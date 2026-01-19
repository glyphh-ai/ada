from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any, Dict, List, Sequence

import numpy as np
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..core import models
from glyphh.nl.chat import generate_response

logger = logging.getLogger(__name__)

QUERY_TEMPLATES: Dict[str, str] = {
    "trade_snapshot": (
        "Fetch the latest glyph whose event_type corresponds to a trade for the requested ticker, "
        "ordering by event_timestamp descending."
    ),
    "quote_snapshot": (
        "Fetch the latest glyph whose event_type corresponds to a quote (bid/ask) for the requested ticker, "
        "ordering by event_timestamp descending."
    ),
    "aggregate_status": (
        "Fetch the most recent aggregate glyph for the requested timeframe (minute/hour/day/etc.) "
        "and ticker, returning the close price, volume, and timestamp fields."
    ),
    "trend_summary": (
        "Fetch a window of aggregate glyphs covering the requested timeframe, compute slope, "
        "returns, moving averages, and volatility, then summarize the trend direction for the ticker."
    ),
}

FIELD_ALIAS_OVERRIDES: Dict[str, str] = {
    "symbol": "ticker",
    "sym": "ticker",
    "ticker symbol": "ticker",
    "ticker_name": "ticker",
    "stock": "ticker",
    "company": "ticker",
    "quote": "event_type",
    "quotes": "event_type",
    "quote event": "event_type",
    "trade event": "event_type",
    "event": "event_type",
    "type": "event_type",
    "event type": "event_type",
    "close": "tick_price_close",
    "last price": "tick_price_close",
    "price close": "tick_price_close",
    "last trade": "trade_price",
    "trade price": "trade_price",
    "price": "tick_price_close",
    "price open": "tick_price_open",
    "open price": "tick_price_open",
    "price high": "tick_price_high",
    "high price": "tick_price_high",
    "price low": "tick_price_low",
    "low price": "tick_price_low",
    "bid": "bid_price",
    "bid price": "bid_price",
    "ask": "ask_price",
    "ask price": "ask_price",
    "bid size": "bid_size",
    "ask size": "ask_size",
    "volume": "tick_volume",
    "size": "trade_size",
    "trade size": "trade_size",
    "time": "event_timestamp",
    "timestamp": "event_timestamp",
    "event time": "event_timestamp",
    "timeframe": "timeframe",
    "interval": "timeframe",
    "duration": "timeframe",
    "period": "timeframe",
    "time period": "timeframe",
    "window": "timeframe",
    "range": "time_range",
    "time range": "time_range",
    "start": "time_range",
    "end": "time_range",
    "start date": "time_range",
    "end date": "time_range",
    "from": "time_range",
    "to": "time_range",
}

TIMEFRAME_NORMALIZATION: Dict[str, str] = {
    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",
    "hour": "hour",
    "hours": "hour",
    "hr": "hour",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "annual": "year",
    "yr": "year",
}


def _collect_roles(roles_config: Dict[str, Any] | None) -> Dict[str, str | None]:
    fields: Dict[str, str | None] = {}
    if not roles_config:
        return fields
    for layer in roles_config.get("layers", []):
        for segment in layer.get("segments", []):
            for entry in segment.get("roles", []):
                if isinstance(entry, str):
                    fields.setdefault(entry, None)
                    continue
                role = entry.get("role")
                if not role:
                    continue
                fields.setdefault(role, entry.get("type"))
    return fields


def _slots_for_intent(intent: Dict[str, Any]) -> set[str]:
    slots: set[str] = set()
    for key in ("role", "find_role", "amount_field", "date_field"):
        val = intent.get(key)
        if isinstance(val, str):
            slots.add(val)
    for flt in intent.get("filters", []) or []:
        if isinstance(flt, dict):
            role = flt.get("role")
            if isinstance(role, str):
                slots.add(role)
    return slots


def _build_alias_map(
    field_map: Dict[str, str | None],
    nl_configs: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    # add role mapping aliases
    for cfg in nl_configs:
        role_mapping = (cfg or {}).get("role_mapping", {})
        for entry in (role_mapping or {}).get("attributes", []):
            if not isinstance(entry, dict):
                continue
            source = entry.get("source")
            target = entry.get("role")
            if not source or not target:
                continue
            alias_map[source.lower()] = target
            alias_map[source.replace("_", " ").lower()] = target
    # add canonical underscored <-> spaced mappings
    for field in field_map:
        alias_map.setdefault(field.lower(), field)
        spaced = field.replace("_", " ").lower()
        alias_map.setdefault(spaced, field)
    # add manual overrides
    for alias, canonical in FIELD_ALIAS_OVERRIDES.items():
        alias_map.setdefault(alias, canonical)
    return alias_map


def _clean_templates(patterns: Sequence[Any]) -> List[str]:
    seen: List[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        cleaned = pattern.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen[:4]


def _collect_intents(
    nl_configs: Sequence[Dict[str, Any]],
    field_map: Dict[str, str | None],
) -> List[Dict[str, Any]]:
    intent_map: Dict[str, Dict[str, Any]] = {}
    default_slot = "ticker" if "ticker" in field_map else None
    for cfg in nl_configs:
        for intent in (cfg or {}).get("intents", []) or []:
            name = intent.get("name")
            if not name:
                continue
            normalized = name.strip()
            entry = intent_map.setdefault(
                normalized,
                {
                    "name": normalized,
                    "type": intent.get("type"),
                    "description": intent.get("description"),
                    "metadata": dict(intent.get("metadata") or {}),
                    "templates": [],
                    "examples": [],
                    "required_slots": set[str](),
                },
            )
            entry["type"] = entry["type"] or intent.get("type")
            entry["metadata"].update(intent.get("metadata") or {})
            patterns = _clean_templates(intent.get("patterns") or [])
            entry["templates"].extend(p for p in patterns if p not in entry["templates"])
            for example in intent.get("examples", []) or []:
                if isinstance(example, dict):
                    text = example.get("text")
                    if isinstance(text, str):
                        entry["examples"].append(text)
            entry["required_slots"].update(_slots_for_intent(intent))
    for intent in intent_map.values():
        if default_slot:
            intent["required_slots"].add(default_slot)
        if intent["metadata"].get("timeframes"):
            intent["required_slots"].add("timeframe")
        if intent["metadata"].get("verbs"):
            intent["metadata"]["verbs"] = intent["metadata"]["verbs"]
    for intent in intent_map.values():
        intent["required_slots"] = sorted(intent["required_slots"])
        intent["templates"] = intent["templates"][:4]
        intent["examples"] = intent["examples"][:3]
        intent["executor_template"] = QUERY_TEMPLATES.get(intent["name"])
    sorted_intents = sorted(intent_map.values(), key=lambda i: i["name"])
    return sorted_intents


def build_capabilities_payload(
    nl_configs: Sequence[Dict[str, Any]],
    roles_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    field_map = _collect_roles(roles_config)
    intents = _collect_intents(nl_configs, field_map)
    alias_map = _build_alias_map(field_map, nl_configs)
    field_entries = [
        {"name": name, "type": field_map[name] or "unknown"} for name in sorted(field_map)
    ]
    return {
        "intents": intents,
        "fields": field_entries,
        "aliases": alias_map,
    }


def _parse_plan_text(text: str) -> Dict[str, Any]:
    trimmed = text.strip()
    if not trimmed:
        raise ValueError("planner returned empty response")
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(trimmed[start : end + 1])


async def plan_intent(
    *,
    text: str,
    capabilities: Dict[str, Any],
    settings: "Settings",
    context: str | None = None,
) -> Dict[str, Any]:
    system_prompt = (
        "You are a planner. Output only JSON that follows the IR schema described below. "
        "Do not add explanations or markup outside the JSON. Follow the schema exactly."
    )
    schema_description = (
        "IR schema:\n"
        "{\n"
        '  "intent": "<intent name from the capabilities>",\n'
        '  "slots": {"ticker": "value", "event_type": "trade", ...},\n'
        '  "metadata": {"tone": "urgent", "verbosity": "low"}\n'
        "}\n"
        "If you cannot decide on an intent or need more detail, respond with:\n"
        '{ "needs_clarification": [{"question": "..."}, ...] }\n'
    )
    user_prompt = (
        f"User query:\n{text.strip()}\n\n"
        f"Capabilities:\n{json.dumps(capabilities, indent=2)}\n\n"
        f"{schema_description}"
    )
    if context:
        user_prompt += f"\nContext:\n{context}\n\n"
    response = await generate_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        openai_model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    logger.debug("planner response=%s", response)
    plan = _parse_plan_text(response["text"])
    return plan


def _normalize_slot_key(key: str, alias_map: Dict[str, str]) -> str:
    normalized = key.strip().lower()
    return alias_map.get(normalized, key.strip())


def _normalize_slots(slots: Dict[str, Any], alias_map: Dict[str, str]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in slots.items():
        if not isinstance(raw_key, str):
            continue
        canonical = _normalize_slot_key(raw_key, alias_map)
        normalized[canonical] = raw_value
    return normalized


def _resolve_timeframe(slots: Dict[str, Any], metadata: Dict[str, Any]) -> str | None:
    timeframe = slots.get("timeframe")
    if isinstance(timeframe, str) and timeframe.strip():
        value = timeframe.strip().lower()
    else:
        candidates = metadata.get("timeframes") or []
        value = candidates[0].strip().lower() if candidates else ""
    if not value:
        return None
    return TIMEFRAME_NORMALIZATION.get(value, value)


def _parse_time_range_slot(value: Any) -> tuple[dt.datetime | None, dt.datetime | None] | None:
    if value is None:
        return None

    def _parse(value_inner: Any) -> dt.datetime | None:
        if not value_inner:
            return None
        if isinstance(value_inner, (int, float)):
            try:
                return dt.datetime.utcfromtimestamp(float(value_inner) / 1000)
            except Exception:
                return None
        if isinstance(value_inner, str):
            cleaned = value_inner.strip()
            if not cleaned:
                return None
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return dt.datetime.fromisoformat(cleaned)
                except ValueError:
                    continue
            try:
                value_float = float(cleaned)
                return dt.datetime.utcfromtimestamp(value_float / 1000)
            except ValueError:
                return None
        return None

    def _parse_relative_range(raw: str) -> tuple[dt.datetime | None, dt.datetime | None] | None:
        token = raw.strip().lower()
        if not token:
            return None
        if token.startswith("last_"):
            token = token.replace("_", " ")
        match = re.match(r"^(last|past)\s*(\d+)\s*(day|days|month|months|year|years)$", token)
        if not match:
            return None
        _, count_raw, unit = match.groups()
        try:
            count = int(count_raw)
        except ValueError:
            return None
        now = dt.datetime.utcnow()
        if unit.startswith("day"):
            delta = dt.timedelta(days=count)
        elif unit.startswith("month"):
            delta = dt.timedelta(days=count * 30)
        else:
            delta = dt.timedelta(days=count * 365)
        return (now - delta, now)

    if isinstance(value, str):
        range_match = _parse_relative_range(value)
        if range_match:
            return range_match
        for separator in ("..", " to ", " - "):
            if separator in value:
                start_raw, end_raw = value.split(separator, 1)
                start = _parse(start_raw)
                end = _parse(end_raw)
                if start or end:
                    return (start, end)
        return None

    if isinstance(value, dict):
        start_raw = value.get("start")
        end_raw = value.get("end")
        if isinstance(start_raw, str):
            range_match = _parse_relative_range(start_raw)
            if range_match and not end_raw:
                return range_match
        start = _parse(start_raw)
        end = _parse(end_raw)
        if not start and not end:
            return None
        return (start, end)

    return None


def _parse_timestamp(raw: Any) -> dt.datetime | None:
    if raw is None:
        return None
    if isinstance(raw, dt.datetime):
        return raw
    if isinstance(raw, (int, float)):
        try:
            return dt.datetime.utcfromtimestamp(float(raw) / 1000)
        except Exception:
            return None
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return None
        try:
            return dt.datetime.fromisoformat(cleaned)
        except ValueError:
            pass
        try:
            return dt.datetime.utcfromtimestamp(float(cleaned) / 1000)
        except ValueError:
            return None
    return None


def _format_timestamp(value: dt.datetime | None) -> str | None:
    if not value:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc).isoformat()
    return value.isoformat()


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_string_field(semantic: Dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = semantic.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    return None


def _extract_number_field(semantic: Dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = semantic.get(key)
        num = _to_number(value)
        if num is not None:
            return num
    return None


def _glyph_timestamp(glyph: models.Glyph) -> dt.datetime | None:
    if not glyph.semantic:
        return None
    return _parse_timestamp(glyph.semantic.get("event_timestamp"))


def _glyph_in_time_range(
    glyph: models.Glyph, time_range: tuple[dt.datetime | None, dt.datetime | None] | None
) -> bool:
    if not time_range:
        return True
    start, end = time_range
    ts = _glyph_timestamp(glyph)
    if not ts:
        return False
    if start and ts < start:
        return False
    if end and ts > end:
        return False
    return True


def _filter_glyphs_by_time_range(
    glyphs: Sequence[models.Glyph],
    time_range: tuple[dt.datetime | None, dt.datetime | None] | None,
) -> list[models.Glyph]:
    if not time_range:
        return list(glyphs)
    return [glyph for glyph in glyphs if _glyph_in_time_range(glyph, time_range)]


def _build_event_candidates(timeframe: str | None, include_aggregate: bool = True) -> list[str]:
    candidates: list[str] = []
    if timeframe:
        candidates.extend(
            [f"{timeframe}_aggregate", f"{timeframe}_bar", timeframe, f"{timeframe}_trend"]
        )
    if include_aggregate:
        candidates.extend(["aggregate", "minute_aggregate", "hour_aggregate", "day_aggregate"])
    seen: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


def _fetch_candidate_glyphs(
    model: models.Model,
    db: Session,
    ticker: str | None,
    event_candidates: Sequence[str] | None,
    limit: int = 200,
) -> list[models.Glyph]:
    query = db.query(models.Glyph).filter(models.Glyph.model_id == model.id)
    if ticker:
        ticker_val = ticker.lower()
        ticker_field = func.lower(func.coalesce(models.Glyph.semantic["ticker"].astext, ""))
        query = query.filter(ticker_field == ticker_val)
    if event_candidates:
        event_field = func.lower(func.coalesce(models.Glyph.semantic["event_type"].astext, ""))
        conditions = []
        for candidate in event_candidates:
            normalized = candidate.lower()
            conditions.append(event_field == normalized)
            conditions.append(event_field.like(f"%{normalized}%"))
        if conditions:
            query = query.filter(or_(*conditions))
    query = query.order_by(func.coalesce(models.Glyph.semantic["event_timestamp"].astext, "").desc())
    return query.limit(limit).all()


def _select_latest_glyph(
    glyphs: Sequence[models.Glyph],
    time_range: tuple[dt.datetime | None, dt.datetime | None] | None,
) -> models.Glyph | None:
    for glyph in glyphs:
        if _glyph_in_time_range(glyph, time_range):
            return glyph
    return None


def _resolve_event_types(slots: Dict[str, Any], fallback: str | None) -> list[str]:
    candidates: list[str] = []
    event_type_raw = slots.get("event_type")
    if isinstance(event_type_raw, str) and event_type_raw.strip():
        candidates.append(event_type_raw.strip().lower())
    if fallback:
        candidates.append(fallback.lower())
    return list(dict.fromkeys([candidate for candidate in candidates if candidate]))


def _extract_slot_value(slots: Dict[str, Any], name: str) -> Any | None:
    return slots.get(name)


def _execute_trade_snapshot(
    model: models.Model,
    db: Session,
    slots: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    ticker = _extract_slot_value(slots, "ticker")
    normalized_ticker = ticker if isinstance(ticker, str) else None
    event_types = _resolve_event_types(slots, "trade")
    glyphs = _fetch_candidate_glyphs(model, db, normalized_ticker, event_types, limit=50)
    glyph = _select_latest_glyph(glyphs, None)
    if not glyph:
        return {
            "result": {},
            "evidence": {},
            "explain": {"reason": "No recent trade glyph found for the requested ticker."},
            "errors": ["No matching trade glyph"],
        }
    sem = glyph.semantic or {}
    price = _extract_number_field(sem, ["trade_price", "tick_price_close", "price_close"])
    size = _extract_number_field(sem, ["trade_size", "tick_volume"])
    evidence = {
        "glyph": glyph.name,
        "ticker": _extract_string_field(sem, ["ticker", "sym", "symbol"]) or normalized_ticker,
        "event_type": _extract_string_field(sem, ["event_type"]),
    }
    timestamp = _format_timestamp(_glyph_timestamp(glyph))
    result = {
        "glyph": glyph.name,
        "ticker": evidence.get("ticker"),
        "event_type": evidence.get("event_type") or "trade",
        "trade_price": price,
        "trade_size": size,
        "timestamp": timestamp,
    }
    return {
        "result": result,
        "evidence": evidence,
        "explain": {
            "reason": f"Selected the most recent trade glyph for {result.get('ticker')} at {timestamp}."
        },
        "errors": [],
    }


def _execute_quote_snapshot(
    model: models.Model,
    db: Session,
    slots: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    ticker = _extract_slot_value(slots, "ticker")
    normalized_ticker = ticker if isinstance(ticker, str) else None
    event_types = _resolve_event_types(slots, "quote")
    glyphs = _fetch_candidate_glyphs(model, db, normalized_ticker, event_types, limit=50)
    glyph = _select_latest_glyph(glyphs, None)
    if not glyph:
        return {
            "result": {},
            "evidence": {},
            "explain": {"reason": "No recent quote glyph found for the requested ticker."},
            "errors": ["No matching quote glyph"],
        }
    sem = glyph.semantic or {}
    bid_price = _extract_number_field(sem, ["bid_price"])
    ask_price = _extract_number_field(sem, ["ask_price"])
    bid_size = _extract_number_field(sem, ["bid_size"])
    ask_size = _extract_number_field(sem, ["ask_size"])
    evidence = {
        "glyph": glyph.name,
        "ticker": _extract_string_field(sem, ["ticker", "sym", "symbol"]) or normalized_ticker,
        "event_type": _extract_string_field(sem, ["event_type"]),
    }
    timestamp = _format_timestamp(_glyph_timestamp(glyph))
    result = {
        "glyph": glyph.name,
        "ticker": evidence.get("ticker"),
        "event_type": evidence.get("event_type") or "quote",
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "timestamp": timestamp,
    }
    return {
        "result": result,
        "evidence": evidence,
        "explain": {
            "reason": f"Selected the most recent quote glyph for {result.get('ticker')} at {timestamp}."
        },
        "errors": [],
    }


def _execute_aggregate_status(
    model: models.Model,
    db: Session,
    slots: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    ticker = _extract_slot_value(slots, "ticker")
    normalized_ticker = ticker if isinstance(ticker, str) else None
    timeframe = _resolve_timeframe(slots, metadata)
    event_candidates = _build_event_candidates(timeframe)
    glyphs = _fetch_candidate_glyphs(model, db, normalized_ticker, event_candidates, limit=100)
    time_range = _parse_time_range_slot(slots.get("time_range"))
    filtered = _filter_glyphs_by_time_range(glyphs, time_range)
    glyph = _select_latest_glyph(filtered, time_range)
    if not glyph and not filtered:
        glyph = _select_latest_glyph(glyphs, time_range)
    if not glyph:
        return {
            "result": {},
            "evidence": {},
            "explain": {
                "reason": "Could not find an aggregate glyph for the requested timeframe and ticker."
            },
            "errors": ["No aggregate glyph"],
        }
    sem = glyph.semantic or {}
    closing = _extract_number_field(sem, ["price_close", "tick_price_close"])
    volume = _extract_number_field(sem, ["tick_volume", "volume", "trade_size"])
    evidence = {
        "glyph": glyph.name,
        "ticker": _extract_string_field(sem, ["ticker", "sym", "symbol"]) or normalized_ticker,
        "timeframe": timeframe,
    }
    timestamp = _format_timestamp(_glyph_timestamp(glyph))
    result = {
        "glyph": glyph.name,
        "ticker": evidence.get("ticker"),
        "timeframe": timeframe,
        "close": closing,
        "volume": volume,
        "timestamp": timestamp,
    }
    return {
        "result": result,
        "evidence": evidence,
        "explain": {
            "reason": f"Selected the latest aggregate glyph for {result.get('ticker')} with timeframe {timeframe}."
        },
        "errors": [],
    }


def _execute_trend_summary(
    model: models.Model,
    db: Session,
    slots: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    ticker = _extract_slot_value(slots, "ticker")
    normalized_ticker = ticker if isinstance(ticker, str) else None
    timeframe = _resolve_timeframe(slots, metadata)
    event_candidates = _build_event_candidates(timeframe)
    glyphs = _fetch_candidate_glyphs(model, db, normalized_ticker, event_candidates, limit=200)
    time_range = _parse_time_range_slot(slots.get("time_range"))
    filtered = _filter_glyphs_by_time_range(glyphs, time_range)
    points: list[tuple[dt.datetime, float]] = []
    for glyph in filtered:
        price = _extract_number_field(glyph.semantic or {}, ["price_close", "tick_price_close"])
        timestamp = _glyph_timestamp(glyph)
        if price is None or not timestamp:
            continue
        points.append((timestamp, price))
    points.sort(key=lambda item: item[0])
    if len(points) < 2:
        return {
            "result": {},
            "evidence": {},
            "explain": {
                "reason": "Not enough aggregate glyphs to compute a trend summary for the requested timeframe."
            },
            "errors": ["Insufficient points for trend"],
        }
    timestamps = np.array([pt[0].timestamp() for pt in points])
    prices = np.array([pt[1] for pt in points])
    try:
        slope, intercept = np.polyfit(timestamps, prices, 1)
    except Exception:
        slope = 0.0
        intercept = float(prices[0]) if prices.size > 0 else 0.0
    latest_price = float(prices[-1]) if prices.size else None
    earliest_price = float(prices[0]) if prices.size else None
    percent_change = None
    if earliest_price and earliest_price != 0:
        percent_change = ((latest_price - earliest_price) / earliest_price) * 100  # type: ignore[arg-type]
    volatility = float(np.std(prices)) if prices.size else None
    window = min(len(prices), 5)
    moving_average = float(np.mean(prices[-window:])) if window else None
    direction = "flat"
    if slope > 0:
        direction = "up"
    elif slope < 0:
        direction = "down"
    evidence = {
        "glyphs": [glyph.name for glyph in filtered[:10]],
        "ticker": _extract_string_field(
            (filtered[0].semantic or {}) if filtered else {},
            ["ticker", "sym", "symbol"],
        )
        or normalized_ticker,
        "timeframe": timeframe,
    }
    result = {
        "glyphs_considered": len(points),
        "timeframe": timeframe,
        "latest_price": latest_price,
        "earliest_price": earliest_price,
        "slope": float(slope),
        "percent_change": percent_change,
        "volatility": volatility,
        "moving_average": moving_average,
        "trend_direction": direction,
        "start_time": points[0][0].isoformat(),
        "end_time": points[-1][0].isoformat(),
    }
    return {
        "result": result,
        "evidence": evidence,
        "explain": {
            "reason": (
                f"Computed trend for {result.get('ticker')} over "
                f"{result.get('glyphs_considered')} points covering {result.get('start_time')} to "
                f"{result.get('end_time')}."
            )
        },
        "errors": [],
    }


EXECUTOR_MAP: Dict[str, Any] = {
    "trade_snapshot": _execute_trade_snapshot,
    "quote_snapshot": _execute_quote_snapshot,
    "aggregate_status": _execute_aggregate_status,
    "trend_summary": _execute_trend_summary,
}


def execute_ir(
    *,
    ir: Dict[str, Any],
    model: models.Model,
    db: Session,
    capabilities: Dict[str, Any],
) -> Dict[str, Any]:
    intent = ir.get("intent")
    if not intent or intent not in EXECUTOR_MAP:
        raise ValueError(f"Unsupported intent {intent}")
    executor = EXECUTOR_MAP[intent]
    alias_map = capabilities.get("aliases") or {}
    slots = _normalize_slots(ir.get("slots") or {}, alias_map)
    metadata = ir.get("metadata") or {}
    execution = executor(model, db, slots, metadata)
    return {
        "ir": ir,
        "result": execution.get("result", {}),
        "evidence": execution.get("evidence", {}),
        "explain": execution.get("explain", {}),
        "errors": execution.get("errors", []),
    }


async def render_execution_response(
    *,
    text: str,
    execution_payload: Dict[str, Any],
    settings: "Settings",
) -> str:
    system_prompt = "Answer only from the provided execution payload; don't invent facts."
    metadata = execution_payload.get("ir", {}).get("metadata") or {}
    user_prompt = (
        f"User query: {text.strip()}\n"
        f"IR plan: {json.dumps(execution_payload.get('ir', {}), separators=(',', ':'))}\n"
        f"Execution payload: {json.dumps(execution_payload, separators=(',', ':'))}\n"
        f"Tone metadata: {json.dumps(metadata, separators=(',', ':'))}\n"
        "Use the evidence fields to justify your response."
    )
    response = await generate_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        openai_model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    return response["text"]
