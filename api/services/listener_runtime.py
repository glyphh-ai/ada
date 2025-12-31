from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import json
import logging
import traceback
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Dict, Deque, List

import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException

from ..core import models
from ..core.db import SessionLocal
from ..core.schemas import ConceptInput
from .history import persist_history_vectors
from .ingest import ingest_concepts
from .usage_metrics import UsageTracker
from sqlalchemy.orm import joinedload


listener_context = contextvars.ContextVar("listener_id", default=None)
logger = logging.getLogger("glyphai.listeners")


class WebsocketLogHandler(logging.Handler):
    def __init__(self, manager: "ListenerManager"):
        super().__init__()
        self.manager = manager

    def emit(self, record: logging.LogRecord) -> None:
        listener_id = listener_context.get()
        if not listener_id:
            return
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        level = record.levelname.lower()
        self.manager._append_log(listener_id, f"{record.name}: {message}", level=level)


@dataclasses.dataclass
class MapConfig:
    id: str
    model_id: str
    field_mappings: List[models.MapFieldMapping]
    source_schema: List[Dict[str, Any]]
    target_schema: List[Dict[str, Any]]
    transformations: list[dict]
    throttle: int
    history_enabled: bool


@dataclasses.dataclass
class ListenerSpec:
    id: str
    url: str
    headers: Dict[str, str]
    payload_template: Any


@dataclasses.dataclass
class ListenerConfig:
    spec: ListenerSpec
    map_config: MapConfig | None
    version: float


class ListenerManager:
    POLL_INTERVAL = 20
    BASE_RECONNECT_DELAY = 3
    MAX_RECONNECT_DELAY = 60

    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._versions: Dict[str, float] = {}
        self._shutdown = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None
        self._logs: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=200))
        self._configs: Dict[str, ListenerConfig] = {}
        self._disabled: set[str] = set()
        self._throttle_counters: Dict[str, int] = defaultdict(int)
        self._ws_log_handler = WebsocketLogHandler(self)
        self._usage_tracker: UsageTracker | None = None
        ws_logger = logging.getLogger("websockets.client")
        if not any(isinstance(handler, WebsocketLogHandler) for handler in ws_logger.handlers):
            ws_logger.setLevel(logging.DEBUG)
            ws_logger.addHandler(self._ws_log_handler)
        self._log_subscribers: Dict[str, list[asyncio.Queue[Dict[str, str]]]] = defaultdict(list)

    async def start(self) -> None:
        if self._monitor_task:
            return
        self._shutdown.clear()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._shutdown.set()
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        await self._stop_all_tasks()

    def set_usage_tracker(self, tracker: UsageTracker | None) -> None:
        self._usage_tracker = tracker

    async def _monitor_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                configs = await asyncio.to_thread(self._load_listener_configs_sync)
                self._configs = configs
                await self._sync_tasks(configs)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("listener monitor failed")
            await self._sleep_with_shutdown(self.POLL_INTERVAL)

    async def _sync_tasks(self, configs: Dict[str, ListenerConfig]) -> None:
        current_ids = set(configs)
        for listener_id in list(self._tasks):
            config = configs.get(listener_id)
            if listener_id not in current_ids or self._versions.get(listener_id) != config.version:
                await self._stop_task(listener_id)
        for listener_id, config in configs.items():
            if listener_id in self._tasks or listener_id in self._disabled:
                continue
            task = asyncio.create_task(self._run_listener(listener_id, config))
            self._tasks[listener_id] = task
            self._versions[listener_id] = config.version

    async def _run_listener(self, listener_id: str, config: ListenerConfig) -> None:
        delay = self.BASE_RECONNECT_DELAY
        map_id = config.map_config.id if config.map_config else "none"
        self._log_listener(listener_id, "info", "Listener %s starting (map=%s)", listener_id, map_id)
        while not self._shutdown.is_set():
            if listener_id in self._disabled:
                self._append_log(listener_id, "Listener disabled; exiting loop")
                break
            try:
                self._append_log(listener_id, "Attempting to connect listener")
                await self._connect_listener(listener_id, config)
                delay = self.BASE_RECONNECT_DELAY
            except asyncio.CancelledError:
                break
            except Exception:
                self._log_listener(listener_id, "error", "listener %s connection failed", listener_id)
                logger.exception("listener %s connection failed", listener_id)
                self._append_log(listener_id, f"Connection failure: {traceback.format_exc()}")
                await self._sleep_with_shutdown(delay)
                delay = min(delay * 2, self.MAX_RECONNECT_DELAY)

    async def _connect_listener(self, listener_id: str, config: ListenerConfig) -> None:
        token = listener_context.set(listener_id)
        try:
            connect_kwargs: Dict[str, Any] = {}
            async with websockets.connect(config.spec.url, **connect_kwargs) as ws:
                await self._run_payload_sequence(ws, config.spec.payload_template, listener_id)
                try:
                    async for raw in ws:
                        if self._shutdown.is_set():
                            break
                        await self._handle_message(raw, config.map_config)
                except ConnectionClosedError as exc:
                    self._append_log(
                        listener_id,
                        f"WebSocket closed: code={exc.code}, reason={exc.reason}",
                        level="error",
                    )
                    raise
        finally:
            listener_context.reset(token)

    async def _run_payload_sequence(self, ws: websockets.WebSocketClientProtocol, payload_template: Any, listener_id: str) -> None:
        entries = self._normalize_sequence(payload_template)
        for idx, entry in enumerate(entries, start=1):
            payload_obj = entry if isinstance(entry, dict) else json.loads(entry) if isinstance(entry, str) else entry
            if isinstance(payload_obj, dict):
                payload_obj = {k: v for k, v in payload_obj.items() if k != "type"}
            payload = payload_obj if isinstance(payload_obj, str) else json.dumps(payload_obj)
            self._append_log(listener_id, f"Sending handshake payload #{idx}: {payload}", level="debug")
            await ws.send(payload)
            try:
                response_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                self._append_log(listener_id, f"Payload handshake #{idx} timed out", level="error")
                raise RuntimeError("Payload handshake timed out")
            response = self._parse_payload(response_raw)
            if not self._handshake_success(response):
                self._append_log(listener_id, f"Payload handshake #{idx} failure: {response}", level="error")
                raise RuntimeError("Payload handshake failed")
            self._append_log(listener_id, f"Payload handshake #{idx} succeeded", level="debug")

    def _normalize_sequence(self, payload_template: Any) -> List[Any]:
        if isinstance(payload_template, dict):
            normalized = payload_template.get("sequence")
            if isinstance(normalized, list):
                return normalized
            if normalized is None:
                return [payload_template]
        if isinstance(payload_template, list):
            return payload_template
        if isinstance(payload_template, str):
            try:
                parsed = json.loads(payload_template)
            except json.JSONDecodeError:
                return []
            return self._normalize_sequence(parsed)
        return [payload_template]

    def _handshake_success(self, response: Any) -> bool:
        success_values = {"ok", "success", "auth_success", "connected"}
        if isinstance(response, dict):
            return response.get("status") in success_values
        if isinstance(response, list):
            return any(
                isinstance(item, dict) and item.get("status") in success_values
                for item in response
            )
        return False

    async def _handle_message(self, raw: Any, map_config: MapConfig | None) -> None:
        payload = self._parse_payload(raw)
        if not isinstance(payload, dict) and not isinstance(payload, list):
            return
        if not map_config:
            return
        listener_id = listener_context.get() or "unknown listener"
        map_id = map_config.id if map_config else "none"
        self._log_listener(
            listener_id,
            "debug",
            "Listener %s received payload from websocket (map=%s): %s",
            listener_id,
            map_id,
            payload,
        )
        records = payload if isinstance(payload, list) else [payload]
        self._log_listener(
            listener_id,
            "debug",
            "Listener %s treating %s record(s) from websocket (map=%s)",
            listener_id,
            len(records),
            map_id,
        )
        await asyncio.to_thread(self._process_records, records, map_config)
        self._append_log(map_config.id, f"Ingested {len(records)} record(s)")

    def _build_concept(
        self,
        record: Dict[str, Any],
        map_config: MapConfig,
        transform_values: Dict[str, Any],
    ) -> ConceptInput | None:
        attributes: Dict[str, Any] = {}
        for mapping in map_config.field_mappings:
            value = self._resolve_source_value(
                record,
                map_config.source_schema,
                mapping.source_path,
                transform_values,
            )
            if value is None:
                continue
            value = self._normalize_attribute_value(value)
            attr_name = self._resolve_target_field_name(map_config.target_schema, mapping.target_path)
            if not attr_name:
                continue
            attributes[attr_name] = value
        if not attributes:
            listener_id = listener_context.get() or "unknown listener"
            self._log_listener(
                listener_id,
                "debug",
                "Listener %s mapped record to no attributes (map=%s record=%s)",
                listener_id,
                map_config.id,
                record,
            )
            return None
        name_value = attributes.get("grapheme") or attributes.get("name")
        if not name_value:
            for val in attributes.values():
                if val is not None:
                    name_value = val
                    break
        if name_value is None:
            listener_id = listener_context.get() or "unknown listener"
            self._log_listener(
                listener_id,
                "debug",
                "Listener %s mapped record to attributes with no name (map=%s attributes=%s record=%s)",
                listener_id,
                map_config.id,
                attributes,
                record,
            )
            return None
        return ConceptInput(name=str(name_value), attributes=attributes)

    def _normalize_attribute_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(self._normalize_attribute_value(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, self._normalize_attribute_value(item))
                    for key, item in value.items()
                )
            )
        return value

    def _resolve_source_value(
        self,
        record: Dict[str, Any],
        schema: List[Dict[str, Any]],
        path: str,
        transform_values: Dict[str, Any] | None = None,
    ) -> Any | None:
        if path.startswith("transform:"):
            if transform_values is None:
                return None
            return transform_values.get(path)
        current: Any = record
        node_schema = schema
        for segment in filter(None, path.split("-")):
            if not segment.isdigit():
                listener_id = listener_context.get() or "unknown listener"
                self._log_listener(
                    listener_id,
                    "debug",
                    "Listener %s encountered non-digit segment '%s' while resolving source path %s",
                    listener_id,
                    segment,
                    path,
                )
                return None
            idx = int(segment)
            if idx < 0 or idx >= len(node_schema):
                listener_id = listener_context.get() or "unknown listener"
                self._log_listener(
                    listener_id,
                    "debug",
                    "Listener %s encountered out-of-range index %s (len=%s) for source path %s",
                    listener_id,
                    idx,
                    len(node_schema),
                    path,
                )
                return None
            field = node_schema[idx]
            field_name = field.get("name")
            if not field_name or not isinstance(current, dict):
                listener_id = listener_context.get() or "unknown listener"
                self._log_listener(
                    listener_id,
                    "debug",
                    "Listener %s missing field_name while resolving source path %s; current=%s",
                    listener_id,
                    path,
                    current,
                )
                return None
            current = current.get(field_name)
            node_schema = field.get("children") or []
        return current

    def _resolve_target_field_name(
        self, schema: List[Dict[str, Any]], path: str
    ) -> str | None:
        node_schema = schema
        field_name: str | None = None
        for segment in filter(None, path.split("-")):
            if not segment.isdigit():
                listener_id = listener_context.get() or "unknown listener"
                self._log_listener(
                    listener_id,
                    "debug",
                    "Listener %s encountered non-digit segment '%s' while resolving target path %s",
                    listener_id,
                    segment,
                    path,
                )
                return None
            idx = int(segment)
            if idx < 0 or idx >= len(node_schema):
                listener_id = listener_context.get() or "unknown listener"
                self._log_listener(
                    listener_id,
                    "debug",
                    "Listener %s encountered out-of-range index %s (len=%s) for target path %s",
                    listener_id,
                    idx,
                    len(node_schema),
                    path,
                )
                return None
            field = node_schema[idx]
            field_name = field.get("name")
            node_schema = field.get("children") or []
        return field_name

    def _parse_payload(self, raw: Any) -> Any:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        if isinstance(raw, dict):
            return raw
        return None

    def _normalize_templates(self, payload_template: Any) -> List[Any]:
        if not payload_template:
            return []
        if isinstance(payload_template, list):
            return payload_template
        return [payload_template]

    def _find_mapping_for_target(self, map_config: MapConfig, target_path: str):
        for mapping in map_config.field_mappings:
            if mapping.target_path == target_path:
                return mapping
        return None

    def _run_transformation_script(
        self,
        transformation: dict,
        inputs: Dict[str, Any],
        listener_id: str,
        transformation_name: str,
    ) -> Dict[str, Any]:
        script = (transformation.get("script") or "").strip()
        if not script:
            self._log_listener(
                listener_id,
                "debug",
                "Listener %s transformation %s has no script defined",
                listener_id,
                transformation_name,
            )
            return {}
        local_vars = dict(inputs)
        try:
            exec(script, {}, local_vars)
        except Exception as exc:
            self._log_listener(
                listener_id,
                "error",
                "Listener %s transformation %s script failed (%s)",
                listener_id,
                transformation_name,
                exc,
            )
            logger.exception("transformation %s execution failed for listener %s", transformation_name, listener_id)
            return {}
        outputs = {}
        for output in transformation.get("outputs", []):
            name = output.get("name")
            outputs[name] = local_vars.get(name)
        return outputs

    def _evaluate_transformations(
        self,
        record: Dict[str, Any],
        map_config: MapConfig,
        listener_id: str,
        transform_values: Dict[str, Any],
    ) -> None:
        for transformation in map_config.transformations:
            trans_id = transformation.get("id")
            trans_name = transformation.get("name") or trans_id
            input_values: Dict[str, Any] = {}
            for idx, input_def in enumerate(transformation.get("inputs", [])):
                target_key = f"transform:{trans_id}:input:{idx}"
                mapping = self._find_mapping_for_target(map_config, target_key)
                value = None
                if mapping:
                    value = self._resolve_source_value(
                        record,
                        map_config.source_schema,
                        mapping.source_path,
                        transform_values,
                    )
                input_values[input_def.get("name") or f"input{idx}"] = value
            self._log_listener(
                listener_id,
                "debug",
                "Listener %s running transformation %s with inputs %s",
                listener_id,
                trans_name,
                input_values,
            )
            outputs = self._run_transformation_script(
                transformation,
                input_values,
                listener_id,
                trans_name,
            )
            self._log_listener(
                listener_id,
                "debug",
                "Listener %s transformation %s returned outputs %s",
                listener_id,
                trans_name,
                outputs,
            )
            for idx, output_def in enumerate(transformation.get("outputs", [])):
                output_key = f"transform:{trans_id}:output:{idx}"
                output_name = output_def.get("name")
                transform_values[output_key] = outputs.get(output_name)

    def _build_concept_for_record(
        self,
        record: Dict[str, Any],
        map_config: MapConfig,
        listener_id: str,
    ) -> ConceptInput | None:
        transform_values: Dict[str, Any] = {}
        self._evaluate_transformations(record, map_config, listener_id, transform_values)
        return self._build_concept(record, map_config, transform_values)

    def _process_records(self, records: List[Any], map_config: MapConfig) -> None:
        db = SessionLocal()
        try:
            model = db.get(models.Model, map_config.model_id)
            if not model:
                return
            concepts: List[ConceptInput] = []
            listener_id = listener_context.get() or "unknown listener"
            throttle = max(1, map_config.throttle if map_config.throttle else 1)
            counter = self._throttle_counters.get(listener_id, 0)
            for record in records:
                if not isinstance(record, dict):
                    continue
                counter += 1
                if throttle > 1 and counter % throttle != 0:
                    self._throttle_counters[listener_id] = counter
                    self._log_listener(
                        listener_id,
                        "debug",
                        "Listener %s skipping record #%s due to throttle=%s",
                        listener_id,
                        counter,
                        throttle,
                    )
                    continue
                concept = self._build_concept_for_record(record, map_config, listener_id)
                self._throttle_counters[listener_id] = counter
                if concept:
                    concepts.append(concept)
            glyph_count = len(concepts)
            listener_id = listener_context.get() or "unknown listener"
            self._log_listener(
                listener_id,
                "info",
                "Listener %s processed %s record(s) using map %s -> model %s (glyphs=%s)",
                listener_id,
                len(records),
                map_config.id,
                map_config.model_id,
                glyph_count,
            )
            if glyph_count:
                ingest_concepts(model, concepts, False, db)
                if map_config.history_enabled:
                    persist_history_vectors(model, concepts, listener_id)
        finally:
            db.close()

    async def _stop_task(self, listener_id: str) -> None:
        task = self._tasks.pop(listener_id, None)
        self._versions.pop(listener_id, None)
        self._throttle_counters.pop(listener_id, None)
        if not task:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _stop_all_tasks(self) -> None:
        tasks = [self._stop_task(listener_id) for listener_id in list(self._tasks)]
        if tasks:
            await asyncio.gather(*tasks)

    async def control_listener(self, listener_id: str, action: str) -> bool:
        configs = await asyncio.to_thread(self._load_listener_configs_sync)
        config = configs.get(listener_id)
        if action == "stop":
            self._disabled.add(listener_id)
            await self._stop_task(listener_id)
            self._append_log(listener_id, "Listener stopped via control API")
            return True
        if action == "start" and config:
            self._disabled.discard(listener_id)
            if listener_id in self._tasks:
                await self._stop_task(listener_id)
            task = asyncio.create_task(self._run_listener(listener_id, config))
            self._tasks[listener_id] = task
            self._versions[listener_id] = config.version
            self._append_log(listener_id, "Listener started via control API")
            return True
        return False

    def _append_log(self, listener_id: str, message: str, level: str = "info") -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        }
        self._logs[listener_id].append(entry)
        for queue in list(self._log_subscribers.get(listener_id, [])):
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass

    def _log_listener(self, listener_id: str, level: str, message: str, *args: Any) -> None:
        formatted = message % args if args else message
        try:
            log_fn = getattr(logger, level)
        except AttributeError:
            log_fn = logger.info
        log_fn(message, *args)
        self._append_log(listener_id, formatted, level=level)

    def subscribe_logs(
        self, listener_id: str
    ) -> tuple[asyncio.Queue[Dict[str, str]], Callable[[], None]]:
        queue: asyncio.Queue[Dict[str, str]] = asyncio.Queue()
        self._log_subscribers[listener_id].append(queue)

        def unsubscribe() -> None:
            subscribers = self._log_subscribers.get(listener_id)
            if not subscribers:
                return
            try:
                subscribers.remove(queue)
            except ValueError:
                pass

        return queue, unsubscribe

    def get_logs(self, listener_id: str) -> List[Dict[str, str]]:
        return list(self._logs.get(listener_id, []))

    async def _sleep_with_shutdown(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    def _load_listener_configs_sync(
        self, enabled_only: bool = True
    ) -> Dict[str, ListenerConfig]:
        db = SessionLocal()
        try:
            configs: Dict[str, ListenerConfig] = {}
            query = db.query(models.WebSocketListener)
            if enabled_only:
                query = query.filter(models.WebSocketListener.enabled == 1)
            listeners = query.all()
            for listener in listeners:
                map_config = None
                max_version = listener.updated_at.timestamp() if listener.updated_at else 0.0
                if listener.map_id:
                    map_obj = (
                        db.query(models.Map)
                        .options(joinedload(models.Map.target_profile), joinedload(models.Map.source_profile))
                        .get(listener.map_id)
                    )
                if (
                    map_obj
                    and map_obj.target_profile
                    and map_obj.target_profile.generated_from_model_id
                ):
                    throttle = max(1, listener.throttle or 1)
                    if throttle <= 0:
                        throttle = 1
                    map_config = MapConfig(
                        id=map_obj.id,
                        model_id=map_obj.target_profile.generated_from_model_id,
                        field_mappings=list(map_obj.field_mappings),
                        source_schema=map_obj.source_profile.schema if map_obj.source_profile else [],
                        target_schema=map_obj.target_profile.schema if map_obj.target_profile else [],
                        transformations=map_obj.transformations or [],
                        throttle=throttle,
                        history_enabled=bool(listener.history_enabled),
                    )
                    updated_at = map_obj.updated_at.timestamp() if map_obj.updated_at else 0.0
                    max_version = max(max_version, updated_at)
                spec = ListenerSpec(
                    id=listener.id,
                    url=listener.url,
                    headers={str(k): str(v) for k, v in (listener.headers or {}).items()},
                    payload_template=listener.payload_template,
                )
                configs[listener.id] = ListenerConfig(
                    spec=spec,
                    map_config=map_config,
                    version=max_version,
                )
            return configs
        finally:
            db.close()

    def _ensure_config(self, listener_id: str) -> ListenerConfig | None:
        config = self._configs.get(listener_id)
        if config:
            return config
        configs = self._load_listener_configs_sync()
        self._configs.update(configs)
        return configs.get(listener_id)

    def refresh_listener_config(
        self, listener_id: str, include_disabled: bool = False
    ) -> ListenerConfig | None:
        configs = self._load_listener_configs_sync(enabled_only=not include_disabled)
        self._configs.update(configs)
        return configs.get(listener_id)

    def import_records(self, listener_id: str, records: List[Any]) -> int:
        config = self._ensure_config(listener_id)
        if not config or not config.map_config:
            raise RuntimeError("Listener missing configuration or map")
        map_config = config.map_config
        db = SessionLocal()
        token = listener_context.set(listener_id)
        try:
            model = db.get(models.Model, map_config.model_id)
            if not model:
                raise RuntimeError("Model not found")
            concepts: List[ConceptInput] = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                concept = self._build_concept_for_record(record, map_config, listener_id)
                if concept:
                    concepts.append(concept)
            if concepts:
                ingest_concepts(model, concepts, False, db)
                if self._usage_tracker:
                    self._usage_tracker.record("listener_calls", count=len(concepts))
            return len(concepts)
        finally:
            listener_context.reset(token)
            db.close()
