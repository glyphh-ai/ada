from __future__ import annotations

import json

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from ..core import models
from ..core.config import get_settings
from ..core.db import get_db
from ..core.schemas import (
    ClarificationEntry,
    ExecutionPayload,
    GlyphEdgeSummary,
    NLChatPayload,
    NLChatResponse,
    NLQueryPayload,
    NLQueryResponse,
)
from ..services.nl_helpers import build_nl_configs, to_nl_glyphs
from ..services.nl_pipeline import (
    build_capabilities_payload,
    execute_ir,
    plan_intent,
    render_execution_response,
)
from .router import api_router
from glyphh.nl.configs import run_nl_query
from glyphh.nl.chat import generate_response


router = api_router(tags=["nl"])
settings = get_settings()


@router.post("/models/{model_id}/nl-query", response_model=NLQueryResponse)
async def nl_query(
    model_id: str,
    payload: NLQueryPayload,
    db: Session = Depends(get_db),
) -> NLQueryResponse:
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OpenAI key missing")
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    nl_configs = build_nl_configs(model, db)
    capabilities = build_capabilities_payload(nl_configs, model.roles_config or {})
    plan_text = payload.text or ""
    try:
        plan = await plan_intent(text=plan_text, capabilities=capabilities, settings=settings)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"planner error: {exc}")
    clarifications = plan.get("needs_clarification")
    if clarifications:
        entries: list[ClarificationEntry] = []
        for candidate in clarifications:
            if isinstance(candidate, dict):
                question = candidate.get("question") or candidate.get("prompt") or candidate.get("text")
            else:
                question = str(candidate)
            if question:
                entries.append(ClarificationEntry(question=question))
        if entries:
            return NLQueryResponse(query=plan_text, needs_clarification=entries)
    intent_name = plan.get("intent")
    if not intent_name:
        raise HTTPException(status_code=400, detail="Planner did not resolve an intent")
    try:
        execution_payload = execute_ir(ir=plan, model=model, db=db, capabilities=capabilities)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    final_text = await render_execution_response(
        text=plan_text,
        execution_payload=execution_payload,
        settings=settings,
    )
    return NLQueryResponse(
        query=plan_text,
        text=final_text,
        execution=ExecutionPayload(**execution_payload),
    )


@router.post("/models/{model_id}/nl-chat", response_model=NLChatResponse)
async def nl_chat(
    model_id: str,
    payload: NLChatPayload,
    db: Session = Depends(get_db),
) -> NLChatResponse:
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OpenAI key missing")
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    glyphs = db.query(models.Glyph).filter(models.Glyph.model_id == model.id).all()
    nl_configs = build_nl_configs(model, db)
    nl_resp = run_nl_query(payload.text or "", nl_configs, model.roles_config or {}, to_nl_glyphs(glyphs))
    if not nl_resp:
        raise HTTPException(status_code=404, detail="No glyph match")
    attributes = nl_resp.get("attributes") or {}
    glyph_edges = {
        "matched_glyph": nl_resp.get("name") or nl_resp.get("matched_glyph"),
        "primary_target": nl_resp.get("primary_target"),
        "primary_edge": nl_resp.get("primary_edge"),
        "secondary_edge": nl_resp.get("secondary_edge"),
        "secondary_target": nl_resp.get("secondary_target"),
        "semantic_edges": nl_resp.get("semantic_edges") or [],
        "neural_edges": nl_resp.get("neural_edges") or [],
        "hierarchy": attributes.get("taxonomy") or [],
        "sequence": nl_resp.get("sequence") or [],
    }
    glyph_data_for_prompt = {
        "glyph": glyph_edges,
        "attributes": attributes,
    }
    system_prompt = (
        "You are an assistant that answers only with facts derived from provided glyph data. "
        "Do not invent anything not in the data."
    )
    user_prompt = (
        f"User query: {payload.text}\n"
        f"Glyph data: {json.dumps(glyph_data_for_prompt, separators=(',', ':'))}\n"
        "Only describe what can be proven from that glyph graph."
    )
    try:
        resp = await generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=payload.temperature or 0.2,
            openai_model=settings.openai_model,
            api_key=settings.openai_api_key,
            previous_response_id=payload.previous_response_id,
        )
        return NLChatResponse(
            text=resp["text"],
            glyph_edges=GlyphEdgeSummary(**glyph_edges),
            message_id=resp.get("previous_response_id"),
            conversation_id=resp.get("conversation_id"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
