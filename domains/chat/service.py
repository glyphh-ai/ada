"""Generic per-model chat service — Glyphh match + optional LLM formatting.

Every deployed model gets the same chat interface:
  User query → Glyphh HDC match → if confident: return pure Glyphh answer
                                 → if low confidence or follow-up: LLM formats

The runtime owns LLM selection (local or external). When no LLM is available
(air-gapped), the raw Glyphh output is returned directly.

This replaces the platform's AssistantChatService — the platform no longer
makes LLM calls. All LLM orchestration lives here in the runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from domains.chat.llm import LLMMessage, LLMResponse, get_llm_provider
from domains.chat.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 6


@dataclass
class ChatTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatResult:
    """Result returned from the chat service."""
    content: str
    command: Optional[str] = None
    code: Optional[str] = None
    confidence: float = 0.0
    match_method: str = "glyphh"
    fact_tree: Optional[dict] = None
    state: str = "DONE"  # DONE, ASK, CONFIRM, ERROR
    provider: str = "glyphh"
    usage: dict = field(default_factory=dict)
    trace_id: str = ""
    action_name: Optional[str] = None
    missing_slots: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _build_system_prompt(org_id: str, model_id: str) -> str:
    """Build a system prompt for the LLM based on the model identity.

    The assistant model gets a Glyphh-specific persona. All other models
    get a generic, neutral prompt that formats Glyphh results conversationally.
    """
    if org_id == "glyphh" and model_id == "assistant":
        return (
            "You are the Glyphh AI assistant — a conversational, warm, and concise guide "
            "for the Glyphh hyperdimensional computing platform.\n\n"
            "You speak in first person, lowercase, like a helpful teammate. You're friendly "
            "but not over-the-top. You get to the point.\n\n"
            "When you receive structured facts from the Glyphh engine:\n"
            "1. Synthesize a clear, concise response from the facts provided.\n"
            "2. If a CLI command is included, mention it naturally.\n"
            "3. If confidence is low, say so honestly.\n"
            "4. Never make up Glyphh features or commands.\n"
            "Keep responses under 100 words unless the user asks for detail.\n"
            "Use lowercase. No corporate speak. Be human."
        )

    return (
        f"You are a helpful assistant for the '{model_id}' knowledge model.\n\n"
        "When you receive structured facts from the knowledge base:\n"
        "1. Synthesize a clear, concise response from the facts provided.\n"
        "2. If confidence is low, say so honestly and suggest rephrasing.\n"
        "3. Never make up information beyond what the facts provide.\n"
        "Keep responses concise and factual."
    )


class ModelChatService:
    """Generic chat service for any deployed Glyphh model.

    Queries the model via the runtime's internal QueryService, then
    optionally formats the result through an LLM.
    """

    def __init__(
        self,
        org_id: str,
        model_id: str,
        llm: LLMProvider | None = None,
    ):
        self.org_id = org_id
        self.model_id = model_id
        self._llm = llm if llm is not None else get_llm_provider()
        self._system_prompt = _build_system_prompt(org_id, model_id)

    async def chat(
        self,
        message: str,
        history: list[ChatTurn] | None = None,
        session_id: str | None = None,
        is_followup: bool = False,
    ) -> ChatResult:
        """Process a user message through Glyphh match + optional LLM.

        1. If follow-up with history: LLM-only with conversation context
        2. Query the Glyphh model via internal service
        3. If confident match with content: return pure Glyphh answer
        4. If match but no content: LLM synthesizes from fact_tree
        5. Fallback: LLM answers from general knowledge
        """
        trace_id = str(uuid.uuid4())[:8]
        history = history or []

        # Follow-ups go straight to LLM with conversation context
        if is_followup and history and self._llm:
            return await self._llm_with_history(message, history, trace_id)

        # Query the Glyphh model
        glyphh_result = await self._query_model(message)

        if glyphh_result is None:
            # Model query failed entirely
            if self._llm:
                return await self._llm_fallback(message, history, trace_id)
            return ChatResult(
                content="the model couldn't process that query. try rephrasing.",
                state="ERROR",
                trace_id=trace_id,
            )

        confidence = glyphh_result.get("confidence", 0.0)
        fact_tree = glyphh_result.get("fact_tree")
        raw_content = fact_tree.get("text", "") if fact_tree else ""
        command = glyphh_result.get("command")
        code = glyphh_result.get("code")
        match_method = glyphh_result.get("match_method", "glyphh")

        # Pure Glyphh path: confident match with content → return directly
        if raw_content and confidence > 0.0:
            return ChatResult(
                content=raw_content,
                command=command,
                code=code,
                confidence=confidence,
                match_method=match_method,
                fact_tree=fact_tree,
                state=glyphh_result.get("state", "DONE"),
                provider="glyphh",
                trace_id=trace_id,
            )

        # Match but no content → LLM synthesizes from fact_tree
        if confidence > 0.0 and self._llm:
            return await self._synthesize(message, glyphh_result, history, trace_id)

        # No match → LLM fallback
        if self._llm:
            return await self._llm_fallback(message, history, trace_id)

        return ChatResult(
            content="no answer found.",
            confidence=confidence,
            match_method=match_method,
            fact_tree=fact_tree,
            state="DONE",
            provider="glyphh",
            trace_id=trace_id,
        )

    async def _query_model(self, query: str) -> dict[str, Any] | None:
        """Query the deployed model via the runtime's internal QueryService."""
        try:
            from domains.query.service import QueryService
            from infrastructure.database import async_session_maker
            from main import model_manager

            if model_manager is None:
                logger.warning("Model manager not initialized")
                return None

            service = QueryService(model_manager, async_session_maker)
            # Use the NL query pipeline for natural language input
            from domains.nl_query.service import NLQueryService
            from domains.nl_query.intent_matcher import IntentMatcher

            intent_matcher = IntentMatcher(confidence_threshold=0.85)
            nl_service = NLQueryService(
                query_service=service,
                intent_matcher=intent_matcher,
                confidence_threshold=0.85,
            )

            result = await nl_service.execute_nl_query(
                org_id=self.org_id,
                model_id=self.model_id,
                query=query,
            )

            response_data = result.to_dict()
            if result.fact_tree is not None:
                response_data["result"] = result.fact_tree.to_json()

            return response_data

        except Exception as e:
            logger.warning(f"Model query failed ({self.org_id}/{self.model_id}): {e}")
            return None

    async def _synthesize(
        self,
        message: str,
        glyphh_result: dict[str, Any],
        history: list[ChatTurn],
        trace_id: str,
    ) -> ChatResult:
        """Send Glyphh facts to the LLM for synthesis."""
        fact_tree = glyphh_result.get("fact_tree")
        confidence = glyphh_result.get("confidence", 0.0)
        command = glyphh_result.get("command")
        code = glyphh_result.get("code")
        raw_content = fact_tree.get("text", "") if fact_tree else ""

        context_parts = [f"Confidence: {confidence:.0%}"]
        if raw_content:
            context_parts.append(f"Answer: {raw_content}")
        if command:
            context_parts.append(f"CLI command: {command}")
        if code:
            context_parts.append(f"Code:\n{code}")
        if fact_tree and fact_tree.get("children"):
            for child in fact_tree["children"][:3]:
                context_parts.append(f"Detail: {child.get('text', '')}")

        context_block = "\n".join(context_parts)

        messages: list[LLMMessage] = [LLMMessage(role="system", content=self._system_prompt)]
        for turn in history[-_MAX_HISTORY_TURNS:]:
            messages.append(LLMMessage(role=turn.role, content=turn.content))

        user_content = (
            f"User question: {message}\n\n"
            f"--- Glyphh facts ---\n{context_block}\n--- end facts ---"
        )
        messages.append(LLMMessage(role="user", content=user_content))

        try:
            llm_resp = await self._llm.complete(messages, max_tokens=512, temperature=0.3)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return ChatResult(
                content=raw_content or "something went wrong generating a response.",
                command=command,
                code=code,
                confidence=confidence,
                match_method="glyphh-fallback",
                fact_tree=fact_tree,
                provider="glyphh",
                trace_id=trace_id,
            )

        content = llm_resp.content.strip()
        state = "ASK" if content.endswith("?") and confidence < 0.5 else "DONE"

        return ChatResult(
            content=content,
            command=command,
            code=code,
            confidence=confidence,
            match_method="llm",
            fact_tree=fact_tree,
            state=state,
            provider=self._llm.provider_name(),
            usage=llm_resp.usage,
            trace_id=trace_id,
        )

    async def _llm_with_history(
        self,
        message: str,
        history: list[ChatTurn],
        trace_id: str,
    ) -> ChatResult:
        """LLM response for conversational follow-ups using history context."""
        extra = (
            "\n\nThis is a conversational follow-up. The user is continuing "
            "a previous topic. Use the conversation history to give a relevant, "
            "grounded answer. Stay factual — don't speculate."
        )
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self._system_prompt + extra),
        ]
        for turn in history[-_MAX_HISTORY_TURNS:]:
            messages.append(LLMMessage(role=turn.role, content=turn.content))
        messages.append(LLMMessage(role="user", content=message))

        try:
            llm_resp = await self._llm.complete(messages, max_tokens=512, temperature=0.3)
            return ChatResult(
                content=llm_resp.content.strip(),
                match_method="llm-followup",
                provider=self._llm.provider_name(),
                usage=llm_resp.usage,
                trace_id=trace_id,
            )
        except Exception as e:
            logger.error(f"LLM follow-up failed: {e}")
            return ChatResult(
                content="couldn't generate a follow-up response. try asking again.",
                state="ERROR",
                trace_id=trace_id,
            )

    async def _llm_fallback(
        self,
        message: str,
        history: list[ChatTurn],
        trace_id: str,
    ) -> ChatResult:
        """LLM-only response when Glyphh model has no match."""
        extra = (
            "\n\nThe knowledge model didn't find a confident match for this query. "
            "Answer based on your general knowledge, but be upfront that the "
            "knowledge base didn't have a direct answer."
        )
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self._system_prompt + extra),
        ]
        for turn in history[-_MAX_HISTORY_TURNS:]:
            messages.append(LLMMessage(role=turn.role, content=turn.content))
        messages.append(LLMMessage(role="user", content=message))

        try:
            llm_resp = await self._llm.complete(messages, max_tokens=512, temperature=0.3)
            return ChatResult(
                content=llm_resp.content.strip(),
                match_method="llm-only",
                provider=self._llm.provider_name(),
                usage=llm_resp.usage,
                trace_id=trace_id,
            )
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return ChatResult(
                content="couldn't reach the language model. try again later.",
                state="ERROR",
                trace_id=trace_id,
            )
