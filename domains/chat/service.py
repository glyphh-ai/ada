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

        Every query — including follow-ups — goes through Glyphh first.
        The LLM is only used when Glyphh has no confident match, and even
        then it's grounded in whatever Glyphh facts are available.

        1. Query the Glyphh model
        2. If confident match with content: return pure Glyphh answer
        3. If follow-up with low confidence: LLM with history + Glyphh facts
        4. If low confidence, not follow-up: LLM fallback with Glyphh facts
        5. No LLM available: return raw Glyphh result or error
        """
        trace_id = str(uuid.uuid4())[:8]
        history = history or []

        # Always query the Glyphh model first — no exceptions
        glyphh_result = await self._query_model(message)

        if glyphh_result is None:
            # Model query failed entirely
            if is_followup and history and self._llm:
                return await self._llm_with_history(message, history, trace_id)
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
        state = glyphh_result.get("state", "DONE")

        # Pure Glyphh path: confident match with content → return directly
        # This applies to ALL queries including follow-ups
        if raw_content and state == "DONE" and confidence >= 0.35:
            return ChatResult(
                content=raw_content,
                command=command,
                code=code,
                confidence=confidence,
                match_method=match_method,
                fact_tree=fact_tree,
                state=state,
                provider="glyphh",
                trace_id=trace_id,
            )

        # Low confidence — LLM assists, grounded in Glyphh facts
        if self._llm:
            if is_followup and history:
                return await self._llm_with_history(
                    message, history, trace_id, glyphh_result=glyphh_result,
                )
            if raw_content and confidence > 0.0:
                return await self._synthesize(message, glyphh_result, history, trace_id)
            return await self._llm_fallback(message, history, trace_id)

        # No LLM — return whatever Glyphh gave us
        return ChatResult(
            content=raw_content or "no answer found.",
            confidence=confidence,
            match_method=match_method,
            fact_tree=fact_tree,
            state=state,
            provider="glyphh",
            trace_id=trace_id,
        )

    async def _query_model(self, query: str) -> dict[str, Any] | None:
        """Query the deployed model through the unified NL pipeline.

        Flow for every model:
          1. Check stored procedures (GQL queries registered for this model)
          2. If model has embedded concepts (.glyphh with custom encoder):
             encode with model's encoder → similarity search against concepts
          3. Otherwise: generic NL pipeline (IntentMatcher → QueryService)
          4. Return FactTree-shaped result for LLM synthesis

        No special cases — every model goes through the same pipeline.
        """
        # Step 1: Check stored procedures first (highest priority)
        proc_result = await self._check_stored_procedures(query)
        if proc_result is not None:
            return proc_result

        # Step 2: Try embedded concept search (models with .glyphh + custom encoder)
        concept_result = await self._query_embedded_concepts(query)
        if concept_result is not None:
            return concept_result

        # Step 3: Generic NL pipeline (DB-backed models)
        return await self._query_generic(query)

    async def _query_embedded_concepts(
        self, query: str,
    ) -> dict[str, Any] | None:
        """Query a model's embedded concepts via its custom HDC encoder.

        This is the path for models that ship with a .glyphh file containing
        pre-encoded concepts (like the assistant model). Uses the model's own
        encoder for similarity search against embedded exemplars.
        """
        from domains.models import load_model
        from pathlib import Path

        runtime_root = Path(__file__).parent.parent.parent
        model_dir = runtime_root / "models" / self.model_id
        if not model_dir.exists():
            model_dir = runtime_root / "custom_models" / self.model_id
        if not model_dir.exists():
            return None

        loaded = load_model(model_dir)
        if not loaded.has_custom_encoder or not loaded.glyphh_path:
            return None

        try:
            cache_key = f"_model_{self.model_id}"
            if not hasattr(ModelChatService, cache_key):
                from glyphh.assistant.core import Assistant, AssistantConfig

                config = AssistantConfig(
                    model_path=loaded.glyphh_path,
                    threshold=0.35,
                )
                assistant = Assistant(config)
                assistant.load()
                setattr(ModelChatService, cache_key, assistant)

            assistant = getattr(ModelChatService, cache_key)
            response = assistant._ask_offline(query)

            return {
                "state": response.state,
                "confidence": response.confidence,
                "match_method": response.match_method,
                "command": response.command,
                "code": response.code,
                "fact_tree": {
                    "text": response.content,
                    "description": f"{self.model_id} response",
                    "value": response.content,
                    "children": [],
                    "citations": [],
                    "data_context": response.metadata or {},
                },
            }

        except Exception as e:
            logger.warning(f"Embedded concept query failed ({self.model_id}): {e}")
            return None
    async def _check_stored_procedures(self, query: str) -> dict[str, Any] | None:
        """Check if query matches a stored procedure for this model.

        Stored procedures are GQL queries registered via the API with
        lexicons for NL matching. They take priority over all other paths.
        """
        try:
            from domains.procedures.service import StoredProcedureService
            from infrastructure.database import async_session_maker

            proc_service = StoredProcedureService(async_session_maker)
            procedures = await proc_service.list(self.org_id, self.model_id)

            if not procedures:
                return None

            # Simple lexicon matching — find best procedure match
            best_match = None
            best_score = 0.0
            query_lower = query.lower()
            query_words = set(query_lower.split())

            for proc in procedures:
                for lexicon in proc.lexicons:
                    lex_words = set(lexicon.lower().split())
                    if not lex_words:
                        continue
                    overlap = len(query_words & lex_words)
                    score = overlap / max(len(lex_words), 1)
                    if score > best_score:
                        best_score = score
                        best_match = proc

            if best_match and best_score >= 0.5:
                logger.info(
                    f"Stored procedure match: {best_match.name} "
                    f"(score={best_score:.2f})"
                )
                # Execute the procedure's GQL query
                return await self._execute_procedure(best_match)

            return None

        except Exception as e:
            logger.debug(f"Stored procedure check skipped: {e}")
            return None

    async def _execute_procedure(self, procedure: Any) -> dict[str, Any] | None:
        """Execute a stored procedure's GQL query and return as FactTree."""
        try:
            from domains.query.service import QueryService
            from domains.models.schemas import SimilaritySearchRequest
            from infrastructure.database import async_session_maker
            from main import model_manager

            if model_manager is None:
                return None

            service = QueryService(model_manager, async_session_maker)
            request = SimilaritySearchRequest(
                query=procedure.gql_query,
                top_k=10,
            )
            fact_tree = await service.similarity_search(
                org_id=self.org_id,
                model_id=self.model_id,
                request=request,
            )

            return {
                "state": "DONE",
                "confidence": 1.0,
                "match_method": "stored_procedure",
                "command": None,
                "code": None,
                "fact_tree": {
                    "text": fact_tree.text if hasattr(fact_tree, 'text') else str(fact_tree),
                    "description": f"Procedure: {procedure.name}",
                    "value": "",
                    "children": [],
                    "citations": [],
                    "data_context": {"procedure": procedure.name},
                },
            }

        except Exception as e:
            logger.warning(f"Procedure execution failed ({procedure.name}): {e}")
            return None



    async def _query_generic(self, query: str) -> dict[str, Any] | None:
        """Query a model via the generic NL query pipeline."""
        try:
            from domains.query.service import QueryService
            from infrastructure.database import async_session_maker
            from main import model_manager

            if model_manager is None:
                logger.warning("Model manager not initialized")
                return None

            service = QueryService(model_manager, async_session_maker)
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
        glyphh_result: dict[str, Any] | None = None,
    ) -> ChatResult:
        """LLM response for conversational follow-ups, grounded in Glyphh facts.

        If Glyphh returned a partial match, include those facts so the LLM
        doesn't hallucinate. If no Glyphh result, the LLM uses history only
        but is instructed to stay within the domain.
        """
        # Build grounding context from Glyphh result if available
        grounding = ""
        confidence = 0.0
        if glyphh_result:
            confidence = glyphh_result.get("confidence", 0.0)
            fact_tree = glyphh_result.get("fact_tree")
            raw_content = fact_tree.get("text", "") if fact_tree else ""
            if raw_content:
                grounding = (
                    f"\n\n--- Glyphh knowledge base (confidence: {confidence:.0%}) ---\n"
                    f"{raw_content}\n"
                    f"--- end ---\n\n"
                    "Use the knowledge base content above to ground your answer. "
                    "If the confidence is low, you may paraphrase but don't invent "
                    "features or details not in the knowledge base."
                )

        extra = (
            "\n\nThis is a conversational follow-up. The user is continuing "
            "a previous topic. Use the conversation history to give a relevant, "
            "grounded answer. Stay factual — don't speculate. If you don't have "
            "enough information from the knowledge base or conversation history, "
            "say so honestly rather than making things up."
        )
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self._system_prompt + extra + grounding),
        ]
        for turn in history[-_MAX_HISTORY_TURNS:]:
            messages.append(LLMMessage(role=turn.role, content=turn.content))
        messages.append(LLMMessage(role="user", content=message))

        try:
            llm_resp = await self._llm.complete(messages, max_tokens=512, temperature=0.3)
            return ChatResult(
                content=llm_resp.content.strip(),
                confidence=confidence,
                match_method="llm-followup",
                provider=self._llm.provider_name(),
                usage=llm_resp.usage,
                trace_id=trace_id,
            )
        except Exception as e:
            logger.error(f"LLM follow-up failed: {e}")
            # Fall back to raw Glyphh content if available
            if glyphh_result:
                ft = glyphh_result.get("fact_tree")
                fallback = ft.get("text", "") if ft else ""
                if fallback:
                    return ChatResult(
                        content=fallback,
                        confidence=confidence,
                        match_method="glyphh-fallback",
                        provider="glyphh",
                        trace_id=trace_id,
                    )
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
