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


def _extract_glyph_metadata(response_data: dict) -> dict:
    """Extract response/command/code from glyph metadata in NL query results.

    Looks through fact_tree children for glyph metadata and surfaces
    the top match's response, command, and code fields.
    """
    fact_tree = response_data.get("fact_tree")
    if not fact_tree:
        return response_data

    children = fact_tree.get("children", []) if isinstance(fact_tree, dict) else []
    if not children:
        return response_data

    top = children[0] if children else {}
    metadata = top.get("data_context", {}) or top.get("metadata", {})

    if metadata.get("response") and not response_data.get("command"):
        response_data.setdefault("command", metadata.get("command"))
    if metadata.get("code") and not response_data.get("code"):
        response_data.setdefault("code", metadata.get("code"))

    # Use response from metadata as fact_tree text if not already set
    if metadata.get("response"):
        if isinstance(fact_tree, dict) and not fact_tree.get("text"):
            fact_tree["text"] = metadata["response"]

    return response_data


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

            # Zero confidence: check if the model is actually functional
            # before letting the LLM generate ungrounded responses
            if confidence == 0.0:
                model_ok = await self._check_model_health()
                if not model_ok:
                    return ChatResult(
                        content=(
                            f"the {self.model_id} model is not deployed or has no data. "
                            "please check that the model directory exists and the runtime "
                            "has deployed it successfully."
                        ),
                        state="ERROR",
                        match_method="none",
                        trace_id=trace_id,
                    )
                # Model is healthy but had zero confidence — no match
                return ChatResult(
                    content="no matching answer found in the knowledge base.",
                    state="NO_MATCH",
                    confidence=0.0,
                    match_method="glyphh",
                    provider="glyphh",
                    trace_id=trace_id,
                )

            return await self._llm_fallback(message, history, trace_id)

        # No LLM — return whatever Glyphh gave us
        if confidence == 0.0:
            model_ok = await self._check_model_health()
            if not model_ok:
                return ChatResult(
                    content=(
                        f"the {self.model_id} model is not deployed or has no data. "
                        "please check that the model directory exists and the runtime "
                        "has deployed it successfully."
                    ),
                    state="ERROR",
                    match_method="none",
                    trace_id=trace_id,
                )
            return ChatResult(
                content="no matching answer found in the knowledge base.",
                state="NO_MATCH",
                confidence=0.0,
                match_method="glyphh",
                provider="glyphh",
                trace_id=trace_id,
            )

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
        """Query the deployed model through the unified pipeline.

        Flow:
          1. Check stored procedures (GQL queries registered for this model)
          2. DB-backed similarity search via QueryService
          3. Return FactTree-shaped result for LLM synthesis
        """
        # Step 1: Check stored procedures first (highest priority)
        proc_result = await self._check_stored_procedures(query)
        if proc_result is not None:
            return proc_result

        # Step 2: DB-backed query (similarity search with custom encode_query_fn)
        return await self._query_generic(query)

    async def _check_model_health(self) -> bool:
        """Check if the model has glyphs deployed and is functional.

        Returns True if the model has data in the DB, False otherwise.
        """
        try:
            from domains.models.storage import GlyphStorage
            from infrastructure.database import async_session_maker

            async with async_session_maker() as session:
                storage = GlyphStorage(session)
                count = await storage.count_glyphs(self.org_id, self.model_id)
                return count > 0
        except Exception as e:
            logger.warning(f"Model health check failed ({self.org_id}/{self.model_id}): {e}")
            return False


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

            result = self._parse_fact_tree(fact_tree, match_method="stored_procedure")
            # Override confidence for stored procedures (exact match)
            result["confidence"] = 1.0
            return result

        except Exception as e:
            logger.warning(f"Procedure execution failed ({procedure.name}): {e}")
            return None



    async def _query_generic(self, query: str) -> dict[str, Any] | None:
        """Query a model via DB similarity search.

        Uses QueryService.similarity_search which now dispatches to
        the model's custom encode_query_fn when available.
        Extracts response/command/code from glyph metadata.
        """
        try:
            from domains.query.service import QueryService
            from domains.models.schemas import SimilaritySearchRequest
            from infrastructure.database import async_session_maker
            from main import model_manager

            if model_manager is None:
                logger.warning("Model manager not initialized")
                return None

            service = QueryService(model_manager, async_session_maker)

            # Chat pipeline always uses direct similarity search.
            # The NL intent matcher is unreliable for chat queries — it
            # misclassifies natural language questions as count/fact_tree/etc.
            # Similarity search with the custom encode_query_fn is the
            # correct path for all chat queries.
            request = SimilaritySearchRequest(query=query, top_k=10)
            fact_tree = await service.similarity_search(
                org_id=self.org_id,
                model_id=self.model_id,
                request=request,
            )

            return self._parse_fact_tree(fact_tree)

        except Exception as e:
            logger.warning(f"Model query failed ({self.org_id}/{self.model_id}): {e}")
            return None

    def _parse_nl_result(self, result: Any) -> dict[str, Any] | None:
        """Parse an NLQueryResult into the dict format chat() expects.

        The NL pipeline returns a FactTree with SDK structure
        (description/value/children). We need to extract the top match's
        metadata (response, command, code) and build a flat dict with
        a fact_tree that has a 'text' key.
        """
        from domains.nl_query.service import ResponseState

        state_str = result.state.value if hasattr(result.state, "value") else str(result.state)

        # Handle error states from the NL pipeline
        if result.state == ResponseState.ERROR:
            error_msg = ""
            if result.error:
                error_msg = result.error.message
            return {
                "state": "ERROR",
                "confidence": 0.0,
                "match_method": result.match_method or "none",
                "command": None,
                "code": None,
                "fact_tree": {
                    "text": error_msg,
                    "description": "error",
                    "value": error_msg,
                    "children": [],
                    "citations": [],
                    "data_context": {},
                },
            }

        # For DONE state, extract match data from the FactTree
        if result.fact_tree is None:
            return {
                "state": state_str,
                "confidence": result.confidence,
                "match_method": result.match_method or "none",
                "command": None,
                "code": None,
                "fact_tree": {"text": "", "children": [], "citations": [], "data_context": {}},
            }

        return self._parse_fact_tree(
            result.fact_tree,
            confidence_override=result.confidence,
            match_method=result.match_method or "glyphh",
        )

    def _parse_fact_tree(
        self,
        fact_tree: Any,
        confidence_override: float | None = None,
        match_method: str = "glyphh",
    ) -> dict[str, Any]:
        """Parse an SDK FactTree into the dict format chat() expects.

        Walks the FactTree children to find match results, extracts
        glyph metadata (response, command, code), and builds a flat
        dict with a fact_tree sub-dict containing a 'text' key.
        """
        ft_json = fact_tree.to_json() if hasattr(fact_tree, "to_json") else {}
        ft_dict = json.loads(ft_json) if isinstance(ft_json, str) else ft_json

        # Walk children to find match results
        # FactTree structure: root -> [query, results, metadata]
        # results -> [match_1, match_2, ...]
        # Each match has value: {concept_text, metadata, similarity_score, ...}
        matches = []
        all_children = ft_dict.get("children", [])

        for child in all_children:
            # Check if this is a "results" group node
            if child.get("description", "").lower() == "results":
                matches = child.get("children", [])
                break
            # Or a direct match node (Match 1, Match 2, ...)
            desc = child.get("description", "")
            if desc.startswith("Match "):
                matches.append(child)

        # Also check for error nodes
        for child in all_children:
            if child.get("description", "").lower() == "error details":
                error_msg = child.get("value", "")
                error_type = (child.get("data_context") or {}).get("error_type", "")
                return {
                    "state": "ERROR",
                    "confidence": 0.0,
                    "match_method": match_method,
                    "command": None,
                    "code": None,
                    "fact_tree": {
                        "text": error_msg,
                        "description": error_type,
                        "value": error_msg,
                        "children": all_children,
                        "citations": [],
                        "data_context": {"error_type": error_type},
                    },
                }

        # Extract top match
        text = ""
        confidence = 0.0
        command = None
        code = None
        top_metadata = {}
        result_children = []

        if matches:
            top = matches[0]
            top_value = top.get("value", {})
            if isinstance(top_value, dict):
                top_metadata = top_value.get("metadata", {})
                text = top_metadata.get("response", "") or top_value.get("concept_text", "")
                command = top_metadata.get("command")
                code = top_metadata.get("code")
                confidence = top_value.get("final_score", 0.0) or top_value.get("similarity_score", 0.0)

            # Build children in the format _extract_glyph_metadata expects
            for m in matches:
                mv = m.get("value", {})
                if isinstance(mv, dict):
                    result_children.append({
                        "text": (mv.get("metadata") or {}).get("response", "") or mv.get("concept_text", ""),
                        "data_context": mv.get("metadata", {}),
                        "confidence": mv.get("final_score", 0.0),
                    })

        if confidence_override is not None:
            # Use the similarity score as primary confidence (it reflects
            # actual match quality). Only fall back to intent confidence
            # when there are no match results (e.g. count operations).
            if confidence == 0.0 and not matches:
                confidence = confidence_override

        return {
            "state": "DONE" if text else "ERROR",
            "confidence": confidence,
            "match_method": match_method,
            "command": command,
            "code": code,
            "fact_tree": {
                "text": text,
                "description": f"{self.model_id} response",
                "value": text,
                "children": result_children,
                "citations": ft_dict.get("citations", []),
                "data_context": top_metadata,
            },
        }

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
