"""
Brain router — cognitive routing through the full Ada architecture.

Uses the cognitive-router capability (a proper Ada model with HDC encoding,
exemplars, and fact trees) to route requests. Returns DONE with a capability
name, or ASK when ambiguous.

No shortcuts — goes through encode_query → similarity_search → fact_tree →
confidence gate, same as any other Ada query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from domains.brain.loader import ADA_ORG_ID, BrainState

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Where the brain decided to send this request."""
    capability: Optional[str] = None
    confidence: float = 0.0
    gate: str = "ASK"  # DONE or ASK from the fact tree
    llm_fallback: bool = False


class BrainRouter:
    """Routes requests through the cognitive-router Ada model.

    This is a proper Ada query — HDC encoding, similarity search against
    exemplars, fact tree, confidence gate. Not a shortcut.
    """

    ROUTER_CAPABILITY = "cognitive-router"

    def __init__(self, brain_state: BrainState, model_manager: Any = None):
        self._state = brain_state
        self._model_manager = model_manager
        self._session_factory = None
        self._initialized = False

    def initialize(self, session_factory: Any = None) -> None:
        """Check that the cognitive-router capability is loaded."""
        if session_factory:
            self._session_factory = session_factory

        if not self._model_manager:
            logger.warning("No model manager — routing degraded")
            return

        key = (ADA_ORG_ID, self.ROUTER_CAPABILITY)
        if key not in self._model_manager._models:
            logger.warning(
                f"Cognitive router capability not loaded — "
                f"check capabilities/{self.ROUTER_CAPABILITY}/ exists"
            )
            return

        self._initialized = True
        logger.info("Cognitive router online")

    async def route(self, query: str) -> RouteResult:
        """Route a query through the cognitive-router model.

        Full Ada pipeline: encode_query → similarity_search → fact_tree.
        Returns DONE with capability name, or ASK when ambiguous.
        """
        if not self._initialized or not self._session_factory:
            return RouteResult()

        try:
            from domains.query.service import QueryService
            query_service = QueryService(self._model_manager, self._session_factory)

            result = await query_service.similarity_search(
                org_id=ADA_ORG_ID,
                model_id=self.ROUTER_CAPABILITY,
                query=query,
            )

            if not result or not isinstance(result, dict):
                return RouteResult()

            return self._interpret_result(result)

        except Exception as e:
            logger.warning(f"Cognitive routing failed: {e}")
            return RouteResult()

    async def route_with_fallback(self, query: str, llm=None) -> RouteResult:
        """Route with LLM disambiguation skill as fallback."""
        result = await self.route(query)

        # DONE — confident route
        if result.gate == "DONE" and result.capability:
            return result

        # ASK — use LLM to disambiguate
        if llm and llm.available:
            cap_names = list(self._state.capabilities.keys())
            if cap_names:
                chosen = await llm.classify(query, cap_names)
                if chosen and chosen in self._state.capabilities:
                    return RouteResult(
                        capability=chosen,
                        confidence=0.5,
                        gate="DONE",
                        llm_fallback=True,
                    )

        return result

    def _interpret_result(self, result: dict) -> RouteResult:
        """Interpret the similarity search result into a routing decision.

        Reads the fact tree to extract the matched capability name
        and confidence. The fact tree's top match metadata contains
        the 'capability' field from the exemplar.
        """
        fact_tree = result.get("fact_tree", {})
        children = fact_tree.get("children", [])

        if not children:
            return RouteResult()

        top = children[0]
        similarity = top.get("value", 0)
        metadata = top.get("data_sample", {}) or top.get("metadata", {})
        capability = metadata.get("capability", "")

        # Also check the description field (concept_text from exemplar)
        if not capability:
            capability = top.get("description", "")

        # Validate capability exists
        if capability and capability not in self._state.capabilities:
            # Maybe it's a memory operation
            if capability == "memory":
                return RouteResult(capability=None, confidence=similarity, gate="DONE")
            # Unknown capability — treat as ASK
            return RouteResult(confidence=similarity, gate="ASK")

        # Confidence gate
        if similarity >= 0.45:
            return RouteResult(capability=capability, confidence=similarity, gate="DONE")
        elif similarity >= 0.30:
            return RouteResult(capability=capability, confidence=similarity, gate="ASK")
        else:
            return RouteResult(confidence=similarity, gate="ASK")
