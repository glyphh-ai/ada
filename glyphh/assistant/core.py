"""
Glyphh Assistant - thin client powered by a deployed Glyphh model.

Online mode: queries the runtime at POST /{org_id}/{model_id}/query
Offline mode: loads assistant.glyphh locally, encodes + similarity search

Uses only public SDK primitives: Encoder, Concept, GlyphhModel, SimilarityCalculator.
No IntentEncoder, no IntentClassifier, no ProcedureRegistry.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from glyphh.assistant.spell_fix import OfflineSpellFixer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

# Hardcoded assistant model coordinates — same regardless of runtime
ASSISTANT_ORG_ID = "glyphh"
ASSISTANT_MODEL_ID = "assistant"
PRODUCTION_RUNTIME_URL = "https://runtime.glyphh.ai"


@dataclass
class AssistantConfig:
    """Configuration for the assistant."""
    runtime_url: Optional[str] = None  # Custom runtime (e.g. local docker)
    platform_url: Optional[str] = None  # Platform API for LLM-assisted chat
    org_id: str = ASSISTANT_ORG_ID
    model_id: str = ASSISTANT_MODEL_ID
    model_path: Optional[Path] = None  # Local .glyphh fallback
    threshold: float = 0.35
    top_k: int = 3


@dataclass
class AssistantResponse:
    """Response from the assistant — mirrors NLQueryResult structure.
    
    States: DONE, ASK, BLOCKED, AUTH_REQUIRED, ERROR
    fact_tree is included for DONE responses to stay aligned with the NL pipeline.
    """
    state: str  # DONE, ASK, BLOCKED, AUTH_REQUIRED, ERROR
    content: str
    fact_tree: Optional[dict] = None
    command: Optional[str] = None
    code: Optional[str] = None
    confidence: float = 0.0
    match_method: str = "offline"  # "offline", "rules", "llm", "auto", "hybrid"
    query_type: str = ""
    trace_id: str = ""
    missing_slot: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Verb / object / domain extraction for query -> Concept
# ---------------------------------------------------------------------------

_VERB_MAP = {
    "how": "help", "what": "explain", "show": "help", "tell": "explain",
    "explain": "explain", "describe": "explain", "why": "explain",
    "create": "build", "build": "build", "init": "build", "initialize": "build",
    "add": "build", "make": "build", "setup": "build", "set": "build",
    "find": "query", "search": "query", "look": "query", "filter": "query",
    "query": "query", "list": "query", "get": "query",
    "deploy": "deploy", "publish": "deploy", "release": "deploy",
    "run": "deploy", "start": "deploy", "launch": "deploy",
    "delete": "manage", "remove": "manage", "update": "manage",
    "navigate": "navigate", "go": "navigate", "open": "navigate",
    "type": "navigate",
}

_KNOWN_OBJECTS = [
    "model", "query", "data", "runtime", "encoder", "config",
    "gql", "hdc", "vector", "glyph", "glyphh", "concept", "role", "segment",
    "layer", "procedure", "pattern", "account", "package", "results",
    "threshold", "similarity", "filter", "limit", "schema",
]

_STOP_WORDS = {
    "how", "do", "i", "a", "the", "to", "is", "what", "my", "an",
    "can", "does", "it", "in", "on", "for", "with", "me", "about",
}

# Common domain misspellings — checked before Levenshtein matching
_COMMON_MISSPELLINGS = {
    "modle": "model", "modl": "model", "moddel": "model",
    "qurey": "query", "qeury": "query", "qury": "query",
    "bild": "build", "buld": "build", "biuld": "build",
    "deplpy": "deploy", "depoly": "deploy", "delpoy": "deploy",
    "encodr": "encoder", "encder": "encoder", "encoer": "encoder",
    "runtme": "runtime", "runtim": "runtime", "runime": "runtime",
    "glpyh": "glyph", "gylph": "glyph", "glpyph": "glyph",
    "glyhh": "glyphh", "glpyhh": "glyphh", "gylphh": "glyphh",
    "concpt": "concept", "concpet": "concept",
    "simlar": "similar", "similiar": "similar",
    "threshld": "threshold", "thresold": "threshold",
    "proceedure": "procedure", "procedre": "procedure",
    "scema": "schema", "shema": "schema",
    "confg": "config", "cofig": "config",
    "sigup": "signup", "singup": "signup",
}


def _extract_verb(words: list[str]) -> str:
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _VERB_MAP:
            return _VERB_MAP[clean]
    return "help"


def _extract_object(words: list[str]) -> str:
    for obj in _KNOWN_OBJECTS:
        if obj in words:
            return obj
    for w in reversed(words):
        clean = re.sub(r"[^a-z]", "", w)
        if clean and clean not in _STOP_WORDS:
            return clean
    return "general"


def _infer_domain(words: list[str]) -> str:
    text = " ".join(words)
    if any(w in text for w in ["build", "create", "init", "add"]):
        return "build"
    if any(w in text for w in ["find", "search", "query", "similar"]):
        return "query"
    if any(w in text for w in ["deploy", "publish", "release"]):
        return "deploy"
    if any(w in text for w in ["what", "explain", "how does"]):
        return "explain"
    return "help"


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

class Assistant:
    """
    Glyphh Assistant - online-first, offline fallback.

    Online: POST to runtime endpoint, parse NLQueryResult.
    Offline: load .glyphh, encode query as Concept, similarity search.
    """

    # Short timeout for runtime probes (seconds)
    _ONLINE_TIMEOUT = 2
    # Separate timeout for normalization calls (seconds) — allows independent tuning
    _NORMALIZE_TIMEOUT = 3

    def __init__(self, config: Optional[AssistantConfig] = None):
        self.config = config or AssistantConfig()
        # Offline model state (lazy loaded)
        self._model = None
        self._encoder = None
        self._calculator = None
        self._exemplar_glyphs = None
        self._exemplar_metadata = None
        # Track which runtimes/platforms are unreachable so we only try once
        self._failed_runtimes: set = set()
        self._failed_platforms: set = set()
        # Conversation history for LLM context
        self._history: list = []
    def load(self):
        """Pre-load the offline model if a model_path is configured."""
        if self.config.model_path or not self.config.runtime_url:
            self._load_offline_model()

    def ask(self, query: str) -> AssistantResponse:
        """Process a query. Try offline (pure Glyphh) first, then LLM fallback.

        The offline model is the primary path — loads the .glyphh file
        into memory and does HDC similarity search. Only falls back to
        the platform LLM when the offline model has no confident match
        or the query looks like a conversational follow-up.
        """
        # 0. Offline local model (primary — pure Glyphh)
        offline_response = self._ask_offline(query)

        # If offline got a confident match AND this isn't a follow-up, use it
        is_followup = self._is_followup_query(query)
        if (offline_response.state == "DONE"
                and offline_response.confidence >= self.config.threshold
                and not is_followup):
            self._update_history(query, offline_response.content)
            return offline_response

        # 1. Platform LLM-assisted chat (fallback for low-confidence or errors)
        platform_urls = []
        if self.config.platform_url and self.config.platform_url not in self._failed_platforms:
            platform_urls.append(self.config.platform_url)
        # Also try well-known platform URL
        default_platform = "https://api.glyphh.ai"
        if default_platform not in self._failed_platforms and default_platform != self.config.platform_url:
            platform_urls.append(default_platform)

        for purl in platform_urls:
            try:
                return self._ask_platform(query, purl, is_followup=is_followup)
            except Exception as e:
                logger.debug(f"Platform unavailable ({purl}): {e}")
                self._failed_platforms.add(purl)

        # 2. No platform available — return the offline response as-is
        self._update_history(query, offline_response.content)
        return offline_response

    def _update_history(self, query: str, response_content: str):
        """Append a user/assistant turn to conversation history."""
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": response_content})
        if len(self._history) > 20:
            self._history = self._history[-12:]

    def _is_followup_query(self, query: str) -> bool:
        """Detect if a query is a conversational follow-up using the Glyphh model.

        Uses numeric binning on the context segment to tag exemplars as
        followup (bin 0) vs standalone (bin 1). Then compares the query's
        intent segment against followup exemplars specifically.

        A query is a follow-up if:
        1. Its best intent match among followup exemplars exceeds a threshold, AND
        2. The followup match is competitive with the standalone match
           (standalone doesn't dominate by a large margin)
        """
        if not self._history or self._encoder is None:
            return False

        try:
            # Encode query as standalone (neutral — we're testing intent similarity)
            query_concept = self._extract_query_concept(query, context_type_value=1.0)
            query_glyph = self._encoder.encode(query_concept)

            q_intent = (
                query_glyph.layers.get("router")
                and query_glyph.layers["router"].segments.get("intent")
            )
            if not q_intent:
                return False

            from glyphh.core.rust_ops import cosine_similarity as _cos_sim

            best_followup_intent = 0.0
            best_standalone_intent = 0.0

            for i, eg in enumerate(self._exemplar_glyphs):
                e_intent = (
                    eg.layers.get("router")
                    and eg.layers["router"].segments.get("intent")
                )
                if not e_intent:
                    continue

                score = float(_cos_sim(q_intent.cortex.data, e_intent.cortex.data))
                ct = self._exemplar_metadata[i].get("content_type", "")
                if ct == "followup":
                    best_followup_intent = max(best_followup_intent, score)
                else:
                    best_standalone_intent = max(best_standalone_intent, score)

            # The query is a follow-up if:
            # - It matches a followup exemplar reasonably well (absolute threshold)
            # - AND standalone doesn't dominate by too much (relative check)
            # The 0.35 threshold catches patterns like "how does it work?" (0.42)
            # The 0.25 margin prevents "what is a glyph?" (fu=0.30, sa=0.60) from
            # being classified as followup
            if best_followup_intent < 0.35:
                return False
            standalone_dominance = best_standalone_intent - best_followup_intent
            return standalone_dominance < 0.25

        except Exception as e:
            logger.debug(f"Follow-up detection failed: {e}")
            return False

        except Exception as e:
            logger.debug(f"Follow-up detection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Platform mode (LLM ↔ Glyphh orchestration)
    # ------------------------------------------------------------------

    _PLATFORM_TIMEOUT = 15  # seconds — LLM calls take longer than raw runtime

    def _ask_platform(self, query: str, platform_url: str, is_followup: bool = False) -> AssistantResponse:
        """POST to platform /api/v1/assistant/chat for LLM-assisted response."""
        import urllib.request
        import urllib.error

        url = f"{platform_url.rstrip('/')}/api/v1/assistant/chat"
        payload = json.dumps({
            "message": query,
            "history": [
                {"role": h["role"], "content": h["content"]}
                for h in self._history[-6:]
            ],
            "is_followup": is_followup,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._PLATFORM_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            raise ConnectionError(f"Platform unreachable: {e}")

        # Update conversation history
        self._update_history(query, data.get("content", ""))

        return AssistantResponse(
            state=data.get("state", "DONE"),
            content=data.get("content", ""),
            fact_tree=data.get("fact_tree"),
            command=data.get("command"),
            code=data.get("code"),
            confidence=data.get("confidence", 0.0),
            match_method=data.get("match_method", "llm"),
            trace_id=data.get("trace_id", ""),
            metadata={
                "provider": data.get("provider", ""),
                "usage": data.get("usage", {}),
                "why": {
                    "query": query,
                    "extracted": {},
                    "matched_question": None,
                    "matched_content_type": None,
                    "score": data.get("confidence", 0.0),
                    "runners_up": [],
                },
            },
        )

    # ------------------------------------------------------------------
    # Online mode
    # ------------------------------------------------------------------

    def _ask_online(self, query: str, runtime_url: str) -> AssistantResponse:
        """POST to runtime endpoint, parse NLQueryResult JSON."""
        import urllib.request
        import urllib.error

        url = (
            f"{runtime_url.rstrip('/')}"
            f"/{self.config.org_id}/{self.config.model_id}/query"
        )
        payload = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._ONLINE_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            raise ConnectionError(f"Runtime unreachable: {e}")

        return self._parse_nl_query_result(data)

    def _normalize_via_runtime(self, query: str) -> tuple[str, bool]:
        """Call POST /{org_id}/{model_id}/normalize on an available runtime.

        Tries custom runtime URL first, then production — same cascade as
        ``_ask_online`` — but skips any URL already in ``_failed_runtimes``.

        Returns ``(normalized_text, was_changed)``.  On *any* failure the
        original query is returned unchanged with ``changed=False``.
        """
        import urllib.request
        import urllib.error

        urls_to_try: list[str] = []
        if self.config.runtime_url and self.config.runtime_url not in self._failed_runtimes:
            urls_to_try.append(self.config.runtime_url)
        if PRODUCTION_RUNTIME_URL not in self._failed_runtimes:
            urls_to_try.append(PRODUCTION_RUNTIME_URL)

        for runtime_url in urls_to_try:
            url = (
                f"{runtime_url.rstrip('/')}"
                f"/{self.config.org_id}/{self.config.model_id}/normalize"
            )
            payload = json.dumps({"query": query}).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._NORMALIZE_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data.get("normalized_query", query), data.get("changed", False)
            except Exception as e:
                logger.debug(f"Normalize call failed ({runtime_url}): {e}")
                continue

        return query, False

    def _parse_nl_query_result(self, data: dict) -> AssistantResponse:
        """Parse a NLQueryResult JSON into an AssistantResponse."""
        state = data.get("state", "ERROR")
        fact_tree = data.get("fact_tree", {})
        content = fact_tree.get("text", data.get("message", "")) if fact_tree else ""
        command = data.get("command")
        code = data.get("code")
        confidence = data.get("confidence", 0.0)
        match_method = data.get("match_method", "online")
        query_type = data.get("query_type", "")
        trace_id = data.get("trace_id", "")

        if state == "ASK":
            missing = data.get("missing_slot")
            return AssistantResponse(
                state="ASK", content=content, fact_tree=fact_tree,
                confidence=confidence, match_method=match_method,
                query_type=query_type, trace_id=trace_id,
                missing_slot=missing, metadata=data,
            )
        if state == "DONE":
            return AssistantResponse(
                state="DONE", content=content, fact_tree=fact_tree,
                command=command, code=code, confidence=confidence,
                match_method=match_method, query_type=query_type,
                trace_id=trace_id, metadata=data,
            )
        # ERROR or unknown
        return AssistantResponse(
            state="ERROR",
            content=content or "Something went wrong with the query.",
            fact_tree=fact_tree if fact_tree else None,
            match_method=match_method, trace_id=trace_id,
            metadata=data,
        )

    # ------------------------------------------------------------------
    # Offline mode
    # ------------------------------------------------------------------

    def _ask_offline(self, query: str) -> AssistantResponse:
        """Load local model, encode query, similarity search."""
        if self._encoder is None:
            self._load_offline_model()

        if self._encoder is None:
            return AssistantResponse(
                state="ERROR",
                content="Assistant model not found. Run `python scripts/build_assistant_model.py` to build it.",
            )

        query_concept = self._extract_query_concept(query)

        try:
            query_glyph = self._encoder.encode(query_concept)
        except Exception as e:
            logger.warning(f"Failed to encode query: {e}")
            return AssistantResponse(
                state="ERROR",
                content="I'm not sure what you're asking. Try 'help' to see what I can do.",
                match_method="offline",
                metadata={
                    "why": {
                        "query": query,
                        "extracted": query_concept.attributes,
                        "matched_question": None,
                        "matched_content_type": None,
                        "score": 0.0,
                        "reason": f"Encoding failed: {e}",
                        "runners_up": [],
                    },
                },
            )

        # Compare intent segments for better discrimination
        from glyphh.core.rust_ops import cosine_similarity as _cos_sim

        all_scores = []  # Track all scores for "why" even below threshold
        q_intent = (
            query_glyph.layers.get("router", None)
            and query_glyph.layers["router"].segments.get("intent")
        )
        for i, eg in enumerate(self._exemplar_glyphs):
            try:
                e_intent = (
                    eg.layers.get("router", None)
                    and eg.layers["router"].segments.get("intent")
                )
                if q_intent and e_intent:
                    score = float(_cos_sim(q_intent.cortex.data, e_intent.cortex.data))
                else:
                    # Fallback to global cortex
                    score = float(_cos_sim(
                        query_glyph.global_cortex.data, eg.global_cortex.data
                    ))
                all_scores.append((score, i))
            except Exception:
                continue

        all_scores.sort(key=lambda x: x[0], reverse=True)
        results = [(s, i) for s, i in all_scores[:self.config.top_k] if s >= self.config.threshold]

        # Build nearest misses for "why" (top 5 scores even if below threshold)
        nearest = []
        for score, idx in all_scores[:5]:
            nm = self._exemplar_metadata[idx]
            nearest.append({
                "score": score,
                "question": nm.get("original_question", ""),
                "content_type": nm.get("content_type", ""),
            })

        # Normalization tracking — set by the cascade below when applicable
        normalized_query = None
        normalization_method = None

        if not results:
            # ---------------------------------------------------------------
            # Normalization cascade: offline spell-fix → LLM normalization
            # ---------------------------------------------------------------

            # --- Tier 1: Offline spell-fix ---
            try:
                exemplar_questions = [
                    m.get("original_question", "")
                    for m in self._exemplar_metadata
                    if m.get("original_question")
                ]
                spell_fixer = OfflineSpellFixer(
                    known_objects=_KNOWN_OBJECTS,
                    verb_map=_VERB_MAP,
                    exemplar_questions=exemplar_questions,
                    common_misspellings=_COMMON_MISSPELLINGS,
                )
                corrected_text, was_changed = spell_fixer.fix(query)

                if was_changed:
                    corrected_concept = self._extract_query_concept(corrected_text)
                    corrected_glyph = self._encoder.encode(corrected_concept)

                    corrected_scores = []
                    cq_intent = (
                        corrected_glyph.layers.get("router", None)
                        and corrected_glyph.layers["router"].segments.get("intent")
                    )
                    for i, eg in enumerate(self._exemplar_glyphs):
                        try:
                            e_intent = (
                                eg.layers.get("router", None)
                                and eg.layers["router"].segments.get("intent")
                            )
                            if cq_intent and e_intent:
                                score = float(_cos_sim(cq_intent.cortex.data, e_intent.cortex.data))
                            else:
                                score = float(_cos_sim(
                                    corrected_glyph.global_cortex.data, eg.global_cortex.data
                                ))
                            corrected_scores.append((score, i))
                        except Exception:
                            continue

                    corrected_scores.sort(key=lambda x: x[0], reverse=True)
                    corrected_results = [
                        (s, i) for s, i in corrected_scores[:self.config.top_k]
                        if s >= self.config.threshold
                    ]

                    if corrected_results:
                        normalized_query = corrected_text
                        normalization_method = "offline_spell_fix"
                        results = corrected_results
                        all_scores = corrected_scores
                        query_concept = corrected_concept
            except Exception as e:
                logger.warning(f"Offline spell-fix failed: {e}")

            # --- Tier 2: Runtime LLM normalization ---
            if not results:
                runtime_available = (
                    (self.config.runtime_url and self.config.runtime_url not in self._failed_runtimes)
                    or PRODUCTION_RUNTIME_URL not in self._failed_runtimes
                )
                if runtime_available:
                    llm_text, llm_changed = self._normalize_via_runtime(query)

                    if llm_changed and llm_text != query:
                        try:
                            llm_concept = self._extract_query_concept(llm_text)
                            llm_glyph = self._encoder.encode(llm_concept)

                            llm_scores = []
                            lq_intent = (
                                llm_glyph.layers.get("router", None)
                                and llm_glyph.layers["router"].segments.get("intent")
                            )
                            for i, eg in enumerate(self._exemplar_glyphs):
                                try:
                                    e_intent = (
                                        eg.layers.get("router", None)
                                        and eg.layers["router"].segments.get("intent")
                                    )
                                    if lq_intent and e_intent:
                                        score = float(_cos_sim(lq_intent.cortex.data, e_intent.cortex.data))
                                    else:
                                        score = float(_cos_sim(
                                            llm_glyph.global_cortex.data, eg.global_cortex.data
                                        ))
                                    llm_scores.append((score, i))
                                except Exception:
                                    continue

                            llm_scores.sort(key=lambda x: x[0], reverse=True)
                            llm_results = [
                                (s, i) for s, i in llm_scores[:self.config.top_k]
                                if s >= self.config.threshold
                            ]

                            if llm_results:
                                normalized_query = llm_text
                                normalization_method = "llm"
                                results = llm_results
                                all_scores = llm_scores
                                query_concept = llm_concept
                        except Exception as e:
                            logger.warning(f"Re-encode after LLM normalization failed: {e}")

            # --- Still no results: return low-confidence fallback ---
            if not results:
                best_score = all_scores[0][0] if all_scores else 0.0
                return AssistantResponse(
                    state="ERROR",
                    content="I'm not sure what you're asking. Try 'help' to see what I can do.",
                    confidence=best_score,
                    match_method="offline",
                    metadata={
                        "why": {
                            "query": query,
                            "extracted": query_concept.attributes,
                            "matched_question": None,
                            "matched_content_type": None,
                            "score": best_score,
                            "reason": f"No results above threshold ({self.config.threshold:.0%})",
                            "runners_up": nearest,
                        },
                    },
                )

        best_score, best_idx = results[0]
        meta = self._exemplar_metadata[best_idx]

        # Build runners-up for "why" explainability (top 5 stack-ranked)
        runners_up = []
        for score, idx in all_scores[1:5]:
            rm = self._exemplar_metadata[idx]
            runners_up.append({
                "score": score,
                "question": rm.get("original_question", ""),
                "content_type": rm.get("content_type", ""),
            })

        # Build fact_tree aligned with NL pipeline structure
        fact_tree = {
            "text": meta.get("response", ""),
            "description": meta.get("original_question", ""),
            "children": [],
            "data_context": {
                "content_type": meta.get("content_type", ""),
                "source": "offline_model",
            },
        }
        if meta.get("code"):
            fact_tree["children"].append({
                "text": meta["code"],
                "description": "code",
                "children": [],
                "data_context": {},
            })

        # Build "why" explanation
        why = {
            "query": query,
            "extracted": query_concept.attributes,
            "matched_question": meta.get("original_question", ""),
            "matched_content_type": meta.get("content_type", ""),
            "score": best_score,
            "runners_up": runners_up,
        }

        # Determine match_method based on normalization
        if normalization_method == "offline_spell_fix":
            match_method = "offline_corrected"
        elif normalization_method == "llm":
            match_method = "llm_assisted"
        else:
            match_method = "offline"

        # Add normalization metadata when normalization was applied
        if normalized_query is not None and normalization_method is not None:
            why["original_query"] = query
            why["normalized_query"] = normalized_query
            why["normalization_method"] = normalization_method

        import uuid
        return AssistantResponse(
            state="DONE",
            content=meta.get("response", ""),
            fact_tree=fact_tree,
            command=meta.get("command"),
            code=meta.get("code"),
            confidence=best_score,
            match_method=match_method,
            query_type=meta.get("content_type", ""),
            trace_id=str(uuid.uuid4())[:8],
            metadata={
                "content_type": meta.get("content_type", ""),
                "original_question": meta.get("original_question", ""),
                "why": why,
            },
        )

    def _load_offline_model(self):
        """Lazy-load the .glyphh file for offline mode."""
        from glyphh.model.package import GlyphhModel
        from glyphh.encoder import Encoder
        from glyphh.similarity import SimilarityCalculator

        model_path = self.config.model_path
        if model_path is None:
            # Default: look next to the data dir
            model_path = Path(__file__).parent / "model" / "assistant.glyphh"

        if not Path(model_path).exists():
            logger.warning(f"Assistant model not found at {model_path}")
            return

        try:
            self._model = GlyphhModel.from_file(str(model_path))
        except Exception as e:
            logger.warning(f"Failed to load assistant model: {e}")
            return

        self._encoder = Encoder(self._model.encoder_config)
        self._calculator = SimilarityCalculator()

        # Encode all embedded concepts into glyphs
        concepts = self._model.get_concepts() or []
        self._exemplar_glyphs = []
        self._exemplar_metadata = []

        from glyphh.core.types import Concept

        for i, record in enumerate(concepts):
            meta = record.get("_metadata", {})
            # Flatten hierarchical record to flat attributes
            attrs = {}
            for layer_name, layer_data in record.items():
                if layer_name.startswith("_"):
                    continue
                if isinstance(layer_data, dict):
                    for seg_name, seg_data in layer_data.items():
                        if isinstance(seg_data, dict):
                            for role_name, value in seg_data.items():
                                attrs[role_name] = value

            concept = Concept(name=f"exemplar_{i}", attributes=attrs, metadata=meta)
            try:
                glyph = self._encoder.encode(concept)
                self._exemplar_glyphs.append(glyph)
                self._exemplar_metadata.append(meta)
            except Exception as e:
                logger.warning(f"Failed to encode exemplar {i}: {e}")

        logger.info(f"Loaded {len(self._exemplar_glyphs)} exemplars from {model_path}")

    def _extract_query_concept(self, query: str, context_type_value: float = 1.0):
        """Extract intent attributes from a NL query string.
        
        Args:
            query: Natural language query
            context_type_value: Numeric bin value for context_type
                (0.0 = followup, 1.0 = standalone). Defaults to standalone.
        """
        from glyphh.core.types import Concept

        # Strip punctuation so "what is glyphh?" matches same as "what is glyphh"
        cleaned = re.sub(r"[^\w\s]", "", query.lower())
        words = cleaned.split()
        verb = _extract_verb(words)
        obj = _extract_object(words)
        domain = _infer_domain(words)
        keywords = " ".join(words)

        return Concept(
            name=f"query_{abs(hash(query)) % 100000000:08d}",
            attributes={
                "verb": verb,
                "object": obj,
                "domain": domain,
                "keywords": keywords,
                "action_type": "",
                "action_id": "",
                "context_type": context_type_value,
            },
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_command(self, text: str) -> bool:
        """Check if text is a CLI command (vs natural language)."""
        if not text:
            return False
        first_word = text.split()[0].lower()
        return first_word in {
            "auth", "build", "test", "package", "runtime",
            "query", "procedure", "models", "hub", "docs", "demo",
            "help", "clear", "exit", "quit",
        }
