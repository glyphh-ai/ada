"""
IntentExtractor — Generic NL intent extraction for Glyphh models.

Provides rule-based fast path (phrase patterns → synonym maps → impact ranking)
with HDC similarity fallback for unknown verbs and nouns.

Designed to be used as a first-class SDK feature by any Glyphh model that needs
intent parsing — toolrouter, BFCL, and custom domain models.

Exports:
    VERB_CONFIG   — EncoderConfig for verb synonym clusters (seed=53, 10000-dim)
    NOUN_CONFIG   — EncoderConfig for noun synonym clusters (seed=53, 10000-dim)
    IntentExtractor — extract(query) → {action, target, domain, keywords}
    get_extractor   — Singleton factory (cached base; fresh instance for packs)
    extract_intent  — Convenience wrapper
    extract_action  — Convenience wrapper
    extract_target  — Convenience wrapper
    infer_domain    — Convenience wrapper

Domain Packs:
    Load optional vocabulary extensions via IntentExtractor(packs=["filesystem"]).
    Pack files live at glyphh/intent/data/packs/{name}.json.
    Available packs: filesystem, trading, travel, social, math, vehicle.

Architecture:
    1. Phrase match  — substring check against phrases list (first match wins)
    2. Word map      — synonym → canonical with _IMPACT_RANK for multi-action queries
    3. Weak filter   — suppress nouns masquerading as verbs unless no strong alternative
    4. HDC fallback  — encode unknown word, cosine-compare against synonym cluster glyphs
    5. Domain infer  — weighted keyword signal scoring across all domain signal dicts
"""

import json
import re
from pathlib import Path

from glyphh import Encoder
from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept

# ---------------------------------------------------------------------------
# Data directory — bundled with the SDK package
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# HDC Configs — seed=53, 10000-dim (independent vector space from main models)
# ---------------------------------------------------------------------------

VERB_CONFIG = EncoderConfig(
    dimension=10000,
    seed=53,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="verbs",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="cluster",
                    roles=[
                        Role(name="canonical", similarity_weight=1.0),
                        Role(name="synonyms", similarity_weight=0.8, text_encoding="bag_of_words"),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="phrases",
                    roles=[
                        Role(name="phrases", similarity_weight=1.0, text_encoding="bag_of_words"),
                    ],
                ),
            ],
        ),
    ],
)

NOUN_CONFIG = EncoderConfig(
    dimension=10000,
    seed=53,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="nouns",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="cluster",
                    roles=[
                        Role(name="canonical", similarity_weight=1.0),
                        Role(name="synonyms", similarity_weight=0.8, text_encoding="bag_of_words"),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="phrases",
                    roles=[
                        Role(name="phrases", similarity_weight=1.0, text_encoding="bag_of_words"),
                    ],
                ),
            ],
        ),
    ],
)

# Role weights for HDC scoring
_VERB_ROLE_WEIGHTS = {"canonical": 1.0, "synonyms": 0.8, "phrases": 1.0}
_NOUN_ROLE_WEIGHTS = {"canonical": 1.0, "synonyms": 0.8, "phrases": 1.0}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# IntentExtractor
# ---------------------------------------------------------------------------

class IntentExtractor:
    """Generic NL intent extraction with rule-based fast path + HDC fallback.

    Usage:
        from glyphh.intent import IntentExtractor

        extractor = IntentExtractor()
        result = extractor.extract("Send a message to #general on Slack")
        # {"action": "send", "target": "channel", "domain": "messaging", "keywords": "message general slack"}

        # With domain packs:
        extractor = IntentExtractor(packs=["filesystem"])
        result = extractor.extract("List all files in the home directory")
        # {"action": "ls", "target": "directory", "domain": "filesystem", "keywords": "files home directory"}
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        packs: list[str] | None = None,
    ):
        data_root = Path(data_dir) if data_dir is not None else _DATA_DIR

        # Load base vocabulary
        self._action_data = _load_jsonl(data_root / "actions.jsonl")
        self._target_data = _load_jsonl(data_root / "targets.jsonl")
        self._domain_data = _load_jsonl(data_root / "domains.jsonl")
        self._phrase_data = _load_jsonl(data_root / "phrases.jsonl")
        self._stop_words = set(_load_json(data_root / "stop_words.json"))

        # Build domain signals from base domains
        self._domain_signals: dict[str, dict[str, int]] = {}
        for entry in self._domain_data:
            self._domain_signals[entry["domain"]] = entry["signals"]

        # Merge optional domain packs
        if packs:
            self._load_packs(packs, data_root / "packs")

        # Build rule-based lookup maps
        self._action_map: dict[str, str] = {}
        self._target_map: dict[str, str] = {}
        self._phrase_actions: list[tuple[str, str]] = []
        self._phrase_targets: list[tuple[str, str]] = []
        self._build_rule_maps()

        # Build HDC synonym cluster glyphs
        self._verb_encoder = Encoder(VERB_CONFIG)
        self._noun_encoder = Encoder(NOUN_CONFIG)
        self._verb_glyphs: list[tuple] = []  # (glyph, canonical)
        self._noun_glyphs: list[tuple] = []  # (glyph, canonical)
        self._build_clusters()

    # -------------------------------------------------------------------
    # Pack loading
    # -------------------------------------------------------------------

    def _load_packs(self, packs: list[str], packs_dir: Path) -> None:
        """Merge domain pack vocabularies into base data (deduplication by canonical)."""
        existing_actions = {e["canonical"] for e in self._action_data}
        existing_targets = {e["canonical"] for e in self._target_data}
        existing_phrases = {e["phrase"] for e in self._phrase_data}

        available = [p.stem for p in packs_dir.glob("*.json")]

        for pack_name in packs:
            pack_path = packs_dir / f"{pack_name}.json"
            if not pack_path.exists():
                raise FileNotFoundError(
                    f"Pack '{pack_name}' not found. Available packs: {available}"
                )
            pack = _load_json(pack_path)

            for entry in pack.get("actions", []):
                if entry["canonical"] not in existing_actions:
                    self._action_data.append(entry)
                    existing_actions.add(entry["canonical"])

            for entry in pack.get("targets", []):
                if entry["canonical"] not in existing_targets:
                    self._target_data.append(entry)
                    existing_targets.add(entry["canonical"])

            for entry in pack.get("phrases", []):
                if entry["phrase"] not in existing_phrases:
                    self._phrase_data.append(entry)
                    existing_phrases.add(entry["phrase"])

            for domain, signals in pack.get("domain_signals", {}).items():
                if domain not in self._domain_signals:
                    self._domain_signals[domain] = {}
                self._domain_signals[domain].update(signals)

    # -------------------------------------------------------------------
    # Rule map construction
    # -------------------------------------------------------------------

    def _build_rule_maps(self) -> None:
        for entry in self._action_data:
            canonical = entry["canonical"]
            for syn in entry["synonyms"]:
                self._action_map[syn] = canonical

        for entry in self._target_data:
            canonical = entry["canonical"]
            for syn in entry["synonyms"]:
                self._target_map[syn] = canonical

        for entry in self._phrase_data:
            self._phrase_actions.append((entry["phrase"], entry["action"]))

        for entry in self._target_data:
            canonical = entry["canonical"]
            for phrase in entry.get("phrases", []):
                self._phrase_targets.append((phrase, canonical))

    def _build_clusters(self) -> None:
        for entry in self._action_data:
            canonical = entry["canonical"]
            synonyms = " ".join(entry["synonyms"])
            phrases_text = " ".join(entry.get("phrases", [])) or canonical
            glyph = self._verb_encoder.encode(Concept(
                name=f"verb_{canonical}",
                attributes={"canonical": canonical, "synonyms": synonyms, "phrases": phrases_text},
            ))
            self._verb_glyphs.append((glyph, canonical))

        for entry in self._target_data:
            canonical = entry["canonical"]
            synonyms = " ".join(entry["synonyms"])
            phrases_text = " ".join(entry.get("phrases", [])) or canonical
            glyph = self._noun_encoder.encode(Concept(
                name=f"noun_{canonical}",
                attributes={"canonical": canonical, "synonyms": synonyms, "phrases": phrases_text},
            ))
            self._noun_glyphs.append((glyph, canonical))

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def extract(self, query: str) -> dict:
        """Extract action, target, domain, and keywords from a natural language query.

        Returns:
            dict with keys: action, target, domain, keywords
        """
        action = self.extract_action(query)
        target = self.extract_target(query, action=action)
        domain = self.infer_domain(query)
        keywords = self.extract_keywords(query)
        return {"action": action, "target": target, "domain": domain, "keywords": keywords}

    def extract_action(self, query: str) -> str:
        """Extract the primary action verb.

        Pipeline: phrase match → word map + impact rank → HDC fallback.
        """
        text_lower = query.lower().strip()
        words = self._tokenize(query)  # pass original for camelCase detection

        # Phase 1: phrase matching (first match wins)
        for phrase, action in self._phrase_actions:
            if phrase in text_lower:
                return action

        # Phase 2: word map with impact ranking and weak-word filtering
        found = self._match_action_words(words)
        if found:
            return found

        # Phase 3: HDC fallback for unknown verbs
        return self._hdc_verb_fallback(words)

    def extract_target(self, query: str, action: str = "") -> str:
        """Extract the primary target noun.

        Pipeline: phrase match → special patterns → word map → HDC fallback.
        """
        text_lower = query.lower().strip()
        words = self._tokenize(query)  # pass original for camelCase detection

        # Phase 1: phrase matching
        for phrase, target in self._phrase_targets:
            if phrase in text_lower:
                return target

        # Phase 2: special patterns (channels, DMs, emails)
        special = self._detect_special_target(text_lower)
        if special:
            return special

        # Phase 3: word map
        for w in words:
            clean = re.sub(r"[^a-z]", "", w)
            if clean in self._target_map:
                return self._target_map[clean]

        # Phase 4: HDC fallback
        return self._hdc_noun_fallback(words)

    def infer_domain(self, query: str) -> str:
        """Infer domain from weighted keyword signals."""
        text_lower = query.lower()
        text_clean = re.sub(r"\S+@\S+\.\S+", "", text_lower)

        boundary_signals = {"cc", "bcc", "dm", "#", "mail", "bug"}
        scores: dict[str, int] = {}

        for domain, signals in self._domain_signals.items():
            score = 0
            for signal, weight in signals.items():
                s_lower = signal.lower()
                if s_lower == "#":
                    if re.search(r"#\w+", text_clean):
                        score += weight
                elif s_lower in boundary_signals or len(s_lower) <= 3:
                    if re.search(r"\b" + re.escape(s_lower) + r"\b", text_clean):
                        score += weight
                else:
                    if s_lower in text_clean:
                        score += weight
            if score > 0:
                scores[domain] = score

        return max(scores, key=scores.get) if scores else "general"

    def extract_keywords(self, query: str) -> str:
        """Extract keywords with stop words removed."""
        cleaned = re.sub(r"[^\w\s#@._\-]", "", query.lower())
        return " ".join(w for w in cleaned.split() if w not in self._stop_words)

    def learn(self, word: str, canonical: str, kind: str = "verb") -> None:
        """Add a synonym mapping at runtime and rebuild the affected HDC cluster.

        Args:
            word: New synonym to learn (e.g., "dispatch")
            canonical: Canonical form it maps to (e.g., "send")
            kind: "verb" or "noun"
        """
        if kind == "verb":
            self._action_map[word] = canonical
            for entry in self._action_data:
                if entry["canonical"] == canonical:
                    if word not in entry["synonyms"]:
                        entry["synonyms"].append(word)
                    break
            else:
                self._action_data.append({"canonical": canonical, "synonyms": [canonical, word], "phrases": []})
            self._rebuild_verb_cluster(canonical)
        elif kind == "noun":
            self._target_map[word] = canonical
            for entry in self._target_data:
                if entry["canonical"] == canonical:
                    if word not in entry["synonyms"]:
                        entry["synonyms"].append(word)
                    break
            else:
                self._target_data.append({"canonical": canonical, "synonyms": [canonical, word], "phrases": []})
            self._rebuild_noun_cluster(canonical)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    # Impact ranking: highest-impact action wins in multi-action queries
    _IMPACT_RANK: dict[str, int] = {
        # Critical / irreversible ops
        "deploy": 11, "rollback": 11, "ban": 11,
        # Destructive / financial ops
        "delete": 10, "charge": 10, "refund": 10, "cancel": 10, "revoke": 10,
        # High-impact state changes
        "create": 9, "subscribe": 9, "approve": 9, "reject": 9, "escalate": 9,
        # Medium-impact ops
        "update": 8, "assign": 8, "share": 8, "upload": 8,
        "enable": 8, "disable": 8, "grant": 8, "kick": 8, "publish": 8,
        # Communication / tracking
        "identify": 7, "track": 7, "comment": 7, "notify": 7, "invite": 7, "build": 7,
        # Standard I/O
        "reply": 6, "set": 6, "sync": 6, "forward": 6, "review": 6,
        # Sending / data ops
        "send": 5, "draft": 5, "export": 5, "import": 5, "restore": 5, "archive": 5,
        # Discovery / analysis
        "search": 3, "find": 3, "analyze": 3, "detect": 3,
        # Retrieval / listing
        "get": 2, "list": 2,
    }

    # Words that are weak action signals (only fire if no strong alternative)
    _WEAK_ACTION_WORDS: set[str] = {
        "new", "open", "file", "start", "record", "book", "move",
        "note", "stop", "end", "drop", "present", "display",
        "capture", "register", "collect", "prepare",
        "mail", "sent", "check", "look",
    }

    @staticmethod
    def _split_camel_snake(name: str) -> str:
        """Split camelCase, snake_case, kebab-case, dot.notation into words.

        Examples:
            getUserById   → "get user by id"
            create_ticket → "create ticket"
            search-emails → "search emails"
            stripe.charge → "stripe charge"
        """
        name = re.sub(r"[_\-.]", " ", name)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
        return name.lower().strip()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text, expanding camelCase/snake_case identifiers.

        Splits BEFORE lowercasing so camelCase boundaries (uppercase letters)
        are detectable. Each resulting token is lowercased after splitting.
        """
        cleaned = re.sub(r"[^\w\s#@._\-]", "", text)
        tokens = []
        for token in cleaned.split():
            if any(c in token for c in ("_", "-", ".")) or any(c.isupper() for c in token):
                tokens.extend(self._split_camel_snake(token).split())
            else:
                tokens.append(token.lower())
        return tokens

    def _match_action_words(self, words: list[str]) -> str | None:
        """Match action words with impact ranking and weak-word filtering."""
        found_actions: list[str] = []
        strong_actions: list[str] = []

        for w in words:
            clean = re.sub(r"[^a-z]", "", w)
            if clean in self._action_map:
                action = self._action_map[clean]
                found_actions.append(action)
                if clean not in self._WEAK_ACTION_WORDS:
                    strong_actions.append(action)

        candidates = strong_actions if strong_actions else found_actions
        if candidates:
            unique = list(dict.fromkeys(candidates))
            if len(unique) > 1:
                return max(unique, key=lambda a: self._IMPACT_RANK.get(a, 0))
            return unique[0]
        return None

    def _detect_special_target(self, text_lower: str) -> str | None:
        """Detect special target patterns (channels, DMs, emails)."""
        if re.search(r"#\w+", text_lower):
            return "channel"
        if "dm " in text_lower or text_lower.startswith("dm ") or "direct message" in text_lower:
            return "dm"
        if re.search(r"\b\S+@\S+\.\S+", text_lower) and not re.search(r"#\w+", text_lower):
            if "message" in text_lower or "send a message" in text_lower:
                return "dm"
        return None

    def _hdc_verb_fallback(self, words: list[str], threshold: float = 0.35) -> str:
        """HDC similarity fallback for unknown verbs."""
        query_text = " ".join(w for w in words if w not in self._stop_words)
        if not query_text:
            return "get"

        query_glyph = self._verb_encoder.encode(Concept(
            name="verb_query",
            attributes={"canonical": "unknown", "synonyms": query_text, "phrases": query_text},
        ))

        best_score = 0.0
        best_canonical = "get"
        for glyph, canonical in self._verb_glyphs:
            score = self._score_roles(query_glyph, glyph, _VERB_ROLE_WEIGHTS)
            if score > best_score:
                best_score = score
                best_canonical = canonical

        return best_canonical if best_score >= threshold else "get"

    def _hdc_noun_fallback(self, words: list[str], threshold: float = 0.35) -> str:
        """HDC similarity fallback for unknown nouns."""
        query_text = " ".join(w for w in words if w not in self._stop_words)
        if not query_text:
            return "general"

        query_glyph = self._noun_encoder.encode(Concept(
            name="noun_query",
            attributes={"canonical": "unknown", "synonyms": query_text, "phrases": query_text},
        ))

        best_score = 0.0
        best_canonical = "general"
        for glyph, canonical in self._noun_glyphs:
            score = self._score_roles(query_glyph, glyph, _NOUN_ROLE_WEIGHTS)
            if score > best_score:
                best_score = score
                best_canonical = canonical

        return best_canonical if best_score >= threshold else "general"

    @staticmethod
    def _score_roles(query_glyph, candidate_glyph, role_weights: dict) -> float:
        """Weighted average of per-role cosine similarities."""
        q_roles: dict = {}
        c_roles: dict = {}
        for layer in query_glyph.layers.values():
            if not layer.name.startswith("_"):
                for seg in layer.segments.values():
                    q_roles.update(seg.roles)
        for layer in candidate_glyph.layers.values():
            if not layer.name.startswith("_"):
                for seg in layer.segments.values():
                    c_roles.update(seg.roles)

        weighted_sum = 0.0
        weight_total = 0.0
        for role_name, weight in role_weights.items():
            if role_name in q_roles and role_name in c_roles:
                sim = float(cosine_similarity(q_roles[role_name].data, c_roles[role_name].data))
                weighted_sum += sim * weight
                weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _rebuild_verb_cluster(self, canonical: str) -> None:
        for entry in self._action_data:
            if entry["canonical"] == canonical:
                synonyms = " ".join(entry["synonyms"])
                phrases_text = " ".join(entry.get("phrases", [])) or canonical
                new_glyph = self._verb_encoder.encode(Concept(
                    name=f"verb_{canonical}",
                    attributes={"canonical": canonical, "synonyms": synonyms, "phrases": phrases_text},
                ))
                self._verb_glyphs = [
                    (new_glyph, c) if c == canonical else (g, c)
                    for g, c in self._verb_glyphs
                ]
                if not any(c == canonical for _, c in self._verb_glyphs):
                    self._verb_glyphs.append((new_glyph, canonical))
                break

    def _rebuild_noun_cluster(self, canonical: str) -> None:
        for entry in self._target_data:
            if entry["canonical"] == canonical:
                synonyms = " ".join(entry["synonyms"])
                phrases_text = " ".join(entry.get("phrases", [])) or canonical
                new_glyph = self._noun_encoder.encode(Concept(
                    name=f"noun_{canonical}",
                    attributes={"canonical": canonical, "synonyms": synonyms, "phrases": phrases_text},
                ))
                self._noun_glyphs = [
                    (new_glyph, c) if c == canonical else (g, c)
                    for g, c in self._noun_glyphs
                ]
                if not any(c == canonical for _, c in self._noun_glyphs):
                    self._noun_glyphs.append((new_glyph, canonical))
                break


# ---------------------------------------------------------------------------
# Module-level convenience functions (singleton pattern)
# ---------------------------------------------------------------------------

_default_extractor: IntentExtractor | None = None


def get_extractor(packs: list[str] | None = None) -> IntentExtractor:
    """Get an IntentExtractor instance.

    Returns a cached singleton when no packs are specified.
    Creates a new instance when packs are requested (not cached).

    Args:
        packs: Optional domain pack names (e.g., ["filesystem", "trading"]).

    Example:
        from glyphh.intent import get_extractor

        extractor = get_extractor()                          # base vocabulary
        fs_extractor = get_extractor(packs=["filesystem"])  # + filesystem pack
    """
    global _default_extractor
    if packs:
        return IntentExtractor(packs=packs)
    if _default_extractor is None:
        _default_extractor = IntentExtractor()
    return _default_extractor


def extract_intent(query: str) -> dict:
    """Extract {action, target, domain, keywords} using the default extractor."""
    return get_extractor().extract(query)


def extract_action(query: str) -> str:
    """Extract the primary action verb using the default extractor."""
    return get_extractor().extract_action(query)


def extract_target(query: str, action: str = "") -> str:
    """Extract the primary target noun using the default extractor."""
    return get_extractor().extract_target(query, action=action)


def infer_domain(query: str) -> str:
    """Infer domain from the query using the default extractor."""
    return get_extractor().infer_domain(query)
