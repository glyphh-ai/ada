"""
Slot/type answerability — does a candidate fact actually answer the question?

The recall score measures *similarity*, not *answer-ness*. A fact can be
topically close ("The user's name is Chris") yet not answer the question
("what is Chris's favorite color?"). Left alone, a high-similarity non-answer
crosses the confidence gate and Ada answers when she should ask to clarify.

This module derives the *expected answer type* from the question (the wh-focus)
and checks whether a candidate fact contains a value of that type. When the
question asks for a closed-class type (a color, a number, a date) and the fact
has none, we down-weight that candidate's confidence so the gate routes it to
ASK/clarify instead of DONE.

Deterministic and LLM-free by design — this is what makes "the brain decides,
not the model" a real property. High precision on closed classes; permissive
(no penalty) on open classes (person/place/org) where absence isn't decisive.
"""

from __future__ import annotations

import re

# ── Expected-type detection (question → type) ───────────────────────────────
# Ordered: most specific first. Closed-class types are the high-precision ones.

COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "violet", "indigo",
    "pink", "brown", "black", "white", "gray", "grey", "cyan", "magenta",
    "teal", "maroon", "navy", "beige", "turquoise", "gold", "silver",
}

OCCUPATIONS = {
    "nurse", "doctor", "physician", "engineer", "teacher", "lawyer", "attorney",
    "accountant", "developer", "programmer", "designer", "manager", "analyst",
    "scientist", "researcher", "chef", "cook", "pilot", "driver", "electrician",
    "plumber", "carpenter", "architect", "professor", "consultant", "technician",
    "surgeon", "dentist", "pharmacist", "therapist", "paramedic", "officer",
}

_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")

# Closed classes (penalize hard when absent) and open classes (don't penalize).
CLOSED = {"color", "number", "date"}


def expected_type(question: str) -> str | None:
    """Return the expected answer type for a question, or None if undetermined.

    None means 'no strong type' → never penalize (let similarity stand).
    """
    q = question.lower().strip()

    # ── Closed classes (high precision) ──
    if "colour" in q or "color" in q:
        return "color"
    if re.search(r"\b(how many|how much|how tall|how old|how far|how long|"
                 r"how big|how heavy|what age|temperature|price|cost|"
                 r"weight|height|distance|count|quantity)\b", q):
        return "number"
    if re.search(r"\b(when|what year|what date|what time|which year|"
                 r"how old)\b", q):
        return "date"

    # ── Open classes (permissive; presence of a proper noun is enough) ──
    if re.search(r"\b(which company|what company|employer|employs|"
                 r"works for|works at)\b", q):
        return "org"
    if re.search(r"\b(what (do|does).*(do for a living|for work)|occupation|"
                 r"profession|job|what.*do for a living)\b", q):
        return "occupation"
    if re.search(r"\b(where|what city|what country|what place|location|"
                 r"which city|which country)\b", q):
        return "place"
    if re.search(r"\b(who|whose|name of|what is .*name|what.s .*name)\b", q):
        return "person"

    return None


# ── Satisfaction detection (does the fact contain that type?) ───────────────

_PROPER_NOUN = re.compile(r"\b[A-Z][a-zA-Z]+\b")
_STOPCAPS = {"The", "A", "An", "It", "He", "She", "They", "I", "We", "You",
             "This", "That", "When", "Where", "Who", "What", "Why", "How",
             "My", "Your", "His", "Her", "Their", "Our", "Ada"}


def _has_proper_noun(fact: str) -> bool:
    """Any capitalized token that isn't a sentence-initial stopword."""
    for m in _PROPER_NOUN.finditer(fact):
        tok = m.group(0)
        # Skip if it's a leading stopword-style cap; keep real proper nouns.
        if tok in _STOPCAPS and m.start() == 0:
            continue
        if tok not in _STOPCAPS:
            return True
    return False


def satisfies(fact: str, atype: str) -> bool:
    """Does `fact` contain a value of the expected answer type `atype`?"""
    f = fact.lower()

    if atype == "color":
        toks = set(re.findall(r"[a-z]+", f))
        return bool(toks & COLORS)

    if atype == "number":
        return bool(re.search(r"\d", fact))

    if atype == "date":
        if re.search(r"\b\d{4}\b", fact):           # year
            return True
        if re.search(r"\b\d{1,2}[:/]\d{2}\b", fact):  # time / date
            return True
        return any(m in f for m in _MONTHS) or any(d in f for d in _WEEKDAYS)

    if atype == "occupation":
        toks = set(re.findall(r"[a-z]+", f))
        return bool(toks & OCCUPATIONS)

    # Open classes: presence of a proper noun is sufficient evidence.
    if atype in ("person", "place", "org"):
        return _has_proper_noun(fact)

    return True  # unknown type → don't penalize


def answerability_factor(question: str, fact: str, penalty: float = 0.5) -> float:
    """Confidence multiplier for `fact` given `question`.

    1.0  → no strong type detected, or the fact satisfies the type.
    penalty → a (closed-class) type was expected but the fact lacks it.

    Open-class misses are penalized only lightly (halfway to `penalty`) since
    proper-noun absence is weaker evidence than a missing color/number/date.
    """
    atype = expected_type(question)
    if atype is None or satisfies(fact, atype):
        return 1.0
    if atype in CLOSED:
        return penalty
    # Open class miss — softer penalty.
    return penalty + (1.0 - penalty) * 0.5
