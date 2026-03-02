"""Prompt templates for each LLM integration point.

Each template is a named constant. The engine formats them with context
at call time via str.format() placeholders.

Two categories:
  1. Schema classification — LLM-primary intent (used by SchemaIntentClassifier)
  2. Legacy fallback — intent/slot/arbitration (used by old loop.py integration)
"""

# ── INTENT DISAMBIGUATION ──
# Fires during PERCEIVE when HDC returns the default "get" action,
# indicating the rule-based + HDC extraction couldn't identify the verb.

INTENT_SYSTEM = (
    "You classify user intent for a function-calling system. "
    "Given the query, identify the primary action (verb), target (noun), "
    "and domain. Use the classify_intent tool to return your answer. "
    "Be concise. Only use actions from the provided list when possible."
)

INTENT_USER = (
    "Available actions: {available_actions}\n"
    "Available domains: {available_domains}\n\n"
    "Query: {query}"
)


# ── SLOT EXTRACTION ──
# Fires during SLOT when rule-based extraction leaves required
# parameters unfilled. Only missing parameters are listed.

SLOT_SYSTEM = (
    "You extract function arguments from user queries. "
    "Given a function signature, current state, and the user's query, "
    "extract values for each listed parameter. "
    "Only include parameters the user explicitly or implicitly specifies. "
    "Use the fill_slots tool to return your answer."
)

SLOT_USER = (
    "Function: {func_name}\n"
    "Missing parameters:\n{param_block}\n\n"
    "Current state: {state_summary}\n\n"
    "Query: {query}"
)


# ── CONFIDENCE ARBITRATION ──
# Fires during CONFIDENCE GATE when the HDC pipeline resolves to
# function calls but confidence is in the uncertain band (0.15-0.50).

ARBITRATION_SYSTEM = (
    "You are a quality gate for a function-calling pipeline. "
    "The system has resolved the user's query to specific function calls. "
    "Review whether the resolution makes sense given the query. "
    "Use the arbitrate tool to accept or reject the result."
)

ARBITRATION_USER = (
    "Query: {query}\n\n"
    "Resolved result:\n"
    "  Action: {action}\n"
    "  Target: {target}\n"
    "  Functions: {functions}\n"
    "  Filled arguments: {filled_slots}\n"
    "  Missing arguments: {missing_slots}\n"
    "  Confidence: {confidence:.2f}\n\n"
    "Is this resolution correct?"
)


# ── SCHEMA CLASSIFICATION ──
# LLM-primary intent classification. The system prompt is built dynamically
# at configure() time from function schemas. The user prompt provides
# per-query context (state, recent actions, query).

SCHEMA_CLASSIFY_SYSTEM = (
    "You are an intent classifier for a function-calling system. "
    "Given the user's query, current state, and recent actions, determine "
    "which function(s) to call and extract their arguments.\n\n"
    "Available functions:\n{function_descriptions}\n\n"
    "Rules:\n"
    "- Select ONLY from the listed functions\n"
    "- Extract argument values from the query when possible\n"
    "- Set confidence based on how clearly the query maps to a function\n"
    "- If the query is ambiguous or doesn't match any function, "
    "set confidence below 0.3\n"
    "- If multiple functions are needed, list them in execution order\n"
    "- Use the classify_and_call tool to return your answer"
)

SCHEMA_CLASSIFY_USER = (
    "Current context: {state_summary}\n"
    "Recent actions: {recent_actions}\n\n"
    "Query: {query}"
)
