"""
Plan-based resource limits for Glyphh Runtime.

Glyph limits per model are determined by the user's plan (from JWT).
The plan slug is included in the JWT `plan` claim, set by the platform
when the user logs in or refreshes their token.
"""

# Plan slug → max glyphs per model (-1 = unlimited)
PLAN_GLYPH_LIMITS: dict[str, int] = {
    "free": 100,
    "advanced": 500,
    "pro": -1,
}


def get_max_glyphs_per_model(plan_slug: str) -> int:
    """Return the glyph-per-model limit for a plan. Defaults to free tier."""
    return PLAN_GLYPH_LIMITS.get(plan_slug, PLAN_GLYPH_LIMITS["free"])
