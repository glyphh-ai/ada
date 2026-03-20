"""
Unit test fixtures — fallbacks for DB-dependent fixtures.

The root conftest only defines `test_db` when runtime deps (asyncpg, etc.)
are importable. On CI these are unavailable, so tests requesting `test_db`
get a fixture-level skip instead of a collection error.
"""

import pytest

_has_runtime_db = True
try:
    from infrastructure.database.connection import Base  # noqa: F401
except Exception:
    _has_runtime_db = False


if not _has_runtime_db:

    @pytest.fixture
    def test_db():
        pytest.skip("Requires database deps (asyncpg/aiosqlite)")
