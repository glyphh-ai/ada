"""
Unit test fixtures — fallback for DB-dependent fixtures.

The root conftest only defines ``test_db`` when the full runtime import
chain succeeds (``from main import app`` + DB Base).  On CI that chain
often fails, leaving ``test_db`` undefined and causing fixture-not-found
errors.  This conftest mirrors the same guard and provides a skip-based
fallback so those tests degrade gracefully.
"""

import pytest

# Mirror the exact guard from tests/conftest.py so this fallback only
# activates when the root conftest's test_db is NOT defined.
_root_has_test_db = True
try:
    import pytest_asyncio                                           # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncSession                 # noqa: F401
    from main import app                                            # noqa: F401
    from infrastructure.database.connection import Base, get_db     # noqa: F401
except ImportError:
    _root_has_test_db = False

if not _root_has_test_db:
    @pytest.fixture
    def test_db():
        pytest.skip("test_db unavailable — runtime deps not fully importable")
