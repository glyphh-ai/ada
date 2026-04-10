"""
Capability Tester — Ada validates new capabilities before they go live.

Spins up an isolated subprocess with its own DB, loads the capability,
runs test queries, and checks the results. Only capabilities that pass
validation get promoted to the live runtime.

The subprocess is fully isolated:
  - Separate SQLite DB (temp file)
  - Separate Python process (no shared state)
  - Timeout per query
  - Killed on completion

Used by:
  - CapabilityBuilder after building (pre-promotion test)
  - Dream loop after minting (background validation)
  - Manual: "test capability X" in the REPL
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum pass rate to promote a capability
MIN_PASS_RATE = 0.6

# Timeout per test subprocess
SUBPROCESS_TIMEOUT = 30


@dataclass
class TestCase:
    """A single test query and expected outcome."""
    query: str
    expected_capability: str  # Which capability should handle it
    passed: bool = False
    similarity: float = 0.0
    result_text: str = ""
    error: Optional[str] = None


@dataclass
class TestReport:
    """Results from testing a capability."""
    capability: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    pass_rate: float = 0.0
    avg_similarity: float = 0.0
    promoted: bool = False
    cases: list[TestCase] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def summary(self) -> str:
        status = "PASS" if self.pass_rate >= MIN_PASS_RATE else "FAIL"
        return (
            f"[{status}] {self.capability}: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.0%}), avg similarity {self.avg_similarity:.2f}, "
            f"{self.elapsed_s:.1f}s"
        )


class CapabilityTester:
    """Tests capabilities in isolated subprocesses.

    Each test spins up a fresh Python process with:
    - A temp SQLite database
    - The capability loaded and exemplars encoded
    - Test queries run through encode_query → similarity search

    The subprocess reports results as JSON on stdout.
    """

    def __init__(self, llm=None):
        self._llm = llm

    async def test(
        self,
        capability_dir: Path,
        capability_name: str,
        test_queries: list[str] | None = None,
        num_generated: int = 10,
    ) -> TestReport:
        """Test a capability end-to-end.

        Args:
            capability_dir: Path to the capability directory
            capability_name: Name of the capability
            test_queries: Explicit test queries (optional)
            num_generated: Number of LLM-generated test queries
        """
        start = time.time()

        # Gather test queries
        queries = list(test_queries or [])

        # Generate additional test queries via LLM
        if self._llm and self._llm.available and num_generated > 0:
            manifest_path = capability_dir / "manifest.yaml"
            if manifest_path.exists():
                import yaml
                manifest = yaml.safe_load(manifest_path.read_text()) or {}
                desc = manifest.get("description", capability_name)
            else:
                desc = capability_name

            generated = await self._llm.generate_exemplars(
                f"Test queries for a '{capability_name}' capability: {desc}",
                count=num_generated,
            )
            queries.extend(generated)

        # Also pull some queries from the exemplars themselves
        exemplar_path = capability_dir / "data" / "exemplars.jsonl"
        if exemplar_path.exists():
            with open(exemplar_path) as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    try:
                        entry = json.loads(line)
                        queries.append(entry.get("text", ""))
                    except json.JSONDecodeError:
                        pass

        queries = [q for q in queries if q.strip()]
        if not queries:
            return TestReport(capability=capability_name, elapsed_s=time.time() - start)

        # Run each query in a subprocess
        cases: list[TestCase] = []
        for query in queries:
            case = await self._run_test_query(
                capability_dir, capability_name, query
            )
            cases.append(case)

        # Compute stats
        report = TestReport(
            capability=capability_name,
            total=len(cases),
            passed=sum(1 for c in cases if c.passed),
            failed=sum(1 for c in cases if not c.passed and not c.error),
            errors=sum(1 for c in cases if c.error),
            cases=cases,
            elapsed_s=time.time() - start,
        )
        report.pass_rate = report.passed / report.total if report.total > 0 else 0.0
        sims = [c.similarity for c in cases if c.similarity > 0]
        report.avg_similarity = sum(sims) / len(sims) if sims else 0.0
        report.promoted = report.pass_rate >= MIN_PASS_RATE

        logger.info(report.summary)
        return report

    async def _run_test_query(
        self,
        capability_dir: Path,
        capability_name: str,
        query: str,
    ) -> TestCase:
        """Run a single test query in an isolated subprocess."""

        # The subprocess script
        test_script = _TEST_SUBPROCESS_SCRIPT

        case = TestCase(query=query, expected_capability=capability_name)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", test_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "ADA_TEST_CAP_DIR": str(capability_dir),
                    "ADA_TEST_CAP_NAME": capability_name,
                    "ADA_TEST_QUERY": query,
                },
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=SUBPROCESS_TIMEOUT
            )

            if proc.returncode != 0:
                case.error = stderr.decode()[:200] if stderr else f"exit code {proc.returncode}"
                return case

            # Parse JSON result from subprocess
            try:
                result = json.loads(stdout.decode().strip())
                case.similarity = result.get("similarity", 0.0)
                case.result_text = result.get("matched", "")
                case.passed = result.get("matched_count", 0) > 0 and case.similarity >= 0.25
            except json.JSONDecodeError:
                case.error = "Invalid JSON from subprocess"

        except asyncio.TimeoutError:
            case.error = f"Timeout ({SUBPROCESS_TIMEOUT}s)"
        except Exception as e:
            case.error = str(e)[:200]

        return case


# ---------------------------------------------------------------------------
# Subprocess script — runs in complete isolation
# ---------------------------------------------------------------------------

_TEST_SUBPROCESS_SCRIPT = '''
import json
import os
import sys
import tempfile

cap_dir = os.environ["ADA_TEST_CAP_DIR"]
cap_name = os.environ["ADA_TEST_CAP_NAME"]
query = os.environ["ADA_TEST_QUERY"]

# Use isolated temp DB
db_path = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"

try:
    import asyncio
    from pathlib import Path

    # Add project root to path
    project_root = Path(cap_dir).parents[1]
    sys.path.insert(0, str(project_root))

    from infrastructure.database import init_db, async_session_maker
    from domains.models.manager import ModelManager

    async def run():
        await init_db()
        mm = ModelManager(async_session_maker)

        # Load the capability
        await mm.load_model_from_directory(
            model_dir=Path(cap_dir),
            org_id="test",
            model_id=cap_name,
        )

        # Wait for exemplar encoding
        for _ in range(60):
            if ("test", cap_name) not in mm._encoding_in_progress:
                break
            await asyncio.sleep(0.5)

        # Query it
        from domains.query.service import QueryService
        qs = QueryService(mm, async_session_maker)
        result = await qs.similarity_search(
            org_id="test",
            model_id=cap_name,
            query=query,
        )

        # Extract results
        matched_count = 0
        top_similarity = 0.0
        top_matched = ""

        if result and isinstance(result, dict):
            ft = result.get("fact_tree", {})
            children = ft.get("children", [])
            matched_count = len(children)
            if children:
                top_similarity = children[0].get("value", 0.0)
                top_matched = children[0].get("description", "")

        return {
            "matched_count": matched_count,
            "similarity": top_similarity,
            "matched": top_matched,
        }

    result = asyncio.run(run())
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({"error": str(e), "matched_count": 0, "similarity": 0.0}))
finally:
    # Clean up temp DB
    try:
        os.unlink(db_path)
    except:
        pass
'''
