"""Unit tests for the enforcement policy store (deterministic guards)."""

import pytest

from domains.brain.policy import PolicyStore, Rule


@pytest.fixture
def store(tmp_path):
    return PolicyStore(path=str(tmp_path / "policy.json"))


class TestPolicyStore:
    def test_add_and_list(self, store):
        store.add(r"git push.*\bmain\b", "No direct pushes to main", tool="Bash")
        assert len(store.rules()) == 1
        assert store.rules()[0].tool == "Bash"

    def test_deny_on_match(self, store):
        store.add(r"git push.*\bmain\b", "No direct pushes to main", tool="Bash")
        hit = store.evaluate("Bash", {"command": "git push origin main"})
        assert hit is not None
        assert hit["decision"] == "deny"
        assert "main" in hit["reason"].lower()

    def test_allow_when_no_match(self, store):
        store.add(r"git push.*\bmain\b", "No direct pushes to main", tool="Bash")
        assert store.evaluate("Bash", {"command": "git push origin feature"}) is None

    def test_tool_filter(self, store):
        store.add(r"main", "x", tool="Bash")
        # Same pattern text, different tool → no match
        assert store.evaluate("Edit", {"file_path": "main.py"}) is None

    def test_wildcard_tool(self, store):
        store.add(r"\.env", "Never touch .env", tool="*")
        assert store.evaluate("Edit", {"file_path": "/proj/.env"})["decision"] == "deny"
        assert store.evaluate("Read", {"file_path": "/proj/.env"})["decision"] == "deny"

    def test_persistence_roundtrip(self, tmp_path):
        p = str(tmp_path / "policy.json")
        PolicyStore(path=p).add(r"rm -rf", "no", tool="Bash")
        reloaded = PolicyStore(path=p)
        assert any("rm -rf" in r.pattern for r in reloaded.rules())

    def test_bad_regex_falls_back_to_substring(self, store):
        store.add(r"a[b", "malformed regex", tool="*")  # invalid regex
        assert store.evaluate("Bash", {"command": "echo a[b"})["decision"] == "deny"


class TestBrainEnforcement:
    def _brain(self):
        from unittest.mock import AsyncMock, MagicMock
        from domains.brain.think import Brain
        from domains.brain.llm import LLMUsage
        from domains.brain.policy import PolicyStore
        state = MagicMock(); state.capabilities = {}; state.capability_names = []
        mm = MagicMock(); mm._models = {}; mm._encoding_in_progress = []
        llm = MagicMock(); llm.available = False
        llm.set_firewall = MagicMock(); llm.ask = AsyncMock(return_value=None)
        llm.usage = LLMUsage()
        brain = Brain(brain_state=state, model_manager=mm, llm=llm,
                      session_factory=MagicMock())
        # isolate policy file per test
        import tempfile, os
        brain._policy = PolicyStore(path=os.path.join(tempfile.mkdtemp(), "p.json"))
        return brain

    def test_add_guard_then_block(self):
        brain = self._brain()
        brain.add_guard(r"git push.*\bmain\b", "No direct pushes to main", tool="Bash")
        denied = brain.check_action("Bash", {"command": "git push origin main"})
        assert denied["decision"] == "deny"
        allowed = brain.check_action("Bash", {"command": "git status"})
        assert allowed["decision"] == "allow"

    def test_add_guard_also_remembers(self):
        brain = self._brain()
        brain.add_guard(r"\.env", "Never read secrets", tool="Read")
        # defense in depth: the guard is also recallable as a reminder
        facts = brain.recall("can I read the .env file?", top_k=5)
        assert any("guard" in f["content"].lower() for f in facts)
