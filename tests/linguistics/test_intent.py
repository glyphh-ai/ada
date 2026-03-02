"""
Tests for LinguisticIntentParser (end-to-end).

Key properties:
  - extract_intent() returns the seeded canonical labels
  - Handles morphological variants: "files" → target="file"
  - Handles misspellings via character similarity
  - normalize() → lemma
  - pos_tag() → (word, pos) list
  - Domain detection from keyword signals
  - Unseeded parser returns "none" for action/target
"""

import pytest

from glyphh.linguistics import LinguisticIntentParser


@pytest.fixture(scope="module")
def parser():
    p = LinguisticIntentParser(dimension=10000, seed=42)
    p.seed_actions({
        "get":    ["get", "fetch", "retrieve", "find", "show", "check", "list"],
        "create": ["create", "make", "build", "generate", "add", "new"],
        "send":   ["send", "email", "mail", "post", "dispatch"],
        "delete": ["delete", "remove", "drop", "erase"],
        "update": ["update", "modify", "edit", "change", "set"],
    })
    p.seed_targets({
        "file":    ["file", "document", "folder", "directory", "attachment"],
        "user":    ["user", "person", "contact", "account", "member"],
        "message": ["message", "email", "notification", "chat", "post"],
        "task":    ["task", "issue", "ticket", "item", "todo"],
    })
    p.seed_domains({
        "storage":  ["gdrive", "dropbox", "s3", "bucket", "drive"],
        "payments": ["stripe", "invoice", "billing", "refund", "charge"],
        "tickets":  ["jira", "github", "issue", "ticket", "sprint"],
    })
    return p


# ── Basic action/target extraction ─────────────────────────────────────────

def test_action_send(parser):
    result = parser.extract_intent("send the file to John")
    assert result["action"] == "send", f"Expected 'send', got {result['action']}"


def test_action_get(parser):
    result = parser.extract_intent("fetch the user profile")
    assert result["action"] == "get", f"Expected 'get', got {result['action']}"


def test_action_create(parser):
    result = parser.extract_intent("create a new task")
    assert result["action"] == "create", f"Expected 'create', got {result['action']}"


def test_action_delete(parser):
    result = parser.extract_intent("remove the old document")
    assert result["action"] == "delete", f"Expected 'delete', got {result['action']}"


def test_target_file(parser):
    # "file" is a direct seed word for "file" target
    result = parser.extract_intent("fetch the file")
    assert result["target"] == "file", f"Expected 'file', got {result['target']}"


def test_target_file_document(parser):
    # "document" is a seed example for "file" target — tests prototype generalisation
    result = parser.extract_intent("fetch the document")
    assert result["target"] == "file", f"Expected 'file' for 'document', got {result['target']}"


def test_target_user(parser):
    result = parser.extract_intent("find the user account")
    assert result["target"] == "user", f"Expected 'user', got {result['target']}"


def test_target_message(parser):
    # Use direct seed word "message" — avoids ambiguity with "notification"
    result = parser.extract_intent("send a message to Alice")
    assert result["target"] == "message", f"Expected 'message', got {result['target']}"


# ── Morphological variant handling ─────────────────────────────────────────

def test_plural_target_files(parser):
    """Key test: 'files' should match target 'file' via MorphologyEngine."""
    result = parser.extract_intent("get all the files")
    assert result["target"] == "file", (
        f"Expected 'file' for 'files' query, got '{result['target']}'"
    )


def test_plural_target_documents(parser):
    result = parser.extract_intent("list all documents")
    assert result["target"] == "file"


def test_gerund_action(parser):
    result = parser.extract_intent("sending the message to Alice")
    assert result["action"] == "send"


def test_past_tense_action(parser):
    result = parser.extract_intent("removed the user from the group")
    assert result["action"] == "delete"


# ── Domain detection ────────────────────────────────────────────────────────

def test_domain_payments(parser):
    result = parser.extract_intent("process the stripe payment invoice")
    assert result["domain"] == "payments"


def test_domain_storage(parser):
    result = parser.extract_intent("upload the file to gdrive")
    assert result["domain"] == "storage"


def test_domain_empty_for_generic(parser):
    result = parser.extract_intent("find the user account")
    assert result["domain"] == ""


# ── Keywords ────────────────────────────────────────────────────────────────

def test_keywords_not_empty(parser):
    result = parser.extract_intent("send the report to Alice")
    assert result["keywords"] != ""


def test_keywords_excludes_stopwords(parser):
    result = parser.extract_intent("send the file")
    # "the" is DET — should not appear in keywords
    assert "the" not in result["keywords"].split()


# ── normalize() helper ──────────────────────────────────────────────────────

def test_normalize_plural(parser):
    lemma = parser.normalize("files")
    assert lemma == "file"


def test_normalize_gerund(parser):
    lemma = parser.normalize("running")
    assert lemma == "run"


def test_normalize_base(parser):
    lemma = parser.normalize("dog")
    assert lemma == "dog"


# ── pos_tag() helper ────────────────────────────────────────────────────────

def test_pos_tag_returns_pairs(parser):
    tags = parser.pos_tag("send the file")
    assert len(tags) == 3
    assert all(len(t) == 2 for t in tags)


def test_pos_tag_det(parser):
    tags = parser.pos_tag("send the file")
    word_to_pos = dict(tags)
    assert word_to_pos.get("the") == "DET"


# ── Edge cases ──────────────────────────────────────────────────────────────

def test_empty_query(parser):
    result = parser.extract_intent("")
    assert result["action"] == "none"
    assert result["target"] == "none"
    assert result["domain"] == ""
    assert result["keywords"] == ""


def test_result_has_all_keys(parser):
    result = parser.extract_intent("fetch the file")
    assert "action"   in result
    assert "target"   in result
    assert "domain"   in result
    assert "keywords" in result


def test_unseeded_parser_returns_none():
    p = LinguisticIntentParser()
    result = p.extract_intent("find the file")
    assert result["action"] == "none"
    assert result["target"] == "none"
