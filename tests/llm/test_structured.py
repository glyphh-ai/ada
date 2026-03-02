"""Tests for structured output parsing."""

from glyphh.llm.structured import (
    LLMResult,
    parse_tool_call_response,
    INTENT_TOOLS,
    SLOT_TOOLS,
    ARBITRATION_TOOLS,
)


class TestLLMResult:
    """LLMResult dataclass."""

    def test_defaults(self):
        r = LLMResult()
        assert r.data == {}
        assert r.raw_text == ""
        assert r.tokens_used == 0
        assert r.latency_ms == 0.0

    def test_with_data(self):
        r = LLMResult(
            data={"action": "delete"},
            raw_text='{"action": "delete"}',
            tokens_used=15,
            latency_ms=42.5,
        )
        assert r.data["action"] == "delete"
        assert r.tokens_used == 15


class TestParseToolCallResponse:
    """parse_tool_call_response() — multiple format handling."""

    def test_direct_json(self):
        result = parse_tool_call_response('{"action": "delete", "confidence": 0.9}')
        assert result["action"] == "delete"
        assert result["confidence"] == 0.9

    def test_markdown_fenced_json(self):
        text = '```json\n{"action": "create"}\n```'
        result = parse_tool_call_response(text)
        assert result["action"] == "create"

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"target": "file"}\n```'
        result = parse_tool_call_response(text)
        assert result["target"] == "file"

    def test_embedded_json_in_text(self):
        text = 'Here is the result: {"action": "search"} done.'
        result = parse_tool_call_response(text)
        assert result["action"] == "search"

    def test_empty_string(self):
        assert parse_tool_call_response("") == {}

    def test_no_json_at_all(self):
        assert parse_tool_call_response("no json here") == {}

    def test_nested_json(self):
        text = '{"arguments": {"file_name": "report.txt"}}'
        result = parse_tool_call_response(text)
        assert result["arguments"]["file_name"] == "report.txt"

    def test_whitespace_handling(self):
        text = '  \n  {"action": "get"}  \n  '
        result = parse_tool_call_response(text)
        assert result["action"] == "get"


class TestToolSchemas:
    """Verify tool schemas have correct structure."""

    def test_intent_tools_structure(self):
        assert len(INTENT_TOOLS) == 1
        func = INTENT_TOOLS[0]["function"]
        assert func["name"] == "classify_intent"
        required = func["parameters"]["required"]
        assert "action" in required
        assert "confidence" in required

    def test_slot_tools_structure(self):
        assert len(SLOT_TOOLS) == 1
        func = SLOT_TOOLS[0]["function"]
        assert func["name"] == "fill_slots"
        assert "arguments" in func["parameters"]["properties"]

    def test_arbitration_tools_structure(self):
        assert len(ARBITRATION_TOOLS) == 1
        func = ARBITRATION_TOOLS[0]["function"]
        assert func["name"] == "arbitrate"
        required = func["parameters"]["required"]
        assert "accept" in required
        assert "reason" in required
