"""
Tests for glyphh.llm.engine — LLM inference engine.

These tests verify the engine structure, backend detection, prompt formatting,
and generation via mocked backends (no real model required).
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from glyphh.llm.engine import (
    LLMEngine,
    GenerateConfig,
    _MLXBackend,
    _LlamaCppBackend,
    _APIBackend,
    _detect_backend,
    _model_dir,
)


# ── GenerateConfig ──────────────────────────────────────────────────────────


class TestGenerateConfig:
    def test_defaults(self):
        c = GenerateConfig()
        assert c.max_tokens == 512
        assert c.temperature == 0.7
        assert c.top_p == 0.9
        assert c.repetition_penalty == 1.1
        assert "<|im_end|>" in c.stop

    def test_override(self):
        c = GenerateConfig(max_tokens=128, temperature=0.0)
        assert c.max_tokens == 128
        assert c.temperature == 0.0


# ── Prompt formatting ───────────────────────────────────────────────────────


class TestPromptFormatting:
    def test_user_only(self):
        engine = LLMEngine()
        result = engine._format_prompt("Hello")
        assert "<|im_start|>user\nHello /no_think<|im_end|>" in result
        assert "<|im_start|>assistant\n" in result
        assert "system" not in result

    def test_with_system_prompt(self):
        engine = LLMEngine(system_prompt="You are Ada.")
        result = engine._format_prompt("Hello")
        assert "<|im_start|>system\nYou are Ada.<|im_end|>" in result
        assert "<|im_start|>user\nHello /no_think<|im_end|>" in result
        assert "<|im_start|>assistant\n" in result

    def test_think_mode(self):
        engine = LLMEngine()
        result = engine._format_prompt("Hello", think=True)
        assert "Hello<|im_end|>" in result
        assert "/no_think" not in result


# ── Engine with mock backend ────────────────────────────────────────────────


class TestEngineGenerate:
    def _mock_backend(self):
        backend = MagicMock()
        backend.name = "mock"
        backend.generate.return_value = "Hello from Ada."
        backend.stream.return_value = iter(["Hello", " from", " Ada."])
        return backend

    def test_generate(self):
        backend = self._mock_backend()
        engine = LLMEngine(backend=backend)
        result = engine.generate("Hi")
        assert result == "Hello from Ada."
        backend.load.assert_called_once()
        backend.generate.assert_called_once()

    def test_stream(self):
        backend = self._mock_backend()
        engine = LLMEngine(backend=backend)
        tokens = list(engine.stream("Hi"))
        assert "".join(tokens) == "Hello from Ada."

    def test_lazy_load(self):
        backend = self._mock_backend()
        engine = LLMEngine(backend=backend)
        backend.load.assert_not_called()
        engine.generate("Hi")
        backend.load.assert_called_once()
        # Second call doesn't reload
        engine.generate("Hi again")
        backend.load.assert_called_once()

    def test_raw_prompt(self):
        backend = self._mock_backend()
        engine = LLMEngine(backend=backend, system_prompt="You are Ada.")
        engine.generate("raw prompt here", raw=True)
        call_args = backend.generate.call_args
        # Raw prompt should NOT have ChatML wrapping
        assert call_args[0][0] == "raw prompt here"

    def test_unload(self):
        backend = self._mock_backend()
        engine = LLMEngine(backend=backend)
        engine.generate("Hi")
        engine.unload()
        assert not engine._loaded
        backend.unload.assert_called_once()

    def test_backend_name(self):
        backend = self._mock_backend()
        engine = LLMEngine(backend=backend)
        assert engine.backend_name == "mock"

    def test_backend_name_before_load(self):
        engine = LLMEngine()
        assert engine.backend_name == "not loaded"


# ── Backend detection ───────────────────────────────────────────────────────


class TestDetectBackend:
    def test_env_override_mlx_dir(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        with patch.dict(os.environ, {"GLYPHH_MODEL_PATH": str(model_dir)}):
            backend = _detect_backend()
            assert isinstance(backend, _MLXBackend)

    def test_env_override_gguf_file(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        with patch.dict(os.environ, {"GLYPHH_MODEL_PATH": str(gguf)}):
            backend = _detect_backend()
            assert isinstance(backend, _LlamaCppBackend)

    def test_env_override_invalid(self, tmp_path):
        with patch.dict(os.environ, {"GLYPHH_MODEL_PATH": str(tmp_path / "nope")}):
            with pytest.raises(FileNotFoundError):
                _detect_backend()

    def test_api_fallback(self):
        env = {
            "GLYPHH_MODEL_PATH": "",
            "GLYPHH_API_BASE": "http://localhost:8080/v1",
            "GLYPHH_API_KEY": "test-key",
            "GLYPHH_API_MODEL": "qwen3-1.5b",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("glyphh.llm.engine._model_dir") as mock_dir:
                mock_dir.return_value = Path("/nonexistent")
                backend = _detect_backend()
                assert isinstance(backend, _APIBackend)

    def test_no_backend_raises(self):
        env = {
            "GLYPHH_MODEL_PATH": "",
            "GLYPHH_API_BASE": "",
            "GLYPHH_API_KEY": "",
            "OPENAI_API_BASE": "",
            "OPENAI_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("glyphh.llm.engine._model_dir") as mock_dir:
                mock_dir.return_value = Path("/nonexistent")
                with pytest.raises(RuntimeError, match="No LLM backend"):
                    _detect_backend()


# ── Backend names ───────────────────────────────────────────────────────────

class TestStripThink:
    def test_strip_think_block(self):
        text = "<think>\nSome reasoning here\n</think>\n\nThe answer is 42."
        assert LLMEngine._strip_think(text) == "The answer is 42."

    def test_no_think_block(self):
        text = "Just a normal response."
        assert LLMEngine._strip_think(text) == "Just a normal response."

    def test_empty_think_block(self):
        text = "<think>\n\n</think>\n\nHello!"
        assert LLMEngine._strip_think(text) == "Hello!"

    def test_stream_filter_think(self):
        backend = MagicMock()
        backend.name = "mock"
        backend.stream.return_value = iter([
            "<think>", "\nreasoning\n", "</think>", "\n\nThe answer."
        ])
        engine = LLMEngine(backend=backend)
        tokens = list(engine.stream("Hi"))
        assert "".join(tokens).strip() == "The answer."


class TestBackendNames:
    def test_mlx_name(self):
        assert _MLXBackend("/fake").name == "mlx"

    def test_llama_cpp_name(self):
        assert _LlamaCppBackend("/fake").name == "llama-cpp"

    def test_api_name(self):
        assert _APIBackend("http://x", "key", "model").name == "api"


from pathlib import Path
