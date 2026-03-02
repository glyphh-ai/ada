"""Tests for model file discovery."""

import os
from pathlib import Path

import pytest

from glyphh.llm.discovery import resolve_model_path, _xdg_data_home


class TestResolveModelPath:
    """resolve_model_path() — 3-tier resolution."""

    def test_explicit_path_found(self, tmp_path):
        model_file = tmp_path / "test.gguf"
        model_file.touch()
        result = resolve_model_path(model_path=str(model_file))
        assert result == model_file

    def test_explicit_path_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            resolve_model_path(model_path=str(tmp_path / "missing.gguf"))

    def test_env_var_found(self, tmp_path, monkeypatch):
        model_file = tmp_path / "env_model.gguf"
        model_file.touch()
        monkeypatch.setenv("GLYPHH_MODEL_PATH", str(model_file))
        result = resolve_model_path()
        assert result == model_file

    def test_env_var_not_found(self, monkeypatch):
        monkeypatch.setenv("GLYPHH_MODEL_PATH", "/nonexistent/model.gguf")
        with pytest.raises(FileNotFoundError, match="GLYPHH_MODEL_PATH"):
            resolve_model_path()

    def test_xdg_default_found(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GLYPHH_MODEL_PATH", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        model_dir = tmp_path / "glyphh" / "models"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "Qwen3-4B-Q4_K_M.gguf"
        model_file.touch()
        result = resolve_model_path()
        assert result == model_file

    def test_nothing_found_gives_instructions(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GLYPHH_MODEL_PATH", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Recommended"):
            resolve_model_path()

    def test_explicit_takes_priority_over_env(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit.gguf"
        explicit.touch()
        env_file = tmp_path / "env.gguf"
        env_file.touch()
        monkeypatch.setenv("GLYPHH_MODEL_PATH", str(env_file))
        result = resolve_model_path(model_path=str(explicit))
        assert result == explicit

    def test_custom_filename(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GLYPHH_MODEL_PATH", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        model_dir = tmp_path / "glyphh" / "models"
        model_dir.mkdir(parents=True)
        custom = model_dir / "custom-model.gguf"
        custom.touch()
        result = resolve_model_path(model_filename="custom-model.gguf")
        assert result == custom


class TestXdgDataHome:
    """_xdg_data_home() platform detection."""

    def test_respects_env_var(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")
        assert _xdg_data_home() == Path("/custom/data")

    def test_default_without_env(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        result = _xdg_data_home()
        assert isinstance(result, Path)
        assert result.is_absolute()
