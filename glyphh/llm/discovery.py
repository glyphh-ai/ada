"""Model file discovery for local GGUF models."""

import os
import platform
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"


def _xdg_data_home() -> Path:
    """Return XDG_DATA_HOME or platform default."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg)
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".local" / "share"


def resolve_model_path(
    model_path: Optional[str | Path] = None,
    model_filename: str = DEFAULT_MODEL_FILENAME,
) -> Path:
    """Resolve GGUF model path from explicit path, env var, or XDG default.

    Search order:
      1. Explicit model_path argument
      2. GLYPHH_MODEL_PATH environment variable
      3. XDG_DATA_HOME/glyphh/models/{model_filename}

    Raises FileNotFoundError with instructions if not found.
    """
    # 1. Explicit
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model file not found: {p}")

    # 2. Environment variable
    env_path = os.environ.get("GLYPHH_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"GLYPHH_MODEL_PATH points to missing file: {p}")

    # 3. XDG default
    default_path = _xdg_data_home() / "glyphh" / "models" / model_filename
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        f"No GGUF model found. Place your model at:\n"
        f"  {default_path}\n"
        f"Or set GLYPHH_MODEL_PATH=/path/to/model.gguf\n"
        f"Or pass model_path= to LLMEngine().\n"
        f"Recommended: Qwen3-4B-Instruct Q4_K_M (~2.5GB)"
    )
