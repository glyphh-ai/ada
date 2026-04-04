"""
CLI setup command — one-command environment bootstrap.

glyphh setup              Auto-detect best backend, download model, smoke test
glyphh setup --backend mlx       Force MLX backend (Apple Silicon)
glyphh setup --backend llama-cpp Force llama-cpp-python backend
glyphh setup --backend api       Use an external OpenAI-compatible API
glyphh setup --cpu               Force CPU-only llama-cpp-python
glyphh setup --status            Show current setup status
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import click

from .. import theme


# ── Constants ────────────────────────────────────────────────────────────────

# Backend identifiers for LLM setup (retained for model download support)
_BACKEND_MLX = "mlx"
_BACKEND_LLAMA_CPP = "llama-cpp"
_BACKEND_API = "api"

# MLX packages
_MLX_PACKAGES = ["mlx>=0.18.0", "mlx-lm>=0.20.0"]

# llama-cpp-python
_LLAMA_CPP_PACKAGE = "llama-cpp-python"
_LLAMA_CPP_MIN_VERSION = "0.3.0"

# Pre-built wheel indices for llama-cpp-python
_WHEEL_INDICES = {
    "cpu":   "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    "metal": "https://abetlen.github.io/llama-cpp-python/whl/metal",
    "cu124": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "cu123": "https://abetlen.github.io/llama-cpp-python/whl/cu123",
}

# Default models — Ada's voice
# MLX: Qwen3.5-2B (newer arch, better quality, Apple Silicon)
# GGUF: Qwen3-1.7B (universal, Linux/Docker)
_MLX_MODEL_REPO = "mlx-community/Qwen3.5-2B-4bit"
_MLX_MODEL_DIRNAME = "Qwen3.5-2B-4bit"

_GGUF_MODEL_REPO = "unsloth/Qwen3-1.7B-GGUF"
_GGUF_MODEL_FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"
_GGUF_MODEL_URL = (
    f"https://huggingface.co/{_GGUF_MODEL_REPO}/resolve/main/{_GGUF_MODEL_FILENAME}"
)


def _xdg_data_home() -> Path:
    """Return XDG_DATA_HOME or platform default."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg)
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".local" / "share"


def _model_dir() -> Path:
    return _xdg_data_home() / "glyphh" / "models"


# ── Platform detection ───────────────────────────────────────────────────────

def _detect_platform() -> dict:
    """Detect OS, architecture, and GPU availability."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        ),
        "gpu": None,
        "recommended_backend": _BACKEND_LLAMA_CPP,
        "llama_cpp_variant": "cpu",
    }

    if info["os"] == "Darwin" and info["arch"] == "arm64":
        info["gpu"] = "Apple Silicon (Metal)"
        info["recommended_backend"] = _BACKEND_MLX
        info["llama_cpp_variant"] = "metal"

    elif info["os"] == "Darwin":
        # Intel Mac — CPU only
        info["llama_cpp_variant"] = "cpu"

    elif info["os"] == "Linux" and shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                info["gpu"] = result.stdout.strip().split("\n")[0]
                info["llama_cpp_variant"] = "cu124"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return info


# ── Status checks ────────────────────────────────────────────────────────────

def _check_mlx_installed() -> str | None:
    """Return installed mlx-lm version, or None."""
    try:
        from importlib.metadata import version
        return version("mlx-lm")
    except Exception:
        return None


def _check_llama_cpp_installed() -> str | None:
    """Return installed llama-cpp-python version, or None."""
    try:
        from importlib.metadata import version
        return version(_LLAMA_CPP_PACKAGE)
    except Exception:
        return None


def _check_api_configured() -> str | None:
    """Return the configured API base URL, or None."""
    return os.environ.get("GLYPHH_API_BASE") or os.environ.get("OPENAI_API_BASE")


def _check_mlx_model_exists() -> Path | None:
    """Return MLX model path if it exists."""
    env_path = os.environ.get("GLYPHH_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir() and (p / "config.json").exists():
            return p

    p = _model_dir() / _MLX_MODEL_DIRNAME
    if p.is_dir() and (p / "config.json").exists():
        return p
    return None


def _check_gguf_model_exists() -> Path | None:
    """Return GGUF model path if it exists."""
    env_path = os.environ.get("GLYPHH_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    p = _model_dir() / _GGUF_MODEL_FILENAME
    if p.is_file():
        return p
    return None


# ── Installation steps ───────────────────────────────────────────────────────

def _install_mlx() -> bool:
    """Install MLX and mlx-lm."""
    cmd = [sys.executable, "-m", "pip", "install"] + _MLX_PACKAGES

    click.secho("  Installing MLX backend...", fg=theme.TEXT)
    click.secho(f"  $ {' '.join(cmd)}", fg=theme.TEXT_DIM)
    click.echo()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            click.secho("  Installed successfully.", fg=theme.SUCCESS)
            return True
        else:
            click.secho("  Installation failed:", fg=theme.ERROR)
            for line in result.stderr.strip().splitlines()[-10:]:
                click.secho(f"    {line}", fg=theme.TEXT_DIM)
            return False
    except subprocess.TimeoutExpired:
        click.secho("  Installation timed out (5 min).", fg=theme.ERROR)
        return False


def _install_llama_cpp(variant: str) -> bool:
    """Install llama-cpp-python with the appropriate pre-built wheel."""
    package = f"{_LLAMA_CPP_PACKAGE}>={_LLAMA_CPP_MIN_VERSION}"
    index_url = _WHEEL_INDICES.get(variant)

    cmd = [sys.executable, "-m", "pip", "install", package]
    if index_url:
        cmd.extend(["--extra-index-url", index_url])

    click.secho(f"  Installing {_LLAMA_CPP_PACKAGE} ({variant})...", fg=theme.TEXT)
    click.secho(f"  $ {' '.join(cmd)}", fg=theme.TEXT_DIM)
    click.echo()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            click.secho("  Installed successfully.", fg=theme.SUCCESS)
            return True
        else:
            click.secho("  Installation failed:", fg=theme.ERROR)
            for line in result.stderr.strip().splitlines()[-10:]:
                click.secho(f"    {line}", fg=theme.TEXT_DIM)
            return False
    except subprocess.TimeoutExpired:
        click.secho("  Installation timed out (5 min).", fg=theme.ERROR)
        return False


def _configure_api_backend() -> bool:
    """Interactive configuration for API backend."""
    click.secho("  API backend configuration:", fg=theme.TEXT)
    click.echo()

    base_url = os.environ.get("GLYPHH_API_BASE") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("GLYPHH_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("GLYPHH_API_MODEL", "")

    if base_url:
        click.secho(f"    Base URL:  {base_url}", fg=theme.SUCCESS)
    else:
        click.secho(
            "    Set GLYPHH_API_BASE (or OPENAI_API_BASE) to your endpoint.",
            fg=theme.WARNING,
        )

    if api_key:
        click.secho(f"    API key:   {'*' * 8}...{api_key[-4:]}", fg=theme.SUCCESS)
    else:
        click.secho(
            "    Set GLYPHH_API_KEY (or OPENAI_API_KEY) for authentication.",
            fg=theme.WARNING,
        )

    if model_name:
        click.secho(f"    Model:     {model_name}", fg=theme.SUCCESS)
    else:
        click.secho(
            "    Set GLYPHH_API_MODEL for model name "
            "(default: auto from endpoint).",
            fg=theme.TEXT_DIM,
        )

    click.echo()

    if not base_url or not api_key:
        click.secho(
            "  Environment variables needed for API backend:",
            fg=theme.INFO,
        )
        click.secho(
            "    export GLYPHH_API_BASE=https://api.openai.com/v1",
            fg=theme.TEXT_DIM,
        )
        click.secho(
            "    export GLYPHH_API_KEY=sk-...",
            fg=theme.TEXT_DIM,
        )
        click.secho(
            "    export GLYPHH_API_MODEL=gpt-4.1-mini  # optional",
            fg=theme.TEXT_DIM,
        )
        click.echo()
        return bool(base_url and api_key)

    return True


def _download_mlx_model() -> bool:
    """Download the MLX model directory from HuggingFace."""
    dest = _model_dir() / _MLX_MODEL_DIRNAME
    dest.parent.mkdir(parents=True, exist_ok=True)

    click.secho(f"  Downloading {_MLX_MODEL_REPO}...", fg=theme.TEXT)
    click.secho(f"  To: {dest}", fg=theme.TEXT_DIM)
    click.echo()

    # Method 1: huggingface_hub (best — progress bar, resume support)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=_MLX_MODEL_REPO,
            local_dir=str(dest),
        )
        click.secho(
            f"  Download complete ({_dir_size_str(dest)}).",
            fg=theme.SUCCESS,
        )
        return True
    except ImportError:
        click.secho(
            "  huggingface-hub not installed. Installing...",
            fg=theme.TEXT_DIM,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface-hub"],
            capture_output=True, timeout=120,
        )
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=_MLX_MODEL_REPO,
                local_dir=str(dest),
            )
            click.secho(
                f"  Download complete ({_dir_size_str(dest)}).",
                fg=theme.SUCCESS,
            )
            return True
        except Exception as e:
            click.secho(f"  Download failed: {e}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Download failed: {e}", fg=theme.ERROR)

    click.secho("  You can download manually:", fg=theme.INFO)
    click.secho(
        f"    huggingface-cli download {_MLX_MODEL_REPO} --local-dir {dest}",
        fg=theme.INFO,
    )
    return False


def _download_gguf_model() -> bool:
    """Download the default GGUF model from HuggingFace."""
    dest = _model_dir() / _GGUF_MODEL_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".gguf.part")

    click.secho(f"  Downloading {_GGUF_MODEL_FILENAME}...", fg=theme.TEXT)
    click.secho(f"  From: {_GGUF_MODEL_URL}", fg=theme.TEXT_DIM)
    click.secho(f"  To:   {dest}", fg=theme.TEXT_DIM)
    click.echo()

    if _download_with_huggingface_hub(dest):
        return True
    if _download_with_httpx(dest, tmp):
        return True
    if _download_with_curl(dest, tmp):
        return True

    click.secho("  Download failed. You can download manually:", fg=theme.ERROR)
    click.secho(f"    curl -L -o {dest} {_GGUF_MODEL_URL}", fg=theme.INFO)
    return False


def _download_with_huggingface_hub(dest: Path) -> bool:
    """Download GGUF using huggingface_hub if available."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_GGUF_MODEL_REPO,
            filename=_GGUF_MODEL_FILENAME,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        downloaded = Path(path)
        if downloaded != dest and downloaded.exists():
            shutil.move(str(downloaded), str(dest))
        click.secho(
            f"  Download complete ({_file_size_str(dest)}).",
            fg=theme.SUCCESS,
        )
        return True
    except ImportError:
        return False
    except Exception as e:
        click.secho(f"  huggingface_hub download failed: {e}", fg=theme.TEXT_DIM)
        return False


def _download_with_httpx(dest: Path, tmp: Path) -> bool:
    """Download GGUF using httpx with streaming."""
    try:
        import httpx
    except ImportError:
        return False

    try:
        with httpx.stream(
            "GET", _GGUF_MODEL_URL, follow_redirects=True, timeout=600,
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            last_pct = -1

            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct and pct % 5 == 0:
                            click.secho(
                                f"\r  Progress: {pct}% "
                                f"({downloaded // (1024*1024)}MB"
                                f" / {total // (1024*1024)}MB)",
                                fg=theme.TEXT_DIM, nl=False,
                            )
                            last_pct = pct

        click.echo()
        shutil.move(str(tmp), str(dest))
        click.secho(
            f"  Download complete ({_file_size_str(dest)}).",
            fg=theme.SUCCESS,
        )
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        click.secho(f"  httpx download failed: {e}", fg=theme.TEXT_DIM)
        return False


def _download_with_curl(dest: Path, tmp: Path) -> bool:
    """Download GGUF using curl as last resort."""
    curl = shutil.which("curl")
    if not curl:
        return False

    try:
        result = subprocess.run(
            [curl, "-L", "--progress-bar", "-o", str(tmp), _GGUF_MODEL_URL],
            timeout=1200,
        )
        if result.returncode == 0 and tmp.exists():
            shutil.move(str(tmp), str(dest))
            click.secho(
                f"  Download complete ({_file_size_str(dest)}).",
                fg=theme.SUCCESS,
            )
            return True
        tmp.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _file_size_str(path: Path) -> str:
    """Human-readable file size."""
    size = path.stat().st_size
    if size >= 1024 ** 3:
        return f"{size / (1024**3):.1f} GB"
    if size >= 1024 ** 2:
        return f"{size / (1024**2):.0f} MB"
    return f"{size / 1024:.0f} KB"


def _dir_size_str(path: Path) -> str:
    """Human-readable total size of a directory."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total >= 1024 ** 3:
        return f"{total / (1024**3):.1f} GB"
    if total >= 1024 ** 2:
        return f"{total / (1024**2):.0f} MB"
    return f"{total / 1024:.0f} KB"


def _smoke_test() -> bool:
    """Quick smoke test: verify core SDK imports work."""
    click.secho("  Running smoke test...", fg=theme.TEXT)

    try:
        from glyphh import Encoder, EncoderConfig, Concept
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        concept = Concept(name="test", attributes={"key": "value"})
        glyph = encoder.encode(concept)
        click.secho(
            f"  Smoke test passed. SDK operational "
            f"(encoded test glyph, dim={len(glyph.global_cortex.data)}).",
            fg=theme.SUCCESS,
        )
        return True
    except Exception as e:
        click.secho(f"  Smoke test failed: {e}", fg=theme.ERROR)
        return False


def _llm_smoke_test() -> bool:
    """Smoke test: verify LLM generates tokens."""
    click.secho("  Running LLM smoke test...", fg=theme.TEXT)

    try:
        from glyphh.llm import LLMEngine
        import time

        engine = LLMEngine()
        start = time.monotonic()
        result = engine.generate("Say hello in one sentence.", max_tokens=32, temperature=0.3)
        elapsed = time.monotonic() - start
        engine.unload()

        tokens = len(result.split())
        click.secho(
            f"  LLM smoke test passed. Generated {tokens} tokens "
            f"in {elapsed:.1f}s (backend={engine.backend_name}).",
            fg=theme.SUCCESS,
        )
        click.secho(f"    → {result.strip()[:80]}", fg=theme.TEXT_DIM)
        return True
    except Exception as e:
        click.secho(f"  LLM smoke test failed: {e}", fg=theme.ERROR)
        return False


# ── CLI command ──────────────────────────────────────────────────────────────

@click.command("setup")
@click.option(
    "--cpu", is_flag=True, default=False,
    help="Force CPU-only llama-cpp-python (skip GPU detection)",
)
@click.option(
    "--backend",
    type=click.Choice(["mlx", "llama-cpp", "api"]),
    default=None,
    help="Force a specific backend",
)
@click.option("--skip-model", is_flag=True, default=False, help="Skip model download")
@click.option("--skip-test", is_flag=True, default=False, help="Skip smoke test")
@click.option("--status", is_flag=True, default=False, help="Show setup status only")
def setup_command(cpu, backend, skip_model, skip_test, status):
    """Bootstrap the Glyphh runtime environment.

    Detects your platform, installs the best LLM inference backend,
    downloads the default model, and verifies everything works.

    \b
    Backends:
      mlx        Apple Silicon native (fastest on M-series Macs)
      llama-cpp  Universal (CPU / CUDA / Metal / Vulkan)
      api        Bring your own LLM via OpenAI-compatible API

    \b
    Examples:
      glyphh setup                      # Auto-detect everything
      glyphh setup --backend mlx        # Force MLX (Apple Silicon only)
      glyphh setup --backend llama-cpp  # Force llama-cpp-python
      glyphh setup --backend api        # Use external API endpoint
      glyphh setup --cpu                # Force CPU-only (no GPU)
      glyphh setup --status             # Check what's installed
    """
    click.echo()
    click.secho("  Glyphh Setup", fg=theme.TEXT, bold=True)
    click.echo()

    # ── Detect platform ─────────────────────────────────────────────────
    info = _detect_platform()

    click.secho("  Platform", fg=theme.ACCENT, bold=True)
    click.secho(f"    OS:       {info['os']} {info['arch']}", fg=theme.TEXT_DIM)
    click.secho(f"    Python:   {info['python']}", fg=theme.TEXT_DIM)
    if info["gpu"]:
        click.secho(f"    GPU:      {info['gpu']}", fg=theme.TEXT_DIM)
    else:
        click.secho("    GPU:      not detected", fg=theme.TEXT_DIM)
    click.echo()

    # ── Check current state ─────────────────────────────────────────────
    mlx_version = _check_mlx_installed()
    llama_version = _check_llama_cpp_installed()
    api_base = _check_api_configured()
    mlx_model = _check_mlx_model_exists()
    gguf_model = _check_gguf_model_exists()

    click.secho("  Status", fg=theme.ACCENT, bold=True)

    if mlx_version:
        click.secho(f"    MLX:          mlx-lm {mlx_version}", fg=theme.SUCCESS)
    else:
        click.secho("    MLX:          not installed", fg=theme.TEXT_DIM)

    if llama_version:
        click.secho(
            f"    llama.cpp:    {_LLAMA_CPP_PACKAGE} {llama_version}",
            fg=theme.SUCCESS,
        )
    else:
        click.secho("    llama.cpp:    not installed", fg=theme.TEXT_DIM)

    if api_base:
        click.secho(f"    API:          {api_base}", fg=theme.SUCCESS)
    else:
        click.secho("    API:          not configured", fg=theme.TEXT_DIM)

    if mlx_model:
        click.secho(
            f"    MLX model:    {mlx_model} ({_dir_size_str(mlx_model)})",
            fg=theme.SUCCESS,
        )
    else:
        click.secho("    MLX model:    not found", fg=theme.TEXT_DIM)

    if gguf_model:
        click.secho(
            f"    GGUF model:   {gguf_model} ({_file_size_str(gguf_model)})",
            fg=theme.SUCCESS,
        )
    else:
        click.secho("    GGUF model:   not found", fg=theme.TEXT_DIM)

    click.echo()

    if status:
        return

    # ── Determine backend ───────────────────────────────────────────────
    if cpu:
        chosen_backend = _BACKEND_LLAMA_CPP
        llama_variant = "cpu"
    elif backend == "mlx":
        if info["os"] != "Darwin" or info["arch"] != "arm64":
            click.secho(
                "  MLX requires Apple Silicon (macOS arm64).",
                fg=theme.ERROR,
            )
            sys.exit(1)
        chosen_backend = _BACKEND_MLX
        llama_variant = None
    elif backend == "llama-cpp":
        chosen_backend = _BACKEND_LLAMA_CPP
        llama_variant = info["llama_cpp_variant"]
    elif backend == "api":
        chosen_backend = _BACKEND_API
        llama_variant = None
    else:
        chosen_backend = info["recommended_backend"]
        llama_variant = info["llama_cpp_variant"]

    click.secho(
        f"  Selected backend: {chosen_backend}",
        fg=theme.INFO,
    )
    click.echo()

    # ── Step 1: Install backend ─────────────────────────────────────────
    if chosen_backend == _BACKEND_API:
        click.secho(
            "  [1/3] Configuring API backend",
            fg=theme.ACCENT, bold=True,
        )
        click.echo()
        _configure_api_backend()

    elif chosen_backend == _BACKEND_MLX:
        if mlx_version:
            click.secho(
                f"  [1/3] MLX already installed (v{mlx_version}), skipping.",
                fg=theme.TEXT_DIM,
            )
        else:
            click.secho(
                "  [1/3] Installing MLX backend",
                fg=theme.ACCENT, bold=True,
            )
            click.echo()
            if not _install_mlx():
                click.echo()
                click.secho(
                    "  Setup failed at backend installation.", fg=theme.ERROR,
                )
                click.secho(
                    "  Try: pip install mlx>=0.18.0 mlx-lm>=0.20.0",
                    fg=theme.INFO,
                )
                sys.exit(1)
    else:
        if llama_version:
            click.secho(
                f"  [1/3] llama-cpp-python already installed "
                f"(v{llama_version}), skipping.",
                fg=theme.TEXT_DIM,
            )
        else:
            click.secho(
                f"  [1/3] Installing llama-cpp-python ({llama_variant})",
                fg=theme.ACCENT, bold=True,
            )
            click.echo()
            if not _install_llama_cpp(llama_variant):
                click.echo()
                click.secho(
                    "  Setup failed at backend installation.",
                    fg=theme.ERROR,
                )
                click.secho(
                    f"  Try: pip install {_LLAMA_CPP_PACKAGE}>="
                    f"{_LLAMA_CPP_MIN_VERSION}",
                    fg=theme.INFO,
                )
                sys.exit(1)

    click.echo()

    # ── Step 2: Download model ──────────────────────────────────────────
    if chosen_backend == _BACKEND_API:
        click.secho(
            "  [2/3] No local model needed for API backend.",
            fg=theme.TEXT_DIM,
        )
    elif skip_model:
        click.secho(
            "  [2/3] Model download skipped (--skip-model).",
            fg=theme.TEXT_DIM,
        )
    elif chosen_backend == _BACKEND_MLX:
        if mlx_model:
            click.secho(
                f"  [2/3] MLX model already present "
                f"({_dir_size_str(mlx_model)}), skipping.",
                fg=theme.TEXT_DIM,
            )
        else:
            click.secho(
                "  [2/3] Downloading model (MLX)",
                fg=theme.ACCENT, bold=True,
            )
            click.echo()
            if not _download_mlx_model():
                click.echo()
                click.secho(
                    "  Setup failed at model download.", fg=theme.ERROR,
                )
                dest = _model_dir() / _MLX_MODEL_DIRNAME
                click.secho(
                    f"  Download manually:\n"
                    f"    huggingface-cli download {_MLX_MODEL_REPO} "
                    f"--local-dir {dest}",
                    fg=theme.INFO,
                )
                sys.exit(1)
    else:
        if gguf_model:
            click.secho(
                f"  [2/3] GGUF model already present "
                f"({_file_size_str(gguf_model)}), skipping.",
                fg=theme.TEXT_DIM,
            )
        else:
            click.secho(
                "  [2/3] Downloading model (GGUF)",
                fg=theme.ACCENT, bold=True,
            )
            click.echo()
            if not _download_gguf_model():
                click.echo()
                click.secho(
                    "  Setup failed at model download.", fg=theme.ERROR,
                )
                dest = _model_dir() / _GGUF_MODEL_FILENAME
                click.secho(
                    f"  Download manually:\n"
                    f"    curl -L -o {dest} {_GGUF_MODEL_URL}",
                    fg=theme.INFO,
                )
                sys.exit(1)

    click.echo()

    # ── Step 3: Smoke test ──────────────────────────────────────────────
    if skip_test:
        click.secho(
            "  [3/4] Smoke test skipped (--skip-test).",
            fg=theme.TEXT_DIM,
        )
    elif chosen_backend == _BACKEND_API:
        click.secho(
            "  [3/4] Smoke test skipped for API backend "
            "(requires running endpoint).",
            fg=theme.TEXT_DIM,
        )
    else:
        click.secho(
            "  [3/4] Verifying SDK", fg=theme.ACCENT, bold=True,
        )
        click.echo()
        _smoke_test()

    click.echo()

    # ── Step 4: LLM smoke test ─────────────────────────────────────────
    if skip_test:
        click.secho(
            "  [4/4] LLM smoke test skipped (--skip-test).",
            fg=theme.TEXT_DIM,
        )
    elif chosen_backend == _BACKEND_API and not (_check_api_configured()):
        click.secho(
            "  [4/4] LLM smoke test skipped (API not configured).",
            fg=theme.TEXT_DIM,
        )
    else:
        click.secho(
            "  [4/4] Verifying LLM generation", fg=theme.ACCENT, bold=True,
        )
        click.echo()
        _llm_smoke_test()

    click.echo()

    # ── Done ────────────────────────────────────────────────────────────
    click.secho("  Setup complete.", fg=theme.SUCCESS, bold=True)
    click.echo()
    click.secho("  Next steps:", fg=theme.TEXT)
    click.secho(
        "    glyphh dev .         Start a local dev server",
        fg=theme.TEXT_DIM,
    )
    click.secho(
        "    glyphh chat          Interactive chat session",
        fg=theme.TEXT_DIM,
    )
    click.echo()
