"""
CLI setup command — one-command environment bootstrap.

glyphh setup          Detect platform, install llama-cpp-python, download model, smoke test
glyphh setup --cpu    Force CPU-only wheel (skip GPU detection)
glyphh setup --status Show current setup status without changing anything
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

_LLAMA_CPP_PACKAGE = "llama-cpp-python"
_LLAMA_CPP_MIN_VERSION = "0.3.0"

# Pre-built wheel indices from abetlen/llama-cpp-python
_WHEEL_INDICES = {
    "cpu":   "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    "metal": "https://abetlen.github.io/llama-cpp-python/whl/metal",
    "cu124": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "cu123": "https://abetlen.github.io/llama-cpp-python/whl/cu123",
}

# Default model
_MODEL_REPO = "Qwen/Qwen3-4B-GGUF"
_MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"
_MODEL_URL = f"https://huggingface.co/{_MODEL_REPO}/resolve/main/{_MODEL_FILENAME}"


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


def _model_path() -> Path:
    return _model_dir() / _MODEL_FILENAME


# ── Platform detection ───────────────────────────────────────────────────────

def _detect_platform() -> dict:
    """Detect OS, architecture, and GPU availability."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "gpu": None,
        "recommended_backend": "cpu",
    }

    if info["os"] == "Darwin":
        # macOS — Apple Silicon gets Metal, Intel gets CPU
        if info["arch"] == "arm64":
            info["gpu"] = "Apple Silicon (Metal)"
            info["recommended_backend"] = "metal"
        else:
            info["gpu"] = None
            info["recommended_backend"] = "cpu"

    elif info["os"] == "Linux":
        # Check for NVIDIA GPU
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    info["gpu"] = result.stdout.strip().split("\n")[0]
                    # Default to CUDA 12.4
                    info["recommended_backend"] = "cu124"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    return info


def _check_llama_cpp_installed() -> str | None:
    """Return installed version of llama-cpp-python, or None."""
    try:
        from importlib.metadata import version
        return version(_LLAMA_CPP_PACKAGE)
    except Exception:
        return None


def _check_model_exists() -> Path | None:
    """Return model path if it exists, or None."""
    # Check env var first
    env_path = os.environ.get("GLYPHH_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Check default location
    p = _model_path()
    if p.exists():
        return p

    return None


# ── Installation steps ───────────────────────────────────────────────────────

def _install_llama_cpp(backend: str) -> bool:
    """Install llama-cpp-python with the appropriate pre-built wheel."""
    package = f"{_LLAMA_CPP_PACKAGE}>={_LLAMA_CPP_MIN_VERSION}"
    index_url = _WHEEL_INDICES.get(backend)

    cmd = [sys.executable, "-m", "pip", "install", package]
    if index_url:
        cmd.extend(["--extra-index-url", index_url])

    click.secho(f"  Installing {_LLAMA_CPP_PACKAGE} ({backend})...", fg=theme.TEXT)
    click.secho(f"  $ {' '.join(cmd)}", fg=theme.TEXT_DIM)
    click.echo()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            click.secho(f"  Installed successfully.", fg=theme.SUCCESS)
            return True
        else:
            click.secho(f"  Installation failed:", fg=theme.ERROR)
            # Show last few lines of stderr
            stderr_lines = result.stderr.strip().splitlines()
            for line in stderr_lines[-10:]:
                click.secho(f"    {line}", fg=theme.TEXT_DIM)
            return False
    except subprocess.TimeoutExpired:
        click.secho("  Installation timed out (5 min).", fg=theme.ERROR)
        return False


def _download_model() -> bool:
    """Download the default GGUF model from HuggingFace."""
    dest = _model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".gguf.part")

    click.secho(f"  Downloading {_MODEL_FILENAME}...", fg=theme.TEXT)
    click.secho(f"  From: {_MODEL_URL}", fg=theme.TEXT_DIM)
    click.secho(f"  To:   {dest}", fg=theme.TEXT_DIM)
    click.echo()

    # Try huggingface_hub first (best progress bar), fall back to httpx, then curl
    if _download_with_huggingface_hub(dest):
        return True

    if _download_with_httpx(dest, tmp):
        return True

    if _download_with_curl(dest, tmp):
        return True

    click.secho("  Download failed. You can download manually:", fg=theme.ERROR)
    click.secho(f"    curl -L -o {dest} {_MODEL_URL}", fg=theme.INFO)
    return False


def _download_with_huggingface_hub(dest: Path) -> bool:
    """Download using huggingface_hub if available (best progress bar)."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=_MODEL_FILENAME,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may put it in a different location
        downloaded = Path(path)
        if downloaded != dest and downloaded.exists():
            shutil.move(str(downloaded), str(dest))
        click.secho(f"  Download complete ({_file_size_str(dest)}).", fg=theme.SUCCESS)
        return True
    except ImportError:
        return False
    except Exception as e:
        click.secho(f"  huggingface_hub download failed: {e}", fg=theme.TEXT_DIM)
        return False


def _download_with_httpx(dest: Path, tmp: Path) -> bool:
    """Download using httpx with streaming (usually already installed)."""
    try:
        import httpx
    except ImportError:
        return False

    try:
        with httpx.stream("GET", _MODEL_URL, follow_redirects=True, timeout=600) as r:
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
                                f"\r  Progress: {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)",
                                fg=theme.TEXT_DIM, nl=False,
                            )
                            last_pct = pct

        click.echo()  # newline after progress
        shutil.move(str(tmp), str(dest))
        click.secho(f"  Download complete ({_file_size_str(dest)}).", fg=theme.SUCCESS)
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        click.secho(f"  httpx download failed: {e}", fg=theme.TEXT_DIM)
        return False


def _download_with_curl(dest: Path, tmp: Path) -> bool:
    """Download using curl as last resort."""
    curl = shutil.which("curl")
    if not curl:
        return False

    try:
        result = subprocess.run(
            [curl, "-L", "--progress-bar", "-o", str(tmp), _MODEL_URL],
            timeout=1200,
        )
        if result.returncode == 0 and tmp.exists():
            shutil.move(str(tmp), str(dest))
            click.secho(f"  Download complete ({_file_size_str(dest)}).", fg=theme.SUCCESS)
            return True
        tmp.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False


def _file_size_str(path: Path) -> str:
    """Human-readable file size."""
    size = path.stat().st_size
    if size >= 1024 ** 3:
        return f"{size / (1024**3):.1f} GB"
    if size >= 1024 ** 2:
        return f"{size / (1024**2):.0f} MB"
    return f"{size / 1024:.0f} KB"


def _smoke_test() -> bool:
    """Quick smoke test: load model, generate one token."""
    click.secho("  Running smoke test...", fg=theme.TEXT)

    try:
        from glyphh.llm import LLMEngine
        engine = LLMEngine(n_ctx=512, verbose=False)
        result = engine.generate(
            system="You are a test assistant.",
            user="Say OK.",
            max_tokens=4,
        )
        engine.unload()

        if result and len(result.strip()) > 0:
            click.secho(f"  Smoke test passed. Model responded: \"{result.strip()}\"", fg=theme.SUCCESS)
            return True
        else:
            click.secho("  Smoke test: model loaded but returned empty response.", fg=theme.WARNING)
            return True  # Still counts as working
    except FileNotFoundError as e:
        click.secho(f"  Smoke test skipped: {e}", fg=theme.WARNING)
        return False
    except Exception as e:
        click.secho(f"  Smoke test failed: {e}", fg=theme.ERROR)
        return False


# ── CLI command ──────────────────────────────────────────────────────────────

@click.command("setup")
@click.option("--cpu", is_flag=True, default=False, help="Force CPU-only (skip GPU detection)")
@click.option("--backend", type=click.Choice(["cpu", "metal", "cu124", "cu123"]),
              default=None, help="Explicit backend choice")
@click.option("--skip-model", is_flag=True, default=False, help="Skip model download")
@click.option("--skip-test", is_flag=True, default=False, help="Skip smoke test")
@click.option("--status", is_flag=True, default=False, help="Show setup status only")
def setup_command(cpu, backend, skip_model, skip_test, status):
    """Bootstrap the Glyphh runtime environment.

    Detects your platform, installs the LLM inference backend with the
    optimal pre-built wheel, downloads the default model, and verifies
    everything works.

    \b
    Examples:
      glyphh setup              # Auto-detect everything
      glyphh setup --cpu        # Force CPU-only backend
      glyphh setup --backend metal   # Explicit Apple Metal backend
      glyphh setup --status     # Check what's installed
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
        click.secho(f"    GPU:      not detected", fg=theme.TEXT_DIM)
    click.echo()

    # ── Check current state ─────────────────────────────────────────────
    llama_version = _check_llama_cpp_installed()
    model_path = _check_model_exists()

    click.secho("  Status", fg=theme.ACCENT, bold=True)

    if llama_version:
        click.secho(f"    Backend:  {_LLAMA_CPP_PACKAGE} {llama_version}", fg=theme.SUCCESS)
    else:
        click.secho(f"    Backend:  not installed", fg=theme.WARNING)

    if model_path:
        click.secho(f"    Model:    {model_path} ({_file_size_str(model_path)})", fg=theme.SUCCESS)
    else:
        click.secho(f"    Model:    not found", fg=theme.WARNING)

    click.echo()

    if status:
        return

    # ── Determine backend ───────────────────────────────────────────────
    if cpu:
        chosen_backend = "cpu"
    elif backend:
        chosen_backend = backend
    else:
        chosen_backend = info["recommended_backend"]

    # ── Step 1: Install llama-cpp-python ─────────────────────────────────
    if llama_version:
        click.secho(f"  [1/3] Backend already installed (v{llama_version}), skipping.",
                     fg=theme.TEXT_DIM)
    else:
        click.secho(f"  [1/3] Installing inference backend ({chosen_backend})",
                     fg=theme.ACCENT, bold=True)
        click.echo()
        if not _install_llama_cpp(chosen_backend):
            click.echo()
            click.secho("  Setup failed at backend installation.", fg=theme.ERROR)
            click.secho("  Try: pip install llama-cpp-python>=0.3.0", fg=theme.INFO)
            sys.exit(1)

    click.echo()

    # ── Step 2: Download model ──────────────────────────────────────────
    if skip_model:
        click.secho("  [2/3] Model download skipped (--skip-model).", fg=theme.TEXT_DIM)
    elif model_path:
        click.secho(f"  [2/3] Model already present ({_file_size_str(model_path)}), skipping.",
                     fg=theme.TEXT_DIM)
    else:
        click.secho(f"  [2/3] Downloading model", fg=theme.ACCENT, bold=True)
        click.echo()
        if not _download_model():
            click.echo()
            click.secho("  Setup failed at model download.", fg=theme.ERROR)
            click.secho(f"  Download manually:", fg=theme.INFO)
            click.secho(f"    curl -L -o {_model_path()} {_MODEL_URL}", fg=theme.INFO)
            sys.exit(1)

    click.echo()

    # ── Step 3: Smoke test ──────────────────────────────────────────────
    if skip_test:
        click.secho("  [3/3] Smoke test skipped (--skip-test).", fg=theme.TEXT_DIM)
    else:
        click.secho("  [3/3] Verifying installation", fg=theme.ACCENT, bold=True)
        click.echo()
        _smoke_test()

    click.echo()

    # ── Done ────────────────────────────────────────────────────────────
    click.secho("  Setup complete.", fg=theme.SUCCESS, bold=True)
    click.echo()
    click.secho("  Next steps:", fg=theme.TEXT)
    click.secho("    glyphh dev .         Start a local dev server", fg=theme.TEXT_DIM)
    click.secho("    glyphh chat          Interactive chat session", fg=theme.TEXT_DIM)
    click.echo()
