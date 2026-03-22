"""
.glyphh package format — zip-based model packaging.

A .glyphh file is a zip archive containing the full model directory:
  manifest.yaml      # model identity metadata (required)
  config.yaml        # encoder/runtime config
  encoder.py         # custom encoder
  intent.py          # intent extraction
  data/              # training data (exemplars, etc.)
  apps/              # per-app modules (Pipedream)
  classes/           # per-class modules (BFCL)
  ...                # any other model-specific files/dirs

Build artifacts (__pycache__, .git, tests/, etc.) are excluded.

The CLI can deploy from either:
  - A .glyphh file directly
  - A model directory (packages it first)
"""

import logging
import zipfile
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Directories excluded from packaging (build artifacts, not needed at runtime)
PACKAGE_EXCLUDE_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".venv", "venv",
    "tests", "results", ".mypy_cache", ".ruff_cache", "node_modules",
}

# Root-level files excluded from packaging (build-time only)
PACKAGE_EXCLUDE_ROOT_FILES = {
    "build.py", "discover.py", "tests.py", "test.py",
    "gap_analysis.py", "run_bfcl.py",
}


def _find_manifest(path: Path) -> Optional[Path]:
    """Find manifest.yaml in a directory — checks root then .glyphh/."""
    root = path / "manifest.yaml"
    if root.exists():
        return root
    glyphh = path / ".glyphh" / "manifest.yaml"
    if glyphh.exists():
        return glyphh
    return None


def is_model_dir(path: Path) -> bool:
    """Check if a directory is a glyphh model project."""
    return path.is_dir() and _find_manifest(path) is not None


def find_model_dir(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from start (default cwd) looking for a model directory."""
    current = (start or Path.cwd()).resolve()
    for _ in range(20):  # safety limit
        if is_model_dir(current):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def read_manifest(model_dir: Path) -> dict:
    """Read manifest.yaml from a model directory."""
    manifest_path = _find_manifest(model_dir)
    if not manifest_path:
        return {}
    try:
        return yaml.safe_load(manifest_path.read_text()) or {}
    except Exception:
        return {}


def package_model(model_dir: Path, output: Optional[Path] = None) -> Path:
    """Package a model directory into a .glyphh file.

    Bundles everything in the model directory except build artifacts.
    Subdirectory structure is preserved (apps/, classes/, data/, etc.).

    Args:
        model_dir: Path to the model directory containing manifest.yaml
        output: Output path for the .glyphh file (default: <model_dir>/<model_id>.glyphh)

    Returns:
        Path to the created .glyphh file
    """
    manifest = read_manifest(model_dir)
    model_id = manifest.get("model_id", model_dir.name)

    if output is None:
        output = model_dir / f"{model_id}.glyphh"

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for child in sorted(model_dir.rglob("*")):
            if not child.is_file():
                continue

            rel = child.relative_to(model_dir)
            parts = rel.parts

            # Skip files in excluded directories
            if any(p in PACKAGE_EXCLUDE_DIRS or p.startswith(".") for p in parts[:-1]):
                continue

            # Skip hidden files
            if rel.name.startswith("."):
                continue

            # Skip excluded root-level files
            if len(parts) == 1 and rel.name in PACKAGE_EXCLUDE_ROOT_FILES:
                continue

            # Skip .glyphh files (don't nest packages)
            if rel.suffix == ".glyphh":
                continue

            zf.write(child, str(rel))

    logger.info(f"Packaged {model_id} -> {output}")
    return output


def unpack_model(glyphh_file: Path, dest: Optional[Path] = None) -> Path:
    """Unpack a .glyphh file into a model directory.

    Args:
        glyphh_file: Path to the .glyphh file
        dest: Destination directory (default: ./<model_id>/)

    Returns:
        Path to the unpacked model directory
    """
    with zipfile.ZipFile(glyphh_file, "r") as zf:
        # Read manifest to get model_id
        if "manifest.yaml" in zf.namelist():
            manifest = yaml.safe_load(zf.read("manifest.yaml")) or {}
        else:
            manifest = {}

        model_id = manifest.get("model_id", glyphh_file.stem)

        if dest is None:
            dest = Path.cwd() / model_id

        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)

    logger.info(f"Unpacked {glyphh_file.name} -> {dest}")
    return dest


def discover_local_models(search_dirs: Optional[list[Path]] = None) -> list[dict]:
    """Discover model directories and .glyphh files on disk.

    Args:
        search_dirs: Directories to scan (default: cwd, ./models/, ./custom_models/)

    Returns:
        List of dicts with model info: model_id, name, path, type (dir|file), glyphs
    """
    if search_dirs is None:
        cwd = Path.cwd()
        search_dirs = [cwd, cwd / "models", cwd / "custom_models"]

    models = []
    seen = set()

    for base in search_dirs:
        if not base.exists():
            continue

        # Check if base itself is a model dir
        if is_model_dir(base) and str(base.resolve()) not in seen:
            seen.add(str(base.resolve()))
            manifest = read_manifest(base)
            models.append(_model_info(base, manifest, "dir"))
            continue

        # Scan children
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            resolved = str(child.resolve())
            if resolved in seen:
                continue

            if child.is_dir() and is_model_dir(child):
                seen.add(resolved)
                manifest = read_manifest(child)
                models.append(_model_info(child, manifest, "dir"))
            elif child.is_file() and child.suffix == ".glyphh":
                seen.add(resolved)
                try:
                    with zipfile.ZipFile(child, "r") as zf:
                        if "manifest.yaml" in zf.namelist():
                            manifest = yaml.safe_load(zf.read("manifest.yaml")) or {}
                        else:
                            manifest = {}
                except Exception:
                    manifest = {}
                models.append(_model_info(child, manifest, "file"))

    return models


def _model_info(path: Path, manifest: dict, model_type: str) -> dict:
    """Build a model info dict."""
    data_dir = path / "data" if path.is_dir() else None
    glyph_count = 0
    if data_dir and data_dir.exists():
        for jsonl in data_dir.glob("*.jsonl"):
            try:
                glyph_count += sum(1 for line in jsonl.open() if line.strip())
            except Exception:
                pass

    return {
        "model_id": manifest.get("model_id", path.stem),
        "name": manifest.get("name", path.stem),
        "version": manifest.get("version", ""),
        "category": manifest.get("category", ""),
        "path": str(path),
        "type": model_type,
        "glyphs": glyph_count,
    }
