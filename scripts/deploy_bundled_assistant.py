#!/usr/bin/env python3
"""
Deploy the bundled assistant.glyphh model to the local runtime on startup.

Finds the assistant model inside the installed glyphh SDK package and
deploys it via the ModelManager directly (in-process) or via HTTP POST.

Two modes:
  1. In-process: called from the runtime's lifespan startup (no HTTP needed)
  2. CLI: POST to a running runtime instance

Usage (CLI):
    python scripts/deploy_bundled_assistant.py [--runtime-url URL] [--retries N]
"""

import argparse
import logging
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)


def find_model() -> Path | None:
    """Locate assistant.glyphh — check models/ dir first, then legacy paths."""
    # Primary: models/ directory at repo root
    repo_root = Path(__file__).parent.parent
    model_path = repo_root / "models" / "assistant" / "assistant.glyphh"
    if model_path.exists():
        return model_path

    # Legacy: inside the glyphh engine package
    try:
        import glyphh
        pkg_dir = Path(glyphh.__file__).parent
        legacy_path = pkg_dir / "assistant" / "model" / "assistant.glyphh"
        if legacy_path.exists():
            return legacy_path
    except ImportError:
        pass

    # Fallback: check common locations
    for candidate in [
        Path("/app/assistant.glyphh"),
        Path("assistant.glyphh"),
    ]:
        if candidate.exists():
            return candidate

    return None


async def deploy_in_process(model_manager) -> bool:
    """Deploy the assistant model directly via ModelManager (no HTTP).

    This is called from the runtime's lifespan startup. It loads the
    .glyphh file and triggers concept encoding, same as the /api/deploy
    endpoint but without the HTTP round-trip.
    """
    model_path = find_model()
    if model_path is None:
        logger.info("No bundled assistant model found — skipping auto-deploy")
        return False

    content = model_path.read_bytes()
    org_id, model_id = "glyphh", "assistant"

    try:
        await model_manager.load_model_from_bytes(content, org_id, model_id)
        logger.info(f"Auto-deployed assistant model from {model_path}")

        # Trigger concept loading (same as deployment endpoint)
        import gzip
        import json
        decompressed = gzip.decompress(content)
        model_data = json.loads(decompressed.decode("utf-8"))
        concepts = model_data.get("concepts")

        if concepts:
            try:
                from api.routes.listeners import get_async_listener_service
                async_service = get_async_listener_service()
                job_id = await async_service.start_load(
                    org_id=org_id,
                    model_id=model_id,
                    records=concepts,
                    batch_size=50,
                )
                logger.info(f"Assistant concept loading started: job {job_id}, {len(concepts)} concepts")
            except Exception as e:
                logger.warning(f"Concept loading failed (model still deployed): {e}")

        return True
    except Exception as e:
        logger.error(f"Failed to auto-deploy assistant model: {e}")
        return False


def deploy_http(model_path: Path, runtime_url: str, retries: int = 10, delay: float = 3.0) -> None:
    """POST the model file to the runtime deploy endpoint (CLI mode)."""
    url = f"{runtime_url.rstrip('/')}/api/deploy?org_id=glyphh&model_id=assistant"
    model_bytes = model_path.read_bytes()

    boundary = "----GlyphhModelBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="assistant.glyphh"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + model_bytes + f"\r\n--{boundary}--\r\n".encode()

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"Deployed assistant model ({len(model_bytes)} bytes) -> {resp.status}")
                return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(delay)

    print("Error: failed to deploy assistant model after all retries")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deploy bundled assistant model to runtime")
    parser.add_argument("--runtime-url", default="http://localhost:8000", help="Runtime URL")
    parser.add_argument("--retries", type=int, default=10, help="Number of retries")
    args = parser.parse_args()

    model_path = find_model()
    if model_path is None:
        print("Error: assistant.glyphh not found")
        sys.exit(1)

    print(f"Found assistant model: {model_path}")
    deploy_http(model_path, args.runtime_url, args.retries)


if __name__ == "__main__":
    main()
