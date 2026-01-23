from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from api.main import app  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise
    raise SystemExit(
        "FastAPI is not installed. Run this script with the runtime venv "
        "or install dependencies from glyphh-runtime/requirements.txt."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Glyphh Runtime OpenAPI spec.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "openapi.json"),
        help="Output path for the OpenAPI spec.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = app.openapi()
    out_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote OpenAPI spec to {out_path}")


if __name__ == "__main__":
    main()
