#!/usr/bin/env python3
"""
Build assistant.glyphh from JSONL knowledge files.

Standard model build script. Run from the model directory or
via the runtime's model build pipeline:

    python models/assistant/build.py
    python models/assistant/build.py --output path/to/output.glyphh
"""

import argparse
import json
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent
RUNTIME_ROOT = MODEL_DIR.parent.parent

# Ensure both the runtime root (for glyphh.*) and model dir (for encoder.py) are importable
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(MODEL_DIR))

from encoder import ENCODER_CONFIG  # noqa: E402 — from MODEL_DIR
from glyphh.model.package import GlyphhModel  # noqa: E402
from glyphh.assistant.concept_converter import entry_to_concept, procedure_to_concepts  # noqa: E402

DATA_DIR = MODEL_DIR / "data"
DEFAULT_OUTPUT = MODEL_DIR / "assistant.glyphh"
JSONL_FILES = [
    "commands.jsonl", "concepts.jsonl", "gql.jsonl",
    "quick_actions.jsonl", "workflows.jsonl", "followups.jsonl",
]


def load_all_jsonl(data_dir: Path) -> list[dict]:
    entries = []
    for filename in JSONL_FILES:
        path = data_dir / filename
        if not path.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue
        count = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError:
                    print(f"  Warning: bad JSON at {filename}:{lineno}, skipping")
        print(f"  {filename}: {count} entries")
    return entries


def concept_to_record(concept) -> dict:
    attrs = concept.attributes
    return {
        "router": {
            "intent": {
                "verb": attrs.get("verb", ""),
                "object": attrs.get("object", ""),
                "domain": attrs.get("domain", ""),
                "keywords": attrs.get("keywords", ""),
            },
            "action": {
                "action_type": attrs.get("action_type", ""),
                "action_id": attrs.get("action_id", ""),
            },
            "context": {
                "context_type": attrs.get("context_type", 1.0),
            },
        },
        "_metadata": concept.metadata,
    }


def build(output_path: Path | None = None) -> None:
    """Standard build entry point."""
    output = output_path or DEFAULT_OUTPUT

    print("Loading JSONL data files...")
    entries = load_all_jsonl(DATA_DIR)
    if not entries:
        print("Error: No entries found.")
        sys.exit(1)

    print(f"\nConverting {len(entries)} entries to Concepts...")
    concepts = [entry_to_concept(e) for e in entries]

    # Try loading procedures (optional)
    try:
        from glyphh.assistant.procedures import load_builtin_procedures
        registry = load_builtin_procedures()
        procs = [{"name": p.name, "description": p.description, "triggers": p.triggers,
                   "category": p.category, "response_template": p.response_template}
                  for p in registry.all_procedures()]
        if procs:
            concepts.extend(procedure_to_concepts(procs))
            print(f"  {len(procs)} procedure concepts")
    except ImportError:
        print("  Note: procedures module not found, skipping")

    print(f"\nTotal concepts: {len(concepts)}")
    records = [concept_to_record(c) for c in concepts]

    model = GlyphhModel(
        name="glyphh-assistant",
        version="0.2.0",
        encoder_config=ENCODER_CONFIG,
        glyphs=[],
        readme="# Glyphh Assistant\n\nRouter-style model for the CLI assistant.",
        metadata={"domain": "assistant", "entry_count": len(records)},
    )
    model.set_concepts(records)

    print(f"Saving to {output}...")
    model.to_file(str(output))
    print(f"\n+ Built {output} ({len(records)} concepts, dim={ENCODER_CONFIG.dimension})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build assistant.glyphh")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)
