#!/usr/bin/env python3
"""
Build assistant.glyphh from JSONL knowledge files.

Self-contained script that lives inside the SDK repo. Can be run
standalone or as part of the release pipeline.

Usage:
    python scripts/build_assistant_model.py
    python scripts/build_assistant_model.py --output path/to/output.glyphh
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the runtime repo is importable
runtime_root = Path(__file__).parent.parent
sys.path.insert(0, str(runtime_root))

from glyphh.model.package import GlyphhModel
from glyphh.assistant.encoder_config import ASSISTANT_ENCODER_CONFIG
from glyphh.assistant.concept_converter import entry_to_concept, procedure_to_concepts

DATA_DIR = runtime_root / "models" / "assistant" / "data"
DEFAULT_OUTPUT = runtime_root / "models" / "assistant" / "assistant.glyphh"
JSONL_FILES = ["commands.jsonl", "concepts.jsonl", "gql.jsonl", "quick_actions.jsonl", "workflows.jsonl", "followups.jsonl"]


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


def build_model(output_path: Path) -> None:
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
        version="1.0.0",
        encoder_config=ASSISTANT_ENCODER_CONFIG,
        glyphs=[],
        readme="# Glyphh Assistant\n\nRouter-style model for the CLI assistant.",
        metadata={"domain": "assistant", "entry_count": len(records)},
    )
    model.set_concepts(records)

    print(f"Saving to {output_path}...")
    model.to_file(str(output_path))
    print(f"\n+ Built {output_path} ({len(records)} concepts, dim={ASSISTANT_ENCODER_CONFIG.dimension})")


def main():
    parser = argparse.ArgumentParser(description="Build assistant.glyphh")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_model(args.output)


if __name__ == "__main__":
    main()
