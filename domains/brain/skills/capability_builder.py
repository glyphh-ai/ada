"""
Capability Builder — Ada creates new capabilities autonomously.

Given a domain description, Ada:
  1. Designs an encoder config (layers, roles, lexicons)
  2. Generates seed exemplars
  3. Creates the intent extraction logic
  4. Writes the capability to disk
  5. Loads it into the runtime

The LLM (Haiku) generates content. Ada validates and assembles
deterministically. No capability goes live without exemplars that
encode cleanly.

Usage (from think pipeline):
    builder = CapabilityBuilder(llm, model_manager, session_factory)
    result = await builder.build("customer-churn",
        description="Analyze customer churn patterns and risk factors",
        seed_queries=["what is our churn rate", "which customers are at risk"]
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Where new capabilities get written
CAPABILITIES_DIR = Path(__file__).resolve().parents[3] / "capabilities"


@dataclass
class BuildResult:
    """Result of building a new capability."""
    success: bool
    name: str
    path: Optional[Path] = None
    exemplar_count: int = 0
    error: Optional[str] = None
    elapsed_s: float = 0.0


class CapabilityBuilder:
    """Builds new Ada capabilities from a description.

    Ada's growth engine. She designs the encoder, generates exemplars,
    writes the files, and loads them into the runtime — all driven by
    her internal LLM but validated deterministically.
    """

    def __init__(self, llm, model_manager=None, session_factory=None):
        self._llm = llm
        self._model_manager = model_manager
        self._session_factory = session_factory

    async def build(
        self,
        name: str,
        description: str,
        seed_queries: list[str] | None = None,
        exemplar_count: int = 30,
    ) -> BuildResult:
        """Build a new capability end-to-end.

        Args:
            name: Capability name (lowercase, hyphenated). e.g. "customer-churn"
            description: What this capability does.
            seed_queries: Optional seed queries to include as exemplars.
            exemplar_count: Target number of exemplars to generate.
        """
        start = time.time()

        # Sanitize name
        name = re.sub(r'[^a-z0-9-]', '-', name.lower()).strip('-')
        if not name:
            return BuildResult(success=False, name=name, error="Invalid name")

        cap_dir = CAPABILITIES_DIR / name
        if cap_dir.exists():
            return BuildResult(success=False, name=name, error=f"Capability '{name}' already exists")

        if not self._llm or not self._llm.available:
            return BuildResult(success=False, name=name, error="LLM offline — cannot generate capability")

        try:
            # 1. Design the encoder
            logger.info(f"Building capability: {name}")
            encoder_design = await self._design_encoder(name, description)
            if not encoder_design:
                return BuildResult(success=False, name=name, error="Failed to design encoder")

            # 2. Generate exemplars
            exemplars = await self._generate_exemplars(
                name, description, seed_queries or [], exemplar_count
            )
            if len(exemplars) < 5:
                return BuildResult(success=False, name=name, error=f"Only generated {len(exemplars)} exemplars (need ≥5)")

            # 3. Generate intent extraction keywords
            keywords = await self._generate_keywords(name, description)

            # 4. Write capability to disk
            cap_dir.mkdir(parents=True)
            (cap_dir / "data").mkdir()

            self._write_manifest(cap_dir, name, description)
            self._write_config(cap_dir, name, description, encoder_design)
            self._write_encoder(cap_dir, name, description, encoder_design, keywords)
            self._write_exemplars(cap_dir, exemplars)

            # 5. Test in isolated subprocess
            from domains.brain.skills.capability_tester import CapabilityTester

            tester = CapabilityTester(llm=self._llm)
            test_report = await tester.test(
                capability_dir=cap_dir,
                capability_name=name,
                test_queries=seed_queries,
            )

            logger.info(f"Test: {test_report.summary}")

            # 6. Promote to live runtime only if tests pass
            loaded = False
            if test_report.promoted and self._model_manager:
                try:
                    from domains.brain.loader import ADA_ORG_ID
                    await self._model_manager.load_model_from_directory(
                        model_dir=cap_dir,
                        org_id=ADA_ORG_ID,
                        model_id=name,
                    )
                    loaded = True
                    logger.info(f"Capability '{name}' promoted to live runtime")
                except Exception as e:
                    logger.warning(f"Tested but failed to load: {e}")
            elif not test_report.promoted:
                logger.warning(
                    f"Capability '{name}' failed testing "
                    f"({test_report.pass_rate:.0%} pass rate, "
                    f"need {CapabilityTester.MIN_PASS_RATE if hasattr(CapabilityTester, 'MIN_PASS_RATE') else '60%'}). "
                    f"Files kept at {cap_dir} for debugging."
                )

            elapsed = time.time() - start
            logger.info(
                f"Built capability '{name}': {len(exemplars)} exemplars, "
                f"test {'PASS' if test_report.promoted else 'FAIL'}, "
                f"{'live' if loaded else 'not promoted'}, {elapsed:.1f}s"
            )

            return BuildResult(
                success=test_report.promoted,
                name=name,
                path=cap_dir,
                exemplar_count=len(exemplars),
                elapsed_s=elapsed,
                error=None if test_report.promoted else f"Tests failed ({test_report.pass_rate:.0%} pass rate)",
            )

        except Exception as e:
            logger.error(f"Capability build failed: {e}")
            # Clean up partial writes
            if cap_dir.exists():
                import shutil
                shutil.rmtree(cap_dir, ignore_errors=True)
            return BuildResult(success=False, name=name, error=str(e))

    # ── Encoder design ───────────────────────────────────────────────

    async def _design_encoder(self, name: str, description: str) -> Optional[dict]:
        """Ask Haiku to design the encoder layers for this domain."""
        prompt = f"""Design an HDC encoder for a capability called "{name}".
Description: {description}

Return a JSON object with this exact structure:
{{
  "layers": [
    {{
      "name": "intent",
      "weight": 0.40,
      "roles": [
        {{"name": "action", "type": "lexicon", "values": ["analyze", "detect", "query", "none"]}},
        {{"name": "signals", "type": "bow"}}
      ]
    }},
    {{
      "name": "domain",
      "weight": 0.35,
      "roles": [
        {{"name": "category", "type": "lexicon", "values": ["category1", "category2", "none"]}},
        {{"name": "signals", "type": "bow"}}
      ]
    }},
    {{
      "name": "semantic",
      "weight": 0.25,
      "roles": [
        {{"name": "description", "type": "bow"}},
        {{"name": "keywords", "type": "bow"}}
      ]
    }}
  ]
}}

Rules:
- 2-4 layers, weights must sum to 1.0
- Lexicon values should be domain-specific actions/categories (8-20 values each)
- Always include a semantic layer with description + keywords BoW roles
- Return ONLY the JSON, no explanation"""

        result = await self._llm.ask(prompt, max_tokens=1024)
        if not result:
            return None

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse encoder design: {e}")

        return None

    # ── Exemplar generation ──────────────────────────────────────────

    async def _generate_exemplars(
        self,
        name: str,
        description: str,
        seed_queries: list[str],
        target_count: int,
    ) -> list[dict]:
        """Generate seed exemplars using Haiku."""
        exemplars = []

        # Include seed queries first
        for q in seed_queries:
            exemplars.append({"text": q, "capability": name})

        # Generate more via LLM
        remaining = target_count - len(exemplars)
        if remaining > 0:
            prompt = f"""Generate {remaining} diverse natural language queries that someone would ask about:
"{description}"

Rules:
- One query per line
- Vary the phrasing: questions, commands, descriptions
- Include both simple ("what is X") and complex ("analyze the relationship between X and Y")
- No numbering, no explanation, just the queries"""

            result = await self._llm.ask(prompt, max_tokens=1024)
            if result:
                for line in result.strip().split("\n"):
                    line = line.strip().strip("-").strip("•").strip()
                    if line and len(line) > 5:
                        exemplars.append({"text": line, "capability": name})

        return exemplars

    # ── Keyword generation ───────────────────────────────────────────

    async def _generate_keywords(self, name: str, description: str) -> dict:
        """Generate action and domain keywords for intent extraction."""
        prompt = f"""For a capability called "{name}" ({description}), generate keyword mappings.

Return JSON:
{{
  "actions": {{"keyword": "action_verb", ...}},
  "domains": {{"keyword": "domain_name", ...}}
}}

"actions" maps domain words to cognitive verbs (analyze, detect, query, track, etc.)
"domains" maps domain words to the domain name "{name}"
Include 10-20 entries per map. Return ONLY JSON."""

        result = await self._llm.ask(prompt, max_tokens=512)
        if not result:
            return {"actions": {}, "domains": {}}

        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {"actions": {}, "domains": {}}

    # ── File writers ─────────────────────────────────────────────────

    def _write_manifest(self, cap_dir: Path, name: str, description: str) -> None:
        (cap_dir / "manifest.yaml").write_text(
            f"model_id: {name}\n"
            f"name: {name.replace('-', ' ').title()}\n"
            f"description: {description}\n"
            f"version: 0.1.0\n"
            f"author: Ada (self-generated)\n"
            f'icon: "\\U0001F9E0"\n'
            f"category: learned\n"
            f"public: false\n"
            f"tags:\n"
            f"  - learned\n"
            f"  - {name}\n"
        )

    def _write_config(self, cap_dir: Path, name: str, description: str, design: dict) -> None:
        (cap_dir / "config.yaml").write_text(
            f"name: {name}\n"
            f"version: 0.1.0\n"
            f"description: |\n"
            f"  {description}\n"
            f"  Auto-generated by Ada.\n"
            f"\n"
            f"encoder:\n"
            f"  dimensions: 2000\n"
            f"  seed: 42\n"
            f"\n"
            f"auto_load_concepts: true\n"
            f"\n"
            f"similarity:\n"
            f"  default_filter:\n"
            f"    min_similarity: 0.30\n"
            f"    top_k: 5\n"
        )

    def _write_encoder(
        self, cap_dir: Path, name: str, description: str,
        design: dict, keywords: dict,
    ) -> None:
        """Generate encoder.py from the LLM-designed schema."""
        layers_code = self._generate_layers_code(design)
        action_map = keywords.get("actions", {})
        domain_map = keywords.get("domains", {})

        # Sanitize for Python string embedding
        action_map_str = json.dumps(action_map, indent=4)
        domain_map_str = json.dumps(domain_map, indent=4)

        encoder_py = textwrap.dedent(f'''\
            """
            Encoder for {name} — auto-generated by Ada.

            {description}
            """

            import hashlib
            import re

            from glyphh.core.config import (
                EncoderConfig,
                Layer,
                Role,
                Segment,
            )

            # ---------------------------------------------------------------------------
            # ENCODER_CONFIG
            # ---------------------------------------------------------------------------

            ENCODER_CONFIG = EncoderConfig(
                dimension=2000,
                seed=42,
                apply_weights_during_encoding=False,
                include_temporal=False,
                layers=[
            {layers_code}
                ],
            )

            # ---------------------------------------------------------------------------
            # Intent extraction — keyword-based
            # ---------------------------------------------------------------------------

            _ACTION_MAP = {action_map_str}

            _DOMAIN_MAP = {domain_map_str}


            def _extract(text: str) -> dict:
                words = re.findall(r"[a-z]+", text.lower())
                action = "none"
                domain = "none"
                for w in words:
                    if w in _ACTION_MAP and action == "none":
                        action = _ACTION_MAP[w]
                    if w in _DOMAIN_MAP and domain == "none":
                        domain = _DOMAIN_MAP[w]
                bow = " ".join(words)
                return {{
                    "action": action,
                    "category": domain,
                    "description": bow,
                    "keywords": bow,
                }}


            def encode_query(query: str) -> dict:
                features = _extract(query)
                stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
                return {{
                    "name": f"q_{{stable_id:08d}}",
                    "attributes": features,
                }}


            def entry_to_record(entry: dict) -> dict:
                text = entry.get("text", "")
                attributes = _extract(text)
                return {{
                    "concept_text": entry.get("capability", "unknown"),
                    "attributes": attributes,
                    "metadata": {{
                        "capability": entry.get("capability", "unknown"),
                        "text": text[:200],
                    }},
                }}
        ''')

        (cap_dir / "encoder.py").write_text(encoder_py)

    def _generate_layers_code(self, design: dict) -> str:
        """Convert the LLM's layer design into Python code for EncoderConfig."""
        lines = []
        for layer in design.get("layers", []):
            layer_name = layer.get("name", "unknown")
            weight = layer.get("weight", 0.25)
            roles = layer.get("roles", [])

            role_lines = []
            for role in roles:
                rname = role.get("name", "unknown")
                rtype = role.get("type", "bow")
                if rtype == "lexicon":
                    values = role.get("values", ["none"])
                    vals_str = ", ".join(f'"{v}"' for v in values)
                    role_lines.append(
                        f"                        Role(\n"
                        f"                            name=\"{rname}\",\n"
                        f"                            similarity_weight=1.0,\n"
                        f"                            lexicons=[{vals_str}],\n"
                        f"                        ),"
                    )
                else:
                    role_lines.append(
                        f"                        Role(\n"
                        f"                            name=\"{rname}\",\n"
                        f"                            similarity_weight=0.7,\n"
                        f"                            text_encoding=\"bag_of_words\",\n"
                        f"                        ),"
                    )

            roles_str = "\n".join(role_lines)
            lines.append(
                f"        Layer(\n"
                f"            name=\"{layer_name}\",\n"
                f"            similarity_weight={weight},\n"
                f"            segments=[\n"
                f"                Segment(\n"
                f"                    name=\"main\",\n"
                f"                    roles=[\n"
                f"{roles_str}\n"
                f"                    ],\n"
                f"                ),\n"
                f"            ],\n"
                f"        ),"
            )

        return "\n".join(lines)

    def _write_exemplars(self, cap_dir: Path, exemplars: list[dict]) -> None:
        with open(cap_dir / "data" / "exemplars.jsonl", "w") as f:
            for ex in exemplars:
                f.write(json.dumps(ex) + "\n")
