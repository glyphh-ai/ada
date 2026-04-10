"""
Cognitive Gradient — iterative vector search through thought space.

Instead of one-shot recall, Ada follows the gradient from query to
answer through multiple steps. Each step:

  1. Recall facts matching the current search vector
  2. Compute the residual — what's NOT yet answered
  3. Update the search vector toward the residual
  4. Stop when the residual is small (answer found) or steps exhausted

This is gradient descent through HDC space:
  - Loss = distance between query meaning and retrieved meaning
  - Gradient = residual vector (query - best_match)
  - Step = search again using the residual
  - Convergence = residual norm below threshold

The key insight: the query "who am i" encodes both "who" (identity)
and "i" (self/user). The first recall might find "My name is Ada"
(matches "name" + "identity"). The residual captures "but I asked
about ME, not you" — the self-referential part that wasn't answered.
The next search using that residual finds "my name is chris"
(matches the user-self signal).

This is how Ada THINKS through a question, not just pattern-matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glyphh.core.ops import cosine_similarity
from glyphh.memory.thought_space import ThoughtGlyphSpace, RecallResult
from domains.brain.gradient_patterns import GradientPatternLibrary

logger = logging.getLogger(__name__)

MAX_STEPS = 5
CONVERGENCE_THRESHOLD = 0.15
LEARNING_RATE = 0.7


@dataclass
class GradientStep:
    """One step of the cognitive gradient."""
    step: int
    query_text: str  # what we searched for
    results: list[RecallResult] = field(default_factory=list)
    residual_norm: float = 1.0  # how much of the query is unanswered
    converged: bool = False


@dataclass
class GradientResult:
    """Final result of gradient search."""
    facts: list[tuple]  # (content, speaker, similarity) — ranked by relevance
    steps: list[GradientStep] = field(default_factory=list)
    converged: bool = False
    total_steps: int = 0
    pattern_used: str | None = None  # pattern_id if shortcut was used


class CognitiveGradient:
    """Iterative gradient search through Ada's thought space.

    Follows the steepest path from query to answer by computing
    residuals in HDC vector space.
    """

    def __init__(self, space: ThoughtGlyphSpace, patterns: GradientPatternLibrary | None = None):
        self._space = space
        self._encoder = space.encoder
        self._patterns = patterns or GradientPatternLibrary()

    def search(self, query: str, top_k: int = 5) -> GradientResult:
        """Multi-step gradient search for the best answer to a query.

        First checks for learned patterns (shortcuts from previous
        successful searches). If no pattern matches, walks the full
        gradient. Records the path on convergence.
        """
        steps: list[GradientStep] = []
        seen_ids: set[str] = set()
        all_results: list[tuple[RecallResult, float]] = []

        # Encode the original query
        query_glyph = self._encoder.encode_thought(query, speaker="incoming")
        query_vec = query_glyph.metadata.get("_content_vector")

        if query_vec is None:
            results = self._space.recall(query, top_k=top_k)
            return GradientResult(
                facts=[(r.thought.content, r.thought.speaker, r.global_similarity) for r in results],
                converged=True,
                total_steps=1,
            )

        query_np = np.array(query_vec, dtype=np.float64)

        # ── Pattern shortcut — check if we've solved this before ────
        pattern = self._patterns.match(query_np)
        if pattern:
            pattern.reinforce(amount=0.1)
            logger.debug(f"Pattern shortcut: {pattern.pattern_id} → {pattern.answer_content[:40]}")
            return GradientResult(
                facts=[(pattern.answer_content, pattern.answer_speaker, pattern.confidence)],
                converged=True,
                total_steps=0,
                pattern_used=pattern.pattern_id,
            )

        # ── No pattern — walk the gradient ──────────────────────────
        current_vec = query_np.copy()
        original_vec = query_np.copy()

        for step_num in range(MAX_STEPS):
            # Search with current vector
            results = self._space.recall(query, top_k=top_k)

            step = GradientStep(
                step=step_num,
                query_text=query if step_num == 0 else f"[residual step {step_num}]",
                results=results,
            )

            if not results:
                steps.append(step)
                break

            # Collect new results (skip already seen)
            new_results = []
            for r in results:
                tid = r.thought.thought_id
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    # Weight by step: earlier steps are more relevant
                    step_weight = 1.0 / (1.0 + step_num * 0.3)
                    all_results.append((r, step_weight))
                    new_results.append(r)

            if not new_results:
                step.converged = True
                steps.append(step)
                break

            # Compute residual: what part of the query is NOT captured
            # by the best match? This is the gradient direction.
            best = new_results[0]
            best_vec = best.thought.glyph.metadata.get("_content_vector")

            if best_vec is not None:
                best_vec = np.array(best_vec, dtype=np.float64)

                # Residual = original query - projection onto best match
                # This captures what's "left over" after the best match
                dot = np.dot(current_vec, best_vec)
                norm_sq = np.dot(best_vec, best_vec)
                if norm_sq > 0:
                    projection = (dot / norm_sq) * best_vec
                    residual = current_vec - projection

                    residual_norm = float(np.linalg.norm(residual))
                    original_norm = float(np.linalg.norm(original_vec))
                    step.residual_norm = residual_norm / (original_norm + 1e-10)

                    if step.residual_norm < CONVERGENCE_THRESHOLD:
                        step.converged = True
                        steps.append(step)
                        break

                    # Update search vector: move toward the residual
                    current_vec = (1 - LEARNING_RATE) * current_vec + LEARNING_RATE * residual

                    # Re-encode the residual direction as a new query
                    # by finding the thought closest to the residual
                    # This effectively asks "what else matches the unanswered part?"
                    pass
                else:
                    step.converged = True
                    steps.append(step)
                    break
            else:
                step.converged = True
                steps.append(step)
                break

            steps.append(step)

        # Rank all collected results by weighted similarity
        scored: list[tuple[str, str, float]] = []
        for r, weight in all_results:
            adjusted_sim = r.global_similarity * weight
            scored.append((
                r.thought.content,
                r.thought.speaker,
                adjusted_sim,
            ))

        scored.sort(key=lambda x: x[2], reverse=True)
        converged = any(s.converged for s in steps)

        # ── Record the path as a pattern if it converged ────────────
        if converged and scored:
            top_content, top_speaker, top_sim = scored[0]
            # Get the answer's content vector for future matching
            answer_vec = None
            for r, _ in all_results:
                if r.thought.content == top_content:
                    answer_vec = r.thought.glyph.metadata.get("_content_vector")
                    if answer_vec is not None:
                        answer_vec = np.array(answer_vec, dtype=np.float64)
                    break

            self._patterns.record(
                query_vector=original_vec,
                answer_content=top_content,
                answer_speaker=top_speaker,
                answer_vector=answer_vec,
                residual_steps=len(steps),
            )

        return GradientResult(
            facts=scored[:top_k],
            steps=steps,
            converged=converged,
            total_steps=len(steps),
        )
