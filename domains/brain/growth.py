"""
Growth engine — extends the dream loop to mine patterns and mint capabilities.

The dream loop already:
- Runs dual REM + Slow-Wave background reasoning
- Crystallizes compound primitives from stable patterns

The growth engine adds:
- Observation ingestion (every think() call gets analyzed)
- Pattern mining (clusters of similar requests)
- Route reinforcement (Hebbian — strengthen working routes)
- Capability minting (new vector spaces from stable clusters)

The growth engine is NOT a separate loop. It plugs into the existing
dream loop by providing callbacks that run during the deep (Slow-Wave)
cycle.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from domains.brain.observation import Observation, ObservationLog

logger = logging.getLogger(__name__)

# Minimum observations in a cluster before it's a minting candidate
MIN_CLUSTER_SIZE = 20

# Minimum stability (fraction of total cluster observations) to mint
MIN_STABILITY = 0.7


@dataclass
class PatternCluster:
    """A group of similar observations that might become a capability."""
    name: Optional[str] = None  # Set by Haiku during crystallization
    observations: list[Observation] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    stability: float = 0.0  # 0.0 = volatile, 1.0 = rock solid
    minted: bool = False  # True once a capability has been created

    @property
    def size(self) -> int:
        return len(self.observations)


@dataclass
class GrowthStats:
    """Track the growth engine's activity."""
    observations_processed: int = 0
    patterns_detected: int = 0
    capabilities_minted: int = 0
    routes_reinforced: int = 0


class GrowthEngine:
    """Mines patterns from observations and mints new capabilities.

    Designed to be called from the dream loop's deep cycle, not as
    a standalone process.
    """

    def __init__(self, observation_log: ObservationLog, llm=None):
        self._observations = observation_log
        self._llm = llm
        self._clusters: dict[str, PatternCluster] = {}
        self._route_hits: Counter = Counter()  # capability → hit count
        self.stats = GrowthStats()

    async def analyze(self) -> list[PatternCluster]:
        """Run one analysis cycle over recent observations.

        Called by the dream loop during the deep (Slow-Wave) phase.
        Returns clusters that are ready for minting.
        """
        recent = self._observations.drain(limit=200)
        if not recent:
            return []

        self.stats.observations_processed += len(recent)

        # Track which capabilities are being hit (Hebbian reinforcement)
        for obs in recent:
            if obs.capability:
                self._route_hits[obs.capability] += 1
                self.stats.routes_reinforced += 1

        # Find observations that fell back to LLM (unrouted)
        unrouted = [o for o in recent if o.llm_fallback or o.capability is None]
        if not unrouted:
            return []

        # Simple clustering: group by the LLM-assigned capability or by None
        groups: dict[Optional[str], list[Observation]] = defaultdict(list)
        for obs in unrouted:
            groups[obs.capability].append(obs)

        ready = []
        for key, obs_list in groups.items():
            cluster_key = key or "unknown"
            if cluster_key not in self._clusters:
                self._clusters[cluster_key] = PatternCluster()

            cluster = self._clusters[cluster_key]
            cluster.observations.extend(obs_list)
            cluster.last_seen = time.time()

            if cluster.size >= MIN_CLUSTER_SIZE:
                # Calculate stability: ratio of consistent routing
                if key:
                    consistent = sum(1 for o in cluster.observations if o.capability == key)
                    cluster.stability = consistent / cluster.size
                else:
                    cluster.stability = 0.0

                if cluster.stability >= MIN_STABILITY and not cluster.minted:
                    self.stats.patterns_detected += 1
                    ready.append(cluster)

        return ready

    async def mint_capability(self, cluster: PatternCluster, model_manager=None, session_factory=None) -> Optional[dict]:
        """Attempt to mint a new capability from a stable cluster.

        Uses the CapabilityBuilder to create a full capability with
        encoder, exemplars, and intent extraction — then loads it
        into the runtime.

        Returns a dict describing the new capability, or None if minting fails.
        """
        if not self._llm or not self._llm.available:
            logger.info("Cannot mint: LLM offline")
            return None

        # Build a description from the cluster's observations
        sample_inputs = [o.input for o in cluster.observations[:20]]
        sample_text = "\n".join(f"- {inp}" for inp in sample_inputs)

        # Ask Haiku to name and describe
        name_prompt = (
            f"These are queries that Ada received but couldn't route to an existing capability:\n\n"
            f"{sample_text}\n\n"
            f"What would you name this capability? Reply with a single lowercase word "
            f"(like 'pricing', 'scheduling', 'onboarding'). No explanation."
        )
        name = await self._llm.ask(name_prompt, max_tokens=16)
        if not name:
            return None
        name = name.strip().lower().replace(" ", "_")

        # Generate seed exemplars
        exemplars = await self._llm.generate_exemplars(
            f"Queries about {name}: {sample_text[:500]}",
            count=15,
        )

        # Get a description
        desc_prompt = (
            f"In one sentence, describe what a '{name}' capability would do. "
            f"Based on these sample queries:\n{sample_text[:300]}"
        )
        description = await self._llm.ask(desc_prompt, max_tokens=64)

        cluster.name = name
        cluster.minted = True
        self.stats.capabilities_minted += 1

        desc = (description or "").strip()

        # Use the CapabilityBuilder to create the full capability
        if model_manager and session_factory:
            try:
                from domains.brain.skills.capability_builder import CapabilityBuilder
                builder = CapabilityBuilder(self._llm, model_manager, session_factory)
                build_result = await builder.build(
                    name=name,
                    description=desc,
                    seed_queries=[o.input for o in cluster.observations[:10]],
                    exemplar_count=max(20, cluster.size),
                )
                if build_result.success:
                    logger.info(
                        f"Dream loop built capability '{name}': "
                        f"{build_result.exemplar_count} exemplars"
                    )
                else:
                    logger.warning(f"Dream loop build failed for '{name}': {build_result.error}")
            except Exception as e:
                logger.warning(f"Dream loop build error for '{name}': {e}")

        result = {
            "name": name,
            "description": desc,
            "exemplars": exemplars,
            "observation_count": cluster.size,
            "stability": cluster.stability,
        }

        logger.info(
            f"Minted new capability: {name} "
            f"({cluster.size} observations, stability={cluster.stability:.2f})"
        )

        return result

    @property
    def route_hits(self) -> dict[str, int]:
        """Which capabilities are being used most."""
        return dict(self._route_hits.most_common())
