"""Raw-text enrichment: the ingest front half (raw -> universal-schema slots)."""
from .enricher import Enricher, HeuristicEnricher, StructuredFact
from .pipe import enrich_to_concept, enrich_and_ingest

__all__ = ["Enricher", "HeuristicEnricher", "StructuredFact",
           "enrich_to_concept", "enrich_and_ingest"]
