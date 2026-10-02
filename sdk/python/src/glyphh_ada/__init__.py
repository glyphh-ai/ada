"""Ada for Python.

An agent builds a model: its types, its records, its rules. This package is
what the software that then runs it calls. Every answer is typed, the same
input always gives the same answer, and no language model is in the path.
"""

from .client import DEFAULT_URL, Ada, AdaError, AdaModel, Transport
from .types import (
    Answer, Args, Change, Check, Contract, Device, Edges, Fact, Facts, FactTree, GqlResult, GraphNode, History, Model, Observation,
    Prediction, Procedure, Reason, Receipt, Recorded, Records, Relation, Scale, Situation, Store, StoredRecord, Trend, Weights, When,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_URL", "Ada", "AdaError", "AdaModel", "Transport", "__version__",
    "Answer", "Args", "Change", "Check", "Contract", "Device", "Edges", "Fact", "Facts", "FactTree", "GqlResult", "GraphNode", "History",
    "Model", "Observation", "Prediction", "Procedure", "Reason", "Receipt", "Recorded", "Records", "Relation", "Scale", "Situation",
    "Store", "StoredRecord", "Trend", "Weights", "When",
]
