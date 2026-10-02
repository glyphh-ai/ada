"""What Ada takes and returns, as typed dictionaries. Every answer is the JSON
the server sent, unchanged: these say what is in it. They follow contract
version 2 (contract/contract.json in the repository)."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Literal, NotRequired, TypedDict
else:  # pragma: no cover
    from typing import Literal

    from typing_extensions import NotRequired, TypedDict

Scalar = Union[str, int, float, bool]
#: Data sent to a model. A flat model takes any JSON object; a typed one takes its spec's shape: layer > segment > role.
Situation = Dict[str, Any]
#: A time: epoch seconds, or an ISO 8601 string.
When = Union[int, float, str]
#: How much each key decides. A number weighs a key; an object weighs inside it, its own weight under "*".
Weights = Dict[str, Any]
#: A statement's $parameters, by name.
Args = Dict[str, Any]
Store = Literal["wins", "vetoes"]
Reason = Literal["reflex", "insufficient", "unseen", "vetoed", "uncalibrated", "mismatch", "low_confidence"]
Match = Literal["same", "lexical", "partial", "near", "different"]


class Scale(TypedDict):
    """What `act` takes for a model."""

    similarity: float
    confidence: float
    records: int
    exact: bool
    must_match: List[str]
    unseen: bool


class Receipt(TypedDict):
    """One situation on record with one outcome, however many times it was recorded."""

    key: str
    outcome: str
    score: float
    sim: float
    count: int
    first: float
    latest: float
    id: str
    hash: str
    author: str


class Answer(TypedDict):
    """What a query returns. Branch on `act`; read `reason` when it is false."""

    contract: int
    model_id: str
    key: str
    act: bool
    reason: Reason
    top: Optional[str]
    sufficient: bool
    exact: bool
    unseen: List[str]
    confidence: float
    support: float
    count: int
    outcomes: Dict[str, float]
    receipts: List[Receipt]
    veto_match: float
    lifted: bool
    calibrated: bool
    scale: Scale


class Fact(TypedDict):
    key: str
    weight: float
    score: float
    share: float
    sim: float
    status: Literal["both", "query_only", "record_only"]
    match: NotRequired[Match]
    query: NotRequired[Any]
    record: NotRequired[Any]
    children: NotRequired[List["Fact"]]


class FactTree(TypedDict):
    id: str
    key: str
    outcome: Optional[str]
    probability: float
    score: float
    sim: float
    count: int
    first: float
    latest: float
    author: str
    ts: float
    hash: str
    tree: Fact


class Facts(TypedDict):
    model_id: str
    wins: List[FactTree]
    vetoes: List[FactTree]


class Check(TypedDict):
    model_id: str
    outcome: str
    win_match: float
    veto_match: float


class Recorded(TypedDict):
    model_id: str
    store: Store
    id: str
    key: str
    ts: float
    version: Optional[int]
    size: int


class StoredRecord(TypedDict):
    seq: int
    id: str
    key: str
    store: Store
    situation: Situation
    outcome: str
    author: str
    ts: float


class Records(TypedDict):
    model_id: str
    total: int
    records: List[StoredRecord]


class CacheStats(TypedDict):
    hits: int
    misses: int
    invalidations: int
    evictions: int
    hit_rate: float
    entries: int


class Model(TypedDict):
    model_id: str
    name: str
    storage: Literal["cloud", "local"]
    device_id: Optional[str]
    weights: Optional[Weights]
    tau: Optional[float]
    calibrated: bool
    scale: Scale
    spec: Optional[Dict[str, Any]]
    edges: Dict[str, List[float]]
    data_version: int
    created_at: str
    updated_at: str
    wins: NotRequired[int]
    vetoes: NotRequired[int]
    things: NotRequired[int]
    cache: NotRequired[CacheStats]


class Device(TypedDict):
    device_id: str
    owner_user_id: str
    name: str
    created_at: str
    online: bool


class Change(TypedDict("Change", {"from": Any})):
    """One role that differs between two versions of a thing."""

    path: str
    to: Any
    score: float


class Observation(TypedDict):
    id: str
    store: Store
    outcome: str
    author: str
    ts: float
    version: NotRequired[int]
    data: NotRequired[Situation]
    changes: NotRequired[List[Change]]
    change: NotRequired[float]


class History(TypedDict):
    model_id: str
    key: str
    count: int
    first: Optional[float]
    latest: Optional[float]
    observations: List[Observation]


class RoleTrend(TypedDict):
    type: str
    first: Any
    latest: Any
    changes: int
    since: float
    slope_per_day: NotRequired[Optional[float]]
    fit: NotRequired[Optional[float]]
    direction: NotRequired[Literal["up", "down", "flat"]]


class Trend(TypedDict):
    model_id: str
    key: str
    versions: int
    first: float
    latest: float
    drift: float
    pace: float
    period: Optional[int]
    roles: Dict[str, RoleTrend]


class RolePrediction(TypedDict("RolePrediction", {"from": Any})):
    to: Any
    basis: Literal["trend", "own_history", "model", "unchanged"]
    confidence: Optional[float]


class Prediction(TypedDict):
    model_id: str
    key: str
    versions: int
    at: float
    data: Situation
    roles: Dict[str, RolePrediction]
    answer: Answer


class Relation(TypedDict("Relation", {"from": Dict[str, str]})):
    """One relation: the thing that has the ref role, the role's path, and the thing it names."""

    relation: str
    to: Dict[str, Any]
    missing: bool


class GraphNode(TypedDict):
    """A thing: its newest version, as the reader may see it."""

    model_id: str
    key: str
    missing: bool
    id: NotRequired[str]
    store: NotRequired[Store]
    outcome: NotRequired[str]
    ts: NotRequired[float]
    data: NotRequired[Situation]
    hops: NotRequired[int]


class Edges(TypedDict):
    model_id: str
    key: str
    level: str
    neural: List[Dict[str, Any]]
    temporal: List[Dict[str, Any]]
    relations: Dict[str, List[Relation]]


class Procedure(TypedDict):
    name: str
    query: str
    description: str
    params: List[str]
    updated_at: str


class GqlResult(TypedDict, total=False):
    """What one GQL statement returns. `statement` says which, and which of
    the other keys are there: see the gql schema in the contract."""

    statement: Literal["find", "list", "count", "compare", "drift", "introspect", "trend", "predict", "aggregate", "follow", "path"]
    model_id: str
    cached: bool
    procedure: str
    # find
    level: str
    threshold: Optional[float]
    count: int
    matches: List[Dict[str, Any]]
    # list
    records: List[Dict[str, Any]]
    # compare, drift
    score: float
    tree: Fact
    drift: float
    drifted: Optional[bool]
    changes: List[Change]
    # introspect
    by: str
    parts: List[Dict[str, Any]]
    # trend, predict
    key: str
    versions: int
    over: float
    pace: float
    period: Optional[int]
    roles: Dict[str, Any]
    alert: Optional[Dict[str, Any]]
    at: float
    data: Situation
    answer: Answer
    # aggregate
    function: str
    path: Optional[str]
    groups: List[Dict[str, Any]]
    # follow, path
    direction: str
    depth: int
    nodes: List[GraphNode]
    edges: List[Relation]
    truncated: bool
    found: bool
    hops: Optional[int]


class Contract(TypedDict):
    """The JSON Schema of every answer and, with a typed model's id, of its data."""

    contract: int
    reasons: Dict[str, str]
    query: Dict[str, Any]
    facts: Dict[str, Any]
    history: Dict[str, Any]
    trend: Dict[str, Any]
    predict: Dict[str, Any]
    gql: Dict[str, Any]
    edges: Dict[str, Any]
    data: NotRequired[Dict[str, Any]]
