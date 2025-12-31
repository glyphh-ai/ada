from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, EmailStr, model_validator
from pydantic.config import ConfigDict


DurationLiteral = Literal[
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
]


class ModelCreate(BaseModel):
    name: str
    description: str | None = None
    roles_config: Dict[str, Any]
    version: int = 1
    data_mode: str | None = None  # "sample" | "live"
    status: str | None = None
    vector_dim: int | None = Field(default=None, ge=1)
    encoder_seed: int | None = None
    space_version: int | None = None


class ModelRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    version: int
    status: str = "draft"
    data_mode: str = "live"
    org_id: str | None = None
    created_by: str | None = None
    created_at: dt.datetime
    roles_config: Dict[str, Any]
    nl_config: Dict[str, Any] | None = None
    linguistic_config: Dict[str, Any] | None = None
    glyph_count: int | None = 0
    vector_dim: int
    encoder_seed: int
    space_id: str
    space_version: int

    class Config:
        from_attributes = True


class ModelList(BaseModel):
    items: list[ModelRead]
    total: int
    page: int
    page_size: int


class GlyphSummary(BaseModel):
    name: str
    node_type: str
    semantic: Dict[str, Any]
    model_id: str


class QueryRequest(BaseModel):
    model_id: str
    text: str | None = None
    glyph_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResult(BaseModel):
    matches: List[GlyphSummary]
    total: int | None = None


class ConceptInput(BaseModel):
    name: str
    attributes: Dict[str, Any]
    node_type: str = "concept"
    taxonomy: List[str] | None = None


class IngestRequest(BaseModel):
    model_id: str
    concepts: List[ConceptInput] = []
    clear_existing: bool = False


class ModelIngestRequest(BaseModel):
    concepts: List[ConceptInput] = []
    clear_existing: bool = False
    force_refresh: bool = False


class ModelSamplePayload(BaseModel):
    concepts: List[ConceptInput]


class ModelSampleResponse(BaseModel):
    model_id: str
    concepts: List[ConceptInput]
    version: int
    updated_at: dt.datetime


class ModelRefreshResponse(BaseModel):
    model_id: str
    glyph_count: int
    status: str


class ModelTestCase(BaseModel):
    name: str
    kind: str  # infer_attribute | find_by_property | find_by_properties | nl_query | similarity_bench
    glyph: str | None = None
    role: str | None = None
    value: str | None = None
    expected: str | None = None
    min_score: float | None = None
    constraints: List[tuple[str, str]] | None = None
    expect_first: str | None = None
    text: str | None = None
    expect: dict | None = None
    pairs: List[tuple[str, str]] | None = None
    groups: List[dict] | None = None
    min_gap: float | None = None
    min_top1: float | None = None
    min_top3: float | None = None


class ModelTestsPayload(BaseModel):
    tests: List[ModelTestCase]


class ModelTestsResponse(BaseModel):
    model_id: str
    tests: List[ModelTestCase]
    version: int
    updated_at: dt.datetime


class ModelTestsRunResponse(BaseModel):
    results: List[dict]


class SimilarityPair(BaseModel):
    left: str
    right: str


class SimilarityGroupConfig(BaseModel):
    name: str
    name_prefix: str | None = None
    taxonomy_includes: str | None = None
    semantic_equals: Dict[str, Any] | None = None
    label_role: str | None = None
    order_role: str | None = None
    expected_neighbor_roles: List[str] | None = None
    top_k: int = Field(default=3, ge=1, le=25)
    far_offset: int | None = Field(default=None, ge=1)


class SimilarityReportRequest(BaseModel):
    pairs: List[SimilarityPair] | None = None
    groups: List[SimilarityGroupConfig]
    enforce_single_space: bool = True


class SimilarityReportResponse(BaseModel):
    report: str
    pairs: List[dict]
    groups: List[dict]
    space: dict | None = None


class ClarificationEntry(BaseModel):
    question: str


class ExecutionPayload(BaseModel):
    ir: Dict[str, Any]
    result: Dict[str, Any]
    evidence: Dict[str, Any]
    explain: Dict[str, Any]
    errors: List[str] = []


class NLQueryPayload(BaseModel):
    text: str


class NLQueryResponse(BaseModel):
    query: str
    text: str | None = None
    execution: ExecutionPayload | None = None
    needs_clarification: list[ClarificationEntry] | None = None


class NLChatPayload(BaseModel):
    text: str
    temperature: float | None = 0.2
    previous_response_id: str | None = None


class GlyphEdgeSummary(BaseModel):
    matched_glyph: str | None = None
    primary_edge: dict | None = None
    primary_target: str | None = None
    secondary_edge: dict | None = None
    secondary_target: str | None = None
    semantic_edges: list[dict] = []
    neural_edges: list[dict] = []
    hierarchy: list[str] | None = None
    sequence: list[str] | None = None


class NLChatResponse(BaseModel):
    text: str
    glyph_edges: GlyphEdgeSummary | None = None
    message_id: str | None = None
    conversation_id: str | None = None


class WebSocketListenerPayload(BaseModel):
    name: str
    description: str | None = None
    url: str
    method: str = "POST"
    headers: Dict[str, str] | None = None
    payload_template: Dict[str, Any] | None = None
    enabled: bool = False
    map_id: str | None = None
    throttle: int = 1
    history_enabled: bool = False


class WebSocketListenerRead(WebSocketListenerPayload):
    id: str
    org_id: str
    created_at: dt.datetime
    updated_at: dt.datetime


class ListenerAssignment(BaseModel):
    model_id: str
    concept_type: str
    mapper: Dict[str, Any] | None = None


class ModelListenerAssignment(BaseModel):
    listener_id: str
    assignments: List[ListenerAssignment]


class ModelListenerResponse(BaseModel):
    listener_id: str
    assignments: List[ListenerAssignment]


class ListenerLogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class ListenerLogs(BaseModel):
    listener_id: str
    entries: List[ListenerLogEntry]


class ListenerControl(BaseModel):
    action: Literal["start", "stop"]


class SampleImportResult(BaseModel):
    category: str
    file: str
    status: Literal["queued", "running", "success", "missing", "error"]
    detail: str | None = None


class SampleUploadEntry(BaseModel):
    file: str
    content: Any


class SampleUploadBundle(BaseModel):
    model: Dict[str, Any] | None = None
    roles_configs: List[SampleUploadEntry] = []
    tests: List[SampleUploadEntry] = []
    encoders: List[SampleUploadEntry] = []
    runtime: List[SampleUploadEntry] = []
    concepts: List[SampleUploadEntry] = []
    profiles: List[SampleUploadEntry] = []
    maps: List[SampleUploadEntry] = []
    listeners: List[SampleUploadEntry] = []


class TrendEntry(BaseModel):
    timestamp: dt.datetime
    role: str
    layer: int
    segment_index: int
    similarity: float | None = None
    delta_norm: float | None = None
    source: str
    attribute_value: Any | None = None


class TrendChartAxis(BaseModel):
    role: str
    color: str
    prediction_color: str | None = Field(None, alias="predictionColor")
    prediction_opacity: float | None = Field(None, alias="predictionOpacity")
    snapshot_color: str | None = Field(None, alias="snapshotColor")
    similarity_threshold: float | None = Field(None, alias="similarityThreshold")
    delta_norm_threshold: float | None = Field(None, alias="deltaNormThreshold")

    class Config:
        validate_by_name = True


class TrendChartPayload(BaseModel):
    name: str
    description: str | None = None
    chart_type: Literal["line", "bar", "area"]
    x_axis: str
    y_axes: List[TrendChartAxis]
    width: int = 360
    height: int = 180
    trail_length: int = 60
    prediction_steps: int = 3
    predictions_enabled: bool = True
    timeline_marker_color: str = "#ef4444"
    snapshot_color: str | None = None


class TrendChartRead(TrendChartPayload):
    id: str
    created_at: dt.datetime
    updated_at: dt.datetime


class ChartSlot(BaseModel):
    index: int
    timestamp: dt.datetime
    type: Literal["history", "horizon", "future"]
    color: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChartDataPoint(BaseModel):
    slot_index: int
    value: float | None = None
    similarity: float | None = None
    delta_norm: float | None = None
    source: str | None = None


class ChartDataRow(BaseModel):
    label: str
    color: str | None = None
    opacity: float | None = None
    type: Literal["line", "bar", "area"] = "line"
    points: List[ChartDataPoint]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChartAggregatedPrediction(BaseModel):
    similarity: float | None = None
    delta_norm: float | None = None
    roles: int


class SnapshotCaptureRequest(BaseModel):
    role: str
    label: str | None = None
    color: str | None = None
    layer: int | None = None
    segment_index: int | None = Field(None, alias="segmentIndex")
    duration: DurationLiteral | None = None
    alpha: float | None = None


class SnapshotCaptureResult(BaseModel):
    role: str
    layer: int
    segment_index: int
    timestamp: dt.datetime
    value: float | None = None
    label: str | None = None
    color: str | None = None
    vector: list[int]


class SnapshotCaptureResponse(BaseModel):
    snapshots: list[SnapshotCaptureResult]


class ChartData(BaseModel):
    slots: List[ChartSlot]
    rows: List[ChartDataRow]
    horizon_slot_index: int
    average_confidence: float | None = None
    duration: DurationLiteral
    aggregated_prediction: ChartAggregatedPrediction | None = None


class HistoricalTrendAxis(BaseModel):
    role: str
    color: str
    layer: int | None = None
    segment_index: int | None = None


class HistoricalTrendView(BaseModel):
    axes: List[HistoricalTrendAxis]
    trail_length: int = 60
    start: dt.datetime | None = None
    end: dt.datetime | None = None


class HistoricalTrendChartRequest(BaseModel):
    model_id: str
    duration: DurationLiteral = "minute"
    trail_length: int = 60
    axes: List[HistoricalTrendAxis]
    start: dt.datetime | None = None
    end: dt.datetime | None = None


class DashboardChartLayoutItem(BaseModel):
    id: str
    trend_id: str | None = None
    chart_id: str | None = None
    trend_name: str | None = None
    chart_name: str
    duration: DurationLiteral
    width: int
    height: int
    order: int
    chart_config: TrendChartPayload
    source: Literal["trend", "historical"] = "trend"
    historical_view: HistoricalTrendView | None = None


class DashboardChartConfigPayload(BaseModel):
    items: List[DashboardChartLayoutItem] = Field(default_factory=list)


class DashboardChartConfigRead(DashboardChartConfigPayload):
    pass


class ChartPreviewRequest(BaseModel):
    chart_config: TrendChartPayload
    model_id: str
    duration: Literal["second", "minute", "hour", "day", "week", "month", "year"] = "minute"
    alpha: float = 1.0


class TrendResponse(BaseModel):
    model_id: str
    entries: list[TrendEntry]
    charts: list[TrendChartRead] | None = None
    trend_definition: TrendDefinitionRead | None = None


class TrendDefinitionPayload(BaseModel):
    name: str
    description: str | None = None
    model_id: str
    roles: List[str]
    duration: Literal["second", "minute", "hour", "day", "week", "month", "year"]
    alpha: float = 1.0
    enabled: bool = True


class TrendDefinitionRead(TrendDefinitionPayload):
    id: str
    created_at: dt.datetime
    updated_at: dt.datetime
    charts: list[TrendChartRead] = []


class TrendListResponse(BaseModel):
    items: list[TrendDefinitionRead]


class PredictionRequest(BaseModel):
    model_id: str
    role: str
    layer: int
    segment_index: int
    duration: Literal["second", "minute", "hour", "day", "week", "month", "year"] = "minute"
    alpha: float | None = None
    steps: int = 1
    history_length: int = 60
    similarity_threshold: float | None = None
    delta_norm_threshold: float | None = None


class PredictionStep(BaseModel):
    step: int
    direction: Literal["history", "future"]
    predicted_timestamp: dt.datetime
    predicted_value: float | None
    similarity_to_previous: float | None
    delta_norm: float
    detail: str | None = None


class PredictionResponse(BaseModel):
    model_id: str
    role: str
    layer: int
    segment_index: int
    predicted_vector: list[int]
    current_vector: list[int] | None = None
    steps: list[PredictionStep]
    similarity_to_previous: float | None
    delta_norm: float
    factor: float


class SampleImportResponse(BaseModel):
    results: List[SampleImportResult]


class Health(BaseModel):
    status: str = "ok"


class FolderBase(BaseModel):
    name: str
    description: str | None = None


class FolderCreate(FolderBase):
    pass


class FolderUpdate(FolderBase):
    pass


class FolderRead(FolderBase):
    id: str
    org_id: str | None = None
    created_by: str | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


class FolderAssets(BaseModel):
    models: int = 0
    encoders: int = 0
    listeners: int = 0
    profiles: int = 0
    maps: int = 0


class FolderSummary(FolderRead):
    assets: FolderAssets = FolderAssets()
    is_root: bool = False


class FolderAssetItem(BaseModel):
    id: str
    name: str


class FolderAssetsDetail(BaseModel):
    models: list[FolderAssetItem] = []
    encoders: list[FolderAssetItem] = []
    listeners: list[FolderAssetItem] = []
    profiles: list[FolderAssetItem] = []
    maps: list[FolderAssetItem] = []


class FolderAssignRequest(BaseModel):
    item_type: Literal["models", "encoders", "listeners", "profiles", "maps"]
    item_id: str


class FieldDefinition(BaseModel):
    name: str
    type: str
    description: str | None = None
    format: str | None = None
    required: bool = False
    children: List["FieldDefinition"] | None = None


FieldDefinition.update_forward_refs()


class ProfileBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str | None = None
    schema_: list[FieldDefinition] | None = Field(default=None, alias="schema")
    fields: list[FieldDefinition] | None = None

    @model_validator(mode="before")
    def ensure_schema(cls, values):
        if values.get("schema") is None and values.get("fields") is not None:
            values["schema"] = values["fields"]
        return values


class ProfileCreate(ProfileBase):
    @model_validator(mode="after")
    def ensure_schema_exists(cls, model):
        if model.schema_ is None:
            raise ValueError("schema is required")
        return model


class ProfileUpdate(ProfileBase):
    @model_validator(mode="after")
    def ensure_schema_exists(cls, model):
        if model.schema_ is None:
            raise ValueError("schema is required")
        return model


class ProfileRead(ProfileBase):
    id: str
    org_id: str | None = None
    created_by: str | None = None
    generated_from_model_id: str | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


class TransformationField(BaseModel):
    name: str | None = None
    type: str


class Transformation(BaseModel):
    id: str
    name: str | None = None
    inputs: list[TransformationField]
    outputs: list[TransformationField]
    script: str | None = None
    y: float


class MapFieldMapping(BaseModel):
    source_path: str
    target_path: str


class MapCreate(BaseModel):
    name: str
    description: str | None = None
    source_profile_id: str
    target_profile_id: str
    field_mappings: list[MapFieldMapping] = []
    transformations: list[Transformation] = []


class MapUpdate(BaseModel):
    name: str
    description: str | None = None
    source_profile_id: str
    target_profile_id: str
    field_mappings: list[MapFieldMapping] = []
    transformations: list[Transformation] = []


class MapRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    source_profile_id: str
    target_profile_id: str
    field_mappings: list[MapFieldMapping]
    transformations: list[Transformation]
    org_id: str | None = None
    created_by: str | None = None
    created_at: dt.datetime
    updated_at: dt.datetime

    class Config:
        from_attributes = True


class MapListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[MapRead]


class MapTargetProfileRequest(BaseModel):
    model_id: str
    map_id: str | None = None


class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    org_id: str
    user_id: str
    email_verification_token: str


class VerifyEmailRequest(BaseModel):
    token: str


class VerifyEmailResponse(BaseModel):
    verified: bool


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    org_id: str | None = None
    role: str | None = None
    first_name: str | None = None
    last_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    token: str
    user: UserPublic


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


class UserUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: EmailStr | None = None
    password: str | None = None
    current_password: str | None = None


class NLConfigPayload(BaseModel):
    config: Dict[str, Any]


class NLConfigResponse(BaseModel):
    model_id: str
    config: Dict[str, Any]
    version: int
    updated_at: dt.datetime


class LinguisticConfigPayload(BaseModel):
    config: Dict[str, Any]


class LinguisticConfigResponse(BaseModel):
    model_id: str
    config: Dict[str, Any]
    version: int
    updated_at: dt.datetime


class EncoderCreate(BaseModel):
    name: str
    domain: str
    description: str | None = None
    type: str | None = "nl"
    status: str | None = None
    version: int | None = 1
    is_public: bool | None = False
    config: Dict[str, Any] | None = None


class EncoderUpdate(BaseModel):
    name: str | None = None
    domain: str | None = None
    description: str | None = None
    type: str | None = None
    status: str | None = None
    is_public: bool | None = None
    config: Dict[str, Any] | None = None


class EncoderRead(BaseModel):
    id: str
    name: str
    domain: str
    description: str | None = None
    type: str = "nl"
    status: str = "draft"
    version: int = 1
    is_public: bool
    org_id: str | None = None
    created_by: str | None = None
    created_at: dt.datetime
    config: Dict[str, Any] | None = None

    class Config:
        from_attributes = True


class EncoderList(BaseModel):
    items: list[EncoderRead]
    total: int
    page: int
    page_size: int


class DataModeUpdate(BaseModel):
    data_mode: str
