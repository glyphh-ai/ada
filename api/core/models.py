from __future__ import annotations

import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import relationship

from glyphh.config import DEFAULT_VECTOR_DIM
from .vector_space import DEFAULT_ENCODER_SEED, DEFAULT_SPACE_VERSION
from .db import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, default="draft", nullable=False)
    version = Column(Integer, default=1, nullable=False)
    roles_config = Column(JSON, nullable=False)
    vector_dim = Column(Integer, nullable=False, default=DEFAULT_VECTOR_DIM, server_default=str(DEFAULT_VECTOR_DIM))
    encoder_seed = Column(Integer, nullable=False, default=DEFAULT_ENCODER_SEED, server_default=str(DEFAULT_ENCODER_SEED))
    space_id = Column(String(64), nullable=False, index=True)
    space_version = Column(Integer, nullable=False, default=DEFAULT_SPACE_VERSION, server_default=str(DEFAULT_SPACE_VERSION))
    data_mode = Column(String, default="live", nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False, onupdate=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False, onupdate=dt.datetime.utcnow)
    default_encoder_ids = Column(JSON, nullable=False, default=list)

    glyphs = relationship("Glyph", back_populates="model", cascade="all, delete-orphan")
    nl_config = relationship("ModelNLConfig", uselist=False, back_populates="model", cascade="all, delete-orphan")
    linguistic_config = relationship("ModelLinguisticConfig", uselist=False, back_populates="model", cascade="all, delete-orphan")
    runtime_config = relationship("ModelRuntimeConfig", uselist=False, back_populates="model", cascade="all, delete-orphan")
    mcp_config = relationship("ModelMcpConfig", uselist=False, back_populates="model", cascade="all, delete-orphan")
    sample = relationship("ModelSample", uselist=False, back_populates="model", cascade="all, delete-orphan")
    tests = relationship("ModelTests", uselist=False, back_populates="model", cascade="all, delete-orphan")
    staged_concepts = relationship(
        "ModelStagedConcept",
        back_populates="model",
        cascade="all, delete-orphan",
    )
    listeners = relationship("WebSocketListener", secondary="model_websocket_listeners", back_populates="models")


class ModelNLConfig(Base):
    __tablename__ = "model_nl_configs"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, unique=True, index=True)
    config = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="nl_config")


class ModelLinguisticConfig(Base):
    __tablename__ = "model_linguistic_configs"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, unique=True, index=True)
    config = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="linguistic_config")


class ModelRuntimeConfig(Base):
    __tablename__ = "model_runtime_configs"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, unique=True, index=True)
    config = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="runtime_config")


class ModelMcpConfig(Base):
    __tablename__ = "model_mcp_configs"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, unique=True, index=True)
    config = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="mcp_config")

class ModelSample(Base):
    __tablename__ = "model_samples"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, unique=True, index=True)
    data = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="sample")


class ModelTests(Base):
    __tablename__ = "model_tests"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, unique=True, index=True)
    tests = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="tests")


class ModelStagedConcept(Base):
    __tablename__ = "model_staged_concepts"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    primary_id = Column(String, nullable=False)
    observed_at = Column(DateTime, nullable=False)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="staged_concepts")

    __table_args__ = (
        UniqueConstraint("model_id", "primary_id", "observed_at", name="uq_model_staged_concepts_key"),
    )


class Encoder(Base):
    __tablename__ = "encoders"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    type = Column(String, default="nl", nullable=False)  # "nl" | "linguistic" | "text"
    status = Column(String, default="draft", nullable=False)
    version = Column(Integer, default=1, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    config = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    schema = Column(JSON, nullable=False, default=dict)
    generated_from_model_id = Column(String, ForeignKey("models.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False, onupdate=dt.datetime.utcnow)

    generated_from_model = relationship("Model")
    source_maps = relationship("Map", back_populates="source_profile", foreign_keys="Map.source_profile_id")
    target_maps = relationship("Map", back_populates="target_profile", foreign_keys="Map.target_profile_id")


class Map(Base):
    __tablename__ = "maps"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source_profile_id = Column(String, ForeignKey("profiles.id"), nullable=False, index=True)
    target_profile_id = Column(String, ForeignKey("profiles.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False, onupdate=dt.datetime.utcnow)
    transformations = Column(JSON, nullable=False, default=list)

    source_profile = relationship("Profile", foreign_keys=[source_profile_id], back_populates="source_maps")
    target_profile = relationship("Profile", foreign_keys=[target_profile_id], back_populates="target_maps")
    field_mappings = relationship("MapFieldMapping", back_populates="map", cascade="all, delete-orphan")


class MapFieldMapping(Base):
    __tablename__ = "map_field_mappings"

    id = Column(String, primary_key=True)
    map_id = Column(String, ForeignKey("maps.id", ondelete="CASCADE"), nullable=False, index=True)
    source_path = Column(String, nullable=False)
    target_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    map = relationship("Map", back_populates="field_mappings")


class Glyph(Base):
    __tablename__ = "glyphs"

    name = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    node_type = Column(String, default="concept", nullable=False)
    semantic = Column(JSON, default=dict, nullable=False)
    cortex = Column(BYTEA, nullable=False)
    prediction_flag = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="glyphs")
    segments = relationship("Segment", back_populates="glyph", cascade="all, delete-orphan")

    __table_args__ = (
        Index(
            "ix_glyphs_name_observed_at",
            "name",
            func.json_extract_path_text(semantic, "observed_at"),
        ),
    )


class Segment(Base):
    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    glyph_name = Column(String, ForeignKey("glyphs.name"), nullable=False)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    layer = Column(Integer, nullable=False)
    seg_index = Column(Integer, nullable=False)
    vec = Column(BYTEA, nullable=False)

    glyph = relationship("Glyph", back_populates="segments")

    __table_args__ = (
        UniqueConstraint("model_id", "glyph_name", "layer", "seg_index", name="uq_segment"),
    )


class Edge(Base):
    __tablename__ = "edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    source = Column(String, nullable=False)
    target = Column(String, nullable=False)
    type = Column(String, nullable=False)
    weight = Column(Numeric, default=1.0, nullable=False)
    layer = Column(Integer, nullable=True)


class Embedding(Base):
    __tablename__ = "embeddings"

    glyph_name = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    embedding = Column(Vector(DEFAULT_VECTOR_DIM), nullable=False)
    node_type = Column(String, default="concept", nullable=False)
    meta = Column("metadata", JSON, default=dict, nullable=False)

    model = relationship(
        "Model",
    )


class WebSocketListener(Base):
    __tablename__ = "listeners"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    url = Column(String, nullable=False)
    method = Column(String, default="POST", nullable=False)
    headers = Column(JSON, default=dict, nullable=False)
    payload_template = Column(JSON, default=dict, nullable=False)
    enabled = Column(Integer, default=1, nullable=False)
    history_enabled = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)
    map_id = Column(String, ForeignKey("maps.id"), nullable=True, index=True)
    throttle = Column(Integer, default=1, nullable=False)

    models = relationship("Model", secondary="model_websocket_listeners", back_populates="listeners")


class ModelWebSocketListener(Base):
    __tablename__ = "model_websocket_listeners"

    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    listener_id = Column(String, ForeignKey("listeners.id"), nullable=False, index=True)
    concept_type = Column(String, nullable=False, default="concept")
    mapper = Column(JSON, default=dict, nullable=False)
    enabled = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)


class GlyphTrend(Base):
    __tablename__ = "glyph_trends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    role = Column(String, nullable=False, index=True)
    layer = Column(Integer, nullable=False)
    segment_index = Column(Integer, nullable=False)
    vector = Column(BYTEA, nullable=False)
    timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False, index=True)
    source = Column(String, nullable=False, default="trend")
    attribute_value = Column(JSON, nullable=True)
    aligned_to_actual = Column(Boolean, nullable=False, default=False, server_default="false")
    trend_definition_id = Column(String, ForeignKey("trends.id"), nullable=True, index=True)

    trend_definition = relationship("TrendDefinition", backref="glyph_trends")


class GlyphHistoryVector(Base):
    __tablename__ = "glyph_history_vectors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    listener_id = Column(String, nullable=True, index=True)
    role = Column(String, nullable=False, index=True)
    layer = Column(Integer, nullable=False)
    segment_index = Column(Integer, nullable=False)
    vector = Column(Vector(DEFAULT_VECTOR_DIM), nullable=False)
    attribute_value = Column(JSON, nullable=True)
    vector_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False, index=True)

    model = relationship("Model")


class TrendChart(Base):
    __tablename__ = "trend_charts"

    id = Column(String, primary_key=True)
    trend_id = Column(String, ForeignKey("trends.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    chart_type = Column(String, nullable=False)
    x_axis = Column(String, nullable=False)
    y_axes = Column(JSON, nullable=False)
    width = Column(Integer, nullable=False, default=360)
    height = Column(Integer, nullable=False, default=180)
    trail_length = Column(Integer, nullable=False, default=60)
    prediction_steps = Column(Integer, nullable=False, default=3)
    predictions_enabled = Column(Boolean, nullable=False, default=True)
    timeline_marker_color = Column(String, nullable=False, default="#ef4444")
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False
    )

    trend_definition = relationship("TrendDefinition", back_populates="charts")


class TrendDefinition(Base):
    __tablename__ = "trends"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    roles = Column(JSON, nullable=False)
    duration = Column(String, nullable=False)
    alpha = Column(Numeric, nullable=False, default=1)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)

    model = relationship("Model")
    charts = relationship("TrendChart", back_populates="trend_definition")


class DashboardChartConfig(Base):
    __tablename__ = "dashboard_chart_configs"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    layout = Column(JSON, default=list, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False
    )

    model = relationship("Model")


class TrendPredictionFrontier(Base):
    __tablename__ = "trend_prediction_frontiers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trend_id = Column(String, ForeignKey("trends.id"), nullable=False, index=True)
    role = Column(String, nullable=False)
    layer = Column(Integer, nullable=False)
    segment_index = Column(Integer, nullable=False)
    last_trend_timestamp = Column(DateTime, nullable=True)
    frontier_timestamp = Column(DateTime, nullable=True)
    target_timestamp = Column(DateTime, nullable=True)
    duration = Column(String, nullable=False)
    prediction_steps = Column(Integer, nullable=False, default=3)
    created_at = Column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False
    )

    trend_definition = relationship("TrendDefinition")

    __table_args__ = (
        UniqueConstraint("trend_id", "role", "layer", "segment_index", name="uq_trend_prediction_frontier"),
    )
