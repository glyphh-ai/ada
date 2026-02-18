"""
SQLAlchemy database models for Glyphh Runtime.

Defines the core tables: glyphs, edges, and model_configs.
Uses pgvector for vector embeddings.

All tables use org_id and model_id for tenant isolation.
org_id is the primary security boundary — every query must filter on it.
model_id scopes the data/vector space within an org.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

# Use JSONB on PostgreSQL, plain JSON on SQLite/others
JSONType = JSON().with_variant(JSONB(), "postgresql")


def VectorType(dim: int):
    """Vector column type: pgvector on PostgreSQL, LargeBinary on SQLite/others."""
    return LargeBinary().with_variant(Vector(dim), "postgresql")

from infrastructure.database.connection import Base


class Glyph(Base):
    """
    Glyph model - vector representation of a concept.
    
    Each glyph belongs to an org and model, containing:
    - Vector embedding (global cortex) for similarity search
    - Concept text (original input)
    - Metadata (arbitrary JSON)
    - Hierarchical vectors (layers, segments, roles) in glyph_vectors table
    """
    __tablename__ = "glyphs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    org_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    concept_text = Column(Text, nullable=False)
    embedding = Column(VectorType(2000), nullable=False)  # Global cortex - max 2000 dims (pgvector HNSW index limit)
    glyph_metadata = Column("metadata", JSONType, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    outgoing_edges = relationship(
        "Edge",
        foreign_keys="Edge.source_glyph_id",
        back_populates="source_glyph",
        cascade="all, delete-orphan"
    )
    incoming_edges = relationship(
        "Edge",
        foreign_keys="Edge.target_glyph_id",
        back_populates="target_glyph",
        cascade="all, delete-orphan"
    )
    vectors = relationship(
        "GlyphVector",
        back_populates="glyph",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        # Composite index for org/model scoped queries
        Index("idx_glyph_org_model", org_id, model_id),
        # Composite index for scoped queries with time ordering
        Index("idx_glyph_org_model_created", org_id, model_id, created_at.desc()),
        # Index for vector similarity search (HNSW - faster than IVFFlat)
        Index(
            "idx_glyph_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )
    
    def __repr__(self) -> str:
        return f"<Glyph(id={self.id}, org_id={self.org_id}, model_id={self.model_id})>"


class GlyphVector(Base):
    """
    Hierarchical vector storage for glyphs.
    
    Stores layer, segment, and role level embeddings for fine-grained
    similarity search. Each vector is associated with a glyph and has:
    - level: 'layer', 'segment', or 'role'
    - path: hierarchical path (e.g., 'semantic', 'semantic.attributes', 'semantic.attributes.color')
    - embedding: the vector for this level
    
    This enables queries like:
    - FIND SIMILAR TO "red car" AT LAYER semantic
    - FIND SIMILAR TO "sports" AT SEGMENT semantic.attributes
    - FIND SIMILAR TO "red" AT ROLE semantic.attributes.color
    """
    __tablename__ = "glyph_vectors"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    glyph_id = Column(
        UUID(as_uuid=True),
        ForeignKey("glyphs.id", ondelete="CASCADE"),
        nullable=False
    )
    org_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    level = Column(String(20), nullable=False)  # 'layer', 'segment', 'role'
    path = Column(String(500), nullable=False)  # e.g., 'semantic.attributes.color'
    embedding = Column(VectorType(2000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship back to glyph
    glyph = relationship("Glyph", back_populates="vectors")
    
    __table_args__ = (
        # Composite index for org/model scoped queries
        Index("idx_glyph_vector_org_model", org_id, model_id),
        # Index for level-based queries
        Index("idx_glyph_vector_level", org_id, model_id, level),
        # Index for path-based queries
        Index("idx_glyph_vector_path", org_id, model_id, level, path),
        # Index for glyph lookup
        Index("idx_glyph_vector_glyph", glyph_id),
        # Vector similarity search index (HNSW)
        Index(
            "idx_glyph_vector_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        # Prevent duplicate vectors for same glyph/level/path
        UniqueConstraint(
            "glyph_id", "level", "path",
            name="uq_glyph_vector_path"
        ),
    )


class Edge(Base):
    """
    Edge model - relationship between two glyphs.
    
    Supports 8 edge types:
    - Spatial: similarity, contrast, analogy, composition
    - Temporal: precedes, follows, causes, prevents
    """
    __tablename__ = "edges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    org_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    source_glyph_id = Column(
        UUID(as_uuid=True),
        ForeignKey("glyphs.id", ondelete="CASCADE"),
        nullable=False
    )
    target_glyph_id = Column(
        UUID(as_uuid=True),
        ForeignKey("glyphs.id", ondelete="CASCADE"),
        nullable=False
    )
    edge_type = Column(String(50), nullable=False)  # similarity, contrast, etc.
    weight = Column(Float, nullable=False)
    edge_metadata = Column("metadata", JSONType, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)  # For TTL-based cache invalidation
    
    # Relationships
    source_glyph = relationship(
        "Glyph",
        foreign_keys=[source_glyph_id],
        back_populates="outgoing_edges"
    )
    target_glyph = relationship(
        "Glyph",
        foreign_keys=[target_glyph_id],
        back_populates="incoming_edges"
    )
    
    __table_args__ = (
        # Composite index for org/model scoped queries
        Index("idx_edge_org_model", org_id, model_id),
        # Composite index for scoped edge type lookups
        Index("idx_edge_org_model_type", org_id, model_id, edge_type),
        # Indexes for edge lookups
        Index("idx_edge_source", source_glyph_id),
        Index("idx_edge_target", target_glyph_id),
        Index("idx_edge_type", edge_type),
        # Prevent duplicate edges
        UniqueConstraint(
            "source_glyph_id", "target_glyph_id", "edge_type",
            name="uq_edge_source_target_type"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<Edge(id={self.id}, type={self.edge_type}, weight={self.weight})>"


class ModelConfig(Base):
    """
    Model configuration - per org/model settings.
    
    Stores configuration for each loaded model including:
    - Similarity weights for edge types
    - Beam search parameters
    - Resource quotas
    - Model metadata for marketplace display
    """
    __tablename__ = "model_configs"
    
    org_id = Column(String(255), primary_key=True)
    model_id = Column(String(255), primary_key=True)
    model_path = Column(Text, nullable=False)
    model_version = Column(String(50), nullable=True)
    sdk_version = Column(String(50), nullable=True)
    
    # Model metadata for marketplace display
    meta_name = Column(String(255), nullable=True)
    short_description = Column(String(200), nullable=True)
    long_description = Column(Text, nullable=True)
    
    # Full encoder config for reconstructing the model without the .glyphh file
    encoder_config = Column(JSONType, nullable=True)
    
    # Similarity weights for each edge type
    similarity_weights = Column(JSONType, default=lambda: {
        "similarity": 1.0,
        "contrast": 0.5,
        "analogy": 0.7,
        "composition": 0.8,
        "precedes": 0.6,
        "follows": 0.6,
        "causes": 0.9,
        "prevents": 0.4,
    })
    
    # Beam search parameters
    beam_width = Column(Integer, default=5)
    max_tree_depth = Column(Integer, default=3)
    
    # Resource quotas
    resource_quotas = Column(JSONType, default=lambda: {
        "memory_mb": 1024,
        "storage_gb": 10,
        "max_glyphs": 1000000,
    })
    
    # Resource usage tracking
    resource_usage = Column(JSONType, default=lambda: {
        "memory_mb": 0,
        "storage_gb": 0,
        "glyph_count": 0,
    })
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"<ModelConfig(org_id={self.org_id}, model_id={self.model_id})>"


class Token(Base):
    """
    Token model - webhook/consumer tokens for API access.
    
    Tokens are scoped to org_id (required) and optionally model_id.
    A token with model_id=None grants access to all models in the org.
    """
    __tablename__ = "tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    token_hash = Column(String(255), nullable=False, unique=True)
    org_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=True, index=True)  # nullable for org-wide tokens
    permissions = Column(JSONType, default=lambda: ["read"])
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index("idx_token_status", status),
        Index("idx_token_org", org_id),
        Index("idx_token_org_model", org_id, model_id),
    )
    
    def __repr__(self) -> str:
        return f"<Token(id={self.id}, org_id={self.org_id}, status={self.status})>"


class ModelVersionHistory(Base):
    """
    Model version history - tracks all deployed versions of a model.
    
    Records version, deployment timestamp, and deployer for each deployment.
    History is retained even when the model is deleted (soft delete).
    """
    __tablename__ = "model_version_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    org_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    deployed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deployed_by = Column(String(255), nullable=True)  # User ID or email
    is_current = Column(Integer, default=1)  # 1 = current version, 0 = historical
    model_metadata = Column("metadata", JSONType, default=dict)  # Additional version metadata
    
    __table_args__ = (
        Index("idx_version_history_org_model", org_id, model_id),
        Index("idx_version_history_current", org_id, model_id, is_current),
        Index("idx_version_history_deployed", org_id, model_id, deployed_at.desc()),
    )
    
    def __repr__(self) -> str:
        return f"<ModelVersionHistory(org_id={self.org_id}, model_id={self.model_id}, version={self.version})>"
