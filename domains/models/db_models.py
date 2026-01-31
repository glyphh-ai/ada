"""
SQLAlchemy database models for Glyphh Runtime.

Defines the core tables: glyphs, edges, and model_configs.
Uses pgvector for vector embeddings.
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
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from infrastructure.database.connection import Base


class Glyph(Base):
    """
    Glyph model - vector representation of a concept.
    
    Each glyph belongs to a namespace (model) and contains:
    - Vector embedding for similarity search
    - Concept text (original input)
    - Metadata (arbitrary JSON)
    """
    __tablename__ = "glyphs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace = Column(String(255), nullable=False, index=True)
    concept_text = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False)  # 768-dim for all-MiniLM-L6-v2
    glyph_metadata = Column("metadata", JSONB, default=dict)
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
    
    __table_args__ = (
        # Index for vector similarity search (IVFFlat)
        Index(
            "idx_glyph_embedding",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        # Composite index for namespace queries
        Index("idx_glyph_namespace_created", namespace, created_at.desc()),
    )
    
    def __repr__(self) -> str:
        return f"<Glyph(id={self.id}, namespace={self.namespace})>"


class Edge(Base):
    """
    Edge model - relationship between two glyphs.
    
    Supports 8 edge types:
    - Spatial: similarity, contrast, analogy, composition
    - Temporal: precedes, follows, causes, prevents
    """
    __tablename__ = "edges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace = Column(String(255), nullable=False, index=True)
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
    edge_metadata = Column("metadata", JSONB, default=dict)
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
        # Indexes for edge lookups
        Index("idx_edge_source", source_glyph_id),
        Index("idx_edge_target", target_glyph_id),
        Index("idx_edge_type", edge_type),
        Index("idx_edge_namespace_type", namespace, edge_type),
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
    Model configuration - per-namespace settings.
    
    Stores configuration for each loaded model including:
    - Similarity weights for edge types
    - Beam search parameters
    - Resource quotas
    """
    __tablename__ = "model_configs"
    
    namespace = Column(String(255), primary_key=True)
    model_path = Column(Text, nullable=False)
    model_version = Column(String(50), nullable=True)
    sdk_version = Column(String(50), nullable=True)
    
    # Similarity weights for each edge type
    similarity_weights = Column(JSONB, default=lambda: {
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
    resource_quotas = Column(JSONB, default=lambda: {
        "memory_mb": 1024,
        "storage_gb": 10,
        "max_glyphs": 1000000,
    })
    
    # Resource usage tracking
    resource_usage = Column(JSONB, default=lambda: {
        "memory_mb": 0,
        "storage_gb": 0,
        "glyph_count": 0,
    })
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"<ModelConfig(namespace={self.namespace})>"


class Token(Base):
    """
    Token model - webhook/consumer tokens for API access.
    
    Tokens are created via Platform UI and stored here for validation.
    """
    __tablename__ = "tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    token_hash = Column(String(255), nullable=False, unique=True)  # Hashed token
    namespace = Column(String(255), nullable=True)  # Optional: restrict to namespace
    permissions = Column(JSONB, default=lambda: ["read"])  # read, write, admin
    status = Column(String(50), default="active")  # active, revoked
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index("idx_token_status", status),
        Index("idx_token_namespace", namespace),
    )
    
    def __repr__(self) -> str:
        return f"<Token(id={self.id}, status={self.status})>"
