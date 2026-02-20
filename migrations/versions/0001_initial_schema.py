"""Initial schema — clean baseline for Glyphh Runtime.

Revision ID: 0001_initial
Revises: None
Create Date: 2026-02-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── glyphs ──
    op.create_table(
        "glyphs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(255), nullable=False, index=True),
        sa.Column("model_id", sa.String(255), nullable=False, index=True),
        sa.Column("concept_text", sa.Text(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),  # VectorType(2000) — pgvector handles via dialect
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_glyph_org_model", "glyphs", ["org_id", "model_id"])
    op.create_index("idx_glyph_org_model_created", "glyphs", ["org_id", "model_id", sa.text("created_at DESC")])

    # pgvector column + HNSW index (raw SQL — alembic can't do vector types natively)
    op.execute("ALTER TABLE glyphs ALTER COLUMN embedding TYPE vector(2000) USING embedding::vector(2000)")
    op.execute("""
        CREATE INDEX idx_glyph_embedding ON glyphs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ── glyph_vectors ──
    op.create_table(
        "glyph_vectors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("glyph_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("glyphs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("org_id", sa.String(255), nullable=False, index=True),
        sa.Column("model_id", sa.String(255), nullable=False, index=True),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("path", sa.String(500), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_glyph_vector_org_model", "glyph_vectors", ["org_id", "model_id"])
    op.create_index("idx_glyph_vector_level", "glyph_vectors", ["org_id", "model_id", "level"])
    op.create_index("idx_glyph_vector_path", "glyph_vectors", ["org_id", "model_id", "level", "path"])
    op.create_index("idx_glyph_vector_glyph", "glyph_vectors", ["glyph_id"])
    op.create_unique_constraint("uq_glyph_vector_path", "glyph_vectors", ["glyph_id", "level", "path"])

    op.execute("ALTER TABLE glyph_vectors ALTER COLUMN embedding TYPE vector(2000) USING embedding::vector(2000)")
    op.execute("""
        CREATE INDEX idx_glyph_vector_embedding ON glyph_vectors
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ── edges ──
    op.create_table(
        "edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(255), nullable=False, index=True),
        sa.Column("model_id", sa.String(255), nullable=False, index=True),
        sa.Column("source_glyph_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("glyphs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_glyph_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("glyphs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("edge_type", sa.String(50), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_edge_org_model", "edges", ["org_id", "model_id"])
    op.create_index("idx_edge_org_model_type", "edges", ["org_id", "model_id", "edge_type"])
    op.create_index("idx_edge_source", "edges", ["source_glyph_id"])
    op.create_index("idx_edge_target", "edges", ["target_glyph_id"])
    op.create_index("idx_edge_type", "edges", ["edge_type"])
    op.create_unique_constraint("uq_edge_source_target_type", "edges", ["source_glyph_id", "target_glyph_id", "edge_type"])

    # ── model_configs ──
    op.create_table(
        "model_configs",
        sa.Column("org_id", sa.String(255), primary_key=True),
        sa.Column("model_id", sa.String(255), primary_key=True),
        sa.Column("model_path", sa.Text(), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("sdk_version", sa.String(50), nullable=True),
        sa.Column("meta_name", sa.String(255), nullable=True),
        sa.Column("short_description", sa.String(200), nullable=True),
        sa.Column("long_description", sa.Text(), nullable=True),
        sa.Column("encoder_config", postgresql.JSONB(), nullable=True),
        sa.Column("similarity_weights", postgresql.JSONB(), server_default='{"similarity":1.0,"contrast":0.5,"analogy":0.7,"composition":0.8,"precedes":0.6,"follows":0.6,"causes":0.9,"prevents":0.4}'),
        sa.Column("beam_width", sa.Integer(), server_default="5"),
        sa.Column("max_tree_depth", sa.Integer(), server_default="3"),
        sa.Column("resource_quotas", postgresql.JSONB(), server_default='{"memory_mb":1024,"storage_gb":10,"max_glyphs":1000000}'),
        sa.Column("resource_usage", postgresql.JSONB(), server_default='{"memory_mb":0,"storage_gb":0,"glyph_count":0}'),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    # ── tokens ──
    op.create_table(
        "tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("token_hash", sa.String(255), nullable=False, unique=True),
        sa.Column("org_id", sa.String(255), nullable=False, index=True),
        sa.Column("model_id", sa.String(255), nullable=True, index=True),
        sa.Column("permissions", postgresql.JSONB(), server_default='["read"]'),
        sa.Column("status", sa.String(50), server_default="active"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_token_status", "tokens", ["status"])
    op.create_index("idx_token_org", "tokens", ["org_id"])
    op.create_index("idx_token_org_model", "tokens", ["org_id", "model_id"])

    # ── stored_procedures ──
    op.create_table(
        "stored_procedures",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(255), nullable=False, index=True),
        sa.Column("model_id", sa.String(255), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("gql_query", sa.Text(), nullable=False),
        sa.Column("lexicons", postgresql.JSONB(), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("stored_procedures")
    op.drop_table("tokens")
    op.drop_table("model_configs")
    op.drop_table("edges")
    op.drop_table("glyph_vectors")
    op.drop_table("glyphs")
    op.execute("DROP EXTENSION IF EXISTS vector")
