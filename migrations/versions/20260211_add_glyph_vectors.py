"""Add glyph_vectors table for hierarchical embeddings.

Revision ID: 20260211_glyph_vectors
Revises: 20260210_add_stored_procedures
Create Date: 2026-02-11

This migration adds the glyph_vectors table to store hierarchical embeddings
(layer, segment, role level) for fine-grained similarity search.
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260211_glyph_vectors'
down_revision = '20260210_add_stored_procedures'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create glyph_vectors table
    op.create_table(
        'glyph_vectors',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('glyph_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', sa.String(255), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('level', sa.String(20), nullable=False),
        sa.Column('path', sa.String(500), nullable=False),
        sa.Column('embedding', Vector(2000), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['glyph_id'], ['glyphs.id'], ondelete='CASCADE'),
    )
    
    # Create indexes
    op.create_index('idx_glyph_vector_org_model', 'glyph_vectors', ['org_id', 'model_id'])
    op.create_index('idx_glyph_vector_level', 'glyph_vectors', ['org_id', 'model_id', 'level'])
    op.create_index('idx_glyph_vector_path', 'glyph_vectors', ['org_id', 'model_id', 'level', 'path'])
    op.create_index('idx_glyph_vector_glyph', 'glyph_vectors', ['glyph_id'])
    
    # Create unique constraint
    op.create_unique_constraint('uq_glyph_vector_path', 'glyph_vectors', ['glyph_id', 'level', 'path'])
    
    # Create HNSW vector index for similarity search
    op.execute("""
        CREATE INDEX idx_glyph_vector_embedding 
        ON glyph_vectors 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.drop_index('idx_glyph_vector_embedding', table_name='glyph_vectors')
    op.drop_constraint('uq_glyph_vector_path', 'glyph_vectors', type_='unique')
    op.drop_index('idx_glyph_vector_glyph', table_name='glyph_vectors')
    op.drop_index('idx_glyph_vector_path', table_name='glyph_vectors')
    op.drop_index('idx_glyph_vector_level', table_name='glyph_vectors')
    op.drop_index('idx_glyph_vector_org_model', table_name='glyph_vectors')
    op.drop_table('glyph_vectors')
