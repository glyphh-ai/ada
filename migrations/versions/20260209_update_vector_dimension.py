"""Update vector dimension from 768 to 2048.

Revision ID: 20260209_vector_dim
Revises: 88d76764e4cb
Create Date: 2026-02-09

This migration updates the glyphs table to support up to 2048 dimensions.
This is the maximum supported by pgvector indexes (IVFFlat and HNSW).
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = '20260209_vector_dim'
down_revision = '88d76764e4cb'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade vector dimension from 768 to 2048."""
    # Drop the existing IVFFlat index first
    op.drop_index('idx_glyph_embedding', table_name='glyphs')
    
    # Alter the column to new dimension
    # Note: This will pad existing vectors with zeros
    op.execute("""
        ALTER TABLE glyphs 
        ALTER COLUMN embedding TYPE vector(2048) 
        USING embedding::vector(2048)
    """)
    
    # Recreate the index using HNSW (generally faster than IVFFlat)
    op.execute("""
        CREATE INDEX idx_glyph_embedding ON glyphs 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    """Downgrade vector dimension from 2048 to 768."""
    # Drop the HNSW index
    op.drop_index('idx_glyph_embedding', table_name='glyphs')
    
    # Alter back to 768 (will truncate vectors!)
    op.execute("""
        ALTER TABLE glyphs 
        ALTER COLUMN embedding TYPE vector(768) 
        USING embedding::vector(768)
    """)
    
    # Recreate the IVFFlat index
    op.create_index(
        'idx_glyph_embedding',
        'glyphs',
        ['embedding'],
        postgresql_using='ivfflat',
        postgresql_with={'lists': 100},
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    )
