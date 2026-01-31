"""Initial schema with glyphs, edges, model_configs, and tokens

Revision ID: 001
Revises: 
Create Date: 2024-01-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create glyphs table
    op.create_table(
        'glyphs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('namespace', sa.String(255), nullable=False),
        sa.Column('concept_text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(768), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for glyphs
    op.create_index('idx_glyph_namespace', 'glyphs', ['namespace'])
    op.create_index(
        'idx_glyph_embedding',
        'glyphs',
        ['embedding'],
        postgresql_using='ivfflat',
        postgresql_with={'lists': 100},
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    )
    op.create_index('idx_glyph_namespace_created', 'glyphs', ['namespace', sa.text('created_at DESC')])
    
    # Create edges table
    op.create_table(
        'edges',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('namespace', sa.String(255), nullable=False),
        sa.Column('source_glyph_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_glyph_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('edge_type', sa.String(50), nullable=False),
        sa.Column('weight', sa.Float(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['source_glyph_id'], ['glyphs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_glyph_id'], ['glyphs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_glyph_id', 'target_glyph_id', 'edge_type', name='uq_edge_source_target_type')
    )
    
    # Create indexes for edges
    op.create_index('idx_edge_namespace', 'edges', ['namespace'])
    op.create_index('idx_edge_source', 'edges', ['source_glyph_id'])
    op.create_index('idx_edge_target', 'edges', ['target_glyph_id'])
    op.create_index('idx_edge_type', 'edges', ['edge_type'])
    op.create_index('idx_edge_namespace_type', 'edges', ['namespace', 'edge_type'])
    
    # Create model_configs table
    op.create_table(
        'model_configs',
        sa.Column('namespace', sa.String(255), nullable=False),
        sa.Column('model_path', sa.Text(), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('sdk_version', sa.String(50), nullable=True),
        sa.Column('similarity_weights', postgresql.JSONB(), nullable=True),
        sa.Column('beam_width', sa.Integer(), nullable=True),
        sa.Column('max_tree_depth', sa.Integer(), nullable=True),
        sa.Column('resource_quotas', postgresql.JSONB(), nullable=True),
        sa.Column('resource_usage', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('namespace')
    )
    
    # Create tokens table
    op.create_table(
        'tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False),
        sa.Column('namespace', sa.String(255), nullable=True),
        sa.Column('permissions', postgresql.JSONB(), nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash')
    )
    
    # Create indexes for tokens
    op.create_index('idx_token_status', 'tokens', ['status'])
    op.create_index('idx_token_namespace', 'tokens', ['namespace'])


def downgrade() -> None:
    op.drop_table('tokens')
    op.drop_table('model_configs')
    op.drop_table('edges')
    op.drop_table('glyphs')
    op.execute('DROP EXTENSION IF EXISTS vector')
