"""drop model_version_history table

Revision ID: 20260212_drop_mvh
Revises: 20260211_glyph_vectors
Create Date: 2026-02-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '20260212_drop_mvh'
down_revision: Union[str, None] = '20260211_glyph_vectors'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop the model_version_history table."""
    op.drop_table('model_version_history')


def downgrade() -> None:
    """Recreate the model_version_history table."""
    op.create_table(
        'model_version_history',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('org_id', sa.String(length=255), nullable=False),
        sa.Column('model_id', sa.String(length=255), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('deployed_at', sa.DateTime(), nullable=False),
        sa.Column('deployed_by', sa.String(length=255), nullable=True),
        sa.Column('is_current', sa.Integer(), server_default='1'),
        sa.Column('metadata', sa.JSON(), server_default='{}'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_model_version_history_org_id', 'model_version_history', ['org_id'])
    op.create_index('ix_model_version_history_model_id', 'model_version_history', ['model_id'])
    op.create_index('idx_version_history_org_model', 'model_version_history', ['org_id', 'model_id'])
    op.create_index('idx_version_history_current', 'model_version_history', ['org_id', 'model_id', 'is_current'])
    op.create_index('idx_version_history_deployed', 'model_version_history', ['org_id', 'model_id', sa.text('deployed_at DESC')])
