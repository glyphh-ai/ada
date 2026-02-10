"""add stored_procedures table

Revision ID: 20260210_stored_procs
Revises: 20260209_vector_dim
Create Date: 2026-02-10 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260210_stored_procs'
down_revision: Union[str, None] = '20260209_vector_dim'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create stored_procedures table."""
    op.create_table(
        'stored_procedures',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('org_id', sa.String(length=255), nullable=False),
        sa.Column('model_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('gql_query', sa.Text(), nullable=False),
        sa.Column('lexicons', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column('description', sa.Text(), server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('org_id', 'model_id', 'name', name='uq_procedure_org_model_name'),
        sa.CheckConstraint("name ~ '^[a-zA-Z][a-zA-Z0-9_]*$'", name='ck_procedure_valid_name'),
        sa.CheckConstraint('array_length(lexicons, 1) > 0', name='ck_procedure_has_lexicons'),
    )
    
    # Create index for efficient lookups by org_id and model_id
    op.create_index(
        'idx_stored_procedures_org_model',
        'stored_procedures',
        ['org_id', 'model_id'],
        unique=False
    )


def downgrade() -> None:
    """Drop stored_procedures table."""
    op.drop_index('idx_stored_procedures_org_model', table_name='stored_procedures')
    op.drop_table('stored_procedures')
