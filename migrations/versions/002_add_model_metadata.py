"""Add model metadata columns for marketplace display.

Revision ID: 002_add_model_metadata
Revises: 001
Create Date: 2026-01-31

Adds meta_name, short_description, and long_description columns
to model_configs table for marketplace display.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_model_metadata'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add model metadata columns to model_configs table."""
    # Add meta_name column
    op.add_column(
        'model_configs',
        sa.Column('meta_name', sa.String(255), nullable=True)
    )
    
    # Add short_description column (max 200 chars for marketplace cards)
    op.add_column(
        'model_configs',
        sa.Column('short_description', sa.String(200), nullable=True)
    )
    
    # Add long_description column (full description, markdown supported)
    op.add_column(
        'model_configs',
        sa.Column('long_description', sa.Text(), nullable=True)
    )
    
    # Set default values for existing rows
    # meta_name defaults to namespace, descriptions default to empty string
    op.execute("""
        UPDATE model_configs 
        SET meta_name = namespace,
            short_description = '',
            long_description = ''
        WHERE meta_name IS NULL
    """)


def downgrade() -> None:
    """Remove model metadata columns from model_configs table."""
    op.drop_column('model_configs', 'long_description')
    op.drop_column('model_configs', 'short_description')
    op.drop_column('model_configs', 'meta_name')
