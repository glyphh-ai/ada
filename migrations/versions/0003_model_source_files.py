"""Add source_files column to model_configs table.

Stores Python source files (encoder.py, intent.py, etc.) as JSON text
so the runtime can restore encode_query_fn from the database without
needing the original files on disk.

Revision ID: 0003
Revises: 0002
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_configs", sa.Column("source_files", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("model_configs", "source_files")
