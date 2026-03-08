"""Add staged_exemplars column to model_configs table.

Stores raw JSONL text during model deploy. A background job encodes
the entries into glyphs, then NULLs this column. Survives dyno restarts.

Revision ID: 0004
Revises: 0003
"""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_configs", sa.Column("staged_exemplars", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("model_configs", "staged_exemplars")
