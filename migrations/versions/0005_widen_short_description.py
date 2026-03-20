"""Widen short_description from VARCHAR(200) to TEXT.

Model manifest descriptions were hitting the 200-char limit on deploy.

Revision ID: 0005
Revises: 0004
"""

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "model_configs",
        "short_description",
        type_=sa.Text(),
        existing_type=sa.String(200),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "model_configs",
        "short_description",
        type_=sa.String(200),
        existing_type=sa.Text(),
        existing_nullable=True,
    )
