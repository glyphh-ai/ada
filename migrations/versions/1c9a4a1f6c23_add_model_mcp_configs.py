"""add model mcp configs

Revision ID: 1c9a4a1f6c23
Revises: 3d3bb55aa9d8
Create Date: 2026-01-27
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1c9a4a1f6c23"
down_revision = "3d3bb55aa9d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_mcp_configs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_model_mcp_configs_model_id"),
        "model_mcp_configs",
        ["model_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_model_mcp_configs_model_id"), table_name="model_mcp_configs")
    op.drop_table("model_mcp_configs")
