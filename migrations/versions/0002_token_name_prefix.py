"""Add name and token_prefix columns to tokens table.

Revision ID: 0002
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tokens", sa.Column("name", sa.String(255), nullable=True))
    op.add_column("tokens", sa.Column("token_prefix", sa.String(12), nullable=True))
    # Backfill existing rows
    op.execute("UPDATE tokens SET name = 'unnamed' WHERE name IS NULL")
    op.alter_column("tokens", "name", nullable=False)
    # Default permissions to read,write for new tokens
    op.execute("UPDATE tokens SET permissions = '[\"read\", \"write\"]'::jsonb WHERE permissions = '[\"read\"]'::jsonb")


def downgrade() -> None:
    op.drop_column("tokens", "token_prefix")
    op.drop_column("tokens", "name")
