"""Replace namespace with org_id and model_id columns.

Revision ID: 003_replace_namespace_with_org_model_id
Revises: 002_add_model_metadata
Create Date: 2026-02-07

Replaces the composite namespace string (org_id/model_id) with first-class
org_id and model_id columns on all four tables. This is a security improvement:
org isolation is now enforced at the database level via mandatory org_id filtering,
not via application-level string parsing.

Migration steps:
1. Add org_id and model_id columns (nullable initially)
2. Populate from existing namespace by splitting on '/'
3. Fail if any namespace lacks a '/' delimiter
4. Make columns non-nullable (except tokens.model_id)
5. Drop namespace columns and old indexes
6. Change model_configs PK from namespace to (org_id, model_id)
7. Create new composite indexes
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_replace_namespace_with_org_model_id'
down_revision: Union[str, None] = '002_add_model_metadata'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # =========================================================================
    # 1. Add new columns (nullable initially for data migration)
    # =========================================================================
    for table in ['glyphs', 'edges', 'model_configs', 'tokens']:
        op.add_column(table, sa.Column('org_id', sa.String(255), nullable=True))
        op.add_column(table, sa.Column('model_id', sa.String(255), nullable=True))

    # =========================================================================
    # 2. Populate from existing namespace (split on '/')
    # =========================================================================
    for table in ['glyphs', 'edges', 'model_configs', 'tokens']:
        # Fail if any row has a namespace without '/'
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM {table}
                    WHERE namespace IS NOT NULL
                      AND namespace NOT LIKE '%/%'
                ) THEN
                    RAISE EXCEPTION 'Malformed namespace in {table}: found rows without / delimiter';
                END IF;
            END $$;
        """)

        # Split namespace into org_id and model_id
        op.execute(f"""
            UPDATE {table}
            SET org_id = split_part(namespace, '/', 1),
                model_id = split_part(namespace, '/', 2)
            WHERE namespace IS NOT NULL
        """)

    # =========================================================================
    # 3. Make columns non-nullable (except tokens.model_id)
    # =========================================================================
    for table in ['glyphs', 'edges', 'model_configs']:
        op.alter_column(table, 'org_id', nullable=False)
        op.alter_column(table, 'model_id', nullable=False)
    op.alter_column('tokens', 'org_id', nullable=False)
    # tokens.model_id stays nullable for org-wide tokens

    # =========================================================================
    # 4. Drop old namespace-based indexes
    # =========================================================================
    op.drop_index('idx_glyph_namespace', 'glyphs')
    op.drop_index('idx_glyph_namespace_created', 'glyphs')
    op.drop_index('idx_edge_namespace', 'edges')
    op.drop_index('idx_edge_namespace_type', 'edges')
    op.drop_index('idx_token_namespace', 'tokens')

    # =========================================================================
    # 5. Drop namespace columns
    # =========================================================================
    for table in ['glyphs', 'edges', 'tokens']:
        op.drop_column(table, 'namespace')

    # model_configs: change PK from namespace to (org_id, model_id)
    op.execute("ALTER TABLE model_configs DROP CONSTRAINT model_configs_pkey")
    op.drop_column('model_configs', 'namespace')
    op.create_primary_key('model_configs_pkey', 'model_configs', ['org_id', 'model_id'])

    # =========================================================================
    # 6. Create new composite indexes
    # =========================================================================
    op.create_index('idx_glyph_org_model', 'glyphs', ['org_id', 'model_id'])
    op.create_index('idx_glyph_org_model_created', 'glyphs',
                     ['org_id', 'model_id', sa.text('created_at DESC')])
    op.create_index('idx_edge_org_model', 'edges', ['org_id', 'model_id'])
    op.create_index('idx_edge_org_model_type', 'edges', ['org_id', 'model_id', 'edge_type'])
    op.create_index('idx_token_org', 'tokens', ['org_id'])
    op.create_index('idx_token_org_model', 'tokens', ['org_id', 'model_id'])


def downgrade() -> None:
    # =========================================================================
    # Reverse: add namespace back, populate, drop org_id/model_id
    # =========================================================================

    # Drop new indexes
    op.drop_index('idx_token_org_model', 'tokens')
    op.drop_index('idx_token_org', 'tokens')
    op.drop_index('idx_edge_org_model_type', 'edges')
    op.drop_index('idx_edge_org_model', 'edges')
    op.drop_index('idx_glyph_org_model_created', 'glyphs')
    op.drop_index('idx_glyph_org_model', 'glyphs')

    # model_configs: change PK back to namespace
    op.execute("ALTER TABLE model_configs DROP CONSTRAINT model_configs_pkey")
    op.add_column('model_configs', sa.Column('namespace', sa.String(255), nullable=True))
    op.execute("UPDATE model_configs SET namespace = org_id || '/' || model_id")
    op.alter_column('model_configs', 'namespace', nullable=False)
    op.create_primary_key('model_configs_pkey', 'model_configs', ['namespace'])
    op.drop_column('model_configs', 'org_id')
    op.drop_column('model_configs', 'model_id')

    # Add namespace back to other tables
    for table in ['glyphs', 'edges', 'tokens']:
        op.add_column(table, sa.Column('namespace', sa.String(255), nullable=True))
        op.execute(f"UPDATE {table} SET namespace = org_id || '/' || model_id")
        if table != 'tokens':
            op.alter_column(table, 'namespace', nullable=False)
        op.drop_column(table, 'org_id')
        op.drop_column(table, 'model_id')

    # Recreate old indexes
    op.create_index('idx_glyph_namespace', 'glyphs', ['namespace'])
    op.create_index('idx_glyph_namespace_created', 'glyphs', ['namespace', sa.text('created_at DESC')])
    op.create_index('idx_edge_namespace', 'edges', ['namespace'])
    op.create_index('idx_edge_namespace_type', 'edges', ['namespace', 'edge_type'])
    op.create_index('idx_token_namespace', 'tokens', ['namespace'])
