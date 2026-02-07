"""add lead soft delete

Revision ID: 005_add_lead_soft_delete
Revises: 004_normalize_clients
Create Date: 2024-01-01 00:00:04.000000

"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "005_add_lead_soft_delete"
down_revision = "004_normalize_clients"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = [
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ",
        "CREATE INDEX IF NOT EXISTS idx_leads_client_deleted_at ON leads (client_id, deleted_at)",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    pass
