"""add indexes

Revision ID: 003_indexes
Revises: 002_add_legacy_columns
Create Date: 2024-01-01 00:00:02.000000

"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "003_indexes"
down_revision = "002_add_legacy_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_leads_client_created ON leads(client_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_leads_client_label ON leads(client_id, virou_cliente)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_api_key ON clients(api_key) WHERE api_key <> ''",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_email ON clients(email) WHERE email IS NOT NULL AND email <> ''",
        "CREATE INDEX IF NOT EXISTS idx_billing_events_client_created ON billing_events(client_id, created_at DESC)",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    pass
