"""add legacy columns

Revision ID: 002_add_legacy_columns
Revises: 001_initial_schema
Create Date: 2024-01-01 00:00:01.000000

"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "002_add_legacy_columns"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = [
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS client_id TEXT",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS nome TEXT",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS email_lead TEXT",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS telefone TEXT",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS origem TEXT",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS tempo_site INTEGER",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS paginas_visitadas INTEGER",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS clicou_preco INTEGER",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS probabilidade DOUBLE PRECISION",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS virou_cliente DOUBLE PRECISION",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS score INTEGER",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS label INTEGER",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "ALTER TABLE leads ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS api_key TEXT",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS nome TEXT",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS email TEXT",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS empresa TEXT",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS telefone TEXT",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS password_hash TEXT",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'trial'",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS usage_month TEXT",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS leads_used_month INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "ALTER TABLE clients ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    pass
