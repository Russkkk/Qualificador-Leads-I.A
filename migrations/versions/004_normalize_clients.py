"""normalize clients

Revision ID: 004_normalize_clients
Revises: 003_indexes
Create Date: 2024-01-01 00:00:03.000000

"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "004_normalize_clients"
down_revision = "003_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = [
        "ALTER TABLE clients ALTER COLUMN api_key DROP NOT NULL",
        "UPDATE clients SET usage_month = TO_CHAR(NOW(), 'YYYY-MM') WHERE usage_month IS NULL OR usage_month = ''",
        "UPDATE clients SET api_key = '' WHERE api_key IS NULL",
        "UPDATE clients SET updated_at = NOW() WHERE updated_at IS NULL",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    pass
