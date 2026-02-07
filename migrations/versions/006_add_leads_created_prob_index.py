"""add leads created prob index

Revision ID: 006_add_leads_created_prob_index
Revises: 005_add_lead_soft_delete
Create Date: 2024-01-01 00:00:05.000000

"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "006_add_leads_created_prob_index"
down_revision = "005_add_lead_soft_delete"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_leads_client_created_prob ON leads (client_id, created_at DESC, probabilidade DESC)"
    )


def downgrade() -> None:
    pass
