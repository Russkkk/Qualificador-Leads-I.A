from flask import Blueprint
from psycopg.rows import dict_row

from extensions import limiter
from services.db import db
from services.demo_service import require_admin_key
from services.utils import json_err, json_ok, month_key

admin_bp = Blueprint("admin", __name__)


@admin_bp.post("/admin/reset_month")
@limiter.limit("100 per minute")
def admin_reset_month():
    ok, _ = require_admin_key()
    if not ok:
        return json_err("Unauthorized", 403)

    mk = month_key()
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    UPDATE clients
                    SET usage_month=%s, leads_used_month=0, updated_at=NOW()
                    WHERE usage_month IS NULL OR usage_month<>%s
                    """,
                    (mk, mk),
                )
                cur.execute("SELECT COUNT(*) AS n FROM clients")
                n = int((cur.fetchone() or {}).get("n") or 0)
        return json_ok({"usage_month": mk, "clients_total": n})
    finally:
        conn.close()
