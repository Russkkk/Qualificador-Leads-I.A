from pathlib import Path
from typing import Any, Dict, Tuple

import psycopg
from psycopg.rows import dict_row

from services import settings
from services.utils import month_key

_SCHEMA_READY = False
_SCHEMA_LOCK = None
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def require_env_db():
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada (Render Environment)")


def db():
    require_env_db()
    return psycopg.connect(settings.DATABASE_URL, row_factory=dict_row)


def get_active_leads_query(alias: str | None = None) -> str:
    if alias:
        return f"FROM leads {alias} WHERE {alias}.deleted_at IS NULL"
    return "FROM leads WHERE deleted_at IS NULL"


def ensure_schema_once() -> Tuple[bool, str]:
    global _SCHEMA_READY, _SCHEMA_LOCK
    if _SCHEMA_READY:
        return True, ""
    if _SCHEMA_LOCK is None:
        import threading
        _SCHEMA_LOCK = threading.Lock()
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return True, ""
        try:
            ensure_schema()
            _SCHEMA_READY = True
            return True, ""
        except Exception as exc:
            return False, repr(exc)


def ensure_schema():
    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute("SELECT version FROM schema_migrations;")
                applied = {row[0] for row in (cur.fetchall() or [])}

                if not _MIGRATIONS_DIR.exists():
                    raise RuntimeError("Diretório migrations/ não encontrado.")

                for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
                    version = path.stem.split("_", 1)[0]
                    if version in applied:
                        continue
                    sql_text = path.read_text(encoding="utf-8")
                    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
                    for stmt in statements:
                        cur.execute(stmt)
                    cur.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)",
                        (version,),
                    )
    finally:
        conn.close()


def ensure_client_row(client_id: str, plan: str = "trial") -> Dict[str, Any]:
    plan = (plan or "trial").strip().lower()
    if plan not in settings.PLAN_CATALOG:
        plan = "trial"

    mk = month_key()

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO clients (client_id, api_key, plan, status, usage_month, leads_used_month, updated_at)
                    VALUES (%s, '', %s, 'active', %s, 0, NOW())
                    ON CONFLICT (client_id) DO NOTHING
                    """,
                    (client_id, plan, mk),
                )

                cur.execute("SELECT * FROM clients WHERE client_id=%s FOR UPDATE", (client_id,))
                row = cur.fetchone() or {}

                if (row.get("usage_month") or "").strip() != mk:
                    cur.execute(
                        "UPDATE clients SET usage_month=%s, leads_used_month=0, updated_at=NOW() WHERE client_id=%s",
                        (mk, client_id),
                    )
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row

                if row.get("api_key") is None:
                    cur.execute("UPDATE clients SET api_key='' WHERE client_id=%s", (client_id,))
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row

        return dict(row)
    finally:
        conn.close()
