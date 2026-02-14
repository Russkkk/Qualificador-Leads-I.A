import atexit
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import psycopg
from psycopg.rows import dict_row
try:
    from psycopg_pool import ConnectionPool
except Exception:  # pragma: no cover - fallback para ambientes sem pacote
    ConnectionPool = None

from services import settings
from services.utils import month_key

_SCHEMA_READY = False
_SCHEMA_LOCK = None
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_POOL: "ConnectionPool | None" = None


class _PooledConn:
    """Adapter para manter compatibilidade com o padrão atual conn.close()."""

    def __init__(self, pool, conn: psycopg.Connection):
        self._pool = pool
        self._conn = conn
        self._released = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)

    def close(self):
        if self._released:
            return
        self._released = True
        try:
            if self._conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                self._conn.rollback()
        except Exception:
            pass
        self._pool.putconn(self._conn)


def require_env_db():
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada (Render Environment)")


def _pool_min() -> int:
    try:
        return max(1, int(os.getenv("DB_POOL_MIN", "1")))
    except Exception:
        return 1


def _pool_max() -> int:
    try:
        return max(_pool_min(), int(os.getenv("DB_POOL_MAX", "10")))
    except Exception:
        return max(_pool_min(), 10)


def _pool_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("DB_POOL_TIMEOUT", "5")))
    except Exception:
        return 5.0


def _conn_timeout() -> int:
    try:
        return max(1, int(os.getenv("DB_CONN_TIMEOUT", "5")))
    except Exception:
        return 5


def _statement_timeout_ms() -> int:
    try:
        return max(0, int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "0")))
    except Exception:
        return 0


def _pool_kwargs() -> dict:
    kwargs = {
        "autocommit": False,
        "row_factory": dict_row,
        "connect_timeout": _conn_timeout(),
    }
    stmt_ms = _statement_timeout_ms()
    if stmt_ms > 0:
        kwargs["options"] = f"-c statement_timeout={stmt_ms}"
    return kwargs


def _get_pool():
    global _POOL
    if _POOL is not None:
        return _POOL
    if ConnectionPool is None:
        return None

    require_env_db()
    _POOL = ConnectionPool(
        conninfo=settings.DATABASE_URL,
        min_size=_pool_min(),
        max_size=_pool_max(),
        timeout=_pool_timeout(),
        kwargs=_pool_kwargs(),
        open=True,
    )
    return _POOL


def close_db_pool():
    global _POOL
    if _POOL is None:
        return
    try:
        _POOL.close()
    finally:
        _POOL = None


atexit.register(close_db_pool)


def db():
    require_env_db()
    pool = _get_pool()
    if pool is None:
        return psycopg.connect(settings.DATABASE_URL, **_pool_kwargs())
    conn = pool.getconn()
    return _PooledConn(pool, conn)


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
