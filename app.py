# app.py - LeadRank / Qualificador de Leads IA (Render + Postgres)
# Versão com auto-migração robusta para bancos antigos:
# - clients: corrige api_key NOT NULL, usage_month, etc.
# - leads: adiciona colunas faltantes (payload JSONB, probabilidade, score, etc.)

import os
import json
import time
import random
import secrets
import hashlib
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify
from flask_cors import CORS

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DEMO_KEY = os.environ.get("DEMO_KEY", "").strip()

ALLOWED_ORIGINS = [
    "https://qualificador-leads-ia.onrender.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

PLAN_CATALOG = {
    "trial":   {"price_brl_month": 0,   "lead_limit_month": 100},
    "demo":    {"price_brl_month": 0,   "lead_limit_month": 30},
    "starter": {"price_brl_month": 79,  "lead_limit_month": 1000},
    "pro":     {"price_brl_month": 179, "lead_limit_month": 5000},
    "vip":     {"price_brl_month": 279, "lead_limit_month": 20000},
}

_DEMO_RL = {}
_DEMO_RL_WINDOW_S = 60 * 60
_DEMO_RL_MAX = 5

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})


# -------------------------
# Utils
# -------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def _month_key(dt: datetime | None = None) -> str:
    dt = dt or _now_utc()
    return dt.strftime("%Y-%m")

def _resp(payload, code=200):
    return jsonify(payload), code

def _require_env_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada no Render (Environment)")

def _db():
    _require_env_db()
    return psycopg2.connect(DATABASE_URL)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _gen_api_key(client_id: str) -> str:
    raw = f"{client_id}:{secrets.token_urlsafe(24)}:{time.time()}"
    return "sk_live_" + _sha256(raw)[:32]

def _get_header(name: str) -> str:
    return (request.headers.get(name) or "").strip()

def _check_demo_key() -> bool:
    return bool(DEMO_KEY) and _get_header("X-DEMO-KEY") == DEMO_KEY

def _get_api_key_from_headers() -> str:
    key = _get_header("X-API-KEY") or _get_header("Authorization")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key

def _client_ip() -> str:
    return (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0].strip()


# -------------------------
# Schema / Migrations
# -------------------------
def _ensure_schema():
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                # LEADS (create + migrate columns)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS leads (
                        id BIGSERIAL PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        probabilidade DOUBLE PRECISION,
                        score INTEGER,
                        label INTEGER,
                        virou_cliente INTEGER,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_created ON leads(client_id, created_at DESC);")

                # bancos antigos: leads pode existir sem essas colunas
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS client_id TEXT;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS probabilidade DOUBLE PRECISION;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS score INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS label INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS virou_cliente INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

                # CLIENTS (create + migrate columns)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS clients (
                        client_id TEXT PRIMARY KEY,
                        api_key TEXT,
                        plan TEXT NOT NULL DEFAULT 'trial',
                        status TEXT NOT NULL DEFAULT 'active',
                        usage_month TEXT NOT NULL DEFAULT '',
                        leads_used_month INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS api_key TEXT;")
                # bancos antigos podem ter api_key NOT NULL; tentamos normalizar
                try:
                    cur.execute("ALTER TABLE clients ALTER COLUMN api_key DROP NOT NULL;")
                except Exception:
                    pass

                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'trial';")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS usage_month TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS leads_used_month INTEGER NOT NULL DEFAULT 0;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

                mk = _month_key()
                cur.execute("UPDATE clients SET usage_month=%s WHERE usage_month IS NULL OR usage_month='';", (mk,))
                cur.execute("UPDATE clients SET updated_at=NOW() WHERE updated_at IS NULL;")

                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_api_key ON clients(api_key) WHERE api_key IS NOT NULL;")

        app._schema_ready = True
        return True, None
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        app._schema_ready = False
        return False, repr(e)
    finally:
        conn.close()

def _ensure_schema_once():
    if getattr(app, "_schema_ready", False):
        return True, None
    return _ensure_schema()

@app.before_request
def _before_any_request():
    _ensure_schema_once()


# -------------------------
# Client helpers / Auth
# -------------------------
def _ensure_client_row(client_id: str, plan: str = "trial"):
    if plan not in PLAN_CATALOG:
        plan = "trial"
    mk = _month_key()

    def _work():
        conn = _db()
        try:
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    fresh_key = _gen_api_key(client_id)
                    cur.execute("""
                        INSERT INTO clients (client_id, api_key, plan, status, usage_month, leads_used_month, updated_at)
                        VALUES (%s, %s, %s, 'active', %s, 0, NOW())
                        ON CONFLICT (client_id) DO NOTHING
                    """, (client_id, fresh_key, plan, mk))

                    cur.execute("SELECT * FROM clients WHERE client_id=%s FOR UPDATE", (client_id,))
                    row = cur.fetchone()

                    if not (row.get("api_key") or "").strip():
                        cur.execute("UPDATE clients SET api_key=%s, updated_at=NOW() WHERE client_id=%s", (fresh_key, client_id))
                        cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                        row = cur.fetchone()

                    usage_month = (row.get("usage_month") or "").strip()
                    if usage_month != mk:
                        cur.execute(
                            "UPDATE clients SET usage_month=%s, leads_used_month=0, updated_at=NOW() WHERE client_id=%s",
                            (mk, client_id),
                        )
                        cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                        row = cur.fetchone()

                    return row
        finally:
            conn.close()

    try:
        return _work()
    except (psycopg2.errors.UndefinedColumn, psycopg2.errors.NotNullViolation):
        _ensure_schema()
        return _work()

def _require_client_auth(client_id: str):
    api_key = _get_api_key_from_headers()
    if not api_key:
        return False, None, "Missing X-API-KEY"

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM clients WHERE api_key=%s", (api_key,))
                row = cur.fetchone()
                if not row:
                    return False, None, "Invalid API key"
                if row.get("client_id") != client_id:
                    return False, None, "API key does not match client_id"
                if (row.get("status") or "active") != "active":
                    return False, None, "Client inactive"

                mk = _month_key()
                if (row.get("usage_month") or "") != mk:
                    cur.execute(
                        "UPDATE clients SET usage_month=%s, leads_used_month=0, updated_at=NOW() WHERE client_id=%s",
                        (mk, client_id),
                    )
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone()

                return True, row, None
    except (psycopg2.errors.UndefinedColumn, psycopg2.errors.NotNullViolation):
        _ensure_schema()
        return _require_client_auth(client_id)
    finally:
        conn.close()


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return jsonify({"ok": True, "service": "LeadRank backend", "ts": _iso(_now_utc())})

@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": _iso(_now_utc())})

@app.get("/health_db")
def health_db():
    if not DATABASE_URL:
        return jsonify({"ok": True, "db": False, "error": "DATABASE_URL missing", "ts": _iso(_now_utc())})
    ok, err = _ensure_schema_once()
    return jsonify({"ok": ok, "db": ok, "error": err, "ts": _iso(_now_utc())})

@app.post("/criar_cliente")
def criar_cliente():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "trial").strip().lower()

    if not client_id:
        return _resp({"ok": False, "error": "client_id obrigatório"}, 400)
    if plan not in PLAN_CATALOG:
        plan = "trial"

    row = _ensure_client_row(client_id, plan=plan)
    return _resp({"ok": True, "client_id": client_id, "api_key": row.get("api_key"), "plan": row.get("plan")})

@app.get("/client_meta")
def client_meta():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return _resp({"ok": False, "error": "client_id obrigatório"}, 400)

    ok_auth, row, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _resp({"ok": False, "error": msg}, 403)

    plan = (row.get("plan") or "trial").lower()
    cat = PLAN_CATALOG.get(plan, PLAN_CATALOG["trial"])

    return _resp({
        "ok": True,
        "client_id": client_id,
        "plan": plan,
        "status": row.get("status") or "active",
        "price_brl_month": cat["price_brl_month"],
        "lead_limit_month": cat["lead_limit_month"],
        "leads_used_this_month": int(row.get("leads_used_month") or 0),
        "usage_month": row.get("usage_month") or _month_key(),
        "ts": _iso(_now_utc()),
    })

@app.post("/prever")
def prever():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead = data.get("lead") or {}

    if not client_id:
        return _resp({"ok": False, "error": "client_id obrigatório"}, 400)

    ok_auth, client_row, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _resp({"ok": False, "error": msg}, 403)

    plan = (client_row.get("plan") or "trial").lower()
    cat = PLAN_CATALOG.get(plan, PLAN_CATALOG["trial"])
    used = int(client_row.get("leads_used_month") or 0)
    limit = int(cat["lead_limit_month"] or 0)

    if limit > 0 and used >= limit:
        return _resp({"ok": False, "code": "plan_limit", "plan": plan, "used": used, "limit": limit}, 402)

    nome = str(lead.get("nome") or "").strip().lower()
    tel = str(lead.get("telefone") or "").strip()
    origem = str(lead.get("origem") or "").strip().lower()

    base = 0.18
    if tel and len(tel) >= 10: base += 0.18
    if nome and len(nome) >= 4: base += 0.12
    if "google" in origem or "ads" in origem: base += 0.12
    prob = max(0.03, min(0.97, base + random.uniform(-0.06, 0.06)))

    score = int(round(prob * 100))
    label = 1 if prob >= 0.7 else (0 if prob < 0.35 else None)

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO leads (client_id, payload, probabilidade, score, label, created_at, updated_at)
                    VALUES (%s, %s::jsonb, %s, %s, %s, NOW(), NOW())
                    RETURNING id
                """, (client_id, json.dumps(lead), float(prob), int(score), label))
                lead_id = cur.fetchone()["id"]

                cur.execute(
                    "UPDATE clients SET leads_used_month = leads_used_month + 1, updated_at=NOW() WHERE client_id=%s",
                    (client_id,),
                )

        return _resp({
            "ok": True,
            "client_id": client_id,
            "lead_id": lead_id,
            "probabilidade": float(prob),
            "score": int(score),
            "label": label,
            "plan": plan,
        })
    except (psycopg2.errors.UndefinedColumn, psycopg2.errors.NotNullViolation):
        _ensure_schema()
        return prever()
    finally:
        conn.close()

@app.post("/set_plan")
def set_plan():
    if not _check_demo_key():
        return _resp({"ok": False, "error": "Unauthorized (DEMO_KEY)"}, 403)

    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "").strip().lower()
    status = (data.get("status") or "").strip().lower()

    if not client_id:
        return _resp({"ok": False, "error": "client_id obrigatório"}, 400)
    if plan and plan not in PLAN_CATALOG:
        return _resp({"ok": False, "error": "plan inválido"}, 400)
    if status and status not in ["active", "inactive"]:
        return _resp({"ok": False, "error": "status inválido"}, 400)

    _ensure_client_row(client_id, plan=plan or "trial")

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sets = []
                vals = []
                if plan:
                    sets.append("plan=%s"); vals.append(plan)
                if status:
                    sets.append("status=%s"); vals.append(status)
                sets.append("updated_at=NOW()")

                q = f"UPDATE clients SET {', '.join(sets)} WHERE client_id=%s"
                vals.append(client_id)
                cur.execute(q, tuple(vals))

                cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                row = cur.fetchone()

        return _resp({"ok": True, "client_id": client_id, "plan": row.get("plan"), "status": row.get("status")})
    except (psycopg2.errors.UndefinedColumn, psycopg2.errors.NotNullViolation):
        _ensure_schema()
        return set_plan()
    finally:
        conn.close()

@app.post("/demo_public")
def demo_public():
    ip = _client_ip()
    now = time.time()

    for k in list(_DEMO_RL.keys()):
        t0, c = _DEMO_RL[k]
        if now - t0 > _DEMO_RL_WINDOW_S:
            del _DEMO_RL[k]

    t0, c = _DEMO_RL.get(ip, (now, 0))
    if now - t0 > _DEMO_RL_WINDOW_S:
        t0, c = now, 0

    if c >= _DEMO_RL_MAX:
        return _resp({"ok": False, "error": "Rate limit demo. Tente mais tarde."}, 429)

    _DEMO_RL[ip] = (t0, c + 1)

    client_id = f"demo_{secrets.token_hex(2)}"
    row = _ensure_client_row(client_id, plan="demo")
    return _resp({"ok": True, "client_id": client_id, "api_key": row.get("api_key"), "plan": "demo"})


if __name__ == "__main__":
    try:
        _ensure_schema_once()
    except Exception as e:
        print("Schema ensure failed:", repr(e))
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
