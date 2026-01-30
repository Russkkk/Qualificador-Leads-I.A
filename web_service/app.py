# app.py - LeadRank (Render + Postgres) - versão completa e estável
# Stack: Flask + psycopg v3 (psycopg[binary])
# Endpoints usados pelo front (static site):
# - POST /signup
# - POST /prever
# - POST /seed_test_leads
# - GET  /dashboard_data?client_id=...
# - GET  /leads_export.csv?client_id=...&api_key=... (api_key opcional para download no browser)
# - POST /label  (rotular conversão/negação)
# - GET  /health_db

import os
import csv
import time
import secrets
import hashlib
import random
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple, Optional

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import psycopg
from psycopg.rows import dict_row

# -------------------------
# Config
# -------------------------
DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()

ALLOWED_ORIGINS = [
    "null",
    "https://qualificador-leads-ia.onrender.com",
    "https://qualificador-leads-i-a.onrender.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://127.0.0.1",
]

PLAN_CATALOG = {
    "trial":   {"price_brl_month": 0,   "lead_limit_month": 100},
    "starter": {"price_brl_month": 79,  "lead_limit_month": 1000},
    "pro":     {"price_brl_month": 179, "lead_limit_month": 5000},
    "vip":     {"price_brl_month": 279, "lead_limit_month": 20000},
}

# -------------------------
# App
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# -------------------------
# Utils
# -------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _month_key(dt: Optional[datetime] = None) -> str:
    dt = dt or _now_utc()
    return dt.strftime("%Y-%m")

def _require_env_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada no ambiente (Render)")

def _db():
    _require_env_db()
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _gen_api_key(client_id: str) -> str:
    raw = f"{client_id}:{secrets.token_urlsafe(24)}:{time.time()}"
    return "sk_live_" + _sha256(raw)[:32]

def _get_api_key_from_headers() -> str:
    key = (request.headers.get("X-API-KEY") or request.headers.get("Authorization") or "").strip()
    if key.lower().startswith("bearer "):
        key = key.split(" ", 1)[1].strip()
    return key

def _json_ok(payload: Dict[str, Any], code: int = 200):
    payload.setdefault("ok", True)
    return jsonify(payload), code

def _json_err(msg: str, code: int = 400, **extra):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return jsonify(payload), code

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def heuristic_prob(tempo_site: int, paginas: int, clicou_preco: int, nome: str, telefone: str) -> float:
    # Heurística simples (boa para demo). Depois você pode plugar modelo treinado.
    base = 0.10
    base += min(tempo_site / 400, 0.25)
    base += min(paginas / 10, 0.25)
    base += 0.20 if clicou_preco else 0.0
    if telefone and len(telefone) >= 10:
        base += 0.06
    if nome and len(nome) >= 4:
        base += 0.04
    return max(0.02, min(0.98, base))

# -------------------------
# Schema (auto)
# -------------------------
_SCHEMA_READY = False

def _ensure_schema_once() -> Tuple[bool, str]:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return True, ""
    try:
        _ensure_schema()
        _SCHEMA_READY = True
        return True, ""
    except Exception as e:
        return False, repr(e)

def _ensure_schema():
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Leads
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS leads (
                        id BIGSERIAL PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        nome TEXT,
                        email_lead TEXT,
                        telefone TEXT,
                        origem TEXT,
                        tempo_site INTEGER,
                        paginas_visitadas INTEGER,
                        clicou_preco INTEGER,
                        probabilidade DOUBLE PRECISION,
                        score INTEGER,
                        virou_cliente DOUBLE PRECISION,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_created ON leads(client_id, created_at DESC);")

                # Clients / workspaces
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS clients (
                        client_id TEXT PRIMARY KEY,
                        api_key TEXT,
                        plan TEXT NOT NULL DEFAULT 'trial',
                        status TEXT NOT NULL DEFAULT 'active',
                        usage_month TEXT NOT NULL DEFAULT '',
                        leads_used_month INTEGER NOT NULL DEFAULT 0,
                        nome TEXT,
                        email TEXT,
                        empresa TEXT,
                        telefone TEXT,
                        valid_until TIMESTAMPTZ,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                # Migrações compatíveis (se DB já existia)
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS api_key TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'trial';")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS usage_month TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS leads_used_month INTEGER NOT NULL DEFAULT 0;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS nome TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS email TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS empresa TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS telefone TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ;")
                try:
                    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_email_unique ON clients(email) WHERE email IS NOT NULL AND email<>'';")
                except Exception:
                    pass
                try:
                    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_api_key ON clients(api_key) WHERE api_key IS NOT NULL AND api_key<>'';")
                except Exception:
                    pass

                mk = _month_key()
                cur.execute("UPDATE clients SET usage_month=%s WHERE usage_month IS NULL OR usage_month='';", (mk,))
                cur.execute("UPDATE clients SET api_key='' WHERE api_key IS NULL;")

    finally:
        conn.close()

# -------------------------
# Auth / Quota
# -------------------------
def _get_client_row(client_id: str) -> Dict[str, Any]:
    _ensure_schema_once()
    mk = _month_key()

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO clients (client_id, api_key, plan, status, usage_month, leads_used_month, updated_at)
                    VALUES (%s, '', 'trial', 'active', %s, 0, NOW())
                    ON CONFLICT (client_id) DO NOTHING
                """, (client_id, mk))

                # lock para reset mensal
                cur.execute("SELECT * FROM clients WHERE client_id=%s FOR UPDATE", (client_id,))
                row = cur.fetchone() or {}

                if (row.get("usage_month") or "").strip() != mk:
                    cur.execute("UPDATE clients SET usage_month=%s, leads_used_month=0, updated_at=NOW() WHERE client_id=%s", (mk, client_id))
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row

                if row.get("api_key") is None:
                    cur.execute("UPDATE clients SET api_key='' WHERE client_id=%s", (client_id,))
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row

        return dict(row)
    finally:
        conn.close()

def _require_client_auth(client_id: str) -> Tuple[bool, Dict[str, Any], str]:
    row = _get_client_row(client_id)
    expected = (row.get("api_key") or "").strip()
    if not expected:
        # compat: se ainda não tem key, deixa passar
        return True, row, ""

    got = _get_api_key_from_headers()
    if not got:
        got = (request.args.get("api_key") or "").strip()  # permite CSV via navegador
    if not got:
        body = request.get_json(silent=True) or {}
        got = (body.get("api_key") or "").strip()

    if got != expected:
        return False, row, "api_key inválida ou ausente."
    return True, row, ""

def _check_quota_and_bump(client_id: str, row: Dict[str, Any], amount: int = 1) -> Tuple[bool, str, Dict[str, Any]]:
    plan = (row.get("plan") or "trial").strip().lower()
    meta = PLAN_CATALOG.get(plan, PLAN_CATALOG["trial"])
    used = int(row.get("leads_used_month") or 0)
    limit = int(meta.get("lead_limit_month") or 0)

    if limit > 0 and used + amount > limit:
        return False, "Limite mensal atingido. Faça upgrade para continuar.", {
            "code": "plan_limit",
            "plan": plan,
            "used": used,
            "limit": limit,
            "price_brl_month": meta.get("price_brl_month"),
        }

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE clients SET leads_used_month = leads_used_month + %s, updated_at=NOW() WHERE client_id=%s",
                    (int(amount), client_id),
                )
        return True, "", {}
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

@app.get("/pricing")
def pricing():
    return _json_ok({"plans": PLAN_CATALOG, "currency": "BRL", "ts": _iso(_now_utc())})

@app.post("/signup")
def signup():
    data = request.get_json(silent=True) or request.form or {}
    nome = (data.get("nome") or "").strip()
    email = (data.get("email") or "").strip().lower()
    empresa = (data.get("empresa") or "").strip()
    telefone = (data.get("telefone") or "").strip()

    if not email or "@" not in email:
        return _json_err("Email válido é obrigatório", 400)

    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT client_id FROM clients WHERE email=%s", (email,))
                if cur.fetchone():
                    return _json_err("Este email já está cadastrado", 409)

                client_id = f"trial-{secrets.token_hex(8)}"
                api_key = _gen_api_key(client_id)
                mk = _month_key()
                valid_until = _now_utc() + timedelta(days=14)

                cur.execute("""
                    INSERT INTO clients
                      (client_id, api_key, plan, status, usage_month, leads_used_month,
                       nome, email, empresa, telefone, valid_until, created_at, updated_at)
                    VALUES
                      (%s,%s,'trial','active',%s,0,%s,%s,%s,%s,%s,NOW(),NOW())
                """, (client_id, api_key, mk, nome or None, email, empresa or None, telefone or None, valid_until))

        return _json_ok({
            "client_id": client_id,
            "api_key": api_key,
            "plan": "trial",
            "valid_until": _iso(valid_until),
        })
    finally:
        conn.close()

@app.post("/prever")
def prever():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok, client_row, err = _require_client_auth(client_id)
    if not ok:
        return _json_err(err, 401)

    nome = (data.get("nome") or "").strip()
    email_lead = (data.get("email_lead") or data.get("email") or "").strip().lower()
    telefone = (data.get("telefone") or "").strip()
    origem = (data.get("origem") or "").strip()
    tempo_site = _safe_int(data.get("tempo_site"), 0)
    paginas_visitadas = _safe_int(data.get("paginas_visitadas"), 0)
    clicou_preco = 1 if _safe_int(data.get("clicou_preco"), 0) else 0

    prob = heuristic_prob(tempo_site, paginas_visitadas, clicou_preco, nome, telefone)
    score = int(round(prob * 100))

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO leads
                      (client_id, nome, email_lead, telefone, origem, tempo_site, paginas_visitadas, clicou_preco,
                       probabilidade, score, created_at, updated_at)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
                    RETURNING id
                """, (client_id, nome or None, email_lead or None, telefone or None, origem or None,
                      tempo_site, paginas_visitadas, clicou_preco, float(prob), int(score)))
                lead_id = int(cur.fetchone()["id"])

        okq, msg, extra = _check_quota_and_bump(client_id, client_row, amount=1)
        if not okq:
            return _json_err(msg, 402, **extra)

        return _json_ok({
            "id": lead_id,
            "score": score,
            "probabilidade": round(float(prob), 4),
        })
    finally:
        conn.close()

@app.post("/seed_test_leads")
def seed_test_leads():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    count = _safe_int(data.get("count"), 10)
    count = max(1, min(50, count))  # evita abuso

    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok, client_row, err = _require_client_auth(client_id)
    if not ok:
        return _json_err(err, 401)

    # quota (reserva)
    okq, msg, extra = _check_quota_and_bump(client_id, client_row, amount=count)
    if not okq:
        return _json_err(msg, 402, **extra)

    rows = []
    for i in range(count):
        tempo_site = random.randint(10, 520)
        paginas = random.randint(1, 12)
        clicou = random.choice([0, 1])
        nome = "Lead Teste " + secrets.token_hex(2).upper()
        email = f"teste{secrets.token_hex(3)}@leadrank.local"
        telefone = "11" + str(random.randint(900000000, 999999999))
        origem = random.choice(["google", "instagram", "whatsapp", "indicacao", "desconhecida"])
        prob = heuristic_prob(tempo_site, paginas, clicou, nome, telefone)
        score = int(round(prob * 100))
        rows.append((client_id, nome, email, telefone, origem, tempo_site, paginas, clicou, float(prob), score))

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO leads
                      (client_id, nome, email_lead, telefone, origem, tempo_site, paginas_visitadas, clicou_preco,
                       probabilidade, score, created_at, updated_at)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
                """, rows)
        return _json_ok({"inserted": count})
    finally:
        conn.close()

@app.get("/dashboard_data")
def dashboard_data():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório (querystring)", 400)

    ok, _, err = _require_client_auth(client_id)
    if not ok:
        return _json_err(err, 401)

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, nome, email_lead, telefone, origem, tempo_site, paginas_visitadas, clicou_preco,
                           probabilidade, score, virou_cliente, created_at
                    FROM leads
                    WHERE client_id=%s
                    ORDER BY created_at DESC
                    LIMIT 200
                """, (client_id,))
                leads = [dict(r) for r in (cur.fetchall() or [])]

                for r in leads:
                    r["created_at"] = _iso(r.get("created_at"))

                convertidos = sum(1 for r in leads if r.get("virou_cliente") in (1, 1.0))
                negados = sum(1 for r in leads if r.get("virou_cliente") in (0, 0.0))
                pendentes = len(leads) - convertidos - negados

                # top origens 30d
                cur.execute("""
                    SELECT COALESCE(NULLIF(TRIM(origem), ''), 'desconhecida') AS origem,
                           COUNT(*)::int AS total
                    FROM leads
                    WHERE client_id=%s AND created_at >= (NOW() - INTERVAL '30 days')
                    GROUP BY 1
                    ORDER BY total DESC, origem ASC
                    LIMIT 6
                """, (client_id,))
                top_origens = cur.fetchall() or []

                # hot leads hoje (UTC window simples)
                cur.execute("""
                    SELECT id, nome, origem, probabilidade, score, created_at
                    FROM leads
                    WHERE client_id=%s AND created_at >= (NOW() - INTERVAL '1 day')
                      AND ( (probabilidade IS NOT NULL AND probabilidade >= 0.70) OR (score IS NOT NULL AND score >= 70) )
                    ORDER BY COALESCE(probabilidade, score/100.0) DESC NULLS LAST, created_at DESC
                    LIMIT 20
                """, (client_id,))
                hot = [dict(r) for r in (cur.fetchall() or [])]
                for r in hot:
                    r["created_at"] = _iso(r.get("created_at"))

        return _json_ok({
            "convertidos": convertidos,
            "negados": negados,
            "pendentes": pendentes,
            "top_origens_30d": top_origens,
            "hot_leads_today": hot,
            "dados": leads,
        })
    finally:
        conn.close()

@app.get("/leads_export.csv")
def leads_export_csv():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório (querystring)", 400)

    ok, _, err = _require_client_auth(client_id)
    if not ok:
        return _json_err(err, 401)

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, created_at, nome, email_lead, telefone, origem, tempo_site, paginas_visitadas,
                           clicou_preco, probabilidade, score, virou_cliente
                    FROM leads
                    WHERE client_id=%s
                    ORDER BY created_at DESC
                    LIMIT 5000
                """, (client_id,))
                rows = cur.fetchall() or []

        def generate():
            import io
            buf = io.StringIO()
            w = csv.writer(buf)

            # header
            w.writerow([
                "id","created_at","nome","email_lead","telefone","origem",
                "tempo_site","paginas_visitadas","clicou_preco","probabilidade","score","virou_cliente"
            ])
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)

            for r in rows:
                w.writerow([
                    r.get("id"),
                    _iso(r.get("created_at")),
                    r.get("nome") or "",
                    r.get("email_lead") or "",
                    r.get("telefone") or "",
                    r.get("origem") or "",
                    r.get("tempo_site") or 0,
                    r.get("paginas_visitadas") or 0,
                    r.get("clicou_preco") or 0,
                    r.get("probabilidade") if r.get("probabilidade") is not None else "",
                    r.get("score") if r.get("score") is not None else "",
                    r.get("virou_cliente") if r.get("virou_cliente") is not None else "",
                ])
                yield buf.getvalue()
                buf.seek(0); buf.truncate(0)

        return Response(generate(), mimetype="text/csv", headers={
            "Content-Disposition": f'attachment; filename="leads_{client_id}.csv"'
        })
    finally:
        conn.close()

@app.post("/label")
def label():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = _safe_int(data.get("id"), 0)
    virou = data.get("virou_cliente", None)

    if not client_id or not lead_id:
        return _json_err("client_id e id são obrigatórios", 400)

    ok, _, err = _require_client_auth(client_id)
    if not ok:
        return _json_err(err, 401)

    if virou not in (0, 1, 0.0, 1.0, None):
        return _json_err("virou_cliente deve ser 0, 1 ou null", 400)

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE leads SET virou_cliente=%s, updated_at=NOW() WHERE client_id=%s AND id=%s",
                    (None if virou is None else float(virou), client_id, lead_id),
                )
        return _json_ok({"updated": True})
    finally:
        conn.close()
