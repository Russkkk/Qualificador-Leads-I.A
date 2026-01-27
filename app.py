# app.py - LeadRank / Qualificador de Leads IA (Render + Postgres)
# Unified build (merge of "v9 1098" + "saas migration" branch):
# - Auto-migrations robustas (leads + clients) para bancos antigos
# - Planos + uso mensal (reset automático por usage_month)
# - Auth por X-API-KEY (quando api_key existe)
# - Endpoints de dashboard/insights/treino (LogReg opcional)
# - Export CSV server-side
#
# Requisitos: psycopg2-binary, flask, flask-cors
# (Opcional): numpy + scikit-learn para /recalc_pending e /auto_threshold

import os
import json
import time
import random
import string
import secrets
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import psycopg2
from psycopg2.extras import RealDictCursor

# optional ML deps (se não existirem, rotas de treino respondem com erro amigável)
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _HAS_ML = True
except Exception:
    _HAS_ML = False


# =========================
# Config
# =========================
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DEMO_KEY = os.environ.get("DEMO_KEY", "").strip()

# ajuste aqui seus domínios permitidos no CORS
ALLOWED_ORIGINS = [
    "https://qualificador-leads-ia.onrender.com",   # Static Site
    "https://qualificador-leads-i-a.onrender.com",  # Web Service (se chamar a si mesmo)
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://127.0.0.1",
]

PLAN_CATALOG = {
    # Campos:
    # - price_brl_month: mensalidade
    # - setup_fee_brl: taxa de setup (0 por padrão; você pode ajustar depois)
    # - lead_limit_month: limite de leads/mês
    "demo":    {"price_brl_month": 0,   "setup_fee_brl": 0, "lead_limit_month": 30},
    "trial":   {"price_brl_month": 0,   "setup_fee_brl": 0, "lead_limit_month": 100},
    "starter": {"price_brl_month": 79,  "setup_fee_brl": 0, "lead_limit_month": 1000},
    "pro":     {"price_brl_month": 179, "setup_fee_brl": 0, "lead_limit_month": 5000},
    "vip":     {"price_brl_month": 279, "setup_fee_brl": 0, "lead_limit_month": 20000},
}

DEFAULT_LIMIT = 200
DEFAULT_THRESHOLD = 0.35
MIN_LABELED_TO_TRAIN = 4

# rate-limit simples de demo pública (por IP/mês)
_DEMO_RL: Dict[str, int] = {}
_DEMO_RL_MAX = 5


# =========================
# App
# =========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})


# =========================
# Utils
# =========================
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

from zoneinfo import ZoneInfo

_SP_TZ = ZoneInfo("America/Sao_Paulo")

def _sp_today_bounds_utc() -> tuple[datetime, datetime]:
    """Retorna (inicio_utc, fim_utc) do dia de hoje em America/Sao_Paulo."""
    now_sp = datetime.now(_SP_TZ)
    start_sp = now_sp.replace(hour=0, minute=0, second=0, microsecond=0)
    end_sp = start_sp.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start_sp.astimezone(timezone.utc), end_sp.astimezone(timezone.utc)

def _top_origens(client_id: str, days: int = 30, limit: int = 6):
    """Top origens (últimos N dias) por quantidade de leads."""
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT COALESCE(NULLIF(TRIM(origem), ''), 'desconhecida') AS origem,
                           COUNT(*)::int AS total
                    FROM leads
                    WHERE client_id=%s
                      AND created_at >= (NOW() - (%s || ' days')::interval)
                    GROUP BY 1
                    ORDER BY total DESC, origem ASC
                    LIMIT %s
                    """,
                    (client_id, int(days), int(limit)),
                )
                return cur.fetchall()
    finally:
        conn.close()

def _hot_leads_today(client_id: str, limit: int = 20):
    """Leads quentes de hoje (probabilidade>=0.70 ou score>=70)."""
    start_utc, end_utc = _sp_today_bounds_utc()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, nome, telefone, email_lead, origem,
                           probabilidade, score, created_at, virou_cliente
                    FROM leads
                    WHERE client_id=%s
                      AND created_at >= %s AND created_at <= %s
                      AND (
                            (probabilidade IS NOT NULL AND probabilidade >= 0.70)
                            OR (score IS NOT NULL AND score >= 70)
                          )
                    ORDER BY COALESCE(probabilidade, score/100.0) DESC NULLS LAST,
                             created_at DESC
                    LIMIT %s
                    """,
                    (client_id, start_utc, end_utc, int(limit)),
                )
                rows = cur.fetchall()
                for r in rows:
                    r["created_at"] = _iso(r.get("created_at"))
                return rows
    finally:
        conn.close()

def _resp(payload: Dict[str, Any], code: int = 200):
    return jsonify(payload), code

def _json_ok(payload: Dict[str, Any], code: int = 200):
    payload.setdefault("ok", True)
    return _resp(payload, code)

def _json_err(msg: str, code: int = 400, **extra):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return _resp(payload, code)

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def _require_env_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada (Render Environment)")

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

def _get_api_key_from_headers() -> str:
    key = _get_header("X-API-KEY") or _get_header("Authorization")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key

def _client_ip() -> str:
    return (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0].strip()

def _check_demo_key() -> bool:
    expected = (os.getenv("DEMO_KEY") or "").strip()
    got = _get_header("X-DEMO-KEY")
    return bool(expected) and got == expected


# =========================
# Schema / Migrations (auto)
# =========================
_SCHEMA_READY = False
_SCHEMA_LOCK = None  # lazy lock


def _ensure_schema_once() -> Tuple[bool, str]:
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
            _ensure_schema()
            _SCHEMA_READY = True
            return True, ""
        except Exception as e:
            return False, repr(e)


def _ensure_schema():
    """
    Cria/migra tabelas para ser compatível com:
    - versões antigas com leads em colunas "nome/email/telefone/..."
    - versões novas com payload JSONB, score/label/updated_at
    - clients antigos com api_key NOT NULL (corrigimos)
    """
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                # -------------------------
                # LEADS
                # -------------------------
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS leads (
                        id BIGSERIAL PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        -- colunas "clássicas" (dashboard atual)
                        nome TEXT,
                        email_lead TEXT,
                        telefone TEXT,
                        origem TEXT,
                        tempo_site INTEGER,
                        paginas_visitadas INTEGER,
                        clicou_preco INTEGER,
                        probabilidade DOUBLE PRECISION,
                        virou_cliente DOUBLE PRECISION,
                        -- colunas "SaaS" (futuro)
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        score INTEGER,
                        label INTEGER,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_created ON leads(client_id, created_at DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_label ON leads(client_id, virou_cliente);")

                # migrações (se leads já existia)
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS client_id TEXT;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS nome TEXT;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS email_lead TEXT;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS telefone TEXT;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS origem TEXT;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS tempo_site INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS paginas_visitadas INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS clicou_preco INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS probabilidade DOUBLE PRECISION;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS virou_cliente DOUBLE PRECISION;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS score INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS label INTEGER;")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
                cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

                # -------------------------
                # CLIENTS
                # -------------------------
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
                # bancos antigos podem ter api_key NOT NULL -> drop
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
                # normaliza NULLs antigos
                cur.execute("UPDATE clients SET usage_month=%s WHERE usage_month IS NULL OR usage_month='';", (mk,))
                cur.execute("UPDATE clients SET api_key='' WHERE api_key IS NULL;")
                cur.execute("UPDATE clients SET updated_at=NOW() WHERE updated_at IS NULL;")

                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_api_key ON clients(api_key) WHERE api_key <> '';")

                # -------------------------
                # THRESHOLDS / MODEL_META (para insights/treino)
                # -------------------------
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS thresholds (
                        client_id TEXT PRIMARY KEY,
                        threshold DOUBLE PRECISION NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_meta (
                        client_id TEXT PRIMARY KEY,
                        can_train BOOLEAN NOT NULL DEFAULT FALSE,
                        labeled_count INTEGER NOT NULL DEFAULT 0,
                        classes_rotuladas TEXT NOT NULL DEFAULT '[]',
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

    finally:
        conn.close()


# tenta migrar na subida (sem derrubar se DB ainda não está pronto)
try:
    _ensure_schema_once()
except Exception:
    pass


# =========================
# Clients / Auth / Quota
# =========================
def _ensure_client_row(client_id: str, plan: str = "trial") -> Dict[str, Any]:
    """Garante row em clients. Reseta contagem mensal quando muda o mês."""
    _ensure_schema_once()

    plan = (plan or "trial").strip().lower()
    if plan not in PLAN_CATALOG:
        plan = "trial"

    mk = _month_key()

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # cria se não existir
                cur.execute("""
                    INSERT INTO clients (client_id, api_key, plan, status, usage_month, leads_used_month, updated_at)
                    VALUES (%s, '', %s, 'active', %s, 0, NOW())
                    ON CONFLICT (client_id) DO NOTHING
                """, (client_id, plan, mk))

                # lock + reset mensal
                cur.execute("SELECT * FROM clients WHERE client_id=%s FOR UPDATE", (client_id,))
                row = cur.fetchone() or {}

                if (row.get("usage_month") or "").strip() != mk:
                    cur.execute(
                        "UPDATE clients SET usage_month=%s, leads_used_month=0, updated_at=NOW() WHERE client_id=%s",
                        (mk, client_id),
                    )
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row

                # garante api_key não nulo (evita NotNullViolation em bancos antigos)
                if row.get("api_key") is None:
                    cur.execute("UPDATE clients SET api_key='' WHERE client_id=%s", (client_id,))
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row

        return dict(row)
    finally:
        conn.close()


def _require_client_auth(client_id: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Regra:
    - se o client não tem api_key (vazio), aceita sem header (compatibilidade)
    - se tem api_key, exige X-API-KEY ou Authorization: Bearer
    """
    row = _ensure_client_row(client_id, plan="trial")
    expected = (row.get("api_key") or "").strip()
    if not expected:
        return True, row, ""

    got = _get_api_key_from_headers()
    if not got:
        data = request.get_json(silent=True) or {}
        got = (data.get("api_key") or "").strip()

    if got != expected:
        return False, row, "api_key inválida ou ausente."
    return True, row, ""


def _check_quota_and_bump(client_id: str, client_row: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Valida limite do plano e incrementa leads_used_month (chamar após inserir lead)."""
    plan = (client_row.get("plan") or "trial").strip().lower()
    meta = PLAN_CATALOG.get(plan, PLAN_CATALOG["trial"])
    used = int(client_row.get("leads_used_month") or 0)
    limit = int(meta.get("lead_limit_month") or 0)

    if limit > 0 and used >= limit:
        return False, "Limite mensal atingido. Faça upgrade para continuar.", {
            "code": "plan_limit",
            "plan": plan,
            "used": used,
            "limit": limit,
            "price_brl_month": meta.get("price_brl_month"),
            "setup_fee_brl": meta.get("setup_fee_brl", 0),
        }

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE clients SET leads_used_month = leads_used_month + 1, updated_at=NOW() WHERE client_id=%s",
                    (client_id,),
                )
        return True, "", {}
    finally:
        conn.close()


# =========================
# Threshold helpers
# =========================
def _get_threshold(client_id: str) -> float:
    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT threshold FROM thresholds WHERE client_id=%s", (client_id,))
                row = cur.fetchone()
                if row and row.get("threshold") is not None:
                    return float(row["threshold"])
        return DEFAULT_THRESHOLD
    finally:
        conn.close()

def _set_threshold(client_id: str, threshold: float):
    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO thresholds (client_id, threshold, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (client_id)
                    DO UPDATE SET threshold=EXCLUDED.threshold, updated_at=NOW()
                    """,
                    (client_id, float(threshold)),
                )
    finally:
        conn.close()


# =========================
# Lead fetch / ML helpers
# =========================
def _fetch_recent_leads(client_id: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                           probabilidade, virou_cliente, created_at
                    FROM leads
                    WHERE client_id=%s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (client_id, int(limit)),
                )
                return [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

def _count_status(rows: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    convertidos = sum(1 for r in rows if r.get("virou_cliente") in (1, 1.0))
    negados = sum(1 for r in rows if r.get("virou_cliente") in (0, 0.0))
    pendentes = len(rows) - convertidos - negados
    return convertidos, negados, pendentes

# ML helpers (apenas se libs existem)
def _features_from_row(r: Dict[str, Any]):
    tempo = _safe_int(r.get("tempo_site"), 0)
    paginas = _safe_int(r.get("paginas_visitadas"), 0)
    clicou = _safe_int(r.get("clicou_preco"), 0)
    return np.array([tempo, paginas, clicou], dtype=float)

def _train_pipeline(X, y):
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200, solver="lbfgs"))])
    pipe.fit(X, y)
    return pipe

def _get_labeled_rows(client_id: str) -> List[Dict[str, Any]]:
    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, tempo_site, paginas_visitadas, clicou_preco, probabilidade, virou_cliente
                    FROM leads
                    WHERE client_id=%s AND virou_cliente IS NOT NULL
                    ORDER BY created_at DESC
                    """,
                    (client_id,),
                )
                return [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

def _can_train(labeled_rows: List[Dict[str, Any]]) -> Tuple[bool, str, List[float]]:
    if len(labeled_rows) < MIN_LABELED_TO_TRAIN:
        classes = sorted(list({float(r["virou_cliente"]) for r in labeled_rows if r.get("virou_cliente") is not None}))
        return False, f"Poucos exemplos rotulados. Recomendo no mínimo {MIN_LABELED_TO_TRAIN} (2 de cada classe) para começar.", classes
    classes = sorted(list({float(r["virou_cliente"]) for r in labeled_rows if r.get("virou_cliente") is not None}))
    if len(classes) < 2:
        return False, "Precisa de exemplos das duas classes (convertido e negado) para treinar.", classes
    return True, "", classes

def _predict_for_rows(pipe, rows: List[Dict[str, Any]]) -> List[float]:
    if not rows:
        return []
    X = np.vstack([_features_from_row(r) for r in rows])
    probs = pipe.predict_proba(X)[:, 1]
    return probs.tolist()

def _update_probabilities(client_id: str, ids: List[int], probs: List[float]) -> int:
    if not ids:
        return 0
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                for lead_id, p in zip(ids, probs):
                    cur.execute(
                        "UPDATE leads SET probabilidade=%s, updated_at=NOW() WHERE client_id=%s AND id=%s",
                        (float(p), client_id, int(lead_id)),
                    )
        return len(ids)
    finally:
        conn.close()

def _compute_precision_recall(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    y_true = []
    y_pred = []
    for r in rows:
        y = r.get("virou_cliente")
        p = r.get("probabilidade")
        if y is None or p is None:
            continue
        y_true.append(1 if float(y) == 1.0 else 0)
        y_pred.append(1 if float(p) >= threshold else 0)

    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

def _best_threshold(rows: List[Dict[str, Any]]) -> float:
    candidates = [i / 100 for i in range(5, 96, 5)]
    best_t = DEFAULT_THRESHOLD
    best_f1 = -1.0
    for t in candidates:
        m = _compute_precision_recall(rows, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
    return float(best_t)


# =========================
# Routes
# =========================
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
    """Retorna o catálogo de planos (para UI)."""
    return _json_ok({
        "plans": PLAN_CATALOG,
        "currency": "BRL",
        "checkout_enabled": bool(STRIPE_SECRET_KEY and STRIPE_PRICE_IDS_JSON),
        "ts": _iso(_now_utc()),
    })

@app.post("/criar_cliente")
def criar_cliente():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "trial").strip().lower()

    if not client_id:
        return _json_err("client_id obrigatório", 400)
    if plan not in PLAN_CATALOG:
        plan = "trial"

    _ensure_schema_once()
    row = _ensure_client_row(client_id, plan=plan)

    # emite api_key se ainda não existe
    api_key = (row.get("api_key") or "").strip()
    if not api_key:
        api_key = _gen_api_key(client_id)
        conn = _db()
        try:
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "UPDATE clients SET api_key=%s, plan=%s, updated_at=NOW() WHERE client_id=%s",
                        (api_key, plan, client_id),
                    )
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row
        finally:
            conn.close()

    meta = PLAN_CATALOG.get((row.get("plan") or plan).lower(), PLAN_CATALOG["trial"])
    return _json_ok({
        "client_id": client_id,
        "api_key": api_key,
        "plan": (row.get("plan") or plan),
        "price_brl_month": meta["price_brl_month"],
        "setup_fee_brl": meta.get("setup_fee_brl", 0),
        "lead_limit_month": meta["lead_limit_month"],
    })

@app.get("/client_meta")
def client_meta():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, row, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    plan = (row.get("plan") or "trial").lower()
    cat = PLAN_CATALOG.get(plan, PLAN_CATALOG["trial"])

    return _json_ok({
        "client_id": client_id,
        "plan": plan,
        "status": row.get("status") or "active",
        "price_brl_month": cat["price_brl_month"],
        "setup_fee_brl": cat.get("setup_fee_brl", 0),
        "lead_limit_month": cat["lead_limit_month"],
        "leads_used_this_month": int(row.get("leads_used_month") or 0),
        "usage_month": row.get("usage_month") or _month_key(),
        "ts": _iso(_now_utc()),
    })

@app.post("/set_plan")
def set_plan():
    """Admin: ajusta plano/status de um client_id. Protegido por DEMO_KEY."""
    if not _check_demo_key():
        return _json_err("Unauthorized (DEMO_KEY)", 403)

    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "").strip().lower()
    status = (data.get("status") or "").strip().lower()

    if not client_id:
        return _json_err("client_id obrigatório", 400)
    if plan and plan not in PLAN_CATALOG:
        return _json_err("plan inválido", 400, allowed=list(PLAN_CATALOG.keys()))
    if status and status not in ["active", "inactive"]:
        return _json_err("status inválido", 400, allowed=["active", "inactive"])

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
                row = cur.fetchone() or {}
        return _json_ok({"client_id": client_id, "plan": row.get("plan"), "status": row.get("status")})
    finally:
        conn.close()

@app.post("/prever")
def prever():
    """
    POST /prever
    Body:
      {
        "client_id": "workspace",
        "lead": {...}          # opcional (formato "SaaS")
        "nome": "...",         # opcional (formato legado)
        "email_lead": "...",
        "telefone": "...",
        "tempo_site": 120,
        "paginas_visitadas": 5,
        "clicou_preco": 1
      }
    """
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, client_row, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    if (client_row.get("status") or "active") != "active":
        return _json_err("Workspace inativo. Fale com o suporte para reativar.", 403, code="inactive")

    plan = (client_row.get("plan") or "trial").lower()
    cat = PLAN_CATALOG.get(plan, PLAN_CATALOG["trial"])
    used = int(client_row.get("leads_used_month") or 0)
    limit = int(cat.get("lead_limit_month") or 0)
    if limit > 0 and used >= limit:
        return _json_err("Limite mensal atingido. Faça upgrade para continuar.", 402,
                         code="plan_limit", plan=plan, used=used, limit=limit,
                         price_brl_month=cat.get("price_brl_month"), setup_fee_brl=cat.get("setup_fee_brl", 0))

    # aceita ambos formatos
    lead = data.get("lead") or {}
    nome = (data.get("nome") or lead.get("nome") or "").strip()
    email = (data.get("email_lead") or data.get("email") or lead.get("email_lead") or lead.get("email") or "").strip()
    telefone = (data.get("telefone") or lead.get("telefone") or "").strip()

    origem = (data.get("origem") or lead.get("origem") or lead.get("source") or "").strip().lower()
    if not origem:
        origem = "desconhecida"

    tempo_site = _safe_int(data.get("tempo_site") if "tempo_site" in data else lead.get("tempo_site"), 0)
    paginas_visitadas = _safe_int(data.get("paginas_visitadas") if "paginas_visitadas" in data else lead.get("paginas_visitadas"), 0)
    clicou_preco = _safe_int(data.get("clicou_preco") if "clicou_preco" in data else lead.get("clicou_preco"), 0)

    # heurística inicial (estável e barata)
    base = 0.10
    base += min(tempo_site / 400, 0.25)
    base += min(paginas_visitadas / 10, 0.25)
    base += 0.20 if clicou_preco else 0.0
    # upgrades sutis (melhora "qualidade percebida" sem inventar dados)
    if telefone and len(telefone) >= 10:
        base += 0.06
    if nome and len(nome) >= 4:
        base += 0.04

    prob = max(0.02, min(0.98, base))
    score = int(round(prob * 100))
    label = 1 if prob >= 0.70 else (0 if prob < 0.35 else None)

    payload = lead if isinstance(lead, dict) else {}
    # também salva o formato legado dentro do payload para auditoria/CRM
    payload.setdefault("nome", nome)
    payload.setdefault("email", email)
    payload.setdefault("email_lead", email)
    payload.setdefault("telefone", telefone)
    payload.setdefault("origem", origem)
    payload.setdefault("tempo_site", tempo_site)
    payload.setdefault("paginas_visitadas", paginas_visitadas)
    payload.setdefault("clicou_preco", clicou_preco)

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO leads
                      (client_id, nome, email_lead, telefone, origem, tempo_site, paginas_visitadas, clicou_preco,
                       payload, probabilidade, score, label, virou_cliente, created_at, updated_at)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,NULL,NOW(),NOW())
                    RETURNING id, created_at
                    """,
                    (client_id, nome, email, telefone, origem, tempo_site, paginas_visitadas, clicou_preco,
                     json.dumps(payload), float(prob), int(score), label),
                )
                row = cur.fetchone() or {}
                ok, err, extra = _check_quota_and_bump(client_id, client_row)
                if not ok:
                    # se estourou limite exatamente aqui, mantemos a inserção (você decide se quer rollback).
                    # Para comportamento estrito, dá para lançar exceção e reverter.
                    return _json_err(err, 402, **extra)

        return _json_ok({
            "client_id": client_id,
            "lead_id": int(row.get("id") or 0),
            "probabilidade": float(prob),
            "score": int(score),
            "label": label,
            "plan": plan,
            "created_at": _iso(row.get("created_at")),
        })
    except (psycopg2.errors.UndefinedColumn, psycopg2.errors.NotNullViolation):
        # segurança extra: se banco estiver em versão antiga, migra e tenta de novo
        _ensure_schema()
        return prever()
    finally:
        conn.close()


@app.get("/dashboard_data")
def dashboard_data():
    client_id = (request.args.get("client_id") or "").strip()
    limit = _safe_int(request.args.get("limit"), DEFAULT_LIMIT)
    limit = max(10, min(limit, 1000))

    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    rows = _fetch_recent_leads(client_id, limit=limit)
    convertidos, negados, pendentes = _count_status(rows)

    # Premium (C1): Top origens (30d) + Hot leads de hoje (America/Sao_Paulo)
    top_origens = _top_origens(client_id, days=30, limit=6)
    hot_leads_today = _hot_leads_today(client_id, limit=20)

    def norm(r: Dict[str, Any]) -> Dict[str, Any]:
        rr = dict(r)
        rr["created_at"] = _iso(rr.get("created_at"))
        return rr

    return _json_ok({
        "client_id": client_id,
        "convertidos": convertidos,
        "negados": negados,
        "pendentes": pendentes,
        "top_origens_30d": top_origens,
        "hot_leads_today": hot_leads_today,
        "hot_leads_today_tz": "America/Sao_Paulo",
        "dados": [norm(r) for r in rows],
        "total_recentes_considerados": len(rows),
    })


@app.post("/confirmar_venda")
def confirmar_venda():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = _safe_int(data.get("lead_id"), 0)
    if not client_id or not lead_id:
        return _json_err("client_id e lead_id obrigatórios", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE leads SET virou_cliente=1, updated_at=NOW() WHERE client_id=%s AND id=%s",
                    (client_id, lead_id),
                )
        return _json_ok({"client_id": client_id, "lead_id": lead_id, "virou_cliente": 1})
    finally:
        conn.close()


@app.post("/negar_venda")
def negar_venda():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = _safe_int(data.get("lead_id"), 0)
    if not client_id or not lead_id:
        return _json_err("client_id e lead_id obrigatórios", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE leads SET virou_cliente=0, updated_at=NOW() WHERE client_id=%s AND id=%s",
                    (client_id, lead_id),
                )
        return _json_ok({"client_id": client_id, "lead_id": lead_id, "virou_cliente": 0})
    finally:
        conn.close()


@app.get("/metrics")
def metrics():
    """Métricas simples (debug/monitoramento)."""
    if not DATABASE_URL:
        return _json_ok({"db": False, "reason": "DATABASE_URL ausente", "ts": _iso(_now_utc())})

    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT COUNT(*) AS total FROM leads;")
                total = int(cur.fetchone()["total"])
                cur.execute("SELECT COUNT(*) AS labeled FROM leads WHERE virou_cliente IS NOT NULL;")
                labeled = int(cur.fetchone()["labeled"])
                cur.execute("SELECT COUNT(*) AS pending FROM leads WHERE virou_cliente IS NULL;")
                pending = int(cur.fetchone()["pending"])
        return _json_ok({"db": True, "total_leads": total, "labeled": labeled, "pending": pending, "ts": _iso(_now_utc())})
    finally:
        conn.close()


@app.get("/recalc_pending")
def recalc_pending():
    """Recalcula probabilidade para pendentes com base nos rotulados (requer numpy/sklearn)."""
    if not _HAS_ML:
        return _json_err("Dependências ML ausentes (numpy/scikit-learn).", 501, code="ml_missing")

    client_id = (request.args.get("client_id") or "").strip()
    limit = _safe_int(request.args.get("limit"), 500)
    limit = max(10, min(limit, 5000))
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    labeled = _get_labeled_rows(client_id)
    can, reason, classes = _can_train(labeled)
    if not can:
        return _json_ok({"client_id": client_id, "can_train": False, "classes_rotuladas": classes, "labeled_count": len(labeled), "reason": reason, "updated": 0})

    X = np.vstack([_features_from_row(r) for r in labeled])
    y = np.array([1 if float(r["virou_cliente"]) == 1.0 else 0 for r in labeled], dtype=int)
    pipe = _train_pipeline(X, y)

    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, tempo_site, paginas_visitadas, clicou_preco
                    FROM leads
                    WHERE client_id=%s AND virou_cliente IS NULL
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (client_id, int(limit)),
                )
                pending = [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

    ids = [int(r["id"]) for r in pending]
    probs = _predict_for_rows(pipe, pending)
    updated = _update_probabilities(client_id, ids, probs)

    return _json_ok({
        "client_id": client_id,
        "can_train": True,
        "classes_rotuladas": classes,
        "labeled_count": len(labeled),
        "updated": updated,
        "min_prob": float(min(probs)) if probs else None,
        "max_prob": float(max(probs)) if probs else None,
        "sample": [{"id": ids[i], "prob": float(probs[i])} for i in range(min(5, len(ids)))]
    })


@app.post("/auto_threshold")
def auto_threshold():
    """Calcula e salva threshold que maximiza F1 (requer numpy/sklearn)."""
    if not _HAS_ML:
        return _json_err("Dependências ML ausentes (numpy/scikit-learn).", 501, code="ml_missing")

    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    labeled = _get_labeled_rows(client_id)
    can, reason, classes = _can_train(labeled)
    if not can:
        return _json_ok({
            "client_id": client_id,
            "can_train": False,
            "classes_rotuladas": classes,
            "labeled_count": len(labeled),
            "reason": reason,
            "threshold": _get_threshold(client_id),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        })

    missing = [r for r in labeled if r.get("probabilidade") is None]
    if missing:
        X = np.vstack([_features_from_row(r) for r in labeled])
        y = np.array([1 if float(r["virou_cliente"]) == 1.0 else 0 for r in labeled], dtype=int)
        pipe = _train_pipeline(X, y)
        ids = [int(r["id"]) for r in missing]
        probs = _predict_for_rows(pipe, missing)
        _update_probabilities(client_id, ids, probs)
        labeled = _get_labeled_rows(client_id)

    best_t = _best_threshold(labeled)
    _set_threshold(client_id, best_t)

    m = _compute_precision_recall(labeled, best_t)
    return _json_ok({"client_id": client_id, "threshold": float(best_t), "precision": float(m["precision"]), "recall": float(m["recall"]), "f1": float(m["f1"])})


@app.get("/insights")
def insights():
    """Insights para dashboard (conversão por faixa e série diária)."""
    client_id = (request.args.get("client_id") or "").strip()
    days = _safe_int(request.args.get("days"), 14)
    days = max(7, min(days, 90))
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    threshold = _get_threshold(client_id)
    since = _now_utc() - timedelta(days=days)

    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT probabilidade, virou_cliente, created_at
                    FROM leads
                    WHERE client_id=%s AND created_at >= %s
                    ORDER BY created_at ASC
                    """,
                    (client_id, since),
                )
                rows = [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

    bands_def = [
        ("0-0.2", 0.0, 0.2),
        ("0.2-0.4", 0.2, 0.4),
        ("0.4-0.6", 0.4, 0.6),
        ("0.6-0.8", 0.6, 0.8),
        ("0.8-1.0", 0.8, 1.01),
    ]

    bands = []
    for name, lo, hi in bands_def:
        subset = [r for r in rows if r.get("probabilidade") is not None and lo <= float(r["probabilidade"]) < hi]
        labeled = [r for r in subset if r.get("virou_cliente") is not None]
        conv = sum(1 for r in labeled if float(r["virou_cliente"]) == 1.0)
        total = len(labeled)
        rate = (conv / total) if total else 0.0
        bands.append({"band": name, "labeled": total, "converted": conv, "conversion_rate": round(float(rate), 4)})

    by_day: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        dt = r.get("created_at")
        if not dt:
            continue
        day = dt.date().isoformat()
        by_day.setdefault(day, {"day": day, "total": 0, "converted": 0, "denied": 0, "pending": 0})
        by_day[day]["total"] += 1
        vc = r.get("virou_cliente")
        if vc is None:
            by_day[day]["pending"] += 1
        elif float(vc) == 1.0:
            by_day[day]["converted"] += 1
        else:
            by_day[day]["denied"] += 1

    series = [by_day[k] for k in sorted(by_day.keys())]

    labeled_all = [r for r in rows if r.get("virou_cliente") is not None]
    conv_all = sum(1 for r in labeled_all if float(r["virou_cliente"]) == 1.0)
    den_all = sum(1 for r in labeled_all if float(r["virou_cliente"]) == 0.0)
    overall_rate = (conv_all / len(labeled_all)) if labeled_all else 0.0

    return _json_ok({
        "client_id": client_id,
        "threshold": float(threshold),
        "overall": {
            "window_total": len(rows),
            "labeled": len(labeled_all),
            "converted": conv_all,
            "denied": den_all,
            "conversion_rate": round(float(overall_rate), 4),
        },
        "bands": bands,
        "series": series,
        "window_days": days,
    })


@app.get("/leads_export.csv")
def leads_export_csv():
    """Export CSV server-side. Útil para CRM / planilha."""
    client_id = (request.args.get("client_id") or "").strip()
    limit = _safe_int(request.args.get("limit"), 5000)
    limit = max(10, min(limit, 20000))
    if not client_id:
        return Response("client_id obrigatório", status=400, mimetype="text/plain")

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return Response("auth_required", status=403, mimetype="text/plain")

    _ensure_schema_once()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, nome, email_lead, telefone, probabilidade, virou_cliente,
                           tempo_site, paginas_visitadas, clicou_preco, created_at
                    FROM leads
                    WHERE client_id=%s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (client_id, int(limit)),
                )
                rows = [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

    headers = ["id","nome","email_lead","telefone","probabilidade","virou_cliente","tempo_site","paginas_visitadas","clicou_preco","created_at"]
    lines = [",".join(headers)]
    for r in rows:
        def esc(v):
            s = "" if v is None else str(v)
            s = s.replace('"', '""')
            return f'"{s}"'
        row = []
        for h in headers:
            v = r.get(h)
            if h == "created_at" and isinstance(v, datetime):
                v = _iso(v)
            row.append(esc(v))
        lines.append(",".join(row))
    csv = "\n".join(lines)

    return Response(
        csv,
        status=200,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="leadrank_{client_id}.csv"'},
    )


@app.post("/demo_public")
def demo_public():
    """Demo pública controlada (SEM DEMO_KEY) com rate-limit por IP/mês."""
    mk = _month_key()
    ip = _client_ip()
    key = f"{ip}:{mk}"
    if _DEMO_RL.get(key, 0) >= _DEMO_RL_MAX:
        return _json_err("Limite de demos atingido para este IP neste mês.", 429, code="rate_limit")

    _DEMO_RL[key] = _DEMO_RL.get(key, 0) + 1

    data = request.get_json(silent=True) or {}
    n = max(10, min(_safe_int(data.get("n"), 30), 30))

    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    client_id = f"demo_{suffix}"
    _ensure_client_row(client_id, plan="demo")

    inserted = 0
    conv = 0
    neg = 0

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                for _ in range(n):
                    tempo_site = random.randint(15, 420)
                    paginas = random.randint(1, 10)
                    clicou_preco = random.choice([0, 1])

                    base = 0.08
                    base += min(tempo_site / 450, 0.25)
                    base += min(paginas / 12, 0.25)
                    base += 0.22 if clicou_preco else 0.0
                    prob = max(0.03, min(0.97, base + random.uniform(-0.05, 0.05)))

                    label_vc = random.choices([None, 1.0, 0.0], weights=[0.45, 0.30, 0.25])[0]
                    if label_vc == 1.0:
                        conv += 1
                    elif label_vc == 0.0:
                        neg += 1

                    nome = "Demo " + "".join(random.choice(string.ascii_uppercase) for _ in range(4))
                    email = "demo@leadrank.local"
                    telefone = "11999990000"
                    payload = {"nome": nome, "email": email, "telefone": telefone, "tempo_site": tempo_site, "paginas_visitadas": paginas, "clicou_preco": clicou_preco}

                    cur.execute(
                        """
                        INSERT INTO leads
                          (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                           payload, probabilidade, virou_cliente, created_at, updated_at)
                        VALUES
                          (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,NOW(),NOW())
                        """,
                        (client_id, nome, email, telefone, tempo_site, paginas, clicou_preco, json.dumps(payload), float(prob), label_vc),
                    )
                    inserted += 1
        return _json_ok({"client_id": client_id, "inserted": inserted, "converted": conv, "denied": neg, "pending": inserted - conv - neg})
    finally:
        conn.close()


@app.post("/seed_demo")
def seed_demo():
    """Gera dados demo para um client_id (protegido por DEMO_KEY)."""
    if not _check_demo_key():
        return _json_err("Unauthorized (DEMO_KEY)", 403)

    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    n = max(10, min(_safe_int(data.get("n"), 30), 300))

    if not client_id:
        suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
        client_id = f"demo_{suffix}"

    _ensure_client_row(client_id, plan="demo")

    inserted = 0
    conv = 0
    neg = 0
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                for _ in range(n):
                    tempo_site = random.randint(15, 420)
                    paginas = random.randint(1, 10)
                    clicou_preco = random.choice([0, 1])

                    base = 0.08
                    base += min(tempo_site / 450, 0.25)
                    base += min(paginas / 12, 0.25)
                    base += 0.22 if clicou_preco else 0.0
                    prob = max(0.03, min(0.97, base + random.uniform(-0.05, 0.05)))

                    label_vc = random.choices([None, 1.0, 0.0], weights=[0.45, 0.30, 0.25])[0]
                    if label_vc == 1.0: conv += 1
                    elif label_vc == 0.0: neg += 1

                    nome = "Demo " + "".join(random.choice(string.ascii_uppercase) for _ in range(4))
                    email = "demo@leadrank.local"
                    telefone = "11999990000"
                    payload = {"nome": nome, "email": email, "telefone": telefone, "tempo_site": tempo_site, "paginas_visitadas": paginas, "clicou_preco": clicou_preco}

                    cur.execute(
                        """
                        INSERT INTO leads
                          (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                           payload, probabilidade, virou_cliente, created_at, updated_at)
                        VALUES
                          (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,NOW(),NOW())
                        """,
                        (client_id, nome, email, telefone, tempo_site, paginas, clicou_preco, json.dumps(payload), float(prob), label_vc),
                    )
                    inserted += 1
        return _json_ok({"client_id": client_id, "inserted": inserted, "converted": conv, "denied": neg, "pending": inserted - conv - neg})
    finally:
        conn.close()


# =========================
# Run local
# =========================

# =========================
# Premium / Billing helpers
# =========================
def _stripe_price_id(plan: str) -> Optional[str]:
    if not STRIPE_PRICE_IDS_JSON:
        return None
    try:
        mp = json.loads(STRIPE_PRICE_IDS_JSON)
        return (mp.get(plan) or "").strip() or None
    except Exception:
        return None

def _admin_required() -> bool:
    return _check_demo_key()

def _upsert_subscription(client_id: str, plan: str, status: str, provider: str = "manual",
                         period_start: Optional[datetime] = None, period_end: Optional[datetime] = None,
                         cancel_at_period_end: bool = False):
    if plan not in PLAN_CATALOG:
        plan = "trial"
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO subscriptions (client_id, provider, status, plan, current_period_start, current_period_end, cancel_at_period_end, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (client_id) DO UPDATE SET
                      provider=EXCLUDED.provider,
                      status=EXCLUDED.status,
                      plan=EXCLUDED.plan,
                      current_period_start=EXCLUDED.current_period_start,
                      current_period_end=EXCLUDED.current_period_end,
                      cancel_at_period_end=EXCLUDED.cancel_at_period_end,
                      updated_at=NOW()
                """, (client_id, provider, status, plan, period_start, period_end, cancel_at_period_end))

                # política de ativação: subscription active -> client ativo e plano pago
                if status == "active":
                    cur.execute("UPDATE clients SET plan=%s, status='active', updated_at=NOW() WHERE client_id=%s", (plan, client_id))
                elif status in ("past_due", "canceled", "inactive"):
                    # por padrão, desativa. Se quiser "grace period", ajuste aqui.
                    cur.execute("UPDATE clients SET status='inactive', updated_at=NOW() WHERE client_id=%s", (client_id,))
    finally:
        conn.close()


# =========================
# Opção B: Operação / Cronless reset + Billing
# =========================
@app.post("/admin/reset_month")
def admin_reset_month():
    if not _admin_required():
        return _json_err("Unauthorized (DEMO_KEY)", 403)

    mk = _month_key()
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # reseta apenas workspaces que ainda estão no mês anterior
                cur.execute("""
                    UPDATE clients
                    SET usage_month=%s, leads_used_month=0, updated_at=NOW()
                    WHERE usage_month IS NULL OR usage_month<>%s
                """, (mk, mk))
                cur.execute("SELECT COUNT(*) AS n FROM clients")
                n = int((cur.fetchone() or {}).get("n") or 0)
        return _json_ok({"usage_month": mk, "clients_total": n})
    finally:
        conn.close()


@app.get("/billing_status")
def billing_status():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, client_row, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM subscriptions WHERE client_id=%s", (client_id,))
                sub = cur.fetchone()
        enabled = bool(STRIPE_SECRET_KEY and STRIPE_PRICE_IDS_JSON)
        return _json_ok({
            "client_id": client_id,
            "checkout_enabled": enabled,
            "subscription": sub,
            "client": {
                "plan": client_row.get("plan"),
                "status": client_row.get("status"),
                "usage_month": client_row.get("usage_month"),
                "leads_used_month": int(client_row.get("leads_used_month") or 0),
            }
        })
    finally:
        conn.close()


@app.post("/billing/checkout")
def billing_checkout():
    """
    Cria uma sessão de checkout (Stripe) se configurado.
    Caso não configurado, retorna ok=false com fallback="whatsapp".
    Requer X-API-KEY do workspace.
    """
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "").strip().lower()
    success_url = (data.get("success_url") or "").strip()
    cancel_url = (data.get("cancel_url") or "").strip()

    if not client_id:
        return _json_err("client_id obrigatório", 400)
    if plan not in PLAN_CATALOG or plan in ("trial", "demo"):
        return _json_err("plan inválido para checkout", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    if not (STRIPE_SECRET_KEY and STRIPE_PRICE_IDS_JSON):
        return _json_err("Checkout ainda não configurado. Use WhatsApp para ativar.", 501, fallback="whatsapp")

    price_id = _stripe_price_id(plan)
    if not price_id:
        return _json_err("Price ID do Stripe não encontrado para este plano.", 500)

    # cria sessão via API REST (sem lib stripe)
    import requests
    url = "https://api.stripe.com/v1/checkout/sessions"
    headers = {"Authorization": f"Bearer {STRIPE_SECRET_KEY}"}
    payload = {
        "mode": "subscription",
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": "1",
        "success_url": success_url or "https://qualificador-leads-ia.onrender.com/?checkout=success",
        "cancel_url": cancel_url or "https://qualificador-leads-ia.onrender.com/?checkout=cancel",
        "client_reference_id": client_id,
        "metadata[client_id]": client_id,
        "metadata[plan]": plan,
    }
    r = requests.post(url, headers=headers, data=payload, timeout=20)
    if r.status_code >= 400:
        return _json_err("Falha ao criar checkout no Stripe.", 502, stripe_status=r.status_code, stripe_body=r.text[:500])

    j = r.json()
    return _json_ok({"checkout_url": j.get("url"), "session_id": j.get("id"), "provider": "stripe"})


@app.post("/billing/webhook")
def billing_webhook():
    """
    Webhook genérico (Stripe/MercadoPago/etc.) com segredo simples.
    Header esperado: X-BILLING-SECRET: <BILLING_WEBHOOK_SECRET>
    """
    if not BILLING_WEBHOOK_SECRET:
        return _json_err("Webhook não configurado (BILLING_WEBHOOK_SECRET ausente).", 501)

    got = _get_header("X-BILLING-SECRET")
    if got != BILLING_WEBHOOK_SECRET:
        return _json_err("Unauthorized", 403)

    payload = request.get_json(silent=True) or {}
    provider = (payload.get("provider") or "manual").strip().lower()
    event_type = (payload.get("type") or payload.get("event_type") or "unknown").strip()
    client_id = (payload.get("client_id") or (payload.get("data") or {}).get("client_id") or "").strip()

    # log do evento
    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO billing_events (provider, event_type, client_id, payload)
                    VALUES (%s,%s,%s,%s::jsonb)
                """, (provider, event_type, client_id or None, json.dumps(payload)))
    finally:
        conn.close()

    # política simples (você pode expandir conforme provider real)
    plan = (payload.get("plan") or "").strip().lower()
    status = (payload.get("status") or "").strip().lower()

    if client_id and plan and status:
        try:
            _upsert_subscription(client_id, plan=plan, status=status, provider=provider)
        except Exception as e:
            return _json_err("Evento recebido, mas falhou ao aplicar.", 500, detail=repr(e))

    return _json_ok({"received": True})


# =========================
# Opção C: Recursos Premium (métricas + explicação)
# =========================
@app.get("/funnels")
def funnels():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                      COUNT(*) AS total,
                      SUM(CASE WHEN probabilidade >= 0.70 THEN 1 ELSE 0 END) AS hot,
                      SUM(CASE WHEN probabilidade >= 0.35 AND probabilidade < 0.70 THEN 1 ELSE 0 END) AS warm,
                      SUM(CASE WHEN probabilidade < 0.35 THEN 1 ELSE 0 END) AS cold,
                      SUM(CASE WHEN virou_cliente = 1 THEN 1 ELSE 0 END) AS convertidos,
                      SUM(CASE WHEN virou_cliente = 0 THEN 1 ELSE 0 END) AS negados,
                      SUM(CASE WHEN virou_cliente IS NULL THEN 1 ELSE 0 END) AS pendentes
                    FROM leads WHERE client_id=%s
                """, (client_id,))
                row = cur.fetchone() or {}

        total = int(row.get("total") or 0)
        conv = int(row.get("convertidos") or 0)
        labeled = int(row.get("convertidos") or 0) + int(row.get("negados") or 0)
        conv_rate = (conv / labeled) if labeled else 0.0
        return _json_ok({
            "client_id": client_id,
            "total": total,
            "hot": int(row.get("hot") or 0),
            "warm": int(row.get("warm") or 0),
            "cold": int(row.get("cold") or 0),
            "convertidos": conv,
            "negados": int(row.get("negados") or 0),
            "pendentes": int(row.get("pendentes") or 0),
            "conversion_rate_labeled": conv_rate,
        })
    finally:
        conn.close()


@app.get("/lead_explain")
def lead_explain():
    client_id = (request.args.get("client_id") or "").strip()
    lead_id = _safe_int(request.args.get("lead_id"), 0)
    if not client_id or not lead_id:
        return _json_err("client_id e lead_id obrigatórios", 400)

    ok_auth, _, msg = _require_client_auth(client_id)
    if not ok_auth:
        return _json_err(msg, 403, code="auth_required")

    conn = _db()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM leads WHERE client_id=%s AND id=%s", (client_id, lead_id))
                lead = cur.fetchone()
        if not lead:
            return _json_err("Lead não encontrado", 404)

        tempo = _safe_int(lead.get("tempo_site"), 0)
        pags = _safe_int(lead.get("paginas_visitadas"), 0)
        clicou = _safe_int(lead.get("clicou_preco"), 0)

        parts = []
        base = 0.10
        parts.append({"factor": "base", "delta": base, "why": "ponto de partida do modelo"} )

        d_tempo = min(tempo / 400, 0.25)
        parts.append({"factor": "tempo_site", "delta": d_tempo, "why": f"{tempo}s no site"})

        d_pags = min(pags / 10, 0.25)
        parts.append({"factor": "paginas_visitadas", "delta": d_pags, "why": f"{pags} páginas visitadas"})

        d_click = 0.20 if clicou else 0.0
        parts.append({"factor": "clicou_preco", "delta": d_click, "why": "clicou em preço" if clicou else "não clicou em preço"})

        tel = (lead.get("telefone") or "").strip()
        nome = (lead.get("nome") or "").strip()
        d_tel = 0.06 if (tel and len(tel) >= 10) else 0.0
        d_nome = 0.04 if (nome and len(nome) >= 4) else 0.0
        if d_tel: parts.append({"factor": "telefone", "delta": d_tel, "why": "telefone válido"})
        if d_nome: parts.append({"factor": "nome", "delta": d_nome, "why": "nome preenchido"})

        score = float(lead.get("probabilidade") or 0.0)
        return _json_ok({
            "client_id": client_id,
            "lead_id": lead_id,
            "probabilidade": score,
            "explain": parts,
            "note": "Explicação do score heurístico (antes do treino)."
        })
    finally:
        conn.close()
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)