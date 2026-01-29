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

# Stripe/Billing envs (evita NameError se não configurar ainda)
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_IDS_JSON = os.environ.get("STRIPE_PRICE_IDS_JSON", "").strip()
BILLING_WEBHOOK_SECRET = os.environ.get("BILLING_WEBHOOK_SECRET", "").strip()

# ajuste aqui seus domínios permitidos no CORS
ALLOWED_ORIGINS = [
    "null",  # permite testar abrindo HTML via file://
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

def _require_demo_key() -> tuple[bool, str | None]:
    """Valida DEMO_KEY de forma compatível (header/query/body)."""
    expected = (os.getenv("DEMO_KEY") or "").strip()
    if not expected:
        return False, "DEMO_KEY não configurada no ambiente"

    # aceita header (qualquer case), Authorization Bearer, query param e body
    got = (
        (request.headers.get("x-demo-key") or request.headers.get("X-DEMO-KEY") or "").strip()
    )
    if not got:
        auth = (request.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            got = auth.split(" ", 1)[1].strip()

    if not got:
        got = (request.args.get("demo_key") or "").strip()

    if not got:
        data = request.get_json(silent=True) or {}
        got = (data.get("demo_key") or "").strip()

    if not got:
        return False, "DEMO_KEY ausente"

    if got != expected:
        return False, "DEMO_KEY inválida"

    return True, None


def _get_api_key_from_headers() -> str:
    key = _get_header("X-API-KEY") or _get_header("Authorization")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key

def _client_ip() -> str:
    return (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0].strip()


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
    - signup (nome/email/empresa/valid_until) -> colunas opcionais
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
                        -- dados opcionais (para signup)
                        nome TEXT,
                        email TEXT,
                        empresa TEXT,
                        valid_until TIMESTAMPTZ,
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

                # colunas opcionais p/ signup
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS nome TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS email TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS empresa TEXT;")
                cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ;")

                # index/unique de email (opcional, e sem quebrar DB antigo)
                try:
                    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_email_unique ON clients(email) WHERE email IS NOT NULL AND email<>'';")
                except Exception:
                    pass

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

                # -------------------------
                # Billing tables (para endpoints billing_*)
                # -------------------------
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS subscriptions (
                        client_id TEXT PRIMARY KEY,
                        provider TEXT NOT NULL DEFAULT 'manual',
                        status TEXT NOT NULL DEFAULT 'inactive',
                        plan TEXT NOT NULL DEFAULT 'trial',
                        current_period_start TIMESTAMPTZ,
                        current_period_end TIMESTAMPTZ,
                        cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS billing_events (
                        id BIGSERIAL PRIMARY KEY,
                        provider TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        client_id TEXT,
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
def _fetch_recent_leads(client_id: str, limit: int = 200) -> List[Dict[str, Any]]:
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

@app.route('/signup', methods=['POST'])
def signup():
    """
    Signup (trial):
    - Antes quebrava porque a tabela 'clients' do seu schema não tem coluna 'id'.
    - Agora usa 'client_id' (PK do schema atual) e adiciona colunas opcionais (nome/email/empresa/valid_until).
    """
    data = request.get_json(silent=True) or request.form or {}
    nome = (data.get('nome') or '').strip()
    email = (data.get('email') or '').strip().lower()
    empresa = (data.get('empresa') or '').strip()
    telefone = (data.get('telefone') or '').strip()

    if not email or '@' not in email:
        return jsonify({"ok": False, "error": "Email válido é obrigatório"}), 400

    _ensure_schema_once()

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                # Verifica se email já existe (email é coluna opcional criada via auto-migration)
                cur.execute("SELECT client_id FROM clients WHERE email = %s", (email,))
                if cur.fetchone():
                    return jsonify({"ok": False, "error": "Este email já está cadastrado"}), 409

                api_key = _gen_api_key("trial")  # key padrão do projeto
                client_id = f"trial-{secrets.token_hex(8)}"
                mk = _month_key()
                valid_until = datetime.now(timezone.utc) + timedelta(days=14)

                cur.execute("""
                    INSERT INTO clients (
                        client_id, api_key, plan, status, usage_month, leads_used_month,
                        nome, email, empresa, telefone, valid_until, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (
                    client_id, api_key, 'trial', 'active', mk, 0,
                    nome or None, email, empresa or None, telefone or None, valid_until
                ))

        return jsonify({
            "ok": True,
            "success": True,
            "client_id": client_id,
            "api_key": api_key,
            "plan": "trial",
            "valid_until": _iso(valid_until),
            "message": "Conta trial criada!"
        })
    except Exception as e:
        # Loga traceback completo no Render (senão fica impossível debugar 500)
        try:
            app.logger.exception("/signup failed")
        except Exception:
            pass
        detail = str(e)
        pgcode = getattr(e, 'pgcode', None) or getattr(getattr(e, 'diag', None), 'sqlstate', None)
        return jsonify({"ok": False, "success": False, "error": detail, "pgcode": pgcode}), 500
    finally:
        conn.close()


# --------- (o resto do arquivo permanece igual ao seu, sem mudanças funcionais) ---------

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


# =========================
# (A partir daqui, cole/mescle o restante do seu app.py original)
# =========================
# Para caber aqui, mantive apenas o trecho necessário para corrigir o erro do /signup.
# Se você quiser que eu gere o arquivo COMPLETO com tudo (todas as rotas),
# me envie o restante do arquivo (depois de /criar_cliente), ou diga “pode repetir o resto igual”.
