# app.py - LeadRank / Qualificador de Leads IA (Postgres / Render)
# Versão "clean": corrige indentation error, imports faltando e remove duplicações.

import os
import re
import math
import random
import string
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import psycopg2
import psycopg2.extras

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================
# App / Config
# =========================
app = Flask(__name__)

# ✅ CORS: libera o seu Static Site (Render) e também localhost para testes
CORS(app, resources={r"/*": {"origins": [
    "https://qualificador-leads-ia.onrender.com",
    "http://localhost",
    "http://127.0.0.1",
    "*"
]}})

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# --- DB bootstrap/migrations (auto) ---
_SCHEMA_READY = False
_SCHEMA_LOCK = None  # lazy init

def _raw_conn():
    """Connection without triggering schema bootstrapping (used by migrations)."""
    _db_required()
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def _ensure_schema_once():
    """Run lightweight migrations once per process. Safe to call many times."""
    global _SCHEMA_READY, _SCHEMA_LOCK
    if _SCHEMA_READY:
        return
    if _SCHEMA_LOCK is None:
        import threading as _t
        _SCHEMA_LOCK = _t.Lock()
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        _ensure_schema()
        _SCHEMA_READY = True

DEMO_KEY = (os.getenv("DEMO_KEY") or "").strip()

# Parâmetros padrões
DEFAULT_LIMIT = 200
MIN_LABELED_TO_TRAIN = 4  # 2 de cada classe recomendado

# Threshold default (pode ser ajustado por /auto_threshold)
DEFAULT_THRESHOLD = 0.35


# =========================
# Helpers
# =========================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _json_ok(payload: Dict[str, Any], code: int = 200):
    payload.setdefault("ok", True)
    return jsonify(payload), code

def _json_err(msg: str, code: int = 400, **extra):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return jsonify(payload), code

def _db_required():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada.")

def db_conn():
    _db_required()
    # Ensure schema/migrations before returning connections for normal operations
    _ensure_schema_once()
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _require_demo_key():
    """
    Protect demo-only endpoints. Accepts key via:
      - Header: X-DEMO-KEY
      - Query: demo_key
      - JSON: demo_key
    Always reads DEMO_KEY fresh from env (Render-safe).
    """
    expected = (os.getenv("DEMO_KEY") or "").strip()
    if not expected:
        return False, "DEMO_KEY não configurada no servidor."

    key = (request.headers.get("X-DEMO-KEY") or "").strip()

    if not key:
        key = (request.args.get("demo_key") or "").strip()

    if not key:
        data = request.get_json(silent=True) or {}
        key = (data.get("demo_key") or "").strip()

    # (extra tolerância: alguns clients mandam "Bearer <key>")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()

    if key != expected:
        return False, "Chave DEMO_KEY inválida."
    return True, ""


# =========================
# Monetização: planos / api_key / uso mensal
# =========================
PLAN_CATALOG = {
    "demo":    {"price_brl_month": 0,   "lead_limit_month": 30},
    "trial":   {"price_brl_month": 0,   "lead_limit_month": 100},
    "starter": {"price_brl_month": 79,  "lead_limit_month": 1000},
    "pro":     {"price_brl_month": 179, "lead_limit_month": 5000},
    "vip":     {"price_brl_month": 279, "lead_limit_month": 20000},
}

def _month_key(dt: Optional[datetime] = None) -> str:
    dt = dt or _now_utc()
    return dt.strftime("%Y-%m")

def _gen_api_key() -> str:
    return "sk_live_" + secrets.token_urlsafe(24)

def _ensure_client_row(client_id: str, plan: str = "trial") -> Dict[str, Any]:
    """Garante que existe um registro em clients. Reseta uso mensal ao virar o mês."""
    plan = (plan or "trial").strip().lower()
    if plan not in PLAN_CATALOG:
        plan = "trial"

    mk = _month_key()

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO clients (client_id, api_key, plan, status, usage_month, leads_used_month, updated_at)
                VALUES (%s, NULL, %s, 'active', %s, 0, NOW())
                ON CONFLICT (client_id) DO NOTHING
            """, (client_id, plan, mk))

            cur.execute("""SELECT * FROM clients WHERE client_id=%s FOR UPDATE""", (client_id,))
            row = cur.fetchone() or {}
            usage_month = (row.get("usage_month") or "").strip()
            if usage_month != mk:
                cur.execute("""UPDATE clients SET usage_month=%s, leads_used_month=0, updated_at=NOW() WHERE client_id=%s""", (mk, client_id))
                cur.execute("""SELECT * FROM clients WHERE client_id=%s""", (client_id,))
                row = cur.fetchone() or {}

        conn.commit()

    return row

def _require_api_key(client_row: Dict[str, Any]) -> Tuple[bool, str]:
    """Se o cliente tem api_key cadastrada, exige X-API-KEY correta."""
    expected = (client_row.get("api_key") or "").strip()
    if not expected:
        return True, ""  # compatibilidade: workspace antigo sem key

    got = (request.headers.get("X-API-KEY") or "").strip()
    if not got:
        data = request.get_json(silent=True) or {}
        got = (data.get("api_key") or "").strip()

    if got != expected:
        return False, "api_key inválida ou ausente."
    return True, ""
def _ensure_schema():
    """
    Cria tabelas mínimas necessárias para Postgres e aplica migrações leves (ADD COLUMN IF NOT EXISTS)
    para evitar 500 quando o schema já existe em versão antiga.

    IMPORTANTE: Render/Postgres às vezes já tem a tabela criada por versões anteriores.
    CREATE TABLE IF NOT EXISTS não adiciona colunas novas — por isso usamos ALTER TABLE.
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            # --- leads ---
            cur.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id SERIAL PRIMARY KEY,
                client_id TEXT NOT NULL,
                nome TEXT,
                email_lead TEXT,
                telefone TEXT,
                tempo_site INTEGER,
                paginas_visitadas INTEGER,
                clicou_preco INTEGER,
                probabilidade DOUBLE PRECISION,
                virou_cliente DOUBLE PRECISION,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """)
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_leads_client_created ON leads(client_id, created_at DESC);""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_leads_client_label ON leads(client_id, virou_cliente);""")

            # --- clients (workspace / monetização) ---
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
            # Migrações (para bancos já existentes)
            cur.execute("""ALTER TABLE clients ADD COLUMN IF NOT EXISTS api_key TEXT;""")
            cur.execute("""ALTER TABLE clients ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'trial';""")
            cur.execute("""ALTER TABLE clients ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';""")
            cur.execute("""ALTER TABLE clients ADD COLUMN IF NOT EXISTS usage_month TEXT;""")
            cur.execute("""ALTER TABLE clients ADD COLUMN IF NOT EXISTS leads_used_month INTEGER NOT NULL DEFAULT 0;""")
            cur.execute("""ALTER TABLE clients ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();""")
            cur.execute("""ALTER TABLE clients ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();""")

            # Garante defaults onde fizer sentido
            mk = _month_key()
            cur.execute("""UPDATE clients SET usage_month=%s WHERE usage_month IS NULL OR usage_month='';""", (mk,))
            cur.execute("""UPDATE clients SET updated_at=NOW() WHERE updated_at IS NULL;""")

            cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_api_key ON clients(api_key) WHERE api_key IS NOT NULL;""")

            # --- thresholds ---
            cur.execute("""
            CREATE TABLE IF NOT EXISTS thresholds (
                client_id TEXT PRIMARY KEY,
                threshold DOUBLE PRECISION NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """)

            # --- model_meta ---
            cur.execute("""
            CREATE TABLE IF NOT EXISTS model_meta (
                client_id TEXT PRIMARY KEY,
                can_train BOOLEAN NOT NULL DEFAULT FALSE,
                labeled_count INTEGER NOT NULL DEFAULT 0,
                classes_rotuladas TEXT NOT NULL DEFAULT '[]',
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """)
        conn.commit()

def _get_threshold(client_id: str) -> float:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT threshold FROM thresholds WHERE client_id=%s", (client_id,))
            row = cur.fetchone()
            if row and row.get("threshold") is not None:
                return float(row["threshold"])
    return DEFAULT_THRESHOLD

def _set_threshold(client_id: str, threshold: float):
    threshold = float(threshold)
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO thresholds (client_id, threshold, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (client_id)
                DO UPDATE SET threshold=EXCLUDED.threshold, updated_at=NOW()
            """, (client_id, threshold))
        conn.commit()

def _fetch_recent_leads(client_id: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM leads
                WHERE client_id=%s
                ORDER BY created_at DESC
                LIMIT %s
            """, (client_id, limit))
            rows = cur.fetchall()
            return rows or []

def _fetch_all_leads(client_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM leads
                WHERE client_id=%s
                ORDER BY created_at DESC
                LIMIT %s
            """, (client_id, limit))
            return cur.fetchall() or []

def _count_status(rows: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    convertidos = sum(1 for r in rows if r.get("virou_cliente") == 1 or r.get("virou_cliente") == 1.0)
    negados = sum(1 for r in rows if r.get("virou_cliente") == 0 or r.get("virou_cliente") == 0.0)
    pendentes = len(rows) - convertidos - negados
    return convertidos, negados, pendentes

def _features_from_row(r: Dict[str, Any]) -> np.ndarray:
    # mesmas features usadas no projeto
    tempo = _safe_int(r.get("tempo_site"), 0)
    paginas = _safe_int(r.get("paginas_visitadas"), 0)
    clicou = _safe_int(r.get("clicou_preco"), 0)
    return np.array([tempo, paginas, clicou], dtype=float)

def _train_pipeline(X: np.ndarray, y: np.ndarray) -> Pipeline:
    # pipeline simples e robusto
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])
    pipe.fit(X, y)
    return pipe

def _get_labeled_rows(client_id: str) -> List[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM leads
                WHERE client_id=%s AND virou_cliente IS NOT NULL
                ORDER BY created_at DESC
            """, (client_id,))
            return cur.fetchall() or []

def _can_train(labeled_rows: List[Dict[str, Any]]) -> Tuple[bool, str, List[float]]:
    if len(labeled_rows) < MIN_LABELED_TO_TRAIN:
        classes = sorted(list({float(r["virou_cliente"]) for r in labeled_rows if r.get("virou_cliente") is not None}))
        return False, f"Poucos exemplos rotulados. Recomendo no mínimo {MIN_LABELED_TO_TRAIN} (2 de cada classe) para começar.", classes
    classes = sorted(list({float(r["virou_cliente"]) for r in labeled_rows if r.get("virou_cliente") is not None}))
    if len(classes) < 2:
        return False, "Precisa de exemplos das duas classes (convertido e negado) para treinar.", classes
    return True, "", classes

def _predict_for_rows(pipe: Pipeline, rows: List[Dict[str, Any]]) -> List[float]:
    if not rows:
        return []
    X = np.vstack([_features_from_row(r) for r in rows])
    # probabilidade da classe 1
    probs = pipe.predict_proba(X)[:, 1]
    return probs.tolist()

def _update_probabilities(client_id: str, ids: List[int], probs: List[float]) -> int:
    if not ids:
        return 0
    with db_conn() as conn:
        with conn.cursor() as cur:
            for lead_id, p in zip(ids, probs):
                cur.execute("""
                    UPDATE leads SET probabilidade=%s
                    WHERE client_id=%s AND id=%s
                """, (float(p), client_id, int(lead_id)))
        conn.commit()
    return len(ids)

def _compute_precision_recall(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    """
    Usa rows rotulados (virou_cliente 0/1) e probabilidade para métricas simples.
    """
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
    """
    Busca threshold que maximiza F1 em uma grade simples.
    """
    candidates = [i/100 for i in range(5, 96, 5)]  # 0.05..0.95
    best_t = DEFAULT_THRESHOLD
    best_f1 = -1.0
    for t in candidates:
        m = _compute_precision_recall(rows, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
    return float(best_t)


# =========================
# Boot schema on import (safe)
# =========================
try:
    _ensure_schema()
except Exception:
    # Em Render, se DATABASE_URL não estiver setado ainda, o app pode subir e /health responde.
    pass


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return jsonify({"ok": True, "service": "LeadRank backend", "ts": _iso(_now_utc())})

@app.get("/health")
def health():
    # Não falha mesmo se DB off, mas avisa
    db_ok = bool(DATABASE_URL)
    return jsonify({"ok": True, "db_configured": db_ok, "ts": _iso(_now_utc())})

@app.get("/health_db")
def health_db():
    """Healthcheck que realmente toca no Postgres e aplica migrações leves."""
    try:
        _ensure_schema_once()
        with _raw_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok;")
                row = cur.fetchone()
        return jsonify({"ok": True, "db": True, "ts": _iso(_now_utc()), "select1": (row.get("ok") if row else 1)})
    except Exception as e:
        return jsonify({"ok": False, "db": False, "error": repr(e), "ts": _iso(_now_utc())}), 500

@app.get("/metrics")
def metrics():
    """
    Métricas gerais simples do sistema (para debug/monitoramento).
    """
    if not DATABASE_URL:
        return _json_ok({"db": False, "reason": "DATABASE_URL ausente"})

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM leads;")
            total = int(cur.fetchone()["total"])
            cur.execute("SELECT COUNT(*) AS labeled FROM leads WHERE virou_cliente IS NOT NULL;")
            labeled = int(cur.fetchone()["labeled"])
            cur.execute("SELECT COUNT(*) AS pending FROM leads WHERE virou_cliente IS NULL;")
            pending = int(cur.fetchone()["pending"])

    return _json_ok({
        "db": True,
        "total_leads": total,
        "labeled": labeled,
        "pending": pending,
        "ts": _iso(_now_utc())
    })

@app.post("/criar_cliente")
def criar_cliente():
    """
    Cria/garante o workspace (client_id) e emite api_key (para SaaS).
    Body JSON:
      - client_id (obrigatório)
      - plan (opcional: trial/starter/pro/vip) — default: trial
    """
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório")

    plan = (data.get("plan") or "trial").strip().lower()
    if plan not in PLAN_CATALOG:
        plan = "trial"

    _ensure_schema()

    # cria meta do modelo se não existir
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_meta (client_id, can_train, labeled_count, classes_rotuladas, updated_at)
                VALUES (%s, FALSE, 0, '[]', NOW())
                ON CONFLICT (client_id) DO NOTHING
            """, (client_id,))
        conn.commit()

    # cria/atualiza client + api_key
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT * FROM clients WHERE client_id=%s""", (client_id,))
            row = cur.fetchone()

            if not row:
                row = _ensure_client_row(client_id, plan=plan)

            api_key = (row.get("api_key") or "").strip()
            if not api_key:
                api_key = _gen_api_key()
                cur.execute("""UPDATE clients SET api_key=%s, plan=%s, updated_at=NOW() WHERE client_id=%s""", (api_key, plan, client_id))
                cur.execute("""SELECT * FROM clients WHERE client_id=%s""", (client_id,))
                row = cur.fetchone() or row

        conn.commit()

    meta = PLAN_CATALOG.get((row.get("plan") or plan).strip().lower(), PLAN_CATALOG["trial"])
    return _json_ok({
        "client_id": client_id,
        "plan": (row.get("plan") or plan),
        "api_key": (row.get("api_key") or api_key),
        "price_brl_month": meta["price_brl_month"],
        "lead_limit_month": meta["lead_limit_month"],
    })

@app.get("/client_meta")
def client_meta():
    """Retorna plano, preço e consumo mensal para o dashboard."""
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório")

    _ensure_schema()
    row = _ensure_client_row(client_id, plan="trial")
    meta = PLAN_CATALOG.get((row.get("plan") or "trial").strip().lower(), PLAN_CATALOG["trial"])

    return _json_ok({
        "client_id": client_id,
        "plan": (row.get("plan") or "trial"),
        "status": (row.get("status") or "active"),
        "price_brl_month": meta["price_brl_month"],
        "lead_limit_month": meta["lead_limit_month"],
        "leads_used_this_month": int(row.get("leads_used_month") or 0),
        "usage_month": (row.get("usage_month") or _month_key()),
    })

@app.post("/set_plan")
def set_plan():
    """Admin: ajusta plano/status de um client_id. Protegido por DEMO_KEY."""
    ok, err = _require_demo_key()
    if not ok:
        return _json_err(err, 401, code="demo_key_required")

    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório")

    plan = (data.get("plan") or "").strip().lower()
    status = (data.get("status") or "").strip().lower()

    if plan and plan not in PLAN_CATALOG:
        return _json_err("Plano inválido", 400, allowed=list(PLAN_CATALOG.keys()))
    if status and status not in ["active", "inactive"]:
        return _json_err("Status inválido", 400, allowed=["active","inactive"])

    _ensure_schema()
    _ensure_client_row(client_id, plan=plan or "trial")

    with db_conn() as conn:
        with conn.cursor() as cur:
            sets = []
            vals = []
            if plan:
                sets.append("plan=%s")
                vals.append(plan)
            if status:
                sets.append("status=%s")
                vals.append(status)
            sets.append("updated_at=NOW()")
            vals.append(client_id)
            cur.execute(f"UPDATE clients SET {', '.join(sets)} WHERE client_id=%s", tuple(vals))
        conn.commit()

    return _json_ok({"client_id": client_id, "plan": plan or None, "status": status or None})


@app.post("/prever")
def prever():
    """
    Recebe features e retorna probabilidade estimada.
    Também grava o lead no banco (com probabilidade).
    """
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório")

    _ensure_schema()
    client_row = _ensure_client_row(client_id, plan="trial")
    ok_auth, err = _require_api_key(client_row)
    if not ok_auth:
        return _json_err(err, 401, code="auth_required")

    if (client_row.get("status") or "active") != "active":
        return _json_err("Workspace inativo. Fale com o suporte para reativar.", 403, code="inactive")

    plan = (client_row.get("plan") or "trial").strip().lower()
    meta = PLAN_CATALOG.get(plan, PLAN_CATALOG["trial"])
    used = int(client_row.get("leads_used_month") or 0)
    limit = int(meta.get("lead_limit_month") or 0)

    if limit > 0 and used >= limit:
        return _json_err("Limite mensal atingido. Faça upgrade para continuar.", 402,
                         code="plan_limit", plan=plan, used=used, limit=limit,
                         price_brl_month=meta.get("price_brl_month"))

    nome = (data.get("nome") or "").strip()
    email = (data.get("email_lead") or data.get("email") or "").strip()
    telefone = (data.get("telefone") or "").strip()

    tempo_site = _safe_int(data.get("tempo_site"), 0)
    paginas_visitadas = _safe_int(data.get("paginas_visitadas"), 0)
    clicou_preco = _safe_int(data.get("clicou_preco"), 0)

    # prob inicial simples até treinar: função heurística
    # (o treino real entra via /recalc_pending)
    base = 0.10
    base += min(tempo_site / 400, 0.25)
    base += min(paginas_visitadas / 10, 0.25)
    base += 0.20 if clicou_preco else 0.0
    prob = max(0.02, min(0.98, base))

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO leads
                  (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco, probabilidade, virou_cliente, created_at)
                VALUES
                  (%s,%s,%s,%s,%s,%s,%s,%s,NULL,NOW())
                RETURNING id, created_at
            """, (client_id, nome, email, telefone, tempo_site, paginas_visitadas, clicou_preco, float(prob)))
            row = cur.fetchone()
            cur.execute("""UPDATE clients SET leads_used_month = leads_used_month + 1, updated_at=NOW() WHERE client_id=%s""", (client_id,))
        conn.commit()

    return _json_ok({
        "client_id": client_id,
        "lead_id": int(row["id"]),
        "probabilidade": float(prob),
        "created_at": _iso(row["created_at"])
    })

@app.get("/dashboard_data")
def dashboard_data():
    """
    Dados para dashboard (últimos 200, já com probabilidade).
    """
    client_id = (request.args.get("client_id") or "").strip()
    limit = _safe_int(request.args.get("limit"), DEFAULT_LIMIT)
    limit = max(10, min(limit, 1000))

    if not client_id:
        return _json_err("client_id obrigatório")

    rows = _fetch_recent_leads(client_id, limit=limit)
    convertidos, negados, pendentes = _count_status(rows)

    # últimos rotulados e pendentes (para debug)
    labeled = [r for r in rows if r.get("virou_cliente") is not None]
    pending = [r for r in rows if r.get("virou_cliente") is None]

    def normalize_row(r):
        rr = dict(r)
        rr["created_at"] = _iso(rr.get("created_at"))
        return rr

    return _json_ok({
        "client_id": client_id,
        "convertidos": convertidos,
        "negados": negados,
        "pendentes": pendentes,
        "dados": [normalize_row(r) for r in rows],
        "last_labeled": [normalize_row(r) for r in labeled[:10]],
        "last_pending": [normalize_row(r) for r in pending[:10]],
        "total_recentes_considerados": len(rows)
    })

@app.post("/confirmar_venda")
def confirmar_venda():
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = _safe_int(data.get("lead_id"), 0)

    if not client_id or not lead_id:
        return _json_err("client_id e lead_id obrigatórios")

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE leads
                SET virou_cliente=1
                WHERE client_id=%s AND id=%s
            """, (client_id, lead_id))
        conn.commit()

    return _json_ok({"client_id": client_id, "lead_id": lead_id, "virou_cliente": 1})

@app.post("/negar_venda")
def negar_venda():
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = _safe_int(data.get("lead_id"), 0)

    if not client_id or not lead_id:
        return _json_err("client_id e lead_id obrigatórios")

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE leads
                SET virou_cliente=0
                WHERE client_id=%s AND id=%s
            """, (client_id, lead_id))
        conn.commit()

    return _json_ok({"client_id": client_id, "lead_id": lead_id, "virou_cliente": 0})

@app.get("/recalc_pending")
def recalc_pending():
    """
    Recalcula a probabilidade para leads pendentes com base no modelo treinado nos rotulados.
    """
    client_id = (request.args.get("client_id") or "").strip()
    limit = _safe_int(request.args.get("limit"), 500)
    limit = max(10, min(limit, 5000))

    if not client_id:
        return _json_err("client_id obrigatório")

    labeled = _get_labeled_rows(client_id)
    can, reason, classes = _can_train(labeled)
    if not can:
        return _json_ok({
            "client_id": client_id,
            "can_train": False,
            "classes_rotuladas": classes,
            "labeled_count": len(labeled),
            "reason": reason,
            "updated": 0
        })

    X = np.vstack([_features_from_row(r) for r in labeled])
    y = np.array([1 if float(r["virou_cliente"]) == 1.0 else 0 for r in labeled], dtype=int)
    pipe = _train_pipeline(X, y)

    # pendentes mais recentes
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, tempo_site, paginas_visitadas, clicou_preco
                FROM leads
                WHERE client_id=%s AND virou_cliente IS NULL
                ORDER BY created_at DESC
                LIMIT %s
            """, (client_id, limit))
            pending = cur.fetchall() or []

    ids = [int(r["id"]) for r in pending]
    probs = _predict_for_rows(pipe, pending)
    updated = _update_probabilities(client_id, ids, probs)

    return _json_ok({
        "client_id": client_id,
        "can_train": True,
        "classes_rotuladas": classes,
        "labeled_count": len(labeled),
        "reason": "",
        "updated": updated,
        "min_prob": float(min(probs)) if probs else None,
        "max_prob": float(max(probs)) if probs else None,
        "sample": [{"id": ids[i], "prob": float(probs[i])} for i in range(min(5, len(ids)))]
    })

@app.post("/auto_threshold")
def auto_threshold():
    """
    Calcula e salva threshold que maximiza F1 para o client_id (com base em rotulados).
    """
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return _json_err("client_id obrigatório")

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

    # garante probs atualizadas para rotulados também (se faltando)
    missing = [r for r in labeled if r.get("probabilidade") is None]
    if missing:
        X = np.vstack([_features_from_row(r) for r in labeled])
        y = np.array([1 if float(r["virou_cliente"]) == 1.0 else 0 for r in labeled], dtype=int)
        pipe = _train_pipeline(X, y)
        ids = [int(r["id"]) for r in missing]
        probs = _predict_for_rows(pipe, missing)
        _update_probabilities(client_id, ids, probs)

        # refetch labeled
        labeled = _get_labeled_rows(client_id)

    best_t = _best_threshold(labeled)
    _set_threshold(client_id, best_t)

    m = _compute_precision_recall(labeled, best_t)
    return _json_ok({
        "client_id": client_id,
        "threshold": float(best_t),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"])
    })

@app.get("/insights")
def insights():
    """
    Retorna insights para a tela de métricas:
    - conversão por faixa de probabilidade
    - série diária (últimos X dias)
    """
    client_id = (request.args.get("client_id") or "").strip()
    days = _safe_int(request.args.get("days"), 14)
    days = max(7, min(days, 90))

    if not client_id:
        return _json_err("client_id obrigatório")

    threshold = _get_threshold(client_id)

    # janela de tempo
    since = _now_utc() - timedelta(days=days)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT probabilidade, virou_cliente, created_at
                FROM leads
                WHERE client_id=%s AND created_at >= %s
                ORDER BY created_at ASC
            """, (client_id, since))
            rows = cur.fetchall() or []

    # bands
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
        bands.append({
            "band": name,
            "labeled": total,
            "converted": conv,
            "conversion_rate": round(float(rate), 4)
        })

    # series daily
    by_day = {}
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

    # overall
    labeled_all = [r for r in rows if r.get("virou_cliente") is not None]
    conv_all = sum(1 for r in labeled_all if float(r["virou_cliente"]) == 1.0)
    overall_rate = (conv_all / len(labeled_all)) if labeled_all else 0.0

    return _json_ok(
        {
            "client_id": client_id,
            "threshold": float(threshold),
            "overall": {
                "window_total": len(rows),
                "window_labeled": len(labeled_all),
                "window_converted": conv_all,
                "conversion_rate": round(float(overall_rate), 4),
            },
            "bands": bands,
            "series": series,
            "window_days": days,
        }
    ), 200



@app.post("/demo_public")
def demo_public():
    """Demo pública controlada (SEM DEMO_KEY)."""
    _ensure_schema()
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0].strip()
    mk = _month_key()

    if not hasattr(app, "_demo_rl"):
        app._demo_rl = {}
    rl = app._demo_rl
    key = f"{ip}:{mk}"
    if rl.get(key, 0) >= 5:
        return _json_err("Limite de demos atingido para este IP neste mês.", 429, code="rate_limit")
    rl[key] = rl.get(key, 0) + 1

    data = request.get_json(silent=True) or {}
    n = max(10, min(_safe_int(data.get("n"), 30), 30))

    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    client_id = f"demo_{suffix}"

    _ensure_client_row(client_id, plan="demo")

    inserted = 0
    conv = 0
    neg = 0

    with db_conn() as conn:
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

                label = random.choices([None, 1.0, 0.0], weights=[0.45, 0.30, 0.25])[0]
                if label == 1.0:
                    conv += 1
                elif label == 0.0:
                    neg += 1

                nome = "Demo " + "".join(random.choice(string.ascii_uppercase) for _ in range(4))
                email = "demo@leadrank.local"
                telefone = "11999990000"

                cur.execute("""
                    INSERT INTO leads
                      (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco, probabilidade, virou_cliente, created_at)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                """, (client_id, nome, email, telefone, tempo_site, paginas, clicou_preco, float(prob), label))
                inserted += 1
        conn.commit()

    return _json_ok({
        "client_id": client_id,
        "inserted": inserted,
        "converted": conv,
        "denied": neg,
        "pending": inserted - conv - neg
    })


@app.post("/seed_demo")
def seed_demo():
    """
    Gera dados realistas de demonstração para um client_id.
    Protegido por DEMO_KEY (env var).
    """
    ok, msg = _require_demo_key()
    if not ok:
        return jsonify({"ok": False, "error": msg}), 403

    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    n = int(data.get("n") or 30)
    n = max(10, min(n, 300))

    if not client_id:
        suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
        client_id = f"demo_{suffix}"

    # gera registros
    inserted = 0
    conv = 0
    neg = 0

    with db_conn() as conn:
        with conn.cursor() as cur:
            for _ in range(n):
                tempo_site = random.randint(15, 420)
                paginas = random.randint(1, 10)
                clicou_preco = random.choice([0, 1])

                # prob coerente com features
                base = 0.08
                base += min(tempo_site / 450, 0.25)
                base += min(paginas / 12, 0.25)
                base += 0.22 if clicou_preco else 0.0
                prob = max(0.03, min(0.97, base + random.uniform(-0.05, 0.05)))

                # rótulo parcial (para simular treino)
                label = random.choices([None, 1.0, 0.0], weights=[0.45, 0.30, 0.25])[0]
                if label == 1.0:
                    conv += 1
                elif label == 0.0:
                    neg += 1

                nome = "Demo " + "".join(random.choice(string.ascii_uppercase) for _ in range(4))
                email = "demo@leadrank.local"
                telefone = "11999990000"

                cur.execute("""
                    INSERT INTO leads
                      (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco, probabilidade, virou_cliente, created_at)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                """, (client_id, nome, email, telefone, tempo_site, paginas, clicou_preco, float(prob), label))
                inserted += 1
        conn.commit()

    return jsonify({
        "ok": True,
        "client_id": client_id,
        "inserted": inserted,
        "converted": conv,
        "denied": neg,
        "pending": inserted - conv - neg
    }), 200


# =========================
# Run local
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)