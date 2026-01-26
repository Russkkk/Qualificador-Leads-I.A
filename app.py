from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import re
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from werkzeug.security import generate_password_hash, check_password_hash

# =========================================================
# CONFIG
# =========================================================
app = Flask(__name__)
CORS(app)

DATABASE_URL = os.environ.get("DATABASE_URL")  # Set in Render Environment
DEFAULT_FALLBACK_PROB = float(os.environ.get("FALLBACK_PROB", "0.35"))
LEAD_HOT_THRESHOLD = float(os.environ.get("LEAD_HOT_THRESHOLD", "0.7"))

# Training guardrails (avoid nonsense training on tiny samples)
MIN_LABELED_TOTAL = int(os.environ.get("MIN_LABELED_TOTAL", "4"))  # minimum labeled samples
MIN_PER_CLASS = int(os.environ.get("MIN_PER_CLASS", "2"))          # minimum per class (0 and 1)
TRAIN_MAX_ROWS = int(os.environ.get("TRAIN_MAX_ROWS", "1000"))     # cap rows used for training


# =========================================================
# DB helpers
# =========================================================
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL não está definido. No Render, configure um PostgreSQL e "
            "adicione a env var DATABASE_URL (Internal Database URL)."
        )
    # sslmode=require works for external URLs; internal usually also accepts it.
    return psycopg2.connect(DATABASE_URL, sslmode=os.environ.get("PGSSLMODE", "require"))


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id SERIAL PRIMARY KEY,
                  client_id TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS leads (
                  id SERIAL PRIMARY KEY,
                  client_id TEXT NOT NULL,
                  tempo_site INTEGER,
                  paginas_visitadas INTEGER,
                  clicou_preco INTEGER,
                  virou_cliente INTEGER, -- 1=convertido, 0=negado, NULL=pendente
                  probabilidade DOUBLE PRECISION,
                  nome TEXT,
                  telefone TEXT,
                  email_lead TEXT,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_id ON leads(client_id);")
            conn.commit()
    finally:
        conn.close()


# Initialize schema at startup (Render will call module on boot)
try:
    init_db()
except Exception as e:
    # Don't crash import in some edge cases; Render will show logs.
    # The endpoints will fail with a clear message if DB isn't configured.
    print("DB init warning:", repr(e))


# =========================================================
# Auth helpers
# =========================================================
def sanitize_client_id(email: str) -> str:
    cid = (email.split("@")[0] if "@" in email else email).strip().lower()
    cid = re.sub(r"[^a-z0-9_]+", "_", cid).strip("_")
    return cid or "cliente"


def authenticate(email: str, password: str):
    email = (email or "").strip().lower()
    password = (password or "").strip()

    if not email or not password:
        return None

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT client_id, password_hash FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
            if not row:
                return None
            if not check_password_hash(row["password_hash"], password):
                return None
            return row["client_id"]
    finally:
        conn.close()


# =========================================================
# ML helpers
# =========================================================
def _fetch_training_rows(client_id: str):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente
                FROM leads
                WHERE client_id=%s AND virou_cliente IS NOT NULL
                ORDER BY id DESC
                LIMIT %s
                """,
                (client_id, TRAIN_MAX_ROWS),
            )
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def _can_train(rows):
    if not rows:
        return False, "Sem leads rotulados (virou_cliente 0/1) ainda."
    y = np.array([int(r["virou_cliente"]) for r in rows], dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    classes = classes.tolist()
    count_map = {int(c): int(n) for c, n in zip(classes, counts)}

    if len(classes) < 2:
        return False, "Só existe uma classe rotulada (apenas 0 ou apenas 1)."
    if len(rows) < MIN_LABELED_TOTAL:
        return False, f"Poucos exemplos rotulados. Recomendo no mínimo {MIN_LABELED_TOTAL} ({MIN_PER_CLASS} de cada classe) para começar."
    if count_map.get(0, 0) < MIN_PER_CLASS or count_map.get(1, 0) < MIN_PER_CLASS:
        return False, f"Poucos exemplos por classe. Precisa de pelo menos {MIN_PER_CLASS} negados (0) e {MIN_PER_CLASS} convertidos (1)."
    return True, "OK"


def _train_model(client_id: str):
    rows = _fetch_training_rows(client_id)
    ok, reason = _can_train(rows)
    if not ok:
        return None, reason

    X = np.array([[r["tempo_site"] or 0, r["paginas_visitadas"] or 0, r["clicou_preco"] or 0] for r in rows], dtype=float)
    y = np.array([int(r["virou_cliente"]) for r in rows], dtype=int)

    # Simple model for MVP
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, "Treinado"


def _predict_prob(model, tempo_site, paginas_visitadas, clicou_preco):
    X = np.array([[tempo_site, paginas_visitadas, clicou_preco]], dtype=float)
    prob = float(model.predict_proba(X)[0][1])
    return prob


# =========================================================
# ROUTES
# =========================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/criar_cliente", methods=["POST"])
def criar_cliente():
    """Onboarding: creates user and returns client_id."""
    email = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "").strip()

    if not email or not password:
        return "Dados inválidos", 400

    client_id = sanitize_client_id(email)
    pw_hash = generate_password_hash(password)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (client_id, email, password_hash) VALUES (%s, %s, %s)",
                (client_id, email, pw_hash),
            )
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return "Usuário já existe (email ou client_id).", 409
    finally:
        conn.close()

    # keep compatibility with your onboarding.html regex extraction
    return f"✅ Conta criada! <b>client_id</b>: {client_id}"


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    client_id = authenticate(email, password)
    if not client_id:
        return jsonify({"erro": "login inválido"}), 401

    return jsonify({"status": "ok", "client_id": client_id})


@app.route("/prever", methods=["POST"])
def prever():
    data = request.get_json(silent=True) or {}

    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"erro": "client_id é obrigatório"}), 400

    tempo_site = int(data.get("tempo_site") or 0)
    paginas_visitadas = int(data.get("paginas_visitadas") or 0)
    clicou_preco = int(data.get("clicou_preco") or 0)

    nome = (data.get("nome") or "").strip()
    telefone = (data.get("telefone") or "").strip()
    email_lead = (data.get("email_lead") or "").strip()

    model, reason = _train_model(client_id)
    if model is None:
        prob = DEFAULT_FALLBACK_PROB
        modelo_treinado = False
        fallback_usado = True
    else:
        prob = _predict_prob(model, tempo_site, paginas_visitadas, clicou_preco)
        modelo_treinado = True
        fallback_usado = False

    lead_quente = 1 if prob >= LEAD_HOT_THRESHOLD else 0

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO leads (client_id, tempo_site, paginas_visitadas, clicou_preco,
                                   virou_cliente, probabilidade, nome, telefone, email_lead, created_at)
                VALUES (%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    client_id,
                    tempo_site,
                    paginas_visitadas,
                    clicou_preco,
                    prob,
                    nome if nome else None,
                    telefone if telefone else None,
                    email_lead if email_lead else None,
                    datetime.now(timezone.utc),
                ),
            )
            lead_id = cur.fetchone()[0]
            conn.commit()
    finally:
        conn.close()

    return jsonify(
        {
            "lead_id": lead_id,
            "probabilidade_de_compra": prob,
            "lead_quente": lead_quente,
            "modelo_treinado": modelo_treinado,
            "fallback_usado": fallback_usado,
            "treino_status": reason,
        }
    )


@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = data.get("lead_id")

    if not client_id or lead_id is None:
        return jsonify({"erro": "client_id e lead_id são obrigatórios"}), 400

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE leads SET virou_cliente=1 WHERE client_id=%s AND id=%s",
                (client_id, int(lead_id)),
            )
            conn.commit()
            if cur.rowcount == 0:
                return jsonify({"erro": "Lead não encontrado"}), 404
    finally:
        conn.close()

    return jsonify({"status": "ok", "lead_id": int(lead_id), "virou_cliente": 1})


@app.route("/negar_venda", methods=["POST"])
def negar_venda():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = data.get("lead_id")

    if not client_id or lead_id is None:
        return jsonify({"erro": "client_id e lead_id são obrigatórios"}), 400

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE leads SET virou_cliente=0 WHERE client_id=%s AND id=%s",
                (client_id, int(lead_id)),
            )
            conn.commit()
            if cur.rowcount == 0:
                return jsonify({"erro": "Lead não encontrado"}), 404
    finally:
        conn.close()

    return jsonify({"status": "ok", "lead_id": int(lead_id), "virou_cliente": 0})


@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"erro": "client_id é obrigatório"}), 400

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, tempo_site, paginas_visitadas, clicou_preco,
                       COALESCE(virou_cliente, -1) AS virou_cliente,
                       probabilidade, nome, telefone, email_lead, created_at
                FROM leads
                WHERE client_id=%s
                ORDER BY id DESC
                """,
                (client_id,),
            )
            rows = cur.fetchall()

        total = len(rows)
        convertidos = sum(1 for r in rows if int(r["virou_cliente"]) == 1)
        negados = sum(1 for r in rows if int(r["virou_cliente"]) == 0)
        pendentes = total - convertidos - negados

        return jsonify(
            {
                "client_id": client_id,
                "total_leads": total,
                "convertidos": convertidos,
                "negados": negados,
                "pendentes": pendentes,
                "dados": rows,
            }
        )
    finally:
        conn.close()


@app.route("/debug_model", methods=["GET"])
def debug_model():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"erro": "client_id é obrigatório"}), 400

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, tempo_site, paginas_visitadas, clicou_preco,
                       COALESCE(virou_cliente, -1) AS virou_cliente,
                       probabilidade, nome, telefone, email_lead, created_at
                FROM leads
                WHERE client_id=%s
                ORDER BY id DESC
                LIMIT 2000
                """,
                (client_id,),
            )
            all_rows = cur.fetchall()

            labeled = [r for r in all_rows if int(r["virou_cliente"]) in (0, 1)]
            pending = [r for r in all_rows if int(r["virou_cliente"]) not in (0, 1)]

            convertidos = sum(1 for r in labeled if int(r["virou_cliente"]) == 1)
            negados = sum(1 for r in labeled if int(r["virou_cliente"]) == 0)

            classes_rotuladas = sorted(list({float(int(r["virou_cliente"])) for r in labeled}))
            labeled_count = len(labeled)

            can_train = False
            reason = "Sem leads no banco ainda."
            if all_rows:
                # reuse can-train logic, but on labeled subset
                ok, why = _can_train([{"virou_cliente": r["virou_cliente"], "tempo_site": r["tempo_site"], "paginas_visitadas": r["paginas_visitadas"], "clicou_preco": r["clicou_preco"]} for r in labeled])
                can_train = ok
                reason = why

            return jsonify(
                {
                    "client_id": client_id,
                    "total_recentes_considerados": len(all_rows),
                    "labeled_count": labeled_count,
                    "classes_rotuladas": classes_rotuladas,
                    "convertidos": convertidos,
                    "negados": negados,
                    "pendentes": len(pending),
                    "can_train": can_train,
                    "reason": reason,
                    "last_labeled": labeled[:10],
                    "last_pending": pending[:10],
                }
            )
    finally:
        conn.close()
        
        
@app.get("/recalc_pending")
def recalc_pending():
    client_id = request.args.get("client_id", "").strip()
    limit = int(request.args.get("limit", "200"))  # quantos pendentes recalcular por chamada (pra não pesar)
    if not client_id:
        return jsonify({"error": "client_id é obrigatório"}), 400

    conn = get_conn()
    cur = conn.cursor()

    # 1) Buscar rotulados (0/1) para treino
    cur.execute("""
        SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente
        FROM leads
        WHERE client_id = %s
          AND virou_cliente IN (0, 1)
        ORDER BY id DESC
        LIMIT 2000
    """, (client_id,))
    labeled = cur.fetchall()

    if len(labeled) < 4:
        conn.close()
        return jsonify({
            "ok": False,
            "reason": "Poucos exemplos rotulados. Rotule no mínimo 4 (2 de cada classe).",
            "labeled_count": len(labeled)
        }), 400

    y = np.array([row[3] for row in labeled], dtype=float)
    # precisa ter as duas classes
    if len(set(y.tolist())) < 2:
        conn.close()
        return jsonify({
            "ok": False,
            "reason": "Só existe 1 classe rotulada (apenas 0 ou apenas 1).",
            "classes": sorted(list(set(y.tolist())))
        }), 400

    X = np.array([[row[0], row[1], row[2]] for row in labeled], dtype=float)

    # 2) Treinar modelo
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(Xs, y)

    # 3) Buscar pendentes para recalcular
    cur.execute("""
        SELECT id, tempo_site, paginas_visitadas, clicou_preco
        FROM leads
        WHERE client_id = %s
          AND (virou_cliente IS NULL OR virou_cliente = -1)
        ORDER BY id DESC
        LIMIT %s
    """, (client_id, limit))
    pending = cur.fetchall()

    if not pending:
        conn.close()
        return jsonify({
            "ok": True,
            "updated": 0,
            "reason": "Não há leads pendentes para recalcular."
        }), 200

    # 4) Recalcular e atualizar
    ids = [p[0] for p in pending]
    Xp = np.array([[p[1], p[2], p[3]] for p in pending], dtype=float)
    Xps = scaler.transform(Xp)

    probs = model.predict_proba(Xps)[:, 1]  # prob de classe 1

    # update em lote
    updates = list(zip([float(p) for p in probs], ids))
    cur.executemany("""
        UPDATE leads
        SET probabilidade = %s
        WHERE id = %s
    """, updates)

    conn.commit()
    conn.close()

    return jsonify({
        "ok": True,
        "updated": len(ids),
        "limit": limit,
        "min_prob": float(np.min(probs)),
        "max_prob": float(np.max(probs)),
        "sample": [{"id": ids[i], "prob": float(probs[i])} for i in range(min(5, len(ids)))]
    }), 200


if __name__ == "__main__":
    # Render uses PORT env var
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
