# app.py - Qualificador de Leads IA (Postgres / Render)
import os
import re
from datetime import datetime, timezone

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")  # required on Render
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.7"))
FALLBACK_PROB = float(os.environ.get("FALLBACK_PROB", "0.35"))

# Minimum labeled examples to train / set threshold
MIN_LABELED_TO_TRAIN = int(os.environ.get("MIN_LABELED_TO_TRAIN", "4"))   # 2 of each class recommended
MIN_LABELED_TO_AUTO_THRESHOLD = int(os.environ.get("MIN_LABELED_TO_AUTO_THRESHOLD", "10"))

app = Flask(__name__)
CORS(app)

# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada. Defina no Render (Environment) ou no seu ambiente local.")
    # Render Postgres requires SSL
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            client_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # Leads table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id SERIAL PRIMARY KEY,
            client_id TEXT NOT NULL,
            tempo_site DOUBLE PRECISION NOT NULL,
            paginas_visitadas INTEGER NOT NULL,
            clicou_preco INTEGER NOT NULL,
            virou_cliente INTEGER NOT NULL DEFAULT -1, -- -1 pendente, 0 negado, 1 convertido
            probabilidade DOUBLE PRECISION,
            nome TEXT,
            telefone TEXT,
            email_lead TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_id ON leads (client_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_status ON leads (client_id, virou_cliente)")

    # Client settings (threshold)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS client_settings (
            client_id TEXT PRIMARY KEY,
            threshold DOUBLE PRECISION NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    conn.commit()
    conn.close()

def get_threshold(cur, client_id: str):
    cur.execute("SELECT threshold FROM client_settings WHERE client_id=%s", (client_id,))
    row = cur.fetchone()
    return float(row[0]) if row else None

def set_threshold(cur, client_id: str, threshold: float):
    cur.execute("""
      INSERT INTO client_settings (client_id, threshold, updated_at)
      VALUES (%s, %s, NOW())
      ON CONFLICT (client_id)
      DO UPDATE SET threshold=EXCLUDED.threshold, updated_at=NOW()
    """, (client_id, float(threshold)))

def slugify_client_id(email: str) -> str:
    base = (email.split("@")[0] if "@" in email else email).strip().lower()
    base = re.sub(r"[^a-z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "cliente"

def ensure_unique_client_id(cur, base_id: str) -> str:
    candidate = base_id
    i = 2
    while True:
        cur.execute("SELECT 1 FROM users WHERE client_id=%s", (candidate,))
        if not cur.fetchone():
            return candidate
        candidate = f"{base_id}_{i}"
        i += 1

# -----------------------------
# ML helpers
# -----------------------------
def train_model_for_client(cur, client_id: str, max_rows: int = 5000):
    """Train model on labeled leads (virou_cliente in 0/1)."""
    cur.execute("""
        SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente
        FROM leads
        WHERE client_id=%s AND virou_cliente IN (0,1)
        ORDER BY id DESC
        LIMIT %s
    """, (client_id, max_rows))
    rows = cur.fetchall()

    labeled_count = len(rows)
    if labeled_count < MIN_LABELED_TO_TRAIN:
        return None, None, {
            "can_train": False,
            "reason": f"Poucos exemplos rotulados. Recomendo no mínimo {MIN_LABELED_TO_TRAIN} (2 de cada classe) para começar.",
            "labeled_count": labeled_count,
            "classes": []
        }

    y = np.array([float(r[3]) for r in rows], dtype=float)
    classes = sorted(list(set(y.tolist())))
    if len(classes) < 2:
        return None, None, {
            "can_train": False,
            "reason": "Só existe 1 classe rotulada (apenas 0 ou apenas 1).",
            "labeled_count": labeled_count,
            "classes": classes
        }

    X = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows], dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=300)
    model.fit(Xs, y)

    return model, scaler, {
        "can_train": True,
        "reason": "Modelo treinado com sucesso.",
        "labeled_count": labeled_count,
        "classes": classes
    }

def iso_utc(dt):
    if dt and isinstance(dt, datetime):
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/criar_cliente")
def criar_cliente():
    email = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "").strip()

    if not email or not password:
        return "Email e senha são obrigatórios.", 400

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT client_id FROM users WHERE email=%s", (email,))
    if cur.fetchone():
        conn.close()
        return "Usuário já existe.", 409

    base_id = slugify_client_id(email)
    client_id = ensure_unique_client_id(cur, base_id)

    pw_hash = generate_password_hash(password)
    cur.execute(
        "INSERT INTO users (client_id, email, password_hash) VALUES (%s,%s,%s)",
        (client_id, email, pw_hash)
    )

    conn.commit()
    conn.close()

    return f"""
    <html><body style="font-family:Arial">
      <h2>Conta criada ✅</h2>
      <p><b>client_id</b>: {client_id}</p>
      <p>Abra o dashboard (Static Site) e use esse client_id.</p>
    </body></html>
    """, 200

@app.post("/login")
def login():
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email e senha são obrigatórios."}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT client_id, password_hash FROM users WHERE email=%s", (email,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Credenciais inválidas."}), 401

    client_id, pw_hash = row[0], row[1]
    if not check_password_hash(pw_hash, password):
        return jsonify({"error": "Credenciais inválidas."}), 401

    return jsonify({"ok": True, "client_id": client_id}), 200

@app.post("/prever")
def prever():
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"error": "client_id é obrigatório"}), 400

    try:
        tempo_site = float(data.get("tempo_site"))
        paginas_visitadas = int(data.get("paginas_visitadas"))
        clicou_preco = int(data.get("clicou_preco"))
    except Exception:
        return jsonify({"error": "Campos inválidos: tempo_site, paginas_visitadas, clicou_preco"}), 400

    nome = (data.get("nome") or "").strip() or None
    telefone = (data.get("telefone") or "").strip() or None
    email_lead = (data.get("email_lead") or "").strip() or None

    conn = get_conn()
    cur = conn.cursor()

    model, scaler, train_status = train_model_for_client(cur, client_id)

    if model is None:
        prob = FALLBACK_PROB
        modelo_treinado = False
        fallback_usado = True
        threshold = DEFAULT_THRESHOLD
    else:
        X = np.array([[tempo_site, paginas_visitadas, clicou_preco]], dtype=float)
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[0, 1])
        modelo_treinado = True
        fallback_usado = False

        th_saved = get_threshold(cur, client_id)
        threshold = float(th_saved) if th_saved is not None else DEFAULT_THRESHOLD

    lead_quente = bool(prob >= threshold)

    cur.execute("""
        INSERT INTO leads (client_id, tempo_site, paginas_visitadas, clicou_preco, virou_cliente,
                           probabilidade, nome, telefone, email_lead)
        VALUES (%s,%s,%s,%s,-1,%s,%s,%s,%s)
        RETURNING id
    """, (client_id, tempo_site, paginas_visitadas, clicou_preco, prob, nome, telefone, email_lead))
    lead_id = cur.fetchone()[0]

    conn.commit()
    conn.close()

    return jsonify({
        "ok": True,
        "lead_id": lead_id,
        "probabilidade_de_compra": prob,
        "lead_quente": lead_quente,
        "threshold_usado": threshold,
        "modelo_treinado": modelo_treinado,
        "fallback_usado": fallback_usado,
        "treino_status": train_status
    }), 200

@app.get("/dashboard_data")
def dashboard_data():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"error": "client_id é obrigatório"}), 400

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
          COUNT(*) AS total_leads,
          SUM(CASE WHEN virou_cliente = 1 THEN 1 ELSE 0 END) AS convertidos,
          SUM(CASE WHEN virou_cliente = 0 THEN 1 ELSE 0 END) AS negados,
          SUM(CASE WHEN virou_cliente NOT IN (0,1) THEN 1 ELSE 0 END) AS pendentes
        FROM leads
        WHERE client_id = %s
    """, (client_id,))
    summary = cur.fetchone() or {}

    cur.execute("""
        SELECT id, tempo_site, paginas_visitadas, clicou_preco, virou_cliente,
               probabilidade, nome, telefone, email_lead, created_at
        FROM leads
        WHERE client_id=%s
        ORDER BY id DESC
        LIMIT 200
    """, (client_id,))
    dados = cur.fetchall()
    conn.close()

    for d in dados:
        d["created_at"] = iso_utc(d.get("created_at"))

    return jsonify({
        "total_leads": int(summary.get("total_leads") or 0),
        "convertidos": int(summary.get("convertidos") or 0),
        "negados": int(summary.get("negados") or 0),
        "pendentes": int(summary.get("pendentes") or 0),
        "dados": dados
    }), 200

@app.post("/confirmar_venda")
def confirmar_venda():
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = data.get("lead_id")

    if not client_id or lead_id is None:
        return jsonify({"error": "client_id e lead_id são obrigatórios"}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE leads SET virou_cliente=1 WHERE client_id=%s AND id=%s", (client_id, int(lead_id)))
    updated = cur.rowcount
    conn.commit()
    conn.close()

    if updated == 0:
        return jsonify({"ok": False, "reason": "Lead não encontrado"}), 404
    return jsonify({"ok": True}), 200

@app.post("/negar_venda")
def negar_venda():
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = data.get("lead_id")

    if not client_id or lead_id is None:
        return jsonify({"error": "client_id e lead_id são obrigatórios"}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE leads SET virou_cliente=0 WHERE client_id=%s AND id=%s", (client_id, int(lead_id)))
    updated = cur.rowcount
    conn.commit()
    conn.close()

    if updated == 0:
        return jsonify({"ok": False, "reason": "Lead não encontrado"}), 404
    return jsonify({"ok": True}), 200

@app.get("/debug_model")
def debug_model():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"error": "client_id é obrigatório"}), 400

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN virou_cliente = 1 THEN 1 ELSE 0 END) AS convertidos,
          SUM(CASE WHEN virou_cliente = 0 THEN 1 ELSE 0 END) AS negados,
          SUM(CASE WHEN virou_cliente NOT IN (0,1) THEN 1 ELSE 0 END) AS pendentes
        FROM leads
        WHERE client_id=%s
    """, (client_id,))
    s = cur.fetchone() or {}

    cur.execute("""
        SELECT id, tempo_site, paginas_visitadas, clicou_preco, virou_cliente,
               probabilidade, nome, telefone, email_lead, created_at
        FROM leads
        WHERE client_id=%s AND virou_cliente IN (0,1)
        ORDER BY id DESC
        LIMIT 10
    """, (client_id,))
    last_labeled = cur.fetchall()

    cur.execute("""
        SELECT id, tempo_site, paginas_visitadas, clicou_preco, virou_cliente,
               probabilidade, nome, telefone, email_lead, created_at
        FROM leads
        WHERE client_id=%s AND virou_cliente NOT IN (0,1)
        ORDER BY id DESC
        LIMIT 10
    """, (client_id,))
    last_pending = cur.fetchall()

    cur.execute("""
        SELECT DISTINCT virou_cliente
        FROM leads
        WHERE client_id=%s AND virou_cliente IN (0,1)
        ORDER BY virou_cliente
    """, (client_id,))
    classes = [float(r["virou_cliente"]) for r in cur.fetchall()]

    total = int(s.get("total") or 0)
    convertidos = int(s.get("convertidos") or 0)
    negados = int(s.get("negados") or 0)
    pendentes = int(s.get("pendentes") or 0)
    labeled_count = convertidos + negados

    can_train = labeled_count >= MIN_LABELED_TO_TRAIN and (0.0 in classes and 1.0 in classes)
    if total == 0:
        reason = "Sem leads no banco ainda."
    elif not (0.0 in classes and 1.0 in classes):
        reason = "Precisa ter exemplos rotulados das duas classes (0 e 1)."
    elif labeled_count < MIN_LABELED_TO_TRAIN:
        reason = f"Poucos exemplos rotulados. Recomendo no mínimo {MIN_LABELED_TO_TRAIN} (2 de cada classe) para começar."
    else:
        reason = "Pronto para treinar."

    # iso
    for d in last_labeled:
        d["created_at"] = iso_utc(d.get("created_at"))
    for d in last_pending:
        d["created_at"] = iso_utc(d.get("created_at"))

    conn.close()

    return jsonify({
        "client_id": client_id,
        "total_recentes_considerados": total,
        "convertidos": convertidos,
        "negados": negados,
        "pendentes": pendentes,
        "labeled_count": labeled_count,
        "classes_rotuladas": classes,
        "can_train": bool(can_train),
        "reason": reason,
        "last_labeled": last_labeled,
        "last_pending": last_pending
    }), 200

@app.get("/recalc_pending")
def recalc_pending():
    client_id = (request.args.get("client_id") or "").strip()
    limit = int(request.args.get("limit", "200"))
    if not client_id:
        return jsonify({"error": "client_id é obrigatório"}), 400

    conn = get_conn()
    cur = conn.cursor()

    model, scaler, train_status = train_model_for_client(cur, client_id)
    if model is None:
        conn.close()
        return jsonify({"ok": False, **train_status}), 400

    cur.execute("""
        SELECT id, tempo_site, paginas_visitadas, clicou_preco
        FROM leads
        WHERE client_id=%s AND virou_cliente NOT IN (0,1)
        ORDER BY id DESC
        LIMIT %s
    """, (client_id, limit))
    pending = cur.fetchall()

    if not pending:
        conn.close()
        return jsonify({"ok": True, "updated": 0, "reason": "Não há leads pendentes para recalcular."}), 200

    ids = [int(p[0]) for p in pending]
    Xp = np.array([[float(p[1]), float(p[2]), float(p[3])] for p in pending], dtype=float)
    Xps = scaler.transform(Xp)
    probs = model.predict_proba(Xps)[:, 1].astype(float)

    # Update in batch
    cur.executemany(
        "UPDATE leads SET probabilidade=%s WHERE client_id=%s AND id=%s",
        [(float(probs[i]), client_id, ids[i]) for i in range(len(ids))]
    )

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

@app.post("/auto_threshold")
def auto_threshold():
    data = request.get_json(force=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"error": "client_id é obrigatório"}), 400

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente
        FROM leads
        WHERE client_id=%s AND virou_cliente IN (0,1)
        ORDER BY id DESC
        LIMIT 5000
    """, (client_id,))
    rows = cur.fetchall()

    if len(rows) < MIN_LABELED_TO_AUTO_THRESHOLD:
        conn.close()
        return jsonify({
            "ok": False,
            "reason": f"Poucos exemplos rotulados. Recomendo pelo menos {MIN_LABELED_TO_AUTO_THRESHOLD} para threshold automático.",
            "labeled_count": len(rows)
        }), 400

    y = np.array([float(r[3]) for r in rows], dtype=float)
    classes = sorted(list(set(y.tolist())))
    if len(classes) < 2:
        conn.close()
        return jsonify({"ok": False, "reason": "Precisa ter 0 e 1 rotulados.", "classes": classes}), 400

    X = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows], dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=300)
    model.fit(Xs, y)

    probs = model.predict_proba(Xs)[:, 1].astype(float)

    best = {"threshold": DEFAULT_THRESHOLD, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (probs >= t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f = f1_score(y, pred, zero_division=0)
        if f > best["f1"]:
            best = {"threshold": float(t), "f1": float(f), "precision": float(p), "recall": float(r)}

    set_threshold(cur, client_id, best["threshold"])
    conn.commit()
    conn.close()

    return jsonify({"ok": True, "client_id": client_id, **best}), 200


@app.get("/metrics")
def metrics():
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"error": "client_id é obrigatório"}), 400

    test_size = float(request.args.get("test_size", "0.2"))
    if not (0.1 <= test_size <= 0.5):
        return jsonify({"error": "test_size deve estar entre 0.1 e 0.5"}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente
        FROM leads
        WHERE client_id = %s AND virou_cliente IN (0,1)
        ORDER BY id DESC
        LIMIT 5000
    """, (client_id,))
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 20:
        return jsonify({
            "ok": False,
            "reason": "Poucos rotulados (recomendo >= 20).",
            "labeled_count": len(rows)
        }), 400

    X = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows], dtype=float)
    y = np.array([int(r[3]) for r in rows], dtype=int)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos < 2 or neg < 2:
        return jsonify({
            "ok": False,
            "reason": "Precisa de pelo menos 2 positivos e 2 negados.",
            "positivos": pos,
            "negados": neg
        }), 400

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, y_train)

    probs = model.predict_proba(Xte)[:, 1].astype(float)

    # usa threshold salvo se existir
    conn = get_conn()
    cur = conn.cursor()
    th_saved = get_threshold(cur, client_id)
    conn.close()
    threshold = float(th_saved) if th_saved is not None else DEFAULT_THRESHOLD

    pred = (probs >= threshold).astype(int)

    precision = float(precision_score(y_test, pred, zero_division=0))
    recall = float(recall_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    return jsonify({
        "ok": True,
        "client_id": client_id,
        "labeled_count": len(rows),
        "positivos": pos,
        "negados": neg,
        "test_size": test_size,
        "threshold_usado": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }), 200

# -----------------------------
# Boot
# -----------------------------
# On Render with gunicorn, __main__ isn't executed.
# We'll init DB at import time safely.
try:
    init_db()
except Exception as e:
    # Avoid crashing import in some contexts; endpoints will raise if DB is misconfigured.
    print("DB init warning:", str(e))
    
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
