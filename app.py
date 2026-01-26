from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import socket
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import hashlib
import requests
import re
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
app = Flask(__name__)
CORS(app)

DB_PATH = os.environ.get("DB_PATH", "/var/data/users.db")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_DB = os.path.join(BASE_DIR, "users.db")

# Z-API (WhatsApp) — configure via environment variables on Render
# Example:
#   ZAPI_INSTANCE=xxxx
#   ZAPI_TOKEN=yyyy
#   WHATSAPP_DESTINO=5511999999999
ZAPI_INSTANCE = os.getenv("ZAPI_INSTANCE", "").strip()
ZAPI_TOKEN = os.getenv("ZAPI_TOKEN", "").strip()
WHATSAPP_DESTINO = os.getenv("WHATSAPP_DESTINO", "").strip()


def zapi_url():
    if not ZAPI_INSTANCE or not ZAPI_TOKEN:
        return ""
    return f"https://api.z-api.io/instances/{ZAPI_INSTANCE}/token/{ZAPI_TOKEN}/send-text"


def enviar_whatsapp(numero: str, mensagem: str) -> bool:
    """Sends a WhatsApp message via Z-API. Returns True if request was sent."""
    url = zapi_url()
    if not url or not numero:
        return False
    try:
        requests.post(url, json={"phone": numero, "message": mensagem}, timeout=10)
        return True
    except Exception:
        return False


# =========================================================
# USERS DB (Login)
# =========================================================
def init_users_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def autenticar(email: str, password: str):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT client_id FROM users WHERE email=? AND password=?",
        (email, hash_password(password)),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


init_users_db()


# =========================================================
# CLIENT DB (one DB per client_id)
# =========================================================
LEADS_COLUMNS = [
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("tempo_site", "INTEGER"),
    ("paginas_visitadas", "INTEGER"),
    ("clicou_preco", "INTEGER"),
    ("virou_cliente", "INTEGER"),
    # --- Upgrades (v1.1)
    ("nome", "TEXT"),
    ("telefone", "TEXT"),
    ("email_lead", "TEXT"),
    ("probabilidade", "REAL"),
    ("created_at", "TEXT"),
]


def _ensure_leads_schema(conn: sqlite3.Connection, c: sqlite3.Cursor):
    """Ensures the 'leads' table exists and has all expected columns.
    This lets us deploy upgrades without wiping customer DBs.
    """
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tempo_site INTEGER,
            paginas_visitadas INTEGER,
            clicou_preco INTEGER,
            virou_cliente INTEGER
        )
        """
    )
    # Add missing columns (SQLite has no IF NOT EXISTS for columns)
    existing = {row[1] for row in c.execute("PRAGMA table_info(leads)").fetchall()}
    for col, col_type in LEADS_COLUMNS:
        if col not in existing:
            # 'id' is already created in base schema above
            if col == "id":
                continue
            c.execute(f"ALTER TABLE leads ADD COLUMN {col} {col_type}")
    conn.commit()


def get_db(client_id: str):
    db_path = os.path.join(DATA_DIR, f"{client_id}.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    _ensure_leads_schema(conn, c)
    return conn, c


# =========================================================
# ML
# =========================================================
def treinar_modelo(client_id: str):
    conn, _ = get_db(client_id)
    df = pd.read_sql(
        "SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente FROM leads",
        conn,
    )
    conn.close()

    if df.empty:
        return None

    # Train only on labeled rows (0/1). Pending (NULL) is ignored.
    df_treino = df[df["virou_cliente"].isin([0, 1])].copy()
    if df_treino.empty or df_treino["virou_cliente"].nunique() < 2:
        return None

    X = df_treino[["tempo_site", "paginas_visitadas", "clicou_preco"]]
    y = df_treino["virou_cliente"]

    model = LogisticRegression()
    model.fit(X, y)
    return model


# =========================================================
# ROUTES
# =========================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password")

    if not email or not password:
        return jsonify({"erro": "email e senha obrigatórios"}), 400

    client_id = autenticar(email, password)
    if not client_id:
        return jsonify({"erro": "login inválido"}), 401

    return jsonify({"status": "ok", "client_id": client_id})


@app.route("/criar_cliente", methods=["POST"])
def criar_cliente():
    """Simple onboarding endpoint (form POST). Creates a user in users.db."""
    email = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "").strip()

    if not email or not password:
        return "Dados inválidos", 400

    # client_id derived from email prefix (safe-ish). You can customize later.
    client_id = email.split("@")[0]
    client_id = re.sub(r"[^a-z0-9_]+", "_", client_id.lower()).strip("_")

    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (client_id, email, password) VALUES (?, ?, ?)",
            (client_id, email, hash_password(password)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return "Usuário já existe (email ou client_id).", 409
    conn.close()

    # Ensure client's DB exists (creates table + schema upgrades)
    conn2, _ = get_db(client_id)
    conn2.close()

    return (
        f"""
        <h2>Conta criada!</h2>
        <p><b>client_id</b>: {client_id}</p>
        <p>Agora você já pode entrar no dashboard e usar a API.</p>
        """,
        200,
    )


@app.route("/prever", methods=["POST"])
def prever():
    """Receives features + client_id, stores lead, returns probability + flag.
    Upgrades:
      - accepts optional nome/telefone/email_lead
      - saves probabilidade in DB (so dashboard can show it later)
    """
    data = request.get_json(silent=True) or {}

    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    # Optional lead identity fields (good for sales use)
    nome = (data.get("nome") or "").strip()
    telefone = (data.get("telefone") or "").strip()
    email_lead = (data.get("email_lead") or "").strip().lower()

    try:
        tempo_site = int(data.get("tempo_site", 0))
        paginas = int(data.get("paginas_visitadas", 0))
        clicou = int(data.get("clicou_preco", 0))
    except Exception:
        return (
            jsonify(
                {
                    "erro": "campos inválidos (tempo_site/paginas_visitadas/clicou_preco devem ser números)"
                }
            ),
            400,
        )

    model = treinar_modelo(client_id)

    if model:
        X = pd.DataFrame(
            [
                {
                    "tempo_site": tempo_site,
                    "paginas_visitadas": paginas,
                    "clicou_preco": clicou,
                }
            ]
        )
        prob = float(model.predict_proba(X)[0][1])
    else:
        prob = 0.35  # cold-start fallback

    lead_quente = int(prob >= 0.7)

    conn, c = get_db(client_id)
    c.execute(
        """
        INSERT INTO leads (
            tempo_site, paginas_visitadas, clicou_preco, virou_cliente,
            nome, telefone, email_lead, probabilidade, created_at
        )
        VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        (
            tempo_site,
            paginas,
            clicou,
            nome or None,
            telefone or None,
            email_lead or None,
            prob,
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ),
    )
    conn.commit()
    lead_id = c.lastrowid
    conn.close()

    # WhatsApp only if configured and lead is hot
    if lead_quente == 1 and WHATSAPP_DESTINO:
        enviar_whatsapp(
            WHATSAPP_DESTINO,
            "🔥 Novo lead quente!\n\n"
            f"Client: {client_id}\n"
            f"Lead ID: {lead_id}\n"
            f"Probabilidade: {round(prob, 2)}\n"
            f"Tempo: {tempo_site}s | Páginas: {paginas} | Clicou preço: {'Sim' if clicou else 'Não'}",
        )

    return jsonify(
        {
            "lead_id": lead_id,
            "probabilidade_de_compra": round(prob, 4),
            "lead_quente": lead_quente,
            "modelo_treinado": bool(model),
            "fallback_usado": (model is None),
        }
    )


@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    """Marks a lead as converted (virou_cliente=1)."""
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = data.get("lead_id")

    if not client_id or lead_id is None:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    try:
        lead_id_int = int(lead_id)
    except Exception:
        return jsonify({"erro": "lead_id deve ser número"}), 400

    conn, c = get_db(client_id)
    c.execute("UPDATE leads SET virou_cliente = 1 WHERE id = ?", (lead_id_int,))
    conn.commit()
    updated = c.rowcount
    conn.close()

    if updated == 0:
        return jsonify({"erro": "lead_id não encontrado para este client_id"}), 404

    if WHATSAPP_DESTINO:
        enviar_whatsapp(
            WHATSAPP_DESTINO,
            f"✅ Venda confirmada!\nClient: {client_id}\nLead ID: {lead_id_int}",
        )

    return jsonify({"status": "venda_confirmada"})


@app.route("/negar_venda", methods=["POST"])
def negar_venda():
    """Upgrade: marks a lead as not converted (virou_cliente=0)."""
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    lead_id = data.get("lead_id")

    if not client_id or lead_id is None:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    try:
        lead_id_int = int(lead_id)
    except Exception:
        return jsonify({"erro": "lead_id deve ser número"}), 400

    conn, c = get_db(client_id)
    c.execute("UPDATE leads SET virou_cliente = 0 WHERE id = ?", (lead_id_int,))
    conn.commit()
    updated = c.rowcount
    conn.close()

    if updated == 0:
        return jsonify({"erro": "lead_id não encontrado para este client_id"}), 404

    return jsonify({"status": "venda_negada"})


@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    """Returns aggregated stats + row data for dashboard."""
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    conn, _ = get_db(client_id)
    df = pd.read_sql("SELECT * FROM leads ORDER BY id DESC", conn)
    conn.close()

    if df.empty:
        return jsonify(
            {
                "total_leads": 0,
                "leads_quentes": 0,
                "leads_frios": 0,
                "pendentes": 0,
                "dados": [],
            }
        )

    total = int(len(df))
    convertidos = int((df["virou_cliente"] == 1).sum())
    negados = int((df["virou_cliente"] == 0).sum())
    pendentes = int(df["virou_cliente"].isna().sum())

    # Backward compat for old dashboard
    leads_quentes = convertidos
    leads_frios_dashboard = negados + pendentes

    df_out = df.copy()
    df_out["virou_cliente"] = df_out["virou_cliente"].fillna(-1).astype(int)

    # Ensure missing columns exist in JSON even for old rows
    for col in ["nome", "telefone", "email_lead", "probabilidade", "created_at"]:
        if col not in df_out.columns:
            df_out[col] = None

    return jsonify(
        {
            "total_leads": total,
            "leads_quentes": leads_quentes,
            "leads_frios": leads_frios_dashboard,
            "pendentes": pendentes,
            "convertidos": convertidos,
            "negados": negados,
            "dados": df_out.to_dict(orient="records"),
        }
    )


@app.route("/debug_model", methods=["GET"])
def debug_model():
    """Diagnóstico para entender por que o /prever está retornando fallback (ex: 0.35).

    Retorna:
      - contagem de leads total / convertidos / negados / pendentes
      - quantos rótulos existem (0/1) e se dá para treinar
      - últimos exemplos rotulados e pendentes (para inspeção)
    """
    client_id = (request.args.get("client_id") or "").strip()
    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    conn, _ = get_db(client_id)

    # Pegamos um recorte recente para debug (evita resposta gigante).
    df = pd.read_sql(
        """SELECT id, tempo_site, paginas_visitadas, clicou_preco,
                  virou_cliente, probabilidade, nome, telefone, email_lead, created_at
             FROM leads
             ORDER BY id DESC
             LIMIT 200""",
        conn,
    )
    conn.close()

    total = int(len(df))
    convertidos = int((df["virou_cliente"] == 1).sum()) if total else 0
    negados = int((df["virou_cliente"] == 0).sum()) if total else 0
    pendentes = total - convertidos - negados

    df_labeled = df[df["virou_cliente"].isin([0, 1])] if total else df
    labeled_count = int(len(df_labeled)) if total else 0
    classes = sorted(df_labeled["virou_cliente"].dropna().unique().tolist()) if labeled_count else []
    can_train = (len(classes) >= 2) and (labeled_count >= 4)

    # Explicação simples para o usuário entender o 0.35
    reason = None
    if total == 0:
        reason = "Sem leads no banco ainda."
    elif labeled_count == 0:
        reason = "Sem leads rotulados (virou_cliente=0/1). Todos estão pendentes (-1/NULL)."
    elif len(classes) < 2:
        reason = "Só existe uma classe rotulada (apenas 0 ou apenas 1). Precisa ter 0 e 1."
    elif labeled_count < 4:
        reason = "Poucos exemplos rotulados. Recomendo no mínimo 4 (2 de cada classe) para começar."
    else:
        reason = "Ok — dá para treinar. Se ainda vier 0.35, confira se está usando o mesmo client_id no /prever."

    # Últimos exemplos úteis para inspeção
    last_labeled = (
        df_labeled.head(10).to_dict(orient="records") if labeled_count else []
    )
    last_pending = (
        df[df["virou_cliente"].isin([-1])].head(10).to_dict(orient="records")
        if total
        else []
    )

    return jsonify(
        {
            "client_id": client_id,
            "total_recentes_considerados": total,
            "convertidos": convertidos,
            "negados": negados,
            "pendentes": pendentes,
            "labeled_count": labeled_count,
            "classes_rotuladas": classes,
            "can_train": can_train,
            "reason": reason,
            "last_labeled": last_labeled,
            "last_pending": last_pending,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
