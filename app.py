from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import hashlib
import requests
import re

# =========================================================
# CONFIG
# =========================================================
app = Flask(__name__)
CORS(app)

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
    """
    Sends a WhatsApp message via Z-API.
    Returns True if request was sent; False if not configured or failed.
    """
    url = zapi_url()
    if not url or not numero:
        return False
    try:
        requests.post(
            url,
            json={"phone": numero, "message": mensagem},
            timeout=10
        )
        return True
    except Exception:
        return False

# =========================================================
# USERS DB (Login)
# =========================================================
def init_users_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_users_db()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def autenticar(email: str, password: str):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT client_id FROM users WHERE email=? AND password=?",
        (email, hash_password(password))
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# =========================================================
# CLIENT DB (one DB per client_id)
# =========================================================
def get_db(client_id: str):
    db_path = os.path.join(DATA_DIR, f"{client_id}.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tempo_site INTEGER,
            paginas_visitadas INTEGER,
            clicou_preco INTEGER,
            virou_cliente INTEGER
        )
    """)
    conn.commit()
    return conn, c

# =========================================================
# ML
# =========================================================
def treinar_modelo(client_id: str):
    conn, _ = get_db(client_id)
    df = pd.read_sql("SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente FROM leads", conn)
    conn.close()

    if df.empty:
        return None

    # We only train if we have at least one positive and one negative label.
    # Here: NULL is ignored (not used as training label).
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
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"erro": "email e senha obrigatórios"}), 400

    client_id = autenticar(email, password)
    if not client_id:
        return jsonify({"erro": "login inválido"}), 401

    return jsonify({"status": "ok", "client_id": client_id})

@app.route("/criar_cliente", methods=["POST"])
def criar_cliente():
    """
    Simple onboarding endpoint (form POST).
    Creates a user in users.db.
    """
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    if not email or not password:
        return "Dados inválidos", 400

    # client_id derived from email prefix (safe-ish). You can customize later.
    client_id = email.split("@")[0]
    # normalize
    client_id = re.sub(r"[^a-z0-9_]+", "_", client_id.lower())

    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (client_id, email, password) VALUES (?, ?, ?)",
            (client_id, email, hash_password(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return "Usuário já existe (email ou client_id).", 409
    conn.close()

    # Ensure client's DB exists (creates table)
    conn2, _ = get_db(client_id)
    conn2.close()

    return f"""
    <h2>Conta criada!</h2>
    <p><b>client_id</b>: {client_id}</p>
    <p>Agora você já pode entrar no dashboard e usar a API.</p>
    """, 200

@app.route("/prever", methods=["POST"])
def prever():
    """
    Receives features + client_id, stores lead, returns probability + flag.
    Optionally sends WhatsApp if lead is hot and WhatsApp configured.
    """
    data = request.get_json(silent=True) or {}

    client_id = data.get("client_id")
    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    # normalize client_id to avoid accidental multiple DBs
    client_id = str(client_id).strip()

    try:
        tempo_site = int(data.get("tempo_site", 0))
        paginas = int(data.get("paginas_visitadas", 0))
        clicou = int(data.get("clicou_preco", 0))
    except Exception:
        return jsonify({"erro": "campos inválidos (tempo_site/paginas_visitadas/clicou_preco devem ser números)"}), 400

    conn, c = get_db(client_id)
    c.execute("""
        INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
        VALUES (?, ?, ?, NULL)
    """, (tempo_site, paginas, clicou))
    conn.commit()
    lead_id = c.lastrowid
    conn.close()

    model = treinar_modelo(client_id)

    if model:
        X = pd.DataFrame([{
            "tempo_site": tempo_site,
            "paginas_visitadas": paginas,
            "clicou_preco": clicou
        }])
        prob = float(model.predict_proba(X)[0][1])
    else:
        # cold-start fallback
        prob = 0.35

    lead_quente = int(prob >= 0.7)

    # WhatsApp only if configured and lead is hot
    if lead_quente == 1 and WHATSAPP_DESTINO:
        enviar_whatsapp(
            WHATSAPP_DESTINO,
            "🔥 Novo lead quente!\n\n"
            f"Client: {client_id}\n"
            f"Lead ID: {lead_id}\n"
            f"Probabilidade: {round(prob, 2)}\n"
            f"Tempo: {tempo_site}s | Páginas: {paginas} | Clicou preço: {'Sim' if clicou else 'Não'}"
        )

    return jsonify({
        "lead_id": lead_id,
        "probabilidade_de_compra": round(prob, 2),
        "lead_quente": lead_quente
    })

@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    """
    Marks a lead as converted (virou_cliente=1).
    IMPORTANT: We do NOT mass-set NULL leads to 0 here (that was breaking things).
    """
    data = request.get_json(silent=True) or {}
    client_id = data.get("client_id")
    lead_id = data.get("lead_id")

    if not client_id or lead_id is None:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    client_id = str(client_id).strip()
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
            f"✅ Venda confirmada!\nClient: {client_id}\nLead ID: {lead_id_int}"
        )

    return jsonify({"status": "venda_confirmada"})

@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    """
    Returns aggregated stats + row data for Chart.js / table.
    """
    client_id = request.args.get("client_id", "").strip()
    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    conn, _ = get_db(client_id)
    df = pd.read_sql("SELECT * FROM leads ORDER BY id DESC", conn)
    conn.close()

    if df.empty:
        return jsonify({
            "total_leads": 0,
            "leads_quentes": 0,
            "leads_frios": 0,
            "dados": []
        })

    # For dashboard classification:
    # virou_cliente == 1 => quente (converted)
    # virou_cliente == 0 => frio (explicitly not converted)
    # NULL => "pendente" (still lead). We'll treat as frio in counts to keep simple,
    # but table will show 'Pendente'.
    total = int(len(df))
    quentes = int((df["virou_cliente"] == 1).sum())
    frios = int((df["virou_cliente"] == 0).sum())
    pendentes = int(df["virou_cliente"].isna().sum())

    # For the existing dashboard.html, keep "leads_frios" as non-converted total (0 + NULL)
    leads_frios_dashboard = frios + pendentes

    # Convert NaN/None safely for JSON
    df_out = df.copy()
    df_out["virou_cliente"] = df_out["virou_cliente"].fillna(-1).astype(int)

    return jsonify({
        "total_leads": total,
        "leads_quentes": quentes,
        "leads_frios": leads_frios_dashboard,
        "dados": df_out.to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
