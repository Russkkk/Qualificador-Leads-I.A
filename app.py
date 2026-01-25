from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import hashlib
import requests

# =====================
# APP CONFIG
# =====================
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_DB = os.path.join(BASE_DIR, "users.db")

ZAPI_URL = "https://api.z-api.io/instances/SEU_ID/token/SEU_TOKEN/send-text"
WHATSAPP_DESTINO = "5518999999999"

# =====================
# WHATSAPP
# =====================
def enviar_whatsapp(numero, mensagem):
    try:
        requests.post(
            ZAPI_URL,
            json={"phone": numero, "message": mensagem},
            timeout=5
        )
    except:
        pass

# =====================
# USERS DB
# =====================
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

# =====================
# CLIENT DB
# =====================
def get_db(client_id):
    path = os.path.join(DATA_DIR, f"{client_id}.db")
    conn = sqlite3.connect(path)
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

# =====================
# IA
# =====================
def treinar_modelo(client_id):
    conn, _ = get_db(client_id)
    df = pd.read_sql("SELECT * FROM leads", conn)

    if df.empty or df["virou_cliente"].isnull().all():
        return None

    df["virou_cliente"] = df["virou_cliente"].fillna(0)

    if df["virou_cliente"].nunique() < 2:
        return None

    X = df[["tempo_site", "paginas_visitadas", "clicou_preco"]]
    y = df["virou_cliente"]

    model = LogisticRegression()
    model.fit(X, y)
    return model

# =====================
# LOGIN
# =====================
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    senha = hashlib.sha256(password.encode()).hexdigest()

    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT client_id FROM users WHERE email=? AND password=?",
        (email, senha)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"erro": "login inválido"}), 401

    return jsonify({"client_id": row[0]})

# =====================
# PREVER
# =====================
@app.route("/prever", methods=["POST"])
def prever():
    data = request.get_json()
    client_id = data.get("client_id")

    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    tempo = int(data.get("tempo_site", 0))
    paginas = int(data.get("paginas_visitadas", 0))
    clicou = int(data.get("clicou_preco", 0))

    conn, c = get_db(client_id)

    c.execute("""
        INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
        VALUES (?, ?, ?, NULL)
    """, (tempo, paginas, clicou))
    conn.commit()
    lead_id = c.lastrowid

    model = treinar_modelo(client_id)

    prob = float(model.predict_proba([[tempo, paginas, clicou]])[0][1]) if model else 0.35
    lead_quente = prob >= 0.7

    if lead_quente:
        enviar_whatsapp(
            WHATSAPP_DESTINO,
            f"🔥 Lead quente!\nTempo: {tempo}s\nPáginas: {paginas}\nClicou preço: {clicou}"
        )

    return jsonify({
        "lead_id": lead_id,
        "probabilidade": round(prob, 2),
        "lead_quente": int(lead_quente)
    })

# =====================
# CONFIRMAR VENDA
# =====================
@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    data = request.get_json()
    client_id = data.get("client_id")
    lead_id = data.get("lead_id")

    if not client_id or not lead_id:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    conn, c = get_db(client_id)
    c.execute("UPDATE leads SET virou_cliente = 1 WHERE id = ?", (lead_id,))
    conn.commit()

    return jsonify({"status": "ok"})

# =====================
# START
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
