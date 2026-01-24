from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)
CORS(app)

BASE_DIR = "data"
MODELS_DIR = "models"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# UTILIDADES
# -----------------------------

def get_db_path(client_id):
    return os.path.join(BASE_DIR, f"{client_id}.db")

def get_model_path(client_id):
    return os.path.join(MODELS_DIR, f"{client_id}.pkl")

def conectar_db(client_id):
    conn = sqlite3.connect(get_db_path(client_id))
    return conn

def criar_tabela(client_id):
    conn = conectar_db(client_id)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tempo_site INTEGER,
            paginas_visitadas INTEGER,
            clicou_preco INTEGER,
            virou_cliente INTEGER
        )
    """)
    conn.commit()
    conn.close()

def carregar_dados(client_id):
    conn = conectar_db(client_id)
    df = pd.read_sql_query("SELECT * FROM leads", conn)
    conn.close()
    return df

# -----------------------------
# MODELO
# -----------------------------

def treinar_modelo(df):
    df = df.dropna(subset=["virou_cliente"])

    if len(df) < 4:
        return None

    if df["virou_cliente"].nunique() < 2:
        return None

    X = df[["tempo_site", "paginas_visitadas", "clicou_preco"]]
    y = df["virou_cliente"]

    model = LogisticRegression()
    model.fit(X, y)

    return model

def salvar_modelo(client_id, model):
    joblib.dump(model, get_model_path(client_id))

def carregar_modelo(client_id):
    path = get_model_path(client_id)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# -----------------------------
# ROTAS
# -----------------------------

@app.route("/prever", methods=["POST"])
def prever():
    dados = request.json
    client_id = dados.get("client_id")

    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    criar_tabela(client_id)

    tempo_site = dados["tempo_site"]
    paginas = dados["paginas_visitadas"]
    clicou = dados["clicou_preco"]

    df = carregar_dados(client_id)
    model = carregar_modelo(client_id)

    if model is None:
        model = treinar_modelo(df)
        if model:
            salvar_modelo(client_id, model)

    if model:
        prob = model.predict_proba([[tempo_site, paginas, clicou]])[0][1]
    else:
        prob = 0.35  # fallback consciente

    lead_quente = 1 if prob >= 0.7 else 0

    conn = conectar_db(client_id)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
        VALUES (?, ?, ?, NULL)
    """, (tempo_site, paginas, clicou))
    conn.commit()
    conn.close()

    return jsonify({
        "probabilidade_de_compra": round(float(prob), 2),
        "lead_quente": lead_quente
    })

@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    dados = request.json
    client_id = dados.get("client_id")
    lead_id = dados.get("lead_id")

    if not client_id or not lead_id:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    conn = conectar_db(client_id)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE leads SET virou_cliente = 1 WHERE id = ?",
        (lead_id,)
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "venda confirmada"})


@app.route("/dashboard", methods=["GET"])
def dashboard():
    client_id = request.args.get("client_id")
    dias = int(request.args.get("dias", 7))

    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    df = carregar_dados(client_id)

    if df.empty:
        return jsonify([])

    total = len(df)
    quentes = df[df["virou_cliente"] == 1].shape[0]
    frios = total - quentes

    return jsonify({
        "total_leads": total,
        "leads_quentes": quentes,
        "leads_frios": frios
    })

if __name__ == "__main__":
    app.run(debug=True)
