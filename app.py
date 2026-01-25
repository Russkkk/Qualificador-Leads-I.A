import os
import sqlite3
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIGURAÇÕES BÁSICAS
# =========================

app = Flask(__name__)
CORS(app)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# =========================
# FUNÇÕES AUXILIARES
# =========================

def get_db(client_id):
    path = f"data/{client_id}.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def criar_tabela(client_id):
    conn = get_db(client_id)
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


def treinar_modelo(client_id):
    conn = get_db(client_id)
    df = pd.read_sql("""
        SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente
        FROM leads
        WHERE virou_cliente IS NOT NULL
    """, conn)
    conn.close()

    if len(df) < 2:
        return None

    X = df[["tempo_site", "paginas_visitadas", "clicou_preco"]]
    y = df["virou_cliente"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, f"models/{client_id}.joblib")
    return model


def carregar_modelo(client_id):
    path = f"models/{client_id}.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return None


# =========================
# ROTAS
# =========================

@app.route("/prever", methods=["POST"])
def prever():
    data = request.get_json(silent=True) or {}

    client_id = data.get("client_id")
    tempo_site = data.get("tempo_site")
    paginas_visitadas = data.get("paginas_visitadas")
    clicou_preco = data.get("clicou_preco")

    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    criar_tabela(client_id)

    conn = get_db(client_id)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
        VALUES (?, ?, ?, NULL)
    """, (tempo_site, paginas_visitadas, clicou_preco))

    conn.commit()

    lead_id = cursor.lastrowid
    conn.close()

    model = carregar_modelo(client_id)

    if not model:
        prob = 0.35
    else:
        X = [[tempo_site, paginas_visitadas, clicou_preco]]
        prob = float(model.predict_proba(X)[0][1])

    return jsonify({
        "lead_id": lead_id,
        "probabilidade_de_compra": round(prob, 2),
        "lead_quente": int(prob >= 0.6)
    })


@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    data = request.get_json(silent=True) or {}

    client_id = data.get("client_id")
    lead_id = data.get("lead_id")

    if not client_id or not lead_id:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    criar_tabela(client_id)

    conn = get_db(client_id)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE leads SET virou_cliente = 1 WHERE id = ?",
        (lead_id,)
    )

    conn.commit()
    conn.close()

    treinar_modelo(client_id)

    return jsonify({"status": "venda_confirmada"})


# =========================
# START
# =========================

if __name__ == "__main__":
    app.run(debug=True)
