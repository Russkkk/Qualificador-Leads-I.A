from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# Banco por cliente
# ---------------------------
def get_db(client_id):
    db_path = os.path.join(DATA_DIR, f"{client_id}.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
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
    return conn, cursor


# ---------------------------
# Treinar modelo
# ---------------------------
def treinar_modelo(client_id):
    conn, cursor = get_db(client_id)

    df = pd.read_sql("SELECT * FROM leads", conn)

    if df.empty:
        return None

    df["virou_cliente"] = df["virou_cliente"].fillna(0)

    if df["virou_cliente"].nunique() < 2:
        return None

    X = df[["tempo_site", "paginas_visitadas", "clicou_preco"]]
    y = df["virou_cliente"]

    model = LogisticRegression()
    model.fit(X, y)

    return model


# ---------------------------
# PREVER LEAD
# ---------------------------
@app.route("/prever", methods=["POST"])
def prever():
    data = request.get_json()

    client_id = data.get("client_id")
    tempo_site = data.get("tempo_site")
    paginas = data.get("paginas_visitadas")
    clicou = data.get("clicou_preco")

    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    conn, cursor = get_db(client_id)

    cursor.execute("""
        INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
        VALUES (?, ?, ?, NULL)
    """, (tempo_site, paginas, clicou))
    conn.commit()

    lead_id = cursor.lastrowid

    model = treinar_modelo(client_id)

    if model:
        prob = model.predict_proba([[tempo_site, paginas, clicou]])[0][1]
    else:
        prob = 0.35

    return jsonify({
        "lead_id": lead_id,
        "probabilidade_de_compra": round(float(prob), 2),
        "lead_quente": 1 if prob >= 0.7 else 0
    })


# ---------------------------
# CONFIRMAR VENDA
# ---------------------------
@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    data = request.get_json()

    client_id = data.get("client_id")
    lead_id = data.get("lead_id")

    if not client_id or not lead_id:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    conn, cursor = get_db(client_id)

    cursor.execute("""
        UPDATE leads SET virou_cliente = 1 WHERE id = ?
    """, (lead_id,))

    cursor.execute("""
        UPDATE leads SET virou_cliente = 0 WHERE virou_cliente IS NULL
    """)

    conn.commit()

    treinar_modelo(client_id)

    return jsonify({"status": "venda_confirmada"})


# ---------------------------
# DASHBOARD API
# ---------------------------
@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    client_id = request.args.get("client_id")

    if not client_id:
        return jsonify({"erro": "client_id obrigatório"}), 400

    conn, cursor = get_db(client_id)

    df = pd.read_sql("SELECT * FROM leads", conn)

    total = len(df)
    quentes = len(df[df["virou_cliente"] == 1])
    frios = total - quentes

    return jsonify({
        "total_leads": total,
        "leads_quentes": quentes,
        "leads_frios": frios,
        "dados": df.fillna(0).to_dict(orient="records")
    })


# ---------------------------
# START
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
