import os
import sqlite3
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression

# ======================
# APP CONFIG
# ======================

app = Flask(__name__)
CORS(app)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ======================
# DATABASE
# ======================

def db_path(client_id):
    return f"data/{client_id}.db"


def conectar_db(client_id):
    return sqlite3.connect(db_path(client_id))


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

# ======================
# MODELO
# ======================

def treinar_modelo(client_id):
    conn = conectar_db(client_id)
    cursor = conn.cursor()

    # 🔥 FORÇA LEADS NÃO CONFIRMADOS A VIRAREM 0
    cursor.execute("""
        UPDATE leads
        SET virou_cliente = 0
        WHERE virou_cliente IS NULL
    """)
    conn.commit()

    df = pd.read_sql("""
        SELECT tempo_site, paginas_visitadas, clicou_preco, virou_cliente
        FROM leads
        WHERE virou_cliente IN (0, 1)
    """, conn)

    conn.close()

    # precisa de pelo menos 2 classes
    if df["virou_cliente"].nunique() < 2:
        return None

    X = df[["tempo_site", "paginas_visitadas", "clicou_preco"]]
    y = df["virou_cliente"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, f"models/{client_id}.joblib")
    return model


def carregar_modelo(client_id):
    caminho = f"models/{client_id}.joblib"
    if os.path.exists(caminho):
        return joblib.load(caminho)
    return None

# ======================
# ROTAS
# ======================

@app.route("/prever", methods=["POST"])
def prever():
    try:
        data = request.get_json(silent=True) or {}

        client_id = data.get("client_id")
        tempo_site = data.get("tempo_site")
        paginas = data.get("paginas_visitadas")
        clicou = data.get("clicou_preco")

        if not client_id:
            return jsonify({"erro": "client_id obrigatório"}), 400

        criar_tabela(client_id)

        conn = conectar_db(client_id)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
            VALUES (?, ?, ?, NULL)
        """, (tempo_site, paginas, clicou))

        conn.commit()
        lead_id = cursor.lastrowid
        conn.close()

        model = carregar_modelo(client_id)

        if not model:
            prob = 0.35
        else:
            X = pd.DataFrame([{
                "tempo_site": tempo_site,
                "paginas_visitadas": paginas,
                "clicou_preco": clicou
            }])
            prob = float(model.predict_proba(X)[0][1])

        return jsonify({
            "lead_id": lead_id,
            "probabilidade_de_compra": round(prob, 2),
            "lead_quente": int(prob >= 0.6)
        })

    except Exception as e:
        # 🔥 ISSO MOSTRA O ERRO REAL NO RENDER
        return jsonify({
            "erro": "erro interno no /prever",
            "detalhe": str(e)
        }), 500


@app.route("/confirmar_venda", methods=["POST"])
def confirmar_venda():
    data = request.get_json(silent=True) or {}

    client_id = data.get("client_id")
    lead_id = data.get("lead_id")

    if not client_id or not lead_id:
        return jsonify({"erro": "client_id e lead_id obrigatórios"}), 400

    criar_tabela(client_id)

    conn = conectar_db(client_id)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE leads
        SET virou_cliente = 1
        WHERE id = ?
    """, (lead_id,))

    conn.commit()
    conn.close()

    treinar_modelo(client_id)

    return jsonify({"status": "venda_confirmada"})

# ======================
# START
# ======================

if __name__ == "__main__":
    app.run(debug=True)
