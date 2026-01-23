import os
import sqlite3
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# ===============================
# CONFIGURAÇÕES GERAIS
# ===============================

app = Flask(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WHATSAPP_NUMERO_PADRAO = "5511999999999"  # troque depois por cliente

# ===============================
# FUNÇÕES AUXILIARES
# ===============================

def get_model_path(client_id: str):
    return os.path.join(DATA_DIR, f"{client_id}_model.pkl")


def carregar_modelo(client_id):
    path = get_model_path(client_id)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def salvar_modelo(client_id, modelo):
    joblib.dump(modelo, get_model_path(client_id))
    

def get_db_path(client_id: str):
    return os.path.join(DATA_DIR, f"{client_id}.db")


def gerar_link_whatsapp():
    mensagem = (
        "Olá! Vi seu interesse e posso te ajudar agora 😊\n\n"
        "Atendimento rápido e sem compromisso."
    )
    texto = mensagem.replace(" ", "%20").replace("\n", "%0A")
    return f"https://wa.me/{WHATSAPP_NUMERO_PADRAO}?text={texto}"


def carregar_dados(client_id):
    db_path = get_db_path(client_id)
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("SELECT * FROM leads", conn)
    conn.close()

    return df


def treinar_modelo(df):
    if len(df) < 10:
        return None

    X = df[["tempo_site", "paginas_visitadas", "clicou_preco"]]
    y = df["virou_cliente"]

    modelo = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    )

    modelo.fit(X, y)
    return modelo

# ===============================
# ROTA PRINCIPAL
# ===============================

@app.route("/prever", methods=["POST"])
def prever():
    dados = request.json

    # -------- validações básicas --------
    client_id = dados.get("client_id")
    if not client_id:
        return jsonify({"erro": "client_id é obrigatório"}), 400

    for campo in ["tempo_site", "paginas_visitadas", "clicou_preco"]:
        if campo not in dados:
            return jsonify({"erro": f"Campo ausente: {campo}"}), 400

    # -------- banco do cliente --------
     df = carregar_dados(client_id)
    modelo = treinar_modelo(df)

    if modelo:
        salvar_modelo(client_id, modelo)
    
    db_path = get_db_path(client_id)
    conn = sqlite3.connect(db_path)
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

    # -------- carrega dados e treina --------
    df = carregar_dados(client_id)
    
    # -------- MODELO DO CLIENTE --------
    modelo = carregar_modelo(client_id)

    if modelo is None:
        df = carregar_dados(client_id)
        modelo = treinar_modelo(df)

        if modelo:
            salvar_modelo(client_id, modelo)

    # -------- PREDIÇÃO --------
    if modelo is None:
        prob = 0.35
    else:
        entrada = pd.DataFrame([{
            "tempo_site": dados["tempo_site"],
            "paginas_visitadas": dados["paginas_visitadas"],
            "clicou_preco": dados["clicou_preco"]
        }])
        prob = min(modelo.predict_proba(entrada)[0][1], 0.95)
        
    # -------- salva lead --------
    cursor.execute("""
        INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
        VALUES (?, ?, ?, ?)
    """, (
        dados["tempo_site"],
        dados["paginas_visitadas"],
        dados["clicou_preco"],
        decisao
    ))

    conn.commit()
    conn.close()

    # -------- resposta --------
    resposta = {
        "client_id": client_id,
        "probabilidade_de_compra": round(prob, 2),
        "lead_quente": decisao
    }

    if decisao == 1:
        resposta["whatsapp"] = gerar_link_whatsapp()

    return jsonify(resposta)


# ===============================
# HEALTH CHECK
# ===============================

@app.route("/")
def home():
    return {"status": "API IA ativa 🚀"}


if __name__ == "__main__":
    app.run()

