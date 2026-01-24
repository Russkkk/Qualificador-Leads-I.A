import os
import sqlite3
import joblib
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# ===============================
# CONFIGURAÇÕES
# ===============================

app = Flask(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WHATSAPP_NUMERO_PADRAO = "5511999999999"

# ===============================
# FUNÇÕES AUXILIARES
# ===============================

def get_db_path(client_id):
    return os.path.join(DATA_DIR, f"{client_id}.db")

def get_model_path(client_id):
    return os.path.join(DATA_DIR, f"{client_id}_model.pkl")

def gerar_link_whatsapp(numero, mensagem):
    texto = mensagem.replace(" ", "%20").replace("\n", "%0A")
    return f"https://wa.me/{numero}?text={texto}"

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

def carregar_modelo(client_id):
    path = get_model_path(client_id)
    if os.path.exists(path):
        return joblib.load(path)
    return None

def salvar_modelo(client_id, modelo):
    joblib.dump(modelo, get_model_path(client_id))

# ===============================
# ROTAS
# ===============================

@app.route("/")
def home():
    return {"status": "API IA ativa \U0001f680"}

@app.route("/prever", methods=["POST"])
def prever():
    dados = request.json

    client_id = dados.get("client_id")
    if not client_id:
        return jsonify({"erro": "client_id é obrigatório"}), 400

    for campo in ["tempo_site", "paginas_visitadas", "clicou_preco"]:
        if campo not in dados:
            return jsonify({"erro": f"Campo ausente: {campo}"}), 400

    # -------- BANCO DO CLIENTE --------
    db_path = get_db_path(client_id)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tabela de leads
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tempo_site INTEGER,
            paginas_visitadas INTEGER,
            clicou_preco INTEGER,
            virou_cliente INTEGER,
            data TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabela de configuração
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS config (
            id INTEGER PRIMARY KEY,
            whatsapp TEXT,
            threshold REAL,
            mensagem TEXT
        )
    """)

    conn.commit()

    # Config padrão (se não existir)
    cursor.execute("SELECT COUNT(*) FROM config")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO config (id, whatsapp, threshold, mensagem)
            VALUES (1, ?, ?, ?)
        """, (
            WHATSAPP_NUMERO_PADRAO,
            0.8,
            "Olá! Vi seu interesse e posso te ajudar agora 😊"
        ))
        conn.commit()

    # Lê config
    cursor.execute("SELECT whatsapp, threshold, mensagem FROM config WHERE id = 1")
    whatsapp, threshold, mensagem = cursor.fetchone()

    # -------- MODELO --------
    modelo = carregar_modelo(client_id)

    if modelo is None:
        df = carregar_dados(client_id)
        modelo = treinar_modelo(df)
        if modelo:
            salvar_modelo(client_id, modelo)

    if modelo is None:
        prob = 0.35
    else:
        entrada = pd.DataFrame([{
            "tempo_site": dados["tempo_site"],
            "paginas_visitadas": dados["paginas_visitadas"],
            "clicou_preco": dados["clicou_preco"]
        }])
        prob = min(modelo.predict_proba(entrada)[0][1], 0.95)

    decisao = 1 if prob >= threshold else 0

    # -------- SALVAR LEAD --------
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

    # -------- RE-TREINO --------
    df = carregar_dados(client_id)
    modelo = treinar_modelo(df)
    if modelo:
        salvar_modelo(client_id, modelo)

    resposta = {
        "client_id": client_id,
        "probabilidade_de_compra": round(prob, 2),
        "lead_quente": decisao
    }

    if decisao == 1:
        resposta["whatsapp"] = gerar_link_whatsapp(whatsapp, mensagem)

    return jsonify(resposta)

@app.route("/dashboard", methods=["GET"])
def dashboard():
    client_id = request.args.get("client_id")
    dias = int(request.args.get("dias", 30))

    if not client_id:
        return jsonify({"erro": "client_id é obrigatório"}), 400

    db_path = get_db_path(client_id)

    if not os.path.exists(db_path):
        return jsonify({"erro": "Cliente não encontrado"}), 404

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # total de leads
    cursor.execute("""
    SELECT COUNT(*) FROM leads
    WHERE data >= datetime('now', ?)
""", (f"-{dias} days",))
    total = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(*) FROM leads
    WHERE virou_cliente = 1
    AND data >= datetime('now', ?)
""", (f"-{dias} days",))
    quentes = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(*) FROM leads
    WHERE virou_cliente = 0
    AND data >= datetime('now', ?)
""", (f"-{dias} days",))
    frios = cursor.fetchone()[0]

    taxa = round((quentes / total) * 100, 2) if total > 0 else 0

    return jsonify({
        "client_id": client_id,
        "total_leads": total,
        "leads_quentes": quentes,
        "leads_frios": frios,
        "taxa_conversao_percent": taxa
    })

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    app.run()
