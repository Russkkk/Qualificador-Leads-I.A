import os
import time
import logging
import sqlite3
import threading
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from dotenv import load_dotenv
from urllib.parse import quote
from pydantic import BaseModel, ValidationError

load_dotenv()

app = Flask(__name__)

# ================= CONFIGURAÇÕES =================
DB_FILE = os.getenv('DB_FILE', 'leads.db')
MODEL_DIR = os.getenv('MODEL_DIR', 'modelos')
WHATSAPP_NUMERO = os.getenv('WHATSAPP_NUMERO', '5518981621797')
MIN_DADOS_TREINO = int(os.getenv('MIN_DADOS_TREINO', 15))
TREINO_APOS_N_FEEDBACKS = int(os.getenv('TREINO_APOS_N_FEEDBACKS', 5))

os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================= DATABASE =================

def conectar_db():
    try:
        return sqlite3.connect(DB_FILE, check_same_thread=False)
    except sqlite3.Error as e:
        logging.error(f"Erro ao conectar DB: {e}")
        raise

def init_db():
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tempo_site REAL,
            paginas_visitadas INTEGER,
            clicou_preco INTEGER,
            virou_cliente INTEGER
        )
    """)
    conn.commit()
    conn.close()

def carregar_dados():
    conn = conectar_db()
    df = pd.read_sql_query("SELECT * FROM leads", conn)
    conn.close()
    return df

def salvar_novo_lead(dados, virou_cliente=-1):
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente) VALUES (?, ?, ?, ?)",
        (dados['tempo_site'], dados['paginas_visitadas'], dados['clicou_preco'], virou_cliente)
    )
    lead_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return lead_id

def atualizar_feedback(lead_id, virou_cliente):
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE leads SET virou_cliente = ? WHERE id = ?", (virou_cliente, lead_id))
    conn.commit()
    conn.close()

# ================= ML =================

def criar_modelo():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42
        ))
    ])

def treinar_modelo():
    global modelo, acuracia, contador_feedbacks

    df = carregar_dados()
    df = df[df['virou_cliente'] != -1]

    if len(df) < MIN_DADOS_TREINO:
        logging.info("Dados insuficientes para treino.")
        return

    X = df[['tempo_site', 'paginas_visitadas', 'clicou_preco']]
    y = df['virou_cliente']

    if len(y.unique()) < 2:
        logging.warning("Apenas uma classe disponível.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    modelo = criar_modelo()
    modelo.fit(X_train, y_train)

    # Cross-validation (cv=3 para evitar erros com datasets pequenos)
    cv_scores = cross_val_score(modelo, X, y, cv=3, scoring='roc_auc')
    logging.info(f"AUC CV: {cv_scores.mean():.2f}")

    preds = modelo.predict(X_test)
    probas = modelo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)

    version = int(time.time())
    model_path = os.path.join(MODEL_DIR, f"modelo_{version}.joblib")
    dump(modelo, model_path)

    # Limpeza de modelos antigos (manter apenas os 5 mais recentes)
    arquivos = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("modelo_")], 
                      key=lambda f: int(f.split('_')[1].split('.')[0]), reverse=True)
    for f in arquivos[5:]:
        os.remove(os.path.join(MODEL_DIR, f))

    acuracia = round(acc, 2)
    logging.info(f"Modelo treinado | ACC: {acc:.2f} | AUC: {auc:.2f}")

    contador_feedbacks = 0

def carregar_modelo_mais_recente():
    arquivos = [f for f in os.listdir(MODEL_DIR) if f.startswith("modelo_")]
    if arquivos:
        mais_recente = max(arquivos, key=lambda f: int(f.split('_')[1].split('.')[0]))
        return load(os.path.join(MODEL_DIR, mais_recente))
    return None

# ================= WHATSAPP =================

def gerar_link_whatsapp():
    msg = "Olá! Vi seu interesse e posso te ajudar agora 😊"
    texto = quote(msg)
    return f"https://wa.me/{WHATSAPP_NUMERO}?text={texto}"

# ================= INIT =================
init_db()
contador_feedbacks = 0
modelo = carregar_modelo_mais_recente()
acuracia = None  # Pode ser carregada se necessário

# ================= ROTAS =================

class LeadData(BaseModel):
    tempo_site: float
    paginas_visitadas: int
    clicou_preco: int

class FeedbackData(BaseModel):
    lead_id: int
    virou_cliente: int

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'erro': 'Erro interno do servidor'}), 500

@app.route('/prever', methods=['POST'])
def prever():
    global modelo
    try:
        dados = LeadData(**request.json).model_dump()
    except ValidationError as e:
        return jsonify({'erro': str(e)}), 400

    entrada = pd.DataFrame([dados])
    lead_id = salvar_novo_lead(dados)

    if not modelo:
        return jsonify({'lead_id': lead_id, 'mensagem': 'Modelo ainda não treinado'}), 200

    prob = modelo.predict_proba(entrada)[0][1]
    score = int(prob * 100)

    resposta = {
        'lead_id': lead_id,
        'score': score,
        'lead_quente': score >= 80
    }

    if score >= 80:
        resposta['whatsapp'] = gerar_link_whatsapp()

    return jsonify(resposta)

@app.route('/feedback', methods=['POST'])
def feedback():
    global contador_feedbacks
    try:
        dados = FeedbackData(**request.json).model_dump()
    except ValidationError as e:
        return jupytext({'erro': str(e)}), 400

    atualizar_feedback(dados['lead_id'], dados['virou_cliente'])
    contador_feedbacks += 1

    if contador_feedbacks >= TREINO_APOS_N_FEEDBACKS:
        threading.Thread(target=treinar_modelo).start()

    return jsonify({'mensagem': 'Feedback registrado'})

@app.route('/metrics', methods=['GET'])
def metrics():
    df = carregar_dados()
    num_leads = len(df)
    num_treinados = len(df[df['virou_cliente'] != -1])
    return jsonify({
        'num_leads': num_leads,
        'num_treinados': num_treinados,
        'acuracia': acuracia if acuracia else 'Não disponível'
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)