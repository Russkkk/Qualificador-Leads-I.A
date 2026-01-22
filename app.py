import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
ARQUIVO = "leads.csv"

# ===============================
# CARREGAR OU CRIAR DADOS
# ===============================
if os.path.exists(ARQUIVO):
    df = pd.read_csv(ARQUIVO)
else:
    df = pd.DataFrame(columns=[
        'tempo_site',
        'paginas_visitadas',
        'clicou_preco',
        'virou_cliente'
    ])

# ===============================
# TREINAR MODELO
# ===============================
def treinar_modelo():
    if len(df) < 5:
        return None, None

    X = df[['tempo_site', 'paginas_visitadas', 'clicou_preco']]
    y = df['virou_cliente']

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=2,
    random_state=42
)
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)
    acc = accuracy_score(y_teste, previsoes)

    return modelo, round(acc, 2)

modelo, acuracia = treinar_modelo()

# ===============================
# ROTA DE PREVISÃO
# ===============================
@app.route('/prever', methods=['POST'])
def prever():
    global df, modelo, acuracia

    dados = request.json
    entrada = pd.DataFrame([dados])

    if modelo is None:
        return jsonify({"erro": "Poucos dados para prever"}), 400

    prob = modelo.predict_proba(entrada)[0][1]

    decisao = 1 if prob >= 0.8 else 0

    # salvar previsão (feedback vem depois)
    novo = entrada.copy()
    novo['virou_cliente'] = decisao
    df = pd.concat([df, novo], ignore_index=True)
    df.to_csv(ARQUIVO, index=False)

    return jsonify({
        "probabilidade_de_compra": round(prob, 2),
        "lead_quente": decisao,
        "acuracia_atual_modelo": acuracia
    })

# ===============================
# ROTA DE FEEDBACK REAL
# ===============================
@app.route('/feedback', methods=['POST'])
def feedback():
    global df, modelo, acuracia

    dados = request.json
    df.loc[df.index[-1], 'virou_cliente'] = dados['virou_cliente']
    df.to_csv(ARQUIVO, index=False)

    modelo, acuracia = treinar_modelo()

    return jsonify({
        "mensagem": "Feedback recebido",
        "nova_acuracia": acuracia
    })

if __name__ == "__main__":
    app.run()
