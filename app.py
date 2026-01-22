from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# ===== TREINANDO A IA =====
dados = {
    'tempo_site': [1,2,5,8,10,15,20,25],
    'paginas_visitadas': [1,1,2,3,4,5,6,7],
    'clicou_preco': [0,0,0,1,1,1,1,1],
    'lead_quente': [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(dados)

X = df[['tempo_site', 'paginas_visitadas', 'clicou_preco']]
y = df['lead_quente']

modelo = DecisionTreeClassifier()
modelo.fit(X, y)

# ===== ROTA DA API =====
@app.route('/prever', methods=['POST'])
def prever():
    data = request.json
    entrada = pd.DataFrame([data])
    resultado = modelo.predict(entrada)

    return jsonify({
        'lead_quente': int(resultado[0])
    })

# ===== START =====
if __name__ == "__main__":
    app.run()