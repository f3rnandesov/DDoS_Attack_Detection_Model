import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# Lê a chave configurada nas variáveis de ambiente do Render
API_KEY = os.getenv("API_KEY")

# Carrega modelo e scaler (Caminhos relativos à raiz do projeto no Render)
modelo = tf.keras.models.load_model("models/modelo_ddos.keras")
scaler = joblib.load("outputs/scaler.pkl")

# Definimos o lado da matriz da CNN (7x7 = 49)
LADO = 7 

# Decorator para checar a chave de segurança
@app.before_request
def check_api_key():
    if request.path == '/detectar':
        key = request.headers.get("X-API-KEY")
        if not API_KEY: # Segurança: se a chave no servidor não estiver definida, bloqueia
            return jsonify({"erro": "Configuração de servidor inválida"}), 500
        if key != API_KEY:
            return jsonify({"erro": "Acesso negado: Chave inválida"}), 403

@app.route('/detectar', methods=['POST'])
def detectar():
    try:
        # Recebe os dados
        data = request.json.get('features')
        if not data:
            return jsonify({"erro": "Campo 'features' ausente"}), 400
        
        vetor = np.array(data).reshape(1, -1)
        
        # 1. Normaliza (o scaler espera 44 colunas, que é o que vem do cliente)
        v_scaled = scaler.transform(vetor)
        
        # 2. Adiciona o padding de 5 zeros para completar as 49 características (7x7)
        # O np.pad adiciona 5 colunas de valor 0 ao final do vetor
        v_padded = np.pad(v_scaled, ((0, 0), (0, 5)), mode='constant', constant_values=0)
        
        # 3. Reshape para o formato da CNN (1, 7, 7, 1)
        X_img = v_padded.reshape(1, LADO, LADO, 1).astype(np.float32)
        
        # Predição
        prob = float(modelo.predict(X_img, verbose=0)[0][0])
        label = "Ataque" if prob >= 0.70 else "Normal"
        
        return jsonify({"probabilidade": prob, "classificacao": label})

    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)