import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# Lê a chave configurada nas variáveis de ambiente do Render
API_KEY = os.getenv("API_KEY")

# Carrega modelo e scaler
modelo = tf.keras.models.load_model("models/modelo_ddos.keras")
scaler = joblib.load("outputs/scaler.pkl")
lado = 8 

# Decorator para checar a chave de segurança
@app.before_request
def check_api_key():
    # Verifica apenas na rota de detecção
    if request.path == '/detectar':
        key = request.headers.get("X-API-KEY")
        if key != API_KEY:
            return jsonify({"erro": "Acesso negado: Chave inválida ou ausente"}), 403

@app.route('/detectar', methods=['POST'])
def detectar():
    data = request.json['features'] 
    vetor = np.array(data).reshape(1, -1)
    
    # Normaliza e formata para a CNN
    v_scaled = scaler.transform(vetor)
    X_img = v_scaled.reshape(1, lado, lado, 1).astype(np.float32)
    
    prob = float(modelo.predict(X_img, verbose=0)[0][0])
    label = "Ataque" if prob >= 0.70 else "Normal"
    
    return jsonify({"probabilidade": prob, "classificacao": label})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)