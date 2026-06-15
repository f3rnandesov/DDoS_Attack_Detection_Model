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
LADO = 7 

@app.before_request
def check_api_key():
    if request.path == '/detectar':
        key = request.headers.get("X-API-KEY")
        if not API_KEY:
            return jsonify({"erro": "Configuração de servidor inválida"}), 500
        if key != API_KEY:
            return jsonify({"erro": "Acesso negado: Chave inválida"}), 403

@app.route('/detectar', methods=['POST'])
def detectar():
    try:
        data = request.json.get('features')
        if not data:
            return jsonify({"erro": "Campo 'features' ausente"}), 400
        
        vetor = np.array(data).reshape(1, -1)
        v_scaled = scaler.transform(vetor)
        v_padded = np.pad(v_scaled, ((0, 0), (0, 5)), mode='constant', constant_values=0)
        X_img = v_padded.reshape(1, LADO, LADO, 1).astype(np.float32)
        
        # --- CORREÇÃO DA MEMÓRIA AQUI ---
        # Chamamos o modelo diretamente. Consome uma fração da RAM do .predict()
        predicao = modelo(X_img, training=False)
        prob = float(predicao.numpy()[0][0])
        
        label = "Ataque" if prob >= 0.70 else "Normal"
        return jsonify({"probabilidade": prob, "classificacao": label})

    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)