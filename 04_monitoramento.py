
import os, sys, time, warnings, collections, threading
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf
import requests
from scapy.all import sniff, IP, TCP, UDP

# ── Configurações ────────────────────────────────────────────
INTERFACE    = "wlp0s20f3"          # interface do hotspot no notebook
MODEL_PATH   = "models/modelo_ddos.keras"
SCALER_PATH  = "outputs/scaler.pkl"

# IPs monitorados
IPS_ALVO = {
    "10.42.0.100": "Master",
    "10.42.0.101": "Worker1-LDR",
    "10.42.0.102": "Worker2-DHT11",
}

WINDOW_SIZE  = 20        # mesma janela do treino
STEP_SIZE    = 5         # inferência a cada 5 pacotes novos
THRESHOLD    = 0.70      # probabilidade acima = alerta de ataque

FEATURE_COLS = [
    "protocol", "pkt_size", "ttl",
    "src_port", "dst_port",
    "tcp_flags", "is_syn", "tcp_payload"
]


# ════════════════════════════════════════════════════════════
#  BUFFER GLOBAL de pacotes capturados
# ════════════════════════════════════════════════════════════
buffer_lock   = threading.Lock()
packet_buffer = collections.deque(maxlen=500)   # últimos 500 pacotes
stats = {
    "total":   0,
    "alertas": 0,
    "normal":  0,
}


# ════════════════════════════════════════════════════════════
#  EXTRAÇÃO DE FEATURES de um pacote
# ════════════════════════════════════════════════════════════
def extrair_features_pkt(pkt):
    if IP not in pkt:
        return None
    return [
        pkt[IP].proto,
        len(pkt),
        pkt[IP].ttl,
        pkt[TCP].sport       if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0),
        pkt[TCP].dport       if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
        int(pkt[TCP].flags)  if TCP in pkt else 0,
        1 if (TCP in pkt and pkt[TCP].flags == 0x002) else 0,
        len(bytes(pkt[TCP].payload)) if TCP in pkt else 0,
    ]


# ════════════════════════════════════════════════════════════
#  MONTAR JANELA → vetor de features estatísticas
# ════════════════════════════════════════════════════════════
def montar_janela(janela):
    arr    = np.array(janela, dtype=np.float32)   # (WINDOW_SIZE, n_features)
    media  = arr.mean(axis=0)
    desvio = arr.std(axis=0)
    maximo = arr.max(axis=0)
    minimo = arr.min(axis=0)
    amplit = maximo - minimo

    n_syn        = arr[:, FEATURE_COLS.index("is_syn")].sum()
    flags_unicas = len(np.unique(arr[:, FEATURE_COLS.index("tcp_flags")]))
    taxa_tcp     = (arr[:, FEATURE_COLS.index("protocol")] == 6).mean()
    taxa_udp     = (arr[:, FEATURE_COLS.index("protocol")] == 17).mean()

    vetor = np.concatenate([
        media, desvio, maximo, minimo, amplit,
        [n_syn, flags_unicas, taxa_tcp, taxa_udp]
    ])
    return vetor


# ════════════════════════════════════════════════════════════
#  INFERÊNCIA COM A API DO RENDER
# ════════════════════════════════════════════════════════════
URL_API = "https://api-detection-ddos-iot.onrender.com/detectar"
API_KEY = "minha_chave_secreta_2026"

def inferir(vetor, lado):
    headers = {"X-API-KEY": API_KEY}
    payload = {"features": vetor.tolist()}
    
    try:
        response = requests.post(URL_API, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            resultado = response.json()
            return resultado['probabilidade'], resultado['classificacao']
        else:
            print(f"Erro na API ({response.status_code}): {response.text}")
            return 0, "Erro"
            
    except Exception as e:
        print(f"Falha ao conectar com a nuvem: {e}")
        return 0, "Erro"


# ════════════════════════════════════════════════════════════
#  CALLBACK do Scapy — chamado a cada pacote capturado
# ════════════════════════════════════════════════════════════
def callback_pacote(pkt):
    features = extrair_features_pkt(pkt)
    if features is None:
        return

    # Filtra apenas pacotes dos IPs de interesse
    src = pkt[IP].src
    dst = pkt[IP].dst
    if src not in IPS_ALVO and dst not in IPS_ALVO:
        return

    with buffer_lock:
        packet_buffer.append(features)
        stats["total"] += 1


# ════════════════════════════════════════════════════════════
#  THREAD DE INFERÊNCIA — roda em paralelo ao sniff
# ════════════════════════════════════════════════════════════
def thread_inferencia(modelo, scaler, lado, stop_event):
    janela_local = []

    print(f"\n{'='*60}")
    print(f"  Monitoramento ativo — threshold={THRESHOLD}")
    print(f"  IPs monitorados: {list(IPS_ALVO.values())}")
    print(f"{'='*60}\n")

    while not stop_event.is_set():
        time.sleep(0.05)

        with buffer_lock:
            novos = list(packet_buffer)
            packet_buffer.clear()

        janela_local.extend(novos)

        while len(janela_local) >= WINDOW_SIZE:
            janela = janela_local[:WINDOW_SIZE]
            janela_local = janela_local[STEP_SIZE:]

            vetor = montar_janela(janela)
            prob, label = inferir(vetor, lado)

            ts = time.strftime("%H:%M:%S")

            # --- CORREÇÃO AQUI: Compara com a String "Ataque" retornada pela API ---
            if label == "Ataque":
                stats["alertas"] += 1
                print(f"[{ts}] ATAQUE DETECTADO  "
                      f"prob={prob:.3f}  "
                      f"alertas_até_agora={stats['alertas']}")
            else:
                stats["normal"] += 1
                print(f"[{ts}] Normal            "
                      f"prob={prob:.3f}  "
                      f"alertas_até_agora={stats['alertas']}")

        # Relatório a cada 200 pacotes totais analisados
        if stats["total"] > 0 and stats["total"] % 200 == 0:
            total = stats["alertas"] + stats["normal"]
            if total > 0:
                taxa = stats["alertas"] / total * 100
                print(f"\n{'─'*60}")
                print(f"  Relatório parcial:")
                print(f"  Janelas classificadas : {total}")
                print(f"  Normal                : {stats['normal']}")
                print(f"  Ataque                : {stats['alertas']}  ({taxa:.1f}%)")
                print(f"{'─'*60}\n")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if os.geteuid() != 0:
        print("[ERRO] Execute com sudo: sudo python3 04_monitoramento.py")
        sys.exit(1)

    print("Carregando configurações locais...")
    scaler = joblib.load(SCALER_PATH)

    # Descobre lado do quadrado para fins informativos
    dados = np.load("outputs/dados_treino.npz")
    lado  = int(dados["lado"][0])
    print(f"Scaler carregado    : {SCALER_PATH}")
    print(f"Input shape Conv2D  : {lado}×{lado}×1")
    print(f"Interface de rede   : {INTERFACE}")

    # Inicia thread de inferência
    stop_event = threading.Event()
    t = threading.Thread(
        target=thread_inferencia,
        args=(None, scaler, lado, stop_event), # O modelo roda remoto, passamos None para o lugar do modelo local
        daemon=True
    )
    t.start()

    # Captura ao vivo (bloqueia até Ctrl+C)
    filtro_bpf = " or ".join([f"host {ip}" for ip in IPS_ALVO])
    print(f"\nCapturando na interface {INTERFACE}...")
    print(f"Filtro BPF: {filtro_bpf}")
    print("Pressione Ctrl+C para parar.\n")

    try:
        sniff(
            iface=INTERFACE,
            filter=filtro_bpf,
            prn=callback_pacote,
            store=False,
        )
    except KeyboardInterrupt:
        stop_event.set()
        t.join(timeout=2)

        # Relatório final
        total = stats["alertas"] + stats["normal"]
        print(f"\n{'='*60}")
        print("  RELATÓRIO FINAL")
        print(f"{'='*60}")
        print(f"  Pacotes capturados    : {stats['total']}")
        print(f"  Janelas classificadas : {total}")
        print(f"  Normal                : {stats['normal']}")
        print(f"  Ataque detectado      : {stats['alertas']}")
        if total > 0:
            print(f"  Taxa de detecção      : {stats['alertas']/total*100:.1f}%")
        print(f"{'='*60}")