
import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, ICMP

PCAP_NORMAL = "dataset_pcap/normal.pcapng"
PCAP_ATAQUE = "dataset_pcap/DoS_traffic.pcapng"

WINDOW_SIZE = 20   # pacotes por janela
STEP_SIZE   = 5    # passo da janela deslizante (overlap)

OUTPUT_NPZ  = "outputs/dataset_janelas.npz"
OUTPUT_CSV  = "outputs/dataset_raw.csv"

# ── Features extraídas por pacote ────────────────────────────
# protocol   : 1=ICMP, 6=TCP, 17=UDP
# pkt_size   : tamanho total do pacote em bytes
# ttl        : time to live (IPs falsos têm TTL variável)
# src_port   : porta de origem
# dst_port   : porta de destino
# tcp_flags  : flags TCP em inteiro (SYN=2, ACK=16, PSH+ACK=24...)
# is_syn     : 1 se for SYN puro (flag SYN flood)
# tcp_payload: tamanho do payload TCP
FEATURE_COLS = [
    "protocol", "pkt_size", "ttl",
    "src_port", "dst_port",
    "tcp_flags", "is_syn", "tcp_payload"
]

#  1. EXTRAÇÃO DE FEATURES DOS PCAPNG
def extrair_pacotes(filepath, label):
    """
    Lê um arquivo pcapng e extrai features de cada pacote IP.
    label = 0 (normal) ou 1 (ataque)
    """
    print(f"\nLendo: {filepath}")
    packets = rdpcap(filepath)
    print(f"  Total de pacotes no arquivo: {len(packets)}")

    rows = []
    ignorados = 0

    for pkt in packets:
        if IP not in pkt:
            ignorados += 1
            continue

        proto      = pkt[IP].proto
        pkt_size   = len(pkt)
        ttl        = pkt[IP].ttl
        src_port   = pkt[TCP].sport       if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dst_port   = pkt[TCP].dport       if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        tcp_flags  = int(pkt[TCP].flags)  if TCP in pkt else 0
        is_syn     = 1 if (TCP in pkt and pkt[TCP].flags == 0x002) else 0
        tcp_payload= len(bytes(pkt[TCP].payload)) if TCP in pkt else 0

        rows.append({
            "timestamp":   float(pkt.time),
            "src_ip":      pkt[IP].src,
            "dst_ip":      pkt[IP].dst,
            "protocol":    proto,
            "pkt_size":    pkt_size,
            "ttl":         ttl,
            "src_port":    src_port,
            "dst_port":    dst_port,
            "tcp_flags":   tcp_flags,
            "is_syn":      is_syn,
            "tcp_payload": tcp_payload,
            "label":       label
        })

    df = pd.DataFrame(rows)
    print(f"  Pacotes IP extraídos : {len(df)}")
    print(f"  Pacotes ignorados    : {ignorados} (não-IP)")
    return df


#  2. JANELAS DESLIZANTES → features estatísticas
def construir_janelas(df, window_size, step_size):
    """
    Cria janelas deslizantes de `window_size` pacotes.

    Para cada janela calcula, por feature:
      média, desvio padrão, máximo, mínimo, amplitude (max-min)

    Features globais da janela:
      total de SYNs, quantidade de flags TCP únicas

    Resultado: X com shape (n_janelas, n_features_estatisticas)
               y com label majoritário da janela
    """
    vals   = df[FEATURE_COLS].values.astype(float)
    labels = df["label"].values
    X, y   = [], []

    n_features = len(FEATURE_COLS)
    total = 0

    for i in range(0, len(vals) - window_size, step_size):
        janela = vals[i : i + window_size]            # (window_size, n_features)

        media   = janela.mean(axis=0)                 # (n_features,)
        desvio  = janela.std(axis=0)
        maximo  = janela.max(axis=0)
        minimo  = janela.min(axis=0)
        amplit  = maximo - minimo

        # Features globais
        n_syn         = janela[:, FEATURE_COLS.index("is_syn")].sum()
        flags_unicas  = len(np.unique(janela[:, FEATURE_COLS.index("tcp_flags")]))
        taxa_tcp      = (janela[:, FEATURE_COLS.index("protocol")] == 6).mean()
        taxa_udp      = (janela[:, FEATURE_COLS.index("protocol")] == 17).mean()

        row = np.concatenate([
            media, desvio, maximo, minimo, amplit,
            [n_syn, flags_unicas, taxa_tcp, taxa_udp]
        ])
        X.append(row)

        # Label da janela = majoritário
        labels_janela = labels[i : i + window_size]
        y.append(1 if labels_janela.mean() >= 0.5 else 0)
        total += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n  Janelas geradas        : {X.shape[0]}")
    print(f"  Features por janela    : {X.shape[1]}")
    print(f"  Label 0 (normal)       : {(y==0).sum()}")
    print(f"  Label 1 (ataque)       : {(y==1).sum()}")

    return X, y


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  PASSO 1 — Extração de features")
    print("=" * 55)

    # Extrai pacotes
    df_normal = extrair_pacotes(PCAP_NORMAL, label=0)
    df_ataque = extrair_pacotes(PCAP_ATAQUE, label=1)

    # Concatena e ordena por tempo
    df = pd.concat([df_normal, df_ataque]) \
           .sort_values("timestamp") \
           .reset_index(drop=True)

    print(f"\nDataset bruto combinado:")
    print(f"  Total de pacotes : {len(df)}")
    print(f"  Normal  (0)      : {(df.label==0).sum()}")
    print(f"  Ataque  (1)      : {(df.label==1).sum()}")

    # Salva CSV bruto
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV bruto salvo em: {OUTPUT_CSV}")

    # Constrói janelas
    print("\n" + "=" * 55)
    print("  Construindo janelas deslizantes")
    print("=" * 55)
    X, y = construir_janelas(df, WINDOW_SIZE, STEP_SIZE)

    # Salva arrays
    np.savez_compressed(OUTPUT_NPZ, X=X, y=y)
    print(f"\nArrays salvos em: {OUTPUT_NPZ}")
    print("\n[OK] Passo 1 concluído. Execute 02_preprocessamento.py")
