# ================================================================
#  PASSO 2 — Normalização + SMOTE + Divisão CORRETA
#
#  CORREÇÃO: Divisão ANTES do SMOTE e do Scaler
#  para evitar Data Leakage
#
#  Entrada : outputs/dataset_janelas.npz
#  Saída   : outputs/dados_treino.npz
#            outputs/scaler.pkl
#            graficos/distribuicao_classes.png
# ================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling  import SMOTE

INPUT_NPZ     = "outputs/dataset_janelas.npz"
OUTPUT_NPZ    = "outputs/dados_treino.npz"
OUTPUT_SCALER = "outputs/scaler.pkl"
OUTPUT_GRAF   = "graficos/distribuicao_classes.png"

# ════════════════════════════════════════════════════════════
#  1. CARREGAR
# ════════════════════════════════════════════════════════════
def carregar(path):
    data = np.load(path)
    X, y = data["X"], data["y"]
    print(f"Dados carregados : X={X.shape}  y={y.shape}")
    print(f"  Normal  (0) : {(y==0).sum()}")
    print(f"  Ataque  (1) : {(y==1).sum()}")
    return X, y

# ════════════════════════════════════════════════════════════
#  2. DIVISÃO PRIMEIRO — 65 / 15 / 20
#     ⚠️  SMOTE e Scaler só depois, apenas no treino
# ════════════════════════════════════════════════════════════
def dividir(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.35, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.571, stratify=y_tmp, random_state=42)

    print(f"\nDivisão (sem SMOTE/Scaler ainda):")
    print(f"  Treino     : {X_tr.shape[0]:>7}  |  N={( y_tr==0).sum()}  A={(y_tr==1).sum()}")
    print(f"  Validação  : {X_val.shape[0]:>7}  |  N={(y_val==0).sum()}  A={(y_val==1).sum()}")
    print(f"  Teste      : {X_te.shape[0]:>7}  |  N={( y_te==0).sum()}  A={(y_te==1).sum()}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te

# ════════════════════════════════════════════════════════════
#  3. SCALER — fit APENAS no treino, transform nos demais
# ════════════════════════════════════════════════════════════
def normalizar(X_tr, X_val, X_te):
    scaler = MinMaxScaler()
    X_tr  = scaler.fit_transform(X_tr)   # fit+transform no treino
    X_val = scaler.transform(X_val)       # só transform
    X_te  = scaler.transform(X_te)        # só transform
    print(f"\nScaler fit no treino. Range treino: [{X_tr.min():.3f}, {X_tr.max():.3f}]")
    return X_tr, X_val, X_te, scaler

# ════════════════════════════════════════════════════════════
#  4. SMOTE — apenas no conjunto de treino
# ════════════════════════════════════════════════════════════
def aplicar_smote(X_tr, y_tr):
    print(f"\nSMOTE apenas no treino:")
    print(f"  Antes : N={(y_tr==0).sum()}  A={(y_tr==1).sum()}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_tr_bal, y_tr_bal = sm.fit_resample(X_tr, y_tr)
    print(f"  Depois: N={(y_tr_bal==0).sum()}  A={(y_tr_bal==1).sum()}")
    return X_tr_bal, y_tr_bal

# ════════════════════════════════════════════════════════════
#  5. RESHAPE para Conv2D → (samples, H, W, 1)
# ════════════════════════════════════════════════════════════
def reshape_conv2d(X):
    n_samples, n_feat = X.shape
    lado = int(np.ceil(np.sqrt(n_feat)))
    pad  = lado * lado - n_feat
    if pad > 0:
        X = np.pad(X, ((0,0),(0,pad)), constant_values=0)
    return X.reshape(n_samples, lado, lado, 1).astype(np.float32), lado

# ════════════════════════════════════════════════════════════
#  6. GRÁFICO distribuição
# ════════════════════════════════════════════════════════════
def grafico_distribuicao(y_tr_orig, y_tr_bal, y_val, y_te):
    os.makedirs("graficos", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distribuição das Classes por Split",
                 fontsize=14, fontweight="bold")

    conjuntos = [
        (y_tr_orig, "Treino — antes SMOTE"),
        (y_tr_bal,  "Treino — após SMOTE"),
        (np.concatenate([y_val, y_te]), "Val + Teste (sem SMOTE)"),
    ]
    cores = [["#2196F3","#E53935"],["#43A047","#E53935"],["#FF9800","#E53935"]]

    for ax, (y, titulo), cor in zip(axes, conjuntos, cores):
        counts = [(y==0).sum(), (y==1).sum()]
        bars = ax.bar(["Normal (0)","Ataque (1)"], counts,
                      color=cor, edgecolor="white", width=0.5)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_ylabel("Amostras")
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(counts)*0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(counts)*1.15)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_GRAF, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico salvo: {OUTPUT_GRAF}")

# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*55)
    print("  PASSO 2 — Pré-processamento (SEM data leakage)")
    print("="*55)

    X, y = carregar(INPUT_NPZ)

    # 1. Divide PRIMEIRO
    X_tr, X_val, X_te, y_tr, y_val, y_te = dividir(X, y)
    y_tr_orig = y_tr.copy()

    # 2. Scaler fit só no treino
    X_tr, X_val, X_te, scaler = normalizar(X_tr, X_val, X_te)

    # 3. SMOTE só no treino
    X_tr, y_tr = aplicar_smote(X_tr, y_tr)

    # 4. Gráfico
    grafico_distribuicao(y_tr_orig, y_tr, y_val, y_te)

    # 5. Reshape Conv2D
    print("\nReshape para Conv2D:")
    X_tr,  lado = reshape_conv2d(X_tr)
    X_val, _    = reshape_conv2d(X_val)
    X_te,  _    = reshape_conv2d(X_te)
    print(f"  Shape : ({lado}×{lado}×1)")

    # 6. Salva
    np.savez_compressed(OUTPUT_NPZ,
        X_tr=X_tr, X_val=X_val, X_te=X_te,
        y_tr=y_tr, y_val=y_val, y_te=y_te,
        lado=np.array([lado])
    )
    joblib.dump(scaler, OUTPUT_SCALER)

    print(f"\nDados salvos : {OUTPUT_NPZ}")
    print(f"Scaler salvo : {OUTPUT_SCALER}")
    print("\n[OK] Passo 2 concluído. Execute 03_treino.py")
