# ================================================================
#  PASSO 3 — Modelo Conv2D + Treino + Avaliação gráfica
#
#  Entrada : outputs/dados_treino.npz
#  Saída   : models/modelo_ddos.keras
#            graficos/curvas_treino.png
#            graficos/matriz_confusao.png
#            graficos/curva_roc.png
#            graficos/metricas_barras.png
# ================================================================

import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Reprodutibilidade
tf.random.set_seed(42)
np.random.seed(42)

# ── Caminhos ─────────────────────────────────────────────────
INPUT_NPZ    = "outputs/dados_treino.npz"
OUTPUT_MODEL = "models/modelo_ddos.keras"
G_CURVAS     = "graficos/curvas_treino.png"
G_MATRIZ     = "graficos/matriz_confusao.png"
G_ROC        = "graficos/curva_roc.png"
G_METRICAS   = "graficos/metricas_barras.png"
G_RESUMO     = "graficos/resumo_avaliacao.png"

# ── Hiperparâmetros ──────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 100
PATIENCE   = 15     # EarlyStopping
LR         = 1e-3   # learning rate inicial


# ════════════════════════════════════════════════════════════
#  1. ARQUITETURA Conv2D
# ════════════════════════════════════════════════════════════
def construir_modelo(input_shape):
    """
    CNN 2D para classificação binária de tráfego IoT.

    Arquitetura:
      Input (H×W×1)
        → Conv2D(32, 3×3) + BN + ReLU + MaxPool
        → Conv2D(64, 3×3) + BN + ReLU + MaxPool
        → Conv2D(128, 3×3) + BN + ReLU + GlobalAvgPool
        → Dense(128, relu, L2) + Dropout(0.4)
        → Dense(64,  relu, L2) + Dropout(0.3)
        → Dense(1, sigmoid)   ← saída binária
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # ── Bloco 1
    x = layers.Conv2D(32, (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2), padding="same", name="pool1")(x)

    # ── Bloco 2
    x = layers.Conv2D(64, (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2), padding="same", name="pool2")(x)

    # ── Bloco 3
    x = layers.Conv2D(128, (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Classificador
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, output, name="CNN2D_DDoS_IoT")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]
    )
    return model


# ════════════════════════════════════════════════════════════
#  2. TREINO
# ════════════════════════════════════════════════════════════
def treinar(model, X_tr, y_tr, X_val, y_val):
    os.makedirs("models", exist_ok=True)

    callbacks = [
        # Para quando val_auc não melhora por PATIENCE épocas
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduz LR quando val_loss estagna
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-7, verbose=1
        ),
        # Salva o melhor modelo
        keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL, monitor="val_auc",
            mode="max", save_best_only=True, verbose=0
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\nModelo salvo em: {OUTPUT_MODEL}")
    return history


# ════════════════════════════════════════════════════════════
#  3. AVALIAÇÃO — MÉTRICAS
# ════════════════════════════════════════════════════════════
def calcular_metricas(model, X_te, y_te):
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob)

    print("\n" + "="*55)
    print("  RESULTADO NO CONJUNTO DE TESTE")
    print("="*55)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print("\nRelatório completo:")
    print(classification_report(y_te, y_pred,
          target_names=["Normal", "Ataque"]))

    return y_pred, y_prob, {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
        "AUC-ROC":   auc,
    }


# ════════════════════════════════════════════════════════════
#  4. GRÁFICOS DE AVALIAÇÃO
# ════════════════════════════════════════════════════════════
def plot_curvas_treino(history):
    """Curvas de Loss, Accuracy, Precision, Recall e F1 por época."""
    os.makedirs("graficos", exist_ok=True)
    h = history.history

    # Calcula F1 por época manualmente
    def f1_epoca(prec, rec):
        return [2*p*r/(p+r+1e-8) for p, r in zip(prec, rec)]

    f1_tr  = f1_epoca(h["precision"], h["recall"])
    f1_val = f1_epoca(h["val_precision"], h["val_recall"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Curvas de Treino — CNN2D DDoS IoT",
                 fontsize=15, fontweight="bold")

    metricas = [
        ("loss",      "Loss (Binary Crossentropy)"),
        ("accuracy",  "Accuracy"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("auc",       "AUC-ROC"),
    ]

    for ax, (chave, titulo) in zip(axes.flat, metricas):
        ax.plot(h[chave],         label="Treino",    color="steelblue",  lw=2)
        ax.plot(h[f"val_{chave}"],label="Validação", color="darkorange", lw=2, linestyle="--")
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xlabel("Época")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=0.3)

    # Último subplot: F1-Score
    ax = axes.flat[5]
    ax.plot(f1_tr,  label="Treino",    color="steelblue",  lw=2)
    ax.plot(f1_val, label="Validação", color="darkorange", lw=2, linestyle="--")
    ax.set_title("F1-Score", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(G_CURVAS, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {G_CURVAS}")


def plot_matriz_confusao(y_te, y_pred):
    """Matriz de confusão com valores absolutos e percentuais."""
    cm      = confusion_matrix(y_te, y_pred)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matriz de Confusão — CNN2D DDoS IoT",
                 fontsize=14, fontweight="bold")

    labels = ["Normal", "Ataque"]

    # Absoluta
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[0], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title("Valores absolutos", fontsize=12)
    axes[0].set_ylabel("Real",     fontsize=11)
    axes[0].set_xlabel("Previsto", fontsize=11)

    # Percentual
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Oranges",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[1], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title("Percentual (%)", fontsize=12)
    axes[1].set_ylabel("Real",     fontsize=11)
    axes[1].set_xlabel("Previsto", fontsize=11)

    # Legenda dos quadrantes
    fig.text(0.5, -0.02,
             "TP=Verdadeiro Positivo  TN=Verdadeiro Negativo  "
             "FP=Falso Positivo  FN=Falso Negativo",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(G_MATRIZ, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {G_MATRIZ}")


def plot_curva_roc(y_te, y_prob):
    """Curva ROC com AUC destacado."""
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc         = roc_auc_score(y_te, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2.5,
            label=f"CNN2D  (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Aleatório (AUC = 0.5)")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)", fontsize=12)
    ax.set_title("Curva ROC — Detecção de DDoS em IoT",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(G_ROC, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {G_ROC}")


def plot_metricas_barras(metricas):
    """Barras das 4 métricas principais + AUC."""
    nomes  = list(metricas.keys())
    valores= list(metricas.values())
    cores  = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(nomes, valores, color=cores, edgecolor="white",
                  width=0.55, zorder=3)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_title("Métricas de Avaliação — CNN2D DDoS IoT",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0.9, color="gray", linestyle="--", lw=1, alpha=0.6,
               label="Referência 0.90")
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(G_METRICAS, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {G_METRICAS}")


def plot_resumo_avaliacao(y_te, y_pred, y_prob, metricas, history):
    """Painel resumo com todas as métricas em um único gráfico."""
    os.makedirs("graficos", exist_ok=True)
    h = history.history

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Avaliação Completa — CNN2D para Detecção de DDoS em IoT",
                 fontsize=16, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Gráfico 1: Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(h["loss"],     label="Treino",    color="steelblue", lw=2)
    ax1.plot(h["val_loss"], label="Validação", color="darkorange", lw=2, ls="--")
    ax1.set_title("Loss por Época", fontweight="bold")
    ax1.set_xlabel("Época"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    # ── Gráfico 2: Accuracy + F1
    f1_tr  = [2*p*r/(p+r+1e-8) for p,r in zip(h["precision"],     h["recall"])]
    f1_val = [2*p*r/(p+r+1e-8) for p,r in zip(h["val_precision"], h["val_recall"])]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(h["accuracy"],     label="Acc Treino",  color="steelblue", lw=2)
    ax2.plot(h["val_accuracy"], label="Acc Val",     color="darkorange", lw=2, ls="--")
    ax2.plot(f1_tr,             label="F1 Treino",   color="green",  lw=1.5, ls=":")
    ax2.plot(f1_val,            label="F1 Val",      color="red",    lw=1.5, ls="-.")
    ax2.set_title("Accuracy & F1 por Época", fontweight="bold")
    ax2.set_xlabel("Época"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    # ── Gráfico 3: AUC-ROC por época
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(h["auc"],     label="Treino",    color="steelblue", lw=2)
    ax3.plot(h["val_auc"], label="Validação", color="darkorange", lw=2, ls="--")
    ax3.set_title("AUC-ROC por Época", fontweight="bold")
    ax3.set_xlabel("Época"); ax3.legend(); ax3.grid(alpha=0.3)
    ax3.spines[["top","right"]].set_visible(False)

    # ── Gráfico 4: Matriz de confusão
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Ataque"],
                yticklabels=["Normal","Ataque"],
                ax=ax4, cbar=False,
                annot_kws={"size":13,"weight":"bold"})
    ax4.set_title("Matriz de Confusão", fontweight="bold")
    ax4.set_ylabel("Real"); ax4.set_xlabel("Previsto")

    # ── Gráfico 5: Curva ROC
    ax5 = fig.add_subplot(gs[1, 1])
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc = roc_auc_score(y_te, y_prob)
    ax5.plot(fpr, tpr, color="steelblue", lw=2.5, label=f"AUC={auc:.4f}")
    ax5.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    ax5.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax5.set_title("Curva ROC", fontweight="bold")
    ax5.set_xlabel("FPR"); ax5.set_ylabel("TPR")
    ax5.legend(fontsize=10); ax5.grid(alpha=0.3)
    ax5.spines[["top","right"]].set_visible(False)

    # ── Gráfico 6: Barras de métricas
    ax6 = fig.add_subplot(gs[1, 2])
    nomes  = ["Accuracy","Precision","Recall","F1-Score"]
    vals   = [metricas[k] for k in nomes]
    cores  = ["#2196F3","#4CAF50","#FF9800","#E91E63"]
    bars = ax6.bar(nomes, vals, color=cores, edgecolor="white", width=0.55, zorder=3)
    for bar, val in zip(bars, vals):
        ax6.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.01,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax6.set_ylim(0, 1.15)
    ax6.set_title("Métricas Finais (Teste)", fontweight="bold")
    ax6.axhline(0.9, color="gray", ls="--", lw=1, alpha=0.6)
    ax6.spines[["top","right"]].set_visible(False)
    ax6.grid(axis="y", alpha=0.3, zorder=0)

    plt.savefig(G_RESUMO, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {G_RESUMO}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  PASSO 3 — Treino e Avaliação")
    print("=" * 55)

    # Carrega dados
    data  = np.load(INPUT_NPZ)
    X_tr  = data["X_tr"];  y_tr  = data["y_tr"]
    X_val = data["X_val"]; y_val = data["y_val"]
    X_te  = data["X_te"];  y_te  = data["y_te"]
    lado  = int(data["lado"][0])

    print(f"Shape treino    : {X_tr.shape}")
    print(f"Shape validação : {X_val.shape}")
    print(f"Shape teste     : {X_te.shape}")
    print(f"Input Conv2D    : ({lado}×{lado}×1)")

    # Constrói modelo
    print("\n" + "="*55)
    print("  Arquitetura Conv2D")
    print("="*55)
    model = construir_modelo(input_shape=(lado, lado, 1))
    model.summary()

    # Treino
    print("\n" + "="*55)
    print("  Treinando...")
    print("="*55)
    history = treinar(model, X_tr, y_tr, X_val, y_val)

    # Avaliação
    y_pred, y_prob, metricas = calcular_metricas(model, X_te, y_te)

    # Gráficos individuais
    print("\nGerando gráficos...")
    plot_curvas_treino(history)
    plot_matriz_confusao(y_te, y_pred)
    plot_curva_roc(y_te, y_prob)
    plot_metricas_barras(metricas)

    # Painel resumo (todos em um)
    plot_resumo_avaliacao(y_te, y_pred, y_prob, metricas, history)

    print("\n" + "="*55)
    print("  Arquivos gerados")
    print("="*55)
    print(f"  Modelo  : {OUTPUT_MODEL}")
    print(f"  Gráficos: graficos/")
    print("\n[OK] Passo 3 concluído. Execute 04_monitoramento.py")
