# Detecção de DDoS em Redes IoT com CNN2D

## Estrutura do projeto

```
ddos_iot/
├── extracao.py          # Extrai features dos pcapng + janelas deslizantes
├── preprocessamento.py  # Normalização + SMOTE + divisão treino/val/teste
├── treino.py            # CNN2D + treino + avaliação gráfica completa
├── monitoramento.py     # Monitoramento em tempo real com modelo treinado
├── requirements.txt
├── dataset_pcap/           # Coloque seus arquivos aqui
│   ├── normal.pcapng
│   └── ataque.pcapng
├── outputs/                # Gerado automaticamente
├── models/                 # Gerado automaticamente
└── graficos/               # Gerado automaticamente
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução — ordem obrigatória

### Passo 1 — Extração
```bash
python3 extracao.py
```
Gera: `outputs/dataset_janelas.npz`, `outputs/dataset_raw.csv`

### Passo 2 — Pré-processamento
```bash
python3 preprocessamento.py
```
Gera: `outputs/dados_treino.npz`, `outputs/scaler.pkl`,
      `graficos/distribuicao_classes.png`

### Passo 3 — Treino e avaliação
```bash
python3 treino.py
```
Gera:
- `models/modelo_ddos.keras`
- `graficos/curvas_treino.png`
- `graficos/matriz_confusao.png`
- `graficos/curva_roc.png`
- `graficos/metricas_barras.png`
- `graficos/resumo_avaliacao.png`  ← painel completo

### Passo 4 — Monitoramento ao vivo
```bash
sudo python3 monitoramento.py
```
Monitora os Workers e Master em tempo real, classificando janelas de
pacotes a cada 5 novos pacotes capturados.

## Dispositivos monitorados

| Dispositivo | IP           |
|-------------|--------------|
| Master      | 10.42.0.100  |
| Worker 1    | 10.42.0.101  |
| Worker 2    | 10.42.0.102  |

## Arquivos pcapng esperados

Renomeie seus arquivos de captura para:
- `dataset_pcap/normal.pcapng`  — tráfego benigno
- `dataset_pcap/ataque.pcapng`  — tráfego sob ataque DDoS
