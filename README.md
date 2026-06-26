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

