# 🔍 Detector de Hotspots Comportamentais & FACS Forense

Este é um projeto em **Python** que utiliza conceitos do **FACS (Facial Action Coding System)** combinados com **Visão Computacional** e **Inteligência Artificial** para analisar vídeos em busca de microexpressões de estresse, assimetrias faciais repentinas e variações na taxa de piscadas.

O sistema conta com uma interface gráfica moderna desenvolvida em **PyQt6** e exporta automaticamente clipes de prova de 5 segundos para cada anomalia detectada.

## 🚀 Funcionalidades
- 🎥 **Player Integrado:** Interface gráfica nativa para carregar e assistir à análise do vídeo.
- ⏱️ **Detecção de Linha do Tempo:** Identifica e exibe carimbos de tempo (timestamps) de pontos suspeitos em tempo real.
- 💾 **Corte de Vídeo Automático:** Salva trechos de 5 segundos contendo o momento exato do alerta na pasta `trechos_suspeitos/`.
- 🧵 **Processamento Multi-thread:** A interface não trava durante o processamento pesado do MediaPipe.

## 🛠️ Como Instalar e Rodar

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o programa:**
   ```bash
   python main.py
   ```
