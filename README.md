# 🔍 Analisador de FACS Forense Híbrido (GUI / CLI)

Este software utiliza Inteligência Artificial e a arquitetura geométrica do **MediaPipe** para realizar análises comportamentais em arquivos de vídeo baseando-se no sistema **FACS (Facial Action Coding System)**.

## 🌟 Detecção Inteligente de Inicialização
- **Modo Gráfico (GUI):** Ativado automaticamente em desktops convencionais. Abre a janela do player em **PyQt6**.
- **Modo Terminal (CLI):** Ativado em servidores remotos (SSH ou nuvem). Opera inteiramente por comandos de texto.

## 🎭 Action Units e Emoções
As AUs faciais vêm dos 52 *blendshapes* do MediaPipe FaceLandmarker, e as AUs de cabeça (51–58) da geometria dos landmarks. Cada AU recebe intensidade FACS de **A** (traço) a **E** (máxima) e lado **L**/**R** quando é unilateral.

- **Detectadas:** 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 18, 20, 22, 23/24, 25, 26, 27, 28, 29, 30, 34, 43, 45, 46, 51–58, 61–64.
- **Não detectáveis** (sem sinal no modelo): 8, 11, 13, 19, 21, 31–33, 35–39, 41, 42, 44. A AU24 não se distingue da AU23 e é reportada como AU23/24.

Emoções pelas combinações EMFACS (protótipos e variantes de Ekman & Friesen), definidas em `EMOCOES` no `facs.py`:

| Emoção | Combinações principais |
|---|---|
| Felicidade | 6+12 |
| Tristeza | 1+4+15, 1+4+15+17, 6+15 |
| Surpresa | 1+2+5+26/27, 1+2+26, 1+2+5 |
| Medo | 1+2+4+5+20(+26), 1+2+4+5, 1+2+4+20, 5+20 |
| Raiva | 4+5+7+23, 4+5+23, 4+7+23, 4+5+7 |
| Nojo | 9, 9+15+16, 9+17, 10+17, 10+16+25, 10+15 |
| Desprezo | AU12 ou AU14 unilateral |

**Vídeo de webcam:** webcams em modo selfie gravam a imagem espelhada, o que inverte esquerda/direita. Marque *"Vídeo de webcam (imagem espelhada)"* na interface gráfica (ou responda `s` no modo terminal). Assim cada frame é desespelhado antes da análise, e o lado das AUs (L/R), o olhar (61/62) e a cabeça (51/52, 55/56) ficam no referencial real do sujeito. Os recortes salvos continuam idênticos ao arquivo original.

**Calibração:** o primeiro segundo com rosto visível define a linha de base neutra do sujeito (os blendshapes têm viés individual). O vídeo deve começar com o rosto em repouso.

**Saídas:** no vídeo, cada AU ativa é marcada no ponto do rosto onde ocorre, na cor da emoção que integra. A interface gráfica mostra uma linha do tempo por emoção, e o modo terminal imprime um resumo dos trechos. Em `trechos_suspeitos/` ficam o `relatorio_emocoes.csv` (início, fim, emoção e combinação de AUs) e um recorte de 5 s de cada episódio.

## 🛠️ Instalação das Dependências
```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Na primeira execução o modelo `face_landmarker.task` (MediaPipe Tasks) é baixado automaticamente para a pasta do projeto.

## 🚀 Como Executar
```bash
python main.py
```
