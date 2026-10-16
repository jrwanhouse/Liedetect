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

## ◆ Pontos de Atenção
> ⚠ **Pontos de atenção indicam variação emocional ou de estresse para revisão humana. NÃO indicam mentira.** Não existe sinal facial confiável de mentira (DePaulo et al., 2003; Bond & DePaulo, 2006). Nervosismo, constrangimento e a própria situação de entrevista produzem os mesmos sinais.

| Ponto | Critério |
|---|---|
| Microexpressão | emoção com duração < 0,5 s |
| Assimetria | AU ativa só de um lado do rosto por ≥ 0,2 s (exceto 30 e 46, laterais por natureza) |
| Sorriso sem AU6 | AU12 bilateral sem AU6 por ≥ 0,2 s (sorriso que não chega aos olhos) |
| Piscadas acima do padrão / supressão | janela de 10 s com ≥ 2× ou ≤ 0,3× a mediana das janelas anteriores do mesmo vídeo |

Os critérios ficam no `atencao.py`.

## ❓ Perguntas (opcional)
Marcar perguntas é opcional; sem elas a análise funciona normalmente. Cada pergunta cobre o trecho da sua marcação até a próxima.

- **No player (interface gráfica):** digite o texto, que é opcional, e aperte **F2** ou *Marcar pergunta*. A marcação usa o tempo do frame que está na tela. No modo *Revisão* dá para pausar e voltar antes de marcar, e o resumo é recalculado na hora.
- **Arquivo:** em *Importar perguntas...* ou no modo terminal, use uma linha por pergunta, no formato `tempo;texto`. O tempo pode ser em segundos, `mm:ss` ou `hh:mm:ss`:
  ```
  0:00;Qual seu nome?
  0:20;Onde estava ontem?
  0:45;Conhece a vítima?
  ```

No fim, o **resumo por pergunta** (`relatorio_perguntas.csv`) mostra para cada pergunta:
- as emoções detectadas e a reação nos 3 s seguintes à marcação;
- os pontos de atenção e as piscadas por minuto;
- se a pergunta **destoa das demais**, quando há pelo menos 3 perguntas para comparar.

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

## ⚙️ Preferências
As preferências ficam em um arquivo JSON, usado tanto pela interface gráfica quanto pelo modo terminal. Por padrão é o `preferencias.json`, na pasta do programa, e outro arquivo pode ser indicado com `--config`:
```bash
python main.py --config minhas_preferencias.json
```
Na interface gráfica elas são editadas em **Arquivo › Preferências** (Ctrl+,) e salvas no mesmo arquivo. Chaves ausentes ou inválidas voltam ao valor padrão.

| Chave | Padrão | Uso |
|---|---|---|
| `modo_player` | `"analise"` | GUI. `"analise"`: o vídeo é exibido durante a análise, e a velocidade do áudio acompanha o ritmo da análise, mantendo o tom da voz. `"revisao"`: a análise roda primeiro, depois abre um player com áudio em tempo real, play/pausa (Espaço) e navegação por clique na linha do tempo ou duplo clique num evento |
| `audio` / `volume` | `true` / `80` | GUI: liga/desliga o som e define o volume (0–100) |
| `webcam_espelhada` | `false` | GUI e terminal: valor inicial da opção de vídeo espelhado |
| `calibracao_segundos` | `1.0` | GUI e terminal: duração da calibração neutra |
| `pasta_saida` | `"trechos_suspeitos"` | GUI e terminal: onde ficam relatórios e recortes |
| `salvar_recortes` | `true` | GUI e terminal: grava o recorte de 5 s de cada evento. Desligar deixa a análise mais rápida |
