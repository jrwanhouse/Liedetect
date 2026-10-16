# 🔍 Analisador de FACS Forense Híbrido (GUI / CLI)

Este software utiliza Inteligência Artificial e a arquitetura geométrica do **MediaPipe** para realizar análises comportamentais em arquivos de vídeo baseando-se no sistema **FACS (Facial Action Coding System)**.

## 🌟 Detecção Inteligente de Inicialização
- **Modo Gráfico (GUI):** Ativado automaticamente em desktops convencionais. Abre a janela do player em **PySide6** (Qt for Python, licença LGPL).
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
| Medo | 1+2+4+5+7+20+26, 1+2+4+5+20(+26), 1+2+4+5(+25/26/27), 1+2+4+20, 5+20 |
| Raiva | 4+5+7+23, 4+5+23, 4+7+23, 4+5+7 (nunca com AU2) |
| Nojo | 9, 9+15+16, 9+17, 10+17, 10+16+25, 10+15 |
| Desprezo | AU12 ou AU14 unilateral |

**Vídeo de webcam:** webcams em modo selfie gravam a imagem espelhada, o que inverte esquerda/direita. Marque *"Vídeo de webcam (imagem espelhada)"* na interface gráfica (ou responda `s` no modo terminal). Assim cada frame é desespelhado antes da análise, e o lado das AUs (L/R), o olhar (61/62) e a cabeça (51/52, 55/56) ficam no referencial real do sujeito. Os recortes salvos continuam idênticos ao arquivo original.

**Calibração:** o primeiro segundo com rosto visível define a linha de base neutra do sujeito (os blendshapes têm viés individual). O vídeo deve começar com o rosto em repouso.

**Saídas:** no vídeo, cada AU ativa é marcada no ponto do rosto onde ocorre, na cor da emoção que integra. A interface gráfica mostra uma linha do tempo por emoção, e o modo terminal imprime um resumo dos trechos. Na pasta de saída (`<projeto>_saida/` ou, no terminal sem projeto, `trechos_suspeitos/`) ficam o `relatorio_emocoes.csv` (início, fim, emoção e combinação de AUs) e um recorte de 5 s de cada episódio.

## 🔬 Detector alternativo: Py-Feat (opcional)
Cada projeto pode usar o **MediaPipe** (padrão) ou o **Py-Feat** para medir AUs e emoções. A escolha fica em **Arquivo › Configurações do projeto** (ou nas **Preferências**, como padrão para novos projetos). O resto do sistema é o mesmo nos dois: episódios, pontos de atenção, perguntas, player de revisão, relatórios e validação FACS. Cada análise registra qual detector usou.

> ⚠ **Licença:** o código do Py-Feat é MIT, mas **os pesos dos modelos não estão liberados para uso comercial**: as bases de treino são acadêmicas, e o img2pose é CC BY-NC. Use o Py-Feat para pesquisa e comparação, e não o distribua no produto vendido sem análise jurídica.

**Instalação.** O Py-Feat precisa de PyTorch, que não existe para o Python 3.15. Por isso ele roda num **ambiente separado**, com Python 3.11 a 3.14, chamado pelo programa como processo externo (`pyfeat_worker.py`):
```bash
py -3.14 -m venv .venv-pyfeat
.venv-pyfeat\Scripts\pip install -r requirements-pyfeat.txt
```
Se o ambiente estiver em outro lugar, informe o Python dele em **Preferências › Python do Py-Feat**. A primeira análise baixa os pesos dos modelos (alguns minutos).

**Como funciona.** O Py-Feat processa o vídeo inteiro primeiro, com o progresso e o tempo restante na barra de status. Depois o vídeo segue pelo mesmo caminho do MediaPipe. A extração fica guardada na pasta de saída do projeto e é reaproveitada ao analisar de novo, por exemplo ao trocar o modo de emoções.

| | MediaPipe | Py-Feat |
|---|---|---|
| AUs faciais | 27, por regras sobre blendshapes | 20 do classificador treinado com FACS (1, 2, 4, 5, 6, 7, 9, 10, **11**, 12, 14, 15, 17, 20, 23, **24**, 25, 26, 28) |
| Intensidade A–E | escala do sinal | aproximada, pela probabilidade (0,5–1,0) |
| Lado L/R e Desprezo | sim | não (o Py-Feat não informa lado) |
| Piscadas (43/45/46) | blendshapes | abertura das pálpebras nos 68 pontos (EAR). O AU43 do modelo é ignorado, porque dispara em sorrisos de olhos abertos |
| Cabeça (51–58) | geometria dos pontos | a mesma geometria, sobre os 68 pontos |
| Olhar (61–64) | sim | não |
| Emoções | regras EMFACS | **classificador do Py-Feat** (padrão) ou **regras EMFACS** sobre as AUs do Py-Feat |
| Velocidade em CPU | tempo real | **~2,4 s por frame** (Detectorv2; o v1 ~3,5 s): 1 min de vídeo ≈ 70 min. A opção *analisar 1 a cada N frames* acelera, mas pode perder microexpressões |

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

## ✅ Validação com codificadores FACS
As AUs saem de regras sobre os blendshapes do MediaPipe, que foram feitos para animação e não treinados com FACS. Antes de usar os resultados como evidência, valide o sistema contra codificação humana.

1. **Exportar uma amostra** de um projeto analisado (menu **Validação FACS › Exportar amostra para codificação**, ou `python main.py --projeto caso.liedetect --exportar-validacao PASTA --amostras 100`). O sorteio é reprodutível (`--semente`) e mistura frames em que o sistema viu cada AU com frames ao acaso, a pelo menos 0,5 s um do outro. Para cada amostra são gravados a imagem (sem marcações) e um trecho de ~1 s, mais a planilha em branco `codificacao_modelo.csv` e um `LEIA-ME.txt`.
2. **Codificação cega:** cada codificador certificado preenche a sua cópia da planilha, sem ver as respostas do sistema, que ficam em `predicoes_sistema.csv` (não entregue esse arquivo). Em cada célula: vazio/0 = ausente, 1/X = presente, A–E = presente com intensidade.
3. **Avaliar** (menu **Validação FACS › Avaliar codificação**, ou `python main.py --avaliar-validacao PASTA cod1.csv cod2.csv ...`). O consenso é a maioria dos codificadores, e as amostras empatadas ficam de fora. O `relatorio_validacao.csv` traz, por AU:
   - **Sistema × consenso:** VP/FP/FN/VN, precisão, revocação, F1 e kappa de Cohen.
   - **Entre codificadores:** kappa médio, que é o teto de referência.
   - **Intensidade:** erro médio em letras.
   - **Limiar sugerido:** o que maximiza o F1.
   - **Situação:** *validada* (F1 ≥ 0,6), *validada com limiar ajustado*, *não validada* ou *amostra insuficiente* (< 10 casos positivos).
4. **Aplicar os limiares sugeridos** (botão no resultado da avaliação, ou `--aplicar-limiares PASTA`). Eles ficam em `limiares_aus.json`, na pasta do programa, e valem para as próximas análises. Cada análise registra no projeto os limiares ajustados que usou (`limiares_ajustados`), para rastreabilidade.

Use vídeos próprios, gravados com consentimento: as bases públicas codificadas em FACS (DISFA, BP4D etc.) costumam ter licença só acadêmica.

## 🧪 Testes
```bash
python -m unittest discover -s tests -v
```
- `tests/test_aus.py`: cada AU isolada deve gerar só ela mesma. Cobre ainda AU6×AU7 no sorriso, AU25/26/27 pela abertura da boca, pálpebras (AU43/45/46, sem AU7 nem olhar falsos ao piscar), olhar (61–64), lado L/R, cabeça (51–58, sem AU53 falsa com a boca aberta), intensidade e coerência das tabelas com o FACS.
- `tests/test_validacao.py`: kappa, F1 e leitura das células. Fluxo completo: exportação cega, consenso com empates, falsos positivos detectados, limiar sugerido e aplicação.
- `tests/test_gui.py`: a interface sem janela visível (criar, analisar, marcar perguntas, salvar, reabrir, trocar de projeto durante a análise, menu de validação) e a criação do detector **real** do MediaPipe numa thread do PySide6.
- `tests/test_pyfeat.py`: o detector Py-Feat sem o Py-Feat instalado (linhas sintéticas e um worker falso com o mesmo protocolo): AUs, emoções nos dois modos, piscadas pela abertura das pálpebras, cabeça, *1 a cada N frames*, erros, interrupção e reaproveitamento. O teste com o Py-Feat real roda com `LIEDETECT_TESTE_PYFEAT=1`.
- `tests/test_medo.py`: as regras EMFACS de medo, o caminho blendshapes → AUs → emoção (com calibração) e uma sequência de vídeo simulada, com microexpressão e episódio sustentado.

Os testes usam um detector simulado: verificam a lógica do sistema, não a precisão do MediaPipe em rostos reais. Duas convenções do modelo foram conferidas em foto real: o "Left" dos blendshapes é o lado esquerdo do sujeito, e fechar os olhos também eleva `eyeSquint` e `eyeLookDown`, por isso essas AUs são suprimidas durante a piscada.

## 📜 Licenças
O sistema é Apache 2.0. As dependências permitem uso comercial e código fechado:
- **PySide6:** LGPL-3.0.
- **MediaPipe e OpenCV:** Apache 2.0.
- **NumPy:** BSD.

A interface usava PyQt6, que é GPL-3.0 e foi trocado pelo PySide6. Ao distribuir o programa, respeite a LGPL do Qt: use as bibliotecas do Qt como bibliotecas dinâmicas e inclua o aviso de licença.

**Python 3.15 + PySide6 6.12:** o `compat.py` desliga os imports adiados (PEP 810), porque o shiboken6 impede que eles sejam resolvidos e o MediaPipe falhava ("'lazy_import' object is not callable"). Ele precisa ser o primeiro import.

## 🛠️ Instalação das Dependências
```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Na primeira execução o modelo `face_landmarker.task` (MediaPipe Tasks) é baixado automaticamente para a pasta do projeto.

## 🚀 Como Executar
```bash
python main.py                                  # interface gráfica (ou terminal, sem ambiente gráfico)
python main.py --projeto caso01.liedetect       # abre o projeto na interface gráfica
python main.py --terminal --projeto caso01.liedetect   # analisa o projeto no terminal e salva o resultado nele
python main.py --terminal                       # modo terminal interativo, sem projeto
```

## 📁 Projetos
Cada análise é um **projeto** (`.liedetect`), um arquivo JSON legível que guarda:
- o vídeo, com o caminho relativo ao projeto, para que vídeo e projeto possam ser movidos juntos;
- as configurações (vídeo de webcam, calibração, recortes);
- as perguntas;
- o resultado da análise.

Relatórios, recortes e os dados de cada frame (`quadros.json.gz`, usados pelo player de revisão) ficam em `<projeto>_saida/`, ao lado do arquivo do projeto.

Na interface gráfica, pelo menu **Arquivo**:
- **Novo projeto** (Ctrl+N): escolha o vídeo e onde salvar o projeto. As configurações iniciais vêm das preferências.
- **Abrir projeto** (Ctrl+O): se o projeto já foi analisado, a linha do tempo, os eventos, o resumo e o player de revisão voltam sem precisar analisar de novo. Se o vídeo tiver mudado de lugar, o sistema pede para localizá-lo.
- **Salvar** (Ctrl+S) / **Salvar como** (Ctrl+Shift+S): o "Salvar como" copia junto a pasta de saída. O título mostra `*` quando há alterações, e o sistema avisa antes de fechar ou trocar de projeto.
- **Configurações do projeto**: webcam, calibração e recortes. Mudanças valem para a próxima análise.

Ao terminar, a análise é salva automaticamente no projeto, e o botão **Parar análise** guarda o resultado parcial. Perguntas marcadas ou importadas depois da análise recalculam o resumo na hora. Para remover uma pergunta, selecione-a na lista e aperte **Delete**.

## ⚙️ Preferências
As preferências ficam em um arquivo JSON, usado tanto pela interface gráfica quanto pelo modo terminal. Por padrão é o `preferencias.json`, na pasta do programa, e outro arquivo pode ser indicado com `--config`:
```bash
python main.py --config minhas_preferencias.json
```
Na interface gráfica elas são editadas em **Arquivo › Preferências** (Ctrl+,) e salvas no mesmo arquivo. Chaves ausentes ou inválidas voltam ao valor padrão.

| Chave | Padrão | Uso |
|---|---|---|
| `modo_player` | `"analise"` | GUI. `"analise"`: o vídeo é exibido durante a análise, com o áudio acompanhando o ritmo da análise e mantendo o tom da voz. `"revisao"`: a análise roda o mais rápido possível, mostrando só uma prévia. Nos dois modos, ao fim da análise (ou ao abrir um projeto analisado), abre o player de revisão: áudio em tempo real, play/pausa (Espaço) e navegação por clique na linha do tempo ou duplo clique num evento |
| `audio` / `volume` | `true` / `80` | GUI: liga/desliga o som e define o volume (0–100) |
| `webcam_espelhada` | `false` | Padrão para novos projetos e para o terminal sem projeto |
| `calibracao_segundos` | `1.0` | Idem: duração da calibração neutra |
| `pasta_saida` | `"trechos_suspeitos"` | Terminal sem projeto: onde ficam relatórios e recortes |
| `salvar_recortes` | `true` | Padrão para novos projetos e para o terminal sem projeto: grava o recorte de 5 s de cada evento. Desligar deixa a análise mais rápida |
