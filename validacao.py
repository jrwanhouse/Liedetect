"""Validacao das AUs do sistema contra codificacao FACS humana.

1. exportar_amostra(): sorteia frames de um projeto analisado e grava, para cada um, a imagem
   (sem marcacoes do sistema) e um trecho de ~1 s, mais a planilha em branco que os
   codificadores preenchem. As predicoes do sistema vao para predicoes_sistema.csv, que NAO
   deve ser entregue aos codificadores (codificacao cega).
2. avaliar(): le as planilhas preenchidas e mede, por AU, a concordancia do sistema com o
   consenso dos codificadores e a dos codificadores entre si; sugere o limiar de cada AU.
"""
import csv
import json
import os
import random
from datetime import datetime
from itertools import combinations
from statistics import median

import cv2

import facs

PASTA_AMOSTRAS = "amostras"
ARQUIVO_META = "amostra.json"
ARQUIVO_MODELO = "codificacao_modelo.csv"
ARQUIVO_PREDICOES = "predicoes_sistema.csv"
ARQUIVO_RELATORIO = "relatorio_validacao.csv"
ARQUIVO_LIMIARES_SUGERIDOS = "limiares_sugeridos.json"

# AUs faciais codificaveis em frame/trecho curto (cabeca e olhar ficam de fora por padrao)
AUS_FACIAIS = sorted(au for au in facs.NOMES_AU if au < 50)
ESPACO_MINIMO_S = 0.5      # entre frames sorteados, para nao repetir o mesmo instante
DURACAO_TRECHO_S = 1.0
MIN_POSITIVOS = 10         # casos positivos no consenso para avaliar uma AU
F1_VALIDADA = 0.6
GANHO_MINIMO_F1 = 0.05  # melhora no F1 para sugerir trocar o limiar
LETRAS = {letra: i + 1 for i, letra in enumerate("ABCDE")}

LEIA_ME = """CODIFICACAO FACS - INSTRUCOES

Cada amostra tem uma imagem (.png) e um trecho de ~1 s (.mp4) centrado nela.
Codifique o frame da IMAGEM; use o trecho so como contexto de movimento.

Preencha uma copia de {modelo} (uma por codificador, sem consultar os demais):
  - vazio ou 0  -> AU ausente
  - 1 ou X      -> AU presente (sem intensidade)
  - A, B, C, D, E -> AU presente com essa intensidade

Nao abra {predicoes}: ela contem as respostas do sistema e quebraria a codificacao cega.
"""


def _eh_amostravel(r):
    return not r.calibrando and r.landmarks is not None


def _score(r, au):
    if r.scores:
        return r.scores.get(au, 0.0)
    return r.aus[au].score if au in r.aus else 0.0  # projetos sem scores gravados


def sortear_frames(resultados, fps, n, aus=AUS_FACIAIS, semente=0):
    """Metade dos frames com cada AU ativa no sistema (estratificado), metade ao acaso."""
    rng = random.Random(semente)
    candidatos = [i for i, r in enumerate(resultados) if _eh_amostravel(r)]
    espaco = max(1, round(fps * ESPACO_MINIMO_S))
    escolhidos = set()

    def livre(i):
        return all(abs(i - j) >= espaco for j in escolhidos)

    ocorrem = {au: [i for i in candidatos if au in resultados[i].aus] for au in aus}
    ocorrem = {au: frames for au, frames in ocorrem.items() if frames}
    por_au = max(1, (n // 2) // max(1, len(ocorrem)))
    for au, com_au in ocorrem.items():
        rng.shuffle(com_au)
        for i in [i for i in com_au if livre(i)][:por_au]:
            if livre(i) and len(escolhidos) < n // 2:
                escolhidos.add(i)
    resto = candidatos[:]
    rng.shuffle(resto)
    for i in resto:
        if len(escolhidos) >= n:
            break
        if livre(i):
            escolhidos.add(i)
    return sorted(escolhidos)


def exportar_amostra(proj, pasta, n=100, aus=AUS_FACIAIS, semente=0):
    """Grava a amostra para codificacao; retorna o numero de amostras exportadas."""
    if not (proj.analise and proj.resultados):
        raise ValueError("o projeto precisa estar analisado (com resultados por frame)")
    fps = proj.analise["fps"]
    frames = sortear_frames(proj.resultados, fps, n, aus, semente)
    if not frames:
        raise ValueError("nenhum frame com rosto analisado para sortear")
    os.makedirs(os.path.join(pasta, PASTA_AMOSTRAS), exist_ok=True)
    cap = cv2.VideoCapture(proj.video)
    meia = int(fps * DURACAO_TRECHO_S / 2)
    linhas_modelo, linhas_pred = [], []
    for n_amostra, idx in enumerate(frames, 1):
        nome = f"a{n_amostra:03d}"
        r = proj.resultados[idx]
        inicio = max(0, idx - meia)
        cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
        trecho, imagem = [], None
        for i in range(inicio, idx + meia + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if r.espelhado:  # o codificador ve o lado real do sujeito, como o sistema
                frame = cv2.flip(frame, 1)
            trecho.append(frame)
            if i == idx:
                imagem = frame
        if imagem is None:
            continue
        cv2.imwrite(os.path.join(pasta, PASTA_AMOSTRAS, nome + ".png"), imagem)
        h, w = imagem.shape[:2]
        out = cv2.VideoWriter(os.path.join(pasta, PASTA_AMOSTRAS, nome + ".mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in trecho:
            out.write(f)
        out.release()
        linhas_modelo.append([nome, f"{PASTA_AMOSTRAS}/{nome}.png", f"{PASTA_AMOSTRAS}/{nome}.mp4",
                              facs.formatar_tempo(r.tempo)] + [""] * len(aus))
        linhas_pred.append([nome, idx, f"{r.tempo:.3f}"] + [
            v for au in aus for v in (f"{_score(r, au):.3f}", r.aus[au].intensidade if au in r.aus else "")])
    cap.release()

    _gravar_csv(os.path.join(pasta, ARQUIVO_MODELO),
                ["amostra", "imagem", "trecho", "tempo"] + [f"AU{au}" for au in aus], linhas_modelo)
    _gravar_csv(os.path.join(pasta, ARQUIVO_PREDICOES),
                ["amostra", "frame", "tempo_s"] + [c for au in aus for c in (f"AU{au}_score", f"AU{au}_sistema")],
                linhas_pred)
    with open(os.path.join(pasta, ARQUIVO_META), "w", encoding="utf-8") as f:
        json.dump({"projeto": proj.caminho, "video": proj.video, "fps": fps, "semente": semente,
                   "aus": aus, "amostras": len(linhas_modelo), "exportado_em": datetime.now().isoformat(timespec="seconds"),
                   # limiares com que o projeto foi analisado (as colunas _sistema seguem estes)
                   "limiares": {str(au): float(proj.analise.get("limiares_ajustados", {}).get(str(au), facs.LIMIARES_PADRAO[au]))
                                for au in aus}}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(pasta, "LEIA-ME.txt"), "w", encoding="utf-8") as f:
        f.write(LEIA_ME.format(modelo=ARQUIVO_MODELO, predicoes=ARQUIVO_PREDICOES))
    return len(linhas_modelo)


def _gravar_csv(caminho, cabecalho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(cabecalho)
        wr.writerows(linhas)


def _ler_csv(caminho):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        texto = f.read()
    delimitador = ";" if texto.count(";") >= texto.count(",") else ","
    return list(csv.DictReader(texto.splitlines(), delimiter=delimitador))


def ler_celula(valor):
    """(presente, intensidade 1-5 ou None) de uma celula de codificacao."""
    v = (valor or "").strip().upper()
    if v in ("", "0", "-"):
        return False, None
    if v in LETRAS:
        return True, LETRAS[v]
    if v in ("1", "X", "S", "SIM"):
        return True, None
    raise ValueError(f"valor de codificacao invalido: {valor!r}")


def ler_codificacao(caminho, aus):
    """{amostra: {au: (presente, intensidade)}} de uma planilha preenchida."""
    codigos = {}
    for n_linha, linha in enumerate(_ler_csv(caminho), 2):
        try:
            codigos[linha["amostra"].strip()] = {au: ler_celula(linha.get(f"AU{au}")) for au in aus}
        except ValueError as e:
            raise ValueError(f"{os.path.basename(caminho)}, linha {n_linha}: {e}") from None
    return codigos


def kappa(a, b):
    """Kappa de Cohen entre duas listas de booleanos."""
    n = len(a)
    if n == 0:
        return None
    concordancia = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    esperada = pa * pb + (1 - pa) * (1 - pb)
    if esperada == 1:
        return 1.0 if concordancia == 1 else 0.0
    return (concordancia - esperada) / (1 - esperada)


def metricas(sistema, verdade):
    vp = sum(s and v for s, v in zip(sistema, verdade))
    fp = sum(s and not v for s, v in zip(sistema, verdade))
    fn = sum(v and not s for s, v in zip(sistema, verdade))
    precisao = vp / (vp + fp) if vp + fp else None
    revocacao = vp / (vp + fn) if vp + fn else None
    f1 = 2 * vp / (2 * vp + fp + fn) if 2 * vp + fp + fn else None
    return {"vp": vp, "fp": fp, "fn": fn, "vn": len(verdade) - vp - fp - fn,
            "precisao": precisao, "revocacao": revocacao, "f1": f1, "kappa": kappa(sistema, verdade)}


def sugerir_limiar(scores, verdade):
    """Limiar (0.05..0.95) que maximiza o F1 contra o consenso; (limiar, f1)."""
    melhor = (None, -1.0)
    for passo in range(1, 20):
        lim = passo / 20
        f1 = metricas([s >= lim for s in scores], verdade)["f1"] or 0.0
        if f1 > melhor[1]:
            melhor = (lim, f1)
    return melhor


def avaliar(pasta, planilhas):
    """Compara o sistema com o consenso das planilhas; grava relatorio e limiares sugeridos."""
    with open(os.path.join(pasta, ARQUIVO_META), encoding="utf-8") as f:
        meta = json.load(f)
    aus = meta["aus"]
    predicoes = {linha["amostra"]: linha for linha in _ler_csv(os.path.join(pasta, ARQUIVO_PREDICOES))}
    codificadores = [ler_codificacao(p, aus) for p in planilhas]
    if not codificadores:
        raise ValueError("nenhuma planilha de codificacao informada")

    resultado = []
    for au in aus:
        lim = meta["limiares"].get(str(au), facs.LIMIAR_PADRAO)
        sistema, verdade, scores, dif_intensidade, empates = [], [], [], [], 0
        for amostra, pred in predicoes.items():
            votos = [c[amostra][au] for c in codificadores if amostra in c]
            if not votos:
                continue
            presentes = sum(p for p, _ in votos)
            if presentes * 2 == len(votos):
                empates += 1  # sem maioria: sem verdade de referencia para esta amostra
                continue
            consenso = presentes * 2 > len(votos)
            score = float(pred[f"AU{au}_score"])
            letra_sistema = pred[f"AU{au}_sistema"]
            sistema.append(bool(letra_sistema))
            verdade.append(consenso)
            scores.append(score)
            intensidades = [i for p, i in votos if p and i]
            if consenso and letra_sistema and intensidades:
                dif_intensidade.append(abs(LETRAS[letra_sistema] - median(intensidades)))

        m = metricas(sistema, verdade)
        positivos = sum(verdade)
        entre = [kappa([c[a][au][0] for a in comuns], [d[a][au][0] for a in comuns])
                 for c, d in combinations(codificadores, 2) if (comuns := sorted(set(c) & set(d)))]
        entre = [k for k in entre if k is not None]
        limiar_sugerido, f1_sugerido = sugerir_limiar(scores, verdade) if positivos else (None, None)
        f1_atual = m["f1"] or 0.0
        sugerir = (positivos >= MIN_POSITIVOS and f1_sugerido is not None and f1_sugerido >= F1_VALIDADA
                   and f1_sugerido >= f1_atual + GANHO_MINIMO_F1 and limiar_sugerido != lim)
        if positivos < MIN_POSITIVOS:
            situacao = f"amostra insuficiente ({positivos} positivos; minimo {MIN_POSITIVOS})"
        elif f1_atual >= F1_VALIDADA:
            situacao = "validada" + (" (o limiar sugerido melhora o F1)" if sugerir else "")
        elif sugerir:
            situacao = "validada com limiar ajustado"
        else:
            situacao = "nao validada"
        resultado.append({
            "au": au, "nome": facs.NOMES_AU[au], "amostras": len(verdade), "empates": empates,
            "positivos": positivos, **m, "limiar": lim,
            "limiar_sugerido": limiar_sugerido, "f1_sugerido": f1_sugerido, "sugerir": sugerir,
            "kappa_codificadores": sum(entre) / len(entre) if entre else None,
            "erro_intensidade": sum(dif_intensidade) / len(dif_intensidade) if dif_intensidade else None,
            "situacao": situacao,
        })

    _gravar_relatorio(os.path.join(pasta, ARQUIVO_RELATORIO), resultado, len(codificadores))
    sugeridos = {str(r["au"]): r["limiar_sugerido"] for r in resultado if r["sugerir"]}
    with open(os.path.join(pasta, ARQUIVO_LIMIARES_SUGERIDOS), "w", encoding="utf-8") as f:
        json.dump(sugeridos, f, indent=2)
    return resultado


def _fmt(v, casas=2):
    return "" if v is None else f"{v:.{casas}f}"


def _gravar_relatorio(caminho, resultado, n_codificadores):
    linhas = [[f"AU{r['au']}", r["nome"], r["amostras"], r["positivos"], r["empates"], r["vp"], r["fp"], r["fn"], r["vn"],
               _fmt(r["precisao"]), _fmt(r["revocacao"]), _fmt(r["f1"]), _fmt(r["kappa"]),
               _fmt(r["kappa_codificadores"]), _fmt(r["erro_intensidade"]), _fmt(r["limiar"]),
               _fmt(r["limiar_sugerido"]), _fmt(r["f1_sugerido"]), r["situacao"]] for r in resultado]
    _gravar_csv(caminho, ["au", "nome", "amostras", "positivos_consenso", "empates", "vp", "fp", "fn", "vn",
                          "precisao", "revocacao", "f1", "kappa_sistema",
                          f"kappa_entre_codificadores (n={n_codificadores})", "erro_intensidade_letras",
                          "limiar_atual", "limiar_sugerido", "f1_com_limiar_sugerido", "situacao"], linhas)


def texto_resumo(resultado):
    """Tabela em texto para terminal e interface."""
    linhas = [f"{'AU':6s} {'pos':>4s} {'F1':>5s} {'kappa':>6s} {'k.cod':>6s} {'lim':>5s} {'sug':>5s}  situacao"]
    for r in resultado:
        linhas.append(f"AU{r['au']:<4d} {r['positivos']:>4d} {_fmt(r['f1']):>5s} {_fmt(r['kappa']):>6s} "
                      f"{_fmt(r['kappa_codificadores']):>6s} {_fmt(r['limiar']):>5s} "
                      f"{_fmt(r['limiar_sugerido']):>5s}  {r['situacao']}")
    return "\n".join(linhas)


def aplicar_limiares_sugeridos(pasta, destino=facs.ARQUIVO_LIMIARES):
    """Mescla os limiares sugeridos em limiares_aus.json e os aplica; retorna os ajustes ativos."""
    with open(os.path.join(pasta, ARQUIVO_LIMIARES_SUGERIDOS), encoding="utf-8") as f:
        sugeridos = json.load(f)
    atuais = {}
    if os.path.exists(destino):
        with open(destino, encoding="utf-8") as f:
            atuais = json.load(f)
    atuais.update(sugeridos)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(atuais, f, indent=2)
    return facs.carregar_limiares(destino)
