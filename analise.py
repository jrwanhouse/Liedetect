"""Laco de analise de um video, compartilhado pela interface grafica e pelo terminal."""
import os
import time
from datetime import datetime

import cv2

from atencao import resumo_perguntas, salvar_resumo_perguntas
import facs
from facs import AnalisadorFACS, compactar, salvar_relatorio
from recortes import registrar_corte


def criar_analisador(video, cfg, pasta, fps, ao_etapa=None, continuar=None):
    """Analisador do detector configurado; o Py-Feat extrai antes as AUs do video inteiro."""
    detector = cfg.get("detector", facs.MEDIAPIPE)
    avisar = ao_etapa or (lambda texto: None)
    if detector == facs.OPENFACE:
        return _analisador_openface(video, cfg, pasta, fps, avisar, continuar)
    if detector != facs.PYFEAT:
        return AnalisadorFACS(fps, cfg["webcam_espelhada"], cfg["calibracao_segundos"])
    import pyfeat_backend as pf
    python = pf.localizar_python(cfg.get("pyfeat_python", ""))
    if not python:
        raise pf.PyFeatIndisponivel("ambiente do Py-Feat nao encontrado (.venv-pyfeat); veja o README")
    modelo, pular = cfg.get("pyfeat_modelo", "v2"), max(1, int(cfg.get("pyfeat_pular", 1)))
    saida = pf.caminho_cache(video, pasta, modelo, pular, cfg["webcam_espelhada"])
    if os.path.exists(saida):
        avisar("Py-Feat: reaproveitando a extracao anterior deste video")
    else:
        inicio = time.monotonic()

        def progresso(feitos, total):
            texto = f"Py-Feat: {feitos}/{total} frames"
            if feitos:
                restante = (time.monotonic() - inicio) / feitos * (total - feitos)
                texto += f" (faltam ~{restante / 60:.0f} min)"
            avisar(texto)
        avisar("Py-Feat: carregando modelos...")
        pf.extrair(video, saida, python, modelo, pular, cfg["webcam_espelhada"], progresso, continuar)
    return pf.AnalisadorPyFeat(fps, cfg["webcam_espelhada"], cfg["calibracao_segundos"], pf.carregar(saida),
                               pular, cfg.get("emocoes_pyfeat", "modelo"))


def _analisador_openface(video, cfg, pasta, fps, avisar, continuar):
    import openface_backend as of
    executavel = of.localizar_executavel(cfg.get("openface_exe", ""))
    if not executavel:
        raise of.OpenFaceIndisponivel("OpenFace nao encontrado (FeatureExtraction.exe); veja o README")
    saida = of.caminho_cache(video, pasta, cfg["webcam_espelhada"])
    if os.path.exists(saida):
        avisar("OpenFace: reaproveitando a extracao anterior deste video")
    else:
        avisar("OpenFace: carregando modelos...")
        of.extrair(video, saida, executavel, cfg["webcam_espelhada"],
                   lambda pct, total: avisar(f"OpenFace: {pct}%"), continuar)
    return of.AnalisadorOpenFace(fps, cfg["webcam_espelhada"], cfg["calibracao_segundos"], of.carregar(saida))


def executar_analise(video, cfg, pasta, ao_evento=None, ao_frame=None, continuar=None,
                     tempo_real=False, guardar_quadros=True, ao_etapa=None):
    """Analisa o video e grava relatorio_emocoes.csv (e recortes, se configurado) em `pasta`.

    cfg: webcam_espelhada, calibracao_segundos, salvar_recortes e, opcionais, detector
         ("mediapipe"/"pyfeat"/"openface"), emocoes_pyfeat, pyfeat_modelo, pyfeat_pular,
         pyfeat_python, openface_exe.
    ao_etapa(texto): avisos de etapas longas (extracao do Py-Feat ou do OpenFace).
    ao_evento(tipo, dado): tipo "emocao" (episodio), "atencao" (ponto) ou "alerta" ((tempo, texto)).
    ao_frame(frame_idx, total, frame, res, analisador): chamado a cada frame analisado.
    continuar(): retornar False interrompe a analise.
    tempo_real: limita o ritmo a velocidade normal do video (exibicao ao vivo).
    Retorna o dicionario da analise (o que vai para o projeto) com "resultados" por frame.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise OSError(f"nao foi possivel abrir o video '{video}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(pasta, exist_ok=True)
    try:
        analisador = criar_analisador(video, cfg, pasta, fps, ao_etapa, continuar)
    except BaseException:
        cap.release()
        raise
    avisar = ao_evento or (lambda tipo, dado: None)
    alertas, cortes, resultados = [], set(), []

    def corte(t, motivo):
        if cfg["salvar_recortes"]:
            registrar_corte(video, t, motivo, cortes, pasta)

    frame_idx = 0
    inicio_relogio = time.monotonic()
    while continuar is None or continuar():
        success, frame = cap.read()
        if not success: break

        res = analisador.processar(frame, frame_idx)
        frame_idx += 1
        for ep in res.novos_episodios:
            avisar("emocao", ep)
            corte(res.tempo, f"{ep['emocao']} {ep['aus']}")
        for p in res.novos_pontos:
            avisar("atencao", p)
            corte(p["inicio"], f"Atencao {p['tipo']}")
        for alerta in res.alertas:
            alertas.append((res.tempo, alerta))
            avisar("alerta", (res.tempo, alerta))
            corte(res.tempo, alerta)
        if guardar_quadros:
            resultados.append(compactar(res))
        if ao_frame:
            ao_frame(frame_idx, total, frame, res, analisador)

        if tempo_real:
            # acompanha o tempo real; se o processamento atrasar, segue sem pausa
            atraso = frame_idx / fps - (time.monotonic() - inicio_relogio)
            if atraso > 0:
                time.sleep(atraso)

    cap.release()
    for p in analisador.finalizar():
        avisar("atencao", p)
    analisador.close()
    salvar_relatorio(os.path.join(pasta, "relatorio_emocoes.csv"), analisador.episodios, alertas,
                     analisador.atencao.pontos)
    return {
        "data": datetime.now().isoformat(timespec="seconds"),
        "fps": fps, "duracao": frame_idx / fps, "completa": total <= 0 or frame_idx >= total,
        "episodios": analisador.episodios, "pontos": analisador.atencao.pontos,
        "tempos_piscadas": analisador.atencao.tempos_piscadas,
        "alertas": [list(a) for a in alertas],
        "detector": analisador.detector,
        "opcoes_detector": {k: cfg[k] for k in ("emocoes_pyfeat", "pyfeat_modelo", "pyfeat_pular")
                            if analisador.detector == facs.PYFEAT and k in cfg},
        "limiares_ajustados": {str(au): v for au, v in
                               facs.LIMIARES_AJUSTADOS_POR_DETECTOR.get(analisador.detector, {}).items()},
        "resultados": resultados if guardar_quadros else None,
    }


def resumir_perguntas(perguntas, analise, pasta):
    """Resumo por pergunta de uma analise concluida; grava relatorio_perguntas.csv."""
    resumo = resumo_perguntas(list(perguntas), analise["duracao"], analise["episodios"],
                              analise["pontos"], analise["tempos_piscadas"])
    salvar_resumo_perguntas(os.path.join(pasta, "relatorio_perguntas.csv"), resumo)
    return resumo


def separar_resultados(dados):
    """Divide o retorno de executar_analise em (analise para o projeto, resultados por frame)."""
    analise = {k: v for k, v in dados.items() if k != "resultados"}
    return analise, dados["resultados"]
