"""Laco de analise de um video, compartilhado pela interface grafica e pelo terminal."""
import os
import time
from datetime import datetime

import cv2

from atencao import resumo_perguntas, salvar_resumo_perguntas
import facs
from facs import AnalisadorFACS, compactar, salvar_relatorio
from recortes import registrar_corte


def executar_analise(video, cfg, pasta, ao_evento=None, ao_frame=None, continuar=None,
                     tempo_real=False, guardar_quadros=True):
    """Analisa o video e grava relatorio_emocoes.csv (e recortes, se configurado) em `pasta`.

    cfg: webcam_espelhada, calibracao_segundos, salvar_recortes.
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
    analisador = AnalisadorFACS(fps, cfg["webcam_espelhada"], cfg["calibracao_segundos"])
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
        "limiares_ajustados": {str(au): v for au, v in facs.LIMIARES_AJUSTADOS.items()},
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
