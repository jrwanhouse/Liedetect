"""Recorte de trechos de video em torno dos eventos detectados."""
import os

import cv2

from facs import formatar_tempo


def registrar_corte(video_origem, segundo_alerta, motivo, cortes, pasta_destino):
    """Recorta o trecho do evento, no maximo um corte por segundo de video."""
    if int(segundo_alerta) in cortes:
        return
    cortes.add(int(segundo_alerta))
    recortar_trecho(video_origem, segundo_alerta, formatar_tempo(segundo_alerta), motivo, pasta_destino)


def recortar_trecho(video_origem, segundo_alerta, timestamp_str, motivo, pasta_destino):
    sub_cap = cv2.VideoCapture(video_origem)
    fps = sub_cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(sub_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(sub_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    segundo_inicio = max(0, segundo_alerta - 2.5)
    sub_cap.set(cv2.CAP_PROP_POS_MSEC, segundo_inicio * 1000)

    nome_m = "".join(c if c.isalnum() else "_" for c in motivo)
    caminho_saida = os.path.join(pasta_destino, f"hotspot_{timestamp_str.replace(':', 'm')}s_{nome_m}.mp4")
    out = cv2.VideoWriter(caminho_saida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for _ in range(int(fps * 5)):
        ret, f = sub_cap.read()
        if not ret: break
        out.write(f)
    sub_cap.release(); out.release()
