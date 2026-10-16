"""Imita o pyfeat_worker.py (mesmo protocolo) sem o Py-Feat: 1 s neutro, depois sorriso.

FALSO_MODO=erro faz o processo falhar; FALSO_MODO=lento demora entre os lotes; FALSO_MODO=mudo
fica 2 s sem escrever nada (como o Py-Feat carregando os modelos) e grava o CSV sem avisar.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402

from auxiliares import FELIZ_PF, NEUTRO_PF  # noqa: E402

video, saida = sys.argv[1], sys.argv[2]
pular = int(sys.argv[sys.argv.index("--pular") + 1]) if "--pular" in sys.argv else 1
modo = os.environ.get("FALSO_MODO", "")
if modo == "erro":
    print("ERRO modelo nao encontrado", flush=True)
    sys.exit(1)
total = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
frames = list(range(0, total, pular))
print(f"PROGRESSO 0 {len(frames)}", flush=True)
if modo == "mudo":
    time.sleep(2)
    with open(saida, "w", encoding="utf-8") as f:
        f.write("frame;FaceScore\n")
    sys.exit(0)
colunas = list(NEUTRO_PF)
with open(saida + ".parcial", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["frame"] + colunas)
    for n, i in enumerate(frames, 1):
        linha = NEUTRO_PF if i < 30 else FELIZ_PF
        w.writerow([i] + [linha[c] for c in colunas])
        if n % 10 == 0 or n == len(frames):
            if modo == "lento":
                time.sleep(0.3)
            print(f"PROGRESSO {n} {len(frames)}", flush=True)
os.replace(saida + ".parcial", saida)
print("PRONTO", flush=True)
