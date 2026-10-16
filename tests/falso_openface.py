"""Imita o FeatureExtraction.exe do OpenFace (argumentos, progresso e CSV) sem o OpenFace.

1 s neutro, depois sorriso; o frame 45 sai sem rosto (success 0). FALSO_MODO=erro falha;
FALSO_MODO=lento demora entre os marcos de progresso. Grava em FALSO_ENTRADA o -f recebido.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402

from auxiliares import FELIZ_OF, NEUTRO_OF  # noqa: E402

args = sys.argv[1:]
video, pasta, nome = (args[args.index(o) + 1] for o in ("-f", "-out_dir", "-of"))
if os.environ.get("FALSO_ENTRADA"):
    open(os.environ["FALSO_ENTRADA"], "w").write(video)
if os.environ.get("FALSO_MODO") == "erro":
    print("Failed to open the video file at location: " + video, flush=True)
    sys.exit(1)
total = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
colunas = list(NEUTRO_OF)
with open(os.path.join(pasta, nome + ".csv"), "w", encoding="utf-8") as f:
    f.write(", ".join(["frame", "face_id", "timestamp"] + colunas) + "\n")
    for i in range(total):
        linha = dict(NEUTRO_OF if i < 30 else FELIZ_OF)
        if i == 45:
            linha["success"] = 0
        f.write(", ".join([str(i + 1), "0", f"{i / 30:.3f}"] + [str(linha[c]) for c in colunas]) + "\n")
print("Starting tracking", flush=True)
for pct in range(0, 101, 10):
    print(f"{pct}% ", end="", flush=True)
    if os.environ.get("FALSO_MODO") == "lento":
        time.sleep(0.3)
print("\nClosed successfully", flush=True)
