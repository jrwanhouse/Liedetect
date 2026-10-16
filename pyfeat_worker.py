"""Processo auxiliar do Py-Feat; roda no ambiente Python do Py-Feat, separado do programa.

Uso: python pyfeat_worker.py VIDEO SAIDA.csv [--modelo v2|v1] [--pular N] [--espelhar] [--lote N]

Le os frames com OpenCV (o leitor de video do proprio Py-Feat depende de um FFmpeg com DLLs
no Windows), desvira se --espelhar, passa ao Py-Feat como tensores e grava um CSV (;) com uma
linha por frame analisado com rosto: frame, FaceScore, AUxx, emocoes e os 68 pontos (x_i, y_i,
em pixels do frame). Escreve em SAIDA.csv.parcial e renomeia so ao terminar.

Protocolo no stdout: "PROGRESSO feitos total", "PRONTO" ou "ERRO mensagem".
Este arquivo nao importa nada do programa principal (outro Python, outras dependencias).
"""
import argparse
import csv
import os
import sys
import warnings

EMOCOES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
# o Detectorv2 usa outros nomes para as mesmas emocoes
NOMES_V2 = {"Anger": "anger", "Disgust": "disgust", "Fear": "fear", "Happy": "happiness",
            "Sad": "sadness", "Surprise": "surprise", "Neutral": "neutral"}


def avisar(texto):
    print(texto, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("saida")
    ap.add_argument("--modelo", choices=["v2", "v1"], default="v2")
    ap.add_argument("--pular", type=int, default=1, help="analisa 1 a cada N frames")
    ap.add_argument("--espelhar", action="store_true", help="desvira os frames (webcam em modo selfie)")
    ap.add_argument("--lote", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    try:
        import cv2
        import torch
        import feat
    except ImportError as e:
        avisar(f"ERRO dependencia ausente no ambiente do Py-Feat: {e}")
        return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        avisar(f"ERRO nao foi possivel abrir o video '{args.video}'")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a_analisar = (total + args.pular - 1) // args.pular if total > 0 else 0
    avisar(f"PROGRESSO 0 {a_analisar}")

    classe = feat.Detectorv2 if args.modelo == "v2" else feat.Detectorv1
    detector = classe(device="cpu", identity_model=None)

    parcial = args.saida + ".parcial"
    arquivo = open(parcial, "w", newline="", encoding="utf-8")
    escritor = None
    lote, indices, feitos = [], [], 0

    def processar():
        nonlocal escritor, feitos
        if not lote:
            return
        r = detector.detect(torch.stack(lote), data_type="tensor", batch_size=len(lote), progress_bar=False)
        r = r.rename(columns={k: v for k, v in NOMES_V2.items() if k in r.columns})
        if escritor is None:
            colunas = ["FaceScore"] + [c for c in r.columns if c.startswith("AU")] + \
                      [c for c in EMOCOES if c in r.columns] + \
                      [f"{eixo}_{i}" for i in range(68) for eixo in ("x", "y")]
            escritor = csv.writer(arquivo, delimiter=";")
            escritor.writerow(["frame"] + colunas)
            processar.colunas = colunas
        r = r.dropna(subset=["FaceScore"])
        # um rosto por frame: o de maior confianca
        r = r.sort_values("FaceScore", ascending=False).drop_duplicates(subset=["frame"])
        for _, linha in r.sort_values("frame").iterrows():
            escritor.writerow([indices[int(linha["frame"])]] +
                              [f"{float(linha[c]):.5g}" for c in processar.colunas])
        arquivo.flush()
        feitos += len(lote)
        lote.clear()
        indices.clear()
        avisar(f"PROGRESSO {feitos} {a_analisar}")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.pular == 0:
            if args.espelhar:
                frame = cv2.flip(frame, 1)
            lote.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous())
            indices.append(i)
            if len(lote) >= args.lote:
                processar()
        i += 1
    processar()
    cap.release()
    if escritor is None:  # video sem frames
        csv.writer(arquivo, delimiter=";").writerow(["frame", "FaceScore"])
    arquivo.close()
    os.replace(parcial, args.saida)
    avisar("PRONTO")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001  (o programa principal so le o stdout)
        avisar(f"ERRO {type(e).__name__}: {e}")
        sys.exit(1)
