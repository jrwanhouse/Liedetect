"""Detector OpenFace 2.2 (CMU MultiComp Lab): AUs com intensidade 0-5 de modelos treinados em bases FACS.

Roda o FeatureExtraction.exe (programa C++ separado, ~20 frames/s em CPU) e le o CSV que ele
grava. Cada linha vira um Resultado pelo mesmo caminho dos outros detectores. As emocoes saem
das regras EMFACS (o OpenFace nao tem classificador de emocoes nem informa o lado das AUs).

LICENCA: "academic or non-profit organization noncommercial research use only"; proibe
redistribuir o OpenFace ou dar acesso a terceiros. Por isso ele nao vai no repositorio: cada
instalacao baixa o seu (ver README).
"""
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile

from facs import AUS_OPENFACE, CALIBRACAO_SEGUNDOS, OPENFACE
from malha68 import AnalisadorMalha68
from processos import OPCOES_POPEN, encerrar

PASTA = os.path.dirname(os.path.abspath(__file__))
NOTA_OPENFACE = ("OpenFace 2.2: licença CMU para pesquisa não comercial; não redistribua o OpenFace. "
                 "~20 frames/s em CPU.")
CONFIANCA_MINIMA = 0.75
COLUNAS = ["confidence", "success"] + [f"AU{au:02d}_r" for au in AUS_OPENFACE] + ["AU28_c"] + \
          [f"{eixo}_{i}" for i in range(68) for eixo in ("x", "y")]
_PORCENTAGEM = re.compile(r"(\d{1,3})%")


class OpenFaceIndisponivel(RuntimeError):
    pass


class AnaliseInterrompida(RuntimeError):
    pass


def executavel_padrao():
    return os.path.join(PASTA, "openface", "OpenFace_2.2.0_win_x64", "FeatureExtraction.exe")


def localizar_executavel(configurado=""):
    """FeatureExtraction do OpenFace: o configurado nas preferencias ou o da pasta openface/ do programa."""
    for caminho in (configurado, executavel_padrao()):
        if caminho and os.path.exists(caminho):
            return caminho
    return None


def caminho_cache(video, pasta, espelhar):
    info = os.stat(video)
    return os.path.join(pasta, f"openface{'_esp' if espelhar else ''}_{info.st_size}_{int(info.st_mtime)}.csv")


def _desespelhar(video, destino):
    """Copia do video desvirada (o OpenFace nao tem essa opcao)."""
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(destino, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.flip(frame, 1))
    cap.release()
    out.release()


def extrair(video, saida, executavel, espelhar=False, ao_progresso=None, continuar=None):
    """Roda o FeatureExtraction ate gravar `saida` (CSV do OpenFace)."""
    temporaria = tempfile.mkdtemp(prefix="openface_", dir=os.path.dirname(os.path.abspath(saida)))
    try:
        entrada = os.path.abspath(video)
        if espelhar:
            entrada = os.path.join(temporaria, "desespelhado.mp4")
            _desespelhar(video, entrada)
        cmd = [executavel, "-f", entrada, "-out_dir", temporaria, "-of", "openface", "-aus", "-pose", "-2Dfp"]
        if executavel.endswith(".py"):  # testes usam um FeatureExtraction falso em Python
            cmd.insert(0, sys.executable)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **OPCOES_POPEN)
        texto, ultimo = "", -1
        try:
            while True:
                if continuar is not None and not continuar():
                    raise AnaliseInterrompida("analise interrompida durante a extracao do OpenFace")
                pedaco = proc.stdout.read1(256) if hasattr(proc.stdout, "read1") else proc.stdout.read(256)
                if not pedaco:
                    break
                texto = (texto + pedaco.decode("utf-8", "replace"))[-4000:]
                # o progresso chega como "0% 10% 20% ..." numa linha que vai sendo escrita
                for pct in map(int, _PORCENTAGEM.findall(texto[-300:])):
                    if pct > ultimo:
                        ultimo = pct
                        if ao_progresso:
                            ao_progresso(pct, 100)
            proc.wait()
        finally:
            encerrar(proc)
            proc.stdout.close()
        csv_gerado = os.path.join(temporaria, "openface.csv")
        if proc.returncode != 0 or not os.path.exists(csv_gerado):
            detalhe = "\n".join(l for l in texto.splitlines()[-10:] if l.strip())
            raise RuntimeError(f"OpenFace falhou (codigo {proc.returncode})" + (f"\n{detalhe}" if detalhe else ""))
        os.replace(csv_gerado, saida)
    finally:
        shutil.rmtree(temporaria, ignore_errors=True)


def carregar(caminho):
    """{frame (a partir de 0): {coluna: valor}} das linhas com rosto rastreado."""
    linhas = {}
    with open(caminho, encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f, skipinitialspace=True):
            if linha.get("success", "0").strip() != "1":
                continue
            linhas[int(linha["frame"]) - 1] = {c: float(linha[c]) for c in COLUNAS if linha.get(c, "").strip()}
    return linhas


class AnalisadorOpenFace(AnalisadorMalha68):
    """AUs pela intensidade 0-5 do OpenFace; emocoes por EMFACS; olhos e cabeca pela malha (malha68.py)."""
    detector = OPENFACE

    def __init__(self, fps, espelhado, calibracao_segundos=CALIBRACAO_SEGUNDOS, linhas=None):
        super().__init__(fps, espelhado, calibracao_segundos, linhas)

    def _rosto_valido(self, linha):
        return linha.get("confidence", 0) >= CONFIANCA_MINIMA and "x_67" in linha

    def _scores(self, linha):
        s = {au: linha[f"AU{au:02d}_r"] / 5 for au in AUS_OPENFACE if f"AU{au:02d}_r" in linha}
        if "AU28_c" in linha:
            s[28] = linha["AU28_c"]
        return s
