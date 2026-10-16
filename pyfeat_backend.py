"""Detector Py-Feat: alternativa ao MediaPipe para AUs e emocoes.

O Py-Feat precisa de PyTorch, que nao existe para o Python do programa; ele roda num ambiente
Python separado (.venv-pyfeat, ver README) por meio de pyfeat_worker.py. Este modulo dispara
esse processo, le o CSV que ele grava e transforma cada linha num Resultado, que segue o mesmo
caminho do MediaPipe (episodios, atencao, perguntas, revisao, relatorios, validacao).

LICENCA: o codigo do Py-Feat e MIT, mas os pesos dos modelos tem termos proprios (bases de
treino academicas; img2pose CC BY-NC). Nao liberado para uso comercial sem analise juridica.
"""
import bisect
import csv
import os
import subprocess
import sys
from collections import deque

import numpy as np

from facs import (AUS_PYFEAT, AnalisadorBase, CALIBRACAO_SEGUNDOS, EMOCOES, MALHA_68, POSE_68, PYFEAT,
                  Ponto, _pose, classificar_emocoes, limiar_de)

PASTA = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(PASTA, "pyfeat_worker.py")
AVISO_LICENCA = ("Py-Feat: os pesos dos modelos nao estao liberados para uso comercial "
                 "(bases de treino academicas; img2pose CC BY-NC). Use para pesquisa e comparacao.")

EMOCOES_PYFEAT = {"anger": "Raiva", "disgust": "Nojo", "fear": "Medo", "happiness": "Felicidade",
                  "sadness": "Tristeza", "surprise": "Surpresa"}
MODOS_EMOCAO = {"modelo": "Classificador de emoções do Py-Feat", "emfacs": "Regras EMFACS sobre as AUs do Py-Feat"}
MODELOS = {"v2": "Detectorv2 (multitarefa)", "v1": "Detectorv1 (XGBoost)"}
LIMIAR_ROSTO = 0.5
LIMIAR_EMOCAO = 0.5
# Olho direito e esquerdo do sujeito na malha de 68: canto externo, 2 superiores, canto interno, 2 inferiores
OLHO_D, OLHO_E = (36, 37, 38, 39, 40, 41), (45, 44, 43, 42, 47, 46)


class PyFeatIndisponivel(RuntimeError):
    pass


class AnaliseInterrompida(RuntimeError):
    pass


def python_padrao():
    if sys.platform == "win32":
        return os.path.join(PASTA, ".venv-pyfeat", "Scripts", "python.exe")
    return os.path.join(PASTA, ".venv-pyfeat", "bin", "python")


def localizar_python(configurado=""):
    """Python do ambiente do Py-Feat: o configurado nas preferencias ou o .venv-pyfeat do programa."""
    for caminho in (configurado, python_padrao()):
        if caminho and os.path.exists(caminho):
            return caminho
    return None


def caminho_cache(video, pasta, modelo, pular, espelhar):
    """CSV da extracao; o nome muda se o video ou as opcoes mudarem, entao reanalisar reaproveita."""
    info = os.stat(video)
    nome = f"pyfeat_{modelo}_p{pular}{'_esp' if espelhar else ''}_{info.st_size}_{int(info.st_mtime)}.csv"
    return os.path.join(pasta, nome)


def extrair(video, saida, python, modelo="v2", pular=1, espelhar=False, ao_progresso=None, continuar=None):
    """Roda o pyfeat_worker.py no ambiente do Py-Feat ate gravar `saida`."""
    cmd = [python, "-u", WORKER, video, saida, "--modelo", modelo, "--pular", str(pular)]
    if espelhar:
        cmd.append("--espelhar")
    opcoes = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"), **opcoes)
    ultimas, erro = deque(maxlen=15), None
    try:
        for linha in proc.stdout:
            if continuar is not None and not continuar():
                raise AnaliseInterrompida("analise interrompida durante a extracao do Py-Feat")
            linha = linha.strip()
            if linha.startswith("PROGRESSO "):
                feitos, total = map(int, linha.split()[1:3])
                if ao_progresso:
                    ao_progresso(feitos, total)
            elif linha.startswith("ERRO "):
                erro = linha[5:]
            elif linha:
                ultimas.append(linha)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
    if proc.returncode != 0 or erro or not os.path.exists(saida):
        detalhe = "\n".join(ultimas)
        raise RuntimeError(f"Py-Feat falhou: {erro or f'codigo {proc.returncode}'}" + (f"\n{detalhe}" if detalhe else ""))


def carregar(caminho):
    """{frame: {coluna: valor}} do CSV gravado pelo worker."""
    with open(caminho, encoding="utf-8", newline="") as f:
        return {int(l["frame"]): {k: float(v) for k, v in l.items() if k != "frame" and v not in ("", "nan")}
                for l in csv.DictReader(f, delimiter=";")}


def _abertura(lm, w, h, olho):
    """Razao de aspecto do olho (EAR, Soukupova & Cech 2016): cai perto de zero com o olho fechado."""
    p = [np.array([lm[i].x * w, lm[i].y * h]) for i in olho]
    largura = np.linalg.norm(p[0] - p[3])
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * largura) if largura else 0.0


class AnalisadorPyFeat(AnalisadorBase):
    """AUs e emocoes pelas linhas do Py-Feat; cabeca e piscadas pela malha de 68 pontos."""
    detector = PYFEAT
    malha = MALHA_68

    def __init__(self, fps, espelhado, calibracao_segundos=CALIBRACAO_SEGUNDOS, linhas=None,
                 pular=1, modo_emocoes="modelo"):
        super().__init__(fps, espelhado, calibracao_segundos)
        self.linhas = linhas or {}
        self.frames = sorted(self.linhas)
        self.pular = max(1, pular)
        self.modo_emocoes = modo_emocoes
        self._amostras_olhos = []
        self.base_olhos = None

    def _linha(self, frame_idx):
        """Linha do frame; com --pular, repete a do ultimo frame analisado."""
        pos = bisect.bisect_right(self.frames, frame_idx) - 1
        if pos >= 0 and frame_idx - self.frames[pos] < self.pular:
            return self.linhas[self.frames[pos]]
        return None

    def _medir(self, frame_bgr, frame_idx, res):
        linha = self._linha(frame_idx)
        if not linha or linha.get("FaceScore", 0) < LIMIAR_ROSTO or "x_67" not in linha:
            return
        h, w = frame_bgr.shape[:2]
        lm = [Ponto(linha[f"x_{i}"] / w, linha[f"y_{i}"] / h) for i in range(68)]
        res.landmarks = lm
        pose = _pose(lm, w, h, POSE_68)
        olhos = np.array([_abertura(lm, w, h, OLHO_E), _abertura(lm, w, h, OLHO_D)])
        if self.base_pose is None:
            self._amostras_olhos.append(olhos)
            if self._calibrar_pose(pose):
                self.base_olhos = np.median(self._amostras_olhos, axis=0)
            return

        s = {au: linha[f"AU{au:02d}"] for au in AUS_PYFEAT if f"AU{au:02d}" in linha}
        # o AU43 do modelo dispara em sorriso com olhos apertados; fechamento vem da abertura medida
        s.pop(43, None)
        fechamento = np.clip(1 - olhos / np.maximum(self.base_olhos, 1e-6), 0, 1)
        lim = limiar_de(45, PYFEAT)
        fechado = fechamento >= lim
        lados = {}
        if fechado.all():  # o FACS nao codifica AU5/AU7 com a palpebra fechada
            s.pop(5, None)
            s.pop(7, None)
        self._olhos(s, lados, bool(fechado[0]), bool(fechado[1]), float(fechamento[0]), float(fechamento[1]))
        self._cabeca(pose, s)
        res.aus = self._limiarizar(s, lados)
        res.scores = s
        res.emocoes = self._emocoes(linha, res.aus)

    def _emocoes(self, linha, aus):
        if self.modo_emocoes == "emfacs":
            return classificar_emocoes(aus)
        probabilidades = {nome: linha.get(chave, 0.0) for chave, nome in EMOCOES_PYFEAT.items()}
        nome, p = max(probabilidades.items(), key=lambda kv: kv[1])
        if p < LIMIAR_EMOCAO or p <= linha.get("neutral", 0.0):
            return {}
        # AUs ativas que pertencem a alguma combinacao EMFACS da emocao (contexto para o laudo)
        explicam = {t for combo in EMOCOES[nome] for t in combo if isinstance(t, int)} & set(aus)
        return {nome: explicam}
