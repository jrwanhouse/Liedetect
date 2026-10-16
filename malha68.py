"""Base dos detectores externos que devolvem a malha de 68 pontos (iBUG): Py-Feat e OpenFace.

Cada detector grava um CSV por frame; o subclasse diz como ler AUs e emocoes de uma linha. Aqui
fica o que e comum: calibracao da pose e da abertura dos olhos, piscadas pela abertura das
palpebras (EAR, Soukupova & Cech 2016), AUs de cabeca pela geometria e limiarizacao.
"""
import bisect

import numpy as np

from facs import AnalisadorBase, CALIBRACAO_SEGUNDOS, MALHA_68, POSE_68, Ponto, _pose, classificar_emocoes, limiar_de

# Olho direito e esquerdo do sujeito: canto externo, 2 superiores, canto interno, 2 inferiores
OLHO_D, OLHO_E = (36, 37, 38, 39, 40, 41), (45, 44, 43, 42, 47, 46)


def abertura_olho(lm, w, h, olho):
    """Razao de aspecto do olho (EAR): cai perto de zero com o olho fechado."""
    p = [np.array([lm[i].x * w, lm[i].y * h]) for i in olho]
    largura = np.linalg.norm(p[0] - p[3])
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * largura) if largura else 0.0


class AnalisadorMalha68(AnalisadorBase):
    """Subclasses definem detector e implementam _rosto_valido, _pontos e _scores (e, se quiserem, _emocoes)."""
    malha = MALHA_68

    def __init__(self, fps, espelhado, calibracao_segundos=CALIBRACAO_SEGUNDOS, linhas=None, pular=1):
        super().__init__(fps, espelhado, calibracao_segundos)
        self.linhas = linhas or {}
        self.frames = sorted(self.linhas)
        self.pular = max(1, pular)
        self._amostras_olhos = []
        self.base_olhos = None

    def _linha(self, frame_idx):
        """Linha do frame; se o detector pulou frames, repete a do ultimo frame analisado."""
        pos = bisect.bisect_right(self.frames, frame_idx) - 1
        if pos >= 0 and frame_idx - self.frames[pos] < self.pular:
            return self.linhas[self.frames[pos]]
        return None

    def _rosto_valido(self, linha):
        raise NotImplementedError

    def _pontos(self, linha, w, h):
        """68 Pontos normalizados (0-1)."""
        return [Ponto(linha[f"x_{i}"] / w, linha[f"y_{i}"] / h) for i in range(68)]

    def _scores(self, linha):
        """{au: score 0-1} das AUs do detector (olhos e cabeca sao medidos aqui pela malha)."""
        raise NotImplementedError

    def _emocoes(self, linha, aus):
        return classificar_emocoes(aus)

    def _medir(self, frame_bgr, frame_idx, res):
        linha = self._linha(frame_idx)
        if not linha or not self._rosto_valido(linha):
            return
        h, w = frame_bgr.shape[:2]
        lm = self._pontos(linha, w, h)
        res.landmarks = lm
        pose = _pose(lm, w, h, POSE_68)
        olhos = np.array([abertura_olho(lm, w, h, OLHO_E), abertura_olho(lm, w, h, OLHO_D)])
        if self.base_pose is None:
            self._amostras_olhos.append(olhos)
            if self._calibrar_pose(pose):
                self.base_olhos = np.median(self._amostras_olhos, axis=0)
            return

        s = self._scores(linha)
        fechamento = np.clip(1 - olhos / np.maximum(self.base_olhos, 1e-6), 0, 1)
        fechado = fechamento >= limiar_de(45, self.detector)
        lados = {}
        if fechado.all():  # o FACS nao codifica AU5/AU7 com a palpebra fechada
            s.pop(5, None)
            s.pop(7, None)
        self._olhos(s, lados, bool(fechado[0]), bool(fechado[1]), float(fechamento[0]), float(fechamento[1]))
        self._cabeca(pose, s)
        res.aus = self._limiarizar(s, lados)
        res.scores = s
        res.emocoes = self._emocoes(linha, res.aus)
