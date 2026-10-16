"""Detector simulado e rosto neutro compartilhados pelos testes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compat  # noqa: E402,F401  (antes de qualquer outro import)
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import facs  # noqa: E402

# Os 52 blendshapes emitidos pelo FaceLandmarker
BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

# Rosto neutro com os vieses tipicos do modelo (ex.: browDown alto mesmo sem franzir)
NEUTRO = dict.fromkeys(BLENDSHAPES, 0.02) | {
    "browDownLeft": 0.35, "browDownRight": 0.33, "eyeSquintLeft": 0.25, "eyeSquintRight": 0.22,
    "eyeBlinkLeft": 0.15, "eyeBlinkRight": 0.15, "jawOpen": 0.05,
    "eyeLookDownLeft": 0.12, "eyeLookDownRight": 0.12,
}

# Pontos usados pela pose (x, y normalizados): cantos dos olhos, bochechas, nariz, queixo
POSE_FRONTAL = {33: (0.40, 0.40), 263: (0.60, 0.40), 234: (0.32, 0.50), 454: (0.68, 0.50),
                1: (0.50, 0.52), 152: (0.50, 0.75)}


def landmarks(pontos=None):
    """478 pontos; os da pose vem de POSE_FRONTAL, sobrescritos por `pontos`."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for idx, (x, y) in (POSE_FRONTAL | (pontos or {})).items():
        lm[idx] = SimpleNamespace(x=x, y=y)
    return lm


def girar(pontos, graus, centro=(0.5, 0.5)):
    """Rotaciona os pontos no plano da imagem (positivo = anti-horario na tela)."""
    a = np.radians(-graus)  # y da imagem cresce para baixo
    cx, cy = centro
    return {i: (cx + (x - cx) * np.cos(a) - (y - cy) * np.sin(a),
                cy + (x - cx) * np.sin(a) + (y - cy) * np.cos(a)) for i, (x, y) in pontos.items()}


class DetectorFalso:
    """Imita o FaceLandmarker: cada item da sequencia e um dict de blendshapes ou (blendshapes, landmarks)."""

    def __init__(self, sequencia):
        self.sequencia = sequencia
        self.padrao = landmarks()
        self.i = 0

    def detect_for_video(self, _imagem, _ms):
        item = self.sequencia[min(self.i, len(self.sequencia) - 1)]
        self.i += 1
        bs, lm = item if isinstance(item, tuple) else (item, self.padrao)
        categorias = [SimpleNamespace(category_name=k, score=v) for k, v in bs.items()]
        return SimpleNamespace(face_landmarks=[lm], face_blendshapes=[categorias])

    def close(self):
        pass


def analisar_sequencia(sequencia, fps=30, espelhado=False):
    """Roda o AnalisadorFACS real sobre a sequencia simulada; retorna (analisador, resultados)."""
    frame = np.zeros((480, 640, 3), np.uint8)
    with mock.patch.object(facs, "criar_detector", lambda: DetectorFalso(sequencia)):
        analisador = facs.AnalisadorFACS(fps, espelhado)
        resultados = [analisador.processar(frame.copy(), i) for i in range(len(sequencia))]
        analisador.finalizar()
    return analisador, resultados


def aus_de(blendshapes=None, pontos=None, frames=1, fps=30):
    """AUs do ultimo frame: 30 frames neutros de calibracao seguidos de `frames` com a expressao."""
    expressao = NEUTRO | (blendshapes or {})
    item = (expressao, landmarks(pontos)) if pontos else expressao
    _, resultados = analisar_sequencia([NEUTRO] * 30 + [item] * frames, fps)
    return resultados[-1].aus


# --- Py-Feat: linhas do CSV do pyfeat_worker.py, com a malha de 68 pontos em pixels (640x480) ---
AUS_PF = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]
EMOCOES_PF = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


def malha68(abertura=(1.0, 1.0), nariz=(0.0, 0.0)):
    """68 pontos de um rosto frontal; abertura = (olho esquerdo, olho direito) do sujeito, 1 = aberto."""
    p = {i: (320.0, 300.0) for i in range(68)}
    p |= {0: (200, 230), 16: (440, 230), 30: (320 + nariz[0], 260 + nariz[1])}
    ae, ad = abertura
    # olho direito do sujeito (lado esquerdo da imagem): 36 externo, 39 interno, 37/38 em cima, 41/40 embaixo
    p |= {36: (230, 200), 39: (270, 200), 37: (243, 200 - 6 * ad), 38: (257, 200 - 6 * ad),
          41: (243, 200 + 6 * ad), 40: (257, 200 + 6 * ad)}
    # olho esquerdo: 42 interno, 45 externo, 43/44 em cima, 47/46 embaixo
    p |= {42: (370, 200), 45: (410, 200), 43: (377, 200 - 6 * ae), 44: (403, 200 - 6 * ae),
          47: (377, 200 + 6 * ae), 46: (403, 200 + 6 * ae)}
    return p


def linha_pyfeat(aus=None, emocoes=None, abertura=(1.0, 1.0), nariz=(0.0, 0.0), rosto=0.99):
    linha = {"FaceScore": rosto} | {f"AU{au:02d}": 0.05 for au in AUS_PF}
    linha |= {f"AU{au:02d}": v for au, v in (aus or {}).items()}
    linha |= {e: 0.01 for e in EMOCOES_PF} | {"neutral": 0.9} | (emocoes or {})
    for i, (x, y) in malha68(abertura, nariz).items():
        linha[f"x_{i}"], linha[f"y_{i}"] = float(x), float(y)
    return linha


NEUTRO_PF = linha_pyfeat()
FELIZ_PF = linha_pyfeat({6: 0.8, 12: 0.9}, {"happiness": 0.95, "neutral": 0.02})


# --- OpenFace: linhas do CSV do FeatureExtraction (intensidade 0-5, malha de 68 pontos em pixels) ---
AUS_OF = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26]


def linha_openface(intensidades=None, au28=0.0, abertura=(1.0, 1.0), nariz=(0.0, 0.0), confianca=0.98, sucesso=1):
    linha = {"confidence": confianca, "success": sucesso} | {f"AU{au:02d}_r": 0.0 for au in AUS_OF}
    linha |= {f"AU{au:02d}_r": v for au, v in (intensidades or {}).items()} | {"AU28_c": au28}
    for i, (x, y) in malha68(abertura, nariz).items():
        linha[f"x_{i}"], linha[f"y_{i}"] = float(x), float(y)
    return linha


NEUTRO_OF = linha_openface()
FELIZ_OF = linha_openface({6: 3.5, 12: 4.2})
