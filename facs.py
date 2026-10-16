"""Deteccao de Action Units (FACS) e de emocoes (EMFACS) com o MediaPipe FaceLandmarker.

As AUs faciais vem dos 52 blendshapes do modelo; as AUs de cabeca (51-58) da geometria
dos landmarks. Os primeiros CALIBRACAO_SEGUNDOS de rosto visivel definem a linha de base
neutra do sujeito, que e descontada de todos os valores.
"""
import csv
import math
import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELO_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

CALIBRACAO_SEGUNDOS = 1.0

# AU -> (nome, blendshape base, tem lado esquerdo/direito)
AUS_BLENDSHAPE = {
    1:  ("Elevador interno da sobrancelha", "browInnerUp", False),
    2:  ("Elevador externo da sobrancelha", "browOuterUp", True),
    4:  ("Abaixador da sobrancelha", "browDown", True),
    5:  ("Elevador da palpebra superior", "eyeWide", True),
    6:  ("Elevador da bochecha", "cheekSquint", True),
    7:  ("Tensor da palpebra", "eyeSquint", True),
    9:  ("Franzidor do nariz", "noseSneer", True),
    10: ("Elevador do labio superior", "mouthUpperUp", True),
    12: ("Puxador do canto do labio", "mouthSmile", True),
    14: ("Covinha", "mouthDimple", True),
    15: ("Depressor do canto do labio", "mouthFrown", True),
    16: ("Depressor do labio inferior", "mouthLowerDown", True),
    17: ("Elevador do queixo", "mouthShrugLower", False),
    18: ("Franzidor dos labios (bico)", "mouthPucker", False),
    20: ("Estirador dos labios", "mouthStretch", True),
    22: ("Afunilador dos labios", "mouthFunnel", False),
    23: ("Tensor/pressor dos labios", "mouthPress", True),
    29: ("Projecao da mandibula", "jawForward", False),
    30: ("Mandibula lateral", "jaw", True),
    34: ("Inflar bochechas", "cheekPuff", False),
}
AUS_DERIVADAS = {
    25: "Labios separados",
    26: "Queda da mandibula",
    27: "Estiramento da boca",
    28: "Sugar os labios",
    43: "Olhos fechados",
    45: "Piscada",
    46: "Piscadela",
    51: "Cabeca vira a esquerda",
    52: "Cabeca vira a direita",
    53: "Cabeca para cima",
    54: "Cabeca para baixo",
    55: "Cabeca inclina a esquerda",
    56: "Cabeca inclina a direita",
    57: "Cabeca para frente",
    58: "Cabeca para tras",
    61: "Olhos a esquerda",
    62: "Olhos a direita",
    63: "Olhos para cima",
    64: "Olhos para baixo",
}
NOMES_AU = {au: spec[0] for au, spec in AUS_BLENDSHAPE.items()} | AUS_DERIVADAS

# AUs do FACS sem sinal correspondente no modelo
NAO_DETECTAVEIS = {
    8: "Labios um em direcao ao outro", 11: "Aprofundador nasolabial", 13: "Puxador agudo do labio",
    19: "Mostrar a lingua (o modelo nao emite tongueOut)",
    21: "Tensor do pescoco", 24: "Pressor dos labios (indistinguivel da AU23; reportada como AU23/24)",
    31: "Apertar a mandibula", 32: "Morder o labio", 33: "Soprar", 35: "Sugar as bochechas",
    36: "Protuberancia da lingua", 37: "Limpar os labios", 38: "Dilatar narinas", 39: "Comprimir narinas",
    41: "Palpebra caida", 42: "Fenda palpebral", 44: "Estreitar os olhos",
}

LIMIAR_PADRAO = 0.3
LIMIARES = {25: 0.15, 26: 0.25, 27: 0.65, 43: 0.5, 45: 0.5, 46: 0.5}
LIMIARES |= {au: 0.4 for au in (51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64)}

# Landmarks onde cada AU aparece; AUs bilaterais listam [lado direito do sujeito, lado esquerdo]
PONTOS_AU = {
    1: [107, 336], 2: [70, 300], 4: [9], 5: [159, 386], 6: [205, 425], 7: [145, 374],
    9: [198, 420], 10: [37, 267], 12: [61, 291], 14: [61, 291], 15: [61, 291], 16: [84, 314],
    17: [152], 18: [13], 20: [61, 291], 22: [13], 23: [13], 25: [14], 26: [152],
    27: [152], 28: [14], 29: [152], 30: [172, 397], 34: [205, 425], 43: [159, 386],
    45: [159, 386], 46: [159, 386], 51: [1], 52: [1], 53: [1], 54: [1], 55: [1], 56: [1],
    57: [1], 58: [1], 61: [468, 473], 62: [468, 473], 63: [468, 473], 64: [468, 473],
}

# Combinacoes EMFACS (prototipos e variantes principais de Ekman & Friesen).
# "U12"/"U14" = AU12/AU14 unilateral.
EMOCOES = {
    "Felicidade": [{6, 12}],
    "Tristeza":   [{1, 4, 15}, {1, 4, 15, 17}, {6, 15}],
    "Surpresa":   [{1, 2, 5, 26}, {1, 2, 5, 27}, {1, 2, 26}, {1, 2, 5}],
    "Medo":       [{1, 2, 4, 5, 20, 26}, {1, 2, 4, 5, 20}, {1, 2, 4, 5}, {1, 2, 4, 20}, {5, 20}],
    "Raiva":      [{4, 5, 7, 23}, {4, 5, 23}, {4, 7, 23}, {4, 5, 7}],
    "Nojo":       [{9}, {9, 15, 16}, {9, 17}, {10, 17}, {10, 16, 25}, {10, 15}],
    "Desprezo":   [{"U12"}, {"U14"}],
}
EXCLUSOES = {"Nojo": {12}, "Desprezo": {6}}

COR_EMOCAO = {  # BGR
    "Felicidade": (0, 215, 255),
    "Tristeza":   (220, 120, 40),
    "Surpresa":   (230, 220, 0),
    "Medo":       (200, 60, 170),
    "Raiva":      (40, 40, 230),
    "Nojo":       (60, 180, 60),
    "Desprezo":   (0, 140, 255),
}
COR_AU_NEUTRA = (200, 200, 200)


@dataclass
class AU:
    score: float
    intensidade: str  # A (traco) .. E (maxima)
    lado: str = ""    # "L"/"R" quando unilateral


@dataclass
class Resultado:
    tempo: float
    calibrando: bool
    espelhado: bool = False
    landmarks: list | None = None
    aus: dict = field(default_factory=dict)       # num -> AU
    emocoes: dict = field(default_factory=dict)   # emocao -> combinacao (set)
    novos_episodios: list = field(default_factory=list)
    alertas: list = field(default_factory=list)


def criar_detector():
    """Cria o FaceLandmarker com blendshapes, baixando o modelo na primeira execucao."""
    if not os.path.exists(MODELO_PATH):
        print("Baixando modelo face_landmarker.task...")
        urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
    opcoes = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.7,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(opcoes)


def intensidade(score, limiar):
    passo = (1 - limiar) / 5
    return "ABCDE"[min(4, int((score - limiar) / passo))]


def num_au(token):
    return int(str(token).lstrip("U"))


def rotulo_au(num, au):
    nome = "AU23/24" if num == 23 else f"AU{num}"
    return f"{nome}{au.intensidade}" + (f"({au.lado})" if au.lado else "")


def rotulo_combinacao(combo, aus):
    nums = sorted({num_au(t) for t in combo})
    return "+".join(rotulo_au(n, aus[n]) for n in nums)


def classificar_emocoes(aus):
    """Retorna {emocao: combinacao} para cada emocao cuja combinacao de AUs esta ativa."""
    tokens = set(aus) | {f"U{n}" for n in (12, 14) if n in aus and aus[n].lado}
    achadas = {}
    for emocao, combos in EMOCOES.items():
        if EXCLUSOES.get(emocao, set()) & tokens:
            continue
        melhor = max((c for c in combos if c <= tokens), key=len, default=None)
        if melhor:
            achadas[emocao] = melhor
    # Descarta emocao cuja combinacao esta contida na de outra (ex.: 1+2+5 surpresa dentro de 1+2+4+5 medo)
    return {e: c for e, c in achadas.items()
            if not any(c < o for k, o in achadas.items() if k != e)}


def _pose(lm, w, h):
    p = lambda i: np.array([lm[i].x * w, lm[i].y * h])
    olho_d, olho_e = p(33), p(263)
    bochecha_d, bochecha_e = p(234), p(454)
    nariz, queixo = p(1), p(152)
    largura = np.linalg.norm(bochecha_e - bochecha_d)
    yaw = (nariz[0] - (bochecha_d[0] + bochecha_e[0]) / 2) / largura
    roll = math.degrees(math.atan2(olho_e[1] - olho_d[1], olho_e[0] - olho_d[0]))
    meio_olhos = (olho_d + olho_e) / 2
    pitch = (nariz[1] - meio_olhos[1]) / (queixo[1] - meio_olhos[1])
    return np.array([yaw, pitch, roll, largura])


def _escala(delta, limite):
    """Converte um desvio de pose em score 0..1; vale 0.4 (limiar) quando delta == limite."""
    return min(1.0, abs(delta) / (limite * 2.5))


class AnalisadorFACS:
    def __init__(self, fps, espelhado=False):
        """espelhado=True para video de webcam em modo selfie (imagem invertida horizontalmente)."""
        self.fps = fps
        self.espelhado = espelhado
        self.detector = criar_detector()
        self.frames_calibracao = max(1, int(fps * CALIBRACAO_SEGUNDOS))
        self._amostras_bs, self._amostras_pose = [], []
        self.base_bs = self.base_pose = None
        self.min_frames_emocao = max(2, round(fps * 0.07))
        self.folga_frames = round(fps * 0.5)
        self.janela_piscadas = int(fps * 5)
        self._frames_fechado = 0
        self.piscadas = 0
        self._rastreio = {e: {"seq": 0, "ausente": 0, "ep": None} for e in EMOCOES}
        self.episodios = []

    def close(self):
        self.detector.close()

    def processar(self, frame_bgr, frame_idx):
        """Analisa um frame (frame_idx comeca em 0).

        Em video espelhado o frame e desespelhado no proprio array antes da analise, para que
        lados (L/R), olhar e movimentos de cabeca fiquem no referencial real do sujeito.
        """
        if self.espelhado:
            frame_bgr[:] = cv2.flip(frame_bgr, 1)
        tempo = frame_idx / self.fps
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                           int(tempo * 1000))
        res = Resultado(tempo=tempo, calibrando=self.base_bs is None, espelhado=self.espelhado)

        if r.face_landmarks:
            lm = r.face_landmarks[0]
            bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
            pose = _pose(lm, w, h)
            res.landmarks = lm
            if self.base_bs is None:
                self._calibrar(bs, pose)
            else:
                res.aus = self._calcular_aus(bs, pose)
                res.emocoes = classificar_emocoes(res.aus)

        res.novos_episodios = self._atualizar_episodios(res, tempo)
        if frame_idx > 0 and frame_idx % self.janela_piscadas == 0:
            if self.piscadas > 6:
                res.alertas.append("Frequencia Critica de Piscadas (Estresse)")
            self.piscadas = 0
        return res

    def _calibrar(self, bs, pose):
        self._amostras_bs.append(bs)
        self._amostras_pose.append(pose)
        if len(self._amostras_bs) >= self.frames_calibracao:
            self.base_bs = {k: float(np.median([a[k] for a in self._amostras_bs])) for k in bs}
            self.base_pose = np.median(self._amostras_pose, axis=0)

    def _calcular_aus(self, bs_bruto, pose):
        bs = {k: max(0.0, (v - self.base_bs[k]) / (1 - self.base_bs[k])) if self.base_bs[k] < 1 else 0.0
              for k, v in bs_bruto.items()}
        s, lados = {}, {}
        for au, (_, base, bilateral) in AUS_BLENDSHAPE.items():
            if bilateral:
                lados[au] = (bs[base + "Left"], bs[base + "Right"])
                s[au] = max(lados[au])
            else:
                s[au] = bs[base]

        # O MediaPipe quase nao ativa cheekSquint: o estreitamento do olho (eyeSquint) conta
        # como AU6 durante o sorriso (Duchenne) e como AU7 fora dele.
        if s[12] >= LIMIAR_PADRAO:
            lados[6] = tuple(max(a, b) for a, b in zip(lados[6], lados[7]))
            s[6], s[7], lados[7] = max(lados[6]), 0.0, (0.0, 0.0)

        jaw = bs["jawOpen"]
        s[25] = max(jaw, s[16])
        s[26] = jaw if jaw < LIMIARES[27] else 0.0
        s[27] = jaw if jaw >= LIMIARES[27] else 0.0
        s[28] = max(bs["mouthRollUpper"], bs["mouthRollLower"])

        pisca_e, pisca_d = bs["eyeBlinkLeft"], bs["eyeBlinkRight"]
        olhos_fechados = min(pisca_e, pisca_d) >= LIMIARES[45]
        if olhos_fechados:
            if self._frames_fechado == 0:
                self.piscadas += 1
            self._frames_fechado += 1
            s[43 if self._frames_fechado > self.fps * 0.5 else 45] = min(pisca_e, pisca_d)
        else:
            self._frames_fechado = 0
            lados[46] = (pisca_e, pisca_d)
            s[46] = max(pisca_e, pisca_d)
            # olhar so e confiavel com olhos abertos
            s[61] = (bs["eyeLookOutLeft"] + bs["eyeLookInRight"]) / 2
            s[62] = (bs["eyeLookInLeft"] + bs["eyeLookOutRight"]) / 2
            s[63] = (bs["eyeLookUpLeft"] + bs["eyeLookUpRight"]) / 2
            s[64] = (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"]) / 2

        d_yaw, d_pitch, d_roll = pose[:3] - self.base_pose[:3]
        razao_largura = pose[3] / self.base_pose[3]
        s[51 if d_yaw > 0 else 52] = _escala(d_yaw, 0.08)
        s[53 if d_pitch < 0 else 54] = _escala(d_pitch, 0.06)
        s[55 if d_roll > 0 else 56] = _escala(d_roll, 8.0)
        if abs(d_yaw) < 0.08:
            if razao_largura > 1:
                s[57] = _escala(razao_largura - 1, 0.15)
            else:
                s[58] = _escala(1 - razao_largura, 0.13)

        aus = {}
        for num, score in s.items():
            limiar = LIMIARES.get(num, LIMIAR_PADRAO)
            if score < limiar:
                continue
            lado = ""
            if num in lados:
                e, d = lados[num]
                if e >= limiar and d < limiar * 0.5:
                    lado = "L"
                elif d >= limiar and e < limiar * 0.5:
                    lado = "R"
            if num == 46 and not lado:
                continue  # os dois olhos meio fechados nao e piscadela
            aus[num] = AU(score, intensidade(score, limiar), lado)
        return aus

    def _atualizar_episodios(self, res, tempo):
        novos = []
        for emocao, r in self._rastreio.items():
            if emocao in res.emocoes:
                r["ausente"] = 0
                r["seq"] += 1
                if r["seq"] == 1:
                    r["inicio"] = tempo
                    r["aus"] = rotulo_combinacao(res.emocoes[emocao], res.aus)
                if r["ep"] is not None:
                    r["ep"]["fim"] = tempo
                elif r["seq"] >= self.min_frames_emocao:
                    r["ep"] = {"emocao": emocao, "inicio": r["inicio"], "fim": tempo, "aus": r["aus"]}
                    self.episodios.append(r["ep"])
                    novos.append(r["ep"])
            else:
                r["seq"] = 0
                if r["ep"] is not None:
                    r["ausente"] += 1
                    if r["ausente"] > self.folga_frames:
                        r["ep"] = None
        return novos


def formatar_tempo(segundos):
    return f"{int(segundos) // 60:02d}:{int(segundos) % 60:02d}"


def desenhar(frame, res):
    """Marca no frame as AUs ativas (na cor da emocao que integram) e lista as emocoes."""
    h, w = frame.shape[:2]
    escala = max(0.45, w / 1600)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    linhas = [(f"Tempo: {formatar_tempo(res.tempo)}", (255, 255, 255))]
    if res.espelhado:
        linhas.append(("Webcam: espelhamento corrigido", (255, 255, 255)))
    if res.calibrando:
        linhas.append(("Calibrando - mantenha o rosto neutro", (0, 200, 255)))

    if res.landmarks is not None:
        cor_au = {}
        for emocao, combo in res.emocoes.items():
            for t in combo:
                cor_au[num_au(t)] = COR_EMOCAO[emocao]
        for num, au in sorted(res.aus.items()):
            pontos = PONTOS_AU[num]
            if au.lado and len(pontos) == 2:
                pontos = [pontos[0] if au.lado == "R" else pontos[1]]
            cor = cor_au.get(num, COR_AU_NEUTRA)
            for i, idx in enumerate(pontos):
                x, y = int(res.landmarks[idx].x * w), int(res.landmarks[idx].y * h)
                cv2.circle(frame, (x, y), max(3, int(4 * escala)), cor, -1)
                if i == 0:
                    cv2.putText(frame, f"{num}{au.intensidade}", (x + 6, y - 6), fonte,
                                escala * 0.7, cor, 1, cv2.LINE_AA)
        for emocao, combo in res.emocoes.items():
            linhas.append((f"{emocao}: {rotulo_combinacao(combo, res.aus)}", COR_EMOCAO[emocao]))
        if res.aus:
            linhas.append(("AUs: " + " ".join(rotulo_au(n, a) for n, a in sorted(res.aus.items())),
                           COR_AU_NEUTRA))

    altura_linha = int(30 * escala * 1.4)
    for i, (texto, cor) in enumerate(linhas):
        pos = (15, 15 + altura_linha * (i + 1))
        cv2.putText(frame, texto, pos, fonte, escala, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, texto, pos, fonte, escala, cor, 1, cv2.LINE_AA)


def salvar_relatorio(caminho, episodios, alertas):
    """Grava CSV (separador ;) com os episodios de emocao e os alertas (tempo, texto)."""
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["tipo", "inicio", "fim", "duracao_s", "descricao", "aus"])
        for ep in sorted(episodios, key=lambda e: e["inicio"]):
            wr.writerow(["emocao", formatar_tempo(ep["inicio"]), formatar_tempo(ep["fim"]),
                         f"{ep['fim'] - ep['inicio']:.2f}", ep["emocao"], ep["aus"]])
        for tempo, texto in alertas:
            wr.writerow(["alerta", formatar_tempo(tempo), "", "", texto, ""])
