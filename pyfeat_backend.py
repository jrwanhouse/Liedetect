"""Detector Py-Feat: alternativa ao MediaPipe para AUs e emocoes.

O Py-Feat precisa de PyTorch, que nao existe para o Python do programa; ele roda num ambiente
Python separado (.venv-pyfeat, ver README) por meio de pyfeat_worker.py. Este modulo dispara
esse processo, le o CSV que ele grava e transforma cada linha num Resultado, que segue o mesmo
caminho do MediaPipe (episodios, atencao, perguntas, revisao, relatorios, validacao).

LICENCA: o codigo do Py-Feat e MIT; os pesos dos modelos tem termos proprios (bases de treino
academicas; img2pose CC BY-NC) que permitem uso nao comercial, que e o caso deste sistema.
Se o uso passar a ser comercial, o Py-Feat precisa ser reavaliado.
"""
import csv
import os
import subprocess
import sys
from collections import deque


from facs import AUS_PYFEAT, CALIBRACAO_SEGUNDOS, EMOCOES, PYFEAT, classificar_emocoes
from malha68 import AnalisadorMalha68
from processos import OPCOES_POPEN, encerrar

PASTA = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(PASTA, "pyfeat_worker.py")
NOTA_PYFEAT = ("Py-Feat: modelos de uso nao comercial (bases academicas; img2pose CC BY-NC). "
               "Em CPU leva ~2-3 s por frame.")

EMOCOES_PYFEAT = {"anger": "Raiva", "disgust": "Nojo", "fear": "Medo", "happiness": "Felicidade",
                  "sadness": "Tristeza", "surprise": "Surpresa"}
MODOS_EMOCAO = {"modelo": "Classificador de emoções do Py-Feat", "emfacs": "Regras EMFACS sobre as AUs do Py-Feat"}
MODELOS = {"v2": "Detectorv2 (multitarefa)", "v1": "Detectorv1 (XGBoost)"}
LIMIAR_ROSTO = 0.5
LIMIAR_EMOCAO = 0.5


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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"), **OPCOES_POPEN)
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
        encerrar(proc)
        proc.stdout.close()
    if proc.returncode != 0 or erro or not os.path.exists(saida):
        detalhe = "\n".join(ultimas)
        raise RuntimeError(f"Py-Feat falhou: {erro or f'codigo {proc.returncode}'}" + (f"\n{detalhe}" if detalhe else ""))


def carregar(caminho):
    """{frame: {coluna: valor}} do CSV gravado pelo worker."""
    with open(caminho, encoding="utf-8", newline="") as f:
        return {int(l["frame"]): {k: float(v) for k, v in l.items() if k != "frame" and v not in ("", "nan")}
                for l in csv.DictReader(f, delimiter=";")}


class AnalisadorPyFeat(AnalisadorMalha68):
    """AUs e emocoes pelas linhas do Py-Feat; cabeca e piscadas pela malha de 68 pontos (malha68.py)."""
    detector = PYFEAT

    def __init__(self, fps, espelhado, calibracao_segundos=CALIBRACAO_SEGUNDOS, linhas=None,
                 pular=1, modo_emocoes="modelo"):
        super().__init__(fps, espelhado, calibracao_segundos, linhas, pular)
        self.modo_emocoes = modo_emocoes

    def _rosto_valido(self, linha):
        return linha.get("FaceScore", 0) >= LIMIAR_ROSTO and "x_67" in linha

    def _scores(self, linha):
        s = {au: linha[f"AU{au:02d}"] for au in AUS_PYFEAT if f"AU{au:02d}" in linha}
        # o AU43 do modelo dispara em sorriso com olhos apertados; o fechamento vem da abertura medida
        s.pop(43, None)
        return s

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
