"""Projeto de analise: video, configuracoes, perguntas e resultados.

O arquivo do projeto (.liedetect) e um JSON. Relatorios, recortes e os resultados por
frame (quadros.json.gz, usados pelo player de revisao) ficam na pasta <projeto>_saida.
"""
import gzip
import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from facs import AU, Ponto, Resultado

EXTENSAO = ".liedetect"
VERSAO = 1
ARQUIVO_QUADROS = "quadros.json.gz"


@dataclass
class Projeto:
    video: str                     # caminho absoluto
    webcam_espelhada: bool = False
    calibracao_segundos: float = 1.0
    salvar_recortes: bool = True
    detector: str = "mediapipe"     # "mediapipe" ou "pyfeat"
    emocoes_pyfeat: str = "modelo"  # "modelo" (classificador do Py-Feat) ou "emfacs"
    pyfeat_modelo: str = "v2"
    pyfeat_pular: int = 1           # Py-Feat: analisa 1 a cada N frames
    perguntas: list = field(default_factory=list)  # (tempo s, texto)
    analise: dict | None = None    # fps, duracao, episodios, pontos, tempos_piscadas, alertas, data
    resultados: list | None = None # Resultado compacto por frame (so em memoria; gravado a parte)
    caminho: str | None = None
    modificado: bool = False

    @property
    def nome(self):
        if self.caminho:
            return os.path.splitext(os.path.basename(self.caminho))[0]
        return os.path.splitext(os.path.basename(self.video))[0]

    def pasta_saida(self):
        return os.path.splitext(self.caminho)[0] + "_saida"

    def configuracoes(self):
        return {"webcam_espelhada": self.webcam_espelhada, "calibracao_segundos": self.calibracao_segundos,
                "salvar_recortes": self.salvar_recortes, "detector": self.detector,
                "emocoes_pyfeat": self.emocoes_pyfeat, "pyfeat_modelo": self.pyfeat_modelo,
                "pyfeat_pular": self.pyfeat_pular}


def novo(video, prefs, caminho):
    """Projeto novo com as configuracoes padrao das preferencias."""
    return Projeto(video=os.path.abspath(video), webcam_espelhada=prefs["webcam_espelhada"],
                   calibracao_segundos=prefs["calibracao_segundos"],
                   salvar_recortes=prefs["salvar_recortes"], detector=prefs["detector"],
                   emocoes_pyfeat=prefs["emocoes_pyfeat"], pyfeat_modelo=prefs["pyfeat_modelo"],
                   pyfeat_pular=prefs["pyfeat_pular"], caminho=os.path.abspath(caminho))


def _caminho_video_relativo(video, pasta_projeto):
    try:
        return os.path.relpath(video, pasta_projeto)
    except ValueError:  # outra unidade no Windows
        return video


def salvar(proj, caminho=None):
    """Grava o projeto (e os resultados por frame, se houver). Escrita atomica."""
    if caminho:
        proj.caminho = os.path.abspath(caminho)
    pasta_projeto = os.path.dirname(proj.caminho)
    dados = {
        "versao": VERSAO,
        "salvo_em": datetime.now().isoformat(timespec="seconds"),
        "video": _caminho_video_relativo(proj.video, pasta_projeto),
        "video_absoluto": proj.video,
        "configuracoes": proj.configuracoes(),
        "perguntas": [{"tempo": t, "texto": texto} for t, texto in sorted(proj.perguntas)],
        "analise": proj.analise,
    }
    _gravar_atomico(proj.caminho, json.dumps(dados, ensure_ascii=False, indent=2).encode("utf-8"))
    if proj.resultados is not None:
        os.makedirs(proj.pasta_saida(), exist_ok=True)
        quadros = json.dumps([_resultado_para_dict(r) for r in proj.resultados], separators=(",", ":"))
        _gravar_atomico(os.path.join(proj.pasta_saida(), ARQUIVO_QUADROS),
                        gzip.compress(quadros.encode("utf-8"), compresslevel=6))
    proj.modificado = False


def abrir(caminho):
    """Le um projeto. Se o video nao estiver no caminho relativo, tenta o absoluto gravado."""
    caminho = os.path.abspath(caminho)
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    if dados.get("versao", 0) > VERSAO:
        raise ValueError(f"projeto criado por versao mais nova do sistema ({dados['versao']})")
    pasta_projeto = os.path.dirname(caminho)
    video = os.path.normpath(os.path.join(pasta_projeto, dados["video"]))
    if not os.path.exists(video) and os.path.exists(dados.get("video_absoluto", "")):
        video = dados["video_absoluto"]
    cfg = dados.get("configuracoes", {})
    proj = Projeto(
        video=video,
        webcam_espelhada=bool(cfg.get("webcam_espelhada", False)),
        calibracao_segundos=float(cfg.get("calibracao_segundos", 1.0)),
        salvar_recortes=bool(cfg.get("salvar_recortes", True)),
        detector=cfg.get("detector", "mediapipe"),  # projetos antigos: MediaPipe
        emocoes_pyfeat=cfg.get("emocoes_pyfeat", "modelo"),
        pyfeat_modelo=cfg.get("pyfeat_modelo", "v2"),
        pyfeat_pular=max(1, int(cfg.get("pyfeat_pular", 1))),
        perguntas=[(float(p["tempo"]), p.get("texto", "")) for p in dados.get("perguntas", [])],
        analise=dados.get("analise"),
        caminho=caminho,
    )
    arquivo_quadros = os.path.join(proj.pasta_saida(), ARQUIVO_QUADROS)
    if proj.analise and os.path.exists(arquivo_quadros):
        with gzip.open(arquivo_quadros, "rt", encoding="utf-8") as f:
            proj.resultados = [_dict_para_resultado(d) for d in json.load(f)]
    return proj


def _gravar_atomico(caminho, conteudo):
    temporario = caminho + ".tmp"
    with open(temporario, "wb") as f:
        f.write(conteudo)
    os.replace(temporario, caminho)


def _resultado_para_dict(r):
    return {
        "t": round(r.tempo, 4), "c": r.calibrando, "e": r.espelhado,
        "aus": {str(n): [round(a.score, 3), a.intensidade, a.lado] for n, a in r.aus.items()},
        "sc": {str(n): round(v, 3) for n, v in r.scores.items() if v > 0.005},
        "emo": {e: sorted(map(str, c)) for e, c in r.emocoes.items()},
        "lm": None if r.landmarks is None else {str(i): [round(p.x, 5), round(p.y, 5)]
                                                 for i, p in r.landmarks.items()},
        "pa": r.pontos_ativos,
        "m": r.malha,
    }


def _token(t):
    return int(t) if t.isdigit() else t


def _dict_para_resultado(d):
    return Resultado(
        tempo=d["t"], calibrando=d["c"], espelhado=d["e"],
        landmarks=None if d["lm"] is None else {int(i): Ponto(*xy) for i, xy in d["lm"].items()},
        aus={int(n): AU(*v) for n, v in d["aus"].items()},
        scores={int(n): v for n, v in d.get("sc", {}).items()},  # projetos antigos nao tem
        emocoes={e: {_token(t) for t in c} for e, c in d["emo"].items()},
        pontos_ativos=d["pa"],
        malha=d.get("m", "mp478"),
    )
