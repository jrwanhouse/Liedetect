"""Preferencias do sistema, gravadas em JSON e compartilhadas pelos modos GUI e terminal."""
import json
import os

ARQUIVO_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preferencias.json")

MODOS_PLAYER = {
    "analise": "Análise com áudio em tempo real",
    "revisao": "Revisão após a análise (player com áudio e navegação)",
}

PADRAO = {
    "modo_player": "analise",      # so GUI
    "audio": True,                 # so GUI
    "volume": 80,                  # so GUI, 0-100
    "webcam_espelhada": False,
    "calibracao_segundos": 1.0,
    "pasta_saida": "trechos_suspeitos",
    "salvar_recortes": True,
    "detector": "mediapipe",       # padrao para novos projetos
    "emocoes_pyfeat": "modelo",
    "pyfeat_modelo": "v2",
    "pyfeat_pular": 1,
    "pyfeat_python": "",           # vazio: .venv-pyfeat na pasta do programa
}

DETECTORES = {
    "mediapipe": "MediaPipe (padrão, licença Apache 2.0)",
    "pyfeat": "Py-Feat (pesquisa: pesos não liberados para uso comercial; lento em CPU)",
}
OPCOES = {"detector": DETECTORES, "emocoes_pyfeat": {"modelo", "emfacs"}, "pyfeat_modelo": {"v2", "v1"}}


def carregar(caminho=ARQUIVO_PADRAO):
    """Le as preferencias; chaves ausentes ou invalidas assumem o valor padrao."""
    prefs = dict(PADRAO)
    if not os.path.exists(caminho):
        return prefs
    try:
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Aviso: preferencias em '{caminho}' ignoradas ({e}).")
        return prefs
    for chave, padrao in PADRAO.items():
        valor = dados.get(chave, padrao)
        if isinstance(padrao, bool):
            valido = isinstance(valor, bool)
        elif isinstance(padrao, (int, float)):
            valido = isinstance(valor, (int, float)) and not isinstance(valor, bool) and valor >= 0
        else:
            valido = isinstance(valor, str) and (valor or chave == "pyfeat_python")
        if chave == "modo_player":
            valido = valor in MODOS_PLAYER
        if chave in OPCOES:
            valido = valor in OPCOES[chave]
        if chave == "pyfeat_pular":
            valido = isinstance(valor, int) and not isinstance(valor, bool) and valor >= 1
        prefs[chave] = valor if valido else padrao
    return prefs


def salvar(prefs, caminho=ARQUIVO_PADRAO):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump({k: prefs[k] for k in PADRAO}, f, ensure_ascii=False, indent=2)
