import argparse
import sys
import os
import cv2
import preferencias
from facs import AnalisadorFACS, EMOCOES, formatar_tempo, salvar_relatorio
from atencao import (AVISO, carregar_perguntas, linhas_resumo, resumo_perguntas,
                     salvar_resumo_perguntas, texto_ponto)
from recortes import registrar_corte

def verificar_suporte_grafico():
    """Detecta se há suporte a ambiente gráfico ativo no sistema operacional."""
    if sys.platform != "win32" and "DISPLAY" not in os.environ:
        return False
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        return True
    except Exception:
        return False

# --- MODO TERMINAL (CLI) ---
def rodar_modo_terminal(prefs):
    print("\n" + "="*50)
    print("🖥️  MODO TERMINAL DETECTADO (Sem ambiente grafico ativo)")
    print("="*50)

    video_input = input("Digite o nome ou caminho do video para analisar: ").strip()
    if not os.path.exists(video_input):
        print(f"❌ Erro: O arquivo '{video_input}' nao foi encontrado.")
        return
    padrao_webcam = prefs["webcam_espelhada"]
    resposta = input(f"O video vem de webcam com imagem espelhada? [{'S/n' if padrao_webcam else 's/N'}]: ")
    resposta = resposta.strip().lower()
    espelhado = resposta in ("s", "sim") if resposta else padrao_webcam
    perguntas = []
    arquivo_perguntas = input("Arquivo de perguntas (tempo;texto por linha) - Enter para nao usar: ").strip()
    if arquivo_perguntas:
        try:
            perguntas = carregar_perguntas(arquivo_perguntas)
            print(f"❓ {len(perguntas)} pergunta(s) carregada(s).")
        except OSError as e:
            print(f"❌ Erro ao ler perguntas ({e}); seguindo sem perguntas.")

    print(f"🔍 Inicializando analise computacional em segundo plano...")

    cap = cv2.VideoCapture(video_input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pasta = prefs["pasta_saida"]
    os.makedirs(pasta, exist_ok=True)
    analisador = AnalisadorFACS(fps, espelhado, prefs["calibracao_segundos"])

    def corte(t, motivo):
        if prefs["salvar_recortes"]:
            registrar_corte(video_input, t, motivo, cortes, pasta)

    frame_idx = 0
    alertas, cortes = [], set()
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        res = analisador.processar(frame, frame_idx)
        frame_idx += 1
        timestamp_str = formatar_tempo(res.tempo)

        if frame_idx % int(fps * 5) == 0 and total_frames > 0:
            porcentagem = (frame_idx / total_frames) * 100
            print(f"Progresso: {porcentagem:.1f}% completo... [Analise atual em {timestamp_str}]")

        for ep in res.novos_episodios:
            print(f"🎭 [{timestamp_str}] {ep['emocao']} -> {ep['aus']}")
            corte(res.tempo, f"{ep['emocao']} {ep['aus']}")
        for p in res.novos_pontos:
            print(texto_ponto(p))
            corte(p["inicio"], f"Atencao {p['tipo']}")
        for alerta in res.alertas:
            alertas.append((res.tempo, alerta))
            print(f"⚠️  [HOTSPOT DETECTADO em {timestamp_str}] -> {alerta}")
            corte(res.tempo, alerta)

    cap.release()
    for p in analisador.finalizar():
        print(texto_ponto(p))
    analisador.close()
    pontos = analisador.atencao.pontos

    print("\n📊 Resumo por emocao:")
    for emocao in EMOCOES:
        eps = [e for e in analisador.episodios if e["emocao"] == emocao]
        if eps:
            trechos = ", ".join(f"{formatar_tempo(e['inicio'])}-{formatar_tempo(e['fim'])}" for e in eps)
            print(f"  {emocao} ({len(eps)}x): {trechos}")
    if not analisador.episodios:
        print("  Nenhuma emocao detectada.")

    print("\n◆ Pontos de atencao:")
    for p in sorted(pontos, key=lambda p: p["inicio"]):
        print("  " + texto_ponto(p))
    if not pontos:
        print("  Nenhum.")

    relatorio = os.path.join(pasta, "relatorio_emocoes.csv")
    salvar_relatorio(relatorio, analisador.episodios, alertas, pontos)

    if perguntas:
        resumo = resumo_perguntas(perguntas, frame_idx / fps, analisador.episodios, pontos,
                                  analisador.atencao.tempos_piscadas)
        print("\n❓ Resumo por pergunta:")
        for texto, _ in linhas_resumo(resumo):
            print("  " + texto)
        caminho = os.path.join(pasta, "relatorio_perguntas.csv")
        salvar_resumo_perguntas(caminho, resumo)
        print(f"  Relatorio por pergunta em '{caminho}'.")

    print(f"\n⚠ {AVISO}")
    print(f"✅ Concluido! Amostras em '{pasta}/', relatorio em '{relatorio}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisador de FACS forense (GUI / terminal).")
    parser.add_argument("--config", default=preferencias.ARQUIVO_PADRAO,
                        help="arquivo JSON de preferencias (padrao: preferencias.json na pasta do programa)")
    args = parser.parse_args()
    prefs = preferencias.carregar(args.config)

    if verificar_suporte_grafico():
        from gui import rodar_modo_grafico
        rodar_modo_grafico(prefs, args.config)
    else:
        rodar_modo_terminal(prefs)
