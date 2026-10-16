import argparse
import sys
import os
import preferencias
import projeto as proj_mod
from analise import executar_analise, resumir_perguntas, separar_resultados
from atencao import AVISO, carregar_perguntas, linhas_resumo, texto_ponto
from facs import EMOCOES, formatar_tempo

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
def analisar_no_terminal(video, cfg, pasta, perguntas, guardar_quadros=False):
    """Roda a analise imprimindo eventos, resumos e relatorios; retorna os dados da analise."""
    print(f"🔍 Inicializando analise computacional em segundo plano...")

    def ao_evento(tipo, dado):
        if tipo == "emocao":
            print(f"🎭 [{formatar_tempo(dado['inicio'])}] {dado['emocao']} -> {dado['aus']}")
        elif tipo == "atencao":
            print(texto_ponto(dado))
        else:
            t, texto = dado
            print(f"⚠️  [HOTSPOT DETECTADO em {formatar_tempo(t)}] -> {texto}")

    def ao_frame(frame_idx, total, frame, res, analisador):
        if frame_idx % int(analisador.fps * 5) == 0 and total > 0:
            print(f"Progresso: {frame_idx / total * 100:.1f}% completo... "
                  f"[Analise atual em {formatar_tempo(res.tempo)}]")

    dados = executar_analise(video, cfg, pasta, ao_evento=ao_evento, ao_frame=ao_frame,
                             guardar_quadros=guardar_quadros)

    print("\n📊 Resumo por emocao:")
    for emocao in EMOCOES:
        eps = [e for e in dados["episodios"] if e["emocao"] == emocao]
        if eps:
            trechos = ", ".join(f"{formatar_tempo(e['inicio'])}-{formatar_tempo(e['fim'])}" for e in eps)
            print(f"  {emocao} ({len(eps)}x): {trechos}")
    if not dados["episodios"]:
        print("  Nenhuma emocao detectada.")

    print("\n◆ Pontos de atencao:")
    for p in sorted(dados["pontos"], key=lambda p: p["inicio"]):
        print("  " + texto_ponto(p))
    if not dados["pontos"]:
        print("  Nenhum.")

    if perguntas:
        print("\n❓ Resumo por pergunta:")
        for texto, _ in linhas_resumo(resumir_perguntas(perguntas, dados, pasta)):
            print("  " + texto)
        print(f"  Relatorio por pergunta em '{os.path.join(pasta, 'relatorio_perguntas.csv')}'.")

    print(f"\n⚠ {AVISO}")
    print(f"✅ Concluido! Amostras e relatorio em '{pasta}/'.")
    return dados

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

    cfg = {"webcam_espelhada": espelhado, "calibracao_segundos": prefs["calibracao_segundos"],
           "salvar_recortes": prefs["salvar_recortes"]}
    analisar_no_terminal(video_input, cfg, prefs["pasta_saida"], perguntas)

def analisar_projeto_no_terminal(caminho):
    """Analisa um projeto existente com as configuracoes e perguntas dele e grava o resultado."""
    try:
        proj = proj_mod.abrir(caminho)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"❌ Erro ao abrir o projeto '{caminho}': {e}")
        return 1
    if not os.path.exists(proj.video):
        print(f"❌ Video do projeto nao encontrado: {proj.video}")
        return 1
    print(f"📁 Projeto: {proj.caminho}\n🎬 Video: {proj.video}\n❓ Perguntas: {len(proj.perguntas)}")
    if proj.analise:
        print("ℹ️  A analise anterior do projeto sera substituida.")
    try:
        dados = analisar_no_terminal(proj.video, proj.configuracoes(), proj.pasta_saida(), proj.perguntas,
                                     guardar_quadros=True)
    except OSError as e:
        print(f"❌ {e}")
        return 1
    proj.analise, proj.resultados = separar_resultados(dados)
    proj_mod.salvar(proj)
    print(f"💾 Resultado salvo no projeto '{proj.caminho}'.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisador de FACS forense (GUI / terminal).")
    parser.add_argument("--config", default=preferencias.ARQUIVO_PADRAO,
                        help="arquivo JSON de preferencias (padrao: preferencias.json na pasta do programa)")
    parser.add_argument("--projeto", help=f"projeto ({proj_mod.EXTENSAO}) a abrir na interface grafica "
                                          "ou, no modo terminal, a analisar")
    parser.add_argument("--terminal", action="store_true",
                        help="forca o modo terminal mesmo com ambiente grafico")
    args = parser.parse_args()
    prefs = preferencias.carregar(args.config)

    if not args.terminal and verificar_suporte_grafico():
        from gui import rodar_modo_grafico
        rodar_modo_grafico(prefs, args.config, args.projeto)
    elif args.projeto:
        sys.exit(analisar_projeto_no_terminal(args.projeto))
    else:
        rodar_modo_terminal(prefs)
