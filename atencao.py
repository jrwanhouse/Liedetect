"""Pontos de atencao para revisao humana e resumo por pergunta.

Nada aqui detecta mentira: os pontos sinalizam variacao emocional ou de estresse em
relacao ao proprio sujeito, que podem ter muitas causas (nervosismo, constrangimento,
cansaco, a propria situacao de entrevista).
"""
import csv
from collections import Counter
from statistics import median

AVISO = ("Pontos de atencao indicam variacao emocional ou de estresse para revisao humana. "
         "NAO indicam mentira.")

DURACAO_MICROEXPRESSAO = 0.5   # s
JANELA_PISCADAS = 10.0         # s
JANELA_REACAO = 3.0            # s apos a marcacao da pergunta
MIN_PERGUNTAS_COMPARACAO = 3


class DetectorAtencao:
    def __init__(self, fps):
        self.fps = fps
        self.min_frames = max(3, round(fps * 0.2))
        self.folga_frames = round(fps * 0.3)
        self._estados = {}
        self.pontos = []
        self.tempos_piscadas = []
        self._fim_janela = JANELA_PISCADAS
        self._piscadas_por_janela = []

    def atualizar(self, res, piscou, episodios_fechados):
        """Recebe o Resultado do frame; retorna (novos pontos, descricoes dos pontos ativos)."""
        from facs import rotulo_au
        t = res.tempo
        novos = []
        if piscou:
            self.tempos_piscadas.append(t)

        condicoes = {}
        for num, au in res.aus.items():
            if au.lado and num not in (30, 46):  # mandibula lateral e piscadela sao laterais por natureza
                condicoes[f"assimetria{num}"] = ("Assimetria", f"{rotulo_au(num, au)} so de um lado do rosto")
        if 12 in res.aus and not res.aus[12].lado and 6 not in res.aus:
            condicoes["sorriso"] = ("Sorriso sem AU6",
                                    f"{rotulo_au(12, res.aus[12])} sem AU6: sorriso que nao chega aos olhos")
        novos += self._rastrear(condicoes, t)

        for ep in episodios_fechados:
            duracao = ep["fim"] - ep["inicio"] + 1 / self.fps
            if duracao < DURACAO_MICROEXPRESSAO:
                novos.append(self._registrar("Microexpressao", ep["inicio"], ep["fim"],
                                             f"{ep['emocao']} por {duracao * 1000:.0f} ms ({ep['aus']})"))

        if t >= self._fim_janela:
            novos += self._avaliar_piscadas()

        ativos = [p["descricao"] for e in self._estados.values() if (p := e["ponto"])]
        return novos, ativos

    def finalizar(self, episodios_abertos):
        """Fecha o video: microexpressoes de episodios ainda abertos."""
        novos = []
        for ep in episodios_abertos:
            duracao = ep["fim"] - ep["inicio"] + 1 / self.fps
            if duracao < DURACAO_MICROEXPRESSAO:
                novos.append(self._registrar("Microexpressao", ep["inicio"], ep["fim"],
                                             f"{ep['emocao']} por {duracao * 1000:.0f} ms ({ep['aus']})"))
        return novos

    def _registrar(self, tipo, inicio, fim, descricao):
        ponto = {"tipo": tipo, "inicio": inicio, "fim": fim, "descricao": descricao}
        self.pontos.append(ponto)
        return ponto

    def _rastrear(self, condicoes, t):
        """Condicoes precisam durar min_frames para virar ponto; terminam apos folga_frames ausentes."""
        novos = []
        for chave in set(self._estados) | set(condicoes):
            e = self._estados.setdefault(chave, {"seq": 0, "ausente": 0, "ponto": None, "inicio": t})
            if chave in condicoes:
                e["ausente"] = 0
                e["seq"] += 1
                if e["seq"] == 1:
                    e["inicio"] = t
                if e["ponto"] is not None:
                    e["ponto"]["fim"] = t
                elif e["seq"] >= self.min_frames:
                    tipo, descricao = condicoes[chave]
                    e["ponto"] = self._registrar(tipo, e["inicio"], t, descricao)
                    novos.append(e["ponto"])
            else:
                e["seq"] = 0
                if e["ponto"] is not None:
                    e["ausente"] += 1
                    if e["ausente"] > self.folga_frames:
                        e["ponto"] = None
        return novos

    def _avaliar_piscadas(self):
        """Compara a janela atual com a mediana das janelas anteriores do mesmo video."""
        inicio = self._fim_janela - JANELA_PISCADAS
        n = sum(inicio <= p < self._fim_janela for p in self.tempos_piscadas)
        novos = []
        if len(self._piscadas_por_janela) >= 2:
            padrao = median(self._piscadas_por_janela)
            if n >= max(2 * padrao, padrao + 4):
                novos.append(self._registrar("Piscadas acima do padrao", inicio, self._fim_janela,
                                             f"{n} piscadas em {JANELA_PISCADAS:.0f} s (padrao do sujeito: {padrao:.0f})"))
            elif padrao >= 3 and n <= padrao * 0.3:
                novos.append(self._registrar("Supressao de piscadas", inicio, self._fim_janela,
                                             f"{n} piscadas em {JANELA_PISCADAS:.0f} s (padrao do sujeito: {padrao:.0f})"))
        self._piscadas_por_janela.append(n)
        self._fim_janela += JANELA_PISCADAS
        return novos


def _parse_tempo(texto):
    segundos = 0.0
    for parte in texto.strip().split(":"):
        segundos = segundos * 60 + float(parte.replace(",", "."))
    return segundos


def carregar_perguntas(caminho):
    """Le perguntas de um arquivo com uma por linha: tempo;texto (tempo em s, mm:ss ou hh:mm:ss).

    Linhas que nao comecam com um tempo valido (ex.: cabecalho) sao ignoradas.
    """
    perguntas = []
    with open(caminho, encoding="utf-8-sig") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            sep = next((s for s in (";", "\t", ",") if s in linha), None)
            tempo, _, texto = linha.partition(sep) if sep else (linha, "", "")
            try:
                perguntas.append((_parse_tempo(tempo), texto.strip()))
            except ValueError:
                continue
    return sorted(perguntas)


def resumo_perguntas(perguntas, duracao, episodios, pontos, tempos_piscadas):
    """Agrega emocoes, pontos de atencao e piscadas no trecho de cada pergunta
    (da marcacao ate a proxima) e aponta as que destoam das demais perguntas."""
    perguntas = sorted(perguntas)
    resumo = []
    for i, (inicio, texto) in enumerate(perguntas):
        fim = perguntas[i + 1][0] if i + 1 < len(perguntas) else max(duracao, inicio)
        dur_min = max(fim - inicio, 1e-6) / 60
        no_trecho = lambda t: inicio <= t < fim
        emocoes = Counter(e["emocao"] for e in episodios if no_trecho(e["inicio"]))
        pts = [p for p in pontos if no_trecho(p["inicio"])]
        reacao = sorted({e["emocao"] for e in episodios if inicio <= e["inicio"] < min(fim, inicio + JANELA_REACAO)})
        resumo.append({
            "numero": i + 1, "texto": texto, "inicio": inicio, "fim": fim,
            "emocoes": emocoes, "pontos": Counter(p["tipo"] for p in pts),
            "pontos_min": len(pts) / dur_min,
            "piscadas_min": sum(map(no_trecho, tempos_piscadas)) / dur_min,
            "reacao_imediata": reacao, "observacoes": [], "nota": "",
        })

    if len(resumo) >= MIN_PERGUNTAS_COMPARACAO:
        med_pontos = median(r["pontos_min"] for r in resumo)
        med_pisc = median(r["piscadas_min"] for r in resumo)
        for r in resumo:
            if sum(r["pontos"].values()) >= 2 and r["pontos_min"] > 2 * max(med_pontos, 0.5):
                r["observacoes"].append(f"pontos de atencao acima das demais perguntas "
                                        f"({r['pontos_min']:.1f}/min vs mediana {med_pontos:.1f}/min)")
            if med_pisc > 0 and r["piscadas_min"] > 1.5 * med_pisc:
                r["observacoes"].append(f"piscadas acima das demais ({r['piscadas_min']:.0f}/min vs {med_pisc:.0f}/min)")
            elif med_pisc > 0 and r["piscadas_min"] < 0.5 * med_pisc:
                r["observacoes"].append(f"piscadas abaixo das demais ({r['piscadas_min']:.0f}/min vs {med_pisc:.0f}/min)")
    elif resumo:
        for r in resumo:
            r["nota"] = f"comparacao entre perguntas requer {MIN_PERGUNTAS_COMPARACAO}+ perguntas"
    return resumo


def formatar_contagem(contador):
    return ", ".join(f"{k} ({v}x)" if v > 1 else k for k, v in contador.most_common()) or "-"


def salvar_resumo_perguntas(caminho, resumo):
    from facs import formatar_tempo
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow([f"# {AVISO}"])
        wr.writerow(["pergunta", "texto", "inicio", "fim", "emocoes", "reacao_imediata_3s",
                     "pontos_de_atencao", "pontos_por_min", "piscadas_por_min", "observacoes"])
        for r in resumo:
            wr.writerow([r["numero"], r["texto"], formatar_tempo(r["inicio"]), formatar_tempo(r["fim"]),
                         formatar_contagem(r["emocoes"]), ", ".join(r["reacao_imediata"]) or "-",
                         formatar_contagem(r["pontos"]), f"{r['pontos_min']:.1f}",
                         f"{r['piscadas_min']:.0f}", " | ".join(r["observacoes"]) or r["nota"]])


def texto_ponto(p):
    from facs import formatar_tempo
    return f"[{formatar_tempo(p['inicio'])}] ◆ Atenção — {p['tipo']}: {p['descricao']}"


def linhas_resumo(resumo):
    """Texto do resumo por pergunta: lista de (texto, destoa)."""
    from facs import formatar_tempo
    linhas = []
    for r in resumo:
        titulo = f"Pergunta {r['numero']}" + (f" ({r['texto']})" if r["texto"] else "")
        texto = (f"{titulo} [{formatar_tempo(r['inicio'])}-{formatar_tempo(r['fim'])}]\n"
                 f"   Emoções: {formatar_contagem(r['emocoes'])}"
                 f" | Reação imediata (3 s): {', '.join(r['reacao_imediata']) or '-'}\n"
                 f"   Pontos de atenção: {formatar_contagem(r['pontos'])}"
                 f" | Piscadas: {r['piscadas_min']:.0f}/min")
        if r["observacoes"]:
            texto += "\n   ▲ " + "; ".join(r["observacoes"])
        elif r["nota"]:
            texto += f"\n   ({r['nota']})"
        linhas.append((texto, bool(r["observacoes"])))
    return linhas
