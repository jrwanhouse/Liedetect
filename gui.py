"""Interface grafica (PySide6/Qt, licenca LGPL), organizada em projetos.

Cada projeto guarda video, configuracoes, perguntas e resultados (projeto.py). Dois modos
de player, escolhidos em Preferencias:
- "analise": o video e exibido enquanto e analisado, com o audio ajustando a velocidade
  ao ritmo da analise para continuar sincronizado;
- "revisao": a analise roda o mais rapido possivel, exibindo so uma previa.
Nos dois modos, ao fim da analise (ou ao abrir um projeto ja analisado) abre o player de
revisao, com audio em tempo real, play/pausa e navegacao.
"""
import compat  # noqa: F401  (antes de qualquer outro import)
import os
import shutil
import sys
import time

import cv2
from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

import preferencias
import projeto as proj_mod
from analise import executar_analise, resumir_perguntas, separar_resultados
from atencao import AVISO, carregar_perguntas, linhas_resumo, texto_ponto
from facs import COR_ATENCAO, COR_EMOCAO, EMOCOES, desenhar, formatar_tempo

FILTRO_PROJETO = f"Projeto Liedetect (*{proj_mod.EXTENSAO})"
FILTRO_VIDEO = "Videos (*.mp4 *.avi *.mov *.mkv)"
PAPEL_TEMPO = Qt.ItemDataRole.UserRole
PAPEL_PERGUNTA = Qt.ItemDataRole.UserRole + 1


def bgr_qt(bgr):
    b, g, r = bgr
    return QColor(r, g, b)


CORES_EVENTO = {"alerta": QColor("#FF8A80"), "atencao": bgr_qt(COR_ATENCAO),
                "pergunta": QColor("#80D8FF"), "resumo": QColor("#E0E0E0"),
                "destaque": QColor("#FFD180")}


def cor_evento(categoria):
    if categoria in COR_EMOCAO:
        return bgr_qt(COR_EMOCAO[categoria])
    return CORES_EVENTO.get(categoria, QColor("#FFFFFF"))


def descrever_evento(tipo, dado):
    """(texto, categoria de cor, tempo) de um evento da analise ou de uma pergunta."""
    if tipo == "emocao":
        return f"[{formatar_tempo(dado['inicio'])}] {dado['emocao']} — {dado['aus']}", dado["emocao"], dado["inicio"]
    if tipo == "atencao":
        return texto_ponto(dado), "atencao", dado["inicio"]
    if tipo == "alerta":
        t, texto = dado
        return f"[{formatar_tempo(t)}] {texto}", "alerta", t
    t, texto = dado
    return f"[{formatar_tempo(t)}] ❓ Pergunta: {texto or '-'}", "pergunta", t


def para_qimage(frame):
    h, w, ch = frame.shape
    return QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, w, h, ch * w,
                  QImage.Format.Format_RGB888).copy()


class AnaliseThread(QThread):
    frame_signal = Signal(QImage)
    status_signal = Signal(str)
    evento_signal = Signal(str, object)              # tipo, dado (ver descrever_evento)
    linha_tempo_signal = Signal(object, object, float)  # episodios, pontos, posicao (s)
    concluido_signal = Signal(object)

    def __init__(self, video, cfg, pasta, tempo_real):
        super().__init__()
        self.video, self.cfg, self.pasta = video, cfg, pasta
        self.tempo_real = tempo_real
        self._run_flag = True

    def run(self):
        self.status_signal.emit("Analisando... (F2 marca uma pergunta)" if self.tempo_real
                                else "Analisando...")
        try:
            dados = executar_analise(self.video, self.cfg, self.pasta,
                                     ao_evento=self.evento_signal.emit, ao_frame=self._ao_frame,
                                     continuar=lambda: self._run_flag, tempo_real=self.tempo_real)
        except OSError as e:
            self.status_signal.emit(f"Erro: {e}")
            self.concluido_signal.emit(None)
            return
        self.linha_tempo_signal.emit([dict(e) for e in dados["episodios"]],
                                     [dict(p) for p in dados["pontos"]], dados["duracao"])
        self.concluido_signal.emit(dados)

    def _ao_frame(self, frame_idx, total, frame, res, analisador):
        # sem tempo real o frame so serve de previa; exibir todos atrasaria a analise
        if not (self.tempo_real or frame_idx % 10 == 1):
            return
        desenhar(frame, res)
        self.frame_signal.emit(para_qimage(frame))
        self.linha_tempo_signal.emit([dict(e) for e in analisador.episodios],
                                     [dict(p) for p in analisador.atencao.pontos], res.tempo)
        if not self.tempo_real and total:
            self.status_signal.emit(f"Analisando... {100 * frame_idx / total:.0f}%")

    def stop(self):
        self._run_flag = False
        self.wait()


class PlayerAudio:
    """Toca o audio do video. Se o QMediaPlayer falhar (ex.: sem saida de audio),
    a posicao passa a vir de um relogio interno, para o player de revisao continuar."""

    def __init__(self, video, prefs):
        self.saida = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.saida)
        self.aplicar_prefs(prefs)
        try:
            self.player.setPitchCompensation(True)  # velocidade menor sem voz grave
        except AttributeError:
            pass
        self.falhou = False
        self.player.errorOccurred.connect(self._erro)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(video)))
        self._base, self._inicio, self._taxa = 0.0, None, 1.0

    def aplicar_prefs(self, prefs):
        self.saida.setVolume(prefs["volume"] / 100)
        self.saida.setMuted(not prefs["audio"])

    def _erro(self, *_):
        self.falhou = True

    def posicao(self):
        if not self.falhou:
            return self.player.position() / 1000
        if self._inicio is None:
            return self._base
        return self._base + (time.monotonic() - self._inicio) * self._taxa

    def tocar(self):
        if self._inicio is None:
            self._inicio = time.monotonic()
        self.player.play()

    def pausar(self):
        self._base, self._inicio = self.posicao(), None
        self.player.pause()

    def posicionar(self, segundos):
        self._base = segundos
        if self._inicio is not None:
            self._inicio = time.monotonic()
        self.player.setPosition(int(segundos * 1000))

    def taxa(self):
        return self._taxa

    def definir_taxa(self, taxa):
        self._base = self.posicao()
        if self._inicio is not None:
            self._inicio = time.monotonic()
        self._taxa = taxa
        self.player.setPlaybackRate(taxa)

    def parar(self):
        self.player.stop()
        self.player.setSource(QUrl())


class PlayerRevisao(QObject):
    """Exibe o video no relogio do audio, redesenhando as AUs gravadas na analise."""
    frame_signal = Signal(QImage)
    posicao_signal = Signal(float)
    tocando_signal = Signal(bool)

    def __init__(self, video, fps, duracao, resultados, prefs):
        super().__init__()
        self.resultados = resultados
        self.fps = fps
        self.duracao = duracao
        self.cap = cv2.VideoCapture(video)
        self.audio = PlayerAudio(video, prefs)
        self.ultimo = -1
        self.tocando = False
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._mostrar(0.0)

    def posicao(self):
        return max(0, self.ultimo) / self.fps

    def alternar(self):
        if self.tocando:
            self.pausar()
        else:
            if self.posicao() >= self.duracao - 1 / self.fps:
                self.ir_para(0.0)
            self.audio.tocar()
            self.tocando = True
            self.tocando_signal.emit(True)

    def pausar(self):
        self.audio.pausar()
        self.tocando = False
        self.tocando_signal.emit(False)

    def ir_para(self, segundos):
        segundos = min(max(0.0, segundos), self.duracao)
        self.audio.posicionar(segundos)
        self._mostrar(segundos)

    def _tick(self):
        if not self.tocando:
            return
        t = self.audio.posicao()
        if t >= self.duracao:
            self.pausar()
            t = self.duracao
        self._mostrar(t)

    def _mostrar(self, t):
        idx = min(int(t * self.fps), len(self.resultados) - 1)
        if idx < 0 or idx == self.ultimo:
            return
        if self.ultimo < idx <= self.ultimo + 15:
            for _ in range(idx - self.ultimo - 1):  # avancar e mais rapido que buscar
                self.cap.grab()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.ultimo = idx
        res = self.resultados[idx]
        if res.espelhado:
            frame = cv2.flip(frame, 1)
        desenhar(frame, res)
        self.frame_signal.emit(para_qimage(frame))
        self.posicao_signal.emit(idx / self.fps)

    def fechar(self):
        self.timer.stop()
        self.audio.parar()
        self.cap.release()


class LinhaDoTempo(QWidget):
    """Uma faixa por emoção e uma de pontos de atenção, com as perguntas marcadas."""
    ALTURA_FAIXA, LARGURA_ROTULO, TOPO = 18, 90, 14
    clicado = Signal(float)

    def __init__(self):
        super().__init__()
        self.duracao, self.posicao, self.episodios, self.pontos = 0.0, 0.0, [], []
        self.perguntas = []
        self.setMinimumHeight(self.TOPO + self.ALTURA_FAIXA * (len(EMOCOES) + 1) + 4)

    def atualizar(self, episodios, pontos, posicao):
        self.episodios, self.pontos, self.posicao = episodios, pontos, posicao
        self.update()

    def definir_posicao(self, posicao):
        self.posicao = posicao
        self.update()

    def _escala(self):
        return self.width() - self.LARGURA_ROTULO - 5, max(self.duracao, self.posicao, 0.001)

    def mousePressEvent(self, evento):
        largura, duracao = self._escala()
        x = evento.position().x() - self.LARGURA_ROTULO
        if 0 <= x <= largura:
            self.clicado.emit(x / largura * duracao)

    def paintEvent(self, _):
        p = QPainter(self)
        largura, duracao = self._escala()
        x_de = lambda t: self.LARGURA_ROTULO + int(t / duracao * largura)
        faixas = [(e, cor_evento(e), [ep for ep in self.episodios if ep["emocao"] == e]) for e in EMOCOES]
        faixas.append(("Atenção", CORES_EVENTO["atencao"], self.pontos))
        for i, (nome, cor, trechos) in enumerate(faixas):
            y = self.TOPO + i * self.ALTURA_FAIXA
            p.setPen(QColor("#CCCCCC"))
            p.drawText(0, y, self.LARGURA_ROTULO - 5, self.ALTURA_FAIXA - 2,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, nome)
            p.fillRect(self.LARGURA_ROTULO, y + 2, largura, self.ALTURA_FAIXA - 6, QColor("#2A2A2A"))
            for t in trechos:
                x0 = x_de(t["inicio"])
                p.fillRect(x0, y + 2, max(2, x_de(t["fim"]) - x0), self.ALTURA_FAIXA - 6, cor)
        p.setPen(QPen(CORES_EVENTO["pergunta"], 1, Qt.PenStyle.DashLine))
        for n, (t, _) in enumerate(sorted(self.perguntas), 1):
            p.drawLine(x_de(t), self.TOPO, x_de(t), self.height())
            p.drawText(x_de(t) + 2, self.TOPO - 3, f"P{n}")
        p.setPen(QColor("#FFFFFF"))
        p.drawLine(x_de(self.posicao), 0, x_de(self.posicao), self.height())
        p.end()


def _campos_analise(form, webcam, calibracao, recortes, rotulo_webcam):
    """Campos de configuracao de analise, comuns as preferencias e ao projeto."""
    chk_webcam = QCheckBox(rotulo_webcam)
    chk_webcam.setChecked(webcam)
    form.addRow("", chk_webcam)
    spn_calibracao = QDoubleSpinBox()
    spn_calibracao.setRange(0.0, 10.0)
    spn_calibracao.setSingleStep(0.5)
    spn_calibracao.setSuffix(" s")
    spn_calibracao.setValue(calibracao)
    form.addRow("Calibração neutra:", spn_calibracao)
    chk_recortes = QCheckBox("Salvar recorte de 5 s de cada evento")
    chk_recortes.setChecked(recortes)
    form.addRow("", chk_recortes)
    return chk_webcam, spn_calibracao, chk_recortes


def _botoes(dialogo, form):
    botoes = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    botoes.accepted.connect(dialogo.accept)
    botoes.rejected.connect(dialogo.reject)
    form.addRow(botoes)


class DialogoPreferencias(QDialog):
    def __init__(self, prefs, caminho, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferências")
        form = QFormLayout(self)

        self.cmb_modo = QComboBox()
        for chave, nome in preferencias.MODOS_PLAYER.items():
            self.cmb_modo.addItem(nome, chave)
        self.cmb_modo.setCurrentIndex(self.cmb_modo.findData(prefs["modo_player"]))
        form.addRow("Modo do player:", self.cmb_modo)
        self.chk_audio = QCheckBox("Reproduzir áudio")
        self.chk_audio.setChecked(prefs["audio"])
        form.addRow("", self.chk_audio)
        self.sld_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_volume.setRange(0, 100)
        self.sld_volume.setValue(int(prefs["volume"]))
        form.addRow("Volume:", self.sld_volume)

        form.addRow(QLabel("<b>Padrões para novos projetos</b>"))
        self.chk_webcam, self.spn_calibracao, self.chk_recortes = _campos_analise(
            form, prefs["webcam_espelhada"], prefs["calibracao_segundos"], prefs["salvar_recortes"],
            "Vídeo de webcam (espelhado)")

        form.addRow(QLabel("<b>Modo terminal sem projeto</b>"))
        pasta = QHBoxLayout()
        self.txt_pasta = QLineEdit(prefs["pasta_saida"])
        btn_pasta = QPushButton("...")
        btn_pasta.clicked.connect(self._escolher_pasta)
        pasta.addWidget(self.txt_pasta)
        pasta.addWidget(btn_pasta)
        form.addRow("Pasta de saída:", pasta)

        arquivo = QLabel(f"Arquivo: {caminho}")
        arquivo.setStyleSheet("color: #999999; font-size: 11px;")
        form.addRow(arquivo)
        _botoes(self, form)

    def _escolher_pasta(self):
        pasta = QFileDialog.getExistingDirectory(self, "Pasta de saída", self.txt_pasta.text())
        if pasta:
            self.txt_pasta.setText(pasta)

    def valores(self):
        return {
            "modo_player": self.cmb_modo.currentData(),
            "audio": self.chk_audio.isChecked(),
            "volume": self.sld_volume.value(),
            "webcam_espelhada": self.chk_webcam.isChecked(),
            "calibracao_segundos": self.spn_calibracao.value(),
            "pasta_saida": self.txt_pasta.text().strip() or preferencias.PADRAO["pasta_saida"],
            "salvar_recortes": self.chk_recortes.isChecked(),
        }


class DialogoProjeto(QDialog):
    def __init__(self, proj, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configurações do projeto — {proj.nome}")
        form = QFormLayout(self)
        form.addRow("Vídeo:", QLabel(proj.video))
        form.addRow("Saída:", QLabel(proj.pasta_saida()))
        self.chk_webcam, self.spn_calibracao, self.chk_recortes = _campos_analise(
            form, proj.webcam_espelhada, proj.calibracao_segundos, proj.salvar_recortes,
            "Vídeo de webcam (imagem espelhada) — corrigir lados das AUs")
        if proj.analise:
            nota = QLabel("Mudanças valem para a próxima análise (Analisar novamente).")
            nota.setStyleSheet("color: #FFD180;")
            form.addRow(nota)
        _botoes(self, form)

    def valores(self):
        return {"webcam_espelhada": self.chk_webcam.isChecked(),
                "calibracao_segundos": self.spn_calibracao.value(),
                "salvar_recortes": self.chk_recortes.isChecked()}


class MainWindow(QMainWindow):
    def __init__(self, prefs, caminho_prefs):
        super().__init__()
        self.prefs = prefs
        self.caminho_prefs = caminho_prefs
        self.setGeometry(100, 100, 1280, 820)
        self.setStyleSheet("background-color: #121212; color: #FFFFFF;")
        self.projeto = None   # projeto aberto
        self.thread_analise = None    # analise em andamento
        self.audio = None     # audio durante a analise em tempo real
        self.revisao = None   # player de revisao
        self.initUI()
        self._atualizar_titulo()

    def initUI(self):
        menus = {
            "Arquivo": [
                ("Novo projeto...", "Ctrl+N", self.novo_projeto),
                ("Abrir projeto...", "Ctrl+O", self.abrir_projeto),
                ("Salvar projeto", "Ctrl+S", self.salvar_projeto),
                ("Salvar projeto como...", "Ctrl+Shift+S", self.salvar_projeto_como),
                None,
                ("Configurações do projeto...", None, self.configurar_projeto),
                ("Preferências...", "Ctrl+,", self.abrir_preferencias),
                None,
                ("Sair", None, self.close),
            ],
        }
        for titulo, acoes in menus.items():
            menu = self.menuBar().addMenu(titulo)
            for item in acoes:
                if item is None:
                    menu.addSeparator()
                    continue
                nome, atalho, funcao = item
                acao = QAction(nome, self)
                if atalho:
                    acao.setShortcut(QKeySequence(atalho))
                acao.triggered.connect(lambda _=False, f=funcao: f())
                menu.addAction(acao)

        layout = QHBoxLayout()
        left = QVBoxLayout()
        self.lbl_projeto = QLabel()
        self.lbl_projeto.setStyleSheet("color: #BBBBBB;")
        left.addWidget(self.lbl_projeto)
        self.video_label = QLabel("Crie um projeto (Ctrl+N) ou abra um existente (Ctrl+O).")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #1E1E1E; border-radius: 5px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.video_label, stretch=1)

        controles = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.alternar_play)
        self.lbl_tempo = QLabel("00:00 / 00:00")
        controles.addWidget(self.btn_play)
        controles.addWidget(self.lbl_tempo)
        controles.addStretch(1)
        self.lbl_modo = QLabel()
        self.lbl_modo.setStyleSheet("color: #999999;")
        controles.addWidget(self.lbl_modo)
        left.addLayout(controles)
        QShortcut(QKeySequence("Space"), self, self._atalho_play)

        self.linha_tempo = LinhaDoTempo()
        self.linha_tempo.clicado.connect(self.ir_para)
        left.addWidget(self.linha_tempo)

        perguntas = QHBoxLayout()
        self.txt_pergunta = QLineEdit()
        self.txt_pergunta.setPlaceholderText("Texto da pergunta (opcional)")
        self.txt_pergunta.returnPressed.connect(self.marcar_pergunta)
        self.btn_marcar = QPushButton("Marcar pergunta (F2)")
        self.btn_marcar.clicked.connect(self.marcar_pergunta)
        self.btn_importar = QPushButton("Importar perguntas...")
        self.btn_importar.clicked.connect(self.importar_perguntas)
        perguntas.addWidget(self.txt_pergunta, stretch=1)
        perguntas.addWidget(self.btn_marcar)
        perguntas.addWidget(self.btn_importar)
        left.addLayout(perguntas)
        QShortcut(QKeySequence("F2"), self, self.marcar_pergunta)

        self.chk_webcam = QCheckBox("Vídeo de webcam (imagem espelhada) — corrigir lados das AUs")
        self.chk_webcam.toggled.connect(self._webcam_alterada)
        left.addWidget(self.chk_webcam)

        self.btn_analisar = QPushButton("Analisar")
        self.btn_analisar.setStyleSheet("background-color: #007ACC; padding: 10px; font-weight: bold;")
        self.btn_analisar.clicked.connect(self.analisar)
        left.addWidget(self.btn_analisar)

        right = QVBoxLayout()
        self.list_alerts = QListWidget()
        self.list_alerts.setStyleSheet("background-color: #1E1E1E;")
        self.list_alerts.setWordWrap(True)
        self.list_alerts.itemDoubleClicked.connect(self._ir_para_item)
        right.addWidget(self.list_alerts, stretch=2)
        dica = QLabel("Duplo clique: ir ao trecho · Delete: remover pergunta selecionada")
        dica.setStyleSheet("color: #888888; font-size: 11px;")
        right.addWidget(dica)
        apagar = QShortcut(QKeySequence("Delete"), self.list_alerts)
        apagar.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        apagar.activated.connect(self.remover_pergunta)
        right.addWidget(QLabel("Resumo por pergunta"))
        self.list_resumo = QListWidget()
        self.list_resumo.setStyleSheet("background-color: #1E1E1E;")
        self.list_resumo.setWordWrap(True)
        self.list_resumo.itemDoubleClicked.connect(self._ir_para_item)
        right.addWidget(self.list_resumo, stretch=1)
        aviso = QLabel(f"⚠ {AVISO}")
        aviso.setWordWrap(True)
        aviso.setStyleSheet("color: #FFD180; font-size: 11px; padding: 4px;")
        right.addWidget(aviso)

        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=1)
        c = QWidget(); c.setLayout(layout); self.setCentralWidget(c)
        self._atualizar_rotulo_modo()

        self._sync = QTimer(self)
        self._sync.setInterval(250)
        self._sync.timeout.connect(self._sincronizar_audio)
        self._atualizar_controles()

    # --- estado da janela ---
    def _analisando(self):
        return self.thread_analise is not None and self.thread_analise.isRunning()

    def _atualizar_titulo(self):
        titulo = "Detector Forense"
        if self.projeto:
            titulo += f" — {self.projeto.nome}{' *' if self.projeto.modificado else ''}"
            self.lbl_projeto.setText(f"Projeto: {self.projeto.caminho}   |   Vídeo: {self.projeto.video}")
        else:
            self.lbl_projeto.setText("Nenhum projeto aberto")
        self.setWindowTitle(titulo)

    def _atualizar_controles(self):
        tem = self.projeto is not None
        analisando = self._analisando()
        self.btn_analisar.setEnabled(tem)
        self.btn_analisar.setText("■ Parar análise" if analisando else
                                  ("Analisar novamente" if tem and self.projeto.analise else "Analisar"))
        for w in (self.btn_marcar, self.btn_importar, self.txt_pergunta, self.chk_webcam):
            w.setEnabled(tem)
        self.chk_webcam.setEnabled(tem and not analisando)
        self.btn_play.setEnabled(self.revisao is not None)
        if self.revisao is None:
            self.btn_play.setText("▶ Play")

    def _modificado(self):
        if self.projeto:
            self.projeto.modificado = True
            self._atualizar_titulo()

    def _atualizar_rotulo_modo(self):
        self.lbl_modo.setText(f"Player: {preferencias.MODOS_PLAYER[self.prefs['modo_player']]}")

    # --- projeto ---
    def _confirmar_descarte(self):
        """True se pode seguir (salvo, descartado ou sem alteracoes)."""
        if not (self.projeto and self.projeto.modificado):
            return True
        botao = QMessageBox.question(
            self, "Projeto alterado", f"Salvar as alterações do projeto '{self.projeto.nome}'?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if botao == QMessageBox.StandardButton.Save:
            return self.salvar_projeto()
        return botao == QMessageBox.StandardButton.Discard

    def _caminho_projeto(self, titulo, sugestao):
        caminho, _ = QFileDialog.getSaveFileName(self, titulo, sugestao, FILTRO_PROJETO)
        if caminho and not caminho.lower().endswith(proj_mod.EXTENSAO):
            caminho += proj_mod.EXTENSAO
        return caminho

    def novo_projeto(self):
        if not self._confirmar_descarte():
            return
        video, _ = QFileDialog.getOpenFileName(self, "Vídeo do novo projeto", "", FILTRO_VIDEO)
        if not video:
            return
        caminho = self._caminho_projeto("Salvar novo projeto", os.path.splitext(video)[0] + proj_mod.EXTENSAO)
        if not caminho:
            return
        proj = proj_mod.novo(video, self.prefs, caminho)
        try:
            proj_mod.salvar(proj)
        except OSError as e:
            QMessageBox.warning(self, "Novo projeto", f"Não foi possível criar o projeto: {e}")
            return
        self.carregar_projeto(proj)
        self.statusBar().showMessage("Projeto criado. Importe ou marque perguntas se quiser e clique em Analisar.")

    def abrir_projeto(self, caminho=None):
        if not self._confirmar_descarte():
            return
        if not caminho:
            caminho, _ = QFileDialog.getOpenFileName(self, "Abrir projeto", "", FILTRO_PROJETO)
            if not caminho:
                return
        try:
            proj = proj_mod.abrir(caminho)
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, "Abrir projeto", f"Não foi possível abrir '{caminho}':\n{e}")
            return
        if not os.path.exists(proj.video):
            QMessageBox.information(self, "Vídeo não encontrado",
                                    f"O vídeo do projeto não foi encontrado:\n{proj.video}\n\nIndique onde ele está.")
            video, _ = QFileDialog.getOpenFileName(self, "Localizar vídeo", "", FILTRO_VIDEO)
            if video:
                proj.video = os.path.abspath(video)
                proj.modificado = True
        self.carregar_projeto(proj)
        self.statusBar().showMessage(f"Projeto aberto: {proj.caminho}")

    def carregar_projeto(self, proj):
        self._encerrar()
        self.projeto = proj
        self.linha_tempo.perguntas = proj.perguntas
        self.chk_webcam.blockSignals(True)
        self.chk_webcam.setChecked(proj.webcam_espelhada)
        self.chk_webcam.blockSignals(False)
        self.list_alerts.clear()
        self.list_resumo.clear()
        self.video_label.clear()
        self.video_label.setText("Vídeo não encontrado." if not os.path.exists(proj.video) else "")

        if proj.analise:
            a = proj.analise
            self.linha_tempo.duracao = a["duracao"]
            self.linha_tempo.atualizar(a["episodios"], a["pontos"], 0.0)
            self._preencher_eventos()
            self.atualizar_resumo()
            self._abrir_revisao()
        else:
            self.linha_tempo.duracao = self._duracao_video(proj.video)
            self.linha_tempo.atualizar([], [], 0.0)
            for pergunta in sorted(proj.perguntas):
                self.adicionar_evento("pergunta", pergunta)
            self._mostrar_primeiro_frame(proj.video)
        self._atualizar_tempo(0.0)
        self._atualizar_titulo()
        self._atualizar_controles()

    def _duracao_video(self, video):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duracao = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        return max(duracao, 0.0)

    def _mostrar_primeiro_frame(self, video):
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        cap.release()
        if ok:
            if self.projeto.webcam_espelhada:
                frame = cv2.flip(frame, 1)
            self.update_img(para_qimage(frame))

    def _preencher_eventos(self):
        """Reconstroi a lista de eventos a partir da analise salva e das perguntas."""
        a = self.projeto.analise
        eventos = ([("emocao", e) for e in a["episodios"]] + [("atencao", p) for p in a["pontos"]]
                   + [("alerta", tuple(al)) for al in a["alertas"]]
                   + [("pergunta", q) for q in self.projeto.perguntas])
        for tipo, dado in sorted(eventos, key=lambda ev: descrever_evento(*ev)[2]):
            self.adicionar_evento(tipo, dado)

    def salvar_projeto(self):
        if not self.projeto:
            return False
        try:
            proj_mod.salvar(self.projeto)
        except OSError as e:
            QMessageBox.warning(self, "Salvar projeto", f"Não foi possível salvar: {e}")
            return False
        self._atualizar_titulo()
        self.statusBar().showMessage(f"Projeto salvo: {self.projeto.caminho}")
        return True

    def salvar_projeto_como(self):
        if not self.projeto:
            return
        caminho = self._caminho_projeto("Salvar projeto como", self.projeto.caminho)
        if not caminho:
            return
        saida_anterior = self.projeto.pasta_saida()
        self.projeto.caminho = os.path.abspath(caminho)
        if os.path.isdir(saida_anterior) and saida_anterior != self.projeto.pasta_saida():
            try:  # relatorios e recortes acompanham o projeto
                shutil.copytree(saida_anterior, self.projeto.pasta_saida(), dirs_exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, "Salvar projeto como", f"Relatórios e recortes não foram copiados: {e}")
        self.salvar_projeto()

    def configurar_projeto(self):
        if not self.projeto:
            self.statusBar().showMessage("Nenhum projeto aberto.")
            return
        dialogo = DialogoProjeto(self.projeto, self)
        if not dialogo.exec():
            return
        valores = dialogo.valores()
        if valores != self.projeto.configuracoes():
            self.projeto.webcam_espelhada = valores["webcam_espelhada"]
            self.projeto.calibracao_segundos = valores["calibracao_segundos"]
            self.projeto.salvar_recortes = valores["salvar_recortes"]
            self.chk_webcam.blockSignals(True)
            self.chk_webcam.setChecked(self.projeto.webcam_espelhada)
            self.chk_webcam.blockSignals(False)
            self._modificado()

    def _webcam_alterada(self, marcado):
        if self.projeto and self.projeto.webcam_espelhada != marcado:
            self.projeto.webcam_espelhada = marcado
            self._modificado()
            if self.projeto.analise:
                self.statusBar().showMessage("Configuração alterada: vale para a próxima análise (Analisar novamente).")

    # --- preferencias ---
    def abrir_preferencias(self):
        dialogo = DialogoPreferencias(self.prefs, self.caminho_prefs, self)
        if not dialogo.exec():
            return
        self.prefs.update(dialogo.valores())
        try:
            preferencias.salvar(self.prefs, self.caminho_prefs)
            self.statusBar().showMessage(f"Preferências salvas em {self.caminho_prefs}.")
        except OSError as e:
            QMessageBox.warning(self, "Preferências", f"Não foi possível salvar: {e}")
        for player in (self.audio, self.revisao and self.revisao.audio):
            if player:
                player.aplicar_prefs(self.prefs)
        self._atualizar_rotulo_modo()

    # --- analise ---
    def _encerrar(self):
        self._sync.stop()
        if self._analisando():
            self.thread_analise.blockSignals(True)  # o resultado nao pertence mais a janela
            self.thread_analise.stop()
        if self.audio:
            self.audio.parar()
            self.audio = None
        if self.revisao:
            self.revisao.fechar()
            self.revisao = None

    def analisar(self):
        if self._analisando():
            self.thread_analise.stop()  # o resultado parcial chega por analise_concluida
            return
        proj = self.projeto
        if not proj:
            return
        if not os.path.exists(proj.video):
            QMessageBox.warning(self, "Analisar", f"Vídeo não encontrado:\n{proj.video}")
            return
        if proj.analise:
            botao = QMessageBox.question(self, "Analisar novamente",
                                         "Substituir a análise atual do projeto? As perguntas são mantidas.")
            if botao != QMessageBox.StandardButton.Yes:
                return
        self._encerrar()
        self.list_alerts.clear()
        self.list_resumo.clear()
        for pergunta in sorted(proj.perguntas):
            self.adicionar_evento("pergunta", pergunta)
        self.linha_tempo.duracao = self._duracao_video(proj.video)
        self.linha_tempo.atualizar([], [], 0.0)

        tempo_real = self.prefs["modo_player"] == "analise"
        self.thread_analise = AnaliseThread(proj.video, proj.configuracoes(), proj.pasta_saida(), tempo_real)
        self.thread_analise.frame_signal.connect(self.update_img)
        self.thread_analise.evento_signal.connect(self.adicionar_evento)
        self.thread_analise.status_signal.connect(self.statusBar().showMessage)
        self.thread_analise.linha_tempo_signal.connect(self.linha_tempo.atualizar)
        self.thread_analise.concluido_signal.connect(self.analise_concluida)
        if tempo_real:
            self.audio = PlayerAudio(proj.video, self.prefs)
            self._sync_estado = None
            self.thread_analise.linha_tempo_signal.connect(self._iniciar_audio)
        self.thread_analise.start()
        self._atualizar_controles()

    def _atualizar_tempo(self, t):
        self.lbl_tempo.setText(f"{formatar_tempo(t)} / {formatar_tempo(self.linha_tempo.duracao)}")

    def _iniciar_audio(self, *_):
        self.thread_analise.linha_tempo_signal.disconnect(self._iniciar_audio)
        if self.audio:
            self.audio.tocar()
            self._sync.start()

    def _sincronizar_audio(self):
        """Modo analise: a velocidade do audio segue o ritmo em que os frames sao analisados."""
        if not self.audio:
            return
        v, agora = self.linha_tempo.posicao, time.monotonic()
        self._atualizar_tempo(v)
        diferenca = v - self.audio.posicao()  # > 0: audio atrasado
        if abs(diferenca) > 0.6:
            self.audio.posicionar(v)
            diferenca = 0.0
        if self._sync_estado is not None:
            v_ant, t_ant, ritmo = self._sync_estado
            medida = (v - v_ant) / max(agora - t_ant, 1e-3)
            ritmo = 0.5 * ritmo + 0.5 * min(1.0, max(0.25, medida))
            # segue o ritmo da analise e recupera a diferenca em ~2 s
            taxa = min(1.15, max(0.25, ritmo + diferenca / 2))
            if abs(taxa - self.audio.taxa()) > 0.03:
                self.audio.definir_taxa(round(taxa, 2))
        else:
            ritmo = 1.0
        self._sync_estado = (v, agora, ritmo)

    def analise_concluida(self, dados):
        self.thread_analise.wait()  # o sinal sai no fim do run(); garante que a thread terminou
        self._sync.stop()
        if self.audio:
            self.audio.parar()
            self.audio = None
        if dados is None:  # erro ao abrir o video
            self._atualizar_controles()
            return
        proj = self.projeto
        proj.analise, proj.resultados = separar_resultados(dados)
        self.linha_tempo.duracao = proj.analise["duracao"]
        self.atualizar_resumo()
        self._abrir_revisao()
        status = "Análise concluída" if proj.analise["completa"] else "Análise interrompida (resultado parcial)"
        if self.salvar_projeto():  # analise e cara: salva automaticamente
            status += " e salva no projeto"
        self.statusBar().showMessage(f"{status}. Espaço = play/pausa; clique na linha do tempo para navegar.")
        self._atualizar_controles()

    def _abrir_revisao(self):
        proj = self.projeto
        if not (proj and proj.analise and proj.resultados and os.path.exists(proj.video)):
            return
        self.revisao = PlayerRevisao(proj.video, proj.analise["fps"], proj.analise["duracao"],
                                     proj.resultados, self.prefs)
        self.revisao.frame_signal.connect(self.update_img)
        self.revisao.posicao_signal.connect(self.linha_tempo.definir_posicao)
        self.revisao.posicao_signal.connect(self._atualizar_tempo)
        self.revisao.tocando_signal.connect(
            lambda tocando: self.btn_play.setText("⏸ Pausa" if tocando else "▶ Play"))
        self.revisao.ir_para(0.0)

    def alternar_play(self):
        if self.revisao:
            self.revisao.alternar()

    def _atalho_play(self):
        if not self.txt_pergunta.hasFocus():  # Espaco no campo de texto e digitacao
            self.alternar_play()

    def ir_para(self, segundos):
        if self.revisao:
            self.revisao.ir_para(segundos)

    def _ir_para_item(self, item):
        t = item.data(PAPEL_TEMPO)
        if t is not None:
            self.ir_para(t)

    # --- perguntas ---
    def posicao_atual(self):
        if self.revisao:
            return self.revisao.posicao()
        if self._analisando():
            return self.linha_tempo.posicao
        return None

    def marcar_pergunta(self):
        if not self.projeto:
            return
        tempo = self.posicao_atual()
        if tempo is None:
            self.statusBar().showMessage("Marque perguntas durante a análise ou no player de revisão; "
                                         "antes da análise, importe um arquivo de perguntas.")
            return
        pergunta = (tempo, self.txt_pergunta.text().strip())
        self.projeto.perguntas.append(pergunta)
        self.txt_pergunta.clear()
        self.adicionar_evento("pergunta", pergunta)
        self._perguntas_alteradas()

    def importar_perguntas(self):
        if not self.projeto:
            return
        caminho, _ = QFileDialog.getOpenFileName(self, "Importar perguntas", "",
                                                 "Perguntas (*.csv *.txt);;Todos (*)")
        if not caminho:
            return
        try:
            novas = carregar_perguntas(caminho)
        except OSError as e:
            self.statusBar().showMessage(f"Erro ao ler perguntas: {e}")
            return
        self.projeto.perguntas.extend(novas)
        for pergunta in novas:
            self.adicionar_evento("pergunta", pergunta)
        self._perguntas_alteradas()
        self.statusBar().showMessage(f"{len(novas)} pergunta(s) importada(s) de {os.path.basename(caminho)}.")

    def remover_pergunta(self):
        item = self.list_alerts.currentItem()
        pergunta = item.data(PAPEL_PERGUNTA) if item else None
        if not (self.projeto and pergunta):
            return
        pergunta = (pergunta[0], pergunta[1])
        if pergunta in self.projeto.perguntas:
            self.projeto.perguntas.remove(pergunta)
        self.list_alerts.takeItem(self.list_alerts.row(item))
        self._perguntas_alteradas()

    def _perguntas_alteradas(self):
        self.linha_tempo.update()
        self._modificado()
        self.atualizar_resumo()

    def atualizar_resumo(self):
        """Recalcula o resumo por pergunta a partir da analise do projeto."""
        self.list_resumo.clear()
        proj = self.projeto
        if not (proj and proj.analise and proj.perguntas) or self._analisando():
            return
        try:
            resumo = resumir_perguntas(proj.perguntas, proj.analise, proj.pasta_saida())
        except OSError as e:
            self.statusBar().showMessage(f"Erro ao salvar resumo: {e}")
            return
        for (texto, destaque), r in zip(linhas_resumo(resumo), resumo):
            item = QListWidgetItem(texto)
            item.setForeground(cor_evento("destaque" if destaque else "resumo"))
            item.setData(PAPEL_TEMPO, r["inicio"])
            self.list_resumo.addItem(item)

    def adicionar_evento(self, tipo, dado):
        texto, categoria, tempo = descrever_evento(tipo, dado)
        item = QListWidgetItem(texto)
        item.setForeground(cor_evento(categoria))
        item.setData(PAPEL_TEMPO, tempo)
        if tipo == "pergunta":
            item.setData(PAPEL_PERGUNTA, list(dado))
        self.list_alerts.addItem(item)

    def update_img(self, img):
        px = QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(px)

    def closeEvent(self, evento):
        if not self._confirmar_descarte():
            evento.ignore()
            return
        self._encerrar()
        super().closeEvent(evento)


def rodar_modo_grafico(prefs, caminho_prefs, caminho_projeto=None):
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow(prefs, caminho_prefs)
    w.show()
    if caminho_projeto:
        w.abrir_projeto(caminho_projeto)
    sys.exit(app.exec())
