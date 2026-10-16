"""Interface grafica (PyQt6).

Dois modos de player, escolhidos em Preferencias:
- "analise": o video e exibido enquanto e analisado, com o audio ajustando a velocidade
  ao ritmo da analise para continuar sincronizado;
- "revisao": a analise roda primeiro, o mais rapido possivel, e depois abre um player com
  audio em tempo real, play/pausa e navegacao, redesenhando as AUs a partir dos resultados.
"""
import os
import sys
import time

import cv2
from PyQt6.QtCore import QObject, QThread, QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

import preferencias
from atencao import (AVISO, carregar_perguntas, linhas_resumo, resumo_perguntas,
                     salvar_resumo_perguntas, texto_ponto)
from facs import (AnalisadorFACS, COR_ATENCAO, COR_EMOCAO, EMOCOES, compactar, desenhar,
                  formatar_tempo, salvar_relatorio)
from recortes import registrar_corte


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


def para_qimage(frame):
    h, w, ch = frame.shape
    return QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, w, h, ch * w,
                  QImage.Format.Format_RGB888).copy()


class AnaliseThread(QThread):
    frame_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)
    evento_signal = pyqtSignal(str, str, float)        # texto, categoria, tempo (s)
    duracao_signal = pyqtSignal(float)
    linha_tempo_signal = pyqtSignal(object, object, float)  # episodios, pontos, posicao (s)
    concluido_signal = pyqtSignal(object)

    def __init__(self, video_path, prefs, espelhado, tempo_real):
        super().__init__()
        self.video_path = video_path
        self.prefs = prefs
        self.espelhado = espelhado
        self.tempo_real = tempo_real  # False: analisa sem pausas e guarda os resultados para revisao
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status_signal.emit("Erro ao abrir o vídeo.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duracao_signal.emit(total / fps)
        pasta = self.prefs["pasta_saida"]
        os.makedirs(pasta, exist_ok=True)
        analisador = AnalisadorFACS(fps, self.espelhado, self.prefs["calibracao_segundos"])
        resultados = None if self.tempo_real else []
        self.status_signal.emit("Analisando... (F2 marca uma pergunta)" if self.tempo_real
                                else "Analisando para revisão...")

        def corte(t, motivo):
            if self.prefs["salvar_recortes"]:
                registrar_corte(self.video_path, t, motivo, cortes, pasta)

        frame_idx = 0
        alertas, cortes = [], set()
        inicio_relogio = time.monotonic()
        while self._run_flag:
            success, frame = cap.read()
            if not success: break

            res = analisador.processar(frame, frame_idx)
            frame_idx += 1
            timestamp_str = formatar_tempo(res.tempo)

            for ep in res.novos_episodios:
                self.evento_signal.emit(f"[{timestamp_str}] {ep['emocao']} — {ep['aus']}", ep["emocao"], ep["inicio"])
                corte(res.tempo, f"{ep['emocao']} {ep['aus']}")
            for p in res.novos_pontos:
                self.evento_signal.emit(texto_ponto(p), "atencao", p["inicio"])
                corte(p["inicio"], f"Atencao {p['tipo']}")
            for alerta in res.alertas:
                alertas.append((res.tempo, alerta))
                self.evento_signal.emit(f"[{timestamp_str}] {alerta}", "alerta", res.tempo)
                corte(res.tempo, alerta)

            if resultados is not None:
                resultados.append(compactar(res))
            # na revisao o frame so serve de previa; exibir todos atrasaria a analise
            if self.tempo_real or frame_idx % 10 == 1:
                desenhar(frame, res)
                self.frame_signal.emit(para_qimage(frame))
                self.linha_tempo_signal.emit([dict(e) for e in analisador.episodios],
                                             [dict(p) for p in analisador.atencao.pontos], res.tempo)
                if not self.tempo_real and total:
                    self.status_signal.emit(f"Analisando para revisão... {100 * frame_idx / total:.0f}%")

            if self.tempo_real:
                # acompanha o tempo real; se o processamento atrasar, segue sem pausa
                atraso = frame_idx / fps - (time.monotonic() - inicio_relogio)
                if atraso > 0:
                    self.msleep(int(atraso * 1000))

        cap.release()
        for p in analisador.finalizar():
            self.evento_signal.emit(texto_ponto(p), "atencao", p["inicio"])
        analisador.close()
        relatorio = os.path.join(pasta, "relatorio_emocoes.csv")
        salvar_relatorio(relatorio, analisador.episodios, alertas, analisador.atencao.pontos)
        self.linha_tempo_signal.emit([dict(e) for e in analisador.episodios],
                                     [dict(p) for p in analisador.atencao.pontos], frame_idx / fps)
        self.concluido_signal.emit({
            "video": self.video_path, "fps": fps, "duracao": frame_idx / fps, "pasta": pasta,
            "espelhado": self.espelhado, "episodios": analisador.episodios,
            "pontos": analisador.atencao.pontos, "tempos_piscadas": analisador.atencao.tempos_piscadas,
            "resultados": resultados, "relatorio": relatorio,
        })

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
    frame_signal = pyqtSignal(QImage)
    posicao_signal = pyqtSignal(float)
    tocando_signal = pyqtSignal(bool)

    def __init__(self, dados, prefs):
        super().__init__()
        self.dados = dados
        self.resultados = dados["resultados"]
        self.fps = dados["fps"]
        self.cap = cv2.VideoCapture(dados["video"])
        self.audio = PlayerAudio(dados["video"], prefs)
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
            if self.posicao() >= self.dados["duracao"] - 1 / self.fps:
                self.ir_para(0.0)
            self.audio.tocar()
            self.tocando = True
            self.tocando_signal.emit(True)

    def pausar(self):
        self.audio.pausar()
        self.tocando = False
        self.tocando_signal.emit(False)

    def ir_para(self, segundos):
        segundos = min(max(0.0, segundos), self.dados["duracao"])
        self.audio.posicionar(segundos)
        self._mostrar(segundos)

    def _tick(self):
        if not self.tocando:
            return
        t = self.audio.posicao()
        if t >= self.dados["duracao"]:
            self.pausar()
            t = self.dados["duracao"]
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
        if self.dados["espelhado"]:
            frame = cv2.flip(frame, 1)
        desenhar(frame, self.resultados[idx])
        self.frame_signal.emit(para_qimage(frame))
        self.posicao_signal.emit(idx / self.fps)

    def fechar(self):
        self.timer.stop()
        self.audio.parar()
        self.cap.release()


class LinhaDoTempo(QWidget):
    """Uma faixa por emoção e uma de pontos de atenção, com as perguntas marcadas."""
    ALTURA_FAIXA, LARGURA_ROTULO, TOPO = 18, 90, 14
    clicado = pyqtSignal(float)

    def __init__(self, perguntas):
        super().__init__()
        self.duracao, self.posicao, self.episodios, self.pontos = 0.0, 0.0, [], []
        self.perguntas = perguntas
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

        self.chk_webcam = QCheckBox("Marcar 'vídeo de webcam (espelhado)' por padrão")
        self.chk_webcam.setChecked(prefs["webcam_espelhada"])
        form.addRow("", self.chk_webcam)

        self.spn_calibracao = QDoubleSpinBox()
        self.spn_calibracao.setRange(0.0, 10.0)
        self.spn_calibracao.setSingleStep(0.5)
        self.spn_calibracao.setSuffix(" s")
        self.spn_calibracao.setValue(prefs["calibracao_segundos"])
        form.addRow("Calibração neutra:", self.spn_calibracao)

        pasta = QHBoxLayout()
        self.txt_pasta = QLineEdit(prefs["pasta_saida"])
        btn_pasta = QPushButton("...")
        btn_pasta.clicked.connect(self._escolher_pasta)
        pasta.addWidget(self.txt_pasta)
        pasta.addWidget(btn_pasta)
        form.addRow("Pasta de saída:", pasta)

        self.chk_recortes = QCheckBox("Salvar recorte de 5 s de cada evento")
        self.chk_recortes.setChecked(prefs["salvar_recortes"])
        form.addRow("", self.chk_recortes)

        arquivo = QLabel(f"Arquivo: {caminho}")
        arquivo.setStyleSheet("color: #999999; font-size: 11px;")
        form.addRow(arquivo)
        botoes = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        botoes.accepted.connect(self.accept)
        botoes.rejected.connect(self.reject)
        form.addRow(botoes)

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


class MainWindow(QMainWindow):
    def __init__(self, prefs, caminho_prefs):
        super().__init__()
        self.prefs = prefs
        self.caminho_prefs = caminho_prefs
        self.setWindowTitle("Detector Forense - Modo Grafico")
        self.setGeometry(100, 100, 1280, 820)
        self.setStyleSheet("background-color: #121212; color: #FFFFFF;")
        self.perguntas = []   # (tempo s, texto); opcional
        self.thread = None    # analise em andamento
        self.audio = None     # audio do modo "analise"
        self.revisao = None   # player do modo "revisao"
        self.dados = None     # resultado da ultima analise concluida
        self.initUI()

    def initUI(self):
        menu = self.menuBar().addMenu("Arquivo")
        acao_prefs = QAction("Preferências...", self)
        acao_prefs.setShortcut(QKeySequence("Ctrl+,"))
        acao_prefs.triggered.connect(self.abrir_preferencias)
        menu.addAction(acao_prefs)
        acao_sair = QAction("Sair", self)
        acao_sair.triggered.connect(self.close)
        menu.addAction(acao_sair)

        layout = QHBoxLayout()
        left = QVBoxLayout()
        self.video_label = QLabel("Nenhum video carregado.")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #1E1E1E; border-radius: 5px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.video_label, stretch=1)

        controles = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.alternar_play)
        self.btn_play.setEnabled(False)
        self.lbl_tempo = QLabel("00:00 / 00:00")
        controles.addWidget(self.btn_play)
        controles.addWidget(self.lbl_tempo)
        controles.addStretch(1)
        self.lbl_modo = QLabel()
        self.lbl_modo.setStyleSheet("color: #999999;")
        controles.addWidget(self.lbl_modo)
        left.addLayout(controles)
        QShortcut(QKeySequence("Space"), self, self._atalho_play)

        self.linha_tempo = LinhaDoTempo(self.perguntas)
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
        self.chk_webcam.setChecked(self.prefs["webcam_espelhada"])
        left.addWidget(self.chk_webcam)

        self.btn_open = QPushButton("Selecionar Video")
        self.btn_open.setStyleSheet("background-color: #007ACC; padding: 10px; font-weight: bold;")
        self.btn_open.clicked.connect(self.open_file)
        left.addWidget(self.btn_open)

        right = QVBoxLayout()
        self.list_alerts = QListWidget()
        self.list_alerts.setStyleSheet("background-color: #1E1E1E;")
        self.list_alerts.setWordWrap(True)
        self.list_alerts.itemDoubleClicked.connect(self._ir_para_item)
        right.addWidget(self.list_alerts, stretch=2)
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

    # --- preferencias ---
    def abrir_preferencias(self):
        dialogo = DialogoPreferencias(self.prefs, self.caminho_prefs, self)
        if dialogo.exec() != QDialog.DialogCode.Accepted:
            return
        self.prefs.update(dialogo.valores())
        try:
            preferencias.salvar(self.prefs, self.caminho_prefs)
            self.statusBar().showMessage(f"Preferências salvas em {self.caminho_prefs}. "
                                         "O modo do player vale a partir do próximo vídeo.")
        except OSError as e:
            QMessageBox.warning(self, "Preferências", f"Não foi possível salvar: {e}")
        self.chk_webcam.setChecked(self.prefs["webcam_espelhada"])
        for player in (self.audio, self.revisao and self.revisao.audio):
            if player:
                player.aplicar_prefs(self.prefs)
        self._atualizar_rotulo_modo()

    def _atualizar_rotulo_modo(self):
        self.lbl_modo.setText(f"Player: {preferencias.MODOS_PLAYER[self.prefs['modo_player']]}")

    # --- video ---
    def _encerrar(self):
        self._sync.stop()
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        if self.audio:
            self.audio.parar()
            self.audio = None
        if self.revisao:
            self.revisao.fechar()
            self.revisao = None
        self.btn_play.setEnabled(False)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir Video", "", "Videos (*.mp4 *.avi)")
        if not file_name:
            return
        if self.perguntas and self.dados is not None:
            manter = QMessageBox.question(
                self, "Perguntas", f"Manter as {len(self.perguntas)} pergunta(s) do vídeo anterior?",
                defaultButton=QMessageBox.StandardButton.No)
            if manter != QMessageBox.StandardButton.Yes:
                self.perguntas.clear()
        self._encerrar()
        self.dados = None
        self.list_alerts.clear()
        self.list_resumo.clear()
        for t, texto in sorted(self.perguntas):
            self.adicionar_evento(f"[{formatar_tempo(t)}] ❓ Pergunta: {texto or '-'}", "pergunta", t)

        tempo_real = self.prefs["modo_player"] == "analise"
        self.thread = AnaliseThread(file_name, dict(self.prefs), self.chk_webcam.isChecked(), tempo_real)
        self.thread.frame_signal.connect(self.update_img)
        self.thread.evento_signal.connect(self.adicionar_evento)
        self.thread.status_signal.connect(self.statusBar().showMessage)
        self.thread.duracao_signal.connect(self._definir_duracao)
        self.thread.linha_tempo_signal.connect(self.linha_tempo.atualizar)
        self.thread.concluido_signal.connect(self.analise_concluida)
        if tempo_real:
            self.audio = PlayerAudio(file_name, self.prefs)
            self._sync_estado = None
            self.thread.linha_tempo_signal.connect(self._iniciar_audio)
        self.thread.start()

    def _definir_duracao(self, duracao):
        self.linha_tempo.duracao = duracao
        self._atualizar_tempo(0.0)

    def _atualizar_tempo(self, t):
        self.lbl_tempo.setText(f"{formatar_tempo(t)} / {formatar_tempo(self.linha_tempo.duracao)}")

    def _iniciar_audio(self, *_):
        self.thread.linha_tempo_signal.disconnect(self._iniciar_audio)
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
        self.dados = dados
        self._sync.stop()
        if self.audio:
            self.audio.parar()
            self.audio = None
        self.atualizar_resumo()
        status = f"Análise concluída. Relatório: {dados['relatorio']}"
        if dados["resultados"] is not None and dados["resultados"]:
            self.revisao = PlayerRevisao(dados, self.prefs)
            self.revisao.frame_signal.connect(self.update_img)
            self.revisao.posicao_signal.connect(self.linha_tempo.definir_posicao)
            self.revisao.posicao_signal.connect(self._atualizar_tempo)
            self.revisao.tocando_signal.connect(
                lambda tocando: self.btn_play.setText("⏸ Pausa" if tocando else "▶ Play"))
            self.btn_play.setEnabled(True)
            status += " | Revisão: Espaço = play/pausa; clique na linha do tempo ou dê duplo clique num evento para navegar."
        self.statusBar().showMessage(status)

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
        t = item.data(Qt.ItemDataRole.UserRole)
        if t is not None:
            self.ir_para(t)

    # --- perguntas ---
    def posicao_atual(self):
        if self.revisao:
            return self.revisao.posicao()
        if self.thread is not None and self.thread.isRunning():
            return self.linha_tempo.posicao
        return None

    def marcar_pergunta(self):
        tempo = self.posicao_atual()
        if tempo is None:
            self.statusBar().showMessage("Abra um vídeo para marcar perguntas.")
            return
        texto = self.txt_pergunta.text().strip()
        self.perguntas.append((tempo, texto))
        self.txt_pergunta.clear()
        self.adicionar_evento(f"[{formatar_tempo(tempo)}] ❓ Pergunta marcada: {texto or '-'}", "pergunta", tempo)
        self.linha_tempo.update()
        self.atualizar_resumo()

    def importar_perguntas(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Importar perguntas", "",
                                                 "Perguntas (*.csv *.txt);;Todos (*)")
        if not caminho:
            return
        try:
            novas = carregar_perguntas(caminho)
        except OSError as e:
            self.statusBar().showMessage(f"Erro ao ler perguntas: {e}")
            return
        self.perguntas.extend(novas)
        self.statusBar().showMessage(f"{len(novas)} pergunta(s) importada(s) de {os.path.basename(caminho)}.")
        if self.thread is not None:
            for t, texto in novas:
                self.adicionar_evento(f"[{formatar_tempo(t)}] ❓ Pergunta importada: {texto or '-'}", "pergunta", t)
        self.linha_tempo.update()
        self.atualizar_resumo()

    def atualizar_resumo(self):
        """Recalcula o resumo por pergunta (apos a analise; tambem quando perguntas mudam na revisao)."""
        self.list_resumo.clear()
        if self.dados is None or not self.perguntas:
            return
        d = self.dados
        resumo = resumo_perguntas(list(self.perguntas), d["duracao"], d["episodios"], d["pontos"],
                                  d["tempos_piscadas"])
        try:
            salvar_resumo_perguntas(os.path.join(d["pasta"], "relatorio_perguntas.csv"), resumo)
        except OSError as e:
            self.statusBar().showMessage(f"Erro ao salvar resumo: {e}")
        for (texto, destaque), r in zip(linhas_resumo(resumo), resumo):
            item = QListWidgetItem(texto)
            item.setForeground(cor_evento("destaque" if destaque else "resumo"))
            item.setData(Qt.ItemDataRole.UserRole, r["inicio"])
            self.list_resumo.addItem(item)

    def adicionar_evento(self, texto, categoria, tempo=None):
        item = QListWidgetItem(texto)
        item.setForeground(cor_evento(categoria))
        item.setData(Qt.ItemDataRole.UserRole, tempo)
        self.list_alerts.addItem(item)

    def update_img(self, img):
        px = QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(px)

    def closeEvent(self, evento):
        self._encerrar()
        super().closeEvent(evento)


def rodar_modo_grafico(prefs, caminho_prefs):
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow(prefs, caminho_prefs)
    w.show()
    sys.exit(app.exec())
