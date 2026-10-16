import sys
import os
import cv2
from facs import (AnalisadorFACS, EMOCOES, COR_EMOCAO, desenhar, formatar_tempo,
                  salvar_relatorio)

PASTA_SAIDA = "trechos_suspeitos"

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

# --- MODO INTERFACE GRÁFICA (GUI) ---
def rodar_modo_grafico():
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QListWidget,
                                 QListWidgetItem, QPushButton, QCheckBox, QVBoxLayout, QHBoxLayout,
                                 QFileDialog)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter

    def cor_qt(emocao):
        b, g, r = COR_EMOCAO[emocao]
        return QColor(r, g, b)

    class VideoProcessorThread(QThread):
        change_pixmap_signal = pyqtSignal(QImage)
        status_signal = pyqtSignal(str)
        evento_signal = pyqtSignal(str, str)        # texto, emocao ("" para alerta)
        duracao_signal = pyqtSignal(float)
        linha_tempo_signal = pyqtSignal(object, float)  # episodios, posicao (s)

        def __init__(self, video_path, espelhado):
            super().__init__()
            self.video_path = video_path
            self.espelhado = espelhado
            self._run_flag = True

        def run(self):
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.status_signal.emit("Erro ao abrir o vídeo.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            self.duracao_signal.emit(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
            os.makedirs(PASTA_SAIDA, exist_ok=True)
            analisador = AnalisadorFACS(fps, self.espelhado)
            self.status_signal.emit("Analisando...")

            frame_idx = 0
            alertas, cortes = [], set()
            while self._run_flag:
                success, frame = cap.read()
                if not success: break

                res = analisador.processar(frame, frame_idx)
                frame_idx += 1
                timestamp_str = formatar_tempo(res.tempo)

                for ep in res.novos_episodios:
                    self.evento_signal.emit(f"[{timestamp_str}] {ep['emocao']} — {ep['aus']}", ep["emocao"])
                    registrar_corte(self.video_path, res.tempo, f"{ep['emocao']} {ep['aus']}", cortes)
                for alerta in res.alertas:
                    alertas.append((res.tempo, alerta))
                    self.evento_signal.emit(f"[{timestamp_str}] {alerta}", "")
                    registrar_corte(self.video_path, res.tempo, alerta, cortes)

                desenhar(frame, res)
                h, w, ch = frame.shape
                qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, w, h, ch*w,
                                  QImage.Format.Format_RGB888).copy()
                self.change_pixmap_signal.emit(qt_image)
                self.linha_tempo_signal.emit([dict(e) for e in analisador.episodios], res.tempo)
                self.msleep(int(1000 / fps))

            cap.release()
            analisador.close()
            relatorio = os.path.join(PASTA_SAIDA, "relatorio_emocoes.csv")
            salvar_relatorio(relatorio, analisador.episodios, alertas)
            self.status_signal.emit(f"Análise concluída. Relatório: {relatorio}")

        def stop(self):
            self._run_flag = False
            self.wait()

    class LinhaDoTempo(QWidget):
        """Uma faixa por emoção, com os trechos do vídeo onde ela foi detectada."""
        ALTURA_FAIXA, LARGURA_ROTULO = 18, 90

        def __init__(self):
            super().__init__()
            self.duracao, self.posicao, self.episodios = 0.0, 0.0, []
            self.setMinimumHeight(self.ALTURA_FAIXA * len(EMOCOES) + 8)

        def atualizar(self, episodios, posicao):
            self.episodios, self.posicao = episodios, posicao
            self.update()

        def paintEvent(self, _):
            p = QPainter(self)
            largura = self.width() - self.LARGURA_ROTULO - 5
            duracao = max(self.duracao, self.posicao, 0.001)
            x_de = lambda t: self.LARGURA_ROTULO + int(t / duracao * largura)
            for i, emocao in enumerate(EMOCOES):
                y = 4 + i * self.ALTURA_FAIXA
                p.setPen(QColor("#CCCCCC"))
                p.drawText(0, y, self.LARGURA_ROTULO - 5, self.ALTURA_FAIXA - 2,
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, emocao)
                p.fillRect(self.LARGURA_ROTULO, y + 2, largura, self.ALTURA_FAIXA - 6, QColor("#2A2A2A"))
                for ep in self.episodios:
                    if ep["emocao"] == emocao:
                        x0 = x_de(ep["inicio"])
                        p.fillRect(x0, y + 2, max(2, x_de(ep["fim"]) - x0), self.ALTURA_FAIXA - 6, cor_qt(emocao))
            p.setPen(QColor("#FFFFFF"))
            p.drawLine(x_de(self.posicao), 0, x_de(self.posicao), self.height())
            p.end()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Detector Forense - Modo Grafico")
            self.setGeometry(100, 100, 1200, 750)
            self.setStyleSheet("background-color: #121212; color: #FFFFFF;")
            self.initUI()

        def initUI(self):
            layout = QHBoxLayout()
            left = QVBoxLayout()
            self.video_label = QLabel("Nenhum video carregado.")
            self.video_label.setMinimumSize(640, 480)
            self.video_label.setStyleSheet("background-color: #1E1E1E; border-radius: 5px;")
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left.addWidget(self.video_label, stretch=1)

            self.linha_tempo = LinhaDoTempo()
            left.addWidget(self.linha_tempo)

            self.chk_webcam = QCheckBox("Vídeo de webcam (imagem espelhada) — corrigir lados das AUs")
            left.addWidget(self.chk_webcam)

            self.btn_open = QPushButton("Selecionar Video")
            self.btn_open.setStyleSheet("background-color: #007ACC; padding: 10px; font-weight: bold;")
            self.btn_open.clicked.connect(self.open_file)
            left.addWidget(self.btn_open)

            self.list_alerts = QListWidget()
            self.list_alerts.setStyleSheet("background-color: #1E1E1E;")
            layout.addLayout(left, stretch=3)
            layout.addWidget(self.list_alerts, stretch=1)

            c = QWidget(); c.setLayout(layout); self.setCentralWidget(c)

        def open_file(self):
            file_name, _ = QFileDialog.getOpenFileName(self, "Abrir Video", "", "Videos (*.mp4 *.avi)")
            if file_name:
                self.list_alerts.clear()
                self.thread = VideoProcessorThread(file_name, self.chk_webcam.isChecked())
                self.thread.change_pixmap_signal.connect(self.update_img)
                self.thread.evento_signal.connect(self.adicionar_evento)
                self.thread.status_signal.connect(self.statusBar().showMessage)
                self.thread.duracao_signal.connect(lambda d: setattr(self.linha_tempo, "duracao", d))
                self.thread.linha_tempo_signal.connect(self.linha_tempo.atualizar)
                self.thread.start()

        def adicionar_evento(self, texto, emocao):
            item = QListWidgetItem(texto)
            item.setForeground(cor_qt(emocao) if emocao else QColor("#FF8A80"))
            self.list_alerts.addItem(item)

        def update_img(self, img):
            px = QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(px)

    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

# --- MODO TERMINAL (CLI) ---
def rodar_modo_terminal():
    print("\n" + "="*50)
    print("🖥️  MODO TERMINAL DETECTADO (Sem ambiente grafico ativo)")
    print("="*50)

    video_input = input("Digite o nome ou caminho do video para analisar: ").strip()
    if not os.path.exists(video_input):
        print(f"❌ Erro: O arquivo '{video_input}' nao foi encontrado.")
        return
    espelhado = input("O video vem de webcam com imagem espelhada? [s/N]: ").strip().lower() in ("s", "sim")

    print(f"🔍 Inicializando analise computacional em segundo plano...")

    cap = cv2.VideoCapture(video_input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    analisador = AnalisadorFACS(fps, espelhado)

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
            registrar_corte(video_input, res.tempo, f"{ep['emocao']} {ep['aus']}", cortes)
        for alerta in res.alertas:
            alertas.append((res.tempo, alerta))
            print(f"⚠️  [HOTSPOT DETECTADO em {timestamp_str}] -> {alerta}")
            registrar_corte(video_input, res.tempo, alerta, cortes)

    cap.release()
    analisador.close()

    print("\n📊 Resumo por emocao:")
    for emocao in EMOCOES:
        eps = [e for e in analisador.episodios if e["emocao"] == emocao]
        if eps:
            trechos = ", ".join(f"{formatar_tempo(e['inicio'])}-{formatar_tempo(e['fim'])}" for e in eps)
            print(f"  {emocao} ({len(eps)}x): {trechos}")
    if not analisador.episodios:
        print("  Nenhuma emocao detectada.")

    relatorio = os.path.join(PASTA_SAIDA, "relatorio_emocoes.csv")
    salvar_relatorio(relatorio, analisador.episodios, alertas)
    print(f"\n✅ Concluido! Amostras em '{PASTA_SAIDA}/', relatorio em '{relatorio}'.")

# --- ESCREVER ARQUIVOS DE VIDEO ---
def registrar_corte(video_origem, segundo_alerta, motivo, cortes):
    """Recorta o trecho do evento, no maximo um corte por segundo de video."""
    if int(segundo_alerta) in cortes:
        return
    cortes.add(int(segundo_alerta))
    recortar_trecho(video_origem, segundo_alerta, formatar_tempo(segundo_alerta), motivo, PASTA_SAIDA)

def recortar_trecho(video_origem, segundo_alerta, timestamp_str, motivo, pasta_destino):
    sub_cap = cv2.VideoCapture(video_origem)
    fps = sub_cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(sub_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(sub_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    segundo_inicio = max(0, segundo_alerta - 2.5)
    sub_cap.set(cv2.CAP_PROP_POS_MSEC, segundo_inicio * 1000)

    nome_m = "".join(c if c.isalnum() else "_" for c in motivo)
    caminho_saida = os.path.join(pasta_destino, f"hotspot_{timestamp_str.replace(':', 'm')}s_{nome_m}.mp4")
    out = cv2.VideoWriter(caminho_saida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for _ in range(int(fps * 5)):
        ret, f = sub_cap.read()
        if not ret: break
        out.write(f)
    sub_cap.release(); out.release()

if __name__ == "__main__":
    if verificar_suporte_grafico():
        rodar_modo_grafico()
    else:
        rodar_modo_terminal()
