import sys
import os
import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QFileDialog, QStyle, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

class VideoProcessorThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)
    alert_signal = pyqtSignal(str, float)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._run_flag = True

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status_signal.emit("Erro ao abrir o arquivo de vídeo.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30

        output_dir = "trechos_suspeitos"
        os.makedirs(output_dir, exist_ok=True)

        piscadas_no_intervalo = 0
        janela_frames = int(fps * 5)
        frame_counter = 0
        olhos_fechados = False
        alertas_registrados_segundos = []

        while self._run_flag:
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1
            milissegundos_atuais = cap.get(cv2.CAP_PROP_POS_MSEC)
            segundos_totais = milissegundos_atuais / 1000
            
            minutos = int(segundos_totais) // 60
            segundos = int(segundos_totais) % 60
            timestamp_str = f"{minutos:02d}:{segundos:02d}"

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            alerta_neste_frame = False
            tipo_alerta = ""

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    p_sup = np.array([face_landmarks.landmark[159].x * w, face_landmarks.landmark[159].y * h])
                    p_inf = np.array([face_landmarks.landmark[145].x * w, face_landmarks.landmark[145].y * h])
                    dist_olho = np.linalg.norm(p_sup - p_inf)

                    if dist_olho < 4.5:
                        if not olhos_fechados:
                            piscadas_no_intervalo += 1
                            olhos_fechados = True
                    else:
                        olhos_fechados = False

                    c_esq = np.array([face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h])
                    c_dir = np.array([face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h])
                    nariz = np.array([face_landmarks.landmark[4].x * w, face_landmarks.landmark[4].y * h])

                    dist_esq = np.linalg.norm(c_esq - nariz)
                    dist_dir = np.linalg.norm(c_dir - nariz)
                    assimetria = abs(dist_esq - dist_dir)

                    if assimetria > 15.0:
                        alerta_neste_frame = True
                        tipo_alerta = "Assimetria Facial Microexpressao (AU14)"

            if frame_counter % janela_frames == 0:
                if piscadas_no_intervalo > 6:
                    alerta_neste_frame = True
                    tipo_alerta = "Frequencia Critica de Piscadas (Estresse)"
                piscadas_no_intervalo = 0

            segundo_arredondado = int(segundos_totais)
            if alerta_neste_frame and (segundo_arredondado not in alertas_registrados_segundos):
                alertas_registrados_segundos.append(segundo_arredondado)
                msg_alerta = f"[{timestamp_str}] {tipo_alerta}"
                self.alert_signal.emit(msg_alerta, segundos_totais)
                self.recortar_trecho_suspeito(self.video_path, segundos_totais, timestamp_str, tipo_alerta, output_dir)

            cv2.putText(frame, f"Tempo do Video: {timestamp_str}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
            self.msleep(int(1000 / fps))

        cap.release()
        self.status_signal.emit("Analise finalizada com sucesso.")

    def stop(self):
        self._run_flag = False
        self.wait()

    def recortar_trecho_suspeito(self, video_origem, segundo_alerta, timestamp_str, motivo, pasta_destino):
        sub_cap = cv2.VideoCapture(video_origem)
        fps = sub_cap.get(cv2.CAP_PROP_FPS)
        w = int(sub_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(sub_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        segundo_inicio = max(0, segundo_alerta - 2.5)
        sub_cap.set(cv2.CAP_PROP_POS_MSEC, segundo_inicio * 1000)
        
        nome_motivo = motivo.replace(" ", "_").replace("(", "").replace(")", "")
        timestamp_arquivo = timestamp_str.replace(":", "m") + "s"
        caminho_saida = os.path.join(pasta_destino, f"hotspot_{timestamp_arquivo}_{nome_motivo}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(caminho_saida, fourcc, fps, (w, h))
        
        for _ in range(int(fps * 5)):
            ret, f = sub_cap.read()
            if not ret:
                break
            out.write(f)
            
        sub_cap.release()
        out.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Hotspots Comportamentais & FACS Forense")
        self.setGeometry(100, 100, 1100, 650)
        self.setStyleSheet("background-color: #121212; color: #FFFFFF;")
        self.thread = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Nenhum video carregado.\nClique em 'Selecionar Video' para iniciar.")
        self.video_label.setFont(QFont("Arial", 14))
        self.video_label.setStyleSheet("background-color: #1E1E1E; border: 2px dashed #333333; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)

        buttons_layout = QHBoxLayout()
        self.btn_open = QPushButton(" Selecionar Video")
        self.btn_open.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.btn_open.setStyleSheet("background-color: #007ACC; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.btn_open.clicked.connect(self.open_file)
        buttons_layout.addWidget(self.btn_open)

        self.btn_stop = QPushButton(" Interromper")
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.btn_stop.setStyleSheet("background-color: #D32F2F; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        buttons_layout.addWidget(self.btn_stop)

        left_layout.addLayout(buttons_layout)

        self.lbl_status = QLabel("Status: Aguardando arquivo...")
        self.lbl_status.setStyleSheet("color: #AAAAAA; font-style: italic; padding: 5px;")
        left_layout.addWidget(self.lbl_status)

        right_layout = QVBoxLayout()
        lbl_panel_title = QLabel("Hotspots de Inconsistencia Detectados")
        lbl_panel_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        lbl_panel_title.setStyleSheet("color: #FF5252; padding-bottom: 5px;")
        right_layout.addWidget(lbl_panel_title)

        self.list_alerts = QListWidget()
        self.list_alerts.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px;
                color: #FF8A80;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2A2A2A;
            }
        """)
        right_layout.addWidget(self.list_alerts)
        
        lbl_info_export = QLabel("ℹ️ Clipes de 5 segundos de cada alerta sao exportados em /trechos_suspeitos")
        lbl_info_export.setWordWrap(True)
        lbl_info_export.setStyleSheet("color: #81C784; font-size: 11px;")
        right_layout.addWidget(lbl_info_export)

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Selecionar Depoimento em Video", "", "Arquivos de Video (*.mp4 *.avi *.mov *.mkv)")
        if file_name:
            self.list_alerts.clear()
            self.lbl_status.setText(f"Status: Analisando {os.path.basename(file_name)}...")
            self.btn_open.setEnabled(False)
            self.btn_stop.setEnabled(True)

            self.thread = VideoProcessorThread(file_name)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.status_signal.connect(self.update_status)
            self.thread.alert_signal.connect(self.add_alert_to_list)
            self.thread.start()

    def stop_analysis(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.lbl_status.setText("Status: Analise interrompida pelo usuario.")
            self.btn_open.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_status(self, text):
        self.lbl_status.setText(f"Status: {text}")
        if "finalizada" in text.lower() or "erro" in text.lower():
            self.btn_open.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def add_alert_to_list(self, alert_text, timestamp):
        self.list_alerts.addItem(alert_text)
        self.list_alerts.scrollToBottom()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
