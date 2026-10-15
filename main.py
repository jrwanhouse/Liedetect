import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import time

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
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                 QPushButton, QVBoxLayout, QHBoxLayout, QListWidget, QFileDialog)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap

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
                self.status_signal.emit("Erro ao abrir o vídeo.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            output_dir = "trechos_suspeitos"
            os.makedirs(output_dir, exist_ok=True)

            piscadas = 0
            janela_frames = int(fps * 5)
            frame_counter = 0
            olhos_fechados = False
            alertas_registrados = []

            while self._run_flag:
                success, frame = cap.read()
                if not success: break

                frame_counter += 1
                segundos_totais = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                timestamp_str = f"{int(segundos_totais)//60:02d}:{int(segundos_totais)%60:02d}"
                h, w, ch = frame.shape

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                alerta_neste_frame, tipo_alerta = False, ""

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Rastreamento de Piscada (AU 45 aproximada)
                        p_sup = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                        p_inf = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                        if np.linalg.norm(p_sup - p_inf) < 4.5:
                            if not olhos_fechados: piscadas += 1; olhos_fechados = True
                        else: olhos_fechados = False

                        # Assimetria Facial (AU 14)
                        c_esq = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                        c_dir = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                        nariz = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                        if abs(np.linalg.norm(c_esq - nariz) - np.linalg.norm(c_dir - nariz)) > 15.0:
                            alerta_neste_frame = True
                            tipo_alerta = "Assimetria Facial Microexpressao (AU14)"
                            cv2.circle(frame, (int(c_esq[0]), int(c_esq[1])), 5, (0,0,255), -1)

                if frame_counter % janela_frames == 0:
                    if piscadas > 6: alerta_neste_frame = True; tipo_alerta = "Frequencia Critica de Piscadas (Estresse)"
                    piscadas = 0

                if alerta_neste_frame and (int(segundos_totais) not in alertas_registrados):
                    alertas_registrados.append(int(segundos_totais))
                    self.alert_signal.emit(f"[{timestamp_str}] {tipo_alerta}", segundos_totais)
                    recortar_trecho(self.video_path, segundos_totais, timestamp_str, tipo_alerta, output_dir)

                cv2.putText(frame, f"Tempo: {timestamp_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, w, h, ch*w, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
                self.msleep(int(1000 / fps))

            cap.release()
            self.status_signal.emit("Analise concluida.")

        def stop(self):
            self._run_flag = False
            self.wait()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Detector Forense - Modo Grafico")
            self.setGeometry(100, 100, 1000, 600)
            self.setStyleSheet("background-color: #121212; color: #FFFFFF;")
            self.initUI()

        def initUI(self):
            layout = QHBoxLayout()
            left = QVBoxLayout()
            self.video_label = QLabel("Nenhum video carregado.")
            self.video_label.setMinimumSize(640, 480)
            self.video_label.setStyleSheet("background-color: #1E1E1E; border-radius: 5px;")
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left.addWidget(self.video_label)
            
            self.btn_open = QPushButton("Selecionar Video")
            self.btn_open.setStyleSheet("background-color: #007ACC; padding: 10px; font-weight: bold;")
            self.btn_open.clicked.connect(self.open_file)
            left.addWidget(self.btn_open)
            
            self.list_alerts = QListWidget()
            self.list_alerts.setStyleSheet("background-color: #1E1E1E; color: #FF8A80;")
            layout.addLayout(left, stretch=3)
            layout.addWidget(self.list_alerts, stretch=1)
            
            c = QWidget(); c.setLayout(layout); self.setCentralWidget(c)

        def open_file(self):
            file_name, _ = QFileDialog.getOpenFileName(self, "Abrir Video", "", "Videos (*.mp4 *.avi)")
            if file_name:
                self.thread = VideoProcessorThread(file_name)
                self.thread.change_pixmap_signal.connect(self.update_img)
                self.thread.alert_signal.connect(lambda msg, _: self.list_alerts.addItem(msg))
                self.thread.start()

        def update_img(self, img):
            px = QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(px)

    app = QApplication(sys.argv)
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

    print(f"🔍 Inicializando analise computacional em segundo plano...")
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(video_input)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir = "trechos_suspeitos"
    os.makedirs(output_dir, exist_ok=True)

    piscadas = 0
    janela_frames = int(fps * 5)
    frame_counter = 0
    olhos_fechados = False
    alertas_registrados = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame_counter += 1
        segundos_totais = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        timestamp_str = f"{int(segundos_totais)//60:02d}:{int(segundos_totais)%60:02d}"
        
        if frame_counter % int(fps * 5) == 0 and total_frames > 0:
            porcentagem = (frame_counter / total_frames) * 100
            print(f"Progresso: {porcentagem:.1f}% completo... [Analise atual em {timestamp_str}]")

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        alerta_neste_frame, tipo_alerta = False, ""

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                p_sup = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                p_inf = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                if np.linalg.norm(p_sup - p_inf) < 4.5:
                    if not olhos_fechados: piscadas += 1; olhos_fechados = True
                else: olhos_fechados = False

                c_esq = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                c_dir = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                nariz = np.array([face_landmarks.landmark.x * w, face_landmarks.landmark.y * h])
                if abs(np.linalg.norm(c_esq - nariz) - np.linalg.norm(c_dir - nariz)) > 15.0:
                    alerta_neste_frame = True
                    tipo_alerta = "Assimetria Facial Microexpressao (AU14)"

        if frame_counter % janela_frames == 0:
            if piscadas > 6: alerta_neste_frame = True; tipo_alerta = "Frequencia Critica de Piscadas"
            piscadas = 0

        if alerta_neste_frame and (int(segundos_totais) not in alertas_registrados):
            alertas_registrados.append(int(segundos_totais))
            print(f"⚠️  [HOTSPOT DETECTADO em {timestamp_str}] -> {tipo_alerta}")
            recortar_trecho(video_input, segundos_totais, timestamp_str, tipo_alerta, output_dir)

    cap.release()
    print(f"\n✅ Concluido! Amostras isoladas geradas em '{output_dir}/'.")

# --- ESCREVER ARQUIVOS DE VIDEO ---
def recortar_trecho(video_origem, segundo_alerta, timestamp_str, motivo, pasta_destino):
    sub_cap = cv2.VideoCapture(video_origem)
    fps = sub_cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(sub_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(sub_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    segundo_inicio = max(0, segundo_alerta - 2.5)
    sub_cap.set(cv2.CAP_PROP_POS_MSEC, segundo_inicio * 1000)
    
    nome_m = motivo.replace(" ", "_").replace("(", "").replace(")", "")
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
