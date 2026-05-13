from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("МАЛАФЬЯ МАЛАФЬЮШЕЧКА")
        self.resize(700, 500)

        self.wave_label = QLabel("Ожидание аудио...")
        self.wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wave_label.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
        self.wave_label.setMinimumSize(640, 320)

        self.result_label = QLabel("Ожидание")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; background-color: #eee;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.wave_label)
        layout.addWidget(self.result_label)
        self.setCentralWidget(container)

    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        q_image = QImage(
            frame_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        self.wave_label.setPixmap(pixmap.scaled(
            self.wave_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def update_result(self, label):

        
        self.result_label.setText(
            f' <span style="color:#333">{label}</span> &nbsp;|&nbsp; '
        )