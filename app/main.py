import sys
from PyQt6.QtWidgets import QApplication
from worker import AsyncWorker
from window import MainWindow

class Application:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        
        self.worker = AsyncWorker()  # audio_len больше не нужен
        self.worker.frame_ready.connect(self.window.update_frame)
        self.worker.result_ready.connect(self.window.update_result)
        
        self.app.aboutToQuit.connect(self.cleanup)

    def run(self):
        self.worker.start()
        self.window.show()
        sys.exit(self.app.exec())

    def cleanup(self):
        self.worker.stop()
        self.worker.wait()

if __name__ == "__main__":
    app = Application()
    app.run()