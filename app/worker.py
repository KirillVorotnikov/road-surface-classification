import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
import time
import sounddevice as sd
import threading
import librosa
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model_loader import load_model

class AsyncWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    result_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._running = True
        self.SAMPLE_RATE = 16000
        self.HISTORY_SEC = 3.0
        self.PRED_SEC = 1
        self.last_prediction = "..."
        self.last_confidence = 0.0
        self.last_inference_ms = 0.0
        self.fps = 0
        self._fps_counter = 0
        self._fps_time = time.time()
        self.HISTORY_SAMPLES = int(self.SAMPLE_RATE * self.HISTORY_SEC)
        self.PRED_SAMPLES = int(self.SAMPLE_RATE * self.PRED_SEC)

        self.audio_buffer = np.zeros(self.HISTORY_SAMPLES, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.frame_counter = 0

        try:
            self.model, self.scaler = load_model()
            print(f" Модель загружена. Ожидается {self.scaler.n_features_in_} признаков.")
        except Exception as e:
            print(f" Ошибка загрузки модели: {e}")
            self.model = None
            self.scaler = None

    def run(self):
        if self.model is None:
            print(" Модель не загружена.")
            return
        print(sd.query_devices())
        self.audio_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=512,
            device=0,
        )
        self.audio_stream.start()

        while self._running:
            with self.buffer_lock:
                buffer_copy = self.audio_buffer.copy()

            # Рендерим график КАЖДЫЙ кадр (~60 FPS)
            wave_img = self._generate_waveform(buffer_copy)
            self.frame_ready.emit(wave_img)

            # Предсказываем реже (раз в 20 кадров ≈ 0.3 сек)
            self.frame_counter += 1
            if self.frame_counter >= 20:
                self.frame_counter = 0
                pred = self._predict_audio(buffer_copy)
                self.result_ready.emit(pred)

            self.msleep(15)  # ~66 FPS цикл обновления интерфейса

        self.audio_stream.stop()
        self.audio_stream.close()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        new_data = indata[:, 0].astype(np.float32)
        with self.buffer_lock:
            self.audio_buffer[:-len(new_data)] = self.audio_buffer[len(new_data):]
            self.audio_buffer[-len(new_data):] = new_data

    def _generate_waveform(self, y):
        w, h = 640, 320

        # =========================
        # SPECTROGRAM
        # =========================

        y_vis = y[-self.HISTORY_SAMPLES:]

        D = librosa.stft(
            y_vis,
            n_fft=1024,
            hop_length=256,
            window='hann'
        )

        S = np.abs(D)

        S_db = librosa.amplitude_to_db(
            S,
            ref=1.0
        )

        # normalize
        S_db = np.clip(S_db, -80, 0)

        spec = ((S_db + 80) / 80.0 * 255).astype(np.uint8)

        # low freq bottom
        spec = np.flipud(spec)

        # resize
        spec = cv2.resize(
            spec,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        # grayscale -> BGR
        img = cv2.cvtColor(spec, cv2.COLOR_GRAY2BGR)

        # =========================
        # ANALYZING ZONE
        # =========================

        x_start = int(
            w * (self.HISTORY_SEC - self.PRED_SEC)
            / self.HISTORY_SEC
        )

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (x_start, 0),
            (w, h),
            (255, 255, 255),
            -1
        )

        img = cv2.addWeighted(
            overlay,
            0.08,
            img,
            0.92,
            0
        )

        cv2.rectangle(
            img,
            (x_start, 0),
            (w, h),
            (255, 255, 255),
            1
        )

        # =========================
        # AUDIO DEBUG
        # =========================

        rms = float(np.sqrt(np.mean(y_vis ** 2)))

        # fps counter
        self._fps_counter += 1

        now = time.time()

        if now - self._fps_time >= 1.0:
            self.fps = self._fps_counter
            self._fps_counter = 0
            self._fps_time = now

        # =========================
        # TEXT OVERLAY
        # =========================

        font = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            f"PRED: {self.last_prediction}",
            f"CONF: {self.last_confidence:.3f}",
            f"RMS:  {rms:.5f}",
            f"INF:  {self.last_inference_ms:.1f} ms",
            f"FPS:  {self.fps}",
        ]

        y0 = 25

        for line in lines:
            cv2.putText(
                img,
                line,
                (10, y0),
                font,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y0 += 24

        # =========================
        # LIVE INDICATOR
        # =========================

        blink = int(time.time() * 2) % 2

        color = (255, 255, 255) if blink else (80, 80, 80)

        cv2.circle(
            img,
            (w - 20, 20),
            6,
            color,
            -1
        )

        cv2.putText(
            img,
            "LIVE",
            (w - 70, 25),
            font,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

        return img

    def _predict_audio(self, y):
        if self.model is None or self.scaler is None:
            return "N/A"

        try:
            t0 = time.time()

            y_pred = y[-self.PRED_SAMPLES:]

            features = self._extract_features(y_pred)

            if features is None:
                return "N/A"

            features_scaled = self.scaler.transform(
                features.reshape(1, -1)
            )

            pred = self.model.predict(features_scaled)

            confidence = 0.0

            # если есть predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probs))

            inference_ms = (time.time() - t0) * 1000

            self.last_prediction = str(pred[0])
            self.last_confidence = confidence
            self.last_inference_ms = inference_ms

            return str(pred[0])

        except Exception as e:
            print(f" Ошибка предсказания: {e}")

            self.last_prediction = "ERROR"
            self.last_confidence = 0.0

            return "Error"

    def _extract_features(self, y):
        if len(y) < 1024:
            return None
        n_fft = 1024
        hop_length = 256

        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=self.SAMPLE_RATE, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        features.append(np.mean(mfcc, axis=1))
        features.append(np.std(mfcc, axis=1))
        features.append([np.mean(librosa.feature.spectral_centroid(y=y, sr=self.SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length)),
                         np.std(librosa.feature.spectral_centroid(y=y, sr=self.SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length))])
        features.append([np.mean(librosa.feature.spectral_bandwidth(y=y, sr=self.SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length)),
                         np.std(librosa.feature.spectral_bandwidth(y=y, sr=self.SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length))])
        features.append([np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length)),
                         np.std(librosa.feature.spectral_rolloff(y=y, sr=self.SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length))])
        features.append([np.mean(librosa.feature.zero_crossing_rate(y)),
                         np.std(librosa.feature.zero_crossing_rate(y))])
        features.append([np.mean(librosa.feature.rms(y=y)),
                         np.std(librosa.feature.rms(y=y))])
        return np.hstack(features)

    def stop(self):
        self._running = False
        self.wait()