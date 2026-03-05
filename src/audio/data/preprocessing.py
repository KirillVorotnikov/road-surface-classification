import os
import numpy as np
import librosa
import torch
from typing import Union


class AudioPreprocessor:
    def __init__(
        self,
        target_sample_rate: int = 16000,  # Целевая частота дискретизации (Гц)
        duration_sec: float = 5.0,  # Фиксированная длительность аудио в секундах
        n_mels: int = 128,  # Кол-во Mel-фильтров для спектрограммы
        n_ffts: int = 2048,  # Размер окна FFT
        hop_length: int = 512,  # Шаг окна
        use_mfcc: bool = False,  # При True будет возвращать MFCC вместо Mel-спектрограммы
        n_mfcc: int = 40,  # Кол-во коэффициентов MFCC
    ):
        self.target_sample_rate = target_sample_rate
        self.duration_sec = duration_sec
        self.target_length = int(target_sample_rate * duration_sec)

        # Параметры для спектральных признаков
        self.n_mels = n_mels
        self.n_ffts = n_ffts
        self.hop_length = hop_length
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc

    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Загружает аудиофайл, конвертирует в моно и ресемплит на лету.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # librosa.load автоматически делает mono=True и resample=sr
        audio, sr = librosa.load(
            file_path,
            sr=self.target_sample_rate,
            mono=True,
            res_type="kaiser_best",  # Качественный ресемплинг
        )
        return audio

    def pad_or_crop(self, audio: np.ndarray) -> np.ndarray:
        """
        Приводит аудио к фиксированной длине
        Если короче, то pad нулями в конец
        Если длиннее, то crop с начала
        """
        current_length = audio.shape[0]

        if current_length < self.target_length:
            # Padding
            pad_width = self.target_length - current_length
            audio = np.pad(audio, (0, pad_width), mode="constant")
        elif current_length > self.target_length:
            # Cropping (берем центральный фрагмент для большей информативности)
            start = (current_length - self.target_length) // 2
            audio = audio[start : start + self.target_length]

        return audio

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Конвертация waveform в Mel-спектрограмму или MFCC
        """
        if self.use_mfcc:
            features = librosa.feature.mfcc(
                y=audio,
                sr=self.target_sample_rate,
                n_mfcc=self.n_mfcc,
                n_mels=self.n_mels,
                n_fft=self.n_ffts,
                hop_length=self.hop_length,
            )
        else:
            # Mel-Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.target_sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_ffts,
                hop_length=self.hop_length,
            )
            # Конвертация в децибелы (логарифмическая шкала)
            features = librosa.power_to_db(mel_spec, ref=1.0)

        return features

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Нормализация признаков (Mean=0, Std=1) по всему тензору.
        """
        mean = np.mean(features)
        std = np.std(features)

        # Защита от деления на ноль
        if std < 1e-8:
            std = 1e-8

        return (features - mean) / std

    def process(self, file_path: str) -> torch.Tensor:
        """
        Полный пайплайн обработки одного файла
        """
        # 1. Загрузка + Моно + Ресемплинг
        audio = self.load_audio(file_path)

        # 2. Pad/Crop
        audio = self.pad_or_crop(audio)

        # 3. Извлечение признаков (Mel или MFCC)
        features = self.extract_features(audio)

        # 4. Нормализация
        features = self.normalize(features)

        return torch.from_numpy(features).float()

    def __call__(self, file_path: str) -> torch.Tensor:
        """Вызов как функция"""
        return self.process(file_path)
