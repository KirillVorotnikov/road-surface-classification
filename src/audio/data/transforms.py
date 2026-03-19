"""
Аугментации для аудио-данных.

Поддерживает два типа аугментаций:
1. Waveform-аугментации (библиотека audiomentations)
2. Spectrogram-аугментации (SpecAugment)
"""

from typing import Literal

import numpy as np
from audiomentations import (
    AddGaussianNoise,
    Gain,
    PitchShift,
    Shift,
    TimeStretch,
)
from audiomentations import Compose as AudiomentationCompose

PresetType = Literal["light", "medium", "heavy"]


class WaveformAugmentations:
    """
    Аугментации на уровне waveform с использованием библиотеки audiomentations.

    Применяются к сырому аудио-сигналу перед конвертацией в спектрограмму.
    """

    def __init__(
        self,
        preset: PresetType = "light",
        sample_rate: int = 16000,
        p: float = 0.5,
    ):
        """
        Инициализация waveform аугментаций.

        Args:
            preset: Уровень аугментации ("light", "medium", "heavy")
            sample_rate: Частота дискретизации аудио
            p: Вероятность применения каждой аугментации
        """
        self.preset = preset
        self.sample_rate = sample_rate
        self.p = p

        self.transform = self._build_transform(preset, sample_rate, p)

    def _build_transform(
        self, preset: PresetType, sample_rate: int, p: float
    ) -> AudiomentationCompose:
        """Построение цепочки аугментаций в зависимости от пресета."""

        # Параметры для разных пресетов
        params = {
            "light": {
                "noise_std": 0.001,
                "stretch_min": 0.95,
                "stretch_max": 1.05,
                "semitones": 2,
                "shift_min": -0.1,
                "shift_max": 0.1,
                "gain_min": -3,
                "gain_max": 3,
            },
            "medium": {
                "noise_std": 0.005,
                "stretch_min": 0.9,
                "stretch_max": 1.1,
                "semitones": 4,
                "shift_min": -0.2,
                "shift_max": 0.2,
                "gain_min": -6,
                "gain_max": 6,
            },
            "heavy": {
                "noise_std": 0.01,
                "stretch_min": 0.8,
                "stretch_max": 1.2,
                "semitones": 6,
                "shift_min": -0.3,
                "shift_max": 0.3,
                "gain_min": -10,
                "gain_max": 10,
            },
        }

        cfg = params[preset]

        return AudiomentationCompose(
            [
                AddGaussianNoise(
                    min_amplitude=cfg["noise_std"],
                    max_amplitude=cfg["noise_std"] * 2,
                    p=p,
                ),
                TimeStretch(
                    min_rate=cfg["stretch_min"],
                    max_rate=cfg["stretch_max"],
                    p=p,
                ),
                PitchShift(
                    min_semitones=-cfg["semitones"],
                    max_semitones=cfg["semitones"],
                    p=p,
                ),
                Shift(
                    min_shift=cfg["shift_min"],
                    max_shift=cfg["shift_max"],
                    rollover=True,
                    p=p,
                ),
                Gain(
                    min_gain_db=cfg["gain_min"],
                    max_gain_db=cfg["gain_max"],
                    p=p,
                ),
            ]
        )

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        Применение аугментаций к waveform.

        Args:
            waveform: Аудио-сигнал в формате numpy array (mono, float32, [-1, 1])

        Returns:
            Аугментированный аудио-сигнал
        """
        return self.transform(waveform, sample_rate=self.sample_rate)


class SpecAugment:
    """
    SpecAugment - аугментации на уровне спектрограммы.

    Применяется к mel-спектрограмме или MFCC после логарифмической компрессии.
    """

    def __init__(
        self,
        preset: PresetType = "light",
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        freq_mask_width: int = 10,
        time_mask_width: int = 20,
        p: float = 0.5,
    ):
        """
        Инициализация SpecAugment.

        Args:
            preset: Уровень аугментации ("light", "medium", "heavy")
            n_freq_masks: Количество масок по частоте
            n_time_masks: Количество масок по времени
            freq_mask_width: Максимальная ширина маски по частоте
            time_mask_width: Максимальная ширина маски по времени
            p: Вероятность применения аугментации
        """
        self.preset = preset
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.p = p

        # Переопределяем параметры для пресетов
        if preset == "light":
            self.n_freq_masks = 1
            self.n_time_masks = 1
            self.freq_mask_width = 10
            self.time_mask_width = 20
        elif preset == "medium":
            self.n_freq_masks = 2
            self.n_time_masks = 2
            self.freq_mask_width = 20
            self.time_mask_width = 40
        elif preset == "heavy":
            self.n_freq_masks = 3
            self.n_time_masks = 3
            self.freq_mask_width = 30
            self.time_mask_width = 60

    def _frequency_mask(self, spectrogram: np.ndarray, mask_width: int) -> np.ndarray:
        """
        Применение маски по частоте.

        Args:
            spectrogram: Спектрограмма (n_freqs, n_time)
            mask_width: Ширина маски

        Returns:
            Замаскированная спектрограмма
        """
        n_freqs = spectrogram.shape[0]
        if mask_width >= n_freqs:
            return spectrogram

        # Случайная позиция маски
        start = np.random.randint(0, n_freqs - mask_width)

        # Заполняем средним значением по частоте
        mask_value = spectrogram[start : start + mask_width, :].mean()
        spectrogram[start : start + mask_width, :] = mask_value

        return spectrogram

    def _time_mask(self, spectrogram: np.ndarray, mask_width: int) -> np.ndarray:
        """
        Применение маски по времени.

        Args:
            spectrogram: Спектрограмма (n_freqs, n_time)
            mask_width: Ширина маски

        Returns:
            Замаскированная спектрограмма
        """
        n_time = spectrogram.shape[1]
        if mask_width >= n_time:
            return spectrogram

        # Случайная позиция маски
        start = np.random.randint(0, n_time - mask_width)

        # Заполняем средним значением по времени
        mask_value = spectrogram[:, start : start + mask_width].mean()
        spectrogram[:, start : start + mask_width] = mask_value

        return spectrogram

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Применение SpecAugment к спектрограмме.

        Args:
            spectrogram: Спектрограмма в формате numpy array (n_freqs, n_time)

        Returns:
            Замаскированная спектрограмма
        """
        spectrogram = spectrogram.copy()

        # Применяем маски по частоте
        if np.random.random() < self.p:
            for _ in range(self.n_freq_masks):
                mask_width = np.random.randint(0, self.freq_mask_width + 1)
                if mask_width > 0:
                    spectrogram = self._frequency_mask(spectrogram, mask_width)

        # Применяем маски по времени
        if np.random.random() < self.p:
            for _ in range(self.n_time_masks):
                mask_width = np.random.randint(0, self.time_mask_width + 1)
                if mask_width > 0:
                    spectrogram = self._time_mask(spectrogram, mask_width)

        return spectrogram


class AudioAugmentations:
    """
    Комбинированный класс для аугментации аудио-данных.

    Применяет waveform и spectrogram аугментации последовательно.
    """

    def __init__(
        self,
        preset: PresetType = "light",
        sample_rate: int = 16000,
        waveform_p: float = 0.5,
        spec_p: float = 0.5,
        use_waveform: bool = True,
        use_spec: bool = True,
    ):
        """
        Инициализация комбинированных аугментаций.

        Args:
            preset: Уровень аугментации ("light", "medium", "heavy")
            sample_rate: Частота дискретизации
            waveform_p: Вероятность применения waveform аугментаций
            spec_p: Вероятность применения spectrogram аугментаций
            use_waveform: Использовать ли waveform аугментации
            use_spec: Использовать ли spectrogram аугментации
        """
        self.preset = preset
        self.sample_rate = sample_rate
        self.use_waveform = use_waveform
        self.use_spec = use_spec

        if use_waveform:
            self.waveform_aug = WaveformAugmentations(
                preset=preset, sample_rate=sample_rate, p=waveform_p
            )
        else:
            self.waveform_aug = None

        if use_spec:
            self.spec_aug = SpecAugment(preset=preset, p=spec_p)
        else:
            self.spec_aug = None

    def augment_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Применить waveform аугментации."""
        if self.waveform_aug is None:
            return waveform
        return self.waveform_aug(waveform)

    def augment_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Применить spectrogram аугментации."""
        if self.spec_aug is None:
            return spectrogram
        return self.spec_aug(spectrogram)

    def __call__(
        self,
        waveform: np.ndarray | None = None,
        spectrogram: np.ndarray | None = None,
    ) -> np.ndarray | tuple:
        """
        Применение аугментаций.

        Args:
            waveform: Сырой аудио-сигнал (для waveform аугментаций)
            spectrogram: Спектрограмма (для spectrogram аугментаций)

        Returns:
            Аугментированный waveform и/или spectrogram
        """
        result = []

        if waveform is not None:
            result.append(self.augment_waveform(waveform))

        if spectrogram is not None:
            result.append(self.augment_spectrogram(spectrogram))

        if len(result) == 1:
            return result[0]
        return tuple(result)
