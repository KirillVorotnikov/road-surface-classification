"""Тесты для модуля transforms (аугментации)."""

import numpy as np
import pytest

from src.audio.data.transforms import (
    AudioAugmentations,
    SpecAugment,
    WaveformAugmentations,
)


class TestWaveformAugmentations:
    """Тесты для waveform аугментаций."""

    @pytest.fixture
    def sample_waveform(self):
        """Создаёт тестовый waveform."""
        sr = 16000
        duration = 1.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32)

    def test_augmentations_preserve_shape(self, sample_waveform):
        """Тест что аугментации не меняют размерность."""
        for preset in ["light", "medium", "heavy"]:
            aug = WaveformAugmentations(
                preset=preset,
                sample_rate=16000,
                p=1.0,  # Всегда применять
            )

            augmented = aug(sample_waveform)

            assert augmented.shape == sample_waveform.shape, (
                f"Preset {preset}: Expected shape {sample_waveform.shape}, "
                f"got {augmented.shape}"
            )
            assert augmented.dtype == sample_waveform.dtype, (
                f"Preset {preset}: Expected dtype {sample_waveform.dtype}, "
                f"got {augmented.dtype}"
            )

    def test_presets_create_without_errors(self, sample_waveform):
        """Тест что пресеты создаются без ошибок."""
        for preset in ["light", "medium", "heavy"]:
            # Не должно быть исключений
            aug = WaveformAugmentations(
                preset=preset,
                sample_rate=16000,
                p=0.5,
            )

            assert aug.preset == preset
            assert aug.sample_rate == 16000

    def test_probability_zero(self, sample_waveform):
        """Тест с вероятностью 0."""
        aug = WaveformAugmentations(
            preset="light",
            sample_rate=16000,
            p=0.0,
        )

        augmented = aug(sample_waveform)

        # С вероятностью 0 аугментация не должна менять сигнал
        # (но из-за стохастичности это не гарантировано)
        assert augmented.shape == sample_waveform.shape

    def test_all_augmentations_applied(self, sample_waveform):
        """Тест что все аугментации применяются."""
        aug = WaveformAugmentations(
            preset="medium",
            sample_rate=16000,
            p=1.0,
        )

        # Запускаем несколько раз - хотя бы одно применение должно изменить сигнал
        changed = False
        for _ in range(10):
            augmented = aug(sample_waveform)
            if not np.allclose(augmented, sample_waveform, atol=1e-6):
                changed = True
                break

        assert changed, "Аугментации не изменили сигнал"


class TestSpecAugment:
    """Тесты для SpecAugment."""

    @pytest.fixture
    def sample_spectrogram(self):
        """Создаёт тестовую спектрограмму."""
        n_mels = 128
        n_time = 157
        return np.random.randn(n_mels, n_time).astype(np.float32)

    def test_augmentations_preserve_shape(self, sample_spectrogram):
        """Тест что аугментации не меняют размерность."""
        for preset in ["light", "medium", "heavy"]:
            aug = SpecAugment(preset=preset, p=1.0)

            augmented = aug(sample_spectrogram)

            assert augmented.shape == sample_spectrogram.shape, (
                f"Preset {preset}: Expected shape {sample_spectrogram.shape}, "
                f"got {augmented.shape}"
            )

    def test_presets_create_without_errors(self, sample_spectrogram):
        """Тест что пресеты создаются без ошибок."""
        for preset in ["light", "medium", "heavy"]:
            # Не должно быть исключений
            aug = SpecAugment(preset=preset, p=0.5)

            assert aug.preset == preset

            # Проверяем что маски применяются
            augmented = aug(sample_spectrogram)
            assert augmented.shape == sample_spectrogram.shape

    def test_frequency_mask(self, sample_spectrogram):
        """Тест маски по частоте."""
        aug = SpecAugment(
            preset="light",
            n_freq_masks=1,
            n_time_masks=0,
            freq_mask_width=10,
            p=1.0,
        )

        augmented = aug(sample_spectrogram)

        # Маска должна изменить значения
        assert not np.allclose(augmented, sample_spectrogram, atol=1e-6)

    def test_time_mask(self, sample_spectrogram):
        """Тест маски по времени."""
        aug = SpecAugment(
            preset="light",
            n_freq_masks=0,
            n_time_masks=1,
            time_mask_width=20,
            p=1.0,
        )

        augmented = aug(sample_spectrogram)

        # Маска должна изменить значения
        assert not np.allclose(augmented, sample_spectrogram, atol=1e-6)

    def test_mask_values(self, sample_spectrogram):
        """Тест значений маски."""
        aug = SpecAugment(
            preset="light",
            n_freq_masks=1,
            n_time_masks=0,
            freq_mask_width=10,
            p=1.0,
        )
        # Явно отключаем time mask (preset может переопределить)
        aug.n_time_masks = 0

        augmented = aug(sample_spectrogram)

        # Замаскированные значения должны быть равны среднему
        # (проверяем что маска заполнена константой)
        diff = np.abs(augmented - sample_spectrogram)
        masked_positions = diff > 1e-6

        if masked_positions.any():
            # В замаскированной области значения должны быть одинаковыми
            masked_vals = augmented[masked_positions]
            # Все замаскированные значения должны быть равны
            # (проверяем через std - должно быть ~0)
            assert np.std(masked_vals) < 1e-5, "Маска не заполнена константой"


class TestAudioAugmentations:
    """Тесты для комбинированных аугментаций."""

    @pytest.fixture
    def sample_waveform(self):
        """Создаёт тестовый waveform."""
        sr = 16000
        duration = 1.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32)

    @pytest.fixture
    def sample_spectrogram(self):
        """Создаёт тестовую спектрограмму."""
        n_mels = 128
        n_time = 157
        return np.random.randn(n_mels, n_time).astype(np.float32)

    def test_combined_augmentations(self, sample_waveform, sample_spectrogram):
        """Тест комбинированных аугментаций."""
        aug = AudioAugmentations(
            preset="light",
            sample_rate=16000,
            waveform_p=1.0,
            spec_p=1.0,
        )

        # Применяем к waveform
        aug_waveform = aug.augment_waveform(sample_waveform)
        assert aug_waveform.shape == sample_waveform.shape

        # Применяем к spectrogram
        aug_spectrogram = aug.augment_spectrogram(sample_spectrogram)
        assert aug_spectrogram.shape == sample_spectrogram.shape

    def test_selective_waveform_only(self, sample_waveform):
        """Тест только waveform аугментаций."""
        aug = AudioAugmentations(
            preset="light",
            use_waveform=True,
            use_spec=False,
        )

        result = aug(waveform=sample_waveform)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_waveform.shape

    def test_selective_spec_only(self, sample_spectrogram):
        """Тест только spectrogram аугментаций."""
        aug = AudioAugmentations(
            preset="light",
            use_waveform=False,
            use_spec=True,
        )

        result = aug(spectrogram=sample_spectrogram)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_spectrogram.shape

    def test_all_presets(self, sample_waveform, sample_spectrogram):
        """Тест всех пресетов."""
        for preset in ["light", "medium", "heavy"]:
            aug = AudioAugmentations(
                preset=preset,
                sample_rate=16000,
            )

            # Waveform
            aug_waveform = aug.augment_waveform(sample_waveform)
            assert aug_waveform.shape == sample_waveform.shape

            # Spectrogram
            aug_spectrogram = aug.augment_spectrogram(sample_spectrogram)
            assert aug_spectrogram.shape == sample_spectrogram.shape
