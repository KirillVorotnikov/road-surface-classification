"""Тесты для модуля preprocessing."""

import pytest
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

from src.audio.data.preprocessing import AudioPreprocessor


@pytest.fixture
def temp_audio_file():
    """Создаёт временный аудиофайл для тестов."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = Path(f.name)
    
    # Создаём тестовый сигнал (5 секунд, 44100 Hz, стерео)
    sr = 44100
    duration = 5.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    # Синусоида + шум
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + np.random.randn(samples) * 0.05
    audio_stereo = np.column_stack([audio, audio])
    
    sf.write(temp_path, audio_stereo, sr)
    
    yield temp_path
    
    # Очистка
    if temp_path.exists():
        temp_path.unlink()


class TestPadOrCrop:
    """Тесты для pad/crop функциональности."""
    
    def test_pad_short_audio(self, temp_audio_file):
        """Тест padding короткого аудио."""
        # Аудио 5 секунд, target 10 секунд
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=10.0,
        )
        
        audio = preprocessor.load_audio(temp_audio_file)
        padded = preprocessor.pad_or_crop(audio)
        
        expected_samples = int(16000 * 10.0)
        assert len(padded) == expected_samples, (
            f"Expected {expected_samples} samples, got {len(padded)}"
        )
        
        # Первые 5 секунд должны содержать оригинальный сигнал
        original_length = int(16000 * 5.0)
        assert not np.allclose(padded[:original_length], 0, atol=1e-6)
        
        # Конец должен быть заполнен нулями
        assert np.allclose(padded[original_length:], 0, atol=1e-6)
    
    def test_crop_long_audio(self, temp_audio_file):
        """Тест crop длинного аудио."""
        # Аудио 5 секунд, target 2 секунды
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=2.0,
        )
        
        audio = preprocessor.load_audio(temp_audio_file)
        cropped = preprocessor.pad_or_crop(audio)
        
        expected_samples = int(16000 * 2.0)
        assert len(cropped) == expected_samples, (
            f"Expected {expected_samples} samples, got {len(cropped)}"
        )
        
        # Кроп должен брать из центра
        assert not np.allclose(cropped, 0, atol=1e-6)
    
    def test_exact_duration(self, temp_audio_file):
        """Тест аудио точной длительности."""
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
        )
        
        audio = preprocessor.load_audio(temp_audio_file)
        result = preprocessor.pad_or_crop(audio)
        
        expected_samples = int(16000 * 5.0)
        assert len(result) == expected_samples


class TestMelSpectrogram:
    """Тесты для mel-спектрограммы."""
    
    def test_mel_spectrogram_shape(self, temp_audio_file):
        """Тест формы mel-спектрограммы."""
        n_mels = 128
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
            n_mels=n_mels,
            use_mfcc=False,
        )
        
        features = preprocessor.process(temp_audio_file)
        
        # Проверяем количество mel-полос
        assert features.shape[0] == n_mels, (
            f"Expected {n_mels} mel bands, got {features.shape[0]}"
        )
        
        # Проверяем временную ось
        hop_length = 512
        expected_time_steps = int((5.0 * 16000) / hop_length)
        assert abs(features.shape[1] - expected_time_steps) < 5, (
            f"Expected ~{expected_time_steps} time steps, got {features.shape[1]}"
        )
    
    def test_mel_spectrogram_values(self, temp_audio_file):
        """Тест значений mel-спектрограммы."""
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
            n_mels=128,
            use_mfcc=False,
        )
        
        features = preprocessor.process(temp_audio_file)
        
        # После нормализации mean должен быть ~0
        assert abs(features.mean().item()) < 1e-5, (
            f"Expected mean ~0, got {features.mean().item()}"
        )
        
        # После нормализации std должен быть ~1 (с допуском на float precision)
        assert abs(features.std().item() - 1.0) < 1e-4, (
            f"Expected std ~1, got {features.std().item()}"
        )
    
    def test_different_n_mels(self, temp_audio_file):
        """Тест с разным количеством mel-полос."""
        for n_mels in [64, 128, 256]:
            preprocessor = AudioPreprocessor(
                target_sample_rate=16000,
                duration_sec=5.0,
                n_mels=n_mels,
                use_mfcc=False,
            )
            
            features = preprocessor.process(temp_audio_file)
            assert features.shape[0] == n_mels, (
                f"Expected {n_mels} mel bands, got {features.shape[0]}"
            )


class TestResampling:
    """Тесты для ресемплинга."""
    
    def test_resampling_from_44100_to_16000(self, temp_audio_file):
        """Тест ресемплинга с 44100 до 16000."""
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
        )
        
        # Загружаем аудио (исходник 44100 Hz)
        audio = preprocessor.load_audio(temp_audio_file)
        
        # Проверяем что частота дискретизации изменена
        # После load_audio частота должна быть target_sample_rate
        expected_samples = int(16000 * 5.0)
        assert abs(len(audio) - expected_samples) < 100, (
            f"Expected ~{expected_samples} samples after resampling, got {len(audio)}"
        )
    
    def test_resampling_from_8000_to_16000(self):
        """Тест ресемплинга с 8000 до 16000."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Создаём аудио 8000 Hz
            sr = 8000
            duration = 2.0
            samples = int(sr * duration)
            audio = np.random.randn(samples) * 0.1
            sf.write(temp_path, audio, sr)
            
            preprocessor = AudioPreprocessor(
                target_sample_rate=16000,
                duration_sec=duration,
            )
            
            result = preprocessor.load_audio(temp_path)
            expected_samples = int(16000 * duration)
            assert abs(len(result) - expected_samples) < 50, (
                f"Expected ~{expected_samples} samples, got {len(result)}"
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_mono_conversion(self, temp_audio_file):
        """Тест конвертации в моно."""
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
        )
        
        # Загружаем стерео аудио
        audio = preprocessor.load_audio(temp_audio_file)
        
        # Проверяем что аудио моно (1D массив)
        assert audio.ndim == 1, (
            f"Expected 1D array (mono), got {audio.ndim}D array"
        )


class TestMFCC:
    """Тесты для MFCC."""
    
    def test_mfcc_shape(self, temp_audio_file):
        """Тест формы MFCC."""
        n_mfcc = 40
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
            n_mfcc=n_mfcc,
            use_mfcc=True,
        )
        
        features = preprocessor.process(temp_audio_file)
        
        assert features.shape[0] == n_mfcc, (
            f"Expected {n_mfcc} MFCC coefficients, got {features.shape[0]}"
        )
    
    def test_mfcc_vs_mel(self, temp_audio_file):
        """Сравнение MFCC и mel-спектрограммы."""
        preprocessor_mel = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
            n_mels=128,
            use_mfcc=False,
        )
        
        preprocessor_mfcc = AudioPreprocessor(
            target_sample_rate=16000,
            duration_sec=5.0,
            n_mfcc=40,
            use_mfcc=True,
        )
        
        mel_features = preprocessor_mel.process(temp_audio_file)
        mfcc_features = preprocessor_mfcc.process(temp_audio_file)
        
        # Разные размеры
        assert mel_features.shape[0] == 128
        assert mfcc_features.shape[0] == 40
        
        # Одинаковая временная ось
        assert mel_features.shape[1] == mfcc_features.shape[1]
