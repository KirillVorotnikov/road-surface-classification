"""PyTorch Dataset for Audio Classification of Road Surfaces."""

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from src.core.registry import DATASET_REGISTRY
from .preprocessing import AudioPreprocessor
from .transforms import AudioAugmentations


@DATASET_REGISTRY.register("audio_mel_dataset")
class AudioMelDataset(Dataset):
    """
    Dataset: audio files -> mel spectrograms (or MFCC).

    CSV format:
        filepath,label
        clips/dry_001.wav,dry_asphalt
        clips/wet_042.wav,wet_asphalt

    Output __getitem__:
        features: torch.Tensor shape (1, n_mels, time_steps)
        label: int
    """

    CLASS_NAMES = [
        "dry_asphalt", "wet_asphalt", "snow", "ice", "gravel"
    ]
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

    def __init__(
        self,
        csv_path: str | Path,
        audio_root: str | Path,
        sample_rate: int = 16000,
        duration_sec: float = 5.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        use_mfcc: bool = False,
        n_mfcc: int = 40,
        augmentation_preset: str | None = None,
        augmentation_waveform_p: float = 0.5,
        augmentation_spec_p: float = 0.5,
    ):
        """
        Args:
            csv_path: Path to CSV with filepath and label columns.
            audio_root: Root folder with audio files.
            sample_rate: Sample rate.
            duration_sec: Fixed clip duration.
            n_mels: Number of mel filters.
            n_fft: FFT window size.
            hop_length: Window step.
            use_mfcc: Use MFCC instead of mel spectrogram.
            n_mfcc: Number of MFCC coefficients.
            augmentation_preset: Augmentation preset (light/medium/heavy/None).
            augmentation_waveform_p: Probability of waveform augmentations.
            augmentation_spec_p: Probability of spectrogram augmentations.
        """
        self.df = pd.read_csv(csv_path)
        self.audio_root = Path(audio_root)

        # Validation
        required_columns = {"filepath", "label"}
        actual_columns = set(self.df.columns)
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            raise ValueError(f"CSV missing columns: {missing}")

        unknown_labels = set(self.df["label"].unique()) - set(self.CLASS_NAMES)
        if unknown_labels:
            raise ValueError(f"Unknown labels in CSV: {unknown_labels}")

        self.preprocessor = AudioPreprocessor(
            target_sample_rate=sample_rate,
            duration_sec=duration_sec,
            n_mels=n_mels,
            n_ffts=n_fft,
            hop_length=hop_length,
            use_mfcc=use_mfcc,
            n_mfcc=n_mfcc,
        )

        # Augmentation
        if augmentation_preset is not None:
            self.augmentations = AudioAugmentations(
                preset=augmentation_preset,
                sample_rate=sample_rate,
                waveform_p=augmentation_waveform_p,
                spec_p=augmentation_spec_p,
            )
        else:
            self.augmentations = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        audio_path = self.audio_root / row["filepath"]
        label = self.CLASS_TO_IDX[row["label"]]

        # 1. Loading + resampling + mono
        waveform = self.preprocessor.load_audio(str(audio_path))

        # 2. Waveform augmentation (to spectrogram)
        if self.augmentations is not None:
            waveform = self.augmentations.augment_waveform(waveform)

        # 3. Pad/Crop to a fixed length
        waveform = self.preprocessor.pad_or_crop(waveform)

        # 4. Feature Extraction (mel/mfcc)c
        features = self.preprocessor.extract_features(waveform)

        # 5. Spectrogram augmentation (SpecAugment)
        if self.augmentations is not None:
            features = self.augmentations.augment_spectrogram(features)

        # 6. Normalization
        features = self.preprocessor.normalize(features)

        # 7. Add a channel dimension to the tensor
        features = torch.from_numpy(features).float()
        features = features.unsqueeze(0)

        return features, label

    def get_class_distribution(self) -> dict[str, int]:
        """Class distribution in the dataset."""
        return dict(self.df["label"].value_counts())

    def get_sample_weights(self) -> torch.Tensor:
        """Weights for WeightedRandomSampler (combating imbalances)."""
        class_counts = self.df["label"].value_counts()
        weights_per_class = 1.0 / class_counts
        sample_weights = self.df["label"].map(weights_per_class).values
        return torch.tensor(sample_weights, dtype=torch.float)