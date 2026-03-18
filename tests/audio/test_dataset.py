import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.audio.data.dataset import AudioMelDataset


@pytest.fixture
def sample_dataset():
    """Creates a temporary dataset with wav and CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clips_dir = os.path.join(tmpdir, "clips")
        os.makedirs(clips_dir)

        files = []
        labels = []
        for i, label in enumerate(["dry_asphalt", "wet_asphalt", "snow"]):
            for j in range(3):
                filename = f"{label}_{j:03d}.wav"
                filepath = os.path.join(clips_dir, filename)
                audio = np.random.randn(16000).astype(np.float32)
                sf.write(filepath, audio, 16000)
                files.append(f"clips/{filename}")
                labels.append(label)

        # CSV
        csv_path = os.path.join(tmpdir, "data.csv")
        df = pd.DataFrame({"filepath": files, "label": labels})
        df.to_csv(csv_path, index=False)

        yield csv_path, tmpdir


class TestAudioMelDataset:
    """AudioMelDataset Tests."""

    def test_len(self, sample_dataset):
        """Dataset length = number of rows in CSV."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(csv_path=csv_path, audio_root=root)
        assert len(dataset) == 9  # 3 classes × 3 files

    def test_getitem_shapes(self, sample_dataset):
        """__getitem__ returns (features, label) with correct shapes."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(
            csv_path=csv_path, audio_root=root,
            sample_rate=16000, duration_sec=1.0,
            n_mels=64, n_fft=512, hop_length=128,
        )

        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(label, int)
        assert features.dim() == 3
        assert features.shape[0] == 1
        assert features.shape[1] == 64
        assert features.shape[2] > 0

    def test_labels_are_valid(self, sample_dataset):
        """All labels are valid indices."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(csv_path=csv_path, audio_root=root)

        for i in range(len(dataset)):
            _, label = dataset[i]
            assert 0 <= label < len(dataset.CLASS_NAMES)

    def test_augmentation_none_for_val(self, sample_dataset):
        """Without augmentations - the result is deterministic."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(
            csv_path=csv_path, audio_root=root,
            augmentation_preset=None,
        )

        f1, l1 = dataset[0]
        f2, l2 = dataset[0]

        assert torch.equal(f1, f2)
        assert l1 == l2

    def test_augmentation_preset(self, sample_dataset):
        """With augmentations - does not fall."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(
            csv_path=csv_path, audio_root=root,
            augmentation_preset="light",
        )

        features, label = dataset[0]
        assert features.dim() == 3

    def test_class_distribution(self, sample_dataset):
        """get_class_distribution returns correct calculations."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(csv_path=csv_path, audio_root=root)

        dist = dataset.get_class_distribution()
        assert dist["dry_asphalt"] == 3
        assert dist["wet_asphalt"] == 3
        assert dist["snow"] == 3

    def test_sample_weights_shape(self, sample_dataset):
        """get_sample_weights returns a tensor of the desired length."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(csv_path=csv_path, audio_root=root)

        weights = dataset.get_sample_weights()
        assert len(weights) == len(dataset)
        assert weights.dtype == torch.float

    def test_unknown_label_raises(self):
        """Unknown label in CSV -> error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "bad.csv")
            pd.DataFrame({
                "filepath": ["fake.wav"],
                "label": ["unknown_surface"],
            }).to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="Unknown labels"):
                AudioMelDataset(csv_path=csv_path, audio_root=tmpdir)

    def test_missing_column_raises(self):
        """Missing column in CSV → error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "bad.csv")
            pd.DataFrame({
                "file": ["fake.wav"],
                "label": ["dry_asphalt"],
            }).to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="missing columns"):
                AudioMelDataset(csv_path=csv_path, audio_root=tmpdir)

    def test_mfcc_mode(self, sample_dataset):
        """MFCC mode is working."""
        csv_path, root = sample_dataset
        dataset = AudioMelDataset(
            csv_path=csv_path, audio_root=root,
            use_mfcc=True, n_mfcc=20,
            sample_rate=16000, duration_sec=1.0,
        )

        features, _ = dataset[0]
        assert features.shape[1] == 20