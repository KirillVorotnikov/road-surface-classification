"""Tests for audio models."""

import pytest
import torch
import torch.nn as nn

from src.audio.models.classifier import AudioPretrainedClassifier, AudioResNetClassifier
from src.audio.models.factory import create_audio_model
from src.audio.models.heads import LinearHead, MLPHead
from src.audio.models.simple_cnn import AudioSimpleCNN
from src.core.registry import MODEL_REGISTRY


@pytest.fixture
def mel_batch():
    """Batch of mel spectrograms: (batch=4, channels=1, n_mels=128, time=87)."""
    return torch.randn(4, 1, 128, 87)


@pytest.fixture
def small_mel_batch():
    """Smaller batch for faster pretrained model tests."""
    return torch.randn(2, 1, 64, 44)


class TestHeads:
    """Tests for classification heads."""

    def test_linear_head(self):
        head = LinearHead(in_features=256, num_classes=5)
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 5)

    def test_mlp_head(self):
        head = MLPHead(in_features=512, num_classes=5, hidden_dim=128)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 5)

    def test_mlp_head_custom_dropout(self):
        head = MLPHead(in_features=256, num_classes=3, dropout=0.5)
        x = torch.randn(2, 256)
        out = head(x)
        assert out.shape == (2, 3)


class TestSimpleCNN:
    """Tests for AudioSimpleCNN."""

    def test_forward_shape(self, mel_batch):
        model = AudioSimpleCNN(num_classes=5)
        out = model(mel_batch)
        assert out.shape == (4, 5)

    def test_different_num_classes(self, mel_batch):
        model = AudioSimpleCNN(num_classes=10)
        out = model(mel_batch)
        assert out.shape == (4, 10)

    def test_get_features(self, mel_batch):
        model = AudioSimpleCNN(num_classes=5)
        features = model.get_features(mel_batch)
        assert features.shape == (4, 256)

    def test_variable_time_steps(self):
        """Model handles different time dimensions."""
        model = AudioSimpleCNN(num_classes=5)
        for time_steps in [44, 87, 157, 200]:
            x = torch.randn(2, 1, 128, time_steps)
            out = model(x)
            assert out.shape == (2, 5)

    def test_variable_n_mels(self):
        """Model handles different n_mels."""
        model = AudioSimpleCNN(num_classes=5)
        for n_mels in [32, 64, 128, 256]:
            x = torch.randn(2, 1, n_mels, 87)
            out = model(x)
            assert out.shape == (2, 5)

    def test_gradient_flow(self, mel_batch):
        """Gradients flow through entire model."""
        model = AudioSimpleCNN(num_classes=5)
        out = model(mel_batch)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestPretrainedClassifier:
    """Tests for AudioPretrainedClassifier (timm)."""

    def test_forward_shape(self, small_mel_batch):
        model = AudioPretrainedClassifier(
            backbone="efficientnet_b0",
            num_classes=5,
            pretrained=False,
            in_channels=1,
        )
        out = model(small_mel_batch)
        assert out.shape == (2, 5)

    def test_get_features(self, small_mel_batch):
        model = AudioPretrainedClassifier(
            backbone="efficientnet_b0",
            num_classes=5,
            pretrained=False,
            in_channels=1,
        )
        features = model.get_features(small_mel_batch)
        assert features.dim() == 2
        assert features.shape[0] == 2

    def test_mobilenet_backbone(self, small_mel_batch):
        model = AudioPretrainedClassifier(
            backbone="mobilenetv3_small_100",
            num_classes=5,
            pretrained=False,
            in_channels=1,
        )
        out = model(small_mel_batch)
        assert out.shape == (2, 5)

    def test_custom_head_dim(self, small_mel_batch):
        model = AudioPretrainedClassifier(
            backbone="efficientnet_b0",
            num_classes=5,
            pretrained=False,
            head_hidden_dim=512,
        )
        out = model(small_mel_batch)
        assert out.shape == (2, 5)


class TestResNetClassifier:
    """Tests for AudioResNetClassifier (torchvision)."""

    def test_resnet18_forward(self, small_mel_batch):
        model = AudioResNetClassifier(
            backbone="resnet18",
            num_classes=5,
            pretrained=False,
        )
        out = model(small_mel_batch)
        assert out.shape == (2, 5)

    def test_resnet34_forward(self, small_mel_batch):
        model = AudioResNetClassifier(
            backbone="resnet34",
            num_classes=5,
            pretrained=False,
        )
        out = model(small_mel_batch)
        assert out.shape == (2, 5)

    def test_resnet50_forward(self, small_mel_batch):
        model = AudioResNetClassifier(
            backbone="resnet50",
            num_classes=5,
            pretrained=False,
        )
        out = model(small_mel_batch)
        assert out.shape == (2, 5)

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            AudioResNetClassifier(backbone="resnet999")

    def test_get_features(self, small_mel_batch):
        model = AudioResNetClassifier(
            backbone="resnet18",
            num_classes=5,
            pretrained=False,
        )
        features = model.get_features(small_mel_batch)
        assert features.dim() == 2
        assert features.shape[0] == 2


class TestModelRegistry:
    """Tests for model registration."""

    def test_simple_cnn_registered(self):
        assert "audio_simple_cnn" in MODEL_REGISTRY.list()

    def test_pretrained_cnn_registered(self):
        assert "audio_pretrained_cnn" in MODEL_REGISTRY.list()

    def test_resnet_registered(self):
        assert "audio_resnet" in MODEL_REGISTRY.list()

    def test_create_from_registry(self):
        model = MODEL_REGISTRY.create(
            "audio_simple_cnn",
            num_classes=5,
            in_channels=1,
        )
        assert isinstance(model, nn.Module)
        x = torch.randn(2, 1, 128, 87)
        out = model(x)
        assert out.shape == (2, 5)


class TestFactory:
    """Tests for create_audio_model factory."""

    def test_create_by_registry_key(self):
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "model": {
                    "name": "audio_simple_cnn",
                    "num_classes": 5,
                    "params": {"in_channels": 1, "dropout": 0.2},
                },
            }
        )
        model = create_audio_model(config)
        assert isinstance(model, AudioSimpleCNN)

    def test_create_resnet_by_short_name(self):
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "model": {
                    "name": "resnet18",
                    "num_classes": 5,
                    "pretrained": False,
                    "dropout": 0.2,
                },
            }
        )
        model = create_audio_model(config)
        assert isinstance(model, AudioResNetClassifier)

    def test_create_timm_model_by_name(self):
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "model": {
                    "name": "mobilenetv3_small_100",
                    "num_classes": 5,
                    "pretrained": False,
                    "dropout": 0.2,
                },
            }
        )
        model = create_audio_model(config)
        assert isinstance(model, AudioPretrainedClassifier)

    def test_factory_output_shape(self):
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "model": {
                    "name": "audio_simple_cnn",
                    "num_classes": 3,
                    "params": {},
                },
            }
        )
        model = create_audio_model(config)
        x = torch.randn(2, 1, 128, 87)
        out = model(x)
        assert out.shape == (2, 3)


class TestModelSize:
    """Verify models are reasonable size."""

    def test_simple_cnn_params(self):
        model = AudioSimpleCNN(num_classes=5)
        params = sum(p.numel() for p in model.parameters())
        assert params < 5_000_000

    def test_resnet18_params(self):
        model = AudioResNetClassifier(
            backbone="resnet18",
            num_classes=5,
            pretrained=False,
        )
        params = sum(p.numel() for p in model.parameters())
        assert params < 15_000_000
