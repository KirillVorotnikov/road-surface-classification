"""Pretrained backbone classifier for audio spectrograms.

Uses ImageNet-pretrained CNNs (via timm) on mel spectrograms
treated as single-channel images. timm handles in_chans
adaptation automatically.
"""

import torch
import torch.nn as nn
import timm

from src.core.registry import MODEL_REGISTRY
from .heads import MLPHead


@MODEL_REGISTRY.register("audio_pretrained_cnn")
class AudioPretrainedClassifier(nn.Module):
    """Pretrained CNN backbone + classification head.

    Supports any timm model as backbone. The first conv layer
    is adapted for single-channel input (spectrogram).

    Input:  (batch, 1, n_mels, time_steps)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        in_channels: int = 1,
        head_hidden_dim: int = 256,
    ):
        """
        Args:
            backbone: timm model name (e.g. efficientnet_b0, resnet18).
            num_classes: Number of output classes.
            pretrained: Use ImageNet pretrained weights.
            dropout: Dropout rate in classification head.
            in_channels: Input channels (1 for mono spectrogram).
            head_hidden_dim: Hidden dimension in MLP head.
        """
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
        )

        feature_dim = self.backbone.num_features

        self.head = MLPHead(
            in_features=feature_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x):
        """Extract features without classification (for fusion)."""
        return self.backbone(x)


@MODEL_REGISTRY.register("audio_resnet")
class AudioResNetClassifier(nn.Module):
    """ResNet classifier using torchvision (no timm dependency).

    Compatible with existing train_config.yaml which uses
    torchvision ResNet models.

    Input:  (batch, 1, n_mels, time_steps)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()

        from torchvision import models

        weights_map = {
            "resnet18": ("resnet18", "IMAGENET1K_V1"),
            "resnet34": ("resnet34", "IMAGENET1K_V1"),
            "resnet50": ("resnet50", "IMAGENET1K_V1"),
        }

        if backbone not in weights_map:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Available: {list(weights_map.keys())}"
            )

        model_fn_name, weights = weights_map[backbone]
        model_fn = getattr(models, model_fn_name)
        self.backbone = model_fn(weights=weights if pretrained else None)

        # Adapt first conv for single-channel input
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )

        # Replace classifier
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = MLPHead(
            in_features=feature_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=dropout,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x):
        """Extract features without classification (for fusion)."""
        return self.backbone(x)