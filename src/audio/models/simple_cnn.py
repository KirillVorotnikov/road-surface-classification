"""Simple CNN for mel spectrogram classification. Baseline model."""

import torch.nn as nn

from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("audio_simple_cnn")
class AudioSimpleCNN(nn.Module):
    """4-block CNN for spectrograms.

    Input:  (batch, 1, n_mels, time_steps)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 128, T) -> (32, 64, T/2)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: -> (64, 32, T/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: -> (128, 16, T/8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: -> (256, 1, 1)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def get_features(self, x):
        """Extract features without classification (for fusion)."""
        x = self.features(x)
        return x.view(x.size(0), -1)  # (batch, 256)
