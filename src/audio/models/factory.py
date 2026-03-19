"""Audio model factory.

Creates models from config using the global MODEL_REGISTRY.
Importing this module registers all audio models.
"""

import torch.nn as nn

from src.core.registry import MODEL_REGISTRY

from .classifier import (  # noqa: F401
    AudioPretrainedClassifier,
    AudioResNetClassifier,
)

# Importing registers models in MODEL_REGISTRY
from .simple_cnn import AudioSimpleCNN  # noqa: F401


def create_audio_model(config) -> nn.Module:
    """Create audio model from config.

    Supports two config styles:

    Style 1 (standalone audio config):
        model.name: registry key (e.g. "audio_simple_cnn")
        model.params: dict of constructor kwargs

    Style 2 (train_config.yaml / Hydra):
        model.name: short name (e.g. "resnet18")
        model.num_classes, model.pretrained, model.dropout

    Args:
        config: OmegaConf config.

    Returns:
        Instantiated nn.Module.
    """
    model_name = config.model.name

    # Check if model_name is a registry key
    if model_name in MODEL_REGISTRY.list():
        return MODEL_REGISTRY.create(
            model_name,
            num_classes=config.model.get("num_classes", 5),
            **config.model.get("params", {}),
        )

    # Fallback: treat as backbone name (e.g. "resnet18")
    # for compatibility with train_config.yaml
    if "resnet" in model_name.lower():
        return AudioResNetClassifier(
            backbone=model_name,
            num_classes=config.model.get("num_classes", 5),
            pretrained=config.model.get("pretrained", True),
            dropout=config.model.get("dropout", 0.2),
        )

    # Try as timm model
    return AudioPretrainedClassifier(
        backbone=model_name,
        num_classes=config.model.get("num_classes", 5),
        pretrained=config.model.get("pretrained", True),
        dropout=config.model.get("dropout", 0.2),
    )
