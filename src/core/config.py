from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import List


@dataclass
class ProjectConfig:
    name: str = 'road-surface-classification'
    classes: List[str] = field(default_factory=lambda: [
        "dry_asphalt", "wet_asphalt", "snow", "ice", "gravel"
    ])
    num_classes: int = 5
    seed: int = 42

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 32
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    early_stopping_patience: int = 10
    label_smoothing: float = 0.1
    num_workers: int = 4

@dataclass
class LoggingConfig:
    tool : str = 'mlflow'
    project: str = "road-surface-classification"
    log_every_n_steps: int = 10
    save_top_k: int = 5

def load_config(config_path: str) -> DictConfig:
    """Load config from a yaml file."""
    config = OmegaConf.load(config_path)
    return config

def merge_configs(base_path: str, override_path: str) -> DictConfig:
    """Merge configs from a yaml file."""
    base_config = OmegaConf.load(base_path)
    override_config = OmegaConf.load(override_path)
    config = OmegaConf.merge(base_config, override_config)
    return config