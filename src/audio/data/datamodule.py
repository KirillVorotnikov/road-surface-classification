"""DataLoader creation for audio classification.

Supports both Hydra config (train_config.yaml) and
standalone OmegaConf configs.
"""

from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import AudioMelDataset


def create_audio_dataloaders(
    config,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders from config.

    Supports two config styles:

    Style 1 (standalone audio config):
        data.params.train_csv, data.params.audio_root, etc.

    Style 2 (train_config.yaml / Hydra):
        data.processed_dir, model.audio.*, training.*, etc.

    Args:
        config: OmegaConf config object.

    Returns:
        (train_loader, val_loader)
    """
    # Resolve config style
    if hasattr(config, "data") and hasattr(config.data, "params"):
        # Style 1: standalone audio config
        return _create_from_audio_config(config)
    else:
        # Style 2: Hydra train_config
        return _create_from_hydra_config(config)


def _create_from_audio_config(config) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders from standalone audio config."""
    data_cfg = config.data.params

    common_kwargs = dict(
        sample_rate=data_cfg.get("sample_rate", 16000),
        duration_sec=data_cfg.get("duration_sec", 5.0),
        n_mels=data_cfg.get("n_mels", 128),
        n_fft=data_cfg.get("n_fft", 2048),
        hop_length=data_cfg.get("hop_length", 512),
        use_mfcc=data_cfg.get("use_mfcc", False),
        n_mfcc=data_cfg.get("n_mfcc", 40),
    )

    train_dataset = AudioMelDataset(
        csv_path=data_cfg.train_csv,
        audio_root=data_cfg.audio_root,
        augmentation_preset=config.data.get("augmentations", "medium"),
        **common_kwargs,
    )

    val_dataset = AudioMelDataset(
        csv_path=data_cfg.val_csv,
        audio_root=data_cfg.audio_root,
        augmentation_preset=None,
        **common_kwargs,
    )

    return _build_loaders(train_dataset, val_dataset, config)


def _create_from_hydra_config(config) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders from Hydra train_config.yaml.

    Expects:
        data.processed_dir: directory with clips/ and CSVs
        model.audio.*: audio params (sample_rate, n_mels, etc.)
    """
    processed_dir = config.data.processed_dir
    audio_cfg = config.model.audio

    common_kwargs = dict(
        audio_root=processed_dir,
        sample_rate=audio_cfg.get("sample_rate", 16000),
        n_mels=audio_cfg.get("n_mels", 128),
        n_fft=audio_cfg.get("n_fft", 2048),
        hop_length=audio_cfg.get("hop_length", 512),
        duration_sec=config.training.get("clip_duration", 5.0),
    )

    aug_preset = "medium" if config.augmentations.get("enabled", True) else None

    train_dataset = AudioMelDataset(
        csv_path=f"{processed_dir}/train.csv",
        augmentation_preset=aug_preset,
        **common_kwargs,
    )

    val_dataset = AudioMelDataset(
        csv_path=f"{processed_dir}/val.csv",
        augmentation_preset=None,
        **common_kwargs,
    )

    return _build_loaders(train_dataset, val_dataset, config)


def _build_loaders(
    train_dataset: AudioMelDataset,
    val_dataset: AudioMelDataset,
    config,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders with optional weighted sampling."""
    batch_size = config.training.get("batch_size", 32)
    num_workers = config.get("validation", config.get("training", {})).get(
        "num_workers", 4
    )
    use_weighted = config.training.get("use_weighted_sampler", False)

    train_sampler = None
    train_shuffle = True

    if use_weighted:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader