"""Creating DataLoaders for Audio."""

from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import AudioMelDataset


def create_audio_dataloaders(
    config,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates train and val DataLoaders from the config.

    Args:
        config: OmegaConf конфиг с секциями data и training.

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = config.data.params

    train_dataset = AudioMelDataset(
        csv_path=data_cfg.train_csv,
        audio_root=data_cfg.audio_root,
        sample_rate=data_cfg.get("sample_rate", 16000),
        duration_sec=data_cfg.get("duration_sec", 5.0),
        n_mels=data_cfg.get("n_mels", 128),
        n_fft=data_cfg.get("n_fft", 2048),
        hop_length=data_cfg.get("hop_length", 512),
        use_mfcc=data_cfg.get("use_mfcc", False),
        n_mfcc=data_cfg.get("n_mfcc", 40),
        augmentation_preset=config.data.get("augmentations", "medium"),
    )

    val_dataset = AudioMelDataset(
        csv_path=data_cfg.val_csv,
        audio_root=data_cfg.audio_root,
        sample_rate=data_cfg.get("sample_rate", 16000),
        duration_sec=data_cfg.get("duration_sec", 5.0),
        n_mels=data_cfg.get("n_mels", 128),
        n_fft=data_cfg.get("n_fft", 2048),
        hop_length=data_cfg.get("hop_length", 512),
        use_mfcc=data_cfg.get("use_mfcc", False),
        n_mfcc=data_cfg.get("n_mfcc", 40),
        augmentation_preset=None,
    )

    # Weighted sampler for train (combating imbalance)
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
        batch_size=config.training.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.training.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader