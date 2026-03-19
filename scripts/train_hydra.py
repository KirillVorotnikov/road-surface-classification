#!/usr/bin/env python3
"""Hydra-based training script.

Backward-compatible wrapper for train_config.yaml.

Usage:
    python scripts/train_hydra.py --config-name train_config
    python scripts/train_hydra.py --config-name train_config model.name=resnet34
    python scripts/train_hydra.py --multirun model.name=resnet18,resnet34
"""

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.seed import set_seed
from src.core.trainer import Trainer
from src.core.logger import create_logger, MlflowLogger
from src.core.losses import create_criterion
from src.core.callbacks import EarlyStopping, ModelCheckpoint
from src.audio.models.factory import create_audio_model
from src.audio.data.datamodule import create_audio_dataloaders

console = Console()


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_config",
)
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra config.

    Args:
        cfg: Hydra configuration.
    """
    console.print("\n[cyan bold]Configuration:[/cyan bold]")
    console.print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    if cfg.mlflow.get("enabled", True) and cfg.mlflow.get("tracking_uri"):
        logger = MlflowLogger(
            tracking_uri=cfg.mlflow.tracking_uri or None,
            experiment_name=cfg.mlflow.experiment_name,
            artifact_location=cfg.mlflow.get("artifact_location"),
            run_name=cfg.experiment_name,
        )
    else:
        from src.core.logger import FileLogger
        logger = FileLogger(run_name=cfg.experiment_name)

    try:
        flat = OmegaConf.to_container(cfg, resolve=True)
        logger.log_params(flat)
    except Exception:
        pass

    console.print("\n[cyan]Creating data loaders...[/cyan]")
    try:
        train_loader, val_loader = create_audio_dataloaders(cfg)
    except Exception as e:
        console.print(f"[red]Data error: {e}[/red]")
        console.print("[yellow]Run 'dvc pull' to download data[/yellow]")
        logger.finish()
        return

    console.print(f"  Train: {len(train_loader.dataset)} samples")
    console.print(f"  Val:   {len(val_loader.dataset)} samples")

    model = create_audio_model(cfg)
    console.print(f"  Model: {cfg.model.name}")
    console.print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs,
    )

    criterion = torch.nn.CrossEntropyLoss()

    monitor = cfg.checkpoint.get("monitor", "val/loss")
    mode = cfg.checkpoint.get("mode", "min")

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=cfg.training.patience,
        ),
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_dir=cfg.checkpoint.get("dir", "checkpoints"),
            save_top_k=cfg.checkpoint.get("save_top_k", 3),
        ),
    ]

    if not OmegaConf.select(cfg, "project.classes", default=None):
        from src.audio.data.dataset import AudioMelDataset
        OmegaConf.update(cfg, "project.classes", AudioMelDataset.CLASS_NAMES)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=cfg,
        logger=logger,
        scheduler=scheduler,
        callbacks=callbacks,
    )

    best_score = trainer.train()

    if cfg.mlflow.get("log_model", True):
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_checkpoint_path:
                try:
                    logger.log_artifact(cb.best_checkpoint_path)
                except Exception:
                    pass

    logger.finish()
    console.print(f"\n[bold green]Done. Best score: {best_score:.4f}[/bold green]")


if __name__ == "__main__":
    main()