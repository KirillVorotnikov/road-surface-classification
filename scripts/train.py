#!/usr/bin/env python3
"""Training script for Road Surface Classification.

Supports standalone audio configs and Hydra-style configs.

Usage:
    # Standalone audio config
    python scripts/train.py --config configs/audio/models/simple_cnn.yaml

    # Override parameters
    python scripts/train.py --config configs/audio/models/resnet18_mel.yaml \
        --override training.lr=0.0005 training.epochs=100

    # Use file logger (no MLflow server needed)
    python scripts/train.py --config configs/audio/models/simple_cnn.yaml \
        --logger file
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.data.datamodule import create_audio_dataloaders

# Importing registers models and datasets
from src.audio.models.factory import create_audio_model
from src.core.callbacks import EarlyStopping, ModelCheckpoint
from src.core.config import load_config
from src.core.logger import create_logger
from src.core.losses import create_criterion
from src.core.seed import set_seed
from src.core.trainer import Trainer

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train road surface classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with SimpleCNN
    python scripts/train.py --config configs/audio/models/simple_cnn.yaml

    # Train with ResNet18
    python scripts/train.py --config configs/audio/models/resnet18_mel.yaml

    # Override hyperparameters
    python scripts/train.py --config configs/audio/models/simple_cnn.yaml \\
        --override training.lr=0.0005 training.batch_size=64

    # Local development (no MLflow)
    python scripts/train.py --config configs/audio/models/simple_cnn.yaml \\
        --logger file
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in dot notation (e.g. training.lr=0.001)",
    )
    parser.add_argument(
        "--logger",
        type=str,
        choices=["mlflow", "file"],
        default=None,
        help="Override logger type (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps (default: auto)",
    )

    return parser.parse_args()


def apply_overrides(config: DictConfig, overrides: list[str]) -> DictConfig:
    """Apply CLI overrides to config.

    Args:
        config: Base config.
        overrides: List of "key=value" strings in dot notation.

    Returns:
        Config with overrides applied.
    """
    for override in overrides:
        if "=" not in override:
            console.print(f"[yellow]Skipping invalid override: {override}[/yellow]")
            continue

        key, value = override.split("=", 1)

        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False

        OmegaConf.update(config, key, value, merge=True)

    return config


def create_optimizer(
    model: torch.nn.Module, config: DictConfig
) -> torch.optim.Optimizer:
    """Create optimizer from config.

    Args:
        model: Model to optimize.
        config: Training config section.

    Returns:
        Optimizer instance.
    """
    training_cfg = config.training
    name = training_cfg.get("optimizer", "AdamW").lower()
    lr = training_cfg.get("lr", training_cfg.get("learning_rate", 1e-4))
    weight_decay = training_cfg.get("weight_decay", 0.01)

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        momentum = training_cfg.get("momentum", 0.9)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DictConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create LR scheduler from config.

    Args:
        optimizer: Optimizer instance.
        config: Training config section.

    Returns:
        Scheduler or None.
    """
    training_cfg = config.training
    name = training_cfg.get("scheduler", "cosine").lower()
    epochs = training_cfg.get("epochs", 50)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    elif name == "step":
        step_size = training_cfg.get("step_size", 10)
        gamma = training_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif name == "none" or name == "":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def create_callbacks(config: DictConfig) -> list:
    """Create training callbacks from config.

    Args:
        config: Full config.

    Returns:
        List of TrainingCallback instances.
    """
    training_cfg = config.training
    callbacks = []

    # Early stopping
    patience = training_cfg.get(
        "early_stopping_patience",
        training_cfg.get("patience", 10),
    )
    monitor_es = config.get("checkpoint", {}).get("monitor", "val/balanced_accuracy")
    mode_es = config.get("checkpoint", {}).get("mode", "max")

    if "loss" in monitor_es:
        mode_es = "min"

    callbacks.append(
        EarlyStopping(
            monitor=monitor_es,
            mode=mode_es,
            patience=patience,
            min_delta=0.001,
        )
    )

    # Model checkpoint
    checkpoint_dir = config.get("checkpoint", {}).get(
        "dir",
        config.get("checkpoint_dir", "checkpoints"),
    )
    save_top_k = config.get("checkpoint", {}).get("save_top_k", 3)

    callbacks.append(
        ModelCheckpoint(
            monitor=monitor_es,
            mode=mode_es,
            save_dir=checkpoint_dir,
            save_top_k=save_top_k,
        )
    )

    return callbacks


def print_config_summary(config: DictConfig) -> None:
    """Print config summary to console."""
    console.print("\n[bold cyan]Configuration Summary[/bold cyan]")
    console.print(f"  Experiment:  {config.get('experiment_name', 'unnamed')}")
    console.print(f"  Model:       {config.model.name}")
    console.print(
        f"  Classes:     {config.get('project', {}).get('num_classes', config.model.get('num_classes', '?'))}"
    )

    training = config.training
    console.print(f"  Epochs:      {training.get('epochs', '?')}")
    console.print(f"  Batch size:  {training.get('batch_size', '?')}")
    console.print(
        f"  LR:          {training.get('lr', training.get('learning_rate', '?'))}"
    )
    console.print(f"  Optimizer:   {training.get('optimizer', '?')}")
    console.print(f"  Loss:        {training.get('loss', 'cross_entropy')}")
    console.print(f"  Logger:      {config.get('logging', {}).get('tool', 'file')}")
    console.print()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    console.print(f"\n[cyan]Loading config: {args.config}[/cyan]")
    config = load_config(args.config)

    if args.override:
        config = apply_overrides(config, args.override)

    if args.logger:
        if "logging" not in config:
            config.logging = {}
        OmegaConf.update(config, "logging.tool", args.logger)

    print_config_summary(config)

    seed = config.get("project", config).get("seed", 42)
    set_seed(seed)
    console.print(f"[cyan]Seed: {seed}[/cyan]")

    logger = create_logger(config)

    try:
        flat_config = OmegaConf.to_container(config, resolve=True)
        logger.log_params(flat_config)
    except Exception as e:
        console.print(f"[yellow]Could not log params: {e}[/yellow]")

    console.print("[cyan]Creating data loaders...[/cyan]")
    try:
        train_loader, val_loader = create_audio_dataloaders(config)
        console.print(f"  Train: {len(train_loader.dataset)} samples")
        console.print(f"  Val:   {len(val_loader.dataset)} samples")
    except Exception as e:
        console.print(f"[red]Error creating data loaders: {e}[/red]")
        console.print(
            "[yellow]Check that data exists (dvc pull) and CSV paths are correct[/yellow]"
        )
        logger.finish()
        sys.exit(1)

    console.print("[cyan]Creating model...[/cyan]")
    model = create_audio_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Total params:     {num_params:,}")
    console.print(f"  Trainable params: {trainable_params:,}")

    optimizer = create_optimizer(model, config)

    scheduler = create_scheduler(optimizer, config)

    criterion = create_criterion(config)

    callbacks = create_callbacks(config)

    class_names = config.get("project", {}).get("classes", None)
    if class_names is None:
        from src.audio.data.dataset import AudioMelDataset

        class_names = AudioMelDataset.CLASS_NAMES

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        logger=logger,
        scheduler=scheduler,
        callbacks=callbacks,
        class_names=class_names,
    )

    best_score = trainer.train()

    # Log best model
    try:
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_checkpoint_path:
                logger.log_artifact(cb.best_checkpoint_path)
                console.print(
                    f"\n[green]Best checkpoint: {cb.best_checkpoint_path}[/green]"
                )
    except Exception as e:
        console.print(f"[yellow]Could not log model artifact: {e}[/yellow]")

    logger.finish()

    console.print(
        f"\n[bold green]Training complete. Best score: {best_score:.4f}[/bold green]"
    )


if __name__ == "__main__":
    main()
