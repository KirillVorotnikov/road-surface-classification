#!/usr/bin/env python3
"""Training script for Road Surface Classification.

Supports:
- Hydra configuration management
- MLflow experiment tracking
- DVC data management

Usage:
    python scripts/train.py --config-name train_config
    python scripts/train.py --config-name train_config model.name=resnet34
    python scripts/train.py --multirun \
        model.name=resnet18,resnet34 training.lr=0.001,0.01
"""

import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.callbacks import EarlyStopping, ModelCheckpoint
from src.core.logger import MlflowLogger


def create_model(cfg: DictConfig) -> nn.Module:
    """Create model from configuration.

    Args:
        cfg: Model configuration.

    Returns:
        PyTorch model.
    """
    model_name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    rprint(f"[cyan]Creating model: {model_name} (classes={num_classes})")

    # Simple ResNet-based classifier for audio spectrograms
    if "resnet" in model_name.lower():
        from torchvision import models

        if model_name == "resnet18":
            backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        elif model_name == "resnet34":
            backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        elif model_name == "resnet50":
            backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Modify first layer for single-channel audio (spectrogram)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify classifier for our number of classes
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(cfg.model.dropout),
            nn.Linear(in_features, num_classes),
        )

        return backbone
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every_n: int = 10,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        log_every_n: Log metrics every n batches.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [train]", leave=False)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        if hasattr(model, "grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_every_n == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model.

    Args:
        model: The model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="[val]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_config",
)
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration.
    """
    # Print configuration
    rprint("\n[cyan bold]Configuration:[/cyan bold]")
    rprint(OmegaConf.to_yaml(cfg))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"\n[cyan]Device:[/cyan] {device}")

    # Create model
    model = create_model(cfg)
    model = model.to(device)

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
    )

    # Create dummy dataloaders for demonstration
    # Replace with your actual data loading logic
    rprint("\n[cyan]Creating data loaders...[/cyan]")
    # TODO: Replace with actual dataset
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 1, 128, 64),  # batch, channel, mel, time
        torch.randint(0, cfg.model.num_classes, (100,)),
    )
    train_loader = torch.utils.data.DataLoader(
        dummy_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dummy_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    rprint(f"  Train samples: {len(train_loader.dataset)}")
    rprint(f"  Val samples: {len(val_loader.dataset)}")

    # Initialize MLflow logger
    rprint("\n[cyan]Initializing MLflow logger...[/cyan]")

    # Setup early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=cfg.training.patience,
        min_delta=0.001,
    )

    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_dir="checkpoints",
        save_best_only=True,
    )
    checkpoint.set_model(model)

    with MlflowLogger(
        tracking_uri=cfg.mlflow.tracking_uri or None,
        experiment_name=cfg.mlflow.experiment_name,
        artifact_location=cfg.mlflow.artifact_location,
        run_name=cfg.experiment_name,
    ) as logger:
        # Log hyperparameters
        logger.log_params(
            {
                "model/name": cfg.model.name,
                "model/num_classes": cfg.model.num_classes,
                "training/epochs": cfg.training.epochs,
                "training/batch_size": cfg.training.batch_size,
                "training/learning_rate": cfg.training.learning_rate,
                "training/weight_decay": cfg.training.weight_decay,
            }
        )

        # Training loop
        best_val_loss = float("inf")

        for epoch in range(cfg.training.epochs):
            # Train
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                log_every_n=cfg.mlflow.log_every_n_epochs,
            )

            # Validate
            val_loss, val_acc = validate(
                model,
                val_loader,
                criterion,
                device,
            )

            # Update scheduler
            scheduler.step()

            # Log metrics
            metrics = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "training/lr": scheduler.get_last_lr()[0],
            }
            logger.log_metrics(metrics, step=epoch)

            rprint(
                f"Epoch {epoch+1}/{cfg.training.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.2f}%"
            )

            # Early stopping check
            epoch_metrics = {"val/loss": val_loss, "val/accuracy": val_acc}
            early_stopping.on_epoch_end(logger, epoch, epoch_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint.save_best_model(epoch, val_loss)

                # Save best model to MLflow
                if cfg.mlflow.log_model:
                    logger.log_model(model, artifact_name="best_model")

            if early_stopping.should_stop():
                rprint("\n[yellow]Early stopping triggered[/yellow]")
                break

        # Log final model
        if cfg.mlflow.log_model:
            logger.log_model(model, artifact_name="final_model")

        rprint("\n[green bold]Training complete![/green bold]")

    rprint("\n[cyan]Done![/cyan]")


if __name__ == "__main__":
    main()
