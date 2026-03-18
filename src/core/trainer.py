"""A universal trainer for any modality."""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from .metrics import compute_metrics
from .callbacks import TrainingCallback, EarlyStopping, ModelCheckpoint
from .logger import BaseLogger
from .device import get_device

console = Console()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Any,
        logger: BaseLogger,
        scheduler=None,
        callbacks: list[TrainingCallback] | None = None,
        class_names: list[str] | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.logger = logger
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.class_names = class_names or config.project.classes
        self.device = get_device()

        self.model.to(self.device)

        for cb in self.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.set_model(self.model)

    def train(self) -> float:
        """Full learning cycle. Returns the best score."""
        num_params = sum(p.numel() for p in self.model.parameters())
        console.print(f"\n[bold blue]Training on {self.device}[/bold blue]")
        console.print(f"Parameters: {num_params:,}")
        console.print(f"Train: {len(self.train_loader.dataset)} samples")
        console.print(f"Val: {len(self.val_loader.dataset)} samples\n")

        # on_train_start
        for cb in self.callbacks:
            cb.on_train_start(self.logger, self.config)

        best_score = 0.0

        for epoch in range(self.config.training.epochs):
            # on_epoch_start
            for cb in self.callbacks:
                cb.on_epoch_start(self.logger, epoch)

            epoch_start = time.time()

            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()

            epoch_time = time.time() - epoch_start

            # We collect all metrics with prefixes
            all_metrics = {
                **{f"train/{k}": v for k, v in train_metrics.items()
                   if not isinstance(v, dict)},
                **{f"val/{k}": v for k, v in val_metrics.items()
                   if not isinstance(v, dict)},
                "epoch_time": epoch_time,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            # Per-class metrics
            if "f1_per_class" in val_metrics:
                for cls_name, f1_val in val_metrics["f1_per_class"].items():
                    all_metrics[f"val/f1_class/{cls_name}"] = f1_val

            self.logger.log_metrics(all_metrics, step=epoch)

            self._print_epoch(epoch, train_metrics, val_metrics, epoch_time)

            # Scheduler
            if self.scheduler:
                self.scheduler.step()

            # Callbacks (on_epoch_end)
            for cb in self.callbacks:
                cb.on_epoch_end(self.logger, epoch, all_metrics)

            # Best score
            score = val_metrics.get("balanced_accuracy", 0.0)
            if score > best_score:
                best_score = score

            # Early stopping
            for cb in self.callbacks:
                if isinstance(cb, EarlyStopping) and cb.should_stop:
                    console.print(f"\n[bold red]Stopped at epoch {epoch}[/bold red]")
                    # on_train_end
                    for cb2 in self.callbacks:
                        cb2.on_train_end(self.logger)
                    return best_score

        # on_train_end
        for cb in self.callbacks:
            cb.on_train_end(self.logger)

        console.print(f"\n[bold green]Best balanced accuracy: {best_score:.4f}[/bold green]")
        return best_score

    def _train_epoch(self) -> dict:
        self.model.train()
        running_loss = 0.0
        all_preds, all_targets = [], []

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

        metrics = compute_metrics(all_targets, all_preds, self.class_names)
        metrics["loss"] = running_loss / len(self.train_loader)
        return metrics

    def _validate_epoch(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        metrics = compute_metrics(all_targets, all_preds, self.class_names)
        metrics["loss"] = running_loss / len(self.val_loader)
        return metrics

    def _print_epoch(self, epoch, train_metrics, val_metrics, epoch_time):
        table = Table(title=f"Epoch {epoch + 1}/{self.config.training.epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="green")
        table.add_column("Val", style="yellow")

        table.add_row(
            "Loss",
            f"{train_metrics['loss']:.4f}",
            f"{val_metrics['loss']:.4f}",
        )
        table.add_row(
            "Balanced Acc",
            f"{train_metrics['balanced_accuracy']:.4f}",
            f"{val_metrics['balanced_accuracy']:.4f}",
        )
        table.add_row(
            "F1 Macro",
            f"{train_metrics['f1_macro']:.4f}",
            f"{val_metrics['f1_macro']:.4f}",
        )
        table.add_row("Time", f"{epoch_time:.1f}s", "")

        console.print(table)