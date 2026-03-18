"""Training callbacks.

- EarlyStopping: stop when metric stops improving
- ModelCheckpoint: save best model checkpoints
"""

import os
from typing import Literal

import torch
from rich.console import Console

from .logger import BaseLogger

console = Console()


class TrainingCallback:
    """Basic callback with empty hooks."""

    def on_train_start(self, logger: BaseLogger, config: dict) -> None:
        pass

    def on_train_end(self, logger: BaseLogger) -> None:
        pass

    def on_epoch_start(self, logger: BaseLogger, epoch: int) -> None:
        pass

    def on_epoch_end(
        self, logger: BaseLogger, epoch: int, metrics: dict[str, float]
    ) -> None:
        pass


class EarlyStopping(TrainingCallback):
    """Stop if there is no improvement."""

    def __init__(
        self,
        monitor: str = "val/balanced_accuracy",
        mode: Literal["min", "max"] = "max",
        patience: int = 10,
        min_delta: float = 0.001,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best_value: float | None = None
        self._counter = 0
        self._should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.mode == "min":
            return value < self._best_value - self.min_delta
        return value > self._best_value + self.min_delta

    def on_epoch_end(self, logger, epoch, metrics):
        value = metrics.get(self.monitor)
        if value is None:
            return

        if self._is_improvement(value):
            self._best_value = value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._should_stop = True
                console.print(
                    f"[red]Early stopping at epoch {epoch}: "
                    f"no improvement for {self.patience} epochs[/red]"
                )

    @property
    def should_stop(self) -> bool:
        return self._should_stop


class ModelCheckpoint(TrainingCallback):
    """Saving the best checkpoints."""

    def __init__(
        self,
        monitor: str = "val/balanced_accuracy",
        mode: Literal["min", "max"] = "max",
        save_dir: str = "checkpoints",
        save_top_k: int = 3,
    ):
        self.monitor = monitor
        self.mode = mode
        self.save_dir = save_dir
        self.save_top_k = save_top_k
        self._best_value: float | None = None
        self._saved: list[tuple[float, str]] = []
        self._model: torch.nn.Module | None = None

        os.makedirs(save_dir, exist_ok=True)

    def set_model(self, model: torch.nn.Module) -> None:
        """Called from Trainer once during initialization."""
        self._model = model

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def on_epoch_end(self, logger, epoch, metrics):
        value = metrics.get(self.monitor)
        if value is None or self._model is None:
            return

        if self._is_improvement(value):
            self._best_value = value

            filename = f"epoch_{epoch:03d}_{value:.4f}.pt"
            filepath = os.path.join(self.save_dir, filename)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self._model.state_dict(),
                    self.monitor: value,
                },
                filepath,
            )
            self._saved.append((value, filepath))

            console.print(
                f"[green]Checkpoint saved: {filename} "
                f"({self.monitor}={value:.4f})[/green]"
            )

            logger.log_artifact(filepath)

            self._cleanup()

    def _cleanup(self):
        reverse = self.mode == "max"
        self._saved.sort(key=lambda x: x[0], reverse=reverse)
        while len(self._saved) > self.save_top_k:
            _, path = self._saved.pop()
            if os.path.exists(path):
                os.remove(path)

    @property
    def best_value(self) -> float | None:
        return self._best_value

    @property
    def best_checkpoint_path(self) -> str | None:
        if not self._saved:
            return None
        return self._saved[0][1]