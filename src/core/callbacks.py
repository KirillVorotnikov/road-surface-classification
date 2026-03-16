"""Training callbacks module.

Provides callbacks for training loops:
- EarlyStopping: Stop training when metric stops improving
- ModelCheckpoint: Save model checkpoints
- TrainingCallback: Base class with hooks
"""

import os
from abc import ABC
from typing import Any, Literal

import torch

from src.core.logger import BaseLogger


class TrainingCallback(ABC):
    """Base callback for training loops.

    Provides hooks for:
    - Training start/end
    - Epoch start/end
    - Batch start/end
    """

    def on_train_start(self, logger: BaseLogger, config: dict[str, Any]) -> None:
        """Called when training starts.

        Args:
            logger: Logger instance.
            config: Training configuration.
        """
        pass

    def on_train_end(self, logger: BaseLogger) -> None:
        """Called when training ends.

        Args:
            logger: Logger instance.
        """
        pass

    def on_epoch_start(
        self,
        logger: BaseLogger,
        epoch: int,
    ) -> None:
        """Called at the start of each epoch.

        Args:
            logger: Logger instance.
            epoch: Current epoch number.
        """
        pass

    def on_epoch_end(
        self,
        logger: BaseLogger,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch.

        Args:
            logger: Logger instance.
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
        """
        pass

    def on_batch_end(
        self,
        logger: BaseLogger,
        epoch: int,
        batch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each batch.

        Args:
            logger: Logger instance.
            epoch: Current epoch number.
            batch: Current batch number.
            metrics: Dictionary of metrics.
        """
        pass


class EarlyStopping(TrainingCallback):
    """Early stopping callback.

    Stops training when a monitored metric stops improving.

    Attributes:
        monitor: Metric name to monitor (e.g., "val/loss").
        mode: "min" or "max" - whether to minimize or maximize the metric.
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.

    Example:
        ```python
        early_stop = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=5,
            min_delta=0.001,
        )
        ```
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        mode: Literal["min", "max"] = "min",
        patience: int = 5,
        min_delta: float = 0.001,
    ):
        """Initialize early stopping.

        Args:
            monitor: Metric name to monitor.
            mode: "min" or "max".
            patience: Number of epochs to wait.
            min_delta: Minimum improvement threshold.
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self._best_value: float | None = None
        self._counter = 0
        self._should_stop = False

    def _is_improvement(self, value: float) -> bool:
        """Check if value is improvement over best.

        Args:
            value: Current metric value.

        Returns:
            True if improvement, False otherwise.
        """
        if self._best_value is None:
            return True

        if self.mode == "min":
            return value < self._best_value - self.min_delta
        else:
            return value > self._best_value + self.min_delta

    def on_epoch_end(
        self,
        logger: BaseLogger,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Check for early stopping.

        Args:
            logger: Logger instance.
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
        """
        if self.monitor not in metrics:
            return

        value = metrics[self.monitor]

        if self._is_improvement(value):
            self._best_value = value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._should_stop = True

    def should_stop(self) -> bool:
        """Check if training should stop.

        Returns:
            True if early stopping triggered, False otherwise.
        """
        return self._should_stop


class ModelCheckpoint(TrainingCallback):
    """Model checkpoint callback.

    Saves model checkpoints based on monitored metric.

    Attributes:
        monitor: Metric name to monitor (e.g., "val/accuracy").
        mode: "min" or "max" - whether to minimize or maximize.
        save_dir: Directory to save checkpoints.
        save_best_only: Only save when metric improves.
        filename_pattern: Pattern for checkpoint filenames.

    Example:
        ```python
        checkpoint = ModelCheckpoint(
            monitor="val/accuracy",
            mode="max",
            save_dir="checkpoints/",
            save_best_only=True,
        )
        ```
    """

    def __init__(
        self,
        monitor: str = "val/accuracy",
        mode: Literal["min", "max"] = "max",
        save_dir: str = "checkpoints",
        save_best_only: bool = True,
        filename_pattern: str = "epoch_{epoch:03d}_{monitor:.4f}.pt",
    ):
        """Initialize model checkpoint.

        Args:
            monitor: Metric name to monitor.
            mode: "min" or "max".
            save_dir: Directory to save checkpoints.
            save_best_only: Only save best model.
            filename_pattern: Filename pattern with {epoch} and {monitor}.
        """
        self.monitor = monitor
        self.mode = mode
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.filename_pattern = filename_pattern

        self._best_value: float | None = None
        self._saved_checkpoints: list[str] = []

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def _is_improvement(self, value: float) -> bool:
        """Check if value is improvement over best.

        Args:
            value: Current metric value.

        Returns:
            True if improvement, False otherwise.
        """
        if self._best_value is None:
            return True

        if self.mode == "min":
            return value < self._best_value
        else:
            return value > self._best_value

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        value: float,
    ) -> str:
        """Save model checkpoint.

        Args:
            model: PyTorch model.
            epoch: Current epoch number.
            value: Current metric value.

        Returns:
            Path to saved checkpoint.
        """
        filename = self.filename_pattern.format(epoch=epoch, monitor=value)
        filepath = os.path.join(self.save_dir, filename)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                self.monitor: value,
            },
            filepath,
        )

        self._saved_checkpoints.append(filepath)
        return filepath

    def on_epoch_end(
        self,
        logger: BaseLogger,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save checkpoint if needed.

        Args:
            logger: Logger instance.
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
        """
        if self.monitor not in metrics:
            return

        value = metrics[self.monitor]

        if not self.save_best_only or self._is_improvement(value):
            # Save checkpoint (model should be passed from training loop)
            # This is a simplified version - full implementation needs model access
            self._best_value = value

    def set_model(self, model: torch.nn.Module) -> None:
        """Set model for checkpointing.

        Args:
            model: PyTorch model to save.
        """
        self._model = model

    def save_best_model(self, epoch: int, value: float) -> str:
        """Save the best model.

        Args:
            epoch: Epoch number.
            value: Metric value.

        Returns:
            Path to saved checkpoint.
        """
        return self._save_checkpoint(self._model, epoch, value)
