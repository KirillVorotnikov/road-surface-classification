"""Training callbacks for WandB and MLflow synchronization.

Provides unified logging to both WandB (real-time visualization)
and MLflow (artifact tracking and model registry).
"""

from typing import Any

import mlflow
import wandb
from torch import Tensor


class DualLogger:
    """Logs metrics and artifacts to both WandB and MLflow.

    This class provides synchronized logging to:
    - WandB: Real-time visualization and dashboards
    - MLflow: Experiment tracking, model registry, artifacts

    Example:
        ```python
        logger = DualLogger(experiment_name="road-surface")

        with logger:
            for epoch in range(epochs):
                loss = train_step()
                logger.log_metrics({"train/loss": loss}, step=epoch)

                val_acc = validate()
                logger.log_metrics({"val/accuracy": val_acc}, step=epoch)

            logger.log_model(model, artifact_name="best_model")
        ```
    """

    def __init__(
        self,
        project: str = "road-surface-classification",
        experiment_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        log_to_wandb: bool = True,
        log_to_mlflow: bool = True,
    ):
        """Initialize dual logger.

        Args:
            project: Project name for WandB.
            experiment_name: MLflow experiment name.
            wandb_config: Optional config dict for WandB run.
            log_to_wandb: Enable WandB logging.
            log_to_mlflow: Enable MLflow logging.
        """
        self.project = project
        self.experiment_name = experiment_name
        self.wandb_config = wandb_config or {}
        self.log_to_wandb = log_to_wandb
        self.log_to_mlflow = log_to_mlflow

        self._wandb_run = None

    def __enter__(self):
        """Start logging sessions."""
        # Initialize WandB if enabled
        if self.log_to_wandb and not wandb.run:
            config = {
                "project": self.project,
                "name": self.experiment_name,
                **self.wandb_config,
            }
            self._wandb_run = wandb.init(**config)

        # MLflow run should be started externally via mlflow_config
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup logging sessions."""
        if self._wandb_run:
            wandb.finish()

    def log_metrics(
        self,
        metrics: dict[str, float | Tensor],
        step: int | None = None,
    ) -> None:
        """Log metrics to both WandB and MLflow.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step number (epoch/batch).
        """
        # Convert tensors to floats
        processed = {
            k: v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()
        }

        if self.log_to_wandb and self._wandb_run:
            wandb.log(processed, step=step)

        if self.log_to_mlflow:
            if step is not None:
                for name, value in processed.items():
                    mlflow.log_metric(name, value, step=step)
            else:
                for name, value in processed.items():
                    mlflow.log_metric(name, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameter name to value.
        """
        if self.log_to_wandb and self._wandb_run:
            wandb.config.update(params)

        if self.log_to_mlflow:
            mlflow.log_params(params)

    def log_model(
        self,
        model: Any,
        artifact_name: str = "model",
        input_example: Any | None = None,
    ) -> None:
        """Log model artifact.

        Args:
            model: PyTorch model or model path.
            artifact_name: Name for the artifact.
            input_example: Optional input example for model signature.
        """
        if self.log_to_mlflow:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_name,
                input_example=input_example,
            )

    def log_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        """Log file artifact.

        Args:
            file_path: Path to the file.
            artifact_path: Optional path within artifact storage.
        """
        if self.log_to_mlflow:
            mlflow.log_artifact(file_path, artifact_path)

        if self.log_to_wandb and self._wandb_run:
            wandb.save(file_path)

    def log_image(
        self,
        image: Any,
        name: str,
        caption: str | None = None,
    ) -> None:
        """Log image artifact.

        Args:
            image: Image array (numpy) or PIL Image.
            name: Name for the image.
            caption: Optional caption.
        """
        if self.log_to_wandb and self._wandb_run:
            wandb.log({name: wandb.Image(image, caption=caption)})

        if self.log_to_mlflow:
            import tempfile

            import matplotlib.pyplot as plt
            import numpy as np

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                if hasattr(image, "numpy"):
                    image = image.numpy()
                if isinstance(image, np.ndarray):
                    plt.imsave(f.name, image)
                    mlflow.log_artifact(f.name, "images")


class TrainingCallback:
    """Base callback for training loops.

    Provides hooks for epoch start/end, batch start/end, etc.
    """

    def on_train_start(self, logger: DualLogger, config: dict[str, Any]) -> None:
        """Called when training starts."""
        pass

    def on_train_end(self, logger: DualLogger) -> None:
        """Called when training ends."""
        pass

    def on_epoch_start(
        self,
        logger: DualLogger,
        epoch: int,
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self,
        logger: DualLogger,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_end(
        self,
        logger: DualLogger,
        epoch: int,
        batch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each batch."""
        pass
