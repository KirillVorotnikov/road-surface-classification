"""MLflow logger module.

Provides centralized MLflow logging with support for:
- Metrics logging (batch and single)
- Parameters logging
- Model and artifact logging
- Image logging with proper cleanup
"""

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import mlflow
import numpy as np
from torch import Tensor


class BaseLogger(ABC):
    """Abstract base class for experiment loggers.

    Defines the interface for logging metrics, parameters, and artifacts.
    """

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float | Tensor],
        step: int | None = None,
    ) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step number (epoch/batch).
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameter name to value.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def log_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        """Log file artifact.

        Args:
            file_path: Path to the file.
            artifact_path: Optional path within artifact storage.
        """
        pass

    @abstractmethod
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
        pass


class MlflowLogger(BaseLogger):
    """MLflow implementation of BaseLogger.

    Provides logging to MLflow tracking server with support for:
    - Metrics (batch and single)
    - Parameters
    - Models
    - Artifacts
    - Images

    Example:
        ```python
        logger = MlflowLogger(
            tracking_uri="http://localhost:5000",
            experiment_name="road-surface-classification",
        )

        with logger:
            logger.log_metrics({"train/loss": 0.5}, step=epoch)
            logger.log_params({"lr": 0.001, "batch_size": 32})
            logger.log_model(model, artifact_name="best_model")
        ```
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "road-surface-classification",
        artifact_location: str | None = None,
        run_name: str | None = None,
        tags: dict | None = None,
    ):
        """Initialize MLflow logger.

        Args:
            tracking_uri: MLflow server URL.
            experiment_name: Name of the MLflow experiment.
            artifact_location: S3-compatible storage for artifacts.
            run_name: Optional name for the current run.
            tags: Additional tags for the run.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.run_name = run_name
        self.tags = tags or {}

        self._run_active = False

    def setup(self) -> None:
        """Configure and initialize MLflow tracking."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.artifact_location and self.artifact_location.startswith("s3://"):
            mlflow.set_experiment(
                self.experiment_name, artifact_location=self.artifact_location
            )
        else:
            mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str | None = None, tags: dict | None = None) -> None:
        """Start MLflow run.

        Args:
            run_name: Run name (overrides instance default).
            tags: Additional tags (merged with instance tags).
        """
        merged_tags = {**self.tags, **(tags or {})}
        run_name = run_name or self.run_name

        if run_name:
            merged_tags["mlflow.runName"] = run_name

        mlflow.start_run(tags=merged_tags)
        self._run_active = True

    def end_run(self) -> None:
        """End MLflow run."""
        if self._run_active:
            mlflow.end_run()
            self._run_active = False

    def __enter__(self):
        """Start logging session."""
        self.setup()
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup logging session."""
        self.end_run()

    def log_metrics(
        self,
        metrics: dict[str, float | Tensor],
        step: int | None = None,
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step number (epoch/batch).
        """
        # Convert tensors to floats
        processed = {
            k: v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()
        }

        if step is not None:
            mlflow.log_metrics(processed, step=step)
        else:
            mlflow.log_metrics(processed)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to MLflow.

        Args:
            params: Dictionary of parameter name to value.
        """
        mlflow.log_params(params)

    def log_model(
        self,
        model: Any,
        artifact_name: str = "model",
        input_example: Any | None = None,
    ) -> None:
        """Log model artifact to MLflow.

        Args:
            model: PyTorch model or model path.
            artifact_name: Name for the artifact.
            input_example: Optional input example for model signature.
        """
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_name,
            input_example=input_example,
        )

    def log_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        """Log file artifact to MLflow.

        Args:
            file_path: Path to the file.
            artifact_path: Optional path within artifact storage.
        """
        mlflow.log_artifact(file_path, artifact_path)

    def log_image(
        self,
        image: Any,
        name: str,
        caption: str | None = None,
    ) -> None:
        """Log image artifact to MLflow.

        Args:
            image: Image array (numpy) or PIL Image.
            name: Name for the image.
            caption: Optional caption (used for WandB compatibility).
        """
        import matplotlib.pyplot as plt

        # Create temporary file with proper cleanup
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name

                if hasattr(image, "numpy"):
                    image = image.numpy()
                if isinstance(image, np.ndarray):
                    plt.imsave(temp_path, image)
                    mlflow.log_artifact(temp_path, "images")
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
