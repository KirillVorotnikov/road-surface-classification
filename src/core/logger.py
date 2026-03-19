"""Unified logging module.

Supports:
- MLflow (production tracking)
- File (local development / testing)
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import Tensor


class BaseLogger(ABC):
    """Abstract base class for experiment loggers."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float | Tensor], step: int | None = None
    ) -> None: ...

    @abstractmethod
    def log_artifact(
        self, file_path: str, artifact_path: str | None = None
    ) -> None: ...

    @abstractmethod
    def log_model(
        self,
        model: Any,
        artifact_name: str = "model",
        input_example: Any | None = None,
    ) -> None: ...

    @abstractmethod
    def log_image(
        self,
        image: Any,
        name: str,
        caption: str | None = None,
    ) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...

    # Common utilities

    def _flatten_params(self, params: dict, prefix: str = "") -> dict[str, str]:
        """Expanding nested dicts for MLflow/file."""
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_params(v, key))
            elif isinstance(v, (list, tuple)):
                flat[key] = str(v)
            else:
                flat[key] = str(v)
        return flat

    def _flatten_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Unfolding nested dicts and converting Tensors."""
        flat = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, Tensor):
                        flat[f"{k}/{sub_k}"] = sub_v.item()
                    elif isinstance(sub_v, (int, float)):
                        flat[f"{k}/{sub_k}"] = float(sub_v)
            elif isinstance(v, Tensor):
                flat[k] = v.item()
            elif isinstance(v, (int, float)):
                flat[k] = float(v)
        return flat


class MlflowLogger(BaseLogger):
    """MLflow logger. Uses MLflowConfig for connection."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "road-surface-classification",
        artifact_location: str | None = None,
        run_name: str | None = None,
        tags: dict | None = None,
    ):
        from .mlflow_config import MLflowConfig

        self._config = MLflowConfig(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            artifact_location=artifact_location,
            run_name=run_name,
            tags=tags or {},
        )
        self._config.setup()
        self._config.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow

        flat = self._flatten_params(params)
        truncated = {k: v[:250] if len(v) > 250 else v for k, v in flat.items()}
        mlflow.log_params(truncated)

    def log_metrics(
        self, metrics: dict[str, float | Tensor], step: int | None = None
    ) -> None:
        import mlflow

        flat = self._flatten_metrics(metrics)
        if flat:
            mlflow.log_metrics(flat, step=step)

    def log_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        import mlflow

        mlflow.log_artifact(file_path, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_name: str = "model",
        input_example: Any | None = None,
    ) -> None:
        import mlflow

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_name,
            input_example=input_example,
        )

    def log_image(
        self,
        image: Any,
        name: str,
        caption: str | None = None,
    ) -> None:
        import os
        import tempfile

        import mlflow
        import numpy as np

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name

            if hasattr(image, "numpy"):
                image = image.numpy()
            if isinstance(image, np.ndarray):
                import matplotlib.pyplot as plt

                plt.imsave(temp_path, image)
                mlflow.log_artifact(temp_path, "images")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def finish(self) -> None:
        import mlflow

        mlflow.end_run()


class FileLogger(BaseLogger):
    """Logger to file. For local development and testing."""

    def __init__(self, log_dir: str = "runs", run_name: str = "run"):
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"

    def log_params(self, params: dict[str, Any]) -> None:
        flat = self._flatten_params(params)
        with open(self.log_dir / "params.json", "w") as f:
            json.dump(flat, f, indent=2)

    def log_metrics(
        self, metrics: dict[str, float | Tensor], step: int | None = None
    ) -> None:
        flat = self._flatten_metrics(metrics)
        entry = {"step": step, **flat}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        pass

    def log_model(
        self,
        model: Any,
        artifact_name: str = "model",
        input_example: Any | None = None,
    ) -> None:
        pass

    def log_image(
        self,
        image: Any,
        name: str,
        caption: str | None = None,
    ) -> None:
        pass

    def finish(self) -> None:
        pass


def create_logger(config) -> BaseLogger:
    """Logger factory from config.

    config.logging.tool: "mlflow" | "file"
    """
    logging_cfg = config.get("logging", {})
    tool = logging_cfg.get("tool", "file")

    if tool == "mlflow":
        return MlflowLogger(
            tracking_uri=logging_cfg.get("tracking_uri"),
            experiment_name=logging_cfg.get(
                "experiment_name", "road-surface-classification"
            ),
            artifact_location=logging_cfg.get("artifact_location"),
            run_name=config.get("experiment_name"),
            tags=logging_cfg.get("tags"),
        )
    else:
        return FileLogger(
            run_name=config.get("experiment_name", "unnamed"),
        )
