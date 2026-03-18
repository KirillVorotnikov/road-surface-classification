"""MLflow configuration module.

Provides centralized MLflow tracking setup with support for:
- External MLflow server
- Environment-based configuration
- Yandex Cloud artifact storage
"""

import os
from dataclasses import dataclass, field

import mlflow


@dataclass
class MLflowConfig:
    """MLflow configuration settings."""

    tracking_uri: str | None = None
    experiment_name: str = "road-surface-classification"
    artifact_location: str | None = None
    run_name: str | None = None
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.tracking_uri is None:
            self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

        if self.artifact_location is None:
            bucket = os.environ.get("BUCKET_NAME")
            if bucket:
                self.artifact_location = f"s3://{bucket}/mlflow-artifacts"

    def setup(self) -> None:
        """Configure and initialize MLflow tracking."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        kwargs = {}
        if self.artifact_location and self.artifact_location.startswith("s3://"):
            kwargs["artifact_location"] = self.artifact_location

        mlflow.set_experiment(self.experiment_name, **kwargs)

    def start_run(
        self, run_name: str | None = None, tags: dict | None = None, **kwargs
    ):
        """Start MLflow run with configuration."""
        merged_tags = {**self.tags, **(tags or {})}
        run_name = run_name or self.run_name

        return mlflow.start_run(
            run_name=run_name,
            tags=merged_tags,
            **kwargs,
        )


def get_mlflow_config() -> MLflowConfig:
    """Get MLflow configuration from environment."""
    return MLflowConfig(
        experiment_name=os.environ.get(
            "MLFLOW_EXPERIMENT", "road-surface-classification"
        ),
        run_name=os.environ.get("MLFLOW_RUN_NAME"),
    )