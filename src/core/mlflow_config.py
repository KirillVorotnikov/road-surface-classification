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
    """MLflow configuration settings.

    Attributes:
        tracking_uri: MLflow server URL. Can be set via MLFLOW_TRACKING_URI env var.
        experiment_name: Name of the MLflow experiment.
        artifact_location: S3-compatible storage for artifacts (Yandex Cloud).
        run_name: Optional name for the current run.
        tags: Additional tags for the run.
    """

    tracking_uri: str | None = None
    experiment_name: str = "road-surface-classification"
    artifact_location: str | None = None
    run_name: str | None = None
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        """Apply environment variables if not explicitly set."""
        if self.tracking_uri is None:
            self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

        if self.artifact_location is None:
            bucket = os.environ.get("BUCKET_NAME")
            if bucket:
                self.artifact_location = f"s3://{bucket}/mlflow-artifacts"

    def setup(self) -> None:
        """Configure and initialize MLflow tracking.

        Sets up:
        - Tracking URI
        - Experiment
        - Default tags
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        if self.artifact_location and self.artifact_location.startswith("s3://"):
            mlflow.set_experiment(
                self.experiment_name, artifact_location=self.artifact_location
            )

    def start_run(
        self, run_name: str | None = None, tags: dict | None = None, **kwargs
    ):
        """Start MLflow run with configuration.

        Args:
            run_name: Run name (overrides instance default).
            tags: Additional tags (merged with instance tags).
            **kwargs: Additional arguments passed to mlflow.start_run.

        Returns:
            MLflow active run context.
        """
        merged_tags = {**self.tags, **(tags or {})}
        run_name = run_name or self.run_name

        if run_name:
            merged_tags["mlflow.runName"] = run_name

        return mlflow.start_run(tags=merged_tags, **kwargs)


def get_mlflow_config() -> MLflowConfig:
    """Get MLflow configuration from environment.

    Returns:
        MLflowConfig instance with environment-based settings.
    """
    return MLflowConfig(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        experiment_name=os.environ.get(
            "MLFLOW_EXPERIMENT", "road-surface-classification"
        ),
        run_name=os.environ.get("MLFLOW_RUN_NAME"),
    )


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "road-surface-classification",
    artifact_location: str | None = None,
) -> MLflowConfig:
    """Convenience function to setup MLflow tracking.

    Args:
        tracking_uri: MLflow server URL.
        experiment_name: Experiment name.
        artifact_location: S3 location for artifacts.

    Returns:
        Configured MLflowConfig instance.
    """
    config = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_location=artifact_location,
    )
    config.setup()
    return config
