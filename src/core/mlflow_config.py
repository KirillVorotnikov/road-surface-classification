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

        # Pass artifact_location on first call to avoid double setup
        if self.artifact_location and self.artifact_location.startswith("s3://"):
            mlflow.set_experiment(
                self.experiment_name, artifact_location=self.artifact_location
            )
        else:
            mlflow.set_experiment(self.experiment_name)

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
