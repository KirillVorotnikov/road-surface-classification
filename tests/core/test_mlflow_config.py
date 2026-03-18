"""Tests for MLflowConfig class.

Tests cover:
- Configuration initialization
- Environment variable handling
- MLflow setup with/without S3 artifact location
- Run creation with tags
"""

import os
from unittest.mock import patch

from src.core.mlflow_config import MLflowConfig


class TestMLflowConfigStartRun:
    """Test start_run() method."""

    @patch("mlflow.start_run")
    def test_start_run_default(self, mock_start_run):
        """Test start_run with default values."""
        config = MLflowConfig()
        config.start_run()

        mock_start_run.assert_called_once_with(
            run_name=None,
            tags={},
        )

    @patch("mlflow.start_run")
    def test_start_run_with_instance_values(self, mock_start_run):
        """Test start_run with instance run_name and tags."""
        config = MLflowConfig(run_name="my-run", tags={"project": "rsc"})
        config.start_run()

        mock_start_run.assert_called_once_with(
            run_name="my-run",
            tags={"project": "rsc"},
        )

    @patch("mlflow.start_run")
    def test_start_run_with_override_values(self, mock_start_run):
        """Test start_run with override values."""
        config = MLflowConfig(run_name="my-run", tags={"project": "rsc"})
        config.start_run(run_name="override-run", tags={"env": "test"})

        mock_start_run.assert_called_once_with(
            run_name="override-run",
            tags={"project": "rsc", "env": "test"},
        )

    @patch("mlflow.start_run")
    def test_start_run_override_run_name_none_falls_back(self, mock_start_run):
        """Test that passing run_name=None falls back to instance run_name."""
        config = MLflowConfig(run_name="instance-run")
        config.start_run(run_name=None)

        mock_start_run.assert_called_once_with(
            run_name="instance-run",
            tags={},
        )

    @patch("mlflow.start_run")
    def test_start_run_no_run_name_anywhere(self, mock_start_run):
        """Test start_run when run_name is None everywhere."""
        config = MLflowConfig()  # run_name=None
        config.start_run()       # run_name=None

        mock_start_run.assert_called_once_with(
            run_name=None,
            tags={},
        )

class TestMLflowConfigPostInit:
    """Test __post_init__ behavior."""

    @patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://env-uri:5000"})
    def test_tracking_uri_from_env(self):
        """Test tracking_uri is set from environment."""
        config = MLflowConfig()
        assert config.tracking_uri == "http://env-uri:5000"

    @patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://env-uri:5000"})
    def test_explicit_tracking_uri_overrides_env(self):
        """Test explicit tracking_uri overrides environment."""
        config = MLflowConfig(tracking_uri="http://explicit:5000")
        assert config.tracking_uri == "http://explicit:5000"

    @patch.dict(os.environ, {"BUCKET_NAME": "my-bucket"})
    def test_artifact_location_from_env(self):
        """Test artifact_location is set from BUCKET_NAME env var."""
        config = MLflowConfig()
        assert config.artifact_location == "s3://my-bucket/mlflow-artifacts"

    @patch.dict(os.environ, {"BUCKET_NAME": "my-bucket"})
    def test_explicit_artifact_location_overrides_env(self):
        """Test explicit artifact_location overrides environment."""
        config = MLflowConfig(artifact_location="s3://other-bucket/artifacts")
        assert config.artifact_location == "s3://other-bucket/artifacts"


class TestMLflowConfigSetup:
    """Test setup() method."""

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_with_tracking_uri(self, mock_set_experiment, mock_set_tracking_uri):
        """Test setup sets tracking URI."""
        config = MLflowConfig(tracking_uri="http://test:5000")
        config.setup()
        mock_set_tracking_uri.assert_called_once_with("http://test:5000")
        mock_set_experiment.assert_called_once_with("road-surface-classification")

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_with_s3_artifact_location(
        self, mock_set_experiment, mock_set_tracking_uri
    ):
        """Test setup with S3 artifact location - single call with artifact_location."""
        config = MLflowConfig(
            tracking_uri="http://test:5000",
            artifact_location="s3://bucket/mlflow-artifacts",
        )
        config.setup()

        mock_set_tracking_uri.assert_called_once_with("http://test:5000")
        # Verify single call with artifact_location
        mock_set_experiment.assert_called_once_with(
            "road-surface-classification",
            artifact_location="s3://bucket/mlflow-artifacts",
        )

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_with_non_s3_artifact_location(
        self, mock_set_experiment, mock_set_tracking_uri
    ):
        """Test setup with non-S3 artifact location falls back to default."""
        config = MLflowConfig(
            tracking_uri="http://test:5000",
            artifact_location="file:///local/path",
        )
        config.setup()

        mock_set_tracking_uri.assert_called_once_with("http://test:5000")
        # Should call without artifact_location since it's not s3://
        mock_set_experiment.assert_called_once_with("road-surface-classification")

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_without_tracking_uri(
        self, mock_set_experiment, mock_set_tracking_uri
    ):
        """Test setup without tracking URI."""
        config = MLflowConfig()
        config.setup()

        mock_set_tracking_uri.assert_not_called()
        mock_set_experiment.assert_called_once_with("road-surface-classification")


class TestMLflowConfigStartRun:
    """Test start_run() method."""

    @patch("mlflow.start_run")
    def test_start_run_default(self, mock_start_run):
        """Test start_run with default values."""
        config = MLflowConfig()
        config.start_run()

        mock_start_run.assert_called_once_with(tags={}, **{})

    @patch("mlflow.start_run")
    def test_start_run_with_instance_values(self, mock_start_run):
        """Test start_run with instance run_name and tags."""
        config = MLflowConfig(run_name="my-run", tags={"project": "rsc"})
        config.start_run()

        mock_start_run.assert_called_once_with(
            tags={"project": "rsc", "mlflow.runName": "my-run"}, **{}
        )

    @patch("mlflow.start_run")
    def test_start_run_with_override_values(self, mock_start_run):
        """Test start_run with override values."""
        config = MLflowConfig(run_name="my-run", tags={"project": "rsc"})
        config.start_run(run_name="override-run", tags={"env": "test"})

        mock_start_run.assert_called_once_with(
            tags={"project": "rsc", "env": "test", "mlflow.runName": "override-run"},
            **{},
        )
