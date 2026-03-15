"""Tests for MLflowConfig class.

Tests cover:
- Configuration initialization
- Environment variable handling
- MLflow setup with/without S3 artifact location
- Run creation with tags
"""

import os
from unittest.mock import patch

from src.core.mlflow_config import MLflowConfig, get_mlflow_config, setup_mlflow


class TestMLflowConfigInit:
    """Test MLflowConfig initialization."""

    def test_default_values(self):
        """Test default attribute values."""
        config = MLflowConfig()
        assert config.tracking_uri is None
        assert config.experiment_name == "road-surface-classification"
        assert config.artifact_location is None
        assert config.run_name is None
        assert config.tags == {}

    def test_custom_values(self):
        """Test custom attribute values."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test-experiment",
            artifact_location="s3://bucket/mlflow-artifacts",
            run_name="test-run",
            tags={"key": "value"},
        )
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "test-experiment"
        assert config.artifact_location == "s3://bucket/mlflow-artifacts"
        assert config.run_name == "test-run"
        assert config.tags == {"key": "value"}


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


class TestHelperFunctions:
    """Test helper functions."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("mlflow.set_experiment")
    def test_get_mlflow_config_defaults(self, mock_set_experiment):
        """Test get_mlflow_config with no environment."""
        config = get_mlflow_config()
        assert config.experiment_name == "road-surface-classification"
        assert config.tracking_uri is None

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://env:5000",
            "MLFLOW_EXPERIMENT": "env-experiment",
            "MLFLOW_RUN_NAME": "env-run",
        },
    )
    @patch("mlflow.set_experiment")
    def test_get_mlflow_config_from_env(self, mock_set_experiment):
        """Test get_mlflow_config reads environment variables."""
        config = get_mlflow_config()
        assert config.tracking_uri == "http://env:5000"
        assert config.experiment_name == "env-experiment"
        assert config.run_name == "env-run"

    @patch("mlflow.set_experiment")
    def test_setup_mlflow(self, mock_set_experiment):
        """Test setup_mlflow convenience function."""
        config = setup_mlflow(
            tracking_uri="http://test:5000",
            experiment_name="test-exp",
            artifact_location="s3://bucket/artifacts",
        )

        assert config.tracking_uri == "http://test:5000"
        assert config.experiment_name == "test-exp"
        assert config.artifact_location == "s3://bucket/artifacts"
