import json
import tempfile
from pathlib import Path

from src.core.logger import BaseLogger, FileLogger, create_logger


class TestFileLogger:
    """Tests FileLogger."""

    def test_log_metrics(self):
        """Metrics are written to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(log_dir=tmpdir, run_name="test")

            logger.log_metrics({"loss": 0.5, "acc": 0.8}, step=0)
            logger.log_metrics({"loss": 0.3, "acc": 0.9}, step=1)

            metrics_file = Path(tmpdir) / "test" / "metrics.jsonl"
            assert metrics_file.exists()

            lines = metrics_file.read_text().strip().split("\n")
            assert len(lines) == 2

            first = json.loads(lines[0])
            assert first["step"] == 0
            assert first["loss"] == 0.5
            assert first["acc"] == 0.8

    def test_log_nested_metrics(self):
        """Nested dict (f1_per_class) are expanded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(log_dir=tmpdir, run_name="test")

            logger.log_metrics(
                {
                    "loss": 0.5,
                    "f1_per_class": {"dry": 0.9, "wet": 0.7},
                },
                step=0,
            )

            metrics_file = Path(tmpdir) / "test" / "metrics.jsonl"
            entry = json.loads(metrics_file.read_text().strip())

            assert entry["f1_per_class/dry"] == 0.9
            assert entry["f1_per_class/wet"] == 0.7
            assert entry["loss"] == 0.5

    def test_log_params(self):
        """Parameters are written, nested ones are expanded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(log_dir=tmpdir, run_name="test")

            logger.log_params(
                {
                    "lr": 0.001,
                    "model": {"name": "cnn", "layers": 5},
                }
            )

            params_file = Path(tmpdir) / "test" / "params.json"
            params = json.loads(params_file.read_text())

            assert params["lr"] == "0.001"
            assert params["model.name"] == "cnn"
            assert params["model.layers"] == "5"

    def test_finish_safe(self):
        """finish() does not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(log_dir=tmpdir, run_name="test")
            logger.finish()

    def test_creates_run_directory(self):
        """Creates a folder for run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _ = FileLogger(log_dir=tmpdir, run_name="my_run")

            assert (Path(tmpdir) / "my_run").is_dir()


class TestCreateLogger:
    """create_logger factory tests."""

    def test_file_logger_by_default(self):
        """Creates a FileLogger by default."""
        config = {"experiment_name": "test"}
        logger = create_logger(config)
        assert isinstance(logger, FileLogger)

    def test_file_logger_explicit(self):
        """tool=file -> FileLogger."""
        config = {
            "logging": {"tool": "file"},
            "experiment_name": "test",
        }
        logger = create_logger(config)
        assert isinstance(logger, FileLogger)

    def test_all_loggers_are_base_logger(self):
        """Any logger is a successor to BaseLogger."""
        config = {"experiment_name": "test"}
        logger = create_logger(config)
        assert isinstance(logger, BaseLogger)
