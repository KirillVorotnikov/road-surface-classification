import os
import tempfile

import torch
import torch.nn as nn

from src.core.callbacks import EarlyStopping, ModelCheckpoint
from src.core.logger import FileLogger


class DummyModel(nn.Module):
    """A simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestEarlyStopping:
    """Early Stopping Tests."""

    def _make_logger(self):
        return FileLogger(log_dir=tempfile.mkdtemp())

    def test_triggers_after_patience(self):
        """Stops after patience epochs without improvement."""
        es = EarlyStopping(monitor="val/loss", mode="min", patience=3)
        logger = self._make_logger()

        es.on_epoch_end(logger, 0, {"val/loss": 1.0})
        assert not es.should_stop

        es.on_epoch_end(logger, 1, {"val/loss": 1.0})
        es.on_epoch_end(logger, 2, {"val/loss": 1.0})
        es.on_epoch_end(logger, 3, {"val/loss": 1.0})
        assert es.should_stop

    def test_resets_on_improvement(self):
        """Reset counter when upgrading."""
        es = EarlyStopping(monitor="val/acc", mode="max", patience=2)
        logger = self._make_logger()

        es.on_epoch_end(logger, 0, {"val/acc": 0.5})
        es.on_epoch_end(logger, 1, {"val/acc": 0.4})
        es.on_epoch_end(logger, 2, {"val/acc": 0.6})
        es.on_epoch_end(logger, 3, {"val/acc": 0.5})
        assert not es.should_stop

    def test_mode_min(self):
        """mode=min: improvement = value decreases."""
        es = EarlyStopping(monitor="val/loss", mode="min", patience=2)
        logger = self._make_logger()

        es.on_epoch_end(logger, 0, {"val/loss": 1.0})
        es.on_epoch_end(logger, 1, {"val/loss": 0.5})
        assert not es.should_stop

    def test_mode_max(self):
        """mode=max: enhancement = value increases."""
        es = EarlyStopping(monitor="val/acc", mode="max", patience=2)
        logger = self._make_logger()

        es.on_epoch_end(logger, 0, {"val/acc": 0.5})
        es.on_epoch_end(logger, 1, {"val/acc": 0.8})
        assert not es.should_stop

    def test_min_delta(self):
        """Improvement less than min_delta is not counted."""
        es = EarlyStopping(
            monitor="val/loss", mode="min",
            patience=2, min_delta=0.1,
        )
        logger = self._make_logger()

        es.on_epoch_end(logger, 0, {"val/loss": 1.0})
        es.on_epoch_end(logger, 1, {"val/loss": 0.95})
        es.on_epoch_end(logger, 2, {"val/loss": 0.92})
        assert es.should_stop

    def test_missing_metric_ignored(self):
        """If there is no metric -> do nothing."""
        es = EarlyStopping(monitor="val/loss", patience=1)
        logger = self._make_logger()

        es.on_epoch_end(logger, 0, {"train/loss": 1.0})
        es.on_epoch_end(logger, 1, {"train/loss": 0.5})
        assert not es.should_stop

    def test_first_epoch_is_always_best(self):
        """The first era is always considered the best."""
        es = EarlyStopping(monitor="val/loss", mode="min", patience=1)
        logger = self._make_logger()

        es.on_epoch_end(logger, 0, {"val/loss": 999.0})
        assert not es.should_stop


class TestModelCheckpoint:
    """Tests ModelCheckpoint."""

    def test_saves_on_improvement(self):
        """Keeps checkpoint when upgrading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            logger = FileLogger(log_dir=tmpdir)

            mc = ModelCheckpoint(
                monitor="val/acc", mode="max",
                save_dir=tmpdir, save_top_k=3,
            )
            mc.set_model(model)

            mc.on_epoch_end(logger, 0, {"val/acc": 0.7})

            pt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
            assert len(pt_files) == 1
            assert mc.best_value == 0.7

    def test_no_save_on_regression(self):
        """Does not save when deteriorating."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            logger = FileLogger(log_dir=tmpdir)

            mc = ModelCheckpoint(
                monitor="val/acc", mode="max",
                save_dir=tmpdir, save_top_k=3,
            )
            mc.set_model(model)

            mc.on_epoch_end(logger, 0, {"val/acc": 0.8})
            mc.on_epoch_end(logger, 1, {"val/acc": 0.7})

            pt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
            assert len(pt_files) == 1

    def test_save_top_k_cleanup(self):
        """Removes unnecessary checkpoints (save_top_k)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            logger = FileLogger(log_dir=tmpdir)

            mc = ModelCheckpoint(
                monitor="val/acc", mode="max",
                save_dir=tmpdir, save_top_k=2,
            )
            mc.set_model(model)

            mc.on_epoch_end(logger, 0, {"val/acc": 0.7})
            mc.on_epoch_end(logger, 1, {"val/acc": 0.8})
            mc.on_epoch_end(logger, 2, {"val/acc": 0.9})

            pt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
            assert len(pt_files) == 2
            assert mc.best_value == 0.9

    def test_best_checkpoint_path_exists(self):
        """best_checkpoint_path points to a real file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            logger = FileLogger(log_dir=tmpdir)

            mc = ModelCheckpoint(
                monitor="val/acc", mode="max",
                save_dir=tmpdir, save_top_k=3,
            )
            mc.set_model(model)

            mc.on_epoch_end(logger, 0, {"val/acc": 0.8})

            assert mc.best_checkpoint_path is not None
            assert os.path.exists(mc.best_checkpoint_path)

    def test_checkpoint_contains_state_dict(self):
        """The checkpoint contains a model_state_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            logger = FileLogger(log_dir=tmpdir)

            mc = ModelCheckpoint(
                monitor="val/acc", mode="max",
                save_dir=tmpdir, save_top_k=3,
            )
            mc.set_model(model)

            mc.on_epoch_end(logger, 0, {"val/acc": 0.8})

            ckpt = torch.load(mc.best_checkpoint_path, weights_only=True)
            assert "model_state_dict" in ckpt
            assert "epoch" in ckpt
            assert ckpt["epoch"] == 0

    def test_missing_metric_ignored(self):
        """If there is no metric, we do nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            logger = FileLogger(log_dir=tmpdir)

            mc = ModelCheckpoint(
                monitor="val/acc", mode="max",
                save_dir=tmpdir, save_top_k=3,
            )
            mc.set_model(model)

            mc.on_epoch_end(logger, 0, {"train/loss": 0.5})

            pt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
            assert len(pt_files) == 0

    def test_no_model_set(self):
        """If the model is not installed -> do nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(log_dir=tmpdir)

            mc = ModelCheckpoint(
                monitor="val/acc", mode="max",
                save_dir=tmpdir, save_top_k=3,
            )
            mc.on_epoch_end(logger, 0, {"val/acc": 0.8})

            pt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
            assert len(pt_files) == 0