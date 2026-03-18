import os
import tempfile

import pandas as pd

from src.audio.data.split import split_by_session


class TestSplitBySession:
    """split_by_session Tests."""

    def test_split_with_session_column(self):
        """Session breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "filepath": [f"clip_{i}.wav" for i in range(20)],
                "label": ["dry_asphalt"] * 10 + ["wet_asphalt"] * 10,
                "session": [f"session_{i // 5}" for i in range(20)],
            })
            csv_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(csv_path, index=False)

            train_path, val_path, test_path = split_by_session(
                csv_path=csv_path,
                output_dir=tmpdir,
                val_size=0.2,
                test_size=0.2,
            )

            train = pd.read_csv(train_path)
            val = pd.read_csv(val_path)
            test = pd.read_csv(test_path)

            # All data covered
            assert len(train) + len(val) + len(test) == 20

            # Sessions don't overlap
            train_sessions = set(train["session"])
            val_sessions = set(val["session"])
            test_sessions = set(test["session"])
            assert train_sessions.isdisjoint(val_sessions)
            assert train_sessions.isdisjoint(test_sessions)
            assert val_sessions.isdisjoint(test_sessions)

    def test_split_without_session_column(self):
        """Fallback: Stratified split without sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "filepath": [f"clip_{i}.wav" for i in range(30)],
                "label": ["dry_asphalt"] * 15 + ["wet_asphalt"] * 15,
            })
            csv_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(csv_path, index=False)

            train_path, val_path, test_path = split_by_session(
                csv_path=csv_path,
                output_dir=tmpdir,
            )

            train = pd.read_csv(train_path)
            val = pd.read_csv(val_path)
            test = pd.read_csv(test_path)

            assert len(train) + len(val) + len(test) == 30

    def test_output_files_exist(self):
        """train.csv, val.csv, test.csv are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "filepath": [f"clip_{i}.wav" for i in range(20)],
                "label": ["dry_asphalt"] * 10 + ["wet_asphalt"] * 10,
            })
            csv_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(csv_path, index=False)

            split_by_session(csv_path=csv_path, output_dir=tmpdir)

            assert os.path.exists(os.path.join(tmpdir, "train.csv"))
            assert os.path.exists(os.path.join(tmpdir, "val.csv"))
            assert os.path.exists(os.path.join(tmpdir, "test.csv"))