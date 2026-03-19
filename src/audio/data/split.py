"""Split data into train/val/test by recording session.

Prevents data leakage: clips from the same recording session
never appear in both train and val/test sets.

Output CSVs are saved to data/processed/ (DVC-tracked).
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_by_session(
    csv_path: str | Path,
    output_dir: str | Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    session_column: str = "session",
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Split data by recording session to prevent leakage.

    If session_column is not present, falls back to
    stratified split by label.

    Args:
        csv_path: Path to master CSV with all samples.
        output_dir: Where to save train.csv, val.csv, test.csv.
        val_size: Validation fraction.
        test_size: Test fraction.
        session_column: Column name for session ID.
        seed: Random seed.

    Returns:
        Paths to (train.csv, val.csv, test.csv).
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if session_column in df.columns:
        train_df, val_df, test_df = _split_by_sessions(
            df, session_column, val_size, test_size, seed
        )
    else:
        train_df, val_df, test_df = _split_stratified(
            df, val_size, test_size, seed
        )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    return train_path, val_path, test_path


def _split_by_sessions(df, session_column, val_size, test_size, seed):
    """Split by unique session IDs."""
    sessions = df[session_column].unique()

    train_sessions, temp_sessions = train_test_split(
        sessions,
        test_size=val_size + test_size,
        random_state=seed,
    )

    if test_size > 0:
        relative_test = test_size / (val_size + test_size)
        val_sessions, test_sessions = train_test_split(
            temp_sessions,
            test_size=relative_test,
            random_state=seed,
        )
    else:
        val_sessions = temp_sessions
        test_sessions = []

    return (
        df[df[session_column].isin(train_sessions)],
        df[df[session_column].isin(val_sessions)],
        df[df[session_column].isin(test_sessions)],
    )


def _split_stratified(df, val_size, test_size, seed):
    """Fallback: stratified split by label."""
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],
        random_state=seed,
    )

    if test_size > 0:
        relative_test = test_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            stratify=temp_df["label"],
            random_state=seed,
        )
    else:
        val_df = temp_df
        test_df = pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df