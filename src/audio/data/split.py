"""Splitting data into train/val/test by recording sessions."""

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
    """
    Splits data by sessions (not by individual frames/clips).

    This prevents data leakage: adjacent clips
    from the same record are not included in both train and val.

    Args:
    csv_path: Path to the CSV file containing the data.
    output_dir: Where to save train.csv, val.csv, and test.csv.
    val_size: Validation share.
    test_size: Test share.
    session_column: Name of the column containing the session ID.
    seed: Random seed.

    Returns:
    Paths to train.csv, val.csv, and test.csv.
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if session_column in df.columns:
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

        train_df = df[df[session_column].isin(train_sessions)]
        val_df = df[df[session_column].isin(val_sessions)]
        test_df = df[df[session_column].isin(test_sessions)]

    else:
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