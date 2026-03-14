#!/usr/bin/env python3
"""Kaggle environment setup script.

This script sets up the Kaggle notebook environment with all required
dependencies, DVC configuration, and project installation.

Usage in Kaggle Notebook:
    !python kaggle/setup.py

Or run individual functions:
    from kaggle.setup import setup_environment, setup_dvc, install_project
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command.

    Args:
        cmd: Command and arguments as list.
        check: Raise exception on non-zero exit code.

    Returns:
        CompletedProcess instance.
    """
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=False)


def setup_environment() -> None:
    """Install core dependencies for Kaggle environment."""
    print("=" * 60)
    print("Setting up Kaggle environment...")
    print("=" * 60)

    # Core ML packages
    packages = [
        "torch",
        "torchaudio",
        "torchvision",
        "mlflow>=2.10",
        "wandb>=0.15",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "rich>=13.0",
        "tqdm>=4.65",
    ]

    print("\n[1/4] Installing core ML packages...")
    run_command([sys.executable, "-m", "pip", "install", "-q"] + packages)

    # Audio processing
    audio_packages = [
        "librosa>=0.10",
        "soundfile>=0.12",
        "audiomentations>=0.30",
    ]

    print("\n[2/4] Installing audio packages...")
    run_command([sys.executable, "-m", "pip", "install", "-q"] + audio_packages)

    # Video processing
    video_packages = [
        "timm>=0.9",
        "albumentations>=1.3",
        "opencv-python>=4.8",
        "pillow>=10.0",
    ]

    print("\n[3/4] Installing video packages...")
    run_command([sys.executable, "-m", "pip", "install", "-q"] + video_packages)

    # DVC with S3 support
    dvc_packages = [
        "dvc>=3.60",
        "dvc-s3>=3.0",
    ]

    print("\n[4/4] Installing DVC...")
    run_command([sys.executable, "-m", "pip", "install", "-q"] + dvc_packages)

    print("\n✓ Environment setup complete")


def setup_dvc(
    bucket_name: str,
    endpoint_url: str = "https://storage.yandexcloud.net",
) -> None:
    """Initialize and configure DVC for Yandex Cloud Storage.

    Args:
        bucket_name: Yandex Cloud bucket name.
        endpoint_url: S3-compatible endpoint URL.
    """
    print("\n" + "=" * 60)
    print("Configuring DVC...")
    print("=" * 60)

    # Initialize DVC (without SCM for Kaggle)
    print("\nInitializing DVC...")
    run_command(["dvc", "init", "--no-scm"])

    # Add remote storage
    print(f"Adding remote storage: s3://{bucket_name}/dvc-storage")
    run_command(
        ["dvc", "remote", "add", "-d", "storage", f"s3://{bucket_name}/dvc-storage"]
    )

    # Configure endpoint
    print(f"Setting endpoint URL: {endpoint_url}")
    run_command(["dvc", "remote", "modify", "storage", "endpointurl", endpoint_url])

    # Verify configuration
    print("\nDVC configuration:")
    run_command(["dvc", "remote", "list"])

    print("\n✓ DVC setup complete")


def setup_mlflow(
    tracking_uri: str,
    experiment_name: str = "road-surface-classification",
    artifact_location: str | None = None,
) -> None:
    """Configure MLflow tracking.

    Args:
        tracking_uri: MLflow server URL.
        experiment_name: Name of the experiment.
        artifact_location: Optional S3 location for artifacts.
    """
    print("\n" + "=" * 60)
    print("Configuring MLflow...")
    print("=" * 60)

    import mlflow

    # Set tracking URI
    print(f"Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    # Set or create experiment
    print(f"Experiment: {experiment_name}")
    if artifact_location:
        mlflow.set_experiment(experiment_name, artifact_location=artifact_location)
    else:
        mlflow.set_experiment(experiment_name)

    # Verify configuration
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"Experiment ID: {experiment.experiment_id}")

    print("\n✓ MLflow setup complete")


def install_project(
    repo_url: str,
    work_dir: Path | None = None,
    editable: bool = False,
) -> Path:
    """Clone and install the project repository.

    Args:
        repo_url: Git repository URL.
        work_dir: Working directory (default: temp directory).
        editable: Install in editable mode.

    Returns:
        Path to the cloned repository.
    """
    import tempfile

    print("\n" + "=" * 60)
    print("Installing project...")
    print("=" * 60)

    # Create working directory
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="kaggle_rsc_"))

    print(f"Working directory: {work_dir}")

    # Clone repository
    print(f"Cloning: {repo_url}")
    run_command(["git", "clone", repo_url, str(work_dir)])

    # Install project
    if editable:
        print("Installing in editable mode...")
        run_command([sys.executable, "-m", "pip", "install", "-e", str(work_dir)])
    else:
        print("Installing...")
        run_command([sys.executable, "-m", "pip", "install", str(work_dir)])

    print(f"\n✓ Project installed at {work_dir}")
    return work_dir


def pull_data() -> None:
    """Pull data from DVC remote storage."""
    print("\n" + "=" * 60)
    print("Pulling data from DVC remote...")
    print("=" * 60)

    run_command(["dvc", "pull"])

    print("\n✓ Data pulled successfully")


def main() -> None:
    """Run full Kaggle setup."""
    # Get credentials from environment (set via Kaggle Secrets)
    bucket_name = os.environ.get("BUCKET_NAME")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    repo_url = os.environ.get(
        "REPO_URL", "https://github.com/your-username/road-surface-classification.git"
    )

    if not bucket_name:
        raise ValueError("BUCKET_NAME environment variable not set")
    if not mlflow_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set")

    # Run setup
    setup_environment()
    setup_dvc(bucket_name)
    setup_mlflow(mlflow_uri)
    work_dir = install_project(repo_url)

    # Change to project directory
    os.chdir(work_dir)
    print(f"\n✓ Working directory changed to {work_dir}")

    # Pull data (optional, can be skipped for testing)
    pull_data()

    print("\n" + "=" * 60)
    print("KAGGLE SETUP COMPLETE")
    print("=" * 60)
    print(f"\nProject directory: {work_dir}")
    print(f"Data directory: {work_dir / 'data'}")
    print(f"MLflow tracking: {mlflow_uri}")
    print("\nYou can now start training:")
    print("  python scripts/train.py --config-name train_config")


if __name__ == "__main__":
    main()
