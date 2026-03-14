#!/usr/bin/env python3
"""Kaggle quick setup script.

This script provides a quick start for running training on Kaggle.
It sets up secrets, pulls data, and prepares the environment.

Usage:
    python scripts/kaggle_setup.py [--no-pull] [--no-verify]

Options:
    --no-pull      Skip pulling data from DVC remote
    --no-verify    Skip secrets verification
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quick setup for Kaggle training environment"
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip pulling data from DVC remote",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip secrets verification",
    )
    return parser.parse_args()


def main() -> None:
    """Run Kaggle setup."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("KAGGLE SETUP")
    print("=" * 60 + "\n")

    # Check if running on Kaggle
    is_kaggle = os.environ.get("KAGGLE_CONTAINER_NAME") is not None
    if is_kaggle:
        print("✓ Running on Kaggle")
    else:
        print("⚠ Not running on Kaggle (development mode)")

    # Setup secrets
    print("\n[1/4] Setting up secrets...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from kaggle.kaggle_secrets import print_secrets_status, setup_secrets

        if not args.no_verify:
            configured = setup_secrets()
            print(f"  AWS: {configured.get('aws', {}).get('bucket', 'N/A')}")
            print(
                f"  MLflow: {configured.get('mlflow', {}).get('tracking_uri', 'N/A')}"
            )
            wandb_status = (
                "configured"
                if configured.get("wandb", {}).get("api_key")
                else "not configured"
            )
            print(f"  WandB: {wandb_status}")
        else:
            setup_secrets()
            print("  Secrets configured (verification skipped)")

        print("  ✓ Secrets setup complete")
    except ImportError:
        print("  ⚠ kaggle_secrets not available (not on Kaggle?)")
        print("  Ensure environment variables are set manually")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        if not args.no_verify:
            print_secrets_status()
            sys.exit(1)

    # Verify DVC configuration
    print("\n[2/4] Checking DVC configuration...")
    try:
        import subprocess

        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            print(f"  Remotes: {result.stdout.strip()}")
        else:
            print("  No DVC remotes configured")
            bucket = os.environ.get("BUCKET_NAME")
            if bucket:
                print(f"  Configuring remote: s3://{bucket}/dvc-storage")
                subprocess.run(
                    [
                        "dvc",
                        "remote",
                        "add",
                        "-d",
                        "storage",
                        f"s3://{bucket}/dvc-storage",
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "dvc",
                        "remote",
                        "modify",
                        "storage",
                        "endpointurl",
                        "https://storage.yandexcloud.net",
                    ],
                    check=True,
                )
        print("  ✓ DVC configuration complete")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ DVC error: {e}")
    except FileNotFoundError:
        print("  ⚠ DVC not installed")

    # Pull data
    if not args.no_pull:
        print("\n[3/4] Pulling data from DVC remote...")
        try:
            import subprocess

            subprocess.run(["dvc", "pull"], check=True)
            print("  ✓ Data pulled successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error pulling data: {e}")
            print("  You can skip this step with --no-pull")
    else:
        print("\n[3/4] Skipping data pull (--no-pull)")

    # Setup MLflow
    print("\n[4/4] Setting up MLflow...")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("road-surface-kaggle")
            experiment = mlflow.get_experiment_by_name("road-surface-kaggle")
            print(f"  Tracking URI: {mlflow_uri}")
            print(f"  Experiment ID: {experiment.experiment_id}")
            print("  ✓ MLflow setup complete")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ MLFLOW_TRACKING_URI not set")

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    print(f"\nProject root: {project_root}")
    print(f"Data directory: {project_root / 'data'}")

    # Check data availability
    data_dir = project_root / "data"
    if data_dir.exists():
        files = list(data_dir.rglob("*"))
        print(f"Data files: {len([f for f in files if f.is_file()])} files")
    else:
        print("⚠ Data directory not found")

    print("\nNext steps:")
    print("  python scripts/train.py --config-name train_config")
    print("\nOr run the notebook: kaggle/notebook_template.ipynb")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
