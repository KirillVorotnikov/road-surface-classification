"""Kaggle Secrets utility module.

Provides convenient access to Kaggle Secrets API for configuring
environment variables required by DVC, MLflow, and WandB.

Usage:
    from kaggle.kaggle_secrets import setup_secrets

    # Setup all required secrets
    setup_secrets()

    # Or access individual secrets
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    aws_key = client.get_secret("AWS_ACCESS_KEY_ID")
"""

import os

from kaggle_secrets import UserSecretsClient

# Secret names (as configured in Kaggle UI)
SECRET_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
SECRET_AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
SECRET_BUCKET_NAME = "BUCKET_NAME"
SECRET_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
SECRET_WANDB_API_KEY = "WANDB_API_KEY"
SECRET_WANDB_ENTITY = "WANDB_ENTITY"
SECRET_REPO_URL = "REPO_URL"

# Environment variable names
ENV_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
ENV_AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
ENV_BUCKET_NAME = "BUCKET_NAME"
ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_WANDB_API_KEY = "WANDB_API_KEY"
ENV_WANDB_ENTITY = "WANDB_ENTITY"
ENV_REPO_URL = "REPO_URL"


class SecretsError(Exception):
    """Raised when a required secret is not found."""

    pass


def get_secret(
    secret_name: str,
    required: bool = True,
    default: str | None = None,
) -> str | None:
    """Get a secret from Kaggle Secrets.

    Args:
        secret_name: Name of the secret as configured in Kaggle UI.
        required: If True, raise exception when secret not found.
        default: Default value if secret not found and required=False.

    Returns:
        Secret value or None/default.

    Raises:
        SecretsError: If secret is required but not found.
    """
    client = UserSecretsClient()

    try:
        value = client.get_secret(secret_name)
        return value
    except Exception as e:
        if required:
            raise SecretsError(
                f"Required secret '{secret_name}' not found. "
                f"Please add it in Kaggle Settings → Secrets."
            ) from e
        return default


def setup_secrets(
    setup_aws: bool = True,
    setup_mlflow: bool = True,
    setup_wandb: bool = True,
    setup_repo: bool = True,
) -> dict:
    """Setup environment variables from Kaggle Secrets.

    Args:
        setup_aws: Configure AWS/Yandex Cloud credentials.
        setup_mlflow: Configure MLflow tracking URI.
        setup_wandb: Configure WandB credentials.
        setup_repo: Configure repository URL.

    Returns:
        Dictionary of configured environment variables.

    Raises:
        SecretsError: If any required secret is not found.
    """
    configured = {}

    # AWS credentials for DVC
    if setup_aws:
        aws_key = get_secret(SECRET_AWS_ACCESS_KEY_ID, required=True)
        aws_secret = get_secret(SECRET_AWS_SECRET_ACCESS_KEY, required=True)
        bucket = get_secret(SECRET_BUCKET_NAME, required=True)

        os.environ[ENV_AWS_ACCESS_KEY_ID] = aws_key
        os.environ[ENV_AWS_SECRET_ACCESS_KEY] = aws_secret
        os.environ[ENV_BUCKET_NAME] = bucket

        configured["aws"] = {
            "access_key_id": aws_key[:4] + "..." if aws_key else None,
            "bucket": bucket,
        }

    # MLflow tracking
    if setup_mlflow:
        mlflow_uri = get_secret(SECRET_MLFLOW_TRACKING_URI, required=True)

        os.environ[ENV_MLFLOW_TRACKING_URI] = mlflow_uri

        configured["mlflow"] = {
            "tracking_uri": mlflow_uri,
        }

    # WandB configuration
    if setup_wandb:
        wandb_key = get_secret(
            SECRET_WANDB_API_KEY,
            required=False,
            default=None,
        )
        wandb_entity = get_secret(
            SECRET_WANDB_ENTITY,
            required=False,
            default=None,
        )

        if wandb_key:
            os.environ[ENV_WANDB_API_KEY] = wandb_key
            configured["wandb"] = {
                "api_key": "configured" if wandb_key else None,
                "entity": wandb_entity,
            }
        else:
            configured["wandb"] = {"api_key": None, "entity": None}

    # Repository URL
    if setup_repo:
        repo_url = get_secret(
            SECRET_REPO_URL,
            required=False,
            default="https://github.com/your-username/road-surface-classification.git",
        )

        os.environ[ENV_REPO_URL] = repo_url

        configured["repo"] = {
            "url": repo_url,
        }

    return configured


def verify_secrets() -> dict:
    """Verify that all required secrets are configured.

    Returns:
        Dictionary with verification status for each service.
    """
    status = {}

    # Check AWS
    status["aws"] = {
        "access_key_id": ENV_AWS_ACCESS_KEY_ID in os.environ,
        "secret_access_key": ENV_AWS_SECRET_ACCESS_KEY in os.environ,
        "bucket_name": ENV_BUCKET_NAME in os.environ,
    }
    status["aws"]["configured"] = all(status["aws"].values())

    # Check MLflow
    status["mlflow"] = {
        "tracking_uri": ENV_MLFLOW_TRACKING_URI in os.environ,
    }
    status["mlflow"]["configured"] = all(status["mlflow"].values())

    # Check WandB
    status["wandb"] = {
        "api_key": ENV_WANDB_API_KEY in os.environ,
        "entity": ENV_WANDB_ENTITY in os.environ,
    }
    status["wandb"]["configured"] = status["wandb"]["api_key"]

    # Check repo
    status["repo"] = {
        "url": ENV_REPO_URL in os.environ,
    }
    status["repo"]["configured"] = all(status["repo"].values())

    return status


def print_secrets_status() -> None:
    """Print current secrets configuration status."""
    status = verify_secrets()

    print("\n" + "=" * 50)
    print("SECRETS CONFIGURATION STATUS")
    print("=" * 50)

    for service, checks in status.items():
        configured = checks.pop("configured", False)
        icon = "✓" if configured else "✗"
        print(f"\n{icon} {service.upper()}")

        for key, value in checks.items():
            icon = "✓" if value else "✗"
            print(f"  {icon} {key}: {value}")

    print("\n" + "=" * 50)

    all_configured = all(s.get("configured", False) for s in status.values())
    if all_configured:
        print("✓ All secrets configured successfully")
    else:
        print("✗ Some secrets are missing. Check Kaggle Settings → Secrets.")
    print("=" * 50 + "\n")
