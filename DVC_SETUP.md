# DVC Setup Guide

## Quick Setup

### 1. Initialize DVC

```bash
conda activate ml-base
dvc init
```

### 2. Configure Yandex Cloud Storage

```bash
# Set your bucket name
BUCKET_NAME="your-bucket-name"

# Add remote storage
dvc remote add -d storage s3://$BUCKET_NAME/dvc-storage

# Configure Yandex Cloud endpoint
dvc remote modify storage endpointurl https://storage.yandexcloud.net

# Configure credentials (choose one option):

# Option A: Store in DVC config (less secure)
dvc remote modify storage access_key_id YOUR_ACCESS_KEY
dvc remote modify storage secret_access_key YOUR_SECRET_KEY

# Option B: Use environment variables (recommended)
# Add to .env or set in your shell
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
```

### 3. Add Data

```bash
# Create data directories
mkdir -p data/raw/audio
mkdir -p data/raw/video
mkdir -p data/processed

# Add data to DVC tracking
dvc add data/raw
dvc add data/processed

# Commit
git add data/.gitignore data/raw.dvc data/processed.dvc
git commit -m "feat: add data tracking with DVC"
```

### 4. Sync Data

```bash
# Upload to Yandex Cloud
dvc push

# Download from Yandex Cloud
dvc pull

# Check status
dvc status
```

## Directory Structure

```
data/
├── raw/                    # Raw data (tracked by DVC)
│   ├── audio/              # Audio files (.wav, .mp3)
│   └── video/              # Video files (.mp4, .avi)
├── processed/              # Processed data (tracked by DVC)
└── .gitignore              # Ignores data files, keeps .dvc files
```

## Common Commands

| Command | Description |
|---------|-------------|
| `dvc add <path>` | Track file/directory with DVC |
| `dvc push` | Upload data to Yandex Cloud |
| `dvc pull` | Download data from Yandex Cloud |
| `dvc status` | Check if data is up to date |
| `dvc gc` | Garbage collect unused data |
| `dvc ls <remote> <path>` | List files in remote storage |
| `dvc get <remote> <path>` | Download file without pulling |

## Environment Variables

```bash
# Yandex Cloud credentials
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# Optional: specify remote
export DVC_REMOTE=storage
```

## Security Notes

- Never commit `.dvc/config` with credentials
- Use environment variables or AWS credentials file
- Keep `.dvc` files (metadata) in Git
- Keep actual data files out of Git (handled by DVC)
