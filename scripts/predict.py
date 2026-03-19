#!/usr/bin/env python3
"""Run inference on audio files.

Usage:
    # Single file
    python scripts/predict.py \
        --checkpoint checkpoints/best.pt \
        --config configs/audio/models/resnet18_mel.yaml \
        --input recording.wav

    # Directory of files
    python scripts/predict.py \
        --checkpoint checkpoints/best.pt \
        --config configs/audio/models/resnet18_mel.yaml \
        --input data/test_clips/
"""

import sys
import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.device import get_device
from src.audio.models.factory import create_audio_model
from src.audio.data.preprocessing import AudioPreprocessor
from src.audio.data.dataset import AudioMelDataset

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on audio files")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Audio file or directory")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    device = get_device(args.device)

    # Load model
    model = create_audio_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocessor
    data_cfg = config.get("data", {}).get("params", {})
    preprocessor = AudioPreprocessor(
        target_sample_rate=data_cfg.get("sample_rate", 16000),
        duration_sec=data_cfg.get("duration_sec", 5.0),
        n_mels=data_cfg.get("n_mels", 128),
        n_ffts=data_cfg.get("n_fft", 2048),
        hop_length=data_cfg.get("hop_length", 512),
    )

    # Find audio files
    input_path = Path(args.input)
    if input_path.is_dir():
        audio_files = sorted(
            list(input_path.glob("*.wav")) +
            list(input_path.glob("*.mp3")) +
            list(input_path.glob("*.flac"))
        )
    else:
        audio_files = [input_path]

    if not audio_files:
        console.print("[red]No audio files found[/red]")
        return

    # Predict
    class_names = AudioMelDataset.CLASS_NAMES

    table = Table(title="Predictions")
    table.add_column("File", style="cyan")
    table.add_column("Prediction", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("All probabilities", style="dim")

    with torch.no_grad():
        for audio_file in audio_files:
            try:
                features = preprocessor(audio_file)
                features = features.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, T)

                outputs = model(features)
                probs = torch.softmax(outputs, dim=-1).squeeze()
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()

                prob_str = " | ".join(
                    f"{name[:5]}:{p:.2f}"
                    for name, p in zip(class_names, probs.tolist())
                )

                table.add_row(
                    audio_file.name,
                    class_names[pred_idx],
                    f"{confidence:.3f}",
                    prob_str,
                )
            except Exception as e:
                table.add_row(audio_file.name, f"[red]ERROR: {e}[/red]", "", "")

    console.print(table)


if __name__ == "__main__":
    main()