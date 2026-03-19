#!/usr/bin/env python3
"""Evaluate a trained model on test data.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/epoch_042_0.9100.pt \
        --config configs/audio/models/resnet18_mel.yaml \
        --test-csv data/processed/test.csv

    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --config configs/audio/models/simple_cnn.yaml \
        --save-predictions predictions.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.data.dataset import AudioMelDataset
from src.audio.models.factory import create_audio_model
from src.core.config import load_config
from src.core.device import get_device
from src.core.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    full_classification_report,
)
from src.core.seed import set_seed

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--test-csv", type=str, default=None, help="Test CSV (overrides config)"
    )
    parser.add_argument(
        "--audio-root", type=str, default=None, help="Audio root dir (overrides config)"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--save-predictions", type=str, default=None, help="Save predictions to CSV"
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    set_seed(config.get("project", config).get("seed", 42))
    device = get_device(args.device)

    # Load model
    console.print(f"\n[cyan]Loading checkpoint: {args.checkpoint}[/cyan]")
    model = create_audio_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test data
    data_cfg = config.get("data", {}).get("params", {})
    test_csv = args.test_csv or data_cfg.get("test_csv", "data/processed/test.csv")
    audio_root = args.audio_root or data_cfg.get("audio_root", "data/processed")

    console.print(f"[cyan]Test data: {test_csv}[/cyan]")

    test_dataset = AudioMelDataset(
        csv_path=test_csv,
        audio_root=audio_root,
        sample_rate=data_cfg.get("sample_rate", 16000),
        duration_sec=data_cfg.get("duration_sec", 5.0),
        n_mels=data_cfg.get("n_mels", 128),
        n_fft=data_cfg.get("n_fft", 2048),
        hop_length=data_cfg.get("hop_length", 512),
        augmentation_preset=None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    console.print(f"  Samples: {len(test_dataset)}")

    # Predict
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=-1)

            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.tolist())
            all_probs.extend(probs.cpu().tolist())

    # Metrics
    class_names = AudioMelDataset.CLASS_NAMES
    metrics = compute_metrics(all_targets, all_preds, class_names)
    report = full_classification_report(all_targets, all_preds, class_names)

    # Print results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")
    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("F1 Macro", f"{metrics['f1_macro']:.4f}")
    table.add_row("F1 Weighted", f"{metrics['f1_weighted']:.4f}")

    console.print(table)

    # Per-class metrics
    class_table = Table(title="Per-class F1")
    class_table.add_column("Class", style="cyan")
    class_table.add_column("F1", style="green")
    class_table.add_column("Precision", style="yellow")
    class_table.add_column("Recall", style="yellow")
    class_table.add_column("Support", style="dim")

    for cls_name in class_names:
        if cls_name in report:
            cls = report[cls_name]
            class_table.add_row(
                cls_name,
                f"{cls['f1-score']:.4f}",
                f"{cls['precision']:.4f}",
                f"{cls['recall']:.4f}",
                str(int(cls["support"])),
            )

    console.print(class_table)

    # Confusion matrix
    cm = compute_confusion_matrix(all_targets, all_preds, class_names)
    console.print("\n[cyan]Confusion Matrix:[/cyan]")
    console.print("  Rows: true labels, Columns: predicted")

    cm_table = Table()
    cm_table.add_column("", style="cyan")
    for name in class_names:
        cm_table.add_column(name[:8])

    for i, name in enumerate(class_names):
        row = [name[:8]] + [str(cm[i, j]) for j in range(len(class_names))]
        cm_table.add_row(*row)

    console.print(cm_table)

    # Save predictions
    if args.save_predictions:
        test_df = pd.read_csv(test_csv)
        test_df["predicted"] = [AudioMelDataset.IDX_TO_CLASS[p] for p in all_preds]
        test_df["correct"] = [t == p for t, p in zip(all_targets, all_preds)]
        for i, cls_name in enumerate(class_names):
            test_df[f"prob_{cls_name}"] = [p[i] for p in all_probs]
        test_df.to_csv(args.save_predictions, index=False)
        console.print(f"\n[green]Predictions saved to {args.save_predictions}[/green]")

    console.print("\n[bold green]Evaluation complete.[/bold green]")


if __name__ == "__main__":
    main()
