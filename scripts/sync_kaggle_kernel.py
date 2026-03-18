#!/usr/bin/env python3
"""Sync notebook to Kaggle kernel.

Pushes a local Jupyter notebook to Kaggle for GPU-accelerated training.

Usage:
    # Basic push
    python scripts/sync_kaggle_kernel.py notebooks/02_train_baseline.ipynb

    # With custom title and datasets
    python scripts/sync_kaggle_kernel.py notebooks/train.ipynb \\
        --title "Road Surface Training v2" \\
        --dataset road-surface/audio-data \\
        --wait

    # GPU disabled
    python scripts/sync_kaggle_kernel.py notebooks/eda.ipynb --no-gpu
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kaggle.kernels import KernelStatus, push_kernel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Push notebook to Kaggle kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s notebooks/train.ipynb
    %(prog)s notebooks/train.ipynb --title "My Training" --wait
    %(prog)s notebooks/eda.ipynb --no-gpu --dataset username/dataset-name
        """,
    )

    parser.add_argument(
        "notebook",
        type=Path,
        help="Path to .ipynb notebook file",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Kernel title (default: notebook filename)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=True,
        help="Make kernel public (default: True)",
    )
    parser.add_argument(
        "--private",
        action="store_false",
        dest="public",
        help="Make kernel private",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        dest="enable_gpu",
        help="Enable GPU accelerator (default: True)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="enable_gpu",
        help="Disable GPU accelerator",
    )
    parser.add_argument(
        "--internet",
        action="store_true",
        default=True,
        help="Enable internet access for S3 (default: True)",
    )
    parser.add_argument(
        "--no-internet",
        action="store_false",
        dest="enable_internet",
        help="Disable internet access",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for kernel to complete",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Max wait time in seconds (default: 7200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to download kernel output (when using --wait)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate notebook
    if not args.notebook.exists():
        print(f"Error: Notebook not found: {args.notebook}")
        sys.exit(1)

    if args.notebook.suffix != ".ipynb":
        print(f"Error: Must be a .ipynb file: {args.notebook}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("PUSHING TO KAGGLE")
    print(f"{'=' * 60}")
    print(f"\nNotebook: {args.notebook}")
    print(f"Title: {args.title or args.notebook.stem}")
    print(f"GPU: {'Enabled' if args.enable_gpu else 'Disabled'}")
    print(f"Internet: {'Enabled' if args.enable_internet else 'Disabled'}")
    print(f"Public: {'Yes' if args.public else 'No'}")
    print()

    # Push to Kaggle
    try:
        info = push_kernel(
            args.notebook,
            title=args.title,
            is_public=args.public,
            enable_gpu=args.enable_gpu,
            enable_internet=args.enable_internet,
            wait=args.wait,
            timeout=args.timeout,
        )

        print(f"\n{'=' * 60}")
        print("RESULT")
        print(f"{'=' * 60}")
        print(f"\nKernel URL: {info.url}")
        print(f"Status: {info.status.value}")

        if info.error_message:
            print(f"Error: {info.error_message}")

        if info.execution_time:
            print(f"Execution time: {info.execution_time:.1f}s")

        # Download output if requested
        if args.wait and args.output and info.status == KernelStatus.COMPLETE:
            print(f"\nDownloading output to {args.output}...")
            from kaggle.kernels import get_kernel_output

            output_path = get_kernel_output(
                info.slug,
                args.output,
                force=True,
            )
            print(f"Output saved to: {output_path}")

        print(f"\n{'=' * 60}\n")

        # Exit with error if kernel failed
        if info.status == KernelStatus.ERROR:
            sys.exit(1)
        elif info.status == KernelStatus.CANCELLED:
            sys.exit(2)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
