"""Kaggle API integration module.

Provides local Jupyter notebook integration with Kaggle:
- Kernel submission and monitoring
- Output retrieval

Usage:
    from kaggle import push_kernel, KernelStatus

    info = push_kernel("notebooks/my_training.ipynb", title="Train")
    print(f"Kernel URL: {info.url}")
"""

from kaggle.kernels import KernelStatus, get_kernel_output, push_kernel

__all__ = [
    "KaggleClient",
    "KernelStatus",
    "push_kernel",
    "get_kernel_output",
]
