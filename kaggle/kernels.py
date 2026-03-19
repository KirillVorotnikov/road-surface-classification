"""Kaggle kernel management.

Функционал:
- Отправка ноутбуков на Kaggle как kernel
- Мониторинг статуса выполнения
- Скачивание результатов

Важно: Dataset management не используется.
Данные загружаются напрямую из Yandex Cloud через DVC/boto3.

Usage:
    from kaggle.kernels import push_kernel, KernelStatus

    info = push_kernel("notebooks/train.ipynb", title="Train")
    print(f"Status: {info.status}")
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


class KernelStatus(Enum):
    """Статус выполнения kernel."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class KernelInfo:
    """Информация о выполнении kernel."""

    slug: str
    title: str
    status: KernelStatus
    url: str
    error_message: str | None = None
    execution_time: float | None = None

    @classmethod
    def from_api_response(cls, response: dict[str, Any]) -> "KernelInfo":
        """Создать KernelInfo из API response.

        Args:
            response: Raw API response dict.

        Returns:
            KernelInfo instance.
        """
        status_map = {
            "queued": KernelStatus.QUEUED,
            "running": KernelStatus.RUNNING,
            "complete": KernelStatus.COMPLETE,
            "error": KernelStatus.ERROR,
            "cancelled": KernelStatus.CANCELLED,
        }

        raw_status = response.get("status", "").lower()
        status = status_map.get(raw_status, KernelStatus.UNKNOWN)

        return cls(
            slug=response.get("ref", ""),
            title=response.get("title", ""),
            status=status,
            url=response.get("url", ""),
            error_message=response.get("error"),
            execution_time=response.get("elapsedSeconds"),
        )


def push_kernel(
    notebook_path: str | Path,
    *,
    title: str | None = None,
    is_public: bool = True,
    enable_gpu: bool = True,
    enable_internet: bool = True,  # Нужно для скачивания из S3
    wait: bool = False,
    timeout: int = 7200,
    poll_interval: int = 30,
) -> KernelInfo:
    """Отправить ноутбук на Kaggle как kernel.

    Создаёт kernel-metadata.json и отправляет ноутбук на выполнение
    на GPU инфраструктуре Kaggle.

    Args:
        notebook_path: Путь к .ipynb файлу.
        title: Заголовок kernel (по умолчанию имя файла).
        is_public: Публичный kernel (default True).
        enable_gpu: Включить GPU (default True).
        enable_internet: Включить интернет (default True, нужно для S3).
        wait: Ждать завершения (default False).
        timeout: Максимальное время ожидания в секундах (default 7200).
        poll_interval: Интервал опроса статуса в секундах (default 30).

    Returns:
        KernelInfo со статусом выполнения.

    Example:
        >>> info = push_kernel(
        ...     "notebooks/train_model.ipynb",
        ...     title="Road Surface Training",
        ...     wait=True
        ... )
        >>> print(f"Status: {info.status}")
    """
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    if nb_path.suffix != ".ipynb":
        raise ValueError("Must be a .ipynb notebook file")

    # Временная директория с kernel-metadata.json
    import shutil
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())

    # Копирование ноутбука
    shutil.copy(nb_path, temp_dir / nb_path.name)

    # Создание kernel-metadata.json
    kernel_title = title or nb_path.stem.replace("_", " ").title()
    metadata = {
        "id": _generate_kernel_id(),
        "title": kernel_title,
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_public": is_public,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,  # Важно для доступа к S3
        "accelerator": "gpu" if enable_gpu else "none",
    }

    with open(temp_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Отправка на Kaggle
    api = KaggleApi()
    api.authenticate()

    try:
        response = api.kernels_push(str(temp_dir))
        kernel_slug = response.url.split("/")[-1]

        if wait:
            return _wait_for_kernel(
                kernel_slug, timeout=timeout, poll_interval=poll_interval
            )

        return KernelInfo(
            slug=kernel_slug,
            title=kernel_title,
            status=KernelStatus.QUEUED,
            url=response.url,
        )

    finally:
        # Очистка
        shutil.rmtree(temp_dir, ignore_errors=True)


def _wait_for_kernel(
    kernel_slug: str,
    timeout: int = 7200,
    poll_interval: int = 30,
) -> KernelInfo:
    """Ждать завершения выполнения kernel.

    Args:
        kernel_slug: Идентификатор kernel.
        timeout: Максимальное время ожидания в секундах.
        poll_interval: Интервал опроса в секундах.

    Returns:
        Финальный KernelInfo с результатами.
    """
    api = KaggleApi()
    api.authenticate()

    start_time = time.time()
    print(f"Waiting for kernel {kernel_slug}...")

    while time.time() - start_time < timeout:
        status_response = api.kernels_status(kernel_slug)

        # Парсинг статуса
        if isinstance(status_response, dict):
            status_str = status_response.get("status", "").lower()
        else:
            status_str = str(status_response).lower()

        if status_str in ("complete", "error", "cancelled"):
            # Получение финальной информации
            info_response = api.kernels_list(
                search=kernel_slug.split("/")[-1],
                user=kernel_slug.split("/")[0] if "/" in kernel_slug else None,
            )

            for kernel in info_response:
                if kernel.ref == kernel_slug:
                    return KernelInfo.from_api_response(kernel.__dict__)

            # Fallback
            return KernelInfo(
                slug=kernel_slug,
                title=kernel_slug,
                status=KernelStatus(status_str),
                url=f"https://www.kaggle.com/{kernel_slug}",
            )

        print(f"  Status: {status_str}...")
        time.sleep(poll_interval)

    raise TimeoutError(f"Kernel did not complete within {timeout}s")


def get_kernel_output(
    kernel_slug: str,
    output_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Скачать результаты выполнения kernel.

    Args:
        kernel_slug: Идентификатор kernel (e.g., "username/kernel-name").
        output_path: Директория для сохранения.
        force: Перезаписать существующие файлы.

    Returns:
        Путь к директории с результатами.
    """
    from kaggle.api import KaggleClient

    client = KaggleClient()
    return client.download_kernel_output(
        kernel_slug,
        output_path,
        force=force,
    )


def _generate_kernel_id() -> str:
    """Генерация уникального ID для kernel."""
    import hashlib
    import time

    timestamp = str(time.time()).encode()
    return hashlib.md5(timestamp).hexdigest()[:22]


def create_kernel_metadata(
    title: str,
    notebook_name: str,
    *,
    is_public: bool = True,
    enable_gpu: bool = True,
    enable_internet: bool = True,
) -> dict[str, Any]:
    """Создать содержимое kernel-metadata.json.

    Args:
        title: Заголовок kernel.
        notebook_name: Имя файла ноутбука.
        is_public: Публичный kernel.
        enable_gpu: Включить GPU.
        enable_internet: Включить интернет (нужно для S3).

    Returns:
        Metadata dict для JSON сериализации.
    """
    return {
        "id": _generate_kernel_id(),
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_public": is_public,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "accelerator": "gpu" if enable_gpu else "none",
    }
