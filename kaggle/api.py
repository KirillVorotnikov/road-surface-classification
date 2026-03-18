"""Kaggle API client для скачивания результатов kernel.

Используется только для получения output файлов после выполнения kernel.
Dataset management не используется (данные хранятся в DVC / Yandex Cloud).

Usage:
    from kaggle.api import KaggleClient

    client = KaggleClient()
    client.download_kernel_output("username/kernel-name", "outputs/")
"""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleClient:
    """Минималистичный Kaggle API client.

    Только для kernel operations. Dataset API не используется.

    Authentication:
        Требуется ~/.kaggle/kaggle.json или env переменные:
        - KAGGLE_USERNAME
        - KAGGLE_KEY

        Получить токен: https://www.kaggle.com/account
    """

    def __init__(self, config_dir: str | None = None):
        """Инициализация клиента.

        Args:
            config_dir: Путь к директории с kaggle.json (по умолчанию ~/.kaggle).
        """
        self.api = KaggleApi(config_dir=config_dir)
        self.api.authenticate()
        self._username: str | None = None

    @property
    def username(self) -> str:
        """Получить имя пользователя."""
        if self._username is None:
            self._username = self.api._config_values["username"]
        return self._username

    def download_kernel_output(
        self,
        kernel_slug: str,
        path: str | Path,
        *,
        force: bool = False,
    ) -> Path:
        """Скачать результаты выполнения kernel.

        Args:
            kernel_slug: Идентификатор kernel (e.g., "username/kernel-name").
            path: Локальная директория для сохранения.
            force: Перезаписать существующие файлы.

        Returns:
            Путь к директории с результатами.

        Example:
            >>> client = KaggleClient()
            >>> client.download_kernel_output(
            ...     "my-username/train-model",
            ...     "outputs/kaggle"
            ... )
        """
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)

        if not force and any(dest.iterdir()):
            raise FileExistsError(
                f"Directory {dest} is not empty. Use force=True to overwrite."
            )

        self.api.kernels_output(
            kernel=kernel_slug,
            path=str(dest),
            force=force,
        )

        return dest

    def __repr__(self) -> str:
        return f"KaggleClient(username={self.username!r})"
