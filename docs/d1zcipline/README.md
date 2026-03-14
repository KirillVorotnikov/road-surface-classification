# Road Surface Classification

Классификация типа дорожного покрытия по аудиоданным.

## Быстрый старт

```bash
# Установка
pip install -e ".[dev]"

# Скачать данные (DVC)
make dvc-pull

# Запуск обучения
make train-audio

# Тесты
make test
```

## Документация

- **[KAGGLE.md](KAGGLE.md)** — Интеграция с Kaggle (локальные ноутбуки, отправка на GPU)
- **[DVC_SETUP.md](DVC_SETUP.md)** — Настройка DVC и удалённого хранилища

## Kaggle Integration

Для работы с Kaggle локально (без перехода на сайт):

```bash
# 1. Установка
make setup-kaggle

# 2. Настройка (получить токен на https://www.kaggle.com/account)
# Скопируйте ~/.kaggle/kaggle.json или заполните .env

# 3. Запуск ноутбуков
make notebook-lab

# 4. Отправка на Kaggle GPU
make kaggle-push NOTEBOOK=notebooks/kaggle/train_model.ipynb WAIT=1
```

Подробности в [KAGGLE.md](KAGGLE.md).

## Структура

```
├── src/                    # Исходный код
│   ├── audio/              # Аудио модуль (основной)
│   ├── video/              # Видео модуль
│   └── core/               # Базовые утилиты
├── notebooks/              # Jupyter ноутбуки
│   ├── 01_eda_audio.ipynb
│   └── kaggle/
├── kaggle/                 # Kaggle API интеграция
├── scripts/                # Скрипты обучения
├── configs/                # Hydra конфигурации
├── tests/                  # Тесты
└── data/                   # Данные (DVC)
```

## Technology Stack

- **Core:** PyTorch, scikit-learn, Hydra, MLflow, WandB
- **Audio:** torchaudio, librosa, audiomentations
- **Video:** torchvision, timm, albumentations
- **Data:** DVC (Yandex Cloud S3)
- **Kaggle:** kaggle API

## License

MIT
