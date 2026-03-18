# Road Surface Classification — Документация

Документация для команды разработки проекта.

---

## Назначение документов

| Документ | Для кого | Описание |
|----------|----------|----------|
| [SETUP.md](SETUP.md) | Все | Настройка окружения для работы |
| [DVC_GUIDE.md](DVC_GUIDE.md) | Команда датасета | Загрузка и управление данными в Yandex Cloud |
| [KAGGLE.md](KAGGLE.md) | Команда ML | Обучение на Kaggle GPU из локального репозитория |
| [PROJECT.md](PROJECT.md) | Все | Общая информация о проекте |

---

## Быстрый старт для новой команды

### Команда ML

1. Настроить окружение — [SETUP.md](SETUP.md)
2. Скачать данные — `make dvc-pull`
3. Запустить ноутбуки — `make notebook-lab`
4. Обучение на Kaggle GPU — [KAGGLE.md](KAGGLE.md)

### Команда датасета

1. Настроить DVC — [SETUP.md](SETUP.md), раздел "DVC + Yandex Cloud"
2. Загрузить данные — [DVC_GUIDE.md](DVC_GUIDE.md)
3. Проверить статус — `make dvc-status`

---

## Команды Makefile

### Установка
```bash
make setup           # Базовая установка (dev)
make setup-dvc       # + DVC зависимости
make setup-kaggle    # + Kaggle зависимости
```

### Данные
```bash
make dvc-pull        # Скачать данные из облака
make dvc-push        # Загрузить данные в облако
make dvc-status      # Проверить статус
```

### Обучение
```bash
make train-audio     # Запуск обучения аудио модели
```

### Тесты
```bash
make test            # Все тесты
make test-audio      # Тесты аудио модуля
make test-cov        # Тесты с покрытием (htmlcov/)
```

### Код
```bash
make lint            # Проверка ruff
make format          # Форматирование ruff
make clean           # Очистка кэшей
```

### Notebooks
```bash
make notebook        # Jupyter Notebook
make notebook-lab    # Jupyter Lab
```

### Kaggle
```bash
make kaggle-push NOTEBOOK=<path> [TITLE=<title>] [WAIT=1]
```

---

## Структура проекта

```
road-surface-classification/
├── src/
│   ├── audio/              # Аудио модуль (основной)
│   │   ├── data/           # Данные и препроцессинг
│   │   ├── models/         # Архитектуры моделей
│   │   └── evaluation/     # Метрики и инференс
│   ├── video/              # Видео модуль (опционально)
│   ├── fusion/             # Слияние модальностей
│   └── core/               # Базовые утилиты
├── notebooks/              # Jupyter ноутбуки
├── kaggle/                 # Kaggle API интеграция
├── scripts/                # Скрипты обучения
├── configs/                # Hydra конфигурации
├── tests/                  # Тесты
├── data/                   # Данные (DVC)
└── docs/d1zcipline/        # Документация
```

---

## Ссылки

- [Setup Guide](SETUP.md) — настройка окружения
- [DVC Guide](DVC_GUIDE.md) — работа с данными
- [Kaggle Guide](KAGGLE.md) — обучение на GPU
