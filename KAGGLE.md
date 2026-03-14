# Kaggle Integration Guide

Инструкция по интеграции проекта с Kaggle для обучения моделей на GPU/TPU.

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                     Yandex Cloud Storage                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ DVC Data    │  │ MLflow       │  │ Kaggle Dataset         │ │
│  │ (datasets)  │  │ (artifacts)  │  │ (snapshots, backups)   │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                    │
         │ dvc pull           │ mlflow tracking
         ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Kaggle Notebook                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Code (Git)  │  │ Training     │  │ WandB (visualization)  │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Быстрый старт

### Шаг 1: Настройка Kaggle Secrets

1. Откройте любой Kaggle Notebook
2. Нажмите **Settings** (справа) → **Secrets**
3. Добавьте следующие секреты:

| Название секрета | Описание | Пример значения |
|-----------------|----------|-----------------|
| `AWS_ACCESS_KEY_ID` | Yandex Cloud access key | `YCAXXXXXXXXXXXXX` |
| `AWS_SECRET_ACCESS_KEY` | Yandex Cloud secret key | `XXXXXXXXXXXXXXXXXXXXXXXX` |
| `BUCKET_NAME` | Имя бакета Yandex Cloud | `road-surface-bucket` |
| `MLFLOW_TRACKING_URI` | URL вашего MLflow сервера | `https://mlflow.example.com` |
| `WANDB_API_KEY` | Ключ WandB (опционально) | `XXXXXXXXXXXXXXXXXXXX` |
| `REPO_URL` | URL Git репозитория (опционально) | `https://github.com/user/repo.git` |

### Шаг 2: Создание Notebook

1. На Kaggle нажмите **New Notebook**
2. Включите **Internet** (Settings → Internet → On)
3. Включите **GPU** (Settings → Accelerator → GPU P100)
4. Скопируйте содержимое `kaggle/notebook_template.ipynb`

### Шаг 3: Запуск обучения

```python
# В начале notebook
!python kaggle/setup.py

# Или по шагам:
from kaggle.kaggle_secrets import setup_secrets
setup_secrets()

!dvc pull
!python scripts/train.py --config-name train_config
```

## Компоненты

### 1. DVC для данных

**Назначение:** Версионирование и загрузка данных из Yandex Cloud.

**Команды:**
```bash
# Инициализация
dvc init --no-scm
dvc remote add -d storage s3://$BUCKET_NAME/dvc-storage
dvc remote modify storage endpointurl https://storage.yandexcloud.net

# Загрузка данных
dvc pull

# Проверка статуса
dvc status
```

### 2. MLflow для экспериментов

**Назначение:** Трекинг метрик, параметров, моделей.

**Использование:**
```python
import mlflow
from src.core.mlflow_config import setup_mlflow

# Настройка
mlflow_config = setup_mlflow(
    tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
    experiment_name="road-surface-kaggle",
)

# Запуск эксперимента
with mlflow.start_run(run_name="experiment-1"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.pytorch.log_model(model, "model")
```

### 3. WandB для визуализации

**Назначение:** Real-time визуализация обучения.

**Использование:**
```python
from src.core.callbacks import DualLogger

with DualLogger(
    project="road-surface-classification",
    log_to_wandb=True,
    log_to_mlflow=True,
) as logger:
    logger.log_metrics({"loss": 0.5}, step=epoch)
```

## Структура файлов

```
road-surface-classification/
├── kaggle/
│   ├── notebook_template.ipynb  # Шаблон Notebook
│   ├── setup.py                 # Скрипт установки
│   └── kaggle_secrets.py        # Работа с Secrets
├── scripts/
│   ├── train.py                 # Обучение с MLflow
│   └── kaggle_setup.py          # Быстрая настройка
├── src/core/
│   ├── mlflow_config.py         # Конфигурация MLflow
│   └── callbacks.py             # Callbacks для логирования
├── configs/
│   └── train_config.yaml        # Hydra конфиг
└── KAGGLE.md                    # Этот файл
```

## Troubleshooting

### Ошибка: "Secret not found"

**Проблема:** Секрет не добавлен в Kaggle Secrets.

**Решение:**
1. Проверьте название секрета (case-sensitive)
2. Убедитесь, что секрет добавлен в том же аккаунте
3. Перезапустите notebook после добавления секрета

### Ошибка: "DVC pull failed"

**Проблема:** Неверные credentials или настройки DVC.

**Решение:**
```bash
# Проверьте переменные окружения
echo $AWS_ACCESS_KEY_ID
echo $BUCKET_NAME

# Проверьте подключение к bucket
aws s3 ls s3://$BUCKET_NAME --endpoint-url https://storage.yandexcloud.net

# Перенастройте DVC
dvc remote remove storage
dvc remote add -d storage s3://$BUCKET_NAME/dvc-storage
dvc remote modify storage endpointurl https://storage.yandexcloud.net
```

### Ошибка: "MLflow connection refused"

**Проблема:** MLflow сервер недоступен из Kaggle.

**Решение:**
1. Убедитесь, что MLflow сервер имеет публичный URL
2. Проверьте firewall правила
3. Используйте HTTPS для подключения

### Ошибка: "CUDA out of memory"

**Проблема:** Недостаточно GPU памяти.

**Решение:**
```yaml
# В train_config.yaml уменьшите batch_size
training:
  batch_size: 16  # было 32

# Или используйте gradient accumulation
training:
  batch_size: 16
  grad_accum_steps: 2
```

## Оптимизация для Kaggle

### 1. Кэширование данных

```python
# Сохраняйте данные в Kaggle Dataset для быстрого доступа
!zip -r data.zip data/
# Загрузите data.zip как Kaggle Dataset
# В следующем notebook: !unzip data.zip
```

### 2. Сохранение чекпоинтов

```python
# Сохраняйте модели в Kaggle Output
import torch
torch.save(model.state_dict(), "/kaggle/working/best_model.pth")

# Или в MLflow artifacts
mlflow.pytorch.log_model(model, "model")
```

### 3. Эффективное использование сессии

- Kaggle сессия: до 12 часов
- Сохраняйте чекпоинты каждые N эпох
- Используйте early stopping

## Примеры использования

### Обучение с базовой конфигурацией

```python
!python scripts/train.py --config-name train_config
```

### Обучение с переопределением параметров

```python
!python scripts/train.py --config-name train_config \
    model.name=resnet34 \
    training.epochs=100 \
    training.batch_size=16 \
    training.learning_rate=0.01
```

### Hyperparameter sweep

```python
!python scripts/train.py --multirun \
    model.name=resnet18,resnet34 \
    training.learning_rate=0.001,0.01,0.1 \
    training.batch_size=16,32
```

### Инференс на новых данных

```python
import torch
import mlflow

# Загрузка модели из MLflow
model_uri = "models:/road-surface-classification/latest"
model = mlflow.pytorch.load_model(model_uri)

# Инференс
model.eval()
with torch.no_grad():
    predictions = model(input_tensor)
```

## Ссылки

- [Kaggle Documentation](https://www.kaggle.com/docs)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs)
- [WandB Documentation](https://docs.wandb.ai)
- [Yandex Cloud Storage Docs](https://cloud.yandex.ru/docs/storage/)
