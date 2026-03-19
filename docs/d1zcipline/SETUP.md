# Setup — Настройка окружения

Инструкция по установке и настройке проекта для работы.

---

## Требования

### Системные
- ОС: Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- Python: 3.10 или выше
- RAM: минимум 8 GB (рекомендуется 16 GB)
- Disk: 20 GB свободного места + место для данных (~10 GB)

### Опционально
- GPU: NVIDIA с поддержкой CUDA (для ускорения обучения)
- Git: для работы с репозиторием
- Conda/Mamba: для управления окружениями (рекомендуется)

---

## Установка

### Шаг 1: Клонирование репозитория

```bash
git clone <repository-url>
cd road-surface-classification
```

### Шаг 2: Создание окружения

**Вариант A: Conda (рекомендуется)**
```bash
conda create -n ml-base python=3.10
conda activate ml-base
```

**Вариант B: venv**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Шаг 3: Установка зависимостей

**Базовая установка (разработка)**
```bash
pip install -e ".[dev]"
```

**С DVC (для работы с данными)**
```bash
pip install -e ".[dev,dvc]"
```

**С Kaggle (для локальных ноутбуков)**
```bash
pip install -e ".[dev,dvc,kaggle]"
```

**Или через Makefile**
```bash
make setup           # dev зависимости
make setup-dvc       # dev + dvc
make setup-kaggle    # kaggle зависимости
```

### Шаг 4: Pre-commit hooks

```bash
pre-commit install
```

Hooks автоматически проверяют код перед коммитом:
- Ruff (lint + format)
- Trailing whitespace
- End of files
- YAML syntax
- Merge conflicts

---

## Настройка доступа

### DVC + Yandex Cloud

Данные хранятся в Yandex Cloud Object Storage.

**1. Получить credentials:**
- Зайдите в Yandex Cloud Console
- Создайте Service Account с правами `storage.editor`
- Создайте Authorized Key (JSON)
- Сохраните `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`

**2. Настроить переменные окружения:**

```bash
# Windows (PowerShell)
$env:AWS_ACCESS_KEY_ID="your-key"
$env:AWS_SECRET_ACCESS_KEY="your-secret"

# Linux/Mac
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
```

**Или через .env файл:**
```bash
cp .env.example .env
# Отредактируйте .env, вставив ваши ключи
```

**3. Проверить DVC:**
```bash
make dvc-status
```

**4. Скачать данные:**
```bash
make dvc-pull
```

### Kaggle API

Для отправки ноутбуков на Kaggle GPU.

**1. Получить API токен:**
- https://www.kaggle.com/account
- Нажмите "Create New API Token"
- Скачается файл `kaggle.json`

**2. Разместить файл:**

```bash
# Windows (PowerShell)
mkdir C:\Users\<user>\.kaggle
move C:\Users\<user>\Downloads\kaggle.json C:\Users\<user>\.kaggle\

# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**3. Проверить:**
```bash
kaggle whoami
```

### MLflow Tracking (опционально)

**Вариант A: Локальный сервер**
```bash
mlflow server --host 0.0.0.0 --port 5000
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**Вариант B: Удалённый сервер**
```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-server.com
```

### Weights & Biases (опционально)

```bash
wandb login
export WANDB_API_KEY=your-key
export WANDB_ENTITY=your-username
```

---

## Проверка установки

### Тесты
```bash
make test
```

### DVC
```bash
make dvc-status
make dvc-pull
```

### Kaggle
```bash
kaggle whoami
```

### Обучение (demo)
```bash
make train-audio
```

---

## Troubleshooting

### Ошибка: `No module named 'torch'`
```bash
pip install -e ".[dev]"
```

### Ошибка: `DVC failed to pull some files`
```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
make dvc-pull
```

### Ошибка: `Could not find kaggle.json`
```bash
# Проверить наличие файла
ls ~/.kaggle/kaggle.json  # Linux/Mac
dir C:\Users\<user>\.kaggle\kaggle.json  # Windows

# Проверить права (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json
```

### Ошибка: `pre-commit failed`
```bash
pre-commit run --all-files
ruff check src/ scripts/
ruff format src/ scripts/
```
