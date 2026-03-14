# Kaggle Integration — Руководство по локальной работе

Интеграция с Kaggle для обучения моделей на GPU без перехода на сайт.

**Важно:** Данные хранятся в **Yandex Cloud** и управляются через **DVC**.
Kaggle Datasets **не используется**.

---

## 📦 Установка

```bash
# Установка Kaggle зависимостей
make setup-kaggle

# Или вручную
pip install -e ".[kaggle]"
```

---

## 🔑 Настройка аутентификации

### 1. Получение API токена

1. Зайдите на https://www.kaggle.com/account
2. Нажмите **"Create New API Token"**
3. Скачается файл `kaggle.json`

### 2. Размещение токена

**Вариант A: Глобально (рекомендуется)**
```bash
# Windows (PowerShell)
mkdir C:\Users\<user>\.kaggle
move C:\Users\<user>\Downloads\kaggle.json C:\Users\<user>\.kaggle\
```

**Вариант B: Переменные окружения**
```bash
cp .env.example .env

# Заполните:
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

---

## 📊 Работа с данными

### DVC (единственный способ)

Данные хранятся в Yandex Cloud Object Storage:

```bash
# Скачать данные
make dvc-pull

# Загрузить изменения
make dvc-push

# Проверить статус
make dvc-status
```

**На Kaggle:** данные скачиваются напрямую из S3 в начале ноутбука
(требуется `enable_internet=True` при отправке kernel).

---

## 💻 Локальные ноутбуки

### Запуск Jupyter

```bash
# Jupyter Notebook
make notebook

# Jupyter Lab (рекомендуется)
make notebook-lab
```

### Структура ноутбуков

```
notebooks/
├── 01_eda_audio.ipynb      # Разведочный анализ
└── kaggle/
    └── train_model.ipynb   # Шаблон для Kaggle GPU
```

---

## 🚀 Отправка на Kaggle GPU

### Быстрый запуск

```bash
# Отправить ноутбук на Kaggle
make kaggle-push NOTEBOOK=notebooks/kaggle/train_model.ipynb

# С кастомным заголовком
make kaggle-push NOTEBOOK=notebooks/kaggle/train_model.ipynb \
    TITLE="Road Surface Training v2"

# С ожиданием завершения
make kaggle-push NOTEBOOK=notebooks/kaggle/train_model.ipynb \
    WAIT=1
```

### Python API

```python
from kaggle.kernels import push_kernel, KernelStatus

# Отправка без ожидания
info = push_kernel(
    "notebooks/kaggle/train_model.ipynb",
    title="My Training",
    enable_gpu=True,
    enable_internet=True,  # Нужно для доступа к S3
)
print(f"Kernel URL: {info.url}")

# Отправка с ожиданием
info = push_kernel(
    "notebooks/kaggle/train_model.ipynb",
    wait=True,
    timeout=7200,  # 2 часа
)
print(f"Status: {info.status}")

# Скачивание результатов
from kaggle.kernels import get_kernel_output

output_path = get_kernel_output(
    info.slug,
    "outputs/kaggle"
)
```

### CLI скрипт

```bash
# Базовая отправка
python scripts/sync_kaggle_kernel.py notebooks/kaggle/train_model.ipynb

# С опциями
python scripts/sync_kaggle_kernel.py notebooks/train.ipynb \
    --title "Road Surface Training" \
    --wait \
    --timeout 7200 \
    --output outputs/kaggle
```

---

## 📈 Workflow

### Локальная разработка

```bash
# 1. Скачать данные (из Yandex Cloud)
make dvc-pull

# 2. Запустить Jupyter
make notebook-lab

# 3. Редактировать ноутбук, запускать ячейки локально
```

### Обучение на Kaggle GPU

```bash
# 1. Подготовить ноутбук для Kaggle
#    (ноутбук сам скачает данные из S3 при запуске)

# 2. Отправить на Kaggle
make kaggle-push NOTEBOOK=notebooks/kaggle/train_model.ipynb \
    TITLE="Training Run 1" \
    WAIT=1

# 3. Получить результаты
#    Файлы автоматически скачаются в outputs/kaggle/
```

### Мониторинг

- **MLflow**: Логируются метрики, модели, артефакты
- **WandB**: Реальная визуализация графиков
- **Kaggle UI**: https://www.kaggle.com/code/<username>/<kernel-name>

---

## 📁 Структура файлов

```
road-surface-classification/
├── kaggle/                      # Kaggle API модуль
│   ├── __init__.py
│   ├── api.py                   # KaggleClient (output download)
│   └── kernels.py               # Push/мониторинг
├── notebooks/
│   ├── 01_eda_audio.ipynb
│   └── kaggle/
│       └── train_model.ipynb    # Шаблон для Kaggle
├── scripts/
│   └── sync_kaggle_kernel.py    # CLI для отправки
├── .env.example                 # Шаблон переменных
└── Makefile
```

---

## 🔧 Конфигурация

### Переменные окружения (.env)

```bash
# Kaggle API
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key

# DVC (Yandex Cloud)
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
BUCKET_NAME=road-surface-classification-storage-1

# MLflow
MLFLOW_TRACKING_URI=https://your-mlflow-server.com

# WandB (опционально)
WANDB_API_KEY=xxx
WANDB_ENTITY=xxx
```

---

## ⚠️ Ограничения Kaggle

| Параметр | Значение |
|----------|----------|
| GPU сессия | До 12 часов |
| Недельный лимит | ~30 часов GPU |
| RAM | 16 GB |
| Disk | 77 GB |
| Internet | Отключен по умолчанию (нужно включить для S3) |

---

## 🔐 Kaggle Secrets (для доступа к S3)

Для скачивания данных из Yandex Cloud **на Kaggle** нужно добавить AWS credentials:

1. Откройте https://www.kaggle.com/settings
2. Раздел **"Secrets"** → **"Add a new Secret"**
3. Добавьте:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
4. В ноутбуке включите доступ к Secrets
5. В коде используйте: `os.environ['AWS_ACCESS_KEY_ID']`

**Альтернатива:** Использовать локальное выполнение без отправки на Kaggle.

---

## 🐛 Troubleshooting

### Ошибка аутентификации
```
Error: Could not find kaggle.json
```
**Решение:** Убедитесь, что `~/.kaggle/kaggle.json` существует и имеет права `600`.

### Ошибка отправки
```
Error: 400 Bad Request - code_file not found
```
**Решение:** Проверьте, что `kernel-metadata.json` создан корректно.

### Таймаут ожидания
```
TimeoutError: Kernel did not complete within 7200s
```
**Решение:** Увеличьте `--timeout` или проверьте ноутбук на ошибки.

### Нет доступа к S3 на Kaggle
```
Error: Access Denied / 403 Forbidden
```
**Решение:**
1. Проверьте, что `enable_internet=True` в `push_kernel()`
2. Добавьте AWS credentials в Kaggle Secrets
3. Проверьте права доступа к Yandex Cloud bucket

---

## 📚 Ссылки

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle Kernels](https://www.kaggle.com/code)
- [Kaggle Secrets](https://www.kaggle.com/settings)
- [DVC Documentation](https://dvc.org/doc)
