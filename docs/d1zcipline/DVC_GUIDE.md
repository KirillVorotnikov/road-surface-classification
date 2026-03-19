# DVC Guide — Работа с данными

Инструкция для команды датасета по загрузке и управлению данными в Yandex Cloud.

---

## Обзор

Данные проекта управляются через DVC (Data Version Control). Хранение — Yandex Cloud Object Storage (S3-совместимый).

**Что хранится в DVC:**
- Сырые аудиоданные (WAV файлы)
- Обработанные данные (спектрограммы, признаки)
- Разметка (CSV, JSON)

**Что НЕ хранится в DVC:**
- Модели (.pt, .pth) — логируются в MLflow
- Логи (wandb/, mlruns/)
- Временные файлы

---

## Настройка DVC

### 1. Установка

```bash
pip install -e ".[dev,dvc]"
```

Или:
```bash
make setup-dvc
```

### 2. Инициализация (если не инициализирован)

```bash
dvc init
```

### 3. Настройка remote

Remote уже настроен в `.dvc/config`:
```
[core]
    remote = storage
['remote "storage"']
    url = s3://road-surface-classification-storage-1/dvc-storage
    endpointurl = https://storage.yandexcloud.net
```

### 4. Credentials

**Вариант A: Переменные окружения**
```bash
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
```

**Вариант B: .env файл**
```bash
cp .env.example .env
# Отредактируйте .env
```

---

## Загрузка данных в облако

### Сценарий 1: Новые данные

**Шаг 1: Разместить файлы в `data/raw/`**
```bash
# Пример структуры
data/raw/
├── asphalt/
│   ├── sample_001.wav
│   └── sample_002.wav
├── concrete/
├── gravel/
├── dirt/
└── snow/
```

**Шаг 2: Добавить в DVC**
```bash
# Добавить всю папку
dvc add data/raw

# Или отдельные файлы
dvc add data/raw/asphalt/sample_001.wav
```

**Шаг 3: Проверить статус**
```bash
dvc status
```

**Шаг 4: Загрузить в облако**
```bash
dvc push
```

**Шаг 5: Закоммитить .dvc файл**
```bash
git add data/raw.dvc .gitignore
git commit -m "data: add new audio samples (asphalt, concrete)"
```

### Сценарий 2: Обновление существующих данных

**Шаг 1: Добавить новые файлы**
```bash
# Скопировать новые файлы в data/raw/
cp /path/to/new/*.wav data/raw/asphalt/
```

**Шаг 2: Обновить DVC**
```bash
# Пересоздать .dvc файл
dvc add data/raw
```

**Шаг 3: Загрузить изменения**
```bash
dvc push
git add data/raw.dvc
git commit -m "data: add 50 new asphalt samples"
```

### Сценарий 3: Исправление ошибок в данных

**Шаг 1: Удалить проблемные файлы**
```bash
rm data/raw/asphalt/corrupted_sample.wav
```

**Шаг 2: Обновить DVC**
```bash
dvc add data/raw
dvc push
git add data/raw.dvc
git commit -m "data: remove corrupted sample"
```

---

## Скачивание данных из облака

### Полная синхронизация
```bash
dvc pull
```

### Проверка статуса (что нужно скачать)
```bash
dvc status
```

### Скачать конкретную папку
```bash
dvc pull data/raw.dvc
```

---

## Команды DVC

| Команда | Описание |
|---------|----------|
| `dvc init` | Инициализация DVC в репозитории |
| `dvc add <path>` | Добавить файл/папку в DVC |
| `dvc status` | Проверить статус данных |
| `dvc push` | Загрузить данные в remote |
| `dvc pull` | Скачать данные из remote |
| `dvc remote list` | Показать настроенные remote |
| `dvc gc` | Очистить неиспользуемые данные из cache |

---

## Структура данных

### Raw данные
```
data/raw/
├── asphalt/          # Асфальт
├── concrete/         # Бетон
├── gravel/           # Гравий
├── dirt/             # Грунт
└── snow/             # Снег
```

### Processed данные
```
data/processed/
├── train/            # Обучающая выборка
│   ├── spectrograms/
│   └── labels.csv
├── val/              # Валидационная выборка
└── test/             # Тестовая выборка
```

---

## .dvcignore

Файлы, которые игнорируются DVC (указаны в `.dvcignore`):

```
# Сырые данные (хранятся в DVC, не в Git)
data/raw/*.wav
data/raw/**/*.wav

# Обработанные данные
data/processed/*.wav
data/processed/*.csv

# Кэш DVC
.dvc/cache

# Модели
*.pt
*.pth
*.onnx

# Логи
wandb/
mlruns/
```

---

## Best Practices

### Именование файлов
```
# Правильно
{class_name}/sample_{id}.wav
asphalt/sample_001.wav

# Неправильно
asphalt/new_file.wav
asphalt/recordings/final_v2_fixed.wav
```

### Коммиты
```bash
# Правильно — атомарный коммит
git add data/raw.dvc
git commit -m "data: add 100 gravel samples"

# Неправильно — несколько изменений в одном коммите
git commit -m "updated data and fixed stuff"
```

### Проверка перед push
```bash
# Всегда проверяйте статус
dvc status

# Проверьте размер данных
dvc status --json | jq '.[].size'
```

---

## Troubleshooting

### Ошибка: `Failed to connect to S3`
```bash
# Проверьте credentials
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Проверьте доступность remote
dvc remote list
```

### Ошибка: `File already exists in remote`
```bash
# Очистите локальный cache
dvc gc

# Попробуйте снова
dvc push --force
```

### Ошибка: `Checksum mismatch`
```bash
# Пересоздайте .dvc файл
dvc remove data/raw.dvc
dvc add data/raw
dvc push
```

### Данные не скачиваются
```bash
# Проверьте, что .dvc файл закоммичен
git status

# Принудительное скачивание
dvc pull --force
```

---

## Ссылки

- [DVC Documentation](https://dvc.org/doc)
- [Yandex Cloud Storage API](https://yandex.cloud/docs/storage/api-ref/)
- [DVC + S3](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
