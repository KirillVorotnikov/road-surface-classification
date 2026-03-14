# Kaggle Secrets Setup Guide

## Шаг 1: Откройте Kaggle Secrets

1. Зайдите на [kaggle.com](https://www.kaggle.com)
2. Создайте новый Notebook или откройте существующий
3. Нажмите **Settings** (правая панель) → **Secrets**

## Шаг 2: Добавьте секреты

### Обязательные секреты

| Название | Описание | Пример |
|----------|----------|--------|
| `AWS_ACCESS_KEY_ID` | Yandex Cloud access key | `YCAXXXXXXXXXXXXX` |
| `AWS_SECRET_ACCESS_KEY` | Yandex Cloud secret key | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` |
| `BUCKET_NAME` | Имя бакета Yandex Cloud | `road-surface-bucket` |
| `MLFLOW_TRACKING_URI` | URL вашего MLflow сервера | `https://mlflow.example.com` |

### Опциональные секреты

| Название | Описание | Пример |
|----------|----------|--------|
| `WANDB_API_KEY` | Ключ WandB для визуализации | `XXXXXXXXXXXXXXXXXXXX` |
| `WANDB_ENTITY` | Имя команды/пользователя WandB | `my-team` |
| `REPO_URL` | URL Git репозитория | `https://github.com/user/repo.git` |

## Шаг 3: Получение Yandex Cloud credentials

1. Зайдите в [Yandex Cloud Console](https://console.cloud.yandex.ru/)
2. Перейдите в **Storage** → **Бакеты**
3. Создайте новый бакет или выберите существующий
4. Нажмите **Ключи доступа** → **Создать новый ключ**
5. Сохраните:
   - **Access Key ID** → `AWS_ACCESS_KEY_ID`
   - **Secret Access Key** → `AWS_SECRET_ACCESS_KEY`
   - **Имя бакета** → `BUCKET_NAME`

## Шаг 4: Проверка

После добавления секретов запустите в Kaggle Notebook:

```python
from kaggle.kaggle_secrets import print_secrets_status
print_secrets_status()
```

Ожидаемый вывод:
```
==================================================
SECRETS CONFIGURATION STATUS
==================================================

✓ AWS
  ✓ access_key_id: True
  ✓ secret_access_key: True
  ✓ bucket_name: True

✓ MLFLOW
  ✓ tracking_uri: True

✓ WANDB
  ✓ api_key: True/False
  ✓ entity: True/False

==================================================
✓ All secrets configured successfully
==================================================
```

## Шаг 5: Использование в Notebook

```python
# В начале notebook
from kaggle.kaggle_secrets import setup_secrets
setup_secrets()

# Теперь переменные окружения доступны
import os
print(os.environ["AWS_ACCESS_KEY_ID"])  # YCAXXXXX...
print(os.environ["MLFLOW_TRACKING_URI"])  # https://...
```

## Примечания

- Секреты не отображаются в выводе ячеек
- Не печатайте секреты в логах
- Один набор секретов используется для всех Notebook аккаунта
- Для изменения секрета: Settings → Secrets → Edit → Save
