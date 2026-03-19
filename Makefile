.PHONY: setup setup-dvc setup-kaggle train-audio train-video test lint format clean dvc-init dvc-push dvc-pull dvc-status kaggle-push notebook notebook-lab train-cnn train-resnet train-effnet train-local predict check

# ── Установка ──
setup:
	pip install -e ".[dev]"
	pre-commit install

setup-dvc:
	pip install -e ".[dev,dvc]"

setup-kaggle:
	pip install -e ".[kaggle]"

# ── DVC ──
dvc-init:
	dvc init
	dvc remote add -d storage s3://$(BUCKET)/dvc-storage
	dvc remote modify storage endpointurl https://storage.yandexcloud.net

dvc-push:
	dvc push

dvc-pull:
	dvc pull

dvc-status:
	dvc status

# ── Обучение ──
train-audio:
	python scripts/train.py --config configs/audio/models/resnet18_mel.yaml

train-cnn:
	python scripts/train.py --config configs/audio/models/simple_cnn.yaml

train-resnet:
	python scripts/train.py --config configs/audio/models/resnet18_mel.yaml

train-effnet:
	python scripts/train.py --config configs/audio/models/efficientnet_b0.yaml

train-local:
	python scripts/train.py --config $(CONFIG) --logger file

train-hydra:
	python scripts/train_hydra.py --config-name train_config

train-video:
	python scripts/train.py --config configs/video/models/efficientnet_b2.yaml

# ── Оценка ──
evaluate:
	python scripts/evaluate.py --checkpoint $(CKPT) --config $(CONFIG)

predict:
	python scripts/predict.py --checkpoint $(CKPT) --config $(CONFIG) --input $(INPUT)

# ── Тесты ──
test:
	pytest tests/ -v --tb=short

test-core:
	pytest tests/core/ -v

test-audio:
	pytest tests/audio/ -v

test-video:
	pytest tests/video/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ── Код ──
lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

check: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .ruff_cache
	rm -rf outputs/

# ── Kaggle ──
kaggle-push:
	python scripts/sync_kaggle_kernel.py $(NOTEBOOK) \
		$(if $(TITLE),--title "$(TITLE)",) \
		$(if $(WAIT),--wait,) \
		$(if $(GPU),,$(if $(filter $(GPU),0 false),--no-gpu,)) \
		$(if $(INTERNET),,$(if $(filter $(INTERNET),0 false),--no-internet,))

# ── Notebooks ──
notebook:
	jupyter notebook notebooks/

notebook-lab:
	jupyter lab notebooks/