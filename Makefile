.PHONY: setup train-audio train-video test lint format clean

# ── Установка ──
setup:
	pip install -e ".[dev]"
	pre-commit install

# ── Обучение ──
train-audio:
	python scripts/train.py --config-path ../configs/audio/models \
	                        --config-name resnet18_mel

train-video:
	python scripts/train.py --config-path ../configs/video/models \
	                        --config-name efficientnet_b2

# ── Оценка ──
evaluate:
	python scripts/evaluate.py --checkpoint $(CKPT) --config $(CONFIG)

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
	pytest tests/ -v --cov=src --cov-report=html

# ── Код ──
lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .ruff_cache