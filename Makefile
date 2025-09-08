.PHONY: help install install-dev test format lint type-check clean run-dev train

# Default target
help:
	@echo "Italian Teacher Multi-Agent Framework"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install core dependencies"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run test suite"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run flake8 linter"
	@echo "  type-check   Run mypy type checker"
	@echo "  clean        Clean build artifacts"
	@echo "  run-dev      Start development server"
	@echo "  train        Run training pipeline"
	@echo "  setup        Run development environment setup"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev,training,audio]"

setup:
	python scripts/setup_dev.py

# Testing and Quality
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "unit or not integration"

test-integration:
	pytest tests/integration/ -v -m integration

test-fast:
	pytest tests/ -v -m "not slow and not ml"

test-ml:
	pytest tests/ -v --ml-tests -m ml

test-cov:
	pytest tests/ --cov=italian_teacher --cov-report=html --cov-report=term

test-cov-unit:
	pytest tests/unit/ --cov=italian_teacher --cov-report=html

test-watch:
	pytest tests/ -v --tb=short -x --lf

# Code formatting
format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Linting
lint:
	flake8 src/ tests/

# Type checking  
type-check:
	mypy src/

# All quality checks
check: format lint type-check test

# Clean up
clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Development server
run-dev:
	uvicorn italian_teacher.api.main:app --reload --host localhost --port 8000

# Training
train:
	python scripts/train_agents.py --config configs/development.yaml

# Data preparation
prepare-data:
	python scripts/prepare_training_data.py

# Database migration
migrate:
	alembic upgrade head

# Create new migration
migration:
	alembic revision --autogenerate -m "$(MSG)"