# Makefile for ome-zarr-io

.PHONY: help install install-dev test test-cov lint format clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=src/ome_zarr_io --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests excluding slow tests
	pytest tests/ -v -m "not slow"

lint:  ## Run linting
	flake8 src/ tests/
	mypy src/

format:  ## Format code
	black src/ tests/ examples/

format-check:  ## Check code formatting
	black --check src/ tests/ examples/

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

example:  ## Run schema example
	python examples/schema_example.py

example-integration:  ## Run integration example
	python examples/integration_example.py

all-examples:  ## Run all examples
	python examples/schema_example.py
	python examples/integration_example.py
