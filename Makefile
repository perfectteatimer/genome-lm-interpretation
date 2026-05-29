.PHONY: help install install-dev lint format test train-zdna train-g4 train-promoter clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with all extras
	pip install -e ".[interpret,data]"

install-dev:  ## Install with dev tooling (ruff, black, pytest)
	pip install -e ".[interpret,data,dev]"

lint:  ## Run ruff
	ruff check src scripts

format:  ## Auto-format with black + ruff
	black src scripts
	ruff check --fix src scripts

test:  ## Run the test suite
	pytest

train-zdna:  ## Fine-tune the Z-DNA model
	python scripts/train.py --config configs/zdna.yaml

train-g4:  ## Fine-tune the G-quadruplex model
	python scripts/train.py --config configs/g4.yaml

train-promoter:  ## Fine-tune the promoter model
	python scripts/train.py --config configs/promoter.yaml

clean:  ## Remove caches and build artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
