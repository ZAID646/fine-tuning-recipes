.PHONY: install dev test train-via-lora train-qlora train-dpo push lint

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

train-lora:
	python -m src.cli train --config recipes/lora.yaml

train-qlora:
	python -m src.cli train --config recipes/qlora.yaml

train-dpo:
	python -m src.cli train --config recipes/dpo.yaml

push:
	python -m src.cli push --adapter-path ./outputs

lint:
	ruff check src/ tests/
