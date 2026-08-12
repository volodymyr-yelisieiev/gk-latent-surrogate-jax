PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif [ -x /opt/homebrew/bin/python3.12 ]; then echo /opt/homebrew/bin/python3.12; elif command -v python3.12 >/dev/null 2>&1; then command -v python3.12; elif command -v python3.11 >/dev/null 2>&1; then command -v python3.11; else command -v python3; fi)
UV ?= uv
RUN ?= $(UV) run --python $(PYTHON) --extra dev

.PHONY: install install-dev agent-check test test-fast lint format type-check check smoke-encoder smoke-direct-diagnostics smoke-simsiam smoke-sequence smoke-all build clean

install:
	$(UV) sync --python $(PYTHON)

install-dev:
	$(UV) sync --python $(PYTHON) --extra dev

agent-check:
	$(RUN) python scripts/verify_agent_setup.py

test:
	JAX_PLATFORM_NAME=cpu $(RUN) pytest --cov=gk_surrogate --cov-report=term-missing

test-fast:
	JAX_PLATFORM_NAME=cpu $(RUN) pytest -m "not slow" --cov=gk_surrogate --cov-report=term-missing --cov-fail-under=95

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff format .

type-check:
	$(RUN) mypy src/gk_surrogate

check: agent-check lint type-check test-fast

smoke-encoder:
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-encoder --config configs/experiment/smoke_encoder_supervised.yaml

smoke-direct-diagnostics:
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-direct-diagnostics --config configs/experiment/smoke_encoder_supervised.yaml --output-dir outputs/smoke_direct_diagnostics

smoke-simsiam:
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-encoder --config configs/experiment/smoke_encoder_simsiam.yaml

smoke-sequence:
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-sequence --config configs/experiment/smoke_sequence.yaml

smoke-all:
	JAX_PLATFORM_NAME=cpu $(RUN) gks inspect-data --config configs/data/tiny_dummy.yaml --dry-run
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-encoder --config configs/experiment/smoke_encoder_supervised.yaml
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-direct-diagnostics --config configs/experiment/smoke_encoder_supervised.yaml --output-dir outputs/smoke_direct_diagnostics
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-encoder --config configs/experiment/smoke_encoder_simsiam.yaml
	JAX_PLATFORM_NAME=cpu $(RUN) gks embed-dataset --config configs/experiment/smoke_embed_dataset.yaml
	JAX_PLATFORM_NAME=cpu $(RUN) gks evaluate-flux-head --config configs/experiment/smoke_evaluate_flux_head.yaml
	JAX_PLATFORM_NAME=cpu $(RUN) gks plot-representation --config configs/experiment/smoke_plot_representation.yaml
	JAX_PLATFORM_NAME=cpu $(RUN) gks train-sequence --config configs/experiment/smoke_sequence.yaml
	JAX_PLATFORM_NAME=cpu $(RUN) gks evaluate-rollout --config configs/experiment/smoke_evaluate_rollout.yaml

build:
	$(UV) build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov build dist *.egg-info outputs/smoke_*
