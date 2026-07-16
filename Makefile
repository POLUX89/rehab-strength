.PHONY: help setup install-dev ingest app test lint format clean check-secrets

help:
	@echo "setup         Crea el venv e instala todo para desarrollo"
	@echo "ingest        Corre el pipeline completo (Sheets + Strong -> data/processed)"
	@echo "app           Levanta el dashboard de Streamlit en local"
	@echo "test          Corre los tests"
	@echo "lint          Chequea estilo con ruff"
	@echo "format        Formatea el código"
	@echo "check-secrets Escanea el repo en busca de secretos"
	@echo "clean         Borra caches y artefactos de build"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pip install -e .
	.venv/bin/pre-commit install
	@echo "✅ Listo. Activá con: source .venv/bin/activate"

install-dev:
	pip install -r requirements-dev.txt && pip install -e .

ingest:
	python -m rehab_strength.ingest.run_all

app:
	streamlit run streamlit_app.py

test:
	pytest -v

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

check-secrets:
	gitleaks detect --source . --redact --no-banner

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
