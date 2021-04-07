
.PHONY: deps

deps: requirements.txt dev-requirements.txt

requirements.txt: pyproject.toml
	poetry export -f requirements.txt -o requirements.txt

dev-requirements.txt: pyproject.toml
	poetry export --dev -f requirements.txt -o dev-requirements.txt

.PHONY: test

test:
	poetry run pytest

.PHONY: check

check: test
	poetry run mypy cognimaker --show-error-codes