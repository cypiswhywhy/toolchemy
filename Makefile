SHELL := /bin/bash

.PHONY: help lint test test-all test-perf docs-agents _publish publish

help:
	@echo "Available targets:"
	@grep -hF "##" $(MAKEFILE_LIST) | grep -Fv MAKEFILE_LIST | sed 's/^/  /'

lint: 		                     ## Lints source code
	@pylint --rcfile pyproject.toml toolchemy

test:                            ## Run all unit tests
	poetry run pytest ./tests/unit

test-all:                       ## Run all tests
	poetry run pytest ./tests

test-perf:                       ## Run perf tests
	poetry run pytest ./tests/perf

docs-agents:                     ## Regenerate AGENTS_MANIFEST.md from package introspection
	poetry run python scripts/generate_agents_manifest.py

_publish:
	poetry version patch
	poetry publish --build

publish: _publish
	git add .
	git commit
	git tag $(shell poetry version --short)
