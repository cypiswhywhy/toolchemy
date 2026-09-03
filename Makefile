SHELL := /bin/bash

.PHONY: help lint test test-cov test-all test-int test-perf docs-agents _publish publish

help:
	@echo "Available targets:"
	@grep -hF "##" $(MAKEFILE_LIST) | grep -Fv MAKEFILE_LIST | sed 's/^/  /'

lint: 		                     ## Lints source code
	poetry run ruff check toolchemy tests scripts

test:                            ## Run all unit tests
	poetry run pytest ./tests/unit

test-cov:                        ## Run unit tests with a branch-coverage report
	poetry run pytest ./tests/unit --cov=toolchemy --cov-branch --cov-report=term-missing

test-all:                        ## Run every test that needs no external service
	poetry run pytest ./tests -m "not integration"

test-int:                        ## Run integration tests (needs TOOLCHEMY_WHISPER_URL and a live server)
	poetry run pytest ./tests/int -m integration

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
