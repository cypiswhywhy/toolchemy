# Agent Guide for toolchemy

This document is for automated coding agents working in this repo.
Keep changes small, respect existing patterns, and follow the conventions below.

## Quick start

- Project type: Python library managed with Poetry.
- Python version: 3.12 (see `pyproject.toml`).
- Source root: `toolchemy/`
- Tests: `tests/unit`, `tests/perf`

## Build / lint / test commands

Use Poetry for all Python execution.

### Install deps

- `poetry install`

### Lint

- `make lint`
- Equivalent: `pylint --rcfile pyproject.toml toolchemy`

### Tests (all)

- Unit tests: `make test`
- All tests: `make test-all`
- Performance tests: `make test-perf`

### Single test

- Run one test file:
  - `poetry run pytest tests/unit/path/to/test_file.py`
- Run one test by node id:
  - `poetry run pytest tests/unit/path/to/test_file.py::TestClass::test_name`
- Run by keyword:
  - `poetry run pytest tests/unit -k "keyword"`

### Build / publish

- Build: `poetry build`
- Publish: `poetry publish --repository pypi`
- Makefile target: `make publish` (bumps patch, builds, publishes, commits, tags)

## Code style and conventions

### Imports

- Order: standard library, third-party, then local imports.
- Separate groups with a blank line.
- Prefer explicit imports over wildcard imports.

### Formatting

- Indentation: 4 spaces.
- Keep line length at or under 160 characters; wrap long calls with hanging indents.
- Strings: no enforced quote style; follow surrounding file.
- Use f-strings for interpolation.

### Types

- Use Python 3.10+ union syntax (`str | None`).
- Prefer built-in generics (`list[str]`, `dict[str, Any]`).
- Use `TypedDict` for structured dict payloads.
- Use `@dataclass` for simple data containers.
- Use `pydantic.BaseModel` for model config data.

### Naming

- Modules and functions: `snake_case`.
- Classes and exceptions: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: prefix with `_`.

### Error handling

- Define domain-specific exceptions for clear failure modes.
- Log context before re-raising when helpful.
- Prefer raising early with clear messages over silent failures.
- For optional behavior, return early instead of deeply nested conditionals.

### Logging

- Use `toolchemy.utils.logger.get_logger`.
- Keep log messages concise and include context.
- Avoid noisy logs in hot paths unless `DEBUG` level.

### Caching patterns

- Cache keys: use `ICacher.create_cache_key` with clear plain and hashed parts.
- When caching is disabled, return or raise explicitly.
- When caching fails, raise the dedicated cache errors.

### Tests

- Use `pytest` fixtures and `@pytest.mark.parametrize`.
- Test file names end with `_test.py`.
- Keep tests deterministic; use `freezegun` or seeded RNGs where needed.
- Always create or update unit tests to cover behavior changes and new features.

### CLI entry points

- Console script: `prompt-studio = toolchemy.ai.prompting.prompter_mlflow:run_studio`.

## Repository structure

- `toolchemy/ai`: model clients, prompting, trackers.
- `toolchemy/utils`: helpers, logging, caching, timers.
- `toolchemy/vision`: image utilities.
- `tests/unit`: unit tests.
- `tests/perf`: performance benchmarks.

## Existing rules

- Cursor rules: none found (`.cursor/rules/**` and `.cursorrules` absent).
- Copilot rules: none found (`.github/copilot-instructions.md` absent).

## Agent workflow expectations

- Read nearby code before editing to match style and patterns.
- Avoid reformatting unrelated code.
- Do not add new dependencies unless necessary.
- Prefer small, well-scoped changes.
- Update tests if behavior changes.
- Keep edits ASCII-only unless the file already uses Unicode.

## Common patterns in this codebase

- Retry logic uses `tenacity` in client code.
- JSON decoding errors are handled with recovery attempts and logging.
- Logging uses structured context strings instead of deep objects.
- Caching implementations include explicit error classes and `persist()`.

## Tips for single-file debugging

- Focus on the public APIs in `toolchemy/ai/clients/common.py` and related client modules.
- Cache behavior and errors live in `toolchemy/utils/cacher/`.
- Utility helpers (formatting, hashing, conversions) are in `toolchemy/utils/utils.py`.

## When adding new code

- Add type hints for all public functions and methods.
- Keep function signatures explicit rather than `*args/**kwargs` unless required.
- Add small helpers to `toolchemy/utils` instead of duplicating logic.
- If a new feature has configuration, model it as a dataclass or BaseModel.

## Linting considerations (pylint)

- Ensure new modules/classes follow the same naming scheme.
- Avoid unused imports or variables.
- Provide docstrings where pylint would require them (if configured later).

## Testing considerations

- Prefer unit tests; perf tests live in `tests/perf` with `pytest-benchmark`.
- For cache tests, assert error types (`CacheEntryDoesNotExistError`, etc.).
- For client tests, use dummy or fake clients to avoid network calls.
