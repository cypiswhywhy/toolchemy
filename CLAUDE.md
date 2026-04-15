# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `AGENTS.md` for the full contributor guide (style, conventions, per-module pointers). This file captures the highest-signal items.

## Commands

All Python execution goes through Poetry. Python 3.12 only.

- Install: `poetry install`
- Lint: `make lint` (runs `pylint --rcfile pyproject.toml toolchemy`)
- Unit tests: `make test` (`poetry run pytest ./tests/unit`)
- All tests incl. perf: `make test-all`
- Perf benchmarks only: `make test-perf` (uses `pytest-benchmark`)
- Single test: `poetry run pytest tests/unit/path/to/test_file.py::TestClass::test_name`
- By keyword: `poetry run pytest tests/unit -k "keyword"`
- Build: `poetry build`
- Release: `make publish` bumps the patch version, builds, publishes to PyPI, commits, and tags — do not run casually.

Test file naming: files end with `_test.py` (not `test_*.py`).

## Architecture

Toolchemy is a Python library of reusable tooling organized around three concerns: talking to model providers, tracking experiments, and caching/utility plumbing.

### `toolchemy/ai/clients/`
Unified client layer over multiple LLM/ASR providers (OpenAI, Gemini, Ollama, Whisper, plus a dummy client for tests). `common.py` defines the shared client interface/base behavior; `factory.py` constructs the right client by name. `pricing.py` holds per-model token pricing used for cost accounting. Client code uses `tenacity` for retries and has explicit JSON-decode recovery paths — match these patterns when adding providers.

### `toolchemy/ai/prompting/`
Prompt iteration and optimization. `prompter_mlflow.py` exposes the `prompt-studio` console script (entry point `toolchemy.ai.prompting.prompter_mlflow:run_studio`). `simple_llm_prompt_optimizer.py` is the optimizer loop.

### `toolchemy/ai/trackers/`
Pluggable experiment trackers with a common interface in `common.py` and implementations for MLflow, Neptune, and an in-memory tracker (useful as a test double).

### `toolchemy/utils/cacher/`
Cache backends (`diskcache`, `pickle`, `shelve`) behind a shared `ICacher` interface in `common.py`. Important conventions:
- Build keys via `ICacher.create_cache_key` with clear plain + hashed parts.
- Surface dedicated error types (e.g. `CacheEntryDoesNotExistError`) rather than generic exceptions; tests assert on these.
- When caching is disabled, return/raise explicitly — don't silently no-op.

### `toolchemy/utils/`
Cross-cutting helpers: `logger.get_logger` (always use this for logging — do not call `logging.getLogger` directly), `timer.py`, `datestimes.py`, `locations.py`, `at_exit_collector.py`, and general helpers in `utils.py`. Add new small helpers here rather than duplicating them elsewhere.

### `toolchemy/vision/`, `toolchemy/nlp/`, `toolchemy/db/`
Domain-specific utilities (image ops, text helpers, DB wrappers — notably `tinydb`).

## Conventions worth remembering

- Line length ≤ 160 chars.
- Python 3.10+ typing syntax (`str | None`, `list[str]`); `TypedDict` for dict payloads, `@dataclass` for plain containers, `pydantic.BaseModel` for config models.
- Explicit signatures — avoid `*args/**kwargs` unless truly needed.
- Keep edits ASCII-only unless the file already uses Unicode.
- Don't add dependencies unless necessary; check `pyproject.toml` first.
