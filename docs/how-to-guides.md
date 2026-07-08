# How-To Guides

## Install and set up the dev environment

```bash
poetry install
```

All commands below run through Poetry against Python 3.12.

## Run lint and tests

```bash
make lint            # pylint --rcfile pyproject.toml toolchemy
make test             # unit tests: poetry run pytest ./tests/unit
make test-all         # unit + int + perf
make test-perf        # pytest-benchmark suite only
```

Run a single test:

```bash
poetry run pytest tests/unit/path/to/test_file.py::TestClass::test_name
poetry run pytest tests/unit -k "keyword"
```

Test files end in `_test.py`, not `test_*.py`.

## Create an LLM client without guessing the provider

`create_llm` infers the provider from the model name (`gpt*` → OpenAI, `gemini*` → Gemini). For anything else (Ollama, local models), pass `uri` explicitly:

```python
from toolchemy.ai.clients import create_llm

llm = create_llm(name="mistral-small3.2:24b", uri="http://localhost:11434")
```

To disable caching for a single client (e.g. in tests), pass `no_cache=True`, or use `DummyModelClient` from `toolchemy.ai.clients` as a network-free test double.

## Choose a cache backend

All caches implement `ICacher` (`toolchemy/utils/cacher/common.py`). Pick one directly instead of going through `create_llm`'s default:

```python
from toolchemy.utils.cacher.cacher_diskcache import CacherDiskcache
from toolchemy.utils.cacher.cacher_pickle import CacherPickle
from toolchemy.utils.cacher.cacher_shelve import CacherShelve

cacher = CacherPickle(name="my-cache", cache_base_dir="./data/cache")
```

Pass it into a client via `cacher=...`, or into `LightTinyDB` (`toolchemy/db/lightdb.py`). To disable caching entirely (rather than skip a single call), use `DummyCacher` from `toolchemy.utils`.

Cache keys should be built with `ICacher.create_cache_key(parts_plain=..., parts_hashed=...)` — plain parts stay human-readable, hashed parts (e.g. full prompt text) get MD5'd.

## Add pricing for a new model

Token cost accounting lives in `toolchemy/ai/clients/pricing.py`. Add an entry to `Pricing.pricing_per_1_mln`:

```python
"my-new-model": {
    KEY_INPUT_TOKENS: 1.0,   # USD per 1M input tokens
    KEY_OUTPUT_TOKENS: 3.0,  # USD per 1M output tokens
},
```

If a model isn't listed, `Pricing.estimate` logs a warning and returns `0.0` rather than raising — cost accounting degrades gracefully for unsupported models.

## Run the prompt-studio UI

`prompt-studio` is a console script backed by `toolchemy.ai.prompting.prompter_mlflow.PrompterMLflow`, which stores prompt templates and versions in a local MLflow registry (SQLite-backed, under `prompts_mlflow/` by default):

```bash
poetry run prompt-studio
# or with a custom registry location:
poetry run prompt-studio --registry-dir ./my_registry
```

This launches `mlflow ui` pointed at the registry/tracking SQLite DBs. Use `PrompterMLflow.create_template` / `.render` / `.template` programmatically to register and render versioned prompts with variable substitution.

## Wire `agent-synergy` into a downstream project

Any project that depends on `toolchemy` can run:

```bash
poetry run agent-synergy --path /path/to/that/project
```

This inserts a marker block into that project's `AGENTS.md` and/or `CLAUDE.md` pointing at the installed package's `AGENTS_MANIFEST.md` (the generated API capability list — see [reference.md](reference.md#agents_manifestmd)), so coding agents working in that project know to reuse `toolchemy` symbols instead of reimplementing them. It's idempotent: re-running refreshes the resolved path without duplicating the block.

## Regenerate the API manifest

After adding or changing public functions/classes, regenerate `toolchemy/AGENTS_MANIFEST.md`:

```bash
make docs-agents
```

This introspects every public symbol in `toolchemy` and writes a fresh manifest (docstring first line, or an inferred description if none exists). Commit the regenerated file — it ships inside the package (see `pyproject.toml`'s `include`).

## Build and publish

```bash
poetry build
```

`make publish` bumps the patch version, builds, publishes to PyPI, commits, and tags — don't run it casually; it's a release action.
