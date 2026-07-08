# Reference

## Full API manifest

`toolchemy/AGENTS_MANIFEST.md` is a generated, exhaustive listing of every public function and class in the package (import path, signature, one-line description). It's regenerated with `make docs-agents` (see [how-to-guides.md](how-to-guides.md#regenerate-the-api-manifest)) and is the authoritative symbol reference — this page indexes modules and the stable surfaces (console scripts, config shapes) instead of duplicating it.

## Console scripts

Defined under `[tool.poetry.scripts]` in `pyproject.toml`:

| Script | Entry point | Purpose |
|---|---|---|
| `prompt-studio` | `toolchemy.ai.prompting.prompter_mlflow:run_studio` | Launches an MLflow-backed UI for browsing/versioning prompt templates. |
| `agent-synergy` | `toolchemy.agent_synergy:main` | Injects a pointer to the installed `AGENTS_MANIFEST.md` into a downstream project's `AGENTS.md`/`CLAUDE.md`. |

There's also a `toolchemy` CLI group (`toolchemy/__main__.py`, run via `python -m toolchemy`) exposing `prompt-studio` as a subcommand with a `--registry-dir/-r` option.

## Module map

| Module | Contents |
|---|---|
| `toolchemy.ai.clients` | `ILLMClient`/`LLMClientBase` interface, provider clients (`OpenAIClient`, `GeminiClient`, `OllamaClient`, `DummyModelClient`, `WhisperClient` for ASR), `create_llm` factory, `Pricing`. |
| `toolchemy.ai.prompting` | `PrompterBase`/`IPromptOptimizer`/`Prompt` common types, `PrompterMLflow` (the `prompt-studio` backend), `SimpleLLMPromptOptimizer`. |
| `toolchemy.ai.trackers` | `ITracker`/`TrackerBase` interface plus `MLflowTracker`, `NeptuneTracker`, `InMemoryTracker` (test double). |
| `toolchemy.utils.cacher` | `ICacher`/`BaseCacher` interface, `CacherDiskcache`, `CacherPickle`, `CacherShelve`, `DummyCacher`, dedicated error types. |
| `toolchemy.utils` | `get_logger`, `Timer`, `Locations`, `AtExitCollector`/`ICollectable`, and general helpers (`ff`, `pp`, `hash_dict`, `to_json`, `truncate`, `batchize`, `split_text`, ...). |
| `toolchemy.db` | `LightTinyDB` — a `tinydb`-backed document store with optional field indexing and cache-guarded inserts. |
| `toolchemy.nlp` | `clean_text` — strips zero-width/control characters, URLs, and news boilerplate from text. |
| `toolchemy.vision` | `ImageProcessor`, `Caption`/`add_caption` for overlaying captions on images. |

## Key config/data shapes

- **`ModelConfig`** (`toolchemy.ai.clients.common`) — pydantic model: `model_name`, `max_new_tokens`, `presence_penalty`, `temperature`, `top_p`. `.raw()` / `.from_raw()` round-trip to/from plain dicts; `str(config)` renders a cache-key-safe fingerprint.
- **`Usage`** (`toolchemy.ai.clients.common`) — dataclass: `input_tokens`, `output_tokens`, `duration`, `cached`. Accumulated per-client and summarized by `usage_summary`.
- **`Filter`** / `FilterOp` (`toolchemy.db.lightdb`) — query filter for `LightTinyDB.search` (`EQUAL`, `GREATER`, `GREATER_OR_EQUAL`, `LESS`, `LESS_OR_EQUAL`).

## Pricing table

`toolchemy.ai.clients.pricing.Pricing.pricing_per_1_mln` holds USD-per-1M-token rates (input/output) for supported models (OpenAI GPT family, several local Ollama models, and a `dummy-model` entry for tests). Unlisted models return an estimated cost of `0.0` with a logged warning rather than raising — see [how-to-guides.md](how-to-guides.md#add-pricing-for-a-new-model) to add one.

## Environment / secrets

No environment variables are read implicitly by the library. Provider API keys (`api_key` for `create_llm`/`OpenAIClient`/`GeminiClient`) must be passed explicitly by the caller — `toolchemy` does not read them from the environment itself.
