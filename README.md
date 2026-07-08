# Toolchemy

A Python library of reusable tooling for LLM-backed projects: unified clients for OpenAI/Gemini/Ollama/Whisper with built-in caching, retry, and cost accounting; pluggable experiment/prompt trackers; and general-purpose caching, logging, and utility helpers.

## Quickstart

```bash
poetry add toolchemy
```

```python
from toolchemy.ai.clients import create_llm

llm = create_llm(name="gpt-4.1-mini", api_key="sk-...")
print(llm.completion("Summarize the plot of Hamlet in one sentence."))
```

## Documentation

- [Tutorials](docs/tutorials.md) — a first walkthrough of the LLM client + cache + cost accounting.
- [How-To Guides](docs/how-to-guides.md) — install, test, choose a cache backend, add pricing, run `prompt-studio`/`agent-synergy`, publish.
- [Reference](docs/reference.md) — console scripts, module map, config shapes; links to the generated `AGENTS_MANIFEST.md`.
- [Explanation](docs/explanation.md) — architecture, caching design, retry/repair flow.

See `AGENTS.md` / `CLAUDE.md` for contributor conventions.

## Development

```bash
poetry install
make lint
make test
```

### Publishing

```bash
poetry build
poetry publish --repository pypi
```
