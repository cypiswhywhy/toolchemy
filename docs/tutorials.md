# Tutorials

## Your first cached LLM call

This walks through wiring up an LLM client, calling it, and inspecting cost/usage — the core loop most `toolchemy` consumers need.

### 1. Install

```bash
poetry add toolchemy
```

### 2. Create a client

`toolchemy.ai.clients.create_llm` picks the right provider client from the model name:

```python
from toolchemy.ai.clients import create_llm

llm = create_llm(
    name="gpt-4.1-mini",
    api_key="sk-...",
    system_prompt="You are a terse assistant.",
)
```

Model names starting with `gpt` (but not `gpt-oss`) resolve to the OpenAI client, `gemini*` resolves to Gemini. Anything else needs an explicit `uri` (e.g. `uri="http://localhost:11434"` for a local Ollama model).

### 3. Get a completion

```python
answer = llm.completion("Summarize the plot of Hamlet in one sentence.")
print(answer)
```

Under the hood this:
- builds a cache key from the prompt, system prompt, and model config,
- returns the cached response if one exists, skipping the network call entirely,
- otherwise calls the provider (retrying transient failures via `tenacity`) and caches the result.

Run the same call twice — the second run returns instantly from cache and is marked `cached=True` in the usage log.

### 4. Ask for structured JSON

```python
data = llm.completion_json(
    "Return {\"title\": str, \"year\": int} for the movie Inception.",
    validation_schema={
        "type": "object",
        "properties": {"title": {"type": "string"}, "year": {"type": "integer"}},
        "required": ["title", "year"],
    },
)
```

If the model returns malformed JSON, the client automatically asks the model to fix it before giving up (see [explanation.md](explanation.md) for the retry/repair flow).

### 5. Check what it cost

```python
print(llm.usage_summary)
```

`usage_summary` aggregates token counts, request counts, and an estimated dollar cost (via `toolchemy.ai.clients.pricing.Pricing`) split into cached vs. non-cached calls.

### Next steps

- Swap the disk cache for a different backend — see [how-to-guides.md](how-to-guides.md#choose-a-cache-backend).
- Add prompt versioning with `prompt-studio` — see [how-to-guides.md](how-to-guides.md#run-the-prompt-studio-ui).
- Read [explanation.md](explanation.md) for why the library is structured this way.
