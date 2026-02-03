import json
from toolchemy.ai.clients.common import ILLMClient, ModelConfig, Usage


class DummyModelClient(ILLMClient):
    def __init__(self, name: str = "dummy", fixed_response: str | None = None):
        self._name = name
        self._fixed_response = fixed_response
        self._metadata = {"name": self._name}
        self._usages = []

    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> dict:
        return self._metadata

    def usage(self, tail: int | None = None) -> list[Usage]:
        tail = tail or len(self._usages)
        usages = self._usages[-tail:]
        return usages

    @property
    def system_prompt(self) -> str:
        return "You are a dummy AI Assistant"

    def embeddings(self, text: str, model_name: str = "nomic-embed-text") -> list[float]:
        return 32 * [0.98]

    def _completion(self, prompt: str, system_prompt: str | None, model_config: ModelConfig | None = None,
                    images_base64: list[str] | None = None) -> tuple[str, Usage]:
        if self._fixed_response:
            model_response = self._fixed_response
        else:
            model_response = f"Echo: {prompt}"
        return model_response, Usage(input_tokens=0, output_tokens=0, duration=0.0)

    def completion(self, prompt: str, model_config: ModelConfig | None = None,
                   images_base64: list[str] | None = None, no_cache: bool = False, cache_only: bool = False) -> str:
        response, usage = self._completion(prompt=prompt, system_prompt=self.system_prompt)
        self._usages.append(usage)
        return response

    def completion_json(self, prompt: str, model_config: ModelConfig | None = None,
                        images_base64: list[str] | None = None, validation_schema: dict | None = None,
                        no_cache: bool = False, cache_only: bool = False) -> dict | list[dict]:
        result_str = self.completion(prompt=prompt, model_config=model_config, images_base64=images_base64, cache_only=cache_only)
        return json.loads(result_str)

    def model_config(self, base_config: ModelConfig | None = None,
                     default_model_name: str | None = None) -> ModelConfig:
        return base_config

    @property
    def usage_summary(self) -> dict:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
