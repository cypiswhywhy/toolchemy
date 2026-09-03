import logging

from toolchemy.ai.clients.common import LLMClientBase, ModelConfig, Usage


class DummyModelClient(LLMClientBase):
    """
    Client that answers locally instead of calling a model, for tests and offline runs.

    Echoes the prompt back, or returns `fixed_response` when one is given.
    """

    DEFAULT_MODEL_NAME = "dummy-model"
    EMBEDDINGS_SIZE = 32

    def __init__(self, name: str = DEFAULT_MODEL_NAME, fixed_response: str | None = None,
                 system_prompt: str | None = None, default_model_config: ModelConfig | None = None,
                 disable_cache: bool = True, log_level: int = logging.INFO):
        super().__init__(default_model_name=name, system_prompt=system_prompt,
                         default_model_config=default_model_config,
                         disable_cache=disable_cache, log_level=log_level)
        self._fixed_response = fixed_response

    @property
    def system_prompt(self) -> str:
        return self._system_prompt or "You are a dummy AI Assistant"

    @property
    def embeddings_size(self) -> int:
        return self.EMBEDDINGS_SIZE

    def embeddings(self, text: str) -> list[float]:
        return self.EMBEDDINGS_SIZE * [0.98]

    def _completion(self, prompt: str, system_prompt: str | None, model_config: ModelConfig | None = None,
                    images_base64: list[str] | None = None) -> tuple[str, Usage]:
        response = self._fixed_response if self._fixed_response is not None else f"Echo: {prompt}"
        return response, Usage(input_tokens=0, output_tokens=0, duration=0.0)
