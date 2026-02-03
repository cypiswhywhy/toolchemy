import logging
from ollama import Client

from toolchemy.ai.clients.common import LLMClientBase, ModelConfig, Usage
from toolchemy.utils.cacher import ICacher
from toolchemy.utils.datestimes import Seconds


class OllamaClient(LLMClientBase):
    def __init__(self, uri: str, model_name: str | None = None, embedding_model_name: str | None = "nomic-embed-text",
                 default_model_config: ModelConfig | None = None, system_prompt: str | None = None,
                 keep_chat_session: bool = False,
                 retry_attempts: int = 5, retry_min_wait: int = 2, retry_max_wait: int = 60,
                 truncate_log_messages_to: int = 200, fix_malformed_json: bool = True,
                 cacher: ICacher | None = None, disable_cache: bool = False, log_level: int = logging.INFO):
        super().__init__(default_model_name=model_name, default_embedding_model_name=embedding_model_name,
                         default_model_config=default_model_config,
                         system_prompt=system_prompt, keep_chat_session=keep_chat_session,
                         retry_attempts=retry_attempts, retry_min_wait=retry_min_wait, retry_max_wait=retry_max_wait,
                         truncate_log_messages_to=truncate_log_messages_to, fix_malformed_json=fix_malformed_json,
                         cacher=cacher, disable_cache=disable_cache, log_level=log_level)
        self._uri = uri
        self._metadata["uri"] = self._uri
        assert self._uri, f"The model uri cannot be empty!"

        self._client = Client(host=self._uri)
        self._logger.debug(f"OLlama client has been initialized ({self._uri})")

    def embeddings(self, text: str) -> list[float]:
        cache_key = self._cacher.create_cache_key(["llm_embeddings"], [text])
        if self._cacher.exists(cache_key):
            self._logger.debug(f"Cache for the text embeddings already exists")
            return self._cacher.get(cache_key)

        results_raw = self._client.embed(model=self.embedding_name, input=text)
        results = [v for v in results_raw.embeddings[0]]

        self._cacher.set(cache_key, results)

        return results

    def _completion(self, prompt: str, system_prompt: str | None = None, model_config: ModelConfig | None = None,
                   images_base64: list[str] | None = None) -> tuple[str, Usage]:

        system_prompt = system_prompt or self.system_prompt
        result = self._client.generate(model=model_config.model_name, system=system_prompt, prompt=prompt,
                                       options={
                                           "temperature": model_config.temperature,
                                           "top_p": model_config.top_p,
                                           "num_predict": model_config.max_new_tokens,
                                       }, images=images_base64)

        total_duration_s = result.total_duration * Seconds.NANOSECOND
        usage = Usage(input_tokens=result.prompt_eval_count, output_tokens=result.eval_count, duration=total_duration_s)

        self._logger.debug(f"Completion done.")

        return result.response, usage
