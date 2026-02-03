import time
import logging
from google import genai
from google.genai import types

from toolchemy.utils.cacher import ICacher
from toolchemy.ai.clients.common import LLMClientBase, ModelConfig, Usage


class GeminiClient(LLMClientBase):
    def __init__(self, api_key: str, default_model_name: str | None = None, default_embedding_model_name: str | None = None,
                 default_model_config: ModelConfig | None = None,
                 system_prompt: str | None = None, keep_chat_session: bool = False,
                 retry_attempts: int = 5, retry_min_wait: int = 2, retry_max_wait: int = 60,
                 truncate_log_messages_to: int = 200,
                 fix_malformed_json: bool = True,
                 cacher: ICacher | None = None, disable_cache: bool = False, log_level: int = logging.INFO):
        super().__init__(default_model_name=default_model_name, default_embedding_model_name=default_embedding_model_name,
                         default_model_config=default_model_config,
                         system_prompt=system_prompt, keep_chat_session=keep_chat_session,
                         retry_attempts=retry_attempts, retry_min_wait=retry_min_wait, retry_max_wait=retry_max_wait,
                         truncate_log_messages_to=truncate_log_messages_to,
                         fix_malformed_json=fix_malformed_json,
                         cacher=cacher, disable_cache=disable_cache, log_level=log_level)
        self._client = genai.Client(api_key=api_key)
        self._logger.debug(f"Gemini client has been initialized")

    def embeddings(self, text: str) -> list[float]:
        raise NotImplementedError()

    def _completion(self, prompt: str, system_prompt: str | None, model_config: ModelConfig | None = None,
                    images_base64: list[str] | None = None) -> tuple[str, Usage]:
        system_prompt = system_prompt or self._system_prompt

        duration_time_start = time.time()

        response = self._client.models.generate_content(
            model=model_config.model_name,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=prompt
        )

        duration = time.time() - duration_time_start

        usage = Usage(input_tokens=response.usage_metadata.prompt_token_count,
                      output_tokens=response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count, duration=duration)

        return response.text, usage
