import time
from abc import ABC, abstractmethod
from openai import OpenAI, AzureOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing import Iterable

from toolchemy.ai.clients.common import LLMClientBase, ModelConfig, Usage, prepare_chat_messages


class BaseOpenAIClient(LLMClientBase, ABC):
    @property
    @abstractmethod
    def _client(self) -> OpenAI:
        pass

    def embeddings(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self.embedding_name
        )
        return response.data[0].embedding

    def _completion(self, prompt: str, system_prompt: str | None, model_config: ModelConfig | None = None,
                    images_base64: list[str] | None = None) -> tuple[str, Usage]:
        messages = self._prepare_chat_messages(prompt=prompt, system_prompt=system_prompt, images_base64=images_base64)

        duration_time_start = time.time()
        response = self._client.chat.completions.create(
            model=model_config.model_name,
            messages=messages,
            top_p=model_config.top_p,
        )
        duration = time.time() - duration_time_start

        usage = Usage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                      duration=duration)

        return response.choices[0].message.content, usage

    def _prepare_chat_messages(self, prompt: str, system_prompt: str | None = None, images_base64: list[str] | None = None) -> list[ChatCompletionMessageParam]:
        messages_all = []
        system_prompt = system_prompt or self._system_prompt or NOT_GIVEN

        if self._keep_chat_session:
            messages_all.extend(self._session_messages)

        return prepare_chat_messages(prompt=prompt, system_prompt=system_prompt, images_base64=images_base64, messages_history=messages_all)


class OpenAIClient(BaseOpenAIClient):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo",
                 embedding_model_name: str = "text-embedding-3-large", default_model_config: ModelConfig | None = None,
                 system_prompt: str | None = None, keep_chat_session: bool = False, no_cache: bool = False):
        super().__init__(default_model_name=model_name, default_embedding_model_name=embedding_model_name,
                         default_model_config=default_model_config,
                         system_prompt=system_prompt, keep_chat_session=keep_chat_session, disable_cache=no_cache)
        self._openai_client = OpenAI(api_key=api_key)

    @property
    def _client(self) -> OpenAI:
        return self._openai_client


class AzureOpenAIClient(BaseOpenAIClient):
    def __init__(self, api_key: str, api_endpoint: str, api_version: str, model_name: str = "gpt-3.5-turbo",
                 embedding_model_name: str = "text-embedding-3-large", default_model_config: ModelConfig | None = None,
                 system_prompt: str | None = None, keep_chat_session: bool = False):
        super().__init__(default_model_name=model_name, default_embedding_model_name=embedding_model_name,
                         default_model_config=default_model_config,
                         system_prompt=system_prompt, keep_chat_session=keep_chat_session)
        self._openai_client = AzureOpenAI(api_key=api_key, azure_endpoint=api_endpoint, api_version=api_version,
                                          azure_deployment=model_name)

    @property
    def _client(self) -> OpenAI:
        return self._openai_client
