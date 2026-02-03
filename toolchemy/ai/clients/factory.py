import logging

from toolchemy.ai.clients.common import LLMClientBase, ModelConfig
from toolchemy.ai.clients import OpenAIClient, OllamaClient, GeminiClient, DummyModelClient
from toolchemy.utils.logger import get_logger

URI_OPENAI = "openai"
URI_GEMINI = "gemini"
URI_DUMMY = "dummy"


def create_llm(name: str, uri: str | None = None, api_key: str | None = None, default_model_config: ModelConfig | None = None, system_prompt: str | None = None, log_level: int = logging.INFO, no_cache: bool = False) -> LLMClientBase:
    logger = get_logger(level=log_level)
    logger.debug(f"Creating llm instance")
    logger.debug(f"> name: {name}")
    if name.startswith("gpt") and not name.startswith("gpt-oss"):
        uri = URI_OPENAI
    elif name.startswith("gemini"):
        uri = URI_GEMINI
    elif uri is None:
        raise ValueError(f"Cannot assume the LLM provider based on the model name: '{name}'. You can pass the uri explicitly as parameter for this function.'")
    logger.debug(f"> uri: {uri}")
    logger.debug(f"> uri assumed: {uri}")

    if uri == URI_OPENAI:
        if not api_key:
            raise ValueError(f"you must pass the 'api_key' explicitly as parameter for this function.")
        return OpenAIClient(model_name=name, api_key=api_key, system_prompt=system_prompt, default_model_config=default_model_config, no_cache=no_cache)

    if uri == URI_GEMINI:
        if not api_key:
            raise ValueError(f"you must pass the 'api_key' explicitly as parameter for this function.")
        return GeminiClient(default_model_name=name, api_key=api_key, system_prompt=system_prompt, default_model_config=default_model_config,
                            disable_cache=no_cache, log_level=log_level)

    return OllamaClient(uri=uri, model_name=name, system_prompt=system_prompt, default_model_config=default_model_config, truncate_log_messages_to=2000,
                        disable_cache=no_cache, log_level=log_level)
