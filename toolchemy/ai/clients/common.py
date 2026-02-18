import logging
import json

from jsonschema import validate, ValidationError
from tenacity import wait_exponential, Retrying, before_sleep_log, stop_after_attempt
from dataclasses import dataclass
from json import JSONDecodeError
from json.decoder import JSONDecodeError as JSONDecoderDecodeError
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypedDict, NotRequired

from toolchemy.utils.logger import get_logger
from toolchemy.utils.utils import ff, truncate
from toolchemy.utils.cacher import Cacher, DummyCacher, ICacher
from toolchemy.utils.at_exit_collector import ICollectable, AtExitCollector
from toolchemy.ai.clients.pricing import Pricing


class LLMCacheDoesNotExist(Exception):
    pass


class ModelResponseError(Exception):
    pass


class ModelConfig(BaseModel):
    model_name: str | None = None

    max_new_tokens: int | None = 2000
    presence_penalty: float | None = 0.0

    temperature: float = 0.7
    top_p: float = 1.0

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.model_name}__{self.max_new_tokens}__{ff(self.presence_penalty)}__{ff(self.temperature)}__{ff(self.top_p)}"

    @classmethod
    def from_raw(cls, data: dict) -> "ModelConfig":
        return cls(model_name=data["model_name"], max_new_tokens=int(data["max_new_tokens"]),
                   presence_penalty=data["presence_penalty"], temperature=data["temperature"], top_p=data["top_p"])

    def raw(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "presence_penalty": self.presence_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    duration: float
    cached: bool = False

    def __eq__(self, other: "Usage"):
        return other.input_tokens == self.input_tokens and other.output_tokens == self.output_tokens and other.duration == self.duration and other.duration == self.duration


class ILLMClient(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def model_config(self, base_config: ModelConfig | None = None, default_model_name: str | None = None) -> ModelConfig:
        pass

    @abstractmethod
    def completion(self, prompt: str, system_prompt: str | None = None, model_config: ModelConfig | None = None, images_base64: list[str] | None = None,
                   no_cache: bool = False, cache_only: bool = False) -> str:
        pass

    @abstractmethod
    def completion_json(self, prompt: str, system_prompt: str | None = None, model_config: ModelConfig | None = None, images_base64: list[str] | None = None,
                        validation_schema: dict | None = None,
                        no_cache: bool = False, cache_only: bool = False) -> dict | list[dict]:
        pass

    @abstractmethod
    def invalidate_completion_cache(self, prompt: str, model_config: ModelConfig | None = None, images_base64: list[str] | None = None):
        pass

    @abstractmethod
    def embeddings(self, text: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def embeddings_size(self) -> int:
        pass

    @property
    @abstractmethod
    def usage_summary(self) -> dict:
        pass

    @abstractmethod
    def usage(self, tail: int | None = None) -> list[Usage]:
        pass


class ChatMessageContent(TypedDict):
    type: str
    text: NotRequired[str]
    image_url: NotRequired[str]


class ChatMessage(TypedDict):
    role: str
    content: str | list[ChatMessageContent]


class ChatMessages(TypedDict):
    messages: list[ChatMessage]


class LLMClientBase(ILLMClient, ICollectable, ABC):
    def __init__(self, default_model_name: str | None = None, default_embedding_model_name: str | None = None,
                 default_model_config: ModelConfig | None = None,
                 system_prompt: str | None = None, keep_chat_session: bool = False,
                 retry_attempts: int = 5, retry_min_wait: int = 2, retry_max_wait: int = 60,
                 truncate_log_messages_to: int = 200,
                 fix_malformed_json: bool = True,
                 cacher: ICacher | None = None, disable_cache: bool = False, log_level: int = logging.INFO):
        self._logger = get_logger(__name__, level=log_level)
        if disable_cache:
            self._cacher = DummyCacher()
        else:
            if cacher is None:
                self._cacher = Cacher(log_level=log_level)
            else:
                self._cacher = cacher.sub_cacher(log_level=log_level)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self._model_name = default_model_name
        self._metadata = {"name": self._model_name}
        self._embedding_model_name = default_embedding_model_name
        self._default_model_config = default_model_config
        if self._default_model_config is None:
            self._default_model_config = ModelConfig()
        self._session_messages = []
        self._system_prompt = system_prompt
        self._keep_chat_session = keep_chat_session
        self._usages = []
        self._truncate_log_messages_to = truncate_log_messages_to
        self._retryer = Retrying(stop=stop_after_attempt(retry_attempts),
                                 wait=wait_exponential(multiplier=1, min=retry_min_wait, max=retry_max_wait),
                                 before_sleep=before_sleep_log(self._logger, log_level=log_level))
        self._fix_malformed_json = fix_malformed_json
        self._prompter = None
        if self._fix_malformed_json:
            self._fix_json_prompt_template = """Below is a malformed JSON object. Your task is to fix it to be a valid JSON, preserving all the data it already contains.
You must return only the fixed JSON object, no additional comments or explanations, just a fixed valid JSON!

Malformed JSON object:
{json_object}"""
        AtExitCollector.register(self)

    def name(self) -> str:
        model_config = self.model_config(default_model_name=self._model_name)
        return model_config.model_name

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def usage_summary(self) -> dict:
        input_tokens = sum([usage.input_tokens for usage in self._usages if not usage.cached])
        output_tokens = sum([usage.output_tokens for usage in self._usages if not usage.cached])
        cached_input_tokens = sum([usage.input_tokens for usage in self._usages if usage.cached])
        cached_output_tokens = sum([usage.output_tokens for usage in self._usages if usage.cached])
        total_usage = {
            "request_count": len([usage for usage in self._usages if not usage.cached]),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": 0,
            "duration": sum([round(usage.duration) for usage in self._usages if not usage.cached]),
            "duration_avg": 0.0,
            "cost": Pricing.estimate(self.name(), input_tokens=input_tokens, output_tokens=output_tokens),
            "cached_request_count": len([usage for usage in self._usages if usage.cached]),
            "cached_input_tokens": cached_input_tokens,
            "cached_output_tokens": cached_output_tokens,
            "cached_total_tokens": 0,
            "cached_duration": sum([round(usage.duration) for usage in self._usages if usage.cached]),
            "cached_duration_avg": 0.0,
            "cached_cost": Pricing.estimate(self.name(), input_tokens=cached_input_tokens, output_tokens=cached_output_tokens)
        }

        total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
        total_usage["cached_total_tokens"] = total_usage["cached_input_tokens"] + total_usage["cached_output_tokens"]
        total_usage["duration_avg"] = float(total_usage["duration"] / total_usage["request_count"]) if total_usage["request_count"] else 0.0
        total_usage["cached_duration_avg"] = float(total_usage["cached_duration"] / total_usage["cached_request_count"]) if total_usage["cached_request_count"] else 0.0

        return total_usage

    def collect(self) -> dict:
        return self.usage_summary

    def label(self) -> str:
        return f"{self.__class__.__name__}({self.name()})"

    def usage(self, tail: int | None = None) -> list[Usage]:
        tail = tail or len(self._usages)
        usages = self._usages[-tail:]
        return usages

    @property
    def embedding_name(self) -> str:
        return self._embedding_model_name

    @property
    def embeddings_size(self) -> int:
        raise NotImplementedError("not yet implemented")

    def model_config(self, base_config: ModelConfig | None = None, default_model_name: str | None = None) -> ModelConfig:
        if base_config is None:
            base_config = self._default_model_config.model_copy()
        if base_config.model_name is None:
            if default_model_name is None:
                raise RuntimeError(f"Model name or default model must be set")
            base_config.model_name = self._model_name
        return base_config

    @abstractmethod
    def _completion(self, prompt: str, system_prompt: str | None, model_config: ModelConfig | None = None,
                   images_base64: list[str] | None = None) -> tuple[str, Usage]:
        pass

    def invalidate_completion_cache(self, prompt: str, model_config: ModelConfig | None = None, images_base64: list[str] | None = None):
        model_config = self.model_config(model_config, self._model_name)
        cache_key, cache_key_usage = self._cache_keys_completion(system_prompt=self._system_prompt, prompt=prompt,
                                                                 model_config=model_config, images_base64=images_base64,
                                                                 is_json=False)
        cache_key_json, cache_key_usage_json = self._cache_keys_completion(system_prompt=self._system_prompt, prompt=prompt,
                                                                           model_config=model_config, images_base64=images_base64,
                                                                           is_json=True)

        self._cacher.unset(cache_key)
        self._cacher.unset(cache_key_usage)
        self._cacher.unset(cache_key_json)
        self._cacher.unset(cache_key_usage_json)

    def completion_json(self, prompt: str, system_prompt: str | None = None, model_config: ModelConfig | None = None,
                        images_base64: list[str] | None = None, validation_schema: dict | None = None,
                        no_cache: bool = False, cache_only: bool = False) -> dict | list[dict]:
        model_cfg = self.model_config(model_config, self._model_name)
        system_prompt = system_prompt or self._system_prompt
        self._logger.debug(f"CompletionJSON started (model: '{model_cfg.model_name}', max_len: {model_cfg.max_new_tokens}, temp: {model_cfg.max_new_tokens}), top_p: {model_cfg.top_p})")
        self._logger.debug(f"> Model config (mod): model: {model_cfg.model_name}, max_new_tokens: {model_cfg.max_new_tokens}, temp: {model_cfg.temperature}")

        cache_key, cache_key_usage = self._cache_keys_completion(system_prompt=system_prompt, prompt=prompt,
                                                                 model_config=model_cfg, images_base64=images_base64,
                                                                 is_json=True)
        if not no_cache and self._cacher.exists(cache_key) and self._cacher.exists(cache_key_usage):
            self._logger.debug(f"Cache for completion_json already exists ('{cache_key}')")
            usage = self._cacher.get(cache_key_usage)
            usage.cached = True
            self._usages.append(usage)
            return self._cacher.get(cache_key)

        if cache_only:
            raise LLMCacheDoesNotExist()

        self._logger.debug(f"Cache for completion_json does not exists, generating new response")

        try:
            response_json, usage = self._retryer(self._completion_json, prompt=prompt, system_prompt=system_prompt,
                                                 model_config=model_cfg,
                                                 images_base64=images_base64,
                                                 validation_schema=validation_schema)
        except Exception:
            self._logger.error(f"> system prompt: {system_prompt}")
            self._logger.error(f"> prompt: {prompt}")
            self._logger.error(f"> model config: {model_config.raw()}")
            raise
        self._usages.append(usage)

        if not no_cache:
            self._cacher.set(cache_key, response_json)
            self._cacher.set(cache_key_usage, usage)

        return response_json

    def _completion_json(self, prompt: str, system_prompt: str, model_config: ModelConfig, images_base64: list[str] | None,
                         validation_schema: dict | None = None) -> tuple[dict | list[dict], Usage]:
        response_str, usage = self._completion(prompt=prompt, system_prompt=system_prompt, model_config=model_config,
                                               images_base64=images_base64)

        response_json = None

        response_str = response_str.replace("```json", "").replace("```", "").strip()

        try:
            response_json = self._decode_json(response_str)
            if validation_schema:
                try:
                    validate(instance=response_json, schema=validation_schema)
                except ValidationError as e:
                    self._logger.error(f"Invalid schema: {e}")
                    raise e

        except (JSONDecodeError, JSONDecoderDecodeError) as e:
            if self._fix_malformed_json and self._fix_json_prompt_template:
                self._logger.warning("Malformed JSON, trying to fix it...")
                self._logger.warning(f"Malformed JSON:\n'{response_str}'")
                fix_json_prompt = self._fix_json_prompt_template.format(json_object=response_str)
                response_json = self.completion_json(fix_json_prompt)
                self._logger.debug(f"Fixed JSON:\n{response_json}")
            if response_json is None:
                self._logger.error(f"Invalid JSON:\n{truncate(response_str, self._truncate_log_messages_to)}\n")
                self._logger.error(f"> prompt:\n{truncate(prompt, self._truncate_log_messages_to)}")
                raise e

        return response_json, usage

    @staticmethod
    def _decode_json(json_str: str) -> dict | list[dict]:
        try:
            content = json.loads(json_str)
            return content
        except (JSONDecodeError, JSONDecodeError) as e:
            pass

        lines = [line.strip() for line in json_str.strip().split('\n') if line.strip()]
        parsed_data = []

        starting_line = 0
        if len(lines[0]) == 1 and lines[0][0] == "l":
            starting_line = 1

        for i, line in enumerate(lines):
            if starting_line == 1 and i == 0:
                continue
            parsed_data.append(json.loads(line))

        if len(parsed_data) == 1 and starting_line == 0:
            parsed_data = parsed_data[0]

        return parsed_data

    def completion(self, prompt: str, system_prompt: str | None = None, model_config: ModelConfig | None = None,
                   images_base64: list[str] | None = None, no_cache: bool = False, cache_only: bool = False) -> str:
        model_config = self.model_config(model_config, self._model_name)
        system_prompt = system_prompt or self._system_prompt
        self._logger.debug(f"Completion started (model: {model_config.model_name})")

        cache_key, cache_key_usage = self._cache_keys_completion(system_prompt=system_prompt, prompt=prompt,
                                                                 model_config=model_config, images_base64=images_base64,
                                                                 is_json=False)
        if not no_cache and self._cacher.exists(cache_key) and self._cacher.exists(cache_key_usage):
            self._logger.debug(f"Cache for the prompt already exists ('{cache_key}')")
            usage_cached = self._cacher.get(cache_key_usage)
            usage_cached.cached = True
            self._usages.append(usage_cached)
            return self._cacher.get(cache_key)

        if cache_only:
            raise LLMCacheDoesNotExist()

        try:
            response, usage = self._retryer(self._completion, prompt=prompt, system_prompt=system_prompt,
                                        model_config=model_config,
                                        images_base64=images_base64)
        except Exception:
            self._logger.error(f"> system prompt: {system_prompt}")
            self._logger.error(f"> prompt: {prompt}")
            self._logger.error(f"> model config: {model_config.raw()}")
            raise

        self._usages.append(usage)

        if not no_cache:
            self._cacher.set(cache_key, response)
            self._cacher.set(cache_key_usage, usage)

        self._logger.debug(f"Completion done.")

        return response

    def _cache_keys_completion(self, system_prompt: str, prompt: str, model_config: ModelConfig,
                               images_base64: list[str] | None = None, is_json: bool = False) -> tuple[str, str]:
        cache_key_part_images = images_base64 or []
        json_suffix = "_json" if is_json else ""
        cache_key = self._cacher.create_cache_key([f"llm_completion{json_suffix}"],
                                                  [system_prompt, prompt, str(model_config),
                                                   "_".join(cache_key_part_images)])
        cache_key_usage = self._cacher.create_cache_key([f"llm_completion{json_suffix}__usage"],
                                                  [system_prompt, prompt, str(model_config),
                                                   "_".join(cache_key_part_images)])

        return cache_key, cache_key_usage


def prepare_chat_messages(prompt: str, system_prompt: str | None = None, images_base64: list[str] | None = None,
                          messages_history: list[ChatMessage] | None = None, envelope: bool = False) -> list[ChatMessage] | ChatMessages:
    messages_all = messages_history or []
    if system_prompt:
        if messages_all and len(messages_all) > 0:
            if not messages_all[0]["role"] == "system":
                messages_all = [{"role": "system", "content": system_prompt}] + messages_all
        else:
            messages_all = [{"role": "system", "content": system_prompt}]

    user_message: ChatMessage = {"role": "user", "content": prompt}

    if images_base64:
        user_message = {"role": "user", "content": []}
        user_message["content"].append({"type": "input_text", "text": prompt})
        for b64_image in images_base64:
            user_message["content"].append({"type": "input_message", "image_url": f"data:image/png;base64,{b64_image}"})

    messages_all.append(user_message)

    if envelope:
        return ChatMessages(messages=messages_all)

    return messages_all
