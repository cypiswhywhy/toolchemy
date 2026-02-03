import pytest
from unittest.mock import patch, call
from tenacity import RetryError

from toolchemy.ai.clients.ollama_client import OllamaClient, ModelConfig, Usage
from toolchemy.utils.cacher import DummyCacher
from toolchemy.utils.datestimes import Seconds


class _OllamaClientGenerateResponse:
    def __init__(self, response: str):
        self._response: str = response
        self._prompt_eval_count = 1
        self._eval_count = 1
        self._total_duration = round(1.0 / Seconds.NANOSECOND)

    @property
    def response(self) -> str:
        return self._response

    @property
    def prompt_eval_count(self) -> int:
        return self._prompt_eval_count

    @property
    def eval_count(self) -> int:
        return self._eval_count

    @property
    def total_duration(self) -> float:
        return self._total_duration


class _OllamaClientChatMessageResponse:
    def __init__(self, prompt: str, response: str):
        self._prompt = prompt
        self._response: str = response

    @property
    def role(self) -> str:
        return "assistant"

    @property
    def content(self) -> str:
        return self._response


class _OllamaClientChatResponse:
    def __init__(self, prompt: str, response: str):
        self._prompt = prompt
        self._response = response
        self._message = _OllamaClientChatMessageResponse(prompt, response)

    @property
    def message(self) -> _OllamaClientChatMessageResponse:
        return self._message

    @property
    def prompt_eval_count(self) -> int:
        return len(self._prompt)

    @property
    def eval_count(self) -> int:
        return len(self._response)


def _prepare_client_response(value: str | Exception) -> _OllamaClientGenerateResponse | Exception:
    if isinstance(value, Exception):
        return value
    return _OllamaClientGenerateResponse(value)


def test_metadata():
    expected_metadata = {
        "name": "phi4",
        "uri": "http://uri"
    }
    client = OllamaClient(uri=expected_metadata["uri"], model_name=expected_metadata["name"], disable_cache=True, retry_min_wait=1, retry_attempts=2)
    assert expected_metadata == client.metadata


@pytest.mark.parametrize("client_class_mock_return_values,model_config", [
    (["expected response"], ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88)),
    (["expected response"], None),
    ([Exception("error"), "expected response"], ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88)),
    ([Exception("error"), "expected response"], None),
])
@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion(client_class_mock, client_class_mock_return_values, model_config):
    expected_model_name = "dummy-model"
    expected_response = "expected response"
    prompt = "test"

    expected_model_config = model_config or ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.side_effect = [_prepare_client_response(client_response) for client_response in client_class_mock_return_values]

    client = OllamaClient(uri="http://uri", model_name=expected_model_name, disable_cache=True, retry_min_wait=1, retry_attempts=2)
    response = client.completion(prompt=prompt, model_config=model_config)

    mock_client.generate.assert_called_with(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": expected_model_config.temperature,
        "top_p": expected_model_config.top_p,
        "num_predict": expected_model_config.max_new_tokens,
    }, images=None)

    assert response == expected_response

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 0,
        "cached_input_tokens": 0,
        "cached_output_tokens": 0,
        "cached_total_tokens": 0,
        "cached_duration": 0,
        "cached_duration_avg": 0.0,
        "cached_cost": 0.0,
    }

    assert expected_usage == client.usage_summary

    last_usages = client.usage(1)
    assert len(last_usages) == 1
    expected_usage = Usage(input_tokens=1, output_tokens=1, duration=1.0, cached=False)
    assert last_usages[0] == expected_usage


@pytest.mark.parametrize("client_class_mock_return_values,model_config", [
    (["expected response"], ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88)),
    (["expected response"], None),
    ([Exception("error"), "expected response"], ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88)),
    ([Exception("error"), "expected response"], None),
])
@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_cached(client_class_mock, client_class_mock_return_values, model_config):
    expected_model_name = "dummy-model"
    expected_response = "expected response"
    prompt = "test"

    cacher = DummyCacher(with_memory_store=True)

    expected_model_config = model_config or ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.side_effect = [_prepare_client_response(client_response) for client_response in client_class_mock_return_values]

    client = OllamaClient(uri="http://uri", model_name=expected_model_name, retry_min_wait=1, retry_attempts=2, cacher=cacher)

    response = client.completion(prompt=prompt, model_config=model_config)

    assert response == expected_response

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 0,
        "cached_input_tokens": 0,
        "cached_output_tokens": 0,
        "cached_total_tokens": 0,
        "cached_duration": 0,
        "cached_duration_avg": 0.0,
        "cached_cost": 0.0,
    }

    assert expected_usage == client.usage_summary

    response2 = client.completion(prompt=prompt, model_config=model_config)

    mock_client.generate.assert_called_with(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": expected_model_config.temperature,
        "top_p": expected_model_config.top_p,
        "num_predict": expected_model_config.max_new_tokens,
    }, images=None)

    assert response2 == expected_response

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 1,
        "cached_input_tokens": 1,
        "cached_output_tokens": 1,
        "cached_total_tokens": 2,
        "cached_duration": 1.0,
        "cached_duration_avg": 1.0,
        "cached_cost": 0.0000011,
    }

    assert expected_usage == client.usage_summary


@pytest.mark.parametrize("client_class_mock_return_values,model_config,validation_schema", [
    (["{\"expected\": \"response\"}"], ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88), None),
    (["{\"expected\": \"response\"}"], None, None),
    (["{\"expected\": \"response\"}"], None, {"type": "object", "properties": {"expected": {"type": "string"}}}),
    (["broken json str", "{\"expected\": \"response\"}"], ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88), None),
    (["broken json str", "{\"expected\": \"response\"}"], None, None),
])
@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_json(client_class_mock, client_class_mock_return_values, model_config, validation_schema):
    expected_model_name = "dummy-model"
    expected_response = {"expected": "response"}
    prompt = "test"

    expected_model_config = model_config or ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.side_effect = [_OllamaClientGenerateResponse(client_response) for client_response in client_class_mock_return_values]

    client = OllamaClient(uri="http://uri", model_name=expected_model_name, disable_cache=True, retry_min_wait=1, retry_attempts=2, fix_malformed_json=False)
    response = client.completion_json(prompt=prompt, model_config=model_config, validation_schema=validation_schema)

    mock_client.generate.assert_called_with(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": expected_model_config.temperature,
        "top_p": expected_model_config.top_p,
        "num_predict": expected_model_config.max_new_tokens,
    }, images=None)

    assert response == expected_response

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 0,
        "cached_input_tokens": 0,
        "cached_output_tokens": 0,
        "cached_total_tokens": 0,
        "cached_duration": 0,
        "cached_duration_avg": 0.0,
        "cached_cost": 0.0,
    }

    assert expected_usage == client.usage_summary


@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_json_invalid_schema(client_class_mock):
    expected_model_name = "dummy-model"
    expected_response = {"expected": "response"}
    prompt = "test"

    expected_model_config = ModelConfig()
    client_class_mock_return_values = "{\"expected\": \"response\"}"
    validation_schema = {"type": "object", "properties": {"unexpected": {"type": "string"}}}

    mock_client = client_class_mock.return_value
    mock_client.generate.side_effect = [_OllamaClientGenerateResponse(client_response) for client_response in client_class_mock_return_values]

    client = OllamaClient(uri="http://uri", model_name=expected_model_name, disable_cache=True, retry_min_wait=1, retry_attempts=2, fix_malformed_json=False)

    with pytest.raises(RetryError):
        _ = client.completion_json(prompt=prompt, model_config=expected_model_config, validation_schema=validation_schema)


@pytest.mark.parametrize("client_class_mock_return_values", [
    (["l\n{\"expected\": \"jsonl response 1\"}\n{\"expected\": \"jsonl response 2\"}"]),
    (["l\n{\"expected\": \"jsonl response 1\"}\n\n{\"expected\": \"jsonl response 2\"}"])
])
@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_json_jsonl_format(client_class_mock, client_class_mock_return_values):
    expected_model_name = "dummy-model"
    expected_response = [{"expected": "jsonl response 1"}, {"expected": "jsonl response 2"}]
    prompt = "test"

    model_config = ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.side_effect = [_OllamaClientGenerateResponse(client_response) for client_response in client_class_mock_return_values]

    client = OllamaClient(uri="http://uri", model_name=expected_model_name, disable_cache=True, retry_min_wait=1, retry_attempts=2, fix_malformed_json=False)
    response = client.completion_json(prompt=prompt, model_config=model_config)

    mock_client.generate.assert_called_with(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "num_predict": model_config.max_new_tokens,
    }, images=None)

    assert response == expected_response

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 0,
        "cached_input_tokens": 0,
        "cached_output_tokens": 0,
        "cached_total_tokens": 0,
        "cached_duration": 0,
        "cached_duration_avg": 0.0,
        "cached_cost": 0.0,
    }

    assert expected_usage == client.usage_summary


@pytest.mark.parametrize("client_class_mock_return_values", [
    (["l\n{\"expected\": \"jsonl response 1\"}\n{\"expected\": \"jsonl response 2\"}"])
])
@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_json_jsonl_format_with_fix_json_failed(client_class_mock, client_class_mock_return_values):
    expected_model_name = "dummy-model"
    expected_response = [{"expected": "jsonl response 1"}, {"expected": "jsonl response 2"}]
    prompt = "test"

    model_config = ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.side_effect = [_OllamaClientGenerateResponse(client_response) for client_response in client_class_mock_return_values]

    client = OllamaClient(uri="http://uri", model_name=expected_model_name, disable_cache=True, retry_min_wait=1, retry_attempts=2, fix_malformed_json=False)
    response = client.completion_json(prompt=prompt, model_config=model_config)

    expected_calls = [call(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "num_predict": model_config.max_new_tokens,
    }, images=None)]

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 0,
        "cached_input_tokens": 0,
        "cached_output_tokens": 0,
        "cached_total_tokens": 0,
        "cached_duration": 0,
        "cached_duration_avg": 0.0,
        "cached_cost": 0.0,
    }

    if client_class_mock_return_values[0].startswith("broken"):
        fix_json_prompt = client._fix_json_prompt_template.format(json_object=client_class_mock_return_values[0])
        expected_calls.append(call(model=expected_model_name, system=None, prompt=fix_json_prompt, options={
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "num_predict": model_config.max_new_tokens,
        }, images=None))

        expected_usage["input_tokens"] = 2
        expected_usage["output_tokens"] = 2
        expected_usage["total_tokens"] = 4
        expected_usage["duration"] = 2.0
        expected_usage["request_count"] = 2
        expected_usage["cost"] = 0.0000022

    mock_client.generate.assert_has_calls(expected_calls)

    assert response == expected_response

    assert expected_usage == client.usage_summary


@pytest.mark.parametrize("client_class_mock_return_values", [
    (["broken json str", "{\"expected\": \"response\"}"]),
])
@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_json_with_fix_json(client_class_mock, client_class_mock_return_values):
    expected_model_name = "dummy-model"
    expected_response = {"expected": "response"}
    prompt = "test"

    model_config = ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.side_effect = [_OllamaClientGenerateResponse(client_response) for client_response in client_class_mock_return_values]

    client = OllamaClient(uri="http://uri", model_name=expected_model_name, disable_cache=True, retry_min_wait=1, retry_attempts=2, fix_malformed_json=True)
    response = client.completion_json(prompt=prompt, model_config=model_config)

    expected_calls = [call(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "num_predict": model_config.max_new_tokens,
    }, images=None)]

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 0,
        "cached_input_tokens": 0,
        "cached_output_tokens": 0,
        "cached_total_tokens": 0,
        "cached_duration": 0,
        "cached_duration_avg": 0.0,
        "cached_cost": 0.0,
    }

    if client_class_mock_return_values[0].startswith("broken"):
        fix_json_prompt = client._fix_json_prompt_template.format(json_object=client_class_mock_return_values[0])
        expected_calls.append(call(model=expected_model_name, system=None, prompt=fix_json_prompt, options={
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "num_predict": model_config.max_new_tokens,
        }, images=None))

        expected_usage["input_tokens"] = 2
        expected_usage["output_tokens"] = 2
        expected_usage["total_tokens"] = 4
        expected_usage["duration"] = 2.0
        expected_usage["request_count"] = 2
        expected_usage["cost"] = 0.0000022

    mock_client.generate.assert_has_calls(expected_calls)

    assert response == expected_response

    assert expected_usage == client.usage_summary


@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_with_cache(client_class_mock):
    model_config = ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88)
    expected_model_name = "dummy-model"
    expected_response = "expected response"
    prompt = "test"

    expected_model_config = model_config or ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.return_value = _OllamaClientGenerateResponse(expected_response)

    cacher = DummyCacher(with_memory_store=True)
    client = OllamaClient(uri="http://uri", model_name=expected_model_name, cacher=cacher)
    response = client.completion(prompt=prompt, model_config=model_config)

    mock_client.generate.assert_called_with(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": expected_model_config.temperature,
        "top_p": expected_model_config.top_p,
        "num_predict": expected_model_config.max_new_tokens,
    }, images=None)

    expected_cache_key = cacher.create_cache_key(["llm_completion"], [None, prompt, str(model_config), "_".join([])])
    expected_cache_key_usage = cacher.create_cache_key(["llm_completion__usage"], [None, prompt, str(model_config), "_".join([])])

    assert response == expected_response
    assert client._cacher.exists(expected_cache_key)
    assert client._cacher.exists(expected_cache_key_usage)
    assert client._cacher.get(expected_cache_key) == response

    response = client.completion(prompt=prompt, model_config=model_config)

    mock_client.generate.assert_called_once()

    assert response == expected_response

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 1,
        "cached_input_tokens": 1,
        "cached_output_tokens": 1,
        "cached_total_tokens": 2,
        "cached_duration": 1.0,
        "cached_duration_avg": 1.0,
        "cached_cost": 0.0000011,
    }

    assert expected_usage == client.usage_summary


@patch("toolchemy.ai.clients.ollama_client.Client")
def test_completion_json_with_cache(client_class_mock):
    model_config = ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88)
    expected_model_name = "dummy-model"
    expected_response_str = "{\"expected\": \"response\"}"
    expected_response = {"expected": "response"}
    prompt = "test"

    expected_model_config = model_config or ModelConfig()

    mock_client = client_class_mock.return_value
    mock_client.generate.return_value = _OllamaClientGenerateResponse(expected_response_str)

    cacher = DummyCacher(with_memory_store=True)
    client = OllamaClient(uri="http://uri", model_name=expected_model_name, cacher=cacher)
    response = client.completion_json(prompt=prompt, model_config=model_config)

    mock_client.generate.assert_called_with(model=expected_model_name, system=None, prompt=prompt, options={
        "temperature": expected_model_config.temperature,
        "top_p": expected_model_config.top_p,
        "num_predict": expected_model_config.max_new_tokens,
    }, images=None)

    expected_cache_key_json = cacher.create_cache_key(["llm_completion_json"],
                                                      [None, prompt, str(model_config), "_".join([])])
    expected_cache_key_usage = cacher.create_cache_key(["llm_completion_json__usage"],
                                                     [None, prompt, str(model_config), "_".join([])])


    assert response == expected_response
    assert client._cacher.exists(expected_cache_key_json)
    assert client._cacher.exists(expected_cache_key_usage)
    assert client._cacher.get(expected_cache_key_json) == response

    response = client.completion_json(prompt=prompt, model_config=model_config)

    mock_client.generate.assert_called_once()

    assert response == expected_response

    expected_usage = {
        "request_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "duration": 1.0,
        "duration_avg": 1.0,
        "cost": 0.0000011,
        "cached_request_count": 1,
        "cached_input_tokens": 1,
        "cached_output_tokens": 1,
        "cached_total_tokens": 2,
        "cached_duration": 1.0,
        "cached_duration_avg": 1.0,
        "cached_cost": 0.0000011,
    }

    assert expected_usage == client.usage_summary


@patch("toolchemy.ai.clients.ollama_client.Client")
def test_invalidate_completion_cache(client_class_mock):
    model_config = ModelConfig(max_new_tokens=300, presence_penalty=0.5, temperature=0.6, top_p=0.88)
    expected_model_name = "dummy-model"
    expected_response_str = "{\"expected\": \"response\"}"
    prompt = "test"

    mock_client = client_class_mock.return_value
    mock_client.generate.return_value = _OllamaClientGenerateResponse(expected_response_str)

    cacher = DummyCacher(with_memory_store=True)
    client = OllamaClient(uri="http://uri", model_name=expected_model_name, cacher=cacher)
    _ = client.completion_json(prompt=prompt, model_config=model_config)

    expected_cache_key_json = cacher.create_cache_key(["llm_completion_json"],
                                                      [None, prompt, str(model_config), "_".join([])])
    expected_cache_key_usage = cacher.create_cache_key(["llm_completion_json__usage"],
                                                     [None, prompt, str(model_config), "_".join([])])

    assert client._cacher.exists(expected_cache_key_json)
    assert client._cacher.exists(expected_cache_key_usage)

    client.invalidate_completion_cache(prompt=prompt, model_config=model_config)

    assert not client._cacher.exists(expected_cache_key_json)
    assert not client._cacher.exists(expected_cache_key_usage)
