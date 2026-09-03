import pytest

from toolchemy.ai.clients import DummyModelClient, ModelConfig, create_llm
from toolchemy.ai.clients.common import LLMCacheDoesNotExist


def test_dummy_client_can_be_instantiated_and_echoes_the_prompt():
    client = DummyModelClient()

    assert client.name() == "dummy-model"
    assert client.completion("hello") == "Echo: hello"


def test_dummy_client_returns_the_fixed_response_when_given_one():
    client = DummyModelClient(fixed_response='{"answer": 42}')

    assert client.completion("anything") == '{"answer": 42}'
    assert client.completion_json("anything") == {"answer": 42}


def test_dummy_client_records_usage():
    client = DummyModelClient()
    client.completion("one")
    client.completion("two")

    summary = client.usage_summary

    assert summary["request_count"] == 2
    assert len(client.usage()) == 2


def test_dummy_client_supports_the_llm_client_interface():
    client = DummyModelClient()

    assert client.embeddings_size == 32
    assert len(client.embeddings("text")) == 32
    assert client.model_config(ModelConfig(model_name="other")).model_name == "other"
    client.invalidate_completion_cache("hello")


def test_dummy_client_honours_cache_only():
    client = DummyModelClient()

    with pytest.raises(LLMCacheDoesNotExist):
        client.completion("hello", cache_only=True)


def test_create_llm_dispatches_the_dummy_uri_to_the_dummy_client():
    client = create_llm(name="dummy-model", uri="dummy")

    assert isinstance(client, DummyModelClient)
    assert client.completion("hello") == "Echo: hello"
