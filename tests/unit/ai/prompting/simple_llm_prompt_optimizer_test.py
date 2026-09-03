import json
from unittest.mock import patch

import pytest

from toolchemy.ai.clients import DummyModelClient
from toolchemy.ai.prompting.common import InvalidPromptError, Prompt
from toolchemy.ai.prompting.simple_llm_prompt_optimizer import SimpleLLMPromptOptimizer
from toolchemy.utils.cacher import DummyCacher

REFACTORED = {"refactored_system": "SYSTEM!", "refactored_user": "USER!"}


def _memory_cacher(**_kwargs) -> DummyCacher:
    """Stand-in for Cacher. DummyCacher does not accept Cacher's log_level kwarg."""
    return DummyCacher(with_memory_store=True)


@pytest.fixture
def optimizer():
    # SimpleLLMPromptOptimizer builds a real on-disk Cacher in __init__ with no way to
    # inject one, so it has to be patched out to keep this test hermetic.
    llm = DummyModelClient(fixed_response=json.dumps(REFACTORED))
    with patch("toolchemy.ai.prompting.simple_llm_prompt_optimizer.Cacher", _memory_cacher):
        yield SimpleLLMPromptOptimizer(llm=llm, target_model_name="target-model")


def test_refactor_rejects_a_prompt_without_templates(optimizer):
    with pytest.raises(InvalidPromptError, match="Templates for user and system must be present"):
        optimizer.refactor(Prompt(system="s", user="u"))


def test_refactor_rewrites_both_templates(optimizer):
    prompt = Prompt(template_system="old system", template_user="old user")

    result = optimizer.refactor(prompt)

    assert result.template_system == "SYSTEM!"
    assert result.template_user == "USER!"


def test_refactor_also_rewrites_the_rendered_prompts_by_default(optimizer):
    prompt = Prompt(system="old system", user="old user", template_system="old system", template_user="old user")

    result = optimizer.refactor(prompt)

    assert result.system == "SYSTEM!"
    assert result.user == "USER!"


def test_refactor_leaves_the_rendered_prompts_alone_when_templates_only(optimizer):
    prompt = Prompt(system="old system", user="old user", template_system="old system", template_user="old user")

    result = optimizer.refactor(prompt, templates_only=True)

    assert result.template_system == "SYSTEM!"
    assert result.system == "old system"
    assert result.user == "old user"


def test_refactor_raises_when_the_model_omits_the_expected_keys():
    llm = DummyModelClient(fixed_response=json.dumps({"something_else": "?"}))
    with patch("toolchemy.ai.prompting.simple_llm_prompt_optimizer.Cacher", _memory_cacher):
        optimizer = SimpleLLMPromptOptimizer(llm=llm, target_model_name="target-model")

    with pytest.raises(KeyError):
        optimizer.refactor(Prompt(template_system="s", template_user="u"))


def test_target_model_name_falls_back_to_the_client_name():
    llm = DummyModelClient(fixed_response=json.dumps(REFACTORED))
    with patch("toolchemy.ai.prompting.simple_llm_prompt_optimizer.Cacher", _memory_cacher):
        optimizer = SimpleLLMPromptOptimizer(llm=llm)

    assert optimizer._target_model_name == llm.name()


def test_refactor_serves_a_repeated_prompt_from_the_cache(optimizer):
    prompt = Prompt(template_system="old system", template_user="old user")

    first = optimizer.refactor(prompt)
    requests_after_first = optimizer._llm.usage_summary["request_count"]

    second = optimizer.refactor(Prompt(template_system="old system", template_user="old user"))

    assert second.template_system == first.template_system
    assert second.template_user == first.template_user
    assert optimizer._llm.usage_summary["request_count"] == requests_after_first
