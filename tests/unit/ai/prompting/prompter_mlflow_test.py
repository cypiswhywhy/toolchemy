import pytest

from toolchemy.ai.prompting.common import Prompt, PrompterBase


@pytest.mark.parametrize("version,version_system", [
    ("1", None),
    ("1", "1"),
    ("2", None),
    ("2", "1"),
    (None, None),
    (None, "1"),
])
def test_mlflow_template(prompter, version: str | None, version_system: str | None):
    prompt_name = "test_prompt"
    expected_suffix = "First version." if version == "1" else "Second version."
    expected_prompt = Prompt(
        system=None,
        user=None,
        template_system=PrompterBase.DEFAULT_PROMPT_SYSTEM,
        template_user=f"Yolo! I say {{{{foo}}}}, you say {{{{bar}}}}. {expected_suffix}",
    )
    template = prompter.template(name=prompt_name, version=version, version_system=version_system)

    assert expected_prompt == template


@pytest.mark.parametrize("version,version_system", [
    ("1", None),
    ("1", "1"),
    ("2", None),
    ("2", "1"),
    (None, None),
    (None, "1"),
])
def test_mlflow_render(prompter, version: str | None, version_system: str | None):
    prompt_name = "test_prompt"
    expected_suffix = "First version." if version == "1" else "Second version."
    expected_prompt = Prompt(
        system=PrompterBase.DEFAULT_PROMPT_SYSTEM,
        user=f"Yolo! I say cat, you say dog. {expected_suffix}",
        template_system=PrompterBase.DEFAULT_PROMPT_SYSTEM,
        template_user=f"Yolo! I say {{{{foo}}}}, you say {{{{bar}}}}. {expected_suffix}",
    )
    rendered_prompt = prompter.render(name=prompt_name, version=version, version_system=version_system, foo="cat", bar="dog")

    assert expected_prompt == rendered_prompt


@pytest.mark.parametrize("version,version_system", [
    ("1", None),
    ("1", "1"),
    ("1", "2"),
    ("2", None),
    ("2", "1"),
    ("2", "2"),
    (None, None),
    (None, "1"),
    (None, "2"),
])
def test_mlflow_render_v2(prompter, version: str | None, version_system: str | None):
    prompt_name = "test_prompt_2"
    expected_suffix = "First version." if version == "1" else "Second version."
    expected_suffix_system = "First version." if version_system == "1" else "Second version."
    expected_prompt = Prompt(
        system=f"You are an awesome assistant, dog, mice. {expected_suffix_system}",
        user=f"Yolo! I say cat, you say dog. {expected_suffix}",
        template_system=f"You are an awesome assistant, {{{{bar}}}}, {{{{foobar}}}}. {expected_suffix_system}",
        template_user=f"Yolo! I say {{{{foo}}}}, you say {{{{bar}}}}. {expected_suffix}",
    )
    rendered_prompt = prompter.render(name=prompt_name, version=version, version_system=version_system, foo="cat", bar="dog", foobar="mice")

    assert expected_prompt == rendered_prompt


@pytest.mark.parametrize("version_system", [
    None,
    "1",
])
def test_mlflow_render_unsynced_versions(prompter, version_system: str | None):
    prompt_name = "test_prompt_3"
    expected_suffix = "Second version."
    expected_suffix_system = "First version."
    expected_prompt = Prompt(
        system=f"You are an awesome assistant, dog, mice. {expected_suffix_system}",
        user=f"Yolo! I say cat, you say dog. {expected_suffix}",
        template_system=f"You are an awesome assistant, {{{{bar}}}}, {{{{foobar}}}}. {expected_suffix_system}",
        template_user=f"Yolo! I say {{{{foo}}}}, you say {{{{bar}}}}. {expected_suffix}",
    )
    rendered_prompt = prompter.render(name=prompt_name, version_system=version_system, foo="cat", bar="dog", foobar="mice")

    assert expected_prompt == rendered_prompt


@pytest.mark.parametrize("version", [None, "2"])
def test_render_with_cache(prompter, version: str):
    prompt_name = "test_prompt"
    expected_prompt = Prompt(
        system=PrompterBase.DEFAULT_PROMPT_SYSTEM,
        user=f"Yolo! I say cat, you say dog. Second version.",
        template_system=PrompterBase.DEFAULT_PROMPT_SYSTEM,
        template_user=f"Yolo! I say {{{{foo}}}}, you say {{{{bar}}}}. Second version.",
    )
    rendered_prompt = prompter.render(name=prompt_name, version=version, foo="cat", bar="dog")

    assert expected_prompt == rendered_prompt

    cache_key = prompter._cacher.create_cache_key(
        ["render", prompt_name, version, None, "optimized_False", False],
        [{"foo": "cat", "bar": "dog"}])

    assert prompter._cacher.exists(cache_key)
    rendered_prompt = Prompt.from_json(prompter._cacher.get(cache_key))

    assert expected_prompt == rendered_prompt
