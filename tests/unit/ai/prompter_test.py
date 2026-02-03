import pytest


@pytest.mark.parametrize("version,expected_suffix", [
    (None, "Second version."),
    ("1", "First version."),
    ("2", "Second version."),
    ("2", "Second version."),
])
def test_mlflow_render(prompter, version: str | None, expected_suffix: str):
    prompt_name = "test_prompt"
    expected_prompt = f"Yolo! I say cat, you say dog. {expected_suffix}"
    rendered_prompt = prompter.render(name=prompt_name, version=version, foo="cat", bar="dog")

    assert expected_prompt == rendered_prompt


@pytest.mark.parametrize("version", [(None), ("2")])
def test_render_with_cache(prompter, version: str):
    prompt_name = "test_prompt"
    expected_prompt = f"Yolo! I say cat, you say dog. Second version."
    rendered_prompt = prompter.render(name=prompt_name, version=version, foo="cat", bar="dog")

    assert expected_prompt == rendered_prompt
    cache_key = prompter._cacher.create_cache_key(["prompt_render", prompter._build_prompt_uri(name=prompt_name, version=version)], [{"foo": "cat", "bar": "dog"}])
    assert prompter._cacher.exists(cache_key)
    assert expected_prompt == prompter._cacher.get(cache_key)


def test_mlflow_template(prompter):
    prompt_name = "test_prompt"
    expected_prompt_template = "Yolo! I say {{foo}}, you say {{bar}}. Second version."
    prompt_template = prompter.template(name=prompt_name, version="2")

    assert expected_prompt_template == prompt_template


def test_mlflow_template_with_no_version(prompter):
    prompt_name = "test_prompt"
    expected_prompt_template = "Yolo! I say {{foo}}, you say {{bar}}. Second version."
    prompt_template = prompter.template(name=prompt_name)

    assert expected_prompt_template == prompt_template


@pytest.mark.parametrize("version", [(None), ("2")])
def test_mlflow_template_with_cache(prompter, version: str):
    prompt_name = "test_prompt"
    expected_prompt_template = "Yolo! I say {{foo}}, you say {{bar}}. Second version."
    prompt_template = prompter.template(name=prompt_name, version=version)

    assert expected_prompt_template == prompt_template
    cache_key = prompter._cacher.create_cache_key(["prompt_template", prompter._build_prompt_uri(name=prompt_name, version=version)])
    assert prompter._cacher.exists(cache_key)
    assert expected_prompt_template == prompter._cacher.get(cache_key)
