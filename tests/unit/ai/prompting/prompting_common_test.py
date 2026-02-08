from toolchemy.ai.prompting.common import Prompt


def test_render_user():
    prompt = Prompt(template_user="This is a template for {{placeholder}} with some double curly {}")
    rendered = prompt.format_user(placeholder="an awesome guy").user
    expected_rendered = "This is a template for an awesome guy with some double curly {}"

    assert rendered == expected_rendered
