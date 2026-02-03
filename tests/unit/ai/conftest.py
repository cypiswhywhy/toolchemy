import pytest

from toolchemy.ai.prompter import PrompterMLflow
from toolchemy.utils import DummyCacher, Locations


@pytest.fixture
def prompter():
    locations = Locations()
    cacher = DummyCacher(with_memory_store=True)
    prompter = PrompterMLflow(registry_store_dir=locations.in_resources("tests/prompts_mlflow"), cacher=cacher)

    prompt_template_1 = "Yolo! I say {{foo}}, you say {{bar}}. First version."
    prompt_template_2 = "Yolo! I say {{foo}}, you say {{bar}}. Second version."

    prompter.create_template("test_prompt", prompt_template_1, overwrite=True)
    prompter.create_template("test_prompt", prompt_template_2, overwrite=True)

    return prompter
