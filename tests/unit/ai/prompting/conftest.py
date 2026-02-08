import logging
import pytest

from toolchemy.ai.prompting.prompter_mlflow import PrompterMLflow
from toolchemy.utils import DummyCacher, Locations


@pytest.fixture
def prompter():
    locations = Locations()
    cacher = DummyCacher(with_memory_store=True)
    prompter = PrompterMLflow(registry_store_dir=locations.in_resources("tests/prompts_mlflow"), cacher=cacher, log_level=logging.DEBUG)


    user_template_1_v1 = "Yolo! I say {{foo}}, you say {{bar}}. First version."
    user_template_1_v2 = "Yolo! I say {{foo}}, you say {{bar}}. Second version."
    user_template_2_v1 = "Yolo! I say {{foo}}, you say {{bar}}. First version."
    user_template_2_v2 = "Yolo! I say {{foo}}, you say {{bar}}. Second version."
    user_template_3_v1 = "Yolo! I say {{foo}}, you say {{bar}}. First version."
    user_template_3_v2 = "Yolo! I say {{foo}}, you say {{bar}}. Second version."
    system_template_2_v1 = "You are an awesome assistant, {{bar}}, {{foobar}}. First version."
    system_template_2_v2 = "You are an awesome assistant, {{bar}}, {{foobar}}. Second version."
    system_template_3_v1 = "You are an awesome assistant, {{bar}}, {{foobar}}. First version."

    prompter.delete("test_prompt")
    prompter.delete("test_prompt_2")
    prompter.delete("test_prompt_3")
    prompter.create_template("test_prompt", user_template_1_v1, overwrite=True)
    prompter.create_template("test_prompt", user_template_1_v2, overwrite=True)
    prompter.create_template("test_prompt_2", user_template_2_v1, system_template_2_v1, overwrite=True)
    prompter.create_template("test_prompt_2", user_template_2_v2, system_template_2_v2, overwrite=True)
    prompter.create_template("test_prompt_3", user_template_3_v1, overwrite=True)
    prompter.create_template("test_prompt_3", user_template_3_v2, system_template_3_v1, overwrite=True)

    return prompter
