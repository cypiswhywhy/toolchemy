from toolchemy.ai.clients.common import ILLMClient
from toolchemy.ai.prompting.common import IPrompter


class PromptOptimizer:
    def __init__(self, llm: ILLMClient, prompter: IPrompter):
        self._llm = llm
        self._prompter = prompter

    def refactor_prompt(self, system_prompt: str, user_prompt: str, target_model_name: str) -> tuple[str, str]:
        meta_instruction = self._prompter.render("refactor_prompt_system", model_name=target_model_name)
        user_input = self._prompter.render("refactor_prompt", user_prompt=user_prompt, system_prompt=system_prompt, model_name=target_model_name)

        validation_schema = {"type": "object", "properties": {
            "refactored_system": {"type": "string"},
            "refactored_user": {"type": "string"},
        }}
        result = self._llm.completion_json(prompt=user_input, system_prompt=meta_instruction, validation_schema=validation_schema)

        return result["refactored_system"], result["refactored_user"]
