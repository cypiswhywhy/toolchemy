import logging
from toolchemy.ai.clients.common import ILLMClient
from toolchemy.ai.prompting.common import IPrompter, IPromptOptimizer
from toolchemy.utils.logger import get_logger


class PromptOptimizer(IPromptOptimizer):
    def __init__(self, llm: ILLMClient, prompter: IPrompter, log_level: int = logging.INFO):
        self._logger = get_logger(level=log_level)
        self._llm = llm
        self._prompter = prompter
        self._logger.info(f"Prompt Optimizer initialized (llm: {self._llm.name()})")

    def refactor_prompt(self, system_prompt: str, user_prompt: str, target_model_name: str) -> tuple[str, str]:
        meta_instruction = self._prompter.render("refactor_prompt_system", model_name=target_model_name)
        user_input = self._prompter.render("refactor_prompt", user_prompt=user_prompt, system_prompt=system_prompt, model_name=target_model_name)

        validation_schema = {"type": "object", "properties": {
            "refactored_system": {"type": "string"},
            "refactored_user": {"type": "string"},
        }}
        result = self._llm.completion_json(prompt=user_input, system_prompt=meta_instruction, validation_schema=validation_schema)

        self._logger.debug(f"=========================== BEFORE ===")
        self._logger.debug(f"--------------------------- system:\n{result["refactored_system"]}")
        self._logger.debug(f"--------------------------- user:\n{result["refactored_user"]}\n")

        return result["refactored_system"], result["refactored_user"]
