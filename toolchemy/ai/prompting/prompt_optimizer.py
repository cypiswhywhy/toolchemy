import logging
from toolchemy.ai.clients.common import ILLMClient
from toolchemy.ai.prompting.common import IPrompter, IPromptOptimizer
from toolchemy.utils.logger import get_logger
from toolchemy.utils.cacher import Cacher


class PromptOptimizer(IPromptOptimizer):
    def __init__(self, llm: ILLMClient, prompter: IPrompter, log_level: int = logging.INFO):
        self._logger = get_logger(level=log_level)
        self._cacher = Cacher(log_level=log_level)
        self._llm = llm
        self._prompter = prompter
        self._logger.info(f"Prompt Optimizer initialized (llm: {self._llm.name()})")

    def refactor_prompt(self, system_prompt: str, user_prompt: str, target_model_name: str) -> tuple[str, str]:
        meta_instruction = self._prompter.render("refactor_prompt_system", model_name=target_model_name)
        user_input = self._prompter.render("refactor_prompt", user_prompt=user_prompt, system_prompt=system_prompt, model_name=target_model_name)

        cache_key = self._cacher.create_cache_key("prompt_refactor", [meta_instruction, user_input, target_model_name])
        if self._cacher.exists(cache_key):
            system_prompt, user_prompt = self._cacher.get(cache_key)
            return system_prompt, user_prompt

        validation_schema = {"type": "object", "properties": {
            "refactored_system": {"type": "string"},
            "refactored_user": {"type": "string"},
        }}
        result = self._llm.completion_json(prompt=user_input, system_prompt=meta_instruction, validation_schema=validation_schema)

        system_prompt_refactored = result["refactored_system"]
        prompt_refactored = result["refactored_user"]

        self._cacher.set(cache_key, (system_prompt_refactored, prompt_refactored))

        self._logger.debug(f"=========================== BEFORE ===")
        self._logger.debug(f"--------------------------- system:\n{system_prompt_refactored}")
        self._logger.debug(f"--------------------------- user:\n{prompt_refactored}\n")

        return system_prompt_refactored, prompt_refactored
