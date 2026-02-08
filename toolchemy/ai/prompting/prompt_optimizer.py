import logging
from toolchemy.ai.clients.common import ILLMClient
from toolchemy.ai.prompting.common import IPromptOptimizer, Prompt, InvalidPromptError
from toolchemy.utils.logger import get_logger
from toolchemy.utils.cacher import Cacher


class PromptOptimizer(IPromptOptimizer):
    def __init__(self, llm: ILLMClient, target_model_name: str | None = None, log_level: int = logging.INFO):
        self._logger = get_logger(level=log_level)
        self._cacher = Cacher(log_level=log_level)
        self._llm = llm
        self._target_model_name = target_model_name or llm.name()
        self._logger.info(f"Prompt Optimizer initialized (llm: {self._llm.name()}, target model: {target_model_name})")

    def refactor(self, prompt: Prompt) -> Prompt:
        if not prompt.template_user or not prompt.template_system:
            raise InvalidPromptError("Templates for user and system must be present")

        self._logger.debug(f"Prompt optimizer started")
        self._logger.debug(f"> System template BEFORE:\n{prompt.template_system}")
        self._logger.debug(f"> User template BEFORE:\n{prompt.template_user}")

        cache_key = self._cacher.create_cache_key("refactor", prompt.json())
        if self._cacher.exists(cache_key):
            prompt_json = self._cacher.get(cache_key)
            prompt = Prompt.from_json(prompt_json)
            self._logger.debug(f"> System template AFTER (cached):\n{prompt.template_system}")
            self._logger.debug(f"> User template AFTER (cached):\n{prompt.template_user}")
            return prompt

        if prompt.template_user and prompt.template_system:
            prompt.template_system, prompt.template_user = self._refactor(user_prompt=prompt.template_user, system_prompt=prompt.template_system,
                                                                          target_model_name=self._target_model_name)
            self._logger.debug(f"> System template AFTER:\n{prompt.template_system}")
            self._logger.debug(f"> User template AFTER:\n{prompt.template_user}")

        if prompt.user and prompt.system:
            prompt.system, prompt.user = self._refactor(user_prompt=prompt.user, system_prompt=prompt.system,
                                                        target_model_name=self._target_model_name)
            self._logger.debug(f"> System AFTER:\n{prompt.system}")
            self._logger.debug(f"> User AFTER:\n{prompt.user}")

        self._cacher.set(cache_key, prompt.json())

        return prompt

    def _refactor(self, user_prompt: str, system_prompt: str, target_model_name: str) -> tuple[str, str]:
        meta_instruction = self._render_prompt_system(target_model_name=target_model_name)
        user_input = self._render_prompt_user(user_prompt=user_prompt, system_prompt=system_prompt, target_model_name=target_model_name)

        validation_schema = {"type": "object", "properties": {
            "refactored_system": {"type": "string"},
            "refactored_user": {"type": "string"},
        }}
        result = self._llm.completion_json(prompt=user_input, system_prompt=meta_instruction, validation_schema=validation_schema)

        try:
            system_prompt_refactored = result["refactored_system"]
            prompt_refactored = result["refactored_user"]
        except Exception:
            self._logger.error(f"Invalid response format. The response:\n''{result}")
            raise

        return system_prompt_refactored, prompt_refactored

    def _render_prompt_system(self, target_model_name: str) -> str:
        return f"""You are an expert Prompt Engineer. Your task is to refactor a System and User prompt to be perfectly optimized for the target model: "{target_model_name}".

STRATEGY:
1. Information Architecture: Use structural delimiters (XML tags for Claude, Markdown for GPT/Llama).
2. Role Clarity: Strengthen the persona in the system prompt.
3. Variables: Ensure the user prompt is a clean template for the task.
4. Model Quirks: If the target is Claude, use XML. If GPT, use clear headers and triple quotes.

OUTPUT FORMAT:
Return only the refactored prompts with no additional comments, headlines, etc. You must return a valid JSON object with exactly two keys:
"refactored_system": "...",
"refactored_user": "..."
"""

    def _render_prompt_user(self, system_prompt: str, user_prompt: str, target_model_name: str) -> str:
        return f"""--- ORIGINAL SYSTEM PROMPT ---
{system_prompt}

--- ORIGINAL USER PROMPT ---
{user_prompt}

--- TARGET MODEL ---
{target_model_name}

Please provide the optimized versions in the requested JSON format.
"""
