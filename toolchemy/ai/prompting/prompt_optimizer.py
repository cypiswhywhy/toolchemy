from abc import ABC, abstractmethod

from toolchemy.ai.clients.common import ILLMClient
from toolchemy.ai.prompter import IPrompter


class IPromptOptimizer(ABC):
    @abstractmethod
    def refactor_prompt(self, system_prompt: str, user_prompt: str, model_name: str) -> tuple[str, str]:
        pass


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


def testing():
    from toolchemy.ai.clients.factory import create_llm
    from toolchemy.ai.prompter import PrompterMLflow
    llm = create_llm("gpt-oss:120b", uri="http://hal:11434")
    prompter = PrompterMLflow()

    optimizer = PromptOptimizer(llm=llm, prompter=prompter)
    prompt, system_prompt = optimizer.refactor_prompt(
        system_prompt="You're a helpful assistant",
        user_prompt="Generate a short story about a crazy cat. Return it as a valid JSON with one property: 'story'",
        target_model_name="claude-opus-4.5",
    )

    print("---")
    print(prompt.strip())
    print("---")
    print(system_prompt.strip())


if __name__ == "__main__":
    testing()
