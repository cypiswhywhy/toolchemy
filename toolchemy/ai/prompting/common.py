import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jinja2 import Template
from toolchemy.utils.cacher import Cacher, ICacher
from toolchemy.utils.logger import get_logger


class InvalidPromptError(Exception):
    pass


@dataclass
class Prompt:
    system: str | None = None
    user: str | None = None
    template_system: str | None = None
    template_user: str | None = None

    def format(self, render_user: bool = True, render_system: bool = True, **variables) -> "Prompt":
        if render_user and not self.template_user:
            raise InvalidPromptError("missing user template")
        if render_system and  not self.template_system:
            raise InvalidPromptError("missing system template")

        return Prompt(system=self._format_template(self.template_system, **variables),
                      user=self._format_template(self.template_user, **variables),
                      template_system=self.template_system, template_user=self.template_user)

    def format_user(self, **variables) -> "Prompt":
        if not self.template_user:
            raise InvalidPromptError("missing user template")

        return Prompt(system=self.system, user=self._format_template(self.template_user, **variables),
                      template_system=self.template_system, template_user=self.template_user)

    def format_system(self, **variables) -> "Prompt":
        if not self.template_system:
            raise InvalidPromptError("missing system template")
        return Prompt(system=self._format_template(self.template_system, **variables), user=self.user, template_system=self.template_system,
                      template_user=self.template_user)

    def _format_template(self, template: str | None, **variables) -> str | None:
        if template is None:
            return None
        t = Template(template)
        return t.render(**variables)

    def json(self) -> dict:
        return {
            "system": self.system,
            "user": self.user,
            "template_system": self.template_system,
            "template_user": self.template_user,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Prompt":
        return cls(system=data.get("system", None), user=data.get("user", None), template_system=data.get("template_system", None),
                   template_user=data.get("template_user", None))


class IPrompter(ABC):
    @abstractmethod
    def render(self, name: str, version: str | None = None, version_system: str | None = None, optimize_rendered: bool = False, **variables) -> Prompt:
        pass

    @abstractmethod
    def create_template(self, name: str, template_user: str, template_system: str | None = None, overwrite: bool = False):
        pass

    @abstractmethod
    def delete(self, name: str):
        pass

    @abstractmethod
    def run_studio(self):
        pass


class IPromptOptimizer(ABC):
    @abstractmethod
    def refactor(self, prompt: Prompt, templates_only: bool = False) -> Prompt:
        pass


class PrompterBase(IPrompter, ABC):
    DEFAULT_PROMPT_SYSTEM = "You are a helpful assistant"

    def __init__(self, default_system_prompt: str | None = None, prompt_optimizer: IPromptOptimizer | None = None, cacher: ICacher | None = None,
                 no_cache: bool = False, log_level: int = logging.INFO):
        self._logger = get_logger(level=log_level)
        self._cacher = cacher or Cacher(disabled=no_cache)
        self._prompt_optimizer = prompt_optimizer
        self._default_system_prompt = default_system_prompt or self.DEFAULT_PROMPT_SYSTEM

    def _prompt_version(self, name: str, version_mapping: str | dict[str, str | list[str]] | None = None) -> str | None:
        if version_mapping is None:
            return self._latest_value()
        if isinstance(version_mapping, int):
            return str(version_mapping)
        if isinstance(version_mapping, str):
            return version_mapping
        if name not in version_mapping:
            return None
        if isinstance(version_mapping[name], str):
            return version_mapping[name]
        return version_mapping[name][0]

    @abstractmethod
    def _latest_value(self) -> str |  None:
        pass
