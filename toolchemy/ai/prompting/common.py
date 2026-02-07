import logging
from abc import ABC, abstractmethod
from toolchemy.utils.cacher import Cacher, ICacher
from toolchemy.utils.logger import get_logger


class IPrompter(ABC):
    @abstractmethod
    def render(self, name: str, version: str | dict[str, str | list[str]] | None = None, **variables) -> str:
        pass

    @abstractmethod
    def template(self, name: str, version: str | dict[str, str | list[str]] | None = None) -> str:
        pass

    @abstractmethod
    def create_template(self, name: str, template: str, overwrite: bool = False):
        pass

    @abstractmethod
    def run_studio(self):
        pass


class PrompterBase(IPrompter, ABC):
    def __init__(self, cacher: ICacher | None = None, no_cache: bool = False, log_level: int = logging.INFO):
        self._logger = get_logger(level=log_level)
        self._cacher = cacher or Cacher(disabled=no_cache)

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


class IPromptOptimizer(ABC):
    @abstractmethod
    def refactor_prompt(self, system_prompt: str, user_prompt: str, target_model_name: str) -> tuple[str, str]:
        pass
