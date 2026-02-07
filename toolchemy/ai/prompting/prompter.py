import logging
import mlflow
import sys
import subprocess
from abc import ABC, abstractmethod

from toolchemy.utils.cacher import Cacher, ICacher
from toolchemy.utils.logger import get_logger
from toolchemy.utils.locations import Locations


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


def run_studio(registry_path: str | None = None):
    prompter = PrompterMLflow(registry_store_dir=registry_path)
    prompter.run_studio()


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


class PrompterMLflow(PrompterBase):
    DEFAULT_PROMPT_REGISTRY_NAME = "prompts_mlflow"

    def __init__(self, registry_store_dir: str | None = None, cacher: ICacher | None = None, no_cache: bool = False, log_level: int = logging.INFO):
        super().__init__(cacher=cacher, no_cache=no_cache, log_level=log_level)
        locations = Locations()
        if registry_store_dir is None:
            registry_store_dir = locations.in_root(self.DEFAULT_PROMPT_REGISTRY_NAME)
        registry_store_dir = locations.abs(registry_store_dir).rstrip("/")

        self._registry_store_uri = f"sqlite:///{registry_store_dir}/registry.db"
        self._tracking_uri = f"sqlite:///{registry_store_dir}/tracking.db"

        self._client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri, registry_uri=self._registry_store_uri)

        self._logger.info(f"Prompter-MLflow initialized")
        self._logger.info(f"> tracking store uri: {self._tracking_uri}")
        self._logger.info(f"> registry store uri: {self._registry_store_uri}")

    def render(self, name: str, version: str | dict[str, str | list[str]] | None = None, **variables) -> str:
        prompt_uri = self._build_prompt_uri(name=name, version=version)

        cache_key = self._cacher.create_cache_key(["prompt_render", prompt_uri], [variables])
        if self._cacher.exists(cache_key):
            self._logger.debug(f"Retrieving from the cache")
            return self._cacher.get(cache_key)

        self._logger.debug(f"Rendering prompt: '{name}' (version: '{version}') -> prompt uri: '{prompt_uri}'")

        prompt_template = self._client.load_prompt(prompt_uri)
        prompt = prompt_template.format(**variables)

        self._cacher.set(cache_key, prompt)

        return prompt

    def template(self, name: str, version: str | dict[str, str | list[str]] | None = None) -> str:
        prompt_uri = self._build_prompt_uri(name=name, version=version)

        cache_key = self._cacher.create_cache_key(["prompt_template", prompt_uri])
        if self._cacher.exists(cache_key):
            return self._cacher.get(cache_key)

        self._logger.debug(f"Getting prompt template: '{name}' (version: '{version}') -> prompt uri: '{prompt_uri}'")

        prompt_template = self._client.load_prompt(prompt_uri)

        self._cacher.set(cache_key, prompt_template.template)

        return prompt_template.template

    def create_template(self, name: str, template: str, overwrite: bool = False):
        if self._client.load_prompt(name_or_uri=name, allow_missing=True) and not overwrite:
            return
        self._client.register_prompt(name=name, template=template)

    def _build_prompt_uri(self, name: str, version: str | int | None = None) -> str:
        prompt_version = self._prompt_version(name, version_mapping=version)
        prompt_uri = f"prompts:/{name}"
        prompt_uri += f"@{prompt_version}" if isinstance(prompt_version, str) and not prompt_version.isdigit() else f"/{prompt_version}"
        return prompt_uri

    def run_studio(self):
        command = [sys.executable, "-m", "mlflow", "ui", "--registry-store-uri", self._registry_store_uri, "--backend-store-uri", self._tracking_uri]
        sys.exit(subprocess.call(command))

    def _latest_value(self) -> str | None:
        return "latest"


def testing():
    import logging
    locations = Locations()
    prompter = PrompterMLflow(registry_store_dir=locations.in_resources("tests/prompts_mlflow"), log_level=logging.DEBUG)
    p = prompter.render("create_model_task_prompt")
    print(p)


if __name__ == "__main__":
    testing()
