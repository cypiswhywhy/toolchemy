import logging
import mlflow
import sys
import subprocess

from toolchemy.ai.prompting.common import PrompterBase, IPromptOptimizer, Prompt
from toolchemy.utils.cacher import ICacher
from toolchemy.utils.locations import Locations


def run_studio(registry_path: str | None = None):
    prompter = PrompterMLflow(registry_store_dir=registry_path)
    prompter.run_studio()


class PrompterMLflow(PrompterBase):
    DEFAULT_PROMPT_REGISTRY_NAME = "prompts_mlflow"

    def __init__(self, default_system_prompt: str | None = None, registry_store_dir: str | None = None, prompt_optimizer: IPromptOptimizer | None = None,
                 cacher: ICacher | None = None, no_cache: bool = False, log_level: int = logging.INFO):
        super().__init__(default_system_prompt=default_system_prompt, prompt_optimizer=prompt_optimizer, cacher=cacher, no_cache=no_cache, log_level=log_level)
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

    def render(self, name: str, version: str | None = None, version_system: str | None = None, optimize_formatted: bool = False, **variables) -> Prompt:
        cache_key = self._cacher.create_cache_key(
            ["render", name, version, version_system, f"optimized_{self._prompt_optimizer is not None}", optimize_formatted], [variables])

        if self._cacher.exists(cache_key):
            self._logger.debug(f"Retrieving from the cache")
            prompt_json = self._cacher.get(cache_key)
            return Prompt.from_json(prompt_json)

        prompt = self.template(name=name, version=version, version_system=version_system)
        prompt = prompt.format(**variables)
        if optimize_formatted and self._prompt_optimizer:
            prompt = self._prompt_optimizer.refactor(prompt)

        self._cacher.set(cache_key, prompt.json())

        return prompt

    def template(self, name: str, version: str | None = None, version_system: str | None = None, optimize_rendered: bool = False) -> Prompt:
        name_system = f"{name}_system"

        prompt_uri_user = self._build_prompt_uri(name=name, version=version)
        prompt_uri_system = self._build_prompt_uri(name=name_system, version=version_system)

        self._logger.debug(f"Rendering prompt:")
        self._logger.debug(f"> user: '{name}' (version: '{version}') -> uri: '{prompt_uri_user}'")
        self._logger.debug(f"> user: '{name_system}' (version: '{version_system}') -> uri: '{prompt_uri_system}'")

        cache_key = self._cacher.create_cache_key(["template", prompt_uri_user, prompt_uri_system, f"optimized_{self._prompt_optimizer is not None}"])

        if self._cacher.exists(cache_key):
            self._logger.debug(f"Retrieving from the cache")
            prompt_json = self._cacher.get(cache_key)
            return Prompt.from_json(prompt_json)

        prompt_template_user = self._client.load_prompt(prompt_uri_user).template
        prompt_template_system = self._client.load_prompt(prompt_uri_system, allow_missing=True)
        prompt_template_system = prompt_template_system.template if prompt_template_system else None
        if not prompt_template_system or prompt_template_system == "None":
            prompt_template_system = self.DEFAULT_PROMPT_SYSTEM

        prompt = Prompt(template_user=prompt_template_user, template_system=prompt_template_system)
        if self._prompt_optimizer:
            prompt = self._prompt_optimizer.refactor(prompt)

        self._cacher.set(cache_key, prompt.json())

        return prompt

    def create_template(self, name: str, template_user: str, template_system: str | None = None, overwrite: bool = False):
        user_ver = None
        system_ver = None
        if not self._client.load_prompt(name_or_uri=name, allow_missing=True) or overwrite:
            user_ver = self._client.register_prompt(name=name, template=template_user)
            user_ver = user_ver.version if user_ver else None

        name_system = f"{name}_system"
        if template_system and (not self._client.load_prompt(name_or_uri=name_system, allow_missing=True) or overwrite):
            system_ver = self._client.register_prompt(name=name_system, template=template_system)
            system_ver = system_ver.version if system_ver else None

        self._logger.debug(f"Registered prompt, user: '{name}' (ver: {user_ver}), system: '{name_system}' (ver: {system_ver})")

    def delete(self, name: str):
        name_system = f"{name}_system"
        if self._client.load_prompt(name, allow_missing=True):
            self._client.delete_prompt(name)
        if self._client.load_prompt(name_system, allow_missing=True):
            self._client.delete_prompt(name_system)

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
