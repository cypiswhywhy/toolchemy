import logging
import os
import sys
from typing import Optional, Any
from diskcache import Cache

from toolchemy.utils.cacher.common import BaseCacher, CacheEntryDoesNotExistError, CacheEntryHasNotBeenSetError, CacherInitializationError
from toolchemy.utils.logger import get_logger
from toolchemy.utils.locations import get_external_caller_path
from toolchemy.utils.utils import _caller_module_name


class CacherDiskcache(BaseCacher):
    def __init__(self, name: str | None = None, cache_base_dir: Optional[str] = None, disabled: bool = False,
                 log_level: int = logging.INFO):
        super().__init__()
        self._disabled = disabled

        self._log_level = log_level
        self._logger = get_logger(level=self._log_level)

        self._name = name
        if not self._name:
            self._name = _caller_module_name()

        self._cache_base_dir = cache_base_dir
        if self._cache_base_dir is None:
            self._cache_base_dir = get_external_caller_path()

        self._cache_dir = os.path.join(self._cache_base_dir, self.CACHER_MAIN_NAME, self._name)

        if self._disabled:
            return

        try:
            self._cache = Cache(self._cache_dir, cull_limit=0, size_limit=2**38)
        except Exception as e:
            raise CacherInitializationError(f"Failed to initialize disk cache for name '{self._name}' (cache dir: '{self._cache_dir}')") from e

        self._logger.debug(
            f"Cacher '{self._name}' initialized (cache path: '{self._cache_dir}', log_level: '{logging.getLevelName(log_level)}')")
        self._logger.debug(f"Cacher logging DEBUG level enabled")

    @property
    def cache_location(self) -> str:
        return self._cache_dir

    def sub_cacher(self, log_level: int | None = None, suffix: str | None = None) -> "ICacher":
        name = _caller_module_name()
        if suffix:
            name += f"__{suffix}"
        if log_level is None:
            log_level = self._log_level
        self._logger.debug(f"Creating sub cacher")
        self._logger.debug(f"> base cache dir: {self._cache_dir}")
        self._logger.debug(f"> name: {name}")
        self._logger.debug(f"> log level: {log_level} ({logging.getLevelName(log_level)})")
        self._logger.debug(f"> is disabled: {self._disabled})")

        return CacherDiskcache(name=os.path.join(self._name, name).strip("/"),
                               cache_base_dir=self._cache_base_dir,
                               log_level=log_level, disabled=self._disabled)

    def _exists(self, name: str) -> bool:
        if self._disabled:
            self._logger.debug("Cacher disabled")
            return False

        if name in self._cache:
            return True
        self._logger.debug("Cache entry %s::%s does not exist", self._cache_dir, name)
        return False

    def set(self, name: str, content: Any, ttl_s: int | None = None):
        """
        Dumps a given object under a given cache entry name. The object must be pickleable.
        """
        if self._disabled:
            return

        result = self._cache.set(name, content, expire=ttl_s)
        does_exist = self.exists(name)
        if not result or not does_exist:
            self._logger.error(f"Cache entry '{name}' not set for name '{self._name}' ({result}, {does_exist})")
            self._logger.error(f"> cache dir: {self._cache_dir}")
            self._logger.error(f"> type of the content: {type(content)}")
            self._logger.error(f"> size of the content: {sys.getsizeof(content)}")
            raise CacheEntryHasNotBeenSetError()

        self._logger.debug("Cache set %s::%s", self._cache_dir, name)

    def get(self, name: str) -> Any:
        """
        Loads an object for a given cache entry name. If it doesn't exist an exception is thrown.
        """
        if self._disabled:
            raise CacheEntryDoesNotExistError(f"Caching is disabled...")

        self._logger.debug("Cache get: %s::%s", self._cache_dir, name)

        if name in self._cache:
            return self._cache.get(name)

        raise CacheEntryDoesNotExistError(f"Cache does not exist: {self._cache_dir}::{name}.")

    def unset(self, name: str):
        """
        Removes a cache entry for a given name
        """
        self._logger.debug("Cache unset: %s::%s", self._cache_dir, name)
        if self._disabled:
            return

        if name in self._cache:
            del self._cache[name]
            self._logger.debug("Cache entry %s::%s removed", self._name, name)
        else:
            self._logger.warning("Cache entry %s::%s does not exist, nothing to remove", self._name, name)

    def persist(self):
        self._cache.close()
