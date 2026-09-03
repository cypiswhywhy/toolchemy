import logging
import sys
import sqlite3
import threading
from typing import Optional, Any
from diskcache import Cache, FanoutCache

from toolchemy.utils.cacher.common import (BaseCacher, CacheEntryDoesNotExistError, CacheEntryHasNotBeenSetError,
                                           CacherInitializationError, CacheEntrySeemMalformedError, ICacher)


class DummyLock:
    def acquire(self, blocking: bool = False, timeout: int = -1) -> bool:
        return False

    def release(self):
        pass

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        pass


class CacherDiskcache(BaseCacher):
    def __init__(self, name: str | None = None, cache_base_dir: Optional[str] = None, shards: int = 1, timeout: int = 30,
                 thread_safe: bool = False, disabled: bool = False, log_level: int = logging.INFO):
        super().__init__()
        self._thread_safe = thread_safe
        self._init_common(name=name, cache_base_dir=cache_base_dir, disabled=disabled, log_level=log_level)

        if self._disabled:
            return

        self._lock = DummyLock()
        if self._thread_safe:
            self._lock = threading.RLock()

        try:
            with self._lock:
                if shards > 1:
                    self._cache = FanoutCache(self._cache_dir, shards=shards, timeout=timeout)
                else:
                    self._cache = Cache(self._cache_dir, cull_limit=0, size_limit=2**38, timeout=timeout)
        except Exception as e:
            raise CacherInitializationError(f"Failed to initialize disk cache for name '{self._name}' (cache dir: '{self._cache_dir}')") from e

        self._log_initialized()

    @property
    def cache_location(self) -> str:
        return self._cache_dir

    def sub_cacher(self, log_level: int | None = None, suffix: str | None = None) -> "ICacher":
        name, log_level = self._sub_cacher_params(log_level, suffix)
        return CacherDiskcache(name=name,
                               cache_base_dir=self._cache_base_dir,
                               log_level=log_level, disabled=self._disabled)

    def _exists(self, name: str) -> bool:
        if self._disabled:
            self._logger.debug("Cacher disabled")
            return False

        try:
            with self._lock:
                if name in self._cache:
                    return True
        except sqlite3.OperationalError as e:
            raise CacheEntrySeemMalformedError(f"Checking the existence of '{name}' failed with: {e!s}")
        self._logger.debug("Cache entry %s::%s does not exist", self._cache_dir, name)
        return False

    def set(self, name: str, content: Any, ttl_s: int | None = None):
        """
        Dumps a given object under a given cache entry name. The object must be pickleable.
        """
        if self._disabled:
            return

        with self._lock:
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
            raise CacheEntryDoesNotExistError("Caching is disabled...")

        self._logger.debug("Cache get: %s::%s", self._cache_dir, name)

        with self._lock:
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

        with self._lock:
            if name in self._cache:
                del self._cache[name]
                self._logger.debug("Cache entry %s::%s removed", self._name, name)
            else:
                self._logger.warning("Cache entry %s::%s does not exist, nothing to remove", self._name, name)

    def persist(self):
        with self._lock:
            self._cache.close()
