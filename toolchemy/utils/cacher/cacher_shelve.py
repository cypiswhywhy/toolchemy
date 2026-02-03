import logging
import os
import threading
from typing import Optional, Any
import shelve

from toolchemy.utils.cacher.common import BaseCacher, DummyLock, CacheEntryDoesNotExistError
from toolchemy.utils.logger import get_logger
from toolchemy.utils.locations import get_external_caller_path
from toolchemy.utils.utils import _caller_module_name
from toolchemy.utils.datestimes import current_unix_timestamp


class CacherShelve(BaseCacher):
    def __init__(self, name: str | None = None, cache_base_dir: Optional[str] = None, disabled: bool = False,
                 log_level: int = logging.INFO, enable_thread_safeness: bool = False):
        super().__init__()
        self._disabled = disabled
        self._enable_thread_safeness = enable_thread_safeness
        if enable_thread_safeness:
            self._lock = threading.Lock()
        else:
            self._lock = DummyLock()
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

        self._cache_path = os.path.join(self._cache_dir, "cache")

        with self._lock:
            os.makedirs(self._cache_dir, exist_ok=True)
        if not self._enable_thread_safeness:
            self._open()

        self._logger.debug(f"Cacher '{self._name}' initialized (cache path: '{self._cache_dir}', log_level: '{logging.getLevelName(log_level)}')")
        self._logger.debug(f"Cacher logging DEBUG level enabled")

    def _open(self):
        self._cache = shelve.open(self._cache_path, writeback=False)

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

        return CacherShelve(name=os.path.join(self._name, name).strip("/"),
                            cache_base_dir=self._cache_base_dir,
                            log_level=log_level, disabled=self._disabled)


    def _exists(self, name: str) -> bool:
        if self._disabled:
            self._logger.debug("Cacher disabled")
            return False

        ret_val = False
        with self._lock:
            if self._enable_thread_safeness:
                self._open()
            if name in self._cache:
                self._logger.debug("Cache entry %s::%s exists", self._cache_dir, name)
                try:
                    existing_entry = self._cache[name]
                except Exception as e:
                    self._logger.error(f"Cache entry failed to fetch cached entry: {e}")
                    self._logger.error(f"Existing keys: {self._cache.keys()}")
                    if self._enable_thread_safeness:
                        self._close()
                    raise e
                self._logger.debug(f"Cache existing entry: {existing_entry}")
                entry = self._migrate(name, existing_entry)
                if self._cache[name]['ttl_s'] is None:
                    ret_val = True
                else:
                    current_time = current_unix_timestamp()
                    if current_time - entry['timestamp'] < entry['ttl_s']:
                        ret_val = True
                    else:
                        del self._cache[name]
            if self._enable_thread_safeness:
                self._close()
            self._logger.debug("Cache entry %s::%s does not exist", self._cache_dir, name)
        return ret_val

    def set(self, name: str, content: Any, ttl_s: int | None = None):
        """
        Dumps a given object under a given cache entry name. The object must be pickleable.
        """
        if self._disabled:
            return

        with self._lock:
            if self._enable_thread_safeness:
                self._open()
            self._cache[name] = self._envelop(content, ttl_s=ttl_s)
            if self._enable_thread_safeness:
                self._close()

        self._logger.debug("Cache set %s::%s", self._cache_dir, name)

    def get(self, name: str) -> Any:
        """
        Loads an object for a given cache entry name. If it doesn't exist an exception is thrown.
        """
        if self._disabled:
            raise CacheEntryDoesNotExistError(f"Caching is disabled...")

        self._logger.debug("Cache get: %s::%s", self._cache_dir, name)

        ret_val = None

        with self._lock:
            if self._enable_thread_safeness:
                self._open()
            if name in self._cache:
                entry = self._migrate(name, self._cache[name])
                if entry['ttl_s'] is None:
                    ret_val = entry['data']
                else:
                    current_time = current_unix_timestamp()
                    if current_time - entry['timestamp'] < entry['ttl_s']:
                        ret_val = entry['data']
                    else:
                        del self._cache[name]
            if self._enable_thread_safeness:
                self._close()
        if ret_val is not None:
            return ret_val
        raise CacheEntryDoesNotExistError(f"Cache does not exist: {self._cache_dir}::{name}.")

    def unset(self, name: str):
        """
        Removes a cache entry for a given name
        """
        self._logger.debug("Cache unset: %s::%s", self._cache_dir, name)
        if self._disabled:
            return

        if name in self._cache:
            with self._lock:
                if self._enable_thread_safeness:
                    self._open()
                del self._cache[name]
                if self._enable_thread_safeness:
                    self._close()
            self._logger.debug("Cache entry %s::%s removed", self._name, name)
        else:
            self._logger.warning("Cache entry %s::%s does not exist, nothing to remove", self._name, name)

    def _close(self):
        self._cache.close()

    def persist(self):
        if not self._enable_thread_safeness:
            with self._lock:
                self._close()

    def _migrate(self, name: str, entry: Any) -> dict[str, Any]:
        if not isinstance(entry, dict) or ("data" not in entry and "timestamp" not in entry and "ttl_s" not in entry):
            self._logger.info(f"Migrating data entry to handle TTL")
            self._logger.info(f"> entry: {entry} (type: {type(entry)})")
            self.set(name, entry)
            entry = self._cache[name]
        return entry


def testing():
    cacher = CacherShelve()
    print(cacher._name)


if __name__ == "__main__":
    testing()
