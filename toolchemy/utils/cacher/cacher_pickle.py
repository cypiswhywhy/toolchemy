import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Optional, Any
import shutil

from toolchemy.utils.cacher.common import BaseCacher, DummyLock, CacheEntryDoesNotExistError
from toolchemy.utils.logger import get_logger
from toolchemy.utils.locations import get_external_caller_path
from toolchemy.utils.utils import _caller_module_name
from toolchemy.utils.datestimes import current_unix_timestamp


class CacherPickle(BaseCacher):
    """
    Cacher implementation where cache is stored as a pickled local file
    """

    CACHER_MAIN_NAME = ".cache"

    def __init__(self, name: str | None = None, cache_base_dir: Optional[str] = None, disabled: bool = False,
                 log_level: int = logging.INFO, enable_thread_safeness: bool = False):
        """
        Initialize cache with its name. It creates .cache/name subdir in cache_base_dir directory
        """
        super().__init__()
        self._disabled = disabled
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

        with self._lock:
            os.makedirs(self._cache_dir, exist_ok=True)

        self._logger.debug(f"Cacher '{self._name}' initialized (cache dir: '{self._cache_dir}', log_level: '{logging.getLevelName(log_level)}')")
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
        self._logger.debug(f"> base name: {self._name}")
        self._logger.debug(f"> name: {name}")
        self._logger.debug(f"> log level: {log_level} ({logging.getLevelName(log_level)})")
        self._logger.debug(f"> is disabled: {self._disabled})")
        return CacherPickle(name=os.path.join(self._name, name).strip("/"), cache_base_dir=self._cache_base_dir,
                            log_level=log_level, disabled=self._disabled)

    def _exists(self, name: str) -> bool:
        if self._disabled:
            self._logger.debug("Cacher disabled")
            return False
        target_filename = self._cache_name(name)
        target_file = Path(target_filename)

        ret_val = False
        if target_file.is_file() or target_file.is_symlink():
            try:
                entry = self._get(name)
                if entry['ttl_s'] is None:
                    ret_val = True
                else:
                    current_time = current_unix_timestamp()
                    if current_time - entry['timestamp'] < entry['ttl_s']:
                        ret_val = True
                    else:
                        self.unset(name)
                        ret_val = False
            except CacheEntryDoesNotExistError:
                return False
            self._logger.debug("Cache entry %s::%s (%s) exists", self._name, name, target_filename)

        self._logger.debug("Cache entry %s::%s does not exist", self._name, name)
        return ret_val

    def set(self, name: str, content: Any, ttl_s: int | None = None):
        if self._disabled:
            return
        target_filename = self._cache_name(name)
        with self._lock:
            with open(target_filename, "wb") as file:
                try:
                    pickle.dump(self._envelop(content, ttl_s=ttl_s), file)  # type: ignore
                except TypeError as e:
                    self._logger.error(f"Wrong type of the serialized content: {type(content)}. Target: {target_filename}. Content:\n{content}")
                    shutil.rmtree(target_filename)
                    raise e
        self._logger.debug("Cache set %s::%s (file: %s)", self._name, name, target_filename)

    def get(self, name: str) -> Any:
        entry = self._get(name)
        return entry["data"]

    def _get(self, name: str) -> Any:
        if self._disabled:
            raise CacheEntryDoesNotExistError(f"Caching is disabled...")
        target_filename = self._cache_name(name)
        self._logger.debug("Cache get: %s::%s (file: %s)", self._name, name, target_filename)
        target_file = Path(target_filename)
        with self._lock:
            if target_file.is_file() or target_file.is_symlink():
                with open(target_filename, "rb") as file:
                    try:
                        restored_object = pickle.load(file, encoding="utf-8")
                    except ModuleNotFoundError as e:
                        self._logger.error(f"{e} while loading from file: '{target_filename}'")
                        raise e
                if restored_object['ttl_s'] is not None:
                    current_time = current_unix_timestamp()
                    if current_time - restored_object['timestamp'] >= restored_object['ttl_s']:
                        self.unset(name)
                        raise CacheEntryDoesNotExistError(f"Cache does not exist: {self._name}::{name}. Path: {target_filename}")
                return restored_object
        raise CacheEntryDoesNotExistError(f"Cache does not exist: {self._name}::{name}. Path: {target_filename}")

    def unset(self, name: str):
        """
        Removes a cache entry for a given name
        """
        target_filename = self._cache_name(name)
        self._logger.debug("Cache unset: %s::%s (file: %s)", self._name, name, target_filename)
        target_file = Path(target_filename)
        if target_file.is_file() or target_file.is_symlink():
            os.remove(target_filename)
            self._logger.debug("Cache entry %s::%s removed", self._name, name)
        else:
            self._logger.warning("Cache entry %s::%s does not exist, nothing to remove", self._name, name)

    def _cache_name(self, name: str):
        return os.path.join(self._cache_dir, f"{name}.pkl")
