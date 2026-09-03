import abc
import hashlib
import copy
import logging
import os
from abc import abstractmethod
from typing import Any

from toolchemy.utils.at_exit_collector import ICollectable, AtExitCollector
from toolchemy.utils.datestimes import current_date_str, current_unix_timestamp
from toolchemy.utils.locations import get_external_caller_path
from toolchemy.utils.logger import get_logger
from toolchemy.utils.utils import _caller_module_name

# Only correct when called directly from a concrete cacher, because _caller_module_name
# walks the stack: it -> _init_common/_sub_cacher_params -> Subclass.__init__ -> caller.
_CALLER_STACK_OFFSET = 3


class CacherInitializationError(Exception):
    pass


class CacheEntryDoesNotExistError(Exception):
    pass


class CacheEntryHasNotBeenSetError(Exception):
    pass


class CacheEntrySeemMalformedError(Exception):
    pass


class ICacher(abc.ABC):
    """
    Cacher interface
    """

    CACHER_MAIN_NAME = ".cache"

    @abstractmethod
    def sub_cacher(self, log_level: int | None = None, suffix: str | None = None) -> "ICacher":
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """
        Checks if there is a cache entry for a given name
        """

    @abstractmethod
    def set(self, name: str, content: Any, ttl_s: int | None = None):
        """
        Dumps a given object under a given cache entry name. The object must be pickleable.
        """

    @abstractmethod
    def unset(self, name: str):
        """
        Removes a cache entry for a given name
        """

    @abstractmethod
    def get(self, name: str) -> Any:
        """
        Loads an object for a given cache entry name. If it doesn't exist, an exception is thrown.
        """

    @staticmethod
    @abstractmethod
    def create_cache_key(parts_plain: list | dict | str | None = None, parts_hashed: list | dict | str | None = None,
                         with_current_date: bool = False) -> str:
        pass

    @property
    @abstractmethod
    def cache_location(self) -> str:
        pass


class BaseCacher(ICacher, ICollectable, abc.ABC):
    def __init__(self):
        self._name = self.__module__
        self._cache_stats = {
            "hit": 0,
            "miss": 0,
        }
        AtExitCollector.register(self)

    def collect(self) -> dict:
        return self._cache_stats

    def label(self) -> str:
        return f"{self.__class__.__name__}({self._name})"

    def _init_common(self, name: str | None, cache_base_dir: str | None, disabled: bool, log_level: int) -> None:
        """
        Sets up the logger, cache name and cache directory shared by every file-backed cacher.

        Must be called directly from a concrete cacher's __init__: when `name` is not given it
        falls back to the name of the module that constructed the cacher (see _CALLER_STACK_OFFSET).
        """
        self._disabled = disabled
        self._log_level = log_level
        self._logger = get_logger(level=self._log_level)

        self._name = name
        if not self._name:
            self._name = _caller_module_name(_CALLER_STACK_OFFSET)

        self._cache_base_dir = cache_base_dir
        if self._cache_base_dir is None:
            self._cache_base_dir = get_external_caller_path()

        self._cache_dir = os.path.join(self._cache_base_dir, self.CACHER_MAIN_NAME, self._name)

    def _sub_cacher_params(self, log_level: int | None, suffix: str | None) -> tuple[str, int]:
        """
        Builds the (name, log_level) a sub cacher is constructed with.

        Must be called directly from a concrete cacher's sub_cacher(), for the same reason
        as _init_common.
        """
        name = _caller_module_name(_CALLER_STACK_OFFSET)
        if suffix:
            name += f"__{suffix}"
        if log_level is None:
            log_level = self._log_level

        self._logger.debug("Creating sub cacher")
        self._logger.debug(f"> base name: {self._name}")
        self._logger.debug(f"> base cache dir: {self._cache_dir}")
        self._logger.debug(f"> name: {name}")
        self._logger.debug(f"> log level: {log_level} ({logging.getLevelName(log_level)})")
        self._logger.debug(f"> is disabled: {self._disabled})")

        return os.path.join(self._name, name).strip("/"), log_level

    def _log_initialized(self) -> None:
        self._logger.debug(
            f"Cacher '{self._name}' initialized (cache dir: '{self._cache_dir}', log_level: '{logging.getLevelName(self._log_level)}')")

    def exists(self, name: str) -> bool:
        does_exist = self._exists(name)
        if does_exist:
            self._cache_stats["hit"] += 1
        else:
            self._cache_stats["miss"] += 1
        return does_exist

    @abc.abstractmethod
    def _exists(self, name: str) -> bool:
        """
        Checks if there is a cache entry for a given name
        """

    def persist(self):
        pass

    @staticmethod
    def hash(name: str) -> str:
        """
        Shortens one cache key component.

        md5 is a key shortener here, never a security primitive. usedforsecurity=False
        states that and keeps this working on FIPS builds; it does not change the digest,
        so cache entries written by earlier versions stay addressable.
        """
        hash_object = hashlib.md5(name.encode('utf-8'), usedforsecurity=False)
        return hash_object.hexdigest()

    @staticmethod
    def create_cache_key(parts_plain: list | dict | str | None = None, parts_hashed: list | dict | str | None = None,
                         with_current_date: bool = False) -> str:
        replaceable_chars = "*.,'\"|<>[]?!-:;()@#$%^&{} "
        if parts_plain is None and parts_hashed is None:
            raise ValueError("You must provide the key components")
        if parts_plain is None:
            parts_plain = []
        if parts_hashed is None:
            parts_hashed = []
        if isinstance(parts_plain, str):
            parts_plain = [parts_plain]
        if isinstance(parts_plain, dict):
            parts_plain = [f"{k}_{v}" for k, v in parts_plain.items()]
        if isinstance(parts_hashed, str):
            parts_hashed = [parts_hashed]
        if isinstance(parts_hashed, dict):
            parts_hashed = [f"{k}_{v}" for k, v in parts_hashed.items()]

        # build a new list rather than assigning into parts_plain: when the caller passes a
        # list it is theirs, and rewriting its elements in place is a side effect on their data
        sanitized_plain = []
        for part_plain in parts_plain:
            sanitized = str(part_plain)
            for char_to_replace in replaceable_chars:
                sanitized = sanitized.replace(char_to_replace, "_")
            sanitized_plain.append(sanitized)

        parts_hashed = [BaseCacher.hash(str(part_hashed)) for part_hashed in parts_hashed]
        parts = sanitized_plain + parts_hashed
        if with_current_date:
            parts.append(current_date_str("%Y%m%d"))

        return "_".join(parts)

    def _envelop(self, content: Any, ttl_s: int | None = None) -> dict[str, Any]:
        if not isinstance(content, dict) or ("data" not in content and "timestamp" not in content and "ttl_s" not in content):
            entry_timestamp = current_unix_timestamp()
            content = {'data': content, 'timestamp': entry_timestamp, 'ttl_s': ttl_s}
        return content


class DummyLock:
    """No-op stand-in for threading.RLock, used when thread safety is off."""

    def acquire(self, blocking: bool = False, timeout: int = -1) -> bool:
        return False

    def release(self):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyCacher(BaseCacher):
    def __init__(self, with_memory_store: bool = False, log_level: int = logging.INFO):
        super().__init__()
        self._data = {}
        self._with_memory_store = with_memory_store
        self._log_level = log_level
        self._logger = get_logger(level=log_level)

    @property
    def cache_location(self) -> str:
        return ""

    def sub_cacher(self, log_level: int | None = None, suffix: str | None = None) -> "ICacher":
        return DummyCacher(with_memory_store=self._with_memory_store,
                           log_level=self._log_level if log_level is None else log_level)

    def _exists(self, name: str) -> bool:
        if not self._with_memory_store:
            return False
        return name in self._data

    def set(self, name: str, content: Any, ttl_s: int | None = None):
        if not self._with_memory_store:
            return
        self._data[name] = copy.deepcopy(content)

    def unset(self, name: str):
        if name in self._data:
            del self._data[name]

    def get(self, name: str) -> Any:
        if not self._with_memory_store:
            return None
        if name not in self._data:
            raise CacheEntryDoesNotExistError()
        return self._data[name]
