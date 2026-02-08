import abc
import hashlib
import copy
from abc import abstractmethod
from typing import Any

from toolchemy.utils.at_exit_collector import ICollectable, AtExitCollector
from toolchemy.utils.datestimes import current_date_str, current_unix_timestamp


class CacherInitializationError(Exception):
    pass


class CacheEntryDoesNotExistError(Exception):
    pass


class CacheEntryHasNotBeenSetError(Exception):
    pass


class ICacher(abc.ABC):
    CACHER_MAIN_NAME = ".cache"

    """
    Cacher interface
    """

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
        hash_object = hashlib.md5(name.encode('utf-8'))
        return hash_object.hexdigest()

    @staticmethod
    def create_cache_key(parts_plain: list | dict | str | None = None, parts_hashed: list | dict | str | None = None,
                         with_current_date: bool = False) -> str:
        replaceable_chars = "*.,'\"|<>[]?!-:;()@#$%^&{} "
        if parts_plain is None and parts_hashed is None:
            raise ValueError(f"You must provide the key components")
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

        for i, part_plain in enumerate(parts_plain):
            for char_to_replace in list(replaceable_chars):
                parts_plain[i] = str(parts_plain[i]).replace(char_to_replace, "_")

        parts_hashed = [BaseCacher.hash(str(part_hashed)) for part_hashed in parts_hashed]
        parts = parts_plain + parts_hashed
        if with_current_date:
            parts.append(current_date_str("%Y%m%d"))

        return "_".join(parts)

    def _envelop(self, content: Any, ttl_s: int | None = None) -> dict[str, Any]:
        if not isinstance(content, dict) or ("data" not in content and "timestamp" not in content and "ttl_s" not in content):
            entry_timestamp = current_unix_timestamp()
            content = {'data': content, 'timestamp': entry_timestamp, 'ttl_s': ttl_s}
        return content


class DummyLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyCacher(BaseCacher):
    def __init__(self, with_memory_store: bool = False):
        super().__init__()
        self._data = {}
        self._with_memory_store = with_memory_store

    @property
    def cache_location(self) -> str:
        return ""

    def sub_cacher(self, log_level: int | None = None, suffix: str | None = None) -> "ICacher":
        return DummyCacher(with_memory_store=self._with_memory_store)

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
