from .common import ICacher, BaseCacher, CacheEntryDoesNotExistError, DummyCacher
from .cacher_pickle import CacherPickle
from .cacher_shelve import CacherShelve
from .cacher_diskcache import CacherDiskcache


class Cacher(CacherDiskcache):
    pass


__all__ = [
    "ICacher",
    "BaseCacher",
    "Cacher",
    "CacherPickle",
    "CacherShelve",
    "CacherDiskcache",
    "DummyCacher",
    "CacheEntryDoesNotExistError",
]
