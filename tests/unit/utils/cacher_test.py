import logging
import tempfile
import pytest
import os
import random
import string
import threading
from datetime import datetime
from unittest.mock import patch

from toolchemy.utils.cacher import BaseCacher, CacherPickle, CacherDiskcache, CacheEntryDoesNotExistError, CacherShelve


@pytest.mark.parametrize("parts_plain,parts_hashed,with_current_date,expected_key", [
    ([], ["h"], False, BaseCacher.hash("h")),
    (["replace:!?*:"], [], False, "replace_____"),
    ("replace:!?*:", [], False, "replace_____"),
    (["a"], ["h"], False, f"a_{BaseCacher.hash('h')}"),
    (["a", "b"], ["h", "h2"], False, f"a_b_{BaseCacher.hash('h')}_{BaseCacher.hash('h2')}"),
    ("a", ["h"], False, f"a_{BaseCacher.hash('h')}"),
    ("a", "h", False, f"a_{BaseCacher.hash('h')}"),
    ("a", "h", True, f"a_{BaseCacher.hash('h')}_20250613"),
    (["a", 2], ["b", 3], False, f"a_2_{BaseCacher.hash('b')}_{BaseCacher.hash('3')}"),
    (["with none", None], [], False, "with_none_None"),
    (["with int", 13], [], False, "with_int_13"),
    (["with hashed int"], [13], False, f"with_hashed_int_{BaseCacher.hash("13")}"),
])
@patch("toolchemy.utils.cacher.common.current_date_str", return_value="20250613")
def test_create_cache_key(_, parts_plain, parts_hashed, with_current_date, expected_key):
    cache_key = BaseCacher.create_cache_key(parts_plain, parts_hashed, with_current_date)
    assert cache_key == expected_key


@pytest.mark.parametrize("cacher_impl", [(CacherPickle), (CacherShelve), (CacherDiskcache)])
def test_polish_encoding(cacher_impl):
    with tempfile.TemporaryDirectory() as base_dir:
        input_object = {
            "text": "Zażąłóć gęślą jaźń",
        }

        cacher = cacher_impl(cache_base_dir=base_dir)
        cacher.set("test", input_object)
        retrieved_object = cacher.get("test")

        assert retrieved_object["text"] == input_object["text"]
        cacher.persist()


@pytest.mark.parametrize("cacher_impl", [(CacherPickle), (CacherShelve), (CacherDiskcache)])
def test_sub_cacher(cacher_impl):
    with tempfile.TemporaryDirectory() as base_dir:
        cacher = cacher_impl(cache_base_dir=base_dir)
        sub_cacher = cacher.sub_cacher()

        sub_cacher.set("testing", True)

        assert os.path.exists(base_dir)
        assert os.path.exists(os.path.join(base_dir, ".cache"))
        assert os.path.exists(os.path.join(base_dir, ".cache", "cacher_test"))
        assert os.path.exists(os.path.join(base_dir, ".cache", "cacher_test", "cacher_test"))

        cacher.persist()
        sub_cacher.persist()


@pytest.mark.parametrize("cacher_impl", [(CacherPickle), (CacherShelve), (CacherDiskcache)])
def test_sub_cacher_with_suffix_shelve(cacher_impl):
    with tempfile.TemporaryDirectory() as base_dir:
        expected_suffix = "variation"

        cacher = CacherShelve(cache_base_dir=base_dir, log_level=logging.DEBUG)
        sub_cacher = cacher.sub_cacher(suffix=expected_suffix)

        sub_cacher.set("testing", True)

        assert os.path.exists(base_dir)
        assert os.path.exists(os.path.join(base_dir, ".cache"))
        assert os.path.exists(os.path.join(base_dir, ".cache", "cacher_test"))
        assert os.path.exists(os.path.join(base_dir, ".cache", "cacher_test", "cache.dat"))
        assert os.path.exists(os.path.join(base_dir, ".cache", "cacher_test", f"cacher_test__{expected_suffix}", "cache.dat"))

        cacher.persist()
        sub_cacher.persist()


class Foo:
    z: float = 123.123

    def __init__(self):
        self._x = 123
        self._y = "abc"

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z


input_test_data = [
    ([{"foo": {"bar": 1}, "hmm": True}, 123], CacherPickle),
    ([{"foo": {"bar": 1}, "hmm": True}, 123], CacherShelve),
    ([{"foo": {"bar": 1}, "hmm": True}, 123], CacherDiskcache),
    (123, CacherPickle),
    (123, CacherShelve),
    (123, CacherDiskcache),
    (True, CacherPickle),
    (True, CacherShelve),
    (True, CacherDiskcache),
    (["a", "b"], CacherPickle),
    (["a", "b"], CacherShelve),
    (["a", "b"], CacherDiskcache),
    (Foo(), CacherPickle),
    (Foo(), CacherShelve),
    (Foo(), CacherDiskcache),
    (datetime.now(), CacherShelve),
    (datetime.now(), CacherDiskcache),
]
@pytest.mark.parametrize("input_data,cacher_impl", input_test_data)
def test_get_set_exists(input_data, cacher_impl):
    with tempfile.TemporaryDirectory() as base_dir:
        cache_key = "testing"
        cacher = cacher_impl(cache_base_dir=base_dir)
        assert not cacher.exists(cache_key)
        cacher.set(cache_key, input_data)
        assert cacher.exists(cache_key)
        cacher.persist()

        cacher2 = cacher_impl(cache_base_dir=base_dir)
        assert cacher2.exists(cache_key)
        retrieved_data = cacher2.get(cache_key)
        assert retrieved_data == input_data


@pytest.mark.parametrize("cacher_impl", [(CacherPickle), (CacherShelve), (CacherDiskcache)])
@patch('time.time')
def test_set_exists_with_ttl(time_mock, cacher_impl):
    input_data = {"foo": "bar"}
    with tempfile.TemporaryDirectory() as base_dir:
        cache_key = "testing"
        cacher = cacher_impl(cache_base_dir=base_dir)
        time_mock.return_value = 1000
        assert not cacher.exists(cache_key)
        time_mock.return_value = 1001
        cacher.set(cache_key, input_data, ttl_s=2)
        time_mock.return_value = 1002
        assert cacher.exists(cache_key)
        time_mock.return_value = 1004
        assert not cacher.exists(cache_key)
        cacher.persist()


@pytest.mark.parametrize("cacher_impl", [(CacherPickle), (CacherShelve), (CacherDiskcache)])
@patch('time.time')
def test_set_get_with_ttl(time_mock, cacher_impl):
    input_data = {"foo": "bar"}
    with tempfile.TemporaryDirectory() as base_dir:
        cache_key = "testing"
        cacher = cacher_impl(cache_base_dir=base_dir)
        time_mock.return_value = 1000
        assert not cacher.exists(cache_key)
        time_mock.return_value = 1001
        cacher.set(cache_key, input_data, ttl_s=2)
        time_mock.return_value = 1002
        result = cacher.get(cache_key)
        assert result == input_data
        time_mock.return_value = 1010
        with pytest.raises(CacheEntryDoesNotExistError):
            _ = cacher.get(cache_key)
        cacher.persist()


@pytest.mark.parametrize("size_in_bytes,cacher_impl", [
    (1000000, CacherPickle),
    (1000000, CacherShelve),
    (1000000, CacherDiskcache),
    (10000000, CacherPickle),
    (10000000, CacherShelve),
    (10000000, CacherDiskcache)
])
def test_set_get_large_data(size_in_bytes: int, cacher_impl):
    cache_key = "testing"
    input_data = {"random_chars": ''.join(random.choices(string.ascii_letters + string.digits, k=size_in_bytes))}
    with tempfile.TemporaryDirectory() as base_dir:
        cacher = cacher_impl(cache_base_dir=base_dir)
        cacher.set(cache_key, input_data)
        assert cacher.exists(cache_key)
        cacher.persist()


@pytest.mark.parametrize("cacher_impl", [(CacherPickle), (CacherShelve)])
def test_thread_safeness(cacher_impl):
    n_threads = 4
    threads = []

    input_data = {"foo": "bar"}

    base_dir = tempfile.TemporaryDirectory()

    cacher = cacher_impl(cache_base_dir=base_dir.name, enable_thread_safeness=True)

    def run_cacher():
        cache_key = f"testing"
        cacher.set(cache_key, input_data)
        assert cacher.exists(cache_key)
        retrieved_data = cacher.get(cache_key)
        assert retrieved_data == input_data

    for _ in range(n_threads):
        t = threading.Thread(target=run_cacher)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        assert not t.is_alive()
    base_dir.cleanup()
