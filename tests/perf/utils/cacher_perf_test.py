import pytest
import tempfile
from toolchemy.utils.cacher import ICacher, CacherPickle, CacherShelve, CacherDiskcache

DATA_COUNT = 100
DATA_SIZE = 100
DATA_COUNT_LARGE = 100
DATA_SIZE_LARGE = 100000
SHARDS = 8


def _generate_input_data(item_count: int, item_size: int) -> list[dict]:
    data = []
    for i in range(item_count):
        entry = {}
        for j in range(item_size):
            entry[f"entry_{str(i)}_{str(j)}"] = f"value_{str(i)}_{j}"
        data.append(entry)
    return data

@pytest.fixture
def input_data():
    return _generate_input_data(DATA_COUNT, DATA_SIZE)


@pytest.fixture
def input_data_large():
    return _generate_input_data(DATA_COUNT_LARGE, DATA_SIZE_LARGE)


def benchmark_set(cacher: ICacher, data: list):
    for i, item in enumerate(data):
        cacher.set(f"cache_key_{str(i)}", item)


def benchmark_get(cacher: ICacher, item_count: int):
    for i in range(item_count):
        _ = cacher.get(f"entry_{str(i)}")


def benchmark_exists(cacher: ICacher, item_count: int):
    for i in range(item_count):
        cacher.exists(f"entry_{str(i)}")


def _prefill_cacher(cacher: ICacher, input_data):
    for i, entry in enumerate(input_data):
        cacher.set(f"entry_{str(i)}", entry)


@pytest.mark.benchmark(group="set")
def test_pickle_set(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir)
        benchmark(benchmark_set, cacher=cacher, data=input_data)


@pytest.mark.benchmark(group="set")
def test_shelve_set(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir)
        benchmark(benchmark_set, cacher=cacher, data=input_data)
        cacher.persist()


@pytest.mark.benchmark(group="set")
def test_diskcache_set(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir)
        benchmark(benchmark_set, cacher=cacher, data=input_data)
        cacher.persist()


@pytest.mark.benchmark(group="set")
def test_pickle_t_safe_set(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data)


@pytest.mark.benchmark(group="set")
def test_shelve_t_safe_set(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data)
        cacher.persist()


@pytest.mark.benchmark(group="set")
def test_diskcache_t_safe_set(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, thread_safe=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data)
        cacher.persist()


@pytest.mark.benchmark(group="set")
def test_diskcache_t_safe_fanout_set(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, shards=SHARDS, thread_safe=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data)
        cacher.persist()


@pytest.mark.benchmark(group="set_large")
def test_pickle_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)


@pytest.mark.benchmark(group="set_large")
def test_shelve_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)
        cacher.persist()


@pytest.mark.benchmark(group="set_large")
def test_diskcache_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)
        cacher.persist()


@pytest.mark.benchmark(group="set_large")
def test_diskcache_fanout_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, shards=SHARDS)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)
        cacher.persist()


@pytest.mark.benchmark(group="set_large")
def test_pickle_t_safe_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)


@pytest.mark.benchmark(group="set_large")
def test_shelve_t_safe_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)
        cacher.persist()


@pytest.mark.benchmark(group="set_large")
def test_diskcache_t_safe_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, thread_safe=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)
        cacher.persist()


@pytest.mark.benchmark(group="set_large")
def test_diskcache_t_safe_set_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, shards=SHARDS, thread_safe=True)
        benchmark(benchmark_set, cacher=cacher, data=input_data_large)
        cacher.persist()


@pytest.mark.benchmark(group="get")
def test_pickle_get(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data))


@pytest.mark.benchmark(group="get")
def test_shelve_get(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="get")
def test_diskcache_get(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="get")
def test_pickle_t_safe_get(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data))


@pytest.mark.benchmark(group="get")
def test_shelve_t_safe_get(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="get")
def test_diskcache_t_safe_get(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="get")
def test_diskcache_t_safe_fanout_get(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, shards=8, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="get_large")
def test_pickle_get_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data_large))


@pytest.mark.benchmark(group="get_large")
def test_shelve_get_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="get_large")
def test_diskcache_get_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="get_large")
def test_pickle_t_safe_get_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data_large))


@pytest.mark.benchmark(group="get_large")
def test_shelve_t_safe_get_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir, enable_thread_safeness=True)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="get_large")
def test_diskcache_t_safe_get_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="get_large")
def test_diskcache_t_safe_get_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, shards=SHARDS, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_get, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="exists")
def test_pickle_exists(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data))


@pytest.mark.benchmark(group="exists")
def test_shelve_exists(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="exists")
def test_diskcache_exists(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="exists")
def test_diskcache_t_safe_exists(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="exists")
def test_diskcache_t_safe_fanout_exists(benchmark, input_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, shards=SHARDS, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data))
        cacher.persist()


@pytest.mark.benchmark(group="exists_large")
def test_pickle_exists_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherPickle(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data_large))


@pytest.mark.benchmark(group="exists_large")
def test_shelve_exists_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherShelve(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="exists_large")
def test_diskcache_exists_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="exists_large")
def test_diskcache_t_safe_exists_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()


@pytest.mark.benchmark(group="exists_large")
def test_diskcache_t_safe_fanout_exists_large(benchmark, input_data_large):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cacher = CacherDiskcache(cache_base_dir=tmp_dir, shards=SHARDS, thread_safe=True)
        _prefill_cacher(cacher=cacher, input_data=input_data_large)
        benchmark(benchmark_exists, cacher=cacher, item_count=len(input_data_large))
        cacher.persist()
