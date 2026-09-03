import pytest
import tempfile
from toolchemy.utils.cacher import ICacher, BaseCacher, CacherPickle, CacherShelve, CacherDiskcache

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
            entry[f"entry_{i!s}_{j!s}"] = f"value_{i!s}_{j}"
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
        cacher.set(f"cache_key_{i!s}", item)


def benchmark_get(cacher: ICacher, item_count: int):
    for i in range(item_count):
        _ = cacher.get(f"entry_{i!s}")


def benchmark_exists(cacher: ICacher, item_count: int):
    for i in range(item_count):
        cacher.exists(f"entry_{i!s}")


def _prefill_cacher(cacher: ICacher, input_data):
    for i, entry in enumerate(input_data):
        cacher.set(f"entry_{i!s}", entry)


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
def test_diskcache_t_safe_fanout_set_large(benchmark, input_data_large):
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
def test_diskcache_t_safe_fanout_get_large(benchmark, input_data_large):
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


@pytest.mark.benchmark(group="cache_key")
def test_create_cache_key_short(benchmark):
    """
    Benchmarks create_cache_key, which sits on the hot path of every cached completion.

    Its plain-part sanitiser runs one str.replace per replaceable character, which looks
    quadratic but measures faster than str.translate, because replace short-circuits on
    absent characters. These benchmarks exist so that trade-off is re-checked with data
    rather than by eye.
    """
    benchmark(BaseCacher.create_cache_key, ["llm_completion_json"], ["system prompt", "prompt"])


@pytest.mark.benchmark(group="cache_key")
def test_create_cache_key_long_plain_part(benchmark):
    long_part = "Summarize the following document, keeping names, dates and figures intact. " * 20
    benchmark(BaseCacher.create_cache_key, [long_part], ["system prompt", "prompt"])
