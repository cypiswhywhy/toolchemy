from toolchemy.utils.logger import get_logger
from freezegun import freeze_time


@freeze_time("2025-09-12 09:07:12 CEST")
def test_info(capsys):
    logger = get_logger(say_hi=False)

    expected_msg = "\x1b[32m|09:07:12 logger_test INFO|\x1b[0m test\x1b[0m"

    logger.info("test")
    captured = capsys.readouterr()
    captured_msg = captured.err.strip()
    print({"x": captured_msg})

    assert captured_msg == expected_msg


@freeze_time("2025-09-12 09:07:12 CEST")
def test_info_without_time(capsys):
    logger = get_logger(with_time=False, say_hi=False)

    expected_msg = "\x1b[32m|logger_test INFO|\x1b[0m test\x1b[0m"

    logger.info("test")
    captured = capsys.readouterr()
    captured_msg = captured.err.strip()

    assert captured_msg == expected_msg


@freeze_time("2025-09-12 09:07:12 CEST")
def test_info_short_module_name(capsys):
    logger = get_logger(short_module_name=True, say_hi=False)

    expected_msg = "\x1b[32m|09:07:12 logger_test INFO|\x1b[0m test\x1b[0m"

    logger.info("test")
    captured = capsys.readouterr()
    captured_msg = captured.err.strip()

    assert captured_msg == expected_msg


def test_name_when_in_parent():
    from toolchemy.ai.trackers.in_memory_tracker import InMemoryTracker

    tracker = InMemoryTracker()

    assert tracker._logger.name == "toolchemy.ai.trackers.in_memory_tracker"
