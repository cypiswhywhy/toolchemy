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


@freeze_time("2025-09-12 09:07:12 CEST")
def test_get_logger_does_not_override_existing_format(capsys):
    logger_name = "toolchemy.tests.logger_isolation"
    logger = get_logger(name=logger_name, say_hi=False)

    expected_msg = (
        "\x1b[32m|09:07:12 toolchemy.tests.logger_isolation INFO|\x1b[0m first\x1b[0m"
    )

    logger.info("first")
    captured = capsys.readouterr()
    captured_msg = captured.err.strip()

    assert captured_msg == expected_msg

    other_logger = get_logger(
        name=logger_name, with_module_name=False, with_log_level=False, say_hi=False
    )

    logger.info("second")
    captured = capsys.readouterr()
    captured_msg = captured.err.strip()
    expected_msg = (
        "\x1b[32m|09:07:12 toolchemy.tests.logger_isolation INFO|\x1b[0m second\x1b[0m"
    )
    assert captured_msg == expected_msg

    other_logger.info("third")
    captured = capsys.readouterr()
    captured_msg = captured.err.strip()
    expected_msg = "\x1b[32m|09:07:12|\x1b[0m third\x1b[0m"
    assert captured_msg == expected_msg


def test_name_when_in_parent():
    from toolchemy.ai.trackers.in_memory_tracker import InMemoryTracker

    tracker = InMemoryTracker()

    assert tracker._logger.name == "toolchemy.ai.trackers.in_memory_tracker"
