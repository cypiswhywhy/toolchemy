from unittest.mock import patch

from toolchemy.utils.timer import Timer


@patch('time.time')
def test_tap(time_mock):
    time_mock.return_value = 1000.0
    timer = Timer()
    time_mock.return_value = 1010.0
    _ = timer.tap()
    time_mock.return_value = 1020.0
    time_diff = timer.tap()

    assert time_diff == 20.0


@patch('time.time')
def test_tap_since_last(time_mock):
    time_mock.return_value = 1000.0
    timer = Timer()
    time_mock.return_value = 1010.0
    _ = timer.tap()
    time_mock.return_value = 1020.0
    time_diff = timer.tap(since_last=True)

    assert time_diff == 10.0


@patch('time.time')
def test_tap_since_last_first_tap(time_mock):
    time_mock.return_value = 1000.0
    timer = Timer()
    time_mock.return_value = 1010.0
    time_diff = timer.tap(since_last=True)

    assert time_diff == 10.0


@patch('time.time')
def test_tap_returns_zero_until_an_unstarted_timer_is_reset(time_mock):
    time_mock.return_value = 1000.0
    timer = Timer(started=False)

    time_mock.return_value = 1010.0
    assert timer.tap() == 0.0

    timer.reset()
    time_mock.return_value = 1020.0
    assert timer.tap() == 10.0
