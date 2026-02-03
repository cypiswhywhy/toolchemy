import pytest
import datetime
from freezegun import freeze_time

from toolchemy.utils.datestimes import seconds_to_time_str, datetime_to_unix_timestamp, unix_timestamp_to_datetime, format_str, current_date_str


@pytest.mark.parametrize("seconds,expected_time", [
        (1, "00:00:01"),
        (59, "00:00:59"),
        (60, "00:01:00"),
        (3599, "00:59:59"),
        (3600, "01:00:00"),
        (360000, "100:00:00"),
    ])
def test_seconds_to_time_str(seconds: int, expected_time: str):
    assert expected_time == seconds_to_time_str(seconds)


def test_datetime_to_unix_timestamp_to_datetime():
    dt = datetime.datetime.now()
    dt2 = unix_timestamp_to_datetime(datetime_to_unix_timestamp(dt))

    assert datetime_to_unix_timestamp(dt) == datetime_to_unix_timestamp(dt2)


def test_format_str():
    datetime_str = "26-01-1985 11:12:13"
    expected_datetime_str = "26/01/1985 11:12"

    assert expected_datetime_str == format_str(datetime_str, "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M")


@freeze_time("2023-12-26")
def test_current_date_str():
    current_date = current_date_str()
    assert current_date == "26-12-2023"


@freeze_time("2023-12-26")
def test_current_date_str_with_delta():
    current_date = current_date_str(time_delta_days=10)
    assert current_date == "16-12-2023"
