import datetime
import time
from enum import Enum


class Seconds(float, Enum):
    NANOSECOND = 1 / 1000000000
    MICROSECOND = 1 / 1000000
    MILLISECOND = 1 / 1000
    SECOND = 1
    MINUTE = 60
    HOUR = 60 * MINUTE
    DAY = 24 * HOUR
    WEEK = 7 * DAY
    MONTH = 30 * DAY


DEFAULT_DATE_FORMAT = "%d-%m-%Y"
DEFAULT_DATETIME_FORMAT = "%d-%m-%Y %H:%M:%S"


def format_str(datetime_str: str, date_format: str, target_datetime_format: str = DEFAULT_DATETIME_FORMAT) -> str:
    datetime_obj = datetime.datetime.strptime(datetime_str, date_format)
    return datetime_to_str(datetime_obj, datetime_format=target_datetime_format)


def date_to_str(date_obj: datetime.date, date_format: str = DEFAULT_DATE_FORMAT) -> str:
    return date_obj.strftime(date_format)


def datetime_to_str(datetime_obj: datetime.datetime, datetime_format: str = DEFAULT_DATETIME_FORMAT) -> str:
    return datetime_obj.strftime(datetime_format)


def str_to_datetime(datetime_str: str, datetime_format: str = DEFAULT_DATETIME_FORMAT) -> datetime.datetime:
    return datetime.datetime.strptime(datetime_str, datetime_format)


def str_to_date(date_str: str, date_format: str = DEFAULT_DATE_FORMAT) -> datetime.date:
    return datetime.datetime.strptime(date_str, date_format).date()


def current_date_str(date_format: str = DEFAULT_DATE_FORMAT, time_delta_days: int | None = None) -> str:
    date_ = datetime.date.today()
    if time_delta_days is not None:
        date_ -= datetime.timedelta(days=time_delta_days)
    return date_to_str(date_, date_format)


def current_datetime_str(datetime_format: str = DEFAULT_DATETIME_FORMAT, time_delta_days: int | None = None) -> str:
    datetime_ = datetime.datetime.now()
    if time_delta_days is not None:
        datetime_ -= datetime.timedelta(days=time_delta_days)
    return datetime_to_str(datetime_, datetime_format)


def seconds_to_time_str(seconds: int | float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def datetime_to_unix_timestamp(datetime_ob: datetime.datetime) -> int:
    return int(datetime_ob.timestamp())


def unix_timestamp_to_datetime(unix_timestamp: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(unix_timestamp)


def str_to_unix_timestamp(datetime_str: str, datetime_format: str = DEFAULT_DATETIME_FORMAT) -> int:
    return datetime_to_unix_timestamp(str_to_datetime(datetime_str, datetime_format))


def current_unix_timestamp() -> int:
    return int(time.time())
