import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from toolchemy.utils.utils import pp, pp_cast, to_json, truncate
from toolchemy.utils.datestimes import str_to_datetime, datetime_to_str


@dataclass
class Foo:
    a: int


@dataclass
class Bar:
    b: str
    c: list[Foo]


class FooBar:
    def __init__(self, a: int, b: str, c: list[Foo]):
        self.a = a
        self.b = b
        self.c = c

    def to_dict(self):
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
        }


def test_pp_cast_dict():
    input_data = {
        "a": 1,
        "b": "b_value",
        "c": {
            "d": 1.123124312412,
            "e": [1, 2.123213123, 3],
            "f": np.array([1.123, 2.456])
        },
        "g": "foo".encode("utf-8")
    }

    expected_output = {
        "a": 1,
        "b": "b_value",
        "c": {
            "d": "1.12",
            "e": [1, "2.12", 3],
            "f": ["1.12", "2.46"]
        },
        "g": "foo",
    }

    output = pp_cast(input_data)

    assert output == expected_output

def test_pp_cast_no_serializable():
    input_data = {
        "foo": Foo(1),
        "bar": Bar("bar2", [Foo(2), Foo(22)]),
        "foobar": FooBar(3, "foobar3", [Foo(3), Foo(33)]),
    }

    expected_output = {
        "foo": {"a": 1},
        "bar": {"b": "bar2", "c": [{"a": 2}, {"a": 22}]},
        "foobar": {"a": 3, "b": "foobar3", "c": [{"a": 3}, {"a": 33}]},
    }

    output = pp_cast(input_data)

    print(output)
    print(expected_output)

    assert output == expected_output


def test_pp_cast_datetime_within_dict():
    expected_datetime_str = "21-01-1988 13:42:55"
    expected_datetime = str_to_datetime(expected_datetime_str)
    input_data = {
        "a": 1,
        "b": expected_datetime,
    }

    expected_output = {
        "a": 1,
        "b": expected_datetime_str,
    }

    output = pp_cast(input_data)

    assert output == expected_output


def test_pp_cast_datetime_within_list():
    expected_datetime_str = "21-01-1988 13:42:55"
    expected_datetime = str_to_datetime(expected_datetime_str)
    input_data = [1, expected_datetime]

    expected_output = [1, expected_datetime_str]

    output = pp_cast(input_data)

    assert output == expected_output


def test_pp_cast_dict_skip_field():
    input_data = {
        "a": 1,
        "b": "b_value",
        "c": {
            "d": 1.123124312412,
            "e": [1, 2.123213123, 3],
            "f": np.array([1.123, 2.456])
        }
    }

    expected_output = {
        "a": 1,
        "c": {
            "d": "1.12",
            "e": [1, "2.12", 3],
        }
    }

    output = pp_cast(input_data, skip_fields=["b", "f"])

    assert output == expected_output


def test_pp_cast_list():
    input_data = [{
        "a": 1,
        "b": "b_value",
        "c": {
            "d": 1.123124312412,
            "e": [1, 2.123213123, 3],
            "f": np.array([1.123, 2.456])
        }
    }, 123]

    expected_output = [{
        "a": 1,
        "b": "b_value",
        "c": {
            "d": "1.12",
            "e": [1, "2.12", 3],
            "f": ["1.12", "2.46"]
        }
    }, 123]

    output = pp_cast(input_data)

    assert output == expected_output


def test_pp_cast_list_of_lists():
    input_data = [{
        "a": 1,
        "b": "b_value",
        "c": {
            "d": 1.123124312412,
            "e": [1, 2.123213123, 3],
            "f": [np.array([1.123, 2.456])]
        }
    }, 123]

    expected_output = [{
        "a": 1,
        "b": "b_value",
        "c": {
            "d": "1.12",
            "e": [1, "2.12", 3],
            "f": [["1.12", "2.46"]]
        }
    }, 123]

    output = pp_cast(input_data)

    assert output == expected_output


def test_pp_cast_safety_dict():
    input_data = {
        "d": 1.123124312412,
        "e": [1, 2.123213123, 3],
        "f": [[1.123, 2.456]],
        "g": {
            "h": 1.555222,
        }
    }
    expected_input_data = {
        "d": 1.123124312412,
        "e": [1, 2.123213123, 3],
        "f": [[1.123, 2.456]],
        "g": {
            "h": 1.555222,
        }
    }
    _ = pp_cast(input_data)

    assert input_data == expected_input_data


def test_pp_with_datetime_within_dict():
    input_data = {
        "foo": "bar",
        "d": datetime.now(),
    }

    expected_msg = {
        "foo": "bar",
        "d": datetime_to_str(input_data["d"]),
    }

    assert pp(expected_msg, print_msg=False) == pp(input_data, print_msg=False)


def test_pp_with_datetime_with_timezone_within_dict():
    input_data = {
        "foo": "bar",
        "d": datetime.now(timezone(timedelta(hours=2))),
    }

    expected_msg = {
        "foo": "bar",
        "d": datetime_to_str(input_data["d"]),
    }

    assert pp(expected_msg, print_msg=False) == pp(input_data, print_msg=False)


def test_pp_with_datetime_within_list():
    input_data = ["foo", datetime.now()]

    expected_msg = ["foo", datetime_to_str(input_data[1])]

    assert pp(expected_msg, print_msg=False) == pp(input_data, print_msg=False)


def test_pp_with_bytes():
    input_data = {
        "foo": "bar",
        "bar": "foo".encode("utf-8"),
    }

    expected_msg = {
        "foo": "bar",
        "bar": "foo",
    }

    assert pp(expected_msg, print_msg=False) == pp(input_data, print_msg=False)


def test_pp_with_bytes_unknown_encoding():
    input_data = {
        "foo": "bar",
        "bar": b"\x01\x15(\xab$(\xad\x92\xab\xf8\x1aO-7p\n\xfbT\xcf\xe4\xb7`\x0fa\xd2\xa2S\x19\xc0\x04\xe3\xb9\xa9\x18\x85\x10.\xee\t\x07\xa1\xa8\xc8O\xee\x90>\xb51\x01\x81\xc6q\x9e28\xa9\x16\x15d\xdc_\x18\xe3'\xd6\x8b\x86\x85\x07^\xeasE]\x96\x15H\xf2\x1f\xf2\xa2\x98\x86\xcc\xc3\x83\xb4t\xcd!\x8d\x03\x9c.=\xb3E\x15=\x86H\x0e3\xec*\xb0\x99\x8c\xa1\t\xf9\t\x07\x14QM\x12\\h\xd5\xadI=y\xfd(\xa2\x8a\x04\x7flalala"
    }

    assert "lalala" in pp(input_data, print_msg=False)


def test_to_dict():
    input_data = Bar(b="test", c=[Foo(1), Foo(2)])

    expected_result = {
        "b": "test",
        "c": [{"a": 1}, {"a": 2}]
    }

    assert expected_result == to_json(input_data)


@pytest.mark.parametrize("input_str,limit,expected_str", [
    ("test", 4, "test"),
    ("test", 5, "test"),
    ("test test", 4, "test (...5 more chars)"),
])
def test_truncate(input_str: str, limit: int, expected_str: str):
    assert expected_str == truncate(input_str, limit)
