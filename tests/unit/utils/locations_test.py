import json
import pytest
import tempfile
import os
from pathlib import Path

from toolchemy.utils.locations import Locations, get_external_caller_path


@pytest.mark.parametrize("objective_path_mode", [True, False])
def test_locations_in_root_dir(objective_path_mode: bool):
    locations = Locations(objective_path_mode=objective_path_mode)
    expected_path = os.path.join(get_external_caller_path(), "test")
    if objective_path_mode:
        expected_path = Path(expected_path)
    assert locations.in_root("test") == expected_path


@pytest.mark.parametrize("objective_path_mode", [True, False])
def test_locations_root_dir(objective_path_mode: bool):
    locations = Locations(objective_path_mode=objective_path_mode)
    expected_path = get_external_caller_path()
    if objective_path_mode:
        expected_path = Path(expected_path)
    assert locations.in_root() == expected_path


def test_get_external_caller_path():
    caller_path = get_external_caller_path()
    assert caller_path.endswith("toolchemy")


def test_custom_root_path():
    locations = Locations(root_path="/tmp")
    assert locations.in_root("test") == "/tmp/test"


def test_locations_in_data():
    locations = Locations()
    result_path = locations.in_data(["a", "b", "c"])
    assert result_path.endswith("data/a/b/c")


def test_locations_in_data_with_prefix_dirs():
    locations = Locations(prefix_dirs={"data": "subdir1/subdir2"})
    result_path = locations.in_resources("a")
    assert "subdir" not in result_path
    result_path = locations.in_data(["a", "b", "c"])
    assert result_path.endswith("data/subdir1/subdir2/a/b/c")


def test_locations_data_with_prefix_dirs():
    locations = Locations(prefix_dirs={"data": "subdir1/subdir2"})
    result_path = locations.in_resources("a")
    assert "subdir" not in result_path
    result_path = locations.in_data()
    assert result_path.endswith("data/subdir1/subdir2"
                                ""
                                ""
                                ""
                                ""
                                ""
                                ""
                                ""
                                ""
                                ""
                                "")


@pytest.mark.parametrize("objective_path_mode", [True, False])
def test_read_content(objective_path_mode: bool):
    expected_content = json.dumps({"key": "value", "number": 42})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        json.dump(json.loads(expected_content), f)
        temp_path = f.name
        if objective_path_mode:
            temp_path = Path(temp_path)

    try:
        result = Locations.read_content(temp_path)
        assert isinstance(result, str)
        assert result == expected_content
    finally:
        os.unlink(temp_path)


def test_read_json_objective_path():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value", "number": 42}, f)
        temp_path = Path(f.name)

    try:
        result = Locations.read_json(temp_path)
        assert isinstance(result, dict)
        assert result == {"key": "value", "number": 42}
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize("expected_data", [
    [
        {"title": "some title", "content": "some content"},
        {"title": "another title", "content": "another content"}
    ],
    [],
    {},
    {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ],
        "metadata": {"version": "1.0"}
    }
])
def test_read_json(expected_data):
    expected_type = type(expected_data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(expected_data, f)
        temp_path = f.name

    try:
        result = Locations.read_json(temp_path)
        assert isinstance(result, expected_type)
        assert result == expected_data
    finally:
        os.unlink(temp_path)


def test_read_json_invalid_file():
    with pytest.raises(FileNotFoundError):
        Locations.read_json("/nonexistent/path/file.json")


def test_read_json_invalid_json():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("this is not valid json")
        temp_path = f.name

    try:
        with pytest.raises(json.decoder.JSONDecodeError):
            Locations.read_json(temp_path)
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize("rel_path,expected_path", [
    ("path.txt", os.path.join("/tmp", "path.txt")),
    ("./path.txt", os.path.join("/tmp", "path.txt")),
    ("./subdir/path.txt", os.path.join("/tmp", "subdir", "path.txt")),
    ("./subdir/../path.txt", os.path.join("/tmp", "path.txt")),
    ("/already/abs/path.txt", "/already/abs/path.txt"),
    ("~/in/home/path.txt", os.path.join(os.path.expanduser("~"), "in", "home", "path.txt")),
    ("~/in/../home/../path.txt", os.path.join(os.path.expanduser("~"), "path.txt")),
])
def test_abs(rel_path: str, expected_path: str):
    current_cwd = os.getcwd()
    os.chdir("/tmp")
    abs_path = Locations.abs(rel_path)
    os.chdir(current_cwd)
    assert abs_path == expected_path


@pytest.mark.parametrize("path,expected_path", [
    ("path.txt", "./path.txt"),
    ("./path.txt", "./path.txt"),
    ("./subdir/path.txt", "./subdir/path.txt"),
    ("./subdir/subsubdir/../path.txt", "./subdir/path.txt"),
    ("{root}", "."),
    ("{root}/", "."),
    ("{root}/path.txt", "./path.txt"),
    ("{root}/abs/path.txt", "./abs/path.txt"),
    ("{root}/abs/subdir/path.txt", "./abs/subdir/path.txt"),
    ("{root}/abs/subdir/path.txt", "./abs/subdir/path.txt"),
])
def test_project_rel(path: str, expected_path: str):
    root_path = os.getcwd().rstrip("/")
    path = path.format(root=root_path)
    locations = Locations(root_path=root_path)
    rel_path = locations.project_rel(path)

    assert rel_path == expected_path
