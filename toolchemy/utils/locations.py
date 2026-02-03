import inspect
import os
import json
from pathlib import Path


class PathDoesNotExistError(Exception):
    pass


def _find_project_root(path: Path) -> Path:
    for parent in [path] + list(path.parents):
        if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
            return parent
    return path


def get_external_caller_path(exclude_prefixes=None) -> str:
    if exclude_prefixes is None:
        exclude_prefixes = [str(Path(__file__).resolve().parent)]

    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if not module or not hasattr(module, '__file__'):
            continue

        path = Path(module.__file__).resolve()
        if all(not str(path).startswith(prefix) for prefix in exclude_prefixes):
            project_root_path = str(_find_project_root(path.parents[1]))
            if "site-packages" in project_root_path:
                return str(Path.cwd())
            return project_root_path

    if exclude_prefixes and len(exclude_prefixes) > 0:
        return str(exclude_prefixes[0])
    raise RuntimeError("Could not find external caller")


class Locations:
    def __init__(self, prefix_dirs: dict | None = None, root_path: str | None = None, objective_path_mode: bool = False) -> None:
        if root_path is None:
            root_path = get_external_caller_path()
        self._dirs = {
            "root": root_path.rstrip("/"),
            "resources": os.path.join(root_path, "resources").rstrip("/"),
            "data": os.path.join(root_path, "data").rstrip("/"),
            "logs": os.path.join(root_path, "logs").rstrip("/"),
        }
        self._prefix_dirs = prefix_dirs
        self._objective_path_mode = objective_path_mode

    @property
    def root(self) -> str:
        ret = self._dirs["root"]
        if self._objective_path_mode:
            ret = Path(ret)
        return ret

    @staticmethod
    def read_content(path: str | Path) -> str:
        if isinstance(path, str):
            path = Path(path)
        return path.read_text()

    @staticmethod
    def read_json(path: str | Path) -> dict | list:
        content = Locations.read_content(path)
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"{e} (Failed to parse JSON from {path}: {e}. Parsed content:\n{content})", e.doc, e.pos) from e
        return json_data

    @staticmethod
    def save_json(data: dict | list, path: str | Path):
        with open(str(path), "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def abs(path: str | Path) -> str:
        return os.path.abspath(os.path.expanduser(str(path)))

    def project_rel(self, path: str | Path) -> str | Path:
        abs_path = str(self.abs(path))
        rel_path = abs_path.replace(self.root, ".")
        if self._objective_path_mode:
            rel_path = Path(rel_path)
        return rel_path

    def in_root(self, elements: str | list[str] | None = None) -> str | Path:
        return self.in_("root", elements)

    def in_resources(self, elements: str | list[str] | None = None) -> str | Path:
        return self.in_("resources", elements)

    def in_data(self, elements: str | list[str] | None = None) -> str | Path:
        return self.in_("data", elements)

    def in_(self, base_dir: str, elements: str | list[str] | None = None) -> str | Path:
        if isinstance(elements, str):
            elements = [elements]
        if elements is None:
            elements = []
        if base_dir not in self._dirs:
            raise ValueError(f"There is no '{base_dir}' dir defined")
        if self._prefix_dirs and base_dir in self._prefix_dirs:
            elements = [self._prefix_dirs[base_dir]] + elements
        ret = str(os.path.join(self._dirs[base_dir], *elements)).rstrip("/")
        if self._objective_path_mode:
            ret = Path(ret)
        return ret
