from abc import ABC, abstractmethod
import statistics
from typing import Dict, Any

from toolchemy.utils.logger import get_logger


class ITracker(ABC):
    @property
    @abstractmethod
    def experiment_name(self) -> str:
        pass

    @property
    @abstractmethod
    def experiment_id(self) -> str:
        pass

    @property
    @abstractmethod
    def run_name(self) -> str:
        pass

    @property
    @abstractmethod
    def run_id(self) -> str:
        pass

    @abstractmethod
    def start_run(
            self, run_id: str = None,
            run_name: str = None,
            parent_run_id: str = None,
            user_specified_tags: Dict[str, str] = None
    ):
        pass

    @abstractmethod
    def end_run(self):
        pass

    @abstractmethod
    def log(self, name: str, value: Any):
        pass

    @abstractmethod
    def log_param(self, name: str, value: Any):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int | None = None, metric_metadata: dict | None = None):
        pass

    @abstractmethod
    def log_text(self, name: str, value: str):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float | list], step: int | None = None):
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: str, save_dir: str = None):
        pass

    @abstractmethod
    def log_figure(self, figure, save_path: str):
        """
        Parameters:
             figure (matplotlib.figure.Figure, _matplotlib.figure.Figure, _plotly.graph_objects.Figure): plot figure
             save_path (str): the run-relative artifact file path in posixpath format to which
                              the figure is saved (e.g. "dir/file.png").
        """
        pass

    @abstractmethod
    def set_run_tag(self, name: str, value: str | int | float):
        pass

    @abstractmethod
    def set_experiment_tag(self, name: str, value: str | int | float):
        pass

    @abstractmethod
    def disable(self):
        pass

    @abstractmethod
    def get_data(self) -> dict:
        pass

    @abstractmethod
    def get_traces(self, filter_name: str | None = None):
        pass


class TrackerBase(ITracker, ABC):
    def __init__(self, experiment_name: str, with_artifact_logging: bool = True, disabled: bool = False):
        self._disabled = disabled
        self._logger = get_logger()
        self._experiment_name = experiment_name
        self._artifact_logging = with_artifact_logging
        self._metrics = {}
        self._params = {}
        self._tags = {
            "experiment": {},
            "runs": {},
        }

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def experiment_id(self) -> str:
        return self._experiment_name

    def get_max_metric_value(self, name: str) -> float:
        return max(self._metrics[name], key=lambda el: el['value'])

    def get_min_metric_value(self, name: str) -> float:
        return min(self._metrics[name], key=lambda el: el['value'])

    def get_avg_metric_value(self, name: str) -> float:
        metric_values = [m['value'] for m in self._metrics[name]]
        return statistics.mean(metric_values)

    def get_data(self) -> dict:
        return {
            "metrics": self._metrics.copy(),
            "params": self._params.copy(),
            "tags": self._tags.copy(),
        }

    def get_traces(self, filter_name: str | None = None):
        raise NotImplementedError()

    def _store_param(self, name: str, value: Any):
        if self._disabled:
            raise RuntimeError(f"Disabled trackers cannot store params!")

        self._params[name] = value

    def _store_tag(self, name: str, value: str | int | float, run_name: str | None = None):
        if run_name is None:
            self._tags["experiment"][name] = value
            return
        if run_name not in self._tags["runs"]:
            self._tags["runs"][run_name] = {}
        self._tags["runs"][run_name][name] = value

    def _store_metric(self, name: str, value: float, metric_metadata: dict | None = None) -> float:
        if self._disabled:
            raise RuntimeError(f"Disabled trackers cannot store metrics!")
        if name not in self._metrics:
            self._metrics[name] = []

        if isinstance(value, dict):
            new_entry = value
        else:
            new_entry = {
                'value': value
            }

        if metric_metadata:
            new_entry.update(metric_metadata)

        self._metrics[name] += [new_entry]

        return new_entry['value']

    def disable(self):
        self._disabled = True


class InMemoryTracker(TrackerBase):
    def __init__(self, experiment_name: str = "dummy", disabled: bool = False):
        super().__init__(experiment_name=experiment_name, disabled=disabled)
        self._run_name = None
        self._reset()

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def run_id(self) -> str:
        return self._run_name

    def start_run(
            self, run_id: str = None,
            run_name: str = None,
            parent_run_id: str = None,
            user_specified_tags: Dict[str, str] = None
    ):
        self._run_name = run_name

    def _reset(self):
        self._run_name = None
        self._data = {}
        self._params = {}
        self._metrics = {}

    def end_run(self):
        self._reset()

    def log(self, name: str, value: Any):
        self._data[name] = value

    def log_param(self, name: str, value: Any):
        self._params[name] = value

    def log_params(self, params: Dict[str, Any]):
        for name, value in params.items():
            self.log_param(name, value)

    def log_text(self, name: str, value: str):
        self.log(name, value)

    def log_metric(self, name: str, value: float, step: int | None = None, metric_metadata: dict | None = None):
        self._metrics[name] = value

    def log_metrics(self, metrics: Dict[str, float | list], step: int | None = None):
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, artifact_path: str, save_dir: str = None):
        raise NotImplementedError()

    def log_figure(self, figure, save_path: str):
        raise NotImplementedError()

    def set_run_tag(self, name: str, value: str | int | float):
        self._store_tag(name, value, run_name=self.run_name)

    def set_experiment_tag(self, name: str, value: str | int | float):
        self._store_tag(name, value)
