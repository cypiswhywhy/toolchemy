from typing import Dict, Any

from toolchemy.ai.trackers.common import TrackerBase


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
