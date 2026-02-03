from typing import Dict, Optional, Any
from neptune_scale import Run

from toolchemy.ai.trackers.common import TrackerBase
from toolchemy.utils.datestimes import current_datetime_str


class NeptuneAITracker(TrackerBase):
    def __init__(self, project_name: str, experiment_name: str, api_token: str, with_artifact_logging: bool = True,
                 disabled: bool = False):
        super().__init__(experiment_name=experiment_name, with_artifact_logging=with_artifact_logging, disabled=disabled)
        self._api_token = api_token
        self._project_name = project_name
        self._active_run: Run | None = None
        self._active_run_id: str | None = None

    @property
    def run_name(self) -> str:
        if not self._active_run or not self._active_run_id:
            raise RuntimeError("There is no active run!")
        return self._active_run_id

    @property
    def run_id(self) -> str:
        return self.run_name


    def start_run(
            self, run_id: str = None,
            run_name: str = None,
            parent_run_id: str = None,
            user_specified_tags: Dict[str, str] = None
    ):
        if self._disabled:
            return
        if self._active_run or self._active_run_id:
            raise RuntimeError(f"Cannot start a new run, there is already an active run")
        if run_name is not None:
            self._logger.warning(f"Neptune tracker uses 'run_id' as the run name. Use 'run_id' for the custom run name.")
            if run_id is None:
                run_id = run_name

        if run_id is None:
            run_id = self._generate_run_name()

        self._active_run_id = run_id

        self._active_run = Run(
            project=self._project_name,
            api_token=self._api_token,
            experiment_name=self._experiment_name,
            run_id=self._active_run_id,
            enable_console_log_capture=True,
        )
        self._logger.info(f"Neptune tracking run started. Experiment name: {self.experiment_name}")

    def end_run(self):
        if self._disabled:
            return
        if self._active_run is None:
            raise ValueError(f"No active run to stop")
        self._active_run.close()
        self._active_run = None
        self._active_run_id = None

    def log(self, name: str, value: Any):
        if self._disabled:
            return
        raise NotImplemented()

    def log_param(self, name: str, val, step: Optional[int] = None):
        if self._disabled:
            return
        self._store_param(name, val)
        self._active_run.log_configs({name: val})

    def log_params(self, params: Dict[str, Any]):
        if self._disabled:
            return
        for param_name, param_value in params.items():
            self._store_param(param_name, param_value)
        self._active_run.log_configs(params)

    def log_text(self, name: str, value: str):
        if self._disabled:
            return
        self._active_run.log_configs({name: value})

    def log_metric(self, name: str, value: float, step: int | None = None, metric_metadata: dict | None = None):
        if self._disabled:
            return
        self.log_metrics({name: value}, step)

    def log_metrics(self, metrics: Dict[str, float | list], step: Optional[int] = None):
        if self._disabled:
            return

        for k, v in metrics.items():
            metric_value = self._store_metric(k, v)
            metrics[k] = metric_value

        self._active_run.log_metrics(metrics, step=step)

    def log_artifact(self, artifact_path: str, save_dir: str = None):
        if self._disabled:
            return
        if not self._artifact_logging:
            return
        if save_dir is None:
            save_dir = ""

        self._active_run.assign_files({save_dir: artifact_path})

    def log_figure(self, figure, save_path: str):
        if self._disabled:
            return
        if save_path is None:
            save_path = ""

        self._active_run.assign_files({save_path: figure})

    def set_run_tag(self, name: str, value: str | int | float):
        self._store_tag(name, value, run_name=self.run_name)
        self._active_run.add_tags([f"{name}__{value}"])

    def set_experiment_tag(self, name: str, value: str | int | float):
        self._store_tag(name, value)
        if self._active_run:
            self.set_run_tag(name, value)

    def get_id(self):
        if self._disabled:
            return ""
        if not self._active_run:
            raise ValueError(f"No active run")
        return self._active_run

    def _generate_run_name(self) -> str:
        return f"{self.experiment_name}__{current_datetime_str('%Y_%m_%d__%H_%M_%S')}"
