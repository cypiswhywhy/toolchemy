import time
from typing import Dict, Optional, Any

import mlflow
mlflow.autolog(disable=True)
from mlflow.client import MlflowClient
from mlflow.entities import RunStatus, Metric, Param
from mlflow.tracking.context.registry import resolve_tags
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from mlflow import MlflowException

from toolchemy.ai.trackers.common import TrackerBase
from toolchemy.utils.logger import get_logger


class MLFlowTracker(TrackerBase):
    def __init__(self, tracking_uri: str, experiment_name: str, with_artifact_logging=True, registry_uri: str | None = None,
                 tracking_client: MlflowClient | None = None):
        super().__init__(experiment_name, with_artifact_logging)
        self._client = None
        self._active_run = None
        self._active_run_id = None
        self._experiment_id = None
        self._reset_run()

        if tracking_client:
            self._client = tracking_client
        else:
            self._client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

        logger = get_logger()
        logger.info(f"Mlflow tracker created")
        logger.info(f"> tracking uri: {tracking_uri}")
        logger.info(f"> artifact logging: {self._artifact_logging}")

    @property
    def run_name(self) -> str:
        if not self._active_run:
            raise RuntimeError("There is no active run!")
        return self._active_run.info.run_name

    @property
    def run_id(self) -> str:
        if not self._active_run:
            raise RuntimeError("There is no active run!")
        return self._active_run.info.run_id

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def start_run(
            self, run_id: str = None,
            run_name: str = None,
            parent_run_id: str = None,
            user_specified_tags: Dict[str, str] = None
    ):
        """
        Starts the run

        :param run_id: If specified, get the run with the specified ID and log parameters and metrics under that run.
        :param run_name: Name of new run. Used only when run_id is unspecified.
        :param parent_run_id: If specified: current run will be nested into parent_run_id
        :param user_specified_tags: dict of with custom tags
        """
        if self._disabled:
            return

        if user_specified_tags is None:
            user_specified_tags = {}
        if parent_run_id is not None:
            user_specified_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        if run_name:
            user_specified_tags[MLFLOW_RUN_NAME] = run_name

        tags = resolve_tags(user_specified_tags)

        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if experiment:
            experiment_comment_msg = "(already exists)"
            self._logger.debug(f"Experiment '{self._experiment_name}' already exists")
            self._experiment_id = experiment.experiment_id
            if experiment.lifecycle_stage == "deleted":
                self._logger.info(f"Restoring deleted experiment")
                self._client.restore_experiment(self._experiment_id)
        else:
            experiment_comment_msg = "(does not exist, creating a new one)"
            self._logger.debug(f"Experiment '{self._experiment_name}' does not exist, creating a new one")
            self._experiment_id = self._client.create_experiment(self._experiment_name)

        self._logger.info(f"Starting the experiment tracking")
        self._logger.info(f"> experiment name: {self._experiment_name} {experiment_comment_msg}")
        self._logger.info(f"> experiment id: {self._experiment_id}")
        self._logger.info(f"> run name: {run_name}")

        self._active_run = self._client.create_run(
            experiment_id=self._experiment_id,
            start_time=None,
            run_name=run_name,
            tags=tags)

        self._active_run_id = self._active_run.info.run_id

    def end_run(self):
        if self._disabled:
            return

        status = RunStatus.to_string(RunStatus.FINISHED)
        self._client.set_terminated(self._active_run_id, status)
        self._reset_run()

    def log(self, name: str, value: Any):
        if self._disabled:
            return

        if isinstance(value, dict):
            self._client.log_dict(self._active_run_id, value, f"{name}.json")

        raise ValueError(f"Unsupported logged object type: {type(value)}")

    def log_param(self, name: str, value: Any):
        if self._disabled:
            return

        self._store_param(name, value)
        self._client.log_param(self._active_run_id, name, value)

    def log_params(self, params: Dict[str, Any]):
        if self._disabled:
            return

        params_to_store = []
        for key, value in params.items():
            if isinstance(value, list):
                for v in value:
                    params_to_store.append(Param(key, str(v)))
            else:
                params_to_store.append(Param(key, str(value)))
            self._store_param(key, value)

        self._client.log_batch(self._active_run_id, [], params_to_store)

    def log_text(self, name: str, value: str):
        if self._disabled:
            return
        try:
            self._client.log_text(run_id=self._active_run_id, text=value, artifact_file=name)
        except MlflowException as e:
            self._logger.error(f"An error occurred during text logging: {e}")
            self._logger.error(f"> tracking uri: {self._client.tracking_uri}")
            self._logger.error(f"> artifact uri: {self._active_run.info.artifact_uri}")
            raise e

    def log_metric(self, name: str, value: float, step: int | None = None, metric_metadata: dict | None = None):
        if self._disabled:
            return

        metric_value = self._store_metric(name, value, metric_metadata)
        self._client.log_metric(self._active_run_id, name, metric_value, step)

    def log_metrics(self, metrics: Dict[str, float | list], step: Optional[int] = None):
        if self._disabled:
            return

        metrics_to_store = []
        timestamp = int(time.time() * 1000)
        for k, value in metrics.items():
            if isinstance(value, list):
                for v in value:
                    metric_value = self._store_metric(k, v)
                    metrics_to_store.append(Metric(k, metric_value, timestamp, step or 0))
            else:
                metric_value = self._store_metric(k, value)
                metrics_to_store.append(Metric(k, metric_value, timestamp, step or 0))

        self._client.log_batch(self._active_run_id, metrics_to_store)

    def log_artifact(self, artifact_path: str, save_dir: str = None):
        if self._disabled:
            return

        if self._artifact_logging:
            self._client.log_artifact(self._active_run_id, artifact_path, save_dir)

    def log_figure(self, figure, save_path: str):
        if self._disabled:
            return
        self._client.log_figure(self._active_run_id, figure, save_path)

    def set_run_tag(self, name: str, value: str | int | float):
        self._store_tag(name, value, run_name=self.run_name)
        self._client.set_tag(self._active_run_id, name, value)

    def set_experiment_tag(self, name: str, value: str | int | float):
        self._store_tag(name, value)
        self._client.set_experiment_tag(self._experiment_id, name, value)

    def _reset_run(self):
        self._active_run = None
        self._active_run_id = None


def play():
    from toolchemy.utils.datestimes import current_datetime_str
    from toolchemy.ai.prompter import PrompterMLflow
    from toolchemy.utils import Locations

    locations = Locations()
    tracker = MLFlowTracker("http://hal:5000", f"test-{current_datetime_str()}")
    tracker.start_run()
    prompter = PrompterMLflow(locations.in_resources("tests/prompts_mlflow"))
    print(prompter.render("test_prompt", foo="foo1", bar="bar1"))
    tracker.log_param("param1", "param1value")
    tracker.log_param("param2", 2)
    tracker.log_metric("metric1", 1.0)
    tracker.log_metric("metric2", 2.0, metric_metadata={"info": "metric 2 metadata"})
    tracker.log_text("text_test", "some longer piece of text")
    print(prompter.render("test_prompt", foo="foo1", bar="bar1"))
    tracker.end_run()
    print(prompter.render("test_prompt", foo="foo1", bar="bar1"))


if __name__ == "__main__":
    play()
