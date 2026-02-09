import time
from unittest import mock, TestCase
from mlflow.entities import RunStatus, Metric, Param

from toolchemy.ai.trackers import MLFlowTracker


class MLFlowTrackerTest(TestCase):
    def setUp(self):
        self.patcher = mock.patch('mlflow.tracking.MlflowClient')
        self.mlflow_client_mock = self.patcher.start()

        self.sut = MLFlowTracker('uri', 'name', with_artifact_logging=True,
                                 tracking_client=self.mlflow_client_mock)
        self.expected_run_id = "111"
        self.mlflow_client_mock.create_run.return_value = mock.Mock(info=mock.Mock(run_id=self.expected_run_id, run_name="main"))
        self.sut.start_run("main")
        # self.sut._active_run_id = self.expected_run_id
        self.sut._get_git_username = lambda: None

    def tearDown(self):
        self.sut.end_run()
        self.patcher.stop()

    @classmethod
    def setUpClass(cls):
        cls.expected_tags = {
            'mlflow.user': "test_user",
            'mlflow.source.name': '/Users/{}/experiment_tracker/venv/bin/pytest',
            'mlflow.source.type': 'LOCAL',
            'mlflow.source.git.commit': '4c1da533076a4db50a1911c2379ce7203462d1ee',
            'mlflow.parentRunId': '222',
        }
        cls.mlflow_resolve_tags_mock = mock.patch("toolchemy.ai.trackers.mlflow_tracker.resolve_tags").start()
        cls.mlflow_resolve_tags_mock.return_value = cls.expected_tags

    @classmethod
    def tearDownClass(cls):
        cls.mlflow_resolve_tags_mock.stop()

    def test_start_run_when_experiment_does_not_exist(self):
        self.mlflow_client_mock.reset_mock()

        expected_experiment_id = 123
        user_specified_tags = {"description": "some description", "additional_info": "some info"}

        self.mlflow_client_mock.get_experiment_by_name.return_value = None
        self.mlflow_client_mock.create_experiment.return_value = expected_experiment_id
        self.mlflow_client_mock.create_run.return_value = mock.Mock(info=mock.Mock(run_id=self.expected_run_id, run_name="main"))

        self.sut.start_run(user_specified_tags=user_specified_tags, run_name="main")

        self.mlflow_client_mock.create_experiment.assert_called_once_with('name')
        self.mlflow_client_mock.create_run.assert_called_once_with(
            experiment_id=expected_experiment_id,
            start_time=None,
            run_name="main",
            tags=self.expected_tags)

        assert self.sut.run_name == "main"

    def test_start_run_when_experiment_exists(self):
        self.mlflow_client_mock.reset_mock()

        expected_experiment_id = 124

        type(self.mlflow_client_mock.get_experiment_by_name.return_value).experiment_id = expected_experiment_id
        self.mlflow_client_mock.create_experiment.return_value = expected_experiment_id
        self.mlflow_client_mock.create_run.return_value = mock.Mock(info=mock.Mock(run_id=self.expected_run_id, run_name="main"))

        self.sut.start_run(run_name="main")

        self.mlflow_client_mock.create_experiment.assert_not_called()
        self.mlflow_client_mock.create_run.assert_called_once_with(
            experiment_id=expected_experiment_id,
            start_time=None,
            run_name="main",
            tags=self.expected_tags)

        assert self.sut.run_name == "main"

    def test_start_run_as_nested_one(self):
        self.mlflow_client_mock.reset_mock()
        expected_experiment_id = 124
        expected_parent_run_id = "222"
        expected_run_id = "333"

        type(self.mlflow_client_mock.get_experiment_by_name.return_value).experiment_id = expected_experiment_id

        self.mlflow_client_mock.create_run.side_effect = [
            mock.Mock(info=mock.Mock(run_id=expected_parent_run_id, run_name="parent")),
            mock.Mock(info=mock.Mock(run_id=expected_run_id, run_name="child"))
        ]

        self.sut.start_run(run_name="parent")
        self.sut.start_run(run_name="child", parent_run_id=expected_parent_run_id)

        self.mlflow_client_mock.create_experiment.assert_not_called()

        expected_calls = [
            mock.call(experiment_id=expected_experiment_id, start_time=None, run_name="parent", tags=self.expected_tags),
            mock.call(experiment_id=expected_experiment_id, start_time=None, run_name="child", tags=self.expected_tags),
        ]

        self.mlflow_client_mock.create_run.assert_has_calls(expected_calls)

        assert self.mlflow_client_mock.create_run.call_count == 2

        self.sut.end_run()
        self.sut.end_run()

    def test_end_run(self):
        status = RunStatus.to_string(RunStatus.FINISHED)
        self.sut.end_run()

        self.mlflow_client_mock.set_terminated.assert_called_once_with(self.expected_run_id, status)

        assert self.sut._active_run is None
        assert self.sut._active_run_id is None

    def test_log_param(self):
        expected_name = 'name'
        expected_value = 'value'

        self.sut.log_param(expected_name, expected_value)

        self.mlflow_client_mock.log_param.assert_called_once_with(self.expected_run_id, expected_name, expected_value)

    def test_log_params(self):
        expected_name = 'name'
        expected_value = 'value'

        params = {
            expected_name: expected_value
        }

        self.sut.log_params(params)

        expected_params = [Param(key, value) for key, value in params.items()]

        args = self.mlflow_client_mock.log_batch.call_args[0]

        self.assertEqual(self.expected_run_id, args[0])
        for i, expected_param in enumerate(expected_params):
            self.assertEqual(expected_param.key, args[2][i].key)
            self.assertEqual(expected_param.value, args[2][i].value)

    def test_log_text(self):
        expected_name = 'name'
        expected_value = 'value'

        self.sut.log_text(expected_name, expected_value)

        self.mlflow_client_mock.log_text.assert_called_once_with(run_id=self.expected_run_id, text=expected_value, artifact_file=expected_name)

    def test_log_metric(self):
        expected_name = 'name'
        expected_value = 6.66
        expected_step = None

        self.sut.log_metric(expected_name, expected_value)

        self.mlflow_client_mock.log_metric.assert_called_once_with(self.expected_run_id, expected_name, expected_value,
                                                                   expected_step)

    def test_log_metrics(self):
        expected_name = 'name'
        expected_value = 6.66
        expected_step = None
        metrics = {
            expected_name: expected_value
        }

        self.sut.log_metrics(metrics, expected_step)

        timestamp = int(time.time() * 1000)
        expected_metrics = [Metric(key, value, timestamp, 0) for key, value in metrics.items()]

        args = self.mlflow_client_mock.log_batch.call_args[0]

        self.assertEqual(self.expected_run_id, args[0])
        for i, expected_metric in enumerate(expected_metrics):
            self.assertEqual(expected_metric.key, args[1][i].key)
            self.assertEqual(expected_metric.value, args[1][i].value)

    def test_set_run_tag(self):
        expected_name = 'tag'
        expected_value = 'some_tag'
        expected_run_name = 'main'

        self.mlflow_client_mock.create_run.return_value = mock.Mock(info=mock.Mock(run_id=self.expected_run_id, run_name=expected_run_name))

        self.sut.start_run(run_name=expected_run_name)

        expected_run_id = self.sut._active_run_id

        self.sut.set_run_tag(expected_name, expected_value)
        self.sut.end_run()

        data = self.sut.get_data()

        expected_data = {
            "metrics": {},
            "params": {},
            "tags": {"experiment": {}, "runs": {
                expected_run_name: {
                    expected_name: expected_value
                }
            }}
        }

        self.assertEqual(expected_data, data)

        self.mlflow_client_mock.set_tag.assert_called_once_with(expected_run_id, expected_name, expected_value)

    def test_set_experiment_tag(self):
        expected_name = 'tag'
        expected_value = 'some_tag'
        expected_experiment_id = self.sut._experiment_id

        self.sut.set_experiment_tag(expected_name, expected_value)

        data = self.sut.get_data()

        expected_data = {
            "metrics": {},
            "params": {},
            "tags": {"experiment": {expected_name: expected_value}, "runs": {}}
        }

        self.assertEqual(expected_data, data)

        self.mlflow_client_mock.set_experiment_tag.assert_called_once_with(expected_experiment_id, expected_name, expected_value)

    def test_log_artifact(self):
        expected_artifact_path = '/some/path'
        expected_save_path='/some/other/path'

        self.sut.log_artifact(expected_artifact_path, expected_save_path)

        self.mlflow_client_mock.log_artifact.assert_called_once_with(
            self.expected_run_id, expected_artifact_path, expected_save_path
        )

    def test_log_figure(self):
        expected_save_path = '/some/path'
        expected_figure = mock.MagicMock()

        self.sut.log_figure(expected_figure, expected_save_path)

        self.mlflow_client_mock.log_figure.assert_called_once_with(
            self.expected_run_id,
            expected_figure,
            expected_save_path
        )

    def test_get_max_metric_value(self):
        self.sut.log_metric('a', 2)
        self.sut.log_metric('a', 4)
        self.sut.log_metric('a', 1)
        self.sut.log_metric('a', 3)

        expected_max_value = {
            'value': 4
        }

        assert expected_max_value == self.sut.get_max_metric_value('a')

    def test_get_min_metric_value(self):
        self.sut.log_metric('a', 2)
        self.sut.log_metric('a', 4)
        self.sut.log_metric('a', 1, metric_metadata={'b': 1, 'c': 2})
        self.sut.log_metric('a', 3)

        expected_min_value = {
            'value': 1,
            'b': 1,
            'c': 2
        }

        assert expected_min_value == self.sut.get_min_metric_value('a')

    def test_get_avg_metric_value(self):
        self.sut.log_metric('a', 2)
        self.sut.log_metric('a', 4)
        self.sut.log_metric('a', 1, metric_metadata={'b': 1, 'c': 2})
        self.sut.log_metric('a', 3)

        expected_avg_value = 2.5

        assert expected_avg_value == self.sut.get_avg_metric_value('a')

    def test_get_data(self):
        self.sut.log_metric('a', 1)
        self.sut.log_param('b', "1")

        expected_data_1 = {
            "metrics": {"a": [{"value": 1}]},
            "params": {"b": "1"},
            "tags": {"experiment" :{}, "runs": {}}
        }

        first_iteration_data = self.sut.get_data()

        assert first_iteration_data == expected_data_1

        self.sut.log_param('b', "2")
        expected_data_2 = {
            "metrics": {"a": [{"value": 1}]},
            "params": {"b": "2"},
            "tags": {"experiment": {}, "runs": {}}
        }
        assert self.sut.get_data() == expected_data_2
        assert first_iteration_data == expected_data_1
