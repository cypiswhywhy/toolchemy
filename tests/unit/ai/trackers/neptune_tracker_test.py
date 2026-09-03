import pytest

from toolchemy.ai.trackers.neptune_tracker import NeptuneAITracker


@pytest.fixture
def tracker() -> NeptuneAITracker:
    return NeptuneAITracker(project_name="workspace/project", experiment_name="experiment", api_token="token")


def test_log_reports_itself_as_not_implemented(tracker):
    with pytest.raises(NotImplementedError):
        tracker.log("name", 1)


def test_log_is_a_no_op_while_the_tracker_is_disabled():
    tracker = NeptuneAITracker(project_name="workspace/project", experiment_name="experiment",
                               api_token="token", disabled=True)

    assert tracker.log("name", 1) is None


def test_end_run_without_an_active_run_raises(tracker):
    with pytest.raises(ValueError, match="No active run to stop"):
        tracker.end_run()


def test_run_name_without_an_active_run_raises(tracker):
    with pytest.raises(RuntimeError, match="There is no active run!"):
        _ = tracker.run_name
