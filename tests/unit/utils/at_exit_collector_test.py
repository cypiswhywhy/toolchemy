import logging
import pytest
import unittest
from unittest.mock import Mock, patch, call
from toolchemy.utils.at_exit_collector import ICollectable, AtExitCollector


class MockCollectable(ICollectable):
    def __init__(self, name: str, data: dict):
        self._name = name
        self._data = data

    def label(self) -> str:
        return self._name

    def collect(self) -> dict:
        return {**self._data}


class TestAtExitCollector(unittest.TestCase):
    def setUp(self):
        AtExitCollector.enable()
        AtExitCollector.reset()

    def tearDown(self):
        AtExitCollector.disable()

    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_register_collectable(self, mock_atexit_register):
        collectable = MockCollectable("test", {"count": 5})

        AtExitCollector.register(collectable)

        assert len(AtExitCollector._collectables) == 1
        assert AtExitCollector._collectables[0] == collectable

    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_register_multiple_collectables(self, mock_atexit_register):
        collectable1 = MockCollectable("test1", {"count": 5})
        collectable2 = MockCollectable("test2", {"count": 10})

        AtExitCollector.register(collectable1)
        AtExitCollector.register(collectable2)

        assert len(AtExitCollector._collectables) == 2


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_atexit_reset(self, mock_atexit_register):
        AtExitCollector.register(MockCollectable("test1", {"count": 5}))
        AtExitCollector.reset()
        assert len(AtExitCollector._collectables) == 0


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_collector_summary_single_collectable(self, mock_atexit_register):
        collectable = MockCollectable("test_service", {"requests": 10, "errors": 2})
        AtExitCollector.register(collectable)

        with patch.object(AtExitCollector._collector_logger, "info") as mock_info:
            AtExitCollector._collector_summary()

            first_call_args = mock_info.call_args_list[1][0][0]
            assert "test_service" in first_call_args
            assert "requests" in first_call_args

            second_call_args = mock_info.call_args_list[2][0][0]
            assert "aggregated summary" in second_call_args


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_collector_summary_multiple_same_name(self, mock_atexit_register):
        collectable1 = MockCollectable("api_client", {"requests": 10, "errors": 2})
        collectable2 = MockCollectable("api_client", {"requests": 15, "errors": 1})
        AtExitCollector.register(collectable1)
        AtExitCollector.register(collectable2)

        with patch.object(AtExitCollector._collector_logger, "info") as mock_info:
            AtExitCollector._collector_summary()

            aggregated_call = mock_info.call_args_list[-2][0][0]
            assert "api_client" in aggregated_call
            assert "instances" in aggregated_call


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_collector_summary_aggregation(self, mock_atexit_register):
        collectable1 = MockCollectable("service", {"calls": 10, "time": 5.0})
        collectable2 = MockCollectable("service", {"calls": 20, "time": 10.0})
        AtExitCollector.register(collectable1)
        AtExitCollector.register(collectable2)

        with patch.object(AtExitCollector._collector_logger, "info"):
            AtExitCollector._collector_summary()


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_collector_summary_with_non_numeric_values(self, mock_atexit_register):
        collectable = MockCollectable("service", {
            "count": 10,
            "message": "test message",
            "status": "active"
        })
        AtExitCollector.register(collectable)

        with patch.object(AtExitCollector._collector_logger, "info") as mock_info:
            AtExitCollector._collector_summary()

            assert mock_info.call_count == 4


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_collector_summary_calculates_averages(self, mock_atexit_register):
        collectable1 = MockCollectable("calc", {"value": 10})
        collectable2 = MockCollectable("calc", {"value": 20})
        collectable3 = MockCollectable("calc", {"value": 30})
        AtExitCollector.register(collectable1)
        AtExitCollector.register(collectable2)
        AtExitCollector.register(collectable3)

        with patch.object(AtExitCollector._collector_logger, "info") as mock_info:
            AtExitCollector._collector_summary()

            aggregated_call = mock_info.call_args_list[-2][0][0]
            assert "avg_value" in aggregated_call


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_update_dict_simple(self, mock_atexit_register):
        d = {"a": 1, "b": 2}
        u = {"b": 3, "c": 4}

        result = AtExitCollector._update_dict(d, u)

        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_update_dict_nested(self, mock_atexit_register):
        d = {"outer": {"inner": 1}}
        u = {"outer": {"inner": 2, "new": 3}}

        result = AtExitCollector._update_dict(d, u)

        assert result["outer"]["inner"] == 2
        assert result["outer"]["new"] == 3


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_update_dict_empty_dict(self, mock_atexit_register):
        d = {}
        u = {"a": 1}

        result = AtExitCollector._update_dict(d, u)

        assert result["a"] == 1


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_multiple_different_collectables(self, mock_atexit_register):
        collectable1 = MockCollectable("service_a", {"metric1": 100})
        collectable2 = MockCollectable("service_b", {"metric2": 200})
        collectable3 = MockCollectable("service_a", {"metric1": 50})

        AtExitCollector.register(collectable1)
        AtExitCollector.register(collectable2)
        AtExitCollector.register(collectable3)

        with patch.object(AtExitCollector._collector_logger, "info") as mock_info:
            AtExitCollector._collector_summary()

            assert mock_info.call_count == 6


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_collector_summary_with_float_values(self, mock_atexit_register):
        collectable1 = MockCollectable("timer", {"duration": 1.5, "count": 3})
        collectable2 = MockCollectable("timer", {"duration": 2.5, "count": 2})

        AtExitCollector.register(collectable1)
        AtExitCollector.register(collectable2)

        with patch.object(AtExitCollector._collector_logger, "info") as mock_info:
            AtExitCollector._collector_summary()

            aggregated_call = mock_info.call_args_list[-2][0][0]
            assert "avg_duration" in aggregated_call
            assert "avg_count" in aggregated_call


    @patch("toolchemy.utils.at_exit_collector.atexit.register")
    def test_collector_summary_empty_collectables(self, mock_atexit_register):
        with patch.object(AtExitCollector._collector_logger, "info") as mock_info:
            AtExitCollector._collector_summary()

            assert  mock_info.call_count == 1
            info_msg = mock_info.call_args_list[0][0][0]

            assert info_msg == "No collectable registered, skipping AtExitCollector summary."
