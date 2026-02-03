from abc import ABC, abstractmethod
import atexit
import collections

from toolchemy.utils.logger import get_logger
from toolchemy.utils.utils import pp


class ICollectable(ABC):
    FIELD_NAME_KEY = "name"

    @abstractmethod
    def label(self) -> str:
        pass

    @abstractmethod
    def collect(self) -> dict:
        pass


class AtExitCollector:
    _collectables: list[ICollectable] = []
    _collector_logger = get_logger()
    _is_enabled = False
    _is_registered = False

    @classmethod
    def enable(cls):
        cls._is_enabled = True

    @classmethod
    def disable(cls):
        cls._is_enabled = False

    @classmethod
    def reset(cls):
        cls._collectables = []
        atexit.unregister(cls._collector_summary)
        cls._is_registered = False

    @classmethod
    def register(cls, collectable: ICollectable):
        if not cls._is_enabled:
            return
        if not cls._is_registered:
            atexit.register(cls._collector_summary)
            cls._collector_logger.info("AtExitCollector registered.")
            cls._is_registered = True
        cls._collector_logger.info(f"Registering collectable: {collectable} (type: {type(collectable)})")
        assert isinstance(collectable, ICollectable), f"Expected ICollectable, got {type(collectable)}"
        cls._collectables.append(collectable)
        for c in cls._collectables:
            cls._collector_logger.info(f"Registered collectable: {c} (type: {c})")

    @classmethod
    def _collector_summary(cls) -> None:
        if not cls._is_enabled:
            return
        if len(cls._collectables) == 0:
            cls._collector_logger.info(f"No collectable registered, skipping AtExitCollector summary.")
            return
        cls._collector_logger.info("AtExitCollector| generating summary...")
        aggregated = {}
        for collectable in cls._collectables:
            try:
                data = collectable.collect()
            except TypeError as e:
                cls._collector_logger.error(f"Collectable of wrong type: {type(collectable)} (is ICollectable: {isinstance(collectable, ICollectable)}): {collectable} (err msg: {e})")
                raise e

            name = collectable.label()
            if name not in aggregated:
                aggregated[name] = {
                    "instances": 0,
                }
            for k, v in data.items():
                if not isinstance(v, (int, float)):
                    continue
                if k not in aggregated[name]:
                    aggregated[name][k] = 0
                aggregated[name][k] += v
            aggregated[name]["instances"] += 1

            cls._collector_logger.info(f"AtExitCollector| summary for {name}:\n{pp(data, print_msg=False)}")

        averages = {}
        for instance_name, instance_data in aggregated.items():
            if instance_name not in averages:
                averages[instance_name] = {}
            for k, v in instance_data.items():
                if k == "instances":
                    continue

                avg_key = f"avg_{k}"
                averages[instance_name][avg_key] = v / instance_data["instances"]

        cls._update_dict(aggregated, averages)

        cls._collector_logger.info(f"AtExitCollector| aggregated summary:\n{pp(aggregated, print_msg=False)}")
        cls._collector_logger.info("AtExitCollector| summary generation DONE")

    @classmethod
    def _update_dict(cls, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = cls._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
