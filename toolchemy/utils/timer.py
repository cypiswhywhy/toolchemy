import time


class Timer:
    def __init__(self, started: bool = True):
        self._started = started
        self._started_time = None
        self._last_tap_time = None
        self.reset()

    def reset(self):
        self._started_time = time.time()
        self._last_tap_time = time.time()

    def tap(self, since_last: bool = False) -> float:
        now_time = time.time()
        since_time = self._started_time
        if since_last and self._last_tap_time:
            since_time = self._last_tap_time

        self._last_tap_time = now_time

        return now_time - since_time
