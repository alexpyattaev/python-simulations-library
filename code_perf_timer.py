import time


class Context_Timer:
    def __init__(self):
        self._start_time = 0
        self.total_time_ns = 0

    def __enter__(self):
        """Start a new timer as a context manager"""
        self._start_time = time.time_ns()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        assert self._start_time != 0
        self.total_time_ns += time.time_ns() - self._start_time
        self._start_time = 0

    @property
    def seconds(self) -> float:
        return self.total_time_ns * 1e-9

