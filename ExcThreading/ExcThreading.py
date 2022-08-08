import threading
from typing import Any, Optional, Callable, Iterable

# Modified Threading class to raise exception in parent thread if any child thread crashes


class ExcThreading(threading.Thread):
    def __init__(
            self,
            target: Optional[Callable] = None,
            args: Optional[Iterable] = (),
            daemon: Optional[bool] = None,
            *init_args,
            **kwargs
    ):
        self._target = None
        self._args = None
        self._kwargs = None
        self._started: Optional[threading.Event] = None
        super().__init__(target=target, args=args, daemon=daemon, *init_args, **kwargs)
        self._exception = None
        self._output = None

    @property
    def started(self) -> bool:
        return self._started.is_set()

    def start(self):
        if not self._started.is_set():
            super().start()

    def run(self):
        self._exception = None
        try:
            self._output = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self._exception = e

    def join(self, timeout=None, exc=True) -> Any:
        super(ExcThreading, self).join(timeout=timeout)
        if self._exception and exc:
            raise self._exception
        return self._output

    def raise_error(self):
        if not self.started:
            raise RuntimeError("cannot run raise_error before the thread has started")
        if self.is_alive():
            self.join(exc=True)
        elif self._exception:
            raise self._exception
        return False
