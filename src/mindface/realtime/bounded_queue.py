from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition
from time import monotonic
from typing import Generic, Literal, TypeVar


T = TypeVar("T")
OverflowPolicy = Literal["block", "drop_oldest", "drop_newest"]


class QueueStopped(RuntimeError):
    """Raised when no more queue items can be produced or consumed."""


class QueueWorkerError(RuntimeError):
    """Raised in another pipeline stage when a worker has failed."""


@dataclass(frozen=True, slots=True)
class QueueStats:
    accepted: int
    dropped: int
    size: int
    stopped: bool


class BoundedDropQueue(Generic[T]):
    def __init__(self, maxsize: int, overflow: OverflowPolicy = "block") -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        if overflow not in {"block", "drop_oldest", "drop_newest"}:
            raise ValueError(f"Unsupported overflow policy: {overflow}")
        self.maxsize = maxsize
        self.overflow = overflow
        self._items: deque[T] = deque()
        self._condition = Condition()
        self._stopped = False
        self._error: BaseException | None = None
        self._accepted = 0
        self._dropped = 0

    def _raise_if_unavailable(self) -> None:
        if self._error is not None:
            raise QueueWorkerError(str(self._error)) from self._error
        if self._stopped:
            raise QueueStopped("queue is stopped")

    def put(self, item: T, timeout: float | None = None) -> bool:
        with self._condition:
            self._raise_if_unavailable()
            if len(self._items) >= self.maxsize:
                if self.overflow == "drop_newest":
                    self._dropped += 1
                    return False
                if self.overflow == "drop_oldest":
                    self._items.popleft()
                    self._dropped += 1
                else:
                    deadline = None if timeout is None else monotonic() + timeout
                    while len(self._items) >= self.maxsize:
                        self._raise_if_unavailable()
                        remaining = None if deadline is None else deadline - monotonic()
                        if remaining is not None and remaining <= 0:
                            return False
                        self._condition.wait(remaining)
            self._items.append(item)
            self._accepted += 1
            self._condition.notify_all()
            return True

    def get(self, timeout: float | None = None) -> T:
        with self._condition:
            deadline = None if timeout is None else monotonic() + timeout
            while not self._items:
                if self._error is not None:
                    raise QueueWorkerError(str(self._error)) from self._error
                if self._stopped:
                    raise QueueStopped("queue is stopped")
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("queue get timed out")
                self._condition.wait(remaining)
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
            self._stopped = True
            self._condition.notify_all()

    def stats(self) -> QueueStats:
        with self._condition:
            return QueueStats(self._accepted, self._dropped, len(self._items), self._stopped)

