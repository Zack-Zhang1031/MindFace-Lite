from __future__ import annotations

from threading import Event, Thread

import pytest

from mindface.realtime.bounded_queue import BoundedDropQueue, QueueStopped, QueueWorkerError


def test_drop_newest_keeps_existing_item_and_counts_drop() -> None:
    queue = BoundedDropQueue[int](maxsize=1, overflow="drop_newest")

    assert queue.put(1)
    assert not queue.put(2)

    assert queue.get() == 1
    assert queue.stats().accepted == 1
    assert queue.stats().dropped == 1


def test_drop_oldest_keeps_latest_item_and_counts_drop() -> None:
    queue = BoundedDropQueue[int](maxsize=1, overflow="drop_oldest")

    assert queue.put(1)
    assert queue.put(2)

    assert queue.get() == 2
    assert queue.stats().accepted == 2
    assert queue.stats().dropped == 1


def test_stop_releases_waiting_consumer() -> None:
    queue = BoundedDropQueue[int](maxsize=1)
    started = Event()
    stopped = Event()

    def consume() -> None:
        started.set()
        with pytest.raises(QueueStopped):
            queue.get()
        stopped.set()

    thread = Thread(target=consume)
    thread.start()
    assert started.wait(timeout=1.0)
    queue.stop()
    thread.join(timeout=1.0)

    assert stopped.is_set()
    assert not thread.is_alive()


def test_worker_failure_is_propagated_to_consumer() -> None:
    queue = BoundedDropQueue[int](maxsize=1)
    cause = ValueError("producer failed")

    queue.fail(cause)

    with pytest.raises(QueueWorkerError, match="producer failed") as captured:
        queue.get()
    assert captured.value.__cause__ is cause


def test_block_policy_times_out_when_queue_remains_full() -> None:
    queue = BoundedDropQueue[int](maxsize=1, overflow="block")
    assert queue.put(1)

    assert not queue.put(2, timeout=0.01)
    assert queue.stats().dropped == 0

