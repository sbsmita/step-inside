"""Several copies of one OpenCV DNN model, for parallel inference.

An OpenCV Net must never run on two threads at once, and a small model (a 100x32 word recognizer) can't keep many
cores busy by itself. A pool of copies solves both: each call borrows a free copy, loading a new one while under the
limit, and waits for one to come back otherwise."""
import os
import queue
import threading
from contextlib import contextmanager


def pool_size(env: str, default: int) -> int:
    return max(1, int(os.environ.get(env, default)))


class NetPool:
    def __init__(self, load, size: int):
        self._load, self._size = load, max(1, size)
        self._free: queue.LifoQueue = queue.LifoQueue()
        self._made = 0
        self._lock = threading.Lock()

    @contextmanager
    def net(self):
        try:
            net = self._free.get_nowait()
        except queue.Empty:
            with self._lock:
                make = self._made < self._size
                self._made += make
            if make:
                try:
                    net = self._load()
                except Exception:
                    with self._lock:
                        self._made -= 1
                    raise
            else:
                net = self._free.get()
        try:
            yield net
        finally:
            self._free.put(net)
