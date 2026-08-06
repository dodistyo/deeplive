import threading


class FrameHolder:
    """Thread-safe 'latest frame wins' holder.

    put() overwrites the previous value (old frames are dropped, never
    queued). get_new(last_seq, timeout) blocks until a frame with a
    sequence number different from last_seq is available, or returns None
    on timeout. seq increments on every put() so identical buffer objects
    still register as 'new'.
    """

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._seq = 0
        self._frame = None
        self._dropped = 0

    def put(self, frame) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._seq += 1
            self._frame = frame
            self._lock.notify_all()

    def get(self):
        """Return the latest frame or None if nothing has been put yet."""
        with self._lock:
            return self._frame

    def get_new(self, last_seq: int | None, timeout: float | None = None):
        """Block until a frame newer than last_seq exists.

        last_seq=None means 'first frame ever' (waits for any frame).
        Returns (frame, seq), or (None, last_seq) on timeout.
        """
        with self._lock:
            if last_seq is None:
                while self._frame is None:
                    if not self._lock.wait(timeout):
                        return None, last_seq
            else:
                while self._frame is None or self._seq == last_seq:
                    if not self._lock.wait(timeout):
                        return None, last_seq
            return self._frame, self._seq

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
