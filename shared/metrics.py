import threading
import time
from dataclasses import dataclass

_EMA_ALPHA = 0.3


@dataclass
class MetricsSnapshot:
    frames_in: int = 0
    frames_out: int = 0
    dropped: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    network_rtt_ms: float = 0.0
    queue_size: int = 0  # 0 or 1 by design (newest-wins)
    uptime_s: float = 0.0


class Metrics:
    """Thread-safe metric collector. Durations are EMA-smoothed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._frames_in = 0
        self._frames_out = 0
        self._dropped = 0
        self._fps_window_start = self._start
        self._fps_window_frames = 0
        self._fps = 0.0
        self._inference_ms = 0.0
        self._encode_ms = 0.0
        self._decode_ms = 0.0
        self._rtt_ms = 0.0
        self._queue_size = 0

    def _ema(self, old: float, sample: float) -> float:
        return old + _EMA_ALPHA * (sample - old) if old else sample

    def record_incoming(self, queue_size: int = 0) -> None:
        with self._lock:
            self._frames_in += 1
            self._queue_size = queue_size
            now = time.monotonic()
            self._fps_window_frames += 1
            if now - self._fps_window_start >= 1.0:
                self._fps = self._fps_window_frames / (now - self._fps_window_start)
                self._fps_window_start = now
                self._fps_window_frames = 0

    def record_outgoing(self) -> None:
        with self._lock:
            self._frames_out += 1

    def record_dropped(self, n: int = 1) -> None:
        with self._lock:
            self._dropped += n

    def record_inference_ms(self, ms: float) -> None:
        with self._lock:
            self._inference_ms = self._ema(self._inference_ms, ms)

    def record_encode_ms(self, ms: float) -> None:
        with self._lock:
            self._encode_ms = self._ema(self._encode_ms, ms)

    def record_decode_ms(self, ms: float) -> None:
        with self._lock:
            self._decode_ms = self._ema(self._decode_ms, ms)

    def record_rtt_ms(self, ms: float) -> None:
        with self._lock:
            self._rtt_ms = self._ema(self._rtt_ms, ms)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                frames_in=self._frames_in,
                frames_out=self._frames_out,
                dropped=self._dropped,
                fps=self._fps,
                inference_ms=round(self._inference_ms, 2),
                encode_ms=round(self._encode_ms, 2),
                decode_ms=round(self._decode_ms, 2),
                network_rtt_ms=round(self._rtt_ms, 2),
                queue_size=self._queue_size,
                uptime_s=round(time.monotonic() - self._start, 2),
            )

    def to_dict(self) -> dict:
        return self.snapshot().__dict__

    def reset(self) -> None:
        with self._lock:
            self._frames_in = self._frames_out = self._dropped = 0
