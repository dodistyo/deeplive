import logging
import threading
import time

from shared.frame_holder import FrameHolder
from shared.metrics import Metrics

logger = logging.getLogger("server.processor")


class Processor:
    """Runs a blocking process_frame on a worker thread.

    feed() -> inbound holder (drop-old) -> worker -> outbound holder
    (drop-old). inference_ms is recorded per frame. Control messages
    (face + settings) are forwarded to the underlying DeepLiveCamProcessor.
    """

    def __init__(self, deeplivecam, poll_ms: int = 50, drop_frames: bool = True) -> None:
        self._dlc = deeplivecam
        self._drop_frames = drop_frames
        self._in = FrameHolder()
        self._out = FrameHolder()
        self._metrics = Metrics()
        self._poll = poll_ms / 1000
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dlc-processor", daemon=True)
        self._last_dropped = 0

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    @property
    def in_holder(self) -> FrameHolder:
        return self._in

    @property
    def out_holder(self) -> FrameHolder:
        return self._out

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=1.0)

    def feed(self, bgr_frame) -> None:
        if not self._drop_frames and self._in.get() is not None:
            logger.warning("drop_frames disabled but holder non-empty; frame would be dropped")
        self._in.put(bgr_frame)
        dropped = self._in.dropped
        if dropped > self._last_dropped:
            self._metrics.record_dropped(dropped - self._last_dropped)
            self._last_dropped = dropped
        self._metrics.record_incoming(queue_size=1 if self._in.get() is not None else 0)

    def latest_out(self):
        return self._out.get()

    def apply_control(self, payload: dict) -> bool:
        """Apply a laptop control message (face + settings) to the
        underlying DeepLiveCamProcessor. Returns True on success."""
        ok = True
        face_bytes = payload.get("face")
        if face_bytes:
            import base64
            raw = base64.b64decode(face_bytes) if isinstance(face_bytes, str) else face_bytes
            ok = self._dlc.set_source_face(raw)
        settings = payload.get("settings")
        if settings:
            self._dlc.update_settings(settings)
        return ok

    def _run(self) -> None:
        last_seq = None
        while not self._stop_evt.is_set():
            frame, seq = self._in.get_new(last_seq, timeout=self._poll)
            if frame is None:
                continue
            last_seq = seq
            t0 = time.perf_counter()
            try:
                out = self._dlc.process_frame(frame)
            except Exception:  # never kill the worker
                logger.exception("inference failed")
                out = frame
            dt_ms = (time.perf_counter() - t0) * 1000
            self._metrics.record_inference_ms(dt_ms)
            if out is not None:
                self._out.put(out)
