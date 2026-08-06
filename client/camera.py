import logging
import threading

import cv2

from shared.frame_holder import FrameHolder

logger = logging.getLogger("client.camera")


class CameraCapture:
    """OpenCV webcam capture on a background thread; newest frame wins."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 60) -> None:
        api = getattr(cv2, "CAP_V4L2", 0)  # Linux; falls back to any backend
        self._cap = cv2.VideoCapture(index, api)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._holder = FrameHolder()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        if not self._cap.isOpened():
            raise RuntimeError("cannot open camera")
        self._thread.start()
        logger.info("camera opened: %s", self._cap)

    def latest(self):
        return self._holder.get()

    def stop(self) -> None:
        self._stop.set()
        self._cap.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            if frame is not None:
                self._holder.put(frame)
