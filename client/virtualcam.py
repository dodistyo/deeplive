import logging
import threading

import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat

from shared.frame_holder import FrameHolder

logger = logging.getLogger("client.virtualcam")


class VirtualCamera:
    """Writes newest frames to a virtual camera device (v4l2loopback)."""

    def __init__(self, holder: FrameHolder, device: str = "/dev/video10",
                 width: int = 1280, height: int = 720, fps: int = 60, cam=None) -> None:
        self._holder = holder
        self._cam = cam or pyvirtualcam.Camera(device=device, width=width,
                                               height=height, fps=fps,
                                               fmt=PixelFormat.YUYV)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.run, name="virtualcam", daemon=True)
        logger.info("virtual camera: %s %dx%d@%d", self._cam.device, width, height, fps)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._cam.close()
        except Exception:
            logger.debug("virtual cam close failed", exc_info=True)

    def run(self) -> None:
        last_seq = None
        while not self._stop.is_set():
            frame, seq = self._holder.get_new(last_seq, timeout=0.05)
            if frame is None or seq == last_seq:
                continue
            if frame.shape[:2] != (self._cam.height, self._cam.width):
                logger.error(
                    "frame shape mismatch: got %s, vcam expects %dx%d — check camera/virtualcam resolution",
                    frame.shape, self._cam.width, self._cam.height,
                )
                last_seq = seq
                continue
            try:
                self._cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_YUYV))
                last_seq = seq
                self._cam.sleep_until_next_frame()
            except Exception:
                logger.exception("failed to send frame to virtual camera")
