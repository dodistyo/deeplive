import asyncio
import time

import numpy as np
from aiortc import MediaStreamError, VideoStreamTrack
from av import VideoFrame

from shared.frame_holder import FrameHolder


def frame_to_bgr(video_frame: VideoFrame) -> np.ndarray:
    return video_frame.to_ndarray(format="bgr24")


def bgr_to_frame(bgr: np.ndarray, pts: int | None = None, time_base=None) -> VideoFrame:
    vf = VideoFrame.from_ndarray(bgr, format="bgr24")
    if pts is not None and time_base is not None:
        vf.pts = pts
        vf.time_base = time_base
    return vf


class InboundVideoTrack:
    """Feeds received WebRTC frames into a processor or FrameHolder.

    Not a MediaStreamTrack subclass: it wraps the remote track aiortc gives
    us and drains it, so recv() keeps the RTP flow healthy. Pass either
    on_frame (a callable taking a BGR ndarray) or holder (a FrameHolder).
    """

    def __init__(self, remote_track: VideoStreamTrack, on_frame=None, holder: FrameHolder | None = None,
                 metrics=None) -> None:
        if on_frame is None and holder is None:
            raise ValueError("provide on_frame or holder")
        self._track = remote_track
        self._on_frame = on_frame
        self._holder = holder
        self._metrics = metrics
        self._decode_ms = 0.0
        self._task = None

    async def run(self) -> None:
        async def _pump():
            while True:
                try:
                    frame = await self._track.recv()
                except MediaStreamError:
                    break
                t0 = time.perf_counter()
                bgr = frame_to_bgr(frame)
                self._decode_ms = (time.perf_counter() - t0) * 1000
                if self._on_frame is not None:
                    self._on_frame(bgr)
                else:
                    self._holder.put(bgr)
                if self._metrics is not None:
                    self._metrics.record_incoming()
                    self._metrics.record_decode_ms(self._decode_ms)

        try:
            await _pump()
        except (MediaStreamError, ConnectionError):
            pass

    async def start(self) -> None:
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class ProcessedVideoTrack(VideoStreamTrack):
    """Outbound track: sends the newest processed frame and blocks until a
    genuinely new frame arrives (never re-sends the same frame in a loop —
    newest-frame-wins, no flooding)."""

    def __init__(self, holder: FrameHolder, poll_ms: int = 2, metrics=None) -> None:
        super().__init__()
        self._holder = holder
        self._poll = poll_ms / 1000
        self._metrics = metrics
        self._last = None

    async def recv(self) -> VideoFrame:
        while True:
            frame, seq = self._holder.get_new(self._last, timeout=0)
            if frame is not None and seq != self._last:
                self._last = seq
                break
            await asyncio.sleep(self._poll)

        t0 = time.perf_counter()
        pts, time_base = await self.next_timestamp()
        vf = bgr_to_frame(frame, pts, time_base)
        if self._metrics is not None:
            self._metrics.record_encode_ms((time.perf_counter() - t0) * 1000)
            self._metrics.record_outgoing()
        return vf
