import asyncio
import logging

from aiortc import MediaStreamError, VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger("client.tracks")


class CameraVideoTrack(VideoStreamTrack):
    """Publishes the latest camera frame to WebRTC."""

    def __init__(self, camera, poll_ms: int = 2) -> None:
        super().__init__()
        self._camera = camera
        self._poll = poll_ms / 1000

    async def recv(self) -> VideoFrame:
        while True:
            frame = self._camera.latest()
            if frame is not None:
                break
            await asyncio.sleep(self._poll)
        pts, time_base = await self.next_timestamp()
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf


class ReceivingVideoTrack:
    """Wraps the remote (processed) track; pushes BGR frames to a holder."""

    def __init__(self, remote_track: VideoStreamTrack, holder, metrics=None) -> None:
        self._track = remote_track
        self._holder = holder
        self._metrics = metrics
        self._task = None

    async def run(self) -> None:
        async def _pump():
            while True:
                try:
                    frame = await self._track.recv()
                except MediaStreamError:
                    break
                bgr = frame.to_ndarray(format="bgr24")
                self._holder.put(bgr)
                if self._metrics is not None:
                    self._metrics.record_incoming()

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
