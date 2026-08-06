import asyncio

import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from av import VideoFrame

from shared.codecs import prefer_codec


class T(VideoStreamTrack):
    async def recv(self) -> VideoFrame:
        pts, tb = await self.next_timestamp()
        f = VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="bgr24")
        f.pts = pts
        f.time_base = tb
        return f


def test_prefer_h264_applies():
    async def _t():
        pc = RTCPeerConnection()
        pc.addTrack(T())
        try:
            assert prefer_codec(pc, "video", "h264") is True
            tc = pc.getTransceivers()[0]
            mime_types = [c.mimeType for c in tc._preferred_codecs]
            assert any("H264" in m.upper() for m in mime_types)
        finally:
            await pc.close()

    asyncio.run(_t())


def test_prefer_unknown_codec_noop():
    async def _t():
        pc = RTCPeerConnection()
        pc.addTrack(T())
        try:
            assert prefer_codec(pc, "video", "bogus") is False
        finally:
            await pc.close()

    asyncio.run(_t())
