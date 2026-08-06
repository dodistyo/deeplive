import asyncio

import numpy as np
import pytest

from client.tracks import CameraVideoTrack, ReceivingVideoTrack
from server.tracks import ProcessedVideoTrack, bgr_to_frame, frame_to_bgr
from shared.frame_holder import FrameHolder


def test_ndarray_roundtrip():
    arr = np.zeros((720, 1280, 3), np.uint8)
    arr[:, :, 0] = 7  # blue plane
    vf = bgr_to_frame(arr)
    assert vf.width == 1280 and vf.height == 720
    out = frame_to_bgr(vf)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8
    assert (out == arr).all()


def test_roundtrip_preserves_pts_and_timebase():
    import fractions
    arr = np.zeros((64, 64, 3), np.uint8)
    tb = fractions.Fraction(1, 90000)
    vf = bgr_to_frame(arr, pts=12345, time_base=tb)
    assert vf.pts == 12345
    assert vf.time_base == tb


@pytest.mark.asyncio
async def test_processed_track_waits_for_first_frame():
    holder = FrameHolder()
    track = ProcessedVideoTrack(holder, poll_ms=1)
    put_task = asyncio.create_task(_put_after_delay(holder))
    vf = await asyncio.wait_for(track.recv(), timeout=1.0)
    assert vf.width == 64 and vf.height == 64
    await put_task
    track.stop()


@pytest.mark.asyncio
async def test_processed_track_blocks_until_new_frame():
    holder = FrameHolder()
    holder.put(np.zeros((64, 64, 3), np.uint8))
    track = ProcessedVideoTrack(holder, poll_ms=1)
    first = await asyncio.wait_for(track.recv(), timeout=0.5)
    assert first.width == 64
    # no new frame yet -> recv must NOT return (blocks, never re-sends)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(track.recv(), timeout=0.1)
    # new frame arrives -> recv returns it
    holder.put(np.ones((64, 64, 3), np.uint8))
    second = await asyncio.wait_for(track.recv(), timeout=0.5)
    assert (frame_to_bgr(second) == 1).all()
    track.stop()


async def _put_after_delay(holder):
    await asyncio.sleep(0.02)
    holder.put(np.zeros((64, 64, 3), np.uint8))


class FakeCam:
    def __init__(self):
        self._h = FrameHolder()
        self._h.put(np.zeros((128, 128, 3), np.uint8))

    def latest(self):
        return self._h.get()


class FakeRemoteTrack:
    """A stand-in remote track: yields two synthetic frames then ends
    via MediaStreamError (matching the real aiortc recv() contract)."""

    def __init__(self):
        self._frames = [
            bgr_to_frame(np.full((32, 32, 3), 5, np.uint8)),
            bgr_to_frame(np.full((32, 32, 3), 9, np.uint8)),
        ]

    async def recv(self):
        from aiortc import MediaStreamError
        if not self._frames:
            raise MediaStreamError()
        return self._frames.pop(0)


@pytest.mark.asyncio
async def test_camera_track_recv_returns_videoframe():
    track = CameraVideoTrack(FakeCam(), poll_ms=1)
    vf = await asyncio.wait_for(track.recv(), timeout=0.5)
    assert vf.width == 128 and vf.height == 128
    track.stop()


@pytest.mark.asyncio
async def test_receiving_track_pushes_frames_to_holder():
    holder = FrameHolder()
    recv = ReceivingVideoTrack(FakeRemoteTrack(), holder)
    await recv.start()
    await asyncio.sleep(0.1)
    await recv.stop()
    latest, seq = holder.get_new(None, timeout=0.1)
    assert latest is not None
    assert latest.shape == (32, 32, 3)
    assert (latest == 9).all()  # newest frame won
    assert seq == 2
