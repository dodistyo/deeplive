import pytest

from server.rtc import Session
from shared.frame_holder import FrameHolder
from shared.metrics import Metrics


class FakeProcessor:
    def __init__(self):
        self.acked = []
        self.stopped = False
        self.feed_calls = 0
        self.metrics = Metrics()
        self.out_holder = FrameHolder()

    def apply_control(self, payload):
        self.acked.append(payload)
        return True

    def feed(self, bgr):
        self.feed_calls += 1

    def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_control_returns_ack():
    proc = FakeProcessor()
    s = Session("c1", proc)
    try:
        reply = await s.handle({"type": "control", "payload": {"face": "x"}})
        assert reply["type"] == "ack"
        assert proc.acked == [{"face": "x"}]
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_control_rejected_returns_error():
    class Reject(FakeProcessor):
        def apply_control(self, payload):
            return False

    s = Session("c1", Reject())
    try:
        reply = await s.handle({"type": "control", "payload": {"face": "x"}})
        assert reply["type"] == "error"
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_ping_returns_pong():
    s = Session("c1", FakeProcessor())
    try:
        assert await s.handle({"type": "ping"}) == {"type": "pong"}
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_unknown_type_raises():
    s = Session("c1", FakeProcessor())
    try:
        with pytest.raises(ValueError):
            await s.handle({"type": "nonsense"})
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_close_stops_processor():
    proc = FakeProcessor()
    s = Session("c1", proc)
    await s.close()
    assert proc.stopped is True
