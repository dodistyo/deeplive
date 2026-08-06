"""End-to-end integration test: real client (WebRTCClient + fake camera) talking
to a real server (uvicorn + create_app + stub processor) over a real WebSocket.

Exercises the full signaling protocol (config -> control -> offer -> answer),
the client reader/ping loops, the server Session routing, and the media path.
No GPU/webcam/v4l2loopback needed: the server echoes frames via a stub DLC.
"""
import asyncio
import socket

import numpy as np
import pytest
import uvicorn

from client.webrtc import WebRTCClient
from server.app import create_app
from server.processor import Processor
from shared.config import ClientConfig, ServerConfig
from shared.frame_holder import FrameHolder


class StubDLC:
    def set_source_face(self, image_bytes):
        return True

    def update_settings(self, settings):
        pass

    def process_frame(self, frame):
        return frame


class FakeCamSource:
    def __init__(self):
        self._h = FrameHolder()

    def latest(self):
        return self._h.get()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_e2e_client_server_media_flows():
    port = _free_port()
    app = create_app(ServerConfig(), processor_factory=lambda: Processor(StubDLC()))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error", access_log=False))
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started, "server failed to start"

        cfg = ClientConfig(client={"server_url": f"ws://127.0.0.1:{port}/ws/test",
                                   "client_id": "test"})
        cfg.processing.source_face = "/nonexistent-face.jpg"  # -> face="", settings only

        cam = FakeCamSource()
        out_holder = FrameHolder()
        client = WebRTCClient(cfg, cam, out_holder)
        await asyncio.wait_for(client.connect(), timeout=30)

        # feed synthetic frames from the "webcam"
        for i in range(30):
            cam._h.put(np.full((64, 64, 3), i % 255, np.uint8))
            await asyncio.sleep(0.02)

        # frames must come back into the virtual-camera holder
        got = None
        for _ in range(200):
            got = out_holder.get()
            if got is not None:
                break
            await asyncio.sleep(0.05)
        assert got is not None, "client received no processed frames"
        assert got.shape == (64, 64, 3)
        assert client.metrics.snapshot().network_rtt_ms >= 0

        await client.close()
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            server_task.cancel()
