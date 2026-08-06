import asyncio
import base64
import json
import logging
import time
from pathlib import Path

import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

from shared.frame_holder import FrameHolder
from shared.metrics import Metrics
from shared.protocol import Signal

logger = logging.getLogger("client.webrtc")

# First control message can trigger server-side face detection + ONNX model
# load; generous to cover a cold-start / partial warmup.
_CONTROL_ACK_TIMEOUT = 120.0


async def _gather_complete(pc: RTCPeerConnection) -> None:
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.02)


class WebRTCClient:
    """Client peer. Offers a camera track; receives a processed track.

    Control point: sends the source face + swap settings at connect and on
    every change (hot-swap) over the signaling WebSocket. A background
    reader task consumes ack/answer/error/metrics so control messages can
    be sent after media is already flowing.
    """

    def __init__(self, config, camera, out_holder: FrameHolder) -> None:
        self.config = config
        self.camera = camera
        self.out_holder = out_holder      # where received processed frames go
        self.metrics = Metrics()
        self.negotiated = {}
        self.pc = RTCPeerConnection()
        self._receiver = None
        self._ws = None
        self._reader = None
        self._ping_task = None
        self._connected = asyncio.Event()
        self._control_ack = asyncio.Event()
        self._last_error = None
        self._pong_future = None
        self._ping_t0 = 0.0

        from client.tracks import CameraVideoTrack
        self.pc.addTrack(CameraVideoTrack(camera))

        @self.pc.on("track")
        def on_track(track):
            from client.tracks import ReceivingVideoTrack
            self._receiver = ReceivingVideoTrack(track, self.out_holder, self.metrics)
            asyncio.create_task(self._receiver.start())
            logger.info("received remote track: %s", track.kind)

        @self.pc.on("connectionstatechange")
        async def on_state():
            logger.info("connection state=%s", self.pc.connectionState)
            if self.pc.connectionState == "connected":
                self._connected.set()
            elif self.pc.connectionState == "failed":
                self._last_error = "connection failed"
                self._control_ack.set()

    # ---- control plane (laptop is the control point) ----

    def _control_payload(self) -> dict:
        p = self.config.processing
        path = Path(p.source_face)
        data = path.read_bytes() if path.exists() else b""
        return {
            "face": base64.b64encode(data).decode() if data else "",
            "settings": {
                "frame_processors": p.frame_processors,
                "many_faces": p.many_faces,
                "enhancer": p.enhancer,
                "opacity": p.opacity,
                "mouth_mask": p.mouth_mask,
                "poisson_blend": p.poisson_blend,
                "detection_interval_ms": p.detection_interval_ms,
            },
        }

    async def _send_control(self, payload: dict | None = None) -> None:
        self._control_ack.clear()
        self._last_error = None
        await self._ws.send(Signal.control(payload or self._control_payload()).encode())
        try:
            await asyncio.wait_for(self._control_ack.wait(), timeout=_CONTROL_ACK_TIMEOUT)
        except TimeoutError:
            logger.error(
                "control ack timeout after %ss: ws_open=%s connected=%s last_error=%s",
                _CONTROL_ACK_TIMEOUT, not self._ws.closed, self._connected.is_set(),
                self._last_error,
            )
            raise
        if self._last_error:
            raise RuntimeError(self._last_error)

    async def set_face(self, path: str) -> None:
        """Hot-swap the source face at runtime."""
        self.config.processing.source_face = path
        await self._send_control()

    async def update_settings(self, settings: dict) -> None:
        await self._send_control({"face": "", "settings": settings})

    # ---- connection lifecycle ----

    async def connect(self) -> None:
        url = self.config.server_url.replace("http", "ws", 1)
        self._ws = await websockets.connect(url)
        try:
            # 1) read negotiated config from the server
            while True:
                msg = Signal.decode(await self._ws.recv())
                if msg.type == "config":
                    self.negotiated = msg.data.get("payload", {})
                    break
            # 2) background reader handles ack/answer/error/pong/metrics
            self._reader = asyncio.create_task(self._read_loop())
            # 3) control plane: face + settings
            await self._send_control()
            # 4) media: offer camera track
            if self.config.network.prefer_codec:
                from shared.codecs import prefer_codec
                prefer_codec(self.pc, "video", self.config.network.prefer_codec)
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            await _gather_complete(self.pc)
            # NOTE: serialize pc.localDescription (post-gather, contains the
            # ICE candidates) — the createOffer() object has none.
            await self._ws.send(json.dumps({"type": "offer", "sdp": self.pc.localDescription.sdp}))
            # 5) wait for media to flow
            await asyncio.wait_for(self._wait_media(), timeout=20)
            # 6) periodic RTT ping over the signaling socket
            self._ping_task = asyncio.create_task(self._ping_loop())
            logger.info("connected; media flowing")
        except Exception:
            await self.close()
            raise

    async def _read_loop(self) -> None:
        try:
            while True:
                msg = Signal.decode(await self._ws.recv())
                if msg.type == "answer":
                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=msg.data["sdp"], type="answer")
                    )
                elif msg.type == "ack":
                    self._control_ack.set()
                elif msg.type == "pong":
                    if self._pong_future is not None and not self._pong_future.done():
                        self._pong_future.set_result(time.monotonic())
                elif msg.type == "error":
                    self._last_error = msg.data.get("message", "server error")
                    self._control_ack.set()
                elif msg.type == "metrics":
                    self.metrics.record_rtt_ms(float(
                        msg.data.get("snapshot", {}).get("network_rtt_ms", 0)))
        except websockets.ConnectionClosed:
            self._connected.clear()

    async def _ping_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(5)
                fut = asyncio.get_running_loop().create_future()
                self._pong_future = fut
                self._ping_t0 = time.monotonic()
                await self._ws.send(Signal.ping().encode())
                await asyncio.wait_for(fut, timeout=2)
                rtt_ms = (time.monotonic() - self._ping_t0) * 1000
                self.metrics.record_rtt_ms(rtt_ms)
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            self._connected.clear()

    async def _wait_media(self) -> None:
        while True:
            if self._last_error:
                raise RuntimeError(self._last_error)
            if self._connected.is_set() and self._receiver is not None:
                return
            await asyncio.sleep(0.02)

    async def close(self) -> None:
        if self._receiver:
            await self._receiver.stop()
        for task in (self._reader, self._ping_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws:
            await self._ws.close()
        await self.pc.close()
