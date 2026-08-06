import asyncio
import logging

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from server.tracks import InboundVideoTrack, ProcessedVideoTrack

logger = logging.getLogger("server.rtc")


async def _gather_complete(pc: RTCPeerConnection) -> None:
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.02)


class Session:
    """One client session: receive webcam, process, send back on one PC.

    The processor is built by app.py and owns the DLC worker + holders;
    this class only wires the WebRTC side and routes signaling messages.
    """

    def __init__(self, client_id: str, processor, poll_ms: int = 2, prefer_codec: str | None = None) -> None:
        self.client_id = client_id
        self.processor = processor
        self.prefer_codec = prefer_codec
        self.pc = RTCPeerConnection()
        self._inbound = None
        self._processed_track: VideoStreamTrack = ProcessedVideoTrack(
            processor.out_holder, poll_ms=poll_ms, metrics=processor.metrics
        )
        self.pc.addTrack(self._processed_track)

        @self.pc.on("track")
        async def on_track(track: VideoStreamTrack):
            if track.kind == "video":
                self._inbound = InboundVideoTrack(
                    track, on_frame=self.processor.feed, metrics=self.processor.metrics
                )
                await self._inbound.start()
                logger.info("session %s receiving video", self.client_id)

        @self.pc.on("connectionstatechange")
        async def on_state():
            logger.info("session %s state=%s", self.client_id, self.pc.connectionState)
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

    async def handle(self, msg: dict) -> dict:
        mtype = msg.get("type")
        if mtype == "control":
            # laptop -> server: set source face + swap settings.
            # apply_control does blocking work (face detection + ONNX model
            # load), so run it off the event loop or the whole signaling
            # socket stalls and the client's ack wait times out.
            payload = msg.get("payload", {})
            ok = await asyncio.to_thread(self.processor.apply_control, payload)
            if not ok:
                return {"type": "error", "message": "control rejected: no face detected in source image"}
            return {"type": "ack"}
        if mtype == "ping":
            return {"type": "pong"}
        if mtype == "offer":
            await self.pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type="offer")
            )
            if self.prefer_codec:
                from shared.codecs import prefer_codec
                prefer_codec(self.pc, "video", self.prefer_codec)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            await _gather_complete(self.pc)
            return {"type": "answer", "sdp": self.pc.localDescription.sdp}
        raise ValueError(f"unexpected message type: {mtype}")

    async def close(self) -> None:
        if self._inbound:
            await self._inbound.stop()
        self.processor.stop()
        await self.pc.close()
