import asyncio

import numpy as np
import pytest
from aiortc import RTCPeerConnection

from client.tracks import CameraVideoTrack
from server.processor import Processor
from server.tracks import InboundVideoTrack, ProcessedVideoTrack
from shared.frame_holder import FrameHolder
from tests.fake_dlc import EchoDLC


async def _gather(pc: RTCPeerConnection) -> None:
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.02)


class FakeCamSource:
    def __init__(self, holder):
        self._h = holder

    def latest(self):
        return self._h.get()


@pytest.mark.asyncio
async def test_video_flows_both_ways():
    # ---- server side ----
    server_pc = RTCPeerConnection()
    server_proc = Processor(EchoDLC())
    server_proc.start()
    received_on_server = []

    @server_pc.on("track")
    async def _on_server_track(track):
        def feed_and_record(bgr):
            server_proc.feed(bgr)
            received_on_server.append(bgr.copy())

        inbound = InboundVideoTrack(track, on_frame=feed_and_record)
        await inbound.start()

    server_pc.addTrack(ProcessedVideoTrack(server_proc.out_holder))

    # ---- client side ----
    client_pc = RTCPeerConnection()
    cam_holder = FrameHolder()
    received_on_client = []

    @client_pc.on("track")
    async def _on_client_track(track):
        async def _pump():
            from aiortc import MediaStreamError
            while True:
                try:
                    f = await track.recv()
                except MediaStreamError:
                    break
                received_on_client.append(f.to_ndarray(format="bgr24").copy())
        asyncio.create_task(_pump())

    client_pc.addTrack(CameraVideoTrack(FakeCamSource(cam_holder)))

    # ---- non-trickle offer/answer ----
    offer = await client_pc.createOffer()
    await client_pc.setLocalDescription(offer)
    await _gather(client_pc)
    await server_pc.setRemoteDescription(client_pc.localDescription)
    answer = await server_pc.createAnswer()
    await server_pc.setLocalDescription(answer)
    await _gather(server_pc)
    await client_pc.setRemoteDescription(server_pc.localDescription)

    # ---- feed frames for ~1s ----
    async def feed():
        for i in range(60):
            cam_holder.put(np.full((64, 64, 3), i % 255, np.uint8))
            await asyncio.sleep(0.016)

    feed_task = asyncio.create_task(feed())
    try:
        for _ in range(200):  # up to ~4s
            if received_on_server and received_on_client:
                break
            await asyncio.sleep(0.02)
    finally:
        feed_task.cancel()
        try:
            await feed_task
        except asyncio.CancelledError:
            pass

    await client_pc.close()
    await server_pc.close()
    server_proc.stop()

    assert len(received_on_server) > 0, "server received no webcam frames"
    assert len(received_on_client) > 0, "client received no processed frames"
    # echo path: the client must receive frames of the sent shape back
    assert received_on_client[0].shape == (64, 64, 3)
