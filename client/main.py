import argparse
import asyncio
import logging
import signal
from pathlib import Path

from client.camera import CameraCapture
from client.virtualcam import VirtualCamera
from client.webrtc import WebRTCClient
from shared.config import ClientConfig, load_config
from shared.frame_holder import FrameHolder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client.main")


async def run(cfg: ClientConfig) -> None:
    out_holder = FrameHolder()
    cam = CameraCapture(index=cfg.camera.index, width=cfg.camera.width,
                        height=cfg.camera.height, fps=cfg.camera.fps)
    cam.start()
    vcam = VirtualCamera(out_holder, device=cfg.virtualcam.device,
                         width=cfg.virtualcam.width, height=cfg.virtualcam.height,
                         fps=cfg.virtualcam.fps)
    vcam.start()

    client = WebRTCClient(cfg, cam, out_holder)
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    try:
        # generous: first control can include server model warmup/load
        await asyncio.wait_for(client.connect(), timeout=180)
        logger.info("connected; streaming to %s", cfg.server_url)
        # periodic metrics log
        while not stop.is_set():
            logger.info("metrics=%s", client.metrics.to_dict())
            try:
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
    finally:
        cam.stop()
        vcam.stop()
        await client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="remote-deeplivecam client")
    ap.add_argument("--config", default="configs/client.yaml", type=Path)
    ap.add_argument("--face", default=None,
                    help="override the source face image (hot-swap on start)")
    args = ap.parse_args()
    cfg = load_config(ClientConfig, args.config)
    if args.face:
        cfg.processing.source_face = args.face
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
