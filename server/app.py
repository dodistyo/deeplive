import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from server.deeplivecam import DeepLiveCamProcessor
from server.processor import Processor
from server.rtc import Session
from shared.config import ServerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server.app")

SESSIONS: dict[str, Session] = {}


def _warmup_models(cfg: ServerConfig, processor_factory=None) -> None:
    """Preload DLC models at startup so the first client control message
    and first frame don't pay MIGraphX's model-compilation cost (~150s on
    a cold server), which blows the client connect timeout."""
    if processor_factory is not None:  # test stub
        return
    try:
        DeepLiveCamProcessor(cfg.model_dump()).warmup()
    except Exception:
        logger.warning("model warmup failed; first client session may be slow",
                       exc_info=True)


def create_app(cfg: ServerConfig, processor_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        logger.info("server starting")
        _warmup_models(cfg, processor_factory)
        yield
        for s in list(SESSIONS.values()):
            await s.close()
        SESSIONS.clear()

    app = FastAPI(title="remote-deeplivecam", lifespan=lifespan)

    def build_processor():
        if processor_factory:
            return processor_factory()
        # Only machine-side settings (execution_provider, model_dir) come
        # from server config; face + swap settings arrive per-session via
        # the control channel.
        return Processor(
            DeepLiveCamProcessor(cfg.model_dump()),
            drop_frames=cfg.performance.drop_frames,
        )

    def negotiate_payload() -> dict:
        return {
            "camera": cfg.camera.model_dump(),
            "network": cfg.network.model_dump(),
            "performance": cfg.performance.model_dump(),
        }

    @app.get("/")
    async def root():
        return {"app": "remote-deeplivecam", "sessions": len(SESSIONS)}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/metrics")
    async def metrics():
        out = {"sessions": {}}
        for cid, s in SESSIONS.items():
            out["sessions"][cid] = s.processor.metrics.to_dict()
        return JSONResponse(out)

    @app.websocket("/ws/{client_id}")
    async def signaling(ws: WebSocket, client_id: str):
        await ws.accept()
        proc = build_processor()
        proc.start()
        session = Session(client_id, proc, prefer_codec=cfg.network.prefer_codec)
        SESSIONS[client_id] = session
        try:
            # tell the client what the server negotiated (resolution etc.)
            await ws.send_text(json.dumps({"type": "config", "payload": negotiate_payload()}))
            while True:
                raw = await ws.receive_text()
                reply = await session.handle(json.loads(raw))
                if reply is not None:
                    await ws.send_text(json.dumps(reply))
        except WebSocketDisconnect:
            logger.info("client %s disconnected", client_id)
        except Exception as exc:  # noqa: BLE001 - session-level error handling
            logger.warning("session %s error: %s", client_id, exc)
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            except Exception:
                logger.debug("failed to send error to client", exc_info=True)
        finally:
            await session.close()
            SESSIONS.pop(client_id, None)

    return app


if __name__ == "__main__":
    import uvicorn

    cfg = ServerConfig.from_yaml("configs/server.yaml") if __import__("os").path.exists("configs/server.yaml") else ServerConfig()
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port)
