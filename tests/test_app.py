from fastapi.testclient import TestClient

from server.app import create_app
from shared.config import ServerConfig


def test_health():
    app = create_app(ServerConfig(), processor_factory=_stub_processor)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"ok": True}


def test_root():
    app = create_app(ServerConfig(), processor_factory=_stub_processor)
    with TestClient(app) as client:
        body = client.get("/").json()
        assert body["app"] == "remote-deeplivecam"
        assert body["sessions"] == 0


def test_metrics_empty():
    app = create_app(ServerConfig(), processor_factory=_stub_processor)
    with TestClient(app) as client:
        assert client.get("/metrics").json() == {"sessions": {}}


def test_ws_handshake_config_then_ping_pong():
    app = create_app(ServerConfig(), processor_factory=_stub_processor)
    with TestClient(app) as client, client.websocket_connect("/ws/laptop1") as ws:
        # server sends negotiated config first
        cfg_msg = ws.receive_json()
        assert cfg_msg["type"] == "config"
        assert cfg_msg["payload"]["camera"]["fps"] == 60
        # ping -> pong round trip
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def _stub_processor():
    class P:
        def __init__(self):
            from shared.metrics import Metrics
            self.metrics = Metrics()

        def start(self):
            pass

        def stop(self):
            pass

        def out_holder(self):
            return None

    return P()
