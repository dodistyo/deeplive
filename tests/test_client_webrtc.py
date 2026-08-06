import base64

import numpy as np

from client.webrtc import WebRTCClient
from shared.config import ClientConfig


def _cfg_with_face(path) -> ClientConfig:
    cfg = ClientConfig(client={"server_url": "ws://x/ws/me", "client_id": "me"})
    cfg.processing.source_face = str(path)
    return cfg


def test_control_payload_encodes_face_base64(tmp_path):
    img = np.full((64, 64, 3), 128, np.uint8)
    import cv2
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    face_path = tmp_path / "face.jpg"
    face_path.write_bytes(buf.tobytes())

    client = WebRTCClient(_cfg_with_face(face_path), camera=None, out_holder=None)
    payload = client._control_payload()
    assert payload["face"] == base64.b64encode(buf.tobytes()).decode()
    settings = payload["settings"]
    assert settings["frame_processors"] == ["face_swapper"]
    assert settings["detection_interval_ms"] == 66
    assert settings["many_faces"] is False


def test_control_payload_missing_face_file_is_empty(tmp_path):
    client = WebRTCClient(_cfg_with_face(tmp_path / "missing.jpg"), camera=None, out_holder=None)
    assert client._control_payload()["face"] == ""


def test_control_api_methods_exist():
    import inspect
    from pathlib import Path
    client = WebRTCClient(_cfg_with_face(Path("/tmp/none.jpg")), camera=None, out_holder=None)
    assert inspect.iscoroutinefunction(client.set_face)
    assert inspect.iscoroutinefunction(client.update_settings)
    assert inspect.iscoroutinefunction(client.connect)
    assert inspect.iscoroutinefunction(client.close)
