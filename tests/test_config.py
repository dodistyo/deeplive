from shared.config import ClientConfig, ServerConfig, load_config


def test_server_defaults(tmp_path):
    p = tmp_path / "server.yaml"
    p.write_text("server:\n  host: 0.0.0.0\n  port: 8000\n")
    cfg = load_config(ServerConfig, p)
    assert cfg.server.host == "0.0.0.0"
    assert cfg.camera.fps == 60
    assert cfg.performance.drop_frames is True


def test_client_missing_required_raises(tmp_path):
    p = tmp_path / "client.yaml"
    p.write_text("camera:\n  width: 1280\n")
    try:
        load_config(ClientConfig, p)
    except ValueError:
        return
    raise AssertionError("missing client.server_url should raise")


def test_client_processing_defaults(tmp_path):
    p = tmp_path / "client.yaml"
    p.write_text("client:\n  server_url: ws://x/ws/me\n")
    cfg = load_config(ClientConfig, p)
    assert cfg.processing.frame_processors == ["face_swapper"]
    assert cfg.processing.detection_interval_ms == 66


def test_invalid_enum(tmp_path):
    p = tmp_path / "client.yaml"
    p.write_text("client:\n  server_url: ws://x\nprocessing:\n  many_faces: notabool\n")
    try:
        load_config(ClientConfig, p)
    except ValueError:
        return
    raise AssertionError("invalid value should raise")


def test_from_yaml_helper(tmp_path):
    p = tmp_path / "server.yaml"
    p.write_text("server:\n  host: 127.0.0.1\n  port: 9000\n")
    cfg = ServerConfig.from_yaml(p)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 9000
