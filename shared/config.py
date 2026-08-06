from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, StrictBool, ValidationError


class CameraConfig(BaseModel):
    width: int = Field(default=1280, ge=160)
    height: int = Field(default=720, ge=120)
    fps: int = Field(default=60, ge=1, le=120)
    index: int = 0  # client only: OpenCV camera index


class NetworkConfig(BaseModel):
    stun: bool = False
    turn: bool = False
    prefer_codec: Literal["h264", "vp8"] = "h264"


class ServerProcessingConfig(BaseModel):
    """Machine-side inference config (lives ONLY on the desktop).

    UX settings (which face, enhancer, many_faces, opacity...) do NOT
    belong here — the laptop owns those and sends them per-session over
    the signaling channel as a `control` message.
    """
    execution_provider: Literal["auto", "cpu", "rocm", "migraphx", "cuda", "coreml", "openvino", "dml"] = "auto"
    model_dir: str = "third_party/Deep-Live-Cam/models"
    warmup_face: str = ""  # server-local face image to preload models at startup


class ClientProcessingConfig(BaseModel):
    """Swap settings chosen on the laptop (control point). Sent to the
    server as a `control` message at connect and on every change."""
    source_face: str = ""  # laptop-local path to the source face image
    frame_processors: list[str] = ["face_swapper"]
    many_faces: StrictBool = False
    enhancer: StrictBool = False
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    mouth_mask: StrictBool = False
    poisson_blend: StrictBool = False
    detection_interval_ms: int = Field(default=66, ge=0)  # 0 = detect every frame


class PerformanceConfig(BaseModel):
    drop_frames: StrictBool = True
    poll_interval_ms: int = Field(default=2, ge=1, le=50)


class MetricsConfig(BaseModel):
    enabled: bool = True


class ServerBindConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8881, ge=1, le=65535)


class ServerConfig(BaseModel):
    server: ServerBindConfig = ServerBindConfig()
    camera: CameraConfig = CameraConfig()
    network: NetworkConfig = NetworkConfig()
    processing: ServerProcessingConfig = ServerProcessingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    metrics: MetricsConfig = MetricsConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "ServerConfig":
        return load_config(cls, path)


class VirtualCamConfig(BaseModel):
    device: str = "/dev/video10"
    width: int = 1280
    height: int = 720
    fps: int = 60


class ClientBindConfig(BaseModel):
    server_url: str  # e.g. "ws://192.168.1.50:8881/ws/laptop1"
    client_id: str = "laptop1"


class ClientConfig(BaseModel):
    client: ClientBindConfig
    camera: CameraConfig = CameraConfig()
    virtualcam: VirtualCamConfig = VirtualCamConfig()
    network: NetworkConfig = NetworkConfig()
    processing: ClientProcessingConfig = ClientProcessingConfig()
    performance: PerformanceConfig = PerformanceConfig()

    @property
    def server_url(self) -> str:
        return self.client.server_url

    @property
    def client_id(self) -> str:
        return self.client.client_id

    @classmethod
    def from_yaml(cls, path: Path) -> "ClientConfig":
        return load_config(cls, path)


def load_config(model_type, path: Path):
    if not Path(path).exists():
        raise ValueError(f"config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return model_type(**raw)
    except ValidationError as exc:
        raise ValueError(f"invalid config {path}: {exc.errors()}") from exc
