# Remote Deep-Live-Cam (Client/Server) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A LAN client/server app where a laptop streams its webcam over WebRTC to a desktop (RX 7900 XTX) running headless Deep-Live-Cam inference, and receives the processed stream back into a Linux virtual camera (`/dev/video10`), with end-to-end latency <100 ms at 720p.

**Architecture:** The laptop (client) captures webcam frames with OpenCV and pushes them up a single `aiortc` `RTCPeerConnection` as a custom `VideoStreamTrack`. The server receives that track, converts each frame to a numpy BGR array, runs it through a headless Deep-Live-Cam `process_frame(source_face, frame, target_face)` pipeline on a dedicated worker thread using a **newest-frame-wins** holder (never queues — drops old frames), then pushes results back down the **same** connection. The client receives the processed track and writes frames to `pyvirtualcam` → v4l2loopback → `/dev/video10`. Signaling is one FastAPI WebSocket; SDP is exchanged non-trickle (we wait for ICE gathering to complete so candidates are embedded in the SDP), so no STUN/TURN is needed on a LAN. GPU inference happens only on the server.

**Control plane (user sits at the laptop):** The laptop is the control point. It owns the UX — the source face image and all swap settings (frame processors, enhancer, many-faces, opacity, detection interval). These travel **laptop → server** over the same signaling WebSocket as a `control` message (sent at connect, and again whenever the user hot-swaps the face or changes settings). The desktop holds no config of its own for faces: it keeps the currently-selected source-face embedding in memory, applied to incoming frames until told otherwise. If no face has been set, the server passes frames through unmodified.

```
Laptop Webcam --(OpenCV)--> CameraVideoTrack --(WebRTC)--> Server InboundTrack
                                                                    |
                                                          FrameHolder (maxsize=1, drop old)
                                                                    |
                                                          DLC process_frame (worker thread)
                                                                    |
                                                          FrameHolder (maxsize=1, drop old)
                                                                    |
Laptop /dev/video10 <--(pyvirtualcam)<-- ReceivingTrack <--(WebRTC)<-- ProcessedVideoTrack
```

**Tech Stack:** Python 3.12 (system has 3.12.3; spec's 3.11 target is fine — code is 3.11-compatible), FastAPI + uvicorn, aiortc + av (PyAV), OpenCV, numpy, pydantic + PyYAML, pyvirtualcam + v4l2loopback, pytest + pytest-asyncio. Deep-Live-Cam (hacksider/main) is vendored under `third_party/` and imported as a headless library. ONNX Runtime on the server via **ROCm** (`onnxruntime-rocm`, pinned) with **MIGraphX** (`onnxruntime-migraphx`) as the ROCm-7 path. **Vulkan is explicitly rejected**: the Vulkan EP is not shipped in desktop wheels (it exists only as the WebGPU EP's backend), has zero real-world usage, and its poor op coverage would run the inswapper/detection graphs on CPU anyway.

**Latency budget (720p60, LAN):** cam read ~0 ms → encode ~3 ms → network ~2 ms → decode ~3 ms → inference <25 ms → encode ~3 ms → network ~2 ms → decode ~3 ms → vcam write ~0 ms ≈ **~41 ms**. Headroom to <100 ms.

---

## Repository Layout

```
remote-deeplivecam/
├── README.md
├── .gitignore
├── pyproject.toml               # dev tooling only: pytest, pytest-asyncio, ruff
├── requirements-server.txt
├── requirements-client.txt
├── requirements-dev.txt
├── configs/
│   ├── server.yaml
│   └── client.yaml
├── shared/
│   ├── __init__.py
│   ├── config.py
│   ├── protocol.py
│   ├── frame_holder.py
│   └── metrics.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── rtc.py
│   ├── tracks.py
│   ├── processor.py
│   └── deeplivecam.py
├── client/
│   ├── __init__.py
│   ├── main.py
│   ├── camera.py
│   ├── tracks.py
│   ├── webrtc.py
│   └── virtualcam.py
├── scripts/
│   ├── setup-dlc.sh
│   └── setup-client.sh          # v4l2loopback + modprobe
├── third_party/                 # Deep-Live-Cam clone (gitignored)
│   └── Deep-Live-Cam/
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_protocol.py
    ├── test_frame_holder.py
    ├── test_metrics.py
    ├── test_tracks.py
    ├── test_processor.py
    ├── test_deeplivecam.py
    └── test_rtc_loopback.py
```

**Key upstream facts this plan relies on (verified against hacksider/Deep-Live-Cam `main`, 2026-08-05):**

- The per-frame entry point is `modules.processors.frame.face_swapper.process_frame(source_face, temp_frame, target_face=None)` (returns a BGR `np.ndarray`). Passing a cached `target_face` skips the internal per-frame face detection (~30–40 ms saved). `swap_face(source_face, target_face, frame)` does the actual paste-back.
- Global knobs live in `modules.globals`: `headless`, `frame_processors`, `execution_providers`, `many_faces`, `mouth_mask`, `poisson_blend`, `opacity`, `sharpness`, `enable_interpolation`.
- `modules.core.update_status()` calls `ui.update_status()` **only when `not headless`**, so setting `modules.globals.headless = True` avoids touching the GUI. However `modules.core` does `import modules.ui as ui` at import time, which pulls in customtkinter. We avoid this entirely by pre-inserting a stub for `modules.ui` into `sys.modules` before importing any `modules.*` (see Task 7). This removes the tkinter/customtkinter dependency from the server.
- Execution providers: `onnxruntime.get_available_providers()` → filter for `cuda`/`rocm`/`migraphx`/`coreml`/`openvino`/`dml`/`cpu` (mirror DLC's `decode_execution_providers`). The community AMD path is `onnxruntime-rocm` pinned to a known-good ORT↔ROCm pair (facefusion ships `onnxruntime-rocm==1.22.2.post1` with ROCm 6.x); on ROCm 7.x use `onnxruntime-migraphx`. RDNA3 (gfx1100) is "best-effort" supported by ROCm — expect kernel-JIT and version-mismatch flakiness; mitigate with exact version pinning, MIOpen cache pre-warm, and a runtime check that `ROCMExecutionProvider` is genuinely active (not silently on CPU).

**aiortc gotchas (verified against 1.15.0, 2026-08-06):**
- `RemoteStreamTrack` does **not** support `async for` — drain with `while True: frame = await track.recv()` and break on `MediaStreamError`.
- `createOffer()`/`createAnswer()` return SDP **without ICE candidates**. You must `await pc.setLocalDescription(...)`, wait until `pc.iceGatheringState == "complete"`, then serialize **`pc.localDescription.sdp`** (which contains `a=candidate:` + `a=end-of-candidates`). Passing the pre-gather object to the peer leaves the remote with zero candidates and ICE stuck at "checking".
- Both peers on one event loop connect fine with host candidates; the in-process loopback test proves the non-trickle flow.

---

## Phase 0 — Project Skeleton & Shared Foundation

### Task 1: Repo skeleton, dependency files, tooling

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `requirements-server.txt`, `requirements-client.txt`, `requirements-dev.txt`, `README.md` (stub), `shared/__init__.py`, `server/__init__.py`, `client/__init__.py`, `tests/__init__.py`

**Step 1:** Create the files.

`.gitignore`:
```gitignore
__pycache__/
*.pyc
.venv/
.venv-*/
third_party/
configs/*.local.yaml
*.log
.pytest_cache/
.ruff_cache/
```

`requirements-server.txt`:
```
fastapi>=0.115
uvicorn[standard]>=0.30
aiortc>=1.9.0
av>=12.0
numpy<2.0
opencv-python-headless>=4.9
pydantic>=2.7
PyYAML>=6.0
# AMD GPU inference on the server — pin ONE of these (NOT both):
#   ROCm 6.x: onnxruntime-rocm==1.22.2.post1
#   ROCm 7.x: onnxruntime-migraphx
# DLC's requirements.txt must not silently downgrade this afterwards.
```

`requirements-client.txt`:
```
aiortc>=1.9.0
av>=12.0
numpy<2.0
opencv-python>=4.9
pyvirtualcam>=0.4.0
pydantic>=2.7
PyYAML>=6.0
websockets>=12.0
```

`requirements-dev.txt`:
```
pytest>=8.0
pytest-asyncio>=0.23
ruff>=0.4
websockets>=12.0
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

**Step 2:** Create venv and install dev deps (client-side deps install now; server DLC deps come in Task 6).
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-client.txt
```
Expected: installs succeed, `pytest --version` prints pytest 8.x.

**Step 3:** Sanity-check empty test run.
```bash
pytest -q
```
Expected: `no tests ran` exit code 5 (fine) or a placeholder passing test.

**Step 4:** Commit.
```bash
git add .
git commit -m "chore: project skeleton, deps, tooling"
```

---

### Task 2: `shared/config.py` — typed YAML config

**Files:**
- Create: `shared/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test** (`tests/test_config.py`):
```python
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
    p.write_text("client:\n  server_url: ws://x\n  processing:\n    many_faces: notabool\n")
    try:
        load_config(ClientConfig, p)
    except ValueError:
        return
    raise AssertionError("invalid value should raise")
```

**Step 2:** Run to verify it fails.
```bash
pytest tests/test_config.py -q
```
Expected: FAIL (ModuleNotFoundError: shared.config).

**Step 3: Implement** (`shared/config.py`):
```python
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


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


class ClientProcessingConfig(BaseModel):
    """Swap settings chosen on the laptop (control point). Sent to the
    server as a `control` message at connect and on every change."""
    source_face: str = ""  # laptop-local path to the source face image
    frame_processors: list[str] = ["face_swapper"]
    many_faces: bool = False
    enhancer: bool = False
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    mouth_mask: bool = False
    poisson_blend: bool = False
    detection_interval_ms: int = Field(default=66, ge=0)  # 0 = detect every frame


class PerformanceConfig(BaseModel):
    drop_frames: bool = True
    poll_interval_ms: int = Field(default=2, ge=1, le=50)


class MetricsConfig(BaseModel):
    enabled: bool = True


class ServerConfig(BaseModel):
    server: dict = {"host": "0.0.0.0", "port": 8000}
    camera: CameraConfig = CameraConfig()
    network: NetworkConfig = NetworkConfig()
    processing: ServerProcessingConfig = ServerProcessingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    metrics: MetricsConfig = MetricsConfig()


class VirtualCamConfig(BaseModel):
    device: str = "/dev/video10"
    width: int = 1280
    height: int = 720
    fps: int = 60


class ClientConfig(BaseModel):
    server_url: str  # e.g. "ws://192.168.1.50:8000/ws/laptop1"
    client_id: str = "laptop1"
    camera: CameraConfig = CameraConfig()
    virtualcam: VirtualCamConfig = VirtualCamConfig()
    network: NetworkConfig = NetworkConfig()
    processing: ClientProcessingConfig = ClientProcessingConfig()
    performance: PerformanceConfig = PerformanceConfig()


def load_config(model_type, path: Path):
    if not Path(path).exists():
        raise ValueError(f"config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return model_type(**raw)
    except ValidationError as exc:
        raise ValueError(f"invalid config {path}: {exc.errors()}") from exc
```

**Step 4:** Run to verify it passes.
```bash
pytest tests/test_config.py -q
```
Expected: 4 passed.

**Step 5:** Commit.
```bash
git add shared/config.py tests/test_config.py
git commit -m "feat(shared): typed yaml config models"
```

---

### Task 3: `shared/protocol.py` — signaling messages

**Files:**
- Create: `shared/protocol.py`
- Test: `tests/test_protocol.py`

**Step 1: Write the failing test** (`tests/test_protocol.py`):
```python
import json
from shared.protocol import Signal


def test_offer_roundtrip():
    sig = Signal.offer(sdp="v=0\r\n")
    msg = json.loads(sig.encode())
    assert msg["type"] == "offer"
    assert msg["sdp"] == "v=0\r\n"
    assert Signal.decode(sig.encode()) == sig

def test_ice_and_answer():
    ice = Signal.ice({"candidate": "c1", "sdpMid": "0", "sdpMLineIndex": 0})
    ans = Signal.answer(sdp="v=0")
    assert Signal.decode(ice.encode()).type == "ice"
    assert Signal.decode(ans.encode()).type == "answer"

def test_control_carries_face_and_settings():
    payload = {
        "settings": {"frame_processors": ["face_swapper"], "many_faces": False,
                     "detection_interval_ms": 66},
        "face": "iVBORw0KGgo=...",  # base64 jpeg bytes of the source face
    }
    sig = Signal.control(payload)
    decoded = Signal.decode(sig.encode())
    assert decoded.type == "control"
    assert decoded.data["payload"]["settings"]["frame_processors"] == ["face_swapper"]
    assert decoded.data["payload"]["face"] == "iVBORw0KGgo=..."

def test_ping_pong():
    assert Signal.decode(Signal.ping().encode()).type == "ping"
    assert Signal.decode(Signal.pong().encode()).type == "pong"

def test_malformed_raises():
    try:
        Signal.decode('{"type": "bogus"}')
    except ValueError:
        return
    raise AssertionError("unknown type should raise")
```

**Step 2:** Run to verify it fails. Expected: ModuleNotFoundError.

**Step 3: Implement** (`shared/protocol.py`):
```python
import json
from dataclasses import dataclass, field
from typing import Any

# Signaling message types exchanged over the WebSocket.
# SDP is exchanged non-trickle (candidates embedded in the SDP after
# ICE gathering completes), so 'ice' is defined for future/extensibility.


@dataclass
class Signal:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def offer(sdp: str) -> "Signal":
        return Signal("offer", {"sdp": sdp})

    @staticmethod
    def answer(sdp: str) -> "Signal":
        return Signal("answer", {"sdp": sdp})

    @staticmethod
    def ice(candidate: dict) -> "Signal":
        return Signal("ice", {"candidate": candidate})

    @staticmethod
    def config(payload: dict) -> "Signal":
        return Signal("config", {"payload": payload})

    @staticmethod
    def control(payload: dict) -> "Signal":
        """Laptop -> server: source face (base64 image bytes) + swap settings."""
        return Signal("control", {"payload": payload})

    @staticmethod
    def metrics(snapshot: dict) -> "Signal":
        return Signal("metrics", {"snapshot": snapshot})

    @staticmethod
    def ping() -> "Signal":
        return Signal("ping", {})

    @staticmethod
    def pong() -> "Signal":
        return Signal("pong", {})

    @staticmethod
    def ack() -> "Signal":
        return Signal("ack", {})

    @staticmethod
    def error(message: str) -> "Signal":
        return Signal("error", {"message": message})

    def encode(self) -> str:
        return json.dumps({"type": self.type, **self.data})

    @classmethod
    def decode(cls, raw: str) -> "Signal":
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid signal json: {exc}") from exc
        if not isinstance(msg, dict) or "type" not in msg:
            raise ValueError(f"signal missing 'type': {raw!r}")
        known = {"offer", "answer", "ice", "config", "control", "metrics",
                 "ping", "pong", "ack", "error"}
        if msg["type"] not in known:
            raise ValueError(f"unknown signal type: {msg['type']}")
        return cls(type=msg["type"], data={k: v for k, v in msg.items() if k != "type"})
```

**Step 4:** Run to verify it passes.

**Step 5:** Commit.
```bash
git add shared/protocol.py tests/test_protocol.py
git commit -m "feat(shared): signaling protocol messages"
```

---

### Task 4: `shared/frame_holder.py` — newest-frame-wins primitive

This is the heart of the "never queue, drop old frames" requirement. It is a thread-safe holder with a monotonically increasing sequence number so `get_new()` reliably detects "a newer frame arrived" even if the same numpy buffer object is reused.

**Files:**
- Create: `shared/frame_holder.py`
- Test: `tests/test_frame_holder.py`

**Step 1: Write the failing test** (`tests/test_frame_holder.py`):
```python
import threading
import time

from shared.frame_holder import FrameHolder


def test_put_overwrites_older():
    h = FrameHolder()
    h.put("a")
    h.put("b")          # old frame dropped
    h.put("c")
    assert h.get() == "c"


def test_get_blocks_until_first_frame():
    h = FrameHolder()
    results = []

    def reader():
        results.append(h.get_new(None, timeout=1.0))

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    h.put("x")
    t.join(2.0)
    assert not t.is_alive()
    assert results == ["x"]


def test_get_new_waits_for_different_frame():
    h = FrameHolder()
    h.put("old")
    got = []

    def reader():
        got.append(h.get_new(0, timeout=0.2))  # seq 0 -> first frame is seq 1

    t = threading.Thread(target=reader)
    t.start()
    t.join(0.5)
    assert got == ["old"]


def test_get_new_timeout_returns_none():
    h = FrameHolder()
    h.put("a")
    assert h.get_new(1, timeout=0.05) is None  # seq unchanged
```

**Step 2:** Run to verify it fails. Expected: ModuleNotFoundError.

**Step 3: Implement** (`shared/frame_holder.py`):
```python
import threading


class FrameHolder:
    """Thread-safe 'latest frame wins' holder.

    put() overwrites the previous value (old frames are dropped, never
    queued). get_new(last_seq, timeout) blocks until a frame with a
    sequence number different from last_seq is available, or returns None
    on timeout. seq increments on every put() so identical buffer objects
    still register as 'new'.
    """

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._seq = 0
        self._frame = None
        self._dropped = 0

    def put(self, frame) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._seq += 1
            self._frame = frame
            self._lock.notify_all()

    def get(self):
        """Return the latest frame or None if nothing has been put yet."""
        with self._lock:
            return self._frame

    def get_new(self, last_seq: int | None, timeout: float | None = None):
        """Block until a frame newer than last_seq exists.

        last_seq=None means 'first frame ever' (waits for any frame).
        Returns (frame, seq), or (None, last_seq) on timeout.
        """
        with self._lock:
            if last_seq is None:
                while self._frame is None:
                    if not self._lock.wait(timeout):
                        return None, last_seq
            else:
                while self._frame is None or self._seq == last_seq:
                    if not self._lock.wait(timeout):
                        return None, last_seq
            return self._frame, self._seq

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
```

**Step 4:** Run to verify it passes.

**Step 5:** Commit.
```bash
git add shared/frame_holder.py tests/test_frame_holder.py
git commit -m "feat(shared): latest-frame-wins holder with drop counter"
```

---

### Task 5: `shared/metrics.py` — metrics collector

**Files:**
- Create: `shared/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing test** (`tests/test_metrics.py`):
```python
from shared.metrics import Metrics, MetricsSnapshot


def test_counts_and_ema():
    m = Metrics()
    m.record_incoming()
    m.record_incoming()
    m.record_inference_ms(10.0)
    m.record_inference_ms(30.0)
    m.record_encode_ms(2.0)
    m.record_decode_ms(3.0)
    m.record_rtt_ms(4.0)
    snap = m.snapshot()
    assert isinstance(snap, MetricsSnapshot)
    assert snap.frames_in >= 2
    assert 10.0 <= snap.inference_ms <= 30.0
    assert snap.encode_ms > 0
    assert snap.decode_ms == 3.0  # first sample
    assert snap.rtt_ms == 4.0


def test_ema_smoothing():
    m = Metrics()
    for i in range(10):
        m.record_inference_ms(20.0)
    m.record_inference_ms(100.0)
    assert m.snapshot().inference_ms < 50.0  # smoothed, not spiked
```

**Step 2:** Run to verify it fails.

**Step 3: Implement** (`shared/metrics.py`):
```python
import threading
import time
from dataclasses import dataclass

_EMA_ALPHA = 0.3


@dataclass
class MetricsSnapshot:
    frames_in: int = 0
    frames_out: int = 0
    dropped: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    network_rtt_ms: float = 0.0
    queue_size: int = 0  # 0 or 1 by design (newest-wins)
    uptime_s: float = 0.0


class Metrics:
    """Thread-safe metric collector. Durations are EMA-smoothed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._frames_in = 0
        self._frames_out = 0
        self._dropped = 0
        self._fps_window_start = self._start
        self._fps_window_frames = 0
        self._fps = 0.0
        self._inference_ms = 0.0
        self._encode_ms = 0.0
        self._decode_ms = 0.0
        self._rtt_ms = 0.0
        self._queue_size = 0

    def _ema(self, old: float, sample: float) -> float:
        return old + _EMA_ALPHA * (sample - old) if old else sample

    def record_incoming(self, queue_size: int = 0) -> None:
        with self._lock:
            self._frames_in += 1
            self._queue_size = queue_size
            now = time.monotonic()
            self._fps_window_frames += 1
            if now - self._fps_window_start >= 1.0:
                self._fps = self._fps_window_frames / (now - self._fps_window_start)
                self._fps_window_start = now
                self._fps_window_frames = 0

    def record_outgoing(self) -> None:
        with self._lock:
            self._frames_out += 1

    def record_dropped(self, n: int = 1) -> None:
        with self._lock:
            self._dropped += n

    def record_inference_ms(self, ms: float) -> None:
        with self._lock:
            self._inference_ms = self._ema(self._inference_ms, ms)

    def record_encode_ms(self, ms: float) -> None:
        with self._lock:
            self._encode_ms = self._ema(self._encode_ms, ms)

    def record_decode_ms(self, ms: float) -> None:
        with self._lock:
            self._decode_ms = self._ema(self._decode_ms, ms)

    def record_rtt_ms(self, ms: float) -> None:
        with self._lock:
            self._rtt_ms = self._ema(self._rtt_ms, ms)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                frames_in=self._frames_in,
                frames_out=self._frames_out,
                dropped=self._dropped,
                fps=self._fps,
                inference_ms=round(self._inference_ms, 2),
                encode_ms=round(self._encode_ms, 2),
                decode_ms=round(self._decode_ms, 2),
                network_rtt_ms=round(self._rtt_ms, 2),
                queue_size=self._queue_size,
                uptime_s=round(time.monotonic() - self._start, 2),
            )

    def to_dict(self) -> dict:
        return self.snapshot().__dict__

    def reset(self) -> None:
        with self._lock:
            self._frames_in = self._frames_out = self._dropped = 0
```

**Step 4:** Run to verify it passes.

**Step 5:** Commit.
```bash
git add shared/metrics.py tests/test_metrics.py
git commit -m "feat(shared): thread-safe metrics collector"
```

---

## Phase 1 — Deep-Live-Cam Standalone

### Task 6: `scripts/setup-dlc.sh` — clone DLC, models, verify `run.py`

Manual/hardware task (no unit tests). Server-side machine (desktop with RX 7900 XTX).

**Files:**
- Create: `scripts/setup-dlc.sh`

**Step 1:** Write `scripts/setup-dlc.sh`:
```bash
#!/usr/bin/env bash
# Clone Deep-Live-Cam and download models; verify headless CLI swap.
# Run on the SERVER (GPU) machine. Idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DLC_DIR="$REPO_ROOT/third_party/Deep-Live-Cam"
MODELS_DIR="$DLC_DIR/models"
VENV="$REPO_ROOT/.venv"

if [ ! -d "$DLC_DIR" ]; then
  git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git "$DLC_DIR"
fi

mkdir -p "$MODELS_DIR"
# Models (GFPGANv1.4 optional, needed only for enhancer)
if [ ! -f "$MODELS_DIR/inswapper_128.onnx" ]; then
  curl -L -o "$MODELS_DIR/inswapper_128.onnx" \
    https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx
fi
if [ ! -f "$MODELS_DIR/inswapper_128_fp16.onnx" ]; then
  curl -L -o "$MODELS_DIR/inswapper_128_fp16.onnx" \
    https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx
fi

# Install DLC deps into the shared venv (customtkinter, insightface, onnxruntime, torch...)
source "$VENV/bin/activate"
pip install -r "$DLC_DIR/requirements.txt"

# ROCm GPU inference: pin the ORT<->ROCm pair to the version the DLC
# ecosystem ships. facefusion (the maintained fork) pins
# onnxruntime-rocm==1.22.2.post1 with ROCm 6.x. For ROCm 7.x use
# onnxruntime-migraphx instead. Do NOT let DLC's requirements.txt pull in
# a random onnxruntime afterwards.
if [ "${OR_ROCM:-1}" = "1" ]; then
  pip install "onnxruntime-rocm==1.22.2.post1" 2>/dev/null || \
  pip install "onnxruntime-migraphx"            || \
  echo "WARNING: no ROCm/MIGraphX onnxruntime installed - will fall back to CPU"
fi

echo "DLC installed. Test with:"
echo "  source $VENV/bin/activate"
echo "  python $DLC_DIR/run.py -s /path/to/source.jpg -t /path/to/target.jpg -o /tmp/out.jpg"
```

**Step 2:** Make executable and run.
```bash
chmod +x scripts/setup-dlc.sh
./scripts/setup-dlc.sh
```
Expected: clone succeeds, models download (~300 MB), pip install completes, ROCm provider installed.

**Step 3: Verify Phase 1 deliverable** (headless, no GUI):
```bash
source .venv/bin/activate
python third_party/Deep-Live-Cam/run.py -s <source_face.jpg> -t <target.jpg> -o /tmp/out.jpg
```
Expected: prints `Output image saved to: /tmp/out.jpg`; open `/tmp/out.jpg` and confirm the face is swapped. (The repo's CLI mode is triggered by `-s`; no window opens.)

**Step 4 (AMD GPU, REQUIRED for the latency target):** verify the provider is actually available **and** genuinely active — RDNA3's #1 failure mode is ROCm reporting available but silently falling back to CPU:
```bash
python - <<'PY'
import onnxruntime as ort, numpy as np
print("available:", ort.get_available_providers())
sess = ort.InferenceSession("third_party/Deep-Live-Cam/models/inswapper_128.onnx",
                            providers=ort.get_available_providers())
print("active providers:", sess.get_providers())  # MUST start with ROCMExecutionProvider
PY
```
Expected: `active providers` lists `ROCMExecutionProvider` first. If it falls back to `CPUExecutionProvider`, the ORT↔ROCm versions don't match — pin the exact pair (ROCm 6.x → `onnxruntime-rocm==1.22.2.post1`; ROCm 7.x → `onnxruntime-migraphx`). Pre-warm the MIOpen kernel cache once with a real inference so first-frame JIT doesn't stutter:
```bash
python scripts/bench.py /path/to/source.jpg 5
```
If no GPU provider works at all, fall back to `cpu` and target 30 fps at 720p (documented ceiling in README).

**Step 5:** Commit.
```bash
git add scripts/setup-dlc.sh
git commit -m "feat(scripts): dlc setup and verification"
```

---

## Phase 2 — Headless Inference Module

### Task 7: `server/deeplivecam.py` — `process_frame(frame) -> frame`

The server-facing wrapper. It imports Deep-Live-Cam as a library **without** triggering the GUI by pre-inserting a stub for `modules.ui` into `sys.modules` before any `modules.*` import. It is **control-plane driven**: the source face and swap settings arrive at runtime from the laptop (`set_source_face`, `update_settings`); until a face is set, `process_frame(frame)` passes frames through unchanged. It caches face detection (`target_face`) across frames for the latency target.

**Files:**
- Create: `server/deeplivecam.py`
- Test: `tests/test_deeplivecam.py`

**Step 1: Write the failing test** — uses a **fake** processor injected so no DLC install is required for CI:
```python
import cv2
import numpy as np
from server.deeplivecam import DeepLiveCamProcessor


def face_img_bytes() -> bytes:
    img = np.full((200, 200, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class FakeFrameProc:
    def __init__(self, name):
        self.name = name
        self.processed_with_source = False

    def process_frame(self, source_face, frame, target_face=None):
        if source_face is not None:
            self.processed_with_source = True
        return frame


class FakeDLC:
    """Stand-in for the real modules.* functions; records calls."""

    def __init__(self):
        self.calls = []
        self.fp_chain = []

    def get_one_face(self, frame):
        self.calls.append(("detect", frame.shape))
        return "target-face"

    def make_frame_processors(self, names):
        self.fp_chain = [FakeFrameProc(n) for n in names]
        return self.fp_chain


def test_no_face_set_passes_through_unchanged():
    p = DeepLiveCamProcessor({}, dlc=FakeDLC())
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert p.process_frame(frame) is frame  # same object: no swap attempted


def test_set_source_face_enables_processing():
    fake = FakeDLC()
    p = DeepLiveCamProcessor({}, dlc=fake)
    assert p.set_source_face(face_img_bytes()) is True
    out = p.process_frame(np.zeros((64, 64, 3), np.uint8))
    assert out.shape == (64, 64, 3)
    assert any(fp.processed_with_source for fp in fake.fp_chain)


def test_detection_cached_by_interval():
    fake = FakeDLC()
    p = DeepLiveCamProcessor({}, dlc=fake)
    p.update_settings({"detection_interval_ms": 1000})
    p.set_source_face(face_img_bytes())
    for _ in range(5):
        p.process_frame(np.zeros((720, 1280, 3), np.uint8))
    detects = [c for c in fake.calls if c[0] == "detect"]
    assert len(detects) == 1  # only first frame triggered detection


def test_no_face_in_frame_still_returns_frame():
    class NoFace(FakeDLC):
        def get_one_face(self, frame):
            return None

    p = DeepLiveCamProcessor({}, dlc=NoFace())
    p.set_source_face(face_img_bytes())
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert p.process_frame(frame).shape == frame.shape


def test_update_settings_rebuilds_processors():
    fake = FakeDLC()
    p = DeepLiveCamProcessor({}, dlc=fake)
    p.set_source_face(face_img_bytes())
    p.update_settings({"frame_processors": ["face_swapper", "face_enhancer"],
                       "many_faces": True, "opacity": 0.8})
    assert [fp.name for fp in fake.fp_chain] == ["face_swapper", "face_enhancer"]
    assert p.many_faces is True
```

**Step 2:** Run to verify it fails. Expected: ModuleNotFoundError.

**Step 3: Implement** (`server/deeplivecam.py`):
```python
import logging
import sys
import time
from types import ModuleType

import cv2
import numpy as np

logger = logging.getLogger("server.deeplivecam")

# Headless trick: every DLC module that touches the GUI imports
# 'modules.ui' (customtkinter). Stub it out BEFORE importing anything
# from Deep-Live-Cam so the server never needs tkinter.
_UI_STUB = ModuleType("modules.ui")
_UI_STUB.update_status = lambda *a, **k: None
if "modules.ui" not in sys.modules:
    sys.modules["modules.ui"] = _UI_STUB


def _select_providers(provider: str) -> list:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "auto":
        for pref in ("cuda", "rocm", "migraphx", "coreml", "openvino", "dml"):
            if any(pref.lower() in p.lower() for p in available):
                return [p for p in available if pref.lower() in p.lower()]
        return ["CPUExecutionProvider"]
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    return [p for p in available if provider.lower() in p.lower()] or ["CPUExecutionProvider"]


class DeepLiveCamProcessor:
    """Headless wrapper exposing process_frame(frame) -> frame.

    The laptop is the control point: the source face (JPEG/PNG bytes) and
    the swap settings arrive at runtime via set_source_face() and
    update_settings() over the signaling channel. This class holds NO face
    config of its own; until a source face is set, process_frame() passes
    frames through unchanged.

    dlc is a dependency-injection hook (tests inject a fake). When dlc is
    None, the real Deep-Live-Cam modules are imported lazily from
    third_party/Deep-Live-Cam (added to sys.path).
    """

    def __init__(self, cfg: dict, dlc=None, dlc_repo_path: str = "third_party/Deep-Live-Cam"):
        proc_cfg = (cfg or {}).get("processing", {})
        self._detection_interval_ms = 66
        self._last_detect_time = 0.0
        self._target_face = None
        self._source_face = None
        self._many_faces = False
        self._frame_processors = []
        self._dlc = dlc

        if dlc is not None:
            self._get_one_face = dlc.get_one_face
            self._dlc_make_processors = dlc.make_frame_processors
            self._globals = None
            return

        # ---- real DLC wiring (server runtime) ----
        if dlc_repo_path not in sys.path:
            sys.path.insert(0, dlc_repo_path)
        import modules.globals as g

        g.headless = True
        g.execution_providers = _select_providers(proc_cfg.get("execution_provider", "auto"))
        g.max_memory = 16

        from modules.face_analyser import get_one_face
        from modules.processors.frame.core import get_frame_processors_modules

        self._get_one_face = get_one_face
        self._get_frame_processors_modules = get_frame_processors_modules
        self._globals = g
        logger.info("DLC ready: providers=%s", g.execution_providers)

    # ---- control-plane API (called from the signaling handler) ----

    def set_source_face(self, image_bytes: bytes) -> bool:
        """Decode a source face image sent by the laptop and store its
        embedding. Returns True on success, False if undecodable or no face."""
        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cannot decode source face image (%d bytes)", len(image_bytes))
            return False
        face = self._get_one_face(img)
        if face is None:
            logger.warning("no face detected in source image")
            return False
        self._source_face = face
        logger.info("source face set")
        return True

    def update_settings(self, settings: dict) -> None:
        """Apply swap settings from the laptop (hot-swappable)."""
        names = list(settings.get("frame_processors", ["face_swapper"]))
        if settings.get("enhancer") and "face_enhancer" not in names:
            names = names + ["face_enhancer"]
        self._many_faces = bool(settings.get("many_faces", False))
        self._detection_interval_ms = int(settings.get("detection_interval_ms", 66))
        if self._dlc is not None:
            self._frame_processors = self._dlc_make_processors(names)
            return
        g = self._globals
        g.frame_processors = names
        g.many_faces = self._many_faces
        g.mouth_mask = bool(settings.get("mouth_mask", False))
        g.poisson_blend = bool(settings.get("poisson_blend", False))
        g.opacity = float(settings.get("opacity", 1.0))
        g.sharpness = 0.0
        g.enable_interpolation = False
        self._frame_processors = self._get_frame_processors_modules(names)

    def _refresh_target_face(self, frame: np.ndarray) -> None:
        now = time.monotonic() * 1000
        if self._detection_interval_ms <= 0 or self._target_face is None \
                or (now - self._last_detect_time) >= self._detection_interval_ms:
            self._target_face = self._get_one_face(frame)
            self._last_detect_time = now

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Swap face(s) on a single BGR frame. Passes through unchanged until
        a source face is set. Never raises for media errors."""
        if frame is None or frame.size == 0:
            return frame
        if self._source_face is None or not self._frame_processors:
            return frame
        self._refresh_target_face(frame)
        processed = frame
        for fp in self._frame_processors:
            try:
                processed = fp.process_frame(self._source_face, processed, target_face=self._target_face)
            except TypeError:
                processed = fp.process_frame(self._source_face, processed)
            except Exception as exc:  # keep the stream alive on model hiccups
                logger.warning("frame processor error: %s", exc)
        return processed if processed is not None else frame

    def close(self) -> None:
        try:
            g = self._globals
            if g is None or "ROCMExecutionProvider" not in g.execution_providers:
                return
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
```

**Step 4:** Run to verify it passes.
```bash
pytest tests/test_deeplivecam.py -q
```
Expected: 5 passed.

**Step 5:** Add a **smoke test** that the real DLC import works in the venv (guards against the tkinter stub breaking):
```bash
source .venv/bin/activate
python - <<'PY'
import sys; sys.path.insert(0, "third_party/Deep-Live-Cam")
from server.deeplivecam import DeepLiveCamProcessor
p = DeepLiveCamProcessor({})
p.update_settings({"frame_processors": ["face_swapper"]})
print("OK, providers:", p._globals.execution_providers)
PY
```
Expected: prints `OK, providers: [...]` with no tkinter error.

**Step 6:** Commit.
```bash
git add server/deeplivecam.py tests/test_deeplivecam.py
git commit -m "feat(server): control-plane-driven headless dlc wrapper"
```

---

## Phase 3 — WebRTC Plumbing (no processing)

### Task 8: `server/tracks.py` — inbound + processed video tracks

**Files:**
- Create: `server/tracks.py`
- Test: `tests/test_tracks.py`

**Step 1: Write the failing test** — conversion helpers and latest-frame-resend semantics, using the av/aiortc loop:
```python
import asyncio
import numpy as np
import pytest
from av import VideoFrame
from server.tracks import frame_to_bgr, bgr_to_frame, ProcessedVideoTrack
from shared.frame_holder import FrameHolder


def test_ndarray_roundtrip():
    arr = np.zeros((720, 1280, 3), np.uint8)
    arr[:, :, 0] = 7  # blue plane
    vf = bgr_to_frame(arr)
    assert vf.width == 1280 and vf.height == 720
    out = frame_to_bgr(vf)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8
    assert (out == arr).all()


@pytest.mark.asyncio
async def test_processed_track_waits_then_resends_latest():
    holder = FrameHolder()
    track = ProcessedVideoTrack(holder)
    first = await asyncio.wait_for(track.recv(), timeout=0.3)
    holder.put(np.zeros((64, 64, 3), np.uint8))
    second = await asyncio.wait_for(track.recv(), timeout=0.3)
    assert first.width == 640  # no data yet -> place holder frame behavior TBD
    assert second.width == 64
    await track.stop()
```

(If the initial-wait behavior is "block until first frame", the test above should instead wait on a frame put *before* recv — see Step 3; adjust the test to match the chosen contract: **recv() blocks until the first processed frame, then returns the newest immediately, never queueing.** Write the test to assert that contract.)

**Step 2:** Run to verify it fails.

**Step 3: Implement** (`server/tracks.py`):
```python
import asyncio
import time

import numpy as np
from aiortc import MediaStreamError, VideoStreamTrack
from av import VideoFrame

from shared.frame_holder import FrameHolder


def frame_to_bgr(video_frame: VideoFrame) -> np.ndarray:
    return video_frame.to_ndarray(format="bgr24")


def bgr_to_frame(bgr: np.ndarray, pts: int | None = None, time_base=None) -> VideoFrame:
    vf = VideoFrame.from_ndarray(bgr, format="bgr24")
    if pts is not None and time_base is not None:
        vf.pts = pts
        vf.time_base = time_base
    return vf


class InboundVideoTrack:
    """Feeds received WebRTC frames into a FrameHolder for processing.

    Not a MediaStreamTrack subclass: it wraps the remote track aiortc gives
    us and drains it, so recv() keeps the RTP flow healthy.
    """

    def __init__(self, remote_track: VideoStreamTrack, holder: FrameHolder, metrics=None) -> None:
        self._track = remote_track
        self._holder = holder
        self._metrics = metrics
        self._decode_ms = 0.0
        self._task = None

    async def run(self) -> None:
        async def _pump():
            async for frame in self._track:
                t0 = time.perf_counter()
                bgr = frame_to_bgr(frame)
                self._decode_ms = (time.perf_counter() - t0) * 1000
                self._holder.put(bgr)
                if self._metrics is not None:
                    self._metrics.record_incoming()
                    self._metrics.record_decode_ms(self._decode_ms)

        try:
            await _pump()
        except (MediaStreamError, ConnectionError):
            pass

    async def start(self) -> None:
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class ProcessedVideoTrack(VideoStreamTrack):
    """Outbound track: returns the newest processed frame; duplicates the
    latest if no new frame has arrived (keeps the stream alive at cadence)."""

    def __init__(self, holder: FrameHolder, poll_ms: int = 2, metrics=None) -> None:
        super().__init__()
        self._holder = holder
        self._poll = poll_ms / 1000
        self._metrics = metrics
        self._last = None

    async def recv(self) -> VideoFrame:
        while True:
            frame, seq = self._holder.get()
            if frame is not None and seq != self._last:
                self._last = seq
                break
            await asyncio.sleep(self._poll)

        t0 = time.perf_counter()
        pts, time_base = await self.next_timestamp()
        vf = bgr_to_frame(frame, pts, time_base)
        if self._metrics is not None:
            self._metrics.record_encode_ms((time.perf_counter() - t0) * 1000)
            self._metrics.record_outgoing()
        return vf
```

**Step 4:** Run to verify it passes. Fix the initial-wait contract test: `recv()` with an empty holder polls until a frame appears (the test must put a frame before expecting data).

**Step 5:** Commit.
```bash
git add server/tracks.py tests/test_tracks.py
git commit -m "feat(server): inbound/processed video tracks + frame conversion"
```

---

### Task 9: `server/processor.py` — worker thread + newest-wins

The processor runs Deep-Live-Cam inference on a dedicated thread so the asyncio event loop never blocks. It reads from an inbound `FrameHolder` and writes to an outbound `FrameHolder`, both drop-old.

**Files:**
- Create: `server/processor.py`
- Test: `tests/test_processor.py`

**Step 1: Write the failing test** (`tests/test_processor.py`):
```python
import threading
import time

import numpy as np

from server.processor import Processor


class SlowFake:
    def __init__(self):
        self.calls = 0

    def process_frame(self, frame):
        self.calls += 1
        time.sleep(0.02)  # 50 fps ceiling; we feed faster
        return frame


def test_newest_frame_wins_when_processing_is_slow():
    fake = SlowFake()
    p = Processor(fake)
    p.start()
    try:
        # feed 10 frames quickly; processor only gets ~2
        for i in range(10):
            p.feed(np.full((64, 64, 3), i, np.uint8))
        time.sleep(0.15)
        out = p.latest_out()
        assert out is not None
        # the latest fed frame must have won (or an older one if it was mid-inference)
        assert p.metrics.snapshot().frames_in >= 10
        assert fake.calls <= 10
        assert fake.calls >= 1
    finally:
        p.stop()


def test_dropped_counter_increments():
    fake = SlowFake()
    p = Processor(fake)
    p.start()
    try:
        for i in range(5):
            p.feed(np.zeros((64, 64, 3), np.uint8))
        time.sleep(0.12)
        assert p.metrics.snapshot().dropped >= 1
    finally:
        p.stop()
```

**Step 2:** Run to verify it fails.

**Step 3: Implement** (`server/processor.py`):
```python
import logging
import threading
import time

from shared.frame_holder import FrameHolder
from shared.metrics import Metrics

logger = logging.getLogger("server.processor")


class Processor:
    """Runs a blocking process_frame on a worker thread.

    feed() -> inbound holder (drop-old) -> worker -> outbound holder
    (drop-old). inference_ms is recorded per frame.
    """

    def __init__(self, deeplivecam, poll_ms: int = 50) -> None:
        self._dlc = deeplivecam
        self._in = FrameHolder()
        self._out = FrameHolder()
        self._metrics = Metrics()
        self._poll = poll_ms / 1000
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dlc-processor", daemon=True)
        self._last_seq = None

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=1.0)

    def feed(self, bgr_frame) -> None:
        self._in.put(bgr_frame)
        self._metrics.record_incoming(queue_size=1 if self._in.get() is not None else 0)

    def latest_out(self):
        return self._out.get()

    def apply_control(self, payload: dict) -> bool:
        """Apply a laptop control message (face + settings) to the
        underlying DeepLiveCamProcessor. Returns True on success."""
        ok = True
        face_bytes = payload.get("face")
        if face_bytes is not None:
            import base64
            raw = base64.b64decode(face_bytes) if isinstance(face_bytes, str) else face_bytes
            ok = self._dlc.set_source_face(raw)
        settings = payload.get("settings")
        if settings:
            self._dlc.update_settings(settings)
        return ok

    def _run(self) -> None:
        last_seq = None
        while not self._stop_evt.is_set():
            frame, seq = self._in.get_new(last_seq, timeout=self._poll)
            if frame is None:
                continue
            last_seq = seq
            t0 = time.perf_counter()
            try:
                out = self._dlc.process_frame(frame)
            except Exception as exc:  # never kill the worker
                logger.exception("inference failed: %s", exc)
                out = frame
            dt_ms = (time.perf_counter() - t0) * 1000
            self._metrics.record_inference_ms(dt_ms)
            if out is not None:
                self._out.put(out)
```

**Step 4:** Run to verify it passes (tune sleeps if flaky; assertion `dropped >= 1` must hold because 10 fast feeds while processing takes 20 ms each guarantees drops).

**Step 5:** Commit.
```bash
git add server/processor.py tests/test_processor.py
git commit -m "feat(server): threaded newest-wins processor"
```

---

### Task 10: `server/rtc.py` + `server/app.py` — session & signaling

The FastAPI app exposes:
- `GET /` — JSON status (uptime, sessions).
- `GET /health` — `{"ok": true}`.
- `GET /metrics` — metrics snapshot (populated in Phase 7; returns zeros until then).
- `WS /ws/{client_id}` — signaling: client sends `offer`, server answers with `answer`. Non-trickle: both sides wait for ICE gathering to complete before serializing SDP.

**Files:**
- Create: `server/rtc.py`, `server/app.py`

**Step 1:** Implement `server/rtc.py`:
```python
import asyncio
import json
import logging

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from server.tracks import InboundVideoTrack, ProcessedVideoTrack

logger = logging.getLogger("server.rtc")


async def _gather_complete(pc: RTCPeerConnection) -> None:
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.02)


class Session:
    """One client session: receive webcam, process, send back on one PC."""

    def __init__(self, client_id: str, processor) -> None:
        self.client_id = client_id
        self.processor = processor
        self.pc = RTCPeerConnection()
        self._inbound = None
        self._processed_track: VideoStreamTrack = ProcessedVideoTrack(
            FrameOut(processor)  # see note below
        )
        self.pc.addTrack(self._processed_track)

        @self.pc.on("track")
        async def on_track(track: VideoStreamTrack):
            if track.kind == "video":
                self._inbound = InboundVideoTrack(track, processor_input_holder, metrics)
                await self._inbound.start()

        @self.pc.on("connectionstatechange")
        async def on_state():
            logger.info("session %s state=%s", self.client_id, self.pc.connectionState)
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

    async def handle(self, msg: dict) -> dict:
        mtype = msg.get("type")
        if mtype == "control":
            # laptop -> server: set source face + swap settings
            payload = msg.get("payload", {})
            ok = self.processor.apply_control(payload)
            if not ok:
                return {"type": "error", "message": "control rejected: no face detected in source image"}
            return {"type": "ack"}
        if mtype == "ping":
            return {"type": "pong"}
        if mtype == "offer":
            await self.pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type="offer")
            )
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
```

> **Note on wiring:** to keep `rtc.py` decoupled, the session is constructed by `app.py` with concrete objects: an inbound `FrameHolder`, an outbound `FrameHolder` (fed by the processor), and the `Processor`. `ProcessedVideoTrack(out_holder)` reads the outbound holder; `InboundVideoTrack(remote_track, in_holder)` writes the inbound holder. The sketch above compresses this; implement `app.py` to own the holders and pass them in (no magic). `Processor.feed` is called from the inbound pump via the holder — i.e. inbound track → `in_holder`, processor consumes `in_holder` and fills `out_holder`, processed track reads `out_holder`.

**Step 2:** Implement `server/app.py`:
```python
import asyncio
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


def create_app(cfg: ServerConfig, processor_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        logger.info("server starting (providers configured at first session)")
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
        return Processor(DeepLiveCamProcessor(cfg.model_dump()))

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
        session = Session(client_id, proc)
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
        except Exception as exc:
            logger.warning("session %s error: %s", client_id, exc)
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            except Exception:
                pass
        finally:
            await session.close()
            SESSIONS.pop(client_id, None)

    return app


if __name__ == "__main__":
    import uvicorn

    cfg = ServerConfig.from_yaml("configs/server.yaml") if hasattr(ServerConfig, "from_yaml") else ServerConfig()
    uvicorn.run(create_app(cfg), host=cfg.server.get("host", "0.0.0.0"),
                port=int(cfg.server.get("port", 8000)))
```

(Add `ServerConfig.from_yaml` classmethod in `shared/config.py` in Phase 8; for now the `__main__` guard uses `ServerConfig()` defaults.)

**Step 3:** Start the server and verify endpoints.
```bash
source .venv/bin/activate
python -m server.app
```
Then in a second shell:
```bash
curl -s localhost:8000/health        # {"ok": true}
curl -s localhost:8000/              # {"app": "remote-deeplivecam", "sessions": 0}
```
Expected: JSON responses as shown.

**Step 4:** Commit.
```bash
git add server/rtc.py server/app.py
git commit -m "feat(server): fastapi app, ws signaling, session handling"
```

---

### Task 11: `client/camera.py` + `client/tracks.py`

**Files:**
- Create: `client/camera.py`, `client/tracks.py`

**Step 1:** Implement `client/camera.py`:
```python
import logging
import threading

import cv2

from shared.frame_holder import FrameHolder

logger = logging.getLogger("client.camera")


class CameraCapture:
    """OpenCV webcam capture on a background thread; newest frame wins."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 60) -> None:
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._holder = FrameHolder()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        if not self._cap.isOpened():
            raise RuntimeError("cannot open camera")
        self._thread.start()
        logger.info("camera opened: %s", self._cap)

    def latest(self):
        return self._holder.get()

    def stop(self) -> None:
        self._stop.set()
        self._cap.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            if frame is not None:
                self._holder.put(frame)
```

**Step 2:** Implement `client/tracks.py`:
```python
import asyncio
import logging

import numpy as np
from aiortc import MediaStreamError, VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger("client.tracks")


class CameraVideoTrack(VideoStreamTrack):
    """Publishes the latest camera frame to WebRTC."""

    def __init__(self, camera, poll_ms: int = 2) -> None:
        super().__init__()
        self._camera = camera
        self._poll = poll_ms / 1000

    async def recv(self) -> VideoFrame:
        while True:
            frame = self._camera.latest()
            if frame is not None:
                break
            await asyncio.sleep(self._poll)
        pts, time_base = await self.next_timestamp()
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf


class ReceivingVideoTrack:
    """Wraps the remote (processed) track; pushes BGR frames to a holder."""

    def __init__(self, remote_track: VideoStreamTrack, holder, metrics=None) -> None:
        self._track = remote_track
        self._holder = holder
        self._metrics = metrics
        self._task = None

    async def run(self) -> None:
        async def _pump():
            async for frame in self._track:
                bgr = frame.to_ndarray(format="bgr24")
                self._holder.put(bgr)
                if self._metrics is not None:
                    self._metrics.record_incoming()

        try:
            await _pump()
        except (MediaStreamError, ConnectionError):
            pass

    async def start(self) -> None:
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
```

**Step 3:** Add a quick sanity test that a synthetic frame flows through `CameraVideoTrack.recv()` without a webcam (`tests/test_tracks.py` — reuse the FrameHolder pattern from the server test):
```python
import asyncio
import numpy as np
import pytest
from client.tracks import CameraVideoTrack
from shared.frame_holder import FrameHolder


class FakeCam:
    def __init__(self):
        self._h = FrameHolder()
        self._h.put(np.zeros((128, 128, 3), np.uint8))

    def latest(self):
        return self._h.get()


@pytest.mark.asyncio
async def test_camera_track_recv_returns_videoframe():
    track = CameraVideoTrack(FakeCam(), poll_ms=1)
    vf = await asyncio.wait_for(track.recv(), timeout=0.5)
    assert vf.width == 128 and vf.height == 128
    await track.stop()
```

**Step 4:** Run tests.
```bash
pytest tests/test_tracks.py -q
```
Expected: all pass.

**Step 5:** Commit.
```bash
git add client/camera.py client/tracks.py tests/test_tracks.py
git commit -m "feat(client): camera capture thread and webcam/recv tracks"
```

---

### Task 12: `client/webrtc.py` — offerer + signaling client

**Files:**
- Create: `client/webrtc.py`

**Step 1:** Implement:
```python
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


async def _gather_complete(pc: RTCPeerConnection) -> None:
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.02)


class WebRTCClient:
    """Client peer. Offers a camera track; receives a processed track.
    Control point: sends the source face + swap settings at connect and on
    every change (hot-swap)."""

    def __init__(self, config, camera, out_holder: FrameHolder) -> None:
        self.config = config
        self.camera = camera
        self.out_holder = out_holder      # where received processed frames go
        self.metrics = Metrics()
        self.pc = RTCPeerConnection()
        self._receiver = None
        self._ws = None
        self.negotiated = {}
        self._connected = asyncio.Event()
        from client.tracks import CameraVideoTrack, ReceivingVideoTrack
        self.pc.addTrack(CameraVideoTrack(camera))

        @self.pc.on("track")
        def on_track(track):
            from client.tracks import ReceivingVideoTrack
            self._receiver = ReceivingVideoTrack(track, self.out_holder, self.metrics)
            asyncio.create_task(self._receiver.start())
            logger.info("received remote track: %s", track.kind)

        @self.pc.on("connectionstatechange")
        async def on_state():
            if self.pc.connectionState == "connected":
                self._connected.set()
            elif self.pc.connectionState == "failed":
                logger.error("connection failed")

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

    async def _send_control(self) -> None:
        await self._ws.send(Signal.control(self._control_payload()).encode())
        # wait for the ack (or error) so a bad face fails fast
        while True:
            msg = Signal.decode(await self._ws.recv())
            if msg.type == "ack":
                return
            if msg.type == "error":
                raise RuntimeError(msg.data.get("message", "control rejected"))

    async def connect(self) -> None:
        url = self.config.server_url.replace("http", "ws", 1)
        async with websockets.connect(url) as ws:
            self._ws = ws
            # 1) read negotiated config from the server
            while True:
                msg = Signal.decode(await ws.recv())
                if msg.type == "config":
                    self.negotiated = msg.data.get("payload", {})
                    break
            # 2) control plane: face + settings (laptop is the control point)
            await self._send_control()
            # 3) media: offer camera track
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            await _gather_complete(self.pc)
            await ws.send(json.dumps({"type": "offer", "sdp": offer.sdp}))
            # 4) keep reading signaling; media runs in background tasks
            while not self._connected.is_set() or self._receiver is None:
                msg = Signal.decode(await ws.recv())
                if msg.type == "answer":
                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=msg.data["sdp"], type="answer")
                    )
                elif msg.type == "error":
                    raise RuntimeError(msg.data.get("message", "server error"))
                elif msg.type == "metrics":
                    self.metrics.record_rtt_ms(float(
                        msg.data.get("snapshot", {}).get("network_rtt_ms", 0)))
            # keep the signaling socket open for control/ping
            await ws.wait_closed()

    async def set_face(self, path: str) -> None:
        """Hot-swap the source face at runtime."""
        self.config.processing.source_face = path
        await self._send_control()

    async def update_settings(self, settings: dict) -> None:
        await self._ws.send(Signal.control({"face": "", "settings": settings}).encode())

    async def close(self) -> None:
        if self._receiver:
            await self._receiver.stop()
        if self._ws:
            await self._ws.close()
        await self.pc.close()
```

> **Note:** `connect()` holds the WS open (after media flows) so `set_face`/`update_settings` can push control messages later; `main.py` drives shutdown on keyboard interrupt. Keep the single-PC, non-trickle contract.

**Step 2:** Static check.
```bash
source .venv/bin/activate
python -m py_compile client/webrtc.py client/camera.py client/tracks.py
```
Expected: exit 0.

**Step 3:** Commit.
```bash
git add client/webrtc.py
git commit -m "feat(client): webrtc offerer + signaling client"
```

---

### Task 13: `tests/test_rtc_loopback.py` — full in-process WebRTC video flow

Validates the entire Phase-3 plumbing (both directions) without a webcam, GPU, or network, using two `RTCPeerConnection`s on one event loop and non-trickle SDP exchange.

**Files:**
- Create: `tests/test_rtc_loopback.py`

**Step 1: Write the test:**
```python
import asyncio

import numpy as np
import pytest

from aiortc import RTCPeerConnection, RTCSessionDescription

from server.processor import Processor
from server.tracks import ProcessedVideoTrack, InboundVideoTrack
from client.tracks import CameraVideoTrack
from shared.frame_holder import FrameHolder

from tests.fake_dlc import EchoDLC  # returns the input frame unchanged


async def _gather(pc):
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_video_flows_both_ways():
    # server side
    server_pc = RTCPeerConnection()
    in_holder = FrameHolder()
    out_holder = FrameHolder()
    server_processor = Processor(EchoDLC())
    server_processor.start()

    received_on_server = []

    @server_pc.on("track")
    async def on_track(track):
        def collect():
            async def _pump():
                async for f in track:
                    received_on_server.append(f.to_ndarray(format="bgr24").copy())
            asyncio.create_task(_pump())
        collect()

    processed_track = ProcessedVideoTrack(out_holder)
    server_pc.addTrack(processed_track)

    # client side
    client_pc = RTCPeerConnection()
    cam_holder = FrameHolder()
    received_on_client = []

    @client_pc.on("track")
    async def on_track(track):
        async def _pump():
            async for f in track:
                received_on_client.append(f.to_ndarray(format="bgr24").copy())
        asyncio.create_task(_pump())

    client_cam = CameraVideoTrack(FakeCamSource(cam_holder))
    client_pc.addTrack(client_cam)

    # non-trickle offer/answer
    offer = await client_pc.createOffer()
    await client_pc.setLocalDescription(offer)
    await _gather(client_pc)
    await server_pc.setRemoteDescription(client_pc.localDescription)  # post-gather! contains candidates
    answer = await server_pc.createAnswer()
    await server_pc.setLocalDescription(answer)
    await _gather(server_pc)
    await client_pc.setRemoteDescription(server_pc.localDescription)

    # pump frames for ~1s
    async def feed():
        for i in range(60):
            cam_holder.put(np.full((64, 64, 3), i, np.uint8))
            await asyncio.sleep(0.016)

    await asyncio.wait_for(asyncio.gather(feed(), _drain(server_pc, out_holder, received_on_client, received_on_server)), timeout=5)

    await client_pc.close()
    await server_pc.close()
    server_processor.stop()

    assert len(received_on_server) > 0, "server received no webcam frames"
    assert len(received_on_client) > 0, "client received no processed frames"
    # echo path: client frames arrived unchanged on the client's receive side
    assert received_on_client[0].shape == (64, 64, 3)
```

(Define `tests/fake_dlc.py` with `class EchoDLC: def process_frame(self, frame): return frame` and a `FakeCamSource` that reads `cam_holder`. Wire the `Processor` to the server side: the processor consumes `in_holder` and feeds `out_holder` — the real `InboundVideoTrack` is exercised by running its `run()` against the incoming remote track instead of the simplified `collect()` above. Implementation detail: run `InboundVideoTrack(track, in_holder)` in a task and let the `Processor` bridge `in_holder → out_holder`.)

**Step 2:** Run to verify it fails/needs the wiring, then make it pass by completing the processor bridge (Task 9's `Processor` already reads `in_holder`/writes `out_holder`; add `server_processor.feed()` calls — or better, have `InboundVideoTrack.run()` call `processor.feed(bgr)` directly via a callback so the loopback test exercises the real path).

**Step 3:** Confirm green.
```bash
pytest tests/test_rtc_loopback.py -v
```
Expected: PASS — frames flow client→server→client on one machine.

**Step 4:** Commit.
```bash
git add tests/fake_dlc.py tests/test_rtc_loopback.py
git commit -m "test(rtc): in-process loopback video flow"
```

---

## Phase 4 — Insert Deep-Live-Cam Processing

### Task 14: Wire the real processor into the session

**Files:**
- Modify: `server/app.py` (already builds `Processor(DeepLiveCamProcessor(cfg))`), `server/rtc.py`
- No new code is required beyond connecting `InboundVideoTrack` → `processor.feed` → outbound holder → `ProcessedVideoTrack`.

**Step 1:** In `server/rtc.py`, change the inbound wiring so each received frame is pushed into the processor:
```python
self._inbound = InboundVideoTrack(track, holder=processor_in_holder, metrics=processor.metrics)
```
and have `Processor.feed` be the single ingestion point (already implemented). Ensure `Processor.start()` is called in `app.py` before media arrives.

**Step 2:** Manual verification on the server machine with a real source face:
```bash
source .venv/bin/activate
python -m server.app
```
Then from the client machine, run the client (Task 16) and confirm the virtual camera shows a swapped face. Until Task 16 exists, verify with the loopback test only.

**Step 3:** Commit.
```bash
git add server/rtc.py server/app.py
git commit -m "feat(server): wire dlc processor into session pipeline"
```

---

## Phase 5 — Client Virtual Camera

### Task 15: `client/virtualcam.py` — pyvirtualcam writer

**Files:**
- Create: `client/virtualcam.py`
- Test: `tests/test_virtualcam.py`

**Step 1: Write the failing test** — uses a fake camera backend so CI needs no v4l2loopback:
```python
import threading
import time

import numpy as np

from client.virtualcam import VirtualCamera
from shared.frame_holder import FrameHolder


class FakeCam:
    def __init__(self):
        self.sent = []
        self.width, self.height, self.fps = 64, 64, 30

    def send(self, frame):
        self.sent.append(frame.copy())

    def sleep_until_next_frame(self):
        time.sleep(0.001)


def test_virtual_camera_writes_latest_frames():
    holder = FrameHolder()
    fake = FakeCam()
    vcam = VirtualCamera(holder, cam=fake)
    t = threading.Thread(target=vcam.run, daemon=True)
    t.start()
    time.sleep(0.02)
    holder.put(np.full((64, 64, 3), 1, np.uint8))
    time.sleep(0.05)
    holder.put(np.full((64, 64, 3), 2, np.uint8))
    time.sleep(0.05)
    vcam.stop()
    t.join(1.0)
    assert len(fake.sent) >= 2
    # newest frame must have been sent
    assert (fake.sent[-1] == 2).all()
```

**Step 2:** Run to verify it fails.

**Step 3: Implement** (`client/virtualcam.py`):
```python
import logging
import threading

import pyvirtualcam
from pyvirtualcam import PixelFormat

from shared.frame_holder import FrameHolder

logger = logging.getLogger("client.virtualcam")


class VirtualCamera:
    """Writes newest frames to a virtual camera device (v4l2loopback)."""

    def __init__(self, holder: FrameHolder, device: str = "/dev/video10",
                 width: int = 1280, height: int = 720, fps: int = 60, cam=None) -> None:
        self._holder = holder
        self._cam = cam or pyvirtualcam.Camera(device=device, width=width,
                                               height=height, fps=fps,
                                               fmt=PixelFormat.BGR)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.run, name="virtualcam", daemon=True)
        logger.info("virtual camera: %s %dx%d@%d", self._cam.device, width, height, fps)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._cam.close()
        except Exception:
            pass

    def run(self) -> None:
        last_seq = None
        try:
            while not self._stop.is_set():
                frame, seq = self._holder.get_new(last_seq, timeout=0.05)
                if frame is not None and seq != last_seq:
                    self._cam.send(frame)
                    last_seq = seq
                    self._cam.sleep_until_next_frame()
        except pyvirtualcam.errors.FrameSendError:
            logger.exception("failed to send frame to virtual camera")
```

**Step 4:** Run to verify it passes.

**Step 5:** Set up the real device and verify on the laptop:
```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepLiveCam" exclusive_caps=1
v4l2-ctl --list-devices   # expect /dev/video10
```
Then a smoke test:
```bash
source .venv/bin/activate
python - <<'PY'
import numpy as np, time, pyvirtualcam
from pyvirtualcam import PixelFormat
with pyvirtualcam.Camera(device="/dev/video10", width=320, height=240, fps=30, fmt=PixelFormat.BGR) as cam:
    for i in range(60):
        cam.send(np.full((240, 320, 3), i * 4, np.uint8))
        cam.sleep_until_next_frame()
print("ok")
PY
```
Open OBS/Zoom/Discord and select "DeepLiveCam" — a gradient image must appear.

**Step 6:** Commit.
```bash
git add client/virtualcam.py tests/test_virtualcam.py
git commit -m "feat(client): pyvirtualcam writer thread"
```

---

### Task 16: `client/main.py` — wiring + entrypoint

**Files:**
- Create: `client/main.py`

**Step 1:** Implement:
```python
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
        await asyncio.wait_for(client.connect(), timeout=30)
        logger.info("connected; streaming to %s", cfg.server_url)
        # periodic metrics log
        while not stop.is_set():
            logger.info("metrics=%s", client.metrics.to_dict())
            await asyncio.wait_for(stop.wait(), timeout=2.0)
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
```

**Step 2:** Add `configs/client.yaml` and `configs/server.yaml` (Phase 8 fills details; stub now):
```yaml
# configs/client.yaml  (the laptop = control point)
client:
  server_url: "ws://192.168.1.50:8000/ws/laptop1"
  client_id: laptop1
camera:
  index: 0
  width: 1280
  height: 720
  fps: 60
virtualcam:
  device: /dev/video10
  width: 1280
  height: 720
  fps: 60
processing:                     # chosen on the laptop, applied on the desktop
  source_face: /home/laptop/faces/me.jpg
  frame_processors: [face_swapper]
  many_faces: false
  enhancer: false
  opacity: 1.0
  mouth_mask: false
  poisson_blend: false
  detection_interval_ms: 66
performance:
  drop_frames: true
```

**Step 3:** Compile-check.
```bash
source .venv/bin/activate
python -m py_compile client/main.py
```

**Step 4:** Commit.
```bash
git add client/main.py configs/client.yaml
git commit -m "feat(client): main entrypoint wiring"
```

---

## Phase 6 — Optimization

### Task 17: Enforce drop-frames policy everywhere

**Files:**
- Modify: `server/processor.py`, `client/tracks.py`, `shared/config.py`

The `FrameHolder` already drops old frames; make it configurable and assert it:
- `shared/config.py`: `PerformanceConfig.drop_frames` (exists). When `drop_frames=True` (default), `Processor.feed()` keeps overwriting (current behavior). When `False` (diagnostic), fail loudly if the inbound holder is non-empty at put time (i.e. raise in `feed`), so backpressure is visible.

**Step 1:** Add the diagnostic branch:
```python
def feed(self, bgr_frame) -> None:
    if not self._drop_frames and self._in.get() is not None:
        logger.warning("drop_frames disabled but holder non-empty; frame would be dropped")
    self._in.put(bgr_frame)
    self._metrics.record_incoming(queue_size=1 if self._in.get() is not None else 0)
```
(Constructor takes `drop_frames: bool = True` from config.)

**Step 2:** Extend `tests/test_processor.py` with a test asserting that when `drop_frames=False`, the latest frame still wins (documented diagnostic behavior) — or that the flag plumbed through. Keep it simple: assert the config value is honored by `Processor`.

**Step 3:** Run tests + commit.
```bash
pytest tests/test_processor.py -q
git add server/processor.py tests/test_processor.py
git commit -m "perf(server): configurable drop-frames policy"
```

### Task 18: Face detection caching (latency critical)

Already implemented in `server/deeplivecam.py` via `detection_interval_ms` (default 66 ms ≈ 15 Hz detection, ~30–40 ms saved per frame). Add config plumbing and a perf note:
- Expose `detection_interval_ms` in `ProcessingConfig` (exists).
- Add a benchmark script `scripts/bench.py` that runs `process_frame` on N synthetic/real frames and prints avg/median/p95 inference ms (Phase 7 companion).

**Step 1:** Create `scripts/bench.py`:
```python
"""Benchmark DeepLiveCamProcessor.process_frame on a set of frames."""
import sys, time, numpy as np
sys.path.insert(0, "third_party/Deep-Live-Cam")
from server.deeplivecam import DeepLiveCamProcessor

def main():
    p = DeepLiveCamProcessor({"processing": {"execution_provider": "auto"}})
    ok = p.set_source_face(open(sys.argv[1], "rb").read())
    if not ok:
        print("ERROR: no face detected in", sys.argv[1]); sys.exit(1)
    p.update_settings({"frame_processors": ["face_swapper"],
                       "detection_interval_ms": 66})
    print("active providers:", p._globals.execution_providers)  # must include ROCM/MIGraphX
    frame = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(3):
        p.process_frame(frame)  # warmup (also pre-warms MIOpen kernel cache)
    times = []
    for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 100):
        t0 = time.perf_counter()
        p.process_frame(frame)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"n={len(times)} avg={np.mean(times):.1f}ms median={times[len(times)//2]:.1f}ms "
          f"p95={times[int(len(times)*0.95)]:.1f}ms max={times[-1]:.1f}ms")

if __name__ == "__main__":
    main()
```
Run:
```bash
source .venv/bin/activate
python scripts/bench.py /path/to/source.jpg 200
```
Expected (RX 7900 XTX / ROCm or MIGraphX): `median` well under 25 ms and the printed providers list starting with `ROCMExecutionProvider`. If it prints `CPUExecutionProvider`, the ORT↔ROCm pair is mismatched (see Task 6). If CPU-only, expect 60–150 ms → 30 fps at 720p is the ceiling.

**Step 2:** Commit.
```bash
git add scripts/bench.py
git commit -m "perf: frame latency benchmark script"
```

### Task 19: Codec preference (H.264) and poll interval tuning

**Files:**
- Modify: `server/rtc.py`, `client/webrtc.py`

H.264 (OpenH264 in aiortc) encodes faster at 720p than VP8 for the same quality on LAN. Prefer it when `network.prefer_codec == "h264"`.

**Step 1:** Add a helper in `shared/` (used by both sides):
```python
# shared/codecs.py
from aiortc import RTCPeerConnection, RTCRtpSender


def prefer_codec(pc: RTCPeerConnection, kind: str, codec: str) -> None:
    caps = RTCRtpSender.getCapabilities(kind)
    wanted = [c for c in caps.codecs if codec.lower() in c.mimeType.lower()]
    if not wanted:
        return
    for transceiver in pc.getTransceivers():
        if transceiver.kind == kind:
            transceiver.setCodecPreferences(wanted)
```
Call it on each PC for `"video"` before `createOffer`/`createAnswer`.

**Step 2:** Wire into `server/rtc.py` (before `createAnswer`) and `client/webrtc.py` (before `createOffer`), gated by `cfg.network.prefer_codec`.

**Step 3:** Unit test the helper is called with the right kind (light test asserting no crash on a fresh PC).

**Step 4:** Commit.
```bash
git add shared/codecs.py server/rtc.py client/webrtc.py tests/test_codecs.py
git commit -m "perf: prefer h264 codec on both peers"
```

---

## Phase 7 — Metrics

### Task 20: `GET /metrics` + client-side logging + latency components

**Files:**
- Modify: `server/app.py` (metrics endpoint exists; make it live), `client/main.py` (already logs), `shared/metrics.py`

Latency breakdown implemented so far:
- **inference_ms** — `Processor._run` (real).
- **decode_ms** — `InboundVideoTrack` (real: `frame.to_ndarray`).
- **encode_ms** — `ProcessedVideoTrack.recv` (proxy: ndarray→VideoFrame conversion; actual libvpx/openh264 encode happens inside aiortc's RTP sender and is not exposed).
- **network_rtt_ms** — proxy via the signaling WebSocket (client measures a WS ping/pong during `connect`). Document that this is signaling-path RTT, not RTCP-measured media RTT (aiortc does not expose SR/RR RTT).
- **fps** — `Metrics.record_incoming` window.
- **dropped / queue_size** — `FrameHolder` + `Metrics`.

**Step 1:** Add a WS ping/pong RTT measurement to `client/webrtc.py`:
```python
async def _measure_rtt(ws, n=10):
    rtts = []
    for _ in range(n):
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "ping"}))
        await ws.recv()
        rtts.append((time.perf_counter() - t0) * 1000)
    return sum(rtts) / len(rtts) if rtts else 0.0
```
Server: handle `{"type": "ping"}` by echoing it back (add to `app.py`'s message loop — `Signal` gains a `ping`/`pong` type in `protocol.py`).

**Step 2:** Verify endpoint.
```bash
curl -s localhost:8000/metrics
```
Expected: JSON with per-session `frames_in`, `inference_ms`, `dropped`, `fps`, etc.

**Step 3:** Commit.
```bash
git add shared/protocol.py shared/metrics.py client/webrtc.py server/app.py tests/test_protocol.py
git commit -m "feat(metrics): live /metrics endpoint and rtt ping"
```

---

## Phase 8 — Config End-to-End

### Task 21: Real YAML configs + `from_yaml` helpers + README

**Files:**
- Modify: `shared/config.py` (add `from_yaml`), `configs/server.yaml`, `configs/client.yaml`, create `README.md`

**Step 1:** Add to `shared/config.py`:
```python
def from_yaml(cls, path: Path):
    return load_config(cls, path)
```
(Add as a base-class/classmethod pattern so both `ServerConfig.from_yaml` and `ClientConfig.from_yaml` exist.)

**Step 2:** Write `configs/server.yaml`:
```yaml
server:
  host: 0.0.0.0
  port: 8000
camera:                    # negotiated target the server asks the client to send
  width: 1280
  height: 720
  fps: 60
network:
  stun: false
  turn: false
  prefer_codec: h264
processing:                # machine-side ONLY; face + swap settings come from the laptop
  execution_provider: auto   # auto picks rocm/migraphx on the RX 7900 XTX
  model_dir: third_party/Deep-Live-Cam/models
performance:
  drop_frames: true
  poll_interval_ms: 2
metrics:
  enabled: true
```

**Step 3:** Finalize `configs/client.yaml` (Task 16 stub + `network.prefer_codec`).

**Step 4:** Write `README.md` covering: prerequisites (two Linux machines, v4l2loopback on client, GPU ROCm/MIGraphX on server), install (`scripts/setup-dlc.sh` + pinned `onnxruntime-rocm`/`onnxruntime-migraphx`, pip installs), run instructions (server: `python -m server.app`; laptop: `python -m client.main --face ~/faces/me.jpg`), hot-swap (`--face` on start; `set_face()` programmatically), latency budget, metrics, troubleshooting (ROCm↔ORT version pinning, `sess.get_providers()` falling back to CPU, MIOpen cache pre-warm, v4l2loopback load, firewall ports 8000 + RTP UDP), nice-to-haves (multi-client, auth, TLS, auto-reconnect, Prometheus export).

**Step 5:** Run full test suite.
```bash
pytest -q
```
Expected: all green.

**Step 6:** Final commit.
```bash
git add .
git commit -m "feat(config): end-to-end yaml config + README"
```

---

## Nice-to-haves (explicitly out of scope for this plan)

Multiple clients (keyed by `client_id`, per-session processor already isolated — but note DLC's module-level processor caches are shared per-process, so N clients share one loaded model; that's fine and desirable), authentication (token in the signaling handshake), TLS (uvicorn ssl certs + `wss`), auto-reconnect (retry loop in `WebRTCClient.connect`), a laptop-side face-picker GUI (adds a UI in front of `set_face()`/`update_settings()`, which already exist), OBS plugin, Docker compose (GPU passthrough for onnxruntime-rocm), Prometheus metrics endpoint.

Hot-swap identity is **core** (not nice-to-have): `WebRTCClient.set_face()` → `control` message → `Processor.apply_control()` → `DeepLiveCamProcessor.set_source_face()`.

## Performance Verification Checklist (final gate)

1. `python scripts/bench.py <source.jpg> 200` → prints `ROCMExecutionProvider` first and `median < 25 ms` — else drop resolution, enable detection caching, or fix the ORT↔ROCm pair.
2. `pytest -q` → all green.
3. Client `main.py` streaming to server: `curl localhost:8000/metrics` shows `inference_ms < 25`, `fps ~ 30–60`, `dropped` climbing (drop-frames working).
4. Control plane: start client with `--face A.jpg`, confirm swap; then `--face B.jpg` in a second run (or call `set_face()`) and confirm the swapped identity changes without a server restart.
5. OBS preview of `/dev/video10` with a stopwatch in frame: measure motion-to-display latency < 100 ms (LAN).
