# Remote Deep-Live-Cam

LAN client/server app where a **laptop** streams its webcam over WebRTC to a
**desktop** (RX 7900 XTX) running headless Deep-Live-Cam inference, and receives
the swapped video back into a Linux virtual camera (`/dev/video10`), ready for
Zoom / Discord / OBS.

The **laptop is the control point**: it picks the source face and swap settings,
which are sent to the desktop over the signaling channel at connect and whenever
they change (hot-swap). The desktop is a dumb inference engine with no face
config of its own.

```
Laptop Webcam --(OpenCV)--> WebRTC --> Desktop GPU (ROCm) --> DLC swap --> WebRTC --> pyvirtualcam --> /dev/video10
```

## Architecture

| Component | Machine | Role |
|---|---|---|
| `client/` | laptop | webcam capture, WebRTC, control plane, virtual camera |
| `server/` | desktop (GPU) | signaling, WebRTC, headless Deep-Live-Cam inference |
| `shared/` | both | config, protocol, newest-frame-wins holder, metrics |

Key design points:

- **Newest-frame-wins**: every hop drops stale frames (`FrameHolder`), so
  end-to-end latency never grows with queue backlog. Target <100 ms at 720p60.
- **One RTCPeerConnection** carries webcam up + processed video back. SDP is
  exchanged **non-trickle**: each side waits for ICE gathering to complete and
  serializes `pc.localDescription.sdp` (which contains the candidates). No
  STUN/TURN needed on a LAN.
- **Control plane**: `control` signal carries the source face (base64 image) +
  swap settings. `set_face()` / `update_settings()` hot-swap at runtime.
- **GPU only on the server** via ONNX Runtime ROCm / MIGraphX.

## Prerequisites

- Two Linux machines. Python 3.11–3.12, `git`, `ffmpeg`.
- **Client (laptop):** `v4l2loopback` kernel module.
- **Server (desktop):** AMD GPU (RX 7900 XTX), ROCm 6.x (or 7.x), enough disk
  for models (~300 MB).

## Install

### 1. Repo + venv (both machines)

```bash
git clone <this repo> remote-deeplivecam && cd remote-deeplivecam
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-client.txt
```

### 2. Server GPU machine only — Deep-Live-Cam + ROCm

```bash
./scripts/setup-dlc.sh
```

This clones Deep-Live-Cam into `third_party/` (pinned to a specific commit — see
`PIN_COMMIT` at the top of the script), downloads the inswapper models,
and pins the GPU runtime. **Verify the provider is genuinely active** (RDNA3's
#1 failure mode is silently falling back to CPU):

```bash
python - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession("third_party/Deep-Live-Cam/models/inswapper_128.onnx",
                            providers=ort.get_available_providers())
print(sess.get_providers())  # must include ROCMExecutionProvider or MIGraphXExecutionProvider
PY
```

If it falls back to CPU, the ORT↔ROCm pair is mismatched:
- ROCm 6.x → `pip install onnxruntime-rocm==1.22.2.post1`
- ROCm 7.x → `pip install onnxruntime-migraphx`
(`setup-dlc.sh` does this automatically from `/opt/rocm/.info/version`)

Then pre-warm the MIOpen kernel cache once: `python scripts/bench.py <source.jpg> 5`.

### 3. Client laptop only — virtual camera

```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils
# Load WITH exclusive_caps=1 so the device advertises capture-only. Browsers
# (Chrome/Firefox getUserMedia) refuse a capture+output device and show
# "your webcam is being used or blocked by another application". The old
# "load WITHOUT exclusive_caps" advice only applied to the broken 0.12.7.
# (Kernel 7.0: apt's 0.12.7 is broken — build 0.15.4 from source, see RUNBOOK.)
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepLiveCam" exclusive_caps=1
```

Persist it across reboots:

```bash
echo 'options v4l2loopback devices=1 video_nr=10 card_label="DeepLiveCam" exclusive_caps=1' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
sudo update-initramfs -u
```

## Run

**Desktop (server):**
```bash
source .venv/bin/activate
python -m server.app            # or: python -m uvicorn server.app:create_app --factory
```
Endpoints: `GET /health`, `GET /`, `GET /metrics` (per-session latency breakdown).

**Laptop (client):** edit `configs/client.yaml` (server URL, camera, virtual cam,
source face) then:
```bash
source .venv/bin/activate
python -m client.main --face ~/faces/me.jpg
```

Open Zoom / Discord / OBS and select **DeepLiveCam**. To hot-swap the face,
restart with a different `--face` (programmatic hot-swap: `WebRTCClient.set_face()`).

## Latency budget (720p60, wired LAN)

| Stage | ms |
|---|---|
| Capture wait (60fps) | 4–8 |
| H.264 encode / decode | 3–6 each |
| Network (LAN) | 1–2 each way |
| Face detection (cached @15 Hz) | ~5 |
| **Swap inference (GPU)** | **10–25** |
| Virtualcam write | ~1 |
| **Total** | **~35–65** |

Watch `curl localhost:8881/metrics` for the real numbers. If `inference_ms`
climbs above ~50, the GPU provider likely fell back to CPU (see above).

## Repository & syncing

The repo lives at **`git@github.com:dodistyo/deeplive.git`**. Both the laptop and
the desktop clone/push to the same repo. The laptop is the canonical machine for
edits; keep the desktop in sync with `git pull`.

`third_party/` (the Deep-Live-Cam clone + models) and `.venv/` are **gitignored** —
not part of the repo. Deep-Live-Cam's version is pinned via `PIN_COMMIT` in
`scripts/setup-dlc.sh`; update it deliberately, never via a blind `main` pull.

Desktop machine (existing checkout → become a git clone without losing `third_party/`):
```bash
cd ~/Applications/Deepfake/remote-deeplivecam
git init
git remote add origin git@github.com:dodistyo/deeplive.git
git fetch origin
git reset --hard origin/main   # tracked files now match the laptop; ignored dirs kept
```
(The laptop is initialized normally: `git init && git add . && git commit` and
`git push -u origin main`.)

## Development

```bash
pytest -q          # full suite
ruff check .       # lint
```

Tests cover the shared primitives, the DLC wrapper (control plane), the
WebRTC tracks, the threaded processor (newest-wins + drops), and an
**in-process loopback test** that pushes synthetic frames client→server→client
through two real `RTCPeerConnection`s. GPU/webcam/v4l2loopback are NOT required.

## Nice-to-haves (not yet built)

Multi-client (per-session processors already isolated), auth token in the
signaling handshake, TLS (`wss`), auto-reconnect, laptop face-picker GUI
(a thin UI over `set_face()`), Docker compose with GPU passthrough, Prometheus
export.
