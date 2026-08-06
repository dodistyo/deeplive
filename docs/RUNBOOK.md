# Runbook — copy-paste operations guide

Two machines, same repo checked out on both:

| Machine | Runs | Needs |
|---|---|---|
| **Desktop** (RX 7900 XTX) | `server/` | GPU + ROCm, Deep-Live-Cam models |
| **Laptop** | `client/` | webcam, v4l2loopback |

Run everything with the venv active: `source .venv/bin/activate`.

**Repo / sync**: both machines use the same repo at `git@github.com:dodistyo/deeplive.git`.
The **laptop is canonical** for edits; on the desktop just `git pull` after changes
are pushed. `third_party/` and `.venv/` are gitignored (Deep-Live-Cam is pinned via
`PIN_COMMIT` in `scripts/setup-dlc.sh` — bump deliberately, never blind `main`).
To turn an existing desktop checkout into a git clone (keeps `third_party/` + models):
`git init && git remote add origin git@github.com:dodistyo/deeplive.git && git fetch origin && git reset --hard origin/main`.

---

## 1. Desktop — one-time setup (GPU machine)

```bash
cd remote-deeplivecam
source .venv/bin/activate
./scripts/setup-dlc.sh
```

Verify the GPU provider is **genuinely active** (RDNA3's #1 failure: silently falls back to CPU):

```bash
python - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession("third_party/Deep-Live-Cam/models/inswapper_128.onnx",
                            providers=ort.get_available_providers())
print(sess.get_providers())   # must include ROCMExecutionProvider or MIGraphXExecutionProvider
PY
```

If it says `CPUExecutionProvider`:
- ROCm 6.x → `pip install onnxruntime-rocm==1.22.2.post1`
- ROCm 7.x → `pip install onnxruntime-migraphx`
(`setup-dlc.sh` does this automatically from `/opt/rocm/.info/version`)
- then re-run the check above.

Pre-warm the MIOpen kernel cache and measure inference (first real inference, expect `median < 25 ms` on GPU):

```bash
python scripts/bench.py /path/to/source.jpg 200
```

Edit `configs/server.yaml` if needed (default: `0.0.0.0:8881`, provider `auto`).

---

## 2. Desktop — run the server

```bash
cd remote-deeplivecam
source .venv/bin/activate
python -m server.app
```

Smoke-check from any shell on the desktop:

```bash
curl -s localhost:8881/health        # {"ok":true}
```

**Firewall note:** allow TCP 8881 (signaling) and UDP RTP ports on the desktop LAN IP,
e.g. `sudo ufw allow 8881/tcp` (UDP media usually negotiates ephemeral ports).

---

## 3. Laptop — one-time setup (client machine)

```bash
cd remote-deeplivecam
source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-client.txt
sudo apt install v4l2loopback-dkms v4l2loopback-utils
# NOTE: do NOT pass exclusive_caps=1 — on v4l2loopback 0.12.7 the device gets
# stuck advertising neither Video Capture nor Video Output after a producer
# stops, forcing a module reload. Without it the device always advertises both.
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepLiveCam"
v4l2-ctl --list-devices        # expect /dev/video10
```

**Kernel 7.0 note (Ubuntu 24.04 HWE)**: the packaged `v4l2loopback-dkms` 0.12.7 is
broken on kernel 7.x — producers write frames but readers get nothing (0 bytes).
Build + sign v4l2loopback 0.15.4 from source instead:

```bash
git clone https://github.com/umlaeute/v4l2loopback ~/Applications/v4l2loopback
cd ~/Applications/v4l2loopback && make
sudo dkms add . && sudo dkms install v4l2loopback/0.15.4
# dkms signs with a non-enrolled key; re-sign with your MOK key (Secure Boot):
zstd -d -f /lib/modules/$(uname -r)/updates/dkms/v4l2loopback.ko.zst -o /tmp/v4l2loopback.ko
# strip any existing signature, then:
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha512 \
  /var/lib/shim-signed/mok/MOK.priv /var/lib/shim-signed/mok/MOK.der /tmp/v4l2loopback.ko
sudo rm -f /lib/modules/$(uname -r)/updates/dkms/v4l2loopback.ko.zst
sudo install -D -m 644 /tmp/v4l2loopback.ko /lib/modules/$(uname -r)/updates/dkms/v4l2loopback.ko
sudo depmod -a && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepLiveCam"
```

To auto-load the module at boot: add `v4l2loopback` to `/etc/modules-load.d/` and a
`/etc/modprobe.d/v4l2loopback.conf` with `options v4l2loopback devices=1 video_nr=10 card_label="DeepLiveCam"`.

---

## 4. Laptop — configure

Edit `configs/client.yaml`:

```yaml
client:
  server_url: "ws://<DESKTOP_LAN_IP>:8881/ws/laptop1"   # e.g. 192.168.1.50
processing:
  source_face: /home/laptop/faces/me.jpg                 # path on the LAPTOP
```

Find the desktop's LAN IP: `hostname -I` on the desktop.

---

## 5. Laptop — run the client

```bash
cd remote-deeplivecam
source .venv/bin/activate
python -m client.main --face ~/faces/me.jpg
```

Wait for `connected; streaming to ...`, then in Zoom / Discord / OBS select **DeepLiveCam**.

Hot-swap the face: stop and re-run with a different `--face` (programmatic hot-swap
available via `WebRTCClient.set_face()`).

---

## 6. Verify it's actually working

On the **desktop** while streaming:

```bash
curl -s localhost:8881/metrics
```

Look for `"inference_ms"` < 50 (target < 25), `"fps"` ~ 30–60, `"dropped"` climbing
(the newest-frame-wins policy discarding stale frames — that's correct behavior).

If `inference_ms` is 60–150, the GPU provider fell back to CPU — see step 1.

**Latency check:** open an OBS preview of `DeepLiveCam`, hold a stopwatch phone app in
the webcam frame, and compare motion. Target < 100 ms (~2–4 frames of 60 fps).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot open camera` | Webcam busy or wrong `camera.index` in `client.yaml`; close OBS/Zoom first |
| `/dev/video10` missing | re-run the `modprobe` line in step 3 |
| `Device /dev/video10 is not a video output device` | v4l2loopback got stuck (0.12.7 bug with `exclusive_caps=1`); reload without it: `sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepLiveCam"` |
| Google Meet shows no DeepLiveCam camera | the client must be running (it only advertises capture while writing); start the client first, keep it running, then select the camera |
| `control rejected: no face detected` | server couldn't find a face in your `--face` image — use a clear, front-facing photo |
| ICE stuck / `connection failed` | wrong LAN IP, firewall on the desktop (TCP 8881 + UDP), or both machines not on the same subnet |
| GPU provider absent / CPU fallback | ORT↔ROCm version mismatch — `setup-dlc.sh` now auto-detects ROCm major and installs `onnxruntime-migraphx` (ROCm 7) or `onnxruntime-rocm` (ROCm 6); re-run it |
| High `inference_ms` | CPU fallback (above); or enable detection caching / lower `camera.width` |
