#!/usr/bin/env bash
# Clone Deep-Live-Cam and download models; verify headless CLI swap.
# Run on the SERVER (GPU) machine. Idempotent.
#
# ROCm note: pin the ORT<->ROCm pair to the version the DLC ecosystem ships.
#   ROCm 6.x -> onnxruntime-rocm==1.22.2.post1 (facefusion-pinned)
#   ROCm 7.x -> onnxruntime-migraphx
# RDNA3 (gfx1100) is best-effort in ROCm: always verify the provider is
# actually ACTIVE (sess.get_providers()), not silently on CPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DLC_DIR="$REPO_ROOT/third_party/Deep-Live-Cam"
MODELS_DIR="$DLC_DIR/models"
VENV="$REPO_ROOT/.venv"

# Pin Deep-Live-Cam to a specific commit (controlled update). To bump DLC,
# change PIN_COMMIT and re-run this script. Do NOT track upstream `main`
# blindly: hacksider changes (e.g. requirements.txt) can break the ROCm/ORT
# pinning and numpy/opencv versions this project relies on.
PIN_COMMIT="230217ec11595d979922a4f636e92ade50608702"

if [ ! -d "$DLC_DIR" ]; then
  git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git "$DLC_DIR"
fi
git -C "$DLC_DIR" fetch --depth 1 origin "$PIN_COMMIT" 2>/dev/null || git -C "$DLC_DIR" fetch origin "$PIN_COMMIT"
git -C "$DLC_DIR" checkout --detach "$PIN_COMMIT"
echo "Deep-Live-Cam pinned to: $(git -C "$DLC_DIR" rev-parse --short HEAD)"

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

# Install DLC deps into the shared venv (customtkinter, insightface, torch...)
source "$VENV/bin/activate"
pip install -r "$DLC_DIR/requirements.txt"

# GPU inference: remove ANY conflicting onnxruntime wheels first. DLC's
# requirements.txt pulls in onnxruntime-gpu (CUDA); keeping both breaks the
# provider (they fight over the same onnxruntime/ files) and you get a silent
# CPU fallback.
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-rocm onnxruntime-migraphx 2>/dev/null || true

# Pin the ORT<->ROCm pair to the installed ROCm version:
#   ROCm 6.x -> onnxruntime-rocm==1.22.2.post1 (facefusion-pinned)
#   ROCm 7.x -> onnxruntime-migraphx
ROCM_VERSION=""
if [ -f /opt/rocm/.info/version ]; then
  ROCM_VERSION=$(cat /opt/rocm/.info/version)
fi
ROCM_MAJOR="${ROCM_VERSION%%.*}"
echo "Detected ROCm $ROCM_VERSION (major $ROCM_MAJOR)"
if [ "$ROCM_MAJOR" = "7" ]; then
  pip install onnxruntime-migraphx
else
  pip install "onnxruntime-rocm==1.22.2.post1"
fi

# Verify the GPU provider is ACTIVE (not silently on CPU). RDNA3's #1 failure.
python - <<'PY'
import onnxruntime as ort, sys
try:
    providers = ort.get_available_providers()
except Exception as e:
    print("ERROR: onnxruntime import failed:", e); sys.exit(1)
print("available providers:", providers)
if not any(p in providers for p in ("ROCMExecutionProvider", "MIGraphXExecutionProvider")):
    print("WARNING: no AMD GPU provider available - inference will run on CPU")
PY

echo "DLC installed. Test with:"
echo "  source $VENV/bin/activate"
echo "  python $DLC_DIR/run.py -s /path/to/source.jpg -t /path/to/target.jpg -o /tmp/out.jpg"
echo
echo "GPU provider verified above. For a real end-to-end speed check run:"
echo "  python scripts/bench.py /path/to/source.jpg 200"
