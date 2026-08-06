"""Benchmark DeepLiveCamProcessor.process_frame on a set of frames.

Usage:
    python scripts/bench.py /path/to/source.jpg [n_frames]

Prints the active execution providers and avg/median/p95 inference latency.
Runs on the SERVER (GPU) machine after setup-dlc.sh.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Deep-Live-Cam"))

from server.deeplivecam import DeepLiveCamProcessor


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    p = DeepLiveCamProcessor({"processing": {"execution_provider": "auto"}})
    with open(sys.argv[1], "rb") as fh:
        ok = p.set_source_face(fh.read())
    if not ok:
        print("ERROR: no face detected in", sys.argv[1])
        sys.exit(1)
    p.update_settings({"frame_processors": ["face_swapper"], "detection_interval_ms": 66})
    print("active providers:", p._globals.execution_providers)
    print("ROCM/MIGraphX ACTIVE:", "ROCM" in " ".join(p._globals.execution_providers).upper()
          or "MIGRAPHX" in " ".join(p._globals.execution_providers).upper())

    frame = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(3):
        p.process_frame(frame)  # warmup (also pre-warms the MIOpen kernel cache)

    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        p.process_frame(frame)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"n={len(times)} avg={np.mean(times):.1f}ms "
          f"median={times[len(times) // 2]:.1f}ms "
          f"p95={times[int(len(times) * 0.95)]:.1f}ms max={times[-1]:.1f}ms")


if __name__ == "__main__":
    main()
