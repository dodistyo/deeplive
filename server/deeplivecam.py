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
        self._warmup_face = proc_cfg.get("warmup_face", "")

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
        if not self._frame_processors:  # ensure a default pipeline exists
            self.update_settings({})
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

    @property
    def many_faces(self) -> bool:
        return self._many_faces

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
            except Exception as exc:  # noqa: BLE001 - keep the stream alive on model hiccups
                logger.warning("frame processor error: %s", exc)
        return processed if processed is not None else frame

    def warmup(self) -> None:
        """Preload the face-analyser + swapper models at startup.

        MIGraphX compiles models on first use; on a cold server that is
        ~90s for the face analyser and ~60s for the swapper, which blows
        the client's control-ack / connect timeouts. The model caches are
        process-global, so one warmup covers every session.

        A real face is needed to also preload the recognition model (its
        first inference is only triggered when a face is detected); falls
        back to a partial warmup if no readable warmup_face is configured.
        """
        if self._dlc is not None:  # injected fake; nothing to warm
            return
        logger.info("warming up face analyser...")
        if self._warmup_face:
            img = cv2.imread(self._warmup_face)
            if img is None:
                logger.warning("warmup_face unreadable: %s", self._warmup_face)
                img = np.zeros((160, 160, 3), np.uint8)
        else:
            img = np.zeros((160, 160, 3), np.uint8)
        self._get_one_face(img)
        logger.info("warming up face swapper...")
        from modules.processors.frame.face_swapper import get_face_swapper
        get_face_swapper()
        logger.info("model warmup complete")

    def close(self) -> None:
        try:
            g = self._globals
            if g is None or "ROCMExecutionProvider" not in g.execution_providers:
                return
            import torch
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("close() cleanup skipped", exc_info=True)
