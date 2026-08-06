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
    # count target-frame detections only (set_source_face detects the source image once)
    detects = [c for c in fake.calls if c[0] == "detect" and c[1] == (720, 1280, 3)]
    assert len(detects) == 1  # only first frame triggered target detection


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


def test_enhancer_appended_to_swapper():
    fake = FakeDLC()
    p = DeepLiveCamProcessor({}, dlc=fake)
    p.set_source_face(face_img_bytes())
    p.update_settings({"frame_processors": ["face_swapper"], "enhancer": True})
    assert [fp.name for fp in fake.fp_chain] == ["face_swapper", "face_enhancer"]


def test_bad_source_face_rejected():
    p = DeepLiveCamProcessor({}, dlc=FakeDLC())
    assert p.set_source_face(b"this is not an image") is False
