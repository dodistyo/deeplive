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
        for _ in range(5):
            p.feed(np.zeros((64, 64, 3), np.uint8))
        time.sleep(0.12)
        assert p.metrics.snapshot().dropped >= 1
    finally:
        p.stop()


def test_apply_control_forwards_face_and_settings():
    class FakeDLC:
        def __init__(self):
            self.face = None
            self.settings = None

        def set_source_face(self, image_bytes):
            self.face = image_bytes
            return True

        def update_settings(self, settings):
            self.settings = settings

    dlc = FakeDLC()
    p = Processor(dlc)
    ok = p.apply_control({"face": "aGVsbG8=", "settings": {"many_faces": True}})
    assert ok is True
    assert dlc.face == b"hello"
    assert dlc.settings == {"many_faces": True}


def test_apply_control_rejects_bad_face():
    class FakeDLC:
        def set_source_face(self, image_bytes):
            return False

        def update_settings(self, settings):
            pass

    p = Processor(FakeDLC())
    assert p.apply_control({"face": "AAAA", "settings": {}}) is False


def test_drop_frames_default_on():
    p = Processor(SlowFake())
    assert p._drop_frames is True


def test_drop_frames_disabled_warns(caplog):
    import logging

    fake = SlowFake()
    p = Processor(fake, drop_frames=False)
    p.feed(np.zeros((4, 4, 3), np.uint8))
    with caplog.at_level(logging.WARNING, logger="server.processor"):
        p.feed(np.zeros((4, 4, 3), np.uint8))  # holder non-empty -> diagnostic warning
    assert any("drop_frames disabled" in r.message for r in caplog.records)
