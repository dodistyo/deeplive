import time

import numpy as np

from client.virtualcam import VirtualCamera
from shared.frame_holder import FrameHolder


class FakeCam:
    def __init__(self):
        self.device = "/dev/fake0"
        self.sent = []
        self.width, self.height, self.fps = 64, 64, 30
        self.closed = False

    def send(self, frame):
        self.sent.append(frame.copy())

    def sleep_until_next_frame(self):
        time.sleep(0.001)

    def close(self):
        self.closed = True


def test_virtual_camera_writes_latest_frames():
    holder = FrameHolder()
    fake = FakeCam()
    vcam = VirtualCamera(holder, cam=fake)
    vcam.start()
    time.sleep(0.02)
    holder.put(np.full((64, 64, 3), 1, np.uint8))
    time.sleep(0.05)
    holder.put(np.full((64, 64, 3), 2, np.uint8))
    time.sleep(0.05)
    vcam.stop()
    assert len(fake.sent) >= 2
    # newest frame must have been sent
    assert (fake.sent[-1] == 2).all()
    assert fake.closed is True


def test_virtual_camera_start_stop_clean():
    holder = FrameHolder()
    fake = FakeCam()
    vcam = VirtualCamera(holder, cam=fake)
    vcam.start()
    time.sleep(0.02)
    vcam.stop()
    assert fake.closed is True
