import threading
import time

from shared.frame_holder import FrameHolder


def test_put_overwrites_older():
    h = FrameHolder()
    h.put("a")
    h.put("b")          # old frame dropped
    h.put("c")
    assert h.get() == "c"
    assert h.dropped == 2


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
    assert results == [("x", 1)]


def test_get_new_waits_for_different_frame():
    h = FrameHolder()
    h.put("old")  # seq 1
    got = []

    def reader():
        got.append(h.get_new(1, timeout=0.2))  # wait for seq != 1

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    h.put("new")
    t.join(0.5)
    assert got == [("new", 2)]


def test_get_new_timeout_returns_none():
    h = FrameHolder()
    h.put("a")
    assert h.get_new(1, timeout=0.05) == (None, 1)  # seq unchanged


def test_identical_object_still_registers_as_new():
    h = FrameHolder()
    buf = object()
    h.put(buf)       # seq 1
    h.put(buf)       # same object again -> seq 2
    frame, seq = h.get_new(1, timeout=0.05)
    assert frame is buf and seq == 2
