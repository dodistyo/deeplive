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
    assert snap.network_rtt_ms == 4.0


def test_ema_smoothing():
    m = Metrics()
    for _ in range(10):
        m.record_inference_ms(20.0)
    m.record_inference_ms(100.0)
    assert m.snapshot().inference_ms < 50.0  # smoothed, not spiked


def test_dropped_and_outgoing():
    m = Metrics()
    m.record_dropped(3)
    m.record_outgoing()
    snap = m.snapshot()
    assert snap.dropped == 3
    assert snap.frames_out == 1


def test_reset():
    m = Metrics()
    m.record_incoming()
    m.record_dropped(1)
    m.reset()
    snap = m.snapshot()
    assert snap.frames_in == 0
    assert snap.dropped == 0


def test_to_dict_jsonable():
    m = Metrics()
    m.record_inference_ms(12.345)
    d = m.to_dict()
    assert isinstance(d["inference_ms"], float)
    assert 0 < d["inference_ms"] < 100
    import json
    json.dumps(d)  # must be serializable
