import json

from shared.protocol import Signal


def test_offer_roundtrip():
    sig = Signal.offer(sdp="v=0\r\n")
    msg = json.loads(sig.encode())
    assert msg["type"] == "offer"
    assert msg["sdp"] == "v=0\r\n"
    assert Signal.decode(sig.encode()) == sig


def test_ice_and_answer():
    ice = Signal.ice({"candidate": "c1", "sdpMid": "0", "sdpMLineIndex": 0})
    ans = Signal.answer(sdp="v=0")
    assert Signal.decode(ice.encode()).type == "ice"
    assert Signal.decode(ans.encode()).type == "answer"


def test_control_carries_face_and_settings():
    payload = {
        "settings": {"frame_processors": ["face_swapper"], "many_faces": False,
                     "detection_interval_ms": 66},
        "face": "iVBORw0KGgo=...",  # base64 jpeg bytes of the source face
    }
    sig = Signal.control(payload)
    decoded = Signal.decode(sig.encode())
    assert decoded.type == "control"
    assert decoded.data["payload"]["settings"]["frame_processors"] == ["face_swapper"]
    assert decoded.data["payload"]["face"] == "iVBORw0KGgo=..."


def test_ping_pong_ack():
    assert Signal.decode(Signal.ping().encode()).type == "ping"
    assert Signal.decode(Signal.pong().encode()).type == "pong"
    assert Signal.decode(Signal.ack().encode()).type == "ack"


def test_malformed_raises():
    try:
        Signal.decode('{"type": "bogus"}')
    except ValueError:
        return
    raise AssertionError("unknown type should raise")


def test_non_json_raises():
    try:
        Signal.decode("not-json{")
    except ValueError:
        return
    raise AssertionError("invalid json should raise")
