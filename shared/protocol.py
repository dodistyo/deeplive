import json
from dataclasses import dataclass, field
from typing import Any

# Signaling message types exchanged over the WebSocket.
# SDP is exchanged non-trickle (candidates embedded in the SDP after
# ICE gathering completes), so 'ice' is defined for future/extensibility.


@dataclass
class Signal:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def offer(sdp: str) -> "Signal":
        return Signal("offer", {"sdp": sdp})

    @staticmethod
    def answer(sdp: str) -> "Signal":
        return Signal("answer", {"sdp": sdp})

    @staticmethod
    def ice(candidate: dict) -> "Signal":
        return Signal("ice", {"candidate": candidate})

    @staticmethod
    def config(payload: dict) -> "Signal":
        return Signal("config", {"payload": payload})

    @staticmethod
    def control(payload: dict) -> "Signal":
        """Laptop -> server: source face (base64 image bytes) + swap settings."""
        return Signal("control", {"payload": payload})

    @staticmethod
    def metrics(snapshot: dict) -> "Signal":
        return Signal("metrics", {"snapshot": snapshot})

    @staticmethod
    def ping() -> "Signal":
        return Signal("ping", {})

    @staticmethod
    def pong() -> "Signal":
        return Signal("pong", {})

    @staticmethod
    def ack() -> "Signal":
        return Signal("ack", {})

    @staticmethod
    def error(message: str) -> "Signal":
        return Signal("error", {"message": message})

    def encode(self) -> str:
        return json.dumps({"type": self.type, **self.data})

    @classmethod
    def decode(cls, raw: str) -> "Signal":
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid signal json: {exc}") from exc
        if not isinstance(msg, dict) or "type" not in msg:
            raise ValueError(f"signal missing 'type': {raw!r}")
        known = {"offer", "answer", "ice", "config", "control", "metrics",
                 "ping", "pong", "ack", "error"}
        if msg["type"] not in known:
            raise ValueError(f"unknown signal type: {msg['type']}")
        return cls(type=msg["type"], data={k: v for k, v in msg.items() if k != "type"})
