from aiortc import RTCPeerConnection, RTCRtpSender


def prefer_codec(pc: RTCPeerConnection, kind: str, codec: str) -> bool:
    """Prefer a specific codec (e.g. 'h264') on all transceivers of `kind`.

    Returns True if at least one matching codec was found and applied.
    """
    caps = RTCRtpSender.getCapabilities(kind)
    wanted = [c for c in caps.codecs if codec.lower() in c.mimeType.lower()]
    if not wanted:
        return False
    applied = False
    for transceiver in pc.getTransceivers():
        if transceiver.kind == kind:
            transceiver.setCodecPreferences(wanted)
            applied = True
    return applied
