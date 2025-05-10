import enum
from dataclasses import dataclass
from dataclasses import field


class MetaEventType(enum.IntEnum):
    """MIDI meta event types"""
    SEQUENCE_NUMBER = 0x00
    TEXT_EVENT = 0x01
    COPYRIGHT_NOTICE = 0x02
    TRACK_NAME = 0x03
    INSTRUMENT_NAME = 0x04
    LYRIC = 0x05
    MARKER = 0x06
    CUE_POINT = 0x07
    MIDI_CHANNEL_PREFIX = 0x20
    END_OF_TRACK = 0x2F
    SET_TEMPO = 0x51
    SMPTE_OFFSET = 0x54
    TIME_SIGNATURE = 0x58
    KEY_SIGNATURE = 0x59
    SEQUENCER_SPECIFIC = 0x7F
    UNKNOWN = -1


@dataclass(frozen=True)
class MetaEventData:
    """Meta event data class"""
    data: bytes
    meta_type: MetaEventType


@dataclass(frozen=True)
class MetaEventSequenceNumber(MetaEventData):
    """Meta event for sequence number"""
    ssss: int
    meta_type: MetaEventType = field(default=MetaEventType.SEQUENCE_NUMBER, init=False)


@dataclass(frozen=True)
class MetaEventText(MetaEventData):
    """Meta event for text"""
    meta_type: MetaEventType = field(default=MetaEventType.TEXT_EVENT, init=False)
    text: str


@dataclass(frozen=True)
class MetaEventCopyright(MetaEventData):
    """Meta event for copyright"""
    text: str
    meta_type: MetaEventType = field(default=MetaEventType.COPYRIGHT_NOTICE, init=False)


@dataclass(frozen=True)
class MetaEventTrackName(MetaEventData):
    """Meta event for track name"""
    text: str
    meta_type: MetaEventType = field(default=MetaEventType.TRACK_NAME, init=False)


@dataclass(frozen=True)
class MetaEventInstrumentName(MetaEventData):
    """Meta event for instrument name"""
    text: str
    meta_type: MetaEventType = field(default=MetaEventType.INSTRUMENT_NAME, init=False)


@dataclass(frozen=True)
class MetaEventLyric(MetaEventData):
    """Meta event for lyric"""
    text: str
    meta_type: MetaEventType = field(default=MetaEventType.LYRIC, init=False)


@dataclass(frozen=True)
class MetaEventMarker(MetaEventData):
    """Meta event for marker"""
    text: str
    meta_type: MetaEventType = field(default=MetaEventType.MARKER, init=False)


@dataclass(frozen=True)
class MetaEventCuePoint(MetaEventData):
    """Meta event for cue point"""
    text: str
    meta_type: MetaEventType = field(default=MetaEventType.CUE_POINT, init=False)


@dataclass(frozen=True)
class MetaEventMIDIChannelPrefix(MetaEventData):
    """Meta event for MIDI channel prefix"""
    channel: int
    meta_type: MetaEventType = field(default=MetaEventType.MIDI_CHANNEL_PREFIX, init=False)


@dataclass(frozen=True)
class MetaEventEndOfTrack(MetaEventData):
    """Meta event for end of track"""
    meta_type: MetaEventType = field(default=MetaEventType.END_OF_TRACK, init=False)


@dataclass(frozen=True)
class MetaEventSetTempo(MetaEventData):
    """Meta event for set tempo"""
    tttttt: int
    meta_type: MetaEventType = field(default=MetaEventType.SET_TEMPO, init=False)


@dataclass(frozen=True)
class MetaEventSMPTEOffset(MetaEventData):
    """Meta event for SMPTE offset"""
    hr: int
    mn: int
    se: int
    fr: int
    ff: int
    meta_type: MetaEventType = field(default=MetaEventType.SMPTE_OFFSET, init=False)


@dataclass(frozen=True)
class MetaEventTimeSignature(MetaEventData):
    """Meta event for time signature"""
    nn: int
    dd: int
    cc: int
    bb: int
    meta_type: MetaEventType = field(default=MetaEventType.TIME_SIGNATURE, init=False)


@dataclass(frozen=True)
class MetaEventKeySignature(MetaEventData):
    """Meta event for key signature"""
    sf: int
    mi: int
    meta_type: MetaEventType = field(default=MetaEventType.KEY_SIGNATURE, init=False)

    def to_string(self) -> str:
        """Returns the key signature as a string"""
        lof_idx = (self.sf + 1) // 7
        if lof_idx > 0:
            lof = "#" * lof_idx
        elif lof_idx < 0:
            lof = "b" * -lof_idx
        else:
            lof = ""
        key = "FCGDAEB"[(self.sf + 1) % 7]
        mode = "minor" if self.mi == 1 else "major"
        return f"{key}{lof} {mode}"


@dataclass(frozen=True)
class MetaEventSequencerSpecific(MetaEventData):
    """Meta event for sequencer specific"""
    meta_type: MetaEventType = field(default=MetaEventType.SEQUENCER_SPECIFIC, init=False)
