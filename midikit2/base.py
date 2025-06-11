from __future__ import annotations
import enum
import typing
import io
import struct
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from .exceptions import MidiDecodeError, MidiEOFError


class ChunkType(enum.IntEnum):
    """MIDI chunk types"""

    HEADER = 0x00
    TRACK = 0x01
    SYSEX = 0x02
    META = 0x03

    def __repr__(self) -> str:
        return self.name


class EventType(enum.IntEnum):
    """MIDI event types"""

    MIDI_EVENT = 0x00
    SYSEX_EVENT = 0x01
    META_EVENT = 0x02

    def __repr__(self) -> str:
        return self.name


### Base classes ###

@dataclass(frozen=True)
class VariableLengthInt:
    """Variable length integer class as specified in the MIDI 1.0 specification"""

    value: int
    data: bytes

    @staticmethod
    def read(infile: typing.BinaryIO, max_size: int = 99) -> VariableLengthInt:
        """Reads a variable length integer from the infile"""
        delta = 0
        bts: list[int] = []

        for _ in range(max_size):
            byte = read_byte(infile)
            delta = (delta << 7) | (byte & 0x7f)
            bts.append(byte)
            if byte < 0x80:
                return VariableLengthInt(delta, bytes(bts))
        raise MidiDecodeError('Variable length integer too long')

    @staticmethod
    @lru_cache(maxsize=1000)
    def from_int(value: int):
        """Creates a VariableLengthInt from an integer"""
        # if not (0 <= value < (1 << 64)):
        if not (0 <= value):
            raise ValueError(f'Variable length integer value out of range: {value}')
        orig = value
        bts: list[int] = []
        while value > 0:
            bts.append(value & 0x7f)
            value >>= 7
        return VariableLengthInt(orig, bytes(bts[::-1]))

    def __repr__(self):
        return self.value.__repr__()


@dataclass(frozen=True)
class MidiData(ABC):
    """Base class for any structured collection of MIDI data"""

    @abstractmethod
    def get_data(self) -> bytes:
        """Returns the underlying data as bytes

        For chunks, this includes the chunk header, length, and contents
        For events, this includes the delta time and the event data"""
        raise NotImplementedError

    def __post_init__(self):
        # Exists to apease mypy
        pass


@dataclass(frozen=True)
class Chunk(MidiData):
    """MIDI chunk class"""
    chunk_type: ChunkType

    @property
    @abstractmethod
    def events(self) -> typing.Iterator[MtrkEvent]:
        """Returns an iterator of events in this chunk"""
        raise NotImplementedError


@dataclass(frozen=True)
class MtrkEvent(MidiData):
    """MIDI event base class"""

    event_type: EventType
    delta_time: VariableLengthInt

    @property
    def time(self) -> int:
        """Returns the delta time in ticks"""
        return self.delta_time.value


def read_byte(infile: typing.BinaryIO):
    byte = infile.read(1)
    if byte == b'':
        raise MidiEOFError('EOF reached while reading byte')
    return ord(byte)


def read_bytes(infile: typing.BinaryIO, size: int, max_length: int = 1000000):
    if size > max_length:
        raise MidiDecodeError('Message length {} exceeds maximum length {}'.format(size, max_length))
    bts = [read_byte(infile) for _ in range(size)]
    return bts


def tick2second(tick: int, ticks_per_beat: int, tempo: int):
    return tick * tempo * 1e-6 / ticks_per_beat


def second2tick(second: float, ticks_per_beat: int, tempo: int):
    return int(round(second / tempo * 1e6 * ticks_per_beat))


def _fix_eot(messages: list[MtrkEvent]):
    from .events import MetaEvent
    from .meta import MetaEventType
    accum = 0
    msgs: list[MtrkEvent] = []

    for msg in messages:
        if isinstance(msg, MetaEvent) and msg.meta_type == MetaEventType.END_OF_TRACK:
            accum += msg.time
        else:
            if accum:
                delta = accum + msg.time
                msg = copy.copy(msg)
                object.__setattr__(msg, "delta_time", VariableLengthInt.from_int(delta))
                accum = 0
            msgs.append(msg)

    msgs.append(MetaEvent(
        VariableLengthInt.from_int(accum),
        meta_msg_type=MetaEventType.END_OF_TRACK.value,
        length=VariableLengthInt.from_int(0),
        data=bytes()
    ))
    return msgs


def merge_chunks(chunks: typing.Iterable[Chunk]) -> list[MtrkEvent]:
    """Merges all tracks in the chunks into a single track"""
    msgs: list[tuple[MtrkEvent, int]] = []  # (event, abstime)
    for chunk in chunks:
        t = 0
        for event in chunk.events:
            t += event.delta_time.value
            msgs.append((event, t))
    msgs.sort(key=lambda x: x[1])
    events: list[MtrkEvent] = []
    t = 0
    for event, abstime in msgs:
        delta = abstime - t
        delta_time = VariableLengthInt.from_int(delta)
        event = copy.copy(event)
        object.__setattr__(event, 'delta_time', delta_time)
        t = abstime
        events.append(event)
    return _fix_eot(events)
    # return events


def calculate_note_deltas(chunks: typing.Iterable[Chunk], ticks_per_beat: int) -> list[float]:
    """Calculates the music duration of the events in seconds"""
    from .events import MetaEvent, MidiEvent, MidiMessage
    from .message import NoteOnMessage, NoteOffMessage
    from .meta import MetaEventType, MetaEventSetTempo
    cum_t = 0
    tempo = 500000
    note_dts: list[float] = []
    for msg in merge_chunks(chunks):
        delta = tick2second(msg.time, ticks_per_beat, tempo) if msg.time > 0 else 0
        cum_t += delta
        # Only accumulate this final time if the event is a note on or note off
        if isinstance(msg, MidiEvent) and (
            isinstance(msg.message, NoteOnMessage) or
            isinstance(msg.message, NoteOffMessage)
        ):
            note_dts.append(cum_t)
            cum_t = 0
        if isinstance(msg, MetaEvent) and msg.meta_type == MetaEventType.SET_TEMPO:
            data = msg.get_event()
            assert isinstance(data, MetaEventSetTempo), f"Got {data}"
            tempo = data.tttttt
    return note_dts
