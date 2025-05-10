# Implements the MIDI message
from __future__ import annotations
import struct
import typing
import enum
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .exceptions import MidiDecodeError, MidiEOFError
from .message import MidiMessage, lookup_message_type, UnknownMessage
from .base import VariableLengthInt, read_byte, read_bytes, ChunkType, EventType, Chunk, MtrkEvent, MidiData
from functools import cached_property

if typing.TYPE_CHECKING:
    from .meta import MetaEventData


### Chunks ###


@dataclass(frozen=True)
class HeaderChunk(Chunk):
    """Header chunk class"""

    data: bytes = field(init=False)
    chunk_type: ChunkType = field(default=ChunkType.HEADER, init=False)
    format_type: int
    num_tracks: int
    division: int

    def __post_init__(self):
        if not (0 <= self.format_type <= 2):
            raise MidiDecodeError(f"Format type {self.format_type} is out of range.")
        if not (1 <= self.num_tracks <= 65535):
            raise MidiDecodeError(f"Number of tracks {self.num_tracks} is out of range.")
        if not (0 <= self.division <= 65535):
            raise MidiDecodeError(f"Division {self.division} is out of range.")
        if self.format_type == 0 and self.num_tracks != 1:
            raise MidiDecodeError("Format type 0 can only have 1 track.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>4sLHHH', b'MThd', 6, self.format_type, self.num_tracks, self.division)

    @property
    def events(self) -> typing.Iterator[MtrkEvent]:
        return iter([])


@dataclass(frozen=True)
class TrackChunk(Chunk):
    """Track chunk class"""
    chunk_type: ChunkType = field(default=ChunkType.TRACK, init=False)
    size: int
    data: bytes  # The raw data of the track chunk, excluding the header (aka the <MTrkEvent>+ part)

    @cached_property
    def _events(self) -> list[MtrkEvent]:
        """Returns the events in this track"""
        return decode_events(self.data)

    def __post_init__(self):
        if len(self.data) != self.size:
            raise MidiDecodeError('Track chunk size does not match data length')
        # Decode events once to check for errors
        decode_events(self.data)

    @property
    def events(self) -> typing.Iterator[MtrkEvent]:
        """Returns an iterator of events in this track"""
        for event in self._events:
            yield event

    def get_data(self) -> bytes:
        return struct.pack('>4sL', b'MTrk', self.size) + self.data

### Events ###


@dataclass(frozen=True)
class MidiEvent(MtrkEvent):
    """An Mtrk event that's a MIDI event"""
    event_type: EventType = field(default=EventType.MIDI_EVENT, init=False)
    message: MidiMessage

    def __post_init__(self):
        if not isinstance(self.message, MidiMessage):
            raise MidiDecodeError('Mtrk event is not a MIDI message')
        return super().__post_init__()

    def get_data(self) -> bytes:
        # Might not be exactly the same as the original data but we strip away the running status shenanigans
        return self.delta_time.data + self.message.get_data()


@dataclass(frozen=True)
class SysexEvent(MtrkEvent):
    """An Mtrk event that's a SysEx event"""
    event_type: EventType = field(default=EventType.SYSEX_EVENT, init=False)
    length: VariableLengthInt
    status_byte: int
    data: bytes  # Should be all the data after the length byte

    def __post_init__(self):
        if self.status_byte not in (0xF0, 0xF7):
            raise MidiDecodeError('Mtrk event is not a SysEx message')
        return super().__post_init__()

    @property
    def is_start_of_packet(self) -> bool:
        """Returns True if this is the start of a SysEx packet"""
        return self.data[0] == 0xF0

    @property
    def is_end_of_packet(self) -> bool:
        """Returns True if this is the end of a SysEx packet"""
        return self.data[0] == 0xF7

    def get_data(self) -> bytes:
        return self.delta_time.data + bytes([self.status_byte]) + self.length.data + self.data


@dataclass(frozen=True)
class MetaEvent(MtrkEvent):
    """An Mtrk event that's a meta event"""
    event_type: EventType = field(default=EventType.META_EVENT, init=False)
    meta_msg_type: int
    length: VariableLengthInt
    data: bytes  # The chunk after FF, type, and length

    def __post_init__(self):
        return super().__post_init__()

    def get_data(self) -> bytes:
        return self.delta_time.data + bytes([0xFF, self.meta_msg_type]) + self.length.data + self.data

    def get_event(self) -> MetaEventData:
        """Returns the meta event data"""
        from .meta import (
            MetaEventSequenceNumber,
            MetaEventText,
            MetaEventCopyright,
            MetaEventTrackName,
            MetaEventInstrumentName,
            MetaEventEndOfTrack,
            MetaEventSetTempo,
            MetaEventSMPTEOffset,
            MetaEventTimeSignature,
            MetaEventLyric,
            MetaEventMarker,
            MetaEventCuePoint,
            MetaEventKeySignature,
            MetaEventSequencerSpecific,
            MetaEventMIDIChannelPrefix,
            MetaEventType,
            MetaEventData
        )
        if self.meta_msg_type == 0x00:
            data = MetaEventSequenceNumber(self.data, int.from_bytes(self.data, 'big'))
        elif self.meta_msg_type == 0x01:
            data = MetaEventText(self.data, self.data.decode('utf-8'))
        elif self.meta_msg_type == 0x02:
            data = MetaEventCopyright(self.data, self.data.decode('utf-8'))
        elif self.meta_msg_type == 0x03:
            data = MetaEventTrackName(self.data, self.data.decode('utf-8'))
        elif self.meta_msg_type == 0x04:
            data = MetaEventInstrumentName(self.data, self.data.decode('utf-8'))
        elif self.meta_msg_type == 0x05:
            data = MetaEventLyric(self.data, self.data.decode('utf-8'))
        elif self.meta_msg_type == 0x06:
            data = MetaEventMarker(self.data, self.data.decode('utf-8'))
        elif self.meta_msg_type == 0x07:
            data = MetaEventCuePoint(self.data, self.data.decode('utf-8'))
        elif self.meta_msg_type == 0x20:
            data = MetaEventMIDIChannelPrefix(self.data, self.data[0])
        elif self.meta_msg_type == 0x2F:
            data = MetaEventEndOfTrack(self.data)
        elif self.meta_msg_type == 0x51:
            data = MetaEventSetTempo(self.data, int.from_bytes(self.data, 'big'))
        elif self.meta_msg_type == 0x54:
            data = MetaEventSMPTEOffset(self.data, self.data[0], self.data[1], self.data[2], self.data[3], self.data[4])
        elif self.meta_msg_type == 0x58:
            data = MetaEventTimeSignature(self.data, self.data[0], self.data[1], self.data[2], self.data[3])
        elif self.meta_msg_type == 0x59:
            data = MetaEventKeySignature(self.data, self.data[0], self.data[1])
        elif self.meta_msg_type == 0x7F:
            data = MetaEventSequencerSpecific(self.data)
        else:
            data = MetaEventData(self.data, MetaEventType.UNKNOWN)
        assert data.meta_type == MetaEventType.UNKNOWN or data.meta_type.value == self.meta_msg_type, f"Meta event mismatch: expected {self.meta_msg_type}, got {data.meta_type}"
        return data

    @property
    def meta_type(self):
        from .meta import MetaEventType
        if self.meta_msg_type in MetaEventType:
            return MetaEventType(self.meta_msg_type)
        return MetaEventType.UNKNOWN


def read_chunk_header(infile: typing.BinaryIO) -> tuple[bytes, int]:
    header = infile.read(8)
    if len(header) < 8:
        raise MidiEOFError('Not enough data to read chunk header')
    return struct.unpack('>4sL', header)


def read_header_chunk(infile: typing.BinaryIO) -> HeaderChunk:
    name, size = read_chunk_header(infile)
    if name != b'MThd':
        raise MidiDecodeError('Incorrect header for a MIDI file')
    if size != 6:
        raise MidiDecodeError('Incorrect header size for a MIDI file')
    data = infile.read(size)
    if len(data) < 6:
        raise MidiEOFError('Not enough data to read header chunk')
    format_, ntrks, division = struct.unpack('>hhh', data[:6])
    return HeaderChunk(format_, ntrks, division)


def read_track(infile: typing.BinaryIO) -> TrackChunk:
    """Reads the infile and returns a track chunk"""
    name, size = read_chunk_header(infile)
    if name != b'MTrk':
        raise MidiDecodeError(f'no MTrk header at start of track: found {name}')
    data = infile.read(size)
    if len(data) < size:
        raise MidiEOFError('EOF reached')
    return TrackChunk(size, data)


def read_message(infile: io.BytesIO, status_byte: int, peek_data: list[int], delta: VariableLengthInt):
    msgtype = lookup_message_type(status_byte)
    if msgtype.is_unknown:
        # Read until the next status byte
        data_buffer: list[int] = []
        for p in peek_data:
            data_buffer.append(p)
        peek_data.clear()
        while True:
            byte = read_byte(infile)
            if byte >= 0x80:
                # Found a new status byte
                infile.seek(-1, io.SEEK_CUR)
                final_buffer = bytes(data_buffer)
                return MidiEvent(
                    delta_time=delta,
                    message=UnknownMessage(
                        msg_type=msgtype,
                        data=final_buffer,
                    ),
                )
            data_buffer.append(byte)

    # Subtract 1 for status byte.
    size = msgtype.size - 1 - len(peek_data)
    data_bytes = peek_data + read_bytes(infile, size)

    return MidiEvent(
        delta_time=delta,
        message=MidiMessage.from_bytes([status_byte] + data_bytes)
    )


def decode_events(data: bytes) -> list[MtrkEvent]:
    size = len(data)
    infile = io.BytesIO(data)
    start = infile.tell()
    last_status = None
    track: list[MtrkEvent] = []

    # Necessary to implement running status - fills in the missing status byte when needed
    peek_data: list[int] = []

    while True:
        assert infile.tell() - start <= size

        if infile.tell() - start == size:
            break

        delta_time = VariableLengthInt.read(infile)

        status_byte = read_byte(infile)

        if status_byte < 0x80:
            if last_status is None:
                raise MidiDecodeError('running status without last_status')
            peek_data = [status_byte]
            status_byte = last_status
        else:
            if status_byte != 0xff:
                last_status = status_byte
            peek_data = []

        msg: MtrkEvent

        # TODO process abrupt change status if needed
        if status_byte == 0xff:
            meta_type = read_byte(infile)
            length = VariableLengthInt.read(infile)
            data_ = read_bytes(infile, length.value)
            msg = MetaEvent(delta_time, meta_type, length, bytes(data_))
        elif status_byte in [0xf0, 0xf7]:
            length = VariableLengthInt.read(infile)
            data_ = read_bytes(infile, length.value)
            msg = SysexEvent(delta_time, length, status_byte, bytes(data_))
        else:
            msg = read_message(infile, status_byte, peek_data, delta_time)
        track.append(msg)

    return track
