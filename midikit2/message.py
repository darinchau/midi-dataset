# Implements the MIDI message
from __future__ import annotations
import struct
import typing
import enum
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .exceptions import MidiDecodeError, MidiEOFError
from .base import MidiData


class MessageType(enum.IntEnum):
    """MIDI message types"""
    UNKNOWN = 0x00  # Reserve this for unknown messages - status bytes must have MSB set to 1 so this should be fine

    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    POLY_PRESSURE = 0xA0
    CONTROL_CHANGE = 0xB0
    PROGRAM_CHANGE = 0xC0
    CHANNEL_PRESSURE = 0xD0
    PITCH_BEND = 0xE0

    # Reserved for future use
    SYSTEM_EXCLUSIVE = 0xF0
    TIME_CODE_QUARTER_FRAME = 0xF1
    SONG_POSITION_POINTER = 0xF2
    SONG_SELECT = 0xF3
    TUNE_REQUEST = 0xF6
    CLOCK = 0xF8
    START = 0xFA
    CONTINUE = 0xFB
    STOP = 0xFC
    ACTIVE_SENSING = 0xFE


@dataclass(frozen=True)
class MessageSpec:
    status_byte: int
    size: int  # number of bytes. -1 means infinite, -2 is reserved for unknown

    @property
    def is_unknown(self) -> bool:
        """Returns True if this message type is unknown"""
        return self.size == -2

    @property
    def message_type(self) -> MessageType:
        """Returns the message type"""
        status_byte = self.status_byte & 0xF0 if 0x80 <= self.status_byte < 0xF0 else self.status_byte
        if status_byte in [t.value for t in MessageType]:
            return MessageType(status_byte)
        return MessageType.UNKNOWN


def lookup_message_type(status_byte: int) -> MessageSpec:
    """Looks up the message type for a given status byte."""
    assert 0 <= status_byte <= 0xFF, f"Status byte {status_byte} is out of range."

    # Use a dictionary to map status bytes to MessageSpec
    message_spec_map = {
        0x80: (MessageSpec(status_byte, 3), MessageType.NOTE_OFF),
        0x90: (MessageSpec(status_byte, 3), MessageType.NOTE_ON),
        0xA0: (MessageSpec(status_byte, 3), MessageType.POLY_PRESSURE),
        0xB0: (MessageSpec(status_byte, 3), MessageType.CONTROL_CHANGE),
        0xC0: (MessageSpec(status_byte, 2), MessageType.PROGRAM_CHANGE),
        0xD0: (MessageSpec(status_byte, 2), MessageType.CHANNEL_PRESSURE),
        0xE0: (MessageSpec(status_byte, 3), MessageType.PITCH_BEND),
        0xF0: (MessageSpec(status_byte, -1), MessageType.SYSTEM_EXCLUSIVE),
        0xF1: (MessageSpec(status_byte, 2), MessageType.TIME_CODE_QUARTER_FRAME),
        0xF2: (MessageSpec(status_byte, 3), MessageType.SONG_POSITION_POINTER),
        0xF3: (MessageSpec(status_byte, 2), MessageType.SONG_SELECT),
        0xF6: (MessageSpec(status_byte, 1), MessageType.TUNE_REQUEST),
        0xF8: (MessageSpec(status_byte, 1), MessageType.CLOCK),
        0xFA: (MessageSpec(status_byte, 1), MessageType.START),
        0xFB: (MessageSpec(status_byte, 1), MessageType.CONTINUE),
        0xFC: (MessageSpec(status_byte, 1), MessageType.STOP),
        0xFE: (MessageSpec(status_byte, 1), MessageType.ACTIVE_SENSING),
    }

    for key, (spec, t) in message_spec_map.items():
        if spec.message_type == t:
            return spec
    return MessageSpec(status_byte, -2)  # Unknown message type


@dataclass(frozen=True)
class ModeNumber:
    value: int

    def __post_init__(self):
        if not (0 <= self.value <= 127):
            raise MidiDecodeError(f"Mode number {self.value} is out of range.")


@dataclass(frozen=True)
class MidiMessage(MidiData):
    """Base class for MIDI messages"""
    msg_type: MessageSpec

    @staticmethod
    def from_bytes(bs: list[int]) -> MidiMessage:
        """Creates a MidiMessage from a list of bytes"""
        if len(bs) == 0:
            raise ValueError("Empty byte list")

        status_byte = bs[0]
        msg_type = lookup_message_type(status_byte)
        if msg_type.size != -1 and len(bs) != msg_type.size:
            raise ValueError(f"Byte list length {len(bs)} (from {msg_type.message_type.name}) does not match expected length {msg_type.size}: {bs}")

        if msg_type.message_type == MessageType.NOTE_ON:
            return NoteOnMessage(
                msg_type=msg_type,
                note=bs[1],
                velocity=bs[2],
            )
        if msg_type.message_type == MessageType.NOTE_OFF:
            return NoteOffMessage(
                msg_type=msg_type,
                note=bs[1],
                velocity=bs[2],
            )
        if msg_type.message_type == MessageType.CONTROL_CHANGE:
            return ControlChangeMessage(
                msg_type=msg_type,
                controller=bs[1],
                value=bs[2],
            )
        if msg_type.message_type == MessageType.PROGRAM_CHANGE:
            return ProgramChangeMessage(
                msg_type=msg_type,
                program=bs[1],
            )
        if msg_type.message_type == MessageType.PITCH_BEND:
            return PitchBendMessage(
                msg_type=msg_type,
                value=bs[1] | (bs[2] << 7),
            )
        if msg_type.message_type == MessageType.CHANNEL_PRESSURE:
            return ChannelPressureMessage(
                msg_type=msg_type,
                pressure=bs[1],
            )
        if msg_type.message_type == MessageType.POLY_PRESSURE:
            return PolyPressureMessage(
                msg_type=msg_type,
                note=bs[1],
                pressure=bs[2],
            )
        return UnknownMessage(
            msg_type=msg_type,
            data=bytes(bs[1:]),
        )


@dataclass(frozen=True)
class NoteOnMessage(MidiMessage):
    """Note On message class"""

    msg_type: MessageSpec
    note: int
    velocity: int

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.note <= 127):
            raise MidiDecodeError(f"Note {self.note} is out of range.")
        if not (0 <= self.velocity <= 127):
            raise MidiDecodeError(f"Velocity {self.velocity} is out of range.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>BBB', self.msg_type.status_byte, self.note, self.velocity)

    @property
    def is_note_off(self) -> bool:
        """Returns True if this should instruct an instrument to stop playing"""
        return self.velocity == 0


@dataclass(frozen=True)
class NoteOffMessage(MidiMessage):
    """Note Off message class"""

    msg_type: MessageSpec
    note: int
    velocity: int

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.note <= 127):
            raise MidiDecodeError(f"Note {self.note} is out of range.")
        if not (0 <= self.velocity <= 127):
            raise MidiDecodeError(f"Velocity {self.velocity} is out of range.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>BBB', self.msg_type.status_byte, self.note, self.velocity)


@dataclass(frozen=True)
class ControlChangeMessage(MidiMessage):
    """Control Change message class"""

    msg_type: MessageSpec
    controller: int
    value: int

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.controller <= 127):
            raise MidiDecodeError(f"Controller {self.controller} is out of range.")
        if not (0 <= self.value <= 127):
            raise MidiDecodeError(f"Value {self.value} is out of range.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>BBB', self.msg_type.status_byte, self.controller, self.value)


@dataclass(frozen=True)
class ProgramChangeMessage(MidiMessage):
    """Program Change message class"""

    msg_type: MessageSpec
    program: int

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.program <= 127):
            raise MidiDecodeError(f"Program {self.program} is out of range.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>BB', self.msg_type.status_byte, self.program)


@dataclass(frozen=True)
class PitchBendMessage(MidiMessage):
    """Pitch Bend message class"""

    msg_type: MessageSpec
    value: int

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.value <= 16383):
            raise MidiDecodeError(f"Value {self.value} is out of range.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>BB', self.msg_type.status_byte, self.value & 0x7F) + struct.pack('>B', (self.value >> 7) & 0x7F)


@dataclass(frozen=True)
class ChannelPressureMessage(MidiMessage):
    """Channel Pressure message class"""

    msg_type: MessageSpec
    pressure: int

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.pressure <= 127):
            raise MidiDecodeError(f"Pressure {self.pressure} is out of range.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>BB', self.msg_type.status_byte, self.pressure)


@dataclass(frozen=True)
class PolyPressureMessage(MidiMessage):
    """Poly Pressure message class"""

    msg_type: MessageSpec
    note: int
    pressure: int

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.note <= 127):
            raise MidiDecodeError(f"Note {self.note} is out of range.")
        if not (0 <= self.pressure <= 127):
            raise MidiDecodeError(f"Pressure {self.pressure} is out of range.")

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>BBB', self.msg_type.status_byte, self.note, self.pressure)


@dataclass(frozen=True)
class UnknownMessage(MidiMessage):
    """Unknown message class"""

    msg_type: MessageSpec
    data: bytes

    def get_data(self) -> bytes:
        """Returns the underlying data as bytes"""
        return struct.pack('>B', self.msg_type.status_byte) + self.data
