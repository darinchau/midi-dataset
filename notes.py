# A code snippet taken from my own classical-music representation library
# https://github.com/darinchau/classical_music/blob/main/src/reps/notes.py

from __future__ import annotations
import re
from dataclasses import dataclass
from functools import reduce
from mido import MidiFile, MidiTrack, Message
from typing import Literal
import typing


PIANO_A0 = 21
PIANO_C8 = 108
_PITCH_NAME_REGEX = re.compile(r"([CDEFGAB])(#+|b+)?(-?[0-9]+)")


@dataclass(frozen=True)
class Note:
    """A piano note is a representation of a note on the piano, with a note name and an octave
    The convention being middle C is C4. The lowest note is A0 and the highest note is C8.

    If the note is in real time, then the duration and offset is timed with respect to quarter length,
    otherwise it is timed with respect to real-time seconds."""
    index: int
    octave: int
    duration: float
    offset: float
    real_time: bool
    velocity: int

    def __post_init__(self):
        # Sanity Check
        assert PIANO_A0 <= self.midi_number <= PIANO_C8, f"Note must be between A0 and C8, but found {self.midi_number}"
        assert self.duration >= 0, f"Duration must be greater than or equal to 0, but found {self.duration}"
        assert self.offset >= 0, f"Offset must be greater than or equal to 0, but found {self.offset}"
        assert 0 <= self.velocity < 128, f"Velocity must be between 0 and 127, but found {self.velocity}"

    def __repr__(self):
        return f"Note({self.note_name}, duration={self.duration}, offset={self.offset}, velocity={self.velocity})"

    @property
    def pitch_name(self) -> str:
        """Returns a note name of the pitch. e.g. A, C#, etc."""
        alter = self.alter
        if alter == 0:
            return self.step
        elif alter == 2:
            return f"{self.step}x"
        elif alter > 0:
            return f"{self.step}{'#' * alter}"
        else:
            return f"{self.step}{'b' * -alter}"

    @property
    def note_name(self):
        """The note name of the note. e.g. A4, C#5, etc."""
        return f"{self.pitch_name}{self.octave}"

    @property
    def step(self) -> Literal["C", "D", "E", "F", "G", "A", "B"]:
        """Returns the diatonic step of the note"""
        idx = self.index % 7
        return ("C", "G", "D", "A", "E", "B", "F")[idx]

    @property
    def step_number(self) -> int:
        """Returns the diatonic step number of the note, where C is 0, D is 1, etc."""
        idx = self.index % 7
        return (0, 4, 1, 5, 2, 6, 3)[idx]

    @property
    def alter(self):
        """Returns the alteration of the note aka number of sharps. Flats are represented as negative numbers."""
        return (self.index + 1) // 7

    @property
    def pitch_number(self):
        """Returns the chromatic pitch number of the note. C is 0, D is 2, etc. There are edge cases like B# returning 12 or Cb returning -1"""
        return ([0, 2, 4, 5, 7, 9, 11][self.step_number] + self.alter)

    @property
    def midi_number(self):
        """The chromatic pitch number of the note, using the convention that A4=440Hz converts to 69
        This is also the MIDI number of the note."""
        return self.pitch_number + 12 * self.octave + 12

    def transpose(self, interval: int, compound: int = 0) -> Note:
        """Transposes the note by a given interval. The interval is given by the relative LOF index.
        So unison is 0, perfect fifths is 1, major 3rds is 4, etc.
        Assuming transposing up. If you want to transpose down, say a perfect fifth,
        then transpose up a perfect fourth and compound by -1."""
        new_index = self.index + interval
        # Make a draft note to detect octave changes
        draft_note = Note(
            index=new_index,
            octave=self.octave,
            duration=self.duration,
            offset=self.offset,
            real_time=self.real_time,
            velocity=self.velocity
        )
        new_octave = self.octave + compound
        if (draft_note.pitch_number % 12) < (self.pitch_number % 12):
            new_octave += 1
        return Note(
            index=new_index,
            octave=new_octave,
            duration=self.duration,
            offset=self.offset,
            real_time=self.real_time,
            velocity=self.velocity
        )

    @classmethod
    def from_str(cls, note: str, real_time: bool = True) -> Note:
        """Creates a Note from a string note.

        Example: A4[0, 1, 64] is A in the 4th octave with a duration of 0 and offset of 1 and velocity of 64.
        A4[0, 1] is A in the 4th octave with a duration of 0 and offset of 1.
        A4 is A in the 4th octave with a duration of 0 and offset of 0.
        A is A in the (implied) 4th octave with a duration of 0 and offset of 0."""
        duration = 0
        offset = 0
        velocity = 64
        if "[" in note:
            note, rest = note.split("[")
            rest = rest.rstrip("]")
            assert len(rest.split(",")) in (2, 3), f"Rest must have 2 or 3 elements, but found {len(rest.split(','))}"
            if len(rest.split(",")) == 3:
                duration, offset, velocity = rest.split(",")
                duration = float(duration)
                offset = float(offset)
                velocity = int(velocity)
            else:
                duration, offset = rest.split(",")
                duration = float(duration)
                offset = float(offset)

        match = _PITCH_NAME_REGEX.match(note)
        if not match:
            # Add the implied octave
            match = _PITCH_NAME_REGEX.match(note + "4")

        assert match and len(match.groups()) == 3, f"The name {note} is not a valid note name"
        pitch_name, alter, octave = match.groups()
        if alter is None:
            alter = ""
        alter = alter.replace("x", "##").replace("-", "b").replace("+", "#")
        sharps = reduce(lambda x, y: x + 1 if y == "#" else x - 1, alter, 0)
        assert pitch_name in ("C", "D", "E", "F", "G", "A", "B"), f"Step must be one of CDEFGAB, but found {pitch_name}"  # to pass the typechecker

        return cls(
            index=_step_alter_to_lof_index(pitch_name, sharps),
            octave=int(octave),
            duration=duration,
            offset=offset,
            real_time=real_time,
            velocity=velocity
        )

    @classmethod
    def from_midi_number(cls, midi_number: int, duration: float = 0., offset: float = 0., real_time: bool = True, velocity: int = 64) -> Note:
        """Creates a Note from a MIDI number. A4 maps to 69. If accidentals are needed, assumes the note is sharp."""
        octave = (midi_number // 12) - 1
        pitch = [0, 7, 2, 9, 4, -1, 6, 1, 8, 3, 10, 5][midi_number % 12]
        return cls(
            index=pitch,
            octave=octave,
            duration=duration,
            offset=offset,
            real_time=real_time,
            velocity=velocity
        )


class _Notes:
    def __init__(self, notes: list[Note], real_time: bool):
        assert all(note.real_time == real_time for note in notes), f"All notes must be {'real' if real_time else 'notated'} time"
        self._notes = sorted(notes, key=lambda x: (x.offset, x.duration))

    def __getitem__(self, index: int) -> Note:
        return self._notes[index]

    def __iter__(self):
        return iter(self._notes)

    def __bool__(self):
        return len(self._notes) > 0

    def normalize(self):
        min_offset = min(note.offset for note in self._notes)
        for note in self._notes:
            # Use python black magic - this is safe because the object only has reference here
            object.__setattr__(note, "offset", note.offset - min_offset)
        return self


class RealTimeNotes(_Notes):
    def __init__(self, notes: list[Note]):
        super().__init__(notes, real_time=True)

    def __add__(self, other: RealTimeNotes) -> RealTimeNotes:
        return RealTimeNotes(self._notes + other._notes)


class NotatedTimeNotes(_Notes):
    def __init__(self, notes: list[Note]):
        super().__init__(notes, real_time=False)

    def __add__(self, other: NotatedTimeNotes) -> NotatedTimeNotes:
        return NotatedTimeNotes(self._notes + other._notes)

    def to_real_time(self, tempo: float = 120.) -> RealTimeNotes:
        notes = self._notes
        assert all(not note.real_time for note in notes), "All notes must be timed against quarter length"
        return RealTimeNotes([Note(
            index=n.index,
            octave=n.octave,
            duration=n.duration * 60 / tempo,
            offset=n.offset * 60 / tempo,
            real_time=True,
            velocity=n.velocity
        ) for n in notes])


def _step_alter_to_lof_index(step: Literal["C", "D", "E", "F", "G", "A", "B"], alter: int) -> int:
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter
