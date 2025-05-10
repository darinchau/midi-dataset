from __future__ import annotations
import os
import sys
import typing
from pathlib import Path
from .base import MidiData, merge_chunks, calculate_note_deltas
from .exceptions import MidiDecodeError, MidiEOFError
from .events import MtrkEvent, MidiEvent, MetaEvent, SysexEvent, read_header_chunk, read_track, TrackChunk, HeaderChunk, Chunk
from dataclasses import dataclass, field
from functools import cached_property
import line_profiler


class Midifile:
    """A class for midi files"""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._sanity_check()
        self._read()
        self._file_check()

    def __enter__(self):
        """Enter the context manager: Returns the raw bytes of the file"""
        self._file = open(self._path, "rb")
        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager"""
        self._file.close()
        return False

    def _sanity_check(self):
        """Check if the file is a valid midi file"""
        if not self._path.is_file():
            raise MidiDecodeError(f"File {self._path} does not exist")
        if self._path.suffix != ".mid" and self._path.suffix != ".midi":
            raise MidiDecodeError(f"File {self._path} is not a midi file: found {self._path.suffix}")

    @line_profiler.profile
    def _read(self):
        """Read the header and file chunks"""
        with self as f:
            self._header = read_header_chunk(f)
            self._tracks: list[TrackChunk] = []
            while not f.tell() == os.fstat(f.fileno()).st_size:
                track_chunk = read_track(f)
                self._tracks.append(track_chunk)

    def _file_check(self):
        """Check if the file is a valid midi file"""
        if not self._header:
            raise MidiDecodeError("Header chunk not found")
        if not self._tracks:
            raise MidiDecodeError("No tracks found")
        if self._header.format_type == 0 and len(self._tracks) != 1:
            raise MidiDecodeError("Format type 0 can only have 1 track")

    @property
    def format_type(self):
        """Format type of the midi file"""
        assert self._header.format_type in (0, 1, 2)
        return self._header.format_type

    @property
    def num_tracks(self):
        """Number of tracks in the midi file"""
        return self._header.num_tracks

    @property
    def division(self):
        """Division of the midi file"""
        return self._header.division

    @property
    def path(self) -> Path:
        """Path to the midi file"""
        return self._path

    @property
    def chunks(self) -> list[Chunk]:
        """Return a list of all the chunks in the file"""
        # Shallow copy the chunks and also apease the typechecker (shallow copy is good enough since chunk is immutable)
        s: list[Chunk] = []
        s.append(self._header)
        for track in self._tracks:
            s.append(track)
        return s

    @property
    def events(self):
        for track in self.chunks:
            for event in track.events:
                yield event

    @cached_property
    def length(self) -> float:
        """Return the music length of this midi in seconds"""
        secs = calculate_note_deltas(self.chunks, self._header.division)
        return sum(secs)
