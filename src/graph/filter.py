# Implements a filter and normalization criterion
# Stores the filter function we use to get clean XMLs in the current phase of training
from typing import List, Optional
from ..extract import musicxml_to_notes, MusicXMLNote


class MIDIFilterCriterion:
    def index_range(self) -> List[int]:
        """Returns the acceptable range of note indices."""
        return list(range(-15, 20))

    def pitch_range(self) -> List[int]:
        """Returns the acceptable range of note pitches."""
        return list(range(21, 109))

    def instrument_range(self) -> List[int]:
        """Returns the acceptable range of note instruments."""
        # Use a subset of the more common General MIDI instruments
        # 0: Acoustic Grand Piano
        # 1: Bright Acoustic Piano
        # 2: Electric Grand Piano
        # 6: Harpsichord
        # 40: Violin
        # 41: Viola
        # 42: Cello
        # 43: Contrabass
        return [0, 1, 2, 6, 40, 41, 42, 43]

    def get_bad_midi_reason(self, file_path: str | list[MusicXMLNote]) -> str:
        """Returns an empty string if the MIDI file is good, otherwise a reason for rejection."""
        if isinstance(file_path, str):
            try:
                notes = musicxml_to_notes(file_path, no_barline=True)
            except Exception as e:
                return "ParseError"
        else:
            notes = file_path

        if len(notes) < 100:
            return "TooFewNotes"

        if any(note.timesig is None for note in notes):
            return "MissingTimeSignature"

        index_range = self.index_range()
        if any(note.index not in index_range for note in notes):
            # Not very rigorous, but no triple sharps or flats allowed for now
            return "InvalidIndex"

        if any(note.timesig is None for note in notes):
            return "MissingTimeSignature"

        pitch_range = self.pitch_range()
        if any(not note.barline and (note.pitch not in pitch_range) for note in notes):
            # Outside piano range
            return "InvalidPitch"

        permitted_instruments = self.instrument_range()
        if any(not note.barline and note.instrument not in permitted_instruments for note in notes):
            return "InvalidInstrument"

        return ""


def is_good_midi(file_path: str | list[MusicXMLNote]) -> bool:
    """Returns True if the MIDI file is considered good based on our filtering criteria.
    Just a convenience wrapper around MIDIFilterCriterion."""
    return not MIDIFilterCriterion().get_bad_midi_reason(file_path)
