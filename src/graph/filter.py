# Stores the filter function we use to get clean XMLs in the current phase of training
from typing import List, Optional
from ..extract import musicxml_to_notes, MusicXMLNote


def get_bad_midi_reason(file_path: str) -> str:
    """Attempt to get the cleanest XMLs from within our dataset. Returns empty string if good, otherwise a reason for rejection."""
    try:
        notes = musicxml_to_notes(file_path, no_barline=True)
    except Exception as e:
        return "ParseError"

    if len(notes) < 100:
        return "TooFewNotes"

    if any(note.timesig is None for note in notes):
        return "MissingTimeSignature"

    if any(note.index > 20 or note.index < -15 for note in notes):
        # Not very rigorous, but no triple sharps or flats allowed for now
        return "InvalidIndex"

    if any(note.timesig is None for note in notes):
        return "MissingTimeSignature"

    if any(not note.barline and (note.pitch < 21 or note.pitch > 108) for note in notes):
        # Outside piano range
        return "InvalidPitch"

    # Use a subset of the more common General MIDI instruments
    # 0: Acoustic Grand Piano
    # 1: Bright Acoustic Piano
    # 2: Electric Grand Piano
    # 6: Harpsichord
    # 40: Violin
    # 41: Viola
    # 42: Cello
    # 43: Contrabass
    permitted_instruments = [0, 1, 2, 6, 40, 41, 42, 43]
    if any(not note.barline and note.instrument not in permitted_instruments for note in notes):
        return "InvalidInstrument"

    return ""


def is_good_midi(file_path: str) -> bool:
    """Returns True if the MIDI file is considered good based on our filtering criteria."""
    return not get_bad_midi_reason(file_path)
