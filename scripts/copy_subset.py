from src.utils import iterate_subset
import os
import shutil
from tqdm import tqdm
from src.extract import musicxml_to_notes
import xml.etree.ElementTree as ET


def is_good_midi(file_path: str) -> str:
    """Attempt to get the cleanest XMLs from within our dataset."""
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


def main():
    old_path = "E:/data/giant-xml-archive"
    new_path = "D:/data/giant-xml-archive"
    subset = ["v2", "classical", "xmidi"]

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    copied = 0
    bar = tqdm(iterate_subset(old_path, subset), desc="Copying files")
    for file_path in bar:
        subpath = os.path.relpath(file_path, old_path)
        new_file_path = os.path.join(new_path, subpath)
        if os.path.exists(new_file_path):
            copied += 1
            tqdm.write(f"File already exists, skipping: {new_file_path}")
            continue

        rejection_reason = is_good_midi(file_path)
        if rejection_reason:
            tqdm.write(f"Skipping bad MIDI file: {file_path} ({rejection_reason})")
            continue

        new_file_dir = os.path.dirname(new_file_path)
        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)
        shutil.copy2(file_path, new_file_path)
        copied += 1
        bar.set_postfix(copied=copied)


if __name__ == "__main__":
    main()
