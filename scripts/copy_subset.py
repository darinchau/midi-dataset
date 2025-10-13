from src.utils import iterate_subset
import os
import shutil
from tqdm import tqdm
from src.extract import musicxml_to_notes
import xml.etree.ElementTree as ET


def is_good_midi(file_path: str) -> bool:
    try:
        notes = musicxml_to_notes(file_path)
    except ET.ParseError:
        return False

    if len(notes) < 100:
        return False

    if any(note.timesig is None for note in notes):
        return False

    if all(note.barline for note in notes):
        return False

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
        return False

    return True


def main():
    old_path = "E:/data/giant-xml-archive"
    new_path = "D:/data/giant-xml-archive"
    subset = ["v2", "classical", "xmidi"]

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for file_path in tqdm(iterate_subset(old_path, subset), desc="Copying files"):
        subpath = os.path.relpath(file_path, old_path)
        new_file_path = os.path.join(new_path, subpath)
        if os.path.exists(new_file_path):
            tqdm.write(f"File already exists, skipping: {new_file_path}")
            continue
        if not is_good_midi(file_path):
            tqdm.write(f"Skipping bad MIDI file: {file_path}")
            continue
        new_file_dir = os.path.dirname(new_file_path)
        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)
        shutil.copy2(file_path, new_file_path)


if __name__ == "__main__":
    main()
