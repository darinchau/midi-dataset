"""
Script to copy a subset of MIDI files from one directory to another, filtering out bad files.
"""

from src.utils import iterate_subset
import os
import shutil
from tqdm import tqdm
from src.graph.filter import MIDIFilterCriterion


def main():
    old_path = "E:/data/giant-xml-archive"
    new_path = "D:/data/giant-xml-archive"
    subset = ["v2", "classical", "xmidi"]

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    midi_filter = MIDIFilterCriterion()

    copied = 0
    bar = tqdm(iterate_subset(old_path, subset), desc="Copying files")
    for file_path in bar:
        subpath = os.path.relpath(file_path, old_path)
        new_file_path = os.path.join(new_path, subpath)
        if os.path.exists(new_file_path):
            copied += 1
            bar.set_postfix(copied=copied)
            tqdm.write(f"File already exists, skipping: {new_file_path}")
            continue

        rejection_reason = midi_filter.get_bad_midi_reason(file_path)
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
