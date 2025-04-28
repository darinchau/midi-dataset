import os
import mido
import json
import typing
from src import GiantMidiDataset
import vlc
import time

T = typing.TypeVar("T")


def instruments(midi_path: str) -> list[int]:
    try:
        mid = mido.MidiFile(midi_path)
        instrument_set = []

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    instrument_name = msg.program
                    instrument_set.append(instrument_name)
        return instrument_set
    except Exception as e:
        return []


def length(file_path):
    vlc_options = [
        f"--soundfont=./resources/gs.sf2",
        "--no-audio",
        "--quiet"
    ]

    instance = vlc.Instance(vlc_options)
    player = instance.media_player_new()  # type: ignore
    media = instance.media_new(file_path)  # type: ignore
    player.set_media(media)
    player.play()

    for i in range(100):
        duration = player.get_length()
        if duration <= 1e-5:
            time.sleep(0.1)
        else:
            break
    else:
        return -1
    player.stop()
    player.release()

    if duration <= 1e-5:
        return -1
    return duration / 1000.0


def analyse(f: typing.Callable[[str], T], name: str | None = None) -> None:
    """Analyse the dataset with a function and save the result to a json file.

    The name of the json file will be the name of the function by default"""
    ds = GiantMidiDataset("./giant-midi-archive")

    if name is None:
        name = f.__name__
    print(f"Analysing {len(ds)} files with {name}")

    path = f"./giant-midi-archive/{name}.json"
    if os.path.exists(path):
        print(f"File {path} already exists, skipping")
        return

    result = ds.accumulate(
        f,
        num_threads=16,
    )
    # Save result to a json file
    with open(path, "w") as file:
        json.dump(result, file)
    print(f"Saved result to {name}.json")


if __name__ == "__main__":
    analyse(length)
