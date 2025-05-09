import os
import mido
import json
import typing
from src import GiantMidiDataset
import vlc
import time

T = typing.TypeVar("T")


def analyse(f: typing.Callable[[str], T], name: str | None = None, num_threads: int = 16) -> None:
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
        num_threads=num_threads,
    )
    # Save result to a json file
    with open(path, "w") as file:
        json.dump(result, file)
    print(f"Saved result to {name}.json")


if __name__ == "__main__":
    import sys
    fn = sys.argv[1]
    f = vars()[fn]
    num_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    analyse(f, num_threads=num_threads)
