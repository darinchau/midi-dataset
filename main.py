from src import GiantMidiDataset
import mido
import json


def length(path: str) -> float:
    try:
        midi = mido.MidiFile(path)
        return midi.length
    except Exception as e:
        return -1


def main():
    ds = GiantMidiDataset("./giant-midi-archive")
    print(f"Number of files: {len(ds)}")

    result = ds.accumulate(
        length,
        num_threads=16,
        first_n=1000
    )
    # Save result to a json file
    with open("./giant-midi-archive/lengths.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
