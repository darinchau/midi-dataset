import os
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import subprocess
from multiprocessing import Lock
from datetime import datetime

INPUT_DIR = r"E:\data\giant-midi-archive"
OUTPUT_DIR = r"E:\data\giant-test-archive"
MUSESCORE_CLI = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
LOGS_DIR = "./logs.txt"


def iterate_midi_files(input_dir: str, output_dir: str):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(('.mid', '.midi')):
                continue
            idx = os.path.splitext(file)[0]
            input_path = Path(root) / file
            output_path = Path(output_dir) / Path(root).relative_to(input_dir) / f"{idx}.xml"
            if output_path.exists():
                # print(f"Skipping {input_path} as {output_path} already exists.")
                continue
            os.makedirs(output_path.parent, exist_ok=True)
            yield input_path, output_path


def convert_one_file(midi_path: Path, output_path: Path):
    # Adjust path if needed
    try:
        assert midi_path.exists(), f"MIDI file {midi_path} does not exist."
        result = subprocess.run([MUSESCORE_CLI, "-o", str(output_path), str(midi_path)], capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        try:
            tqdm.write(f"Error converting {midi_path}: {e}")
        except Exception as e2:
            f"Error converting {midi_path}: {e}"
        return False


def main():
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(convert_one_file, midi_path, output_path): midi_path
            for midi_path, output_path in iterate_midi_files(INPUT_DIR, OUTPUT_DIR)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting MIDI files"):
            midi_path = futures[future]
            try:
                success = future.result()
                if not success:
                    tqdm.write(f"Failed to convert {midi_path}")
            except Exception as e:
                tqdm.write(f"Exception occurred while converting {midi_path}: {e}")


if __name__ == "__main__":
    main()
