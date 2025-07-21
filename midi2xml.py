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
lock = Lock()


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
        with lock:
            with open(LOGS_DIR, 'a') as log_file:
                log_file.write(f"Error converting {midi_path}: {(type(e))} {str(e)}\n")
        return False


def main():
    # Refresh the logs file
    with open(LOGS_DIR, 'w') as log_file:
        log_file.write(f"Conversion started at: {datetime.now()}\n")
    print(f"Logs will be saved to {LOGS_DIR}")

    for input_path, output_path in tqdm(iterate_midi_files(INPUT_DIR, OUTPUT_DIR), desc="Preparing MIDI files"):
        if not convert_one_file(input_path, output_path):
            tqdm.write(f"Failed to convert {input_path}.")


if __name__ == "__main__":
    main()
