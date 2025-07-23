import os
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import subprocess
from multiprocessing import Lock
from datetime import datetime
import argparse
import sys

INPUT_DIR = r"E:\data\giant-midi-archive"
OUTPUT_DIR = r"E:\data\giant-test-archive"
MUSESCORE_CLI = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
LOGS_DIR = "./logs.txt"


def iterate_midi_files(input_dir: str, output_dir: str, limit: int | None = None):
    count = 0
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
            count += 1
            if limit is not None and count >= limit:
                return


def convert_one_file(midi_path: Path, output_path: Path):
    # Adjust path if needed
    try:
        url = "http://localhost:8129/convert"
        response = requests.post(url, files={"file": open(midi_path, "rb")})
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            tqdm.write(f"Failed to convert {midi_path}: {response.status_code} {response.reason}")
            return False
    except Exception as e:
        try:
            tqdm.write(f"Error converting {midi_path}: {e}")
        except Exception:
            print(f"Error converting {midi_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert MIDI files to MusicXML format using MuseScore CLI.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of MIDI files to process.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes to use.")
    args = parser.parse_args()
    limit = args.limit
    assert isinstance(limit, int) or limit is None, "Limit must be an integer or None."
    if limit is not None and limit <= 0:
        print("Limit must be a positive integer or None.")
        sys.exit(1)

    workers = args.workers
    assert isinstance(workers, int) and workers > 0, "Number of workers must be a positive integer."
    if workers < 0:
        print("Number of workers must be a positive integer.")
        sys.exit(1)

    if workers == 0:
        for midi_path, output_path in tqdm(iterate_midi_files(INPUT_DIR, OUTPUT_DIR, limit=limit), desc="Converting MIDI files"):
            success = convert_one_file(midi_path, output_path)
            if not success:
                tqdm.write(f"Failed to convert {midi_path}")
    else:
        print(f"Using {workers} worker processes for conversion.")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(convert_one_file, midi_path, output_path): midi_path
                for midi_path, output_path in iterate_midi_files(INPUT_DIR, OUTPUT_DIR, limit=limit)
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
