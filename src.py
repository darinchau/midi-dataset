from __future__ import annotations
import os
import logging
import filecmp
import hashlib
import shutil
import zipfile
import json
import random
import string
import tempfile
import typing
from collections import defaultdict
from threading import Lock
from multiprocessing import Lock as ProcessLock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

ILLEGAL_WINDOWS_NAMES = ("con", "prn", "nul", "aux", "com", "lpt", "lst")


class _ExistsAlready(Exception):
    """Custom exception for when a file already exists."""
    pass


def _safe_write_json(data, filename):
    temp_fd, temp_path = tempfile.mkstemp()

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, indent=4, ensure_ascii=False)
        shutil.move(temp_path, filename)
    except Exception as e:
        print(f"Failed to write data: {e}")
        os.unlink(temp_path)


def extract_zip(file_path: str, root: str) -> str:
    """Extracts a zip file to a specified directory. Return the path to the extracted directory."""
    extract_to = os.path.join(root, os.path.splitext(os.path.basename(file_path))[0])
    if os.path.exists(extract_to) and os.listdir(extract_to):
        raise _ExistsAlready(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def extract_rar(file_path: str, root: str, unrar_path: str) -> str:
    """Extracts a rar file to a specified directory. Return the path to the extracted directory."""
    try:
        import rarfile  # Import rarfile only if needed to avoid unnecessary dependency
    except ImportError:
        raise ImportError("Please install rarfile: pip install rarfile")
    rarfile.UNRAR_TOOL = unrar_path
    extract_to = os.path.join(root, os.path.splitext(os.path.basename(file_path))[0])
    if os.path.exists(extract_to) and os.listdir(extract_to):
        raise _ExistsAlready(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    with rarfile.RarFile(file_path) as rar_ref:
        rar_ref.extractall(extract_to)
    return extract_to


def extract_archives(base_path: str, unrar_path: str) -> None:
    """Finds and extracts all zip and rar files in the directory and subdirectories."""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith('.zip'):
                try:
                    path = extract_zip(full_path, root)
                    logging.info(f"Extracted {full_path}")
                except _ExistsAlready as e:
                    path = e.args[0]
                except zipfile.BadZipFile as e:
                    logging.warning(f"Bad zip file: {full_path}: {e}")
                    continue
                extract_archives(path, unrar_path)
            elif file.endswith('.rar'):
                try:
                    path = extract_rar(full_path, root, unrar_path)
                    logging.info(f"Extracted {full_path}")
                except _ExistsAlready as e:
                    path = e.args[0]
                except Exception as e:
                    logging.warning(f"Bad rar file: {full_path}: {e}")
                    continue
                extract_archives(path, unrar_path)


def find_midi_files(
    base_path: str,
    exts: tuple[str, ...] = ('.mid', '.midi', '.kar', '.rmi'),
    verbose: bool = True,
) -> list[str]:
    """Returns a list of all MIDI files in the directory and subdirectories."""
    midi_files = []
    midi_files_found = tqdm(desc=" MIDI files found...", unit="file", leave=True, disable=not verbose)
    non_midi_extensions = set()
    for ext in exts:
        for path in Path(base_path).rglob('*' + ext):
            midi_files.append(str(path))
            midi_files_found.update(1)
    for path in Path(base_path).rglob('*'):
        if path.suffix not in exts and path.is_file():
            non_midi_extensions.add(path.suffix)
    if verbose:
        logging.info(f"Non-MIDI file extensions found: {sorted(non_midi_extensions)}")
    midi_files_found.close()
    return midi_files


def deduplicate_files(root: str, file_paths: list[str], n_processes: int = 1) -> dict:
    """Remove duplicate files based on file content, adding a progress bar for hashing, and return a mapping of original to new file paths."""
    def compute_hash(file_path):
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest(), file_path
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return "", file_path

    hashes: dict[str, list[str]] = defaultdict(list)

    if n_processes == 0:
        # Single-threaded hashing
        for file_path in tqdm(file_paths, desc="Hashing files", unit="file"):
            file_hash, path = compute_hash(file_path)
            if file_hash:
                hashes[file_hash].append(path)
    else:
        # Multi-threaded hashing
        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            for file_hash, file_path in tqdm(
                executor.map(compute_hash, file_paths),
                desc="Hashing files",
                unit="file",
                total=len(file_paths)
            ):
                if file_hash:
                    hashes[file_hash].append(file_path)

    logging.info(f"Found {len(hashes)} unique file hashes.")

    assert False, "TODO: Implement logic that skips the files that are already in the dataset. Remember to handle hash collisions correctly"

    def process_collision(hash_val: str, paths: list[str]):
        local_unique_files: dict[str, str] = {}
        if any(path in paths_with_deepseek_labels for path in paths):
            for path in paths:
                if path in paths_with_deepseek_labels:
                    local_unique_files[path] = hash_val
                    break
        else:
            first_path = paths[0]
            local_unique_files[first_path] = hash_val
        return local_unique_files

    if n_processes == 0:
        # Single-threaded collision processing
        for hash_val, paths in tqdm(hashes.items(), desc="Processing collisions", unit="group"):
            unique_files.update(process_collision(hash_val, paths))
    else:
        # Multi-threaded collision processing
        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            results = list(
                tqdm(
                    executor.map(lambda item: process_collision(*item), hashes.items()),
                    desc="Processing collisions",
                    unit="group",
                    total=len(hashes),
                )
            )
        for result in results:
            unique_files.update(result)
    return unique_files


def make_mapping(unique_files: dict[str, str], csv_file_path: str):
    assert False, "TODO: evaluate whether we need to handle the case where the mapping csv already exists"
    mapping = {v: k for k, v in unique_files.items()}
    indices: list[str] = sorted(mapping.keys())
    values: list[str] = [mapping[index] for index in indices]
    df = pd.DataFrame({
        "index": indices,
        "original_path": values
    })
    df.to_csv(csv_file_path, index=False, sep="\t", encoding="utf-8")


def copy_and_rename_files(unique_files: dict[str, str], target_dir: str, name_dir_hierachies: tuple[int, ...]):
    """Copy files to a new directory with new names and return a mapping of new names to original paths."""
    for original_path, index in tqdm(unique_files.items(), desc="Copying files", unit="file"):
        new_name = f"{index}.mid"
        parent_dirs = [index[:i] for i in name_dir_hierachies]
        new_path = os.path.join(target_dir, *parent_dirs, new_name)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copyfile(original_path, new_path)


def main(root: str, target_directory: str, exts: tuple[str, ...], unrar_path: str, name_dir_hierachies: tuple[int, ...], n_processes: int):
    """Create a dataset from a directory.

    The target directory can be an existing dataset. We will try to process only the new files that are added since the last run.
    The existing labels will be matched to their MD5 hashes and labels will be updated accordingly."""
    if not os.path.exists(root):
        raise FileNotFoundError(f"Directory {root} does not exist.")
    if not os.path.isdir(root):
        raise NotADirectoryError(f"{root} is not a directory.")

    csv_file_path = os.path.join(target_directory, "mapping.csv")

    base_directory = "./data"

    extract_archives(base_directory, unrar_path)
    midi_files = find_midi_files(base_directory, exts=exts)
    logging.info(f"Found MIDI files: {len(midi_files)}")

    unique_files = deduplicate_files(target_directory, midi_files, n_processes=n_processes)
    logging.info(f"Unique MIDI files after deduplication: {len(unique_files)}")

    os.makedirs(target_directory, exist_ok=True)

    make_mapping(unique_files, csv_file_path)
    logging.info(f"Mapping saved to CSV file at {csv_file_path}")

    copy_and_rename_files(unique_files, target_directory, name_dir_hierachies)
    logging.info(f"Files copied and renamed. Total: {len(unique_files)}")


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Create a MIDI dataset from a directory.")
    parser.add_argument("--root", type=str, default="./data", help="Root directory to process.")
    parser.add_argument("--target_directory", type=str, default="E:/giant-midi-archive", help="Target directory for the dataset.")
    parser.add_argument("--unrar_path", type=str, default="C:/Program Files/WinRAR/UnRAR.exe", help="Path to the UnRAR executable.")
    parser.add_argument("--exts", type=str, nargs="+", default=['.mid', '.midi'], help="MIDI file extensions to include.")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of processes for parallel processing.")

    args = parser.parse_args()

    dataset = main(
        root=args.root,
        target_directory=args.target_directory,
        unrar_path=args.unrar_path,
        exts=tuple(args.exts),
        n_processes=args.n_processes,
        name_dir_hierachies=(2, 4)
    )
