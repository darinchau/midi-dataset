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
import typing as t
from collections import defaultdict
from threading import Lock
from multiprocessing import Lock as ProcessLock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    exts: tuple[str, ...] = ('.mid', '.midi'),
    verbose: bool = True,
) -> t.Generator[str, None, None]:
    """Returns a list of all MIDI files in the directory and subdirectories."""
    midi_files_found = tqdm(desc=" MIDI files found...", unit="file", leave=True, disable=not verbose)
    for ext in exts:
        for path in Path(base_path).rglob('*' + ext):
            yield str(path)
            midi_files_found.update(1)
    midi_files_found.close()


def compute_hash(file_path: str) -> tuple[str, str]:
    """Return (file_path, md5hex) or (file_path, '') on error."""
    try:
        with open(file_path, 'rb') as f:
            return file_path, hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return file_path, ""

def deduplicate_files(
    files: t.Iterable[str],
    n_processes: int = 1,
    total: t.Optional[int] = None,
    use_threads: bool = False,
) -> t.Generator[t.Tuple[str, str], None, None]:
    """
    Consume an iterable/generator of file paths, compute MD5 in parallel,
    and yield (file_path, hash) for unique contents as soon as ready.

    - files: iterable or generator of file paths
    - n_processes: number of worker processes (0 or 1 -> run in main process)
    - total: optional count for tqdm progress bar
    - use_threads: set True to use threads instead of processes
    """
    used_hashes = set()

    # Single-process (no parallelism)
    if n_processes in (0, 1):
        for file_path in tqdm(files, total=total, desc="Hashing files", unit="file"):
            _, file_hash = compute_hash(file_path)
            if file_hash and file_hash not in used_hashes:
                used_hashes.add(file_hash)
                yield file_path, file_hash
        return

    # Parallel path: use processes by default, threads if requested
    ExecutorClass = ProcessPoolExecutor
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor
        ExecutorClass = ThreadPoolExecutor

    # We submit tasks as we iterate over files to support generators and limit memory.
    # Keep a small queue of in-flight futures to avoid overwhelming the system.
    max_in_flight = max(n_processes * 4, 16)  # modest buffer
    in_flight = set()

    with ExecutorClass(max_workers=n_processes) as executor:
        pbar = tqdm(desc="Hashing files", unit="file", total=total)

        def drain_completed(block: bool = False):
            nonlocal in_flight
            done = []
            for fut in as_completed(in_flight, timeout=None if block else 0):
                done.append(fut)
            for fut in done:
                in_flight.remove(fut)
                file_path, file_hash = fut.result()
                pbar.update(1)
                if file_hash and file_hash not in used_hashes:
                    used_hashes.add(file_hash)
                    yield file_path, file_hash

        # Submit tasks progressively
        for file_path in files:
            # throttle submissions
            while len(in_flight) >= max_in_flight:
                # yield any completed results without blocking too long
                yield from drain_completed(block=False)
                if len(in_flight) >= max_in_flight:
                    # if still full, block until at least one completes
                    yield from drain_completed(block=True)

            in_flight.add(executor.submit(compute_hash, file_path))

        # After submissions complete, drain remaining futures
        while in_flight:
            yield from drain_completed(block=True)

        pbar.close()

    print(f"Total unique files: {len(used_hashes)}")


def make_mapping(deduped_files: t.Iterator[tuple[str, str]], csv_file_path: str):
    mapping = {index: path for path, index in deduped_files}
    indices: list[str] = sorted(mapping.keys())
    values: list[str] = [mapping[index].replace('\\', '/') for index in indices]
    df = pd.DataFrame({
        "index": indices,
        "original_path": values
    })
    df.to_csv(csv_file_path, index=False, sep="\t", encoding="utf-8")
    return mapping


def copy_and_rename_files(deduped_files: t.Iterator[tuple[str, str]], target_dir: str, name_dir_hierachies: tuple[int, ...]):
    """Copy files to a new directory with new names."""
    for original_path, index in tqdm(deduped_files, desc="Copying files", unit="file"):
        new_name = f"{index}.mid"
        parent_dirs = [index[:i] for i in name_dir_hierachies]
        new_path = os.path.join(target_dir, *parent_dirs, new_name)
        try:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copyfile(original_path, new_path)
            yield original_path, index
        except Exception as e:
            logging.error(f"Error copying file {original_path} to {new_path}: {e}")


def main(base_directory: str, target_directory: str, exts: tuple[str, ...], unrar_path: str, name_dir_hierachies: tuple[int, ...], n_processes: int):
    """Create a dataset from a directory.

    The target directory can be an existing dataset. We will try to process only the new files that are added since the last run.
    The existing labels will be matched to their MD5 hashes and labels will be updated accordingly."""
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f"Directory {base_directory} does not exist.")
    if not os.path.isdir(base_directory):
        raise NotADirectoryError(f"{base_directory} is not a directory.")

    csv_file_path = os.path.join(target_directory, "mapping.csv")
    os.makedirs(target_directory, exist_ok=True)

    extract_archives(base_directory, unrar_path)
    midi_files = find_midi_files(base_directory, exts=exts)
    unique_files = deduplicate_files(midi_files, n_processes=n_processes)
    unique_files = copy_and_rename_files(unique_files, target_directory, name_dir_hierachies)
    processed = make_mapping(unique_files, csv_file_path)
    logging.info(f"Mapping saved to CSV file at {csv_file_path}")
    logging.info(f"Files copied and renamed. Total: {len(processed)}")


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Create a MIDI dataset from a directory.")
    parser.add_argument("--root", type=str, default="D:/data/raw-midi-data", help="Root directory to process.")
    parser.add_argument("--target_directory", type=str, default="E:/data/giant-midi-archive", help="Target directory for the dataset.")
    parser.add_argument("--unrar_path", type=str, default="C:/Program Files/WinRAR/UnRAR.exe", help="Path to the UnRAR executable.")
    parser.add_argument("--exts", type=str, nargs="+", default=['.mid', '.midi'], help="MIDI file extensions to include.")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of processes for parallel processing.")

    args = parser.parse_args()

    dataset = main(
        base_directory=args.root,
        target_directory=args.target_directory,
        unrar_path=args.unrar_path,
        exts=tuple(args.exts),
        n_processes=args.n_processes,
        name_dir_hierachies=(2, 4)
    )
