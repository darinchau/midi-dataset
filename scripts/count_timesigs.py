import os
from functools import partial
from multiprocessing import Pool, cpu_count
from src.util import iterate_dataset
from src.constants import *
from src.util import get_time_signature_map
from src.extract.analyze import musicxml_to_tokens as parse_musicxml
import numpy as np
import traceback
from tqdm.auto import tqdm
from collections import defaultdict, Counter


def main():
    time_sig_counts = defaultdict(int)

    for file_path in iterate_dataset(XML_ROOT):
        try:
            print(f"Processing file: {file_path}")
            notes_data = parse_musicxml(file_path)
            for note in notes_data:
                if not note.barline and note.timesig:  # Skip barlines and notes without timesig
                    time_sig_counts[note.timesig] += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            traceback.print_exc()
            continue


def process_single_file(file_path, debug=False):
    """
    Process a single MusicXML file and return time signature counts.
    """
    time_sigs = []
    try:
        notes_data = parse_musicxml(file_path, debug=debug)
        for note in notes_data:
            if not note.barline and note.timesig:  # Skip barlines and notes without timesig
                time_sigs.append(note.timesig)

        return file_path, time_sigs, None
    except Exception as e:
        error_msg = f"Error processing file {file_path}: {e}\n{traceback.format_exc()}"
        return file_path, [], error_msg


def aggregate_counts(results, time_sig_counts):
    """
    Aggregate time signature counts from multiple files.
    """
    for file_path, time_sigs, error in results:
        if error:
            print(error)
        else:
            for time_sig in time_sigs:
                time_sig_counts[time_sig] += 1


def main_parallel(n_workers=16, chunk_size=10):
    """
    Parallel version using multiprocessing.

    Args:
        n_workers: Number of worker processes. Default = 16
        chunk_size: Number of files to process in each batch
    """
    print(f"Using {n_workers} worker processes")

    # Get all file paths
    file_paths = list(tqdm(iterate_dataset(XML_ROOT), desc="Collecting file paths", unit="file"))
    total_files = len(file_paths)
    print(f"Found {total_files} files to process")

    # Initialize time signature counter
    time_sig_counts = Counter()

    # Process files in parallel with progress bar
    with Pool(processes=n_workers) as pool:
        # Use imap for better memory efficiency and progress tracking
        process_func = partial(process_single_file, debug=False)

        # Progress bar with more detailed information
        with tqdm(total=total_files,
                  desc="Processing files",
                  unit="files",
                  smoothing=0.1) as pbar:
            # Process in chunks for better performance
            results_batch = []
            errors_count = 0

            for result in pool.imap(process_func, file_paths, chunksize=chunk_size):
                results_batch.append(result)

                # Count errors
                if result[2] is not None:  # error occurred
                    errors_count += 1

                # Update progress bar with additional stats
                pbar.update(1)
                pbar.set_postfix({
                    'errors': errors_count,
                    'batch_size': len(results_batch)
                })

                # Aggregate results periodically to avoid memory buildup
                if len(results_batch) >= 100:
                    aggregate_counts(results_batch, time_sig_counts)
                    results_batch = []

            # Aggregate remaining results
            if results_batch:
                aggregate_counts(results_batch, time_sig_counts)

    # Print summary statistics
    print(f"\nProcessing complete!")
    if errors_count > 0:
        print(f"Files with errors: {errors_count}")

    # Print time signature statistics
    print(f"\n{'='*60}")
    print("TIME SIGNATURE STATISTICS:")
    print(f"{'='*60}")
    print(f"Total distinct time signatures found: {len(time_sig_counts)}")
    print(f"Total time signature annotations: {sum(time_sig_counts.values()):,}")

    # Sort time signatures by frequency
    sorted_time_sigs = sorted(time_sig_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nAll time signatures found (sorted by frequency):")
    for i, (time_sig, count) in enumerate(sorted_time_sigs):
        percentage = (count / sum(time_sig_counts.values())) * 100
        print(f"  {i+1:2d}. {time_sig:8s}: {count:10,} occurrences ({percentage:6.2f}%)")

    time_sig_map = get_time_signature_map()
    mapped_sigs = set(v for v in time_sig_map.values() if v not in ["UNK", "unused"])

    print(f"\n{'='*60}")
    print("TIME SIGNATURE MAPPING COVERAGE:")
    print(f"{'='*60}")
    print(f"Time signatures in our mapping: {sorted(mapped_sigs)}")

    unmapped_sigs = []
    for time_sig, count in sorted_time_sigs:
        if time_sig not in mapped_sigs and time_sig != "UNK":
            unmapped_sigs.append((time_sig, count))

    if unmapped_sigs:
        print(f"\nTime signatures NOT in our mapping:")
        total_unmapped = sum(count for _, count in unmapped_sigs)
        unmapped_percentage = (total_unmapped / sum(time_sig_counts.values())) * 100
        print(f"Total unmapped occurrences: {total_unmapped:,} ({unmapped_percentage:.2f}%)")
        for time_sig, count in unmapped_sigs:
            percentage = (count / sum(time_sig_counts.values())) * 100
            print(f"  {time_sig:8s}: {count:10,} occurrences ({percentage:6.2f}%)")
    else:
        print("\nAll found time signatures are covered by our mapping!")

    return time_sig_counts


if __name__ == "__main__":
    # Use 16 workers
    time_sig_counts = main_parallel(n_workers=16, chunk_size=10)

    # Save time signature counts
    import json
    with open('time_signature_counts.json', 'w') as f:
        json.dump(dict(time_sig_counts), f, indent=2)
    print("\nSaved time signature counts to 'time_signature_counts.json'")
