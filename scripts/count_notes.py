import os
from functools import partial
from multiprocessing import Pool, cpu_count
from src.util import iterate_dataset
from src.constants import *
from src.extract.analyze import musicxml_to_tokens as parse_musicxml
import numpy as np
import traceback
from tqdm.auto import tqdm


def main():
    # instrument, pitch, velocity
    counts = np.zeros((128, 128, 128), dtype=int)
    for file_path in iterate_dataset(XML_ROOT):
        try:
            print(f"Processing file: {file_path}")
            notes_data = parse_musicxml(file_path)
            for note in notes_data:
                instrument = note.instrument
                pitch = note.pitch
                velocity = note.velocity
                counts[instrument, pitch, velocity] += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            traceback.print_exc()
            continue


def process_single_file(file_path, debug=False):
    """
    Process a single MusicXML file and return counts.
    Returns a sparse representation of counts to save memory.
    """
    counts_sparse = []
    try:
        notes_data = parse_musicxml(file_path, debug=debug)
        for note in notes_data:
            instrument = min(max(0, note.instrument), 127)  # Ensure within bounds
            pitch = min(max(0, note.pitch), 127)
            velocity = min(max(0, note.velocity), 127)
            counts_sparse.append((instrument, pitch, velocity))
        return file_path, counts_sparse, None
    except Exception as e:
        error_msg = f"Error processing file {file_path}: {e}\n{traceback.format_exc()}"
        return file_path, [], error_msg


def aggregate_counts(results, counts):
    """
    Aggregate sparse counts from multiple files into the main counts array.
    """
    for file_path, sparse_counts, error in results:
        if error:
            print(error)
        else:
            for instrument, pitch, velocity in sparse_counts:
                counts[instrument, pitch, velocity] += 1


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

    # Initialize counts array
    counts = np.zeros((128, 128, 128), dtype=np.int64)  # Using int64 to avoid overflow

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
                    aggregate_counts(results_batch, counts)
                    results_batch = []

            # Aggregate remaining results
            if results_batch:
                aggregate_counts(results_batch, counts)

    # Print summary statistics
    total_notes = np.sum(counts)
    non_zero_entries = np.count_nonzero(counts)

    print(f"\nProcessing complete!")
    print(f"Total notes counted: {total_notes:,}")
    print(f"Non-zero entries in counts array: {non_zero_entries:,}")
    print(f"Memory usage of counts array: {counts.nbytes / 1024 / 1024:.2f} MB")
    if errors_count > 0:
        print(f"Files with errors: {errors_count}")

    # Find most common combinations
    if total_notes > 0:
        top_k = 10
        flat_indices = np.argpartition(counts.ravel(), -top_k)[-top_k:]
        top_indices = np.unravel_index(flat_indices, counts.shape)

        print(f"\nTop {top_k} most common (instrument, pitch, velocity) combinations:")
        for i in range(top_k):
            inst = top_indices[0][i]
            pitch = top_indices[1][i]
            vel = top_indices[2][i]
            count = counts[inst, pitch, vel]
            if count > 0:
                print(f"  Instrument {inst}, Pitch {pitch}, Velocity {vel}: {count:,} occurrences")

    return counts


if __name__ == "__main__":
    # Use 16 workers as requested
    counts = main_parallel(n_workers=16, chunk_size=10)

    # Save the final counts as npz
    np.savez_compressed("./resources/note_counts.npz", counts)
    print("\nSaved counts to 'note_counts.npz'")
