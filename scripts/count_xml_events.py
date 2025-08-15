# Count notes in MusicXML files
# python -m scripts.count_xml_events
import os
import csv
from functools import partial
from multiprocessing import Pool, cpu_count
from src.util import iterate_dataset
from src.constants import *
from src.extract.analyze import parse_musicxml as parse_musicxml
import numpy as np
import traceback
from tqdm.auto import tqdm


def process_single_file(file_path):
    """
    Process a single MusicXML file and return the count of notes.
    Returns the file basename and note count.
    """
    try:
        notes_data = parse_musicxml(file_path)
        note_count = 0
        for note in notes_data:
            if not note.barline:
                continue
            note_count += 1
        basename = os.path.basename(file_path)
        return basename, note_count, None
    except Exception as e:
        error_msg = f"Error processing file {file_path}: {e}\n{traceback.format_exc()}"
        basename = os.path.basename(file_path)
        return basename, 0, error_msg


def main_parallel(n_workers=16, chunk_size=10, output_csv="note_counts.csv"):
    """
    Parallel version using multiprocessing.

    Args:
        n_workers: Number of worker processes. Default = 16
        chunk_size: Number of files to process in each batch
        output_csv: Path to output CSV file
    """
    print(f"Using {n_workers} worker processes")

    # Get all file paths
    file_paths = list(tqdm(iterate_dataset(XML_ROOT), desc="Collecting file paths", unit="file"))
    total_files = len(file_paths)
    print(f"Found {total_files} files to process")

    # Store results as list of tuples (basename, count)
    file_counts = []

    # Process files in parallel with progress bar
    with Pool(processes=n_workers) as pool:
        process_func = process_single_file

        # Progress bar with more detailed information
        with tqdm(total=total_files,
                  desc="Processing files",
                  unit="files",
                  smoothing=0.1) as pbar:
            errors_count = 0
            total_notes = 0

            for result in pool.imap(process_func, file_paths, chunksize=chunk_size):
                basename, note_count, error = result

                if error is not None:
                    errors_count += 1
                    print(error)
                else:
                    file_counts.append((basename, note_count))
                    total_notes += note_count

                # Update progress bar with additional stats
                pbar.update(1)
                pbar.set_postfix({
                    'errors': errors_count,
                    'total_notes': total_notes
                })

    # Sort by basename for consistent output
    file_counts.sort(key=lambda x: x[0])

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['filename', 'note_count'])
        # Write data
        writer.writerows(file_counts)

    # Print summary statistics
    print(f"\nProcessing complete!")
    print(f"Total files processed successfully: {len(file_counts):,}")
    print(f"Total notes counted across all files: {total_notes:,}")
    if errors_count > 0:
        print(f"Files with errors: {errors_count}")

    # Calculate and display some statistics
    if file_counts:
        counts_only = [count for _, count in file_counts]
        avg_notes = np.mean(counts_only)
        median_notes = np.median(counts_only)
        max_notes = max(counts_only)
        min_notes = min(counts_only)

        print(f"\nStatistics:")
        print(f"  Average notes per file: {avg_notes:.2f}")
        print(f"  Median notes per file: {median_notes:.0f}")
        print(f"  Maximum notes in a file: {max_notes}")
        print(f"  Minimum notes in a file: {min_notes}")

        # Show top 10 files with most notes
        file_counts_sorted = sorted(file_counts, key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 files with most notes:")
        for i, (filename, count) in enumerate(file_counts_sorted[:10], 1):
            print(f"  {i}. {filename}: {count:,} notes")

    print(f"\nResults saved to '{output_csv}'")

    return file_counts


if __name__ == "__main__":
    # Use 16 workers as requested
    file_counts = main_parallel(n_workers=16, chunk_size=10, output_csv="note_counts_per_file.csv")
