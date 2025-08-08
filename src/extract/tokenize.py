# Extracts sparse compound word tokens from MusicXML files.

import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from dataclasses import dataclass
from .analyze import MusicXMLNote
from ..util import get_time_signature_map
from ..util import get_inv_time_signature_map,  get_text_or_raise, dynamics_to_velocity


def element_to_nparray(element: MusicXMLNote):
    """
    Convert a music xml token or a barline to a 1D vector
    """
    v = np.zeros((128 + 128 + 1 + 1 + 1 + 16 + 1,))
    if element.barline:
        v[-1] = 1
        return v
    time_sig_idx = get_inv_time_signature_map().get(element.timesig, 0)
    v[element.instrument] = 1
    v[128 + element.pitch] = 1
    # This should be the velocity field but we found that the parsed xml only uses 80 for the velocity
    v[128 + 128] = 1
    v[128 + 128 + 1] = element.start_ql
    v[128 + 128 + 1 + 1] = element.duration_ql
    v[128 + 128 + 1 + 1 + 1 + time_sig_idx] = 1
    v[-1] = 0
    return v


def musicxml_to_tokens(notes_data: list[MusicXMLNote]):
    """
    Tokenize a list of MusicXMLNote objects into a one-hot encoded 3D piano roll.

    Args:
        xml_path (str): Path to the MusicXML file.

    Returns:
        np.ndarray: A (T, d) 2D array where T is the number of time steps and d is the number of dimensions
        The dimesions are:
        - Instrument (128x, one-hot encoded)
        - Pitch (128x, one-hot encoded)
        - Velocity (0-1)
        - Onset (# Quarter notes from start of the bar)
        - Duration (# Quarter notes)
        - Current time signature (16x, one-hot encoded)
        - Bar line? (0 or 1)
    """
    x = [element_to_nparray(n) for n in notes_data]
    x = np.stack(x)
    return x


def notes_data_to_piano_roll(notes_data: list[MusicXMLNote], steps_per_second=24):
    # Get time signature mapping
    time_sig_map = get_time_signature_map()
    # Create reverse mapping from time signature string to index
    time_sig_to_index = {v: k for k, v in time_sig_map.items() if k < 12}

    # Calculate total time steps
    # Filter out barlines when calculating max time
    actual_notes = [n for n in notes_data if not n.barline]
    if not actual_notes:
        return np.zeros((128, 128, 1), dtype=np.float32), np.zeros((16, 1), dtype=np.float32)

    max_time = max(n.start + n.duration for n in actual_notes)
    total_steps = int(np.ceil(max_time * steps_per_second))

    # Initialize arrays
    piano_roll_3d = np.zeros((128, 128, total_steps), dtype=np.float32)
    metadata_array = np.zeros((16, total_steps), dtype=np.float32)

    # Track current time signature and barline positions
    current_time_sig_index = 0  # Default to "UNK"
    barline_times = []

    # First pass: collect barline times and process notes
    for i, note in enumerate(notes_data):
        if note.barline:
            # Find the time of the next non-barline note
            for j in range(i + 1, len(notes_data)):
                if not notes_data[j].barline:
                    barline_times.append(notes_data[j].start)
                    break
        else:
            # Process regular notes for piano roll
            instrument = max(0, min(127, note.instrument))
            pitch = max(0, min(127, note.pitch))
            start_step = int(note.start * steps_per_second)
            end_step = int((note.start + note.duration) * steps_per_second)

            # Convert MIDI velocity (0-127) to 0-1 range
            velocity_normalized = note.velocity / 127.0

            if start_step < total_steps:
                end_step = min(end_step, total_steps)
                piano_roll_3d[instrument, pitch, start_step:end_step] = velocity_normalized

    # Second pass: fill metadata array
    # Sort notes by start time (excluding barlines)
    sorted_notes = sorted([n for n in notes_data if not n.barline], key=lambda x: x.start)
    note_idx = 0

    for t in range(total_steps):
        current_time = t / steps_per_second

        # Update current time signature based on notes at or before this time
        while note_idx < len(sorted_notes) and sorted_notes[note_idx].start <= current_time:
            note = sorted_notes[note_idx]
            if note.timesig in time_sig_to_index:
                current_time_sig_index = time_sig_to_index[note.timesig]
            note_idx += 1

        # Set time signature one-hot encoding (positions 0-11)
        if 0 <= current_time_sig_index < 12:
            metadata_array[current_time_sig_index, t] = 1.0

        # Check if this is the first position after a barline (position 15)
        for barline_time in barline_times:
            barline_step = int(barline_time * steps_per_second)
            if t == barline_step:
                metadata_array[15, t] = 1.0
                break

    return piano_roll_3d, metadata_array
