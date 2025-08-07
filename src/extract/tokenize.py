# Extracts sparse compound word tokens from MusicXML files.

import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from dataclasses import dataclass
from ..util import get_inv_time_signature_map,  get_text_or_raise, dynamics_to_velocity

BARLINE = object()


def check_element(element):
    """Check if the element is a valid MusicXML note or barline. Raise ValueError if not."""
    if element is BARLINE:
        return True
    if isinstance(element, tuple) and len(element) == 6:
        instrument, pitch, velocity, onset, duration, time_signature = element
        if not (0 <= instrument < 128):
            raise ValueError(f"Invalid instrument: {instrument}")
        if not (0 <= pitch < 128):
            raise ValueError(f"Invalid pitch: {pitch}")
        if not (0 <= velocity <= 127):
            raise ValueError(f"Invalid velocity: {velocity}")
        if not (onset >= 0):
            raise ValueError(f"Invalid onset: {onset}")
        if not (duration > 0):
            raise ValueError(f"Invalid duration: {duration}")
        return True
    raise ValueError("Element must be a tuple of 6 components or BARLINE")


def element_to_nparray(element):
    """
    Convert a music xml token or a barline to a 1D vector
    """
    v = np.zeros((128 + 128 + 1 + 1 + 1 + 16 + 1,))
    if element is BARLINE:
        v[-1] = 1
        return v
    check_element(element)
    instrument, pitch, velocity, onset, duration, time_signature = element
    time_sig_idx = get_inv_time_signature_map().get(time_signature, 0)
    v[instrument] = 1
    v[128 + pitch] = 1
    v[128 + 128] = velocity / 127.0  # Normalize velocity to [0, 1]
    v[128 + 128 + 1] = onset
    v[128 + 128 + 1 + 1] = duration
    v[128 + 128 + 1 + 1 + 1 + time_sig_idx] = 1
    v[-1] = 0
    return v


def musicxml_to_tokens(xml_path: str, debug=True):
    """
    Tokenize a list of MusicXMLNote objects into a one-hot encoded 3D piano roll.

    Args:
        xml_path (str): Path to the MusicXML file.
        debug (bool): If True, prints debug information.

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
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Default tempo
    tempo_bpm = 120
    divisions = 1
    current_dynamics = 80  # Default velocity (mf)
    current_time_signature = None  # Track current time signature

    # Find tempo and divisions
    for element in root.iter():
        if element.tag == 'divisions':
            divisions = int(get_text_or_raise(element))
            if debug:
                print(f"Divisions per quarter note: {divisions}")
        elif element.tag == 'sound' and 'tempo' in element.attrib:
            tempo_bpm = float(element.attrib['tempo'])
            if debug:
                print(f"Tempo: {tempo_bpm} BPM")

    notes_data = []
    current_time = 0
    current_instrument = 0

    if debug:
        print("\n" + "="*80)
        print("PARSING NOTES:")
        print("="*80)

    for part in root.findall('.//part'):
        part_id = part.get('id')

        if debug:
            print(f"\nProcessing Part: {part_id}")

        # Try to find instrument for this part
        for score_part in root.findall('.//score-part[@id="{}"]'.format(part_id)):
            midi_instr = score_part.find('.//midi-instrument/midi-program')
            if midi_instr is not None:
                current_instrument = int(get_text_or_raise(midi_instr)) - 1  # MIDI programs are 1-indexed in MusicXML
                if debug:
                    # Also try to get instrument name
                    part_name = score_part.find('.//part-name')
                    instr_name = get_text_or_raise(part_name) if part_name is not None else "Unknown"
                    print(f"  Instrument: {instr_name} (MIDI Program: {current_instrument})")

        current_time = 0
        measure_number = 0

        for measure in part.findall('.//measure'):
            measure_number += 1
            measure_start_time = current_time  # Track start of measure

            if debug and measure.get('number'):
                print(f"\n  Measure {measure.get('number')}:")

            # Check for time signature changes
            for attributes in measure.findall('.//attributes'):
                time_elem = attributes.find('time')
                if time_elem is not None:
                    beats_elem = time_elem.find('beats')
                    beat_type_elem = time_elem.find('beat-type')
                    beats = get_text_or_raise(beats_elem)
                    beat_type = get_text_or_raise(beat_type_elem)
                    current_time_signature = f"{beats}/{beat_type}"
                    if debug:
                        print(f"    Time signature: {current_time_signature}")

            # Check for dynamics changes in this measure
            for direction in measure.findall('.//direction'):
                dynamics = direction.find('.//dynamics')
                if dynamics is not None:
                    for dyn in dynamics:
                        current_dynamics = dynamics_to_velocity(dyn.tag)
                        if debug:
                            print(f"    Dynamics change: {dyn.tag} (velocity: {current_dynamics})")

            for element in measure:
                if element.tag == 'note':
                    pitch_elem = element.find('pitch')
                    if pitch_elem is not None:
                        # Get pitch components
                        step_elem = pitch_elem.find('step')
                        step = get_text_or_raise(step_elem)
                        assert step in {'C', 'D', 'E', 'F', 'G', 'A', 'B'}, f"Invalid step: {step}"

                        octave_elem = pitch_elem.find('octave')
                        octave = int(get_text_or_raise(octave_elem))
                        alter_elem = pitch_elem.find('alter')
                        alter = int(get_text_or_raise(alter_elem)) if alter_elem is not None else 0

                        # Convert to MIDI note number
                        note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                        midi_note = 12 * (octave + 1) + note_map[step] + alter

                        # Get duration
                        duration_elem = element.find('duration')
                        duration_ticks = int(get_text_or_raise(duration_elem))

                        # Get velocity - first check for explicit velocity element
                        velocity = current_dynamics
                        velocity_elem = element.find('velocity')
                        if velocity_elem is not None:
                            velocity = int(get_text_or_raise(velocity_elem))
                        else:
                            # Check for dynamics in notations
                            notations = element.find('notations')
                            if notations is not None:
                                dynamics = notations.find('.//dynamics')
                                if dynamics is not None:
                                    for dyn in dynamics:
                                        velocity = dynamics_to_velocity(dyn.tag)
                                        current_dynamics = velocity  # Update current dynamics

                        # Convert to quarter notes
                        duration_quarters = duration_ticks / divisions
                        onset_quarters_from_bar = (current_time - measure_start_time) / divisions

                        # Add note data as tuple
                        note_data = (
                            current_instrument,       # Instrument (0-127)
                            midi_note,                # Pitch (MIDI number)
                            velocity,                 # Velocity (0-127)
                            onset_quarters_from_bar,  # Onset (quarter notes from start of bar)
                            duration_quarters,        # Duration (quarter notes)
                            current_time_signature    # Time signature string
                        )
                        notes_data.append(note_data)

                    # Update time if not a chord
                    if element.find('chord') is None and element.find('duration') is not None:
                        duration_elem = element.find('duration')
                        duration = int(get_text_or_raise(duration_elem))
                        current_time += duration

                elif element.tag == 'backup':
                    duration_elem = element.find('duration')
                    duration = int(get_text_or_raise(duration_elem))
                    current_time -= duration
                    if debug:
                        print(f"    Backup: {duration} ticks")

                elif element.tag == 'forward':
                    duration_elem = element.find('duration')
                    duration = int(get_text_or_raise(duration_elem))
                    current_time += duration
                    if debug:
                        print(f"    Forward: {duration} ticks")

                elif element.tag == 'rest':
                    duration_elem = element.find('duration')
                    duration = int(get_text_or_raise(duration_elem))
                    current_time += duration
                    if debug:
                        duration_seconds = (duration / divisions * 60.0) / tempo_bpm
                        print(f"    Rest: {duration_seconds:.3f} seconds")

            # Insert BARLINE after each measure
            notes_data.append(BARLINE)

    if debug:
        print("\n" + "="*80)
        print(f"SUMMARY:")
        # Count actual notes (excluding barlines)
        actual_notes = [n for n in notes_data if n != BARLINE]
        print(f"Total notes parsed: {len(actual_notes)}")
        print(f"Total barlines: {notes_data.count(BARLINE)}")
        if actual_notes:
            instruments_used = set(n[0] for n in actual_notes)
            print(f"Instruments used: {sorted(instruments_used)}")
            pitch_range = [min(n[1] for n in actual_notes), max(n[1] for n in actual_notes)]
            print(f"Pitch range: MIDI {pitch_range[0]} - {pitch_range[1]}")
            velocity_range = [min(n[2] for n in actual_notes), max(n[2] for n in actual_notes)]
            print(f"Velocity range: {velocity_range[0]} - {velocity_range[1]}")
            time_signatures_used = set(n[5] for n in actual_notes if n[5] is not None)
            print(f"Time signatures used: {sorted(time_signatures_used)}")
        print("="*80 + "\n")

    # Convert notes_data to numpy array
    notes_data = [element_to_nparray(n) for n in notes_data]
    notes_data = np.stack(notes_data)
    return notes_data
