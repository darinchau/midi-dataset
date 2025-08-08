# Extracts and analyzes MusicXML files to create a 3D piano roll representation.
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List
from ..util import get_text_or_raise, get_inv_gm_instruments_map, dynamics_to_velocity, get_time_signature_map


@dataclass(frozen=True)
class MusicXMLNote:
    instrument: int
    pitch: int  # MIDI pitch number (0-127)
    start: float  # seconds
    duration: float  # seconds
    start_ql: float  # quarter notes from start of the bar
    duration_ql: float  # quarter notes length
    index: int  # line of fifths index
    octave: int  # Octave number (0-9)
    velocity: int  # MIDI velocity (0-127)
    timesig: str
    barline: bool  # True if this note is not a note but a barline

    @classmethod
    def get_barline(cls):
        return cls(
            instrument=0,
            pitch=0,
            start=0.0,
            duration=0.0,
            start_ql=0.0,
            duration_ql=0.0,
            index=0,
            octave=0,
            velocity=0,
            timesig="",
            barline=True
        )

    def __post_init__(self):
        if not (0 <= self.instrument < 128):
            raise ValueError(f"Invalid instrument: {self.instrument}")
        if not (0 <= self.pitch < 128):
            raise ValueError(f"Invalid pitch: {self.pitch}")
        if self.start < 0 or self.duration < 0:
            raise ValueError("Start and duration must be non-negative")
        if not (0 <= self.velocity <= 127):
            raise ValueError(f"Invalid velocity: {self.velocity}")
        calculated_pitch = 12 * (self.octave + 1) + ([0, 7, 2, 9, 4, 11, 5][self.index % 7] + (self.index + 1) // 7)
        if not self.barline and calculated_pitch != self.pitch:
            raise ValueError(f"Pitch {self.pitch} does not match calculated pitch {calculated_pitch}")


def _step_alter_to_lof_index(step: str, alter: int) -> int:
    assert step in {"C", "D", "E", "F", "G", "A", "B"}, f"Invalid step: {step}"
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter


def _lof_index_to_step_alter(index: int) -> tuple[str, int]:
    idx = index % 7
    step = ("C", "G", "D", "A", "E", "B", "F")[idx]
    alter = (index + 1) // 7
    return step, alter


def parse_musicxml(xml_path: str, debug=True):
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

    # Collect all notes with their timing and instrument info
    notes_data: list[MusicXMLNote] = []
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

                        # Convert to seconds
                        duration_seconds = (duration_quarters * 60.0) / tempo_bpm
                        start_seconds = (current_time / divisions * 60.0) / tempo_bpm

                        # Add note data as tuple
                        # Calculate line of fifths index
                        lof_index = _step_alter_to_lof_index(step, alter)

                        note = MusicXMLNote(
                            instrument=current_instrument,
                            pitch=midi_note,
                            start=start_seconds,
                            duration=duration_seconds,
                            start_ql=onset_quarters_from_bar,
                            duration_ql=duration_quarters,
                            index=lof_index,
                            octave=octave,
                            velocity=velocity,
                            timesig=current_time_signature if current_time_signature else "UNK",
                            barline=False
                        )
                        notes_data.append(note)

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
            notes_data.append(MusicXMLNote.get_barline())

    if debug:
        print("\n" + "="*80)
        print(f"SUMMARY:")
        # Count actual notes (excluding barlines)
        actual_notes = [n for n in notes_data if not n.barline]
        print(f"Total notes parsed: {len(actual_notes)}")
        print(f"Total barlines: {sum(1 for n in notes_data if n.barline)}")
        if actual_notes:
            print(f"Time range: 0.000 - {max(n.start + n.duration for n in actual_notes):.3f} seconds")
            instruments_used = set(n.instrument for n in actual_notes)
            print(f"Instruments used: {sorted(instruments_used)}")
            pitch_range = [min(n.pitch for n in actual_notes), max(n.pitch for n in actual_notes)]
            print(f"Pitch range: MIDI {pitch_range[0]} - {pitch_range[1]}")
            velocity_range = [min(n.velocity for n in actual_notes), max(n.velocity for n in actual_notes)]
            print(f"Velocity range: {velocity_range[0]} - {velocity_range[1]}")
        print("="*80 + "\n")

    return notes_data


def is_valid_xml(xml_path: str) -> bool:
    """
    Check if the given file is a valid MusicXML file. Should be used as a filter for dataset iteration.

    Args:
        xml_path (str): Path to the MusicXML file.

    Returns:
        bool: True if the file is a valid MusicXML file, False otherwise.
    """
    try:
        notes = parse_musicxml(xml_path, debug=False)
        return len(notes) > 0
    except ET.ParseError:
        return False


def fix_time_signature(notes_data: List[MusicXMLNote]) -> List[MusicXMLNote]:
    """
    Fix time signature for each note in the notes_data list.
    If a note has no time signature, set it to "UNK".
    Modifies the notes_data in place and returns the original list with updated note objects

    Args:
        notes_data (List[MusicXMLNote]): List of MusicXMLNote objects.

    Returns:
        List[MusicXMLNote]: Updated list with fixed time signatures.
    """
    mapping = get_time_signature_map()

    fixes = {
        "2/2": "4/4",
        "4/2": "4/4",
        "3/2": "6/4",
        "8/4": "4/4",
        "8/8": "4/4",
        "4/8": "2/4",
        "16/16": "4/4",
    }

    for note in notes_data:
        new_timesig = note.timesig
        if new_timesig not in mapping:
            if new_timesig in fixes:
                new_timesig = fixes[new_timesig]
            else:
                new_timesig = "UNK"
            assert new_timesig in mapping, f"Invalid time signature: {new_timesig}"
        # This in place modification is explitly stated in the docstring
        object.__setattr__(note, 'timesig', new_timesig)

    return notes_data
