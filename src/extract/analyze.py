# Extracts and analyzes MusicXML files to create a 3D piano roll representation.
import numpy as np
import xml.etree.ElementTree as ET
import logging
from dataclasses import dataclass
from typing import List
from functools import lru_cache
from .utils import get_text_or_raise, get_inv_gm_instruments_map, dynamics_to_velocity, get_time_signature_map
from ..utils import get_gm_instruments_map

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MusicXMLNote:
    instrument: int  # Program number of the instrument assuming GM (0-127)
    start: float  # seconds from beginning of the piece
    duration: float  # seconds
    start_ql: float  # quarter notes from start of the bar
    duration_ql: float  # quarter notes length
    index: int  # line of fifths index
    octave: int  # Octave number (0-9)
    velocity: int  # MIDI velocity (0-127)
    timesig: str | None  # Time signature as a string, e.g., "4/4", or None if unknown
    barline: bool  # True if this note is not a note but a barline

    @classmethod
    def get_barline(cls, current_time: float, current_timesig: str | None) -> 'MusicXMLNote':
        return cls(
            instrument=-1,
            start=current_time,
            duration=0.0,
            start_ql=0.0,
            duration_ql=0.0,
            index=0,
            octave=0,
            velocity=0,
            timesig=current_timesig,
            barline=True
        )

    def __post_init__(self):
        if self.barline:
            if not self.duration == 0.0 and self.duration_ql == 0.0:
                raise ValueError("Barline note should have zero duration and quarter length")
            return  # No further validation needed for barlines

        if not (0 <= self.instrument < 128):
            raise ValueError(f"Invalid instrument: {self.instrument}")
        if not (0 <= self.pitch < 128):
            raise ValueError(f"Invalid pitch: {self.pitch}")
        if self.start < 0 or self.duration < 0:
            raise ValueError("Start and duration must be non-negative")
        if not (0 <= self.velocity <= 127):
            raise ValueError(f"Invalid velocity: {self.velocity}")
        # Check time signature validity
        if self.timesig is not None:
            num, denom = self.timesig.split('/')
            if not (num.isdigit() and denom.isdigit() and int(denom) in {1, 2, 4, 8, 16, 32, 64}):
                raise ValueError(f"Invalid time signature: {self.timesig}")
            max_start_ql = (int(num) * 4) / int(denom)
            if not (0 <= self.start_ql < max_start_ql):
                raise ValueError(f"start_ql {self.start_ql} out of range for time signature {self.timesig}")

    @property
    def pitch(self):
        """Return the MIDI pitch number. A0 is 21 and C8 is 108. Barlines are 0."""
        if self.barline:
            return -1
        return index_octave_to_pitch(self.index, self.octave)

    @property
    def name(self):
        """Return the note name in scientific pitch notation, e.g., C4, A#3."""
        if self.barline:
            return "BARLINE"
        step, alter = _lof_index_to_step_alter(self.index)
        pitch_class = step
        if alter < 0:
            pitch_class += 'b' * (-alter)
        elif alter > 0:
            pitch_class += '#' * alter  # Don't bother with double sharp symbols

        instrument_name = get_gm_instruments_map().get(self.instrument, "Unknown")
        return f"{pitch_class}{self.octave} ({instrument_name}={self.instrument})"

    @staticmethod
    def pitch_to_index(pitch: int) -> int:
        """Convert MIDI pitch number to line of fifths index."""
        assert 0 <= pitch < 128, f"Invalid pitch: {pitch}"
        return [0, 7, 5, 9, 4, -1, 6, 1, 8, 3, 10, 5][pitch % 12]

    @staticmethod
    def pitch_to_octave(pitch: int) -> int:
        """Convert MIDI pitch number to octave number."""
        assert 0 <= pitch < 128, f"Invalid pitch: {pitch}"
        return (pitch // 12) - 1


@lru_cache(maxsize=None)
def index_octave_to_pitch(index: int, octave: int) -> int:
    return 12 * (octave + 1) + ([0, 7, 2, 9, 4, 11, 5][index % 7] + (index + 1) // 7)


def _step_alter_to_lof_index(step: str, alter: int) -> int:
    assert step in {"C", "D", "E", "F", "G", "A", "B"}, f"Invalid step: {step}"
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter


def _lof_index_to_step_alter(index: int) -> tuple[str, int]:
    idx = index % 7
    step = ("C", "G", "D", "A", "E", "B", "F")[idx]
    alter = (index + 1) // 7
    return step, alter


def parse_musicxml(xml_path: str):
    """
    Tokenize a list of MusicXMLNote objects into a one-hot encoded 3D piano roll.

    Args:
        xml_path (str): Path to the MusicXML file.

    Returns:
        List[MusicXMLNote]: List of MusicXMLNote objects parsed from the file.
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
            logger.debug(f"Divisions per quarter note: {divisions}")
        elif element.tag == 'sound' and 'tempo' in element.attrib:
            tempo_bpm = float(element.attrib['tempo'])
            logger.debug(f"Tempo: {tempo_bpm} BPM")

    # Collect all notes with their timing and instrument info
    notes_data: list[MusicXMLNote] = []
    current_time = 0
    current_instrument = 0

    logger.info("Starting to parse notes")

    for part in root.findall('.//part'):
        part_id = part.get('id')
        logger.debug(f"Processing Part: {part_id}")

        # Try to find instrument for this part
        for score_part in root.findall('.//score-part[@id="{}"]'.format(part_id)):
            midi_instr = score_part.find('.//midi-instrument/midi-program')
            if midi_instr is not None:
                current_instrument = int(get_text_or_raise(midi_instr)) - 1  # MIDI programs are 1-indexed in MusicXML
                # Also try to get instrument name
                part_name = score_part.find('.//part-name')
                instr_name = get_text_or_raise(part_name) if part_name is not None else "Unknown"
                logger.debug(f"  Instrument: {instr_name} (MIDI Program: {current_instrument})")

        current_time = 0
        measure_number = 0

        # Dictionary to track ongoing tied notes: key is (pitch, octave, alter)
        tied_notes = {}

        for measure in part.findall('.//measure'):
            measure_number += 1
            measure_start_time = current_time  # Track start of measure

            if measure.get('number'):
                logger.debug(f"  Measure {measure.get('number')}:")

            # Check for time signature changes
            for attributes in measure.findall('.//attributes'):
                time_elem = attributes.find('time')
                if time_elem is not None:
                    beats_elem = time_elem.find('beats')
                    beat_type_elem = time_elem.find('beat-type')
                    beats = get_text_or_raise(beats_elem)
                    beat_type = get_text_or_raise(beat_type_elem)
                    current_time_signature = f"{beats}/{beat_type}"
                    logger.debug(f"    Time signature: {current_time_signature}")

            # Check for dynamics changes in this measure
            for direction in measure.findall('.//direction'):
                dynamics = direction.find('.//dynamics')
                if dynamics is not None:
                    for dyn in dynamics:
                        current_dynamics = dynamics_to_velocity(dyn.tag)
                        logger.debug(f"    Dynamics change: {dyn.tag} (velocity: {current_dynamics})")

            for element in measure:
                if element.tag == 'note':
                    # Skip grace notes (for now?)
                    is_grace = element.find('grace') is not None
                    if is_grace:
                        continue

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

                        # Calculate line of fifths index
                        lof_index = _step_alter_to_lof_index(step, alter)

                        # Create a key for tracking tied notes
                        note_key = (step, octave, alter)

                        # Check for tie elements
                        tie_start = False
                        tie_stop = False
                        for tie in element.findall('.//tie'):
                            tie_type = tie.get('type')
                            if tie_type == 'start':
                                tie_start = True
                            elif tie_type == 'stop':
                                tie_stop = True

                        # Handle tied notes
                        if tie_stop and note_key in tied_notes:
                            # This note ends a tie - add its duration to the ongoing tied note
                            tied_note_data = tied_notes[note_key]
                            tied_note_data['duration_ticks'] += duration_ticks
                            tied_note_data['duration_quarters'] += duration_quarters
                            tied_note_data['duration_seconds'] += duration_seconds

                            if not tie_start:
                                # This is the end of the tie - create the combined note
                                note = MusicXMLNote(
                                    instrument=current_instrument,
                                    start=tied_note_data['start_seconds'],
                                    duration=tied_note_data['duration_seconds'],
                                    start_ql=tied_note_data['onset_quarters_from_bar'],
                                    duration_ql=tied_note_data['duration_quarters'],
                                    index=lof_index,
                                    octave=octave,
                                    velocity=tied_note_data['velocity'],
                                    timesig=tied_note_data['timesig'],
                                    barline=False
                                )
                                notes_data.append(note)
                                # Remove from tied notes tracking
                                del tied_notes[note_key]
                        elif tie_start and not tie_stop:
                            # This note starts a new tie - store it for later
                            tied_notes[note_key] = {
                                'start_seconds': start_seconds,
                                'duration_seconds': duration_seconds,
                                'onset_quarters_from_bar': onset_quarters_from_bar,
                                'duration_quarters': duration_quarters,
                                'duration_ticks': duration_ticks,
                                'velocity': velocity,
                                'timesig': current_time_signature if current_time_signature else None
                            }
                        elif not tie_start and not tie_stop:
                            # Regular note (not tied)
                            note = MusicXMLNote(
                                instrument=current_instrument,
                                start=start_seconds,
                                duration=duration_seconds,
                                start_ql=onset_quarters_from_bar,
                                duration_ql=duration_quarters,
                                index=lof_index,
                                octave=octave,
                                velocity=velocity,
                                timesig=current_time_signature if current_time_signature else None,
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
                    logger.debug(f"    Backup: {duration} ticks")

                elif element.tag == 'forward':
                    duration_elem = element.find('duration')
                    duration = int(get_text_or_raise(duration_elem))
                    current_time += duration
                    logger.debug(f"    Forward: {duration} ticks")

                elif element.tag == 'rest':
                    duration_elem = element.find('duration')
                    duration = int(get_text_or_raise(duration_elem))
                    current_time += duration
                    duration_seconds = (duration / divisions * 60.0) / tempo_bpm
                    logger.debug(f"    Rest: {duration_seconds:.3f} seconds")

            # Insert BARLINE after each measure
            start_seconds = (current_time / divisions * 60.0) / tempo_bpm
            notes_data.append(MusicXMLNote.get_barline(start_seconds, current_time_signature if current_time_signature else None))

        # Check for any unresolved tied notes (shouldn't happen in valid MusicXML)
        if tied_notes:
            logger.warning(f"Found {len(tied_notes)} unresolved tied notes in part {part_id}")
            # Optionally create notes for them anyway
            for note_key, tied_note_data in tied_notes.items():
                step, octave, alter = note_key
                lof_index = _step_alter_to_lof_index(step, alter)
                note = MusicXMLNote(
                    instrument=current_instrument,
                    start=tied_note_data['start_seconds'],
                    duration=tied_note_data['duration_seconds'],
                    start_ql=tied_note_data['onset_quarters_from_bar'],
                    duration_ql=tied_note_data['duration_quarters'],
                    index=lof_index,
                    octave=octave,
                    velocity=tied_note_data['velocity'],
                    timesig=tied_note_data['timesig'],
                    barline=False
                )
                notes_data.append(note)

    # Count actual notes (excluding barlines)
    actual_notes = [n for n in notes_data if not n.barline]
    logger.info(f"Total notes parsed: {len(actual_notes)}")
    logger.info(f"Total barlines: {sum(1 for n in notes_data if n.barline)}")
    if actual_notes:
        logger.info(f"Time range: 0.000 - {max(n.start + n.duration for n in actual_notes):.3f} seconds")
        instruments_used = set(n.instrument for n in actual_notes)
        logger.info(f"Instruments used: {sorted(instruments_used)}")
        pitch_range = [min(n.pitch for n in actual_notes), max(n.pitch for n in actual_notes)]
        logger.info(f"Pitch range: MIDI {pitch_range[0]} - {pitch_range[1]}")
        velocity_range = [min(n.velocity for n in actual_notes), max(n.velocity for n in actual_notes)]
        logger.info(f"Velocity range: {velocity_range[0]} - {velocity_range[1]}")

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
        notes = parse_musicxml(xml_path)
        return len(notes) > 0
    except ET.ParseError:
        return False


def fix_time_signature(notes_data: List[MusicXMLNote]) -> List[MusicXMLNote]:
    """
    Fix time signature for each note in the notes_data list.
    If a note has no time signature, set it to None.
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
        if not note.barline and new_timesig not in mapping.values():
            logger.debug(f"Fixing time signature for note: {new_timesig}")
            if new_timesig in fixes:
                new_timesig = fixes[new_timesig]
            else:
                new_timesig = None  # Unknown time signature
            assert new_timesig in mapping.values(), f"Invalid time signature: {new_timesig}"
            # This in place modification is explitly stated in the docstring
            object.__setattr__(note, 'timesig', new_timesig)

    return notes_data


def musicxml_to_notes(xml_path: str, no_barline: bool = False) -> List[MusicXMLNote]:
    """
    Parse a MusicXML file and return a list of MusicXMLNote objects.

    Args:
        xml_path (str): Path to the MusicXML file.
        no_barline (bool): If True, remove barline notes from the output list.

    Returns:
        List[MusicXMLNote]: List of MusicXMLNote objects parsed from the file.
    """
    notes = parse_musicxml(xml_path)
    notes = fix_time_signature(notes)
    notes = sorted(notes, key=lambda n: (n.start, not n.barline, n.instrument, n.pitch))
    # Remove duplicate barlines
    current_barline = None
    new_notes = []
    for note in notes:
        if note.barline:
            if current_barline is None or current_barline.start != note.start:
                if not no_barline:
                    new_notes.append(note)
                current_barline = note
        else:
            new_notes.append(note)
    return new_notes
