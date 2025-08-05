import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List
import music21


@dataclass(frozen=True)
class MusicXMLNote:
    instrument: int
    pitch: int
    start: float  # seconds
    duration: float  # seconds
    index: int  # line of fifths index
    octave: int
    velocity: int  # MIDI velocity (0-127)


def _step_alter_to_lof_index(step: str, alter: int) -> int:
    assert step in {"C", "D", "E", "F", "G", "A", "B"}, f"Invalid step: {step}"
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter


def _lof_index_to_step_alter(index: int) -> tuple[str, int]:
    idx = index % 7
    step = ("C", "G", "D", "A", "E", "B", "F")[idx]
    alter = (index + 1) // 7
    return step, alter


def dynamics_to_velocity(dynamics_tag: str) -> int:
    """
    Convert musical dynamics notation to MIDI velocity.
    """
    dynamics_map = {
        'pppp': 8, 'ppp': 20, 'pp': 33, 'p': 45,
        'mp': 60, 'mf': 75, 'f': 88, 'ff': 103,
        'fff': 117, 'ffff': 127,
        'sf': 100, 'sfz': 100, 'sffz': 115,
        'fp': 88, 'rfz': 100, 'rf': 100
    }
    return dynamics_map.get(dynamics_tag, 80)  # Default to mezzo-forte


def parse_musicxml(xml_path: str, debug=True) -> List[MusicXMLNote]:
    """
    Manually parse MusicXML and create 3D piano roll with debug output.
    Raises an error if the file is not a valid MusicXML.

    Args:
        xml_path (str): Path to the MusicXML file.
        debug (bool): If True, prints debug information.

    Returns:
        List[MusicXMLNote]: A list of MusicXMLNote objects containing parsed note data.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Default tempo
    tempo_bpm = 120
    divisions = 1
    current_dynamics = 80  # Default velocity (mf)

    # Find tempo and divisions
    for element in root.iter():
        if element.tag == 'divisions':
            divisions = int(element.text)
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
                current_instrument = int(midi_instr.text) - 1  # MIDI programs are 1-indexed in MusicXML
                if debug:
                    # Also try to get instrument name
                    part_name = score_part.find('.//part-name')
                    instr_name = part_name.text if part_name is not None else "Unknown"
                    print(f"  Instrument: {instr_name} (MIDI Program: {current_instrument})")

        current_time = 0
        measure_number = 0

        for measure in part.findall('.//measure'):
            measure_number += 1
            if debug and measure.get('number'):
                print(f"\n  Measure {measure.get('number')}:")

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
                        step = pitch_elem.find('step').text
                        assert step in {'C', 'D', 'E', 'F', 'G', 'A', 'B'}, f"Invalid step: {step}"

                        octave = int(pitch_elem.find('octave').text)
                        alter_elem = pitch_elem.find('alter')
                        alter = int(alter_elem.text) if alter_elem is not None else 0

                        # Convert to MIDI note number
                        note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                        midi_note = 12 * (octave + 1) + note_map[step] + alter

                        # Get duration
                        duration_ticks = int(element.find('duration').text)

                        # Get velocity - first check for explicit velocity element
                        velocity = current_dynamics
                        velocity_elem = element.find('velocity')
                        if velocity_elem is not None:
                            velocity = int(velocity_elem.text)
                        else:
                            # Check for dynamics in notations
                            notations = element.find('notations')
                            if notations is not None:
                                dynamics = notations.find('.//dynamics')
                                if dynamics is not None:
                                    for dyn in dynamics:
                                        velocity = dynamics_to_velocity(dyn.tag)
                                        current_dynamics = velocity  # Update current dynamics

                        # Convert to seconds
                        quarter_length = duration_ticks / divisions
                        duration_seconds = (quarter_length * 60.0) / tempo_bpm
                        start_seconds = (current_time / divisions * 60.0) / tempo_bpm

                        # Check if it's part of a chord
                        is_chord = element.find('chord') is not None

                        if debug:
                            print(f"    Note: {step}{alter if alter != 0 else ''}{octave}")
                            print(f"      - Pitch: {step}")
                            print(f"      - Alter: {alter} {'(sharp)' if alter > 0 else '(flat)' if alter < 0 else '(natural)'}")
                            print(f"      - Octave: {octave}")
                            print(f"      - MIDI Note Number: {midi_note}")
                            print(f"      - Instrument: {current_instrument}")
                            print(f"      - Velocity: {velocity}")
                            print(f"      - Onset(ticks): {current_time}")
                            print(f"      - Onset: {start_seconds:.3f} seconds")
                            print(f"      - Duration: {duration_seconds:.3f} seconds")
                            print(f"      - Duration (ticks): {duration_ticks}")
                            print(f"      - Is Chord: {is_chord}")

                            # Additional info if available
                            voice = element.find('voice')
                            if voice is not None:
                                print(f"      - Voice: {voice.text}")

                        notes_data.append(MusicXMLNote(
                            instrument=current_instrument,
                            pitch=midi_note,
                            start=start_seconds,
                            duration=duration_seconds,
                            index=_step_alter_to_lof_index(step, alter),
                            octave=octave,
                            velocity=velocity,
                        ))

                    # Update time if not a chord
                    if element.find('chord') is None and element.find('duration') is not None:
                        duration = int(element.find('duration').text)
                        current_time += duration

                elif element.tag == 'backup':
                    duration = int(element.find('duration').text)
                    current_time -= duration
                    if debug:
                        print(f"    Backup: {duration} ticks")

                elif element.tag == 'forward':
                    duration = int(element.find('duration').text)
                    current_time += duration
                    if debug:
                        print(f"    Forward: {duration} ticks")

                elif element.tag == 'rest':
                    duration = int(element.find('duration').text)
                    current_time += duration
                    if debug:
                        duration_seconds = (duration / divisions * 60.0) / tempo_bpm
                        print(f"    Rest: {duration_seconds:.3f} seconds")

    if debug:
        print("\n" + "="*80)
        print(f"SUMMARY:")
        print(f"Total notes parsed: {len(notes_data)}")
        if notes_data:
            print(f"Time range: 0.000 - {max(n.start + n.duration for n in notes_data):.3f} seconds")
            instruments_used = set(n.instrument for n in notes_data)
            print(f"Instruments used: {sorted(instruments_used)}")
            pitch_range = [min(n.pitch for n in notes_data), max(n.pitch for n in notes_data)]
            print(f"Pitch range: MIDI {pitch_range[0]} - {pitch_range[1]}")
            velocity_range = [min(n.velocity for n in notes_data), max(n.velocity for n in notes_data)]
            print(f"Velocity range: {velocity_range[0]} - {velocity_range[1]}")
        print("="*80 + "\n")

    return notes_data


def get_instrument_name(midi_program):
    """
    Get the General MIDI instrument name from program number.
    """
    # General MIDI instrument names
    gm_instruments = [
        "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
        "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavi", "Celesta", "Glockenspiel",
        "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
        "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
        "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
        "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
        "Distortion Guitar", "Guitar harmonics", "Acoustic Bass", "Electric Bass (finger)",
        "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1",
        "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
        "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1", "String Ensemble 2",
        "SynthStrings 1", "SynthStrings 2", "Choir Aahs", "Voice Oohs", "Synth Voice",
        "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn",
        "Brass Section", "SynthBrass 1", "SynthBrass 2", "Soprano Sax", "Alto Sax",
        "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet",
        "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi",
        "Whistle", "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
        "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
        "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)",
        "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
        "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
        "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
        "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bag pipe", "Fiddle", "Shanai",
        "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom",
        "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise", "Seashore",
        "Bird Tweet", "Telephone Ring", "Helicopter", "Applause", "Gunshot"
    ]

    if 0 <= midi_program < len(gm_instruments):
        return gm_instruments[midi_program]
    return f"Unknown Instrument {midi_program}"


def notes_data_to_piano_roll(notes_data: list[MusicXMLNote], steps_per_second=24):
    max_time = max(n.start + n.duration for n in notes_data)
    total_steps = int(np.ceil(max_time * steps_per_second))

    # Changed from uint8 to float32 to store velocity values
    piano_roll_3d = np.zeros((128, 128, total_steps), dtype=np.float32)

    for note in notes_data:
        instrument = max(0, min(127, note.instrument))
        pitch = max(0, min(127, note.pitch))
        start_step = int(note.start * steps_per_second)
        end_step = int((note.start + note.duration) * steps_per_second)

        # Convert MIDI velocity (0-127) to 0-1 range
        velocity_normalized = note.velocity / 127.0

        if start_step < total_steps:
            end_step = min(end_step, total_steps)
            piano_roll_3d[instrument, pitch, start_step:end_step] = velocity_normalized

    return piano_roll_3d
