import music21 as m21
import partitura as pt
from partitura.utils.music import ensure_notearray
from util import get_path


def tokenize_score_partitura(midi_path: str):
    # The load_music21 method doesnt seem to work properly. This is more consistent
    m21score = m21.converter.parse(midi_path)
    tmp_path = m21score.write("musicxml")
    assert tmp_path is not None, "Failed to write MusicXML from MIDI file."
    extended_score_note_array = ensure_notearray(
        pt.load_score(tmp_path),
        include_pitch_spelling=True,  # adds 3 fields: step, alter, octave
        include_key_signature=True,  # adds 2 fields: ks_fifths, ks_mode
        include_time_signature=True,  # adds 2 fields: ts_beats, ts_beat_type
        include_metrical_position=True,  # adds 3 fields: is_downbeat, rel_onset_div, tot_measure_div
        include_grace_notes=True  # adds 2 fields: is_grace, grace_type
    )
