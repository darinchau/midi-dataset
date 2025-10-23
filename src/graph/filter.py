# Implements a filter and normalization criterion
# Stores the filter function we use to get clean XMLs in the current phase of training
from typing import List, Optional
from functools import cache
import numpy as np

from ..extract.utils import get_time_signature_map
from ..extract import musicxml_to_notes, MusicXMLNote
from .make_graph import NoteGraph


class MIDIFilterCriterion:
    @cache
    def permitted_index(self) -> List[int]:
        """Returns the acceptable range of note indices."""
        return list(range(-15, 20))

    @cache
    def permitted_pitch(self) -> List[int]:
        """Returns the acceptable range of note pitches."""
        return list(range(21, 109))

    @cache
    def permitted_instruments(self) -> List[int]:
        """Returns the acceptable range of note instruments."""
        # Use a subset of the more common General MIDI instruments
        # 0: Acoustic Grand Piano
        # 1: Bright Acoustic Piano
        # 2: Electric Grand Piano
        # 6: Harpsichord
        # 40: Violin
        # 41: Viola
        # 42: Cello
        # 43: Contrabass
        return [0, 1, 2, 6, 40, 41, 42, 43]

    @cache
    def permitted_octaves(self) -> List[int]:
        """Returns the acceptable range of note octaves."""
        # Calculated automatically from permitted pitches and permitted index
        # from the relation:
        # pitch = 12 * octave + ([0, 7, 2, 9, 4, 11, 5][index % 7] + (index + 1) // 7
        min_pitch = min(self.permitted_pitch())
        max_pitch = max(self.permitted_pitch())
        min_index = min(self.permitted_index())
        max_index = max(self.permitted_index())
        min_octave = (min_pitch - (min_index + 1) // 7 - 11) // 12
        max_octave = (max_pitch - (max_index + 1) // 7) // 12
        return list(range(min_octave, max_octave + 1))

    @cache
    def permitted_timesigs(self) -> List[int]:
        """Returns the acceptable time signatures in their index form as defined in utils."""
        ls = get_time_signature_map().keys()
        return [i for i in ls if get_time_signature_map()[i] is not None]

    def get_bad_midi_reason(self, file_path: str | list[MusicXMLNote]) -> str:
        """Returns an empty string if the MIDI file is good, otherwise a reason for rejection."""
        if isinstance(file_path, str):
            try:
                notes = musicxml_to_notes(file_path, no_barline=True)
            except Exception as e:
                return "ParseError"
        else:
            notes = file_path

        if len(notes) < 100:
            return "TooFewNotes"

        if any(note.timesig is None for note in notes):
            return "MissingTimeSignature"

        permitted_timesig_values = {get_time_signature_map()[i] for i in self.permitted_timesigs()}
        if any(note.timesig not in permitted_timesig_values for note in notes):
            return "InvalidTimeSignature"

        index_range = self.permitted_index()
        if any(note.index not in index_range for note in notes):
            # Not very rigorous, but no triple sharps or flats allowed for now
            return "InvalidIndex"

        if any(note.timesig is None for note in notes):
            return "MissingTimeSignature"

        pitch_range = self.permitted_pitch()
        if any(not note.barline and (note.pitch not in pitch_range) for note in notes):
            # Outside piano range
            return "InvalidPitch"

        permitted_instruments = self.permitted_instruments()
        if any(not note.barline and note.instrument not in permitted_instruments for note in notes):
            return "InvalidInstrument"

        return ""

    def check_graph(self, graph: NoteGraph) -> Optional[str]:
        """Checks if the graph satisfies the filter criteria. Returns None if good, otherwise a reason."""
        index_emb = graph.feature_info.feature_names.index("index")
        instrument_emb = graph.feature_info.feature_names.index("instrument")
        pitch_emb = graph.feature_info.feature_names.index("pitch")
        timesig_emb = graph.feature_info.feature_names.index("timesig")

        index_values = set(graph.node_features[:, index_emb].astype(int).tolist())
        for idx in index_values:
            if idx not in self.permitted_index():
                return "InvalidIndex"

        instrument_values = set(graph.node_features[:, instrument_emb].astype(int).tolist())
        for inst in instrument_values:
            if inst not in self.permitted_instruments():
                return "InvalidInstrument"

        pitch_values = set(graph.node_features[:, pitch_emb].astype(int).tolist())
        for pitch in pitch_values:
            if pitch not in self.permitted_pitch():
                return "InvalidPitch"

        timesig_values = set(graph.node_features[:, timesig_emb].astype(int).tolist())
        permitted_timesig_indices = set(self.permitted_timesigs())
        for ts in timesig_values:
            if ts not in permitted_timesig_indices:
                return "InvalidTimeSignature"

        # Check if start times are non-negative
        start_idx = graph.feature_info.feature_names.index("start")
        if np.any(graph.node_features[:, start_idx] < 0):
            return "NegativeStartTime"

        return None

    def normalize(self, graph: NoteGraph) -> NoteGraph:
        """Normalizes the note indices, instruments, and pitches to start from 0."""
        # Create mapping dictionaries for each categorical feature
        index_map = {index: i for i, index in enumerate(sorted(self.permitted_index()))}
        instrument_map = {inst: i for i, inst in enumerate(sorted(self.permitted_instruments()))}
        pitch_map = {pitch: i for i, pitch in enumerate(sorted(self.permitted_pitch()))}
        timesig_map = {ts: i for i, ts in enumerate(sorted(self.permitted_timesigs()))}

        # Get column indices for each feature
        index_emb = graph.feature_info.feature_names.index("index")
        instrument_emb = graph.feature_info.feature_names.index("instrument")
        pitch_emb = graph.feature_info.feature_names.index("pitch")
        timesig_emb = graph.feature_info.feature_names.index("timesig")

        # Apply normalization via mappings
        graph.node_features[:, index_emb] = np.array(
            [index_map[int(v.item())] for v in graph.node_features[:, index_emb]],
        )

        graph.node_features[:, instrument_emb] = np.array(
            [instrument_map[int(v.item())] for v in graph.node_features[:, instrument_emb]],
        )

        graph.node_features[:, pitch_emb] = np.array(
            [pitch_map[int(v.item())] for v in graph.node_features[:, pitch_emb]],
        )

        graph.node_features[:, timesig_emb] = np.array(
            [timesig_map[int(v.item())] for v in graph.node_features[:, timesig_emb]],
        )

        # Move the start time to 0
        start_idx = graph.feature_info.feature_names.index("start")
        min_start_time = np.min(graph.node_features[:, start_idx])
        graph.node_features[:, start_idx] -= min_start_time

        return graph


def is_good_midi(file_path: str | list[MusicXMLNote]) -> bool:
    """Returns True if the MIDI file is considered good based on our filtering criteria.
    Just a convenience wrapper around MIDIFilterCriterion."""
    return not MIDIFilterCriterion().get_bad_midi_reason(file_path)


def is_good_notegraph(graph: NoteGraph) -> bool:
    """Returns True if the NoteGraph is considered good based on our filtering criteria.
    Just a convenience wrapper around MIDIFilterCriterion."""
    return MIDIFilterCriterion().check_graph(graph) is None
