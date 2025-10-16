# Implements a filter and normalization criterion
# Stores the filter function we use to get clean XMLs in the current phase of training
from typing import List, Optional
from functools import cache
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

    def normalize(self, graph: NoteGraph) -> NoteGraph:
        """Normalizes the note indices, instruments, and pitches to start from 0."""
        import torch

        # Create mapping dictionaries for each categorical feature
        index_map = {index: i for i, index in enumerate(sorted(self.permitted_index()))}
        instrument_map = {inst: i for i, inst in enumerate(sorted(self.permitted_instruments()))}
        pitch_map = {pitch: i for i, pitch in enumerate(sorted(self.permitted_pitch()))}

        # Get column indices for each feature
        index_emb = graph.feature_info["feature_names"].index("index")
        instrument_emb = graph.feature_info["feature_names"].index("instrument")
        pitch_emb = graph.feature_info["feature_names"].index("pitch")

        # Apply normalization via mappings
        graph.node_features[:, index_emb] = torch.tensor(
            [index_map[int(v.item())] for v in graph.node_features[:, index_emb]],
            dtype=torch.long
        )

        graph.node_features[:, instrument_emb] = torch.tensor(
            [instrument_map[int(v.item())] for v in graph.node_features[:, instrument_emb]],
            dtype=torch.long
        )

        graph.node_features[:, pitch_emb] = torch.tensor(
            [pitch_map[int(v.item())] for v in graph.node_features[:, pitch_emb]],
            dtype=torch.long
        )

        return graph


def is_good_midi(file_path: str | list[MusicXMLNote]) -> bool:
    """Returns True if the MIDI file is considered good based on our filtering criteria.
    Just a convenience wrapper around MIDIFilterCriterion."""
    return not MIDIFilterCriterion().get_bad_midi_reason(file_path)
