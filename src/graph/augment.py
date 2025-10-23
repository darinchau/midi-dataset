import numpy as np
from typing import Optional, Tuple, List
from copy import deepcopy
import random
from abc import ABC, abstractmethod

from .make_graph import NoteGraph
from ..extract.utils import get_ql_idx_map, get_ql_timesig_map
from ..extract import MusicXMLNote, index_octave_to_pitch
from .filter import MIDIFilterCriterion
from functools import lru_cache


@lru_cache(maxsize=256)
def transpose_idx_octave(index: int, octave: int, tranpose: int, compound: int) -> Tuple[int, int]:
    """Transpose a note given its index and octave by a number of semitones and octaves.
    Args:
        index (int): The index of the note.
        octave (int): The octave of the note.
        tranpose (int): The number of semitones to transpose.
        compound (int): The number of octaves to transpose."""
    new_index = index + tranpose
    new_octave = octave + compound

    new_pitch = index_octave_to_pitch(new_index, octave)
    old_pitch = index_octave_to_pitch(index, octave)
    if (new_pitch % 12) < (old_pitch % 12):
        new_octave += 1

    return new_index, new_octave


class NoteAugmentor(ABC):
    """Class for augmentation strategies on lists of MusicXMLNote objects."""

    @abstractmethod
    def augment(self, notes: List[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> List[MusicXMLNote]:
        """Apply a series of augmentations to the input notes."""
        pass


class AugmentError(ValueError):
    """Custom exception for augmentation errors. Indicates when an augmentation cannot be applied."""
    pass


class Transpose(NoteAugmentor):
    """Transpose all notes by a random number of semitones."""

    def __init__(self, min_semitones: int = -6, max_semitones: int = 6, min_compound: int = -1, max_compound: int = 1):
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.min_compound = min_compound
        self.max_compound = max_compound

    def augment(self, notes: list[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> list[MusicXMLNote]:
        """Transpose all notes by a given interval."""
        if rng is None:
            rng = np.random.default_rng()

        intervals_lof = rng.integers(self.min_semitones, self.max_semitones + 1).item()
        compound = rng.integers(self.min_compound, self.max_compound + 1).item()

        new_notes: list[MusicXMLNote] = []
        for note in notes:
            new_index, new_octave = transpose_idx_octave(note.index, note.octave, intervals_lof, compound)
            new_note = deepcopy(note)
            object.__setattr__(new_note, 'index', new_index)
            object.__setattr__(new_note, 'octave', new_octave)
            new_notes.append(new_note)
        return new_notes


class Retrograde(NoteAugmentor):
    """Reverse the temporal order of notes."""

    def augment(self, notes: List[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> List[MusicXMLNote]:
        """Reverse the temporal order of notes."""
        augmented = deepcopy(notes)
        max_end_time = max(note.start + note.duration for note in notes)
        max_end_time_ql = max(note.start_ql + note.duration_ql for note in notes if note.timesig is not None)
        new_notes: list[MusicXMLNote] = []
        for note in augmented:
            if note.timesig is not None:
                note_timesig_ql = get_ql_timesig_map()[note.timesig]
                new_start = max_end_time - (note.start + note.duration)
                new_start_ql = note_timesig_ql - (note.start_ql + note.duration_ql)
                new_note = deepcopy(note)
                object.__setattr__(new_note, 'start', new_start)
                object.__setattr__(new_note, 'start_ql', new_start_ql)
            else:
                new_start = max_end_time - (note.start + note.duration)
                new_start_ql = max_end_time_ql - (note.start_ql + note.duration_ql)
                new_note = deepcopy(note)
                object.__setattr__(new_note, 'start', new_start)
                object.__setattr__(new_note, 'start_ql', new_start_ql)
            new_notes.append(new_note)
        return new_notes


class AugmentNotes(NoteAugmentor):
    """Scale all note durations by 2x"""

    def __init__(self, change_time: bool = True) -> None:
        self.change_time = change_time

    def augment(self, notes: list[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> list[MusicXMLNote]:
        if rng is None:
            rng = np.random.default_rng()

        new_notes: list[MusicXMLNote] = []
        for note in notes:
            new_note = deepcopy(note)
            object.__setattr__(new_note, 'start_ql', note.start_ql * 2)
            if note.timesig is None:
                object.__setattr__(new_note, 'duration_ql', note.duration_ql * 2)
            else:
                total_ql = get_ql_timesig_map()[note.timesig]
                object.__setattr__(new_note, 'duration_ql', (note.duration_ql * 2) % total_ql)

            if self.change_time:
                object.__setattr__(new_note, 'start', note.start * 2)
                object.__setattr__(new_note, 'duration', note.duration * 2)
            new_notes.append(new_note)
        return new_notes


class AddPassingNote(NoteAugmentor):
    """Add a passing note between two existing notes."""

    def __init__(self, interval: Optional[int] = None, compound: Optional[int] = None):
        self.interval = interval
        self.compound = compound

    def augment(self, notes: list[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> list[MusicXMLNote]:
        if rng is None:
            rng = np.random.default_rng()

        interval = self.interval if self.interval is not None else rng.integers(-12, 12).item()
        compound = self.compound if self.compound is not None else rng.integers(-1, 1).item()

        # Pick one note, cut it in half, and insert a passing note
        note_idx = rng.integers(0, len(notes)).item()
        note = notes[note_idx]
        new_index, new_octave = transpose_idx_octave(note.index, note.octave, interval, compound)
        passing_note = MusicXMLNote(
            instrument=note.instrument,
            start=note.start + note.duration / 2,
            duration=note.duration / 2,
            start_ql=note.start_ql + note.duration_ql / 2,
            duration_ql=note.duration_ql / 2,
            index=new_index,
            octave=new_octave,
            timesig=note.timesig,
            barline=False,
            velocity=note.velocity
        )
        old_note = MusicXMLNote(
            instrument=note.instrument,
            start=note.start,
            duration=note.duration / 2,
            start_ql=note.start_ql,
            duration_ql=note.duration_ql / 2,
            index=note.index,
            octave=note.octave,
            timesig=note.timesig,
            barline=note.barline,
            velocity=note.velocity
        )
        new_notes = notes[:note_idx] + [old_note, passing_note] + notes[note_idx + 1:]
        return new_notes


class RhythmicVariation(NoteAugmentor):
    """Apply rhythmic variations while preserving pitch sequence."""

    def __init__(self, variation_strength: float = 1.0):
        self.variation_strength = variation_strength

    def sample_list_normal(self, ls: np.ndarray, rng: np.random.Generator) -> float:
        values = np.array(ls)
        weights = -0.5 * ((values) / self.variation_strength)**2
        log_p_max = weights.max()
        weights = {k: np.exp(v - log_p_max) for k, v in zip(ls, weights)}
        total_weight = sum(weights.values())
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for k, w in weights.items():
            cumulative += w
            if cumulative >= threshold:
                return k
        return ls[-1]

    def augment(self, notes: list[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> list[MusicXMLNote]:
        if rng is None:
            rng = np.random.default_rng()

        new_notes: list[MusicXMLNote] = []
        cumulative_ql_shift = 0.0
        base_ql_shifts = np.array([-1, -0.5, -0.25, 0.0, 0.25, 0.5, 1])
        for note in notes:
            ql_shifts = base_ql_shifts + cumulative_ql_shift
            ql_shift = self.sample_list_normal(ql_shifts, rng)
            new_start_ql = note.start_ql + cumulative_ql_shift + ql_shift
            new_note = deepcopy(note)
            object.__setattr__(new_note, 'start_ql', new_start_ql)
            new_notes.append(new_note)
            cumulative_ql_shift += ql_shift

        return new_notes


class ReplaceEnharmonic(NoteAugmentor):
    """Randomly replace notes with their enharmonic equivalents."""

    def __init__(self, replace_prob: float = 0.1):
        self.replace_prob = replace_prob

    def augment(self, notes: list[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> list[MusicXMLNote]:
        if rng is None:
            rng = np.random.default_rng()

        new_notes: list[MusicXMLNote] = []
        for note in notes:
            new_note = deepcopy(note)
            if rng.random() < self.replace_prob:
                choices: list[tuple[int, int]] = []
                if note.index - 12 in MIDIFilterCriterion.permitted_index():
                    choices.append((note.index - 12, 0))  # Transpose by diminished second
                if note.index + 12 in MIDIFilterCriterion.permitted_index():
                    choices.append((note.index + 12, -1))  # Transpose by augmented 7th -> lower octave
                if choices:
                    (new_index, new_octave) = transpose_idx_octave(note.index, note.octave, *rng.choice(choices))
                    object.__setattr__(new_note, 'index', new_index)
                    object.__setattr__(new_note, 'octave', new_octave)
            new_notes.append(new_note)
        return new_notes


class SubstituteNote(NoteAugmentor):
    """Randomly substitute notes with other valid notes."""

    def __init__(self, substitute_prob: float = 0.1):
        self.substitute_prob = substitute_prob

    def augment(self, notes: list[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> list[MusicXMLNote]:
        if rng is None:
            rng = np.random.default_rng()

        permitted_indices = MIDIFilterCriterion.permitted_index()
        permitted_octaves = MIDIFilterCriterion.permitted_octaves()
        shift = [
            (2, 0),  # Up major second
            (-2, -1),  # Down major second
            (3, -1),  # Down minor third
            (-3, 0),  # Up minor third
            (4, 0),  # Up major third
            (-4, -1),  # Down major third
            (5, -1),  # Down minor second
            (-5, 0),  # Up minor second
            (7, 0),  # Add sharp
            (-7, -1)  # Add flat
        ]

        new_notes: list[MusicXMLNote] = []
        for note in notes:
            new_note = deepcopy(note)
            if rng.random() < self.substitute_prob:
                transpose_idx, transpose_octave = rng.choice(shift)
                new_index, new_octave = transpose_idx_octave(note.index, note.octave, transpose_idx, transpose_octave)
                if new_index in permitted_indices and new_octave in permitted_octaves:
                    object.__setattr__(new_note, 'index', new_index)
                    object.__setattr__(new_note, 'octave', new_octave)
            new_notes.append(new_note)
        return new_notes


class TimePerturbations(NoteAugmentor):
    """Apply small time perturbations to note start times."""

    def __init__(self, max_perturbation: float = 0.1, strength: float = 1.0, perturb_probability: float = 0.25):
        self.max_perturbation = max_perturbation
        self.strength = strength
        self.perturb_probability = perturb_probability

    def augment(self, notes: list[MusicXMLNote], rng: Optional[np.random.Generator] = None) -> list[MusicXMLNote]:
        if rng is None:
            rng = np.random.default_rng()

        new_notes: list[MusicXMLNote] = []
        cumulative_shift = 0.0
        for note in notes:
            new_start = note.start + cumulative_shift
            if rng.random() < self.perturb_probability:
                perturbation = rng.normal(0, self.strength)
                perturbation = np.clip(perturbation, -self.max_perturbation, self.max_perturbation)
                cumulative_shift += perturbation
                new_start += perturbation
            new_note = deepcopy(note)
            object.__setattr__(new_note, 'start', new_start)
            new_notes.append(new_note)
        return new_notes
