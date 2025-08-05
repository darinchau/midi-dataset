import os
import numpy as np
from .analyse import parse_musicxml, MusicXMLNote
from .constants import XML_ROOT


def tokenize_midi_one_hot(notes: list[MusicXMLNote]) -> np.ndarray:
    """
    Tokenize a list of MusicXMLNote objects into a one-hot encoded 3D piano roll.

    Args:
        notes (list[MusicXMLNote]): List of MusicXMLNote objects.

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
