# Extracts sparse compound word tokens from MusicXML files.

import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from dataclasses import dataclass
from .analyze import MusicXMLNote, musicxml_to_tokens as parse_musicxml
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
    notes_data = parse_musicxml(xml_path, debug=debug)
    notes_data = [element_to_nparray(n) for n in notes_data]
    notes_data = np.stack(notes_data)
    return notes_data
