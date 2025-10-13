from functools import cache
from xml.etree.ElementTree import Element
from ..utils import get_gm_instruments_map


@cache
def get_time_signature_map():
    return {
        0: None,
        1: "4/4",
        2: "3/4",
        3: "2/4",
        4: "6/8",
        5: "5/4",
        6: "7/8",
        7: "9/8",
        8: "10/8",
        9: "11/8",
        10: "12/8",
        11: "3/8",
        12: "6/4",
        13: "7/4",
        14: "5/8",
        # Position 15 is reserved for barlines
    }


@cache
def get_inv_time_signature_map():
    return {v: k for k, v in get_time_signature_map().items()}


@cache
def get_inv_gm_instruments_map():
    return {v: k for k, v in get_gm_instruments_map().items()}


def get_text_or_raise(elem: Element | None) -> str:
    """
    Get text from an XML element, raise ValueError if not found.
    """
    if elem is None or elem.text is None:
        raise ValueError("Element text is None")
    return elem.text


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
