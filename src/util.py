import os
import pandas as pd
from functools import cache
from xml.etree.ElementTree import Element
from .constants import METADATA_PATH, MIDI_ROOT, XML_ROOT


def get_path(root: str, index: str) -> str:
    # Look through the giant-midi-archive directory for the file with the given index like a trie
    # and return the path to that file.
    path = [root]
    while True:
        p = os.path.join(*path)
        for pt in os.listdir(p):
            if index.startswith(pt) and os.path.isdir(os.path.join(*path, pt)):
                path.append(pt)
                break
            elif pt.startswith(index) and os.path.isfile(os.path.join(*path, pt)):
                return os.path.join(*path, pt)
        else:
            break
    raise FileNotFoundError(f"File with index {index} not found in {root}.")


def iterate_dataset(root: str):
    """Creates a iterator that yields the absolute path of each file in the dataset."""
    df = pd.read_csv(os.path.join(METADATA_PATH, "mapping.csv"), sep='\t')
    for index in df['index']:
        try:
            yield get_path(root, index)
        except FileNotFoundError as e:
            print(f"File not found for index {index}: {e}")
            continue


def iterate_midis():
    return iterate_dataset(MIDI_ROOT)


def iterate_xmls(check: bool = True):
    """Iterate through all XML files in the dataset.
    If `check` is True, it will validate each XML file using `is_valid_xml`.
    """
    from .extract.analyze import is_valid_xml
    for path in iterate_dataset(XML_ROOT):
        if check:
            if is_valid_xml(path):
                yield path
            else:
                print(f"Invalid XML file: {path}")
                continue
        else:
            yield path


@cache
def get_gm_instruments_map():
    """
    Get the General MIDI instrument name from program number.
    """
    # General MIDI instrument names
    return {
        0: "Acoustic Grand Piano",
        1: "Bright Acoustic Piano",
        2: "Electric Grand Piano",
        3: "Honky-tonk Piano",
        4: "Electric Piano 1",
        5: "Electric Piano 2",
        6: "Harpsichord",
        7: "Clavi",
        8: "Celesta",
        9: "Glockenspiel",
        10: "Music Box",
        11: "Vibraphone",
        12: "Marimba",
        13: "Xylophone",
        14: "Tubular Bells",
        15: "Dulcimer",
        16: "Drawbar Organ",
        17: "Percussive Organ",
        18: "Rock Organ",
        19: "Church Organ",
        20: "Reed Organ",
        21: "Accordion",
        22: "Harmonica",
        23: "Tango Accordion",
        24: "Acoustic Guitar (nylon)",
        25: "Acoustic Guitar (steel)",
        26: "Electric Guitar (jazz)",
        27: "Electric Guitar (clean)",
        28: "Electric Guitar (muted)",
        29: "Overdriven Guitar",
        30: "Distortion Guitar",
        31: "Guitar harmonics",
        32: "Acoustic Bass",
        33: "Electric Bass (finger)",
        34: "Electric Bass (pick)",
        35: "Fretless Bass",
        36: "Slap Bass 1",
        37: "Slap Bass 2",
        38: "Synth Bass 1",
        39: "Synth Bass 2",
        40: "Violin",
        41: "Viola",
        42: "Cello",
        43: "Contrabass",
        44: "Tremolo Strings",
        45: "Pizzicato Strings",
        46: "Orchestral Harp",
        47: "Timpani",
        48: "String Ensemble 1",
        49: "String Ensemble 2",
        50: "SynthStrings 1",
        51: "SynthStrings 2",
        52: "Choir Aahs",
        53: "Voice Oohs",
        54: "Synth Voice",
        55: "Orchestra Hit",
        56: "Trumpet",
        57: "Trombone",
        58: "Tuba",
        59: "Muted Trumpet",
        60: "French Horn",
        61: "Brass Section",
        62: "SynthBrass 1",
        63: "SynthBrass 2",
        64: "Soprano Sax",
        65: "Alto Sax",
        66: "Tenor Sax",
        67: "Baritone Sax",
        68: "Oboe",
        69: "English Horn",
        70: "Bassoon",
        71: "Clarinet",
        72: "Piccolo",
        73: "Flute",
        74: "Recorder",
        75: "Pan Flute",
        76: "Blown Bottle",
        77: "Shakuhachi",
        78: "Whistle",
        79: "Ocarina",
        80: "Lead 1 (square)",
        81: "Lead 2 (sawtooth)",
        82: "Lead 3 (calliope)",
        83: "Lead 4 (chiff)",
        84: "Lead 5 (charang)",
        85: "Lead 6 (voice)",
        86: "Lead 7 (fifths)",
        87: "Lead 8 (bass + lead)",
        88: "Pad 1 (new age)",
        89: "Pad 2 (warm)",
        90: "Pad 3 (polysynth)",
        91: "Pad 4 (choir)",
        92: "Pad 5 (bowed)",
        93: "Pad 6 (metallic)",
        94: "Pad 7 (halo)",
        95: "Pad 8 (sweep)",
        96: "FX 1 (rain)",
        97: "FX 2 (soundtrack)",
        98: "FX 3 (crystal)",
        99: "FX 4 (atmosphere)",
        100: "FX 5 (brightness)",
        101: "FX 6 (goblins)",
        102: "FX 7 (echoes)",
        103: "FX 8 (sci-fi)",
        104: "Sitar",
        105: "Banjo",
        106: "Shamisen",
        107: "Koto",
        108: "Kalimba",
        109: "Bag pipe",
        110: "Fiddle",
        111: "Shanai",
        112: "Tinkle Bell",
        113: "Agogo",
        114: "Steel Drums",
        115: "Woodblock",
        116: "Taiko Drum",
        117: "Melodic Tom",
        118: "Synth Drum",
        119: "Reverse Cymbal",
        120: "Guitar Fret Noise",
        121: "Breath Noise",
        122: "Seashore",
        123: "Bird Tweet",
        124: "Telephone Ring",
        125: "Helicopter",
        126: "Applause",
        127: "Gunshot"
    }


@cache
def get_time_signature_map():
    return {
        0: "UNK",
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
