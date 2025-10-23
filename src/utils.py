from collections import OrderedDict
import os
import pandas as pd
from functools import cache
from .constants import METADATA_PATH, MIDI_ROOT, XML_ROOT
import pickle
from tqdm.auto import tqdm

CACHE_PATH = "./resources/cache"


def get_path(root: str, index: str) -> str:
    # Look through the giant-midi-archive directory for the file with the given index like a trie
    # and return the path to that file.
    # Optimization: First try the hardcoded 2-4 digit index, then try the prefix search method
    target_dir = os.path.join(root, index[:2], index[:4])
    path = None
    if os.path.exists(target_dir):
        if root == MIDI_ROOT:
            path = os.path.join(target_dir, index + ".mid")
        elif root == XML_ROOT:
            path = os.path.join(target_dir, index + ".xml")
        else:
            files = os.listdir(target_dir)
            for file in files:
                if file.startswith(index):
                    path = os.path.join(target_dir, file)
                    break

    if path is not None and os.path.exists(path):
        return path

    path_dirs = [root]
    while True:
        p = os.path.join(*path_dirs)
        for pt in os.listdir(p):
            if index.startswith(pt) and os.path.isdir(os.path.join(*path_dirs, pt)):
                path_dirs.append(pt)
                break
            elif pt.startswith(index) and os.path.isfile(os.path.join(*path_dirs, pt)):
                return os.path.join(*path_dirs, pt)
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


def iterate_subset(root: str, subsets: list[str] | str):
    subset_prefix = {
        "aria": "data/v2/aria-midi-v1-ext",
        "mmd": "data/v2/MMD_MIDI",
        "xmidi": "data/v1/XMIDI_Dataset",
        "classical": "data/v1/hf-midi-classical-music",
        "v1": "data/v1",
        "v2": "data/v2",
        "all": "data"
    }

    if isinstance(subsets, str):
        subsets = [subsets]

    for subset in subsets:
        if subset not in subset_prefix:
            raise ValueError(f"Subset {subset} not recognized. Available subsets: {sorted(subset_prefix.keys())}")

    df = pd.read_csv(os.path.join(METADATA_PATH, "mapping.csv"), sep='\t')
    for _, row in df.iterrows():
        index = row['index']
        mapping_value = row['original_path']
        if any(mapping_value.startswith(subset_prefix[s]) for s in subsets):
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


def get_all_xml_paths() -> list[str]:
    """
    Get all valid XML file paths in the dataset.
    If the cache exists, it will load from there.
    Otherwise, it will iterate through the dataset and create the cache, taking an absurd amount of time.
    """
    cache_file = os.path.join(CACHE_PATH, "valid_xmls_cp.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            files = pickle.load(f)
    else:
        files = list(tqdm(iterate_dataset(XML_ROOT), desc="Loading XML files"))
        with open(cache_file, 'wb') as f:
            pickle.dump(files, f)
    return files


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


def clear_cuda():
    import torch
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def count_parameters(module):
    """Count the number of parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def count_direct_parameters(module):
    """Count only the parameters directly belonging to this module, not its children."""
    direct_params = set(module.parameters(recurse=False))
    return sum(p.numel() for p in direct_params)


def print_model_hierarchy(model, indent=0, prefix="", show_shapes=False, max_depth=None):
    """
    Print a hierarchical structure of a PyTorch model with parameter counts.

    Args:
        model: PyTorch model
        indent: Current indentation level
        prefix: Prefix for the current line
        show_shapes: If True, also show parameter shapes
        max_depth: Maximum depth to traverse (None for unlimited)
    """
    if max_depth is not None and indent > max_depth:
        return

    # Count parameters for this module and its children
    total_params = count_parameters(model)
    direct_params = count_direct_parameters(model)

    # Format the parameter count
    param_str = f"{total_params:,}"
    if direct_params > 0 and direct_params != total_params:
        param_str = f"{total_params:,} (direct: {direct_params:,})"

    # Get module type name
    module_type = model.__class__.__name__

    # Print current module
    indent_str = "  " * indent
    print(f"{indent_str}{prefix}{module_type}: {param_str} params")

    # Optionally show parameter shapes
    if show_shapes and direct_params > 0:
        for name, param in model.named_parameters(recurse=False):
            print(f"{indent_str}    └─ {name}: {list(param.shape)} = {param.numel():,} params")

    # Recursively print children
    children = list(model.named_children())
    for i, (name, child) in enumerate(children):
        is_last = (i == len(children) - 1)
        child_prefix = f"{name} - "
        print_model_hierarchy(child, indent + 1, child_prefix, show_shapes, max_depth)


def analyze_model(model, show_shapes=False, max_depth=None):
    """
    Analyze and return a comprehensive summary of a PyTorch model.

    Args:
        model: PyTorch model to analyze
        show_shapes: If True, show parameter shapes
        max_depth: Maximum depth to traverse

    Returns:
        str: Formatted model analysis string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL ARCHITECTURE HIERARCHY")
    lines.append("=" * 80)

    # Calculate parameters
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    lines.append("")
    lines.append(f"Total Parameters: {total_params:,}")
    lines.append(f"Trainable Parameters: {trainable_params:,}")
    lines.append(f"Non-trainable Parameters: {non_trainable_params:,}")
    lines.append("-" * 80)
    lines.append("")

    # Get hierarchical structure (you'll need to modify print_model_hierarchy too)
    hierarchy_lines = get_model_hierarchy_string(model, show_shapes=show_shapes, max_depth=max_depth)
    lines.extend(hierarchy_lines)
    lines.append("=" * 80)

    return "\n".join(lines)


def get_model_hierarchy_string(model, indent=0, prefix="", show_shapes=False, max_depth=None):
    """
    Get hierarchical structure of a PyTorch model as list of strings.
    """
    if max_depth is not None and indent > max_depth:
        return []

    lines = []

    # Count parameters for this module and its children
    total_params = count_parameters(model)
    direct_params = count_direct_parameters(model)

    # Format the parameter count
    param_str = f"{total_params:,}"
    if direct_params > 0 and direct_params != total_params:
        param_str = f"{total_params:,} (direct: {direct_params:,})"

    # Get module type name
    module_type = model.__class__.__name__

    # Add current module
    indent_str = "  " * indent
    lines.append(f"{indent_str}{prefix}{module_type}: {param_str} params")

    # Optionally show parameter shapes
    if show_shapes and direct_params > 0:
        for name, param in model.named_parameters(recurse=False):
            lines.append(f"{indent_str}    └─ {name}: {list(param.shape)} = {param.numel():,} params")

    # Recursively add children
    children = list(model.named_children())
    for i, (name, child) in enumerate(children):
        child_prefix = f"{name} - "
        child_lines = get_model_hierarchy_string(child, indent + 1, child_prefix, show_shapes, max_depth)
        lines.extend(child_lines)

    return lines


def get_model_summary_dict(model, prefix=""):
    """
    Get a dictionary representation of the model hierarchy with parameter counts.

    Returns:
        dict: Nested dictionary with module names and parameter counts
    """
    # Add current module info
    total_params = count_parameters(model)
    direct_params = count_direct_parameters(model)

    module_info = {
        "type": model.__class__.__name__,
        "total_params": total_params,
        "direct_params": direct_params,
        "children": OrderedDict()
    }

    # Add children recursively
    for name, child in model.named_children():
        module_info["children"][name] = get_model_summary_dict(child, f"{prefix}{name}.")

    return module_info
