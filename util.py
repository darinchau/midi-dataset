import os
import pandas as pd
from constants import METADATA_PATH


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
