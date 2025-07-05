import pandas as pd
import os
import json
from tqdm import tqdm

# Stores all the functions that have been used to create labels for the dataset.
# Maybe exports a few useful ones idk


def load_mapping():
    df = pd.read_csv(os.path.join("E:/giant-midi-archive/mapping.csv"), sep='\t')
    df['original_path'] = df['original_path'].apply(lambda x: x.replace("\\", "/"))
    return df


def create_aria_labels(df: pd.DataFrame):
    print(f"Creating aria labels for {len(df)} files...")

    with open("data/v2/aria-midi-v1-ext/aria-midi-v1-ext/metadata.json", 'r') as f:
        metadata = json.load(f)
    df['aria_midi_number'] = df['original_path'].apply(lambda x: os.path.basename(x).split("_")[0] if "aria-midi-v1-ext" in x else "")

    def apply(key):
        if not key:
            return ""
        info = metadata[str(int(key))]['metadata']
        tags = []
        if "difficulty" in info:
            tags.append(f"{info["difficulty"]} difficulty")
        if "form" in info:
            tags.append(f"{info['form']} form")
        if "key_signature" in info:
            tags.append(f"Key: {info['key_signature']}")
        if "time_signature" in info:
            tags.append(f"Time: {info['time_signature']}")
        if "music_period" in info:
            tags.append(f"Period: {info['music_period']}")
        if "style" in info:
            tags.append(f"Style: {info['style']}")
        if "composer" in info:
            tags.append(f"Composer: {info['composer']}")
        return ", ".join(tags)

    df['aria_labels'] = df['aria_midi_number'].apply(apply)
    df.drop(columns=['aria_midi_number'], inplace=True)
    return df


def create_deepseek_labels(df: pd.DataFrame):
    print(f"Creating deepseek labels for {len(df)} files...")
    infos = pd.read_csv("E:/giant-midi-archive/info.csv", sep='\t')
    df1 = df.merge(infos, on='original_path', how='outer')
    df1['deepseek_desc'] = df1['deepseek_desc'].replace(pd.NA, '', regex=True)
    df1 = df1.fillna('')
    df1 = df1[df1['index'] != ""].drop_duplicates(subset=['index'], keep='first')
    return df1


def create_lengths(df: pd.DataFrame):
    import pretty_midi

    def midi_file_length(root: str, index: str) -> float:
        try:
            path = get_path(root, index)
            midi_data = pretty_midi.PrettyMIDI(path)
            length_in_seconds = midi_data.get_end_time()
            return length_in_seconds
        except Exception as e:
            return -1

    tqdm.pandas()

    df['length'] = df['index'].progress_map(lambda idx: midi_file_length("E:/giant-midi-archive", idx))
    return df


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
