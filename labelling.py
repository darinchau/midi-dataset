# This script stores all the scripts that have been used to analyse metadata and label MIDI files in the Giant MIDI Archive.
# And maybe in the process exports a few useful ones
import pandas as pd
import os
import json
from tqdm import tqdm
from util import get_path


def load_mapping(root: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, "mapping.csv"), sep='\t')
    df['original_path'] = df['original_path'].apply(lambda x: x.replace("\\", "/"))
    return df


def create_aria_labels(raw_data_path: str, df: pd.DataFrame):
    print(f"Creating aria labels for {len(df)} files...")

    with open(os.path.join(raw_data_path, "/v2/aria-midi-v1-ext/aria-midi-v1-ext/metadata.json"), 'r') as f:
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


def create_deepseek_labels(root: str, df: pd.DataFrame):
    print(f"Creating deepseek labels for {len(df)} files...")
    infos = pd.read_csv(os.path.join(root, "info.csv"), sep='\t')
    df1 = df.merge(infos, on='original_path', how='outer')
    df1['deepseek_desc'] = df1['deepseek_desc'].replace(pd.NA, '', regex=True)
    df1 = df1.fillna('')
    df1 = df1[df1['index'] != ""].drop_duplicates(subset=['index'], keep='first')
    return df1


def create_lengths(root: str, df: pd.DataFrame):
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

    df['length'] = df['index'].progress_map(lambda idx: midi_file_length(root, idx))
    return df


def get_note_counts(root: str, df: pd.DataFrame):
    import pretty_midi

    def count_notes_in_midi(midi_file_path: str):
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            note_count = 0
            for instrument in midi_data.instruments:
                note_count += len(instrument.notes)
            return note_count
        except Exception as e:
            return -1
    tqdm.pandas(desc="Counting notes...")
    df['note_count'] = df['index'].progress_apply(lambda x: count_notes_in_midi(get_path(root, x)))
    return df
