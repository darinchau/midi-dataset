import pandas as pd
import os
import json


def create_aria_labels(root: str):
    df = pd.read_csv(os.path.join(root, "mapping.csv"), sep='\t')
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
    df = df[df['aria_labels'] != ""]
    df.drop(columns=['original_path'], inplace=True)
    df.to_csv(os.path.join(root, "aria.csv"), sep='\t', index=False)
