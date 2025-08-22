# This script stores all the scripts that have been used to analyse metadata and label MIDI files in the Giant MIDI Archive.
# And maybe in the process exports a few useful ones
# The functions might not run because the paths might not be correct.
import pandas as pd
import os
import json
from tqdm import tqdm
from src.utils import get_path
import logging
import traceback
from threading import Lock
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
from datetime import datetime
import os
import time
import pandas as pd
from src.constants import XML_ROOT, METADATA_PATH, MIDI_ROOT


def load_mapping() -> pd.DataFrame:
    """Loads the mapping of MIDI files to their metadata."""
    root = METADATA_PATH
    df = pd.read_csv(os.path.join(root, "mapping.csv"), sep='\t')
    df['original_path'] = df['original_path'].apply(lambda x: x.replace("\\", "/"))
    return df


def create_xml_note_counts():
    """Creates the num."""
    from src.extract.analyze import musicxml_to_notes

    df = load_mapping()

    def xml_note_count(index: str) -> int:
        try:
            fp = get_path(XML_ROOT, index)
            notes_data = musicxml_to_notes(fp)
            return len([x for x in notes_data if not x.barline])
        except Exception as e:
            print(f"Error processing {index}: {e}")
            return -1

    tqdm.pandas()

    df['xml_note_count'] = df['index'].progress_apply(xml_note_count)
    df.drop("original_path", axis=1, inplace=True)
    df.to_csv(os.path.join(METADATA_PATH, "xml_note_counts.csv"), index=False, sep="\t")
    return df


def create_aria_labels():
    """Creates the key aria labels: difficulty, form, key_signature, time_signature, music_period, style, composer in the ARIA dataset"""
    df = load_mapping()
    print(f"Creating aria labels for {len(df)} files...")

    json_path = "E:/data/raw-midi-data/data/data/v2/aria-midi-v1-ext/aria-midi-v1-ext/metadata.json"
    with open(json_path, 'r') as f:
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

    df.to_csv(os.path.join(METADATA_PATH, "aria_labels.csv"), index=False, sep="\t")
    return df


def create_lengths():
    """Creates the lengths of the MIDI files in seconds."""
    import pretty_midi
    df = load_mapping()

    def midi_file_length(index: str) -> float:
        try:
            path = get_path(MIDI_ROOT, index)
            midi_data = pretty_midi.PrettyMIDI(path)
            length_in_seconds = midi_data.get_end_time()
            return length_in_seconds
        except Exception as e:
            return -1

    tqdm.pandas()

    df['length'] = df['index'].progress_map(midi_file_length)
    df.to_csv(os.path.join(METADATA_PATH, "lengths.csv"), index=False, sep="\t")
    return df


def create_note_counts():
    """Counts the number of notes in each MIDI file."""
    import pretty_midi

    df = load_mapping()

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
    df['note_count'] = df['index'].progress_apply(lambda x: count_notes_in_midi(get_path(MIDI_ROOT, x)))
    df.to_csv(os.path.join(METADATA_PATH, "note_counts.csv"), index=False, sep="\t")
    return df


def make_deepseek_labels():
    """Labels MIDI files using DeepSeek API. Does not create the key since deepseek results will benefit so much from some data cleaning"""
    from openai import OpenAI

    df = load_mapping()
    print(f"Creating deepseek labels for {len(df)} files...")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    dotenv.load_dotenv()

    DEEPSEEK_API = os.getenv("DEEPSEEK_API_KEY")
    assert DEEPSEEK_API is not None, "Please set the DEEPSEEK_API_KEY environment variable."

    file_lock = Lock()
    usage_lock = Lock()

    def is_discount_time():
        """
        Check if the current time is within the deepseek off peak hours (00:30 to 08:29).
        """
        current_time = datetime.now().time()
        return current_time >= datetime.strptime("00:30", "%H:%M").time() and current_time <= datetime.strptime("08:29", "%H:%M").time()

    def labelling_deepseek(key: str, save_path: str, usage_path: str, mapping: list[str], existing_labels: dict[str, list[str]]):
        if key in existing_labels:
            tqdm.write(f"Skipping {key}, already labelled.")
            return

        prompt = """You are a music expert. You are really good at inferring the music genre, style, mood, and other 
        characteristics of a piece of music based on some text clues and your vast knowledge of music.
        You will be given a list of words that describe a piece of music. You will try to tell me information about the music based on these words.
        The information could include genre, style, mood, instruments, composer, form, texture, key etc.
        Only give me the information that you are likely certain about.
        If the information is a BPM, you should prepend it with 'BPM: '.
        If the information is a key, you should prepend it with 'Key: '.
        You will return a list of comma-separated words that describe the music.

        For example, the input 'midi-classical-music/bach-bwv001-_400_chorales-008807b.mid' could be
        classical music, baroque, bach, chorale, polyphonic, choral

        The input 'XMIDI_Dataset/XMIDI_fear_pop_LTFY2KJS.midi' could be
        pop, fear

        The input 'Samples Depot - 999,000 Midi Pack Collection/Various Midi 370.000 Super Pack/Sounds To Sample/Sounds To Sample Deep House Melodics Vol 3/Lead & Addon MIDI/29 S2S DHM3 AddonMIDI 124 Dm.mid' could be
        deep house, melodic, addon, BPM: 124, D minor

        Some are probably straight up impossible to label. In that case: you should return the word 'unknown' as the only word in the list.
        For example, the input 'The_Magic_of_MIDI/MIDI/Midi.MID' might as well be impossible and you should return
        'unknown'.

        Give me as many words as you can, but ONLY insofar as you are certain about them.
        """

        client = OpenAI(api_key=DEEPSEEK_API, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "/".join(mapping)},
            ],
            stream=False,
            temperature=1.0
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            if "\n" in content:
                content = content.replace("\n", " -# ").strip()
            if "\t" in content:
                content = content.replace("\t", " -# ").strip()
            if content == "unknown":
                labels = ["unknown"]
            else:
                labels = [x.strip() for x in content.split(",") if x.strip()]
            labels = [x for x in labels if x]
            existing_labels[key] = labels
            with file_lock:
                with open(save_path, "a", encoding='utf-8') as f:
                    f.write(f"{key}\t{', '.join(labels)}\n")

        if response.usage is not None:
            hits = response.usage.prompt_cache_hit_tokens  # type: ignore
            misses = response.usage.prompt_cache_miss_tokens  # type: ignore
            completions = response.usage.completion_tokens
            money_one_billionth_usd = hits * 35 + misses * 35 + completions * 550
            if not is_discount_time():
                money_one_billionth_usd *= 2
            with usage_lock:
                with open(usage_path, "a", encoding='utf-8') as f:
                    f.write(f"{key}\t{hits}\t{misses}\t{completions}\t{money_one_billionth_usd}\n")

    save_path = os.path.join(".", "deepseek_labels.txt")
    usage_path = os.path.join(".", "deepseek_usage.txt")

    new_mappings = {}
    for k, v in zip(df["index"], df["original_path"]):
        new_mappings[k] = v.split("\\")[2:]

    existing_labels: dict[str, list[str]] = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    key, labels = line.strip().split("\t")
                except ValueError as e:
                    print(f"Error parsing line: {line.strip()}")
                    raise e
                existing_labels_key = labels.split(", ")
                existing_labels[key] = [label.strip() for label in existing_labels_key if label.strip()]

    if not os.path.exists(usage_path):
        with open(usage_path, "w") as f:
            f.write("key\tprompt_cache_hit_tokens\tprompt_cache_miss_tokens\tcompletion_tokens\tmoney(1e-9USD)\n")

    while not is_discount_time():
        print("Waiting for discount time (00:30 to 08:29)...")
        time.sleep(60)

    with ThreadPoolExecutor() as executor:
        futures = []
        for key, mapping in tqdm(new_mappings.items(), desc="Creating jobs with DeepSeek"):
            if key in existing_labels:
                continue
            futures.append(executor.submit(labelling_deepseek, key, save_path, usage_path, mapping, existing_labels))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DeepSeek responses"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing future: {e}")
                print(f"Traceback: {traceback.format_exc()}")

    print("Labelling complete.")


def create_mmd_genre():
    """Create a mapping of MetaMidi Dataset MIDI files to their genres."""
    import pandas as pd
    df = load_mapping()

    def apply(path):
        is_mmd = "v2/MMD_MIDI" in path
        if not is_mmd:
            return ""
        return os.path.splitext(os.path.basename(path))[0]

    df['mmd_hash'] = df['original_path'].apply(apply)

    # JSONL available at: https://zenodo.org/records/5142664#.YQN3c5NKgWo
    genres_path = f"E:/data/raw-midi-data/data/data/v2/MMD_MIDI/MMD_MIDI/MMD_scraped_genre.jsonl"

    def extract_genre(genre):
        def inner(genre):
            if isinstance(genre, list):
                for g in genre:
                    yield from inner(g)
            else:
                yield genre
        return list(inner(genre))

    import json
    with open(genres_path, 'r', encoding='utf-8') as f:
        genres = [json.loads(line) for line in f]

    genres_df = pd.DataFrame(genres)

    genres_df['genre'] = genres_df['genre'].apply(extract_genre)

    def map_genre(genre):
        new_genre = []
        mapping = {
            "traditional_(folk)": "traditional folk",
            "early_20th_century": "early 20th century",
            "easylistening": "easy listening",
            "italian%2cfrench%2cspanish": "italian french spanish",
            "musical%2cfilm%2ctv": "musical film tv",
        }
        for g in genre:
            if g in ["unknown era", "unconfirmed category"]:
                continue
            if g in mapping:
                new_genre.append(mapping[g])
            else:
                new_genre.append(g)
        return new_genre

    genres_df['genre'] = genres_df['genre'].apply(map_genre)

    import numpy as np
    from collections import defaultdict

    genre_counts = defaultdict(int)

    def count(genre):
        for g in genre:
            genre_counts[g] += 1

    genres_df['genre'].apply(count)
    genres_df['genre'] = genres_df['genre'].apply(", ".join)

    df = df.merge(genres_df, how='left', left_on='mmd_hash', right_on='md5')
    df.drop(['original_path', 'md5', 'mmd_hash'], axis=1, inplace=True)
    df = df[df['genre'].notna()]

    df.to_csv(os.path.join(METADATA_PATH, "mmd_genres.csv"), index=False, sep="\t")
    print(f"Created MMD genres mapping with {len(df)} entries.")
    return df


if __name__ == "__main__":
    create_xml_note_counts()
