import logging
import traceback
from threading import Lock
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
from datetime import datetime
from openai import OpenAI
import os
import time
import pandas as pd

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


def main(root: str):
    df = pd.read_csv(os.path.join(root, "mapping.csv"), sep='\t')

    save_path = os.path.join(root, "deepseek_labels.txt")
    usage_path = os.path.join(root, "deepseek_usage.txt")

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
    # response = labelling_deepseek("testdata", ["midi-classical-music", "bach-bwv001-_400_chorales-008807b.mid"])
    print("Labelling complete.")


if __name__ == "__main__":
    main("E:/giant-midi-archive")
