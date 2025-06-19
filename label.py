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
from src import GiantMidiDataset

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
dotenv.load_dotenv()

DEEPSEEK_API = os.getenv("DEEPSEEK_API_KEY")
assert DEEPSEEK_API is not None, "Please set the DEEPSEEK_API_KEY environment variable."

SAVE_PATH = "./giant-midi-archive/deepseek_labels_2.txt"
USAGE_PATH = "./giant-midi-archive/deepseek_usage_2.txt"
file_lock = Lock()
usage_lock = Lock()


def is_discount_time():
    """
    Check if the current time is within the deepseek off peak hours (00:30 to 08:29).
    """
    current_time = datetime.now().time()
    return current_time >= datetime.strptime("00:30", "%H:%M").time() and current_time <= datetime.strptime("08:29", "%H:%M").time()


def labelling_deepseek(key: str, mapping: list[str], existing_labels: dict[str, list[str]]):
    if key in existing_labels:
        tqdm.write(f"Skipping {key}, already labelled.")
        return

    if not is_discount_time():
        tqdm.write(f"Skipping {key}, not within the allowed time range. We poor D:")
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
        if content == "unknown":
            labels = ["unknown"]
        else:
            labels = [x.strip() for x in content.split(",") if x.strip()]
        labels = [x for x in labels if x]
        existing_labels[key] = labels
        with file_lock:
            with open(SAVE_PATH, "a", encoding='utf-8') as f:
                f.write(f"{key}\t{', '.join(labels)}\n")

    if response.usage is not None:
        hits = response.usage.prompt_cache_hit_tokens  # type: ignore
        misses = response.usage.prompt_cache_miss_tokens  # type: ignore
        completions = response.usage.completion_tokens
        money_one_billionth_usd = hits * 35 + misses * 35 + completions * 550
        with usage_lock:
            with open(USAGE_PATH, "a", encoding='utf-8') as f:
                f.write(f"{key}\t{hits}\t{misses}\t{completions}\t{money_one_billionth_usd}\n")


def main():
    ds = GiantMidiDataset.load()
    mappings = ds.lookup_info("mapping")

    new_mappings = {}
    for k, v in mappings.items():
        new_mappings[k] = v.split("\\")[2:]

    existing_labels: dict[str, list[str]] = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            for line in f:
                key, labels = line.strip().split("\t")
                existing_labels_key = labels.split(", ")
                existing_labels[key] = [label.strip() for label in existing_labels_key if label.strip()]

    if not os.path.exists(USAGE_PATH):
        with open(USAGE_PATH, "w") as f:
            f.write("key\tprompt_cache_hit_tokens\tprompt_cache_miss_tokens\tcompletion_tokens\tmoney(1e-9USD)\n")

    # Wait here
    waiting_tqdm = tqdm(desc="Waiting for discount time to start...", unit="s")
    while not is_discount_time():
        time.sleep(1)
        waiting_tqdm.update(1)
    waiting_tqdm.close()

    with ThreadPoolExecutor() as executor:
        futures = []
        for key, mapping in tqdm(new_mappings.items(), desc="Creating jobs with DeepSeek"):
            if key in existing_labels:
                continue
            # Split half the job to another file
            if key[0] not in "qwertyuiopasd":
                continue
            futures.append(executor.submit(labelling_deepseek, key, mapping, existing_labels))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DeepSeek responses"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing future: {e}")
                print(f"Traceback: {traceback.format_exc()}")
    # response = labelling_deepseek("testdata", ["midi-classical-music", "bach-bwv001-_400_chorales-008807b.mid"])
    print("Labelling complete.")


if __name__ == "__main__":
    main()
