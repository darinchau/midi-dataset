import numpy as np
import random
import mido
from tqdm import tqdm
from src import GiantMidiDataset
from midikit2 import Midifile
from mido import MidiFile as MidoFile


def canonical_filter(ds: GiantMidiDataset, key: str):
    max_delta = ds.lookup_info_idx("delta_time", key)
    if max_delta >= 10:
        return False
    if max_delta < 0:
        return False
    length = ds.lookup_info_idx("length", key)
    if length > 3600:
        return False
    if length < 1:
        return False
    try:
        midifile = Midifile(ds.get_path(key))
    except Exception as e:
        print(f"Error processing {key}: {e}")
        return False
    return True


def yield_midi_note_info(midi_path):
    mid = MidoFile(midi_path)
    current_instruments = [0] * 16  # Default instruments (0: Acoustic Grand Piano)
    tempo = 500000  # Default tempo (500,000 microseconds per beat)
    time_per_tick = tempo / mid.ticks_per_beat

    for track in mid.tracks:
        time_since_last_event = 0
        for msg in track:
            time_since_last_event += msg.time
            seconds_since_last_event = time_since_last_event * time_per_tick / 1_000_000

            if msg.type == 'program_change':
                current_instruments[msg.channel] = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                # Note on message (note, velocity > 0)
                note_number = msg.note
                instrument = current_instruments[msg.channel]
                duration_seconds = seconds_since_last_event
                yield (note_number, instrument, duration_seconds)
            elif msg.type == 'set_tempo':
                tempo = msg.tempo
                time_per_tick = tempo / mid.ticks_per_beat


def main():
    ds = GiantMidiDataset("./giant-midi-archive")
    ds.add_filter(canonical_filter)
    hotness_array = np.zeros((128, 128), dtype=np.float32)
    files = ds.get_all_paths()
    random.shuffle(files)
    for path in tqdm(files, desc="Processing MIDI files"):
        index = path.stem
        try:
            if ds.is_outlier(index):
                continue
            local_hotness_array = np.zeros((128, 128), dtype=np.float32)
            for i, j, t in yield_midi_note_info(path):
                local_hotness_array[i, j] += t
            hotness_array += local_hotness_array
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # save the hotness array to a file
    np.save("hotness_array.npy", hotness_array)


if __name__ == "__main__":
    main()
    print("Hotness array saved to hotness_array.npy")
