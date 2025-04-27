import os
import mido
import json
import typing
from src import GiantMidiDataset
from notes import Note, RealTimeNotes, NotatedTimeNotes
T = typing.TypeVar("T")


def _get_notes(path: str) -> RealTimeNotes:
    mid = mido.MidiFile(path)
    tempo = 500000  # Default tempo (500,000 microseconds per beat)
    ticks_per_beat = mid.ticks_per_beat

    tempo_changes = [(0, tempo)]
    events = []

    # Convert delta times to absolute times and collect all events
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_changes.append((current_tick, msg.tempo))
            if msg.type in ['note_on', 'note_off']:
                events.append((current_tick, msg.note, msg.type, msg.velocity))

    tempo_changes.sort()
    events.sort()
    notes: list[Note] = []
    current_time = 0
    last_tick = 0
    note_on_dict: dict[int, tuple[float, int]] = {}

    # Convert ticks in events to real time using the global tempo map
    for event in events:
        tick, note, tp, velocity = event

        while tempo_changes and tempo_changes[0][0] <= tick:
            tempo_change_tick, new_tempo = tempo_changes.pop(0)
            if tempo_change_tick > last_tick:
                current_time += mido.tick2second(tempo_change_tick - last_tick, ticks_per_beat, tempo)
                last_tick = tempo_change_tick
            tempo = new_tempo

            # Update current time up to the event tick
        if tick > last_tick:
            current_time += mido.tick2second(tick - last_tick, ticks_per_beat, tempo)
            last_tick = tick

        if tp == 'note_on' and velocity > 0:
            note_on_dict[note] = (current_time, velocity)
        elif (tp == 'note_off' or (tp == 'note_on' and velocity == 0)) and note in note_on_dict:
            start_time, velocity = note_on_dict.pop(note)
            duration = current_time - start_time
            note = Note.from_midi_number(midi_number=note, duration=duration, offset=start_time, velocity=velocity, real_time=True)
            notes.append(note)

    return RealTimeNotes(notes)


def instruments(midi_path: str) -> list[int]:
    try:
        mid = mido.MidiFile(midi_path)
        instrument_set = []

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    instrument_name = msg.program
                    instrument_set.append(instrument_name)
        return instrument_set
    except Exception as e:
        return []


def length(midi_path: str) -> float:
    try:
        notes = _get_notes(midi_path)
        if len(notes._notes) == 0:
            return -1
        return notes[-1].offset + notes[-1].duration
    except Exception as e:
        return -1


def analyse(f: typing.Callable[[str], T], name: str | None = None) -> None:
    """Analyse the dataset with a function and save the result to a json file.

    The name of the json file will be the name of the function by default"""
    ds = GiantMidiDataset("./giant-midi-archive")

    if name is None:
        name = f.__name__
    print(f"Analysing {len(ds)} files with {name}")

    path = f"./giant-midi-archive/{name}.json"
    if os.path.exists(path):
        print(f"File {path} already exists, skipping")
        return

    result = ds.accumulate(
        f,
        num_threads=16,
    )
    # Save result to a json file
    with open(path, "w") as file:
        json.dump(result, file)
    print(f"Saved result to {name}.json")


if __name__ == "__main__":
    analyse(length)
