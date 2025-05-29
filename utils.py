from typing import List, Dict
import pretty_midi

def generate_midi_from_melody(melody:List[Dict], output_path:str, tempo:int=120):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=1) # this is piano
    
    current_time = 0.0
    for note_info in melody:
        note = pretty_midi.Note(
            velocity=80,
            pitch=int(note_info['pitch']),
            start=current_time + note_info['start_time'],
            end=current_time + note_info['start_time'] + note_info['duration']
        )
        instrument.notes.append(note)
        current_time += note_info['duration']
    
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"Generated melody saved to {output_path}")
    
     