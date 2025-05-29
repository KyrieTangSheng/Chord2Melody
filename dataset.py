from torch.utils.data import Dataset
import pickle
import json
import torch
class ChordMelodyDataset(Dataset):
    def __init__(self, data_path:str, vocab_path:str, max_melody_length:int=32, max_chord_input_length:int=10):
        with open(data_path, 'rb') as f:
            self.training_sequences = pickle.load(f)
        with open(vocab_path, 'r') as f:
            self.vocabularies = json.load(f)
            
        self.chord_to_idx = self.vocabularies['chord_vocab']
        self.note_to_idx = self.vocabularies['note_vocab']
        self.max_melody_length = max_melody_length
        self.max_chord_input_length = max_chord_input_length
        # filter out sequences that are too long
        self.training_sequences = [seq for seq in self.training_sequences if len(seq['output_melody']) <= max_melody_length]
        
        print(f"Loaded {len(self.training_sequences)} training sequences")

    def __len__(self):
        return len(self.training_sequences)
    
    def __getitem__(self, idx):
        sequence = self.training_sequences[idx]
        
        # chord progression encoding
        chord_input = []
        for chord in sequence['input_chords']:
            chord_input.append(self.chord_to_idx.get(chord, self.chord_to_idx['<UNK>']))

        # shouldn't happen. But just in case.
        while len(chord_input) < self.max_chord_input_length:
            chord_input.append(self.chord_to_idx['<PAD>'])  
        
        # melody encoding
        melody_pitches, melody_durations, melody_starts = [], [], []
        for note in sequence['output_melody']:
            melody_pitches.append(note['pitch'])
            melody_durations.append(min(note['duration'], 4)) # ?? why 4?
            melody_starts.append(min(note['start'], 16))    # ?? why 16?

        while len(melody_pitches) < self.max_melody_length:
            melody_pitches.append(0) 
            melody_durations.append(0.0)
            melody_starts.append(0.0)
        
        melody_pitches = melody_pitches[:self.max_melody_length]
        melody_durations = melody_durations[:self.max_melody_length]
        melody_starts = melody_starts[:self.max_melody_length]
        
        return {
            'chord_input': torch.tensor(chord_input, dtype=torch.long),
            'melody_pitch': torch.tensor(melody_pitches, dtype=torch.long),
            'melody_duration': torch.tensor(melody_durations, dtype=torch.float32),
            'melody_start': torch.tensor(melody_starts, dtype=torch.float32),
            'sequence_length': len(sequence['output_melody']),
            'song_id': sequence['song_id']
        }
        

if __name__ == "__main__":
    data_path = "processed_pop909_chord_melody/training_sequences.pkl"
    vocab_path = "processed_pop909_chord_melody/vocabularies.json"
    
    dataset = ChordMelodyDataset(data_path, vocab_path)
    print(dataset[200])