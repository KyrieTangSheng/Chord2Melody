from torch.utils.data import Dataset
import pickle
import json
import torch
import numpy as np

class ChordMelodyDataset(Dataset):
    def __init__(self, data_path: str, vocab_path: str, max_melody_length: int = 32, max_chord_length: int = 100):
        with open(data_path, 'rb') as f:
            self.training_sequences = pickle.load(f)
        with open(vocab_path, 'r') as f:
            self.vocabularies = json.load(f)
            
        self.chord_to_idx = self.vocabularies['chord_vocab']
        self.note_to_idx = self.vocabularies['note_vocab']
        self.max_melody_length = max_melody_length
        self.max_chord_length = max_chord_length
        
        # Filter sequences that are too long
        self.training_sequences = [seq for seq in self.training_sequences 
                                 if len(seq['output_melody']) <= max_melody_length 
                                 and len(seq['full_chord_sequence']) <= max_chord_length]
        
        print(f"Loaded {len(self.training_sequences)} training sequences")
        if self.training_sequences:
            avg_chord_len = np.mean([len(seq['full_chord_sequence']) for seq in self.training_sequences])
            avg_melody_len = np.mean([len(seq['output_melody']) for seq in self.training_sequences])
            print(f"Average chord sequence length: {avg_chord_len:.1f}")
            print(f"Average melody length: {avg_melody_len:.1f}")

    def __len__(self):
        return len(self.training_sequences)
    
    def __getitem__(self, idx):
        sequence = self.training_sequences[idx]
        
        # Full chord sequence encoding
        full_chords = []
        for chord in sequence['full_chord_sequence']:
            full_chords.append(self.chord_to_idx.get(chord, self.chord_to_idx['<UNK>']))

        # Pad or truncate chord sequence
        original_chord_length = len(full_chords)
        if len(full_chords) < self.max_chord_length:
            full_chords.extend([self.chord_to_idx['<PAD>']] * (self.max_chord_length - len(full_chords)))
        else:
            full_chords = full_chords[:self.max_chord_length]
        
        # Focus position (clamped to valid range)
        focus_position = min(sequence['focus_position'], original_chord_length - 1, self.max_chord_length - 1)
        
        # Melody encoding
        melody_pitches, melody_durations, melody_starts = [], [], []
        for note in sequence['output_melody']:
            melody_pitches.append(note['pitch'])
            melody_durations.append(min(note['duration'], 4.0))  # Cap duration
            melody_starts.append(min(note['start'], 16.0))       # Cap start time

        # Pad melody sequences
        actual_melody_length = len(melody_pitches)
        while len(melody_pitches) < self.max_melody_length:
            melody_pitches.append(0)  # Use 0 for padding (rest)
            melody_durations.append(0.0)
            melody_starts.append(0.0)
        
        # Truncate if too long
        melody_pitches = melody_pitches[:self.max_melody_length]
        melody_durations = melody_durations[:self.max_melody_length]
        melody_starts = melody_starts[:self.max_melody_length]
        
        # Create masks
        chord_mask = [1 if i < original_chord_length else 0 for i in range(self.max_chord_length)]
        melody_mask = [1 if i < actual_melody_length else 0 for i in range(self.max_melody_length)]
        
        return {
            'full_chord_sequence': torch.tensor(full_chords, dtype=torch.long),
            'chord_mask': torch.tensor(chord_mask, dtype=torch.bool),
            'focus_position': torch.tensor(focus_position, dtype=torch.long),
            'melody_pitch': torch.tensor(melody_pitches, dtype=torch.long),
            'melody_duration': torch.tensor(melody_durations, dtype=torch.float32),
            'melody_start': torch.tensor(melody_starts, dtype=torch.float32),
            'melody_mask': torch.tensor(melody_mask, dtype=torch.bool),
            'segment_position': torch.tensor(sequence.get('segment_position', 0.0), dtype=torch.float32),
            'chord_length': torch.tensor(original_chord_length, dtype=torch.long),
            'melody_length': torch.tensor(actual_melody_length, dtype=torch.long),
            'song_id': sequence['song_id']
        }

if __name__ == "__main__":
    data_path = "processed_pop909_chord_melody/training_sequences.pkl"
    vocab_path = "processed_pop909_chord_melody/vocabularies.json"
    
    dataset = ChordMelodyDataset(data_path, vocab_path)
    sample = dataset[0]
    print("Sample data shapes:")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.shape} - {value.dtype}")
        else:
            print(f"{key}: {value}")