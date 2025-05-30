from torch.utils.data import Dataset
import pickle
import json
import torch
import numpy as np

class ChordMelodyDataset(Dataset):
    def __init__(self, data_path: str, vocab_path: str, max_melody_length: int = 16, max_chord_length: int = 12):
        with open(data_path, 'rb') as f:
            self.training_sequences = pickle.load(f)
        with open(vocab_path, 'r') as f:
            self.vocabularies = json.load(f)
            
        self.chord_to_idx = self.vocabularies['chord_vocab']
        self.note_to_idx = self.vocabularies['note_vocab']
        self.max_melody_length = max_melody_length
        self.max_chord_length = max_chord_length
        
        # Filter sequences
        self.training_sequences = [seq for seq in self.training_sequences 
                                 if len(seq['output_melody']) <= max_melody_length 
                                 and len(seq['full_chord_sequence']) <= max_chord_length]
        
        print(f"Loaded {len(self.training_sequences)} chord-aligned training sequences")
        if self.training_sequences:
            avg_chord_len = np.mean([len(seq['full_chord_sequence']) for seq in self.training_sequences])
            avg_melody_len = np.mean([len(seq['output_melody']) for seq in self.training_sequences])
            print(f"Average chord sequence length: {avg_chord_len:.1f}")
            print(f"Average melody length: {avg_melody_len:.1f}")

    def __len__(self):
        return len(self.training_sequences)
    
    def __getitem__(self, idx):
        sequence = self.training_sequences[idx]
        
        # Encode chord sequence
        chord_indices = []
        for chord in sequence['full_chord_sequence']:
            chord_indices.append(self.chord_to_idx.get(chord, self.chord_to_idx['<UNK>']))
        
        # Encode chord durations (normalized)
        chord_durations = sequence['chord_durations']
        max_duration = max(chord_durations) if chord_durations else 4.0
        normalized_durations = [d / max_duration for d in chord_durations]
        
        # Pad sequences
        original_chord_length = len(chord_indices)
        while len(chord_indices) < self.max_chord_length:
            chord_indices.append(self.chord_to_idx['<PAD>'])
            normalized_durations.append(0.0)
        
        # Focus position (which chord we're generating melody for)
        focus_position = min(sequence['focus_position'], original_chord_length - 1)
        
        # Target chord duration
        target_chord_duration = sequence['chord_duration']
        
        # Melody encoding - notes relative to target chord timing
        melody_pitches, melody_starts, melody_durations = [], [], []
        for note in sequence['output_melody']:
            melody_pitches.append(note['pitch'])
            # Normalize timing to chord duration
            melody_starts.append(note['start_time'] / target_chord_duration)
            melody_durations.append(note['duration'] / target_chord_duration)
        
        # Pad melody
        actual_melody_length = len(melody_pitches)
        while len(melody_pitches) < self.max_melody_length:
            melody_pitches.append(0)
            melody_starts.append(0.0)
            melody_durations.append(0.0)
        
        # Create masks
        chord_mask = [1 if i < original_chord_length else 0 for i in range(self.max_chord_length)]
        melody_mask = [1 if i < actual_melody_length else 0 for i in range(self.max_melody_length)]
        
        return {
            'full_chord_sequence': torch.tensor(chord_indices, dtype=torch.long),
            'chord_durations': torch.tensor(normalized_durations, dtype=torch.float32),
            'chord_mask': torch.tensor(chord_mask, dtype=torch.bool),
            'focus_position': torch.tensor(focus_position, dtype=torch.long),
            'target_chord_duration': torch.tensor(target_chord_duration, dtype=torch.float32),
            'melody_pitch': torch.tensor(melody_pitches, dtype=torch.long),
            'melody_start': torch.tensor(melody_starts, dtype=torch.float32),
            'melody_duration': torch.tensor(melody_durations, dtype=torch.float32),
            'melody_mask': torch.tensor(melody_mask, dtype=torch.bool),
            'chord_length': torch.tensor(original_chord_length, dtype=torch.long),
            'melody_length': torch.tensor(actual_melody_length, dtype=torch.long),
            'song_id': sequence['song_id']
        }