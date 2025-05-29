import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pretty_midi
import numpy as np
import json
import pickle

class POP909DataProcessor:
    """
    This is the data processor for the POP909 dataset.
    This processor is going to
    (1) extracts melody from the midi file
    (2) parse the chord annotations from the txt files
    (3) aligns chord with melody segments
    (4) create input/output pairs for training
    """
    def __init__(self, data_path:str, output_path:str, chord_window_size:int=8):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Musical parameters NOTE: not sure yet if these parameters are correct
        self.time_resolution = 0.125  # 1/8 note resolution (in seconds)
        self.min_chord_duration = 0.5  # Minimum chord duration to consider
        self.min_melody_segment = 2.0  # Minimum melody segment length
        
        self.chord_vocab = set()
        self.note_vocab = set()
        
        self.processed_data = []
        self.chord_window_size = chord_window_size
        
    def parse_chord_file(self, chord_file_path:str) -> List[Tuple[float, float, str]]:
        chords = []
        
        try:
            with open(chord_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        start_time, end_time, chord_symbol = parts[:3]
                        if (float(end_time)-float(start_time)) >= self.min_chord_duration and chord_symbol != "N":
                            chords.append((float(start_time), float(end_time), chord_symbol))
                            self.chord_vocab.add(chord_symbol)
        except Exception as e:
            error_msg = f"Error parsing chord file {chord_file_path}: {e}"
            print(error_msg)
        
        return chords
    
    def extract_melody_track(self, midi_file_path:str) -> Optional[pretty_midi.Instrument]:
        """ 
        NOTE: here melody track is the first track of the midi file (according my observation of the dataset)
        """
        midi_file_path = str(midi_file_path)
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            for instrument in midi_data.instruments:
                if instrument.name and "MELODY" in instrument.name.upper():
                    return instrument
            
            if len(midi_data.instruments) > 0:
                return midi_data.instruments[0]
            
        except Exception as e:
            print(f"Error loading MIDI file {midi_file_path}: {e}")
            
        return None

    def quantize_melody(self, melody_track:pretty_midi.Instrument, song_duration:float) -> List[Dict]:
        """
        quantize melody to fixed time grid and extract features
        """
        quantized_melody = []
        time_steps = np.arange(0, song_duration, self.time_resolution)
        
        for note in melody_track.notes:
            start_step = int(np.round(note.start / self.time_resolution))
            end_step = int(np.round(note.end / self.time_resolution))
            
            if start_step >= len(time_steps) or end_step > len(time_steps):
                continue
                
            note_info = {
                'start_time': time_steps[start_step],
                'end_time': time_steps[end_step-1],
                'pitch': note.pitch,
                'velocity': note.velocity,
                'duration': note.end - note.start
            }
            quantized_melody.append(note_info)
            self.note_vocab.add(note.pitch)
            
            
        return quantized_melody
    
    def align_chords_melody(self, chords:List[Tuple[float, float, str]], melody:List[Dict]) -> List[Dict]:
        aligned_pairs = []
        
        for i, chord in enumerate(chords):
            if i > len(chords) - self.chord_window_size:
                break
            
            chord_melody = []
            current_chords = []
            j = 0
            while j < self.chord_window_size:
                chord_start, chord_end, chord_symbol = chords[i+j]
                current_chords.append((chord_start, chord_end, chord_symbol))
                j += 1
            
            for note in melody:
                if (note['start_time'] < current_chords[-1][1] and note['end_time'] > current_chords[0][0]):
                    relative_note = note.copy()
                    relative_note['relative_start'] = max(0, note['start_time'] - current_chords[0][0])
                    relative_note['relative_end'] = min(current_chords[-1][1] - current_chords[0][0], 
                                                       note['end_time'] - current_chords[0][0])
                    chord_melody.append(relative_note)
            
            if chord_melody:
                chord_context = []
                
                # previous chord
                if i > 0:
                    chord_context.append(chords[i-1][2])
                else:
                    chord_context.append('<START>')
                #current chords
                for chord in current_chords:
                    chord_context.append(chord[2])
                #next chord
                if i + self.chord_window_size < len(chords) - 1:
                    chord_context.append(chords[i+self.chord_window_size][2])
                else:
                    chord_context.append('<END>')
                

                pair = {
                    'chord_context': chord_context,
                    'chord_duration': current_chords[-1][1] - current_chords[0][0],
                    'melody_notes': chord_melody,
                    'chord_start_time': current_chords[0][0],
                    'chord_end_time': current_chords[-1][1]
                }
                aligned_pairs.append(pair)
        return aligned_pairs
            
    # def align_chords_melody(self, chords:List[Tuple[float, float, str]], melody:List[Dict]) -> List[Dict]:
    #     """
    #     Create chord-melody alignment pairs for training.
    #     Each pair contains a chord progression segment and corresponding melody.
    #     """
        
    #     aligned_pairs = []
    #     for i, chord in enumerate(chords):
    #         chord_start, chord_end, chord_symbol = chord
    #         chord_melody = []
            
    #         for note in melody:
    #             if (note['start_time'] < chord_end and note['end_time'] > chord_start):
    #                 relative_note = note.copy()
    #                 relative_note['relative_start'] = max(0, note['start_time'] - chord_start)
    #                 relative_note['relative_end'] = min(chord_end - chord_start, 
    #                                                   note['end_time'] - chord_start)
    #                 chord_melody.append(relative_note)
            
    #         if chord_melody:
    #             chord_context = []
                
    #             # previous chord
    #             if i > 0:
    #                 chord_context.append(chords[i-1][2])
    #             else:
    #                 chord_context.append('<START>')
                    
    #             #current chord
    #             chord_context.append(chord_symbol)
                
    #             #next chord
    #             if i < len(chords) - 1:
    #                 chord_context.append(chords[i+1][2])
    #             else:
    #                 chord_context.append('<END>')
                
    #             pair = {
    #                 'chord_context': chord_context,
    #                 'chord_duration': chord_end - chord_start,
    #                 'melody_notes': chord_melody,
    #                 'chord_start_time': chord_start,
    #                 'chord_end_time': chord_end
    #             }
    #             aligned_pairs.append(pair)
            
    #     return aligned_pairs
    
    def normalize_chord_symbol(self, chord_symbol:str) -> str:
        """
        Return the original chord symbol without normalization.
        Only handle empty/None cases.
        """
        # Handle no-chord cases
        if chord_symbol == 'N' or not chord_symbol or chord_symbol.strip() == '':
            return 'N'
        
        return chord_symbol.strip()

    def process_song(self, song_folder:Path) -> None:
        """
        Process a single song from the dataset.
        """
        song_id = song_folder.name
        midi_file_path = song_folder / f"{song_id}.mid"
        chord_file_path = song_folder / "chord_audio.txt"
        
        if not midi_file_path.exists() or not chord_file_path.exists():
            print(f"Skipping {song_id} due to missing files")
            return None
        
        chords = self.parse_chord_file(chord_file_path)
        if not chords:
            print(f"No valid chords found for {song_id}")
            return None
        
        melody_track = self.extract_melody_track(midi_file_path)
        if not melody_track:
            print(f"No melody track found for {song_id}")
            return None
        
        song_duration = max(chords[-1][1], melody_track.notes[-1].end)
        
        quantized_melody = self.quantize_melody(melody_track, song_duration)
        if not quantized_melody:
            print(f"No quantized melody for song {song_id}")
            return None
        
        normalized_chords = [(start, end, self.normalize_chord_symbol(chord)) 
                           for start, end, chord in chords]
        
        aligned_pairs = self.align_chords_melody(normalized_chords, quantized_melody)
        if not aligned_pairs:
            print(f"No aligned pairs for song {song_id}")
            return None
        
        return {
            'song_id': song_id,
            'chord_melody_pairs': aligned_pairs,
            'total_duration': song_duration,
            'num_chords': len(chords),
            'num_melody_notes': len(quantized_melody)
        }
    
    def process_dataset(self) -> None:
        """
        Process all songs in the dataset and save the aligned pairs.
        """
        pop909_folder = self.data_path / "POP909"
        if not pop909_folder.exists():
            print(f"POP909 folder not found at {pop909_folder}")
            return
        
        song_folders = [f for f in pop909_folder.iterdir() if f.is_dir()]
        print(f"Found {len(song_folders)} song folders in POP909 dataset")
        
        successful_songs = 0
        for song_folder in sorted(song_folders):
            song_data = self.process_song(song_folder)
            if song_data:
                self.processed_data.append(song_data)
                successful_songs += 1
                
                if successful_songs % 50 == 0:
                    print(f"Processed {successful_songs}/{len(song_folders)} songs")
                    
        print(f"Completed processing. {successful_songs}/{len(song_folders)} songs processed successfully")

    def create_vocabularies(self) -> None:
        chord_vocab_list = ['<PAD>', '<START>', '<END>', '<UNK>'] + sorted(list(self.chord_vocab))
        note_vocab_list = list(range(0, 128))
        
        vocab_data = {
            'chord_vocab': {chord: idx for idx, chord in enumerate(chord_vocab_list)},
            'note_vocab': {note: idx for idx, note in enumerate(note_vocab_list)},
            'idx_to_chord': {idx: chord for idx, chord in enumerate(chord_vocab_list)},
            'idx_to_note': {idx: note for idx, note in enumerate(note_vocab_list)}
        }
        
        with open(self.output_path / "vocabularies.json", 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Chord vocabulary size: {len(chord_vocab_list)}")
        print(f"Note vocabulary size: {len(note_vocab_list)}")
        
        return vocab_data
    
    def save_processed_data(self) -> None:
        with open(self.output_path / "chord_melody_data.pkl", 'wb') as f:
            pickle.dump(self.processed_data, f)
            
        with open(self.output_path / "chord_melody_data.json", 'w') as f:
            json.dump(self.processed_data, f, indent=2)
            
        total_pairs = sum(len(song['chord_melody_pairs']) for song in self.processed_data)
        avg_pairs_per_song = total_pairs / len(self.processed_data) if self.processed_data else 0
        
        stats = {
            'total_songs': len(self.processed_data),
            'total_chord_melody_pairs': total_pairs,
            'avg_pairs_per_song': avg_pairs_per_song,
            'unique_chords': len(self.chord_vocab),
            'note_range': [min(self.note_vocab), max(self.note_vocab)] if self.note_vocab else [0, 127]
        }
        
        with open(self.output_path / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Processed data saved to {self.output_path}")
        print(f"Dataset statistics: {stats}")

    def create_training_sequences(self, max_seq_length:int=32) -> None:
        training_sequences = []
        
        for song in self.processed_data:
            for pair in song['chord_melody_pairs']:
                chord_input = pair['chord_context']
                
                melody_output = []
                for note in pair['melody_notes']:
                    melody_output.append({
                        'pitch': note['pitch'],
                        'start': note['relative_start'],
                        'duration': note['relative_end'] - note['relative_start']
                    })
                
                if len(melody_output) <= max_seq_length:
                    training_sequences.append({
                        'input_chords': chord_input,
                        'output_melody': melody_output,
                        'song_id': song['song_id']
                    })
                else:
                    break
        
        with open(self.output_path / "training_sequences.pkl", 'wb') as f:
            pickle.dump(training_sequences, f)
        
        print(f"Created {len(training_sequences)} training sequences")
        return training_sequences

def process_dataset(dataset_path:str, output_path:str):
    processor = POP909DataProcessor(dataset_path, output_path)
    processor.process_dataset()
    processor.create_vocabularies()
    processor.save_processed_data()
    processor.create_training_sequences()

if __name__ == "__main__":
    dataset_path = "POP909-Dataset"
    output_path = "processed_pop909_chord_melody"
    process_dataset(dataset_path, output_path)
                
                
                    
        
        
        
            
            
            
                
        
                
