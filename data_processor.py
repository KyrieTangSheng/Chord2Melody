import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pretty_midi
import numpy as np
import json
import pickle

class POP909DataProcessor:
    """
    Data processor for attention-based global context approach.
    This processor stores full chord sequences and creates training pairs
    with focus positions for attention mechanism.
    """
    def __init__(self, data_path: str, output_path: str, melody_segment_length: int = 32):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.time_resolution = 0.125  
        self.min_chord_duration = 0.5 
        self.melody_segment_length = melody_segment_length 
        
        self.chord_vocab = set()
        self.note_vocab = set()
        
        self.processed_data = []
        
    def parse_chord_file(self, chord_file_path: str) -> List[Tuple[float, float, str]]:
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
                        if (float(end_time) - float(start_time)) >= self.min_chord_duration and chord_symbol != "N":
                            simplified_chord = self.normalize_chord_symbol(chord_symbol)
                            chords.append((float(start_time), float(end_time), simplified_chord))
                            self.chord_vocab.add(simplified_chord)
        except Exception as e:
            error_msg = f"Error parsing chord file {chord_file_path}: {e}"
            print(error_msg)
        
        return chords
    
    def extract_melody_track(self, midi_file_path: str) -> Optional[pretty_midi.Instrument]:
        """ 
        Extract melody track from MIDI file
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
    
    def quantize_melody(self, melody_track: pretty_midi.Instrument, song_duration: float) -> List[Dict]:
        """
        Quantize melody to fixed time grid and extract features
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
                'end_time': time_steps[end_step-1] if end_step > 0 else time_steps[start_step],
                'pitch': note.pitch,
                'velocity': note.velocity,
                'duration': note.end - note.start
            }
            quantized_melody.append(note_info)
            self.note_vocab.add(note.pitch)
            
        return quantized_melody
    
    def normalize_chord_symbol(self, chord_symbol: str) -> str:
        """
        Normalize and simplify chord symbols
        """
        if chord_symbol == 'N' or not chord_symbol or chord_symbol.strip() == '':
            return 'N'
        
        cleaned = chord_symbol.strip()
        simplified = self.simplify_chord_symbol(cleaned)
        return simplified

    def process_song(self, song_folder: Path) -> Optional[Dict]:
        """
        Process a single song with simplified chord vocabulary
        """
        song_id = song_folder.name
        midi_file_path = song_folder / f"{song_id}.mid"
        chord_file_path = song_folder / "chord_audio.txt"
        
        if not midi_file_path.exists() or not chord_file_path.exists():
            print(f"Skipping {song_id} due to missing files")
            return None
        
        chords = self.parse_chord_file(chord_file_path)
        if len(chords) < 4:
            print(f"Skipping {song_id} - not enough chords ({len(chords)})")
            return None
        
        combined_track = self.extract_melody_track(midi_file_path)
        
        song_duration = max(chords[-1][1], combined_track.notes[-1].end)
        
        quantized_melody = self.quantize_melody(combined_track, song_duration)
        if not quantized_melody:
            print(f"No quantized melody for song {song_id}")
            return None
        
        melody_segments = self.create_chord_aligned_segments(chords, quantized_melody)
        if not melody_segments:
            print(f"No melody segments for song {song_id}")
            return None
        
        return {
            'song_id': song_id,
            'melody_segments': melody_segments,
            'total_duration': song_duration,
            'num_chords': len(chords),
            'num_melody_notes': len(quantized_melody),
            'full_chord_sequence': [chord[2] for chord in chords]
        }
    
    def process_dataset(self) -> None:
        """
        Process all songs in the dataset and save the melody segments.
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

    def create_vocabularies(self) -> Dict:
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
            
        total_segments = sum(len(song['melody_segments']) for song in self.processed_data)
        avg_segments_per_song = total_segments / len(self.processed_data) if self.processed_data else 0
        
        stats = {
            'total_songs': len(self.processed_data),
            'total_melody_segments': total_segments,
            'avg_segments_per_song': avg_segments_per_song,
            'unique_chords': len(self.chord_vocab),
            'note_range': [min(self.note_vocab), max(self.note_vocab)] if self.note_vocab else [0, 127],
            'melody_segment_length': self.melody_segment_length
        }
        
        with open(self.output_path / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Processed data saved to {self.output_path}")
        print(f"Dataset statistics: {stats}")

    def create_training_sequences(self) -> List[Dict]:
        """
        Create training sequences with proper chord-melody alignment
        """
        training_sequences = []
        
        for song in self.processed_data:
            for segment in song['melody_segments']:
                for chord_pair in segment['chord_melody_pairs']:
                    
                    chord_sequence = segment['full_chord_sequence']
                    current_chord_pos = chord_pair['chord_position_in_segment']
                    
                    melody_notes = []
                    for note in chord_pair['notes']:
                        melody_notes.append({
                            'pitch': note['pitch'],
                            'start_time': note['start_in_chord'],
                            'duration': note['end_in_chord'] - note['start_in_chord'],
                            'velocity': note['velocity']
                        })
                    
                    training_example = {
                        'full_chord_sequence': chord_sequence,
                        'chord_durations': segment['chord_durations'],
                        'focus_position': current_chord_pos,
                        'target_chord': chord_pair['chord'],
                        'chord_duration': chord_pair['chord_duration'],
                        'output_melody': melody_notes,
                        'song_id': song['song_id'],
                        'segment_id': segment['segment_id'],
                        'timing_context': {
                            'bpm_estimate': 60.0 / (chord_pair['chord_duration'] / 2),
                            'time_signature': '4/4',
                            'chord_position_in_song': segment['segment_start_time'] / song['total_duration']
                        }
                    }
                    
                    if len(melody_notes) > 0:
                        training_sequences.append(training_example)
        
        with open(self.output_path / "training_sequences.pkl", 'wb') as f:
            pickle.dump(training_sequences, f)
        
        print(f"Created {len(training_sequences)} chord-aligned training sequences")
        return training_sequences
    
    
    def simplify_chord_symbol(self, chord_symbol: str) -> str:
        """
        Simplify chord symbols to reduce vocabulary size and improve generalization.
        
        Maps complex chords to basic chord types:
        - All major variations -> maj
        - All minor variations -> min  
        - All dominant 7th variations -> 7
        - All major 7th variations -> maj7
        - All minor 7th variations -> min7
        - All diminished variations -> dim
        - All augmented variations -> aug
        - All suspended variations -> sus
        """
        
        if chord_symbol == 'N' or not chord_symbol or chord_symbol.strip() == '':
            return 'N'
        
        if ':' not in chord_symbol:
            return chord_symbol
        
        root, quality = chord_symbol.split(':', 1)
        
        simplified_quality = None
        
        if any(pattern in quality for pattern in ['maj7', 'M7']):
            if any(pattern in quality for pattern in ['min', 'm7', 'minmaj']):
                simplified_quality = 'minmaj7'
            else:
                simplified_quality = 'maj7'
        elif 'maj' in quality:
            simplified_quality = 'maj'
        
        elif any(pattern in quality for pattern in ['min7', 'm7']) and 'maj' not in quality:
            simplified_quality = 'min7'
        elif 'min' in quality or quality.startswith('m'):
            simplified_quality = 'min'
        
        elif any(pattern in quality for pattern in ['7', '9', '11', '13']) and 'maj' not in quality and 'min' not in quality:
            simplified_quality = '7'
        
        elif any(pattern in quality for pattern in ['dim', 'hdim', 'ø']):
            if '7' in quality or 'hdim' in quality:
                simplified_quality = 'dim7'
            else:
                simplified_quality = 'dim'
        
        elif 'aug' in quality or '+' in quality:
            simplified_quality = 'aug'
        
        elif any(pattern in quality for pattern in ['sus2', 'sus4', 'sus']):
            if 'sus2' in quality:
                simplified_quality = 'sus2'
            else:
                simplified_quality = 'sus4'
        
        elif '6' in quality and 'min' not in quality:
            simplified_quality = 'maj'
        elif '6' in quality and 'min' in quality:
            simplified_quality = 'min'
        
        if simplified_quality is None:
            if quality == '' or quality.isdigit():
                simplified_quality = 'maj'
            else:
                simplified_quality = 'maj'
        
        return f"{root}:{simplified_quality}"

    def create_chord_aligned_segments(self, chords: List[Tuple[float, float, str]], melody: List[Dict]) -> List[Dict]:
        """
        Create training segments aligned to chord boundaries for better timing relationships.
        Each segment represents a sequence of chords with their corresponding melody notes.
        """
        segments = []
        
        if not chords or not melody:
            return segments
        
        min_chords_per_segment = 4   
        max_chords_per_segment = 12  
        overlap_chords = 2              
        
        chord_idx = 0
        while chord_idx < len(chords):
            segment_end_idx = min(chord_idx + max_chords_per_segment, len(chords))
            segment_end_idx = min(chord_idx + max_chords_per_segment, len(chords))
            
            if segment_end_idx - chord_idx < min_chords_per_segment and segment_end_idx < len(chords):
                segment_end_idx = min(chord_idx + min_chords_per_segment, len(chords))
            
            segment_chords = chords[chord_idx:segment_end_idx]
            segment_start_time = segment_chords[0][0]
            segment_end_time = segment_chords[-1][1]
            
            chord_melody_pairs = []
            
            for local_idx, (chord_start, chord_end, chord_symbol) in enumerate(segment_chords):
                chord_notes = []
                
                for note in melody:
                    note_start = note['start_time']
                    note_end = note['end_time']
                    
                    if not (note_end <= chord_start or note_start >= chord_end):
                        overlap_start = max(note_start, chord_start)
                        overlap_end = min(note_end, chord_end)
                        overlap_duration = overlap_end - overlap_start
                        note_duration = note_end - note_start
                        
                        if note_duration > 0 and overlap_duration / note_duration > 0.5:
                            relative_note = {
                                'pitch': note['pitch'],
                                'start_in_chord': max(0, note_start - chord_start),
                                'end_in_chord': min(chord_end - chord_start, note_end - chord_start),
                                'duration': note.get('duration', note_end - note_start),
                                'velocity': note.get('velocity', 80),
                                'chord_duration': chord_end - chord_start
                            }
                            chord_notes.append(relative_note)
                
                chord_melody_pairs.append({
                    'chord': chord_symbol,
                    'chord_duration': chord_end - chord_start,
                    'chord_position_in_segment': local_idx,
                    'notes': chord_notes,
                    'absolute_start_time': chord_start,
                    'absolute_end_time': chord_end
                })
            
            total_notes = sum(len(pair['notes']) for pair in chord_melody_pairs)
            if total_notes > 0 and len(segment_chords) >= min_chords_per_segment:
                segment = {
                    'chord_melody_pairs': chord_melody_pairs,
                    'full_chord_sequence': [pair['chord'] for pair in chord_melody_pairs],
                    'chord_durations': [pair['chord_duration'] for pair in chord_melody_pairs],
                    'segment_start_time': segment_start_time,
                    'segment_end_time': segment_end_time,
                    'total_duration': segment_end_time - segment_start_time,
                    'num_chords': len(segment_chords),
                    'total_notes': total_notes,
                    'segment_id': len(segments)
                }
                segments.append(segment)
            
            step_size = max_chords_per_segment - overlap_chords
            chord_idx += step_size
            
            if chord_idx >= len(chords) - min_chords_per_segment:
                break
        
        return segments
    
def process_dataset(dataset_path: str, output_path: str, melody_segment_length: int = 32):
    processor = POP909DataProcessor(dataset_path, output_path, melody_segment_length)
    processor.process_dataset()
    processor.create_vocabularies()
    processor.save_processed_data()
    processor.create_training_sequences()
    
if __name__ == "__main__":
    dataset_path = "POP909-Dataset"
    output_path = "processed_pop909_chord_melody"
    process_dataset(dataset_path, output_path)
    
    
    