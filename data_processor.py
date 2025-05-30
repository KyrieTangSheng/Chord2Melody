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
        
        # Musical parameters
        self.time_resolution = 0.125  # 1/8 note resolution (in seconds)
        self.min_chord_duration = 0.5  # Minimum chord duration to consider
        self.melody_segment_length = melody_segment_length  # Length of melody segments to generate
        
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
                            # FIXED: Simplify chord BEFORE adding to vocabulary
                            simplified_chord = self.normalize_chord_symbol(chord_symbol)
                            chords.append((float(start_time), float(end_time), simplified_chord))
                            self.chord_vocab.add(simplified_chord)  # Add simplified chord to vocab
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
                ## NOTE: TARGET INSTRUMENT CHANGE HERE
                if instrument.name and "MELODY" in instrument.name.upper():
                    return instrument
            
            if len(midi_data.instruments) > 0:
                ## NOTE: TARGET INSTRUMENT CHANGE HERE
                return midi_data.instruments[0]
            
        except Exception as e:
            print(f"Error loading MIDI file {midi_file_path}: {e}")
            
        return None

    def extract_bridge_track(self, midi_file_path: str) -> Optional[pretty_midi.Instrument]:
        """
        Extract bridge track from MIDI file
        """
        midi_file_path = str(midi_file_path)
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            for instrument in midi_data.instruments:
                ## NOTE: TARGET INSTRUMENT CHANGE HERE
                if instrument.name and "BRIDGE" in instrument.name.upper():
                    return instrument
            
            if len(midi_data.instruments) > 0:
                ## NOTE: TARGET INSTRUMENT CHANGE HERE
                return midi_data.instruments[1]
            
        except Exception as e:
            print(f"Error loading MIDI file {midi_file_path}: {e}")
            
        return None
    
    def combine_tracks(self, melody_track: pretty_midi.Instrument, bridge_track: pretty_midi.Instrument) -> pretty_midi.Instrument:
        """
        Combine melody and bridge tracks
        """
        program = melody_track.program
        combined_track = pretty_midi.Instrument(name="Combined Melody and Bridge", program=program)
        combined_track.notes = sorted(
            melody_track.notes + bridge_track.notes,
            key=lambda note: note.start
        )
        return combined_track
    
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
    
    def create_melody_segments(self, chords: List[Tuple[float, float, str]], melody: List[Dict]) -> List[Dict]:
        """
        Create training segments by dividing the song into overlapping melody segments.
        Each segment has the full chord sequence but focuses on a specific time range.
        """
        segments = []
        
        if not chords or not melody:
            return segments
        
        song_duration = chords[-1][1]
        segment_duration = self.melody_segment_length * self.time_resolution * 2  # Rough estimate
        
        # Create overlapping segments
        segment_overlap = segment_duration * 0.25  # 25% overlap
        current_time = 0.0
        
        while current_time < song_duration - segment_duration * 0.5:
            segment_start = current_time
            segment_end = min(current_time + segment_duration, song_duration)
            
            # Find melody notes in this segment
            segment_melody = []
            for note in melody:
                if (note['start_time'] >= segment_start and note['start_time'] < segment_end):
                    relative_note = note.copy()
                    relative_note['relative_start'] = note['start_time'] - segment_start
                    relative_note['relative_end'] = note['end_time'] - segment_start
                    segment_melody.append(relative_note)
            
            # Only create segment if it has melody notes
            if len(segment_melody) > 0 and len(segment_melody) <= self.melody_segment_length:
                # Find the focus chord position (chord active during this segment)
                focus_chord_idx = 0
                focus_time = segment_start + segment_duration / 2  # Middle of segment
                
                for i, (chord_start, chord_end, _) in enumerate(chords):
                    if chord_start <= focus_time < chord_end:
                        focus_chord_idx = i
                        break
                
                segment = {
                    'full_chord_sequence': [chord[2] for chord in chords],  # All chords
                    'chord_times': [(chord[0], chord[1]) for chord in chords],  # Timing info
                    'focus_position': focus_chord_idx,  # Which chord to focus on
                    'segment_start_time': segment_start,
                    'segment_end_time': segment_end,
                    'melody_notes': segment_melody,
                    'segment_position': current_time / song_duration  # Normalized position in song
                }
                segments.append(segment)
            
            current_time += segment_duration - segment_overlap
        
        return segments
    
    def normalize_chord_symbol(self, chord_symbol: str) -> str:
        """
        Normalize and simplify chord symbols
        """
        if chord_symbol == 'N' or not chord_symbol or chord_symbol.strip() == '':
            return 'N'
        
        # First clean up the symbol
        cleaned = chord_symbol.strip()
        
        # Then simplify it
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
        
        melody_track = self.extract_melody_track(midi_file_path)
        bridge_track = self.extract_bridge_track(midi_file_path)
        
        combined_track = self.combine_tracks(melody_track, bridge_track)
        if not combined_track:
            print(f"No combined track found for {song_id}")
            return None
        
        song_duration = max(chords[-1][1], combined_track.notes[-1].end)
        
        quantized_melody = self.quantize_melody(combined_track, song_duration)
        if not quantized_melody:
            print(f"No quantized melody for song {song_id}")
            return None
        
        # CHANGED: Use chord-aligned segments instead of time-based segments
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
                # For each chord in the segment, create a training example
                for chord_pair in segment['chord_melody_pairs']:
                    
                    # Prepare input: chord sequence with focus on current chord
                    chord_sequence = segment['full_chord_sequence']
                    current_chord_pos = chord_pair['chord_position_in_segment']
                    
                    # Prepare output: melody notes for this specific chord
                    melody_notes = []
                    for note in chord_pair['notes']:
                        melody_notes.append({
                            'pitch': note['pitch'],
                            'start_time': note['start_in_chord'],  # Relative to chord start
                            'duration': note['end_in_chord'] - note['start_in_chord'],
                            'velocity': note['velocity']
                        })
                    
                    # Create training example
                    training_example = {
                        'full_chord_sequence': chord_sequence,
                        'chord_durations': segment['chord_durations'],
                        'focus_position': current_chord_pos,  # Which chord we're generating for
                        'target_chord': chord_pair['chord'],
                        'chord_duration': chord_pair['chord_duration'],
                        'output_melody': melody_notes,
                        'song_id': song['song_id'],
                        'segment_id': segment['segment_id'],
                        'timing_context': {
                            'bpm_estimate': 60.0 / (chord_pair['chord_duration'] / 2),  # Rough BPM estimate
                            'time_signature': '4/4',  # Could be extracted from MIDI
                            'chord_position_in_song': segment['segment_start_time'] / song['total_duration']
                        }
                    }
                    
                    # Only add if there are melody notes for this chord
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
        
        # Extract root note (everything before the colon)
        if ':' not in chord_symbol:
            return chord_symbol  # Handle edge cases
        
        root, quality = chord_symbol.split(':', 1)
        
        # Define simplification rules
        simplified_quality = None
        
        # Major chords (including extensions, inversions, added notes)
        if any(pattern in quality for pattern in ['maj7', 'M7']):
            if any(pattern in quality for pattern in ['min', 'm7', 'minmaj']):
                simplified_quality = 'minmaj7'  # Minor-major 7th
            else:
                simplified_quality = 'maj7'
        elif 'maj' in quality:
            simplified_quality = 'maj'
        
        # Minor chords
        elif any(pattern in quality for pattern in ['min7', 'm7']) and 'maj' not in quality:
            simplified_quality = 'min7'
        elif 'min' in quality or quality.startswith('m'):
            simplified_quality = 'min'
        
        # Dominant 7th chords (including 9th, 11th, 13th extensions)
        elif any(pattern in quality for pattern in ['7', '9', '11', '13']) and 'maj' not in quality and 'min' not in quality:
            simplified_quality = '7'
        
        # Diminished chords
        elif any(pattern in quality for pattern in ['dim', 'hdim', 'ø']):
            if '7' in quality or 'hdim' in quality:
                simplified_quality = 'dim7'
            else:
                simplified_quality = 'dim'
        
        # Augmented chords
        elif 'aug' in quality or '+' in quality:
            simplified_quality = 'aug'
        
        # Suspended chords
        elif any(pattern in quality for pattern in ['sus2', 'sus4', 'sus']):
            if 'sus2' in quality:
                simplified_quality = 'sus2'
            else:
                simplified_quality = 'sus4'
        
        # 6th chords (treat as major with added 6th -> simplify to maj)
        elif '6' in quality and 'min' not in quality:
            simplified_quality = 'maj'
        elif '6' in quality and 'min' in quality:
            simplified_quality = 'min'
        
        # Default fallback
        if simplified_quality is None:
            # Try to infer from common patterns
            if quality == '' or quality.isdigit():
                simplified_quality = 'maj'  # Default to major
            else:
                simplified_quality = 'maj'  # Conservative fallback
        
        return f"{root}:{simplified_quality}"

    def create_chord_aligned_segments(self, chords: List[Tuple[float, float, str]], melody: List[Dict]) -> List[Dict]:
        """
        Create training segments aligned to chord boundaries for better timing relationships.
        Each segment represents a sequence of chords with their corresponding melody notes.
        """
        segments = []
        
        if not chords or not melody:
            return segments
        
        # Parameters for chord-aligned segments
        min_chords_per_segment = 4    # Minimum musical context
        max_chords_per_segment = 12   # Maximum for attention span
        overlap_chords = 2            # Overlap between segments
        
        # Create chord-aligned segments
        chord_idx = 0
        while chord_idx < len(chords):
            # Define segment boundaries based on chords
            segment_end_idx = min(chord_idx + max_chords_per_segment, len(chords))
            
            # Ensure minimum segment size
            if segment_end_idx - chord_idx < min_chords_per_segment and segment_end_idx < len(chords):
                segment_end_idx = min(chord_idx + min_chords_per_segment, len(chords))
            
            segment_chords = chords[chord_idx:segment_end_idx]
            segment_start_time = segment_chords[0][0]
            segment_end_time = segment_chords[-1][1]
            
            # Create chord-to-melody mappings for this segment
            chord_melody_pairs = []
            
            for local_idx, (chord_start, chord_end, chord_symbol) in enumerate(segment_chords):
                # Find melody notes that belong to this chord
                chord_notes = []
                
                for note in melody:
                    note_start = note['start_time']
                    note_end = note['end_time']
                    
                    # Note belongs to chord if it overlaps with chord duration
                    if not (note_end <= chord_start or note_start >= chord_end):
                        # Calculate overlap percentage
                        overlap_start = max(note_start, chord_start)
                        overlap_end = min(note_end, chord_end)
                        overlap_duration = overlap_end - overlap_start
                        note_duration = note_end - note_start
                        
                        # Only include note if significant overlap (>50% of note duration)
                        if note_duration > 0 and overlap_duration / note_duration > 0.5:
                            # Make timing relative to chord start
                            relative_note = {
                                'pitch': note['pitch'],
                                'start_in_chord': max(0, note_start - chord_start),
                                'end_in_chord': min(chord_end - chord_start, note_end - chord_start),
                                'duration': note.get('duration', note_end - note_start),  # Handle missing duration
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
            
            # Only create segment if it has reasonable musical content
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
            
            # Move to next segment with overlap
            step_size = max_chords_per_segment - overlap_chords
            chord_idx += step_size
            
            # Safety check
            if chord_idx >= len(chords) - min_chords_per_segment:
                break
        
        return segments
    
    
def process_dataset(dataset_path: str, output_path: str, melody_segment_length: int = 32):
    processor = POP909DataProcessor(dataset_path, output_path, melody_segment_length)
    processor.process_dataset()
    processor.create_vocabularies()
    processor.save_processed_data()
    processor.create_training_sequences()


def test_chord_simplification():
    """Test the chord simplification on your vocabulary"""
    
    # Sample of complex chords from your vocabulary
    test_chords = [
        "C:maj", "C:maj7", "C:maj6", "C:maj6(9)", "C:maj7/3", "C:maj(9)/5",
        "C:min", "C:min7", "C:min6", "C:min7(11)", "C:min9/5",
        "C:7", "C:9", "C:11", "C:7/3", "C:9(13)", "C:7/b7",
        "C:dim", "C:dim7", "C:hdim7", "C:dim/b3",
        "C:aug", "C:aug(b7)",
        "C:sus4", "C:sus2", "C:sus4(b7)", "C:sus4(b7,9)",
        "F#:maj6(9)/5", "Bb:min7(4)/b7", "Ab:maj9(13)"
    ]
    
    processor = POP909DataProcessor("", "")  # Dummy paths for testing
    
    print("Chord Simplification Test:")
    print("Original -> Simplified")
    print("-" * 30)
    
    simplified_set = set()
    for chord in test_chords:
        simplified = processor.simplify_chord_symbol(chord)
        print(f"{chord:<20} -> {simplified}")
        simplified_set.add(simplified)
    
    print(f"\nOriginal count: {len(test_chords)}")
    print(f"Simplified count: {len(simplified_set)}")
    print(f"Reduction: {len(test_chords) - len(simplified_set)} chords")
    print(f"Simplified vocabulary: {sorted(simplified_set)}")
    
if __name__ == "__main__":
    dataset_path = "POP909-Dataset"
    output_path = "processed_pop909_chord_melody"
    process_dataset(dataset_path, output_path)