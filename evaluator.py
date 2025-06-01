import numpy as np
from typing import List, Dict, Tuple

class SimpleMelodyEvaluator:
    def __init__(self):
        # Simple chord-to-pitch mappings
        self.chord_pitch_maps = {
            'maj': [0, 4, 7],           # C major: C, E, G
            'maj7': [0, 4, 7, 11],      # Cmaj7: C, E, G, B
            'min': [0, 3, 7],           # C minor: C, Eb, G
            'min7': [0, 3, 7, 10],      # Cm7: C, Eb, G, Bb
            'minmaj7': [0, 3, 7, 11],   # CmMaj7: C, Eb, G, B
            '7': [0, 4, 7, 10],         # C7: C, E, G, Bb
            'dim': [0, 3, 6],           # Cdim: C, Eb, Gb
            'aug': [0, 4, 8],           # Caug: C, E, G#
            'sus2': [0, 2, 7],          # Csus2: C, D, G
            'sus4': [0, 5, 7],          # Csus4: C, F, G
        }

    def parse_chord_symbol(self, chord_symbol: str) -> Tuple[int, str]:
        """Parse chord symbol into root note and quality"""
        if ':' not in chord_symbol:
            return 0, 'maj'
        
        root_str, quality = chord_symbol.split(':', 1)
        
        root_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        root = root_map.get(root_str, 0)
        return root, quality

    def get_chord_pitches(self, chord_symbol: str) -> List[int]:
        """Get pitch classes that belong to a chord"""
        root, quality = self.parse_chord_symbol(chord_symbol)
        
        if quality in self.chord_pitch_maps:
            intervals = self.chord_pitch_maps[quality]
        else:
            intervals = self.chord_pitch_maps['maj'] 
        
        return [(root + interval) % 12 for interval in intervals]

    def find_chord_at_time(self, time: float, chord_times: List[Tuple]) -> int:
        """Find which chord is active at a given time"""
        for i, chord_time in enumerate(chord_times):
            if len(chord_time) >= 2:
                start, end = chord_time[0], chord_time[1]
                if start <= time < end:
                    return i
        return len(chord_times) - 1 if chord_times else 0

    def chord_alignment_score(self, melody: List[Dict], chord_sequence: List[str], chord_times: List[Tuple]) -> float:
        """How well do melody notes fit the chords? (0.0 to 1.0)"""
        if not melody or not chord_sequence:
            return 0.0
        
        alignment_scores = []
        
        for note in melody:
            chord_idx = self.find_chord_at_time(note['start_time'], chord_times)
            if chord_idx < len(chord_sequence):
                chord = chord_sequence[chord_idx]
                chord_notes = self.get_chord_pitches(chord)
                pitch_class = note['pitch'] % 12
                
                if pitch_class in chord_notes:
                    alignment_scores.append(1.0)
                else:
                    alignment_scores.append(0.0)
            else:
                alignment_scores.append(0.5)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0

    def timing_accuracy_score(self, melody: List[Dict], expected_duration: float) -> float:
        """How well does generated melody match expected duration? (0.0 to 1.0)"""
        if not melody or expected_duration <= 0:
            return 0.0
        
        actual_duration = max(n['start_time'] + n['duration'] for n in melody)
        
        ratio = min(actual_duration, expected_duration) / max(actual_duration, expected_duration)
        return ratio

    def note_density_score(self, melody: List[Dict]) -> float:
        """How reasonable is the note density? (0.0 to 1.0)"""
        if not melody:
            return 0.0
        
        total_duration = max(n['start_time'] + n['duration'] for n in melody)
        if total_duration <= 0:
            return 0.0
        
        density = len(melody) / total_duration
        
        if 0.5 <= density <= 2.0:
            return 1.0
        elif density < 0.5:
            return density / 0.5
        else:
            return max(0.0, 1.0 - (density - 2.0) / 3.0)

    def evaluate(self, melody: List[Dict], chord_sequence: List[str], chord_times: List[Tuple]) -> Dict[str, float]:
        """Main evaluation function - returns scores between 0.0 and 1.0"""
        
        if chord_times and len(chord_times[0]) >= 2:
            expected_duration = chord_times[-1][1]
        else:
            expected_duration = len(chord_sequence) * 2.0
        
        scores = {
            'chord_alignment': self.chord_alignment_score(melody, chord_sequence, chord_times),
            'timing_accuracy': self.timing_accuracy_score(melody, expected_duration),
            'note_density': self.note_density_score(melody)
        }
        
        scores['overall'] = (scores['chord_alignment'] + scores['timing_accuracy'] + scores['note_density']) / 3.0
        
        return scores

    def print_evaluation(self, melody: List[Dict], chord_sequence: List[str], chord_times: List[Tuple]):
        """Print a nice evaluation report"""
        scores = self.evaluate(melody, chord_sequence, chord_times)
        
        print(f"\n🎵 MELODY EVALUATION")
        print("=" * 30)
        print(f"Overall Score:     {scores['overall']:.3f}")
        print(f"Chord Alignment:   {scores['chord_alignment']:.3f}")
        print(f"Timing Accuracy:   {scores['timing_accuracy']:.3f}")
        print(f"Note Density:      {scores['note_density']:.3f}")
        
        if scores['overall'] > 0.8:
            print("🌟 Excellent!")
        elif scores['overall'] > 0.6:
            print("✅ Good")
        elif scores['overall'] > 0.4:
            print("⚠️  Okay")
        else:
            print("❌ Needs work")
        
        if melody:
            duration = max(n['start_time'] + n['duration'] for n in melody)
            density = len(melody) / duration if duration > 0 else 0
            pitch_range = max(n['pitch'] for n in melody) - min(n['pitch'] for n in melody)
            
            print(f"\n📊 Stats:")
            print(f"Notes: {len(melody)}")
            print(f"Duration: {duration:.1f}s")
            print(f"Density: {density:.2f} notes/sec")
            print(f"Pitch range: {pitch_range} semitones")
        
        return scores


def evaluate_melody(melody, chord_sequence, chord_times):
    """Simple function to evaluate a melody"""
    evaluator = SimpleMelodyEvaluator()
    return evaluator.print_evaluation(melody, chord_sequence, chord_times)