from typing import List, Dict, Optional
import pretty_midi
import numpy as np

def generate_midi_from_melody(melody: List[Dict], output_path: str, tempo: int = 120, 
                             original_midi_path: Optional[str] = None):
    """
    Generate MIDI file from melody with proper timing
    
    Args:
        melody: List of note dictionaries with 'pitch', 'start_time', 'duration'
        output_path: Where to save the generated MIDI
        tempo: Target tempo in BPM (default 120)
        original_midi_path: Path to original MIDI file for tempo reference
    """
    
    # Get original tempo if available
    original_tempo = tempo
    if original_midi_path:
        try:
            original_midi = pretty_midi.PrettyMIDI(original_midi_path)
            original_tempo = original_midi.estimate_tempo()
            print(f"Original tempo: {original_tempo:.1f} BPM")
        except:
            print(f"Could not load original MIDI, using default tempo: {tempo} BPM")
            original_tempo = tempo
    
    # Create new MIDI with proper tempo
    midi = pretty_midi.PrettyMIDI(initial_tempo=original_tempo)
    
    # Create melody instrument
    melody_instrument = pretty_midi.Instrument(program=1, name="Generated Melody")  # Piano
    
    if not melody:
        print("Warning: Empty melody provided")
        midi.instruments.append(melody_instrument)
        midi.write(output_path)
        return
    
    # Convert melody timing to proper MIDI timing
    for note_info in melody:
        try:
            pitch = int(note_info['pitch'])
            start_time = float(note_info['start_time'])
            duration = float(note_info['duration'])
            
            # Ensure minimum duration and valid pitch
            duration = max(0.1, duration)  # Minimum 100ms
            if not (21 <= pitch <= 108):  # Valid piano range
                continue
                
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start_time,
                end=start_time + duration
            )
            melody_instrument.notes.append(note)
            
        except (KeyError, ValueError, TypeError) as e:
            print(f"Warning: Skipping invalid note {note_info}: {e}")
            continue
    
    # Sort notes by start time
    melody_instrument.notes.sort(key=lambda n: n.start)
    
    # Add instrument to MIDI
    midi.instruments.append(melody_instrument)
    
    # Add original piano track if available
    if original_midi_path:
        try:
            original_midi = pretty_midi.PrettyMIDI(original_midi_path)
            # Find piano/accompaniment track (usually not the melody track)
            for i, instrument in enumerate(original_midi.instruments):
                if i > 0 or "melody" not in instrument.name.lower():  # Skip melody track
                    instrument.name = f"Original_{instrument.name}"
                    midi.instruments.append(instrument)
            print(f"Added {len(original_midi.instruments)-1} original tracks")
        except Exception as e:
            print(f"Could not add original tracks: {e}")
    
    # Write MIDI file
    midi.write(output_path)
    
    # Report timing info
    if melody_instrument.notes:
        total_duration = melody_instrument.notes[-1].end
        note_count = len(melody_instrument.notes)
        print(f"Generated melody: {note_count} notes, {total_duration:.1f}s duration")
    else:
        print("Warning: No valid notes generated")


def fix_melody_timing(melody: List[Dict], chord_sequence: List[str], 
                     chords_per_measure: int = 4, measures_per_minute: int = 30) -> List[Dict]:
    """
    Fix melody timing to align with musical structure
    
    Args:
        melody: Generated melody with abstract timing
        chord_sequence: The chord progression used
        chords_per_measure: How many chords per measure (default 4)
        measures_per_minute: Tempo in measures per minute (default 30 = 120 BPM)
    
    Returns:
        melody: Melody with corrected timing in seconds
    """
    if not melody:
        return melody
    
    # Calculate timing parameters
    seconds_per_measure = 60.0 / measures_per_minute
    seconds_per_chord = seconds_per_measure / chords_per_measure
    total_song_duration = len(chord_sequence) * seconds_per_chord
    
    print(f"Timing fix: {len(chord_sequence)} chords, {seconds_per_chord:.2f}s per chord")
    print(f"Expected song duration: {total_song_duration:.1f}s")
    
    # Find the time scale of generated melody
    if melody:
        max_generated_time = max(note['start_time'] + note.get('duration', 0) for note in melody)
        if max_generated_time > 0:
            time_scale = total_song_duration / max_generated_time
            print(f"Applying time scale: {time_scale:.3f}")
        else:
            time_scale = 1.0
    else:
        time_scale = 1.0
    
    # Apply timing corrections
    fixed_melody = []
    for note in melody:
        fixed_note = note.copy()
        fixed_note['start_time'] = note['start_time'] * time_scale
        fixed_note['duration'] = note.get('duration', 0.5) * time_scale
        
        # Ensure reasonable durations
        fixed_note['duration'] = max(0.1, min(2.0, fixed_note['duration']))
        
        fixed_melody.append(fixed_note)
    
    return fixed_melody


def extract_chord_timing_from_midi(midi_path: str) -> List[tuple]:
    """
    Extract actual chord timing from original MIDI file
    
    Returns:
        List of (start_time, end_time, chord_index) tuples
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Find accompaniment track (usually has chords)
        accompaniment_track = None
        for instrument in midi.instruments:
            if not instrument.is_drum and "melody" not in instrument.name.lower():
                accompaniment_track = instrument
                break
        
        if not accompaniment_track:
            # Fallback: use any non-drum track
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    accompaniment_track = instrument
                    break
        
        if not accompaniment_track:
            print("No suitable track found for chord timing")
            return []
        
        # Extract chord timing by grouping simultaneous notes
        chord_times = []
        notes = sorted(accompaniment_track.notes, key=lambda n: n.start)
        
        if not notes:
            return []
        
        current_chord_start = notes[0].start
        current_chord_notes = []
        tolerance = 0.1  # 100ms tolerance for "simultaneous" notes
        
        for note in notes:
            if note.start - current_chord_start <= tolerance:
                # Part of current chord
                current_chord_notes.append(note)
            else:
                # New chord
                if current_chord_notes:
                    chord_end = max(n.end for n in current_chord_notes)
                    chord_times.append((current_chord_start, chord_end, len(chord_times)))
                
                current_chord_start = note.start
                current_chord_notes = [note]
        
        # Add final chord
        if current_chord_notes:
            chord_end = max(n.end for n in current_chord_notes)
            chord_times.append((current_chord_start, chord_end, len(chord_times)))
        
        print(f"Extracted {len(chord_times)} chord segments from MIDI")
        return chord_times
        
    except Exception as e:
        print(f"Error extracting chord timing: {e}")
        return []


def align_melody_to_chord_timing(melody: List[Dict], chord_times: List[tuple]) -> List[Dict]:
    """
    Align generated melody to actual chord timing from original MIDI
    
    Args:
        melody: Generated melody with abstract timing
        chord_times: List of (start_time, end_time, chord_index) from original MIDI
    
    Returns:
        melody: Melody aligned to real chord timing
    """
    if not melody or not chord_times:
        return melody
    
    # Calculate total duration from chord times
    total_chord_duration = chord_times[-1][1] - chord_times[0][0]
    
    # Find melody time span
    melody_start = min(note['start_time'] for note in melody)
    melody_end = max(note['start_time'] + note.get('duration', 0) for note in melody)
    melody_span = melody_end - melody_start
    
    if melody_span <= 0:
        return melody
    
    # Calculate time scaling
    time_scale = total_chord_duration / melody_span
    time_offset = chord_times[0][0] - melody_start * time_scale
    
    print(f"Aligning melody: scale={time_scale:.3f}, offset={time_offset:.3f}s")
    
    # Apply alignment
    aligned_melody = []
    for note in melody:
        aligned_note = note.copy()
        aligned_note['start_time'] = note['start_time'] * time_scale + time_offset
        aligned_note['duration'] = note.get('duration', 0.5) * time_scale
        
        # Ensure reasonable bounds
        aligned_note['start_time'] = max(0, aligned_note['start_time'])
        aligned_note['duration'] = max(0.1, min(4.0, aligned_note['duration']))
        
        aligned_melody.append(aligned_note)
    
    return aligned_melody