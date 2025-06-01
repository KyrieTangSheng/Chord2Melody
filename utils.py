from typing import List, Dict, Optional
import pretty_midi
import numpy as np

def generate_midi_from_melody(melody: List[Dict], output_path: str, tempo: int = 120, 
                             original_midi_path: Optional[str] = None):
    """
    Generate MIDI file from melody with proper timing
    """
    
    original_tempo = tempo
    if original_midi_path:
        try:
            original_midi = pretty_midi.PrettyMIDI(original_midi_path)
            original_tempo = original_midi.estimate_tempo()
            print(f"Original tempo: {original_tempo:.1f} BPM")
        except:
            print(f"Could not load original MIDI, using default tempo: {tempo} BPM")
            original_tempo = tempo
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=original_tempo)
    
    melody_instrument = pretty_midi.Instrument(program=1, name="Generated Melody")
    
    if not melody:
        print("Warning: Empty melody provided")
        midi.instruments.append(melody_instrument)
        midi.write(output_path)
        return
    
    for note_info in melody:
        try:
            pitch = int(note_info['pitch'])
            start_time = float(note_info['start_time'])
            duration = float(note_info['duration'])
            
            duration = max(0.1, duration)
            if not (21 <= pitch <= 108):
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
    
    melody_instrument.notes.sort(key=lambda n: n.start)
    
    midi.instruments.append(melody_instrument)
    
    if original_midi_path:
        try:
            original_midi = pretty_midi.PrettyMIDI(original_midi_path)
            for i, instrument in enumerate(original_midi.instruments):
                if i > 0 or "melody" not in instrument.name.lower():
                    instrument.name = f"Original_{instrument.name}"
                    midi.instruments.append(instrument)
            print(f"Added {len(original_midi.instruments)-1} original tracks")
        except Exception as e:
            print(f"Could not add original tracks: {e}")
    
    midi.write(output_path)
    
    if melody_instrument.notes:
        total_duration = melody_instrument.notes[-1].end
        note_count = len(melody_instrument.notes)
        print(f"Generated melody: {note_count} notes, {total_duration:.1f}s duration")
    else:
        print("Warning: No valid notes generated")

def extract_chord_timing_from_midi(midi_path: str, target_chord_count: int = None) -> List[tuple]:
    """
    Extract actual chord timing from original MIDI file with better alignment
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        total_duration = midi.get_end_time()
        
        print(f"MIDI Analysis: {midi_path}")
        print(f"  Total duration: {total_duration:.1f}s")
        print(f"  Target chord count: {target_chord_count}")
        
        accompaniment_track = None
        max_polyphony = 0
        
        for i, instrument in enumerate(midi.instruments):
            if instrument.is_drum:
                continue
                
            polyphony = calculate_average_polyphony(instrument)
            print(f"  Track {i} ({instrument.name}): {len(instrument.notes)} notes, polyphony: {polyphony:.1f}")
            
            if "melody" in instrument.name.lower() or "lead" in instrument.name.lower():
                continue
                
            if polyphony > max_polyphony:
                max_polyphony = polyphony
                accompaniment_track = instrument
        
        if not accompaniment_track:
            print("  No suitable accompaniment track found, using fallback")
            return create_fallback_timing(total_duration, target_chord_count)
        
        print(f"  Using track: {accompaniment_track.name} (polyphony: {max_polyphony:.1f})")
        
        chord_changes = detect_chord_changes(accompaniment_track, total_duration)
        
        print(f"  Detected {len(chord_changes)} chord changes")
        
        if target_chord_count and len(chord_changes) != target_chord_count:
            chord_changes = align_chord_timing(chord_changes, target_chord_count, total_duration)
            print(f"  Aligned to {len(chord_changes)} chord segments")
        
        return chord_changes
        
    except Exception as e:
        print(f"Error extracting chord timing: {e}")
        if target_chord_count:
            return create_fallback_timing(60.0, target_chord_count) 
        return []

def calculate_average_polyphony(instrument):
    """Calculate average number of simultaneous notes"""
    if not instrument.notes:
        return 0.0
    
    duration = max(note.end for note in instrument.notes)
    sample_points = int(duration * 4)
    
    total_polyphony = 0
    for i in range(sample_points):
        time_point = i * 0.25
        active_notes = sum(1 for note in instrument.notes 
                          if note.start <= time_point < note.end)
        total_polyphony += active_notes
    
    return total_polyphony / sample_points if sample_points > 0 else 0

def detect_chord_changes(instrument, total_duration):
    """Detect when chords change in the instrument"""
    if not instrument.notes:
        return []
    
    chord_events = []
    tolerance = 0.15
    
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
    
    i = 0
    while i < len(sorted_notes):
        chord_start = sorted_notes[i].start
        chord_notes = [sorted_notes[i]]
        
        j = i + 1
        while j < len(sorted_notes) and sorted_notes[j].start - chord_start <= tolerance:
            chord_notes.append(sorted_notes[j])
            j += 1
        
        if len(chord_notes) >= 2 or (j < len(sorted_notes) and sorted_notes[j].start - chord_start > 0.5):
            chord_events.append({
                'start': chord_start,
                'notes': chord_notes,
                'polyphony': len(chord_notes)
            })
        
        i = j
    
    chord_times = []
    for i, event in enumerate(chord_events):
        start_time = event['start']
        
        if i + 1 < len(chord_events):
            end_time = chord_events[i + 1]['start']
        else:
            end_time = max(note.end for note in event['notes'])
            end_time = min(end_time, total_duration)
        
        if end_time - start_time >= 0.3:
            chord_times.append((start_time, end_time, len(chord_times)))
    
    return chord_times

def align_chord_timing(detected_changes, target_count, total_duration):
    """Align detected chord changes to target chord count"""
    
    if len(detected_changes) == target_count:
        return detected_changes
    
    print(f"  Aligning {len(detected_changes)} detected changes to {target_count} target chords")
    
    if len(detected_changes) > target_count:
        return merge_chord_segments(detected_changes, target_count)
    else:
        return interpolate_chord_segments(detected_changes, target_count, total_duration)
        return interpolate_chord_segments(detected_changes, target_count, total_duration)

def merge_chord_segments(segments, target_count):
    """Merge chord segments to reach target count"""
    if target_count >= len(segments):
        return segments
    
    indices = [int(i * len(segments) / target_count) for i in range(target_count)]
    merged = []
    
    for i, idx in enumerate(indices):
        start_time = segments[idx][0]
        
        if i + 1 < len(indices):
            next_idx = indices[i + 1]
            end_time = segments[next_idx][0]
        else:
            end_time = segments[-1][1]
        
        merged.append((start_time, end_time, i))
    
    return merged

def interpolate_chord_segments(segments, target_count, total_duration):
    """Interpolate to create more chord segments"""
    if not segments:
        return create_fallback_timing(total_duration, target_count)
    
    chord_duration = total_duration / target_count
    interpolated = []
    
    for i in range(target_count):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration
        interpolated.append((start_time, end_time, i))
    
    return interpolated

def create_fallback_timing(duration, chord_count):
    """Create uniform chord timing as fallback"""
    if chord_count <= 0:
        return []
    
    chord_duration = duration / chord_count
    return [(i * chord_duration, (i + 1) * chord_duration, i) 
            for i in range(chord_count)]


def align_melody_to_chord_timing(melody: List[Dict], chord_times: List[tuple]) -> List[Dict]:
    """
    Align generated melody to actual chord timing from original MIDI
    """
    if not melody or not chord_times:
        return melody
    
    total_chord_duration = chord_times[-1][1] - chord_times[0][0]
    
    melody_start = min(note['start_time'] for note in melody)
    melody_end = max(note['start_time'] + note.get('duration', 0) for note in melody)
    melody_span = melody_end - melody_start
    
    if melody_span <= 0:
        return melody
    
    time_scale = total_chord_duration / melody_span
    time_offset = chord_times[0][0] - melody_start * time_scale
    
    print(f"Aligning melody: scale={time_scale:.3f}, offset={time_offset:.3f}s")
    
    aligned_melody = []
    for note in melody:
        aligned_note = note.copy()
        aligned_note['start_time'] = note['start_time'] * time_scale + time_offset
        aligned_note['duration'] = note.get('duration', 0.5) * time_scale
        
        aligned_note['start_time'] = max(0, aligned_note['start_time'])
        aligned_note['duration'] = max(0.1, min(4.0, aligned_note['duration']))
        
        aligned_melody.append(aligned_note)
    
    return aligned_melody




