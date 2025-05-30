import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import math

class AttentionChordToMelodyTransformer(nn.Module):
    def __init__(self, 
                chord_vocab_size: int,
                note_vocab_size: int,
                d_model: int = 256,
                nhead: int = 8,
                num_layers: int = 6,
                max_melody_length: int = 32,
                max_chord_length: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.max_melody_length = max_melody_length
        self.max_chord_length = max_chord_length
        
        # Embeddings
        self.chord_embedding = nn.Embedding(chord_vocab_size, d_model)
        self.note_embedding = nn.Embedding(note_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_melody_length, d_model)
        self.chord_position_embedding = nn.Embedding(max_chord_length, d_model)
        
        # Segment position embedding (for song-level position)
        self.segment_position_embedding = nn.Linear(1, d_model)
        
        # Distance-aware attention for chord encoder
        self.chord_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, 
                nhead, 
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers // 2
        )
        
        # Melody decoder
        self.melody_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, 
                nhead, 
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers // 2
        )
        
        # Focus attention mechanism
        self.focus_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.focus_projection = nn.Linear(d_model, d_model)
        
        # Output heads
        self.pitch_head = nn.Linear(d_model, note_vocab_size)
        self.duration_head = nn.Linear(d_model, 1)
        self.start_head = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for transformer"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
    
    def create_distance_attention_mask(self, seq_len: int, focus_positions: torch.Tensor, focus_window: int = 8):
        """
        Create attention mask that focuses on nearby chords with some global context
        """
        batch_size = focus_positions.size(0)
        device = focus_positions.device
        
        # Create base attention mask (allow all positions initially)
        mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
        
        for b in range(batch_size):
            focus_pos = focus_positions[b].item()
            
            # Create distance-based weights
            distances = torch.arange(seq_len, device=device).float()
            focus_distances = torch.abs(distances - focus_pos)
            
            # Local focus: strong attention to nearby chords
            local_weights = torch.exp(-focus_distances / (focus_window / 2))
            
            # Global context: periodic attention to distant chords
            global_weights = torch.where(
                (torch.arange(seq_len, device=device) % 4) == 0,
                torch.ones(seq_len, device=device) * 0.3,
                torch.zeros(seq_len, device=device)
            )
            
            # Combine local and global weights
            combined_weights = local_weights + global_weights
            combined_weights = combined_weights / combined_weights.sum()
            
            # Create attention mask for this batch item
            for i in range(seq_len):
                mask[b, i, :] = combined_weights
        
        return mask
    
    # TIMING ENHANCEMENT METHODS - Added from first document
    def _align_simultaneous_notes(self, melody, tolerance=0.1):
        """
        Align notes that should be played simultaneously
        
        Args:
            melody: List of note dictionaries with 'start_time', 'duration', 'pitch'
            tolerance: Time tolerance for considering notes simultaneous (in seconds)
        
        Returns:
            melody: Melody with aligned simultaneous notes
        """
        if len(melody) <= 1:
            return melody
        
        # Sort melody by start time
        sorted_melody = sorted(melody, key=lambda x: x['start_time'])
        aligned_melody = []
        
        i = 0
        while i < len(sorted_melody):
            current_note = sorted_melody[i].copy()
            simultaneous_notes = [current_note]
            
            # Find all notes that should start at the same time
            j = i + 1
            while j < len(sorted_melody):
                next_note = sorted_melody[j]
                time_diff = abs(next_note['start_time'] - current_note['start_time'])
                
                if time_diff <= tolerance:
                    # This note should be simultaneous
                    simultaneous_notes.append(next_note.copy())
                    j += 1
                else:
                    break
            
            if len(simultaneous_notes) > 1:
                # Align all simultaneous notes
                avg_start_time = sum(note['start_time'] for note in simultaneous_notes) / len(simultaneous_notes)
                
                print(f"  Aligning {len(simultaneous_notes)} simultaneous notes at {avg_start_time:.2f}s")
                print(f"    Pitches: {[note['pitch'] for note in simultaneous_notes]}")
                
                # Align start times
                for note in simultaneous_notes:
                    note['start_time'] = avg_start_time
                
                # Optional: Align durations for better harmony
                avg_duration = sum(note['duration'] for note in simultaneous_notes) / len(simultaneous_notes)
                for note in simultaneous_notes:
                    note['duration'] = avg_duration
            
            aligned_melody.extend(simultaneous_notes)
            i = j
        
        return aligned_melody

    def _quantize_note_timing(self, melody, beat_duration=0.25):
        """
        Quantize note timing to musical beats for cleaner rhythm
        
        Args:
            melody: List of note dictionaries
            beat_duration: Duration of one beat in seconds (0.25 = 16th note at 120 BPM)
        
        Returns:
            melody: Melody with quantized timing
        """
        print(f"  Quantizing notes to {beat_duration:.3f}s grid...")
        
        quantized_melody = []
        
        for note in melody:
            quantized_note = note.copy()
            
            # Quantize start time to nearest beat
            quantized_start = round(note['start_time'] / beat_duration) * beat_duration
            
            # Quantize duration to musical values
            raw_duration = note['duration']
            
            # Common musical durations (in beats at 120 BPM)
            musical_durations = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]  # 16th to whole note
            
            # Find closest musical duration
            closest_duration = min(musical_durations, key=lambda x: abs(x - raw_duration))
            
            quantized_note['start_time'] = quantized_start
            quantized_note['duration'] = closest_duration
            
            quantized_melody.append(quantized_note)
        
        return quantized_melody

    def _remove_timing_conflicts(self, melody, min_gap=0.05):
        """
        Remove notes that create timing conflicts while preserving harmonies
        
        Args:
            melody: List of note dictionaries
            min_gap: Minimum gap between non-simultaneous notes
        
        Returns:
            melody: Cleaned melody without conflicts
        """
        if len(melody) <= 1:
            return melody
        
        sorted_melody = sorted(melody, key=lambda x: (x['start_time'], x['pitch']))
        cleaned_melody = []
        
        for note in sorted_melody:
            should_add = True
            
            for existing_note in cleaned_melody:
                existing_end = existing_note['start_time'] + existing_note['duration']
                note_start = note['start_time']
                note_end = note['start_time'] + note['duration']
                
                # Check if notes are truly simultaneous (same start time)
                if abs(existing_note['start_time'] - note_start) < 0.01:
                    # Simultaneous notes are OK - this creates harmony
                    continue
                
                # Check for problematic overlaps
                elif (note_start < existing_end - min_gap and 
                      note_end > existing_note['start_time'] + min_gap):
                    
                    # Notes overlap but aren't simultaneous - this sounds bad
                    print(f"    Removing conflicting note: pitch {note['pitch']} at {note_start:.2f}s")
                    should_add = False
                    break
            
            if should_add:
                cleaned_melody.append(note)
        
        return cleaned_melody

    def _enhance_note_timing(self, melody):
        """
        Master function to enhance note timing for better musical quality
        """
        if not melody:
            return melody
        
        print(f"Enhancing timing for {len(melody)} notes...")
        
        # Step 1: Remove obvious conflicts
        melody = self._remove_timing_conflicts(melody, min_gap=0.05)
        print(f"  After conflict removal: {len(melody)} notes")
        
        # Step 2: Align simultaneous notes
        melody = self._align_simultaneous_notes(melody, tolerance=0.15)
        print(f"  After alignment: {len(melody)} notes")
        
        # Step 3: Quantize to musical grid (optional - comment out if too rigid)
        # melody = self._quantize_note_timing(melody, beat_duration=0.25)
        # print(f"  After quantization: {len(melody)} notes")
        
        # Step 4: Final cleanup
        melody = sorted(melody, key=lambda x: x['start_time'])
        
        # Step 5: Report simultaneous note groups
        self._report_simultaneous_groups(melody)
        
        return melody

    def _report_simultaneous_groups(self, melody):
        """Report groups of simultaneous notes for debugging"""
        simultaneous_groups = []
        i = 0
        
        while i < len(melody):
            current_time = melody[i]['start_time']
            group = [melody[i]]
            
            j = i + 1
            while j < len(melody) and abs(melody[j]['start_time'] - current_time) < 0.01:
                group.append(melody[j])
                j += 1
            
            if len(group) > 1:
                pitches = [note['pitch'] for note in group]
                simultaneous_groups.append((current_time, pitches))
            
            i = j
        
        if simultaneous_groups:
            print(f"  Found {len(simultaneous_groups)} simultaneous note groups:")
            for time, pitches in simultaneous_groups[:5]:  # Show first 5
                print(f"    {time:.2f}s: pitches {pitches}")
            if len(simultaneous_groups) > 5:
                print(f"    ... and {len(simultaneous_groups) - 5} more groups")

    def _calculate_song_bpm(self, chord_times):
        """Calculate BPM from chord timing for better quantization"""
        if len(chord_times) < 4:
            return 120  # Default BPM
        
        # Calculate average chord duration
        durations = [end - start for start, end in chord_times]
        avg_chord_duration = sum(durations) / len(durations)
        
        # Estimate BPM (assuming 1 chord = 1 beat roughly)
        estimated_bpm = 60.0 / avg_chord_duration
        
        # Snap to common BPM values
        common_bpms = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180]
        closest_bpm = min(common_bpms, key=lambda x: abs(x - estimated_bpm))
        
        print(f"  Estimated BPM: {estimated_bpm:.1f} -> Using: {closest_bpm}")
        return closest_bpm

    def _enhance_note_timing_with_bpm(self, melody, beat_duration):
        """Enhanced timing with BPM awareness"""
        if not melody:
            return melody
        
        print(f"Enhancing timing for {len(melody)} notes (beat={beat_duration:.3f}s)...")
        
        # Step 1: Remove conflicts
        melody = self._remove_timing_conflicts(melody)
        
        # Step 2: Align simultaneous notes  
        melody = self._align_simultaneous_notes(melody, tolerance=beat_duration * 0.5)
        
        # Step 3: Light quantization (only if notes are close to beat grid)
        quantized_melody = []
        for note in melody:
            quantized_note = note.copy()
            
            # Only quantize if note is close to a beat boundary
            closest_beat_time = round(note['start_time'] / beat_duration) * beat_duration
            time_diff = abs(note['start_time'] - closest_beat_time)
            
            if time_diff < beat_duration * 0.3:  # Within 30% of beat duration
                quantized_note['start_time'] = closest_beat_time
            
            quantized_melody.append(quantized_note)
        
        self._report_simultaneous_groups(quantized_melody)
        return sorted(quantized_melody, key=lambda x: x['start_time'])
    
    # END OF TIMING ENHANCEMENT METHODS

    def forward(self, full_chord_sequence, chord_mask, focus_positions, chord_durations=None,
                melody_pitch=None, training=True):
        batch_size = full_chord_sequence.size(0)
        chord_seq_len = full_chord_sequence.size(1)
        
        # Embed full chord sequence
        chord_embedded = self.chord_embedding(full_chord_sequence)
        
        # Add positional encoding for chords
        chord_positions = torch.arange(chord_seq_len, device=full_chord_sequence.device)
        chord_pos_emb = self.chord_position_embedding(chord_positions).unsqueeze(0).expand(batch_size, -1, -1)
        chord_embedded = chord_embedded + chord_pos_emb
        
        # Add chord duration information if available
        if chord_durations is not None:
            duration_emb = chord_durations.unsqueeze(-1).expand(-1, -1, self.d_model) * 0.1
            chord_embedded = chord_embedded + duration_emb
        
        chord_embedded = self.dropout(chord_embedded)
        
        # Apply padding mask for chord sequence
        padding_mask = ~chord_mask  # Invert because True means "ignore"
        
        # Encode chords with attention
        chord_encoded = self.chord_encoder(
            chord_embedded,
            src_key_padding_mask=padding_mask
        )
        
        # Apply focus attention to emphasize relevant chords
        focus_context, focus_weights = self.focus_attention(
            chord_encoded, chord_encoded, chord_encoded,
            key_padding_mask=padding_mask
        )
        
        # Combine original and focused representations
        chord_encoded = chord_encoded + self.focus_projection(focus_context)
        chord_encoded = self.layer_norm(chord_encoded)
        
        if training:
            # Training mode with teacher forcing
            target_melody = melody_pitch[:, :-1]  # Remove last token
            
            melody_embedded = self.note_embedding(target_melody)
            melody_seq_len = target_melody.size(1)
            
            # Add positional encoding for melody
            melody_positions = torch.arange(melody_seq_len, device=target_melody.device)
            melody_pos_emb = self.position_embedding(melody_positions).unsqueeze(0).expand(batch_size, -1, -1)
            melody_embedded = melody_embedded + melody_pos_emb
            melody_embedded = self.dropout(melody_embedded)

            # Create causal mask for melody
            causal_mask = torch.triu(torch.ones(melody_seq_len, melody_seq_len, device=target_melody.device), diagonal=1).bool()
            
            # Decode melody
            decoded = self.melody_decoder(
                melody_embedded, 
                chord_encoded, 
                tgt_mask=causal_mask,
                memory_key_padding_mask=padding_mask
            )
            
            # Generate outputs
            pitch_logits = self.pitch_head(decoded)
            duration_pred = self.duration_head(decoded).squeeze(-1)
            start_pred = self.start_head(decoded).squeeze(-1)
            
            return pitch_logits, duration_pred, start_pred
        
        else:
            # Generation mode
            generated_sequence = []
            current_input = torch.zeros(batch_size, 1, dtype=torch.long, device=full_chord_sequence.device)
            
            for i in range(self.max_melody_length):
                melody_embedded = self.note_embedding(current_input)
                melody_seq_len = current_input.size(1)
                
                # Add positional encoding for current sequence
                melody_positions = torch.arange(melody_seq_len, device=current_input.device)
                melody_pos_emb = self.position_embedding(melody_positions).unsqueeze(0).expand(batch_size, -1, -1)
                melody_embedded = melody_embedded + melody_pos_emb
                melody_embedded = self.dropout(melody_embedded)
                
                # Create causal mask for current sequence
                causal_mask = torch.triu(torch.ones(melody_seq_len, melody_seq_len, device=current_input.device), diagonal=1).bool()
                
                # Decode current sequence
                decoded = self.melody_decoder(
                    melody_embedded,
                    chord_encoded,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=padding_mask
                )
                
                # Generate predictions for next token
                last_decoded = decoded[:, -1:]
                pitch_logits = self.pitch_head(last_decoded)
                duration_pred = self.duration_head(last_decoded).squeeze(-1)
                start_pred = self.start_head(last_decoded).squeeze(-1)
                
                # Sample next pitch
                pitch_probs = torch.softmax(pitch_logits.squeeze(1), dim=-1)
                next_pitch = torch.multinomial(pitch_probs, 1)
                
                # Store predictions
                generated_sequence.append({
                    'pitch': next_pitch,
                    'duration': duration_pred,
                    'start': start_pred
                })
                
                # Early stopping if we generate a rest or reach max length
                if next_pitch.item() == 0 or len(generated_sequence) >= self.max_melody_length:
                    break
                
                # Update input sequence for next iteration
                current_input = torch.cat([current_input, next_pitch], dim=1)
            
            return generated_sequence
    
    def generate_full_song_melody(self, full_chord_sequence, vocab_path, segment_length=16, overlap=4, 
                    target_density=0.6, density_window=8, temperature=1.0,
                    original_midi_path=None):
        """
        Generate melody for an entire song by processing it in segments with proper timing
        """
        self.eval()
        device = next(self.parameters()).device
        
        with open(vocab_path, 'r') as f:
            vocabularies = json.load(f)
            
        chord_to_idx = vocabularies['chord_vocab']
        idx_to_note = vocabularies['idx_to_note']
        
        # Convert chord sequence to indices
        chord_indices = []
        for chord in full_chord_sequence:
            chord_idx = chord_to_idx.get(chord, chord_to_idx.get('<UNK>', 0))
            chord_indices.append(chord_idx)
        
        if len(chord_indices) > self.max_chord_length:
            print(f"Warning: Chord sequence too long ({len(chord_indices)} > {self.max_chord_length}), truncating")
            chord_indices = chord_indices[:self.max_chord_length]
        
        # Extract real timing if original MIDI available
        chord_times = []
        if original_midi_path:
            from utils import extract_chord_timing_from_midi
            chord_times = extract_chord_timing_from_midi(original_midi_path)
        
        # Pad chord sequence
        original_length = len(chord_indices)
        padded_chords = chord_indices + [chord_to_idx.get('<PAD>', 0)] * (self.max_chord_length - len(chord_indices))
        
        # Create chord mask
        chord_mask = [True] * original_length + [False] * (self.max_chord_length - original_length)
        
        full_melody = []
        step_size = max(1, segment_length - overlap)
        
        print(f"Generating melody for {len(chord_indices)} chords...")
        print(f"Segment length: {segment_length}, Overlap: {overlap}, Step size: {step_size}")
        print(f"Target density: {target_density:.2f}, Temperature: {temperature:.2f}")
        
        # Calculate timing parameters - FIXED to prevent over-extension
        if chord_times and len(chord_times) >= len(chord_indices):
            # Use real timing from original MIDI
            print("Using real chord timing from original MIDI")
            total_song_duration = chord_times[-1][1] - chord_times[0][0]
            segment_durations = []
            for i in range(0, len(chord_indices), step_size):
                seg_start_idx = i
                seg_end_idx = min(i + segment_length, len(chord_indices))
                
                if seg_start_idx < len(chord_times) and seg_end_idx <= len(chord_times):
                    seg_start_time = chord_times[seg_start_idx][0]
                    seg_end_time = chord_times[seg_end_idx-1][1]
                    segment_durations.append((seg_start_time, seg_end_time - seg_start_time))
                else:
                    # Fallback to estimated timing
                    estimated_duration = segment_length * 2.0
                    estimated_start = i * 2.0
                    segment_durations.append((estimated_start, estimated_duration))
        else:
            # Use estimated timing
            print("Using estimated chord timing (2 seconds per chord)")
            total_song_duration = len(chord_indices) * 2.0
            segment_durations = []
            for i in range(0, len(chord_indices), step_size):
                estimated_start = i * 2.0
                estimated_duration = min(segment_length, len(chord_indices) - i) * 2.0
                segment_durations.append((estimated_start, estimated_duration))
        
        # Generate melody segment by segment
        current_position = 0
        segment_idx = 0
        
        print(f"Total song duration: {total_song_duration:.1f}s")
        print(f"Will generate segments until position {original_length}")
        
        while current_position < original_length:
            # Calculate focus position for this segment
            segment_start = current_position
            segment_end = min(segment_start + segment_length, original_length)
            focus_position = (segment_start + segment_end) // 2
            focus_position = min(focus_position, original_length - 1)
            
            # Get timing for this segment
            if segment_idx < len(segment_durations):
                segment_start_time, segment_duration = segment_durations[segment_idx]
            else:
                # Calculate timing for segments beyond pre-calculated ones
                segment_start_time = current_position * 2.0
                remaining_chords = original_length - current_position
                segment_duration = min(segment_length, remaining_chords) * 2.0
            
            # Create tensors
            chord_tensor = torch.tensor([padded_chords], dtype=torch.long, device=device)
            mask_tensor = torch.tensor([chord_mask], dtype=torch.bool, device=device)
            focus_tensor = torch.tensor([focus_position], dtype=torch.long, device=device)
            segment_pos = torch.tensor([current_position / max(1, original_length - 1)], dtype=torch.float32, device=device)
            
            # Generate melody for this segment using density control
            with torch.no_grad():
                generated_sequence = self._generate_with_density_control(
                    chord_tensor, 
                    mask_tensor, 
                    focus_tensor,
                    target_density=target_density,
                    density_window=density_window,
                    temperature=temperature
                )
            
            # FIXED: Process and add generated notes with CONTROLLED timing
            segment_melody = []
            valid_notes = [note for note in generated_sequence if note['pitch'].item() > 0]
            
            if valid_notes:  # Only process if we have notes
                for i, note_info in enumerate(valid_notes):
                    pitch_idx = note_info['pitch'].item()
                    pitch = int(idx_to_note.get(str(pitch_idx), 60))
                    
                    # SIMPLE timing distribution - spread notes evenly across segment
                    note_position = i / len(valid_notes)  # 0 to 1
                    relative_start = note_info['start'].item() / 8.0  # 0 to 1
                    
                    # Blend even distribution with model prediction (favor even distribution)
                    blended_start = 0.7 * note_position + 0.3 * relative_start
                    
                    # Scale to segment duration with bounds checking
                    actual_start = segment_start_time + (blended_start * segment_duration)
                    
                    # Simple duration scaling - cap at reasonable values
                    relative_duration = min(1.0, note_info['duration'].item() / 4.0)
                    actual_duration = max(0.1, min(1.0, relative_duration * segment_duration * 0.2))
                    
                    segment_melody.append({
                        'pitch': pitch,
                        'duration': actual_duration,
                        'start_time': actual_start
                    })
            
            # Add segment melody to full melody
            if segment_idx == 0:
                # First segment - add all notes
                full_melody.extend(segment_melody)
            else:
                # Subsequent segments - simple overlap handling
                overlap_end_time = segment_start_time + (overlap * segment_duration / segment_length)
                
                for note in segment_melody:
                    if note['start_time'] < overlap_end_time:
                        # In overlap region - only add some notes to reduce density
                        if len(full_melody) == 0 or note['start_time'] > full_melody[-1]['start_time'] + 0.3:
                            full_melody.append(note)
                    else:
                        # Non-overlap region - add all notes
                        full_melody.append(note)
            
            print(f"Generated segment {segment_idx + 1} "
                f"(position: {current_position}-{segment_end}, focus: chord {focus_position}, "
                f"time: {segment_start_time:.1f}-{segment_start_time+segment_duration:.1f}s, "
                f"notes: {len(segment_melody)})")
            
            # Move to next segment
            current_position += step_size
            segment_idx += 1
            
            # Safety check to prevent infinite loops
            if segment_idx > 200:
                print("Warning: Too many segments generated, stopping")
                break
        
        # Sort melody by start time and apply final bounds check
        full_melody.sort(key=lambda x: x['start_time'])
        
        # CRITICAL: Ensure melody doesn't exceed expected duration
        if full_melody:
            max_allowed_time = total_song_duration * 1.1  # Allow 10% overage
            full_melody = [note for note in full_melody if note['start_time'] <= max_allowed_time]
            
            # If last note extends too far, trim it
            if full_melody and full_melody[-1]['start_time'] + full_melody[-1]['duration'] > max_allowed_time:
                full_melody[-1]['duration'] = max(0.1, max_allowed_time - full_melody[-1]['start_time'])
        
        print(f"Generated {len(full_melody)} notes for full song")
        if full_melody:
            generated_duration = max(note['start_time'] + note['duration'] for note in full_melody)
            print(f"Generated melody duration: {generated_duration:.1f} seconds")
            print(f"Expected song duration: {total_song_duration:.1f} seconds")
            coverage_ratio = generated_duration / total_song_duration if total_song_duration > 0 else 0
            print(f"Coverage ratio: {coverage_ratio:.2f}")
        
        return full_melody

    def generate_melody_from_chords(self, chord_sequence, vocab_path, target_density=0.6, 
                                density_window=8, temperature=1.0, chord_duration=2.0):
        """Generate melody for a single chord sequence with proper timing"""
        self.eval()
        
        # Move model to CPU for generation
        original_device = next(self.parameters()).device
        self.to('cpu')
        
        with open(vocab_path, 'r') as f:
            vocabularies = json.load(f)
            
        chord_to_idx = vocabularies['chord_vocab']
        idx_to_note = vocabularies['idx_to_note']
        
        # Convert chord sequence to indices
        chord_indices = []
        for chord in chord_sequence:
            chord_idx = chord_to_idx.get(chord, chord_to_idx.get('<UNK>', 0))
            chord_indices.append(chord_idx)
        
        # Pad to max length
        original_length = len(chord_indices)
        while len(chord_indices) < self.max_chord_length:
            chord_indices.append(chord_to_idx.get('<PAD>', 0))
        chord_indices = chord_indices[:self.max_chord_length]
        
        # Create tensors (on CPU)
        chord_tensor = torch.tensor([chord_indices], dtype=torch.long)
        chord_mask = torch.tensor([[True] * original_length + [False] * (self.max_chord_length - original_length)], 
                                dtype=torch.bool)
        focus_position = torch.tensor([original_length // 2], dtype=torch.long)
        
        # Calculate total duration
        total_duration = len(chord_sequence) * chord_duration
        
        # Modified generation with density control
        with torch.no_grad():
            generated_sequence = self._generate_with_density_control(
                chord_tensor, chord_mask, focus_position, 
                target_density, density_window, temperature
            )
        
        melody = []
        for i, note_info in enumerate(generated_sequence):
            pitch_idx = note_info['pitch'].item()
            if pitch_idx > 0:  # Skip rest notes
                pitch = int(idx_to_note.get(str(pitch_idx), 60))
                
                # Scale timing to total duration
                relative_start = note_info['start'].item()  # 0 to ~8
                relative_duration = note_info['duration'].item()  # 0 to ~4
                
                # Normalize and scale
                normalized_start = min(1.0, relative_start / 8.0)
                normalized_duration = min(1.0, relative_duration / 4.0)
                
                actual_start = normalized_start * total_duration
                actual_duration = max(0.1, normalized_duration * total_duration * 0.1)  # Max 10% of total duration
                
                melody.append({
                    'pitch': pitch,
                    'duration': actual_duration,
                    'start_time': actual_start
                })
        
        # Move model back to original device
        self.to(original_device)
        
        return melody
    
    
    def _generate_with_density_control(self, chord_tensor, chord_mask, focus_position, 
                             target_density=0.6, density_window=8, temperature=1.0):
        """
        Generate sequence with balanced density control
        """
        batch_size = chord_tensor.size(0)
        device = chord_tensor.device
        
        # Encode chords first (same as original)
        chord_embedded = self.chord_embedding(chord_tensor)
        chord_seq_len = chord_tensor.size(1)
        
        # Add positional encoding for chords
        chord_positions = torch.arange(chord_seq_len, device=device)
        chord_pos_emb = self.chord_position_embedding(chord_positions).unsqueeze(0).expand(batch_size, -1, -1)
        chord_embedded = chord_embedded + chord_pos_emb
        chord_embedded = self.dropout(chord_embedded)
        
        # Apply padding mask
        padding_mask = ~chord_mask
        
        # Encode chords
        chord_encoded = self.chord_encoder(chord_embedded, src_key_padding_mask=padding_mask)
        
        # Apply focus attention
        focus_context, _ = self.focus_attention(chord_encoded, chord_encoded, chord_encoded, key_padding_mask=padding_mask)
        chord_encoded = chord_encoded + self.focus_projection(focus_context)
        chord_encoded = self.layer_norm(chord_encoded)
        
        # Generation with balanced density control
        generated_sequence = []
        current_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Track recent decisions for density control
        recent_predictions = []  # Track recent note/rest decisions
        
        print(f"Generating with target density: {target_density:.2f}")
        
        for i in range(self.max_melody_length):
            # Embed current sequence
            melody_embedded = self.note_embedding(current_input)
            melody_seq_len = current_input.size(1)
            
            # Add positional encoding
            melody_positions = torch.arange(melody_seq_len, device=device)
            melody_pos_emb = self.position_embedding(melody_positions).unsqueeze(0).expand(batch_size, -1, -1)
            melody_embedded = melody_embedded + melody_pos_emb
            melody_embedded = self.dropout(melody_embedded)
            
            # Decode
            decoded = self.melody_decoder(melody_embedded, chord_encoded, memory_key_padding_mask=padding_mask)
            
            # Get predictions for the last position
            last_output = decoded[:, -1:]
            pitch_logits = self.pitch_head(last_output)
            duration_pred = self.duration_head(last_output).squeeze(-1)
            start_pred = self.start_head(last_output).squeeze(-1)
            
            # SIMPLIFIED DENSITY CONTROL - focus on main issue
            if len(recent_predictions) >= density_window:
                # Calculate recent density (ratio of notes to total positions)
                recent_notes = sum(1 for x in recent_predictions[-density_window:] if x > 0)
                current_density = recent_notes / density_window
                
                # Calculate density deviation
                density_deviation = current_density - target_density
                
                # Apply bias to encourage/discourage note generation
                density_bias = self._calculate_density_bias(density_deviation, temperature)
                
                # REDUCED temporal penalty - much gentler
                if len(recent_predictions) >= 3:
                    recent_note_count = sum(1 for x in recent_predictions[-3:] if x > 0)
                    if recent_note_count >= 3:  # 3 consecutive notes
                        temporal_bias = -1.0  # Gentle bias toward rest
                    elif recent_note_count == 0:  # 3 consecutive rests
                        temporal_bias = 0.5   # Gentle bias toward note
                    else:
                        temporal_bias = 0.0
                else:
                    temporal_bias = 0.0
                
                # Apply combined bias (much more conservative)
                total_bias = density_bias + temporal_bias
                pitch_logits = self._apply_density_bias(pitch_logits, total_bias)
                
                # Debug info
                if i % 8 == 0:
                    print(f"Step {i}: Density: {current_density:.3f}/{target_density:.3f}, "
                        f"Bias: {total_bias:.3f}")
            
            # Sample next note with temperature
            if temperature != 1.0:
                pitch_logits = pitch_logits / temperature
            
            # Use sampling instead of argmax for more variety
            pitch_probs = torch.softmax(pitch_logits.squeeze(1), dim=-1)
            next_pitch = torch.multinomial(pitch_probs, 1)
            
            # Track this prediction for density calculation
            recent_predictions.append(next_pitch.item())
            
            # Keep only recent predictions
            if len(recent_predictions) > density_window * 2:
                recent_predictions = recent_predictions[-density_window:]
            
            # Update input sequence
            current_input = torch.cat([current_input, next_pitch], dim=1)
            
            # Store generated note info
            generated_sequence.append({
                'pitch': next_pitch.squeeze(-1),
                'duration': duration_pred,
                'start': start_pred
            })
        
        return generated_sequence

    def _calculate_temporal_clustering_penalty(self, recent_start_times, current_start):
        """
        Calculate penalty for temporal clustering of notes
        
        Args:
            recent_start_times: List of recent note start times
            current_start: Predicted start time for current note
        
        Returns:
            penalty: Higher values discourage note generation (encourage rest)
        """
        if len(recent_start_times) < 2:
            return 0.0
        
        # Calculate how clustered the recent notes are
        time_diffs = []
        for i in range(1, len(recent_start_times)):
            diff = abs(recent_start_times[i] - recent_start_times[i-1])
            time_diffs.append(diff)
        
        # Add current prediction to the mix
        time_diffs.append(abs(current_start - recent_start_times[-1]))
        
        # If notes are very close together (small time differences), apply penalty
        avg_gap = sum(time_diffs) / len(time_diffs)
        min_gap = min(time_diffs)
        
        # Strong penalty if average gap is small OR if any gap is very small
        if avg_gap < 0.5:  # Average gap less than 500ms
            penalty = 2.0 - (avg_gap / 0.5)  # Scale from 0 to 2
        elif min_gap < 0.2:  # Any gap less than 200ms
            penalty = 1.5
        else:
            penalty = 0.0
        
        return penalty

    def _apply_temporal_spacing_bias(self, pitch_logits, temporal_bias):
        """
        Apply temporal spacing bias to encourage better note distribution
        
        Args:
            pitch_logits: [batch, seq_len, vocab_size] logits from model
            temporal_bias: Positive values discourage notes (encourage spacing)
        """
        if temporal_bias == 0.0:
            return pitch_logits
        
        # Clone logits to avoid in-place modification
        adjusted_logits = pitch_logits.clone()
        
        if temporal_bias > 0:  # Encourage spacing by discouraging notes
            # Boost rest token and reduce note tokens
            adjusted_logits[:, :, 0] += temporal_bias  # Boost rest
            adjusted_logits[:, :, 1:] -= temporal_bias * 0.5  # Reduce notes
        
        return adjusted_logits

    def _calculate_density_bias(self, density_deviation, temperature):
        """
        Calculate bias to apply to pitch logits based on density deviation
        
        Args:
            density_deviation: current_density - target_density
            temperature: Controls strength of bias
        
        Returns:
            bias: Positive = encourage notes, Negative = discourage notes
        """
        # Scale bias based on deviation magnitude
        bias_strength = 2.0 / temperature  # Stronger bias at lower temperatures
        
        if density_deviation > 0.15:  # Too many recent notes
            return -bias_strength  # Bias toward rest (pitch 0)
        elif density_deviation < -0.15:  # Too few recent notes
            return bias_strength   # Bias toward notes (pitch > 0)
        else:
            return 0.0  # No bias needed
        
    def _apply_density_bias(self, pitch_logits, bias):
        """
        Apply density bias to pitch logits
        
        Args:
            pitch_logits: [batch, seq_len, vocab_size] logits from model
            bias: Positive = encourage notes, Negative = encourage rest
        """
        if bias == 0.0:
            return pitch_logits
        
        # Clone logits to avoid in-place modification
        adjusted_logits = pitch_logits.clone()
        
        if bias > 0:  # Encourage notes
            # Boost all non-zero pitches
            adjusted_logits[:, :, 1:] += bias
        else:  # Encourage rest
            # Boost the rest token (pitch 0)
            adjusted_logits[:, :, 0] += abs(bias)
        
        return adjusted_logits
    
    def generate_chord_aligned_melody(self, chord_sequence, chord_times, vocab_path, temperature=1.0):
        """
        Generate melody that aligns with real song timing by processing chord-by-chord
        ENHANCED: Now uses timing enhancement methods
        
        Args:
            chord_sequence: List of chord symbols ['C:maj', 'Am:min', ...]
            chord_times: List of (start_time, end_time) tuples for each chord
            vocab_path: Path to vocabulary file
            temperature: Sampling temperature
        
        Returns:
            List of melody notes with absolute timing
        """
        self.eval()
        device = next(self.parameters()).device
        
        with open(vocab_path, 'r') as f:
            vocabularies = json.load(f)
        
        chord_to_idx = vocabularies['chord_vocab']
        idx_to_note = vocabularies['idx_to_note']
        
        # Convert chord sequence to indices
        chord_indices = []
        for chord in chord_sequence:
            chord_idx = chord_to_idx.get(chord, chord_to_idx.get('<UNK>', 0))
            chord_indices.append(chord_idx)
        
        # Limit to model's maximum chord length
        if len(chord_indices) > self.max_chord_length:
            print(f"Warning: Chord sequence too long, using sliding window approach")
            return self._generate_with_sliding_window(chord_sequence, chord_times, vocab_path, temperature)
        
        # Calculate BPM for better timing
        song_bpm = self._calculate_song_bpm(chord_times)
        beat_duration = 60.0 / (song_bpm * 4)  # 16th note duration
        
        # Prepare full chord context
        padded_chords = chord_indices + [chord_to_idx.get('<PAD>', 0)] * (self.max_chord_length - len(chord_indices))
        chord_mask = [True] * len(chord_indices) + [False] * (self.max_chord_length - len(chord_indices))
        
        # Calculate normalized chord durations
        chord_durations = [end - start for start, end in chord_times[:len(chord_indices)]]
        max_duration = max(chord_durations) if chord_durations else 4.0
        normalized_durations = [d / max_duration for d in chord_durations]
        normalized_durations += [0.0] * (self.max_chord_length - len(normalized_durations))
        
        # Generate melody for each chord
        full_melody = []
        
        print(f"Generating melody for {len(chord_indices)} chords...")
        
        for focus_idx in range(len(chord_indices)):
            chord_start, chord_end = chord_times[focus_idx]
            chord_duration = chord_end - chord_start
            current_chord = chord_sequence[focus_idx]
            
            print(f"  Chord {focus_idx + 1}/{len(chord_indices)}: {current_chord} ({chord_duration:.2f}s)")
            
            # Create tensors for this generation
            chord_tensor = torch.tensor([padded_chords], dtype=torch.long, device=device)
            chord_duration_tensor = torch.tensor([normalized_durations], dtype=torch.float32, device=device)
            chord_mask_tensor = torch.tensor([chord_mask], dtype=torch.bool, device=device)
            focus_tensor = torch.tensor([focus_idx], dtype=torch.long, device=device)
            target_duration_tensor = torch.tensor([chord_duration], dtype=torch.float32, device=device)
            
            # Generate melody notes for this specific chord
            with torch.no_grad():
                chord_melody = self._generate_single_chord_melody(
                    chord_tensor, 
                    chord_duration_tensor,
                    chord_mask_tensor, 
                    focus_tensor,
                    target_duration_tensor,
                    temperature
                )
            
            # Convert to absolute timing
            for note_info in chord_melody:
                pitch_idx = note_info['pitch'].item()
                if pitch_idx > 0:  # Skip rest notes
                    pitch = int(idx_to_note.get(str(pitch_idx), 60))
                    
                    # Convert relative timing to absolute timing
                    relative_start = note_info['start'].item()  # 0.0 to 1.0
                    relative_duration = note_info['duration'].item()  # 0.0 to 1.0
                    
                    absolute_start = chord_start + (relative_start * chord_duration)
                    absolute_duration = max(0.1, relative_duration * chord_duration)
                    
                    # Ensure note doesn't extend beyond chord boundary
                    if absolute_start + absolute_duration > chord_end:
                        absolute_duration = chord_end - absolute_start
                    
                    if absolute_duration > 0.05:  # Minimum note duration
                        full_melody.append({
                            'pitch': pitch,
                            'start_time': absolute_start,
                            'duration': absolute_duration,
                            'chord': current_chord,
                            'chord_index': focus_idx
                        })
        
        # Sort by start time and apply ENHANCED timing cleanup
        full_melody.sort(key=lambda x: x['start_time'])
        
        # ENHANCED: Use new timing enhancement instead of just removing overlaps
        cleaned_melody = self._enhance_note_timing_with_bpm(full_melody, beat_duration)
        
        print(f"Generated {len(cleaned_melody)} notes total")
        return cleaned_melody

    def _generate_single_chord_melody(self, chord_tensor, chord_durations, chord_mask, focus_position, target_duration, temperature):
        """Generate melody for a single chord using the trained model"""
        
        # This uses the same generation logic as your current model but focuses on one chord
        batch_size = chord_tensor.size(0)
        device = chord_tensor.device
        
        # Encode chord context (same as your current model)
        chord_embedded = self.chord_embedding(chord_tensor)
        chord_seq_len = chord_tensor.size(1)
        
        chord_positions = torch.arange(chord_seq_len, device=device)
        chord_pos_emb = self.chord_position_embedding(chord_positions).unsqueeze(0).expand(batch_size, -1, -1)
        chord_embedded = chord_embedded + chord_pos_emb
        chord_embedded = self.dropout(chord_embedded)
        
        padding_mask = ~chord_mask
        chord_encoded = self.chord_encoder(chord_embedded, src_key_padding_mask=padding_mask)
        
        focus_context, _ = self.focus_attention(chord_encoded, chord_encoded, chord_encoded, key_padding_mask=padding_mask)
        chord_encoded = chord_encoded + self.focus_projection(focus_context)
        chord_encoded = self.layer_norm(chord_encoded)
        
        # Generate melody (adapted from your existing generation)
        generated_sequence = []
        current_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Reduce max length for single chord generation
        max_notes_per_chord = 8  # Reasonable limit for one chord
        
        for i in range(max_notes_per_chord):
            melody_embedded = self.note_embedding(current_input)
            melody_seq_len = current_input.size(1)
            
            melody_positions = torch.arange(melody_seq_len, device=device)
            melody_pos_emb = self.position_embedding(melody_positions).unsqueeze(0).expand(batch_size, -1, -1)
            melody_embedded = melody_embedded + melody_pos_emb
            melody_embedded = self.dropout(melody_embedded)
            
            decoded = self.melody_decoder(melody_embedded, chord_encoded, memory_key_padding_mask=padding_mask)
            
            last_output = decoded[:, -1:]
            pitch_logits = self.pitch_head(last_output)
            duration_pred = self.duration_head(last_output).squeeze(-1)
            start_pred = self.start_head(last_output).squeeze(-1)
            
            # Apply temperature
            if temperature != 1.0:
                pitch_logits = pitch_logits / temperature
            
            pitch_probs = torch.softmax(pitch_logits.squeeze(1), dim=-1)
            next_pitch = torch.multinomial(pitch_probs, 1)
            
            current_input = torch.cat([current_input, next_pitch], dim=1)
            
            generated_sequence.append({
                'pitch': next_pitch.squeeze(-1),
                'duration': torch.clamp(duration_pred, 0.0, 1.0),  # Clamp to 0-1 range
                'start': torch.clamp(start_pred, 0.0, 1.0)
            })
            
            # Early stopping if we generate a rest
            if next_pitch.item() == 0:
                break
        
        return generated_sequence

    def _remove_overlapping_notes(self, melody):
        """Remove overlapping notes to create clean melody line"""
        if len(melody) <= 1:
            return melody
        
        cleaned = [melody[0]]
        
        for note in melody[1:]:
            last_note = cleaned[-1]
            
            # If notes overlap, keep the one that starts first
            if note['start_time'] < last_note['start_time'] + last_note['duration']:
                # Shorten the previous note
                if last_note['start_time'] + 0.1 < note['start_time']:
                    last_note['duration'] = note['start_time'] - last_note['start_time']
                    cleaned.append(note)
                # Otherwise skip the new note if it's too close
            else:
                cleaned.append(note)
        
        return cleaned

    def _generate_with_sliding_window(self, chord_sequence, chord_times, vocab_path, temperature):
        """Handle very long chord sequences with sliding window approach"""
        print(f"Using sliding window for {len(chord_sequence)} chords")
        
        window_size = self.max_chord_length - 2  # Leave room for context (10 for max_chord_length=12)
        overlap = window_size // 3  # 1/3 overlap (3-4 chords)
        full_melody = []
        
        num_windows = (len(chord_sequence) + window_size - overlap - 1) // (window_size - overlap)
        print(f"Processing {num_windows} windows (window_size={window_size}, overlap={overlap})")
        
        for window_idx in range(num_windows):
            start_idx = window_idx * (window_size - overlap)
            end_idx = min(start_idx + window_size, len(chord_sequence))
            
            print(f"  Window {window_idx + 1}/{num_windows}: chords {start_idx}-{end_idx} ({end_idx - start_idx} chords)")
            
            # Extract window
            window_chords = chord_sequence[start_idx:end_idx]
            window_times = chord_times[start_idx:end_idx]
            
            # Adjust timing to start from 0 for this window
            if window_times:
                time_offset = window_times[0][0]
                adjusted_times = [(start - time_offset, end - time_offset) for start, end in window_times]
            else:
                adjusted_times = []
            
            # Generate for this window (recursive call with smaller sequence)
            if len(window_chords) <= self.max_chord_length:
                window_melody = self._generate_window_melody(
                    window_chords, 
                    adjusted_times, 
                    vocab_path, 
                    temperature,
                    time_offset
                )
            else:
                # If still too long, truncate
                window_chords = window_chords[:self.max_chord_length]
                adjusted_times = adjusted_times[:self.max_chord_length]
                window_melody = self._generate_window_melody(
                    window_chords, 
                    adjusted_times, 
                    vocab_path, 
                    temperature,
                    time_offset
                )
            
            # Handle overlap - only add non-overlapping notes
            if window_idx == 0:
                # First window - add all notes
                full_melody.extend(window_melody)
                print(f"    Added {len(window_melody)} notes from first window")
            else:
                # Subsequent windows - filter overlap region
                overlap_chord_count = overlap
                if start_idx + overlap_chord_count < len(chord_times):
                    overlap_end_time = chord_times[start_idx + overlap_chord_count][0]
                else:
                    overlap_end_time = chord_times[-1][1]  # End of song
                
                new_notes = 0
                for note in window_melody:
                    if note['start_time'] >= overlap_end_time:
                        full_melody.append(note)
                        new_notes += 1
                
                print(f"    Added {new_notes} notes from window {window_idx + 1} (after overlap filtering)")
        
        print(f"Sliding window complete: {len(full_melody)} total notes")
        return full_melody
    def _generate_window_melody(self, chord_sequence, chord_times, vocab_path, temperature, time_offset):
        """
        Generate melody for a single window
        ENHANCED: Apply timing enhancement to window melody
        """
        
        with open(vocab_path, 'r') as f:
            vocabularies = json.load(f)
        
        chord_to_idx = vocabularies['chord_vocab']
        idx_to_note = vocabularies['idx_to_note']
        device = next(self.parameters()).device
        
        # Convert chord sequence to indices
        chord_indices = []
        for chord in chord_sequence:
            chord_idx = chord_to_idx.get(chord, chord_to_idx.get('<UNK>', 0))
            chord_indices.append(chord_idx)
        
        # Prepare model inputs
        original_length = len(chord_indices)
        padded_chords = chord_indices + [chord_to_idx.get('<PAD>', 0)] * (self.max_chord_length - len(chord_indices))
        chord_mask = [True] * original_length + [False] * (self.max_chord_length - original_length)
        
        # Calculate normalized chord durations
        chord_durations = [end - start for start, end in chord_times[:original_length]]
        max_duration = max(chord_durations) if chord_durations else 4.0
        normalized_durations = [d / max_duration for d in chord_durations]
        normalized_durations += [0.0] * (self.max_chord_length - len(normalized_durations))
        
        # Generate melody for each chord
        window_melody = []
        
        for focus_idx in range(original_length):
            if focus_idx >= len(chord_times):
                break
                
            chord_start, chord_end = chord_times[focus_idx]
            chord_duration = chord_end - chord_start
            current_chord = chord_sequence[focus_idx]
            
            # Create tensors for this generation
            chord_tensor = torch.tensor([padded_chords], dtype=torch.long, device=device)
            chord_duration_tensor = torch.tensor([normalized_durations], dtype=torch.float32, device=device)
            chord_mask_tensor = torch.tensor([chord_mask], dtype=torch.bool, device=device)
            focus_tensor = torch.tensor([focus_idx], dtype=torch.long, device=device)
            target_duration_tensor = torch.tensor([chord_duration], dtype=torch.float32, device=device)
            
            # Generate melody notes for this specific chord
            with torch.no_grad():
                chord_melody = self._generate_single_chord_melody(
                    chord_tensor, 
                    chord_duration_tensor,
                    chord_mask_tensor, 
                    focus_tensor,
                    target_duration_tensor,
                    temperature
                )
            
            # Convert to absolute timing (add back the time offset)
            for note_info in chord_melody:
                pitch_idx = note_info['pitch'].item()
                if pitch_idx > 0:  # Skip rest notes
                    pitch = int(idx_to_note.get(str(pitch_idx), 60))
                    
                    # Convert relative timing to absolute timing
                    relative_start = max(0.0, min(1.0, note_info['start'].item()))
                    relative_duration = max(0.0, min(1.0, note_info['duration'].item()))
                    
                    absolute_start = time_offset + chord_start + (relative_start * chord_duration)
                    absolute_duration = max(0.1, relative_duration * chord_duration * 0.5)
                    
                    # Ensure note doesn't extend beyond chord boundary
                    chord_end_absolute = time_offset + chord_end
                    if absolute_start + absolute_duration > chord_end_absolute:
                        absolute_duration = max(0.1, chord_end_absolute - absolute_start)
                    
                    if absolute_duration > 0.05:  # Minimum note duration
                        window_melody.append({
                            'pitch': pitch,
                            'start_time': absolute_start,
                            'duration': absolute_duration,
                            'chord': current_chord,
                            'chord_index': focus_idx
                        })
        
        # ENHANCED: Apply timing enhancement to window melody
        enhanced_melody = self._enhance_note_timing(window_melody)
        
        return enhanced_melody
