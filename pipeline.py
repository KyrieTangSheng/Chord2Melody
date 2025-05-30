import torch
import json
from data_processor import process_dataset
from dataset import ChordMelodyDataset
from torch.utils.data import DataLoader
from model import AttentionChordToMelodyTransformer
from trainer import ChordToMelodyTrainer
from utils import generate_midi_from_melody
from pathlib import Path
import argparse
import os
import pretty_midi

def main(args):
    print("Chord to Melody Generation - Attention-Based Global Context Approach")
    print("=" * 70)
    
    # Process dataset if needed
    if args.process_data:
        print("Processing dataset...")
        dataset_path, output_path = "POP909-Dataset", "processed_pop909_chord_melody"
        process_dataset(dataset_path, output_path, melody_segment_length=32)
        print("Dataset processing completed!")
        return
    
    # Choose device - always use CPU for generation due to MPS limitations
    training_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    generation_device = torch.device('cpu')
    print(f"Using device: {training_device} (training) / {generation_device} (generation)")
    
    # Load vocabularies & dataset
    data_path = "processed_pop909_chord_melody/training_sequences.pkl"
    vocab_path = "processed_pop909_chord_melody/vocabularies.json"
    
    try:
        full_dataset = ChordMelodyDataset(data_path, vocab_path, max_chord_length=12, max_melody_length=16)
        print(f"Dataset loaded successfully!")
    except FileNotFoundError:
        print("Processed data not found. Please run with --process_data flag first.")
        return
        
    # Split dataset into train / val
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
    # Create dataloaders
    batch_size = 8 if training_device.type == 'mps' else 16  # Smaller batch for MPS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
    # Create model
    model = AttentionChordToMelodyTransformer(
        chord_vocab_size=len(full_dataset.chord_to_idx),
        note_vocab_size=len(full_dataset.note_to_idx),
        d_model=256,
        nhead=8,
        num_layers=6,
        max_chord_length=12
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    if args.train:
        # Move model to training device
        model = model.to(training_device)
        # Create trainer and train
        trainer = ChordToMelodyTrainer(model, train_loader, val_loader, training_device)
        trainer.train(num_epochs=args.epochs)
        trainer.plot_training_history()
        print("Training completed!")
    else:
        # Load trained model
        try:
            model.load_state_dict(torch.load('best_chord_melody_model.pt', map_location=generation_device))
            model = model.to(generation_device)
            print("Loaded trained model successfully!")
        except FileNotFoundError:
            print("No trained model found. Please train first with --train flag.")
            return
    
    # Move model to CPU for generation
    model = model.to(generation_device)
    
    # Generate sample melodies
    print("\nGENERATING SAMPLE MELODY")
    print("="*70)
    
    # Test with real POP909 song
    if args.test_real_song:
        print("\nReal POP909 song with chord-aligned timing:")
        try:
            with open("processed_pop909_chord_melody/chord_melody_data.json", 'r') as f:
                song_data = json.load(f)
            
            if song_data:
                test_song = song_data[300]  # Take first song
                real_chords = test_song['full_chord_sequence']
                song_id = test_song['song_id']
                
                # Path to original MIDI file
                original_midi_path = f"POP909-Dataset/POP909/{song_id}/{song_id}.mid"
                
                print(f"Using song: {song_id}")
                print(f"Chord progression: {len(real_chords)} chords")
                print(f"Chords: {' | '.join(real_chords[:8])}{'...' if len(real_chords) > 8 else ''}")
                
                # DEBUG: Check original MIDI properties
                try:
                    original_midi = pretty_midi.PrettyMIDI(original_midi_path)
                    original_duration = original_midi.get_end_time()
                    original_tempo = original_midi.estimate_tempo()
                    
                    print(f"\nOriginal MIDI Analysis:")
                    print(f"  Duration: {original_duration:.1f}s")
                    print(f"  Estimated tempo: {original_tempo:.1f} BPM")
                    print(f"  Instruments: {len(original_midi.instruments)}")
                    
                    for i, inst in enumerate(original_midi.instruments):
                        inst_duration = max([n.end for n in inst.notes]) if inst.notes else 0
                        print(f"    {i}: {inst.name} - {len(inst.notes)} notes, {inst_duration:.1f}s")
                        
                except Exception as e:
                    print(f"Could not analyze original MIDI: {e}")
                    original_duration = len(real_chords) * 2.0  # Fallback
                
                # Extract chord timing from original MIDI
                print(f"\nExtracting chord timing...")
                if original_midi_path and Path(original_midi_path).exists():
                    from utils import extract_chord_timing_from_midi
                    chord_times = extract_chord_timing_from_midi(
                        original_midi_path, 
                        target_chord_count=len(real_chords)
                    )
                else:
                    print("  Original MIDI not found, using estimated timing")
                    chord_times = [(i * 2.0, (i + 1) * 2.0, i) for i in range(len(real_chords))]
                
                if chord_times:
                    extracted_duration = chord_times[-1][1] - chord_times[0][0]
                    print(f"  Extracted {len(chord_times)} chord segments")
                    print(f"  Extracted duration: {extracted_duration:.1f}s")
                    print(f"  Average chord duration: {extracted_duration/len(chord_times):.1f}s")
                    
                    # Show first few chord timings
                    print(f"  First few chords:")
                    for i, (start, end, idx) in enumerate(chord_times[:5]):
                        duration = end - start
                        chord = real_chords[i] if i < len(real_chords) else "N/A"
                        print(f"    {i+1}. {start:5.1f}s-{end:5.1f}s ({duration:4.1f}s) -> {chord}")
                else:
                    print("  Failed to extract timing, using fallback")
                    chord_times = [(i * 2.0, (i + 1) * 2.0, i) for i in range(len(real_chords))]
                
                # Generate melody with chord-aligned timing
                print(f"\nGenerating chord-aligned melody...")
                
                # Move model to CPU for generation (avoid MPS issues)
                model = model.to(generation_device)
                
                try:
                    # Generate using the chord-aligned method
                    generated_melody = model.generate_chord_aligned_melody(
                        chord_sequence=real_chords,
                        chord_times=[(start, end) for start, end, _ in chord_times],  # Remove index
                        vocab_path=vocab_path,
                        temperature=1.1
                    )
                    
                    if generated_melody:
                        print(f"✓ Generated {len(generated_melody)} notes")
                        
                        # Calculate generated duration
                        generated_duration = max(note['start_time'] + note['duration'] for note in generated_melody)
                        expected_duration = chord_times[-1][1] if chord_times else len(real_chords) * 2.0
                        
                        print(f"  Generated duration: {generated_duration:.1f}s")
                        print(f"  Expected duration: {expected_duration:.1f}s")
                        print(f"  Coverage ratio: {generated_duration/expected_duration:.2f}")
                        
                        # Save generated melody with original tracks
                        os.makedirs("generated_real_song_aligned", exist_ok=True)
                        output_path = f"generated_real_song_aligned/{song_id}.mid"
                        
                        try:
                            generate_midi_from_melody(
                                generated_melody, 
                                output_path,
                                tempo=original_tempo if 'original_tempo' in locals() else 120,
                                original_midi_path=original_midi_path
                            )
                            print(f"✓ Saved to {output_path}")
                            
                        except Exception as e:
                            print(f"  Error saving MIDI: {e}")
                            # Fallback: save without original tracks
                            simple_output = f"generated_melody_only_{song_id}.mid"
                            generate_midi_from_melody(generated_melody, simple_output)
                            print(f"  Saved melody-only version to {simple_output}")
                        
                        # Performance comparison
                        if 'original_duration' in locals():
                            print(f"\nTiming Comparison:")
                            print(f"  Original MIDI:     {original_duration:.1f}s")
                            print(f"  Generated melody:  {generated_duration:.1f}s")
                            print(f"  Timing accuracy:   {1 - abs(generated_duration - original_duration)/original_duration:.1%}")
                            
                            if abs(generated_duration - original_duration) / original_duration < 0.1:
                                print(f"  ✅ Excellent timing alignment!")
                            elif abs(generated_duration - original_duration) / original_duration < 0.2:
                                print(f"  ✓ Good timing alignment")
                            else:
                                print(f"  ⚠️ Timing could be improved")
                    
                    else:
                        print("  ❌ No melody generated")
                        
                except Exception as e:
                    print(f"  ❌ Generation failed: {e}")
                    print(f"  This might be because the chord-aligned generation method isn't implemented yet")
                    
            else:
                print("No song data found in processed files")
                    
        except FileNotFoundError:
            print("Processed song data not found. Please run with --process_data flag first.")
        except Exception as e:
            print(f"Error in real song test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention-Based Chord to Melody Generation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--process_data", action="store_true", help="Process the dataset")
    parser.add_argument("--test_real_song", action="store_true", help="Test with real song chord progression")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()
    
    main(args)