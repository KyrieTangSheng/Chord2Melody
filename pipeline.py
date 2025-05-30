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
        full_dataset = ChordMelodyDataset(data_path, vocab_path, max_chord_length=80)
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
        max_chord_length=80
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
    print("\n" + "="*70)
    print("GENERATING SAMPLE MELODIES")
    print("="*70)
    
    # Updated pipeline usage with proper timing

    # Example 1: Simple chord progression with fixed timing
    print("\n1. Simple chord progression with proper timing:")
    simple_chords = ['C:maj', 'Am:min', 'F:maj', 'G:maj', 'C:maj', 'Am:min', 'F:maj', 'G:maj']
    print(f"Chords: {' | '.join(simple_chords)}")

    simple_melody = model.generate_melody_from_chords(
        simple_chords, 
        vocab_path, 
        chord_duration=2.0  # 2 seconds per chord
    )

    if simple_melody:
        # Fix timing before saving
        from utils import fix_melody_timing
        fixed_melody = fix_melody_timing(simple_melody, simple_chords, chords_per_measure=4)
        
        output_path = "generated_simple_melody.mid"
        generate_midi_from_melody(fixed_melody, output_path)
        print(f"✓ Simple melody saved to {output_path} ({len(fixed_melody)} notes)")

    # Example 2: Full song with real timing reference
    print("\n2. Full song with original MIDI timing:")
    long_chords = [
        # 12-bar blues progression repeated ~4 times with variations
        'C:7', 'C:7', 'C:7', 'C:7',
        'F:7', 'F:7', 'C:7', 'C:7', 
        'G:7', 'F:7', 'C:7', 'G:7',
        
        'C:7', 'C:7', 'C:7', 'C:7',
        'F:7', 'F:7', 'C:7', 'C:7',
        'G:7', 'F:7', 'C:7', 'C:7',
        
        'C:7', 'C:7', 'C:7', 'C:7', 
        'F:7', 'F:7', 'C:7', 'C:7',
        'G:7', 'F:7', 'C:7', 'G:7',
        
        'C:7', 'C:7', 'C:7', 'C:7',
        'F:7', 'F:7', 'C:7', 'C:7',
        'G:7', 'F:7', 'C:7', 'C:7',
        
        # Final variation with some extra color
        'C:7', 'C:7', 'C:7', 'C:7',
        'F:7', 'F:7', 'C:7', 'C:7',
        'G:7', 'F:7', 'C:7', 'C:7'
    ]

    full_song_melody = model.generate_full_song_melody(
        long_chords, 
        vocab_path, 
        segment_length=16,
        overlap=4,
    )

    if full_song_melody:
        output_path = "generated_full_song_proper_timing.mid"
        generate_midi_from_melody(
            full_song_melody, 
            output_path, 
        )
        print(f"✓ Full song saved to {output_path} ({len(full_song_melody)} notes)")

    # Example 3: Test with real POP909 song
    if args.test_real_song:
        # Create output directory for real song generations if it doesn't exist
        real_song_output_dir = Path("generated_real_song")
        real_song_output_dir.mkdir(exist_ok=True)
        print("\n3. Real POP909 song with proper timing:")
        try:
            with open("processed_pop909_chord_melody/chord_melody_data.json", 'r') as f:
                song_data = json.load(f)
            
            if song_data:
                test_song = song_data[302]  # Take first song
                real_chords = test_song['full_chord_sequence']
                song_id = test_song['song_id']
                
                # Path to original MIDI file
                original_midi_path = f"POP909-Dataset/POP909/{song_id}/{song_id}.mid"
                
                print(f"Using song: {song_id} ({len(real_chords)} chords)")
                
                real_melody = model.generate_full_song_melody(
                    real_chords, 
                    vocab_path,
                    segment_length=12,
                    overlap=3,
                    original_midi_path=original_midi_path
                )
                
                if real_melody:
                    output_path = f"generated_real_song/{song_id}.mid"
                    generate_midi_from_melody(
                        real_melody, 
                        output_path,
                        original_midi_path=original_midi_path
                    )
                    print(f"✓ Real song melody saved to {output_path}")
                    
                    # Compare durations
                    try:
                        original_midi = pretty_midi.PrettyMIDI(original_midi_path)
                        original_duration = original_midi.get_end_time()
                        generated_duration = max(note['start_time'] + note['duration'] for note in real_melody)
                        print(f"Duration comparison: Original {original_duration:.1f}s vs Generated {generated_duration:.1f}s")
                    except:
                        print("Could not compare durations")
                
        except FileNotFoundError:
            print("Processed song data not found. Skipping real song test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention-Based Chord to Melody Generation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--process_data", action="store_true", help="Process the dataset")
    parser.add_argument("--test_real_song", action="store_true", help="Test with real song chord progression")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()
    
    main(args)