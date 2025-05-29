import torch
import json
from data_processor import process_dataset
from dataset import ChordMelodyDataset
from torch.utils.data import DataLoader
from model import ChordToMelodyTransformer
from trainer import ChordToMelodyTrainer
from utils import generate_midi_from_melody
import argparse
def main(args):
    print("Task2 - Chord to Melody Generation")
    
    # process dataset
    # dataset_path, output_path = "POP909-Dataset", "processed_pop909_chord_melody"
    # process_dataset(dataset_path, output_path)
    
    # choose device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load vocabularies & dataset
    data_path = "processed_pop909_chord_melody/training_sequences.pkl"
    vocab_path = "processed_pop909_chord_melody/vocabularies.json"
    full_dataset = ChordMelodyDataset(data_path, vocab_path)
        
    # split dataset into train / val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
        
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
    # create model
    model = ChordToMelodyTransformer(
        chord_vocab_size=len(full_dataset.chord_to_idx),
        note_vocab_size=len(full_dataset.note_to_idx),
        d_model=256,
        nhead=8,
        num_layers=6
    )
        
    if args.train:
        # create trainer
        trainer = ChordToMelodyTrainer(model, train_loader, val_loader, device)
        # train model
        trainer.train(num_epochs=100)
        # plot training history
        trainer.plot_training_history()
    else:
        # load model
        model.load_state_dict(torch.load('best_chord_melody_model.pt', map_location=device))
        model = model.to(device)
    # generate sample melody
    example_chords = ['C:7', 'C:7', 'F:7', 'G:7','C:7', 'C:7', 'F:7', 'G:7', 'C:7', 'C:7']
    generated_melody = model.generate_melody_from_chords(example_chords, vocab_path)
    # save melody as midi
    output_path = "generated_melody.mid"
    generate_midi_from_melody(generated_melody, output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chord to Melody Generation Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()
    print(args.train)
    main(args)