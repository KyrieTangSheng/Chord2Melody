import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import json

class ChordToMelodyTransformer(nn.Module):
    def __init__(self, 
                chord_vocab_size:int,
                note_vocab_size:int,
                d_model:int = 256,
                nhead:int = 8,
                num_layers:int = 6,
                max_melody_length:int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.max_melody_length = max_melody_length
        
        self.chord_embedding = nn.Embedding(chord_vocab_size, d_model)
        self.note_embedding = nn.Embedding(note_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_melody_length, d_model)
        
        self.chord_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers//2
        )
        
        self.melody_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers//2
        )
        
        self.pitch_head = nn.Linear(d_model, note_vocab_size)
        self.duration_head = nn.Linear(d_model, 1)
        self.start_head = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, chord_input, melody_pitch = None, training = True):
        batch_size = chord_input.size(0)
        
        chord_embedded = self.chord_embedding(chord_input)
        chord_encoded = self.chord_encoder(chord_embedded)
        
        
        if training:
            seq_len = melody_pitch.size(1)
            
            target_melody = melody_pitch[:, :-1]  # Remove last token
            
            melody_embedded = self.note_embedding(target_melody)
            positions = torch.arange(target_melody.size(1), device=target_melody.device)
            pos_embedded = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
            melody_embedded = melody_embedded + pos_embedded
            melody_embedded = self.dropout(melody_embedded)

            seq_len = target_melody.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(target_melody.device)
            
            decoded = self.melody_decoder(melody_embedded, chord_encoded, tgt_mask=causal_mask)
            
            pitch_logits = self.pitch_head(decoded)
            duration_pred = self.duration_head(decoded).squeeze(-1)
            start_pred = self.start_head(decoded).squeeze(-1)
            
            return pitch_logits, duration_pred, start_pred
        else:
            generated_sequence = []
            current_input = torch.zeros(batch_size, 1, dtype=torch.long, device=chord_input.device)
            
            for i in range(self.max_melody_length):
                # Embed current sequence
                melody_embedded = self.note_embedding(current_input)
                positions = torch.arange(current_input.size(1), device=current_input.device)
                pos_embedded = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
                melody_embedded = melody_embedded + pos_embedded
                melody_embedded = self.dropout(melody_embedded)
                
                # Decode
                decoded = self.melody_decoder(melody_embedded, chord_encoded)
                
                # Predict next token
                pitch_logits = self.pitch_head(decoded[:, -1:])  # Last position only
                duration_pred = self.duration_head(decoded[:, -1:]).squeeze(-1)
                start_pred = self.start_head(decoded[:, -1:]).squeeze(-1)
                
                # Sample next note
                next_pitch = torch.argmax(pitch_logits, dim=-1)
                
                # Append to sequence
                current_input = torch.cat([current_input, next_pitch], dim=1)
                
                generated_sequence.append({
                    'pitch': next_pitch.squeeze(-1),
                    'duration': duration_pred,
                    'start': start_pred
                })
                          
            return generated_sequence
        
    def generate_melody_from_chords(self, chord_sequence, vocab_path):
        self.eval()
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        with open(vocab_path, 'r') as f:
            vocabularies = json.load(f)
            
        chord_to_idx = vocabularies['chord_vocab']
        idx_to_note = vocabularies['note_vocab']
        
        chord_input = []
        for chord in chord_sequence:
            chord_idx = chord_to_idx.get(chord, chord_to_idx.get('<UNK>', 0))
            chord_input.append(chord_idx)
        
        while len(chord_input) < 10: # NOTE: this can be changed
            chord_input.append(chord_to_idx.get('<PAD>', 0))
        
        chord_tensor = torch.tensor([chord_input], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated_sequence = self(chord_tensor, training=False)
        
        melody = []
        for note_info in generated_sequence:
            pitch_idx = note_info['pitch'].item()
            pitch = idx_to_note.get(str(pitch_idx), 60)  # Default to middle C
            duration = note_info['duration'].item()
            start = note_info['start'].item()
            
            if pitch > 0:  # Skip rest notes
                melody.append({
                    'pitch': pitch,
                    'duration': max(0.1, duration),  # Minimum duration
                    'start_time': max(0.0, start)
                })
        
        return melody
            
        
            