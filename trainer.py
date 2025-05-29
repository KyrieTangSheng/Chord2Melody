import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time
class ChordToMelodyTrainer:
    def __init__(self, model, train_loader, val_loader, device='mps'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.pitch_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.duration_criterion = nn.MSELoss()
        self.start_criterion = nn.MSELoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            chord_input = batch['chord_input'].to(self.device)
            melody_pitch = batch['melody_pitch'].to(self.device)
            melody_duration = batch['melody_duration'].to(self.device)
            melody_start = batch['melody_start'].to(self.device)
            
            self.optimizer.zero_grad()
            
            pitch_logits, duration_pred, start_pred = self.model(
                chord_input, melody_pitch, training=True
            )
            
            # Calculate losses
            # Shift targets for next-token prediction
            target_pitch = melody_pitch[:, 1:]  # Remove first token
            target_duration = melody_duration[:, :-1]  # Remove last token
            target_start = melody_start[:, :-1]  # Remove last token
            
            pitch_loss = self.pitch_criterion(
                pitch_logits.reshape(-1, pitch_logits.size(-1)), 
                target_pitch.reshape(-1)
            )
            duration_loss = self.duration_criterion(duration_pred, target_duration)
            start_loss = self.start_criterion(start_pred, target_start)
            
            # Combined loss
            total_batch_loss = pitch_loss + 0.5 * duration_loss + 0.5 * start_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                chord_input = batch['chord_input'].to(self.device)
                melody_pitch = batch['melody_pitch'].to(self.device)
                melody_duration = batch['melody_duration'].to(self.device)
                melody_start = batch['melody_start'].to(self.device)
                
                # Forward pass
                pitch_logits, duration_pred, start_pred = self.model(
                    chord_input, melody_pitch, training=True
                )
                
                # Calculate losses
                target_pitch = melody_pitch[:, 1:]
                target_duration = melody_duration[:, :-1]
                target_start = melody_start[:, :-1]
                
                pitch_loss = self.pitch_criterion(
                    pitch_logits.reshape(-1, pitch_logits.size(-1)), 
                    target_pitch.reshape(-1)
                )
                duration_loss = self.duration_criterion(duration_pred, target_duration)
                start_loss = self.start_criterion(start_pred, target_start)
                
                total_batch_loss = pitch_loss + 0.5 * duration_loss + 0.5 * start_loss
                total_loss += total_batch_loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Save best model
            if epoch == 0 or val_loss < min(self.val_losses[:-1]):
                torch.save(self.model.state_dict(), 'best_chord_melody_model.pt')
                print("New best model saved!")
        
        end_time = time.time()
        training_time_in_minutes = (end_time - start_time) // 60
        print(f"Training completed in {training_time_in_minutes} minutes")
        
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Chord-to-Melody Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')