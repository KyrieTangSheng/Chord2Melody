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
        
        # Simple loss functions
        self.pitch_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.duration_criterion = nn.MSELoss()
        self.start_criterion = nn.MSELoss()
        
        # Simple optimizer setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=3,
            factor=0.5,
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def calculate_accuracy(self, pitch_logits, target_pitch, mask):
        """Calculate pitch prediction accuracy for non-padded positions"""
        with torch.no_grad():
            pred_pitch = torch.argmax(pitch_logits, dim=-1)
            target_flat = target_pitch.reshape(-1)
            pred_flat = pred_pitch.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            if mask_flat.sum() > 0:
                correct = (pred_flat == target_flat) & mask_flat
                accuracy = correct.sum().float() / mask_flat.sum().float()
                return accuracy.item()
            return 0.0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device - updated for new dataset structure
            full_chord_sequence = batch['full_chord_sequence'].to(self.device)
            chord_durations = batch['chord_durations'].to(self.device)  # NEW
            chord_mask = batch['chord_mask'].to(self.device)
            focus_positions = batch['focus_position'].to(self.device)
            target_chord_duration = batch['target_chord_duration'].to(self.device)  # NEW
            melody_pitch = batch['melody_pitch'].to(self.device)
            melody_duration = batch['melody_duration'].to(self.device)
            melody_start = batch['melody_start'].to(self.device)
            melody_mask = batch['melody_mask'].to(self.device)
            
            # REMOVED: segment_position is not in the new dataset
            
            self.optimizer.zero_grad()
            
            # Forward pass - updated for new model inputs
            pitch_logits, duration_pred, start_pred = self.model(
                full_chord_sequence,
                chord_mask,
                focus_positions,
                chord_durations=chord_durations,  # NEW: pass chord durations
                melody_pitch=melody_pitch,
                training=True
            )
            
            # Prepare targets (shift for next-token prediction)
            target_pitch = melody_pitch[:, 1:]  # Remove first token
            target_duration = melody_duration[:, :-1]  # Remove last token
            target_start = melody_start[:, :-1]  # Remove last token
            target_mask = melody_mask[:, :-1]  # Remove last token
            
            # Calculate losses
            pitch_loss = self.pitch_criterion(
                pitch_logits.reshape(-1, pitch_logits.size(-1)), 
                target_pitch.reshape(-1)
            )
            
            # Only compute regression losses on non-padded positions
            if target_mask.sum() > 0:
                duration_loss = self.duration_criterion(
                    duration_pred[target_mask], 
                    target_duration[target_mask]
                )
                start_loss = self.start_criterion(
                    start_pred[target_mask], 
                    target_start[target_mask]
                )
            else:
                duration_loss = torch.tensor(0.0, device=self.device)
                start_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss (pitch is most important)
            total_batch_loss = pitch_loss + 0.3 * duration_loss + 0.3 * start_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(pitch_logits, target_pitch, target_mask)
            
            total_loss += total_batch_loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Print progress occasionally
            if batch_idx % 50 == 0 and batch_idx > 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = total_accuracy / (batch_idx + 1)
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss {current_loss:.4f}, Acc {current_acc:.3f}")
        
        return total_loss / num_batches, total_accuracy / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device - updated for new dataset structure
                full_chord_sequence = batch['full_chord_sequence'].to(self.device)
                chord_durations = batch['chord_durations'].to(self.device)  # NEW
                chord_mask = batch['chord_mask'].to(self.device)
                focus_positions = batch['focus_position'].to(self.device)
                target_chord_duration = batch['target_chord_duration'].to(self.device)  # NEW
                melody_pitch = batch['melody_pitch'].to(self.device)
                melody_duration = batch['melody_duration'].to(self.device)
                melody_start = batch['melody_start'].to(self.device)
                melody_mask = batch['melody_mask'].to(self.device)
                
                try:
                    # Try normal forward pass
                    pitch_logits, duration_pred, start_pred = self.model(
                        full_chord_sequence,
                        chord_mask,
                        focus_positions,
                        chord_durations=chord_durations,  # NEW: pass chord durations
                        melody_pitch=melody_pitch,
                        training=True
                    )
                except RuntimeError as e:
                    if "MPS" in str(e):
                        # Fall back to CPU only for this operation
                        self.model.cpu()
                        pitch_logits, duration_pred, start_pred = self.model(
                            full_chord_sequence.cpu(),
                            chord_mask.cpu(),
                            focus_positions.cpu(),
                            chord_durations=chord_durations.cpu(),  # NEW
                            melody_pitch=melody_pitch.cpu(),
                            training=True
                        )
                        # Move model and results back to MPS
                        self.model.to(self.device)
                        pitch_logits = pitch_logits.to(self.device)
                        duration_pred = duration_pred.to(self.device)
                        start_pred = start_pred.to(self.device)
                    else:
                        raise e
                
                # Rest of validation logic remains the same
                target_pitch = melody_pitch[:, 1:]
                target_duration = melody_duration[:, :-1]
                target_start = melody_start[:, :-1]
                target_mask = melody_mask[:, :-1]
                
                pitch_loss = self.pitch_criterion(
                    pitch_logits.reshape(-1, pitch_logits.size(-1)), 
                    target_pitch.reshape(-1)
                )
                
                if target_mask.sum() > 0:
                    duration_loss = self.duration_criterion(
                        duration_pred[target_mask], 
                        target_duration[target_mask]
                    )
                    start_loss = self.start_criterion(
                        start_pred[target_mask], 
                        target_start[target_mask]
                    )
                else:
                    duration_loss = torch.tensor(0.0, device=self.device)
                    start_loss = torch.tensor(0.0, device=self.device)
                
                total_batch_loss = pitch_loss + 0.3 * duration_loss + 0.3 * start_loss
                accuracy = self.calculate_accuracy(pitch_logits, target_pitch, target_mask)
                
                total_loss += total_batch_loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        return total_loss / num_batches, total_accuracy / num_batches
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training and validation
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.plot_training_history()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_chord_melody_model.pt')
                print("✓ New best model saved!")
            
            # Early stopping if loss increases too much
            if epoch > 10 and val_loss > min(self.val_losses) * 1.2:
                print("Early stopping - validation loss increasing")
                break
                
            print("-" * 50)
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        print(f"\nTraining completed in {training_time:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
    def plot_training_history(self):
        """Plot training and validation loss."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if len(self.val_losses) > 3:
            # Show smoothed validation loss
            window = min(3, len(self.val_losses) // 2)
            smoothed_val = []
            for i in range(len(self.val_losses)):
                start_idx = max(0, i - window)
                end_idx = min(len(self.val_losses), i + window + 1)
                smoothed_val.append(sum(self.val_losses[start_idx:end_idx]) / (end_idx - start_idx))
            
            plt.plot(smoothed_val, label='Smoothed Val Loss', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Smoothed Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')