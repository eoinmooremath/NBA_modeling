"""
Training Methods for NBA Player Performance Prediction

This module provides the training functionality for the SAINT model, including
custom loss functions, training loops, and evaluation methods.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from IPython import display
import numpy as np

class DistributionPinballLoss(nn.Module):
    """Loss function that combines MSE with pinball loss to maintain statistical distributions."""
    def __init__(self, column_weights=None, base_weight=0.7, pinball_weight=0.3, quantiles=None):
        super().__init__()
        self.register_buffer('column_weights', 
                           column_weights if column_weights is not None 
                           else torch.ones(7))
        self.base_weight = base_weight
        self.pinball_weight = pinball_weight
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.25, 0.5, 0.75, 0.9]
        
    def pinball_loss(self, y_pred, y_true, tau):
        diff = y_pred - y_true
        return torch.max(tau * diff, (tau - 1) * diff)
    
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=0.0)
        mse_loss = torch.mean((y_pred - y_true) ** 2 * self.column_weights.view(1, 1, -1))
        
        pinball_losses = []
        for tau in self.quantiles:
            tau_tensor = torch.tensor(tau, device=y_pred.device)
            pinball = self.pinball_loss(y_pred, y_true, tau_tensor)
            weighted_pinball = pinball * self.column_weights.view(1, 1, -1)
            pinball_losses.append(torch.mean(weighted_pinball))
        
        avg_pinball_loss = torch.mean(torch.stack(pinball_losses))
        return self.base_weight * mse_loss + self.pinball_weight * avg_pinball_loss


class AdaptiveLearningRate:
    """Simplified learning rate management with cyclic scheduling."""
    def __init__(self, base_lr, max_lr, step_size):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.step_count = 0
        
    def get_lr(self):
        """Calculate learning rate using cosine annealing"""
        cycle_progress = (self.step_count % self.step_size) / self.step_size
        lr = self.base_lr + (self.max_lr - self.base_lr) * \
             (1 + np.cos(np.pi * cycle_progress)) / 2
        self.step_count += 1
        return lr
    
    def step(self, optimizer):
        """Update optimizer's learning rate"""
        lr = self.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

from IPython import display

class TrainingMonitor:
    """Essential training visualization and monitoring."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        
        # Create the figure and axis
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.setup_plot()
        
    def setup_plot(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title('Training Progress', fontsize=12)
        self.ax.set_xlabel('Epoch', fontsize=10)
        self.ax.set_ylabel('Loss', fontsize=10)
        plt.tight_layout()
        
    def update(self, train_loss, val_loss, lr=None, batch_size=None):
        # Clear previous plots
        display.clear_output(wait=True)
        
        # Append new losses
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        
        epochs = list(range(1, len(self.train_losses) + 1))
        
        # Clear the axis but keep the figure
        self.ax.clear()
        
        # Plot training and validation losses
        self.ax.plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        self.ax.plot(epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        
        # Set labels and title
        self.ax.set_title('Training Progress', fontsize=12)
        self.ax.set_xlabel('Epoch', fontsize=10)
        self.ax.set_ylabel('Loss', fontsize=10)
        self.ax.legend(fontsize=10)
        self.ax.grid(True)
        
        # Add current loss values to the plot
        if len(epochs) > 0:
            self.ax.text(0.02, 0.98, 
                        f'Train Loss: {train_loss:.4f}\nValid Loss: {val_loss:.4f}',
                        transform=self.ax.transAxes,
                        verticalalignment='top',
                        fontsize=10)
        
        # Update display
        plt.tight_layout()
        display.display(self.fig)  # Display the figure


def train_model(
    model, 
    train_dataset, 
    valid_dataset, 
    device,
    save_dir='./checkpoints',
    batch_stages=[(512, 5), (128, 5), (32, 5)],
    num_epochs=100,
    initial_learning_rate=0.001,
    min_learning_rate=1e-6,
    weight_decay=0.01,
    gradient_clip=1.0,
    num_workers=4,
    column_weights=None,
    base_weight=0.7,
    pinball_weight=0.3,
    quantiles=None
):
    """
    Training function streamlined for SAINT model with dynamic batch sizing
    and monitoring.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize training components
    criterion = DistributionPinballLoss(
        column_weights=column_weights.to(device) if column_weights is not None else None,
        base_weight=base_weight,
        pinball_weight=pinball_weight,
        quantiles=quantiles
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize monitoring and tracking
    monitor = TrainingMonitor()
    scaler = torch.amp.GradScaler('cuda')  
    best_valid_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop with epoch progress bar
    epoch_bar = tqdm(range(num_epochs), desc='Epochs', position=0)
    
    for epoch in epoch_bar:
        # Determine current batch size stage
        current_stage = 0
        for i, (batch_size, epochs_in_stage) in enumerate(batch_stages):
            if epoch >= sum(stage[1] for stage in batch_stages[:i]):
                current_stage = i
        
        current_batch_size, _ = batch_stages[current_stage]
        
        # Create data loaders for current batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Initialize learning rate scheduler
        lr_scheduler = AdaptiveLearningRate(
            min_learning_rate,
            initial_learning_rate,
            len(train_loader)
        )
        
        # Training phase
        model.train()
        train_loss = 0
        
        # Batch progress bar
        batch_bar = tqdm(
            enumerate(train_loader),
            desc=f'Training (batch size={current_batch_size})',
            total=len(train_loader),
            position=1,
            leave=False
        )
        
        for batch_idx, (x_categ, x_cont, y_true) in batch_bar:
            x_categ = x_categ.to(device)
            x_cont = x_cont.to(device)
            y_true = y_true.to(device)
            
            with torch.amp.autocast('cuda'):
                y_pred = model(x_categ, x_cont)
                loss = criterion(y_pred, y_true)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            current_lr = lr_scheduler.step(optimizer)
            train_loss += loss.item()
            
            # Update batch progress bar
            avg_loss = train_loss / (batch_idx + 1)
            batch_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        valid_loss = 0
        
        # Validation progress bar
        valid_bar = tqdm(
            valid_loader,
            desc='Validating',
            position=1,
            leave=False
        )
        
        with torch.no_grad():
            for x_categ, x_cont, y_true in valid_bar:
                x_categ = x_categ.to(device)
                x_cont = x_cont.to(device)
                y_true = y_true.to(device)
                
                with torch.amp.autocast('cuda'):
                    y_pred = model(x_categ, x_cont)
                    loss = criterion(y_pred, y_true)
                
                valid_loss += loss.item()
                valid_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        valid_loss /= len(valid_loader)
        
        # Update monitoring
        monitor.update(train_loss, valid_loss, current_lr, current_batch_size)
        
        # Update epoch progress bar
        epoch_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'valid_loss': f'{valid_loss:.4f}',
            'best': f'{best_valid_loss:.4f}',
            'patience': f'{patience_counter}/10',
            'batch_size': current_batch_size
        })
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            print("\nEarly stopping triggered!")
            break
    
    # Save final model
    save_path = save_dir / f'model_loss{best_valid_loss:.4f}_time_{datetime.now():%Y%m%d_%H%M}.pt'
    torch.save({
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'epoch': epoch
    }, save_path)
    
    print(f"\nSaved model to {save_path}")
    plt.close(monitor.fig)

    return best_model_state, best_valid_loss, save_path

