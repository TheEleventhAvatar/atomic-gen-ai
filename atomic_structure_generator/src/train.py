"""
Training script for molecular VAE.

This module provides utilities for training the Variational Autoencoder
on molecular data, including loss computation, optimization, and model saving.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm
import json
from pathlib import Path

from .model import MolecularVAE, create_vae_model
from .data_loader import create_data_loaders
from .preprocess import MolecularPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAETrainer:
    """
    Trainer class for molecular VAE.
    
    Handles training loop, validation, model saving, and logging.
    """
    
    def __init__(
        self,
        model: MolecularVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: str = "models",
        log_dir: str = "logs",
        beta: float = 1.0,
        reconstruction_weight: float = 1.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: VAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            device: Device to train on
            save_dir: Directory to save models
            log_dir: Directory for tensorboard logs
            beta: KL divergence weight
            reconstruction_weight: Weight for reconstruction loss
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.beta = beta
        self.reconstruction_weight = reconstruction_weight
        
        # Create directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_atom_loss': [],
            'train_coord_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_atom_loss': [],
            'val_coord_loss': [],
            'val_kl_loss': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"Initialized trainer with device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_atom_loss = 0.0
        total_coord_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.model.compute_loss(batch, self.reconstruction_weight)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total_loss'].item()
            total_atom_loss += losses['atom_loss'].item()
            total_coord_loss += losses['coord_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'atom': f"{losses['atom_loss'].item():.4f}",
                'coord': f"{losses['coord_loss'].item():.4f}",
                'kl': f"{losses['kl_loss'].item():.4f}"
            })
            
            # Log to tensorboard (every 100 batches)
            if batch_idx % 100 == 0:
                step = len(self.train_history['epoch']) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', losses['total_loss'].item(), step)
                self.writer.add_scalar('Train/BatchAtomLoss', losses['atom_loss'].item(), step)
                self.writer.add_scalar('Train/BatchCoordLoss', losses['coord_loss'].item(), step)
                self.writer.add_scalar('Train/BatchKLLoss', losses['kl_loss'].item(), step)
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_atom_loss = total_atom_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return {
            'loss': avg_loss,
            'atom_loss': avg_atom_loss,
            'coord_loss': avg_coord_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_atom_loss = 0.0
        total_coord_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Compute loss
                losses = self.model.compute_loss(batch, self.reconstruction_weight)
                
                # Accumulate losses
                total_loss += losses['total_loss'].item()
                total_atom_loss += losses['atom_loss'].item()
                total_coord_loss += losses['coord_loss'].item()
                total_kl_loss += losses['kl_loss'].item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'atom': f"{losses['atom_loss'].item():.4f}",
                    'coord': f"{losses['coord_loss'].item():.4f}",
                    'kl': f"{losses['kl_loss'].item():.4f}"
                })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_atom_loss = total_atom_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return {
            'loss': avg_loss,
            'atom_loss': avg_atom_loss,
            'coord_loss': avg_coord_loss,
            'kl_loss': avg_kl_loss
        }
    
    def train(self, num_epochs: int, save_every: int = 10) -> None:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save model every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_atom_loss'].append(train_metrics['atom_loss'])
            self.train_history['train_coord_loss'].append(train_metrics['coord_loss'])
            self.train_history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_atom_loss'].append(val_metrics['atom_loss'])
            self.train_history['val_coord_loss'].append(val_metrics['coord_loss'])
            self.train_history['val_kl_loss'].append(val_metrics['kl_loss'])
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch + 1)
            self.writer.add_scalar('Train/AtomLoss', train_metrics['atom_loss'], epoch + 1)
            self.writer.add_scalar('Train/CoordLoss', train_metrics['coord_loss'], epoch + 1)
            self.writer.add_scalar('Train/KLLoss', train_metrics['kl_loss'], epoch + 1)
            
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch + 1)
            self.writer.add_scalar('Val/AtomLoss', val_metrics['atom_loss'], epoch + 1)
            self.writer.add_scalar('Val/CoordLoss', val_metrics['coord_loss'], epoch + 1)
            self.writer.add_scalar('Val/KLLoss', val_metrics['kl_loss'], epoch + 1)
            
            # Print epoch summary
            logger.info(f"Train Loss: {train_metrics['loss']:.4f} "
                       f"(Atom: {train_metrics['atom_loss']:.4f}, "
                       f"Coord: {train_metrics['coord_loss']:.4f}, "
                       f"KL: {train_metrics['kl_loss']:.4f})")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f} "
                       f"(Atom: {val_metrics['atom_loss']:.4f}, "
                       f"Coord: {val_metrics['coord_loss']:.4f}, "
                       f"KL: {val_metrics['kl_loss']:.4f})")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch + 1
                self.save_model('best_model.pth')
                logger.info(f"New best model saved (epoch {epoch + 1})")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_training_history()
        
        # Final save
        self.save_model('final_model.pth')
        self.save_training_history()
        
        logger.info(f"Training completed. Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
    
    def save_model(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_model(self, filename: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint['train_history']
        
        logger.info(f"Loaded model from {filename}")
    
    def save_training_history(self) -> None:
        """Save training history to JSON file."""
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)


def train_vae(
    data_path: str,
    save_dir: str = "models",
    log_dir: str = "logs",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    latent_dim: int = 32,
    beta: float = 1.0,
    reconstruction_weight: float = 1.0,
    device: Optional[str] = None
) -> None:
    """
    Train a VAE model on molecular data.
    
    Args:
        data_path: Path to QM9 dataset
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        latent_dim: Dimensionality of latent space
        beta: KL divergence weight
        reconstruction_weight: Weight for reconstruction loss
        device: Device to train on (auto-detect if None)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    logger.info(f"Training on device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        train_split=0.8,
        num_workers=4
    )
    
    # Create model
    model = create_vae_model(
        max_atoms=29,
        num_atom_types=5,
        latent_dim=latent_dim,
        beta=beta
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        beta=beta,
        reconstruction_weight=reconstruction_weight
    )
    
    # Train model
    trainer.train(num_epochs=num_epochs, save_every=10)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    # Example usage
    train_vae(
        data_path="../data",
        save_dir="../models",
        log_dir="../logs",
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        latent_dim=32
    )
