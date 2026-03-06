"""
Variational Autoencoder model for molecular generation.

This module implements a VAE architecture that can learn from molecular
atomic coordinates and generate new 3D molecular structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MolecularEncoder(nn.Module):
    """
    Encoder network for molecular VAE.
    
    Takes atomic numbers and 3D coordinates as input and outputs
    latent space parameters (mean and log variance).
    """
    
    def __init__(
        self,
        max_atoms: int = 29,
        num_atom_types: int = 5,
        latent_dim: int = 32,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize encoder.
        
        Args:
            max_atoms: Maximum number of atoms per molecule
            num_atom_types: Number of different atom types
            latent_dim: Dimensionality of latent space
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.latent_dim = latent_dim
        
        # Input dimension: (atom_type + 3_coords) * max_atoms
        input_dim = (num_atom_types + 3) * max_atoms
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder_layers = nn.Sequential(*layers)
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        
    def forward(self, atomic_nums: torch.Tensor, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            atomic_nums: Tensor of atom type indices [batch_size, max_atoms]
            coordinates: Tensor of 3D coordinates [batch_size, max_atoms, 3]
            
        Returns:
            Tuple of (mean, log_variance) for latent distribution
        """
        batch_size = atomic_nums.size(0)
        
        # Embed atom types
        atom_embedded = self.atom_embedding(atomic_nums)  # [batch_size, max_atoms, num_atom_types]
        
        # Flatten and concatenate with coordinates
        atom_flat = atom_embedded.view(batch_size, -1)  # [batch_size, max_atoms * num_atom_types]
        coord_flat = coordinates.view(batch_size, -1)    # [batch_size, max_atoms * 3]
        
        # Concatenate features
        x = torch.cat([atom_flat, coord_flat], dim=-1)  # [batch_size, (num_atom_types + 3) * max_atoms]
        
        # Pass through encoder layers
        encoded = self.encoder_layers(x)
        
        # Get mean and log variance
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar


class MolecularDecoder(nn.Module):
    """
    Decoder network for molecular VAE.
    
    Takes latent vector as input and reconstructs atomic numbers and 3D coordinates.
    """
    
    def __init__(
        self,
        max_atoms: int = 29,
        num_atom_types: int = 5,
        latent_dim: int = 32,
        hidden_dims: list = [64, 128, 256],
        dropout: float = 0.1
    ):
        """
        Initialize decoder.
        
        Args:
            max_atoms: Maximum number of atoms per molecule
            num_atom_types: Number of different atom types
            latent_dim: Dimensionality of latent space
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.latent_dim = latent_dim
        
        # Output dimension: (atom_type + 3_coords) * max_atoms
        output_dim = (num_atom_types + 3) * max_atoms
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Final output layer
        self.fc_output = nn.Linear(prev_dim, output_dim)
        
        # Separate outputs for atom types and coordinates
        self.output_dim_atom = num_atom_types * max_atoms
        self.output_dim_coord = 3 * max_atoms
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            Tuple of (atom_logits, coordinates)
        """
        # Pass through decoder layers
        decoded = self.decoder_layers(z)
        
        # Get final output
        output = self.fc_output(decoded)
        
        # Split into atom types and coordinates
        atom_output = output[:, :self.output_dim_atom]
        coord_output = output[:, self.output_dim_atom:]
        
        # Reshape to proper dimensions
        atom_logits = atom_output.view(-1, self.max_atoms, self.num_atom_types)
        coordinates = coord_output.view(-1, self.max_atoms, 3)
        
        return atom_logits, coordinates


class MolecularVAE(nn.Module):
    """
    Complete Variational Autoencoder for molecular generation.
    
    Combines encoder and decoder with reparameterization trick for training.
    """
    
    def __init__(
        self,
        max_atoms: int = 29,
        num_atom_types: int = 5,
        latent_dim: int = 32,
        encoder_hidden_dims: list = [256, 128, 64],
        decoder_hidden_dims: list = [64, 128, 256],
        dropout: float = 0.1,
        beta: float = 1.0  # KL divergence weight
    ):
        """
        Initialize VAE.
        
        Args:
            max_atoms: Maximum number of atoms per molecule
            num_atom_types: Number of different atom types
            latent_dim: Dimensionality of latent space
            encoder_hidden_dims: Hidden dimensions for encoder
            decoder_hidden_dims: Hidden dimensions for decoder
            dropout: Dropout rate
            beta: Weight for KL divergence loss
        """
        super().__init__()
        
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Initialize encoder and decoder
        self.encoder = MolecularEncoder(
            max_atoms=max_atoms,
            num_atom_types=num_atom_types,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout
        )
        
        self.decoder = MolecularDecoder(
            max_atoms=max_atoms,
            num_atom_types=num_atom_types,
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, atomic_nums: torch.Tensor, coordinates: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            atomic_nums: Tensor of atom type indices [batch_size, max_atoms]
            coordinates: Tensor of 3D coordinates [batch_size, max_atoms, 3]
            
        Returns:
            Dictionary with model outputs
        """
        # Encode
        mu, logvar = self.encoder(atomic_nums, coordinates)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode
        atom_logits, reconstructed_coords = self.decoder(z)
        
        return {
            'atom_logits': atom_logits,
            'reconstructed_coords': reconstructed_coords,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def generate(self, num_samples: int = 1, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        Generate new molecules by sampling from latent space.
        
        Args:
            num_samples: Number of molecules to generate
            device: Device to generate on
            
        Returns:
            Dictionary with generated molecules
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode
        atom_logits, coordinates = self.decoder(z)
        
        return {
            'atom_logits': atom_logits,
            'coordinates': coordinates,
            'z': z
        }
    
    def encode(self, atomic_nums: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encode molecules to latent space.
        
        Args:
            atomic_nums: Tensor of atom type indices
            coordinates: Tensor of 3D coordinates
            
        Returns:
            Latent vectors
        """
        mu, _ = self.encoder(atomic_nums, coordinates)
        return mu
    
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent vectors to molecules.
        
        Args:
            z: Latent vectors
            
        Returns:
            Dictionary with decoded molecules
        """
        atom_logits, coordinates = self.decoder(z)
        return {
            'atom_logits': atom_logits,
            'coordinates': coordinates
        }
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        reconstruction_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            batch: Batch of data
            reconstruction_weight: Weight for reconstruction loss
            
        Returns:
            Dictionary with loss components
        """
        # Forward pass
        outputs = self.forward(batch['atomic_nums'], batch['coordinates'])
        
        # Get outputs
        atom_logits = outputs['atom_logits']
        reconstructed_coords = outputs['reconstructed_coords']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss for atom types (cross-entropy)
        atom_loss = F.cross_entropy(
            atom_logits.view(-1, self.num_atom_types),
            batch['atomic_nums'].view(-1),
            reduction='none'
        )
        
        # Apply mask to ignore padding
        mask_expanded = batch['mask'].unsqueeze(-1).expand_as(atom_loss.view(-1, self.num_atom_types))
        atom_loss = (atom_loss.view(-1, self.num_atom_types) * mask_expanded).sum() / (batch['mask'].sum() + 1e-8)
        
        # Reconstruction loss for coordinates (MSE)
        coord_loss = F.mse_loss(
            reconstructed_coords * batch['mask'].unsqueeze(-1),
            batch['coordinates'] * batch['mask'].unsqueeze(-1),
            reduction='sum'
        ) / (batch['mask'].sum() + 1e-8)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch['atomic_nums'].size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = reconstruction_weight * (atom_loss + coord_loss) + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'atom_loss': atom_loss,
            'coord_loss': coord_loss,
            'kl_loss': kl_loss
        }


def create_vae_model(
    max_atoms: int = 29,
    num_atom_types: int = 5,
    latent_dim: int = 32,
    **kwargs
) -> MolecularVAE:
    """
    Create and return a VAE model.
    
    Args:
        max_atoms: Maximum number of atoms per molecule
        num_atom_types: Number of different atom types
        latent_dim: Dimensionality of latent space
        **kwargs: Additional arguments for MolecularVAE
        
    Returns:
        MolecularVAE instance
    """
    return MolecularVAE(
        max_atoms=max_atoms,
        num_atom_types=num_atom_types,
        latent_dim=latent_dim,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MolecularVAE(
        max_atoms=29,
        num_atom_types=5,
        latent_dim=32
    ).to(device)
    
    # Create sample data
    batch_size = 4
    atomic_nums = torch.randint(0, 5, (batch_size, 29)).to(device)
    coordinates = torch.randn(batch_size, 29, 3).to(device)
    mask = torch.ones(batch_size, 29).to(device)
    
    batch = {
        'atomic_nums': atomic_nums,
        'coordinates': coordinates,
        'mask': mask
    }
    
    # Test forward pass
    with torch.no_grad():
        outputs = model.forward(atomic_nums, coordinates)
        print("Forward pass outputs:")
        for key, tensor in outputs.items():
            print(f"{key}: {tensor.shape}")
        
        # Test generation
        generated = model.generate(num_samples=2)
        print("\nGenerated molecules:")
        for key, tensor in generated.items():
            print(f"{key}: {tensor.shape}")
        
        # Test loss computation
        losses = model.compute_loss(batch)
        print("\nLoss components:")
        for key, value in losses.items():
            print(f"{key}: {value.item():.4f}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
