"""
Molecule generation utilities.

This module provides functions for generating new molecular structures
using a trained VAE model, including sampling strategies and post-processing.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

from .model import MolecularVAE
from .preprocess import MolecularPreprocessor

logger = logging.getLogger(__name__)


class MolecularGenerator:
    """
    Generator class for creating new molecular structures.
    
    Handles sampling from latent space, decoding, and post-processing
    of generated molecules.
    """
    
    def __init__(
        self,
        model: MolecularVAE,
        preprocessor: MolecularPreprocessor,
        device: torch.device = None
    ):
        """
        Initialize generator.
        
        Args:
            model: Trained VAE model
            preprocessor: Molecular preprocessor
            device: Device for generation
        """
        self.model = model
        self.preprocessor = preprocessor
        
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        
        logger.info(f"Initialized generator on device: {device}")
    
    def generate_molecules(
        self,
        num_samples: int = 1,
        temperature: float = 1.0,
        sampling_strategy: str = "random",
        latent_vectors: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Generate new molecular structures.
        
        Args:
            num_samples: Number of molecules to generate
            temperature: Temperature for sampling (higher = more diverse)
            sampling_strategy: Strategy for latent sampling ("random", "interpolate", "cluster")
            latent_vectors: Optional pre-specified latent vectors
            
        Returns:
            List of generated molecules
        """
        self.model.eval()
        
        with torch.no_grad():
            if latent_vectors is not None:
                z = latent_vectors.to(self.device)
                num_samples = z.size(0)
            else:
                # Sample latent vectors based on strategy
                z = self._sample_latent_vectors(
                    num_samples, temperature, sampling_strategy
                )
            
            # Decode to molecules
            outputs = self.model.decode(z)
            
            # Post-process outputs
            molecules = []
            for i in range(num_samples):
                mol_data = self._postprocess_molecule(
                    outputs['atom_logits'][i],
                    outputs['coordinates'][i]
                )
                mol_data['latent_vector'] = z[i].cpu().numpy()
                molecules.append(mol_data)
            
            return molecules
    
    def _sample_latent_vectors(
        self,
        num_samples: int,
        temperature: float,
        strategy: str
    ) -> torch.Tensor:
        """
        Sample latent vectors using different strategies.
        
        Args:
            num_samples: Number of vectors to sample
            temperature: Sampling temperature
            strategy: Sampling strategy
            
        Returns:
            Sampled latent vectors
        """
        if strategy == "random":
            # Random sampling from normal distribution
            z = torch.randn(num_samples, self.model.latent_dim, device=self.device)
            z = z * temperature
            
        elif strategy == "interpolate":
            # Interpolate between random points
            z_start = torch.randn(1, self.model.latent_dim, device=self.device)
            z_end = torch.randn(1, self.model.latent_dim, device=self.device)
            
            alphas = torch.linspace(0, 1, num_samples, device=self.device)
            z = z_start + alphas.view(-1, 1) * (z_end - z_start)
            z = z * temperature
            
        elif strategy == "cluster":
            # Sample around cluster centers
            num_clusters = min(4, num_samples)
            cluster_centers = torch.randn(num_clusters, self.model.latent_dim, device=self.device)
            
            samples_per_cluster = num_samples // num_clusters
            remainder = num_samples % num_clusters
            
            z_list = []
            for i in range(num_clusters):
                cluster_size = samples_per_cluster + (1 if i < remainder else 0)
                noise = torch.randn(cluster_size, self.model.latent_dim, device=self.device) * 0.5
                cluster_samples = cluster_centers[i:i+1] + noise
                z_list.append(cluster_samples)
            
            z = torch.cat(z_list, dim=0) * temperature
            
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        return z
    
    def _postprocess_molecule(
        self,
        atom_logits: torch.Tensor,
        coordinates: torch.Tensor
    ) -> Dict:
        """
        Post-process generated molecule data.
        
        Args:
            atom_logits: Predicted atom type logits
            coordinates: Predicted coordinates
            
        Returns:
            Processed molecule data
        """
        # Convert to numpy
        atom_logits = atom_logits.cpu().numpy()
        coordinates = coordinates.cpu().numpy()
        
        # Get predicted atom types
        predicted_atoms = np.argmax(atom_logits, axis=-1)
        
        # Convert indices to atomic numbers
        atomic_nums = np.array([
            self.preprocessor.idx_to_atom.get(idx, self.preprocessor.atom_types[0])
            for idx in predicted_atoms
        ])
        
        # Denormalize coordinates
        denormalized_coords = self.preprocessor.denormalize_coordinates(coordinates)
        
        # Create mask for valid atoms (non-zero coordinates)
        valid_mask = np.any(np.abs(denormalized_coords) > 1e-6, axis=1)
        
        # Filter valid atoms
        valid_atomic_nums = atomic_nums[valid_mask]
        valid_coords = denormalized_coords[valid_mask]
        
        # Center the molecule
        if len(valid_coords) > 0:
            center = np.mean(valid_coords, axis=0)
            valid_coords = valid_coords - center
        
        return {
            'atomic_nums': valid_atomic_nums,
            'coordinates': valid_coords,
            'num_atoms': len(valid_atomic_nums),
            'atom_logits': atom_logits,
            'raw_coordinates': denormalized_coords
        }
    
    def generate_diverse_set(
        self,
        num_molecules: int = 10,
        diversity_factor: float = 2.0
    ) -> List[Dict]:
        """
        Generate a diverse set of molecules.
        
        Args:
            num_molecules: Number of molecules to generate
            diversity_factor: Factor to increase diversity
            
        Returns:
            List of diverse molecules
        """
        molecules = []
        
        # Generate molecules with different temperatures
        temperatures = np.linspace(0.5, 2.0, 5)
        molecules_per_temp = num_molecules // len(temperatures)
        
        for temp in temperatures:
            temp_molecules = self.generate_molecules(
                num_samples=molecules_per_temp,
                temperature=temp * diversity_factor,
                sampling_strategy="random"
            )
            molecules.extend(temp_molecules)
        
        # Generate remaining molecules with interpolation
        remaining = num_molecules - len(molecules)
        if remaining > 0:
            interp_molecules = self.generate_molecules(
                num_samples=remaining,
                temperature=1.0,
                sampling_strategy="interpolate"
            )
            molecules.extend(interp_molecules)
        
        return molecules
    
    def interpolate_between_molecules(
        self,
        mol1_data: Dict,
        mol2_data: Dict,
        num_steps: int = 10
    ) -> List[Dict]:
        """
        Interpolate between two molecules in latent space.
        
        Args:
            mol1_data: First molecule data
            mol2_data: Second molecule data
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated molecules
        """
        # Ensure molecules have latent vectors
        if 'latent_vector' not in mol1_data or 'latent_vector' not in mol2_data:
            # Encode molecules to get latent vectors
            self.model.eval()
            
            with torch.no_grad():
                # Convert to tensors
                atomic_nums1 = torch.tensor(mol1_data['atomic_nums'], dtype=torch.long).unsqueeze(0)
                coords1 = torch.tensor(mol1_data['coordinates'], dtype=torch.float32).unsqueeze(0)
                
                atomic_nums2 = torch.tensor(mol2_data['atomic_nums'], dtype=torch.long).unsqueeze(0)
                coords2 = torch.tensor(mol2_data['coordinates'], dtype=torch.float32).unsqueeze(0)
                
                # Pad to max_atoms
                padded_atomic_nums1 = torch.zeros(1, self.model.max_atoms, dtype=torch.long)
                padded_coords1 = torch.zeros(1, self.model.max_atoms, 3, dtype=torch.float32)
                
                padded_atomic_nums2 = torch.zeros(1, self.model.max_atoms, dtype=torch.long)
                padded_coords2 = torch.zeros(1, self.model.max_atoms, 3, dtype=torch.float32)
                
                # Fill with actual data
                num_atoms1 = min(len(mol1_data['atomic_nums']), self.model.max_atoms)
                num_atoms2 = min(len(mol2_data['atomic_nums']), self.model.max_atoms)
                
                padded_atomic_nums1[0, :num_atoms1] = atomic_nums1[0, :num_atoms1]
                padded_coords1[0, :num_atoms1] = coords1[0, :num_atoms1]
                
                padded_atomic_nums2[0, :num_atoms2] = atomic_nums2[0, :num_atoms2]
                padded_coords2[0, :num_atoms2] = coords2[0, :num_atoms2]
                
                # Normalize coordinates
                padded_coords1 = self.preprocessor.normalize_coordinates(padded_coords1)
                padded_coords2 = self.preprocessor.normalize_coordinates(padded_coords2)
                
                # Encode
                z1 = self.model.encode(padded_atomic_nums1.to(self.device), padded_coords1.to(self.device))
                z2 = self.model.encode(padded_atomic_nums2.to(self.device), padded_coords2.to(self.device))
        else:
            z1 = torch.tensor(mol1_data['latent_vector'], dtype=torch.float32).unsqueeze(0)
            z2 = torch.tensor(mol2_data['latent_vector'], dtype=torch.float32).unsqueeze(0)
        
        # Interpolate
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        interpolated_z = z1 + alphas.view(-1, 1) * (z2 - z1)
        
        # Decode interpolated vectors
        molecules = []
        with torch.no_grad():
            outputs = self.model.decode(interpolated_z)
            
            for i in range(num_steps):
                mol_data = self._postprocess_molecule(
                    outputs['atom_logits'][i],
                    outputs['coordinates'][i]
                )
                mol_data['latent_vector'] = interpolated_z[i].cpu().numpy()
                mol_data['interpolation_alpha'] = alphas[i].item()
                molecules.append(mol_data)
        
        return molecules
    
    def save_molecules(self, molecules: List[Dict], filepath: str) -> None:
        """
        Save generated molecules to file.
        
        Args:
            molecules: List of molecules to save
            filepath: Path to save file
        """
        # Convert tensors to numpy for JSON serialization
        serializable_molecules = []
        
        for i, mol in enumerate(molecules):
            mol_data = {
                'id': i,
                'atomic_nums': mol['atomic_nums'].tolist(),
                'coordinates': mol['coordinates'].tolist(),
                'num_atoms': int(mol['num_atoms']),
                'latent_vector': mol['latent_vector'].tolist()
            }
            
            if 'interpolation_alpha' in mol:
                mol_data['interpolation_alpha'] = mol['interpolation_alpha']
            
            serializable_molecules.append(mol_data)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(serializable_molecules, f, indent=2)
        
        logger.info(f"Saved {len(molecules)} molecules to {filepath}")
    
    def load_molecules(self, filepath: str) -> List[Dict]:
        """
        Load molecules from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            List of loaded molecules
        """
        with open(filepath, 'r') as f:
            molecules_data = json.load(f)
        
        molecules = []
        for mol_data in molecules_data:
            mol = {
                'atomic_nums': np.array(mol_data['atomic_nums']),
                'coordinates': np.array(mol_data['coordinates']),
                'num_atoms': mol_data['num_atoms'],
                'latent_vector': np.array(mol_data['latent_vector'])
            }
            
            if 'interpolation_alpha' in mol_data:
                mol['interpolation_alpha'] = mol_data['interpolation_alpha']
            
            molecules.append(mol)
        
        logger.info(f"Loaded {len(molecules)} molecules from {filepath}")
        return molecules


def load_generator(
    model_path: str,
    preprocessor: MolecularPreprocessor,
    device: Optional[str] = None
) -> MolecularGenerator:
    """
    Load a trained generator from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        preprocessor: Molecular preprocessor
        device: Device for generation
        
    Returns:
        Loaded generator
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model architecture (assuming default parameters)
    # In practice, you might want to save these parameters during training
    model = MolecularVAE(
        max_atoms=29,
        num_atom_types=5,
        latent_dim=32
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create generator
    generator = MolecularGenerator(model, preprocessor, device)
    
    logger.info(f"Loaded generator from {model_path}")
    return generator


if __name__ == "__main__":
    # Example usage
    from .preprocess import create_preprocessor
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Create a dummy model for testing
    model = MolecularVAE(max_atoms=29, num_atom_types=5, latent_dim=32)
    
    # Create generator
    generator = MolecularGenerator(model, preprocessor)
    
    # Generate some molecules
    molecules = generator.generate_molecules(num_samples=5, temperature=1.0)
    
    print(f"Generated {len(molecules)} molecules:")
    for i, mol in enumerate(molecules):
        print(f"Molecule {i+1}: {mol['num_atoms']} atoms")
        print(f"Atomic numbers: {mol['atomic_nums']}")
        print(f"Coordinate shape: {mol['coordinates'].shape}")
        print()
