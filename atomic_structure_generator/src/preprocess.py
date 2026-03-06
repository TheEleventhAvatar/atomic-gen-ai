"""
Preprocessing utilities for molecular data.

This module provides functions for normalizing coordinates, encoding atom types,
and preparing tensors for VAE training.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MolecularPreprocessor:
    """
    Preprocessor for molecular data.
    
    Handles coordinate normalization, atom type encoding, and tensor preparation
    for VAE training and inference.
    """
    
    def __init__(
        self,
        atom_types: List[int] = [1, 6, 7, 8, 9],  # H, C, N, O, F
        max_atoms: int = 29,
        coord_normalize: bool = True,
        coord_mean: Optional[float] = None,
        coord_std: Optional[float] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            atom_types: List of atomic numbers to consider
            max_atoms: Maximum number of atoms per molecule
            coord_normalize: Whether to normalize coordinates
            coord_mean: Mean coordinate value for normalization
            coord_std: Standard deviation for normalization
        """
        self.atom_types = atom_types
        self.max_atoms = max_atoms
        self.coord_normalize = coord_normalize
        
        # Create atom type mappings
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}
        self.idx_to_atom = {idx: atom for idx, atom in enumerate(atom_types)}
        self.num_atom_types = len(atom_types)
        
        # Coordinate normalization parameters
        self.coord_mean = coord_mean if coord_mean is not None else 0.0
        self.coord_std = coord_std if coord_std is not None else 1.0
        
        logger.info(f"Initialized preprocessor with {self.num_atom_types} atom types")
        logger.info(f"Atom types: {self.atom_types}")
    
    def encode_atoms(self, atomic_nums: np.ndarray) -> np.ndarray:
        """
        Encode atomic numbers to one-hot vectors.
        
        Args:
            atomic_nums: Array of atomic numbers
            
        Returns:
            One-hot encoded atom types
        """
        encoded = np.zeros((len(atomic_nums), self.num_atom_types), dtype=np.float32)
        
        for i, atomic_num in enumerate(atomic_nums):
            if atomic_num in self.atom_to_idx:
                idx = self.atom_to_idx[atomic_num]
                encoded[i, idx] = 1.0
            else:
                # Unknown atom type, use first type as default
                encoded[i, 0] = 1.0
                
        return encoded
    
    def decode_atoms(self, encoded_atoms: np.ndarray) -> np.ndarray:
        """
        Decode one-hot encoded atoms back to atomic numbers.
        
        Args:
            encoded_atoms: One-hot encoded atom types
            
        Returns:
            Array of atomic numbers
        """
        atomic_nums = []
        
        for i in range(encoded_atoms.shape[0]):
            idx = np.argmax(encoded_atoms[i])
            atomic_num = self.idx_to_atom.get(idx, self.atom_types[0])
            atomic_nums.append(atomic_num)
            
        return np.array(atomic_nums)
    
    def normalize_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates using z-score normalization.
        
        Args:
            coordinates: Array of 3D coordinates
            
        Returns:
            Normalized coordinates
        """
        if self.coord_normalize:
            return (coordinates - self.coord_mean) / (self.coord_std + 1e-8)
        return coordinates
    
    def denormalize_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Denormalize coordinates back to original scale.
        
        Args:
            coordinates: Normalized coordinates
            
        Returns:
            Denormalized coordinates
        """
        if self.coord_normalize:
            return coordinates * self.coord_std + self.coord_mean
        return coordinates
    
    def pad_molecule(
        self,
        atomic_nums: np.ndarray,
        coordinates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pad molecule to max_atoms size.
        
        Args:
            atomic_nums: Array of atomic numbers
            coordinates: Array of 3D coordinates
            
        Returns:
            Tuple of (padded_atomic_nums, padded_coordinates, mask)
        """
        num_atoms = len(atomic_nums)
        
        # Create padded arrays
        padded_atomic_nums = np.zeros(self.max_atoms, dtype=np.int64)
        padded_coordinates = np.zeros((self.max_atoms, 3), dtype=np.float32)
        mask = np.zeros(self.max_atoms, dtype=np.float32)
        
        # Fill with actual data
        padded_atomic_nums[:num_atoms] = atomic_nums
        padded_coordinates[:num_atoms] = coordinates
        mask[:num_atoms] = 1.0
        
        return padded_atomic_nums, padded_coordinates, mask
    
    def preprocess_molecule(
        self,
        atomic_nums: np.ndarray,
        coordinates: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single molecule for VAE input.
        
        Args:
            atomic_nums: Array of atomic numbers
            coordinates: Array of 3D coordinates
            
        Returns:
            Dictionary with preprocessed tensors
        """
        # Pad molecule
        padded_atomic_nums, padded_coordinates, mask = self.pad_molecule(
            atomic_nums, coordinates
        )
        
        # Normalize coordinates
        normalized_coords = self.normalize_coordinates(padded_coordinates)
        
        # Encode atoms to indices
        atomic_indices = np.array([
            self.atom_to_idx.get(num, 0) for num in padded_atomic_nums
        ])
        
        # Convert to tensors
        return {
            'atomic_nums': torch.from_numpy(atomic_indices),
            'coordinates': torch.from_numpy(normalized_coords),
            'mask': torch.from_numpy(mask),
            'num_atoms': torch.tensor(len(atomic_nums), dtype=torch.long)
        }
    
    def postprocess_output(
        self,
        atomic_logits: torch.Tensor,
        coordinates: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Postprocess VAE output back to molecular format.
        
        Args:
            atomic_logits: Predicted atom type logits
            coordinates: Predicted coordinates
            mask: Atom mask
            
        Returns:
            Dictionary with postprocessed arrays
        """
        # Convert to numpy
        atomic_logits = atomic_logits.detach().cpu().numpy()
        coordinates = coordinates.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        
        # Get predicted atom types
        predicted_atoms = np.argmax(atomic_logits, axis=-1)
        
        # Convert indices to atomic numbers
        atomic_nums = np.array([
            self.idx_to_atom.get(idx, self.atom_types[0]) for idx in predicted_atoms
        ])
        
        # Denormalize coordinates
        denormalized_coords = self.denormalize_coordinates(coordinates)
        
        # Apply mask
        valid_mask = mask > 0.5
        
        return {
            'atomic_nums': atomic_nums[valid_mask],
            'coordinates': denormalized_coords[valid_mask],
            'mask': mask[valid_mask]
        }
    
    def compute_stats(self, molecules: List[Dict]) -> Dict[str, float]:
        """
        Compute dataset statistics.
        
        Args:
            molecules: List of molecule dictionaries
            
        Returns:
            Dictionary with statistics
        """
        all_coords = []
        atom_counts = []
        
        for mol in molecules:
            all_coords.append(mol['coordinates'])
            atom_counts.append(len(mol['atomic_nums']))
        
        all_coords = np.concatenate(all_coords, axis=0)
        atom_counts = np.array(atom_counts)
        
        stats = {
            'coord_mean': float(np.mean(all_coords)),
            'coord_std': float(np.std(all_coords)),
            'coord_min': float(np.min(all_coords)),
            'coord_max': float(np.max(all_coords)),
            'num_atoms_mean': float(np.mean(atom_counts)),
            'num_atoms_std': float(np.std(atom_counts)),
            'num_atoms_min': int(np.min(atom_counts)),
            'num_atoms_max': int(np.max(atom_counts))
        }
        
        return stats
    
    def create_atom_features(self, atomic_nums: np.ndarray) -> np.ndarray:
        """
        Create additional atom features beyond atomic numbers.
        
        Args:
            atomic_nums: Array of atomic numbers
            
        Returns:
            Feature matrix for atoms
        """
        features = []
        
        for atomic_num in atomic_nums:
            # Basic atomic properties
            atomic_weight = self._get_atomic_weight(atomic_num)
            electronegativity = self._get_electronegativity(atomic_num)
            valence_electrons = self._get_valence_electrons(atomic_num)
            
            # Create feature vector
            atom_features = [
                atomic_num / 100.0,  # Normalize atomic number
                atomic_weight / 200.0,  # Normalize atomic weight
                electronegativity / 4.0,  # Normalize electronegativity
                valence_electrons / 8.0  # Normalize valence electrons
            ]
            
            features.append(atom_features)
        
        return np.array(features, dtype=np.float32)
    
    def _get_atomic_weight(self, atomic_num: int) -> float:
        """Get atomic weight for element."""
        atomic_weights = {
            1: 1.008,   # H
            6: 12.011,  # C
            7: 14.007,  # N
            8: 15.999,  # O
            9: 18.998   # F
        }
        return atomic_weights.get(atomic_num, 12.011)  # Default to carbon
    
    def _get_electronegativity(self, atomic_num: int) -> float:
        """Get Pauling electronegativity for element."""
        electronegativities = {
            1: 2.20,   # H
            6: 2.55,   # C
            7: 3.04,   # N
            8: 3.44,   # O
            9: 3.98    # F
        }
        return electronegativities.get(atomic_num, 2.55)  # Default to carbon
    
    def _get_valence_electrons(self, atomic_num: int) -> int:
        """Get number of valence electrons for element."""
        valence_electrons = {
            1: 1,   # H
            6: 4,   # C
            7: 5,   # N
            8: 6,   # O
            9: 7    # F
        }
        return valence_electrons.get(atomic_num, 4)  # Default to carbon


def create_preprocessor(
    coord_mean: Optional[float] = None,
    coord_std: Optional[float] = None,
    **kwargs
) -> MolecularPreprocessor:
    """
    Create and return a MolecularPreprocessor instance.
    
    Args:
        coord_mean: Mean coordinate for normalization
        coord_std: Standard deviation for normalization
        **kwargs: Additional arguments for MolecularPreprocessor
        
    Returns:
        MolecularPreprocessor instance
    """
    return MolecularPreprocessor(
        coord_mean=coord_mean,
        coord_std=coord_std,
        **kwargs
    )


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = MolecularPreprocessor()
    
    # Sample molecule (water)
    atomic_nums = np.array([8, 1, 1])  # O, H, H
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0]
    ])
    
    # Test preprocessing
    processed = preprocessor.preprocess_molecule(atomic_nums, coordinates)
    
    print("Preprocessed tensors:")
    for key, tensor in processed.items():
        print(f"{key}: {tensor.shape}")
    
    # Test encoding/decoding
    encoded = preprocessor.encode_atoms(atomic_nums)
    decoded = preprocessor.decode_atoms(encoded)
    
    print(f"\nOriginal atomic nums: {atomic_nums}")
    print(f"Decoded atomic nums: {decoded}")
    print(f"Encoding/decoding correct: {np.array_equal(atomic_nums, decoded)}")
