"""
Data loader for QM9 molecular dataset.

This module provides utilities to load and handle the QM9 dataset,
extracting atomic coordinates and atom types for training the VAE model.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QM9Dataset(Dataset):
    """
    PyTorch Dataset for QM9 molecular data.
    
    Loads molecular structures and converts them to tensor format
    suitable for VAE training.
    """
    
    def __init__(
        self,
        data_path: str,
        max_atoms: int = 29,
        normalize_coords: bool = True,
        atom_types: Optional[List[int]] = None
    ):
        """
        Initialize QM9 dataset.
        
        Args:
            data_path: Path to QM9 dataset files
            max_atoms: Maximum number of atoms per molecule (QM9 max is 29)
            normalize_coords: Whether to normalize coordinates
            atom_types: List of atomic numbers to consider
        """
        self.data_path = data_path
        self.max_atoms = max_atoms
        self.normalize_coords = normalize_coords
        
        # Default atom types in QM9 (H, C, N, O, F)
        if atom_types is None:
            self.atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
        else:
            self.atom_types = atom_types
            
        # Create atom type to index mapping
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}
        
        # Load molecular data
        self.molecules = self._load_molecules()
        
        # Calculate coordinate statistics for normalization
        if self.normalize_coords:
            self.coord_mean, self.coord_std = self._calculate_coord_stats()
        else:
            self.coord_mean, self.coord_std = 0.0, 1.0
            
        logger.info(f"Loaded {len(self.molecules)} molecules from {data_path}")
    
    def _load_molecules(self) -> List[Dict]:
        """Load molecules from QM9 dataset files."""
        molecules = []
        
        # Try to load from pickle file first (faster)
        pickle_path = os.path.join(self.data_path, 'qm9_molecules.pkl')
        if os.path.exists(pickle_path):
            logger.info(f"Loading molecules from {pickle_path}")
            with open(pickle_path, 'rb') as f:
                molecules = pickle.load(f)
            return molecules
        
        # Load from raw XYZ files
        xyz_dir = os.path.join(self.data_path, 'xyz')
        if not os.path.exists(xyz_dir):
            raise FileNotFoundError(f"XYZ directory not found: {xyz_dir}")
        
        xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith('.xyz')]
        logger.info(f"Found {len(xyz_files)} XYZ files")
        
        for i, xyz_file in enumerate(xyz_files[:1000]):  # Limit to 1000 for demo
            if i % 100 == 0:
                logger.info(f"Processing file {i}/{len(xyz_files)}")
                
            file_path = os.path.join(xyz_dir, xyz_file)
            try:
                mol_data = self._parse_xyz_file(file_path)
                if mol_data:
                    molecules.append(mol_data)
            except Exception as e:
                logger.warning(f"Error processing {xyz_file}: {e}")
                continue
        
        # Save to pickle for faster loading next time
        with open(pickle_path, 'wb') as f:
            pickle.dump(molecules, f)
            
        return molecules
    
    def _parse_xyz_file(self, file_path: str) -> Optional[Dict]:
        """Parse XYZ file to extract atomic coordinates and types."""
        try:
            mol = rdmolfiles.MolFromXYZFile(file_path)
            if mol is None:
                return None
                
            # Get atomic numbers and coordinates
            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            coords = mol.GetConformer().GetPositions()
            
            # Filter to only allowed atom types
            valid_atoms = []
            valid_coords = []
            
            for i, (atomic_num, coord) in enumerate(zip(atomic_nums, coords)):
                if atomic_num in self.atom_types:
                    valid_atoms.append(atomic_num)
                    valid_coords.append(coord)
            
            if len(valid_atoms) == 0:
                return None
                
            return {
                'atomic_nums': np.array(valid_atoms),
                'coordinates': np.array(valid_coords),
                'num_atoms': len(valid_atoms)
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _calculate_coord_stats(self) -> Tuple[float, float]:
        """Calculate mean and std of coordinates for normalization."""
        all_coords = []
        for mol in self.molecules:
            all_coords.append(mol['coordinates'])
            
        if all_coords:
            all_coords = np.concatenate(all_coords, axis=0)
            return float(np.mean(all_coords)), float(np.std(all_coords))
        return 0.0, 1.0
    
    def __len__(self) -> int:
        return len(self.molecules)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single molecule as tensors."""
        mol = self.molecules[idx]
        
        # Pad to max_atoms
        atomic_nums = mol['atomic_nums']
        coordinates = mol['coordinates']
        num_atoms = mol['num_atoms']
        
        # Create padded arrays
        padded_atomic_nums = np.zeros(self.max_atoms, dtype=np.int64)
        padded_coords = np.zeros((self.max_atoms, 3), dtype=np.float32)
        mask = np.zeros(self.max_atoms, dtype=np.float32)
        
        # Fill with actual data
        padded_atomic_nums[:num_atoms] = atomic_nums
        padded_coords[:num_atoms] = coordinates
        mask[:num_atoms] = 1.0
        
        # Normalize coordinates
        if self.normalize_coords:
            padded_coords = (padded_coords - self.coord_mean) / (self.coord_std + 1e-8)
        
        # Convert atomic numbers to indices
        atomic_indices = np.array([self.atom_to_idx.get(num, 0) for num in padded_atomic_nums])
        
        # Convert to tensors
        return {
            'atomic_nums': torch.from_numpy(atomic_indices),
            'coordinates': torch.from_numpy(padded_coords),
            'mask': torch.from_numpy(mask),
            'num_atoms': torch.tensor(num_atoms, dtype=torch.long)
        }


def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_path: Path to QM9 dataset
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        num_workers: Number of worker processes
        **kwargs: Additional arguments for QM9Dataset
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = QM9Dataset(data_path, **kwargs)
    
    # Split into train and validation
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def download_qm9_dataset(data_path: str) -> None:
    """
    Download QM9 dataset (placeholder function).
    
    In practice, you would download the QM9 dataset from:
    https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134k_molecules/978904
    
    Args:
        data_path: Directory to save the dataset
    """
    os.makedirs(data_path, exist_ok=True)
    logger.info(f"Please download QM9 dataset to {data_path}")
    logger.info("Visit: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134k_molecules/978904")


if __name__ == "__main__":
    # Test the data loader
    data_path = "../data"
    
    if not os.path.exists(data_path):
        download_qm9_dataset(data_path)
    else:
        # Create a small sample dataset for testing
        logger.info("Creating sample dataset for testing...")
        
        # Create sample XYZ file
        xyz_dir = os.path.join(data_path, "xyz")
        os.makedirs(xyz_dir, exist_ok=True)
        
        sample_xyz = """9
Water molecule
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
"""
        
        with open(os.path.join(xyz_dir, "water.xyz"), "w") as f:
            f.write(sample_xyz)
        
        # Test dataset
        dataset = QM9Dataset(data_path)
        logger.info(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample keys: {sample.keys()}")
            logger.info(f"Atomic nums shape: {sample['atomic_nums'].shape}")
            logger.info(f"Coordinates shape: {sample['coordinates'].shape}")
