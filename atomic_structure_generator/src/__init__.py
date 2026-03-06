"""
3D Molecular Structure Generator

A generative AI project for creating novel 3D molecular structures
using Variational Autoencoders trained on the QM9 dataset.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__description__ = "Generative AI for 3D Molecular Structure Design"

# Import main components
from .model import MolecularVAE, create_vae_model
from .data_loader import QM9Dataset, create_data_loaders
from .preprocess import MolecularPreprocessor, create_preprocessor
from .generate import MolecularGenerator, load_generator
from .visualize import MolecularVisualizer, create_visualizer
from .train import VAETrainer, train_vae

__all__ = [
    "MolecularVAE",
    "create_vae_model", 
    "QM9Dataset",
    "create_data_loaders",
    "MolecularPreprocessor",
    "create_preprocessor",
    "MolecularGenerator", 
    "load_generator",
    "MolecularVisualizer",
    "create_visualizer",
    "VAETrainer",
    "train_vae"
]
