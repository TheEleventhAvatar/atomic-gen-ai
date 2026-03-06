# 🧬 3D Molecular Structure Generator

A Generative AI project that creates novel 3D molecular structures using a Variational Autoencoder (VAE) trained on the QM9 molecular dataset. This project demonstrates the application of deep learning to computational chemistry and molecular design.

## 🎯 Project Overview

This project implements a complete pipeline for:
- **Data Processing**: Loading and preprocessing the QM9 molecular dataset
- **Model Training**: Training a VAE to learn molecular representations
- **Molecule Generation**: Generating new 3D molecular structures
- **Visualization**: Interactive 3D visualization of generated molecules
- **Web Interface**: User-friendly Streamlit application

## 🏗️ Architecture

The system consists of several key components:

### Core Modules
- **`data_loader.py`**: Handles QM9 dataset loading and PyTorch Dataset creation
- **`preprocess.py`**: Molecular preprocessing and coordinate normalization
- **`model.py`**: VAE architecture with encoder/decoder networks
- **`train.py`**: Training pipeline with loss computation and model saving
- **`generate.py`**: Molecule generation and sampling strategies
- **`visualize.py`**: 3D visualization utilities (py3Dmol, Plotly, Matplotlib)
- **`app.py`**: Streamlit web interface

### Model Architecture
- **Encoder**: Linear layers with ReLU activations
- **Latent Space**: 16-32 dimensional representation
- **Decoder**: Reconstructs atomic coordinates from latent vectors
- **Loss Function**: Reconstruction loss + KL divergence

## 📦 Project Structure

```
atomic_structure_generator/
├── data/                   # QM9 dataset files
├── models/                 # Trained model checkpoints
├── notebooks/              # Jupyter notebooks for EDA
├── src/                    # Source code modules
│   ├── data_loader.py      # Dataset handling
│   ├── preprocess.py       # Data preprocessing
│   ├── model.py           # VAE model architecture
│   ├── train.py           # Training pipeline
│   ├── generate.py        # Molecule generation
│   └── visualize.py       # 3D visualization
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd atomic_structure_generator
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download QM9 Dataset**
```bash
# Create data directory
mkdir -p data/xyz

# Download QM9 dataset from:
# https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134k_molecules/978904

# Extract XYZ files to data/xyz/
```

## 🏋️ Training

### Basic Training
```bash
# Train the VAE model
python -m src.train

# Or with custom parameters
python -c "from src.train import train_vae; train_vae(
    data_path='data',
    save_dir='models',
    num_epochs=100,
    batch_size=32,
    latent_dim=32
)"
```

### Training Parameters
- `data_path`: Path to QM9 dataset
- `num_epochs`: Number of training epochs (default: 100)
- `batch_size`: Training batch size (default: 32)
- `latent_dim`: Latent space dimensionality (default: 32)
- `learning_rate`: Adam optimizer learning rate (default: 1e-3)
- `beta`: KL divergence weight (default: 1.0)

### Monitoring Training
```bash
# View training logs
tensorboard --logdir=logs
```

## 🧪 Generating Molecules

### Command Line Generation
```bash
python -c "
from src.generate import load_generator
from src.preprocess import create_preprocessor

# Load preprocessor
preprocessor = create_preprocessor()

# Load trained generator
generator = load_generator('models/best_model.pth', preprocessor)

# Generate molecules
molecules = generator.generate_molecules(num_samples=10, temperature=1.0)
print(f'Generated {len(molecules)} molecules')
"
```

### Python API
```python
from src.generate import MolecularGenerator
from src.model import create_vae_model
from src.preprocess import create_preprocessor

# Create components
preprocessor = create_preprocessor()
model = create_vae_model(latent_dim=32)
generator = MolecularGenerator(model, preprocessor)

# Generate diverse molecules
molecules = generator.generate_diverse_set(num_molecules=10)

# Interpolate between molecules
interpolated = generator.interpolate_between_molecules(mol1, mol2, num_steps=10)
```

## 🌐 Web Interface

### Launch Streamlit App
```bash
streamlit run app.py
```

### Features
- **Interactive 3D Visualization**: Rotate, zoom, and inspect molecules
- **Generation Controls**: Adjust temperature and sampling strategies
- **Multiple Backends**: Choose between Plotly and py3Dmol visualization
- **Download Options**: Export molecules as JSON or XYZ files
- **Gallery View**: Browse multiple generated molecules

### Web Interface Usage
1. **Load Model**: Use the sidebar to load a trained model
2. **Configure Settings**: Adjust generation parameters
3. **Generate Molecules**: Click the generate button
4. **Explore Results**: View and interact with generated structures
5. **Export Data**: Download molecules in various formats

## 📊 Visualization Options

### Supported Backends
1. **Plotly**: Interactive 3D plots with zoom/rotate
2. **py3Dmol**: Professional molecular visualization
3. **Matplotlib**: Static 3D plots for publications

### Visualization Features
- Atom coloring by element type
- Bond detection and display
- Atom labels and element symbols
- Multiple molecule comparison
- Coordinate display and export

## 🔬 Dataset Details

### QM9 Dataset
- **Size**: ~134,000 organic molecules
- **Elements**: H, C, N, O, F
- **Properties**: 3D coordinates, atomic numbers, molecular properties
- **Format**: XYZ files with molecular structures

### Data Processing
- Coordinate normalization (z-score)
- Atom type encoding (one-hot)
- Padding to fixed molecule size
- Train/validation split (80/20)

## 🧮 Model Details

### VAE Architecture
```
Input: (atom_types + coordinates) × max_atoms
├── Encoder
│   ├── Linear(256) → ReLU → Dropout
│   ├── Linear(128) → ReLU → Dropout
│   ├── Linear(64) → ReLU → Dropout
│   └── Linear(latent_dim × 2)  # μ and logσ
├── Latent Space
│   └── Reparameterization trick
└── Decoder
    ├── Linear(64) → ReLU → Dropout
    ├── Linear(128) → ReLU → Dropout
    ├── Linear(256) → ReLU → Dropout
    └── Linear(output_dim)  # atom_logits + coordinates
```

### Loss Components
- **Reconstruction Loss**: Cross-entropy (atoms) + MSE (coordinates)
- **KL Divergence**: Regularizes latent space
- **Total Loss**: Weighted combination with β parameter

## 📈 Performance Metrics

### Training Metrics
- Reconstruction loss convergence
- KL divergence behavior
- Latent space organization
- Generation quality assessment

### Evaluation Methods
- Visual inspection of generated structures
- Chemical validity checks
- Diversity analysis
- Property prediction accuracy

## 🛠️ Development

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- Modular architecture
- Error handling and logging

### Testing
```bash
# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
mypy src/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📚 Example Usage

### Basic Generation
```python
from src.generate import load_generator
from src.preprocess import create_preprocessor

# Load trained model
preprocessor = create_preprocessor()
generator = load_generator('models/best_model.pth', preprocessor)

# Generate 5 molecules
molecules = generator.generate_molecules(
    num_samples=5,
    temperature=1.0,
    sampling_strategy='random'
)

# Visualize first molecule
from src.visualize import create_visualizer
visualizer = create_visualizer()
fig = visualizer.visualize_plotly(molecules[0])
fig.show()
```

### Advanced Generation
```python
# Generate diverse set
diverse_mols = generator.generate_diverse_set(
    num_molecules=20,
    diversity_factor=2.0
)

# Interpolate between molecules
interpolated = generator.interpolate_between_molecules(
    diverse_mols[0],
    diverse_mols[1],
    num_steps=10
)

# Save results
generator.save_molecules(diverse_mols, 'generated_molecules.json')
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller latent dimension
   - Enable gradient checkpointing

2. **Model Loading Errors**
   - Check model file path
   - Verify model architecture matches checkpoint
   - Ensure consistent PyTorch versions

3. **Dataset Issues**
   - Verify QM9 dataset download
   - Check XYZ file format
   - Ensure correct directory structure

4. **Visualization Problems**
   - Update visualization packages
   - Check browser compatibility
   - Verify molecule data format

### Performance Tips
- Use GPU for training (CUDA recommended)
- Enable mixed precision training
- Use data loading with multiple workers
- Cache preprocessed data

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **QM9 Dataset**: Ramakrishnan et al., "Quantum Chemistry Structures and Properties of 134k Molecules"
- **RDKit**: Open-source cheminformatics library
- **PyTorch**: Deep learning framework
- **Streamlit**: Web app framework

## 📞 Contact

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: [your-email@example.com]

---

**Note**: This is a research project for educational purposes. Generated molecules should be validated for chemical feasibility before any practical use.
