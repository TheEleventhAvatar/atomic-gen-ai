"""
Streamlit web interface for molecular generation.

This module provides a user-friendly web interface for generating
and visualizing 3D molecular structures using the trained VAE model.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import py3Dmol
import base64
from io import BytesIO
import json
import logging
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MolecularVAE, create_vae_model
from src.preprocess import MolecularPreprocessor, create_preprocessor
from src.generate import MolecularGenerator, load_generator
from src.visualize import MolecularVisualizer, create_visualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="3D Molecular Structure Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .molecule-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


class MolecularApp:
    """
    Main application class for the molecular generator web interface.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.generator = None
        self.preprocessor = None
        self.visualizer = None
        
        # Initialize session state
        if 'generated_molecules' not in st.session_state:
            st.session_state.generated_molecules = []
        if 'current_molecule' not in st.session_state:
            st.session_state.current_molecule = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
    
    def load_model(self, model_path: str = "models/best_model.pth"):
        """
        Load the trained model and initialize components.
        
        Args:
            model_path: Path to the trained model checkpoint
        """
        try:
            # Check if model file exists
            if not Path(model_path).exists():
                st.error(f"Model file not found: {model_path}")
                return False
            
            # Create preprocessor
            self.preprocessor = create_preprocessor()
            
            # Create model
            self.model = create_vae_model(
                max_atoms=29,
                num_atom_types=5,
                latent_dim=32
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Create generator
            self.generator = MolecularGenerator(
                self.model, self.preprocessor, self.device
            )
            
            # Create visualizer
            self.visualizer = create_visualizer()
            
            st.session_state.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            logger.error(f"Error loading model: {e}")
            return False
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🧬 3D Molecular Structure Generator</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        <p style="text-align: center; color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
        Generate novel 3D molecular structures using a trained Variational Autoencoder (VAE) model.
        Explore the latent space of molecular geometry and create new compounds with AI.
        </p>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.markdown('<h2 class="sub-header">⚙️ Generation Settings</h2>', 
                           unsafe_allow_html=True)
        
        # Model loading
        st.sidebar.subheader("📂 Model Configuration")
        model_path = st.sidebar.text_input(
            "Model Path",
            value="models/best_model.pth",
            help="Path to the trained model checkpoint"
        )
        
        if st.sidebar.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                if self.load_model(model_path):
                    st.sidebar.success("✅ Model loaded successfully!")
                else:
                    st.sidebar.error("❌ Failed to load model")
        
        # Generation parameters
        st.sidebar.subheader("🎲 Generation Parameters")
        
        num_molecules = st.sidebar.slider(
            "Number of Molecules",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of molecules to generate"
        )
        
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Controls diversity (higher = more diverse)"
        )
        
        sampling_strategy = st.sidebar.selectbox(
            "Sampling Strategy",
            options=["random", "interpolate", "cluster"],
            value="random",
            help="Strategy for sampling from latent space"
        )
        
        # Visualization settings
        st.sidebar.subheader("🎨 Visualization Settings")
        
        show_bonds = st.sidebar.checkbox(
            "Show Bonds",
            value=True,
            help="Display chemical bonds between atoms"
        )
        
        show_labels = st.sidebar.checkbox(
            "Show Atom Labels",
            value=True,
            help="Display element symbols on atoms"
        )
        
        viz_backend = st.sidebar.selectbox(
            "Visualization Backend",
            options=["plotly", "py3dmol"],
            value="plotly",
            help="Choose visualization library"
        )
        
        return {
            'num_molecules': num_molecules,
            'temperature': temperature,
            'sampling_strategy': sampling_strategy,
            'show_bonds': show_bonds,
            'show_labels': show_labels,
            'viz_backend': viz_backend
        }
    
    def generate_molecules(self, params: dict):
        """Generate molecules using the current model."""
        if not st.session_state.model_loaded:
            st.error("❌ Please load a model first!")
            return
        
        with st.spinner("🧬 Generating molecules..."):
            try:
                molecules = self.generator.generate_molecules(
                    num_samples=params['num_molecules'],
                    temperature=params['temperature'],
                    sampling_strategy=params['sampling_strategy']
                )
                
                st.session_state.generated_molecules = molecules
                
                if molecules:
                    st.success(f"✅ Successfully generated {len(molecules)} molecules!")
                else:
                    st.warning("⚠️ No molecules were generated. Try different parameters.")
                    
            except Exception as e:
                st.error(f"❌ Error generating molecules: {str(e)}")
                logger.error(f"Error generating molecules: {e}")
    
    def visualize_molecule(self, molecule: dict, params: dict):
        """Visualize a single molecule."""
        if params['viz_backend'] == 'plotly':
            fig = self.visualizer.visualize_plotly(
                molecule,
                show_bonds=params['show_bonds'],
                show_atom_labels=params['show_labels']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif params['viz_backend'] == 'py3dmol':
            # Create py3Dmol visualization
            view = self.visualizer.visualize_py3dmol(
                molecule,
                width=800,
                height=600,
                show_bonds=params['show_bonds']
            )
            
            # Convert to HTML for display
            html = view._make_html()
            st.components.v1.html(html, height=600)
    
    def display_molecule_info(self, molecule: dict, idx: int):
        """Display information about a molecule."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Atoms", molecule['num_atoms'])
        
        with col2:
            # Calculate molecular weight (simplified)
            atomic_weights = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998}
            weight = sum(atomic_weights.get(num, 12.011) for num in molecule['atomic_nums'])
            st.metric("Weight", f"{weight:.2f} Da")
        
        with col2:
            # Count atom types
            atom_counts = {}
            for num in molecule['atomic_nums']:
                element = self.visualizer._get_element_symbol(num)
                atom_counts[element] = atom_counts.get(element, 0) + 1
            
            formula = ''.join([f"{element}{count}" for element, count in sorted(atom_counts.items())])
            st.metric("Formula", formula)
        
        with col3:
            # Calculate bounds
            coords = molecule['coordinates']
            if len(coords) > 0:
                size = np.max(coords) - np.min(coords)
                st.metric("Size", f"{size:.2f} Å")
        
        # Display atomic composition
        with st.expander("🔬 Atomic Composition"):
            composition_data = []
            for num in molecule['atomic_nums']:
                element = self.visualizer._get_element_symbol(num)
                composition_data.append({"Element": element, "Atomic Number": num})
            
            df = pd.DataFrame(composition_data)
            st.dataframe(df, use_container_width=True)
        
        # Display coordinates
        with st.expander("📐 3D Coordinates"):
            coord_data = []
            for i, (num, coord) in enumerate(zip(molecule['atomic_nums'], molecule['coordinates'])):
                element = self.visualizer._get_element_symbol(num)
                coord_data.append({
                    "Atom": i,
                    "Element": element,
                    "X": f"{coord[0]:.4f}",
                    "Y": f"{coord[1]:.4f}",
                    "Z": f"{coord[2]:.4f}"
                })
            
            df = pd.DataFrame(coord_data)
            st.dataframe(df, use_container_width=True)
    
    def download_molecule_data(self, molecule: dict, idx: int):
        """Provide download options for molecule data."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as JSON
            json_data = {
                'atomic_nums': molecule['atomic_nums'].tolist(),
                'coordinates': molecule['coordinates'].tolist(),
                'num_atoms': int(molecule['num_atoms'])
            }
            
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name=f"molecule_{idx}.json",
                mime="application/json"
            )
        
        with col2:
            # Download as XYZ
            xyz_lines = [str(molecule['num_atoms']), f"Generated Molecule {idx}"]
            
            for atomic_num, coord in zip(molecule['atomic_nums'], molecule['coordinates']):
                element = self.visualizer._get_element_symbol(atomic_num)
                xyz_lines.append(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
            
            xyz_str = "\n".join(xyz_lines)
            st.download_button(
                label="🧪 Download XYZ",
                data=xyz_str,
                file_name=f"molecule_{idx}.xyz",
                mime="chemical/x-xyz"
            )
    
    def render_main_content(self, params: dict):
        """Render the main content area."""
        # Generation section
        st.markdown('<h2 class="sub-header">🎯 Generate Molecules</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info("🔬 Click the button below to generate new molecular structures using the loaded VAE model.")
        
        with col2:
            if st.button("🚀 Generate Molecules", type="primary", use_container_width=True):
                self.generate_molecules(params)
        
        with col3:
            if st.button("🔄 Generate Diverse Set", use_container_width=True):
                if st.session_state.model_loaded:
                    with st.spinner("Generating diverse set..."):
                        try:
                            molecules = self.generator.generate_diverse_set(
                                num_molecules=params['num_molecules'],
                                diversity_factor=2.0
                            )
                            st.session_state.generated_molecules = molecules
                            st.success(f"✅ Generated diverse set of {len(molecules)} molecules!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        # Display generated molecules
        if st.session_state.generated_molecules:
            st.markdown('<h2 class="sub-header">🧪 Generated Molecules</h2>', 
                       unsafe_allow_html=True)
            
            molecules = st.session_state.generated_molecules
            
            # Molecule selector
            selected_idx = st.selectbox(
                "Select Molecule to View:",
                options=range(len(molecules)),
                format_func=lambda i: f"Molecule {i+1} ({molecules[i]['num_atoms']} atoms)",
                index=0
            )
            
            selected_molecule = molecules[selected_idx]
            
            # Display selected molecule
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"🔬 Molecule {selected_idx + 1}")
                self.visualize_molecule(selected_molecule, params)
            
            with col2:
                st.subheader("📊 Molecule Info")
                self.display_molecule_info(selected_molecule, selected_idx)
                
                st.subheader("💾 Download")
                self.download_molecule_data(selected_molecule, selected_idx)
            
            # Gallery view
            if len(molecules) > 1:
                st.markdown('<h2 class="sub-header">🖼️ Molecule Gallery</h2>', 
                           unsafe_allow_html=True)
                
                # Create grid of molecules
                cols = st.columns(min(3, len(molecules)))
                
                for i, mol in enumerate(molecules[:6]):  # Show max 6 molecules
                    with cols[i % 3]:
                        st.subheader(f"Mol {i+1}")
                        
                        # Small plotly visualization
                        fig = self.visualizer.visualize_plotly(
                            mol,
                            show_bonds=params['show_bonds'],
                            show_atom_labels=False,
                            width=300,
                            height=300
                        )
                        
                        fig.update_layout(
                            title=f"{mol['num_atoms']} atoms",
                            font=dict(size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Quick info
                        st.write(f"Atoms: {mol['num_atoms']}")
                        
                        if st.button(f"View Mol {i+1}", key=f"view_{i}"):
                            st.session_state.selected_molecule_idx = i
                            st.rerun()
    
    def render_footer(self):
        """Render the application footer."""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; padding: 1rem;">
            <p>🧬 3D Molecular Structure Generator | Built with PyTorch, RDKit, and Streamlit</p>
            <p>Generative AI for Molecular Design | Research Project</p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the main application."""
        # Render components
        self.render_header()
        params = self.render_sidebar()
        
        # Main content
        if st.session_state.model_loaded:
            self.render_main_content(params)
        else:
            st.markdown("""
            <div class="warning-message">
                <h3>🚀 Getting Started</h3>
                <p>Please load a trained model using the sidebar to begin generating molecules.</p>
                <p>If you don't have a trained model, please run the training script first:</p>
                <code>python -m src.train</code>
            </div>
            """, unsafe_allow_html=True)
            
            # Show demo with sample molecule
            st.markdown('<h2 class="sub-header">🎬 Demo Visualization</h2>', 
                       unsafe_allow_html=True)
            
            # Create sample molecule for demo
            sample_molecule = {
                'atomic_nums': np.array([6, 6, 6, 6, 1, 1, 1, 1]),  # Benzene
                'coordinates': np.array([
                    [1.40, 0.00, 0.00],
                    [0.70, 1.21, 0.00],
                    [-0.70, 1.21, 0.00],
                    [-1.40, 0.00, 0.00],
                    [-0.70, -1.21, 0.00],
                    [0.70, -1.21, 0.00],
                    [2.48, 0.00, 0.00],
                    [-2.48, 0.00, 0.00]
                ]),
                'num_atoms': 8
            }
            
            if self.visualizer is None:
                self.visualizer = create_visualizer()
            
            fig = self.visualizer.visualize_plotly(
                sample_molecule,
                show_bonds=True,
                show_atom_labels=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 This is a sample benzene molecule. Load a model to generate new structures!")
        
        self.render_footer()


def main():
    """Main function to run the Streamlit app."""
    app = MolecularApp()
    app.run()


if __name__ == "__main__":
    main()
