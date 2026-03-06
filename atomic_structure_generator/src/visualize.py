"""
3D visualization utilities for molecular structures.

This module provides functions for visualizing molecular structures in 3D
using py3Dmol, matplotlib, and other visualization tools.
"""

import numpy as np
import py3Dmol
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union
import logging
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class MolecularVisualizer:
    """
    Visualizer for molecular structures.
    
    Supports multiple visualization backends including py3Dmol, matplotlib,
    and Plotly for interactive 3D visualization.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        # Atom colors for visualization
        self.atom_colors = {
            1: 'white',   # H
            6: 'gray',    # C
            7: 'blue',    # N
            8: 'red',     # O
            9: 'green',   # F
            12: 'magenta', # Mg
            14: 'orange', # Si
            16: 'yellow', # S
            17: 'purple', # Cl
        }
        
        # Atom radii for visualization
        self.atom_radii = {
            1: 0.31,   # H
            6: 0.76,   # C
            7: 0.71,   # N
            8: 0.66,   # O
            9: 0.57,   # F
            12: 1.60,  # Mg
            14: 1.11,  # Si
            16: 1.05,  # S
            17: 1.02,  # Cl
        }
        
        logger.info("Initialized molecular visualizer")
    
    def visualize_py3dmol(
        self,
        molecule: Dict,
        width: int = 400,
        height: int = 400,
        style: str = "stick",
        show_bonds: bool = True,
        background_color: str = "white"
    ) -> py3Dmol.view:
        """
        Create 3D visualization using py3Dmol.
        
        Args:
            molecule: Molecule data with atomic_nums and coordinates
            width: Viewer width in pixels
            height: Viewer height in pixels
            style: Visualization style ("stick", "sphere", "line")
            show_bonds: Whether to show bonds
            background_color: Background color
            
        Returns:
            py3Dmol view object
        """
        view = py3Dmol.view(width=width, height=height)
        
        # Set background color
        view.setBackgroundColor(background_color)
        
        # Get molecule data
        atomic_nums = molecule['atomic_nums']
        coordinates = molecule['coordinates']
        
        # Create XYZ string
        xyz_lines = [str(len(atomic_nums)), "Generated Molecule"]
        
        for i, (atomic_num, coord) in enumerate(zip(atomic_nums, coordinates)):
            element = self._get_element_symbol(atomic_num)
            xyz_lines.append(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        
        xyz_string = "\n".join(xyz_lines)
        
        # Add molecule to view
        view.addModel(xyz_string, "xyz")
        
        # Set style
        if style == "stick":
            view.setStyle({'stick': {}})
        elif style == "sphere":
            view.setStyle({'sphere': {'radius': 0.5}})
        elif style == "line":
            view.setStyle({'line': {}})
        else:
            view.setStyle({'stick': {}})
        
        # Add bonds if requested
        if show_bonds:
            view.addBonds()
        
        # Zoom to fit
        view.zoomTo()
        
        return view
    
    def visualize_matplotlib(
        self,
        molecule: Dict,
        figsize: Tuple[int, int] = (10, 8),
        show_atom_labels: bool = True,
        atom_size: float = 100,
        bond_width: float = 2.0,
        elev: int = 20,
        azim: int = 45
    ) -> plt.Figure:
        """
        Create 3D visualization using matplotlib.
        
        Args:
            molecule: Molecule data
            figsize: Figure size
            show_atom_labels: Whether to show atom labels
            atom_size: Size of atom spheres
            bond_width: Width of bond lines
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get molecule data
        atomic_nums = molecule['atomic_nums']
        coordinates = molecule['coordinates']
        
        # Plot atoms
        for i, (atomic_num, coord) in enumerate(zip(atomic_nums, coordinates)):
            color = self.atom_colors.get(atomic_num, 'gray')
            element = self._get_element_symbol(atomic_num)
            
            ax.scatter(
                coord[0], coord[1], coord[2],
                c=color, s=atom_size, alpha=0.8, edgecolors='black'
            )
            
            if show_atom_labels:
                ax.text(
                    coord[0], coord[1], coord[2],
                    element, fontsize=8, ha='center', va='center'
                )
        
        # Plot bonds
        if len(atomic_nums) > 1:
            bonds = self._detect_bonds(atomic_nums, coordinates)
            for bond in bonds:
                i, j = bond
                coords_i = coordinates[i]
                coords_j = coordinates[j]
                
                ax.plot(
                    [coords_i[0], coords_j[0]],
                    [coords_i[1], coords_j[1]],
                    [coords_i[2], coords_j[2]],
                    'k-', linewidth=bond_width, alpha=0.6
                )
        
        # Set labels and title
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Generated Molecule ({len(atomic_nums)} atoms)')
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Make axes equal
        self._set_axes_equal(ax)
        
        plt.tight_layout()
        return fig
    
    def visualize_plotly(
        self,
        molecule: Dict,
        show_bonds: bool = True,
        show_atom_labels: bool = True,
        width: int = 800,
        height: int = 600
    ) -> go.Figure:
        """
        Create interactive 3D visualization using Plotly.
        
        Args:
            molecule: Molecule data
            show_bonds: Whether to show bonds
            show_atom_labels: Whether to show atom labels
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Get molecule data
        atomic_nums = molecule['atomic_nums']
        coordinates = molecule['coordinates']
        
        # Create scatter plot for atoms
        colors = [self.atom_colors.get(num, 'gray') for num in atomic_nums]
        symbols = [self._get_element_symbol(num) for num in atomic_nums]
        
        fig = go.Figure()
        
        # Add atoms
        fig.add_trace(go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode='markers+text' if show_atom_labels else 'markers',
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=symbols if show_atom_labels else None,
            textposition='middle center',
            name='Atoms'
        ))
        
        # Add bonds
        if show_bonds and len(atomic_nums) > 1:
            bonds = self._detect_bonds(atomic_nums, coordinates)
            
            for bond in bonds:
                i, j = bond
                bond_coords = np.array([coordinates[i], coordinates[j]])
                
                fig.add_trace(go.Scatter3d(
                    x=bond_coords[:, 0],
                    y=bond_coords[:, 1],
                    z=bond_coords[:, 2],
                    mode='lines',
                    line=dict(width=4, color='gray'),
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=f'Generated Molecule ({len(atomic_nums)} atoms)',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube'
            ),
            width=width,
            height=height,
            showlegend=True
        )
        
        return fig
    
    def visualize_multiple_molecules(
        self,
        molecules: List[Dict],
        backend: str = "plotly",
        max_molecules: int = 9,
        **kwargs
    ) -> Union[go.Figure, plt.Figure]:
        """
        Visualize multiple molecules in a grid.
        
        Args:
            molecules: List of molecules to visualize
            backend: Visualization backend ("plotly", "matplotlib")
            max_molecules: Maximum number of molecules to show
            **kwargs: Additional arguments for visualization
            
        Returns:
            Figure with multiple subplots
        """
        molecules = molecules[:max_molecules]
        n_mols = len(molecules)
        
        if backend == "plotly":
            return self._visualize_multiple_plotly(molecules, **kwargs)
        else:
            return self._visualize_multiple_matplotlib(molecules, **kwargs)
    
    def _visualize_multiple_plotly(
        self,
        molecules: List[Dict],
        cols: int = 3,
        **kwargs
    ) -> go.Figure:
        """Visualize multiple molecules using Plotly subplots."""
        from plotly.subplots import make_subplots
        
        n_mols = len(molecules)
        rows = (n_mols + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=[f'Molecule {i+1}' for i in range(n_mols)]
        )
        
        for idx, mol in enumerate(molecules):
            row = idx // cols + 1
            col = idx % cols + 1
            
            atomic_nums = mol['atomic_nums']
            coordinates = mol['coordinates']
            colors = [self.atom_colors.get(num, 'gray') for num in atomic_nums]
            
            fig.add_trace(
                go.Scatter3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    z=coordinates[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors,
                        line=dict(width=1, color='black')
                    ),
                    name=f'Mol {idx+1}',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f'Generated Molecules ({n_mols} molecules)',
            height=300 * rows
        )
        
        return fig
    
    def _visualize_multiple_matplotlib(
        self,
        molecules: List[Dict],
        cols: int = 3,
        figsize: Tuple[int, int] = (15, 10),
        **kwargs
    ) -> plt.Figure:
        """Visualize multiple molecules using matplotlib subplots."""
        n_mols = len(molecules)
        rows = (n_mols + cols - 1) // cols
        
        fig = plt.figure(figsize=figsize)
        
        for idx, mol in enumerate(molecules):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            
            atomic_nums = mol['atomic_nums']
            coordinates = mol['coordinates']
            
            # Plot atoms
            for atomic_num, coord in zip(atomic_nums, coordinates):
                color = self.atom_colors.get(atomic_num, 'gray')
                ax.scatter(coord[0], coord[1], coord[2], c=color, s=50, alpha=0.8)
            
            # Plot bonds
            if len(atomic_nums) > 1:
                bonds = self._detect_bonds(atomic_nums, coordinates)
                for bond in bonds:
                    i, j = bond
                    coords_i = coordinates[i]
                    coords_j = coordinates[j]
                    
                    ax.plot(
                        [coords_i[0], coords_j[0]],
                        [coords_i[1], coords_j[1]],
                        [coords_i[2], coords_j[2]],
                        'k-', linewidth=1, alpha=0.6
                    )
            
            ax.set_title(f'Mol {idx+1} ({len(atomic_nums)} atoms)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set equal aspect ratio
            self._set_axes_equal(ax)
        
        plt.tight_layout()
        return fig
    
    def _detect_bonds(
        self,
        atomic_nums: np.ndarray,
        coordinates: np.ndarray,
        tolerance: float = 1.6
    ) -> List[Tuple[int, int]]:
        """
        Detect bonds between atoms based on distance.
        
        Args:
            atomic_nums: Array of atomic numbers
            coordinates: Array of 3D coordinates
            tolerance: Distance tolerance for bond detection
            
        Returns:
            List of bond tuples (atom_index1, atom_index2)
        """
        bonds = []
        n_atoms = len(atomic_nums)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Calculate distance
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                
                # Get covalent radii sum
                r_i = self.atom_radii.get(atomic_nums[i], 0.76)
                r_j = self.atom_radii.get(atomic_nums[j], 0.76)
                max_dist = (r_i + r_j) * tolerance
                
                if dist < max_dist:
                    bonds.append((i, j))
        
        return bonds
    
    def _get_element_symbol(self, atomic_num: int) -> str:
        """Get element symbol from atomic number."""
        elements = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
            12: 'Mg', 14: 'Si', 16: 'S', 17: 'Cl'
        }
        return elements.get(atomic_num, f'X{atomic_num}')
    
    def _set_axes_equal(self, ax):
        """Set equal aspect ratio for 3D axes."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    def save_visualization(
        self,
        fig: Union[go.Figure, plt.Figure, py3Dmol.view],
        filepath: str,
        **kwargs
    ):
        """
        Save visualization to file.
        
        Args:
            fig: Figure to save
            filepath: Output file path
            **kwargs: Additional arguments for saving
        """
        if isinstance(fig, go.Figure):
            if filepath.endswith('.html'):
                fig.write_html(filepath)
            else:
                fig.write_image(filepath, **kwargs)
        elif isinstance(fig, plt.Figure):
            fig.savefig(filepath, **kwargs)
        elif isinstance(fig, py3Dmol.view):
            # For py3Dmol, save as PNG
            png_data = fig.png()
            with open(filepath, 'wb') as f:
                f.write(png_data)
        
        logger.info(f"Saved visualization to {filepath}")


def create_visualizer() -> MolecularVisualizer:
    """
    Create and return a MolecularVisualizer instance.
    
    Returns:
        MolecularVisualizer instance
    """
    return MolecularVisualizer()


if __name__ == "__main__":
    # Test the visualizer
    visualizer = create_visualizer()
    
    # Create a sample molecule (water)
    sample_molecule = {
        'atomic_nums': np.array([8, 1, 1]),  # O, H, H
        'coordinates': np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0]
        ])
    }
    
    # Test different visualization methods
    print("Testing py3Dmol visualization...")
    view = visualizer.visualize_py3dmol(sample_molecule)
    
    print("Testing matplotlib visualization...")
    fig = visualizer.visualize_matplotlib(sample_molecule)
    
    print("Testing plotly visualization...")
    plotly_fig = visualizer.visualize_plotly(sample_molecule)
    
    print("Visualizations created successfully!")
