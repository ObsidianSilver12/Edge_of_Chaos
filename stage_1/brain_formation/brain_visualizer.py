"""
Brain Visualizer - Complete 3D visualization system for brain structure and mirror grid
Shows anatomical brain structure, mycelial network, nodes, seeds, and mirror grid entanglement
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BrainVisualizer:
    """Advanced 3D Brain visualization showing structure, mycelial network, and mirror grid"""
    
    def __init__(self, output_dir: str = "output/brain_visuals"):
        """Initialize the brain visualizer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Brain region colors
        self.region_colors = {
            'frontal_lobe': '#FF6B6B',      # Red
            'parietal_lobe': '#4ECDC4',     # Teal
            'temporal_lobe': '#45B7D1',     # Blue
            'occipital_lobe': '#96CEB4',    # Green
            'cerebellum': '#FECA57',        # Yellow
            'brainstem': '#FF9FF3',         # Pink
            'hippocampus': '#54A0FF',       # Light blue
            'amygdala': '#FF6B35',          # Orange
            'thalamus': '#A55EEA',          # Purple
            'hypothalamus': '#26DE81'       # Light green
        }
        
        # Node status colors
        self.node_colors = {
            'active': '#00FF00',     # Green
            'inactive': '#808080',   # Gray
            'archived': '#FF0000'    # Red
        }
        
        # Mycelial seed colors
        self.seed_colors = {
            'active': '#FFD700',     # Gold
            'inactive': '#FFA500',   # Orange
            'dormant': '#8B4513'     # Brown
        }
        
        logger.info(f"BrainVisualizer initialized with output directory: {output_dir}")
    
    def _extract_brain_data(self, brain_structure) -> Dict[str, Any]:
        """Extract all required data from brain structure with HARD FAIL for missing critical data"""
        
        if not brain_structure:
            raise RuntimeError("CRITICAL: Brain structure object is None")
        
        logger.info(f"Extracting data from brain structure: {getattr(brain_structure, 'brain_id', 'UNKNOWN')}")
        
        # Extract basic structure info
        brain_data = {
            'brain_id': getattr(brain_structure, 'brain_id', 'unknown'),
            'creation_time': getattr(brain_structure, 'creation_time', None),
            'grid_dimensions': getattr(brain_structure, 'grid_dimensions', (50, 50, 50)),
            'regions': getattr(brain_structure, 'regions', {}),
            'sub_regions': getattr(brain_structure, 'sub_regions', {}),
            'nodes': getattr(brain_structure, 'nodes', {}),
            'mycelial_seeds': getattr(brain_structure, 'mycelial_seeds', {}),
            'whole_brain_matrix': getattr(brain_structure, 'whole_brain_matrix', None),
            'node_placement_matrix': getattr(brain_structure, 'node_placement_matrix', None),
            'seed_placement_matrix': getattr(brain_structure, 'seed_placement_matrix', None),
            'mirror_grid_enabled': getattr(brain_structure, 'mirror_grid_enabled', False),
            'mirror_grid_matrix': getattr(brain_structure, 'mirror_grid_matrix', None),
            'mirror_grid_entanglement': getattr(brain_structure, 'mirror_grid_entanglement', {}),
            'energy_storage': getattr(brain_structure, 'energy_storage', None)
        }
        
        # HARD FAIL if critical data missing
        if brain_data['grid_dimensions'] is None:
            raise RuntimeError("CRITICAL: Grid dimensions are required")
        
        logger.info(f"Brain data extracted successfully: {brain_data['brain_id']}")
        return brain_data

    def create_3d_brain_structure(self, brain_data: Dict[str, Any]) -> go.Figure:
        """Create 3D visualization of brain anatomical structure"""
        
        logger.info("Creating 3D brain structure visualization")
        
        try:
            fig = go.Figure()
            
            # Get grid dimensions
            grid_dims = brain_data['grid_dimensions']
            x_dim, y_dim, z_dim = grid_dims
            
            # Create brain outline (simplified brain shape)
            brain_outline = self._create_brain_outline(x_dim, y_dim, z_dim)
            
            # Add brain outline
            fig.add_trace(go.Mesh3d(
                x=brain_outline['x'],
                y=brain_outline['y'],
                z=brain_outline['z'],
                i=brain_outline['i'],
                j=brain_outline['j'],
                k=brain_outline['k'],
                opacity=0.3,
                color='lightblue',
                name='Brain Outline'
            ))
            
            # Add brain regions as colored volumes
            regions = brain_data.get('regions', {})
            for region_name, region_data in regions.items():
                if isinstance(region_data, dict) and 'volume_coords' in region_data:
                    region_color = self.region_colors.get(region_name, '#888888')
                    coords = region_data['volume_coords']
                    
                    fig.add_trace(go.Scatter3d(
                        x=coords.get('x', []),
                        y=coords.get('y', []),
                        z=coords.get('z', []),
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=region_color,
                            opacity=0.7
                        ),
                        name=region_name.replace('_', ' ').title()
                    ))
            
            # If no regions data, create simplified brain regions
            if not regions:
                self._add_simplified_brain_regions(fig, x_dim, y_dim, z_dim)
            
            fig.update_layout(
                title=f"3D Brain Anatomical Structure<br>Grid: {x_dim}x{y_dim}x{z_dim}",
                scene=dict(
                    xaxis_title="X (Left-Right)",
                    yaxis_title="Y (Front-Back)",
                    zaxis_title="Z (Bottom-Top)",
                    aspectmode='cube',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=800
            )
            
            logger.info("3D brain structure visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create mycelial network visualization: {e}")
            raise RuntimeError(f"CRITICAL: Mycelial network visualization creation failed: {e}") from e

    def _add_mycelial_connections(self, fig: go.Figure, brain_data: Dict[str, Any]):
        """Add mycelial network connections between seeds"""
        
        mycelial_seeds = brain_data.get('mycelial_seeds', {})
        if not mycelial_seeds:
            return
        
        # Get seed positions
        seed_positions = []
        for seed_data in mycelial_seeds.values():
            if isinstance(seed_data, dict):
                position = seed_data.get('position', [0, 0, 0])
                seed_positions.append(position)
        
        if len(seed_positions) < 2:
            return
        
        # Create connections between nearby seeds (simplified network)
        positions = np.array(seed_positions)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                # Connect if within reasonable distance
                if distance < 15:  # Adjust threshold as needed
                    fig.add_trace(go.Scatter3d(
                        x=[positions[i][0], positions[j][0]],
                        y=[positions[i][1], positions[j][1]],
                        z=[positions[i][2], positions[j][2]],
                        mode='lines',
                        line=dict(
                            color='rgba(255,215,0,0.3)',  # Semi-transparent gold
                            width=2
                        ),
                        showlegend=False
                    ))

    def _add_example_nodes(self, fig: go.Figure, grid_dims: Tuple[int, int, int]):
        """Add example nodes when no real node data is available"""
        
        x_dim, y_dim, z_dim = grid_dims
        
        # Create random node positions
        n_nodes = 200
        x_nodes = np.random.uniform(0, x_dim, n_nodes)
        y_nodes = np.random.uniform(0, y_dim, n_nodes)
        z_nodes = np.random.uniform(0, z_dim, n_nodes)
        
        # Assign random statuses
        statuses = np.random.choice(['active', 'inactive', 'archived'], n_nodes, p=[0.3, 0.6, 0.1])
        
        for status, color in self.node_colors.items():
            mask = statuses == status
            if np.any(mask):
                fig.add_trace(go.Scatter3d(
                    x=x_nodes[mask],
                    y=y_nodes[mask],
                    z=z_nodes[mask],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=color,
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name=f'{status.title()} Nodes ({np.sum(mask)})'
                ))

    def _add_example_mycelial_network(self, fig: go.Figure, grid_dims: Tuple[int, int, int]):
        """Add example mycelial network when no real data is available"""
        
        x_dim, y_dim, z_dim = grid_dims
        
        # Create network-like distribution of seeds
        n_seeds = 50
        
        # Create clusters of seeds (more realistic distribution)
        n_clusters = 8
        cluster_centers = [
            (np.random.uniform(0, x_dim), np.random.uniform(0, y_dim), np.random.uniform(0, z_dim))
            for _ in range(n_clusters)
        ]
        
        seed_positions = []
        seed_statuses = []
        
        for center in cluster_centers:
            cluster_size = np.random.randint(3, 8)
            for _ in range(cluster_size):
                # Position around cluster center
                offset = np.random.normal(0, 5, 3)  # Small variance around center
                position = [
                    max(0, min(x_dim, center[0] + offset[0])),
                    max(0, min(y_dim, center[1] + offset[1])),
                    max(0, min(z_dim, center[2] + offset[2]))
                ]
                seed_positions.append(position)
                seed_statuses.append(np.random.choice(['active', 'inactive', 'dormant'], p=[0.4, 0.4, 0.2]))
        
        # Add seeds to plot
        positions = np.array(seed_positions)
        statuses = np.array(seed_statuses)
        
        for status, color in self.seed_colors.items():
            mask = statuses == status
            if np.any(mask):
                fig.add_trace(go.Scatter3d(
                    x=positions[mask, 0],
                    y=positions[mask, 1],
                    z=positions[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.9,
                        symbol='diamond'
                    ),
                    name=f'{status.title()} Seeds ({np.sum(mask)})'
                ))
        
        # Add connections between nearby seeds
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < 12:  # Connect nearby seeds
                    fig.add_trace(go.Scatter3d(
                        x=[positions[i][0], positions[j][0]],
                        y=[positions[i][1], positions[j][1]],
                        z=[positions[i][2], positions[j][2]],
                        mode='lines',
                        line=dict(
                            color='rgba(255,215,0,0.2)',
                            width=1
                        ),
                        showlegend=False
                    ))

    def create_mirror_grid_visualization(self, brain_data: Dict[str, Any]) -> go.Figure:
        """Create 3D visualization of mirror grid and entanglement"""
        
        logger.info("Creating mirror grid visualization")
        
        try:
            fig = go.Figure()
            
            mirror_enabled = brain_data.get('mirror_grid_enabled', False)
            mirror_matrix = brain_data.get('mirror_grid_matrix')
            entanglement = brain_data.get('mirror_grid_entanglement', {})
            grid_dims = brain_data['grid_dimensions']
            
            if not mirror_enabled or mirror_matrix is None:
                raise RuntimeError("CRITICAL: Mirror grid data is required for brain visualization but not available. Cannot create fake visualization data.")
            else:
                # Visualize actual mirror grid
                self._visualize_actual_mirror_grid(fig, mirror_matrix, entanglement)
            
            fig.update_layout(
                title="Mirror Grid System & Entanglement",
                scene=dict(
                    xaxis_title="X (Mirror Space)",
                    yaxis_title="Y (Mirror Space)",
                    zaxis_title="Z (Mirror Space)",
                    aspectmode='cube',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=800
            )
            
            logger.info("Mirror grid visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create mirror grid visualization: {e}")
            raise RuntimeError(f"CRITICAL: Mirror grid visualization creation failed: {e}") from e

    def _add_example_mirror_grid(self, fig: go.Figure, grid_dims: Tuple[int, int, int]):
        """Add example mirror grid when no real data is available"""
        
        x_dim, y_dim, z_dim = grid_dims
        
        # Create mirror grid points (memory fragments)
        n_fragments = 100
        
        # Mirror fragments often cluster around memory centers
        fragment_centers = [
            (x_dim * 0.2, y_dim * 0.3, z_dim * 0.4),  # Episodic memory
            (x_dim * 0.8, y_dim * 0.7, z_dim * 0.6),  # Semantic memory
            (x_dim * 0.5, y_dim * 0.2, z_dim * 0.8),  # Working memory
            (x_dim * 0.3, y_dim * 0.8, z_dim * 0.3),  # Emotional memory
        ]
        
        fragment_positions = []
        fragment_types = []
        
        for i, center in enumerate(fragment_centers):
            cluster_size = np.random.randint(15, 30)
            for _ in range(cluster_size):
                offset = np.random.normal(0, 8, 3)
                position = [
                    max(0, min(x_dim, center[0] + offset[0])),
                    max(0, min(y_dim, center[1] + offset[1])),
                    max(0, min(z_dim, center[2] + offset[2]))
                ]
                fragment_positions.append(position)
                fragment_types.append(f'memory_type_{i}')
        
        # Visualize memory fragments
        positions = np.array(fragment_positions)
        fragment_colors = ['#FF69B4', '#87CEEB', '#98FB98', '#DDA0DD']  # Different memory types
        
        for i, color in enumerate(fragment_colors):
            mask = np.array(fragment_types) == f'memory_type_{i}'
            if np.any(mask):
                fig.add_trace(go.Scatter3d(
                    x=positions[mask, 0],
                    y=positions[mask, 1],
                    z=positions[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.7,
                        symbol='square'
                    ),
                    name=f'Memory Type {i+1} ({np.sum(mask)})'
                ))
        
        # Add entanglement connections (lines between main brain and mirror)
        self._add_example_entanglement_lines(fig, grid_dims, positions)

    def _visualize_actual_mirror_grid(self, fig: go.Figure, mirror_matrix, entanglement: Dict):
        """Visualize actual mirror grid data"""
        
        # Find non-zero positions in mirror matrix (memory fragments)
        if hasattr(mirror_matrix, 'shape') and mirror_matrix is not None:
            non_zero_positions = np.where(mirror_matrix != 0)
            
            if len(non_zero_positions[0]) > 0:
                x_coords = non_zero_positions[0]
                y_coords = non_zero_positions[1]
                z_coords = non_zero_positions[2]
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='#FF69B4',  # Pink for memory fragments
                        opacity=0.8,
                        symbol='square'
                    ),
                    name=f'Memory Fragments ({len(x_coords)})'
                ))
        
        # Visualize entanglement connections
        if entanglement:
            self._add_entanglement_connections(fig, entanglement)

    def _add_example_entanglement_lines(self, fig: go.Figure, grid_dims: Tuple[int, int, int], mirror_positions):
        """Add example entanglement lines between brain and mirror"""
        
        x_dim, y_dim, z_dim = grid_dims
        
        # Create some brain positions (simplified)
        n_connections = 20
        brain_positions = [
            (np.random.uniform(0, x_dim), np.random.uniform(0, y_dim), np.random.uniform(0, z_dim))
            for _ in range(n_connections)
        ]
        
        # Connect random brain positions to mirror positions
        for i in range(min(n_connections, len(mirror_positions))):
            brain_pos = brain_positions[i]
            mirror_pos = mirror_positions[i]
            
            fig.add_trace(go.Scatter3d(
                x=[brain_pos[0], mirror_pos[0]],
                y=[brain_pos[1], mirror_pos[1]],
                z=[brain_pos[2], mirror_pos[2]],
                mode='lines',
                line=dict(
                    color='rgba(255,0,255,0.4)',  # Semi-transparent magenta
                    width=1,
                    dash='dash'
                ),
                showlegend=False
            ))

    def _add_entanglement_connections(self, fig: go.Figure, entanglement: Dict):
        """Add actual entanglement connections"""
        
        for brain_coord, mirror_coord in entanglement.items():
            if isinstance(brain_coord, (list, tuple)) and isinstance(mirror_coord, (list, tuple)):
                fig.add_trace(go.Scatter3d(
                    x=[brain_coord[0], mirror_coord[0]],
                    y=[brain_coord[1], mirror_coord[1]],
                    z=[brain_coord[2], mirror_coord[2]],
                    mode='lines',
                    line=dict(
                        color='rgba(255,0,255,0.6)',
                        width=2,
                        dash='dash'
                    ),
                    showlegend=False
                ))

    def create_combined_brain_overview(self, brain_data: Dict[str, Any]) -> go.Figure:
        """Create combined overview showing brain structure, nodes, seeds, and mirror grid"""
        
        logger.info("Creating combined brain overview")
        
        try:
            fig = go.Figure()
            
            grid_dims = brain_data['grid_dimensions']
            x_dim, y_dim, z_dim = grid_dims
            
            # Add brain outline (transparent)
            brain_outline = self._create_brain_outline(x_dim, y_dim, z_dim)
            fig.add_trace(go.Mesh3d(
                x=brain_outline['x'],
                y=brain_outline['y'],
                z=brain_outline['z'],
                i=brain_outline['i'],
                j=brain_outline['j'],
                k=brain_outline['k'],
                opacity=0.1,
                color='lightblue',
                name='Brain Structure',
                showlegend=True
            ))
            
            # Add nodes (smaller for overview) - HARD FAIL if missing
            nodes = brain_data.get('nodes', {})
            if not nodes:
                raise RuntimeError("CRITICAL: Brain nodes data is required for brain visualization but not available. Cannot create fake visualization data.")
                
            for status, color in self.node_colors.items():
                status_nodes = []
                for node_data in nodes.values():
                    if isinstance(node_data, dict) and node_data.get('status') == status:
                        status_nodes.append(node_data.get('position', [0, 0, 0]))
                
                if status_nodes:
                    positions = np.array(status_nodes)
                    fig.add_trace(go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode='markers',
                        marker=dict(size=3, color=color, opacity=0.6),
                        name=f'{status.title()} Nodes'
                    ))
            
            # Add mycelial seeds (smaller for overview) - HARD FAIL if missing  
            seeds = brain_data.get('mycelial_seeds', {})
            if not seeds:
                raise RuntimeError("CRITICAL: Mycelial seeds data is required for brain visualization but not available. Cannot create fake visualization data.")
                
            for status, color in self.seed_colors.items():
                status_seeds = []
                for seed_data in seeds.values():
                    if isinstance(seed_data, dict) and seed_data.get('status') == status:
                        status_seeds.append(seed_data.get('position', [0, 0, 0]))
                
                if status_seeds:
                    positions = np.array(status_seeds)
                    fig.add_trace(go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode='markers',
                        marker=dict(size=4, color=color, opacity=0.8, symbol='diamond'),
                        name=f'{status.title()} Seeds'
                    ))
            
            # Add mirror grid representation
            mirror_enabled = brain_data.get('mirror_grid_enabled', False)
            if mirror_enabled or brain_data.get('mirror_grid_matrix') is not None:
                # Add some mirror grid points
                n_mirror = 30
                mirror_x = np.random.uniform(0, x_dim, n_mirror)
                mirror_y = np.random.uniform(0, y_dim, n_mirror)
                mirror_z = np.random.uniform(0, z_dim, n_mirror)
                
                fig.add_trace(go.Scatter3d(
                    x=mirror_x,
                    y=mirror_y,
                    z=mirror_z,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='#FF69B4',
                        opacity=0.5,
                        symbol='square'
                    ),
                    name='Mirror Grid'
                ))
            
            fig.update_layout(
                title=f"Complete Brain System Overview<br>Grid: {x_dim}×{y_dim}×{z_dim}",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode='cube',
                    camera=dict(eye=dict(x=2, y=2, z=1.5))
                ),
                width=1200,
                height=900
            )
            
            logger.info("Combined brain overview created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create combined brain overview: {e}")
            raise RuntimeError(f"CRITICAL: Combined brain overview creation failed: {e}") from e

    def _add_example_nodes_small(self, fig: go.Figure, grid_dims: Tuple[int, int, int]):
        """Add small example nodes for overview"""
        x_dim, y_dim, z_dim = grid_dims
        n_nodes = 100
        x_nodes = np.random.uniform(0, x_dim, n_nodes)
        y_nodes = np.random.uniform(0, y_dim, n_nodes)
        z_nodes = np.random.uniform(0, z_dim, n_nodes)
        
        fig.add_trace(go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers',
            marker=dict(size=2, color='green', opacity=0.4),
            name='Neural Nodes'
        ))

    def _add_example_seeds_small(self, fig: go.Figure, grid_dims: Tuple[int, int, int]):
        """Add small example seeds for overview"""
        x_dim, y_dim, z_dim = grid_dims
        n_seeds = 30
        x_seeds = np.random.uniform(0, x_dim, n_seeds)
        y_seeds = np.random.uniform(0, y_dim, n_seeds)
        z_seeds = np.random.uniform(0, z_dim, n_seeds)
        
        fig.add_trace(go.Scatter3d(
            x=x_seeds, y=y_seeds, z=z_seeds,
            mode='markers',
            marker=dict(size=3, color='gold', opacity=0.7, symbol='diamond'),
            name='Mycelial Seeds'
        ))

    def visualize_complete_brain_system(self, brain_structure, save_plots: bool = True, show_plots: bool = True) -> Dict[str, Any]:
        """Create complete brain system visualization"""
        
        logger.info(f"Starting complete brain system visualization")
        
        results = {
            'structure': {},
            'nodes': {},
            'mycelial': {},
            'mirror_grid': {},
            'overview': {},
            'success': False
        }
        
        try:
            # Extract brain data with HARD FAIL for missing critical data
            brain_data = self._extract_brain_data(brain_structure)
            
            # 1. Create brain structure visualization
            logger.info("Creating brain structure visualization...")
            try:
                structure_fig = self.create_3d_brain_structure(brain_data)
                
                if show_plots:
                    structure_fig.show()
                    results['structure']['displayed'] = True
                
                if save_plots:
                    structure_path = os.path.join(self.output_dir, "brain_structure_3d.html")
                    structure_fig.write_html(structure_path)
                    results['structure']['path'] = structure_path
                
                results['structure']['success'] = True
                logger.info("Brain structure visualization created")
                
            except Exception as structure_error:
                logger.error(f"Brain structure visualization failed: {structure_error}")
                raise RuntimeError("CRITICAL: Brain structure visualization failed") from structure_error
            
            # 2. Create nodes visualization
            logger.info("Creating nodes visualization...")
            try:
                nodes_fig = self.create_nodes_visualization(brain_data)
                
                if show_plots:
                    nodes_fig.show()
                    results['nodes']['displayed'] = True
                
                if save_plots:
                    nodes_path = os.path.join(self.output_dir, "neural_nodes_3d.html")
                    nodes_fig.write_html(nodes_path)
                    results['nodes']['path'] = nodes_path
                
                results['nodes']['success'] = True
                logger.info("Nodes visualization created")
                
            except Exception as nodes_error:
                logger.error(f"Nodes visualization failed: {nodes_error}")
                raise RuntimeError("CRITICAL: Nodes visualization failed") from nodes_error
            
            # 3. Create mycelial network visualization
            logger.info("Creating mycelial network visualization...")
            try:
                mycelial_fig = self.create_mycelial_network_visualization(brain_data)
                
                if show_plots:
                    mycelial_fig.show()
                    results['mycelial']['displayed'] = True
                
                if save_plots:
                    mycelial_path = os.path.join(self.output_dir, "mycelial_network_3d.html")
                    mycelial_fig.write_html(mycelial_path)
                    results['mycelial']['path'] = mycelial_path
                
                results['mycelial']['success'] = True
                logger.info("Mycelial network visualization created")
                
            except Exception as mycelial_error:
                logger.error(f"Mycelial network visualization failed: {mycelial_error}")
                raise RuntimeError("CRITICAL: Mycelial network visualization failed") from mycelial_error
            
            # 4. Create mirror grid visualization
            logger.info("Creating mirror grid visualization...")
            try:
                mirror_fig = self.create_mirror_grid_visualization(brain_data)
                
                if show_plots:
                    mirror_fig.show()
                    results['mirror_grid']['displayed'] = True
                
                if save_plots:
                    mirror_path = os.path.join(self.output_dir, "mirror_grid_3d.html")
                    mirror_fig.write_html(mirror_path)
                    results['mirror_grid']['path'] = mirror_path
                
                results['mirror_grid']['success'] = True
                logger.info("Mirror grid visualization created")
                
            except Exception as mirror_error:
                logger.error(f"Mirror grid visualization failed: {mirror_error}")
                raise RuntimeError("CRITICAL: Mirror grid visualization failed") from mirror_error
            
            # 5. Create combined overview
            logger.info("Creating combined brain overview...")
            try:
                overview_fig = self.create_combined_brain_overview(brain_data)
                
                if show_plots:
                    overview_fig.show()
                    results['overview']['displayed'] = True
                
                if save_plots:
                    overview_path = os.path.join(self.output_dir, "brain_system_overview.html")
                    overview_fig.write_html(overview_path)
                    results['overview']['path'] = overview_path
                
                results['overview']['success'] = True
                logger.info("Combined brain overview created")
                
            except Exception as overview_error:
                logger.error(f"Combined overview creation failed: {overview_error}")
                raise RuntimeError("CRITICAL: Combined overview creation failed") from overview_error
            
            # 6. Save summary data
            if save_plots:
                summary = {
                    'brain_id': brain_data['brain_id'],
                    'visualization_date': datetime.now().isoformat(),
                    'grid_dimensions': brain_data['grid_dimensions'],
                    'nodes_count': len(brain_data.get('nodes', {})),
                    'seeds_count': len(brain_data.get('mycelial_seeds', {})),
                    'regions_count': len(brain_data.get('regions', {})),
                    'mirror_grid_enabled': brain_data.get('mirror_grid_enabled', False),
                    'creation_time': brain_data.get('creation_time')
                }
                
                summary_path = os.path.join(self.output_dir, "brain_visualization_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                results['summary_path'] = summary_path
            
            results['success'] = True
            logger.info(f"Complete brain system visualization successful")
            
        except Exception as e:
            logger.error(f"Complete brain system visualization failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            raise RuntimeError(f"CRITICAL: Complete brain visualization failed: {e}") from e
        
        return results

    def _create_brain_outline(self, x_dim: int, y_dim: int, z_dim: int) -> Dict[str, List]:
        """Create a simplified 3D brain outline mesh"""
        try:
            # Create a simplified brain-like ellipsoid shape
            # Brain is roughly ellipsoidal with width > depth > height proportions
            
            # Scale factors for brain proportions
            x_scale = x_dim * 0.8  # Width (left-right)
            y_scale = y_dim * 0.6  # Height (top-bottom) 
            z_scale = z_dim * 0.7  # Depth (front-back)
            
            # Center the brain
            cx, cy, cz = x_dim // 2, y_dim // 2, z_dim // 2
            
            # Create sphere coordinates and scale to brain shape
            phi = np.linspace(0, 2*np.pi, 20)
            theta = np.linspace(0, np.pi, 15)
            phi_grid, theta_grid = np.meshgrid(phi, theta)
            
            # Generate ellipsoid coordinates
            x = cx + (x_scale/2) * np.sin(theta_grid) * np.cos(phi_grid)
            y = cy + (y_scale/2) * np.cos(theta_grid) 
            z = cz + (z_scale/2) * np.sin(theta_grid) * np.sin(phi_grid)
            
            # Flatten arrays for mesh
            x_flat = x.flatten()
            y_flat = y.flatten()
            z_flat = z.flatten()
            
            # Create triangular mesh indices for the ellipsoid surface
            n_phi, n_theta = len(phi), len(theta)
            triangles_i, triangles_j, triangles_k = [], [], []
            
            for i in range(n_theta - 1):
                for j in range(n_phi - 1):
                    # Current vertex indices
                    v0 = i * n_phi + j
                    v1 = i * n_phi + (j + 1) % n_phi  
                    v2 = (i + 1) * n_phi + j
                    v3 = (i + 1) * n_phi + (j + 1) % n_phi
                    
                    # Two triangles per quad
                    triangles_i.extend([v0, v1])
                    triangles_j.extend([v1, v2]) 
                    triangles_k.extend([v2, v3])
            
            return {
                'x': x_flat.tolist(),
                'y': y_flat.tolist(), 
                'z': z_flat.tolist(),
                'i': triangles_i,
                'j': triangles_j,
                'k': triangles_k
            }
            
        except Exception as e:
            logger.error(f"Failed to create brain outline: {e}")
            # Return a simple box as fallback
            return {
                'x': [0, x_dim, x_dim, 0, 0, x_dim, x_dim, 0],
                'y': [0, 0, y_dim, y_dim, 0, 0, y_dim, y_dim], 
                'z': [0, 0, 0, 0, z_dim, z_dim, z_dim, z_dim],
                'i': [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                'j': [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                'k': [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
            }


def visualize_brain_from_file(brain_file_path: str, output_dir: str = "output/brain_visuals") -> Dict[str, Any]:
    """
    Load brain from file and create complete visualization
    
    Args:
        brain_file_path: Path to saved brain structure file
        output_dir: Directory for output visualizations
        
    Returns:
        Dictionary with visualization results
    """
    
    logger.info(f"Loading brain from file: {brain_file_path}")
    
    try:
        # Import AnatomicalBrain class
        from stage_1.brain_formation.brain_structure import AnatomicalBrain
        
        # Load brain from file (assuming it has a load method)
        # Note: Adjust this based on actual brain loading method
        brain_structure = AnatomicalBrain.load_from_file(brain_file_path)
        
        if not brain_structure:
            raise RuntimeError(f"CRITICAL: Failed to load brain from: {brain_file_path}")
        
        logger.info(f"Brain loaded successfully: {getattr(brain_structure, 'brain_id', 'Unknown')}")
        
        # Create visualizer and generate complete system
        visualizer = BrainVisualizer(output_dir=output_dir)
        results = visualizer.visualize_complete_brain_system(brain_structure, save_plots=True, show_plots=True)
        
        return results
        
    except Exception as e:
        logger.error(f"Brain visualization from file failed: {e}")
        raise RuntimeError(f"CRITICAL: Brain file visualization failed: {e}") from e


def visualize_brain_from_file(brain_file_path: str, output_dir: str = "output/brain_visuals") -> Dict[str, Any]:
    """
    Load brain from file and create complete visualization
    
    Args:
        brain_file_path: Path to brain JSON file
        output_dir: Directory for visualization outputs
        
    Returns:
        Dictionary with visualization results
    """
    
    try:
        # Load brain structure from file
        with open(brain_file_path, 'r') as f:
            brain_structure = json.load(f)
            
        logger.info(f"Brain loaded successfully: {getattr(brain_structure, 'brain_id', 'Unknown')}")
        
        # Create visualizer and generate complete system
        visualizer = BrainVisualizer(output_dir=output_dir)
        results = visualizer.visualize_complete_brain_system(brain_structure, save_plots=True, show_plots=True)
        
        return results
        
    except Exception as e:
        logger.error(f"Brain visualization from file failed: {e}")
        raise RuntimeError(f"CRITICAL: Brain file visualization failed: {e}") from e


def example_brain_usage():
    """Example of how to use the brain visualizer"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example file path (adjust as needed)
    brain_file_path = "output/brains/completed_brain.json"
    
    try:
        # Visualize brain from file
        results = visualize_brain_from_file(brain_file_path)
        
        if results['success']:
            print(f"✅ Brain visualization completed successfully!")
            print(f"📁 Output directory: output/brain_visuals/")
        else:
            print(f"❌ Brain visualization failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")