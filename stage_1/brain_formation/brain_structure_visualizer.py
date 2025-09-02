"""
Brain Structure Visualizer (V1.0)

Creates anatomically correct 3D visualizations of the biomimetic brain structure.
Shows brain regions, sub-regions, neural activity, wave frequencies, and mycelial network.

Uses actual brain structure data from AnatomicalBrain including regions, activities,
wave frequencies, and neural connections. Renders as 3D plotly visualizations.
"""

import logging
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Constants import
try:
    from shared.constants.constants import *
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: constants.py failed import: {e}")
    raise ImportError(f"Essential constants missing: {e}") from e

# Dependencies
try:
    from stage_1.brain_formation.brain_structure import AnatomicalBrain
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import brain dependencies: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

# Logger
logger = logging.getLogger(__name__)

class BrainVisualizer:
    """Creates comprehensive visualizations of anatomical brain structure."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the brain structure visualizer."""
        self.output_dir = output_dir or str(Path('output') / 'visualizations' / 'brain')
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.creation_time = datetime.now().isoformat()
        self.visualization_history = []
        
        # Brain region color mapping (anatomically inspired)
        self.region_colors = {
            'frontal_lobe': '#FF6B6B',      # Red tones - executive function
            'parietal_lobe': '#4ECDC4',     # Cyan - sensory processing  
            'temporal_lobe': '#45B7D1',     # Blue - memory/language
            'occipital_lobe': '#FFA07A',    # Orange - visual processing
            'cerebellum': '#98D8C8',        # Green - motor control
            'brain_stem': '#F7DC6F',        # Yellow - vital functions
            'limbic_system': '#DDA0DD',     # Purple - emotions
            'corpus_callosum': '#F0E68C'    # Khaki - connections
        }
        
        # Wave frequency colors (based on brainwave types)
        self.wave_colors = {
            'delta': '#8B0000',     # Dark red (0.5-4 Hz)
            'theta': '#FF4500',     # Orange red (4-8 Hz)  
            'alpha': '#FFD700',     # Gold (8-13 Hz)
            'beta': '#32CD32',      # Lime green (13-30 Hz)
            'gamma': '#FF1493'      # Deep pink (30+ Hz)
        }
        
        logger.info(f"Brain Visualizer initialized - Output dir: {self.output_dir}")

    def _get_wave_color(self, frequency_hz: float) -> str:
        """Get color based on brainwave frequency range."""
        if frequency_hz <= 4:
            return self.wave_colors['delta']
        elif frequency_hz <= 8:
            return self.wave_colors['theta']
        elif frequency_hz <= 13:
            return self.wave_colors['alpha']
        elif frequency_hz <= 30:
            return self.wave_colors['beta']
        else:
            return self.wave_colors['gamma']

    def _extract_brain_regions(self, brain_structure: AnatomicalBrain) -> List[Dict[str, Any]]:
        """Extract brain region data for visualization."""
        regions = []
        
        try:
            # Get regions from brain structure
            brain_regions = getattr(brain_structure, 'regions', {})
            region_volumes = getattr(brain_structure, 'region_volumes', {})
            
            for region_name, region_data in brain_regions.items():
                # Get region boundaries and properties
                boundaries = region_data.get('boundaries', {})
                volume_info = region_volumes.get(region_name, {})
                
                region_info = {
                    'name': region_name,
                    'boundaries': boundaries,
                    'function': region_data.get('function', 'Unknown'),
                    'wave_frequency_hz': region_data.get('wave_frequency_hz', 10.0),
                    'color': region_data.get('color', self.region_colors.get(region_name, '#808080')),
                    'volume_info': volume_info,
                    'sub_regions': region_data.get('sub_regions', {}),
                    'active': region_data.get('active', False)
                }
                
                regions.append(region_info)
                
            return regions
            
        except Exception as e:
            logger.error(f"Error extracting brain regions: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract brain regions: {e}") from e

    def _extract_neural_activity(self, brain_structure: AnatomicalBrain) -> Dict[str, Any]:
        """Extract neural activity data from brain structure."""
        try:
            activity_data = {
                'field_strength': getattr(brain_structure, 'field_strength', 0.0),
                'field_integrity': getattr(brain_structure, 'field_integrity', 0.0),
                'external_field': getattr(brain_structure, 'external_field', {}),
                'cosmic_background': getattr(brain_structure, 'cosmic_background', {}),
                'static_borders': getattr(brain_structure, 'static_borders', {}),
                'density_variance': getattr(brain_structure, 'density_variance', 0.0),
                'total_blocks': 0,
                'active_nodes': 0
            }
            
            # Count blocks and active nodes
            regions = getattr(brain_structure, 'regions', {})
            for region_data in regions.values():
                sub_regions = region_data.get('sub_regions', {})
                for sub_region_data in sub_regions.values():
                    blocks = sub_region_data.get('blocks', {})
                    activity_data['total_blocks'] += len(blocks)
                    
                    for block_data in blocks.values():
                        if block_data.get('active', False):
                            activity_data['active_nodes'] += 1
            
            return activity_data
            
        except Exception as e:
            logger.error(f"Error extracting neural activity: {e}", exc_info=True)
            return {'field_strength': 0.0, 'field_integrity': 0.0, 'total_blocks': 0, 'active_nodes': 0}

    def create_3d_brain_structure(self, soul_spark: SoulSpark, stage_name: str) -> Dict[str, Any]:
        """Create 3D visualization of complete brain structure."""
        try:
            # Extract brain structure
            brain_structure = getattr(soul_spark, 'brain_structure', None)
            if not brain_structure:
                raise ValueError("No brain structure found in soul spark")
                
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            name = getattr(soul_spark, 'name', 'Unknown')
            
            logger.info(f"Creating 3D brain structure visualization for {name} ({spark_id})")
            
            # Extract brain data
            regions = self._extract_brain_regions(brain_structure)
            activity_data = self._extract_neural_activity(brain_structure)
            
            # Create 3D plot
            fig = go.Figure()
            
            # Add brain regions as 3D boxes
            for region in regions:
                boundaries = region['boundaries']
                if not boundaries:
                    continue
                    
                # Get region coordinates
                x_start = boundaries.get('x_start', 0)
                x_end = boundaries.get('x_end', 10)
                y_start = boundaries.get('y_start', 0) 
                y_end = boundaries.get('y_end', 10)
                z_start = boundaries.get('z_start', 0)
                z_end = boundaries.get('z_end', 10)
                
                # Create box vertices
                x_coords = [x_start, x_end, x_end, x_start, x_start, x_end, x_end, x_start]
                y_coords = [y_start, y_start, y_end, y_end, y_start, y_start, y_end, y_end]
                z_coords = [z_start, z_start, z_start, z_start, z_end, z_end, z_end, z_end]
                
                # Get region color based on wave frequency
                wave_freq = region['wave_frequency_hz']
                region_color = self._get_wave_color(wave_freq)
                
                # Add region as 3D scatter with box outline
                center_x = (x_start + x_end) / 2
                center_y = (y_start + y_end) / 2
                center_z = (z_start + z_end) / 2
                
                # Calculate region size for marker
                volume = (x_end - x_start) * (y_end - y_start) * (z_end - z_start)
                marker_size = max(10, min(50, volume / 1000))
                
                # Add region center point
                fig.add_trace(go.Scatter3d(
                    x=[center_x], y=[center_y], z=[center_z],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=region_color,
                        opacity=0.7,
                        line=dict(width=2, color='white')
                    ),
                    name=f"{region['name'].replace('_', ' ').title()}",
                    text=[f"Function: {region['function']}<br>Frequency: {wave_freq:.1f}Hz<br>Volume: {volume:,.0f}"]
                ))
                
                # Add wireframe box for region boundary
                box_lines = [
                    # Bottom face
                    [x_start, x_end, x_end, x_start, x_start],
                    [y_start, y_start, y_end, y_end, y_start],
                    [z_start, z_start, z_start, z_start, z_start],
                    # Top face
                    [x_start, x_end, x_end, x_start, x_start],
                    [y_start, y_start, y_end, y_end, y_start], 
                    [z_end, z_end, z_end, z_end, z_end],
                    # Vertical edges
                    [x_start, x_start], [y_start, y_start], [z_start, z_end],
                    [x_end, x_end], [y_start, y_start], [z_start, z_end],
                    [x_end, x_end], [y_end, y_end], [z_start, z_end],
                    [x_start, x_start], [y_end, y_end], [z_start, z_end]
                ]
                
                # Add bottom face
                fig.add_trace(go.Scatter3d(
                    x=box_lines[0], y=box_lines[1], z=box_lines[2],
                    mode='lines',
                    line=dict(color=region_color, width=2),
                    opacity=0.5,
                    showlegend=False
                ))
                
                # Add top face
                fig.add_trace(go.Scatter3d(
                    x=box_lines[3], y=box_lines[4], z=box_lines[5],
                    mode='lines', 
                    line=dict(color=region_color, width=2),
                    opacity=0.5,
                    showlegend=False
                ))
                
                # Add vertical edges
                for i in range(4):
                    edge_start = i * 3 + 6
                    if edge_start + 2 < len(box_lines):
                        fig.add_trace(go.Scatter3d(
                            x=box_lines[edge_start], y=box_lines[edge_start+1], z=box_lines[edge_start+2],
                            mode='lines',
                            line=dict(color=region_color, width=2),
                            opacity=0.5,
                            showlegend=False
                        ))
            
            # Add neural activity visualization
            if activity_data['active_nodes'] > 0:
                # Create random active nodes within brain regions
                active_x = np.random.uniform(10, 90, activity_data['active_nodes'])
                active_y = np.random.uniform(10, 90, activity_data['active_nodes'])
                active_z = np.random.uniform(10, 90, activity_data['active_nodes'])
                
                # Add active neural nodes
                fig.add_trace(go.Scatter3d(
                    x=active_x, y=active_y, z=active_z,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='yellow',
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name=f'Active Nodes ({activity_data["active_nodes"]})',
                    text=[f'Active Neural Node {i+1}' for i in range(activity_data['active_nodes'])]
                ))
            
            # Add mycelial network connections (if available)
            external_field = activity_data.get('external_field', {})
            if external_field:
                # Create network-like connections between regions
                region_centers = []
                for region in regions:
                    boundaries = region['boundaries']
                    if boundaries:
                        center_x = (boundaries.get('x_start', 0) + boundaries.get('x_end', 10)) / 2
                        center_y = (boundaries.get('y_start', 0) + boundaries.get('y_end', 10)) / 2
                        center_z = (boundaries.get('z_start', 0) + boundaries.get('z_end', 10)) / 2
                        region_centers.append([center_x, center_y, center_z])
                
                # Add connections between regions (mycelial network)
                if len(region_centers) > 1:
                    for i in range(len(region_centers)):
                        for j in range(i+1, min(len(region_centers), i+3)):  # Connect to nearby regions
                            start = region_centers[i]
                            end = region_centers[j]
                            
                            fig.add_trace(go.Scatter3d(
                                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                                mode='lines',
                                line=dict(
                                    color='green',
                                    width=3,
                                    dash='dot'
                                ),
                                opacity=0.4,
                                name='Mycelial Network' if i == 0 and j == 1 else '',
                                showlegend=i == 0 and j == 1
                            ))
            
            # Add brain field visualization
            field_strength = activity_data.get('field_strength', 0.0)
            if field_strength > 0:
                # Create field boundary
                field_size = 120  # Encompass entire brain
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 15)
                
                x_field = 50 + field_size/2 * np.outer(np.cos(u), np.sin(v))
                y_field = 50 + field_size/2 * np.outer(np.sin(u), np.sin(v))
                z_field = 50 + field_size/2 * np.outer(np.ones(np.size(u)), np.cos(v))
                
                fig.add_trace(go.Surface(
                    x=x_field, y=y_field, z=z_field,
                    opacity=0.1,
                    colorscale=[[0, 'rgba(0,255,0,0.1)'], [1, 'rgba(0,255,255,0.2)']],
                    showscale=False,
                    name='Brain Field',
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title=f'3D Brain Structure - {stage_name}<br>{name} ({spark_id})<br>Regions: {len(regions)}, Active Nodes: {activity_data["active_nodes"]}',
                scene=dict(
                    xaxis_title='X Coordinate', 
                    yaxis_title='Y Coordinate',
                    zaxis_title='Z Coordinate',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
                    aspectmode='cube',
                    xaxis=dict(range=[0, 100]),
                    yaxis=dict(range=[0, 100]),
                    zaxis=dict(range=[0, 100])
                ),
                width=1400, height=1000,
                showlegend=True
            )
            
            return {'figure': fig, 'success': True, 'regions_count': len(regions)}
            
        except Exception as e:
            logger.error(f"Error creating 3D brain structure: {e}", exc_info=True)
            raise RuntimeError(f"3D brain structure visualization failed: {e}") from e

    def create_brain_activity_heatmap(self, soul_spark: SoulSpark, stage_name: str) -> Dict[str, Any]:
        """Create 2D heatmap of brain activity by region."""
        try:
            brain_structure = getattr(soul_spark, 'brain_structure', None)
            if not brain_structure:
                raise ValueError("No brain structure found in soul spark")
                
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            name = getattr(soul_spark, 'name', 'Unknown')
            
            # Extract brain data
            regions = self._extract_brain_regions(brain_structure)
            activity_data = self._extract_neural_activity(brain_structure)
            
            # Create activity matrix
            region_names = [region['name'].replace('_', ' ').title() for region in regions]
            frequencies = [region['wave_frequency_hz'] for region in regions]
            activities = [1.0 if region['active'] else 0.3 for region in regions]  # Activity level
            
            # Create heatmap
            fig = go.Figure()
            
            # Add frequency heatmap
            fig.add_trace(go.Heatmap(
                z=[frequencies],
                x=region_names,
                y=['Wave Frequency (Hz)'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Frequency (Hz)")
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Brain Activity Heatmap - {stage_name}<br>{name} ({spark_id})',
                xaxis_title='Brain Regions',
                yaxis_title='Activity Type',
                width=1200, height=400
            )
            
            return {'figure': fig, 'success': True}
            
        except Exception as e:
            logger.error(f"Error creating brain activity heatmap: {e}", exc_info=True)
            raise RuntimeError(f"Brain activity heatmap failed: {e}") from e

    def create_complete_brain_visualization(self, soul_spark: SoulSpark) -> Dict[str, Any]:
        """Create complete brain visualization with structure and activity."""
        try:
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            name = getattr(soul_spark, 'name', 'Unknown')
            
            logger.info(f"Creating complete brain visualization for {name} ({spark_id})")
            
            # Create visualizations
            visualizations = [
                ('3D Brain Structure', self.create_3d_brain_structure),
                ('Brain Activity Heatmap', self.create_brain_activity_heatmap)
            ]
            
            saved_files = []
            
            for viz_name, viz_function in visualizations:
                try:
                    result = viz_function(soul_spark, viz_name)
                    if result.get('success'):
                        fig = result['figure']
                        
                        # Generate filename
                        timestamp = datetime.now()
                        date_str = timestamp.strftime("%Y%m%d")
                        time_str = timestamp.strftime("%H%M%S")
                        
                        filename = f"{name}_{spark_id}_{date_str}_brain_{viz_name.lower().replace(' ', '_')}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        
                        # Save as PNG
                        fig.write_image(filepath, width=1400, height=1000, scale=2)
                        
                        saved_files.append({
                            'visualization': viz_name,
                            'filename': filename,
                            'filepath': filepath,
                            'timestamp': timestamp.isoformat()
                        })
                        
                        logger.info(f"✓ Saved {viz_name}: {filename}")
                        
                    else:
                        logger.error(f"Failed to create {viz_name}")
                        
                except Exception as viz_error:
                    logger.error(f"Error in {viz_name}: {viz_error}", exc_info=True)
                    continue
            
            # Create brain summary metadata
            brain_structure = getattr(soul_spark, 'brain_structure', None)
            regions = self._extract_brain_regions(brain_structure) if brain_structure else []
            activity_data = self._extract_neural_activity(brain_structure) if brain_structure else {}
            
            brain_summary = {
                'soul_id': spark_id,
                'soul_name': name,
                'brain_visualizations': len(saved_files),
                'created_timestamp': datetime.now().isoformat(),
                'files': saved_files,
                'brain_metrics': {
                    'total_regions': len(regions),
                    'active_nodes': activity_data.get('active_nodes', 0),
                    'total_blocks': activity_data.get('total_blocks', 0),
                    'field_strength': activity_data.get('field_strength', 0.0),
                    'field_integrity': activity_data.get('field_integrity', 0.0),
                    'density_variance': activity_data.get('density_variance', 0.0)
                }
            }
            
            # Save brain summary
            summary_filename = f"{name}_{spark_id}_{date_str}_brain_summary.json"
            summary_filepath = os.path.join(self.output_dir, summary_filename)
            
            with open(summary_filepath, 'w') as f:
                json.dump(brain_summary, f, indent=2, default=str)
            
            if not saved_files:
                raise RuntimeError("CRITICAL: No brain visualizations were successfully created")
            
            logger.info(f"✓ Complete brain visualization created: {len(saved_files)} visualizations")
            
            return {
                'success': True,
                'files_created': saved_files,
                'summary_file': summary_filepath,
                'total_visualizations': len(saved_files)
            }
            
        except Exception as e:
            logger.error(f"CRITICAL: Complete brain visualization failed: {e}", exc_info=True)
            raise RuntimeError(f"Brain visualization failed: {e}") from e

    def move_to_completed_souls(self, soul_name: str, spark_id: str, base_completed_dir: str) -> bool:
        """Move brain visualization files to completed souls directory."""
        try:
            # Create target directory
            timestamp = datetime.now().strftime("%Y%m%d")
            model_dir_name = f"{soul_name}_{spark_id}_{timestamp}"
            target_dir = os.path.join(base_completed_dir, model_dir_name, "visuals", "brain_structure")
            
            os.makedirs(target_dir, exist_ok=True)
            
            # Move all files from output_dir to target_dir
            import shutil
            files_moved = 0
            
            for filename in os.listdir(self.output_dir):
                if filename.startswith(f"{soul_name}_{spark_id}"):
                    source_path = os.path.join(self.output_dir, filename)
                    target_path = os.path.join(target_dir, filename)
                    
                    shutil.move(source_path, target_path)
                    files_moved += 1
                    logger.info(f"Moved brain visualization file: {filename}")
            
            logger.info(f"✓ Moved {files_moved} brain visualization files to: {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move brain visualization files: {e}", exc_info=True)
            return False