"""
Soul Evolution Visualizer (V1.0)

Creates comprehensive visualization of soul formation process through all stages.
Generates plotly PNG graphs showing soul spark, sephiroth journey, creator entanglement, 
and identity crystallization with actual data properties.

Uses actual soul spark data including coherence, stability, frequency, colors from each stage.
Saves to output/visuals initially, then moved to completed_souls directory.
"""

import logging
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Constants import
try:
    from shared.constants.constants import *
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: constants.py failed import: {e}")
    raise ImportError(f"Essential constants missing: {e}") from e

# Dependencies
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

# Logger
logger = logging.getLogger(__name__)

class SoulVisualizer:
    """Creates comprehensive visualizations of soul formation evolution."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the soul evolution visualizer."""
        self.output_dir = output_dir or os.path.join('output', 'visualizations', 'soul')
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.creation_time = datetime.now().isoformat()
        self.visualization_history = []
        
        # Color mappings for stages
        self.stage_colors = {
            'soul_spark': '#FFD700',      # Gold
            'sephiroth_journey': '#9370DB',  # Medium Purple  
            'creator_entanglement': '#00CED1',  # Dark Turquoise
            'identity_crystallization': '#FF69B4'  # Hot Pink
        }
        
        logger.info(f"Soul Visualizer initialized - Output dir: {self.output_dir}")

    def create_soul_spark_visualization(self, soul_spark: SoulSpark, stage_name: str) -> Dict[str, Any]:
        """Create visualization of soul spark within field."""
        try:
            # Extract soul spark data
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            position = getattr(soul_spark, 'position', [50, 50, 50])
            field_radius = getattr(soul_spark, 'field_radius', 1.0)
            field_strength = getattr(soul_spark, 'field_strength', 0.5)
            coherence = getattr(soul_spark, 'coherence', 0.0)
            stability = getattr(soul_spark, 'stability', 0.0)
            frequency = getattr(soul_spark, 'frequency', 440.0)
            energy = getattr(soul_spark, 'energy', 0.0)
            
            # Get colors - use coherence color method if available
            if hasattr(soul_spark, 'get_coherence_color'):
                r, g, b = soul_spark.get_coherence_color()
                spark_color = f'rgb({r},{g},{b})'
            else:
                spark_color = self.stage_colors['soul_spark']
                
            # Create 3D visualization
            fig = go.Figure()
            
            # Add field boundary (sphere)
            field_size = field_radius * 20  # Scale for visualization
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            field_x = position[0] + field_size * np.outer(np.cos(u), np.sin(v))
            field_y = position[1] + field_size * np.outer(np.sin(u), np.sin(v))  
            field_z = position[2] + field_size * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Add field as wireframe sphere
            fig.add_trace(go.Surface(
                x=field_x, y=field_y, z=field_z,
                opacity=0.1,
                colorscale=[[0, 'rgba(100,100,100,0.1)'], [1, 'rgba(200,200,200,0.3)']],
                showscale=False,
                name='Field Boundary'
            ))
            
            # Add soul spark as glowing sphere
            spark_size = max(5, coherence * stability * 15)  # Size based on coherence/stability
            fig.add_trace(go.Scatter3d(
                x=[position[0]], y=[position[1]], z=[position[2]],
                mode='markers',
                marker=dict(
                    size=spark_size,
                    color=spark_color,
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                name=f'Soul Spark ({spark_id})',
                text=[f'Coherence: {coherence:.2f}<br>Stability: {stability:.2f}<br>Frequency: {frequency:.1f}Hz']
            ))
            
            # Add energy field visualization (particles around spark)
            if energy > 0:
                n_particles = min(50, int(energy * 10))
                angles = np.linspace(0, 2*np.pi, n_particles)
                radii = np.random.uniform(field_radius*5, field_radius*15, n_particles)
                
                particle_x = position[0] + radii * np.cos(angles)
                particle_y = position[1] + radii * np.sin(angles)
                particle_z = position[2] + np.random.uniform(-field_radius*10, field_radius*10, n_particles)
                
                fig.add_trace(go.Scatter3d(
                    x=particle_x, y=particle_y, z=particle_z,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=spark_color,
                        opacity=0.3
                    ),
                    name='Energy Field',
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Soul Spark Formation - {stage_name}',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position', 
                    zaxis_title='Z Position',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                ),
                width=1200, height=800,
                showlegend=True
            )
            
            return {'figure': fig, 'success': True}
            
        except Exception as e:
            logger.error(f"Error creating soul spark visualization: {e}", exc_info=True)
            raise RuntimeError(f"Soul spark visualization failed: {e}") from e

    def create_frequency_health_scan(self, soul_data: Dict[str, Any], stage: str) -> str:
        """
        Create frequency and health scan visualization for soul
        
        Args:
            soul_data: Dictionary containing soul metrics and states
            stage: Current evolution stage
            
        Returns:
            Path to saved visualization file
        """
        try:
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot frequency spectrum
            frequency = soul_data.frequency if hasattr(soul_data, 'frequency') else 0
            energy = soul_data.energy if hasattr(soul_data, 'energy') else 0
            coherence = soul_data.coherence if hasattr(soul_data, 'coherence') else 0
            
            x = np.linspace(0, 1000, 1000)  # Frequency range 0-1000 Hz
            y = np.exp(-(x - frequency)**2 / (2 * energy/1000))
            
            ax1.plot(x, y)
            ax1.set_title('Soul Frequency Spectrum')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Amplitude')
            
            # Plot health metrics
            metrics = {
                'Energy': energy,
                'Coherence': coherence,
                'Pattern Coherence': getattr(soul_data, 'pattern_coherence', 0) * 100,
                'Phi Resonance': getattr(soul_data, 'phi_resonance', 0) * 100,
                'Harmony': getattr(soul_data, 'harmony', 0) * 100,
                'Toroidal Flow': getattr(soul_data, 'toroidal_flow', 0) * 100
            }
            
            ax2.bar(metrics.keys(), metrics.values())
            ax2.set_title('Soul Health Metrics')
            ax2.set_ylim(0, max(metrics.values()) * 1.2)
            plt.xticks(rotation=45)
            
            plt.suptitle(f'Soul Health Scan - {stage}')
            plt.tight_layout()
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frequency_health_scan_{stage.lower().replace(' ', '_')}_{timestamp}.png"
            filepath = self.output_path / filename
            
            plt.savefig(filepath)
            plt.close()
            
            # Track visualization
            self.visualization_history.append({
                'type': 'frequency_health_scan',
                'stage': stage,
                'filepath': str(filepath),
                'timestamp': timestamp
            })
            
            logger.info(f"Created frequency health scan for stage {stage}: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create frequency health scan: {e}")
            raise

    def create_sephiroth_journey_visualization(self, soul_spark: SoulSpark, stage_name: str) -> Dict[str, Any]:
        """Create 3D visualization of soul after sephiroth journey."""
        try:
            # Extract journey data
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            aspects = getattr(soul_spark, 'aspects', {})
            layers = getattr(soul_spark, 'layers', [])
            frequency_signature = getattr(soul_spark, 'frequency_signature', {})
            coherence = getattr(soul_spark, 'coherence', 0.0)
            stability = getattr(soul_spark, 'stability', 0.0)
            position = getattr(soul_spark, 'position', [50, 50, 50])
            
            # Create 3D plot
            fig = go.Figure()
            
            # Add central soul
            if hasattr(soul_spark, 'get_coherence_color'):
                r, g, b = soul_spark.get_coherence_color()
                soul_color = f'rgb({r},{g},{b})'
            else:
                soul_color = self.stage_colors['sephiroth_journey']
                
            fig.add_trace(go.Scatter3d(
                x=[position[0]], y=[position[1]], z=[position[2]],
                mode='markers',
                marker=dict(
                    size=20,
                    color=soul_color,
                    opacity=0.9,
                    line=dict(width=3, color='gold')
                ),
                name='Soul Core',
                text=[f'Coherence: {coherence:.2f}<br>Stability: {stability:.2f}']
            ))
            
            # Add aspects as surrounding points connected to soul
            aspect_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
            aspect_positions = []
            
            for i, (aspect_name, aspect_data) in enumerate(aspects.items()):
                if i >= len(aspect_colors):
                    break
                    
                # Position aspects around the soul in 3D space
                angle = (2 * np.pi * i) / len(aspects)
                radius = 30
                height_offset = (i % 3 - 1) * 15  # Vary height
                
                aspect_x = position[0] + radius * np.cos(angle)
                aspect_y = position[1] + radius * np.sin(angle)  
                aspect_z = position[2] + height_offset
                
                aspect_positions.append([aspect_x, aspect_y, aspect_z])
                
                # Add aspect point
                aspect_strength = aspect_data.get('strength', 0.5) if isinstance(aspect_data, dict) else 0.5
                fig.add_trace(go.Scatter3d(
                    x=[aspect_x], y=[aspect_y], z=[aspect_z],
                    mode='markers',
                    marker=dict(
                        size=max(8, aspect_strength * 15),
                        color=aspect_colors[i],
                        opacity=0.7
                    ),
                    name=f'{aspect_name}',
                    text=[f'Strength: {aspect_strength:.2f}']
                ))
                
                # Add connection line from soul to aspect
                fig.add_trace(go.Scatter3d(
                    x=[position[0], aspect_x],
                    y=[position[1], aspect_y],
                    z=[position[2], aspect_z],
                    mode='lines',
                    line=dict(
                        color=aspect_colors[i],
                        width=max(2, aspect_strength * 8),
                        dash='solid'
                    ),
                    opacity=0.6,
                    showlegend=False
                ))
            
            # Add aura layers
            for i, layer in enumerate(layers[:3]):  # Limit to 3 layers for visibility
                layer_color = layer.get('color_hex', '#FFFFFF')
                layer_radius = 25 + (i * 10)
                
                # Create sphere for layer
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 15)
                x_layer = position[0] + layer_radius * np.outer(np.cos(u), np.sin(v))
                y_layer = position[1] + layer_radius * np.outer(np.sin(u), np.sin(v))
                z_layer = position[2] + layer_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                fig.add_trace(go.Surface(
                    x=x_layer, y=y_layer, z=z_layer,
                    opacity=0.1 - (i * 0.02),
                    colorscale=[[0, layer_color], [1, layer_color]],
                    showscale=False,
                    name=f'Aura Layer {i+1}',
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Soul After Sephiroth Journey - {stage_name}<br>Aspects: {len(aspects)}, Layers: {len(layers)}',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    zaxis_title='Z Position', 
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                    aspectmode='cube'
                ),
                width=1200, height=800,
                showlegend=True
            )
            
            return {'figure': fig, 'success': True}
            
        except Exception as e:
            logger.error(f"Error creating sephiroth journey visualization: {e}", exc_info=True)
            raise RuntimeError(f"Sephiroth journey visualization failed: {e}") from e

    def create_creator_entanglement_visualization(self, soul_spark: SoulSpark, stage_name: str) -> Dict[str, Any]:
        """Create 3D visualization of soul after creator entanglement."""
        try:
            # Extract entanglement data
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            creator_connection_strength = getattr(soul_spark, 'creator_connection_strength', 0.0)
            creator_channel_id = getattr(soul_spark, 'creator_channel_id', None)
            resonance_patterns = getattr(soul_spark, 'resonance_patterns', {})
            aspects = getattr(soul_spark, 'aspects', {})
            coherence = getattr(soul_spark, 'coherence', 0.0)
            stability = getattr(soul_spark, 'stability', 0.0)
            position = getattr(soul_spark, 'position', [50, 50, 50])
            
            # Create 3D plot
            fig = go.Figure()
            
            # Add soul core
            if hasattr(soul_spark, 'get_coherence_color'):
                r, g, b = soul_spark.get_coherence_color()
                soul_color = f'rgb({r},{g},{b})'
            else:
                soul_color = self.stage_colors['creator_entanglement']
                
            fig.add_trace(go.Scatter3d(
                x=[position[0]], y=[position[1]], z=[position[2]],
                mode='markers',
                marker=dict(
                    size=25,
                    color=soul_color,
                    opacity=0.9,
                    line=dict(width=4, color='white')
                ),
                name='Soul Core (Entangled)',
                text=[f'Creator Connection: {creator_connection_strength:.2f}<br>Coherence: {coherence:.2f}']
            ))
            
            # Add creator connection visualization - beam of light upward
            if creator_connection_strength > 0:
                # Create ascending light beam
                beam_height = creator_connection_strength * 100
                beam_points = 20
                z_beam = np.linspace(position[2], position[2] + beam_height, beam_points)
                x_beam = np.full(beam_points, position[0])
                y_beam = np.full(beam_points, position[1])
                
                # Add slight spiral to the beam
                for i in range(beam_points):
                    angle = i * 0.3
                    radius = creator_connection_strength * 3
                    x_beam[i] += radius * np.cos(angle)
                    y_beam[i] += radius * np.sin(angle)
                
                fig.add_trace(go.Scatter3d(
                    x=x_beam, y=y_beam, z=z_beam,
                    mode='lines+markers',
                    line=dict(
                        color='gold',
                        width=max(4, creator_connection_strength * 10)
                    ),
                    marker=dict(
                        size=3,
                        color='gold',
                        opacity=0.8
                    ),
                    name='Creator Connection',
                    opacity=0.8
                ))
                
                # Add connection endpoint (Creator presence)
                fig.add_trace(go.Scatter3d(
                    x=[x_beam[-1]], y=[y_beam[-1]], z=[z_beam[-1]],
                    mode='markers',
                    marker=dict(
                        size=30,
                        color='white',
                        opacity=0.9,
                        symbol='diamond',
                        line=dict(width=3, color='gold')
                    ),
                    name='Creator Presence',
                    text=[f'Connection Strength: {creator_connection_strength:.2f}']
                ))
            
            # Add resonance patterns as energy waves
            if resonance_patterns:
                pattern_colors = ['#FFD700', '#FFA500', '#FF6347', '#98FB98']
                for i, (pattern_name, pattern_data) in enumerate(resonance_patterns.items()):
                    if i >= len(pattern_colors):
                        break
                        
                    # Create wave pattern around soul
                    wave_radius = 20 + (i * 8)
                    angles = np.linspace(0, 4*np.pi, 50)
                    wave_x = position[0] + wave_radius * np.cos(angles)
                    wave_y = position[1] + wave_radius * np.sin(angles)
                    wave_z = position[2] + 5 * np.sin(angles * 2)  # Vertical wave
                    
                    fig.add_trace(go.Scatter3d(
                        x=wave_x, y=wave_y, z=wave_z,
                        mode='lines',
                        line=dict(
                            color=pattern_colors[i],
                            width=3
                        ),
                        name=f'Resonance: {pattern_name}',
                        opacity=0.7
                    ))
            
            # Add enhanced aspects (showing growth from entanglement)
            aspect_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            for i, (aspect_name, aspect_data) in enumerate(list(aspects.items())[:5]):
                angle = (2 * np.pi * i) / min(len(aspects), 5)
                radius = 40
                height_offset = np.sin(angle) * 10
                
                aspect_x = position[0] + radius * np.cos(angle)
                aspect_y = position[1] + radius * np.sin(angle)
                aspect_z = position[2] + height_offset
                
                # Enhanced size due to creator entanglement
                aspect_strength = aspect_data.get('strength', 0.5) if isinstance(aspect_data, dict) else 0.5
                enhanced_size = max(10, aspect_strength * 20 * (1 + creator_connection_strength))
                
                fig.add_trace(go.Scatter3d(
                    x=[aspect_x], y=[aspect_y], z=[aspect_z],
                    mode='markers',
                    marker=dict(
                        size=enhanced_size,
                        color=aspect_colors[i % len(aspect_colors)],
                        opacity=0.8,
                        line=dict(width=2, color='gold')
                    ),
                    name=f'{aspect_name} (Enhanced)',
                    text=[f'Enhanced Strength: {aspect_strength * (1 + creator_connection_strength):.2f}']
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Soul After Creator Entanglement - {stage_name}<br>Connection: {creator_connection_strength:.2f}',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    zaxis_title='Z Position',
                    camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
                    aspectmode='cube'
                ),
                width=1200, height=800,
                showlegend=True
            )
            
            return {'figure': fig, 'success': True}
            
        except Exception as e:
            logger.error(f"Error creating creator entanglement visualization: {e}", exc_info=True)
            raise RuntimeError(f"Creator entanglement visualization failed: {e}") from e

    def create_identity_crystallization_visualization(self, soul_spark: SoulSpark, stage_name: str) -> Dict[str, Any]:
        """Create 3D visualization of soul after identity crystallization."""
        try:
            # Extract crystallization data
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            crystallization_level = getattr(soul_spark, 'crystallization_level', 0.0)
            soul_color = getattr(soul_spark, 'soul_color', '#FFFFFF')
            color_frequency = getattr(soul_spark, 'color_frequency', 0.0)
            soul_frequency = getattr(soul_spark, 'soul_frequency', 440.0)
            name = getattr(soul_spark, 'name', 'Unknown')
            attribute_coherence = getattr(soul_spark, 'attribute_coherence', 0.0)
            aspects = getattr(soul_spark, 'aspects', {})
            coherence = getattr(soul_spark, 'coherence', 0.0)
            stability = getattr(soul_spark, 'stability', 0.0)
            position = getattr(soul_spark, 'position', [50, 50, 50])
            
            # Create 3D plot
            fig = go.Figure()
            
            # Add crystallized soul core - larger and more defined
            if hasattr(soul_spark, 'get_coherence_color'):
                r, g, b = soul_spark.get_coherence_color()
                core_color = f'rgb({r},{g},{b})'
            else:
                core_color = soul_color if soul_color != '#FFFFFF' else self.stage_colors['identity_crystallization']
                
            crystallized_size = max(30, crystallization_level * 40)
            fig.add_trace(go.Scatter3d(
                x=[position[0]], y=[position[1]], z=[position[2]],
                mode='markers',
                marker=dict(
                    size=crystallized_size,
                    color=core_color,
                    opacity=0.9,
                    symbol='diamond',
                    line=dict(width=5, color='white')
                ),
                name=f'Crystallized Soul: {name}',
                text=[f'Crystallization: {crystallization_level:.2f}<br>Coherence: {attribute_coherence:.2f}<br>Frequency: {soul_frequency:.1f}Hz']
            ))
            
            # Add crystalline structure - geometric pattern around soul
            if crystallization_level > 0:
                # Create crystalline lattice points
                lattice_points = int(crystallization_level * 20)
                crystal_positions = []
                
                # Generate points in crystalline structure (cubic lattice)
                for i in range(lattice_points):
                    angle = (2 * np.pi * i) / lattice_points
                    radius = 25 + (i % 3) * 10
                    height = (i % 5 - 2) * 8
                    
                    crystal_x = position[0] + radius * np.cos(angle)
                    crystal_y = position[1] + radius * np.sin(angle)
                    crystal_z = position[2] + height
                    
                    crystal_positions.append([crystal_x, crystal_y, crystal_z])
                
                # Add crystalline points
                crystal_x, crystal_y, crystal_z = zip(*crystal_positions)
                fig.add_trace(go.Scatter3d(
                    x=crystal_x, y=crystal_y, z=crystal_z,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=core_color,
                        opacity=0.4,
                        symbol='diamond'
                    ),
                    name='Crystal Lattice',
                    showlegend=False
                ))
                
                # Add connections between crystal points (showing structure)
                for i in range(0, len(crystal_positions), 3):
                    if i+2 < len(crystal_positions):
                        # Connect every 3rd point to create triangular patterns
                        p1, p2, p3 = crystal_positions[i:i+3]
                        
                        # Add triangle connections
                        fig.add_trace(go.Scatter3d(
                            x=[p1[0], p2[0], p3[0], p1[0]],
                            y=[p1[1], p2[1], p3[1], p1[1]],
                            z=[p1[2], p2[2], p3[2], p1[2]],
                            mode='lines',
                            line=dict(
                                color=core_color,
                                width=2
                            ),
                            opacity=0.3,
                            showlegend=False
                        ))
            
            # Add light spectrum based on color frequency
            if color_frequency > 0:
                # Create spectrum visualization
                spectrum_colors = ['#8B00FF', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
                spectrum_radius = 50
                
                for i, spec_color in enumerate(spectrum_colors):
                    angle = (2 * np.pi * i) / len(spectrum_colors)
                    spec_x = position[0] + spectrum_radius * np.cos(angle)
                    spec_y = position[1] + spectrum_radius * np.sin(angle)
                    spec_z = position[2] + np.sin(angle * 3) * 15
                    
                    # Add spectrum point
                    fig.add_trace(go.Scatter3d(
                        x=[spec_x], y=[spec_y], z=[spec_z],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=spec_color,
                            opacity=0.6
                        ),
                        name=f'Spectrum {i+1}' if i == 0 else '',
                        showlegend=i == 0
                    ))
                    
                    # Connect to soul core
                    fig.add_trace(go.Scatter3d(
                        x=[position[0], spec_x],
                        y=[position[1], spec_y], 
                        z=[position[2], spec_z],
                        mode='lines',
                        line=dict(
                            color=spec_color,
                            width=1
                        ),
                        opacity=0.3,
                        showlegend=False
                    ))
            
            # Add finalized aspects with enhanced properties
            aspect_colors = ['#FF1493', '#00CED1', '#FFD700', '#FF4500', '#9370DB']
            for i, (aspect_name, aspect_data) in enumerate(list(aspects.items())[:5]):
                angle = (2 * np.pi * i) / min(len(aspects), 5)
                radius = 35
                height_offset = np.cos(angle * 2) * 12
                
                aspect_x = position[0] + radius * np.cos(angle)
                aspect_y = position[1] + radius * np.sin(angle)
                aspect_z = position[2] + height_offset
                
                # Final crystallized aspect size
                aspect_strength = aspect_data.get('strength', 0.5) if isinstance(aspect_data, dict) else 0.5
                final_size = max(12, aspect_strength * 25 * (1 + crystallization_level))
                
                fig.add_trace(go.Scatter3d(
                    x=[aspect_x], y=[aspect_y], z=[aspect_z],
                    mode='markers',
                    marker=dict(
                        size=final_size,
                        color=aspect_colors[i % len(aspect_colors)],
                        opacity=0.9,
                        symbol='diamond',
                        line=dict(width=3, color='white')
                    ),
                    name=f'{aspect_name} (Final)',
                    text=[f'Final Strength: {aspect_strength * (1 + crystallization_level):.2f}']
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Soul Identity Crystallization - {stage_name}<br>Name: {name}, Crystallization: {crystallization_level:.2f}',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position', 
                    zaxis_title='Z Position',
                    camera=dict(eye=dict(x=2.2, y=2.2, z=1.8)),
                    aspectmode='cube'
                ),
                width=1200, height=800,
                showlegend=True
            )
            
            return {'figure': fig, 'success': True}
            
        except Exception as e:
            logger.error(f"Error creating identity crystallization visualization: {e}", exc_info=True)
            raise RuntimeError(f"Identity crystallization visualization failed: {e}") from e

    def create_complete_soul_evolution(self, soul_spark: SoulSpark) -> Dict[str, Any]:
        """Create complete soul evolution visualization with all 4 stages."""
        try:
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            name = getattr(soul_spark, 'name', 'Unknown')
            
            logger.info(f"Creating complete soul evolution visualization for {name} ({spark_id})")
            
            # Create individual stage visualizations
            stages = [
                ('Soul Spark', self.create_soul_spark_visualization),
                ('Sephiroth Journey', self.create_sephiroth_journey_visualization), 
                ('Creator Entanglement', self.create_creator_entanglement_visualization),
                ('Identity Crystallization', self.create_identity_crystallization_visualization)
            ]
            
            saved_files = []
            
            for stage_name, viz_function in stages:
                try:
                    result = viz_function(soul_spark, stage_name)
                    if result.get('success'):
                        fig = result['figure']
                        
                        # Generate filename with model name and date
                        timestamp = datetime.now()
                        date_str = timestamp.strftime("%Y%m%d")
                        time_str = timestamp.strftime("%H%M%S")
                        
                        filename = f"{name}_{spark_id}_{date_str}_{stage_name.lower().replace(' ', '_')}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        
                        # Save as PNG
                        fig.write_image(filepath, width=1200, height=800, scale=2)
                        
                        saved_files.append({
                            'stage': stage_name,
                            'filename': filename,
                            'filepath': filepath,
                            'timestamp': timestamp.isoformat()
                        })
                        
                        logger.info(f"✓ Saved {stage_name} visualization: {filename}")
                        
                    else:
                        logger.error(f"Failed to create {stage_name} visualization")
                        
                except Exception as stage_error:
                    logger.error(f"Error in {stage_name} visualization: {stage_error}", exc_info=True)
                    # Continue with other stages - hard fail only if no visualizations created
                    continue
            
            # Create evolution summary metadata
            evolution_summary = {
                'soul_id': spark_id,
                'soul_name': name,
                'evolution_stages': len(saved_files),
                'created_timestamp': datetime.now().isoformat(),
                'files': saved_files,
                'final_metrics': {
                    'coherence': getattr(soul_spark, 'coherence', 0.0),
                    'stability': getattr(soul_spark, 'stability', 0.0),
                    'crystallization_level': getattr(soul_spark, 'crystallization_level', 0.0),
                    'creator_connection_strength': getattr(soul_spark, 'creator_connection_strength', 0.0),
                    'soul_frequency': getattr(soul_spark, 'soul_frequency', 0.0),
                    'aspects_count': len(getattr(soul_spark, 'aspects', {})),
                    'layers_count': len(getattr(soul_spark, 'layers', []))
                }
            }
            
            # Save evolution summary
            summary_filename = f"{name}_{spark_id}_{date_str}_evolution_summary.json"
            summary_filepath = os.path.join(self.output_dir, summary_filename)
            
            with open(summary_filepath, 'w') as f:
                json.dump(evolution_summary, f, indent=2, default=str)
            
            if not saved_files:
                raise RuntimeError("CRITICAL: No soul evolution visualizations were successfully created")
            
            logger.info(f"✓ Complete soul evolution visualization created: {len(saved_files)} stages")
            
            return {
                'success': True,
                'files_created': saved_files,
                'summary_file': summary_filepath,
                'total_visualizations': len(saved_files)
            }
            
        except Exception as e:
            logger.error(f"CRITICAL: Complete soul evolution visualization failed: {e}", exc_info=True)
            raise RuntimeError(f"Soul evolution visualization failed: {e}") from e

    def move_to_completed_souls(self, soul_name: str, spark_id: str, base_completed_dir: str) -> bool:
        """Move visualization files to completed souls directory."""
        try:
            # Create target directory
            timestamp = datetime.now().strftime("%Y%m%d")
            model_dir_name = f"{soul_name}_{spark_id}_{timestamp}"
            target_dir = os.path.join(base_completed_dir, model_dir_name, "visuals", "soul_evolution")
            
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
                    logger.info(f"Moved soul evolution file: {filename}")
            
            logger.info(f"✓ Moved {files_moved} soul evolution files to: {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move soul evolution files: {e}", exc_info=True)
            return False