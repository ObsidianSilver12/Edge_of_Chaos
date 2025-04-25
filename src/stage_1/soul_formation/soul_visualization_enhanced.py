"""
Enhanced Soul Visualization

This module provides detailed visualization capabilities for fully formed souls,
creating rich visual representations of the soul's properties, structure, and attributes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

from src.stage_1.soul_formation.soul_spark import SoulSpark
from src.constants.constants import *


class EnhancedSoulVisualizer:
    """
    Creates detailed visualizations for a fully formed soul.
    """
    
    def __init__(self, soul: SoulSpark, output_dir: str):
        """
        Initialize with a completed soul to visualize.
        
        Args:
            soul: Fully formed SoulSpark object
            output_dir: Directory to save visualizations
        """
        if not isinstance(soul, SoulSpark):
            raise TypeError("A SoulSpark object is required")
            
        self.soul = soul
        self.output_dir = output_dir
        
        # Create custom colormaps for soul visualization
        self.stability_cmap = self._create_stability_colormap()
        self.aspect_cmap = self._create_aspect_colormap()
        
    def _create_stability_colormap(self):
        """Create a custom colormap for stability visualization."""
        return LinearSegmentedColormap.from_list(
            "stability", 
            [(0.0, "red"), (0.5, "yellow"), (0.85, "lightblue"), (1.0, "purple")]
        )
        
    def _create_aspect_colormap(self):
        """Create a custom colormap for aspect visualization."""
        return LinearSegmentedColormap.from_list(
            "aspects", 
            [(0.0, "blue"), (0.2, "cyan"), (0.4, "green"), 
             (0.6, "gold"), (0.8, "orange"), (1.0, "magenta")]
        )
    
    def generate_all_visualizations(self, show: bool = True, save: bool = True) -> Dict[str, Any]:
        """
        Generate all visualizations for the soul.
        
        Args:
            show: Whether to display visualizations
            save: Whether to save visualizations to files
            
        Returns:
            Dictionary of visualization results
        """
        results = {}
        
        # Core structure visualization (3D)
        results["core_structure"] = self.visualize_core_structure(show, save)
        
        # Energy and harmony visualization
        results["energy_harmony"] = self.visualize_energy_harmony(show, save)
        
        # Frequency signature
        results["frequency_signature"] = self.visualize_frequency_signature(show, save)
        
        # Aspects map
        results["aspects_map"] = self.visualize_aspects_map(show, save)
        
        # Dimensional resonance
        results["dimensional_resonance"] = self.visualize_dimensional_resonance(show, save)
        
        # Coherence patterns
        results["coherence_patterns"] = self.visualize_coherence_patterns(show, save)
        
        # Life cord structure
        results["life_cord"] = self.visualize_life_cord(show, save)
        
        # Identity crystallization
        results["identity"] = self.visualize_identity(show, save)
        
        # Create a dashboard
        results["dashboard"] = self.create_soul_dashboard(show, save)
        
        return results
    
    def visualize_core_structure(self, show: bool = True, save: bool = True) -> plt.Figure:
        """
        Create a 3D visualization of the soul's core structure.
        
        Args:
            show: Whether to display visualization
            save: Whether to save visualization to file
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract core field representation (simplified if no detailed field)
        energy_field = getattr(self.soul, 'energy_field', None)
        if energy_field is None:
            # Create approximate field from available metrics
            field_size = 30
            energy_field = np.zeros((field_size, field_size, field_size))
            
            # Use stability and resonance to shape the field
            stability = self.soul.stability
            resonance = self.soul.resonance
            coherence = getattr(self.soul, 'coherence', 0.7)
            
            # Create a basic sphere with variations based on properties
            x, y, z = np.indices((field_size, field_size, field_size))
            center = field_size // 2
            r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
            
            # Base sphere
            sphere_mask = r < field_size/3
            energy_field[sphere_mask] = 0.5
            
            # Add variations based on stability (more stable = smoother)
            noise_level = max(0.1, 1.0 - stability)
            energy_field += np.random.normal(0, noise_level, energy_field.shape)
            
            # Add coherence patterns (higher coherence = more structured patterns)
            if coherence > 0.6:
                for i in range(5):
                    freq = 0.5 + i * 0.1
                    pattern = np.sin(freq * x/5) * np.cos(freq * y/5) * np.sin(freq * z/5)
                    energy_field += coherence * pattern * 0.2
            
            # Normalize the field
            energy_field = (energy_field - np.min(energy_field)) / (np.max(energy_field) - np.min(energy_field))
        
        # Create threshold for isosurface visualization
        threshold = 0.5
        
        # Create a mask for the isosurface
        verts, faces, _, _ = marching_cubes(energy_field, threshold)
        
        # Scale to appropriate size
        verts = verts / np.array(energy_field.shape)
        
        # Create colors based on soul properties
        colors = np.zeros(verts.shape)
        for i in range(verts.shape[0]):
            # Calculate position-dependent colors based on soul metrics
            height = verts[i, 2]
            radius = np.sqrt(verts[i, 0]**2 + verts[i, 1]**2)
            angle = np.arctan2(verts[i, 1], verts[i, 0])
            
            # Map properties to RGB (simplified approximation)
            r = 0.5 + 0.5 * np.sin(height * 5)
            g = 0.5 + 0.5 * np.sin(radius * 6)
            b = 0.5 + 0.5 * np.sin(angle * 4)
            
            # Adjust colors based on soul stability and resonance
            r = r * stability
            g = g * resonance
            b = b * coherence
            
            colors[i] = [r, g, b]
        
        # Plot the isosurface with colors
        mesh = Poly3DCollection(verts[faces], alpha=0.8)
        mesh.set_facecolor(colors[faces[:, 0]])
        ax.add_collection3d(mesh)
        
        # Add annotations for key soul metrics
        ax.text(0, 0, 1.2, f"Stability: {stability:.2f}", color='black', fontsize=12)
        ax.text(0, 0, 1.1, f"Resonance: {resonance:.2f}", color='black', fontsize=12)
        ax.text(0, 0, 1.0, f"Coherence: {coherence:.2f}", color='black', fontsize=12)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Soul Core Structure')
        
        # Set equal aspect ratio
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        if save:
            plt.savefig(f"{self.output_dir}/soul_core_structure.png", dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def visualize_energy_harmony(self, show: bool = True, save: bool = True) -> plt.Figure:
        """
        Visualize the soul's energy harmony patterns.
        
        Args:
            show: Whether to display visualization
            save: Whether to save visualization to file
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create a radial visualization of energy harmony
        # Extract relevant metrics
        stability = self.soul.stability
        resonance = self.soul.resonance
        coherence = getattr(self.soul, 'coherence', 0.7)
        
        # Get aspects if available
        aspects = getattr(self.soul, 'aspects', {})
        aspect_count = len(aspects)
        
        # Generate the plot
        ax = fig.add_subplot(111, polar=True)
        
        # Create harmony metrics - if we don't have detailed ones, approximate them
        harmony_metrics = getattr(self.soul, 'harmony_metrics', None)
        if harmony_metrics is None:
            # Create approximate metrics
            num_metrics = 12
            theta = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
            
            # Base values on stability, resonance, and add some variation
            base_value = (stability + resonance + coherence) / 3
            variation = 0.2
            values = base_value + variation * np.sin(theta * 5)
            
            # Ensure values are in [0, 1] range
            values = np.clip(values, 0, 1)
            
            # Create labels
            labels = [
                "Energy Flow", "Stability", "Resonance", "Coherence", 
                "Creator Bond", "Aspect Integration", "Earth Attunement", 
                "Sephiroth Harmony", "Identity Clarity", "Life Cord", 
                "Vibrational State", "Divine Alignment"
            ]
        else:
            # Use actual metrics
            values = []
            labels = []
            for key, value in harmony_metrics.items():
                labels.append(key)
                values.append(value)
            theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        
        # Create the radar chart
        ax.fill(theta, values, alpha=0.25)
        ax.plot(theta, values, marker='o', linewidth=2)
        ax.set_thetagrids(theta * 180/np.pi, labels)
        
        # Set limits and title
        ax.set_ylim(0, 1)
        ax.set_title("Soul Energy Harmony")
        
        # Add overall harmony score in center
        overall_harmony = np.mean(values)
        ax.text(0, 0, f"{overall_harmony:.2f}", ha='center', va='center', 
                fontsize=20, bbox=dict(facecolor='white', alpha=0.8))
        
        if save:
            plt.savefig(f"{self.output_dir}/soul_energy_harmony.png", dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def visualize_frequency_signature(self, show: bool = True, save: bool = True) -> plt.Figure:
        """
        Visualize the soul's frequency signature.
        
        Args:
            show: Whether to display visualization
            save: Whether to save visualization to file
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(14, 8))
        
        # Extract frequencies or generate them from soul properties
        frequencies = getattr(self.soul, 'frequencies', None)
        
        if frequencies is None:
            # Generate approximate frequencies based on soul properties
            base_freq = getattr(self.soul, 'base_frequency', 432.0)
            
            # Generate harmonic series with variations based on soul properties
            harmonics = 15
            harmonic_series = np.arange(1, harmonics + 1)
            
            # Apply stability variations to harmonic amplitudes
            stability = self.soul.stability
            resonance = self.soul.resonance
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
            # Frequencies using harmonics and phi variations
            frequencies = base_freq * harmonic_series
            
            # Add phi-based variations to make it more "alive"
            phi_variations = np.array([base_freq * phi**(i % 5) for i in range(harmonics)])
            frequencies = np.concatenate((frequencies, phi_variations))
            
            # Amplitudes with exponential decay modified by stability
            decay_factor = 0.9 - (stability * 0.2)  # More stable = slower decay
            amplitudes = np.exp(-decay_factor * np.arange(len(frequencies)))
            
            # Add variations based on resonance
            phase_shifts = resonance * np.sin(np.arange(len(frequencies)))
            
            # Create frequency dataframe
            # Create plot
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

        # Plot each frequency point
        for i in range(len(frequencies)):
            ax.plot([frequencies[i]], [amplitudes[i]], 'bo', alpha=0.6)
            # Add phase modulation
            ax.plot([frequencies[i]-2, frequencies[i]+2], 
                   [amplitudes[i]*np.cos(phase_shifts[i])]*2, 'b-', alpha=0.3)
        else:
            # Use actual frequencies if provided
            for freq, amp in frequencies.items():
                ax.plot([freq], [amp], 'bo', alpha=0.6)
        
        # Create the plot
        ax = fig.add_subplot(111)
        
        # Extract coherence value from soul properties
        coherence = getattr(self.soul, 'coherence', 0.7)  # Default to 0.7 if not found
        ax.set_title(f'Soul Coherence Patterns (Coherence: {coherence:.2f})')
        
        # Add interpretive note
        text = "Stable souls show regular patterns\n"
        text += "Resonant souls show stronger patterns\n"
        text += "Coherent souls show clear structure"
        ax.text(0.02, 0.02, text, transform=ax.transAxes, 
               bbox=dict(facecolor='white', alpha=0.8))
        
        if save:
            plt.savefig(f"{self.output_dir}/soul_coherence_patterns.png", dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def visualize_life_cord(self, show: bool = True, save: bool = True) -> plt.Figure:
        """
        Visualize the soul's life cord structure.
        
        Args:
            show: Whether to display visualization
            save: Whether to save visualization to file
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get life cord data or create placeholder
        life_cord = getattr(self.soul, 'life_cord', None)
        
        if life_cord is None:
            # Create placeholder life cord data
            life_cord = {
                'integrity': 0.85,
                'channels': 7,
                'harmonic_nodes': 12,
                'soul_anchor': {'position': [0, 0, 2], 'strength': 0.9},
                'earth_anchor': {'position': [0, 0, -2], 'strength': 0.8}
            }
        
        # Extract life cord properties
        integrity = life_cord.get('integrity', 0.8)
        num_channels = life_cord.get('channels', 7)
        soul_anchor = life_cord.get('soul_anchor', {'position': [0, 0, 2], 'strength': 0.9})
        earth_anchor = life_cord.get('earth_anchor', {'position': [0, 0, -2], 'strength': 0.8})
        harmonic_nodes = life_cord.get('harmonic_nodes', 12)
        
        # Plot the soul and earth anchors
        soul_pos = soul_anchor.get('position', [0, 0, 2])
        earth_pos = earth_anchor.get('position', [0, 0, -2])
        
        ax.scatter([soul_pos[0]], [soul_pos[1]], [soul_pos[2]], 
                  color='purple', s=300, alpha=0.8, label='Soul Anchor')
        ax.scatter([earth_pos[0]], [earth_pos[1]], [earth_pos[2]], 
                  color='green', s=300, alpha=0.8, label='Earth Anchor')
        
        # Create the life cord channels
        for i in range(num_channels):
            # Generate a slightly different path for each channel
            t = np.linspace(0, 1, 50)
            
            # Determine channel type (primary or secondary)
            is_primary = (i == 0)
            
            # Create base helix path from soul to earth
            radius = 0.2 if is_primary else 0.1 + (i * 0.02)
            frequency = 4 if is_primary else 3 + (i % 3)
            phase = 0 if is_primary else (i * np.pi / num_channels)
            
            # Modify path based on integrity
            radius *= 0.8 + (integrity * 0.4)  # Higher integrity = more regular
            
            # Calculate path coordinates
            x = radius * np.sin(2 * np.pi * frequency * t + phase)
            y = radius * np.cos(2 * np.pi * frequency * t + phase)
            z = np.linspace(soul_pos[2], earth_pos[2], len(t))
            
            # Add some variation based on position (less for higher integrity)
            variation = (1 - integrity) * 0.2
            x += variation * np.sin(4 * np.pi * t)
            y += variation * np.cos(3 * np.pi * t)
            
            # Determine color and width based on channel type
            if is_primary:
                color = 'gold'
                linewidth = 5
                alpha = 0.8
            else:
                color = plt.cm.cool(i / num_channels)
                linewidth = 2
                alpha = 0.6
                
            # Plot the channel
            ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)
        
        # Add harmonic nodes along the cord
        for i in range(harmonic_nodes):
            # Position node along the cord
            node_t = (i + 1) / (harmonic_nodes + 1)
            
            # Calculate node position with slight offset from center
            node_x = 0.1 * np.sin(node_t * np.pi * 2)
            node_y = 0.1 * np.cos(node_t * np.pi * 2)
            node_z = soul_pos[2] - node_t * (soul_pos[2] - earth_pos[2])
            
            # Node size based on position (larger in middle)
            node_size = 80 * (1 - 2 * abs(0.5 - node_t))
            
            # Node color based on function
            node_color = plt.cm.plasma(node_t)
            
            # Plot the node
            ax.scatter([node_x], [node_y], [node_z], color=node_color, 
                      s=node_size, alpha=0.8, edgecolor='white')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Soul Life Cord (Integrity: {integrity:.2f}, Channels: {num_channels})')
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio
        max_range = max(
            abs(soul_pos[2] - earth_pos[2]),
            2 * radius
        )
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(earth_pos[2] - max_range/4, soul_pos[2] + max_range/4)
        
        if save:
            plt.savefig(f"{self.output_dir}/soul_life_cord.png", dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def visualize_identity(self, show: bool = True, save: bool = True) -> plt.Figure:
        """
        Visualize the soul's identity crystallization.
        
        Args:
            show: Whether to display visualization
            save: Whether to save visualization to file
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Get identity data or create placeholder
        identity = getattr(self.soul, 'identity', None)
        
        if identity is None:
            # Create placeholder identity data
            identity = {
                'name': 'Luminara',
                'soul_color': [0.7, 0.3, 0.9],
                'voice_frequency': 432.0,
                'crystallization': 0.88,
                'divine_aspects': ['wisdom', 'compassion', 'creativity'],
                'resonance_signature': [0.8, 0.7, 0.9, 0.85, 0.75],
                'unique_pattern': np.random.random((20, 20))
            }
        
        # Extract identity properties
        name = identity.get('name', 'Unnamed')
        soul_color = identity.get('soul_color', [0.5, 0.5, 0.8])
        voice_freq = identity.get('voice_frequency', 432.0)
        crystallization = identity.get('crystallization', 0.8)
        divine_aspects = identity.get('divine_aspects', ['unknown'])
        
        # Create a 2x2 subplot grid
        grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)
        
        # 1. Name and Soul Color visualization (top left)
        ax1 = fig.add_subplot(grid[0, 0])
        
        # Convert soul color to RGB
        if isinstance(soul_color, list) and len(soul_color) == 3:
            rgb_color = soul_color
        else:
            rgb_color = [0.5, 0.5, 0.8]  # Default blue-purple
            
        # Create a color swatch
        gradient = np.linspace(0, 1, 100)
        gradient = np.vstack((gradient, gradient))
        
        # Plot color swatch with variations
        color_img = np.zeros((100, 100, 3))
        for i in range(3):
            # Add some variation to make it more interesting
            channel = np.outer(np.linspace(max(0, rgb_color[i]-0.2), min(1, rgb_color[i]+0.2), 100), 
                              np.ones(100))
            # Add some patterns
            channel += 0.05 * np.sin(np.outer(np.linspace(0, 4*np.pi, 100), np.ones(100)))
            # Ensure values are in [0, 1]
            channel = np.clip(channel, 0, 1)
            color_img[:, :, i] = channel
            
        ax1.imshow(color_img)
        ax1.axis('off')
        
        # Add name as title
        ax1.set_title(f"Soul Name: {name}", fontsize=14)
        
        # Add color information
        rgb_text = f"RGB: ({rgb_color[0]:.2f}, {rgb_color[1]:.2f}, {rgb_color[2]:.2f})"
        ax1.text(0.5, -0.1, rgb_text, transform=ax1.transAxes, ha='center')
        
        # 2. Voice Frequency visualization (top right)
        ax2 = fig.add_subplot(grid[0, 1])
        
        # Create time array for voice visualization
        t = np.linspace(0, 1, 1000)
        
        # Generate voice waveform
        voice_wave = np.sin(2 * np.pi * voice_freq * t)
        
        # Add harmonics based on crystallization
        for i in range(1, 5):
            harmonic_amp = crystallization / (i * 2)
            voice_wave += harmonic_amp * np.sin(2 * np.pi * voice_freq * (i+1) * t)
            
        # Plot voice waveform
        ax2.plot(t, voice_wave, color=rgb_color, linewidth=2)
        
        # Set labels and title
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Voice Frequency: {voice_freq:.2f} Hz')
        ax2.set_xlim(0, 0.05)  # Show just a portion for clarity
        
        # 3. Divine Aspects visualization (bottom left)
        ax3 = fig.add_subplot(grid[1, 0], polar=True)
        
        # Create a polar plot for divine aspects
        num_aspects = len(divine_aspects)
        theta = np.linspace(0, 2*np.pi, num_aspects, endpoint=False)
        
        # Equal strength for placeholder, or use actual data if available
        if hasattr(identity, 'aspect_strengths'):
            strengths = identity.get('aspect_strengths', [0.8] * num_aspects)
        else:
            strengths = [0.8] * num_aspects
            
        # Create aspect plot
        ax3.bar(theta, strengths, width=2*np.pi/num_aspects, bottom=0.2, 
               alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, num_aspects)))
        
        # Set aspect labels
        ax3.set_xticks(theta)
        ax3.set_xticklabels(divine_aspects)
        
        # Set title
        ax3.set_title('Divine Aspects')
        
        # 4. Crystallization Pattern (bottom right)
        ax4 = fig.add_subplot(grid[1, 1])
        
        # Get unique pattern or create one
        unique_pattern = identity.get('unique_pattern', None)
        if unique_pattern is None:
            # Create a crystalline pattern based on soul properties
            size = 50
            unique_pattern = np.zeros((size, size))
            
            # Create crystalline structure with symmetry
            for i in range(size):
                for j in range(size):
                    # Calculate distance from center
                    x = i - size/2
                    y = j - size/2
                    r = np.sqrt(x**2 + y**2)
                    
                    # Calculate angle
                    theta = np.arctan2(y, x)
                    
                    # Create crystalline pattern with 6-fold symmetry
                    pattern = np.sin(6 * theta) * np.exp(-r/(size/3))
                    
                    # Add more details based on crystallization
                    if crystallization > 0.7:
                        pattern += 0.3 * np.sin(12 * theta) * np.exp(-r/(size/6))
                        
                    # More crystallization = more structure
                    noise = np.random.random() * (1 - crystallization)
                    
                    unique_pattern[i, j] = pattern + noise
        
        # Normalize pattern
        unique_pattern = (unique_pattern - np.min(unique_pattern)) / (np.max(unique_pattern) - np.min(unique_pattern))
        
        # Plot the crystallization pattern
        im = ax4.imshow(unique_pattern, cmap='plasma', interpolation='bilinear')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax4)
        cbar.set_label('Pattern Intensity')
        
        # Set title
        ax4.set_title(f'Identity Crystallization: {crystallization:.2f}')
        ax4.axis('off')
        
        # Add overall title
        fig.suptitle(f'Soul Identity Profile: {name}', fontsize=16)
        
        if save:
            plt.savefig(f"{self.output_dir}/soul_identity.png", dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def create_soul_dashboard(self, show: bool = True, save: bool = True) -> plt.Figure:
        """
        Create a comprehensive dashboard with key soul metrics.
        
        Args:
            show: Whether to display visualization
            save: Whether to save visualization to file
                Returns:
            Figure object
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Set up subplot grid
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
        
        # Extract core soul metrics
        stability = self.soul.stability
        resonance = self.soul.resonance
        coherence = getattr(self.soul, 'coherence', 0.7)
        identity_data = getattr(self.soul, 'identity', {})
        name = identity_data.get('name', 'Unnamed Soul')
        crystallization = identity_data.get('crystallization', 0.8)
        life_cord = getattr(self.soul, 'life_cord', {})
        cord_integrity = life_cord.get('integrity', 0.8)
        aspects = getattr(self.soul, 'aspects', {})
        
        # 1. Core metrics gauge chart (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Stability', 'Resonance', 'Coherence', 'Crystallization', 'Cord Integrity']
        values = [stability, resonance, coherence, crystallization, cord_integrity]
        colors = ['green', 'blue', 'purple', 'gold', 'teal']
        
        bars = ax1.barh(metrics, values, color=colors, alpha=0.7)
        ax1.set_xlim(0, 1)
        ax1.set_title('Core Soul Metrics')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center')
        
        # 2. Simplified 3D structure (top middle)
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        
        # Create a simple soul field visualization
        phi = (1 + np.sqrt(5)) / 2
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        # Base sphere
        x = stability * np.outer(np.cos(u), np.sin(v))
        y = stability * np.outer(np.sin(u), np.sin(v))
        z = stability * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Add variations based on resonance
        variation = 0.1 * resonance * np.sin(phi * u[:, np.newaxis] + v[np.newaxis, :])
        x += variation
        y += variation * np.cos(u[:, np.newaxis])
        z += variation * np.sin(v[np.newaxis, :])
        
        # Create colormap based on coherence
        theta = np.arctan2(y, x)
        phi_angle = np.arctan2(np.sqrt(x**2 + y**2), z)
        
        colors = np.zeros((*x.shape, 4))
        colors[..., 0] = 0.5 + 0.5 * np.sin(phi_angle + theta)  # Red
        colors[..., 1] = 0.4 + 0.4 * np.cos(phi_angle * phi)    # Green
        colors[..., 2] = 0.6 + 0.4 * np.sin(theta * phi)        # Blue
        colors[..., 3] = 0.7  # Alpha
        
        # Plot the surface
        ax2.plot_surface(x, y, z, facecolors=colors, antialiased=True)
        
        # Set labels and title
        ax2.set_title('Soul Energy Field')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_box_aspect([1,1,1])
        
        # 3. Identity summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Get soul color
        soul_color = identity_data.get('soul_color', [0.5, 0.5, 0.8])
        
        # Create identity summary
        identity_text = f"Name: {name}\n"
        identity_text += f"Crystallization: {crystallization:.2f}\n"
        
        # Add voice frequency if available
        voice_freq = identity_data.get('voice_frequency', None)
        if voice_freq:
            identity_text += f"Voice Frequency: {voice_freq:.1f} Hz\n"
            
        # Add divine aspects if available
        divine_aspects = identity_data.get('divine_aspects', [])
        if divine_aspects:
            identity_text += f"Divine Aspects: {', '.join(divine_aspects[:3])}"
            if len(divine_aspects) > 3:
                identity_text += f" +{len(divine_aspects)-3} more"
        
        # Create a gradient background using soul color
        gradient = np.zeros((100, 100, 3))
        for i in range(3):
            gradient[:, :, i] = np.linspace(1, soul_color[i], 100)[:, np.newaxis]
        
        ax3.imshow(gradient)
        
        # Add identity text
        ax3.text(0.5, 0.5, identity_text, ha='center', va='center',
                transform=ax3.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax3.set_title('Soul Identity')
        ax3.axis('off')
        
        # 4. Aspect distribution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Count aspects by category/Sephirah
        aspect_categories = {}
        for aspect, data in aspects.items():
            # Try to determine category
            category = "Other"
            
            for sep in ["kether", "chokmah", "binah", "chesed", "geburah", 
                      "tiphareth", "netzach", "hod", "yesod", "malkuth", "daath"]:
                if sep in aspect.lower():
                    category = sep.capitalize()
                    break
            
            # Add to counter
            if category not in aspect_categories:
                aspect_categories[category] = 0
            aspect_categories[category] += 1
        
        # If no aspects, create placeholder data
        if not aspect_categories:
            aspect_categories = {
                "Kether": 3, "Chokmah": 2, "Binah": 4, 
                "Chesed": 3, "Geburah": 2, "Tiphareth": 5,
                "Netzach": 2, "Hod": 3, "Yesod": 4, "Malkuth": 3
            }
        
        # Create pie chart of aspect distribution
        labels = aspect_categories.keys()
        sizes = aspect_categories.values()
        
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
               shadow=True, startangle=90, 
               colors=plt.cm.tab20(np.linspace(0, 1, len(labels))))
        
        ax4.set_title('Soul Aspect Distribution')
        
        # 5. Frequency signature (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Get frequency data or generate placeholder
        base_freq = getattr(self.soul, 'base_frequency', 432.0)
        
        # Generate frequency spectrum
        x = np.linspace(base_freq * 0.8, base_freq * 2.5, 1000)
        
        # Create peaks at key frequencies (base freq and harmonics)
        y = np.zeros_like(x)
        phi = (1 + np.sqrt(5)) / 2
        
        for i in range(5):
            # Main harmonics
            freq = base_freq * (i + 1)
            amp = 1.0 / (i + 1) * stability
            width = 5 + 10 * (1 - coherence)  # More coherent = sharper peaks
            y += amp * np.exp(-(x - freq)**2 / width)
            
            # Phi-based harmonics
            if i < 3:
                phi_freq = base_freq * phi**(i+1)
                phi_amp = 0.3 / (i + 1) * resonance
                y += phi_amp * np.exp(-(x - phi_freq)**2 / width)
        
        # Plot frequency spectrum
        ax5.plot(x, y, color='blue', linewidth=2)
        
        # Mark key frequencies
        for i in range(3):
            freq = base_freq * (i + 1)
            amp = 1.0 / (i + 1) * stability
            ax5.plot([freq, freq], [0, amp], 'r--', alpha=0.5)
            ax5.text(freq, amp + 0.02, f"{freq:.1f} Hz", ha='center')
        
        # Set labels and title
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Amplitude')
        ax5.set_title('Soul Frequency Signature')
        
        # 6. Life cord visualization (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Extract life cord properties or use defaults
        channels = life_cord.get('channels', 7)
        nodes = life_cord.get('harmonic_nodes', 12)
        
        # Create simplified 2D life cord visualization
        cord_length = 100
        x = np.linspace(0, 1, cord_length)
        
        # Plot the main cord
        ax6.plot(x, np.zeros_like(x), 'k-', linewidth=3, alpha=0.7)
        
        # Add channels
        for i in range(channels):
            # Primary channel is straight, others are wavy
            if i == 0:
                y = np.zeros_like(x)
                color = 'gold'
                lw = 4
                alpha = 0.8
            else:
                freq = 8 + i
                phase = i * np.pi / channels
                amp = 0.1 + 0.02 * i
                y = amp * np.sin(freq * np.pi * x + phase)
                color = plt.cm.cool(i / channels)
                lw = 2
                alpha = 0.6
            
            ax6.plot(x, y, '-', color=color, linewidth=lw, alpha=alpha)
        
        # Add harmonic nodes
        for i in range(nodes):
            node_x = (i + 1) / (nodes + 1)
            node_y = 0
            
            size = 100 * (1 - 2 * abs(0.5 - node_x))
            color = plt.cm.plasma(node_x)
            
            ax6.scatter(node_x, node_y, s=size, color=color, 
                      alpha=0.8, edgecolor='white')
        
        # Add anchors
        ax6.scatter(0, 0, s=200, color='purple', alpha=0.7, label='Soul Anchor')
        ax6.scatter(1, 0, s=200, color='green', alpha=0.7, label='Earth Anchor')
        
        # Set labels and title
        ax6.set_xlabel('Position')
        ax6.set_title(f'Life Cord (Integrity: {cord_integrity:.2f}, Channels: {channels})')
        ax6.set_ylim(-0.3, 0.3)
        ax6.legend(loc='upper right')
        ax6.axis('off')
        
        # 7. Dimensional resonance (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        
        # Create simplified dimensional resonance
        dimensions = ["Earth", "Physical", "Malkuth", "Yesod", "Tiphareth", "Kether"]
        
        # Create resonance values based on available metrics
        values = []
        for dim in dimensions:
            if dim == "Earth":
                # Earth connection should be strong for a birth-ready soul
                val = 0.8 + 0.2 * cord_integrity
            elif dim == "Physical":
                # Physical resonance should be good for birth-ready soul
                val = 0.7 + 0.3 * stability
            elif dim == "Malkuth":
                # Malkuth similar to Earth
                val = 0.75 + 0.25 * cord_integrity
            elif dim == "Yesod":
                # Yesod (foundation) important for identity
                val = 0.6 + 0.4 * crystallization
            elif dim == "Tiphareth":
                # Tiphareth (beauty) connected to coherence
                val = 0.5 + 0.5 * coherence
            elif dim == "Kether":
                # Kether (crown) indicates ongoing creator connection
                val = 0.4 + 0.6 * resonance
            else:
                val = 0.5
                
            values.append(min(1.0, val))  # Ensure value is at most 1.0
            
        # Create horizontal bar chart
        bars = ax7.barh(dimensions, values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(dimensions))), 
                      alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax7.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center')
        
        # Set labels and title
        ax7.set_xlim(0, 1.1)
        ax7.set_title('Dimensional Resonance')
        
        # 8. Coherence pattern (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        
        # Create a simplified coherence visualization
        grid_size = 50
        pattern = np.zeros((grid_size, grid_size))
        
        # Create base pattern with phi ratio influence
        phi = (1 + np.sqrt(5)) / 2
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Pattern based on soul metrics
        pattern = np.sin(stability * X * np.pi) * np.cos(resonance * Y * np.pi)
        pattern += coherence * np.sin(phi * X * Y * np.pi) * 0.3
        
        # Add structured noise - less for more coherent souls
        noise = np.random.normal(0, 0.3 * (1 - coherence), pattern.shape)
        pattern += noise
        
        # Normalize pattern
        pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))
        
        # Plot the coherence pattern
        im = ax8.imshow(pattern, cmap='plasma', origin='lower')
        ax8.set_title(f'Coherence Pattern (Coherence: {coherence:.2f})')
        ax8.axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax8)
        cbar.set_label('Pattern Energy')
        
        # 9. Summary metrics with divine connection (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        
        # Create a radar chart with key final metrics
        categories = ['Stability', 'Resonance', 'Coherence', 
                     'Identity', 'Life Cord', 'Earth Connection']
        
        # Create values based on soul properties
        values = [
            stability,
            resonance,
            coherence,
            crystallization,
            cord_integrity,
            cord_integrity * 0.8 + stability * 0.2
        ]
        
        # Complete the loop for the radar chart
        values = np.concatenate((values, [values[0]]))
        categories = np.concatenate((categories, [categories[0]]))
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
        
        # Create radar chart
        ax9.plot(angles, values, 'o-', linewidth=2)
        ax9.fill(angles, values, alpha=0.25)
        ax9.set_thetagrids(angles * 180/np.pi, categories)
        
        # Set radar chart limits
        ax9.set_ylim(0, 1)
        ax9.set_title('Soul Balance Summary')
        
        # Calculate overall readiness score
        readiness = (stability + resonance + coherence + 
                    crystallization + cord_integrity) / 5
        
        # Add readiness score in center
        ax9.text(0, 0, f"{readiness:.2f}", ha='center', va='center', 
               fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        # Add soul name as overall title
        birth_ready = "Birth Ready" if readiness > 0.8 else "Developing"
        fig.suptitle(f"Soul Dashboard: {name} ({birth_ready})", fontsize=20, y=0.98)
        
        if save:
            plt.savefig(f"{self.output_dir}/soul_dashboard.png", dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig


def generate_sample_soul_visualizations(output_dir: str = "soul_visualizations"):
    """
    Generate sample visualizations for a test soul.
    
    Args:
        output_dir: Directory to save visualizations
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample SoulSpark
    # This would normally be from loading an actual SoulSpark
    # but we'll create a simple placeholder here
    class SampleSoul:
        def __init__(self):
            self.stability = 0.92
            self.resonance = 0.88
            self.coherence = 0.85
            self.base_frequency = 432.0
            
            # Identity
            self.identity = {
                'name': 'Seraphina',
                'soul_color': [0.7, 0.3, 0.9],
                'voice_frequency': 432.0,
                'crystallization': 0.89,
                'divine_aspects': ['wisdom', 'compassion', 'creativity', 'joy'],
                'resonance_signature': [0.8, 0.7, 0.9, 0.85, 0.75]
            }
            
            # Life cord
            self.life_cord = {
                'integrity': 0.91,
                'channels': 7,
                'harmonic_nodes': 12,
                'soul_anchor': {'position': [0, 0, 2], 'strength': 0.9},
                'earth_anchor': {'position': [0, 0, -2], 'strength': 0.85}
            }
            
            # Aspects
            self.aspects = {
                'kether_light': {'strength': 0.85, 'integration': 0.82},
                'chokmah_wisdom': {'strength': 0.78, 'integration': 0.73},
                'binah_understanding': {'strength': 0.82, 'integration': 0.78},
                'chesed_mercy': {'strength': 0.75, 'integration': 0.77},
                'geburah_strength': {'strength': 0.72, 'integration': 0.68},
                'tiphareth_beauty': {'strength': 0.90, 'integration': 0.85},
                'netzach_victory': {'strength': 0.68, 'integration': 0.72},
                'hod_splendor': {'strength': 0.74, 'integration': 0.69},
                'yesod_foundation': {'strength': 0.88, 'integration': 0.82},
                'malkuth_kingdom': {'strength': 0.86, 'integration': 0.89}
            }
            
            # Placeholder metrics
            self.harmony_metrics = {
                "Energy Flow": 0.88,
                "Stability": 0.92,
                "Resonance": 0.88, 
                "Coherence": 0.85,
                "Creator Bond": 0.79,
                "Aspect Integration": 0.83,
                "Earth Attunement": 0.90,
                "Sephiroth Harmony": 0.87,
                "Identity Clarity": 0.89,
                "Life Cord": 0.91,
                "Vibrational State": 0.86,
                "Divine Alignment": 0.84
            }
    
    # Create sample soul
    sample_soul = SampleSoul()
    
    # Create visualizer
    visualizer = EnhancedSoulVisualizer(sample_soul, output_dir)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(show=True, save=True)
    
    print(f"Sample soul visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Generate sample visualizations when run directly
    generate_sample_soul_visualizations()



