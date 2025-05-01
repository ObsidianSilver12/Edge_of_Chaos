"""
Soul Journey Visualization Module

Provides visualization capabilities for tracking and displaying the soul's
journey through different stages of formation and development.
"""

import logging
import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

# Constants (import from project constants if needed)
PI = 3.14159265358979323846

logger = logging.getLogger(__name__)

class SoulJourneyVisualizer:
    """Creates visualizations for soul journeys through development stages."""
    
    def __init__(self, output_dir: str = "output/visualizations/journey"):
        """Initialize the soul journey visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise
        
        # Define colormaps for different journey properties
        self.stage_cmap = LinearSegmentedColormap.from_list(
            "stage_cmap", ["#001f3f", "#0074D9", "#7FDBFF", "#39CCCC", "#3D9970", "#2ECC40", "#01FF70", "#FFDC00", "#FF851B", "#FF4136"]
        )  # Dark Blue -> Blue -> Light Blue -> ... -> Red (progression)
        
        logger.info(f"SoulJourneyVisualizer initialized")
    
    def _save_or_show(self, fig, filename: str, show: bool, save: bool):
        """Helper to save and/or show a matplotlib figure."""
        try:
            if save:
                full_path = os.path.join(self.output_dir, filename)
                fig.savefig(full_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {full_path}")
            if show:
                # Check if running in interactive environment before showing
                if hasattr(plt, 'isinteractive') and plt.isinteractive():
                    plt.show()
                else:
                    logger.info("Non-interactive environment detected, plot not shown automatically.")
            if not show:  # Close if not shown to free memory
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error saving/showing plot {filename}: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
    
    def visualize_journey_timeline(self, soul, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the soul's journey timeline through all formation stages.
        
        Args:
            soul: SoulSpark instance
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        try:
            # Get soul attributes
            soul_name = getattr(soul, 'name', 'Unknown Soul')
            soul_id = getattr(soul, 'spark_id', 'unknown')
            
            # Define journey stages in order
            stages = [
                {"name": "Guff Strengthening", "flag": "guff_strengthened", "ready_flag": "ready_for_guff", "desc": "Energy infusion in the Hall of Souls"},
                {"name": "Sephiroth Journey", "flag": "sephiroth_journey_complete", "ready_flag": "ready_for_journey", "desc": "Passage through the Tree of Life fields"},
                {"name": "Creator Entanglement", "flag": "creator_channel_id", "ready_flag": "ready_for_entanglement", "desc": "Direct resonance with divine essence"},
                {"name": "Harmonic Strengthening", "flag": "harmonically_strengthened", "ready_flag": "ready_for_strengthening", "desc": "Enhancement of soul frequencies"},
                {"name": "Life Cord Formation", "flag": "cord_formation_complete", "ready_flag": "ready_for_life_cord", "desc": "Creation of energetic connection"},
                {"name": "Earth Harmonization", "flag": "earth_harmonized", "ready_flag": "ready_for_earth", "desc": "Attunement to Earth's frequencies"},
                {"name": "Identity Crystallization", "flag": "identity_crystallized", "ready_flag": "ready_for_identity", "desc": "Stabilization of soul attributes"},
                {"name": "Birth Process", "flag": "incarnated", "ready_flag": "ready_for_birth", "desc": "Final incarnation into physical form"}
            ]
            
            # Extract completion and readiness data
            completion_states = []
            readiness_states = []
            for stage in stages:
                # Check if flag exists and is True
                completed = getattr(soul, stage["flag"], False)
                if stage["flag"] == "creator_channel_id":  # Special case for entanglement
                    completed = getattr(soul, stage["flag"], "") != ""
                    
                ready = getattr(soul, stage["ready_flag"], False)
                
                completion_states.append(completed)
                readiness_states.append(ready)
            
            # Create the timeline
            stage_names = [s["name"] for s in stages]
            stage_descs = [s["desc"] for s in stages]
            y_positions = np.arange(len(stages))
            
            # Plot horizontal timeline
            ax.plot([0, len(stages)-1], [0, 0], 'k-', alpha=0.3, linewidth=2)
            
            # Plot completion markers
            for i, (completed, ready) in enumerate(zip(completion_states, readiness_states)):
                if completed:
                    # Completed stage (filled circle)
                    ax.scatter(i, 0, s=300, marker='o', color='green', edgecolors='darkgreen', zorder=10)
                    
                    # Draw line to next point if not the last one
                    if i < len(stages) - 1:
                        # Line to next stage, style depends if next is ready or not
                        next_color = 'darkgreen' if completion_states[i+1] else ('orange' if readiness_states[i+1] else 'gray')
                        ax.plot([i, i+1], [0, 0], color=next_color, linewidth=3, alpha=0.7)
                elif ready:
                    # Ready but not completed (half-filled circle)
                    ax.scatter(i, 0, s=200, marker='o', color='orange', edgecolors='darkorange', alpha=0.7, zorder=5)
                else:
                    # Not ready, not completed (empty circle)
                    ax.scatter(i, 0, s=150, marker='o', facecolors='none', edgecolors='gray', alpha=0.5)
            
            # Add stage names with markers
            for i, (name, desc) in enumerate(zip(stage_names, stage_descs)):
                # Determine text position (alternating above/below)
                y_pos = 0.2 if i % 2 == 0 else -0.2
                align = 'bottom' if i % 2 == 0 else 'top'
                
                # Determine text color based on completion state
                if completion_states[i]:
                    color = 'darkgreen'
                    weight = 'bold'
                elif readiness_states[i]:
                    color = 'darkorange'
                    weight = 'normal'
                else:
                    color = 'gray'
                    weight = 'normal'
                
                # Add stage name
                ax.text(i, y_pos, name, ha='center', va=align, color=color, 
                       fontweight=weight, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                
                # Add description in smaller text
                desc_y = y_pos + (0.15 if i % 2 == 0 else -0.15)
                ax.text(i, desc_y, desc, ha='center', va=align, color=color, 
                       fontsize=8, alpha=0.8, 
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
            
            # Format plot
            ax.set_title(f"Soul Journey Timeline: {soul_name}", fontsize=14)
            ax.set_xlim(-0.5, len(stages)-0.5)
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Add progress indicator
            completed_count = sum(completion_states)
            progress_text = f"Journey Progress: {completed_count}/{len(stages)} stages completed"
            ax.text(len(stages)/2, -0.8, progress_text, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Add soul information
            ax.text(0, 0.8, f"Soul ID: {soul_id}", fontsize=10, ha='left',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            self._save_or_show(fig, f"soul_journey_timeline_{soul_id}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating journey timeline visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_sephiroth_journey(self, soul, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the soul's journey through Sephiroth fields.
        
        Args:
            soul: SoulSpark instance
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(12, 14))
        
        try:
            # Get soul attributes
            soul_name = getattr(soul, 'name', 'Unknown Soul')
            soul_id = getattr(soul, 'spark_id', 'unknown')
            journey_complete = getattr(soul, 'sephiroth_journey_complete', False)
            memory_echoes = getattr(soul, 'memory_echoes', {})
            
            # Extract Sephiroth journey data from memory echoes
            sephiroth_echoes = {}
            sephiroth_order = []
            current_sephiroth = None
            
            for echo_id, echo in memory_echoes.items():
                if "entered_sephiroth" in echo_id:
                    sephiroth_name = echo.get('sephiroth_name', 'Unknown')
                    sephiroth_order.append(sephiroth_name)
                    time_stamp = echo.get('timestamp', 0)
                    
                    # Get metrics at time of entry
                    metrics = {
                        'resonance': echo.get('resonance', 0),
                        'stability': echo.get('stability', 0),
                        'coherence': echo.get('coherence', 0),
                        'energy': echo.get('energy', 0)
                    }
                    
                    sephiroth_echoes[sephiroth_name] = {
                        'timestamp': time_stamp,
                        'metrics': metrics,
                        'completed': True  # Assume completed if in memory
                    }
            
            # If no journey data, show empty visualization
            if not sephiroth_echoes:
                ax = fig.add_subplot(111)
                if journey_complete:
                    msg = "Sephiroth Journey completed, but no memory echoes found"
                else:
                    msg = "Sephiroth Journey not yet started"
                ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f"Sephiroth Journey: {soul_name}")
                self._save_or_show(fig, f"sephiroth_journey_{soul_id}.png", show, save)
                return fig
            
            # Define Tree of Life positions for visualization
            tree_positions = {
                "Kether": (5, 9.5),
                "Chokmah": (3, 8),
                "Binah": (7, 8),
                "Chesed": (3, 6),
                "Geburah": (7, 6),
                "Tiphareth": (5, 5),
                "Netzach": (3, 4),
                "Hod": (7, 4),
                "Yesod": (5, 2.5),
                "Malkuth": (5, 1)
            }
            
            # Define standard paths through the Tree of Life
            tree_paths = [
                ("Kether", "Chokmah"), ("Kether", "Binah"), ("Chokmah", "Binah"),
                ("Chokmah", "Tiphareth"), ("Chokmah", "Chesed"), ("Binah", "Tiphareth"),
                ("Binah", "Geburah"), ("Chesed", "Geburah"), ("Chesed", "Tiphareth"),
                ("Chesed", "Netzach"), ("Geburah", "Tiphareth"), ("Geburah", "Hod"),
                ("Tiphareth", "Netzach"), ("Tiphareth", "Hod"), ("Tiphareth", "Yesod"),
                ("Netzach", "Hod"), ("Netzach", "Yesod"), ("Netzach", "Malkuth"),
                ("Hod", "Yesod"), ("Hod", "Malkuth"), ("Yesod", "Malkuth")
            ]
            
            # Create the Tree of Life visualization
            ax = fig.add_subplot(111)
            
            # Draw standard paths (dim)
            for path in tree_paths:
                if path[0] in tree_positions and path[1] in tree_positions:
                    x = [tree_positions[path[0]][0], tree_positions[path[1]][0]]
                    y = [tree_positions[path[0]][1], tree_positions[path[1]][1]]
                    ax.plot(x, y, 'k-', linewidth=1, alpha=0.3)
            
            # Draw the soul's journey path
            if len(sephiroth_order) > 1:
                for i in range(len(sephiroth_order) - 1):
                    start = sephiroth_order[i]
                    end = sephiroth_order[i+1]
                    
                    if start in tree_positions and end in tree_positions:
                        x = [tree_positions[start][0], tree_positions[end][0]]
                        y = [tree_positions[start][1], tree_positions[end][1]]
                        ax.plot(x, y, 'r-', linewidth=2.5, alpha=0.7)
            
            # Plot all Sephiroth nodes
            for name, position in tree_positions.items():
                if name in sephiroth_echoes:
                    # Soul has visited this Sephirah
                    metrics = sephiroth_echoes[name]['metrics']
                    resonance = metrics['resonance']
                    size = 700 + 500 * resonance
                    
                    # Use color based on position in journey
                    journey_pos = sephiroth_order.index(name) / max(1, len(sephiroth_order) - 1)
                    color = self.stage_cmap(journey_pos)
                    
                    # Plot with journey order number
                    ax.scatter(position[0], position[1], s=size, color=color, alpha=0.8, 
                              edgecolors='white', linewidth=1, zorder=10)
                    ax.text(position[0], position[1], str(sephiroth_order.index(name) + 1), 
                           ha='center', va='center', fontsize=14, color='white', fontweight='bold')
                else:
                    # Soul hasn't visited this Sephirah
                    ax.scatter(position[0], position[1], s=300, color='lightgray', alpha=0.3, edgecolors='gray')
                
                # Add Sephirah name
                ax.text(position[0], position[1] - 0.4, name, ha='center', va='top', fontsize=10)
            
            # Add legend/info for journey order
            if sephiroth_order:
                journey_text = "Journey Order:\n" + " → ".join(sephiroth_order)
                ax.text(0.5, 0.02, journey_text, ha='center', va='bottom', fontsize=10,
                       transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            # Format plot
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add title and completion status
            title = f"Sephiroth Journey: {soul_name}"
            if journey_complete:
                title += " (COMPLETED)"
            ax.set_title(title, fontsize=16)
            
            # Add metrics evolution if journey has multiple steps
            if len(sephiroth_order) > 1:
                # Create inset axes for metrics evolution
                inset_ax = fig.add_axes([0.15, 0.15, 0.25, 0.2], facecolor='whitesmoke')
                
                # Extract metrics for each step
                steps = range(1, len(sephiroth_order) + 1)
                resonance_values = [sephiroth_echoes[name]['metrics']['resonance'] for name in sephiroth_order]
                stability_values = [sephiroth_echoes[name]['metrics']['stability'] for name in sephiroth_order]
                coherence_values = [sephiroth_echoes[name]['metrics']['coherence'] for name in sephiroth_order]
                
                # Plot metrics evolution
                inset_ax.plot(steps, resonance_values, 'b-o', label='Resonance', linewidth=2, markersize=4)
                inset_ax.plot(steps, stability_values, 'g-o', label='Stability', linewidth=2, markersize=4)
                inset_ax.plot(steps, coherence_values, 'r-o', label='Coherence', linewidth=2, markersize=4)
                
                inset_ax.set_title('Metrics Evolution', fontsize=10)
                inset_ax.set_xlim(0.5, len(sephiroth_order) + 0.5)
                inset_ax.set_ylim(0, 1)
                inset_ax.set_xlabel('Journey Step', fontsize=8)
                inset_ax.set_ylabel('Value', fontsize=8)
                inset_ax.legend(loc='upper left', fontsize=8)
                inset_ax.grid(True, alpha=0.3)
            
            self._save_or_show(fig, f"sephiroth_journey_{soul_id}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating Sephiroth journey visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_creator_entanglement(self, soul, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the creator entanglement process and connection.
        
        Args:
            soul: SoulSpark instance
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(10, 8))
        
        try:
            # Get soul attributes
            soul_name = getattr(soul, 'name', 'Unknown Soul')
            soul_id = getattr(soul, 'spark_id', 'unknown')
            creator_channel_id = getattr(soul, 'creator_channel_id', '')
            creator_connection_strength = getattr(soul, 'creator_connection_strength', 0.0)
            creator_alignment = getattr(soul, 'creator_alignment', 0.0)
            creator_aspects = getattr(soul, 'creator_aspects', {})
            
            # If no entanglement, show empty visualization
            if not creator_channel_id:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Creator Entanglement not established", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f"Creator Entanglement: {soul_name}")
                self._save_or_show(fig, f"creator_entanglement_{soul_id}.png", show, save)
                return fig
            
            # Create the visualization - 3D entanglement
            ax = fig.add_subplot(111, projection='3d')
            
            # Define coordinates
            soul_pos = [0, 0, 0]  # Soul at origin
            creator_pos = [0, 0, 3]  # Creator above
            
            # Plot soul and creator points
            ax.scatter(*soul_pos, s=300, color='blue', alpha=0.7, label='Soul')
            ax.scatter(*creator_pos, s=500, color='gold', alpha=0.7, label='Creator Essence')
            
            # Create connection visualization based on strength
            if creator_connection_strength > 0:
                # Draw connection as a tube/helix
                t = np.linspace(0, 1, 100)
                helix_radius = 0.2 + 0.3 * creator_connection_strength
                helix_freq = 3 + 5 * creator_alignment  # More aligned = higher frequency
                
                # Create helix between points
                x = soul_pos[0] + helix_radius * np.sin(2 * PI * helix_freq * t)
                y = soul_pos[1] + helix_radius * np.cos(2 * PI * helix_freq * t)
                z = soul_pos[2] + (creator_pos[2] - soul_pos[2]) * t
                
                # Color based on connection strength
                colors = plt.colormaps['viridis'](t)  # Using viridis colormap which is always available
                
                # Plot connection
                for i in range(len(t)-1):
                    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=2 + 3 * creator_connection_strength, alpha=0.7)
                
                # Add connection aura
                if creator_connection_strength > 0.5:
                    # Create a semi-transparent aura around the connection
                    u = np.linspace(0, 2 * PI, 20)
                    v = np.linspace(0, PI, 20)
                    aura_radius = 0.8 * creator_connection_strength
                    
                    # Create ellipsoid between points
                    aura_x = aura_radius * np.outer(np.cos(u), np.sin(v))
                    aura_y = aura_radius * np.outer(np.sin(u), np.sin(v))
                    aura_z = 1.5 * np.outer(np.ones_like(u), np.cos(v) * 0.5 + 0.5)
                    
                    # Plot with semi-transparency
                    ax.plot_surface(aura_x, aura_y, aura_z, color='gold', alpha=0.1)
            
            # Plot creator aspects as smaller points around the creator
            if creator_aspects:
                aspect_count = len(creator_aspects)
                aspect_radius = 0.8
                
                for i, (aspect, value) in enumerate(creator_aspects.items()):
                    # Position around creator in a circle
                    angle = 2 * PI * i / aspect_count
                    aspect_x = creator_pos[0] + aspect_radius * np.cos(angle)
                    aspect_y = creator_pos[1] + aspect_radius * np.sin(angle)
                    aspect_z = creator_pos[2] + 0.3
                    
                    # Size and alpha based on aspect strength
                    aspect_size = 50 + 100 * value
                    aspect_alpha = 0.3 + 0.7 * value
                    
                    # Plot aspect point
                    ax.scatter(aspect_x, aspect_y, aspect_z, s=aspect_size, 
                              color='orange', alpha=aspect_alpha)
                    
                    # Add aspect name
                    ax.text(aspect_x, aspect_y, aspect_z + 0.2, aspect, 
                           ha='center', va='center', fontsize=8, color='darkgoldenrod')
            
            # Customize 3D plot
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(-0.5, 4)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            
            # Add connection strength indicator
            connection_info = (
                f"Connection Strength: {creator_connection_strength:.2f}\n"
                f"Creator Alignment: {creator_alignment:.2f}\n"
                f"Channel ID: {creator_channel_id[:8]}..."
            )
            ax.text2D(0.02, 0.02, connection_info, transform=ax.transAxes, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            # Title
            ax.set_title(f"Creator Entanglement: {soul_name}", fontsize=14, y=1.05)
            
            self._save_or_show(fig, f"creator_entanglement_{soul_id}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating Creator Entanglement visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_birth_process(self, soul, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the birth process including mother resonance if present.
        
        Args:
            soul: SoulSpark instance
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(12, 9))
        
        try:
            # Get soul attributes
            soul_name = getattr(soul, 'name', 'Unknown Soul')
            soul_id = getattr(soul, 'spark_id', 'unknown')
            incarnated = getattr(soul, 'incarnated', False)
            birth_data = getattr(soul, 'birth_data', {})
            memory_echoes = getattr(soul, 'memory_echoes', {})
            
            # If not incarnated, show empty visualization
            if not incarnated:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Birth Process not completed", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f"Birth Process: {soul_name}")
                self._save_or_show(fig, f"birth_process_{soul_id}.png", show, save)
                return fig
            
            # Extract birth metrics from memory echoes
            birth_metrics = {}
            mother_influence = {}
            
            # Look for birth-related echoes
            for echo_id, echo in memory_echoes.items():
                if "birth_process" in echo_id:
                    # Extract metrics
                    birth_metrics = {
                        'birth_intensity': echo.get('birth_intensity', 0.0),
                        'trauma_level': echo.get('trauma_level', 0.0),
                        'memory_retention': echo.get('memory_retention', 0.0),
                        'cord_efficiency': echo.get('cord_efficiency', 0.0),
                        'breath_synch': echo.get('breath_synchronization', 0.0),
                        'acceptance': echo.get('acceptance_level', 0.0)
                    }
                    
                    # Extract mother influence if present
                    if 'mother_influence' in echo:
                        mother_influence = echo['mother_influence']
            
            # Create a grid for the visualizations
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.2, 1, 0.8], hspace=0.3, wspace=0.3)
            
            # --- 1. Birth Process Diagram (Top Left) ---
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Draw simplified birth diagram
            ax1.set_xlim(0, 10)
            ax1.set_ylim(0, 10)
            
            # Draw the physical plane
            ax1.axhspan(0, 2, alpha=0.2, color='brown', label='Physical Plane')
            
            # Draw the veil of forgetting
            ax1.axhspan(3, 4, alpha=0.3, color='silver', label='Memory Veil')
            
            # Draw the spiritual plane
            ax1.axhspan(6, 10, alpha=0.2, color='blue', label='Spiritual Plane')
            
            # Draw the soul's path
            # Path shape influenced by birth intensity and trauma
            birth_intensity = birth_metrics.get('birth_intensity', 0.5)
            trauma_level = birth_metrics.get('trauma_level', 0.5)
            
            # Create path points (more trauma = more chaotic path)
            path_x = [5]  # Start at center top
            path_y = [9]
            
            # Add intermediate points with some randomness based on trauma
            steps = 8
            for i in range(1, steps):
                y_pos = 9 - i * (9 / steps)
                jitter = trauma_level * 2  # More trauma = more jitter
                x_jitter = 5 + np.random.uniform(-jitter, jitter)
                path_x.append(x_jitter)
                path_y.append(y_pos)
            
            # Add end point
            path_x.append(5)  # End at center bottom
            path_y.append(1)
            
            # Draw the path
            ax1.plot(path_x, path_y, 'r-', linewidth=2 + birth_intensity * 3, alpha=0.7)
            
            # Add mother influence if present
            if mother_influence:
                # Draw mother's resonance as a protective field
                theta = np.linspace(0, 2*PI, 100)
                influence_strength = mother_influence.get('overall_strength', 0.5)
                radius = 1.5 + influence_strength * 2
                
                # Create a protective bubble around part of the path
                bubble_x = 5 + radius * np.cos(theta)
                bubble_y = 5 + radius * np.sin(theta)
                ax1.plot(bubble_x, bubble_y, color='purple', alpha=0.4, linewidth=2, linestyle='--')
                ax1.fill(bubble_x, bubble_y, color='lavender', alpha=0.2)
                
                # Add label
                ax1.text(5, 5 + radius + 0.5, "Mother's Resonance", ha='center', fontsize=9, color='purple')
            
            # Mark the physical birth point
            ax1.scatter(5, 1, s=200, color='red', marker='*', zorder=10, label='Birth Point')
            
            # Add labels for planes
            ax1.text(1, 8, "Spirit Realm", fontsize=10, color='darkblue')
            ax1.text(1, 3.5, "Veil of Forgetting", fontsize=10, color='gray')
            ax1.text(1, 1, "Physical Realm", fontsize=10, color='saddlebrown')
            
            ax1.set_title("Birth Process Diagram")
            ax1.axis('off')
            
            # --- 2. Birth Metrics (Top Middle) ---
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Display birth metrics as a radar chart
            if birth_metrics:
                # Define metrics for radar
                metrics = ['Birth Intensity', 'Trauma Level', 'Memory Retention', 
                          'Cord Efficiency', 'Breath Synch', 'Acceptance']
                values = [birth_metrics.get('birth_intensity', 0.0),
                         birth_metrics.get('trauma_level', 0.0),
                         birth_metrics.get('memory_retention', 0.0),
                         birth_metrics.get('cord_efficiency', 0.0),
                         birth_metrics.get('breath_synch', 0.0),
                         birth_metrics.get('acceptance', 0.0)]
                
                # Convert to numpy arrays and ensure 0-1 range
                values = np.clip(values, 0, 1)
                
                # Create radar chart
                angles = np.linspace(0, 2*PI, len(metrics), endpoint=False).tolist()
                values = values.tolist()
                
                # Close the polygon
                values += values[:1]
                angles += angles[:1]
                metrics += metrics[:1]
                
                # Plot radar
                ax2.plot(angles, values, 'b-', linewidth=2, alpha=0.7)
                ax2.fill(angles, values, 'skyblue', alpha=0.2)
                
                # Set ticks and labels
                ax2.set_xticks(angles[:-1])
                ax2.set_xticklabels(metrics[:-1], fontsize=8)
                ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
                ax2.set_ylim(0, 1)
                
                # Add grid
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "No birth metrics available", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            
            ax2.set_title("Birth Metrics")
            
            # --- 3. Mother Influence (Top Right) ---
            ax3 = fig.add_subplot(gs[0, 2])
            
            if mother_influence:
                # Display mother influence metrics
                influence_metrics = [
                    ('Connection', mother_influence.get('connection_influence', 0.0)),
                    ('Trauma Red.', mother_influence.get('trauma_reduction', 0.0)),
                    ('Acceptance', mother_influence.get('acceptance_influence', 0.0)),
                    ('Cord Eff.', mother_influence.get('cord_efficiency_influence', 0.0)),
                    ('Memory', mother_influence.get('memory_influence', 0.0)),
                    ('Breath', mother_influence.get('breath_influence', 0.0))
                ]
                
                # Sort by influence strength
                influence_metrics.sort(key=lambda x: x[1], reverse=True)
                
                # Create horizontal bar chart
                names = [m[0] for m in influence_metrics]
                values = [m[1] for m in influence_metrics]
                
                y_pos = range(len(names))
                bars = ax3.barh(y_pos, values, align='center', color='purple', alpha=0.6)
                
                # Add labels
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(names, fontsize=9)
                ax3.set_xlim(0, 1)
                ax3.set_xlabel('Influence Strength', fontsize=9)
                
                # Add text labels
                for bar in bars:
                    width = bar.get_width()
                    ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                            va='center', fontsize=8)
                
                # Add overall strength
                overall = mother_influence.get('overall_strength', 0.0)
                ax3.text(0.5, -0.15, f"Overall Influence: {overall:.2f}", 
                        ha='center', transform=ax3.transAxes, fontsize=10,
                        bbox=dict(facecolor='lavender', alpha=0.3, boxstyle='round,pad=0.2'))
            else:
                ax3.text(0.5, 0.5, "No Mother\nInfluence", 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            
            ax3.set_title("Mother's Influence")
            
            # --- 4. Memory Veil Visualization (Bottom Left) ---
            ax4 = fig.add_subplot(gs[1, 0])
            
            # Create a visual representation of the memory veil
            memory_retention = birth_metrics.get('memory_retention', 0.5)
            
            # Generate a grid of "memory points"
            x = np.linspace(0, 10, 20)
            y = np.linspace(0, 10, 20)
            X, Y = np.meshgrid(x, y)
            
            # Generate a pattern for the veil effect
            veil_pattern = np.exp(-0.1 * ((X - 5)**2 + (Y - 5)**2))
            
            # Apply memory retention - higher retention = more points visible
            memory_mask = np.random.rand(*veil_pattern.shape) < (memory_retention**2)
            visible_pattern = veil_pattern * memory_mask
            
            # Create a custom blue gradient for memory intensity
            memory_cmap = LinearSegmentedColormap.from_list(
                "memory_cmap", ["#FFFFFF00", "#AADDFF", "#0066CC"]
            )
            
            # Plot as scatter with size and color based on pattern
            sizes = 100 * visible_pattern.flatten()
            colors = visible_pattern.flatten()
            
            ax4.scatter(X.flatten(), Y.flatten(), s=sizes, c=colors, cmap=memory_cmap, 
                       alpha=0.7, edgecolors='none')
            
            # Add gradient overlay for the veil effect
            veil_gradient = np.linspace(0, 1, 100)[:, np.newaxis]
            veil_extent = [0, 10, 0, 10]
            ax4.imshow(veil_gradient, extent=veil_extent, aspect='auto', 
                      origin='lower', cmap='gray', alpha=0.3)
            
            # Add memory retention indicator
            ax4.text(5, 9, f"Memory Retention: {memory_retention:.2f}", 
                    ha='center', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            ax4.set_xlim(0, 10)
            ax4.set_ylim(0, 10)
            ax4.set_title("Memory Veil Effect")
            ax4.axis('off')
            
            # --- 5. Life Cord Connection (Bottom Middle) ---
            ax5 = fig.add_subplot(gs[1, 1])
            
            # Visualize the life cord connection
            cord_efficiency = birth_metrics.get('cord_efficiency', 0.5)
            
            # Draw a stylized cord
            cord_x = np.linspace(0, 10, 100)
            
            # Cord shape affected by efficiency
            if cord_efficiency > 0.7:
                # Smooth cord for high efficiency
                cord_y = 5 + np.sin(cord_x * PI / 5) * 0.5
            elif cord_efficiency > 0.4:
                # Slightly wavy cord for medium efficiency
                cord_y = 5 + np.sin(cord_x * PI / 2) * 1.0
            else:
                # Very wavy/distorted cord for low efficiency
                cord_y = 5 + np.sin(cord_x * PI / 2) * 1.5 + np.sin(cord_x * PI) * 0.7
            
            # Plot cord with thickness based on efficiency
            ax5.plot(cord_x, cord_y, color='silver', linewidth=2 + cord_efficiency * 5, alpha=0.7)
            
            # Add glow effect for cord
            for i in range(5):
                alpha = 0.1 - i * 0.015
                width = (2 + cord_efficiency * 5) + i * 3
                ax5.plot(cord_x, cord_y, color='cyan', linewidth=width, alpha=alpha)
            
            # Add anchors at each end
            ax5.scatter(0, cord_y[0], s=300, color='blue', alpha=0.7, marker='o')
            ax5.scatter(10, cord_y[-1], s=300, color='brown', alpha=0.7, marker='o')
            
            # Add labels
            ax5.text(1, cord_y[10] + 2, "Soul Anchor", ha='center', va='center', fontsize=10)
            ax5.text(9, cord_y[-10] + 2, "Earth Anchor", ha='center', va='center', fontsize=10)
            
            # Add efficiency indicator
            ax5.text(5, 1, f"Cord Efficiency: {cord_efficiency:.2f}", 
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            ax5.set_xlim(0, 10)
            ax5.set_ylim(0, 10)
            ax5.set_title("Life Cord Connection")
            ax5.axis('off')
            
            # --- 6. First Breath Synchronization (Bottom Right) ---
            ax6 = fig.add_subplot(gs[1, 2])
            
            # Visualize breath synchronization
            breath_synch = birth_metrics.get('breath_synch', 0.5)
            
            # Generate time values
            t = np.linspace(0, 4*PI, 1000)
            
            # Generate earth frequency (constant)
            earth_freq = 1.0
            earth_wave = np.sin(earth_freq * t)
            
            # Generate soul frequency (varies with synchronization)
            if breath_synch > 0.8:
                # Nearly perfect sync
                soul_freq = earth_freq * 1.02
            elif breath_synch > 0.5:
                # Moderate sync
                soul_freq = earth_freq * 1.2
            else:
                # Poor sync
                soul_freq = earth_freq * 1.5
            
            soul_wave = np.sin(soul_freq * t)
            
            # Plot waves
            ax6.plot(t, earth_wave, 'g-', linewidth=2, label='Earth Rhythm')
            ax6.plot(t, soul_wave, 'b-', linewidth=2, label='Soul Rhythm')
            
            # Highlight synchronized regions
            sync_mask = np.abs(earth_wave - soul_wave) < 0.3
            sync_indices = np.where(sync_mask)[0]
            
            if len(sync_indices) > 0:
                for i in range(len(sync_indices)):
                    if i == 0 or sync_indices[i] > sync_indices[i-1] + 1:
                        # Start of a new sync region
                        start_idx = sync_indices[i]
                        # Find end of this sync region
                        end_idx = start_idx
                        while end_idx + 1 < len(sync_mask) and sync_mask[end_idx + 1]:
                            end_idx += 1
                        
                        # Highlight region if it's long enough
                        if end_idx - start_idx > 10:
                            ax6.axvspan(t[start_idx], t[end_idx], alpha=0.2, color='yellow')
            
            # Format plot
            ax6.set_xlim(0, max(t))
            ax6.set_ylim(-1.2, 1.2)
            ax6.legend(loc='upper right', fontsize=8)
            ax6.set_ylabel('Amplitude', fontsize=9)
            ax6.set_xlabel('Time', fontsize=9)
            
            # Add sync level
            ax6.text(0.5, -0.15, f"Breath Sync: {breath_synch:.2f}", 
                    ha='center', transform=ax6.transAxes, fontsize=10,
                    bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round,pad=0.2'))
            
            ax6.set_title("Breath Synchronization")
            
            # Main title
            fig.suptitle(f"Birth Process Analysis: {soul_name}", fontsize=16, y=0.98)
            
            self._save_or_show(fig, f"birth_process_{soul_id}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating birth process visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
def create_journey_dashboard(self, soul, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
    """Create a comprehensive dashboard view of the soul's journey progress.
    
    Args:
        soul: SoulSpark instance
        show: Whether to display the plot
        save: Whether to save the plot to file
        
    Returns:
        Matplotlib figure or None on error
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
    
    try:
        # Get soul attributes
        soul_name = getattr(soul, 'name', 'Unknown Soul')
        soul_id = getattr(soul, 'spark_id', 'unknown')
        
        # --- 1. Journey Timeline (Top Row, Spans all columns) ---
        ax1 = fig.add_subplot(gs[0, :])
        
        # Define journey stages in order
        stages = [
            {"name": "Guff Strengthening", "flag": "guff_strengthened", "ready_flag": "ready_for_guff"},
            {"name": "Sephiroth Journey", "flag": "sephiroth_journey_complete", "ready_flag": "ready_for_journey"},
            {"name": "Creator Entanglement", "flag": "creator_channel_id", "ready_flag": "ready_for_entanglement"},
            {"name": "Harmonic Strengthening", "flag": "harmonically_strengthened", "ready_flag": "ready_for_strengthening"},
            {"name": "Life Cord Formation", "flag": "cord_formation_complete", "ready_flag": "ready_for_life_cord"},
            {"name": "Earth Harmonization", "flag": "earth_harmonized", "ready_flag": "ready_for_earth"},
            {"name": "Identity Crystallization", "flag": "identity_crystallized", "ready_flag": "ready_for_identity"},
            {"name": "Birth Process", "flag": "incarnated", "ready_flag": "ready_for_birth"}
        ]
        
        # Extract completion and readiness data
        completion_states = []
        readiness_states = []
        for stage in stages:
            # Check if flag exists and is True
            completed = getattr(soul, stage["flag"], False)
            if stage["flag"] == "creator_channel_id":  # Special case for entanglement
                completed = getattr(soul, stage["flag"], "") != ""
                
            ready = getattr(soul, stage["ready_flag"], False)
            
            completion_states.append(completed)
            readiness_states.append(ready)
        
        # Plot horizontal timeline
        ax1.plot([0, len(stages)-1], [0, 0], 'k-', alpha=0.3, linewidth=2)
        
        # Plot completion markers
        for i, (completed, ready) in enumerate(zip(completion_states, readiness_states)):
            if completed:
                # Completed stage (filled circle)
                ax1.scatter(i, 0, s=300, marker='o', color='green', edgecolors='darkgreen', zorder=10)
                
                # Draw line to next point if not the last one
                if i < len(stages) - 1:
                    # Line to next stage, style depends if next is ready or not
                    next_color = 'darkgreen' if completion_states[i+1] else ('orange' if readiness_states[i+1] else 'gray')
                    ax1.plot([i, i+1], [0, 0], color=next_color, linewidth=3, alpha=0.7)
            elif ready:
                # Ready but not completed (half-filled circle)
                ax1.scatter(i, 0, s=200, marker='o', color='orange', edgecolors='darkorange', alpha=0.7, zorder=5)
            else:
                # Not ready, not completed (empty circle)
                ax1.scatter(i, 0, s=150, marker='o', facecolors='none', edgecolors='gray', alpha=0.5)
        
        # Add stage names with markers
        for i, stage in enumerate(stages):
            # Determine text position (alternating above/below)
            y_pos = 0.2 if i % 2 == 0 else -0.2
            align = 'bottom' if i % 2 == 0 else 'top'
            
            # Determine text color based on completion state
            if completion_states[i]:
                color = 'darkgreen'
                weight = 'bold'
            elif readiness_states[i]:
                color = 'darkorange'
                weight = 'normal'
            else:
                color = 'gray'
                weight = 'normal'
            
            # Add stage name
            ax1.text(i, y_pos, stage["name"], ha='center', va=align, color=color, 
                    fontweight=weight, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Format plot
        ax1.set_title("Soul Journey Progress", fontsize=14)
        ax1.set_xlim(-0.5, len(stages)-0.5)
        ax1.set_ylim(-0.8, 0.8)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        
        # Add progress indicator
        completed_count = sum(completion_states)
        next_stage = stages[completed_count]["name"] if completed_count < len(stages) else "All Complete"
        progress_text = f"Journey Progress: {completed_count}/{len(stages)} stages completed\nNext Stage: {next_stage}"
        ax1.text(len(stages)/2, -0.6, progress_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # --- 2. Soul Core Metrics (Middle Left) ---
        ax2 = fig.add_subplot(gs[1, 0], polar=True)
        
        # Create a radar chart for core metrics
        core_metrics = {
            "Energy": getattr(soul, 'energy', 0.0),
            "Stability": getattr(soul, 'stability', 0.0),
            "Coherence": getattr(soul, 'coherence', 0.0),
            "Resonance": getattr(soul, 'resonance', 0.0),
            "Response": getattr(soul, 'response_level', 0.0),
            "Harmony": getattr(soul, 'harmony', 0.0)
        }
        
        # Setup radar chart
        labels = list(core_metrics.keys())
        values = list(core_metrics.values())
        values = np.clip(values, 0, 1).tolist()
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax2.plot(angles, values, 'b-', linewidth=2, marker='o', markersize=5)
        ax2.fill(angles, values, 'cyan', alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.set_title("Soul Core Metrics", fontsize=12)
        
        # --- 3. Identity Formation (Middle Center) ---
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Get identity attributes
        identity_crystallized = getattr(soul, 'identity_crystallized', False)
        crystallization_level = getattr(soul, 'crystallization_level', 0.0)
        soul_color = getattr(soul, 'soul_color', 'grey')
        voice_frequency = getattr(soul, 'voice_frequency', 0.0)
        sephiroth_aspect = getattr(soul, 'sephiroth_aspect', 'Unknown')
        elemental_affinity = getattr(soul, 'elemental_affinity', 'Unknown')
        platonic_symbol = getattr(soul, 'platonic_symbol', 'Unknown')
        
        # Create a display layout
        
        # Draw a central circle representing the soul identity
        identity_circle = plt.Circle((0.5, 0.5), 0.2, color=soul_color, alpha=0.7)
        ax3.add_patch(identity_circle)
        
        # Add crystal effect if crystallized
        if identity_crystallized:
            # Add crystal-like rays emanating from center
            for angle in range(0, 360, 30):
                rad = angle * np.pi / 180
                length = 0.15 + 0.1 * crystallization_level
                ax3.plot([0.5, 0.5 + length * np.cos(rad)], 
                        [0.5, 0.5 + length * np.sin(rad)], 
                        color='white', alpha=0.7, linewidth=1.5)
        
        # Add crystallization level indicator
        ax3.barh([0.85], [crystallization_level], left=0.2, height=0.05, 
                color='gold', alpha=0.8)
        ax3.barh([0.85], [1.0-crystallization_level], left=0.2+crystallization_level, 
                height=0.05, color='gray', alpha=0.3)
        ax3.text(0.2, 0.85, "Crystallization", ha='right', va='center', fontsize=9)
        ax3.text(0.2 + crystallization_level/2, 0.85, f"{crystallization_level:.2f}", 
                ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add identity attributes as text
        identity_text = f"Name: {soul_name}\nColor: {soul_color}\nVoice: {voice_frequency:.1f} Hz"
        ax3.text(0.5, 0.25, identity_text, ha='center', va='center', fontsize=9)
        
        # Add affinities
        affinities_text = f"Sephirah: {sephiroth_aspect}\nElement: {elemental_affinity}\nSymbol: {platonic_symbol}"
        ax3.text(0.5, 0.12, affinities_text, ha='center', va='center', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Identity Formation", fontsize=12)
        
        # --- 4. Earth Connection (Middle Right) ---
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Get earth connection attributes
        earth_harmonized = getattr(soul, 'earth_harmonized', False)
        earth_resonance = getattr(soul, 'earth_resonance', 0.0)
        gaia_connection = getattr(soul, 'gaia_connection', 0.0)
        planetary_resonance = getattr(soul, 'planetary_resonance', 0.0)
        elemental_alignment = getattr(soul, 'elemental_alignment', 0.0)
        cycle_synchronization = getattr(soul, 'cycle_synchronization', 0.0)
        
        # Create earth connection visualization
        earth_metrics = {
            'Earth Resonance': earth_resonance,
            'Gaia Connection': gaia_connection,
            'Planetary Res.': planetary_resonance,
            'Elemental Align.': elemental_alignment,
            'Cycle Synch.': cycle_synchronization
        }
        
        # Sort by value
        sorted_metrics = sorted(earth_metrics.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_metrics]
        values = [item[1] for item in sorted_metrics]
        
        # Create horizontal bars
        y_pos = np.arange(len(labels))
        bars = ax4.barh(y_pos, values, align='center', 
                       color='forestgreen', alpha=0.7)
        
        # Add labels
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=9)
        ax4.set_xlim(0, 1)
        ax4.set_xlabel('Connection Strength', fontsize=9)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                    va='center', fontsize=8)
        
        # Add harmonization indicator
        if earth_harmonized:
            status_text = "Status: HARMONIZED"
            status_color = 'darkgreen'
        else:
            status_text = "Status: NOT HARMONIZED"
            status_color = 'gray'
            
        ax4.text(0.5, -0.15, status_text, ha='center', transform=ax4.transAxes, 
                fontsize=10, fontweight='bold', color=status_color)
        
        ax4.set_title("Earth Connection", fontsize=12)
        
        # --- 5. Life Cord Status (Bottom Left) ---
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Get life cord attributes
        cord_formed = getattr(soul, 'cord_formation_complete', False)
        cord_integrity = getattr(soul, 'cord_integrity', 0.0)
        field_integration = getattr(soul, 'field_integration', 0.0)
        physical_integration = getattr(soul, 'physical_integration', 0.0)
        
        # Create a visual representation of the cord
        if cord_formed:
            # Draw a stylized cord
            t = np.linspace(0, np.pi, 100)
            x = 0.5 + 0.4 * np.sin(t)
            y = np.linspace(0.1, 0.9, 100)
            
            # Cord stability affected by integrity
            jitter = max(0.01, 0.1 - cord_integrity * 0.1)
            x += np.random.normal(0, jitter, 100)
            
            # Plot with thickness based on integrity
            ax5.plot(x, y, color='silver', linewidth=5 * cord_integrity + 1, alpha=0.7)
            
            # Add glow effect
            for i in range(3):
                alpha = 0.1 - i * 0.03
                width = (5 * cord_integrity + 1) + i * 2
                ax5.plot(x, y, color='cyan', linewidth=width, alpha=alpha)
            
            # Add anchors at each end
            ax5.scatter(x[0], y[0], s=200, color='brown', alpha=0.7, marker='o')
            ax5.scatter(x[-1], y[-1], s=200, color='blue', alpha=0.7, marker='o')
            
            # Add labels
            ax5.text(x[0] - 0.1, y[0] - 0.05, "Earth", ha='right', va='top', fontsize=9)
            ax5.text(x[-1] + 0.1, y[-1] + 0.05, "Soul", ha='left', va='bottom', fontsize=9)
            
            # Add metrics as text
            metrics_text = (
                f"Cord Integrity: {cord_integrity:.2f}\n"
                f"Field Integration: {field_integration:.2f}\n"
                f"Physical Integration: {physical_integration:.2f}"
            )
            ax5.text(0.5, 0.02, metrics_text, ha='center', va='bottom', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        else:
            ax5.text(0.5, 0.5, "Life Cord Not Formed", ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
        
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title("Life Cord Status", fontsize=12)
        
        # --- 6. Creator Connection (Bottom Center) ---
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Get creator connection attributes
        creator_channel_id = getattr(soul, 'creator_channel_id', '')
        creator_connection_strength = getattr(soul, 'creator_connection_strength', 0.0)
        creator_alignment = getattr(soul, 'creator_alignment', 0.0)
        creator_aspects = getattr(soul, 'creator_aspects', {})
        
        if creator_channel_id:
            # Draw a stylized connection visualization
            
            # Draw a golden circle for creator
            creator_circle = plt.Circle((0.5, 0.85), 0.1, color='gold', alpha=0.7)
            ax6.add_patch(creator_circle)
            
            # Draw a smaller circle for soul
            soul_circle = plt.Circle((0.5, 0.3), 0.07, color='blue', alpha=0.7)
            ax6.add_patch(soul_circle)
            
            # Draw connection beam with width based on strength
            beam_width = 0.02 + 0.08 * creator_connection_strength
            
            # Create beam coordinates
            beam_x = [0.5 - beam_width/2, 0.5 + beam_width/2, 
                     0.5 + beam_width/2, 0.5 - beam_width/2]
            beam_y = [0.3 + 0.07, 0.3 + 0.07, 0.85 - 0.1, 0.85 - 0.1]
            
            # Plot beam with golden gradient
            beam_poly = plt.Polygon(np.column_stack([beam_x, beam_y]), 
                                   color='gold', alpha=0.4)
            ax6.add_patch(beam_poly)
            
            # Add aspects if present
            if creator_aspects:
                aspect_items = list(creator_aspects.items())
                max_aspects = min(4, len(aspect_items))
                
                for i in range(max_aspects):
                    aspect, value = aspect_items[i]
                    angle = np.pi / 4 + i * np.pi / 2
                    r = 0.15 + 0.05 * value
                    x = 0.5 + r * np.cos(angle)
                    y = 0.85 + r * np.sin(angle)
                    
                    # Size based on value
                    size = 30 + 100 * value
                    
                    ax6.scatter(x, y, s=size, color='orange', alpha=0.6)
                    ax6.text(x, y + 0.05, aspect, ha='center', va='bottom', fontsize=8)
            
            # Add metrics as text
            metrics_text = (
                f"Connection Strength: {creator_connection_strength:.2f}\n"
                f"Creator Alignment: {creator_alignment:.2f}\n"
                f"Channel ID: {creator_channel_id[:8]}..."
            )
            ax6.text(0.5, 0.15, metrics_text, ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        else:
            ax6.text(0.5, 0.5, "Creator Connection Not Established", 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_title("Creator Connection", fontsize=12)
        
        # --- 7. Incarnation Status (Bottom Right) ---
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Get incarnation attributes
        incarnated = getattr(soul, 'incarnated', False)
        ready_for_birth = getattr(soul, 'ready_for_birth', False)
        birth_data = getattr(soul, 'birth_data', {})
        
        # Create a visual representation of incarnation status
        if incarnated:
            # Draw a physical body silhouette
            body_x = [0.4, 0.35, 0.3, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.7, 0.65, 0.6]
            body_y = [0.8, 0.75, 0.65, 0.4, 0.25, 0.2, 0.15, 0.2, 0.25, 0.4, 0.65, 0.75, 0.8]
            
            # Plot body outline
            ax7.fill(body_x, body_y, color='tan', alpha=0.6)
            
            # Plot head
            head_circle = plt.Circle((0.5, 0.85), 0.1, color='tan', alpha=0.6)
            ax7.add_patch(head_circle)
            
            # Add a glowing aura
            for i in range(5):
                alpha = 0.08 - i * 0.01
                width = 2 + i * 2
                ax7.plot(body_x + [0.4], body_y + [0.8], color='cyan', 
                        linewidth=width, alpha=alpha)
            
            # Extract birth metrics
            birth_intensity = birth_data.get('birth_intensity', 0.0)
            trauma_level = birth_data.get('trauma_level', 0.0)
            memory_retention = birth_data.get('memory_retention', 0.0)
            
            # Add birth data metrics as text
            metrics_text = (
                f"Birth Intensity: {birth_intensity:.2f}\n"
                f"Trauma Level: {trauma_level:.2f}\n"
                f"Memory Retention: {memory_retention:.2f}"
            )
            
            ax7.text(0.5, 0.05, metrics_text, ha='center', va='bottom', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Add incarnated status
            status_text = "Status: INCARNATED"
            ax7.text(0.5, 0.95, status_text, ha='center', va='top', fontsize=12, 
                    color='darkgreen', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        elif ready_for_birth:
            # Soul is ready but not incarnated
            # Draw a soul ready to incarnate
            
            # Draw a physical body outline (ghosted)
            body_x = [0.4, 0.35, 0.3, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.7, 0.65, 0.6]
            body_y = [0.8, 0.75, 0.65, 0.4, 0.25, 0.2, 0.15, 0.2, 0.25, 0.4, 0.65, 0.75, 0.8]
            
            # Plot body outline
            ax7.plot(body_x + [0.4], body_y + [0.8], color='gray', alpha=0.4, linewidth=1.5)
            
            # Draw a soul sphere above body
            soul_circle = plt.Circle((0.5, 1.05), 0.12, color='blue', alpha=0.6)
            ax7.add_patch(soul_circle)
            
            # Add a glowing aura for soul
            for i in range(5):
                alpha = 0.1 - i * 0.015
                radius = 0.12 + i * 0.03
                soul_aura = plt.Circle((0.5, 1.05), radius, color='cyan', alpha=alpha, fill=False)
                ax7.add_patch(soul_aura)
            
            # Add descent arrow
            ax7.arrow(0.5, 0.9, 0, -0.25, head_width=0.07, head_length=0.07, 
                     fc='blue', ec='blue', alpha=0.6)
            
            # Add ready status
            status_text = "Status: READY FOR BIRTH"
            ax7.text(0.5, 0.05, status_text, ha='center', va='bottom', fontsize=12, 
                    color='darkorange', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
        else:
            # Not ready, not incarnated
            # Show message only
            ax7.text(0.5, 0.5, "Not Ready for Birth", ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=14)
        
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1.2)
        ax7.set_xticks([])
        ax7.set_yticks([])
        ax7.set_title("Incarnation Status", fontsize=12)
        
        # Main title
        fig.suptitle(f"Soul Journey Dashboard: {soul_name}", fontsize=16, y=0.98)
        
        self._save_or_show(fig, f"soul_journey_dashboard_{soul_id}.png", show, save)
        return fig
    except Exception as e:
        logger.error(f"Error generating journey dashboard: {e}", exc_info=True)
        if 'fig' in locals() and fig:
            plt.close(fig)
        return None