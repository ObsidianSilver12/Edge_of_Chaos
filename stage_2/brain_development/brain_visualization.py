"""
brain_visualization.py - Module for visualizing brain structure and soul connection.

This module provides visualization tools for the brain seed, hemisphere structure,
brain regions, and soul connections.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import cm
from matplotlib.font_manager import FontProperties
import logging
from typing import Dict, List, Tuple, Optional, Any
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainVisualization')

# Color maps for different visualization aspects
COLOR_MAPS = {
    'energy': cm.get_cmap('viridis'),
    'frequency': cm.get_cmap('plasma'),
    'resonance': cm.get_cmap('inferno'),
    'connection': cm.get_cmap('Blues'),
    'soul': cm.get_cmap('magma')
}

# Define sephiroth aspects with their frequencies
SEPHIROTH_ASPECTS = {
    'kether': {'frequency': 1000.0},    
    'chokmah': {'frequency': 900.0},
    'binah': {'frequency': 800.0},
    'chesed': {'frequency': 700.0},
    'geburah': {'frequency': 600.0},
    'tiphareth': {'frequency': 500.0},
    'netzach': {'frequency': 400.0},
    'hod': {'frequency': 300.0},
    'yesod': {'frequency': 200.0},
    'malkuth': {'frequency': 100.0}
}

def visualize_brain_seed(brain_seed, save_path=None, show=True):
    """
    Generate a basic visualization of the brain seed structure.
    
    Parameters:
        brain_seed: The brain seed object to visualize
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating brain seed visualization")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize seed core
    if hasattr(brain_seed, 'seed_core') and brain_seed.seed_core:
        # Extract seed core properties
        position = brain_seed.seed_core.get('position', np.array([0, 0, 0]))
        radius = brain_seed.seed_core.get('radius', 0.1)
        energy_density = brain_seed.seed_core.get('energy_density', 50)
        
        color_intensity = min(1.0, energy_density / 200)
        core_color = cm.get_cmap('Blues')(color_intensity)
        core_color = cm.get_cmap('viridis')(color_intensity)
        
        # Plot seed core as a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = position[0] + radius * np.cos(u) * np.sin(v)
        y = position[1] + radius * np.sin(u) * np.sin(v)
        z = position[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=core_color, alpha=0.7)
    
    # Visualize energy generators
    if hasattr(brain_seed, 'energy_generators') and brain_seed.energy_generators:
        # Define color mapping for generator types
        type_colors = {
            'resonant_field': 'blue',
            'vortex_node': 'green',
            'scalar_amplifier': 'red',
            'harmonic_oscillator': 'purple',
            'quantum_field_stabilizer': 'orange'
        }
        
        for i, generator in enumerate(brain_seed.energy_generators):
            position = generator.get('position', np.array([0, 0, 0]))
            output = generator.get('output', 10)
            g_type = generator.get('type', 'unknown')
            
            # Size based on output
            size = output / 10
            
            # Color based on type
            color = type_colors.get(g_type, 'gray')
            
            # Plot generator as a point
            ax.scatter(position[0], position[1], position[2], color=color, s=size*100, 
                      label=f"{g_type}" if i == 0 else "", alpha=0.8)
            
            # Add energy lines from generator to core
            if hasattr(brain_seed, 'seed_core') and brain_seed.seed_core:
                core_pos = brain_seed.seed_core.get('position', np.array([0, 0, 0]))
                ax.plot([position[0], core_pos[0]], 
                        [position[1], core_pos[1]], 
                        [position[2], core_pos[2]], 
                        color=color, alpha=0.3, linestyle='--')
    
    # Set up axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Brain Seed Structure')
    
    # Add legend
    if hasattr(brain_seed, 'energy_generators') and brain_seed.energy_generators:
        types = set(g.get('type', 'unknown') for g in brain_seed.energy_generators)
        type_patches = []
        for g_type in types:
            color = type_colors.get(g_type, 'gray')
            type_patches.append(plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, label=g_type))
        
        ax.legend(handles=type_patches, loc='upper right')
    
    # Add metrics text
    if hasattr(brain_seed, 'get_metrics'):
        metrics = brain_seed.get_metrics()
        metrics_text = (
            f"Energy: {metrics.get('energy_level', 0):.1f}/{metrics.get('energy_capacity', 0):.1f}\n"
            f"Complexity: {metrics.get('complexity', 0):.1f}\n"
            f"Progress: {metrics.get('formation_progress', 0)*100:.1f}%\n"
            f"Integrity: {metrics.get('structural_integrity', 0)*100:.1f}%\n"
            f"Stability: {metrics.get('stability', 0)*100:.1f}%"
        )
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_soul_distribution(brain_seed, save_path=None, show=True):
    """
    Visualize the distribution of soul aspects throughout the brain.
    
    Parameters:
        brain_seed: The brain seed object to visualize
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating soul distribution visualization")
    
    # Check if soul distribution is present
    if not hasattr(brain_seed, 'soul_aspect_distribution'):
        logger.warning("Soul aspects not distributed in brain, cannot visualize")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get distribution data
    distribution = brain_seed.soul_aspect_distribution
    
    # Region positions (approximate based on real brain)
    region_positions = {
        'frontal': (-2, 2.5),
        'parietal': (0, 2.5),
        'temporal': (-2.5, 0),
        'occipital': (2.5, 0),
        'limbic': (0, 0.5),
        'cerebellum': (0, -2),
        'brainstem': (0, -3)
    }
    
    # Region sizes
    region_sizes = {
        'frontal': (3, 2),
        'parietal': (3, 2),
        'temporal': (2, 2.5),
        'occipital': (2, 2.5),
        'limbic': (1.5, 1.5),
        'cerebellum': (3, 1.5),
        'brainstem': (1, 2)
    }
    
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=9, height=8, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2, alpha=0.5)
    ax.add_patch(brain_outline)
    
    # Draw hemispheres dividing line
    ax.plot([0, 0], [-4, 4], 'k--', linewidth=1, alpha=0.5)
    
    # Draw regions with soul aspects
    for region_name, mappings in distribution.get('region_mappings', {}).items():
        if not mappings:
            continue
            
        # Get region position and size
        pos = region_positions.get(region_name, (0, 0))
        size = region_sizes.get(region_name, (1, 1))
        
        # Create base region shape
        if region_name == 'limbic':
            # Limbic as a circle
            base_patch = Circle(pos, size[0]/2, fill=True, alpha=0.1,
                              edgecolor='black', linewidth=1, facecolor='gray')
        elif region_name in ['cerebellum', 'brainstem']:
            # Cerebellum and brainstem as rectangles
            base_patch = Rectangle((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                                 fill=True, alpha=0.1, edgecolor='black', linewidth=1,
                                 facecolor='gray')
        else:
            # Others as fancy rounded rectangles
            base_patch = FancyBboxPatch((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                                      boxstyle="round,pad=0.3",
                                      fill=True, alpha=0.1, edgecolor='black', linewidth=1,
                                      facecolor='gray')
        
        # Add base patch
        ax.add_patch(base_patch)
        
        # Add region label
        ax.text(pos[0], pos[1] + 0.3, region_name.title(), fontsize=10, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Calculate total influence
        total_influence = sum(m.get('influence', 0) for m in mappings)
        
        # Draw pie chart of aspects
        aspect_colors = {
            'kether': '#FFFFFF',    # White
            'chokmah': '#C8C8FF',   # Light blue
            'binah': '#7F00FF',     # Purple
            'chesed': '#0000FF',    # Blue
            'geburah': '#FF0000',   # Red
            'tiphareth': '#FFD700',  # Gold
            'netzach': '#00FF00',   # Green
            'hod': '#FFA500',       # Orange
            'yesod': '#8200FF',     # Violet
            'malkuth': '#8B4513'    # Brown
        }
        
        # Create wedges
        wedges = []
        colors = []
        labels = []
        sizes = []
        
        for mapping in mappings:
            aspect = mapping.get('aspect', 'unknown')
            influence = mapping.get('influence', 0)
            
            # Normalized influence for pie
            if total_influence > 0:
                size = influence / total_influence
            else:
                size = 0
                
            # Add to lists
            wedges.append(aspect)
            colors.append(aspect_colors.get(aspect, '#808080'))
            labels.append(f"{aspect}\n{influence:.2f}")
            sizes.append(size)
        
        # Draw pie chart
        if sizes:
            # Position slightly offset from center
            pie_pos = (pos[0], pos[1] - 0.5)
            pie_size = min(size[0], size[1]) * 0.6
            
            wedges, texts = ax.pie(sizes, colors=colors, startangle=90, radius=pie_size,
                                 center=pie_pos, wedgeprops=dict(width=pie_size*0.5, alpha=0.7))
            
            # Add influence text
            ax.text(pos[0], pos[1] - 0.1, f"Aspects: {len(mappings)}", 
                   fontsize=8, ha='center', va='center')
        
        # Draw some soul aspect pockets
        if 'soul_field' in distribution and 'regions' in distribution['soul_field']:
            field = distribution['soul_field']['regions'].get(region_name, {})
            
            # Draw pockets
            for pocket in field.get('pockets', []):
                pocket_pos = pocket.get('position', np.array([0, 0, 0]))
                aspect = pocket.get('aspect', 'unknown')
                intensity = pocket.get('intensity', 0.5)
                
                # Map to 2D position
                pocket_2d = (
                    pos[0] + pocket_pos[0] * size[0] * 0.3,
                    pos[1] + pocket_pos[1] * size[1] * 0.3
                )
                
                # Color from aspect
                color = aspect_colors.get(aspect, '#808080')
                
                # Size and alpha based on intensity
                pocket_size = 20 + 50 * intensity
                alpha = 0.3 + 0.7 * intensity
                
                # Plot pocket
                ax.scatter(pocket_2d[0], pocket_2d[1], s=pocket_size, color=color, 
                          alpha=alpha, edgecolor='white', linewidth=0.5)
    
    # Draw hemisphere mappings
    for hemi, mappings in distribution.get('hemisphere_mappings', {}).items():
        if not mappings:
            continue
            
        # Position for hemisphere label
        pos = (-3, 1.5) if hemi == 'left' else (3, 1.5)
        
        # Add hemisphere label
        ax.text(pos[0], pos[1], f"{hemi.title()} Hemisphere", fontsize=12, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # List aspects
        aspect_text = "Soul Aspects:\n" + "\n".join(f"- {m.get('aspect', 'unknown')}" for m in mappings)
        
        # Add aspect list
        ax.text(pos[0], pos[1] - 1, aspect_text, fontsize=9, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5))
        
        # Draw aspect fields in hemisphere
        if 'soul_field' in distribution and 'hemispheres' in distribution['soul_field']:
            field = distribution['soul_field']['hemispheres'].get(hemi, {})
            
            # For each aspect, draw some representative points
            for aspect_name, intensity in field.get('aspect_intensities', {}).items():
                # Get aspect color
                color = aspect_colors.get(aspect_name, '#808080')
                
                # Draw points
                n_points = int(10 * intensity)
                
                for _ in range(n_points):
                    # Random position in hemisphere
                    x = np.random.random() * 3 * (-1 if hemi == 'left' else 1)
                    y = (np.random.random() - 0.5) * 5
                    
                    # Skip if outside the ellipse
                    if (x/4.5)**2 + (y/4)**2 > 1:
                        continue
                        
                    # Size and alpha
                    size = 30 + 70 * np.random.random() * intensity
                    alpha = 0.2 + 0.7 * np.random.random() * intensity
                    
                    # Plot point
                    ax.scatter(x, y, s=size, color=color, alpha=alpha, edgecolor=None)
    
    # Create legend for sephiroth aspects
    aspect_patches = []
    for aspect, color in aspect_colors.items():
        if aspect in distribution.get('aspects', {}):
            aspect_patches.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color, markersize=10, 
                                          label=f"{aspect} - {distribution['aspects'][aspect].get('quality', '')}"))
    
    # Add legend
    if aspect_patches:
        ax.legend(handles=aspect_patches, title="Soul Aspects", 
                loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add integration level
    integration = distribution.get('integration_level', 0)
    integration_text = f"Soul Integration: {integration*100:.1f}%"
    integration_color = cm.get_cmap('RdYlGn')(integration)
    
    ax.text(0, -4.5, integration_text, fontsize=14, ha='center', va='center',
           bbox=dict(facecolor=integration_color, alpha=0.7))
    
    # Set up axes
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Soul Aspects Distribution in Brain', fontsize=16)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_combined_visualization(brain_seed, save_path=None, show=True):
    """
    Create a combined visualization showing brain structure, soul attachment,
    and soul distribution.
    
    Parameters:
        brain_seed: The brain seed object to visualize
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating combined visualization")
    
    # Check if we have enough data to visualize
    if not hasattr(brain_seed, 'region_structure') or not brain_seed.region_structure:
        logger.warning("Brain regions not developed, cannot create combined visualization")
        return None
    
    if not hasattr(brain_seed, 'soul_connection'):
        logger.warning("Soul not attached to brain, cannot create combined visualization")
        return None
    
    # Create figure with subfigures
    fig = plt.figure(figsize=(18, 12))
    
    # Brain regions subplot
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Region positions and sizes (simplified for combined view)
    region_positions = {
        'frontal': (-2, 2.5),
        'parietal': (0, 2.5),
        'temporal': (-2.5, 0),
        'occipital': (2.5, 0),
        'limbic': (0, 0.5),
        'cerebellum': (0, -2),
        'brainstem': (0, -3)
    }
    
    region_sizes = {
        'frontal': (3, 2),
        'parietal': (3, 2),
        'temporal': (2, 2.5),
        'occipital': (2, 2.5),
        'limbic': (1.5, 1.5),
        'cerebellum': (3, 1.5),
        'brainstem': (1, 2)
    }
    
    # Draw regions
    for region_name, region in brain_seed.region_structure.items():
        # Get position and size
        pos = region_positions.get(region_name, (0, 0))
        size = region_sizes.get(region_name, (1, 1))
        
        # Create region shape
        if region_name == 'limbic':
            patch = Circle(pos, size[0]/2, fill=True, alpha=0.5,
                         edgecolor='black', linewidth=1,
                         facecolor='lightblue')
        elif region_name in ['cerebellum', 'brainstem']:
            patch = Rectangle((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                            fill=True, alpha=0.5, edgecolor='black', linewidth=1,
                            facecolor='lightblue')
        else:
            patch = FancyBboxPatch((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                                 boxstyle="round,pad=0.3",
                                 fill=True, alpha=0.5, edgecolor='black', linewidth=1,
                                 facecolor='lightblue')
        
        # Add patch
        ax1.add_patch(patch)
        
        # Add label
        ax1.text(pos[0], pos[1], region_name.title(), fontsize=8, ha='center', va='center')
    
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=9, height=8, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2, alpha=0.5)
    ax1.add_patch(brain_outline)
    
    # Set up axes for brain regions
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Brain Structure', fontsize=14)
    
    # Soul attachment subplot
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Draw brain
    brain_outline = Ellipse((0, 0), width=8, height=6, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2)
    ax2.add_patch(brain_outline)
    
    # Draw soul
    soul_pos = (0, 5)
    soul_size = 50
    ax2.scatter(soul_pos[0], soul_pos[1], s=soul_size*5, color='purple', alpha=0.7, 
              edgecolor='white', linewidth=1)
    
    # Draw connection cord
    connection_strength = brain_seed.soul_connection.get('connection_strength', 0.5)
    ax2.plot([0, 0], [0, soul_pos[1]], color='gold', linewidth=2, alpha=connection_strength)
    
    # Draw attachment points
    for point in brain_seed.soul_connection.get('brain_attachment_points', [])[:5]:  # limit to 5 for clarity
        # Get position and properties
        point_pos = point.get('position', np.array([0, 0, 0]))
        strength = point.get('strength', 0.5)
        purpose = point.get('purpose', 'unknown')
        
        # Map to 2D
        pos_2d = (point_pos[0], point_pos[1] * 0.8)
        
        # Skip if outside visualization area
        if abs(pos_2d[0]) > 4 or abs(pos_2d[1]) > 3:
            continue
        
        # Purpose-based color
        color = 'gold' if purpose == 'primary_connection' else 'cyan'
        
        # Draw point
        ax2.scatter(pos_2d[0], pos_2d[1], s=strength*100, color=color, alpha=strength)
        
        # Connect to cord
        ax2.plot([pos_2d[0], 0], [pos_2d[1], pos_2d[1]], color=color, 
               alpha=0.5*strength, linestyle='--')
    
    # Set up axes for soul attachment
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-4, 6)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Soul Attachment', fontsize=14)
    
    # Soul distribution subplot
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Draw brain
    brain_outline = Ellipse((0, 0), width=9, height=8, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2, alpha=0.5)
    ax3.add_patch(brain_outline)
    
    # Define colors for aspects
    aspect_colors = {
        'kether': '#FFFFFF',    # White
        'chokmah': '#C8C8FF',   # Light blue
        'binah': '#7F00FF',     # Purple
        'chesed': '#0000FF',    # Blue
        'geburah': '#FF0000',   # Red
        'tiphareth': '#FFD700',  # Gold
        'netzach': '#00FF00',   # Green
        'hod': '#FFA500',       # Orange
        'yesod': '#8200FF',     # Violet
        'malkuth': '#8B4513'    # Brown
    }
    
    # Draw distributions if available
    if hasattr(brain_seed, 'soul_aspect_distribution'):
        distribution = brain_seed.soul_aspect_distribution
        
        # Draw aspects by region
        for region_name, mappings in distribution.get('region_mappings', {}).items():
            if not mappings:
                continue
                
            # Get region position
            pos = region_positions.get(region_name, (0, 0))
            
            # Get primary aspect (highest influence)
            primary_aspect = None
            max_influence = 0
            
            for mapping in mappings:
                aspect = mapping.get('aspect', 'unknown')
                influence = mapping.get('influence', 0)
                
                if influence > max_influence:
                    max_influence = influence
                    primary_aspect = aspect
            
            if primary_aspect:
                # Get color
                color = aspect_colors.get(primary_aspect, '#808080')
                
                # Draw region with this color
                if region_name == 'limbic':
                    patch = Circle(pos, region_sizes[region_name][0]/2, fill=True, alpha=0.7,
                                 edgecolor='black', linewidth=1, facecolor=color)
                elif region_name in ['cerebellum', 'brainstem']:
                    patch = Rectangle((pos[0]-region_sizes[region_name][0]/2, 
                                     pos[1]-region_sizes[region_name][1]/2), 
                                    region_sizes[region_name][0], region_sizes[region_name][1], 
                                    fill=True, alpha=0.7, edgecolor='black', linewidth=1,
                                    facecolor=color)
                else:
                    patch = FancyBboxPatch((pos[0]-region_sizes[region_name][0]/2, 
                                          pos[1]-region_sizes[region_name][1]/2), 
                                         region_sizes[region_name][0], region_sizes[region_name][1], 
                                         boxstyle="round,pad=0.3",
                                         fill=True, alpha=0.7, edgecolor='black', linewidth=1,
                                         facecolor=color)
                
                # Add patch
                ax3.add_patch(patch)
                
                # Add label
                ax3.text(pos[0], pos[1], f"{region_name}\n({primary_aspect})", 
                       fontsize=8, ha='center', va='center')
    
    # Set up axes for soul distribution
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Soul Aspect Distribution', fontsize=14)
    
    # Combined metrics subplot
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Turn off axes
    ax4.axis('off')
    
    # Get metrics for display
    metrics = {}
    
    # Brain formation metrics
    if hasattr(brain_seed, 'formation_progress'):
        metrics['Brain Formation'] = f"{brain_seed.formation_progress * 100:.1f}%"
    
    if hasattr(brain_seed, 'structural_integrity'):
        metrics['Structural Integrity'] = f"{brain_seed.structural_integrity * 100:.1f}%"
    
    if hasattr(brain_seed, 'stability'):
        metrics['Brain Stability'] = f"{brain_seed.stability * 100:.1f}%"
    
    # Connection metrics
    if hasattr(brain_seed, 'soul_connection'):
        metrics['Connection Strength'] = f"{brain_seed.soul_connection.get('connection_strength', 0) * 100:.1f}%"
        metrics['Resonance Coherence'] = f"{brain_seed.soul_connection.get('resonance_coherence', 0) * 100:.1f}%"
        metrics['Attachment Points'] = str(len(brain_seed.soul_connection.get('brain_attachment_points', [])))
    
    # Distribution metrics
    if hasattr(brain_seed, 'soul_aspect_distribution'):
        dist = brain_seed.soul_aspect_distribution
        metrics['Soul Integration'] = f"{dist.get('integration_level', 0) * 100:.1f}%"
        metrics['Soul Aspects'] = str(len(dist.get('aspects', {})))
        
        # Count total mappings
        total_mappings = sum(len(mappings) for mappings in dist.get('region_mappings', {}).values())
        metrics['Aspect Mappings'] = str(total_mappings)
    
    # Create metrics table
    cells = []
    for label, value in metrics.items():
        cells.append([label, value])
    
    # Add table
    table = ax4.table(cellText=cells, loc='center', cellLoc='left', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        
        # Color code some values
        if j == 1 and "%" in cell.get_text().get_text():
            value = float(cell.get_text().get_text().strip('%'))
            color = cm.get_cmap('RdYlGn')(value / 100)
            cell.set_facecolor(color)
    
    ax4.set_title('Combined Metrics', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    plt.suptitle(f"Brain-Soul System: {brain_seed.formation_progress * 100:.1f}% Formed", 
               fontsize=18, y=0.98)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_brain_development_timeline(brain_seed, save_path=None, show=True):
    """
    Visualize the development timeline of the brain seed.
    
    Parameters:
        brain_seed: The brain seed object to visualize
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating brain development timeline visualization")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define development stages
    stages = [
        {'name': 'Seed Core Formation', 'progress': 0.1, 'description': 'Formation of the energetic seed core'},
        {'name': 'Hemisphere Development', 'progress': 0.3, 'description': 'Development of left & right hemispheres'},
        {'name': 'Region Formation', 'progress': 0.6, 'description': 'Formation of specialized brain regions'},
        {'name': 'White Noise Application', 'progress': 0.7, 'description': 'Application of white noise to unstructured areas'},
        {'name': 'Soul Attachment Preparation', 'progress': 0.9, 'description': 'Preparation for soul attachment'},
        {'name': 'Soul Connection', 'progress': 0.95, 'description': 'Connection of soul via life cord'},
        {'name': 'Soul Distribution', 'progress': 1.0, 'description': 'Distribution of soul aspects through brain'}
    ]
    
    # Get current progress
    current_progress = getattr(brain_seed, 'formation_progress', 0)
    
    # Plot timeline
    y_pos = 0
    for i, stage in enumerate(stages):
        progress = stage['progress']
        name = stage['name']
        description = stage['description']
        
        # Determine status
        if current_progress >= progress:
            status = 'completed'
            color = 'green'
            alpha = 0.8
        elif current_progress >= progress - 0.1:
            status = 'in progress'
            color = 'orange'
            alpha = 0.6
        else:
            status = 'pending'
            color = 'gray'
            alpha = 0.3
        
        # Plot stage point
        ax.scatter(progress, y_pos, s=100, color=color, alpha=alpha, zorder=3)
        
        # Add stage name
        ax.text(progress, y_pos + 0.1, name, ha='center', va='bottom', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7), rotation=45)
        
        # Add description
        status_text = f"[{status.upper()}]" if status != 'pending' else ""
        ax.text(progress, y_pos - 0.1, f"{description} {status_text}", ha='center', va='top', 
               fontsize=8, alpha=0.7 if status != 'pending' else 0.5)
        
        # Connect stages with line
        if i > 0:
            prev_progress = stages[i-1]['progress']
            ax.plot([prev_progress, progress], [y_pos, y_pos], color=color, 
                   alpha=alpha, linestyle='-', linewidth=2, zorder=2)
    
    # Add current progress marker
    ax.axvline(x=current_progress, color='red', linestyle='--', alpha=0.7, zorder=1)
    ax.text(current_progress, -0.3, f"Current: {current_progress*100:.1f}%", 
           color='red', ha='center', va='top', fontsize=12,
           bbox=dict(facecolor='white', alpha=0.9))
    
    # Set up axes
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Progress')
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Brain Development Timeline', fontsize=16)
    
    # Add progress ticks
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Add soul-brain system metrics if available
    y_text_pos = -0.4
    if hasattr(brain_seed, 'get_metrics'):
        metrics = brain_seed.get_metrics()
        
        metrics_text = []
        
        if 'complexity' in metrics:
            metrics_text.append(f"Complexity: {metrics['complexity']:.1f}")
        
        if 'energy_level' in metrics and 'energy_capacity' in metrics:
            metrics_text.append(f"Energy: {metrics['energy_level']:.1f}/{metrics['energy_capacity']:.1f}")
        
        if 'structural_integrity' in metrics:
            metrics_text.append(f"Integrity: {metrics['structural_integrity']*100:.1f}%")
        
        if 'stability' in metrics:
            metrics_text.append(f"Stability: {metrics['stability']*100:.1f}%")
        
        if hasattr(brain_seed, 'soul_connection'):
            metrics_text.append(f"Soul Connection: {brain_seed.soul_connection.get('connection_strength', 0)*100:.1f}%")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_hemispheres(brain_seed, save_path=None, show=True):
    """
    Visualize the hemisphere structure of the brain.
    
    Parameters:
        brain_seed: The brain seed object to visualize
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating hemisphere visualization")
    
    # Check if hemispheres are developed
    if not hasattr(brain_seed, 'hemisphere_structure') or not (
        brain_seed.hemisphere_structure.get('left', {}).get('developed', False) and 
        brain_seed.hemisphere_structure.get('right', {}).get('developed', False)
    ):
        logger.warning("Hemispheres not developed, cannot visualize")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=8, height=6, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2)
    ax.add_patch(brain_outline)
    
    # Draw dividing line
    ax.plot([0, 0], [-4, 4], 'k--', linewidth=1, alpha=0.5)
    
    # Get hemisphere data
    left_hemi = brain_seed.hemisphere_structure['left']
    right_hemi = brain_seed.hemisphere_structure['right']
    
    # Calculate properties
    left_complexity = left_hemi.get('complexity', 5)
    right_complexity = right_hemi.get('complexity', 5)
    left_energy = left_hemi.get('energy', 50)
    right_energy = right_hemi.get('energy', 50)
    
    # Map to visual properties
    left_dots = int(20 + left_complexity * 5)
    right_dots = int(20 + right_complexity * 5)
    left_color = cm.get_cmap('Blues')(min(1.0, left_energy / 100))
    right_color = cm.get_cmap('Reds')(min(1.0, right_energy / 100))
    
    # Generate random points for left hemisphere
    for _ in range(left_dots):
        # Random position within the left hemisphere
        x = -np.random.random() * 3.5
        y = (np.random.random() - 0.5) * 5
        
        # Skip if outside the ellipse
        if (x/4)**2 + (y/3)**2 > 1:
            continue
            
        # Plot dot
        size = np.random.random() * 50 + 10
        alpha = np.random.random() * 0.8 + 0.2
        ax.scatter(x, y, s=size, color=left_color, alpha=alpha, edgecolor=None)
    
    # Generate random points for right hemisphere
    for _ in range(right_dots):
        # Random position within the right hemisphere
        x = np.random.random() * 3.5
        y = (np.random.random() - 0.5) * 5
        
        # Skip if outside the ellipse
        if (x/4)**2 + (y/3)**2 > 1:
            continue
            
        # Plot dot
        size = np.random.random() * 50 + 10
        alpha = np.random.random() * 0.8 + 0.2
        ax.scatter(x, y, s=size, color=right_color, alpha=alpha, edgecolor=None)
    
    # Draw energy channels
    left_channels = left_hemi.get('energy_channels', 3)
    right_channels = right_hemi.get('energy_channels', 3)
    
    # Draw left channels
    for i in range(left_channels):
        angle = (np.pi/2) - (np.pi * i / (left_channels * 2))
        length = 3 + np.random.random()
        x = -length * np.cos(angle)
        y = length * np.sin(angle)
        
        ax.plot([0, x], [0, y], color='blue', linewidth=2, alpha=0.6)
        ax.scatter(x, y, color='blue', s=50, alpha=0.8)
    
    # Draw right channels
    for i in range(right_channels):
        angle = (np.pi/2) - (np.pi * i / (right_channels * 2))
        length = 3 + np.random.random()
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        
        ax.plot([0, x], [0, y], color='red', linewidth=2, alpha=0.6)
        ax.scatter(x, y, color='red', s=50, alpha=0.8)
    
    # Add function labels
    left_functions = "\n".join(left_hemi.get('primary_function', 'analytical').split('_'))
    right_functions = "\n".join(right_hemi.get('primary_function', 'creative').split('_'))
    
    ax.text(-3, 0, left_functions, fontsize=14, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.7))
    ax.text(3, 0, right_functions, fontsize=14, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Set up axes
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Brain Hemisphere Structure', fontsize=16)
    
    # Add labels
    ax.text(-3, -3.5, f"Left Hemisphere\nComplexity: {left_complexity:.1f}\nEnergy: {left_energy:.1f}", 
           fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    ax.text(3, -3.5, f"Right Hemisphere\nComplexity: {right_complexity:.1f}\nEnergy: {right_energy:.1f}", 
           fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add balance indicator
    balance = 1.0 - abs(left_complexity - right_complexity) / (left_complexity + right_complexity)
    balance_text = f"Hemisphere Balance: {balance*100:.1f}%"
    balance_color = cm.get_cmap('RdYlGn')(balance)
    
    ax.text(0, 3.5, balance_text, fontsize=12, ha='center', va='center',
           bbox=dict(facecolor=balance_color, alpha=0.7))
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_brain_regions(brain_seed, save_path=None, show=True):
    """
    Visualize the brain regions and their connections.
    
    Parameters:
        brain_seed: The brain seed object to visualize
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating brain regions visualization")
    
    # Check if regions are developed
    if not hasattr(brain_seed, 'region_structure') or not brain_seed.region_structure:
        logger.warning("Brain regions not developed, cannot visualize")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Region positions (approximate based on real brain)
    region_positions = {
        'frontal': (-2, 2.5),
        'parietal': (0, 2.5),
        'temporal': (-2.5, 0),
        'occipital': (2.5, 0),
        'limbic': (0, 0.5),
        'cerebellum': (0, -2),
        'brainstem': (0, -3)
    }
    
    # Region sizes
    region_sizes = {
        'frontal': (3, 2),
        'parietal': (3, 2),
        'temporal': (2, 2.5),
        'occipital': (2, 2.5),
        'limbic': (1.5, 1.5),
        'cerebellum': (3, 1.5),
        'brainstem': (1, 2)
    }
    
    # Draw regions
    region_patches = {}
    region_centers = {}
    
    for region_name, region in brain_seed.region_structure.items():
        # Get position and size
        pos = region_positions.get(region_name, (0, 0))
        size = region_sizes.get(region_name, (1, 1))
        
        # Get region properties
        complexity = region.get('complexity', 5)
        development = region.get('development_level', 0.5)
        energy = region.get('energy', 50)
        
        # Calculate visual properties
        color_intensity = min(1.0, energy / 100)
        alpha = 0.3 + (0.7 * development)
        linewidth = 1 + complexity / 5
        
        # Create region shape
        if region_name == 'limbic':
            # Limbic as a circle
            patch = Circle(pos, size[0]/2, fill=True, alpha=alpha,
                         edgecolor='black', linewidth=linewidth,
                         facecolor=cm.get_cmap('viridis')(color_intensity))
        elif region_name in ['cerebellum', 'brainstem']:
            # Cerebellum and brainstem as rectangles
            patch = Rectangle((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                            fill=True, alpha=alpha, edgecolor='black', linewidth=linewidth,
                            facecolor=cm.get_cmap('viridis')(color_intensity))
        else:
            # Others as fancy rounded rectangles
            patch = FancyBboxPatch((pos[0]-size[0]/2, pos[1]-size[1]/2), size[0], size[1], 
                                 boxstyle=f"round,pad=0.3,rounding_size={0.2+0.3*development}",
                                 fill=True, alpha=alpha, edgecolor='black', linewidth=linewidth,
                                 facecolor=cm.get_cmap('viridis')(color_intensity))
        
        # Add patch
        ax.add_patch(patch)
        
        # Store for connections
        region_patches[region_name] = patch
        region_centers[region_name] = pos
        
        # Add label
        ax.text(pos[0], pos[1], region_name.title(), fontsize=10, ha='center', va='center')
        
        # Add complexity indicator
        ax.text(pos[0], pos[1] - 0.3, f"C: {complexity:.1f}", fontsize=8, ha='center', va='center')
    
    # Draw connections
    for region_name, region in brain_seed.region_structure.items():
        if 'connections' not in region:
            continue
            
        # Get source position
        source_pos = region_centers.get(region_name, (0, 0))
        
        for connection in region.get('connections', []):
            target_name = connection.get('target', '')
            if target_name not in region_centers:
                continue
                
            # Get target position
            target_pos = region_centers.get(target_name, (0, 0))
            
            # Get connection properties
            strength = connection.get('strength', 0.5)
            pathways = connection.get('pathways', 1)
            
            # Draw connection lines
            for i in range(pathways):
                # Add slight offset for multiple pathways
                if pathways > 1:
                    offset = (i / (pathways - 1) - 0.5) * 0.2
                    modified_source = (source_pos[0] + offset, source_pos[1] + offset)
                    modified_target = (target_pos[0] + offset, target_pos[1] + offset)
                else:
                    modified_source = source_pos
                    modified_target = target_pos
                
                # Draw line with properties based on strength
                ax.plot([modified_source[0], modified_target[0]], 
                        [modified_source[1], modified_target[1]], 
                        color='blue', alpha=0.2 + 0.7 * strength, 
                        linewidth=0.5 + 2 * strength)
    
    # Set up axes
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Brain Regions and Connections', fontsize=16)
    
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=9, height=8, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2, alpha=0.5)
    ax.add_patch(brain_outline)
    
    # Add formation progress indicator
    if hasattr(brain_seed, 'formation_progress'):
        progress = brain_seed.formation_progress
        progress_text = f"Brain Formation: {progress*100:.1f}%"
        progress_color = cm.get_cmap('RdYlGn')(progress)
        
        ax.text(0, -4.5, progress_text, fontsize=12, ha='center', va='center',
               bbox=dict(facecolor=progress_color, alpha=0.7))
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_soul_attachment(brain_seed, life_cord=None, save_path=None, show=True):
    """
    Visualize the soul attachment to the brain.
    
    Parameters:
        brain_seed: The brain seed object to visualize
        life_cord (optional): The life cord connecting to the soul
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating soul attachment visualization")
    
    # Check if soul connection is present
    if not hasattr(brain_seed, 'soul_connection'):
        logger.warning("Soul not attached to brain, cannot visualize connection")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=8, height=6, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2)
    ax.add_patch(brain_outline)
    
    # Draw soul representation
    soul_pos = (0, 5)
    soul_size = 3
    
    # Gradient representing soul
    n_points = 300
    soul_points_x = np.random.normal(soul_pos[0], 0.8, n_points)
    soul_points_y = np.random.normal(soul_pos[1], 0.8, n_points)
    soul_colors = np.zeros((n_points, 4))
    
    for i in range(n_points):
        # Distance from center
        dist = np.sqrt((soul_points_x[i] - soul_pos[0])**2 + (soul_points_y[i] - soul_pos[1])**2)
        
        # Color based on distance
        color_val = max(0, 1 - dist/soul_size)
        
        # Get a color from a colormap
        cmap_color = cm.get_cmap('magma')(color_val)
        
        # Store color
        soul_colors[i] = cmap_color
    
    # Plot soul points
    ax.scatter(soul_points_x, soul_points_y, c=soul_colors, s=50, edgecolor=None)
    
    # Draw life cord
    life_cord_points_x = []
    life_cord_points_y = []
    
    # Create points along curve from soul to brain
    steps = 50
    for i in range(steps):
        t = i / (steps - 1)
        
        # Parameterized curve (bezier-like)
        x = 0  # Centerline
        y = soul_pos[1] - t * (soul_pos[1])
        
        # Add some oscillation
        period = 10
        amplitude = 0.4 * (1 - t)  # Decreases as it approaches the brain
        x += amplitude * np.sin(period * t * np.pi)
        
        life_cord_points_x.append(x)
        life_cord_points_y.append(y)
    
    # Plot life cord
    connection_strength = brain_seed.soul_connection.get('connection_strength', 0.5)
    ax.plot(life_cord_points_x, life_cord_points_y, 'white', linewidth=10, alpha=0.1, zorder=1)
    ax.plot(life_cord_points_x, life_cord_points_y, 'gold', linewidth=5, alpha=0.2, zorder=2)
    ax.plot(life_cord_points_x, life_cord_points_y, 'white', linewidth=2, alpha=connection_strength, zorder=3)
    
    # Draw attachment points
    for point in brain_seed.soul_connection.get('brain_attachment_points', []):
        # Get point properties
        point_pos = point.get('position', np.array([0, 0, 0]))
        strength = point.get('strength', 0.5)
        purpose = point.get('purpose', 'unknown')
        
        # Map to 2D position (simple mapping)
        pos_2d = (point_pos[0], point_pos[1] * 0.8)
        
        # Skip if outside visualization area
        if abs(pos_2d[0]) > 4 or abs(pos_2d[1]) > 3:
            continue
        
        # Color based on purpose
        purpose_colors = {
            'primary_connection': 'gold',
            'region_connection': 'cyan',
            'hemisphere_connection': 'magenta',
            'auxiliary': 'silver'
        }
        color = purpose_colors.get(purpose, 'white')
        
        # Size based on strength
        size = 50 + 100 * strength
        
        # Plot point
        ax.scatter(pos_2d[0], pos_2d[1], s=size, color=color, alpha=strength, 
                  edgecolor='white', linewidth=1, zorder=4)
        
        # Draw connection to life cord
        # Find closest point on life cord
        distances = [(x - pos_2d[0])**2 + (y - pos_2d[1])**2 
                    for x, y in zip(life_cord_points_x, life_cord_points_y)]
        closest_idx = np.argmin(distances)
        closest_point = (life_cord_points_x[closest_idx], life_cord_points_y[closest_idx])
        
        # Draw connection line
        ax.plot([pos_2d[0], closest_point[0]], [pos_2d[1], closest_point[1]], 
               color=color, linewidth=1, alpha=0.7*strength, zorder=3, 
               linestyle='--' if purpose != 'primary_connection' else '-')
    
    # Draw resonance field if present
    if 'resonance_field' in brain_seed.soul_connection:
        resonance_field = brain_seed.soul_connection['resonance_field']
        coherence = resonance_field.get('coherence', 0.5)
        intensity = resonance_field.get('intensity', 0.5)
        
        # Draw field indication
        field_radius = 4.5
        field_circle = Circle((0, 0), field_radius, fill=True, 
                            facecolor='blue', alpha=0.05 + 0.1 * intensity, 
                            edgecolor=None, zorder=0)
        ax.add_patch(field_circle)
        
        # Draw harmonic nodes
        for node in resonance_field.get('harmonic_nodes', []):
            # Calculate position based on frequencies
            brain_freq = node.get('brain_frequency', 10)
            soul_freq = node.get('soul_frequency', 10)
            band = node.get('band', 'alpha')
            resonance = node.get('resonance', 0.5)
            
            # Map frequency to position
            # Higher brain_freq = closer to brain
            # Higher soul_freq = closer to soul
            rel_pos = brain_freq / (brain_freq + soul_freq)
            y_pos = soul_pos[1] * (1 - rel_pos)
            
            # Map band to x position
            band_positions = {
                'delta': -2.5,
                'theta': -1.5,
                'alpha': -0.5,
                'beta': 0.5,
                'gamma': 1.5,
                'lambda': 2.5
            }
            x_pos = band_positions.get(band, 0) * (1 - rel_pos)  # Closer to center as it approaches brain
            
            # Add some randomness
            x_pos += np.random.normal(0, 0.3)
            y_pos += np.random.normal(0, 0.3)
            
            # Color based on band
            band_colors = {
                'delta': 'purple',
                'theta': 'blue',
                'alpha': 'teal',
                'beta': 'green',
                'gamma': 'orange',
                'lambda': 'red'
            }
            color = band_colors.get(band, 'white')
            
            # Size and alpha based on resonance and energy
            size = 30 + 100 * resonance
            alpha = 0.3 + 0.7 * resonance
            
            # Plot harmonic node
            ax.scatter(x_pos, y_pos, s=size, color=color, alpha=alpha, 
                      edgecolor='white', linewidth=0.5, zorder=2)
    
    # Add connection metrics
    metrics_text = (
        f"Connection Strength: {connection_strength*100:.1f}%\n"
        f"Resonance Coherence: {brain_seed.soul_connection.get('resonance_coherence', 0)*100:.1f}%\n"
        f"Attachment Points: {len(brain_seed.soul_connection.get('brain_attachment_points', []))}\n"
        f"Soul Connection Points: {len(brain_seed.soul_connection.get('soul_connection_points', []))}"
    )
    
    ax.text(4, -3, metrics_text, fontsize=10, ha='left', va='center',
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Set up axes
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Soul-Brain Connection via Life Cord', fontsize=16)
    
    # Add labels
    ax.text(0, 5.5, "Soul", fontsize=14, ha='center', va='center',
           color='white', bbox=dict(facecolor='purple', alpha=0.7))
    
    ax.text(0, -3.5, "Brain", fontsize=14, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def _create_distributed_soul_field(distribution, resonant_soul, brain_seed):
    """
    Create distributed soul field throughout brain.
    
    Parameters:
        distribution: The distribution structure
        resonant_soul: The resonant soul
        brain_seed: The brain seed
        
    Returns:
        dict: Updated distribution structure
    """
    # Create soul field
    soul_field = {
        'overall_intensity': 0.0,
        'overall_coherence': 0.0,
        'regions': {},
        'hemispheres': {}
    }
    
    # Create field in each mapped region
    for region_name, mappings in distribution['region_mappings'].items():
        # Skip empty mappings
        if not mappings:
            continue
        
        # Get region
        region = brain_seed.region_structure.get(region_name, {})
        
        # Create field
        field = {
            'intensity': 0.0,
            'coherence': 0.0,
            'aspect_intensities': {},
            'pockets': []
        }
        
        # Calculate aspect intensities
        total_influence = sum(m['influence'] for m in mappings)
        
        for mapping in mappings:
            aspect_name = mapping['aspect']
            influence = mapping['influence']
            
            # Normalize influence
            normalized_influence = influence / total_influence if total_influence > 0 else 0
            
            # Store aspect intensity
            field['aspect_intensities'][aspect_name] = normalized_influence
            
            # Create soul pockets in region pockets
            if 'pockets' in region:
                for i, pocket in enumerate(region['pockets']):
                    # Only create soul pocket in some brain pockets
                    if np.random.random() < normalized_influence:
                        soul_pocket = {
                            'aspect': aspect_name,
                            'intensity': normalized_influence * (0.7 + 0.3 * np.random.random()),
                            'position': pocket['position'],
                            'frequency': SEPHIROTH_ASPECTS[aspect_name]['frequency'],
                            'color': _get_aspect_color(aspect_name),
                            'brain_pocket_id': pocket.get('id', f'unknown_{i}')
                        }
                        field['pockets'].append(soul_pocket)
        
        # Calculate field properties
        field['intensity'] = 0.3 + (0.7 * min(1.0, total_influence))
        field['coherence'] = 0.5 + (0.3 * np.random.random())
        
        # Store field
        soul_field['regions'][region_name] = field
    
    # Create field in each mapped hemisphere
    for hemi, mappings in distribution['hemisphere_mappings'].items():
        # Skip empty mappings
        if not mappings:
            continue
        
        # Create field
        field = {
            'intensity': 0.0,
            'coherence': 0.0,
            'aspect_intensities': {}
        }
        
        # Calculate aspect intensities
        total_influence = sum(m['influence'] for m in mappings)
        
        for mapping in mappings:
            aspect_name = mapping['aspect']
            influence = mapping['influence']
            
            # Normalize influence
            normalized_influence = influence / total_influence if total_influence > 0 else 0
            
            # Store aspect intensity
            field['aspect_intensities'][aspect_name] = normalized_influence
        
        # Calculate field properties
        field['intensity'] = 0.4 + (0.6 * min(1.0, total_influence))
        field['coherence'] = 0.6 + (0.3 * np.random.random())
        
        # Store field
        soul_field['hemispheres'][hemi] = field
    
    # Calculate overall field properties
    region_intensities = [r['intensity'] for r in soul_field['regions'].values()]
    hemisphere_intensities = [h['intensity'] for h in soul_field['hemispheres'].values()]
    
    # Overall intensity is weighted average of region and hemisphere intensities
    if region_intensities and hemisphere_intensities:
        region_avg = np.mean(region_intensities)
        hemi_avg = np.mean(hemisphere_intensities)
        
        soul_field['overall_intensity'] = 0.4 * region_avg + 0.6 * hemi_avg
    elif region_intensities:
        soul_field['overall_intensity'] = np.mean(region_intensities)
    elif hemisphere_intensities:
        soul_field['overall_intensity'] = np.mean(hemisphere_intensities)
    else:
        soul_field['overall_intensity'] = 0.0
    
    # Calculate coherence
    region_coherences = [r['coherence'] for r in soul_field['regions'].values()]
    hemisphere_coherences = [h['coherence'] for h in soul_field['hemispheres'].values()]
    
    if region_coherences and hemisphere_coherences:
        region_avg = np.mean(region_coherences)
        hemi_avg = np.mean(hemisphere_coherences)
        
        soul_field['overall_coherence'] = 0.3 * region_avg + 0.7 * hemi_avg
    elif region_coherences:
        soul_field['overall_coherence'] = np.mean(region_coherences)
    elif hemisphere_coherences:
        soul_field['overall_coherence'] = np.mean(hemisphere_coherences)
    else:
        soul_field['overall_coherence'] = 0.0
    
    # Store soul field
    distribution['soul_field'] = soul_field
    
    logger.info(f"Created distributed soul field with intensity {soul_field['overall_intensity']:.2f} "
               f"and coherence {soul_field['overall_coherence']:.2f}")
    
    return distribution

def _get_aspect_color(aspect_name):
    """Generate a color for a soul aspect."""
    # Define colors for each sephiroth aspect
    aspect_colors = {
        'kether': {'r': 255, 'g': 255, 'b': 255},  # White
        'chokmah': {'r': 200, 'g': 200, 'b': 255},  # Light blue
        'binah': {'r': 127, 'g': 0, 'b': 255},  # Purple
        'chesed': {'r': 0, 'g': 0, 'b': 255},  # Blue
        'geburah': {'r': 255, 'g': 0, 'b': 0},  # Red
        'tiphareth': {'r': 255, 'g': 215, 'b': 0},  # Gold
        'netzach': {'r': 0, 'g': 255, 'b': 0},  # Green
        'hod': {'r': 255, 'g': 165, 'b': 0},  # Orange
        'yesod': {'r': 130, 'g': 0, 'b': 255},  # Violet
        'malkuth': {'r': 139, 'g': 69, 'b': 19}   # Brown
    }
    
    # Return color for aspect or a default color
    return aspect_colors.get(aspect_name, {'r': 128, 'g': 128, 'b': 128})

