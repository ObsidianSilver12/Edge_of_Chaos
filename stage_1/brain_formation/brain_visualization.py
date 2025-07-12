"""
brain_visualization.py V8 - Module for visualizing brain structure and soul connection.

This module provides visualization tools for the new brain formation system including:
- Brain structure with hierarchical regions and energy systems
- Mycelial network distribution and energy flows
- Memory distribution and sephiroth aspects
- Stress monitoring and mother resonance
- Neural network connections and development

Updated for V8 brain formation architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import cm
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
color = mcolors.to_hex(cm.get_cmap('YlOrRd')(0.5))
import logging
from typing import Dict, List, Tuple, Optional, Any
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainVisualizationV8')

# Color maps for different visualization aspects
COLOR_MAPS = {
    'energy': cm.get_cmap('viridis'),
    'frequency': cm.get_cmap('plasma'),
    'resonance': cm.get_cmap('inferno'),
    'connection': cm.get_cmap('Blues'),
    'soul': cm.get_cmap('magma'),
    'mycelial': cm.get_cmap('YlOrRd'),
    'stress': cm.get_cmap('Reds'),
    'memory': cm.get_cmap('coolwarm')
}

# Sephiroth aspects with colors and frequencies
SEPHIROTH_ASPECTS = {
    'kether': {'frequency': 1000.0, 'color': '#FFFFFF'},    
    'chokmah': {'frequency': 900.0, 'color': '#C8C8FF'},
    'binah': {'frequency': 800.0, 'color': '#7F00FF'},
    'chesed': {'frequency': 700.0, 'color': '#0000FF'},
    'geburah': {'frequency': 600.0, 'color': '#FF0000'},
    'tiphareth': {'frequency': 500.0, 'color': '#FFD700'},
    'netzach': {'frequency': 400.0, 'color': '#00FF00'},
    'hod': {'frequency': 300.0, 'color': '#FFA500'},
    'yesod': {'frequency': 200.0, 'color': '#8200FF'},
    'malkuth': {'frequency': 100.0, 'color': '#8B4513'}
}

# Brain region positions for visualization
REGION_POSITIONS = {
    # Major regions
    'frontal_lobe': (-2.5, 2.0),
    'parietal_lobe': (0, 2.5),
    'temporal_lobe': (-3.0, 0),
    'occipital_lobe': (2.5, 0.5),
    'limbic_system': (0, 0),
    'cerebellum': (0, -2.5),
    'brainstem': (0, -3.5),
    
    # Sub-regions
    'prefrontal_cortex': (-3.0, 2.5),
    'motor_cortex': (-1.5, 2.8),
    'somatosensory_cortex': (1.0, 2.8),
    'visual_cortex': (3.0, 0.5),
    'auditory_cortex': (-3.5, 0.5),
    'hippocampus': (-0.5, 0),
    'amygdala': (0.5, 0),
    'thalamus': (0, 0.5),
    'hypothalamus': (0, -0.5),
    'pons': (0, -3.0),
    'medulla': (0, -4.0)
}

REGION_SIZES = {
    # Major regions
    'frontal_lobe': (2.5, 1.5),
    'parietal_lobe': (2.0, 1.5),
    'temporal_lobe': (2.0, 2.0),
    'occipital_lobe': (1.5, 1.5),
    'limbic_system': (1.8, 1.5),
    'cerebellum': (2.5, 1.2),
    'brainstem': (0.8, 1.5),
    
    # Sub-regions
    'prefrontal_cortex': (1.5, 1.0),
    'motor_cortex': (1.0, 0.8),
    'somatosensory_cortex': (1.0, 0.8),
    'visual_cortex': (1.0, 1.0),
    'auditory_cortex': (0.8, 0.8),
    'hippocampus': (0.6, 0.4),
    'amygdala': (0.4, 0.3),
    'thalamus': (0.5, 0.4),
    'hypothalamus': (0.4, 0.3),
    'pons': (0.6, 0.5),
    'medulla': (0.5, 0.4)
}

def visualize_complete_brain_system(brain_structure=None, mycelial_network=None, 
                                   energy_storage=None, memory_distribution=None,
                                   stress_monitoring=None, neural_network=None,
                                   save_path=None, show=True):
    """
    Create comprehensive visualization of the complete brain formation system.
    
    Parameters:
        brain_structure: BrainStructure instance
        mycelial_network: MycelialNetwork instance
        energy_storage: EnergyStorage instance
        memory_distribution: MemoryDistribution instance
        stress_monitoring: StressMonitoring instance
        neural_network: NeuralNetwork instance
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating complete brain system visualization")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Brain Structure Overview (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    _plot_brain_structure(ax1, brain_structure)
    ax1.set_title('Brain Structure & Regions', fontsize=14, fontweight='bold')
    
    # 2. Mycelial Network (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    _plot_mycelial_network(ax2, mycelial_network, brain_structure)
    ax2.set_title('Mycelial Network & Energy', fontsize=14, fontweight='bold')
    
    # 3. Memory Distribution (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    _plot_memory_distribution(ax3, memory_distribution, brain_structure)
    ax3.set_title('Memory & Soul Aspects', fontsize=14, fontweight='bold')
    
    # 4. Neural Network (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    _plot_neural_network(ax4, neural_network, brain_structure)
    ax4.set_title('Neural Network & Synapses', fontsize=14, fontweight='bold')
    
    # 5. Energy & Stress Monitoring (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    _plot_energy_and_stress(ax5, energy_storage, stress_monitoring)
    ax5.set_title('Energy Storage & Stress', fontsize=14, fontweight='bold')
    
    # 6. System Metrics (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    _plot_system_metrics(ax6, brain_structure, mycelial_network, energy_storage, 
                        memory_distribution, stress_monitoring, neural_network)
    ax6.set_title('System Metrics & Status', fontsize=14, fontweight='bold')
    
    # Add overall title with formation status
    formation_status = "Unknown"
    if brain_structure and hasattr(brain_structure, 'formation_complete'):
        formation_status = "Complete" if brain_structure.formation_complete else "In Progress"
    
    plt.suptitle(f'Complete Brain Formation System - Status: {formation_status}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Complete brain system visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def _plot_brain_structure(ax, brain_structure):
    """Plot brain structure with hierarchical regions."""
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=8, height=6, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=2, alpha=0.7)
    ax.add_patch(brain_outline)
    
    # Draw hemisphere divider
    ax.plot([0, 0], [-3, 3], 'k--', linewidth=1, alpha=0.5)
    
    if not brain_structure:
        ax.text(0, 0, 'Brain Structure\nNot Available', ha='center', va='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        return
    
    # Get brain regions if available
    regions = getattr(brain_structure, 'regions', {})
    active_cells = getattr(brain_structure, 'active_cells', {})
    
    # Plot major regions
    for region_name in ['frontal_lobe', 'parietal_lobe', 'temporal_lobe', 
                       'occipital_lobe', 'limbic_system', 'cerebellum', 'brainstem']:
        if region_name in REGION_POSITIONS:
            pos = REGION_POSITIONS[region_name]
            size = REGION_SIZES.get(region_name, (1.0, 1.0))
            
            # Get region activity level
            activity = 0.5  # Default
            if regions and region_name in regions:
                region_data = regions[region_name]
                activity = region_data.get('activity_level', 0.5)
            
            # Color based on activity
            color = COLOR_MAPS['energy'](activity)
            
            # Create region shape
            if region_name == 'limbic_system':
                patch = Circle(pos, size[0]/2, fill=True, alpha=0.6,
                             facecolor=color, edgecolor='black', linewidth=1)
            elif region_name in ['brainstem']:
                patch = Rectangle((pos[0]-size[0]/2, pos[1]-size[1]/2), 
                                size[0], size[1], fill=True, alpha=0.6,
                                facecolor=color, edgecolor='black', linewidth=1)
            else:
                patch = FancyBboxPatch((pos[0]-size[0]/2, pos[1]-size[1]/2), 
                                     size[0], size[1], boxstyle="round,pad=0.1",
                                     fill=True, alpha=0.6, facecolor=color, 
                                     edgecolor='black', linewidth=1)
            
            ax.add_patch(patch)
            
            # Add label
            display_name = region_name.replace('_', ' ').title()
            ax.text(pos[0], pos[1], display_name, ha='center', va='center', 
                   fontsize=8, fontweight='bold')
    
    # Plot active cells as small dots
    if active_cells:
        cell_count = 0
        for coord, cell_data in active_cells.items():
            if cell_count > 100:  # Limit display for performance
                break
            
            # Convert 3D coordinates to 2D for display
            if isinstance(coord, tuple) and len(coord) >= 2:
                x, y = coord[0] * 0.03, coord[1] * 0.03  # Scale down
                
                # Skip if outside brain outline
                if x*x/16 + y*y/9 > 1:
                    continue
                
                # Get cell type and activity
                cell_type = cell_data.get('type', 'unknown')
                activity = cell_data.get('activity', 0.5)
                
                # Color based on cell type
                type_colors = {
                    'neuron': 'blue',
                    'memory': 'purple',
                    'energy': 'red',
                    'mycelial': 'orange'
                }
                color = type_colors.get(cell_type, 'gray')
                
                # Plot cell
                ax.scatter(x, y, s=20*activity, c=color, alpha=0.7, edgecolors=None)
                cell_count += 1
    
    # Add formation progress if available
    if hasattr(brain_structure, 'formation_complete'):
        progress_text = "Formation: Complete" if brain_structure.formation_complete else "Formation: In Progress"
        ax.text(-4, -3.5, progress_text, fontsize=10, 
               bbox=dict(facecolor='lightgreen' if brain_structure.formation_complete else 'lightyellow', 
                        alpha=0.8))
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

def _plot_mycelial_network(ax, mycelial_network, brain_structure):
    """Plot mycelial network distribution and energy flows."""
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=8, height=6, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.add_patch(brain_outline)
    
    if not mycelial_network:
        ax.text(0, 0, 'Mycelial Network\nNot Available', ha='center', va='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        return
    
    # Get mycelial network data
    network_nodes = getattr(mycelial_network, 'network_nodes', {})
    energy_flows = getattr(mycelial_network, 'energy_flows', [])
    processing_hubs = getattr(mycelial_network, 'processing_hubs', {})
    
    # Plot network nodes
    for node_id, node_data in network_nodes.items():
        coord = node_data.get('coordinate', (0, 0, 0))
        energy_level = node_data.get('energy_level', 0.5)
        node_type = node_data.get('type', 'standard')
        
        # Convert to 2D
        x, y = coord[0] * 0.03, coord[1] * 0.03
        
        # Skip if outside brain
        if x*x/16 + y*y/9 > 1:
            continue
        
        # Size and color based on energy and type
        size = 30 + 50 * energy_level
        if node_type == 'hub':
            color = 'red'
            size *= 1.5
        elif node_type == 'seed':
            color = 'gold'
        else:
            color = COLOR_MAPS['mycelial'](energy_level)
        
        ax.scatter(x, y, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # Plot energy flows as lines
    node_positions = {}
    for node_id, node_data in network_nodes.items():
        coord = node_data.get('coordinate', (0, 0, 0))
        x, y = coord[0] * 0.03, coord[1] * 0.03
        node_positions[node_id] = (x, y)
    
    for flow in energy_flows:
        source_id = flow.get('source')
        target_id = flow.get('target')
        flow_rate = flow.get('flow_rate', 0.5)
        
        if source_id in node_positions and target_id in node_positions:
            source_pos = node_positions[source_id]
            target_pos = node_positions[target_id]
            
            # Skip if either position is outside brain
            if (source_pos[0]**2/16 + source_pos[1]**2/9 > 1 or 
                target_pos[0]**2/16 + target_pos[1]**2/9 > 1):
                continue
            
            # Draw flow line
            ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]],
                   color='orange', alpha=0.3 + 0.5*flow_rate, linewidth=1 + 2*flow_rate)
    
    # Plot processing hubs with special markers
    for hub_id, hub_data in processing_hubs.items():
        coord = hub_data.get('coordinate', (0, 0, 0))
        processing_capacity = hub_data.get('processing_capacity', 0.5)
        
        x, y = coord[0] * 0.03, coord[1] * 0.03
        
        # Skip if outside brain
        if x*x/16 + y*y/9 > 1:
            continue
        
        # Draw hub with special symbol
        size = 100 + 100 * processing_capacity
        ax.scatter(x, y, s=size, c='none', edgecolors='red', linewidth=3, marker='s')
        ax.scatter(x, y, s=size*0.5, c='red', alpha=0.3)
    
    # Add network stats
    node_count = len(network_nodes)
    flow_count = len(energy_flows)
    hub_count = len(processing_hubs)
    
    stats_text = f"Nodes: {node_count}\nFlows: {flow_count}\nHubs: {hub_count}"
    ax.text(-4.5, 3, stats_text, fontsize=10, 
           bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

def _plot_memory_distribution(ax, memory_distribution, brain_structure):
    """Plot memory distribution and sephiroth aspects."""
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=8, height=6, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.add_patch(brain_outline)
    
    if not memory_distribution:
        ax.text(0, 0, 'Memory Distribution\nNot Available', ha='center', va='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        return
    
    # Get memory data
    sephiroth_memories = getattr(memory_distribution, 'sephiroth_memories', {})
    identity_aspects = getattr(memory_distribution, 'identity_aspects', {})
    memory_coordinates = getattr(memory_distribution, 'memory_coordinates', {})
    
    # Plot sephiroth aspects
    for aspect_name, aspect_data in sephiroth_memories.items():
        if aspect_name in SEPHIROTH_ASPECTS:
            coord = aspect_data.get('coordinate', (0, 0, 0))
            intensity = aspect_data.get('intensity', 0.5)
            
            # Convert to 2D
            x, y = coord[0] * 0.03, coord[1] * 0.03
            
            # Skip if outside brain
            if x*x/16 + y*y/9 > 1:
                continue
            
            # Get aspect color and properties
            aspect_info = SEPHIROTH_ASPECTS[aspect_name]
            color = aspect_info['color']
            
            # Plot aspect
            size = 50 + 100 * intensity
            ax.scatter(x, y, s=size, c=color, alpha=0.7, 
                      edgecolors='white', linewidth=1)
            
            # Add label for larger aspects
            if intensity > 0.7:
                ax.text(x, y-0.3, aspect_name, ha='center', va='top', 
                       fontsize=8, fontweight='bold')
    
    # Plot identity aspects
    for aspect_name, aspect_data in identity_aspects.items():
        coord = aspect_data.get('coordinate', (0, 0, 0))
        strength = aspect_data.get('strength', 0.5)
        
        # Convert to 2D
        x, y = coord[0] * 0.03, coord[1] * 0.03
        
        # Skip if outside brain
        if x*x/16 + y*y/9 > 1:
            continue
        
        # Plot identity aspect
        size = 30 + 60 * strength
        ax.scatter(x, y, s=size, c='cyan', alpha=0.5, 
                  edgecolors='blue', linewidth=1, marker='d')
    
    # Plot memory coordinates as small points
    coord_count = 0
    for memory_id, coord in memory_coordinates.items():
        if coord_count > 50:  # Limit display
            break
        
        # Convert to 2D
        x, y = coord[0] * 0.03, coord[1] * 0.03
        
        # Skip if outside brain
        if x*x/16 + y*y/9 > 1:
            continue
        
        # Plot memory point
        ax.scatter(x, y, s=10, c='purple', alpha=0.3)
        coord_count += 1
    
    # Add memory stats
    seph_count = len(sephiroth_memories)
    identity_count = len(identity_aspects)
    coord_count = len(memory_coordinates)
    
    stats_text = f"Sephiroth: {seph_count}\nIdentity: {identity_count}\nCoords: {coord_count}"
    ax.text(3, 3, stats_text, fontsize=10, 
           bbox=dict(facecolor='lightcyan', alpha=0.8))
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

def _plot_neural_network(ax, neural_network, brain_structure):
    """Plot neural network nodes and synaptic connections."""
    # Draw brain outline
    brain_outline = Ellipse((0, 0), width=8, height=6, fill=False, 
                          edgecolor='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.add_patch(brain_outline)
    
    if not neural_network:
        ax.text(0, 0, 'Neural Network\nNot Available', ha='center', va='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        return
    
    # Get neural data
    nodes = getattr(neural_network, 'nodes', {})
    synapses = getattr(neural_network, 'synapses', {})
    
    # Plot neural nodes
    node_positions = {}
    for node_id, node_data in nodes.items():
        coord = node_data.get('coordinate', (0, 0, 0))
        activation = node_data.get('activation_level', 0.5)
        node_type = node_data.get('type', 'standard')
        
        # Convert to 2D
        x, y = coord[0] * 0.03, coord[1] * 0.03
        
        # Skip if outside brain
        if x*x/16 + y*y/9 > 1:
            continue
        
        node_positions[node_id] = (x, y)
        
        # Size and color based on activation and type
        size = 20 + 40 * activation
        if node_type == 'motor':
            color = 'red'
        elif node_type == 'sensory':
            color = 'blue'
        elif node_type == 'memory':
            color = 'purple'
        else:
            color = COLOR_MAPS['connection'](activation)
        
        ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Plot synapses as connections
    synapse_count = 0
    for synapse_id, synapse_data in synapses.items():
        if synapse_count > 100:  # Limit display for performance
            break
        
        source_id = synapse_data.get('source_node')
        target_id = synapse_data.get('target_node')
        strength = synapse_data.get('strength', 0.5)
        
        if source_id in node_positions and target_id in node_positions:
            source_pos = node_positions[source_id]
            target_pos = node_positions[target_id]
            
            # Draw synapse
            ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]],
                   color='gray', alpha=0.2 + 0.6*strength, linewidth=0.5 + 1.5*strength)
            synapse_count += 1
    
    # Add neural stats
    node_count = len(nodes)
    synapse_count = len(synapses)
    
    # Calculate average activation
    if nodes:
        avg_activation = np.mean([n.get('activation_level', 0.5) for n in nodes.values()])
    else:
        avg_activation = 0.0
    
    stats_text = f"Nodes: {node_count}\nSynapses: {synapse_count}\nAvg Act: {avg_activation:.2f}"
    ax.text(-4.5, -3, stats_text, fontsize=10, 
           bbox=dict(facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

def _plot_energy_and_stress(ax, energy_storage, stress_monitoring):
    """Plot energy storage and stress monitoring systems."""
    if not energy_storage and not stress_monitoring:
        ax.text(0.5, 0.5, 'Energy & Stress\nSystems\nNot Available', ha='center', va='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8),
                transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Energy storage visualization
    if energy_storage:
        # Get energy data
        energy_pools = getattr(energy_storage, 'energy_pools', {})
        total_capacity = getattr(energy_storage, 'total_capacity', 1000)
        current_energy = getattr(energy_storage, 'current_total_energy', 500)
        
        # Energy gauge
        energy_ratio = current_energy / total_capacity if total_capacity > 0 else 0
        
        # Draw energy gauge
        theta = np.linspace(0, 2*np.pi*energy_ratio, 50)
        x_gauge = 0.5 + 0.3 * np.cos(theta)
        y_gauge = 0.8 + 0.3 * np.sin(theta)
        
        ax.fill_between(x_gauge, y_gauge, 0.8, alpha=0.6, 
                       color=COLOR_MAPS['energy'](energy_ratio), 
                       transform=ax.transAxes)
        
        # Energy gauge outline
        theta_full = np.linspace(0, 2*np.pi, 100)
        x_outline = 0.5 + 0.3 * np.cos(theta_full)
        y_outline = 0.8 + 0.3 * np.sin(theta_full)
        ax.plot(x_outline, y_outline, 'k-', linewidth=2, transform=ax.transAxes)
        
        # Energy text
        ax.text(0.5, 0.8, f'{current_energy:.0f}/{total_capacity:.0f}', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               transform=ax.transAxes)
        
        ax.text(0.5, 0.65, 'Energy Level', ha='center', va='center', 
               fontsize=10, transform=ax.transAxes)
        
        # Energy pool breakdown
        pool_text = "Energy Pools:\n"
        for pool_name, pool_data in energy_pools.items():
            current = pool_data.get('current_energy', 0)
            capacity = pool_data.get('capacity', 100)
            pool_text += f"{pool_name}: {current:.0f}/{capacity:.0f}\n"
        
        ax.text(0.05, 0.5, pool_text, ha='left', va='top', fontsize=8,
               bbox=dict(facecolor='lightyellow', alpha=0.8),
               transform=ax.transAxes)
    
    # Stress monitoring visualization
    if stress_monitoring:
        # Get stress data
        stress_level = getattr(stress_monitoring, 'current_stress_level', 0.3)
        stress_threshold = getattr(stress_monitoring, 'stress_threshold', 0.8)
        mother_resonance = getattr(stress_monitoring, 'mother_resonance_active', False)
        
        # Stress gauge
        stress_ratio = stress_level / stress_threshold if stress_threshold > 0 else 0
        
        # Draw stress gauge (bottom half circle)
        theta = np.linspace(np.pi, 2*np.pi, 50)
        theta_stress = np.linspace(np.pi, np.pi + np.pi*stress_ratio, int(50*stress_ratio))
        
        x_stress_outline = 0.5 + 0.25 * np.cos(theta)
        y_stress_outline = 0.3 + 0.25 * np.sin(theta)
        ax.plot(x_stress_outline, y_stress_outline, 'k-', linewidth=2, transform=ax.transAxes)
        
        if len(theta_stress) > 0:
            x_stress = 0.5 + 0.25 * np.cos(theta_stress)
            y_stress = 0.3 + 0.25 * np.sin(theta_stress)
            
            # Color based on stress level
            if stress_ratio > 0.8:
                stress_color = 'red'
            elif stress_ratio > 0.5:
                stress_color = 'orange'
            else:
                stress_color = 'green'
            
            ax.fill_between(x_stress, y_stress, 0.3, alpha=0.7, color=stress_color,
                           transform=ax.transAxes)
        
        # Stress text
        ax.text(0.5, 0.3, f'{stress_level:.2f}/{stress_threshold:.2f}', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               transform=ax.transAxes)
        
        ax.text(0.5, 0.15, 'Stress Level', ha='center', va='center', 
               fontsize=10, transform=ax.transAxes)
        
        # Mother resonance indicator
        resonance_color = 'lightgreen' if mother_resonance else 'lightgray'
        resonance_text = 'Mother Resonance: ' + ('ACTIVE' if mother_resonance else 'INACTIVE')
        ax.text(0.95, 0.5, resonance_text, ha='right', va='center', fontsize=9,
               bbox=dict(facecolor=resonance_color, alpha=0.8),
               transform=ax.transAxes)
        
        # Stress monitoring events (if available)
        monitoring_events = getattr(stress_monitoring, 'monitoring_events', [])
        if monitoring_events:
            recent_events = monitoring_events[-3:]  # Last 3 events
            events_text = "Recent Events:\n"
            for event in recent_events:
                event_type = event.get('event_type', 'unknown')
                stress_change = event.get('stress_change', 0)
                events_text += f"• {event_type}: {stress_change:+.2f}\n"
            
            ax.text(0.95, 0.15, events_text, ha='right', va='top', fontsize=8,
                   bbox=dict(facecolor='lightcyan', alpha=0.8),
                   transform=ax.transAxes)
    
    ax.axis('off')

def _plot_system_metrics(ax, brain_structure, mycelial_network, energy_storage, 
                        memory_distribution, stress_monitoring, neural_network):
    """Plot overall system metrics and status."""
    ax.axis('off')
    
    # Collect metrics from all systems
    metrics = {}
    
    # Brain structure metrics
    if brain_structure:
        metrics['Brain Formation'] = 'Complete' if getattr(brain_structure, 'formation_complete', False) else 'In Progress'
        metrics['Active Cells'] = len(getattr(brain_structure, 'active_cells', {}))
        metrics['Regions'] = len(getattr(brain_structure, 'regions', {}))
    
    # Mycelial network metrics
    if mycelial_network:
        metrics['Network Nodes'] = len(getattr(mycelial_network, 'network_nodes', {}))
        metrics['Energy Flows'] = len(getattr(mycelial_network, 'energy_flows', []))
        metrics['Processing Hubs'] = len(getattr(mycelial_network, 'processing_hubs', {}))
    
    # Energy storage metrics
    if energy_storage:
        total_capacity = getattr(energy_storage, 'total_capacity', 1000)
        current_energy = getattr(energy_storage, 'current_total_energy', 500)
        energy_ratio = current_energy / total_capacity if total_capacity > 0 else 0
        metrics['Energy Level'] = f'{energy_ratio*100:.1f}%'
        metrics['Energy Pools'] = len(getattr(energy_storage, 'energy_pools', {}))
    
    # Memory distribution metrics
    if memory_distribution:
        metrics['Sephiroth Aspects'] = len(getattr(memory_distribution, 'sephiroth_memories', {}))
        metrics['Identity Aspects'] = len(getattr(memory_distribution, 'identity_aspects', {}))
        metrics['Memory Coordinates'] = len(getattr(memory_distribution, 'memory_coordinates', {}))
    
    # Stress monitoring metrics
    if stress_monitoring:
        stress_level = getattr(stress_monitoring, 'current_stress_level', 0.3)
        metrics['Stress Level'] = f'{stress_level:.2f}'
        metrics['Mother Resonance'] = 'Active' if getattr(stress_monitoring, 'mother_resonance_active', False) else 'Inactive'
    
    # Neural network metrics
    if neural_network:
        metrics['Neural Nodes'] = len(getattr(neural_network, 'nodes', {}))
        metrics['Synapses'] = len(getattr(neural_network, 'synapses', {}))
    
    # Create metrics table
    if metrics:
        y_pos = 0.95
        ax.text(0.5, y_pos, 'System Metrics', ha='center', va='top', 
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        y_pos -= 0.08
        
        for label, value in metrics.items():
            # Color code certain values
            color = 'black'
            if 'Complete' in str(value):
                color = 'green'
            elif 'In Progress' in str(value):
                color = 'orange'
            elif 'Active' in str(value):
                color = 'green'
            elif 'Inactive' in str(value):
                color = 'red'
            
            ax.text(0.05, y_pos, f'{label}:', ha='left', va='top', 
                   fontsize=10, fontweight='bold', transform=ax.transAxes)
            ax.text(0.95, y_pos, str(value), ha='right', va='top', 
                   fontsize=10, color=color, transform=ax.transAxes)
            y_pos -= 0.06
    
    # Add overall system status
    status_y = 0.15
    ax.text(0.5, status_y, 'Overall System Status', ha='center', va='center', 
           fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Determine overall status
    formation_complete = brain_structure and getattr(brain_structure, 'formation_complete', False)
    has_energy = energy_storage and getattr(energy_storage, 'current_total_energy', 0) > 0
    has_network = mycelial_network and len(getattr(mycelial_network, 'network_nodes', {})) > 0
    has_memory = memory_distribution and len(getattr(memory_distribution, 'sephiroth_memories', {})) > 0
    
    if formation_complete and has_energy and has_network and has_memory:
        status = "FULLY OPERATIONAL"
        status_color = 'green'
    elif formation_complete:
        status = "FORMATION COMPLETE"
        status_color = 'lightgreen'
    elif has_network and has_energy:
        status = "DEVELOPING"
        status_color = 'orange'
    else:
        status = "INITIALIZING"
        status_color = 'yellow'
    
    ax.text(0.5, status_y - 0.05, status, ha='center', va='center', 
           fontsize=14, fontweight='bold', color='white',
           bbox=dict(facecolor=status_color, alpha=0.8, pad=10),
           transform=ax.transAxes)

def visualize_brain_development_progress(brain_structure=None, mycelial_network=None, 
                                       energy_storage=None, memory_distribution=None,
                                       stress_monitoring=None, neural_network=None,
                                       save_path=None, show=True):
    """
    Visualize the development progress of the brain formation system.
    
    Parameters:
        brain_structure: BrainStructure instance
        mycelial_network: MycelialNetwork instance
        energy_storage: EnergyStorage instance
        memory_distribution: MemoryDistribution instance
        stress_monitoring: StressMonitoring instance
        neural_network: NeuralNetwork instance
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating brain development progress visualization")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define development stages
    stages = [
        {'name': 'Brain Seed', 'progress': 0.1, 'description': 'Initial energy spark creation'},
        {'name': 'Brain Structure', 'progress': 0.25, 'description': 'Hierarchical brain regions formation'},
        {'name': 'Neural Network', 'progress': 0.4, 'description': 'Neural nodes and synaptic connections'},
        {'name': 'Mycelial Network', 'progress': 0.6, 'description': 'Energy management and processing system'},
        {'name': 'Energy Storage', 'progress': 0.75, 'description': 'Limbic energy pools and distribution'},
        {'name': 'Memory Distribution', 'progress': 0.85, 'description': 'Sephiroth aspects and identity mapping'},
        {'name': 'Stress Monitoring', 'progress': 0.95, 'description': 'Mother resonance and protection systems'},
        {'name': 'System Integration', 'progress': 1.0, 'description': 'Complete brain formation ready for birth'}
    ]
    
    # Determine current progress based on available systems
    current_progress = 0.0
    
    if brain_structure:
        current_progress = max(current_progress, 0.25)
        if getattr(brain_structure, 'formation_complete', False):
            current_progress = max(current_progress, 0.3)
    
    if neural_network and len(getattr(neural_network, 'nodes', {})) > 0:
        current_progress = max(current_progress, 0.4)
        if len(getattr(neural_network, 'synapses', {})) > 0:
            current_progress = max(current_progress, 0.5)
    
    if mycelial_network and len(getattr(mycelial_network, 'network_nodes', {})) > 0:
        current_progress = max(current_progress, 0.6)
        if len(getattr(mycelial_network, 'processing_hubs', {})) > 0:
            current_progress = max(current_progress, 0.65)
    
    if energy_storage and getattr(energy_storage, 'current_total_energy', 0) > 0:
        current_progress = max(current_progress, 0.75)
    
    if memory_distribution and len(getattr(memory_distribution, 'sephiroth_memories', {})) > 0:
        current_progress = max(current_progress, 0.85)
    
    if stress_monitoring:
        current_progress = max(current_progress, 0.95)
        if getattr(stress_monitoring, 'mother_resonance_active', False):
            current_progress = max(current_progress, 1.0)
    
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
        ax.scatter(progress, y_pos, s=150, color=color, alpha=alpha, zorder=3, edgecolors='white', linewidth=2)
        
        # Add stage name
        ax.text(progress, y_pos + 0.15, name, ha='center', va='bottom', fontsize=11,
               fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        # Add description
        status_text = f"[{status.upper()}]" if status != 'pending' else ""
        ax.text(progress, y_pos - 0.15, f"{description}\n{status_text}", ha='center', va='top', 
               fontsize=9, alpha=0.8 if status != 'pending' else 0.5,
               bbox=dict(facecolor='white', alpha=0.6))
        
        # Connect stages with line
        if i > 0:
            prev_progress = stages[i-1]['progress']
            line_color = color if current_progress >= prev_progress else 'lightgray'
            ax.plot([prev_progress, progress], [y_pos, y_pos], color=line_color, 
                   alpha=alpha, linestyle='-', linewidth=3, zorder=2)
    
    # Add current progress marker
    ax.axvline(x=current_progress, color='red', linestyle='--', linewidth=3, alpha=0.8, zorder=1)
    ax.text(current_progress, -0.4, f"Current Progress: {current_progress*100:.1f}%", 
           color='red', ha='center', va='top', fontsize=14, fontweight='bold',
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', linewidth=2))
    
    # Set up axes
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.6, 0.4)
    ax.set_xlabel('Development Progress', fontsize=12, fontweight='bold')
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Brain Formation Development Timeline', fontsize=16, fontweight='bold')
    
    # Add progress ticks
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Brain development progress visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_3d_brain_structure(brain_structure=None, mycelial_network=None,
                                memory_distribution=None, save_path=None, show=True):
    """
    Create a 3D visualization of the brain structure with mycelial network and memory distribution.
    
    Parameters:
        brain_structure: BrainStructure instance
        mycelial_network: MycelialNetwork instance
        memory_distribution: MemoryDistribution instance
        save_path (str, optional): Path to save the visualization image
        show (bool): Whether to display the visualization
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    logger.info("Generating 3D brain structure visualization")
    
    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')  # Ensure 3D axes
    
    # Brain outline as wireframe sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_brain = 4 * np.outer(np.cos(u), np.sin(v))
    y_brain = 3 * np.outer(np.sin(u), np.sin(v))
    z_brain = 2.5 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot wireframe brain outline
    if hasattr(ax, 'plot_wireframe'):
        ax.plot_wireframe(x_brain, y_brain, z_brain, alpha=0.1, color='gray')
    else:
        logger.warning("3D plotting is not available on this axes object.")
    
    # Plot active brain cells
    if brain_structure:
        active_cells = getattr(brain_structure, 'active_cells', {})
        
        cell_count = 0
        for coord, cell_data in active_cells.items():
            if cell_count > 200:  # Limit for performance
                break
            
            if isinstance(coord, tuple) and len(coord) >= 3:
                x, y, z = coord[0] * 0.03, coord[1] * 0.03, coord[2] * 0.03
                
                # Skip if outside brain ellipsoid
                if (x*x/16 + y*y/9 + z*z/6.25) > 1:
                    continue
                
                cell_type = cell_data.get('type', 'unknown')
                activity = cell_data.get('activity', 0.5)
                
                # Color based on cell type
                type_colors = {
                    'neuron': 'blue',
                    'memory': 'purple',
                    'energy': 'red',
                    'mycelial': 'orange'
                }
                color = type_colors.get(cell_type, 'gray')
                
                ax.scatter(x, y, z, s=20*activity, c=color, alpha=0.6)
                cell_count += 1
    
    # Plot mycelial network nodes
    if mycelial_network:
        network_nodes = getattr(mycelial_network, 'network_nodes', {})
        
        node_positions = {}
        for node_id, node_data in network_nodes.items():
            coord = node_data.get('coordinate', (0, 0, 0))
            energy_level = node_data.get('energy_level', 0.5)
            node_type = node_data.get('type', 'standard')
            
            x, y, z = coord[0] * 0.03, coord[1] * 0.03, coord[2] * 0.03
            
            # Skip if outside brain
            if (x*x/16 + y*y/9 + z*z/6.25) > 1:
                continue
            
            node_positions[node_id] = (x, y, z)
            
            # Size and color based on energy and type
            node_size = 50 + 100 * energy_level
            if node_type == 'hub':
                color = 'red'
                node_size *= 1.5
            elif node_type == 'seed':
                color = 'gold'
            else:
                color = COLOR_MAPS['mycelial'](energy_level)
            
            ax.scatter(x, y, z, s=node_size, c=color, alpha=0.8, edgecolors='white')
        
        # Plot energy flows
        energy_flows = getattr(mycelial_network, 'energy_flows', [])
        for flow in energy_flows[:50]:  # Limit for performance
            source_id = flow.get('source')
            target_id = flow.get('target')
            flow_rate = flow.get('flow_rate', 0.5)
            
            if source_id in node_positions and target_id in node_positions:
                source_pos = node_positions[source_id]
                target_pos = node_positions[target_id]
                
                ax.plot([source_pos[0], target_pos[0]], 
                       [source_pos[1], target_pos[1]],
                       [source_pos[2], target_pos[2]],
                       color='orange', alpha=0.3 + 0.5*flow_rate, 
                       linewidth=1 + 2*flow_rate)
    
    # Plot memory aspects
    if memory_distribution:
        sephiroth_memories = getattr(memory_distribution, 'sephiroth_memories', {})
        
        for aspect_name, aspect_data in sephiroth_memories.items():
            if aspect_name in SEPHIROTH_ASPECTS:
                coord = aspect_data.get('coordinate', (0, 0, 0))
                intensity = aspect_data.get('intensity', 0.5)
                
                x, y, z = coord[0] * 0.03, coord[1] * 0.03, coord[2] * 0.03
                
                # Skip if outside brain
                if (x*x/16 + y*y/9 + z*z/6.25) > 1:
                    continue
                
                # Get aspect color
                aspect_info = SEPHIROTH_ASPECTS[aspect_name]
                color = aspect_info['color']
                
                # Plot with size based on intensity
                aspect_size = 100 + 200 * intensity
                ax.scatter(x, y, z, s=aspect_size, c=color, alpha=0.7, 
                          edgecolors='white', linewidth=1, marker='D')
    
    # Set up 3D plot
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('3D Brain Structure with Networks', fontsize=16, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = 5
    ax.set_xlim((-max_range, max_range))
    ax.set_ylim((-max_range, max_range))
    ax.set_zlim3d((-max_range, max_range))
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=8, label='Neural Cells'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=8, label='Mycelial Nodes'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', 
                   markersize=8, label='Memory Aspects'),
        Line2D([0], [0], color='orange', linewidth=2, alpha=0.7, 
                   label='Energy Flows')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"3D brain structure visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
