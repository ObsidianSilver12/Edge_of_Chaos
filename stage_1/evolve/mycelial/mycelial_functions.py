# --- START OF FILE stage_1/evolve/mycelial_functions.py ---

"""
Mycelial Network Functions (V4.5.0 - Stage 1 Implementation)

Provides limited subset of mycelial network functionality needed for Stage 1 brain development.
Handles initial network setup, basic operations, and energy distribution.
Follows proper physics principles with hard validation and no fallbacks.
"""

import numpy as np
import logging
import os
import sys
import json
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import math
import random
from stage_1.evolve.core.evolve_constants import *
from constants.constants import *

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logging.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()
# --- Logging Setup ---
logger = logging.getLogger("MycelialFunctions")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Try to import metrics tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not available. Metrics will not be recorded.")
    METRICS_AVAILABLE = False


# --- Basic Network Initialization ---
def initialize_basic_network(brain_structure, seed_position: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Initialize the basic mycelial network structure radiating from seed position.
    This creates the foundation for the subconscious system.
    
    Args:
        brain_structure: The brain structure object
        seed_position: Position of the brain seed
        
    Returns:
        Dict with initialization metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info(f"Initializing basic mycelial network from seed position {seed_position}")
    
    # Validate inputs
    if not hasattr(brain_structure, 'dimensions'):
        raise ValueError("Invalid brain_structure object. Missing dimensions attribute.")
    
    x, y, z = seed_position
    if not (0 <= x < brain_structure.dimensions[0] and 
            0 <= y < brain_structure.dimensions[1] and 
            0 <= z < brain_structure.dimensions[2]):
        raise ValueError(f"Seed position {seed_position} out of brain structure bounds {brain_structure.dimensions}")
    
    # Define network parameters
    initial_radius = min(20, max(10, min(brain_structure.dimensions) // 10))
    base_density = MYCELIAL_DEFAULT_DENSITY
    radial_falloff_factor = 2.0  # Controls how quickly density falls off with distance
    
    # Track cells affected
    cells_affected = 0
    max_density = 0.0
    
    # Create radial mycelial network around seed with phi-based distribution
    for dx in range(-initial_radius, initial_radius + 1):
        for dy in range(-initial_radius, initial_radius + 1):
            for dz in range(-initial_radius, initial_radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                        0 <= ny < brain_structure.dimensions[1] and 
                        0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Calculate distance from seed
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                # Apply mycelial density if within radius
                if dist <= initial_radius:
                    # Calculate phi-harmonic density pattern
                    # Use PHI (golden ratio) for natural-looking distribution
                    phi_factor = 0.5 + 0.5 * math.cos(dist * PHI)
                    
                    # Apply radial falloff with exponential decay
                    falloff = math.exp(-dist / (initial_radius / radial_falloff_factor))
                    
                    # Calculate final density
                    density = base_density * falloff * (1.0 + 0.3 * phi_factor)
                    
                    # Apply value to brain structure (never reducing existing density)
                    brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_density_grid[nx, ny, nz],
                        density
                    )
                    
                    # Track metrics
                    cells_affected += 1
                    max_density = max(max_density, density)
                    
                    # Add small amount of energy to network
                    brain_structure.mycelial_energy_grid[nx, ny, nz] += 0.01 * density
    
    # Calculate region coverage
    region_coverage = {}
    for region_name in brain_structure.regions:
        # Find cells for this region
        region_indices = np.where(brain_structure.region_grid == region_name)
        if len(region_indices[0]) > 0:
            # Calculate mycelial presence in this region
            region_mycelial = brain_structure.mycelial_density_grid[region_indices]
            mycelial_cells = np.sum(region_mycelial > 0.01)
            total_cells = len(region_indices[0])
            
            region_coverage[region_name] = mycelial_cells / total_cells
    
    logger.info(f"Basic mycelial network initialized with {cells_affected} cells affected. "
               f"Max density: {max_density:.2f}")
    
    # Return metrics
    metrics_result = {
        'cells_affected': cells_affected,
        'initial_radius': initial_radius,
        'base_density': base_density,
        'max_density': float(max_density),
        'region_coverage': {k: float(v) for k, v in region_coverage.items()},
        'seed_position': seed_position
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'cells_affected': cells_affected,
                'initial_radius': initial_radius,
                'max_density': float(max_density),
                'region_coverage': {k: float(v) for k, v in region_coverage.items()}
            }
            metrics.record_metrics("mycelial_initialization", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record mycelial initialization metrics: {e}")
    
    return metrics_result


# --- Primary Pathway Establishment ---
def establish_primary_pathways(brain_structure) -> Dict[str, Any]:
    """
    Creates a few primary pathways between major regions.
    These form the backbone of the mycelial network.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with pathway metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info("Establishing primary mycelial pathways between major regions")
    
    # Validate input
    if not hasattr(brain_structure, 'regions'):
        raise ValueError("Invalid brain_structure object. Missing regions attribute.")
    
    if len(brain_structure.regions) < 2:
        logger.warning("Insufficient regions for primary pathways. Need at least 2 regions.")
        return {
            'pathways_created': 0,
            'total_cells': 0,
            'message': "Insufficient regions for primary pathways"
        }
    
    # Define important region pairs
    # These are the critical connections needed for basic brain function
    primary_connections = [
        (REGION_LIMBIC, REGION_BRAIN_STEM),      # Core survival connection
        (REGION_LIMBIC, REGION_FRONTAL),         # Emotion-reasoning connection
        (REGION_LIMBIC, REGION_TEMPORAL),        # Emotion-memory connection
        (REGION_BRAIN_STEM, REGION_CEREBELLUM),  # Movement control connection
        (REGION_FRONTAL, REGION_PARIETAL),       # Reasoning-sensory connection
        (REGION_TEMPORAL, REGION_OCCIPITAL)      # Memory-visual connection
    ]
    
    # Filter to existing regions in brain structure
    valid_connections = []
    for region1, region2 in primary_connections:
        if region1 in brain_structure.regions and region2 in brain_structure.regions:
            valid_connections.append((region1, region2))
    
    if not valid_connections:
        logger.warning("No valid region pairs for primary pathways.")
        return {
            'pathways_created': 0,
            'total_cells': 0,
            'message': "No valid region pairs for primary pathways"
        }
    
    # Track created pathways
    pathways_created = 0
    pathway_metrics = []
    total_cells = 0
    max_cells_per_pathway = 500  # Limit to prevent excessive computation
    
    # Create pathway for each valid connection
    for region1, region2 in valid_connections:
        # Find centers of regions
        if 'center' not in brain_structure.regions[region1] or 'center' not in brain_structure.regions[region2]:
            logger.warning(f"Missing center for regions {region1} or {region2}. Skipping.")
            continue
        
        center1 = brain_structure.regions[region1]['center']
        center2 = brain_structure.regions[region2]['center']
        
        # Calculate direct distance
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        dz = center2[2] - center1[2]
        direct_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Skip if too far
        if direct_dist > MYCELIAL_MAXIMUM_PATHWAY_LENGTH:
            logger.warning(f"Distance between {region1} and {region2} too large ({direct_dist:.2f}). Skipping.")
            continue
        
        # Create curving pathway with slight randomness
        pathway_metrics_entry = create_mycelial_pathway(
            brain_structure, center1, center2, region1, region2, max_cells_per_pathway)
        
        if pathway_metrics_entry['created']:
            pathways_created += 1
            pathway_metrics.append(pathway_metrics_entry)
            total_cells += pathway_metrics_entry['cells_affected']
    
    logger.info(f"Created {pathways_created} primary pathways with {total_cells} total cells")
    
    # Return metrics
    result = {
        'pathways_created': pathways_created,
        'total_cells': total_cells,
        'pathway_metrics': pathway_metrics
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE and pathways_created > 0:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'pathways_created': pathways_created,
                'total_cells': total_cells,
                'region_pairs': [(m['region1'], m['region2']) for m in pathway_metrics]
            }
            metrics.record_metrics("mycelial_primary_pathways", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record primary pathway metrics: {e}")
    
    return result


def create_mycelial_pathway(brain_structure, start_pos: Tuple[int, int, int], 
                          end_pos: Tuple[int, int, int], region1: str, region2: str, 
                          max_cells: int = 500) -> Dict[str, Any]:
    """
    Create a mycelial pathway between two positions.
    
    Args:
        brain_structure: The brain structure object
        start_pos: Start position
        end_pos: End position
        region1: Source region name
        region2: Target region name
        max_cells: Maximum cells to affect (for performance)
        
    Returns:
        Dict with pathway metrics
    """
    # Calculate pathway parameters
    sx, sy, sz = start_pos
    ex, ey, ez = end_pos
    
    # Calculate direct distance
    direct_dist = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
    
    # Skip if too far - mycelial pathways have distance constraints
    if direct_dist > MYCELIAL_MAXIMUM_PATHWAY_LENGTH:
        logger.warning(f"Distance between {region1} and {region2} too large ({direct_dist:.2f}). Skipping.")
        return {
            'created': False,
            'region1': region1,
            'region2': region2,
            'error': "Distance exceeds maximum pathway length"
        }
    
    # Calculate step count based on distance for smooth pathway
    step_count = int(direct_dist * 1.5)  # 50% more steps than distance for smooth curve
    step_count = max(10, min(50, step_count))  # Between 10 and 50 steps
    
    # Generate curve points with slight randomness (controlled chaos)
    curve_points = []
    midpoint_jitter = min(10, direct_dist * 0.2)  # Max 20% of distance
    
    # Calculate midpoint with random jitter
    mx = (sx + ex) / 2 + np.random.uniform(-midpoint_jitter, midpoint_jitter)
    my = (sy + ey) / 2 + np.random.uniform(-midpoint_jitter, midpoint_jitter)
    mz = (sz + ez) / 2 + np.random.uniform(-midpoint_jitter, midpoint_jitter)
    
    # Constrain to brain bounds
    mx = max(0, min(brain_structure.dimensions[0] - 1, mx))
    my = max(0, min(brain_structure.dimensions[1] - 1, my))
    mz = max(0, min(brain_structure.dimensions[2] - 1, mz))
    
    # Generate quadratic Bezier curve through midpoint
    for i in range(step_count + 1):
        t = i / step_count
        
        # Quadratic Bezier curve formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        x = int((1 - t)**2 * sx + 2 * (1 - t) * t * mx + t**2 * ex)
        y = int((1 - t)**2 * sy + 2 * (1 - t) * t * my + t**2 * ey)
        z = int((1 - t)**2 * sz + 2 * (1 - t) * t * mz + t**2 * ez)
        
        # Constrain to brain bounds
        x = max(0, min(brain_structure.dimensions[0] - 1, x))
        y = max(0, min(brain_structure.dimensions[1] - 1, y))
        z = max(0, min(brain_structure.dimensions[2] - 1, z))
        
        curve_points.append((x, y, z))
    
    # Calculate parameters for mycelial growth
    pathway_radius = 1  # Basic pathway thickness
    
    # Mycelial density varies between regions
    start_density = 0.7  # High density at start
    end_density = 0.7    # High density at end
    mid_density = 0.4    # Lower density at midpoint
    
    # Track cells affected
    cells_affected = 0
    added_density = 0.0
    pathway_cells = set()
    
    # Apply mycelial density along pathway
    for i, point in enumerate(curve_points):
        # Skip some points to meet max_cells constraint if needed
        if i % max(1, len(curve_points) // max_cells) != 0:
            continue
            
        x, y, z = point
        
        # Calculate progress along pathway (0 to 1)
        progress = i / len(curve_points)
        
        # Calculate density with natural bell curve distribution (stronger at endpoints)
        # This creates stronger connectivity at region centers and weaker in between
        if progress < 0.5:
            # Start to middle (decreasing)
            t = progress * 2  # 0 to 1
            density = start_density * (1 - t) + mid_density * t
        else:
            # Middle to end (increasing)
            t = (progress - 0.5) * 2  # 0 to 1
            density = mid_density * (1 - t) + end_density * t
        
        # Apply density at this point and small radius around it
        for dx in range(-pathway_radius, pathway_radius + 1):
            for dy in range(-pathway_radius, pathway_radius + 1):
                for dz in range(-pathway_radius, pathway_radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < brain_structure.dimensions[0] and 
                           0 <= ny < brain_structure.dimensions[1] and 
                           0 <= nz < brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from pathway center
                    point_dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if point_dist <= pathway_radius:
                        # Calculate falloff for this point
                        point_falloff = 1.0 - (point_dist / pathway_radius)
                        
                        # Calculate final density for this point
                        point_density = density * point_falloff
                        
                        # Apply to brain structure (max with existing)
                        old_density = brain_structure.mycelial_density_grid[nx, ny, nz]
                        new_density = max(old_density, point_density)
                        
                        # Apply only if it will increase density
                        if new_density > old_density:
                            brain_structure.mycelial_density_grid[nx, ny, nz] = new_density
                            added_density += (new_density - old_density)
                            
                            # Track unique cells affected
                            pathway_cells.add((nx, ny, nz))
                            
                            # Add small coherence boost along pathway
                            brain_structure.coherence_grid[nx, ny, nz] = max(
                                brain_structure.coherence_grid[nx, ny, nz],
                                0.3 * point_density
                            )
                        
                        cells_affected += 1
                        
                        # Break if max cells reached
                        if cells_affected >= max_cells:
                            break
                    
                if cells_affected >= max_cells:
                    break
            
            if cells_affected >= max_cells:
                break
        
        if cells_affected >= max_cells:
            logger.debug(f"Max cells ({max_cells}) reached for pathway. Stopping early.")
            break
    
    # Calculate metrics
    unique_cells = len(pathway_cells)
    avg_added_density = added_density / unique_cells if unique_cells > 0 else 0.0
    
    logger.info(f"Created mycelial pathway from {region1} to {region2} with {unique_cells} unique cells, "
               f"avg added density {avg_added_density:.2f}")
    
    # Return metrics
    return {
        'created': True,
        'region1': region1,
        'region2': region2,
        'direct_distance': float(direct_dist),
        'curve_points': len(curve_points),
        'cells_affected': cells_affected,
        'unique_cells': unique_cells,
        'avg_added_density': float(avg_added_density),
        'pathway_radius': pathway_radius,
        'max_cells_constraint': max_cells
    }


# --- Energy Distribution Setup ---
def setup_energy_distribution_channels(brain_structure) -> Dict[str, Any]:
    """
    Sets up basic channels for energy flow in the mycelial network.
    Creates energy distribution infrastructure.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with channel metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info("Setting up energy distribution channels in mycelial network")
    
    # Validate input
    if not hasattr(brain_structure, 'mycelial_density_grid'):
        raise ValueError("Invalid brain_structure object. Missing mycelial_density_grid attribute.")
    
    # Track energy channels
    channels_created = 0
    energy_nodes = []
    total_energy_capacity = 0.0
    
    # First identify high-density mycelial cells as potential energy nodes
    density_threshold = 0.5  # Only cells with high density
    high_density_indices = np.where(brain_structure.mycelial_density_grid > density_threshold)
    
    if len(high_density_indices[0]) == 0:
        logger.warning("No high-density mycelial cells found for energy channels.")
        return {
            'channels_created': 0,
            'nodes_created': 0,
            'total_capacity': 0.0,
            'message': "No high-density mycelial cells found"
        }
    
    # Sample a subset of high-density cells for energy nodes (for performance)
    max_nodes = 50  # Limit node count
    sample_size = min(max_nodes, len(high_density_indices[0]))
    sample_indices = np.random.choice(len(high_density_indices[0]), sample_size, replace=False)
    
    # Process each energy node
    for i in sample_indices:
        x, y, z = high_density_indices[0][i], high_density_indices[1][i], high_density_indices[2][i]
        
        # Get node properties
        density = brain_structure.mycelial_density_grid[x, y, z]
        region = brain_structure.region_grid[x, y, z]
        
        # Calculate energy capacity based on density
        node_capacity = density * 10.0  # 10 units per density
        
        # Initialize energy node
        node = {
            'position': (x, y, z),
            'density': float(density),
            'capacity': float(node_capacity),
            'region': region,
            'channels': []
        }
        
        # Create energy channels from this node (to nearby high-density cells)
        channels_from_node = create_energy_channels(brain_structure, (x, y, z), node_capacity)
        
        node['channels'] = channels_from_node
        channels_created += len(channels_from_node)
        
        # Add node to list
        energy_nodes.append(node)
        total_energy_capacity += node_capacity
        
        # Set energy capacity in energy grid
        brain_structure.mycelial_energy_grid[x, y, z] = max(
            brain_structure.mycelial_energy_grid[x, y, z],
            node_capacity * 0.2  # Initial 20% fill
        )
    
    logger.info(f"Created {len(energy_nodes)} energy nodes with {channels_created} channels. "
               f"Total capacity: {total_energy_capacity:.2f} BEU.")
    
    # Return metrics
    result = {
        'channels_created': channels_created,
        'nodes_created': len(energy_nodes),
        'total_capacity': float(total_energy_capacity),
        'nodes': energy_nodes[:10]  # Only include first 10 nodes for brevity
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE and len(energy_nodes) > 0:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'channels_created': channels_created,
                'nodes_created': len(energy_nodes),
                'total_capacity': float(total_energy_capacity),
                'avg_capacity_per_node': float(total_energy_capacity / len(energy_nodes))
            }
            metrics.record_metrics("mycelial_energy_channels", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record energy channel metrics: {e}")
    
    return result


def create_energy_channels(brain_structure, node_position: Tuple[int, int, int], 
                         node_capacity: float, max_channels: int = 5, max_radius: int = 10) -> List[Dict[str, Any]]:
    """
    Create energy channels from a node to nearby high-density cells.
    
    Args:
        brain_structure: The brain structure object
        node_position: Position of energy node
        node_capacity: Energy capacity of node
        max_channels: Maximum channels to create per node
        max_radius: Maximum search radius for channel endpoints
        
    Returns:
        List of channel data dictionaries
    """
    x, y, z = node_position
    
    # Find nearby high-density cells as potential channel endpoints
    candidate_endpoints = []
    
    # Search in gradually expanding radius
    for radius in range(3, max_radius + 1, 2):
        # Search spherical shell at this radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # Skip if not near the shell
                    shell_dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    if not (radius - 1 <= shell_dist <= radius + 1):
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < brain_structure.dimensions[0] and 
                           0 <= ny < brain_structure.dimensions[1] and 
                           0 <= nz < brain_structure.dimensions[2]):
                        continue
                    
                    # Check mycelial density
                    density = brain_structure.mycelial_density_grid[nx, ny, nz]
                    
                    # Consider as endpoint if density is high enough
                    if density > 0.3:
                        # Calculate straight-line distance (more precise)
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)
                        
                        # Add candidate
                        candidate_endpoints.append({
                            'position': (nx, ny, nz),
                            'density': float(density),
                            'distance': float(distance),
                            'region': brain_structure.region_grid[nx, ny, nz]
                        })
        
        # If we have enough candidates, stop searching
        if len(candidate_endpoints) >= max_channels * 2:
            break
    
    # Select best endpoints (prioritize high density + different regions)
    selected_endpoints = []
    
    # First, sort by density
    candidate_endpoints.sort(key=lambda e: e['density'], reverse=True)
    
    # Track regions to ensure diversity
    selected_regions = set()
    
    # Select diverse endpoints
    for endpoint in candidate_endpoints:
        # Prioritize different regions first
        if len(selected_endpoints) < max_channels:
            # If we don't have this region yet, or we already have all regions
            if endpoint['region'] not in selected_regions or len(selected_regions) >= len(brain_structure.regions):
                selected_endpoints.append(endpoint)
                selected_regions.add(endpoint['region'])
        
        # Stop if we have enough
        if len(selected_endpoints) >= max_channels:
            break
    
    # Create channels to selected endpoints
    channels = []
    
    for endpoint in selected_endpoints:
        ex, ey, ez = endpoint['position']
        region = endpoint['region']
        distance = endpoint['distance']
        
        # Calculate channel capacity (depends on distance and endpoint density)
        channel_capacity = node_capacity * endpoint['density'] * (1.0 / (1.0 + distance * 0.1))
        
        # Create channel
        channel = {
            'start': node_position,
            'end': endpoint['position'],
            'distance': float(distance),
            'capacity': float(channel_capacity),
            'efficiency': float(MYCELIAL_ENERGY_EFFICIENCY * (0.9 + 0.1 * endpoint['density'])),
            'end_region': region
        }
        
        channels.append(channel)
        
        # Set energy channel in the brain
        # Enhance mycelial density along the channel
        line_points = get_line_points(node_position, endpoint['position'])
        
        for point in line_points:
            px, py, pz = point
            
            # Skip if outside brain bounds
            if not (0 <= px < brain_structure.dimensions[0] and 
                   0 <= py < brain_structure.dimensions[1] and 
                   0 <= pz < brain_structure.dimensions[2]):
                continue
            
            # Enhance mycelial density along channel
            brain_structure.mycelial_density_grid[px, py, pz] = max(
                brain_structure.mycelial_density_grid[px, py, pz],
                0.3  # Minimum density for channels
            )
            
            # Set small initial energy
            brain_structure.mycelial_energy_grid[px, py, pz] = max(
                brain_structure.mycelial_energy_grid[px, py, pz],
                channel_capacity * 0.05  # 5% of capacity as initial energy
            )
        
        # Set endpoint energy capacity
        brain_structure.mycelial_energy_grid[ex, ey, ez] = max(
            brain_structure.mycelial_energy_grid[ex, ey, ez],
            channel_capacity * 0.2  # 20% of capacity at endpoint
        )
    
    return channels


def get_line_points(start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    Get points along a 3D line using Bresenham's line algorithm.
    
    Args:
        start: Start position (x, y, z)
        end: End position (x, y, z)
        
    Returns:
        List of (x, y, z) points along the line
    """
    x1, y1, z1 = start
    x2, y2, z2 = end
    
    points = []
    
    # Calculate distances
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    # Calculate step directions
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    sz = 1 if z2 > z1 else -1
    
    # Initial discriminants
    dm = max(dx, dy, dz)
    i = dm
    x = x1
    y = y1
    z = z1
    
    # Begin loop
    while True:
        points.append((x, y, z))
        
        # Check if we reached the end
        if x == x2 and y == y2 and z == z2:
            break
        
        # Update coordinates
        i1 = 2 * i
        if i1 >= dz:
            i -= dz
            x += sx
        if i1 >= dy:
            i -= dy
            y += sy
        if i1 >= dx:
            i -= dx
            z += sz
    
    return points


# --- Soul-Specific Functions ---
def prepare_for_soul_attachment(brain_structure) -> Dict[str, Any]:
    """
    Prepares the mycelial network for eventual soul connection.
    Sets up special pathways and nodes for soul energy.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with preparation metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info("Preparing mycelial network for soul attachment")
    
    # Validate input
    if not hasattr(brain_structure, 'region_grid'):
        raise ValueError("Invalid brain_structure object. Missing region_grid attribute.")
    
    # Find limbic region (primary soul connection point)
    limbic_indices = np.where(brain_structure.region_grid == REGION_LIMBIC)
    
    if len(limbic_indices[0]) == 0:
        logger.warning("Limbic region not found. Checking brain stem region.")
        
        # Try brain stem as alternative
        brain_stem_indices = np.where(brain_structure.region_grid == REGION_BRAIN_STEM)
        
        if len(brain_stem_indices[0]) == 0:
            raise ValueError("Neither limbic nor brain stem regions found. Cannot prepare for soul attachment.")
        
        soul_region_indices = brain_stem_indices
        soul_region_name = REGION_BRAIN_STEM
    else:
        soul_region_indices = limbic_indices
        soul_region_name = REGION_LIMBIC
    
    # Calculate optimal position within region (highest mycelial density + resonance)
    best_position = None
    best_score = -1.0
    
    for i in range(len(soul_region_indices[0])):
        x, y, z = soul_region_indices[0][i], soul_region_indices[1][i], soul_region_indices[2][i]
        
        # Calculate score based on mycelial density and resonance
        mycelial_density = brain_structure.mycelial_density_grid[x, y, z]
        resonance = brain_structure.resonance_grid[x, y, z]
        
        score = mycelial_density * 0.6 + resonance * 0.4
        
        if score > best_score:
            best_score = score
            best_position = (x, y, z)
    
    if best_position is None:
        raise ValueError(f"No suitable position found in {soul_region_name} region.")
    
    logger.info(f"Found optimal soul attachment position at {best_position} in {soul_region_name} region")
    
    # Create soul attachment node with high capacity
    soul_node_radius = 5
    soul_node_capacity = 100.0  # High capacity for soul energy
    cells_prepared = 0
    
    # Enhance area around soul attachment point
    x, y, z = best_position
    
    for dx in range(-soul_node_radius, soul_node_radius + 1):
        for dy in range(-soul_node_radius, soul_node_radius + 1):
            for dz in range(-soul_node_radius, soul_node_radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                       0 <= ny < brain_structure.dimensions[1] and 
                       0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Calculate distance from center
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist <= soul_node_radius:
                    # Calculate falloff (stronger at center)
                    falloff = 1.0 - (dist / soul_node_radius)
                    
                    # Enhance mycelial density
                    brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_density_grid[nx, ny, nz],
                        0.8 * falloff  # High density
                    )
                    
                    # Enhance resonance (for soul frequency)
                    brain_structure.resonance_grid[nx, ny, nz] = max(
                        brain_structure.resonance_grid[nx, ny, nz],
                        0.9 * falloff  # Very high resonance
                    )
                    
                    # Enhance coherence (for soul coherence)
                    brain_structure.coherence_grid[nx, ny, nz] = max(
                        brain_structure.coherence_grid[nx, ny, nz],
                        0.8 * falloff  # High coherence
                    )
                    
                    # Set capacity in energy grid
                    brain_structure.mycelial_energy_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_energy_grid[nx, ny, nz],
                        soul_node_capacity * falloff * 0.05  # 5% initial fill
                    )
                    
                    # Mark as prepared for soul
                    brain_structure.soul_presence_grid[nx, ny, nz] = max(
                        brain_structure.soul_presence_grid[nx, ny, nz],
                        0.6 * falloff  # Moderate presence before attachment
                    )
                    
                    cells_prepared += 1
    
    # Create connection pathways to key brain regions for soul influence
    target_regions = [
        REGION_FRONTAL,   # For higher reasoning
        REGION_TEMPORAL,  # For memory
        REGION_PARIETAL,  # For perception
        REGION_OCCIPITAL  # For vision
    ]
    
    soul_pathways = []
    
    for target_region in target_regions:
        # Skip if region doesn't exist
        if target_region not in brain_structure.regions:
            continue
        
        # Get target region center
        if 'center' not in brain_structure.regions[target_region]:
            continue
            
        target_center = brain_structure.regions[target_region]['center']
        
        # Create high-capacity pathway
        pathway_metrics = create_soul_influence_pathway(
            brain_structure, best_position, target_center, soul_region_name, target_region)
        
        if pathway_metrics['created']:
            soul_pathways.append(pathway_metrics)
    
    # Record preparation status
    soul_preparation = {
        'position': best_position,
        'region': soul_region_name,
        'node_radius': soul_node_radius,
        'node_capacity': float(soul_node_capacity),
        'cells_prepared': cells_prepared,
        'pathways_created': len(soul_pathways),
        'energy_capacity': float(soul_node_capacity * cells_prepared * 0.05),
        'pathways': soul_pathways
    }
    
    logger.info(f"Prepared {cells_prepared} cells for soul attachment with {len(soul_pathways)} influence pathways")
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'position': best_position,
                'region': soul_region_name,
                'cells_prepared': cells_prepared,
                'pathways_created': len(soul_pathways),
                'energy_capacity': float(soul_node_capacity * cells_prepared * 0.05)
            }
            metrics.record_metrics("mycelial_soul_preparation", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record soul preparation metrics: {e}")
    
    return soul_preparation


def create_soul_influence_pathway(brain_structure, start_pos: Tuple[int, int, int],
                                end_pos: Tuple[int, int, int], start_region: str,
                                end_region: str) -> Dict[str, Any]:
    """
    Create a specialized pathway for soul influence to flow to brain regions.
    These pathways have higher capacity and resonance than regular mycelial pathways.
    
    Args:
        brain_structure: The brain structure object
        start_pos: Soul attachment position
        end_pos: Target region position
        start_region: Soul attachment region name
        end_region: Target region name
        
    Returns:
        Dict with pathway metrics
    """
    logger.info(f"Creating soul influence pathway from {start_region} to {end_region}")
    
    # Calculate pathway parameters
    sx, sy, sz = start_pos
    ex, ey, ez = end_pos
    
    # Calculate direct distance
    direct_dist = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
    
    # Skip if too far
    if direct_dist > MYCELIAL_MAXIMUM_PATHWAY_LENGTH * 1.5:  # Allow 50% further for soul pathways
        logger.warning(f"Distance between {start_region} and {end_region} too large ({direct_dist:.2f}). Skipping.")
        return {
            'created': False,
            'start_region': start_region,
            'end_region': end_region,
            'error': "Distance exceeds maximum pathway length"
        }
    
    # Create straight-line pathway with periodic phi-harmonic nodes
    step_count = min(50, int(direct_dist))
    pathway_radius = 2  # Thicker pathway for soul influence
    
    # Create points along pathway
    pathway_points = []
    for i in range(step_count + 1):
        t = i / step_count
        
        # Linear interpolation
        x = int(sx + t * (ex - sx))
        y = int(sy + t * (ey - sy))
        z = int(sz + t * (ez - sz))
        
        # Constrain to brain bounds
        x = max(0, min(brain_structure.dimensions[0] - 1, x))
        y = max(0, min(brain_structure.dimensions[1] - 1, y))
        z = max(0, min(brain_structure.dimensions[2] - 1, z))
        
        pathway_points.append((x, y, z))
    
    # Soul pathway has phi-harmonic energy nodes
    harmonic_points = []
    for i, point in enumerate(pathway_points):
        # Create phi-harmonic nodes along pathway
        progress = i / len(pathway_points)
        
        # Check if this is a phi-harmonic point
        is_harmonic = False
        for j in range(1, 6):  # Check first 5 phi harmonics
            phi_point = (PHI * j) % 1.0  # Get fractional part
            if abs(progress - phi_point) < 0.05:  # Within 5% of harmonic
                is_harmonic = True
                break
        
        if is_harmonic:
            harmonic_points.append(point)
    
    # Track cells affected
    cells_affected = 0
    pathway_cells = set()
    total_energy_capacity = 0.0
    
    # Apply soul influence along pathway
    for point in pathway_points:
        x, y, z = point
        
        # Different handling for normal points and harmonic points
        is_harmonic = point in harmonic_points
        point_radius = pathway_radius * 2 if is_harmonic else pathway_radius
        
        # Apply influence in radius around point
        for dx in range(-point_radius, point_radius + 1):
            for dy in range(-point_radius, point_radius + 1):
                for dz in range(-point_radius, point_radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < brain_structure.dimensions[0] and 
                           0 <= ny < brain_structure.dimensions[1] and 
                           0 <= nz < brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from pathway
                    point_dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if point_dist <= point_radius:
                        # Calculate falloff
                        falloff = 1.0 - (point_dist / point_radius)
                        
                        # Enhance mycelial properties
                        # Higher values for harmonic points
                        density_boost = 0.9 if is_harmonic else 0.7
                        resonance_boost = 0.9 if is_harmonic else 0.6
                        
                        # Apply enhancements
                        brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                            brain_structure.mycelial_density_grid[nx, ny, nz],
                            density_boost * falloff
                        )
                        
                        brain_structure.resonance_grid[nx, ny, nz] = max(
                            brain_structure.resonance_grid[nx, ny, nz],
                            resonance_boost * falloff
                        )
                        
                        # Set small initial soul presence
                        if is_harmonic:
                            brain_structure.soul_presence_grid[nx, ny, nz] = max(
                                brain_structure.soul_presence_grid[nx, ny, nz],
                                0.3 * falloff
                            )
                        
                        # Add energy capacity at harmonic points
                        if is_harmonic:
                            capacity = 20.0 * falloff  # High capacity at harmonic points
                            brain_structure.mycelial_energy_grid[nx, ny, nz] = max(
                                brain_structure.mycelial_energy_grid[nx, ny, nz],
                                capacity * 0.1  # 10% initial fill
                            )
                            total_energy_capacity += capacity
                        
                        # Track unique cells affected
                        pathway_cells.add((nx, ny, nz))
                        cells_affected += 1
    
    # Calculate metrics
    unique_cells = len(pathway_cells)
    
    logger.info(f"Created soul influence pathway from {start_region} to {end_region} with "
               f"{unique_cells} cells and {len(harmonic_points)} harmonic nodes")
    
    # Return metrics
    return {
        'created': True,
        'start_region': start_region,
        'end_region': end_region,
        'direct_distance': float(direct_dist),
        'cells_affected': cells_affected,
        'unique_cells': unique_cells,
        'harmonic_points': len(harmonic_points),
        'pathway_radius': pathway_radius,
        'energy_capacity': float(total_energy_capacity)
    }


def create_soul_connection_channel(brain_structure, soul_position: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Creates a dedicated channel for soul connection to the brain.
    This channel is the primary conduit for soul energy and information.
    
    Args:
        brain_structure: The brain structure object
        soul_position: Position of soul container
        
    Returns:
        Dict with channel metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info(f"Creating dedicated soul connection channel at position {soul_position}")
    
    # Validate input
    if not hasattr(brain_structure, 'soul_presence_grid'):
        raise ValueError("Invalid brain_structure object. Missing soul_presence_grid attribute.")
    
    x, y, z = soul_position
    if not (0 <= x < brain_structure.dimensions[0] and 
           0 <= y < brain_structure.dimensions[1] and 
           0 <= z < brain_structure.dimensions[2]):
        raise ValueError(f"Soul position {soul_position} out of brain structure bounds")
    
    # Define channel parameters
    channel_radius = 3
    phi_points = 8  # Number of phi-harmonic points
    
    # Create phi-harmonic channel structure (think double-helix DNA-like structure)
    phi_points_coords = []
    
    for i in range(phi_points):
        # Calculate position on phi-spiral
        angle = 2 * math.pi * PHI * i
        
        # Create spiral point
        dx = int(channel_radius * math.cos(angle))
        dy = int(channel_radius * math.sin(angle))
        dz = i  # Increase z with each point
        
        # Ensure within brain bounds
        nx = max(0, min(brain_structure.dimensions[0] - 1, x + dx))
        ny = max(0, min(brain_structure.dimensions[1] - 1, y + dy))
        nz = max(0, min(brain_structure.dimensions[2] - 1, z + dz))
        
        phi_points_coords.append((nx, ny, nz))
    
    # Apply channel properties
    cells_affected = 0
    channel_cells = set()
    total_capacity = 0.0
    
    # First, enhance the center point with high values
    center_radius = 2
    for dx in range(-center_radius, center_radius + 1):
        for dy in range(-center_radius, center_radius + 1):
            for dz in range(-center_radius, center_radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                       0 <= ny < brain_structure.dimensions[1] and 
                       0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Calculate distance from center
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist <= center_radius:
                    # Apply high values at center
                    falloff = 1.0 - (dist / center_radius)
                    
                    # Set very high values at center
                    brain_structure.soul_presence_grid[nx, ny, nz] = max(
                        brain_structure.soul_presence_grid[nx, ny, nz],
                        0.9 * falloff  # Extremely high presence
                    )
                    
                    brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_density_grid[nx, ny, nz],
                        0.95 * falloff  # Extremely high density
                    )
                    
                    brain_structure.resonance_grid[nx, ny, nz] = max(
                        brain_structure.resonance_grid[nx, ny, nz],
                        0.95 * falloff  # Extremely high resonance
                    )
                    
                    # Set high capacity
                    capacity = 200.0 * falloff  # Very high capacity
                    brain_structure.mycelial_energy_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_energy_grid[nx, ny, nz],
                        capacity * 0.1  # 10% initial fill
                    )
                    total_capacity += capacity
                    
                    # Track affected cells
                    channel_cells.add((nx, ny, nz))
                    cells_affected += 1
    
    # Connect phi-harmonic points with pathways
    for i in range(len(phi_points_coords)):
        # Connect to next point (circular)
        next_i = (i + 1) % len(phi_points_coords)
        
        start_point = phi_points_coords[i]
        end_point = phi_points_coords[next_i]
        
        # Create pathway between points
        pathway_points = get_line_points(start_point, end_point)
        
        # Apply enhanced properties along pathway
        for point in pathway_points:
            px, py, pz = point
            
            # Skip if outside brain bounds
            if not (0 <= px < brain_structure.dimensions[0] and 
                   0 <= py < brain_structure.dimensions[1] and 
                   0 <= pz < brain_structure.dimensions[2]):
                continue
            
            # Apply high values along pathway
            brain_structure.soul_presence_grid[px, py, pz] = max(
                brain_structure.soul_presence_grid[px, py, pz],
                0.7  # High presence
            )
            
            brain_structure.mycelial_density_grid[px, py, pz] = max(
                brain_structure.mycelial_density_grid[px, py, pz],
                0.8  # High density
            )
            
            brain_structure.resonance_grid[px, py, pz] = max(
                brain_structure.resonance_grid[px, py, pz],
                0.8  # High resonance
            )
            
            # Add energy capacity
            brain_structure.mycelial_energy_grid[px, py, pz] = max(
                brain_structure.mycelial_energy_grid[px, py, pz],
                15.0 * 0.1  # 10% of 15.0 capacity
            )
            total_capacity += 15.0
            
            # Track affected cells
            channel_cells.add((px, py, pz))
            cells_affected += 1
    
    # Enhance phi points with higher values
    for point in phi_points_coords:
        px, py, pz = point
        
        # Skip if outside brain bounds
        if not (0 <= px < brain_structure.dimensions[0] and 
               0 <= py < brain_structure.dimensions[1] and 
               0 <= pz < brain_structure.dimensions[2]):
            continue
        
        # Apply very high values at phi points
        brain_structure.soul_presence_grid[px, py, pz] = max(
            brain_structure.soul_presence_grid[px, py, pz],
            0.85  # Very high presence
        )
        
        brain_structure.mycelial_density_grid[px, py, pz] = max(
            brain_structure.mycelial_density_grid[px, py, pz],
            0.9  # Very high density
        )
        
        brain_structure.resonance_grid[px, py, pz] = max(
            brain_structure.resonance_grid[px, py, pz],
            0.9  # Very high resonance
        )
        
        # Set high capacity
        capacity = 50.0  # High capacity at phi points
        brain_structure.mycelial_energy_grid[px, py, pz] = max(
            brain_structure.mycelial_energy_grid[px, py, pz],
            capacity * 0.1  # 10% initial fill
        )
        total_capacity += capacity
        
        # Track affected cells
        channel_cells.add((px, py, pz))
        cells_affected += 1
    
    # Calculate metrics
    unique_cells = len(channel_cells)
    
    logger.info(f"Created soul connection channel with {unique_cells} cells and {phi_points} phi-harmonic points. "
               f"Total capacity: {total_capacity:.2f} BEU.")
    
    # Return metrics
    result = {
        'soul_position': soul_position,
        'channel_radius': channel_radius,
        'phi_points': phi_points,
        'cells_affected': cells_affected,
        'unique_cells': unique_cells,
        'total_capacity': float(total_capacity),
        'phi_harmonic_locations': phi_points_coords
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'cells_affected': cells_affected,
                'unique_cells': unique_cells,
                'phi_points': phi_points,
                'total_capacity': float(total_capacity)
            }
            metrics.record_metrics("mycelial_soul_channel", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record soul channel metrics: {e}")
    
    return result


def connect_soul_to_limbic_region(brain_structure, soul_position: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Connects the soul to the limbic region specifically.
    This is the primary emotional and subconscious connection point.
    
    Args:
        brain_structure: The brain structure object
        soul_position: Position of soul container
        
    Returns:
        Dict with connection metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info(f"Connecting soul to limbic region from position {soul_position}")
    
    # Validate inputs
    if not hasattr(brain_structure, 'region_grid'):
        raise ValueError("Invalid brain_structure object. Missing region_grid attribute.")
    
    # Find limbic region
    limbic_indices = np.where(brain_structure.region_grid == REGION_LIMBIC)
    
    if len(limbic_indices[0]) == 0:
        logger.warning("Limbic region not found. Cannot establish connection.")
        return {
            'success': False,
            'message': "Limbic region not found"
        }
    
    # Check if soul position is already in limbic region
    x, y, z = soul_position
    if brain_structure.region_grid[x, y, z] == REGION_LIMBIC:
        logger.info("Soul already positioned in limbic region. Creating local connections.")
        
        # Create local connections within limbic region
        local_connections = create_local_limbic_connections(brain_structure, soul_position)
        
        return {
            'success': True,
            'already_in_limbic': True,
            'local_connections': local_connections
        }
    
    # Find center of limbic region
    limbic_center = (
        int(np.mean(limbic_indices[0])),
        int(np.mean(limbic_indices[1])),
        int(np.mean(limbic_indices[2]))
    )
    
    # Create high-bandwidth connection to limbic region
    connection = create_soul_influence_pathway(
        brain_structure, soul_position, limbic_center, 
        brain_structure.region_grid[x, y, z], REGION_LIMBIC)
    
    if not connection['created']:
        logger.warning("Failed to create connection to limbic region.")
        return {
            'success': False,
            'message': "Failed to create connection pathway"
        }
    
    # Create emotional subcenters (phi-harmonic points within limbic region)
    emotional_centers = []
    
    # Find suitable cells in limbic region
    # Prioritize cells with high resonance
    resonance_threshold = 0.5
    resonant_indices = []
    
    for i in range(len(limbic_indices[0])):
        lx, ly, lz = limbic_indices[0][i], limbic_indices[1][i], limbic_indices[2][i]
        
        # Check resonance
        if brain_structure.resonance_grid[lx, ly, lz] >= resonance_threshold:
            resonant_indices.append((lx, ly, lz))
    
    # If not enough resonant cells, use random cells
    if len(resonant_indices) < 5:
        # Sample random limbic cells
        sample_size = min(100, len(limbic_indices[0]))
        sample_indices = np.random.choice(len(limbic_indices[0]), sample_size, replace=False)
        
        resonant_indices = [
            (limbic_indices[0][i], limbic_indices[1][i], limbic_indices[2][i])
            for i in sample_indices
        ]
    
    # Create emotional centers at phi-harmonic positions
    for i in range(5):  # Create 5 emotional centers
        if i >= len(resonant_indices):
            break
            
        # Get position
        ex, ey, ez = resonant_indices[i]
        
        # Create emotional center
        emotional_center = create_emotional_center(brain_structure, (ex, ey, ez))
        emotional_centers.append(emotional_center)
    
    # Create connections between emotional centers for emotional network
    emotional_pathways = []
    
    for i in range(len(emotional_centers)):
        for j in range(i+1, len(emotional_centers)):
            # Connect centers i and j
            center_i = emotional_centers[i]['position']
            center_j = emotional_centers[j]['position']
            
            # Create pathway between emotional centers
            pathway = create_emotional_pathway(brain_structure, center_i, center_j)
            
            if pathway['created']:
                emotional_pathways.append(pathway)
    
    logger.info(f"Connected soul to limbic region with {len(emotional_centers)} emotional centers "
               f"and {len(emotional_pathways)} emotional pathways")
    
    # Return metrics
    result = {
        'success': True,
        'already_in_limbic': False,
        'connection': connection,
        'limbic_center': limbic_center,
        'emotional_centers': emotional_centers,
        'emotional_pathways': emotional_pathways
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'connection_cells': connection['unique_cells'],
                'emotional_centers': len(emotional_centers),
                'emotional_pathways': len(emotional_pathways)
            }
            metrics.record_metrics("mycelial_limbic_connection", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record limbic connection metrics: {e}")
    
    return result

def create_local_limbic_connections(brain_structure, soul_position: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Create local connections within the limbic region when soul is already positioned there.
    
    Args:
        brain_structure: The brain structure object
        soul_position: Position of soul in limbic region
        
    Returns:
        Dict with connection metrics
    """
    x, y, z = soul_position
    
    # Define search radius
    radius = 20
    
    # Find high-resonance cells in limbic region
    limbic_cells = []
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                       0 <= ny < brain_structure.dimensions[1] and 
                       0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Check if in limbic region
                if brain_structure.region_grid[nx, ny, nz] == REGION_LIMBIC:
                    # Calculate distance
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if dist <= radius and dist > 0:  # Exclude soul position itself
                        # Get resonance
                        resonance = brain_structure.resonance_grid[nx, ny, nz]
                        
                        # Consider cells with good resonance
                        if resonance >= 0.4:
                            limbic_cells.append({
                                'position': (nx, ny, nz),
                                'resonance': float(resonance),
                                'distance': float(dist)
                            })
    
    # Sort by resonance (highest first)
    limbic_cells.sort(key=lambda c: c['resonance'], reverse=True)
    
    # Select top cells for emotional centers (max 5)
    max_centers = 5
    selected_centers = limbic_cells[:max_centers]
    
    # Create emotional centers
    emotional_centers = []
    for center_data in selected_centers:
        position = center_data['position']
        emotional_center = create_emotional_center(brain_structure, position)
        emotional_centers.append(emotional_center)
    
    # Create connections between all emotional centers and the soul
    pathways = []
    
    for center in emotional_centers:
        center_pos = center['position']
        
        # Create pathway from soul to center
        pathway = create_emotional_pathway(brain_structure, soul_position, center_pos)
        
        if pathway['created']:
            pathways.append(pathway)
    
    # Create connections between emotional centers (fully connected network)
    for i in range(len(emotional_centers)):
        for j in range(i+1, len(emotional_centers)):
            center_i_pos = emotional_centers[i]['position']
            center_j_pos = emotional_centers[j]['position']
            
            # Create pathway between centers
            pathway = create_emotional_pathway(brain_structure, center_i_pos, center_j_pos)
            
            if pathway['created']:
                pathways.append(pathway)
    
    logger.info(f"Created local limbic connections with {len(emotional_centers)} emotional centers "
               f"and {len(pathways)} emotional pathways")
    
    # Return metrics
    return {
        'emotional_centers': emotional_centers,
        'pathways': pathways,
        'total_centers': len(emotional_centers),
        'total_pathways': len(pathways)
    }


def create_emotional_center(brain_structure, position: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Create an emotional center node within the limbic system.
    These are specialized nodes for emotional processing.
    
    Args:
        brain_structure: The brain structure object
        position: Position for emotional center
        
    Returns:
        Dict with center metrics
    """
    x, y, z = position
    center_radius = 2
    
    # Track affected cells
    cells_affected = 0
    total_capacity = 0.0
    
    # Enhance area around emotional center
    for dx in range(-center_radius, center_radius + 1):
        for dy in range(-center_radius, center_radius + 1):
            for dz in range(-center_radius, center_radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                       0 <= ny < brain_structure.dimensions[1] and 
                       0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Calculate distance from center
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist <= center_radius:
                    # Calculate falloff
                    falloff = 1.0 - (dist / center_radius)
                    
                    # Enhance properties
                    brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_density_grid[nx, ny, nz],
                        0.8 * falloff  # High density
                    )
                    
                    brain_structure.resonance_grid[nx, ny, nz] = max(
                        brain_structure.resonance_grid[nx, ny, nz],
                        0.85 * falloff  # Very high resonance for emotions
                    )
                    
                    # Set energy capacity
                    capacity = 30.0 * falloff  # Good capacity for emotional processing
                    brain_structure.mycelial_energy_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_energy_grid[nx, ny, nz],
                        capacity * 0.1  # 10% initial fill
                    )
                    total_capacity += capacity
                    
                    cells_affected += 1
    
    # Return metrics
    return {
        'position': position,
        'radius': center_radius,
        'cells_affected': cells_affected,
        'total_capacity': float(total_capacity)
    }


def create_emotional_pathway(brain_structure, start_pos: Tuple[int, int, int], 
                          end_pos: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Create a pathway specifically for emotional information flow.
    These have different characteristics from standard pathways.
    
    Args:
        brain_structure: The brain structure object
        start_pos: Start position
        end_pos: End position
        
    Returns:
        Dict with pathway metrics
    """
    # Calculate direct distance
    sx, sy, sz = start_pos
    ex, ey, ez = end_pos
    
    direct_dist = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
    
    # Skip if too far
    if direct_dist > MYCELIAL_MAXIMUM_PATHWAY_LENGTH:
        return {
            'created': False,
            'error': "Distance exceeds maximum pathway length"
        }
    
    # Create curved pathway (emotions flow in waves)
    pathway_radius = 1
    
    # Calculate step count - use golden ratio for natural wave
    step_count = int(direct_dist * (1 + 1/PHI))
    step_count = max(10, min(40, step_count))
    
    # Create wave-like pathway using sine wave
    pathway_points = []
    
    for i in range(step_count + 1):
        t = i / step_count
        
        # Linear interpolation with sine wave offset
        wave_amplitude = direct_dist * 0.1  # 10% of distance
        wave_frequency = 3  # 3 cycles along pathway
        
        # Calculate base position with linear interpolation
        base_x = sx + t * (ex - sx)
        base_y = sy + t * (ey - sy)
        base_z = sz + t * (ez - sz)
        
        # Calculate direction perpendicular to pathway
        if abs(ex - sx) > abs(ey - sy):
            # Pathway is more horizontal, so wave vertically
            wave_offset = wave_amplitude * math.sin(wave_frequency * 2 * math.pi * t)
            base_y += wave_offset
        else:
            # Pathway is more vertical, so wave horizontally
            wave_offset = wave_amplitude * math.sin(wave_frequency * 2 * math.pi * t)
            base_x += wave_offset
        
        # Convert to integer coordinates
        x = int(base_x)
        y = int(base_y)
        z = int(base_z)
        
        # Constrain to brain bounds
        x = max(0, min(brain_structure.dimensions[0] - 1, x))
        y = max(0, min(brain_structure.dimensions[1] - 1, y))
        z = max(0, min(brain_structure.dimensions[2] - 1, z))
        
        pathway_points.append((x, y, z))
    
    # Track cells affected
    cells_affected = 0
    pathway_cells = set()
    
    # Apply pathway properties
    for point in pathway_points:
        x, y, z = point
        
        # Create a small radius around this point
        for dx in range(-pathway_radius, pathway_radius + 1):
            for dy in range(-pathway_radius, pathway_radius + 1):
                for dz in range(-pathway_radius, pathway_radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < brain_structure.dimensions[0] and 
                           0 <= ny < brain_structure.dimensions[1] and 
                           0 <= nz < brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from pathway center
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if dist <= pathway_radius:
                        # Calculate falloff
                        falloff = 1.0 - (dist / pathway_radius)
                        
                        # Enhance properties - emotional pathways have high resonance
                        brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                            brain_structure.mycelial_density_grid[nx, ny, nz],
                            0.7 * falloff
                        )
                        
                        brain_structure.resonance_grid[nx, ny, nz] = max(
                            brain_structure.resonance_grid[nx, ny, nz],
                            0.75 * falloff
                        )
                        
                        # Track unique cells
                        pathway_cells.add((nx, ny, nz))
                        cells_affected += 1
    
    # Calculate pathway length (following the wave)
    actual_length = 0.0
    for i in range(len(pathway_points) - 1):
        p1 = pathway_points[i]
        p2 = pathway_points[i + 1]
        
        segment_length = math.sqrt(
            (p2[0] - p1[0])**2 + 
            (p2[1] - p1[1])**2 + 
            (p2[2] - p1[2])**2
        )
        
        actual_length += segment_length
    
    # Return metrics
    return {
        'created': True,
        'start': start_pos,
        'end': end_pos,
        'direct_distance': float(direct_dist),
        'actual_length': float(actual_length),
        'cells_affected': cells_affected,
        'unique_cells': len(pathway_cells),
        'pathway_radius': pathway_radius,
        'step_count': step_count
    }


# --- Memory Fragment Handling ---
def store_soul_aspect_fragment(brain_structure, aspect_data: Dict[str, Any], 
                             temporal_region_position: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
    """
    Store a soul aspect as a memory fragment in the temporal region.
    
    Args:
        brain_structure: The brain structure object
        aspect_data: Soul aspect data dictionary
        temporal_region_position: Optional specific position in temporal region
        
    Returns:
        Dict with storage metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info("Storing soul aspect as memory fragment in temporal region")
    
    # Validate input
    if not hasattr(brain_structure, 'region_grid'):
        raise ValueError("Invalid brain_structure object. Missing region_grid attribute.")
    
    if not isinstance(aspect_data, dict):
        raise ValueError("Invalid aspect_data. Must be a dictionary.")
    
    # Find temporal region if position not provided
    if temporal_region_position is None:
        temporal_indices = np.where(brain_structure.region_grid == REGION_TEMPORAL)
        
        if len(temporal_indices[0]) == 0:
            raise ValueError("Temporal region not found in brain structure.")
        
        # Find optimal position in temporal region (high resonance)
        best_position = None
        best_resonance = -1.0
        
        # Sample points for efficiency
        sample_size = min(100, len(temporal_indices[0]))
        sample_indices = np.random.choice(len(temporal_indices[0]), sample_size, replace=False)
        
        for i in sample_indices:
            x, y, z = temporal_indices[0][i], temporal_indices[1][i], temporal_indices[2][i]
            
            # Get resonance
            resonance = brain_structure.resonance_grid[x, y, z]
            
            if resonance > best_resonance:
                best_resonance = resonance
                best_position = (x, y, z)
        
        if best_position is None:
            # Fallback to first temporal cell
            best_position = (temporal_indices[0][0], temporal_indices[1][0], temporal_indices[2][0])
        
        temporal_region_position = best_position
    
    # Verify position is in temporal region
    x, y, z = temporal_region_position
    if not (0 <= x < brain_structure.dimensions[0] and 
           0 <= y < brain_structure.dimensions[1] and 
           0 <= z < brain_structure.dimensions[2]):
        raise ValueError(f"Position {temporal_region_position} out of brain structure bounds")
    
    if brain_structure.region_grid[x, y, z] != REGION_TEMPORAL:
        logger.warning(f"Position {temporal_region_position} is not in temporal region. "
                     f"Using anyway but this may affect memory integration.")
    
    # Create memory fragment container
    fragment_radius = 3
    cells_affected = 0
    fragment_cells = set()
    
    # Get aspect properties
    aspect_id = aspect_data.get('id', str(uuid.uuid4()))
    aspect_frequency = aspect_data.get('frequency', DEFAULT_BRAIN_SEED_FREQUENCY)
    aspect_energy = aspect_data.get('energy', 1.0)
    
    # Apply fragment properties in radius around position
    for dx in range(-fragment_radius, fragment_radius + 1):
        for dy in range(-fragment_radius, fragment_radius + 1):
            for dz in range(-fragment_radius, fragment_radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                       0 <= ny < brain_structure.dimensions[1] and 
                       0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Calculate distance from center
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist <= fragment_radius:
                    # Calculate falloff
                    falloff = 1.0 - (dist / fragment_radius)
                    
                    # Set aspect frequency in grid
                    # Memory fragments have specific frequencies
                    brain_structure.frequency_grid[nx, ny, nz] = max(
                        brain_structure.frequency_grid[nx, ny, nz] * 0.5,  # Reduce existing
                        aspect_frequency * falloff  # Apply aspect frequency
                    )
                    
                    # Enhance mycelial density for memory storage
                    brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_density_grid[nx, ny, nz],
                        0.7 * falloff  # High density for memory
                    )
                    
                    # Set small soul presence (aspect contains soul information)
                    brain_structure.soul_presence_grid[nx, ny, nz] = max(
                        brain_structure.soul_presence_grid[nx, ny, nz],
                        0.3 * falloff  # Low presence
                    )
                    
                    # Add energy for memory activation
                    brain_structure.mycelial_energy_grid[nx, ny, nz] = max(
                        brain_structure.mycelial_energy_grid[nx, ny, nz],
                        aspect_energy * falloff * 0.2  # 20% initial energy
                    )
                    
                    # Track cells
                    fragment_cells.add((nx, ny, nz))
                    cells_affected += 1
    
    logger.info(f"Stored soul aspect {aspect_id} as memory fragment with {cells_affected} cells")
    
    # Return metrics
    result = {
        'aspect_id': aspect_id,
        'position': temporal_region_position,
        'region': brain_structure.region_grid[x, y, z],
        'frequency': float(aspect_frequency),
        'cells_affected': cells_affected,
        'fragment_radius': fragment_radius,
        'unique_cells': len(fragment_cells)
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'aspect_id': aspect_id,
                'position': temporal_region_position,
                'frequency': float(aspect_frequency),
                'cells_affected': cells_affected
            }
            metrics.record_metrics("mycelial_soul_aspect_storage", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record soul aspect storage metrics: {e}")
    
    return result


def assign_fragment_frequency(brain_structure, fragment_position: Tuple[int, int, int], 
                            frequency: float, radius: int = 3) -> Dict[str, Any]:
    """
    Assign a specific frequency to a memory fragment.
    This sets the resonant frequency for this fragment.
    
    Args:
        brain_structure: The brain structure object
        fragment_position: Position of memory fragment
        frequency: Frequency to assign (Hz)
        radius: Radius of effect
        
    Returns:
        Dict with assignment metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info(f"Assigning frequency {frequency:.2f} Hz to memory fragment at {fragment_position}")
    
    # Validate inputs
    if not hasattr(brain_structure, 'frequency_grid'):
        raise ValueError("Invalid brain_structure object. Missing frequency_grid attribute.")
    
    if not isinstance(frequency, (int, float)) or frequency <= 0:
        raise ValueError(f"Invalid frequency: {frequency}. Must be positive.")
    
    x, y, z = fragment_position
    if not (0 <= x < brain_structure.dimensions[0] and 
           0 <= y < brain_structure.dimensions[1] and 
           0 <= z < brain_structure.dimensions[2]):
        raise ValueError(f"Position {fragment_position} out of brain structure bounds")
    
    # Track cells affected
    cells_affected = 0
    
    # Apply frequency in radius around position
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                       0 <= ny < brain_structure.dimensions[1] and 
                       0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Calculate distance from center
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist <= radius:
                    # Calculate falloff
                    falloff = 1.0 - (dist / radius)
                    
                    # Set frequency in grid
                    brain_structure.frequency_grid[nx, ny, nz] = (
                        frequency * falloff + 
                        brain_structure.frequency_grid[nx, ny, nz] * (1.0 - falloff)
                    )
                    
                    cells_affected += 1
    
    logger.info(f"Assigned frequency to {cells_affected} cells around fragment position")
    
    # Return metrics
    return {
        'position': fragment_position,
        'frequency': float(frequency),
        'radius': radius,
        'cells_affected': cells_affected,
        'region': brain_structure.region_grid[x, y, z]
    }


def place_fragment_in_temporal_region(brain_structure, fragment_id: str) -> Dict[str, Any]:
    """
    Place a memory fragment in the temporal region.
    This function is a placeholder for actual fragment placement.
    
    Args:
        brain_structure: The brain structure object
        fragment_id: ID of the memory fragment
        
    Returns:
        Dict with placement metrics
    """
    logger.info(f"Placing memory fragment {fragment_id} in temporal region")
    
    # This is a placeholder function since actual fragment data
    # would be handled by the memory_fragment_system
    
    # Find temporal region
    temporal_indices = np.where(brain_structure.region_grid == REGION_TEMPORAL)
    
    if len(temporal_indices[0]) == 0:
        logger.warning("Temporal region not found. Cannot place fragment.")
        return {
            'success': False,
            'message': "Temporal region not found"
        }
    
    # Find optimal position (high resonance)
    best_position = None
    best_resonance = -1.0
    
    # Sample points for efficiency
    sample_size = min(100, len(temporal_indices[0]))
    sample_indices = np.random.choice(len(temporal_indices[0]), sample_size, replace=False)
    
    for i in sample_indices:
        x, y, z = temporal_indices[0][i], temporal_indices[1][i], temporal_indices[2][i]
        
        # Get resonance
        resonance = brain_structure.resonance_grid[x, y, z]
        
        if resonance > best_resonance:
            best_resonance = resonance
            best_position = (x, y, z)
    
    if best_position is None:
        # Fallback to first temporal cell
        best_position = (temporal_indices[0][0], temporal_indices[1][0], temporal_indices[2][0])
    
    # Return placement information
    return {
        'success': True,
        'fragment_id': fragment_id,
        'position': best_position,
        'region': REGION_TEMPORAL,
        'resonance': float(best_resonance)
    }


# --- Energy Management Functions ---
def distribute_initial_energy(brain_structure, brain_seed, energy_amount: float) -> Dict[str, Any]:
    """
    Distribute initial energy throughout the brain structure from seed.
    Sets up initial energy distribution for Stage 1.
    
    Args:
        brain_structure: The brain structure object
        brain_seed: The brain seed object
        energy_amount: Amount of energy to distribute
        
    Returns:
        Dict with distribution metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info(f"Distributing initial energy of {energy_amount:.2f} BEU from brain seed")
    
    # Validate inputs
    if not hasattr(brain_structure, 'energy_grid'):
        raise ValueError("Invalid brain_structure object. Missing energy_grid attribute.")
    
    if not hasattr(brain_seed, 'position'):
        raise ValueError("Invalid brain_seed object. Missing position attribute.")
    
    if not isinstance(energy_amount, (int, float)) or energy_amount <= 0:
        raise ValueError(f"Invalid energy amount: {energy_amount}. Must be positive.")
    
    # Get seed position
    seed_position = brain_seed.position
    if seed_position is None:
        raise ValueError("Brain seed position not set.")
    
    x, y, z = seed_position
    
    # Use energy from seed
    energy_result = brain_seed.use_energy(energy_amount, "initial_distribution")
    
    if not energy_result.get('success', False):
        logger.warning(f"Failed to use energy from seed: {energy_result.get('message', 'Unknown error')}")
        return {
            'success': False,
            'message': energy_result.get('message', 'Failed to use energy from seed')
        }
    
    # Define distribution parameters
    distribution_radius = min(50, max(20, int(energy_amount)))  # Scale with energy
    energy_distributed = 0.0
    cells_energized = 0
    
    # Define region weights (importance for energy)
    region_weights = {
        REGION_LIMBIC: 1.0,       # Highest priority
        REGION_BRAIN_STEM: 0.9,   # Very high priority
        REGION_FRONTAL: 0.7,      # High priority
        REGION_TEMPORAL: 0.7,     # High priority
        REGION_PARIETAL: 0.6,     # Medium priority
        REGION_OCCIPITAL: 0.6,    # Medium priority
        REGION_CEREBELLUM: 0.5    # Lower priority
    }
    
    # Track energy by region
    region_energy = {}
    
    # Create energy distribution via distance from seed and region importance
    for dx in range(-distribution_radius, distribution_radius + 1):
        for dy in range(-distribution_radius, distribution_radius + 1):
            for dz in range(-distribution_radius, distribution_radius + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Skip if outside brain bounds
                if not (0 <= nx < brain_structure.dimensions[0] and 
                       0 <= ny < brain_structure.dimensions[1] and 
                       0 <= nz < brain_structure.dimensions[2]):
                    continue
                
                # Calculate distance from seed
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist <= distribution_radius:
                    # Calculate basic energy distribution
                    # Use exponential decay for natural falloff
                    energy_falloff = math.exp(-dist / (distribution_radius / 3))
                    
                    # Scale by region importance
                    region = brain_structure.region_grid[nx, ny, nz]
                    region_factor = region_weights.get(region, 0.5) if region else 0.2
                    
                    # Calculate cell energy
                    cell_energy = energy_amount * energy_falloff * region_factor * 0.001
                    
                    # Skip if negligible
                    if cell_energy < 0.001:
                        continue
                    
                    # Apply to brain structure
                    brain_structure.energy_grid[nx, ny, nz] += cell_energy
                    
                    # Track distribution
                    energy_distributed += cell_energy
                    cells_energized += 1
                    
                    # Track by region
                    if region:
                        if region not in region_energy:
                            region_energy[region] = 0.0
                        region_energy[region] += cell_energy
    
    logger.info(f"Distributed {energy_distributed:.2f} BEU to {cells_energized} cells "
               f"({energy_distributed/energy_amount:.1%} efficiency)")
    
    # Return metrics
    result = {
        'success': True,
        'energy_used': float(energy_amount),
        'energy_distributed': float(energy_distributed),
        'efficiency': float(energy_distributed/energy_amount),
        'cells_energized': cells_energized,
        'distribution_radius': distribution_radius,
        'region_energy': {k: float(v) for k, v in region_energy.items()}
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'energy_used': float(energy_amount),
                'energy_distributed': float(energy_distributed),
                'efficiency': float(energy_distributed/energy_amount),
                'cells_energized': cells_energized,
                'region_distribution': {k: float(v) for k, v in region_energy.items()}
            }
            metrics.record_metrics("mycelial_energy_distribution", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record energy distribution metrics: {e}")
    
    return result


def establish_energy_store(brain_structure, capacity: float) -> Dict[str, Any]:
    """
    Establish an energy storage system in the mycelial network.
    Creates distributed storage system within the network.
    
    Args:
        brain_structure: The brain structure object
        capacity: Desired energy storage capacity
        
    Returns:
        Dict with storage metrics
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info(f"Establishing mycelial energy storage system with capacity {capacity:.2f} BEU")
    
    # Validate inputs
    if not hasattr(brain_structure, 'mycelial_density_grid'):
        raise ValueError("Invalid brain_structure object. Missing mycelial_density_grid attribute.")
    
    if not isinstance(capacity, (int, float)) or capacity <= 0:
        raise ValueError(f"Invalid capacity: {capacity}. Must be positive.")
    
    # Find cells with high mycelial density for storage
    density_threshold = 0.5  # Only cells with high density
    storage_cells = np.where(brain_structure.mycelial_density_grid > density_threshold)
    
    if len(storage_cells[0]) == 0:
        logger.warning("No suitable cells found for energy storage.")
        return {
            'success': False,
            'message': "No suitable cells found for energy storage"
        }
    
    # Calculate capacity per cell
    cells_count = len(storage_cells[0])
    capacity_per_cell = capacity / cells_count
    
    # Create storage nodes
    storage_nodes = []
    actual_capacity = 0.0
    
    for i in range(cells_count):
        x, y, z = storage_cells[0][i], storage_cells[1][i], storage_cells[2][i]
        
        # Get cell density and calculate node capacity
        density = brain_structure.mycelial_density_grid[x, y, z]
        node_capacity = capacity_per_cell * density
        
        # Set energy capacity in grid
        # Use 5% initial fill
        brain_structure.mycelial_energy_grid[x, y, z] = max(
            brain_structure.mycelial_energy_grid[x, y, z],
            node_capacity * 0.05
        )
        
        # Add storage node
        storage_nodes.append({
            'position': (x, y, z),
            'capacity': float(node_capacity),
            'density': float(density),
            'region': brain_structure.region_grid[x, y, z]
        })
        
        actual_capacity += node_capacity
    
    logger.info(f"Established energy storage system with {len(storage_nodes)} nodes "
               f"and {actual_capacity:.2f} BEU total capacity")
    
    # Track storage by region
    region_storage = {}
    for node in storage_nodes:
        region = node['region']
        if region:
            if region not in region_storage:
                region_storage[region] = {
                    'nodes': 0,
                    'capacity': 0.0
                }
            region_storage[region]['nodes'] += 1
            region_storage[region]['capacity'] += node['capacity']
    
    # Return metrics
    result = {
        'success': True,
        'storage_nodes': len(storage_nodes),
        'total_capacity': float(actual_capacity),
        'average_capacity_per_node': float(actual_capacity / len(storage_nodes)),
        'region_storage': region_storage
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'storage_nodes': len(storage_nodes),
                'total_capacity': float(actual_capacity),
                'region_storage': {r: float(d['capacity']) for r, d in region_storage.items()}
            }
            metrics.record_metrics("mycelial_energy_store", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record energy store metrics: {e}")
    
    return result


def setup_energy_conservation_protocol(brain_structure) -> Dict[str, Any]:
    """
    Set up energy conservation protocols in the mycelial network.
    This creates pathways and mechanisms to efficiently use energy.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with protocol metrics
    """
    logger.info("Setting up energy conservation protocols in mycelial network")
    
    # Define key protocols and their efficiency values
    conservation_protocols = [
        {
            'name': 'resonance_amplification',
            'description': 'Amplifies weak signals through resonance',
            'efficiency_gain': 0.15,  # 15% efficiency improvement
            'required_resonance': 0.7
        },
        {
            'name': 'field_coherence_enhancement',
            'description': 'Enhances field coherence to reduce energy loss',
            'efficiency_gain': 0.2,  # 20% efficiency improvement
            'required_coherence': 0.6
        },
        {
            'name': 'harmonic_frequency_alignment',
            'description': 'Aligns frequencies to harmonic patterns for efficiency',
            'efficiency_gain': 0.12,  # 12% efficiency improvement
            'min_frequency': 5.0
        },
        {
            'name': 'phi_pattern_distribution',
            'description': 'Distributes energy in phi-harmonic patterns',
            'efficiency_gain': 0.18,  # 18% efficiency improvement
            'min_density': 0.5
        }
    ]
    
    # Track protocol implementation
    implemented_protocols = []
    total_efficiency_gain = 0.0
    cells_affected = 0
    
    # Implement each protocol where applicable
    for protocol in conservation_protocols:
        protocol_cells = 0
        efficiency_gain = 0.0
        
        # Search for cells that meet protocol requirements
        if protocol['name'] == 'resonance_amplification':
            # Find high-resonance cells
            high_res_cells = np.where(brain_structure.resonance_grid > protocol['required_resonance'])
            
            if len(high_res_cells[0]) > 0:
                # Enhance these cells for efficiency
                for i in range(len(high_res_cells[0])):
                    x, y, z = high_res_cells[0][i], high_res_cells[1][i], high_res_cells[2][i]
                    
                    # Apply efficiency boost
                    # For Stage 1, this is represented by enhanced resonance
                    # Actual efficiency would be calculated by mycelial system
                    brain_structure.resonance_grid[x, y, z] = min(
                        1.0,
                        brain_structure.resonance_grid[x, y, z] * (1 + protocol['efficiency_gain'] * 0.5)
                    )
                    
                    protocol_cells += 1
                
                # Calculate protocol efficiency gain
                efficiency_gain = protocol['efficiency_gain'] * protocol_cells / brain_structure.total_grid_cells
                
                # Add to implemented protocols
                implemented_protocols.append({
                    'name': protocol['name'],
                    'cells_affected': protocol_cells,
                    'efficiency_gain': float(efficiency_gain)
                })
                
                total_efficiency_gain += efficiency_gain
                cells_affected += protocol_cells
                
        elif protocol['name'] == 'field_coherence_enhancement':
            # Find high-coherence cells
            high_coh_cells = np.where(brain_structure.coherence_grid > protocol['required_coherence'])
            
            if len(high_coh_cells[0]) > 0:
                # Enhance these cells for efficiency
                for i in range(len(high_coh_cells[0])):
                    x, y, z = high_coh_cells[0][i], high_coh_cells[1][i], high_coh_cells[2][i]
                    
                    # Apply efficiency boost
                    brain_structure.coherence_grid[x, y, z] = min(
                        1.0,
                        brain_structure.coherence_grid[x, y, z] * (1 + protocol['efficiency_gain'] * 0.5)
                    )
                    
                    protocol_cells += 1
                
                # Calculate protocol efficiency gain
                efficiency_gain = protocol['efficiency_gain'] * protocol_cells / brain_structure.total_grid_cells
                
                # Add to implemented protocols
                implemented_protocols.append({
                    'name': protocol['name'],
                    'cells_affected': protocol_cells,
                    'efficiency_gain': float(efficiency_gain)
                })
                
                total_efficiency_gain += efficiency_gain
                cells_affected += protocol_cells
                
        elif protocol['name'] == 'harmonic_frequency_alignment':
            # Find cells with sufficient frequency
            freq_cells = np.where(brain_structure.frequency_grid > protocol['min_frequency'])
            
            if len(freq_cells[0]) > 0:
                # Sample a subset for processing
                sample_size = min(10000, len(freq_cells[0]))
                sample_indices = np.random.choice(len(freq_cells[0]), sample_size, replace=False)
                
                for i in sample_indices:
                    x, y, z = freq_cells[0][i], freq_cells[1][i], freq_cells[2][i]
                    
                    # Get current frequency
                    frequency = brain_structure.frequency_grid[x, y, z]
                    
                    # Find nearest harmonic frequency
                    base_freq = SCHUMANN_FREQUENCY
                    harmonic_ratios = [0.5, 1.0, 1.5, 2.0, PHI, 1/PHI]
                    
                    best_harmonic = None
                    min_distance = float('inf')
                    
                    for ratio in harmonic_ratios:
                        harmonic = base_freq * ratio
                        distance = abs(frequency - harmonic)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_harmonic = harmonic
                    
                    # Only adjust if reasonably close to harmonic
                    if min_distance < 1.0:
                        # Adjust frequency toward harmonic
                        brain_structure.frequency_grid[x, y, z] = (
                            frequency * 0.7 + best_harmonic * 0.3
                        )
                        
                        protocol_cells += 1
                
                # Calculate protocol efficiency gain
                efficiency_gain = protocol['efficiency_gain'] * protocol_cells / brain_structure.total_grid_cells
                
                # Add to implemented protocols
                implemented_protocols.append({
                    'name': protocol['name'],
                    'cells_affected': protocol_cells,
                    'efficiency_gain': float(efficiency_gain)
                })
                
                total_efficiency_gain += efficiency_gain
                cells_affected += protocol_cells
                
        elif protocol['name'] == 'phi_pattern_distribution':
            # Find cells with sufficient mycelial density
            density_cells = np.where(brain_structure.mycelial_density_grid > protocol['min_density'])
            
            if len(density_cells[0]) > 0:
                # Create phi-pattern distribution for efficient energy flow
                # Calculate clusters of cells in phi-harmonic distances
                
                # First find a few seed points for pattern
                sample_size = min(5, len(density_cells[0]))
                sample_indices = np.random.choice(len(density_cells[0]), sample_size, replace=False)
                
                pattern_seeds = [
                    (density_cells[0][i], density_cells[1][i], density_cells[2][i])
                    for i in sample_indices
                ]
                
                # Create phi-harmonic patterns around each seed
                for seed in pattern_seeds:
                    sx, sy, sz = seed
                    
                    # Generate points at phi-harmonic distances
                    phi_points = []
                    radius = 20  # Maximum radius for pattern
                    
                    for i in range(1, 6):  # 5 phi harmonics
                        distance = radius * (1 - 1 / (i * PHI))
                        
                        # Create points at this distance
                        points_at_distance = 5  # Number of points at each distance
                        
                        for j in range(points_at_distance):
                            # Calculate position on phi-spiral
                            angle = 2 * math.pi * j / points_at_distance
                            
                            # Create spiral point
                            px = int(sx + distance * math.cos(angle))
                            py = int(sy + distance * math.sin(angle))
                            pz = sz
                            
                            # Ensure within brain bounds
                            if (0 <= px < brain_structure.dimensions[0] and 
                                0 <= py < brain_structure.dimensions[1] and 
                                0 <= pz < brain_structure.dimensions[2]):
                                phi_points.append((px, py, pz))
                    
                    # Enhance points in phi-harmonic pattern
                    for point in phi_points:
                        px, py, pz = point
                        
                        # Enhance mycelial network at these points
                        brain_structure.mycelial_density_grid[px, py, pz] = max(
                            brain_structure.mycelial_density_grid[px, py, pz],
                            0.7  # High density
                        )
                        
                        # Enhance resonance for efficiency
                        brain_structure.resonance_grid[px, py, pz] = max(
                            brain_structure.resonance_grid[px, py, pz],
                            0.75  # High resonance
                        )
                        
                        protocol_cells += 1
                    
                    # Connect phi points with pathways
                    for i in range(len(phi_points)):
                        # Connect to next point (circular)
                        next_i = (i + 1) % len(phi_points)
                        
                        start_point = phi_points[i]
                        end_point = phi_points[next_i]
                        
                        # Create pathway between points
                        pathway_points = get_line_points(start_point, end_point)
                        
                        # Enhance points along pathway
                        for point in pathway_points:
                            px, py, pz = point
                            
                            # Skip if outside brain bounds
                            if not (0 <= px < brain_structure.dimensions[0] and 
                                   0 <= py < brain_structure.dimensions[1] and 
                                   0 <= pz < brain_structure.dimensions[2]):
                                continue
                            
                            # Enhance mycelial density
                            brain_structure.mycelial_density_grid[px, py, pz] = max(
                                brain_structure.mycelial_density_grid[px, py, pz],
                                0.6  # Good density
                            )
                            
                            protocol_cells += 1
                
                # Calculate protocol efficiency gain
                efficiency_gain = protocol['efficiency_gain'] * protocol_cells / brain_structure.total_grid_cells
                
                # Add to implemented protocols
                implemented_protocols.append({
                    'name': protocol['name'],
                    'cells_affected': protocol_cells,
                    'efficiency_gain': float(efficiency_gain)
                })
                
                total_efficiency_gain += efficiency_gain
                cells_affected += protocol_cells
    
    logger.info(f"Established {len(implemented_protocols)} energy conservation protocols "
               f"with {total_efficiency_gain:.2f} total efficiency gain")
    
    # Return metrics
    result = {
        'protocols_implemented': len(implemented_protocols),
        'total_efficiency_gain': float(total_efficiency_gain),
        'cells_affected': cells_affected,
        'protocols': implemented_protocols
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'protocols_implemented': len(implemented_protocols),
                'total_efficiency_gain': float(total_efficiency_gain),
                'cells_affected': cells_affected,
                'protocol_details': {p['name']: float(p['efficiency_gain']) for p in implemented_protocols}
            }
            metrics.record_metrics("mycelial_energy_conservation", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record energy conservation metrics: {e}")
    
    return result


# --- State Transition Setup ---
def initialize_liminal_state(brain_structure) -> Dict[str, Any]:
    """
    Initialize the liminal consciousness state.
    This is the initial state before dream or awareness.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with initialization metrics
    """
    logger.info("Initializing liminal consciousness state")
    
    # Define liminal state parameters
    liminal_frequency = 3.5  # Delta/theta boundary
    liminal_coherence = 0.4  # Moderate coherence
    liminal_resonance = 0.5  # Moderate resonance
    cells_affected = 0
    
    # Apply liminal state properties to brain
    # This state affects all cells with some energy/mycelial density
    active_cells = np.where(
        (brain_structure.energy_grid > 0.1) & 
        (brain_structure.mycelial_density_grid > 0.2)
    )
    
    if len(active_cells[0]) == 0:
        logger.warning("No active cells found for liminal state.")
        return {
            'success': False,
            'message': "No active cells found"
        }
    
    # Apply liminal state properties
    for i in range(len(active_cells[0])):
        x, y, z = active_cells[0][i], active_cells[1][i], active_cells[2][i]
        
        # Set liminal frequency (with some variation)
        variation = 1.0 + 0.2 * (random.random() - 0.5)
        current_freq = brain_structure.frequency_grid[x, y, z]
        
        # Blend current with liminal (weighted transition)
        brain_structure.frequency_grid[x, y, z] = (
            current_freq * 0.3 + liminal_frequency * variation * 0.7
        )
        
        # Set coherence (lower in liminal state)
        current_coherence = brain_structure.coherence_grid[x, y, z]
        
        # Blend current with liminal (weighted transition)
        brain_structure.coherence_grid[x, y, z] = (
            current_coherence * 0.5 + liminal_coherence * 0.5
        )
        
        # Set resonance (moderate in liminal state)
        current_resonance = brain_structure.resonance_grid[x, y, z]
        
        # Blend current with liminal (weighted transition)
        brain_structure.resonance_grid[x, y, z] = (
            current_resonance * 0.5 + liminal_resonance * 0.5
        )
        
        cells_affected += 1
    
    logger.info(f"Initialized liminal state in {cells_affected} cells")
    
    # Return metrics
    result = {
        'success': True,
        'state': 'liminal',
        'cells_affected': cells_affected,
        'base_frequency': float(liminal_frequency),
        'base_coherence': float(liminal_coherence),
        'base_resonance': float(liminal_resonance),
        'percentage_active': float(cells_affected / brain_structure.total_grid_cells)
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'state': 'liminal',
                'cells_affected': cells_affected,
                'base_frequency': float(liminal_frequency),
                'percentage_active': float(cells_affected / brain_structure.total_grid_cells)
            }
            metrics.record_metrics("mycelial_liminal_state", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record liminal state metrics: {e}")
    
    return result


def prepare_for_dream_state(brain_structure) -> Dict[str, Any]:
    """
    Prepare the brain for dream state transition.
    Sets up the necessary pathways and patterns for dream state.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with preparation metrics
    """
    logger.info("Preparing for dream state")
    
    # Define dream state parameters
    dream_frequency = 5.5  # Theta wave
    dream_coherence = 0.6  # Higher coherence than liminal
    dream_resonance = 0.7  # Higher resonance than liminal
    
    # Define key regions for dream state
    dream_regions = [
        REGION_LIMBIC,    # Emotions in dreams
        REGION_TEMPORAL,  # Memory incorporation
        REGION_OCCIPITAL  # Visual processing
    ]
    
    # Track affected cells
    cells_affected = 0
    dream_cells = set()
    
    # Create dream pathways between key regions
    dream_pathways = []
    
    # Connect each dream region to others
    for i, region1 in enumerate(dream_regions):
        for j, region2 in enumerate(dream_regions[i+1:], i+1):
            # Skip if either region doesn't exist
            if region1 not in brain_structure.regions or region2 not in brain_structure.regions:
                continue
                
            # Get region centers
            if 'center' not in brain_structure.regions[region1] or 'center' not in brain_structure.regions[region2]:
                continue
                
            center1 = brain_structure.regions[region1]['center']
            center2 = brain_structure.regions[region2]['center']
            
            # Create pathway between regions
            pathway = create_dream_pathway(brain_structure, center1, center2, region1, region2)
            
            if pathway['created']:
                dream_pathways.append(pathway)
                
                # Add cells to tracking set
                for cell in pathway['pathway_cells']:
                    dream_cells.add(cell)
    
    # Prepare key dream regions
    for region_name in dream_regions:
        if region_name not in brain_structure.regions:
            continue
            
        # Find cells in this region
        region_indices = np.where(brain_structure.region_grid == region_name)
        
        if len(region_indices[0]) == 0:
            continue
            
        # Sample a subset of cells for efficiency
        sample_size = min(1000, len(region_indices[0]))
        sample_indices = np.random.choice(len(region_indices[0]), sample_size, replace=False)
        
        # Prepare each sampled cell
        for i in sample_indices:
            x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
            
            # Skip if no mycelial density
            if brain_structure.mycelial_density_grid[x, y, z] < 0.1:
                continue
                
            # Apply dream state properties
            variation = 1.0 + 0.3 * (random.random() - 0.5)  # More variation in dreams
            
            # Adjust frequency toward dream frequency
            current_freq = brain_structure.frequency_grid[x, y, z]
            dream_freq = dream_frequency * variation
            
            # Blend current with dream (weighted)
            brain_structure.frequency_grid[x, y, z] = (
                current_freq * 0.4 + dream_freq * 0.6
            )
            
            # Adjust coherence toward dream coherence
            current_coherence = brain_structure.coherence_grid[x, y, z]
            brain_structure.coherence_grid[x, y, z] = (
                current_coherence * 0.5 + dream_coherence * 0.5
            )
            
            # Adjust resonance toward dream resonance
            current_resonance = brain_structure.resonance_grid[x, y, z]
            brain_structure.resonance_grid[x, y, z] = (
                current_resonance * 0.5 + dream_resonance * 0.5
            )
            
            # Add to tracking
            dream_cells.add((x, y, z))
    
    # Count affected cells
    cells_affected = len(dream_cells)
    
    logger.info(f"Prepared {cells_affected} cells for dream state with {len(dream_pathways)} dream pathways")
    
    # Return metrics
    result = {
        'success': True,
        'cells_affected': cells_affected,
        'dream_pathways': len(dream_pathways),
        'base_frequency': float(dream_frequency),
        'base_coherence': float(dream_coherence),
        'base_resonance': float(dream_resonance),
        'percentage_prepared': float(cells_affected / brain_structure.total_grid_cells),
        'pathways': dream_pathways
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'cells_affected': cells_affected,
                'dream_pathways': len(dream_pathways),
                'base_frequency': float(dream_frequency),
                'percentage_prepared': float(cells_affected / brain_structure.total_grid_cells)
            }
            metrics.record_metrics("mycelial_dream_preparation", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record dream preparation metrics: {e}")
    
    return result


def create_dream_pathway(brain_structure, start_pos: Tuple[int, int, int],
                       end_pos: Tuple[int, int, int], start_region: str, 
                       end_region: str) -> Dict[str, Any]:
    """
    Create a specialized pathway for dream state.
    Dream pathways have unique wave-like properties.
    
    Args:
        brain_structure: The brain structure object
        start_pos: Start position
        end_pos: End position
        start_region: Start region name
        end_region: End region name
        
    Returns:
        Dict with pathway metrics
    """
    # Calculate direct distance
    sx, sy, sz = start_pos
    ex, ey, ez = end_pos
    
    direct_dist = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
    
    # Skip if too far
    if direct_dist > MYCELIAL_MAXIMUM_PATHWAY_LENGTH:
        return {
            'created': False,
            'error': "Distance exceeds maximum pathway length"
        }
    
    # Create pathway with dream-like wave pattern (multiple frequencies)
    pathway_radius = 1
    
    # Calculate step count - dreams are more complex
    step_count = int(direct_dist * (1 + 1/PHI))
    step_count = max(10, min(50, step_count))
    
    # Create wave-like pathway using multiple overlapping waves
    pathway_points = []
    
    for i in range(step_count + 1):
        t = i / step_count
        
        # Linear interpolation with multiple wave offsets
        # Dreams have complex wave patterns
        wave1_amplitude = direct_dist * 0.15  # 15% of distance
        wave1_frequency = 3  # 3 cycles
        
        wave2_amplitude = direct_dist * 0.07  # 7% of distance
        wave2_frequency = 5  # 5 cycles
        
        # Calculate base position with linear interpolation
        base_x = sx + t * (ex - sx)
        base_y = sy + t * (ey - sy)
        base_z = sz + t * (ez - sz)
        
        # Apply wave offsets
        if abs(ex - sx) > abs(ey - sy):
            # Pathway is more horizontal, so wave vertically
            wave1_offset = wave1_amplitude * math.sin(wave1_frequency * 2 * math.pi * t)
            wave2_offset = wave2_amplitude * math.sin(wave2_frequency * 2 * math.pi * t)
            base_y += wave1_offset + wave2_offset
        else:
            # Pathway is more vertical, so wave horizontally
            wave1_offset = wave1_amplitude * math.sin(wave1_frequency * 2 * math.pi * t)
            wave2_offset = wave2_amplitude * math.sin(wave2_frequency * 2 * math.pi * t)
            base_x += wave1_offset + wave2_offset
        
        # Convert to integer coordinates
        x = int(base_x)
        y = int(base_y)
        z = int(base_z)
        
        # Constrain to brain bounds
        x = max(0, min(brain_structure.dimensions[0] - 1, x))
        y = max(0, min(brain_structure.dimensions[1] - 1, y))
        z = max(0, min(brain_structure.dimensions[2] - 1, z))
        
        pathway_points.append((x, y, z))
    
    # Track cells affected
    pathway_cells = set()
    
    # Apply pathway properties
    dream_frequency = 5.5  # Theta wave
    dream_coherence = 0.6  # Higher coherence
    
    for point in pathway_points:
        x, y, z = point
        
        # Create a small radius around this point
        for dx in range(-pathway_radius, pathway_radius + 1):
            for dy in range(-pathway_radius, pathway_radius + 1):
                for dz in range(-pathway_radius, pathway_radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < brain_structure.dimensions[0] and 
                           0 <= ny < brain_structure.dimensions[1] and 
                           0 <= nz < brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from pathway center
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if dist <= pathway_radius:
                        # Apply dream pathway properties
                        # Dreams have high variation
                        variation = 1.0 + 0.4 * (random.random() - 0.5)
                        
                        # Adjust frequency toward dream frequency
                        current_freq = brain_structure.frequency_grid[nx, ny, nz]
                        brain_structure.frequency_grid[nx, ny, nz] = (
                            current_freq * 0.3 + dream_frequency * variation * 0.7
                        )
                        
                        # Adjust coherence toward dream coherence
                        current_coherence = brain_structure.coherence_grid[nx, ny, nz]
                        brain_structure.coherence_grid[nx, ny, nz] = (
                            current_coherence * 0.4 + dream_coherence * 0.6
                        )
                        
                        # Add to mycelial density for pathway
                        brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                            brain_structure.mycelial_density_grid[nx, ny, nz],
                            0.6  # Good density for dreams
                        )
                        
                        # Track affected cells
                        pathway_cells.add((nx, ny, nz))
    
# Calculate pathway length (following the wave)
    actual_length = 0.0
    for i in range(len(pathway_points) - 1):
        p1 = pathway_points[i]
        p2 = pathway_points[i + 1]
        
        segment_length = math.sqrt(
            (p2[0] - p1[0])**2 + 
            (p2[1] - p1[1])**2 + 
            (p2[2] - p1[2])**2
        )
        
        actual_length += segment_length
    
    # Return metrics
    return {
        'created': True,
        'start_region': start_region,
        'end_region': end_region,
        'direct_distance': float(direct_dist),
        'actual_length': float(actual_length),
        'pathway_points': len(pathway_points),
        'unique_cells': len(pathway_cells),
        'pathway_radius': pathway_radius,
        'pathway_cells': list(pathway_cells)  # Convert set to list for serialization
    }


def define_state_transition_triggers(brain_structure) -> Dict[str, Any]:
    """
    Define triggers for state transitions in consciousness.
    Creates the conditions that trigger changes between liminal, dream, and awareness states.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with trigger metrics
    """
    logger.info("Defining state transition triggers")
    
    # Define transition trigger parameters
    triggers = [
        {
            'name': 'liminal_to_dream',
            'from_state': 'liminal',
            'to_state': 'dream',
            'frequency_threshold': 5.0,  # Move to dream state when frequency rises above 5.0 Hz
            'coherence_threshold': 0.5,  # With coherence above 0.5
            'energy_threshold': 10.0,     # And sufficient energy
            'cells_required': 0.2         # In at least 20% of active cells
        },
        {
            'name': 'dream_to_liminal',
            'from_state': 'dream',
            'to_state': 'liminal',
            'frequency_threshold': 4.0,  # Fall back to liminal when frequency drops below 4.0 Hz
            'coherence_threshold': 0.4,  # Or coherence drops
            'energy_threshold': 5.0,     # Or energy is low
            'cells_required': 0.1        # In at least 10% of active cells
        },
        {
            'name': 'dream_to_awareness',
            'from_state': 'dream',
            'to_state': 'awareness',
            'frequency_threshold': 8.0,  # Move to awareness when frequency rises above 8.0 Hz
            'coherence_threshold': 0.7,  # With high coherence
            'energy_threshold': 20.0,    # And high energy
            'cells_required': 0.3        # In at least 30% of active cells
        },
        {
            'name': 'awareness_to_dream',
            'from_state': 'awareness',
            'to_state': 'dream',
            'frequency_threshold': 7.0,  # Drop to dream when frequency falls below 7.0 Hz
            'coherence_threshold': 0.6,  # Or coherence drops
            'energy_threshold': 15.0,    # Or energy decreases
            'cells_required': 0.2        # In at least 20% of active cells
        }
    ]
    
    # Create trigger regions in brain (areas that monitor state)
    trigger_regions = []
    
    # For each trigger, create monitoring nodes in appropriate regions
    for trigger in triggers:
        # Different regions are responsible for different triggers
        if trigger['name'] == 'liminal_to_dream':
            # Limbic and brain stem trigger dream state
            monitor_regions = [REGION_LIMBIC, REGION_BRAIN_STEM]
        elif trigger['name'] == 'dream_to_liminal':
            # Brain stem triggers return to liminal
            monitor_regions = [REGION_BRAIN_STEM]
        elif trigger['name'] == 'dream_to_awareness':
            # Frontal and parietal trigger awareness
            monitor_regions = [REGION_FRONTAL, REGION_PARIETAL]
        elif trigger['name'] == 'awareness_to_dream':
            # Limbic triggers return to dream
            monitor_regions = [REGION_LIMBIC]
        else:
            monitor_regions = []
        
        # Create monitoring nodes in each relevant region
        trigger_monitoring_nodes = []
        
        for region_name in monitor_regions:
            if region_name not in brain_structure.regions:
                continue
                
            # Find cells in this region
            region_indices = np.where(brain_structure.region_grid == region_name)
            
            if len(region_indices[0]) == 0:
                continue
                
            # Use cells with good mycelial density and resonance for monitoring
            region_cells = []
            
            for i in range(len(region_indices[0])):
                x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
                
                # Check for good monitoring properties
                density = brain_structure.mycelial_density_grid[x, y, z]
                resonance = brain_structure.resonance_grid[x, y, z]
                
                if density > 0.5 and resonance > 0.5:
                    region_cells.append((x, y, z, float(density), float(resonance)))
            
            # Sort by combined quality (density + resonance)
            region_cells.sort(key=lambda c: c[3] + c[4], reverse=True)
            
            # Select top cells for monitoring nodes (max 5 per region)
            monitor_cells = region_cells[:5]
            
            # Create monitoring nodes
            for cell in monitor_cells:
                x, y, z, density, resonance = cell
                
                # Enhance monitoring properties
                brain_structure.mycelial_density_grid[x, y, z] = max(
                    brain_structure.mycelial_density_grid[x, y, z],
                    0.8  # High density for monitoring
                )
                
                brain_structure.resonance_grid[x, y, z] = max(
                    brain_structure.resonance_grid[x, y, z],
                    0.8  # High resonance for monitoring
                )
                
                # Add to monitoring nodes
                trigger_monitoring_nodes.append({
                    'position': (x, y, z),
                    'region': region_name,
                    'density': density,
                    'resonance': resonance
                })
        
        # Add monitor nodes to trigger
        trigger['monitoring_nodes'] = trigger_monitoring_nodes
        trigger_regions.append({
            'trigger_name': trigger['name'],
            'monitoring_nodes': len(trigger_monitoring_nodes),
            'monitor_regions': monitor_regions
        })
    
    logger.info(f"Defined {len(triggers)} state transition triggers with monitoring nodes")
    
    # Return metrics
    result = {
        'triggers_defined': len(triggers),
        'trigger_regions': trigger_regions,
        'triggers': triggers
    }
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'triggers_defined': len(triggers),
                'trigger_regions': trigger_regions
            }
            metrics.record_metrics("mycelial_state_triggers", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record state trigger metrics: {e}")
    
    return result


# --- Data Export for System Handoff ---
def prepare_network_handoff_data(brain_structure, brain_seed) -> Dict[str, Any]:
    """
    Prepare mycelial network data for handoff to full system.
    Creates a comprehensive data package for the system stage.
    
    Args:
        brain_structure: The brain structure object
        brain_seed: The brain seed object
        
    Returns:
        Dict with handoff data
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info("Preparing mycelial network data for system handoff")
    
    # Validate inputs
    if not hasattr(brain_structure, 'mycelial_density_grid'):
        raise ValueError("Invalid brain_structure object. Missing mycelial_density_grid attribute.")
    
    if not hasattr(brain_seed, 'position'):
        raise ValueError("Invalid brain_seed object. Missing position attribute.")
    
    # Create handoff data structure
    # This contains all the information needed by the full system
    handoff_data = {
        'mycelial_network_id': str(uuid.uuid4()),
        'creation_time': datetime.now().isoformat(),
        'brain_id': brain_structure.brain_id,
        'seed_id': brain_seed.seed_id,
        'seed_position': brain_seed.position,
        'network_metrics': {}
    }
    
    # Calculate network metrics
    network_metrics = calculate_network_metrics(brain_structure)
    handoff_data['network_metrics'] = network_metrics
    
    # Define energy distribution and storage information
    energy_data = {
        'total_energy': float(np.sum(brain_structure.mycelial_energy_grid)),
        'energy_cells': int(np.sum(brain_structure.mycelial_energy_grid > 0.1)),
        'average_density': float(np.mean(brain_structure.mycelial_density_grid)),
        'maximum_density': float(np.max(brain_structure.mycelial_density_grid)),
        'seed_energy_level': float(brain_seed.base_energy_level),
        'seed_mycelial_energy': float(brain_seed.mycelial_energy_store),
        'energy_by_region': {}
    }
    
    # Calculate energy by region
    for region_name in brain_structure.regions:
        region_indices = np.where(brain_structure.region_grid == region_name)
        
        if len(region_indices[0]) > 0:
            region_energy = np.sum(brain_structure.mycelial_energy_grid[region_indices])
            region_density = np.mean(brain_structure.mycelial_density_grid[region_indices])
            
            energy_data['energy_by_region'][region_name] = {
                'total_energy': float(region_energy),
                'average_density': float(region_density)
            }
    
    handoff_data['energy_data'] = energy_data
    
    # Define state information
    state_data = {
        'current_state': 'liminal',  # Initial state
        'state_cells': {
            'liminal': int(np.sum(
                (brain_structure.frequency_grid > 2.0) & 
                (brain_structure.frequency_grid < 5.0)
            )),
            'dream': int(np.sum(
                (brain_structure.frequency_grid >= 5.0) & 
                (brain_structure.frequency_grid < 8.0)
            )),
            'awareness': int(np.sum(brain_structure.frequency_grid >= 8.0))
        },
        'average_frequency': float(np.mean(brain_structure.frequency_grid[brain_structure.frequency_grid > 0])),
        'average_coherence': float(np.mean(brain_structure.coherence_grid)),
        'average_resonance': float(np.mean(brain_structure.resonance_grid)),
        'soul_presence': float(np.mean(brain_structure.soul_presence_grid))
    }
    
    handoff_data['state_data'] = state_data
    
    # Define high-density pathway information
    # This helps the full system identify existing pathways
    pathway_data = identify_existing_pathways(brain_structure)
    handoff_data['pathway_data'] = pathway_data
    
    # Add soul connection information if available
    if hasattr(brain_seed, 'soul_connection') and brain_seed.soul_connection is not None:
        handoff_data['soul_connection'] = brain_seed.soul_connection
    
    # Add high-activity node information
    high_activity_nodes = identify_high_activity_nodes(brain_structure)
    handoff_data['high_activity_nodes'] = high_activity_nodes
    
    logger.info("Mycelial network handoff data prepared successfully")
    
    # Return handoff data
    return handoff_data


def calculate_network_metrics(brain_structure) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for the mycelial network.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with network metrics
    """
    # Calculate basic metrics
    mycelial_cells = np.sum(brain_structure.mycelial_density_grid > 0.1)
    coverage_percent = mycelial_cells / brain_structure.total_grid_cells * 100
    
    # Calculate distribution metrics
    energy_distribution = np.sum(brain_structure.mycelial_energy_grid)
    active_cells = np.sum(brain_structure.mycelial_energy_grid > 0.1)
    
    # Calculate field metrics
    avg_energy = float(np.mean(brain_structure.energy_grid))
    avg_frequency = float(np.mean(brain_structure.frequency_grid[brain_structure.frequency_grid > 0]))
    avg_resonance = float(np.mean(brain_structure.resonance_grid))
    avg_coherence = float(np.mean(brain_structure.coherence_grid))
    avg_stability = float(np.mean(brain_structure.stability_grid))
    
    # Calculate density distribution
    density_bins = [0.1, 0.3, 0.5, 0.7, 0.9]
    density_distribution = {}
    
    for i in range(len(density_bins)):
        if i == 0:
            # First bin: density > bin threshold
            count = np.sum(brain_structure.mycelial_density_grid > density_bins[i])
        else:
            # Middle bins: between this and previous threshold
            count = np.sum((brain_structure.mycelial_density_grid > density_bins[i]) & 
                          (brain_structure.mycelial_density_grid <= density_bins[i-1]))
        
        density_distribution[f'>{density_bins[i]}'] = int(count)
    
    # Calculate region coverage
    region_coverage = {}
    for region_name in brain_structure.regions:
        region_indices = np.where(brain_structure.region_grid == region_name)
        
        if len(region_indices[0]) > 0:
            region_cells = len(region_indices[0])
            region_mycelial = np.sum(brain_structure.mycelial_density_grid[region_indices] > 0.1)
            coverage = region_mycelial / region_cells
            
            region_coverage[region_name] = {
                'total_cells': int(region_cells),
                'mycelial_cells': int(region_mycelial),
                'coverage_percent': float(coverage * 100)
            }
    
    # Compile all metrics
    return {
        'mycelial_cells': int(mycelial_cells),
        'coverage_percent': float(coverage_percent),
        'energy_distribution': float(energy_distribution),
        'active_cells': int(active_cells),
        'field_metrics': {
            'avg_energy': avg_energy,
            'avg_frequency': avg_frequency,
            'avg_resonance': avg_resonance,
            'avg_coherence': avg_coherence,
            'avg_stability': avg_stability
        },
        'density_distribution': density_distribution,
        'region_coverage': region_coverage
    }


def identify_existing_pathways(brain_structure) -> List[Dict[str, Any]]:
    """
    Identify existing high-density pathways in the mycelial network.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        List of pathway data dictionaries
    """
    # Find cells with high mycelial density
    high_density_indices = np.where(brain_structure.mycelial_density_grid > 0.6)
    
    if len(high_density_indices[0]) == 0:
        return []
    
    # Create list of high-density points
    high_density_points = [
        (high_density_indices[0][i], high_density_indices[1][i], high_density_indices[2][i])
        for i in range(len(high_density_indices[0]))
    ]
    
    # Group points into contiguous pathways
    pathways = []
    points_assigned = set()
    
    # For each unassigned point, find its neighbors and build a pathway
    for point in high_density_points:
        if point in points_assigned:
            continue
        
        # Start a new pathway with this point
        pathway_points = [point]
        points_assigned.add(point)
        
        # Grow the pathway by adding connected neighbors
        i = 0
        while i < len(pathway_points):
            x, y, z = pathway_points[i]
            
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # Skip self
                        
                        nx, ny, nz = x + dx, y + dy, z + dz
                        neighbor = (nx, ny, nz)
                        
                        # If neighbor is high-density and not assigned, add to pathway
                        if (neighbor in high_density_points and 
                            neighbor not in points_assigned):
                            pathway_points.append(neighbor)
                            points_assigned.add(neighbor)
            
            i += 1
        
        # Only add pathways with enough points
        if len(pathway_points) >= 5:
            # Calculate pathway properties
            avg_density = np.mean([
                brain_structure.mycelial_density_grid[x, y, z]
                for x, y, z in pathway_points
            ])
            
            # Identify regions this pathway connects
            regions = set()
            for x, y, z in pathway_points:
                region = brain_structure.region_grid[x, y, z]
                if region:
                    regions.add(region)
            
            # Calculate endpoints as most distant points
            if len(pathway_points) >= 2:
                # Find two most distant points
                max_dist = 0
                endpoints = (pathway_points[0], pathway_points[-1])
                
                for i, p1 in enumerate(pathway_points):
                    for j, p2 in enumerate(pathway_points[i+1:], i+1):
                        dist = math.sqrt(
                            (p2[0] - p1[0])**2 + 
                            (p2[1] - p1[1])**2 + 
                            (p2[2] - p1[2])**2
                        )
                        
                        if dist > max_dist:
                            max_dist = dist
                            endpoints = (p1, p2)
            else:
                endpoints = (pathway_points[0], pathway_points[0])
                max_dist = 0
            
            # Measure pathway length (approximate)
            pathway_length = max_dist
            
            # Add pathway data
            pathways.append({
                'points': len(pathway_points),
                'average_density': float(avg_density),
                'regions_connected': list(regions),
                'endpoints': endpoints,
                'length': float(pathway_length)
            })
    
    # Sort pathways by length (longest first)
    pathways.sort(key=lambda p: p['length'], reverse=True)
    
    # Return top 20 for system integration
    return pathways[:20]


def identify_high_activity_nodes(brain_structure) -> List[Dict[str, Any]]:
    """
    Identify high-activity nodes in the mycelial network.
    These are cells with combination of high energy, high density, and high resonance.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        List of high-activity node data dictionaries
    """
    # Define criteria for high-activity nodes
    energy_threshold = 0.5  # High energy
    density_threshold = 0.7  # High mycelial density
    resonance_threshold = 0.7  # High resonance
    
    # Find cells meeting all criteria
    high_indices = np.where(
        (brain_structure.mycelial_energy_grid > energy_threshold) &
        (brain_structure.mycelial_density_grid > density_threshold) &
        (brain_structure.resonance_grid > resonance_threshold)
    )
    
    if len(high_indices[0]) == 0:
        return []
    
    # Create list of high-activity nodes
    high_nodes = []
    
    for i in range(len(high_indices[0])):
        x, y, z = high_indices[0][i], high_indices[1][i], high_indices[2][i]
        
        # Get cell properties
        energy = float(brain_structure.mycelial_energy_grid[x, y, z])
        density = float(brain_structure.mycelial_density_grid[x, y, z])
        resonance = float(brain_structure.resonance_grid[x, y, z])
        coherence = float(brain_structure.coherence_grid[x, y, z])
        frequency = float(brain_structure.frequency_grid[x, y, z])
        region = brain_structure.region_grid[x, y, z]
        
        # Calculate activity score
        activity_score = (
            energy * 0.3 + 
            density * 0.3 + 
            resonance * 0.2 + 
            coherence * 0.2
        )
        
        # Add node data
        high_nodes.append({
            'position': (int(x), int(y), int(z)),
            'region': region,
            'energy': energy,
            'density': density,
            'resonance': resonance,
            'coherence': coherence,
            'frequency': frequency,
            'activity_score': float(activity_score)
        })
    
    # Sort by activity score (highest first)
    high_nodes.sort(key=lambda n: n['activity_score'], reverse=True)
    
    # Return top 50 for system integration
    return high_nodes[:50]


def export_memory_fragment_mapping(brain_structure) -> Dict[str, Any]:
    """
    Export mapping of memory fragments in the brain.
    Creates a map of fragment locations and properties.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with fragment mapping data
    """
    logger.info("Exporting memory fragment mapping")
    
    # Define criteria for fragment identification
    # Fragments have high resonance but aren't necessarily activated yet
    resonance_threshold = 0.6
    
    # Find cells with good resonance (potential fragments)
    fragment_indices = np.where(brain_structure.resonance_grid > resonance_threshold)
    
    if len(fragment_indices[0]) == 0:
        logger.warning("No memory fragments found for mapping.")
        return {
            'fragments_found': 0,
            'message': "No memory fragments found"
        }
    
    # Group points into fragment clusters
    fragments = []
    points_assigned = set()
    
    # For each unassigned point, find its neighbors and build a fragment
    for i in range(len(fragment_indices[0])):
        x, y, z = fragment_indices[0][i], fragment_indices[1][i], fragment_indices[2][i]
        point = (x, y, z)
        
        if point in points_assigned:
            continue
        
        # Start a new fragment with this point
        fragment_points = [point]
        points_assigned.add(point)
        
        # Grow the fragment by adding connected neighbors
        j = 0
        while j < len(fragment_points):
            fx, fy, fz = fragment_points[j]
            
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # Skip self
                        
                        nx, ny, nz = fx + dx, fy + dy, fz + dz
                        
                        # Skip if outside brain bounds
                        if not (0 <= nx < brain_structure.dimensions[0] and 
                               0 <= ny < brain_structure.dimensions[1] and 
                               0 <= nz < brain_structure.dimensions[2]):
                            continue
                        
                        # Check if neighbor has high resonance
                        if brain_structure.resonance_grid[nx, ny, nz] > resonance_threshold:
                            neighbor = (nx, ny, nz)
                            
                            # If not already assigned, add to fragment
                            if neighbor not in points_assigned:
                                fragment_points.append(neighbor)
                                points_assigned.add(neighbor)
            
            j += 1
        
        # Only add fragments with enough points
        if len(fragment_points) >= 3:
            # Calculate fragment centroid
            center_x = int(np.mean([p[0] for p in fragment_points]))
            center_y = int(np.mean([p[1] for p in fragment_points]))
            center_z = int(np.mean([p[2] for p in fragment_points]))
            
            # Get fragment properties
            region = brain_structure.region_grid[center_x, center_y, center_z]
            
            # Calculate fragment properties
            avg_resonance = np.mean([
                brain_structure.resonance_grid[x, y, z]
                for x, y, z in fragment_points
            ])
            
            avg_frequency = np.mean([
                brain_structure.frequency_grid[x, y, z]
                for x, y, z in fragment_points
            ])
            
            soul_presence = np.mean([
                brain_structure.soul_presence_grid[x, y, z]
                for x, y, z in fragment_points
            ])
            
            # Add fragment data
            fragments.append({
                'center': (center_x, center_y, center_z),
                'points': len(fragment_points),
                'region': region,
                'avg_resonance': float(avg_resonance),
                'avg_frequency': float(avg_frequency),
                'soul_presence': float(soul_presence),
                'fragment_id': str(uuid.uuid4())
            })
    
    # Sort fragments by resonance (highest first)
    fragments.sort(key=lambda f: f['avg_resonance'], reverse=True)
    
    # Count fragments by region
    region_counts = {}
    for fragment in fragments:
        region = fragment['region']
        if region not in region_counts:
            region_counts[region] = 0
        region_counts[region] += 1
    
    logger.info(f"Exported mapping of {len(fragments)} memory fragments")
    
    # Return mapping data
    return {
        'fragments_found': len(fragments),
        'fragments': fragments,
        'region_counts': region_counts
    }


def export_energy_distribution_state(brain_structure) -> Dict[str, Any]:
    """
    Export the current energy distribution state of the mycelial network.
    
    Args:
        brain_structure: The brain structure object
        
    Returns:
        Dict with energy distribution data
    """
    logger.info("Exporting energy distribution state")
    
    # Calculate overall energy stats
    total_energy = float(np.sum(brain_structure.mycelial_energy_grid))
    max_energy = float(np.max(brain_structure.mycelial_energy_grid))
    avg_energy = float(np.mean(brain_structure.mycelial_energy_grid[brain_structure.mycelial_energy_grid > 0]))
    energy_cells = int(np.sum(brain_structure.mycelial_energy_grid > 0))
    
    # Calculate energy by region
    region_energy = {}
    for region_name in brain_structure.regions:
        region_indices = np.where(brain_structure.region_grid == region_name)
        
        if len(region_indices[0]) > 0:
            region_total = float(np.sum(brain_structure.mycelial_energy_grid[region_indices]))
            region_max = float(np.max(brain_structure.mycelial_energy_grid[region_indices]))
            region_cells = int(np.sum(brain_structure.mycelial_energy_grid[region_indices] > 0))
            region_avg = (
                float(np.mean(brain_structure.mycelial_energy_grid[region_indices][
                    brain_structure.mycelial_energy_grid[region_indices] > 0
                ])) 
                if region_cells > 0 else 0.0
            )
            
            region_energy[region_name] = {
                'total_energy': region_total,
                'max_energy': region_max,
                'avg_energy': region_avg,
                'energy_cells': region_cells,
                'percentage': float(region_total / total_energy) if total_energy > 0 else 0.0
            }
    
    # Find high-energy concentration points
    high_energy_threshold = max_energy * 0.7  # 70% of maximum
    high_energy_indices = np.where(brain_structure.mycelial_energy_grid > high_energy_threshold)
    
    high_energy_points = []
    for i in range(len(high_energy_indices[0])):
        x, y, z = high_energy_indices[0][i], high_energy_indices[1][i], high_energy_indices[2][i]
        
        high_energy_points.append({
            'position': (int(x), int(y), int(z)),
            'energy': float(brain_structure.mycelial_energy_grid[x, y, z]),
            'region': brain_structure.region_grid[x, y, z]
        })
    
    # Sort by energy (highest first)
    high_energy_points.sort(key=lambda p: p['energy'], reverse=True)
    
    logger.info(f"Exported energy distribution state with {energy_cells} active cells")
    
    # Return energy distribution data
    return {
        'total_energy': total_energy,
        'max_energy': max_energy,
        'avg_energy': avg_energy,
        'energy_cells': energy_cells,
        'region_energy': region_energy,
        'high_energy_points': high_energy_points[:20]  # Top 20 for reference
    }


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("MycelialFunctions module test execution")
    
    try:
        # Import necessary modules
        from stage_1.evolve.brain_structure.brain_structure import BrainGrid, create_brain_structure
        from stage_1.evolve.brain_structure.brain_seed import BrainSeed, create_brain_seed
        
        # Create test brain structure
        brain_structure = create_brain_structure(
            dimensions=(64, 64, 64),  # Smaller for testing
            initialize_regions=True,
            initialize_sound=True
        )
        
        # Create test brain seed
        brain_seed = create_brain_seed(
            initial_beu=10.0,
            initial_mycelial_beu=5.0,
            initial_frequency=7.83
        )
        
        # Set seed position
        if brain_structure.regions_defined:
            # Find limbic region
            limbic_indices = np.where(brain_structure.region_grid == REGION_LIMBIC)
            
            if len(limbic_indices[0]) > 0:
                # Calculate center of limbic region
                center_x = int(np.mean(limbic_indices[0]))
                center_y = int(np.mean(limbic_indices[1]))
                center_z = int(np.mean(limbic_indices[2]))
                
                # Set seed position
                seed_position = (center_x, center_y, center_z)
                region = REGION_LIMBIC
            else:
                # Fallback to center of brain
                seed_position = (
                    brain_structure.dimensions[0] // 2,
                    brain_structure.dimensions[1] // 2,
                    brain_structure.dimensions[2] // 2
                )
                region = brain_structure.region_grid[seed_position]
            
            # Set position in brain seed
            brain_seed.set_position(seed_position, region)
        
        # Test functions
        logger.info("Testing MycelialFunctions module functions:")
        
        # 1. Initialize basic network
        logger.info("1. Testing initialize_basic_network()...")
        init_result = initialize_basic_network(brain_structure, brain_seed.position)
        logger.info(f"   Result: {init_result['cells_affected']} cells affected")
        
        # 2. Establish primary pathways
        logger.info("2. Testing establish_primary_pathways()...")
        pathway_result = establish_primary_pathways(brain_structure)
        logger.info(f"   Result: {pathway_result['pathways_created']} pathways created")
        
        # 3. Setup energy distribution channels
        logger.info("3. Testing setup_energy_distribution_channels()...")
        channel_result = setup_energy_distribution_channels(brain_structure)
        logger.info(f"   Result: {channel_result['channels_created']} channels created")
        
        # 4. Prepare for soul attachment
        logger.info("4. Testing prepare_for_soul_attachment()...")
        soul_prep_result = prepare_for_soul_attachment(brain_structure)
        logger.info(f"   Result: {soul_prep_result['cells_prepared']} cells prepared")
        
        # 5. Initialize liminal state
        logger.info("5. Testing initialize_liminal_state()...")
        liminal_result = initialize_liminal_state(brain_structure)
        logger.info(f"   Result: {liminal_result['cells_affected']} cells affected")
        
        # 6. Prepare for dream state
        logger.info("6. Testing prepare_for_dream_state()...")
        dream_result = prepare_for_dream_state(brain_structure)
        logger.info(f"   Result: {dream_result['cells_affected']} cells affected")
        
        # 7. Prepare network handoff data
        logger.info("7. Testing prepare_network_handoff_data()...")
        handoff_data = prepare_network_handoff_data(brain_structure, brain_seed)
        logger.info(f"   Result: Handoff data prepared with {len(handoff_data)} top-level keys")
        
        # Save test results
        try:
            with open("mycelial_test_results.json", "w") as f:
                import json
                json.dump({
                    "init_result": init_result,
                    "pathway_result": pathway_result,
                    "channel_result": channel_result,
                    "soul_prep_result": soul_prep_result,
                    "liminal_result": liminal_result,
                    "dream_result": dream_result,
                    "handoff_data_keys": list(handoff_data.keys())
                }, f, indent=2, default=str)
            
            logger.info("Test results saved to mycelial_test_results.json")
        except Exception as e:
            logger.warning(f"Could not save test results: {e}")
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        print(f"ERROR: Test execution failed: {e}")
        sys.exit(1)
    
    logger.info("MycelialFunctions module test execution completed")

# --- END OF FILE stage_1/evolve/mycelial_functions.py ---