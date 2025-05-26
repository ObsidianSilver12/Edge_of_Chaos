# --- START OF FILE stage_1/evolve/brain_seed_integration.py ---

"""
Brain Seed Integration Module (V4.5.0 - Wave Physics & Field Dynamics)

Handles the integration of a brain seed with the brain structure.
Uses field dynamics to determine optimal placement and integration patterns.
Establishes energy and frequency connections between seed and brain structure.
"""

import logging
import os
import sys
import numpy as np
import uuid
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import math
# Import constants from the main constants module
from constants.constants import *

# --- Logging ---
logger = logging.getLogger('BrainSeedIntegration')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()

class BrainSeedIntegration:
    """
    Handles the integration of a brain seed with the brain structure.
    Uses field dynamics to find optimal position and establish connections.
    """
    
    def __init__(self):
        """Initialize the brain seed integration handler."""
        self.integration_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        
        # Integration state
        self.integration_completed = False
        self.seed_id = None
        self.optimal_position = None
        self.integration_metrics = {}
        
        logger.info("BrainSeedIntegration initialized")
    
    def integrate_seed_with_structure(self, brain_seed, brain_structure, 
                                     seed_position: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Integrate the brain seed with the brain structure.
        
        Args:
            brain_seed: The BrainSeed instance
            brain_structure: The BrainGrid instance
            seed_position: Optional position for seed (if None, optimal position is determined)
            
        Returns:
            Dict containing integration metrics
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If integration fails
        """
        # Validate inputs
        if not hasattr(brain_seed, 'seed_id'):
            raise ValueError("Invalid brain_seed object. Missing seed_id.")
            
        if not hasattr(brain_structure, 'brain_id'):
            raise ValueError("Invalid brain_structure object. Missing brain_id.")
        
        self.seed_id = brain_seed.seed_id
        
        logger.info(f"Integrating brain seed {brain_seed.seed_id} with brain structure {brain_structure.brain_id}")
        
        # Initialize metrics
        integration_metrics = {
            'integration_id': self.integration_id,
            'seed_id': brain_seed.seed_id,
            'brain_id': brain_structure.brain_id,
            'integration_start': datetime.now().isoformat(),
            'seed_energy_beu': brain_seed.base_energy_level,
            'mycelial_energy_beu': brain_seed.mycelial_energy_store,
            'seed_frequency_hz': brain_seed.base_frequency_hz,
            'integration_phases': {}
        }
        
        try:
            # Validate brain structure preparation
            prep_status = brain_structure.prepare_for_soul_distribution()
            if not prep_status.get('success', False):
                raise ValueError(f"Brain structure not ready for integration: {prep_status.get('messages', [])}")
            
            # Phase 1: Position seed in brain structure using field dynamics
            if seed_position is None:
                seed_position = self._find_optimal_position_based_on_fields(brain_structure, brain_seed)
                logger.info(f"Optimal seed position determined using field dynamics: {seed_position}")
            
            # Validate seed position
            x, y, z = seed_position
            if not (0 <= x < brain_structure.dimensions[0] and 
                   0 <= y < brain_structure.dimensions[1] and 
                   0 <= z < brain_structure.dimensions[2]):
                raise ValueError(f"Seed position {seed_position} out of brain structure bounds")
            
            # Determine brain region at seed position
            seed_region = brain_structure.region_grid[x, y, z]
            seed_subregion = brain_structure.sub_region_grid[x, y, z]
            
            if not seed_region:
                raise ValueError(f"No brain region defined at position {seed_position}")
            
            # Store optimal position
            self.optimal_position = seed_position
            
            # Validate seed position is in limbic region (preferred)
            if seed_region != REGION_LIMBIC and seed_region != REGION_BRAIN_STEM:
                logger.warning(f"Seed position {seed_position} not in limbic or brain stem region. "
                              f"Current region: {seed_region}")
            
            # Set seed position in brain seed
            brain_seed.set_position(seed_position, seed_region, seed_subregion)
            
            # Phase 1 metrics
            phase1_metrics = {
                'seed_position': seed_position,
                'seed_region': seed_region,
                'seed_subregion': seed_subregion if seed_subregion else "none",
                'position_field_properties': {
                    'energy': float(brain_structure.energy_grid[x, y, z]),
                    'frequency': float(brain_structure.frequency_grid[x, y, z]),
                    'stability': float(brain_structure.stability_grid[x, y, z]),
                    'coherence': float(brain_structure.coherence_grid[x, y, z]),
                    'resonance': float(brain_structure.resonance_grid[x, y, z])
                },
                'field_dynamics_score': self._calculate_field_dynamics_score(brain_structure, seed_position)
            }
            integration_metrics['integration_phases']['positioning'] = phase1_metrics
            
            # Phase 2: Establish energy field connection
            energy_connection_metrics = self._establish_energy_connection(brain_seed, brain_structure, seed_position)
            integration_metrics['integration_phases']['energy_connection'] = energy_connection_metrics
            
            # Phase 3: Establish frequency resonance
            frequency_resonance_metrics = self._establish_frequency_resonance(brain_seed, brain_structure, seed_position)
            integration_metrics['integration_phases']['frequency_resonance'] = frequency_resonance_metrics
            
            # Phase 4: Create initial mycelial pathways
            mycelial_pathway_metrics = self._create_initial_mycelial_pathways(brain_seed, brain_structure, seed_position)
            integration_metrics['integration_phases']['mycelial_pathways'] = mycelial_pathway_metrics
            
            # Mark integration as complete
            self.integration_completed = True
            self.last_updated = datetime.now().isoformat()
            self.integration_metrics = integration_metrics
            
            # Final metrics
            integration_metrics['integration_end'] = datetime.now().isoformat()
            integration_metrics['success'] = True
            
            logger.info(f"Brain seed integration completed successfully at position {seed_position}")
            return integration_metrics
            
        except Exception as e:
            logger.error(f"Error during brain seed integration: {e}", exc_info=True)
            integration_metrics['integration_end'] = datetime.now().isoformat()
            integration_metrics['success'] = False
            integration_metrics['error'] = str(e)
            
            # Store failed metrics
            self.integration_metrics = integration_metrics
            
            raise RuntimeError(f"Brain seed integration failed: {e}")    
    def _find_optimal_position_based_on_fields(self, brain_structure, brain_seed) -> Tuple[int, int, int]:
        """
        Find the optimal position for the brain seed based on field dynamics.
        Uses field intersections, edge of chaos, and resonance patterns to find the best position.
        
        Args:
            brain_structure: The brain structure
            brain_seed: The brain seed to place
            
        Returns:
            Tuple of (x, y, z) coordinates for optimal position
        """
        logger.info("Finding optimal seed position using field dynamics")
        
        # First preference: find intersection of energy/stability fields in limbic region
        limbic_indices = np.where(brain_structure.region_grid == REGION_LIMBIC)
        if len(limbic_indices[0]) == 0:
            logger.warning("Limbic region not found. Checking brain stem region.")
            
            # Check brain stem as alternative
            brain_stem_indices = np.where(brain_structure.region_grid == REGION_BRAIN_STEM)
            if len(brain_stem_indices[0]) == 0:
                logger.warning("Neither limbic nor brain stem regions found. Using field dynamics alone.")
                return self._find_optimal_position_with_field_dynamics(brain_structure, brain_seed)
            
            # Use brain stem
            region_indices = brain_stem_indices
            region_name = REGION_BRAIN_STEM
        else:
            # Use limbic region
            region_indices = limbic_indices
            region_name = REGION_LIMBIC
        
        logger.info(f"Using {region_name} region for seed placement with {len(region_indices[0])} cells")
        
        # Find field edge cells within target region
        highest_dynamics_score = -1.0
        optimal_position = None
        
        # Sample points in region for computational efficiency
        sample_size = min(500, len(region_indices[0]))
        sample_indices = np.random.choice(len(region_indices[0]), sample_size, replace=False)
        
        for i in sample_indices:
            x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
            position = (x, y, z)
            
            # Calculate field dynamics score
            dynamics_score = self._calculate_field_dynamics_score(brain_structure, position)
            
            # Update if better
            if dynamics_score > highest_dynamics_score:
                highest_dynamics_score = dynamics_score
                optimal_position = position
        
        if optimal_position:
            logger.info(f"Found optimal position in {region_name} at {optimal_position} "
                       f"with field dynamics score {highest_dynamics_score:.4f}")
            return optimal_position
        
        # Fallback to general field dynamics
        logger.warning(f"No suitable position found in {region_name}. Using general field dynamics.")
        return self._find_optimal_position_with_field_dynamics(brain_structure, brain_seed)
    
    def _find_optimal_position_with_field_dynamics(self, brain_structure, brain_seed) -> Tuple[int, int, int]:
        """
        Find optimal position using general field dynamics (not region-specific).
        
        Args:
            brain_structure: The brain structure
            brain_seed: The brain seed
            
        Returns:
            Tuple of (x, y, z) coordinates for optimal position
        """
        # Find field edge points in general
        edge_points = self._find_edge_of_chaos_points(brain_structure, 100)
        
        if not edge_points:
            # Fallback to center of brain
            logger.warning("No edge of chaos points found. Using brain center.")
            return (brain_structure.dimensions[0] // 2,
                   brain_structure.dimensions[1] // 2,
                   brain_structure.dimensions[2] // 2)
        
        # Find point with highest field dynamics score
        highest_dynamics_score = -1.0
        optimal_position = None
        
        for position in edge_points:
            # Calculate field dynamics score
            dynamics_score = self._calculate_field_dynamics_score(brain_structure, position)
            
            # Update if better
            if dynamics_score > highest_dynamics_score:
                highest_dynamics_score = dynamics_score
                optimal_position = position
        
        if optimal_position:
            logger.info(f"Found optimal position at {optimal_position} "
                       f"with field dynamics score {highest_dynamics_score:.4f}")
            return optimal_position
        
        # Fallback to first edge point
        logger.warning("Could not find optimal position with field dynamics. Using first edge point.")
        return edge_points[0]
    
    def _find_edge_of_chaos_points(self, brain_structure, max_points: int = 100) -> List[Tuple[int, int, int]]:
        """
        Find points at the edge of chaos (high energy gradient).
        These are good candidates for seed placement.
        
        Args:
            brain_structure: The brain structure
            max_points: Maximum number of points to return
            
        Returns:
            List of (x, y, z) coordinates
        """
        # Calculate energy gradient magnitude
        grad_x, grad_y, grad_z = np.gradient(brain_structure.energy_grid)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Find cells with high gradient magnitude
        # Use 80th percentile as threshold
        threshold = np.percentile(gradient_magnitude, 80)
        high_gradient_cells = gradient_magnitude > threshold
        
        # Get coordinates of high gradient cells
        high_gradient_indices = np.where(high_gradient_cells)
        points = [(int(high_gradient_indices[0][i]), 
                 int(high_gradient_indices[1][i]), 
                 int(high_gradient_indices[2][i])) 
                 for i in range(min(len(high_gradient_indices[0]), max_points))]
        
        # If no points found, reduce threshold
        if not points:
            threshold = np.percentile(gradient_magnitude, 50)
            high_gradient_cells = gradient_magnitude > threshold
            high_gradient_indices = np.where(high_gradient_cells)
            points = [(int(high_gradient_indices[0][i]), 
                     int(high_gradient_indices[1][i]), 
                     int(high_gradient_indices[2][i])) 
                     for i in range(min(len(high_gradient_indices[0]), max_points))]
        
        logger.debug(f"Found {len(points)} edge of chaos points with threshold {threshold:.4f}")
        return points
    
    def _calculate_field_dynamics_score(self, brain_structure, position: Tuple[int, int, int]) -> float:
            """
            Calculate a score for the field dynamics at a given position.
            Higher scores indicate better positions for seed placement.
            
            Args:
                brain_structure: The brain structure
                position: Position to evaluate
                
            Returns:
                Field dynamics score (0-1 scale)
            """
            x, y, z = position
            
            # Extract field values at position
            energy = brain_structure.energy_grid[x, y, z]
            frequency = brain_structure.frequency_grid[x, y, z]
            stability = brain_structure.stability_grid[x, y, z]
            coherence = brain_structure.coherence_grid[x, y, z]
            resonance = brain_structure.resonance_grid[x, y, z]
            
            # Calculate energy gradient at position
            energy_grad_x, energy_grad_y, energy_grad_z = np.gradient(brain_structure.energy_grid)
            energy_gradient_magnitude = math.sqrt(energy_grad_x[x, y, z]**2 + 
                                            energy_grad_y[x, y, z]**2 + 
                                            energy_grad_z[x, y, z]**2)
            
            # Calculate frequency gradient at position
            freq_grad_x, freq_grad_y, freq_grad_z = np.gradient(brain_structure.frequency_grid)
            freq_gradient_magnitude = math.sqrt(freq_grad_x[x, y, z]**2 + 
                                            freq_grad_y[x, y, z]**2 + 
                                            freq_grad_z[x, y, z]**2)
            
            # Normalize energy and frequency to 0-1 range
            energy_norm = min(1.0, energy / 1.0) if energy > 0 else 0.0
            freq_norm = min(1.0, frequency / 30.0) if frequency > 0 else 0.0
            
            # Normalize gradients to 0-1 range
            energy_grad_norm = min(1.0, energy_gradient_magnitude / 0.2)
            freq_grad_norm = min(1.0, freq_gradient_magnitude / 5.0)
            
            # Calculate base score from field properties
            base_score = (
                energy_norm * 0.2 +         # Energy level
                stability * 0.2 +           # Stability
                coherence * 0.2 +           # Coherence
                resonance * 0.2 +           # Resonance
                energy_grad_norm * 0.1 +    # Energy gradient (edge of chaos)
                freq_grad_norm * 0.1        # Frequency gradient
            )
            
            # Add bonus for phi-harmonic frequency relationships
            # PHI is the golden ratio (~1.618)
            phi_resonance = 0.0
            phi_ratios = [PHI, 1/PHI, PHI*2, PHI/2]
            for ratio in phi_ratios:
                if frequency > 0:
                    common_frequencies = [7.83, 8.0, 10.0, 12.0, 13.0, 14.0, 20.0, 30.0]
                    for common_freq in common_frequencies:
                        freq_ratio = frequency / common_freq
                        # Check proximity to phi ratio
                        phi_proximity = 1.0 - min(abs(freq_ratio - ratio) / ratio, 1.0)
                        phi_resonance = max(phi_resonance, phi_proximity)
            
            # Add phi resonance bonus (up to 20%)
            final_score = base_score + (phi_resonance * 0.2)
            
            # Normalize to 0-1 range
            final_score = min(1.0, max(0.0, final_score))
            
            return final_score
    
    def _establish_energy_connection(self, brain_seed, brain_structure, 
                                    seed_position: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Establish energy field connection between seed and brain structure.
        Creates energy field around seed and distributes energy.
        
        Args:
            brain_seed: The brain seed
            brain_structure: The brain structure
            seed_position: Seed position
            
        Returns:
            Dict with energy connection metrics
        """
        logger.info(f"Establishing energy connection at position {seed_position}")
        
        x, y, z = seed_position
        energy_amount = brain_seed.base_energy_level * 0.3  # Use 30% of seed energy
        
        # Ensure sufficient energy
        if energy_amount < 1.0:
            logger.warning(f"Low energy for connection: {energy_amount:.2E} BEU. "
                            f"Using minimum 1.0 BEU.")
            energy_amount = 1.0
        
        # Use energy from seed
        energy_result = brain_seed.use_energy(energy_amount, "brain_connection")
        if not energy_result['success']:
            raise RuntimeError(f"Insufficient energy for connection: {energy_result['message']}")
        
        # Create radial energy field
        radius = brain_seed.seed_field['radius']
        energy_distributed = 0.0
        cells_energized = 0
        
        # Create connection points
        connection_points = []
        
        # Apply energy in spherical pattern with phi harmonics
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < brain_structure.dimensions[0] and 
                            0 <= ny < brain_structure.dimensions[1] and 
                            0 <= nz < brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from seed
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Apply energy if within radius
                    if dist <= radius:
                        # Calculate phi harmonics
                        # Use standing wave pattern with phi ratio nodes
                        phi_harmonic = 0.5 + 0.5 * math.cos(dist / radius * PHI * math.pi)
                        
                        # Calculate energy distribution factor
                        dist_factor = (1.0 - dist / radius) ** 2  # Squared falloff
                        energy_factor = dist_factor * phi_harmonic
                        
                        # Calculate energy for this cell
                        cell_energy = energy_amount * energy_factor * 0.01  # 1% per cell
                        
                        # Add energy to brain structure
                        brain_structure.energy_grid[nx, ny, nz] += cell_energy
                        energy_distributed += cell_energy
                        cells_energized += 1
                        
                        # Store connection point
                        if energy_factor > 0.3:  # Only store significant connections
                            connection_points.append({
                                'position': (nx, ny, nz),
                                'energy': float(cell_energy),
                                'distance': float(dist),
                                'region': brain_structure.region_grid[nx, ny, nz],
                                'sub_region': brain_structure.sub_region_grid[nx, ny, nz]
                            })
        
        # Calculate region distribution
        region_distribution = {}
        for nx in range(max(0, x - radius), min(brain_structure.dimensions[0], x + radius + 1)):
            for ny in range(max(0, y - radius), min(brain_structure.dimensions[1], y + radius + 1)):
                for nz in range(max(0, z - radius), min(brain_structure.dimensions[2], z + radius + 1)):
                    region = brain_structure.region_grid[nx, ny, nz]
                    if region:
                        energy = brain_structure.energy_grid[nx, ny, nz]
                        if region not in region_distribution:
                            region_distribution[region] = 0.0
                        region_distribution[region] += energy
        
        # Return metrics
        return {
            'energy_used': float(energy_amount),
            'energy_distributed': float(energy_distributed),
            'distribution_efficiency': float(energy_distributed / energy_amount),
            'cells_energized': cells_energized,
            'region_distribution': {k: float(v) for k, v in region_distribution.items()},
            'connection_points': connection_points[:10]  # Store only top 10 for brevity
        }

    def _establish_frequency_resonance(self, brain_seed, brain_structure, 
                                        seed_position: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Establish frequency resonance between seed and brain structure.
        Creates resonance patterns around seed.
        
        Args:
            brain_seed: The brain seed
            brain_structure: The brain structure
            seed_position: Seed position
            
        Returns:
            Dict with frequency resonance metrics
        """
        logger.info(f"Establishing frequency resonance at position {seed_position}")
        
        x, y, z = seed_position
        base_frequency = brain_seed.base_frequency_hz
        harmonics = brain_seed.frequency_harmonics
        resonance_radius = brain_seed.seed_field['radius'] * 1.5  # Wider radius for resonance
        
        # Apply frequency pattern
        cells_affected = 0
        region_resonance = {}
        
        # Create resonance field with standing wave patterns
        for dx in range(-resonance_radius, resonance_radius + 1):
            for dy in range(-resonance_radius, resonance_radius + 1):
                for dz in range(-resonance_radius, resonance_radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < brain_structure.dimensions[0] and 
                            0 <= ny < brain_structure.dimensions[1] and 
                            0 <= nz < brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from seed
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Apply resonance if within radius
                    if dist <= resonance_radius:
                        # Calculate position in normalized spherical coordinates
                        dist_norm = dist / resonance_radius  # 0-1 range
                        
                        # Calculate resonance field
                        # Use standing waves with harmonic interference
                        wave_sum = 0.0
                        for i, harmonic in enumerate(harmonics[:3]):  # Use first 3 harmonics
                            # Each harmonic creates a standing wave
                            # Use decreasing amplitude for higher harmonics
                            amplitude = 1.0 / (i + 1)
                            
                            # Create standing wave based on distance
                            # sin(π*r*h) creates nodes at specific distances based on harmonic
                            wave = amplitude * math.sin(math.pi * dist_norm * (i + 1))
                            wave_sum += wave
                        
                        # Normalize wave sum to 0-1 range
                        wave_norm = (wave_sum + len(harmonics[:3])) / (2 * len(harmonics[:3]))
                        
                        # Get current brain frequency
                        brain_freq = brain_structure.frequency_grid[nx, ny, nz]
                        
                        # Calculate resonance as proximity to harmonic relationship
                        resonance = 0.0
                        if brain_freq > 0:
                            # Check proximity to harmonics
                            for harmonic in harmonics:
                                freq_ratio = harmonic / brain_freq
                                # Check if close to integer ratio
                                for ratio in [0.5, 1.0, 1.5, 2.0, PHI, 1/PHI]:
                                    proximity = 1.0 - min(abs(freq_ratio - ratio) / max(ratio, 0.1), 1.0)
                                    resonance = max(resonance, proximity)
                        
                        # Apply resonance enhancement (influenced by wave pattern)
                        resonance_enhancement = wave_norm * resonance * 0.5
                        new_resonance = min(1.0, brain_structure.resonance_grid[nx, ny, nz] + resonance_enhancement)
                        
                        # Update resonance in brain
                        brain_structure.resonance_grid[nx, ny, nz] = new_resonance
                        cells_affected += 1
                        
                        # Track region resonance
                        region = brain_structure.region_grid[nx, ny, nz]
                        if region:
                            if region not in region_resonance:
                                region_resonance[region] = []
                            region_resonance[region].append(new_resonance)
        
        # Calculate average resonance per region
        region_resonance_avg = {}
        for region, values in region_resonance.items():
            region_resonance_avg[region] = float(np.mean(values))
        
        # Return metrics
        return {
            'base_frequency': float(base_frequency),
            'harmonics': [float(h) for h in harmonics],
            'resonance_radius': resonance_radius,
            'cells_affected': cells_affected,
            'region_resonance_avg': region_resonance_avg
        }

    def _create_initial_mycelial_pathways(self, brain_seed, brain_structure, 
                                        seed_position: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Create initial mycelial pathways from seed to key brain regions.
        
        Args:
            brain_seed: The brain seed
            brain_structure: The brain structure
            seed_position: Seed position
            
        Returns:
            Dict with mycelial pathway metrics
        """
        logger.info(f"Creating initial mycelial pathways from position {seed_position}")
        
        x, y, z = seed_position
        
        # Define target regions
        target_regions = [
            REGION_LIMBIC,
            REGION_BRAIN_STEM,
            REGION_FRONTAL,
            REGION_TEMPORAL,
            REGION_PARIETAL,
            REGION_OCCIPITAL
        ]
        
        # Find closest cell in each target region
        pathways = []
        total_length = 0
        total_density = 0.0
        
        for target_region in target_regions:
            # Skip seed's own region
            if target_region == brain_structure.region_grid[x, y, z]:
                continue
            
            # Find cells in target region
            region_indices = np.where(brain_structure.region_grid == target_region)
            
            if len(region_indices[0]) == 0:
                logger.warning(f"Target region {target_region} not found. Skipping.")
                continue
            
            # Find closest cell
            closest_dist = float('inf')
            closest_pos = None
            
            # Sample a subset for efficiency
            sample_size = min(100, len(region_indices[0]))
            sample_indices = np.random.choice(len(region_indices[0]), sample_size, replace=False)
            
            for i in sample_indices:
                tx, ty, tz = region_indices[0][i], region_indices[1][i], region_indices[2][i]
                dist = math.sqrt((tx - x)**2 + (ty - y)**2 + (tz - z)**2)
                
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = (tx, ty, tz)
            
            if closest_pos is None:
                continue
            
            # Create pathway
            pathway = self._create_mycelial_pathway(brain_structure, seed_position, closest_pos, target_region)
            
            if pathway.get('created', False):
                pathways.append(pathway)
                total_length += pathway['actual_length']
                total_density += pathway['avg_density'] * pathway['actual_length']
        
        # Calculate metrics
        average_density = total_density / total_length if total_length > 0 else 0.0
        
        # Return metrics
        return {
            'pathways_created': len(pathways),
            'total_pathway_length': total_length,
            'average_density': float(average_density),
            'pathways': pathways
        }
    
    def _create_mycelial_pathway(self, brain_structure, start_pos: Tuple[int, int, int], 
                                end_pos: Tuple[int, int, int], target_region: str) -> Dict[str, Any]:
        """
        Create a mycelial pathway between two positions in the brain.
        
        Args:
            brain_structure: The brain structure
            start_pos: Starting position (seed position)
            end_pos: Ending position (target in region)
            target_region: Target region name
            
        Returns:
            Dict with pathway creation metrics
        """
        sx, sy, sz = start_pos
        ex, ey, ez = end_pos
        
        # Calculate pathway vector
        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        
        # Calculate pathway length
        pathway_length = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Check if pathway exceeds maximum length
        if pathway_length > MYCELIAL_MAXIMUM_PATHWAY_LENGTH:
            logger.warning(f"Pathway to {target_region} exceeds maximum length: {pathway_length:.1f}")
            return {
                'created': False,
                'reason': 'exceeds_max_length',
                'requested_length': float(pathway_length),
                'max_length': MYCELIAL_MAXIMUM_PATHWAY_LENGTH
            }
        
        # Create pathway points along the line
        num_steps = max(int(pathway_length), 3)  # At least 3 steps
        pathway_points = []
        total_density = 0.0
        
        for i in range(num_steps + 1):
            t = i / num_steps  # 0 to 1
            
            # Calculate interpolated position
            px = int(sx + t * dx)
            py = int(sy + t * dy)
            pz = int(sz + t * dz)
            
            # Check bounds
            if not (0 <= px < brain_structure.dimensions[0] and
                    0 <= py < brain_structure.dimensions[1] and
                    0 <= pz < brain_structure.dimensions[2]):
                continue
            
            # Calculate mycelial density based on distance from start
            # Higher density near start and end, lower in middle
            dist_from_start = t * pathway_length
            dist_from_end = (1 - t) * pathway_length
            
            # Create density profile - higher at endpoints
            density_factor = 0.3 + 0.7 * (1.0 / (1.0 + min(dist_from_start, dist_from_end) / 10.0))
            mycelial_density = min(1.0, density_factor * 0.5)  # Cap at 0.5 density
            
            # Add mycelial density to brain structure
            current_density = brain_structure.mycelial_density_grid.get((px, py, pz), 0.0)
            new_density = min(1.0, current_density + mycelial_density)
            brain_structure.mycelial_density_grid[(px, py, pz)] = new_density
            
            # Track pathway points
            pathway_points.append({
                'position': (px, py, pz),
                'step': i,
                'density': float(mycelial_density),
                'region': brain_structure.region_grid.get((px, py, pz), 'unknown')
            })
            
            total_density += mycelial_density
        
        # Calculate metrics
        avg_density = total_density / len(pathway_points) if pathway_points else 0.0
        actual_length = len(pathway_points)
        
        return {
            'created': True,
            'target_region': target_region,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'requested_length': float(pathway_length),
            'actual_length': actual_length,
            'avg_density': float(avg_density),
            'pathway_points': pathway_points[:5]  # Store first 5 points for brevity
        }


    # --- Helper Functions ---

    @staticmethod
    def integrate_brain_seed(brain_seed, brain_structure, 
                        seed_position: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Helper function to integrate a brain seed with a brain structure.

        Args:
            brain_seed: The brain seed
            brain_structure: The brain structure
            seed_position: Optional seed position
            
        Returns:
            Integration metrics
        """
        integrator = BrainSeedIntegration()
        return integrator.integrate_seed_with_structure(brain_seed, brain_structure, seed_position)

    # --- Main Execution ---
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        logger.info("BrainSeedIntegration module test execution")

        try:
            # Import necessary modules
            from stage_1.evolve.brain_structure.brain_structure import BrainGrid, create_brain_structure
            from stage_1.evolve.brain_structure.brain_seed import BrainSeed, create_brain_seed
            
            # Create brain structure
            brain_structure = create_brain_structure(
                dimensions=(128, 128, 128),
                initialize_regions=True,
                initialize_sound=True
            )
            
            # Create brain seed
            brain_seed = create_brain_seed(
                initial_beu=10.0,
                initial_mycelial_beu=5.0,
                initial_frequency=7.83
            )
            
            # Integrate seed with structure
            integration_metrics = integrate_brain_seed(brain_seed, brain_structure)
            
            # Print results
            print(f"Integration successful: {integration_metrics['success']}")
            print(f"Seed position: {integration_metrics['integration_phases']['positioning']['seed_position']}")
            print(f"Seed region: {integration_metrics['integration_phases']['positioning']['seed_region']}")
            
            # Save integration metrics
            import json
            with open("brain_seed_integration_test.json", "w") as f:
                json.dump(integration_metrics, f, indent=2, default=str)
            
            print("Integration metrics saved to brain_seed_integration_test.json")
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}", exc_info=True)
            print(f"ERROR: Integration test failed: {e}")
            sys.exit(1)

        logger.info("BrainSeedIntegration module test execution completed")

# --- END OF FILE stage_1/evolve/brain_seed_integration.py ---
    
