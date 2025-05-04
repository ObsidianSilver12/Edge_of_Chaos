# --- mycelial_network.py - Core mycelial network implementation ---

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
import os
import json

# Import constants and brain structure references
try:
    from constants.constants import *
    from stage_1.soul_formation.region_definitions import *
except ImportError as e:
    logging.critical(f"Failed to import required modules: {e}")
    raise ImportError(f"Mycelial network requires region_definitions.py and constants.py: {e}")

# Configure logging
logger = logging.getLogger("MycelialNetwork")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MycelialNetwork:
    """
    Mycelial network for energy distribution through brain structure.
    Acts as the neural connectivity system for soul propagation.
    """
    
    def __init__(self, brain_grid):
        """Initialize with reference to the brain grid"""
        self.brain_grid = brain_grid
        self.dimensions = brain_grid.dimensions
        
        # Validate that brain grid has the necessary attributes
        if not hasattr(brain_grid, 'mycelial_density_grid') or not hasattr(brain_grid, 'mycelial_energy_grid'):
            raise ValueError("Brain grid missing mycelial grid attributes")
        
        # Network configuration
        self.pathway_count = 0
        self.branch_count = 0
        self.seed_points = []
        self.major_pathways = []
        self.initialized = False
        self.energy_flowing = False
        
        # Tracking metrics
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        self.metrics = {}
        
        logger.info("Mycelial network initialized with brain grid reference")
    
    def initialize_network(self, seed_position: Optional[Tuple[int, int, int]] = None):
        """
        Initialize the mycelial network with pathways and branches.
        
        Args:
            seed_position: Optional central seed position where brain seed will be located
        """
        logger.info("Initializing mycelial network pathways")
        
        # Ensure brain regions are defined
        if not hasattr(self.brain_grid, 'region_grid') or np.all(self.brain_grid.region_grid == ""):
            raise ValueError("Brain grid regions not defined. Cannot initialize mycelial network.")
        
        # Create seed points in each region
        self._create_seed_points(seed_position)
        
        # Connect seed points with mycelial pathways
        self._connect_seed_points()
        
        # Create secondary branches through each region
        self._create_regional_branches()
        
        # Create special pathways based on brain function
        self._create_functional_pathways()
        
        # Normalize densities
        self._normalize_density_values()
        
        # Update metrics
        self.metrics.update({
            "seed_points": len(self.seed_points),
            "major_pathways": len(self.major_pathways),
            "total_branches": self.branch_count,
            "avg_density": float(np.mean(self.brain_grid.mycelial_density_grid[self.brain_grid.mycelial_density_grid > 0])),
            "coverage_percent": float(np.sum(self.brain_grid.mycelial_density_grid > 0.1) / np.prod(self.dimensions) * 100)
        })
        
        self.initialized = True
        self.last_updated = datetime.now().isoformat()
        logger.info(f"Mycelial network initialized with {len(self.seed_points)} seed points and {self.branch_count} branches")
        
        return True
    
    def _create_seed_points(self, central_seed_position: Optional[Tuple[int, int, int]] = None):
        """Create seed points in each major region"""
        logger.info("Creating mycelial network seed points")
        
        self.seed_points = []
        
        # If central seed position is provided, add it first
        if central_seed_position:
            x, y, z = central_seed_position
            # Set maximum density at seed
            self.brain_grid.mycelial_density_grid[x, y, z] = 1.0
            self.seed_points.append((x, y, z, "central"))
            logger.debug(f"  Central seed point added at {central_seed_position}")
        
        # Create a seed point in each major region
        for region_name in MAJOR_REGIONS:
            # Skip if this is the region containing the central seed
            if central_seed_position and self.brain_grid.region_grid[central_seed_position] == region_name:
                logger.debug(f"  Skipping additional seed in {region_name} (contains central seed)")
                continue
                
            # Find all cells for this region
            region_indices = np.where(self.brain_grid.region_grid == region_name)
            if len(region_indices[0]) == 0:
                logger.warning(f"  No cells found for region {region_name}")
                continue
            
            # Choose a central point for the seed
            center_idx = len(region_indices[0]) // 2
            seed_x = region_indices[0][center_idx]
            seed_y = region_indices[1][center_idx]
            seed_z = region_indices[2][center_idx]
            
            # Set high mycelial density at seed
            self.brain_grid.mycelial_density_grid[seed_x, seed_y, seed_z] = 0.9
            
            # Add to seed points
            self.seed_points.append((seed_x, seed_y, seed_z, region_name))
            logger.debug(f"  Seed point added for {region_name} at ({seed_x}, {seed_y}, {seed_z})")
        
        # Add seeds for key sub-regions
        key_subregions = ["hippocampus", "amygdala", "thalamus", "prefrontal", "visual_cortex"]
        for sub_name in key_subregions:
            # Find all cells for this sub-region
            sub_indices = np.where(self.brain_grid.sub_region_grid == sub_name)
            if len(sub_indices[0]) == 0:
                continue
                
            # Choose a central point for the seed
            center_idx = len(sub_indices[0]) // 2
            seed_x = sub_indices[0][center_idx]
            seed_y = sub_indices[1][center_idx]
            seed_z = sub_indices[2][center_idx]
            
            # Only add if not too close to existing seeds
            too_close = False
            for x, y, z, _ in self.seed_points:
                dist = np.sqrt((seed_x-x)**2 + (seed_y-y)**2 + (seed_z-z)**2)
                if dist < 10:  # Avoid seeds that are too close together
                    too_close = True
                    break
            
            if not too_close:
                # Set high mycelial density at seed
                self.brain_grid.mycelial_density_grid[seed_x, seed_y, seed_z] = 0.85
                
                # Add to seed points
                self.seed_points.append((seed_x, seed_y, seed_z, sub_name))
                logger.debug(f"  Seed point added for {sub_name} at ({seed_x}, {seed_y}, {seed_z})")
        
        logger.info(f"Created {len(self.seed_points)} mycelial seed points")
    
    def _connect_seed_points(self):
        """Connect seed points with primary mycelial pathways"""
        logger.info("Connecting mycelial seed points with primary pathways")
        
        if not self.seed_points:
            logger.warning("No seed points available to connect")
            return
        
        # Identify central seed point if present
        central_seed = None
        for seed in self.seed_points:
            if seed[3] == "central":
                central_seed = seed
                break
        
        # Create a pathway from each seed to the central seed or to all others
        pathways_created = 0
        
        if central_seed:
            # Connect all seeds to central seed (star topology)
            central_x, central_y, central_z, _ = central_seed
            for seed_x, seed_y, seed_z, seed_name in self.seed_points:
                if seed_name == "central":
                    continue
                
                # Create pathway
                path_info = self._create_mycelial_pathway(
                    (central_x, central_y, central_z),
                    (seed_x, seed_y, seed_z),
                    0.8,
                    f"central-to-{seed_name}")
                
                self.major_pathways.append(path_info)
                pathways_created += 1
        else:
            # Connect each seed to several others (mesh topology)
            for i, seed1 in enumerate(self.seed_points):
                seed1_x, seed1_y, seed1_z, seed1_name = seed1
                
                # Connect to a subset of other seeds
                connections_needed = min(3, len(self.seed_points) - 1)
                connections_made = 0
                
                for j, seed2 in enumerate(self.seed_points):
                    if i == j:
                        continue
                        
                    seed2_x, seed2_y, seed2_z, seed2_name = seed2
                    
                    # Only create some connections (not a full mesh)
                    if connections_made >= connections_needed:
                        break
                    
                    # Create pathway
                    path_info = self._create_mycelial_pathway(
                        (seed1_x, seed1_y, seed1_z),
                        (seed2_x, seed2_y, seed2_z),
                        0.7,
                        f"{seed1_name}-to-{seed2_name}")
                    
                    self.major_pathways.append(path_info)
                    pathways_created += 1
                    connections_made += 1
        
        logger.info(f"Created {pathways_created} primary mycelial pathways")
    
    def _create_mycelial_pathway(self, point1: Tuple[int, int, int], 
                               point2: Tuple[int, int, int], 
                               base_density: float,
                               pathway_name: str) -> Dict:
        """
        Create a mycelial pathway between two points with natural variation.
        
        Returns:
            Dict containing pathway information
        """
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        
        # Calculate path parameters
        dx, dy, dz = x2-x1, y2-y1, z2-z1
        path_length = int(np.sqrt(dx**2 + dy**2 + dz**2))
        
        # Track cells in this pathway
        path_cells = []
        
        # Create path with natural variation
        for step in range(path_length + 1):
            # Interpolate position with slight noise
            t = step / path_length if path_length > 0 else 0
            noise_x = np.random.normal(0, 0.15 * path_length)
            noise_y = np.random.normal(0, 0.15 * path_length)
            noise_z = np.random.normal(0, 0.15 * path_length)
            
            x = int(x1 + t*dx + noise_x)
            y = int(y1 + t*dy + noise_y)
            z = int(z1 + t*dz + noise_z)
            
            # Ensure within grid
            if (0 <= x < self.dimensions[0] and 
                0 <= y < self.dimensions[1] and 
                0 <= z < self.dimensions[2]):
                
                # Decrease density with distance from endpoints
                endpoint_distance = min(t, 1-t)
                local_density = base_density * (0.7 + 0.3 * endpoint_distance)
                
                # Set mycelial density
                current_density = self.brain_grid.mycelial_density_grid[x, y, z]
                # Use max blending to avoid overwriting stronger connections
                if local_density > current_density:
                    self.brain_grid.mycelial_density_grid[x, y, z] = local_density
                    path_cells.append((x, y, z))
        
        # Return pathway information
        return {
            "name": pathway_name,
            "start": point1,
            "end": point2,
            "length": path_length,
            "base_density": base_density,
            "cell_count": len(path_cells)
        }
    
    def _create_regional_branches(self):
        """Create secondary branching pathways within each region"""
        logger.info("Creating regional mycelial branches")
        
        # Process each major region
        branches_created = 0
        
        for region_name in MAJOR_REGIONS:
            # Find all cells for this region
            region_indices = np.where(self.brain_grid.region_grid == region_name)
            if len(region_indices[0]) == 0:
                continue
                
            # Find cells with existing mycelial presence in this region
            mycelial_indices = np.where(
                (self.brain_grid.region_grid == region_name) & 
                (self.brain_grid.mycelial_density_grid > 0.1))
            
            if len(mycelial_indices[0]) == 0:
                logger.debug(f"  No mycelial network found in {region_name} for branching")
                continue
                
            # Create branches from existing pathways
            num_branches = min(15, len(mycelial_indices[0]) // 10)
            logger.debug(f"  Creating {num_branches} branches in {region_name}")
            
            for _ in range(num_branches):
                # Choose random start point on existing network
                idx = np.random.randint(0, len(mycelial_indices[0]))
                start_x = mycelial_indices[0][idx]
                start_y = mycelial_indices[1][idx]
                start_z = mycelial_indices[2][idx]
                
                # Choose random end point in region
                idx = np.random.randint(0, len(region_indices[0]))
                end_x = region_indices[0][idx]
                end_y = region_indices[1][idx]
                end_z = region_indices[2][idx]
                
                # Create branch
                self._create_mycelial_pathway(
                    (start_x, start_y, start_z),
                    (end_x, end_y, end_z),
                    0.5,
                    f"{region_name}-branch-{_}")
                
                branches_created += 1
        
        self.branch_count = branches_created
        logger.info(f"Created {branches_created} regional mycelial branches")
    
    def _create_functional_pathways(self):
        """Create special pathways based on brain function"""
        logger.info("Creating functional mycelial pathways")
        
        # Create memory pathway (hippocampus to prefrontal)
        self._create_memory_pathway()
        
        # Create emotional pathway (amygdala to prefrontal)
        self._create_emotional_pathway()
        
        # Create sensory pathways
        self._create_sensory_pathways()
    
    def _create_memory_pathway(self):
        """Create memory pathway connecting hippocampus to prefrontal cortex"""
        # Find hippocampus
        hippo_indices = np.where(self.brain_grid.sub_region_grid == "hippocampus")
        if len(hippo_indices[0]) == 0:
            logger.warning("Could not find hippocampus for memory pathway")
            return
            
        # Find prefrontal
        pref_indices = np.where(self.brain_grid.sub_region_grid == "prefrontal")
        if len(pref_indices[0]) == 0:
            logger.warning("Could not find prefrontal cortex for memory pathway")
            return
        
        # Get central points
        hippo_idx = len(hippo_indices[0]) // 2
        hippo_x = hippo_indices[0][hippo_idx]
        hippo_y = hippo_indices[1][hippo_idx]
        hippo_z = hippo_indices[2][hippo_idx]
        
        pref_idx = len(pref_indices[0]) // 2
        pref_x = pref_indices[0][pref_idx]
        pref_y = pref_indices[1][pref_idx]
        pref_z = pref_indices[2][pref_idx]
        
        # Create high-density memory pathway
        path_info = self._create_mycelial_pathway(
            (hippo_x, hippo_y, hippo_z),
            (pref_x, pref_y, pref_z),
            0.85,  # Higher density for memory pathway
            "memory-pathway")
            
        logger.info(f"Memory pathway created with {path_info['cell_count']} cells")
    
    def _create_emotional_pathway(self):
        """Create emotional pathway connecting amygdala to limbic regions"""
        # Find amygdala
        amyg_indices = np.where(self.brain_grid.sub_region_grid == "amygdala")
        if len(amyg_indices[0]) == 0:
            logger.warning("Could not find amygdala for emotional pathway")
            return
            
        # Find other emotional processing regions
        target_regions = ["prefrontal", "cingulate", "orbitofrontal"]
        
        for target in target_regions:
            target_indices = np.where(self.brain_grid.sub_region_grid == target)
            if len(target_indices[0]) == 0:
                continue
            
            # Get central points
            amyg_idx = len(amyg_indices[0]) // 2
            amyg_x = amyg_indices[0][amyg_idx]
            amyg_y = amyg_indices[1][amyg_idx]
            amyg_z = amyg_indices[2][amyg_idx]
            
            target_idx = len(target_indices[0]) // 2
            target_x = target_indices[0][target_idx]
            target_y = target_indices[1][target_idx]
            target_z = target_indices[2][target_idx]
            
            # Create emotional pathway
            path_info = self._create_mycelial_pathway(
                (amyg_x, amyg_y, amyg_z),
                (target_x, target_y, target_z),
                0.8,  # Higher density for emotional pathway
                f"emotional-pathway-{target}")
                
            logger.info(f"Emotional pathway to {target} created with {path_info['cell_count']} cells")
    
    def _create_sensory_pathways(self):
        """Create sensory pathways connecting primary sensory regions"""
        sensory_regions = ["primary_visual", "primary_auditory", "somatosensory_cortex"]
        integration_regions = ["prefrontal", "limbic"]
        
        for sense in sensory_regions:
            # Find sensory region
            sense_indices = np.where(self.brain_grid.sub_region_grid == sense)
            if len(sense_indices[0]) == 0:
                continue
                
            # Connect to integration regions
            for target in integration_regions:
                target_is_major = target in MAJOR_REGIONS
                
                if target_is_major:
                    target_indices = np.where(self.brain_grid.region_grid == target)
                else:
                    target_indices = np.where(self.brain_grid.sub_region_grid == target)
                    
                if len(target_indices[0]) == 0:
                    continue
                
                # Get central points
                sense_idx = len(sense_indices[0]) // 2
                sense_x = sense_indices[0][sense_idx]
                sense_y = sense_indices[1][sense_idx]
                sense_z = sense_indices[2][sense_idx]
                
                target_idx = len(target_indices[0]) // 2
                target_x = target_indices[0][target_idx]
                target_y = target_indices[1][target_idx]
                target_z = target_indices[2][target_idx]
                
                # Create sensory pathway
                path_info = self._create_mycelial_pathway(
                    (sense_x, sense_y, sense_z),
                    (target_x, target_y, target_z),
                    0.75,
                    f"sensory-pathway-{sense}-to-{target}")
                    
                logger.debug(f"Sensory pathway from {sense} to {target} created with {path_info['cell_count']} cells")
    
    def _normalize_density_values(self):
        """Normalize mycelial density values to 0-1 range"""
        # Find cells with mycelial presence
        mycelial_indices = np.where(self.brain_grid.mycelial_density_grid > 0)
        if len(mycelial_indices[0]) == 0:
            logger.warning("No mycelial network found for normalization")
            return
            
        # Get density statistics
        max_density = np.max(self.brain_grid.mycelial_density_grid)
        min_density = np.min(self.brain_grid.mycelial_density_grid[mycelial_indices])
        
        # Only normalize if necessary
        if max_density > 1.0:
            # Scale to 0-1 range
            self.brain_grid.mycelial_density_grid = np.clip(
                self.brain_grid.mycelial_density_grid / max_density, 
                0.0, 1.0)
            
            logger.debug(f"Normalized mycelial densities from [{min_density:.3f}, {max_density:.3f}] to [0.0, 1.0]")
    
    def distribute_energy(self, seed_position: Tuple[int, int, int], 
                        energy_amount: float, soul_frequency: float,
                        soul_stability: float, soul_coherence: float) -> Dict:
        """
        Distribute energy from seed position through mycelial network
        
        Args:
            seed_position: Position of brain seed (x, y, z)
            energy_amount: Amount of energy to distribute (BEU)
            soul_frequency: Soul's base frequency (Hz)
            soul_stability: Soul's stability factor (0-1)
            soul_coherence: Soul's coherence factor (0-1)
            
        Returns:
            Dict containing distribution metrics
        """
        if not self.initialized:
            logger.warning("Mycelial network not initialized. Cannot distribute energy.")
            return {"success": False, "error": "Network not initialized"}
            
        logger.info(f"Distributing energy ({energy_amount:.2f} BEU) through mycelial network")
        
        # Reset mycelial energy grid
        self.brain_grid.mycelial_energy_grid.fill(0.0)
        
        # Calculate distances from seed
        x0, y0, z0 = seed_position
        
        # Create distance array
        distances = np.zeros(self.dimensions, dtype=np.float32)
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                for z in range(self.dimensions[2]):
                    distances[x, y, z] = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
        
        # Normalize distances to 0-1
        max_dist = np.max(distances)
        if max_dist > 0:
            distances = distances / max_dist
        
        # Find cells with mycelial density
        mycelial_indices = np.where(self.brain_grid.mycelial_density_grid > 0.05)
        mycelial_cell_count = len(mycelial_indices[0])
        
        if mycelial_cell_count == 0:
            logger.warning("No mycelial network found for energy distribution")
            return {"success": False, "error": "No mycelial network"}
        
        # Calculate per-cell energy allocation
        energy_per_cell = energy_amount / mycelial_cell_count
        
        # Soul factors affect distribution
        stability_factor = 0.5 + 0.5 * soul_stability
        coherence_factor = 0.5 + 0.5 * soul_coherence
        
        # Track energy distribution
        total_distributed = 0.0
        cell_count = 0
        region_energy = {}
        
        # Distribute energy based on mycelial density and distance
        for i in range(mycelial_cell_count):
            x, y, z = mycelial_indices[0][i], mycelial_indices[1][i], mycelial_indices[2][i]
            
            # Calculate energy for this cell
            cell_distance = distances[x, y, z]
            cell_density = self.brain_grid.mycelial_density_grid[x, y, z]
            
            # Energy attenuates with distance and increases with density
            # Modulated by soul stability (more stable souls have better energy distribution)
            distance_factor = 1.0 - (cell_distance ** 2) * (1.0 - stability_factor * 0.5)
            density_factor = cell_density ** 0.7  # Non-linear scaling
            
            # Calculate cell energy
            cell_energy = energy_per_cell * distance_factor * density_factor * stability_factor
            
            # Update energy grid
            self.brain_grid.mycelial_energy_grid[x, y, z] += cell_energy
            
            # Update tracking
            total_distributed += cell_energy
            cell_count += 1
            
            # Track region energy
            region = self.brain_grid.region_grid[x, y, z]
            if region:
                if region not in region_energy:
                    region_energy[region] = 0.0
                region_energy[region] += cell_energy
        
        # Update the main energy grid of the brain
        # Mycelial energy contributes to overall brain energy
        self.brain_grid.energy_grid += self.brain_grid.mycelial_energy_grid * 0.8
        
        # Calculate energy distribution metrics
        distribution_metrics = {
            "success": True,
            "total_energy_distributed": float(total_distributed),
            "distribution_efficiency": float(total_distributed / energy_amount),
            "cells_energized": cell_count,
            "region_distribution": {k: float(v) for k, v in region_energy.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        self.energy_flowing = True
        self.last_updated = datetime.now().isoformat()
        logger.info(f"Energy distributed: {total_distributed:.2f} BEU across {cell_count} cells")
        
        return distribution_metrics

    def propagate_frequencies(self, soul_frequency: float, soul_coherence: float) -> Dict:
        """
        Propagate foundational learning frequencies through the mycelial network.
        These frequencies correspond to basic learning patterns for language,
        shapes, sounds, and creative expression.
        
        Args:
            soul_frequency: Soul's base frequency (Hz)
            soul_coherence: Soul's coherence factor (0-1)
            
        Returns:
            Dict containing propagation metrics
        """
        if not self.initialized or not self.energy_flowing:
            logger.warning("Mycelial network not initialized or no energy flowing. Cannot propagate frequencies.")
            return {"success": False, "error": "Network not initialized or no energy"}
        
        logger.info(f"Propagating foundational learning frequencies from {soul_frequency:.2f} Hz")
        
        # Find cells with mycelial energy
        active_indices = np.where(self.brain_grid.mycelial_energy_grid > 0.01)
        active_cell_count = len(active_indices[0])
        
        if active_cell_count == 0:
            logger.warning("No energized cells found for frequency propagation")
            return {"success": False, "error": "No energized cells"}
        
        # Define learning frequency bands based on soul frequency
        # These are aligned with different learning modalities
        learning_frequencies = {
            "language": soul_frequency * 1.2,  # Language/alphabet frequencies
            "visual": soul_frequency * 0.9,    # Visual/shape frequencies
            "auditory": soul_frequency * 1.1,  # Sound/auditory frequencies
            "motor": soul_frequency * 0.8,     # Movement/physical learning
            "creative": soul_frequency * 1.15  # Creative expression
        }
        
        # Determine how coherence affects learning clarity
        coherence_factor = 0.3 + 0.7 * soul_coherence  # Even low coherence allows some learning
        
        # Track propagation metrics
        cells_affected = 0
        region_coverage = {}
        modality_distribution = {k: 0 for k in learning_frequencies.keys()}
        
        # Propagate learning frequencies to different brain regions
        for i in range(active_cell_count):
            x, y, z = active_indices[0][i], active_indices[1][i], active_indices[2][i]
            
            # Get region information (adjust based on your brain_grid structure)
            region = self.brain_grid.region_grid[x, y, z] if hasattr(self.brain_grid, 'region_grid') else ""
            sub_region = self.brain_grid.sub_region_grid[x, y, z] if hasattr(self.brain_grid, 'sub_region_grid') else ""
            
            # Determine appropriate learning frequency based on brain region
            learning_freq = soul_frequency  # Default
            learning_type = "general"
            
            # Map brain regions to learning modalities (adjust based on your regions)
            if region == "temporal" or sub_region in ["wernicke", "broca"]:
                learning_freq = learning_frequencies["language"]
                learning_type = "language"
            elif region == "occipital" or sub_region in ["primary_visual", "visual_association"]:
                learning_freq = learning_frequencies["visual"]
                learning_type = "visual"
            elif region == "temporal" or sub_region in ["primary_auditory"]:
                learning_freq = learning_frequencies["auditory"]
                learning_type = "auditory"
            elif region == "frontal" or sub_region in ["motor_cortex"]:
                learning_freq = learning_frequencies["motor"]
                learning_type = "motor"
            elif region == "parietal" or sub_region in ["somatosensory_cortex"]:
                learning_freq = learning_frequencies["creative"]
                learning_type = "creative"
            
            # Add natural variation to avoid uniform patterns
            # This simulates the natural variation in infant learning
            learning_freq *= (0.95 + 0.1 * np.random.random())
            
            # Set frequency in brain grid, modulated by coherence
            # Higher coherence means clearer learning patterns
            if hasattr(self.brain_grid, 'frequency_grid'):
                self.brain_grid.frequency_grid[x, y, z] = learning_freq * coherence_factor
            
            # Track metrics
            cells_affected += 1
            modality_distribution[learning_type] = modality_distribution.get(learning_type, 0) + 1
            
            # Track region coverage
            if region:
                if region not in region_coverage:
                    region_coverage[region] = 0
                region_coverage[region] += 1
        
        # Calculate coverage
        coverage_percent = cells_affected / active_cell_count * 100.0 if active_cell_count > 0 else 0.0
        
        # Normalize modality distribution to percentages
        for key in modality_distribution:
            modality_distribution[key] = (modality_distribution[key] / cells_affected * 100.0) if cells_affected > 0 else 0.0
        
        # Create propagation metrics
        propagation_metrics = {
            "success": True,
            "cells_affected": cells_affected,
            "coverage_percent": float(coverage_percent),
            "coherence_factor": float(coherence_factor),
            "learning_modalities": modality_distribution,
            "region_coverage": region_coverage,
            "learning_frequencies": {k: float(v) for k, v in learning_frequencies.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        self.last_updated = datetime.now().isoformat()
        logger.info(f"Learning frequencies propagated to {cells_affected} cells ({coverage_percent:.1f}% coverage)")
        
        return propagation_metrics

    def create_memory_pathways(self, aspect_distribution: Dict[str, Any]) -> Dict:
        """
        Create specialized learning pathways for early childhood development.
        Maps soul aspects to foundational learning domains like language acquisition,
        shape recognition, sound processing, and creative expression.
        
        Args:
            aspect_distribution: Dictionary mapping aspect names to strength values
            
        Returns:
            Dict containing memory pathway metrics
        """
        if not self.initialized:
            logger.warning("Mycelial network not initialized. Cannot create learning pathways.")
            return {"success": False, "error": "Network not initialized"}
        
        logger.info(f"Creating early learning pathways for {len(aspect_distribution)} soul aspects")
        
        # Define learning domains and their associated brain regions
        learning_domains = {
            "language": {
                "regions": ["temporal", "frontal"],
                "sub_regions": ["wernicke", "broca"],
                "description": "Language acquisition and processing"
            },
            "visual": {
                "regions": ["occipital"],
                "sub_regions": ["primary_visual", "visual_association"],
                "description": "Visual perception and shape recognition"
            },
            "auditory": {
                "regions": ["temporal"],
                "sub_regions": ["primary_auditory"],
                "description": "Sound processing and recognition"
            },
            "motor": {
                "regions": ["frontal", "cerebellum"],
                "sub_regions": ["motor_cortex"],
                "description": "Motor skills and physical coordination"
            },
            "emotional": {
                "regions": ["limbic"],
                "sub_regions": ["amygdala", "cingulate"],
                "description": "Emotional processing and development"
            },
            "cognitive": {
                "regions": ["prefrontal", "parietal"],
                "sub_regions": ["prefrontal"],
                "description": "Reasoning and problem-solving"
            },
            "creative": {
                "regions": ["parietal", "temporal", "occipital"],
                "sub_regions": ["visual_association"],
                "description": "Creative expression and imagination"
            }
        }
        
        # Track pathways created
        pathways_created = 0
        aspects_connected = 0
        aspect_pathways = {}
        
        # Map aspect types to learning domains
        aspect_domain_mapping = {
            # Language-related aspects
            "verbal": "language",
            "linguistic": "language",
            "communication": "language",
            "speech": "language",
            "word": "language",
            "alphabet": "language",
            
            # Visual-related aspects
            "visual": "visual",
            "spatial": "visual",
            "shape": "visual",
            "color": "visual",
            "geometric": "visual",
            
            # Auditory-related aspects
            "auditory": "auditory",
            "sound": "auditory",
            "musical": "auditory",
            "rhythm": "auditory",
            "harmony": "auditory",
            
            # Movement-related aspects
            "motor": "motor",
            "physical": "motor",
            "movement": "motor",
            "coordination": "motor",
            "dexterity": "motor",
            
            # Emotional aspects
            "emotional": "emotional",
            "feeling": "emotional",
            "empathy": "emotional",
            "sensitivity": "emotional",
            "connection": "emotional",
            
            # Cognitive aspects
            "logical": "cognitive",
            "analytical": "cognitive",
            "problem": "cognitive",
            "reasoning": "cognitive",
            "thinking": "cognitive",
            
            # Creative aspects
            "creative": "creative",
            "artistic": "creative",
            "imaginative": "creative",
            "expressive": "creative",
            "playful": "creative"
        }
        
        # Default mapping for unrecognized aspects
        default_domains = ["cognitive", "emotional"]
        
        # For each aspect, create pathways to appropriate learning domains
        for aspect_name, aspect_data in aspect_distribution.items():
            # Extract aspect strength
            if isinstance(aspect_data, dict):
                aspect_strength = aspect_data.get('strength', 0.5)
            elif isinstance(aspect_data, (int, float)):
                aspect_strength = float(aspect_data)
            else:
                aspect_strength = 0.5  # Default
            
            # Determine learning domain based on aspect name
            domain_match = None
            for keyword, domain in aspect_domain_mapping.items():
                if keyword in aspect_name.lower():
                    domain_match = domain
                    break
            
            # Use default domain if no match
            domains_to_connect = [domain_match] if domain_match else default_domains
            
            # Create pathways to each matching domain
            aspect_paths = []
            
            for domain_name in domains_to_connect:
                domain = learning_domains.get(domain_name)
                if not domain:
                    continue
                
                # Find target regions for this domain
                target_regions = []
                
                # Try sub-regions first (more specific)
                for sub_region in domain["sub_regions"]:
                    if hasattr(self.brain_grid, 'sub_region_grid'):
                        sub_indices = np.where(self.brain_grid.sub_region_grid == sub_region)
                        if len(sub_indices[0]) > 0:
                            target_regions.append((sub_region, sub_indices, "sub"))
                
                # Then try major regions
                for region in domain["regions"]:
                    if hasattr(self.brain_grid, 'region_grid'):
                        region_indices = np.where(self.brain_grid.region_grid == region)
                        if len(region_indices[0]) > 0:
                            target_regions.append((region, region_indices, "major"))
                
                # Skip if no target regions found
                if not target_regions:
                    continue
                
                # Find existing high-density mycelial cells as connection starting points
                mycelial_indices = np.where(self.brain_grid.mycelial_density_grid > 0.3)
                if len(mycelial_indices[0]) == 0:
                    # If no suitable cells, use random cells
                    if hasattr(self.brain_grid, 'region_grid'):
                        random_indices = np.where(self.brain_grid.region_grid != "")
                        if len(random_indices[0]) > 0:
                            mycelial_indices = random_indices
                
                # Skip if still no suitable cells
                if len(mycelial_indices[0]) == 0:
                    continue
                
                # For each target region, create a pathway
                for target_name, target_indices, region_type in target_regions:
                    # Choose start point (mycelial cell)
                    start_idx = np.random.randint(0, len(mycelial_indices[0]))
                    start_x = mycelial_indices[0][start_idx]
                    start_y = mycelial_indices[1][start_idx]
                    start_z = mycelial_indices[2][start_idx]
                    
                    # Choose end point (in target region)
                    end_idx = np.random.randint(0, len(target_indices[0]))
                    end_x = target_indices[0][end_idx]
                    end_y = target_indices[1][end_idx]
                    end_z = target_indices[2][end_idx]
                    
                    # Create pathway with strength based on aspect strength
                    # Stronger aspects make stronger pathways for better learning
                    path_density = 0.4 + 0.4 * aspect_strength
                    
                    # Create pathway with a descriptive name
                    pathway_name = f"learning-{domain_name}-{aspect_name}-to-{target_name}"
                    path_info = self._create_mycelial_pathway(
                        (start_x, start_y, start_z),
                        (end_x, end_y, end_z),
                        path_density,
                        pathway_name
                    )
                    
                    # Record information about this pathway
                    learning_path = {
                        "learning_domain": domain_name,
                        "target_region": target_name,
                        "region_type": region_type,
                        "path_info": path_info,
                        "path_strength": path_density,
                        "description": domain["description"]
                    }
                    
                    aspect_paths.append(learning_path)
                    pathways_created += 1
            
            # Record pathways for this aspect
            if aspect_paths:
                aspect_pathways[aspect_name] = aspect_paths
                aspects_connected += 1
        
        # After creating all pathways, strengthen special learning connections
        self._strengthen_early_learning_connections()
        
        # Calculate memory pathway metrics
        memory_metrics = {
            "success": True,
            "pathways_created": pathways_created,
            "aspects_connected": aspects_connected,
            "learning_domains_covered": list(set([path["learning_domain"] for paths in aspect_pathways.values() for path in paths])),
            "aspect_pathways_count": {k: len(v) for k, v in aspect_pathways.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        self.last_updated = datetime.now().isoformat()
        logger.info(f"Created {pathways_created} learning pathways for {aspects_connected} aspects")
        
        return memory_metrics

    def _strengthen_early_learning_connections(self):
        """
        Create reinforced connections between key learning areas
        that are important for early development.
        """
        # Key connections for early learning (cross-domain integration)
        key_connections = [
            # Language-Auditory connection (crucial for speech development)
            {
                "source_region": "wernicke",
                "target_region": "primary_auditory",
                "strength": 0.9,
                "name": "language-sound-integration"
            },
            # Visual-Motor connection (eye-hand coordination)
            {
                "source_region": "primary_visual",
                "target_region": "motor_cortex",
                "strength": 0.85,
                "name": "visual-motor-coordination"
            },
            # Emotional-Language connection (emotional expression)
            {
                "source_region": "amygdala",
                "target_region": "broca",
                "strength": 0.8,
                "name": "emotional-expression"
            },
            # Visual-Language connection (reading preparation)
            {
                "source_region": "visual_association",
                "target_region": "wernicke",
                "strength": 0.85,
                "name": "visual-language-integration"
            },
            # Auditory-Motor connection (rhythm and movement)
            {
                "source_region": "primary_auditory",
                "target_region": "cerebellum",
                "strength": 0.8,
                "name": "sound-movement-coordination"
            }
        ]
        
        # Create each key connection
        for connection in key_connections:
            source_region = connection["source_region"]
            target_region = connection["target_region"]
            
            # Find source and target regions
            source_indices = np.where(self.brain_grid.sub_region_grid == source_region)
            target_indices = np.where(self.brain_grid.sub_region_grid == target_region)
            
            # Skip if either region is missing
            if len(source_indices[0]) == 0 or len(target_indices[0]) == 0:
                continue
            
            # Choose start and end points
            source_idx = len(source_indices[0]) // 2  # Use middle point for stability
            source_x = source_indices[0][source_idx]
            source_y = source_indices[1][source_idx]
            source_z = source_indices[2][source_idx]
            
            target_idx = len(target_indices[0]) // 2  # Use middle point for stability
            target_x = target_indices[0][target_idx]
            target_y = target_indices[1][target_idx]
            target_z = target_indices[2][target_idx]
            
            # Create a strong pathway
            self._create_mycelial_pathway(
                (source_x, source_y, source_z),
                (target_x, target_y, target_z),
                connection["strength"],
                f"early-learning-{connection['name']}"
            )
            
        logger.info("Strengthened key early learning connections")

    def get_network_metrics(self) -> Dict:
        """
        Get comprehensive metrics about the mycelial network state.
        
        Returns:
            Dict containing network metrics
        """
        if not self.initialized:
            return {
                "initialized": False,
                "error": "Network not initialized"
            }
        
        # Calculate coverage
        total_cells = np.prod(self.dimensions)
        cells_with_mycelium = np.sum(self.brain_grid.mycelial_density_grid > 0.05)
        coverage_percent = cells_with_mycelium / total_cells * 100.0
        
        # Calculate energy metrics if energy is flowing
        energy_metrics = {}
        if self.energy_flowing:
            total_energy = np.sum(self.brain_grid.mycelial_energy_grid)
            energized_cells = np.sum(self.brain_grid.mycelial_energy_grid > 0.01)
            energy_coverage = energized_cells / cells_with_mycelium * 100.0 if cells_with_mycelium > 0 else 0.0
            
            # Regional energy distribution
            region_energy = {}
            for region in MAJOR_REGIONS:
                region_indices = np.where(self.brain_grid.region_grid == region)
                if len(region_indices[0]) > 0:
                    region_energy[region] = float(np.sum(
                        self.brain_grid.mycelial_energy_grid[region_indices]
                    ))
            
            energy_metrics = {
                "total_energy": float(total_energy),
                "energized_cells": int(energized_cells),
                "energy_coverage_percent": float(energy_coverage),
                "region_energy_distribution": region_energy
            }
        
        # Compile network metrics
        network_metrics = {
            "initialized": self.initialized,
            "energy_flowing": self.energy_flowing,
            "dimensions": self.dimensions,
            "seed_points": len(self.seed_points),
            "major_pathways": len(self.major_pathways),
            "total_branches": self.branch_count,
            "cells_with_mycelium": int(cells_with_mycelium),
            "coverage_percent": float(coverage_percent),
            "creation_time": self.creation_time,
            "last_updated": self.last_updated
        }
        
        # Add energy metrics if available
        if energy_metrics:
            network_metrics.update({"energy": energy_metrics})
        
        return network_metrics

    def visualize_network(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate visualization data for the mycelial network.
        
        Args:
            output_path: Optional path to save visualization data
            
        Returns:
            Dict containing visualization metrics and paths
        """
        # This is a placeholder for actual visualization code
        # In a full implementation, this would generate 3D visualization data
        
        if not self.initialized:
            return {"success": False, "error": "Network not initialized"}
        
        logger.info("Generating mycelial network visualization data")
        
        visualization_data = {
            "dimensions": self.dimensions,
            "seed_points": [
                {"position": [x, y, z], "region": region}
                for x, y, z, region in self.seed_points
            ],
            "pathways": [
                {
                    "name": pathway["name"],
                    "start": pathway["start"],
                    "end": pathway["end"],
                    "density": pathway["base_density"]
                }
                for pathway in self.major_pathways
            ],
            "network_metrics": self.get_network_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save visualization data if path provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(visualization_data, f, indent=2, default=str)
                logger.info(f"Visualization data saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving visualization data: {e}")
                return {"success": False, "error": f"Failed to save data: {e}"}
        
        return {
            "success": True,
            "visualization_data": "Generated",
            "output_path": output_path if output_path else None
        }

    def reset_network(self):
        """Reset the mycelial network to an uninitialized state."""
        logger.info("Resetting mycelial network")
        
        # Reset grid arrays
        self.brain_grid.mycelial_density_grid.fill(0.0)
        self.brain_grid.mycelial_energy_grid.fill(0.0)
        
        # Reset tracking
        self.pathway_count = 0
        self.branch_count = 0
        self.seed_points = []
        self.major_pathways = []
        self.initialized = False
        self.energy_flowing = False
        
        # Update timestamp
        self.last_updated = datetime.now().isoformat()
        
        logger.info("Mycelial network reset complete")
        return True





















