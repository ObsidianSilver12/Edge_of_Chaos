# --- START OF FILE stage_1/evolve/brain_structure/brain_structure.py ---

"""
Brain Structure Implementation (Simplified V5.0)

Creates a block-based 3D structure for the brain with regions, subregions, and blocks.
Uses sparse representation for active cells and predefined boundaries.
Optimized for performance with minimal memory footprint.
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

# Import constants
from stage_1.evolve.core.evolve_constants import *
from constants.constants import *
# Import brain structure dictionary model
from stage_1.evolve.brain_structure.brain_structure_dictionary import get_brain_structure_dictionary

# --- Logging Setup ---
logger = logging.getLogger("BrainStructure")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Try to import metrics tracking ---
try:
    import metrics_tracking
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics_tracking = MetricsPlaceholder()


class BrainGrid:
    """3D grid representing the brain with simplified block-based structure."""
    
    def __init__(self, dimensions: Tuple[int, int, int] = GRID_DIMENSIONS):
        """
        Initialize brain structure with simplified grid.
        
        Args:
            dimensions: 3D dimensions for brain grid (x, y, z)
        """
        logger.info(f"Initializing simplified brain grid with dimensions {dimensions}")
        
        self.dimensions = dimensions
        self.brain_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        
        # Load brain structure dictionary model
        self.structure_model = get_brain_structure_dictionary()
        
        # Core sparse data structures
        # These will store only active cells to minimize memory usage
        
        # Region mapping - stores for each coordinate which region it belongs to
        self.region_grid = {}  # (x,y,z) -> region_name
        self.sub_region_grid = {}  # (x,y,z) -> subregion_name
        
        # Field value tracking (sparse representation)
        self.energy_grid = {}          # (x,y,z) -> energy value
        self.frequency_grid = {}       # (x,y,z) -> frequency value
        self.resonance_grid = {}       # (x,y,z) -> resonance value
        self.stability_grid = {}       # (x,y,z) -> stability value
        self.coherence_grid = {}       # (x,y,z) -> coherence value
        
        # Mycelial tracking
        self.mycelial_density_grid = {}       # (x,y,z) -> mycelial density
        self.mycelial_energy_grid = {}        # (x,y,z) -> mycelial energy
        
        # Soul presence tracking
        self.soul_presence_grid = {}          # (x,y,z) -> soul presence
        self.soul_frequency_grid = {}         # (x,y,z) -> soul frequency
        
        # Structure definitions
        self.regions = {}                # Region definitions
        self.sub_regions = {}            # Sub-region definitions
        self.blocks = {}                 # Block definitions
        self.boundaries = {}             # Boundary definitions
        
        # Cell tracking
        self.active_cell_count = 0
        self.boundary_cell_count = 0
        self.soul_filled_cells = 0
        
        # Status tracking
        self.regions_defined = False
        self.sub_regions_defined = False
        self.boundaries_defined = False
        self.field_initialized = False
        self.mycelial_paths_formed = False
        
        # Statistics
        self.total_grid_cells = dimensions[0] * dimensions[1] * dimensions[2]
        
        # Block size (dividing brain into blocks for efficient processing)
        # Using larger blocks to reduce the total number of blocks
        self.block_size = (
            max(4, dimensions[0] // 32),
            max(4, dimensions[1] // 32),
            max(4, dimensions[2] // 32)
        )
        
        logger.info(f"Brain structure initialized with block size {self.block_size}")
    
    def define_major_regions(self) -> bool:
        """
        Define major brain regions based on proportions and locations.
        
        Returns:
            True if successful
        """
        logger.info("Defining major brain regions")
        
        # Clear any existing regions
        self.regions = {}
        
        # Define each region from constants
        for region_name, proportion in REGION_PROPORTIONS.items():
            # Get region center from constants
            region_center = REGION_LOCATIONS.get(region_name, (0.5, 0.5, 0.5))
            
            # Convert to grid coordinates
            center_x = int(region_center[0] * self.dimensions[0])
            center_y = int(region_center[1] * self.dimensions[1])
            center_z = int(region_center[2] * self.dimensions[2])
            
            # Define region
            self.regions[region_name] = {
                'name': region_name,
                'center': (center_x, center_y, center_z),
                'proportion': proportion,
                'volume': int(proportion * self.total_grid_cells),
                'default_frequency': REGION_DEFAULT_FREQUENCIES.get(region_name, SCHUMANN_FREQUENCY),
                'creation_time': datetime.now().isoformat()
            }
            
            logger.info(f"Defined region {region_name} with proportion {proportion:.2f}")
        
        # Fill the region grid with appropriate regions
        self._fill_region_grid()
        
        self.regions_defined = True
        self.last_updated = datetime.now().isoformat()
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'brain_id': self.brain_id,
                    'timestamp': datetime.now().isoformat(),
                    'regions_defined': len(self.regions),
                    'regions': list(self.regions.keys())
                }
                metrics_tracking.record_metrics("brain_structure_regions", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record region metrics: {e}")
        
        return True
    
    def _fill_region_grid(self) -> None:
        """
        Fill the region grid with appropriate regions.
        This uses a distance-based approach to assign regions.
        """
        logger.info("Filling region grid (sparse representation)")
        
        # Create blocks for more efficient region assignment
        self._create_region_blocks()
        
        # Process each block
        for block_id, block in self.blocks.items():
            # Get block region
            region_name = block.get('region')
            if not region_name:
                continue
            
            # Get block position and size
            bx, by, bz = block['position']
            bsx, bsy, bsz = block['size']
            
            # Mark cells in this block with region
            # Only mark active cells to save memory
            # Most cells in internal regions can remain inactive
            is_boundary_block = block.get('is_boundary', False)
            
            # For boundary blocks, mark all cells for accuracy
            if is_boundary_block:
                # Mark all cells in boundary blocks
                for x in range(bx, bx + bsx):
                    for y in range(by, by + bsy):
                        for z in range(bz, bz + bsz):
                            self.region_grid[(x, y, z)] = region_name
            else:
                # For internal blocks, mark fewer cells (sparse)
                # Use higher density near center of region for accuracy
                region_center = self.regions[region_name]['center']
                center_distance = np.sqrt(
                    (bx - region_center[0])**2 + 
                    (by - region_center[1])**2 + 
                    (bz - region_center[2])**2
                )
                
                # Calculate density based on distance from center
                # Higher density near center, lower at periphery
                if center_distance < 20:
                    density = 0.2  # 20% near center
                elif center_distance < 40:
                    density = 0.1  # 10% middle distance
                else:
                    density = 0.05  # 5% far from center
                
                # Mark cells with this density
                for x in range(bx, bx + bsx):
                    for y in range(by, by + bsy):
                        for z in range(bz, bz + bsz):
                            if random.random() < density:
                                self.region_grid[(x, y, z)] = region_name
        
        # Count cells assigned
        region_counts = {region: 0 for region in self.regions}
        for region in self.region_grid.values():
            region_counts[region] = region_counts.get(region, 0) + 1
        
        logger.info(f"Region grid filled with {len(self.region_grid)} cells marked")
        for region, count in region_counts.items():
            logger.info(f"  {region}: {count} cells")
    
    def _create_region_blocks(self) -> None:
        """Create blocks for each region."""
        logger.info("Creating region blocks")
        
        # Calculate block grid dimensions
        block_grid_x = self.dimensions[0] // self.block_size[0]
        block_grid_y = self.dimensions[1] // self.block_size[1]
        block_grid_z = self.dimensions[2] // self.block_size[2]
        
        # Create blocks
        block_count = 0
        
        # Find closest region for each block
        for bx in range(block_grid_x):
            for by in range(block_grid_y):
                for bz in range(block_grid_z):
                    # Calculate block center
                    center_x = bx * self.block_size[0] + self.block_size[0] // 2
                    center_y = by * self.block_size[1] + self.block_size[1] // 2
                    center_z = bz * self.block_size[2] + self.block_size[2] // 2
                    
                    # Find closest region
                    closest_region = None
                    min_distance = float('inf')
                    
                    for region_name, region in self.regions.items():
                        rx, ry, rz = region['center']
                        
                        # Calculate distance
                        distance = math.sqrt(
                            (center_x - rx)**2 + 
                            (center_y - ry)**2 + 
                            (center_z - rz)**2
                        )
                        
                        # Adjust distance by region importance
                        # More important regions get priority
                        adjusted_distance = distance / (region['proportion'] * 10)
                        
                        if adjusted_distance < min_distance:
                            min_distance = adjusted_distance
                            closest_region = region_name
                    
                    # Create block ID
                    block_id = f"B_{bx}_{by}_{bz}"
                    
                    # Calculate block position
                    block_x = bx * self.block_size[0]
                    block_y = by * self.block_size[1]
                    block_z = bz * self.block_size[2]
                    
                    # Create block
                    self.blocks[block_id] = {
                        'id': block_id,
                        'grid_position': (bx, by, bz),
                        'position': (block_x, block_y, block_z),
                        'size': self.block_size,
                        'region': closest_region,
                        'sub_region': None,  # To be assigned later
                        'is_boundary': False,  # To be determined
                        'creation_time': datetime.now().isoformat()
                    }
                    
                    block_count += 1
        
        # Count blocks per region
        region_block_counts = {region: 0 for region in self.regions}
        for block in self.blocks.values():
            region = block['region']
            region_block_counts[region] = region_block_counts.get(region, 0) + 1
        
        logger.info(f"Created {block_count} blocks across {len(self.regions)} regions")
        for region, count in region_block_counts.items():
            logger.info(f"  {region}: {count} blocks")
    
    def define_sub_regions(self) -> bool:
        """
        Define sub-regions within major regions.
        
        Returns:
            True if successful
        """
        logger.info("Defining sub-regions within major regions")
        
        # Ensure regions are defined
        if not self.regions_defined:
            logger.warning("Regions not defined. Defining regions first.")
            self.define_major_regions()
        
        # Clear existing sub-regions
        self.sub_regions = {}
        
        # Define sub-regions for each region
        for region_name, region in self.regions.items():
            # Determine number of sub-regions based on region importance
            num_sub_regions = max(2, int(region['proportion'] * 10))
            
            # For each sub-region
            for i in range(num_sub_regions):
                # Sub-region name
                sub_region_name = f"{region_name}_{i+1}"
                
                # Calculate offset from region center
                # This distributes sub-regions within the region
                angle = 2 * np.pi * i / num_sub_regions
                offset_factor = 0.3  # How far from center
                
                offset_x = int(np.cos(angle) * offset_factor * self.dimensions[0] * region['proportion'])
                offset_y = int(np.sin(angle) * offset_factor * self.dimensions[1] * region['proportion'])
                offset_z = int((i % 3 - 1) * offset_factor * self.dimensions[2] * region['proportion'] / 2)
                
                # Calculate sub-region center
                rx, ry, rz = region['center']
                center_x = min(max(0, rx + offset_x), self.dimensions[0] - 1)
                center_y = min(max(0, ry + offset_y), self.dimensions[1] - 1)
                center_z = min(max(0, rz + offset_z), self.dimensions[2] - 1)
                
                # Create sub-region
                sub_region = {
                    'name': sub_region_name,
                    'parent_region': region_name,
                    'center': (center_x, center_y, center_z),
                    'proportion': region['proportion'] / num_sub_regions,
                    'creation_time': datetime.now().isoformat()
                }
                
                self.sub_regions[sub_region_name] = sub_region
                
                logger.info(f"Defined sub-region {sub_region_name} in {region_name}")
        
        # Fill sub-region grid
        self._fill_subregion_grid()
        
        self.sub_regions_defined = True
        self.last_updated = datetime.now().isoformat()
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'brain_id': self.brain_id,
                    'timestamp': datetime.now().isoformat(),
                    'sub_regions_defined': len(self.sub_regions),
                    'sub_regions_per_region': {
                        r: sum(1 for sr in self.sub_regions.values() if sr['parent_region'] == r)
                        for r in self.regions
                    }
                }
                metrics_tracking.record_metrics("brain_structure_subregions", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record sub-region metrics: {e}")
        
        return True
    
    def _fill_subregion_grid(self) -> None:
        """
        Fill the sub-region grid with appropriate sub-regions.
        Only fills cells that already have a region assigned.
        """
        logger.info("Filling sub-region grid (sparse representation)")
        
        # Process cells that already have a region assigned
        for coord, region_name in self.region_grid.items():
            # Find sub-regions for this region
            region_subregions = [
                sr for sr_name, sr in self.sub_regions.items()
                if sr['parent_region'] == region_name
            ]
            
            if not region_subregions:
                continue
            
            # Find closest sub-region
            closest_subregion = None
            min_distance = float('inf')
            
            x, y, z = coord
            
            for subregion in region_subregions:
                sx, sy, sz = subregion['center']
                
                # Calculate distance
                distance = math.sqrt(
                    (x - sx)**2 + 
                    (y - sy)**2 + 
                    (z - sz)**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_subregion = subregion
            
            # Assign sub-region
            if closest_subregion:
                self.sub_region_grid[coord] = closest_subregion['name']
        
        # Count sub-region cells
        subregion_counts = {sr: 0 for sr in self.sub_regions}
        for subregion in self.sub_region_grid.values():
            subregion_counts[subregion] = subregion_counts.get(subregion, 0) + 1
        
        logger.info(f"Sub-region grid filled with {len(self.sub_region_grid)} cells marked")
    
    def define_boundaries(self) -> bool:
        """
        Define boundaries between brain regions.
        Creates high-density boundary cells for wave transmission.
        
        Returns:
            True if successful
        """
        logger.info("Defining region boundaries")
        
        # Ensure regions are defined
        if not self.regions_defined:
            logger.warning("Regions not defined. Defining regions first.")
            self.define_major_regions()
            
        if not self.sub_regions_defined:
            logger.warning("Sub-regions not defined. Defining sub-regions first.")
            self.define_sub_regions()
        
        # Clear existing boundaries
        self.boundaries = {}
        boundaries_created = 0
        boundary_cells = 0
        
        # For each region pair, check cells for boundaries
        processed_pairs = set()
        
        # Iterate through cells with region assignment
        for coord, region in self.region_grid.items():
            x, y, z = coord
            
            # Check 6-connected neighboring cells
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                neighbor_coord = (nx, ny, nz)
                
                # Skip if outside grid
                if not (0 <= nx < self.dimensions[0] and
                        0 <= ny < self.dimensions[1] and
                        0 <= nz < self.dimensions[2]):
                    continue
                
                # Skip if neighbor doesn't have a region
                if neighbor_coord not in self.region_grid:
                    continue
                
                neighbor_region = self.region_grid[neighbor_coord]
                
                # Skip if same region
                if neighbor_region == region:
                    continue
                
                # Create boundary key (consistently ordered)
                regions_key = tuple(sorted([region, neighbor_region]))
                
                # Skip if already processed
                if regions_key in processed_pairs:
                    continue
                
                processed_pairs.add(regions_key)
                
                # Get boundary type
                boundary_type = REGION_BOUNDARIES.get(
                    (regions_key[0], regions_key[1]), 
                    REGION_BOUNDARIES.get(
                        (regions_key[1], regions_key[0]), 
                        BOUNDARY_TYPE_GRADUAL
                    )
                )
                
                # Get boundary parameters
                params = BOUNDARY_PARAMETERS.get(boundary_type, BOUNDARY_PARAMETERS[BOUNDARY_TYPE_GRADUAL])
                
                # Create boundary
                boundary_id = f"B_{regions_key[0]}_{regions_key[1]}"
                
                self.boundaries[boundary_id] = {
                    'id': boundary_id,
                    'regions': regions_key,
                    'type': boundary_type,
                    'transition_width': params['transition_width'],
                    'permeability': params['permeability'],
                    'creation_time': datetime.now().isoformat()
                }
                
                boundaries_created += 1
                
                # Mark boundary cells
                # This is where we create high-density boundary cells
                self._create_boundary_cells(regions_key, boundary_type, params)
                
                logger.debug(f"Created boundary {boundary_id} between {regions_key[0]} and {regions_key[1]}")
        
        # Count boundary cells
        boundary_cells = sum(1 for coord, value in self.resonance_grid.items() 
                          if value > 0.7 and coord in self.region_grid)
        
        self.boundary_cell_count = boundary_cells
        self.boundaries_defined = True
        self.last_updated = datetime.now().isoformat()
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'brain_id': self.brain_id,
                    'timestamp': datetime.now().isoformat(),
                    'boundaries_created': boundaries_created,
                    'boundary_cells': boundary_cells
                }
                metrics_tracking.record_metrics("brain_structure_boundaries", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record boundary metrics: {e}")
        
        logger.info(f"Defined {boundaries_created} boundaries with {boundary_cells} boundary cells")
        return True
    
    def _create_boundary_cells(self, regions_pair: Tuple[str, str], boundary_type: str, params: Dict[str, Any]) -> None:
        """
        Create high-density boundary cells between two regions.
        
        Args:
            regions_pair: Tuple of two region names
            boundary_type: Type of boundary
            params: Boundary parameters
        """
        # Find all cells in first region that are adjacent to second region
        boundary_coords = []
        
        # Find cells in first region
        region1_coords = [coord for coord, r in self.region_grid.items() if r == regions_pair[0]]
        region2_coords = [coord for coord, r in self.region_grid.items() if r == regions_pair[1]]
        
        # Convert to sets for faster lookups
        region1_set = set(region1_coords)
        region2_set = set(region2_coords)
        
        # Find boundary cells - cells in region1 adjacent to region2
        for coord in region1_coords:
            x, y, z = coord
            
            # Check 6-connected neighbors
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                neighbor_coord = (nx, ny, nz)
                
                # Skip if outside grid
                if not (0 <= nx < self.dimensions[0] and
                        0 <= ny < self.dimensions[1] and
                        0 <= nz < self.dimensions[2]):
                    continue
                
                # Check if neighbor is in region2
                if neighbor_coord in region2_set:
                    boundary_coords.append(coord)
                    break
        
        # Do the same for region2 cells adjacent to region1
        for coord in region2_coords:
            x, y, z = coord
            
            # Check 6-connected neighbors
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                neighbor_coord = (nx, ny, nz)
                
                # Skip if outside grid
                if not (0 <= nx < self.dimensions[0] and
                        0 <= ny < self.dimensions[1] and
                        0 <= nz < self.dimensions[2]):
                    continue
                
                # Check if neighbor is in region1
                if neighbor_coord in region1_set:
                    boundary_coords.append(coord)
                    break
        
        # Set boundary cell properties
        for coord in boundary_coords:
            # Set high resonance for boundary cells
            self.resonance_grid[coord] = 0.8
            
            # Set frequency based on boundary type
            if boundary_type == BOUNDARY_TYPE_SHARP:
                # Sharp boundaries have higher frequencies
                frequency = 15.0 + random.uniform(-1.0, 1.0)
            elif boundary_type == BOUNDARY_TYPE_GRADUAL:
                # Gradual boundaries have moderate frequencies
                frequency = 10.0 + random.uniform(-1.0, 1.0)
            else:  # BOUNDARY_TYPE_PERMEABLE
                # Permeable boundaries have lower frequencies
                frequency = 7.0 + random.uniform(-1.0, 1.0)
            
            self.frequency_grid[coord] = frequency
            
            # Set coherence based on permeability
            self.coherence_grid[coord] = params['permeability']
            
            # Set stability
            self.stability_grid[coord] = 0.7
            
            # Set initial energy
            self.energy_grid[coord] = 0.1
    
    def initialize_base_fields(self) -> bool:
        """
        Initialize base field values across the brain.
        Applies frequency, resonance and other field properties.
        
        Returns:
            True if successful
        """
        logger.info("Initializing base field values")
        
        # Make sure regions and boundaries are defined
        if not self.regions_defined:
            logger.warning("Regions not defined. Defining regions first.")
            self.define_major_regions()
            
        if not self.sub_regions_defined:
            logger.warning("Sub-regions not defined. Defining sub-regions first.")
            self.define_sub_regions()
            
        if not self.boundaries_defined:
            logger.warning("Boundaries not defined. Defining boundaries first.")
            self.define_boundaries()
        
        # Field initialization is more efficient with sparse representation
        # We only initialize cells that are already in the region grid
        cells_initialized = 0
        
        # Initialize field values for cells with region assignment
        for coord, region in self.region_grid.items():
            # Skip if already a boundary cell (already initialized)
            if coord in self.resonance_grid and self.resonance_grid[coord] > 0.7:
                continue
            
            # Get region default frequency
            region_freq = self.regions[region]['default_frequency']
            
            # Add slight variation to frequency
            freq_variation = 1.0 + 0.1 * (random.random() - 0.5)
            frequency = region_freq * freq_variation
            
            # Set field values
            self.frequency_grid[coord] = frequency
            self.resonance_grid[coord] = 0.5  # Medium resonance
            self.stability_grid[coord] = 0.5  # Medium stability
            self.coherence_grid[coord] = 0.5  # Medium coherence
            self.energy_grid[coord] = 0.1  # Low initial energy
            
            # Initial mycelial values
            self.mycelial_density_grid[coord] = 0.1  # Low initial density
            
            cells_initialized += 1
        
        self.field_initialized = True
        self.last_updated = datetime.now().isoformat()
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'brain_id': self.brain_id,
                    'timestamp': datetime.now().isoformat(),
                    'cells_initialized': cells_initialized,
                    'boundary_cells': self.boundary_cell_count,
                    'total_cells': len(self.region_grid)
                }
                metrics_tracking.record_metrics("brain_structure_field_init", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record field initialization metrics: {e}")
        
        logger.info(f"Initialized fields with {cells_initialized} cells")
        return True
    
    def get_region_at_position(self, position: Tuple[int, int, int]) -> Optional[str]:
        """
        Get region at specified position.
        
        Args:
            position: (x, y, z) position
            
        Returns:
            Region name or None if not assigned
        """
        return self.region_grid.get(position)
    
    def get_subregion_at_position(self, position: Tuple[int, int, int]) -> Optional[str]:
        """
        Get subregion at specified position.
        
        Args:
            position: (x, y, z) position
            
        Returns:
            Subregion name or None if not assigned
        """
        return self.sub_region_grid.get(position)
    
    def get_field_value(self, position: Tuple[int, int, int], field_type: str) -> float:
        """
        Get field value at specified position.
        
        Args:
            position: (x, y, z) position
            field_type: Field type (energy, frequency, resonance, etc.)
            
        Returns:
            Field value or 0.0 if not set
        """
        if field_type == 'energy':
            return self.energy_grid.get(position, 0.0)
        elif field_type == 'frequency':
            return self.frequency_grid.get(position, 0.0)
        elif field_type == 'resonance':
            return self.resonance_grid.get(position, 0.0)
        elif field_type == 'stability':
            return self.stability_grid.get(position, 0.0)
        elif field_type == 'coherence':
            return self.coherence_grid.get(position, 0.0)
        elif field_type == 'mycelial_density':
            return self.mycelial_density_grid.get(position, 0.0)
        elif field_type == 'mycelial_energy':
            return self.mycelial_energy_grid.get(position, 0.0)
        elif field_type == 'soul_presence':
            return self.soul_presence_grid.get(position, 0.0)
        else:
            return 0.0
    
    def set_field_value(self, position: Tuple[int, int, int], field_type: str, value: float) -> bool:
            """
            Set field value at specified position.
            
            Args:
                position: (x, y, z) position
                field_type: Field type (energy, frequency, resonance, etc.)
                value: Value to set
                
            Returns:
                True if set successfully, False otherwise
            """
            # Check if position is valid
            x, y, z = position
            if not (0 <= x < self.dimensions[0] and 
                    0 <= y < self.dimensions[1] and 
                    0 <= z < self.dimensions[2]):
                return False
            
            # Set value
            if field_type == 'energy':
                self.energy_grid[position] = value
            elif field_type == 'frequency':
                self.frequency_grid[position] = value
            elif field_type == 'resonance':
                self.resonance_grid[position] = value
            elif field_type == 'stability':
                self.stability_grid[position] = value
            elif field_type == 'coherence':
                self.coherence_grid[position] = value
            elif field_type == 'mycelial_density':
                self.mycelial_density_grid[position] = value
            elif field_type == 'mycelial_energy':
                self.mycelial_energy_grid[position] = value
            elif field_type == 'soul_presence':
                self.soul_presence_grid[position] = value
            else:
                return False
            
            # If position doesn't have a region, assign the closest one
            if position not in self.region_grid:
                region = self._find_closest_region(position)
                if region:
                    self.region_grid[position] = region
                
                # Also try to assign a subregion
                if region and position not in self.sub_region_grid:
                    subregion = self._find_closest_subregion(position, region)
                    if subregion:
                        self.sub_region_grid[position] = subregion
            
            return True
        
    def _find_closest_region(self, position: Tuple[int, int, int]) -> Optional[str]:
        """Find closest region to the given position."""
        x, y, z = position
        closest_region = None
        min_distance = float('inf')
        
        for region_name, region in self.regions.items():
            rx, ry, rz = region['center']
            
            # Calculate distance
            distance = math.sqrt(
                (x - rx)**2 + (y - ry)**2 + (z - rz)**2
            )
            
            # Adjust distance by region importance
            adjusted_distance = distance / (region['proportion'] * 10)
            
            if adjusted_distance < min_distance:
                min_distance = adjusted_distance
                closest_region = region_name
        
        return closest_region

    def _find_closest_subregion(self, position: Tuple[int, int, int], region: str) -> Optional[str]:
        """Find closest subregion in the given region to the position."""
        x, y, z = position
        closest_subregion = None
        min_distance = float('inf')
        
        # Find subregions in this region
        for subregion_name, subregion in self.sub_regions.items():
            if subregion['parent_region'] != region:
                continue
                
            sx, sy, sz = subregion['center']
            
            # Calculate distance
            distance = math.sqrt(
                (x - sx)**2 + (y - sy)**2 + (z - sz)**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_subregion = subregion_name
        
        return closest_subregion

    def find_optimal_seed_position(self) -> Tuple[int, int, int]:
        """
        Find optimal position for brain seed placement.
        Looks for limbic or brain stem regions.
        
        Returns:
            (x, y, z) coordinates for optimal seed position
        """
        logger.info("Finding optimal seed position")
        
        # Look for limbic region cells
        limbic_cells = [pos for pos, region in self.region_grid.items() 
                        if region == REGION_LIMBIC]
        
        if limbic_cells:
            # Find cell near center of limbic region
            limbic_center = self.regions[REGION_LIMBIC]['center']
            closest_cell = min(limbic_cells, 
                                key=lambda pos: math.sqrt(
                                    (pos[0] - limbic_center[0])**2 + 
                                    (pos[1] - limbic_center[1])**2 + 
                                    (pos[2] - limbic_center[2])**2
                                ))
            
            logger.info(f"Found optimal seed position in limbic region: {closest_cell}")
            return closest_cell
        
        # Check brain stem as backup
        brain_stem_cells = [pos for pos, region in self.region_grid.items() 
                            if region == REGION_BRAIN_STEM]
        
        if brain_stem_cells:
            # Find cell near center of brain stem
            brain_stem_center = self.regions[REGION_BRAIN_STEM]['center']
            closest_cell = min(brain_stem_cells, 
                                key=lambda pos: math.sqrt(
                                    (pos[0] - brain_stem_center[0])**2 + 
                                    (pos[1] - brain_stem_center[1])**2 + 
                                    (pos[2] - brain_stem_center[2])**2
                                ))
            
            logger.info(f"Found optimal seed position in brain stem: {closest_cell}")
            return closest_cell
        
        # Fallback to brain center
        center = (
            self.dimensions[0] // 2,
            self.dimensions[1] // 2,
            self.dimensions[2] // 2
        )
        logger.info(f"Falling back to brain center for seed position: {center}")
        return center

    def get_cells_by_region(self, region_name: str) -> List[Tuple[int, int, int]]:
        """
        Get all cell positions in a specific region.
        
        Args:
            region_name: Region name
            
        Returns:
            List of (x, y, z) positions
        """
        return [pos for pos, region in self.region_grid.items() if region == region_name]

    def get_cells_by_subregion(self, subregion_name: str) -> List[Tuple[int, int, int]]:
        """
        Get all cell positions in a specific subregion.
        
        Args:
            subregion_name: Subregion name
            
        Returns:
            List of (x, y, z) positions
        """
        return [pos for pos, subregion in self.sub_region_grid.items() if subregion == subregion_name]

    def get_cells_by_frequency(self, target_freq: float, tolerance: float = 1.0) -> List[Tuple[int, int, int]]:
        """
        Find cells with frequencies close to target.
        
        Args:
            target_freq: Target frequency
            tolerance: Allowable frequency deviation
            
        Returns:
            List of (x, y, z) positions
        """
        return [pos for pos, freq in self.frequency_grid.items() 
                if abs(freq - target_freq) <= tolerance]

    def get_region_size(self, region_name: str) -> Tuple[int, int, int]:
        """
        Get approximate region size in grid units.
        
        Args:
            region_name: Region name
            
        Returns:
            (x_size, y_size, z_size) size tuple
        """
        if region_name not in self.regions:
            return (0, 0, 0)
        
        # Get all cells in region
        region_cells = self.get_cells_by_region(region_name)
        
        if not region_cells:
            return (0, 0, 0)
        
        # Calculate min/max coordinates
        min_x = min(pos[0] for pos in region_cells)
        max_x = max(pos[0] for pos in region_cells)
        min_y = min(pos[1] for pos in region_cells)
        max_y = max(pos[1] for pos in region_cells)
        min_z = min(pos[2] for pos in region_cells)
        max_z = max(pos[2] for pos in region_cells)
        
        # Calculate dimensions
        x_size = max_x - min_x + 1
        y_size = max_y - min_y + 1
        z_size = max_z - min_z + 1
        
        return (x_size, y_size, z_size)

    def prepare_for_soul_distribution(self) -> Dict[str, Any]:
        """
        Prepare brain structure for soul distribution.
        Checks readiness and returns status.
        
        Returns:
            Dict with preparation status
        """
        preparation_status = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'messages': [],
            'preparations_completed': []
        }
        
        # Check regions
        if not self.regions_defined:
            try:
                self.define_major_regions()
                preparation_status['preparations_completed'].append('regions_defined')
                preparation_status['messages'].append("Major regions defined successfully.")
            except Exception as e:
                preparation_status['success'] = False
                preparation_status['messages'].append(f"Failed to define major regions: {e}")
                return preparation_status
        
        # Check sub-regions
        if not self.sub_regions_defined:
            try:
                self.define_sub_regions()
                preparation_status['preparations_completed'].append('sub_regions_defined')
                preparation_status['messages'].append("Sub-regions defined successfully.")
            except Exception as e:
                preparation_status['success'] = False
                preparation_status['messages'].append(f"Failed to define sub-regions: {e}")
                return preparation_status
        
        # Check boundaries
        if not self.boundaries_defined:
            try:
                self.define_boundaries()
                preparation_status['preparations_completed'].append('boundaries_defined')
                preparation_status['messages'].append("Boundaries defined successfully.")
            except Exception as e:
                preparation_status['success'] = False
                preparation_status['messages'].append(f"Failed to define boundaries: {e}")
                return preparation_status
        
        # Check field initialization
        if not self.field_initialized:
            try:
                self.initialize_base_fields()
                preparation_status['preparations_completed'].append('field_initialized')
                preparation_status['messages'].append("Base fields initialized successfully.")
            except Exception as e:
                preparation_status['success'] = False
                preparation_status['messages'].append(f"Failed to initialize base fields: {e}")
                return preparation_status
        
        # Find optimal seed position
        try:
            seed_position = self.find_optimal_seed_position()
            seed_region = self.get_region_at_position(seed_position)
            seed_subregion = self.get_subregion_at_position(seed_position)
            
            preparation_status['seed_position'] = seed_position
            preparation_status['seed_region'] = seed_region
            preparation_status['seed_subregion'] = seed_subregion
            preparation_status['messages'].append(f"Optimal seed position found at {seed_position} in {seed_region or 'unknown'} region.")
        except Exception as e:
            preparation_status['success'] = False
            preparation_status['messages'].append(f"Failed to find optimal seed position: {e}")
            return preparation_status
        
        # All checks passed
        preparation_status['metrics'] = self.get_metrics()
        preparation_status['messages'].append("Brain structure fully prepared for soul distribution.")
        
        return preparation_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive structure metrics."""
        metrics = {
            'brain_id': self.brain_id,
            'dimensions': self.dimensions,
            'creation_time': self.creation_time,
            'last_updated': self.last_updated,
            'structure': {
                'regions_count': len(self.regions),
                'sub_regions_count': len(self.sub_regions),
                'blocks_count': len(self.blocks),
                'boundaries_count': len(self.boundaries),
                'regions_defined': self.regions_defined,
                'sub_regions_defined': self.sub_regions_defined,
                'boundaries_defined': self.boundaries_defined,
                'field_initialized': self.field_initialized
            },
            'cells': {
                'total_grid_cells': self.total_grid_cells,
                'region_cells': len(self.region_grid),
                'subregion_cells': len(self.sub_region_grid),
                'energy_cells': len(self.energy_grid),
                'frequency_cells': len(self.frequency_grid),
                'resonance_cells': len(self.resonance_grid),
                'boundary_cell_count': self.boundary_cell_count,
                'soul_filled_cells': self.soul_filled_cells,
                'utilization_percentage': len(self.region_grid) / self.total_grid_cells * 100
            },
            'regions': {
                region_name: {
                    'cells': len([pos for pos, r in self.region_grid.items() if r == region_name]),
                    'proportion': region_info['proportion']
                } for region_name, region_info in self.regions.items()
            }
        }
        
        return metrics

    def save_state(self, filename: str) -> bool:
        """
        Save brain structure state to file.
        
        Args:
            filename: Path to output file
            
        Returns:
            True if saved successfully
        """
        logger.info(f"Saving brain structure state to {filename}")
        
        try:
            # Create dictionary with essential data
            # Use string keys for coordinates to make JSON serializable
            state = {
                'brain_id': self.brain_id,
                'dimensions': self.dimensions,
                'creation_time': self.creation_time,
                'last_updated': datetime.now().isoformat(),
                'regions': self.regions,
                'sub_regions': self.sub_regions,
                'blocks': self.blocks,
                'boundaries': self.boundaries,
                'region_grid': {str(k): v for k, v in self.region_grid.items()},
                'sub_region_grid': {str(k): v for k, v in self.sub_region_grid.items()},
                'energy_grid': {str(k): v for k, v in self.energy_grid.items()},
                'frequency_grid': {str(k): v for k, v in self.frequency_grid.items()},
                'resonance_grid': {str(k): v for k, v in self.resonance_grid.items()},
                'stability_grid': {str(k): v for k, v in self.stability_grid.items()},
                'coherence_grid': {str(k): v for k, v in self.coherence_grid.items()},
                'mycelial_density_grid': {str(k): v for k, v in self.mycelial_density_grid.items()},
                'mycelial_energy_grid': {str(k): v for k, v in self.mycelial_energy_grid.items()},
                'soul_presence_grid': {str(k): v for k, v in self.soul_presence_grid.items()},
                'status': {
                    'regions_defined': self.regions_defined,
                    'sub_regions_defined': self.sub_regions_defined,
                    'boundaries_defined': self.boundaries_defined,
                    'field_initialized': self.field_initialized,
                    'mycelial_paths_formed': self.mycelial_paths_formed
                },
                'statistics': {
                    'total_grid_cells': self.total_grid_cells,
                    'boundary_cell_count': self.boundary_cell_count,
                    'soul_filled_cells': self.soul_filled_cells
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Brain structure state saved successfully to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving brain structure state: {e}", exc_info=True)
            return False


    @classmethod
    def load_state(cls, filename: str) -> 'BrainGrid':
        """
        Load brain structure state from file.
        
        Args:
            filename: Path to state file
            
        Returns:
            BrainGrid instance
        """
        logger.info(f"Loading brain structure state from {filename}")
        
        try:
            # Load from file
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Create instance with saved dimensions
            dimensions = tuple(state['dimensions'])
            brain = BrainGrid(dimensions)
            
            # Restore simple properties
            brain.brain_id = state['brain_id']
            brain.creation_time = state['creation_time']
            brain.last_updated = state['last_updated']
            
            # Restore dictionaries
            brain.regions = state['regions']
            brain.sub_regions = state['sub_regions']
            brain.blocks = state['blocks']
            brain.boundaries = state['boundaries']
            
            # Helper for string tuple conversion
            def str_to_tuple(s):
                # Remove parentheses and split by comma
                parts = s.strip('()').split(',')
                return tuple(int(p.strip()) for p in parts)
            
            # Restore coordinate grids
            brain.region_grid = {str_to_tuple(k): v for k, v in state['region_grid'].items()}
            brain.sub_region_grid = {str_to_tuple(k): v for k, v in state['sub_region_grid'].items()}
            brain.energy_grid = {str_to_tuple(k): v for k, v in state['energy_grid'].items()}
            brain.frequency_grid = {str_to_tuple(k): v for k, v in state['frequency_grid'].items()}
            brain.resonance_grid = {str_to_tuple(k): v for k, v in state['resonance_grid'].items()}
            brain.stability_grid = {str_to_tuple(k): v for k, v in state['stability_grid'].items()}
            brain.coherence_grid = {str_to_tuple(k): v for k, v in state['coherence_grid'].items()}
            brain.mycelial_density_grid = {str_to_tuple(k): v for k, v in state['mycelial_density_grid'].items()}
            brain.mycelial_energy_grid = {str_to_tuple(k): v for k, v in state['mycelial_energy_grid'].items()}
            brain.soul_presence_grid = {str_to_tuple(k): v for k, v in state['soul_presence_grid'].items()}
            
            # Restore status flags
            brain.regions_defined = state['status']['regions_defined']
            brain.sub_regions_defined = state['status']['sub_regions_defined']
            brain.boundaries_defined = state['status']['boundaries_defined']
            brain.field_initialized = state['status']['field_initialized']
            brain.mycelial_paths_formed = state['status']['mycelial_paths_formed']
            
            # Restore statistics
            brain.total_grid_cells = state['statistics']['total_grid_cells']
            brain.boundary_cell_count = state['statistics']['boundary_cell_count']
            brain.soul_filled_cells = state['statistics']['soul_filled_cells']
            
            logger.info(f"Brain structure state loaded successfully from {filename}")
            return brain
            
        except Exception as e:
            logger.error(f"Error loading brain structure state: {e}", exc_info=True)
            raise ValueError(f"Failed to load brain structure: {e}")


# --- Create Brain Structure Function ---
def create_brain_structure(dimensions: Optional[Tuple[int, int, int]] = None,
                    initialize_regions: bool = True,
                    initialize_sound: bool = False) -> BrainGrid:
    """
    Create a brain structure with default settings.
    
    Args:
        dimensions: Optional custom dimensions (defaults to GRID_DIMENSIONS from constants)
        initialize_regions: Whether to initialize regions
        initialize_sound: Whether to initialize sound patterns
        
    Returns:
        Initialized BrainGrid instance
    """
    logger.info("Creating brain structure")
    
    # Use default dimensions if none provided
    if dimensions is None:
        dimensions = GRID_DIMENSIONS
    
    # Create brain structure
    brain = BrainGrid(dimensions)
    
    # Initialize regions if requested
    if initialize_regions:
        logger.info("Initializing brain regions")
        brain.define_major_regions()
        brain.define_sub_regions()
        brain.define_boundaries()
        brain.initialize_base_fields()
    
    # Initialize sound patterns if requested
    if initialize_sound:
        logger.info("Initializing sound patterns")
        _initialize_sound_patterns(brain)
    
    logger.info(f"Brain structure created with dimensions {dimensions}")
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'brain_id': brain.brain_id,
                'timestamp': datetime.now().isoformat(),
                'dimensions': dimensions,
                'regions_initialized': initialize_regions,
                'sound_initialized': initialize_sound,
                'total_grid_cells': brain.total_grid_cells
            }
            metrics_tracking.record_metrics("brain_structure_creation", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record brain creation metrics: {e}")
    
    return brain

def _initialize_sound_patterns(brain: BrainGrid) -> bool:
    """
    Initialize sound patterns in boundary cells.
    These are used for wave transmission between regions.
    
    Args:
        brain: BrainGrid instance
        
    Returns:
        True if successful
    """
    logger.info("Initializing sound patterns in boundary cells")
    
    # Ensure boundaries are defined
    if not brain.boundaries_defined:
        logger.warning("Boundaries not defined. Creating boundaries first.")
        brain.define_boundaries()
    
    # Set sound patterns in boundary cells
    patterns_set = 0
    
    # Identify boundary cells (high resonance cells)
    boundary_cells = [pos for pos, res in brain.resonance_grid.items() if res > 0.7]
    
    for pos in boundary_cells:
        # Get region for this cell
        region = brain.region_grid.get(pos)
        if not region:
            continue
        
        # Find which boundary this cell belongs to
        # We look at neighboring cells to see if they have different regions
        x, y, z = pos
        neighbor_regions = set()
        
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            neighbor_pos = (nx, ny, nz)
            
            # Skip if outside grid
            if not (0 <= nx < brain.dimensions[0] and
                    0 <= ny < brain.dimensions[1] and
                    0 <= nz < brain.dimensions[2]):
                continue
            
            # Get neighbor region
            neighbor_region = brain.region_grid.get(neighbor_pos)
            if neighbor_region and neighbor_region != region:
                neighbor_regions.add(neighbor_region)
        
        # Skip if no neighboring regions
        if not neighbor_regions:
            continue
        
        # Find the boundary type
        boundary_type = None
        for neighbor_region in neighbor_regions:
            regions_pair = tuple(sorted([region, neighbor_region]))
            
            # Check boundaries
            for boundary_id, boundary in brain.boundaries.items():
                if tuple(boundary['regions']) == regions_pair:
                    boundary_type = boundary['type']
                    break
            
            if boundary_type:
                break
        
        # Skip if no boundary type found
        if not boundary_type:
            continue
        
        # Adjust frequency based on boundary type
        current_freq = brain.frequency_grid.get(pos, 0.0)
        
        if boundary_type == BOUNDARY_TYPE_SHARP:
            # Sharp boundaries have higher frequencies
            freq_adjustment = 1.2  # 20% increase
        elif boundary_type == BOUNDARY_TYPE_GRADUAL:
            # Gradual boundaries have moderate adjustment
            freq_adjustment = 1.0  # No change
        else:  # BOUNDARY_TYPE_PERMEABLE
            # Permeable boundaries have lower frequencies
            freq_adjustment = 0.9  # 10% decrease
        
        # Apply frequency adjustment
        brain.frequency_grid[pos] = current_freq * freq_adjustment
        
        patterns_set += 1
    
    logger.info(f"Set sound patterns in {patterns_set} boundary cells")
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'brain_id': brain.brain_id,
                'timestamp': datetime.now().isoformat(),
                'patterns_set': patterns_set,
                'boundary_cells': len(boundary_cells)
            }
            metrics_tracking.record_metrics("brain_structure_sound_patterns", metrics_data)
        except Exception as e:
            logger.warning(f"Failed to record sound pattern metrics: {e}")
    
    return patterns_set > 0

# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Brain Structure module test execution")
    
    try:
        # Create brain structure
        brain = create_brain_structure()
        
        # Print metrics
        metrics = brain.get_metrics()
        print(f"Brain structure created with ID {brain.brain_id}")
        print(f"Dimensions: {metrics['dimensions']}")
        print(f"Total grid cells: {metrics['cells']['total_grid_cells']}")
        print(f"Region cells: {metrics['cells']['region_cells']}")
        print(f"Utilization: {metrics['cells']['utilization_percentage']:.2f}%")
        
        # Find optimal seed position
        seed_pos = brain.find_optimal_seed_position()
        print(f"Optimal seed position: {seed_pos}")
        print(f"Region at seed position: {brain.get_region_at_position(seed_pos)}")
        
        # Save state
        brain.save_state("test_brain_structure.json")
        print("Brain structure state saved to test_brain_structure.json")
        
    except Exception as e:
        logger.error(f"Error during test execution: {e}", exc_info=True)
        print(f"ERROR: Test execution failed: {e}")
        sys.exit(1)
    
    logger.info("Brain Structure module test execution completed successfully")

# --- END OF FILE stage_1/evolve/brain_structure/brain_structure.py ---