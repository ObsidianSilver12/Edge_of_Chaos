# brain_structure.py V9 - FIXED COMPLETE SYSTEM
"""
Anatomically Correct Brain Structure with proper matrices, frequency tracking, and mirror grid.
Creates regions → sub-regions → blocks → nodes/seeds with proper ID structure and frequency management.
Includes mirror grid system for fragment storage with entanglement capabilities.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import uuid
import random
import numpy as np

# Import constants and definitions
from shared.constants.constants import *
from shared.dictionaries.region_definitions import MAJOR_REGIONS, SUB_REGIONS
from stage_1.brain_formation.energy_storage import create_energy_storage_with_brain

# --- Logging Setup ---
logger = logging.getLogger("BrainStructure")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AnatomicalBrain:
    """
    Anatomically correct brain structure with proper node/seed placement and frequency management.
    
    Architecture:
    - Grid → Regions → Sub-regions → Blocks
    - Blocks contain either nodes OR mycelial seeds
    - Proper frequency tracking for search capabilities
    - Mirror grid for memory fragments with entanglement
    - No artificial hemisphere division
    """
    
    def __init__(self, dimensions: Tuple[int, int, int] = GRID_DIMENSIONS, brain_seed: Dict[str, Any] = None):
        """Initialize anatomical brain structure."""
        self.brain_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.grid_dimensions = dimensions
        self.brain_seed = brain_seed
        
        # --- Core Structure Storage ---
        self.regions = {}                    # Region definitions and data
        self.sub_regions = {}               # Sub-region data with matrices
        self.static_borders = {}            # Border definitions between regions
        
        # --- Node and Seed Management ---
        self.nodes = {}                     # All nodes indexed by ID
        self.mycelial_seeds = {}           # All seeds indexed by ID  
        self.node_status_frequencies = {   # Frequency tracking for search
            'active': NODE_FREQUENCY_ACTIVE,
            'inactive': NODE_FREQUENCY_INACTIVE, 
            'archived': NODE_FREQUENCY_ARCHIVED
        }
        self.seed_status_frequencies = {
            'active': MYCELIAL_SEED_FREQUENCY_ACTIVE,
            'inactive': MYCELIAL_SEED_FREQUENCY_INACTIVE,
            'dormant': MYCELIAL_SEED_FREQUENCY_DORMANT
        }
        
        # --- Grid Matrices (3D NumPy Arrays) ---
        self.whole_brain_matrix = np.full(dimensions, None, dtype=object)  # Complete brain overview
        self.node_placement_matrix = np.zeros(dimensions, dtype=object)    # Node tracking matrix
        self.seed_placement_matrix = np.zeros(dimensions, dtype=object)    # Seed tracking matrix
        self.sub_region_matrices = {}                                      # Per sub-region matrices
        
        # --- Mirror Grid System (for memory fragments) ---
        self.mirror_grid_enabled = False
        self.mirror_grid_matrix = None     # Created on demand
        self.mirror_grid_entanglement = {} # Entanglement mapping brain ↔ mirror
        
        # --- Energy and Processing ---
        self.energy_storage = None
        self.seed_allocation_data = {}
        
        # --- Configuration ---
        self.density_variance = self._calculate_density_variance()
        self.region_volumes = {}
        
        logger.info(f"🧠 Anatomical brain initialized: {self.brain_id[:8]}")
    
    def _calculate_density_variance(self) -> Dict[str, float]:
        """Calculate density variance for each region (biomimetic individual differences)."""
        variance = {}
        all_regions = list(MAJOR_REGIONS.keys())
        dense_regions = random.sample(all_regions, random.randint(2, 3))
        
        for region_name in all_regions:
            if region_name in dense_regions:
                variance[region_name] = random.uniform(0.85, 0.95)  # More dense
            else:
                variance[region_name] = random.uniform(0.70, 0.80)  # Standard density
        
        logger.info(f"Density variance calculated: dense regions = {dense_regions}")
        return variance
    
    def load_brain_seed(self):
        """Load brain seed coordinates and validate position."""
        logger.info("Loading brain seed coordinates...")
        
        try:
            if not self.brain_seed:
                # Default center position with variance
                center = (
                    self.grid_dimensions[0] // 2 + random.randint(-10, 10),
                    self.grid_dimensions[1] // 2 + random.randint(-10, 10), 
                    self.grid_dimensions[2] // 2 + random.randint(-10, 10)
                )
                self.brain_seed = {
                    'position': center, 
                    'energy': BASE_BRAIN_FREQUENCY,  # 7.83 Hz
                    'frequency': BASE_BRAIN_FREQUENCY
                }
                logger.warning("Created default brain seed")
            
            # Validate position within brain area (not external buffer)
            pos = self.brain_seed['position']
            buffer_size = 26
            
            if (pos[0] < buffer_size or pos[0] >= self.grid_dimensions[0] - buffer_size or
                pos[1] < buffer_size or pos[1] >= self.grid_dimensions[1] - buffer_size or
                pos[2] < buffer_size or pos[2] >= self.grid_dimensions[2] - buffer_size):
                
                center_pos = (
                    self.grid_dimensions[0] // 2,
                    self.grid_dimensions[1] // 2, 
                    self.grid_dimensions[2] // 2
                )
                self.brain_seed['position'] = center_pos
                logger.warning(f"Brain seed repositioned to center: {center_pos}")
            
            logger.info(f"✅ Brain seed loaded at: {self.brain_seed['position']}")
            
        except Exception as e:
            logger.error(f"Failed to load brain seed: {e}")
            raise RuntimeError(f"Brain seed loading failed: {e}") from e
    
    def calculate_anatomical_volumes(self):
        """Calculate volumes for each brain region based on anatomical proportions."""
        logger.info("Calculating anatomical region volumes...")
        
        try:
            buffer_size = 26
            region_border_size = 4
            
            # Available brain space
            total_x = self.grid_dimensions[0] - (2 * buffer_size) - (2 * region_border_size)
            total_y = self.grid_dimensions[1] - (2 * buffer_size) - (2 * region_border_size)
            total_z = self.grid_dimensions[2] - (2 * buffer_size) - (2 * region_border_size)
            total_brain_volume = total_x * total_y * total_z
            
            logger.info(f"Total brain volume: {total_brain_volume:,} units")
            
            # Calculate region volumes with density variance
            self.region_volumes = {}
            for region_name, region_config in MAJOR_REGIONS.items():
                base_volume = int(total_brain_volume * region_config['proportion'])
                density_factor = self.density_variance.get(region_name, 0.80)
                usable_volume = int(base_volume * density_factor)
                
                # Account for border space
                border_volume = region_border_size ** 3 * 6  # Approximate border volume
                actual_usable_volume = max(1000, usable_volume - border_volume)  # Minimum volume
                
                self.region_volumes[region_name] = {
                    'total_volume': base_volume,
                    'usable_volume': actual_usable_volume,
                    'border_volume': border_volume,
                    'density_factor': density_factor,
                    'region_bounds': self._calculate_region_bounds(
                        region_name, total_x, total_y, total_z, buffer_size, region_border_size
                    )
                }
            
            logger.info(f"✅ Calculated volumes for {len(self.region_volumes)} regions")
            
        except Exception as e:
            logger.error(f"Failed to calculate volumes: {e}")
            raise RuntimeError(f"Volume calculation failed: {e}") from e
    
    def _calculate_region_bounds(self, region_name: str, brain_x: int, brain_y: int, 
                                brain_z: int, buffer_size: int, region_border_size: int) -> Dict[str, int]:
        """Calculate anatomical bounds for a specific region."""
        try:
            region_config = MAJOR_REGIONS[region_name]
            location_bias = region_config['location_bias']
            
            # Calculate region size based on proportion
            region_volume = self.region_volumes.get(region_name, {}).get('usable_volume', brain_x * brain_y * brain_z * 0.1)
            estimated_side = int((region_volume ** (1/3)) * 1.2)  # Cube root with adjustment
            
            # Apply location bias to position region anatomically
            start_x = buffer_size + region_border_size + int((brain_x - estimated_side) * location_bias[0])
            start_y = buffer_size + region_border_size + int((brain_y - estimated_side) * location_bias[1])
            start_z = buffer_size + region_border_size + int((brain_z - estimated_side) * location_bias[2])
            
            # Ensure bounds stay within brain area
            end_x = min(self.grid_dimensions[0] - buffer_size - region_border_size, start_x + estimated_side)
            end_y = min(self.grid_dimensions[1] - buffer_size - region_border_size, start_y + estimated_side)
            end_z = min(self.grid_dimensions[2] - buffer_size - region_border_size, start_z + estimated_side)
            
            return {
                'x_start': start_x, 'x_end': end_x,
                'y_start': start_y, 'y_end': end_y,
                'z_start': start_z, 'z_end': end_z
            }
            
        except KeyError as e:
            logger.error(f"Region {region_name} not found in MAJOR_REGIONS: {e}")
            raise ValueError(f"Invalid region name: {region_name}") from e
        except Exception as e:
            logger.error(f"Failed to calculate bounds for {region_name}: {e}")
            raise RuntimeError(f"Bounds calculation failed for {region_name}: {e}") from e
    
    def create_regions_and_sub_regions(self):
        """Create all brain regions and sub-regions with proper anatomical placement."""
        logger.info("Creating brain regions and sub-regions...")
        
        try:
            total_regions_created = 0
            total_sub_regions_created = 0
            
            # Create each major region
            for region_name, region_config in MAJOR_REGIONS.items():
                if region_name not in self.region_volumes:
                    logger.error(f"Volume not calculated for region: {region_name}")
                    raise RuntimeError(f"Missing volume data for region: {region_name}")
                
                region_bounds = self.region_volumes[region_name]['region_bounds']
                
                # Create region data structure
                region_data = {
                    'region_id': region_name,
                    'region_type': 'major_region',
                    'function': region_config['function'],
                    'boundaries': region_bounds,
                    'default_wave': region_config['default_wave'],
                    'wave_frequency_hz': region_config['wave_frequency_hz'],
                    'color': region_config['color'].value,
                    'sub_regions': {},
                    'creation_time': datetime.now().isoformat(),
                    'volume_info': self.region_volumes[region_name]
                }
                
                # Create sub-regions for this region
                sub_region_names = region_config['sub_regions']
                region_sub_regions_created = 0
                
                for sub_region_name in sub_region_names:
                    if sub_region_name in SUB_REGIONS:
                        sub_region_config = SUB_REGIONS[sub_region_name]
                        
                        # Calculate sub-region bounds within parent region
                        sub_region_bounds = self._calculate_sub_region_bounds(
                            region_bounds, sub_region_config, len(sub_region_names), 
                            region_sub_regions_created
                        )
                        
                        # Create sub-region data
                        sub_region_data = {
                            'sub_region_id': sub_region_name,
                            'parent_region': region_name,
                            'function': sub_region_config['function'],
                            'boundaries': sub_region_bounds,
                            'wave_frequency_hz': sub_region_config['wave_frequency_hz'],
                            'blocks': {},
                            'nodes': {},      # Will track nodes in this sub-region
                            'seeds': {},      # Will track seeds in this sub-region
                            'matrix': None,   # Will be 3D numpy array for this sub-region
                            'creation_time': datetime.now().isoformat()
                        }
                        
                        # Store sub-region data
                        region_data['sub_regions'][sub_region_name] = sub_region_data
                        self.sub_regions[f"{region_name}-{sub_region_name}"] = sub_region_data
                        
                        region_sub_regions_created += 1
                        total_sub_regions_created += 1
                    else:
                        logger.warning(f"Sub-region {sub_region_name} not found in SUB_REGIONS")
                
                # Store region data
                self.regions[region_name] = region_data
                total_regions_created += 1
                
                logger.info(f"Region {region_name}: {region_sub_regions_created} sub-regions created")
            
            logger.info(f"✅ Created {total_regions_created} regions, {total_sub_regions_created} sub-regions")
            
        except Exception as e:
            logger.error(f"Failed to create regions and sub-regions: {e}")
            raise RuntimeError(f"Region creation failed: {e}") from e
    
    def _calculate_sub_region_bounds(self, parent_bounds: Dict[str, int], sub_region_config: Dict[str, Any],
                                   total_sub_regions: int, sub_region_index: int) -> Dict[str, int]:
        """Calculate bounds for a sub-region within its parent region."""
        try:
            # Calculate available space in parent region
            parent_width = parent_bounds['x_end'] - parent_bounds['x_start']
            parent_height = parent_bounds['y_end'] - parent_bounds['y_start']
            parent_depth = parent_bounds['z_end'] - parent_bounds['z_start']
            
            # Determine sub-region size based on proportion
            proportion = sub_region_config.get('proportion', 1.0 / total_sub_regions)
            
            # Calculate dimensions
            sub_width = max(10, int(parent_width * proportion ** (1/3)))
            sub_height = max(10, int(parent_height * proportion ** (1/3)))
            sub_depth = max(10, int(parent_depth * proportion ** (1/3)))
            
            # Position sub-region (simple grid placement for now)
            grid_x = sub_region_index % 2
            grid_y = (sub_region_index // 2) % 2
            grid_z = (sub_region_index // 4) % 2
            
            start_x = parent_bounds['x_start'] + grid_x * (parent_width // 2)
            start_y = parent_bounds['y_start'] + grid_y * (parent_height // 2)
            start_z = parent_bounds['z_start'] + grid_z * (parent_depth // 2)
            
            # Ensure sub-region fits within parent
            end_x = min(parent_bounds['x_end'], start_x + sub_width)
            end_y = min(parent_bounds['y_end'], start_y + sub_height)
            end_z = min(parent_bounds['z_end'], start_z + sub_depth)
            
            return {
                'x_start': start_x, 'x_end': end_x,
                'y_start': start_y, 'y_end': end_y,
                'z_start': start_z, 'z_end': end_z
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate sub-region bounds: {e}")
            raise RuntimeError(f"Sub-region bounds calculation failed: {e}") from e
    
    def populate_blocks_nodes_and_seeds(self):
        """Populate each sub-region with blocks, then designate blocks as nodes or seeds."""
        logger.info("Populating blocks and designating nodes/seeds...")
        
        try:
            block_size = 8  # Size of each block
            total_blocks = 0
            total_nodes = 0
            total_seeds = 0
            
            for region_name, region_data in self.regions.items():
                region_blocks = 0
                
                for sub_region_name, sub_region_data in region_data['sub_regions'].items():
                    bounds = sub_region_data['boundaries']
                    
                    # Calculate how many blocks fit in this sub-region
                    width = bounds['x_end'] - bounds['x_start']
                    height = bounds['y_end'] - bounds['y_start']
                    depth = bounds['z_end'] - bounds['z_start']
                    
                    blocks_x = max(1, width // block_size)
                    blocks_y = max(1, height // block_size)
                    blocks_z = max(1, depth // block_size)
                    max_blocks = blocks_x * blocks_y * blocks_z
                    
                    # Apply density to determine actual blocks
                    density_factor = self.density_variance.get(region_name, 0.80)
                    actual_blocks = max(1, int(max_blocks * density_factor))
                    
                    # Calculate seed allocation (5-10% of blocks become seeds)
                    seed_percentage = self._calculate_seed_percentage(actual_blocks)
                    seeds_needed = max(1, int(actual_blocks * seed_percentage))
                    nodes_needed = actual_blocks - seeds_needed
                    
                    # Create sub-region matrix
                    matrix_dims = (blocks_x, blocks_y, blocks_z)
                    sub_region_matrix = np.zeros(matrix_dims, dtype=object)
                    
                    # Place blocks randomly and designate as nodes or seeds
                    blocks_placed = 0
                    nodes_created = 0
                    seeds_created = 0
                    occupied_positions = set()
                    
                    while blocks_placed < actual_blocks:
                        # Random position within sub-region grid
                        bx = random.randint(0, blocks_x - 1)
                        by = random.randint(0, blocks_y - 1)
                        bz = random.randint(0, blocks_z - 1)
                        
                        if (bx, by, bz) not in occupied_positions:
                            # Calculate absolute coordinates
                            abs_x = bounds['x_start'] + bx * block_size + block_size // 2
                            abs_y = bounds['y_start'] + by * block_size + block_size // 2
                            abs_z = bounds['z_start'] + bz * block_size + block_size // 2
                            
                            # Determine if this block becomes a node or seed
                            if seeds_created < seeds_needed:
                                # Create mycelial seed
                                seed_id = f"{region_name}-{sub_region_name}-seed-{seeds_created+1:03d}"
                                seed_data = self._create_mycelial_seed(seed_id, (abs_x, abs_y, abs_z), 
                                                                    region_name, sub_region_name)
                                
                                # Store seed
                                self.mycelial_seeds[seed_id] = seed_data
                                sub_region_data['seeds'][seed_id] = seed_data
                                sub_region_matrix[bx, by, bz] = seed_data
                                self.seed_placement_matrix[abs_x, abs_y, abs_z] = seed_data
                                self.whole_brain_matrix[abs_x, abs_y, abs_z] = seed_data
                                
                                seeds_created += 1
                                total_seeds += 1
                                
                            else:
                                # Create node
                                node_id = f"{region_name}-{sub_region_name}-node-{nodes_created+1:03d}"
                                node_data = self._create_brain_node(node_id, (abs_x, abs_y, abs_z), 
                                                                  region_name, sub_region_name)
                                
                                # Store node
                                self.nodes[node_id] = node_data
                                sub_region_data['nodes'][node_id] = node_data
                                sub_region_matrix[bx, by, bz] = node_data
                                self.node_placement_matrix[abs_x, abs_y, abs_z] = node_data
                                self.whole_brain_matrix[abs_x, abs_y, abs_z] = node_data
                                
                                nodes_created += 1
                                total_nodes += 1
                            
                            occupied_positions.add((bx, by, bz))
                            blocks_placed += 1
                            region_blocks += 1
                            total_blocks += 1
                    
                    # Store sub-region matrix
                    sub_region_data['matrix'] = sub_region_matrix
                    sub_region_data['blocks_created'] = blocks_placed
                    sub_region_data['nodes_created'] = nodes_created
                    sub_region_data['seeds_created'] = seeds_created
                    
                    # Store matrix in organized collection
                    self.sub_region_matrices[f"{region_name}-{sub_region_name}"] = sub_region_matrix
                
                logger.info(f"Region {region_name}: {region_blocks} blocks created")
            
            logger.info(f"✅ Created {total_blocks} blocks: {total_nodes} nodes, {total_seeds} seeds")
            
        except Exception as e:
            logger.error(f"Failed to populate blocks: {e}")
            raise RuntimeError(f"Block population failed: {e}") from e
    
    def _calculate_seed_percentage(self, total_blocks: int) -> float:
        """
        Calculate percentage of blocks that become mycelial seeds based on region size.
        
        Uses size-based density scaling from constants:
        - ≤5 blocks: 5% seeds (very small regions)
        - ≤20 blocks: 6% seeds (small regions)  
        - ≤50 blocks: 7% seeds (medium regions)
        - ≤100 blocks: 8% seeds (large regions)
        - >100 blocks: 10% seeds (very large regions)
        
        This ensures appropriate seed density relative to region complexity.
        """
        # Use size-based density calculations from constants
        if total_blocks <= 5:
            return MYCELIAL_SEED_DENSITY_VERY_SMALL  # 5% for very small regions
        elif total_blocks <= 20:
            return MYCELIAL_SEED_DENSITY_SMALL       # 6% for small regions
        elif total_blocks <= 50:
            return MYCELIAL_SEED_DENSITY_MEDIUM      # 7% for medium regions
        elif total_blocks <= 100:
            return MYCELIAL_SEED_DENSITY_LARGE       # 8% for large regions
        else:
            return MYCELIAL_SEED_DENSITY_VERY_LARGE  # 10% for very large regions
    
    def _create_brain_node(self, node_id: str, coordinates: Tuple[int, int, int], 
                          region: str, sub_region: str) -> Dict[str, Any]:
        """Create a brain node with proper ID structure and frequency tracking."""
        try:
            # Get sub-region frequency
            sub_region_config = SUB_REGIONS.get(sub_region, {})
            sub_region_frequency = sub_region_config.get('wave_frequency_hz', BASE_BRAIN_FREQUENCY)
            
            node_data = {
                'node_id': node_id,
                'node_type': 'neural',
                'coordinates': coordinates,
                'region': region,
                'sub_region': sub_region,
                'hierarchical_name': node_id,
                
                # Frequency management
                'sub_region_frequency': sub_region_frequency,  # Frequency from sub-region
                'status_frequency': self.node_status_frequencies['inactive'],  # Current status frequency
                'status': 'inactive',  # Start inactive, activated later
                
                # State tracking
                'active': False,
                'last_activation': None,
                'activation_count': 0,
                
                # Processing capabilities
                'processing_capacity': 1.0,
                'energy_level': 0.0,
                'connections': [],
                
                # Creation metadata
                'creation_time': datetime.now().isoformat(),
                'node_signature': f"{region}_{sub_region}_{coordinates[0]}_{coordinates[1]}_{coordinates[2]}"
            }
            
            return node_data
            
        except Exception as e:
            logger.error(f"Failed to create node {node_id}: {e}")
            raise RuntimeError(f"Node creation failed: {node_id}") from e
    
    def _create_mycelial_seed(self, seed_id: str, coordinates: Tuple[int, int, int], 
                             region: str, sub_region: str) -> Dict[str, Any]:
        """Create a mycelial seed with proper ID structure and specialized capabilities."""
        try:
            seed_data = {
                'seed_id': seed_id,
                'seed_type': 'mycelial',
                'coordinates': coordinates,
                'region': region,
                'sub_region': sub_region,
                'hierarchical_name': seed_id,
                
                # Frequency management
                'seed_frequency': MYCELIAL_SEED_FREQUENCY_BASE,  # Shared seed frequency
                'status_frequency': self.seed_status_frequencies['dormant'],  # Current status frequency
                'status': 'dormant',  # Start dormant, activated on demand
                
                # State tracking
                'active': False,
                'last_activation': None,
                'activation_count': 0,
                
                # Mycelial capabilities
                'communication_channels': [],
                'field_modulation_active': False,
                'energy_distribution_capacity': MYCELIAL_SEED_BASE_ENERGY_MULTIPLIER,
                'quantum_entangled': False,
                
                # Energy management
                'energy_level': 0.0,
                'energy_storage_capacity': SYNAPSE_ENERGY_JOULES * MYCELIAL_SEED_BASE_ENERGY_MULTIPLIER,
                
                # Creation metadata
                'creation_time': datetime.now().isoformat(),
                'seed_signature': f"seed_{region}_{sub_region}_{coordinates[0]}_{coordinates[1]}_{coordinates[2]}"
            }
            
            return seed_data
            
        except Exception as e:
            logger.error(f"Failed to create seed {seed_id}: {e}")
            raise RuntimeError(f"Seed creation failed: {seed_id}") from e
    
    def create_mirror_grid(self, nodes_disabled: bool = True) -> Dict[str, Any]:
        """
        Create mirror grid that mirrors brain structure for memory fragments.
        The mirror grid has same structure but nodes disabled to allow fragment storage.
        """
        logger.info(f"🪞 Creating mirror grid (nodes disabled: {nodes_disabled})...")
        
        try:
            if self.mirror_grid_enabled:
                logger.warning("Mirror grid already exists")
                return {'success': True, 'reason': 'already_exists'}
            
            # Create mirror grid matrix (same dimensions as brain)
            self.mirror_grid_matrix = np.full(self.grid_dimensions, None, dtype=object)
            
            # Copy structure but modify for fragment storage
            mirror_regions = {}
            mirror_sub_regions = {}
            total_fragment_spaces = 0
            
            for region_name, region_data in self.regions.items():
                # Copy region structure
                mirror_region = region_data.copy()
                mirror_region['mirror_grid'] = True
                mirror_region['nodes_disabled'] = nodes_disabled
                mirror_region['fragment_storage_active'] = True
                
                mirror_region_sub_regions = {}
                
                for sub_region_name, sub_region_data in region_data['sub_regions'].items():
                    # Copy sub-region structure
                    mirror_sub_region = sub_region_data.copy()
                    mirror_sub_region['mirror_grid'] = True
                    mirror_sub_region['nodes_disabled'] = nodes_disabled
                    mirror_sub_region['fragment_storage'] = {}  # For storing memory fragments
                    
                    # Create fragment storage spaces (only for subset of nodes for lower density)
                    if nodes_disabled:
                        node_list = list(sub_region_data.get('nodes', {}).items())
                        # Only create fragment spaces for 20% of nodes to achieve lower density
                        fragment_density = 0.20
                        num_fragment_spaces = max(1, int(len(node_list) * fragment_density))
                        
                        # Randomly select which nodes get fragment spaces
                        import random
                        selected_nodes = random.sample(node_list, min(num_fragment_spaces, len(node_list)))
                        
                        for node_id, node_data in selected_nodes:
                            # Create fragment storage space at node location
                            fragment_space_id = node_id.replace('node', 'fragment_space')
                            fragment_space = {
                                'fragment_space_id': fragment_space_id,
                                'coordinates': node_data['coordinates'],
                                'region': node_data['region'],
                                'sub_region': node_data['sub_region'],
                                'capacity': 10,  # Can hold up to 10 memory fragments
                                'fragments': {},
                                'entanglement_link': node_id,  # Link to corresponding brain node
                                'creation_time': datetime.now().isoformat()
                            }
                            
                            mirror_sub_region['fragment_storage'][fragment_space_id] = fragment_space
                            
                            # Place in mirror grid matrix
                            coords = fragment_space['coordinates']
                            self.mirror_grid_matrix[coords[0], coords[1], coords[2]] = fragment_space
                            
                            # Create entanglement mapping
                            self.mirror_grid_entanglement[node_id] = fragment_space_id
                            
                            total_fragment_spaces += 1
                    
                    # Keep seeds in mirror grid (they help with cosmic field effects)
                    mirror_sub_region['seeds'] = sub_region_data.get('seeds', {}).copy()
                    
                    mirror_region_sub_regions[sub_region_name] = mirror_sub_region
                    mirror_sub_regions[f"{region_name}-{sub_region_name}"] = mirror_sub_region
                
                mirror_region['sub_regions'] = mirror_region_sub_regions
                mirror_regions[region_name] = mirror_region
            
            # Store mirror grid data
            self.mirror_grid = {
                'enabled': True,
                'nodes_disabled': nodes_disabled,
                'regions': mirror_regions,
                'sub_regions': mirror_sub_regions,
                'matrix': self.mirror_grid_matrix,
                'entanglement_mapping': self.mirror_grid_entanglement,
                'fragment_spaces': total_fragment_spaces,
                'creation_time': datetime.now().isoformat()
            }
            
            self.mirror_grid_enabled = True
            
            logger.info(f"✅ Mirror grid created: {total_fragment_spaces} fragment spaces")
            
            return {
                'success': True,
                'fragment_spaces_created': total_fragment_spaces,
                'entanglement_links': len(self.mirror_grid_entanglement),
                'nodes_disabled': nodes_disabled
            }
            
        except Exception as e:
            logger.error(f"Failed to create mirror grid: {e}")
            raise RuntimeError(f"Mirror grid creation failed: {e}") from e
    
    def activate_initial_nodes_with_energy(self):
        """
        Create energy storage system and activate initial nodes.
        This prepares the brain for mycelial seed activation.
        """
        logger.info("🔋 Creating energy storage and activating initial nodes...")
        
        try:
            # Create energy storage system
            self.energy_storage = create_energy_storage_with_brain(self.get_brain_structure())
            
            # Collect nodes for activation
            nodes_to_activate = []
            seed_allocation_summary = {}
            
            for region_name, region_data in self.regions.items():
                for sub_region_name, sub_region_data in region_data['sub_regions'].items():
                    sub_region_key = f"{region_name}-{sub_region_name}"
                    
                    # Track allocation
                    nodes_count = len(sub_region_data.get('nodes', {}))
                    seeds_count = len(sub_region_data.get('seeds', {}))
                    total_blocks = nodes_count + seeds_count
                    
                    seed_allocation_summary[sub_region_key] = {
                        'total_blocks': total_blocks,
                        'seeds_allocated': seeds_count,
                        'nodes_allocated': nodes_count,
                        'seed_percentage': seeds_count / total_blocks if total_blocks > 0 else 0
                    }
                    
                    # Add nodes to activation list
                    for node_id, node_data in sub_region_data.get('nodes', {}).items():
                        nodes_to_activate.append({
                            'node_id': node_id,
                            'coordinates': node_data['coordinates'],
                            'region': node_data['region'],
                            'sub_region': node_data['sub_region']
                        })
            
            # Activate nodes using energy storage
            activation_results = self.energy_storage.activate_initial_nodes(nodes_to_activate)
            
            # Update node states in brain matrices
            for node_info in nodes_to_activate[:activation_results['successful_activations']]:
                node_id = node_info['node_id']
                coords = node_info['coordinates']
                
                # Update node status
                if node_id in self.nodes:
                    self.nodes[node_id]['active'] = True
                    self.nodes[node_id]['status'] = 'active'
                    self.nodes[node_id]['status_frequency'] = self.node_status_frequencies['active']
                    self.nodes[node_id]['last_activation'] = datetime.now().isoformat()
                    self.nodes[node_id]['activation_count'] += 1
                
                # Update matrices
                if (0 <= coords[0] < self.grid_dimensions[0] and 
                    0 <= coords[1] < self.grid_dimensions[1] and 
                    0 <= coords[2] < self.grid_dimensions[2]):
                    
                    if self.whole_brain_matrix[coords] is not None:
                        self.whole_brain_matrix[coords]['active'] = True
                    if self.node_placement_matrix[coords] is not None:
                        self.node_placement_matrix[coords]['active'] = True
            
            # Store activation results
            self.brain_structure = self.get_brain_structure()
            self.brain_structure['node_activation'] = {
                'activation_time': datetime.now().isoformat(),
                'total_nodes_activated': activation_results['successful_activations'],
                'total_energy_used': activation_results['total_energy_used'],
                'seed_allocation_summary': seed_allocation_summary
            }
            
            self.seed_allocation_data = seed_allocation_summary
            
            logger.info(f"✅ Activated {activation_results['successful_activations']} nodes")
            logger.info(f"   Energy used: {activation_results['total_energy_used']:.1f} SEU")
            
            return activation_results
            
        except Exception as e:
            logger.error(f"Failed to activate nodes with energy: {e}")
            raise RuntimeError(f"Node activation failed: {e}") from e
    
    def search_nodes_by_frequency(self, target_frequency: float, tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """Search for nodes by their status frequency (for pattern detection)."""
        try:
            matching_nodes = []
            
            for node_id, node_data in self.nodes.items():
                status_freq = node_data.get('status_frequency', 0.0)
                if abs(status_freq - target_frequency) <= tolerance:
                    matching_nodes.append({
                        'node_id': node_id,
                        'coordinates': node_data['coordinates'],
                        'status': node_data['status'],
                        'frequency': status_freq,
                        'region': node_data['region'],
                        'sub_region': node_data['sub_region']
                    })
            
            return matching_nodes
            
        except Exception as e:
            logger.error(f"Failed to search nodes by frequency: {e}")
            raise RuntimeError(f"Node frequency search failed: {e}") from e
    
    def search_seeds_by_frequency(self, target_frequency: float, tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """Search for seeds by their status frequency."""
        try:
            matching_seeds = []
            
            for seed_id, seed_data in self.mycelial_seeds.items():
                status_freq = seed_data.get('status_frequency', 0.0)
                if abs(status_freq - target_frequency) <= tolerance:
                    matching_seeds.append({
                        'seed_id': seed_id,
                        'coordinates': seed_data['coordinates'],
                        'status': seed_data['status'],
                        'frequency': status_freq,
                        'region': seed_data['region'],
                        'sub_region': seed_data['sub_region']
                    })
            
            return matching_seeds
            
        except Exception as e:
            logger.error(f"Failed to search seeds by frequency: {e}")
            raise RuntimeError(f"Seed frequency search failed: {e}") from e
    
    def update_node_status(self, node_id: str, new_status: str) -> bool:
        """Update node status and corresponding frequency for search capabilities."""
        try:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found")
            
            if new_status not in self.node_status_frequencies:
                raise ValueError(f"Invalid node status: {new_status}")
            
            # Update node data
            self.nodes[node_id]['status'] = new_status
            self.nodes[node_id]['status_frequency'] = self.node_status_frequencies[new_status]
            self.nodes[node_id]['active'] = (new_status == 'active')
            
            # Update in matrices
            coords = self.nodes[node_id]['coordinates']
            if (0 <= coords[0] < self.grid_dimensions[0] and 
                0 <= coords[1] < self.grid_dimensions[1] and 
                0 <= coords[2] < self.grid_dimensions[2]):
                
                if self.whole_brain_matrix[coords] is not None:
                    self.whole_brain_matrix[coords]['status'] = new_status
                    self.whole_brain_matrix[coords]['active'] = (new_status == 'active')
                
                if self.node_placement_matrix[coords] is not None:
                    self.node_placement_matrix[coords]['status'] = new_status
                    self.node_placement_matrix[coords]['active'] = (new_status == 'active')
            
            logger.info(f"Node {node_id[:8]} status updated: {new_status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update node status: {e}")
            raise RuntimeError(f"Node status update failed: {node_id}") from e
    
    def update_seed_status(self, seed_id: str, new_status: str) -> bool:
        """Update seed status and corresponding frequency."""
        try:
            if seed_id not in self.mycelial_seeds:
                raise ValueError(f"Seed {seed_id} not found")
            
            if new_status not in self.seed_status_frequencies:
                raise ValueError(f"Invalid seed status: {new_status}")
            
            # Update seed data
            self.mycelial_seeds[seed_id]['status'] = new_status
            self.mycelial_seeds[seed_id]['status_frequency'] = self.seed_status_frequencies[new_status]
            self.mycelial_seeds[seed_id]['active'] = (new_status == 'active')
            
            # Update in matrices
            coords = self.mycelial_seeds[seed_id]['coordinates']
            if (0 <= coords[0] < self.grid_dimensions[0] and 
                0 <= coords[1] < self.grid_dimensions[1] and 
                0 <= coords[2] < self.grid_dimensions[2]):
                
                if self.whole_brain_matrix[coords] is not None:
                    self.whole_brain_matrix[coords]['status'] = new_status
                    self.whole_brain_matrix[coords]['active'] = (new_status == 'active')
                
                if self.seed_placement_matrix[coords] is not None:
                    self.seed_placement_matrix[coords]['status'] = new_status
                    self.seed_placement_matrix[coords]['active'] = (new_status == 'active')
            
            logger.info(f"Seed {seed_id[:8]} status updated: {new_status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update seed status: {e}")
            raise RuntimeError(f"Seed status update failed: {seed_id}") from e
    
    def get_sub_region_matrix(self, region_name: str, sub_region_name: str) -> Optional[np.ndarray]:
        """Get the 3D matrix for a specific sub-region."""
        try:
            sub_region_key = f"{region_name}-{sub_region_name}"
            return self.sub_region_matrices.get(sub_region_key)
            
        except Exception as e:
            logger.error(f"Failed to get sub-region matrix: {e}")
            return None
    
    # =============================================================================
    # MATRIX OVERLAY & VISUAL SWITCHING FUNCTIONS
    # =============================================================================
    
    def switch_matrix_overlay(self, overlay_type: str) -> Dict[str, Any]:
        """
        Switch between different matrix overlays for visualization.
        
        Args:
            overlay_type: 'nodes', 'fragments', 'both', 'none'
            
        Returns:
            Dictionary with overlay coordinates and metadata
        """
        try:
            overlay_data = {
                'overlay_type': overlay_type,
                'timestamp': datetime.now().isoformat(),
                'coordinates': [],
                'metadata': {}
            }
            
            if overlay_type == 'nodes':
                # Extract all node coordinates from node_placement_matrix
                for node_id, node_data in self.nodes.items():
                    coords = node_data['coordinates']
                    overlay_data['coordinates'].append({
                        'id': node_id,
                        'position': coords,
                        'type': 'node',
                        'status': node_data.get('status', 'inactive'),
                        'region': node_data.get('region'),
                        'sub_region': node_data.get('sub_region'),
                        'frequency': node_data.get('status_frequency', 0.0)
                    })
                overlay_data['metadata'] = {
                    'total_nodes': len(self.nodes),
                    'active_nodes': sum(1 for n in self.nodes.values() if n.get('active', False)),
                    'grid_type': 'brain_matrix'
                }
                
            elif overlay_type == 'fragments':
                # Extract fragment spaces from mirror grid if available
                if self.mirror_grid_enabled and hasattr(self, 'mirror_grid'):
                    for region_name, region_data in self.mirror_grid['regions'].items():
                        for sub_region_name, sub_region_data in region_data['sub_regions'].items():
                            for space_id, space_data in sub_region_data.get('fragment_storage', {}).items():
                                coords = space_data['coordinates']
                                overlay_data['coordinates'].append({
                                    'id': space_id,
                                    'position': coords,
                                    'type': 'fragment_space',
                                    'capacity': space_data.get('capacity', 0),
                                    'region': space_data.get('region'),
                                    'sub_region': space_data.get('sub_region'),
                                    'entanglement_link': space_data.get('entanglement_link'),
                                    'fragments_stored': len(space_data.get('fragments', {}))
                                })
                    overlay_data['metadata'] = {
                        'total_fragment_spaces': len(overlay_data['coordinates']),
                        'mirror_grid_enabled': True,
                        'grid_type': 'mirror_matrix'
                    }
                else:
                    overlay_data['metadata'] = {
                        'total_fragment_spaces': 0,
                        'mirror_grid_enabled': False,
                        'error': 'Mirror grid not available'
                    }
                
            elif overlay_type == 'both':
                # Combine both node and fragment overlays
                nodes_overlay = self.switch_matrix_overlay('nodes')
                fragments_overlay = self.switch_matrix_overlay('fragments')
                
                combined_coordinates = []
                if nodes_overlay['success']:
                    combined_coordinates.extend(nodes_overlay['overlay_data']['coordinates'])
                if fragments_overlay['success']:
                    combined_coordinates.extend(fragments_overlay['overlay_data']['coordinates'])
                
                overlay_data['coordinates'] = combined_coordinates
                overlay_data['metadata'] = {
                    'total_nodes': nodes_overlay['overlay_data']['metadata'].get('total_nodes', 0) if nodes_overlay['success'] else 0,
                    'total_fragment_spaces': fragments_overlay['overlay_data']['metadata'].get('total_fragment_spaces', 0) if fragments_overlay['success'] else 0,
                    'grid_type': 'combined_matrix'
                }
                
            elif overlay_type == 'none':
                # Return empty overlay (grid only)
                overlay_data['metadata'] = {
                    'grid_only': True,
                    'grid_type': 'base_grid'
                }
            
            logger.info(f"Matrix overlay switched to: {overlay_type} ({len(overlay_data['coordinates'])} items)")
            return {'success': True, 'overlay_data': overlay_data}
            
        except Exception as e:
            logger.error(f"Failed to switch matrix overlay: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_overlay_coordinates_by_region(self, overlay_type: str, region_name: str) -> Dict[str, Any]:
        """
        Get overlay coordinates filtered by specific region.
        
        Args:
            overlay_type: 'nodes' or 'fragments'
            region_name: Name of brain region
            
        Returns:
            Filtered coordinates for the specified region
        """
        try:
            overlay_result = self.switch_matrix_overlay(overlay_type)
            if not overlay_result['success']:
                return overlay_result
            
            all_coordinates = overlay_result['overlay_data']['coordinates']
            region_coordinates = [
                coord for coord in all_coordinates 
                if coord.get('region') == region_name
            ]
            
            region_overlay = {
                'overlay_type': overlay_type,
                'region_filter': region_name,
                'coordinates': region_coordinates,
                'metadata': {
                    'total_in_region': len(region_coordinates),
                    'region_name': region_name,
                    'original_total': len(all_coordinates)
                }
            }
            
            return {'success': True, 'overlay_data': region_overlay}
            
        except Exception as e:
            logger.error(f"Failed to get region overlay coordinates: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_density_compliance(self) -> Dict[str, Any]:
        """
        Validate that brain grid has ~80% nodes and mirror matrix has ~20% fragments.
        Also check edge margins are respected.
        """
        try:
            validation_result = {
                'brain_grid_compliance': {},
                'mirror_grid_compliance': {},
                'edge_margin_compliance': {},
                'overall_compliant': True
            }
            
            # Calculate brain grid density
            total_brain_volume = np.prod(self.grid_dimensions)
            buffer_volume = 26 * 26 * 26 * 8  # 8 corners of buffer space
            usable_brain_volume = total_brain_volume - buffer_volume
            
            total_nodes = len(self.nodes)
            total_seeds = len(self.mycelial_seeds)
            total_populated = total_nodes + total_seeds
            
            brain_density = total_populated / usable_brain_volume if usable_brain_volume > 0 else 0
            node_percentage = total_nodes / total_populated if total_populated > 0 else 0
            seed_percentage = total_seeds / total_populated if total_populated > 0 else 0
            
            # Brain grid validation (should be exactly 80% nodes, 20% seeds)
            brain_compliant = (0.78 <= node_percentage <= 0.82) and (0.18 <= seed_percentage <= 0.22)
            validation_result['brain_grid_compliance'] = {
                'compliant': brain_compliant,
                'node_percentage': node_percentage,
                'seed_percentage': seed_percentage,
                'total_density': brain_density,
                'total_nodes': total_nodes,
                'total_seeds': total_seeds,
                'expected_node_range': '78-82% (target: 80%)',
                'expected_seed_range': '18-22% (target: 20%)'
            }
            
            # Mirror grid validation (if enabled)
            if self.mirror_grid_enabled:
                fragment_spaces = 0
                for region_data in self.mirror_grid['regions'].values():
                    for sub_region_data in region_data['sub_regions'].values():
                        fragment_spaces += len(sub_region_data.get('fragment_storage', {}))
                
                mirror_density = fragment_spaces / usable_brain_volume if usable_brain_volume > 0 else 0
                expected_density_ratio = 0.25  # Should be ~25% of brain density  
                actual_density_ratio = mirror_density / (brain_density or 1)
                mirror_compliant = 0.15 <= actual_density_ratio <= 0.35
                
                validation_result['mirror_grid_compliance'] = {
                    'compliant': mirror_compliant,
                    'fragment_spaces': fragment_spaces,
                    'mirror_density': mirror_density,
                    'density_ratio_to_brain': actual_density_ratio,
                    'expected_ratio_range': f'15-35% of brain density (target: {expected_density_ratio:.0%})'
                }
            else:
                validation_result['mirror_grid_compliance'] = {
                    'compliant': True,
                    'note': 'Mirror grid not enabled'
                }
            
            # Edge margin validation
            buffer_size = 26
            edge_violations = 0
            
            for node_id, node_data in self.nodes.items():
                coords = node_data['coordinates']
                if (coords[0] < buffer_size or coords[0] >= self.grid_dimensions[0] - buffer_size or
                    coords[1] < buffer_size or coords[1] >= self.grid_dimensions[1] - buffer_size or
                    coords[2] < buffer_size or coords[2] >= self.grid_dimensions[2] - buffer_size):
                    edge_violations += 1
            
            edge_compliant = edge_violations == 0
            validation_result['edge_margin_compliance'] = {
                'compliant': edge_compliant,
                'edge_violations': edge_violations,
                'buffer_size': buffer_size,
                'total_nodes_checked': len(self.nodes)
            }
            
            # Overall compliance
            validation_result['overall_compliant'] = (
                brain_compliant and 
                validation_result['mirror_grid_compliance']['compliant'] and 
                edge_compliant
            )
            
            logger.info(f"Density validation: {'PASSED' if validation_result['overall_compliant'] else 'FAILED'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate density compliance: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_brain_structure(self) -> Dict[str, Any]:
        """Get complete brain structure for external systems."""
        try:
            structure = {
                'brain_id': self.brain_id,
                'creation_time': self.creation_time,
                'grid_dimensions': self.grid_dimensions,
                'regions': self.regions,
                'sub_regions': self.sub_regions,
                'nodes': self.nodes,
                'mycelial_seeds': self.mycelial_seeds,
                'mirror_grid': getattr(self, 'mirror_grid', None),
                'brain_signature': self._generate_brain_signature(),
                'statistics': {
                    'total_regions': len(self.regions),
                    'total_sub_regions': len(self.sub_regions),
                    'total_nodes': len(self.nodes),
                    'total_seeds': len(self.mycelial_seeds),
                    'active_nodes': sum(1 for n in self.nodes.values() if n.get('active', False)),
                    'active_seeds': sum(1 for s in self.mycelial_seeds.values() if s.get('active', False))
                }
            }
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to get brain structure: {e}")
            raise RuntimeError(f"Brain structure retrieval failed: {e}") from e
    
    def _generate_brain_signature(self) -> str:
        """Generate unique signature for this brain instance."""
        try:
            signature_parts = [
                self.brain_id[:8],
                str(len(self.regions)),
                str(len(self.nodes)),
                str(len(self.mycelial_seeds)),
                str(hash(str(sorted(self.density_variance.items()))))[:8]
            ]
            return "-".join(signature_parts)
            
        except Exception as e:
            logger.warning(f"Failed to generate brain signature: {e}")
            return f"brain-{self.brain_id[:8]}-unknown"
    
    def validate_brain_structure(self) -> Dict[str, Any]:
        """Comprehensive validation of brain structure integrity."""
        logger.info("Validating brain structure...")
        
        try:
            issues = []
            
            # Validate basic structure
            if not self.regions:
                issues.append("No regions created")
            if not self.sub_regions:
                issues.append("No sub-regions created") 
            if not self.nodes and not self.mycelial_seeds:
                issues.append("No nodes or seeds created")
            
            # Validate matrices
            if self.whole_brain_matrix is None:
                issues.append("Whole brain matrix not initialized")
            if self.node_placement_matrix is None:
                issues.append("Node placement matrix not initialized")
            if self.seed_placement_matrix is None:
                issues.append("Seed placement matrix not initialized")
            
            # Validate frequency consistency
            frequency_issues = 0
            for node_id, node_data in self.nodes.items():
                expected_freq = self.node_status_frequencies.get(node_data.get('status', 'inactive'))
                actual_freq = node_data.get('status_frequency')
                if expected_freq != actual_freq:
                    frequency_issues += 1
            
            if frequency_issues > 0:
                issues.append(f"Frequency inconsistencies in {frequency_issues} nodes")
            
            # Validate ID structure
            id_issues = 0
            for node_id in self.nodes.keys():
                if not node_id.count('-') >= 3:  # region-subregion-node-number format
                    id_issues += 1
            
            if id_issues > 0:
                issues.append(f"Invalid ID structure in {id_issues} nodes")
            
            # Statistics
            stats = {
                'structure_valid': len(issues) == 0,
                'issues_found': issues,
                'regions_count': len(self.regions),
                'sub_regions_count': len(self.sub_regions),
                'nodes_count': len(self.nodes),
                'seeds_count': len(self.mycelial_seeds),
                'mirror_grid_enabled': self.mirror_grid_enabled,
                'frequency_issues': frequency_issues,
                'id_structure_issues': id_issues
            }
            
            if len(issues) == 0:
                logger.info("✅ Brain structure validation passed")
            else:
                logger.warning(f"⚠️ Brain structure validation found {len(issues)} issues")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to validate brain structure: {e}")
            raise RuntimeError(f"Brain structure validation failed: {e}") from e
    
    def run_brain_formation_sequence(self, create_mirror_grid: bool = True) -> Dict[str, Any]:
        """Run complete brain formation sequence."""
        logger.info("🧠 Starting complete brain formation sequence...")
        
        try:
            # Step 1: Load brain seed
            self.load_brain_seed()
            
            # Step 2: Calculate volumes
            self.calculate_anatomical_volumes()
            
            # Step 3: Create regions and sub-regions
            self.create_regions_and_sub_regions()
            
            # Step 4: Populate with blocks, nodes, and seeds
            self.populate_blocks_nodes_and_seeds()
            
            # Step 5: Create mirror grid if requested
            mirror_grid_result = None
            if create_mirror_grid:
                mirror_grid_result = self.create_mirror_grid(nodes_disabled=True)
            
            # Step 6: Activate initial nodes with energy
            activation_result = self.activate_initial_nodes_with_energy()
            
            # Step 7: Validate structure
            validation_result = self.validate_brain_structure()
            
            # Compile results
            formation_result = {
                'success': validation_result['structure_valid'],
                'brain_id': self.brain_id,
                'formation_time': datetime.now().isoformat(),
                'brain_signature': self._generate_brain_signature(),
                'regions_created': len(self.regions),
                'sub_regions_created': len(self.sub_regions),
                'nodes_created': len(self.nodes),
                'seeds_created': len(self.mycelial_seeds),
                'nodes_activated': activation_result['successful_activations'],
                'mirror_grid_created': mirror_grid_result is not None,
                'validation_result': validation_result,
                'energy_storage_id': self.energy_storage.storage_id if self.energy_storage else None
            }
            
            if formation_result['success']:
                logger.info("✅ Brain formation sequence completed successfully")
            else:
                logger.error("❌ Brain formation sequence completed with issues")
            
            return formation_result
            
        except Exception as e:
            logger.error(f"Brain formation sequence failed: {e}")
            raise RuntimeError(f"Brain formation failed: {e}") from e








# # --- brain_structure.py V9 - ANATOMICALLY CORRECT WITH PROPER VOLUME CALCULATIONS ---
# """
# Create anatomically correct brain structure with:
# - Real neuroanatomical regions from region_definitions.py
# - Proper volume-based block placement (80% utilization)
# - Static sound borders around sub-regions
# - Hierarchical naming: region-subregion-block
# - Variance per simulation run
# - Safe frequency ranges only
# - Cosmic background in empty spaces
# - Standing wave external fields
# - Dual matrix storage (whole brain + individual sub-regions)

# Architecture: 256³ Grid → External Field → Regions → Sub-regions → Blocks
# """
# # Import corrected region definitions
# from shared.dictionaries.region_definitions import *
# from shared.constants.constants import *
# from stage_1.brain_formation.energy_storage import create_energy_storage_with_brain
# from datetime import datetime
# from typing import Dict, List, Tuple, Any, Optional
# import sys
# import os
# import logging
# import uuid
# import random
# import numpy as np
# import math
# import traceback


# # Add project root to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# # Import sound generators
# try:
#     from shared.sound.sound_generator import SoundGenerator
#     from shared.sound.sounds_of_universe import UniverseSounds
#     from shared.sound.noise_generator import NoiseGenerator
#     SOUND_AVAILABLE = True
# except ImportError as e:
#     print(f"Warning: Sound modules not available: {e}")
#     SOUND_AVAILABLE = False

# # Logging setup
# logger = logging.getLogger("BrainStructure")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# class AnatomicalBrain:
#     """
#     Anatomically correct brain structure with proper volume calculations.
    
#     Key Features:
#     - Real neuroanatomical regions (no artificial hemispheres)
#     - Volume-based block placement (80% utilization per region)
#     - Static sound borders around sub-regions
#     - Safe frequency ranges only
#     - Hierarchical naming: region-subregion-block
#     - Dual matrix storage for efficiency
#     - Variance per simulation run
#     """
    
#     def __init__(self):
#         """Initialize anatomically correct brain structure."""
#         self.brain_id = str(uuid.uuid4())
#         self.creation_time = datetime.now().isoformat()
#         self.grid_dimensions = GRID_DIMENSIONS  # (256, 256, 256)
        
#         # Brain components
#         self.external_field = {}
#         self.regions = {}           # Major anatomical regions
#         self.sub_regions = {}       # Individual sub-region matrices
#         self.whole_brain_matrix = np.zeros(self.grid_dimensions, dtype=object)  # Complete brain
#         self.static_borders = {}    # Sound borders around sub-regions
#         self.brain_waves = {}       # Wave patterns per region
#         self.cosmic_background = {} # Background in empty spaces
#         self.field_integrity = {}
#         self.brain_structure = {}
#         self.brain_seed = None
#         self.region_volumes = {}  # Initialize region volumes
#         self.block_validation_details = []  # Initialize block validation details
#         self.seed_allocation_data = None  # Initialize seed allocation data
        
#         # Sound generators
#         self.sound_generator = None
#         self.universe_sounds = None
#         self.noise_generator = None
#         self.create_sound_generators()
        
#         # Simulation variance parameters
#         self.density_variance = self._generate_density_variance()
        
#         # Energy storage system
#         self.energy_storage = None
        
#         logger.info(f"Anatomical brain initialized: {self.brain_id[:8]}")
    
#     def create_sound_generators(self):
#         """Create sound generators with error handling."""
#         try:
#             if SOUND_AVAILABLE:
#                 self.sound_generator = SoundGenerator(output_dir="output/sounds/brain")
#                 self.universe_sounds = UniverseSounds(output_dir="output/sounds/brain")
#                 self.noise_generator = NoiseGenerator(output_dir="output/sounds/brain")
#                 logger.info("Sound generators created successfully")
#             else:
#                 logger.warning("Sound modules not available - brain will function without sound")
#         except RuntimeError as e:
#             logger.error(f"Error creating sound generators: {e}")
#             self.sound_generator = None
#             self.universe_sounds = None
#             self.noise_generator = None
    
#     def _generate_density_variance(self) -> Dict[str, float]:
#         """Generate random density variance for this simulation."""
#         variance = {}
        
#         # Randomly select 2-3 regions to be more dense
#         all_regions = list(MAJOR_REGIONS.keys())
#         dense_regions = random.sample(all_regions, random.randint(2, 3))
        
#         for region_name in all_regions:
#             if region_name in dense_regions:
#                 variance[region_name] = random.uniform(0.85, 0.95)  # More dense
#             else:
#                 variance[region_name] = random.uniform(0.70, 0.80)  # Standard density
        
#         logger.info(f"Density variance: dense regions = {dense_regions}")
#         return variance
    
    
#     def load_brain_seed(self):
#         """Load brain seed coordinates."""
#         logger.info("Loading brain seed")
        
#         try:
#             if not self.brain_seed:
#                 # Default center position
#                 center = (self.grid_dimensions[0] // 2, 
#                          self.grid_dimensions[1] // 2, 
#                          self.grid_dimensions[2] // 2)
#                 self.brain_seed = {
#                     'position': center, 
#                     'energy': 7.83,  # Schumann resonance
#                     'frequency': BASE_BRAIN_FREQUENCY
#                 }
#                 logger.warning("Using default brain seed position")
            
#             # Validate position within brain area (not external buffer)
#             pos = self.brain_seed['position']
#             buffer_size = 26
            
#             if (pos[0] < buffer_size or pos[0] >= self.grid_dimensions[0] - buffer_size or
#                 pos[1] < buffer_size or pos[1] >= self.grid_dimensions[1] - buffer_size or
#                 pos[2] < buffer_size or pos[2] >= self.grid_dimensions[2] - buffer_size):
                
#                 center_pos = (self.grid_dimensions[0] // 2, 
#                              self.grid_dimensions[1] // 2, 
#                              self.grid_dimensions[2] // 2)
#                 self.brain_seed['position'] = center_pos
#                 logger.warning(f"Brain seed moved to center: {center_pos}")
            
#             logger.info(f"Brain seed loaded at: {self.brain_seed['position']}")
            
#         except RuntimeError as e:
#             logger.error(f"Failed to load brain seed: {e}")
#             raise RuntimeError(f"Brain seed loading failed: {e}") from e
    
#     def create_external_standing_wave_field(self):
#         """Create external standing wave field around brain."""
#         logger.info("Creating external standing wave field")
        
#         try:
#             buffer_size = 26  # 10% buffer around brain
            
#             # Safe frequency for standing waves (avoid unsafe ranges)
#             base_frequency = 40.0  # Safe gamma range
#             wavelength = 343.0 / base_frequency  # Sound wavelength in air
            
#             # Create 3D standing wave pattern
#             standing_waves = []
#             for axis in ['x', 'y', 'z']:
#                 wave_data = {
#                     'axis': axis,
#                     'frequency': base_frequency,
#                     'wavelength': wavelength,
#                     'amplitude': 0.5,
#                     'phase': random.uniform(0, 2 * math.pi),  # Random phase for variance
#                     'nodes': [],
#                     'antinodes': []
#                 }
                
#                 # Calculate node/antinode positions
#                 axis_length = self.grid_dimensions[0]
#                 num_wavelengths = int(axis_length / wavelength)
                
#                 for i in range(num_wavelengths):
#                     node_pos = i * wavelength
#                     antinode_pos = (i + 0.5) * wavelength
                    
#                     if node_pos < axis_length:
#                         wave_data['nodes'].append(node_pos)
#                     if antinode_pos < axis_length:
#                         wave_data['antinodes'].append(antinode_pos)
                
#                 standing_waves.append(wave_data)
            
#             # Generate protective boundary sound
#             boundary_sound_file = None
#             if SOUND_AVAILABLE and self.universe_sounds:
#                 try:
#                     boundary_sound = self.universe_sounds.generate_cosmic_background(
#                         duration=3.0,
#                         amplitude=0.4,
#                         frequency_band='low'  # Safe low frequencies
#                     )
                    
#                     if boundary_sound is not None:
#                         boundary_sound_file = f"brain_boundary_{self.brain_id[:8]}.wav"
#                         self.universe_sounds.save_sound(
#                             boundary_sound,
#                             boundary_sound_file,
#                             f"Brain Boundary Field - {self.brain_id[:8]}"
#                         )
#                         logger.info(f"Boundary sound generated: {boundary_sound_file}")
                
#                 except (RuntimeError, ValueError, KeyError) as e:
#                     logger.warning(f"Failed to generate boundary sound: {e}")
            
#             self.external_field = {
#                 'field_id': str(uuid.uuid4()),
#                 'field_type': 'standing_wave_protection',
#                 'creation_time': datetime.now().isoformat(),
#                 'boundaries': {
#                     'inner_bounds': (buffer_size, buffer_size, buffer_size),
#                     'outer_bounds': (
#                         self.grid_dimensions[0] - buffer_size,
#                         self.grid_dimensions[1] - buffer_size,
#                         self.grid_dimensions[2] - buffer_size
#                     ),
#                     'buffer_thickness': buffer_size
#                 },
#                 'standing_waves': standing_waves,
#                 'field_strength': 0.618,  # Golden ratio
#                 'sound_file': boundary_sound_file,
#                 'applied': True
#             }
            
#             logger.info(f"External standing wave field created with {len(standing_waves)} axes")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to create external field: {e}")
#             raise RuntimeError(f"External field creation failed: {e}") from e
    
#     def calculate_anatomical_volumes(self):
#         """Calculate proper anatomical volumes for each region."""
#         logger.info("Calculating anatomical volumes")
        
#         try:
#             buffer_size = 26
#             region_border_size = 4  # Space for static borders
            
#             # Available brain space (accounting for both external buffer and region borders)
#             total_x = self.grid_dimensions[0] - (2 * buffer_size) - (2 * region_border_size)
#             total_y = self.grid_dimensions[1] - (2 * buffer_size) - (2 * region_border_size)
#             total_z = self.grid_dimensions[2] - (2 * buffer_size) - (2 * region_border_size)
#             total_brain_volume = total_x * total_y * total_z
            
#             logger.info(f"Total available brain volume: {total_brain_volume:,} units")
#             logger.info(f"Region border space reserved: {region_border_size} units per side")
            
#             # Calculate region volumes
#             self.region_volumes = {}
#             for region_name, region_config in MAJOR_REGIONS.items():
#                 base_volume = int(total_brain_volume * region_config['proportion'])
                
#                 # Apply density variance
#                 density_factor = self.density_variance.get(region_name, 0.80)
#                 usable_volume = int(base_volume * density_factor)
                
#                 # Account for border space (region_border_size now properly used)
#                 border_volume = region_border_size * region_border_size * region_border_size * 6  # 6 faces
#                 actual_usable_volume = max(1, usable_volume - border_volume)
                
#                 self.region_volumes[region_name] = {
#                     'total_volume': base_volume,
#                     'usable_volume': actual_usable_volume,
#                     'border_volume': border_volume,
#                     'region_border_size': region_border_size,
#                     'density_factor': density_factor,
#                     'region_bounds': self._calculate_region_bounds(
#                         region_name, total_x, total_y, total_z, buffer_size, region_border_size  # FIX: Added missing parameter
#                     )
#                 }
            
#             logger.info(f"Calculated volumes for {len(self.region_volumes)} regions")
#             logger.info(f"Border space per region: {border_volume} units")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to calculate volumes: {e}")
#             raise RuntimeError(f"Volume calculation failed: {e}") from e
    
#     def _calculate_region_bounds(self, region_name: str, brain_x: int, brain_y: int, 
#                             brain_z: int, buffer_size: int, region_border_size: int) -> Dict[str, int]:
#         """Calculate anatomical bounds for a region."""
        
#         # Get location bias from region definitions
#         region_config = MAJOR_REGIONS[region_name]
#         location_bias = region_config.get('location_bias', (0.5, 0.5, 0.5))
        
#         # Calculate region size based on proportion
#         proportion = region_config['proportion']
#         region_volume = brain_x * brain_y * brain_z * proportion
        
#         # Approximate cubic dimensions
#         base_size = int(region_volume ** (1/3))
        
#         # Apply anatomical shape variance (but account for borders)
#         max_width = brain_x - (2 * region_border_size)
#         max_height = brain_y - (2 * region_border_size)
#         max_depth = brain_z - (2 * region_border_size)
        
#         width = int(min(base_size * random.uniform(0.8, 1.2), max_width))
#         height = int(min(base_size * random.uniform(0.8, 1.2), max_height))
#         depth = int(min(base_size * random.uniform(0.8, 1.2), max_depth))
        
#         # Ensure minimum size
#         width = max(width, 20)
#         height = max(height, 20)
#         depth = max(depth, 20)
        
#         # Position based on anatomical location (including border space)
#         x_start = buffer_size + region_border_size + int((brain_x - width - (2 * region_border_size)) * location_bias[0])
#         y_start = buffer_size + region_border_size + int((brain_y - height - (2 * region_border_size)) * location_bias[1])
#         z_start = buffer_size + region_border_size + int((brain_z - depth - (2 * region_border_size)) * location_bias[2])
        
#         return {
#             'x_start': x_start,
#             'x_end': x_start + width,
#             'y_start': y_start,
#             'y_end': y_start + height,
#             'z_start': z_start,
#             'z_end': z_start + depth,
#             'border_size': region_border_size
#         }
    
#     def create_anatomical_regions(self):
#         """Create anatomically correct regions."""
#         logger.info("Creating anatomical regions")
        
#         try:
#             for region_name, volume_data in self.region_volumes.items():
#                 region_config = MAJOR_REGIONS[region_name]
#                 bounds = volume_data['region_bounds']
                
#                 # Create region data
#                 region_data = {
#                     'region_id': region_name,
#                     'region_name': region_name,
#                     'function': region_config['function'],
#                     'boundaries': bounds,
#                     'total_volume': volume_data['total_volume'],
#                     'usable_volume': volume_data['usable_volume'],
#                     'density_factor': volume_data['density_factor'],
#                     'default_wave': region_config['default_wave'],
#                     'wave_frequency_hz': region_config['wave_frequency_hz'],
#                     'color': region_config['color'].value,
#                     'sound_base_note': region_config['sound_base_note'],
#                     'boundary_type': region_config['boundary_type'],
#                     'sub_regions': {},
#                     'creation_time': datetime.now().isoformat()
#                 }
                
#                 # Create sub-regions
#                 sub_region_names = region_config['sub_regions']
#                 for sub_region_name in sub_region_names:
#                     sub_region_data = self._create_sub_region(
#                         region_name, sub_region_name, bounds
#                     )
#                     region_data['sub_regions'][sub_region_name] = sub_region_data
                
#                 self.regions[region_name] = region_data
            
#             logger.info(f"Created {len(self.regions)} anatomical regions")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to create regions: {e}")
#             raise RuntimeError(f"Region creation failed: {e}") from e
    
#     def _create_sub_region(self, parent_region: str, sub_region_name: str, 
#                           parent_bounds: Dict[str, int]) -> Dict[str, Any]:
#         """Create a single sub-region within parent region."""
        
#         # Get sub-region configuration
#         sub_config = SUB_REGIONS[sub_region_name]
        
#         # Calculate sub-region bounds within parent
#         parent_width = parent_bounds['x_end'] - parent_bounds['x_start']
#         parent_height = parent_bounds['y_end'] - parent_bounds['y_start']
#         parent_depth = parent_bounds['z_end'] - parent_bounds['z_start']
#         parent_volume = parent_width * parent_height * parent_depth
        
#         # Sub-region size based on proportion
#         sub_proportion = sub_config['proportion']
#         sub_volume = int(parent_volume * sub_proportion)
#         sub_size = int(sub_volume ** (1/3))
        
#         # Random position within parent (anatomical variance)
#         border_size = 2
#         max_x_offset = max(0, parent_width - sub_size - border_size)
#         max_y_offset = max(0, parent_height - sub_size - border_size)
#         max_z_offset = max(0, parent_depth - sub_size - border_size)
        
#         x_offset = random.randint(0, max_x_offset) if max_x_offset > 0 else 0
#         y_offset = random.randint(0, max_y_offset) if max_y_offset > 0 else 0
#         z_offset = random.randint(0, max_z_offset) if max_z_offset > 0 else 0
        
#         sub_bounds = {
#             'x_start': parent_bounds['x_start'] + x_offset,
#             'x_end': parent_bounds['x_start'] + x_offset + sub_size,
#             'y_start': parent_bounds['y_start'] + y_offset,
#             'y_end': parent_bounds['y_start'] + y_offset + sub_size,
#             'z_start': parent_bounds['z_start'] + z_offset,
#             'z_end': parent_bounds['z_start'] + z_offset + sub_size
#         }
        
#         # Ensure bounds stay within parent
#         sub_bounds['x_end'] = min(sub_bounds['x_end'], parent_bounds['x_end'] - border_size)
#         sub_bounds['y_end'] = min(sub_bounds['y_end'], parent_bounds['y_end'] - border_size)
#         sub_bounds['z_end'] = min(sub_bounds['z_end'], parent_bounds['z_end'] - border_size)
        
#         return {
#             'sub_region_id': f"{parent_region}-{sub_region_name}",
#             'sub_region_name': sub_region_name,
#             'parent_region': parent_region,
#             'function': sub_config['function'],
#             'brodmann_areas': sub_config.get('brodmann_areas', []),
#             'boundaries': sub_bounds,
#             'volume': sub_volume,
#             'wave_frequency_hz': sub_config['wave_frequency_hz'],
#             'default_wave': sub_config['default_wave'],
#             'color': sub_config['color'].value,
#             'sound_modifier': sub_config.get('sound_modifier', 'perfect_fifth'),
#             'sound_pattern': sub_config.get('sound_pattern', 'flowing_mid'),
#             'boundary_type': sub_config.get('boundary_type', 'gradual'),
#             'connections': sub_config.get('connections', []),
#             'blocks': {},
#             'matrix': None,  # Will be created during block population
#             'creation_time': datetime.now().isoformat()
#         }
    
#     # ADD this helper method to the AnatomicalBrain class:
#     def _calculate_seed_percentage(self, total_blocks: int) -> float:
#         """
#         Calculate mycelial seed percentage based on sub-region size.
#         Bigger areas get higher percentage (5-10% range).
        
#         Args:
#             total_blocks: Number of blocks in sub-region
        
#         Returns:
#             Seed percentage (0.05 to 0.10)
#         """
#         if total_blocks <= 5:
#             return 0.05  # 5% for very small sub-regions
#         elif total_blocks <= 20:
#             return 0.06  # 6% for small sub-regions  
#         elif total_blocks <= 50:
#             return 0.07  # 7% for medium sub-regions
#         elif total_blocks <= 100:
#             return 0.08  # 8% for large sub-regions
#         else:
#             return 0.10  # 10% for very large sub-regions
    
#     def populate_blocks_in_regions(self):
#         """Populate blocks in regions with proper volume utilization."""
#         logger.info("Populating blocks in regions (80% volume utilization)")
        
#         try:
#             block_size = 10  # 10³ = 1000 units per block
#             total_blocks_created = 0
            
#             for region_name, region_data in self.regions.items():
#                 region_blocks = 0
                
#                 for sub_region_name, sub_region_data in region_data['sub_regions'].items():
#                     bounds = sub_region_data['boundaries']
                    
#                     # Calculate available space for blocks
#                     width = bounds['x_end'] - bounds['x_start']
#                     height = bounds['y_end'] - bounds['y_start']
#                     depth = bounds['z_end'] - bounds['z_start']
#                     total_volume = width * height * depth
                    
#                     # 80% utilization
#                     usable_volume = int(total_volume * 0.80)
#                     max_blocks = usable_volume // (block_size ** 3)
                    
#                     # Apply region density variance
#                     density_factor = self.density_variance.get(region_name, 0.80)
#                     actual_blocks = int(max_blocks * density_factor)
                    
#                     if actual_blocks == 0:
#                         actual_blocks = 1  # Minimum one block per sub-region
                    
#                     # Create sub-region matrix
#                     matrix_dims = (
#                         (width // block_size) + 1,
#                         (height // block_size) + 1,
#                         (depth // block_size) + 1
#                     )
#                     sub_region_matrix = np.zeros(matrix_dims, dtype=object)
                    
#                     # Place blocks randomly within sub-region
#                     blocks_placed = 0
#                     occupied_positions = set()
#                     max_attempts = actual_blocks * 3  # Prevent infinite loop
#                     attempts = 0
                    
#                     while blocks_placed < actual_blocks and attempts < max_attempts:
#                         # Random position within sub-region
#                         bx = random.randint(0, max(0, (width // block_size) - 1))
#                         by = random.randint(0, max(0, (height // block_size) - 1))
#                         bz = random.randint(0, max(0, (depth // block_size) - 1))
                        
#                         if (bx, by, bz) not in occupied_positions:
#                             # Create block
#                             block_id = f"{region_name}-{sub_region_name}-{blocks_placed+1:03d}"
                            
#                             block_bounds = {
#                                 'x_start': bounds['x_start'] + bx * block_size,
#                                 'x_end': min(bounds['x_end'], bounds['x_start'] + (bx + 1) * block_size),
#                                 'y_start': bounds['y_start'] + by * block_size,
#                                 'y_end': min(bounds['y_end'], bounds['y_start'] + (by + 1) * block_size),
#                                 'z_start': bounds['z_start'] + bz * block_size,
#                                 'z_end': min(bounds['z_end'], bounds['z_start'] + (bz + 1) * block_size)
#                             }
                            
#                             block_data = {
#                                 'block_id': block_id,
#                                 'parent_region': region_name,
#                                 'parent_sub_region': sub_region_name,
#                                 'hierarchical_name': block_id,
#                                 'grid_position': (bx, by, bz),
#                                 'boundaries': block_bounds,
#                                 'center': (
#                                     (block_bounds['x_start'] + block_bounds['x_end']) // 2,
#                                     (block_bounds['y_start'] + block_bounds['y_end']) // 2,
#                                     (block_bounds['z_start'] + block_bounds['z_end']) // 2
#                                 ),
#                                 'volume': (
#                                     (block_bounds['x_end'] - block_bounds['x_start']) *
#                                     (block_bounds['y_end'] - block_bounds['y_start']) *
#                                     (block_bounds['z_end'] - block_bounds['z_start'])
#                                 ),
#                                 'node_position': None,  # One node per block, calculated later
#                                 'active': False,
#                                 'wave_properties': {},
#                                 'creation_time': datetime.now().isoformat()
#                             }
                            
#                             # Store in sub-region
#                             sub_region_data['blocks'][block_id] = block_data
#                             sub_region_matrix[bx, by, bz] = block_data
                            
#                             # Store in whole brain matrix
#                             center = block_data['center']
#                             self.whole_brain_matrix[center[0], center[1], center[2]] = block_data
                            
#                             occupied_positions.add((bx, by, bz))
#                             blocks_placed += 1
#                             region_blocks += 1
#                             total_blocks_created += 1
                        
#                         attempts += 1
                    
#                     # Store sub-region matrix
#                     sub_region_data['matrix'] = sub_region_matrix
#                     sub_region_data['blocks_created'] = blocks_placed
                    
#                     # Store in separate sub-region storage for efficiency
#                     self.sub_regions[f"{region_name}-{sub_region_name}"] = sub_region_data
                
#                 logger.info(f"Region {region_name}: {region_blocks} blocks created")
            
#             logger.info(f"Total blocks created: {total_blocks_created}")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to populate blocks: {e}")
#             raise RuntimeError(f"Block population failed: {e}") from e
        
#     def activate_initial_nodes_with_energy(self):
#         """
#         Create energy storage system and activate initial nodes in each region.
#         Activates majority of blocks as nodes, reserving space for mycelial seeds.
#         Called after brain structure is complete but before mycelial seeds.
#         """
#         logger.info("🔋 Creating energy storage and activating initial nodes...")
        
#         try:
#             # Create energy storage system with this brain structure
#             self.energy_storage = create_energy_storage_with_brain(self.get_brain_structure())
            
#             # Collect nodes that need activation (accounting for mycelial seed space)
#             nodes_to_activate = []
#             seed_allocation_summary = {}
            
#             for region_name, region_data in self.regions.items():
#                 for sub_region_name, sub_region_data in region_data['sub_regions'].items():
#                     # Get all blocks in this sub-region
#                     blocks = sub_region_data['blocks']
#                     total_blocks = len(blocks)
                    
#                     # Calculate mycelial seed allocation (5-10% based on sub-region size)
#                     seed_percentage = self._calculate_seed_percentage(total_blocks)
#                     seeds_needed = max(1, int(total_blocks * seed_percentage))
                    
#                     # Remaining blocks become nodes
#                     nodes_to_activate_count = total_blocks - seeds_needed
                    
#                     # Store seed allocation for later use
#                     seed_allocation_summary[f"{region_name}-{sub_region_name}"] = {
#                         'total_blocks': total_blocks,
#                         'seeds_allocated': seeds_needed,
#                         'seed_percentage': seed_percentage,
#                         'nodes_allocated': nodes_to_activate_count
#                     }
                    
#                     # Select blocks for node activation (first N blocks)
#                     block_items = list(blocks.items())
                    
#                     for i in range(min(nodes_to_activate_count, len(block_items))):
#                         block_id, block_data = block_items[i]
                        
#                         # Create node activation data
#                         node_activation = {
#                             'node_id': block_id,
#                             'coordinates': block_data['center'],
#                             'region': region_name,
#                             'sub_region': sub_region_name,
#                             'block_data': block_data
#                         }
                        
#                         nodes_to_activate.append(node_activation)
            
#             # Activate nodes using energy storage
#             activation_results = self.energy_storage.activate_initial_nodes(nodes_to_activate)
            
#             # Update brain structure with activation results
#             for activated_node in activation_results['activated_nodes']:
#                 node_id = activated_node['node_id']
                
#                 # Find and update the block in brain structure
#                 for region_name, region_data in self.regions.items():
#                     for sub_region_name, sub_region_data in region_data['sub_regions'].items():
#                         if node_id in sub_region_data['blocks']:
#                             block = sub_region_data['blocks'][node_id]
#                             block['node_active'] = True
#                             block['node_energy'] = activated_node['energy_placed']
#                             block['node_status'] = 'active'
#                             block['activation_time'] = datetime.now().isoformat()
                            
#                             # Update in whole brain matrix if applicable
#                             center = block['center']
#                             if (0 <= center[0] < self.grid_dimensions[0] and 
#                                 0 <= center[1] < self.grid_dimensions[1] and 
#                                 0 <= center[2] < self.grid_dimensions[2]):
#                                 if self.whole_brain_matrix[center] is not None:
#                                     self.whole_brain_matrix[center]['node_active'] = True
            
#             # Store activation summary in brain structure
#             self.brain_structure = self.get_brain_structure()  # Refresh structure
#             self.brain_structure['node_activation'] = {
#                 'activation_time': datetime.now().isoformat(),
#                 'total_nodes_activated': activation_results['successful_activations'],
#                 'total_energy_used': activation_results['total_energy_used'],
#                 'energy_storage_id': self.energy_storage.storage_id,
#                 'activation_results': activation_results,
#                 'seed_allocation_summary': seed_allocation_summary
#             }
            
#             # Store seed allocation for mycelial seed creation
#             self.seed_allocation_data = seed_allocation_summary
            
#             logger.info("✅ Node activation complete:")
#             logger.info(f"   Nodes activated: {activation_results['successful_activations']}")
#             logger.info(f"   Energy used: {activation_results['total_energy_used']:.1f} SEU")
#             logger.info(f"   Mycelial seed space reserved in {len(seed_allocation_summary)} sub-regions")
#             logger.info(f"   Energy storage: {self.energy_storage.storage_id[:8]}")
            
#             return activation_results
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to activate initial nodes with energy: {e}")
#             raise RuntimeError(f"Node activation failed: {e}") from e

#     def trigger_brain_development(self, visualize: bool = False):
#         """Trigger brain development with optional visualization."""
#         logger.info("🧠 Starting anatomically correct brain development")
        
#         try:
#             self.load_brain_seed()
#             self.create_external_standing_wave_field()
#             self.calculate_anatomical_volumes()
#             self.create_anatomical_regions()
            
#             if visualize:
#                 logger.info("📊 Visualizing region volumes...")
#                 self.visualize_brain_development("volumes")
            
#             self.populate_blocks_in_regions()
#             self.activate_initial_nodes_with_energy()
            
#             # ADD THIS LINE - Create mycelial seeds after node activation:
#             self.create_mycelial_seeds_in_reserved_space()
            
#             if visualize:
#                 logger.info("📊 Visualizing block distribution...")
#                 self.visualize_brain_development("blocks")
            
#             self.create_static_sound_borders()
#             self.assign_brain_wave_properties()
#             self.apply_cosmic_background()
#             self.test_field_integrity()
#             self.save_brain_structure()
            
#             logger.info("✅ Anatomical brain development completed successfully")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Brain development failed: {e}")
#             raise RuntimeError(f"Brain development failed: {e}") from e

#     # ADD this method to create mycelial seeds after node activation:
#     def create_mycelial_seeds_in_reserved_space(self):
#         """
#         Create mycelial seeds in the reserved space (blocks not activated as nodes).
#         Must be called after activate_initial_nodes_with_energy().
#         """
#         logger.info("🌱 Creating mycelial seeds in reserved space...")
        
#         try:
#             if not hasattr(self, 'seed_allocation_data'):
#                 raise RuntimeError("Seed allocation data not found - run node activation first")
            
#             brain_grid_seeds = {}
#             mirror_grid_seeds = {}
#             total_seeds_created = 0
            
#             for sub_region_key, allocation_data in self.seed_allocation_data.items():
#                 region_name, sub_region_name = sub_region_key.split('-', 1)
#                 sub_region_data = self.regions[region_name]['sub_regions'][sub_region_name]
                
#                 # Get blocks not activated as nodes (remaining blocks)
#                 all_blocks = list(sub_region_data['blocks'].items())
#                 nodes_allocated = allocation_data['nodes_allocated']
#                 seeds_needed = allocation_data['seeds_allocated']
                
#                 # Blocks for seeds are the ones NOT used for nodes
#                 seed_blocks = all_blocks[nodes_allocated:nodes_allocated + seeds_needed]
#                 if len(seed_blocks) < seeds_needed:
#                     raise RuntimeError(f"Insufficient blocks for seeds in {sub_region_key}")
                
#                 for i, (block_id, block_data) in enumerate(seed_blocks):
#                     # Create seed ID
#                     seed_id = f"mycelial_seed_{region_name}_{sub_region_name}_{i+1:03d}"
                    
#                     # Create seed data
#                     seed_data = {
#                         'seed_id': seed_id,
#                         'coordinates': block_data['center'],
#                         'region': region_name,
#                         'sub_region': sub_region_name,
#                         'block_id': block_id,
#                         'frequency': MYCELIAL_SEED_FREQUENCY_DORMANT,
#                         'energy_level': 0.0,
#                         'state': 'dormant',
#                         'creation_time': datetime.now().isoformat(),
#                         'seed_type': 'energy_tower',
#                         'quantum_channel_capable': True,
#                         'field_modulation_capable': True
#                     }
                    
#                     # Store in both grids
#                     brain_grid_seeds[seed_id] = seed_data.copy()
#                     mirror_grid_seeds[seed_id] = seed_data.copy()
#                     mirror_grid_seeds[seed_id]['grid_type'] = 'mirror'
                    
#                     # Mark block as seed location
#                     block_data['mycelial_seed'] = True
#                     block_data['seed_id'] = seed_id
#                     block_data['seed_data'] = seed_data
                    
#                     total_seeds_created += 1
            
#             # Store mycelial seeds in brain structure
#             mycelial_seeds_data = {
#                 'creation_time': datetime.now().isoformat(),
#                 'total_seeds_created': total_seeds_created,
#                 'brain_grid': brain_grid_seeds,
#                 'mirror_grid': mirror_grid_seeds,
#                 'seed_allocation_summary': self.seed_allocation_data,
#                 'seeds_ready': True
#             }
            
#             # Add to brain structure
#             if not hasattr(self, 'brain_structure') or not self.brain_structure:
#                 self.brain_structure = self.get_brain_structure()
            
#             self.brain_structure['mycelial_seeds'] = mycelial_seeds_data
            
#             logger.info("✅ Mycelial seeds created:")
#             logger.info(f"   Total seeds: {total_seeds_created}")
#             logger.info(f"   Brain grid seeds: {len(brain_grid_seeds)}")
#             logger.info(f"   Mirror grid seeds: {len(mirror_grid_seeds)}")
            
#             return mycelial_seeds_data
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to create mycelial seeds: {e}")
#             raise RuntimeError(f"Mycelial seed creation failed: {e}") from e
        
#     def create_static_sound_borders(self):
#         """Create static sound borders around sub-regions."""
#         logger.info("Creating static sound borders around sub-regions")
        
#         try:
#             for region_name, region_data in self.regions.items():
#                 for sub_region_name, sub_region_data in region_data['sub_regions'].items():
#                     border_id = f"{region_name}-{sub_region_name}-border"
                    
#                     # Generate appropriate static noise for this sub-region
#                     static_sound_file = None
#                     if SOUND_AVAILABLE and self.noise_generator:
#                         try:
#                             # Different noise types for different functions
#                             noise_type = self._get_noise_type_for_function(
#                                 sub_region_data['function']
#                             )
                            
#                             static_noise = self.noise_generator.generate_noise(
#                                 noise_type=noise_type,
#                                 duration=2.0,
#                                 amplitude=0.2  # Safe amplitude
#                             )
                            
#                             if static_noise is not None:
#                                 static_sound_file = f"static_{region_name}_{sub_region_name}_{self.brain_id[:8]}.wav"
#                                 self.noise_generator.save_noise(
#                                     static_noise,
#                                     static_sound_file,
#                                     f"Static Border - {region_name} {sub_region_name}"
#                                 )
                        
#                         except (RuntimeError, ValueError, KeyError) as e:
#                             logger.warning(f"Failed to generate static for {border_id}: {e}")
                    
#                     # Create border data
#                     bounds = sub_region_data['boundaries']
#                     border_data = {
#                         'border_id': border_id,
#                         'parent_region': region_name,
#                         'parent_sub_region': sub_region_name,
#                         'boundaries': bounds,
#                         'border_thickness': 3,  # 3 voxel thick border
#                         'field_strength': random.uniform(0.4, 0.7),
#                         'permeability': random.uniform(0.3, 0.5),  # Permeable
#                         'sound_file': static_sound_file,
#                         'noise_type': self._get_noise_type_for_function(sub_region_data['function']),
#                         'frequency_range': self._get_safe_frequency_range(sub_region_data['default_wave']),
#                         'creation_time': datetime.now().isoformat()
#                     }
                    
#                     self.static_borders[border_id] = border_data
#                     sub_region_data['static_border'] = border_data
            
#             logger.info(f"Created {len(self.static_borders)} static sound borders")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to create static borders: {e}")
#             raise RuntimeError(f"Static border creation failed: {e}") from e
#     # ADD this method to get brain structure:
#     def get_brain_structure(self) -> Dict[str, Any]:
#         """Get current brain structure state."""
#         return {
#             'brain_id': self.brain_id,
#             'creation_time': self.creation_time,
#             'grid_dimensions': self.grid_dimensions,
#             'regions': self.regions,
#             'sub_regions': self.sub_regions,
#             'external_field': self.external_field,
#             'static_borders': self.static_borders,
#             'cosmic_background': self.cosmic_background
#         }

#     def _get_noise_type_for_function(self, function: str) -> str:
#         """Get appropriate noise type based on region function."""
#         if 'visual' in function or 'color' in function:
#             return 'pink'  # Visual processing
#         elif 'auditory' in function or 'sound' in function:
#             return 'white'  # Audio processing
#         elif 'motor' in function or 'movement' in function:
#             return 'brown'  # Motor functions
#         elif 'memory' in function or 'formation' in function:
#             return 'blue'   # Memory functions
#         elif 'emotion' in function or 'fear' in function:
#             return 'violet' # Emotional processing
#         else:
#             return 'white'  # Default
    
#     def _get_safe_frequency_range(self, wave_type: str) -> Tuple[float, float]:
#         """Get safe frequency range for wave type."""
#         safe_ranges = {
#             'delta': (0.5, 4.0),
#             'theta': (4.0, 8.0),
#             'alpha': (8.0, 13.0),
#             'beta': (13.0, 30.0),
#             'gamma': (30.0, 100.0)
#         }
#         return safe_ranges.get(wave_type, (8.0, 13.0))  # Default to alpha
    
#     def assign_brain_wave_properties(self):
#         """Assign brain wave properties to regions and blocks."""
#         logger.info("Assigning brain wave properties")
        
#         try:
#             for region_name, region_data in self.regions.items():
#                 logger.info(f"Processing region: {region_name}")  # Now using region_name
                
#                 # Assign region wave properties
#                 region_data['wave_properties'] = {
#                     'default_wave': region_data['default_wave'],
#                     'frequency_hz': region_data['wave_frequency_hz'],
#                     'base_note': region_data['sound_base_note'],
#                     'safe_range': self._get_safe_frequency_range(region_data['default_wave']),
#                     'region_name': region_name  # Using region_name here
#                 }
                
#                 for sub_region_name, sub_region_data in region_data['sub_regions'].items():
#                     # Get sub-region specific properties
#                     sub_config = SUB_REGIONS[sub_region_name]
                    
#                     sub_region_data['wave_properties'] = {
#                         'default_wave': sub_config['default_wave'],
#                         'frequency_hz': sub_config['wave_frequency_hz'],
#                         'sound_modifier': sub_config.get('sound_modifier', 'perfect_fifth'),
#                         'sound_pattern': sub_config.get('sound_pattern', 'flowing_mid'),
#                         'safe_range': self._get_safe_frequency_range(sub_config['default_wave']),
#                         'parent_region': region_name  # Using region_name here too
#                     }
                    
#                     # Assign to blocks
#                     blocks_processed = 0
#                     for block_id, block_data in sub_region_data['blocks'].items():
#                         logger.debug(f"Processing block: {block_id}")  # Now using block_id
                        
#                         # Add frequency variance to blocks
#                         base_freq = sub_config['wave_frequency_hz']
#                         frequency_variance = random.uniform(0.95, 1.05)
#                         block_frequency = base_freq * frequency_variance
                        
#                         # Ensure frequency stays in safe range
#                         safe_range = self._get_safe_frequency_range(sub_config['default_wave'])
#                         block_frequency = max(safe_range[0], min(safe_range[1], block_frequency))
                        
#                         block_data['wave_properties'] = {
#                             'wave_type': sub_config['default_wave'],
#                             'frequency_hz': block_frequency,
#                             'amplitude': random.uniform(0.3, 0.8),
#                             'phase': random.uniform(0, 2 * math.pi),
#                             'safe_range': safe_range,
#                             'block_id': block_id,  # Using block_id here
#                             'parent_region': region_name,
#                             'parent_sub_region': sub_region_name
#                         }
                        
#                         # Calculate node position (one per block)
#                         center = block_data['center']
#                         # Add small random offset within block
#                         node_offset = (
#                             random.randint(-3, 3),
#                             random.randint(-3, 3),
#                             random.randint(-3, 3)
#                         )
#                         block_data['node_position'] = (
#                             center[0] + node_offset[0],
#                             center[1] + node_offset[1],
#                             center[2] + node_offset[2]
#                         )
                        
#                         # Store block_id reference for quick lookup
#                         block_data['quick_lookup_id'] = block_id
#                         blocks_processed += 1
                    
#                     logger.info(f"  Sub-region {sub_region_name}: {blocks_processed} blocks processed")
            
#             logger.info("Brain wave properties assigned successfully")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to assign wave properties: {e}")
#             raise RuntimeError(f"Wave property assignment failed: {e}") from e
    
#     def apply_cosmic_background(self):
#         """Apply cosmic background to empty spaces."""
#         logger.info("Applying cosmic background to empty spaces")
        
#         try:
#             # Generate cosmic background sound
#             cosmic_sound_file = None
#             if SOUND_AVAILABLE and self.universe_sounds:
#                 try:
#                     cosmic_background = self.universe_sounds.generate_cosmic_background(
#                         duration=5.0,
#                         amplitude=0.1,  # Very low amplitude for background
#                         frequency_band='full'
#                     )
                    
#                     if cosmic_background is not None:
#                         cosmic_sound_file = f"cosmic_background_{self.brain_id[:8]}.wav"
#                         self.universe_sounds.save_sound(
#                             cosmic_background,
#                             cosmic_sound_file,
#                             f"Cosmic Background - Brain {self.brain_id[:8]}"
#                         )
#                         logger.info(f"Cosmic background sound generated: {cosmic_sound_file}")
                
#                 except (RuntimeError, ValueError, KeyError) as e:
#                     logger.warning(f"Failed to generate cosmic background: {e}")
            
#             # Apply to empty grid spaces
#             empty_spaces = 0
#             buffer_size = 26
            
#             for x in range(buffer_size, self.grid_dimensions[0] - buffer_size):
#                 for y in range(buffer_size, self.grid_dimensions[1] - buffer_size):
#                     for z in range(buffer_size, self.grid_dimensions[2] - buffer_size):
#                         if self.whole_brain_matrix[x, y, z] is None or self.whole_brain_matrix[x, y, z] == 0:
#                             # Apply cosmic background
#                             self.whole_brain_matrix[x, y, z] = {
#                                 'type': 'cosmic_background',
#                                 'position': (x, y, z),
#                                 'amplitude': random.uniform(0.05, 0.15),
#                                 'frequency': BASE_BRAIN_FREQUENCY + random.uniform(-1.0, 1.0),
#                                 'sound_file': cosmic_sound_file
#                             }
#                             empty_spaces += 1
            
#             self.cosmic_background = {
#                 'sound_file': cosmic_sound_file,
#                 'empty_spaces_filled': empty_spaces,
#                 'base_frequency': BASE_BRAIN_FREQUENCY,
#                 'amplitude_range': (0.05, 0.15),
#                 'applied': True
#             }
            
#             logger.info(f"Cosmic background applied to {empty_spaces:,} empty spaces")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to apply cosmic background: {e}")
#             raise RuntimeError(f"Cosmic background application failed: {e}") from e
    
#     def test_field_integrity(self):
#         """Test field integrity and detect harmful frequency clashes."""
#         logger.info("Testing field integrity and frequency safety")
        
#         try:
#             integrity_issues = []
#             critical_issues = []
            
#             # Test 1: External field applied
#             if not self.external_field.get('applied', False):
#                 critical_issues.append("External standing wave field not applied")
            
#             # Test 2: All regions have static borders
#             regions_without_borders = []
#             for region_name, region_data in self.regions.items():
#                 for sub_region_name in region_data['sub_regions'].keys():
#                     border_id = f"{region_name}-{sub_region_name}-border"
#                     if border_id not in self.static_borders:
#                         regions_without_borders.append(f"{region_name}-{sub_region_name}")
            
#             if regions_without_borders:
#                 integrity_issues.append(f"Missing static borders: {regions_without_borders}")
            
#             # Test 3: Frequency safety check
#             unsafe_frequencies = []
#             total_field_energy = 0.0
            
#             for region_data in self.regions.values():
#                 for sub_region_data in region_data['sub_regions'].values():
#                     for block_data in sub_region_data['blocks'].values():
#                         wave_props = block_data.get('wave_properties', {})
#                         frequency = wave_props.get('frequency_hz', 0)
#                         amplitude = wave_props.get('amplitude', 0)
                        
#                         # Check if frequency is in safe range
#                         safe_range = wave_props.get('safe_range', (0, 100))
#                         if not (safe_range[0] <= frequency <= safe_range[1]):
#                             unsafe_frequencies.append({
#                                 'block_id': block_data['block_id'],
#                                 'frequency': frequency,
#                                 'safe_range': safe_range
#                             })
                        
#                         # Accumulate field energy
#                         total_field_energy += amplitude
            
#             if unsafe_frequencies:
#                 critical_issues.append(f"Unsafe frequencies detected: {len(unsafe_frequencies)} blocks")
            
#             # Test 4: Electromagnetic safety (total field energy)
#             if total_field_energy > 100.0:  # Safety threshold
#                 critical_issues.append(f"Dangerous electromagnetic field level: {total_field_energy:.2f}")
            
#             # Test 5: Border permeability check
#             impermeable_borders = []
#             for _, border_data in self.static_borders.items():
#                 if border_data.get('permeability', 1.0) < 0.2:  # Too impermeable
#                     impermeable_borders.append(border_id)
            
#             if impermeable_borders:
#                 integrity_issues.append(f"Borders too impermeable: {impermeable_borders}")
            
#             # Check for critical issues
#             if critical_issues:
#                 error_msg = f"CRITICAL field integrity issues: {critical_issues}"
#                 logger.error(error_msg)
#                 raise RuntimeError(error_msg)
            
#             # Store integrity results
#             self.field_integrity = {
#                 'test_passed': len(integrity_issues) == 0,
#                 'issues_found': integrity_issues,
#                 'critical_issues': critical_issues,
#                 'total_field_energy': total_field_energy,
#                 'electromagnetic_safe': total_field_energy <= 100.0,
#                 'unsafe_frequencies': unsafe_frequencies,
#                 'frequency_safety_passed': len(unsafe_frequencies) == 0,
#                 'test_time': datetime.now().isoformat()
#             }
            
#             if len(integrity_issues) == 0:
#                 logger.info("✅ Field integrity test PASSED - All systems safe")
#             else:
#                 logger.warning(f"⚠️ Field integrity test completed with {len(integrity_issues)} minor issues")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Field integrity test failed: {e}")
#             raise RuntimeError(f"Field integrity test failed: {e}") from e
    
#     def save_brain_structure(self):
#         """Save complete brain structure."""
#         logger.info("Saving anatomically correct brain structure")
        
#         try:
#             # Calculate final statistics
#             total_blocks = 0
#             total_sub_regions = 0
#             total_borders = len(self.static_borders)
            
#             for region_data in self.regions.values():
#                 for sub_region_data in region_data['sub_regions'].values():
#                     total_blocks += len(sub_region_data['blocks'])
#                     total_sub_regions += 1
            
#             # Create complete brain structure
#             self.brain_structure = {
#                 'brain_id': self.brain_id,
#                 'creation_time': self.creation_time,
#                 'completion_time': datetime.now().isoformat(),
#                 'brain_type': 'anatomically_correct',
#                 'grid_dimensions': self.grid_dimensions,
                
#                 # Core components
#                 'external_field': self.external_field,
#                 'regions': self.regions,
#                 'sub_regions': self.sub_regions,
#                 'static_borders': self.static_borders,
#                 'cosmic_background': self.cosmic_background,
#                 'field_integrity': self.field_integrity,
                
#                 # Storage matrices
#                 'whole_brain_matrix': self.whole_brain_matrix,
#                 'sub_region_matrices': {
#                     name: data['matrix'] for name, data in self.sub_regions.items()
#                 },
                
#                 # Statistics
#                 'statistics': {
#                     'total_regions': len(self.regions),
#                     'total_sub_regions': total_sub_regions,
#                     'total_blocks': total_blocks,
#                     'total_static_borders': total_borders,
#                     'volume_utilization': 0.80,
#                     'density_variance': self.density_variance,
#                     'anatomically_correct': True,
#                     'ready_for_neural_network': True
#                 },
                
#                 # Sound files generated
#                 'sound_files': {
#                     'external_field': self.external_field.get('sound_file'),
#                     'cosmic_background': self.cosmic_background.get('sound_file'),
#                     'static_borders': [
#                         border.get('sound_file') for border in self.static_borders.values()
#                         if border.get('sound_file')
#                     ]
#                 },
                
#                 'structure_complete': True
#             }
            
#             # Set completion flag
#             setattr(self, 'FLAG_BRAIN_STRUCTURE_CREATED', True)
            
#             logger.info("✅ Brain structure saved successfully")
#             logger.info(f"   📊 Regions: {len(self.regions)}")
#             logger.info(f"   📊 Sub-regions: {total_sub_regions}")
#             logger.info(f"   📊 Blocks: {total_blocks:,}")
#             logger.info(f"   📊 Static borders: {total_borders}")
#             logger.info(f"   📊 Sound files: {len([f for f in self.brain_structure['sound_files'].values() if f])}")
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Failed to save brain structure: {e}")
#             raise RuntimeError(f"Brain structure save failed: {e}") from e
        
#     def _plot_region_volumes(self, ax):
#         """Plot wireframe boxes for each region."""
#         region_colors = {
#             'frontal_cortex': 'blue', 'parietal_cortex': 'green', 
#             'temporal_cortex': 'gold', 'occipital_cortex': 'purple',
#             'limbic_system': 'red', 'cerebellum': 'cyan', 'brainstem': 'gray'
#         }
        
#         for region_name, region_data in self.regions.items():
#             bounds = region_data['boundaries']
#             color = region_colors.get(region_name, 'black')
#             self._draw_wireframe_box(ax, bounds, color, alpha=0.3)

#     def _plot_block_distribution(self, ax):
#         """Plot block centers as scattered points."""
#         region_colors = {
#             'frontal_cortex': 'blue', 'parietal_cortex': 'green',
#             'temporal_cortex': 'gold', 'occipital_cortex': 'purple',
#             'limbic_system': 'red', 'cerebellum': 'cyan', 'brainstem': 'gray'
#         }
        
#         for region_name, region_data in self.regions.items():
#             color = region_colors.get(region_name, 'black')
#             x_coords, y_coords, z_coords = [], [], []
            
#             for sub_region_data in region_data['sub_regions'].values():
#                 for block_data in sub_region_data['blocks'].values():
#                     center = block_data['center']
#                     x_coords.append(center[0])
#                     y_coords.append(center[1]) 
#                     z_coords.append(center[2])
            
#             if x_coords:
#                 ax.scatter(x_coords, y_coords, z_coords, 
#                         c=color, s=20, alpha=0.6, label=region_name)
#         ax.legend()

#     def _plot_complete_structure(self, ax):
#         """Plot blocks + borders for final validation."""
#         # Plot blocks (smaller points)
#         self._plot_block_distribution(ax)
        
#         # Add border wireframes
#         for _, border_data in self.static_borders.items():
#             bounds = self.static_borders[border_data['border_id']]['boundaries']
#             # Use thin red wireframe for borders
#             self._draw_wireframe_box(ax, bounds, 'red', alpha=0.2, linewidth=0.5)

#     def _draw_wireframe_box(self, ax, bounds, color, alpha=0.3, linewidth=1.0):
#         """Draw wireframe box."""
#         x_min, x_max = bounds['x_start'], bounds['x_end']
#         y_min, y_max = bounds['y_start'], bounds['y_end'] 
#         z_min, z_max = bounds['z_start'], bounds['z_end']
        
#         edges = [
#             ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
#             ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
#             ([x_max, x_min], [y_max, y_max], [z_min, z_min]),
#             ([x_max, x_max], [y_max, y_min], [z_min, z_min]),
#             ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
#             ([x_min, x_min], [y_min, y_max], [z_max, z_max]),
#             ([x_max, x_min], [y_max, y_max], [z_max, z_max]),
#             ([x_max, x_max], [y_max, y_min], [z_max, z_max]),
#             ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
#             ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
#             ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
#             ([x_max, x_max], [y_max, y_max], [z_min, z_max])
#         ]
        
#         for edge in edges:
#             ax.plot(edge[0], edge[1], edge[2], color=color, alpha=alpha, linewidth=linewidth)

#     def visualize_brain_development(self, stage: str = "blocks"):
#         """Simple 3D visualization of brain development stages."""
#         try:
#             import matplotlib.pyplot as plt
#             from mpl_toolkits.mplot3d import Axes3D
            
#             fig = plt.figure(figsize=(12, 8))
#             ax = fig.add_subplot(111, projection='3d')
            
#             if stage == "volumes":
#                 self._plot_region_volumes(ax)
#             elif stage == "blocks":
#                 self._plot_block_distribution(ax)
#             elif stage == "complete":
#                 self._plot_complete_structure(ax)
            
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y') 
#             ax.set_zlabel('Z')
#             ax.set_title(f'Brain Structure - {stage.title()} Stage')
#             ax.set_xlim(0, self.grid_dimensions[0])
#             ax.set_ylim(0, self.grid_dimensions[1])
#             ax.set_zlim(0, self.grid_dimensions[2])
            
#             plt.tight_layout()
#             plt.show()
#             logger.info(f"✅ Brain visualization ({stage}) completed")
            
#         except ImportError:
#             logger.warning("Matplotlib not available - skipping visualization")
#         except (ValueError, RuntimeError) as visualization_error:
#             logger.error(f"Visualization failed for stage '{stage}': {visualization_error}")
    
#     def run_test_simulation(self, test_seed: Dict[str, Any] = None, visualize: bool = False) -> Dict[str, Any]:
#         """Run test simulation with optional test seed and visualization."""
#         logger.info("🧪 Running anatomical brain test simulation")
        
#         try:
#             # Set test seed if provided
#             if test_seed:
#                 self.brain_seed = test_seed
#             else:
#                 self.brain_seed = {
#                     'position': (128, 128, 128),  # Center
#                     'energy': BASE_BRAIN_FREQUENCY,
#                     'frequency': 432.0 * 1.618  # Golden ratio frequency
#                 }
            
#             # Run full development with visualization flag
#             self.trigger_brain_development(visualize=visualize)
            
#             # Verify structure
#             verification_results = self._verify_anatomical_structure()
            
#             # Create test results
#             test_results = {
#                 'test_passed': verification_results['structure_valid'],
#                 'brain_id': self.brain_id,
#                 'test_seed': self.brain_seed,
#                 'anatomical_verification': verification_results,
#                 'statistics': self.brain_structure['statistics'],
#                 'field_integrity': self.field_integrity,
#                 'sound_files_generated': len([f for f in self.brain_structure['sound_files'].values() if f]),
#                 'unique_brain_signature': self._generate_brain_signature(),
#                 'test_time': datetime.now().isoformat()
#             }
            
#             if test_results['test_passed']:
#                 logger.info("✅ Test simulation PASSED - Anatomical brain created successfully")
#             else:
#                 logger.error("❌ Test simulation FAILED - Check verification results")
            
#             return test_results
            
#         except (RuntimeError, ValueError, KeyError) as e:
#             logger.error(f"Test simulation failed: {e}")
#             return {
#                 'test_passed': False,
#                 'error': str(e),
#                 'brain_id': self.brain_id,
#                 'test_time': datetime.now().isoformat()
#             }
    
#     def _verify_anatomical_structure(self) -> Dict[str, Any]:
#         """Verify the anatomical structure is correct."""
#         issues = []
        
#         # Check all major regions exist
#         expected_regions = set(MAJOR_REGIONS.keys())
#         actual_regions = set(self.regions.keys())
#         if expected_regions != actual_regions:
#             issues.append(f"Region mismatch: expected {expected_regions}, got {actual_regions}")
        
#         # Check sub-regions exist for each region - optimized iteration
#         for region_name, region_data in self.regions.items():  # Direct iteration instead of .keys()
#             expected_sub_regions = set(MAJOR_REGIONS[region_name]['sub_regions'])
#             actual_sub_regions = set(region_data['sub_regions'].keys())
#             if expected_sub_regions != actual_sub_regions:
#                 issues.append(f"Sub-region mismatch in {region_name}")
        
#         # Check blocks have proper hierarchical naming - now using block_data
#         naming_issues = 0
#         block_validation_details = []
        
#         for sub_region_name, sub_region_data in self.sub_regions.items():  # Direct iteration
#             for block_id, block_data in sub_region_data['blocks'].items():  # Direct iteration
#                 expected_pattern = f"{sub_region_data['parent_region']}-{sub_region_data['sub_region_name']}-"
                
#                 # Now properly using block_data for validation
#                 if not block_id.startswith(expected_pattern):
#                     naming_issues += 1
#                     block_validation_details.append({
#                         'block_id': block_id,
#                         'expected_pattern': expected_pattern,
#                         'actual_hierarchical_name': block_data.get('hierarchical_name', 'MISSING'),
#                         'parent_region': block_data.get('parent_region', 'UNKNOWN'),
#                         'parent_sub_region': block_data.get('parent_sub_region', 'UNKNOWN'),
#                         'has_node_position': 'node_position' in block_data,
#                         'has_wave_properties': 'wave_properties' in block_data,
#                         'block_active': block_data.get('active', False)
#                     })
        
#         if naming_issues > 0:
#             issues.append(f"Hierarchical naming issues: {naming_issues} blocks")
#             self.block_validation_details = block_validation_details  # Store validation details
#             self.block_validation_details = block_validation_details
        
#         # Check static borders exist - optimized iteration
#         missing_borders = 0
#         border_details = []
        
#         for sub_region_name in self.sub_regions:  # Direct iteration instead of .keys()
#             border_id = f"{sub_region_name}-border"
#             if border_id not in self.static_borders:
#                 missing_borders += 1
#                 border_details.append({
#                     'sub_region': sub_region_name,
#                     'expected_border_id': border_id,
#                     'border_exists': False
#                 })
#             else:
#                 border_details.append({
#                     'sub_region': sub_region_name,
#                     'expected_border_id': border_id,
#                     'border_exists': True,
#                     'border_thickness': self.static_borders[border_id].get('border_thickness', 0),
#                     'permeability': self.static_borders[border_id].get('permeability', 0)
#                 })
        
#         if missing_borders > 0:
#             issues.append(f"Missing static borders: {missing_borders}")
        
#         # Additional block_data usage - validate block integrity
#         total_valid_blocks = 0
#         invalid_blocks = 0
        
#         for sub_region_data in self.sub_regions.values():  # Direct iteration
#             for block_data in sub_region_data['blocks'].values():  # Using block_data properly
#                 if (block_data.get('center') and 
#                     block_data.get('boundaries') and 
#                     block_data.get('volume', 0) > 0):
#                     total_valid_blocks += 1
#                 else:
#                     invalid_blocks += 1
        
#         if invalid_blocks > 0:
#             issues.append(f"Invalid block structures: {invalid_blocks} blocks")
        
#         return {
#             'structure_valid': len(issues) == 0,
#             'issues_found': issues,
#             'regions_verified': len(self.regions),
#             'sub_regions_verified': len(self.sub_regions),
#             'blocks_verified': sum(len(sr['blocks']) for sr in self.sub_regions.values()),
#             'borders_verified': len(self.static_borders),
#             'valid_blocks': total_valid_blocks,
#             'invalid_blocks': invalid_blocks,
#             'border_details': border_details,
#             'naming_issues_count': naming_issues,
#             'detailed_validation': {
#                 'block_validation_details': block_validation_details if naming_issues > 0 else [],
#                 'border_validation_details': border_details
#             }
#         }
    
#     def _generate_brain_signature(self) -> str:
#         """Generate unique signature for this brain instance."""
#         # Create signature based on density variance and structure
#         signature_parts = [
#             self.brain_id[:8],
#             str(len(self.regions)),
#             str(sum(len(sr['blocks']) for sr in self.sub_regions.values())),
#             str(hash(str(sorted(self.density_variance.items()))))[:8]
#         ]
#         return "-".join(signature_parts)


# # --- TEST FUNCTION ---
# def test_anatomical_brain():
#     """Test the anatomical brain structure creation."""
#     print("\n" + "="*60)
#     print("🧠 TESTING ANATOMICALLY CORRECT BRAIN STRUCTURE")
#     print("="*60)
    
#     try:
#         # Create brain instance
#         print("1. Creating anatomical brain instance...")
#         brain = AnatomicalBrain()
#         print(f"   ✅ Brain created: {brain.brain_id[:8]}")
#         print(f"   📊 Density variance: {len(brain.density_variance)} regions varied")
        
#         # Run test simulation
#         print("\n2. Running comprehensive test simulation...")
#         test_results = brain.run_test_simulation(visualize=True)  # Enable visualization
        
#         # Display results
#         print("\n" + "="*50)
#         print("🔬 TEST RESULTS")
#         print("="*50)
        
#         if test_results['test_passed']:
#             print("✅ ANATOMICAL BRAIN TEST PASSED!")
#             stats = test_results['statistics']
#             print(f"   🧠 Brain ID: {test_results['brain_id'][:8]}")
#             print(f"   📊 Regions: {stats['total_regions']}")
#             print(f"   📊 Sub-regions: {stats['total_sub_regions']}")
#             print(f"   📊 Blocks: {stats['total_blocks']:,}")
#             print(f"   📊 Static borders: {stats['total_static_borders']}")
#             print(f"   📊 Volume utilization: {stats['volume_utilization']*100:.0f}%")
#             print(f"   🔊 Sound files: {test_results['sound_files_generated']}")
#             print(f"   🔒 Field integrity: {'✅ SAFE' if test_results['field_integrity']['test_passed'] else '❌ ISSUES'}")
#             print(f"   🆔 Brain signature: {test_results['unique_brain_signature']}")
            
#             # Verification details
#             verification = test_results['anatomical_verification']
#             print("\n🔍 ANATOMICAL VERIFICATION:")
#             print(f"   ✅ Structure valid: {verification['structure_valid']}")
#             print(f"   📊 Regions verified: {verification['regions_verified']}")
#             print(f"   📊 Sub-regions verified: {verification['sub_regions_verified']}")
#             print(f"   📊 Blocks verified: {verification['blocks_verified']:,}")
#             print(f"   📊 Borders verified: {verification['borders_verified']}")
            
#             if verification['issues_found']:
#                 print(f"   ⚠️ Issues found: {len(verification['issues_found'])}")
#                 for issue in verification['issues_found']:
#                     print(f"      - {issue}")
            
#         else:
#             print("❌ ANATOMICAL BRAIN TEST FAILED!")
#             if 'error' in test_results:
#                 print(f"   💥 Error: {test_results['error']}")
            
#             if 'anatomical_verification' in test_results:
#                 verification = test_results['anatomical_verification']
#                 if verification['issues_found']:
#                     print(f"   🔍 Issues found: {len(verification['issues_found'])}")
#                     for issue in verification['issues_found']:
#                         print(f"      - {issue}")
        
#         print("="*50)
#         return test_results['test_passed']
        
#     except (RuntimeError, ValueError, KeyError) as e:
#         print(f"❌ CRITICAL ERROR during anatomical brain test: {e}")
        
#         traceback.print_exc()
#         return False


# if __name__ == "__main__":
#     print("🧠 Anatomical Brain Structure Module - Direct Test")
#     SUCCESS = test_anatomical_brain()
    
#     if SUCCESS:
#         print("\n🎉 ALL TESTS PASSED - Anatomical brain structure working perfectly!")
#         print("   Ready for neural network integration")
#     else:
#         print("\n💥 TESTS FAILED - Check errors above")
    
#     print(f"\nTest completed: {'SUCCESS' if SUCCESS else 'FAILED'}")