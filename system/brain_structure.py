# --- brain_structure.py - Core 3D brain structure implementation ---

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
import json
import os
from datetime import datetime

# Import constants and definitions
try:
    from constants.constants import *
    from stage_1.soul_formation.region_definitions import *
except ImportError as e:
    logging.critical(f"Failed to import required modules: {e}")
    raise ImportError(f"Brain structure requires region_definitions.py and constants.py: {e}")

# Configure logging
logger = logging.getLogger("BrainStructure")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class BrainGrid:
    """3D grid representing the brain field structure with regions and basic field properties"""
    
    def __init__(self, dimensions: Tuple[int, int, int] = GRID_DIMENSIONS):
        """Initialize the brain grid with specified dimensions"""
        self.dimensions = dimensions
        logger.info(f"Initializing brain grid with dimensions {dimensions}")
        
        # Core field structures
        self.energy_grid = np.zeros(dimensions, dtype=np.float32)
        self.frequency_grid = np.zeros(dimensions, dtype=np.float32)
        self.resonance_grid = np.zeros(dimensions, dtype=np.float32)
        self.stability_grid = np.zeros(dimensions, dtype=np.float32)
        self.coherence_grid = np.zeros(dimensions, dtype=np.float32)
        
        # Region identification grid (stores string identifiers)
        self.region_grid = np.full(dimensions, "", dtype=object)
        self.sub_region_grid = np.full(dimensions, "", dtype=object)
        self.hemisphere_grid = np.full(dimensions, "", dtype=object)
        
        # Soul presence tracking
        self.soul_presence_grid = np.zeros(dimensions, dtype=np.float32)
        self.soul_frequency_grid = np.zeros(dimensions, dtype=np.float32)
        
        # Placeholders for mycelial network interface
        # Actual implementation will be in mycelial_network.py
        self.mycelial_density_grid = np.zeros(dimensions, dtype=np.float32)
        self.mycelial_energy_grid = np.zeros(dimensions, dtype=np.float32)
        
        # Field wave patterns
        self.wave_pattern_grid = np.zeros(dimensions, dtype=np.complex64)
        
        # Counters and tracking
        self.total_grid_cells = dimensions[0] * dimensions[1] * dimensions[2]
        self.initialized_cells = 0
        self.soul_filled_cells = 0
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        
        # Configuration state
        self.regions_defined = False
        self.field_initialized = False
        self.mycelial_paths_formed = False
        self.geometry_applied = False
        self.sound_integrated = False
        
        # Background field characteristics (baseline noise)
        self.background_frequency = SCHUMANN_FREQUENCY
        self.background_amplitude = 0.05  # Low background amplitude
        self.background_coherence = 0.2   # Low background coherence
        
        logger.info("Brain grid data structures initialized")
    
    def define_hemispheres(self):
        """Define the left and right hemispheres in the brain grid"""
        logger.info("Defining hemisphere boundaries")
        x_mid = self.dimensions[0] // 2
        
        # Set hemisphere boundaries (x-axis split)
        self.hemisphere_grid[:x_mid, :, :] = "left"
        self.hemisphere_grid[x_mid:, :, :] = "right"
        
        # Apply hemisphere characteristics
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                for z in range(self.dimensions[2]):
                    hemi = self.hemisphere_grid[x, y, z]
                    if hemi == "left":
                        # Left hemisphere has more analytical bias
                        self.frequency_grid[x, y, z] += BRAIN_WAVE_TYPES['beta']['frequency_range'][0]
                        self.stability_grid[x, y, z] += 0.6
                    elif hemi == "right":
                        # Right hemisphere has more creative bias
                        self.frequency_grid[x, y, z] += BRAIN_WAVE_TYPES['alpha']['frequency_range'][0]
                        self.coherence_grid[x, y, z] += 0.6
        
        logger.info("Hemisphere definition complete")
                    
    def define_major_regions(self):
        """Define the major brain regions based on proportions and locations"""
        logger.info("Defining major brain regions")
        
        # First make sure hemispheres are defined
        if not np.any(self.hemisphere_grid != ""):
            self.define_hemispheres()
        
        # Calculate total volume
        total_volume = self.total_grid_cells
        
        # Track cells assigned to regions
        assigned_cells = 0
        
        # Process each region and allocate space
        for region_name, region_data in MAJOR_REGIONS.items():
            logger.info(f"Processing region: {region_name}")
            
            # Calculate target cell count based on proportion
            target_cells = int(total_volume * region_data['proportion'])
            region_cells = 0
            
            # Get location bias (center point) for this region
            loc_bias = region_data.get('location_bias', (0.5, 0.5, 0.5))
            
            # Convert to grid coordinates
            center_x = int(loc_bias[0] * self.dimensions[0])
            center_y = int(loc_bias[1] * self.dimensions[1])
            center_z = int(loc_bias[2] * self.dimensions[2])
            
            # Determine radius for this region (proportional to cell count)
            # Using cube root for volume approximation
            radius = int((3 * target_cells / (4 * np.pi))**(1/3))
            
            # First pass: mark cells within radius of center point
            cells_marked = self._mark_region_cells(
                region_name, center_x, center_y, center_z, radius)
            
            # Adjust if needed (grow or shrink region to match proportion)
            while cells_marked < target_cells * 0.95:
                # Grow region if too small
                radius += 1
                self.region_grid = np.full(self.dimensions, "", dtype=object)  # Reset
                cells_marked = self._mark_region_cells(
                    region_name, center_x, center_y, center_z, radius)
                
            assigned_cells += cells_marked
            
            # Apply region-specific field characteristics
            self._apply_region_field_properties(region_name, region_data)
            
            logger.info(f"Region {region_name} allocated {cells_marked} cells (target: {target_cells})")
        
        # Check coverage
        coverage = assigned_cells / total_volume
        logger.info(f"Major region definition complete. Grid coverage: {coverage:.2f}")
        self.regions_defined = True
    
    def _mark_region_cells(self, region_name: str, center_x: int, center_y: int, 
                          center_z: int, radius: int) -> int:
        """Mark cells within radius of center as belonging to region, returns count"""
        cells_marked = 0
        
        # Use spherical distance with some noise for natural-looking boundaries
        for x in range(max(0, center_x-radius), min(self.dimensions[0], center_x+radius)):
            for y in range(max(0, center_y-radius), min(self.dimensions[1], center_y+radius)):
                for z in range(max(0, center_z-radius), min(self.dimensions[2], center_z+radius)):
                    # Calculate distance with slight noise
                    noise = np.random.uniform(0.85, 1.15)
                    dist = noise * np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
                    
                    # Only assign if cell is not already assigned and within radius
                    if dist <= radius and self.region_grid[x, y, z] == "":
                        self.region_grid[x, y, z] = region_name
                        cells_marked += 1
        
        return cells_marked
    
    def _apply_region_field_properties(self, region_name: str, region_data: Dict):
        """Apply field properties to all cells belonging to a region"""
        # Default field properties if not specified
        wave_freq = region_data.get('wave_frequency_hz', SCHUMANN_FREQUENCY)
        default_wave = region_data.get('default_wave', 'alpha')
        wave_range = BRAIN_WAVE_TYPES.get(default_wave, {'frequency_range': (8.0, 12.0)})['frequency_range']
        
        # Find all cells for this region
        region_indices = np.where(self.region_grid == region_name)
        if len(region_indices[0]) == 0:
            return  # No cells found
            
        # Apply base frequency with slight variations
        for i in range(len(region_indices[0])):
            x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
            
            # Add frequency with natural variation
            variation = np.random.uniform(0.9, 1.1)
            self.frequency_grid[x, y, z] = wave_freq * variation
            
            # Set baseline stability and coherence for this region
            base_stability = 0.5 + 0.2 * np.random.random()
            base_coherence = 0.5 + 0.2 * np.random.random()
            self.stability_grid[x, y, z] = base_stability
            self.coherence_grid[x, y, z] = base_coherence
            
            # Initialize energy levels
            self.energy_grid[x, y, z] = 0.1 + 0.1 * np.random.random()
            
    def define_sub_regions(self):
        """Define sub-regions within major regions"""
        if not self.regions_defined:
            self.define_major_regions()
            
        logger.info("Defining sub-regions within major regions")
        
        # Map each sub-region to its parent
        parent_map = {}
        for sub_name, sub_data in SUB_REGIONS.items():
            parent = sub_data['parent']
            if parent not in parent_map:
                parent_map[parent] = []
            parent_map[parent].append(sub_name)
        
        # Process each major region
        for region_name, sub_list in parent_map.items():
            # Find all cells for this major region
            region_indices = np.where(self.region_grid == region_name)
            if len(region_indices[0]) == 0:
                continue  # Skip if no cells in this region
            
            # Calculate cells per sub-region based on proportions
            total_region_cells = len(region_indices[0])
            
            # Sort sub-regions to ensure consistent allocation
            sub_list.sort()
            
            # Track starting position for allocation
            current_pos = 0
            
            # Allocate cells to each sub-region
            for sub_name in sub_list:
                sub_data = SUB_REGIONS[sub_name]
                proportion = sub_data.get('proportion', 1.0 / len(sub_list))
                target_cells = int(total_region_cells * proportion)
                
                # Allocate this batch of cells to the sub-region
                end_pos = min(current_pos + target_cells, total_region_cells)
                for i in range(current_pos, end_pos):
                    if i >= len(region_indices[0]):
                        break
                    x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
                    self.sub_region_grid[x, y, z] = sub_name
                
                # Apply sub-region specific properties
                self._apply_subregion_field_properties(sub_name, sub_data, 
                                                     region_indices, current_pos, end_pos)
                
                current_pos = end_pos
                logger.debug(f"Sub-region {sub_name} allocated {end_pos-current_pos} cells")
        
        logger.info("Sub-region definition complete")
    
    def _apply_subregion_field_properties(self, sub_name: str, sub_data: Dict, 
                                        region_indices, start_idx: int, end_idx: int):
        """Apply field properties specific to a sub-region"""
        # Platonic solid association affects field properties
        platonic = sub_data.get('platonic_solid', PlatonicSolids.CUBE)
        
        # Get field characteristics based on platonic association
        if isinstance(platonic, PlatonicSolids):
            platonic_name = platonic.value
        else:
            platonic_name = str(platonic)
            
        harmonic_ratios = PLATONIC_HARMONIC_RATIOS.get(platonic_name, [1.0, 2.0])
        
        # Apply field properties to allocated cells
        for i in range(start_idx, end_idx):
            if i >= len(region_indices[0]):
                break
                
            x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
            
            # Boost appropriate field characteristics based on sub-region function
            function = sub_data.get('function', '')
            if 'memory' in function.lower():
                self.coherence_grid[x, y, z] *= 1.2
            elif 'control' in function.lower() or 'motor' in function.lower():
                self.stability_grid[x, y, z] *= 1.2
            elif 'processing' in function.lower():
                self.frequency_grid[x, y, z] *= 1.1
    
    def define_boundaries(self):
        """Define boundaries between regions with appropriate transition properties"""
        if not self.regions_defined:
            self.define_major_regions()
            
        logger.info("Defining region boundaries and transition zones")
        
        # Process each pair of regions with shared boundaries
        for (region1, region2), boundary_type in REGION_BOUNDARIES.items():
            if region1 not in MAJOR_REGIONS or region2 not in MAJOR_REGIONS:
                continue
                
            # Get boundary properties
            boundary_props = BOUNDARY_TYPES.get(boundary_type, BOUNDARY_TYPES['gradual'])
            transition_width = boundary_props['transition_width']
            permeability = boundary_props['permeability']
            
            # Find cells near the boundary
            self._apply_boundary_transition(region1, region2, transition_width, permeability)
        
        # Handle default boundaries for unspecified pairs
        self._apply_default_boundaries()
        
        logger.info("Boundary definition complete")
    
    def _apply_boundary_transition(self, region1: str, region2: str, 
                                 transition_width: int, permeability: float):
        """Create transition zones between two regions"""
        # Find all cells for region1 that are neighbors with region2
        for x in range(1, self.dimensions[0]-1):
            for y in range(1, self.dimensions[1]-1):
                for z in range(1, self.dimensions[2]-1):
                    if self.region_grid[x, y, z] != region1:
                        continue
                        
                    # Check neighbors for region2
                    has_neighbor = False
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if (0 <= nx < self.dimensions[0] and 
                                    0 <= ny < self.dimensions[1] and 
                                    0 <= nz < self.dimensions[2] and
                                    self.region_grid[nx, ny, nz] == region2):
                                    has_neighbor = True
                                    break
                            if has_neighbor:
                                break
                        if has_neighbor:
                            break
                    
                    if has_neighbor:
                        # This is a boundary cell - modify properties
                        # Enhance mycelial density at boundaries for energy transfer
                        self.mycelial_density_grid[x, y, z] = max(
                            self.mycelial_density_grid[x, y, z], 
                            permeability * 0.8)
                        
                        # Set resonance to facilitate cross-boundary communication
                        self.resonance_grid[x, y, z] = max(
                            self.resonance_grid[x, y, z],
                            permeability * 0.7)
    
    def _apply_default_boundaries(self):
        """Apply default boundary properties to any unhandled transitions"""
        # Use neighbor analysis to find transitions
        for x in range(1, self.dimensions[0]-1):
            for y in range(1, self.dimensions[1]-1):
                for z in range(1, self.dimensions[2]-1):
                    region = self.region_grid[x, y, z]
                    if not region:
                        continue
                        
                    # Check neighbors for different regions
                    for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        neighbor_region = self.region_grid[nx, ny, nz]
                        
                        if neighbor_region and neighbor_region != region:
                            # This is a boundary without explicit settings
                            # Use default boundary properties
                            boundary_props = BOUNDARY_TYPES['gradual']
                            permeability = boundary_props['permeability']
                            
                            # Set default boundary characteristics
                            self.mycelial_density_grid[x, y, z] = max(
                                self.mycelial_density_grid[x, y, z], 
                                permeability * 0.6)
                            
                            self.resonance_grid[x, y, z] = max(
                                self.resonance_grid[x, y, z],
                                permeability * 0.5)
        
    def integrate_sound_wave_patterns(self, wave_data: Optional[Dict] = None):
        """Integrate sound wave patterns into the brain field structure"""
        logger.info("Integrating sound wave patterns into brain field")
        
        if not self.regions_defined:
            self.define_major_regions()
            self.define_sub_regions()
        
        # Use provided wave data if available, otherwise generate defaults
        if wave_data is None:
            wave_data = self._generate_default_wave_patterns()
        
        # Map wave patterns to brain regions
        self._map_wave_patterns_to_regions(wave_data)
        
        # Create interference patterns at boundaries
        self._generate_wave_interference_patterns()
        
        logger.info("Sound wave patterns integrated into brain field")
        self.sound_integrated = True
    
    def _generate_default_wave_patterns(self) -> Dict:
        """Generate default wave patterns for major brain regions"""
        wave_patterns = {}
        
        # Generate patterns for each major region based on default waves
        for region_name, region_data in MAJOR_REGIONS.items():
            default_wave = region_data.get('default_wave', 'alpha')
            wave_freq = region_data.get('wave_frequency_hz', SCHUMANN_FREQUENCY)
            
            # Create a wave pattern for this region
            wave_patterns[region_name] = {
                'base_frequency': wave_freq,
                'amplitude': 0.7 + 0.3 * np.random.random(),
                'phase': 2 * np.pi * np.random.random(),
                'wave_type': default_wave,
                'harmonics': []
            }
            
            # Add harmonics based on region's function
            if default_wave == 'beta':
                # More active regions get higher harmonics
                wave_patterns[region_name]['harmonics'] = [
                    wave_freq * 2.0,
                    wave_freq * 3.0,
                    wave_freq * 0.5
                ]
            elif default_wave == 'alpha':
                # Balanced regions get phi harmonics
                wave_patterns[region_name]['harmonics'] = [
                    wave_freq * GOLDEN_RATIO,
                    wave_freq / GOLDEN_RATIO
                ]
            elif default_wave == 'theta':
                # Deep processing regions get deeper harmonics
                wave_patterns[region_name]['harmonics'] = [
                    wave_freq * 0.5,
                    wave_freq * 1.5
                ]
            else:
                # Default harmonics
                wave_patterns[region_name]['harmonics'] = [
                    wave_freq * 2.0
                ]
        
        return wave_patterns
    
    def _map_wave_patterns_to_regions(self, wave_data: Dict):
        """Map sound wave patterns to their respective brain regions"""
        for region_name, wave_pattern in wave_data.items():
            if region_name not in MAJOR_REGIONS:
                logger.warning(f"Unknown region {region_name} in wave data")
                continue
            
            # Find all cells for this region
            region_indices = np.where(self.region_grid == region_name)
            if len(region_indices[0]) == 0:
                continue
                
            # Get wave pattern properties
            base_freq = wave_pattern.get('base_frequency', SCHUMANN_FREQUENCY)
            amplitude = wave_pattern.get('amplitude', 0.8)
            phase = wave_pattern.get('phase', 0.0)
            harmonics = wave_pattern.get('harmonics', [])
            
            # Apply wave pattern to all cells in this region
            for i in range(len(region_indices[0])):
                x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
                
                # Set base wave frequency with small variations
                variation = 1.0 + 0.05 * (np.random.random() - 0.5)
                self.frequency_grid[x, y, z] = base_freq * variation
                
                # Create complex wave pattern with phase and amplitude
                # Store as complex number for phase information
                wave_val = amplitude * np.exp(1j * phase)
                self.wave_pattern_grid[x, y, z] = wave_val
                
                # Add harmonic influence to resonance
                harmonic_strength = min(1.0, len(harmonics) / 5)
                self.resonance_grid[x, y, z] = max(
                    self.resonance_grid[x, y, z],
                    0.5 + 0.5 * harmonic_strength)
    
    def _generate_wave_interference_patterns(self):
        """Generate wave interference patterns at region boundaries"""
        # Find boundary cells
        for x in range(1, self.dimensions[0]-1):
            for y in range(1, self.dimensions[1]-1):
                for z in range(1, self.dimensions[2]-1):
                    region = self.region_grid[x, y, z]
                    if not region:
                        continue
                    
                    # Check neighbors for different regions
                    has_different_neighbor = False
                    neighbor_regions = set()
                    
                    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if (0 <= nx < self.dimensions[0] and 
                            0 <= ny < self.dimensions[1] and 
                            0 <= nz < self.dimensions[2]):
                            
                            neighbor_region = self.region_grid[nx, ny, nz]
                            if neighbor_region and neighbor_region != region:
                                has_different_neighbor = True
                                neighbor_regions.add(neighbor_region)
                    
                    if has_different_neighbor:
                        # This is a boundary cell - create interference pattern
                        self._apply_interference_pattern(x, y, z, region, neighbor_regions)
    
    def _apply_interference_pattern(self, x: int, y: int, z: int, 
                                  region: str, neighbor_regions: set):
        """Apply wave interference pattern at boundary cell"""
        # Get current cell's wave parameters
        current_freq = self.frequency_grid[x, y, z]
        current_wave = self.wave_pattern_grid[x, y, z]
        
        # Calculate average neighbor frequency
        neighbor_freqs = []
        for neighbor in neighbor_regions:
            # Find average frequency for this neighbor region
            region_indices = np.where(self.region_grid == neighbor)
            if len(region_indices[0]) > 0:
                region_freqs = self.frequency_grid[region_indices]
                neighbor_freqs.append(np.mean(region_freqs))
        
        if not neighbor_freqs:
            return
            
        avg_neighbor_freq = np.mean(neighbor_freqs)
        
        # Calculate interference pattern
        freq_ratio = current_freq / avg_neighbor_freq if avg_neighbor_freq > 0 else 1.0
        
        # Check if frequencies are in harmonic relationship
        is_harmonic = False
        for ratio in [0.5, 1.0, 1.5, 2.0, GOLDEN_RATIO, 1/GOLDEN_RATIO]:
            if abs(freq_ratio - ratio) < 0.1:
                is_harmonic = True
                break
        
        # Apply interference effects
        if is_harmonic:
            # Harmonic interference - enhance resonance
            self.resonance_grid[x, y, z] = min(1.0, self.resonance_grid[x, y, z] * 1.2)
            # Enhance coherence slightly at harmonic boundaries
            self.coherence_grid[x, y, z] = min(1.0, self.coherence_grid[x, y, z] * 1.1)
        else:
            # Non-harmonic interference - create more dynamic boundary
            # Add slight frequency modulation
            modulation = 0.1 * (avg_neighbor_freq - current_freq)
            self.frequency_grid[x, y, z] += modulation
            
            # Reduce coherence slightly at non-harmonic boundaries
            self.coherence_grid[x, y, z] *= 0.95
    
    def save_brain_structure(self, filename: str):
        """Save the brain structure to file"""
        logger.info(f"Saving brain structure to {filename}")
        
        try:
            # Create a dictionary with essential data
            brain_data = {
                'dimensions': self.dimensions,
                'creation_time': self.creation_time,
                'last_updated': datetime.now().isoformat(),
                'total_grid_cells': self.total_grid_cells,
                'soul_filled_cells': self.soul_filled_cells,
                'coverage_percent': (self.soul_filled_cells / self.total_grid_cells) * 100,
                'metrics': {
                    'avg_energy': float(np.mean(self.energy_grid)),
                    'avg_frequency': float(np.mean(self.frequency_grid[self.frequency_grid > 0])),
                    'avg_resonance': float(np.mean(self.resonance_grid[self.resonance_grid > 0])),
                    'avg_stability': float(np.mean(self.stability_grid[self.stability_grid > 0])),
                    'avg_coherence': float(np.mean(self.coherence_grid[self.coherence_grid > 0])),
                    'avg_soul_presence': float(np.mean(self.soul_presence_grid[self.soul_presence_grid > 0])),
                    'mycelial_coverage': float(np.sum(self.mycelial_density_grid > 0.1) / self.total_grid_cells)
                },
                'regions': {},
                'field_baseline': {
                    'energy_min': float(np.min(self.energy_grid)),
                    'energy_max': float(np.max(self.energy_grid)),
                    'frequency_min': float(np.min(self.frequency_grid[self.frequency_grid > 0])),
                    'frequency_max': float(np.max(self.frequency_grid)),
                    'resonance_min': float(np.min(self.resonance_grid[self.resonance_grid > 0])),
                    'resonance_max': float(np.max(self.resonance_grid)),
                }
            }
            
            # Add region stats
            for region_name in MAJOR_REGIONS:
                indices = np.where(self.region_grid == region_name)
                if len(indices[0]) == 0:
                    continue
                    
                brain_data['regions'][region_name] = {
                    'cell_count': len(indices[0]),
                    'avg_energy': float(np.mean(self.energy_grid[indices])),
                    'avg_frequency': float(np.mean(self.frequency_grid[indices])),
                    'avg_soul_presence': float(np.mean(self.soul_presence_grid[indices])),
                    'sub_regions': {}
                }
                
                # Add sub-region stats
                for sub_name, sub_data in SUB_REGIONS.items():
                    if sub_data['parent'] != region_name:
                        continue
                        
                    sub_indices = np.where(self.sub_region_grid == sub_name)
                    if len(sub_indices[0]) == 0:
                        continue
                        
                    brain_data['regions'][region_name]['sub_regions'][sub_name] = {
                        'cell_count': len(sub_indices[0]),
                        'avg_energy': float(np.mean(self.energy_grid[sub_indices])),
                        'avg_frequency': float(np.mean(self.frequency_grid[sub_indices])),
                        'avg_soul_presence': float(np.mean(self.soul_presence_grid[sub_indices]))
                    }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(brain_data, f, indent=2)
                
            logger.info(f"Brain structure saved successfully to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving brain structure: {e}", exc_info=True)
            return False

    @classmethod
    def load_brain_structure(cls, filename: str):
        """Load a brain structure from file (metadata only)"""
        logger.info(f"Loading brain structure from {filename}")
        
        try:
            import json
            with open(filename, 'r') as f:
                brain_data = json.load(f)
                
            # Create new instance with loaded dimensions
            dimensions = tuple(brain_data['dimensions'])
            brain_grid = cls(dimensions)
            
            # Set metadata
            brain_grid.creation_time = brain_data['creation_time']
            brain_grid.last_updated = brain_data['last_updated']
            brain_grid.total_grid_cells = brain_data['total_grid_cells']
            brain_grid.soul_filled_cells = brain_data['soul_filled_cells']
            
            # Set other attributes to indicate loaded state
            brain_grid.regions_defined = True
            brain_grid.field_initialized = True
            brain_grid.mycelial_paths_formed = True
            brain_grid.geometry_applied = True
            brain_grid.sound_integrated = True
            
            logger.info(f"Brain structure metadata loaded from {filename}")
            return brain_grid, brain_data
            
        except Exception as e:
            logger.error(f"Error loading brain structure: {e}", exc_info=True)
            return None, None
    
    def prepare_for_soul_distribution(self):
        """Ensure brain structure is fully prepared for soul distribution"""
        # Check if regions are defined, define if not
        if not self.regions_defined:
            self.define_major_regions()
            self.define_sub_regions()
            self.define_boundaries()
        
        # Ensure sound patterns are integrated
        if not self.sound_integrated:
            self.integrate_sound_wave_patterns()
        
        # Mycelial network will be initialized by mycelial_network.py
        
        logger.info("Brain structure prepared for soul distribution")
        return True
    
    def find_optimal_seed_position(self) -> Tuple[int, int, int]:
        """Find optimal position for brain seed placement based on field properties"""
        # First, check for limbic region (ideally near thalamus)
        thalamus_indices = np.where(self.sub_region_grid == "thalamus")
        if len(thalamus_indices[0]) > 0:
            # Use central point of thalamus
            center_idx = len(thalamus_indices[0]) // 2
            return (thalamus_indices[0][center_idx], 
                   thalamus_indices[1][center_idx], 
                   thalamus_indices[2][center_idx])
        
        # Fallback to limbic region
        limbic_indices = np.where(self.region_grid == "limbic")
        if len(limbic_indices[0]) > 0:
            # Use central point of limbic region
            center_idx = len(limbic_indices[0]) // 2
            return (limbic_indices[0][center_idx], 
                   limbic_indices[1][center_idx], 
                   limbic_indices[2][center_idx])
        
        # Last resort: use center of brain grid
        return (self.dimensions[0] // 2, 
               self.dimensions[1] // 2, 
               self.dimensions[2] // 2)


def create_brain_structure(dimensions: Optional[Tuple[int, int, int]] = None, 
                          initialize_fully: bool = True) -> BrainGrid:
    """Create a brain structure with default settings"""
    # Use default dimensions if not specified
    if dimensions is None:
        dimensions = GRID_DIMENSIONS
    
    logger.info(f"Creating brain structure with dimensions {dimensions}")
    
    # Create brain grid
    brain_grid = BrainGrid(dimensions)
    
    # Initialize brain structure
    if initialize_fully:
        # Define regions
        brain_grid.define_major_regions()
        brain_grid.define_sub_regions()
        brain_grid.define_boundaries()
        
        # Initialize sound wave patterns
        brain_grid.integrate_sound_wave_patterns()
        
        logger.info("Brain structure fully initialized")
    
    return brain_grid
