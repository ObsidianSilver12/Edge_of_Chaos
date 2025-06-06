# --- brain_structure.py (Complete Code V2.1) ---

import numpy as np
import logging
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import math
import random

# Import constants and region definitions
# Ensure constants.py contains: PHI, STANDING_WAVE_SIGNAL_SPEED, EDGE_OF_CHAOS_RATIO,
# MAX_EOC_ZONES, SCHUMANN_FREQUENCY, MIN_BRAIN_FREQUENCY, MAX_BRAIN_FREQUENCY,
# MYCELIAL_ACTIVITY_THRESHOLD, MAX_NEURAL_NODE_FREQUENCY, FLOAT_EPSILON, etc.
# Also, region_definitions.py should be in the specified path.
from constants.constants import *

# Attempt to import from the provided region_definitions.py location.
# This path might need adjustment based on your actual project structure.
try:
    from stage_1.evolve.core.region_definitions import *
try:
             from stage_1.evolve.core.region_definitions import *
except ImportError as e:
             # Initialize logger first
             logger = logging.getLogger("BrainStructure")
             if not logger.handlers:
                 handler = logging.StreamHandler()
                 formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                 handler.setFormatter(formatter)
                 logger.addHandler(handler)
                 logger.setLevel(logging.INFO)
        
             logger.error(f"Could not import from system.brain_regions.region_definitions: {e}. "
                          f"Ensure this path is correct and the file exists. Using fallback placeholder data.")
             # Fallback placeholder data if import fails - replace with your actual data structure
             MAJOR_REGIONS = {'frontal': {'proportion': 0.28, 'location_bias': (0.7,0.5,0.2), 'wave_frequency_hz':18.5, 'color':None, 'sound_base_note':'C4', 'glyph':None, 'sub_regions':[]}, 'limbic': {'proportion': 0.11, 'location_bias': (0.5,0.5,0.4), 'wave_frequency_hz':6.8, 'color':None, 'sound_base_note':'D4', 'glyph':None, 'sub_regions':[]}}
             SUB_REGIONS = {'hippocampus': {'parent':'limbic', 'proportion':0.2, 'platonic_solid':None, 'wave_frequency_hz':6.5, 'color':None, 'sound_modifier':'major_third', 'glyph':None}}
             REGION_GRID_DIMENSIONS_FROM_DEF = GRID_DIMENSIONS # Fallback
             DEFINED_REGION_BOUNDARIES = {'default': 'gradual'}
             DEFINED_BOUNDARY_PROPERTIES = {'gradual': {'transition_width': 10, 'permeability': 0.6}}
             SOUND_SHARP_FREQUENCY, SOUND_GRADUAL_FREQUENCY, SOUND_PERMEABLE_FREQUENCY, SOUND_FREQUENCY_VARIATION = 100, 50, 20, 5
             SOUND_MODIFIERS = {'major_third': {'interval': 1.25, 'timbre_shift':0.12, 'amplitude_modifier':1.02}}

logger = logging.getLogger("BrainStructure")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- FieldBlock Class ---
class FieldBlock:
    """Field block for efficient regional field calculations and sound properties."""
    def __init__(self, block_id: str, position: Tuple[int, int, int],
                 size: Tuple[int, int, int], overlapping_major_regions: List[str],
                 block_size_config: Tuple[int, int, int],
                 region_base_frequencies: Dict[str, float]):
        self.block_id = block_id
        self.position = position
        self.size = size
        self.overlapping_major_regions = overlapping_major_regions
        self.block_size_config = block_size_config

        self.energy_total = 0.0
        self.weighted_frequency_sum = 0.0
        self.active_cell_count_in_block = 0

        self.is_boundary_block = len(overlapping_major_regions) > 1
        self.boundary_sound_freq = 0.0
        self.boundary_type: Optional[str] = None
        self.permeability: float = 0.5 # Default permeability

        if self.is_boundary_block and len(overlapping_major_regions) >= 2:
            # Determine primary boundary type based on the first two overlapping major regions
            # A more complex model could handle multiple boundary types within a single block
            pair = tuple(sorted((overlapping_major_regions[0], overlapping_major_regions[1])))
            self.boundary_type = DEFINED_REGION_BOUNDARIES.get(pair, DEFINED_REGION_BOUNDARIES.get('default', 'gradual'))
            
            boundary_props = DEFINED_BOUNDARY_PROPERTIES.get(self.boundary_type)
            if boundary_props:
                self.permeability = boundary_props.get('permeability', 0.5)
            self.boundary_sound_freq = self._calculate_boundary_sound_frequency(region_base_frequencies)


    def _calculate_boundary_sound_frequency(self, region_base_frequencies: Dict[str, float]) -> float:
        if not self.is_boundary_block or not self.boundary_type:
            return 0.0

        base_freq = SOUND_GRADUAL_FREQUENCY # Default
        if self.boundary_type == BOUNDARY_TYPE_SHARP: base_freq = SOUND_SHARP_FREQUENCY
        elif self.boundary_type == BOUNDARY_TYPE_PERMEABLE: base_freq = SOUND_PERMEABLE_FREQUENCY
        
        avg_region_freq = np.mean([region_base_frequencies.get(r, SCHUMANN_FREQUENCY)
                                   for r in self.overlapping_major_regions if r in region_base_frequencies]) \
                          if self.overlapping_major_regions else SCHUMANN_FREQUENCY
        
        return base_freq + random.uniform(-SOUND_FREQUENCY_VARIATION, SOUND_FREQUENCY_VARIATION) + (avg_region_freq * 0.05)

    def update_with_cell_activity(self, energy: float, frequency: float):
        """Update block's aggregate values based on an active cell within it."""
        self.energy_total += energy
        self.weighted_frequency_sum += frequency * energy
        self.active_cell_count_in_block += 1

    def get_aggregated_frequency(self) -> float:
        """Returns the energy-weighted average frequency of active cells in the block."""
        if self.energy_total <= FLOAT_EPSILON or self.active_cell_count_in_block == 0:
            # Fallback: if no active cells, return a representative static frequency
            # (e.g., average of its overlapping regions' base frequencies)
            # This requires access to region_base_frequencies, passed at init of FieldBlock
            # For now, simplified:
            return SCHUMANN_FREQUENCY 
        return self.weighted_frequency_sum / self.energy_total

    def reset_dynamic_values(self):
        """Resets values that are recalculated based on active_cells in each update cycle."""
        self.energy_total = 0.0
        self.weighted_frequency_sum = 0.0
        self.active_cell_count_in_block = 0


class BrainStructure:
    def __init__(self, dimensions: Tuple[int, int, int] = GRID_DIMENSIONS): # Use GRID_DIMENSIONS from constants
        logger.info(f"Initializing BrainStructure with dimensions {dimensions}")
        if dimensions != REGION_GRID_DIMENSIONS_FROM_DEF and REGION_GRID_DIMENSIONS_FROM_DEF is not None : # Compare with imported
             logger.warning(f"BrainStructure dimensions {dimensions} differ from region_definitions.py GRID_DIMENSIONS {REGION_GRID_DIMENSIONS_FROM_DEF}. Using provided dimensions.")
        self.dimensions = dimensions
        self.total_voxels = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        self.brain_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time

        self.regions: Dict[str, Dict[str, Any]] = {}
        self.sub_regions: Dict[str, Dict[str, Any]] = {}
        self.region_grid: np.ndarray = np.full(self.dimensions, None, dtype=object)
        
        self.block_size_config: Tuple[int, int, int] = self._calculate_optimal_block_size()
        self.block_grid_dims: Tuple[int, int, int] = (
            math.ceil(dimensions[0] / self.block_size_config[0]),
            math.ceil(dimensions[1] / self.block_size_config[1]),
            math.ceil(dimensions[2] / self.block_size_config[2])
        )
        self.field_blocks: Dict[str, FieldBlock] = {}

        self.static_field_foundation: np.ndarray = np.zeros(self.dimensions, dtype=np.float32)
        self.inherent_regional_standing_waves: Dict[str, List[Dict[str, Any]]] = {}
        self.phi_based_structural_ratios: Dict[str, float] = {'PHI': PHI}
        self.edge_of_chaos_zones: List[Dict[str, Any]] = []

        self.active_cells: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.mycelial_density_grid: np.ndarray = np.zeros(self.dimensions, dtype=np.float32)
        self.potential_neural_node_sites: List[Tuple[int,int,int]] = []

        self._derived_frequency_grid: Optional[np.ndarray] = None
        self._derived_energy_field: Optional[np.ndarray] = None
        self._derived_coherence_grid: Optional[np.ndarray] = None
        self._derived_resonance_grid: Optional[np.ndarray] = None
        self._complexity_field: Optional[np.ndarray] = None

        self._initialize_anatomy_and_static_features()
        self._initialize_field_blocks()
        self._calculate_all_inherent_standing_waves()
        self._calculate_static_complexity_field_and_eoc_zones()
        self.designate_neural_node_locations() # Designate potential sites

        logger.info(f"BrainStructure {self.brain_id[:8]} initialized with {len(self.regions)} major regions, {len(self.sub_regions)} sub-regions, and {len(self.field_blocks)} field blocks.")

    def _calculate_optimal_block_size(self) -> Tuple[int, int, int]:
        min_dim = min(self.dimensions) if min(self.dimensions) > 0 else 16 # Default if dimensions are zero
        # Ensure block size is a power of 2 for potential FFT or grid algorithms later
        # And not larger than the smallest brain dimension.
        # Start with a reasonable base (e.g., 16 or 32) and adjust down if necessary.
        block_dim_base = 16
        while block_dim_base > min_dim and block_dim_base > 2:
            block_dim_base //= 2
        actual_block_dim = max(2, block_dim_base) # Smallest block size of 2x2x2
        return (actual_block_dim, actual_block_dim, actual_block_dim)

    def _initialize_anatomy_and_static_features(self):
        logger.info("Defining brain anatomy, static EM foundation, and regional sound properties...")
        total_volume_units = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        if total_volume_units == 0:
            logger.error("Brain dimensions are zero. Cannot initialize anatomy.")
            return

        # 1. Define Major Regions
        for region_name, props in MAJOR_REGIONS.items():
            size_variation = 1.0 + random.uniform(-0.07, 0.07)
            freq_variation = 1.0 + random.uniform(-0.03, 0.03)

            region_target_volume = total_volume_units * props['proportion'] * size_variation
            
            # Attempt to derive initial dimensions maintaining some aspect ratio from definitions if available
            # Example: if props had 'aspect_ratios': (rx, ry, rz) where rx*ry*rz=1
            # For now, simple cuboid approximation based on volume.
            approx_side = max(1, int(region_target_volume**(1/3)))

            dim_x = max(1, int(approx_side * (props.get('size_factor_xyz', (1.0, 1.0, 1.0))[0] if 'size_factor_xyz' in props else 1.0) * (1 + random.uniform(-0.1, 0.1))))
            dim_y = max(1, int(approx_side * (props.get('size_factor_xyz', (1.0, 1.0, 1.0))[1] if 'size_factor_xyz' in props else 1.0) * (1 + random.uniform(-0.1, 0.1))))
            dim_z = max(1, int(approx_side * (props.get('size_factor_xyz', (1.0, 1.0, 1.0))[2] if 'size_factor_xyz' in props else 1.0) * (1 + random.uniform(-0.1, 0.1))))

            # Adjust center placement to ensure region fits within brain_structure dimensions
            center_x = int(props['location_bias'][0] * self.dimensions[0])
            center_y = int(props['location_bias'][1] * self.dimensions[1])
            center_z = int(props['location_bias'][2] * self.dimensions[2])
            
            center_x = max(dim_x // 2, min(center_x, self.dimensions[0] - (dim_x // 2) - (dim_x % 2)))
            center_y = max(dim_y // 2, min(center_y, self.dimensions[1] - (dim_y // 2) - (dim_y % 2)))
            center_z = max(dim_z // 2, min(center_z, self.dimensions[2] - (dim_z // 2) - (dim_z % 2)))

            x_start = max(0, center_x - dim_x // 2)
            x_end = min(self.dimensions[0], x_start + dim_x)
            y_start = max(0, center_y - dim_y // 2)
            y_end = min(self.dimensions[1], y_start + dim_y)
            z_start = max(0, center_z - dim_z // 2)
            z_end = min(self.dimensions[2], z_start + dim_z)

            actual_dim_x, actual_dim_y, actual_dim_z = (x_end - x_start), (y_end - y_start), (z_end - z_start)
            if actual_dim_x <= 0 or actual_dim_y <= 0 or actual_dim_z <= 0:
                logger.warning(f"Degenerate major region {region_name} after clamping ({actual_dim_x}x{actual_dim_y}x{actual_dim_z}). Skipping.")
                continue

            self.regions[region_name] = {
                'name': region_name, 'center': (center_x, center_y, center_z),
                'bounds': (x_start, x_end, y_start, y_end, z_start, z_end),
                'size_dims': (actual_dim_x, actual_dim_y, actual_dim_z),
                'base_frequency': props['wave_frequency_hz'] * freq_variation,
                'color_enum': props['color'], 'light_intensity': random.uniform(0.6, 0.9),
                'static_sound_note': props['sound_base_note'], 'glyph_enum': props['glyph'],
                'sub_region_names': list(props['sub_regions']),
                'ambient_sound_amplitude': 0.1 + random.uniform(-0.02, 0.02)
            }
            self.region_grid[x_start:x_end, y_start:y_end, z_start:z_end] = region_name

        # 2. Define Sub-Regions
        for sub_region_name, sub_props in SUB_REGIONS.items():
            parent_region_name = sub_props['parent']
            if parent_region_name in self.regions:
                parent_data = self.regions[parent_region_name]
                parent_bounds = parent_data['bounds']
                parent_center_actual = parent_data['center']

                sub_size_variation = 1.0 + random.uniform(-0.1, 0.1)
                sub_freq_variation = 1.0 + random.uniform(-0.05, 0.05)

                parent_actual_volume = parent_data['size_dims'][0] * parent_data['size_dims'][1] * parent_data['size_dims'][2]
                if parent_actual_volume == 0: continue

                sub_region_target_volume = parent_actual_volume * sub_props['proportion'] * sub_size_variation
                sub_approx_side = max(1, int(sub_region_target_volume**(1/3)))

                s_dim_x = max(1, int(min(parent_data['size_dims'][0] * 0.8, sub_approx_side * (1 + random.uniform(-0.15, 0.15)))))
                s_dim_y = max(1, int(min(parent_data['size_dims'][1] * 0.8, sub_approx_side * (1 + random.uniform(-0.15, 0.15)))))
                s_dim_z = max(1, int(min(parent_data['size_dims'][2] * 0.8, sub_approx_side * (1 + random.uniform(-0.15, 0.15)))))

                s_center_x = parent_center_actual[0] + random.randint(-parent_data['size_dims'][0]//5, parent_data['size_dims'][0]//5)
                s_center_y = parent_center_actual[1] + random.randint(-parent_data['size_dims'][1]//5, parent_data['size_dims'][1]//5)
                s_center_z = parent_center_actual[2] + random.randint(-parent_data['size_dims'][2]//5, parent_data['size_dims'][2]//5)

                sx_start = max(parent_bounds[0], s_center_x - s_dim_x // 2)
                sx_end = min(parent_bounds[1], sx_start + s_dim_x)
                sy_start = max(parent_bounds[2], s_center_y - s_dim_y // 2)
                sy_end = min(parent_bounds[3], sy_start + s_dim_y)
                sz_start = max(parent_bounds[4], s_center_z - s_dim_z // 2)
                sz_end = min(parent_bounds[5], sz_start + s_dim_z)

                actual_s_dim_x, actual_s_dim_y, actual_s_dim_z = (sx_end - sx_start), (sy_end - sy_start), (sz_end - sz_start)
                if actual_s_dim_x <= 0 or actual_s_dim_y <= 0 or actual_s_dim_z <= 0:
                    logger.warning(f"Degenerate sub-region {sub_region_name} after clamping. Skipping.")
                    continue

                self.sub_regions[sub_region_name] = {
                    'name': sub_region_name, 'parent_region': parent_region_name,
                    'center': (s_center_x, s_center_y, s_center_z),
                    'bounds': (sx_start, sx_end, sy_start, sy_end, sz_start, sz_end),
                    'size_dims': (actual_s_dim_x, actual_s_dim_y, actual_s_dim_z),
                    'base_frequency': sub_props['wave_frequency_hz'] * sub_freq_variation,
                    'color_enum': sub_props['color'], 'platonic_solid_enum': sub_props['platonic_solid'],
                    'static_sound_modifier_key': sub_props['sound_modifier'], 'glyph_enum': sub_props['glyph'],
                    'ambient_sound_amplitude': self.regions[parent_region_name]['ambient_sound_amplitude'] * SOUND_MODIFIERS.get(sub_props['sound_modifier'], {'amplitude_modifier':1.0})['amplitude_modifier']
                }
                if sx_start < sx_end and sy_start < sy_end and sz_start < sz_end:
                    self.region_grid[sx_start:sx_end, sy_start:sy_end, sz_start:sz_end] = sub_region_name

        # Static EM Field Foundation
        spacing = int(PHI * random.uniform(4.5, 5.5)) # Varied Phi-based spacing
        anchor_strength = 0.05 + random.uniform(-0.01, 0.01)
        for x_ in range(0, self.dimensions[0], spacing):
            for y_ in range(0, self.dimensions[1], spacing):
                for z_ in range(0, self.dimensions[2], spacing):
                    if self._is_valid_position((x_,y_,z_)):
                        self.static_field_foundation[x_, y_, z_] = anchor_strength
        self.last_updated = datetime.now().isoformat()

    def _initialize_field_blocks(self):
        logger.info(f"Initializing field blocks with block size {self.block_size_config}...")
        self.field_blocks = {}
        
        # Pre-calculate all region base frequencies (major and sub)
        all_region_base_freqs = {}
        for r_name, r_data in self.regions.items():
            all_region_base_freqs[r_name] = r_data['base_frequency']
        for sr_name, sr_data in self.sub_regions.items():
            all_region_base_freqs[sr_name] = sr_data['base_frequency']

        for bx in range(self.block_grid_dims[0]):
            for by in range(self.block_grid_dims[1]):
                for bz in range(self.block_grid_dims[2]):
                    start_x = bx * self.block_size_config[0]
                    start_y = by * self.block_size_config[1]
                    start_z = bz * self.block_size_config[2]

                    end_x = min(start_x + self.block_size_config[0], self.dimensions[0])
                    end_y = min(start_y + self.block_size_config[1], self.dimensions[1])
                    end_z = min(start_z + self.block_size_config[2], self.dimensions[2])
                    
                    actual_size = (end_x - start_x, end_y - start_y, end_z - start_z)
                    if actual_size[0] <= 0 or actual_size[1] <= 0 or actual_size[2] <= 0:
                        continue

                    block_id = f"FB_{bx}_{by}_{bz}"
                    
                    # Determine overlapping major regions for this block more accurately
                    overlapping_maj_regions = set()
                    # Sample points within the block to find regions it intersects
                    sample_points_in_block = [
                        (start_x, start_y, start_z),
                        (end_x - 1, start_y, start_z),
                        (start_x, end_y - 1, start_z),
                        (start_x, start_y, end_z - 1),
                        (start_x + actual_size[0]//2, start_y + actual_size[1]//2, start_z + actual_size[2]//2) # Center
                    ]
                    for sp_x, sp_y, sp_z in sample_points_in_block:
                        if self._is_valid_position((sp_x, sp_y, sp_z)):
                            region_name_at_sp = self.region_grid[sp_x, sp_y, sp_z]
                            if region_name_at_sp:
                                if region_name_at_sp in self.sub_regions:
                                    overlapping_maj_regions.add(self.sub_regions[region_name_at_sp]['parent_region'])
                                elif region_name_at_sp in self.regions:
                                    overlapping_maj_regions.add(region_name_at_sp)
                    
                    self.field_blocks[block_id] = FieldBlock(
                        block_id=block_id,
                        position=(start_x, start_y, start_z),
                        size=actual_size,
                        overlapping_major_regions=list(overlapping_maj_regions if overlapping_maj_regions else ["unknown"]),
                        block_size_config=self.block_size_config,
                        region_base_frequencies=all_region_base_freqs
                    )
        logger.info(f"Initialized {len(self.field_blocks)} field blocks.")

    def _calculate_inherent_standing_waves_for_region(self, region_name: str, region_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        waves = []
        dims = region_data['size_dims']
        base_freq = region_data['base_frequency']
        signal_speed = STANDING_WAVE_SIGNAL_SPEED

        if base_freq <= FLOAT_EPSILON: return []

        for i in range(3): # 0 for x, 1 for y, 2 for z
            L = dims[i]
            if L <= FLOAT_EPSILON: continue # Skip if dimension is zero or too small

            wavelength = signal_speed / base_freq if base_freq > FLOAT_EPSILON else float('inf')
            if wavelength <= FLOAT_EPSILON : continue

            num_half_wavelengths_fit = (2 * L) / wavelength

            # Consider the fundamental mode (n=1, or num_half_wavelengths_fit ~ 2)
            # and perhaps a couple of harmonics if they fit well.
            # This is a simplified representation.
            # We focus on waves that would naturally resonate at or near the region's base frequency.
            
            # Primary mode related to base frequency
            num_nodes = max(2, int(round(num_half_wavelengths_fit)) + 1) # Ensure at least 2 nodes for a wave
            node_spacing = L / (num_nodes - 1) if num_nodes > 1 else L

            # Relative node/antinode positions (0 to L)
            # These are schemas; actual field values would be calculated by a physics engine.
            schema_nodes = [j * node_spacing for j in range(num_nodes)]
            schema_antinodes = [(j + 0.5) * node_spacing for j in range(num_nodes - 1)]
            
            waves.append({
                'type': f'dim_{i}_mode_at_base_freq',
                'frequency': base_freq,
                'wavelength': wavelength,
                'amplitude_potential': 0.3 + random.uniform(-0.05, 0.05),
                'dimension_index': i,
                'num_half_wavelengths_fit': num_half_wavelengths_fit,
                'node_schema_local': schema_nodes,
                'antinode_schema_local': schema_antinodes
            })
        return waves

    def _calculate_all_inherent_standing_waves(self):
        logger.info("Calculating inherent standing wave patterns for all regions...")
        self.inherent_regional_standing_waves = {} # Reset
        for r_name, r_data in self.regions.items():
            if r_data['size_dims'][0]>0 and r_data['size_dims'][1]>0 and r_data['size_dims'][2]>0:
                self.inherent_regional_standing_waves[r_name] = self._calculate_inherent_standing_waves_for_region(r_name, r_data)
        for sr_name, sr_data in self.sub_regions.items():
            if sr_data['size_dims'][0]>0 and sr_data['size_dims'][1]>0 and sr_data['size_dims'][2]>0:
                 self.inherent_regional_standing_waves[sr_name] = self._calculate_inherent_standing_waves_for_region(sr_name, sr_data)
        self.last_updated = datetime.now().isoformat()

    def _calculate_static_complexity_field_and_eoc_zones(self):
        logger.info("Calculating static complexity field and identifying EoC zones...")
        self._complexity_field = np.zeros(self.dimensions, dtype=np.float32)
        all_defined_regions = {**self.regions, **self.sub_regions}

        for r_name, r_data in all_defined_regions.items():
            center = r_data['center']
            eff_radius = np.mean(r_data['size_dims']) / 2.0
            if eff_radius <= FLOAT_EPSILON: continue

            # Get complexity factor from MAJOR_REGIONS definition (can be refined)
            major_region_name_for_complexity = r_data.get('parent_region', r_name)
            base_complexity = MAJOR_REGIONS.get(major_region_name_for_complexity, {}).get('complexity_factor', 0.5) # Default if not in MAJOR_REGIONS (e.g. if r_name is a subregion not listed there)
            if 'complexity_factor' not in MAJOR_REGIONS.get(major_region_name_for_complexity, {}): # If major region itself doesn't have it
                 base_complexity = 0.5 # Generic fallback

            bounds = r_data['bounds']
            for x_v in range(bounds[0], bounds[1]):
                for y_v in range(bounds[2], bounds[3]):
                    for z_v in range(bounds[4], bounds[5]):
                        # Ensure we only affect voxels belonging to *this specific* region/sub-region
                        if self.region_grid[x_v, y_v, z_v] == r_name:
                            distance = math.sqrt((x_v - center[0])**2 + (y_v - center[1])**2 + (z_v - center[2])**2)
                            if distance <= eff_radius:
                                falloff = max(0, 1.0 - (distance / eff_radius))
                                self._complexity_field[x_v, y_v, z_v] += base_complexity * falloff
        
        self._complexity_field += np.random.normal(0, 0.05, self.dimensions)
        self._complexity_field = np.clip(self._complexity_field, 0, 1)

        grad_x = np.gradient(self._complexity_field, axis=0)
        grad_y = np.gradient(self._complexity_field, axis=1)
        grad_z = np.gradient(self._complexity_field, axis=2)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        target_gradient = EDGE_OF_CHAOS_RATIO
        tolerance = 0.05
        
        edge_mask = np.abs(gradient_magnitude - target_gradient) < tolerance
        edge_positions = np.argwhere(edge_mask)

        self.edge_of_chaos_zones = []
        selected_indices = np.random.choice(edge_positions.shape[0], min(edge_positions.shape[0], MAX_EOC_ZONES), replace=False)

        for idx in selected_indices:
            pos = tuple(edge_positions[idx])
            zone_region = self.get_region_at_position(pos)
            self.edge_of_chaos_zones.append({
                'position': pos,
                'static_complexity': float(self._complexity_field[pos]),
                'gradient_magnitude': float(gradient_magnitude[pos]),
                'optimization_potential': float(1.0 - abs(gradient_magnitude[pos] - target_gradient)/tolerance),
                'region': zone_region
            })
        logger.info(f"Identified {len(self.edge_of_chaos_zones)} static EoC zones.")
        self.last_updated = datetime.now().isoformat()

    def activate_cell(self, position: Tuple[int, int, int], energy: float,
                      activity_level: float = 0.1, cell_type: str = "generic",
                      initial_frequency: Optional[float] = None,
                      memory_signature: Optional[str] = None,
                      mycelial_presence: float = 0.0):
        if not self._is_valid_position(position):
            logger.warning(f"Attempted to activate cell at invalid position: {position}")
            return
        if position not in self.active_cells:
            self.active_cells[position] = {}

        self.active_cells[position]['energy'] = max(0.0, energy) # Ensure non-negative
        self.active_cells[position]['activity_level'] = max(0.0, min(1.0, activity_level))
        self.active_cells[position]['type'] = cell_type
        self.active_cells[position]['mycelial_presence'] = max(0.0, min(1.0, mycelial_presence))

        if memory_signature:
            self.active_cells[position]['memory_signature'] = memory_signature

        if initial_frequency:
            self.active_cells[position]['current_frequency'] = max(MIN_BRAIN_FREQUENCY, min(MAX_NEURAL_NODE_FREQUENCY, initial_frequency))
        elif 'current_frequency' not in self.active_cells[position]:
            region_name = self.get_region_at_position(position)
            base_reg_freq = SCHUMANN_FREQUENCY
            if region_name:
                region_data = self.sub_regions.get(region_name) or self.regions.get(region_name)
                if region_data:
                    base_reg_freq = region_data['base_frequency']
            self.active_cells[position]['current_frequency'] = base_reg_freq
        
        self.last_updated = datetime.now().isoformat()
        self._flag_derived_grids_for_recalculation()

    def deactivate_cell(self, position: Tuple[int, int, int]):
        if position in self.active_cells:
            del self.active_cells[position]
            self.last_updated = datetime.now().isoformat()
            self._flag_derived_grids_for_recalculation()

    def set_cell_energy(self, position: Tuple[int, int, int], energy: float):
        if not self._is_valid_position(position): return
        clamped_energy = max(0.0, energy) # Energy cannot be negative
        if position not in self.active_cells:
            self.activate_cell(position, energy=clamped_energy)
        else:
            self.active_cells[position]['energy'] = clamped_energy
        
        # If energy is zero or very low, consider deactivating the cell's specific roles
        if clamped_energy <= FLOAT_EPSILON:
            if self.active_cells[position].get('type') == 'neural_node':
                 self.active_cells[position]['activity_level'] = 0.0 # Neural node becomes inactive
            # Mycelial presence might remain, but energy is gone.
            # Could fully deactivate if other systems also confirm inactivity.
            # For now, just reducing energy. Full deactivation is a higher-level decision.

        self._flag_derived_grids_for_recalculation()
        self.last_updated = datetime.now().isoformat()

    def get_cell_properties(self, position: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        return self.active_cells.get(position)

    def _is_valid_position(self, position: Tuple[int, int, int]) -> bool:
        x, y, z = position
        return (0 <= x < self.dimensions[0] and
                0 <= y < self.dimensions[1] and
                0 <= z < self.dimensions[2])

    def get_region_at_position(self, position: Tuple[int, int, int]) -> Optional[str]:
        if not self._is_valid_position(position) or self.region_grid is None:
            return None
        name = self.region_grid[position]
        return name if name is not None else None # Explicitly return None if grid value is None

    def get_major_region_at_position(self, position: Tuple[int, int, int]) -> Optional[str]:
        region_name = self.get_region_at_position(position)
        if region_name:
            if region_name in self.sub_regions:
                return self.sub_regions[region_name]['parent_region']
            elif region_name in self.regions:
                return region_name
        return None

    def update_mycelial_density(self, position: Tuple[int, int, int], density_value: float):
        if not self._is_valid_position(position) or self.mycelial_density_grid is None: return
        self.mycelial_density_grid[position] = max(0.0, min(1.0, density_value))
        self.last_updated = datetime.now().isoformat()
        self._flag_derived_grids_for_recalculation()

    def _flag_derived_grids_for_recalculation(self):
        self._derived_frequency_grid = None
        self._derived_energy_field = None
        self._derived_coherence_grid = None
        self._derived_resonance_grid = None

    def get_derived_frequency_grid(self, force_recalculate: bool = False) -> np.ndarray:
        if self._derived_frequency_grid is not None and not force_recalculate:
            return self._derived_frequency_grid.copy() # Return a copy

        logger.debug("Recalculating derived frequency grid...")
        # Initialize with a base frequency (e.g., Schumann or a low brainwave)
        freq_grid = np.full(self.dimensions, SCHUMANN_FREQUENCY, dtype=np.float32)

        if self.region_grid is not None:
            # Apply base frequencies from defined regions and sub-regions
            # This ensures all voxels have a base frequency from their assigned region
            unique_regions = np.unique(self.region_grid)
            for r_name in unique_regions:
                if r_name is None: continue # Skip unassigned voxels
                region_data = self.sub_regions.get(r_name) or self.regions.get(r_name)
                if region_data:
                    mask = (self.region_grid == r_name)
                    freq_grid[mask] = region_data['base_frequency']
        
        # Static EM foundation might subtly tune local frequencies
        if self.static_field_foundation is not None:
            freq_grid += self.static_field_foundation * 0.1 # Small influence, can be tuned

        # Active cells' current frequencies override or strongly influence local areas
        for pos, cell_data in self.active_cells.items():
            if 'current_frequency' in cell_data:
                # For simplicity, active cell frequency dominates its voxel
                # A more complex model could blend or create a local field effect
                freq_grid[pos] = cell_data['current_frequency']

        # Influence of inherent regional standing waves (conceptual modulation)
        for r_name, sw_list in self.inherent_regional_standing_waves.items():
            region_data = self.regions.get(r_name) or self.sub_regions.get(r_name)
            if region_data:
                mask = (self.region_grid == r_name) # Mask for the specific region/sub-region
                if np.any(mask): # Check if the mask is not empty
                    for sw in sw_list:
                        # Example: slightly modulate frequency based on amplitude potential
                        # This is highly conceptual. A real model would involve wave superposition.
                        # We assume active standing waves slightly shift the dominant frequency.
                        modulation_factor = (1 + sw['amplitude_potential'] * 0.05 * (random.random() * 0.4 + 0.8))
                        freq_grid[mask] *= modulation_factor
        
        self._derived_frequency_grid = np.clip(freq_grid, MIN_BRAIN_FREQUENCY, MAX_BRAIN_FREQUENCY)
        logger.debug("Derived frequency grid recalculated.")
        return self._derived_frequency_grid.copy()

    def get_derived_field_value(self, position: Tuple[int, int, int], field_type: str) -> float:
        if not self._is_valid_position(position):
            logger.warning(f"Invalid position {position} for get_derived_field_value.")
            return 0.0

        if field_type == 'frequency':
            # Ensure the grid is calculated if it hasn't been
            if self._derived_frequency_grid is None:
                self.get_derived_frequency_grid(force_recalculate=True)
            return float(self._derived_frequency_grid[position]) if self._derived_frequency_grid is not None else 0.0
        
        elif field_type == 'static_complexity':
            if self._complexity_field is not None:
                return float(self._complexity_field[position])
            else:
                logger.warning("Static complexity field accessed before calculation.")
                return 0.0
        
        # Fallback for other types: query active_cell data or a base regional value
        cell_props = self.get_cell_properties(position)
        if cell_props and field_type in cell_props:
            # Ensure return type is float if possible
            val = cell_props[field_type]
            return float(val) if isinstance(val, (int, float)) else 0.0

        region_name = self.get_region_at_position(position)
        if region_name:
            region_data = self.regions.get(region_name) or self.sub_regions.get(region_name)
            if region_data:
                if field_type == 'base_frequency': return float(region_data['base_frequency'])
                # Example for other static properties if needed:
                # if field_type == 'light_intensity': return float(region_data.get('light_intensity', 0.0))

        logger.debug(f"Derived field type '{field_type}' not found for position {position}, returning 0.0.")
        return 0.0

    def report_structural_complexity(self) -> Dict[str, Any]:
        active_region_names = set()
        if self.region_grid is not None: # Ensure region_grid is initialized
            for pos in self.active_cells.keys():
                reg_name = self.get_region_at_position(pos)
                if reg_name: active_region_names.add(reg_name)
        
        mycelial_active_voxels = np.sum(self.mycelial_density_grid > MYCELIAL_ACTIVITY_THRESHOLD) if self.mycelial_density_grid is not None else 0
        total_voxels = self.mycelial_density_grid.size if self.mycelial_density_grid is not None else 1 # Avoid division by zero
        if total_voxels == 0: total_voxels = 1 # Ensure not zero for division

        avg_mycelial_density_active = 0.0
        if self.mycelial_density_grid is not None and mycelial_active_voxels > 0:
             avg_mycelial_density_active = np.mean(self.mycelial_density_grid[self.mycelial_density_grid > MYCELIAL_ACTIVITY_THRESHOLD])

        return {
            'active_cell_count': len(self.active_cells),
            'active_regions_count': len(active_region_names),
            'mycelial_coverage_percent': (mycelial_active_voxels / total_voxels) * 100,
            'avg_mycelial_density_active': float(avg_mycelial_density_active),
            'edge_of_chaos_zones_count': len(self.edge_of_chaos_zones)
        }

    def designate_neural_node_locations(self, base_density_factor: float = 0.005): # Further reduced base density
        logger.info(f"Designating potential neural node locations with base density factor {base_density_factor}...")
        num_designated_total = 0
        self.potential_neural_node_sites: List[Tuple[int,int,int]] = []

        if self.region_grid is None:
            logger.error("Region grid not initialized. Cannot designate neural node locations.")
            return

        all_target_regions = list(self.regions.keys()) + list(self.sub_regions.keys())

        for r_name in all_target_regions:
            is_sub_region = r_name in self.sub_regions
            region_data = self.sub_regions.get(r_name) if is_sub_region else self.regions.get(r_name)
            
            if not region_data or not region_data.get('bounds') or not region_data.get('size_dims') \
               or region_data['size_dims'][0] <=0 or region_data['size_dims'][1] <=0 or region_data['size_dims'][2] <=0 :
                # logger.debug(f"Skipping region/sub-region {r_name} due to missing data or zero size.")
                continue

            bounds = region_data['bounds']
            region_slice = (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3]), slice(bounds[4], bounds[5]))
            
            try:
                region_voxels_mask = (self.region_grid[region_slice] == r_name)
            except IndexError:
                logger.warning(f"IndexError accessing region_grid for {r_name} with slice {region_slice}. Skipping.")
                continue

            coords_in_slice = np.argwhere(region_voxels_mask)
            if coords_in_slice.size == 0:
                # logger.debug(f"No voxels found for region {r_name} in region_grid slice. Skipping.")
                continue
                
            absolute_coords = coords_in_slice + np.array([bounds[0], bounds[2], bounds[4]])
            num_voxels_in_region = absolute_coords.shape[0]

            density_multiplier = 1.0
            parent_major_region = self.sub_regions[r_name]['parent_region'] if is_sub_region else r_name
            if parent_major_region == 'frontal': density_multiplier = 1.5
            elif parent_major_region == 'temporal': density_multiplier = 1.2

            num_nodes_to_place = int(num_voxels_in_region * base_density_factor * density_multiplier)
            num_nodes_to_place = max(1, min(num_nodes_to_place, num_voxels_in_region)) # Ensure at least one, but not more than available

            if num_nodes_to_place > 0:
                chosen_indices = np.random.choice(num_voxels_in_region, num_nodes_to_place, replace=False)
                chosen_sites = absolute_coords[chosen_indices]
                
                for site_pos_array in chosen_sites:
                    self.potential_neural_node_sites.append(tuple(map(int, site_pos_array))) # Ensure int tuple
                    num_designated_total += 1
        
        logger.info(f"Conceptually designated {num_designated_total} potential neural node sites.")
        self.last_updated = datetime.now().isoformat()

    def get_brain_stem_center(self) -> Optional[Tuple[int, int, int]]:
        if 'brain_stem' in self.regions and self.regions['brain_stem'].get('center'):
            return self.regions['brain_stem']['center']
        logger.warning("Brain stem region or its center not defined.")
        return (self.dimensions[0]//2, self.dimensions[1]//2, self.dimensions[2]//2) # Fallback

    def get_limbic_center(self) -> Optional[Tuple[int, int, int]]:
        if 'limbic' in self.regions and self.regions['limbic'].get('center'):
            return self.regions['limbic']['center']
        logger.warning("Limbic region or its center not defined.")
        return (self.dimensions[0]//2, self.dimensions[1]//2, self.dimensions[2]//2) # Fallback







# # --- brain_structure.py (Refactored V6.0.0 - Hybrid Blocks/Sparse with Full Complexity) ---

# """
# Brain Structure Implementation - Hybrid Architecture

# Combines efficient sparse representation with block-based field calculations.
# Includes full complexity: standing waves, phi ratios, edge of chaos detection.
# Integrates with architecture's static field patterns and mycelial networks.

# Architecture:
# - Sparse storage for active brain cells (memory efficient)
# - Block-based static field calculations (computational efficient)
# - Standing wave resonance chambers
# - Phi-based harmonic calculations
# - Edge of chaos detection zones
# - Full region/sub-region anatomical structure
# """

# import numpy as np
# import logging
# import os
# import sys
# import json
# import uuid
# from typing import Dict, List, Tuple, Any, Optional, Union
# from datetime import datetime
# import math
# import random

# # Import constants
# from constants.constants import *

# # --- Logging Setup ---
# logger = logging.getLogger("BrainStructure")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# # --- Try to import metrics tracking ---
# try:
#     import metrics_tracking
#     METRICS_AVAILABLE = True
# except ImportError:
#     logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
#     METRICS_AVAILABLE = False
#     class MetricsPlaceholder:
#         def record_metrics(self, *args, **kwargs): pass
#     metrics_tracking = MetricsPlaceholder()


# class HybridBrainStructure:
#     """
#     Hybrid brain structure with blocks for fields and sparse for brain data.
#     Full complexity calculations with efficient storage.
#     """
    
#     def __init__(self, dimensions: Tuple[int, int, int] = GRID_DIMENSIONS):
#         """
#         Initialize hybrid brain structure.
        
#         Args:
#             dimensions: 3D dimensions for brain grid (x, y, z)
#         """
#         logger.info(f"Initializing hybrid brain structure with dimensions {dimensions}")
        
#         self.dimensions = dimensions
#         self.brain_id = str(uuid.uuid4())
#         self.creation_time = datetime.now().isoformat()
#         self.last_updated = self.creation_time
        
#         # === HYBRID STORAGE ARCHITECTURE ===
        
#         # 1. SPARSE BRAIN DATA (Active cells only - memory efficient)
#         self.active_cells = {}          # (x,y,z) -> cell_data
#         self.region_assignments = {}    # (x,y,z) -> region_name
#         self.subregion_assignments = {} # (x,y,z) -> subregion_name
        
#         # 2. BLOCK-BASED FIELD CALCULATIONS (Computational efficient)
#         self.field_blocks = {}          # block_id -> FieldBlock
#         self.block_size = self._calculate_optimal_block_size()
#         self.block_grid_dims = (
#             math.ceil(dimensions[0] / self.block_size[0]),
#             math.ceil(dimensions[1] / self.block_size[1]),
#             math.ceil(dimensions[2] / self.block_size[2])
#         )
        
#         # 3. STATIC FIELD PATTERNS (From architecture)
#         self.static_field_foundation = None  # NumPy array for calculations
#         self.standing_wave_patterns = {}     # Resonance chambers
#         self.field_anchors = {}              # EM foundation points
        
#         # === BRAIN ANATOMY ===
#         self.regions = {}               # Major brain regions
#         self.subregions = {}            # Sub-regions within major regions
#         self.region_boundaries = {}     # Region boundary definitions
        
#         # === COMPLEXITY SYSTEMS ===
#         self.phi_resonance_zones = {}   # Golden ratio resonance areas
#         self.edge_of_chaos_zones = {}   # Optimal processing zones
#         self.complexity_gradients = {} # Complexity field analysis
        
#         # === MYCELIAL INTERFACE ===
#         self.mycelial_interface = None  # Connection to mycelial network
#         self.energy_monitoring = {}     # Energy level tracking
#         self.seed_integration_data = {} # Brain seed connection info
        
#         # === STATUS FLAGS ===
#         self.regions_defined = False
#         self.subregions_defined = False
#         self.field_blocks_initialized = False
#         self.static_fields_created = False
#         self.standing_waves_calculated = False
#         self.complexity_analysis_complete = False
#         self.seed_integrated = False
        
#         logger.info(f"Hybrid brain structure initialized: {self.brain_id[:8]}")
    
#     def _calculate_optimal_block_size(self) -> Tuple[int, int, int]:
#         """Calculate optimal block size for field calculations."""
#         # Balance between computational efficiency and field resolution
#         # Smaller blocks = better field resolution, more computation
#         # Larger blocks = less computation, coarser field resolution
        
#         # Use architecture principle: blocks for sound boundaries
#         min_wavelength = SPEED_OF_SOUND / max(REGION_DEFAULT_FREQUENCIES.values())
        
#         # Block size should be fraction of minimum wavelength for proper sound
#         block_size_float = min_wavelength / 4.0  # Quarter wavelength resolution
        
#         # Convert to grid units and ensure reasonable size
#         grid_scale = 1.0  # Assume 1 grid unit = 1 meter (adjust as needed)
#         block_size = max(8, min(32, int(block_size_float / grid_scale)))
        
#         return (block_size, block_size, block_size)
    
#     # === REGION DEFINITION WITH FULL ANATOMY ===
    
#     def define_complete_brain_anatomy(self) -> bool:
#         """Define complete brain anatomy with regions and sub-regions."""
#         logger.info("Defining complete brain anatomy")
        
#         try:
#             # 1. Define major regions
#             self._define_major_regions()
            
#             # 2. Define sub-regions within major regions
#             self._define_subregions()
            
#             # 3. Create region boundaries
#             self._define_region_boundaries()
            
#             # 4. Initialize field blocks based on anatomy
#             self._initialize_field_blocks()
            
#             self.regions_defined = True
#             self.subregions_defined = True
#             self.last_updated = datetime.now().isoformat()
            
#             logger.info("Complete brain anatomy defined successfully")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to define brain anatomy: {e}")
#             return False
    
#     def _define_major_regions(self):
#         """Define major brain regions with proper anatomical structure."""
#         self.regions = {}
        
#         for region_name, proportion in REGION_PROPORTIONS.items():
#             location = REGION_LOCATIONS.get(region_name, (0.5, 0.5, 0.5))
#             frequency = REGION_DEFAULT_FREQUENCIES.get(region_name, SCHUMANN_FREQUENCY)
            
#             # Convert normalized location to grid coordinates
#             center_x = int(location[0] * self.dimensions[0])
#             center_y = int(location[1] * self.dimensions[1])
#             center_z = int(location[2] * self.dimensions[2])
            
#             # Calculate region volume and approximate size
#             total_volume = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
#             region_volume = int(proportion * total_volume)
            
#             # Approximate region as ellipsoid
#             region_radius = (region_volume * 3 / (4 * np.pi)) ** (1/3)
            
#             self.regions[region_name] = {
#                 'name': region_name,
#                 'center': (center_x, center_y, center_z),
#                 'proportion': proportion,
#                 'volume': region_volume,
#                 'radius': region_radius,
#                 'frequency': frequency,
#                 'subregions': [],  # Will be populated later
#                 'complexity_level': self._calculate_region_complexity(region_name),
#                 'field_strength': self._calculate_region_field_strength(region_name)
#             }
        
#         logger.info(f"Defined {len(self.regions)} major brain regions")
    
#     def _define_subregions(self):
#         """Define sub-regions within major regions."""
#         self.subregions = {}
        
#         # Anatomical sub-regions for each major region
#         subregion_definitions = {
#             REGION_FRONTAL: [
#                 'prefrontal_cortex', 'orbitofrontal_cortex', 'motor_cortex', 
#                 'premotor_cortex', 'supplementary_motor_area', 'broca_area'
#             ],
#             REGION_PARIETAL: [
#                 'primary_somatosensory', 'secondary_somatosensory', 'superior_parietal',
#                 'inferior_parietal', 'precuneus', 'postcentral_gyrus'
#             ],
#             REGION_TEMPORAL: [
#                 'primary_auditory', 'secondary_auditory', 'wernicke_area',
#                 'superior_temporal', 'middle_temporal', 'inferior_temporal',
#                 'hippocampus', 'amygdala', 'entorhinal_cortex'
#             ],
#             REGION_OCCIPITAL: [
#                 'primary_visual', 'secondary_visual', 'visual_association',
#                 'cuneus', 'lingual_gyrus', 'fusiform_gyrus'
#             ],
#             REGION_LIMBIC: [
#                 'anterior_cingulate', 'posterior_cingulate', 'parahippocampal',
#                 'fornix', 'mammillary_bodies', 'septal_nuclei'
#             ],
#             REGION_BRAIN_STEM: [
#                 'midbrain', 'pons', 'medulla_oblongata',
#                 'reticular_formation', 'cranial_nerve_nuclei'
#             ],
#             REGION_CEREBELLUM: [
#                 'anterior_lobe', 'posterior_lobe', 'flocculonodular_lobe',
#                 'deep_cerebellar_nuclei', 'cerebellar_cortex'
#             ]
#         }
        
#         for region_name, subregion_list in subregion_definitions.items():
#             if region_name in self.regions:
#                 region = self.regions[region_name]
                
#                 for i, subregion_name in enumerate(subregion_list):
#                     # Distribute sub-regions around parent region center
#                     angle = 2 * np.pi * i / len(subregion_list)
#                     offset_radius = region['radius'] * 0.6  # 60% of region radius
                    
#                     offset_x = int(offset_radius * np.cos(angle))
#                     offset_y = int(offset_radius * np.sin(angle))
#                     offset_z = int(offset_radius * np.sin(angle * 0.5))  # Vary z less
                    
#                     center_x = region['center'][0] + offset_x
#                     center_y = region['center'][1] + offset_y
#                     center_z = region['center'][2] + offset_z
                    
#                     # Keep within brain bounds
#                     center_x = max(0, min(center_x, self.dimensions[0] - 1))
#                     center_y = max(0, min(center_y, self.dimensions[1] - 1))
#                     center_z = max(0, min(center_z, self.dimensions[2] - 1))
                    
#                     subregion_id = f"{region_name}_{subregion_name}"
                    
#                     self.subregions[subregion_id] = {
#                         'name': subregion_name,
#                         'full_name': subregion_id,
#                         'parent_region': region_name,
#                         'center': (center_x, center_y, center_z),
#                         'radius': region['radius'] / len(subregion_list),
#                         'frequency': region['frequency'] * (1 + 0.1 * (i - len(subregion_list)/2)),
#                         'specialization': self._get_subregion_specialization(subregion_name)
#                     }
                
#                 # Update parent region with subregion list
#                 self.regions[region_name]['subregions'] = [
#                     f"{region_name}_{sr}" for sr in subregion_list
#                 ]
        
#         logger.info(f"Defined {len(self.subregions)} sub-regions")
    
#     def _get_subregion_specialization(self, subregion_name: str) -> str:
#         """Get functional specialization for sub-region."""
#         specializations = {
#             'prefrontal_cortex': 'executive_control',
#             'motor_cortex': 'movement_control',
#             'primary_somatosensory': 'touch_processing',
#             'primary_auditory': 'sound_processing',
#             'primary_visual': 'visual_processing',
#             'hippocampus': 'memory_formation',
#             'amygdala': 'emotional_processing',
#             'wernicke_area': 'language_comprehension',
#             'broca_area': 'speech_production',
#             'anterior_cingulate': 'emotion_regulation',
#             'midbrain': 'reflexes_arousal',
#             'pons': 'breathing_sleep',
#             'medulla_oblongata': 'vital_functions',
#             'anterior_lobe': 'motor_learning',
#             'posterior_lobe': 'coordination'
#         }
#         return specializations.get(subregion_name, 'general_processing')
    
#     def _calculate_region_complexity(self, region_name: str) -> float:
#         """Calculate complexity level for brain region."""
#         complexity_levels = {
#             REGION_FRONTAL: 0.9,      # Highest complexity
#             REGION_TEMPORAL: 0.8,     # High complexity (language, memory)
#             REGION_PARIETAL: 0.7,     # High complexity (integration)
#             REGION_LIMBIC: 0.8,       # High complexity (emotion, memory)
#             REGION_OCCIPITAL: 0.6,    # Moderate complexity (visual)
#             REGION_CEREBELLUM: 0.7,   # High complexity (motor learning)
#             REGION_BRAIN_STEM: 0.4    # Lower complexity (vital functions)
#         }
#         return complexity_levels.get(region_name, 0.5)
    
#     def _calculate_region_field_strength(self, region_name: str) -> float:
#         """Calculate static field strength for brain region."""
#         field_strengths = {
#             REGION_BRAIN_STEM: 1.0,   # Strongest (core functions)
#             REGION_LIMBIC: 0.9,       # Very strong (soul connection)
#             REGION_FRONTAL: 0.8,      # Strong (executive)
#             REGION_TEMPORAL: 0.7,     # Strong (memory/language)
#             REGION_PARIETAL: 0.6,     # Moderate (integration)
#             REGION_CEREBELLUM: 0.6,   # Moderate (motor)
#             REGION_OCCIPITAL: 0.5     # Moderate (visual)
#         }
#         return field_strengths.get(region_name, 0.5)
    
#     def _define_region_boundaries(self):
#         """Define boundaries between brain regions."""
#         self.region_boundaries = {}
        
#         for region1_name in self.regions:
#             for region2_name in self.regions:
#                 if region1_name < region2_name:  # Avoid duplicates
#                     boundary_key = (region1_name, region2_name)
                    
#                     # Get boundary type from constants or determine based on regions
#                     boundary_type = REGION_BOUNDARIES.get(
#                         boundary_key,
#                         REGION_BOUNDARIES.get(
#                             (region2_name, region1_name),
#                             BOUNDARY_TYPE_GRADUAL
#                         )
#                     )
                    
#                     # Get boundary parameters
#                     params = BOUNDARY_PARAMETERS.get(boundary_type, 
#                                                    BOUNDARY_PARAMETERS[BOUNDARY_TYPE_GRADUAL])
                    
#                     self.region_boundaries[boundary_key] = {
#                         'type': boundary_type,
#                         'regions': boundary_key,
#                         'transition_width': params['transition_width'],
#                         'permeability': params['permeability'],
#                         'sound_frequency': self._calculate_boundary_sound_frequency(boundary_type)
#                     }
        
#         logger.info(f"Defined {len(self.region_boundaries)} region boundaries")
    
#     def _calculate_boundary_sound_frequency(self, boundary_type: str) -> float:
#         """Calculate sound frequency for boundary type."""
#         frequencies = {
#             BOUNDARY_TYPE_SHARP: SOUND_SHARP_FREQUENCY,
#             BOUNDARY_TYPE_GRADUAL: SOUND_GRADUAL_FREQUENCY,
#             BOUNDARY_TYPE_PERMEABLE: SOUND_PERMEABLE_FREQUENCY
#         }
#         base_freq = frequencies.get(boundary_type, SOUND_GRADUAL_FREQUENCY)
#         return base_freq + random.uniform(-SOUND_FREQUENCY_VARIATION, SOUND_FREQUENCY_VARIATION)
    
#     # === FIELD BLOCK INITIALIZATION ===
    
#     def _initialize_field_blocks(self):
#         """Initialize field blocks for computational efficiency."""
#         logger.info("Initializing field blocks for static field calculations")
        
#         self.field_blocks = {}
#         block_count = 0
        
#         for bx in range(self.block_grid_dims[0]):
#             for by in range(self.block_grid_dims[1]):
#                 for bz in range(self.block_grid_dims[2]):
#                     # Calculate block boundaries
#                     start_x = bx * self.block_size[0]
#                     start_y = by * self.block_size[1]
#                     start_z = bz * self.block_size[2]
                    
#                     end_x = min(start_x + self.block_size[0], self.dimensions[0])
#                     end_y = min(start_y + self.block_size[1], self.dimensions[1])
#                     end_z = min(start_z + self.block_size[2], self.dimensions[2])
                    
#                     block_id = f"FB_{bx}_{by}_{bz}"
                    
#                     # Determine which region(s) this block overlaps
#                     overlapping_regions = self._find_overlapping_regions(
#                         (start_x, start_y, start_z), (end_x, end_y, end_z)
#                     )
                    
#                     # Create field block
#                     field_block = FieldBlock(
#                         block_id=block_id,
#                         position=(start_x, start_y, start_z),
#                         size=(end_x - start_x, end_y - start_y, end_z - start_z),
#                         overlapping_regions=overlapping_regions,
#                         block_size=self.block_size
#                     )
                    
#                     self.field_blocks[block_id] = field_block
#                     block_count += 1
        
#         self.field_blocks_initialized = True
#         logger.info(f"Initialized {block_count} field blocks")
    
#     def _find_overlapping_regions(self, start_pos: Tuple[int, int, int], 
#                                  end_pos: Tuple[int, int, int]) -> List[str]:
#         """Find regions that overlap with block boundaries."""
#         overlapping = []
        
#         block_center = (
#             (start_pos[0] + end_pos[0]) / 2,
#             (start_pos[1] + end_pos[1]) / 2,
#             (start_pos[2] + end_pos[2]) / 2
#         )
        
#         for region_name, region in self.regions.items():
#             region_center = region['center']
#             distance = math.sqrt(
#                 (block_center[0] - region_center[0])**2 +
#                 (block_center[1] - region_center[1])**2 +
#                 (block_center[2] - region_center[2])**2
#             )
            
#             # Block overlaps if within region radius + block diagonal
#             block_diagonal = math.sqrt(sum(s**2 for s in self.block_size))
#             if distance <= region['radius'] + block_diagonal:
#                 overlapping.append(region_name)
        
#         return overlapping
    
#     # === STATIC FIELD PATTERNS ===
    
#     def create_static_field_foundation(self) -> bool:
#         """Create static electromagnetic field foundation."""
#         logger.info("Creating static field foundation")
        
#         try:
#             # Create base electromagnetic pattern (from architecture)
#             self.static_field_foundation = np.zeros(self.dimensions)
            
#             # EM foundation pattern - every 8th cell (like DPI grid)
#             spacing = 8
#             for x in range(0, self.dimensions[0], spacing):
#                 for y in range(0, self.dimensions[1], spacing):
#                     for z in range(0, self.dimensions[2], spacing):
#                         if (x < self.dimensions[0] and 
#                             y < self.dimensions[1] and 
#                             z < self.dimensions[2]):
                            
#                             # Field strength based on nearby regions
#                             field_strength = self._calculate_field_strength_at_position((x, y, z))
#                             self.static_field_foundation[x, y, z] = field_strength
            
#             # Create field anchor points
#             self._create_field_anchors()
            
#             self.static_fields_created = True
#             logger.info("Static field foundation created")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to create static field foundation: {e}")
#             return False
    
#     def _calculate_field_strength_at_position(self, position: Tuple[int, int, int]) -> float:
#         """Calculate static field strength at given position."""
#         x, y, z = position
#         total_strength = 0.0
        
#         # Field strength influenced by nearby regions
#         for region_name, region in self.regions.items():
#             rx, ry, rz = region['center']
#             distance = math.sqrt((x - rx)**2 + (y - ry)**2 + (z - rz)**2)
            
#             # Field strength decreases with distance
#             if distance < region['radius']:
#                 strength = region['field_strength'] * (1.0 - distance / region['radius'])
#                 total_strength += strength
        
#         return min(1.0, total_strength)  # Cap at 1.0
    
#     def _create_field_anchors(self):
#         """Create field anchor points for resonance."""
#         self.field_anchors = {}
        
#         # Create anchors at strong field points
#         anchor_count = 0
#         threshold = 0.7  # Only strong field points become anchors
        
#         # Find strong field points in foundation
#         strong_points = np.where(self.static_field_foundation >= threshold)
        
#         for i in range(len(strong_points[0])):
#             x, y, z = strong_points[0][i], strong_points[1][i], strong_points[2][i]
#             anchor_id = f"FA_{x}_{y}_{z}"
            
#             self.field_anchors[anchor_id] = {
#                 'position': (x, y, z),
#                 'strength': float(self.static_field_foundation[x, y, z]),
#                 'frequency': self._calculate_anchor_frequency((x, y, z)),
#                 'resonance_radius': 10  # Grid units
#             }
#             anchor_count += 1
        
#         logger.info(f"Created {anchor_count} field anchors")
    
#     def _calculate_anchor_frequency(self, position: Tuple[int, int, int]) -> float:
#         """Calculate resonance frequency for field anchor."""
#         # Base frequency from nearest region
#         nearest_region = self._find_nearest_region(position)
#         if nearest_region:
#             base_freq = self.regions[nearest_region]['frequency']
#         else:
#             base_freq = SCHUMANN_FREQUENCY
        
#         # Add position-based variation
#         x, y, z = position
#         variation = (x + y + z) % 100 / 100.0 * 2.0  # ±1 Hz variation
        
#         return base_freq + variation - 1.0
    
#     def _find_nearest_region(self, position: Tuple[int, int, int]) -> Optional[str]:
#         """Find nearest region to given position."""
#         x, y, z = position
#         min_distance = float('inf')
#         nearest = None
        
#         for region_name, region in self.regions.items():
#             rx, ry, rz = region['center']
#             distance = math.sqrt((x - rx)**2 + (y - ry)**2 + (z - rz)**2)
            
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest = region_name
        
#         return nearest
    
#     # === STANDING WAVE CALCULATIONS ===
    
#     def calculate_standing_wave_patterns(self) -> bool:
#         """Calculate standing wave resonance patterns."""
#         logger.info("Calculating standing wave resonance patterns")
        
#         try:
#             self.standing_wave_patterns = {}
            
#             # Create standing waves for each region frequency
#             for region_name, region in self.regions.items():
#                 frequency = region['frequency']
#                 wavelength = SPEED_OF_SOUND / frequency
                
#                 # Calculate wave pattern for this frequency
#                 wave_pattern = self._calculate_wave_pattern(frequency, wavelength, region)
                
#                 self.standing_wave_patterns[f"{region_name}_wave"] = wave_pattern
            
#             # Create global harmonics using phi ratios
#             self._calculate_phi_harmonic_patterns()
            
#             self.standing_waves_calculated = True
#             logger.info(f"Calculated {len(self.standing_wave_patterns)} standing wave patterns")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to calculate standing waves: {e}")
#             return False
    
#     def _calculate_wave_pattern(self, frequency: float, wavelength: float, 
#                                region: Dict[str, Any]) -> Dict[str, Any]:
#         """Calculate standing wave pattern for frequency."""
#         # Node spacing (zero displacement points)
#         node_spacing = wavelength / 2.0
        
#         # Find nodes and antinodes within region
#         nodes = []
#         antinodes = []
        
#         center = region['center']
#         radius = region['radius']
        
#         # Calculate nodes in region
#         for x in range(int(center[0] - radius), int(center[0] + radius)):
#             for y in range(int(center[1] - radius), int(center[1] + radius)):
#                 for z in range(int(center[2] - radius), int(center[2] + radius)):
#                     if (0 <= x < self.dimensions[0] and
#                         0 <= y < self.dimensions[1] and
#                         0 <= z < self.dimensions[2]):
                        
#                         # Distance from region center
#                         distance = math.sqrt(
#                             (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
#                         )
                        
#                         if distance <= radius:
#                             # Check if this position is a node or antinode
#                             phase = (distance % wavelength) / wavelength * 2 * np.pi
                            
#                             if abs(np.sin(phase)) < 0.1:  # Near zero = node
#                                 nodes.append((x, y, z))
#                             elif abs(np.sin(phase)) > 0.9:  # Near ±1 = antinode
#                                 antinodes.append((x, y, z))
        
#         return {
#             'frequency': frequency,
#             'wavelength': wavelength,
#             'region': region['name'],
#             'nodes': nodes,
#             'antinodes': antinodes,
#             'amplitude': 0.5,  # Base amplitude
#             'phase': 0.0
#         }
    
#     def _calculate_phi_harmonic_patterns(self):
#         """Calculate harmonic patterns using golden ratio."""
#         logger.info("Calculating phi-based harmonic patterns")
        
#         # Create harmonics based on phi ratios
#         base_frequencies = [f for f in REGION_DEFAULT_FREQUENCIES.values()]
        
#         for base_freq in base_frequencies:
#             # Generate phi harmonics
#             phi_harmonics = []
#             current_freq = base_freq
            
#             for i in range(5):  # 5 harmonic levels
#                 phi_harmonics.append(current_freq)
#                 current_freq *= PHI  # Golden ratio multiplication
                
#                 if current_freq > 100:  # Reasonable upper limit
#                     break
            
#             # Create resonance pattern for phi harmonics
#             harmonic_pattern = {
#                 'base_frequency': base_freq,
#                 'phi_harmonics': phi_harmonics,
#                 'resonance_strength': 0.8,
#                 'pattern_type': 'phi_spiral'
#             }
            
#             pattern_id = f"phi_harmonic_{base_freq:.1f}hz"
#             self.standing_wave_patterns[pattern_id] = harmonic_pattern
    
#     # === EDGE OF CHAOS DETECTION ===
    
#     def detect_edge_of_chaos_zones(self) -> bool:
#         """Detect edge of chaos zones for optimal processing."""
#         logger.info("Detecting edge of chaos zones")
        
#         try:
#             self.edge_of_chaos_zones = {}
            
#             # Calculate complexity gradients across brain
#             complexity_field = self._calculate_complexity_field()
            
#             # Find edge of chaos zones (complexity gradient ≈ golden ratio)
#             edge_zones = self._find_edge_of_chaos_regions(complexity_field)
            
#             self.edge_of_chaos_zones = edge_zones
            
#             logger.info(f"Detected {len(edge_zones)} edge of chaos zones")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to detect edge of chaos zones: {e}")
#             return False
    
#     def _calculate_complexity_field(self) -> np.ndarray:
#         """Calculate complexity field across brain structure."""
#         # Create complexity field based on region complexity and field patterns
#         complexity_field = np.zeros(self.dimensions)
        
#         # Base complexity from regions
#         for region_name, region in self.regions.items():
#             center = region['center']
#             radius = region['radius']
#             complexity = region['complexity_level']
            
#             # Create complexity gradient around region center
#             for x in range(max(0, int(center[0] - radius)), 
#                           min(self.dimensions[0], int(center[0] + radius))):
#                 for y in range(max(0, int(center[1] - radius)), 
#                               min(self.dimensions[1], int(center[1] + radius))):
#                     for z in range(max(0, int(center[2] - radius)), 
#                                   min(self.dimensions[2], int(center[2] + radius))):
                        
#                         distance = math.sqrt(
#                             (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
#                         )
                        
#                         if distance <= radius:
#                             # Complexity decreases with distance from center
#                             falloff = 1.0 - (distance / radius)
#                             local_complexity = complexity * falloff
                            
#                             # Add to existing complexity (regions can overlap)
#                             complexity_field[x, y, z] += local_complexity
        
#         # Add noise and variations for edge of chaos detection
#         noise = np.random.normal(0, 0.1, self.dimensions)
#         complexity_field += noise
        
#         # Normalize to [0, 1] range
#         complexity_field = np.clip(complexity_field, 0, 1)
        
#         return complexity_field
    
#     def _find_edge_of_chaos_regions(self, complexity_field: np.ndarray) -> Dict[str, Any]:
#         """Find regions at edge of chaos (optimal complexity)."""
#         edge_zones = {}
        
#         # Calculate gradients to find edge of chaos
#         grad_x = np.gradient(complexity_field, axis=0)
#         grad_y = np.gradient(complexity_field, axis=1)
#         grad_z = np.gradient(complexity_field, axis=2)
        
#         # Gradient magnitude
#         gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
#         # Edge of chaos occurs where gradient ≈ golden ratio reciprocal
#         target_gradient = EDGE_OF_CHAOS_RATIO  # 1/phi ≈ 0.618
#         tolerance = 0.1
        
#         # Find positions near edge of chaos
#         edge_mask = np.abs(gradient_magnitude - target_gradient) < tolerance
#         edge_positions = np.where(edge_mask)
        
#         zone_count = 0
#         for i in range(len(edge_positions[0])):
#             x, y, z = edge_positions[0][i], edge_positions[1][i], edge_positions[2][i]
            
#             zone_id = f"edge_zone_{zone_count}"
#             edge_zones[zone_id] = {
#                 'position': (x, y, z),
#                 'complexity': float(complexity_field[x, y, z]),
#                 'gradient_magnitude': float(gradient_magnitude[x, y, z]),
#                 'optimization_potential': float(1.0 - abs(gradient_magnitude[x, y, z] - target_gradient)),
#                 'region': self._find_nearest_region((x, y, z))
#             }
#             zone_count += 1
            
#             # Limit number of zones to prevent memory issues
#             if zone_count >= 1000:
#                 break
        
#         return edge_zones
    
#     # === PHI RESONANCE CALCULATIONS ===
    
#     def calculate_phi_resonance_zones(self) -> bool:
#         """Calculate phi-based resonance zones."""
#         logger.info("Calculating phi resonance zones")
        
#         try:
#             self.phi_resonance_zones = {}
            
#             # Create phi resonance around field anchors
#             for anchor_id, anchor in self.field_anchors.items():
#                 position = anchor['position']
#                 base_freq = anchor['frequency']
                
#                 # Generate phi harmonic series
#                 phi_frequencies = []
#                 current_freq = base_freq
                
#                 for i in range(8):  # Create 8 harmonics
#                     phi_frequencies.append(current_freq)
#                     current_freq *= PHI
                    
#                     if current_freq > 1000:  # Reasonable upper limit
#                         break
                
#                 # Create resonance zone
#                 zone_id = f"phi_zone_{anchor_id}"
#                 self.phi_resonance_zones[zone_id] = {
#                     'anchor_position': position,
#                     'base_frequency': base_freq,
#                     'phi_harmonics': phi_frequencies,
#                     'resonance_radius': anchor['resonance_radius'],
#                     'resonance_strength': anchor['strength'],
#                     'fibonacci_pattern': self._generate_fibonacci_pattern(position)
#                 }
            
#             logger.info(f"Calculated {len(self.phi_resonance_zones)} phi resonance zones")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to calculate phi resonance zones: {e}")
#             return False
    
#     def _generate_fibonacci_pattern(self, center: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
#         """Generate fibonacci spiral pattern around center point."""
#         pattern_points = []
#         x0, y0, z0 = center
        
#         # Generate fibonacci spiral in 3D
#         fibonacci_seq = [1, 1, 2, 3, 5, 8, 13, 21]
        
#         for i, fib_num in enumerate(fibonacci_seq):
#             angle = i * PHI * 2 * np.pi  # Golden angle
#             radius = fib_num * 2  # Scale by fibonacci number
            
#             # 3D spiral coordinates
#             x = int(x0 + radius * np.cos(angle))
#             y = int(y0 + radius * np.sin(angle))
#             z = int(z0 + fib_num * np.cos(angle * 0.5))  # Vertical component
            
#             # Keep within bounds
#             if (0 <= x < self.dimensions[0] and
#                 0 <= y < self.dimensions[1] and
#                 0 <= z < self.dimensions[2]):
#                 pattern_points.append((x, y, z))
        
#         return pattern_points
    
#     # === MYCELIAL INTERFACE ===
    
#     def create_mycelial_interface(self):
#         """Create interface for mycelial network connection."""
#         logger.info("Creating mycelial network interface")
        
#         class BrainMycelialInterface:
#             def __init__(self, brain_structure):
#                 self.brain = brain_structure
#                 self.energy_storage = {}
#                 self.monitoring_active = True
                
#             def receive_energy_from_seed(self, energy_amount, frequency, position, region):
#                 """Receive energy from brain seed."""
#                 try:
#                     # Store energy in brain structure
#                     self.brain.set_field_value(position, 'energy', energy_amount)
#                     self.brain.set_field_value(position, 'frequency', frequency)
                    
#                     # Update energy monitoring
#                     self.energy_storage[position] = {
#                         'energy': energy_amount,
#                         'frequency': frequency,
#                         'region': region,
#                         'timestamp': datetime.now().isoformat()
#                     }
                    
#                     # Trigger energy distribution through field blocks
#                     self.brain._distribute_seed_energy(position, energy_amount, frequency)
                    
#                     logger.info(f"Mycelial interface received {energy_amount} BEU at {frequency} Hz")
#                     return {'success': True, 'energy_stored': energy_amount}
                    
#                 except Exception as e:
#                     logger.error(f"Mycelial interface error: {e}")
#                     return {'success': False, 'error': str(e)}
            
#             def monitor_energy_levels(self):
#                 """Monitor energy levels across brain."""
#                 total_energy = sum(data['energy'] for data in self.energy_storage.values())
#                 return {
#                     'total_energy': total_energy,
#                     'energy_points': len(self.energy_storage),
#                     'monitoring_active': self.monitoring_active
#                 }
            
#             def get_brain_state(self):
#                 """Get current brain state for mycelial processing."""
#                 return {
#                     'regions_active': len([r for r in self.brain.regions if self.brain.regions[r]]),
#                     'field_blocks_active': len(self.brain.field_blocks),
#                     'complexity_zones': len(self.brain.edge_of_chaos_zones),
#                     'phi_zones': len(self.brain.phi_resonance_zones),
#                     'standing_waves': len(self.brain.standing_wave_patterns)
#                 }
        
#         self.mycelial_interface = BrainMycelialInterface(self)
#         return self.mycelial_interface
    
#     def _distribute_seed_energy(self, position: Tuple[int, int, int], 
#                                energy: float, frequency: float):
#         """Distribute seed energy through field blocks."""
#         x, y, z = position
        
#         # Find which field block contains this position
#         block_x = x // self.block_size[0]
#         block_y = y // self.block_size[1]
#         block_z = z // self.block_size[2]
        
#         block_id = f"FB_{block_x}_{block_y}_{block_z}"
        
#         if block_id in self.field_blocks:
#             field_block = self.field_blocks[block_id]
#             field_block.receive_energy(position, energy, frequency)
            
#             # Propagate to neighboring blocks
#             self._propagate_energy_to_neighbors(block_id, energy * 0.1, frequency)
    
#     def _propagate_energy_to_neighbors(self, source_block_id: str, energy: float, frequency: float):
#         """Propagate energy to neighboring field blocks."""
#         # Parse source block coordinates
#         parts = source_block_id.split('_')
#         bx, by, bz = int(parts[1]), int(parts[2]), int(parts[3])
        
#         # Propagate to 6-connected neighbors
#         for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
#             nx, ny, nz = bx + dx, by + dy, bz + dz
            
#             # Check bounds
#             if (0 <= nx < self.block_grid_dims[0] and
#                 0 <= ny < self.block_grid_dims[1] and
#                 0 <= nz < self.block_grid_dims[2]):
                
#                 neighbor_id = f"FB_{nx}_{ny}_{nz}"
#                 if neighbor_id in self.field_blocks:
#                     neighbor_block = self.field_blocks[neighbor_id]
#                     # Propagate reduced energy to center of neighbor block
#                     center_pos = (
#                         nx * self.block_size[0] + self.block_size[0] // 2,
#                         ny * self.block_size[1] + self.block_size[1] // 2,
#                         nz * self.block_size[2] + self.block_size[2] // 2
#                     )
#                     neighbor_block.receive_energy(center_pos, energy, frequency)
    
#     # === SPARSE BRAIN DATA MANAGEMENT ===
    
#     def set_field_value(self, position: Tuple[int, int, int], field_type: str, value: float) -> bool:
#         """Set field value at position (sparse storage)."""
#         x, y, z = position
        
#         # Validate position
#         if not (0 <= x < self.dimensions[0] and
#                 0 <= y < self.dimensions[1] and
#                 0 <= z < self.dimensions[2]):
#             return False
        
#         # Ensure cell exists in active cells
#         if position not in self.active_cells:
#             self.active_cells[position] = {}
        
#         # Set field value
#         self.active_cells[position][field_type] = value
        
#         # Auto-assign region if not set
#         if position not in self.region_assignments:
#             nearest_region = self._find_nearest_region(position)
#             if nearest_region:
#                 self.region_assignments[position] = nearest_region
        
#         return True
    
#     def get_field_value(self, position: Tuple[int, int, int], field_type: str) -> float:
#         """Get field value at position."""
#         if position in self.active_cells:
#             return self.active_cells[position].get(field_type, 0.0)
#         return 0.0
    
#     def get_active_cell_count(self) -> int:
#         """Get number of active cells."""
#         return len(self.active_cells)
    
#     def get_region_at_position(self, position: Tuple[int, int, int]) -> Optional[str]:
#         """Get region at position."""
#         return self.region_assignments.get(position)
    
#     def get_subregion_at_position(self, position: Tuple[int, int, int]) -> Optional[str]:
#         """Get subregion at position."""
#         return self.subregion_assignments.get(position)
    
#     # === BRAIN SEED INTEGRATION SUPPORT ===
    
#     def notify_seed_integration(self, seed_id: str, position: Tuple[int, int, int],
#                                energy_transferred: float, frequency: float) -> Dict[str, Any]:
#         """Notification that brain seed has been integrated."""
#         logger.info(f"Brain seed {seed_id[:8]} integrated at {position}")
        
#         self.seed_integration_data = {
#             'seed_id': seed_id,
#             'position': position,
#             'energy_transferred': energy_transferred,
#             'frequency': frequency,
#             'integration_time': datetime.now().isoformat()
#         }
        
#         self.seed_integrated = True
        
#         # Initialize energy monitoring at seed position
#         self.energy_monitoring[position] = {
#             'initial_energy': energy_transferred,
#             'current_energy': energy_transferred,
#             'frequency': frequency,
#             'monitoring_start': datetime.now().isoformat()
#         }
        
#         return {
#             'success': True,
#             'seed_position_set': True,
#             'energy_monitoring_started': True,
#             'mycelial_interface_ready': self.mycelial_interface is not None
#         }
    
#     def prepare_for_seed_integration(self, seed_position: Tuple[int, int, int] = None) -> Dict[str, Any]:
#         """Prepare brain structure for seed integration."""
#         logger.info("Preparing brain structure for seed integration")
        
#         preparation_status = {
#             'success': True,
#             'preparations_completed': [],
#             'errors': []
#         }
        
#         # Ensure all systems are initialized
#         if not self.regions_defined:
#             if self.define_complete_brain_anatomy():
#                 preparation_status['preparations_completed'].append('brain_anatomy')
#             else:
#                 preparation_status['success'] = False
#                 preparation_status['errors'].append('Failed to define brain anatomy')
        
#         if not self.static_fields_created:
#             if self.create_static_field_foundation():
#                 preparation_status['preparations_completed'].append('static_fields')
#             else:
#                 preparation_status['success'] = False
#                 preparation_status['errors'].append('Failed to create static fields')
        
#         if not self.standing_waves_calculated:
#             if self.calculate_standing_wave_patterns():
#                 preparation_status['preparations_completed'].append('standing_waves')
#             else:
#                 preparation_status['success'] = False
#                 preparation_status['errors'].append('Failed to calculate standing waves')
        
#         # Initialize complex systems
#         if not self.edge_of_chaos_zones:
#             if self.detect_edge_of_chaos_zones():
#                 preparation_status['preparations_completed'].append('edge_of_chaos_zones')
        
#         if not self.phi_resonance_zones:
#             if self.calculate_phi_resonance_zones():
#                 preparation_status['preparations_completed'].append('phi_resonance_zones')
        
#         # Create mycelial interface if not exists
#         if not self.mycelial_interface:
#             self.create_mycelial_interface()
#             preparation_status['preparations_completed'].append('mycelial_interface')
        
#         # Find optimal seed position if not provided
#         if seed_position is None:
#             seed_position = self.find_optimal_seed_position()
        
#         preparation_status['optimal_seed_position'] = seed_position
#         preparation_status['optimal_region'] = self._find_nearest_region(seed_position)
        
#         return preparation_status
    
#     def find_optimal_seed_position(self) -> Tuple[int, int, int]:
#         """Find optimal position for brain seed (limbic region)."""
#         # Use limbic region center as optimal position
#         if REGION_LIMBIC in self.regions:
#             return self.regions[REGION_LIMBIC]['center']
        
#         # Fallback to brain center
#         return (
#             self.dimensions[0] // 2,
#             self.dimensions[1] // 2,
#             self.dimensions[2] // 2
#         )
    
#     def analyze_brain_state(self) -> str:
#         """Analyze current brain state based on complexity and energy."""
#         try:
#             # Calculate metrics from brain structure data
#             total_energy = sum(cell_data.get('energy', 0) for cell_data in self.active_cells.values())
#             avg_energy = total_energy / len(self.active_cells) if self.active_cells else 0
            
#             # Count edge of chaos zones
#             chaos_count = len(self.edge_of_chaos_zones)
            
#             # Calculate complexity
#             complexity_ratio = chaos_count / 1000.0  # Normalize
            
#             # Determine state based on energy and complexity
#             if avg_energy < 0.2:
#                 return 'dormant'
#             elif avg_energy < 0.4 and complexity_ratio < 0.3:
#                 return 'formation'
#             elif complexity_ratio > 0.7:
#                 return 'aware_processing'
#             elif avg_energy > 0.6 and complexity_ratio > 0.4:
#                 return 'soul_attached_settling'
#             else:
#                 return 'formation'
                
#         except Exception as e:
#             logger.error(f"Error analyzing brain state: {e}")
#             return 'dormant'

#     def get_active_regions(self) -> set[str]:
#         """Get currently active regions based on energy levels."""
#         active_regions = set()
        
#         try:
#             for position, cell_data in self.active_cells.items():
#                 if cell_data.get('energy', 0) > 0.3:  # Active threshold
#                     region = self.region_assignments.get(position)
#                     if region:
#                         active_regions.add(region)
#         except Exception as e:
#             logger.error(f"Error getting active regions: {e}")
        
#         return active_regions
#     def calculate_processing_intensity(self) -> float:
#         """Calculate current processing intensity."""
#         try:
#             if not self.active_cells:
#                 return 0.0
            
#             # Average energy across active cells
#             energies = [cell_data.get('energy', 0) for cell_data in self.active_cells.values()]
#             return sum(energies) / len(energies)
            
#         except Exception as e:
#             logger.error(f"Error calculating processing intensity: {e}")
#             return 0.0

#     def update_field_dynamics(self, field_dynamics):
#         """Update field dynamics based on current brain state."""
#         try:
#             current_state = self.analyze_brain_state()
#             active_regions = self.get_active_regions()
#             processing_intensity = self.calculate_processing_intensity()
            
#             # Update field dynamics with current state
#             if hasattr(field_dynamics, 'update_fields_for_new_state'):
#                 field_dynamics.update_fields_for_new_state(
#                     new_brain_state=current_state,
#                     active_regions_now=active_regions,
#                     processing_intensity=processing_intensity
#                 )
            
#             return {
#                 'success': True,
#                 'state': current_state,
#                 'active_regions': list(active_regions),
#                 'intensity': processing_intensity
#             }
            
#         except Exception as e:
#             logger.error(f"Error updating field dynamics: {e}")
#             return {'success': False, 'error': str(e)}
    
#     # === STATUS AND METRICS ===
    
#     def get_brain_metrics(self) -> Dict[str, Any]:
#         """Get comprehensive brain structure metrics."""
#         return {
#             'brain_id': self.brain_id,
#             'creation_time': self.creation_time,
#             'last_updated': self.last_updated,
#             'dimensions': self.dimensions,
#             'structure': {
#                 'regions_count': len(self.regions),
#                 'subregions_count': len(self.subregions),
#                 'field_blocks_count': len(self.field_blocks),
#                 'boundaries_count': len(self.region_boundaries),
#                 'active_cells_count': len(self.active_cells)
#             },
#             'complexity': {
#                 'edge_of_chaos_zones': len(self.edge_of_chaos_zones),
#                 'phi_resonance_zones': len(self.phi_resonance_zones),
#                 'standing_wave_patterns': len(self.standing_wave_patterns),
#                 'field_anchors': len(self.field_anchors)
#             },
#             'integration': {
#                 'seed_integrated': self.seed_integrated,
#                 'mycelial_interface_active': self.mycelial_interface is not None,
#                 'energy_monitoring_points': len(self.energy_monitoring)
#             },
#             'status': {
#                 'regions_defined': self.regions_defined,
#                 'subregions_defined': self.subregions_defined,
#                 'field_blocks_initialized': self.field_blocks_initialized,
#                 'static_fields_created': self.static_fields_created,
#                 'standing_waves_calculated': self.standing_waves_calculated,
#                 'complexity_analysis_complete': bool(self.edge_of_chaos_zones)
#             }
#         }


# class FieldBlock:
#     """Field block for efficient field calculations."""
    
#     def __init__(self, block_id: str, position: Tuple[int, int, int], 
#                  size: Tuple[int, int, int], overlapping_regions: List[str],
#                  block_size: Tuple[int, int, int]):
#         self.block_id = block_id
#         self.position = position  # Start position
#         self.size = size          # Actual size
#         self.overlapping_regions = overlapping_regions
#         self.block_size = block_size  # Standard block size
        
#         # Field data
#         self.energy_total = 0.0
#         self.frequency_avg = 0.0
#         self.energy_points = {}  # (relative_x, relative_y, relative_z) -> energy
        
#         # Sound properties for boundaries
#         self.is_boundary_block = len(overlapping_regions) > 1
#         self.boundary_sound_freq = 0.0
        
#         if self.is_boundary_block:
#             self.boundary_sound_freq = self._calculate_boundary_frequency()
    
#     def _calculate_boundary_frequency(self) -> float:
#         """Calculate sound frequency for boundary block."""
#         if len(self.overlapping_regions) >= 2:
#             # Average of region frequencies
#             avg_freq = 10.0  # Default if no region data
#             # This would use actual region frequency data
#             return avg_freq
#         return 0.0
    
#     def receive_energy(self, position: Tuple[int, int, int], energy: float, frequency: float):
#         """Receive energy at specific position within block."""
#         # Convert to relative position within block
#         rel_x = position[0] - self.position[0]
#         rel_y = position[1] - self.position[1]
#         rel_z = position[2] - self.position[2]
        
#         # Store energy
#         self.energy_points[(rel_x, rel_y, rel_z)] = {
#             'energy': energy,
#             'frequency': frequency
#         }
        
#         # Update totals
#         self.energy_total += energy
        
#         # Update average frequency (weighted by energy)
#         if self.energy_total > 0:
#             total_freq_energy = sum(
#                 data['energy'] * data['frequency'] 
#                 for data in self.energy_points.values()
#             )
#             self.frequency_avg = total_freq_energy / self.energy_total


# # === UTILITY FUNCTIONS ===

# def create_hybrid_brain_structure(dimensions: Optional[Tuple[int, int, int]] = None,
#                                  initialize_all: bool = True) -> HybridBrainStructure:
#     """Create hybrid brain structure with optional full initialization."""
#     if dimensions is None:
#         dimensions = GRID_DIMENSIONS
    
#     logger.info(f"Creating hybrid brain structure with dimensions {dimensions}")
    
#     brain = HybridBrainStructure(dimensions)
    
#     if initialize_all:
#         logger.info("Performing full brain structure initialization")
        
#         # Define complete anatomy
#         brain.define_complete_brain_anatomy()
        
#         # Create static fields
#         brain.create_static_field_foundation()
        
#         # Calculate standing waves
#         brain.calculate_standing_wave_patterns()
        
#         # Detect complexity zones
#         brain.detect_edge_of_chaos_zones()
#         brain.calculate_phi_resonance_zones()
        
#         # Create mycelial interface
#         brain.create_mycelial_interface()
        
#         logger.info("Full brain structure initialization completed")
    
#     return brain


# def demonstrate_hybrid_brain_structure():
#     """Demonstrate the hybrid brain structure."""
#     print("\n=== Hybrid Brain Structure Demonstration ===")
    
#     try:
#         # Create brain structure
#         brain = create_hybrid_brain_structure(initialize_all=True)
        
#         print(f"Created brain: {brain.brain_id[:8]}")
        
#         # Get metrics
#         metrics = brain.get_brain_metrics()
#         print(f"Regions: {metrics['structure']['regions_count']}")
#         print(f"Sub-regions: {metrics['structure']['subregions_count']}")
#         print(f"Field blocks: {metrics['structure']['field_blocks_count']}")
#         print(f"Edge of chaos zones: {metrics['complexity']['edge_of_chaos_zones']}")
#         print(f"Phi resonance zones: {metrics['complexity']['phi_resonance_zones']}")
#         print(f"Standing wave patterns: {metrics['complexity']['standing_wave_patterns']}")
        
#         # Test seed preparation
#         prep_result = brain.prepare_for_seed_integration()
#         print(f"Seed preparation success: {prep_result['success']}")
#         print(f"Optimal seed position: {prep_result['optimal_seed_position']}")
        
#         return brain
        
#     except Exception as e:
#         print(f"ERROR: {e}")
#         return None


# # === MAIN EXECUTION ===
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     # Demonstrate hybrid brain structure
#     demo_brain = demonstrate_hybrid_brain_structure()
    
#     if demo_brain:
#         print("\nHybrid Brain Structure demonstration completed successfully!")
#         print("Full complexity maintained with efficient hybrid storage.")
#     else:
#         print("\nERROR: Hybrid Brain Structure demonstration failed")

# # --- End of brain_structure.py ---