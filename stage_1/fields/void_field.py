# --- START OF FILE src/stage_1/fields/void_field.py ---

"""
Void Field Module

Implements the dynamic Void field, the base reality containing patterned potential,
energy, and resonance phenomena. Incorporates noise, sacred geometry, and
platonic solid potential during initialization.
"""

import logging
import numpy as np
import random
from typing import Dict, Any, Tuple, Optional, List

# Assuming field_base is in the same directory
from .field_base import FieldBase
# Assuming constants are accessible via src path setup in main script/controller
from constants.constants import *

# --- Dependency Imports ---
# Noise Generator
try:
    # Assuming sound folder is accessible
    from sound.noise_generator import NoiseGenerator
    NOISE_GEN_AVAILABLE = True
except ImportError as e:
    logging.error(f"NoiseGenerator import failed: {e}. Void noise initialization will be basic.")
    NOISE_GEN_AVAILABLE = False
    NoiseGenerator = None # Define as None if not available

# Sacred Geometry Functions (Assume they exist in src/geometry/ and provide influence grids)
try:
    # Example imports - adjust based on actual function names/modules after refactor
    from geometry.flower_of_life import get_flower_of_life_influence # Needs implementation
    from geometry.seed_of_life import get_seed_of_life_influence   # Needs implementation
    from geometry.metatrons_cube import get_metatrons_cube_influence # Needs implementation
    # ... import other required geometry influence functions
    GEOMETRY_FUNCS_AVAILABLE = True
    # Map names to functions
    GEOMETRY_FUNCTION_MAP = {
        'flower_of_life': get_flower_of_life_influence,
        'seed_of_life': get_seed_of_life_influence,
        'metatrons_cube': get_metatrons_cube_influence,
        # Add other mappings here...
    }
except ImportError as e:
    logging.warning(f"Could not import geometry influence functions: {e}. Geometry potential disabled.")
    GEOMETRY_FUNCS_AVAILABLE = False
    GEOMETRY_FUNCTION_MAP = {}

# Platonic Solid Functions (Assume they exist in src/platonics/ and provide influence fields)
try:
    # Example imports - adjust based on actual function names/modules after refactor
    from platonics.tetrahedron import get_tetrahedron_influence # Needs implementation
    from platonics.hexahedron import get_hexahedron_influence # Needs implementation
    # ... import other required platonic influence functions
    PLATONIC_FUNCS_AVAILABLE = True
    # Map names to functions
    PLATONIC_FUNCTION_MAP = {
        'tetrahedron': get_tetrahedron_influence,
        'hexahedron': get_hexahedron_influence,
         # Add other mappings here...
    }
except ImportError as e:
    logging.warning(f"Could not import platonic influence functions: {e}. Platonic potential disabled.")
    PLATONIC_FUNCS_AVAILABLE = False
    PLATONIC_FUNCTION_MAP = {}


logger = logging.getLogger(__name__)

class VoidField(FieldBase):
    """
    Represents the dynamic Void field, the foundation of the simulated reality.
    It contains inherent patterns, energy fluctuations, and resonance potential.
    """

    def __init__(self, grid_size: Tuple[int, int, int] = GRID_SIZE):
        """
        Initialize the Void field.

        Args:
            grid_size (Tuple[int, int, int]): Dimensions of the field grid.
        """
        super().__init__(name="VoidField", grid_size=grid_size)

        # --- Initialize Void-Specific Properties ---
        self.base_energy: float = VOID_BASE_ENERGY
        self.base_frequency_range: Tuple[float, float] = VOID_BASE_FREQUENCY_RANGE
        self.base_coherence: float = VOID_BASE_COHERENCE
        self.chaos_order_balance: float = VOID_CHAOS_ORDER_BALANCE

        # Dynamics properties
        self.dissipation_rate: float = ENERGY_DISSIPATION_RATE
        self.propagation_speed: float = WAVE_PROPAGATION_SPEED
        self.resonance_threshold: float = HARMONIC_RESONANCE_THRESHOLD
        self.resonance_boost: float = HARMONIC_RESONANCE_ENERGY_BOOST

        # Noise Generator Instance
        self.noise_generator: Optional[NoiseGenerator] = None
        if NOISE_GEN_AVAILABLE and NoiseGenerator:
            try:
                self.noise_generator = NoiseGenerator(sample_rate=SAMPLE_RATE) # Assuming default SR for noise base
                logger.info("NoiseGenerator instantiated for VoidField.")
            except Exception as e:
                logger.error(f"Failed to instantiate NoiseGenerator: {e}. Noise generation will be basic.")
                self.noise_generator = None
        else:
             logger.warning("NoiseGenerator class not available.")


        # Initialize the grids
        self.initialize_grid() # This now includes noise and potential patterns
        logger.info(f"VoidField initialized with grid size {self.grid_size}.")

    def initialize_grid(self) -> None:
        """
        Sets up the initial state of the Void field's grids.
        Includes base energy, frequency noise, coherence, sparse patterns, and color.
        """
        logger.info(f"Initializing VoidField grid ({self.grid_size})...")
        shape = self.grid_size

        # 1. Energy Grid: Baseline + small random fluctuations
        self.energy = np.full(shape, self.base_energy, dtype=np.float32)
        self.energy += np.random.normal(0, self.base_energy * 0.05, shape).astype(np.float32)
        self.energy = np.clip(self.energy, 0.0, 1.0)

        # 2. Frequency Grid: Use NoiseGenerator for pink noise if available
        if self.noise_generator:
            try:
                # Generate base white noise
                white_noise_base = self.noise_generator.generate_noise('white', duration=1.0, amplitude=1.0) # Generate 1s sample
                # Tile/repeat to fill the grid size (approximate way to get noise volume)
                # This is computationally simpler than generating huge 3D noise directly
                num_repeats = int(np.ceil(np.prod(shape) / len(white_noise_base)))
                tiled_noise = np.tile(white_noise_base, num_repeats)[:np.prod(shape)]
                # Apply pink noise filter
                pink_noise = self.noise_generator._apply_filter(tiled_noise, self.noise_generator._generate_pink_noise)
                # Reshape and scale to frequency range
                self.frequency = pink_noise.reshape(shape).astype(np.float32)
                # Scale noise from [-1, 1] range to frequency range
                low, high = self.base_frequency_range
                self.frequency = ((self.frequency + 1.0) / 2.0) * (high - low) + low # Map from [-1,1] to [low,high]
                self.frequency = np.clip(self.frequency, low, high)
                logger.info("Initialized frequency grid with Pink Noise via NoiseGenerator.")
            except Exception as e:
                logger.error(f"NoiseGenerator failed for frequency: {e}. Falling back to uniform.")
                low, high = self.base_frequency_range
                self.frequency = np.random.uniform(low, high, shape).astype(np.float32)
        else: # Fallback
            low, high = self.base_frequency_range
            self.frequency = np.random.uniform(low, high, shape).astype(np.float32)
            logger.warning("Initialized frequency grid with uniform random noise (NoiseGenerator unavailable).")


        # 3. Coherence Grid: Baseline + variations
        self.coherence = np.full(shape, self.base_coherence, dtype=np.float32)
        self.coherence += np.random.normal(0, self.base_coherence * 0.1, shape).astype(np.float32)
        self.coherence = np.clip(self.coherence, 0.0, 1.0)

        # 4. Pattern Influence Grid: Initialize with zeros
        self.pattern_influence = np.zeros(shape, dtype=np.float32)

        # 5. Color Grid: Base void color + noise
        void_base_color = np.array([0.1, 0.05, 0.2], dtype=np.float32)
        self.color = np.tile(void_base_color, shape + (1,)).reshape(shape + (3,))
        self.color += np.random.normal(0, 0.02, shape + (3,)).astype(np.float32)
        self.color = np.clip(self.color, 0.0, 1.0)

        # 6. Chaos/Order Grids (derived)
        self.order = self.coherence * 0.6 + self.energy * 0.4
        self.chaos = 1.0 - self.order
        self.chaos = np.clip(self.chaos, 0.0, 1.0)
        self.order = np.clip(self.order, 0.0, 1.0)

        # --- Sparsely Apply Base Geometric & Platonic Potential ---
        num_patterns_to_place = 5 # Example: Place a few instances of each potential pattern
        # Geometry
        if GEOMETRY_FUNCS_AVAILABLE:
            logger.info("Applying sparse sacred geometry potential...")
            for pattern_name, pattern_func in GEOMETRY_FUNCTION_MAP.items():
                for _ in range(num_patterns_to_place):
                    try:
                        # Determine a random size and location
                        # Size should be smaller than grid dimensions
                        pattern_size = tuple(np.random.randint(max(5, s // 10), max(10, s // 4)) for s in shape)
                        location = tuple(np.random.randint(0, shape[i] - pattern_size[i]) for i in range(3))
                        strength = VOID_GEOMETRY_DENSITY * (0.5 + random.random()) # Add some variation

                        # Generate the influence grid for this pattern
                        influence_grid = pattern_func(shape=pattern_size, strength=strength) # Assumes func exists
                        if influence_grid is not None and influence_grid.shape == pattern_size:
                            # Get slice for placement
                            placement_slice = tuple(slice(loc, loc + size) for loc, size in zip(location, pattern_size))
                            # Add influence (allow overlap)
                            self.pattern_influence[placement_slice] += influence_grid
                        else: logger.warning(f"Failed to generate valid influence grid for {pattern_name}")
                    except Exception as e:
                        logger.error(f"Error applying geometry potential for {pattern_name}: {e}", exc_info=False) # Less verbose logging for loop errors
            self.pattern_influence = np.clip(self.pattern_influence, 0.0, 1.0) # Clamp total influence
            logger.info("Sacred geometry potential applied.")
        else: logger.warning("Skipping geometry potential application (functions unavailable).")

        # Platonics
        if PLATONIC_FUNCS_AVAILABLE:
            logger.info("Applying sparse platonic solid potential...")
            for solid_name, solid_func in PLATONIC_FUNCTION_MAP.items():
                 for _ in range(num_patterns_to_place):
                    try:
                        solid_size = tuple(np.random.randint(max(5, s // 12), max(10, s // 5)) for s in shape)
                        location = tuple(np.random.randint(0, shape[i] - solid_size[i]) for i in range(3))
                        density = VOID_PLATONIC_DENSITY * (0.5 + random.random())

                        influence_field = solid_func(shape=solid_size, strength=density) # Assumes func exists
                        if influence_field is not None and influence_field.shape == solid_size:
                            placement_slice = tuple(slice(loc, loc + size) for loc, size in zip(location, solid_size))
                            # Add influence (e.g., to pattern grid, or maybe directly to energy/coherence?)
                            # Let's add to pattern_influence for now
                            self.pattern_influence[placement_slice] += influence_field
                        else: logger.warning(f"Failed to generate valid influence field for {solid_name}")
                    except Exception as e:
                        logger.error(f"Error applying platonic potential for {solid_name}: {e}", exc_info=False)
            self.pattern_influence = np.clip(self.pattern_influence, 0.0, 1.0) # Clamp total influence
            logger.info("Platonic solid potential applied.")
        else: logger.warning("Skipping platonic potential application (functions unavailable).")

        logger.info("VoidField grid initialization complete.")

    def update_step(self, delta_time: float) -> None:
        """
        Simulates the dynamic evolution of the Void field over one time step.
        Includes resonance checks, energy dissipation, and wave propagation.

        Args:
            delta_time (float): The time step duration.
        """
        if self.energy is None or self.frequency is None or self.coherence is None:
             logger.error("VoidField grids are not initialized. Cannot update.")
             return
        if delta_time <= 0:
             logger.warning("delta_time must be positive for update_step.")
             return

        # --- 1. Harmonic Resonance ---
        freq = self.frequency
        energy_boost = np.zeros_like(self.energy, dtype=np.float32)
        coherence_boost = np.zeros_like(self.coherence, dtype=np.float32)

        for axis in range(3):
            for shift in [-1, 1]:
                neighbor_freq = np.roll(freq, shift, axis=axis)
                # Simple resonance check (can be refined)
                ratio = np.divide(np.maximum(freq, neighbor_freq),
                                  np.maximum(np.minimum(freq, neighbor_freq), FLOAT_EPSILON),
                                  where=np.minimum(freq, neighbor_freq) > FLOAT_EPSILON)
                is_resonant = np.abs(ratio - np.round(ratio)) < self.resonance_threshold # Near integer ratios
                is_resonant |= np.abs(ratio - GOLDEN_RATIO) < self.resonance_threshold # Near phi
                is_resonant |= np.abs(ratio / GOLDEN_RATIO - np.round(ratio / GOLDEN_RATIO)) < self.resonance_threshold # Phi harmonics

                # Apply boost where resonant, potentially modified by pattern influence
                resonance_modifier = 1.0 + self.pattern_influence[is_resonant] # Higher pattern influence slightly increases boost
                energy_boost[is_resonant] += self.resonance_boost * self.energy[is_resonant] * resonance_modifier
                coherence_boost[is_resonant] += self.resonance_boost * 0.5 * resonance_modifier

        self.energy += energy_boost * delta_time # Scale boost by time step
        self.coherence += coherence_boost * delta_time

        # --- 2. Energy Dissipation & Propagation (Simplified Diffusion) ---
        # Simple neighbor averaging for diffusion/propagation effect
        neighbors_sum = np.zeros_like(self.energy)
        for axis in range(3):
            for shift in [-1, 1]:
                neighbors_sum += np.roll(self.energy, shift, axis=axis)
        diffusion_effect = (neighbors_sum / 6.0 - self.energy) * self.propagation_speed * delta_time

        # Apply dissipation and diffusion
        self.energy += diffusion_effect
        self.energy *= (1.0 - self.dissipation_rate * delta_time)

        # --- 3. Update Frequency (Slow drift + noise) ---
        freq_drift = np.random.normal(0, 0.5 * delta_time, self.grid_size).astype(np.float32)
        self.frequency += freq_drift
        self.frequency = np.clip(self.frequency, self.base_frequency_range[0], self.base_frequency_range[1])

        # --- 4. Update Coherence (Drift towards target influenced by energy) ---
        coherence_target = self.base_coherence + (self.energy - self.base_energy) * 0.3 # Coherence influenced by energy deviation
        coherence_target = np.clip(coherence_target, 0.0, 1.0)
        self.coherence += (coherence_target - self.coherence) * 0.1 * delta_time # Slow adjustment
        self.coherence = np.clip(self.coherence, 0.0, 1.0)

        # --- 5. Update Chaos/Order ---
        self.order = self.coherence * 0.6 + self.energy * 0.4
        self.chaos = 1.0 - self.order
        self.order = np.clip(self.order, 0.0, 1.0)
        self.chaos = np.clip(self.chaos, 0.0, 1.0)

        # --- 6. Clamp energy ---
        self.energy = np.clip(self.energy, 0.0, 1.0) # Final clamp


    def get_properties_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """ Get properties, handling potential None grids during initialization issues. """
        if not self._validate_coordinates(coordinates):
            raise IndexError(f"Coordinates {coordinates} out of bounds {self.grid_size}.")

        x, y, z = coordinates
        properties = {"field_type": "void"}
        # Safely access attributes, providing defaults if None (shouldn't happen after init)
        properties["energy"] = float(self.energy[x, y, z]) if self.energy is not None else self.base_energy
        properties["frequency"] = float(self.frequency[x, y, z]) if self.frequency is not None else np.mean(self.base_frequency_range)
        properties["coherence"] = float(self.coherence[x, y, z]) if self.coherence is not None else self.base_coherence
        properties["pattern_influence"] = float(self.pattern_influence[x, y, z]) if self.pattern_influence is not None else 0.0
        properties["color"] = self.color[x, y, z].tolist() if self.color is not None else [0.1, 0.05, 0.2]
        properties["chaos"] = float(self.chaos[x, y, z]) if self.chaos is not None else (1.0 - (self.base_coherence*0.6+self.base_energy*0.4))
        properties["order"] = float(self.order[x, y, z]) if self.order is not None else (self.base_coherence*0.6+self.base_energy*0.4)
        return properties

    def apply_influence(self, position: Tuple[float, float, float], influence_type: str, strength: float, radius: float) -> None:
        """ Apply external influence (basic implementation). """
        logger.debug(f"Applying influence '{influence_type}' at {position} str={strength:.2f}, rad={radius:.1f}")
        center_x, center_y, center_z = [int(round(p)) for p in position]
        radius_sq = radius * radius

        min_x = max(0, center_x - int(radius)); max_x = min(self.grid_size[0], center_x + int(radius) + 1)
        min_y = max(0, center_y - int(radius)); max_y = min(self.grid_size[1], center_y + int(radius) + 1)
        min_z = max(0, center_z - int(radius)); max_z = min(self.grid_size[2], center_z + int(radius) + 1)

        # Use indices for efficient slicing and calculation
        x_idx, y_idx, z_idx = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y), np.arange(min_z, max_z), indexing='ij')
        dist_sq = (x_idx - center_x)**2 + (y_idx - center_y)**2 + (z_idx - center_z)**2
        mask = dist_sq <= radius_sq

        if not np.any(mask): return # No cells within radius

        # Calculate falloff based on distance (e.g., linear or gaussian)
        falloff = np.maximum(0.0, 1.0 - np.sqrt(dist_sq[mask] / max(FLOAT_EPSILON, radius_sq)))

        # Get grid slices corresponding to the mask
        grid_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
        masked_slice_indices = tuple(idx[mask] for idx in (x_idx, y_idx, z_idx)) # Indices within the full grid

        # Apply based on type
        if influence_type == 'energy_infusion':
            self.energy[masked_slice_indices] += strength * falloff
            self.energy[masked_slice_indices] = np.clip(self.energy[masked_slice_indices], 0.0, 1.0)
        elif influence_type == 'stabilizing':
            self.coherence[masked_slice_indices] += strength * 0.1 * falloff
            self.coherence[masked_slice_indices] = np.clip(self.coherence[masked_slice_indices], 0.0, 1.0)
            # Update order/chaos derived from coherence
            self.order[masked_slice_indices] = self.coherence[masked_slice_indices] * 0.6 + self.energy[masked_slice_indices] * 0.4
            self.chaos[masked_slice_indices] = 1.0 - self.order[masked_slice_indices]
            self.order[masked_slice_indices] = np.clip(self.order[masked_slice_indices], 0.0, 1.0)
            self.chaos[masked_slice_indices] = np.clip(self.chaos[masked_slice_indices], 0.0, 1.0)
        elif influence_type == 'frequency_shift':
             self.frequency[masked_slice_indices] += strength * falloff # Strength here is the Hz shift
             self.frequency[masked_slice_indices] = np.clip(self.frequency[masked_slice_indices], self.base_frequency_range[0], self.base_frequency_range[1])
        else:
            logger.warning(f"Unknown influence type '{influence_type}' requested.")


# --- END OF FILE src/stage_1/fields/void_field.py ---