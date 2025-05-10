# --- START OF FILE src/stage_1/fields/void_field.py ---

"""
Void Field Module (Refactored V4.3 - Principle-Driven S/C, PEP8, Optimal Points)

Represents the dynamic Void field using scaled Joules (SEU), Stability Units (SU),
and Coherence Units (CU) for base values and interactions. Includes detailed
resonance calculation, performance guards, improved error handling, and the
find_optimal_development_points method. Adheres to PEP 8 formatting.
"""

from datetime import datetime
import logging
logger = logging.getLogger(__name__)
# Ensure logger level is set appropriately
try:
    import constants.constants as const # Use alias
    logger.setLevel(const.LOG_LEVEL)
except ImportError:
    logger.warning(
        "Constants not loaded, using default INFO level for VoidField logger."
    )
    logger.setLevel(logging.INFO)

import numpy as np
import random
from typing import Dict, Any, Tuple, Optional, List
from abc import ABC, abstractmethod
from math import sqrt, pi as PI, exp

# --- Constants Import (using alias 'const') ---
try:
    from constants.constants import *
except ImportError as e:
    logger.critical("CRITICAL ERROR: constants.py failed import in void_field.py")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependencies ---
try:
    # Noise Generator is optional
    from sound.noise_generator import NoiseGenerator
    NOISE_GEN_AVAILABLE = True
except ImportError:
    NOISE_GEN_AVAILABLE = False
    NoiseGenerator = None # Define as None if unavailable
    logger.warning("NoiseGenerator not found. Void field init will use uniform random.")
try:
    from stage_1.fields.field_base import FieldBase
except ImportError:
    logger.critical("CRITICAL ERROR: Cannot import FieldBase")
    # Define dummy ABC if import fails, to allow class definition
    class FieldBase(ABC):
        @abstractmethod
        def initialize_grid(self):
            pass  # etc.

# --- Resonance Calculation Helper ---
def _calculate_cell_resonance(freq1: np.ndarray, freq2: np.ndarray) -> np.ndarray:
    """Vectorized resonance calculation between frequency grids."""
    # Initialize resonance array
    resonance = np.zeros_like(freq1, dtype=np.float32)
    # Create mask for valid frequencies (avoid division by zero)
    valid_mask = (freq1 > const.FLOAT_EPSILON) & (freq2 > const.FLOAT_EPSILON)
    if not np.any(valid_mask):
        return resonance # Return zeros if no valid pairs

    # Calculate ratio only for valid pairs
    f1_valid = freq1[valid_mask]
    f2_valid = freq2[valid_mask]
    # Ensure ratio is always >= 1
    ratio = np.maximum(f1_valid, f2_valid) / np.maximum(np.minimum(f1_valid, f2_valid), const.FLOAT_EPSILON)
    # Integer Resonance (Vectorized)
    int_res = np.zeros_like(ratio)
    for i in range(1, 5):
        for j in range(1, 5):
            target_ratio = float(max(i, j)) / float(min(i, j))
            # Closeness metric (sharper peak)
            deviation = np.abs(ratio - target_ratio)
            tolerance = const.RESONANCE_INTEGER_RATIO_TOLERANCE * target_ratio
            closeness = np.maximum(0.0, 1.0 - (deviation / max(tolerance, const.FLOAT_EPSILON)))**2
            int_res = np.maximum(int_res, closeness)

    # Phi Resonance (Vectorized)
    phi_res = np.zeros_like(ratio)
    for i in [1, 2]: # Check Phi^1, Phi^2 and inverses
        phi_pow = const.GOLDEN_RATIO ** i
        dev_phi = np.abs(ratio - phi_pow)
        tol_phi = const.RESONANCE_PHI_RATIO_TOLERANCE * phi_pow
        closeness_phi = np.maximum(0.0, 1.0 - (dev_phi / max(tol_phi, const.FLOAT_EPSILON)))**2

        inv_phi_pow = 1.0 / phi_pow
        dev_inv_phi = np.abs(ratio - inv_phi_pow)
        tol_inv_phi = const.RESONANCE_PHI_RATIO_TOLERANCE * inv_phi_pow
        closeness_inv_phi = np.maximum(0.0, 1.0 - (dev_inv_phi / max(tol_inv_phi, const.FLOAT_EPSILON)))**2

        phi_res = np.maximum(phi_res, closeness_phi)
        phi_res = np.maximum(phi_res, closeness_inv_phi)

    # Combine and store results back into the original shape using the mask
    resonance[valid_mask] = np.maximum(int_res, phi_res)
    return resonance


class VoidField(FieldBase):
    """ Dynamic Void field using SEU, SU, CU. Includes detailed resonance. """

    # --- __init__ ---
    def __init__(self, grid_size: Tuple[int, int, int] = const.GRID_SIZE):
        """Initialize the Void Field."""
        super().__init__(name="VoidField", grid_size=grid_size)

        # --- Base Properties (from constants) ---
        self.base_energy_seu = const.VOID_BASE_ENERGY_SEU
        self.update_counter: int = 0
        self.base_energy_seu = const.VOID_BASE_ENERGY_SEU
        self.base_frequency_range = const.VOID_BASE_FREQUENCY_RANGE
        self.base_stability_su = const.VOID_BASE_STABILITY_SU
        self.dissipation_rate = const.ENERGY_DISSIPATION_RATE
        self.propagation_speed = const.WAVE_PROPAGATION_SPEED
        self.resonance_energy_boost_factor = const.HARMONIC_RESONANCE_ENERGY_BOOST
        # Example: Coherence boost can be linked to energy boost
        self.resonance_coherence_boost_factor = (
            const.HARMONIC_RESONANCE_ENERGY_BOOST * 5.0 # Boost coherence more?
        )

        # --- Noise Generator (Optional) ---
        self.noise_generator = None
        if NOISE_GEN_AVAILABLE and NoiseGenerator:
            try:
                self.noise_generator = NoiseGenerator(sample_rate=const.SAMPLE_RATE)
                logger.info("NoiseGenerator initialized successfully for VoidField.")
            except NameError:
                logger.error("const.SAMPLE_RATE missing. Cannot initialize NoiseGenerator.")
            except Exception as e:
                logger.error(f"NoiseGenerator init failed: {e}")

        # --- Grid Initialization ---
        # Initialize grids to None initially, setup in initialize_grid
        self.energy: Optional[np.ndarray] = None
        self.frequency: Optional[np.ndarray] = None
        self.stability: Optional[np.ndarray] = None
        self.coherence: Optional[np.ndarray] = None
        self.pattern_influence: Optional[np.ndarray] = None
        self.color: Optional[np.ndarray] = None
        self.order: Optional[np.ndarray] = None
        self.chaos: Optional[np.ndarray] = None
        # Call initialization
        try:
             self.initialize_grid()
             logger.info(f"VoidField initialized with grid size {self.grid_size}.")
        except Exception as init_err:
             logger.critical("CRITICAL: Failed to initialize VoidField grids.",
                             exc_info=True)
             raise RuntimeError("VoidField grid initialization failed") from init_err

    # --- initialize_grid ---
    def initialize_grid(self) -> None:
        """Initializes the VoidField grids with base values and noise."""
        logger.info(f"Initializing VoidField grid ({self.grid_size}) with "
                    f"SEU/SU/CU units...")
        shape = self.grid_size
        try:
            # Energy (SEU)
            self.energy = np.clip(
                np.full(shape, self.base_energy_seu, dtype=np.float32) +
                np.random.normal(0, self.base_energy_seu * 0.1, shape).astype(np.float32),
                0.0, self.base_energy_seu * 5.0 # Allow higher initial peaks
            )
            logger.debug(f"Initialized energy grid. Shape: {self.energy.shape}")

            # Frequency (Hz) - Attempt noise, fallback to uniform
            low_freq, high_freq = self.base_frequency_range
            if self.noise_generator:
                try:
                    max_amp = getattr(const, 'MAX_AMPLITUDE', 0.95)
                    num_needed = np.prod(shape)
                    # Request slightly more duration than strictly needed
                    duration_needed = (float(num_needed) / float(const.SAMPLE_RATE)) + 0.1
                    logger.debug(f"Requesting noise duration: {duration_needed:.3f}s")
                    # Use Pink Noise for more natural frequency distribution
                    noise_data_raw = self.noise_generator.generate_noise(
                        'pink', duration=duration_needed, amplitude=max_amp
                    )
                    if noise_data_raw is None or len(noise_data_raw) < num_needed:
                         raise RuntimeError("Noise generator returned insufficient data")
                    # Slice and reshape
                    noise_data = noise_data_raw[:num_needed]
                    # Check for invalid noise data (NaN, Inf, or flat)
                    if not np.all(np.isfinite(noise_data)) or (num_needed > 1 and np.max(noise_data) == np.min(noise_data)):
                         raise RuntimeError("Generated noise data is invalid (NaN/Inf/Flat).")
                    min_noise, max_noise = np.min(noise_data), np.max(noise_data)
                    range_noise = max_noise - min_noise
                    # Normalize noise 0-1
                    noise_normalized = (np.zeros_like(noise_data) if range_noise < const.FLOAT_EPSILON
                                       else (noise_data - min_noise) / range_noise)
                    # Scale to frequency range and reshape
                    self.frequency = (noise_normalized * (high_freq - low_freq) + low_freq).reshape(shape).astype(np.float32)
                    if self.frequency.shape != shape: # Safeguard reshape
                         raise RuntimeError(f"Frequency grid reshape failed. Expected {shape}, got {self.frequency.shape}")
                    logger.info("Initialized frequency grid with Pink Noise.")
                except Exception as noise_err:
                    logger.error(f"Noise generation failed: {noise_err}. "
                                 f"Falling back to uniform random frequency.", exc_info=True)
                    self.frequency = np.random.uniform(low_freq, high_freq, shape).astype(np.float32)
            else:
                logger.warning("NoiseGenerator unavailable. Initializing frequency with uniform random.")
                self.frequency = np.random.uniform(low_freq, high_freq, shape).astype(np.float32)
            # Final clip
            self.frequency = np.clip(self.frequency, low_freq, high_freq)
            logger.debug(f"Initialized frequency grid. Shape: {self.frequency.shape}")

            # Stability (SU)
            self.stability = np.clip(
                np.full(shape, self.base_stability_su, dtype=np.float32) +
                np.random.normal(0, self.base_stability_su * 0.15, shape).astype(np.float32),
                0.0, const.MAX_STABILITY_SU
            )
            logger.debug(f"Initialized stability grid. Shape: {self.stability.shape}")

            # Coherence (CU)
            self.coherence = np.clip(
                np.full(shape, VOID_BASE_COHERENCE_CU, dtype=np.float32) +
                np.random.normal(0, VOID_BASE_COHERENCE_CU * 0.2, shape).astype(np.float32),
                0.0, const.MAX_COHERENCE_CU
            )
            logger.debug(f"Initialized coherence grid. Shape: {self.coherence.shape}")

            # Pattern Influence (0-1)
            self.pattern_influence = np.random.uniform(0.0, 0.05, shape).astype(np.float32)
            logger.debug(f"Initialized pattern_influence grid. Shape: {self.pattern_influence.shape}")

            # Color (RGB, 0-1)
            try:
                base_col = np.array([0.1, 0.05, 0.2], dtype=np.float32) # Base void color
                color_noise = np.random.normal(0, 0.02, shape + (3,)).astype(np.float32)
                # Tile base color and add noise
                self.color = np.clip(
                    np.tile(base_col, shape + (1,)).reshape(shape + (3,)) + color_noise,
                    0.0, 1.0
                )
                logger.debug(f"Initialized color grid. Shape: {self.color.shape}")
            except Exception as color_e:
                logger.error(f"Error initializing color grid: {color_e}. Setting default grey.", exc_info=True)
                self.color = np.full(shape + (3,), 0.5, dtype=np.float32)

            # Order/Chaos (0-1, derived)
            try:
                norm_s = self.stability / const.MAX_STABILITY_SU
                norm_c = self.coherence / const.MAX_COHERENCE_CU
                self.order = np.clip((norm_s * 0.6 + norm_c * 0.4), 0.0, 1.0)
                self.chaos = 1.0 - self.order
                logger.debug(f"Initialized order grid. Shape: {self.order.shape}")
                logger.debug(f"Initialized chaos grid. Shape: {self.chaos.shape}")
            except Exception as order_e:
                 logger.error(f"Error initializing order/chaos: {order_e}. Setting defaults.", exc_info=True)
                 self.order = np.full(shape, 0.5, dtype=np.float32)
                 self.chaos = np.full(shape, 0.5, dtype=np.float32)

            logger.info("VoidField grid initialization complete.")

        except Exception as e:
             logger.critical("Critical error during grid array creation.", exc_info=True)
             # Set grids to None to indicate failure
             self.energy=self.frequency=self.stability=self.coherence=None
             self.pattern_influence=self.color=self.order=self.chaos=None
             raise RuntimeError("Failed to create VoidField grid arrays.") from e


    # --- update_step ---
    def update_step(self, delta_time: float) -> None:
        """Simulates Void dynamics: diffusion, dissipation, resonance, drift."""
        # Check if grids are initialized
        if self.energy is None or self.frequency is None or \
           self.stability is None or self.coherence is None or \
           self.pattern_influence is None or self.order is None or \
           self.chaos is None or self.color is None:
            logger.error("Cannot update VoidField: Grids not initialized.")
            return
        if delta_time <= 0: return # No change if no time passed

        # Cap delta_time to prevent instability with large steps
        effective_dt = min(delta_time, 0.5) # Max time step cap
        if effective_dt != delta_time:
             logger.warning(f"Large delta_time ({delta_time:.2f}s) capped to "
                            f"{effective_dt:.2f}s for stability.")
             self.update_counter += 1

        try:
            # --- 1. Harmonic Resonance Effects ---
            freq = self.frequency
            energy_boost_factor = np.zeros_like(self.energy)
            coherence_boost_factor = np.zeros_like(self.coherence)
            stability_boost_factor = np.zeros_like(self.stability)
            light_emission_points = np.zeros(self.grid_size, dtype=bool)

            # Iterate over neighbors (can optimize with convolution/kernel later)
            for axis in range(self.dimensions):
                for shift in [-1, 1]:
                    neighbor_freq = np.roll(freq, shift, axis=axis)
                    resonance_score = _calculate_cell_resonance(freq, neighbor_freq)
                    
                    # Pattern influence enhances resonance effect
                    resonance_modifier = 1.0 + self.pattern_influence * 0.5
                    boost_mask = resonance_score > 0.1 # Only apply boost above threshold
                    
                    if np.any(boost_mask):
                        # Calculate energy boost (standard from original)
                        energy_boost_factor[boost_mask] += (
                            self.resonance_energy_boost_factor *
                            resonance_score[boost_mask] *
                            resonance_modifier[boost_mask]
                        )
                        
                        # Calculate coherence boost (standard from original)
                        coherence_boost_factor[boost_mask] += (
                            self.resonance_coherence_boost_factor *
                            resonance_score[boost_mask] *
                            resonance_modifier[boost_mask]
                        )
                        
                        # Add stability boost (new)
                        stability_boost_factor[boost_mask] += (
                            resonance_score[boost_mask] * 0.5 * 
                            resonance_modifier[boost_mask]
                        )
                        
                        # Mark high resonance points for light emission (new)
                        light_emission_mask = resonance_score > 0.8
                        if np.any(light_emission_mask):
                            light_emission_points[light_emission_mask] = True
                            
            # Apply boosts to SU, CU, and SEU (with logging for significant changes)
            old_energy = self.energy.copy()
            old_coherence = self.coherence.copy()
            old_stability = self.stability.copy()

            # Apply energy boost (SEU)
            self.energy *= (1.0 + energy_boost_factor * effective_dt)
            energy_change = np.sum(self.energy - old_energy)
            if abs(energy_change) > 1000:  # Log if total energy change is significant
                logger.info(f"Harmonic resonance caused energy change of {energy_change:.1f} SEU")

            # Apply coherence boost (CU)
            coherence_boost = coherence_boost_factor * const.MAX_COHERENCE_CU * effective_dt
            self.coherence += coherence_boost
            coherence_change = np.sum(self.coherence - old_coherence)
            if abs(coherence_change) > 10:  # Log if total coherence change is significant
                logger.info(f"Harmonic resonance caused coherence change of {coherence_change:.1f} CU")

            # Apply stability boost (SU)
            stability_boost = stability_boost_factor * const.MAX_STABILITY_SU * effective_dt
            self.stability += stability_boost
            stability_change = np.sum(self.stability - old_stability)
            if abs(stability_change) > 10:  # Log if total stability change is significant
                logger.info(f"Harmonic resonance caused stability change of {stability_change:.1f} SU")

            # Handle light emission at high resonance points
            if np.any(light_emission_points):
                # Generate light emission as energy transfer to neighbors
                emission_grid = np.zeros_like(self.energy)
                emission_grid[light_emission_points] = self.energy[light_emission_points] * 0.05
                
                # Apply emission as energy boost to surrounding points
                for axis in range(self.dimensions):
                    for shift in [-1, 1]:
                        # Transfer energy to neighbors, creating light-like propagation
                        target_grid = np.roll(emission_grid, shift, axis=axis)
                        self.energy += target_grid * 0.2  # Attenuate by distance
                
                # Reduce energy at emission points (conservation)
                self.energy[light_emission_points] *= 0.9
                
                # Log significant light emissions
                num_emission_points = np.sum(light_emission_points)
                if num_emission_points > 100:
                    logger.info(f"High harmonic resonance caused light emission at {num_emission_points} points")

            # --- 2. Energy Diffusion & Dissipation ---
            # Diffusion (Laplacian approximation)
            neighbors_sum = np.zeros_like(self.energy)
            for axis in range(self.dimensions):
                for shift in [-1, 1]: neighbors_sum += np.roll(self.energy, shift, axis=axis)
            # Basic diffusion effect (reduced strength compared to original)
            diffusion_effect = ((neighbors_sum / (2.0 * self.dimensions)) - self.energy)
            diffusion_effect *= (self.propagation_speed * 0.5) * effective_dt  # Reduced to 50%

            # Add light-like directional energy propagation
            light_energy = np.zeros_like(self.energy)
            # Emission probability based on local energy and chaos (more chaos = more emission)
            emission_probability = np.clip(self.energy * 0.001 * (0.5 + self.chaos * 0.5), 0, 0.2)
            # Random emissions based on probability (vectorized)
            random_values = np.random.random(self.energy.shape)
            new_emissions = (random_values < emission_probability) * (self.energy * 0.01)
            # Zero out emissions with insufficient energy
            new_emissions = np.where(self.energy > const.FLOAT_EPSILON * 100, new_emissions, 0)

            # Propagate emissions in all directions (light-like behavior)
            for axis in range(self.dimensions):
                for direction in [-1, 1]:
                    # Directional propagation (apply smaller dt for light-speed effect)
                    shifted_emissions = np.roll(new_emissions, direction, axis=axis) * effective_dt * 10.0
                    light_energy += shifted_emissions

            # Apply combined effects (diffusion + light propagation)
            self.energy += diffusion_effect + light_energy

            # Energy interaction effects (occasional "bursts" of energy)
            if hasattr(self, 'update_counter') and self.update_counter % 3 == 0:  # Only calculate sometimes for performance
                energy_threshold = self.base_energy_seu * 3.0
                high_energy_points = self.energy > energy_threshold
                if np.any(high_energy_points):
                    # Energy bursts at high energy points (simulates energy release)
                    local_chaos = self.chaos[high_energy_points]
                    # Scale burst intensity by chaos (more chaos = larger bursts)
                    burst_intensity = local_chaos * 0.2 * energy_threshold
                    # Apply bursts (add energy to neighbors, remove from source)
                    energy_mask = np.zeros_like(self.energy)
                    energy_mask[high_energy_points] = burst_intensity
                    # Spread to neighbors
                    for axis in range(self.dimensions):
                        for shift in [-1, 1]:
                            self.energy += np.roll(energy_mask, shift, axis=axis) * 0.2
                    # Reduce source energy (conservation)
                    self.energy[high_energy_points] *= 0.8

            # Dissipation (simple exponential decay - retain this from original)
            self.energy *= (1.0 - self.dissipation_rate * effective_dt)

            # --- 3. Property Drifts & Noise ---
            # Frequency drift (random walk)
            freq_drift = np.random.normal(0, 1.0 * effective_dt,
                                          self.grid_size).astype(np.float32)
            self.frequency += freq_drift
            # Drift towards baseline values (slow relaxation)
            self.stability += ((self.base_stability_su - self.stability) *
                               0.01 * effective_dt)
            self.coherence += ((VOID_BASE_COHERENCE_CU - self.coherence) *
                               0.02 * effective_dt)
            # Pattern influence decay
            self.pattern_influence *= (1.0 - 0.005 * effective_dt)

            # --- 4. Clamping ---
            self.energy = np.clip(self.energy, 0.0, const.MAX_SOUL_ENERGY_SEU * 5) # Allow higher peaks in void
            self.stability = np.clip(self.stability, 0.0, const.MAX_STABILITY_SU)
            self.coherence = np.clip(self.coherence, 0.0, const.MAX_COHERENCE_CU)
            self.frequency = np.clip(self.frequency, self.base_frequency_range[0],
                                     self.base_frequency_range[1])
            self.pattern_influence = np.clip(self.pattern_influence, 0.0, 1.0)

            # --- 5. Update Derived Properties (Order/Chaos, Color) ---
            norm_s = self.stability / const.MAX_STABILITY_SU
            norm_c = self.coherence / const.MAX_COHERENCE_CU
            self.order = np.clip((norm_s * 0.6 + norm_c * 0.4), 0.0, 1.0)
            self.chaos = 1.0 - self.order
            # Color update with light-like spectral response
            energy_factor = np.clip(self.energy / (self.base_energy_seu * 5.0), 0.1, 1.0)
            coherence_factor = np.clip(self.coherence / const.MAX_COHERENCE_CU, 0.0, 1.0)
            frequency_factor = np.clip((self.frequency - self.base_frequency_range[0]) / 
                                    (self.base_frequency_range[1] - self.base_frequency_range[0]), 0.0, 1.0)

            # Map frequency to visible spectrum approximation
            r_component = np.clip((1.0 - frequency_factor) * 2.0, 0.0, 1.0)  # Higher at low frequencies (red)
            g_component = np.clip(1.0 - np.abs(frequency_factor * 2.0 - 1.0), 0.0, 1.0)  # Peak in middle
            b_component = np.clip((frequency_factor - 0.5) * 2.0, 0.0, 1.0)  # Higher at high frequencies (blue)

            # Create target color based on frequency
            hue_shift = (coherence_factor - 0.5) * 0.2
            target_r = np.clip(r_component + hue_shift, 0.0, 1.0)
            target_g = np.clip(g_component, 0.0, 1.0)
            target_b = np.clip(b_component - hue_shift, 0.0, 1.0)

            # Blend current color with target (slow transition for stability)
            blend_rate = 0.02 * effective_dt  # Adjust rate based on time step
            self.color[..., 0] = np.clip(self.color[..., 0] * (1.0 - blend_rate) + target_r * blend_rate, 0, 1)
            self.color[..., 1] = np.clip(self.color[..., 1] * (1.0 - blend_rate) + target_g * blend_rate, 0, 1)
            self.color[..., 2] = np.clip(self.color[..., 2] * (1.0 - blend_rate) + target_b * blend_rate, 0, 1)

            # Add "flashes" of light at high-energy points (bursts of brightness)
            if hasattr(self, 'update_counter') and self.update_counter % 5 == 0:
                high_energy_points = self.energy > (self.base_energy_seu * 4.0)
                if np.any(high_energy_points):
                    # Create flash effect at these points
                    flash_intensity = np.zeros_like(self.energy)
                    flash_intensity[high_energy_points] = 1.0
                    # Apply flash to color (temporary brightness boost)
                    self.color[high_energy_points, :] = np.clip(self.color[high_energy_points, :] * 1.5, 0, 1)
                    flash_count = np.sum(high_energy_points)
                    if flash_count > 100:
                        logger.info(f"Light flash event at {flash_count} high-energy points")
                        
                        # Generate sound for significant energy flash events
                        try:
                            # Import locally to avoid circular dependencies
                            try:
                                from sound.sound_generator import SoundGenerator
                                sound_gen_available = True
                            except ImportError:
                                sound_gen_available = False
                                logger.debug("SoundGenerator not available for burst sound generation")
                            
                            if sound_gen_available:
                                # Create a sound generator
                                sound_gen = SoundGenerator(output_dir="output/sounds/energy_events")
                                
                                # Define radius for bounding box calculation
                                burst_radius = 5.0  # Use fixed radius for sound burst area
                                
                                # Calculate center point from high energy points
                                center_points = np.where(high_energy_points)
                                center_x = int(np.mean(center_points[0]))
                                center_y = int(np.mean(center_points[1]))
                                center_z = int(np.mean(center_points[2]))
                                
                                # Calculate bounding box for the affected area
                                min_x = max(0, center_x - int(burst_radius))
                                max_x = min(self.grid_size[0], center_x + int(burst_radius) + 1)
                                min_y = max(0, center_y - int(burst_radius))
                                max_y = min(self.grid_size[1], center_y + int(burst_radius) + 1)
                                min_z = max(0, center_z - int(burst_radius))
                                max_z = min(self.grid_size[2], center_z + int(burst_radius) + 1)
                                
                                # Define box slice using the calculated bounds
                                box_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
                                # Create mask for points within the burst radius
                                x_idx, y_idx, z_idx = np.meshgrid(
                                    np.arange(min_x, max_x),
                                    np.arange(min_y, max_y),
                                    np.arange(min_z, max_z),
                                    indexing='ij'
                                )
                                dist_sq = ((x_idx - center_x)**2 + 
                                         (y_idx - center_y)**2 + 
                                         (z_idx - center_z)**2)
                                mask = dist_sq <= burst_radius**2
                                
                                # Calculate average frequency from affected points
                                avg_freq = np.mean(self.frequency[box_slice][mask])
                                
                                # Create parameters for energy burst sound
                                sound = sound_gen.generate_harmonic_tone(
                                    base_frequency=avg_freq,
                                    harmonics=[1.0, 1.5, 2.0, 2.5],  # More harmonics for a rich sound
                                    amplitudes=[0.8, 0.4, 0.2, 0.1],  # Decreasing amplitudes
                                    duration=3.0,
                                    fade_in_out=0.5
                                )
                                
                                # Generate and save the sound
                                burst_sound = sound
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"energy_burst_{timestamp}.wav"
                                sound_file = sound_gen.save_sound(burst_sound, filename)
                                
                                if sound_file:
                                    logger.info(f"Generated sound file for energy burst: {sound_file}")
                        except Exception as sound_err:
                            logger.error(f"Error generating energy burst sound: {sound_err}", exc_info=True)
                            # Non-critical, continue execution

            # Final modulation by energy (brighter where more energy)
            self.color *= energy_factor[..., np.newaxis]
            self.color = np.clip(self.color, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error during VoidField update step: {e}", exc_info=True)
            # Potentially reset grids or handle error gracefully? For now, log and continue.

    # --- get_properties_at ---
    def get_properties_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """Gets field properties (SEU/SU/CU) at specific coordinates."""
        if not self._validate_coordinates(coordinates):
            raise IndexError(f"Coordinates {coordinates} out of grid bounds {self.grid_size}")
        x, y, z = coordinates

        # Check grid initialization before access
        if any(g is None for g in [self.energy, self.frequency, self.stability,
                                    self.coherence, self.pattern_influence,
                                    self.color, self.order, self.chaos]):
            logger.error(f"Grid not initialized when accessing {coordinates}.")
            # Return minimal dict or raise error? Returning dict allows caller checks.
            return {"field_type": "void", "error": "Uninitialized grid"}

        try:
            properties = {"field_type": "void"}
            # Return properties in defined units
            properties["energy_seu"] = float(self.energy[x, y, z])
            properties["frequency_hz"] = float(self.frequency[x, y, z])
            properties["stability_su"] = float(self.stability[x, y, z])
            properties["coherence_cu"] = float(self.coherence[x, y, z])
            properties["pattern_influence"] = float(self.pattern_influence[x, y, z]) # 0-1
            properties["color_rgb"] = self.color[x, y, z].tolist() # List [r,g,b] 0-1
            properties["order_factor"] = float(self.order[x, y, z]) # 0-1
            properties["chaos_factor"] = float(self.chaos[x, y, z]) # 0-1
            return properties
        except IndexError: # Should be caught by _validate_coordinates, but safeguard
            raise IndexError(f"Coordinates {coordinates} caused IndexError during property access.")
        except Exception as e:
            logger.error(f"Unexpected error in get_properties_at({coordinates}): {e}", exc_info=True)
            raise # Re-raise unexpected errors


    # --- apply_influence ---
    def apply_influence(self, position: Tuple[float, float, float],
                        influence_type: str, strength: float, radius: float) -> None:
        """Applies external influence (e.g., from soul) to the VoidField."""
        logger.debug(f"Applying influence '{influence_type}' at {position} "
                     f"str={strength:.2f}, rad={radius:.1f}")
        try:
            center_x, center_y, center_z = [int(round(p)) for p in position]
            if not self._validate_coordinates((center_x, center_y, center_z)):
                logger.warning(f"Influence center {position} is out of bounds.")
                return # Ignore influence outside grid

            # Calculate radius squared once
            radius_sq = radius * radius
            
            # Define bounding box, ensure it's within grid limits
            min_x = max(0, center_x - int(radius))
            max_x = min(self.grid_size[0], center_x + int(radius) + 1)
            min_y = max(0, center_y - int(radius))
            max_y = min(self.grid_size[1], center_y + int(radius) + 1)
            min_z = max(0, center_z - int(radius))
            max_z = min(self.grid_size[2], center_z + int(radius) + 1)

            # Check if bounding box is valid
            if min_x >= max_x or min_y >= max_y or min_z >= max_z: return # No volume to affect

            # Create index grids ONLY for the bounding box slice
            x_idx, y_idx, z_idx = np.meshgrid(
                np.arange(min_x, max_x), np.arange(min_y, max_y), np.arange(min_z, max_z),
                indexing='ij'
            )
            # Calculate squared distance from center within the box
            dist_sq = ((x_idx - center_x)**2 + (y_idx - center_y)**2 +
                       (z_idx - center_z)**2)
            # Create mask for points within the sphere AND within the box
            mask = dist_sq <= radius_sq
            if not np.any(mask): return # No points affected

            # Calculate falloff factor based on distance (0 at edge, 1 at center)
            falloff = np.maximum(0.0, 1.0 - np.sqrt(dist_sq[mask] / max(const.FLOAT_EPSILON, radius_sq)))

            # Define slice for the affected bounding box
            box_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))

            # Check grids before applying influence
            if any(g is None for g in [self.energy, self.stability, self.coherence,
                                        self.frequency, self.pattern_influence,
                                        self.order, self.chaos]):
                 logger.error("Cannot apply influence: VoidField grids not initialized.")
                 return

            # Apply influence based on type (vectorized on the masked slice)
            if influence_type == 'energy_infusion':
                self.energy[box_slice][mask] += strength * falloff # Add SEU
            elif influence_type == 'stabilizing':
                # Push towards target SU (strength defines the target SU)
                target_su = strength
                current_su = self.stability[box_slice][mask]
                change = (target_su - current_su) * 0.1 * falloff # Gradual push
                self.stability[box_slice][mask] += change
            elif influence_type == 'coherence_boost':
                 # Push towards target CU (strength defines the target CU)
                target_cu = strength
                current_cu = self.coherence[box_slice][mask]
                change = (target_cu - current_cu) * 0.1 * falloff # Gradual push
                self.coherence[box_slice][mask] += change
            elif influence_type == 'frequency_shift':
                # Shift frequency by strength * falloff (Hz change)
                self.frequency[box_slice][mask] += strength * falloff
            elif influence_type == 'pattern_imprint':
                 # Increase pattern influence factor (0-1)
                self.pattern_influence[box_slice][mask] += strength * falloff
            else:
                logger.warning(f"Unknown influence type requested: '{influence_type}'")
                return # Do nothing for unknown types

            # --- Re-clamp and update derived properties for the affected region ---
            affected_indices = tuple(idx[mask] for idx in (x_idx,y_idx,z_idx)) # Get affected indices

            if influence_type in ['energy_infusion']:
                 self.energy[affected_indices] = np.clip(self.energy[affected_indices], 0.0, const.MAX_SOUL_ENERGY_SEU * 5)
            if influence_type in ['stabilizing', 'coherence_boost']:
                 self.stability[affected_indices] = np.clip(self.stability[affected_indices], 0.0, const.MAX_STABILITY_SU)
                 self.coherence[affected_indices] = np.clip(self.coherence[affected_indices], 0.0, const.MAX_COHERENCE_CU)
                 # Update order/chaos only if stability/coherence changed
                 norm_s_affected = self.stability[affected_indices] / const.MAX_STABILITY_SU
                 norm_c_affected = self.coherence[affected_indices] / const.MAX_COHERENCE_CU
                 new_order_affected = np.clip((norm_s_affected * 0.6 + norm_c_affected * 0.4), 0.0, 1.0)
                 self.order[affected_indices] = new_order_affected
                 self.chaos[affected_indices] = 1.0 - new_order_affected
            if influence_type in ['frequency_shift']:
                 self.frequency[affected_indices] = np.clip(self.frequency[affected_indices], self.base_frequency_range[0], self.base_frequency_range[1])
            if influence_type in ['pattern_imprint']:
                 self.pattern_influence[affected_indices] = np.clip(self.pattern_influence[affected_indices], 0.0, 1.0)

        except IndexError:
            logger.error(f"IndexError applying influence at {position}. "
                         f"Calculated box slice: {box_slice}, Mask shape: {mask.shape}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error applying influence: {e}", exc_info=True)

    # --- calculate_edge_of_chaos ---
    def calculate_edge_of_chaos(self, coordinates: Tuple[int, int, int],
                                radius: int = 3) -> float:
        """Calculates edge of chaos metric (0-1) in a local region."""
        if not self._validate_coordinates(coordinates):
            logger.warning(f"Invalid coordinates for EoC calculation: {coordinates}")
            return 0.0 # Return default/neutral value for invalid coords

        x, y, z = coordinates
        # Define local region bounds, clamped to grid size
        min_x = max(0, x - radius); max_x = min(self.grid_size[0], x + radius + 1)
        min_y = max(0, y - radius); max_y = min(self.grid_size[1], y + radius + 1)
        min_z = max(0, z - radius); max_z = min(self.grid_size[2], z + radius + 1)

        # Check if region is valid
        if min_x >= max_x or min_y >= max_y or min_z >= max_z:
             logger.warning(f"Invalid region slice for EoC at {coordinates}")
             return 0.0

        try:
            # Slice the order grid for the local region
            region_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
            if self.order is None:
                logger.error("Order grid not initialized for EoC calculation.")
                return 0.0
            order_vals = self.order[region_slice]
            if order_vals.size == 0: return 0.0 # Empty region

            # Calculate metrics: Balance (avg closeness to 0.5) and Complexity (variance)
            avg_order = np.mean(order_vals)
            # Balance: Higher when avg_order is near 0.5
            balance = 1.0 - 2.0 * abs(avg_order - 0.5)
            # Complexity: Higher variance indicates more complex patterns
            variance = np.var(order_vals)
            # Scale variance to 0-1 roughly (e.g., max variance is 0.25 for 0/1 values)
            # Multiply by balance: complexity requires balance
            complexity = np.clip(sqrt(variance) * 4.0 * balance, 0.0, 1.0)

            # Combine balance and complexity (weighted average)
            edge_metric = balance * 0.6 + complexity * 0.4
            return float(np.clip(edge_metric, 0.0, 1.0)) # Final clamp 0-1

        except Exception as e:
            logger.error(f"Error calculating EoC at {coordinates} (radius {radius}): {e}",
                         exc_info=True)
            return 0.0 # Return default/neutral value on error

    # --- get_edge_of_chaos_grid ---
    def get_edge_of_chaos_grid(self, low_res: bool = True) -> np.ndarray:
        """Calculates the Edge of Chaos metric for the entire grid."""
        if self.order is None:
            logger.error("Cannot calculate EoC grid: Order grid not initialized.")
            return np.zeros(self.grid_size if not low_res else tuple(max(1, s//4) for s in self.grid_size))

        # Performance check for full resolution
        if not low_res and np.prod(self.grid_size) > 100**3: # Arbitrary limit
            logger.warning("Calculating full-res EoC grid for large size - "
                           "this might be slow.")

        target_shape = tuple(max(1, s // 4) for s in self.grid_size) if low_res else self.grid_size
        result = np.zeros(target_shape, dtype=np.float32)
        sample_factor = 4 if low_res else 1
        local_radius = 2 if low_res else 3 # Smaller radius for low-res sampling

        # Iterate over the target grid (low-res or full-res)
        it = np.nditer(result, flags=['multi_index'], op_flags=['writeonly'])
        with it:
             for x in it:
                 # Map target grid index back to original grid index
                 # Take the center of the sampled block
                 orig_coords = tuple(min(self.grid_size[i] - 1,
                                     int(round((it.multi_index[i] + 0.5) * sample_factor)))
                                 for i in range(self.dimensions))

                 # Calculate EoC centered at this original coordinate
                 eoc_value = self.calculate_edge_of_chaos(orig_coords, radius=local_radius)
                 x[...] = eoc_value # Assign value to the target grid

        return result

    # --- find_optimal_development_points ---
    def find_optimal_development_points(self, count: int = 5
                                       ) -> List[Tuple[int, int, int]]:
        """Find points with high Edge of Chaos values."""
        if count <= 0: return []
        logger.debug(f"Finding {count} optimal development points...")
        try:
            # Calculate low-res EoC grid for performance
            edge_grid = self.get_edge_of_chaos_grid(low_res=True)
            if edge_grid is None or np.prod(edge_grid.shape) == 0:
                logger.warning("EoC grid is empty. Returning random points.")
                return [tuple(np.random.randint(0, d)) for d in self.grid_size
                        for _ in range(count)] # Generate 'count' random points

            # Get indices sorted by EoC value (highest first)
            # Flatten the grid and get indices that would sort it descendingly
            flat_indices_desc = np.argsort(edge_grid.flatten())[::-1]

            optimal_points_candidates = []
            sample_factor = 4 # Factor used in get_edge_of_chaos_grid (low_res=True)
            lowres_shape = edge_grid.shape
            seen_coords = set() # Track added coords to avoid duplicates

            for flat_idx in flat_indices_desc:
                # Convert flat index back to 3D low-res coordinates
                coords_lowres = np.unravel_index(flat_idx, lowres_shape)

                # Map low-res coords back to original grid coords (approx center of block)
                orig_coords = tuple(min(self.grid_size[i] - 1, # Clamp to max index
                                     max(0, int(round((coords_lowres[i] + 0.5) * sample_factor))))
                                for i in range(self.dimensions))

                # Validate and add if not already seen
                if self._validate_coordinates(orig_coords) and orig_coords not in seen_coords:
                    optimal_points_candidates.append(orig_coords)
                    seen_coords.add(orig_coords)

                # Stop when we have enough points
                if len(optimal_points_candidates) >= count:
                    break

            # Fill with random points if fewer optimal points were found than requested
            while len(optimal_points_candidates) < count:
                rand_coords = tuple(np.random.randint(0, d) for d in self.grid_size)
                if rand_coords not in seen_coords:
                     optimal_points_candidates.append(rand_coords)
                     seen_coords.add(rand_coords)

            logger.info(f"Found {len(optimal_points_candidates)} candidate points. "
                        f"Returning top {count}: {optimal_points_candidates[:count]}")
            return optimal_points_candidates[:count] # Return only the requested number

        except Exception as e:
            logger.error(f"Error finding optimal development points: {e}", exc_info=True)
            # Fallback to random points on error
            return [tuple(np.random.randint(0, d)) for d in self.grid_size
                    for _ in range(count)]

    # --- Other Helper Methods ---
    def _validate_coordinates(self, coordinates: Tuple[int, int, int]) -> bool:
        """Checks if grid coordinates are within the field bounds."""
        try:
            # Check type and length first for robustness
            if not isinstance(coordinates, tuple) or len(coordinates) != self.dimensions:
                return False
            # Check bounds for each dimension
            return all(0 <= coordinates[i] < self.grid_size[i]
                       for i in range(self.dimensions))
        except:
             return False # Handle any unexpected errors during check

    def apply_geometric_pattern(self, pattern_type: str, location: Tuple[int, int, int],
                               radius: float, strength: float) -> None:
        """Applies a predefined geometric pattern influence."""
        # This method relies on the pattern influence grid being present
        if self.pattern_influence is None:
             logger.error("Cannot apply geometric pattern: Pattern grid not initialized.")
             return
        if not self._validate_coordinates(location):
             logger.warning(f"Cannot apply pattern '{pattern_type}' at invalid "
                            f"location {location}.")
             return
        if radius <= 0 or strength <= 0: return # No effect

        logger.debug(f"Applying geometric pattern '{pattern_type}' at {location} "
                     f"(rad={radius:.1f}, str={strength:.2f})")
        # --- Bounding Box & Mask ---
        center_x, center_y, center_z = location
        radius_sq = radius * radius
        min_x=max(0,center_x-int(radius)); max_x=min(self.grid_size[0],center_x+int(radius)+1)
        min_y=max(0,center_y-int(radius)); max_y=min(self.grid_size[1],center_y+int(radius)+1)
        min_z=max(0,center_z-int(radius)); max_z=min(self.grid_size[2],center_z+int(radius)+1)
        if min_x>=max_x or min_y>=max_y or min_z>=max_z: return

        x_idx, y_idx, z_idx = np.meshgrid(
            np.arange(min_x, max_x), np.arange(min_y, max_y), np.arange(min_z, max_z),
            indexing='ij'
        )
        dist_sq = ((x_idx - center_x)**2 + (y_idx - center_y)**2 +
                   (z_idx - center_z)**2)
        mask = dist_sq <= radius_sq
        if not np.any(mask): return

        # --- Calculate Pattern Influence ---
        # Example: Increase pattern_influence grid based on pattern type & falloff
        # More complex patterns could use precomputed grids (see FieldHarmonics idea)
        falloff = np.maximum(0.0, 1.0 - np.sqrt(dist_sq[mask] / max(const.FLOAT_EPSILON, radius_sq)))
        # Base influence increase
        influence_increase = strength * falloff
        # Modify increase based on pattern type (simple example)
        if 'flower' in pattern_type or 'tree' in pattern_type: influence_increase *= 1.2 # Harmony boost
        elif 'tetrahedron' in pattern_type: influence_increase *= 0.8 # Structure focus
        elif 'merkaba' in pattern_type: influence_increase *= 1.1 # Transformative boost

        # --- Apply to Grid ---
        box_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
        current_influence = self.pattern_influence[box_slice][mask]
        self.pattern_influence[box_slice][mask] = np.clip(
            current_influence + influence_increase, 0.0, 1.0
        )


    # --- __str__ ---
    def __str__(self) -> str:
        """String representation of the VoidField."""
        status = []
        status.append(f"Size:{self.grid_size}")
        if self.energy is not None: status.append(f"E:{np.mean(self.energy):.1f}SEU")
        if self.stability is not None: status.append(f"S:{np.mean(self.stability):.1f}SU")
        if self.coherence is not None: status.append(f"C:{np.mean(self.coherence):.1f}CU")
        if self.order is not None: status.append(f"Order:{np.mean(self.order):.2f}")
        return f"VoidField({', '.join(status)})"

# --- END OF FILE src/stage_1/fields/void_field.py ---