# --- START OF FILE src/stage_1/soul_formation/spark_harmonization.py ---

"""
Spark Harmonization Functions (Refactored V4.3.2 - Natural Principles)

Performs initial harmonization of newly emerged soul spark through natural
resonance principles. Enhances phi_resonance, pattern_coherence, harmony, 
and toroidal_flow_strength through phase-based and structural resonance.
Stability/Coherence emerge naturally from these enhanced factors.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional
from math import pi as PI, sqrt, exp, sin, cos, atan2, tanh

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Constants Import ---
try:
    from shared.constants.constants import *
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

# --- Configure Logger ---
try:
    logger.setLevel(LOG_LEVEL)
except NameError:
    logger.warning("LOG_LEVEL constant not found. Using default INFO level.")
    logger.setLevel(logging.INFO)
if not logger.handlers:
    log_formatter = logging.Formatter(LOG_FORMAT if 'LOG_FORMAT' in globals() else '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

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

# --- Helper Functions ---

def _calculate_circular_variance(phases_array: np.ndarray) -> float:
    """
    Calculates circular variance of phases (0=perfectly aligned, 1=uniform).
    Used to assess phase coherence.
    """
    if phases_array is None or not isinstance(phases_array, np.ndarray) or phases_array.size < 2:
        return 1.0  # Max variance if no/one phase or invalid input
    
    if not np.all(np.isfinite(phases_array)):
        logger.warning("Non-finite values found in phases array during variance calculation.")
        return 1.0  # Max variance if data invalid
    
    mean_cos = np.mean(np.cos(phases_array))
    mean_sin = np.mean(np.sin(phases_array))
    r_len_sq = mean_cos**2 + mean_sin**2
    
    if r_len_sq < 0:  # Protect against precision errors
        r_len_sq = 0.0
    
    r_len = sqrt(r_len_sq)
    return 1.0 - r_len

def _optimize_phase_coherence(soul_spark: SoulSpark, target_factor: float) -> float:
    """
    Optimizes phase coherence by aligning phases.
    Returns improvement amount (0-1 scale).
    """
    if not isinstance(soul_spark.frequency_signature, dict):
        return 0.0
    
    if 'phases' not in soul_spark.frequency_signature:
        return 0.0
    
    phases_data = soul_spark.frequency_signature['phases']
    if phases_data is None:
        return 0.0
    
    if isinstance(phases_data, list) and len(phases_data) == 0:
        return 0.0
    
    phases = np.array(phases_data) if not isinstance(phases_data, np.ndarray) else phases_data
    if phases.size <= 1:
        return 0.0
    
    # Calculate initial phase coherence
    mean_cos = np.mean(np.cos(phases))
    mean_sin = np.mean(np.sin(phases))
    initial_r = np.sqrt(mean_cos**2 + mean_sin**2)
    initial_var = 1.0 - initial_r
    mean_phase = np.arctan2(mean_sin, mean_cos)
    
    # Apply phase adjustment proportional to target_factor
    # Higher target_factor = stronger alignment force
    new_phases = phases * (1.0 - target_factor) + mean_phase * target_factor
    new_phases = new_phases % (2 * PI)
    
    # Calculate new coherence
    new_mean_cos = np.mean(np.cos(new_phases))
    new_mean_sin = np.mean(np.sin(new_phases))
    new_r = np.sqrt(new_mean_cos**2 + new_mean_sin**2)
    new_var = 1.0 - new_r
    
    # Update if improved
    if new_var < initial_var - FLOAT_EPSILON:
        soul_spark.frequency_signature['phases'] = new_phases.tolist()
        return (initial_var - new_var)
    
    return 0.0

def _enhance_phi_resonance(soul_spark: SoulSpark, current_phase: int, intensity: float, max_gain: float) -> float:
    """
    Enhances phi resonance through natural frequency alignment with golden ratio.
    Returns the amount of enhancement applied (0-1 scale).
    """
    current_phi = soul_spark.phi_resonance
    
    # Natural ceiling effect - increasingly difficult to improve as value approaches 1.0
    remaining_potential = 1.0 - current_phi
    if remaining_potential <= FLOAT_EPSILON:
        return 0.0
    
    # Phase-dependent intensity modulation
    # Different phases have different focus areas with natural wave-like progression
    phase_modifier = 0.7 + 0.3 * sin(PI * current_phase / 4)
    
    # Each increment follows a natural logarithmic growth curve
    # Higher values see diminishing returns - natural adaptation pattern
    gain = min(max_gain, remaining_potential * (1.0 - exp(-intensity * phase_modifier)))
    
    # Apply enhancement
    soul_spark.phi_resonance = min(1.0, current_phi + gain)
    
    return gain

def _enhance_pattern_coherence(soul_spark: SoulSpark, current_phase: int, intensity: float, max_gain: float) -> float:
    """
    Enhances pattern coherence through emergent structural organization.
    Returns the amount of enhancement applied (0-1 scale).
    """
    current_pcoh = soul_spark.pattern_coherence
    
    # Natural ceiling effect
    remaining_potential = 1.0 - current_pcoh
    if remaining_potential <= FLOAT_EPSILON:
        return 0.0
    
    # Pattern coherence benefits from phi resonance - natural synergy
    phi_synergy = 0.5 + 0.5 * soul_spark.phi_resonance
    
    # Phase-dependent modulation
    phase_modifier = 0.5 + 0.5 * cos(PI * current_phase / 4)
    
    # Natural logarithmic growth curve with synergistic scaling
    gain = min(max_gain, remaining_potential * (1.0 - exp(-intensity * phase_modifier * phi_synergy)))
    
    # Apply enhancement
    soul_spark.pattern_coherence = min(1.0, current_pcoh + gain)
    
    return gain

def _develop_toroidal_flow(soul_spark: SoulSpark, current_phase: int, intensity: float, max_gain: float) -> float:
    """
    Develops toroidal flow through resonant energy circulation patterns.
    Returns the amount of enhancement applied (0-1 scale).
    """
    current_torus = soul_spark.toroidal_flow_strength
    
    # Natural ceiling effect
    remaining_potential = 1.0 - current_torus
    if remaining_potential <= FLOAT_EPSILON:
        return 0.0
    
    # Toroidal flow development depends on phi resonance and pattern coherence
    # This is a natural synergistic relationship
    synergy_factor = 0.3 + 0.35 * soul_spark.phi_resonance + 0.35 * soul_spark.pattern_coherence
    
    # Phase-dependent modulation - natural development cycles
    phase_modifier = 0.6 + 0.4 * sin(PI * (current_phase + 2) / 4)
    
    # Growth follows a hyperbolic tangent curve - natural systems approach limits smoothly
    normalized_intensity = intensity * synergy_factor * phase_modifier
    gain = min(max_gain, remaining_potential * tanh(normalized_intensity))
    
    # Apply enhancement
    soul_spark.toroidal_flow_strength = min(1.0, current_torus + gain)
    
    return gain

def _enhance_harmony(soul_spark: SoulSpark, current_phase: int, intensity: float, max_gain: float) -> float:
    """
    Develops harmonic integration between all soul components.
    Returns the amount of enhancement applied (0-1 scale).
    """
    current_harmony = soul_spark.harmony
    
    # Natural ceiling effect
    remaining_potential = 1.0 - current_harmony
    if remaining_potential <= FLOAT_EPSILON:
        return 0.0
    
    # Harmony emerges from balanced integration of all other factors
    # Natural emergence principle - the whole is more than sum of parts
    integrated_factors = (
        soul_spark.phi_resonance +
        soul_spark.pattern_coherence +
        soul_spark.toroidal_flow_strength
    ) / 3.0
    
    # Phase-dependent modulation
    phase_modifier = 0.5 + 0.5 * cos(PI * (current_phase + 1) / 4)
    
    # Emergence curve - sigmoid-like natural growth
    normalized_intensity = intensity * integrated_factors * phase_modifier
    gain = min(max_gain, remaining_potential * (1.0 - exp(-normalized_intensity)))
    
    # Apply enhancement
    soul_spark.harmony = min(1.0, current_harmony + gain)
    
    return gain

def _increase_soul_energy(soul_spark: SoulSpark, harmony_gain: float, phase: int) -> float:
    """
    Increases soul energy based on harmony gain - natural energy generation.
    Returns the amount of energy added (SEU).
    """
    # Energy generation scales with harmony improvement
    # More harmony = more efficient energy utilization and generation
    energy_multiplier = 30000.0 + 10000.0 * phase
    energy_gain = harmony_gain * energy_multiplier
    
    # Apply energy increase
    old_energy = soul_spark.energy
    soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, old_energy + energy_gain)
    
    return soul_spark.energy - old_energy

# --- Main Function ---

def perform_spark_harmonization(
    soul_spark: SoulSpark,
    intensity: float = 0.85,
    duration_factor: float = 1.0,
    iterations: int | None = None  # Updated type hint to allow None
) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs natural harmonization of the soul spark through progressive phases.
    Each phase focuses on specific aspects of harmonization, following natural principles.
    
    Args:
        soul_spark: The SoulSpark object to harmonize
        intensity: How strongly to apply harmonization (0.1-1.0)
        duration_factor: Multiplier for process duration (0.1-2.0)
        iterations: Optional parameter to override the number of iterations (if None, uses duration_factor)
        
    Returns:
        Tuple with modified SoulSpark and metrics dictionary
    """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("soul_spark invalid.")
    if not 0.1 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.1 and 1.0")
    if not 0.1 <= duration_factor <= 2.0:
        raise ValueError("duration_factor must be between 0.1 and 2.0")
    if iterations is not None and not isinstance(iterations, int):
        raise TypeError("iterations must be an integer or None")
    
    spark_id = soul_spark.spark_id
    
    # Use iterations if provided, otherwise calculate based on duration_factor
    total_iterations = iterations if iterations is not None else int(244 * duration_factor)
    logger.info(f"--- Starting Spark Harmonization for {spark_id} ({total_iterations} iterations) ---")
    
    # Store initial state
    initial_state = {
        'stability_su': soul_spark.stability,
        'coherence_cu': soul_spark.coherence,
        'energy_seu': soul_spark.energy,
        'phi_resonance': soul_spark.phi_resonance,
        'pattern_coherence': soul_spark.pattern_coherence,
        'harmony': soul_spark.harmony,
        'toroidal_flow_strength': soul_spark.toroidal_flow_strength
    }
    
    logger.info(f"  Initial Harmonization State: S={initial_state['stability_su']:.1f}, "
                f"C={initial_state['coherence_cu']:.1f}, E={initial_state['energy_seu']:.1f}, "
                f"P.Coh={initial_state['pattern_coherence']:.3f}, PhiRes={initial_state['phi_resonance']:.3f}, "
                f"Torus={initial_state['toroidal_flow_strength']:.3f}")
    
    try:
        # --- Required attributes check ---
        required_attrs = [
            'phi_resonance', 'pattern_coherence', 'harmony', 
            'toroidal_flow_strength', 'frequency_signature'
        ]
        missing = [attr for attr in required_attrs if not hasattr(soul_spark, attr)]
        if missing:
            raise AttributeError(f"Soul missing required attributes: {missing}")
        
        # === Harmonization Process in Natural Phases ===
        # Each phase represents a natural developmental stage
        
        # --- Phase 1: Initial Phase Coherence and Pattern Development ---
        logger.debug("Phase 1: Initial Phase Coherence and Pattern Development")
        
        # Start with phase coherence to establish foundational order
        for i in range(5):
            # Apply phase coherence optimization with increasing intensity
            phase_factor = 0.05 + (i * 0.02)
            if hasattr(soul_spark, '_optimize_phase_coherence'):
                soul_spark._optimize_phase_coherence(phase_factor)
            else:
                _optimize_phase_coherence(soul_spark, phase_factor)
            
            # Natural pattern development with moderate intensity
            pattern_gain = _enhance_pattern_coherence(soul_spark, 1, 0.5 * intensity, 0.15)
            
            # Begin toroidal flow development at low intensity
            torus_gain = _develop_toroidal_flow(soul_spark, 1, 0.2 * intensity, 0.03)
            
            # Update derived scores to reflect changes
            soul_spark.update_state()
        
        logger.info(f"  Phase 1 Complete: S={soul_spark.stability:.1f} "
                   f"({soul_spark.stability - initial_state['stability_su']:+.1f}), "
                   f"C={soul_spark.coherence:.1f} "
                   f"({soul_spark.coherence - initial_state['coherence_cu']:+.1f}), "
                   f"P.Coh={soul_spark.pattern_coherence:.3f} "
                   f"({soul_spark.pattern_coherence - initial_state['pattern_coherence']:+.3f}), "
                   f"Torus={soul_spark.toroidal_flow_strength:.3f}")
        
        # Record state after phase 1
        phase1_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'phi_resonance': soul_spark.phi_resonance,
            'pattern_coherence': soul_spark.pattern_coherence,
            'toroidal_flow_strength': soul_spark.toroidal_flow_strength
        }
        
        # --- Phase 2: Phi Resonance Development ---
        logger.debug("Phase 2: Phi Resonance Development")
        
        for i in range(10):
            # Phi resonance is the focus in this phase
            phi_gain = _enhance_phi_resonance(soul_spark, 2, 0.7 * intensity, 0.1)
            
            # Continue pattern enhancement with increased intensity
            pattern_gain = _enhance_pattern_coherence(soul_spark, 2, 0.4 * intensity, 0.05)
            
            # Continue toroidal flow development
            torus_gain = _develop_toroidal_flow(soul_spark, 2, 0.6 * intensity, 0.05)
            
            # Update derived scores
            soul_spark.update_state()
        
        logger.info(f"  Phase 2 Complete: S={soul_spark.stability:.1f} "
                   f"({soul_spark.stability - phase1_state['stability_su']:+.1f}), "
                   f"C={soul_spark.coherence:.1f} "
                   f"({soul_spark.coherence - phase1_state['coherence_cu']:+.1f}), "
                   f"PhiRes={soul_spark.phi_resonance:.3f} "
                   f"({soul_spark.phi_resonance - phase1_state['phi_resonance']:+.3f}), "
                   f"Torus={soul_spark.toroidal_flow_strength:.3f} "
                   f"({soul_spark.toroidal_flow_strength - phase1_state['toroidal_flow_strength']:+.3f})")
        
        # Record state after phase 2
        phase2_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'phi_resonance': soul_spark.phi_resonance,
            'pattern_coherence': soul_spark.pattern_coherence,
            'harmony': soul_spark.harmony,
            'toroidal_flow_strength': soul_spark.toroidal_flow_strength
        }
        
        # --- Phase 3: Harmonic Integration ---
        logger.debug("Phase 3: Harmonic Integration")
        
        for i in range(17):
            # Focus on harmony development
            harmony_gain = _enhance_harmony(soul_spark, 3, 0.8 * intensity, 0.1)
            
            # Integrate all factors
            phi_gain = _enhance_phi_resonance(soul_spark, 3, 0.5 * intensity, 0.04)
            pattern_gain = _enhance_pattern_coherence(soul_spark, 3, 0.5 * intensity, 0.04)
            torus_gain = _develop_toroidal_flow(soul_spark, 3, 0.7 * intensity, 0.1)
            
            # Generate energy from harmony development
            energy_gain = _increase_soul_energy(soul_spark, harmony_gain, 3)
            
            # Update derived scores
            soul_spark.update_state()
        
        logger.info(f"  Phase 3 Complete: S={soul_spark.stability:.1f} "
                   f"({soul_spark.stability - phase2_state['stability_su']:+.1f}), "
                   f"C={soul_spark.coherence:.1f} "
                   f"({soul_spark.coherence - phase2_state['coherence_cu']:+.1f}), "
                   f"Harmony={soul_spark.harmony:.3f} "
                   f"({soul_spark.harmony - phase2_state['harmony']:+.3f}), "
                   f"Freq={soul_spark.frequency:.1f}Hz, "
                   f"Torus={soul_spark.toroidal_flow_strength:.3f} "
                   f"({soul_spark.toroidal_flow_strength - phase2_state['toroidal_flow_strength']:+.3f})")
        
        # Record state after phase 3
        phase3_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'energy_seu': soul_spark.energy
        }
        
        # --- Phase 4: Energy Integration and Final Balancing ---
        logger.debug("Phase 4: Energy Integration and Final Balancing")
        
        for i in range(7):
            # Finalize toroidal flow for energy stabilization
            torus_gain = _develop_toroidal_flow(soul_spark, 4, 0.9 * intensity, 0.25)
            
            # Establish energy balance through harmony
            harmony_gain = _enhance_harmony(soul_spark, 4, 0.7 * intensity, 0.1)
            
            # Generate substantial energy
            energy_gain = _increase_soul_energy(soul_spark, harmony_gain, 4)
            
            # Update derived scores
            soul_spark.update_state()
        
        logger.info(f"  Phase 4 Complete: S={soul_spark.stability:.1f} "
                   f"({soul_spark.stability - phase3_state['stability_su']:+.1f}), "
                   f"C={soul_spark.coherence:.1f} "
                   f"({soul_spark.coherence - phase3_state['coherence_cu']:+.1f}), "
                   f"E={soul_spark.energy:.1f} "
                   f"({soul_spark.energy - phase3_state['energy_seu']:+.1f}), "
                   f"Torus={soul_spark.toroidal_flow_strength:.3f}")
        
        # --- Final Phase: Natural Equilibrium Attainment ---
        # This phase allows the system to find its natural balance point
        
        # Set journey completion flag
        setattr(soul_spark, FLAG_SPARK_HARMONIZED, True)
        setattr(soul_spark, FLAG_READY_FOR_GUFF, True)
        
        # Add memory echo
        if hasattr(soul_spark, 'add_memory_echo'):
            soul_spark.add_memory_echo(f"Harmonized initial spark pattern. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
        
        # --- Compile metrics ---
        end_metrics = {
            'action': 'spark_harmonization',
            'soul_id': spark_id,
            'initial_state': initial_state,
            'final_state': {
                'stability_su': soul_spark.stability,
                'coherence_cu': soul_spark.coherence,
                'energy_seu': soul_spark.energy,
                'phi_resonance': soul_spark.phi_resonance,
                'pattern_coherence': soul_spark.pattern_coherence,
                'harmony': soul_spark.harmony,
                'toroidal_flow_strength': soul_spark.toroidal_flow_strength
            },
            'stability_gain': soul_spark.stability - initial_state['stability_su'],
            'coherence_gain': soul_spark.coherence - initial_state['coherence_cu'],
            'energy_gain': soul_spark.energy - initial_state['energy_seu'],
            'pattern_coherence_gain': soul_spark.pattern_coherence - initial_state['pattern_coherence'],
            'phi_resonance_gain': soul_spark.phi_resonance - initial_state['phi_resonance'],
            'harmony_gain': soul_spark.harmony - initial_state['harmony'],
            'toroidal_flow_gain': soul_spark.toroidal_flow_strength - initial_state['toroidal_flow_strength'],
            'success': True
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('spark_harmonization_summary', end_metrics)
        
        logger.info(f"--- Spark Harmonization Complete ---")
        return soul_spark, end_metrics
        
    except Exception as e:
        logger.error(f"Spark Harmonization failed: {e}", exc_info=True)
        error_metrics = {
            'action': 'spark_harmonization',
            'soul_id': spark_id,
            'initial_state': initial_state,
            'error': str(e),
            'success': False
        }
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('spark_harmonization_summary', error_metrics)
        
        raise RuntimeError(f"Spark Harmonization failed: {e}") from e


# --- END OF FILE src/stage_1/soul_formation/spark_harmonization.py ---