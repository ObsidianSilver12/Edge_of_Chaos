# --- START OF FILE src/stage_1/soul_formation/spark_harmonization.py ---

"""
Spark Harmonization Module (V4.3.9 - Active Phase/Harmonic Refinement, PEP8)

Implements the initial self-harmonization phase for a newly emerged SoulSpark.
This stage iteratively refines internal coherence, stability factors, and energy
based on principles like toroidal flow (pattern coherence) and harmonic/phase
resonance stabilization. Modifies SoulSpark attributes directly.
Adheres strictly to PEP 8 formatting.
"""

import numpy as np
import sys
import time
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
from math import sqrt, pi as PI, exp, atan2 # Use atan2 for mean phase angle

# --- Constants Import ---
try:
    from constants.constants import * 
    
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: constants.py failed import in spark_harmonization.py")
    # Define minimal fallbacks
    HARMONIZATION_ITERATIONS = 144; HARMONIZATION_PATTERN_COHERENCE_RATE = 0.003
    HARMONIZATION_PHI_RESONANCE_RATE = 0.002; HARMONIZATION_HARMONY_RATE = 0.0015
    HARMONIZATION_ENERGY_GAIN_RATE = 0.1; MAX_SOUL_ENERGY_SEU = 1e6
    MAX_STABILITY_SU = 100.0; MAX_COHERENCE_CU = 100.0
    LOG_LEVEL = logging.INFO; LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    FLOAT_EPSILON = 1e-9; PHI = 1.618; GOLDEN_RATIO = PHI
    # Fallbacks for NEW constants
    HARMONIZATION_PHASE_ADJUST_RATE = 0.01
    HARMONIZATION_HARMONIC_ADJUST_RATE = 0.005
    HARMONIZATION_CIRC_VAR_THRESHOLD = 0.1
    HARMONIZATION_HARM_DEV_THRESHOLD = 0.05
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# ... (logger configuration as before) ...
if not logger.handlers:
    try: logger.setLevel(LOG_LEVEL)
    except NameError: logger.warning("LOG_LEVEL constant not found."); logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter(LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(log_formatter); logger.addHandler(ch)

# --- Metrics Tracking ---
# ... (metrics import/placeholder as before) ...
class MetricsPlaceholder:
    @staticmethod
    def record_metrics(*args, **kwargs): pass
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
    logger.debug("Metrics tracking loaded in spark_harmonization.")
except ImportError:
    logger.error("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    metrics = MetricsPlaceholder()
except Exception as e:
    logger.error(f"Unexpected error importing metrics: {e}")
    METRICS_AVAILABLE = False
    metrics = MetricsPlaceholder()

# --- Helper: Calculate Circular Variance ---
def _calculate_circular_variance(phases_array: np.ndarray) -> float:
    """Calculates circular variance (0=sync, 1=uniform)."""
    if phases_array is None or phases_array.size < 2:
        return 1.0 # Max variance if no/one phase
    mean_cos = np.mean(np.cos(phases_array))
    mean_sin = np.mean(np.sin(phases_array))
    # Resultant vector length R
    r_len = sqrt(max(0.0, mean_cos**2 + mean_sin**2))
    # Variance = 1 - R
    return 1.0 - r_len

# --- Helper: Calculate Harmonic Deviation ---
def _calculate_harmonic_deviation(harmonics: List[float], base_freq: float) -> float:
    """Calculates average deviation of harmonics from ideal ratios."""
    if not harmonics or base_freq <= FLOAT_EPSILON:
        return 1.0 # Max deviation if no harmonics/base_freq
    ratios = [h / base_freq for h in harmonics if h > FLOAT_EPSILON]
    if not ratios: return 1.0

    deviations = []
    for r in ratios:
        if r <= 0: continue
        int_dev = min([abs(r - n) for n in range(1, 6)]) # Check against integers 1-5
        phi_dev = min([abs(r - PHI**n) for n in [1, -1, 2, -2]]) # Check phi, 1/phi, phi^2, 1/phi^2
        deviations.append(min(int_dev, phi_dev))
    return np.mean(deviations) if deviations else 1.0


def perform_spark_harmonization(soul_spark: SoulSpark, iterations: int = HARMONIZATION_ITERATIONS) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Harmonizes a newly emerged soul to increase stability, coherence through
    pattern coherence, phi resonance, toroidal flow, and energy optimization.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if iterations <= 0: raise ValueError("iterations must be positive.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Spark Harmonization for {spark_id} ({iterations} iterations) ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    
    # Track detailed metrics for each phase
    detailed_metrics = {
        'phase1': {'pattern_gain': 0.0},
        'phase2': {'phi_gain': 0.0},
        'phase3': {'harmony_gain': 0.0, 'freq_change': 0.0},
        'phase4': {'energy_gain': 0.0, 'torus_gain': 0.0},
        'phase5': {'coherence_gain': 0.0, 'stability_gain': 0.0}
    }
    
    # Store Initial State
    initial_state = {
        'stability': soul_spark.stability, 
        'coherence': soul_spark.coherence,
        'energy': soul_spark.energy, 
        'pattern_coherence': soul_spark.pattern_coherence,
        'phi_resonance': soul_spark.phi_resonance, 
        'harmony': soul_spark.harmony,
        'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.05)
    }
    
    logger.info(f"  Initial Harmonization State: S={initial_state['stability']:.1f}, "
                f"C={initial_state['coherence']:.1f}, E={initial_state['energy']:.1f}, "
                f"P.Coh={initial_state['pattern_coherence']:.3f}, PhiRes={initial_state['phi_resonance']:.3f}, "
                f"Torus={initial_state['toroidal_flow_strength']:.3f}")

    try:
        # --- Perform Harmonization Process ---
        # Phase 1: Establish Base Pattern (10% of iterations)
        phase1_steps = max(20, int(iterations * 0.1))
        detailed_metrics['phase1'] = _harmonize_phase1_establish_pattern(soul_spark, phase1_steps)
        
        # Log after phase 1
        phase1_state = {
            'stability': soul_spark.stability, 
            'coherence': soul_spark.coherence,
            'pattern_coherence': soul_spark.pattern_coherence,
            'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        }
        logger.info(f"  Phase 1 Complete: S={phase1_state['stability']:.1f} (+{phase1_state['stability']-initial_state['stability']:.1f}), "
                   f"C={phase1_state['coherence']:.1f} (+{phase1_state['coherence']-initial_state['coherence']:.1f}), "
                   f"P.Coh={phase1_state['pattern_coherence']:.3f} (+{detailed_metrics['phase1']['pattern_gain']:.3f}), "
                   f"Torus={phase1_state['toroidal_flow_strength']:.3f}")
        
        # Phase 2: Phi Resonance Development (20% of iterations)
        phase2_steps = max(50, int(iterations * 0.2))
        detailed_metrics['phase2'] = _harmonize_phase2_develop_phi(soul_spark, phase2_steps)
        
        # Log after phase 2
        phase2_state = {
            'stability': soul_spark.stability, 
            'coherence': soul_spark.coherence,
            'phi_resonance': soul_spark.phi_resonance,
            'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        }
        logger.info(f"  Phase 2 Complete: S={phase2_state['stability']:.1f} (+{phase2_state['stability']-phase1_state['stability']:.1f}), "
                   f"C={phase2_state['coherence']:.1f} (+{phase2_state['coherence']-phase1_state['coherence']:.1f}), "
                   f"PhiRes={phase2_state['phi_resonance']:.3f} (+{detailed_metrics['phase2']['phi_gain']:.3f}), "
                   f"Torus={phase2_state['toroidal_flow_strength']:.3f} (+{detailed_metrics['phase2']['torus_gain']:.3f})")
        
        # Phase 3: Harmonic Alignment (20% of iterations)
        phase3_steps = max(50, int(iterations * 0.2))
        detailed_metrics['phase3'] = _harmonize_phase3_align_harmonics(soul_spark, phase3_steps)
        
        # Log after phase 3
        phase3_state = {
            'stability': soul_spark.stability, 
            'coherence': soul_spark.coherence,
            'harmony': soul_spark.harmony,
            'frequency': soul_spark.frequency,
            'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        }
        logger.info(f"  Phase 3 Complete: S={phase3_state['stability']:.1f} (+{phase3_state['stability']-phase2_state['stability']:.1f}), "
                   f"C={phase3_state['coherence']:.1f} (+{phase3_state['coherence']-phase2_state['coherence']:.1f}), "
                   f"Harmony={phase3_state['harmony']:.3f} (+{detailed_metrics['phase3']['harmony_gain']:.3f}), "
                   f"Freq={phase3_state['frequency']:.1f}Hz, "
                   f"Torus={phase3_state['toroidal_flow_strength']:.3f} (+{detailed_metrics['phase3']['torus_gain']:.3f})")
        
        # Phase 4: Energy and Toroidal Flow (30% of iterations)
        phase4_steps = max(50, int(iterations * 0.3))
        detailed_metrics['phase4'] = _harmonize_phase4_energy_torus(soul_spark, phase4_steps)
        
        # Log after phase 4
        phase4_state = {
            'stability': soul_spark.stability, 
            'coherence': soul_spark.coherence,
            'energy': soul_spark.energy,
            'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        }
        logger.info(f"  Phase 4 Complete: S={phase4_state['stability']:.1f} (+{phase4_state['stability']-phase3_state['stability']:.1f}), "
                   f"C={phase4_state['coherence']:.1f} (+{phase4_state['coherence']-phase3_state['coherence']:.1f}), "
                   f"E={phase4_state['energy']:.1f} (+{detailed_metrics['phase4']['energy_gain']:.1f}), "
                   f"Torus={phase4_state['toroidal_flow_strength']:.3f} (+{detailed_metrics['phase4']['torus_gain']:.3f})")
        
        # Final phase: Coherence Optimization (20% of iterations)
        remaining_steps = iterations - (phase1_steps + phase2_steps + phase3_steps + phase4_steps)
        detailed_metrics['phase5'] = _harmonize_phase5_optimize_coherence(soul_spark, remaining_steps)
        
        # Final update to consolidate changes
        if hasattr(soul_spark, 'update_state'): 
            logger.debug("  Performing final state update")
            soul_spark.update_state()
            
        if hasattr(soul_spark, 'add_memory_echo'):
            soul_spark.add_memory_echo(f"Harmonized spark patterns. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}, Torus:{getattr(soul_spark, 'toroidal_flow_strength', 0.05):.2f}")

        # --- Finalize and Return ---
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'stability': soul_spark.stability, 
            'coherence': soul_spark.coherence,
            'energy': soul_spark.energy, 
            'pattern_coherence': soul_spark.pattern_coherence,
            'phi_resonance': soul_spark.phi_resonance, 
            'harmony': soul_spark.harmony,
            'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        }
        
        metrics = {
            'action': 'spark_harmonization', 
            'soul_id': spark_id,
            'duration_sec': (end_time_dt - start_time_dt).total_seconds(),
            'iterations_run': iterations,
            'initial_stability_su': float(initial_state['stability']),
            'final_stability_su': float(final_state['stability']),
            'stability_gain_su': float(final_state['stability'] - initial_state['stability']),
            'initial_coherence_cu': float(initial_state['coherence']),
            'final_coherence_cu': float(final_state['coherence']),
            'coherence_gain_cu': float(final_state['coherence'] - initial_state['coherence']),
            'initial_energy_seu': float(initial_state['energy']),
            'final_energy_seu': float(final_state['energy']),
            'energy_gain_seu': float(final_state['energy'] - initial_state['energy']),
            'initial_pattern_coherence': float(initial_state['pattern_coherence']),
            'final_pattern_coherence': float(final_state['pattern_coherence']),
            'initial_phi_resonance': float(initial_state['phi_resonance']),
            'final_phi_resonance': float(final_state['phi_resonance']),
            'initial_harmony': float(initial_state['harmony']),
            'final_harmony': float(final_state['harmony']),
            'initial_toroidal_flow': float(initial_state['toroidal_flow_strength']),
            'final_toroidal_flow': float(final_state['toroidal_flow_strength']),
            'toroidal_flow_gain': float(final_state['toroidal_flow_strength'] - initial_state['toroidal_flow_strength']),
            'detailed_metrics': detailed_metrics,
            'success': True,
        }

        logger.info(f"--- Spark Harmonization Complete for {spark_id}. ---")
        logger.info(f"  Final State: S={final_state['stability']:.1f}, "
                    f"C={final_state['coherence']:.1f}, E={final_state['energy']:.1f}, "
                    f"P.Coh={final_state['pattern_coherence']:.3f}, PhiRes={final_state['phi_resonance']:.3f}, "
                    f"Torus={final_state['toroidal_flow_strength']:.3f}")
        
        return soul_spark, metrics

    except Exception as e:
        logger.error(f"Error during spark harmonization for {spark_id}: {e}", exc_info=True)
        raise RuntimeError(f"Spark harmonization failed: {e}") from e

def _harmonize_phase1_establish_pattern(soul_spark: SoulSpark, steps: int) -> Dict[str, float]:
    """
    Establishes base pattern coherence and toroidal flow foundation.
    Returns detailed metrics for this phase.
    """
    logger.debug(f"Phase 1: Establish Pattern ({steps} steps)")
    metrics = {
        'pattern_gain': 0.0,
        'torus_gain': 0.0,
        'coherence_gain': 0.0,
        'stability_gain': 0.0
    }
    
    initial_pattern = soul_spark.pattern_coherence
    initial_coherence = soul_spark.coherence
    initial_stability = soul_spark.stability
    initial_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
    
    # Start with larger pattern coherence increments
    base_pattern_increment = 0.008  # Substantially increased increment
    
    for step in range(steps):
        # Calculate adaptive increment - larger at the beginning
        decay_factor = 1.0 - (step / steps)**0.5  # Non-linear decay
        pattern_increment = base_pattern_increment * (1.0 + 2.0 * decay_factor)
        
        # Apply increment with diminishing returns check
        current_pattern = soul_spark.pattern_coherence
        new_pattern = min(0.85, current_pattern + pattern_increment)  # Cap at 0.85
        actual_gain = new_pattern - current_pattern
        
        # Apply the change
        soul_spark.pattern_coherence = float(new_pattern)
        
        # Initialize and enhance toroidal flow
        torus_strength = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        torus_increment = 0.002 * (1.0 + decay_factor)  # Start building torus
        new_torus = min(0.3, torus_strength + torus_increment)  # Early cap at 0.3
        setattr(soul_spark, 'toroidal_flow_strength', float(new_torus))
        
        # Small energy gain (proportional to pattern coherence gain)
        energy_gain = HARMONIZATION_ENERGY_GAIN_RATE * actual_gain * 20.0
        soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_gain)
        
        # Update state to reflect changes (with lower frequency to avoid overhead)
        if step % 5 == 0 and hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            
            # Log significant improvements
            if step > 0 and step % 15 == 0:
                logger.debug(f"  P1 Step {step}/{steps}: Pattern={soul_spark.pattern_coherence:.3f}, "
                           f"Torus={getattr(soul_spark, 'toroidal_flow_strength', 0.05):.3f}, "
                           f"C={soul_spark.coherence:.1f}")
    
    # Update final metrics
    metrics['pattern_gain'] = soul_spark.pattern_coherence - initial_pattern
    metrics['torus_gain'] = getattr(soul_spark, 'toroidal_flow_strength', 0.05) - initial_torus
    metrics['coherence_gain'] = soul_spark.coherence - initial_coherence
    metrics['stability_gain'] = soul_spark.stability - initial_stability
    
    return metrics

def _harmonize_phase2_develop_phi(soul_spark: SoulSpark, steps: int) -> Dict[str, float]:
    """
    Enhances phi resonance and continues building toroidal flow.
    Returns detailed metrics for this phase.
    """
    logger.debug(f"Phase 2: Develop Phi Resonance ({steps} steps)")
    metrics = {
        'phi_gain': 0.0,
        'torus_gain': 0.0,
        'coherence_gain': 0.0,
        'stability_gain': 0.0
    }
    
    initial_phi = soul_spark.phi_resonance
    initial_coherence = soul_spark.coherence
    initial_stability = soul_spark.stability
    initial_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
    
    # Larger phi resonance increments
    base_phi_increment = 0.006  # Increased from typical ~0.001
    max_phi_target = 0.90  # Higher target
    
    for step in range(steps):
        # Adaptive increment with decay
        decay_factor = 1.0 - (step / steps)**0.7  # Slower decay than phase 1
        phi_increment = base_phi_increment * (1.0 + 1.5 * decay_factor)
        
        # Apply phi resonance adjustment
        current_phi = soul_spark.phi_resonance
        distance_to_max = max(0.0, max_phi_target - current_phi)
        effective_increment = min(phi_increment, distance_to_max * 0.1)  # Prevent overshooting
        new_phi = current_phi + effective_increment
        
        # Apply the change
        soul_spark.phi_resonance = float(new_phi)
        
        # ENHANCED TOROIDAL FLOW - more aggressive development
        torus_strength = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        # Link torus growth to phi resonance
        torus_rate = 0.003 * (1.0 + (current_phi - 0.5) * 2.0)  # Faster as phi improves
        new_torus = min(0.5, torus_strength + torus_rate)  # Mid-phase cap at 0.5
        setattr(soul_spark, 'toroidal_flow_strength', float(new_torus))
        
        # Energy boost with phi resonance improvement
        energy_gain = HARMONIZATION_ENERGY_GAIN_RATE * effective_increment * 250.0
        soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_gain)
        
        # Update state to reflect changes
        if step % 5 == 0 and hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            
            # Log significant improvements
            if step > 0 and step % 15 == 0:
                logger.debug(f"  P2 Step {step}/{steps}: Phi={soul_spark.phi_resonance:.3f}, "
                           f"Torus={getattr(soul_spark, 'toroidal_flow_strength', 0.05):.3f}, "
                           f"C={soul_spark.coherence:.1f}")
    
    # Update final metrics
    metrics['phi_gain'] = soul_spark.phi_resonance - initial_phi
    metrics['torus_gain'] = getattr(soul_spark, 'toroidal_flow_strength', 0.05) - initial_torus
    metrics['coherence_gain'] = soul_spark.coherence - initial_coherence
    metrics['stability_gain'] = soul_spark.stability - initial_stability
    
    return metrics

def _harmonize_phase3_align_harmonics(soul_spark: SoulSpark, steps: int) -> Dict[str, float]:
    """
    Aligns harmonics and develops harmony, while continuing toroidal flow enhancement.
    Returns detailed metrics for this phase.
    """
    logger.debug(f"Phase 3: Align Harmonics ({steps} steps)")
    metrics = {
        'harmony_gain': 0.0,
        'freq_change': 0.0,
        'torus_gain': 0.0,
        'coherence_gain': 0.0,
        'stability_gain': 0.0
    }
    
    initial_harmony = soul_spark.harmony
    initial_freq = soul_spark.frequency
    initial_coherence = soul_spark.coherence
    initial_stability = soul_spark.stability
    initial_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
    
    # Base frequency adjustments toward favorable harmonics
    favorable_freqs = [432.0, 528.0, 639.0, 741.0, 852.0]  # Favorable base frequencies
    
    for step in range(steps):
        # Find closest favorable frequency
        current_freq = soul_spark.frequency
        closest_freq = min(favorable_freqs, key=lambda f: abs(f - current_freq))
        freq_diff = closest_freq - current_freq
        
        # Calculate nudge with progressive strength
        progress_factor = step / steps  # Increases over time
        base_nudge_factor = 0.003 + 0.01 * progress_factor  # Grows from 0.003 to 0.013
        nudge_amount = freq_diff * base_nudge_factor
        
        # Apply frequency nudge
        new_freq = current_freq + nudge_amount
        soul_spark.frequency = max(10.0, float(new_freq))
        
        # Regenerate harmonics for significant changes
        if abs(nudge_amount) > current_freq * 0.005:
            if hasattr(soul_spark, '_validate_or_init_frequency_structure'):
                soul_spark._validate_or_init_frequency_structure()
        
        # Harmony improvement
        harmony_gain = 0.004 * (1.0 + progress_factor)
        soul_spark.harmony = min(0.9, soul_spark.harmony + harmony_gain)
        
        # ENHANCED TOROIDAL FLOW - linked to harmonic development
        torus_strength = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        # Faster growth, linked to harmony
        torus_rate = 0.004 * (1.0 + progress_factor) * (0.5 + soul_spark.harmony)
        new_torus = min(0.7, torus_strength + torus_rate)  # Cap at 0.7 for this phase
        setattr(soul_spark, 'toroidal_flow_strength', float(new_torus))
        
        # Energy boost
        energy_gain = HARMONIZATION_ENERGY_GAIN_RATE * (0.8 + progress_factor)
        soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_gain)
        
        # Update state with higher frequency due to importance of harmonic alignment
        if step % 3 == 0 and hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            
            # Log significant improvements
            if step > 0 and step % 15 == 0:
                logger.debug(f"  P3 Step {step}/{steps}: Freq={soul_spark.frequency:.1f}Hz, "
                           f"Harmony={soul_spark.harmony:.3f}, "
                           f"Torus={getattr(soul_spark, 'toroidal_flow_strength', 0.05):.3f}, "
                           f"C={soul_spark.coherence:.1f}")
    
    # Update final metrics
    metrics['harmony_gain'] = soul_spark.harmony - initial_harmony
    metrics['freq_change'] = soul_spark.frequency - initial_freq
    metrics['torus_gain'] = getattr(soul_spark, 'toroidal_flow_strength', 0.05) - initial_torus
    metrics['coherence_gain'] = soul_spark.coherence - initial_coherence
    metrics['stability_gain'] = soul_spark.stability - initial_stability
    
    return metrics

def _harmonize_phase4_energy_torus(soul_spark: SoulSpark, steps: int) -> Dict[str, float]:
    """
    Maximizes toroidal flow and energy distribution.
    Returns detailed metrics for this phase.
    """
    logger.debug(f"Phase 4: Energy Distribution & Toroidal Flow ({steps} steps)")
    metrics = {
        'energy_gain': 0.0,
        'torus_gain': 0.0,
        'coherence_gain': 0.0,
        'stability_gain': 0.0
    }
    
    initial_energy = soul_spark.energy
    initial_coherence = soul_spark.coherence
    initial_stability = soul_spark.stability
    initial_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
    
    # Target energy level (proportion of maximum)
    target_energy_factor = 0.15  # Target 15% of maximum energy
    
    # CRITICAL: This phase focuses heavily on toroidal flow maximization
    for step in range(steps):
        # Calculate target energy
        target_energy = MAX_SOUL_ENERGY_SEU * target_energy_factor
        energy_diff = target_energy - soul_spark.energy
        
        # Adaptive energy adjustment
        progress_factor = step / steps
        energy_adjustment_rate = 0.015 + 0.05 * progress_factor  # Grows from 0.015 to 0.065
        energy_change = energy_diff * energy_adjustment_rate
        
        # Apply energy change
        soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, max(0.0, soul_spark.energy + energy_change))
        
        # MAXIMIZE TOROIDAL FLOW - highest priority in this phase
        current_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        
        # Aggressive toroidal flow development with accelerating growth
        base_rate = 0.006  # Much more aggressive base rate
        acceleration = 1.0 + 3.0 * progress_factor  # Strong acceleration over time
        synergy_factor = 1.0 + (soul_spark.phi_resonance + soul_spark.harmony) / 2.0  # Synergistic effect
        
        torus_growth = base_rate * acceleration * synergy_factor
        new_torus = min(0.95, current_torus + torus_growth)  # Allow up to 0.95 (near maximum)
        
        # Apply the change with detailed logging for significant jumps
        if abs(new_torus - current_torus) > 0.01:
            logger.debug(f"  Torus growth: {current_torus:.3f} -> {new_torus:.3f} (+{new_torus-current_torus:.3f})")
            
        setattr(soul_spark, 'toroidal_flow_strength', float(new_torus))
        
        # Update state frequently to properly capture torus effects
        if step % 3 == 0 and hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            
            # Log significant improvements
            if step > 0 and step % 10 == 0:
                logger.debug(f"  P4 Step {step}/{steps}: Energy={soul_spark.energy:.1f}, "
                           f"Torus={getattr(soul_spark, 'toroidal_flow_strength', 0.05):.3f}, "
                           f"S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
    
    # Update final metrics
    metrics['energy_gain'] = soul_spark.energy - initial_energy
    metrics['torus_gain'] = getattr(soul_spark, 'toroidal_flow_strength', 0.05) - initial_torus
    metrics['coherence_gain'] = soul_spark.coherence - initial_coherence
    metrics['stability_gain'] = soul_spark.stability - initial_stability
    
    return metrics

def _harmonize_phase5_optimize_coherence(soul_spark: SoulSpark, steps: int) -> Dict[str, float]:
    """
    Final coherence optimization with all factors.
    Returns detailed metrics for this phase.
    """
    logger.debug(f"Phase 5: Coherence Optimization ({steps} steps)")
    metrics = {
        'coherence_gain': 0.0,
        'stability_gain': 0.0,
        'pattern_gain': 0.0,
        'phi_gain': 0.0,
        'harmony_gain': 0.0,
        'torus_gain': 0.0
    }
    
    initial_coherence = soul_spark.coherence
    initial_stability = soul_spark.stability
    initial_pattern = soul_spark.pattern_coherence
    initial_phi = soul_spark.phi_resonance
    initial_harmony = soul_spark.harmony
    initial_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
    
    # This phase focuses explicitly on S/C improvements through all factors
    for step in range(steps):
        progress_factor = step / steps
        
        # --- Pattern coherence boost ---
        pattern_gain = 0.003 * (1.0 + progress_factor)
        soul_spark.pattern_coherence = min(0.95, soul_spark.pattern_coherence + pattern_gain)
        
        # --- Phi resonance boost ---
        phi_gain = 0.002 * (1.0 + progress_factor)
        soul_spark.phi_resonance = min(0.95, soul_spark.phi_resonance + phi_gain)
        
        # --- Harmony boost ---
        harmony_gain = 0.003 * (1.0 + progress_factor)
        soul_spark.harmony = min(0.95, soul_spark.harmony + harmony_gain)
        
        # --- Final toroidal flow optimization ---
        current_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.05)
        # Careful final adjustments - maintain what's been built but top up
        remaining_headroom = max(0.0, 0.99 - current_torus)
        torus_gain = remaining_headroom * 0.05 * (1.0 + 2.0 * progress_factor)
        new_torus = current_torus + torus_gain
        setattr(soul_spark, 'toroidal_flow_strength', float(new_torus))
        
        # --- Energy top-up ---
        energy_gain = HARMONIZATION_ENERGY_GAIN_RATE * 3.0  # Higher rate for final phase
        soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_gain)
        
        # --- Phase alignment strengthening ---
        if hasattr(soul_spark, '_optimize_phase_coherence'):
            # Progressive phase alignment - becomes more aggressive
            phase_factor = 0.07 + 0.18 * progress_factor  # Grows from 0.07 to 0.25
            soul_spark._optimize_phase_coherence(phase_factor)
        
        # --- Update state very frequently in final phase ---
        if step % 2 == 0 and hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            
            # Log progress at regular intervals
            if step > 0 and step % 5 == 0:
                logger.debug(f"  P5 Step {step}/{steps}: S={soul_spark.stability:.1f}, "
                           f"C={soul_spark.coherence:.1f}, Torus={getattr(soul_spark, 'toroidal_flow_strength', 0.05):.3f}")
    
    # Update final metrics
    metrics['coherence_gain'] = soul_spark.coherence - initial_coherence
    metrics['stability_gain'] = soul_spark.stability - initial_stability
    metrics['pattern_gain'] = soul_spark.pattern_coherence - initial_pattern
    metrics['phi_gain'] = soul_spark.phi_resonance - initial_phi
    metrics['harmony_gain'] = soul_spark.harmony - initial_harmony
    metrics['torus_gain'] = getattr(soul_spark, 'toroidal_flow_strength', 0.05) - initial_torus
    
    return metrics







# def perform_spark_harmonization(
#     soul_spark: SoulSpark,
#     iterations: int = HARMONIZATION_ITERATIONS
# ) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Performs initial self-harmonization with active phase/harmonic refinement.
#     """
#     global metrics # Ensure access to global metrics object
#     # --- Input Validation ---
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("Input must be a SoulSpark instance.")
#     if not isinstance(iterations, int) or iterations <= 0: raise ValueError("Iterations must be positive.")

#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     logger.info(f"--- Starting Spark Harmonization for {spark_id} ({iterations} iterations) ---")
#     start_time = time.time()
#     i = 0 # Initialize loop counter

#     # --- Record Initial State ---
#     initial_s = soul_spark.stability; initial_c = soul_spark.coherence; initial_e = soul_spark.energy
#     initial_p_coh = getattr(soul_spark, 'pattern_coherence', 0.0); initial_phi_res = getattr(soul_spark, 'phi_resonance', 0.0); initial_harmony = getattr(soul_spark, 'harmony', 0.0)
#     logger.info(f"  Initial Harmonization State: S={initial_s:.1f}, C={initial_c:.1f}, E={initial_e:.1f}, P.Coh={initial_p_coh:.3f}, PhiRes={initial_phi_res:.3f}")

#     # --- Harmonization Loop ---
#     try:
#         for i in range(iterations):
#             # --- 1. Structure Building (Pattern Coherence) ---
#             rate_p_coh = HARMONIZATION_PATTERN_COHERENCE_RATE
#             current_p_coh = getattr(soul_spark, 'pattern_coherence', 0.0)
#             increase_p_coh = rate_p_coh * (1.0 - current_p_coh**0.5) # Faster increase when low
#             soul_spark.pattern_coherence = min(1.0, current_p_coh + increase_p_coh)

#             # --- 2. Internal Resonance (Phi & Harmony Factors) ---
#             rate_phi = HARMONIZATION_PHI_RESONANCE_RATE
#             current_phi = getattr(soul_spark, 'phi_resonance', 0.0)
#             increase_phi = rate_phi * (1.0 - current_phi**0.5)
#             soul_spark.phi_resonance = min(1.0, current_phi + increase_phi)

#             rate_harm = HARMONIZATION_HARMONY_RATE
#             current_harm = getattr(soul_spark, 'harmony', 0.0)
#             increase_harm = rate_harm * (1.0 - current_harm**0.5)
#             soul_spark.harmony = min(1.0, current_harm + increase_harm)

#             # --- 3. *** Active Phase Alignment *** ---
#             phases = np.array(soul_spark.frequency_signature.get('phases', []))
#             if phases.size > 1:
#                 current_circ_var = _calculate_circular_variance(phases)
#                 if current_circ_var > HARMONIZATION_CIRC_VAR_THRESHOLD: # Only adjust if variance is high
#                     # Calculate mean angle (-pi to pi)
#                     mean_angle = atan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))
#                     # Calculate adjustment needed for each phase towards mean angle
#                     # Ensure shortest path adjustment (handle wrap-around)
#                     delta_angle = mean_angle - phases
#                     delta_angle = (delta_angle + PI) % (2 * PI) - PI # Wrap to [-pi, pi]
#                     # Apply adjustment scaled by variance and rate
#                     phase_adjustment = delta_angle * current_circ_var * HARMONIZATION_PHASE_ADJUST_RATE
#                     phases = (phases + phase_adjustment) % (2 * PI) # Apply and wrap
#                     soul_spark.frequency_signature['phases'] = phases.tolist()
#                     # logger.debug(f"   Phase Adj (Iter {i+1}): CircVar={current_circ_var:.3f} -> Adjusted towards {mean_angle:.3f}")

#             # --- 4. *** Active Harmonic Purity Adjustment *** ---
#             harmonics_list = soul_spark.harmonics
#             base_freq = soul_spark.frequency
#             if harmonics_list and base_freq > FLOAT_EPSILON:
#                 current_harm_dev = _calculate_harmonic_deviation(harmonics_list, base_freq)
#                 if current_harm_dev > HARMONIZATION_HARM_DEV_THRESHOLD: # Only adjust if deviation is high
#                     new_harmonics = []
#                     for h in harmonics_list:
#                         ratio = h / base_freq
#                         if ratio <= 0: new_harmonics.append(h); continue
#                         # Find nearest ideal ratio
#                         int_dists = [abs(ratio - n) for n in range(1, 6)]
#                         phi_dists = [abs(ratio - PHI**n) for n in [1, -1, 2, -2]]
#                         all_dists = int_dists + phi_dists
#                         best_dist_idx = np.argmin(all_dists)
#                         if best_dist_idx < len(int_dists): # Integer match
#                             nearest_ideal_ratio = range(1, 6)[best_dist_idx]
#                         else: # Phi match
#                             phi_powers = [PHI**n for n in [1, -1, 2, -2]]
#                             nearest_ideal_ratio = phi_powers[best_dist_idx - len(int_dists)]

#                         target_harmonic = base_freq * nearest_ideal_ratio
#                         # Nudge harmonic towards target, scaled by deviation and rate
#                         adjustment = (target_harmonic - h) * current_harm_dev * HARMONIZATION_HARMONIC_ADJUST_RATE
#                         new_harmonics.append(h + adjustment)
#                     # Update harmonics lists (both simple and in signature)
#                     soul_spark.harmonics = sorted(new_harmonics)
#                     soul_spark.frequency_signature['frequencies'] = sorted(new_harmonics)
#                     # logger.debug(f"   Harmonic Adj (Iter {i+1}): AvgDev={current_harm_dev:.3f} -> Adjusted")


#             # --- 5. Energy Gain from Internal Harmony ---
#             internal_order_proxy = (soul_spark.pattern_coherence + soul_spark.phi_resonance) / 2.0
#             energy_gain = internal_order_proxy * HARMONIZATION_ENERGY_GAIN_RATE
#             soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_gain)

#             # --- 6. Update State (Recalculates SU/CU based on ALL factor changes) ---
#             soul_spark.update_state()
            
#             # --- 7. Develop Toroidal Flow ---
#             # Flow develops based on pattern coherence and harmony
#             rate_torus = HARMONIZATION_TORUS_RATE # Add this constant (e.g., 0.002)
#             current_torus = getattr(soul_spark, 'toroidal_flow_strength', 0.0)
#             # Drive torus development by coherence/harmony, faster when low
#             driver = (soul_spark.pattern_coherence + soul_spark.harmony) / 2.0
#             increase_torus = rate_torus * driver * (1.0 - current_torus**0.7)
#             soul_spark.toroidal_flow_strength = min(1.0, current_torus + increase_torus)

#             # --- 8. Energy Gain (Modify to include Torus) ---
#             internal_order_proxy = (soul_spark.pattern_coherence + soul_spark.phi_resonance + soul_spark.toroidal_flow_strength) / 3.0 # Average including torus
#             energy_gain = internal_order_proxy * HARMONIZATION_ENERGY_GAIN_RATE # Use the potentially boosted rate
#             soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_gain)
            
#             # --- Logging (Periodic) ---
#             if (i + 1) % (iterations // 10 or 1) == 0:
#                 logger.debug(f"  Harmonization {i+1}/{iterations}: S={soul_spark.stability:.1f}, "
#                              f"C={soul_spark.coherence:.1f}, E={soul_spark.energy:.1f}, "
#                              f"P.Coh={soul_spark.pattern_coherence:.3f}, PhiRes={soul_spark.phi_resonance:.3f}")
#             soul_spark.update_state()
#     except Exception as e:
#         # ... (Error handling as before) ...
#         logger.error(f"Error during harmonization loop for {spark_id} at iteration {i+1}: {e}", exc_info=True)
#         metrics_summary_fail = {'action': 'spark_harmonization', 'soul_id': spark_id, 'duration_sec': time.time() - start_time, 'iterations_run': i + 1, 'success': False, 'error': str(e)}
#         if METRICS_AVAILABLE:
#             try: metrics.record_metrics('spark_harmonization_summary', metrics_summary_fail)
#             except Exception as metric_err: logger.error(f"Failed record fail metrics: {metric_err}")
#         raise RuntimeError(f"Spark harmonization failed: {e}") from e

#     # --- Finalization & Metrics ---
#     # ... (Metrics calculation and logging as before) ...
#     final_s = soul_spark.stability; final_c = soul_spark.coherence; final_e = soul_spark.energy
#     final_p_coh = soul_spark.pattern_coherence; final_phi_res = soul_spark.phi_resonance; final_harmony = soul_spark.harmony
#     duration = time.time() - start_time
#     metrics_summary = {
#         'action': 'spark_harmonization','soul_id': spark_id,'duration_sec': duration,'iterations_run': iterations,
#         'initial_stability_su': initial_s, 'final_stability_su': final_s, 'stability_gain_su': final_s - initial_s,
#         'initial_coherence_cu': initial_c, 'final_coherence_cu': final_c, 'coherence_gain_cu': final_c - initial_c,
#         'initial_energy_seu': initial_e, 'final_energy_seu': final_e, 'energy_gain_seu': final_e - initial_e,
#         'initial_pattern_coherence': initial_p_coh, 'final_pattern_coherence': final_p_coh,
#         'initial_phi_resonance': initial_phi_res, 'final_phi_resonance': final_phi_res,
#         'initial_harmony': initial_harmony, 'final_harmony': final_harmony, 'success': True
#     }
#     if METRICS_AVAILABLE:
#         try: metrics.record_metrics('spark_harmonization_summary', metrics_summary)
#         except Exception as metric_err: logger.error(f"Failed record success metrics: {metric_err}")

#     logger.info(f"--- Spark Harmonization Complete for {spark_id}. ---")
#     logger.info(f"  Final State: S={final_s:.1f}, C={final_c:.1f}, E={final_e:.1f}, "
#                 f"P.Coh={final_p_coh:.3f}, PhiRes={final_phi_res:.3f}")

#     return soul_spark, metrics_summary

# --- END OF FILE src/stage_1/soul_formation/spark_harmonization.py ---