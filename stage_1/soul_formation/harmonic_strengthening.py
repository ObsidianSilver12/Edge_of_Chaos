# --- START OF FILE src/stage_1/soul_formation/harmonic_strengthening.py ---

"""
Harmonic Strengthening Functions (Natural Principles V1.0)

Enhances soul integration through natural principles of self-organization,
homeostasis, emergent properties, and energy conservation.
Rather than arbitrary weights and targets, this approach allows the soul
to find its own optimal state through natural balancing mechanisms.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional
from math import sqrt, pi as PI, exp, atan2, tanh, log
from shared.constants.constants import *

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from shared.constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Harmonic Strengthening cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}.")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    try: 
        logger.setLevel(LOG_LEVEL)
    except NameError: 
        logger.warning("LOG_LEVEL constant not found."); logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter(LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(log_formatter); logger.addHandler(ch)

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

# --- Natural System Helper Functions ---

def _calculate_circular_variance(phases_array: np.ndarray) -> float:
    """Calculates circular variance (0=sync, 1=uniform)."""
    if phases_array is None or not isinstance(phases_array, np.ndarray) or phases_array.size < 2:
        return 1.0 # Max variance if no/one phase or invalid input
    if not np.all(np.isfinite(phases_array)):
        logger.warning("Non-finite values found in phases array during variance calculation.")
        return 1.0 # Max variance if data invalid
    mean_cos = np.mean(np.cos(phases_array))
    mean_sin = np.mean(np.sin(phases_array))
    r_len_sq = mean_cos**2 + mean_sin**2
    if r_len_sq < 0: # Protect against precision errors
        r_len_sq = 0.0
    r_len = sqrt(r_len_sq)
    return 1.0 - r_len

def _calculate_harmonic_deviation(harmonics: List[float], base_freq: float) -> float:
    """Calculates average deviation of harmonics from ideal natural ratios."""
    if not harmonics or not isinstance(harmonics, list) or base_freq <= FLOAT_EPSILON:
        return 1.0 # Max deviation if no harmonics/base_freq
    valid_harmonics = [h for h in harmonics if isinstance(h, (int,float)) and h > FLOAT_EPSILON]
    if not valid_harmonics: return 1.0

    # Natural systems use both integer and Phi-based ratios
    deviations = []
    for h in valid_harmonics:
        ratio = h / base_freq
        if ratio <= 0 or not np.isfinite(ratio): continue
        
        # Natural harmonic series - include higher harmonics as well
        int_dists = [abs(ratio - n) for n in range(1, 9)]
        
        # Fibonacci/Golden ratio relationships - natural growth patterns
        phi_powers = [PHI**n for n in [-2, -1, 1, 2, 3]]
        phi_dists = [abs(ratio - p) for p in phi_powers]
        
        # Find nearest natural ratio
        min_deviation = min(int_dists + phi_dists)
        
        # Normalize by the ratio itself to account for larger frequency ranges
        normalized_deviation = min_deviation / ratio
        deviations.append(normalized_deviation)

    # Return average deviation, capped at 1.0
    avg_dev = float(np.mean(deviations)) if deviations else 1.0
    return min(1.0, float(avg_dev))

def _identify_energy_distribution(soul_spark: SoulSpark) -> Dict[str, float]:
    """
    Identifies natural energy distribution across soul components.
    Returns a dictionary of factors and their energy distribution weights.
    Higher weights indicate areas that need more energy.
    """
    # Natural systems allocate energy to where it's most needed
    
    # Calculate need based on current state and ideal balance point
    # Higher distance from natural balance = higher energy need
    
    # Phase coherence need
    phases = np.array(soul_spark.frequency_signature.get('phases', []))
    phase_coherence = 1.0 - _calculate_circular_variance(phases)
    phase_need = 1.0 - phase_coherence
    
    # Harmonic purity need
    harmonic_deviation = _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency)
    harmonic_need = harmonic_deviation
    
    # Phi resonance need
    phi_need = 1.0 - soul_spark.phi_resonance
    
    # Pattern coherence need
    pattern_need = 1.0 - soul_spark.pattern_coherence
    
    # Harmony need
    harmony_need = 1.0 - soul_spark.harmony
    
    # Toroidal flow need
    torus_need = 1.0 - soul_spark.toroidal_flow_strength
    
    # Apply natural diminishing returns curve
    # In nature, more energy flows to areas of greatest need,
    # but there are diminishing returns as systems approach balance
    needs = {
        'phase': max(0, 1.0 - exp(-3 * phase_need)),
        'harmonic': max(0, 1.0 - exp(-3 * harmonic_need)),
        'phi': max(0, 1.0 - exp(-3 * phi_need)),
        'pattern': max(0, 1.0 - exp(-3 * pattern_need)),
        'harmony': max(0, 1.0 - exp(-3 * harmony_need)),
        'torus': max(0, 1.0 - exp(-3 * torus_need))
    }
    
    # Normalize to create a natural distribution
    total_need = sum(needs.values())
    if total_need > FLOAT_EPSILON:
        distribution = {k: v / total_need for k, v in needs.items()}
    else:
        # If all needs are minimal, distribute evenly
        distribution = {k: 1.0 / len(needs) for k in needs}
    
    return distribution

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """
    Check prerequisites using natural thresholds based on 
    the soul's current capability rather than arbitrary constants.
    """
    logger.debug(f"Checking prerequisites for harmonic strengthening (Soul: {soul_spark.spark_id})...")
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")

    # Stage Completion Flag - this is still needed for process flow
    if not getattr(soul_spark, FLAG_READY_FOR_STRENGTHENING, False):
        msg = f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_STRENGTHENING}."
        logger.error(msg)
        raise ValueError(msg)

    # Check for existence of required attributes
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if stability_su < 0 or coherence_cu < 0:
        msg = "Prerequisite failed: Soul missing stability or coherence attributes."
        logger.error(msg)
        raise AttributeError(msg)

    # Natural systems don't use arbitrary thresholds
    # Instead, they require a minimum viability threshold
    # to sustain further development
    
    # Base requirement: at least 50% of maximum capacity
    min_stability = MAX_STABILITY_SU * 0.5
    min_coherence = MAX_COHERENCE_CU * 0.5
    
    if stability_su < min_stability:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) below minimum viability threshold ({min_stability:.1f} SU)."
        logger.error(msg)
        raise ValueError(msg)
        
    if coherence_cu < min_coherence:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) below minimum viability threshold ({min_coherence:.1f} CU)."
        logger.error(msg)
        raise ValueError(msg)

    if getattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_HARMONICALLY_STRENGTHENED}. Re-running.")

    logger.debug("Harmonic strengthening prerequisites met through natural viability thresholds.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """Ensure soul has necessary properties for natural strengthening."""
    logger.debug(f"Ensuring properties for harmonic strengthening (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'harmonics', 'phi_resonance',
                'pattern_coherence', 'harmony', 'aspects', 'frequency_signature',
                'toroidal_flow_strength']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for natural strengthening: {missing}")

    if soul_spark.frequency <= FLOAT_EPSILON: 
        raise ValueError("Soul frequency must be positive for natural resonance.")
    if not isinstance(soul_spark.frequency_signature, dict): 
        raise TypeError("frequency_signature must be dict for harmonic structure.")
    if 'phases' not in soul_spark.frequency_signature: 
        raise ValueError("frequency_signature missing 'phases' for coherence calculations.")
    if not isinstance(soul_spark.harmonics, list): 
        raise TypeError("harmonics must be a list for resonance patterns.")

    # Initialize field radius/strength if missing - these create the soul's boundary
    if not hasattr(soul_spark, 'field_radius'): 
        setattr(soul_spark, 'field_radius', 1.0)
    if not hasattr(soul_spark, 'field_strength'): 
        setattr(soul_spark, 'field_strength', 0.5)

    logger.debug("Soul properties ensured for natural strengthening.")

# --- Natural Harmonic Strengthening Functions ---

def _heal_harmonic_structure(soul_spark: SoulSpark, energy_available: float) -> Dict[str, float]:
    """
    Naturally heal the harmonic structure of the soul using available energy.
    Returns metrics about changes made.
    """
    changes = {'energy_used': 0.0, 'harmonic_improvement': 0.0}
    harmonics_list = soul_spark.harmonics
    base_freq = soul_spark.frequency
    
    # Skip if no valid harmonics or frequency
    if not harmonics_list or base_freq <= FLOAT_EPSILON:
        return changes
        
    # Measure current state
    current_deviation = _calculate_harmonic_deviation(harmonics_list, base_freq)
    if current_deviation < FLOAT_EPSILON:  # Already perfect
        return changes
    
    # Natural systems have diminishing returns as they approach perfection
    # Effort required increases exponentially as perfection nears
    effort_scale = 1.0 / (1.0 - min(0.99, current_deviation))
    
    # Calculate energy required based on current deviation
    # More deviation = more energy needed, with diminishing returns
    max_energy_needed = current_deviation * 100.0 * effort_scale
    
    # Limit energy usage to available energy
    energy_used = min(energy_available, max_energy_needed)
    changes['energy_used'] = energy_used
    
    # Calculate potential improvement based on energy investment
    # Natural diminishing returns curve
    energy_effectiveness = 1.0 - exp(-energy_used / 50.0)
    potential_improvement = current_deviation * energy_effectiveness
    
    # Apply the improvement to each harmonic
    new_harmonics = []
    for h in harmonics_list:
        ratio = h / base_freq
        if ratio <= 0 or not np.isfinite(ratio):
            new_harmonics.append(h)
            continue
            
        # Find the nearest natural ratio (integer or phi-based)
        int_ratios = list(range(1, 9))
        phi_ratios = [PHI**n for n in [-2, -1, 1, 2, 3]]
        all_ratios = int_ratios + phi_ratios
        
        all_dists = [abs(ratio - r) for r in all_ratios]
        nearest_idx = np.argmin(all_dists)
        nearest_ratio = all_ratios[nearest_idx]
        
        # Calculate ideal harmonic frequency based on nearest natural ratio
        ideal_harmonic = base_freq * nearest_ratio
        
        # Apply a portion of the correction proportional to available energy
        # More deviation gets more correction - natural priority
        current_deviation = abs(h - ideal_harmonic) / max(h, ideal_harmonic)
        adjustment_factor = current_deviation * energy_effectiveness
        
        # Natural adjustment with increasing resistance as we approach perfection
        adjustment = (ideal_harmonic - h) * adjustment_factor
        adjusted_h = h + adjustment
        new_harmonics.append(adjusted_h)
    
    # Update harmonics in soul with new naturally adjusted values
    soul_spark.harmonics = new_harmonics
    
    # Update frequency_signature if it exists
    if 'frequencies' in soul_spark.frequency_signature:
        soul_spark.frequency_signature['frequencies'] = new_harmonics
    
    # Calculate improvement for metrics
    new_deviation = _calculate_harmonic_deviation(new_harmonics, base_freq)
    improvement = current_deviation - new_deviation
    changes['harmonic_improvement'] = improvement
    
    logger.debug(f"Harmonic healing: Used {energy_used:.2f} energy, Deviation {current_deviation:.4f} → {new_deviation:.4f}")
    return changes

def _heal_phase_coherence(soul_spark: SoulSpark, energy_available: float) -> Dict[str, float]:
    """
    Naturally heal the phase coherence of the soul's frequency signature.
    Returns metrics about changes made.
    """
    changes = {'energy_used': 0.0, 'phase_improvement': 0.0}
    
    # Skip if no valid phase data
    if not isinstance(soul_spark.frequency_signature, dict) or 'phases' not in soul_spark.frequency_signature:
        return changes
        
    phases_data = soul_spark.frequency_signature['phases']
    if not isinstance(phases_data, (list, np.ndarray)) or len(phases_data) < 2:
        return changes
    
    # Convert to numpy array for calculations
    phases = np.array(phases_data)
    
    # Measure current state
    initial_variance = _calculate_circular_variance(phases)
    current_coherence = 1.0 - initial_variance
    
    # Skip if already at perfect coherence
    if current_coherence > 0.99:
        return changes
    
    # Natural systems have diminishing returns as they approach perfection
    # More effort required as coherence increases
    effort_scale = 1.0 / (1.0 - min(0.99, current_coherence))
    
    # Calculate energy required based on current incoherence
    # More incoherence = more energy needed, but with natural scaling
    max_energy_needed = initial_variance * 100.0 * effort_scale
    
    # Limit energy usage to available energy
    energy_used = min(energy_available, max_energy_needed)
    changes['energy_used'] = energy_used
    
    # Calculate adjustment factor based on energy investment
    # Natural diminishing returns curve
    energy_effectiveness = 1.0 - exp(-energy_used / 50.0)
    
    # Calculate mean phase direction
    mean_cos = np.mean(np.cos(phases))
    mean_sin = np.mean(np.sin(phases))
    mean_phase = atan2(mean_sin, mean_cos)
    
    # Adjustment force proportional to incoherence and energy
    # More incoherent = stronger adjustment
    adjustment_force = initial_variance * energy_effectiveness
    
    # Apply the adjustment - phases move toward the mean phase
    # with natural resistance to extreme uniformity
    new_phases = phases * (1.0 - adjustment_force) + mean_phase * adjustment_force
    new_phases = new_phases % (2 * PI)
    
    # Update the soul's phase structure
    soul_spark.frequency_signature['phases'] = new_phases.tolist()
    
    # Calculate improvement for metrics
    new_variance = _calculate_circular_variance(new_phases)
    new_coherence = 1.0 - new_variance
    improvement = new_coherence - current_coherence
    changes['phase_improvement'] = improvement
    logger.debug(f"Phase healing: Used {energy_used:.2f} energy, Coherence {current_coherence:.4f} → {new_coherence:.4f}")
    return changes

def _strengthen_phi_resonance(soul_spark: SoulSpark, energy_available: float) -> Dict[str, float]:
   """
   Naturally strengthen phi resonance through golden ratio alignment.
   Returns metrics about changes made.
   """
   changes = {'energy_used': 0.0, 'phi_improvement': 0.0}
   
   # Get current phi resonance
   current_phi = soul_spark.phi_resonance
   
   # Skip if already at perfect resonance
   if current_phi > 0.99:
       return changes
   
   # Natural systems have increasing resistance as they approach perfection
   potential_improvement = 1.0 - current_phi
   
   # Calculate energy required based on how close to perfection
   # Exponential increase in energy needed as we approach perfection
   effort_scale = exp(4.0 * current_phi)  # Natural exponential scaling
   max_energy_needed = potential_improvement * 100.0 * effort_scale
   
   # Limit energy usage to available energy
   energy_used = min(energy_available, max_energy_needed)
   changes['energy_used'] = energy_used
   
   # Calculate improvement based on energy investment
   # Natural logarithmic improvement curve
   effectiveness = 1.0 - exp(-energy_used / max(10.0, max_energy_needed / 2.0))
   improvement = potential_improvement * effectiveness
   
   # Factor in synergistic effects from other components
   # Pattern coherence and toroidal flow naturally assist phi resonance
   pattern_factor = soul_spark.pattern_coherence
   torus_factor = soul_spark.toroidal_flow_strength
   synergy_multiplier = 1.0 + ((pattern_factor + torus_factor) / 4.0)
   
   # Apply improvement with synergistic effects
   final_improvement = improvement * synergy_multiplier
   soul_spark.phi_resonance = min(1.0, current_phi + final_improvement)
   
   # Track actual improvement
   actual_improvement = soul_spark.phi_resonance - current_phi
   changes['phi_improvement'] = actual_improvement
   
   logger.debug(f"Phi resonance healing: Used {energy_used:.2f} energy, Phi {current_phi:.4f} → {soul_spark.phi_resonance:.4f}")
   return changes

def _enhance_pattern_coherence(soul_spark: SoulSpark, energy_available: float) -> Dict[str, float]:
   """
   Naturally enhance pattern coherence through structural organization.
   Returns metrics about changes made.
   """
   changes = {'energy_used': 0.0, 'pattern_improvement': 0.0}
   
   # Get current pattern coherence
   current_pattern = soul_spark.pattern_coherence
   
   # Skip if already at perfect coherence
   if current_pattern > 0.99:
       return changes
   
   # Natural systems have increasing resistance as they approach perfection
   potential_improvement = 1.0 - current_pattern
   
   # Calculate energy required based on how close to perfection
   # Exponential increase in energy needed as we approach perfection
   effort_scale = exp(3.0 * current_pattern)  # Natural exponential scaling
   max_energy_needed = potential_improvement * 90.0 * effort_scale
   
   # Limit energy usage to available energy
   energy_used = min(energy_available, max_energy_needed)
   changes['energy_used'] = energy_used
   
   # Calculate improvement based on energy investment
   # Natural logarithmic improvement curve
   effectiveness = 1.0 - exp(-energy_used / max(10.0, max_energy_needed / 2.0))
   improvement = potential_improvement * effectiveness
   
   # Factor in synergistic effects from other components
   # Phi resonance and harmony naturally assist pattern coherence
   phi_factor = soul_spark.phi_resonance
   harmony_factor = soul_spark.harmony
   synergy_multiplier = 1.0 + ((phi_factor + harmony_factor) / 4.0)
   
   # Apply improvement with synergistic effects
   final_improvement = improvement * synergy_multiplier
   soul_spark.pattern_coherence = min(1.0, current_pattern + final_improvement)
   
   # Track actual improvement
   actual_improvement = soul_spark.pattern_coherence - current_pattern
   changes['pattern_improvement'] = actual_improvement
   
   logger.debug(f"Pattern coherence healing: Used {energy_used:.2f} energy, Pattern {current_pattern:.4f} → {soul_spark.pattern_coherence:.4f}")
   return changes

def _develop_harmony(soul_spark: SoulSpark, energy_available: float) -> Dict[str, float]:
   """
   Naturally develop harmony through balanced integration of components.
   Returns metrics about changes made.
   """
   changes = {'energy_used': 0.0, 'harmony_improvement': 0.0, 'energy_adjustment': 0.0}
   
   # Get current harmony
   current_harmony = soul_spark.harmony
   
   # Skip if already at perfect harmony
   if current_harmony > 0.99:
       return changes
   
   # Natural systems have increasing resistance as they approach perfection
   potential_improvement = 1.0 - current_harmony
   
   # Calculate energy required based on how close to perfection
   # Exponential increase in energy needed as we approach perfection
   effort_scale = exp(3.5 * current_harmony)  # Natural exponential scaling
   max_energy_needed = potential_improvement * 80.0 * effort_scale
   
   # Limit energy usage to available energy
   energy_used = min(energy_available, max_energy_needed)
   changes['energy_used'] = energy_used
   
   # Calculate improvement based on energy investment
   # Natural logarithmic improvement curve
   effectiveness = 1.0 - exp(-energy_used / max(10.0, max_energy_needed / 2.0))
   improvement = potential_improvement * effectiveness
   
   # Factor in synergistic effects from other components
   # In natural systems, harmony emerges from the balanced interaction of all parts
   phases = np.array(soul_spark.frequency_signature.get('phases', []))
   phase_coherence = float(1.0 - _calculate_circular_variance(phases))
   harmonic_purity = float(1.0 - _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency))
   
   # Multiple components contribute to harmony
   synergy_factors = [
       phase_coherence,
       harmonic_purity,
       soul_spark.pattern_coherence,
       soul_spark.phi_resonance
   ]
   
   # Natural synergy - geometric mean of contributing factors
   synergy_product = float(np.prod([max(0.1, f) for f in synergy_factors]))
   synergy_mean = synergy_product ** (1.0 / len(synergy_factors))
   synergy_multiplier = 0.5 + synergy_mean / 2.0  # Scale to 0.5-1.0 range
   
   # Apply improvement with synergistic effects
   final_improvement = improvement * synergy_multiplier
   soul_spark.harmony = min(1.0, current_harmony + final_improvement)
   
   # In natural systems, improved harmony creates energy
   # When components work in concert, less energy is wasted
   harmony_gain = soul_spark.harmony - current_harmony
   if harmony_gain > 0:
       energy_adjustment = harmony_gain * 20.0  # Generate energy proportional to harmony gain
       soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_adjustment)
       changes['energy_adjustment'] = energy_adjustment
   
   # Track actual improvement
   actual_improvement = soul_spark.harmony - current_harmony
   changes['harmony_improvement'] = actual_improvement
   
   logger.debug(f"Harmony development: Used {energy_used:.2f} energy, Harmony {current_harmony:.4f} → {soul_spark.harmony:.4f}, Energy +{changes['energy_adjustment']:.2f}")
   return changes

def _cultivate_toroidal_flow(soul_spark: SoulSpark, energy_available: float) -> Dict[str, float]:
   """
   Naturally cultivate toroidal flow through self-sustaining energy patterns.
   Returns metrics about changes made.
   """
   changes = {'energy_used': 0.0, 'torus_improvement': 0.0}
   
   # Get current toroidal flow strength
   current_torus = soul_spark.toroidal_flow_strength
   
   # Skip if already at perfect flow
   if current_torus > 0.99:
       return changes
   
   # Natural systems have increasing resistance as they approach perfection
   potential_improvement = 1.0 - current_torus
   
   # Calculate energy required based on how close to perfection
   # Exponential increase in energy needed as we approach perfection
   effort_scale = exp(3.0 * current_torus)  # Natural exponential scaling
   max_energy_needed = potential_improvement * 120.0 * effort_scale
   
   # Limit energy usage to available energy
   energy_used = min(energy_available, max_energy_needed)
   changes['energy_used'] = energy_used
   
   # Calculate improvement based on energy investment
   # Natural logarithmic improvement curve
   effectiveness = 1.0 - exp(-energy_used / max(10.0, max_energy_needed / 2.0))
   improvement = potential_improvement * effectiveness
   
   # Factor in system maturity
   # Toroidal flow depends on the overall maturity of the soul system
   stability_factor = soul_spark.stability / MAX_STABILITY_SU
   coherence_factor = soul_spark.coherence / MAX_COHERENCE_CU
   harmony_factor = soul_spark.harmony
   
   # Natural systems develop toroidal flow as they become more stable and coherent
   maturity_factor = (stability_factor + coherence_factor + harmony_factor) / 3.0
   maturity_multiplier = 0.3 + maturity_factor * 0.7  # Scale to 0.3-1.0 range
   
   # Apply improvement with maturity effects
   final_improvement = improvement * maturity_multiplier
   soul_spark.toroidal_flow_strength = min(1.0, current_torus + final_improvement)
   
   # Track actual improvement
   actual_improvement = soul_spark.toroidal_flow_strength - current_torus
   changes['torus_improvement'] = actual_improvement
   
   logger.debug(f"Toroidal flow cultivation: Used {energy_used:.2f} energy, Torus {current_torus:.4f} → {soul_spark.toroidal_flow_strength:.4f}")
   return changes

def _run_natural_healing_cycle(soul_spark: SoulSpark, cycle_num: int) -> Dict[str, float]:
   """
   Run one natural healing cycle that distributes energy based on need.
   This follows how natural systems allocate resources to where they're most needed.
   """
   changes = {
       'energy_used': 0.0,
       'delta_phi': 0.0,
       'delta_pattern': 0.0,
       'delta_harmony': 0.0,
       'delta_torus': 0.0,
       'delta_phase': 0.0,
       'delta_harmonic': 0.0,
       'energy_generated': 0.0,
       'healing_areas': []
   }
   
   # --- Natural energy allocation ---
   
   # Determine how much energy to use per cycle
   # Natural systems don't deplete all energy at once, but use sustainable amounts
   available_total_energy = soul_spark.energy * 0.02  # Use 2% of available energy per cycle
   
   # Identify areas most in need of healing based on current state
   energy_distribution = _identify_energy_distribution(soul_spark)
   
   # Sort healing needs by priority (highest need first)
   sorted_needs = sorted(energy_distribution.items(), key=lambda x: x[1], reverse=True)
   
   # Track total energy used
   total_energy_used = 0.0
   
   # Apply healing to areas based on need
   for area, distribution in sorted_needs:
       if distribution < 0.01:  # Skip areas with minimal need
           continue
           
       # Allocate energy proportional to need
       area_energy = available_total_energy * distribution
       
       if area == 'phi':
           # Strengthen phi resonance
           result = _strengthen_phi_resonance(soul_spark, area_energy)
           changes['delta_phi'] = result['phi_improvement']
           total_energy_used += result['energy_used']
           if result['phi_improvement'] > FLOAT_EPSILON:
               changes['healing_areas'].append('phi')
               
       elif area == 'pattern':
           # Enhance pattern coherence
           result = _enhance_pattern_coherence(soul_spark, area_energy)
           changes['delta_pattern'] = result['pattern_improvement']
           total_energy_used += result['energy_used']
           if result['pattern_improvement'] > FLOAT_EPSILON:
               changes['healing_areas'].append('pattern')
               
       elif area == 'harmony':
           # Develop harmony
           result = _develop_harmony(soul_spark, area_energy)
           changes['delta_harmony'] = result['harmony_improvement']
           total_energy_used += result['energy_used']
           changes['energy_generated'] += result.get('energy_adjustment', 0.0)
           if result['harmony_improvement'] > FLOAT_EPSILON:
               changes['healing_areas'].append('harmony')
               
       elif area == 'torus':
           # Cultivate toroidal flow
           result = _cultivate_toroidal_flow(soul_spark, area_energy)
           changes['delta_torus'] = result['torus_improvement']
           total_energy_used += result['energy_used']
           if result['torus_improvement'] > FLOAT_EPSILON:
               changes['healing_areas'].append('torus')
               
       elif area == 'phase':
           # Heal phase coherence
           result = _heal_phase_coherence(soul_spark, area_energy)
           changes['delta_phase'] = result['phase_improvement']
           total_energy_used += result['energy_used']
           if result['phase_improvement'] > FLOAT_EPSILON:
               changes['healing_areas'].append('phase')
               
       elif area == 'harmonic':
           # Heal harmonic structure
           result = _heal_harmonic_structure(soul_spark, area_energy)
           changes['delta_harmonic'] = result['harmonic_improvement']
           total_energy_used += result['energy_used']
           if result['harmonic_improvement'] > FLOAT_EPSILON:
               changes['healing_areas'].append('harmonic')
   
   # Update total energy used
   changes['energy_used'] = total_energy_used
   
   # Apply energy consumption
   soul_spark.energy -= total_energy_used
   
   # Log detailed healing information
   if cycle_num % 10 == 0 or cycle_num <= 5:
       logger.info(f"HS Cycle {cycle_num} - Natural healing: Energy used={total_energy_used:.2f}, "
                  f"Areas healed: {', '.join(changes['healing_areas']) if changes['healing_areas'] else 'none needed'}")
       logger.info(f"  Soul state: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}, "
                 f"Phi={soul_spark.phi_resonance:.3f}, PCoh={soul_spark.pattern_coherence:.3f}, "
                 f"Harm={soul_spark.harmony:.3f}, Torus={soul_spark.toroidal_flow_strength:.3f}")
   
   return changes

def perform_harmonic_strengthening(soul_spark: SoulSpark, intensity: float = 0.7, 
                                 duration_factor: float = 1.0) -> Tuple[SoulSpark, Dict[str, Any]]:
   """
   Performs harmonic strengthening using natural principles.
   Allow the soul to find its own optimal state through natural healing processes.
   """
   # --- Input Validation ---
   if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
   if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
   if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")

   spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
   max_cycles = int(HS_MAX_CYCLES * duration_factor)
   logger.info(f"--- Starting Natural Harmonic Strengthening for Soul {spark_id} (Max Cycles={max_cycles}) ---")
   start_time_iso = datetime.now().isoformat()
   start_time_dt = datetime.fromisoformat(start_time_iso)
   cycle_changes_log = [] # Store changes per cycle

   try:
       _ensure_soul_properties(soul_spark)
       if not _check_prerequisites(soul_spark):
           raise ValueError("Soul prerequisites for harmonic strengthening not met.")

       # Verify required methods exist
       if not hasattr(soul_spark, '_calculate_stability_score') or not hasattr(soul_spark, '_calculate_coherence_score'):
           raise AttributeError("SoulSpark missing required calculation methods")

       # Store Initial State
       initial_state = {
           'stability_su': soul_spark.stability, 
           'coherence_cu': soul_spark.coherence,
           'energy_seu': soul_spark.energy, 
           'frequency_hz': soul_spark.frequency,
           'phi_resonance': soul_spark.phi_resonance,
           'pattern_coherence': soul_spark.pattern_coherence, 
           'harmony': soul_spark.harmony,
           'toroidal_flow_strength': soul_spark.toroidal_flow_strength,
           'phase_coherence_initial': 1.0 - _calculate_circular_variance(np.array(soul_spark.frequency_signature.get('phases', []))),
           'harmonic_purity_initial': 1.0 - _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency)
       }
       logger.info(f"Natural HS Initial State: S={initial_state['stability_su']:.1f}, C={initial_state['coherence_cu']:.1f}, "
                  f"Phi={initial_state['phi_resonance']:.3f}, P.Coh={initial_state['pattern_coherence']:.3f}, "
                  f"Harm={initial_state['harmony']:.3f}, Torus={initial_state['toroidal_flow_strength']:.3f}")

       # --- Natural Healing Loop ---
       cycles_run = 0
       stabilization_cycles = 0  # Count cycles where stability improved
       coherence_cycles = 0      # Count cycles where coherence improved
       
       # Records for detecting natural equilibrium
       last_updates = {
           'phi': 0,
           'pattern': 0,
           'harmony': 0,
           'torus': 0,
           'phase': 0,
           'harmonic': 0
       }
       equilibrium_detected = False
       stagnation_counter = 0
       
       for cycle in range(max_cycles):
           cycles_run = cycle + 1
           
           # Record state before cycle
           pre_update_stability = soul_spark.stability
           pre_update_coherence = soul_spark.coherence
           pre_update_phi = soul_spark.phi_resonance
           pre_update_pattern = soul_spark.pattern_coherence
           pre_update_harmony = soul_spark.harmony
           pre_update_torus = soul_spark.toroidal_flow_strength
           
           # Run natural healing cycle
           cycle_changes = _run_natural_healing_cycle(soul_spark, cycles_run)
           cycle_changes_log.append(cycle_changes)
           
           # Update state to recalculate stability/coherence
           soul_spark.update_state()
           
           # Check for new stability/coherence records
           stability_improved = soul_spark.stability > pre_update_stability + FLOAT_EPSILON
           coherence_improved = soul_spark.coherence > pre_update_coherence + FLOAT_EPSILON
           
           if stability_improved:
               stabilization_cycles += 1
           if coherence_improved:
               coherence_cycles += 1
           
           # Track component changes
           if abs(soul_spark.phi_resonance - pre_update_phi) > FLOAT_EPSILON:
               last_updates['phi'] = cycle
           if abs(soul_spark.pattern_coherence - pre_update_pattern) > FLOAT_EPSILON:
               last_updates['pattern'] = cycle
           if abs(soul_spark.harmony - pre_update_harmony) > FLOAT_EPSILON:
               last_updates['harmony'] = cycle
           if abs(soul_spark.toroidal_flow_strength - pre_update_torus) > FLOAT_EPSILON:
               last_updates['torus'] = cycle
           if cycle_changes['delta_phase'] > FLOAT_EPSILON:
               last_updates['phase'] = cycle
           if cycle_changes['delta_harmonic'] > FLOAT_EPSILON:
               last_updates['harmonic'] = cycle
               
           # Add stability/coherence changes to cycle metrics
           cycle_changes['delta_stability'] = soul_spark.stability - pre_update_stability
           cycle_changes['delta_coherence'] = soul_spark.coherence - pre_update_coherence
           cycle_changes['current_stability'] = soul_spark.stability
           cycle_changes['current_coherence'] = soul_spark.coherence
           
           # Natural equilibrium detection
           # In nature, systems reach stable states where no further changes occur
           if cycles_run >= 20:  # Need enough cycles to detect patterns
               # Check if any significant healing has happened in the last 15 cycles
               recent_healing = any(cycles_run - last_cycle <= 15 for last_cycle in last_updates.values())
               
               if not recent_healing:
                   # No recent healing - system may be at equilibrium
                   stagnation_counter += 1
               else:
                   stagnation_counter = 0
                   
               # Consider equilibrium reached after 10 cycles of stability
               if stagnation_counter >= 10:
                   equilibrium_detected = True
                   logger.info(f"Natural equilibrium detected after {cycles_run} cycles - system stabilized")
                   break
           
           # Check if we've reached natural limits
           # Natural systems have physical limits to what they can achieve
           if (soul_spark.phi_resonance > 0.97 and
               soul_spark.pattern_coherence > 0.97 and
               soul_spark.harmony > 0.97 and
               soul_spark.toroidal_flow_strength > 0.97):
               logger.info(f"Natural perfection achieved after {cycles_run} cycles")
               break
               
           # Conserve energy - stop if energy is critically low
           if soul_spark.energy < 0.05 * initial_state['energy_seu']:
               logger.info(f"Energy conservation engaged after {cycles_run} cycles - energy running low")
               break
           
           # Check natural perfection of stability/coherence
           if soul_spark.stability >= MAX_STABILITY_SU - 0.1 and soul_spark.coherence >= MAX_COHERENCE_CU - 0.1:
               logger.info(f"Natural perfection of stability and coherence achieved after {cycles_run} cycles")
               break
       
       # Log if max cycles reached without equilibrium
       if cycles_run >= max_cycles:
           logger.warning(f"Natural healing reached max cycles ({max_cycles}) without reaching equilibrium. "
                         f"Final: S={soul_spark.stability:.2f}, C={soul_spark.coherence:.2f}")
           
       # --- Finalization ---
       setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, True)
       setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, True)
       
       # Update the last_modified timestamp
       last_mod_time = datetime.now().isoformat()
       setattr(soul_spark, 'last_modified', last_mod_time)

       # Add a memory echo to record the harmonization
       if hasattr(soul_spark, 'add_memory_echo'):
           stats_summary = f"S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}"
           if equilibrium_detected:
               soul_spark.add_memory_echo(f"Naturally harmonized to equilibrium (Cycles: {cycles_run}). {stats_summary}")
           else:
               soul_spark.add_memory_echo(f"Natural harmony strengthened (Cycles: {cycles_run}). {stats_summary}")

       # Calculate final phase and harmonic metrics
       final_phases = np.array(soul_spark.frequency_signature.get('phases', []))
       final_phase_coh = 1.0 - _calculate_circular_variance(final_phases)
       final_harm_purity = 1.0 - _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency)

       # Compile Overall Metrics
       end_time_iso = last_mod_time
       end_time_dt = datetime.fromisoformat(end_time_iso)
       final_state = {
           'stability_su': soul_spark.stability, 
           'coherence_cu': soul_spark.coherence,
           'energy_seu': soul_spark.energy, 
           'frequency_hz': soul_spark.frequency,
           'phi_resonance': soul_spark.phi_resonance,
           'pattern_coherence': soul_spark.pattern_coherence, 
           'harmony': soul_spark.harmony,
           'toroidal_flow_strength': soul_spark.toroidal_flow_strength,
           'phase_coherence_final': final_phase_coh,
           'harmonic_purity_final': final_harm_purity,
           FLAG_HARMONICALLY_STRENGTHENED: getattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED)
       }
       
       # Calculate gains
       stability_gain = final_state['stability_su'] - initial_state['stability_su']
       coherence_gain = final_state['coherence_cu'] - initial_state['coherence_cu']
       phi_gain = final_state['phi_resonance'] - initial_state['phi_resonance']
       pcoh_gain = final_state['pattern_coherence'] - initial_state['pattern_coherence']
       harmony_gain = final_state['harmony'] - initial_state['harmony']
       torus_gain = final_state['toroidal_flow_strength'] - initial_state['toroidal_flow_strength']
       phase_coh_gain = final_state['phase_coherence_final'] - initial_state['phase_coherence_initial']
       harm_purity_gain = final_state['harmonic_purity_final'] - initial_state['harmonic_purity_initial']

       # Complete metrics report
       overall_metrics = {
           'action': 'natural_harmonic_strengthening', 
           'soul_id': spark_id,
           'start_time': start_time_iso, 
           'end_time': end_time_iso,
           'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
           'cycles_run': cycles_run, 
           'max_cycles': max_cycles,
           'stabilization_cycles': stabilization_cycles,
           'coherence_cycles': coherence_cycles,
           'equilibrium_detected': equilibrium_detected,
           'initial_state': initial_state, 
           'final_state': final_state,
           'stability_gain_su': float(stability_gain),
           'coherence_gain_cu': float(coherence_gain),
           'phi_resonance_gain': float(phi_gain),
           'pattern_coherence_gain': float(pcoh_gain),
           'harmony_gain': float(harmony_gain),
           'toroidal_flow_gain': float(torus_gain),
           'phase_coherence_gain': float(phase_coh_gain),
           'harmonic_purity_gain': float(harm_purity_gain),
           'success': True,
           'cycle_details': cycle_changes_log[:min(len(cycle_changes_log), 40)]  # Include first 40 cycles
       }

       # Log metrics if available
       if METRICS_AVAILABLE: 
           metrics.record_metrics('harmonic_strengthening_summary', overall_metrics)

       # Final summary log
       logger.info(f"--- Natural Harmonic Strengthening Completed Successfully for Soul {spark_id} ({cycles_run} cycles) ---")
       logger.info(f"  Factor Gains: Phi: {phi_gain:+.3f}, P.Coh: {pcoh_gain:+.3f}, Harm: {harmony_gain:+.3f}, Torus: {torus_gain:+.3f}")
       logger.info(f"  Quality Gains: Phase Coh: {phase_coh_gain:+.3f}, Harm Purity: {harm_purity_gain:+.3f}")
       logger.info(f"  Final State: S={soul_spark.stability:.1f} SU ({stability_gain:+.1f}), C={soul_spark.coherence:.1f} CU ({coherence_gain:+.1f})")
       logger.info(f"  Cycles with improvements: Stability: {stabilization_cycles}/{cycles_run}, Coherence: {coherence_cycles}/{cycles_run}")

       return soul_spark, overall_metrics

   except (ValueError, TypeError, AttributeError) as e_val:
        logger.error(f"Natural Harmonic Strengthening failed for {spark_id} due to validation/attribute error: {e_val}", exc_info=True)
        failed_step = 'prerequisites/validation'
        record_hs_failure(spark_id, start_time_iso, failed_step, str(e_val))
        raise e_val
   except RuntimeError as e_rt:
        logger.critical(f"Natural Harmonic Strengthening failed critically for {spark_id}: {e_rt}", exc_info=True)
        failed_step = 'runtime'
        record_hs_failure(spark_id, start_time_iso, failed_step, str(e_rt))
        setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False)
        setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False)
        raise e_rt
   except Exception as e:
        logger.critical(f"Unexpected error during Natural Harmonic Strengthening for {spark_id}: {e}", exc_info=True)
        failed_step = 'unexpected'
        record_hs_failure(spark_id, start_time_iso, failed_step, str(e))
        setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False)
        setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False)
        raise RuntimeError(f"Unexpected Natural Harmonic Strengthening failure: {e}") from e

# --- Failure Metric Helper ---
def record_hs_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
   """ Helper to record failure metrics consistently. """
   if METRICS_AVAILABLE:
       try:
           end_time = datetime.now().isoformat()
           duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
           metrics.record_metrics('harmonic_strengthening_summary', {
               'action': 'natural_harmonic_strengthening', 'soul_id': spark_id,
               'start_time': start_time_iso, 'end_time': end_time,
               'duration_seconds': duration,
               'success': False, 'error': error_msg, 'failed_step': failed_step
           })
       except Exception as metric_e:
            logger.error(f"Failed to record Natural HS failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/harmonic_strengthening.py ---









