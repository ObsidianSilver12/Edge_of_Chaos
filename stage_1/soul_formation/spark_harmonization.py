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
from math import sqrt, pi as PI, exp, atan2
from constants.constants import *
# --- Logging ---
logger = logging.getLogger(__name__)

# Ensure constants are properly imported
try:
    from constants.constants import *
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: constants.py failed import in spark_harmonization.py")
    raise ImportError(f"Essential constants missing: {e}")


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
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
    logger.debug("Metrics tracking loaded in spark_harmonization.")
except ImportError:
    logger.error("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()


# --- Helper Functions for Harmonization ---
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

def calculate_resonance(freq1: float, freq2: float) -> float:
    """
    Calculate resonance between two frequencies.
    Returns a value between 0 (no resonance) and 1 (perfect resonance).
    """
    if freq1 <= FLOAT_EPSILON or freq2 <= FLOAT_EPSILON:
        return 0.0
        
    # Get ratio between frequencies (ensure smaller freq is denominator)
    ratio = max(freq1, freq2) / min(freq1, freq2)
    
    # Check for simple integer ratios (1:1, 2:1, 3:2, etc)
    int_resonance = 0.0
    for n in range(1, 6):
        for d in range(1, 6):
            if d == 0: continue
            target_ratio = n / d
            ratio_diff = abs(ratio - target_ratio)
            if ratio_diff < 0.1:  # Close to integer ratio
                res_score = 1.0 - ratio_diff * 10  # Scale: 0.0 to 1.0
                int_resonance = max(int_resonance, res_score)
    
    # Check for phi-based ratios
    phi_resonance = 0.0
    phi_ratios = [PHI, 1/PHI, PHI**2, 1/(PHI**2)]
    for phi_r in phi_ratios:
        ratio_diff = abs(ratio - phi_r)
        if ratio_diff < 0.1:  # Close to phi ratio
            res_score = 1.0 - ratio_diff * 10  # Scale: 0.0 to 1.0
            phi_resonance = max(phi_resonance, res_score)
    
    # Return the strongest resonance found
    return max(int_resonance, phi_resonance)

def _enhance_layer_integration(soul_spark) -> float:
    """
    Enhance integration between existing aura layers.
    Returns coherence gain from the process.
    """
    if not hasattr(soul_spark, 'layers') or len(soul_spark.layers) < 2:
        logger.debug(f"Not enough layers for integration: {getattr(soul_spark, 'spark_id', 'unknown')}")
        return 0.0
    
    integration_improved = 0.0
    coherence_gain = 0.0
    layer_integration_metrics = []
    
    # Process all adjacent layer pairs
    for i in range(1, len(soul_spark.layers)):
        prev_layer = soul_spark.layers[i-1]
        curr_layer = soul_spark.layers[i]
        
        # Initialize integration data if needed
        if 'integration' not in curr_layer:
            curr_layer['integration'] = {}
        
        integration_data = curr_layer['integration']
        if f'with_layer_{i-1}' not in integration_data:
            integration_data[f'with_layer_{i-1}'] = 0.2  # Start with basic connection
        
        # Calculate integration factors
        frequency_match = 0.0
        density_harmony = 0.0
        harmonic_match = 0.0
        
        # Check frequency resonance between layers
        if ('resonant_frequencies' in prev_layer and 
            'resonant_frequencies' in curr_layer):
            
            # Find strongest frequency resonance between layers
            best_resonance = 0.0
            for f1 in prev_layer['resonant_frequencies']:
                for f2 in curr_layer['resonant_frequencies']:
                    res = calculate_resonance(f1, f2)
                    best_resonance = max(best_resonance, res)
            
            frequency_match = best_resonance
        
        # Check density compatibility
        if ('density' in prev_layer and 'density' in curr_layer):
            prev_density = prev_layer['density'].get('base_density', 0.5)
            curr_density = curr_layer['density'].get('base_density', 0.5)
            
            # Small density differences are better for integration
            density_diff = abs(prev_density - curr_density)
            density_harmony = 1.0 - min(1.0, density_diff * 2.0)
        
        # Check harmonic relationships
        if ('harmonic_data' in prev_layer and 'harmonic_data' in curr_layer):
            # Compare harmonic ratios - similar ratios integrate better
            prev_ratios = []
            curr_ratios = []
            
            for h in prev_layer['harmonic_data'].values():
                if isinstance(h, dict) and 'ratio_value' in h:
                    prev_ratios.append(h['ratio_value'])
                    
            for h in curr_layer['harmonic_data'].values():
                if isinstance(h, dict) and 'ratio_value' in h:
                    curr_ratios.append(h['ratio_value'])
            
            if prev_ratios and curr_ratios:
                # Find closest ratio match
                best_match = 0.0
                for r1 in prev_ratios:
                    for r2 in curr_ratios:
                        if r1 <= 0 or r2 <= 0:
                            continue
                        ratio_diff = abs(r1 - r2) / max(r1, r2)
                        match_score = 1.0 - min(1.0, ratio_diff * 2.0)
                        best_match = max(best_match, match_score)
                
                harmonic_match = best_match
        
        # Calculate overall integration improvement
        current_integration = integration_data[f'with_layer_{i-1}']
        integration_factor = (
            frequency_match * 0.5 + 
            density_harmony * 0.3 + 
            harmonic_match * 0.2
        )
        
        # Apply gradual improvement based on factor
        max_improvement = 0.15  # Cap per cycle for stability
        improvement = max_improvement * integration_factor * (1.0 - current_integration)
        new_integration = min(1.0, current_integration + improvement)
        
        integration_data[f'with_layer_{i-1}'] = float(new_integration)
        integration_improved += (new_integration - current_integration)
        
        # Record metrics for this layer pair
        layer_pair_metrics = {
            'layer_pair': f"{i-1}_{i}",
            'frequency_match': float(frequency_match),
            'density_harmony': float(density_harmony),
            'harmonic_match': float(harmonic_match),
            'integration_before': float(current_integration),
            'integration_after': float(new_integration),
            'improvement': float(new_integration - current_integration)
        }
        layer_integration_metrics.append(layer_pair_metrics)
        
        logger.debug(f"Layer {i} integration with {i-1}: {current_integration:.3f}->{new_integration:.3f} "
                    f"(FreqMatch={frequency_match:.2f}, DensityHarm={density_harmony:.2f}, "
                    f"HarmMatch={harmonic_match:.2f})")
    
    # Calculate coherence gain from integration improvements
    # Scale to meaningful CU gain - calibrate based on testing
    coherence_gain = integration_improved * 6.0
    
    # Record integration process in soul history if available
    if integration_improved > 0.001 and hasattr(soul_spark, 'interaction_history'):
        integration_record = {
            'type': 'layer_integration_enhancement',
            'timestamp': datetime.now().isoformat(),
            'layer_pairs_count': len(layer_integration_metrics),
            'total_integration_improved': float(integration_improved),
            'coherence_gain': float(coherence_gain),
            'layer_details': layer_integration_metrics
        }
        soul_spark.interaction_history.append(integration_record)
    
    logger.debug(f"Layer integration improved by {integration_improved:.4f}, coherence gain: {coherence_gain:.2f} CU")
    
    # If we have metrics tracking available, record the data
    if METRICS_AVAILABLE:
        try:
            metrics_data = {
                'action': 'layer_integration',
                'soul_id': getattr(soul_spark, 'spark_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'layer_count': len(soul_spark.layers),
                'integration_improved': float(integration_improved),
                'coherence_gain': float(coherence_gain),
                'layer_pairs': layer_integration_metrics
            }
            metrics.record_metrics('layer_integration', metrics_data)
        except Exception as e:
            logger.error(f"Error recording layer integration metrics: {e}")
    
    return coherence_gain

def _enhance_aspect_layer_resonance(soul_spark) -> float:
    """
    Enhance resonance between aspects and aura layers.
    Returns coherence gain from the process.
    """
    if not hasattr(soul_spark, 'aspects') or not soul_spark.aspects:
        logger.debug(f"No aspects found for resonance enhancement: {getattr(soul_spark, 'spark_id', 'unknown')}")
        return 0.0
        
    if not hasattr(soul_spark, 'layers') or not soul_spark.layers:
        logger.debug(f"No layers found for resonance enhancement: {getattr(soul_spark, 'spark_id', 'unknown')}")
        return 0.0
    
    aspect_resonance_improved = 0.0
    coherence_gain = 0.0
    
    # Process each aspect
    for aspect_name, aspect_data in soul_spark.aspects.items():
        # Initialize aspect-layer resonance data
        if 'layer_resonance' not in aspect_data:
            aspect_data['layer_resonance'] = {}
        
        # Get aspect properties that might resonate with layers
        aspect_freq = aspect_data.get('details', {}).get('frequency', None)
        aspect_strength = aspect_data.get('strength', 0.1)
        
        # If no frequency, try to infer from other properties
        if aspect_freq is None or aspect_freq <= FLOAT_EPSILON:
            aspect_harmonics = aspect_data.get('details', {}).get('harmonics', [])
            if aspect_harmonics and len(aspect_harmonics) > 0:
                aspect_freq = aspect_harmonics[0]
        
        # Skip aspects without usable frequency information
        if aspect_freq is None or aspect_freq <= FLOAT_EPSILON:
            continue
        
        # Find most resonant layer for this aspect
        best_layer_idx = None
        best_resonance = 0.1  # Minimum threshold for resonance
        
        for i, layer in enumerate(soul_spark.layers):
            # Check direct frequency match with layer's resonant frequencies
            if 'resonant_frequencies' in layer:
                for layer_freq in layer['resonant_frequencies']:
                    if layer_freq <= FLOAT_EPSILON:
                        continue
                    
                    # Calculate frequency resonance
                    res_score = calculate_resonance(aspect_freq, layer_freq)
                    if res_score > best_resonance:
                        best_resonance = res_score
                        best_layer_idx = i
            
            # Also check harmonic data for more sophisticated matching
            if 'harmonic_data' in layer:
                for harmonic_id, harmonic in layer['harmonic_data'].items():
                    if isinstance(harmonic, dict) and 'target_frequency' in harmonic:
                        h_freq = harmonic['target_frequency']
                        if h_freq <= FLOAT_EPSILON:
                            continue
                            
                        res_score = calculate_resonance(aspect_freq, h_freq)
                        if res_score > best_resonance:
                            best_resonance = res_score
                            best_layer_idx = i
        
        # If a resonant layer was found, enhance the connection
        if best_layer_idx is not None:
            layer_key = f'layer_{best_layer_idx}'
            
            # Get current resonance or initialize
            current_resonance = aspect_data['layer_resonance'].get(layer_key, 0.1)
            
            # Calculate resonance improvement based on strength and current level
            max_improvement = 0.15 * aspect_strength  # Stronger aspects improve faster
            potential = best_resonance * (1.0 - current_resonance)
            improvement = max_improvement * potential
            
            # Apply improvement
            new_resonance = min(1.0, current_resonance + improvement)
            aspect_data['layer_resonance'][layer_key] = float(new_resonance)
            
            # Track improvement
            aspect_resonance_improved += (new_resonance - current_resonance) * aspect_strength
            
            logger.debug(f"Aspect '{aspect_name}' resonance with {layer_key}: "
                        f"{current_resonance:.3f}->{new_resonance:.3f} (score={best_resonance:.2f})")
    
    # Calculate coherence gain from aspect-layer resonance improvements
    # Scale to meaningful CU gain - calibrate based on testing
    coherence_gain = aspect_resonance_improved * 8.0
    
    logger.debug(f"Aspect-layer resonance improved by {aspect_resonance_improved:.4f}, "
                f"coherence gain: {coherence_gain:.2f} CU")
    return coherence_gain
    return coherence_gain

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
        phase1_metrics = _harmonize_phase1_establish_pattern(soul_spark, phase1_steps)

        # Add layer integration enhancement
        layer_integration_gain = _enhance_layer_integration(soul_spark)
        phase1_metrics['layer_integration_gain'] = layer_integration_gain
        phase1_metrics['coherence_gain'] += layer_integration_gain
        logger.debug(f"  Phase 1: Added layer integration gain: +{layer_integration_gain:.2f} CU")

        detailed_metrics['phase1'] = phase1_metrics

        if hasattr(soul_spark, 'frequency_signature') and 'phases' in soul_spark.frequency_signature:
            phases = np.array(soul_spark.frequency_signature['phases'])
            phase_var = _calculate_circular_variance(phases)
            logger.debug(f"  Final P1 Phase coherence: {1.0-phase_var:.3f}")
        
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
        phase3_metrics = _harmonize_phase3_align_harmonics(soul_spark, phase3_steps)

        # Add aspect-layer resonance enhancement
        aspect_resonance_gain = _enhance_aspect_layer_resonance(soul_spark)
        phase3_metrics['aspect_resonance_gain'] = aspect_resonance_gain
        phase3_metrics['coherence_gain'] += aspect_resonance_gain
        logger.debug(f"  Phase 3: Added aspect-layer resonance gain: +{aspect_resonance_gain:.2f} CU")

        detailed_metrics['phase3'] = phase3_metrics
        
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
        harmonics_list = soul_spark.harmonics
        base_freq = soul_spark.frequency
        if harmonics_list and base_freq > FLOAT_EPSILON:
            harm_dev = _calculate_harmonic_deviation(harmonics_list, base_freq)
            logger.debug(f"  P3 Step {steps}/{steps}: Harmonic purity: {1.0-harm_dev:.3f}")
    
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