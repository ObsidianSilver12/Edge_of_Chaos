# --- START OF FILE src/stage_1/soul_formation/sephiroth_journey_processing.py ---

"""
Sephiroth Journey Processing (Refactored V4.3.8 - Natural Principles)

Handles soul interaction via Glyph/Layer model. Resonance calculated naturally.
Layers formed based on resonance. Aspects transferred via resonant exchange.
Energy exchanged (SEU). Influence factors modified for indirect S/C effects.
Geometric effects applied to aura. Modifies SoulSpark.
"""

import logging
import numpy as np
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
from math import sqrt, pi as PI, exp, atan2, tanh, log
import random
from constants.constants import *

# --- Logging ---
logger = logging.getLogger(__name__)
# Ensure logger level is set appropriately
try:
    import constants.constants as const # Use alias
    logger.setLevel(const.LOG_LEVEL)
except ImportError:
    logger.warning(
        "Constants not loaded, using default INFO level for "
        "Sephiroth Journey logger."
    )
    logger.setLevel(logging.INFO)

# --- Constants Import (using alias 'const') ---
try:
    import constants.constants as const
except ImportError as e:
    logger.critical(
        "CRITICAL ERROR: constants.py failed import in "
        "sephiroth_journey_processing.py"
    )
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.fields.sephiroth_field import SephirothField
    # Kept for potential future use (e.g., accessing local void properties)
    from stage_1.fields.field_controller import FieldController
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.error("Metrics tracking module not found.")
    METRICS_AVAILABLE = False
    # Define placeholder if module not found
    class MetricsPlaceholder: record_metrics = lambda *a, **kw: None
    metrics = MetricsPlaceholder()

# --- Resonance Helpers (Frequency & Geometric) ---
def calculate_resonance(freq1: float, freq2: float) -> float:
    """
    Calculate frequency resonance (0-1) using perceptual proximity and harmonic relationships.
    """
    if min(freq1, freq2) <= const.FLOAT_EPSILON:
        return 0.0
    
    # Base resonance on logarithmic difference (perceptual proximity)
    log_diff = abs(np.log(freq1) - np.log(freq2))
    log_tolerance_sigma = 0.15  # Controls width of resonance peak
    
    # Primary resonance based on Gaussian proximity
    main_resonance = np.exp(-(log_diff**2) / (2 * log_tolerance_sigma**2))
    
    # Secondary resonance from harmonic relationships
    ratio = max(freq1, freq2) / min(freq1, freq2)
    harmonic_resonance = 0.0
    
    # Check standard harmonics (broader tolerance)
    for i in range(1, 5):
        for j in range(1, 5):
            target_ratio = float(max(i, j)) / float(min(i, j))
            harmonic_diff = abs(ratio - target_ratio)
            harmonic_tolerance = 0.08 * target_ratio  # More forgiving tolerance
            if harmonic_diff < harmonic_tolerance:
                harmonic_score = (1.0 - harmonic_diff/harmonic_tolerance)
                harmonic_resonance = max(harmonic_resonance, harmonic_score)
    
    # Check phi resonance (broader tolerance)
    phi = const.GOLDEN_RATIO
    for i in [1, 2]:
        phi_target = phi**i
        phi_diff = abs(ratio - phi_target)
        phi_tolerance = 0.1 * phi_target  # More forgiving
        if phi_diff < phi_tolerance:
            phi_score = (1.0 - phi_diff/phi_tolerance)
            harmonic_resonance = max(harmonic_resonance, phi_score)
    
    # Combine resonances with bias toward primary resonance
    final_resonance = 0.7 * main_resonance + 0.3 * harmonic_resonance
    
    logger.debug(f"  ResCalc (Enhanced): F1={freq1:.1f}, F2={freq2:.1f}, LogDiff={log_diff:.4f}, Main={main_resonance:.3f}, Harm={harmonic_resonance:.3f} -> Res={final_resonance:.4f}")
    return max(0.0, min(1.0, final_resonance))

def calculate_geometric_resonance(soul_spark: SoulSpark,
                                  sephirah_influencer: SephirothField) -> float:
    """Calculate resonance (0-1) based on geometric compatibility."""
    try:
        soul_geom = getattr(soul_spark, 'geometric_pattern', None)
        sephirah_name = sephirah_influencer.sephirah_name
        sephirah_glyph_info = const.SEPHIROTH_GLYPH_DATA.get(sephirah_name, {})
        sephirah_platonic = sephirah_glyph_info.get('platonic')
        layer_count = len(getattr(soul_spark, 'layers', []))
        stability_factor = getattr(soul_spark, 'stability', 0.0) / const.MAX_STABILITY_SU

        # Base compatibility (0.3 to 0.6) based on stability
        base_compatibility = 0.3 + (stability_factor * 0.3)
        # Bonus for progression (up to 0.3)
        journey_progress_bonus = min(0.3, layer_count * 0.03)
        compatibility = base_compatibility + journey_progress_bonus

        # Bonus/penalty for matching geometries
        match_bonus = 0.0
        if soul_geom and sephirah_platonic:
            if soul_geom == sephirah_platonic: match_bonus = 0.3
        elif soul_geom is None and sephirah_platonic: match_bonus = 0.1
        elif soul_geom and sephirah_platonic is None: match_bonus = -0.1
        compatibility += match_bonus

        # Special resonance bonus
        if sephirah_name == "tiphareth" and layer_count >= 5: compatibility += 0.2

        final_compatibility = max(0.0, min(1.0, compatibility)) # Clamp 0-1
        logger.debug(
            f"  GeomResCalc: Soul={soul_geom}, Seph={sephirah_platonic}, "
            f"Base={base_compatibility:.3f}, LayerB={journey_progress_bonus:.3f}, "
            f"MatchB={match_bonus:.3f} -> Comp={final_compatibility:.4f}"
        )
        return final_compatibility
    except Exception as e:
        logger.error(f"Error in calculate_geometric_resonance: {e}", exc_info=True)
        raise RuntimeError(f"Geometric resonance calculation failed: {e}")

# --- Field Interaction Helper ---
def calculate_field_energy_transfer(source_potential_seu: float,
                                    target_energy_seu: float,
                                    resonance_coupling: float, # Combined factor (0-1)
                                    rate_k: float, 
                                    delta_time: float) -> float:
    """Calculates energy transfer (SEU) based on potential difference and coupling."""
    potential_diff = source_potential_seu - target_energy_seu
    # Transfer proportional to difference * coupling * rate * time
    energy_transfer = potential_diff * resonance_coupling * rate_k * delta_time
    # No arbitrary limit - natural system
    return energy_transfer

# --- Aspect Acquisition Helper ---
def _acquire_sephirah_aspects(soul_spark: SoulSpark,
                              sephirah_influencer: SephirothField,
                              resonance_strength: float
                              ) -> Tuple[Dict[str, float], int, int]:
    """
    Acquires/Strengthens aspects based on resonance. Modifies SoulSpark.aspects.
    Returns dict of {aspect_name: imparted_strength}, gained_count,
    strengthened_count. Fails hard on critical errors.
    """
    sephirah_name = sephirah_influencer.sephirah_name
    logger.debug(f"Acquiring aspects from {sephirah_name} "
                 f"(Resonance: {resonance_strength:.3f})...")
    if not isinstance(sephirah_influencer, SephirothField): raise TypeError("sephirah_influencer invalid.")
    if not (0.0 <= resonance_strength <= 1.0): raise ValueError("Invalid resonance strength.")
    if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict):
        raise AttributeError("Missing 'aspects' dict.")

    try:
        sephirah_aspects_data = sephirah_influencer.get_aspects() # Use getter
        detailed_aspects = sephirah_aspects_data.get('detailed_aspects', {})
        if not detailed_aspects: 
            logger.warning(f"No detailed aspects found for {sephirah_name}.")
            raise ValueError(f"No detailed aspects found for {sephirah_name}.")

        soul_current_aspects = soul_spark.aspects
        imparted_strengths_dict = {}
        gained_count = 0
        strengthened_count = 0
        transfer_time = datetime.now().isoformat()
        
        # Sort aspects by strength for natural prioritization
        sorted_aspects = sorted(
            [(name, details) for name, details in detailed_aspects.items()],
            key=lambda x: float(x[1].get('strength', 0.0)),
            reverse=True
        )

        # NEW: Calculate aspect acquisition threshold based on resonance
        # Lower resonance = higher threshold = fewer aspects gained
        # This creates natural variation based on resonance quality
        base_threshold = 0.25  # Minimum strength to consider (0-1)
        resonance_modifier = 1.0 - (resonance_strength * 0.8)  # Lower is better
        acquisition_threshold = base_threshold * resonance_modifier
        
        logger.debug(f"  Resonance strength: {resonance_strength:.3f}, Acquisition threshold: {acquisition_threshold:.3f}")
        
        # NEW: Apply random variance to aspect acquisition for natural system behavior
        # Higher resonance = lower variance (more predictable outcome)
        variance_range = 0.2 * (1.0 - resonance_strength)
        
        for aspect_name, details in sorted_aspects:
            # Get aspect strength from Sephirah
            seph_str = float(details.get('strength', 0.0))  # Base 0-1
            if seph_str <= const.FLOAT_EPSILON: 
                continue
                
            # NEW: Apply natural resonance filtering effect
            # Only process aspects above dynamic threshold (resonance-based)
            # Plus small random variance for natural behavior
            variance = random.uniform(-variance_range, variance_range) if variance_range > 0.01 else 0
            adjusted_threshold = max(0.0, min(0.95, acquisition_threshold + variance))
            
            if seph_str < adjusted_threshold:
                logger.debug(f"  Aspect '{aspect_name}' below threshold ({seph_str:.3f} < {adjusted_threshold:.3f}), skipping")
                continue

            # Calculate imparted strength using resonance effects
            # More resonance = more efficient transfer, non-linear scaling effect
            # Natural systems increase transfer efficiency with resonance
            resonance_factor = resonance_strength ** 0.8  # Non-linear scaling
            
            # Aspect specificity - each aspect has its own natural resonance profile
            # This creates variation between aspects based on their "compatibility"
            aspect_specificity = random.uniform(0.7, 1.0)  # Natural variation
            
            imparted_strength = (
                seph_str * 
                resonance_factor * 
                aspect_specificity * 
                const.SEPHIROTH_ASPECT_TRANSFER_FACTOR
            )
            
            # Store calculated strength
            imparted_strengths_dict[aspect_name] = imparted_strength

            # Get current soul strength for this aspect (if any)
            current_soul_strength = float(soul_current_aspects.get(aspect_name, {})
                                          .get('strength', 0.0))
                                          
            # Add imparted strength, clamp to max
            new_soul_strength = min(const.MAX_ASPECT_STRENGTH,
                                    current_soul_strength + imparted_strength)
            actual_gain = new_soul_strength - current_soul_strength

            # Only process if meaningful change occurred
            if actual_gain > const.FLOAT_EPSILON:  # If change occurred
                details_copy = details.copy()
                details_copy['last_interaction_time'] = transfer_time
                
                if aspect_name not in soul_current_aspects or current_soul_strength <= const.FLOAT_EPSILON:
                    # Gain new aspect
                    soul_current_aspects[aspect_name] = {
                        'strength': new_soul_strength, 
                        'source': sephirah_name,
                        'time_acquired': transfer_time, 
                        'details': details_copy,
                        'resonance_factor': resonance_factor  # Store for reference
                    }
                    gained_count += 1
                    logger.info(f"Gained NEW aspect '{aspect_name}' from "
                                f"{sephirah_name} strength: {new_soul_strength:.3f} "
                                f"(resonance: {resonance_factor:.3f})")
                else:
                    # Strengthen existing aspect
                    soul_current_aspects[aspect_name]['strength'] = new_soul_strength
                    soul_current_aspects[aspect_name]['last_strengthened'] = transfer_time
                    soul_current_aspects[aspect_name]['details'].update(details_copy)  # Merge details
                    soul_current_aspects[aspect_name]['resonance_factor'] = resonance_factor  # Update
                    strengthened_count += 1
                    logger.info(f"STRENGTHENED aspect '{aspect_name}' "
                                f"{current_soul_strength:.3f} -> {new_soul_strength:.3f} "
                                f"(resonance: {resonance_factor:.3f})")

        # NEW: More nuanced guarantee mechanism for minimum acquisition
        # Only trigger if truly poor resonance and no aspects transferred
        if resonance_strength > const.FLOAT_EPSILON and gained_count == 0 and strengthened_count == 0:
            # Only guarantee ONE primary aspect with minimum possible strength
            primary_aspects = sephirah_aspects_data.get('primary_aspects', [])
            if primary_aspects:
                # Pick the strongest primary aspect for guaranteed minimal transfer
                for aspect_to_gain in primary_aspects:
                    if aspect_to_gain in detailed_aspects:
                        details = detailed_aspects[aspect_to_gain]
                        seph_str = float(details.get('strength', 0.1))
                        
                        # Significantly lower strength for poor resonance guarantee
                        guaranteed_factor = 0.2 * resonance_strength  # Very low transfer
                        min_imparted = max(0.01, seph_str * guaranteed_factor * 
                                         const.SEPHIROTH_ASPECT_TRANSFER_FACTOR)
                        
                        imparted_strengths_dict[aspect_to_gain] = min_imparted
                        curr_str = float(soul_current_aspects.get(aspect_to_gain, {}).get('strength', 0.0))
                        new_str = min(const.MAX_ASPECT_STRENGTH, curr_str + min_imparted)
                        
                        if new_str > curr_str + const.FLOAT_EPSILON:
                            details_copy = details.copy()
                            details_copy['last_interaction_time'] = transfer_time
                            soul_current_aspects[aspect_to_gain] = {
                                'strength': new_str,
                                'source': sephirah_name,
                                'time_acquired': transfer_time,
                                'details': details_copy,
                                'resonance_factor': guaranteed_factor,  # Store low factor
                                'guaranteed_minimum': True  # Mark as minimum guarantee
                            }
                            gained_count = 1
                            logger.info(f"Guaranteed minimum acquisition of '{aspect_to_gain}' "
                                       f"strength={new_str:.3f} (low resonance: {resonance_strength:.3f}).")
                            break  # Only guarantee one aspect
                
        # Apply changes to soul
        if gained_count > 0 or strengthened_count > 0:
             setattr(soul_spark, 'last_modified', transfer_time)
        
        logger.info(f"Aspect acquisition from {sephirah_name}: "
                    f"Gained={gained_count}, Strengthened={strengthened_count} "
                    f"(Resonance: {resonance_strength:.3f})")
        return imparted_strengths_dict, gained_count, strengthened_count
    except Exception as e:
        logger.error(f"Error acquiring aspects from {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError(f"Aspect acquisition failed: {e}")

# --- Geometric Transformation Helper ---
def _apply_geometric_transformation(soul_spark: SoulSpark,
                                    sephirah_influencer: SephirothField,
                                    resonance_strength: float
                                    ) -> Dict[str, float]:
    """Applies geometric effects based on Sephirah's glyph and resonance to aura layers."""
    changes_applied_dict = {}
    sephirah_name = sephirah_influencer.sephirah_name
    try:
        sephirah_glyph_info = const.SEPHIROTH_GLYPH_DATA.get(sephirah_name, {})
        plat_affinity = sephirah_glyph_info.get('platonic')
        if not plat_affinity: return {} # No geometry defined

        effects_to_apply = const.GEOMETRY_EFFECTS.get(plat_affinity, const.DEFAULT_GEOMETRY_EFFECT)
        if not effects_to_apply: return {}

        transformation_occurred = False
        for effect_name, modifier in effects_to_apply.items():
             if modifier == 0.0: continue
             target_attr = effect_name
             is_boost = '_boost' in effect_name or '_factor' in effect_name
             is_push = '_push' in effect_name
             if is_boost: target_attr = effect_name.split('_boost')[0].split('_factor')[0]
             if is_push: target_attr = effect_name.split('_push')[0]

             if not hasattr(soul_spark, target_attr):
                 logger.warning(f"Geom target '{target_attr}' not on SoulSpark."); continue
             current_value = getattr(soul_spark, target_attr)
             if not isinstance(current_value, (int, float)): continue # Skip non-numeric

             change = 0.0
             max_clamp = 1.0 # Default clamp for 0-1 factors
             if target_attr == 'stability': max_clamp = const.MAX_STABILITY_SU
             elif target_attr == 'coherence': max_clamp = const.MAX_COHERENCE_CU

             if is_boost or not is_push: # Additive boost/penalty scaled by resonance
                  change = modifier * resonance_strength * 0.1 # Scaled change
             elif is_push: # Push towards target value
                  push_target = 0.5 if 'balance' in effect_name else 0.0
                  diff = push_target - current_value
                  change = diff * abs(modifier) * resonance_strength * 0.1

             new_value = current_value + change
             clamped_new_value = max(0.0, min(max_clamp, new_value)) # Clamp 0-max
             actual_change = clamped_new_value - current_value

             if abs(actual_change) > const.FLOAT_EPSILON:
                 setattr(soul_spark, target_attr, float(clamped_new_value))
                 changes_applied_dict[f"{target_attr}_geom_delta"] = float(actual_change)
                 transformation_occurred = True

        if transformation_occurred:
             setattr(soul_spark, 'last_modified', datetime.now().isoformat())
             logger.debug(f"Applied geom effects to soul {soul_spark.spark_id} from {sephirah_name}.")
        return changes_applied_dict
    except Exception as e:
        logger.error(f"Error applying geometric transform for {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError(f"Geometric transformation failed: {e}")

# --- Error Recording Helper ---
def record_interaction_failure(soul_id: str, sephirah: str, timestamp: str,
                             error_type: str, error_details: str) -> None:
    """Records interaction failure details for metrics and debugging."""
    failure_data = {
        'soul_id': soul_id,
        'sephirah': sephirah,
        'timestamp': timestamp,
        'error_type': error_type,
        'error_details': error_details
    }
    
    if METRICS_AVAILABLE:
        metrics.record_metrics('sephiroth_journey_failure', failure_data)
    
    logger.error(f"Interaction failure recorded: {failure_data}")

# --- Layer Resonance Helper ---
def _develop_layer_resonance(soul_spark: SoulSpark, target_frequency: float, resonance_strength: float) -> Dict[str, Any]:
    """
    Add resonance data to the soul's outermost layer based on target frequency.
    This is a key function for aura layer development.
    Returns dict with metrics about the resonance development process.
    """
    if not hasattr(soul_spark, 'layers') or not soul_spark.layers:
        logger.error(f"Cannot develop layer resonance: Soul {soul_spark.spark_id} has no layers")
        raise ValueError("Soul has no layers for resonance development")

    metrics = {
        'harmonic_ratio': '',
        'ratio_value': 0.0,
        'resonance_effort': 0.0,
        'density_gain': 0.0,
        'uniformity_gain': 0.0
    }

    # Get most recent layer to add resonance properties
    latest_layer = soul_spark.layers[-1]
    
    # Initialize resonant frequencies array if not present
    if 'resonant_frequencies' not in latest_layer:
        latest_layer['resonant_frequencies'] = []
        
    # Initialize harmonic_data if not present
    if 'harmonic_data' not in latest_layer:
        latest_layer['harmonic_data'] = {}
    
    # Record soul's base frequency for reference
    base_freq = soul_spark.frequency
    if base_freq <= const.FLOAT_EPSILON:
        logger.error(f"Invalid base frequency {base_freq} for soul {soul_spark.spark_id}")
        raise ValueError(f"Soul base frequency invalid: {base_freq}")
    
    # Calculate natural harmonic relationships
    harmonic_matches = []
    
    # Check integer ratios (1:1, 2:1, 3:2, etc.)
    for n in range(1, 6):
        for d in range(1, 6):
            if n == 0 or d == 0:
                continue
                
            ratio_value = float(n) / float(d)
            harmonic_freq = base_freq * ratio_value
            # Calculate how close this harmonic is to target
            deviation = abs(harmonic_freq - target_frequency) / max(target_frequency, const.FLOAT_EPSILON)
            # Only consider close matches
            if deviation < 0.1:
                harmonic_matches.append({
                    'ratio': f"{n}:{d}",
                    'ratio_value': ratio_value,
                    'frequency': harmonic_freq,
                    'deviation': deviation,
                    'is_integer': True
                })
    
    # Check phi-based ratios (natural growth patterns)
    phi_ratios = [('phi', PHI), ('1/phi', 1.0/PHI), ('phi²', PHI**2), ('1/phi²', 1.0/(PHI**2))]
    for name, ratio in phi_ratios:
        harmonic_freq = base_freq * ratio
        deviation = abs(harmonic_freq - target_frequency) / max(target_frequency, const.FLOAT_EPSILON)
        if deviation < 0.1:
            harmonic_matches.append({
                'ratio': name,
                'ratio_value': ratio,
                'frequency': harmonic_freq,
                'deviation': deviation,
                'is_integer': False
            })
    
    # If no natural harmonics found, create a custom one
    if not harmonic_matches:
        custom_ratio = target_frequency / base_freq
        harmonic_matches.append({
            'ratio': f"custom",
            'ratio_value': custom_ratio,
            'frequency': target_frequency,
            'deviation': 0.0,
            'is_integer': False
        })
    
    # Sort by deviation (lowest first)
    harmonic_matches.sort(key=lambda x: x['deviation'])
    best_match = harmonic_matches[0]
    
    # Calculate resonance effort (lower with stronger resonance)
    effort_base = best_match['deviation'] * (1.0 - resonance_strength)
    # Prefer simpler ratios
    ratio_complexity = 0.0
    if best_match['is_integer']:
        num, denom = best_match['ratio'].split(':')
        ratio_complexity = (int(num) + int(denom)) / 20.0  # Scale to 0-1
    else:
        # Phi ratios get lower complexity score
        ratio_complexity = 0.1
    
    total_effort = min(1.0, effort_base + ratio_complexity * 0.3)
    
    # Record this resonance in the layer
    harmonic_data = {
        'target_frequency': float(target_frequency),
        'harmonic_ratio': best_match['ratio'],
        'ratio_value': float(best_match['ratio_value']),
        'resonance_strength': float(resonance_strength),
        'resonance_effort': float(total_effort)
    }
    
    # Store in layer data
    latest_layer['harmonic_data'][str(len(latest_layer['harmonic_data']))] = harmonic_data
    if target_frequency not in latest_layer['resonant_frequencies']:
        latest_layer['resonant_frequencies'].append(float(target_frequency))
        
    # Limit stored frequencies to avoid bloat
    if len(latest_layer['resonant_frequencies']) > 5:
        latest_layer['resonant_frequencies'] = latest_layer['resonant_frequencies'][-5:]
    
    # Enhance layer density/quality based on resonance
    current_density = latest_layer['density']['base_density']
    # Good resonance with low effort = more density improvement
    density_boost = max(0.0, 0.15 * (1.0 - total_effort) * resonance_strength)
    new_density = min(1.0, current_density + density_boost)
    latest_layer['density']['base_density'] = float(new_density)
    
    # Update metrics
    metrics['harmonic_ratio'] = best_match['ratio']
    metrics['ratio_value'] = best_match['ratio_value']
    metrics['resonance_effort'] = total_effort
    metrics['density_gain'] = new_density - current_density
    
    # Enhance uniformity if resonance is strong
    if 'uniformity' in latest_layer['density'] and resonance_strength > 0.7:
        current_uniformity = latest_layer['density']['uniformity']
        uniformity_boost = max(0.0, 0.1 * resonance_strength * (1.0 - current_uniformity))
        new_uniformity = min(1.0, current_uniformity + uniformity_boost)
        latest_layer['density']['uniformity'] = float(new_uniformity)
        metrics['uniformity_gain'] = new_uniformity - current_uniformity
    
    logger.debug(f"Layer resonance developed: Ratio={best_match['ratio']}, " 
                f"Effort={total_effort:.4f}, Density: {current_density:.3f}->{new_density:.3f}")
                
    # Record interaction in soul metadata if tracking is available
    if hasattr(soul_spark, 'interaction_history'):
        interaction_data = {
            'type': 'layer_resonance_development',
            'timestamp': datetime.now().isoformat(),
            'target_frequency': float(target_frequency),
            'harmonic_ratio': best_match['ratio'],
            'ratio_value': float(best_match['ratio_value']),
            'resonance_strength': float(resonance_strength),
            'effort': float(total_effort),
            'density_boost': float(density_boost)
        }
        soul_spark.interaction_history.append(interaction_data)
    
    return metrics

# --- Layer Formation Helper (Applies Influence Factor) ---
def _form_sephirah_layer(soul_spark: SoulSpark,
                         sephirah_influencer: SephirothField,
                         resonance_strength: float) -> Dict[str, Any]:
    """
    Forms layer, transfers energy, applies geometrics effect and influence.
    Modifies SoulSpark. Returns changes dict. Hard fails on errors.
    """
    sephirah_name = sephirah_influencer.sephirah_name
    timestamp = datetime.now().isoformat()
    # Initialize changes dict focusing on influence
    changes = {
        'energy_transfer_seu': 0.0, 
        'sephirah_influence_increment': 0.0, 
        'phase_coherence_improvement': 0.0,
        'stability_before': soul_spark.stability,
        'coherence_before': soul_spark.coherence,
        'energy_before': soul_spark.energy
    }
    
    try:
        # 1. Calculate Energy Transfer (SEU)
        sephirah_potential_seu = sephirah_influencer.get_target_energy_seu()
        # Coupling depends on resonance and soul coherence
        coupling = resonance_strength * (soul_spark.coherence / const.MAX_COHERENCE_CU)
        coupling = max(0.0, min(1.0, coupling)) # Clamp 0-1
        energy_transfer = calculate_field_energy_transfer(
            sephirah_potential_seu, soul_spark.energy, coupling,
            const.SEPHIROTH_ENERGY_EXCHANGE_RATE_K, delta_time=1.0 # Use rate constant
        )
        soul_spark.energy = min(const.MAX_SOUL_ENERGY_SEU,
                                max(0.0, soul_spark.energy + energy_transfer))
        changes['energy_transfer_seu'] = energy_transfer

        # 2. Determine Layer Properties - Enhanced with natural resonance effects
        # Base density from resonance but with non-linear scaling for more naturalism
        # Higher resonance = higher density, but with natural curve
        base_density = resonance_strength ** 0.8  # Slight curve for naturalism
        
        # Apply minor random variance for organic feel (smaller at high resonance)
        variance_range = 0.1 * (1.0 - resonance_strength)
        density_variance = random.uniform(-variance_range, variance_range)
        layer_density = max(0.1, min(1.0, base_density + density_variance))
        
        # Pattern distortion inversely related to stability - more stable = less distortion
        pattern_distortion = soul_spark.get_pattern_distortion()
        
        # Layer uniformity - how even the energy distribution is
        # Higher resonance = more uniform distribution
        base_uniformity = 1.0 - pattern_distortion
        uniformity_boost = resonance_strength * 0.2  # Resonance improves uniformity
        layer_uniformity = max(0.0, min(1.0, base_uniformity + uniformity_boost))
        
        # NEW: Add resonant vibration effect for high resonance layers
        # More resonant interactions create more active, vibrant layers
        vibration_factor = 0.0
        if resonance_strength > 0.7:
            vibration_factor = (resonance_strength - 0.7) * 3.0  # Scale 0-0.9
            vibration_factor = min(0.9, vibration_factor)  # Cap at 0.9
        
        # NEW: Add depth parameter for 3D visual richness
        # More resonant = deeper penetration of layer into soul structure
        layer_depth = resonance_strength * 0.8
        
        # Create enhanced density map with proper structure
        density_map = {
            'base_density': layer_density,
            'uniformity': layer_uniformity,
            'vibration_factor': vibration_factor,
            'depth': layer_depth
        }

        # 3. Add Layer to Soul - Enhanced with resonance-based color intensity
        layer_color_hex = '#FFFFFF'  # Fallback
        try:
            # Get target color from Sephirah
            rgb = sephirah_influencer.target_color_rgb
            if rgb is not None and len(rgb) == 3:
                # Scale color intensity based on resonance for more vivid colors with stronger resonance
                intensity_factor = 0.6 + (resonance_strength * 0.4)  # 0.6 to 1.0 range
                r, g, b = [min(255, int(c * 255 * intensity_factor)) for c in rgb]
                layer_color_hex = f'#{r:02x}{g:02x}{b:02x}'
        except Exception as color_err:
            logger.warning(f"Could not format color: {color_err}")

        # Pass enhanced density_map directly to add_layer
        soul_spark.add_layer(sephirah_name, density_map, layer_color_hex, timestamp)

        # 4. Calculate & Apply Influence Factor Increment - proportional to resonance
        # No artificial boosting - natural relationship
        base_increment = resonance_strength * const.SEPHIRAH_INFLUENCE_RATE_K
        
        # Natural enhancement for strong resonance (non-linear curve)
        if resonance_strength > 0.3:
            # Additional boost for strong resonance follows natural growth curve
            resonance_boost = (resonance_strength - 0.3) * 0.5
            influence_increment = base_increment * (1.0 + resonance_boost)
        else:
            influence_increment = base_increment
            
        # Small acceleration for early development (natural development principle)
        if len(soul_spark.layers) <= 5:
            influence_increment *= 1.2  # 20% boost for early layers
            
        current_influence = getattr(soul_spark, 'cumulative_sephiroth_influence', 0.0)
        new_influence = min(1.0, current_influence + influence_increment) # Clamp 0-1
        
        setattr(soul_spark, 'cumulative_sephiroth_influence', new_influence)
        changes['sephirah_influence_increment'] = influence_increment # Record increment
        
        logger.debug(f"  Sephiroth influence: {current_influence:.3f} -> {new_influence:.3f} (+{influence_increment:.3f})")

        # 5. Optimize Phase Coherence if resonance is strong
        phase_improvement = 0.0
        if resonance_strength > 0.2 and hasattr(soul_spark, '_optimize_phase_coherence'):
            # Stronger resonance = more aggressive phase optimization
            phase_factor = min(0.15, 0.05 + resonance_strength * 0.1)
            phase_improvement = soul_spark._optimize_phase_coherence(phase_factor)
            changes['phase_coherence_improvement'] = phase_improvement

        # 6. Apply Geometric Transformation Effects to aura
        geom_changes = _apply_geometric_transformation(soul_spark, sephirah_influencer, resonance_strength)
        changes['geometric_effects'] = geom_changes # Store dict of geometric changes

        # 7. Store Interaction Metadata
        interaction_data = {
            'type': 'sephirah_layer_formation', 'timestamp': timestamp,
            'sephirah': sephirah_name, 'resonance': resonance_strength,
            'energy_transfer_seu': energy_transfer,
            'influence_increment': influence_increment, # Store influence increment
            'phase_improvement': phase_improvement,
            'layer_density': layer_density, 'layer_uniformity': layer_uniformity,
            'layer_vibration': vibration_factor, 'layer_depth': layer_depth,
            'geom_changes_applied': list(geom_changes.keys())
        }
        if hasattr(soul_spark, 'interaction_history'):
            soul_spark.interaction_history.append(interaction_data)
        setattr(soul_spark, 'last_modified', timestamp)

        logger.debug(f"Formed layer for {sephirah_name}. "
                    f"E_trans={energy_transfer:.2f}, "
                    f"InfluenceInc={influence_increment:.4f}, "
                    f"PhaseImprove={phase_improvement:.4f}")
        return changes

    except Exception as e:
        logger.error(f"Error forming Sephirah layer for {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError(f"Layer formation failed for {sephirah_name}: {e}")


def process_sephirah_interaction(soul_spark: SoulSpark,
                              sephirah_influencer: SephirothField,
                              field_controller: 'FieldController', 
                              duration: float
                              ) -> Tuple[SoulSpark, Dict[str, Any]]:
   """
   Processes interaction: Resonance -> Aspects -> Layer/Energy/Influence/Geometry
   -> Update State. Modifies SoulSpark through aura layers, not direct SU/CU.
   """
   # --- Input Validation & Setup ---
   if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
   if not isinstance(sephirah_influencer, SephirothField): raise TypeError("sephirah_influencer invalid.")
   if not isinstance(duration, (int, float)) or duration <= const.FLOAT_EPSILON: raise ValueError("Duration must be positive.")

   # --- Variables Accessible Throughout ---
   sephirah_name = sephirah_influencer.sephirah_name
   spark_id = getattr(soul_spark, 'spark_id', 'unknown')

   log_start = (f"--- Starting SephInteraction [Natural Principles]: " 
                f"Soul {spark_id} with {sephirah_name.capitalize()} "
                f"(Dur: {duration}) ---")
   logger.info(log_start)
   start_time_iso = datetime.now().isoformat()
   start_time_dt = datetime.fromisoformat(start_time_iso)

   try:
       # Store initial state
       initial_energy = soul_spark.energy
       initial_stability = soul_spark.stability
       initial_coherence = soul_spark.coherence

       # --- 1. Calculate Resonance ---
       logger.info(f"  {sephirah_name}: Step 1: Calculating resonance...")
       # Frequency resonance - how well soul and sephirah frequencies align
       freq_res = calculate_resonance(soul_spark.frequency,
                                   sephirah_influencer.target_frequency)
       # Geometric resonance - how well soul and sephirah geometries align
       geom_res = calculate_geometric_resonance(soul_spark, sephirah_influencer)
       
       # Combine using weighted average - natural relationship
       total_weight = (const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ +
                       const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM +
                       const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI)
       if total_weight <= const.FLOAT_EPSILON: total_weight = 1.0
       
       # Phi term reflects golden ratio synergy between frequency and geometry
       resonance_strength = (
           (freq_res * const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ +
           geom_res * const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM +
           (freq_res * geom_res * const.GOLDEN_RATIO) * const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI)
           / total_weight
       )
       resonance_strength = max(0.0, min(1.0, resonance_strength))  # Clamp 0-1
       logger.info(f"  {sephirah_name}: Resonance - Freq={freq_res:.3f}, Geom={geom_res:.3f} -> Total={resonance_strength:.4f}")

       # --- 2. Acquire Aspects ---
       logger.info(f"  {sephirah_name}: Step 2: Acquiring aspects...")
       imparted_aspects, gained_count, strengthened_count = _acquire_sephirah_aspects(
           soul_spark, sephirah_influencer, resonance_strength
       )
       # Detailed logging happens inside _acquire_sephirah_aspects

       # --- 3. Form Layer, Transfer Energy, Apply Effects ---
       logger.info(f"  {sephirah_name}: Step 3: Forming layer, applying energy/influence/geometry...")
       layer_changes = _form_sephirah_layer(
           soul_spark, sephirah_influencer, resonance_strength
       )
       log_layer = (f"  {sephirah_name}: Layer     - "
                   f"dE={layer_changes.get('energy_transfer_seu', 0.0):+.2f}, "
                   f"dInf={layer_changes.get('sephirah_influence_increment', 0.0):+.5f}")
       logger.info(log_layer)
       if layer_changes.get('geometric_effects'):
           logger.info(f"  {sephirah_name}: Geometry  - Changes: {layer_changes['geometric_effects']}")
       
       # --- 4. Develop Layer Resonance ---
       logger.info(f"  {sephirah_name}: Step 4: Developing layer resonance...")
       
       layer_resonance_metrics = _develop_layer_resonance(
           soul_spark, sephirah_influencer.target_frequency, resonance_strength
       )
       
       # Log layer resonance results
       if layer_resonance_metrics:
           layer_ratio = layer_resonance_metrics.get('harmonic_ratio', 'unknown')
           layer_effort = layer_resonance_metrics.get('resonance_effort', 0.0)
           density_gain = layer_resonance_metrics.get('density_gain', 0.0)
           uniformity_gain = layer_resonance_metrics.get('uniformity_gain', 0.0)
           
           logger.info(f"  {sephirah_name}: Resonance - Ratio={layer_ratio}, "
                     f"Effort={layer_effort:.3f}, Density+={density_gain:.3f}, "
                     f"Uniformity+={uniformity_gain:.3f}")
           
           # Add resonance metrics to layer_changes for overall metrics
           layer_changes['layer_resonance'] = layer_resonance_metrics
       
       # --- 5. Update Soul's Emergent State (SU/CU Scores) ---
       # This is where indirect effects manifest in SU/CU
       logger.info(f"  {sephirah_name}: Step 5: Updating internal state scores...")
       if hasattr(soul_spark, 'update_state'):
           soul_spark.update_state()
       else:
           logger.error("SoulSpark object missing 'update_state' method!")
           raise AttributeError("SoulSpark needs 'update_state' method.")
           
       # --- 6. Compile Complete Metrics ---
       # Include before/after energy, stability, coherence
       final_stability = soul_spark.stability
       final_coherence = soul_spark.coherence
       final_energy = soul_spark.energy
       
       step_changes = {
           'energy_transfer_seu': layer_changes.get('energy_transfer_seu', 0.0),
           'sephirah_influence_increment': layer_changes.get('sephirah_influence_increment', 0.0),
           'phase_coherence_improvement': layer_changes.get('phase_coherence_improvement', 0.0),
           'geometric_effects': layer_changes.get('geometric_effects', {}),
           'layer_resonance': layer_changes.get('layer_resonance', {}),
           'stability_before': initial_stability,
           'stability_after': final_stability,
           'stability_change': final_stability - initial_stability,
           'coherence_before': initial_coherence,
           'coherence_after': final_coherence,
           'coherence_change': final_coherence - initial_coherence,
           'energy_before': initial_energy,
           'energy_after': final_energy,
           'energy_change': final_energy - initial_energy
       }
           
       end_time_iso = datetime.now().isoformat()
       end_time_dt = datetime.fromisoformat(end_time_iso)
       
       interaction_metrics = {
           'action': 'sephirah_interaction',
           'soul_id': spark_id,
           'sephirah': sephirah_name,
           'start_time': start_time_iso,
           'end_time': end_time_iso,
           'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
           'simulated_duration': duration,
           'resonance_achieved': resonance_strength,
           'freq_resonance_component': freq_res,
           'geom_resonance_component': geom_res,
           'aspects_gained_count': gained_count,
           'aspects_strengthened_count': strengthened_count,
           'layer_formation_changes': step_changes,
           'stability_before': initial_stability,
           'stability_after': final_stability,
           'stability_change': final_stability - initial_stability,
           'coherence_before': initial_coherence,
           'coherence_after': final_coherence,
           'coherence_change': final_coherence - initial_coherence,
           'energy_before': initial_energy,
           'energy_after': final_energy,
           'energy_change': final_energy - initial_energy,
           'success': True,
       }

       if METRICS_AVAILABLE:
           metrics.record_metrics('sephiroth_journey_step', interaction_metrics)

       return soul_spark, interaction_metrics
   except Exception as e:
       logger.error(f"Error during Sephirah interaction for {soul_spark.spark_id} in "
                  f"{sephirah_name}: {e}", exc_info=True)
       raise RuntimeError(f"Sephirah interaction failed for {sephirah_name}: {e}")
# --- END OF FILE src/stage_1/soul_formation/sephiroth_journey_processing.py --- 