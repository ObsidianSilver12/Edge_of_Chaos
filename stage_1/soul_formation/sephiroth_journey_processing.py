# --- START OF FILE src/stage_1/soul_formation/sephiroth_journey_processing.py ---

"""
Sephiroth Journey Processing (Refactored V4.3 - Principle-Driven S/C, PEP8)

Handles soul interaction via Glyph/Layer model. Resonance calculated. Layers formed.
Aspects transferred. Energy exchanged (SEU). *Influence factors* modified for S/C.
Geometric effects applied. Metadata stored. Modifies SoulSpark.
Adheres to PEP 8 formatting.
"""

import logging
import numpy as np
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
from constants.constants import * # Import constants directly for clarity

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
        return 0.1 # Default low resonance on error

# --- Field Interaction Helper ---
def calculate_field_energy_transfer(source_potential_seu: float,
                                    target_energy_seu: float,
                                    resonance_coupling: float, # Combined factor (0-1)
                                    rate_k: float, delta_time: float) -> float:
    """Calculates energy transfer (SEU) based on potential difference and coupling."""
    potential_diff = source_potential_seu - target_energy_seu
    # Transfer proportional to difference * coupling * rate * time
    energy_transfer = potential_diff * resonance_coupling * rate_k * delta_time
    # Optional limit removed for now
    return energy_transfer

# --- Aspect Acquisition Helper ---
def _acquire_sephirah_aspects(soul_spark: SoulSpark,
                              sephirah_influencer: SephirothField,
                              resonance_strength: float
                              ) -> Tuple[Dict[str, float], int, int]:
    """
    Acquires/Strengthens aspects based on resonance. Modifies SoulSpark.aspects.
    NO threshold. Returns dict of {aspect_name: imparted_strength}, gained_count,
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
        if not detailed_aspects: logger.warning(f"No detailed aspects found for {sephirah_name}."); return {}, 0, 0

        soul_current_aspects = soul_spark.aspects; imparted_strengths_dict = {}
        gained_count = 0; strengthened_count = 0; transfer_time = datetime.now().isoformat()

        for aspect_name, details in detailed_aspects.items():
            seph_str = float(details.get('strength', 0.0)) # Base 0-1
            if seph_str <= const.FLOAT_EPSILON: continue

            # Calculate imparted strength (0-1 scale)
            # Use resonance^1.5 for non-linear scaling, transfer factor constant
            imparted_strength = max(0.0, seph_str * (resonance_strength**1.5) *
                                    const.SEPHIROTH_ASPECT_TRANSFER_FACTOR)
            imparted_strengths_dict[aspect_name] = imparted_strength # Record calculation

            current_soul_strength = float(soul_current_aspects.get(aspect_name, {})
                                          .get('strength', 0.0))
            # Add imparted strength, clamp to max
            new_soul_strength = min(const.MAX_ASPECT_STRENGTH,
                                    current_soul_strength + imparted_strength)
            actual_gain = new_soul_strength - current_soul_strength

            if actual_gain > const.FLOAT_EPSILON: # If change occurred
                details_copy = details.copy(); details_copy['last_interaction_time'] = transfer_time
                if aspect_name not in soul_current_aspects or current_soul_strength <= const.FLOAT_EPSILON:
                    # Gain new aspect
                    soul_current_aspects[aspect_name] = {
                        'strength': new_soul_strength, 'source': sephirah_name,
                        'time_acquired': transfer_time, 'details': details_copy
                    }
                    gained_count += 1
                    logger.info(f"Gained NEW aspect '{aspect_name}' from "
                                f"{sephirah_name} strength: {new_soul_strength:.3f}")
                else:
                    # Strengthen existing aspect
                    soul_current_aspects[aspect_name]['strength'] = new_soul_strength
                    soul_current_aspects[aspect_name]['last_strengthened'] = transfer_time
                    soul_current_aspects[aspect_name]['details'].update(details_copy) # Merge details
                    strengthened_count += 1
                    logger.info(f"STRENGTHENED aspect '{aspect_name}' "
                                f"{current_soul_strength:.3f} -> {new_soul_strength:.3f}")

        # Guarantee minimum gain if resonance > 0 and nothing happened
        if resonance_strength > const.FLOAT_EPSILON and gained_count == 0 and strengthened_count == 0:
            primary_aspects = sephirah_aspects_data.get('primary_aspects', [])
            if primary_aspects:
                aspect_to_gain = primary_aspects[0]
                if aspect_to_gain in detailed_aspects:
                    details=detailed_aspects[aspect_to_gain]; seph_str=float(details.get('strength',0.1))
                    min_imparted = max(0.01, seph_str*(resonance_strength**1.5)*const.SEPHIROTH_ASPECT_TRANSFER_FACTOR)
                    imparted_strengths_dict[aspect_to_gain] = min_imparted
                    curr_str=float(soul_current_aspects.get(aspect_to_gain,{}).get('strength',0.0))
                    new_str=min(const.MAX_ASPECT_STRENGTH, curr_str + min_imparted)
                    if new_str > curr_str + const.FLOAT_EPSILON:
                        details_copy=details.copy(); details_copy['last_interaction_time']=transfer_time
                        soul_current_aspects[aspect_to_gain]={'strength':new_str,'source':sephirah_name,'time_acquired':transfer_time,'details':details_copy}
                        gained_count = 1
                        logger.info(f"Guaranteed minimum acquisition of '{aspect_to_gain}' strength={new_str:.3f}.")

        if gained_count > 0 or strengthened_count > 0:
             setattr(soul_spark, 'last_modified', transfer_time)
        logger.info(f"Aspect acquisition from {sephirah_name}: "
                    f"Gained={gained_count}, Strengthened={strengthened_count}")
        return imparted_strengths_dict, gained_count, strengthened_count
    except Exception as e:
        logger.error(f"Error acquiring aspects from {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError("Aspect acquisition failed.") from e

# --- Geometric Transformation Helper ---
def _apply_geometric_transformation(soul_spark: SoulSpark,
                                    sephirah_influencer: SephirothField,
                                    resonance_strength: float
                                    ) -> Dict[str, float]:
    """Applies geometric effects based on Sephirah's glyph and resonance."""
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
             target_attr=effect_name; is_boost='_boost' in effect_name or '_factor' in effect_name; is_push='_push' in effect_name
             if is_boost: target_attr = effect_name.split('_boost')[0].split('_factor')[0]
             if is_push: target_attr = effect_name.split('_push')[0]

             if not hasattr(soul_spark, target_attr):
                 logger.warning(f"Geom target '{target_attr}' not on SoulSpark."); continue
             current_value = getattr(soul_spark, target_attr)
             if not isinstance(current_value, (int, float)): continue # Skip non-numeric

             change=0.0; max_clamp=1.0 # Default clamp for 0-1 factors
             if target_attr=='stability': max_clamp=const.MAX_STABILITY_SU
             elif target_attr=='coherence': max_clamp=const.MAX_COHERENCE_CU

             if is_boost or not is_push: # Additive boost/penalty
                  change = modifier * resonance_strength * 0.1 # Scaled change
             elif is_push: # Push towards target value
                  push_target=0.5 if 'balance' in effect_name else 0.0
                  diff=push_target-current_value; change=diff*abs(modifier)*resonance_strength*0.1

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
        return {}

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

# --- Frequency Nudging Helper ---
def _apply_frequency_nudging(soul_spark: SoulSpark, target_frequency: float, resonance_strength: float) -> None:
    """
    Adjust soul's frequency toward target based on resonance strength.
    Strong resonance = stronger nudge toward target frequency.
    """
    if not hasattr(soul_spark, 'frequency'):
        logger.warning("Soul missing frequency attribute - skipping nudge")
        return

    current_freq = soul_spark.frequency
    freq_diff = target_frequency - current_freq
    
    # Scale adjustment by resonance (stronger resonance = bigger adjustment)
    adjustment_factor = resonance_strength * 0.1  # Max 10% adjustment per step
    freq_adjustment = freq_diff * adjustment_factor
    
    new_freq = current_freq + freq_adjustment
    # Ensure frequency stays positive and within reasonable bounds
    new_freq = max(0.1, min(1000.0, new_freq))
    
    soul_spark.frequency = new_freq
    logger.debug(f"Frequency nudged: {current_freq:.1f} -> {new_freq:.1f} Hz (target={target_frequency:.1f})")

# --- Layer Formation Helper (Applies Influence Factor) ---
def _form_sephirah_layer(soul_spark: SoulSpark,
                         sephirah_influencer: SephirothField,
                         resonance_strength: float) -> Dict[str, Any]:
    """
    Forms layer, transfers energy, applies geom effects, INCREMENTS INFLUENCE.
    Modifies SoulSpark. Returns dict of changes. Fails hard.
    """
    sephirah_name = sephirah_influencer.sephirah_name
    timestamp = datetime.now().isoformat()
    # Initialize changes dict focusing on influence
    changes = {'energy_transfer_seu': 0.0, 'sephirah_influence_increment': 0.0, 'phase_coherence_improvement': 0.0}
    layer_details = {'density': 0.0, 'uniformity': 0.0}

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

        # 2. Determine Layer Properties
        layer_density = resonance_strength # Simple mapping
        pattern_distortion = soul_spark.get_pattern_distortion() # 0=perfect, 1=max distortion
        layer_uniformity = max(0.0, min(1.0, 1.0 - pattern_distortion))
        density_map_placeholder = {'base_density': layer_density, 'uniformity': layer_uniformity}
        layer_details.update(density_map_placeholder)

        # 3. Add Layer to Soul
        layer_color_hex = '#FFFFFF' # Fallback
        try: # Safely format color
            rgb = sephirah_influencer.target_color_rgb
            if rgb is not None and len(rgb) == 3:
                 r, g, b = [int(c * 255) for c in rgb]
                 layer_color_hex = f'#{r:02x}{g:02x}{b:02x}'
        except Exception as color_err: logger.warning(f"Could not format color: {color_err}")
        soul_spark.add_layer(sephirah_name, density_map_placeholder, layer_color_hex, timestamp)

        # 4. Calculate & Apply Influence Factor Increment
        # Enhanced influence increment with resonance boost
        base_increment = resonance_strength * const.SEPHIRAH_INFLUENCE_RATE_K  # Base rate
        
        # Boost for higher resonance values - nonlinear scaling
        resonance_boost = 0.0
        if resonance_strength > 0.3:
            resonance_boost = (resonance_strength - 0.3) * 0.5  # Additional boost for strong resonance
        
        influence_increment = base_increment * (1.0 + resonance_boost)
        
        # Scale up for early layers to accelerate progress
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

        # 6. Apply Geometric Transformation Effects (Modifies factors/attributes)
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
        logger.error(f"Error forming Sephirah layer for {sephirah_name}: {e}",
                     exc_info=True)
        raise RuntimeError(f"Layer formation failed for {sephirah_name}") from e


# --- Main Orchestration Function (Uses Influence Factors) ---
def process_sephirah_interaction(soul_spark: SoulSpark,
                                 sephirah_influencer: SephirothField,
                                 field_controller: 'FieldController', # Keep for context
                                 duration: float
                                 ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Processes interaction: Resonance -> Nudge Freq -> Aspects -> Layer/Energy/Influence/Geometry
    -> Update State. Modifies SoulSpark. Includes DETAILED LOGGING. Fails hard.
    """
    # --- Input Validation & Setup ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(sephirah_influencer, SephirothField): raise TypeError("sephirah_influencer invalid.")
    if not isinstance(duration, (int, float)) or duration <= const.FLOAT_EPSILON: raise ValueError("Duration must be positive.")

    # --- Variables Accessible Throughout ---
    sephirah_name = sephirah_influencer.sephirah_name
    spark_id = getattr(soul_spark, 'spark_id', 'unknown')

    log_start = (f"--- Starting SephInteraction [Principle-Driven]: " # Changed log prefix
                 f"Soul {spark_id} with {sephirah_name.capitalize()} "
                 f"(Dur: {duration}) ---")
    logger.info(log_start)
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)

    # --- Metrics variables ---
    interaction_metrics: Dict[str, Any] = {} # Initialize
    initial_state_metrics: Dict[str, Any] = {}
    final_state_metrics: Dict[str, Any] = {}
    imparted_aspects: Dict[str, float] = {}
    step_changes: Dict[str, Any] = {}
    gained_count = 0
    strengthened_count = 0
    freq_res = 0.0
    geom_res = 0.0
    resonance_strength = 0.0

    try:
        # --- Capture Initial State for this interaction ---
        e_before = soul_spark.energy
        s_before = soul_spark.stability
        c_before = soul_spark.coherence
        inf_before = getattr(soul_spark, 'cumulative_sephiroth_influence', 0.0)
        freq_before = soul_spark.frequency
        initial_state_metrics = { # Capture for final report
            'energy_seu': e_before, 'stability_su': s_before,
            'coherence_cu': c_before, 'aspect_count': len(soul_spark.aspects),
            'layer_count': len(soul_spark.layers)
        }
        logger.info(f"  {sephirah_name}: BEFORE - E={e_before:.1f}, S={s_before:.1f}, C={c_before:.1f}, CumulInf={inf_before:.4f}, Freq={freq_before:.1f}Hz")

        # --- 1. Calculate Resonance ---
        logger.info(f"  {sephirah_name}: Step 1: Calculating resonance...")
        freq_res = calculate_resonance(soul_spark.frequency,
                                       sephirah_influencer.target_frequency)
        geom_res = calculate_geometric_resonance(soul_spark, sephirah_influencer)
        # Combine using weighted average
        total_weight = (const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ +
                        const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM +
                        const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI)
        if total_weight <= const.FLOAT_EPSILON: total_weight = 1.0
        resonance_strength = (
            (freq_res * const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ +
             geom_res * const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM +
             (freq_res * geom_res * const.PHI) * const.SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI) # Used PHI constant
            / total_weight
        )
        resonance_strength = max(0.0, min(1.0, resonance_strength)) # Clamp 0-1
        logger.info(f"  {sephirah_name}: Resonance - Freq={freq_res:.3f}, Geom={geom_res:.3f} -> Total={resonance_strength:.4f}")

        # --- Action 1b: Enhanced Frequency Nudging ---
        logger.info(f"  {sephirah_name}: Starting frequency nudging...")
        _apply_frequency_nudging(soul_spark, sephirah_influencer.target_frequency, resonance_strength)

        # --- 2. Acquire Aspects ---
        logger.info(f"  {sephirah_name}: Step 2: Acquiring aspects...")
        imparted_aspects, gained_count, strengthened_count = _acquire_sephirah_aspects(
            soul_spark, sephirah_influencer, resonance_strength
        )
        # Detailed logging happens inside _acquire_sephirah_aspects

        # --- 3. Form Layer, Transfer Energy, Apply Effects (Modifies Influence) ---
        logger.info(f"  {sephirah_name}: Step 3: Forming layer, applying energy/influence/geometry...")
        # Ensure duration scaling is considered or removed if effect is instantaneous
        # For now, assuming _form_sephirah_layer represents a single 'step' interaction
        step_changes = _form_sephirah_layer(
            soul_spark, sephirah_influencer, resonance_strength
        )
        log_layer = (f"  {sephirah_name}: Layer     - "
                     f"dE={step_changes.get('energy_transfer_seu', 0.0):+.2f}, "
                     f"dInf={step_changes.get('sephirah_influence_increment', 0.0):+.5f}")
        logger.info(log_layer) # Changed to INFO
        if step_changes.get('geometric_effects'):
             logger.info(f"  {sephirah_name}: Geometry  - Changes: {step_changes['geometric_effects']}") # Changed to INFO


        # --- 4. Update Soul's Emergent State (SU/CU Scores) ---
        logger.info(f"  {sephirah_name}: Step 4: Updating internal state scores...")
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
        else:
             logger.error("SoulSpark object missing 'update_state' method!")
             raise AttributeError("SoulSpark needs 'update_state' method.")

        # --- Capture and Log Final State for this interaction ---
        e_after = soul_spark.energy
        s_after = soul_spark.stability
        c_after = soul_spark.coherence
        inf_after = getattr(soul_spark, 'cumulative_sephiroth_influence', 0.0)
        freq_after = soul_spark.frequency
        # Calculate S/C deltas for this *interaction*
        s_delta_interaction = s_after - s_before
        c_delta_interaction = c_after - c_before
        logger.info(f"  {sephirah_name}: AFTER  - E={e_after:.1f}, S={s_after:.1f}({s_delta_interaction:+.2f}), C={c_after:.1f}({c_delta_interaction:+.2f}), CumulInf={inf_after:.4f}, Freq={freq_after:.1f}Hz")

        # --- 5. Compile Metrics ---
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state_metrics = { # Capture final state for report
            'energy_seu': e_after, 'stability_su': s_after,
            'coherence_cu': c_after, 'aspect_count': len(soul_spark.aspects),
            'layer_count': len(soul_spark.layers)
        }
        interaction_metrics = {
            'action': 'sephirah_interaction_glyph', 'soul_id': spark_id,
            'sephirah': sephirah_name, 'start_time': start_time_iso,
            'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'simulated_duration': duration, 'resonance_achieved': resonance_strength,
            'freq_resonance_component': freq_res,
            'geom_resonance_component': geom_res,
            'aspects_gained_count': gained_count,
            'aspects_strengthened_count': strengthened_count,
            'imparted_aspect_strengths_summary': f"{len(imparted_aspects)} aspects impacted", # Summary
            'layer_formation_changes': step_changes, # Includes energy, influence_increment, geom effects
            'initial_state': initial_state_metrics, 'final_state': final_state_metrics,
            'success': True,
        }
        if METRICS_AVAILABLE:
             metrics.record_metrics('sephiroth_journey_step', interaction_metrics)

        logger.info(f"--- Finished SephInteraction: {sephirah_name.capitalize()} (Res={resonance_strength:.4f}) ---")

        return soul_spark, interaction_metrics

    # --- Error Handling ---
    # ... (Keep existing error handling) ...
    except (ValueError, TypeError, AttributeError) as e_val:
        logger.error(f"Sephirah interaction failed for {spark_id} in "
                     f"{sephirah_name}: {e_val}", exc_info=True)
        record_interaction_failure(spark_id, sephirah_name, start_time_iso,
                                   'validation/attribute', str(e_val))
        raise # Re-raise specific errors
    except RuntimeError as e_rt:
        logger.critical(f"Sephirah interaction failed critically for {spark_id} "
                        f"in {sephirah_name}: {e_rt}", exc_info=True)
        record_interaction_failure(spark_id, sephirah_name, start_time_iso,
                                   'runtime', str(e_rt))
        raise # Re-raise runtime errors
    except Exception as e:
        logger.critical(f"Unexpected error during interaction for {spark_id} "
                        f"in {sephirah_name}: {e}", exc_info=True)
        record_interaction_failure(spark_id, sephirah_name, start_time_iso,
                                   'unexpected', str(e))
        raise RuntimeError(f"Unexpected Sephirah interaction failure: {e}") from e

# --- END OF REVISED Journey Function ---