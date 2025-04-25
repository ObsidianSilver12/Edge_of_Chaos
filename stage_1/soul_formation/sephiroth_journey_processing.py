# --- START OF FILE sephiroth_journey_processing.py ---

"""
Sephiroth Journey Processing Functions

Contains the logic for processing a soul's interaction within a specific
Sephiroth dimension. Calculates resonance, transfers aspects, updates
the soul's state, and performs local entanglement based on the
properties of the Sephirah. Modifies SoulSpark directly.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    # Assuming constants are in src/constants/constants.py relative to project root
    from constants.constants import *
except ImportError as e:
    # Fallback logging setup if constants unavailable during standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Sephiroth Journey Processing cannot function correctly.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_formation.soul_spark import SoulSpark
    # Aspect dictionary provides Sephiroth properties
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    DEPENDENCIES_AVAILABLE = True
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary failed to initialize.")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies (SoulSpark, aspect_dictionary): {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

# --- Metrics Tracking (Optional) ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import metrics_tracking: {e}. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    # Define placeholder if metrics not available but code uses it
    class MetricsPlaceholder:
        def record_metrics(*args, **kwargs):
            pass
    metrics = MetricsPlaceholder()


# --- Helper Functions ---

def _calculate_soul_resonance_with_sephirah(soul_spark: SoulSpark, sephirah_name: str, sephirah_aspects_data: Dict) -> float:
    """
    Calculates the soul's resonance with a specific Sephirah. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object.
        sephirah_name (str): Name of the Sephirah (lowercase).
        sephirah_aspects_data (Dict): Pre-loaded aspect data for the Sephirah.

    Returns:
        float: Resonance score (0-1).

    Raises:
        ValueError: If required data is missing or invalid.
        RuntimeError: If calculation fails unexpectedly.
        AttributeError: If required attributes are missing.
    """
    logger.debug(f"Calculating resonance between soul {soul_spark.spark_id} and {sephirah_name}...")
    if not sephirah_aspects_data: raise ValueError("Sephirah aspect data is missing.")

    try:
        # Get required soul properties (use getattr for safety, raise if essential ones missing)
        soul_frequency = getattr(soul_spark, 'soul_frequency', None)
        if soul_frequency is None or soul_frequency <= FLOAT_EPSILON:
            soul_frequency = getattr(soul_spark, 'frequency', None) # Fallback to base frequency
            if soul_frequency is None or soul_frequency <= FLOAT_EPSILON:
                raise ValueError("SoulSpark missing valid 'soul_frequency' or 'frequency'.")
        soul_phi_resonance = getattr(soul_spark, 'phi_resonance', 0.5) # Default if missing
        soul_aspects_dict = getattr(soul_spark, 'aspects', {}) # Default if missing
        if not isinstance(soul_aspects_dict, dict): raise TypeError("Soul 'aspects' attribute must be a dict.")

        # Get required Sephiroth properties from provided data
        sephirah_instance = aspect_dictionary.load_aspect_instance(sephirah_name) # Fails hard if not loadable
        sephirah_freq = getattr(sephirah_instance, 'base_frequency', None)
        if sephirah_freq is None or sephirah_freq <= FLOAT_EPSILON:
            raise ValueError(f"Sephirah base frequency for {sephirah_name} is missing or non-positive.")

        # Use get with defaults for other sephirah properties
        sephirah_primary_aspects = set(sephirah_aspects_data.get('primary_aspects', []))
        sephirah_phi_count = sephirah_aspects_data.get('phi_harmonic_count', DEFAULT_PHI_HARMONIC_COUNT)
        sephirah_resonance_mult = sephirah_aspects_data.get('resonance_multiplier', 1.0)

        # --- Calculate Resonance Components ---
        # 1. Frequency Resonance
        ratio = min(soul_frequency / sephirah_freq, sephirah_freq / soul_frequency)
        freq_resonance = max(0.0, 1.0 - abs(1.0 - ratio))**2
        logger.debug(f"  Frequency Resonance: {freq_resonance:.4f} (Soul: {soul_frequency:.2f}, Seph: {sephirah_freq:.2f})")

        # 2. Aspect Resonance (Matching primary aspects)
        match_count = 0
        if sephirah_primary_aspects: # Avoid division by zero if no primary aspects defined
             match_count = sum(1 for aspect_name in soul_aspects_dict if aspect_name in sephirah_primary_aspects)
             aspect_resonance = match_count / len(sephirah_primary_aspects)
        else:
             aspect_resonance = 0.0 # Or 1.0 if empty means perfect match? Assume 0.0
        logger.debug(f"  Aspect Resonance: {aspect_resonance:.4f} ({match_count}/{len(sephirah_primary_aspects)} primary matches)")

        # 3. Phi Resonance
        phi_factor = (soul_phi_resonance * sephirah_phi_count) / max(1, DEFAULT_PHI_HARMONIC_COUNT) # Normalize
        phi_resonance = min(1.0, max(0.0, phi_factor))
        logger.debug(f"  Phi Resonance: {phi_resonance:.4f} (SoulPhi={soul_phi_resonance:.2f}, SephPhiCount={sephirah_phi_count})")

        # --- Combine Components using Constants ---
        total_resonance = (
            freq_resonance * SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ +
            aspect_resonance * SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_ASPECT +
            phi_resonance * SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI
        )
        total_resonance *= sephirah_resonance_mult
        total_resonance *= (1.0 + (np.random.random() - 0.5) * 0.05) # +/- 2.5% variation
        total_resonance = max(0.0, min(1.0, total_resonance))

        logger.debug(f"  Total Resonance calculated: {total_resonance:.4f}")
        return float(total_resonance)

    except ValueError as ve: raise ve # Propagate validation errors
    except TypeError as te: raise te
    except AttributeError as ae: raise AttributeError(f"Missing required attribute for resonance calc: {ae}") from ae
    except Exception as e:
        logger.error(f"Unexpected error calculating soul resonance with {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError(f"Resonance calculation failed unexpectedly for {sephirah_name}.") from e


def _acquire_sephirah_aspects(soul_spark: SoulSpark, sephirah_name: str, sephirah_aspects_data: Dict, resonance_strength: float) -> Dict[str, float]:
    """
    Acquires aspects based on resonance. Modifies SoulSpark.aspects. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        sephirah_name (str): Name of the current Sephirah (lowercase).
        sephirah_aspects_data (Dict): Pre-loaded aspect data for the Sephirah.
        resonance_strength (float): Calculated resonance between soul and Sephirah (0-1).

    Returns:
        Dict[str, float]: Dictionary of {aspect_name: new_strength} for aspects touched.

    Raises:
        ValueError: If required data missing or invalid resonance.
        AttributeError: If soul_spark missing 'aspects' dictionary.
        RuntimeError: If acquisition fails unexpectedly.
    """
    logger.debug(f"Acquiring aspects from {sephirah_name} for soul {soul_spark.spark_id} (Resonance: {resonance_strength:.3f})...")
    if not sephirah_aspects_data: raise ValueError("Sephirah aspect data is missing.")
    if not (0.0 <= resonance_strength <= 1.0): raise ValueError("Invalid resonance strength.")
    if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict):
        raise AttributeError("SoulSpark object must have a valid 'aspects' dictionary attribute.")

    try:
        # Get aspect details from the loaded data
        primary_aspects = sephirah_aspects_data.get('primary_aspects', [])
        secondary_aspects = sephirah_aspects_data.get('secondary_aspects', [])
        detailed_aspects = sephirah_aspects_data.get('detailed_aspects', {}) # Dict {name: details}
        soul_current_aspects = soul_spark.aspects

        acquired = {} # Track aspects touched {name: new_strength}
        gained_count = 0
        strengthened_count = 0
        aspects_touched_this_step = set() # Track names touched in this call

        # Combine primary and secondary for processing
        all_sephirah_aspect_names = set(primary_aspects) | set(secondary_aspects)

        for aspect_name in all_sephirah_aspect_names:
            if aspect_name not in detailed_aspects:
                 logger.warning(f"Details missing for aspect '{aspect_name}' in {sephirah_name}. Skipping.")
                 continue

            # Determine gain threshold (secondary are harder to gain)
            gain_threshold = SEPHIROTH_JOURNEY_ASPECT_GAIN_THRESHOLD
            strengthen_factor = SEPHIROTH_JOURNEY_ASPECT_STRENGTHEN_FACTOR
            if aspect_name in secondary_aspects and aspect_name not in primary_aspects:
                gain_threshold += 0.1 # Make secondary harder to gain initially
                strengthen_factor *= 0.6 # Strengthen secondary less

            # Calculate acquisition potential (resonance + randomness)
            acquisition_potential = resonance_strength * (0.6 + np.random.random() * 0.4) # 0.6-1.0 * resonance

            if acquisition_potential >= gain_threshold:
                current_strength = soul_current_aspects.get(aspect_name, {}).get('strength', 0.0)
                if not isinstance(current_strength, (int, float)): current_strength = 0.0 # Handle invalid data

                strength_increase = resonance_strength * strengthen_factor
                new_strength = min(1.0, current_strength + strength_increase)
                aspect_details_copy = detailed_aspects[aspect_name].copy() # Copy details

                # Add transfer timestamp to the details being added/updated
                transfer_time = datetime.now().isoformat()
                aspect_details_copy['last_interaction_time'] = transfer_time

                if aspect_name not in soul_current_aspects: # Gained new
                    soul_current_aspects[aspect_name] = {
                        'strength': new_strength,
                        'source': sephirah_name,
                        'time_acquired': transfer_time,
                        'details': aspect_details_copy
                    }
                    acquired[aspect_name] = new_strength
                    aspects_touched_this_step.add(aspect_name)
                    gained_count += 1
                    # Log memory echo
                    echo_msg = f"Gained aspect '{aspect_name}' ({sephirah_name}) Strength: {new_strength:.3f}"
                    logger.info(f"Memory echo created: {echo_msg}")
                    if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
                        soul_spark.memory_echoes.append(f"{transfer_time}: {echo_msg}")
                elif new_strength > current_strength + FLOAT_EPSILON: # Strengthened existing (only if increased noticeably)
                    soul_current_aspects[aspect_name]['strength'] = new_strength
                    soul_current_aspects[aspect_name]['last_strengthened'] = transfer_time
                    soul_current_aspects[aspect_name]['details'].update(aspect_details_copy) # Update details
                    acquired[aspect_name] = new_strength
                    aspects_touched_this_step.add(aspect_name)
                    strengthened_count += 1
                    logger.debug(f"  Strengthened Aspect '{aspect_name}': Strength={new_strength:.4f}")


        # Update last modified time on soul if changes occurred
        if gained_count > 0 or strengthened_count > 0:
             setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        logger.info(f"Aspect acquisition complete for {sephirah_name}: Gained={gained_count}, Strengthened={strengthened_count}")
        # Return only aspects touched *in this step* for potential local entanglement boost
        return {name: acquired[name] for name in aspects_touched_this_step}


    except Exception as e:
        logger.error(f"Error acquiring aspects from {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError(f"Aspect acquisition failed for {sephirah_name}.") from e

# --- Refined _update_soul_state_from_sephirah ---
def _update_soul_state_from_sephirah(soul_spark: SoulSpark, sephirah_name: str, sephirah_aspects_data: Dict, resonance_strength: float) -> Dict[str, float]:
    """
    Updates soul's core attributes and primary divine attribute based on interaction.
    Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        sephirah_name (str): Name of the current Sephirah (lowercase).
        sephirah_aspects_data (Dict): Pre-loaded aspect data for the Sephirah.
        resonance_strength (float): Calculated resonance (0-1).

    Returns:
        Dict[str, float]: Dictionary of changes applied {'attribute_name': change_amount}.

    Raises:
        ValueError: If required data missing or invalid.
        AttributeError: If soul_spark missing attributes.
        RuntimeError: If update fails unexpectedly.
    """
    logger.debug(f"Updating soul state based on interaction with {sephirah_name} (Resonance: {resonance_strength:.3f})...")
    if not sephirah_aspects_data: raise ValueError("Sephirah aspect data is missing.")
    if not (0.0 <= resonance_strength <= 1.0): raise ValueError("Invalid resonance strength.")

    try:
        # Get current soul state safely
        current_stability = getattr(soul_spark, 'stability', SOUL_SPARK_DEFAULT_STABILITY)
        current_coherence = getattr(soul_spark, 'coherence', SOUL_SPARK_DEFAULT_COHERENCE)
        current_strength_attr = getattr(soul_spark, 'strength', (current_stability + current_coherence) / 2.0)

        # Validate current state values
        if not isinstance(current_stability, (int, float)): current_stability = SOUL_SPARK_DEFAULT_STABILITY
        if not isinstance(current_coherence, (int, float)): current_coherence = SOUL_SPARK_DEFAULT_COHERENCE
        if not isinstance(current_strength_attr, (int, float)): current_strength_attr = 0.5

        # --- Calculate Core State Changes ---
        sephirah_stability_mod = sephirah_aspects_data.get('stability_modifier', 1.0)
        sephirah_coherence_mod = sephirah_aspects_data.get('coherence_modifier', 1.0)
        sephirah_strength_mod = sephirah_aspects_data.get('strength_modifier', 0.0)

        stability_change = resonance_strength * (sephirah_stability_mod - 0.5) * SEPHIROTH_JOURNEY_STABILITY_BOOST_FACTOR
        coherence_change = resonance_strength * SEPHIROTH_JOURNEY_COHERENCE_BOOST_FACTOR * (sephirah_coherence_mod * 0.5 + 0.5)
        strength_change = resonance_strength * SEPHIROTH_JOURNEY_STRENGTH_RESONANCE_FACTOR + sephirah_strength_mod * 0.05

        new_stability = min(1.0, max(0.0, current_stability + stability_change))
        new_coherence = min(1.0, max(0.0, current_coherence + coherence_change))
        new_strength_attr = min(1.0, max(0.0, current_strength_attr + strength_change))

        # --- Impart/Strengthen Primary Divine Attribute ---
        sephirah_attribute_map = { # Mapping Sephirah name to primary attribute key
            "kether": "unity", "chokmah": "wisdom", "binah": "understanding",
            "chesed": "mercy", "geburah": "severity", "tiphareth": "beauty",
            "netzach": "victory", "hod": "splendor", "yesod": "foundation",
            "malkuth": "kingdom", "daath": "knowledge"
        }
        target_attribute = sephirah_attribute_map.get(sephirah_name)
        attribute_change = 0.0

        if target_attribute:
            # Ensure the 'attributes' dictionary exists on the soul
            if not hasattr(soul_spark, 'attributes') or not isinstance(soul_spark.attributes, dict):
                setattr(soul_spark, 'attributes', {})

            sephirah_energy = sephirah_aspects_data.get('energy_level', 0.5) # Use preloaded energy level
            attribute_increase = resonance_strength * sephirah_energy * SEPHIROTH_JOURNEY_ATTRIBUTE_IMPART_FACTOR
            attribute_increase = min(0.1, max(0.0, attribute_increase)) # Cap the increase per interaction

            current_value = soul_spark.attributes.get(target_attribute, 0.0)
            if not isinstance(current_value, (int, float)): current_value = 0.0

            new_value = min(1.0, current_value + attribute_increase) # Assume attributes scale 0-1
            soul_spark.attributes[target_attribute] = float(new_value)
            attribute_change = new_value - current_value # Store actual change
            logger.debug(f"  Updated Soul Attribute '{target_attribute}' to {new_value:.4f} (+{attribute_increase:.4f})")
        else:
             logger.warning(f"No primary attribute defined for Sephirah '{sephirah_name}' in map.")

        # --- Apply Elemental Influence ---
        sephirah_element = sephirah_aspects_data.get('element', ELEMENT_NEUTRAL).lower()
        elemental_change = 0.0
        if not hasattr(soul_spark, 'elements') or not isinstance(soul_spark.elements, dict):
            setattr(soul_spark, 'elements', {})

        current_elem_strength = soul_spark.elements.get(sephirah_element, 0.0)
        if not isinstance(current_elem_strength, (int, float)): current_elem_strength = 0.0

        element_increase = resonance_strength * SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR
        new_elem_strength = min(1.0, current_elem_strength + element_increase)
        soul_spark.elements[sephirah_element] = float(new_elem_strength)
        elemental_change = new_elem_strength - current_elem_strength
        logger.debug(f"  Updated element '{sephirah_element}' strength to {new_elem_strength:.3f}")

        # --- Update SoulSpark Object ---
        setattr(soul_spark, 'stability', float(new_stability))
        setattr(soul_spark, 'coherence', float(new_coherence))
        setattr(soul_spark, 'strength', float(new_strength_attr)) # Update the 'strength' attribute

        # Record actual changes applied
        changes = {
            "stability_change": float(new_stability - current_stability),
            "coherence_change": float(new_coherence - current_coherence),
            "strength_change": float(new_strength_attr - current_strength_attr),
            f"{target_attribute}_change" if target_attribute else "attribute_change": attribute_change, # Use specific name if possible
            f"{sephirah_element}_element_change": elemental_change
        }
        logger.info(f"Soul state updated after {sephirah_name}: Stab={new_stability:.4f}, Coh={new_coherence:.4f}, Str={new_strength_attr:.4f}")

        return changes

    except AttributeError as ae: logger.error(f"SoulSpark missing expected attribute during state update: {ae}"); raise
    except ValueError as ve: logger.error(f"ValueError during state update for {sephirah_name}: {ve}"); raise
    except Exception as e: logger.error(f"Unexpected error updating soul state from {sephirah_name}: {e}", exc_info=True); raise RuntimeError(f"Soul state update failed for {sephirah_name}.") from e


# --- NEW Local Entanglement Function ---
def _perform_local_sephiroth_entanglement(soul_spark: SoulSpark, sephirah_name: str, sephirah_aspects_data: Dict, resonance_strength: float, acquired_aspects: Dict[str, float]) -> Dict[str, float]:
    """
    Performs local attunement/entanglement with the current Sephirah after initial interaction.
    Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        sephirah_name (str): Name of the current Sephirah (lowercase).
        sephirah_aspects_data (Dict): Pre-loaded aspect data for the Sephirah.
        resonance_strength (float): Calculated resonance (0-1).
        acquired_aspects (Dict[str, float]): Aspects acquired/strengthened in the previous step
                                            {aspect_name: new_strength}.

    Returns:
        Dict[str, float]: Dictionary of changes applied during local entanglement.

    Raises:
        ValueError: If resonance strength invalid.
        AttributeError: If soul_spark missing attributes.
        RuntimeError: If entanglement fails unexpectedly.
    """
    logger.debug(f"Performing local entanglement: Soul {soul_spark.spark_id} with {sephirah_name} (Resonance: {resonance_strength:.3f})...")
    if not (0.0 <= resonance_strength <= 1.0): raise ValueError("Invalid resonance strength.")

    try:
        changes = {}
        start_time = datetime.now().isoformat() # Use consistent timestamp for all changes in this step

        # --- 1. Frequency Attunement ---
        freq_nudge = 0.0
        current_freq = getattr(soul_spark, 'frequency', None)
        sephirah_freq = sephirah_aspects_data.get('base_frequency')
        if current_freq and current_freq > 0 and sephirah_freq and sephirah_freq > 0:
            freq_diff = sephirah_freq - current_freq
            freq_nudge = freq_diff * resonance_strength * SEPHIROTH_LOCAL_ENTANGLE_FREQ_PULL
            new_freq = max(FLOAT_EPSILON, current_freq + freq_nudge)
            setattr(soul_spark, 'frequency', float(new_freq))
            logger.debug(f"  Frequency locally nudged to {new_freq:.2f} Hz (+{freq_nudge:.2f})")
        else:
            logger.warning(f"Could not perform frequency attunement for {sephirah_name} (Invalid frequencies).")
        changes['local_frequency_nudge'] = freq_nudge


        # --- 2. Stability & Coherence Boost ---
        current_stability = getattr(soul_spark, 'stability', SOUL_SPARK_DEFAULT_STABILITY)
        current_coherence = getattr(soul_spark, 'coherence', SOUL_SPARK_DEFAULT_COHERENCE)

        stability_gain = resonance_strength * SEPHIROTH_LOCAL_ENTANGLE_STABILITY_GAIN
        coherence_gain = resonance_strength * SEPHIROTH_LOCAL_ENTANGLE_COHERENCE_GAIN

        new_stability = min(1.0, current_stability + stability_gain)
        new_coherence = min(1.0, current_coherence + coherence_gain)

        setattr(soul_spark, 'stability', float(new_stability))
        setattr(soul_spark, 'coherence', float(new_coherence))
        changes['local_stability_gain'] = new_stability - current_stability
        changes['local_coherence_gain'] = new_coherence - current_coherence
        logger.debug(f"  Local Stability boosted to {new_stability:.4f} (+{changes['local_stability_gain']:.4f})")
        logger.debug(f"  Local Coherence boosted to {new_coherence:.4f} (+{changes['local_coherence_gain']:.4f})")

        # --- 3. Boost Recently Acquired/Strengthened Aspects ---
        total_aspect_boost = 0.0
        aspect_boost_count = 0
        if hasattr(soul_spark, 'aspects') and isinstance(soul_spark.aspects, dict) and acquired_aspects:
            for aspect_name in acquired_aspects.keys(): # Iterate over names touched in previous step
                if aspect_name in soul_spark.aspects:
                    current_strength = soul_spark.aspects[aspect_name].get('strength', 0.0)
                    if not isinstance(current_strength, (int, float)): continue # Skip if invalid

                    # Boost based on resonance
                    boost = resonance_strength * SEPHIROTH_LOCAL_ENTANGLE_ASPECT_BOOST
                    new_strength = min(1.0, current_strength + boost)
                    if new_strength > current_strength + FLOAT_EPSILON: # Only update if changed
                         soul_spark.aspects[aspect_name]['strength'] = float(new_strength)
                         soul_spark.aspects[aspect_name]['last_attuned'] = start_time # Mark attunement time
                         total_aspect_boost += (new_strength - current_strength)
                         aspect_boost_count += 1
                         logger.debug(f"    Locally boosted aspect '{aspect_name}' to {new_strength:.4f}")

        changes['local_aspect_boost_total'] = total_aspect_boost
        changes['local_aspect_boost_count'] = aspect_boost_count
        if aspect_boost_count > 0:
             logger.debug(f"  Boosted {aspect_boost_count} recently acquired/strengthened aspects.")

        # --- Update Timestamp ---
        setattr(soul_spark, 'last_modified', start_time)

        logger.info(f"Local entanglement with {sephirah_name} complete.")
        return changes

    except AttributeError as ae: logger.error(f"SoulSpark missing attribute during local entanglement: {ae}"); raise
    except ValueError as ve: logger.error(f"ValueError during local entanglement for {sephirah_name}: {ve}"); raise
    except Exception as e: logger.error(f"Unexpected error during local entanglement for {sephirah_name}: {e}", exc_info=True); raise RuntimeError(f"Local entanglement failed for {sephirah_name}.") from e


# --- Main Orchestration Function ---
def process_sephirah_interaction(soul_spark: SoulSpark, sephirah_name: str, duration: float) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Processes the soul's interaction within a single Sephiroth field, including resonance,
    aspect acquisition, state update, and local entanglement.
    Modifies the SoulSpark object in place. Fails hard on critical errors.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        sephirah_name (str): The name of the Sephirah being processed.
        duration (float): The duration (simulation units) spent in this Sephirah.

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object.
            - interaction_metrics (Dict): Summary metrics for this interaction step.

    Raises:
        ValueError: If sephirah_name or duration invalid.
        RuntimeError: If aspect loading or any processing step fails critically.
    """
    sephirah_lower = sephirah_name.lower()
    if not isinstance(duration, (int, float)) or duration <= FLOAT_EPSILON:
        raise ValueError("Duration must be positive.")
    if not ASPECT_DICT_AVAILABLE or aspect_dictionary is None:
         raise RuntimeError("Aspect Dictionary unavailable.")
    if sephirah_lower not in aspect_dictionary.sephiroth_names:
         raise ValueError(f"Invalid Sephiroth name: {sephirah_name}")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Processing Interaction: Soul {spark_id} in {sephirah_lower.capitalize()} (Duration: {duration}) ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)

    try:
        # --- 1. Get Sephiroth Data ---
        sephirah_aspects_data = aspect_dictionary.get_aspects(sephirah_lower)
        if not sephirah_aspects_data: raise RuntimeError(f"Failed to load aspect data for {sephirah_lower}.")

        # --- 2. Calculate Resonance ---
        logger.info("Step 2.1: Calculating soul-sephirah resonance...")
        resonance = _calculate_soul_resonance_with_sephirah(soul_spark, sephirah_lower, sephirah_aspects_data)

        # --- 3. Acquire Aspects ---
        logger.info("Step 2.2: Acquiring aspects...")
        acquired_aspects_dict = _acquire_sephirah_aspects(soul_spark, sephirah_lower, sephirah_aspects_data, resonance)

        # --- 4. Update Soul State & Attributes ---
        logger.info("Step 2.3: Updating soul core state...")
        state_changes = _update_soul_state_from_sephirah(soul_spark, sephirah_lower, sephirah_aspects_data, resonance)

        # --- 5. Perform Local Entanglement/Attunement ---
        logger.info("Step 2.4: Performing local Sephiroth entanglement...")
        entanglement_changes = _perform_local_sephiroth_entanglement(soul_spark, sephirah_lower, sephirah_aspects_data, resonance, acquired_aspects_dict)

        # --- 6. Compile Metrics for this step ---
        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)

        # Calculate counts based on acquired_aspects_dict (aspects touched in step 3)
        gained_count = sum(1 for asp_name in acquired_aspects_dict if getattr(soul_spark, 'aspects', {}).get(asp_name, {}).get('time_acquired') == getattr(soul_spark, 'aspects', {}).get(asp_name, {}).get('details', {}).get('transfer_timestamp'))
        strengthened_count = len(acquired_aspects_dict) - gained_count

        interaction_metrics = {
            'action': 'sephirah_interaction', 'soul_id': spark_id, 'sephirah': sephirah_lower,
            'start_time': start_time_iso, 'end_time': end_time_iso, 'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'resonance_achieved': resonance,
            'aspects_gained_count': gained_count,
            'aspects_strengthened_count': strengthened_count,
            'aspects_touched_names': list(acquired_aspects_dict.keys()), # Names of aspects touched in step 3
            'initial_state_changes': state_changes, # Changes from step 4
            'local_entanglement_changes': entanglement_changes, # Changes from step 5
            'final_stability': getattr(soul_spark, 'stability', 0.0),
            'final_coherence': getattr(soul_spark, 'coherence', 0.0),
            'final_frequency': getattr(soul_spark, 'frequency', 0.0),
            'success': True,
        }
        # Record detailed metrics if available
        if METRICS_AVAILABLE:
            try: metrics.record_metrics('sephiroth_journey_step', interaction_metrics)
            except Exception as e: logger.error(f"Failed to record interaction metrics: {e}")

        logger.info(f"Interaction with {sephirah_lower.capitalize()} complete. Resonance: {resonance:.4f}")
        logger.info(f"  Aspects Touched: {len(acquired_aspects_dict)}, Final Stability: {soul_spark.stability:.4f}, Final Coherence: {soul_spark.coherence:.4f}")

        return soul_spark, interaction_metrics

    except Exception as e:
        # Catch any exception from the steps above
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Sephirah interaction failed critically for soul {spark_id} in {sephirah_lower}: {e}", exc_info=True)
        # Record failure metric if possible
        if METRICS_AVAILABLE:
             try: metrics.record_metrics('sephiroth_journey_step', {
                  'action': 'sephirah_interaction', 'soul_id': spark_id, 'sephirah': sephirah_lower,
                  'start_time': start_time_iso, 'end_time': end_time_iso,
                  'success': False, 'error': str(e)
             })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        # Re-raise as RuntimeError
        raise RuntimeError(f"Sephirah interaction process failed critically for {sephirah_lower}.") from e

# (Keep Example Usage __main__ block if desired for testing this file)

# --- END OF (MODIFIED) sephiroth_journey_processing.py ---