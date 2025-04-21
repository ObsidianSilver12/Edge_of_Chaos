# --- START OF FILE sephiroth_journey_processing.py ---

"""
Sephiroth Journey Processing Functions

Contains the logic for processing a soul's interaction within a specific
Sephiroth dimension. Calculates resonance, transfers aspects, and updates
the soul's state based on the properties of the Sephirah.

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
    from src.constants import * # Import all for convenience
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Sephiroth Journey Processing cannot function correctly.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.void.soul_spark import SoulSpark
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    # No direct need for field classes here, aspect_dictionary provides data
    DEPENDENCIES_AVAILABLE = True
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary failed to initialize.")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies (SoulSpark, aspect_dictionary): {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import metrics_tracking: {e}. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder: def record_metrics(*args, **kwargs): pass
    metrics = MetricsPlaceholder()


# --- Helper Functions ---

def _calculate_soul_resonance_with_sephirah(soul_spark: SoulSpark, sephirah_name: str, sephirah_aspects_data: Dict) -> float:
    """
    Calculates the soul's resonance with a specific Sephirah. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object.
        sephirah_name (str): Name of the Sephirah.
        sephirah_aspects_data (Dict): Pre-loaded aspect data for the Sephirah.

    Returns:
        float: Resonance score (0-1).

    Raises:
        ValueError: If required data is missing or invalid.
    """
    logger.debug(f"Calculating resonance between soul {soul_spark.spark_id} and {sephirah_name}...")
    if not sephirah_aspects_data: raise ValueError("Sephirah aspect data is missing.")

    try:
        # Get required soul properties
        soul_frequency = getattr(soul_spark, 'soul_frequency', SOUL_SPARK_DEFAULT_FREQ) # Use determined soul frequency
        soul_phi_resonance = getattr(soul_spark, 'phi_resonance', 0.5)
        soul_aspects_dict = getattr(soul_spark, 'aspects', {})
        if soul_frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency is non-positive.")

        # Get required Sephiroth properties from provided data
        # Use base_frequency from the aspect instance via aspect_dictionary
        sephirah_freq = aspect_dictionary.load_aspect_instance(sephirah_name).base_frequency
        if sephirah_freq <= FLOAT_EPSILON: raise ValueError(f"Sephirah base frequency for {sephirah_name} is non-positive.")
        sephirah_primary_aspects = set(sephirah_aspects_data.get('primary_aspects', []))
        sephirah_phi_count = sephirah_aspects_data.get('phi_harmonic_count', DEFAULT_PHI_HARMONIC_COUNT)
        sephirah_resonance_mult = sephirah_aspects_data.get('resonance_multiplier', 1.0) # From aspect data

        # --- Calculate Resonance Components ---
        # 1. Frequency Resonance
        freq_resonance = 0.0
        if sephirah_freq > FLOAT_EPSILON:
            ratio = min(soul_frequency / sephirah_freq, sephirah_freq / soul_frequency)
            # Scale more aggressively: 1 = perfect match, 0 = large difference
            freq_resonance = max(0.0, 1.0 - abs(1.0 - ratio))**2 # Squared distance from 1
        logger.debug(f"  Frequency Resonance: {freq_resonance:.4f} (Soul: {soul_frequency:.2f}, Seph: {sephirah_freq:.2f})")

        # 2. Aspect Resonance (Matching primary aspects)
        aspect_resonance = 0.0
        if sephirah_primary_aspects:
            match_count = sum(1 for aspect_name in soul_aspects_dict if aspect_name in sephirah_primary_aspects)
            aspect_resonance = match_count / len(sephirah_primary_aspects)
        logger.debug(f"  Aspect Resonance: {aspect_resonance:.4f} ({match_count}/{len(sephirah_primary_aspects)} primary matches)")

        # 3. Phi Resonance
        # Interaction between soul's phi resonance and Sephiroth's phi harmonic count potential
        phi_factor = (soul_phi_resonance * sephirah_phi_count) / (DEFAULT_PHI_HARMONIC_COUNT) # Normalize against default max count
        phi_resonance = min(1.0, max(0.0, phi_factor))
        logger.debug(f"  Phi Resonance: {phi_resonance:.4f} (SoulPhi={soul_phi_resonance:.2f}, SephPhiCount={sephirah_phi_count})")

        # --- Combine Components using Constants ---
        total_resonance = (
            freq_resonance * SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ +
            aspect_resonance * SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_ASPECT +
            phi_resonance * SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI
        )
        # Apply Sephiroth's own resonance multiplier
        total_resonance *= sephirah_resonance_mult
        # Apply slight random variation
        total_resonance *= (1.0 + (np.random.random() - 0.5) * 0.05) # +/- 2.5% variation
        total_resonance = max(0.0, min(1.0, total_resonance)) # Clamp final result

        logger.debug(f"  Total Resonance calculated: {total_resonance:.4f}")
        return float(total_resonance)

    except Exception as e:
        logger.error(f"Error calculating soul resonance with {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError(f"Resonance calculation failed for {sephirah_name}.") from e


def _acquire_sephirah_aspects(soul_spark: SoulSpark, sephirah_name: str, sephirah_aspects_data: Dict, resonance_strength: float) -> Dict[str, float]:
    """
    Acquires aspects based on resonance. Modifies SoulSpark.aspects. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        sephirah_name (str): Name of the current Sephirah.
        sephirah_aspects_data (Dict): Pre-loaded aspect data for the Sephirah.
        resonance_strength (float): Calculated resonance between soul and Sephirah (0-1).

    Returns:
        Dict[str, float]: Dictionary of {aspect_name: acquired_strength}.

    Raises:
        ValueError: If required data missing or invalid.
        AttributeError: If soul_spark missing 'aspects' dictionary.
    """
    logger.debug(f"Acquiring aspects from {sephirah_name} for soul {soul_spark.spark_id} (Resonance: {resonance_strength:.3f})...")
    if not sephirah_aspects_data: raise ValueError("Sephirah aspect data is missing.")
    if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict):
        raise AttributeError("SoulSpark object must have a valid 'aspects' dictionary attribute.")

    try:
        # Get detailed aspects from the loaded data
        primary_aspects = sephirah_aspects_data.get('primary_aspects', [])
        secondary_aspects = sephirah_aspects_data.get('secondary_aspects', [])
        detailed_aspects = sephirah_aspects_data.get('detailed_aspects', {})
        soul_current_aspects = soul_spark.aspects # Get reference to soul's aspect dict

        acquired = {}
        gained_count = 0
        strengthened_count = 0

        # --- Acquire Primary Aspects ---
        for aspect_name in primary_aspects:
            if aspect_name not in detailed_aspects: continue # Skip if details missing
            # Acquisition strength depends on resonance
            acquisition_strength = resonance_strength * (0.6 + np.random.random() * 0.4) # 0.6-1.0 of resonance
            if acquisition_strength >= SEPHIROTH_JOURNEY_ASPECT_GAIN_THRESHOLD: # Use constant threshold
                current_strength = soul_current_aspects.get(aspect_name, {}).get('strength', 0.0)
                # Strengthening uses a different factor
                strength_increase = resonance_strength * SEPHIROTH_JOURNEY_ASPECT_STRENGTHEN_FACTOR
                new_strength = min(1.0, current_strength + strength_increase)

                if aspect_name not in soul_current_aspects: # Gained new
                    soul_current_aspects[aspect_name] = {
                        'strength': new_strength, 'source': sephirah_name,
                        'time_acquired': datetime.now().isoformat(),
                        'details': detailed_aspects[aspect_name] # Store details from sephirah
                    }
                    acquired[aspect_name] = new_strength
                    gained_count += 1
                    # Log memory echo
                    logger.info(f"Memory echo created: Gained aspect '{aspect_name}' from {sephirah_name}.")
                    if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
                        soul_spark.memory_echoes.append(f"Gained '{aspect_name}' ({sephirah_name}) @ {soul_current_aspects[aspect_name]['time_acquired']}")
                else: # Strengthened existing
                    soul_current_aspects[aspect_name]['strength'] = new_strength
                    soul_current_aspects[aspect_name]['last_strengthened'] = datetime.now().isoformat()
                    acquired[aspect_name] = new_strength # Still record it was touched
                    strengthened_count += 1
                logger.debug(f"  Acquired/Strengthened Primary Aspect '{aspect_name}': Strength={new_strength:.4f}")


        # --- Acquire Secondary Aspects (Lower probability/strength) ---
        secondary_gain_threshold = SEPHIROTH_JOURNEY_ASPECT_GAIN_THRESHOLD + 0.1 # Harder to gain secondary
        for aspect_name in secondary_aspects:
            if aspect_name not in detailed_aspects: continue
            acquisition_strength = resonance_strength * (0.4 + np.random.random() * 0.3) # 0.4-0.7 of resonance
            if acquisition_strength >= secondary_gain_threshold:
                current_strength = soul_current_aspects.get(aspect_name, {}).get('strength', 0.0)
                strength_increase = resonance_strength * SEPHIROTH_JOURNEY_ASPECT_STRENGTHEN_FACTOR * 0.5 # Less strengthening
                new_strength = min(1.0, current_strength + strength_increase)

                if aspect_name not in soul_current_aspects:
                    soul_current_aspects[aspect_name] = {
                        'strength': new_strength, 'source': sephirah_name,
                        'time_acquired': datetime.now().isoformat(),
                        'details': detailed_aspects[aspect_name]
                    }
                    acquired[aspect_name] = new_strength
                    gained_count += 1
                    # Log memory echo
                    logger.info(f"Memory echo created: Gained aspect '{aspect_name}' from {sephirah_name}.")
                    if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
                        soul_spark.memory_echoes.append(f"Gained '{aspect_name}' ({sephirah_name}) @ {soul_current_aspects[aspect_name]['time_acquired']}")
                else:
                    soul_current_aspects[aspect_name]['strength'] = new_strength
                    soul_current_aspects[aspect_name]['last_strengthened'] = datetime.now().isoformat()
                    acquired[aspect_name] = new_strength
                    strengthened_count += 1
                logger.debug(f"  Acquired/Strengthened Secondary Aspect '{aspect_name}': Strength={new_strength:.4f}")

        # Update last modified time on soul
        if gained_count > 0 or strengthened_count > 0:
             setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        logger.info(f"Aspect acquisition complete for {sephirah_name}: Gained={gained_count}, Strengthened={strengthened_count}")
        return acquired # Return dict of aspects touched in this step

    except Exception as e:
        logger.error(f"Error acquiring aspects from {sephirah_name}: {e}", exc_info=True)
        raise RuntimeError(f"Aspect acquisition failed for {sephirah_name}.") from e


def _update_soul_state_from_sephirah(soul_spark: SoulSpark, sephirah_name: str, sephirah_aspects_data: Dict, resonance_strength: float) -> Dict[str, float]:
    """
    Updates soul's core attributes based on interaction. Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        sephirah_name (str): Name of the current Sephirah.
        sephirah_aspects_data (Dict): Pre-loaded aspect data for the Sephirah.
        resonance_strength (float): Calculated resonance (0-1).

    Returns:
        Dict[str, float]: Dictionary of changes applied to core attributes.

    Raises:
        ValueError: If required data missing or invalid.
        AttributeError: If soul_spark missing attributes.
    """
    logger.debug(f"Updating soul state based on interaction with {sephirah_name} (Resonance: {resonance_strength:.3f})...")
    if not sephirah_aspects_data: raise ValueError("Sephirah aspect data is missing.")

    try:
        # Get current soul state
        current_stability = getattr(soul_spark, 'stability')
        current_coherence = getattr(soul_spark, 'coherence')
        current_strength = getattr(soul_spark, 'strength', 0.5) # Use strength if exists, default 0.5

        # Get Sephiroth modifiers
        sephirah_stability_mod = sephirah_aspects_data.get('stability_modifier', 1.0) # Use 1.0 if missing

        # Calculate changes based on resonance and Sephiroth properties
        stability_change = resonance_strength * (sephirah_stability_mod - 0.5) * SEPHIROTH_JOURNEY_STABILITY_BOOST_FACTOR # Use constant factor
        # Coherence boost based purely on resonance (interaction itself harmonizes)
        coherence_change = resonance_strength * SEPHIROTH_JOURNEY_STABILITY_BOOST_FACTOR * 0.8 # Coherence boost slightly less
        # Strength increases with resonance and aspects gained (use resonance here)
        strength_change = resonance_strength * SEPHIROTH_JOURNEY_STRENGTH_RESONANCE_FACTOR

        # Apply changes
        new_stability = min(1.0, max(0.0, current_stability + stability_change))
        new_coherence = min(1.0, max(0.0, current_coherence + coherence_change))
        new_strength = min(1.0, max(0.0, current_strength + strength_change))

        # --- Update SoulSpark ---
        setattr(soul_spark, 'stability', float(new_stability))
        setattr(soul_spark, 'coherence', float(new_coherence))
        setattr(soul_spark, 'strength', float(new_strength)) # Add/update strength attribute

        # Impart divine quality (if defined)
        divine_quality = sephirah_aspects_data.get('divine_quality')
        if isinstance(divine_quality, dict):
            quality_name = divine_quality.get('name', sephirah_name)
            quality_strength = divine_quality.get('strength', 0.5) * resonance_strength * SEPHIROTH_JOURNEY_DIVINE_QUALITY_IMPART_FACTOR
            if not hasattr(soul_spark, 'divine_qualities'): setattr(soul_spark, 'divine_qualities', {})
            # Strengthen existing or add new
            current_qual_strength = soul_spark.divine_qualities.get(quality_name, {}).get('strength', 0.0)
            soul_spark.divine_qualities[quality_name] = {'strength': min(1.0, max(current_qual_strength, quality_strength)), # Take max? Or add? Let's take max.
                                                         'source': sephirah_name, 'time': datetime.now().isoformat()}
            logger.debug(f"  Imparted/Updated Divine Quality '{quality_name}' with strength {soul_spark.divine_qualities[quality_name]['strength']:.3f}")


        # Apply elemental influence
        sephirah_element = sephirah_aspects_data.get('element', ELEMENT_NEUTRAL).lower()
        if hasattr(soul_spark, 'elements') and isinstance(soul_spark.elements, dict):
            current_elem_strength = soul_spark.elements.get(sephirah_element, 0.0)
            soul_spark.elements[sephirah_element] = min(1.0, current_elem_strength + resonance_strength * SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR)
            logger.debug(f"  Updated element '{sephirah_element}' strength to {soul_spark.elements[sephirah_element]:.3f}")

        # Record changes
        changes = {
            "stability_change": float(new_stability - current_stability),
            "coherence_change": float(new_coherence - current_coherence),
            "strength_change": float(new_strength - current_strength)
        }
        logger.info(f"Soul state updated: Stab={new_stability:.4f}, Coh={new_coherence:.4f}, Str={new_strength:.4f}")

        return changes

    except AttributeError as ae: logger.error(f"SoulSpark missing attribute during state update: {ae}"); raise
    except Exception as e: logger.error(f"Error updating soul state from {sephirah_name}: {e}", exc_info=True); raise RuntimeError(f"Soul state update failed for {sephirah_name}.") from e


# --- Orchestration Function ---

def process_sephirah_interaction(soul_spark: SoulSpark, sephirah_name: str, duration: float) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Processes the soul's interaction within a single Sephiroth field.
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
        RuntimeError: If aspect loading, resonance calc, aspect acquisition, or state update fails.
    """
    sephirah_lower = sephirah_name.lower()
    if not isinstance(duration, (int, float)) or duration <= FLOAT_EPSILON:
        raise ValueError("Duration must be positive.")
    if not DEPENDENCIES_AVAILABLE or aspect_dictionary is None:
         raise RuntimeError("Aspect Dictionary unavailable.")
    if sephirah_lower not in aspect_dictionary.sephiroth_names:
         raise ValueError(f"Invalid Sephiroth name: {sephirah_name}")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Processing Interaction: Soul {spark_id} in {sephirah_lower.capitalize()} (Duration: {duration}) ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)

    try:
        # --- Get Sephiroth Data ---
        sephirah_aspects_data = aspect_dictionary.get_aspects(sephirah_lower) # Fails hard if missing
        if not sephirah_aspects_data: raise RuntimeError(f"Failed to load aspect data for {sephirah_lower}.")

        # --- Calculate Resonance ---
        logger.info("Calculating soul-sephirah resonance...")
        resonance = _calculate_soul_resonance_with_sephirah(soul_spark, sephirah_lower, sephirah_aspects_data)

        # --- Acquire Aspects ---
        logger.info("Acquiring aspects...")
        acquired_aspects_dict = _acquire_sephirah_aspects(soul_spark, sephirah_lower, sephirah_aspects_data, resonance)

        # --- Update Soul State ---
        logger.info("Updating soul core state...")
        state_changes = _update_soul_state_from_sephirah(soul_spark, sephirah_lower, sephirah_aspects_data, resonance)

        # --- Compile Metrics for this step ---
        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)
        interaction_metrics = {
            'action': 'sephirah_interaction', 'soul_id': spark_id, 'sephirah': sephirah_lower,
            'start_time': start_time_iso, 'end_time': end_time_iso, 'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'resonance_achieved': resonance,
            'aspects_gained_count': sum(1 for asp, data in acquired_aspects_dict.items() if getattr(soul_spark,'aspects',{}).get(asp,{}).get('time_acquired') == data.get('time_acquired')), # Rough count of NEW aspects
            'aspects_strengthened_count': sum(1 for asp in acquired_aspects_dict if getattr(soul_spark,'aspects',{}).get(asp,{}).get('time_acquired') != data.get('time_acquired')), # Rough count of STRENGTHENED
            'state_changes': state_changes, # Store dict of core changes
            'final_stability': getattr(soul_spark, 'stability', 0.0),
            'final_coherence': getattr(soul_spark, 'coherence', 0.0),
            'success': True,
        }
        try: metrics.record_metrics('sephiroth_journey_step', interaction_metrics)
        except Exception as e: logger.error(f"Failed to record interaction metrics: {e}")

        logger.info(f"Interaction with {sephirah_lower.capitalize()} complete. Resonance: {resonance:.4f}")
        logger.info(f"  Aspects Touched: {len(acquired_aspects_dict)}, Final Stability: {soul_spark.stability:.4f}, Final Coherence: {soul_spark.coherence:.4f}")

        return soul_spark, interaction_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Sephirah interaction failed critically for soul {spark_id} in {sephirah_lower}: {e}", exc_info=True)
        if METRICS_AVAILABLE:
             try: metrics.record_metrics('sephiroth_journey_step', {
                  'action': 'sephirah_interaction', 'soul_id': spark_id, 'sephirah': sephirah_lower,
                  'start_time': start_time_iso, 'end_time': end_time_iso,
                  'success': False, 'error': str(e)
             })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Sephirah interaction process failed for {sephirah_lower}.") from e

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Sephiroth Journey Processing Module Example...")
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        # Create a SoulSpark instance (more developed state)
        test_soul = SoulSpark()
        test_soul.spark_id="test_journey_001"
        test_soul.stability = 0.75
        test_soul.coherence = 0.78
        test_soul.resonance = 0.80 # Has resonance from previous stages
        test_soul.frequency = 432.0
        test_soul.formation_complete = True
        test_soul.harmonically_strengthened = True # Assume previous stage done
        test_soul.creator_alignment = 0.85
        test_soul.phi_resonance = 0.75
        test_soul.aspects = {'unity': {'strength': 0.6, 'source':'Kether'}, 'love': {'strength': 0.5, 'source':'Guff'}} # Some initial aspects

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Coherence: {test_soul.coherence:.4f}")
        print(f"  Resonance: {test_soul.resonance:.4f}")
        print(f"  Aspects: {list(test_soul.aspects.keys())}")

        sephirah_to_process = "malkuth" # Example: start journey in Malkuth

        try:
            print(f"\n--- Processing Interaction with {sephirah_to_process.capitalize()} ---")
            modified_soul, interaction_metrics_result = process_sephirah_interaction(
                soul_spark=test_soul, # Pass the object
                sephirah_name=sephirah_to_process,
                duration=10.0 # Example duration
            )

            print("\n--- Interaction Complete ---")
            print("Resulting Soul State Summary:")
            print(f"  ID: {modified_soul.spark_id}")
            print(f"  Final Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            print(f"  Final Coherence: {getattr(modified_soul, 'coherence', 'N/A'):.4f}")
            print(f"  Aspects ({len(getattr(modified_soul, 'aspects', {}))} total): {list(getattr(modified_soul, 'aspects', {}).keys())}")
            print(f"  Elemental Influence: {getattr(modified_soul, 'elements', {})}")
            print(f"  Divine Qualities: {list(getattr(modified_soul, 'divine_qualities', {}).keys())}")


            print("\nInteraction Metrics:")
            # print(json.dumps(interaction_metrics_result, indent=2, default=str)) # Optional full metrics
            print(f"  Sephirah: {interaction_metrics_result.get('sephirah')}")
            print(f"  Resonance Achieved: {interaction_metrics_result.get('resonance_achieved', 'N/A'):.4f}")
            print(f"  Aspects Gained: {interaction_metrics_result.get('aspects_gained_count', 'N/A')}")
            print(f"  Aspects Strengthened: {interaction_metrics_result.get('aspects_strengthened_count', 'N/A')}")
            print(f"  State Changes: {interaction_metrics_result.get('state_changes')}")


        except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as e:
            print(f"\n--- ERROR during Sephiroth Interaction Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Sephiroth Interaction Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\nSephiroth Journey Processing Module Example Finished.")

# --- END OF FILE sephiroth_journey_processing.py ---