# --- START OF FILE src/stage_1/soul_formation/earth_harmonisation.py ---

"""
Earth Harmonization Functions (Refactored V4.3.8 - Principle-Driven)

Simulates an 'Echo' projection attuning to Earth via the Life Cord.
Attunement is resonance-driven, impacting the main soul's earth_resonance factor.
May induce minor stress feedback affecting main soul S/C emergently.
Modifies SoulSpark directly. Uses constants (defined below if missing). Hard fails.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional
import random # Added for planet selection
from constants.constants import *
# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
# Attempt to import from central location
try:
    from constants.constants import *
except ImportError:
    logging.error("Constants module not found. Using local fallbacks for Earth Harmonization.")
    # Define minimal fallbacks needed for THIS file
    # FLAG_READY_FOR_EARTH = "ready_for_earth"; HARMONY_PREREQ_CORD_INTEGRITY_MIN = 0.7
    # HARMONY_PREREQ_STABILITY_MIN_SU = 75.0; HARMONY_PREREQ_COHERENCE_MIN_CU = 75.0
    # FLAG_EARTH_ATTUNED = "earth_attuned"; FLAG_ECHO_PROJECTED = "echo_projected" # New flags
    # FLOAT_EPSILON = 1e-9; MAX_STABILITY_SU=100.0; MAX_COHERENCE_CU=100.0
    # EARTH_HARMONY_INTENSITY_DEFAULT = 0.7; EARTH_HARMONY_DURATION_FACTOR_DEFAULT = 1.0
    # ECHO_PROJECTION_ENERGY_COST_FACTOR = 0.02 # % of spiritual energy cost
    # ECHO_ATTUNEMENT_CYCLES = 8             # Number of attunement cycles
    # ECHO_ATTUNEMENT_RATE = 0.05            # Base rate of state shift per cycle
    # ATTUNEMENT_RESONANCE_THRESHOLD = 0.3   # Min resonance for attunement shift
    # EARTH_FREQUENCIES = {"schumann": 7.83, "core_resonance": 136.1} # Example Earth frequencies
    # SCHUMANN_FREQUENCY = 7.83
    # EARTH_FREQUENCY = 136.1 # Core resonance example
    # EARTH_ELEMENTS = ["earth", "water", "fire", "air", "aether"]
    # ELEMENTAL_TARGET_EARTH = 0.8; ELEMENTAL_TARGET_OTHER = 0.4 # Targets for element alignment
    # ELEMENTAL_ALIGN_INTENSITY_FACTOR = 0.2 # How strongly alignment happens
    # HARMONY_CYCLE_NAMES = ["circadian", "heartbeat", "breath"]; HARMONY_CYCLE_SYNC_TARGET_BASE = 0.9
    # HARMONY_CYCLE_SYNC_INTENSITY_FACTOR = 0.1; HARMONY_CYCLE_SYNC_DURATION_FACTOR = 1.0
    # PLANETARY_FREQUENCIES = { # Example Frequencies (Hz) - NEEDS PROPER VALUES
    #     'Sun': 126.22, 'Moon': 210.42, 'Mercury': 141.27, 'Venus': 221.23,
    #     'Mars': 144.72, 'Jupiter': 183.58, 'Saturn': 147.85,
    #     'Uranus': 207.36, 'Neptune': 211.44, 'Pluto': 140.25
    # }
    # PLANETARY_ATTUNEMENT_CYCLES = 5 # Cycles for planetary resonance
    # PLANETARY_RESONANCE_RATE = 0.04 # Rate for planetary resonance factor gain
    # GAIA_CONNECTION_CYCLES = 3 # Cycles for Gaia connection
    # GAIA_CONNECTION_FACTOR = 0.1 # Rate for Gaia connection factor gain
    # STRESS_FEEDBACK_FACTOR = 0.01          # How much echo discordance impacts main soul stability variance
    # FLAG_READY_FOR_IDENTITY = "ready_for_identity"
    # # Need resonance calc constants if not imported globally
    # GOLDEN_RATIO = 1.618; RESONANCE_INTEGER_RATIO_TOLERANCE=0.02; RESONANCE_PHI_RATIO_TOLERANCE=0.03

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from .creator_entanglement import calculate_resonance # Use the detailed one
    try: from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
    except ImportError: aspect_dictionary = None # Optional for element mapping
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

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

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds and cord integrity factor. Raises ValueError on failure. """
    logger.debug(f"Checking Earth Attunement prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("Invalid SoulSpark object.")

    if not getattr(soul_spark, FLAG_READY_FOR_EARTH, False): # Set by Life Cord
        msg = f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_EARTH}."; logger.error(msg); raise ValueError(msg)

    cord_integrity = getattr(soul_spark, "cord_integrity", -1.0)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if cord_integrity < 0 or stability_su < 0 or coherence_cu < 0:
        msg = "Prerequisite failed: Soul missing cord_integrity, stability or coherence."; logger.error(msg); raise AttributeError(msg)

    if cord_integrity < HARMONY_PREREQ_CORD_INTEGRITY_MIN:
        msg = f"Prerequisite failed: Cord integrity ({cord_integrity:.3f}) < {HARMONY_PREREQ_CORD_INTEGRITY_MIN}."; logger.error(msg); raise ValueError(msg)
    if stability_su < HARMONY_PREREQ_STABILITY_MIN_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {HARMONY_PREREQ_STABILITY_MIN_SU} SU."; logger.error(msg); raise ValueError(msg)
    if coherence_cu < HARMONY_PREREQ_COHERENCE_MIN_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {HARMONY_PREREQ_COHERENCE_MIN_CU} CU."; logger.error(msg); raise ValueError(msg)

    if getattr(soul_spark, FLAG_EARTH_ATTUNED, False) or getattr(soul_spark, FLAG_ECHO_PROJECTED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_ECHO_PROJECTED}/{FLAG_EARTH_ATTUNED}. Re-running.")

    logger.debug("Earth Attunement prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. Raises error if missing/invalid. """
    logger.debug(f"Ensuring properties for Earth Attunement (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'elements', 'earth_cycles',
                'planetary_resonance', 'gaia_connection', 'earth_resonance',
                'spiritual_energy', 'life_cord', 'aspects', 'frequency_history'] # Added freq history
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Earth Attunement: {missing}")

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    if not isinstance(soul_spark.elements, dict): raise TypeError("elements must be dict.")
    if not isinstance(soul_spark.earth_cycles, dict): raise TypeError("earth_cycles must be dict.")
    if not isinstance(soul_spark.life_cord, dict): raise TypeError("life_cord must be dict.")
    if soul_spark.spiritual_energy < 0: raise ValueError("Spiritual energy cannot be negative.")
    if not isinstance(soul_spark.frequency_history, list): raise TypeError("frequency_history must be a list.")

    # Initialize echo state if needed
    if not hasattr(soul_spark, 'echo_state'): setattr(soul_spark, 'echo_state', None)
    # Initialize planetary state if needed
    if not hasattr(soul_spark, 'governing_planet'): setattr(soul_spark, 'governing_planet', None)
    if not hasattr(soul_spark, 'planetary_resonance'): setattr(soul_spark, 'planetary_resonance', 0.0)
    if not hasattr(soul_spark, 'gaia_connection'): setattr(soul_spark, 'gaia_connection', 0.0)

    logger.debug("Soul properties ensured for Earth Attunement.")

# --- Core Attunement Sub-Functions ---

def _project_soul_echo(soul_spark: SoulSpark) -> Tuple[Dict, float]:
    """ Projects the echo, returns initial echo state and energy cost. """
    logger.info("EH Sub-Step: Projecting Soul Echo...")
    projection_cost = soul_spark.spiritual_energy * ECHO_PROJECTION_ENERGY_COST_FACTOR
    if soul_spark.spiritual_energy < projection_cost + 1.0: # Ensure some energy remains
        raise ValueError(f"Insufficient spiritual energy ({soul_spark.spiritual_energy:.1f}) to project echo (cost ~{projection_cost:.1f}).")

    soul_spark.spiritual_energy -= projection_cost
    timestamp = datetime.now().isoformat()
    # Echo starts with main soul's current state, simplified
    echo_state = {
        'frequency_hz': soul_spark.frequency,
        'elements': soul_spark.elements.copy(), # Copy current elemental balance
        'earth_cycles': soul_spark.earth_cycles.copy(), # Copy current cycle sync
        'projected_time': timestamp,
        'current_resonance_with_earth': 0.0 # Initial resonance
    }
    setattr(soul_spark, 'echo_state', echo_state)
    setattr(soul_spark, FLAG_ECHO_PROJECTED, True)
    soul_spark.last_modified = timestamp
    logger.info(f"  Echo projected. Cost: {projection_cost:.1f} SEU. Remaining Spirit E: {soul_spark.spiritual_energy:.1f} SEU.")
    return echo_state, projection_cost

def _perform_frequency_attunement(soul_spark: SoulSpark, echo_state: Dict) -> None:
    """ Attunes main soul frequency based on echo's resonance with Earth frequencies. """
    logger.debug("  EH Attunement: Frequency...")
    main_freq = soul_spark.frequency
    # Calculate echo's resonance with primary Earth frequencies
    res_schumann = calculate_resonance(echo_state['frequency_hz'], SCHUMANN_FREQUENCY)
    res_core = calculate_resonance(echo_state['frequency_hz'], EARTH_FREQUENCY)
    # Weighted average resonance score for the echo
    echo_earth_resonance = (res_schumann * 0.7 + res_core * 0.3) # Weight Schumann higher?
    echo_state['current_resonance_with_earth'] = echo_earth_resonance # Update echo state
    logger.debug(f"    Echo Earth Freq Resonance: Schum={res_schumann:.3f}, Core={res_core:.3f} -> Avg={echo_earth_resonance:.3f}")

    # Nudge main soul frequency towards a weighted Earth target
    target_freq = (SCHUMANN_FREQUENCY * 0.6 + EARTH_FREQUENCY * 0.2 + main_freq * 0.2) # Target blend
    freq_diff = target_freq - main_freq
    # Nudge amount modulated by echo's success (resonance)
    nudge_factor = echo_earth_resonance * HARMONY_FREQ_TUNING_FACTOR * 0.5 # Gentle nudge rate
    tuning_amount = freq_diff * nudge_factor
    new_freq = max(SCHUMANN_FREQUENCY * 0.5, main_freq + tuning_amount)
    logger.debug(f"    Soul Freq Nudge: Current={main_freq:.1f}, Target={target_freq:.1f}, EchoRes={echo_earth_resonance:.3f}, Nudge={tuning_amount:.2f} -> New={new_freq:.1f}")

    soul_spark.frequency = float(new_freq)
    if hasattr(soul_spark, '_validate_or_init_frequency_structure'):
        soul_spark._validate_or_init_frequency_structure()
    else: raise AttributeError("Missing frequency structure update method.")


def _perform_elemental_alignment(soul_spark: SoulSpark, echo_state: Dict) -> None:
    """ Aligns main soul elements based on echo resonance. """
    logger.debug("  EH Attunement: Elemental...")
    soul_elements = soul_spark.elements
    malkuth_element = "earth" # Default
    if aspect_dictionary:
         try: malkuth_element = aspect_dictionary.get_aspects("malkuth").get("element", "earth").lower()
         except Exception: pass

    total_elem_shift = 0
    echo_resonance = echo_state.get('current_resonance_with_earth', 0.0) # Use overall echo resonance

    for element in EARTH_ELEMENTS:
        element_lower = element.lower()
        current_strength = float(soul_elements.get(element_lower, 0.0)) # Soul's current 0-1 factor
        target_strength = ELEMENTAL_TARGET_EARTH if element_lower == malkuth_element else ELEMENTAL_TARGET_OTHER # Target 0-1 factor
        diff = target_strength - current_strength
        # Adjustment modulated by echo resonance
        adjustment = diff * echo_resonance * ELEMENTAL_ALIGN_INTENSITY_FACTOR # Delta for 0-1 factor
        new_strength = max(0.0, min(1.0, current_strength + adjustment))
        soul_elements[element_lower] = float(new_strength) # Update soul's dict
        total_elem_shift += abs(adjustment)
    logger.debug(f"    Soul Elem Align: Total Shift={total_elem_shift:.4f} (based on EchoRes={echo_resonance:.3f})")
    # Update overall alignment score
    alignment_levels = []
    for elem, target in zip(EARTH_ELEMENTS, [ELEMENTAL_TARGET_EARTH if el.lower()==malkuth_element else ELEMENTAL_TARGET_OTHER for el in EARTH_ELEMENTS]):
        final_str = soul_elements.get(elem.lower(), 0.0)
        level = 1.0 - abs(target - final_str) / max(FLOAT_EPSILON, target) if target > FLOAT_EPSILON else (1.0 if abs(final_str)<FLOAT_EPSILON else 0.0)
        alignment_levels.append(level)
    overall_alignment = np.mean(alignment_levels) if alignment_levels else 0.0
    setattr(soul_spark, "elemental_alignment", float(overall_alignment)) # Store overall 0-1 score

def _perform_cycle_synchronization(soul_spark: SoulSpark, echo_state: Dict) -> None:
    """ Synchronizes main soul cycle factors based on echo resonance. """
    logger.debug("  EH Attunement: Cycle Sync...")
    soul_cycles = soul_spark.earth_cycles
    total_cycle_shift = 0
    echo_resonance = echo_state.get('current_resonance_with_earth', 0.0)

    for cycle_name in HARMONY_CYCLE_NAMES:
        current_sync = float(soul_cycles.get(cycle_name, 0.3)) # Soul's current 0-1 factor
        target_sync = HARMONY_CYCLE_SYNC_TARGET_BASE # Simplified target
        diff = target_sync - current_sync
        # Adjustment modulated by echo resonance
        adjustment = diff * echo_resonance * HARMONY_CYCLE_SYNC_INTENSITY_FACTOR * HARMONY_CYCLE_SYNC_DURATION_FACTOR # Delta for 0-1 factor
        new_sync = max(0.0, min(1.0, current_sync + adjustment))
        soul_cycles[cycle_name] = float(new_sync)
        total_cycle_shift += abs(adjustment)
    logger.debug(f"    Soul Cycle Sync: Total Shift={total_cycle_shift:.4f} (based on EchoRes={echo_resonance:.3f})")
    # Update overall sync score
    sync_levels = []
    for name in HARMONY_CYCLE_NAMES:
        final_sync = soul_cycles.get(name, 0.0)
        target = HARMONY_CYCLE_SYNC_TARGET_BASE
        level = 1.0 - abs(target - final_sync) / max(FLOAT_EPSILON, target) if target > FLOAT_EPSILON else (1.0 if abs(final_sync)<FLOAT_EPSILON else 0.0)
        sync_levels.append(level)
    overall_sync = np.mean(sync_levels) if sync_levels else 0.0
    setattr(soul_spark, "cycle_synchronization", float(overall_sync)) # Store overall 0-1 score


def _perform_planetary_resonance(soul_spark: SoulSpark, echo_state: Dict, cycles: int) -> None:
    """ Attunes soul planetary resonance factor (0-1), gently nudges frequency. """
    logger.info(f"EH Sub-Step: Planetary Resonance ({cycles} cycles)...")
    if cycles <= 0: return
    try:
        # Determine governing planet once
        if getattr(soul_spark, 'governing_planet', None) is None:
            planet_name = random.choice(list(PLANETARY_FREQUENCIES.keys()))
            setattr(soul_spark, 'governing_planet', planet_name)
            logger.debug(f"    Assigned Governing Planet: {planet_name}")
        else:
            planet_name = soul_spark.governing_planet

        target_frequency = PLANETARY_FREQUENCIES.get(planet_name)
        if target_frequency is None or target_frequency <= FLOAT_EPSILON:
            raise ValueError(f"Invalid or missing frequency for planet {planet_name}")
        logger.debug(f"    Target Planet: {planet_name} (Freq: {target_frequency:.2f} Hz)")

        for cycle in range(cycles):
            current_resonance_factor = getattr(soul_spark, "planetary_resonance", 0.0) # Soul's factor
            soul_freq = soul_spark.frequency
            echo_resonance = echo_state.get('current_resonance_with_earth', 0.0) # Use echo's earth resonance as modulator

            # Calculate resonance between soul and planet
            freq_resonance = calculate_resonance(soul_freq, target_frequency)

            # Calculate potential gain, modulated by echo resonance and frequency resonance
            potential_gain = (1.0 - current_resonance_factor) * freq_resonance * echo_resonance
            actual_gain = potential_gain * PLANETARY_RESONANCE_RATE # Apply rate
            new_resonance_factor = min(1.0, current_resonance_factor + actual_gain)
            setattr(soul_spark, "planetary_resonance", float(new_resonance_factor))

            # Gently nudge soul frequency if resonance is high
            if freq_resonance > 0.85: # High resonance threshold for nudge
                freq_diff = target_frequency - soul_freq
                nudge = freq_diff * 0.01 # Very small nudge factor per cycle
                soul_spark.frequency = max(FLOAT_EPSILON * 10, soul_freq + nudge)
                # No need to regen harmonics every cycle unless freq change is large

            logger.debug(f"    Planet Res Cycle {cycle+1}: SoulFreq={soul_freq:.1f}, FreqRes={freq_resonance:.3f}, EchoRes={echo_resonance:.3f} -> Factor {current_resonance_factor:.4f} -> {new_resonance_factor:.4f} (+{actual_gain:.5f})")

        # Final harmonic update if frequency changed significantly overall
        if hasattr(soul_spark, '_validate_or_init_frequency_structure'):
            soul_spark._validate_or_init_frequency_structure()

        logger.info(f"Planetary resonance complete. Final Factor: {soul_spark.planetary_resonance:.4f}")

    except Exception as e:
        logger.error(f"Error during planetary resonance: {e}", exc_info=True)
        raise RuntimeError("Planetary resonance attunement failed critically.") from e

def _perform_gaia_connection(soul_spark: SoulSpark, cycles: int) -> None:
    """ Establishes Gaia connection factor (0-1). """
    logger.info(f"EH Sub-Step: Gaia Connection ({cycles} cycles)...")
    if cycles <= 0: return
    try:
        for cycle in range(cycles):
            current_connection = getattr(soul_spark, "gaia_connection", 0.0) # Soul's factor
            earth_resonance = soul_spark.earth_resonance # Current earth resonance
            cord_integrity = soul_spark.cord_integrity # Current cord integrity

            # Potential increase depends on prerequisites being met (resonance, integrity)
            potential_gain = (1.0 - current_connection) * earth_resonance * cord_integrity
            actual_gain = potential_gain * GAIA_CONNECTION_FACTOR # Apply rate
            new_connection = min(1.0, current_connection + actual_gain)
            setattr(soul_spark, "gaia_connection", float(new_connection))
            logger.debug(f"    Gaia Conn Cycle {cycle+1}: EarthRes={earth_resonance:.3f}, CordInt={cord_integrity:.3f} -> Factor {current_connection:.4f} -> {new_connection:.4f} (+{actual_gain:.5f})")

        logger.info(f"Gaia connection complete. Final Factor: {soul_spark.gaia_connection:.4f}")

    except Exception as e:
        logger.error(f"Error during Gaia connection: {e}", exc_info=True)
        raise RuntimeError("Gaia connection establishment failed critically.") from e

def _update_soul_final_properties(soul_spark: SoulSpark):
    """ Updates final flags and adds memory echo. S/C changes emerge from update_state. """
    logger.info(f"Updating soul {soul_spark.spark_id} final attunement properties...")
    try:
        setattr(soul_spark, FLAG_EARTH_ATTUNED, True) # Use new flag name
        setattr(soul_spark, FLAG_READY_FOR_IDENTITY, True)
        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)

        # --- REMOVED Direct S/C Bonuses ---
        final_earth_resonance = soul_spark.earth_resonance # Get final score for echo
        logger.debug(f"  Final Earth Attunement state: EarthRes={final_earth_resonance:.4f}")

        if hasattr(soul_spark, 'add_memory_echo'):
            echo_msg = f"Earth attunement via echo complete. EarthRes:{final_earth_resonance:.3f}"
            soul_spark.add_memory_echo(echo_msg)

        logger.info("Soul flags updated post-attunement. Ready for Identity.")

    except Exception as e:
        logger.error(f"Error updating final properties: {e}", exc_info=True)
        raise RuntimeError("Final property update failed critically.") from e


# --- Orchestration Function ---
def perform_earth_harmonization(soul_spark: SoulSpark, intensity: float = EARTH_HARMONY_INTENSITY_DEFAULT, duration_factor: float = EARTH_HARMONY_DURATION_FACTOR_DEFAULT) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs echo projection and resonance-driven attunement. Modifies SoulSpark. """
    # --- Input Validation & Setup ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.") # Keep factors for potential modulation

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Earth Attunement [Echo Projection] for Soul {spark_id} ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}
    total_stress_factor = 0.0 # Track potential stress

    try:
        _ensure_soul_properties(soul_spark) # Raises error if fails
        _check_prerequisites(soul_spark) # Raises error if fails

        # Store Initial State
        initial_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'spiritual_energy_seu': soul_spark.spiritual_energy,
            'earth_resonance': soul_spark.earth_resonance,
            'elemental_alignment': soul_spark.elemental_alignment, 'cycle_synchronization': soul_spark.cycle_synchronization,
            'planetary_resonance': soul_spark.planetary_resonance, 'gaia_connection': soul_spark.gaia_connection
        }
        logger.info(f"EH Initial State: S={initial_state['stability_su']:.1f}, C={initial_state['coherence_cu']:.1f}, E_spirit={initial_state['spiritual_energy_seu']:.1f}, EarthRes={initial_state['earth_resonance']:.3f}")

        # --- 1. Project Echo ---
        echo_state, projection_cost = _project_soul_echo(soul_spark)
        process_metrics_summary['steps']['projection'] = {'energy_cost_seu': projection_cost}

        # --- 2. Run Attunement Cycles ---
        logger.info(f"EH Step 2: Running {ECHO_ATTUNEMENT_CYCLES} Attunement Cycles...")
        for cycle in range(ECHO_ATTUNEMENT_CYCLES):
            logger.debug(f"--- EH Attunement Cycle {cycle + 1}/{ECHO_ATTUNEMENT_CYCLES} ---")
            # Attune frequency, elements, cycles based on echo resonance
            _perform_frequency_attunement(soul_spark, echo_state)
            _perform_elemental_alignment(soul_spark, echo_state)
            _perform_cycle_synchronization(soul_spark, echo_state)
            # Calculate stress based on echo-soul difference AFTER adjustments
            current_main_freq = soul_spark.frequency
            freq_diff_stress = abs(current_main_freq - echo_state['frequency_hz']) / max(current_main_freq, FLOAT_EPSILON)
            # (Simplified stress calc - focus on freq difference for now)
            stress = freq_diff_stress * STRESS_FEEDBACK_FACTOR
            total_stress_factor += stress
            logger.debug(f"    Cycle {cycle+1} Stress Factor: {stress:.4f} (Total Accumulated: {total_stress_factor:.4f})")


        # --- 3. Planetary & Gaia Attunement (Post-basic attunement) ---
        _perform_planetary_resonance(soul_spark, echo_state, PLANETARY_ATTUNEMENT_CYCLES)
        _perform_gaia_connection(soul_spark, GAIA_CONNECTION_CYCLES)

        # --- 4. Apply Accumulated Stress Feedback ---
        avg_stress_factor = total_stress_factor / ECHO_ATTUNEMENT_CYCLES if ECHO_ATTUNEMENT_CYCLES > 0 else 0
        if avg_stress_factor > FLOAT_EPSILON and hasattr(soul_spark, 'frequency_history') and soul_spark.frequency_history:
             current_std = np.std(soul_spark.frequency_history)
             # Increase variance slightly based on average stress
             variance_increase = current_std * avg_stress_factor * 0.2 # Small effect
             # Add noise to frequency based on this increased variance estimate
             noise_to_add = np.random.normal(0, variance_increase)
             soul_spark.frequency += noise_to_add
             soul_spark.frequency = max(FLOAT_EPSILON * 10, soul_spark.frequency) # Clamp
             # Optionally update history - maybe not needed if update_state handles it
             # soul_spark.frequency_history.append(soul_spark.frequency)
             # if len(soul_spark.frequency_history) > 20: soul_spark.frequency_history.pop(0)
             logger.debug(f"  Applied Avg Stress Feedback: {avg_stress_factor:.4f}, Freq Noise Added: {noise_to_add:+.2f} Hz")


        # --- 5. Final Update & Metrics ---
        _update_soul_final_properties(soul_spark) # Sets flags

        logger.debug("EH: Calling final update_state...")
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            logger.debug(f"EH: S/C after final update: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        else: raise AttributeError("SoulSpark missing 'update_state' method.")

        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)

        end_time_iso = last_mod_time; end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'spiritual_energy_seu': soul_spark.spiritual_energy,
            'earth_resonance': soul_spark.earth_resonance,
            'elemental_alignment': soul_spark.elemental_alignment, 'cycle_synchronization': soul_spark.cycle_synchronization,
            'planetary_resonance': soul_spark.planetary_resonance, 'gaia_connection': soul_spark.gaia_connection,
            FLAG_EARTH_ATTUNED: getattr(soul_spark, FLAG_EARTH_ATTUNED),
            FLAG_READY_FOR_IDENTITY: getattr(soul_spark, FLAG_READY_FOR_IDENTITY)
        }
        overall_metrics = {
            'action': 'earth_harmonization_echo', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state, 'final_state': final_state,
            'echo_projection_cost_seu': projection_cost,
            'attunement_cycles': ECHO_ATTUNEMENT_CYCLES,
            'avg_stress_feedback': avg_stress_factor,
            'earth_resonance_change': final_state['earth_resonance'] - initial_state['earth_resonance'],
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('earth_harmonization_summary', overall_metrics)

        logger.info(f"--- Earth Attunement Completed Successfully for Soul {spark_id} ---")
        logger.info(f"  Final State: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}, EarthRes={soul_spark.earth_resonance:.3f}, PlanetRes={soul_spark.planetary_resonance:.3f}, GaiaConn={soul_spark.gaia_connection:.3f}")
        return soul_spark, overall_metrics

    # --- Error Handling (Ensure Hard Fail) ---
    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Earth Attunement failed for {spark_id}: {e_val}", exc_info=True)
         failed_step = 'projection' if not getattr(soul_spark, FLAG_ECHO_PROJECTED, False) else 'prerequisites/validation'
         record_eh_failure(spark_id, start_time_iso, failed_step, str(e_val))
         # Hard fail
         raise e_val
    except RuntimeError as e_rt:
         logger.critical(f"Earth Attunement failed critically for {spark_id}: {e_rt}", exc_info=True)
         failed_step = 'runtime'
         record_eh_failure(spark_id, start_time_iso, failed_step, str(e_rt))
         setattr(soul_spark, FLAG_ECHO_PROJECTED, False); setattr(soul_spark, FLAG_EARTH_ATTUNED, False); setattr(soul_spark, FLAG_READY_FOR_IDENTITY, False)
         # Hard fail
         raise e_rt
    except Exception as e:
         logger.critical(f"Unexpected error during Earth Attunement for {spark_id}: {e}", exc_info=True)
         failed_step = 'unexpected'
         record_eh_failure(spark_id, start_time_iso, failed_step, str(e))
         setattr(soul_spark, FLAG_ECHO_PROJECTED, False); setattr(soul_spark, FLAG_EARTH_ATTUNED, False); setattr(soul_spark, FLAG_READY_FOR_IDENTITY, False)
         # Hard fail
         raise RuntimeError(f"Unexpected Earth Attunement failure: {e}") from e

# --- Failure Metric Helper ---
def record_eh_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('earth_harmonization_summary', {
                'action': 'earth_harmonization_echo', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': end_time,
                'duration_seconds': duration,
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record EH failure metrics for {spark_id}: {metric_e}")


# --- END OF FILE src/stage_1/soul_formation/earth_harmonisation.py ---