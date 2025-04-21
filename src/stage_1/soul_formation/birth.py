# --- START OF FILE birth.py ---

"""
Birth Process Functions (Refactored - Operates on SoulSpark Object, Uses Constants)

Handles the process of birthing a soul into physical incarnation, including
attachment, life cord transfer, memory veil deployment, and first breath.
Modifies the SoulSpark object instance directly.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import time
import uuid
from datetime import datetime

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    # Import necessary constants FROM THE CENTRAL FILE
    from src.constants import * # Import all for convenience
    # Extract specific Earth freq if needed
    EARTH_BREATH_FREQUENCY = EARTH_FREQUENCIES.get("breath", 0.2) # Default ~12 breaths/min
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants from src.constants: {e}. Birth process cannot function correctly.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.void.soul_spark import SoulSpark
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}. Birth process cannot function.")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import metrics_tracking: {e}. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder: def record_metrics(*args, **kwargs): pass
    metrics = MetricsPlaceholder()


# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """Checks if the soul meets prerequisites for birth using constants."""
    logger.debug(f"Checking birth prerequisites for soul {soul_spark.spark_id}...")
    # Check prerequisite flags first
    if BIRTH_PREREQ_EARTH_HARMONIZED and not getattr(soul_spark, "earth_harmonized", False):
        logger.error("Prerequisite failed: Soul has not been harmonized with Earth.")
        return False
    if BIRTH_PREREQ_READY_FOR_BIRTH and not getattr(soul_spark, "ready_for_birth", False):
        logger.error("Prerequisite failed: Soul not marked ready for birth (e.g., after Earth Harmonization).")
        return False

    # Check numerical prerequisites using constants
    cord_integrity = getattr(soul_spark, "cord_integrity", 0.0)
    if cord_integrity < BIRTH_PREREQ_CORD_INTEGRITY_MIN:
        logger.error(f"Prerequisite failed: Life cord integrity ({cord_integrity:.3f}) below threshold ({BIRTH_PREREQ_CORD_INTEGRITY_MIN}).")
        return False
    earth_resonance = getattr(soul_spark, "earth_resonance", 0.0)
    if earth_resonance < BIRTH_PREREQ_EARTH_RESONANCE_MIN:
        logger.error(f"Prerequisite failed: Earth resonance ({earth_resonance:.3f}) below threshold ({BIRTH_PREREQ_EARTH_RESONANCE_MIN}).")
        return False

    logger.debug("Birth prerequisites met.")
    return True

def _connect_to_physical_form(soul_spark: SoulSpark, intensity: float) -> Tuple[float, float, Dict[str, Any]]:
    """Connects soul to physical form using constants. Modifies SoulSpark implicitly via return values used later. Fails hard."""
    logger.info("Phase: Connecting to physical form...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0.")

    try:
        earth_resonance = getattr(soul_spark, "earth_resonance", 0.6) # Get required attrs
        cord_integrity = getattr(soul_spark, "cord_integrity", 0.7)
        logger.debug(f"  Input values: EarthRes={earth_resonance:.3f}, CordInteg={cord_integrity:.3f}, Intensity={intensity:.2f}")

        # Calculate connection strength and trauma using constants
        base_strength = (earth_resonance * BIRTH_CONN_WEIGHT_RESONANCE +
                         cord_integrity * BIRTH_CONN_WEIGHT_INTEGRITY)
        trauma_factor = intensity * BIRTH_CONN_TRAUMA_FACTOR
        connection_factor = intensity * BIRTH_CONN_STRENGTH_FACTOR
        max_potential = base_strength * (1.0 + connection_factor)
        connection_strength = min(BIRTH_CONN_STRENGTH_CAP, max_potential) # Use constant cap
        trauma_level = trauma_factor # Trauma directly related to intensity factor
        acceptance = max(BIRTH_ACCEPTANCE_MIN, 1.0 - trauma_level * BIRTH_ACCEPTANCE_TRAUMA_FACTOR) # Use constants

        logger.debug(f"  Calculated: BaseStr={base_strength:.3f}, Trauma={trauma_level:.3f}, ConnFactor={connection_factor:.3f}")
        logger.debug(f"  Result: ConnStrength={connection_strength:.3f}, Acceptance={acceptance:.3f}")

        phase_metrics = {
            "base_strength": float(base_strength), "trauma_level": float(trauma_level),
            "connection_strength": float(connection_strength), "form_acceptance": float(acceptance),
            "timestamp": datetime.now().isoformat() }
        try: metrics.record_metrics('birth_connection', phase_metrics)
        except Exception as e: logger.error(f"Failed to record connection metrics: {e}")

        logger.info(f"Physical form connection phase complete. Connection: {connection_strength:.3f}, Acceptance: {acceptance:.3f}")
        # Return values needed for subsequent phases and finalization
        return float(connection_strength), float(acceptance), phase_metrics

    except AttributeError as ae: logger.error(f"SoulSpark missing attribute for connection: {ae}"); raise
    except Exception as e: logger.error(f"Error connecting to physical form: {e}", exc_info=True); raise RuntimeError("Physical form connection failed.") from e


def _transfer_life_cord(soul_spark: SoulSpark, physical_connection: float, intensity: float) -> Dict[str, Any]:
    """Transfers life cord using constants. Modifies SoulSpark. Fails hard."""
    logger.info("Phase: Transferring life cord...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not hasattr(soul_spark, 'life_cord') or not isinstance(soul_spark.life_cord, dict):
        raise AttributeError("SoulSpark requires a valid 'life_cord' dictionary attribute.")
    if not hasattr(soul_spark, 'cord_integrity'):
         raise AttributeError("SoulSpark missing 'cord_integrity' attribute.")

    try:
        life_cord = soul_spark.life_cord # Get the dict
        cord_integrity = soul_spark.cord_integrity
        logger.debug(f"  Input values: CordInteg={cord_integrity:.3f}, PhysConn={physical_connection:.3f}, Intensity={intensity:.2f}")

        # Calculate transfer efficiency using constants
        transfer_efficiency = cord_integrity * (1.0 - intensity * BIRTH_CORD_TRANSFER_INTENSITY_FACTOR)
        transfer_efficiency = max(0.0, min(1.0, transfer_efficiency)) # Clamp
        logger.debug(f"  Calculated Transfer Efficiency: {transfer_efficiency:.4f}")

        # Calculate form integration using constants
        form_integration = physical_connection * BIRTH_CORD_INTEGRATION_CONN_FACTOR
        form_integration = max(0.0, min(1.0, form_integration))
        logger.debug(f"  Calculated Form Integration: {form_integration:.4f}")

        # Calculate new cord integrity and bandwidth
        new_integrity = cord_integrity * transfer_efficiency
        original_bandwidth = life_cord.get("bandwidth", 100.0) # Get current or default
        new_bandwidth = original_bandwidth * transfer_efficiency
        logger.debug(f"  New Integrity: {new_integrity:.4f}, New Bandwidth: {new_bandwidth:.2f}")

        # --- Update SoulSpark ---
        # Modify the existing life_cord dictionary on the soul_spark
        life_cord["form_integration"] = float(form_integration)
        life_cord["physical_anchored"] = True
        life_cord["bandwidth"] = float(new_bandwidth)
        # Update the top-level integrity attribute
        setattr(soul_spark, "cord_integrity", float(new_integrity))
        # Add form_integration attribute directly? Or keep in cord dict? Keep in cord for now.
        # setattr(soul_spark, "form_integration", float(form_integration))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "transfer_efficiency": float(transfer_efficiency), "form_integration": float(form_integration),
            "original_integrity": float(cord_integrity), "new_integrity": float(new_integrity),
            "integrity_change": float(new_integrity - cord_integrity),
            "original_bandwidth": float(original_bandwidth), "new_bandwidth": float(new_bandwidth),
            "bandwidth_reduction": float(original_bandwidth - new_bandwidth),
            "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('birth_cord_transfer', phase_metrics)
        except Exception as e: logger.error(f"Failed to record cord transfer metrics: {e}")

        logger.info(f"Life cord transferred. Efficiency: {transfer_efficiency:.3f}, New Integrity: {new_integrity:.3f}, Form Integration: {form_integration:.3f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error transferring life cord: {e}", exc_info=True); raise RuntimeError("Life cord transfer failed.") from e


def _deploy_memory_veil(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """Deploys memory veil using constants. Modifies SoulSpark. Fails hard."""
    logger.info("Phase: Deploying memory veil...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")

    try:
        # Calculate veil properties using constants
        veil_strength = BIRTH_VEIL_STRENGTH_BASE + intensity * BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR
        veil_permanence = BIRTH_VEIL_PERMANENCE_BASE + intensity * BIRTH_VEIL_PERMANENCE_INTENSITY_FACTOR
        base_retention = BIRTH_VEIL_RETENTION_BASE + intensity * BIRTH_VEIL_RETENTION_INTENSITY_FACTOR
        base_retention = max(BIRTH_VEIL_RETENTION_MIN, base_retention) # Ensure minimum retention

        # Clamp values
        veil_strength = max(0.0, min(1.0, veil_strength))
        veil_permanence = max(0.0, min(1.0, veil_permanence))
        base_retention = max(0.0, min(1.0, base_retention))
        logger.debug(f"  Calculated Veil Properties: Strength={veil_strength:.3f}, Permanence={veil_permanence:.3f}, BaseRetention={base_retention:.3f}")

        # Calculate retention for specific memory types using constant modifiers
        memory_retentions = {}
        for mem_type, mod in BIRTH_VEIL_MEMORY_RETENTION_MODS.items():
            retention = min(1.0, base_retention + mod) # Add modifier
            memory_retentions[mem_type] = float(retention)
        logger.debug(f"  Specific Memory Retentions: {memory_retentions}")

        # Create veil configuration dictionary
        veil_config = {
            "strength": float(veil_strength), "permanence": float(veil_permanence),
            "base_retention": float(base_retention), "memory_retentions": memory_retentions,
            "deployment_time": datetime.now().isoformat()
        }

        # --- Update SoulSpark ---
        setattr(soul_spark, "memory_veil", veil_config)
        setattr(soul_spark, "memory_retention", float(base_retention)) # Store base retention
        setattr(soul_spark, 'last_modified', veil_config['deployment_time'])

        # --- Calculate & Record Metrics ---
        phase_metrics = veil_config.copy() # Use veil_config as basis for metrics
        phase_metrics["timestamp"] = veil_config['deployment_time']
        try: metrics.record_metrics('birth_memory_veil', phase_metrics)
        except Exception as e: logger.error(f"Failed to record memory veil metrics: {e}")

        logger.info(f"Memory veil deployed. Strength: {veil_strength:.3f}, Base Retention: {base_retention:.3f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error deploying memory veil: {e}", exc_info=True); raise RuntimeError("Memory veil deployment failed.") from e


def _first_breath_integration(soul_spark: SoulSpark, physical_connection: float, intensity: float) -> Dict[str, Any]:
    """Integrates first breath using constants. Modifies SoulSpark. Fails hard."""
    logger.info("Phase: Integrating first breath...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")

    try:
        earth_breath_freq = EARTH_BREATH_FREQUENCY # Use constant
        earth_resonance = getattr(soul_spark, "earth_resonance", 0.6) # Assumed set by harmonization
        logger.debug(f"  Input values: EarthRes={earth_resonance:.3f}, PhysConn={physical_connection:.3f}, Intensity={intensity:.2f}")

        # Calculate breath properties using constants
        breath_amplitude = BIRTH_BREATH_AMP_BASE + intensity * BIRTH_BREATH_AMP_INTENSITY_FACTOR
        breath_depth = BIRTH_BREATH_DEPTH_BASE + intensity * BIRTH_BREATH_DEPTH_INTENSITY_FACTOR
        breath_sync = earth_resonance * BIRTH_BREATH_SYNC_RESONANCE_FACTOR
        integration_strength = physical_connection * BIRTH_BREATH_INTEGRATION_CONN_FACTOR

        # Clamp values
        breath_amplitude = max(0.0, min(1.0, breath_amplitude))
        breath_depth = max(0.0, min(1.0, breath_depth))
        breath_sync = max(0.0, min(1.0, breath_sync))
        integration_strength = max(0.0, min(1.0, integration_strength))
        logger.debug(f"  Calculated Breath Properties: Amp={breath_amplitude:.3f}, Depth={breath_depth:.3f}, Sync={breath_sync:.3f}, Integration={integration_strength:.3f}")

        # Calculate resonance boost and energy shift using constants
        resonance_boost = breath_sync * breath_depth * BIRTH_BREATH_RESONANCE_BOOST_FACTOR
        new_earth_resonance = min(1.0, earth_resonance + resonance_boost)

        energy_shift = breath_depth * integration_strength * BIRTH_BREATH_ENERGY_SHIFT_FACTOR
        physical_energy = BIRTH_BREATH_PHYSICAL_ENERGY_BASE + energy_shift * BIRTH_BREATH_PHYSICAL_ENERGY_SCALE
        spiritual_energy = BIRTH_BREATH_SPIRITUAL_ENERGY_BASE + energy_shift * BIRTH_BREATH_SPIRITUAL_ENERGY_SCALE
        physical_energy = max(0.0, min(1.0, physical_energy))
        spiritual_energy = max(BIRTH_BREATH_SPIRITUAL_ENERGY_MIN, min(1.0, spiritual_energy)) # Ensure min spiritual energy
        logger.debug(f"  Resonance Boost: {resonance_boost:.4f} -> New EarthRes={new_earth_resonance:.3f}")
        logger.debug(f"  Energy Shift: {energy_shift:.4f} -> PhysEnergy={physical_energy:.3f}, SpiritEnergy={spiritual_energy:.3f}")

        # Create breath configuration dictionary
        breath_config = {
            "frequency": float(earth_breath_freq), "amplitude": float(breath_amplitude), "depth": float(breath_depth),
            "sync_factor": float(breath_sync), "integration_strength": float(integration_strength),
            "physical_energy_level": float(physical_energy), "spiritual_energy_level": float(spiritual_energy),
            "timestamp": datetime.now().isoformat() }

        # --- Update SoulSpark ---
        setattr(soul_spark, "breath_pattern", breath_config)
        setattr(soul_spark, "earth_resonance", float(new_earth_resonance))
        setattr(soul_spark, "physical_energy", float(physical_energy))
        setattr(soul_spark, "spiritual_energy", float(spiritual_energy))
        setattr(soul_spark, 'last_modified', breath_config['timestamp'])

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "integration_strength": float(integration_strength), "resonance_boost": float(resonance_boost),
            "final_earth_resonance": float(new_earth_resonance), "physical_energy": float(physical_energy),
            "spiritual_energy": float(spiritual_energy), "timestamp": breath_config['timestamp'] }
        try: metrics.record_metrics('birth_first_breath', phase_metrics)
        except Exception as e: logger.error(f"Failed to record first breath metrics: {e}")

        logger.info(f"First breath integrated. Integration: {integration_strength:.3f}, PhysEnergy: {physical_energy:.3f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error integrating first breath: {e}", exc_info=True); raise RuntimeError("First breath integration failed.") from e


def _finalize_birth(soul_spark: SoulSpark, connection_strength: float, form_acceptance: float) -> Tuple[float, Dict[str, Any]]:
    """Finalizes birth using constants. Modifies SoulSpark. Fails hard."""
    logger.info("Phase: Finalizing birth process...")

    try:
        # Get breath integration strength if available
        breath_integration = getattr(soul_spark, 'breath_pattern', {}).get('integration_strength', 0.7) # Default if missing

        # Calculate total integration using constants
        total_integration = (connection_strength * BIRTH_FINAL_INTEGRATION_WEIGHT_CONN +
                             form_acceptance * BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT +
                             breath_integration * BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH)
        total_integration = max(0.0, min(1.0, total_integration))
        logger.debug(f"  Calculated Final Integration Level: {total_integration:.4f}")

        # Adjust core soul properties for physical existence using constants
        original_frequency = getattr(soul_spark, "frequency")
        original_stability = getattr(soul_spark, "stability")
        physical_frequency = original_frequency * BIRTH_FINAL_FREQ_FACTOR
        physical_stability = original_stability * BIRTH_FINAL_STABILITY_FACTOR

        # Ensure frequency remains positive
        physical_frequency = max(FLOAT_EPSILON, physical_frequency)
        logger.debug(f"  Adjusted Properties: Freq={physical_frequency:.2f} (from {original_frequency:.2f}), Stab={physical_stability:.3f} (from {original_stability:.3f})")

        # --- Update SoulSpark ---
        setattr(soul_spark, "frequency", float(physical_frequency))
        setattr(soul_spark, "stability", float(physical_stability))
        setattr(soul_spark, "physical_integration", float(total_integration))
        setattr(soul_spark, "incarnated", True) # Mark as incarnated
        birth_time = datetime.now().isoformat()
        setattr(soul_spark, "birth_time", birth_time)
        setattr(soul_spark, 'last_modified', birth_time)

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "final_integration": float(total_integration), "final_frequency": float(physical_frequency),
            "final_stability": float(physical_stability), "birth_timestamp": birth_time, "success": True }
        try: metrics.record_metrics('birth_finalization', phase_metrics)
        except Exception as e: logger.error(f"Failed to record finalization metrics: {e}")

        logger.info(f"Birth finalized. Integration: {total_integration:.3f}, Final Freq: {physical_frequency:.2f}, Final Stab: {physical_stability:.3f}")
        return float(total_integration), phase_metrics

    except Exception as e: logger.error(f"Error finalizing birth: {e}", exc_info=True); raise RuntimeError("Birth finalization failed.") from e


# --- Orchestration Function ---

def perform_birth(soul_spark: SoulSpark, intensity: float = 0.7) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs the complete birth process for a soul spark. Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of the birth process (0.1-1.0). Affects speed,
                           trauma, connection strength, veil strength.

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified (incarnated) SoulSpark object.
            - overall_metrics (Dict): Summary metrics for the entire birth process.

    Raises:
        TypeError: If soul_spark is not a SoulSpark instance.
        ValueError: If parameters invalid or prerequisites not met.
        RuntimeError: If any phase fails critically.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Starting Birth Process for Soul {spark_id} (Intensity={intensity:.2f}) ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}
    birth_successful = False # Flag to track success through phases

    try:
        # --- Check Prerequisites ---
        if not _check_prerequisites(soul_spark):
            raise ValueError("Soul prerequisites for birth not met.")
        logger.info("Prerequisites checked successfully.")

        # --- Store Initial State ---
        initial_state = {
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0), 'earth_resonance': getattr(soul_spark, 'earth_resonance', 0.0),
            'cord_integrity': getattr(soul_spark, 'cord_integrity', 0.0) }
        logger.info(f"Initial State: EarthRes={initial_state['earth_resonance']:.4f}, Freq={initial_state['frequency']:.2f}, Stab={initial_state['stability']:.4f}, CordInteg={initial_state['cord_integrity']:.3f}")

        # --- Run Phases (Fail hard within each) ---
        logger.info("Step 1: Connect to Physical Form...")
        connection_strength, form_acceptance, metrics1 = _connect_to_physical_form(soul_spark, intensity)
        process_metrics_summary['steps']['connection'] = metrics1

        logger.info("Step 2: Transfer Life Cord...")
        metrics2 = _transfer_life_cord(soul_spark, connection_strength, intensity)
        process_metrics_summary['steps']['cord_transfer'] = metrics2

        logger.info("Step 3: Deploy Memory Veil...")
        metrics3 = _deploy_memory_veil(soul_spark, intensity)
        process_metrics_summary['steps']['memory_veil'] = metrics3

        logger.info("Step 4: First Breath Integration...")
        metrics4 = _first_breath_integration(soul_spark, connection_strength, intensity)
        process_metrics_summary['steps']['first_breath'] = metrics4

        logger.info("Step 5: Finalize Birth...")
        final_integration, metrics5 = _finalize_birth(soul_spark, connection_strength, form_acceptance)
        process_metrics_summary['steps']['finalization'] = metrics5
        birth_successful = metrics5.get("success", False) # Mark success only if finalization passes

        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Capture final state metrics relevant to this process
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0), 'earth_resonance': getattr(soul_spark, 'earth_resonance', 0.0),
            'cord_integrity': getattr(soul_spark, 'cord_integrity', 0.0),
            'physical_integration': getattr(soul_spark, 'physical_integration', 0.0),
            'incarnated': getattr(soul_spark, 'incarnated', False) }

        overall_metrics = {
            'action': 'birth', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(), 'intensity_setting': intensity,
            'initial_state': initial_state, 'final_state': final_state,
            'final_integration': final_integration, 'success': birth_successful,
            # 'steps_metrics': process_metrics_summary['steps'] # Optional detail
        }
        try: metrics.record_metrics('birth_summary', overall_metrics)
        except Exception as e: logger.error(f"Failed to record summary metrics for birth process: {e}")

        if birth_successful:
             logger.info(f"--- Birth Process Completed Successfully for Soul {spark_id} ---")
             logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
             logger.info(f"Final Physical Integration: {final_integration:.4f}")
             logger.info(f"Final Soul Stability: {final_state['stability']:.4f}")
             logger.info(f"Incarnated Flag: {final_state['incarnated']}")
        else:
             # This path shouldn't be reached if finalization fails hard, but included for completeness
             logger.error(f"--- Birth Process Completed BUT FAILED for Soul {spark_id} ---")


        return soul_spark, overall_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Birth process failed critically for soul {spark_id}: {e}", exc_info=True)
        failed_step = "unknown"; steps_completed = list(process_metrics_summary['steps'].keys())
        if steps_completed: failed_step = steps_completed[-1]

        # Mark soul as failed this stage
        setattr(soul_spark, "incarnated", False)
        setattr(soul_spark, "birth_time", None)

        if METRICS_AVAILABLE:
             try: metrics.record_metrics('birth_summary', {
                  'action': 'birth', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - datetime.fromisoformat(start_time_iso)).total_seconds(),
                  'intensity_setting': intensity, 'success': False, 'error': str(e), 'failed_step': failed_step })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Birth process failed at step '{failed_step}'.") from e

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Birth Process Module Example...")
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        test_soul = SoulSpark()
        test_soul.spark_id="test_birth_001"
        # Set state *after* identity crystallization
        test_soul.stability = 0.90
        test_soul.coherence = 0.92
        test_soul.frequency = 174.0 # Grounded frequency
        test_soul.formation_complete = True
        test_soul.harmonically_strengthened = True
        test_soul.cord_formation_complete = True
        test_soul.earth_harmonized = True
        test_soul.identity_crystallized = True # Prerequisite
        test_soul.cord_integrity = 0.90 # Prerequisite
        test_soul.earth_resonance = 0.85 # Prerequisite
        test_soul.ready_for_birth = True # Prerequisite
        test_soul.name = "TestSoulAlpha"
        test_soul.soul_color = "gold"
        test_soul.last_modified = datetime.now().isoformat()
        test_soul.life_cord = {"bandwidth": 500.0} # Add dummy life cord dict needed by transfer phase

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Coherence: {test_soul.coherence:.4f}")
        print(f"  Earth Resonance: {test_soul.earth_resonance:.4f}")
        print(f"  Cord Integrity: {test_soul.cord_integrity:.4f}")
        print(f"  Ready for Birth: {getattr(test_soul, 'ready_for_birth', False)}")

        try:
            print("\n--- Running Birth Process ---")
            modified_soul, summary_metrics_result = perform_birth(
                soul_spark=test_soul,
                intensity=0.75 # Example intensity
            )

            print("\n--- Birth Complete ---")
            print("Final Soul State Summary:")
            print(f"  ID: {modified_soul.spark_id}")
            print(f"  Incarnated Flag: {getattr(modified_soul, 'incarnated', False)}")
            print(f"  Birth Time: {getattr(modified_soul, 'birth_time', 'N/A')}")
            print(f"  Physical Integration: {getattr(modified_soul, 'physical_integration', 'N/A'):.4f}")
            print(f"  Final Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            print(f"  Final Frequency: {getattr(modified_soul, 'frequency', 'N/A'):.2f} Hz")
            if hasattr(modified_soul, 'memory_veil'): print(f"  Memory Veil Strength: {modified_soul.memory_veil.get('strength'):.3f}")
            if hasattr(modified_soul, 'breath_pattern'): print(f"  Breath Integration: {modified_soul.breath_pattern.get('integration_strength'):.3f}")

            print("\nOverall Process Metrics:")
            # print(json.dumps(summary_metrics_result, indent=2, default=str))
            print(f"  Duration: {summary_metrics_result.get('duration_seconds', 'N/A'):.2f}s")
            print(f"  Success: {summary_metrics_result.get('success')}")
            print(f"  Final Integration Level: {summary_metrics_result.get('final_integration', 'N/A'):.4f}")

        except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as e:
            print(f"\n--- ERROR during Birth Process Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Birth Process Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\nBirth Process Module Example Finished.")


# --- END OF FILE birth.py ---

