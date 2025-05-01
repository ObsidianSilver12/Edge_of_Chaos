"""
Birth Process Functions (Refactored V4.1 - SEU/SU/CU Units)

Handles birth into physical incarnation. Uses 0-1 prerequisites for cord/earth resonance.
Splits energy into physical/spiritual (SEU). Modifies final frequency (Hz) and stability (SU).
Integrates Mother Resonance profile influence (passed as dict). Modifies SoulSpark.
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
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}.")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.error("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()

# Mother Glyph Path (Keep for reference)
MOTHER_GLYPH_PATH = "glyphs/glyph_resonance/encoded_glyphs/encoded_mother_sigil.jpeg"

# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using 0-1 factors for cord/earth resonance. """
    logger.debug(f"Checking birth prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): return False

    # 1. Stage Flag
    if not getattr(soul_spark, FLAG_READY_FOR_BIRTH, False): # Set by Identity
        logger.error(f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_BIRTH}.")
        return False
    # Make sure identity was actually crystallized (redundant check, but safe)
    if not getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False):
        logger.error("Prerequisite failed: Soul identity not crystallized (flag mismatch).")
        return False

    # 2. Minimum State Thresholds (Factors 0-1)
    cord_integrity = getattr(soul_spark, "cord_integrity", 0.0) # 0-1 factor
    earth_resonance = getattr(soul_spark, "earth_resonance", 0.0) # 0-1 factor

    if cord_integrity < BIRTH_PREREQ_CORD_INTEGRITY_MIN:
        logger.error(f"Prerequisite failed: Cord integrity ({cord_integrity:.3f}) < {BIRTH_PREREQ_CORD_INTEGRITY_MIN})"); return False
    if earth_resonance < BIRTH_PREREQ_EARTH_RESONANCE_MIN:
        logger.error(f"Prerequisite failed: Earth Resonance ({earth_resonance:.3f}) < {BIRTH_PREREQ_EARTH_RESONANCE_MIN})"); return False

    logger.debug("Birth prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties for birth. """
    logger.debug(f"Ensuring properties for birth process (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'energy', 'cord_integrity', 'earth_resonance', 'life_cord']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Birth: {missing}")

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    if not isinstance(soul_spark.life_cord, dict): soul_spark.life_cord = {} # Ensure dict

    # Initialize attributes set during birth if missing
    defaults = {
        "memory_veil": None, "breath_pattern": None, "physical_integration": 0.0,
        "incarnated": False, "birth_time": None, "physical_energy": 0.0,
        "spiritual_energy": soul_spark.energy # Initially all energy is spiritual
    }
    for attr, default in defaults.items():
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
             setattr(soul_spark, attr, default)

    logger.debug("Soul properties ensured for Birth.")


# --- Core Birth Functions (Updated energy split, stability units) ---

def _connect_to_physical_form(soul_spark: SoulSpark, intensity: float, mother_profile: Optional[Dict]) -> Tuple[float, float, Dict[str, Any]]:
    """ Connects soul to physical form. Returns connection strength (0-1) and acceptance (0-1). """
    logger.info("Phase: Connecting to physical form...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")

    try:
        # Use 0-1 factors for calculation
        earth_resonance = soul_spark.earth_resonance
        cord_integrity = soul_spark.cord_integrity
        mother_nurturing = mother_profile.get('nurturing_capacity', 0.5) if mother_profile else 0.5
        mother_spiritual = mother_profile.get('spiritual', {}).get('connection', 0.5) if mother_profile else 0.5
        mother_love = mother_profile.get('love_resonance', 0.5) if mother_profile else 0.5

        base_strength = (earth_resonance * BIRTH_CONN_WEIGHT_RESONANCE + cord_integrity * BIRTH_CONN_WEIGHT_INTEGRITY)
        base_strength *= (1.0 + mother_spiritual * BIRTH_CONN_MOTHER_STRENGTH_FACTOR)
        connection_factor = intensity * BIRTH_CONN_STRENGTH_FACTOR
        connection_strength = min(BIRTH_CONN_STRENGTH_CAP, max(0.0, base_strength * (1.0 + connection_factor))) # 0-1 score

        trauma_base = intensity * BIRTH_CONN_TRAUMA_FACTOR
        trauma_reduction = mother_nurturing * BIRTH_CONN_MOTHER_TRAUMA_REDUCTION
        trauma_level = max(0.0, min(1.0, trauma_base - trauma_reduction)) # 0-1 score

        acceptance_base = max(BIRTH_ACCEPTANCE_MIN, 1.0 - trauma_level * BIRTH_ACCEPTANCE_TRAUMA_FACTOR)
        acceptance_boost = mother_love * BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR
        acceptance = min(1.0, max(0.0, acceptance_base + acceptance_boost)) # 0-1 score

        phase_metrics = {
            "connection_strength_factor": float(connection_strength), "form_acceptance_factor": float(acceptance),
            "trauma_level_factor": float(trauma_level), "mother_influence_applied": mother_profile is not None,
            "timestamp": datetime.now().isoformat() }
        if METRICS_AVAILABLE: metrics.record_metrics('birth_connection', phase_metrics)
        logger.info(f"Physical form connection phase complete. ConnFactor: {connection_strength:.3f}, AcceptFactor: {acceptance:.3f}")
        return float(connection_strength), float(acceptance), phase_metrics

    except Exception as e: logger.error(f"Error connecting to physical form: {e}", exc_info=True); raise RuntimeError("Physical form connection failed.") from e

def _transfer_life_cord(soul_spark: SoulSpark, physical_connection: float, intensity: float, mother_profile: Optional[Dict]) -> Dict[str, Any]:
    """ Transfers life cord. Modifies cord_integrity (0-1), bandwidth (Hz). """
    logger.info("Phase: Transferring life cord...")
    if not (0.0 <= physical_connection <= 1.0): raise ValueError("physical_connection invalid.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity invalid.")
    if not isinstance(soul_spark.life_cord, dict): raise AttributeError("Missing 'life_cord' dict.")

    try:
        life_cord = soul_spark.life_cord; current_integrity = soul_spark.cord_integrity # 0-1 factor
        mother_healing = mother_profile.get('healing', {}).get('resonance', 0.5) if mother_profile else 0.5
        mother_protection = mother_profile.get('healing', {}).get('protection', 0.5) if mother_profile else 0.5

        base_efficiency_loss = intensity * BIRTH_CORD_TRANSFER_INTENSITY_FACTOR
        efficiency_boost = mother_healing * BIRTH_CORD_MOTHER_EFFICIENCY_FACTOR
        transfer_efficiency = current_integrity * (1.0 - base_efficiency_loss + efficiency_boost) # 0-1 factor
        transfer_efficiency = max(0.0, min(1.0, transfer_efficiency))

        base_integration = physical_connection * BIRTH_CORD_INTEGRATION_CONN_FACTOR
        integration_boost = mother_protection * BIRTH_CORD_MOTHER_INTEGRATION_FACTOR
        form_integration = min(1.0, max(0.0, base_integration + integration_boost)) # 0-1 factor

        new_integrity = current_integrity * transfer_efficiency # Update 0-1 factor
        original_bandwidth_hz = life_cord.get("bandwidth_hz", 100.0) # Get Hz
        new_bandwidth_hz = original_bandwidth_hz * transfer_efficiency # Scale Hz

        # Update SoulSpark
        life_cord["form_integration_factor"] = float(form_integration)
        life_cord["physical_anchored"] = True
        life_cord["bandwidth_hz"] = float(new_bandwidth_hz) # Store Hz
        setattr(soul_spark, "cord_integrity", float(new_integrity)) # Update 0-1 score
        timestamp = datetime.now().isoformat(); setattr(soul_spark, 'last_modified', timestamp)

        phase_metrics = {
            "transfer_efficiency_factor": transfer_efficiency, "form_integration_factor": form_integration,
            "new_integrity_factor": new_integrity, "new_bandwidth_hz": new_bandwidth_hz,
            "mother_influence_applied": mother_profile is not None, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('birth_cord_transfer', phase_metrics)
        logger.info(f"Life cord transferred. IntegrityFactor: {new_integrity:.3f}, BW: {new_bandwidth_hz:.1f} Hz")
        return phase_metrics

    except Exception as e: logger.error(f"Error transferring life cord: {e}", exc_info=True); raise RuntimeError("Life cord transfer failed.") from e

def _deploy_memory_veil(soul_spark: SoulSpark, intensity: float, mother_profile: Optional[Dict]) -> Dict[str, Any]:
    """ Deploys memory veil. Modifies memory_retention (0-1 factor). """
    logger.info("Phase: Deploying memory veil...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    try:
        mother_love = mother_profile.get('love_resonance', 0.5) if mother_profile else 0.5

        veil_strength = BIRTH_VEIL_STRENGTH_BASE + intensity * BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR # 0-1 factor
        veil_permanence = BIRTH_VEIL_PERMANENCE_BASE + intensity * BIRTH_VEIL_PERMANENCE_INTENSITY_FACTOR # 0-1 factor
        base_retention = BIRTH_VEIL_RETENTION_BASE + intensity * BIRTH_VEIL_RETENTION_INTENSITY_FACTOR # Base 0-1 factor
        retention_boost = mother_love * BIRTH_VEIL_MOTHER_RETENTION_FACTOR
        final_retention = max(BIRTH_VEIL_RETENTION_MIN, min(1.0, base_retention + retention_boost)) # Final 0-1 factor
        veil_strength = max(0.0, min(1.0, veil_strength)); veil_permanence = max(0.0, min(1.0, veil_permanence))

        memory_retentions = { mem_type: float(min(1.0, max(0.0, final_retention + mod))) for mem_type, mod in BIRTH_VEIL_MEMORY_RETENTION_MODS.items() }

        deployment_time = datetime.now().isoformat()
        veil_config = { "strength_factor": veil_strength, "permanence_factor": veil_permanence, "base_retention_factor": final_retention, "memory_retentions": memory_retentions, "deployment_time": deployment_time }

        setattr(soul_spark, "memory_veil", veil_config)
        setattr(soul_spark, "memory_retention", final_retention) # Store final 0-1 factor
        setattr(soul_spark, 'last_modified', deployment_time)

        phase_metrics = {**veil_config, "mother_influence_applied": mother_profile is not None, "timestamp": deployment_time}
        if METRICS_AVAILABLE: metrics.record_metrics('birth_memory_veil', phase_metrics)
        logger.info(f"Memory veil deployed. StrengthFactor: {veil_strength:.3f}, RetentionFactor: {final_retention:.3f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error deploying memory veil: {e}", exc_info=True); raise RuntimeError("Memory veil deployment failed.") from e

def _first_breath_integration(soul_spark: SoulSpark, physical_connection: float, intensity: float, mother_profile: Optional[Dict]) -> Dict[str, Any]:
    """ Integrates first breath. Modifies earth_resonance (0-1), energy split (SEU). """
    logger.info("Phase: Integrating first breath...")
    if not (0.0 <= physical_connection <= 1.0): raise ValueError("physical_connection invalid.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity invalid.")
    try:
        current_earth_resonance = soul_spark.earth_resonance # 0-1 factor
        total_energy_seu = soul_spark.energy # SEU

        mother_breath_freq = mother_profile.get('breath_pattern', {}).get('frequency', EARTH_BREATH_FREQUENCY) if mother_profile else EARTH_BREATH_FREQUENCY
        mother_nurturing = mother_profile.get('nurturing_capacity', 0.5) if mother_profile else 0.5
        mother_healing = mother_profile.get('healing', {}).get('resonance', 0.5) if mother_profile else 0.5

        breath_amplitude = max(0.0, min(1.0, BIRTH_BREATH_AMP_BASE + intensity * BIRTH_BREATH_AMP_INTENSITY_FACTOR)) # 0-1 factor
        breath_depth = max(0.0, min(1.0, BIRTH_BREATH_DEPTH_BASE + intensity * BIRTH_BREATH_DEPTH_INTENSITY_FACTOR)) # 0-1 factor
        base_sync = current_earth_resonance * BIRTH_BREATH_SYNC_RESONANCE_FACTOR
        sync_deviation = abs(mother_breath_freq - EARTH_BREATH_FREQUENCY) / max(FLOAT_EPSILON, EARTH_BREATH_FREQUENCY)
        mother_sync_factor = max(0.0, 1.0 - sync_deviation)
        breath_sync = base_sync * (1.0 + mother_sync_factor * BIRTH_BREATH_MOTHER_SYNC_FACTOR) # 0-1 factor
        breath_sync = max(0.0, min(1.0, breath_sync))
        integration_strength = max(0.0, min(1.0, physical_connection * BIRTH_BREATH_INTEGRATION_CONN_FACTOR)) # 0-1 factor

        # Earth Resonance boost (0-1 score)
        resonance_boost_factor = breath_sync * breath_depth * BIRTH_BREATH_RESONANCE_BOOST_FACTOR
        resonance_boost_factor *= (1.0 + mother_nurturing * BIRTH_BREATH_MOTHER_RESONANCE_BOOST)
        new_earth_resonance = min(1.0, current_earth_resonance + resonance_boost_factor)
        earth_resonance_gain = new_earth_resonance - current_earth_resonance

        # Energy Split (SEU)
        energy_shift_factor = breath_depth * integration_strength * BIRTH_BREATH_ENERGY_SHIFT_FACTOR
        energy_shift_factor *= (1.0 + mother_healing * BIRTH_BREATH_MOTHER_ENERGY_BOOST)
        # Physical energy is a fraction of the *total* available energy, scaled by factors
        physical_energy_fraction = BIRTH_BREATH_PHYSICAL_ENERGY_BASE + energy_shift_factor * BIRTH_BREATH_PHYSICAL_ENERGY_SCALE
        physical_energy_fraction = max(0.05, min(0.95, physical_energy_fraction)) # Clamp fraction 5%-95%
        new_physical_energy_seu = total_energy_seu * physical_energy_fraction
        # Spiritual energy is the remainder
        new_spiritual_energy_seu = total_energy_seu - new_physical_energy_seu
        # Ensure non-negative
        new_physical_energy_seu = max(0.0, new_physical_energy_seu)
        new_spiritual_energy_seu = max(0.0, new_spiritual_energy_seu)

        breath_time = datetime.now().isoformat()
        breath_config = {
            "frequency_hz": float(mother_breath_freq), "amplitude_factor": breath_amplitude, "depth_factor": breath_depth,
            "sync_factor": breath_sync, "integration_strength_factor": integration_strength,
             # Store resulting SEU values
            "physical_energy_seu": new_physical_energy_seu, "spiritual_energy_seu": new_spiritual_energy_seu,
            "timestamp": breath_time }

        # Update SoulSpark
        setattr(soul_spark, "breath_pattern", breath_config)
        setattr(soul_spark, "earth_resonance", new_earth_resonance) # Update 0-1 score
        setattr(soul_spark, "physical_energy", new_physical_energy_seu) # Update SEU
        setattr(soul_spark, "spiritual_energy", new_spiritual_energy_seu) # Update SEU
        setattr(soul_spark, 'last_modified', breath_time)

        phase_metrics = {
             "integration_strength_factor": integration_strength, "final_earth_resonance": new_earth_resonance,
             "earth_resonance_gain": earth_resonance_gain,
             "physical_energy_seu": new_physical_energy_seu, "spiritual_energy_seu": new_spiritual_energy_seu,
             "mother_influence_applied": mother_profile is not None, "timestamp": breath_time }
        if METRICS_AVAILABLE: metrics.record_metrics('birth_first_breath', phase_metrics)
        logger.info(f"First breath integrated. IntegrationFactor: {integration_strength:.3f}, PhysEnergy: {new_physical_energy_seu:.1f} SEU")
        return phase_metrics

    except Exception as e: logger.error(f"Error integrating first breath: {e}", exc_info=True); raise RuntimeError("First breath integration failed.") from e


def _finalize_birth(soul_spark: SoulSpark, connection_strength: float, form_acceptance: float, mother_profile: Optional[Dict]) -> Tuple[float, Dict[str, Any]]:
    """ Finalizes birth. Modifies frequency (Hz), stability (SU), physical_integration (0-1). """
    logger.info("Phase: Finalizing birth process...")
    if not (0.0 <= connection_strength <= 1.0): raise ValueError("connection_strength invalid.")
    if not (0.0 <= form_acceptance <= 1.0): raise ValueError("form_acceptance invalid.")
    try:
        # Use factors for calculation
        breath_integration = getattr(soul_spark, 'breath_pattern', {}).get('integration_strength_factor', 0.7)
        mother_spirit_conn = mother_profile.get('spiritual', {}).get('connection', 0.5) if mother_profile else 0.5

        # Total physical integration factor (0-1)
        base_integration = (connection_strength * BIRTH_FINAL_INTEGRATION_WEIGHT_CONN + form_acceptance * BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT + breath_integration * BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH)
        integration_boost = mother_spirit_conn * BIRTH_FINAL_MOTHER_INTEGRATION_BOOST
        total_integration_factor = min(1.0, max(0.0, base_integration + integration_boost)) # Final 0-1 score

        # Adjust core soul properties: Frequency (Hz) and Stability (SU)
        original_frequency_hz = soul_spark.frequency
        original_stability_su = soul_spark.stability

        # Frequency adjusts based on integration (e.g., lower frequency in physical)
        physical_frequency_hz = original_frequency_hz * (1.0 - (1.0 - BIRTH_FINAL_FREQ_FACTOR) * total_integration_factor)
        physical_frequency_hz = max(FLOAT_EPSILON * 10, physical_frequency_hz) # Ensure positive Hz

        # Stability adjusts based on integration (e.g., slight reduction due to physical constraints)
        stability_adjustment_factor = (1.0 - (1.0 - BIRTH_FINAL_STABILITY_FACTOR) * total_integration_factor)
        physical_stability_su = original_stability_su * stability_adjustment_factor
        physical_stability_su = max(0.0, min(MAX_STABILITY_SU, physical_stability_su)) # Clamp SU
        stability_change_su = physical_stability_su - original_stability_su

        # Update SoulSpark
        setattr(soul_spark, "frequency", float(physical_frequency_hz))
        setattr(soul_spark, "stability", float(physical_stability_su))
        setattr(soul_spark, "physical_integration", float(total_integration_factor)) # Store 0-1 score
        setattr(soul_spark, FLAG_INCARNATED, True)
        birth_time = datetime.now().isoformat()
        setattr(soul_spark, "birth_time", birth_time)
        setattr(soul_spark, 'last_modified', birth_time)
        if hasattr(soul_spark, 'update_state'): soul_spark.update_state() # Update derived scores

        phase_metrics = {
            "final_integration_factor": total_integration_factor,
            "final_frequency_hz": physical_frequency_hz, "frequency_change_hz": physical_frequency_hz - original_frequency_hz,
            "final_stability_su": physical_stability_su, "stability_change_su": stability_change_su,
            "birth_timestamp": birth_time, "mother_influence_applied": mother_profile is not None, "success": True }
        if METRICS_AVAILABLE: metrics.record_metrics('birth_finalization', phase_metrics)
        logger.info(f"Birth finalized. IntegrationFactor: {total_integration_factor:.3f}, Final Freq: {physical_frequency_hz:.1f} Hz, Final Stab: {physical_stability_su:.1f} SU")
        return float(total_integration_factor), phase_metrics

    except Exception as e: logger.error(f"Error finalizing birth: {e}", exc_info=True); raise RuntimeError("Birth finalization failed.") from e


# --- Orchestration Function ---
def perform_birth(soul_spark: SoulSpark, intensity: float = BIRTH_INTENSITY_DEFAULT,
                 mother_profile: Optional[Dict[str, Any]] = None,
                 use_encoded_glyph: bool = True # Keep flag for external logic/logging
                 ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs complete birth process using SU/SEU units where applicable. Modifies SoulSpark. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    active_mother_profile = mother_profile if use_encoded_glyph and mother_profile else None
    log_msg_suffix = "with Mother Profile" if active_mother_profile else "without Mother Profile"
    logger.info(f"--- Starting Birth Process for Soul {spark_id} (Int={intensity:.2f}) {log_msg_suffix} ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}

    try:
        _ensure_soul_properties(soul_spark)
        if not _check_prerequisites(soul_spark): # Uses 0-1 factors
            raise ValueError("Soul prerequisites for birth not met.")

        # Store Initial State (SEU/SU/Factors)
        initial_state = {
            'stability_su': soul_spark.stability, 'frequency_hz': soul_spark.frequency,
            'earth_resonance': soul_spark.earth_resonance, 'cord_integrity': soul_spark.cord_integrity,
            'energy_seu': soul_spark.energy
        }

        # Run Phases
        connection_strength, form_acceptance, metrics1 = _connect_to_physical_form(soul_spark, intensity, active_mother_profile)
        process_metrics_summary['steps']['connection'] = metrics1
        metrics2 = _transfer_life_cord(soul_spark, connection_strength, intensity, active_mother_profile)
        process_metrics_summary['steps']['cord_transfer'] = metrics2
        metrics3 = _deploy_memory_veil(soul_spark, intensity, active_mother_profile)
        process_metrics_summary['steps']['memory_veil'] = metrics3
        metrics4 = _first_breath_integration(soul_spark, connection_strength, intensity, active_mother_profile)
        process_metrics_summary['steps']['first_breath'] = metrics4
        final_integration, metrics5 = _finalize_birth(soul_spark, connection_strength, form_acceptance, active_mother_profile)
        process_metrics_summary['steps']['finalization'] = metrics5
        birth_successful = metrics5.get("success", False)

        # Compile Overall Metrics
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Report state in correct units/scores
             'stability_su': soul_spark.stability, 'frequency_hz': soul_spark.frequency,
             'earth_resonance': soul_spark.earth_resonance, 'cord_integrity': soul_spark.cord_integrity,
             'physical_integration': soul_spark.physical_integration,
             'physical_energy_seu': soul_spark.physical_energy,
             'spiritual_energy_seu': soul_spark.spiritual_energy,
             FLAG_INCARNATED: getattr(soul_spark, FLAG_INCARNATED) }
        overall_metrics = {
            'action': 'birth', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(), 'intensity_setting': intensity,
            'mother_influence_active': active_mother_profile is not None,
            'initial_state': initial_state, 'final_state': final_state,
            'final_integration_factor': final_integration, 'success': birth_successful, }
        if METRICS_AVAILABLE: metrics.record_metrics('birth_summary', overall_metrics)

        if birth_successful:
             logger.info(f"--- Birth Process Completed Successfully for Soul {spark_id} ---")
             if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Incarnated successfully. Integration: {final_integration:.3f}")
        else: logger.error(f"--- Birth Process Completed BUT FAILED for Soul {spark_id} ---") # Should not happen if logic holds

        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Birth process failed for {spark_id} due to validation error: {e_val}", exc_info=True)
         failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'prerequisites'
         record_birth_failure(spark_id, start_time_iso, failed_step, str(e_val), active_mother_profile is not None)
         setattr(soul_spark, FLAG_INCARNATED, False)
         raise
    except RuntimeError as e_rt:
         logger.critical(f"Birth process failed critically for {spark_id}: {e_rt}", exc_info=True)
         failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'runtime'
         record_birth_failure(spark_id, start_time_iso, failed_step, str(e_rt), active_mother_profile is not None)
         setattr(soul_spark, FLAG_INCARNATED, False)
         raise
    except Exception as e:
         logger.critical(f"Unexpected error during birth process for {spark_id}: {e}", exc_info=True)
         failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'unexpected'
         record_birth_failure(spark_id, start_time_iso, failed_step, str(e), active_mother_profile is not None)
         setattr(soul_spark, FLAG_INCARNATED, False)
         raise RuntimeError(f"Unexpected birth process failure: {e}") from e

# --- Failure Metric Helper ---
def record_birth_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str, mother_active: bool):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            metrics.record_metrics('birth_summary', {
                'action': 'birth', 'soul_id': spark_id, 'start_time': start_time_iso,
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(start_time_iso)).total_seconds(),
                'mother_influence_active': mother_active,
                'success': False, 'error': error_msg, 'failed_step': failed_step })
        except Exception as metric_e:
            logger.error(f"Failed to record birth failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/birth.py ---
