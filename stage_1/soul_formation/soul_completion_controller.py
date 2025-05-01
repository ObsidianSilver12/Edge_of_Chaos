# --- START OF FILE src/stage_1/soul_completion_controller.py ---

"""
Soul Completion Controller (Refactored V4.1)

Orchestrates the final stages of soul formation after Creator Entanglement.
Operates on SoulSpark with SEU/SU/CU units and calls refactored stage functions.
"""

import logging
import os
import sys
import uuid
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from constants.constants import *
except ImportError as e:
    # Minimal fallbacks if constants fail at runtime
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}.")
    DATA_DIR_BASE = "output" # Need at least this
    # Default stage parameters (won't be used if called correctly, but prevent NameError)
    HARMONIC_STRENGTHENING_INTENSITY_DEFAULT=0.7; HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT=1.0
    LIFE_CORD_COMPLEXITY_DEFAULT=0.7; EARTH_HARMONY_INTENSITY_DEFAULT=0.7; EARTH_HARMONY_DURATION_FACTOR_DEFAULT=1.0
    BIRTH_INTENSITY_DEFAULT=0.7; IDENTITY_CRYSTALLIZATION_THRESHOLD = 0.85


# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening
    from stage_1.soul_formation.life_cord import form_life_cord
    from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization
    from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization
    from stage_1.soul_formation.birth import perform_birth
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import stage modules/SoulSpark/metrics: {e}", exc_info=True)
    raise ImportError(f"Core stage dependencies missing: {e}") from e


# Controller-specific metric category
CONTROLLER_METRIC_CATEGORY = "soul_completion_controller"

class SoulCompletionController:
    """
    Orchestrates Harmonic Strengthening -> Life Cord -> Earth Harmonization -> Identity -> Birth.
    """

    def __init__(self, data_dir: str = DATA_DIR_BASE, controller_id: Optional[str] = None):
        """ Initialize the Soul Completion Controller. """
        if not data_dir or not isinstance(data_dir, str): raise ValueError("Data directory invalid.")

        self.controller_id: str = controller_id or str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.output_dir: str = os.path.join(data_dir, "controller_data", f"soul_completion_{self.controller_id}")

        try: os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e: logger.critical(f"CRITICAL: Failed to create output dir {self.output_dir}: {e}"); raise

        self.active_souls: Dict[str, Dict[str, Any]] = {} # {soul_id: status_dict}
        logger.info(f"Initializing Soul Completion Controller (ID: {self.controller_id})")
        if METRICS_AVAILABLE:
             metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                 'status': 'initialized', 'controller_id': self.controller_id,
                 'timestamp': self.creation_time, })
        logger.info(f"Soul Completion Controller '{self.controller_id}' initialized.")

    def run_soul_completion(self, soul_spark: SoulSpark, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        """
        Runs the complete sequence of post-entanglement soul formation stages.
        Modifies SoulSpark (with SEU/SU/CU) in place. Fails hard.

        Args:
            soul_spark (SoulSpark): The SoulSpark object ready for completion stages.
            **kwargs: Optional parameters to override defaults for specific stages.

        Returns: Tuple[SoulSpark, Dict[str, Any]]: Modified SoulSpark, overall metrics.
        Raises: TypeError, ValueError, RuntimeError.
        """
        if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        if spark_id in self.active_souls and self.active_souls[spark_id]['status'] == 'processing':
             raise RuntimeError(f"Soul {spark_id} is already being processed.")

        logger.info(f"--- Starting Soul Completion Process for Soul {spark_id} ---")
        start_time_iso = datetime.now().isoformat()
        start_time_dt = datetime.fromisoformat(start_time_iso)
        completion_summary = {'soul_id': spark_id, 'stages': {}}
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': None, 'start_time': start_time_iso}

        try:
            # --- Stage 1: Harmonic Strengthening ---
            stage_name = FLAG_HARMONICALLY_STRENGTHENED.replace('_', ' ').title()
            self.active_souls[spark_id]['current_stage'] = stage_name
            logger.info(f"Stage: {stage_name} for {spark_id}...")
            harmony_intensity = kwargs.get('harmony_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT)
            harmony_duration = kwargs.get('harmony_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT)
            _, metrics1 = perform_harmonic_strengthening(soul_spark, intensity=harmony_intensity, duration_factor=harmony_duration)
            completion_summary['stages'][stage_name] = metrics1 # Store full stage metrics
            logger.info(f"{stage_name} Complete. Stability: {soul_spark.stability:.1f} SU, Coherence: {soul_spark.coherence:.1f} CU")

            # --- Stage 2: Life Cord Formation ---
            stage_name = FLAG_CORD_FORMATION_COMPLETE.replace('_', ' ').title()
            self.active_souls[spark_id]['current_stage'] = stage_name
            logger.info(f"Stage: {stage_name} for {spark_id}...")
            cord_complexity = kwargs.get('cord_complexity', LIFE_CORD_COMPLEXITY_DEFAULT)
            _, metrics2 = form_life_cord(soul_spark, complexity=cord_complexity)
            completion_summary['stages'][stage_name] = metrics2
            logger.info(f"{stage_name} Complete. Cord Integrity: {soul_spark.cord_integrity:.3f}")

            # --- Stage 3: Earth Harmonization ---
            stage_name = FLAG_EARTH_HARMONIZED.replace('_', ' ').title()
            self.active_souls[spark_id]['current_stage'] = stage_name
            logger.info(f"Stage: {stage_name} for {spark_id}...")
            earth_intensity = kwargs.get('earth_intensity', EARTH_HARMONY_INTENSITY_DEFAULT)
            earth_duration = kwargs.get('earth_duration_factor', EARTH_HARMONY_DURATION_FACTOR_DEFAULT)
            _, metrics3 = perform_earth_harmonization(soul_spark, intensity=earth_intensity, duration_factor=earth_duration)
            completion_summary['stages'][stage_name] = metrics3
            logger.info(f"{stage_name} Complete. Earth Resonance: {soul_spark.earth_resonance:.3f}")

            # --- Stage 4: Identity Crystallization ---
            stage_name = FLAG_IDENTITY_CRYSTALLIZED.replace('_', ' ').title()
            self.active_souls[spark_id]['current_stage'] = stage_name
            logger.info(f"Stage: {stage_name} for {spark_id}...")
            id_kwargs = {k:v for k,v in kwargs.items() if k in ['specified_name', 'train_cycles', 'entrainment_bpm', 'entrainment_duration', 'love_cycles', 'geometry_stages', 'crystallization_threshold']}
            # life_cord_data removed from perform_identity_crystallization signature
            _, metrics4 = perform_identity_crystallization(soul_spark, **id_kwargs)
            completion_summary['stages'][stage_name] = metrics4
            logger.info(f"{stage_name} Complete. Name: {soul_spark.name}, Cryst Level: {soul_spark.crystallization_level:.3f}")

            # --- Stage 5: Birth ---
            stage_name = "Birth"
            self.active_souls[spark_id]['current_stage'] = stage_name
            logger.info(f"Stage: {stage_name} for {spark_id}...")
            birth_intensity = kwargs.get('birth_intensity', BIRTH_INTENSITY_DEFAULT)
            # Check if mother glyph is available (constant defined in root or here)
            try: from constants.constants import MOTHER_GLYPH_AVAILABLE as BIRTH_MOTHER_GLYPH_CHECK
            except ImportError: BIRTH_MOTHER_GLYPH_CHECK = False # Fallback if constant missing
            _, metrics5 = perform_birth(
                 soul_spark, intensity=birth_intensity,
                 mother_profile=kwargs.get('mother_profile'), # Pass profile if provided in kwargs
                 use_encoded_glyph=BIRTH_MOTHER_GLYPH_CHECK
            )
            completion_summary['stages'][stage_name] = metrics5
            logger.info(f"{stage_name} Complete.")

            # --- Finalization ---
            end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
            completion_summary['start_time'] = start_time_iso
            completion_summary['end_time'] = end_time_iso
            completion_summary['duration_seconds'] = (end_time_dt - start_time_dt).total_seconds()
            completion_summary['success'] = True
            completion_summary['final_soul_state_summary'] = soul_spark.get_spark_metrics()['core'] # Store core metrics summary

            self.active_souls[spark_id]['status'] = 'completed'
            self.active_souls[spark_id]['end_time'] = end_time_iso
            self.active_souls[spark_id]['current_stage'] = None
            self.active_souls[spark_id]['summary'] = {k: v for k, v in completion_summary.items() if k != 'stages'} # Store summary without detailed stage metrics

            self._save_completed_soul(soul_spark) # Save final state

            # Record overall metrics for this controller's run
            overall_controller_metrics = {
                 'controller_run': 'soul_completion', 'soul_id': spark_id,
                 'start_time': start_time_iso, 'end_time': end_time_iso,
                 'duration_seconds': completion_summary['duration_seconds'],
                 'success': True,
                 'final_energy_seu': soul_spark.energy,
                 'final_stability_su': soul_spark.stability,
                 'final_coherence_cu': soul_spark.coherence,
                 'final_incarnated_status': getattr(soul_spark, FLAG_INCARNATED, False)
            }
            if METRICS_AVAILABLE: metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, overall_controller_metrics)

            logger.info(f"--- Soul Completion Process Finished Successfully for Soul {spark_id} ---")
            logger.info(f"Duration: {completion_summary['duration_seconds']:.2f}s")
            logger.info(f"Final Incarnated Status: {getattr(soul_spark, FLAG_INCARNATED, False)}")

            # Return the completed soul and the summary (excluding detailed stage metrics for brevity?)
            return soul_spark, overall_controller_metrics # Return controller summary

        except Exception as e:
            # ... (Error handling logic unchanged, records failure metric) ...
            end_time_iso = datetime.now().isoformat()
            failed_stage = self.active_souls[spark_id].get('current_stage', 'unknown')
            logger.critical(f"Soul completion failed for {spark_id} at stage '{failed_stage}': {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['end_time'] = end_time_iso
            self.active_souls[spark_id]['error'] = str(e)
            setattr(soul_spark, FLAG_INCARNATED, False) # Ensure not marked incarnated on failure
            # Record failure
            if METRICS_AVAILABLE:
                 metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                      'controller_run': 'soul_completion', 'soul_id': spark_id,
                      'start_time': start_time_iso, 'end_time': end_time_iso,
                      'duration_seconds': (datetime.fromisoformat(end_time_iso) - start_time_dt).total_seconds(),
                      'success': False, 'error': str(e), 'failed_stage': failed_stage })
            raise RuntimeError(f"Soul completion process failed at stage '{failed_stage}'.") from e


    def _save_completed_soul(self, soul_spark: SoulSpark) -> bool:
        """Saves the final state of the completed soul."""
        # ... (implementation unchanged) ...
        spark_id = getattr(soul_spark, 'spark_id', None)
        if not spark_id: return False
        filename = f"soul_completed_{spark_id}.json"
        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving completed soul data for {spark_id} to {save_path}...")
        try:
            if hasattr(soul_spark, 'save_spark_data'): return soul_spark.save_spark_data(save_path)
            else: logger.error("SoulSpark object missing save_spark_data method."); return False
        except Exception as e: logger.error(f"Error saving completed soul {spark_id}: {e}", exc_info=True); return False


    def save_controller_state(self, filename: Optional[str] = None) -> bool:
        """ Save the controller's current state (tracking active souls). """
        # ... (implementation unchanged) ...
        if filename is None: filename = f"soul_completion_controller_state_{self.controller_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        elif not filename.lower().endswith('.json'): filename += '.json'
        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving Soul Completion controller state to {save_path}...")
        try:
            controller_metrics = {}
            if METRICS_AVAILABLE: controller_metrics = metrics.get_category_metrics(CONTROLLER_METRIC_CATEGORY)
            state_data = {
                'controller_id': self.controller_id, 'creation_time': self.creation_time,
                'output_dir': self.output_dir,
                'active_souls_status': {sid: data.get('status') for sid, data in self.active_souls.items()},
                'last_metrics_snapshot': controller_metrics, 'save_timestamp': datetime.now().isoformat() }
            with open(save_path, 'w') as f: json.dump(state_data, f, indent=2, default=str)
            logger.info("Soul Completion controller state saved successfully.")
            return True
        except IOError as e: logger.error(f"IOError saving controller state: {e}", exc_info=True); raise IOError(f"Failed write state: {e}") from e
        except Exception as e: logger.error(f"Unexpected error saving state: {e}", exc_info=True); raise RuntimeError(f"Failed save state: {e}") from e

    # --- __str__, __repr__ (Unchanged) ---
    def __str__(self) -> str:
        processing = sum(1 for d in self.active_souls.values() if d.get('status')=='processing')
        completed = sum(1 for d in self.active_souls.values() if d.get('status')=='completed')
        failed = sum(1 for d in self.active_souls.values() if d.get('status')=='failed')
        return f"SoulCompletionController(ID: {self.controller_id}, Processing: {processing}, Completed: {completed}, Failed: {failed})"
    def __repr__(self) -> str: return f"<SoulCompletionController id='{self.controller_id}' active_souls={len(self.active_souls)}>"

# --- Example Usage (logic unchanged, but output interpretation differs) ---
# Keep __main__ block for standalone testing if needed

# --- END OF FILE soul_completion_controller.py ---
