# --- START OF FILE soul_completion_controller.py ---

"""
Soul Completion Controller

Orchestrates the final stages of soul formation after the Sephiroth journey,
including harmonic strengthening, life cord formation, Earth harmonization,
identity crystallization, and the birth process.

Author: Soul Development Framework Team
"""
from src.stage_1.soul_formation.soul_visualization_enhanced import EnhancedSoulVisualizer
import logging
import os
import sys
import uuid
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
# Add this with your other logging constants
import logging
LOG_LEVEL = logging.INFO  # Default logging level


# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    # Import necessary constants (mostly defaults for stages if not overridden)
    from src.constants.constants import *

except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. SoulCompletionController cannot function correctly.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
# Import the primary functions from the refactored stage modules
# Also import SoulSpark definition
try:
    from src.stage_1.soul_formation.soul_spark import SoulSpark
    from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening
    from stage_1.soul_formation.life_cord import form_life_cord
    from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization
    from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization
    from stage_1.soul_formation.birth import perform_birth
    # Import metrics tracking
    import metrics_tracking as metrics
    DEPENDENCIES_AVAILABLE = True
    METRICS_AVAILABLE = True
    if metrics is None: raise ImportError("Metrics tracking module failed load.") # Check explicit None if metrics fails init
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import stage modules or SoulSpark/metrics: {e}")
    METRICS_AVAILABLE = False
    raise ImportError(f"Core stage dependencies missing: {e}") from e

# Conditional import for visualization (only if controller invokes it)
try:
    import matplotlib.pyplot as plt
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    logger.warning("Matplotlib not found. Visualization methods within controller will be disabled.")


# Configure logging (if not done globally)
# log_file_path = os.path.join("logs", "soul_completion_controller.log")
# os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
# logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')

# Define controller-specific metric category
CONTROLLER_METRIC_CATEGORY = "soul_completion_controller"

class SoulCompletionController:
    """
    Orchestrates the final stages of soul formation:
    Harmonic Strengthening -> Life Cord -> Earth Harmonization -> Identity -> Birth.
    """

    def __init__(self, data_dir: str = DATA_DIR_BASE, controller_id: Optional[str] = None):
        """
        Initialize the Soul Completion Controller.

        Args:
            data_dir (str): Base directory for saving controller state or logs.
            controller_id (Optional[str]): Specific ID for the controller.
        """
        if not data_dir or not isinstance(data_dir, str):
            raise ValueError("Data directory must be a non-empty string.")

        self.controller_id: str = controller_id or str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        # Controller's specific output dir
        self.output_dir: str = os.path.join(data_dir, "controller_data", f"soul_completion_{self.controller_id}")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create controller output directory {self.output_dir}: {e}")
            raise

        # Track souls currently being processed by this controller {soul_id: status_dict}
        self.active_souls: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initializing Soul Completion Controller (ID: {self.controller_id})")

        # Record initial controller state metric
        metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
            'status': 'initialized',
            'controller_id': self.controller_id,
            'timestamp': self.creation_time,
        })
        logger.info(f"Soul Completion Controller '{self.controller_id}' initialized successfully.")

    def run_soul_completion(self, soul_spark: SoulSpark, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        """
        Runs the complete sequence of post-Sephiroth soul formation stages.
        Modifies the SoulSpark object in place. Fails hard on critical errors.

        Args:
            soul_spark (SoulSpark): The SoulSpark object that has completed the Sephiroth journey.
            **kwargs: Optional parameters to override defaults for specific stages, e.g.,
                      harmony_intensity=0.8, cord_complexity=0.75, birth_intensity=0.6

        Returns:
            Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
                - The modified (incarnated) SoulSpark object.
                - overall_metrics (Dict): Summary metrics for the entire completion process.

        Raises:
            TypeError: If soul_spark is not a SoulSpark instance.
            ValueError: If prerequisites for starting completion are not met or kwargs invalid.
            RuntimeError: If any stage fails critically.
        """
        if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
        spark_id = getattr(soul_spark, 'spark_id', 'unknown')
        if spark_id == 'unknown': raise ValueError("SoulSpark missing valid spark_id.")
        if spark_id in self.active_souls and self.active_souls[spark_id]['status'] == 'processing':
             raise RuntimeError(f"Soul {spark_id} is already being processed by this controller.")

        logger.info(f"--- Starting Soul Completion Process for Soul {spark_id} ---")
        start_time_iso = datetime.now().isoformat()
        start_time_dt = datetime.fromisoformat(start_time_iso)
        completion_summary = {'soul_id': spark_id, 'stages': {}}
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': None, 'start_time': start_time_iso}

        try:
            # --- Stage 1: Harmonic Strengthening ---
            self.active_souls[spark_id]['current_stage'] = 'harmonic_strengthening'
            logger.info(f"Stage 1: Harmonic Strengthening for {spark_id}...")
            harmony_intensity = kwargs.get('harmony_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT)
            harmony_duration = kwargs.get('harmony_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT)
            _, metrics1 = perform_harmonic_strengthening(soul_spark, intensity=harmony_intensity, duration_factor=harmony_duration)
            completion_summary['stages']['harmonic_strengthening'] = metrics1['final_state'] # Store final state metrics
            logger.info("Stage 1 Complete.")

            # --- Stage 2: Life Cord Formation ---
            self.active_souls[spark_id]['current_stage'] = 'life_cord'
            logger.info(f"Stage 2: Life Cord Formation for {spark_id}...")
            cord_complexity = kwargs.get('cord_complexity', LIFE_CORD_COMPLEXITY_DEFAULT)
            _, metrics2 = form_life_cord(soul_spark, complexity=cord_complexity)
            completion_summary['stages']['life_cord'] = metrics2['final_state']
            logger.info("Stage 2 Complete.")

            # --- Stage 3: Earth Harmonization ---
            self.active_souls[spark_id]['current_stage'] = 'earth_harmonization'
            logger.info(f"Stage 3: Earth Harmonization for {spark_id}...")
            earth_intensity = kwargs.get('earth_intensity', EARTH_HARMONY_INTENSITY_DEFAULT)
            earth_duration = kwargs.get('earth_duration_factor', EARTH_HARMONY_DURATION_FACTOR_DEFAULT)
            # Life cord data is now attached to soul_spark, no need to pass separately
            _, metrics3 = perform_earth_harmonization(soul_spark, intensity=earth_intensity, duration_factor=earth_duration)
            completion_summary['stages']['earth_harmonization'] = metrics3['final_state']
            logger.info("Stage 3 Complete.")

            # --- Stage 4: Identity Crystallization ---
            self.active_souls[spark_id]['current_stage'] = 'identity_crystallization'
            logger.info(f"Stage 4: Identity Crystallization for {spark_id}...")
            # Pass relevant kwargs if customization is needed
            id_kwargs = {k:v for k,v in kwargs.items() if k in ['specified_name', 'train_cycles', 'entrainment_bpm', 'entrainment_duration', 'love_cycles', 'geometry_stages', 'crystallization_threshold']}
            life_cord_data = getattr(soul_spark, 'life_cord', {}) # Get cord data from soul
            _, metrics4 = perform_identity_crystallization(soul_spark, life_cord_data, **id_kwargs)
            completion_summary['stages']['identity_crystallization'] = metrics4['final_state']
            logger.info("Stage 4 Complete.")

            # --- Stage 5: Birth ---
            self.active_souls[spark_id]['current_stage'] = 'birth'
            logger.info(f"Stage 5: Birth Process for {spark_id}...")
            birth_intensity = kwargs.get('birth_intensity', BIRTH_INTENSITY_DEFAULT)
            # Pass relevant data (soul, cord data from soul, identity data from soul)
            soul_identity_metrics = getattr(soul_spark, 'identity_metrics', {}) # Get metrics stored by ID stage
            life_cord_data = getattr(soul_spark, 'life_cord', {})
            _, metrics5 = perform_birth(soul_spark, life_cord_data, soul_identity_metrics, intensity=birth_intensity)
            completion_summary['stages']['birth'] = metrics5['final_state']
            logger.info("Stage 5 Complete.")


            # --- Finalization ---
            end_time_iso = datetime.now().isoformat()
            end_time_dt = datetime.fromisoformat(end_time_iso)
            completion_summary['start_time'] = start_time_iso
            completion_summary['end_time'] = end_time_iso
            completion_summary['duration_seconds'] = (end_time_dt - start_time_dt).total_seconds()
            completion_summary['success'] = True
            completion_summary['final_soul_state_summary'] = soul_spark.get_spark_metrics() # Get final comprehensive metrics

            self.active_souls[spark_id]['status'] = 'completed'
            self.active_souls[spark_id]['end_time'] = end_time_iso
            self.active_souls[spark_id]['current_stage'] = None
            self.active_souls[spark_id]['summary'] = completion_summary

            # Save final soul state
            self._save_completed_soul(soul_spark)

            # Generate visualizations
            visualizer = EnhancedSoulVisualizer(soul_spark, output_dir="completed_souls")
            # Generate visualizations without storing unused return value
            visualizer.generate_all_visualizations(show=True, save=True)

            # Record overall metrics
            try:
                metrics.record_metrics('soul_completion_summary', completion_summary)
            except Exception as e:
                logger.error(f"Failed to record summary metrics for soul completion: {e}")

            logger.info(f"--- Soul Completion Process Finished Successfully for Soul {spark_id} ---")
            logger.info(f"Duration: {completion_summary['duration_seconds']:.2f}s")
            logger.info(f"Final Incarnated Status: {getattr(soul_spark, 'incarnated', False)}")

            # Return the completed soul and summary
            return soul_spark, completion_summary

        except Exception as e:
            end_time_iso = datetime.now().isoformat()
            failed_stage = self.active_souls[spark_id].get('current_stage', 'unknown')
            logger.critical(f"Soul completion process failed critically for soul {spark_id} at stage '{failed_stage}': {e}", exc_info=True)

            self.active_souls[spark_id]['status'] = 'failed'
            self.active_souls[spark_id]['end_time'] = end_time_iso
            self.active_souls[spark_id]['error'] = str(e)

            # Mark soul as failed this stage?
            setattr(soul_spark, "incarnated", False) # Ensure not marked as incarnated

            if METRICS_AVAILABLE:
                try: metrics.record_metrics('soul_completion_summary', {
                        'action': 'soul_completion', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                        'duration_seconds': (datetime.fromisoformat(end_time_iso) - start_time_dt).total_seconds(),
                        'success': False, 'error': str(e), 'failed_stage': failed_step
                    })
                except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
            raise RuntimeError(f"Soul completion process failed at stage '{failed_step}'.") from e


    def _save_completed_soul(self, soul_spark: SoulSpark) -> bool:
        """Saves the final state of the completed soul."""
        spark_id = getattr(soul_spark, 'spark_id', 'unknown')
        if spark_id == 'unknown': return False

        filename = f"soul_completed_{spark_id}.json"
        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving completed soul data for {spark_id} to {save_path}...")
        try:
            # Use the soul's own save method
            if hasattr(soul_spark, 'save_spark_data'):
                return soul_spark.save_spark_data(save_path)
            else:
                logger.error("SoulSpark object does not have a save_spark_data method.")
                return False
        except Exception as e:
            logger.error(f"Error saving completed soul data: {e}", exc_info=True)
            return False


    def save_controller_state(self, filename: Optional[str] = None) -> bool:
        """
        Save the controller's current state (tracking active souls). Fails hard on error.

        Args:
            filename (Optional[str]): Filename for the state file (JSON). Auto-generated if None.

        Returns:
            bool: True if saving was successful.

        Raises:
            IOError: If saving the state file fails.
            RuntimeError: For other unexpected errors.
        """
        if filename is None:
            filename = f"soul_completion_controller_state_{self.controller_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        elif not filename.lower().endswith('.json'):
             filename += '.json'

        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving Soul Completion controller state to {save_path}...")

        try:
            controller_metrics = metrics.get_category_metrics(CONTROLLER_METRIC_CATEGORY) # Get current controller metrics

            state_data = {
                'controller_id': self.controller_id,
                'creation_time': self.creation_time,
                'output_dir': self.output_dir,
                'active_souls_status': {sid: data.get('status') for sid, data in self.active_souls.items()}, # Save only status
                'last_metrics_snapshot': controller_metrics,
                'save_timestamp': datetime.now().isoformat()
            }

            with open(save_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.info("Soul Completion controller state saved successfully.")
            return True
        except IOError as e:
            logger.error(f"IOError saving controller state to {save_path}: {e}", exc_info=True)
            raise IOError(f"Failed to write controller state file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error saving controller state: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save controller state: {e}") from e

    def __str__(self) -> str:
        """String representation of the Soul Completion Controller."""
        processing = sum(1 for data in self.active_souls.values() if data.get('status') == 'processing')
        completed = sum(1 for data in self.active_souls.values() if data.get('status') == 'completed')
        failed = sum(1 for data in self.active_souls.values() if data.get('status') == 'failed')
        return (f"SoulCompletionController(ID: {self.controller_id}, "
                f"Processing: {processing}, Completed: {completed}, Failed: {failed})")

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<SoulCompletionController id='{self.controller_id}' active_souls={len(self.active_souls)}>"


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Soul Completion Controller Module Example...")
    soul_completion_controller = None
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        try:
            # --- Setup: Create a soul that has finished Sephiroth Journey ---
            # This would typically come from SephirothController output
            soul_after_journey = SoulSpark()
            soul_after_journey.spark_id = "journey_complete_soul_01"
            # Set attributes as if journey completed successfully
            soul_after_journey.stability = 0.85
            soul_after_journey.coherence = 0.88
            soul_after_journey.frequency = 432.0
            soul_after_journey.formation_complete = True
            soul_after_journey.sephiroth_journey_complete = True # Flag from Sephiroth stage
            soul_after_journey.aspects = {'unity': {'strength': 0.9}, 'love': {'strength': 0.8}, 'wisdom': {'strength': 0.75}, 'harmony': {'strength': 0.8}}
            soul_after_journey.creator_alignment = 0.9
            soul_after_journey.memory_echoes = ["Journey through Tiphareth completed."] # Example memory
            soul_after_journey.last_modified = datetime.now().isoformat()
            # Ensure necessary attributes for Harmonization prerequisites
            soul_after_journey.harmonically_strengthened = False # Will be set by the process
            soul_after_journey.cord_formation_complete = False
            soul_after_journey.earth_harmonized = False
            soul_after_journey.identity_crystallized = False
            soul_after_journey.incarnated = False
            soul_after_journey.ready_for_earth = False # Will be set by Life Cord
            soul_after_journey.ready_for_birth = False # Will be set by Earth Harmony

            print(f"\nInitial Soul State (Post-Journey - {soul_after_journey.spark_id}):")
            print(f"  Stability: {soul_after_journey.stability:.4f}")
            print(f"  Coherence: {soul_after_journey.coherence:.4f}")
            print(f"  Frequency: {soul_after_journey.frequency:.4f}")
            print(f"  Aspects Count: {len(soul_after_journey.aspects)}")

            # --- Initialize Controller ---
            soul_completion_controller = SoulCompletionController(
                data_dir="output/soul_completion_example"
            )
            print(soul_completion_controller)

            # --- Run Completion Process ---
            print("\n--- Running Full Soul Completion Process ---")
            final_soul_object, summary_metrics_result = soul_completion_controller.run_soul_completion(
                soul_spark = soul_after_journey, # Pass the prepared soul object
                harmony_intensity=0.7,
                cord_complexity=0.6,
                birth_intensity=0.7
                # Add other kwargs to override stage defaults if needed
            )



            print("\n--- Soul Completion Process Finished ---")
            print("Final Soul State Summary:")
            print(f"  ID: {final_soul_object.spark_id}")
            print(f"  Incarnated Flag: {getattr(final_soul_object, 'incarnated', False)}")
            print(f"  Birth Time: {getattr(final_soul_object, 'birth_time', 'N/A')}")
            print(f"  Final Stability: {getattr(final_soul_object, 'stability', 'N/A'):.4f}")
            print(f"  Final Coherence: {getattr(final_soul_object, 'coherence', 'N/A'):.4f}")
            print(f"  Final Frequency: {getattr(final_soul_object, 'frequency', 'N/A'):.2f} Hz")
            print(f"  Crystallized Name: {getattr(final_soul_object, 'name', 'N/A')}")
            print(f"  Memory Echo Count: {len(getattr(final_soul_object, 'memory_echoes', []))}")


            print("\nOverall Process Metrics:")
            print(f"  Duration: {summary_metrics_result.get('duration_seconds', 'N/A'):.2f}s")
            print(f"  Success: {summary_metrics_result.get('success')}")

            # Save controller state
            soul_completion_controller.save_controller_state()



        except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as e:
            print(f"\n--- ERROR during Soul Completion Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            if soul_completion_controller: print(f"Controller state at error: {soul_completion_controller}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Soul Completion Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            if soul_completion_controller: print(f"Controller state at error: {soul_completion_controller}")
            import traceback; traceback.print_exc()


        
    print("\nSoul Completion Controller Module Example Finished.")


# --- END OF FILE soul_completion_controller.py ---