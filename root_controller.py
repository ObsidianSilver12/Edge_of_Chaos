# --- START OF FILE root_controller.py ---

import logging
import os
import sys
import time
import traceback
import json
from typing import List, Optional
from datetime import datetime
from stage_1.fields.soul_field_controller import SoulFieldController


# --- Setup Project Root Path ---
try:
 sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except (OSError, AttributeError, TypeError) as e:
    print(f"Failed to set up project paths: {e}")
    sys.exit(1)

# --- Core Controller Imports ---

from stage_1.soul_formation.soul_spark import SoulSpark 
from stage_1.soul_formation.soul_completion_controller import SoulCompletionController



# Needed for type checking

# --- Constants Import ---
from constants.constants import * # Import constants for potential use



# --- Metrics Import ---
try:
    # Ensure this path is correct relative to the root controller's location
    # If root_controller is in the project root, 'metrics_tracking' should work if it's also there
    # Assuming metrics_tracking.py is in the root directory alongside root_controller.py for now:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
    logger = logging.getLogger('root_controller') # Ensure logger is defined before using it
    logger.info("Metrics tracking module loaded successfully.")
except ImportError:
    logger = logging.getLogger('root_controller') # Ensure logger is defined before using it
    logger.warning("Metrics tracking module not found. Using placeholder.")
    # Define placeholder class WITH CORRECT INDENTATION
    class MetricsPlaceholder:
        # Corrected: Method definition indented inside the class
        def record_metrics(self, *args, **kwargs): # Added self (conventional)
            pass # Placeholder implementation
    metrics = MetricsPlaceholder() # Assign instance
    METRICS_AVAILABLE = False
except Exception as e: # Catch other potential errors during import
    logger = logging.getLogger('root_controller')
    logger.critical(f"CRITICAL ERROR during metrics import: {e}", exc_info=True)
    # Define placeholder even on unexpected import error
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs):
            pass
    metrics = MetricsPlaceholder()
    METRICS_AVAILABLE = False

# Ensure logger is configured *after* the logger = logging.getLogger(...) lines
# Move the basicConfig call here or ensure it happens before any logging attempts
if not logging.getLogger().hasHandlers():
    log_file_path = os.path.join("logs", "root_controller_run.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL, # Use LOG_LEVEL from constants if available, else fallback
        format=LOG_FORMAT if 'LOG_FORMAT' in globals() else '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path, mode='w')
        ]
    )


# --- Main Application Logic ---

def run_simulation(num_souls: int = 1, journey_duration: float = 5.0, report_path: str = "output/final_report.json"):
    """
    Runs the main soul formation simulation flow.

    Args:
        num_souls: Number of souls to create and process.
        journey_duration: Duration (simulation units) spent in each Sephirah.
        report_path: Path to save the final summary report.
    """
    logger.info(f"--- Starting Soul Development Simulation ---")
    logger.info(f"Number of Souls: {num_souls}")
    logger.info(f"Duration per Sephirah: {journey_duration}")
    start_time = time.time()
    start_time_iso = datetime.now().isoformat() # Capture start time

    all_souls_summary = {}
    field_controller: Optional[SoulFieldController] = None
    completion_controller: Optional[SoulCompletionController] = None

    try:
        # --- 1. Initialize Field System & Controllers ---
        logger.info("Initializing Field Controller and creating Tree of Life...")
        field_controller = SoulFieldController(
            auto_initialize=True,
            create_tree=True
        )
        logger.info("Field Controller initialized.")

        completion_controller = SoulCompletionController(
             data_dir=os.path.join(OUTPUT_DIR_BASE, "completion_data")
        )
        logger.info("Completion Controller initialized.")

        # --- 2. Create Souls ---
        created_soul_ids: List[str] = []
        logger.info(f"Creating {num_souls} soul(s)...")
        for i in range(num_souls):
            soul_id = field_controller.create_soul(
                name=f"SimSoul_{i+1}",
                initial_field_key="guff"
            )
            created_soul_ids.append(soul_id)
        logger.info(f"Created {len(created_soul_ids)} souls.")

        # --- 3. Process Each Soul ---
        for soul_id in created_soul_ids:
            logger.info(f"\n--- Processing Soul: {soul_id} ---")
            soul_summary = {'id': soul_id, 'stages': {}, 'success': True, 'failed_stage': None, 'error': None}
            current_stage_name = "Start" # Track current stage locally for error reporting
            try:
                # Retrieve the actual SoulSpark object
                soul_spark = field_controller.get_soul(soul_id)

                # a. Creator Entanglement
                current_stage_name = "Creator Entanglement"
                logger.info(f"Running {current_stage_name} for {soul_id}...")
                entanglement_success = field_controller.initiate_soul_journey(soul_id) # Uses defaults
                if not entanglement_success:
                     # This function now raises RuntimeError on failure, so this check might be redundant
                     raise RuntimeError(f"Creator Entanglement returned False for soul {soul_id}.")
                soul_summary['stages']['creator_entanglement'] = "Success"
                logger.info(f"{current_stage_name} complete.")

                # b. Sephiroth Journey
                current_stage_name = "Sephiroth Journey"
                logger.info(f"Running {current_stage_name} for {soul_id}...")
                default_journey_path = [
                    "kether", "chokmah", "binah", # "daath", # Optional
                    "chesed", "geburah", "tiphareth",
                    "netzach", "hod", "yesod", "malkuth"
                ]
                journey_success = field_controller.run_sephiroth_journey(
                    soul_id,
                    journey_path=default_journey_path,
                    duration_per_sephirah=journey_duration
                )
                if not journey_success:
                     # This function also raises RuntimeError on failure
                     raise RuntimeError(f"Sephiroth Journey returned False for soul {soul_id}.")
                soul_summary['stages']['sephiroth_journey'] = "Success"
                logger.info(f"{current_stage_name} complete.")

                # c. Completion Stages
                current_stage_name = "Completion Stages"
                logger.info(f"Running {current_stage_name} for {soul_id}...")
                # Ensure the completion controller is available
                if not completion_controller:
                    raise RuntimeError("SoulCompletionController was not initialized.")

                # run_soul_completion modifies soul_spark in place and returns summary
                final_soul, completion_summary = completion_controller.run_soul_completion(
                    soul_spark = soul_spark # Pass the object directly
                    # Add specific kwargs here to override defaults if needed
                    # e.g., birth_intensity=0.7
                )
                # Check success from the completion summary if available
                if not completion_summary.get('success', False):
                    # Completion controller might have internal errors caught, check its summary
                    raise RuntimeError(f"Soul Completion stage failed for soul {soul_id}. Reason: {completion_summary.get('error', 'Unknown')}")
                soul_summary['stages']['completion'] = "Success"
                soul_summary['final_state'] = final_soul.get_spark_metrics() # Get final metrics
                logger.info(f"{current_stage_name} complete.")

            except (ValueError, TypeError, RuntimeError, AttributeError) as stage_err:
                 # Catch errors from any stage *within* the loop for this soul
                 error_message = f"Error processing soul {soul_id} during stage '{current_stage_name}': {stage_err}"
                 logger.error(error_message, exc_info=False) # Log error without full traceback for brevity here
                 soul_summary['error'] = str(stage_err)
                 soul_summary['failed_stage'] = current_stage_name
                 soul_summary['success'] = False # Mark this soul as failed
            except Exception as unexp_err:
                 # Catch unexpected errors during this soul's processing
                 error_message = f"Unexpected critical error processing soul {soul_id} during stage '{current_stage_name}': {unexp_err}"
                 logger.critical(error_message, exc_info=True) # Log full traceback for critical errors
                 soul_summary['error'] = f"Unexpected: {str(unexp_err)}"
                 soul_summary['failed_stage'] = current_stage_name
                 soul_summary['success'] = False

            all_souls_summary[soul_id] = soul_summary # Add individual summary to overall report

        # --- 4. Final Report ---
        logger.info("Simulation loop finished. Generating final report...")
        end_time = time.time()
        final_report = {
            'simulation_start_time': start_time_iso,
            'simulation_end_time': datetime.now().isoformat(),
            'total_duration_seconds': end_time - start_time,
            'souls_processed': len(created_soul_ids),
            'journey_duration_per_sephirah': journey_duration,
            'results_per_soul': all_souls_summary,
            'final_system_status': field_controller.get_system_status() if field_controller else "Controller not initialized"
        }

        # Save report
        try:
            if report_path:
                 # Ensure directory exists before writing
                 report_dir = os.path.dirname(report_path)
                 if report_dir: os.makedirs(report_dir, exist_ok=True)
                 with open(report_path, 'w') as f:
                     # Use default=str to handle potential datetime or numpy types
                     json.dump(final_report, f, indent=2, default=str)
                 logger.info(f"Final simulation report saved to: {report_path}")
        except Exception as report_err:
             logger.error(f"Failed to save final report to {report_path}: {report_err}")

        logger.info(f"--- Soul Development Simulation Finished ---")

    except Exception as main_err:
        # Catch errors during initialization or the main loop setup
        logger.critical(f"Simulation aborted due to critical error: {main_err}", exc_info=True)
        # Optionally try to save partial state or metrics
        if field_controller and hasattr(field_controller, 'save_controller_state'): # Check existence before calling
            try: field_controller.save_controller_state("controller_state_on_error.json")
            except Exception as save_err: logger.error(f"Failed to save controller state on error: {save_err}")
        if METRICS_AVAILABLE and hasattr(metrics, 'persist_metrics'):
            try: metrics.persist_metrics()
            except Exception as persist_err: logger.error(f"Failed to persist metrics on error: {persist_err}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Example: Run simulation with 1 soul, 2 units of time per Sephirah
    try:
        run_simulation(num_souls=1, journey_duration=2.0, report_path="output/simulation_report_main.json")
    except Exception as e:
         # Catch errors raised from run_simulation itself (e.g., init failures)
         logger.critical(f"Main execution failed: {e}", exc_info=True)
         sys.exit(1) # Exit with error code