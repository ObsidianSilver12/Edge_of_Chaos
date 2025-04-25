# --- START OF FILE root_controller.py ---

import logging
import os
import sys
import time
import traceback
from typing import List

# --- Setup Project Root Path ---
# Ensure the 'src' directory is accessible for imports
try:
    # Assumes root_controller.py is in the project root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(ROOT_DIR, "src")
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    # Also add the root itself for potentially importing top-level modules like metrics_tracking
    if ROOT_DIR not in sys.path:
         sys.path.insert(0, ROOT_DIR)

    # --- Core Controller Imports ---
    from stage_1.fields.soul_field_controller import SoulFieldController
    from stage_1.soul_formation.soul_completion_controller import SoulCompletionController
    from stage_1.soul_formation.soul_spark import SoulSpark # Needed for type checking

    # --- Constants Import ---
    from constants.constants import * # Import constants for potential use

    # --- Metrics Import ---
    try:
         import metrics_tracking as metrics
         METRICS_AVAILABLE = True
    except ImportError:
         # Define placeholder if needed
         class MetricsPlaceholder: def record_metrics(*args, **kwargs): pass
         metrics = MetricsPlaceholder()
         METRICS_AVAILABLE = False
         logging.getLogger("root_controller").warning("Metrics tracking module not found.")

except ImportError as e:
    # Use basic print/logging if imports fail early
    print(f"CRITICAL ERROR: Failed to set up paths or import core controllers: {e}", file=sys.stderr)
    print(f"Ensure script is run from project root and src structure is correct.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1) # Exit if core components cannot be imported

# --- Logging Setup ---
# Configure logging for the entire application here
log_file_path = os.path.join("logs", "root_controller_run.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL, # Use LOG_LEVEL from constants
    format=LOG_FORMAT, # Use LOG_FORMAT from constants
    handlers=[
        logging.StreamHandler(sys.stdout), # Log to console
        logging.FileHandler(log_file_path, mode='w') # Log to file
    ]
)
logger = logging.getLogger('root_controller')


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

    all_souls_summary = {}
    field_controller: Optional[SoulFieldController] = None
    completion_controller: Optional[SoulCompletionController] = None

    try:
        # --- 1. Initialize Field System & Controllers ---
        logger.info("Initializing Field Controller and creating Tree of Life...")
        # SoulFieldController handles FieldSystem and Registry init
        field_controller = SoulFieldController(
            auto_initialize=True,
            create_tree=True # Ensure the tree structure is created
        )
        logger.info("Field Controller initialized.")

        # Initialize Completion Controller (can be done later, but good to have ready)
        completion_controller = SoulCompletionController(
            # Pass necessary config if needed, e.g., output dir base from constants
             data_dir=os.path.join(OUTPUT_DIR_BASE, "completion_data")
        )
        logger.info("Completion Controller initialized.")

        # --- 2. Create Souls ---
        created_soul_ids: List[str] = []
        logger.info(f"Creating {num_souls} soul(s)...")
        for i in range(num_souls):
            # Start souls in Guff by default, ready for entanglement
            soul_id = field_controller.create_soul(
                name=f"SimSoul_{i+1}",
                initial_field_key="guff" # Souls typically start in Guff before journey
            )
            created_soul_ids.append(soul_id)
        logger.info(f"Created {len(created_soul_ids)} souls.")

        # --- 3. Process Each Soul ---
        for soul_id in created_soul_ids:
            logger.info(f"\n--- Processing Soul: {soul_id} ---")
            soul_summary = {'id': soul_id, 'stages': {}}
            try:
                # Retrieve the actual SoulSpark object
                soul_spark = field_controller.get_soul(soul_id)

                # a. Creator Entanglement (Run before journey)
                logger.info(f"Running Creator Entanglement for {soul_id}...")
                entanglement_success = field_controller.initiate_soul_journey(soul_id) # Uses defaults
                if not entanglement_success:
                     raise RuntimeError(f"Creator Entanglement failed for soul {soul_id}.")
                soul_summary['stages']['creator_entanglement'] = "Success"
                logger.info("Creator Entanglement complete.")

                # b. Sephiroth Journey (Run after entanglement)
                logger.info(f"Running Sephiroth Journey for {soul_id}...")
                # Define the journey path (e.g., descending)
                default_journey_path = [
                    "kether", "chokmah", "binah", # "daath", # Optionally include Daath
                    "chesed", "geburah", "tiphareth",
                    "netzach", "hod", "yesod", "malkuth"
                ]
                journey_success = field_controller.run_sephiroth_journey(
                    soul_id,
                    journey_path=default_journey_path,
                    duration_per_sephirah=journey_duration
                )
                if not journey_success:
                     raise RuntimeError(f"Sephiroth Journey failed for soul {soul_id}.")
                soul_summary['stages']['sephiroth_journey'] = "Success"
                logger.info("Sephiroth Journey complete.")

                # c. Completion Stages (Run after journey)
                logger.info(f"Running Completion Stages for {soul_id}...")
                # The SoulCompletionController takes the modified soul_spark object
                final_soul, completion_metrics = completion_controller.run_soul_completion(
                    soul_spark = soul_spark # Pass the object directly
                    # Add specific kwargs here to override defaults if needed
                    # e.g., birth_intensity=0.7
                )
                soul_summary['stages']['completion'] = "Success"
                soul_summary['final_state'] = final_soul.get_spark_metrics() # Get final metrics
                logger.info("Completion Stages complete.")

            except (ValueError, TypeError, RuntimeError, AttributeError) as stage_err:
                 logger.error(f"Error processing soul {soul_id}: {stage_err}", exc_info=True)
                 soul_summary['error'] = str(stage_err)
                 soul_summary['failed_stage'] = field_controller.active_souls.get(soul_id,{}).get('current_stage', 'unknown') if field_controller else 'unknown'
            except Exception as unexp_err:
                 logger.critical(f"Unexpected critical error processing soul {soul_id}: {unexp_err}", exc_info=True)
                 soul_summary['error'] = f"Unexpected: {str(unexp_err)}"
                 soul_summary['failed_stage'] = 'unexpected'

            all_souls_summary[soul_id] = soul_summary # Add individual summary to overall report

        # --- 4. Final Report ---
        logger.info("Simulation loop finished. Generating final report...")
        end_time = time.time()
        final_report = {
            'simulation_start_time': start_time_iso if 'start_time_iso' in locals() else datetime.now().isoformat(), # Capture start time properly
            'simulation_end_time': datetime.now().isoformat(),
            'total_duration_seconds': end_time - start_time,
            'souls_processed': len(created_soul_ids),
            'journey_duration_per_sephirah': journey_duration,
            'results_per_soul': all_souls_summary,
            # Include final system status snapshot
            'final_system_status': field_controller.get_system_status() if field_controller else "Controller not initialized"
        }

        # Save report
        try:
            if report_path:
                 report_dir = os.path.dirname(report_path)
                 if report_dir: os.makedirs(report_dir, exist_ok=True)
                 with open(report_path, 'w') as f:
                     json.dump(final_report, f, indent=2, default=str) # Use default=str for complex types
                 logger.info(f"Final simulation report saved to: {report_path}")
        except Exception as report_err:
             logger.error(f"Failed to save final report to {report_path}: {report_err}")

        logger.info(f"--- Soul Development Simulation Finished ---")

    except Exception as main_err:
        logger.critical(f"Simulation aborted due to critical error: {main_err}", exc_info=True)
        # Optionally try to save partial state or metrics
        if field_controller and hasattr(field_controller, 'save_controller_state'):
            try: field_controller.save_controller_state("controller_state_on_error.json")
            except Exception as save_err: logger.error(f"Failed to save controller state on error: {save_err}")
        if METRICS_AVAILABLE and hasattr(metrics, 'persist_metrics'):
            try: metrics.persist_metrics()
            except Exception as persist_err: logger.error(f"Failed to persist metrics on error: {persist_err}")


if __name__ == "__main__":
    # Example: Run simulation with 2 souls, 3 units of time per Sephirah
    run_simulation(num_souls=1, journey_duration=3.0)
    # You can add argument parsing here to control num_souls, etc. from command line


# --- END OF FILE root_controller.py ---