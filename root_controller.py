# --- START OF FILE root_controller.py ---

import logging
import os
import sys
import time
import traceback
import json
from typing import List, Optional
from datetime import datetime


# --- Setup Project Root Path ---
# Assuming this part works correctly
try:
 sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except (OSError, AttributeError, TypeError) as e:
    print(f"FATAL: Failed to set up project paths: {e}") # Use print for early errors
    sys.exit(1)

# --- Early Print Statement ---
print("DEBUG: Project path setup attempted. Starting imports...")

# --- Constants Import (crucial) ---
try:
    from constants.constants import *
    print(f"DEBUG: Constants loaded. LOG_LEVEL={LOG_LEVEL}, OUTPUT_DIR_BASE={OUTPUT_DIR_BASE}")
except ImportError as e:
    print(f"FATAL: Could not import constants: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Error loading constants: {e}")
    sys.exit(1)

# --- Core Controller Imports ---
# Use print statements around potentially problematic imports if needed
try:
    from stage_1.fields.soul_field_controller import SoulFieldController
    print("DEBUG: Imported SoulFieldController")
    from stage_1.soul_formation.soul_spark import SoulSpark
    print("DEBUG: Imported SoulSpark")
    from stage_1.soul_formation.soul_completion_controller import SoulCompletionController
    print("DEBUG: Imported SoulCompletionController")
except ImportError as e:
    print(f"FATAL: Could not import core stage components: {e}")
    sys.exit(1)


# --- Logger Initialization (Do this ONCE and EARLY) ---
# Define logger name
LOGGER_NAME = 'root_controller'
logger = logging.getLogger(LOGGER_NAME)
print(f"DEBUG: Logger '{LOGGER_NAME}' obtained.")

# --- BasicConfig Setup (Call this BEFORE first log message) ---
# Ensure it only runs once
if not logging.getLogger(LOGGER_NAME).hasHandlers():
    print("DEBUG: Configuring logging...")
    try:
        # Use DEBUG level for detailed output during debugging
        effective_log_level = logging.DEBUG # Override constant for debugging
        log_file_path = os.path.join("logs", "root_controller_run.log")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        log_format = LOG_FORMAT # Use constant if defined, else basic
        
        logging.basicConfig(
            level=effective_log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout), # Log to console
                logging.FileHandler(log_file_path, mode='w') # Log to file
            ]
        )
        # Test the logger immediately after config
        logger.debug(f"Logging configured successfully. Level: {logging.getLevelName(effective_log_level)}, File: {log_file_path}")
    except Exception as log_err:
        print(f"FATAL: Failed to configure logging: {log_err}")
        # Continue without logging if it fails, relying on prints
else:
    print("DEBUG: Logging already configured.")


# --- Metrics Import (After Logger Config) ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
    logger.info("Metrics tracking module loaded successfully.") # Now logger should work
except ImportError:
    logger.warning("Metrics tracking module not found. Using placeholder.") # Logger should work
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()
    METRICS_AVAILABLE = False
except Exception as e: # Catch other potential errors during import
    logger.critical(f"CRITICAL ERROR during metrics import: {e}", exc_info=True)
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()
    METRICS_AVAILABLE = False


# --- Main Application Logic ---

def run_simulation(num_souls: int = 1, journey_duration: float = 5.0, report_path: str = "output/final_report.json"):
    """
    Runs the main soul formation simulation flow.
    Args:
        num_souls: Number of souls to create and process.
        journey_duration: Duration (simulation units) spent in each Sephirah.
        report_path: Path to save the final summary report.
    """
    # === Log Start ===
    logger.info("--- Starting Soul Development Simulation ---") # Now this should appear
    logger.info(f"Number of Souls: {num_souls}")
    logger.info(f"Duration per Sephirah: {journey_duration}")
    logger.debug(f"Using OUTPUT_DIR_BASE: {OUTPUT_DIR_BASE}") # Log path being used

    start_time = time.time()
    start_time_iso = datetime.now().isoformat()

    all_souls_summary = {}
    field_controller: Optional[SoulFieldController] = None
    completion_controller: Optional[SoulCompletionController] = None

    try:
        # --- 1. Initialize Field System & Controllers ---
        logger.info("Attempting to initialize Field Controller and create Tree of Life...")
        print("DEBUG: About to initialize SoulFieldController...") # Add print statement
        try:
            field_controller = SoulFieldController(
                auto_initialize=True,
                create_tree=True # This implicitly initializes FieldSystem and runs create_tree
            )
            print("DEBUG: SoulFieldController initialized.") # Add print statement
            logger.info("Field Controller initialized successfully.")
        except Exception as fc_init_err:
            print(f"FATAL: Error during SoulFieldController initialization: {fc_init_err}")
            logger.critical("Error during SoulFieldController initialization", exc_info=True)
            raise # Re-raise to be caught by the main try-except

        logger.info("Attempting to initialize Completion Controller...")
        print("DEBUG: About to initialize SoulCompletionController...") # Add print statement
        try:
             completion_data_dir = os.path.join(OUTPUT_DIR_BASE, "completion_data")
             logger.debug(f"Completion Controller data directory: {completion_data_dir}")
             completion_controller = SoulCompletionController(
                  data_dir=completion_data_dir
             )
             print("DEBUG: SoulCompletionController initialized.") # Add print statement
             logger.info("Completion Controller initialized successfully.")
        except Exception as cc_init_err:
             print(f"FATAL: Error during SoulCompletionController initialization: {cc_init_err}")
             logger.critical("Error during SoulCompletionController initialization", exc_info=True)
             raise # Re-raise

        # === Soul Creation ===
        created_soul_ids: List[str] = []
        logger.info(f"Creating {num_souls} soul(s)...")
        print(f"DEBUG: Starting soul creation loop for {num_souls} souls...") # Add print statement
        for i in range(num_souls):
            soul_num = i + 1
            logger.debug(f"Attempting to create soul {soul_num}...")
            print(f"DEBUG: Attempting field_controller.create_soul for soul {soul_num}...") # Add print statement
            try:
                soul_id = field_controller.create_soul(
                    name=f"SimSoul_{soul_num}",
                    initial_field_key="guff" # Ensure 'guff' is a valid key handled by create_soul
                )
                created_soul_ids.append(soul_id)
                logger.debug(f"Soul {soul_num} created with ID: {soul_id}")
                print(f"DEBUG: Soul {soul_num} created (ID: {soul_id}).") # Add print statement
            except Exception as soul_create_err:
                 print(f"ERROR: Failed to create soul {soul_num}: {soul_create_err}")
                 logger.error(f"Failed to create soul {soul_num}", exc_info=True)
                 # Decide whether to continue or raise
                 raise RuntimeError(f"Critical failure creating soul {soul_num}") from soul_create_err
        logger.info(f"Successfully created {len(created_soul_ids)} souls.")
        print(f"DEBUG: Finished soul creation loop. Created IDs: {created_soul_ids}") # Add print statement


        # === Process Each Soul ===
        print("DEBUG: Starting soul processing loop...") # Add print statement
        for soul_id in created_soul_ids:
            logger.info(f"\n--- Processing Soul: {soul_id} ---")
            soul_summary = {'id': soul_id, 'stages': {}, 'success': True, 'failed_stage': None, 'error': None}
            current_stage_name = "Start"

            try:
                logger.debug(f"Retrieving SoulSpark object for {soul_id}...")
                print(f"DEBUG: Getting soul object {soul_id}...") # Add print statement
                soul_spark = field_controller.get_soul(soul_id) # Use the Field Controller to get the soul
                if not soul_spark: # Handle case where get_soul might return None (should raise error ideally)
                     raise ValueError(f"Could not retrieve SoulSpark object for ID {soul_id} from controller.")
                print(f"DEBUG: Got soul object {soul_id}.") # Add print statement

                # a. Creator Entanglement
                current_stage_name = "Creator Entanglement"
                logger.info(f"Running {current_stage_name} for {soul_id}...")
                print(f"DEBUG: Initiating soul journey (Entanglement) for {soul_id}...") # Add print statement
                # This method should modify soul_spark in place and return True/False or raise error
                entanglement_success = field_controller.initiate_soul_journey(soul_id) # Uses defaults
                if not entanglement_success:
                     # If it returns False instead of raising error on failure
                     raise RuntimeError(f"Creator Entanglement returned False for soul {soul_id}.")
                soul_summary['stages']['creator_entanglement'] = "Success"
                logger.info(f"{current_stage_name} complete.")
                print(f"DEBUG: Entanglement complete for {soul_id}.") # Add print statement


                # b. Sephiroth Journey
                current_stage_name = "Sephiroth Journey"
                logger.info(f"Running {current_stage_name} for {soul_id}...")
                print(f"DEBUG: Running Sephiroth journey for {soul_id}...") # Add print statement
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
                     # This function also returns False or raises error
                     raise RuntimeError(f"Sephiroth Journey returned False for soul {soul_id}.")
                soul_summary['stages']['sephiroth_journey'] = "Success"
                logger.info(f"{current_stage_name} complete.")
                print(f"DEBUG: Sephiroth journey complete for {soul_id}.") # Add print statement

                # c. Completion Stages
                current_stage_name = "Completion Stages"
                logger.info(f"Running {current_stage_name} for {soul_id}...")
                print(f"DEBUG: Running completion stages for {soul_id}...") # Add print statement
                if not completion_controller:
                    raise RuntimeError("SoulCompletionController was not initialized.")

                # run_soul_completion modifies soul_spark in place and returns summary
                final_soul, completion_summary_data = completion_controller.run_soul_completion(
                    soul_spark = soul_spark # Pass the object directly
                )
                # Check success from the completion summary
                if not completion_summary_data.get('success', False):
                    raise RuntimeError(f"Soul Completion stage failed for soul {soul_id}. Reason: {completion_summary_data.get('error', 'Unknown')}")
                soul_summary['stages']['completion'] = "Success"
                soul_summary['final_state'] = final_soul.get_spark_metrics() # Get final metrics
                logger.info(f"{current_stage_name} complete.")
                print(f"DEBUG: Completion stages complete for {soul_id}.") # Add print statement


            except (ValueError, TypeError, RuntimeError, AttributeError) as stage_err:
                 error_message = f"Error processing soul {soul_id} during stage '{current_stage_name}': {stage_err}"
                 print(f"ERROR: {error_message}") # Print error as well
                 logger.error(error_message, exc_info=False)
                 soul_summary['error'] = str(stage_err)
                 soul_summary['failed_stage'] = current_stage_name
                 soul_summary['success'] = False
            except Exception as unexp_err:
                 error_message = f"Unexpected critical error processing soul {soul_id} during stage '{current_stage_name}': {unexp_err}"
                 print(f"CRITICAL ERROR: {error_message}") # Print error as well
                 logger.critical(error_message, exc_info=True)
                 soul_summary['error'] = f"Unexpected: {str(unexp_err)}"
                 soul_summary['failed_stage'] = current_stage_name
                 soul_summary['success'] = False

            all_souls_summary[soul_id] = soul_summary
        print("DEBUG: Finished soul processing loop.") # Add print statement

        # --- 4. Final Report ---
        logger.info("Simulation loop finished. Generating final report...")
        print("DEBUG: Generating final report...") # Add print statement
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
        print(f"DEBUG: Attempting to save report to {report_path}...") # Add print statement
        try:
            if report_path:
                 report_dir = os.path.dirname(report_path)
                 if report_dir:
                      print(f"DEBUG: Ensuring report directory exists: {report_dir}") # Add print statement
                      os.makedirs(report_dir, exist_ok=True)
                 with open(report_path, 'w') as f:
                     json.dump(final_report, f, indent=2, default=str)
                 logger.info(f"Final simulation report saved to: {report_path}")
                 print(f"DEBUG: Report saved to {report_path}") # Add print statement
        except Exception as report_err:
             print(f"ERROR: Failed to save report: {report_err}") # Add print statement
             logger.error(f"Failed to save final report to {report_path}: {report_err}")

        logger.info(f"--- Soul Development Simulation Finished ---")
        print("DEBUG: run_simulation function finished.") # Add print statement

    except Exception as main_err:
        # Catch errors during initialization or the main loop setup outside the per-soul loop
        error_message = f"Simulation aborted due to critical error: {main_err}"
        print(f"FATAL ERROR: {error_message}") # Print error
        logger.critical(error_message, exc_info=True) # Log with traceback
        # Attempt cleanup/save state (optional)
        # ... (cleanup code) ...

# --- Main Execution Block ---
if __name__ == "__main__":
    print("DEBUG: Starting main execution block...") # Add print statement
    try:
        run_simulation(num_souls=1, journey_duration=2.0, report_path="output/simulation_report_main.json")
        print("DEBUG: run_simulation completed without raising an exception to the main block.") # Add print statement
    except Exception as e:
         print(f"FATAL ERROR in main execution block: {e}") # Print error
         # Use logger if configured, otherwise just print traceback
         if logger.hasHandlers():
              logger.critical(f"Main execution failed: {e}", exc_info=True)
         else:
              print("Logging not configured. Printing traceback:")
              traceback.print_exc()
         sys.exit(1)
    print("DEBUG: Main execution block finished.") # Add print statement