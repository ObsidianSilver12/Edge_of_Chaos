# --- START OF FILE guff_controller.py ---

"""
Guff Field Controller (Refactored for 3D and Orchestration)

Manages the Guff field interactions and soul spark strengthening process.
Orchestrates the reception of sparks from the Void field, the strengthening
process within the Guff field, and preparation for transfer to the Sephiroth stage.
Enforces strict error handling and delegates field logic to GuffField3D.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
import uuid
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# --- Constants ---
try:
    from src.constants import (
        DEFAULT_DIMENSIONS_3D, LOG_LEVEL, LOG_FORMAT, DATA_DIR_BASE
    )
except ImportError as e:
    # Basic logging setup if constants failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. GuffController cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    # Assumes GuffField is refactored to GuffField3D in the correct path
    from stage_1.void.guff_field import GuffField3D
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import GuffField3D: {e}. GuffController cannot function.")
    raise ImportError(f"Core dependency GuffField3D missing: {e}") from e

try:
    # Import the refactored metrics tracking module
    import metrics_tracking as metrics
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import metrics_tracking: {e}. GuffController cannot function.")
    raise ImportError(f"Core dependency metrics_tracking missing: {e}") from e

# Conditional import for visualization
try:
    import matplotlib.pyplot as plt
    # We don't need Axes3D here, the field handles its own plotting
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    logging.warning("Matplotlib not found. Visualization methods will be disabled.")

# Configure logging
log_file_path = os.path.join("logs", "guff_controller.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('guff_controller')

# Define controller-specific metric category
CONTROLLER_METRIC_CATEGORY = "guff_controller"

class GuffController:
    """
    Controller for 3D Guff field operations and soul spark strengthening.
    Orchestrates the strengthening stage of soul formation.
    """

    def __init__(self, dimensions: Tuple[int, int, int] = DEFAULT_DIMENSIONS_3D,
                 field_name: str = "guff_field", data_dir: str = DATA_DIR_BASE,
                 controller_id: Optional[str] = None):
        """
        Initialize a new Guff Field Controller. Fails hard on invalid configuration.

        Args:
            dimensions (Tuple[int, int, int]): Dimensions of the Guff field.
            field_name (str): Name for the associated GuffField3D instance.
            data_dir (str): Base directory for field data.
            controller_id (Optional[str]): Specific ID for the controller, generates if None.

        Raises:
            ValueError: If dimensions, field_name, or data_dir are invalid.
            RuntimeError: If GuffField3D initialization fails.
            OSError: If the output directory cannot be created.
        """
        # --- Input Validation ---
        if not isinstance(dimensions, tuple) or len(dimensions) != 3 or not all(isinstance(d, int) and d > 0 for d in dimensions):
            raise ValueError(f"Dimensions must be a tuple of 3 positive integers, got {dimensions}")
        if not field_name or not isinstance(field_name, str):
            raise ValueError("Field name must be a non-empty string.")
        if not data_dir or not isinstance(data_dir, str):
            raise ValueError("Data directory must be a non-empty string.")

        self.controller_id: str = controller_id or str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.field_dimensions: Tuple[int, int, int] = dimensions
        # Controller's specific output dir for its state, logs, etc.
        self.output_dir: str = os.path.join(data_dir, "controller_data", f"guff_{self.controller_id}")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create controller output directory {self.output_dir}: {e}")
            raise

        logger.info(f"Initializing Guff Field Controller (ID: {self.controller_id})")

        # Initialize the Guff field (raises errors on failure)
        try:
            self.guff_field: GuffField3D = GuffField3D(
                dimensions=self.field_dimensions,
                field_name=field_name,
                data_dir=data_dir # Pass base data dir to field
            )
        except (ValueError, RuntimeError, ImportError) as e:
            logger.critical(f"CRITICAL: Failed to initialize GuffField3D: {e}", exc_info=True)
            raise RuntimeError(f"GuffField3D initialization failed: {e}") from e

        # --- State Tracking ---
        # Stores the *transfer data* dict received from VoidController for sparks being processed
        self.processing_spark_data: Dict[str, Dict[str, Any]] = {}
        # Stores the *reception info* dict returned by guff_field.receive_spark
        self.reception_info_data: Dict[str, Dict[str, Any]] = {}
        # Stores the *strengthening result* dict returned by guff_field.strengthen_soul_formation
        self.strengthening_result_data: Dict[str, Dict[str, Any]] = {}
        # Stores the *finalized soul data* dict returned by guff_field.finalize_soul_formation
        self.finalized_soul_data: Dict[str, Dict[str, Any]] = {}
        # Stores the *transfer data* dict returned by guff_field.prepare_for_sephiroth_transfer
        self.transfer_ready_data: Dict[str, Dict[str, Any]] = {}

        self.simulation_step: int = 0 # Tracks number of strengthening cycles run by controller

        # Record initial controller state metric
        metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
            'status': 'initialized',
            'controller_id': self.controller_id,
            'timestamp': self.creation_time,
            'field_uuid': self.guff_field.uuid,
            'dimensions': list(self.field_dimensions),
            'simulation_step': self.simulation_step
        })

        logger.info(f"Guff Field Controller '{self.controller_id}' initialized successfully.")
        logger.info(f"Associated Guff Field: '{self.guff_field.field_name}' (UUID: {self.guff_field.uuid})")

    def receive_spark_from_void(self, spark_transfer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives spark data prepared by the Void stage and initiates processing in Guff.
        Fails hard if input is invalid or field processing fails.

        Args:
            spark_transfer_data (Dict[str, Any]): The dictionary returned by
                                                  `VoidFieldController.transfer_spark_to_guff`.
                                                  Must contain 'transfer_complete', 'spark_id',
                                                  'position', 'void_field_metrics', 'energy', 'stability'.

        Returns:
            Dict[str, Any]: The reception information dictionary returned by `GuffField3D.receive_spark`.

        Raises:
            TypeError: If spark_transfer_data is not a dictionary.
            ValueError: If spark_transfer_data is missing essential keys or is invalid.
            RuntimeError: If the `guff_field.receive_spark` method fails.
        """
        logger.info("Controller receiving spark data from Void stage...")
        if not isinstance(spark_transfer_data, dict):
            raise TypeError("spark_transfer_data must be a dictionary.")

        # Validate essential keys from the transfer data
        required_keys = ["transfer_complete", "spark_id", "position", "energy", "stability", "void_field_metrics"]
        missing_keys = [key for key in required_keys if key not in spark_transfer_data]
        if missing_keys:
            raise ValueError(f"Spark transfer data is missing required keys: {', '.join(missing_keys)}")
        if not spark_transfer_data["transfer_complete"]:
            raise ValueError("Spark transfer data indicates transfer was not complete.")

        spark_id = spark_transfer_data["spark_id"]
        if not spark_id or not isinstance(spark_id, str):
             raise ValueError("Invalid spark_id in transfer data.")
        if spark_id in self.processing_spark_data or spark_id in self.transfer_ready_data:
            logger.warning(f"Spark {spark_id} has already been received or processed. Skipping.")
            # Return existing reception info if available, otherwise raise error? Let's return existing.
            if spark_id in self.reception_info_data: return self.reception_info_data[spark_id].copy()
            else: raise ValueError(f"Spark {spark_id} already processed but reception info missing.")

        logger.info(f"Processing spark {spark_id} reception in Guff field.")
        self.processing_spark_data[spark_id] = spark_transfer_data # Store input data

        try:
            # Delegate spark reception to the field
            reception_info = self.guff_field.receive_spark(spark_transfer_data)
            if not isinstance(reception_info, dict) or not reception_info.get("reception_complete"):
                 # Field method should raise error on failure, but double check return
                 raise RuntimeError(f"GuffField3D.receive_spark did not return valid reception info for spark {spark_id}.")

            self.reception_info_data[spark_id] = reception_info # Store result

            # Record metrics
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'receive_spark',
                'spark_id': spark_id,
                'guff_position': list(reception_info.get('guff_position', [])),
                'success': True,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"Spark {spark_id} successfully received by Guff field at position {reception_info.get('guff_position')}.")
            return reception_info.copy() # Return a copy

        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            logger.error(f"Failed to receive spark {spark_id} in Guff field: {e}", exc_info=True)
            # Clean up state for this spark if reception failed
            if spark_id in self.processing_spark_data: del self.processing_spark_data[spark_id]
            if spark_id in self.reception_info_data: del self.reception_info_data[spark_id]
            raise RuntimeError(f"Spark reception process failed for {spark_id}: {e}") from e

    def strengthen_soul(self, spark_id: str, iterations: int = 10) -> Dict[str, Any]:
        """
        Initiates the strengthening process for a received spark. Fails hard on error.

        Args:
            spark_id (str): The ID of the spark whose reception is complete.
            iterations (int): Number of strengthening iterations to perform.

        Returns:
            Dict[str, Any]: The strengthening result dictionary from `GuffField3D.strengthen_soul_formation`.

        Raises:
            ValueError: If spark_id is invalid, not found, or iterations is invalid.
            RuntimeError: If the `guff_field.strengthen_soul_formation` method fails.
        """
        logger.info(f"Controller initiating strengthening for spark {spark_id} ({iterations} iterations)...")
        if not spark_id or not isinstance(spark_id, str):
            raise ValueError("Invalid spark_id provided.")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Iterations must be a positive integer.")
        if spark_id not in self.reception_info_data:
            raise ValueError(f"Cannot strengthen spark {spark_id}: Reception info not found. Call receive_spark_from_void first.")

        reception_info = self.reception_info_data[spark_id]

        try:
            # Delegate strengthening to the field
            strengthening_result = self.guff_field.strengthen_soul_formation(reception_info, iterations)
            if not isinstance(strengthening_result, dict) or "final_strength" not in strengthening_result:
                 # Field method should raise error, but double check return
                 raise RuntimeError(f"GuffField3D.strengthen_soul_formation did not return valid result for spark {spark_id}.")

            self.strengthening_result_data[spark_id] = strengthening_result # Store result
            self.simulation_step += iterations # Increment controller step count

            # Record metrics
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'strengthen_soul',
                'spark_id': spark_id,
                'iterations': iterations,
                'final_strength': strengthening_result.get('final_strength'),
                'final_resonance': strengthening_result.get('final_resonance'),
                'formation_quality': strengthening_result.get('formation_quality'),
                'success': True,
                'simulation_step': self.simulation_step,
                'timestamp': datetime.now().isoformat()
            })

            # Visualize (conditional)
            self.visualize_formation(spark_id, save=True, show=False)

            logger.info(f"Strengthening complete for spark {spark_id}. Quality: {strengthening_result.get('formation_quality', 'N/A'):.4f}")
            return strengthening_result.copy() # Return a copy

        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            logger.error(f"Failed to strengthen soul {spark_id}: {e}", exc_info=True)
            # Don't clean up reception info here, allow retry maybe? But remove strengthening result.
            if spark_id in self.strengthening_result_data: del self.strengthening_result_data[spark_id]
            raise RuntimeError(f"Soul strengthening process failed for {spark_id}: {e}") from e

    def finalize_soul(self, spark_id: str) -> Dict[str, Any]:
        """
        Initiates the finalization process for a strengthened soul. Fails hard on error.

        Args:
            spark_id (str): The ID of the spark whose strengthening is complete.

        Returns:
            Dict[str, Any]: The finalized soul data dictionary from `GuffField3D.finalize_soul_formation`.

        Raises:
            ValueError: If spark_id is invalid or strengthening result not found.
            RuntimeError: If the `guff_field.finalize_soul_formation` method fails.
        """
        logger.info(f"Controller initiating finalization for soul {spark_id}...")
        if not spark_id or not isinstance(spark_id, str):
            raise ValueError("Invalid spark_id provided.")
        if spark_id not in self.strengthening_result_data:
            raise ValueError(f"Cannot finalize soul {spark_id}: Strengthening result not found. Call strengthen_soul first.")

        strengthening_result = self.strengthening_result_data[spark_id]

        try:
            # Delegate finalization to the field
            finalized_soul_data = self.guff_field.finalize_soul_formation(strengthening_result)
            if not isinstance(finalized_soul_data, dict) or not finalized_soul_data.get("ready_for_sephiroth"):
                 # Field method should raise error for low quality, but double check return
                 raise RuntimeError(f"GuffField3D.finalize_soul_formation did not return valid finalized data for soul {spark_id}.")

            self.finalized_soul_data[spark_id] = finalized_soul_data # Store result

            # Record metrics
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'finalize_soul',
                'spark_id': spark_id,
                'formation_quality': finalized_soul_data.get('formation_quality'),
                'creator_resonance': finalized_soul_data.get('creator_resonance', {}).get('overall_resonance'),
                'success': True,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"Finalization complete for soul {spark_id}. Ready for Sephiroth: {finalized_soul_data.get('ready_for_sephiroth')}")
            return finalized_soul_data.copy() # Return a copy

        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            logger.error(f"Failed to finalize soul {spark_id}: {e}", exc_info=True)
            # Allow retry? Remove finalized data if it exists.
            if spark_id in self.finalized_soul_data: del self.finalized_soul_data[spark_id]
            # Re-raise the original error type if it's ValueError (e.g., low quality)
            if isinstance(e, ValueError): raise
            else: raise RuntimeError(f"Soul finalization process failed for {spark_id}: {e}") from e

    def prepare_soul_for_transfer(self, spark_id: str) -> Dict[str, Any]:
        """
        Prepares a finalized soul for transfer to the Sephiroth stage. Fails hard on error.

        Args:
            spark_id (str): The ID of the soul whose finalization is complete.

        Returns:
            Dict[str, Any]: The transfer data dictionary from `GuffField3D.prepare_for_sephiroth_transfer`.

        Raises:
            ValueError: If spark_id is invalid or finalized soul data not found.
            RuntimeError: If the `guff_field.prepare_for_sephiroth_transfer` method fails.
        """
        logger.info(f"Controller preparing soul {spark_id} for transfer to Sephiroth...")
        if not spark_id or not isinstance(spark_id, str):
            raise ValueError("Invalid spark_id provided.")
        if spark_id not in self.finalized_soul_data:
            raise ValueError(f"Cannot prepare soul {spark_id}: Finalized data not found. Call finalize_soul first.")

        finalized_data = self.finalized_soul_data[spark_id]
        if not finalized_data.get("ready_for_sephiroth"):
             # This case should ideally be caught by finalize_soul raising error, but check again.
             raise ValueError(f"Cannot prepare soul {spark_id}: Soul is not marked ready_for_sephiroth.")

        try:
            # Delegate preparation to the field
            transfer_data = self.guff_field.prepare_for_sephiroth_transfer(finalized_data)
            if not isinstance(transfer_data, dict) or not transfer_data.get("transfer_prepared"):
                 # Field method should raise error, but double check return
                 raise RuntimeError(f"GuffField3D.prepare_for_sephiroth_transfer did not return valid transfer data for soul {spark_id}.")

            self.transfer_ready_data[spark_id] = transfer_data # Store result

            # Record metrics
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'prepare_for_transfer',
                'spark_id': spark_id,
                'sephiroth_connections_count': len(transfer_data.get('sephiroth_connections', {})),
                'success': True,
                'timestamp': datetime.now().isoformat()
            })

            # Clean up intermediate states for this spark_id (optional, depends on desired traceability)
            # if spark_id in self.processing_spark_data: del self.processing_spark_data[spark_id]
            # if spark_id in self.reception_info_data: del self.reception_info_data[spark_id]
            # if spark_id in self.strengthening_result_data: del self.strengthening_result_data[spark_id]
            # if spark_id in self.finalized_soul_data: del self.finalized_soul_data[spark_id]

            logger.info(f"Preparation for Sephiroth transfer complete for soul {spark_id}.")
            return transfer_data.copy() # Return a copy

        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            logger.error(f"Failed to prepare soul {spark_id} for transfer: {e}", exc_info=True)
            # Allow retry? Remove transfer ready data if it exists.
            if spark_id in self.transfer_ready_data: del self.transfer_ready_data[spark_id]
            raise RuntimeError(f"Soul transfer preparation failed for {spark_id}: {e}") from e

    def run_full_guff_stage(self, input_spark_transfer_data: List[Dict[str, Any]],
                             iterations_per_spark: int = 15) -> List[Dict[str, Any]]:
        """
        Runs the complete Guff stage for a list of input sparks from the Void stage.
        Handles errors per spark, allowing the process to continue for others.

        Args:
            input_spark_transfer_data (List[Dict[str, Any]]): List of spark transfer data dictionaries
                                                             from the Void stage.
            iterations_per_spark (int): Number of strengthening iterations per spark.

        Returns:
            List[Dict[str, Any]]: List of transfer data dictionaries for souls successfully
                                 prepared for the Sephiroth stage.
        """
        if not isinstance(input_spark_transfer_data, list):
            raise TypeError("input_spark_transfer_data must be a list of dictionaries.")
        if not isinstance(iterations_per_spark, int) or iterations_per_spark <= 0:
            raise ValueError("iterations_per_spark must be a positive integer.")

        logger.info(f"Starting full Guff stage processing for {len(input_spark_transfer_data)} input sparks...")
        successfully_prepared_souls = []

        for i, spark_data in enumerate(input_spark_transfer_data):
            spark_id = spark_data.get("spark_id", f"unknown_spark_{i}")
            logger.info(f"--- Processing Spark {i+1}/{len(input_spark_transfer_data)} (ID: {spark_id}) ---")

            try:
                # 1. Receive Spark
                reception_info = self.receive_spark_from_void(spark_data)

                # 2. Strengthen Soul
                strengthening_result = self.strengthen_soul(spark_id, iterations=iterations_per_spark)

                # 3. Finalize Soul (Can raise ValueError if quality too low)
                finalized_soul_data = self.finalize_soul(spark_id)

                # 4. Prepare for Transfer
                transfer_data = self.prepare_soul_for_transfer(spark_id)
                successfully_prepared_souls.append(transfer_data)
                logger.info(f"--- Successfully processed Spark {spark_id} ---")

            except ValueError as ve:
                # Catch specific errors like low quality during finalization or invalid input
                logger.error(f"Skipping Spark {spark_id} due to ValueError: {ve}")
                # Continue to the next spark
            except (RuntimeError, TypeError, AttributeError, IOError) as rte:
                # Catch critical processing errors for this specific spark
                logger.error(f"Critical error processing Spark {spark_id}. Skipping. Error: {rte}", exc_info=True)
                # Continue to the next spark
            except Exception as e:
                # Catch any other unexpected errors for this spark
                logger.error(f"Unexpected error processing Spark {spark_id}. Skipping. Error: {e}", exc_info=True)
                # Continue to the next spark

            # Visualize field state after processing each spark (optional)
            if VISUALIZATION_ENABLED:
                self.visualize_guff_field(save=True, show=False, filename=f"guff_field_after_spark_{spark_id}.png")

        # Optional: Persist final metrics state for the controller run
        try: metrics.persist_metrics()
        except Exception as e: logger.error(f"Final metrics persistence failed for Guff stage: {e}")

        logger.info(f"Guff stage processing finished. {len(successfully_prepared_souls)} souls prepared for transfer.")
        return successfully_prepared_souls

    def visualize_guff_field(self, save: bool = False, show: bool = False, filename: Optional[str] = None) -> bool:
        """
        Visualize the current state of the Guff field (3D slice).

        Args:
            save (bool): Whether to save the visualization.
            show (bool): Whether to display the visualization (ignored if matplotlib unavailable).
            filename (Optional[str]): Custom filename for saving (relative to controller output dir).

        Returns:
            bool: True if visualization was attempted successfully, False otherwise.
        """
        if not VISUALIZATION_ENABLED:
            logger.warning("Visualization requested but Matplotlib is not available.")
            return False

        save_path = os.path.join(self.output_dir, filename) if save and filename else None
        if save and not filename: # Auto-generate filename
             step = self.simulation_step # Use controller step for filename
             filename = f"guff_field_viz_step_{step}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
             save_path = os.path.join(self.output_dir, filename)

        logger.debug(f"Attempting to visualize Guff field slice (Show={show}, Save Path={save_path})")
        fig = None
        try:
            # Ask the field to generate the plot figure (using its slice visualizer)
            if hasattr(self.guff_field, 'visualize_field_slice'):
                fig = self.guff_field.visualize_field_slice(
                    axis='z', # Default slice
                    slice_index=None, # Default middle slice
                    show=False, # Controller handles showing/saving
                    save_path=save_path # Pass path if saving
                )
                if fig is None and save_path and os.path.exists(save_path):
                     logger.info(f"Guff field slice visualization saved by field method to {save_path}")
                     return True
                elif fig is None:
                     logger.warning("Field visualization method returned None.")
                     return False
            else:
                 logger.warning("GuffField3D instance lacks 'visualize_field_slice' method.")
                 return False

            # If field method returns figure and doesn't save directly:
            # if save_path and fig: # Already saved by field method if path given
            #      fig.savefig(save_path, dpi=300, bbox_inches='tight')
            #      logger.info(f"Guff field slice visualization saved to {save_path}")

            if show and fig:
                plt.show()
            elif fig:
                plt.close(fig)

            return True

        except Exception as e:
            logger.error(f"Error during Guff field visualization: {e}", exc_info=True)
            if fig: plt.close(fig)
            return False

    def visualize_formation(self, spark_id: str, save: bool = False, show: bool = False) -> bool:
        """
        Visualize the formation state for a specific spark using the field's visualizer.

        Args:
            spark_id (str): The ID of the spark/soul to visualize.
            save (bool): Whether to save the visualization.
            show (bool): Whether to display the visualization (ignored if matplotlib unavailable).

        Returns:
            bool: True if visualization was attempted successfully, False otherwise.
        """
        if not VISUALIZATION_ENABLED:
            logger.warning("Visualization requested but Matplotlib is not available.")
            return False
        if not spark_id or not isinstance(spark_id, str):
            logger.error("Invalid spark_id provided for visualization.")
            return False

        # Get the relevant data for visualization (use finalized if available, else strengthening)
        soul_data_for_viz = None
        if spark_id in self.finalized_soul_data:
            soul_data_for_viz = self.finalized_soul_data[spark_id]
        elif spark_id in self.strengthening_result_data:
            soul_data_for_viz = self.strengthening_result_data[spark_id]
        else:
            logger.error(f"Cannot visualize formation for spark {spark_id}: No finalized or strengthening data found.")
            return False

        save_path = os.path.join(self.output_dir, f"guff_formation_{spark_id}.png") if save else None
        logger.debug(f"Attempting to visualize Guff formation for {spark_id} (Show={show}, Save Path={save_path})")
        fig = None
        try:
            if hasattr(self.guff_field, 'visualize_formation'):
                fig = self.guff_field.visualize_formation(
                    soul_data=soul_data_for_viz,
                    show=False, # Controller handles show/save
                    save_path=save_path
                )
                if fig is None and save_path and os.path.exists(save_path):
                     logger.info(f"Guff formation visualization saved by field method to {save_path}")
                     return True
                elif fig is None:
                     logger.warning("Field formation visualization method returned None.")
                     return False
            else:
                 logger.warning("GuffField3D instance lacks 'visualize_formation' method.")
                 return False

            # If field method returns figure and doesn't save directly:
            # if save_path and fig: # Saved by field method
            #     fig.savefig(save_path, dpi=300, bbox_inches='tight')
            #     logger.info(f"Guff formation visualization saved to {save_path}")

            if show and fig:
                plt.show()
            elif fig:
                plt.close(fig)

            return True

        except Exception as e:
            logger.error(f"Error during Guff formation visualization for {spark_id}: {e}", exc_info=True)
            if fig: plt.close(fig)
            return False

    def save_controller_state(self, filename: Optional[str] = None) -> bool:
        """
        Save the controller's current state. Fails hard on error.

        Args:
            filename (Optional[str]): Filename for the state file (JSON). Auto-generated if None.

        Returns:
            bool: True if saving was successful.

        Raises:
            IOError: If saving the state file fails.
            RuntimeError: For other unexpected errors.
        """
        if filename is None:
            filename = f"guff_controller_state_{self.controller_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        elif not filename.lower().endswith('.json'):
             filename += '.json'

        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving Guff controller state to {save_path}...")

        try:
            # Get current metrics relevant to the controller
            controller_metrics = metrics.get_category_metrics(CONTROLLER_METRIC_CATEGORY)

            state_data = {
                'controller_id': self.controller_id,
                'creation_time': self.creation_time,
                'field_uuid': self.guff_field.uuid,
                'field_dimensions': list(self.field_dimensions),
                'output_dir': self.output_dir,
                'simulation_step': self.simulation_step,
                # Store IDs of processed sparks/souls at different stages
                'processing_spark_ids': list(self.processing_spark_data.keys()),
                'reception_info_ids': list(self.reception_info_data.keys()),
                'strengthening_result_ids': list(self.strengthening_result_data.keys()),
                'finalized_soul_ids': list(self.finalized_soul_data.keys()),
                'transfer_ready_ids': list(self.transfer_ready_data.keys()),
                'last_metrics_snapshot': controller_metrics,
                'save_timestamp': datetime.now().isoformat()
            }
            # Note: We are NOT saving the actual data dicts here, only IDs for traceability.
            # The actual spark/soul/result data should be saved individually when generated.

            with open(save_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.info("Guff controller state saved successfully.")
            return True
        except IOError as e:
            logger.error(f"IOError saving Guff controller state to {save_path}: {e}", exc_info=True)
            raise IOError(f"Failed to write Guff controller state file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error saving Guff controller state: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save Guff controller state: {e}") from e

    def get_transfer_ready_souls(self) -> List[Dict[str, Any]]:
        """
        Get the transfer data dictionaries for all souls ready for the Sephiroth stage.

        Returns:
            List[Dict[str, Any]]: A list of transfer data dictionaries.
        """
        # Returns copies of the data stored in transfer_ready_data
        return [data.copy() for data in self.transfer_ready_data.values()]

    def __str__(self) -> str:
        """String representation of the Guff Field Controller."""
        return (f"GuffController(ID: {self.controller_id}, Dims: {self.field_dimensions}, "
                f"Step: {self.simulation_step}, Processing: {len(self.processing_spark_data)}, "
                f"Strengthened: {len(self.strengthening_result_data)}, Ready: {len(self.transfer_ready_data)})")

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"<GuffController id='{self.controller_id}' field_uuid='{self.guff_field.uuid}' "
                f"dims={self.field_dimensions} step={self.simulation_step}>")


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Guff Field Controller Module Example...")
    controller = None
    try:
        # Assume Void stage produced some spark transfer data (create dummy data)
        dummy_spark_transfer_data = [
            {
                "spark_id": str(uuid.uuid4()),
                "position": (10, 15, 20),
                "energy": 0.85,
                "stability": 0.92,
                "formation_score": 0.88,
                "void_field_metrics": {"dimensions": (64, 64, 64), "max_energy": 1.0},
                "transfer_complete": True,
                "transfer_timestamp": datetime.now().isoformat(),
                # ... other potential keys from Void transfer ...
            },
             {
                "spark_id": str(uuid.uuid4()),
                "position": (45, 50, 30),
                "energy": 0.78,
                "stability": 0.85,
                "formation_score": 0.81,
                "void_field_metrics": {"dimensions": (64, 64, 64), "max_energy": 1.0},
                "transfer_complete": True,
                "transfer_timestamp": datetime.now().isoformat(),
            }
        ]

        # Initialize Controller
        controller = GuffController(
            dimensions=(64, 64, 64),
            field_name="guff_example_field",
            data_dir="output/guff_controller_example"
        )
        print(controller)

        # Run the full Guff stage for the dummy sparks
        prepared_souls_data = controller.run_full_guff_stage(
            input_spark_transfer_data=dummy_spark_transfer_data,
            iterations_per_spark=10 # Fewer iterations for example
        )

        print(f"\nGuff stage processing finished. {len(prepared_souls_data)} souls prepared for transfer.")

        if prepared_souls_data:
            print("\nData for first prepared soul:")
            # Pretty print the first dictionary
            print(json.dumps(prepared_souls_data[0], indent=2, default=str))

        # Save controller state
        controller.save_controller_state()

        print("\nGuff Field Controller Example Finished Successfully.")

    except (ValueError, TypeError, IOError, RuntimeError, ImportError) as e:
        print(f"\n--- ERROR during Guff Field Controller Example ---")
        print(f"An error occurred: {type(e).__name__}: {e}")
        if controller: print(f"Controller state at error: {controller}")
        import traceback
        traceback.print_exc()
        print("--------------------------------------------------")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during Guff Field Controller Example ---")
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        if controller: print(f"Controller state at error: {controller}")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------------------------")


# --- END OF FILE guff_controller.py ---