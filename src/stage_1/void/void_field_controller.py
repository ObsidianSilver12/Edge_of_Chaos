# --- START OF FILE void_field_controller.py ---

"""
Void Field Controller (Refactored for 3D and Robustness)

Controls the void field operations and soul spark formation process in 3D.
Orchestrates pattern embedding, quantum fluctuations, and spark detection,
enforcing strict error handling and validation.

Author: Soul Development Framework Team
"""

import logging
import os
import numpy as np
import sys
import uuid
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# --- Constants ---
try:
    from src.constants import (
        DEFAULT_DIMENSIONS_3D, LOG_LEVEL, LOG_FORMAT, DATA_DIR_BASE,
        VOID_WELL_QUALITY_THRESHOLD # Example threshold for well selection
    )
except ImportError as e:
    # Basic logging setup if constants failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. VoidFieldController cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    # Assumes VoidField is refactored to VoidField3D in the correct path
    from stage_1.void.void_field import VoidField3D
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import VoidField3D: {e}. VoidFieldController cannot function.")
    raise ImportError(f"Core dependency VoidField3D missing: {e}") from e

try:
    # Import the refactored metrics tracking module
    import metrics_tracking as metrics
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import metrics_tracking: {e}. VoidFieldController cannot function.")
    raise ImportError(f"Core dependency metrics_tracking missing: {e}") from e

# Conditional import for visualization
try:
    import matplotlib.pyplot as plt
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    logging.warning("Matplotlib not found. Visualization methods will be disabled.")

# Configure logging
log_file_path = os.path.join("logs", "void_field_controller.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('void_field_controller')

# Define controller-specific metric category
CONTROLLER_METRIC_CATEGORY = "void_controller"

class VoidFieldController:
    """
    Controller for 3D void field operations and soul spark formation.
    Manages pattern embedding, fluctuations, well identification, and spark creation.
    """

    def __init__(self, dimensions: Tuple[int, int, int] = DEFAULT_DIMENSIONS_3D,
                 field_name: str = "void_field", data_dir: str = DATA_DIR_BASE,
                 controller_id: Optional[str] = None):
        """
        Initialize a new Void Field Controller. Fails hard on invalid configuration.

        Args:
            dimensions (Tuple[int, int, int]): Dimensions of the void field.
            field_name (str): Name for the associated VoidField3D instance.
            data_dir (str): Base directory for field data.
            controller_id (Optional[str]): Specific ID for the controller, generates if None.

        Raises:
            ValueError: If dimensions, field_name, or data_dir are invalid.
            RuntimeError: If VoidField3D initialization fails.
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
        self.output_dir: str = os.path.join(data_dir, "controller_data", f"void_{self.controller_id}") # Specific output dir

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create controller output directory {self.output_dir}: {e}")
            raise # Fail hard

        logger.info(f"Initializing Void Field Controller (ID: {self.controller_id})")

        # Initialize the void field (raises errors on failure)
        try:
            self.void_field: VoidField3D = VoidField3D(
                dimensions=self.field_dimensions,
                field_name=field_name,
                data_dir=data_dir # Pass base data dir to field
            )
        except (ValueError, RuntimeError, ImportError) as e:
            logger.critical(f"CRITICAL: Failed to initialize VoidField3D: {e}", exc_info=True)
            raise RuntimeError(f"VoidField3D initialization failed: {e}") from e

        # --- State Tracking ---
        self.formed_sparks: List[Dict[str, Any]] = [] # Stores spark *data* dicts
        self.potential_wells: List[Dict[str, Any]] = [] # Stores well *data* dicts
        self.simulation_step: int = 0
        self.patterns_embedded: bool = False
        self.wells_identified: bool = False

        # Record initial controller state metric
        metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
            'status': 'initialized',
            'timestamp': self.creation_time,
            'field_uuid': self.void_field.uuid,
            'dimensions': list(self.field_dimensions), # Convert tuple for JSON
            'simulation_step': self.simulation_step
        })

        logger.info(f"Void Field Controller '{self.controller_id}' initialized successfully.")
        logger.info(f"Associated Void Field: '{self.void_field.field_name}' (UUID: {self.void_field.uuid})")

    def embed_sacred_patterns(self, geometry_type: str = "flower_of_life", strength: float = 1.0) -> Dict[str, Any]:
        """
        Embed sacred geometry patterns in the void field. Fails hard on error.

        Args:
            geometry_type (str): Type of geometry to embed (e.g., "flower_of_life").
            strength (float): Strength of the embedding (0.0 to 1.0).

        Returns:
            Dict[str, Any]: Information about embedded patterns from the field.

        Raises:
            RuntimeError: If the embedding process fails in the VoidField3D.
            ValueError: If strength is invalid.
        """
        if not (isinstance(strength, (int, float)) and 0.0 <= strength <= 1.0):
            raise ValueError(f"Embedding strength must be between 0.0 and 1.0, got {strength}")

        logger.info(f"Controller initiating embedding of '{geometry_type}' pattern...")
        try:
            # Delegate to the field, which should handle errors/validation
            # Assumes embed_sacred_geometry returns a result dict on success
            embed_result = self.void_field.embed_sacred_geometry(geometry_type, strength)
            self.patterns_embedded = True

            # Record metrics
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'embed_patterns',
                'geometry_type': geometry_type,
                'strength': strength,
                'patterns_embedded': self.patterns_embedded,
                'timestamp': datetime.now().isoformat()
            })
            # Persist metrics after significant action (optional, depends on strategy)
            # try: metrics.persist_metrics() except Exception as e: logger.error(f"Periodic metrics persistence failed: {e}")

            # Visualize (conditional)
            self.visualize_void_field(save=True, show=False, filename=f"void_field_after_pattern_{geometry_type}.png")

            logger.info(f"Sacred geometry pattern '{geometry_type}' embedded successfully.")
            return embed_result # Return result from field method

        except (ValueError, RuntimeError, AttributeError) as e: # Catch errors from field
            logger.error(f"Failed to embed sacred patterns '{geometry_type}': {e}", exc_info=True)
            raise RuntimeError(f"Pattern embedding failed: {e}") from e # Fail hard

    def identify_potential_wells(self, threshold_factor: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Identify potential wells in the void field. Fails hard on error.

        Args:
            threshold_factor (Optional[float]): Factor relative to mean energy for thresholding.
                                               Uses field's default if None.

        Returns:
            List[Dict[str, Any]]: List of identified potential wells (dictionaries).

        Raises:
            RuntimeError: If patterns haven't been embedded or well identification fails.
        """
        if not self.patterns_embedded:
            raise RuntimeError("Cannot identify wells: Sacred patterns must be embedded first.")

        logger.info("Controller initiating potential well identification...")
        try:
            # Delegate to the field
            # Pass threshold_factor if provided, otherwise field uses its default
            wells_kwargs = {}
            if threshold_factor is not None:
                 if not (isinstance(threshold_factor, (int, float)) and threshold_factor > 0):
                      raise ValueError(f"Invalid threshold_factor: {threshold_factor}")
                 wells_kwargs['threshold_factor'] = threshold_factor

            self.potential_wells = self.void_field.identify_potential_wells(**wells_kwargs) # Returns list of dicts
            self.wells_identified = True

            # Record metrics
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'identify_wells',
                'wells_identified_count': len(self.potential_wells),
                'wells_identified': self.wells_identified,
                'threshold_factor_used': threshold_factor if threshold_factor is not None else 'field_default',
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"Identified {len(self.potential_wells)} potential wells.")
            return self.potential_wells.copy() # Return a copy

        except (ValueError, RuntimeError, AttributeError) as e: # Catch errors from field
            logger.error(f"Failed to identify potential wells: {e}", exc_info=True)
            self.wells_identified = False # Reset flag on failure
            self.potential_wells = []
            raise RuntimeError(f"Potential well identification failed: {e}") from e # Fail hard


    def simulate_quantum_fluctuations(self, iterations: int = 10, amplitude: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Simulate quantum fluctuations and check for spark formation. Fails hard on error.

        Args:
            iterations (int): Number of simulation iterations.
            amplitude (Optional[float]): Fluctuation amplitude. Uses field's default if None.

        Returns:
            List[Dict[str, Any]]: List of sparks formed during these iterations (dictionaries).

        Raises:
            RuntimeError: If wells haven't been identified or simulation fails.
            ValueError: If iterations is invalid.
            IOError: If saving spark data fails.
        """
        if not self.wells_identified:
            raise RuntimeError("Cannot simulate fluctuations: Potential wells must be identified first.")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Iterations must be a positive integer.")

        logger.info(f"Controller initiating {iterations} quantum fluctuation steps...")
        try:
            # Delegate to the field, handle amplitude override
            sim_kwargs = {'iterations': iterations}
            if amplitude is not None:
                if not (isinstance(amplitude, (int, float)) and amplitude >= 0):
                    raise ValueError("Fluctuation amplitude must be non-negative.")
                sim_kwargs['amplitude'] = amplitude

            # simulate_quantum_fluctuations in field now returns metrics including high energy points
            # check_spark_formation is separate
            sim_metrics = self.void_field.simulate_quantum_fluctuations(**sim_kwargs)

            # Now check for spark formation based on the updated field state
            newly_formed_spark_data: List[Dict[str, Any]] = []
            # Loop through identified wells (or re-identify if needed) to check thresholds
            # For simplicity, let's assume the field has a method `check_spark_formation`
            # that checks all current wells after fluctuations.
            if hasattr(self.void_field, 'check_spark_formation'):
                # This method should return spark data dict if formed, else None/False/empty dict
                spark_check_result = self.void_field.check_spark_formation()
                if isinstance(spark_check_result, dict) and spark_check_result.get("spark_formed"):
                    # Assume it returns data for *one* spark per check for simplicity,
                    # or modify field to return a list if multiple can form at once.
                    newly_formed_spark_data.append(spark_check_result)
                    logger.info(f"Spark formed during fluctuation check: ID {spark_check_result.get('spark_id')}")
                elif isinstance(spark_check_result, list): # If field returns a list
                    newly_formed_spark_data.extend(spark_check_result)
                    if spark_check_result: logger.info(f"{len(spark_check_result)} sparks formed during check.")

            else:
                logger.warning("VoidField3D instance lacks 'check_spark_formation' method. Cannot check for sparks.")


            # Process newly formed sparks
            for spark_data in newly_formed_spark_data:
                self.formed_sparks.append(spark_data)
                # Record metrics for this specific spark
                metrics.record_metrics('spark_formation', { # Use a dedicated category
                    'spark_id': spark_data.get('spark_id'),
                    'position': list(spark_data.get('position', [-1,-1,-1])),
                    'energy': spark_data.get('energy'),
                    'stability': spark_data.get('stability'),
                    'formation_score': spark_data.get('formation_score'),
                    'timestamp': spark_data.get('timestamp')
                })
                # Save spark data (raises IOError on failure)
                self._save_spark_data(spark_data)

            self.simulation_step += iterations

            # Record controller-level metrics for this step
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'simulate_fluctuations',
                'iterations': iterations,
                'amplitude_used': amplitude if amplitude is not None else 'field_default',
                'sparks_formed_this_step': len(newly_formed_spark_data),
                'total_sparks_formed': len(self.formed_sparks),
                'simulation_step': self.simulation_step,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"Fluctuation simulation step complete. {len(newly_formed_spark_data)} new sparks formed.")
            return newly_formed_spark_data # Return only sparks formed in this batch

        except (ValueError, RuntimeError, AttributeError, IOError) as e: # Catch errors from field/saving
            logger.error(f"Failed during quantum fluctuation simulation: {e}", exc_info=True)
            raise RuntimeError(f"Fluctuation simulation failed: {e}") from e # Fail hard


    def _save_spark_data(self, spark_data: Dict[str, Any]) -> bool:
        """Save a formed soul spark's data to file. Fails hard on error."""
        if not isinstance(spark_data, dict):
            raise TypeError("spark_data must be a dictionary.")
        spark_id = spark_data.get('spark_id')
        if not spark_id or not isinstance(spark_id, str):
            raise ValueError("spark_data dictionary missing valid 'spark_id'.")

        filename = f"spark_{spark_id}.json" # Use spark ID for filename
        save_path = os.path.join(self.output_dir, filename)

        logger.debug(f"Saving spark data for {spark_id} to {save_path}")
        try:
            with open(save_path, 'w') as f:
                # Convert numpy types if necessary for JSON
                def default_serializer(o):
                    if isinstance(o, np.integer): return int(o)
                    if isinstance(o, np.floating): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    if isinstance(o, (datetime, uuid.UUID)): return str(o)
                    # Add other types as needed
                    # Let JSONEncoder handle serializable types, raise for others
                    return json.JSONEncoder().default(o)

                json.dump(spark_data, f, indent=2, default=default_serializer)

            logger.info(f"Soul spark data saved to {save_path}")
            return True
        except (IOError, TypeError) as e:
            logger.error(f"Error saving soul spark data to {save_path}: {e}", exc_info=True)
            raise IOError(f"Failed to save spark data file: {e}") from e # Fail hard

    def run_full_spark_formation(self, target_sparks: int = 3, max_iterations: int = 100,
                                 iterations_per_batch: int = 10) -> List[Dict[str, Any]]:
        """
        Run the complete soul spark formation process. Fails hard on errors.

        Args:
            target_sparks (int): Target number of sparks to form.
            max_iterations (int): Maximum total simulation iterations.
            iterations_per_batch (int): Iterations per simulation step.

        Returns:
            List[Dict[str, Any]]: List of all formed soul spark data dictionaries.

        Raises:
            ValueError: If target_sparks, max_iterations, or iterations_per_batch are invalid.
            RuntimeError: If any core step (embedding, wells, fluctuations) fails critically.
        """
        if not isinstance(target_sparks, int) or target_sparks <= 0:
            raise ValueError("target_sparks must be a positive integer.")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(iterations_per_batch, int) or iterations_per_batch <= 0:
            raise ValueError("iterations_per_batch must be a positive integer.")

        logger.info(f"Starting full spark formation process: Target={target_sparks}, Max Iter={max_iterations}")

        # --- Step 1: Embed Patterns (raises RuntimeError on failure) ---
        if not self.patterns_embedded:
            self.embed_sacred_patterns()

        # --- Step 2: Identify Wells (raises RuntimeError on failure) ---
        if not self.wells_identified:
            self.identify_potential_wells()
        if not self.potential_wells:
             logger.warning("No potential wells identified after embedding. Cannot form sparks.")
             return [] # Return empty list, not an error state necessarily

        # --- Step 3: Simulate Fluctuations Loop ---
        total_iterations = 0
        while len(self.formed_sparks) < target_sparks and total_iterations < max_iterations:
            current_batch_size = min(iterations_per_batch, max_iterations - total_iterations)
            if current_batch_size <= 0: break # Safety check

            logger.info(f"Running fluctuation batch: {current_batch_size} iterations "
                        f"(Total: {total_iterations}/{max_iterations}, Sparks: {len(self.formed_sparks)}/{target_sparks})")

            # Simulate (raises RuntimeError on failure)
            new_sparks = self.simulate_quantum_fluctuations(iterations=current_batch_size)
            total_iterations += current_batch_size

            # Visualize periodically
            if VISUALIZATION_ENABLED and (total_iterations % (iterations_per_batch * 5) == 0 or new_sparks):
                self.visualize_void_field(save=True, show=False, filename=f"void_field_iter_{total_iterations}.png")

        # --- Finalization ---
        if VISUALIZATION_ENABLED:
            self.visualize_void_field(save=True, show=False, filename="void_field_final.png")

        # Optional: Persist final metrics state
        try: metrics.persist_metrics()
        except Exception as e: logger.error(f"Final metrics persistence failed: {e}")

        logger.info(f"Spark formation process complete: {len(self.formed_sparks)} sparks formed after {total_iterations} iterations.")
        return self.formed_sparks.copy() # Return a copy

    def visualize_void_field(self, save: bool = False, show: bool = False, filename: Optional[str] = None) -> bool:
        """
        Visualize the current state of the void field (3D).

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
        if save and not filename: # Auto-generate filename if saving enabled but no name given
             filename = f"void_field_viz_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
             save_path = os.path.join(self.output_dir, filename)

        logger.debug(f"Attempting to visualize void field (Show={show}, Save Path={save_path})")
        fig = None
        try:
            # Ask the field to generate the plot figure
            # Use the 3D visualizer
            if hasattr(self.void_field, 'visualize_field_3d'):
                # Pass relevant info for overlays
                well_positions = [w['position'] for w in self.potential_wells]
                spark_positions = [s['position'] for s in self.formed_sparks]

                fig = self.void_field.visualize_field_3d(
                    threshold=0.4, # Example threshold
                    show_wells=bool(well_positions),
                    well_positions=well_positions,
                    show_sparks=bool(spark_positions),
                    spark_positions=spark_positions,
                    title=f"Void Field State (Step {self.simulation_step})",
                    save_path=save_path # Pass save path directly to field method if supported
                )
                if fig is None and save_path and os.path.exists(save_path):
                     # Field method might save directly
                     logger.info(f"Void field 3D visualization saved by field method to {save_path}")
                     return True
                elif fig is None:
                     logger.warning("Field visualization method returned None.")
                     return False

            else:
                 logger.warning("VoidField3D instance lacks 'visualize_field_3d' method.")
                 return False

            # If field method returns figure and doesn't save directly:
            if save_path and fig:
                 fig.savefig(save_path, dpi=300, bbox_inches='tight')
                 logger.info(f"Void field 3D visualization saved to {save_path}")

            if show and fig:
                plt.show()
            elif fig: # Close figure if not shown
                plt.close(fig)

            return True # Visualization attempted

        except Exception as e:
            logger.error(f"Error during void field visualization: {e}", exc_info=True)
            if fig: plt.close(fig) # Close figure on error
            return False

    def save_controller_state(self, filename: Optional[str] = None) -> bool:
        """
        Save the controller's current state (excluding the full field arrays).
        Fails hard on error.

        Args:
            filename (Optional[str]): Filename for the state file (JSON). Auto-generated if None.

        Returns:
            bool: True if saving was successful.

        Raises:
            IOError: If saving the state file fails.
        """
        if filename is None:
            filename = f"void_controller_state_{self.controller_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        elif not filename.lower().endswith('.json'):
             filename += '.json'

        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving controller state to {save_path}...")

        try:
            # Get current metrics relevant to the controller
            controller_metrics = metrics.get_category_metrics(CONTROLLER_METRIC_CATEGORY)

            state_data = {
                'controller_id': self.controller_id,
                'creation_time': self.creation_time,
                'field_uuid': self.void_field.uuid, # Link to field state
                'field_dimensions': list(self.field_dimensions),
                'output_dir': self.output_dir,
                'simulation_step': self.simulation_step,
                'patterns_embedded': self.patterns_embedded,
                'wells_identified': self.wells_identified,
                'potential_wells_count': len(self.potential_wells),
                 # Optionally save well IDs or positions if needed for reload logic
                'potential_wells_brief': [{'id':w.get('id'), 'pos':w.get('position')} for w in self.potential_wells[:20]], # Limit saved wells
                'formed_sparks_count': len(self.formed_sparks),
                'formed_sparks_ids': [s.get('spark_id') for s in self.formed_sparks], # Save IDs
                'last_metrics_snapshot': controller_metrics,
                'save_timestamp': datetime.now().isoformat()
            }

            with open(save_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str) # Use default=str for non-serializables

            logger.info("Controller state saved successfully.")
            return True
        except IOError as e:
            logger.error(f"IOError saving controller state to {save_path}: {e}", exc_info=True)
            raise IOError(f"Failed to write controller state file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error saving controller state: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save controller state: {e}") from e # Convert other errors


    def get_formed_sparks(self) -> List[Dict[str, Any]]:
        """
        Get data dictionaries for all formed soul sparks.

        Returns:
            List[Dict[str, Any]]: A copy of the list of formed spark data dictionaries.
        """
        return self.formed_sparks.copy()

    def get_best_spark(self) -> Optional[Dict[str, Any]]:
        """
        Get data dictionary for the highest quality soul spark formed so far.

        Returns:
            Optional[Dict[str, Any]]: The spark data dictionary, or None if no sparks formed.
        """
        if not self.formed_sparks:
            return None

        try:
            # Sort sparks by formation score (assuming higher is better)
            # Add safety check for missing key
            sorted_sparks = sorted(
                self.formed_sparks,
                key=lambda s: s.get('formation_score', 0.0) if isinstance(s, dict) else 0.0,
                reverse=True
            )
            return sorted_sparks[0].copy() # Return a copy
        except Exception as e:
             logger.error(f"Error finding best spark: {e}", exc_info=True)
             return None # Return None if sorting/access fails

    def __str__(self) -> str:
        """String representation of the Void Field Controller."""
        return (f"VoidFieldController(ID: {self.controller_id}, Dims: {self.field_dimensions}, "
                f"Step: {self.simulation_step}, Wells: {len(self.potential_wells)}, Sparks: {len(self.formed_sparks)})")

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"<VoidFieldController id='{self.controller_id}' field_uuid='{self.void_field.uuid}' "
                f"dims={self.field_dimensions} step={self.simulation_step} "
                f"patterns={self.patterns_embedded} wells={self.wells_identified}>")

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Void Field Controller Module Example...")
    controller = None # Initialize controller to None
    try:
        # Use smaller dimensions for quick testing
        controller = VoidFieldController(
            dimensions=(32, 32, 32), # Smaller size for example
            field_name="void_example_field",
            data_dir="output/void_controller_example"
        )
        print(controller)

        # Run the full formation process
        formed_sparks_list = controller.run_full_spark_formation(
            target_sparks=2,       # Aim for fewer sparks for example
            max_iterations=50,     # Limit iterations
            iterations_per_batch=5
        )
        print(f"\nFormation process completed. Formed {len(formed_sparks_list)} sparks.")

        # Get the best spark data
        best_spark_data = controller.get_best_spark()
        if best_spark_data:
            print("\nBest Soul Spark Formed:")
            print(f"  ID: {best_spark_data.get('spark_id')}")
            print(f"  Position: {best_spark_data.get('position')}")
            print(f"  Formation Score: {best_spark_data.get('formation_score', 'N/A'):.4f}")

        # Save controller state
        controller.save_controller_state()

        print("\nVoid Field Controller Example Finished Successfully.")

    except (ValueError, TypeError, IOError, RuntimeError, ImportError) as e:
        print(f"\n--- ERROR during Void Field Controller Example ---")
        print(f"An error occurred: {type(e).__name__}: {e}")
        if controller: print(f"Controller state at error: {controller}")
        import traceback
        traceback.print_exc()
        print("--------------------------------------------------")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during Void Field Controller Example ---")
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        if controller: print(f"Controller state at error: {controller}")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------------------------")


# --- END OF FILE void_field_controller.py ---