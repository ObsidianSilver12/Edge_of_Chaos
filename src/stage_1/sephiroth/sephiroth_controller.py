# --- START OF FILE sephiroth_controller.py ---

"""
Sephiroth Controller (Refactored for 3D and Orchestration)

Manages the creation, access, and orchestration of interactions within the
Sephiroth dimensional fields. Guides the soul's journey through the Tree of Life.
Enforces strict error handling and delegates field logic to SephirothField3D instances.

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
        DEFAULT_DIMENSIONS_3D, LOG_LEVEL, LOG_FORMAT, DATA_DIR_BASE
    )
except ImportError as e:
    # Basic logging setup if constants failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. SephirothController cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    # Assumes SephirothField is refactored to SephirothField3D in the correct path
    from stage_1.sephiroth.sephiroth_field import SephirothField3D
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import SephirothField3D: {e}. SephirothController cannot function.")
    raise ImportError(f"Core dependency SephirothField3D missing: {e}") from e

try:
    # Import the aspect dictionary (essential for names and properties)
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    if aspect_dictionary is None: # Check if aspect_dictionary itself failed init
         raise ImportError("aspect_dictionary failed to initialize.")
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import or initialize aspect_dictionary: {e}. SephirothController cannot function.")
    raise ImportError(f"Core dependency aspect_dictionary missing or failed: {e}") from e

try:
    # Import the refactored metrics tracking module
    import metrics_tracking as metrics
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import metrics_tracking: {e}. SephirothController cannot function.")
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
log_file_path = os.path.join("logs", "sephiroth_controller.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('sephiroth_controller')

# Define controller-specific metric category
CONTROLLER_METRIC_CATEGORY = "sephiroth_controller"

class SephirothController:
    """
    Controller for managing Sephiroth dimensional fields and orchestrating soul journeys.
    """

    def __init__(self, dimensions: Tuple[int, int, int] = DEFAULT_DIMENSIONS_3D,
                 data_dir: str = DATA_DIR_BASE, initialize_all: bool = False,
                 controller_id: Optional[str] = None):
        """
        Initialize the Sephiroth controller. Fails hard on invalid configuration.

        Args:
            dimensions (Tuple[int, int, int]): Dimensions for all Sephiroth fields.
            data_dir (str): Base directory for field data.
            initialize_all (bool): If True, create all Sephiroth fields on initialization.
                                   Otherwise, fields are created on demand via get_field.
            controller_id (Optional[str]): Specific ID for the controller, generates if None.

        Raises:
            ValueError: If dimensions or data_dir are invalid.
            RuntimeError: If aspect_dictionary is unavailable or field initialization fails.
            OSError: If the output directory cannot be created.
        """
        # --- Input Validation ---
        if not isinstance(dimensions, tuple) or len(dimensions) != 3 or not all(isinstance(d, int) and d > 0 for d in dimensions):
            raise ValueError(f"Dimensions must be a tuple of 3 positive integers, got {dimensions}")
        if not data_dir or not isinstance(data_dir, str):
            raise ValueError("Data directory must be a non-empty string.")
        if aspect_dictionary is None:
            raise RuntimeError("CRITICAL: SephirothAspectDictionary failed to initialize. Cannot create SephirothController.")

        self.controller_id: str = controller_id or str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.dimensions: Tuple[int, int, int] = dimensions
        self.data_dir: str = data_dir # Base data dir passed to fields
        # Controller's specific output dir for its state, logs, journey records, etc.
        self.output_dir: str = os.path.join(data_dir, "controller_data", f"sephiroth_{self.controller_id}")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create controller output directory {self.output_dir}: {e}")
            raise

        logger.info(f"Initializing Sephiroth Controller (ID: {self.controller_id})")

        # Dictionary to store Sephiroth field instances {sephirah_name: SephirothField3D}
        self.fields: Dict[str, SephirothField3D] = {}

        # Track journey progress for souls {soul_id: journey_data_dict}
        self.soul_journeys: Dict[str, Dict[str, Any]] = {}

        if initialize_all:
            try:
                self.initialize_all_fields()
            except Exception as e:
                # Log critical failure during bulk init but allow controller object creation?
                # Or fail hard? Let's fail hard, as it indicates a fundamental issue.
                logger.critical(f"CRITICAL: Failed during initialize_all_fields: {e}", exc_info=True)
                raise RuntimeError("Failed to initialize all Sephiroth fields during controller setup.") from e

        # Record initial controller state metric
        metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
            'status': 'initialized',
            'controller_id': self.controller_id,
            'timestamp': self.creation_time,
            'dimensions': list(self.dimensions),
            'initialized_fields_count': len(self.fields),
            'initialize_all': initialize_all
        })

        logger.info(f"Sephiroth Controller '{self.controller_id}' initialized successfully.")

    def initialize_all_fields(self):
        """
        Initialize all Sephiroth fields defined in the aspect dictionary.
        Fails hard if any field cannot be created.
        """
        logger.info("Controller initializing all Sephiroth fields...")
        sephiroth_names = aspect_dictionary.sephiroth_names
        if not sephiroth_names:
            raise RuntimeError("No Sephiroth names found in aspect_dictionary.")

        initialized_count = 0
        for sephirah_name in sephiroth_names:
            try:
                # get_field will create if not present and raises errors on failure
                self.get_field(sephirah_name)
                initialized_count += 1
            except (ValueError, RuntimeError, ImportError) as e:
                logger.critical(f"CRITICAL: Failed to initialize field for '{sephirah_name}' during initialize_all: {e}", exc_info=True)
                # Re-raise to halt the process if even one field fails init
                raise RuntimeError(f"Failed to initialize field '{sephirah_name}'") from e

        logger.info(f"All {initialized_count} Sephiroth fields initialized by controller.")

    def get_field(self, sephirah: str) -> SephirothField3D:
        """
        Get a specific Sephiroth field, creating it if it doesn't exist.
        Fails hard if the field cannot be created or retrieved.

        Args:
            sephirah (str): Name of the Sephirah (case-insensitive).

        Returns:
            SephirothField3D: The requested field instance.

        Raises:
            ValueError: If sephirah name is invalid.
            RuntimeError: If field creation fails.
        """
        sephirah_lower = sephirah.lower()
        if sephirah_lower not in aspect_dictionary.sephiroth_names:
            raise ValueError(f"Invalid Sephirah name: '{sephirah}'.")

        # Return existing field if already created
        if sephirah_lower in self.fields:
            return self.fields[sephirah_lower]

        # Create a new field instance
        logger.info(f"Creating new field instance for {sephirah_lower}...")
        try:
            # Pass necessary args to SephirothField3D constructor
            field = SephirothField3D(
                sephirah=sephirah_lower,
                dimensions=self.dimensions,
                data_dir=self.data_dir
            )
            self.fields[sephirah_lower] = field
            logger.info(f"Successfully created {sephirah_lower} field.")

            # Record metric for field creation
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'create_field',
                'sephirah': sephirah_lower,
                'field_uuid': field.uuid,
                'success': True,
                'timestamp': datetime.now().isoformat()
            })
            return field

        except (ValueError, RuntimeError, ImportError, AttributeError, IOError) as e:
            logger.critical(f"CRITICAL: Failed to create Sephiroth field for '{sephirah_lower}': {e}", exc_info=True)
            raise RuntimeError(f"Field creation failed for {sephirah_lower}: {e}") from e

    def get_sephiroth_journey_order(self, start_sephirah: str = "yesod", end_sephirah: str = "kether") -> List[str]:
        """
        Get a recommended order for the soul's journey through Sephiroth.
        Defaults to starting at Yesod and ending at Kether, following pillar ascent.

        Args:
            start_sephirah (str): The starting Sephirah (default: "yesod").
            end_sephirah (str): The ending Sephirah (default: "kether").

        Returns:
            List[str]: List of Sephiroth names in journey order.

        Raises:
            ValueError: If start or end sephirah are invalid.
        """
        start_lower = start_sephirah.lower()
        end_lower = end_sephirah.lower()

        if start_lower not in aspect_dictionary.sephiroth_names or end_lower not in aspect_dictionary.sephiroth_names:
            raise ValueError("Invalid start or end Sephirah provided for journey.")

        # Define pillar groups based on aspect dictionary (robustly)
        pillars = {'left': [], 'middle': [], 'right': []}
        positions = {}
        for name in aspect_dictionary.sephiroth_names:
             try:
                 instance = aspect_dictionary.load_aspect_instance(name)
                 pillar = getattr(instance, 'pillar', 'middle') # Default to middle if missing
                 position = getattr(instance, 'position', 99)   # Default high number if missing
                 if pillar in pillars: pillars[pillar].append(name)
                 else: logger.warning(f"Unknown pillar '{pillar}' for {name}, assigning to middle.") ; pillars['middle'].append(name)
                 positions[name] = position
             except Exception as e:
                  logger.error(f"Could not load pillar/position for {name}: {e}. Skipping.")
                  continue # Skip sephirah if loading fails

        # Sort pillars by position (ascending = higher up the tree)
        for pillar_list in pillars.values():
             pillar_list.sort(key=lambda s: positions.get(s, 99))

        # Simplified path: Middle -> Right -> Left -> Middle -> Crown
        # This is just one possible path logic. Can be customized.
        journey_order = []
        if start_lower == 'yesod':
            journey_order = ['yesod', 'hod', 'netzach', 'tiphareth', 'geburah', 'chesed', 'binah', 'chokmah', 'kether']
            # Remove Daath if present from default aspect list
            if 'daath' in journey_order: journey_order.remove('daath')
            # Ensure start/end are handled (they are in this specific case)
        else:
             # Fallback to simple position-based order if non-standard start
             all_seph = sorted(aspect_dictionary.sephiroth_names, key=lambda s: positions.get(s, 99))
             try:
                 start_idx = all_seph.index(start_lower)
                 journey_order = all_seph[start_idx:] # Simplistic path from start onwards
             except ValueError:
                  logger.error(f"Start sephirah {start_lower} not found in sorted list. Using default order.")
                  journey_order = all_seph # Default to full order

        # Ensure end sephirah is last if possible
        if end_lower in journey_order and journey_order[-1] != end_lower:
             journey_order.remove(end_lower)
             journey_order.append(end_lower)

        logger.info(f"Generated journey order ({start_sephirah} -> {end_sephirah}): {journey_order}")
        return journey_order

    def guide_soul_journey(self, soul_transfer_data: Dict[str, Any],
                           journey_order: Optional[List[str]] = None,
                           duration_per_sephirah: float = 10.0) -> Dict[str, Any]:
        """
        Guides a soul's journey through the Sephiroth fields. Fails hard on critical errors.

        Args:
            soul_transfer_data (Dict[str, Any]): The transfer data dictionary from GuffController.
                                                 Must contain 'spark_id', 'sephiroth_connections', etc.
            journey_order (Optional[List[str]]): Specific order of Sephiroth to visit.
                                                 If None, uses default order from Yesod to Kether.
            duration_per_sephirah (float): Time (simulation units) spent in each field.

        Returns:
            Dict[str, Any]: The final state of the soul data dictionary after the journey.

        Raises:
            TypeError: If soul_transfer_data is not a dictionary.
            ValueError: If soul_transfer_data is missing critical keys, duration invalid,
                        or journey order contains invalid Sephiroth.
            RuntimeError: If a field cannot be retrieved or transformation fails critically.
        """
        if not isinstance(soul_transfer_data, dict):
            raise TypeError("soul_transfer_data must be a dictionary.")
        soul_id = soul_transfer_data.get('spark_id')
        if not soul_id or not isinstance(soul_id, str):
            raise ValueError("soul_transfer_data missing valid 'spark_id'.")
        if not isinstance(duration_per_sephirah, (int, float)) or duration_per_sephirah <= 0:
             raise ValueError("duration_per_sephirah must be positive.")

        logger.info(f"Starting Sephiroth journey for soul {soul_id}...")

        # Use default journey order if none provided
        if journey_order is None:
            journey_order = self.get_sephiroth_journey_order()
        elif not isinstance(journey_order, list) or not journey_order:
            raise ValueError("Provided journey_order must be a non-empty list.")
        else: # Validate provided order
             for sephirah in journey_order:
                  if sephirah.lower() not in aspect_dictionary.sephiroth_names:
                       raise ValueError(f"Invalid Sephiroth '{sephirah}' in provided journey_order.")

        # Initialize journey tracking for this soul
        current_soul_data = soul_transfer_data.copy() # Work on a copy
        journey_path = []
        journey_start_time = datetime.now().isoformat()
        self.soul_journeys[soul_id] = {
            "soul_id": soul_id,
            "start_time": journey_start_time,
            "status": "started",
            "path_taken": journey_path,
            "initial_state": current_soul_data.get("sephiroth_connections", {}) # Example initial state marker
        }

        # Record initial journey metric
        metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
            'action': 'start_journey',
            'soul_id': soul_id,
            'journey_order': journey_order,
            'timestamp': journey_start_time
        })

        # Iterate through the journey
        for i, sephirah_name in enumerate(journey_order):
            logger.info(f"Soul {soul_id} entering {sephirah_name.capitalize()} (Step {i+1}/{len(journey_order)})...")
            step_start_time = datetime.now().isoformat()
            try:
                # Get the field instance (raises error if fails)
                field = self.get_field(sephirah_name)

                # Call the field's transform_soul method
                # Pass the *current* soul data dictionary
                transform_result = field.transform_soul(current_soul_data, duration=duration_per_sephirah)

                # Check result from transform_soul
                if not isinstance(transform_result, dict) or not transform_result.get("success"):
                    error_msg = transform_result.get("error", "Unknown transformation error")
                    logger.error(f"Transformation failed for soul {soul_id} in {sephirah_name}: {error_msg}")
                    # Option: Stop journey, or log and continue? Let's stop for now on explicit failure.
                    self.soul_journeys[soul_id]['status'] = 'failed'
                    self.soul_journeys[soul_id]['failure_reason'] = f"Transformation failed in {sephirah_name}: {error_msg}"
                    raise RuntimeError(f"Journey halted: Transformation failed in {sephirah_name}")

                # Update journey path record
                journey_path.append({
                    "sephirah": sephirah_name,
                    "entry_time": step_start_time,
                    "exit_time": datetime.now().isoformat(),
                    "duration": duration_per_sephirah,
                    "resonance_achieved": transform_result.get("resonance"),
                    "transformation_strength": transform_result.get("strength"),
                    "aspects_gained_count": len(transform_result.get("aspects_gained", [])),
                    "aspects_strengthened_count": len(transform_result.get("aspects_strengthened", []))
                })

                # Record metrics for this step
                metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                    'action': 'journey_step',
                    'soul_id': soul_id,
                    'sephirah': sephirah_name,
                    'step_number': i + 1,
                    'resonance': transform_result.get("resonance"),
                    'strength': transform_result.get("strength"),
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                })

                # Visualize field state after transformation (optional)
                if VISUALIZATION_ENABLED and (i % 3 == 0 or i == len(journey_order) - 1): # Visualize every 3 steps + end
                    self.visualize_sephiroth_field(sephirah_name, save=True, show=False,
                                                  filename=f"sephiroth_{sephirah_name}_soul_{soul_id}_step{i+1}.png")

            except (ValueError, RuntimeError, AttributeError, TypeError, IOError) as e:
                logger.critical(f"CRITICAL ERROR during soul {soul_id} journey at {sephirah_name}: {e}", exc_info=True)
                self.soul_journeys[soul_id]['status'] = 'failed'
                self.soul_journeys[soul_id]['failure_reason'] = f"Critical error in {sephirah_name}: {e}"
                # Re-raise as RuntimeError to halt the process
                raise RuntimeError(f"Journey failed for soul {soul_id} at {sephirah_name}: {e}") from e

        # Journey completed successfully
        journey_end_time = datetime.now().isoformat()
        self.soul_journeys[soul_id]['status'] = 'completed'
        self.soul_journeys[soul_id]['end_time'] = journey_end_time
        self.soul_journeys[soul_id]['final_state'] = self._get_soul_state_summary(current_soul_data) # Get summary

        # Save journey log
        self._save_journey_log(soul_id)

        # Record final journey metric
        metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
            'action': 'complete_journey',
            'soul_id': soul_id,
            'steps_completed': len(journey_path),
            'final_status': 'completed',
            'timestamp': journey_end_time
        })

        logger.info(f"Sephiroth journey completed successfully for soul {soul_id}.")
        return current_soul_data # Return the final, modified soul data dictionary

    def _get_soul_state_summary(self, soul_data: Dict[str, Any]) -> Dict[str, Any]:
         """Extracts key summary info from the potentially large soul data dict."""
         if not isinstance(soul_data, dict): return {}
         return {
             "aspect_count": len(soul_data.get('aspects', {})),
             "strength": soul_data.get('strength', 0.0),
             "stability": soul_data.get('stability', 0.0),
             "divine_qualities": list(soul_data.get('divine_qualities', {}).keys()),
             # Add other key metrics as needed
         }

    def _save_journey_log(self, soul_id: str) -> bool:
        """Save the journey log for a specific soul."""
        if soul_id not in self.soul_journeys:
             logger.error(f"Cannot save journey log: No data found for soul {soul_id}")
             return False

        filename = f"journey_log_{soul_id}.json"
        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving journey log for soul {soul_id} to {save_path}...")

        try:
            with open(save_path, 'w') as f:
                 json.dump(self.soul_journeys[soul_id], f, indent=2, default=str)
            logger.info("Journey log saved successfully.")
            return True
        except (IOError, TypeError) as e:
            logger.error(f"IOError saving journey log to {save_path}: {e}", exc_info=True)
            # Don't fail hard for log saving? Or should we? Let's not fail hard.
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving journey log: {e}", exc_info=True)
            return False

    def prepare_field_for_gateway(self, sephirah: str, gateway_key: str) -> Dict[str, Any]:
        """
        Prepare a specific Sephiroth field for gateway operations. Fails hard on error.

        Args:
            sephirah (str): The name of the Sephirah field to prepare.
            gateway_key (str): The key identifying the gateway (e.g., "tetrahedron").

        Returns:
            Dict[str, Any]: Result dictionary from `SephirothField3D.prepare_for_gateway`.

        Raises:
            ValueError: If sephirah or gateway_key are invalid.
            RuntimeError: If the field cannot be retrieved or preparation fails.
        """
        sephirah_lower = sephirah.lower()
        gateway_key_lower = gateway_key.lower()
        logger.info(f"Controller preparing {sephirah_lower} field for gateway '{gateway_key_lower}'...")

        try:
            # Get the field instance (raises error if fails)
            field = self.get_field(sephirah_lower)

            # Delegate preparation to the field
            preparation_result = field.prepare_for_gateway(gateway_key_lower)
            if not isinstance(preparation_result, dict) or "success" not in preparation_result:
                 raise RuntimeError(f"SephirothField3D.prepare_for_gateway did not return valid result for {sephirah_lower} / {gateway_key_lower}.")

            # Record metrics
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'prepare_gateway',
                'sephirah': sephirah_lower,
                'gateway_key': gateway_key_lower,
                'success': preparation_result.get('success', False),
                'resonance': preparation_result.get('resonance'),
                'stability': preparation_result.get('stability'),
                'timestamp': datetime.now().isoformat()
            })

            if preparation_result.get('success', False):
                 logger.info(f"Successfully prepared {sephirah_lower} field for gateway '{gateway_key_lower}'.")
            else:
                 logger.warning(f"Failed to prepare {sephirah_lower} field for gateway '{gateway_key_lower}'. Reason: {preparation_result.get('reason')}")

            return preparation_result # Return the result from the field

        except (ValueError, RuntimeError, AttributeError, TypeError, IOError) as e:
            logger.error(f"Failed to prepare {sephirah_lower} for gateway '{gateway_key_lower}': {e}", exc_info=True)
            raise RuntimeError(f"Gateway preparation failed: {e}") from e

    def visualize_sephiroth_field(self, sephirah: str, use_3d: bool = False, save: bool = False, show: bool = False, filename: Optional[str] = None) -> bool:
        """
        Visualize the state of a specific Sephiroth field (2D slice or 3D).

        Args:
            sephirah (str): The name of the Sephirah field to visualize.
            use_3d (bool): If True, attempt 3D visualization, otherwise 2D slice.
            save (bool): Whether to save the visualization.
            show (bool): Whether to display the visualization (ignored if matplotlib unavailable).
            filename (Optional[str]): Custom filename for saving (relative to controller output dir).

        Returns:
            bool: True if visualization was attempted successfully, False otherwise.
        """
        if not VISUALIZATION_ENABLED:
            logger.warning("Visualization requested but Matplotlib is not available.")
            return False

        sephirah_lower = sephirah.lower()
        save_path = os.path.join(self.output_dir, filename) if save and filename else None
        if save and not filename: # Auto-generate filename
             viz_type = "3d" if use_3d else "2d_slice"
             filename = f"sephiroth_{sephirah_lower}_{viz_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
             save_path = os.path.join(self.output_dir, filename)

        logger.debug(f"Attempting to visualize {sephirah_lower} field (3D={use_3d}, Show={show}, Save Path={save_path})")
        fig = None
        try:
            # Get the field instance
            field = self.get_field(sephirah_lower) # Can raise error

            # Choose visualization method
            if use_3d and hasattr(field, 'visualize_field_3d'):
                 viz_method = field.visualize_field_3d
                 viz_args = {'save_path': save_path} # Pass path directly
            elif hasattr(field, 'visualize_field_2d'):
                 viz_method = field.visualize_field_2d
                 viz_args = {'save_path': save_path} # Pass path directly
            else:
                 logger.error(f"No suitable visualization method found for {sephirah_lower} field (3D={use_3d}).")
                 return False

            # Call the visualization method
            fig = viz_method(**viz_args)
            if fig is None and save_path and os.path.exists(save_path):
                logger.info(f"Sephiroth field visualization saved by field method to {save_path}")
                return True
            elif fig is None:
                logger.warning("Field visualization method returned None.")
                return False

            # If field method returns figure and doesn't save directly:
            # if save_path and fig: # Saved by field method
            #     fig.savefig(save_path, dpi=300, bbox_inches='tight')
            #     logger.info(f"Sephiroth field visualization saved to {save_path}")

            if show and fig:
                plt.show()
            elif fig:
                plt.close(fig)

            return True

        except (ValueError, RuntimeError, AttributeError, TypeError, IOError) as e:
            logger.error(f"Error during {sephirah_lower} field visualization: {e}", exc_info=True)
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
            filename = f"sephiroth_controller_state_{self.controller_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        elif not filename.lower().endswith('.json'):
             filename += '.json'

        save_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving Sephiroth controller state to {save_path}...")

        try:
            # Get current metrics relevant to the controller
            controller_metrics = metrics.get_category_metrics(CONTROLLER_METRIC_CATEGORY)

            state_data = {
                'controller_id': self.controller_id,
                'creation_time': self.creation_time,
                'field_dimensions': list(self.dimensions),
                'output_dir': self.output_dir,
                'initialized_field_names': list(self.fields.keys()),
                'initialized_field_uuids': {name: field.uuid for name, field in self.fields.items()},
                'soul_journey_ids_in_progress': [sid for sid, data in self.soul_journeys.items() if data.get('status') == 'started'],
                'soul_journey_ids_completed': [sid for sid, data in self.soul_journeys.items() if data.get('status') == 'completed'],
                'soul_journey_ids_failed': [sid for sid, data in self.soul_journeys.items() if data.get('status') == 'failed'],
                'last_metrics_snapshot': controller_metrics,
                'save_timestamp': datetime.now().isoformat()
            }
            # Note: We are NOT saving journey details or field instances here.

            with open(save_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.info("Sephiroth controller state saved successfully.")
            return True
        except IOError as e:
            logger.error(f"IOError saving Sephiroth controller state to {save_path}: {e}", exc_info=True)
            raise IOError(f"Failed to write Sephiroth controller state file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error saving Sephiroth controller state: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save Sephiroth controller state: {e}") from e

    def get_soul_journey_log(self, soul_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the journey log data for a specific soul.

        Args:
            soul_id (str): The ID of the soul whose journey log is requested.

        Returns:
            Optional[Dict[str, Any]]: The journey log dictionary, or None if not found.
        """
        return self.soul_journeys.get(soul_id) # Returns None if key doesn't exist


    def __str__(self) -> str:
        """String representation of the Sephiroth Controller."""
        num_fields = len(self.fields)
        num_journeys = len(self.soul_journeys)
        num_complete = sum(1 for data in self.soul_journeys.values() if data.get('status') == 'completed')
        return (f"SephirothController(ID: {self.controller_id}, Dims: {self.dimensions}, "
                f"Initialized Fields: {num_fields}, Total Journeys Tracked: {num_journeys}, "
                f"Completed Journeys: {num_complete})")

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"<SephirothController id='{self.controller_id}' dims={self.dimensions} "
                f"fields={list(self.fields.keys())}>")


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Sephiroth Controller Module Example...")
    controller = None
    try:
        # Initialize Controller (optionally initialize all fields)
        controller = SephirothController(
            dimensions=(32, 32, 32), # Smaller for example
            data_dir="output/sephiroth_controller_example",
            initialize_all=True # Create all fields at start for this example
        )
        print(controller)

        # --- Example: Simulate guiding a soul ---
        # Assume we have 'finalized_soul_data' from Guff stage
        dummy_finalized_soul_data = {
            "spark_id": "soul_abc123",
            "formation_quality": 0.85,
            "final_strength": 0.9,
            "final_resonance": 0.88,
            "creator_resonance": {"overall_resonance": 0.8},
            "sephiroth_connections": {"kether": 0.8, "yesod": 0.6, "hod": 0.5, "netzach": 0.5}, # Example connections
            "ready_for_sephiroth": True,
            "aspects": {"core_identity": {"strength": 1.0}, "stability_matrix": {"strength": 0.9}}, # Example aspects
            # ... other keys from Guff finalization ...
        }

        print("\n--- Guiding Dummy Soul Journey (Yesod -> Kether) ---")
        try:
            # Use default journey order (Yesod to Kether)
            final_soul_state = controller.guide_soul_journey(
                soul_transfer_data=dummy_finalized_soul_data,
                duration_per_sephirah=2.0 # Short duration for example
            )
            print("\nJourney Completed. Final Soul State Summary:")
            print(f"  Soul ID: {final_soul_state.get('spark_id')}")
            print(f"  Aspect Count: {len(final_soul_state.get('aspects', {}))}")
            print(f"  Strength: {final_soul_state.get('strength', 'N/A'):.4f}")
            print(f"  Stability: {final_soul_state.get('stability', 'N/A'):.4f}")

            # Retrieve and print the journey log
            journey_log = controller.get_soul_journey_log("soul_abc123")
            if journey_log:
                 print("\nJourney Log:")
                 # print(json.dumps(journey_log, indent=2, default=str)) # Can be long
                 print(f"  Status: {journey_log.get('status')}")
                 print(f"  Path Taken ({len(journey_log.get('path_taken',[]))} steps): {[step.get('sephirah') for step in journey_log.get('path_taken',[])]}")
            else:
                 print("\nCould not retrieve journey log.")

        except (ValueError, RuntimeError) as journey_err:
            print(f"\nSoul journey failed: {journey_err}")


        # --- Example: Prepare a field for gateway ---
        print("\n--- Preparing Kether for Dodecahedron Gateway ---")
        try:
            prep_result = controller.prepare_field_for_gateway("kether", "dodecahedron")
            print("Gateway Preparation Result:", prep_result)
            if VISUALIZATION_ENABLED:
                 controller.visualize_sephiroth_field("kether", use_3d=True, save=True, show=False, filename="kether_gateway_prep.png")

        except (ValueError, RuntimeError) as prep_err:
             print(f"\nGateway preparation failed: {prep_err}")


        # Save controller state
        controller.save_controller_state()

        print("\nSephiroth Controller Example Finished Successfully.")

    except (ValueError, TypeError, IOError, RuntimeError, ImportError) as e:
        print(f"\n--- ERROR during Sephiroth Controller Example ---")
        print(f"An error occurred: {type(e).__name__}: {e}")
        if controller: print(f"Controller state at error: {controller}")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------------------")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during Sephiroth Controller Example ---")
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        if controller: print(f"Controller state at error: {controller}")
        import traceback
        traceback.print_exc()
        print("--------------------------------------------------------------")

# --- END OF FILE sephiroth_controller.py ---
