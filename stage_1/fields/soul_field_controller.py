# --- START OF FILE soul_field_controller.py ---

"""
Soul Field Controller Module

Provides a high-level controller for managing the entire soul development framework,
including field initialization, soul creation and management, orchestrating key
soul formation processes like Entanglement and the Sephiroth Journey, and
running development cycles.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
import numpy as np 
import json
import uuid
import random # For default positioning calculation
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime



# --- Field System Imports ---
# Assuming controller is in src/stage_1/fields/
from stage_1.fields.field_system import FieldSystem
from stage_1.fields.field_registry import FieldRegistry
from stage_1.fields.base_field import BaseField
from stage_1.fields.void_field import VoidField
from stage_1.fields.sephiroth_field import SephirothField
from stage_1.fields.guff_field import GuffField
from stage_1.fields.kether_field import KetherField
# Import all specific Sephiroth Fields explicitly
from stage_1.fields.chokmah_field import ChokmahField
from stage_1.fields.binah_field import BinahField
from stage_1.fields.chesed_field import ChesedField
from stage_1.fields.geburah_field import GeburahField
from stage_1.fields.tiphareth_field import TipharethField
from stage_1.fields.netzach_field import NetzachField
from stage_1.fields.hod_field import HodField
from stage_1.fields.yesod_field import YesodField
from stage_1.fields.malkuth_field import MalkuthField
from stage_1.fields.daath_field import DaathField
# --- Soul Formation Process Imports ---
# Assuming these are in src/stage_1/soul_formation/
# Use try-except for robustness
try:
    from stage_1.soul_formation.soul_spark import SoulSpark
    from stage_1.soul_formation.sephiroth_journey_processing import process_sephirah_interaction
    from stage_1.fields.creator_entanglement import run_full_entanglement_process

    SOUL_FORMATION_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).critical(f"CRITICAL ERROR: Failed to import soul formation modules: {e}. Controller orchestration will be limited.")
    SOUL_FORMATION_AVAILABLE = False
    # Define dummy classes/functions if absolutely needed for type hints or structure
    class SoulSpark: pass
    def process_sephirah_interaction(*args, **kwargs): raise NotImplementedError("Sephiroth Journey Processing module not found.")
    def run_full_entanglement_process(*args, **kwargs): raise NotImplementedError("Creator Entanglement module not found.")


# --- Visualization Import ---
try:
    from .field_visualization import visualize_field_layout_3d # Relative import
    # Import matplotlib just for type hinting the return value
    import matplotlib.pyplot as plt
    Figure = plt.Figure
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("field_visualization module or matplotlib not found. Layout visualization will be unavailable.")
    VISUALIZATION_AVAILABLE = False
    visualize_field_layout_3d = None
    Figure = Any # Placeholder type hint

# Default constants in case import fails
DEFAULT_EDGE_OF_CHAOS_RATIO = 0.618  # Fallback value

# --- Constants Import ---
try:
    from constants.constants import *
except ImportError as e:
    logging.getLogger(__name__).critical(f"CRITICAL ERROR: Failed to import constants: {e}. Using fallback values.")

# Configure logging
logger = logging.getLogger(__name__)


class SoulFieldController:
    """
    High-level controller for the Soul Development Framework field system.
    Orchestrates field creation, soul management, and key developmental processes.
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 auto_initialize: bool = True,
                 create_tree: bool = True):
        """
        Initialize the Soul Field Controller.

        Args:
            config_path: Optional path to a configuration file.
            auto_initialize: Automatically initialize the FieldSystem (Void, Kether, Guff).
            create_tree: Automatically create the full Tree of Life structure if auto_initialize is True.

        Raises:
            RuntimeError: If core initialization fails critically.
        """
        self.controller_id: str = str(uuid.uuid4())
        logger.info(f"Initializing Soul Field Controller (ID: {self.controller_id})...")
        self.creation_time = datetime.now().isoformat()
        self.initialized = False
        self.field_system: Optional[FieldSystem] = None
        self.souls: Dict[str, SoulSpark] = {} # Store SoulSpark objects directly
        self.sephiroth_field_ids: Dict[str, str] = {} # Map lowercase name to field ID
        self.config: Dict[str, Any] = {}

        try:
            if config_path:
                self.config = self._load_config(config_path)
                # Apply config settings (e.g., log level, defaults) - implementation needed

            # Initialize the underlying Field System
            logger.info(f"Initializing Field System (Auto-Init: {auto_initialize})...")
            self.field_system = FieldSystem(auto_initialize=auto_initialize)
            if not self.field_system.initialized and auto_initialize:
                raise RuntimeError("FieldSystem failed to auto-initialize.")

            # Create the Tree of Life structure if requested
            if create_tree and self.field_system.initialized:
                logger.info("Controller triggering Tree of Life creation...")
                self.sephiroth_field_ids = self.field_system.create_tree_of_life()
                if not self.sephiroth_field_ids:
                     logger.warning("Tree of Life creation returned empty map.")
                else:
                     logger.info(f"Tree of Life fields mapped: {list(self.sephiroth_field_ids.keys())}")

            self.initialized = True
            logger.info(f"Soul Field Controller '{self.controller_id}' initialized successfully.")

        except Exception as e:
            error_msg = f"CRITICAL FAILURE during Soul Field Controller initialization: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            self.initialized = False
            self.field_system = None
            raise RuntimeError(error_msg) from e

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load config file {config_path}: {e}") from e

    # --- Soul Management Methods ---

    def create_soul(self,
                   name: Optional[str] = None,
                   initial_field_key: str = "guff", # Use key like 'guff', 'kether', 'tiphareth'
                   initial_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Creates a new SoulSpark object and places it in the specified initial field.

        Args:
            name: Optional name for the soul.
            initial_field_key: Key indicating the field ('guff', or lowercase sephirah name).
            initial_data: Optional dictionary to initialize SoulSpark attributes.

        Returns:
            The unique ID of the created soul.

        Raises:
            RuntimeError: If controller or field system not initialized, or creation fails.
            ValueError: If initial_field_key is invalid or placement fails.
            TypeError: If arguments have incorrect types.
        """
        if not self.initialized or not self.field_system:
            raise RuntimeError("Controller or FieldSystem not initialized.")
        if not SOUL_FORMATION_AVAILABLE:
             raise RuntimeError("SoulSpark class is not available.")
        if not isinstance(initial_field_key, str):
             raise TypeError("initial_field_key must be a string.")

        soul_id = str(uuid.uuid4())
        soul_name = name or f"Soul-{soul_id[:8]}"
        logger.info(f"Creating soul '{soul_name}' (ID: {soul_id}) in field '{initial_field_key}'...")

        try:
            # Create the SoulSpark instance
            soul_spark = SoulSpark(initial_data=initial_data, spark_id=soul_id)
            soul_spark.name = soul_name # Ensure name is set

            # Determine target field ID
            target_field_id = None
            field_key_lower = initial_field_key.lower()

            if field_key_lower == "guff":
                guff_field = self.field_system.get_guff_field()
                if guff_field is None:
                    raise ValueError("Guff field not found, cannot place soul.")
                target_field_id = guff_field.field_id
                # Use the specific Guff placement method
                self.field_system.move_soul_to_guff(soul_id, soul_spark.__dict__) # Pass soul data
            elif field_key_lower in self.sephiroth_field_ids:
                target_field_id = self.sephiroth_field_ids[field_key_lower]
                # Use general placement method for Sephiroth
                self.field_system.place_soul_in_field(soul_id, target_field_id)
            elif field_key_lower == "kether": # Handle Kether explicitly if Tree wasn't created
                 kether_field = self.field_system.get_kether_field()
                 if kether_field is None: raise ValueError("Kether field not found.")
                 target_field_id = kether_field.field_id
                 self.field_system.place_soul_in_field(soul_id, target_field_id)
            else:
                raise ValueError(f"Invalid initial field key: '{initial_field_key}'. Must be 'guff' or a valid Sephirah name.")

            # Store the SoulSpark object itself
            self.souls[soul_id] = soul_spark

            # Update soul's history (assuming SoulSpark object doesn't do this automatically)
            if hasattr(soul_spark, 'field_history') and isinstance(soul_spark.field_history, list):
                 field_name = self.field_system.get_field(target_field_id).name
                 soul_spark.field_history.append({
                     'field_id': target_field_id,
                     'field_name': field_name,
                     'entry_time': soul_spark.creation_time,
                     'exit_time': None
                 })

            logger.info(f"Successfully created and placed soul {soul_id} in field {target_field_id}.")
            return soul_id

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            error_msg = f"Failed to create soul '{soul_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Clean up if soul object was created but placement failed
            if soul_id in self.souls: del self.souls[soul_id]
            if soul_id in self.field_system.soul_field_locations: del self.field_system.soul_field_locations[soul_id]
            raise RuntimeError(error_msg) from e
        except Exception as e:
             error_msg = f"Unexpected error creating soul '{soul_name}': {str(e)}"
             logger.error(error_msg, exc_info=True)
             if soul_id in self.souls: del self.souls[soul_id]
             if soul_id in self.field_system.soul_field_locations: del self.field_system.soul_field_locations[soul_id]
             raise RuntimeError(error_msg) from e

    def get_soul(self, soul_id: str) -> SoulSpark:
        """Retrieves the SoulSpark object by its ID."""
        if not isinstance(soul_id, str) or not soul_id:
            raise TypeError("soul_id must be a non-empty string.")
        soul = self.souls.get(soul_id)
        if soul is None:
            raise ValueError(f"Soul with ID '{soul_id}' not found in controller.")
        return soul

    def move_soul(self, soul_id: str, target_field_key: str) -> bool:
        """Moves a soul to a different field using its key ('guff', 'kether', etc.)."""
        if not self.initialized or not self.field_system:
             raise RuntimeError("Controller or FieldSystem not initialized.")
        soul_spark = self.get_soul(soul_id) # Raises ValueError if not found
        field_key_lower = target_field_key.lower()
        logger.info(f"Moving soul {soul_id} to field '{field_key_lower}'...")

        try:
            current_field_id = self.field_system.soul_field_locations.get(soul_id)

            # Update history in soul object if moving *from* a field
            if current_field_id and hasattr(soul_spark, 'field_history') and isinstance(soul_spark.field_history, list):
                 now = datetime.now().isoformat()
                 for entry in reversed(soul_spark.field_history): # Find last entry without exit time
                      if entry.get('field_id') == current_field_id and entry.get('exit_time') is None:
                           entry['exit_time'] = now
                           break

            # Perform move using FieldSystem methods
            target_field_id = None
            if field_key_lower == "guff":
                # Need soul_data dictionary for move_soul_to_guff
                soul_data_dict = soul_spark.__dict__ # Or use get_spark_metrics() if available
                target_field_id = self.field_system.move_soul_to_guff(soul_id, soul_data_dict)
            elif field_key_lower in self.sephiroth_field_ids:
                target_field_id = self.sephiroth_field_ids[field_key_lower]
                self.field_system.place_soul_in_field(soul_id, target_field_id)
            elif field_key_lower == "kether":
                 kether_field = self.field_system.get_kether_field()
                 if kether_field is None: raise ValueError("Kether field not found.")
                 target_field_id = kether_field.field_id
                 self.field_system.place_soul_in_field(soul_id, target_field_id)
            else:
                 raise ValueError(f"Invalid target field key: '{target_field_key}'.")

            # Update history in soul object after move *to* a field
            if target_field_id and hasattr(soul_spark, 'field_history') and isinstance(soul_spark.field_history, list):
                 field_name = self.field_system.get_field(target_field_id).name
                 soul_spark.field_history.append({
                     'field_id': target_field_id,
                     'field_name': field_name,
                     'entry_time': datetime.now().isoformat(),
                     'exit_time': None
                 })

            logger.info(f"Successfully moved soul {soul_id} to field {target_field_id}")
            return True

        except (ValueError, TypeError, RuntimeError) as e:
             error_msg = f"Failed to move soul {soul_id} to '{target_field_key}': {str(e)}"
             logger.error(error_msg)
             raise RuntimeError(error_msg) from e # Re-raise as runtime error
        except Exception as e:
             error_msg = f"Unexpected error moving soul {soul_id} to '{target_field_key}': {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    # --- Orchestration Methods ---

    def initiate_soul_journey(self, soul_id: str,
                             creator_resonance: float = 0.8,
                             edge_of_chaos_ratio: float = DEFAULT_EDGE_OF_CHAOS_RATIO,
                             stabilization_iterations: int = 5) -> bool:
        """
        Performs the Creator Entanglement process for a soul, preparing it for the journey.

        Args:
            soul_id: The ID of the SoulSpark object to entangle.
            creator_resonance: Base resonance strength for connection (0-1).
            edge_of_chaos_ratio: Ideal chaos ratio for connection (0-1).
            stabilization_iterations: Number of stabilization iterations.

        Returns:
            True if entanglement was successful.

        Raises:
            RuntimeError: If entanglement module unavailable or process fails.
            ValueError: If soul not found or prerequisites not met.
            TypeError: If arguments invalid type.
        """
        if not SOUL_FORMATION_AVAILABLE:
            raise RuntimeError("Creator Entanglement module is not available.")

        logger.info(f"Initiating Creator Entanglement for soul {soul_id}...")
        soul_spark = self.get_soul(soul_id) # Raises ValueError if not found

        # --- Check Prerequisites ---
        # Example: Check if soul formation is considered complete enough
        if not getattr(soul_spark, 'formation_complete', False):
            raise ValueError(f"Soul {soul_id} formation is not complete. Cannot initiate entanglement.")
        if getattr(soul_spark, 'creator_channel_id', None):
             logger.warning(f"Soul {soul_id} already has a creator channel ID. Re-running entanglement.")
             # Decide whether to raise error or allow re-entanglement

        try:
            # Call the entanglement orchestration function
            modified_soul, summary_metrics = run_full_entanglement_process(
                soul_spark=soul_spark, # Pass the actual object
                creator_resonance=creator_resonance,
                edge_of_chaos_ratio=edge_of_chaos_ratio,
                stabilization_iterations=stabilization_iterations
            )

            # Update the controller's reference if the object identity changed (unlikely if modified in-place)
            self.souls[soul_id] = modified_soul

            # Set a flag on the soul indicating entanglement complete
            setattr(soul_spark, 'creator_entangled', True)
            setattr(soul_spark, 'last_modified', summary_metrics.get('end_time', datetime.now().isoformat()))

            logger.info(f"Creator Entanglement successful for soul {soul_id}.")
            return True

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            error_msg = f"Creator Entanglement failed for soul {soul_id}: {str(e)}"
            logger.error(error_msg)
            setattr(soul_spark, 'creator_entangled', False) # Ensure flag is false on failure
            raise RuntimeError(error_msg) from e
        except Exception as e:
             error_msg = f"Unexpected error during Creator Entanglement for soul {soul_id}: {str(e)}"
             logger.error(error_msg, exc_info=True)
             setattr(soul_spark, 'creator_entangled', False)
             raise RuntimeError(error_msg) from e

    def run_sephiroth_journey(self, soul_id: str,
                             journey_path: Optional[List[str]] = None,
                             duration_per_sephirah: float = 5.0) -> bool:
        """
        Orchestrates the soul's journey through the specified Sephiroth path.

        Args:
            soul_id: The ID of the SoulSpark object undertaking the journey.
            journey_path: List of lowercase Sephiroth names defining the path.
                          If None, uses a default descending path.
            duration_per_sephirah: Time (simulation units) spent in each Sephirah.

        Returns:
            True if the journey completes successfully.

        Raises:
            RuntimeError: If processing module unavailable or journey fails critically.
            ValueError: If soul not found, path invalid, duration invalid, or prerequisites not met.
            TypeError: If arguments invalid type.
        """
        if not SOUL_FORMATION_AVAILABLE:
            raise RuntimeError("Sephiroth Journey Processing module is not available.")
        if not isinstance(duration_per_sephirah, (int, float)) or duration_per_sephirah <= 0:
             raise ValueError("duration_per_sephirah must be positive.")

        logger.info(f"Starting Sephiroth Journey for soul {soul_id}...")
        soul_spark = self.get_soul(soul_id) # Raises ValueError if not found

        # --- Check Prerequisites ---
        if not getattr(soul_spark, 'creator_entangled', False):
            raise ValueError(f"Soul {soul_id} has not completed Creator Entanglement. Cannot start journey.")
        if getattr(soul_spark, 'sephiroth_journey_complete', False):
             logger.warning(f"Soul {soul_id} has already completed a Sephiroth journey. Running again.")
             # Reset relevant journey state? Or allow multiple journeys?

        # Define default journey path if none provided
        if journey_path is None:
             journey_path = [ # Default descending path (excluding Daath unless specified)
                "kether", "chokmah", "binah", "chesed", "geburah",
                "tiphareth", "netzach", "hod", "yesod", "malkuth"
             ]
        elif not isinstance(journey_path, list) or not all(isinstance(s, str) for s in journey_path):
             raise TypeError("journey_path must be a list of strings.")

        logger.info(f"Journey Path: {' -> '.join(p.capitalize() for p in journey_path)}")

        journey_metrics = {'steps': [], 'success': False}
        try:
            # Iterate through the path
            for i, sephirah_name_lower in enumerate(journey_path):
                sephirah_name_lower = sephirah_name_lower.lower() # Ensure lowercase
                logger.info(f"Journey Step {i+1}/{len(journey_path)}: Soul {soul_id} entering {sephirah_name_lower.capitalize()}...")

                # 1. Move Soul to the Field
                # Need the field ID from the name stored during tree creation
                if sephirah_name_lower not in self.sephiroth_field_ids:
                     # Attempt to find/create field if tree wasn't fully generated
                     logger.warning(f"Field ID for '{sephirah_name_lower}' not in map. Attempting to find/create...")
                     # Try finding by name first
                     found_field = self.field_system.get_field_by_name(f"{sephirah_name_lower.capitalize()} - {self._get_sephirah_title(sephirah_name_lower)}")
                     if found_field:
                          self.sephiroth_field_ids[sephirah_name_lower] = found_field.field_id
                          logger.info(f"Found existing field for {sephirah_name_lower}.")
                     else:
                          # If not found, attempt to create it (needs position logic improvement)
                          # This part is complex - requires knowing where to place it if not pre-created.
                          # For now, we raise an error if it wasn't pre-mapped.
                          raise ValueError(f"Field ID for {sephirah_name_lower} not found and auto-creation during journey is not fully supported yet.")

                target_field_id = self.sephiroth_field_ids[sephirah_name_lower]
                self.move_soul(soul_id, sephirah_name_lower) # Use the name key

                # 2. Process Interaction within the field
                # process_sephirah_interaction modifies soul_spark in place
                _, step_metrics = process_sephirah_interaction(
                    soul_spark=soul_spark,
                    sephirah_name=sephirah_name_lower,
                    duration=duration_per_sephirah
                )
                journey_metrics['steps'].append(step_metrics)
                logger.info(f"Interaction in {sephirah_name_lower.capitalize()} complete.")

            # --- Journey Completion ---
            setattr(soul_spark, 'sephiroth_journey_complete', True)
            setattr(soul_spark, 'last_modified', datetime.now().isoformat())
            journey_metrics['success'] = True
            # Optional: Consolidate journey effects, e.g., average stability change
            # journey_metrics['overall_stability_change'] = sum(s['state_changes'].get('stability_change',0) for s in journey_metrics['steps'])

            logger.info(f"Sephiroth Journey successfully completed for soul {soul_id}.")
            # Optionally save journey metrics
            # self._save_journey_report(soul_id, journey_metrics)
            return True

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            failed_step = sephirah_name_lower if 'sephirah_name_lower' in locals() else 'unknown'
            error_msg = f"Sephiroth Journey failed for soul {soul_id} at step '{failed_step}': {str(e)}"
            logger.error(error_msg, exc_info=True) # Log with traceback
            setattr(soul_spark, 'sephiroth_journey_complete', False) # Mark as failed
            journey_metrics['success'] = False
            journey_metrics['error'] = error_msg
            journey_metrics['failed_step'] = failed_step
            # self._save_journey_report(soul_id, journey_metrics) # Save partial/failed report
            raise RuntimeError(error_msg) from e
        except Exception as e:
             failed_step = sephirah_name_lower if 'sephirah_name_lower' in locals() else 'unknown'
             error_msg = f"Unexpected error during Sephiroth Journey for soul {soul_id} at step '{failed_step}': {str(e)}"
             logger.critical(error_msg, exc_info=True)
             setattr(soul_spark, 'sephiroth_journey_complete', False)
             journey_metrics['success'] = False
             journey_metrics['error'] = error_msg
             journey_metrics['failed_step'] = failed_step
             # self._save_journey_report(soul_id, journey_metrics)
             raise RuntimeError(error_msg) from e

    # --- Visualization Method ---
    def visualize_current_layout(self, **kwargs) -> Optional[Figure]:
        """
        Generates a 3D visualization of the current field layout using the external function.

        Args:
            **kwargs: Arguments passed directly to visualize_field_layout_3d
                      (e.g., show_labels=True, save_path="output/layout.png", show_plot=True).

        Returns:
            Optional[plt.Figure]: Matplotlib Figure object if successful, None otherwise.

        Raises:
            RuntimeError: If visualization module or the field system/registry is unavailable,
                          or if the visualization generation fails.
        """
        if not VISUALIZATION_AVAILABLE or visualize_field_layout_3d is None:
            raise RuntimeError("Field visualization capability is not available (matplotlib or field_visualization module missing).")
        if not self.field_system or not self.field_system.registry:
            raise RuntimeError("Field System or Registry not available for visualization.")
        if not self.field_system.initialized:
             raise RuntimeError("Field System must be initialized to visualize layout.")

        logger.info("Calling field layout visualization...")
        try:
            # Pass the registry instance from the field system
            fig = visualize_field_layout_3d(registry=self.field_system.registry, **kwargs)
            return fig
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to generate field layout visualization via controller: {e}")
            raise RuntimeError(f"Layout visualization failed: {e}") from e # Re-raise as RuntimeError
        except Exception as e:
             logger.error(f"Unexpected error during controller's layout visualization call: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error during layout visualization: {e}") from e


    # --- Status and Reporting ---
    def get_system_status(self) -> Dict[str, Any]:
        """Gets comprehensive system status."""
        if not self.initialized or not self.field_system:
             raise RuntimeError("Controller or FieldSystem not initialized.")
        return self.field_system.get_system_status() # Delegate to FieldSystem

    # (Keep __str__ and __repr__ as previously defined)
    def __str__(self) -> str:
        """String representation of the controller."""
        soul_count = len(self.souls)
        field_count = len(self.field_system.registry.fields) if self.field_system and self.field_system.registry else 0
        return f"SoulFieldController(ID: {self.controller_id[:8]}, Souls: {soul_count}, Fields: {field_count}, Initialized: {self.initialized})"

    def __repr__(self) -> str:
        """Detailed representation."""
        soul_count = len(self.souls)
        field_count = len(self.field_system.registry.fields) if self.field_system and self.field_system.registry else 0
        fs_ok = self.field_system is not None and self.field_system.initialized
        return f"<SoulFieldController id='{self.controller_id}' souls={soul_count} fields={field_count} system_ok={fs_ok}>"


# --- END OF (MODIFIED) soul_field_controller.py ---