# --- START OF (MODIFIED) field_system.py ---

"""
Field System Module

The main interface to the field system architecture. Provides high-level APIs for
working with fields, managing entities across fields, and running field operations.
Interacts with the FieldRegistry and orchestrates operations across field types.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
from typing import Dict, List, Any, Tuple, Optional, Type, Union
from datetime import datetime
import random # Import random for default positioning
import numpy as np # Import numpy for calculations

# Import field registry and ALL field types
from src.stage_1.fields.field_registry import FieldRegistry
from src.stage_1.fields.base_field import BaseField
from src.stage_1.fields.void_field import VoidField
from src.stage_1.fields.sephiroth_field import SephirothField
from src.stage_1.fields.guff_field import GuffField
# Import all specific Sephiroth Fields explicitly for type checking and clarity
from src.stage_1.fields.kether_field import KetherField
from src.stage_1.fields.chokmah_field import ChokmahField
from src.stage_1.fields.binah_field import BinahField
from src.stage_1.fields.chesed_field import ChesedField
from src.stage_1.fields.geburah_field import GeburahField
from src.stage_1.fields.tiphareth_field import TipharethField
from src.stage_1.fields.netzach_field import NetzachField
from src.stage_1.fields.hod_field import HodField
from src.stage_1.fields.yesod_field import YesodField
from src.stage_1.fields.malkuth_field import MalkuthField
from src.stage_1.fields.daath_field import DaathField

# Configure logging
logger = logging.getLogger(__name__)


class FieldSystem:
    """
    Main interface to the field system architecture.
    Provides high-level APIs for working with fields and entities within them.
    """

    def __init__(self, auto_initialize: bool = True):
        """
        Initialize the field system.

        Args:
            auto_initialize: Whether to automatically initialize the basic field structure (Void, Kether, Guff).

        Raises:
            RuntimeError: If field system initialization fails critically.
        """
        logger.info(f"Initializing Field System (Auto-Initialize: {auto_initialize})...")
        try:
            # Get registry instance
            self.registry = FieldRegistry.get_instance()

            # State variables
            self.initialized = False
            self.creation_time = datetime.now().isoformat()

            # Track current active fields for souls {soul_id: field_id}
            self.soul_field_locations: Dict[str, str] = {}

            # Auto-initialize if requested
            if auto_initialize:
                self.initialize_system() # This calls registry.initialize_system()

            logger.info("Field System initialized successfully.")

        except Exception as e:
            # Catch errors from registry init or self.initialize_system
            error_msg = f"CRITICAL FAILURE during Field System initialization: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            # Ensure state reflects failure
            self.initialized = False
            self.registry = None # Indicate registry failed or wasn't fully set up
            raise RuntimeError(error_msg) from e

    def initialize_system(self) -> bool:
        """
        Initialize the basic field structure via the registry.
        Ensures Kether is the correct type if registry created a base version initially.

        Returns:
            True if initialization was successful.

        Raises:
            RuntimeError: If system is already initialized or core initialization fails.
        """
        if self.initialized:
            raise RuntimeError("Field system is already initialized.")
        if not self.registry: # Safety check
             raise RuntimeError("Field Registry not available for initialization.")

        logger.info("Triggering core system structure initialization (Void, Kether, Guff)...")
        try:
            # Let registry handle the core creation
            self.registry.initialize_system()
            # Registry's initialize_system now directly creates KetherField

            # Verify core components exist after registry init
            if self.registry.void_field_id is None:
                raise RuntimeError("Void field ID not set after registry initialization.")
            kether_field = self.get_kether_field()
            if kether_field is None:
                 raise RuntimeError("Kether field not found after registry initialization.")
            guff_field = self.get_guff_field()
            if guff_field is None:
                 raise RuntimeError("Guff field not found after registry initialization.")

            # Verify Kether is the correct type (should be handled by registry now)
            if not isinstance(kether_field, KetherField):
                 logger.warning(f"Kether field (ID: {kether_field.field_id}) is type {type(kether_field)}, expected KetherField. Registry init might need check.")
                 # Allow proceeding but log warning. Could raise RuntimeError for stricter control.

            self.initialized = True
            logger.info("Core system structure initialized successfully.")
            return True

        except RuntimeError as re:
            # Catch errors specifically from registry.initialize_system or self checks
            error_msg = f"Failed to initialize field system structure: {str(re)}"
            logger.critical(error_msg, exc_info=True)
            self.initialized = False # Ensure state reflects failure
            raise re # Re-raise the caught error
        except Exception as e:
            # Catch any other unexpected errors during init
            error_msg = f"Unexpected error during field system structure initialization: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            self.initialized = False
            raise RuntimeError(error_msg) from e

    def create_sephiroth_field(self, sephirah_name: str, position: Tuple[float, float, float],
                              dimensions: Optional[Tuple[float, float, float]] = None, **kwargs) -> str:
        """
        Create a Sephiroth field and place it in the Void. Uses specific field type if available.

        Args:
            sephirah_name: Name of the Sephirah (e.g., "chesed", "kether"). Case-insensitive.
            position: Position (x, y, z) of the field's center in the Void.
            dimensions: Dimensions (x, y, z) of the field (optional, defaults based on Sephirah).
            **kwargs: Additional keyword arguments passed to the field constructor.

        Returns:
            ID of the created field.

        Raises:
            RuntimeError: If system not initialized, Void field missing, or creation fails.
            ValueError: If parameters are invalid, Sephirah type unknown, or position/dimensions invalid.
            TypeError: If arguments have incorrect types.
            AttributeError: If VoidField lacks required methods.
        """
        if not self.initialized:
            raise RuntimeError("Field system must be initialized before creating Sephiroth fields.")
        if not self.registry: # Safety check
             raise RuntimeError("Field Registry not available.")
        if self.registry.void_field_id is None:
             raise RuntimeError("Void field ID not set. Cannot place Sephiroth field.")

        # --- Input Validation ---
        if not isinstance(sephirah_name, str) or not sephirah_name:
            raise ValueError("Sephirah name must be a non-empty string.")
        sephirah_lower = sephirah_name.lower()

        if not isinstance(position, tuple) or len(position) != 3 or not all(isinstance(c, (int, float)) for c in position):
            raise TypeError("Position must be a tuple of 3 numbers.")
        if dimensions is not None:
            if not isinstance(dimensions, tuple) or len(dimensions) != 3 or not all(isinstance(c, (int, float)) and c > 0 for c in dimensions):
                raise ValueError("Dimensions, if provided, must be a tuple of 3 positive numbers.")

        # Check if the specific Sephiroth type exists in the registry
        field_type_to_create = sephirah_lower
        if field_type_to_create not in self.registry.field_types:
            # If specific type (e.g., 'chesed') not found, try generic 'sephiroth' base type
            if "sephiroth" in self.registry.field_types:
                 logger.warning(f"Specific field type '{field_type_to_create}' not registered. Using generic 'sephiroth' type.")
                 field_type_to_create = "sephiroth"
                 # Ensure sephiroth_name is passed for generic type
                 if 'sephiroth_name' not in kwargs: kwargs['sephiroth_name'] = sephirah_lower
            else:
                 raise ValueError(f"Unsupported Sephiroth type: '{sephirah_name}' (Neither specific nor generic 'sephiroth' type registered).")

        logger.info(f"Attempting to create {sephirah_name.capitalize()} field (Type: {field_type_to_create})...")
        try:
            # Get Void field
            void_field = self.registry.get_field(self.registry.void_field_id)
            if not isinstance(void_field, VoidField):
                raise RuntimeError(f"Registry returned non-VoidField for void_field_id {self.registry.void_field_id}.")

            # Set default dimensions if not provided
            if dimensions is None:
                # Use a map for standard dimensions
                dims_map = {
                    "kether": (100.0, 100.0, 100.0), "chokmah": (90.0, 90.0, 90.0), "binah": (90.0, 90.0, 90.0),
                    "chesed": (80.0, 80.0, 80.0), "geburah": (80.0, 80.0, 80.0), "tiphareth": (85.0, 85.0, 85.0),
                    "netzach": (75.0, 75.0, 75.0), "hod": (75.0, 75.0, 75.0), "yesod": (70.0, 70.0, 70.0),
                    "malkuth": (65.0, 65.0, 65.0), "daath": (85.0, 85.0, 85.0)
                }
                default_dims = (70.0, 70.0, 70.0) # Fallback default
                dimensions = dims_map.get(sephirah_lower, default_dims)
                if not all(d > 0 for d in dimensions): # Final check on defaults
                     raise ValueError(f"Internal Error: Default dimensions for {sephirah_name} are not positive: {dimensions}")
                logger.debug(f"Using default dimensions for {sephirah_name}: {dimensions}")

            # Define field name
            field_name = f"{sephirah_name.capitalize()} - {self._get_sephirah_title(sephirah_lower)}"

            # Prepare arguments for field creation
            creation_args = {
                "name": field_name,
                "dimensions": dimensions,
                # Base frequency is usually set by the specific Sephiroth class constructor default
                # Pass kwargs from caller
                **kwargs
            }
            # If using generic 'sephiroth' type, ensure sephiroth_name is passed
            if field_type_to_create == "sephiroth" and 'sephiroth_name' not in creation_args:
                creation_args['sephiroth_name'] = sephirah_lower

            # Create the field using the registry
            field = self.registry.create_field(field_type_to_create, **creation_args)

            # Add the newly created field to the Void field's contained list
            if not hasattr(void_field, 'add_contained_field'):
                 # This check might be redundant if VoidField type check passed, but good practice
                 raise AttributeError("VoidField instance missing 'add_contained_field' method.")
            void_field.add_contained_field(
                field.field_id,
                "sephiroth", # Type within void context
                position,
                dimensions
            )

            # Connect the Void and the new field in the registry
            self.registry.connect_fields(self.registry.void_field_id, field.field_id, "containment", 0.9)

            logger.info(f"Successfully created {field_name} (Type: {field.field_type}, ID: {field.field_id}) at position {position}")
            return field.field_id

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            # Catch errors from validation, registry, or void field interaction
            error_msg = f"Failed to create Sephiroth field '{sephirah_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Ensure field is removed from registry if creation failed mid-way (though unlikely here)
            # if 'field' in locals() and hasattr(field,'field_id') and field.field_id in self.registry.fields:
            #     try: self.registry.delete_field(field.field_id)
            #     except Exception: pass # Ignore errors during rollback cleanup
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = f"Unexpected error creating Sephiroth field '{sephirah_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _get_sephirah_title(self, sephirah_name_lower: str) -> str:
        """
        Get the title for a Sephirah (case-insensitive). Includes Daath.

        Args:
            sephirah_name_lower: Lowercase name of the Sephirah.

        Returns:
            Title string (e.g., "Crown", "Knowledge"). Returns "Unknown" if not found.
        """
        titles = {
            "kether": "Crown", "chokmah": "Wisdom", "binah": "Understanding",
            "daath": "Knowledge", # Added Daath
            "chesed": "Mercy", "geburah": "Severity", "tiphareth": "Beauty",
            "netzach": "Victory", "hod": "Splendor", "yesod": "Foundation",
            "malkuth": "Kingdom"
        }
        return titles.get(sephirah_name_lower, "Unknown")

    def get_field(self, field_id: str) -> BaseField:
        """
        Get a field by ID via the registry.

        Args:
            field_id: ID of the field.

        Returns:
            Field instance.

        Raises:
            ValueError: If field_id is invalid or not found.
            TypeError: If field_id is not a string.
            RuntimeError: If registry is unavailable.
        """
        if not self.registry: raise RuntimeError("Field Registry not available.")
        # Let registry handle validation
        return self.registry.get_field(field_id)

    def get_field_by_name(self, name: str) -> Optional[BaseField]:
        """
        Get a field by name (first matching field, case-insensitive).

        Args:
            name: Name of the field. Must be a non-empty string.

        Returns:
            Field instance or None if not found.

        Raises:
            ValueError: If name is invalid.
            RuntimeError: If registry is unavailable.
        """
        if not self.registry: raise RuntimeError("Field Registry not available.")
        if not isinstance(name, str) or not name:
             raise ValueError("Field name must be a non-empty string.")

        name_lower = name.lower()
        for field in self.registry.fields.values():
            # Check if field name matches (case-insensitive)
            if hasattr(field, 'name') and isinstance(field.name, str) and field.name.lower() == name_lower:
                return field
        logger.debug(f"Field with name '{name}' not found.")
        return None

    def get_void_field(self) -> VoidField:
        """
        Get the Void field instance.

        Returns:
            VoidField instance.

        Raises:
            RuntimeError: If void field ID not set, field not found, or not a VoidField instance.
        """
        if not self.registry: raise RuntimeError("Field Registry not available.")
        if self.registry.void_field_id is None:
            raise RuntimeError("Void field ID not set in registry - system may not be initialized.")

        try:
            field = self.registry.get_field(self.registry.void_field_id)
            if not isinstance(field, VoidField):
                # This indicates a system state error
                raise RuntimeError(f"Field with Void ID '{self.registry.void_field_id}' is not a VoidField instance (Type: {type(field)}).")
            return field
        except ValueError as ve: # Catch if get_field fails
            raise RuntimeError(f"Void field (ID: {self.registry.void_field_id}) not found in registry.") from ve
        except Exception as e:
             raise RuntimeError(f"Unexpected error retrieving Void field: {e}") from e


    def get_guff_field(self) -> Optional[GuffField]:
        """
        Find and return the Guff field instance. Returns None if not found.

        Returns:
            Optional[GuffField]: Guff field instance or None.

        Raises:
            RuntimeError: If registry is unavailable.
        """
        if not self.registry: raise RuntimeError("Field Registry not available.")
        for field in self.registry.fields.values():
            if isinstance(field, GuffField):
                return field
        logger.debug("Guff field not found in the registry.")
        return None

    def get_kether_field(self) -> Optional[KetherField]:
        """
        Find and return the Kether field instance. Returns None if not found.

        Returns:
            Optional[KetherField]: Kether field instance or None.

        Raises:
            RuntimeError: If registry is unavailable.
        """
        if not self.registry: raise RuntimeError("Field Registry not available.")
        for field in self.registry.fields.values():
            # Check for specific KetherField type first
            if isinstance(field, KetherField):
                return field
            # Fallback check removed - registry should handle correct type creation
        logger.debug("Kether field not found in the registry.")
        return None

    def place_soul_in_field(self, soul_id: str, field_id: str, position: Optional[Tuple[float, float, float]] = None) -> bool:
        """
        Place a soul entity in a specific field. Handles transfer if soul is already present elsewhere.

        Args:
            soul_id: Unique ID of the soul.
            field_id: ID of the target field.
            position: Optional (x, y, z) position in the target field. If None, calculates a default position.

        Returns:
            True if soul was placed successfully.

        Raises:
            ValueError: If parameters are invalid, field not found, or placement fails validation within the field.
            TypeError: If arguments have incorrect types.
            RuntimeError: If registry unavailable or transfer/placement fails unexpectedly.
        """
        # --- Input Validation ---
        if not self.registry: raise RuntimeError("Field Registry not available.")
        if not isinstance(soul_id, str) or not soul_id: raise TypeError("soul_id must be a non-empty string.")
        if not isinstance(field_id, str) or not field_id: raise TypeError("field_id must be a non-empty string.")
        if position is not None:
            if not isinstance(position, tuple) or len(position) != 3 or not all(isinstance(c, (int, float)) for c in position):
                raise TypeError("Position, if provided, must be a tuple of 3 numbers.")

        logger.info(f"Placing/Moving soul {soul_id} into field {field_id}...")
        try:
            target_field = self.registry.get_field(field_id) # Raises ValueError if not found

            # Calculate default position if needed
            if position is None:
                dims = target_field.dimensions
                if not isinstance(dims, tuple) or len(dims) != 3 or not all(d > 0 for d in dims):
                     raise ValueError(f"Target field {field_id} has invalid dimensions: {dims}")
                center_pos = (dims[0] / 2, dims[1] / 2, dims[2] / 2)
                # Add slight random offset (e.g., within 10% of dimension)
                offset_factor = 0.1
                position = (
                    center_pos[0] + (random.random() - 0.5) * dims[0] * offset_factor,
                    center_pos[1] + (random.random() - 0.5) * dims[1] * offset_factor,
                    center_pos[2] + (random.random() - 0.5) * dims[2] * offset_factor
                )
                # Ensure calculated position is within bounds (BaseField.add_entity should validate this too)
                position = tuple(max(0, min(p, max_dim)) for p, max_dim in zip(position, dims))
                logger.debug(f"Calculated default position for soul {soul_id}: {position}")

            # Check if soul is currently in another field tracked by this system
            current_field_id = self.soul_field_locations.get(soul_id)

            if current_field_id:
                if current_field_id == field_id:
                    # Soul already in target field, update position (remove/add)
                    logger.debug(f"Soul {soul_id} already in field {field_id}. Updating position to {position}.")
                    try:
                         # Find existing entity data to attempt re-add on failure (robustness)
                         existing_entity_data = None
                         for entity in target_field.entities:
                             if isinstance(entity, dict) and entity.get('id') == soul_id:
                                 existing_entity_data = entity
                                 break

                         target_field.remove_entity(soul_id) # Raises ValueError if not found in field's list
                         target_field.add_entity(soul_id, position) # Raises ValueError on issues
                    except (ValueError, TypeError) as e:
                         logger.error(f"Failed to update position for soul {soul_id} in {field_id}: {e}")
                         # Attempt to re-add entity if it was removed but add failed
                         if existing_entity_data and not any(isinstance(e, dict) and e.get('id') == soul_id for e in target_field.entities):
                              try:
                                   old_pos = existing_entity_data.get('position')
                                   if old_pos: target_field.add_entity(soul_id, old_pos) # Add back at old pos
                              except Exception as rb_err: logger.error(f"Rollback failed for position update: {rb_err}")
                         raise RuntimeError(f"Failed position update for soul {soul_id}: {e}") from e
                    # Location record remains the same
                else:
                    # Soul is in a different field, perform transfer
                    logger.debug(f"Soul {soul_id} is in field {current_field_id}. Transferring to {field_id}.")
                    source_field = self.registry.get_field(current_field_id)
                    # Find current position in source field for registry transfer method
                    source_position = None
                    for entity in source_field.entities:
                        if isinstance(entity, dict) and entity.get('id') == soul_id:
                            source_position = entity.get('position')
                            if not (isinstance(source_position, tuple) and len(source_position) == 3):
                                 source_position = None # Invalid position stored
                            break
                    if source_position is None:
                         # This indicates an inconsistency between soul_field_locations and field state
                         logger.error(f"Inconsistency: Soul {soul_id} tracked in {current_field_id} but not found or position invalid in field's entity list.")
                         # Force add to target field instead of transfer
                         target_field.add_entity(soul_id, position) # Raises ValueError on issues
                         self.soul_field_locations[soul_id] = field_id # Update location record anyway
                    else:
                         # Use registry transfer (which calls remove/add on fields)
                         self.registry.transfer_entity(soul_id, current_field_id, field_id, source_position, position)
                         self.soul_field_locations[soul_id] = field_id # Update location record
            else:
                # Soul not currently tracked, add directly to target field
                logger.debug(f"Soul {soul_id} not tracked. Adding directly to field {field_id}.")
                target_field.add_entity(soul_id, position) # Raises ValueError on issues
                self.soul_field_locations[soul_id] = field_id # Add to location tracking

            logger.info(f"Successfully placed/moved soul {soul_id} in field {field_id} at position {position}")
            return True

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            # Catch specific errors from registry or field methods
            error_msg = f"Failed to place soul {soul_id} in field {field_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Clean up potentially inconsistent state? (e.g., remove from target if add failed after remove)
            # If transfer failed, soul might be stuck removed from source. This is complex.
            # Best approach is to log critical error and raise.
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = f"Unexpected error placing soul {soul_id} in field {field_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def remove_soul_from_field(self, soul_id: str) -> bool:
        """
        Remove a soul entity from its currently tracked field.

        Args:
            soul_id: ID of the soul to remove.

        Returns:
            True if soul was successfully removed.

        Raises:
            ValueError: If soul is not tracked or not found in its tracked field.
            TypeError: If soul_id is not a string.
            RuntimeError: If registry unavailable or removal fails unexpectedly.
        """
        # --- Input Validation ---
        if not self.registry: raise RuntimeError("Field Registry not available.")
        if not isinstance(soul_id, str) or not soul_id: raise TypeError("soul_id must be a non-empty string.")

        logger.info(f"Removing soul {soul_id} from its current field...")
        current_field_id = self.soul_field_locations.get(soul_id)
        if not current_field_id:
            raise ValueError(f"Soul {soul_id} is not currently tracked in any field by this system.")

        try:
            field = self.registry.get_field(current_field_id) # Raises ValueError if field gone
            # Remove from field instance (raises ValueError if not found in field's list)
            field.remove_entity(soul_id)
            # Remove from location tracking
            del self.soul_field_locations[soul_id]

            logger.info(f"Successfully removed soul {soul_id} from field {current_field_id}")
            return True

        except ValueError as ve: # Catch errors from get_field or remove_entity
             error_msg = f"Failed to remove soul {soul_id} from field {current_field_id}: {str(ve)}"
             logger.error(error_msg)
             # If remove_entity failed, the soul might still be tracked incorrectly.
             # Remove from tracking anyway to resolve inconsistency.
             if soul_id in self.soul_field_locations:
                  logger.warning(f"Removing soul {soul_id} from tracking despite error removing from field {current_field_id}.")
                  del self.soul_field_locations[soul_id]
             raise ve # Re-raise the original error
        except Exception as e:
            error_msg = f"Unexpected error removing soul {soul_id} from field {current_field_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def move_soul_to_kether(self, soul_id: str) -> str:
        """
        Convenience method to move a soul to the Kether field.

        Args:
            soul_id: ID of the soul.

        Returns:
            ID of the Kether field.

        Raises:
            RuntimeError: If Kether field not found or move fails.
            ValueError: If soul_id invalid.
            TypeError: If soul_id invalid type.
        """
        logger.info(f"Moving soul {soul_id} to Kether...")
        kether_field = self.get_kether_field() # Raises RuntimeError if registry unavailable
        if kether_field is None:
            raise RuntimeError("Cannot move soul: Kether field not found in the system.")

        try:
            # place_soul_in_field handles transfer logic internally
            self.place_soul_in_field(soul_id, kether_field.field_id)
            logger.info(f"Successfully moved soul {soul_id} to Kether field (ID: {kether_field.field_id})")
            return kether_field.field_id
        except (ValueError, TypeError, RuntimeError) as e:
             # Catch errors from place_soul_in_field
             error_msg = f"Failed to move soul {soul_id} to Kether: {str(e)}"
             logger.error(error_msg) # Log full error
             raise RuntimeError(error_msg) from e # Re-raise as RuntimeError for consistency
        except Exception as e:
             error_msg = f"Unexpected error moving soul {soul_id} to Kether: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    def move_soul_to_guff(self, soul_id: str, soul_data: Dict[str, Any]) -> str:
        """
        Move a soul entity to the Guff field and store its data.

        Args:
            soul_id: ID of the soul.
            soul_data: Data associated with the soul (required by GuffField.store_soul).

        Returns:
            ID of the Guff field.

        Raises:
            RuntimeError: If Guff field not found or operation fails.
            ValueError: If soul_id invalid or Guff storage fails validation.
            TypeError: If arguments invalid type.
        """
        # --- Input Validation ---
        if not isinstance(soul_id, str) or not soul_id: raise TypeError("soul_id must be a non-empty string.")
        if not isinstance(soul_data, dict): raise TypeError("soul_data must be a dictionary.")

        logger.info(f"Moving soul {soul_id} to Guff field for storage...")
        guff_field = self.get_guff_field() # Raises RuntimeError if registry unavailable
        if guff_field is None:
            raise RuntimeError("Cannot move soul: Guff field not found in the system.")

        try:
            # Remove from current field if tracked
            current_field_id = self.soul_field_locations.get(soul_id)
            if current_field_id:
                 # Check if already in Guff - if so, maybe just update data?
                 # For now, assume we always remove first if tracked elsewhere.
                 if current_field_id != guff_field.field_id:
                      self.remove_soul_from_field(soul_id) # Remove from tracking and field instance
                 else:
                      logger.warning(f"Soul {soul_id} is already tracked in Guff. Proceeding to store/update.")
                      # Ensure it's removed from Guff's *entity* list before storing as 'soul'
                      try: guff_field.remove_entity(soul_id)
                      except ValueError: pass # Ignore if not present as entity

            # Store in Guff field's internal storage (handles validation)
            guff_field.store_soul(soul_id, soul_data)

            # Update location tracking
            self.soul_field_locations[soul_id] = guff_field.field_id

            logger.info(f"Successfully moved and stored soul {soul_id} in Guff field (ID: {guff_field.field_id})")
            return guff_field.field_id

        except (ValueError, TypeError, RuntimeError) as e:
            # Catch errors from remove_soul_from_field or guff_field.store_soul
            error_msg = f"Failed to move soul {soul_id} to Guff: {str(e)}"
            logger.error(error_msg) # Log full error
            raise RuntimeError(error_msg) from e # Re-raise as RuntimeError
        except Exception as e:
             error_msg = f"Unexpected error moving soul {soul_id} to Guff: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    def move_soul_to_sephirah(self, soul_id: str, sephirah_name: str) -> str:
        """
        Move a soul entity to a specific Sephirah field by name.

        Args:
            soul_id: ID of the soul.
            sephirah_name: Name of the target Sephirah (case-insensitive).

        Returns:
            ID of the target Sephirah field.

        Raises:
            ValueError: If Sephirah name is invalid or field not found, or move fails validation.
            TypeError: If arguments invalid type.
            RuntimeError: If registry unavailable or move fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(soul_id, str) or not soul_id: raise TypeError("soul_id must be a non-empty string.")
        if not isinstance(sephirah_name, str) or not sephirah_name: raise ValueError("Sephirah name must be a non-empty string.")

        logger.info(f"Moving soul {soul_id} to {sephirah_name.capitalize()} field...")
        if not self.registry: raise RuntimeError("Field Registry not available.")

        sephirah_field = None
        target_field_id = None
        sephirah_lower = sephirah_name.lower()

        # Find the field by checking the sephiroth_name attribute
        for field_id, field in self.registry.fields.items():
            # Ensure field is a SephirothField or derivative and has the attribute
            if isinstance(field, SephirothField) and hasattr(field, 'sephiroth_name') and isinstance(field.sephiroth_name, str):
                 if field.sephiroth_name == sephirah_lower:
                    sephirah_field = field
                    target_field_id = field_id
                    break
            # Include Daath check specifically if it might not have sephiroth_name attribute standardly
            elif isinstance(field, DaathField) and sephirah_lower == "daath":
                 sephirah_field = field
                 target_field_id = field_id
                 break


        if sephirah_field is None:
            raise ValueError(f"Sephirah field '{sephirah_name}' not found in the registry.")

        try:
            # place_soul_in_field handles transfer logic
            self.place_soul_in_field(soul_id, target_field_id)
            logger.info(f"Successfully moved soul {soul_id} to {sephirah_field.name} (ID: {target_field_id})")
            return target_field_id
        except (ValueError, TypeError, RuntimeError) as e:
             # Catch errors from place_soul_in_field
             error_msg = f"Failed to move soul {soul_id} to {sephirah_name}: {str(e)}"
             logger.error(error_msg) # Log full error
             raise RuntimeError(error_msg) from e # Re-raise
        except Exception as e:
             error_msg = f"Unexpected error moving soul {soul_id} to {sephirah_name}: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    # --- calculate_soul_field_resonance, find_most_resonant_field ---
    # Review: Look solid, use field methods, handle errors. OK.

    def calculate_soul_field_resonance(self, soul_id: str, field_id: str, soul_frequency: float) -> float:
        """
        Calculate how strongly a soul resonates with a field.

        Args:
            soul_id: ID of the soul (used for logging).
            field_id: ID of the field.
            soul_frequency: Frequency of the soul (> 0).

        Returns:
            Resonance value between 0.0 and 1.0.

        Raises:
            ValueError: If parameters are invalid (field not found, frequency <= 0).
            TypeError: If arguments invalid type.
            RuntimeError: If registry unavailable or calculation fails unexpectedly.
        """
        # --- Input Validation ---
        if not self.registry: raise RuntimeError("Field Registry not available.")
        if not isinstance(soul_id, str) or not soul_id: raise TypeError("soul_id must be a non-empty string.")
        if not isinstance(field_id, str) or not field_id: raise TypeError("field_id must be a non-empty string.")
        if not isinstance(soul_frequency, (int, float)) or soul_frequency <= 0.0:
            raise ValueError("Soul frequency must be a positive number.")

        logger.debug(f"Calculating resonance: Soul {soul_id} (Freq: {soul_frequency:.2f}Hz) with Field {field_id}...")
        try:
            field = self.registry.get_field(field_id) # Raises ValueError if not found

            # Calculate base resonance using field's method
            if not hasattr(field, 'calculate_field_resonance'):
                 raise AttributeError(f"Field {field_id} (Type: {type(field)}) lacks 'calculate_field_resonance' method.")
            field_resonance = field.calculate_field_resonance(soul_frequency)

            # Check for aspect resonance if field has aspects
            aspect_resonance = 0.0
            if hasattr(field, 'aspects') and isinstance(field.aspects, dict) and field.aspects:
                if hasattr(field, 'calculate_aspect_resonance'):
                    aspect_resonances = []
                    # Iterate over aspect names (keys)
                    for aspect_name in field.aspects.keys():
                         try:
                             # Pass aspect_name, not the full data dict
                             resonance = field.calculate_aspect_resonance(aspect_name, soul_frequency)
                             if isinstance(resonance, (int, float)): # Validate return type
                                 aspect_resonances.append(resonance)
                             else:
                                 logger.warning(f"calculate_aspect_resonance for aspect '{aspect_name}' returned invalid type {type(resonance)}")
                         except (ValueError, TypeError, AttributeError) as aspect_err:
                              logger.warning(f"Could not calculate resonance for aspect '{aspect_name}': {aspect_err}")
                              # Continue with other aspects

                    if aspect_resonances:
                        # Calculate average aspect resonance
                        aspect_resonance = sum(aspect_resonances) / len(aspect_resonances)
                        logger.debug(f"Calculated average aspect resonance: {aspect_resonance:.4f}")
                    else: logger.debug("No valid aspect resonances calculated.")
                else: logger.debug(f"Field {field_id} has aspects but lacks 'calculate_aspect_resonance' method.")

            # Combine resonances (weighted average)
            overall_resonance = 0.7 * field_resonance + 0.3 * aspect_resonance
            # Clamp final result
            overall_resonance = max(0.0, min(1.0, overall_resonance))

            logger.debug(f"Overall resonance for Soul {soul_id} with Field {field_id}: {overall_resonance:.4f}")
            return float(overall_resonance)

        except (ValueError, TypeError, AttributeError) as e:
             # Catch specific errors from registry or field methods
             error_msg = f"Error calculating soul-field resonance ({soul_id} <-> {field_id}): {str(e)}"
             logger.error(error_msg) # Log full error
             raise RuntimeError(error_msg) from e # Re-raise as RuntimeError
        except Exception as e:
             error_msg = f"Unexpected error calculating soul-field resonance ({soul_id} <-> {field_id}): {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    def find_most_resonant_field(self, soul_id: str, soul_frequency: float) -> Tuple[str, float]:
        """
        Find the field in the registry with the strongest resonance for a given soul frequency.

        Args:
            soul_id: ID of the soul (for logging).
            soul_frequency: Frequency of the soul (> 0).

        Returns:
            Tuple of (most_resonant_field_id, highest_resonance_value).

        Raises:
            ValueError: If parameters are invalid or no fields are available.
            RuntimeError: If registry unavailable or calculation fails.
        """
        # --- Input Validation ---
        if not self.registry: raise RuntimeError("Field Registry not available.")
        if not isinstance(soul_id, str) or not soul_id: raise TypeError("soul_id must be a non-empty string.")
        if not isinstance(soul_frequency, (int, float)) or soul_frequency <= 0.0:
            raise ValueError("Soul frequency must be a positive number.")

        logger.info(f"Finding most resonant field for soul {soul_id} (Freq: {soul_frequency:.2f}Hz)...")
        if not self.registry.fields:
            raise ValueError("No fields available in registry for resonance calculation.")

        resonances = []
        # Calculate resonance with each field in the registry
        for field_id in list(self.registry.fields.keys()): # Iterate over copy of keys
            # Check if field still exists (might be deleted during iteration in complex scenarios)
            if field_id not in self.registry.fields: continue
            try:
                resonance = self.calculate_soul_field_resonance(soul_id, field_id, soul_frequency)
                resonances.append((field_id, resonance))
            except RuntimeError as e:
                # Log error for specific field but continue searching others
                logger.error(f"Could not calculate resonance for field {field_id}: {e}")

        if not resonances:
            raise RuntimeError("Failed to calculate resonance for any available fields.")

        # Find the field with the maximum resonance value
        most_resonant = max(resonances, key=lambda item: item[1])

        logger.info(f"Most resonant field found for soul {soul_id}: {most_resonant[0]} (Resonance: {most_resonant[1]:.4f})")
        return most_resonant


    # --- simulate_field_interactions ---
    # Review: Looks solid, uses helper methods, specific Guff/Kether logic, general resonance calc loop. Hard fails. OK.

    def create_tree_of_life(self) -> Dict[str, str]:
        """
        Create the complete Tree of Life structure with all 10 Sephiroth and Daath.
        Checks for existing fields before creating new ones.

        Returns:
            Dictionary mapping Sephirah names (lowercase) to their field IDs.

        Raises:
            RuntimeError: If system not initialized or creation fails critically.
        """
        if not self.initialized:
            raise RuntimeError("Field system must be initialized before creating Tree of Life.")
        if not self.registry: raise RuntimeError("Field Registry not available.")

        logger.info("Ensuring/Creating full Tree of Life structure...")
        sephiroth_fields: Dict[str, str] = {}
        all_sephiroth_names = [ # Canonical order + Daath
            "kether", "chokmah", "binah", "daath", "chesed", "geburah",
            "tiphareth", "netzach", "hod", "yesod", "malkuth"
        ]

        # --- Position Calculation ---
        # Get Void field dimensions to place the tree appropriately
        try:
             void_field = self.get_void_field()
             if not isinstance(void_field.dimensions, tuple) or len(void_field.dimensions) != 3 or not all(d > 0 for d in void_field.dimensions):
                  raise ValueError(f"Void field has invalid dimensions: {void_field.dimensions}")
        except (RuntimeError, ValueError) as e:
             raise RuntimeError(f"Cannot determine Tree of Life positions: {e}") from e

        void_dims = void_field.dimensions
        void_center = (void_dims[0] / 2, void_dims[1] / 2, void_dims[2] / 2)

        # Scale the conceptual 9x16 grid to fit within the void (e.g., using 1/3 of void width/height)
        grid_width_concept = 9.0
        grid_height_concept = 16.0 # From Kether(1) to Malkuth(16)
        tree_scale_x = (void_dims[0] / 3.0) / grid_width_concept
        tree_scale_y = (void_dims[1] / 3.0) / grid_height_concept # Use void Y for tree Height
        tree_scale_z = (void_dims[2] / 4.0) # Use void Z for tree Depth/flatness

        # Calculate positions based on conceptual grid and void center
        # Using SEPHIROTH_POSITIONS defined in tree_of_life.py (assuming accessible or redefined here)
        # If tree_of_life.py is not imported, need to define SEPHIROTH_POSITIONS here.
        # Let's assume it's available via constants or direct definition for simplicity:
        SEPHIROTH_POSITIONS_CONCEPT = { # Conceptual grid coordinates
            'Kether': (4.5, 1), 'Daath': (4.5, 4.75), 'Tipareth': (4.5, 9), 'Yesod': (4.5, 14), 'Malkuth': (4.5, 16),
            'Binah': (0, 3), 'Geburah': (0, 7.5), 'Hod': (0, 12),
            'Chokmah': (9, 3), 'Chesed': (9, 7.5), 'Netzach': (9, 12),
        }
        final_positions = {}
        tree_center_x = void_center[0]
        tree_center_y = void_center[1] # Center tree vertically in void Y
        tree_z_plane = void_center[2] # Place tree at void's Z center for now

        for name, (grid_x, grid_y) in SEPHIROTH_POSITIONS_CONCEPT.items():
            # Calculate offsets from conceptual center (4.5, 8.5) assuming Kether=1, Malkuth=16
            offset_x = grid_x - grid_width_concept / 2.0
            offset_y = grid_y - (grid_height_concept + 1) / 2.0 # Center around 8.5
            # Apply scaling and void center offset
            final_x = tree_center_x + offset_x * tree_scale_x
            final_y = tree_center_y + offset_y * tree_scale_y # Use void Y for tree Y/Height
            final_z = tree_z_plane # Keep planar for now
            final_positions[name.lower()] = (final_x, final_y, final_z)


        # --- Create/Find Fields ---
        for sephirah_lower in all_sephiroth_names:
            # Check if already exists by sephiroth_name attribute
            found_field = None
            for field_id, field in self.registry.fields.items():
                 if isinstance(field, SephirothField) and hasattr(field, 'sephiroth_name') and field.sephiroth_name == sephirah_lower:
                      found_field = field
                      break
                 # Specific check for Daath if it doesn't use standard attribute
                 elif isinstance(field, DaathField) and sephirah_lower == "daath":
                      found_field = field
                      break

            if found_field:
                sephiroth_fields[sephirah_lower] = found_field.field_id
                logger.info(f"Found existing {sephirah_lower.capitalize()} field (ID: {found_field.field_id}).")
            else:
                # Create the field if not found
                logger.info(f"Creating {sephirah_lower.capitalize()} field...")
                if sephirah_lower not in final_positions:
                     raise RuntimeError(f"Internal Error: Position not calculated for {sephirah_lower}.")
                position = final_positions[sephirah_lower]
                # Use create_sephiroth_field which handles defaults and specific types
                field_id = self.create_sephiroth_field(sephirah_lower, position)
                sephiroth_fields[sephirah_lower] = field_id

        # --- Create Connections ---
        logger.info("Connecting Sephiroth fields according to Tree of Life paths...")
        # Define paths (use canonical names)
        PATHS_CONCEPT = [ # Using canonical names for clarity
            ('Kether', 'Tipareth'), ('Tipareth', 'Yesod'), ('Yesod', 'Malkuth'),
            ('Kether', 'Daath'), ('Daath', 'Tipareth'),
            ('Kether', 'Binah'), ('Kether', 'Chokmah'), ('Binah', 'Chokmah'),
            ('Binah', 'Daath'), ('Chokmah', 'Daath'),
            ('Binah', 'Tipareth'), ('Chokmah', 'Tipareth'),
            ('Binah', 'Geburah'), ('Chokmah', 'Chesed'), ('Geburah', 'Chesed'),
            ('Geburah', 'Tipareth'), ('Chesed', 'Tipareth'),
            ('Geburah', 'Hod'), ('Chesed', 'Netzach'), ('Hod', 'Netzach'),
            ('Hod', 'Yesod'), ('Netzach', 'Yesod'),
            ('Tipareth', 'Hod'), ('Tipareth', 'Netzach')
        ]
        connections_created = 0
        connection_errors = 0
        for start_name, end_name in PATHS_CONCEPT:
            start_lower = start_name.lower()
            end_lower = end_name.lower()
            # Check if both fields exist in our map
            if start_lower in sephiroth_fields and end_lower in sephiroth_fields:
                try:
                    # Connect with default strength (can be refined)
                    self._connect_sephiroth(sephiroth_fields, start_lower, end_lower, "path", 0.7)
                    connections_created += 1
                except (ValueError, TypeError, RuntimeError) as conn_err:
                    logger.error(f"Failed to connect {start_lower} to {end_lower}: {conn_err}")
                    connection_errors += 1
            else:
                logger.error(f"Cannot connect path: Fields '{start_lower}' or '{end_lower}' not found in created map.")
                connection_errors += 1

        if connection_errors > 0:
             logger.warning(f"Completed Tree of Life creation with {connection_errors} connection errors.")
        else:
             logger.info(f"Successfully created {connections_created} connections for the Tree of Life.")

        logger.info(f"Finished creating Tree of Life structure with {len(sephiroth_fields)} fields.")
        return sephiroth_fields

        # Outer try-except for the whole method
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            error_msg = f"Failed during Tree of Life creation: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during Tree of Life creation: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _connect_sephiroth(self, sephiroth_fields_map: Dict[str, str], sephirah1_lower: str, sephirah2_lower: str,
                          connection_type: str, strength: float) -> bool:
        """
        Internal helper to connect two Sephiroth fields using their lowercase names and the map.

        Args:
            sephiroth_fields_map: Dictionary mapping lowercase Sephirah names to field IDs.
            sephirah1_lower: Lowercase name of first Sephirah.
            sephirah2_lower: Lowercase name of second Sephirah.
            connection_type: Type of connection.
            strength: Connection strength (0.0-1.0).

        Returns:
            True if connection was created successfully.

        Raises:
            ValueError: If Sephiroth names not found in map or connection fails validation.
            RuntimeError: If registry unavailable or connection fails unexpectedly.
        """
        if not self.registry: raise RuntimeError("Field Registry not available.")
        if sephirah1_lower not in sephiroth_fields_map:
            raise ValueError(f"Sephirah '{sephirah1_lower}' not found in provided field map.")
        if sephirah2_lower not in sephiroth_fields_map:
            raise ValueError(f"Sephirah '{sephirah2_lower}' not found in provided field map.")

        field1_id = sephiroth_fields_map[sephirah1_lower]
        field2_id = sephiroth_fields_map[sephirah2_lower]

        # Check if already connected (optional, registry connect handles overwrite warning)
        # if field1_id in self.registry.field_connections and field2_id in self.registry.field_connections[field1_id]:
        #     logger.debug(f"Connection between {sephirah1_lower} ({field1_id}) and {sephirah2_lower} ({field2_id}) already exists.")
        #     return True # Or False if overwrite is not desired

        try:
            # Use registry to create the bi-directional connection
            self.registry.connect_fields(field1_id, field2_id, connection_type, strength)
            logger.debug(f"Connected {sephirah1_lower} ({field1_id}) to {sephirah2_lower} ({field2_id})")
            return True
        except (ValueError, TypeError, RuntimeError) as e:
            # Log specific error but re-raise to indicate failure in the calling method
            logger.error(f"Connection failed between {sephirah1_lower} ({field1_id}) and {sephirah2_lower} ({field2_id}): {e}")
            raise e # Re-raise the caught error


    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the field system including registry and soul locations.

        Returns:
            Dictionary of system status information.

        Raises:
             RuntimeError: If registry is unavailable.
        """
        if not self.registry: raise RuntimeError("Field Registry not available.")
        logger.debug("Gathering system status...")
        try:
            # Get registry metrics
            registry_metrics = self.registry.get_registry_metrics()

            # Get entity counts per field
            field_entities_summary = {}
            for field_id, field in self.registry.fields.items():
                 # Check field validity before accessing attributes
                 if not isinstance(field, BaseField):
                      logger.warning(f"Invalid object found in registry fields (ID: {field_id}, Type: {type(field)}). Skipping.")
                      continue
                 field_entities_summary[field_id] = {
                    'name': getattr(field, 'name', 'Unnamed Field'),
                    'field_type': getattr(field, 'field_type', 'unknown'),
                    'entity_count': len(getattr(field, 'entities', [])),
                    'is_active': getattr(field, 'active', False)
                 }

            # Get special field info (check if they exist)
            void_info = None; kether_info = None; guff_info = None
            try: void_field = self.get_void_field(); void_info = void_field.get_field_metrics().get('void_specific') if void_field else None
            except RuntimeError: logger.warning("Could not get Void field for status.")
            try: kether_field = self.get_kether_field(); kether_info = kether_field.get_field_metrics().get('kether_specific') if kether_field else None
            except RuntimeError: logger.warning("Could not get Kether field for status.")
            try: guff_field = self.get_guff_field(); guff_info = guff_field.get_field_metrics().get('guff_specific') if guff_field else None
            except RuntimeError: logger.warning("Could not get Guff field for status.")


            # Combine all metrics
            status = {
                'system_info': {
                     'initialized': self.initialized,
                     'creation_time': self.creation_time,
                     'current_time': datetime.now().isoformat(),
                     'tracked_soul_count': len(self.soul_field_locations),
                },
                'registry_metrics': registry_metrics,
                'fields_summary': field_entities_summary,
                'soul_locations': self.soul_field_locations.copy(), # Return copy
                'special_fields': {
                    'void': void_info,
                    'kether': kether_info,
                    'guff': guff_info
                }
            }
            logger.debug("System status gathered successfully.")
            return status

        except Exception as e:
            # Catch any unexpected error during status gathering
            error_msg = f"Failed to get complete system status: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return minimal status with error
            return {
                'system_info': {
                     'initialized': self.initialized,
                     'creation_time': self.creation_time,
                     'current_time': datetime.now().isoformat(),
                     'error': error_msg
                 }
            }

    def __str__(self) -> str:
        """String representation of the field system."""
        field_count = len(self.registry.fields) if self.registry else 0
        return f"FieldSystem(initialized={self.initialized}, fields={field_count}, tracked_souls={len(self.soul_field_locations)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        field_count = len(self.registry.fields) if self.registry else 0
        return f"<FieldSystem initialized={self.initialized} fields={field_count} tracked_souls={len(self.soul_field_locations)} registry_ok={self.registry is not None}>"

# --- END OF (MODIFIED) field_system.py ---