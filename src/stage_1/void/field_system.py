"""
Base Implementation for 3D Fields
Provides common initialization and potential utility methods for 3D fields.
"""

import numpy as np
import logging
import os
import uuid
from datetime import datetime
import json  # For saving/loading state
from typing import Optional, Tuple, Dict, Any, Union

# --- Constants ---
try:
    from src.constants import DEFAULT_DIMENSIONS_3D, DEFAULT_FIELD_DTYPE, DATA_DIR_BASE
except ImportError:
    logging.warning("Could not import constants. Using fallback values in Field3D.")
    DEFAULT_DIMENSIONS_3D = (256, 256, 256)
    DEFAULT_FIELD_DTYPE = np.float32
    DATA_DIR_BASE = "data"

# Configure logging (consider moving to a central logging config)
log_file_path = os.path.join("logs", "field_3d_base.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file_path)
logger = logging.getLogger('field_3d_base')


class Field3D(object):  # Base class inheriting from object
    """Base implementation for 3D fields."""
    
    def __init__(self, field_name: str, dimensions: Tuple[int, int, int] = DEFAULT_DIMENSIONS_3D,
                 data_dir: str = DATA_DIR_BASE, initial_energy: float = 0.1, noise_level: float = 0.05):
        """
        Initialize a 3D field.
        
        Args:
            field_name (str): A unique name for this field instance.
            dimensions (tuple): The dimensions (x, y, z) of the 3D field.
            data_dir (str): Base directory to store data related to this field.
            initial_energy (float): Base energy level to initialize the field with.
            noise_level (float): Amplitude of initial random noise.
        """
        if not isinstance(dimensions, tuple) or len(dimensions) != 3 or not all(isinstance(d, int) and d > 0 for d in dimensions):
            raise ValueError(f"Dimensions must be a tuple of 3 positive integers, got {dimensions}")
        
        if not field_name or not isinstance(field_name, str):
            raise ValueError("Field name must be a non-empty string.")
            
        if not isinstance(initial_energy, (int, float)) or not (0.0 <= initial_energy <= 1.0): 
            logger.warning(f"Initial energy {initial_energy} outside [0,1]. Clamping.")
            initial_energy = max(0.0, min(1.0, initial_energy))
            
        if not isinstance(noise_level, (int, float)) or noise_level < 0: 
            logger.warning(f"Noise level {noise_level} invalid. Setting to 0.05.")
            noise_level = 0.05
            
        self.field_name = field_name
        self.dimensions = dimensions
        
        # Create a specific directory for this field instance's data
        self.data_dir = os.path.join(data_dir, "fields", field_name)  # fields/field_name
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create data directory {self.data_dir}: {e}")
            # Don't use a fallback path, fail gracefully
            raise RuntimeError(f"Cannot create data directory for field: {e}")
            
        self.uuid = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        
        # --- Core field array ---
        # Subclasses might add more arrays (e.g., harmonic_field, light_intensity)
        logger.debug(f"Initializing energy_potential for {self.field_name} with shape {self.dimensions}")
        self.energy_potential = np.random.normal(initial_energy, noise_level, self.dimensions).astype(DEFAULT_FIELD_DTYPE)
        
        # Ensure initial energy is within valid bounds [0, 1]
        self.energy_potential = np.clip(self.energy_potential, 0.0, 1.0)
        
        # --- Metrics tracking ---
        self.metrics = {
            "creation_time": self.creation_time,
            "last_update": self.creation_time,
            "energy_mean": float(np.mean(self.energy_potential)),
            "energy_std": float(np.std(self.energy_potential)),
            "energy_min": float(np.min(self.energy_potential)),
            "energy_max": float(np.max(self.energy_potential)),
        }
        
        logger.info(f"Field3D '{self.field_name}' (UUID: {self.uuid}) initialized with dimensions {self.dimensions}.")

    def update_base_metrics(self) -> None:
        """Updates basic energy metrics."""
        self.metrics['last_update'] = datetime.now().isoformat()
        self.metrics['energy_mean'] = float(np.mean(self.energy_potential))
        self.metrics['energy_std'] = float(np.std(self.energy_potential))
        self.metrics['energy_min'] = float(np.min(self.energy_potential))
        self.metrics['energy_max'] = float(np.max(self.energy_potential))
        logger.debug(f"Base metrics updated for {self.field_name}")

    def get_state(self) -> Dict[str, Any]:
        """Returns a dictionary representing the basic state of the field."""
        self.update_base_metrics()  # Ensure metrics are current
        state = {
            "field_name": self.field_name,
            "class": self.__class__.__name__,
            "uuid": self.uuid,
            "dimensions": self.dimensions,
            "creation_time": self.creation_time,
            "metrics": self.metrics.copy(),
            # Do NOT include the full energy_potential array here by default
        }
        return state

    def save_state(self, filename: Optional[str] = None, include_energy_array: bool = False) -> Optional[str]:
        """
        Saves the field state (metadata and optionally energy data) to files.
        
        Args:
            filename (str, optional): Base name for the saved files (without extension).
                                  Defaults to field_name + timestamp.
            include_energy_array (bool): If True, saves the large energy_potential array
                                     to a separate .npy file. Defaults to False.
        
        Returns:
            str: Path to the saved JSON state file, or None on failure.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.field_name}_state_{timestamp}"
            
        json_filepath = os.path.join(self.data_dir, f"{filename}.json")
        state_data = self.get_state()  # Get current state dict
        
        energy_array_path = None
        if include_energy_array:
            # Save energy_potential array to a separate .npy file
            energy_array_path = self._save_array(self.energy_potential, f"{filename}_energy")
            if energy_array_path:
                # Store the relative path in the JSON metadata
                state_data['energy_potential_path'] = os.path.basename(energy_array_path)
            else:
                logger.error("Failed to save energy potential array, state JSON will not reference it.")
                
        try:
            with open(json_filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)  # Use default=str for non-serializable types like numpy floats
                
            logger.info(f"Saved field state metadata to {json_filepath}")
            if energy_array_path:
                logger.info(f"Saved energy potential array to {energy_array_path}")
                
            return json_filepath
        except TypeError as te: 
            logger.error(f"Serialization error saving state to {json_filepath}: {te}. State data: {state_data}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to save field state to {json_filepath}: {e}", exc_info=True)
            return None

    def load_state(self, filepath: str, load_energy_array: bool = True) -> bool:
        """
        Loads the field state from a JSON file.
        
        Args:
            filepath (str): Path to the JSON state file.
            load_energy_array (bool): If True, attempts to load the energy_potential array
                                  from the path specified in the JSON. Defaults to True.
        
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if not os.path.exists(filepath) or not filepath.endswith(".json"):
            logger.error(f"State file not found or invalid: {filepath}")
            return False
            
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
                
            # --- Basic State Validation ---
            if state_data.get("field_name") != self.field_name:
                logger.warning(f"Loading state with mismatched field name ('{state_data.get('field_name')}' vs '{self.field_name}').")
                
            loaded_dims = tuple(state_data.get("dimensions", []))
            if loaded_dims != self.dimensions:
                logger.error(f"Cannot load state: Dimension mismatch ({loaded_dims} vs {self.dimensions}). Reinitialize field with correct dimensions.")
                # Or potentially resize self.energy_potential if desired and feasible
                return False
                
            # --- Restore Attributes ---
            self.uuid = state_data.get("uuid", self.uuid)
            self.creation_time = state_data.get("creation_time", self.creation_time)
            self.metrics = state_data.get("metrics", self.metrics)
            # Restore other simple attributes if added later
            
            # --- Load Energy Potential Array ---
            if load_energy_array:
                energy_path_relative = state_data.get("energy_potential_path")
                if energy_path_relative:
                    # Construct full path relative to the JSON file's directory
                    energy_path_absolute = os.path.join(os.path.dirname(filepath), energy_path_relative)
                    
                    if os.path.exists(energy_path_absolute):
                        try:
                            loaded_array = np.load(energy_path_absolute)
                            if loaded_array.shape == self.dimensions:
                                self.energy_potential = loaded_array.astype(DEFAULT_FIELD_DTYPE)
                                logger.info(f"Loaded energy potential array from {energy_path_absolute}")
                            else:
                                logger.error(f"Dimension mismatch for energy array {energy_path_absolute} ({loaded_array.shape} vs {self.dimensions}). Array not loaded.")
                        except Exception as load_err:
                            logger.error(f"Failed to load energy array from {energy_path_absolute}: {load_err}")
                    else:
                        logger.warning(f"Energy potential array file not found: {energy_path_absolute}. Array not loaded.")
                else:
                    logger.warning("State file does not contain 'energy_potential_path'. Array not loaded.")
                    
            logger.info(f"Successfully loaded field state from {filepath}")
            return True
            
        except json.JSONDecodeError as jde:
            logger.error(f"Failed to decode JSON state file {filepath}: {jde}")
            return False
        except Exception as e:
            logger.error(f"Failed to load field state from {filepath}: {e}", exc_info=True)
            return False

    def _save_array(self, array: np.ndarray, name_prefix: str) -> Optional[str]:
        """Helper to save a numpy array, returning the relative path."""
        # Uses the base filename derived from state save
        array_filename = f"{name_prefix}.npy"
        array_filepath = os.path.join(self.data_dir, array_filename)
        
        try:
            np.save(array_filepath, array)
            logger.debug(f"Saved array '{name_prefix}' to {array_filepath}")
            # Return the relative path for storage in JSON
            return array_filepath
        except Exception as e:
            logger.error(f"Failed to save array '{name_prefix}' to {array_filepath}: {e}")
            return None

    def _normalize_field(self) -> None:
        """Basic normalization by clipping. Subclasses should override if needed."""
        if hasattr(self, 'energy_potential'):
            self.energy_potential = np.clip(self.energy_potential, 0.0, 1.0)
            logger.debug(f"Normalized energy potential for {self.field_name}")
        # Subclasses should extend this to normalize other field attributes as needed

    def __str__(self) -> str:
        return f"Field3D(name='{self.field_name}', dimensions={self.dimensions}, uuid='{self.uuid}')"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.field_name}' dims={self.dimensions} uuid='{self.uuid}'>"
