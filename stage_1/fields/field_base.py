# --- START OF FILE field_base.py ---

"""
Field Base Module

Provides the abstract base class for all field implementations
(Void, Sephiroth influencers) in the Soul Development Framework.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from constants.constants import GRID_SIZE, FLOAT_EPSILON
except ImportError:
    logger.critical("CRITICAL: Failed to import constants. FieldBase cannot function.")
    raise

class FieldBase(ABC):
    """
    Abstract Base Class for spatial fields within the simulation.
    Defines the common interface and properties.
    """

    def __init__(self, name: str, grid_size: Tuple[int, int, int] = GRID_SIZE):
        """
        Initialize the base field.

        Args:
            name (str): The unique name of the field or influencer.
            grid_size (Tuple[int, int, int]): The dimensions (x, y, z) of the grid.

        Raises:
            ValueError: If name is empty or grid_size is invalid.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Field name must be a non-empty string.")
        if not isinstance(grid_size, tuple) or len(grid_size) != 3 or not all(isinstance(d, int) and d > 0 for d in grid_size):
            raise ValueError(f"grid_size must be a tuple of 3 positive integers, got {grid_size}")

        self.name: str = name
        self.grid_size: Tuple[int, int, int] = grid_size
        self.dimensions: int = 3 # Hardcoded for now

        # --- Core Field Property Grids ---
        # These grids store the state of the field at each point.
        # Concrete subclasses (like VoidField) will initialize these.
        # SephirothField influencers will *modify* the VoidField's grids.
        self.energy: Optional[np.ndarray] = None
        self.frequency: Optional[np.ndarray] = None
        self.coherence: Optional[np.ndarray] = None
        self.pattern_influence: Optional[np.ndarray] = None # Stores IDs or weights of geometric patterns
        self.color: Optional[np.ndarray] = None # Stores (R, G, B) or similar color info
        self.chaos: Optional[np.ndarray] = None # Local chaos level
        self.order: Optional[np.ndarray] = None # Local order level

        logger.info(f"FieldBase '{self.name}' initialized structure for grid size {self.grid_size}.")

    @abstractmethod
    def initialize_grid(self) -> None:
        """
        Abstract method to set up the initial state of the field's grids.
        Must be implemented by subclasses that own a grid (like VoidField).
        """
        pass

    @abstractmethod
    def update_step(self, delta_time: float) -> None:
        """
        Abstract method to simulate the dynamic evolution of the field over one time step.
        Must be implemented by subclasses that own a grid and have dynamics.

        Args:
            delta_time (float): The time elapsed since the last update step.
        """
        pass

    @abstractmethod
    def get_properties_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Abstract method to get the field properties at a specific grid coordinate.

        Args:
            coordinates (Tuple[int, int, int]): The (x, y, z) integer grid coordinates.

        Returns:
            Dict[str, Any]: A dictionary of properties (energy, frequency, etc.) at that point.

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        pass

    @abstractmethod
    def apply_influence(self, position: Tuple[float, float, float], influence_type: str, strength: float, radius: float) -> None:
        """
        Abstract method to apply an external influence (like a soul's presence) to the field.

        Args:
            position (Tuple[float, float, float]): The center position of the influence.
            influence_type (str): Type of influence (e.g., 'stabilizing', 'chaotic', 'frequency_shift').
            strength (float): The magnitude of the influence.
            radius (float): The radius of effect of the influence.
        """
        pass

    def _validate_coordinates(self, coordinates: Tuple[int, int, int]) -> bool:
        """Checks if grid coordinates are within the field bounds."""
        x, y, z = coordinates
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]):
            return False
        return True

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the field grid."""
        return self.grid_size

    def __str__(self) -> str:
        return f"FieldBase(Name: {self.name}, Size: {self.grid_size})"

    def __repr__(self) -> str:
        return f"<FieldBase name='{self.name}' size={self.grid_size}>"

# --- END OF FILE field_base.py ---