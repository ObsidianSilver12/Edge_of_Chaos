"""
Field Base Module

Provides the abstract base class for all field implementations
(Void, Sephiroth influencers) in the Soul Development Framework.
Includes support for geometric patterns, harmonics, and sound parameters.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from shared.constants.constants import GRID_SIZE, FLOAT_EPSILON, GOLDEN_RATIO
except ImportError:
    logger.critical("CRITICAL: Failed to import constants. FieldBase cannot function.")
    raise

class FieldBase(ABC):
    """
    Abstract Base Class for spatial fields within the simulation.
    Defines the common interface and properties, including harmonic and geometric attributes.
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

    def apply_geometric_pattern(self, pattern_type: str, location: Tuple[int, int, int], 
                               radius: float, strength: float) -> None:
        """
        Apply a geometric pattern to the field at a specified location.
        
        Args:
            pattern_type (str): Type of geometric pattern (e.g., 'tetrahedron', 'flower_of_life').
            location (Tuple[int, int, int]): Center location for the pattern.
            radius (float): Radius of effect.
            strength (float): Strength of pattern application.
            
        Returns:
            None
        """
        if self.pattern_influence is None or self.energy is None: 
            logger.warning(f"Pattern influence or energy grid not initialized in field {self.name}.")
            return
            
        # Import here to avoid circular imports
        try:
            from stage_1.fields.field_harmonics import FieldHarmonics
            
            # Calculate bounding box
            x, y, z = location
            int_radius = int(np.ceil(radius))
            min_x = max(0, x - int_radius)
            max_x = min(self.grid_size[0], x + int_radius + 1)
            min_y = max(0, y - int_radius)
            max_y = min(self.grid_size[1], y + int_radius + 1)
            min_z = max(0, z - int_radius)
            max_z = min(self.grid_size[2], z + int_radius + 1)
            
            # Check if bounding box is valid
            if min_x >= max_x or min_y >= max_y or min_z >= max_z:
                logger.warning(f"Invalid bounding box for pattern {pattern_type} at {location}.")
                return
                
            # Get grid modifier from FieldHarmonics
            box_size = (max_x - min_x, max_y - min_y, max_z - min_z)
            pattern_grid = FieldHarmonics.generate_geometry_grid_modifier(pattern_type, box_size)
            
            # Calculate distance mask for strength falloff
            x_idx, y_idx, z_idx = np.meshgrid(
                np.arange(min_x, max_x), 
                np.arange(min_y, max_y), 
                np.arange(min_z, max_z), 
                indexing='ij'
            )
            dist_sq = (x_idx - x)**2 + (y_idx - y)**2 + (z_idx - z)**2
            radius_sq = radius**2
            
            # Create strength mask with distance falloff
            strength_mask = np.maximum(0, 1.0 - np.sqrt(dist_sq / max(FLOAT_EPSILON, radius_sq)))
            strength_mask[dist_sq > radius_sq] = 0.0
            
            # Apply pattern to field's pattern_influence grid
            influence_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
            pattern_effect = pattern_grid * strength_mask * strength
            
            # Add to existing pattern influence
            self.pattern_influence[influence_slice] += pattern_effect
            
            # Clamp to valid range
            self.pattern_influence[influence_slice] = np.clip(
                self.pattern_influence[influence_slice], 
                0.0, 
                1.0
            )
            
            logger.debug(f"Applied {pattern_type} geometric pattern at {location} with radius {radius}.")
        except ImportError:
            logger.error("Failed to import FieldHarmonics. Cannot apply geometric pattern.")
        except Exception as e:
            logger.error(f"Error applying geometric pattern: {e}", exc_info=True)

    def get_sound_parameters(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Get sound parameters for a specific location in the field.
        
        Args:
            coordinates (Tuple[int, int, int]): The (x, y, z) integer grid coordinates.
            
        Returns:
            Dict[str, Any]: Dictionary of sound parameters.
        """
        try:
            # Get field properties at coordinates
            props = self.get_properties_at(coordinates)
            
            # Import here to avoid circular imports
            from stage_1.fields.field_harmonics import FieldHarmonics
            
            # Get sound parameters from FieldHarmonics
            return FieldHarmonics.get_live_sound_parameters(self.name, props)
        except ImportError:
            logger.error("Failed to import FieldHarmonics. Cannot generate sound parameters.")
            return {}
        except Exception as e:
            logger.error(f"Error generating sound parameters: {e}", exc_info=True)
            return {}

    def get_sound_visualization(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Get visualization parameters for sound at a specific location.
        
        Args:
            coordinates (Tuple[int, int, int]): The (x, y, z) integer grid coordinates.
            
        Returns:
            Dict[str, Any]: Dictionary of visualization parameters.
        """
        try:
            # Get field properties and sound parameters
            props = self.get_properties_at(coordinates)
            sound_params = self.get_sound_parameters(coordinates)
            
            # Import here to avoid circular imports
            from stage_1.fields.field_harmonics import FieldHarmonics
            
            # Get visualization parameters
            return FieldHarmonics.generate_live_sound_visualization(
                sound_params, 
                self.name, 
                props
            )
        except ImportError:
            logger.error("Failed to import FieldHarmonics. Cannot generate sound visualization.")
            return {}
        except Exception as e:
            logger.error(f"Error generating sound visualization: {e}", exc_info=True)
            return {}

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