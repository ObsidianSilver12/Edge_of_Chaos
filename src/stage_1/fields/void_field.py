# --- START OF FILE void_field.py ---

"""
Void Field Module

Defines the VoidField class, which represents the all-encompassing dimensional
container that holds all other fields within it. This is the outermost field
in the soul development framework hierarchy.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from src.stage_1.fields.base_field import BaseField

# Configure logging
logger = logging.getLogger(__name__)

class VoidField(BaseField):
    """
    Represents the Void field - the all-encompassing outer dimensional container
    that contains all other fields within the soul development framework.

    Unlike other fields, the Void field is not contained within any other field
    and serves as the primordial space from which all other dimensions emerge.
    """
    
    def __init__(self, 
                 field_id: Optional[str] = None,
                 name: str = "The Void",
                 dimensions: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0),
                 base_frequency: float = 369.0,
                 resonance: float = 0.9,
                 stability: float = 0.99,
                 coherence: float = 0.95):
        """
        Initialize the Void field with its unique properties.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field (typically very large)
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            
        Raises:
            ValueError: If any parameters are invalid
            TypeError: If parameters are of incorrect type
        """
        # Validate void-specific parameters
        if dimensions[0] <= 100 or dimensions[1] <= 100 or dimensions[2] <= 100:
            raise ValueError("Void dimensions must be large enough to contain other fields (>100 in each dimension)")
        
        # Initialize the base field
        super().__init__(
            field_id=field_id,
            name=name,
            dimensions=dimensions,
            base_frequency=base_frequency,
            resonance=resonance,
            stability=stability,
            coherence=coherence
        )
        
        # Override the field type
        self.field_type = "void"
        
        # Void-specific properties
        self.is_bounded = False  # The void has no external boundaries
        self.contained_fields = {}  # Dictionary of fields contained within the void
        self.primordial_energy = 1.0  # Maximum primordial energy
        self.expansion_rate = 0.001  # Rate at which the void expands naturally
        self.creation_capacity = 1.0  # Capacity to create new dimensions
        
        # Void energy properties
        self.void_properties = {
            'primordial_energy': self.primordial_energy,
            'expansion_rate': self.expansion_rate,
            'creation_capacity': self.creation_capacity,
            'is_bounded': self.is_bounded
        }
        
        logger.info(f"Void Field '{self.name}' (ID: {self.field_id}) initialized with dimensions {dimensions}")
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
        """
        Initialize the Void's energy grid with specific void properties.
        
        Args:
            resolution: Number of points along each dimension
            
        Returns:
            The initialized energy grid
            
        Raises:
            ValueError: If resolution is invalid
        """
        if any(r <= 0 for r in resolution):
            raise ValueError("Grid resolution must be positive in all dimensions")
            
        # Create baseline energy grid using parent method
        super().initialize_energy_grid(resolution)
        
        # Add void-specific energy patterns
        # The void has high stability but varying energy density
        
        # Get grid indices
        x, y, z = np.indices(resolution)
        x_norm = x / resolution[0]
        y_norm = y / resolution[1]
        z_norm = z / resolution[2]
        
        # Calculate distance from center
        center_x, center_y, center_z = 0.5, 0.5, 0.5
        distance = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2 + (z_norm - center_z)**2)
        distance = distance / np.sqrt(3)  # Normalize to [0, 1]
        
        # Create energy density gradient - higher near center
        density_gradient = 1.0 - 0.5 * distance
        
        # Apply to energy grid with void modifiers
        void_stability_modifier = 0.2 * np.sin(distance * 10 * np.pi)
        void_energy = self.stability * (1.0 + void_stability_modifier) * density_gradient
        
        # Add some void fluctuations - random but stable patterns
        seed = 42  # Fixed seed for reproducibility
        np.random.seed(seed)
        void_fluctuations = 0.1 * np.random.rand(*resolution) * density_gradient
        
        # Combine everything
        self._energy_grid = void_energy + void_fluctuations
        
        # Ensure values stay in valid range
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0)
        
        logger.debug(f"Void Field energy grid initialized with resolution {resolution}")
        return self._energy_grid
    
    def add_contained_field(self, field_id: str, field_type: str, position: Tuple[float, float, float], 
                          dimensions: Tuple[float, float, float]) -> bool:
        """
        Add a field contained within the Void.
        
        Args:
            field_id: Unique identifier for the contained field
            field_type: Type of the field (e.g., "sephiroth", "guff")
            position: (x, y, z) position of the field's center in the Void
            dimensions: (x, y, z) dimensions of the contained field
            
        Returns:
            True if field was added successfully
            
        Raises:
            ValueError: If position/dimensions are invalid or field already exists
        """
        # Validate position is within void boundaries
        for i, (coord, (min_bound, max_bound)) in enumerate(zip(position, self.boundaries)):
            if coord < min_bound or coord > max_bound:
                raise ValueError(f"Field position {position} is outside void boundaries")
        
        # Validate dimensions to ensure the field fits inside the void
        for i in range(3):
            if dimensions[i] <= 0:
                raise ValueError(f"Field dimension {i} must be positive")
            
            min_pos = position[i] - dimensions[i]/2
            max_pos = position[i] + dimensions[i]/2
            
            if min_pos < self.boundaries[i][0] or max_pos > self.boundaries[i][1]:
                raise ValueError(f"Field extends beyond void boundaries at dimension {i}")
        
        # Check if field already exists
        if field_id in self.contained_fields:
            raise ValueError(f"Field {field_id} already exists in the void")
            
        # Add the field
        self.contained_fields[field_id] = {
            'field_type': field_type,
            'position': position,
            'dimensions': dimensions,
            'addition_time': datetime.now().isoformat()
        }
        
        logger.info(f"Added {field_type} field {field_id} to void at position {position} with dimensions {dimensions}")
        return True
    
    def remove_contained_field(self, field_id: str) -> bool:
        """
        Remove a field from the Void.
        
        Args:
            field_id: ID of field to remove
            
        Returns:
            True if field was removed
            
        Raises:
            ValueError: If field does not exist in the void
        """
        if field_id not in self.contained_fields:
            raise ValueError(f"Field {field_id} not found in the void")
            
        # Remove the field
        field_info = self.contained_fields.pop(field_id)
        
        logger.info(f"Removed {field_info['field_type']} field {field_id} from void")
        return True
    
    def check_field_overlap(self, field_id1: str, field_id2: str) -> float:
        """
        Check the degree of overlap between two contained fields.
        
        Args:
            field_id1: ID of first field
            field_id2: ID of second field
            
        Returns:
            Overlap coefficient (0.0 for no overlap, 1.0 for complete overlap)
            
        Raises:
            ValueError: If either field does not exist
        """
        if field_id1 not in self.contained_fields:
            raise ValueError(f"Field {field_id1} not found in the void")
            
        if field_id2 not in self.contained_fields:
            raise ValueError(f"Field {field_id2} not found in the void")
            
        # Get field information
        field1 = self.contained_fields[field_id1]
        field2 = self.contained_fields[field_id2]
        
        # Calculate overlap in each dimension
        overlap_volume = 1.0
        total_volume = 0.0
        
        for i in range(3):
            min1 = field1['position'][i] - field1['dimensions'][i]/2
            max1 = field1['position'][i] + field1['dimensions'][i]/2
            min2 = field2['position'][i] - field2['dimensions'][i]/2
            max2 = field2['position'][i] + field2['dimensions'][i]/2
            
            # Calculate overlap in this dimension
            overlap = max(0, min(max1, max2) - max(min1, min2))
            overlap_volume *= overlap
            
            # Calculate total volume (for normalization)
            dim1 = max1 - min1
            dim2 = max2 - min2
            total_volume += dim1 * dim2
        
        # No overlap
        if overlap_volume <= 0:
            return 0.0
            
        # Normalize overlap
        total_volume = field1['dimensions'][0] * field1['dimensions'][1] * field1['dimensions'][2] + \
                      field2['dimensions'][0] * field2['dimensions'][1] * field2['dimensions'][2]
                      
        overlap_coefficient = (2 * overlap_volume) / total_volume
        
        return float(min(1.0, max(0.0, overlap_coefficient)))
    
    def expand(self, expansion_factor: float) -> bool:
        """
        Expand the Void field dimensions.
        
        Args:
            expansion_factor: Factor by which to expand (1.0 = no change, 2.0 = double)
            
        Returns:
            True if expansion was successful
            
        Raises:
            ValueError: If expansion_factor is invalid
        """
        if expansion_factor <= 0:
            raise ValueError("Expansion factor must be positive")
            
        if expansion_factor < 1.0:
            raise ValueError("Void cannot contract, expansion factor must be >= 1.0")
            
        # Store old dimensions
        old_dimensions = self.dimensions
        
        # Calculate new dimensions
        new_dimensions = (
            old_dimensions[0] * expansion_factor,
            old_dimensions[1] * expansion_factor,
            old_dimensions[2] * expansion_factor
        )
        
        # Update dimensions
        self.dimensions = new_dimensions
        self.volume = new_dimensions[0] * new_dimensions[1] * new_dimensions[2]
        self.center = (new_dimensions[0]/2, new_dimensions[1]/2, new_dimensions[2]/2)
        self.boundaries = [(0, new_dimensions[0]), (0, new_dimensions[1]), (0, new_dimensions[2])]
        
        logger.info(f"Expanded Void from {old_dimensions} to {new_dimensions} (factor: {expansion_factor})")
        return True
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the void field's current state.
        
        Returns:
            Dictionary of void field metrics
        """
        # Get base metrics
        base_metrics = super().get_field_metrics()
        
        # Add void-specific metrics
        void_metrics = {
            'contained_field_count': len(self.contained_fields),
            'contained_field_types': self._count_field_types(),
            'primordial_energy': self.primordial_energy,
            'expansion_rate': self.expansion_rate,
            'creation_capacity': self.creation_capacity,
            'is_bounded': self.is_bounded
        }
        
        # Combine metrics
        combined_metrics = {**base_metrics, 'void_specific': void_metrics}
        
        return combined_metrics