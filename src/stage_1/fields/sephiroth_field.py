# --- START OF FILE sephiroth_field.py ---

"""
Sephiroth Field Module

Defines the SephirothField class which serves as the base for all Sephiroth-specific
dimensional fields. Implements the 40% base field, 50% Sephiroth-specific, 10% random
composition pattern.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from src.stage_1.fields.base_field import BaseField

# Configure logging
logger = logging.getLogger(__name__)

class SephirothField(BaseField):
    """
    Base class for all Sephiroth fields in the soul development framework.
    Implements the composition pattern of 40% base field, 50% Sephiroth-specific, 10% random.
    """
    
    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Sephiroth Field",
                 dimensions: Tuple[float, float, float] = (100.0, 100.0, 100.0),
                 base_frequency: float = 432.0,
                 resonance: float = 0.8,
                 stability: float = 0.85,
                 coherence: float = 0.8,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1,
                 sephiroth_name: str = "generic"):
        """
        Initialize a Sephiroth field with its composition ratios.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties (typically 0.4)
            sephiroth_ratio: Proportion of Sephiroth-specific properties (typically 0.5)
            random_ratio: Proportion of random variation (typically 0.1)
            sephiroth_name: Name of the specific Sephiroth (lowercase)
            
        Raises:
            ValueError: If any parameters are invalid
            TypeError: If parameters are of incorrect type
        """
        # Validate composition ratios
        total_ratio = base_field_ratio + sephiroth_ratio + random_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-5):
            raise ValueError(f"Field composition ratios must sum to 1.0, got {total_ratio}")
        
        if not 0.0 <= base_field_ratio <= 1.0:
            raise ValueError(f"Base field ratio must be between 0.0 and 1.0, got {base_field_ratio}")
            
        if not 0.0 <= sephiroth_ratio <= 1.0:
            raise ValueError(f"Sephiroth ratio must be between 0.0 and 1.0, got {sephiroth_ratio}")
            
        if not 0.0 <= random_ratio <= 1.0:
            raise ValueError(f"Random ratio must be between 0.0 and 1.0, got {random_ratio}")
            
        if not sephiroth_name:
            raise ValueError("Sephiroth name must be provided")
            
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
        self.field_type = "sephiroth"
        
        # Store composition ratios
        self.base_field_ratio = base_field_ratio
        self.sephiroth_ratio = sephiroth_ratio
        self.random_ratio = random_ratio
        
        # Sephiroth-specific properties
        self.sephiroth_name = sephiroth_name.lower()
        self.divine_attribute = ""  # To be set by derived classes
        self.geometric_correspondence = ""  # To be set by derived classes
        self.element = ""  # To be set by derived classes
        self.primary_color = ""  # To be set by derived classes
        self.aspects = {}  # Dictionary of aspects present in this Sephiroth
        
        # Composition verification
        self.composition_verified = False
        
        # Add to field properties
        self.field_properties.update({
            'sephiroth_name': self.sephiroth_name,
            'base_field_ratio': self.base_field_ratio,
            'sephiroth_ratio': self.sephiroth_ratio,
            'random_ratio': self.random_ratio
        })
        
        logger.info(f"Sephiroth Field '{self.name}' ({self.sephiroth_name}) initialized with composition ratios: Base={base_field_ratio:.2f}, Sephiroth={sephiroth_ratio:.2f}, Random={random_ratio:.2f}")
    
    def add_aspect(self, aspect_name: str, aspect_strength: float, aspect_data: Dict[str, Any]) -> bool:
        """
        Add a Sephiroth-specific aspect to this field.
        
        Args:
            aspect_name: Name of the aspect
            aspect_strength: Strength of the aspect (0.0-1.0)
            aspect_data: Additional data about the aspect
            
        Returns:
            True if aspect was added successfully
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= aspect_strength <= 1.0:
            raise ValueError(f"Aspect strength must be between 0.0 and 1.0, got {aspect_strength}")
            
        if aspect_name in self.aspects:
            raise ValueError(f"Aspect '{aspect_name}' already exists in this field")
            
        # Add the aspect
        self.aspects[aspect_name] = {
            'strength': aspect_strength,
            'data': aspect_data,
            'addition_time': datetime.now().isoformat()
        }
        
        logger.info(f"Added aspect '{aspect_name}' to {self.sephiroth_name} field with strength {aspect_strength:.2f}")
        return True
    
    def remove_aspect(self, aspect_name: str) -> bool:
        """
        Remove an aspect from this field.
        
        Args:
            aspect_name: Name of the aspect to remove
            
        Returns:
            True if aspect was removed
            
        Raises:
            ValueError: If aspect does not exist
        """
        if aspect_name not in self.aspects:
            raise ValueError(f"Aspect '{aspect_name}' not found in this field")
            
        # Remove the aspect
        del self.aspects[aspect_name]
        
        logger.info(f"Removed aspect '{aspect_name}' from {self.sephiroth_name} field")
        return True
    
    def verify_composition(self) -> bool:
        """
        Verify that the field's composition matches the expected ratios.
        This is a validation step to ensure field construction is correct.
        
        Returns:
            True if composition is verified
            
        Raises:
            RuntimeError: If composition verification fails
        """
        # Check if energy grid is initialized
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before verifying composition")
            
        # Get the contribution from each component
        base_contribution = np.mean(self._base_contribution) if hasattr(self, '_base_contribution') else 0.0
        sephiroth_contribution = np.mean(self._sephiroth_contribution) if hasattr(self, '_sephiroth_contribution') else 0.0
        random_contribution = np.mean(self._random_contribution) if hasattr(self, '_random_contribution') else 0.0
        
        # Calculate total
        total_contribution = base_contribution + sephiroth_contribution + random_contribution
        
        # Calculate actual ratios
        actual_base_ratio = base_contribution / total_contribution if total_contribution > 0 else 0.0
        actual_sephiroth_ratio = sephiroth_contribution / total_contribution if total_contribution > 0 else 0.0
        actual_random_ratio = random_contribution / total_contribution if total_contribution > 0 else 0.0
        
        # Check if ratios match expected values (within tolerance)
        tolerance = 0.02  # 2% tolerance
        
        base_ratio_match = abs(actual_base_ratio - self.base_field_ratio) <= tolerance
        sephiroth_ratio_match = abs(actual_sephiroth_ratio - self.sephiroth_ratio) <= tolerance
        random_ratio_match = abs(actual_random_ratio - self.random_ratio) <= tolerance
        
        # All must match
        if not (base_ratio_match and sephiroth_ratio_match and random_ratio_match):
            error_msg = f"Composition verification failed: " \
                       f"Base ratio: expected {self.base_field_ratio:.2f}, got {actual_base_ratio:.2f}; " \
                       f"Sephiroth ratio: expected {self.sephiroth_ratio:.2f}, got {actual_sephiroth_ratio:.2f}; " \
                       f"Random ratio: expected {self.random_ratio:.2f}, got {actual_random_ratio:.2f}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Set verification flag
        self.composition_verified = True
        logger.info(f"Composition verification successful for {self.sephiroth_name} field")
        return True
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the field's energy grid with the 40/50/10 composition pattern.
        
        Args:
            resolution: Number of points along each dimension
            
        Returns:
            The initialized energy grid
            
        Raises:
            ValueError: If resolution is invalid
        """
        if any(r <= 0 for r in resolution):
            raise ValueError("Grid resolution must be positive in all dimensions")
            
        # Create base field energy grid
        super().initialize_energy_grid(resolution)
        
        # Store this as the base contribution
        self._base_contribution = self._energy_grid.copy()
        
        # Create grid indices
        x, y, z = np.indices(resolution)
        x_norm = x / resolution[0]
        y_norm = y / resolution[1]
        z_norm = z / resolution[2]
        
        # Initialize Sephiroth-specific contribution
        # This will be overridden by subclasses
        self._sephiroth_contribution = np.zeros(resolution)
        
        # Initialize random contribution
        np.random.seed(42)  # Fixed seed for reproducibility
        self._random_contribution = np.random.rand(*resolution) * 0.2
        
        # Combine contributions with proper ratios
        self._energy_grid = (
            self.base_field_ratio * self._base_contribution +
            self.sephiroth_ratio * self._sephiroth_contribution +
            self.random_ratio * self._random_contribution
        )
        
        # Ensure values stay in valid range
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0)
        
        logger.debug(f"Initialized {self.sephiroth_name} field energy grid with resolution {resolution}")
        return self._energy_grid
    
    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Sephiroth-specific patterns to the energy grid.
        This is an abstract method that should be implemented by subclasses.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by specific Sephiroth field classes")
    
    def calculate_aspect_resonance(self, aspect_name: str, entity_frequency: float) -> float:
        """
        Calculate how strongly an aspect in this field resonates with an entity.
        
        Args:
            aspect_name: Name of the aspect to check
            entity_frequency: Frequency of the entity to check resonance with
            
        Returns:
            Resonance value between 0.0 and 1.0
            
        Raises:
            ValueError: If aspect does not exist or parameters are invalid
        """
        if aspect_name not in self.aspects:
            raise ValueError(f"Aspect '{aspect_name}' not found in this field")
            
        if entity_frequency <= 0:
            raise ValueError("Entity frequency must be positive")
            
        # Get aspect data
        aspect = self.aspects[aspect_name]
        aspect_strength = aspect['strength']
        
        # Get aspect frequency if available
        aspect_frequency = aspect['data'].get('frequency', self.base_frequency)
        
        # Calculate base resonance
        resonance = 1.0 - min(1.0, abs(aspect_frequency - entity_frequency) / max(aspect_frequency, entity_frequency))
        
        # Apply aspect strength modifier
        resonance *= aspect_strength
        
        # Apply field's inherent resonance property
        resonance *= self.resonance
        
        return float(min(1.0, max(0.0, resonance)))
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Sephiroth field's current state.
        
        Returns:
            Dictionary of Sephiroth field metrics
        """
        # Get base metrics
        base_metrics = super().get_field_metrics()
        
        # Add Sephiroth-specific metrics
        sephiroth_metrics = {
            'sephiroth_name': self.sephiroth_name,
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'aspect_count': len(self.aspects),
            'composition': {
                'base_field_ratio': self.base_field_ratio,
                'sephiroth_ratio': self.sephiroth_ratio,
                'random_ratio': self.random_ratio,
                'composition_verified': self.composition_verified
            }
        }
        
        # Add aspect metrics
        aspect_metrics = {}
        for name, aspect in self.aspects.items():
            aspect_metrics[name] = {
                'strength': aspect['strength'],
                'data': {k: v for k, v in aspect['data'].items() if k != 'complex_data'}  # Exclude complex data
            }
        
        # Combine metrics
        combined_metrics = {
            **base_metrics, 
            'sephiroth_specific': sephiroth_metrics,
            'aspects': aspect_metrics
        }
        
        return combined_metrics
    
    def __str__(self) -> str:
        """String representation of the Sephiroth field."""
        return f"SephirothField(name={self.name}, sephirah={self.sephiroth_name}, aspects={len(self.aspects)})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<SephirothField id='{self.field_id}' name='{self.name}' sephirah='{self.sephiroth_name}' aspects={len(self.aspects)}>"

# --- END OF FILE sephiroth_field.py ---


