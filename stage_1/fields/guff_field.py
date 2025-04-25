# --- START OF FILE guff_field.py ---

"""
Guff Field Module

Defines the GuffField class which represents the specific pocket dimension within Kether.
The Guff is where souls are stored before incarnation, often called the "Treasury of Souls".

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from stage_1.fields.base_field import BaseField
from stage_1.fields.sephiroth_field import SephirothField

# Configure logging
logger = logging.getLogger(__name__)

class GuffField(BaseField):
    """
    Represents the Guff field - a pocket dimension within Kether that serves
    as the "Treasury of Souls" where souls wait before incarnation.
    
    The Guff field has properties of both Kether (as its parent dimension) and
    unique properties for soul storage and preparation.
    """
    
    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Guff - Treasury of Souls",
                 dimensions: Tuple[float, float, float] = (88.0, 88.0, 88.0),
                 base_frequency: float = 963.0,  # Higher frequency for soul preparation
                 resonance: float = 0.92,
                 stability: float = 0.95,
                 coherence: float = 0.93,
                 kether_field_id: Optional[str] = None,
                 kether_influence_ratio: float = 0.4,
                 guff_specific_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Guff field with its unique properties.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            kether_field_id: ID of the parent Kether field
            kether_influence_ratio: Proportion of Kether influence (typically 0.4)
            guff_specific_ratio: Proportion of Guff-specific properties (typically 0.5)
            random_ratio: Proportion of random variation (typically 0.1)
            
        Raises:
            ValueError: If any parameters are invalid
        """
        # Validate composition ratios
        total_ratio = kether_influence_ratio + guff_specific_ratio + random_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-5):
            raise ValueError(f"Field composition ratios must sum to 1.0, got {total_ratio}")
        
        if not 0.0 <= kether_influence_ratio <= 1.0:
            raise ValueError(f"Kether influence ratio must be between 0.0 and 1.0, got {kether_influence_ratio}")
            
        if not 0.0 <= guff_specific_ratio <= 1.0:
            raise ValueError(f"Guff specific ratio must be between 0.0 and 1.0, got {guff_specific_ratio}")
            
        if not 0.0 <= random_ratio <= 1.0:
            raise ValueError(f"Random ratio must be between 0.0 and 1.0, got {random_ratio}")
        
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
        self.field_type = "guff"
        
        # Store composition ratios
        self.kether_influence_ratio = kether_influence_ratio
        self.guff_specific_ratio = guff_specific_ratio
        self.random_ratio = random_ratio
        
        # Kether connection
        self.kether_field_id = kether_field_id
        self.kether_connection_strength = 0.95  # Strong connection to Kether
        
        # Guff-specific properties
        self.soul_capacity = 144000  # Traditional number of souls in Guff
        self.souls_stored = []  # List of soul IDs stored in Guff
        self.soul_maturation_rate = 0.02  # Rate at which souls mature per time unit
        self.incarnation_readiness_threshold = 0.85  # Threshold for incarnation readiness
        
        # Divine properties
        self.divine_light_intensity = 0.95
        self.wisdom_infusion_rate = 0.03
        self.name_inscription_enabled = True  # Whether soul names can be inscribed
        
        # Guff composition verification
        self.composition_verified = False
        
        # Add to field properties
        self.field_properties.update({
            'kether_field_id': self.kether_field_id,
            'kether_influence_ratio': self.kether_influence_ratio,
            'guff_specific_ratio': self.guff_specific_ratio,
            'random_ratio': self.random_ratio,
            'soul_capacity': self.soul_capacity,
            'divine_light_intensity': self.divine_light_intensity
        })
        
        logger.info(f"Guff Field '{self.name}' initialized with soul capacity {self.soul_capacity} and connection to Kether {self.kether_connection_strength:.2f}")
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Guff field's energy grid with the composition pattern.
        
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
        
        # Initialize Kether influence
        # This would normally come from the actual Kether field
        # but we'll simulate it here
        self._kether_contribution = np.zeros(resolution)
        
        # Create a Kether-like pattern (crown energy)
        # Kether is characterized by pure white light, unity, and crown energy
        center_x, center_y, center_z = 0.5, 0.5, 0.5
        distance = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2 + (z_norm - center_z)**2)
        distance = distance / np.sqrt(3)  # Normalize to [0, 1]
        
        # Kether crown pattern - radial with upward emphasis
        crown_pattern = np.exp(-5 * distance) * (1.0 + 0.5 * z_norm)
        
        # Unity pattern - even, stable light
        unity_pattern = 0.8 + 0.2 * np.sin(x_norm * y_norm * z_norm * np.pi)
        
        # Combine into Kether contribution
        self._kether_contribution = 0.6 * crown_pattern + 0.4 * unity_pattern
        
        # Initialize Guff-specific contribution
        self._guff_contribution = np.zeros(resolution)
        
        # Create Guff-specific patterns
        
        # Soul storage pattern - like a honeycomb or matrix
        cell_size = 0.1
        honeycomb = 0.5 + 0.5 * np.sin(x_norm / cell_size * 2 * np.pi) * \
                          np.sin(y_norm / cell_size * 2 * np.pi) * \
                          np.sin(z_norm / cell_size * 2 * np.pi)
        
        # Divine light pattern - emanating from above
        divine_light = self.divine_light_intensity * (0.5 + 0.5 * z_norm)
        
        # Wisdom infusion pattern - rippling waves
        wisdom_waves = 0.5 + 0.5 * np.sin(4 * np.pi * (
            x_norm + y_norm + z_norm + 
            0.1 * np.sin(3 * np.pi * (x_norm + y_norm + z_norm))
        ))
        
        # Combine into Guff contribution
        self._guff_contribution = (
            0.4 * honeycomb +
            0.4 * divine_light +
            0.2 * wisdom_waves
        )
        
        # Initialize random contribution
        np.random.seed(42)  # Fixed seed for reproducibility
        self._random_contribution = np.random.rand(*resolution) * 0.2
        
        # Combine contributions with proper ratios
        self._energy_grid = (
            self.kether_influence_ratio * self._kether_contribution +
            self.guff_specific_ratio * self._guff_contribution +
            self.random_ratio * self._random_contribution
        )
        
        # Ensure values stay in valid range
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0)
        
        # Attempt to verify composition
        try:
            self.verify_composition()
        except RuntimeError as e:
            logger.warning(f"Composition verification failed: {e}")
            # Continue without failing - this is initialization
        
        logger.debug(f"Initialized Guff field energy grid with resolution {resolution}")
        return self._energy_grid
    
    def verify_composition(self) -> bool:
        """
        Verify that the field's composition matches the expected ratios.
        
        Returns:
            True if composition is verified
            
        Raises:
            RuntimeError: If composition verification fails
        """
        # Check if energy grid is initialized
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before verifying composition")
            
        # Get the contribution from each component
        kether_contribution = np.mean(self._kether_contribution) if hasattr(self, '_kether_contribution') else 0.0
        guff_contribution = np.mean(self._guff_contribution) if hasattr(self, '_guff_contribution') else 0.0
        random_contribution = np.mean(self._random_contribution) if hasattr(self, '_random_contribution') else 0.0
        
        # Calculate total
        total_contribution = kether_contribution + guff_contribution + random_contribution
        
        # Calculate actual ratios
        actual_kether_ratio = kether_contribution / total_contribution if total_contribution > 0 else 0.0
        actual_guff_ratio = guff_contribution / total_contribution if total_contribution > 0 else 0.0
        actual_random_ratio = random_contribution / total_contribution if total_contribution > 0 else 0.0
        
        # Check if ratios match expected values (within tolerance)
        tolerance = 0.02  # 2% tolerance
        
        kether_ratio_match = abs(actual_kether_ratio - self.kether_influence_ratio) <= tolerance
        guff_ratio_match = abs(actual_guff_ratio - self.guff_specific_ratio) <= tolerance
        random_ratio_match = abs(actual_random_ratio - self.random_ratio) <= tolerance
        
        # All must match
        if not (kether_ratio_match and guff_ratio_match and random_ratio_match):
            error_msg = f"Composition verification failed: " \
                       f"Kether ratio: expected {self.kether_influence_ratio:.2f}, got {actual_kether_ratio:.2f}; " \
                       f"Guff ratio: expected {self.guff_specific_ratio:.2f}, got {actual_guff_ratio:.2f}; " \
                       f"Random ratio: expected {self.random_ratio:.2f}, got {actual_random_ratio:.2f}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Set verification flag
        self.composition_verified = True
        logger.info(f"Composition verification successful for Guff field")
        return True
    
    def store_soul(self, soul_id: str, soul_data: Dict[str, Any]) -> bool:
        """
        Store a soul in the Guff field.
        
        Args:
            soul_id: Unique identifier for the soul
            soul_data: Data associated with the soul
            
        Returns:
            True if soul was stored successfully
            
        Raises:
            ValueError: If soul already exists or capacity is reached
        """
        # Check if soul is already stored
        if any(s['id'] == soul_id for s in self.souls_stored):
            raise ValueError(f"Soul {soul_id} is already stored in Guff")
            
        # Check capacity
        if len(self.souls_stored) >= self.soul_capacity:
            raise ValueError(f"Guff field has reached maximum soul capacity ({self.soul_capacity})")
            
        # Prepare soul storage record
        soul_record = {
            'id': soul_id,
            'storage_time': datetime.now().isoformat(),
            'maturity': soul_data.get('maturity', 0.1),  # Initial maturity
            'divine_light_exposure': 0.0,
            'wisdom_infusion': 0.0,
            'incarnation_readiness': 0.0,
            'data': soul_data
        }
        
        # Add soul to storage
        self.souls_stored.append(soul_record)
        
        # Also add as entity in the field
        position = self._calculate_soul_position(len(self.souls_stored) - 1)
        self.add_entity(soul_id, position)
        
        logger.info(f"Soul {soul_id} stored in Guff field at position {position}")
        return True
    
    def _calculate_soul_position(self, index: int) -> Tuple[float, float, float]:
        """
        Calculate position for a soul based on its index.
        
        Args:
            index: Index of the soul in storage
            
        Returns:
            (x, y, z) position coordinates
        """
        # Create a spiral pattern for soul storage
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        theta = 2 * np.pi * phi * index
        
        # Radius grows with index, but levels out
        radius = min(0.9, 0.1 + 0.4 * (1 - np.exp(-index / 1000)))
        
        # Z-position varies with index in a wave pattern
        z = 0.5 + 0.4 * np.sin(index / 100 * np.pi)
        
        # Calculate position
        x = 0.5 + radius * np.cos(theta)
        y = 0.5 + radius * np.sin(theta)
        
        # Convert to absolute coordinates
        x_abs = x * self.dimensions[0]
        y_abs = y * self.dimensions[1]
        z_abs = z * self.dimensions[2]
        
        return (x_abs, y_abs, z_abs)
    
    def retrieve_soul(self, soul_id: str) -> Dict[str, Any]:
        """
        Retrieve a soul from the Guff field.
        
        Args:
            soul_id: ID of soul to retrieve
            
        Returns:
            Soul record data
            
        Raises:
            ValueError: If soul is not found
        """
        # Find soul
        soul_index = None
        for i, soul in enumerate(self.souls_stored):
            if soul['id'] == soul_id:
                soul_index = i
                break
                
        if soul_index is None:
            raise ValueError(f"Soul {soul_id} not found in Guff field")
            
        # Retrieve soul
        soul_record = self.souls_stored.pop(soul_index)
        
        # Remove entity from field
        try:
            self.remove_entity(soul_id)
        except ValueError:
            logger.warning(f"Entity {soul_id} not found in field entities when removing")
        
        logger.info(f"Soul {soul_id} retrieved from Guff field")
        return soul_record
    
    def update_soul_maturation(self, time_units: float) -> Dict[str, List[str]]:
        """
        Update maturation of all stored souls over time.
        
        Args:
            time_units: Number of time units to simulate
            
        Returns:
            Dictionary with lists of souls reaching readiness or other thresholds
            
        Raises:
            ValueError: If time_units is invalid
        """
        if time_units <= 0:
            raise ValueError("Time units must be positive")
            
        # Track souls reaching thresholds
        ready_souls = []
        mature_souls = []
        high_wisdom_souls = []
        
        # Update each soul
        for soul in self.souls_stored:
            # Get current values
            current_maturity = soul['maturity']
            current_divine_light = soul['divine_light_exposure']
            current_wisdom = soul['wisdom_infusion']
            current_readiness = soul['incarnation_readiness']
            
            # Calculate updates
            maturity_increase = self.soul_maturation_rate * time_units
            divine_light_increase = self.divine_light_intensity * 0.01 * time_units
            wisdom_increase = self.wisdom_infusion_rate * time_units
            
            # Apply updates
            new_maturity = min(1.0, current_maturity + maturity_increase)
            new_divine_light = min(1.0, current_divine_light + divine_light_increase)
            new_wisdom = min(1.0, current_wisdom + wisdom_increase)
            
            # Calculate readiness
            new_readiness = 0.4 * new_maturity + 0.3 * new_divine_light + 0.3 * new_wisdom
            
            # Update soul record
            soul['maturity'] = new_maturity
            soul['divine_light_exposure'] = new_divine_light
            soul['wisdom_infusion'] = new_wisdom
            soul['incarnation_readiness'] = new_readiness
            
            # Check thresholds
            if new_readiness >= self.incarnation_readiness_threshold and current_readiness < self.incarnation_readiness_threshold:
                ready_souls.append(soul['id'])
                
            if new_maturity >= 0.9 and current_maturity < 0.9:
                mature_souls.append(soul['id'])
                
            if new_wisdom >= 0.8 and current_wisdom < 0.8:
                high_wisdom_souls.append(soul['id'])
        
        # Return thresholds reached
        thresholds = {
            'ready_for_incarnation': ready_souls,
            'high_maturity': mature_souls,
            'high_wisdom': high_wisdom_souls
        }
        
        logger.info(f"Updated soul maturation over {time_units} time units. Souls ready for incarnation: {len(ready_souls)}")
        return thresholds
    
    def inscribe_soul_name(self, soul_id: str, name: str) -> bool:
        """
        Inscribe a name to a soul in the Guff field.
        
        Args:
            soul_id: ID of soul to name
            name: Name to inscribe
            
        Returns:
            True if name was inscribed successfully
            
        Raises:
            ValueError: If soul is not found
            RuntimeError: If name inscription is disabled
        """
        if not self.name_inscription_enabled:
            raise RuntimeError("Name inscription is currently disabled in this Guff field")
            
        if not name:
            raise ValueError("Name cannot be empty")
            
        # Find soul
        soul = None
        for s in self.souls_stored:
            if s['id'] == soul_id:
                soul = s
                break
                
        if soul is None:
            raise ValueError(f"Soul {soul_id} not found in Guff field")
            
        # Update soul record
        soul['data']['name'] = name
        soul['data']['name_inscription_time'] = datetime.now().isoformat()
        
        logger.info(f"Name '{name}' inscribed to soul {soul_id}")
        return True
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Guff field's current state.
        
        Returns:
            Dictionary of Guff field metrics
        """
        # Get base metrics
        base_metrics = super().get_field_metrics()
        
        # Calculate soul statistics
        souls_count = len(self.souls_stored)
        capacity_utilization = souls_count / self.soul_capacity if self.soul_capacity > 0 else 0
        
        avg_maturity = 0.0
        avg_readiness = 0.0
        ready_souls_count = 0
        
        if souls_count > 0:
            avg_maturity = sum(s['maturity'] for s in self.souls_stored) / souls_count
            avg_readiness = sum(s['incarnation_readiness'] for s in self.souls_stored) / souls_count
            ready_souls_count = sum(1 for s in self.souls_stored if s['incarnation_readiness'] >= self.incarnation_readiness_threshold)
        
        # Add Guff-specific metrics
        guff_metrics = {
            'souls_count': souls_count,
            'capacity_utilization': capacity_utilization,
            'available_capacity': self.soul_capacity - souls_count,
            'average_soul_maturity': avg_maturity,
            'average_incarnation_readiness': avg_readiness,
            'souls_ready_for_incarnation': ready_souls_count,
            'divine_light_intensity': self.divine_light_intensity,
            'wisdom_infusion_rate': self.wisdom_infusion_rate,
            'kether_connection_strength': self.kether_connection_strength,
            'name_inscription_enabled': self.name_inscription_enabled,
            'composition': {
                'kether_influence_ratio': self.kether_influence_ratio,
                'guff_specific_ratio': self.guff_specific_ratio,
                'random_ratio': self.random_ratio,
                'composition_verified': self.composition_verified
            }
        }
        
        # Combine metrics
        combined_metrics = {
            **base_metrics,
            'guff_specific': guff_metrics
        }
        
        return combined_metrics
    
    def __str__(self) -> str:
        """String representation of the Guff field."""
        return f"GuffField(name={self.name}, souls={len(self.souls_stored)}/{self.soul_capacity})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<GuffField id='{self.field_id}' name='{self.name}' souls={len(self.souls_stored)}/{self.soul_capacity}>"

# --- END OF FILE guff_field.py ---


