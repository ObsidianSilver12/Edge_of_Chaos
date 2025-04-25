# --- START OF FILE kether_field.py ---

"""
Kether Field Module

Defines the KetherField class which represents the highest Sephiroth in the Tree of Life.
Kether (Crown) is the first emanation, representing divine unity and the source of all creation.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from stage_1.fields.sephiroth_field import SephirothField

# Configure logging
logger = logging.getLogger(__name__)

class KetherField(SephirothField):
    """
    Represents the Kether Sephiroth field - the crown and highest emanation
    in the Tree of Life, embodying divine unity and the source of creation.
    """
    
    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Kether - Crown",
                 dimensions: Tuple[float, float, float] = (100.0, 100.0, 100.0),
                 base_frequency: float = 963.0,  # Kether frequency
                 resonance: float = 0.99,
                 stability: float = 0.99,
                 coherence: float = 0.99,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Kether field with Kether-specific properties.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Kether-specific properties
            random_ratio: Proportion of random variation
            
        Raises:
            ValueError: If any parameters are invalid
            TypeError: If parameters are of incorrect type
        """
        # Initialize the base Sephiroth field
        super().__init__(
            field_id=field_id,
            name=name,
            dimensions=dimensions,
            base_frequency=base_frequency,
            resonance=resonance,
            stability=stability,
            coherence=coherence,
            base_field_ratio=base_field_ratio,
            sephiroth_ratio=sephiroth_ratio,
            random_ratio=random_ratio,
            sephiroth_name="kether"
        )
        
        # Set Kether-specific attributes
        self.divine_attribute = "unity"
        self.geometric_correspondence = "point"
        self.element = "aether/light"
        self.primary_color = "white"
        
        # Kether-specific properties
        self.kether_properties = {
            'divine_unity': 0.99,
            'creation_power': 0.98,
            'transcendence_level': 0.99,
            'pure_consciousness': 0.97,
            'divine_potentiality': 0.98,
            'crown_energy': 0.99
        }
        
        # Unique to Kether: contained subdimensions like Guff
        self.contained_fields = {}  # Dictionary of fields contained within Kether
        
        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'kether_properties': self.kether_properties
        })
        
        # Initialize Kether-specific aspects
        self._initialize_kether_aspects()
        
        logger.info(f"Kether Field '{self.name}' initialized with {len(self.aspects)} aspects")
    
    def add_contained_field(self, field_id: str, field_type: str, position: Tuple[float, float, float], 
                          dimensions: Tuple[float, float, float]) -> bool:
        """
        Add a field contained within Kether (like Guff).
        
        Args:
            field_id: Unique identifier for the contained field
            field_type: Type of the field (e.g., "guff")
            position: (x, y, z) position of the field's center in Kether
            dimensions: (x, y, z) dimensions of the contained field
            
        Returns:
            True if field was added successfully
            
        Raises:
            ValueError: If position/dimensions are invalid or field already exists
        """
        # Validate position is within Kether boundaries
        for i, (coord, (min_bound, max_bound)) in enumerate(zip(position, self.boundaries)):
            if coord < min_bound or coord > max_bound:
                raise ValueError(f"Field position {position} is outside Kether boundaries")
        
        # Validate dimensions to ensure the field fits inside Kether
        for i in range(3):
            if dimensions[i] <= 0:
                raise ValueError(f"Field dimension {i} must be positive")
            
            min_pos = position[i] - dimensions[i]/2
            max_pos = position[i] + dimensions[i]/2
            
            if min_pos < self.boundaries[i][0] or max_pos > self.boundaries[i][1]:
                raise ValueError(f"Field extends beyond Kether boundaries at dimension {i}")
        
        # Check if field already exists
        if field_id in self.contained_fields:
            raise ValueError(f"Field {field_id} already exists in Kether")
            
        # Add the field
        self.contained_fields[field_id] = {
            'field_type': field_type,
            'position': position,
            'dimensions': dimensions,
            'addition_time': datetime.now().isoformat()
        }
        
        logger.info(f"Added {field_type} field {field_id} to Kether at position {position} with dimensions {dimensions}")
        return True
    
    def remove_contained_field(self, field_id: str) -> bool:
        """
        Remove a field from Kether.
        
        Args:
            field_id: ID of field to remove
            
        Returns:
            True if field was removed
            
        Raises:
            ValueError: If field does not exist in Kether
        """
        if field_id not in self.contained_fields:
            raise ValueError(f"Field {field_id} not found in Kether")
            
        # Remove the field
        field_info = self.contained_fields.pop(field_id)
        
        logger.info(f"Removed {field_info['field_type']} field {field_id} from Kether")
        return True
    
    def _initialize_kether_aspects(self) -> None:
        """
        Initialize Kether-specific aspects.
        
        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Kether-specific aspects
            aspects_data = {
                'divine_unity': {
                    'frequency': 963.0,  # Kether base frequency
                    'color': 'white',
                    'element': 'aether',
                    'keywords': ['oneness', 'unity', 'wholeness', 'singularity'],
                    'description': 'The aspect of divine unity and oneness'
                },
                'pure_being': {
                    'frequency': 1000.0,  # High frequency for pure being
                    'color': 'white/clear',
                    'element': 'light',
                    'keywords': ['existence', 'being', 'presence', 'is-ness'],
                    'description': 'The aspect of pure being and existence'
                },
                'divine_will': {
                    'frequency': 986.0,
                    'color': 'white',
                    'element': 'aether',
                    'keywords': ['will', 'intention', 'purpose', 'direction'],
                    'description': 'The aspect of divine will and intention'
                },
                'divine_wisdom': {
                    'frequency': 936.0,
                    'color': 'silvery white',
                    'element': 'light',
                    'keywords': ['wisdom', 'knowledge', 'understanding', 'omniscience'],
                    'description': 'The aspect of divine wisdom and omniscience'
                },
                'creation_source': {
                    'frequency': 852.0,  # Creation frequency
                    'color': 'radiant white',
                    'element': 'aether/light',
                    'keywords': ['creation', 'source', 'beginning', 'potential'],
                    'description': 'The aspect of source and beginning of creation'
                },
                'transcendence': {
                    'frequency': 999.0,  # High frequency for transcendence
                    'color': 'pure light',
                    'element': 'aether',
                    'keywords': ['beyond', 'transcendence', 'limitless', 'infinite'],
                    'description': 'The aspect of transcendence beyond limitation'
                },
                'crown_consciousness': {
                    'frequency': 963.0,  # Crown frequency
                    'color': 'brilliant white',
                    'element': 'light',
                    'keywords': ['consciousness', 'awareness', 'presence', 'being'],
                    'description': 'The aspect of pure consciousness and awareness'
                }
            }
            
            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Kether principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'unity', 'oneness', 'transcendence', 'being', 'consciousness', 'source'}
                )) / len(data['keywords'])
                
                # Frequency alignment with Kether's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)
                
                # Element alignment
                element_alignment = 1.0 if data['element'] == self.element else 0.7
                
                # Calculate overall strength - Kether aspects have high strength
                strength = (0.4 * keywords_alignment + 0.3 * freq_alignment + 0.3 * element_alignment)
                strength = min(1.0, max(0.8, strength))  # Ensure minimum strength of 0.8 for Kether
                
                # Add aspect
                self.add_aspect(name, strength, data)
                
        except Exception as e:
            error_msg = f"Failed to initialize Kether aspects: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Kether-specific patterns to the energy grid.
        
        Raises:
            RuntimeError: If energy grid is not initialized
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Kether patterns")
            
        # Get grid dimensions
        grid_shape = self._energy_grid.shape
        
        # Create grid indices
        x, y, z = np.indices(grid_shape)
        x_norm = x / grid_shape[0]
        y_norm = y / grid_shape[1]
        z_norm = z / grid_shape[2]
        
        # Calculate distance from center
        center_x, center_y, center_z = 0.5, 0.5, 0.5
        distance = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2 + (z_norm - center_z)**2)
        distance = distance / np.sqrt(3)  # Normalize to [0, 1]
        
        # Kether is associated with unity, divine light, and the crown
        # Create patterns that reflect these properties
        
        # 1. Unity pattern - singular point expanding outward
        unity_pattern = np.exp(-5 * distance)
        
        # 2. Divine light - pure white light emanating from center
        light_pattern = np.exp(-2 * distance) * (1.0 + 0.5 * z_norm)
        
        # 3. Crown energy - focused at top (high z)
        crown_pattern = 0.5 + 0.5 * z_norm * np.exp(-3 * distance)
        
        # 4. Creation source - pulsing energy
        creation_pattern = 0.5 + 0.5 * np.sin(10 * np.pi * distance) * np.exp(-2 * distance)
        
        # 5. Transcendence - subtle high-frequency oscillations
        transcendence_pattern = 0.5 + 0.1 * np.sin(20 * np.pi * (
            0.3 * np.sin(5 * np.pi * x_norm) +
            0.3 * np.sin(5 * np.pi * y_norm) +
            0.4 * np.sin(5 * np.pi * z_norm)
        ))
        
        # Combine patterns with appropriate weights
        self._sephiroth_contribution = (
            0.3 * unity_pattern +
            0.3 * light_pattern +
            0.2 * crown_pattern +
            0.1 * creation_pattern +
            0.1 * transcendence_pattern
        )
        
        # Ensure values are in valid range
        self._sephiroth_contribution = np.clip(self._sephiroth_contribution, 0.0, 1.0)
        
        # Update energy grid with new composition
        self._energy_grid = (
            self.base_field_ratio * self._base_contribution +
            self.sephiroth_ratio * self._sephiroth_contribution +
            self.random_ratio * self._random_contribution
        )
        
        # Final clipping
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0)
        
        logger.debug(f"Applied Kether-specific patterns to energy grid")
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Kether field's energy grid with specific patterns.
        
        Args:
            resolution: Number of points along each dimension
            
        Returns:
            The initialized energy grid
            
        Raises:
            ValueError: If resolution is invalid
        """
        # Initialize base grid from SephirothField
        super().initialize_energy_grid(resolution)
        
        # Apply Kether-specific patterns
        self.apply_sephiroth_specific_patterns()
        
        # Verify composition
        try:
            self.verify_composition()
        except RuntimeError as e:
            logger.warning(f"Composition verification failed: {e}")
            # Continue without failing - this is initialization
        
        return self._energy_grid
    
    def emanate_divine_light(self, intensity: float) -> float:
        """
        Emanate divine light - Kether-specific function for influencing other fields.
        
        Args:
            intensity: Intensity of emanation (0.0-1.0)
            
        Returns:
            Actual emanation strength
            
        Raises:
            ValueError: If intensity is invalid
        """
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"Intensity must be between 0.0 and 1.0, got {intensity}")
            
        # Calculate emanation strength
        base_emanation = self.kether_properties['divine_unity'] * self.kether_properties['crown_energy']
        emanation_strength = base_emanation * intensity
        
        # Record emanation event
        emanation_event = {
            'intensity': intensity,
            'emanation_strength': emanation_strength,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update properties
        if not hasattr(self, 'emanation_history'):
            self.emanation_history = []
        self.emanation_history.append(emanation_event)
        
        logger.info(f"Emanated divine light with strength {emanation_strength:.4f} (intensity: {intensity:.2f})")
        return float(emanation_strength)
    
    def increase_potentiality(self, target_field_id: str, potentiality_level: float) -> bool:
        """
        Increase the potentiality of a target field - Kether-specific ability.
        
        Args:
            target_field_id: ID of field to affect
            potentiality_level: Level of potentiality to infuse (0.0-1.0)
            
        Returns:
            True if potentiality was increased
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If operation fails
        """
        if not 0.0 <= potentiality_level <= 1.0:
            raise ValueError(f"Potentiality level must be between 0.0 and 1.0, got {potentiality_level}")
            
        # Check if field is connected
        if target_field_id not in self.connections:
            raise ValueError(f"Field {target_field_id} is not connected to Kether")
            
        # Calculate potentiality effect strength
        potentiality_strength = self.kether_properties['divine_potentiality'] * potentiality_level
        
        # Record potentiality event
        potentiality_event = {
            'target_field_id': target_field_id,
            'potentiality_level': potentiality_level,
            'potentiality_strength': potentiality_strength,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store event
        if not hasattr(self, 'potentiality_increases'):
            self.potentiality_increases = []
        self.potentiality_increases.append(potentiality_event)
        
        logger.info(f"Increased potentiality of field {target_field_id} to {potentiality_strength:.4f}")
        return True
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Kether field's current state.
        
        Returns:
            Dictionary of Kether field metrics
        """
        # Get Sephiroth base metrics
        base_metrics = super().get_field_metrics()
        
        # Add Kether-specific metrics
        kether_metrics = {
            'kether_properties': self.kether_properties,
            'divine_unity': self.kether_properties['divine_unity'] * self.stability,
            'creation_potential': self.kether_properties['creation_power'] * self.resonance,
            'transcendence_level': self.kether_properties['transcendence_level'],
            'contained_field_count': len(self.contained_fields),
            'emanation_count': len(getattr(self, 'emanation_history', [])),
            'potentiality_increase_count': len(getattr(self, 'potentiality_increases', []))
        }
        
        # Add subdimension information
        contained_fields_info = {}
        for field_id, field_data in self.contained_fields.items():
            contained_fields_info[field_id] = {
                'field_type': field_data['field_type'],
                'position': field_data['position'],
                'dimensions': field_data['dimensions'],
                'addition_time': field_data['addition_time']
            }
        
        # Combine metrics
        combined_metrics = {
            **base_metrics,
            'kether_specific': kether_metrics,
            'contained_fields': contained_fields_info
        }
        
        return combined_metrics
    
    def __str__(self) -> str:
        """String representation of the Kether field."""
        return f"KetherField(name={self.name}, aspects={len(self.aspects)}, unity={self.kether_properties['divine_unity']:.2f})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<KetherField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} contained_fields={len(self.contained_fields)}>"

# --- END OF FILE kether_field.py ---