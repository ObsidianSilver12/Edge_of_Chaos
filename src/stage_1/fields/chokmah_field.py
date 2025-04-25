# --- START OF FILE chokmah_field.py ---

"""
Chokmah Field Module

Defines the ChokmahField class which represents the specific Sephiroth field for Chokmah.
Implements Chokmah-specific aspects, frequencies, colors, and other properties.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from src.stage_1.fields.sephiroth_field import SephirothField

# Configure logging
logger = logging.getLogger(__name__)

class ChokmahField(SephirothField):
    """
    Represents the Chokmah Sephiroth field - the embodiment of wisdom, intuition,
    and the masculine principle in the soul development framework.
    """
    
    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Chokmah",
                 dimensions: Tuple[float, float, float] = (90.0, 90.0, 90.0),
                 base_frequency: float = 932.0,  # Chokmah frequency
                 resonance: float = 0.95,
                 stability: float = 0.92,
                 coherence: float = 0.9,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Chokmah field with Chokmah-specific properties.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Chokmah-specific properties
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
            sephiroth_name="chokmah"
        )
        
        # Set Chokmah-specific attributes
        self.divine_attribute = "wisdom"
        self.geometric_correspondence = "line"
        self.element = "fire"
        self.primary_color = "grey"
        
        # Chokmah-specific properties
        self.chokmah_properties = {
            'wisdom_level': 0.98,
            'intuition_factor': 0.95,
            'masculine_principle': 0.9,
            'dynamic_force': 0.92,
            'inspiration_energy': 0.88
        }
        
        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'chokmah_properties': self.chokmah_properties
        })
        
        # Initialize Chokmah-specific aspects
        self._initialize_chokmah_aspects()
        
        logger.info(f"Chokmah Field '{self.name}' initialized with {len(self.aspects)} aspects")
    
    def _initialize_chokmah_aspects(self) -> None:
        """
        Initialize Chokmah-specific aspects.
        
        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Chokmah-specific aspects
            aspects_data = {
                'divine_wisdom': {
                    'frequency': 932.0,  # Chokmah base frequency
                    'color': 'silver-grey',
                    'element': 'fire',
                    'keywords': ['wisdom', 'insight', 'intuition', 'foresight'],
                    'description': 'The aspect of divine wisdom and insight'
                },
                'masculine_principle': {
                    'frequency': 852.0,
                    'color': 'blue-grey',
                    'element': 'fire',
                    'keywords': ['masculine', 'active', 'dynamic', 'yang'],
                    'description': 'The aspect of the divine masculine principle'
                },
                'creative_force': {
                    'frequency': 963.0,
                    'color': 'white-grey',
                    'element': 'fire',
                    'keywords': ['creation', 'force', 'energy', 'dynamism'],
                    'description': 'The aspect of dynamic creative force'
                },
                'divine_father': {
                    'frequency': 741.0,
                    'color': 'deep grey',
                    'element': 'fire',
                    'keywords': ['father', 'paternal', 'protective', 'guiding'],
                    'description': 'The aspect of the divine father principle'
                },
                'pure_intelligence': {
                    'frequency': 528.0,
                    'color': 'light grey',
                    'element': 'fire/air',
                    'keywords': ['intelligence', 'knowing', 'perception', 'cognition'],
                    'description': 'The aspect of pure divine intelligence'
                },
                'supernal_will': {
                    'frequency': 417.0,
                    'color': 'silver',
                    'element': 'fire',
                    'keywords': ['will', 'intention', 'purpose', 'direction'],
                    'description': 'The aspect of divine will and intention'
                },
                'stars_wisdom': {
                    'frequency': 126.22,  # Cosmic frequency
                    'color': 'starlight',
                    'element': 'light',
                    'keywords': ['stars', 'cosmos', 'celestial', 'infinite'],
                    'description': 'The aspect of celestial wisdom of the stars'
                }
            }
            
            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Chokmah principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'wisdom', 'masculine', 'force', 'intelligence', 'will', 'insight'}
                )) / len(data['keywords'])
                
                # Frequency alignment with Chokmah's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)
                
                # Element alignment
                element_alignment = 1.0 if data['element'] == self.element else 0.5
                
                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.5, strength))  # Ensure minimum strength of 0.5
                
                # Add aspect
                self.add_aspect(name, strength, data)
                
        except Exception as e:
            error_msg = f"Failed to initialize Chokmah aspects: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Chokmah-specific patterns to the energy grid.
        
        Raises:
            RuntimeError: If energy grid is not initialized
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Chokmah patterns")
            
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
        
        # Chokmah is associated with wisdom, masculine energy, and dynamic force
        # Create patterns that reflect these properties
        
        # 1. Line pattern (Chokmah's geometric correspondence is a line)
        line_pattern = np.maximum(0, 1.0 - 3 * abs(y_norm - 0.5)) * np.maximum(0, 1.0 - 3 * abs(z_norm - 0.5))
        
        # 2. Wisdom pattern (radial waves emanating from center)
        wisdom_pattern = 0.5 + 0.5 * np.cos(distance * 8 * np.pi)
        
        # 3. Masculine/Active pattern (dynamic, outward-directed energy)
        masculine_pattern = 0.5 + 0.5 * np.sin(8 * np.pi * (
            0.6 * np.sin(3 * np.pi * x_norm) +
            0.4 * np.sin(5 * np.pi * (y_norm + z_norm))
        ))
        
        # 4. Force pattern (concentrated streams of energy)
        force_pattern = 0.5 + 0.5 * np.sin(6 * np.pi * x_norm) * np.sin(6 * np.pi * y_norm) * np.sin(6 * np.pi * z_norm)
        
        # 5. Cosmic pattern (star-like points of light)
        cosmic_pattern = np.random.RandomState(42).rand(*grid_shape)  # Seeded for reproducibility
        cosmic_pattern = np.exp(-5 * distance) + 0.3 * (cosmic_pattern > 0.97).astype(float)
        
        # Combine patterns with appropriate weights
        self._sephiroth_contribution = (
            0.2 * line_pattern +
            0.25 * wisdom_pattern +
            0.25 * masculine_pattern +
            0.2 * force_pattern +
            0.1 * cosmic_pattern
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
        
        logger.debug(f"Applied Chokmah-specific patterns to energy grid")
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Chokmah field's energy grid with specific patterns.
        
        Args:
            resolution: Number of points along each dimension
            
        Returns:
            The initialized energy grid
            
        Raises:
            ValueError: If resolution is invalid
        """
        # Initialize base grid from SephirothField
        super().initialize_energy_grid(resolution)
        
        # Apply Chokmah-specific patterns
        self.apply_sephiroth_specific_patterns()
        
        # Verify composition
        try:
            self.verify_composition()
        except RuntimeError as e:
            logger.warning(f"Composition verification failed: {e}")
            # Continue without failing - this is initialization
        
        return self._energy_grid
    
    def impart_wisdom(self, entity_id: str, wisdom_intensity: float) -> float:
        """
        Impart Chokmah's wisdom to an entity.
        
        Args:
            entity_id: ID of entity to receive wisdom energy
            wisdom_intensity: Intensity of wisdom transfer (0.0-1.0)
            
        Returns:
            Actual wisdom imparted
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If entity not found or operation fails
        """
        # Find entity
        entity_found = False
        for entity in self.entities:
            if entity['id'] == entity_id:
                entity_found = True
                break
                
        if not entity_found:
            raise ValueError(f"Entity {entity_id} not found in Chokmah field")
            
        if not (0.0 <= wisdom_intensity <= 1.0):
            raise ValueError(f"Wisdom intensity must be between 0.0 and 1.0, got {wisdom_intensity}")
            
        # Calculate wisdom transfer
        base_wisdom = self.chokmah_properties['wisdom_level']
        intuition_factor = self.chokmah_properties['intuition_factor']
        
        # Calculate wisdom imparted
        wisdom_imparted = base_wisdom * wisdom_intensity * intuition_factor
        
        # Record wisdom impartation
        wisdom_event = {
            'entity_id': entity_id,
            'wisdom_intensity': wisdom_intensity,
            'wisdom_imparted': wisdom_imparted,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to entity record
        for entity in self.entities:
            if entity['id'] == entity_id:
                if 'chokmah_wisdom' not in entity:
                    entity['chokmah_wisdom'] = []
                entity['chokmah_wisdom'].append(wisdom_event)
                break
        
        logger.info(f"Imparted Chokmah wisdom with intensity {wisdom_intensity:.2f} to entity {entity_id}")
        return float(wisdom_imparted)
    
    def activate_creative_force(self, entity_id: str, activation_level: float) -> Dict[str, Any]:
        """
        Activate the creative force of Chokmah in an entity.
        
        Args:
            entity_id: ID of entity to activate
            activation_level: Level of creative force activation (0.0-1.0)
            
        Returns:
            Dictionary of activation results
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If entity not found or operation fails
        """
        # Find entity
        entity_found = False
        for entity in self.entities:
            if entity['id'] == entity_id:
                entity_found = True
                break
                
        if not entity_found:
            raise ValueError(f"Entity {entity_id} not found in Chokmah field")
            
        if not (0.0 <= activation_level <= 1.0):
            raise ValueError(f"Activation level must be between 0.0 and 1.0, got {activation_level}")
            
        # Calculate creative force activation
        dynamic_force = self.chokmah_properties['dynamic_force']
        inspiration = self.chokmah_properties['inspiration_energy']
        
        # Calculate activation effects
        force_effect = dynamic_force * activation_level
        inspiration_effect = inspiration * activation_level
        
        # Calculate combined effect
        total_effect = (0.6 * force_effect + 0.4 * inspiration_effect)
        
        # Record activation
        activation_event = {
            'entity_id': entity_id,
            'activation_level': activation_level,
            'force_effect': force_effect,
            'inspiration_effect': inspiration_effect,
            'total_effect': total_effect,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to entity record
        for entity in self.entities:
            if entity['id'] == entity_id:
                if 'chokmah_activations' not in entity:
                    entity['chokmah_activations'] = []
                entity['chokmah_activations'].append(activation_event)
                break
        
        logger.info(f"Activated Chokmah creative force at level {activation_level:.2f} for entity {entity_id}")
        return activation_event
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Chokmah field's current state.
        
        Returns:
            Dictionary of Chokmah field metrics
        """
        # Get Sephiroth base metrics
        base_metrics = super().get_field_metrics()
        
        # Add Chokmah-specific metrics
        chokmah_metrics = {
            'chokmah_properties': self.chokmah_properties,
            'wisdom_capacity': self.chokmah_properties['wisdom_level'] * self.stability,
            'intuition_power': self.chokmah_properties['intuition_factor'] * self.resonance,
            'masculine_energy': self.chokmah_properties['masculine_principle'] * self.coherence,
            'creative_potential': self.chokmah_properties['dynamic_force'] * self.chokmah_properties['inspiration_energy']
        }
        
        # Calculate entity wisdom metrics
        entity_wisdom = {}
        for entity in self.entities:
            entity_id = entity['id']
            # Count wisdom events for each entity
            wisdom_count = len(entity.get('chokmah_wisdom', []))
            entity_wisdom[entity_id] = {
                'wisdom_events': wisdom_count
            }
            # Calculate total wisdom imparted if events exist
            if wisdom_count > 0:
                total_wisdom = sum(w['wisdom_imparted'] for w in entity['chokmah_wisdom'])
                entity_wisdom[entity_id]['total_wisdom_imparted'] = total_wisdom
        
        # Combine metrics
        combined_metrics = {
            **base_metrics,
            'chokmah_specific': chokmah_metrics,
            'entity_wisdom': entity_wisdom
        }
        
        return combined_metrics
    
    def __str__(self) -> str:
        """String representation of the Chokmah field."""
        return f"ChokmahField(name={self.name}, aspects={len(self.aspects)}, wisdom={self.chokmah_properties['wisdom_level']:.2f})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<ChokmahField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} wisdom={self.chokmah_properties['wisdom_level']:.2f}>"

# --- END OF FILE chokmah_field.py ---