# --- START OF FILE geburah_field.py ---

"""
Geburah Field Module

Defines the GeburahField class which represents the specific Sephiroth field for Geburah.
Implements Geburah-specific aspects, frequencies, colors, and other properties.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from src.stage_1.fields.sephiroth_field import SephirothField

# Configure logging
logger = logging.getLogger(__name__)

class GeburahField(SephirothField):
    """
    Represents the Geburah Sephiroth field - the embodiment of severity, judgment,
    and discipline in the soul development framework.
    """
    
    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Geburah",
                 dimensions: Tuple[float, float, float] = (70.0, 70.0, 70.0),
                 base_frequency: float = 741.0,  # Geburah frequency
                 resonance: float = 0.8,
                 stability: float = 0.85,
                 coherence: float = 0.82,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Geburah field with Geburah-specific properties.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Geburah-specific properties
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
            sephiroth_name="geburah"
        )
        
        # Set Geburah-specific attributes
        self.divine_attribute = "severity"
        self.geometric_correspondence = "pentagon"
        self.element = "fire"
        self.primary_color = "red"
        
        # Geburah-specific properties
        self.geburah_properties = {
            'judgment_level': 0.95,
            'discipline_energy': 0.9,
            'restriction_principle': 0.85,
            'mars_resonance': 0.88,
            'purification_factor': 0.92
        }
        
        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'geburah_properties': self.geburah_properties
        })
        
        # Initialize Geburah-specific aspects
        self._initialize_geburah_aspects()
        
        logger.info(f"Geburah Field '{self.name}' initialized with {len(self.aspects)} aspects")
    
    def _initialize_geburah_aspects(self) -> None:
        """
        Initialize Geburah-specific aspects.
        
        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Geburah-specific aspects
            aspects_data = {
                'divine_judgment': {
                    'frequency': 741.0,  # Geburah base frequency
                    'color': 'red',
                    'element': 'fire',
                    'keywords': ['judgment', 'discernment', 'evaluation', 'assessment'],
                    'description': 'The aspect of divine judgment and discernment'
                },
                'divine_severity': {
                    'frequency': 741.0,  # Geburah base frequency
                    'color': 'deep red',
                    'element': 'fire',
                    'keywords': ['severity', 'discipline', 'restriction', 'limitation'],
                    'description': 'The aspect of divine severity and discipline'
                },
                'purification': {
                    'frequency': 852.0,
                    'color': 'bright red',
                    'element': 'fire',
                    'keywords': ['purification', 'cleansing', 'removal', 'elimination'],
                    'description': 'The aspect of spiritual purification'
                },
                'discipline': {
                    'frequency': 639.0,
                    'color': 'red-orange',
                    'element': 'fire',
                    'keywords': ['discipline', 'training', 'refinement', 'structure'],
                    'description': 'The aspect of spiritual discipline and training'
                },
                'mars_energy': {
                    'frequency': 144.72,  # Mars frequency
                    'color': 'fiery red',
                    'element': 'fire',
                    'keywords': ['strength', 'courage', 'willpower', 'determination'],
                    'description': 'The planetary aspect of Mars energy'
                },
                'divine_sword': {
                    'frequency': 963.0,
                    'color': 'silver-red',
                    'element': 'fire/air',
                    'keywords': ['separation', 'cutting', 'discernment', 'decision'],
                    'description': 'The aspect of the divine sword that cuts away illusion'
                },
                'karmic_balance': {
                    'frequency': 528.0,
                    'color': 'ruby red',
                    'element': 'fire',
                    'keywords': ['karma', 'balance', 'justice', 'equilibrium'],
                    'description': 'The aspect of karmic balance and justice'
                }
            }
            
            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Geburah principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'judgment', 'severity', 'discipline', 'restriction', 'purification', 'strength'}
                )) / len(data['keywords'])
                
                # Frequency alignment with Geburah's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)
                
                # Element alignment
                element_alignment = 1.0 if data['element'] == self.element else 0.5
                
                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.5, strength))  # Ensure minimum strength of 0.5
                
                # Add aspect
                self.add_aspect(name, strength, data)
                
        except Exception as e:
            error_msg = f"Failed to initialize Geburah aspects: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Geburah-specific patterns to the energy grid.
        
        Raises:
            RuntimeError: If energy grid is not initialized
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Geburah patterns")
            
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
        
        # Geburah is associated with severity, judgment, and fire
        # Create patterns that reflect these properties
        
        # 1. Fire pattern (intense, dynamic energy)
        fire_pattern = 0.5 + 0.5 * np.sin(8 * np.pi * (
            0.4 * np.sin(5 * np.pi * x_norm) +
            0.4 * np.sin(7 * np.pi * y_norm) +
            0.2 * np.sin(3 * np.pi * z_norm)
        ))
        
        # 2. Judgment pattern (clear boundaries, stark contrasts)
        judgment_pattern = 0.5 + 0.5 * np.sign(np.sin(4 * np.pi * distance))
        
        # 3. Severity pattern (intensifying toward edges)
        severity_pattern = 0.3 + 0.7 * distance
        
        # 4. Mars influence (dynamic, forceful energy)
        mars_pattern = 0.5 + 0.5 * np.sin(144.72 * x_norm * 2 * np.pi) * \
                            np.sin(144.72 * y_norm * 2 * np.pi) * \
                            np.sin(144.72 * z_norm * 2 * np.pi)
                            
        # 5. Pentagon pattern (Geburah's geometric correspondence)
        # Approximate pentagon pattern using trigonometry
        pentagon_r = 0.5
        pentagon_points = 5
        angle = np.arctan2(y_norm - center_y, x_norm - center_x)
        pentagon_pattern = 0.5 + 0.5 * np.cos(pentagon_points * angle)
        
        # Combine patterns with appropriate weights
        self._sephiroth_contribution = (
            0.25 * fire_pattern +
            0.25 * judgment_pattern +
            0.2 * severity_pattern +
            0.15 * mars_pattern +
            0.15 * pentagon_pattern
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
        
        logger.debug(f"Applied Geburah-specific patterns to energy grid")
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Geburah field's energy grid with specific patterns.
        
        Args:
            resolution: Number of points along each dimension
            
        Returns:
            The initialized energy grid
            
        Raises:
            ValueError: If resolution is invalid
        """
        # Initialize base grid from SephirothField
        super().initialize_energy_grid(resolution)
        
        # Apply Geburah-specific patterns
        self.apply_sephiroth_specific_patterns()
        
        # Verify composition
        try:
            self.verify_composition()
        except RuntimeError as e:
            logger.warning(f"Composition verification failed: {e}")
            # Continue without failing - this is initialization
        
        return self._energy_grid
    
    def apply_judgment(self, entity_id: str, judgment_intensity: float) -> float:
        """
        Apply Geburah's judgment energy to an entity.
        
        Args:
            entity_id: ID of entity to receive judgment energy
            judgment_intensity: Intensity of judgment (0.0-1.0)
            
        Returns:
            Purification level achieved
            
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
            raise ValueError(f"Entity {entity_id} not found in Geburah field")
            
        if not (0.0 <= judgment_intensity <= 1.0):
            raise ValueError(f"Judgment intensity must be between 0.0 and 1.0, got {judgment_intensity}")
            
        # Calculate judgment effect
        base_judgment = self.geburah_properties['judgment_level']
        purification_factor = self.geburah_properties['purification_factor']
        
        # Calculate purification level
        purification_level = base_judgment * judgment_intensity * purification_factor
        
        # Record judgment application
        judgment_event = {
            'entity_id': entity_id,
            'judgment_intensity': judgment_intensity,
            'purification_level': purification_level,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to entity record
        for entity in self.entities:
            if entity['id'] == entity_id:
                if 'geburah_judgments' not in entity:
                    entity['geburah_judgments'] = []
                entity['geburah_judgments'].append(judgment_event)
                break
        
        logger.info(f"Applied Geburah judgment with intensity {judgment_intensity:.2f} to entity {entity_id}")
        return float(purification_level)
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Geburah field's current state.
        
        Returns:
            Dictionary of Geburah field metrics
        """
        # Get Sephiroth base metrics
        base_metrics = super().get_field_metrics()
        
        # Add Geburah-specific metrics
        geburah_metrics = {
            'geburah_properties': self.geburah_properties,
            'judgment_capacity': self.geburah_properties['judgment_level'] * self.stability,
            'mars_influence': self.geburah_properties['mars_resonance'] * self.coherence,
            'purification_potential': self.geburah_properties['purification_factor'] * self.resonance
        }
        
        # Calculate entity judgment metrics
        entity_judgment = {}
        for entity in self.entities:
            entity_id = entity['id']
            # Count judgment events for each entity
            judgment_count = len(entity.get('geburah_judgments', []))
            entity_judgment[entity_id] = {
                'judgment_events': judgment_count
            }
            # Calculate average purification if judgments exist
            if judgment_count > 0:
                avg_purification = sum(j['purification_level'] for j in entity['geburah_judgments']) / judgment_count
                entity_judgment[entity_id]['avg_purification'] = avg_purification
        
        # Combine metrics
        combined_metrics = {
            **base_metrics,
            'geburah_specific': geburah_metrics,
            'entity_judgments': entity_judgment
        }
        
        return combined_metrics
    
    def __str__(self) -> str:
        """String representation of the Geburah field."""
        return f"GeburahField(name={self.name}, aspects={len(self.aspects)}, judgment={self.geburah_properties['judgment_level']:.2f})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<GeburahField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} judgment={self.geburah_properties['judgment_level']:.2f}>"

# --- END OF FILE geburah_field.py ---