# --- START OF FILE chesed_field.py ---

"""
Chesed Field Module

Defines the ChesedField class which represents the specific Sephiroth field for Chesed.
Implements Chesed-specific aspects, frequencies, colors, and other properties.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from src.stage_1.fields.sephiroth_field import SephirothField

# Configure logging
logger = logging.getLogger(__name__)

class ChesedField(SephirothField):
    """
    Represents the Chesed Sephiroth field - the embodiment of mercy, loving-kindness,
    and benevolence in the soul development framework.
    """
    
    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Chesed",
                 dimensions: Tuple[float, float, float] = (70.0, 70.0, 70.0),
                 base_frequency: float = 396.0,  # Chesed frequency
                 resonance: float = 0.85,
                 stability: float = 0.9,
                 coherence: float = 0.85,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Chesed field with Chesed-specific properties.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Chesed-specific properties
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
            sephiroth_name="chesed"
        )
        
        # Set Chesed-specific attributes
        self.divine_attribute = "mercy"
        self.geometric_correspondence = "tetrahedron"
        self.element = "water"
        self.primary_color = "blue"
        
        # Chesed-specific properties
        self.chesed_properties = {
            'compassion_level': 0.95,
            'healing_energy': 0.9,
            'expansion_principle': 0.85,
            'love_frequency': 528.0,  # Love frequency in Hz
            'jupiter_resonance': 0.8
        }
        
        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'chesed_properties': self.chesed_properties
        })
        
        # Initialize Chesed-specific aspects
        self._initialize_chesed_aspects()
        
        logger.info(f"Chesed Field '{self.name}' initialized with {len(self.aspects)} aspects")
    
    def _initialize_chesed_aspects(self) -> None:
        """
        Initialize Chesed-specific aspects.
        
        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Chesed-specific aspects
            aspects_data = {
                'loving_kindness': {
                    'frequency': 528.0,  # Love frequency
                    'color': 'blue',
                    'element': 'water',
                    'keywords': ['compassion', 'love', 'kindness', 'healing'],
                    'description': 'The aspect of unconditional love and compassion'
                },
                'mercy': {
                    'frequency': 396.0,  # Chesed base frequency
                    'color': 'blue',
                    'element': 'water',
                    'keywords': ['forgiveness', 'grace', 'clemency', 'leniency'],
                    'description': 'The aspect of divine mercy and forgiveness'
                },
                'magnanimity': {
                    'frequency': 417.0,
                    'color': 'royal blue',
                    'element': 'water',
                    'keywords': ['generosity', 'benevolence', 'abundance', 'giving'],
                    'description': 'The aspect of generosity and abundance'
                },
                'expansion': {
                    'frequency': 639.0,  # Connection frequency
                    'color': 'light blue',
                    'element': 'air/water',
                    'keywords': ['growth', 'expansion', 'abundance', 'prosperity'],
                    'description': 'The aspect of expansive growth and prosperity'
                },
                'healing': {
                    'frequency': 741.0,  # Healing frequency
                    'color': 'turquoise',
                    'element': 'water',
                    'keywords': ['restoration', 'recovery', 'wholeness', 'wellness'],
                    'description': 'The aspect of healing and restoration'
                },
                'benevolence': {
                    'frequency': 432.0,  # Wholeness frequency
                    'color': 'deep blue',
                    'element': 'water',
                    'keywords': ['goodwill', 'charity', 'benevolence', 'altruism'],
                    'description': 'The aspect of goodwill and altruism'
                },
                'jupiter': {
                    'frequency': 183.58,  # Jupiter frequency
                    'color': 'royal blue',
                    'element': 'aether',
                    'keywords': ['expansion', 'blessing', 'abundance', 'wisdom'],
                    'description': 'The planetary aspect of Jupiter'
                }
            }
            
            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Chesed principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'love', 'compassion', 'kindness', 'mercy', 'healing', 'abundance'}
                )) / len(data['keywords'])
                
                # Frequency alignment with Chesed's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)
                
                # Element alignment
                element_alignment = 1.0 if data['element'] == self.element else 0.5
                
                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.5, strength))  # Ensure minimum strength of 0.5
                
                # Add aspect
                self.add_aspect(name, strength, data)
                
        except Exception as e:
            error_msg = f"Failed to initialize Chesed aspects: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Chesed-specific patterns to the energy grid.
        
        Raises:
            RuntimeError: If energy grid is not initialized
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Chesed patterns")
            
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
        
        # Chesed is associated with expansion, love, and mercy
        # Create patterns that reflect these properties
        
        # 1. Expansive, outward-flowing energy (tetrahedron shape)
        tetrahedron = np.maximum(0, 1 - 2 * (
            abs(x_norm - 0.5) + abs(y_norm - 0.5) + abs(z_norm - 0.5)
        ))
        
        # 2. Love frequency pattern (528 Hz)
        love_pattern = 0.5 + 0.5 * np.sin(528 * distance * 2 * np.pi)
        
        # 3. Mercy/Compassion pattern (gentle, wave-like)
        mercy_pattern = 0.5 + 0.5 * np.cos(4 * np.pi * distance) * np.exp(-2 * distance)
        
        # 4. Jupiter influence (expansive, magnanimous)
        jupiter_pattern = 0.5 + 0.5 * np.sin(183.58 * x_norm * 2 * np.pi) * \
                               np.sin(183.58 * y_norm * 2 * np.pi) * \
                               np.sin(183.58 * z_norm * 2 * np.pi)
                               
        # 5. Water element pattern (flowing, fluid)
        water_pattern = 0.5 + 0.5 * np.sin(8 * np.pi * (
            0.5 * np.sin(3 * np.pi * x_norm) +
            0.3 * np.sin(5 * np.pi * y_norm) +
            0.2 * np.sin(7 * np.pi * z_norm)
        ))
        
        # Combine patterns with appropriate weights
        self._sephiroth_contribution = (
            0.25 * tetrahedron +
            0.25 * love_pattern +
            0.20 * mercy_pattern +
            0.15 * jupiter_pattern +
            0.15 * water_pattern
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
        
        logger.debug(f"Applied Chesed-specific patterns to energy grid")
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Chesed field's energy grid with specific patterns.
        
        Args:
            resolution: Number of points along each dimension
            
        Returns:
            The initialized energy grid
            
        Raises:
            ValueError: If resolution is invalid
        """
        # Initialize base grid from SephirothField
        super().initialize_energy_grid(resolution)
        
        # Apply Chesed-specific patterns
        self.apply_sephiroth_specific_patterns()
        
        # Verify composition
        try:
            self.verify_composition()
        except RuntimeError as e:
            logger.warning(f"Composition verification failed: {e}")
            # Continue without failing - as this is initialization, not runtime
        
        return self._energy_grid
    
    def calculate_healing_resonance(self, entity_frequency: float) -> float:
        """
        Calculate healing resonance with an entity - Chesed-specific function.
        
        Args:
            entity_frequency: Frequency of the entity
            
        Returns:
            Healing resonance value between 0.0 and 1.0
            
        Raises:
            ValueError: If entity_frequency is invalid
        """
        if entity_frequency <= 0:
            raise ValueError("Entity frequency must be positive")
            
        # Calculate based on love frequency (528 Hz) and mercy frequency (396 Hz)
        love_resonance = 1.0 - min(1.0, abs(528.0 - entity_frequency) / 528.0)
        mercy_resonance = 1.0 - min(1.0, abs(396.0 - entity_frequency) / 396.0)
        
        # Calculate healing resonance
        healing_freq = 741.0  # Healing frequency
        healing_resonance = 1.0 - min(1.0, abs(healing_freq - entity_frequency) / healing_freq)
        
        # Combine resonances with Chesed-specific weights
        chesed_healing_resonance = (
            0.4 * love_resonance +
            0.3 * mercy_resonance +
            0.3 * healing_resonance
        )
        
        # Apply Chesed's compassion modifier
        chesed_healing_resonance *= self.chesed_properties['compassion_level']
        
        return float(min(1.0, max(0.0, chesed_healing_resonance)))
    
    def apply_expansion_energy(self, entity_id: str, expansion_factor: float) -> float:
        """
        Apply Chesed's expansion energy to an entity.
        
        Args:
            entity_id: ID of entity to receive expansion energy
            expansion_factor: Factor to scale expansion energy (0.0-2.0)
            
        Returns:
            Actual expansion energy applied
            
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
            raise ValueError(f"Entity {entity_id} not found in Chesed field")
            
        if not (0.0 <= expansion_factor <= 2.0):
            raise ValueError(f"Expansion factor must be between 0.0 and 2.0, got {expansion_factor}")
            
        # Calculate expansion energy based on Chesed properties
        base_expansion = self.chesed_properties['expansion_principle']
        jupiter_influence = self.chesed_properties['jupiter_resonance']
        
        # Calculate final expansion energy
        expansion_energy = base_expansion * expansion_factor * jupiter_influence
        
        # Record expansion energy application
        expansion_event = {
            'entity_id': entity_id,
            'expansion_factor': expansion_factor,
            'expansion_energy': expansion_energy,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to entity record
        for entity in self.entities:
            if entity['id'] == entity_id:
                if 'chesed_expansions' not in entity:
                    entity['chesed_expansions'] = []
                entity['chesed_expansions'].append(expansion_event)
                break
        
        logger.info(f"Applied Chesed expansion energy {expansion_energy:.2f} to entity {entity_id}")
        return float(expansion_energy)
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Chesed field's current state.
        
        Returns:
            Dictionary of Chesed field metrics
        """
        # Get Sephiroth base metrics
        base_metrics = super().get_field_metrics()
        
        # Add Chesed-specific metrics
        chesed_metrics = {
            'chesed_properties': self.chesed_properties,
            'healing_capacity': self.chesed_properties['healing_energy'] * self.stability,
            'love_frequency_alignment': self.calculate_field_resonance(528.0),  # Alignment with love frequency
            'jupiter_influence': self.chesed_properties['jupiter_resonance'] * self.coherence,
            'mercy_quotient': self.chesed_properties['compassion_level'] * self.resonance
        }
        
        # Calculate entity healing metrics
        entity_healing = {}
        for entity in self.entities:
            entity_id = entity['id']
            # Calculate healing potential for each entity
            healing_potential = self.calculate_healing_resonance(432.0)  # Default frequency if unknown
            entity_healing[entity_id] = healing_potential
        
        # Combine metrics
        combined_metrics = {
            **base_metrics,
            'chesed_specific': chesed_metrics,
            'entity_healing_potentials': entity_healing
        }
        
        return combined_metrics
    
    def __str__(self) -> str:
        """String representation of the Chesed field."""
        return f"ChesedField(name={self.name}, aspects={len(self.aspects)}, compassion={self.chesed_properties['compassion_level']:.2f})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<ChesedField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} compassion={self.chesed_properties['compassion_level']:.2f}>"

# --- END OF FILE chesed_field.py ---