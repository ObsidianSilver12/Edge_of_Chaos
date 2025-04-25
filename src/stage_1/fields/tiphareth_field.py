# --- START OF FILE tiphareth_field.py ---

"""
Tiphareth Field Module

Defines the TipharethField class which represents the specific Sephiroth field for Tiphareth.
Implements Tiphareth-specific aspects, frequencies, colors, and other properties.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Assuming sephiroth_field is in the same directory or accessible via src path
from .sephiroth_field import SephirothField

# Configure logging
logger = logging.getLogger(__name__)

class TipharethField(SephirothField):
    """
    Represents the Tiphareth Sephiroth field - the embodiment of beauty, balance,
    harmony, and the heart center in the soul development framework.
    """

    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Tiphareth - Beauty", # More descriptive default name
                 dimensions: Tuple[float, float, float] = (85.0, 85.0, 85.0),
                 base_frequency: float = 639.0,  # Tiphareth frequency (Solfeggio FA - Connection/Relationships)
                 resonance: float = 0.92,
                 stability: float = 0.95, # Tiphareth is highly stable balance point
                 coherence: float = 0.94,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Tiphareth field with Tiphareth-specific properties.

        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Tiphareth-specific properties
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
            sephiroth_name="tiphareth"
        )

        # Set Tiphareth-specific attributes
        self.divine_attribute = "beauty"
        self.geometric_correspondence = "cube/hexagon" # Central cube or six-pointed star (Star of David)
        self.element = "air" # Element of balance, sometimes Sun/Fire associated
        self.primary_color = "yellow/gold" # Color of the Sun, balance

        # Tiphareth-specific properties
        self.tiphareth_properties = {
            'harmony_level': 0.98,
            'balance_factor': 0.97,
            'beauty_resonance': 0.95,
            'healing_potential': 0.90, # Connection to heart/healing
            'sun_influence': 0.92 # Planetary correspondence
        }

        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'tiphareth_properties': self.tiphareth_properties
        })

        # Initialize Tiphareth-specific aspects
        self._initialize_tiphareth_aspects()

        logger.info(f"Tiphareth Field '{self.name}' initialized with {len(self.aspects)} aspects")

    def _initialize_tiphareth_aspects(self) -> None:
        """
        Initialize Tiphareth-specific aspects.

        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Tiphareth-specific aspects
            aspects_data = {
                'divine_beauty': {
                    'frequency': 639.0,  # Tiphareth base frequency
                    'color': 'gold',
                    'element': 'air/fire',
                    'keywords': ['beauty', 'harmony', 'balance', 'aesthetics', 'perfection'],
                    'description': 'The aspect of divine beauty and perfect harmony'
                },
                'harmony_and_balance': {
                    'frequency': 432.0, # Harmonic frequency related to Earth/Sun
                    'color': 'yellow',
                    'element': 'air',
                    'keywords': ['harmony', 'balance', 'equilibrium', 'center', 'stillness'],
                    'description': 'The aspect of balance and equilibrium in the Tree'
                },
                'the_son': {
                    'frequency': 528.0, # Solfeggio MI (Love/Miracle) frequency
                    'color': 'golden yellow',
                    'element': 'fire/air',
                    'keywords': ['son', 'christ_consciousness', 'sacrifice', 'resurrection', 'redemption'],
                    'description': 'The aspect representing the Son/Redeemer principle'
                },
                'healing_heart': {
                    'frequency': 528.0, # Love frequency
                    'color': 'emerald green/gold', # Heart chakra colors
                    'element': 'air', # Heart often associated with Air
                    'keywords': ['healing', 'heart', 'compassion', 'love', 'empathy'],
                    'description': 'The aspect of the compassionate healing heart'
                },
                'integration_center': {
                    'frequency': 639.0, # Connection frequency
                    'color': 'yellow/white',
                    'element': 'air',
                    'keywords': ['integration', 'center', 'synthesis', 'connection', 'nexus'],
                    'description': 'The aspect of Tiphareth as the central integration point'
                },
                'sun_splendor': {
                    'frequency': 126.22, # Sun frequency
                    'color': 'brilliant gold',
                    'element': 'fire',
                    'keywords': ['sun', 'light', 'splendor', 'radiance', 'life', 'vitality'],
                    'description': 'The planetary aspect of the Sun'
                }
            }

            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Tiphareth principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'beauty', 'harmony', 'balance', 'center', 'healing', 'integration', 'sun', 'love', 'light'}
                )) / max(1, len(data['keywords']))

                # Frequency alignment with Tiphareth's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)

                # Element alignment (Air is primary for balance, Fire for Sun aspect)
                element_alignment = 1.0 if 'air' in data['element'] else (0.8 if 'fire' in data['element'] else 0.4)

                # Calculate overall strength - Tiphareth aspects are generally strong and balanced
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.7, strength)) # Ensure strong minimum strength for Tiphareth

                # Add aspect (will raise ValueError if name exists or strength invalid)
                self.add_aspect(name, strength, data)

        except (ValueError, TypeError) as e:
             # Catch errors from add_aspect
             error_msg = f"Failed to add Tiphareth aspect: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Tiphareth aspects: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Tiphareth-specific patterns to the energy grid. Reflects balance and harmony.

        Raises:
            RuntimeError: If energy grid or required contribution arrays are not initialized.
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Tiphareth patterns")
        if not hasattr(self, '_base_contribution') or self._base_contribution is None:
             raise RuntimeError("Base contribution array (_base_contribution) not initialized.")
        if not hasattr(self, '_random_contribution') or self._random_contribution is None:
             raise RuntimeError("Random contribution array (_random_contribution) not initialized.")

        grid_shape = self._energy_grid.shape
        if any(s <= 0 for s in grid_shape):
             raise RuntimeError(f"Energy grid has invalid shape: {grid_shape}")

        # Normalize coordinates [0, 1]
        indices = np.indices(grid_shape)
        norm_coords = [indices[i] / max(1, grid_shape[i] - 1) for i in range(3)]
        x_norm, y_norm, z_norm = norm_coords

        # Distance from center [0, 1 approx]
        center = np.array([0.5, 0.5, 0.5])
        point_coords = np.stack(norm_coords, axis=-1)
        distance = np.linalg.norm(point_coords - center, axis=-1)
        distance_norm = np.clip(distance / (np.sqrt(3) / 2), 0.0, 1.0)

        # Tiphareth: Beauty, Balance, Harmony, Sun, Center
        # Patterns should be centered, symmetrical, harmonious, and radiating.

        # 1. Central Radiance (Sun-like energy emanating from center)
        # Strong Gaussian core, represents the central sun/heart
        radiance_pattern = np.exp(-distance_norm**2 * 7) # Tighter core than Kether

        # 2. Harmony Pattern (Symmetrical, stable wave patterns - e.g., 6-fold)
        # Calculate angle in XY plane for symmetry reference
        angle_xy = np.arctan2(y_norm - 0.5, x_norm - 0.5)
        # Combine with Z influence for 3D symmetry
        angle_z = np.arccos(np.clip(z_norm - 0.5, -0.5, 0.5) * 2) # Angle from equator
        # 6-fold symmetry modulated by vertical position and distance
        harmony_pattern = 0.5 + 0.5 * np.cos(6 * angle_xy) * np.sin(angle_z) * np.exp(-distance_norm * 1.5)

        # 3. Balance Pattern (Smooth, even energy distribution, minimal extremes)
        # Low-frequency standing wave, centered
        balance_pattern = 0.7 + 0.3 * np.cos(2 * np.pi * distance_norm) * np.cos(np.pi * z_norm)

        # 4. Integration Pattern (Connects surrounding Sephiroth - radial pathways)
        # Radial structure modulated angularly
        num_spokes = 6 # Connecting to 6 surrounding Sephiroth
        radial_structure = 0.5 + 0.5 * np.cos(num_spokes * angle_xy) * np.exp(-distance_norm * 4)
        integration_pattern = radial_structure

        # 5. Sun Influence (Golden light, life-giving warmth)
        # Use frequency associated with the Sun, create warm radial pattern
        sun_freq_scaled = 126.22 / 10 # Scaled planet frequency for grid
        sun_pattern = 0.5 + 0.5 * np.sin(sun_freq_scaled * np.pi * distance_norm * 4)**2 # Positive radiating wave

        # Combine patterns reflecting Tiphareth's balanced, central nature
        self._sephiroth_contribution = (
            0.35 * radiance_pattern +      # Central Sun/Beauty/Heart (Strongest)
            0.20 * harmony_pattern +       # Symmetry/Harmony
            0.20 * balance_pattern +       # Equilibrium/Stability
            0.10 * integration_pattern +   # Central connection point
            0.15 * sun_pattern             # Planetary influence
        )

        # Ensure values are within the valid range [0, 1]
        self._sephiroth_contribution = np.clip(self._sephiroth_contribution, 0.0, 1.0)

        # Update the final energy grid
        self._energy_grid = (
            self.base_field_ratio * self._base_contribution +
            self.sephiroth_ratio * self._sephiroth_contribution +
            self.random_ratio * self._random_contribution
        )
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0) # Final clip
        logger.debug(f"Applied Tiphareth-specific patterns to energy grid")

    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Tiphareth field's energy grid with specific patterns.

        Args:
            resolution: Number of points along each dimension (must be positive ints).

        Returns:
            The initialized energy grid (numpy array).

        Raises:
            ValueError: If resolution is invalid.
            RuntimeError: If initialization or pattern application fails.
        """
        super().initialize_energy_grid(resolution)
        self.apply_sephiroth_specific_patterns()
        try:
            self.verify_composition()
        except RuntimeError as e:
            logger.warning(f"Composition verification failed during Tiphareth initialization: {e}")
        return self._energy_grid

    def harmonize_entity(self, entity_id: str, harmony_intensity: float = 0.8) -> float:
        """
        Apply Tiphareth's harmonizing energy to an entity.

        Args:
            entity_id: ID of entity to harmonize.
            harmony_intensity: Intensity of the harmonization effect (0.0-1.0).

        Returns:
            Resulting harmony level achieved in the entity (conceptual value 0.0-1.0).

        Raises:
            ValueError: If parameters are invalid or entity not found in field.
            TypeError: If argument types are incorrect.
            RuntimeError: If operation fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(entity_id, str) or not entity_id:
             raise TypeError("entity_id must be a non-empty string.")
        if not isinstance(harmony_intensity, (int, float)) or not (0.0 <= harmony_intensity <= 1.0):
            raise ValueError(f"Harmony intensity must be a number between 0.0 and 1.0, got {harmony_intensity}")

        entity_ref = None
        for entity in self.entities:
             if isinstance(entity, dict) and entity.get('id') == entity_id:
                  entity_ref = entity
                  break
        if entity_ref is None:
            raise ValueError(f"Entity '{entity_id}' not found in Tiphareth field {self.field_id}")

        try:
            # Calculate harmonization effect based on Tiphareth properties
            base_harmony = self.tiphareth_properties.get('harmony_level', 0.9) # Default if missing
            balance_factor = self.tiphareth_properties.get('balance_factor', 0.9)

            # Harmonization effectiveness influenced by field state (resonance, stability)
            field_effectiveness = self.resonance * self.stability

            # Simulate harmonization process - increase entity's internal harmony
            # Assumes entity has a 'harmony' attribute (state variable)
            current_entity_harmony = entity_ref.get('harmony', 0.5) # Default if missing
            if not isinstance(current_entity_harmony, (int, float)): current_entity_harmony = 0.5 # Reset if invalid type

            # Calculate the increase in harmony
            harmony_increase = base_harmony * balance_factor * harmony_intensity * field_effectiveness * 0.15 # Scaled effect
            new_harmony = min(1.0, current_entity_harmony + harmony_increase) # Clamp result

            # Record the event
            harmonization_event = {
                'entity_id': entity_id,
                'intensity': harmony_intensity,
                'harmony_increase': harmony_increase,
                'final_harmony': new_harmony,
                'timestamp': datetime.now().isoformat()
            }

            # Add event to entity's record if structure allows
            if 'tiphareth_harmonizations' not in entity_ref: entity_ref['tiphareth_harmonizations'] = []
            if isinstance(entity_ref['tiphareth_harmonizations'], list):
                 entity_ref['tiphareth_harmonizations'].append(harmonization_event)
            else: logger.warning(f"Entity {entity_id} 'tiphareth_harmonizations' is not list.")

            # Update the entity's state directly
            entity_ref['harmony'] = new_harmony
            # Could also update stability/coherence slightly
            entity_ref['stability'] = min(1.0, entity_ref.get('stability', 0.5) + harmony_increase * 0.1)
            entity_ref['coherence'] = min(1.0, entity_ref.get('coherence', 0.5) + harmony_increase * 0.1)


            logger.info(f"Harmonized entity {entity_id}. New harmony level: {new_harmony:.3f}")
            return float(new_harmony)

        except Exception as e:
             error_msg = f"Error harmonizing entity {entity_id}: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e


    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Tiphareth field's current state.

        Returns:
            Dictionary of Tiphareth field metrics.
        """
        base_metrics = super().get_field_metrics()

        tiphareth_metrics = {
            'tiphareth_properties': self.tiphareth_properties.copy(), # Return a copy
            'realized_field_harmony': self.tiphareth_properties.get('harmony_level', 0) * self.stability * self.coherence,
            'realized_balance_potential': self.tiphareth_properties.get('balance_factor', 0) * self.resonance,
            'realized_healing_potential': self.tiphareth_properties.get('healing_potential', 0) * self.coherence,
            'realized_sun_influence': self.tiphareth_properties.get('sun_influence', 0)
        }

        # Calculate average harmony of entities within the field
        total_entity_harmony = 0.0
        valid_entity_count = 0
        for entity in self.entities:
             harmony = entity.get('harmony', None) # Get harmony attribute
             if isinstance(harmony, (int, float)):
                 total_entity_harmony += harmony
                 valid_entity_count += 1

        tiphareth_metrics['average_entity_harmony'] = total_entity_harmony / valid_entity_count if valid_entity_count > 0 else 0.0
        tiphareth_metrics['harmonized_entity_count'] = valid_entity_count

        combined_metrics = {
            **base_metrics,
            'tiphareth_specific': tiphareth_metrics,
        }
        return combined_metrics

    def __str__(self) -> str:
        """String representation of the Tiphareth field."""
        harmony = self.tiphareth_properties.get('harmony_level', 0.0)
        return f"TipharethField(name={self.name}, aspects={len(self.aspects)}, harmony={harmony:.2f})"

    def __repr__(self) -> str:
        """Detailed representation."""
        balance = self.tiphareth_properties.get('balance_factor', 0.0)
        return f"<TipharethField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} balance={balance:.2f}>"

# --- END OF FILE tiphareth_field.py ---