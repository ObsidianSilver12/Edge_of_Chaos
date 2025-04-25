# --- START OF FILE malkuth_field.py ---

"""
Malkuth Field Module

Defines the MalkuthField class which represents the specific Sephiroth field for Malkuth.
Implements Malkuth-specific aspects, frequencies, colors, and other properties.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Assuming sephiroth_field is in the same directory or accessible via src path
from stage_1.fields.sephiroth_field import SephirothField

# Configure logging
logger = logging.getLogger(__name__)

class MalkuthField(SephirothField):
    """
    Represents the Malkuth Sephiroth field - the embodiment of the Kingdom, the physical realm,
    manifestation, grounding, and the synthesis of elements in the soul development framework.
    """

    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Malkuth - Kingdom", # More descriptive default name
                 dimensions: Tuple[float, float, float] = (65.0, 65.0, 65.0), # Smallest dimension
                 base_frequency: float = 285.0, # Malkuth frequency (Solfeggio related, grounding/physical)
                 resonance: float = 0.75,
                 stability: float = 0.98, # Very stable, represents the physical realm
                 coherence: float = 0.70, # Lower coherence due to complexity of physical manifestation
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Malkuth field with Malkuth-specific properties.

        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Malkuth-specific properties
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
            sephiroth_name="malkuth"
        )

        # Set Malkuth-specific attributes
        self.divine_attribute = "kingdom"
        self.geometric_correspondence = "cube/sphere" # Represents physical form/Earth
        self.element = "earth" # The element of manifestation
        self.primary_color = "citrine/olive/russet/black" # Earthy, mixed colors

        # Malkuth-specific properties
        self.malkuth_properties = {
            'manifestation_potential': 0.99, # High potential for physical form
            'grounding_factor': 0.98,      # Strong grounding influence
            'physicality_density': 0.95,   # High density reflecting physical matter
            'elemental_synthesis': 0.85,   # Synthesizes the four lower elements
            'earth_resonance': 0.90        # Planetary correspondence (Earth itself)
        }

        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'malkuth_properties': self.malkuth_properties
        })

        # Initialize energy grid
        self._energy_grid = None
        self._base_contribution = None
        self._random_contribution = None
        self._sephiroth_contribution = None

        # Initialize Malkuth-specific aspects
        self._initialize_malkuth_aspects()

        logger.info(f"Malkuth Field '{self.name}' initialized with {len(self.aspects)} aspects")

    def _initialize_malkuth_aspects(self) -> None:
        """
        Initialize Malkuth-specific aspects.

        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Malkuth-specific aspects
            aspects_data = {
                'divine_kingdom': {
                    'frequency': 285.0,  # Malkuth base frequency
                    'color': 'mixed earthy tones',
                    'element': 'earth',
                    'keywords': ['kingdom', 'realm', 'physical', 'manifest', 'presence', 'domain'],
                    'description': 'The aspect of the manifested divine kingdom, the physical plane'
                },
                'grounding_stability': {
                    'frequency': 174.0, # Solfeggio freq for grounding/lowest physical
                    'color': 'brown/black',
                    'element': 'earth',
                    'keywords': ['grounding', 'stability', 'earth', 'foundation', 'physicality', 'roots'],
                    'description': 'The aspect of grounding energy and physical stability'
                },
                'manifestation': {
                    'frequency': 396.0, # Solfeggio UT - Root chakra frequency/manifestation
                    'color': 'russet/red',
                    'element': 'earth/fire', # Manifestation involves Fire's action on Earth
                    'keywords': ['manifestation', 'creation', 'materialization', 'form', 'action', 'result'],
                    'description': 'The aspect of bringing spirit into physical manifestation'
                },
                'elemental_synthesis': {
                    'frequency': 432.0, # Harmonic frequency associated with Earth balance
                    'color': 'citrine/olive',
                    'element': 'earth/water/air/fire', # All four elements synthesized
                    'keywords': ['elements', 'synthesis', 'integration', 'balance', 'nature', 'world'],
                    'description': 'The aspect representing the synthesis of the four elements in the physical'
                },
                 'shekinah': {
                    'frequency': 528.0, # Presence/Love/Divine Feminine frequency
                    'color': 'white/gold/mixed',
                    'element': 'earth/spirit', # Divine presence within Earth
                    'keywords': ['presence', 'shekinah', 'divine_feminine', 'immanence', 'indwelling', 'bride'],
                    'description': 'The aspect of the indwelling divine presence (Shekinah)'
                },
                'earth_body': {
                    'frequency': 7.83, # Schumann Resonance - Earth's heartbeat
                    'color': 'green/brown',
                    'element': 'earth',
                    'keywords': ['earth', 'gaia', 'body', 'physicality', 'nature', 'planet'],
                    'description': 'The aspect connecting to the body of the Earth (Gaia)'
                }
            }

            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Malkuth principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'kingdom', 'physical', 'manifest', 'grounding', 'earth', 'elements', 'stability', 'body'}
                )) / max(1, len(data['keywords']))

                # Frequency alignment with Malkuth's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)

                # Element alignment (Earth primary)
                element_alignment = 1.0 if 'earth' in data['element'] else 0.3 # Strong preference for Earth

                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.5, strength)) # Ensure decent minimum strength for Malkuth aspects

                # Add aspect (will raise ValueError if name exists or strength invalid)
                self.add_aspect(name, strength, data)

        except (ValueError, TypeError) as e:
             # Catch errors from add_aspect
             error_msg = f"Failed to add Malkuth aspect: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Malkuth aspects: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Malkuth-specific patterns to the energy grid. Reflects physical manifestation.

        Raises:
            RuntimeError: If energy grid or required contribution arrays are not initialized.
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Malkuth patterns")
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

        # Malkuth: Kingdom, Earth, Manifestation, Grounding, Stability
        # Patterns should be dense, stable, ordered, perhaps crystalline or earthy textures.

        # 1. Grounding Pattern (Density increases towards the bottom z=0)
        grounding_pattern = 0.7 + 0.3 * (1.0 - z_norm)**3 # Stronger influence at the bottom

        # 2. Manifestation Grid (Stable, orthogonal structure - like crystal lattice)
        grid_freq = 3 # Lower frequency for stability and density
        # Use squared cosines for more defined grid points
        manifestation_pattern = 0.4 + 0.6 * (np.cos(grid_freq * np.pi * x_norm)**2 *
                                             np.cos(grid_freq * np.pi * y_norm)**2 *
                                             np.cos(grid_freq * np.pi * z_norm)**2)

        # 3. Elemental Synthesis (Interwoven patterns representing the four elements)
        # Example: Combine patterns with frequencies typical of elements
        fire_freq = 8; water_freq = 6; air_freq = 10; earth_freq = 4 # Relative frequencies
        elem_x = np.sin(fire_freq * np.pi * x_norm) * np.cos(earth_freq * np.pi * x_norm)
        elem_y = np.sin(water_freq * np.pi * y_norm) * np.cos(air_freq * np.pi * y_norm)
        elem_z = np.sin(air_freq * np.pi * z_norm) * np.cos(water_freq * np.pi * z_norm)
        element_pattern = 0.5 + 0.5 * (elem_x * elem_y * elem_z) # Mix complexly

        # 4. Physicality Density (Higher density overall, less fluctuation than higher realms)
        # Represents denser energy, stronger base level
        center = np.array([0.5, 0.5, 0.5])
        point_coords = np.stack(norm_coords, axis=-1)
        distance_sq = np.sum((point_coords - center)**2, axis=-1)
        density_pattern = 0.8 + 0.2 * np.exp(-distance_sq * 8) # High base density, slightly higher at center

        # 5. Earth Resonance (Schumann frequency influence - stable, low frequency)
        schumann_freq_scaled = 7.83 / 4 # Scaled very low for grid
        # Stable, slow oscillation
        earth_pattern = 0.5 + 0.5 * np.cos(schumann_freq_scaled * np.pi * (x_norm + y_norm + z_norm))**2

        # Combine patterns reflecting Malkuth's stable, manifest, earthy nature
        self._sephiroth_contribution = (
            0.25 * grounding_pattern +      # Core grounding effect
            0.30 * manifestation_pattern + # Structure of physical reality (Strong)
            0.10 * element_pattern +       # Synthesis of lower elements
            0.20 * density_pattern +       # Physical density representation
            0.15 * earth_pattern           # Earth resonance/Schumann
        )

        # Ensure values are within the valid range [0, 1]
        self._sephiroth_contribution = np.clip(self._sephiroth_contribution, 0.0, 1.0)

        # Update the final energy grid
        self._energy_grid = (
            self.base_field_ratio * self._base_contribution +
            self.sephiroth_ratio * self._sephiroth_contribution +
            self.random_ratio * self._random_contribution
        )
        # Malkuth has high stability, less randomness influence maybe? Let ratio handle it for now.
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0) # Final clip
        logger.debug(f"Applied Malkuth-specific patterns to energy grid")

    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Malkuth field's energy grid with specific patterns.

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
            logger.warning(f"Composition verification failed during Malkuth initialization: {e}")
        return self._energy_grid

    def ground_entity(self, entity_id: str, grounding_strength: float = 0.9) -> float:
        """
        Apply Malkuth's grounding energy to an entity, increasing stability.

        Args:
            entity_id: ID of the entity to ground.
            grounding_strength: Intensity of the grounding effect (0.0-1.0).

        Returns:
            Resulting grounding effect strength applied (0.0-1.0).

        Raises:
            ValueError: If parameters are invalid or entity not found in field.
            TypeError: If argument types are incorrect.
            RuntimeError: If operation fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(entity_id, str) or not entity_id:
             raise TypeError("entity_id must be a non-empty string.")
        if not isinstance(grounding_strength, (int, float)) or not (0.0 <= grounding_strength <= 1.0):
            raise ValueError(f"Grounding strength must be between 0.0 and 1.0, got {grounding_strength}")

        entity_ref = None
        for entity in self.entities:
             if isinstance(entity, dict) and entity.get('id') == entity_id:
                  entity_ref = entity
                  break
        if entity_ref is None:
            raise ValueError(f"Entity '{entity_id}' not found in Malkuth field {self.field_id}")

        try:
            # Calculate grounding effect based on Malkuth properties
            base_grounding = self.malkuth_properties.get('grounding_factor', 0.95) # Default if missing
            earth_resonance = self.malkuth_properties.get('earth_resonance', 0.8)

            # Grounding effectiveness increases with field stability
            grounding_effect = base_grounding * earth_resonance * grounding_strength * self.stability
            grounding_effect = min(1.0, max(0.0, grounding_effect)) # Clamp result

            # Simulate grounding effect on entity state
            # Assumes entity has 'stability' and 'frequency' attributes
            current_entity_stability = entity_ref.get('stability', 0.5)
            current_entity_frequency = entity_ref.get('frequency', 432.0)

            if not isinstance(current_entity_stability, (int, float)): current_entity_stability = 0.5
            if not isinstance(current_entity_frequency, (int, float)): current_entity_frequency = 432.0

            # Grounding increases stability and potentially lowers frequency
            stability_increase = grounding_effect * 0.15 # Scaled effect
            new_stability = min(1.0, current_entity_stability + stability_increase)

            # Lower frequency towards Malkuth's base or Earth resonance
            frequency_change_factor = grounding_effect * 0.2
            target_freq = min(current_entity_frequency, self.base_frequency) # Move towards lower freq
            frequency_decrease = (current_entity_frequency - target_freq) * frequency_change_factor
            new_frequency = max(50.0, current_entity_frequency - frequency_decrease) # Ensure minimum frequency


            # Record the event
            grounding_event = {
                'entity_id': entity_id,
                'strength_applied': grounding_strength,
                'grounding_effect': grounding_effect,
                'new_stability': new_stability,
                'new_frequency': new_frequency,
                'timestamp': datetime.now().isoformat()
            }

            # Add event to entity's record if structure allows
            if 'malkuth_groundings' not in entity_ref: entity_ref['malkuth_groundings'] = []
            if isinstance(entity_ref['malkuth_groundings'], list):
                 entity_ref['malkuth_groundings'].append(grounding_event)
            else: logger.warning(f"Entity {entity_id} 'malkuth_groundings' is not list.")

            # Update entity state directly
            entity_ref['stability'] = new_stability
            entity_ref['frequency'] = new_frequency

            logger.info(f"Grounded entity {entity_id}. Effect: {grounding_effect:.3f}. New Stability: {new_stability:.3f}, New Frequency: {new_frequency:.1f}Hz")
            return float(grounding_effect)

        except Exception as e:
             error_msg = f"Error grounding entity {entity_id}: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Malkuth field's current state.

        Returns:
            Dictionary of Malkuth field metrics.
        """
        base_metrics = super().get_field_metrics()

        malkuth_metrics = {
            'malkuth_properties': self.malkuth_properties.copy(), # Return a copy
            'realized_manifestation_potential': self.malkuth_properties.get('manifestation_potential', 0) * self.stability,
            'current_grounding_strength': self.malkuth_properties.get('grounding_factor', 0) * self.stability,
            'current_physicality_density': self.malkuth_properties.get('physicality_density', 0), # Density property itself
            'elemental_synthesis_level': self.malkuth_properties.get('elemental_synthesis', 0) * self.coherence,
            'realized_earth_resonance': self.malkuth_properties.get('earth_resonance', 0)
        }

        # Aggregate grounding statistics
        total_grounding_effect = 0.0
        grounding_event_count = 0
        grounded_entities = 0
        for entity in self.entities:
             groundings = entity.get('malkuth_groundings', [])
             if isinstance(groundings, list) and groundings:
                 grounded_entities += 1
                 grounding_event_count += len(groundings)
                 total_grounding_effect += sum(g.get('grounding_effect', 0.0) for g in groundings)

        malkuth_metrics['total_grounding_events'] = grounding_event_count
        malkuth_metrics['grounded_entity_count'] = grounded_entities
        malkuth_metrics['average_grounding_effect'] = total_grounding_effect / grounding_event_count if grounding_event_count > 0 else 0.0

        combined_metrics = {
            **base_metrics,
            'malkuth_specific': malkuth_metrics,
        }
        return combined_metrics

    def __str__(self) -> str:
        """String representation of the Malkuth field."""
        manifest = self.malkuth_properties.get('manifestation_potential', 0.0)
        return f"MalkuthField(name={self.name}, aspects={len(self.aspects)}, manifestation={manifest:.2f})"

    def __repr__(self) -> str:
        """Detailed representation."""
        grounding = self.malkuth_properties.get('grounding_factor', 0.0)
        return f"<MalkuthField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} grounding={grounding:.2f}>"

# --- END OF FILE malkuth_field.py ---