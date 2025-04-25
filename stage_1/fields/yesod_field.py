# --- START OF FILE yesod_field.py ---

"""
Yesod Field Module

Defines the YesodField class which represents the specific Sephiroth field for Yesod.
Implements Yesod-specific aspects, frequencies, colors, and other properties.

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

class YesodField(SephirothField):
    """
    Represents the Yesod Sephiroth field - the embodiment of foundation, the etheric plane,
    dreams, illusions, the subconscious, and the astral realm in the soul development framework.
    """

    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Yesod - Foundation", # More descriptive default name
                 dimensions: Tuple[float, float, float] = (70.0, 70.0, 70.0),
                 base_frequency: float = 369.0, # Yesod frequency (Related to Etheric/Astral, 9)
                 resonance: float = 0.86,
                 stability: float = 0.82, # More fluid than Hod/Netzach, astral can shift
                 coherence: float = 0.80, # Can be illusory, affecting coherence
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Yesod field with Yesod-specific properties.

        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Yesod-specific properties
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
            sephiroth_name="yesod"
        )

        # Set Yesod-specific attributes
        self.divine_attribute = "foundation"
        self.geometric_correspondence = "enneagon (9-sided)" # Often associated with 9
        self.element = "air/water" # Etheric/Astral - mixture, reflects Air over Water
        self.primary_color = "purple/violet/indigo" # Colors of the astral, moon

        # Yesod-specific properties
        self.yesod_properties = {
            'foundation_strength': 0.94,
            'etheric_density': 0.85, # How strongly it relates to etheric plane
            'dream_potential': 0.92, # Potential for dream states/imagery
            'illusion_factor': 0.40, # Potential for illusion/glamour
            'moon_influence': 0.88 # Planetary correspondence
        }

        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'yesod_properties': self.yesod_properties
        })

        # Initialize energy grid
        self._energy_grid = None
        
        # Initialize Yesod-specific aspects
        self._initialize_yesod_aspects()

        logger.info(f"Yesod Field '{self.name}' initialized with {len(self.aspects)} aspects")

    def _initialize_yesod_aspects(self) -> None:
        """
        Initialize Yesod-specific aspects.

        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Yesod-specific aspects
            aspects_data = {
                'divine_foundation': {
                    'frequency': 369.0,  # Yesod base frequency
                    'color': 'deep purple',
                    'element': 'earth/water', # Foundation implies Earth stability
                    'keywords': ['foundation', 'basis', 'stability', 'support', 'root', 'structure'],
                    'description': 'The aspect of the divine foundation supporting creation'
                },
                'etheric_plane': {
                    'frequency': 417.0, # Solfeggio RE - Change/transition planes
                    'color': 'violet/silver',
                    'element': 'aether/air',
                    'keywords': ['etheric', 'astral', 'subtle', 'energy_body', 'template', 'liminal'],
                    'description': 'The aspect representing the etheric and astral planes'
                },
                'dream_and_imagination': {
                    'frequency': 396.0, # Solfeggio UT - Root/unconscious
                    'color': 'indigo',
                    'element': 'water',
                    'keywords': ['dream', 'imagination', 'illusion', 'subconscious', 'imagery', 'fantasy'],
                    'description': 'The aspect of dreams, the subconscious, and imagination'
                },
                'reflection_and_imagery': {
                    'frequency': 432.0, # Harmonic frequency
                    'color': 'silver/purple',
                    'element': 'water', # Reflection like water
                    'keywords': ['reflection', 'image', 'mirror', 'perception', 'illusion', 'appearance'],
                    'description': 'The aspect of reflection and the formation of images (Maya)'
                },
                 'generative_force': {
                    'frequency': 528.0, # Connected to creation/love/life force
                    'color': 'purple/pink',
                    'element': 'water', # Life force often linked to water
                    'keywords': ['generation', 'sexuality', 'creativity', 'fertility', 'life_force'],
                    'description': 'The aspect of the generative life force and potential'
                },
                'moon_influence': {
                    'frequency': 210.42, # Moon frequency
                    'color': 'silver/violet',
                    'element': 'water',
                    'keywords': ['moon', 'cycles', 'emotions', 'subconscious', 'intuition', 'tides'],
                    'description': 'The planetary aspect of the Moon'
                }
            }

            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Yesod principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'foundation', 'etheric', 'dream', 'illusion', 'reflection', 'generation', 'moon', 'subconscious'}
                )) / max(1, len(data['keywords']))

                # Frequency alignment with Yesod's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)

                # Element alignment (Water/Air/Aether primary)
                element_alignment = 1.0 if data['element'] in ['water', 'air', 'aether'] else 0.4

                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.5, strength)) # Ensure decent minimum strength for Yesod aspects

                # Add aspect (will raise ValueError if name exists or strength invalid)
                self.add_aspect(name, strength, data)

        except (ValueError, TypeError) as e:
             # Catch errors from add_aspect
             error_msg = f"Failed to add Yesod aspect: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Yesod aspects: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Yesod-specific patterns to the energy grid. Reflects etheric/astral nature.

        Raises:
            RuntimeError: If energy grid or required contribution arrays are not initialized.
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Yesod patterns")
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

        # Yesod: Foundation, Etheric, Dreams, Illusion, Moon
        # Patterns should be fluid, layered, shimmering, reflecting astral qualities.

        # 1. Foundation Pattern (Stable base, perhaps stronger at lower Z reflecting connection to Malkuth)
        foundation_pattern = 0.6 + 0.4 * np.exp(-z_norm * 3) # Stronger at z=0 (bottom)

        # 2. Etheric Weave (Fine, shimmering, interconnected energy strands)
        weave_freq_base = 10
        # Add slight time-like variation for shimmer (use distance as proxy)
        distance = np.sqrt(x_norm**2 + y_norm**2 + z_norm**2)
        shimmer_factor = 1.0 + 0.1 * np.sin(distance * 5 * np.pi)
        weave_freq = weave_freq_base * shimmer_factor
        etheric_pattern = 0.5 + 0.5 * (np.sin(weave_freq * np.pi * x_norm + weave_freq * np.pi * y_norm) *
                                     np.cos(weave_freq * np.pi * y_norm + weave_freq * np.pi * z_norm))

        # 3. Dream/Illusion Pattern (Shifting, wave interference, potentially less coherent areas)
        wave1_freq = 4; wave2_freq = 6; wave3_freq = 9 # Frequencies related to 9?
        illusion_pattern = 0.5 + 0.5 * (np.sin(wave1_freq * np.pi * x_norm + wave2_freq * np.pi * y_norm) *
                                        np.cos(wave2_freq * np.pi * y_norm - wave3_freq * np.pi * z_norm) *
                                        np.sin(wave3_freq * np.pi * z_norm - wave1_freq * np.pi * x_norm))
        # Modulate coherence based on this pattern - where pattern is weak, coherence might be lower
        coherence_modulation = 0.7 + 0.3 * illusion_pattern # Higher energy pattern = higher coherence

        # 4. Reflection Pattern (Mirror-like symmetries, layered feeling)
        # Example: Create layers based on distance from center
        num_layers = 5
        layer_pattern = np.floor(distance * num_layers) % 2 # Alternating layers 0 and 1
        reflection_pattern = 0.6 + 0.4 * layer_pattern # Two distinct layer energy levels

        # 5. Moon Influence (Cyclical, flowing, tidal patterns)
        moon_freq_scaled = 210.42 / 15 # Scaled for grid
        # Tidal flow pattern based on Z and XY angle
        angle_xy = np.arctan2(y_norm - 0.5, x_norm - 0.5)
        moon_pattern = 0.5 + 0.5 * np.sin(moon_freq_scaled * np.pi * z_norm * 2 + angle_xy * 1.5) # Cycle along Z axis, influenced by angle

        # Combine patterns reflecting Yesod's etheric/foundational nature
        self._sephiroth_contribution = (
            0.25 * foundation_pattern +    # Foundation aspect
            0.20 * etheric_pattern +       # Etheric weave/shimmer
            0.25 * illusion_pattern +      # Dream/Illusion (base energy pattern)
            0.15 * reflection_pattern +    # Reflection/Imagery layers
            0.15 * moon_pattern            # Lunar/Cyclical influence
        )
        # Apply coherence modulation derived from illusion pattern
        self._sephiroth_contribution *= coherence_modulation

        # Ensure values are within the valid range [0, 1]
        self._sephiroth_contribution = np.clip(self._sephiroth_contribution, 0.0, 1.0)

        # Update the final energy grid
        self._energy_grid = (
            self.base_field_ratio * self._base_contribution +
            self.sephiroth_ratio * self._sephiroth_contribution +
            self.random_ratio * self._random_contribution
        )
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0) # Final clip
        logger.debug(f"Applied Yesod-specific patterns to energy grid")

    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Yesod field's energy grid with specific patterns.

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
            logger.warning(f"Composition verification failed during Yesod initialization: {e}")
        return self._energy_grid

    def perceive_illusion(self, entity_id: str) -> float:
        """
        Simulates an entity perceiving the illusory nature of the Yesod field.

        Args:
            entity_id: ID of the entity perceiving.

        Returns:
            A value (0.0-1.0) representing the strength of the illusion perceived by the entity.

        Raises:
            ValueError: If entity not found in field.
            TypeError: If entity_id is not a string.
            RuntimeError: If operation fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(entity_id, str) or not entity_id:
             raise TypeError("entity_id must be a non-empty string.")

        entity_ref = None
        for entity in self.entities:
             if isinstance(entity, dict) and entity.get('id') == entity_id:
                  entity_ref = entity
                  break
        if entity_ref is None:
            raise ValueError(f"Entity '{entity_id}' not found in Yesod field {self.field_id}")

        try:
            # Illusion strength depends on field's illusion factor and local coherence
            illusion_factor = self.yesod_properties.get('illusion_factor', 0.4) # Default if missing
            field_coherence = self.coherence # Overall field coherence

            # Get entity position to sample local field coherence/energy
            local_coherence_factor = field_coherence # Default to field average
            if self._energy_grid is not None and 'position' in entity_ref:
                 entity_pos = entity_ref.get('position')
                 # Validate entity position before using it
                 if isinstance(entity_pos, tuple) and len(entity_pos) == 3 and all(isinstance(c, (int, float)) for c in entity_pos):
                     try:
                         # Energy grid values approximate local order/coherence
                         energy_at_point = self.get_energy_at_point(entity_pos)
                         # Higher energy -> higher coherence -> less illusion
                         local_coherence_factor = energy_at_point
                     except (ValueError, RuntimeError) as e:
                         logger.warning(f"Could not get energy at entity point {entity_pos} for illusion calc: {e}")
                 else:
                     logger.warning(f"Entity {entity_id} has invalid position {entity_pos}. Using average coherence.")


            # Perceived illusion strength: Base field illusion factor, reduced by local coherence
            # Randomness reflects shifting nature of illusion
            random_factor = 0.9 + np.random.random() * 0.2 # Randomness between 0.9 and 1.1
            perceived_illusion = illusion_factor * max(0, (1.0 - local_coherence_factor * 0.8)) * random_factor
            perceived_illusion = min(1.0, max(0.0, perceived_illusion)) # Clamp result [0, 1]

            # Record event
            perception_event = {
                'entity_id': entity_id,
                'perceived_illusion_strength': perceived_illusion,
                'field_illusion_factor': illusion_factor,
                'local_coherence_estimate': local_coherence_factor,
                'timestamp': datetime.now().isoformat()
            }

            # Add event to entity's record if structure allows
            if 'yesod_perceptions' not in entity_ref: entity_ref['yesod_perceptions'] = []
            if isinstance(entity_ref['yesod_perceptions'], list):
                 entity_ref['yesod_perceptions'].append(perception_event)
            else: logger.warning(f"Entity {entity_id} 'yesod_perceptions' is not list.")

            logger.info(f"Entity {entity_id} perceived illusion with strength {perceived_illusion:.3f} (Local Coherence Factor: {local_coherence_factor:.3f})")
            return float(perceived_illusion)

        except Exception as e:
             error_msg = f"Error during illusion perception for entity {entity_id}: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Yesod field's current state.

        Returns:
            Dictionary of Yesod field metrics.
        """
        base_metrics = super().get_field_metrics()

        yesod_metrics = {
            'yesod_properties': self.yesod_properties.copy(), # Return a copy
            'realized_foundation_strength': self.yesod_properties.get('foundation_strength', 0) * self.stability,
            'current_etheric_density': self.yesod_properties.get('etheric_density', 0) * self.coherence,
            'current_dream_potential': self.yesod_properties.get('dream_potential', 0) * self.resonance,
            'average_illusion_potential': self.yesod_properties.get('illusion_factor', 0) * (1.0 - self.coherence),
            'realized_lunar_influence': self.yesod_properties.get('moon_influence', 0)
        }

        # Aggregate perception statistics
        total_perceptions = 0
        total_perceived_illusion_strength = 0.0
        perceiving_entities = 0
        for entity in self.entities:
             perceptions = entity.get('yesod_perceptions', [])
             if isinstance(perceptions, list) and perceptions:
                 perceiving_entities += 1
                 total_perceptions += len(perceptions)
                 total_perceived_illusion_strength += sum(p.get('perceived_illusion_strength', 0.0) for p in perceptions)

        yesod_metrics['total_perception_events'] = total_perceptions
        yesod_metrics['perceiving_entity_count'] = perceiving_entities
        yesod_metrics['average_perceived_illusion'] = total_perceived_illusion_strength / total_perceptions if total_perceptions > 0 else 0.0

        combined_metrics = {
            **base_metrics,
            'yesod_specific': yesod_metrics,
        }
        return combined_metrics

    def __str__(self) -> str:
        """String representation of the Yesod field."""
        foundation = self.yesod_properties.get('foundation_strength', 0.0)
        return f"YesodField(name={self.name}, aspects={len(self.aspects)}, foundation={foundation:.2f})"

    def __repr__(self) -> str:
        """Detailed representation."""
        dream = self.yesod_properties.get('dream_potential', 0.0)
        return f"<YesodField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} dream={dream:.2f}>"

# --- END OF FILE yesod_field.py ---