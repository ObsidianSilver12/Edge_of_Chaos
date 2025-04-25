# --- START OF FILE netzach_field.py ---

"""
Netzach Field Module

Defines the NetzachField class which represents the specific Sephiroth field for Netzach.
Implements Netzach-specific aspects, frequencies, colors, and other properties.

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

class NetzachField(SephirothField):
    """
    Represents the Netzach Sephiroth field - the embodiment of victory, endurance,
    emotion, inspiration, and the ' Netzach' aspect in the soul development framework.
    """

    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Netzach - Victory", # More descriptive default name
                 dimensions: Tuple[float, float, float] = (75.0, 75.0, 75.0),
                 base_frequency: float = 528.0, # Netzach frequency (Solfeggio MI - Love/Miracles, related to Venus)
                 resonance: float = 0.88,
                 stability: float = 0.80, # More dynamic/emotional than Hod
                 coherence: float = 0.85,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Netzach field with Netzach-specific properties.

        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Netzach-specific properties
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
            sephiroth_name="netzach"
        )

        # Set Netzach-specific attributes
        self.divine_attribute = "victory"
        self.geometric_correspondence = "seven-pointed star" # Often associated with Venus/7
        self.element = "fire" # Dynamic, emotional fire
        self.primary_color = "green/emerald" # Color of Venus, nature, emotion

        # Netzach-specific properties
        self.netzach_properties = {
            'victory_potential': 0.95,
            'endurance_factor': 0.90,
            'emotional_intensity': 0.88,
            'inspiration_flow': 0.92,
            'venus_resonance': 0.85 # Planetary correspondence
        }

        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'netzach_properties': self.netzach_properties
        })

        # Initialize energy grid with None
        self._energy_grid = None
        self._base_contribution = None
        self._random_contribution = None
        self._sephiroth_contribution = None

        # Initialize Netzach-specific aspects
        self._initialize_netzach_aspects()

        logger.info(f"Netzach Field '{self.name}' initialized with {len(self.aspects)} aspects")

    def _initialize_netzach_aspects(self) -> None:
        """
        Initialize Netzach-specific aspects.

        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Netzach-specific aspects
            aspects_data = {
                'divine_victory': {
                    'frequency': 528.0,  # Netzach base frequency
                    'color': 'emerald green',
                    'element': 'fire',
                    'keywords': ['victory', 'success', 'overcoming', 'achievement', 'triumph'],
                    'description': 'The aspect of divine victory and overcoming obstacles'
                },
                'endurance_and_perseverance': {
                    'frequency': 741.0, # Solfeggio SOL - Power/Expression
                    'color': 'deep green',
                    'element': 'earth/fire', # Endurance has an Earth quality
                    'keywords': ['endurance', 'perseverance', 'stamina', 'persistence', 'fortitude'],
                    'description': 'The aspect of enduring will and perseverance'
                },
                'unfettered_emotion': {
                    'frequency': 417.0, # Solfeggio RE - Change/Undoing, emotional release
                    'color': 'bright green',
                    'element': 'water/fire', # Emotional fire/water
                    'keywords': ['emotion', 'passion', 'feeling', 'desire', 'instinct', 'drive'],
                    'description': 'The aspect of raw, untamed emotion and desire'
                },
                'inspiration_and_creativity': {
                    'frequency': 528.0, # Love/Miracles frequency -> Inspiration
                    'color': 'green/gold',
                    'element': 'fire/air', # Inspiration often linked to Air/Fire
                    'keywords': ['inspiration', 'creativity', 'art', 'imagination', 'muse'],
                    'description': 'The aspect of divine inspiration and artistic creativity'
                },
                'love_and_attraction': {
                    'frequency': 528.0, # Love frequency
                    'color': 'rose green',
                    'element': 'fire', # Attraction/desire as Fire
                    'keywords': ['love', 'attraction', 'desire', 'beauty', 'relationship', 'passion'],
                    'description': 'The aspect of love, attraction, and relational energy'
                },
                'venus_influence': {
                    'frequency': 221.23, # Venus frequency
                    'color': 'emerald/copper',
                    'element': 'fire/water', # Venus related to both
                    'keywords': ['love', 'beauty', 'art', 'harmony', 'relationship', 'pleasure'],
                    'description': 'The planetary aspect of Venus'
                }
            }

            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Netzach principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'victory', 'endurance', 'emotion', 'inspiration', 'love', 'creativity', 'passion', 'desire'}
                )) / max(1, len(data['keywords']))

                # Frequency alignment with Netzach's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)

                # Element alignment (Fire primary, others secondary)
                element_alignment = 1.0 if 'fire' in data['element'] else (0.6 if 'water' in data['element'] else 0.3)

                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.6, strength)) # Ensure strong minimum strength for Netzach aspects

                # Add aspect (will raise ValueError if name exists or strength invalid)
                self.add_aspect(name, strength, data)

        except (ValueError, TypeError) as e:
             # Catch errors from add_aspect
             error_msg = f"Failed to add Netzach aspect: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Netzach aspects: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Netzach-specific patterns to the energy grid. Reflects dynamic emotion.

        Raises:
            RuntimeError: If energy grid or required contribution arrays are not initialized.
        """
        if not hasattr(self, '_base_contribution') or self._base_contribution is None:
            raise RuntimeError("Base contribution array (_base_contribution) not initialized.")
        if not hasattr(self, '_random_contribution') or self._random_contribution is None:
             raise RuntimeError("Random contribution array (_random_contribution) not initialized.")

        # Validate that energy grid was initialized by parent class
        if not hasattr(self, '_energy_grid') or self._energy_grid is None:
            raise RuntimeError("Energy grid not initialized")
            
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

        # Netzach: Victory, Endurance, Emotion, Fire, Venus
        # Patterns should be dynamic, flowing, passionate, creative bursts.

        # 1. Victory Pattern (Expanding, radiating energy bursts)
        # Use distance and time (simulated via a slow wave) for dynamic bursts
        time_factor = np.sin(np.pi * distance_norm * 2) # Simulate expansion wave
        victory_bursts = 0.5 + 0.5 * np.sin(distance_norm * 10 * np.pi + time_factor * 3) * np.exp(-distance_norm * 1.5)

        # 2. Endurance Pattern (Persistent, stable underlying current, less prominent)
        endurance_pattern = 0.6 + 0.4 * np.cos(np.pi * (x_norm + y_norm - z_norm)) # Slow, stable wave

        # 3. Emotional Flow (Dynamic, swirling, less predictable patterns)
        # Use higher frequencies and mixed axes
        swirl_freq1, swirl_freq2, swirl_freq3 = 6, 8, 10
        emotional_flow = 0.5 + 0.5 * (np.sin(swirl_freq1 * np.pi * x_norm + swirl_freq2 * np.pi * z_norm) *
                                     np.cos(swirl_freq2 * np.pi * y_norm + swirl_freq3 * np.pi * x_norm) *
                                     np.sin(swirl_freq3 * np.pi * z_norm + swirl_freq1 * np.pi * y_norm))

        # 4. Inspiration/Creativity (Sparkling, high-energy points)
        # Thresholded noise, representing creative sparks
        sparkle_threshold = 0.97
        sparkle_noise = np.random.rand(*grid_shape)
        inspiration_pattern = 0.4 + 0.6 * (sparkle_noise > sparkle_threshold).astype(float) # Base level + sparks

        # 5. Venus Influence (Harmonic resonance, patterns related to beauty/love)
        # Use frequency associated with Venus, create more complex harmonic structure
        venus_freq_scaled = 221.23 / 15 # Scaled planet frequency for grid
        phi = (1 + np.sqrt(5))/2
        venus_pattern = 0.5 + 0.5 * (np.sin(venus_freq_scaled * np.pi * (x_norm * phi + y_norm))) * \
                                   (np.cos(venus_freq_scaled * np.pi * (y_norm * phi + z_norm))) # Intertwined waves

        # Combine patterns reflecting Netzach's dynamic, emotional nature
        self._sephiroth_contribution = (
            0.25 * victory_bursts +        # Victory/Achievement
            0.10 * endurance_pattern +     # Endurance (Less dominant)
            0.35 * emotional_flow +        # Core emotion/Passion (Strongest)
            0.10 * inspiration_pattern +   # Creativity sparks
            0.20 * venus_pattern           # Venus influence (Love/Beauty)
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
        logger.debug(f"Applied Netzach-specific patterns to energy grid")

    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Netzach field's energy grid with specific patterns.

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
            logger.warning(f"Composition verification failed during Netzach initialization: {e}")
        return self._energy_grid

    def inspire_creativity(self, entity_id: str, inspiration_potential: float) -> float:
        """
        Stimulate creativity and inspiration in an entity within the Netzach field.

        Args:
            entity_id: ID of the entity to inspire.
            inspiration_potential: Base potential for inspiration (0.0-1.0).

        Returns:
            Actual inspiration energy transferred (0.0-1.0).

        Raises:
            ValueError: If parameters are invalid or entity not found in field.
            TypeError: If argument types are incorrect.
            RuntimeError: If operation fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(entity_id, str) or not entity_id:
             raise TypeError("entity_id must be a non-empty string.")
        if not isinstance(inspiration_potential, (int, float)) or not (0.0 <= inspiration_potential <= 1.0):
            raise ValueError(f"Inspiration potential must be between 0.0 and 1.0, got {inspiration_potential}")

        entity_ref = None
        for entity in self.entities:
             if isinstance(entity, dict) and entity.get('id') == entity_id:
                  entity_ref = entity
                  break
        if entity_ref is None:
            raise ValueError(f"Entity '{entity_id}' not found in Netzach field {self.field_id}")

        try:
            # Calculate inspiration effect based on Netzach properties
            inspiration_flow = self.netzach_properties.get('inspiration_flow', 0.8) # Default if missing
            emotional_intensity = self.netzach_properties.get('emotional_intensity', 0.8)

            # Inspiration energy depends on field flow, entity potential, field emotional state, and resonance
            inspiration_energy = inspiration_flow * inspiration_potential * emotional_intensity * self.resonance
            inspiration_energy = min(1.0, max(0.0, inspiration_energy)) # Clamp result

            # Record the event
            inspiration_event = {
                'entity_id': entity_id,
                'potential': inspiration_potential,
                'energy_transferred': inspiration_energy,
                'timestamp': datetime.now().isoformat()
            }

            # Add event to entity's record if structure allows
            if 'netzach_inspirations' not in entity_ref: entity_ref['netzach_inspirations'] = []
            if isinstance(entity_ref['netzach_inspirations'], list):
                 entity_ref['netzach_inspirations'].append(inspiration_event)
            else: logger.warning(f"Entity {entity_id} 'netzach_inspirations' is not list.")

            # Optionally update entity's creativity attribute
            current_creativity = entity_ref.get('creativity', 0.5) # Assume entity has this state
            if isinstance(current_creativity, (int, float)):
                 entity_ref['creativity'] = min(1.0, current_creativity + inspiration_energy * 0.15) # Example update scale

            logger.info(f"Inspired entity {entity_id} with energy {inspiration_energy:.3f}. New creativity: {entity_ref.get('creativity', 'N/A')}")
            return float(inspiration_energy)

        except Exception as e:
             error_msg = f"Error inspiring creativity in entity {entity_id}: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Netzach field's current state.

        Returns:
            Dictionary of Netzach field metrics.
        """
        base_metrics = super().get_field_metrics()

        netzach_metrics = {
            'netzach_properties': self.netzach_properties.copy(), # Return a copy
            'realized_victory_potential': self.netzach_properties.get('victory_potential', 0) * self.coherence,
            'current_emotional_intensity': self.netzach_properties.get('emotional_intensity', 0) * self.resonance * (1.0 - self.stability), # More intense when less stable?
            'current_inspiration_flow': self.netzach_properties.get('inspiration_flow', 0) * self.stability, # More stable = clearer flow?
            'realized_venus_influence': self.netzach_properties.get('venus_resonance', 0)
        }

        # Aggregate inspiration statistics
        total_inspiration_energy = 0.0
        inspiration_event_count = 0
        inspired_entities_count = 0
        for entity in self.entities:
             inspirations = entity.get('netzach_inspirations', [])
             if isinstance(inspirations, list) and inspirations:
                 inspired_entities_count += 1
                 inspiration_event_count += len(inspirations)
                 total_inspiration_energy += sum(insp.get('energy_transferred', 0.0) for insp in inspirations)

        netzach_metrics['total_inspiration_events'] = inspiration_event_count
        netzach_metrics['inspired_entity_count'] = inspired_entities_count
        netzach_metrics['average_inspiration_energy'] = total_inspiration_energy / inspiration_event_count if inspiration_event_count > 0 else 0.0

        combined_metrics = {
            **base_metrics,
            'netzach_specific': netzach_metrics,
        }
        return combined_metrics

    def __str__(self) -> str:
        """String representation of the Netzach field."""
        victory = self.netzach_properties.get('victory_potential', 0.0)
        return f"NetzachField(name={self.name}, aspects={len(self.aspects)}, victory={victory:.2f})"

    def __repr__(self) -> str:
        """Detailed representation."""
        emotion = self.netzach_properties.get('emotional_intensity', 0.0)
        return f"<NetzachField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} emotion={emotion:.2f}>"

# --- END OF FILE netzach_field.py ---