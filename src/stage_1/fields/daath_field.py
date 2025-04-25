# --- START OF FILE daath_field.py ---

"""
Daath Field Module

Defines the DaathField class which represents the specific, often hidden or
transitional, dimension of Daath (Knowledge).

Daath is not typically considered one of the 10 Sephiroth but rather a state
of consciousness or a gateway between Binah and Chokmah. Its properties reflect
this bridging and potentially abyssal nature.

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

class DaathField(SephirothField):
    """
    Represents the Daath dimension - Knowledge, the Abyss, the synthesis of Chokmah and Binah.
    It acts as a bridge or a void depending on the context. Uses SephirothField base for composition.
    """

    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Daath - Knowledge/Abyss",
                 dimensions: Tuple[float, float, float] = (85.0, 85.0, 85.0), # Slightly smaller than Binah/Chokmah
                 base_frequency: float = 618.0, # Frequency related to Phi inverse (1/PHI * 1000 approx)
                 resonance: float = 0.88, # High resonance potential
                 stability: float = 0.75, # Can be unstable (Abyss aspect)
                 coherence: float = 0.80, # Coherence can fluctuate
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Daath field with its unique properties.

        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Daath-specific properties
            random_ratio: Proportion of random variation

        Raises:
            ValueError: If any parameters are invalid
            TypeError: If parameters are of incorrect type
        """
        # Initialize the base Sephiroth field, using "daath" as name
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
            sephiroth_name="daath" # Use "daath" although not a standard Sephirah
        )

        # Set Daath-specific attributes
        self.divine_attribute = "knowledge" # Knowledge, but also potentially Abyss
        self.geometric_correspondence = "point/void" # Point of synthesis or void
        self.element = "aether/void" # Bridge element or absence
        self.primary_color = "lavender/grey" # Often lavender or grey

        # Daath-specific properties
        self.daath_properties = {
            'knowledge_potential': 0.98,
            'integration_factor': 0.90, # Integrates Chokmah/Binah
            'abyss_potential': 0.60, # Potential for instability/void
            'gate_threshold': 0.85, # Threshold to pass through
            'synthesis_power': 0.92
        }

        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'daath_properties': self.daath_properties
        })

        # Initialize Daath-specific aspects
        self._initialize_daath_aspects()

        logger.info(f"Daath Field '{self.name}' initialized with {len(self.aspects)} aspects")

    def _initialize_daath_aspects(self) -> None:
        """
        Initialize Daath-specific aspects. Reflects both knowledge and abyss.

        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Daath-specific aspects
            aspects_data = {
                'unified_knowledge': {
                    'frequency': 618.0, # Daath base frequency
                    'color': 'bright lavender',
                    'element': 'aether',
                    'keywords': ['knowledge', 'gnosis', 'synthesis', 'knowing', 'awareness'],
                    'description': 'The aspect of unified knowledge beyond duality'
                },
                'the_abyss': {
                    'frequency': 144.0, # Lower, potentially chaotic frequency
                    'color': 'dark grey/black',
                    'element': 'void',
                    'keywords': ['abyss', 'void', 'separation', 'illusion', 'unknowing', 'nothingness'],
                    'description': 'The aspect representing the Abyss, separation from unity'
                },
                'integration_point': {
                    'frequency': 528.0, # Frequency of transformation/harmony
                    'color': 'silver/lavender',
                    'element': 'aether',
                    'keywords': ['integration', 'synthesis', 'balance', 'connection', 'bridge'],
                    'description': 'The aspect of integrating higher principles'
                },
                'hidden_wisdom': {
                    'frequency': 932.0, # Chokmah frequency link
                    'color': 'grey/silver',
                    'element': 'light', # Chokmah element
                    'keywords': ['wisdom', 'hidden', 'secret', 'potential', 'unmanifest'],
                    'description': 'The hidden potential of Chokmah reflected through Daath'
                },
                'structured_potential': {
                    'frequency': 852.0, # Binah frequency link
                    'color': 'black/lavender',
                    'element': 'water/aether', # Binah elements
                    'keywords': ['potential', 'structure', 'form', 'latent', 'understanding'],
                    'description': 'The structured potential of Binah accessible via Daath'
                },
                'gateway': {
                    'frequency': 369.0, # A transitional frequency (Yesod related?)
                    'color': 'shifting lavender/grey',
                    'element': 'void/aether',
                    'keywords': ['gateway', 'portal', 'transition', 'threshold', 'crossing'],
                    'description': 'The aspect of Daath as a gateway or threshold'
                }
            }

            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Daath principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'knowledge', 'synthesis', 'integration', 'abyss', 'gateway', 'knowing', 'hidden', 'potential'}
                )) / max(1, len(data['keywords']))

                # Frequency alignment with Daath's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)

                # Element alignment (Daath is complex)
                element_alignment = 1.0 if data['element'] in ['aether', 'void'] else 0.5

                # Calculate overall strength - more variable due to Daath's nature
                strength = (0.4 * keywords_alignment + 0.4 * freq_alignment + 0.2 * element_alignment)
                # Allow lower min strength for abyss aspect
                min_strength = 0.3 if 'abyss' in name else 0.5
                # Add randomness reflecting instability/potential
                strength = min(1.0, max(min_strength, strength)) * (0.8 + np.random.random() * 0.4)
                strength = min(1.0, max(0.1, strength)) # Final clamp

                # Add aspect (will raise ValueError if name exists or strength invalid)
                self.add_aspect(name, strength, data)

        except (ValueError, TypeError) as e:
             # Catch errors from add_aspect
             error_msg = f"Failed to add Daath aspect: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Daath aspects: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Daath-specific patterns to the energy grid. Reflects duality.

        Raises:
            RuntimeError: If energy grid or required contribution arrays are not initialized.
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Daath patterns")
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

        # Daath: Knowledge, Synthesis, Abyss, Gateway
        # Patterns reflect this duality and transition.

        # 1. Synthesis Pattern (Converging energy towards center, strong core)
        synthesis_pattern = 1.0 - distance_norm**0.7 # Stronger towards center

        # 2. Abyss Pattern (Areas of lower energy/stability, potentially dynamic)
        # Example: Chaotic modulation based on position, stronger near edges
        edge_dist = np.minimum.reduce([x_norm, 1-x_norm, y_norm, 1-y_norm, z_norm, 1-z_norm])
        # Use Perlin noise or similar for chaotic feel if available, otherwise simulate with sine waves
        chaos_freq1, chaos_freq2 = 8, 13 # Fibonacci related frequencies
        abyss_factor = 0.3 * np.sin(chaos_freq1 * np.pi * x_norm + chaos_freq2 * np.pi * y_norm) * \
                           np.cos(chaos_freq1 * np.pi * y_norm + chaos_freq2 * np.pi * z_norm) * \
                           np.exp(-edge_dist * 4) # Stronger effect near edges
        abyss_pattern = 0.6 + abyss_factor # Average base, modulated chaotically

        # 3. Knowledge Lattice (Fine, interwoven high-frequency structure)
        knowledge_freq = 16 # Higher frequency grid
        knowledge_pattern = 0.5 + 0.5 * (np.sin(knowledge_freq * np.pi * x_norm) *
                                         np.cos(knowledge_freq * np.pi * y_norm) *
                                         np.sin(knowledge_freq * np.pi * z_norm))

        # 4. Gateway Potential (Fluctuating energy, perhaps stronger along connection axes)
        # Example: Stronger potential along Z-axis (connection Kether-Tiphareth)
        gateway_fluctuation = 0.4 * np.sin(5 * np.pi * z_norm + 3 * np.pi * distance_norm) * np.exp(-distance_norm * 1.5)
        gateway_pattern = 0.5 + gateway_fluctuation

        # Combine patterns with weights reflecting Daath's dual nature
        self._sephiroth_contribution = (
            0.30 * synthesis_pattern +    # Core synthesis/knowledge potential
            0.25 * knowledge_pattern +    # Intricate knowledge aspect
            0.25 * gateway_pattern +      # Transitional/Gateway aspect
            0.20 * abyss_pattern          # Abyss/Unknowing/Instability aspect
        )

        # Apply inherent instability modulation based on Daath property
        abyss_potential = self.daath_properties.get('abyss_potential', 0.5)
        # Random noise scaled by abyss potential
        instability_mod = 1.0 + (abyss_potential * 0.4) * (np.random.rand(*grid_shape) - 0.5) # +/- 20% * abyss_potential
        self._sephiroth_contribution *= instability_mod

        # Clip final contribution
        self._sephiroth_contribution = np.clip(self._sephiroth_contribution, 0.0, 1.0)

        # Update the final energy grid
        self._energy_grid = (
            self.base_field_ratio * self._base_contribution +
            self.sephiroth_ratio * self._sephiroth_contribution +
            self.random_ratio * self._random_contribution
        )
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0) # Final clip
        logger.debug(f"Applied Daath-specific patterns to energy grid")

    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Daath field's energy grid with specific patterns.

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
            logger.warning(f"Composition verification failed during Daath initialization: {e}")
        return self._energy_grid

    def attempt_crossing(self, entity_id: str, entity_knowledge_level: float) -> bool:
        """
        Simulates an entity attempting to cross Daath (pass through the Abyss/Gateway).

        Args:
            entity_id: ID of the entity attempting the crossing.
            entity_knowledge_level: A measure (0.0-1.0) of the entity's knowledge/integration.

        Returns:
            True if the crossing is successful, False otherwise.

        Raises:
            ValueError: If parameters are invalid or entity not found in field.
            TypeError: If arguments have incorrect types.
            RuntimeError: If operation fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(entity_id, str) or not entity_id:
             raise TypeError("entity_id must be a non-empty string.")
        if not isinstance(entity_knowledge_level, (int, float)) or not (0.0 <= entity_knowledge_level <= 1.0):
            raise ValueError(f"Entity knowledge level must be between 0.0 and 1.0, got {entity_knowledge_level}")

        entity_ref = None
        for entity in self.entities:
             if isinstance(entity, dict) and entity.get('id') == entity_id:
                  entity_ref = entity
                  break
        if entity_ref is None:
            raise ValueError(f"Entity '{entity_id}' not found in Daath field {self.field_id}")

        try:
            # Calculate crossing difficulty based on Daath properties
            abyss_potential = self.daath_properties.get('abyss_potential', 0.5)
            field_stability = self.stability
            difficulty = abyss_potential * max(0, (1.0 - field_stability)) # Difficulty higher if field unstable

            # Calculate success threshold based on gate property and difficulty
            gate_threshold = self.daath_properties.get('gate_threshold', 0.8)
            # Effective threshold increases with difficulty
            success_threshold = min(1.0, gate_threshold + difficulty * 0.5)

            # Calculate entity's crossing potential based on knowledge and field support
            synthesis_power = self.daath_properties.get('synthesis_power', 0.8)
            crossing_potential = entity_knowledge_level * synthesis_power * self.resonance # Resonance helps

            # Determine success (add randomness influenced by instability)
            # Higher instability means more randomness in success
            instability_factor = 1.0 - field_stability
            random_swing = instability_factor * 0.5 # Max +/- 25% deviation if stability is 0
            random_roll = 1.0 + (np.random.random() - 0.5) * 2 * random_swing
            success = (crossing_potential * random_roll) >= success_threshold

            # Record the event
            crossing_event = {
                'entity_id': entity_id,
                'knowledge_level': entity_knowledge_level,
                'crossing_potential': crossing_potential,
                'difficulty_factor': difficulty,
                'success_threshold': success_threshold,
                'random_factor': random_roll,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }

            # Add event to entity's record if structure allows
            if 'daath_crossings' not in entity_ref: entity_ref['daath_crossings'] = []
            if isinstance(entity_ref['daath_crossings'], list):
                 entity_ref['daath_crossings'].append(crossing_event)
            else: logger.warning(f"Entity {entity_id} 'daath_crossings' is not a list.")

            if success:
                logger.info(f"Entity {entity_id} successfully crossed Daath (Potential: {crossing_potential:.3f} vs Threshold: {success_threshold:.3f}).")
                # Event indicating success could be triggered here
            else:
                logger.info(f"Entity {entity_id} failed to cross Daath (Potential: {crossing_potential:.3f} vs Threshold: {success_threshold:.3f}).")
                # Event indicating failure could be triggered here

            return success

        except Exception as e:
            error_msg = f"Error during Daath crossing attempt for entity {entity_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Daath field's current state.

        Returns:
            Dictionary of Daath field metrics.
        """
        base_metrics = super().get_field_metrics()

        daath_metrics = {
            'daath_properties': self.daath_properties.copy(), # Return a copy
            'realized_knowledge_potential': self.daath_properties.get('knowledge_potential', 0) * self.coherence,
            'current_abyss_influence': self.daath_properties.get('abyss_potential', 0) * (1.0 - self.stability),
            'realized_synthesis_power': self.daath_properties.get('synthesis_power', 0) * self.resonance,
            'current_gate_threshold': self.daath_properties.get('gate_threshold', 0) * (1.0 - self.daath_properties.get('abyss_potential', 0) * (1.0 - self.stability))
        }

        # Aggregate crossing attempt statistics
        total_attempts = 0
        successful_attempts = 0
        for entity in self.entities:
             crossings = entity.get('daath_crossings', [])
             if isinstance(crossings, list):
                 total_attempts += len(crossings)
                 successful_attempts += sum(1 for c in crossings if c.get('success', False))

        daath_metrics['total_crossing_attempts'] = total_attempts
        daath_metrics['crossing_success_rate'] = successful_attempts / total_attempts if total_attempts > 0 else 0.0

        combined_metrics = {
            **base_metrics,
            'daath_specific': daath_metrics
        }
        return combined_metrics

    def __str__(self) -> str:
        """String representation of the Daath field."""
        knowledge = self.daath_properties.get('knowledge_potential', 0.0)
        return f"DaathField(name={self.name}, aspects={len(self.aspects)}, knowledge={knowledge:.2f})"

    def __repr__(self) -> str:
        """Detailed representation."""
        abyss = self.daath_properties.get('abyss_potential', 0.0)
        return f"<DaathField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} abyss={abyss:.2f}>"

# --- END OF FILE daath_field.py ---