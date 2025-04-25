# --- START OF FILE hod_field.py ---

"""
Hod Field Module

Defines the HodField class which represents the specific Sephiroth field for Hod.
Implements Hod-specific aspects, frequencies, colors, and other properties.

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

class HodField(SephirothField):
    """
    Represents the Hod Sephiroth field - the embodiment of glory, splendor,
    intellect, logic, and communication in the soul development framework.
    """

    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Hod - Splendor", # More descriptive default name
                 dimensions: Tuple[float, float, float] = (75.0, 75.0, 75.0),
                 base_frequency: float = 417.0, # Hod frequency (Solfeggio RE - Change/Communication)
                 resonance: float = 0.85,
                 stability: float = 0.90, # More stable/structured than Netzach
                 coherence: float = 0.88,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Hod field with Hod-specific properties.

        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Hod-specific properties
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
            sephiroth_name="hod"
        )

        # Initialize energy grid as None
        self._energy_grid = None

        # Set Hod-specific attributes
        self.divine_attribute = "glory/splendor"
        self.geometric_correspondence = "eight-pointed star" # Often associated with Mercury/8
        self.element = "water" # Intellectual water, sometimes air associated
        self.primary_color = "orange" # Color of Mercury, intellect

        # Hod-specific properties
        self.hod_properties = {
            'intellect_level': 0.96,
            'communication_clarity': 0.94,
            'rational_structure': 0.90,
            'truth_resonance': 0.88, # Resonance with objective truth
            'mercury_influence': 0.85 # Planetary correspondence
        }

        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'hod_properties': self.hod_properties
        })

        # Initialize Hod-specific aspects
        self._initialize_hod_aspects()

        logger.info(f"Hod Field '{self.name}' initialized with {len(self.aspects)} aspects")

    def _initialize_hod_aspects(self) -> None:
        """
        Initialize Hod-specific aspects.

        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Hod-specific aspects
            aspects_data = {
                'divine_glory': {
                    'frequency': 417.0,  # Hod base frequency
                    'color': 'bright orange',
                    'element': 'water/fire', # Glory has a Fire aspect
                    'keywords': ['glory', 'splendor', 'majesty', 'radiance', 'honor'],
                    'description': 'The aspect of divine glory and intellectual splendor'
                },
                'intellect_and_reason': {
                    'frequency': 852.0, # Solfeggio LA - Intuition/Order (Higher intellect)
                    'color': 'orange/yellow',
                    'element': 'air/water', # Intellect related to Air
                    'keywords': ['intellect', 'reason', 'logic', 'analysis', 'mind', 'thought'],
                    'description': 'The aspect of divine intellect and rational thought'
                },
                'communication_and_language': {
                    'frequency': 639.0, # Solfeggio FA - Connection/Relationships
                    'color': 'orange',
                    'element': 'air', # Communication as Air
                    'keywords': ['communication', 'language', 'expression', 'speech', 'writing', 'symbols'],
                    'description': 'The aspect of clear communication and expression'
                },
                'truth_and_honesty': {
                    'frequency': 528.0, # Frequency associated with truth/integrity/transformation
                    'color': 'clear orange',
                    'element': 'air',
                    'keywords': ['truth', 'honesty', 'veracity', 'sincerity', 'objectivity'],
                    'description': 'The aspect of seeking and communicating objective truth'
                },
                'structured_thought': {
                    'frequency': 417.0, # Base frequency related to structure
                    'color': 'deep orange',
                    'element': 'earth/water', # Structure has Earth quality
                    'keywords': ['structure', 'order', 'logic', 'form', 'pattern', 'system'],
                    'description': 'The aspect of structured and ordered thought processes'
                },
                'mercury_influence': {
                    'frequency': 141.27, # Mercury frequency
                    'color': 'orange/silver',
                    'element': 'air',
                    'keywords': ['intellect', 'communication', 'mind', 'reason', 'speed', 'logic'],
                    'description': 'The planetary aspect of Mercury'
                }
            }

            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Hod principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'glory', 'intellect', 'communication', 'truth', 'reason', 'structure', 'logic', 'mind'}
                )) / max(1, len(data['keywords']))

                # Frequency alignment with Hod's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)

                # Element alignment (Water/Air primary)
                element_alignment = 1.0 if data['element'] in ['water', 'air'] else 0.5

                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.6, strength)) # Ensure reasonable minimum strength for Hod aspects

                # Add aspect (will raise ValueError if name exists or strength invalid)
                self.add_aspect(name, strength, data)

        except (ValueError, TypeError) as e:
             # Catch errors from add_aspect
             error_msg = f"Failed to add Hod aspect: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Hod aspects: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Hod-specific patterns to the energy grid. Reflects intellect and structure.

        Raises:
            RuntimeError: If energy grid or required contribution arrays are not initialized.
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Hod patterns")
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

        # Hod: Glory, Intellect, Communication, Structure, Water/Air, Mercury
        # Patterns should be structured, intricate, perhaps network-like or crystalline.

        # 1. Intellect Pattern (Complex, interwoven geometric structure)
        # Example: Higher-order sinusoidal interference pattern
        freq1, freq2, freq3 = 8, 12, 16 # Related to 8-pointed star?
        intellect_pattern = 0.5 + 0.5 * (np.sin(freq1 * np.pi * x_norm) * np.cos(freq2 * np.pi * y_norm) +
                                         np.sin(freq2 * np.pi * y_norm) * np.cos(freq3 * np.pi * z_norm) +
                                         np.sin(freq3 * np.pi * z_norm) * np.cos(freq1 * np.pi * x_norm)) / 3.0

        # 2. Communication Pattern (Structured network pathways)
        # Simulate defined pathways, less random than Netzach's flow
        path_freq = 6
        comm_pattern_x = 0.5 + 0.5 * np.cos(path_freq * np.pi * y_norm)**2 * np.cos(path_freq * np.pi * z_norm)**2
        comm_pattern_y = 0.5 + 0.5 * np.cos(path_freq * np.pi * x_norm)**2 * np.cos(path_freq * np.pi * z_norm)**2
        comm_pattern_z = 0.5 + 0.5 * np.cos(path_freq * np.pi * x_norm)**2 * np.cos(path_freq * np.pi * y_norm)**2
        communication_pattern = (comm_pattern_x + comm_pattern_y + comm_pattern_z) / 3.0

        # 3. Rational Structure (Geometric order, 8-pointed star influence)
        angle_xy = np.arctan2(y_norm - 0.5, x_norm - 0.5)
        # Modulate based on Z for 3D structure
        z_mod = np.cos(np.pi * z_norm)**2
        structure_pattern = 0.5 + 0.5 * np.cos(8 * angle_xy) * z_mod # Eight-fold symmetry in XY modulated by Z

        # 4. Truth Resonance (Clear, stable regions, low noise)
        # Create stable core region
        center = np.array([0.5, 0.5, 0.5])
        point_coords = np.stack(norm_coords, axis=-1)
        distance_sq = np.sum((point_coords - center)**2, axis=-1)
        truth_pattern = 0.7 + 0.3 * np.exp(-distance_sq * 10) # Stable core

        # 5. Mercury Influence (Fast, precise oscillations/patterns)
        mercury_freq_scaled = 141.27 / 8 # Scaled for grid
        # Fast oscillation based on coordinates
        mercury_pattern = 0.5 + 0.5 * np.sin(mercury_freq_scaled * np.pi * (4*x_norm + 3*y_norm - 2*z_norm))

        # Combine patterns reflecting Hod's intellectual and structured nature
        self._sephiroth_contribution = (
            0.30 * intellect_pattern +      # Core intellect/complexity
            0.25 * communication_pattern + # Network/Communication channels
            0.20 * structure_pattern +     # Rational structure/order
            0.10 * truth_pattern +         # Clarity/Truth core
            0.15 * mercury_pattern         # Planetary influence
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
        logger.debug(f"Applied Hod-specific patterns to energy grid")

    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Hod field's energy grid with specific patterns.

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
            logger.warning(f"Composition verification failed during Hod initialization: {e}")
        return self._energy_grid

    def analyze_information(self, entity_id: str, information_complexity: float) -> float:
        """
        Simulates the analysis of information by an entity within the Hod field.

        Args:
            entity_id: ID of the entity analyzing information.
            information_complexity: Complexity of the information (0.0-1.0).

        Returns:
            Resulting clarity of understanding achieved (0.0-1.0).

        Raises:
            ValueError: If parameters are invalid or entity not found in field.
            TypeError: If argument types are incorrect.
            RuntimeError: If operation fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(entity_id, str) or not entity_id:
            raise TypeError("entity_id must be a non-empty string.")
        if not isinstance(information_complexity, (int, float)) or not (0.0 <= information_complexity <= 1.0):
            raise ValueError(f"Information complexity must be between 0.0 and 1.0, got {information_complexity}")

        entity_ref = None
        for entity in self.entities:
             if isinstance(entity, dict) and entity.get('id') == entity_id:
                  entity_ref = entity
                  break
        if entity_ref is None:
            raise ValueError(f"Entity '{entity_id}' not found in Hod field {self.field_id}")

        try:
            # Calculate analysis capability based on Hod properties
            intellect_level = self.hod_properties.get('intellect_level', 0.9) # Default if missing
            rational_structure = self.hod_properties.get('rational_structure', 0.8)
            communication_clarity = self.hod_properties.get('communication_clarity', 0.9)

            # Entity's analysis potential enhanced by field's intellect, structure, and coherence
            analysis_potential = intellect_level * rational_structure * self.coherence

            # Clarity depends on potential vs complexity, boosted by communication clarity
            clarity = analysis_potential * max(0, (1.0 - information_complexity * 1.1)) # Complexity penalty
            clarity *= communication_clarity # Boosted by field clarity
            clarity = min(1.0, max(0.0, clarity)) # Clamp result [0, 1]

            # Record the event
            analysis_event = {
                'entity_id': entity_id,
                'info_complexity': information_complexity,
                'analysis_potential': analysis_potential,
                'result_clarity': clarity,
                'timestamp': datetime.now().isoformat()
            }

            # Add event to entity's record if structure allows
            if 'hod_analyses' not in entity_ref: entity_ref['hod_analyses'] = []
            if isinstance(entity_ref['hod_analyses'], list):
                 entity_ref['hod_analyses'].append(analysis_event)
            else: logger.warning(f"Entity {entity_id} 'hod_analyses' is not list.")

            # Optionally update entity's understanding attribute
            current_understanding = entity_ref.get('understanding', 0.5) # Assume entity state
            if isinstance(current_understanding, (int, float)):
                 entity_ref['understanding'] = min(1.0, current_understanding + clarity * 0.1) # Example update scale

            logger.info(f"Entity {entity_id} analyzed information (Complexity: {information_complexity:.2f}). Resulting clarity: {clarity:.3f}")
            return float(clarity)

        except Exception as e:
             error_msg = f"Error during information analysis for entity {entity_id}: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e

    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Hod field's current state.

        Returns:
            Dictionary of Hod field metrics.
        """
        base_metrics = super().get_field_metrics()

        hod_metrics = {
            'hod_properties': self.hod_properties.copy(), # Return a copy
            'realized_intellectual_capacity': self.hod_properties.get('intellect_level', 0) * self.coherence,
            'current_communication_clarity': self.hod_properties.get('communication_clarity', 0) * self.stability,
            'rational_structure_level': self.hod_properties.get('rational_structure', 0) * self.stability,
            'truth_resonance_level': self.hod_properties.get('truth_resonance', 0) * self.resonance,
            'realized_mercury_influence': self.hod_properties.get('mercury_influence', 0)
        }

        # Aggregate analysis statistics
        total_analyses = 0
        total_clarity_achieved = 0.0
        analyzing_entities = 0
        for entity in self.entities:
             analyses = entity.get('hod_analyses', [])
             if isinstance(analyses, list) and analyses:
                 analyzing_entities += 1
                 total_analyses += len(analyses)
                 total_clarity_achieved += sum(a.get('result_clarity', 0.0) for a in analyses)

        hod_metrics['total_analysis_events'] = total_analyses
        hod_metrics['analyzing_entity_count'] = analyzing_entities
        hod_metrics['average_analysis_clarity'] = total_clarity_achieved / total_analyses if total_analyses > 0 else 0.0

        combined_metrics = {
            **base_metrics,
            'hod_specific': hod_metrics,
        }
        return combined_metrics

    def __str__(self) -> str:
        """String representation of the Hod field."""
        intellect = self.hod_properties.get('intellect_level', 0.0)
        return f"HodField(name={self.name}, aspects={len(self.aspects)}, intellect={intellect:.2f})"

    def __repr__(self) -> str:
        """Detailed representation."""
        comm_clarity = self.hod_properties.get('communication_clarity', 0.0)
        return f"<HodField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} communication={comm_clarity:.2f}>"

# --- END OF FILE hod_field.py ---