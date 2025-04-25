# --- START OF FILE binah_field.py ---

"""
Binah Field Module

Defines the BinahField class which represents the specific Sephiroth field for Binah.
Implements Binah-specific aspects, frequencies, colors, and other properties.

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

class BinahField(SephirothField):
    """
    Represents the Binah Sephiroth field - the embodiment of understanding, structure,
    form, and the divine feminine principle in the soul development framework.
    """

    def __init__(self,
                 field_id: Optional[str] = None,
                 name: str = "Binah - Understanding", # More descriptive default name
                 dimensions: Tuple[float, float, float] = (90.0, 90.0, 90.0),
                 base_frequency: float = 852.0,  # Binah frequency (Solfeggio LA)
                 resonance: float = 0.90,
                 stability: float = 0.94, # Binah provides structure
                 coherence: float = 0.92,
                 base_field_ratio: float = 0.4,
                 sephiroth_ratio: float = 0.5,
                 random_ratio: float = 0.1):
        """
        Initialize the Binah field with Binah-specific properties.

        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            base_field_ratio: Proportion of base field properties
            sephiroth_ratio: Proportion of Binah-specific properties
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
            sephiroth_name="binah"
        )

        # Set Binah-specific attributes
        self.divine_attribute = "understanding"
        self.geometric_correspondence = "triangle" # Often associated with Triangle or Yoni
        self.element = "water" # Primordial waters, sometimes earth
        self.primary_color = "black" # Color of potential, formlessness

        # Binah-specific properties
        self.binah_properties = {
            'understanding_level': 0.97,
            'structuring_potential': 0.95,
            'feminine_principle': 0.92,
            'form_manifestation': 0.90,
            'receptivity_factor': 0.88,
            'saturn_resonance': 0.85 # Planetary correspondence
        }

        # Add to field properties
        self.field_properties.update({
            'divine_attribute': self.divine_attribute,
            'geometric_correspondence': self.geometric_correspondence,
            'element': self.element,
            'primary_color': self.primary_color,
            'binah_properties': self.binah_properties
        })

        # Initialize Binah-specific aspects
        self._initialize_binah_aspects()

        logger.info(f"Binah Field '{self.name}' initialized with {len(self.aspects)} aspects")

    def _initialize_binah_aspects(self) -> None:
        """
        Initialize Binah-specific aspects.

        Raises:
            RuntimeError: If aspect initialization fails
        """
        try:
            # Define Binah-specific aspects based on correspondences
            # (Ensure these keys match expected data structure for aspects)
            aspects_data = {
                'divine_understanding': {
                    'frequency': 852.0,  # Binah base frequency
                    'color': 'deep black',
                    'element': 'water',
                    'keywords': ['understanding', 'intelligence', 'reason', 'perception', 'intuition'],
                    'description': 'The aspect of profound divine understanding and intelligence'
                },
                'feminine_principle': {
                    'frequency': 741.0, # Solfeggio SOL
                    'color': 'silver-black',
                    'element': 'water',
                    'keywords': ['feminine', 'receptive', 'form', 'yin', 'passivity'],
                    'description': 'The aspect of the divine feminine principle and receptivity'
                },
                'form_and_structure': {
                    'frequency': 396.0, # Solfeggio UT (Root/Structure)
                    'color': 'black',
                    'element': 'earth/water',
                    'keywords': ['form', 'structure', 'limitation', 'order', 'containment', 'template'],
                    'description': 'The aspect of giving form and structure to energy'
                },
                'divine_mother': {
                    'frequency': 528.0, # Solfeggio MI (Love)
                    'color': 'dark blue/black',
                    'element': 'water',
                    'keywords': ['mother', 'maternal', 'nurturing', 'birth', 'creation'],
                    'description': 'The aspect of the divine mother principle'
                },
                'great_sea': {
                    'frequency': 417.0, # Solfeggio RE (Change/Potential)
                    'color': 'black',
                    'element': 'water',
                    'keywords': ['potential', 'sea', 'primordial', 'unmanifest', 'depth'],
                    'description': 'The aspect of the primordial sea of potential'
                },
                'saturn_influence': {
                    'frequency': 147.85, # Saturn frequency
                    'color': 'black/lead',
                    'element': 'earth',
                    'keywords': ['structure', 'discipline', 'time', 'karma', 'limitation'],
                    'description': 'The planetary aspect of Saturn'
                },
                 'sanctification': {
                    'frequency': 963.0, # Solfeggio SI (Divine Order)
                    'color': 'silver-black',
                    'element': 'water/aether',
                    'keywords': ['sacred', 'holy', 'sanctuary', 'consecration', 'purity'],
                    'description': 'The aspect of setting apart as sacred'
                }
            }

            # Add each aspect to the field
            for name, data in aspects_data.items():
                # Calculate strength based on alignment with Binah principles
                keywords_alignment = len(set(data['keywords']).intersection(
                    {'understanding', 'form', 'structure', 'receptive', 'feminine', 'mother', 'potential', 'discipline'}
                )) / max(1, len(data['keywords']))

                # Frequency alignment with Binah's base frequency
                freq_alignment = 1.0 - min(1.0, abs(data['frequency'] - self.base_frequency) / self.base_frequency)

                # Element alignment (more lenient as Binah relates to both water/earth)
                element_alignment = 1.0 if data['element'] == self.element else 0.6 if data['element'] == 'earth' else 0.3

                # Calculate overall strength
                strength = (0.5 * keywords_alignment + 0.3 * freq_alignment + 0.2 * element_alignment)
                strength = min(1.0, max(0.5, strength))  # Ensure minimum strength of 0.5

                # Add aspect (will raise ValueError if name exists or strength invalid)
                self.add_aspect(name, strength, data)

        except (ValueError, TypeError) as e:
             # Catch errors from add_aspect
             error_msg = f"Failed to add Binah aspect: {str(e)}"
             logger.error(error_msg, exc_info=True)
             raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Binah aspects: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def apply_sephiroth_specific_patterns(self) -> None:
        """
        Apply Binah-specific patterns to the energy grid.

        Raises:
            RuntimeError: If energy grid or required contribution arrays are not initialized.
        """
        if self._energy_grid is None:
            raise RuntimeError("Energy grid must be initialized before applying Binah patterns")
        if not hasattr(self, '_base_contribution') or self._base_contribution is None:
            raise RuntimeError("Base contribution array (_base_contribution) not initialized.")
        if not hasattr(self, '_random_contribution') or self._random_contribution is None:
            raise RuntimeError("Random contribution array (_random_contribution) not initialized.")

        grid_shape = self._energy_grid.shape
        if any(s <= 0 for s in grid_shape):
             raise RuntimeError(f"Energy grid has invalid shape: {grid_shape}")

        # Create grid indices safely
        indices = np.indices(grid_shape)
        # Normalize coordinates: divide by (shape - 1) or 1 if shape is 1, to get range [0, 1]
        norm_coords = [indices[i] / max(1, grid_shape[i] - 1) for i in range(3)]
        x_norm, y_norm, z_norm = norm_coords

        # Calculate distance from center
        center = np.array([0.5, 0.5, 0.5])
        point_coords = np.stack(norm_coords, axis=-1)
        distance = np.linalg.norm(point_coords - center, axis=-1)
        # Normalize distance to approx [0, 1] range (max distance is sqrt(3)/2 from center)
        distance_norm = distance / (np.sqrt(3) / 2)
        distance_norm = np.clip(distance_norm, 0.0, 1.0) # Ensure within [0, 1]

        # Binah: Understanding, Structure, Form, Receptivity, Containment
        # Create patterns reflecting these properties.

        # 1. Structuring Grid pattern (lattice-like, stable)
        grid_freq = 6 # Frequency of grid lines
        grid_pattern = 0.5 + 0.5 * (np.cos(grid_freq * np.pi * x_norm) *
                                    np.cos(grid_freq * np.pi * y_norm) *
                                    np.cos(grid_freq * np.pi * z_norm))

        # 2. Containment pattern (higher energy density near boundaries)
        # Calculate distance to nearest edge for each dimension
        edge_dist_x = np.minimum(x_norm, 1.0 - x_norm)
        edge_dist_y = np.minimum(y_norm, 1.0 - y_norm)
        edge_dist_z = np.minimum(z_norm, 1.0 - z_norm)
        min_edge_dist = np.minimum(np.minimum(edge_dist_x, edge_dist_y), edge_dist_z)
        # Exponential increase towards edge (min_edge_dist approaches 0)
        containment_pattern = 0.3 + 0.7 * np.exp(-min_edge_dist * 10)

        # 3. Receptivity pattern (inward focus, stable core)
        # Use distance from center, stronger near center
        receptivity_pattern = 0.4 + 0.6 * np.exp(-distance_norm**2 * 4) # Gaussian focus

        # 4. Saturn influence (slow, deep, structured resonance)
        saturn_freq_scaled = 147.85 / 20 # Scaled planet frequency for grid
        saturn_pattern = 0.5 + 0.5 * np.sin(saturn_freq_scaled * x_norm * 2 * np.pi) * \
                               np.sin(saturn_freq_scaled * y_norm * 2 * np.pi + np.pi/2) * \
                               np.sin(saturn_freq_scaled * z_norm * 2 * np.pi + np.pi) # Phased

        # 5. Water/Form pattern (deep, stable potential energy, subtle flow)
        # Combine stable base with slow undulation
        water_base = 0.6 + 0.4 * np.tanh(5 * (distance_norm - 0.5)) # Stable core
        water_flow = 0.1 * np.sin(np.pi * (x_norm + 2*y_norm + 3*z_norm)) # Slow 3D flow
        water_pattern = water_base + water_flow

        # Combine patterns with weights reflecting Binah's attributes
        self._sephiroth_contribution = (
            0.30 * grid_pattern +         # Structure (significant)
            0.25 * containment_pattern +  # Form/Limitation
            0.15 * receptivity_pattern +  # Receptive core
            0.15 * saturn_pattern +       # Planetary influence
            0.15 * water_pattern          # Elemental/Potential
        )

        # Ensure values are within the valid range [0, 1]
        self._sephiroth_contribution = np.clip(self._sephiroth_contribution, 0.0, 1.0)

        # Update the final energy grid using the calculated contribution
        self._energy_grid = (
            self.base_field_ratio * self._base_contribution +
            self.sephiroth_ratio * self._sephiroth_contribution +
            self.random_ratio * self._random_contribution
        )

        # Final clipping to ensure grid values are strictly within [0, 1]
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0)

        logger.debug(f"Applied Binah-specific patterns to energy grid")

    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the Binah field's energy grid with specific patterns.

        Args:
            resolution: Number of points along each dimension (must be positive ints).

        Returns:
            The initialized energy grid (numpy array).

        Raises:
            ValueError: If resolution is invalid.
            RuntimeError: If initialization or pattern application fails.
        """
        # Let superclass handle initial grid setup and contribution array creation
        super().initialize_energy_grid(resolution)

        # Apply Binah-specific patterns onto the initialized structure
        self.apply_sephiroth_specific_patterns()

        # Optional: Verify composition ratios after applying patterns
        try:
            self.verify_composition()
        except RuntimeError as e:
            # Log warning but don't fail initialization
            logger.warning(f"Composition verification failed during Binah initialization: {e}")

        return self._energy_grid

    def apply_form_constraints(self, entity_id: str, constraint_factor: float) -> float:
        """
        Apply Binah's structuring energy to impose form constraints on an entity.

        Args:
            entity_id: ID of entity to affect.
            constraint_factor: Factor determining the strength of the constraint (0.0-1.0).

        Returns:
            Actual structuring force applied (0.0-1.0).

        Raises:
            ValueError: If parameters are invalid or entity not found in field.
            TypeError: If argument types are incorrect.
            RuntimeError: If operation fails unexpectedly.
        """
        # --- Input Validation ---
        if not isinstance(entity_id, str) or not entity_id:
            raise TypeError("entity_id must be a non-empty string.")
        if not isinstance(constraint_factor, (int, float)) or not (0.0 <= constraint_factor <= 1.0):
            raise ValueError(f"Constraint factor must be a number between 0.0 and 1.0, got {constraint_factor}")

        entity_ref = None
        for entity in self.entities:
             # Check if entity is a dict and has 'id' key before accessing
             if isinstance(entity, dict) and entity.get('id') == entity_id:
                  entity_ref = entity
                  break

        if entity_ref is None:
            raise ValueError(f"Entity '{entity_id}' not found in Binah field {self.field_id}")

        try:
            # Calculate structuring effect based on Binah properties
            base_structuring = self.binah_properties.get('structuring_potential', 0.8) # Default if missing
            saturn_influence = self.binah_properties.get('saturn_resonance', 0.7)

            # Calculate final structuring force, influenced by field state
            structuring_force = base_structuring * constraint_factor * saturn_influence * self.stability
            structuring_force = min(1.0, max(0.0, structuring_force)) # Clamp result

            # Record the constraint application event
            constraint_event = {
                'entity_id': entity_id,
                'constraint_factor': constraint_factor,
                'structuring_force': structuring_force,
                'timestamp': datetime.now().isoformat()
            }

            # Add event to entity's record if structure allows
            if 'binah_constraints' not in entity_ref:
                 entity_ref['binah_constraints'] = []
            if isinstance(entity_ref['binah_constraints'], list):
                 entity_ref['binah_constraints'].append(constraint_event)
            else:
                 logger.warning(f"Entity {entity_id} 'binah_constraints' attribute is not a list. Cannot record event.")

            # Potential side-effect: Modify entity state based on force (e.g., reduce mobility)
            # This depends heavily on the entity's model and is omitted here for simplicity.
            # Example: entity_ref['mobility'] = entity_ref.get('mobility', 1.0) * (1.0 - structuring_force * 0.1)

            logger.info(f"Applied Binah structuring constraint (Force: {structuring_force:.3f}) to entity {entity_id}")
            return float(structuring_force)

        except Exception as e:
            error_msg = f"Error applying form constraints to entity {entity_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the Binah field's current state.

        Returns:
            Dictionary of Binah field metrics.
        """
        # Get base metrics from parent class
        base_metrics = super().get_field_metrics()

        # Calculate Binah-specific derived metrics
        binah_metrics = {
            'binah_properties': self.binah_properties.copy(), # Return a copy
            'realized_structuring_potential': self.binah_properties.get('structuring_potential', 0) * self.stability,
            'realized_receptivity_factor': self.binah_properties.get('receptivity_factor', 0) * self.resonance,
            'realized_saturn_influence': self.binah_properties.get('saturn_resonance', 0) * self.coherence
        }

        # Calculate aggregate metrics from entity interactions if tracked
        total_constraints = 0
        total_force_applied = 0.0
        constrained_entities = 0
        for entity in self.entities:
             constraints = entity.get('binah_constraints', [])
             if isinstance(constraints, list) and constraints:
                 constrained_entities += 1
                 total_constraints += len(constraints)
                 total_force_applied += sum(c.get('structuring_force', 0.0) for c in constraints)

        binah_metrics['total_constraint_events'] = total_constraints
        binah_metrics['constrained_entity_count'] = constrained_entities
        binah_metrics['average_constraint_force'] = total_force_applied / total_constraints if total_constraints > 0 else 0.0

        # Combine base metrics with Binah-specific metrics
        combined_metrics = {
            **base_metrics,
            'binah_specific': binah_metrics
            # Removed 'entity_constraint_stats' for individual entities from top-level metrics
            # It could be retrieved by querying individual entities if needed.
        }

        return combined_metrics

    def __str__(self) -> str:
        """String representation of the Binah field."""
        understanding = self.binah_properties.get('understanding_level', 0.0)
        return f"BinahField(name={self.name}, aspects={len(self.aspects)}, understanding={understanding:.2f})"

    def __repr__(self) -> str:
        """Detailed representation."""
        structure = self.binah_properties.get('structuring_potential', 0.0)
        return f"<BinahField id='{self.field_id}' name='{self.name}' aspects={len(self.aspects)} structure={structure:.2f}>"
    # --- END OF FILE binah_field.py ---