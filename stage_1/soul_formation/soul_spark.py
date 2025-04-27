# --- START OF FILE src/stage_1/soul_formation/soul_spark.py ---

"""
Soul Spark Module (Refactored for New Field System)

Defines the SoulSpark class, representing the evolving soul entity.
Includes position, current field context, and energy attributes.
Holds the core attributes, state, and methods for visualization
and data management as the soul progresses through formation stages.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    # Import necessary constants
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. SoulSpark using fallback defaults.")
    # Define crucial fallbacks if constants unavailable
    SOUL_SPARK_DEFAULT_FREQ = 432.0
    SOUL_SPARK_DEFAULT_STABILITY = 0.6
    SOUL_SPARK_DEFAULT_RESONANCE = 0.6
    SOUL_SPARK_DEFAULT_ALIGNMENT = 0.1
    SOUL_SPARK_DEFAULT_COHERENCE = 0.6
    SOUL_SPARK_DEFAULT_PHI_RESONANCE = 0.5
    SOUL_SPARK_DEFAULT_ENERGY = 1.0 # Default energy level
    FLOAT_EPSILON = 1e-9
    PI = np.pi
    GOLDEN_RATIO = 1.61803398875
    MAX_AMPLITUDE = 0.95
    # Visualization fallbacks if needed by methods
    SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT = 'grey'
    SOUL_SPARK_VIZ_FREQ_SIG_MARKER_FMT = 'bo'
    SOUL_SPARK_VIZ_FREQ_SIG_BASE_FMT = 'r-'
    SOUL_SPARK_VIZ_FREQ_SIG_STEM_LW = 1.5
    SOUL_SPARK_VIZ_FREQ_SIG_MARKER_SZ = 5.0
    SOUL_SPARK_VIZ_FREQ_SIG_XLABEL = 'Frequency (Hz)'
    SOUL_SPARK_VIZ_FREQ_SIG_YLABEL = 'Amplitude'
    SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR = 'red'

# Conditional import for visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    logger.warning("Matplotlib not found. SoulSpark visualization methods will be disabled.")


class SoulSpark:
    """
    Represents the evolving soul entity throughout its formation process.
    Holds attributes modified by various stage functions.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None, spark_id: Optional[str] = None):
        """
        Initialize a new SoulSpark instance.

        Args:
            initial_data (Optional[Dict[str, Any]]): Data dictionary from Void/Guff stage
                                                     to initialize the spark. If None, uses defaults.
            spark_id (Optional[str]): Specify an ID, otherwise a new one is generated.
        """
        self.spark_id: str = spark_id or str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.last_modified: str = self.creation_time

        # --- Initialize Attributes ---
        data = initial_data or {}

        # Core Metrics / State Flags
        self.stability: float = float(data.get('stability', SOUL_SPARK_DEFAULT_STABILITY))
        self.resonance: float = float(data.get('resonance', SOUL_SPARK_DEFAULT_RESONANCE)) # General resonance
        self.coherence: float = float(data.get('coherence', SOUL_SPARK_DEFAULT_COHERENCE))
        self.creator_alignment: float = float(data.get('creator_alignment', SOUL_SPARK_DEFAULT_ALIGNMENT))
        # NEW: Base energy level of the spark itself
        self.energy: float = float(data.get('energy', SOUL_SPARK_DEFAULT_ENERGY))

        # Stage Completion Flags
        self.guff_strengthened: bool = bool(data.get('guff_strengthened', False)) # NEW flag
        self.sephiroth_journey_complete: bool = bool(data.get('sephiroth_journey_complete', False)) # NEW flag (replaces formation_complete?)
        self.harmonically_strengthened: bool = bool(data.get('harmonically_strengthened', False))
        self.cord_formation_complete: bool = bool(data.get('cord_formation_complete', False))
        self.earth_harmonized: bool = bool(data.get('earth_harmonized', False))
        self.identity_crystallized: bool = bool(data.get('identity_crystallized', False))
        self.incarnated: bool = bool(data.get('incarnated', False))
        # Readiness Flags (set by preceding stages)
        self.ready_for_guff: bool = bool(data.get('ready_for_guff', True)) # Assume ready after spark creation
        self.ready_for_journey: bool = bool(data.get('ready_for_journey', False)) # Set by Guff Strengthening
        self.ready_for_entanglement: bool = bool(data.get('ready_for_entanglement', False)) # Set after Journey
        self.ready_for_completion: bool = bool(data.get('ready_for_completion', False)) # Set by Entanglement
        self.ready_for_strengthening: bool = bool(data.get('ready_for_strengthening', False)) # Set after Entanglement
        self.ready_for_life_cord: bool = bool(data.get('ready_for_life_cord', False)) # Set by Strengthening
        self.ready_for_earth: bool = bool(data.get('ready_for_earth', False)) # Set by Life Cord
        self.ready_for_identity: bool = bool(data.get('ready_for_identity', False)) # Set by Earth Harmony
        self.ready_for_birth: bool = bool(data.get('ready_for_birth', False)) # Set by Identity

        # Frequency & Harmonics
        self.frequency: float = float(data.get('frequency', SOUL_SPARK_DEFAULT_FREQ))
        # frequency_signature is now generated/updated by entanglement
        self.frequency_signature: Dict[str, Any] = data.get('frequency_signature', {})
        self.harmonics: List[float] = data.get('harmonics', []) # Simple list updated by stages
        self.phi_resonance: float = float(data.get('phi_resonance', SOUL_SPARK_DEFAULT_PHI_RESONANCE))
        self.pattern_coherence: float = float(data.get('pattern_coherence', 0.0)) # From Entanglement patterns
        self.harmony: float = float(data.get('harmony', 0.5)) # Related to coherence enhancement

        # Aspects & Qualities
        self.aspects: Dict[str, Dict[str, Any]] = data.get('aspects', {}) # {name: {strength, source, time, details...}}
        # creator_aspects removed - incorporated into aspects with source='Kether'/'Entanglement'
        # divine_qualities removed - represented within aspects

        # Entanglement & Life Cord
        self.creator_channel_id: Optional[str] = data.get('creator_channel_id')
        self.creator_connection_strength: float = float(data.get('creator_connection_strength', 0.0))
        self.resonance_patterns: Dict[str, Any] = data.get('resonance_patterns', {}) # From Entanglement
        self.life_cord: Optional[Dict[str, Any]] = data.get('life_cord')
        self.cord_integrity: float = float(data.get('cord_integrity', 0.0))
        # cord_integration removed (now part of life_cord dict?) -> Keep for simplicity? Let's remove.
        # Field Integration (NEW - Set during life cord stage)
        self.field_integration: float = float(data.get('field_integration', 0.0)) # How well cord integrates with soul's field

        # Earth Harmonization Attributes
        self.earth_resonance: float = float(data.get('earth_resonance', 0.0))
        self.elements: Dict[str, float] = data.get('elements', {})
        self.earth_cycles: Dict[str, float] = data.get('earth_cycles', {})
        self.planetary_resonance: float = float(data.get('planetary_resonance', 0.0))
        self.gaia_connection: float = float(data.get('gaia_connection', 0.0))
        self.elemental_alignment: float = float(data.get('elemental_alignment', 0.0))
        self.cycle_synchronization: float = float(data.get('cycle_synchronization', 0.0))

        # Identity Attributes
        self.name: Optional[str] = data.get('name')
        self.gematria_value: int = int(data.get('gematria_value', 0))
        self.name_resonance: float = float(data.get('name_resonance', 0.0))
        self.voice_frequency: float = float(data.get('voice_frequency', 0.0))
        self.response_level: float = float(data.get('response_level', 0.0))
        self.heartbeat_entrainment: float = float(data.get('heartbeat_entrainment', 0.0))
        self.soul_color: Optional[str] = data.get('soul_color')
        self.color_frequency: float = float(data.get('color_frequency', 0.0))
        self.soul_frequency: float = float(data.get('soul_frequency', self.frequency))
        self.sephiroth_aspect: Optional[str] = data.get('sephiroth_aspect')
        self.elemental_affinity: Optional[str] = data.get('elemental_affinity')
        self.platonic_symbol: Optional[str] = data.get('platonic_symbol')
        self.yin_yang_balance: float = float(data.get('yin_yang_balance', 0.5))
        # Ensure emotional_resonance dict has standard keys
        default_emotions = {'love': 0.0, 'joy': 0.0, 'peace': 0.0, 'harmony': 0.0, 'compassion': 0.0}
        loaded_emotions = data.get('emotional_resonance', {})
        self.emotional_resonance: Dict[str, float] = {k: float(loaded_emotions.get(k, default_emotions[k])) for k in default_emotions}
        self.crystallization_level: float = float(data.get('crystallization_level', 0.0))
        self.attribute_coherence: float = float(data.get('attribute_coherence', 0.0))
        self.identity_metrics: Optional[Dict[str, Any]] = data.get('identity_metrics')
        self.sacred_geometry_imprint: Optional[str] = data.get('sacred_geometry_imprint')

        # Birth Related Attributes
        self.physical_integration: float = float(data.get('physical_integration', 0.0))
        self.birth_time: Optional[str] = data.get('birth_time')
        self.memory_veil: Optional[Dict[str, Any]] = data.get('memory_veil')
        self.memory_retention: float = float(data.get('memory_retention', 1.0))
        self.breath_pattern: Optional[Dict[str, Any]] = data.get('breath_pattern')
        self.physical_energy: float = float(data.get('physical_energy', 0.0)) # Energy bound to physical form
        self.spiritual_energy: float = float(data.get('spiritual_energy', 1.0)) # Remaining spiritual energy

        # Consciousness State
        self.consciousness_state: str = data.get('consciousness_state', 'spark') # Initial state
        self.consciousness_frequency: float = float(data.get('consciousness_frequency', self.frequency * 0.1)) # Derived from base freq
        self.state_stability: float = float(data.get('state_stability', self.stability))

        # Memory Echoes
        self.memory_echoes: List[str] = data.get('memory_echoes', [])

        # NEW: Field Interaction Attributes
        # Position uses float for smooth movement potential, converted to int for grid indexing
        self.position: List[float] = data.get('position', [GRID_SIZE[0]/2.0, GRID_SIZE[1]/2.0, GRID_SIZE[2]/2.0]) # Start in Void center
        self.current_field_key: str = data.get('current_field_key', 'void') # Initial field is Void

        # --- Post-Init ---
        # Ensure basic harmonics list exists if frequency is set
        if not self.harmonics and self.frequency > FLOAT_EPSILON:
            self._generate_basic_harmonics()

        # Validate loaded/default attributes
        self._validate_attributes()

        logger.info(f"SoulSpark initialized/loaded: ID={self.spark_id}, Field={self.current_field_key}, Freq={self.frequency:.2f}, Stab={self.stability:.3f}, Energy={self.energy:.3f}")

    def _generate_basic_harmonics(self, count: int = 7) -> None:
        """Generates a simple list of harmonic frequencies based on self.frequency."""
        if self.frequency > FLOAT_EPSILON:
            self.harmonics = [float(self.frequency * n) for n in range(1, count + 1)]
            logger.debug(f"Generated basic harmonics list for {self.spark_id}: {self.harmonics}")
        else:
            self.harmonics = []

    def generate_harmonic_structure(self) -> None:
        """
        Generate the detailed harmonic frequency signature.
        Uses self.frequency as base and populates self.frequency_signature.
        """
        if self.frequency <= FLOAT_EPSILON:
            logger.warning(f"Cannot generate harmonic structure for {self.spark_id}: base frequency is non-positive.")
            self.frequency_signature = {'base_frequency': self.frequency, 'frequencies': [], 'amplitudes': [], 'phases': [], 'num_frequencies': 0}
            return

        base_freq = self.frequency
        num_harmonics = HARMONIC_STRENGTHENING_HARMONIC_COUNT # Use constant for default count
        phi_count = int(num_harmonics * self.phi_resonance) # Number of phi-related harmonics

        frequencies = [base_freq]
        amplitudes = [1.0] # Fundamental amplitude

        # Integer Harmonics
        for i in range(2, num_harmonics - phi_count + 1):
            frequencies.append(base_freq * i)
            amplitudes.append(1.0 / (i**0.8)) # Slightly faster falloff

        # Phi Harmonics
        for i in range(1, phi_count + 1):
            frequencies.append(base_freq * (GOLDEN_RATIO ** i))
            amplitudes.append(0.8 / (GOLDEN_RATIO**(i*0.7))) # Different falloff for phi

        # Sort by frequency for clarity
        sorted_indices = np.argsort(frequencies)
        final_frequencies = np.array(frequencies)[sorted_indices]
        final_amplitudes = np.array(amplitudes)[sorted_indices]

        # Normalize amplitudes
        final_amplitudes /= np.max(final_amplitudes)

        # Random phases
        phases = np.random.uniform(0, 2 * PI, len(final_frequencies))

        self.frequency_signature = {
            'base_frequency': float(base_freq),
            'frequencies': final_frequencies.tolist(), # Store as list for JSON
            'amplitudes': final_amplitudes.tolist(), # Store as list for JSON
            'phases': phases.tolist(),               # Store as list for JSON
            'num_frequencies': len(final_frequencies)
        }
        logger.debug(f"Generated harmonic structure for soul {self.spark_id} with base {base_freq:.2f} Hz, {len(final_frequencies)} harmonics.")


    def get_spark_metrics(self) -> Dict[str, Any]:
        """ Get comprehensive metrics (Updated for new attributes). """
        logger.debug(f"Calculating metrics for soul {self.spark_id}...")
        metrics_data = {
            'core': {
                'stability': getattr(self, 'stability', 0.0),
                'resonance': getattr(self, 'resonance', 0.0),
                'coherence': getattr(self, 'coherence', 0.0),
                'creator_alignment': getattr(self, 'creator_alignment', 0.0),
                'energy': getattr(self, 'energy', 0.0), # Added energy
                'last_modified': getattr(self, 'last_modified', self.creation_time)
            },
            'position': {
                 'coords': getattr(self, 'position', [0.0, 0.0, 0.0]),
                 'current_field': getattr(self, 'current_field_key', 'unknown')
            },
            'frequency': {
                'base_frequency': self.frequency, # Use the direct attribute
                'num_harmonics': len(getattr(self, 'harmonics', [])), # Use simple list count
                'signature_base': self.frequency_signature.get('base_frequency', 0.0),
                'signature_count': self.frequency_signature.get('num_frequencies', 0),
                'soul_frequency': getattr(self, 'soul_frequency', 0.0),
                'voice_frequency': getattr(self, 'voice_frequency', 0.0),
                'color_frequency': getattr(self, 'color_frequency', 0.0),
                'phi_resonance': getattr(self, 'phi_resonance', 0.0),
                'harmony': getattr(self, 'harmony', 0.0)
            },
            'identity': { # Consolidated identity attributes
                'name': getattr(self, 'name', None),
                'gematria': getattr(self, 'gematria_value', 0),
                'name_resonance': getattr(self, 'name_resonance', 0.0),
                'response_level': getattr(self, 'response_level', 0.0),
                'soul_color': getattr(self, 'soul_color', None),
                'sephiroth_aspect': getattr(self, 'sephiroth_aspect', None),
                'elemental_affinity': getattr(self, 'elemental_affinity', None),
                'platonic_symbol': getattr(self, 'platonic_symbol', None),
                'yin_yang_balance': getattr(self, 'yin_yang_balance', 0.5),
                'crystallization_level': getattr(self, 'crystallization_level', 0.0),
                'attribute_coherence': getattr(self, 'attribute_coherence', 0.0),
                'identity_crystallized': getattr(self, 'identity_crystallized', False),
                'sacred_geometry_imprint': getattr(self, 'sacred_geometry_imprint', None)
            },
            'consciousness': { # Consolidated consciousness state
                 'state': getattr(self, 'consciousness_state', 'spark'),
                 'frequency': getattr(self, 'consciousness_frequency', 0.0),
                 'stability': getattr(self, 'state_stability', 0.0),
            },
            'entanglement': { # Consolidated entanglement
                'channel_id': getattr(self, 'creator_channel_id', None),
                'connection_strength': getattr(self, 'creator_connection_strength', 0.0),
                'pattern_coherence': getattr(self, 'pattern_coherence', 0.0),
                'resonance_pattern_count': len(getattr(self, 'resonance_patterns', {}))
            },
            'aspects': { # Summarize aspects
                 'count': len(getattr(self, 'aspects', {})),
                 'average_strength': float(np.mean([d.get('strength', 0.0) for d in getattr(self, 'aspects', {}).values()])) if getattr(self, 'aspects', {}) else 0.0
            },
            'life_cord': { # Consolidated life cord
                'formation_complete': getattr(self, 'cord_formation_complete', False),
                'integrity': getattr(self, 'cord_integrity', 0.0),
                'bandwidth': getattr(self, 'life_cord', {}).get('bandwidth', 0.0),
                'channels': getattr(self, 'life_cord', {}).get('channel_count', 0),
                'earth_connection': getattr(self, 'life_cord', {}).get('earth_connection', 0.0),
                'field_integration': getattr(self, 'field_integration', 0.0) # Added
            },
            'earth_harmony': { # Consolidated earth harmony
                 'harmonized': getattr(self, 'earth_harmonized', False),
                 'earth_resonance': getattr(self, 'earth_resonance', 0.0),
                 'elemental_alignment': getattr(self, 'elemental_alignment', 0.0),
                 'cycle_synchronization': getattr(self, 'cycle_synchronization', 0.0),
                 'planetary_resonance': getattr(self, 'planetary_resonance', 0.0),
                 'gaia_connection': getattr(self, 'gaia_connection', 0.0),
            },
            'incarnation': { # Consolidated incarnation
                'incarnated': getattr(self, 'incarnated', False),
                'birth_time': getattr(self, 'birth_time', None),
                'physical_integration': getattr(self, 'physical_integration', 0.0),
                'memory_retention': getattr(self, 'memory_retention', 1.0),
                'physical_energy': getattr(self, 'physical_energy', 0.0),
                'spiritual_energy': getattr(self, 'spiritual_energy', 0.0),
                'heartbeat_entrainment': getattr(self, 'heartbeat_entrainment', 0.0) # Moved here
            },
            'memory': { 'echo_count': len(getattr(self, 'memory_echoes', [])) },
            'stage_flags': { # Summary of progress flags
                'guff_strengthened': self.guff_strengthened,
                'sephiroth_journey_complete': self.sephiroth_journey_complete,
                'harmonically_strengthened': self.harmonically_strengthened,
                'cord_formation_complete': self.cord_formation_complete,
                'earth_harmonized': self.earth_harmonized,
                'identity_crystallized': self.identity_crystallized,
                'incarnated': self.incarnated
            }
        }
        logger.debug(f"Metrics calculated for soul {self.spark_id}")
        return metrics_data

    # --- add_memory_echo unchanged ---
    def add_memory_echo(self, event_description: str):
        """Adds a descriptive string as a memory echo."""
        if not hasattr(self, 'memory_echoes') or not isinstance(self.memory_echoes, list):
             self.memory_echoes = []
        timestamp = datetime.now().isoformat()
        self.memory_echoes.append(f"{timestamp}: {event_description}")
        logger.debug(f"Memory echo added to soul {self.spark_id}: '{event_description}'")

    # --- Visualization methods (visualize_spark, visualize_energy_signature) ---
    # These need updates to use the potentially changed/added attributes (e.g., energy, position).
    # Keep the structure but adapt the data they pull from 'self'.

    def visualize_spark(self, show: bool = True, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """ Create a 3D visualization (Updated). """
        if not VISUALIZATION_ENABLED:
            logger.warning("Visualization disabled: Matplotlib not found.")
            return None
        logger.info(f"Generating 3D visualization for soul {self.spark_id}...")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        try:
            # Use position attribute if available, default to 0,0,0
            center_x, center_y, center_z = getattr(self, 'position', [0.0, 0.0, 0.0])

            # Size based on energy and stability
            viz_radius = 0.3 + 0.7 * (self.energy * 0.6 + self.stability * 0.4)
            # Color based on soul_color or frequency
            color_name = getattr(self, 'soul_color', 'white').lower()
            viz_color = self._get_viz_color(color_name) # Use helper

            # Create sphere data
            u = np.linspace(0, 2 * PI, 30); v = np.linspace(0, PI, 15)
            x = viz_radius * np.outer(np.cos(u), np.sin(v)) + center_x
            y = viz_radius * np.outer(np.sin(u), np.sin(v)) + center_y
            z = viz_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z

            # Color map intensity based on coherence
            color_metric = self.coherence
            face_colors = plt.cm.get_cmap(viz_color)(color_metric * np.ones_like(x))
            face_colors[..., 3] = 0.4 + 0.5 * self.resonance # Alpha based on resonance

            ax.plot_surface(x, y, z, facecolors=face_colors, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.55) # Reduced alpha slightly

            # Add Aspects Representation (spiral outwards)
            aspects = getattr(self, 'aspects', {})
            num_aspects = len(aspects)
            if num_aspects > 0:
                aspect_radius_start = viz_radius * 1.1
                aspect_radius_end = viz_radius * (1.1 + 0.4 * self.attribute_coherence) # Coherence spreads aspects
                phi_angles = np.linspace(0, num_aspects * GOLDEN_RATIO * PI, num_aspects) # Angle spread
                z_spread = np.linspace(-viz_radius * 0.5, viz_radius * 0.5, num_aspects) # Vertical spread
                radii = np.linspace(aspect_radius_start, aspect_radius_end, num_aspects)
                aspect_colors = plt.cm.viridis(np.linspace(0, 1, num_aspects)) # Color by aspect order/type?
                ax_x = radii * np.cos(phi_angles) + center_x
                ax_y = radii * np.sin(phi_angles) + center_y
                ax_z = z_spread + center_z
                strengths = [d.get('strength', 0.1) for d in aspects.values()]
                sizes = 10 + 80 * np.array(strengths) # Size based on strength
                ax.scatter(ax_x, ax_y, ax_z, c=aspect_colors, s=sizes, alpha=0.7, label=f'{num_aspects} Aspects')

            # Add Creator Connection (as before)
            creator_strength = getattr(self, 'creator_connection_strength', 0.0)
            if creator_strength > 0.1:
                 line_end_z = center_z + viz_radius + 2.0 * creator_strength
                 ax.plot([center_x, center_x], [center_y, center_y], [center_z + viz_radius, line_end_z],
                         color='gold', linestyle='--', linewidth=2.5, alpha=0.7, label=f'Creator Conn ({creator_strength:.2f})')
                 ax.scatter([center_x], [center_y], [line_end_z], color='gold', s=60, marker='*')

            # Add Life Cord Anchor (as before)
            cord_complete = getattr(self, 'cord_formation_complete', False)
            if cord_complete:
                 cord_integrity = getattr(self, 'cord_integrity', 0.0)
                 line_end_z = center_z - viz_radius - 1.5 * cord_integrity
                 ax.plot([center_x, center_x], [center_y, center_y], [center_z - viz_radius, line_end_z],
                         color='saddlebrown', linestyle='-', linewidth=2.0, alpha=0.6, label=f'Life Cord ({cord_integrity:.2f})')
                 ax.scatter([center_x], [center_y], [line_end_z], color='saddlebrown', s=50, marker='v')

            # Plotting Setup (as before, potentially adjust auto-scaling limits)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            title = f"Soul Spark: {getattr(self, 'name', self.spark_id[:8])}\nS:{self.stability:.2f} C:{self.coherence:.2f} E:{self.energy:.2f} A:{self.creator_alignment:.2f}"
            ax.set_title(title)
            # Auto-scaling logic (ensure it handles potential empty points arrays)
            all_points_list = [[center_x, center_y, center_z]]
            if num_aspects > 0: all_points_list.append(np.column_stack([ax_x, ax_y, ax_z]))
            if creator_strength > 0.1: all_points_list.append([[center_x, center_y, center_z + viz_radius + 2.0 * creator_strength]])
            if cord_complete: all_points_list.append([[center_x, center_y, center_z - viz_radius - 1.5 * cord_integrity]])

            if len(all_points_list) > 1:
                 all_points = np.vstack(all_points_list)
                 if all_points.size > 0:
                      min_coords = np.min(all_points, axis=0)
                      max_coords = np.max(all_points, axis=0)
                      means = (min_coords + max_coords) / 2.0
                      ranges = max_coords - min_coords
                      max_range = max(max(ranges), viz_radius * 2.5) # Ensure minimum size based on radius
                      ax.set_xlim(means[0] - max_range / 2, means[0] + max_range / 2)
                      ax.set_ylim(means[1] - max_range / 2, means[1] + max_range / 2)
                      ax.set_zlim(means[2] - max_range / 2, means[2] + max_range / 2)
                 else: # Fallback if no points generated
                      ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            else:
                 ax.set_xlim(center_x-1, center_x+1); ax.set_ylim(center_y-1, center_y+1); ax.set_zlim(center_z-1, center_z+1)


            ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1.0)) # Adjust legend position
            plt.tight_layout()

            if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show: plt.show()
            else: plt.close(fig)
            return fig

        except Exception as e:
            logger.error(f"Error during SoulSpark 3D visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)
            return None

    def _get_viz_color(self, color_name: Optional[str]) -> str:
        """Helper to get a valid matplotlib color/cmap name."""
        if color_name is None: return 'plasma' # Default cmap
        color_name = color_name.lower()
        # Handle multi-colors like 'white/clear' -> 'white'
        if '/' in color_name: color_name = color_name.split('/')[0]
        # Handle specific known mappings
        color_map_viz = {
            'earth_tones': 'YlOrBr', 'gold': 'gold', 'silver': 'silver',
            'black': 'black', 'white': 'white', 'grey': 'grey',
            # Add more mappings if COLOR_SPECTRUM keys differ from matplotlib names
        }
        if color_name in color_map_viz: return color_map_viz[color_name]
        # Try direct name
        try:
            plt.cm.get_cmap(color_name)
            return color_name
        except ValueError:
            # Check if it's a basic color name
             if color_name in plt.colormaps(): # Check available colormaps first
                 return color_name
             elif color_name in plt.colormaps(): # check basic colors
                 return color_name
             else: return 'plasma' # Final fallback cmap


    def visualize_energy_signature(self, show: bool = True, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """ Visualize the energy signature (frequency spectrum - Updated). """
        if not VISUALIZATION_ENABLED:
            logger.warning("Visualization disabled: Matplotlib not found.")
            return None
        logger.info(f"Generating energy signature visualization for soul {self.spark_id}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            ax.set_title(f'Frequency Spectrum: {getattr(self, "name", self.spark_id[:8])}')
            # Use the detailed frequency_signature if available
            fsig = getattr(self, 'frequency_signature', {})
            freqs = np.array(fsig.get('frequencies', []))
            amps = np.array(fsig.get('amplitudes', []))
            base_freq = fsig.get('base_frequency', self.frequency) # Use signature base if possible

            if freqs.size > 0 and amps.size > 0 and freqs.size == amps.size:
                # Filter out very low amplitude harmonics for clarity
                mask = amps > 0.01 # Only show harmonics with > 1% amplitude
                plot_freqs = freqs[mask]
                plot_amps = amps[mask]

                if plot_freqs.size > 0:
                    markerline, stemlines, baseline = ax.stem(
                        plot_freqs, plot_amps,
                        linefmt=SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT,
                        markerfmt=SOUL_SPARK_VIZ_FREQ_SIG_MARKER_FMT,
                        basefmt=SOUL_SPARK_VIZ_FREQ_SIG_BASE_FMT,
                        label='Harmonics'
                    )
                    plt.setp(stemlines, 'linewidth', SOUL_SPARK_VIZ_FREQ_SIG_STEM_LW)
                    plt.setp(markerline, 'markersize', SOUL_SPARK_VIZ_FREQ_SIG_MARKER_SZ)

                    ax.set_xlabel(SOUL_SPARK_VIZ_FREQ_SIG_XLABEL)
                    ax.set_ylabel(SOUL_SPARK_VIZ_FREQ_SIG_YLABEL)
                    ax.set_ylim(bottom=0)
                    ax.grid(True, linestyle='--', alpha=0.6)

                    # Highlight base frequency
                    if base_freq > FLOAT_EPSILON:
                         ax.axvline(x=base_freq, color=SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR, linestyle='--', alpha=0.7, label=f'Base Freq ({base_freq:.1f} Hz)')

                    # Maybe highlight soul_frequency or voice_frequency if different?
                    soul_freq = getattr(self, 'soul_frequency', 0.0)
                    if soul_freq > FLOAT_EPSILON and abs(soul_freq - base_freq) > 5.0: # Show if different
                         ax.axvline(x=soul_freq, color='purple', linestyle=':', alpha=0.7, label=f'Soul Freq ({soul_freq:.1f} Hz)')

                    ax.legend()
                else:
                    ax.text(0.5, 0.5, "No significant frequencies found", ha='center', transform=ax.transAxes)
            else: # Fallback if no signature, just plot base frequency
                 if self.frequency > FLOAT_EPSILON:
                      ax.stem([self.frequency], [1.0], linefmt='grey', markerfmt='bo', basefmt='r-')
                      ax.text(0.5, 0.5, f"Base Frequency: {self.frequency:.1f} Hz\n(No detailed signature)", ha='center', va='center', transform=ax.transAxes)
                 else:
                      ax.text(0.5, 0.5, "No frequency data available", ha='center', transform=ax.transAxes)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            metrics_str = f"Stab: {self.stability:.2f} | Res: {self.resonance:.2f} | Coh: {self.coherence:.2f} | Align: {self.creator_alignment:.2f} | E: {self.energy:.2f}"
            fig.text(0.5, 0.02, metrics_str, ha='center', fontsize=10, bbox=dict(facecolor='whitesmoke', alpha=0.8))

            if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show: plt.show()
            else: plt.close(fig)
            return fig

        except Exception as e:
            logger.error(f"Error during energy signature visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)
            return None

    # --- State Management (save/load - check serialization carefully) ---
    def save_spark_data(self, file_path: str) -> bool:
        """ Saves spark data. Ensures numpy arrays are converted. """
        logger.info(f"Saving soul spark data for {self.spark_id} to {file_path}...")
        try:
            data_to_save = {}
            for attr, value in self.__dict__.items():
                 data_to_save[attr] = value

            def default_serializer(o):
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (datetime, uuid.UUID)): return str(o)
                # Add handling for specific complex types if needed
                # if isinstance(o, SomeComplexType): return o.to_dict()
                try: return json.JSONEncoder().default(o)
                except TypeError:
                    logger.warning(f"Attribute '{attr}' type {type(o)} unserializable. Representing as string.")
                    return str(o) # Fallback to string representation

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=default_serializer)
            logger.info("Soul spark data saved successfully.")
            return True
        except IOError as e: logger.error(f"IOError saving soul spark data: {e}"); raise IOError(f"Failed to write file: {e}") from e
        except TypeError as e: logger.error(f"TypeError saving soul spark data: {e}"); raise TypeError(f"Serialization failed: {e}") from e
        except Exception as e: logger.error(f"Unexpected error saving: {e}"); raise RuntimeError(f"Failed to save: {e}") from e

    @classmethod
    def load_from_file(cls, file_path: str) -> 'SoulSpark':
        """ Loads spark data. Converts lists back to numpy arrays where appropriate. """
        logger.info(f"Loading soul spark data from {file_path}...")
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(file_path, 'r') as f: loaded_data = json.load(f)
            if not isinstance(loaded_data, dict): raise ValueError("Invalid data format.")

            instance = cls(initial_data=None, spark_id=loaded_data.get('spark_id'))
            for attr, value in loaded_data.items():
                # Convert specific attributes back to numpy arrays if they were lists
                if attr == 'frequency_signature' and isinstance(value, dict):
                    for key in ['frequencies', 'amplitudes', 'phases']:
                        if key in value and isinstance(value[key], list):
                            value[key] = np.array(value[key])
                elif attr == 'harmonics' and isinstance(value, list):
                     value = np.array(value) # Convert simple harmonics list too if needed elsewhere
                # Add more conversions if other attributes use numpy arrays
                setattr(instance, attr, value)

            instance._validate_attributes() # Validate after loading
            # Ensure frequency signature counts match array lengths
            fsig = getattr(instance, 'frequency_signature', {})
            if isinstance(fsig, dict):
                for key in ['frequencies', 'amplitudes', 'phases']:
                    if key in fsig and isinstance(fsig[key], np.ndarray):
                        pass # Already numpy array
                    elif key in fsig and isinstance(fsig[key], list):
                        fsig[key] = np.array(fsig[key]) # Convert if still list
                num_freqs = len(fsig.get('frequencies', []))
                fsig['num_frequencies'] = num_freqs
                if len(fsig.get('amplitudes',[])) != num_freqs or len(fsig.get('phases',[])) != num_freqs:
                    logger.warning(f"Freq signature mismatch on load for {instance.spark_id}. Regenerating.")
                    instance.generate_harmonic_structure() # Regenerate if inconsistent

            logger.info(f"Soul spark loaded successfully: ID={instance.spark_id}")
            return instance
        except json.JSONDecodeError as e: logger.error(f"JSON Decode Error: {e}"); raise ValueError(f"Invalid JSON: {e}") from e
        except Exception as e: logger.error(f"Error loading soul spark: {e}"); raise RuntimeError(f"Failed to load: {e}") from e

    def _validate_attributes(self):
         """Internal helper to validate numeric attributes after init/load."""
         logger.debug(f"Validating attributes for soul {self.spark_id}...")
         # Combine all numeric attributes expected
         numeric_attrs = [
             'stability', 'resonance', 'coherence', 'creator_alignment', 'energy',
             'frequency', 'phi_resonance', 'pattern_coherence', 'cord_integrity',
             'field_integration', 'earth_resonance', 'planetary_resonance', 'gaia_connection',
             'elemental_alignment', 'cycle_synchronization', 'gematria_value', 'name_resonance',
             'voice_frequency', 'response_level', 'heartbeat_entrainment', 'color_frequency',
             'soul_frequency', 'yin_yang_balance', 'crystallization_level', 'attribute_coherence',
             'physical_integration', 'memory_retention', 'physical_energy', 'spiritual_energy',
             'consciousness_frequency', 'state_stability'
         ]
         for attr in numeric_attrs:
              val = getattr(self, attr, None)
              if val is None: continue # Skip attributes not yet set (e.g., during early stages)

              # Strict Type Check
              if not isinstance(val, (int, float)):
                   # Attempt conversion for simple cases, otherwise fail hard
                   try:
                       val = float(val)
                       logger.warning(f"Converted attribute '{attr}' from type {type(getattr(self, attr))} to float.")
                       setattr(self, attr, val)
                   except (ValueError, TypeError):
                        raise TypeError(f"Attribute '{attr}' must be numeric (int/float), found {type(val)}.")

              # Finite Check
              if not np.isfinite(val):
                   raise ValueError(f"Attribute '{attr}' has non-finite value {val}. Must be finite.")

              # Specific Value Checks
              if 'frequency' in attr and val <= FLOAT_EPSILON:
                   # Allow 0 for color_frequency maybe? But base frequencies must be > 0
                   if attr not in ['color_frequency']: # Check specific exceptions
                        raise ValueError(f"Frequency attribute '{attr}' ({val}) must be positive.")
              elif attr == 'gematria_value':
                  if not isinstance(val, int): # Ensure Gematria is integer
                       setattr(self, attr, int(round(val)))
                       logger.warning(f"Rounded gematria_value to integer: {getattr(self, attr)}")
              # Clamp most metrics (0-1) unless specified otherwise
              elif attr not in ['frequency', 'voice_frequency', 'color_frequency', 'soul_frequency', 'consciousness_frequency', 'gematria_value']:
                   clamped_val = max(0.0, min(1.0, val))
                   if abs(clamped_val - val) > FLOAT_EPSILON:
                        logger.warning(f"Clamping attribute '{attr}' ({val}) to 0-1 range -> {clamped_val}.")
                        setattr(self, attr, clamped_val)

         # Validate list/dict types
         dict_attrs = ['aspects', 'emotional_resonance', 'earth_cycles', 'frequency_signature', 'resonance_patterns', 'life_cord', 'memory_veil', 'breath_pattern', 'identity_metrics']
         list_attrs = ['harmonics', 'memory_echoes', 'position']
         for attr in dict_attrs:
              if hasattr(self, attr) and getattr(self, attr) is not None and not isinstance(getattr(self, attr), dict):
                   raise TypeError(f"Attribute '{attr}' must be a dictionary or None, found {type(getattr(self, attr))}.")
         for attr in list_attrs:
              if hasattr(self, attr) and getattr(self, attr) is not None and not isinstance(getattr(self, attr), list):
                   raise TypeError(f"Attribute '{attr}' must be a list or None, found {type(getattr(self, attr))}.")
         # Specific check for position list
         if hasattr(self, 'position') and isinstance(getattr(self, 'position'), list):
              pos = getattr(self, 'position')
              if len(pos) != 3 or not all(isinstance(p, (int, float)) and np.isfinite(p) for p in pos):
                   raise ValueError(f"Attribute 'position' must be a list of 3 finite numbers: {pos}")

         logger.debug("Attribute validation complete.")


    def __str__(self) -> str:
        """ String representation (Updated). """
        name = getattr(self, 'name', self.spark_id[:8])
        state = getattr(self, 'consciousness_state', 'N/A')
        field = getattr(self, 'current_field_key', 'N/A')
        stab = getattr(self, 'stability', 0.0)
        coh = getattr(self, 'coherence', 0.0)
        eng = getattr(self, 'energy', 0.0)
        cryst = getattr(self, 'identity_crystallized', False)
        return (f"SoulSpark(Name: {name}, ID: {self.spark_id[:8]}, Field: {field}, State: {state}, "
                f"Stab:{stab:.2f}, Coh:{coh:.2f}, Eng:{eng:.2f}, Cryst:{cryst})")

    def __repr__(self) -> str:
        """ Detailed representation (Updated). """
        return f"<SoulSpark id='{self.spark_id}' name='{getattr(self, 'name', None)}' stability={self.stability:.4f} coherence={self.coherence:.4f} energy={self.energy:.4f}>"

# --- END OF FILE src/stage_1/soul_formation/soul_spark.py ---


