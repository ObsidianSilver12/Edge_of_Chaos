# --- START OF FILE soul_spark.py ---

"""
Soul Spark Module (Refactored)

Defines the SoulSpark class, representing the evolving soul entity.
This class holds the core attributes, state, and methods for visualization
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
    from src.constants import (
        GOLDEN_RATIO, FLOAT_EPSILON, LOG_LEVEL,
        SOUL_SPARK_DEFAULT_FREQ, SOUL_SPARK_DEFAULT_STABILITY, SOUL_SPARK_DEFAULT_RESONANCE,
        SOUL_SPARK_DEFAULT_ALIGNMENT, SOUL_SPARK_DEFAULT_ENERGY,
        SOUL_SPARK_VIABILITY_WEIGHT_STABILITY, SOUL_SPARK_VIABILITY_WEIGHT_RESONANCE,
        SOUL_SPARK_VIABILITY_WEIGHT_DIM_STABILITY, SOUL_SPARK_COMPLEXITY_DIVISOR,
        SOUL_SPARK_POTENTIAL_WEIGHT_ALIGNMENT, SOUL_SPARK_POTENTIAL_WEIGHT_DIM_STABILITY,
        # Visualization Constants (Imported for use within the class)
        SOUL_SPARK_VIZ_POINT_SIZE_FACTOR, SOUL_SPARK_VIZ_POINT_SIZE_BASE,
        SOUL_SPARK_VIZ_POINT_ALPHA_FACTOR, SOUL_SPARK_VIZ_POINT_ALPHA_MAX,
        SOUL_SPARK_VIZ_EDGE_ALPHA_FACTOR, SOUL_SPARK_VIZ_EDGE_WIDTH_FACTOR,
        SOUL_SPARK_VIZ_CENTER_COLOR, SOUL_SPARK_VIZ_CENTER_SIZE_FACTOR,
        SOUL_SPARK_VIZ_CENTER_EDGE_COLOR, SOUL_SPARK_VIZ_FREQ_SIG_BARS,
        SOUL_SPARK_VIZ_FREQ_SIG_XLABEL, SOUL_SPARK_VIZ_FREQ_SIG_YLABEL,
        SOUL_SPARK_VIZ_ENERGY_DIST_XLABEL, SOUL_SPARK_VIZ_ENERGY_DIST_YLABEL,
        SOUL_SPARK_VIZ_DIM_STAB_LABELS, SOUL_SPARK_VIZ_DIM_STAB_COLORS
    )
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants from src.constants: {e}. SoulSpark may use fallback defaults.")
    # Define crucial fallbacks ONLY if constants are absolutely necessary for basic structure
    SOUL_SPARK_DEFAULT_FREQ = 432.0
    SOUL_SPARK_DEFAULT_STABILITY = 0.6
    SOUL_SPARK_DEFAULT_RESONANCE = 0.6
    SOUL_SPARK_DEFAULT_ALIGNMENT = 0.1
    FLOAT_EPSILON = 1e-9
    # Visualization constants would fallback within methods if needed

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
        # Use provided data or defaults from constants
        data = initial_data or {}

        # Core Metrics / State Flags
        self.stability: float = float(data.get('stability', SOUL_SPARK_DEFAULT_STABILITY))
        self.resonance: float = float(data.get('resonance', SOUL_SPARK_DEFAULT_RESONANCE))
        self.coherence: float = float(data.get('coherence', SOUL_SPARK_DEFAULT_RESONANCE)) # Often related to resonance initially
        self.creator_alignment: float = float(data.get('creator_alignment', SOUL_SPARK_DEFAULT_ALIGNMENT))
        self.formation_complete: bool = bool(data.get('formation_complete', False)) # From Guff/Sephiroth
        self.harmonically_strengthened: bool = bool(data.get('harmonically_strengthened', False))
        self.cord_formation_complete: bool = bool(data.get('cord_formation_complete', False))
        self.earth_harmonized: bool = bool(data.get('earth_harmonized', False))
        self.identity_crystallized: bool = bool(data.get('identity_crystallized', False))
        self.incarnated: bool = bool(data.get('incarnated', False))
        self.ready_for_earth: bool = bool(data.get('ready_for_earth', False)) # Set by Life Cord
        self.ready_for_birth: bool = bool(data.get('ready_for_birth', False)) # Set by Earth Harmonization

        # Frequency & Harmonics
        self.frequency: float = float(data.get('frequency', SOUL_SPARK_DEFAULT_FREQ))
        self.frequency_signature: Dict[str, Any] = data.get('frequency_signature', {'base_frequency': self.frequency, 'frequencies': [], 'amplitudes': [], 'phases': [], 'num_frequencies': 0})
        self.harmonics: List[float] = data.get('harmonics', []) # Simple list of harmonic frequencies
        self.phi_resonance: float = float(data.get('phi_resonance', 0.5))
        self.pattern_coherence: float = float(data.get('pattern_coherence', 0.0)) # From Entanglement patterns

        # Aspects & Qualities
        self.aspects: Dict[str, Dict[str, Any]] = data.get('aspects', {}) # {name: {strength, source, time, details...}}
        self.creator_aspects: Dict[str, Dict[str, Any]] = data.get('creator_aspects', {}) # Aspects from Kether
        self.divine_qualities: Dict[str, Dict[str, Any]] = data.get('divine_qualities', {}) # {name: {strength, source, time}}

        # Entanglement & Life Cord
        self.creator_channel_id: Optional[str] = data.get('creator_channel_id')
        self.creator_connection_strength: float = float(data.get('creator_connection_strength', 0.0))
        self.resonance_patterns: Dict[str, Any] = data.get('resonance_patterns', {}) # From Entanglement
        self.life_cord: Optional[Dict[str, Any]] = data.get('life_cord')
        self.cord_integrity: float = float(data.get('cord_integrity', 0.0))
        self.cord_integration: float = float(data.get('cord_integration', 0.0)) # Set by Life Cord stage

        # Harmonization & Identity
        self.earth_resonance: float = float(data.get('earth_resonance', 0.0))
        self.elements: Dict[str, float] = data.get('elements', {}) # {element_name: strength}
        self.earth_cycles: Dict[str, float] = data.get('earth_cycles', {}) # {cycle_name: sync_level}
        self.planetary_resonance: float = float(data.get('planetary_resonance', 0.0))
        self.gaia_connection: float = float(data.get('gaia_connection', 0.0))
        self.elemental_alignment: float = float(data.get('elemental_alignment', 0.0))
        self.cycle_synchronization: float = float(data.get('cycle_synchronization', 0.0))
        self.name: Optional[str] = data.get('name')
        self.gematria_value: int = int(data.get('gematria_value', 0))
        self.name_resonance: float = float(data.get('name_resonance', 0.0))
        self.voice_frequency: float = float(data.get('voice_frequency', 0.0))
        self.response_level: float = float(data.get('response_level', 0.0))
        self.heartbeat_entrainment: float = float(data.get('heartbeat_entrainment', 0.0))
        self.soul_color: Optional[str] = data.get('soul_color')
        self.color_frequency: float = float(data.get('color_frequency', 0.0))
        self.soul_frequency: float = float(data.get('soul_frequency', self.frequency)) # Default to main frequency
        self.sephiroth_aspect: Optional[str] = data.get('sephiroth_aspect')
        self.elemental_affinity: Optional[str] = data.get('elemental_affinity')
        self.platonic_symbol: Optional[str] = data.get('platonic_symbol')
        self.yin_yang_balance: float = float(data.get('yin_yang_balance', 0.5))
        self.emotional_resonance: Dict[str, float] = data.get('emotional_resonance', {'love': 0.0, 'joy': 0.0, 'peace': 0.0, 'harmony': 0.0, 'compassion': 0.0})
        self.crystallization_level: float = float(data.get('crystallization_level', 0.0))
        self.attribute_coherence: float = float(data.get('attribute_coherence', 0.0))
        self.identity_metrics: Optional[Dict[str, Any]] = data.get('identity_metrics') # Store verification results
        self.sacred_geometry_imprint: Optional[str] = data.get('sacred_geometry_imprint')

        # Birth Related
        self.physical_integration: float = float(data.get('physical_integration', 0.0))
        self.birth_time: Optional[str] = data.get('birth_time')
        self.memory_veil: Optional[Dict[str, Any]] = data.get('memory_veil')
        self.memory_retention: float = float(data.get('memory_retention', 1.0)) # Base retention after veil
        self.breath_pattern: Optional[Dict[str, Any]] = data.get('breath_pattern')
        self.physical_energy: float = float(data.get('physical_energy', 0.0))
        self.spiritual_energy: float = float(data.get('spiritual_energy', 1.0))

        # Consciousness State (Store last known state)
        self.consciousness_state: str = data.get('consciousness_state', 'dream')
        self.consciousness_frequency: float = float(data.get('consciousness_frequency', 2.0)) # Default delta
        self.state_stability: float = float(data.get('state_stability', self.stability)) # Init with overall stability

        # Memory Echoes
        self.memory_echoes: List[str] = data.get('memory_echoes', []) # List of strings describing events

        # --- Validation ---
        # Ensure core numeric attributes are valid numbers
        for attr in ['stability', 'resonance', 'coherence', 'creator_alignment', 'frequency', 'phi_resonance', 'pattern_coherence', 'cord_integrity', 'earth_resonance', 'planetary_resonance', 'gaia_connection', 'elemental_alignment', 'cycle_synchronization', 'crystallization_level', 'attribute_coherence']:
             val = getattr(self, attr)
             if not isinstance(val, (int, float)):
                 logger.error(f"Attribute '{attr}' has invalid type {type(val)}. Resetting to default.")
                 setattr(self, attr, 0.5) # Reset to a neutral default
             elif not np.isfinite(val):
                 logger.error(f"Attribute '{attr}' has non-finite value {val}. Resetting to default.")
                 setattr(self, attr, 0.5)
             elif attr != 'frequency' and not (0.0 <= val <= 1.0): # Clamp most metrics to 0-1
                 setattr(self, attr, max(0.0, min(1.0, val)))
             elif attr == 'frequency' and val <= FLOAT_EPSILON:
                 logger.error(f"Attribute 'frequency' has non-positive value {val}. Resetting to default.")
                 setattr(self, attr, SOUL_SPARK_DEFAULT_FREQ)


        logger.info(f"SoulSpark initialized/loaded: ID={self.spark_id}, Freq={self.frequency:.2f}, Stab={self.stability:.3f}, Res={self.resonance:.3f}, Align={self.creator_alignment:.3f}")

    def get_spark_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the current state of the soul spark.

        Returns:
            dict: Metrics dictionary categorized by aspect.
        """
        logger.debug(f"Calculating metrics for soul {self.spark_id}...")
        # Use getattr with defaults for safety, although init should handle most cases
        metrics_data = {
            'core': {
                'stability': getattr(self, 'stability', 0.0),
                'resonance': getattr(self, 'resonance', 0.0),
                'coherence': getattr(self, 'coherence', 0.0),
                'creator_alignment': getattr(self, 'creator_alignment', 0.0),
                'last_modified': getattr(self, 'last_modified', self.creation_time)
            },
            'frequency': {
                'base_frequency': self.frequency_signature.get('base_frequency', self.frequency),
                'num_frequencies': len(self.frequency_signature.get('frequencies', [])),
                'amplitudes_mean': float(np.mean(self.frequency_signature.get('amplitudes', [0]))),
                'soul_frequency': getattr(self, 'soul_frequency', 0.0), # From identity
                'voice_frequency': getattr(self, 'voice_frequency', 0.0) # From identity
            },
            'identity': {
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
                'identity_crystallized': getattr(self, 'identity_crystallized', False)
            },
            'consciousness': {
                 'state': getattr(self, 'consciousness_state', 'dream'),
                 'frequency': getattr(self, 'consciousness_frequency', 0.0),
                 'stability': getattr(self, 'state_stability', 0.0),
            },
            'entanglement': {
                'channel_id': getattr(self, 'creator_channel_id', None),
                'connection_strength': getattr(self, 'creator_connection_strength', 0.0),
                'pattern_coherence': getattr(self, 'pattern_coherence', 0.0),
                'aspect_count': len(getattr(self, 'creator_aspects', {})),
            },
            'life_cord': {
                'formation_complete': getattr(self, 'cord_formation_complete', False),
                'integrity': getattr(self, 'cord_integrity', 0.0),
                'bandwidth': getattr(self, 'life_cord', {}).get('bandwidth', 0.0),
                'channels': getattr(self, 'life_cord', {}).get('channel_count', 0),
                'earth_connection': getattr(self, 'life_cord', {}).get('earth_connection', 0.0)
            },
            'earth_harmony': {
                 'harmonized': getattr(self, 'earth_harmonized', False),
                 'earth_resonance': getattr(self, 'earth_resonance', 0.0),
                 'elemental_alignment': getattr(self, 'elemental_alignment', 0.0),
                 'cycle_synchronization': getattr(self, 'cycle_synchronization', 0.0),
                 'planetary_resonance': getattr(self, 'planetary_resonance', 0.0),
                 'gaia_connection': getattr(self, 'gaia_connection', 0.0),
            },
            'incarnation': {
                'incarnated': getattr(self, 'incarnated', False),
                'birth_time': getattr(self, 'birth_time', None),
                'physical_integration': getattr(self, 'physical_integration', 0.0),
                'memory_retention': getattr(self, 'memory_retention', 1.0),
                'physical_energy': getattr(self, 'physical_energy', 0.0),
                'spiritual_energy': getattr(self, 'spiritual_energy', 0.0),
            },
            'memory': {
                 'echo_count': len(getattr(self, 'memory_echoes', []))
            }
            # Add more derived metrics like viability, complexity, potential using constants
            # 'overall': {
            #     'viability': (metrics_data['core']['stability'] * SOUL_SPARK_VIABILITY_WEIGHT_STABILITY + ...),
            #     'complexity': len(getattr(self, 'aspects', {})) / SOUL_SPARK_COMPLEXITY_DIVISOR,
            #     'potential': (metrics_data['core']['creator_alignment'] * SOUL_SPARK_POTENTIAL_WEIGHT_ALIGNMENT + ...)
            # }
        }
        logger.debug(f"Metrics calculated for soul {self.spark_id}")
        return metrics_data

    def add_memory_echo(self, event_description: str):
        """Adds a descriptive string as a memory echo."""
        if not hasattr(self, 'memory_echoes') or not isinstance(self.memory_echoes, list):
             self.memory_echoes = []
        timestamp = datetime.now().isoformat()
        self.memory_echoes.append(f"{timestamp}: {event_description}")
        # Optional: Limit memory size
        # if len(self.memory_echoes) > MAX_MEMORY_ECHOES:
        #     self.memory_echoes.pop(0)
        logger.debug(f"Memory echo added to soul {self.spark_id}: '{event_description}'")

    # --- Visualization Methods ---
    # Keep visualize_spark and visualize_energy_signature from previous version,
    # but ensure they use self attributes correctly and use constants for styling/layout.

    def visualize_spark(self, show: bool = True, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create a 3D visualization of the soul spark's current state.

        Args:
            show (bool): Whether to display the visualization.
            save_path (Optional[str]): Path to save the visualization image.

        Returns:
            Optional[plt.Figure]: Matplotlib figure object, or None if visualization disabled/failed.
        """
        if not VISUALIZATION_ENABLED:
            logger.warning("Visualization disabled: Matplotlib not found.")
            return None
        logger.info(f"Generating 3D visualization for soul {self.spark_id}...")

        fig = plt.figure(figsize=(12, 10)) # Slightly larger figure
        ax = fig.add_subplot(111, projection='3d')

        try:
            # --- Plot Core Structure (Placeholder - Simplified Representation) ---
            # Using attributes like stability, coherence, frequency to influence a central sphere/cloud
            center_x, center_y, center_z = 0, 0, 0 # Assume centered for visualization

            # Size based on stability & coherence
            viz_radius = 0.5 + 0.5 * (self.stability + self.coherence) / 2.0
            # Color based on soul_color or frequency
            viz_color = getattr(self, 'soul_color', 'white').lower()
            # Handle potential complex colors from constants if needed
            if '/' in viz_color: viz_color = viz_color.split('/')[0] # Take first color
            try: plt.get_cmap(viz_color) # Check if it's a valid matplotlib color/cmap name
            except ValueError: viz_color = 'plasma' # Fallback cmap

            # Create sphere data
            u = np.linspace(0, 2 * PI, 30)
            v = np.linspace(0, PI, 15)
            x = viz_radius * np.outer(np.cos(u), np.sin(v)) + center_x
            y = viz_radius * np.outer(np.sin(u), np.sin(v)) + center_y
            z = viz_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z

            # Use frequency or alignment to modulate color map
            color_metric = self.creator_alignment # Use alignment for color intensity
            face_colors = plt.cm.get_cmap(viz_color)(color_metric * np.ones_like(x))
            face_colors[..., 3] = 0.5 + 0.4 * self.resonance # Alpha based on resonance

            ax.plot_surface(x, y, z, facecolors=face_colors, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.6)

            # --- Add Aspect Representation (Example: colored points around core) ---
            num_aspects = len(getattr(self, 'aspects', {}))
            if num_aspects > 0:
                 phi_angles = np.linspace(0, num_aspects * GOLDEN_RATIO * 2 * PI, num_aspects, endpoint=False)
                 aspect_radius = viz_radius * 1.2
                 ax_x = aspect_radius * np.cos(phi_angles) * np.sin(phi_angles/2.0) + center_x
                 ax_y = aspect_radius * np.sin(phi_angles) * np.sin(phi_angles/2.0) + center_y
                 ax_z = aspect_radius * np.cos(phi_angles/2.0) + center_z
                 ax.scatter(ax_x, ax_y, ax_z, c='cyan', s=30, alpha=0.8, label=f'{num_aspects} Aspects')

            # --- Add Creator Connection Representation ---
            if self.creator_connection_strength > 0.1:
                 # Line to 'creator' point above
                 ax.plot([center_x, center_x], [center_y, center_y], [center_z + viz_radius, center_z + viz_radius + 2.0 * self.creator_connection_strength],
                         color='gold', linestyle='--', linewidth=2.5, alpha=0.7, label=f'Creator Conn ({self.creator_connection_strength:.2f})')
                 ax.scatter([center_x], [center_y], [center_z + viz_radius + 2.0 * self.creator_connection_strength], color='gold', s=60, marker='*')

            # --- Add Life Cord Anchor Representation ---
            if self.cord_formation_complete:
                 cord_integrity = getattr(self, 'cord_integrity', 0.0)
                 # Line to 'earth' point below
                 ax.plot([center_x, center_x], [center_y, center_y], [center_z - viz_radius, center_z - viz_radius - 1.5 * cord_integrity],
                         color='brown', linestyle='-', linewidth=2.0, alpha=0.6, label=f'Life Cord ({cord_integrity:.2f})')
                 ax.scatter([center_x], [center_y], [center_z - viz_radius - 1.5 * cord_integrity], color='saddlebrown', s=50, marker='v')


            # --- Plotting Setup ---
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            title = f"Soul Spark: {getattr(self, 'name', self.spark_id[:8])}\nS:{self.stability:.2f} R:{self.resonance:.2f} C:{self.coherence:.2f} A:{self.creator_alignment:.2f}"
            ax.set_title(title)

            # Auto-scaling axes
            all_points = np.array([[center_x,center_y,center_z]])
            if num_aspects > 0: all_points = np.vstack([all_points, np.column_stack([ax_x, ax_y, ax_z])])
            if self.creator_connection_strength > 0.1: all_points = np.vstack([all_points, [[center_x, center_y, center_z + viz_radius + 2.0 * self.creator_connection_strength]]])
            if self.cord_formation_complete: all_points = np.vstack([all_points, [[center_x, center_y, center_z - viz_radius - 1.5 * self.cord_integrity]]])

            if all_points.shape[0] > 1:
                ranges = np.ptp(all_points, axis=0)
                means = np.mean(all_points, axis=0)
                max_range = max(ranges) * 1.2 # Add 20% padding
                if max_range < 1.0: max_range = 1.0 # Ensure minimum range
                ax.set_xlim(means[0] - max_range / 2, means[0] + max_range / 2)
                ax.set_ylim(means[1] - max_range / 2, means[1] + max_range / 2)
                ax.set_zlim(means[2] - max_range / 2, means[2] + max_range / 2)
            else: # Fallback if only center point
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

            ax.legend()
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SoulSpark 3D visualization saved to {save_path}")
            if show:
                plt.show()
            else:
                plt.close(fig)

            return fig

        except Exception as e:
            logger.error(f"Error during SoulSpark 3D visualization: {e}", exc_info=True)
            if fig: plt.close(fig) # Ensure figure is closed on error
            return None

    def visualize_energy_signature(self, show: bool = True, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create a visualization of the energy signature (frequency spectrum).

        Args:
            show (bool): Whether to display the visualization.
            save_path (Optional[str]): Path to save the visualization image.

        Returns:
            Optional[plt.Figure]: Matplotlib figure object, or None if visualization disabled/failed.
        """
        if not VISUALIZATION_ENABLED:
            logger.warning("Visualization disabled: Matplotlib not found.")
            return None
        logger.info(f"Generating energy signature visualization for soul {self.spark_id}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            # --- Plot Frequency Spectrum ---
            ax.set_title(f'Frequency Spectrum: {getattr(self, "name", self.spark_id[:8])}')
            freqs = self.frequency_signature.get('frequencies', [])
            amps = self.frequency_signature.get('amplitudes', [])

            if freqs and amps and len(freqs) == len(amps):
                valid_data = sorted([(f, a) for f, a in zip(freqs, amps) if f > FLOAT_EPSILON and a > FLOAT_EPSILON])
                if valid_data:
                    sorted_freqs, sorted_amps = zip(*valid_data)
                    # Use constants for styling if available
                    markerline, stemlines, baseline = ax.stem(
                        sorted_freqs, sorted_amps,
                        linefmt=SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT, # Assumed constants
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
                    base_freq = self.frequency_signature.get('base_frequency', self.frequency)
                    ax.axvline(x=base_freq, color=SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR, linestyle='--', alpha=0.7, label=f'Base Freq ({base_freq:.1f} Hz)')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, "No significant frequencies found", ha='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No frequency data available", ha='center', transform=ax.transAxes)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout slightly

            # Add overall metrics text
            metrics_str = f"Stab: {self.stability:.2f} | Res: {self.resonance:.2f} | Coh: {self.coherence:.2f} | Align: {self.creator_alignment:.2f}"
            fig.text(0.5, 0.02, metrics_str, ha='center', fontsize=10, bbox=dict(facecolor='whitesmoke', alpha=0.8))

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Energy signature visualization saved to {save_path}")
            if show:
                plt.show()
            else:
                plt.close(fig)

            return fig

        except Exception as e:
            logger.error(f"Error during energy signature visualization: {e}", exc_info=True)
            if fig: plt.close(fig)
            return None

    # --- State Management ---
    def save_spark_data(self, file_path: str) -> bool:
        """
        Save the current soul spark data to a JSON file. Fails hard on IO/Type error.

        Args:
            file_path (str): Path to save the data file.

        Returns:
            bool: True if save was successful.

        Raises:
            TypeError: If an attribute has an unserializable type.
            IOError: If writing the file fails.
            RuntimeError: For other unexpected errors.
        """
        logger.info(f"Saving soul spark data for {self.spark_id} to {file_path}...")
        try:
            # Create data structure from attributes
            data_to_save = {}
            for attr, value in self.__dict__.items():
                 # Skip potentially unserializable or large objects if needed
                 # For now, attempt to serialize everything
                 data_to_save[attr] = value

            # Custom serializer for complex types if needed
            def default_serializer(o):
                if isinstance(o, np.ndarray): return o.tolist() # Convert numpy arrays
                if isinstance(o, (datetime, uuid.UUID)): return str(o)
                # Let default JSON encoder handle primitive types or raise error
                try:
                    return json.JSONEncoder().default(o)
                except TypeError:
                     logger.warning(f"Attribute '{attr}' of type {type(o)} is not JSON serializable. Skipping.")
                     return None # Skip unserializable types

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=default_serializer)

            logger.info(f"Soul spark data saved successfully.")
            return True

        except IOError as e:
            logger.error(f"IOError saving soul spark data to {file_path}: {e}", exc_info=True)
            raise IOError(f"Failed to write spark data file: {e}") from e
        except TypeError as e:
            logger.error(f"TypeError saving soul spark data (unserializable attribute?): {e}", exc_info=True)
            raise TypeError(f"Failed to serialize soul spark data: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error saving soul spark data: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save spark data: {e}") from e

    @classmethod
    def load_from_file(cls, file_path: str) -> 'SoulSpark':
        """
        Load soul spark data from a saved file. Fails hard on error.

        Args:
            file_path (str): Path to the saved spark data file.

        Returns:
            SoulSpark: An instance of SoulSpark initialized with loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid data format.
            RuntimeError: For other loading errors.
        """
        logger.info(f"Loading soul spark data from {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Soul spark data file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)

            if not isinstance(loaded_data, dict):
                raise ValueError("Invalid data format in file: root element is not a dictionary.")

            # Create a new instance and populate it
            # Pass None initially to avoid default initialization logic interfering
            instance = cls(initial_data=None, spark_id=loaded_data.get('spark_id'))

            # Populate attributes from loaded data
            for attr, value in loaded_data.items():
                # Convert lists back to numpy arrays if needed (e.g., for harmonic signature)
                # This depends on how you decide to store complex attributes
                setattr(instance, attr, value)

            # --- Post-load Validation/Initialization ---
            # Re-validate loaded numeric types
            instance._validate_attributes()
            # Ensure frequency signature components match if loaded
            fsig = getattr(instance, 'frequency_signature', {})
            if fsig:
                 num_freqs = len(fsig.get('frequencies',[]))
                 fsig['num_frequencies'] = num_freqs
                 if len(fsig.get('amplitudes',[])) != num_freqs or len(fsig.get('phases',[])) != num_freqs:
                      logger.warning(f"Frequency signature components mismatch after load for {instance.spark_id}. Regenerating.")
                      # Option: Regenerate signature based on base frequency
                      instance.generate_harmonic_structure() # Assuming this method exists now
                 else:
                      # Convert lists back to numpy arrays if needed by other methods
                      fsig['frequencies'] = np.array(fsig.get('frequencies',[]))
                      fsig['amplitudes'] = np.array(fsig.get('amplitudes',[]))
                      fsig['phases'] = np.array(fsig.get('phases',[]))

            logger.info(f"Soul spark loaded successfully: ID={instance.spark_id}")
            return instance

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file {file_path}: {e}")
            raise ValueError(f"Invalid JSON data in state file: {e}") from e
        except Exception as e:
            logger.error(f"Error loading soul spark data from {file_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load spark data: {e}") from e

    def _validate_attributes(self):
         """Internal helper to validate numeric attributes after init/load."""
         logger.debug(f"Validating attributes for soul {self.spark_id}...")
         for attr in ['stability', 'resonance', 'coherence', 'creator_alignment', 'frequency', 'phi_resonance', 'pattern_coherence', 'cord_integrity', 'earth_resonance', 'planetary_resonance', 'gaia_connection', 'elemental_alignment', 'cycle_synchronization', 'crystallization_level', 'attribute_coherence', 'name_resonance', 'voice_frequency', 'response_level', 'heartbeat_entrainment', 'color_frequency', 'soul_frequency', 'yin_yang_balance', 'physical_integration', 'physical_energy', 'spiritual_energy', 'consciousness_frequency', 'state_stability']:
              val = getattr(self, attr, None)
              if val is None: continue # Skip attributes not yet set
              if not isinstance(val, (int, float)):
                  raise TypeError(f"Attribute '{attr}' must be numeric, found {type(val)}.")
              if not np.isfinite(val):
                  raise ValueError(f"Attribute '{attr}' has non-finite value {val}.")
              # Apply clamping (0-1 for most, positive for frequencies)
              if 'frequency' in attr:
                   if val <= FLOAT_EPSILON: raise ValueError(f"Frequency attribute '{attr}' ({val}) must be positive.")
              elif 'level' not in attr and 'count' not in attr and attr != 'gematria_value' and attr != 'field_radius': # Clamp most metrics 0-1
                   setattr(self, attr, max(0.0, min(1.0, val)))
         logger.debug("Attribute validation complete.")


    def __str__(self) -> str:
        """String representation of the soul spark."""
        name = getattr(self, 'name', self.spark_id[:8])
        state = getattr(self, 'consciousness_state', 'N/A')
        stab = getattr(self, 'stability', 0.0)
        res = getattr(self, 'resonance', 0.0)
        coh = getattr(self, 'coherence', 0.0)
        align = getattr(self, 'creator_alignment', 0.0)
        cryst = getattr(self, 'identity_crystallized', False)
        return (f"SoulSpark(Name: {name}, ID: {self.spark_id[:8]}, State: {state}, "
                f"Stab:{stab:.2f}, Res:{res:.2f}, Coh:{coh:.2f}, Align:{align:.2f}, Cryst:{cryst})")

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<SoulSpark id='{self.spark_id}' name='{getattr(self, 'name', None)}' stability={self.stability:.4f} resonance={self.resonance:.4f}>"

# --- END OF FILE soul_spark.py ---


