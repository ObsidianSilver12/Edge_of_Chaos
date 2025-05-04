# --- START OF FILE stage_1/visualization/soul_visualizer.py ---

"""
Soul Visualization Module (V1.1)

Generates visualizations of the SoulSpark state at different stages.
Includes shape representation based on Platonic Symbol, S/C representation,
energy levels, and aspect distribution. Uses matplotlib. Hard fails on critical errors.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_hex
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# --- Constants ---
# Import necessary constants or define fallbacks
try:
    # Assumes constants.py is accessible from the execution path
    from constants.constants import (
        MAX_STABILITY_SU, MAX_COHERENCE_CU, FLOAT_EPSILON,
        PLATONIC_SOLIDS # List of solid names
        # COLOR_SPECTRUM # Optional, could be used for aspect colors
    )
    # Basic platonic data if full geometry isn't available
    PLATONIC_SOLID_POINTS = {'tetrahedron': 4, 'hexahedron': 8, 'octahedron': 6, 'dodecahedron': 20, 'icosahedron': 12, 'sphere': 100, 'merkaba': 8, 'default': 50}

except ImportError as e:
    logging.error(f"Constants import failed in visualizer: {e}. Using local fallbacks.")
    MAX_STABILITY_SU = 100.0; MAX_COHERENCE_CU = 100.0; FLOAT_EPSILON = 1e-9
    PLATONIC_SOLIDS = ['sphere', 'tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron', 'merkaba']
    PLATONIC_SOLID_POINTS = {'tetrahedron': 4, 'hexahedron': 8, 'octahedron': 6, 'dodecahedron': 20, 'icosahedron': 12, 'sphere': 100, 'merkaba': 8, 'default': 50}

# --- Dependency Imports ---
try:
    # Adjust import path based on project structure
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

logger = logging.getLogger('SoulVisualizer')
if not logger.handlers:
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Helper: Get Color from Frequency ---
def get_frequency_color(frequency: float) -> str:
    """ Maps frequency to a color hex string using a colormap. """
    # Define frequency range for color mapping (e.g., common soul/earth range)
    min_freq = 50.0
    max_freq = 1000.0
    try:
        norm_freq = Normalize(vmin=min_freq, vmax=max_freq)(np.clip(frequency, min_freq, max_freq))
        # Use a perceptually uniform colormap like viridis or plasma
        rgba = cm.plasma(norm_freq)
        hex_color = to_hex(rgba)
        return hex_color
    except Exception as e:
        logger.warning(f"Could not determine frequency color for {frequency} Hz: {e}. Returning grey.")
        return "#808080" # Grey fallback

# --- Main Visualization Function ---
def visualize_soul_state(soul_spark: SoulSpark, stage_name: str, output_dir: str, show: bool = False):
    """ Generates and saves a visualization of the soul state. """
    if not isinstance(soul_spark, SoulSpark):
        logger.error("Invalid SoulSpark object passed to visualizer.")
        return # Don't fail simulation for viz error

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_soul')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stage_name = "".join(c if c.isalnum() else "_" for c in stage_name) # Sanitize stage name
    filename = f"soul_{spark_id}_stage_{safe_stage_name}_{timestamp}.png"

    try:
        # Ensure the output directory exists
        save_path_dir = os.path.join(output_dir, "visuals", spark_id)
        os.makedirs(save_path_dir, exist_ok=True)
        full_path = os.path.join(save_path_dir, filename)
    except OSError as e:
        logger.error(f"Failed to create directory for visualization '{full_path}': {e}")
        return # Don't proceed if directory fails

    logger.info(f"Generating visualization for {spark_id} at stage '{stage_name}' -> {filename}")

    try:
        fig = plt.figure(figsize=(16, 9)) # Wider aspect ratio
        fig.suptitle(f"Soul State: {spark_id} - Stage: {stage_name}", fontsize=16)

        # --- 1. 3D Shape Representation ---
        ax_3d = fig.add_subplot(1, 3, 1, projection='3d')
        shape = getattr(soul_spark, 'platonic_symbol', 'sphere') # Default to sphere if not set
        stability = getattr(soul_spark, 'stability', 50.0)
        coherence = getattr(soul_spark, 'coherence', 50.0)
        frequency = getattr(soul_spark, 'frequency', 432.0)
        base_color = get_frequency_color(frequency) # Color based on frequency

        # --- Basic Shape Representation ---
        num_points = PLATONIC_SOLID_POINTS.get(shape, PLATONIC_SOLID_POINTS['default']) * 5 # More points for viz
        # Generate points on a sphere surface
        indices = np.arange(0, num_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_points)
        theta = np.pi * (1 + 5**0.5) * indices
        # Base sphere points
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)

        # --- Apply Effects based on S/C ---
        # Stability affects sharpness/definition -> Modify point radius/spread
        radius_factor = 0.5 + 0.5 * (stability / MAX_STABILITY_SU) # Sharp (1.0) to diffuse (0.5)
        point_radius = np.random.uniform(radius_factor * 0.8, radius_factor * 1.2, num_points)
        x *= point_radius
        y *= point_radius
        z *= point_radius
        # Coherence affects density/opacity -> Use alpha
        point_alpha = 0.1 + 0.8 * (coherence / MAX_COHERENCE_CU)**2 # Higher coherence = more opaque

        # --- Plotting ---
        ax_3d.scatter(x, y, z, c=base_color, alpha=point_alpha, s=30, label=f"{shape.capitalize()}")
        ax_3d.set_title(f"Structure ({shape.capitalize()})")
        ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
        ax_3d.set_xlim([-1.5, 1.5]); ax_3d.set_ylim([-1.5, 1.5]); ax_3d.set_zlim([-1.5, 1.5])
        ax_3d.legend()


        # --- 2. Core Metrics Display ---
        ax_metrics = fig.add_subplot(1, 3, 2)
        ax_metrics.axis('off') # Turn off axes for text display
        ax_metrics.set_title("Core Metrics")

        # Key metrics to display
        metrics_to_display = [
            ("Stability", f"{stability:.1f} SU", cm.coolwarm),
            ("Coherence", f"{coherence:.1f} CU", cm.viridis),
            ("Frequency", f"{frequency:.1f} Hz", cm.plasma),
            ("Spiritual Energy", f"{getattr(soul_spark, 'spiritual_energy', 0.0):.1f} SEU", cm.Blues),
            ("Physical Energy", f"{getattr(soul_spark, 'physical_energy', 0.0):.1f} SEU", cm.Reds),
            ("Earth Resonance", f"{getattr(soul_spark, 'earth_resonance', 0.0):.3f}", cm.Greens),
            ("Gaia Connection", f"{getattr(soul_spark, 'gaia_connection', 0.0):.3f}", cm.YlGn),
            ("Planetary Res.", f"{getattr(soul_spark, 'planetary_resonance', 0.0):.3f}", cm.Purples),
            ("Harmony", f"{getattr(soul_spark, 'harmony', 0.0):.3f}", cm.magma),
            ("Phi Resonance", f"{getattr(soul_spark, 'phi_resonance', 0.0):.3f}", cm.copper),
            ("Pattern Coherence", f"{getattr(soul_spark, 'pattern_coherence', 0.0):.3f}", cm.bone),
            ("Toroidal Flow", f"{getattr(soul_spark, 'toroidal_flow_strength', 0.0):.3f}", cm.cool),
            ("Crystallization", f"{getattr(soul_spark, 'crystallization_level', 0.0):.3f}", cm.binary)
        ]

        y_pos = 0.95
        for label, value_str, cmap in metrics_to_display:
            try: # Extract numeric value for coloring
                numeric_val = float(value_str.split()[0])
                if "SU" in value_str or "CU" in value_str: norm_val = Normalize(0, 100)(numeric_val)
                elif "Hz" in value_str: norm_val = Normalize(50, 1000)(numeric_val) # Freq range for color
                else: norm_val = Normalize(0, 1)(numeric_val) # Assume 0-1 for factors
                bg_color = cmap(norm_val)
            except:
                bg_color = 'lightgrey' # Fallback color

            ax_metrics.text(0.05, y_pos, f"{label}:", ha='left', va='top', fontsize=10, weight='bold')
            ax_metrics.text(0.95, y_pos, value_str, ha='right', va='top', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", fc=bg_color, alpha=0.6))
            y_pos -= 0.07 # Increment y position

        # --- 3. Aspects & Identity Info ---
        ax_aspects = fig.add_subplot(1, 3, 3)
        ax_aspects.axis('off') # Turn off axes for text display
        ax_aspects.set_title("Identity & Aspects")

        # Identity Info
        id_text = []
        if getattr(soul_spark, 'name'): id_text.append(f"Name: {soul_spark.name}")
        if getattr(soul_spark, 'zodiac_sign'): id_text.append(f"Sign: {soul_spark.zodiac_sign}")
        if getattr(soul_spark, 'elemental_affinity'): id_text.append(f"Element: {soul_spark.elemental_affinity}")
        if getattr(soul_spark, 'sephiroth_aspect'): id_text.append(f"Sephirah: {soul_spark.sephiroth_aspect}")
        if getattr(soul_spark, 'governing_planet'): id_text.append(f"Planet: {soul_spark.governing_planet}")

        ax_aspects.text(0.05, 0.95, "\n".join(id_text), ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", alpha=0.5))


        # Aspects Spider Chart (simplified as text list)
        aspects = getattr(soul_spark, 'aspects', {})
        if aspects:
             top_aspects = sorted(aspects.items(), key=lambda item: item[1].get('strength', 0) * (1.0 - item[1].get('retention_factor', 1.0)), reverse=True)[:7] # Show top 7 based on strength*retention
             aspect_lines = ["Top Aspects (Strength * Veil Factor):"]
             for name, data in top_aspects:
                  strength = data.get('strength', 0.0)
                  retention = data.get('retention_factor', 1.0) # Default to 1 if veil not applied yet
                  effective_strength = strength * retention
                  aspect_lines.append(f"- {name}: {effective_strength:.2f} (S:{strength:.2f}, R:{retention:.2f})")

             ax_aspects.text(0.05, 0.65, "\n".join(aspect_lines), ha='left', va='top', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

        # Adjust layout tightly
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap

        # --- Save and Show ---
        plt.savefig(full_path, dpi=150)
        logger.info(f"Visualization saved to {full_path}")
        if show:
            plt.show()
        else:
            plt.close(fig) # Close the figure if not showing

    except Exception as e:
        logger.error(f"Failed to generate visualization for {spark_id} at stage {stage_name}: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig) # Ensure figure is closed on error
        # Do not hard fail the simulation for a visualization error

# --- END OF FILE stage_1/visualization/soul_visualizer.py ---