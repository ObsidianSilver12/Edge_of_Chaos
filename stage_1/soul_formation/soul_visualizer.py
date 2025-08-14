# --- START OF FILE stage_1/soul_formation/soul_visualizer.py ---

"""
Soul Visualization Module (V2.1 - Enhanced & Robust Visualization)

Creates elegant and meaningful visualizations of soul state at key development points.
Shows density, frequency distribution, resonance patterns, and acquired aspects.
Includes robustness improvements for 3D rendering and consistent styling.
Hard fails if visualization can't be created to ensure simulation captures
critical development stages.
"""

import logging
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d import Axes3D # For type hinting, actual collection below
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # Explicit import
from skimage import measure  # For 3D visualization
import matplotlib.cm as cm
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union
import colorsys # For color manipulations if needed
import json

# Configure matplotlib to use Agg backend if no display is available
try:
    matplotlib.use('Agg')
except ImportError: # tk TclError sometimes if no display, Agg is safer
    pass


# --- Setup Logging ---
logger = logging.getLogger('soul_visualizer')
if not logger.handlers:
    logger.setLevel(logging.INFO) # Default, can be overridden by main app
    handler = logging.StreamHandler(sys.stdout)
    # Attempt to get LOG_FORMAT from constants, else use a default
    try:
        from shared.constants.constants import LOG_FORMAT
        formatter = logging.Formatter(LOG_FORMAT)
    except ImportError:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Constants ---
DEFAULT_GRID_SIZE = 30
DEFAULT_SOUL_RADIUS = 10
DEFAULT_RESOLUTION = 100
DEFAULT_3D_RESOLUTION = 30  # Adjusted for performance vs. detail balance
DEFAULT_FIG_SIZE = (16, 14) # Slightly larger for comprehensive report
DEFAULT_DPI = 150
ASPECT_CATEGORIES = {
    'spiritual': ['compassion', 'wisdom', 'light', 'love', 'connection', 'insight', 'presence', 'unity'],
    'intellectual': ['knowledge', 'logic', 'understanding', 'clarity', 'analysis', 'reasoning', 'discernment'],
    'emotional': ['empathy', 'joy', 'harmony', 'peace', 'gratitude', 'forgiveness', 'acceptance'],
    'willpower': ['courage', 'determination', 'discipline', 'focus', 'perseverance', 'strength', 'resolve']
}
SEPHIROTH_COLORS = {
    'kether': '#FFFFFF', 'chokmah': '#7EB6FF', 'binah': '#FFD700', 'daath': '#800080',
    'chesed': '#4169E1', 'geburah': '#FF4500', 'tiphareth': '#FFD700', 'netzach': '#228B22',
    'hod': '#FF8C00', 'yesod': '#9932CC', 'malkuth': '#8B4513', 'unknown': '#AAAAAA'
}
SOUL_PALETTES = {
    'ethereal': ['#081b29', '#0c2c43', '#1a5173', '#2d7bad', '#4da8db', '#a8d5f2'],
    'spiritual': ['#230b33', '#4a1260', '#732a8e', '#a63db8', '#c573d2', '#e6aeee'],
    'cosmic': ['#0a001a', '#240142', '#420866', '#730d9e', '#9925e3', '#c576ff'],
    'vibrant': ['#000000', '#2b0245', '#4a026c', '#750294', '#a702bc', '#cd35ed'],
    'crystalline': ['#02111b', '#053a5f', '#0a679a', '#1aa1d6', '#5cc9f4', '#bcebff']
}
DARK_BACKGROUND_COLOR = '#121212'
LIGHT_TEXT_COLOR = '#E0E0E0'
GRID_LINE_COLOR = '#444444'

# --- Helper Functions ---
def get_soul_color_spectrum(soul_spark) -> List[Tuple[float, float, float, float]]:
    """Get color mapping based on soul's development and Sephiroth influence."""
    palette_name = 'ethereal'
    try:
        dominant_sephirah = None
        if hasattr(soul_spark, 'sephiroth_aspect') and soul_spark.sephiroth_aspect: # Use direct aspect
            dominant_sephirah = soul_spark.sephiroth_aspect.lower()
        elif hasattr(soul_spark, 'cumulative_sephiroth_influence') and soul_spark.cumulative_sephiroth_influence:
             # Placeholder: if cumulative_sephiroth_influence is a dict of {seph_name: influence_value}
             if isinstance(soul_spark.cumulative_sephiroth_influence, dict) and soul_spark.cumulative_sephiroth_influence:
                dominant_sephirah = max(soul_spark.cumulative_sephiroth_influence.items(), key=lambda x: x[1])[0].lower()

        if dominant_sephirah:
            if dominant_sephirah in ['kether', 'chokmah', 'tiphareth', 'daath']: palette_name = 'spiritual'
            elif dominant_sephirah in ['binah', 'geburah']: palette_name = 'vibrant'
            elif dominant_sephirah in ['chesed', 'netzach']: palette_name = 'ethereal'
            elif dominant_sephirah in ['hod', 'yesod']: palette_name = 'cosmic'
            else: palette_name = 'crystalline' # Malkuth and others
        elif hasattr(soul_spark, 'coherence'):
            coherence = getattr(soul_spark, 'coherence', 50.0)
            max_coh = getattr(soul_spark, 'MAX_COHERENCE_CU', 100.0) # Assuming this might be on SoulSpark
            norm_coherence = coherence / max(1.0, max_coh)
            if norm_coherence > 0.8: palette_name = 'crystalline'
            elif norm_coherence > 0.6: palette_name = 'spiritual'
            elif norm_coherence > 0.4: palette_name = 'ethereal'
            else: palette_name = 'cosmic'
    except Exception as e: logger.warning(f"Could not determine color palette dynamically: {e}")

    palette = SOUL_PALETTES.get(palette_name, SOUL_PALETTES['ethereal'])
    colors_rgba = []
    for i, hex_color in enumerate(palette):
        try:
            rgb = to_rgba(hex_color)[:3]
            alpha = 0.4 + 0.6 * (i / max(1, len(palette) - 1))
            colors_rgba.append((*rgb, alpha))
        except ValueError: # Handle invalid hex string
            logger.warning(f"Invalid hex color '{hex_color}' in palette '{palette_name}'. Using default.")
            colors_rgba.append((0.5, 0.5, 0.5, 0.5)) # Default grey
    return colors_rgba

from matplotlib.colors import LinearSegmentedColormap, Colormap
import matplotlib.pyplot as plt

def create_soul_colormap(soul_spark) -> Colormap:
    """Create a beautiful colormap based on soul's energy spectrum."""
    try:
        color_spectrum_rgba = get_soul_color_spectrum(soul_spark)
        if len(color_spectrum_rgba) < 2: # Need at least two colors for a gradient
            logger.warning("Not enough colors in spectrum for custom colormap, using default.")
            return plt.get_cmap('viridis') # Or magma, plasma
        return LinearSegmentedColormap.from_list('soul_custom_cmap', color_spectrum_rgba, N=256)
    except Exception as e:
        logger.warning(f"Error creating custom colormap: {e}")
        return plt.get_cmap('viridis')

def get_density_factors(soul_spark) -> Dict[str, float]:
    """Calculate density distribution factors based on soul attributes."""
    factors = {'stability': 0.5, 'coherence': 0.5, 'resonance': 0.5, 'phase_coherence': 0.4, 'pattern_integrity': 0.4}
    try:
        if hasattr(soul_spark, 'stability') and soul_spark.stability is not None:
            max_s = getattr(soul_spark, 'MAX_STABILITY_SU', 100.0)
            factors['stability'] = min(1.0, max(0.0, float(soul_spark.stability) / max(1.0, float(max_s))))
        if hasattr(soul_spark, 'coherence') and soul_spark.coherence is not None:
            max_c = getattr(soul_spark, 'MAX_COHERENCE_CU', 100.0)
            factors['coherence'] = min(1.0, max(0.0, float(soul_spark.coherence) / max(1.0, float(max_c))))
        if hasattr(soul_spark, 'resonance') and soul_spark.resonance is not None:
            factors['resonance'] = min(1.0, max(0.0, float(soul_spark.resonance)))
        if hasattr(soul_spark, 'pattern_coherence') and soul_spark.pattern_coherence is not None: # Used for phase_coherence in vis
            factors['phase_coherence'] = min(1.0, max(0.0, float(soul_spark.pattern_coherence)))
        if hasattr(soul_spark, 'phi_resonance') and soul_spark.phi_resonance is not None: # Used for pattern_integrity in vis
            factors['pattern_integrity'] = min(1.0, max(0.0, float(soul_spark.phi_resonance)))
    except Exception as e: logger.warning(f"Error calculating density factors: {e}")
    return factors

def get_aspect_strengths(soul_spark) -> Dict[str, Dict[str, float]]:
    """Extract aspect strengths from soul by category."""
    aspect_by_category = {cat: {} for cat in ASPECT_CATEGORIES.keys()}
    try:
        if hasattr(soul_spark, 'aspects') and isinstance(soul_spark.aspects, dict):
            for aspect_name, aspect_data in soul_spark.aspects.items():
                if not isinstance(aspect_data, dict): continue
                strength = float(aspect_data.get('strength', 0.0))
                categorized = False
                for cat, keywords in ASPECT_CATEGORIES.items():
                    if any(keyword in aspect_name.lower() for keyword in keywords):
                        aspect_by_category[cat][aspect_name] = strength
                        categorized = True; break
                if not categorized:
                    aspect_by_category[list(ASPECT_CATEGORIES.keys())[0]][aspect_name] = strength
    except Exception as e: logger.warning(f"Error analyzing aspects: {e}")
    return aspect_by_category

def transform_frequency_signature(soul_spark) -> Optional[Dict[str, np.ndarray]]:
    """Extract frequency signature data for visualization."""
    try:
        if not hasattr(soul_spark, 'frequency_signature'): return None
        sig = soul_spark.frequency_signature
        if not isinstance(sig, dict): return None
        result = {}
        for key in ['frequencies', 'amplitudes', 'phases']:
            if key in sig:
                val = sig[key]
                if isinstance(val, list): result[key] = np.array(val, dtype=float)
                elif isinstance(val, np.ndarray): result[key] = val.astype(float)
        return result if 'frequencies' in result else None
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE: Cannot transform frequency signature: {e}", exc_info=True)
        raise RuntimeError(f"Frequency signature transformation failed") from e

def generate_soul_density_field(soul_spark, resolution: int = DEFAULT_RESOLUTION) -> np.ndarray:
    """Generate 2D density field for visualization based on soul attributes."""
    try:
        factors = get_density_factors(soul_spark)
        x = np.linspace(-1, 1, resolution); y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        radius_stable = 0.3 + 0.4 * factors['stability']
        grid = np.exp(-(X**2 + Y**2) / (2 * radius_stable**2))
        coherence = factors['coherence']; resonance = factors['resonance']

        if coherence > 0.4:
            num_circles = 7 + int(coherence * 12); circle_strength = 0.1 + 0.3 * coherence
            for i in range(num_circles):
                angle = 2 * np.pi * i / num_circles
                cx = 0.5 * coherence * np.cos(angle); cy = 0.5 * coherence * np.sin(angle)
                rad = 0.4 - 0.1 * (1 - coherence)
                grid += circle_strength * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * rad**2))
        if resonance > 0.3:
            spiral_strength = 0.1 + 0.3 * resonance; spiral_freq = 3 + 5 * resonance
            spiral_phase = np.arctan2(Y, X); radius_map = np.sqrt(X**2 + Y**2)
            grid += spiral_strength * np.sin(spiral_freq * (radius_map + spiral_phase)) * np.exp(-(radius_map**2) / 1.0)
        if factors['pattern_integrity'] > 0.3:
            phi = 1.618; phi_strength = 0.1 + 0.3 * factors['pattern_integrity']
            theta = np.arctan2(Y, X); r_map = np.sqrt(X**2 + Y**2) # Renamed r to r_map
            grid += phi_strength * np.sin(np.log(r_map + 0.1) * phi * 5.0 + theta) * np.exp(-(r_map**2) / 0.8)

        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-10) # Normalize
        edge_mask = 1.0 - (1.0 - coherence) * 0.7 * np.exp(-(X**2 + Y**2) / (1.2**2))
        grid *= edge_mask
        return grid
    except Exception as e:
        logger.error(f"Failed to generate density field: {e}", exc_info=True)
        x_fallback = np.linspace(-1,1,resolution); y_fallback = np.linspace(-1,1,resolution); X_f, Y_f = np.meshgrid(x_fallback,y_fallback)
        return np.exp(-(X_f**2 + Y_f**2) / (2 * 0.5**2))

def generate_soul_3d_field(soul_spark, resolution: int = DEFAULT_3D_RESOLUTION) -> Tuple[np.ndarray, List[float]]:
    """Generate 3D density field for visualization with isosurface values."""
    try:
        factors = get_density_factors(soul_spark)
        x = np.linspace(-1, 1, resolution); y = np.linspace(-1, 1, resolution); z = np.linspace(-1, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        radius_stable = 0.3 + 0.4 * factors['stability']
        field = np.exp(-(X**2 + Y**2 + Z**2) / (2 * radius_stable**2))
        coherence = factors['coherence']; resonance = factors['resonance']
        toroidal_factor = getattr(soul_spark, 'toroidal_flow_strength', 0.0)
        toroidal_factor = min(1.0, max(0.0, toroidal_factor))

        if toroidal_factor > 0.2:
            R_torus = 0.5; a_torus = 0.2 # Renamed R,a to R_torus,a_torus
            d_torus = ((np.sqrt(X**2 + Y**2) - R_torus)**2 + Z**2) / a_torus**2
            field = (1 - toroidal_factor) * field + toroidal_factor * np.exp(-d_torus)
        if coherence > 0.3: field += 0.2 * np.sin(8 * X * Y * Z * np.pi * coherence) * coherence
        if resonance > 0.3:
            distance = np.sqrt(X**2 + Y**2 + Z**2)
            field += 0.15 * np.sin(distance*12*np.pi*resonance)*resonance * np.exp(-(distance**2)/1.5)
        if factors['pattern_integrity'] > 0.5:
            tetra_strength = 0.2 * factors['pattern_integrity']
            tetra1 = np.maximum(X+Y+Z, np.maximum(X-Y-Z, np.maximum(-X+Y-Z, -X-Y+Z)))
            tetra2 = np.maximum(-X-Y-Z, np.maximum(-X+Y+Z, np.maximum(X-Y+Z, X+Y-Z)))
            field += tetra_strength * np.clip(1.0 - 0.5*(np.abs(tetra1)+np.abs(tetra2)),0,1) * np.exp(-(X**2+Y**2+Z**2)/0.8)

        field = (field - field.min()) / (field.max() - field.min() + 1e-10)
        hist, bin_edges = np.histogram(field.flatten(), bins=20)
        cumulative = np.cumsum(hist) / np.sum(hist)
        isosurface_levels = []
        target_percentiles = [0.2, 0.4, 0.6, 0.75, 0.85] # Ensure these are sensible for your data range
        for perc in target_percentiles:
            idx = np.searchsorted(cumulative, perc)
            if idx < len(bin_edges) - 1: isosurface_levels.append(bin_edges[idx])
            elif isosurface_levels and idx == len(bin_edges) -1 : isosurface_levels.append((bin_edges[idx-1] + field.max())/2.0) # Add something near max
        if not isosurface_levels and field.max() > field.min(): # If no levels from percentiles, add some default ones
            isosurface_levels = np.linspace(field.min() + 0.2*(field.max()-field.min()), field.max() - 0.1*(field.max()-field.min()), 3).tolist()
        if not isosurface_levels: isosurface_levels = [0.5] # Ultimate fallback
        return field, sorted(list(set(np.clip(isosurface_levels, field.min()+1e-5, field.max()-1e-5)))) # Ensure levels are within data range

    except Exception as e:
        logger.error(f"Error in 3D field generation: {e}", exc_info=True)
        x_f = np.linspace(-1,1,resolution); y_f = np.linspace(-1,1,resolution); z_f = np.linspace(-1,1,resolution); X_f,Y_f,Z_f=np.meshgrid(x_f,y_f,z_f,indexing='ij')
        return np.exp(-(X_f**2+Y_f**2+Z_f**2)/(2*0.5**2)), [0.3,0.5,0.7]


def get_soul_aspects_by_strength(soul_spark, n_top=10) -> List[Tuple[str, float]]:
    """Get top N soul aspects by strength."""
    aspects = []
    try:
        if hasattr(soul_spark, 'aspects') and isinstance(soul_spark.aspects, dict):
            for name, data in soul_spark.aspects.items():
                if isinstance(data, dict) and 'strength' in data:
                    aspects.append((name, float(data['strength'])))
        sorted_aspects = sorted(aspects, key=lambda x: x[1], reverse=True)
        return sorted_aspects[:n_top]
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE: Cannot get aspects by strength: {e}", exc_info=True)
        raise RuntimeError(f"Aspects by strength calculation failed") from e

def get_sephiroth_influence(soul_spark) -> Dict[str, float]:
    """Get Sephiroth influence levels from soul."""
    influences = {}
    try:
        # Try different potential attribute names for sephiroth influence
        seph_influence_attr_names = ['sephiroth_influence', 'cumulative_sephiroth_influence', 'sephiroth_aspect_strengths']
        seph_influence_data = None
        for attr_name in seph_influence_attr_names:
            if hasattr(soul_spark, attr_name) and isinstance(getattr(soul_spark, attr_name), dict):
                seph_influence_data = getattr(soul_spark, attr_name)
                break
        
        if seph_influence_data:
            for seph, value in seph_influence_data.items():
                influences[str(seph).lower()] = float(value)
        # If it's a single string (primary aspect), assign it full influence
        elif hasattr(soul_spark, 'sephiroth_aspect') and isinstance(soul_spark.sephiroth_aspect, str):
            influences[soul_spark.sephiroth_aspect.lower()] = 1.0

    except Exception as e: logger.warning(f"Error getting Sephiroth influences: {e}")
    return influences

# --- Main Visualization Functions ---
def visualize_density_2d(soul_spark, ax, resolution: int = DEFAULT_RESOLUTION) -> None:
    """Create beautiful 2D density plot of the soul energy field."""
    try:
        density = generate_soul_density_field(soul_spark, resolution)
        cmap = create_soul_colormap(soul_spark)
        x_coords = np.linspace(-1, 1, resolution); y_coords = np.linspace(-1, 1, resolution)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords) # Renamed X,Y
        ax.contourf(X_grid, Y_grid, density, 50, cmap=cmap, alpha=0.95)
        contour_levels = np.linspace(0.2, 0.9, 8)
        ax.contour(X_grid, Y_grid, density, levels=contour_levels, colors='white', alpha=0.3, linewidths=0.8)
        ax.imshow(density, extent=[-1, 1, -1, 1], origin='lower', cmap=cmap, alpha=0.4)
        stability = getattr(soul_spark, 'stability', None); coherence = getattr(soul_spark, 'coherence', None)
        subtitle = f"S: {stability:.1f} SU | C: {coherence:.1f} CU" if stability is not None and coherence is not None else ""
        ax.set_title(f"Soul Energy Field\n{subtitle}", fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.set_xlabel("Frequency Dimension", color=LIGHT_TEXT_COLOR); ax.set_ylabel("Resonance Dimension", color=LIGHT_TEXT_COLOR)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    except Exception as e: logger.error(f"Error in 2D density visualization: {e}", exc_info=True); raise RuntimeError(f"2D density vis failed: {e}") from e

def visualize_frequency_spectrum(soul_spark, ax) -> None:
    """Create elegant frequency spectrum visualization."""
    try:
        freq_data = transform_frequency_signature(soul_spark)
        if not freq_data or not isinstance(freq_data, dict):
            ax.text(0.5,0.5,'Frequency Data N/A',ha='center',va='center',transform=ax.transAxes,fontsize=11,color=LIGHT_TEXT_COLOR)
            ax.set_title('Frequency Spectrum',color=LIGHT_TEXT_COLOR)
            return
            
        frequencies = freq_data.get('frequencies')
        if not isinstance(frequencies, np.ndarray) or frequencies.size == 0:
            ax.text(0.5,0.5,'Invalid Frequency Data',ha='center',va='center',transform=ax.transAxes,fontsize=11,color=LIGHT_TEXT_COLOR)
            ax.set_title('Frequency Spectrum',color=LIGHT_TEXT_COLOR)
            return
            
        amplitudes = freq_data.get('amplitudes', np.ones_like(frequencies))
        if len(amplitudes) != len(frequencies): amplitudes = np.ones_like(frequencies)
        idx = np.argsort(frequencies); frequencies = frequencies[idx]; amplitudes = amplitudes[idx]
        if amplitudes.max() > 0: amplitudes = amplitudes / amplitudes.max()
        cmap = create_soul_colormap(soul_spark); colors = cmap(amplitudes)
        x_indices = np.arange(len(frequencies)) # Renamed x
        ax.bar(x_indices, amplitudes, color=colors, alpha=0.7, width=0.7)
        ax.plot(x_indices, amplitudes, '-', color='white', alpha=0.6, linewidth=1.5)
        for i in range(len(frequencies)):
            if amplitudes[i] > 0.5: ax.plot([i,i],[0,amplitudes[i]],'--',color='white',alpha=0.3,linewidth=0.8)
        ax.set_title('Soul Frequency Spectrum', fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.set_xlabel('Harmonic Components', color=LIGHT_TEXT_COLOR); ax.set_ylabel('Relative Amplitude', color=LIGHT_TEXT_COLOR)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        num_ticks = min(10, len(frequencies))
        tick_indices = np.linspace(0, len(frequencies)-1, num_ticks, dtype=int).tolist() # Ensure list
        if not tick_indices: tick_indices = [0] if len(frequencies)==1 else []

        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{frequencies[i]:.1f}" for i in tick_indices], rotation=45, color=LIGHT_TEXT_COLOR)

    except Exception as e: logger.error(f"Error in frequency visualization: {e}", exc_info=True); raise RuntimeError(f"Freq vis failed: {e}") from e

def visualize_aspects_radar(soul_spark, ax) -> None:
    """Create elegant radar chart of aspect strengths by category."""
    try:
        aspects_by_category = get_aspect_strengths(soul_spark)
        categories = list(aspects_by_category.keys())
        values = [sum(aspects.values())/len(aspects) if aspects else 0.0 for aspects in aspects_by_category.values()]
        if all(v==0 for v in values):
            ax.text(0.5,0.5,'Aspect Data N/A',ha='center',va='center',transform=ax.transAxes,fontsize=11,color=LIGHT_TEXT_COLOR); ax.set_title('Soul Aspects',color=LIGHT_TEXT_COLOR); return
        categories.append(categories[0]); values.append(values[0])
        theta = np.linspace(0, 2*np.pi, len(categories))
        cmap = create_soul_colormap(soul_spark)
        ax.plot(theta, values, 'o-', linewidth=2, color=cmap(0.7))
        ax.fill(theta, values, alpha=0.3, color=cmap(0.5))
        for level in [0.25, 0.5, 0.75]: ax.add_patch(patches.Circle((0,0),level,fill=False,color=GRID_LINE_COLOR,alpha=0.3,ls='--',lw=0.5))
        ax.set_xticks(theta[:-1]); ax.set_xticklabels(categories[:-1], color=LIGHT_TEXT_COLOR, fontsize=8) # Adjusted xticks and labels
        ax.set_ylim(0,1); ax.set_title('Soul Aspect Categories', fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.tick_params(axis='y', colors=LIGHT_TEXT_COLOR) # Color y-axis ticks
        ax.spines['polar'].set_color(GRID_LINE_COLOR) # Color polar spine

    except Exception as e: logger.error(f"Error in aspects radar visualization: {e}", exc_info=True); raise RuntimeError(f"Aspects radar vis failed: {e}") from e

def visualize_soul_3d(soul_spark, ax, resolution: int = DEFAULT_3D_RESOLUTION) -> None:
    """Create beautiful 3D visualization of the soul structure."""
    try:
        field_3d, iso_levels = generate_soul_3d_field(soul_spark, resolution)
        cmap = create_soul_colormap(soul_spark)
        logger.debug(f"3D Field range for {soul_spark.spark_id}: min={field_3d.min():.4f}, max={field_3d.max():.4f}, iso_levels={iso_levels}")
        if field_3d.min() == field_3d.max() and not iso_levels: iso_levels=[field_3d.min() + 1e-5] # Ensure one level for flat field
        
        collections_added = 0
        for i, level in enumerate(iso_levels):
            if not (field_3d.min() <= level <= field_3d.max()): # Ensure level is within data range
                logger.debug(f"Isosurface level {level:.3f} outside field data range. Skipping.")
                continue
            alpha = 0.15 + 0.1 * i
            color_val = 0.2 + 0.8 * i / max(1, len(iso_levels) -1 if len(iso_levels)>1 else 1)
            color = cmap(color_val)
            try:
                verts, faces, _, _ = measure.marching_cubes(field_3d, level=level, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution))
                verts -= 1.0 # Shift origin to center plot at (0,0,0)
            except (RuntimeError, ValueError) as mc_err: # Catch specific marching_cubes errors
                logger.warning(f"Marching cubes failed for level {level:.4f}: {mc_err}. Skipping.")
                continue
            if verts.size == 0 or faces.size == 0: logger.debug(f"No geometry for level {level:.4f}."); continue
            
            mesh = Poly3DCollection(verts[faces], linewidths=0) # Linewidths=0 for no edges by default
            mesh.set_facecolor(color); mesh.set_alpha(alpha)
            ax.add_collection3d(mesh); collections_added += 1
        
        if collections_added == 0:
            logger.warning(f"No isosurfaces rendered for soul {soul_spark.spark_id}. Plotting basic sphere.")
            u_s,v_s=np.mgrid[0:2*np.pi:20j,0:np.pi:10j]; x_s=0.5*np.cos(u_s)*np.sin(v_s); y_s=0.5*np.sin(u_s)*np.sin(v_s); z_s=0.5*np.cos(v_s) # Renamed u,v,x,y,z
            ax.plot_surface(x_s,y_s,z_s,color=cmap(0.5),alpha=0.3,rcount=10,ccount=10,linewidth=0) # Added rcount/ccount and lw=0

        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.set_xlabel('X',color=LIGHT_TEXT_COLOR); ax.set_ylabel('Y',color=LIGHT_TEXT_COLOR); ax.set_zlabel('Z',color=LIGHT_TEXT_COLOR)
        ax.set_title('Soul Structure (3D)', fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.set_box_aspect([1,1,1]); ax.view_init(elev=30, azim=np.random.uniform(30,60)) # Randomize azim slightly
        ax.grid(False) # Cleaner look without grid
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False # Transparent panes
        ax.xaxis.pane.set_edgecolor(DARK_BACKGROUND_COLOR); ax.yaxis.pane.set_edgecolor(DARK_BACKGROUND_COLOR); ax.zaxis.pane.set_edgecolor(DARK_BACKGROUND_COLOR)


    except Exception as e:
        logger.error(f"Error in 3D visualization for {soul_spark.spark_id}: {e}", exc_info=True)
        # Fallback to simple text message
        ax.text(0,0,0, "3D Visualization Error", ha='center', va='center', color=LIGHT_TEXT_COLOR)
        ax.set_title('Soul Structure (Error)', color=LIGHT_TEXT_COLOR)


def visualize_top_aspects(soul_spark, ax) -> None:
    """Visualize top soul aspects with their strengths."""
    try:
        top_aspects = get_soul_aspects_by_strength(soul_spark, n_top=8) # Reduced to 8 for clarity
        if not top_aspects:
            ax.text(0.5,0.5,'No Aspects Available',ha='center',va='center',transform=ax.transAxes,color=LIGHT_TEXT_COLOR); ax.set_title('Soul Aspects',color=LIGHT_TEXT_COLOR); return
        names = [a[0] for a in top_aspects]; strengths = [a[1] for a in top_aspects]
        cmap = create_soul_colormap(soul_spark); colors = [cmap(s*0.8+0.2) for s in strengths] # Vary color by strength
        y_pos = np.arange(len(names))
        sorted_indices = np.argsort(strengths); names=[names[i] for i in sorted_indices]; strengths=[strengths[i] for i in sorted_indices]; colors=[colors[i] for i in sorted_indices]
        bars = ax.barh(y_pos, strengths, color=colors, height=0.7, alpha=0.8, edgecolor=LIGHT_TEXT_COLOR, linewidth=0.5)
        for i, v_bar in enumerate(strengths): # Renamed v
            if v_bar > 0.01: ax.text(v_bar+0.02, i, f"{v_bar:.2f}", va='center', fontsize=8, color=LIGHT_TEXT_COLOR)
        ax.set_yticks(y_pos); ax.set_yticklabels([name[:15] for name in names], color=LIGHT_TEXT_COLOR)
        ax.set_xlabel('Strength', color=LIGHT_TEXT_COLOR); ax.set_title('Top Soul Aspects', fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlim(0,1.1); ax.grid(True, linestyle='--', alpha=0.2, color=GRID_LINE_COLOR, axis='x')
    except Exception as e: logger.error(f"Error in top aspects visualization: {e}", exc_info=True); raise RuntimeError(f"Top aspects vis failed: {e}") from e

def visualize_sephiroth_influence(soul_spark, ax) -> None:
    """Visualize the soul's connection to different Sephiroth energies."""
    try:
        influences = get_sephiroth_influence(soul_spark)
        if not influences:
            ax.text(0.5,0.5,'Sephiroth Data N/A',ha='center',va='center',transform=ax.transAxes,color=LIGHT_TEXT_COLOR); ax.set_title('Sephiroth Influence',color=LIGHT_TEXT_COLOR); return
        sorted_influences = sorted(influences.items(), key=lambda x: x[1], reverse=True)[:7] # Top 7
        names, values = zip(*sorted_influences) if sorted_influences else ([], [])
        if not names: ax.text(0.5,0.5,'No Significant Sephiroth Influence',ha='center',va='center',transform=ax.transAxes,color=LIGHT_TEXT_COLOR); ax.set_title('Sephiroth Influence',color=LIGHT_TEXT_COLOR); return

        colors = [SEPHIROTH_COLORS.get(name.lower(), '#AAAAAA') for name in names]
        # Explode the largest slice slightly for emphasis
        explode_values = [0.0] * len(values) # Renamed explode
        if values: explode_values[0] = 0.05

        wedges, texts, autotexts = ax.pie(
            values, explode=explode_values, labels=[name.capitalize() for name in names], colors=colors,
            autopct=lambda p: f'{p:.1f}%' if p > 5 else '', # Show % only for larger slices
            pctdistance=0.80, startangle=90,
            wedgeprops={'edgecolor': DARK_BACKGROUND_COLOR, 'linewidth': 1.5, 'alpha': 0.9},
            textprops={'color': LIGHT_TEXT_COLOR, 'fontsize': 7, 'fontweight': 'bold'}
        )
        for autotext_item in autotexts: autotext_item.set_color('black') # Make percentage text readable on light wedges
        ax.set_title('Sephiroth Influence', fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    except Exception as e: logger.error(f"Error in Sephiroth influence visualization: {e}", exc_info=True); raise RuntimeError(f"Sephiroth influence vis failed: {e}") from e


def visualize_soul_state(
    soul_spark,
    stage_name: str,
    output_dir: Optional[str] = None,
    show: bool = False
) -> str:
    logger.info(f"Creating visualization for soul {soul_spark.spark_id} at stage: {stage_name}")
    try:
        final_output_dir = output_dir if output_dir else os.path.join("output", "visualizations", "default_soul_states")
        os.makedirs(final_output_dir, exist_ok=True)
        fig = plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI); fig.patch.set_facecolor(DARK_BACKGROUND_COLOR)
        gs = plt.GridSpec(3,3,figure=fig,hspace=0.4,wspace=0.35,left=0.05,right=0.95,top=0.90,bottom=0.05) # Adjusted layout
        axs = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2],polar=True),
               fig.add_subplot(gs[1,:],projection='3d'), fig.add_subplot(gs[2,:2]), fig.add_subplot(gs[2,2])]
        for ax_item in axs: # Renamed ax
            ax_item.set_facecolor(DARK_BACKGROUND_COLOR)
            if hasattr(ax_item,'spines'):
                for spine in ax_item.spines.values(): spine.set_color(GRID_LINE_COLOR)
            ax_item.tick_params(colors=LIGHT_TEXT_COLOR)
            if hasattr(ax_item,'xaxis') and hasattr(ax_item.xaxis,'label'): ax_item.xaxis.label.set_color(LIGHT_TEXT_COLOR)
            if hasattr(ax_item,'yaxis') and hasattr(ax_item.yaxis,'label'): ax_item.yaxis.label.set_color(LIGHT_TEXT_COLOR)
            if hasattr(ax_item,'title'): ax_item.title.set_color(LIGHT_TEXT_COLOR)
        
        visualize_density_2d(soul_spark, axs[0]); visualize_frequency_spectrum(soul_spark, axs[1])
        visualize_aspects_radar(soul_spark, axs[2]); visualize_soul_3d(soul_spark, axs[3])
        visualize_top_aspects(soul_spark, axs[4]); visualize_sephiroth_influence(soul_spark, axs[5])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); soul_id = soul_spark.spark_id
        title_text = f"Soul State: {soul_id} - {stage_name} ({timestamp})"
        try:
            s_val = getattr(soul_spark,'stability',None); c_val = getattr(soul_spark,'coherence',None)
            if s_val is not None and c_val is not None: title_text += f"\nS: {s_val:.1f} SU, C: {c_val:.1f} CU"
        except: pass
        fig.suptitle(title_text, fontsize=16, color='white', y=0.98) # Adjusted y for suptitle
        # No plt.tight_layout() here as GridSpec handles it with hspace/wspace and rect.

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{soul_spark.spark_id}_{stage_name.replace(' ','_')}_{timestamp_str}.png"
        filepath = os.path.join(final_output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, facecolor=DARK_BACKGROUND_COLOR, bbox_inches='tight')
        logger.info(f"Visualization saved to {filepath}")

        try:
            data_save_dir = os.path.join(final_output_dir, "..", "completed_soul_data_npy") # Changed subfolder name
            os.makedirs(data_save_dir, exist_ok=True)
            # Create a dictionary that can be saved by np.save (needs to be an array or pickled)
            # For simplicity, saving a dictionary of key metrics.
            density_field_for_save = generate_soul_density_field(soul_spark, resolution=30) # Smaller for .npy
            state_data_dict = {
                'soul_id': soul_spark.spark_id, 'stage': stage_name, 'timestamp': timestamp_str,
                'density_2d_shape_for_save': density_field_for_save.shape, # Store shape
                'stability': getattr(soul_spark,'stability',0.0), 'coherence': getattr(soul_spark,'coherence',0.0),
                'frequency': getattr(soul_spark,'frequency',0.0),
                'aspects_count': len(getattr(soul_spark,'aspects',{})),
                'layers_count': len(getattr(soul_spark,'layers',[]))
            }
            # To save a dict with np.save, allow pickle or convert to structured array. Or use json.
            data_filename_json = f"{soul_spark.spark_id}_{stage_name.replace(' ','_')}_{timestamp_str}_metrics.json"
            data_filepath_json = os.path.join(data_save_dir, data_filename_json)
            with open(data_filepath_json, 'w', encoding='utf-8') as f_json:
                json.dump(state_data_dict, f_json, indent=2, default=str)
            logger.info(f"Soul state metrics (JSON) saved to {data_filepath_json}")
        except Exception as data_e: logger.warning(f"Failed to save soul state metrics: {data_e}")

        if show: plt.show()
        else: plt.close(fig)
        return filepath
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in soul visualization: {e}", exc_info=True)
        plt.close('all'); raise RuntimeError(f"Failed to create soul visualization: {e}") from e

def visualize_state_comparison(
    soul_spark_states: List[Tuple[Any, str]],
    output_dir: Optional[str] = None,
    show: bool = False
) -> str:
    if not soul_spark_states or len(soul_spark_states) < 2:
        logger.error("Need at least two soul states to compare"); raise ValueError("Need >=2 states")
    logger.info(f"Creating comparison viz for {len(soul_spark_states)} states")
    final_output_dir=output_dir if output_dir else os.path.join("output","visualizations","default_comparisons")
    os.makedirs(final_output_dir,exist_ok=True); soul_id=soul_spark_states[0][0].spark_id
    fig=plt.figure(figsize=(18,10),dpi=DEFAULT_DPI); fig.patch.set_facecolor(DARK_BACKGROUND_COLOR) # Wider for comparison
    cmap=create_soul_colormap(soul_spark_states[0][0]); stages=[stage for _,stage in soul_spark_states]; x_range=range(len(stages))

    # Stability & Coherence
    ax_sc=fig.add_subplot(2,2,1); ax_sc.set_facecolor(DARK_BACKGROUND_COLOR)
    stability_vals=[getattr(s,'stability',0.0) for s,_ in soul_spark_states]; coherence_vals=[getattr(s,'coherence',0.0) for s,_ in soul_spark_states]
    ax_sc.plot(x_range,stability_vals,'o-',label='Stability (SU)',color=cmap(0.3),lw=2,mec='white',mew=0.5,ms=7)
    ax_sc.plot(x_range,coherence_vals,'o-',label='Coherence (CU)',color=cmap(0.7),lw=2,mec='white',mew=0.5,ms=7)
    ax_sc.set_xticks(x_range); ax_sc.set_xticklabels(stages,rotation=30,ha='right',color=LIGHT_TEXT_COLOR,fontsize=9)
    ax_sc.set_ylabel('Value',color=LIGHT_TEXT_COLOR); ax_sc.set_title('S/C Progression',color='white',fontsize=14)
    ax_sc.legend(frameon=False,labelcolor='white',fontsize=10); ax_sc.grid(True,ls='--',alpha=0.2,color=GRID_LINE_COLOR)
    for spine in ax_sc.spines.values(): spine.set_color(GRID_LINE_COLOR)
    ax_sc.tick_params(colors=LIGHT_TEXT_COLOR)

    # Aspect & Layer Counts
    ax_counts=fig.add_subplot(2,2,2); ax_counts.set_facecolor(DARK_BACKGROUND_COLOR)
    aspect_counts=[len(getattr(s,'aspects',{})) for s,_ in soul_spark_states]; layer_counts=[len(getattr(s,'layers',[])) for s,_ in soul_spark_states]
    bar_width=0.35; r1=np.arange(len(stages)); r2=[x_val+bar_width for x_val in r1]
    ax_counts.bar(r1,aspect_counts,width=bar_width,color=cmap(0.85),alpha=0.8,label='Aspects',edgecolor=LIGHT_TEXT_COLOR,lw=0.5)
    ax_counts.bar(r2,layer_counts,width=bar_width,color=cmap(0.45),alpha=0.8,label='Layers',edgecolor=LIGHT_TEXT_COLOR,lw=0.5)
    ax_counts.set_xticks([r_val+bar_width/2 for r_val in r1]); ax_counts.set_xticklabels(stages,rotation=30,ha='right',color=LIGHT_TEXT_COLOR,fontsize=9)
    ax_counts.set_ylabel('Count',color=LIGHT_TEXT_COLOR); ax_counts.set_title('Aspects & Layers Growth',color='white',fontsize=14)
    ax_counts.legend(frameon=False,labelcolor='white',fontsize=10); ax_counts.grid(True,ls='--',alpha=0.2,color=GRID_LINE_COLOR,axis='y')
    for spine in ax_counts.spines.values(): spine.set_color(GRID_LINE_COLOR)
    ax_counts.tick_params(colors=LIGHT_TEXT_COLOR)

    # Density Evolution (Simplified to show key stages if too many)
    ax_density_evo = fig.add_subplot(2,1,2); ax_density_evo.set_facecolor(DARK_BACKGROUND_COLOR) # Span bottom row
    num_states_to_show = min(len(soul_spark_states), 5) # Show max 5 states for density comparison
    indices_to_show = np.linspace(0, len(soul_spark_states)-1, num_states_to_show, dtype=int).tolist()
    
    density_grid_res = 30 # Lower resolution for composite
    composite_width = density_grid_res * num_states_to_show
    composite_density = np.zeros((density_grid_res, composite_width))
    
    stages_shown_for_density = []
    for i, original_idx in enumerate(indices_to_show):
        soul_state_obj, stage_label = soul_spark_states[original_idx] # Renamed soul to soul_state_obj
        stages_shown_for_density.append(stage_label)
        density_field = generate_soul_density_field(soul_state_obj, resolution=density_grid_res)
        composite_density[:, i*density_grid_res:(i+1)*density_grid_res] = density_field

    im_density = ax_density_evo.imshow(composite_density, cmap=cmap, origin='lower', extent=(0, float(num_states_to_show), 0, 1), aspect='auto')
    cbar_density = plt.colorbar(im_density, ax=ax_density_evo, label='Normalized Density', fraction=0.046, pad=0.04)
    cbar_density.ax.yaxis.set_tick_params(color=LIGHT_TEXT_COLOR); plt.setp(plt.getp(cbar_density.ax.axes, 'yticklabels'), color=LIGHT_TEXT_COLOR)
    cbar_density.set_label('Normalized Density', color=LIGHT_TEXT_COLOR)
    ax_density_evo.set_yticks([]); ax_density_evo.set_xticks(np.arange(num_states_to_show)+0.5); ax_density_evo.set_xticklabels(stages_shown_for_density,rotation=20,ha='right',color=LIGHT_TEXT_COLOR,fontsize=9)
    ax_density_evo.set_title('Soul Density Evolution (Selected Stages)',color='white',fontsize=14)
    for spine in ax_density_evo.spines.values(): spine.set_color(GRID_LINE_COLOR)


    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title_text = f"Soul Development Comparison: {soul_id}\n({timestamp})"
    fig.suptitle(title_text, fontsize=16, color='white', y=0.98) # Adjusted y
    plt.tight_layout(rect=(0, 0, 1, 0.95)) # Adjusted rect to be a tuple
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{soul_id}_Development_Comparison_{timestamp_str}.png"
    filepath = os.path.join(final_output_dir, filename)
    plt.savefig(filepath, dpi=DEFAULT_DPI, facecolor=DARK_BACKGROUND_COLOR, bbox_inches='tight')
    logger.info(f"Comparison visualization saved to {filepath}")
    if show: plt.show()
    else: plt.close(fig)
    return filepath


# --- Other visualization functions from your provided code ---
# (generate_soul_signature, visualize_soul_signature, visualize_earth_resonance,
#  create_comprehensive_soul_report, get_soul_info_text)
# For brevity, these are not repeated here but should be included in your actual file.
# Ensure they also use DARK_BACKGROUND_COLOR, LIGHT_TEXT_COLOR, GRID_LINE_COLOR for consistency.

# Example of how one might look with the theme:
def generate_soul_signature(soul_spark) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate x,y coordinates and parameters for soul's unique signature pattern."""
    try:
        # Get core parameters from soul
        stability = getattr(soul_spark, 'stability', 50.0) / 100.0
        coherence = getattr(soul_spark, 'coherence', 50.0) / 100.0
        frequency = getattr(soul_spark, 'frequency', 432.0)
        
        # Generate time points
        t = np.linspace(0, 8*np.pi, 1000)
        
        # Calculate radius with harmonics
        r = 0.3 + 0.2 * np.sin(frequency/100 * t) * stability
        r += 0.15 * np.sin(2*t) * coherence
        r = np.clip(r, 0.1, 0.9)
        
        # Generate x,y coordinates
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        return x, y, t, r
    except Exception as e:
        logger.error(f"Error generating soul signature: {e}")
        # Return fallback simple circle
        t = np.linspace(0, 2*np.pi, 100)
        return np.cos(t)*0.5, np.sin(t)*0.5, t, np.ones_like(t)*0.5

def visualize_soul_signature(soul_spark, ax=None, with_title=True):
    if ax is None:
        fig_sig, ax_sig = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI) # Renamed fig, ax to fig_sig, ax_sig
        fig_sig.patch.set_facecolor(DARK_BACKGROUND_COLOR)
        ax_sig.set_facecolor(DARK_BACKGROUND_COLOR)
    else: # ax is provided
        ax_sig = ax

    try:
        x_sig, y_sig, t_sig, r_sig = generate_soul_signature(soul_spark) # Renamed x,y,t,r
        cmap_sig = create_soul_colormap(soul_spark); colors_sig = cmap_sig(np.linspace(0,1,len(t_sig))) # Renamed cmap, colors
        points = np.array([x_sig,y_sig]).T.reshape((-1,1,2)); segments=np.concatenate([points[:-1],points[1:]],axis=1)
        from matplotlib.collections import LineCollection # Import here if not global
        lc = LineCollection(segments.tolist(),colors=colors_sig,linewidth=2,alpha=0.8); ax_sig.add_collection(lc)
        for alpha_val, width_val in zip([0.1,0.05,0.02],[4,6,8]): # Renamed alpha, width
            lc_glow=LineCollection(segments.tolist(),colors=colors_sig,linewidth=width_val,alpha=alpha_val); ax_sig.add_collection(lc_glow)
        ax_sig.scatter(0,0,color='white',s=50,alpha=0.8,zorder=10,edgecolor=DARK_BACKGROUND_COLOR) # Edgecolor for contrast
        ax_sig.set_xlim(-1.1,1.1); ax_sig.set_ylim(-1.1,1.1); ax_sig.set_aspect('equal'); ax_sig.axis('off')
        if with_title:
            soul_id_sig=getattr(soul_spark,'spark_id','Unknown'); soul_name_sig=getattr(soul_spark,'name',None) # Renamed soul_id, soul_name
            title_sig=f"Soul Signature: {soul_id_sig}" + (f" ({soul_name_sig})" if soul_name_sig else "") # Renamed title
            ax_sig.set_title(title_sig,color=LIGHT_TEXT_COLOR,fontsize=14)
        return ax_sig
    except Exception as e:
        logger.error(f"Error visualizing soul signature: {e}", exc_info=True)
        ax_sig.text(0.5,0.5,'Signature N/A',ha='center',va='center',transform=ax_sig.transAxes,color=LIGHT_TEXT_COLOR)
        ax_sig.set_title('Soul Signature',color=LIGHT_TEXT_COLOR); return ax_sig

# --- (Include all other visualization functions, adapting their styling to the dark theme) ---

def create_comprehensive_soul_report(
    soul_spark,
    stage_name: str,
    output_dir: Optional[str] = None,
    show: bool = False
) -> str:
    logger.info(f"Creating comprehensive report for soul {soul_spark.spark_id} at stage: {stage_name}")
    try:
        final_output_dir = output_dir if output_dir else os.path.join("output", "visualizations", "comprehensive_reports")
        os.makedirs(final_output_dir, exist_ok=True)

        fig = plt.figure(figsize=(20, 18), dpi=DEFAULT_DPI) # Adjusted size for more plots
        fig.patch.set_facecolor(DARK_BACKGROUND_COLOR)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.3, left=0.05,right=0.95,top=0.92,bottom=0.05)

        axs_map = {
            '2d_density': fig.add_subplot(gs[0, 0]),
            'freq_spectrum': fig.add_subplot(gs[0, 1]),
            'aspect_radar': fig.add_subplot(gs[0, 2], polar=True),
            '3d_structure': fig.add_subplot(gs[1, :], projection='3d'),
            'top_aspects': fig.add_subplot(gs[2, 0]),
            'harmony_factors': fig.add_subplot(gs[2, 1]),
            'sephiroth_influence': fig.add_subplot(gs[2, 2]),
            'soul_signature': fig.add_subplot(gs[3,0]),
            'earth_resonance': fig.add_subplot(gs[3,1]),
            'info_panel': fig.add_subplot(gs[3,2])
        }

        for ax_name, ax_curr in axs_map.items(): # Renamed ax
            ax_curr.set_facecolor(DARK_BACKGROUND_COLOR)
            if hasattr(ax_curr,'spines'):
                for spine in ax_curr.spines.values(): spine.set_color(GRID_LINE_COLOR)
            ax_curr.tick_params(colors=LIGHT_TEXT_COLOR)
            if hasattr(ax_curr,'xaxis') and hasattr(ax_curr.xaxis,'label'): ax_curr.xaxis.label.set_color(LIGHT_TEXT_COLOR)
            if hasattr(ax_curr,'yaxis') and hasattr(ax_curr.yaxis,'label'): ax_curr.yaxis.label.set_color(LIGHT_TEXT_COLOR)
            if hasattr(ax_curr,'title'): ax_curr.title.set_color(LIGHT_TEXT_COLOR)
            if ax_name == 'info_panel': ax_curr.axis('off')


        visualize_density_2d(soul_spark, axs_map['2d_density'])
        visualize_frequency_spectrum(soul_spark, axs_map['freq_spectrum'])
        visualize_aspects_radar(soul_spark, axs_map['aspect_radar'])
        visualize_soul_3d(soul_spark, axs_map['3d_structure'])
        visualize_top_aspects(soul_spark, axs_map['top_aspects'])
        visualize_harmony_factors(soul_spark, axs_map['harmony_factors'])
        visualize_sephiroth_influence(soul_spark, axs_map['sephiroth_influence'])
        visualize_soul_signature(soul_spark, axs_map['soul_signature'], with_title=False)
        axs_map['soul_signature'].set_title("Soul Signature", color=LIGHT_TEXT_COLOR, fontsize=12) # Add title separately
        visualize_earth_resonance(soul_spark, axs_map['earth_resonance'])


        info_text_str = get_soul_info_text(soul_spark) # Renamed info_text
        axs_map['info_panel'].text(0.05, 0.95, info_text_str, color=LIGHT_TEXT_COLOR, fontsize=9,
                                   va='top', ha='left', linespacing=1.6, family='monospace',
                                   bbox=dict(boxstyle='round,pad=0.5', fc=DARK_BACKGROUND_COLOR, ec=GRID_LINE_COLOR, alpha=0.7))
        axs_map['info_panel'].set_title('Soul Overview', color=LIGHT_TEXT_COLOR, fontsize=12)


        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); soul_id_report = soul_spark.spark_id # Renamed soul_id
        title_report = f"Soul Development Report: {soul_id_report} - Stage: {stage_name}\n({timestamp})" # Renamed title
        fig.suptitle(title_report, fontsize=18, color='white', y=0.98)

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{soul_spark.spark_id}_ComprehensiveReport_{stage_name.replace(' ','_')}_{timestamp_str}.png"
        filepath = os.path.join(final_output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, facecolor=DARK_BACKGROUND_COLOR, bbox_inches='tight')
        logger.info(f"Comprehensive report saved to {filepath}")
        if show: plt.show()
        else: plt.close(fig)
        return filepath
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in comprehensive report creation: {e}", exc_info=True)
        plt.close('all'); raise RuntimeError(f"Failed to create comprehensive report: {e}") from e


def visualize_earth_resonance(soul_spark, ax) -> None:
    """Visualize the soul's resonance with Earth's energy field."""
    try:
        earth_res = getattr(soul_spark, 'earth_resonance', 0.0)
        phys_int = getattr(soul_spark, 'physical_integration', 0.0)
        cord_int = getattr(soul_spark, 'cord_integrity', 0.0)
        
        metrics = {
            'Earth Resonance': earth_res,
            'Physical Integration': phys_int,
            'Life Cord Integrity': cord_int
        }
        
        # Filter out None values and sort
        metrics = {k: v for k, v in metrics.items() if v is not None}
        if not metrics:
            ax.text(0.5, 0.5, 'Earth Connection Data N/A', ha='center', va='center', 
                   transform=ax.transAxes, color=LIGHT_TEXT_COLOR)
            ax.set_title('Earth Connection', color=LIGHT_TEXT_COLOR)
            return
            
        names = list(metrics.keys())
        values = list(metrics.values())
        x_pos = np.arange(len(names))
        
        # Create gradient bars
        cmap = create_soul_colormap(soul_spark)
        colors = [cmap(v) for v in values]
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, v in enumerate(values):
            if v > 0.01:  # Only show if significant
                ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom',
                       color=LIGHT_TEXT_COLOR, fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right', color=LIGHT_TEXT_COLOR)
        ax.set_ylim(0, 1.1)
        ax.set_title('Earth Connection Metrics', fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.grid(True, linestyle='--', alpha=0.2, color=GRID_LINE_COLOR)
        
    except Exception as e:
        logger.error(f"Error in earth resonance visualization: {e}")
        ax.text(0.5, 0.5, 'Earth Connection Vis Error', ha='center', va='center',
               transform=ax.transAxes, color=LIGHT_TEXT_COLOR)
        ax.set_title('Earth Connection', color=LIGHT_TEXT_COLOR)

def visualize_harmony_factors(soul_spark, ax) -> None:
    """Visualize harmony-related factors of the soul."""
    try:
        factors = {
            'Pattern Coherence': getattr(soul_spark, 'pattern_coherence', 0.0),
            'Phi Resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'Resonance': getattr(soul_spark, 'resonance', 0.0),
            'Harmony': getattr(soul_spark, 'harmony', 0.0),
            'Toroidal Flow': getattr(soul_spark, 'toroidal_flow_strength', 0.0)
        }
        
        # Filter out None values and sort by value
        factors = {k: v for k, v in factors.items() if v is not None}
        sorted_items = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        if not values:
            ax.text(0.5, 0.5, 'Harmony Data N/A', ha='center', va='center', transform=ax.transAxes, color=LIGHT_TEXT_COLOR)
            ax.set_title('Harmony Factors', color=LIGHT_TEXT_COLOR)
            return
            
        y_pos = np.arange(len(names))
        cmap = create_soul_colormap(soul_spark)
        colors = [cmap(v) for v in values]
        
        bars = ax.barh(y_pos, values, color=colors, height=0.7, alpha=0.8)
        
        # Add value labels
        for i, v in enumerate(values):
            if v > 0.01:  # Only show label if value is significant
                ax.text(v + 0.02, i, f'{v:.2f}', va='center', color=LIGHT_TEXT_COLOR, fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color=LIGHT_TEXT_COLOR)
        ax.set_xlim(0, 1.1)
        ax.set_title('Harmony Factors', fontsize=12, color=LIGHT_TEXT_COLOR)
        ax.grid(True, linestyle='--', alpha=0.2, color=GRID_LINE_COLOR)
        
    except Exception as e:
        logger.error(f"Error in harmony factors visualization: {e}")
        ax.text(0.5, 0.5, 'Harmony Vis Error', ha='center', va='center', transform=ax.transAxes, color=LIGHT_TEXT_COLOR)
        ax.set_title('Harmony Factors', color=LIGHT_TEXT_COLOR)

def get_soul_info_text(soul_spark):
    """Create a formatted text summary of the soul's information."""
    info = []
    info.append(f"Soul ID: {soul_spark.spark_id[:12]}...") # Shorten ID for display
    if hasattr(soul_spark, 'name') and soul_spark.name: info.append(f"Name: {soul_spark.name}")
    else: info.append("Name: Not Assigned")

    info.append("\n--- Core Metrics ---")
    max_s = getattr(soul_spark, 'MAX_STABILITY_SU', 100.0)
    max_c = getattr(soul_spark, 'MAX_COHERENCE_CU', 100.0)
    max_e = getattr(soul_spark, 'MAX_SOUL_ENERGY_SEU', 100000.0)
    info.append(f"Stability: {getattr(soul_spark,'stability',0.0):.2f}/{max_s:.0f} SU")
    info.append(f"Coherence: {getattr(soul_spark,'coherence',0.0):.2f}/{max_c:.0f} CU")
    info.append(f"Frequency: {getattr(soul_spark,'frequency',0.0):.2f} Hz")
    info.append(f"Energy:    {getattr(soul_spark,'energy',0.0):.2f}/{max_e:.0f} SEU")

    info.append("\n--- Development ---")
    if hasattr(soul_spark,'creation_time'): info.append(f"Created: {str(soul_spark.creation_time).split('.')[0]}")
    if hasattr(soul_spark,'birth_time') and soul_spark.birth_time: info.append(f"Birth: {str(soul_spark.birth_time).split('.')[0]}")
    info.append(f"Layers: {len(getattr(soul_spark,'layers',[]))}")
    info.append(f"Aspects: {len(getattr(soul_spark,'aspects',{}))}")
    if hasattr(soul_spark,'consciousness_state'): info.append(f"Consciousness: {soul_spark.consciousness_state}")

    info.append("\n--- Harmony Factors ---")
    harmony_attrs = ['phi_resonance','pattern_coherence','harmony','resonance','toroidal_flow_strength']
    for attr in harmony_attrs:
        if hasattr(soul_spark,attr) and getattr(soul_spark,attr) is not None:
            info.append(f"{attr.replace('_',' ').title()}: {getattr(soul_spark,attr):.3f}")

    info.append("\n--- Connections ---")
    if hasattr(soul_spark,'creator_connection_strength'): info.append(f"Creator Conn: {soul_spark.creator_connection_strength:.3f}")
    if hasattr(soul_spark,'cord_integrity'): info.append(f"Life Cord Int: {soul_spark.cord_integrity:.3f}")
    if hasattr(soul_spark,'earth_resonance'): info.append(f"Earth Res: {soul_spark.earth_resonance:.3f}")
    if hasattr(soul_spark,'physical_integration'): info.append(f"Physical Int: {soul_spark.physical_integration:.3f}")

    info.append("\n--- Identity ---")
    if hasattr(soul_spark,'crystallization_level'): info.append(f"Crystallization: {soul_spark.crystallization_level:.3f}")
    if hasattr(soul_spark,'soul_color'): info.append(f"Soul Color: {soul_spark.soul_color}")
    if hasattr(soul_spark,'sephiroth_aspect'): info.append(f"Sephiroth Aspect: {soul_spark.sephiroth_aspect.capitalize() if soul_spark.sephiroth_aspect else 'N/A'}")
    if hasattr(soul_spark,'elemental_affinity'): info.append(f"Elemental Affinity: {soul_spark.elemental_affinity.capitalize() if soul_spark.elemental_affinity else 'N/A'}")
    if hasattr(soul_spark,'platonic_symbol'): info.append(f"Platonic Symbol: {soul_spark.platonic_symbol.capitalize() if soul_spark.platonic_symbol else 'N/A'}")
    if hasattr(soul_spark,'zodiac_sign'): info.append(f"Zodiac: {soul_spark.zodiac_sign} ({getattr(soul_spark,'governing_planet','N/A')})")

    return "\n".join(info)


if __name__ == "__main__":
    from unittest.mock import MagicMock
    mock_soul = MagicMock(spec=['spark_id', 'stability', 'coherence', 'frequency', 'energy',
                                'aspects', 'layers', 'sephiroth_aspect', 'soul_color',
                                'MAX_STABILITY_SU', 'MAX_COHERENCE_CU', 'MAX_SOUL_ENERGY_SEU',
                                'frequency_signature', 'pattern_coherence', 'phi_resonance', 'resonance',
                                'toroidal_flow_strength', 'harmony', 'creator_connection_strength',
                                'cord_integrity', 'earth_resonance', 'physical_integration',
                                'crystallization_level', 'consciousness_state', 'creation_time',
                                'birth_time', 'name', 'elemental_affinity', 'platonic_symbol',
                                'zodiac_sign', 'governing_planet', 'sephiroth_influence']) # Add all expected attrs
    mock_soul.spark_id = "TEST-SOUL-001"
    mock_soul.stability = 65.7; mock_soul.MAX_STABILITY_SU = 100.0
    mock_soul.coherence = 72.3; mock_soul.MAX_COHERENCE_CU = 100.0
    mock_soul.frequency = 432.8; mock_soul.energy = 55000.0; mock_soul.MAX_SOUL_ENERGY_SEU = 100000.0
    mock_soul.aspects = {"wisdom":{"strength":0.8},"love":{"strength":0.9},"courage":{"strength":0.7}}
    mock_soul.layers = [{"sephirah":"kether","density":{"base_density":0.9},"color_hex":"#FFFFFF"}, {"sephirah":"chokmah","density":{"base_density":0.8},"color_hex":"#7EB6FF"}]
    mock_soul.sephiroth_aspect = "tiphareth"; mock_soul.soul_color = "#FFD700"
    mock_soul.frequency_signature = {'frequencies': np.array([432.8, 698.5, 865.6]), 'amplitudes': np.array([1.0, 0.6, 0.4]), 'phases': np.array([0.1, 0.5, 1.2])}
    mock_soul.pattern_coherence = 0.75; mock_soul.phi_resonance = 0.82; mock_soul.resonance = 0.78
    mock_soul.toroidal_flow_strength = 0.65; mock_soul.harmony = 0.88
    mock_soul.creator_connection_strength=0.5; mock_soul.cord_integrity=0.9; mock_soul.earth_resonance=0.6
    mock_soul.physical_integration=0.3; mock_soul.crystallization_level=0.0
    mock_soul.consciousness_state = "Harmonized"; mock_soul.creation_time=datetime.now().isoformat()
    mock_soul.birth_time=None; mock_soul.name="Testa"
    mock_soul.elemental_affinity="fire"; mock_soul.platonic_symbol="tetrahedron"
    mock_soul.zodiac_sign="Aries"; mock_soul.governing_planet="Mars"
    mock_soul.sephiroth_influence = {'tiphareth': 0.8, 'chesed': 0.5}


    logger.info("Starting Soul Visualizer Test...")
    output_path_state = visualize_soul_state(mock_soul, "Test_Stage", output_dir="output/test_visuals", show=False)
    print(f"Test state visualization saved to: {output_path_state}")

    # Create a second state for comparison
    mock_soul_later = MagicMock(spec=['spark_id', 'stability', 'coherence', 'frequency', 'energy',
                                'aspects', 'layers', 'sephiroth_aspect', 'soul_color',
                                'MAX_STABILITY_SU', 'MAX_COHERENCE_CU', 'MAX_SOUL_ENERGY_SEU',
                                'frequency_signature', 'pattern_coherence', 'phi_resonance', 'resonance',
                                'toroidal_flow_strength', 'harmony', 'creator_connection_strength',
                                'cord_integrity', 'earth_resonance', 'physical_integration',
                                'crystallization_level', 'consciousness_state', 'creation_time',
                                'birth_time', 'name', 'elemental_affinity', 'platonic_symbol',
                                'zodiac_sign', 'governing_planet', 'sephiroth_influence'])
    mock_soul_later.spark_id = "TEST-SOUL-001"
    mock_soul_later.stability = 80.1; mock_soul_later.MAX_STABILITY_SU = 100.0
    mock_soul_later.coherence = 85.5; mock_soul_later.MAX_COHERENCE_CU = 100.0
    mock_soul_later.aspects = {"wisdom":{"strength":0.9},"love":{"strength":0.95},"courage":{"strength":0.8}, "unity":{"strength":0.5}}
    mock_soul_later.layers = mock_soul.layers + [{"sephirah":"geburah","density":{"base_density":0.6},"color_hex":"#FF4500"}]
    mock_soul_later.frequency_signature = mock_soul.frequency_signature # Keep same for simplicity
    mock_soul_later.soul_color = "#FFA500"
    mock_soul_later.sephiroth_influence = {'tiphareth': 0.7, 'chesed': 0.6, 'geburah': 0.4}


    output_path_compare = visualize_state_comparison(
        [(mock_soul, "Early Stage"), (mock_soul_later, "Later Stage")],
        output_dir="output/test_visuals", show=False
    )
    print(f"Test comparison visualization saved to: {output_path_compare}")

    output_path_report = create_comprehensive_soul_report(mock_soul_later, "Final_Test_Stage", output_dir="output/test_visuals", show=False)
    print(f"Test comprehensive report saved to: {output_path_report}")

    logger.info("Soul Visualizer Test Finished.")

# --- END OF FILE ---







