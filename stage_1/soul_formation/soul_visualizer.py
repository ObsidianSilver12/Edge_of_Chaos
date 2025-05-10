"""
Soul Visualization Module (V2.0 - Enhanced Visualization)

Creates elegant and meaningful visualizations of soul state at key development points.
Shows density, frequency distribution, resonance patterns, and acquired aspects.
Hard fails if visualization can't be created to ensure simulation captures
critical development stages.
"""

import logging
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union
import colorsys
from skimage import measure, filters

# Configure matplotlib to use Agg backend if no display is available
matplotlib.use('Agg')

# --- Setup Logging ---
logger = logging.getLogger('soul_visualizer')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# --- Constants ---
DEFAULT_GRID_SIZE = 30
DEFAULT_SOUL_RADIUS = 10
DEFAULT_RESOLUTION = 100
DEFAULT_3D_RESOLUTION = 50  # Increased for better detail
DEFAULT_FIG_SIZE = (14, 12)  # Larger figure size
DEFAULT_DPI = 200  # Higher DPI for better quality
ASPECT_CATEGORIES = {
    'spiritual': ['compassion', 'wisdom', 'light', 'love', 'connection', 'insight', 'presence'],
    'intellectual': ['knowledge', 'logic', 'understanding', 'clarity', 'analysis', 'reasoning', 'discernment'],
    'emotional': ['empathy', 'joy', 'harmony', 'peace', 'gratitude', 'forgiveness', 'acceptance'],
    'willpower': ['courage', 'determination', 'discipline', 'focus', 'perseverance', 'strength', 'resolve']
}
SEPHIROTH_COLORS = {
    'kether': '#FFFFFF',   # White
    'chokmah': '#7EB6FF',  # Blue
    'binah': '#FFD700',    # Gold
    'daath': '#800080',    # Purple
    'chesed': '#4169E1',   # Royal Blue
    'geburah': '#FF4500',  # Red-Orange
    'tiphareth': '#FFD700', # Gold
    'netzach': '#228B22',  # Forest Green
    'hod': '#FF8C00',      # Dark Orange
    'yesod': '#9932CC',    # Dark Orchid
    'malkuth': '#8B4513'   # Saddle Brown
}

# Beautiful gradient palettes for soul visualization
SOUL_PALETTES = {
    'ethereal': ['#081b29', '#0c2c43', '#1a5173', '#2d7bad', '#4da8db', '#a8d5f2'],
    'spiritual': ['#230b33', '#4a1260', '#732a8e', '#a63db8', '#c573d2', '#e6aeee'],
    'cosmic': ['#0a001a', '#240142', '#420866', '#730d9e', '#9925e3', '#c576ff'],
    'vibrant': ['#000000', '#2b0245', '#4a026c', '#750294', '#a702bc', '#cd35ed'],
    'crystalline': ['#02111b', '#053a5f', '#0a679a', '#1aa1d6', '#5cc9f4', '#bcebff']
}

# --- Helper Functions ---
def get_soul_color_spectrum(soul_spark) -> List[Tuple[float, float, float, float]]:
    """Get color mapping based on soul's development and Sephiroth influence."""
    # Default color palette if we can't derive from soul
    palette_name = 'ethereal'
    
    # Try to extract palette based on soul attributes
    try:
        # Get dominant sephiroth influence if available
        dominant_sephirah = None
        if hasattr(soul_spark, 'sephiroth_influence') and soul_spark.sephiroth_influence:
            dominant_sephirah = max(soul_spark.sephiroth_influence.items(), key=lambda x: x[1])[0]
        
        # Choose palette based on dominant sephirah or coherence
        if dominant_sephirah:
            if dominant_sephirah in ['kether', 'chokmah', 'tiphareth']:
                palette_name = 'spiritual'
            elif dominant_sephirah in ['binah', 'geburah']:
                palette_name = 'vibrant'
            elif dominant_sephirah in ['chesed', 'netzach']:
                palette_name = 'ethereal'
            elif dominant_sephirah in ['hod', 'yesod', 'daath']:
                palette_name = 'cosmic'
            else:
                palette_name = 'crystalline'
        elif hasattr(soul_spark, 'coherence'):
            coherence = getattr(soul_spark, 'coherence', 50)
            if coherence > 80:
                palette_name = 'crystalline'
            elif coherence > 60:
                palette_name = 'spiritual'
            elif coherence > 40:
                palette_name = 'ethereal'
            elif coherence > 20:
                palette_name = 'cosmic'
            else:
                palette_name = 'vibrant'
                
    except Exception as e:
        logger.warning(f"Could not determine color palette: {e}")
    
    # Convert hex colors to rgba with alpha gradient
    palette = SOUL_PALETTES[palette_name]
    colors_rgba = []
    
    for i, hex_color in enumerate(palette):
        # Convert hex to rgb
        rgb = to_rgba(hex_color)[:3]
        # Calculate alpha based on position (deeper colors more transparent)
        alpha = 0.4 + 0.6 * (i / (len(palette) - 1))
        colors_rgba.append((*rgb, alpha))
    
    return colors_rgba

def get_density_factors(soul_spark) -> Dict[str, float]:
    """Calculate density distribution factors based on soul attributes."""
    try:
        # Start with some defaults
        factors = {
            'stability': 0.5,
            'coherence': 0.5,
            'resonance': 0.5,
            'phase_coherence': 0.4,
            'pattern_integrity': 0.4
        }
        
        # Try to get actual values where available
        if hasattr(soul_spark, 'stability') and soul_spark.stability is not None:
            max_stability = getattr(soul_spark, 'MAX_STABILITY_SU', 100.0)
            factors['stability'] = min(1.0, max(0.0, soul_spark.stability / max_stability))
        
        if hasattr(soul_spark, 'coherence') and soul_spark.coherence is not None:
            max_coherence = getattr(soul_spark, 'MAX_COHERENCE_CU', 100.0) 
            factors['coherence'] = min(1.0, max(0.0, soul_spark.coherence / max_coherence))
        
        if hasattr(soul_spark, 'resonance') and soul_spark.resonance is not None:
            factors['resonance'] = min(1.0, max(0.0, soul_spark.resonance))
        
        if hasattr(soul_spark, 'pattern_coherence') and soul_spark.pattern_coherence is not None:
            factors['phase_coherence'] = min(1.0, max(0.0, soul_spark.pattern_coherence))
            
        if hasattr(soul_spark, 'phi_resonance') and soul_spark.phi_resonance is not None:
            factors['pattern_integrity'] = min(1.0, max(0.0, soul_spark.phi_resonance))
            
        return factors
    except Exception as e:
        logger.warning(f"Error calculating density factors: {e}")
        return {'stability': 0.5, 'coherence': 0.5, 'resonance': 0.5}

def get_aspect_strengths(soul_spark) -> Dict[str, Dict[str, float]]:
    """Extract aspect strengths from soul by category."""
    try:
        # Initialize categories
        aspect_by_category = {cat: {} for cat in ASPECT_CATEGORIES.keys()}
        
        # Check if soul has aspects
        if hasattr(soul_spark, 'aspects') and soul_spark.aspects:
            # For each aspect, try to categorize it
            for aspect_name, aspect_data in soul_spark.aspects.items():
                if not isinstance(aspect_data, dict):
                    continue
                
                strength = aspect_data.get('strength', 0.0)
                categorized = False
                
                # Try to find a category
                for cat, keywords in ASPECT_CATEGORIES.items():
                    for keyword in keywords:
                        if keyword in aspect_name.lower():
                            aspect_by_category[cat][aspect_name] = strength
                            categorized = True
                            break
                    if categorized:
                        break
                
                # If not categorized, put in the first category as fallback
                if not categorized:
                    fallback_category = list(ASPECT_CATEGORIES.keys())[0]
                    aspect_by_category[fallback_category][aspect_name] = strength
        
        return aspect_by_category
    except Exception as e:
        logger.warning(f"Error analyzing aspects: {e}")
        return {cat: {} for cat in ASPECT_CATEGORIES.keys()}
        
def transform_frequency_signature(soul_spark) -> Optional[Dict[str, np.ndarray]]:
    """Extract frequency signature data for visualization."""
    try:
        if not hasattr(soul_spark, 'frequency_signature'):
            return None
            
        sig = soul_spark.frequency_signature
        if not isinstance(sig, dict):
            return None
            
        # Extract key components
        result = {}
        for key in ['frequencies', 'amplitudes', 'phases']:
            if key in sig:
                # Convert to numpy array if it's a list
                if isinstance(sig[key], list):
                    result[key] = np.array(sig[key])
                elif isinstance(sig[key], np.ndarray):
                    result[key] = sig[key]
                else:
                    # Skip if not valid
                    continue
        
        # Only return if we have at least frequencies
        if 'frequencies' in result:
            return result
        return None
    except Exception as e:
        logger.warning(f"Error transforming frequency signature: {e}")
        return None

def generate_soul_density_field(soul_spark, resolution: int = DEFAULT_RESOLUTION) -> np.ndarray:
    """Generate 2D density field for visualization based on soul attributes."""
    try:
        # Get density factors from soul attributes
        factors = get_density_factors(soul_spark)
        
        # Initialize grid with gaussian shape
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Base gaussian with radius controlled by stability
        radius = 0.3 + 0.4 * factors['stability']
        grid = np.exp(-(X**2 + Y**2) / (2 * radius**2))
        
        # Create more detailed structure based on factors
        coherence = factors['coherence']
        resonance = factors['resonance']
        phase_coherence = factors['phase_coherence']
        
        # Create flower of life pattern for more coherent souls
        if coherence > 0.4:
            # Multiple overlapping circles for flower of life pattern
            num_circles = 7 + int(coherence * 12)  # More circles for higher coherence
            circle_strength = 0.1 + 0.3 * coherence
            
            for i in range(num_circles):
                # Calculate circle positions in a flower pattern
                angle = 2 * np.pi * i / num_circles
                cx = 0.5 * coherence * np.cos(angle)
                cy = 0.5 * coherence * np.sin(angle)
                rad = 0.4 - 0.1 * (1 - coherence)
                
                # Add circle
                circle = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * rad**2))
                grid += circle_strength * circle
                
        # Add resonance-based spiral patterns
        if resonance > 0.3:
            # Add spiral wave patterns
            spiral_strength = 0.1 + 0.3 * resonance
            spiral_freq = 3 + 5 * resonance  # Higher frequency for higher resonance
            spiral_phase = np.arctan2(Y, X)
            radius_map = np.sqrt(X**2 + Y**2)
            
            spiral = np.sin(spiral_freq * (radius_map + spiral_phase))
            spiral_mask = np.exp(-(radius_map**2) / 1.0)  # Fade at edges
            grid += spiral_strength * spiral * spiral_mask
            
        # Add phi-resonance golden spiral pattern
        if factors['pattern_integrity'] > 0.3:
            # Golden spiral based on Fibonacci
            phi = 1.618
            phi_strength = 0.1 + 0.3 * factors['pattern_integrity']
            theta = np.arctan2(Y, X)
            r = np.sqrt(X**2 + Y**2)
            
            # Create spiral pattern based on golden ratio
            golden_spiral = np.sin(np.log(r+0.1) * phi * 5.0 + theta)
            spiral_mask = np.exp(-(r**2) / 0.8)  # Fade at edges
            grid += phi_strength * golden_spiral * spiral_mask
            
        # Normalize the grid
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-10)
        
        # Apply smooth gradient with depth effect
        edge_mask = 1.0 - (1.0 - coherence) * 0.7 * np.exp(-(X**2 + Y**2) / (1.2**2))
        grid = grid * edge_mask
        
        return grid
    except Exception as e:
        logger.error(f"Failed to generate density field: {e}")
        # Return fallback simple gaussian
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        return np.exp(-(X**2 + Y**2) / (2 * 0.5**2))

def generate_soul_3d_field(soul_spark, resolution: int = DEFAULT_3D_RESOLUTION) -> Tuple[np.ndarray, List[float]]:
    """Generate 3D density field for visualization with isosurface values."""
    try:
        # Get density factors from soul attributes
        factors = get_density_factors(soul_spark)
        
        # Initialize 3D grid
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        z = np.linspace(-1, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Base spherical field with radius controlled by stability
        radius = 0.3 + 0.4 * factors['stability']
        field = np.exp(-(X**2 + Y**2 + Z**2) / (2 * radius**2))
        
        # Get key soul properties for advanced patterns
        coherence = factors['coherence']
        resonance = factors['resonance']
        
        # Get toroidal factor if available
        toroidal_factor = 0.0
        if hasattr(soul_spark, 'toroidal_flow_strength'):
            toroidal_factor = min(1.0, max(0.0, soul_spark.toroidal_flow_strength))
        
        # Create more complex soul structure based on factors
        
        # 1. For more coherent souls, create flower of life-inspired structure
        if coherence > 0.4:
            # Add vesica piscis structure (overlapping spheres)
            sphere_dist = 0.4 * coherence
            for axis in [(1,0,0), (0,1,0), (0,0,1)]:
                dx, dy, dz = [val * sphere_dist for val in axis]
                # Create two overlapping spheres
                sphere1 = np.exp(-((X+dx)**2 + (Y+dy)**2 + (Z+dz)**2) / (2 * (radius*0.7)**2))
                sphere2 = np.exp(-((X-dx)**2 + (Y-dy)**2 + (Z-dz)**2) / (2 * (radius*0.7)**2))
                field += 0.3 * coherence * (sphere1 + sphere2)
        
        # 2. Add toroidal component for souls with toroidal flow
        if toroidal_factor > 0.2:
            # Create torus parameters
            R = 0.5  # Major radius
            a = 0.2  # Minor radius - smaller for more elegant effect
            
            # Calculate distance to torus ring
            torus_dist = ((np.sqrt(X**2 + Y**2) - R)**2 + Z**2) / a**2
            torus = np.exp(-torus_dist)
            
            # Blend with core field
            field = (1 - toroidal_factor) * field + toroidal_factor * torus
        
        # 3. Add resonance-based harmonic patterns
        if resonance > 0.3:
            # Create standing wave patterns
            wave_strength = 0.15 * resonance
            wave_freq = 4 + 8 * resonance  # More complex patterns with higher resonance
            wave_pattern = wave_strength * np.sin(wave_freq * X) * np.sin(wave_freq * Y) * np.sin(wave_freq * Z)
            
            # Apply waves with radial falloff
            radius_mask = np.exp(-(X**2 + Y**2 + Z**2) / (1.5**2))
            field += wave_pattern * radius_mask
        
        # 4. Add Sri Yantra inspired patterns for souls with high pattern integrity
        if factors['pattern_integrity'] > 0.5:
            # Create tetrahedron-like structures
            tetra_strength = 0.2 * factors['pattern_integrity']
            
            # First tetrahedron - upward pointing
            tetra1 = np.maximum(X + Y + Z, np.maximum(X - Y - Z, np.maximum(-X + Y - Z, -X - Y + Z)))
            
            # Second tetrahedron - downward pointing
            tetra2 = np.maximum(-X - Y - Z, np.maximum(-X + Y + Z, np.maximum(X - Y + Z, X + Y - Z)))
            
            # Combine with star tetrahedron effect
            star_tetra = 1.0 - 0.5 * (np.abs(tetra1) + np.abs(tetra2))
            star_tetra = np.clip(star_tetra, 0, 1)
            
            # Add to field with appropriate masking and blending
            radius_mask = np.exp(-(X**2 + Y**2 + Z**2) / 0.8)
            field += tetra_strength * star_tetra * radius_mask
            
        # Normalize the field
        field = (field - field.min()) / (field.max() - field.min() + 1e-10)
        
        # Calculate appropriate isosurface levels based on field distribution
        # Generate 4-6 isosurface levels for rendering
        hist, bin_edges = np.histogram(field.flatten(), bins=20)
        cumulative = np.cumsum(hist) / np.sum(hist)
        
        # Select levels that represent meaningful contours
        isosurface_levels = []
        target_percentiles = [0.2, 0.4, 0.6, 0.75, 0.85]
        
        for perc in target_percentiles:
            idx = np.searchsorted(cumulative, perc)
            if idx < len(bin_edges) - 1:
                isosurface_levels.append(bin_edges[idx])
        
        return field, isosurface_levels
        
    except Exception as e:
        logger.error(f"Error in 3D field generation: {e}")
        # Return fallback simple gaussian
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        z = np.linspace(-1, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        field = np.exp(-(X**2 + Y**2 + Z**2) / (2 * 0.5**2))
        return field, [0.3, 0.5, 0.7]

def create_soul_colormap(soul_spark) -> LinearSegmentedColormap:
    """Create a beautiful colormap based on soul's energy spectrum."""
    try:
        # Get soul color spectrum
        color_spectrum = get_soul_color_spectrum(soul_spark)
        
        # Create custom colormap
        return LinearSegmentedColormap.from_list('soul_colormap', color_spectrum, N=256)
    except Exception as e:
        logger.warning(f"Error creating custom colormap: {e}")
        # Fallback to a beautiful preset
        return plt.get_cmap('magma')

def get_soul_aspects_by_strength(soul_spark, n_top=10) -> List[Tuple[str, float]]:
    """Get top N soul aspects by strength."""
    try:
        aspects = []
        if hasattr(soul_spark, 'aspects') and soul_spark.aspects:
            for name, data in soul_spark.aspects.items():
                if isinstance(data, dict) and 'strength' in data:
                    aspects.append((name, data['strength']))
                
        # Sort by strength and take top N
        sorted_aspects = sorted(aspects, key=lambda x: x[1], reverse=True)
        return sorted_aspects[:n_top]
    except Exception as e:
        logger.warning(f"Error getting aspects by strength: {e}")
        return []

def get_sephiroth_influence(soul_spark) -> Dict[str, float]:
    """Get Sephiroth influence levels from soul."""
    try:
        influences = {}
        if hasattr(soul_spark, 'sephiroth_influence') and soul_spark.sephiroth_influence:
            for seph, value in soul_spark.sephiroth_influence.items():
                influences[seph.lower()] = float(value)
        return influences
    except Exception as e:
        logger.warning(f"Error getting Sephiroth influences: {e}")
        return {}

# --- Main Visualization Functions ---
def visualize_density_2d(soul_spark, ax, resolution: int = DEFAULT_RESOLUTION) -> None:
    """Create beautiful 2D density plot of the soul energy field."""
    try:
        # Generate the density field
        density = generate_soul_density_field(soul_spark, resolution)
        
        # Create custom colormap
        cmap = create_soul_colormap(soul_spark)
        
        # Plot density as a beautiful contour-filled plot
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Plot as filled contours for more elegant appearance
        contourf = ax.contourf(X, Y, density, 50, cmap=cmap, alpha=0.95)
        
        # Add subtle contour lines
        contour_levels = np.linspace(0.2, 0.9, 8)
        contours = ax.contour(X, Y, density, levels=contour_levels, colors='white', 
                             alpha=0.3, linewidths=0.8)
        
        # Add glow effect
        ax.imshow(density, extent=[-1, 1, -1, 1], origin='lower', 
                 cmap=cmap, alpha=0.4)
        
        # Get stability and coherence for title
        stability = getattr(soul_spark, 'stability', None)
        coherence = getattr(soul_spark, 'coherence', None)
        subtitle = ""
        if stability is not None and coherence is not None:
            subtitle = f"S: {stability:.1f} SU | C: {coherence:.1f} CU"
            
        ax.set_title(f"Soul Energy Field\n{subtitle}", fontsize=12)
        ax.set_xlabel("Frequency Dimension")
        ax.set_ylabel("Resonance Dimension")
        
        # Set clean, minimal axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        
    except Exception as e:
        logger.error(f"Error in 2D density visualization: {e}")
        raise RuntimeError(f"Failed to create 2D density visualization: {e}")

def visualize_frequency_spectrum(soul_spark, ax) -> None:
    """Create elegant frequency spectrum visualization."""
    try:
        # Transform frequency data
        freq_data = transform_frequency_signature(soul_spark)
        
        if not freq_data or 'frequencies' not in freq_data:
            # Draw placeholder if no data
            ax.text(0.5, 0.5, 'Frequency Data Not Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=11)
            ax.set_title('Frequency Spectrum')
            return
            
        # Extract data
        frequencies = freq_data['frequencies']
        amplitudes = freq_data.get('amplitudes', np.ones_like(frequencies))
        
        # Ensure we have amplitudes
        if len(amplitudes) != len(frequencies):
            amplitudes = np.ones_like(frequencies)
            
        # Sort by frequency
        idx = np.argsort(frequencies)
        frequencies = frequencies[idx]
        amplitudes = amplitudes[idx]
        
        # Scale amplitudes to 0-1
        if amplitudes.max() > 0:
            amplitudes = amplitudes / amplitudes.max()
        
        # Create colormap for the bars
        cmap = create_soul_colormap(soul_spark)
        colors = cmap(amplitudes)
        
        # Create elegant visualization with smooth curves
        x = np.arange(len(frequencies))
        
        # Plot bars with gradient colors
        ax.bar(x, amplitudes, color=colors, alpha=0.7, width=0.7)
        
        # Add smooth curve connecting the peaks
        ax.plot(x, amplitudes, '-', color='white', alpha=0.6, linewidth=1.5)
        
        # Add subtle vertical lines to highlight peaks
        for i in range(len(frequencies)):
            if amplitudes[i] > 0.5:  # Only highlight significant frequencies
                ax.plot([i, i], [0, amplitudes[i]], '--', color='white', 
                        alpha=0.3, linewidth=0.8)
        
        ax.set_title('Soul Frequency Spectrum', fontsize=12)
        ax.set_xlabel('Harmonic Components')
        ax.set_ylabel('Relative Amplitude')
        
        # Clean up the chart
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set tick labels only if not too many
        if len(frequencies) <= 10:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{f:.1f}" for f in frequencies], rotation=45)
        else:
            # Sample some ticks
            ticks = np.linspace(0, len(frequencies)-1, 7, dtype=int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{frequencies[i]:.1f}" for i in ticks], rotation=45)
            
    except Exception as e:
        logger.error(f"Error in frequency visualization: {e}")
        raise RuntimeError(f"Failed to create frequency visualization: {e}")

def visualize_aspects_radar(soul_spark, ax) -> None:
    """Create elegant radar chart of aspect strengths by category."""
    try:
        # Get categorized aspects
        aspects_by_category = get_aspect_strengths(soul_spark)
        
        # Calculate category averages
        categories = list(aspects_by_category.keys())
        values = []
        
        for category, aspects in aspects_by_category.items():
            if aspects:
                avg_strength = sum(aspects.values()) / len(aspects)
                values.append(avg_strength)
            else:
                values.append(0.0)
                
        # If no aspects found, draw placeholder
        if all(v == 0 for v in values):
            ax.text(0.5, 0.5, 'Aspect Data Not Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=11)
            ax.set_title('Soul Aspects')
            return
                
        # Ensure radar is complete circle by duplicating first value
        categories.append(categories[0])
        values.append(values[0])
        
        # Convert to radians for plotting
        theta = np.linspace(0, 2*np.pi, len(categories))
        
        # Get colormap for elegant styling
        cmap = create_soul_colormap(soul_spark)
        
        # Plot radar with beautiful styling
        ax.plot(theta, values, 'o-', linewidth=2, color=cmap(0.7))
        
        # Fill with gradient and transparency
        ax.fill(theta, values, alpha=0.3, color=cmap(0.5))
        
        # Add subtle grid circles
        grid_levels = [0.25, 0.5, 0.75]
        for level in grid_levels:
            circle = plt.Circle((0, 0), level, fill=False, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
        
        # Add labels with nicer formatting
        for i, cat in enumerate(categories[:-1]):
            angle = i * 2 * np.pi / len(categories[:-1])
            x = 1.2 * np.cos(angle)
            y = 1.2 * np.sin(angle)
            if x < 0:
                ax.text(x, y, cat, ha='right', va='center', fontsize=9)
            else:
                ax.text(x, y, cat, ha='left', va='center', fontsize=9)
        
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_title('Soul Aspect Categories', fontsize=12)
        
    except Exception as e:
        logger.error(f"Error in aspects radar visualization: {e}")
        raise RuntimeError(f"Failed to create aspects radar visualization: {e}")

def visualize_soul_3d(soul_spark, ax, resolution: int = DEFAULT_3D_RESOLUTION) -> None:
    """Create beautiful 3D visualization of the soul structure."""
    try:
        # Generate 3D field and isosurface levels
        field_3d, iso_levels = generate_soul_3d_field(soul_spark, resolution)
        
        # Create coordinate grids
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        z = np.linspace(-1, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Get colormap
        cmap = create_soul_colormap(soul_spark)
        
        # Create elegant isosurfaces
        for i, level in enumerate(iso_levels):
            # Higher levels (inner) are more opaque
            alpha = 0.2 + 0.1 * i
            color = cmap(0.3 + 0.7 * i/len(iso_levels))
            
            # Create isosurface
            verts, faces, _, _ = measure.marching_cubes(field_3d, level)
            
            # Scale vertices to our coordinate system
            verts = verts / resolution * 2 - 1
            
            # Plot as triangular mesh with smooth shading
            mesh = Poly3DCollection(verts[faces])
            mesh.set_facecolor(color)
            mesh.set_edgecolor('none')
            mesh.set_alpha(alpha)
            ax.add_collection3d(mesh)
        
        # Create a subtle wireframe for the outermost layer
        if len(iso_levels) > 0:
            level = iso_levels[0]
            verts, faces, _, _ = measure.marching_cubes(field_3d, level)
            verts = verts / resolution * 2 - 1
            
            mesh_wire = Poly3DCollection(verts[faces], facecolors='none', edgecolors='white', alpha=0.1)
            ax.add_collection3d(mesh_wire)
        
        # Set limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Soul Structure (3D)', fontsize=12)
        
        # Equal aspect ratio for better visualization
        ax.set_box_aspect([1, 1, 1])
        
        # Set initial view angle
        ax.view_init(30, 45)
        
    except Exception as e:
        logger.error(f"Error in 3D visualization: {e}")
        # Try a simpler 3D visualization approach as fallback
        try:
            # Create simple scatter plot visualization
            resolution = 20  # Lower resolution for fallback
            field_3d, _ = generate_soul_3d_field(soul_spark, resolution)
            
            # Create a mask for points to display
            mask = field_3d > 0.5
            x_idx, y_idx, z_idx = np.where(mask)
            
            # Convert indices to coordinates
            x_coords = x_idx / resolution * 2 - 1
            y_coords = y_idx / resolution * 2 - 1
            z_coords = z_idx / resolution * 2 - 1
            
            # Get colormap
            cmap = create_soul_colormap(soul_spark)
            colors = [cmap(field_3d[x, y, z]) for x, y, z in zip(x_idx, y_idx, z_idx)]
            
            # Plot
            ax.scatter(x_coords, y_coords, z_coords, c=colors, alpha=0.5, s=20)
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_title('Soul Structure (Simplified)')
            
        except Exception as fallback_error:
            logger.error(f"Even fallback 3D visualization failed: {fallback_error}")
            ax.text(0, 0, 0, "3D Visualization Failed", fontsize=12)
            ax.set_title('Soul Structure (Error)')

def visualize_top_aspects(soul_spark, ax) -> None:
    """Visualize top soul aspects with their strengths."""
    try:
        # Get top aspects by strength
        top_aspects = get_soul_aspects_by_strength(soul_spark, n_top=10)
        
        if not top_aspects:
            ax.text(0.5, 0.5, 'No Aspects Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_title('Soul Aspects')
            return
        
        # Unpack aspect names and strengths
        names = [a[0] for a in top_aspects]
        strengths = [a[1] for a in top_aspects]
        
        # Create colormap
        cmap = create_soul_colormap(soul_spark)
        colors = [cmap(s) for s in strengths]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        
        # Sort aspects by strength (highest at top)
        sorted_indices = np.argsort(strengths)
        names = [names[i] for i in sorted_indices]
        strengths = [strengths[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        
        # Plot horizontal bars with gradient colors and rounded caps
        bars = ax.barh(y_pos, strengths, color=colors, height=0.7, alpha=0.8)
        
        # Add value labels inside bars for strong aspects
        for i, v in enumerate(strengths):
            if v > 0.3:  # Only add text for significant values
                ax.text(v - 0.1, i, f"{v:.2f}", va='center', ha='right', 
                        color='white', fontweight='bold', fontsize=8)
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:20] for name in names])  # Truncate long names
        ax.set_xlabel('Strength')
        ax.set_title('Top Soul Aspects', fontsize=12)
        
        # Clean up the chart
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1.1)
        
    except Exception as e:
        logger.error(f"Error in top aspects visualization: {e}")
        raise RuntimeError(f"Failed to create top aspects visualization: {e}")

def visualize_harmony_factors(soul_spark, ax) -> None:
    """Visualize the harmony and resonance factors."""
    try:
        # Collect relevant factors
        factors = {}
        factor_names = [
            'stability', 'coherence', 'phi_resonance', 'pattern_coherence', 
            'harmony', 'resonance', 'toroidal_flow_strength', 'creator_alignment', 
            'cord_integrity', 'earth_resonance', 'physical_integration',
            'crystallization_level'
        ]
        
        # Get actual values or defaults
        for name in factor_names:
            if hasattr(soul_spark, name) and getattr(soul_spark, name) is not None:
                # Normalize based on expected range if possible
                value = getattr(soul_spark, name)
                if name == 'stability':
                    max_val = getattr(soul_spark, 'MAX_STABILITY_SU', 100.0)
                    norm_value = min(1.0, max(0.0, value / max_val))
                elif name == 'coherence':
                    max_val = getattr(soul_spark, 'MAX_COHERENCE_CU', 100.0)
                    norm_value = min(1.0, max(0.0, value / max_val))
                else:
                    # Assume 0-1 range for other factors
                    norm_value = min(1.0, max(0.0, value))
                    
                factors[name.replace('_', ' ').title()] = norm_value
        
        # If no factors found, show placeholder
        if not factors:
            ax.text(0.5, 0.5, 'Harmony Factors Not Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_title('Harmony Factors')
            return
            
        # Sort factors by value
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        # Unpack for plotting
        names, values = zip(*sorted_factors)
        
        # Create colormap
        cmap = create_soul_colormap(soul_spark)
        
        # Create horizontal bar chart with gradient colors
        y_pos = np.arange(len(names))
        gradient_colors = [cmap(v) for v in values]
        
        bars = ax.barh(y_pos, values, color=gradient_colors, height=0.7, alpha=0.8)
        
        # Add subtle gradient shading to bars
        for i, bar in enumerate(bars):
            # Get bar dimensions
            width = bar.get_width()
            height = bar.get_height()
            x = bar.get_x()
            y = bar.get_y()
            
            # Add small glowing effect
            if width > 0.5:
                # Only add glow to significant bars
                ax.axhspan(y, y+height, alpha=0.1, color=gradient_colors[i])
            
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=8)
            
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:15] for name in names])  # Truncate long names
        ax.set_xlim(0, 1.1)
        ax.set_title('Soul Harmony Factors', fontsize=12)
        
        # Clean up the chart
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    except Exception as e:
        logger.error(f"Error in harmony factors visualization: {e}")
        raise RuntimeError(f"Failed to create harmony factors visualization: {e}")

def visualize_sephiroth_influence(soul_spark, ax) -> None:
    """Visualize the soul's connection to different Sephiroth energies."""
    try:
        # Get Sephiroth influences
        influences = get_sephiroth_influence(soul_spark)
        
        if not influences:
            # Draw placeholder if no data
            ax.text(0.5, 0.5, 'Sephiroth Data Not Available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes)
            ax.set_title('Sephiroth Influence')
            return
            
        # Sort by influence
        sorted_influences = sorted(influences.items(), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_influences)
        
        # Get colors from Sephiroth color map
        colors = [SEPHIROTH_COLORS.get(name.lower(), '#AAAAAA') for name in names]
        
        # Create pie chart with subtle 3D effect
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=None,
            colors=colors, 
            autopct='%1.1f%%',
            pctdistance=0.85,
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.5, 'alpha': 0.7, 'width': 0.5},
            textprops={'color': 'white', 'fontsize': 8}
        )
        
        # Add some glow effects
        for i, wedge in enumerate(wedges):
            wedge.set_alpha(0.8)
        
        # Create custom legend with sephiroth names
        legend_elements = []
        for name, color in zip(names, colors):
            # Capitalize sephiroth name
            display_name = name.capitalize()
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=8, label=display_name))
        
        # Add legend with appropriate positioning
        ax.legend(handles=legend_elements, loc='center', fontsize=8, 
                 frameon=False, bbox_to_anchor=(0.5, -0.1))
        
        ax.set_title('Sephiroth Influence', fontsize=12)
        
    except Exception as e:
        logger.error(f"Error in Sephiroth influence visualization: {e}")
        raise RuntimeError(f"Failed to create Sephiroth influence visualization: {e}")

def visualize_soul_state(
    soul_spark, 
    stage_name: str, 
    output_dir: str = None, 
    show: bool = False
) -> str:
    """
    Create a comprehensive visualization of the soul's current state.
    Returns the path to the saved visualization file.
    Hard fails if visualization cannot be created.
    """
    logger.info(f"Creating visualization for soul {soul_spark.spark_id} at stage: {stage_name}")
    
    try:
        # Import necessary modules
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage import measure
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
        
        # Create a darker background for the figure
        fig.patch.set_facecolor('#121212')
        
        # Create grid layout with specific dimensions
        gs = plt.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Create named subplots
        ax_2d = fig.add_subplot(gs[0, 0])
        ax_freq = fig.add_subplot(gs[0, 1])
        ax_radar = fig.add_subplot(gs[0, 2], polar=True)
        ax_3d = fig.add_subplot(gs[1, :], projection='3d')
        ax_aspects = fig.add_subplot(gs[2, :2])
        ax_sephiroth = fig.add_subplot(gs[2, 2])
        
        # Set dark background for all axes
        for ax in [ax_2d, ax_freq, ax_radar, ax_3d, ax_aspects, ax_sephiroth]:
            ax.set_facecolor('#121212')
            for spine in ax.spines.values():
                spine.set_color('#333333')
            ax.tick_params(colors='#CCCCCC')
            ax.xaxis.label.set_color('#CCCCCC')
            ax.yaxis.label.set_color('#CCCCCC')
            ax.title.set_color('#FFFFFF')
        
        # Create visualizations
        visualize_density_2d(soul_spark, ax_2d)
        visualize_frequency_spectrum(soul_spark, ax_freq)
        visualize_aspects_radar(soul_spark, ax_radar)
        visualize_soul_3d(soul_spark, ax_3d)
        visualize_top_aspects(soul_spark, ax_aspects)
        visualize_sephiroth_influence(soul_spark, ax_sephiroth)
        
        # Add metadata to figure
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        soul_id = soul_spark.spark_id
        title = f"Soul State: {soul_id} - {stage_name} ({timestamp})"
        
        # Add stability and coherence values if available
        try:
            s_val = getattr(soul_spark, 'stability', None)
            c_val = getattr(soul_spark, 'coherence', None)
            if s_val is not None and c_val is not None:
                title += f"\nStability: {s_val:.1f} SU | Coherence: {c_val:.1f} CU"
        except Exception:
            pass
            
        fig.suptitle(title, fontsize=16, color='white')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout with room for title
        
        # Save visualization
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{soul_spark.spark_id}_{stage_name.replace(' ','_')}_{timestamp_str}.png"
        if output_dir:
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
            
        plt.savefig(filepath, dpi=DEFAULT_DPI, facecolor='#121212')
        logger.info(f"Visualization saved to {filepath}")
        
        # Also save a NumPy version of the soul state for later analysis
        try:
            data_save_dir = os.path.join(output_dir, "../completed") if output_dir else "output/completed"
            os.makedirs(data_save_dir, exist_ok=True)
            
            # Capture key state data for memory-efficient storage
            state_data = {
                'soul_id': soul_spark.spark_id,
                'stage': stage_name,
                'timestamp': timestamp_str,
                'density_2d': generate_soul_density_field(soul_spark, resolution=50),  # Lower resolution for storage
                'stability': getattr(soul_spark, 'stability', 0.0),
                'coherence': getattr(soul_spark, 'coherence', 0.0),
                'frequency': getattr(soul_spark, 'frequency', 0.0),
                'aspects_count': len(getattr(soul_spark, 'aspects', {})),
                'layers_count': len(getattr(soul_spark, 'layers', [])),
            }
            
            # Save numpy data
            data_filename = f"{soul_spark.spark_id}_{stage_name.replace(' ','_')}_{timestamp_str}.npy"
            data_filepath = os.path.join(data_save_dir, data_filename)
            np.save(data_filepath, state_data)
            logger.info(f"Soul state data saved to {data_filepath}")
        except Exception as data_e:
            logger.warning(f"Failed to save soul state data: {data_e}")
        
        # Show visualization if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in soul visualization: {e}", exc_info=True)
        plt.close('all')  # Clean up any open figures
        raise RuntimeError(f"Failed to create soul visualization: {e}")

# --- Helper Function for State Comparison ---
def visualize_state_comparison(
    soul_spark_states: List[Tuple[Any, str]], 
    output_dir: str = None,
    show: bool = False
) -> str:
    """
    Compare multiple soul states across different development stages.
    
    Args:
        soul_spark_states: List of tuples (soul_spark, stage_name) to compare
        output_dir: Directory to save visualization
        show: Whether to display the visualization
        
    Returns:
        Path to saved visualization
    """
    if not soul_spark_states or len(soul_spark_states) < 2:
        logger.error("Need at least two soul states to compare")
        raise ValueError("Need at least two soul states to compare")
        
    logger.info(f"Creating comparison visualization for {len(soul_spark_states)} soul states")
    
    try:
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Get soul ID from first state
        soul_id = soul_spark_states[0][0].spark_id
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12), dpi=DEFAULT_DPI)
        fig.patch.set_facecolor('#121212')
        
        # Get a consistent colormap for this soul
        cmap = create_soul_colormap(soul_spark_states[0][0])
        
        # Setup grid
        n_states = len(soul_spark_states)
        
        # Extract stages for x-axis
        stages = [stage for _, stage in soul_spark_states]
        
        # 1. Stability and Coherence progression - Upper left
        ax_sc = fig.add_subplot(2, 2, 1)
        ax_sc.set_facecolor('#121212')
        
        stability_values = []
        coherence_values = []
        
        for soul, _ in soul_spark_states:
            stability_values.append(getattr(soul, 'stability', 0.0))
            coherence_values.append(getattr(soul, 'coherence', 0.0))
            
        x = range(len(stages))
        
        # Plot stability with gradient line
        color1 = cmap(0.3)
        for i in range(len(x)-1):
            ax_sc.plot(x[i:i+2], stability_values[i:i+2], '-', 
                      color=color1, linewidth=2.5, alpha=0.8)
            ax_sc.plot(x[i:i+2], stability_values[i:i+2], '-', 
                      color='white', linewidth=1, alpha=0.3)
                      
        # Plot coherence with gradient line
        color2 = cmap(0.7)
        for i in range(len(x)-1):
            ax_sc.plot(x[i:i+2], coherence_values[i:i+2], '-', 
                      color=color2, linewidth=2.5, alpha=0.8)
            ax_sc.plot(x[i:i+2], coherence_values[i:i+2], '-', 
                      color='white', linewidth=1, alpha=0.3)
        
        # Add markers
        ax_sc.scatter(x, stability_values, s=80, color=color1, alpha=0.9, 
                     edgecolor='white', linewidth=1, zorder=10)
        ax_sc.scatter(x, coherence_values, s=80, color=color2, alpha=0.9, 
                     edgecolor='white', linewidth=1, zorder=10)
        
        # Styling
        ax_sc.set_xticks(x)
        ax_sc.set_xticklabels(stages, rotation=45, ha='right', color='#CCCCCC')
        ax_sc.set_ylabel('Value', color='#CCCCCC')
        ax_sc.set_title('Stability and Coherence Progression', color='white', fontsize=14)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color1, 
                  markersize=8, label='Stability (SU)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color2, 
                  markersize=8, label='Coherence (CU)')
        ]
        ax_sc.legend(handles=legend_elements, loc='upper left', frameon=False, 
                    labelcolor='white')
        
        # Add subtle grid
        ax_sc.grid(True, linestyle='--', alpha=0.2, color='#555555')
        for spine in ax_sc.spines.values():
            spine.set_color('#333333')
        ax_sc.tick_params(colors='#CCCCCC')
        
        # 2. Soul Energy Density Evolution - Upper right
        ax_density = fig.add_subplot(2, 2, 2)
        ax_density.set_facecolor('#121212')
        
        # Create a series of small density plots showing evolution
        n_cols = len(soul_spark_states)
        grid_size = 50  # Small grid for each stage
        
        # Create a grid to hold all density plots
        full_grid = np.zeros((grid_size, grid_size * n_cols))
        
        # Fill the grid with density plots
        for i, (soul, _) in enumerate(soul_spark_states):
            density = generate_soul_density_field(soul, resolution=grid_size)
            full_grid[:, i*grid_size:(i+1)*grid_size] = density
            
        # Plot with custom colormap
        extent = [0, n_cols, 0, 1]
        im = ax_density.imshow(full_grid, extent=extent, origin='lower', 
                             cmap=cmap, aspect='auto')
                             
        # Add stage labels
        for i in range(n_cols):
            ax_density.text(i + 0.5, 0.05, stages[i], 
                          ha='center', va='bottom', color='white',
                          fontsize=8, rotation=90)
            
        # Add vertical lines between stages
        for i in range(1, n_cols):
            ax_density.axvline(i, color='white', linewidth=0.5, alpha=0.3)
            
        ax_density.set_title('Soul Energy Evolution', color='white', fontsize=14)
        ax_density.set_yticks([])
        ax_density.set_xticks([])
        
        # 3. Aspect Count and Layer Count - Lower left
        ax_counts = fig.add_subplot(2, 2, 3)
        ax_counts.set_facecolor('#121212')
        
        # Get aspect and layer counts
        aspect_counts = []
        layer_counts = []
        
        for soul, _ in soul_spark_states:
            aspect_counts.append(len(getattr(soul, 'aspects', {})))
            layer_counts.append(len(getattr(soul, 'layers', [])))
            
        # Create double bar chart with gradient fill
        bar_width = 0.35
        r1 = np.arange(len(stages))
        r2 = [x + bar_width for x in r1]
        
        # Custom colors with alpha gradient
        aspect_color = cmap(0.9)
        layer_color = cmap(0.4)
        
        # Create bars with lighter edges
        aspect_bars = ax_counts.bar(r1, aspect_counts, width=bar_width, 
                                  color=aspect_color, alpha=0.8,
                                  edgecolor='white', linewidth=0.5,
                                  label='Aspects')
        layer_bars = ax_counts.bar(r2, layer_counts, width=bar_width, 
                                 color=layer_color, alpha=0.8,
                                 edgecolor='white', linewidth=0.5,
                                 label='Layers')
        
        # Add count labels above bars
        for i, v in enumerate(aspect_counts):
            ax_counts.text(r1[i], v + 0.3, str(v), ha='center', va='bottom', 
                         color='white', fontsize=9)
        for i, v in enumerate(layer_counts):
            ax_counts.text(r2[i], v + 0.3, str(v), ha='center', va='bottom', 
                         color='white', fontsize=9)
        
        # Add styling
        ax_counts.set_xticks([r + bar_width/2 for r in range(len(stages))])
        ax_counts.set_xticklabels(stages, rotation=45, ha='right', color='#CCCCCC')
        ax_counts.set_ylabel('Count', color='#CCCCCC')
        ax_counts.set_title('Aspects and Layers Growth', color='white', fontsize=14)
        ax_counts.legend(frameon=False, labelcolor='white')
        
        # Add subtle grid
        ax_counts.grid(True, linestyle='--', alpha=0.2, color='#555555', axis='y')
        for spine in ax_counts.spines.values():
            spine.set_color('#333333')
        ax_counts.tick_params(colors='#CCCCCC')
        
        # 4. Harmony Factors Comparison - Lower right
        ax_harmony = fig.add_subplot(2, 2, 4)
        ax_harmony.set_facecolor('#121212')
        
        # Get harmony factors for each state
        factor_names = [
            'stability', 'coherence', 'phi_resonance', 'pattern_coherence', 
            'harmony', 'resonance', 'toroidal_flow_strength'
        ]
        
        # Collect factor data across states
        factor_data = {name: [] for name in factor_names}
        
        for soul, _ in soul_spark_states:
            for name in factor_names:
                if hasattr(soul, name) and getattr(soul, name) is not None:
                    # Normalize value to 0-1 range
                    value = getattr(soul, name)
                    if name == 'stability':
                        max_val = getattr(soul, 'MAX_STABILITY_SU', 100.0)
                        norm_value = min(1.0, max(0.0, value / max_val))
                    elif name == 'coherence':
                        max_val = getattr(soul, 'MAX_COHERENCE_CU', 100.0)
                        norm_value = min(1.0, max(0.0, value / max_val))
                    else:
                        # Assume 0-1 range for other factors
                        norm_value = min(1.0, max(0.0, value))
                    factor_data[name].append(norm_value)
                else:
                    factor_data[name].append(0.0)
        
        # Plot radar chart for comparison
        # Keep only factors with data
        valid_factors = [name for name in factor_names if any(v > 0 for v in factor_data[name])]
        
        if valid_factors:
            # Set number of variables
            N = len(valid_factors)
            
            # Create angle array
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Create new axes with polar projection
            ax_harmony = plt.subplot(2, 2, 4, polar=True)
            ax_harmony.set_facecolor('#121212')
            
            # Draw one line per stage and fill area
            for i, (soul, stage) in enumerate(soul_spark_states):
                # Prepare data
                values = [factor_data[name][i] for name in valid_factors]
                values += values[:1]  # Close the loop
                
                # Get color from colormap with different saturation for each stage
                color = cmap(0.2 + 0.8 * (i / max(1, len(soul_spark_states) - 1)))
                
                # Plot line and fill
                ax_harmony.plot(angles, values, linewidth=2, linestyle='-', 
                             color=color, alpha=0.8, label=stage)
                ax_harmony.fill(angles, values, color=color, alpha=0.1)
            
            # Draw labels
            ax_harmony.set_xticks(angles[:-1])
            ax_harmony.set_xticklabels([name.replace('_', ' ').capitalize() for name in valid_factors], 
                                     color='#CCCCCC', fontsize=8)
            
            # Draw ylabels
            ax_harmony.set_rlabel_position(0)
            ax_harmony.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_harmony.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color='#CCCCCC')
            ax_harmony.set_ylim(0, 1)
            
            # Add legend
            ax_harmony.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), 
                           frameon=False, labelcolor='white')
            
            # Add title
            ax_harmony.set_title('Harmony Factors Comparison', color='white', fontsize=14, pad=15)
            
            # Add subtle gridlines
            ax_harmony.grid(color='#555555', alpha=0.2)
        else:
            # No valid factors, show text instead
            ax_harmony.text(0.5, 0.5, 'Harmony Factor Data Not Available', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_harmony.transAxes, color='white')
            ax_harmony.set_title('Harmony Factors Comparison', color='white')
        
        # Add metadata to figure
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = f"Soul Development Comparison: {soul_id}\n({timestamp})"
        fig.suptitle(title, fontsize=16, color='white')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout with room for title
        
        # Save visualization
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{soul_id}_Development_Comparison_{timestamp_str}.png"
        if output_dir:
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
            
        plt.savefig(filepath, dpi=DEFAULT_DPI, facecolor='#121212')
        logger.info(f"Comparison visualization saved to {filepath}")
        
        # Show visualization if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in comparison visualization: {e}", exc_info=True)
        plt.close('all')  # Clean up any open figures
        raise RuntimeError(f"Failed to create comparison visualization: {e}")


# Additional utility function to generate a beautiful soul signature pattern
def generate_soul_signature(soul_spark, resolution=200):
    """
    Generates a unique visual signature pattern for the soul based on its properties.
    This creates an artistic representation unique to each soul.
    """
    try:
        # Get soul properties
        stability = getattr(soul_spark, 'stability', 50.0) / 100.0
        coherence = getattr(soul_spark, 'coherence', 50.0) / 100.0
        frequency = getattr(soul_spark, 'frequency', 432.0)
        
        # Create coordinate grid
        t = np.linspace(0, 2*np.pi, resolution)
        
        # Base parameters derived from soul properties
        base_radius = 0.3 + 0.2 * stability
        complexity = 3 + 10 * coherence
        variation = 0.1 + 0.3 * coherence
        
        # Create frequency-based parameters
        freq_factor = (frequency % 100) / 100.0  # Normalize to 0-1
        phase_shift = 2 * np.pi * freq_factor
        
        # Generate unique pattern based on soul ID
        id_seed = 0
        try:
            # Convert spark_id to a numeric seed
            id_str = str(soul_spark.spark_id)
            id_seed = sum(ord(c) for c in id_str) / 1000.0
        except:
            id_seed = 0.5
        
        # Create harmonic components
        r = base_radius
        for i in range(1, int(complexity) + 1):
            harmonic = variation * np.sin(i * t + phase_shift * i + id_seed * i)
            r += harmonic / i  # Higher harmonics contribute less
        
        # Convert to cartesian coordinates
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        return x, y, t, r
        
    except Exception as e:
        logger.error(f"Error generating soul signature: {e}")
        # Return a simple circle as fallback
        t = np.linspace(0, 2*np.pi, resolution)
        r = 0.5 * np.ones_like(t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return x, y, t, r

def visualize_soul_signature(soul_spark, ax=None, with_title=True):
    """
    Creates a beautiful artistic visualization of the soul's unique signature pattern.
    This can be used as a visual identifier for the soul.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=DEFAULT_DPI)
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
    
    try:
        # Generate soul signature pattern
        x, y, t, r = generate_soul_signature(soul_spark)
        
        # Get soul properties for coloring
        stability = getattr(soul_spark, 'stability', 50.0) / 100.0
        coherence = getattr(soul_spark, 'coherence', 50.0) / 100.0
        
        # Create color gradient based on soul spectrum
        cmap = create_soul_colormap(soul_spark)
        colors = cmap(np.linspace(0, 1, len(t)))
        
        # Plot with beautiful styling
        points = np.array([x, y]).T.reshape((-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create line collection with variable colors
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8)
        line = ax.add_collection(lc)
        
        # Add subtle glow effect
        for alpha, width in zip([0.1, 0.05, 0.02], [4, 6, 8]):
            lc_glow = LineCollection(segments, colors=colors, linewidth=width, alpha=alpha)
            ax.add_collection(lc_glow)
        
        # Add center marker
        ax.scatter(0, 0, color='white', s=50, alpha=0.8, zorder=10)
        
        # Add radial lines for more complex souls
        if coherence > 0.6:
            n_lines = int(5 + 5 * coherence)
            for i in range(n_lines):
                theta = i * 2 * np.pi / n_lines
                line_x = [0, 0.9 * np.cos(theta)]
                line_y = [0, 0.9 * np.sin(theta)]
                ax.plot(line_x, line_y, color='white', alpha=0.1, linewidth=0.5)
        
        # Add circular markers along the path based on stability
        if stability > 0.4:
            # More stable souls have more defined nodes
            n_markers = int(4 + 8 * stability)
            marker_indices = np.linspace(0, len(x) - 1, n_markers, dtype=int)
            ax.scatter(x[marker_indices], y[marker_indices], color='white', 
                     s=30, alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Set limits and aspect
        margin = 0.1
        max_range = max(np.max(np.abs(x)), np.max(np.abs(y))) + margin
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')
        
        # Remove axes
        ax.axis('off')
        
        # Add title if requested
        if with_title:
            soul_id = getattr(soul_spark, 'spark_id', 'Unknown')
            soul_name = getattr(soul_spark, 'name', None)
            title = f"Soul Signature: {soul_id}"
            if soul_name:
                title += f" ({soul_name})"
            ax.set_title(title, color='white', fontsize=14)
        
        return ax
        
    except Exception as e:
        logger.error(f"Error visualizing soul signature: {e}")
        # Draw a simple placeholder
        ax.text(0.5, 0.5, 'Soul Signature Unavailable', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, color='white')
        ax.set_title('Soul Signature', color='white')
        return ax

def visualize_earth_resonance(soul_spark, ax=None):
    """
    Visualizes the soul's resonance with Earth frequencies and elements.
    Shows connection strength to different elemental and planetary energies.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=DEFAULT_DPI)
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
    
    try:
        # Collect Earth resonance data
        resonance_data = {}
        
        # Check for elemental affinities
        if hasattr(soul_spark, 'elemental_affinities') and soul_spark.elemental_affinities:
            for element, value in soul_spark.elemental_affinities.items():
                resonance_data[f"Element: {element.capitalize()}"] = float(value)
        
        # Check for planetary resonance
        if hasattr(soul_spark, 'planetary_resonance') and soul_spark.planetary_resonance:
            for planet, value in soul_spark.planetary_resonance.items():
                resonance_data[f"Planet: {planet.capitalize()}"] = float(value)
        
        # Check for Earth resonance value
        earth_res = getattr(soul_spark, 'earth_resonance', None)
        if earth_res is not None:
            resonance_data['Earth Resonance'] = float(earth_res)
        
        # Check for Schumann resonance
        schumann_res = getattr(soul_spark, 'schumann_resonance_alignment', None)
        if schumann_res is not None:
            resonance_data['Schumann Resonance'] = float(schumann_res)
            
        # If no data available, show placeholder
        if not resonance_data:
            ax.text(0.5, 0.5, 'Earth Resonance Data Not Available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, color='white')
            ax.set_title('Earth Resonance', color='white')
            return ax
        
        # Sort by value
        sorted_items = sorted(resonance_data.items(), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_items)
        
        # Create color map based on categories
        colors = []
        for name in names:
            if 'element' in name.lower():
                colors.append('#4CAF50')  # Green for elements
            elif 'planet' in name.lower():
                colors.append('#2196F3')  # Blue for planets
            elif 'schumann' in name.lower():
                colors.append('#FFC107')  # Yellow for Schumann
            else:
                colors.append('#9C27B0')  # Purple for other
        
        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=colors, height=0.7, alpha=0.8)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(max(0.02, v - 0.15) if v > 0.3 else v + 0.02, 
                   i, f"{v:.2f}", va='center', 
                   color='white' if v > 0.3 else '#CCCCCC',
                   fontsize=9)
        
        # Add category coloring in the y-tick labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color='#CCCCCC')
        
        # Add title and clean up axes
        ax.set_title('Earth & Elemental Resonance', color='white', fontsize=14)
        ax.set_xlabel('Resonance Strength', color='#CCCCCC')
        ax.set_xlim(0, 1.1)
        
        # Clean up the chart
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        ax.tick_params(colors='#CCCCCC')
        
        # Add subtle grid
        ax.grid(True, linestyle='--', alpha=0.2, color='#555555', axis='x')
        
        return ax
        
    except Exception as e:
        logger.error(f"Error visualizing Earth resonance: {e}")
        # Draw placeholder on error
        ax.text(0.5, 0.5, 'Earth Resonance Visualization Failed', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, color='white')
        ax.set_title('Earth Resonance', color='white')
        return ax

def create_comprehensive_soul_report(
    soul_spark, 
    stage_name: str, 
    output_dir: str = None, 
    show: bool = False
) -> str:
    """
    Creates a comprehensive soul development report with all visualizations.
    Returns the path to the saved report file.
    """
    logger.info(f"Creating comprehensive report for soul {soul_spark.spark_id} at stage: {stage_name}")
    
    try:
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create larger figure with more visualizations
        fig = plt.figure(figsize=(20, 16), dpi=DEFAULT_DPI)
        fig.patch.set_facecolor('#121212')
        
        # Create grid layout
        gs = plt.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Create all visualization panels
        ax_2d = fig.add_subplot(gs[0, 0])
        ax_freq = fig.add_subplot(gs[0, 1])
        ax_radar = fig.add_subplot(gs[0, 2], polar=True)
        ax_signature = fig.add_subplot(gs[0, 3])
        ax_3d = fig.add_subplot(gs[1, :2], projection='3d')
        ax_earth = fig.add_subplot(gs[1, 2:])
        ax_aspects = fig.add_subplot(gs[2, :2])
        ax_harmony = fig.add_subplot(gs[2, 2:])
        ax_sephiroth = fig.add_subplot(gs[3, 0])
        ax_layers = fig.add_subplot(gs[3, 1:3])
        ax_info = fig.add_subplot(gs[3, 3])
        
        # Set dark background for all axes
        for ax in fig.get_axes():
            ax.set_facecolor('#121212')
            if hasattr(ax, 'spines'):
                for spine in ax.spines.values():
                    spine.set_color('#333333')
            ax.tick_params(colors='#CCCCCC')
            if hasattr(ax, 'xaxis') and hasattr(ax.xaxis, 'label'):
                ax.xaxis.label.set_color('#CCCCCC')
            if hasattr(ax, 'yaxis') and hasattr(ax.yaxis, 'label'):
                ax.yaxis.label.set_color('#CCCCCC')
            if hasattr(ax, 'title'):
                ax.title.set_color('#FFFFFF')
        
        # Create all visualizations
        visualize_density_2d(soul_spark, ax_2d)
        visualize_frequency_spectrum(soul_spark, ax_freq)
        visualize_aspects_radar(soul_spark, ax_radar)
        visualize_soul_signature(soul_spark, ax_signature)
        visualize_soul_3d(soul_spark, ax_3d)
        visualize_earth_resonance(soul_spark, ax_earth)
        visualize_top_aspects(soul_spark, ax_aspects)
        visualize_harmony_factors(soul_spark, ax_harmony)
        visualize_sephiroth_influence(soul_spark, ax_sephiroth)
        
        # Additional visualizations can be added here
        
        # Create soul information text panel
        ax_info.axis('off')
        info_text = get_soul_info_text(soul_spark)
        ax_info.text(0.1, 0.9, info_text, color='white', fontsize=10,
                   va='top', ha='left', linespacing=1.5)
        ax_info.set_title('Soul Information', color='white')
        
        # Add metadata to figure
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        soul_id = soul_spark.spark_id
        title = f"Soul Development Report: {soul_id} - {stage_name}\n({timestamp})"
        
        fig.suptitle(title, fontsize=18, color='white')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout with room for title
        
        # Save report
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{soul_spark.spark_id}_Report_{stage_name.replace(' ','_')}_{timestamp_str}.png"
        if output_dir:
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
            
        plt.savefig(filepath, dpi=DEFAULT_DPI, facecolor='#121212')
        logger.info(f"Comprehensive report saved to {filepath}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in comprehensive report creation: {e}", exc_info=True)
        plt.close('all')  # Clean up any open figures
        raise RuntimeError(f"Failed to create comprehensive report: {e}")

def get_soul_info_text(soul_spark):
    """Create a formatted text summary of the soul's information."""
    info = []
    # Basic soul info
    info.append(f"Soul ID: {soul_spark.spark_id}")
    if hasattr(soul_spark, 'name'):
        info.append(f"Name: {soul_spark.name}")
    
    # Core metrics
    info.append("\nCore Metrics:")
    if hasattr(soul_spark, 'stability'):
        max_stability = getattr(soul_spark, 'MAX_STABILITY_SU', 100.0)
        info.append(f"Stability: {soul_spark.stability:.2f} SU/{max_stability:.0f} SU")
    if hasattr(soul_spark, 'coherence'):
        max_coherence = getattr(soul_spark, 'MAX_COHERENCE_CU', 100.0)
        info.append(f"Coherence: {soul_spark.coherence:.2f} CU/{max_coherence:.0f} CU")
    if hasattr(soul_spark, 'frequency'):
        info.append(f"Frequency: {soul_spark.frequency:.2f} Hz")
    if hasattr(soul_spark, 'energy'):
        max_energy = getattr(soul_spark, 'MAX_SOUL_ENERGY_SEU', 100.0)
        info.append(f"Energy: {soul_spark.energy:.2f} SEU/{max_energy:.0f} SEU")
    
    # Formation details
    info.append("\nFormation Details:")
    if hasattr(soul_spark, 'creation_datetime'):
        info.append(f"Created: {soul_spark.creation_datetime}")
    if hasattr(soul_spark, 'conceptual_birth_datetime'):
        info.append(f"Birth: {soul_spark.conceptual_birth_datetime}")
    if hasattr(soul_spark, 'layers'):
        info.append(f"Layers: {len(soul_spark.layers)}")
    if hasattr(soul_spark, 'aspects'):
        info.append(f"Aspects: {len(soul_spark.aspects)}")
    
    # Harmony factors
    info.append("\nHarmony Factors:")
    harmony_attrs = [
        'phi_resonance', 'pattern_coherence', 'resonance', 
        'toroidal_flow_strength', 'creator_alignment', 
        'earth_resonance', 'physical_integration'
    ]
    for attr in harmony_attrs:
        if hasattr(soul_spark, attr) and getattr(soul_spark, attr) is not None:
            value = getattr(soul_spark, attr)
            info.append(f"{attr.replace('_', ' ').title()}: {value:.2f}")
    
    # Primary sephiroth influence
    if hasattr(soul_spark, 'sephiroth_aspect'):
        info.append(f"\nPrimary Sephiroth: {soul_spark.sephiroth_aspect.capitalize()}")
    
    # Additional properties
    if hasattr(soul_spark, 'soul_color'):
        info.append(f"Soul Color: {soul_spark.soul_color}")
    if hasattr(soul_spark, 'consciousness_state'):
        info.append(f"Consciousness: {soul_spark.consciousness_state}")
    
    return "\n".join(info)

# If imported as a module, set up the module
if __name__ == "__main__":
    # Simple test code for debugging
    from unittest.mock import MagicMock
    
    # Create mock soul for testing
    mock_soul = MagicMock()
    mock_soul.spark_id = "TEST-SOUL-001"
    mock_soul.stability = 65.0
    mock_soul.coherence = 70.0
    mock_soul.frequency = 432.0
    mock_soul.energy = 50.0
    mock_soul.aspects = {
        "wisdom": {"strength": 0.8, "type": "spiritual"},
        "love": {"strength": 0.9, "type": "emotional"},
        "courage": {"strength": 0.7, "type": "willpower"},
        "insight": {"strength": 0.6, "type": "spiritual"},
        "logic": {"strength": 0.5, "type": "intellectual"}
    }
    mock_soul.layers = [
        {"sephirah": "kether", "density": 0.9},
        {"sephirah": "chokmah", "density": 0.8},
        {"sephirah": "binah", "density": 0.7}
    ]
    mock_soul.sephiroth_aspect = "tiphareth"
    mock_soul.soul_color = "#8A2BE2"  # BlueViolet
    
    # Test visualization
    output_path = visualize_soul_state(mock_soul, "Testing", output_dir="output", show=True)
    print(f"Test visualization saved to: {output_path}")







