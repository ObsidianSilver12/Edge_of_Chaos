"""
Earth Harmonization Functions (FIXED VERSION - V4.4.1 - Proper Earth Resonance Calculation)

"""

"""
Earth Harmonization Functions (Refactored V4.4.0 - Wave Physics & Aura Integration)

Establishes resonant connection with Mother Earth/Gaia using standing wave physics.
Creates resonant chambers in aura layers rather than directly modifying frequency.
Implements proper Schumann resonance principles and acoustic energy transfer.
Uses light-based information exchange with planetary and Earth energetic systems.
Modifies SoulSpark directly. Uses constants. Hard fails.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from math import sqrt, exp, sin, cos, pi as PI, atan2, tanh
import time
import random
import uuid
from typing import Dict, List, Any, Tuple, Optional
from shared.constants.constants import *  # type: ignore

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from .creator_entanglement import calculate_resonance
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import core dependencies: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e
    
# --- Sound Module Imports ---
try:
    from shared.sound.sound_generator import SoundGenerator
    from shared.sound.sounds_of_universe import UniverseSounds
    from shared.sound.noise_generator import NoiseGenerator
    SOUND_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Sound modules not available: {e}. Earth harmonization will use simulated sound.")
    SOUND_MODULES_AVAILABLE = False

# Import aspect dictionary
try:
    from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
except ImportError:
    aspect_dictionary = None  # Optional for element mapping
    logging.warning("Sephiroth aspect dictionary not available. Element mapping will be limited.")

try:
    pass  # Keep error handling structure
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()

# --- Helper Functions ---

def find_best_harmonic_resonance(freq1: float, freq2: float) -> float:
    """
    Find the best harmonic resonance between two frequencies using musical ratios.
    This preserves the integrity of both frequencies by finding their natural
    harmonic relationships rather than forcing direct matching.
    """
    if freq1 <= FLOAT_EPSILON or freq2 <= FLOAT_EPSILON:
        return 0.0
        
    # Musical intervals (perfect harmonics)
    ratios = [1.0, 1.25, 1.333, 1.5, 1.667, 2.0, 0.5, 0.667, 0.75, 0.8]
    
    best_resonance = 0.0
    for ratio in ratios:
        # Check both directions for harmonics
        harmonic1 = freq1 * ratio
        harmonic2 = freq2 * ratio
        
        # Calculate resonance with both potential matches
        res1 = 1.0 - min(1.0, abs(harmonic1 - freq2) / freq2)
        res2 = 1.0 - min(1.0, abs(harmonic2 - freq1) / freq1)
        
        # Use the best resonance found
        best_resonance = max(best_resonance, res1, res2)
        
    return best_resonance

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds and cord integrity factor. Raises ValueError on failure. """
    logger.debug(f"Checking Earth Attunement prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("Invalid SoulSpark object.")

    if not getattr(soul_spark, FLAG_READY_FOR_EARTH, False): # Set by Life Cord
        msg = f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_EARTH}."; logger.error(msg); raise ValueError(msg)

    cord_integrity = getattr(soul_spark, "cord_integrity", -1.0)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if cord_integrity < 0 or stability_su < 0 or coherence_cu < 0:
        msg = "Prerequisite failed: Soul missing cord_integrity, stability or coherence."; logger.error(msg); raise AttributeError(msg)

    if cord_integrity < HARMONY_PREREQ_CORD_INTEGRITY_MIN:
        msg = f"Prerequisite failed: Cord integrity ({cord_integrity:.3f}) < {HARMONY_PREREQ_CORD_INTEGRITY_MIN}."; logger.error(msg); raise ValueError(msg)
    if stability_su < HARMONY_PREREQ_STABILITY_MIN_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {HARMONY_PREREQ_STABILITY_MIN_SU} SU."; logger.error(msg); raise ValueError(msg)
    if coherence_cu < HARMONY_PREREQ_COHERENCE_MIN_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {HARMONY_PREREQ_COHERENCE_MIN_CU} CU."; logger.error(msg); raise ValueError(msg)

    if getattr(soul_spark, FLAG_EARTH_ATTUNED, False) or getattr(soul_spark, FLAG_ECHO_PROJECTED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_ECHO_PROJECTED}/{FLAG_EARTH_ATTUNED}. Re-running.")

    logger.debug("Earth Attunement prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. Initializes if missing. """
    logger.debug(f"Ensuring properties for earth harmonization (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'aspects', 'earth_resonance', 
                'planetary_resonance', 'gaia_connection', 'layers', 'cord_integrity']
    
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Earth Harmonization: {missing}")
    
    # Initialize soul_color if missing
    if not hasattr(soul_spark, 'soul_color') or soul_spark.soul_color is None:
        # Generate a default color based on frequency
        freq = soul_spark.frequency
        # Map frequency range to RGB hue (0-360)
        hue = (freq % 360)
        # Convert HSV to RGB (simplified - assumes S=1, V=1)
        h_prime = hue / 60.0
        x = 1.0 - abs(h_prime % 2.0 - 1.0)
        
        if 0 <= h_prime < 1: rgb = [1.0, x, 0.0]
        elif 1 <= h_prime < 2: rgb = [x, 1.0, 0.0]
        elif 2 <= h_prime < 3: rgb = [0.0, 1.0, x]
        elif 3 <= h_prime < 4: rgb = [0.0, x, 1.0]
        elif 4 <= h_prime < 5: rgb = [x, 0.0, 1.0]
        else: rgb = [1.0, 0.0, x]
        
        # Scale to 0-255 and convert to hex
        r, g, b = [int(255 * c) for c in rgb]
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        
        # Set the soul color
        soul_spark.soul_color = color_hex
        logger.debug(f"  Initialized soul_color to {color_hex} (based on frequency {freq:.1f}Hz)")
    
    # Initialize elements and earth_cycles dictionaries if missing
    if not hasattr(soul_spark, 'elements') or not isinstance(soul_spark.elements, dict):
        soul_spark.elements = {'earth': 0.2, 'air': 0.2, 'fire': 0.2, 'water': 0.2, 'aether': 0.2}
    
    if not hasattr(soul_spark, 'earth_cycles') or not isinstance(soul_spark.earth_cycles, dict):
        soul_spark.earth_cycles = {cycle_name: 0.2 for cycle_name in HARMONY_CYCLE_NAMES}
    
    # Ensure layer resonance properties exist
    for i, layer in enumerate(soul_spark.layers):
        if not isinstance(layer, dict):
            continue
            
        # Initialize earth_resonance in layer if needed
        if 'earth_resonance' not in layer:
            layer['earth_resonance'] = {
                'schumann': 0.0,
                'core': 0.0,
                'total': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # FIXED: Ensure all required keys exist in existing earth_resonance
            if 'schumann' not in layer['earth_resonance']:
                layer['earth_resonance']['schumann'] = 0.0
            if 'core' not in layer['earth_resonance']:
                layer['earth_resonance']['core'] = 0.0
            if 'total' not in layer['earth_resonance']:
                layer['earth_resonance']['total'] = 0.0
            if 'timestamp' not in layer['earth_resonance']:
                layer['earth_resonance']['timestamp'] = datetime.now().isoformat()
        
        # Initialize resonant_chambers in layer if needed
        if 'resonant_chambers' not in layer:
            layer['resonant_chambers'] = []
    
    # Ensure planetary_resonance and gaia_connection are initialized
    if not isinstance(soul_spark.planetary_resonance, (int, float)) or soul_spark.planetary_resonance < 0:
        soul_spark.planetary_resonance = 0.0
    
    if not isinstance(soul_spark.gaia_connection, (int, float)) or soul_spark.gaia_connection < 0:
        soul_spark.gaia_connection = 0.0
    
    logger.debug("Soul properties ensured for Earth Harmonization.")

def _create_resonant_layer_chamber(soul_spark: SoulSpark, layer_idx: int, 
                                 target_freq: float, chamber_type: str,
                                 initial_resonance: float = 0.0) -> Dict[str, Any]:
    """
    Creates a resonant chamber in a specific layer tuned to a target frequency.
    This is the mechanism for layers to resonate with Earth frequencies.
    Returns the chamber data including its resonance properties.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= len(soul_spark.layers):
        raise ValueError(f"Invalid layer index: {layer_idx}")
    if not isinstance(target_freq, (int, float)) or target_freq <= FLOAT_EPSILON:
        raise ValueError(f"Invalid target frequency: {target_freq}")
    if not isinstance(chamber_type, str) or not chamber_type:
        raise ValueError("Chamber type must be a non-empty string.")
    
    layer = soul_spark.layers[layer_idx]
    if not isinstance(layer, dict):
        raise TypeError(f"Layer at index {layer_idx} is not a dictionary.")
    
    # Get layer frequencies
    layer_freqs = []
    if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
        layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
    
    # Initialize chamber resonance
    chamber_resonance = initial_resonance
    
    # Find best resonance between layer frequencies and target frequency
    if layer_freqs:
        for freq in layer_freqs:
            res = calculate_resonance(freq, target_freq)
            chamber_resonance = max(chamber_resonance, res)
    
    # Calculate wavelength and dimensions
    wavelength = SPEED_OF_SOUND / target_freq if target_freq > FLOAT_EPSILON else 0.0
    
    # Create chamber with geometric properties
    chamber = {
        "id": str(uuid.uuid4()),
        "type": chamber_type,
        "target_frequency": float(target_freq),
        "resonance": float(chamber_resonance),
        "wavelength": float(wavelength),
        "creation_time": datetime.now().isoformat(),
        # Geometric properties based on chamber type
        "geometry": _get_chamber_geometry(chamber_type, wavelength, chamber_resonance)
    }
    
    # Add to layer's resonant_chambers
    if 'resonant_chambers' not in layer:
        layer['resonant_chambers'] = []
    
    layer['resonant_chambers'].append(chamber)
    
    # FIXED: Initialize layer's earth_resonance property with ALL required keys
    if 'earth_resonance' not in layer:
        layer['earth_resonance'] = {
            'schumann': 0.0,
            'core': 0.0,
            'total': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    else:
        # FIXED: Ensure all required keys exist in existing earth_resonance
        if 'schumann' not in layer['earth_resonance']:
            layer['earth_resonance']['schumann'] = 0.0
        if 'core' not in layer['earth_resonance']:
            layer['earth_resonance']['core'] = 0.0
        if 'total' not in layer['earth_resonance']:
            layer['earth_resonance']['total'] = 0.0
    
    # Update the appropriate resonance based on chamber type
    if chamber_type == 'schumann' or 'schumann' in chamber_type:
        layer['earth_resonance']['schumann'] = max(
            layer['earth_resonance']['schumann'],
            chamber_resonance
        )
    elif chamber_type == 'core' or 'core' in chamber_type:
        layer['earth_resonance']['core'] = max(
            layer['earth_resonance']['core'],
            chamber_resonance
        )
    
    # Recalculate total resonance (NOW all keys are guaranteed to exist)
    layer['earth_resonance']['total'] = (
        layer['earth_resonance']['schumann'] * 0.7 +
        layer['earth_resonance']['core'] * 0.3
    )
    layer['earth_resonance']['timestamp'] = datetime.now().isoformat()
    
    logger.debug(f"Created {chamber_type} resonant chamber in layer {layer_idx}: "
                f"Freq={target_freq:.1f}Hz, Res={chamber_resonance:.3f}")
    
    return chamber


def _get_chamber_geometry(chamber_type: str, wavelength: float, resonance: float) -> Dict[str, Any]:
    """
    Calculates geometric properties of a resonant chamber based on type and wavelength.
    Returns chamber geometry parameters used for resonance and visualization.
    """
    # Default dimensions
    width = wavelength * 0.25  # Quarter wavelength for width
    height = wavelength * 0.5  # Half wavelength for height
    depth = wavelength * 0.25  # Quarter wavelength for depth
    
    # Adjust by type
    if chamber_type == 'schumann':
        # Schumann chambers are flatter, wider
        width *= 1.2
        height *= 0.8
        shape = "dome"
    elif chamber_type == 'core':
        # Core chambers are taller, narrower
        width *= 0.9
        height *= 1.1
        shape = "column"
    elif chamber_type == 'planetary':
        # Planetary chambers are more spherical
        width = height = depth
        shape = "sphere"
    else:
        # Default generic chamber
        shape = "cube"
    
    # PHI-based scaling for natural resonance
    phi_scale = PHI ** (resonance * 2)  # Higher resonance = more PHI influence
    width *= (0.8 + 0.4 * phi_scale)
    height *= (0.8 + 0.4 * phi_scale)
    
    # Calculate volume
    if shape == "sphere":
        volume = (4/3) * PI * (width/2)**3
    elif shape == "dome":
        volume = (2/3) * PI * (width/2)**2 * height
    elif shape == "column":
        volume = PI * (width/2)**2 * height
    else:  # cube
        volume = width * height * depth
    
    # Add sacred geometry proportions if resonance is high
    sacred_geometry = {}
    if resonance > 0.7:
        sacred_geometry["phi_ratio"] = True
        sacred_geometry["vesica_piscis"] = True
        sacred_geometry["golden_spiral"] = resonance > 0.85
    
    return {
        "shape": shape,
        "width": float(width),
        "height": float(height),
        "depth": float(depth),
        "volume": float(volume),
        "phi_scale": float(phi_scale),
        "sacred_geometry": sacred_geometry
    }

def _create_earth_resonance_layer(soul_spark: SoulSpark, earth_freq: float, 
                                  resonance_strength: float, layer_type: str) -> Dict[str, Any]:
    """
    NATURAL VERSION: Creates an aura layer that resonates with Earth frequency.
    Uses same principle as Sephiroth layer formation - natural aura adaptation.
    
    Args:
        soul_spark: The soul to add layer to
        earth_freq: Earth frequency to resonate with  
        resonance_strength: Natural resonance achieved (0-1)
        layer_type: Type of layer ('schumann' or 'earth_core')
        
    Returns:
        Dict with layer creation metrics
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(earth_freq, (int, float)) or earth_freq <= FLOAT_EPSILON:
        raise ValueError(f"Invalid Earth frequency: {earth_freq}")
    if not (0.0 <= resonance_strength <= 1.0):
        raise ValueError("Invalid resonance strength.")
    
    timestamp = datetime.now().isoformat()
    
    # NATURAL: Use same layer formation logic as Sephiroth system
    # Base density from resonance with natural curve (same as _form_sephirah_layer)
    base_density = resonance_strength ** 0.8  # Natural scaling
    
    # Apply minor random variance for organic feel (same as Sephiroth)
    variance_range = 0.1 * (1.0 - resonance_strength)
    density_variance = random.uniform(-variance_range, variance_range)
    layer_density = max(0.1, min(1.0, base_density + density_variance))
    
    # Layer uniformity based on soul's pattern coherence (natural principle)
    pattern_distortion = soul_spark.get_pattern_distortion()
    base_uniformity = 1.0 - pattern_distortion
    uniformity_boost = resonance_strength * 0.2  # Resonance improves uniformity
    layer_uniformity = max(0.0, min(1.0, base_uniformity + uniformity_boost))
    
    # Vibration factor for high resonance layers (same as Sephiroth)
    vibration_factor = 0.0
    if resonance_strength > 0.7:
        vibration_factor = (resonance_strength - 0.7) * 3.0
        vibration_factor = min(0.9, vibration_factor)
    
    # Layer depth based on resonance strength (natural penetration)
    layer_depth = resonance_strength * 0.8
    
    # Create enhanced density map (same structure as Sephiroth layers)
    density_map = {
        'base_density': layer_density,
        'uniformity': layer_uniformity,
        'vibration_factor': vibration_factor,
        'depth': layer_depth
    }
    
    # NATURAL: Earth-specific layer colors
    if layer_type == 'schumann':
        layer_color_hex = '#8B7355'  # Earth brown
    elif layer_type == 'earth_core':
        layer_color_hex = '#CD853F'  # Deeper earth tone
    else:
        layer_color_hex = '#A0522D'  # Generic earth color
    
    # Scale color intensity based on resonance (same as Sephiroth)
    try:
        # Parse hex color
        r = int(layer_color_hex[1:3], 16)
        g = int(layer_color_hex[3:5], 16) 
        b = int(layer_color_hex[5:7], 16)
        
        # Scale intensity
        intensity_factor = 0.6 + (resonance_strength * 0.4)
        r = min(255, int(r * intensity_factor))
        g = min(255, int(g * intensity_factor))
        b = min(255, int(b * intensity_factor))
        
        layer_color_hex = f'#{r:02x}{g:02x}{b:02x}'
    except Exception:
        pass  # Use default color if parsing fails
    
    # Add layer to soul using existing method
    layer_name = f"Earth_{layer_type.capitalize()}"
    soul_spark.add_layer(layer_name, density_map, layer_color_hex, timestamp)
    
    # NATURAL: Add Earth frequency to layer's resonant frequencies
    # This enables future harmonic interactions (same as Sephiroth)
    if hasattr(soul_spark, 'layers') and soul_spark.layers:
        latest_layer = soul_spark.layers[-1]
        if isinstance(latest_layer, dict):
            if 'resonant_frequencies' not in latest_layer:
                latest_layer['resonant_frequencies'] = []
            
            # Add Earth frequency to layer's resonant spectrum
            if earth_freq not in latest_layer['resonant_frequencies']:
                latest_layer['resonant_frequencies'].append(float(earth_freq))
            
            # Initialize earth_resonance tracking in layer
            if 'earth_resonance' not in latest_layer:
                latest_layer['earth_resonance'] = {
                    'frequency': earth_freq,
                    'resonance_strength': resonance_strength,
                    'layer_type': layer_type,
                    'timestamp': timestamp
                }
    
    # Create metrics (same pattern as Sephiroth)
    layer_metrics = {
        'layer_type': layer_type,
        'earth_frequency': float(earth_freq),
        'resonance_strength': float(resonance_strength),
        'layer_density': float(layer_density),
        'layer_uniformity': float(layer_uniformity),
        'vibration_factor': float(vibration_factor),
        'layer_depth': float(layer_depth),
        'layer_color': layer_color_hex,
        'timestamp': timestamp
    }
    
    logger.debug(f"Created {layer_type} Earth resonance layer: "
                f"freq={earth_freq:.1f}Hz, resonance={resonance_strength:.3f}, "
                f"density={layer_density:.3f}")
    
    return layer_metrics

def _establish_standing_waves(soul_spark: SoulSpark, earth_freq: float, 
                            chamber_type: str) -> float:
    """
    FIXED VERSION: Uses actively tuned layer frequencies for Earth resonance.
    Expects layers to have been tuned to Earth frequencies via active tuning process.
    
    Args:
        soul_spark: The soul to establish standing waves with
        earth_freq: Earth frequency to resonate with
        chamber_type: Type of resonance ('schumann' or 'core')
        
    Returns:
        Natural resonance level achieved (0-1)
    """
    logger.info(f"Establishing {chamber_type} standing waves with Earth frequency {earth_freq:.1f}Hz...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(earth_freq, (int, float)) or earth_freq <= FLOAT_EPSILON:
        raise ValueError(f"Invalid Earth frequency: {earth_freq}")
    
    # Get the best resonance from tuned layer frequencies
    best_layer_resonance = 0.0
    tuned_freq_used = None
    tuned_frequencies_found = 0
    
    if hasattr(soul_spark, 'layers') and soul_spark.layers:
        for layer in soul_spark.layers:
            if isinstance(layer, dict) and 'resonant_frequencies' in layer:
                layer_freqs = layer.get('resonant_frequencies', [])
                for layer_freq in layer_freqs:
                    if layer_freq > FLOAT_EPSILON:
                        tuned_frequencies_found += 1
                        # Calculate resonance with tuned frequency
                        layer_resonance = calculate_resonance(layer_freq, earth_freq)
                        if layer_resonance > best_layer_resonance:
                            best_layer_resonance = layer_freq
                            tuned_freq_used = layer_freq
    
    # Check if we found tuned frequencies
    if tuned_frequencies_found == 0:
        raise ValueError(f"No tuned layer frequencies found for {chamber_type} resonance calculation. Active tuning may have failed.")
    
    # Use the best tuned frequency resonance
    combined_resonance = best_layer_resonance
    logger.debug(f"Using tuned layer frequency {tuned_freq_used:.1f}Hz for {chamber_type} resonance: {combined_resonance:.3f}")
    
    # Apply chamber-type specific harmonic relationships
    if chamber_type == 'schumann':
        if earth_freq == SCHUMANN_FREQUENCY:
            combined_resonance *= 1.1  # Natural biological resonance
    elif chamber_type == 'core':
        earth_element = getattr(soul_spark, 'elements', {}).get('earth', 0.2)
        if earth_element > 0.3:
            combined_resonance *= (0.9 + earth_element * 0.2)
    
    # Final resonance
    final_resonance = min(1.0, max(0.0, combined_resonance))
    
    # Record resonance in soul attributes
    if chamber_type == 'schumann':
        setattr(soul_spark, 'schumann_resonance_level', float(final_resonance))
    elif chamber_type == 'core':
        setattr(soul_spark, 'earth_core_resonance_level', float(final_resonance))
    
    logger.debug(f"{chamber_type} resonance with tuned frequencies: {final_resonance:.3f}")
    
    return final_resonance


def _attune_to_schumann_resonance(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    NATURAL VERSION: Attunes soul to Schumann resonance through natural aura adaptation.
    Creates aura layers that resonate with Schumann frequencies naturally.
    No artificial boosting - follows same principles as Sephiroth system.
    """
    logger.info("Attuning to Schumann resonance...")
    
    # Primary Schumann frequency
    schumann_freq = SCHUMANN_FREQUENCY
    
    # NATURAL: Calculate natural resonance (no artificial amplification)
    primary_resonance = _establish_standing_waves(soul_spark, schumann_freq, 'schumann')
    
    # NATURAL: Create aura layer if meaningful resonance achieved
    layer_metrics = {}
    if primary_resonance > 0.1:  # Only create layer if natural resonance exists
        layer_metrics = _create_earth_resonance_layer(
            soul_spark, schumann_freq, primary_resonance, 'schumann'
        )
    
    # Schumann harmonics (natural overtones)
    schumann_harmonics = [14.3, 20.8, 27.3, 33.8]
    harmonic_resonances = []
    harmonic_layers = []
    
    for h_freq in schumann_harmonics:
        h_resonance = _establish_standing_waves(soul_spark, h_freq, 'schumann_harmonic')
        harmonic_resonances.append(h_resonance)
        
        # Create harmonic layer if resonance is meaningful
        if h_resonance > 0.1:
            h_layer_metrics = _create_earth_resonance_layer(
                soul_spark, h_freq, h_resonance, f'schumann_harmonic_{h_freq:.1f}'
            )
            harmonic_layers.append(h_layer_metrics)
    
    # NATURAL: Calculate weighted average (same logic as Sephiroth)
    if harmonic_resonances:
        primary_weight = 0.6
        harmonic_weight = 0.4
        harmonic_average = sum(harmonic_resonances) / len(harmonic_resonances)
        weighted_average = (primary_resonance * primary_weight + harmonic_average * harmonic_weight)
    else:
        weighted_average = primary_resonance
    
    # NATURAL: Update soul's earth_resonance based on actual achieved resonance
    # No artificial boosting - this is what the aura naturally achieved
    current_earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
    schumann_contribution = weighted_average  # Pure natural contribution
    
    # Update earth resonance (use max to accumulate best resonance achieved)
    new_earth_resonance = max(current_earth_resonance, schumann_contribution)
    setattr(soul_spark, 'earth_resonance', float(new_earth_resonance))
    
    # Create metrics
    metrics_data = {
        'schumann_primary_resonance': primary_resonance,
        'schumann_harmonic_resonances': harmonic_resonances,
        'schumann_weighted_average': weighted_average,
        'schumann_contribution': schumann_contribution,
        'layers_created': 1 if layer_metrics else 0,
        'harmonic_layers_created': len(harmonic_layers),
        'natural_earth_resonance': new_earth_resonance,
        'layer_metrics': layer_metrics,
        'harmonic_layer_metrics': harmonic_layers,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Schumann attunement complete (NATURAL): "
               f"Primary: {primary_resonance:.3f}, "
               f"Harmonics Avg: {sum(harmonic_resonances) / len(harmonic_resonances) if harmonic_resonances else 0.0:.3f}, "
               f"Weighted: {weighted_average:.3f}, "
               f"Earth Resonance: {new_earth_resonance:.3f}")
    
    return metrics_data

def _attune_to_earth_core(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    NATURAL VERSION: Attunes soul to Earth core frequency through natural aura adaptation.
    Creates aura layers that resonate with Earth core frequency naturally.
    """
    logger.info("Attuning to Earth core frequency...")
    
    # Earth core frequency
    earth_core_freq = EARTH_FREQUENCY
    
    # NATURAL: Calculate natural resonance (no artificial amplification)
    core_resonance = _establish_standing_waves(soul_spark, earth_core_freq, 'core')
    
    # NATURAL: Create aura layer if meaningful resonance achieved
    layer_metrics = {}
    if core_resonance > 0.1:  # Only create layer if natural resonance exists
        layer_metrics = _create_earth_resonance_layer(
            soul_spark, earth_core_freq, core_resonance, 'earth_core'
        )
    
    # NATURAL: Update soul's earth_resonance based on achieved resonance
    current_earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
    core_contribution = core_resonance  # Pure natural contribution
    
    # Combine with existing earth resonance using natural weighted average
    # This reflects how multiple resonant layers work together
    if current_earth_resonance > 0.0:
        # Weight both contributions based on their natural intensity
        schumann_weight = HARMONY_SCHUMANN_INTENSITY
        core_weight = HARMONY_CORE_INTENSITY
        total_weight = schumann_weight + core_weight
        
        combined_earth_resonance = (
            current_earth_resonance * schumann_weight +
            core_contribution * core_weight
        ) / total_weight
    else:
        combined_earth_resonance = core_contribution
    
    setattr(soul_spark, 'earth_resonance', float(combined_earth_resonance))
    
    # Create metrics
    metrics_data = {
        'core_resonance': core_resonance,
        'core_contribution': core_contribution,
        'combined_earth_resonance': combined_earth_resonance,
        'layers_created': 1 if layer_metrics else 0,
        'layer_metrics': layer_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Earth core attunement complete (NATURAL): "
               f"Core Resonance: {core_resonance:.3f}, "
               f"Combined Earth Resonance: {combined_earth_resonance:.3f}")
    
    return metrics_data

def _perform_light_spectrum_integration(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    Integrates light spectrum frequencies corresponding to Earth resonance.
    Creates light-based information exchange between soul and Earth.
    Returns metrics about the integration process.
    """
    logger.info("Performing light spectrum integration...")
    
    # Audio frequency to light frequency scaling factor
    audio_to_light_scale = 1e12  # Example scaling
    
    # Get layers
    layers = soul_spark.layers
    if not layers:
        logger.warning("No layers available for light spectrum integration.")
        return {'light_resonance': 0.0}
    
    # Track light spectrum integration
    light_resonance = 0.0
    light_spectrum_entries = []
    
    # Process each layer with earth resonance
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
            
        # Check if layer has earth resonance
        if 'earth_resonance' not in layer or not isinstance(layer['earth_resonance'], dict):
            continue
            
        total_res = layer['earth_resonance'].get('total', 0.0)
        if total_res <= FLOAT_EPSILON:
            continue
            
        # Get layer frequencies that resonate with Earth
        earth_freqs = []
        if 'resonant_chambers' in layer and isinstance(layer['resonant_chambers'], list):
            for chamber in layer['resonant_chambers']:
                if 'type' in chamber and ('schumann' in chamber['type'] or 'core' in chamber['type']):
                    freq = chamber.get('target_frequency', 0.0)
                    if freq > FLOAT_EPSILON:
                        earth_freqs.append(freq)
        
        if not earth_freqs:
            continue
            
        # Create light spectrum entry for each frequency
        for freq in earth_freqs:
            # Convert audio frequency to light spectrum
            light_freq = freq * audio_to_light_scale
            
            # Calculate wavelength in nanometers
            # c = λν where c is speed of light, λ is wavelength, ν is frequency
            speed_of_light = 299792458  # m/s
            wavelength_m = speed_of_light / light_freq if light_freq > 0 else 0
            wavelength_nm = wavelength_m * 1e9  # Convert to nanometers
            
            # Map wavelength to color spectrum (approximate)
            color = _wavelength_to_color(wavelength_nm)
            
            # Create light spectrum entry
            light_entry = {
                "audio_frequency": float(freq),
                "light_frequency": float(light_freq),
                "wavelength_nm": float(wavelength_nm),
                "color": color,
                "resonance": float(total_res),
                "layer_idx": i
            }
            
            light_spectrum_entries.append(light_entry)
            
            # Update overall light resonance
            light_resonance = max(light_resonance, total_res)
            
            # Add light spectrum information to the layer
            if 'light_spectrum' not in layer:
                layer['light_spectrum'] = []
            
            layer['light_spectrum'].append(light_entry)
    
    # Create metrics
    metrics_data = {
        'light_resonance': light_resonance,
        'light_spectrum_entries': len(light_spectrum_entries),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Light spectrum integration complete. "
               f"Entries: {len(light_spectrum_entries)}, "
               f"Resonance: {light_resonance:.3f}")
    
    return metrics_data

def _wavelength_to_color(wavelength_nm: float) -> str:
    """
    Converts a wavelength in nanometers to an approximate RGB color.
    Returns the color as a hex string.
    """
    if wavelength_nm < 380 or wavelength_nm > 750:
        # Outside visible spectrum
        return "#000000"  # Black
    
    # Map wavelength to RGB using approximate model
    if 380 <= wavelength_nm < 440:
        # Violet
        r = (440 - wavelength_nm) / 60
        g = 0
        b = 1
    elif 440 <= wavelength_nm < 490:
        # Blue
        r = 0
        g = (wavelength_nm - 440) / 50
        b = 1
    elif 490 <= wavelength_nm < 510:
        # Green-blue
        r = 0
        g = 1
        b = (510 - wavelength_nm) / 20
    elif 510 <= wavelength_nm < 580:
        # Green-yellow
        r = (wavelength_nm - 510) / 70
        g = 1
        b = 0
    elif 580 <= wavelength_nm < 645:
        # Yellow-red
        r = 1
        g = (645 - wavelength_nm) / 65
        b = 0
    else:  # 645 - 750
        # Red
        r = 1
        g = 0
        b = 0
        
        # Intensity adjustment for deep red
        if wavelength_nm > 700:
            intensity = 0.3 + 0.7 * (750 - wavelength_nm) / 50
            r *= intensity
    
    # Convert to hex
    r_int = int(r * 255)
    g_int = int(g * 255)
    b_int = int(b * 255)
    
    return f"#{r_int:02X}{g_int:02X}{b_int:02X}"

def calculate_earth_cycle_resonance(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    Calculate resonance with Earth's natural cycles (day/night, seasons, etc.).
    Creates resonant patterns with these cycles in soul layers.
    """
    logger.info("Calculating Earth cycle resonance...")
    
    # Get existing earth_cycles or initialize
    earth_cycles = getattr(soul_spark, 'earth_cycles', {})
    if not isinstance(earth_cycles, dict):
        earth_cycles = {cycle_name: 0.0 for cycle_name in HARMONY_CYCLE_NAMES}
    
    # Get layers
    layers = soul_spark.layers
    if not layers:
        raise ValueError("Soul layers missing for Earth cycle resonance calculation.")
    
    # Define Earth cycles and their frequencies
    cycle_frequencies = {
        'diurnal': 1.0 / (24 * 3600),  # Once per day (Hz)
        'lunar': 1.0 / (29.53 * 24 * 3600),  # Once per lunar month
        'seasonal': 1.0 / (365.25 * 24 * 3600),  # Once per year
        'biorhythm': 1.0 / (28 * 24 * 3600),  # ~28 day cycle
        'circadian': 1.0 / (24 * 3600)  # Similar to diurnal but for internal rhythms
    }
    
    # Calculate resonance for each cycle
    total_resonance = 0.0
    resonance_details = {}
    
    for cycle_name, cycle_freq in cycle_frequencies.items():
        # Scale to audible range for calculations
        audible_freq = cycle_freq * EARTH_CYCLE_FREQ_SCALING
        
        # Find best resonant layer for this cycle
        best_layer_idx = -1
        best_resonance = 0.0
        
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                continue
                
            # Get layer frequencies
            layer_freqs = []
            if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
                layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
            
            # Calculate resonance with this layer
            layer_resonance = 0.0
            for freq in layer_freqs:
                res = calculate_resonance(freq, audible_freq)
                layer_resonance = max(layer_resonance, res)
            
            if layer_resonance > best_resonance:
                best_resonance = layer_resonance
                best_layer_idx = i
        
        # Create resonant chamber in best layer if found
        if best_layer_idx >= 0 and best_resonance > 0.2:
            chamber = _create_resonant_layer_chamber(
                soul_spark, best_layer_idx, audible_freq, f'earth_cycle_{cycle_name}', best_resonance
            )
            
            # Update earth_cycles with new resonance
            cycle_resonance = max(earth_cycles.get(cycle_name, 0.0), best_resonance)
            earth_cycles[cycle_name] = float(cycle_resonance)
            
            total_resonance += cycle_resonance
            
            resonance_details[cycle_name] = {
                "frequency": float(audible_freq),
                "resonance": float(cycle_resonance),
                "layer_idx": best_layer_idx
            }
            
            logger.debug(f"  Earth cycle '{cycle_name}' resonance: {cycle_resonance:.3f} (Layer {best_layer_idx})")
    
    # Calculate average resonance
    avg_resonance = total_resonance / len(cycle_frequencies) if cycle_frequencies else 0.0
    
    # Update soul's earth_cycles
    setattr(soul_spark, 'earth_cycles', earth_cycles)
    
    # Create metrics
    metrics_data = {
        'earth_cycles': earth_cycles,
        'average_resonance': float(avg_resonance),
        'resonance_details': resonance_details,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Earth cycle resonance calculation complete. Avg resonance: {avg_resonance:.3f}")
    return metrics_data

def calculate_elemental_balance(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    Calculate soul's resonance with elemental energies (earth, air, fire, water, aether).
    Updates the elements dictionary in the soul spark.
    """
    logger.info("Calculating elemental balance...")
    
    # Get existing elements or initialize
    elements = getattr(soul_spark, 'elements', {})
    if not isinstance(elements, dict) or not elements:
        elements = {'earth': 0.2, 'air': 0.2, 'fire': 0.2, 'water': 0.2, 'aether': 0.2}
    
    # Get soul properties for calculation
    soul_color = getattr(soul_spark, 'soul_color', None)
    elemental_affinity = getattr(soul_spark, 'elemental_affinity', None)
    sephiroth_aspect = getattr(soul_spark, 'sephiroth_aspect', None)
    layers = soul_spark.layers
    
    # Calculate base elemental resonance from soul properties
    base_elements = {'earth': 0.0, 'air': 0.0, 'fire': 0.0, 'water': 0.0, 'aether': 0.0}
    
    # From elemental affinity (strongest influence)
    if elemental_affinity:
        affinity_lower = elemental_affinity.lower()
        if affinity_lower in base_elements:
            base_elements[affinity_lower] += 0.4
            # Secondary elements based on common relationships
            if affinity_lower == 'earth':
                base_elements['water'] += 0.1
            elif affinity_lower == 'air':
                base_elements['fire'] += 0.1
            elif affinity_lower == 'fire':
                base_elements['air'] += 0.1
            elif affinity_lower == 'water':
                base_elements['earth'] += 0.1
            elif affinity_lower == 'aether':
                # Aether balances all
                for elem in base_elements:
                    if elem != 'aether':
                        base_elements[elem] += 0.05
    
    # From soul color (secondary influence)
    if soul_color:
        color_lower = soul_color.lower()
        elemental_color_map = {
            '#ff0000': 'fire',    # Red -> Fire
            '#00ff00': 'earth',   # Green -> Earth
            '#0000ff': 'water',   # Blue -> Water
            '#ffff00': 'air',     # Yellow -> Air
            '#ffffff': 'aether',  # White -> Aether
            '#800080': 'aether'   # Purple -> Aether
        }
        
        # Find closest color match
        best_element = None
        min_distance = float('inf')
        for color_hex, element in elemental_color_map.items():
            # Simple color distance (Manhattan)
            try:
                r1, g1, b1 = int(soul_color[1:3], 16), int(soul_color[3:5], 16), int(soul_color[5:7], 16)
                r2, g2, b2 = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                distance = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_element = element
            except (ValueError, IndexError):
                logger.warning(f"Invalid color format for soul_color: {soul_color}")
                continue
        
        if best_element and best_element in base_elements:
            base_elements[best_element] += 0.2
    
    # From sephiroth aspect
    if sephiroth_aspect and aspect_dictionary is not None:
        try:
            seph_elements = aspect_dictionary.get_aspects(sephiroth_aspect).get('element', '')
            if '/' in seph_elements:
                # Multiple elements
                elem_list = seph_elements.split('/')
                weight = 0.2 / max(1, len(elem_list))
                for elem in elem_list:
                    elem_lower = elem.lower()
                    if elem_lower in base_elements:
                        base_elements[elem_lower] += weight
            elif seph_elements and seph_elements.lower() in base_elements:
                base_elements[seph_elements.lower()] += 0.2
        except (KeyError, AttributeError):
            logger.warning(f"Failed to get element for sephiroth aspect: {sephiroth_aspect}")
    
    # From layer resonance with elemental frequencies
    elemental_frequencies = {
        'earth': 256.0,  # Hz - Earth resonance
        'water': 384.0,  # Hz - Water resonance
        'fire': 528.0,   # Hz - Fire resonance
        'air': 432.0,    # Hz - Air resonance
        'aether': 432.0 * PHI  # Hz - Aether/Spirit resonance
    }
    
    for element, freq in elemental_frequencies.items():
        # Find resonant layers
        resonant_layers = find_resonant_layers(soul_spark, freq)
        if resonant_layers:
            # Add resonance contribution (avg of top 3 if available)
            top_layers = sorted(resonant_layers, key=lambda x: x[1], reverse=True)[:3]
            avg_resonance = sum(res for _, res in top_layers) / len(top_layers)
            base_elements[element] += avg_resonance * 0.2
    
    # Normalize to ensure sum is 1.0
    total = sum(base_elements.values())
    if total > 0:
        normalized_elements = {elem: val / total for elem, val in base_elements.items()}
    else:
        normalized_elements = {elem: 0.2 for elem in base_elements}
    
    # Update with existing values using weighted average
    # (70% new calculation, 30% existing to ensure stability)
    updated_elements = {}
    for elem in normalized_elements:
        existing = elements.get(elem, 0.0)
        updated_elements[elem] = 0.7 * normalized_elements[elem] + 0.3 * existing
    
    # Ensure all elements have values and save
    for elem in ['earth', 'air', 'fire', 'water', 'aether']:
        if elem not in updated_elements:
            updated_elements[elem] = 0.1
    
    # Re-normalize to ensure sum is exactly 1.0
    total = sum(updated_elements.values())
    if total > 0:
        for elem in updated_elements:
            updated_elements[elem] = updated_elements[elem] / total
    
    # Update soul's elements
    setattr(soul_spark, 'elements', updated_elements)
    
    # Create metrics
    metrics_data = {
        'elements': updated_elements,
        'base_elements': base_elements,
        'normalized_elements': normalized_elements,
        'dominant_element': max(updated_elements.items(), key=lambda x: x[1])[0],
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Elemental balance calculation complete. Dominant: {metrics_data['dominant_element']}")
    return metrics_data

def _calculate_planetary_resonance(soul_spark: SoulSpark, birth_datetime: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculate resonance with planetary bodies based on astrological positions.
    Uses birth_datetime if provided, otherwise uses soul's conceptual_birth_datetime.
    """
    logger.info("Calculating planetary resonance...")
    
    # Get or create birth datetime
    if birth_datetime is None:
        # Use conceptual birth datetime from soul if available
        birth_dt_str = getattr(soul_spark, 'conceptual_birth_datetime', None)
        if birth_dt_str:
            try:
                birth_datetime = datetime.fromisoformat(birth_dt_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid conceptual_birth_datetime: {birth_dt_str}. Using current time.")
                birth_datetime = datetime.now()
        else:
            # Generate a random birth datetime if none exists
            # This is a simplified approach - in a real system, this would be more deterministic
            # based on soul properties
            now = datetime.now()
            year_ago = now - timedelta(days=365)
            random_seconds = random.randint(0, int((now - year_ago).total_seconds()))
            birth_datetime = year_ago + timedelta(seconds=random_seconds)
            # Save to soul
            setattr(soul_spark, 'conceptual_birth_datetime', birth_datetime.isoformat())
    
    # Define planetary frequencies based on cosmic correspondence
    # These are conceptual mappings, not direct orbital frequencies
    planetary_frequencies = {
        'sun': PLANETARY_FREQUENCIES.get('sun', 126.22),        # Hz
        'moon': PLANETARY_FREQUENCIES.get('moon', 210.42),      # Hz
        'mercury': PLANETARY_FREQUENCIES.get('mercury', 141.27), # Hz
        'venus': PLANETARY_FREQUENCIES.get('venus', 221.23),    # Hz
        'mars': PLANETARY_FREQUENCIES.get('mars', 144.72),      # Hz
        'jupiter': PLANETARY_FREQUENCIES.get('jupiter', 183.58), # Hz
        'saturn': PLANETARY_FREQUENCIES.get('saturn', 147.85),  # Hz
        'uranus': PLANETARY_FREQUENCIES.get('uranus', 207.36),  # Hz
        'neptune': PLANETARY_FREQUENCIES.get('neptune', 211.44), # Hz
        'pluto': PLANETARY_FREQUENCIES.get('pluto', 140.25)     # Hz
    }
    
    # Calculate resonance with each planetary body
    resonance_data = {}
    total_resonance = 0.0
    
    # Determine relative planetary positions based on birth date
    # This is a simplified approximation - a real implementation would use proper astronomical calculations
    
    # Simplified algorithm to determine planetary strength based on birth date components
    month = birth_datetime.month
    day = birth_datetime.day
    hour = birth_datetime.hour
    
    # Generate deterministic strength factors based on birth components
    # These are meant to approximate astrological influences rather than exact positions
    planet_strengths = {
        'sun': 0.4 + 0.6 * ((month + day) % 12) / 12,
        'moon': 0.4 + 0.6 * ((month + hour) % 24) / 24,
        'mercury': 0.3 + 0.7 * ((day + hour) % 10) / 10,
        'venus': 0.3 + 0.7 * ((month + 2*day) % 8) / 8,
        'mars': 0.3 + 0.7 * ((day + month*2) % 15) / 15,
        'jupiter': 0.3 + 0.7 * ((month * day) % 12) / 12,
        'saturn': 0.3 + 0.7 * ((month + day + hour) % 10) / 10,
        'uranus': 0.2 + 0.8 * ((day * hour) % 24) / 24,
        'neptune': 0.2 + 0.8 * ((month * hour) % 12) / 12,
        'pluto': 0.2 + 0.8 * ((day + month + hour) % 10) / 10
    }
    
    # Determine governing planet (highest strength)
    governing_planet = max(planet_strengths.items(), key=lambda x: x[1])[0]
    # Set it in the soul
    setattr(soul_spark, 'governing_planet', governing_planet)
    
    # Find resonant layers for each planet and establish resonance
    for planet, freq in planetary_frequencies.items():
        # Skip invalid frequencies
        if freq <= FLOAT_EPSILON:
            continue
            
        # Find resonant layers
        resonant_layers = find_resonant_layers(soul_spark, freq)
        
        # Calculate overall resonance for this planet
        planet_resonance = 0.0
        resonant_layer_ids = []
        
        if resonant_layers:
            # Choose top 3 resonant layers if available
            top_layers = sorted(resonant_layers, key=lambda x: x[1], reverse=True)[:3]
            
            # Create resonant chambers in these layers
            for layer_idx, layer_res in top_layers:
                # Weight by planetary strength from birth date
                strength_factor = planet_strengths.get(planet, 0.5)
                effective_resonance = layer_res * strength_factor
                
                # Create resonant chamber in layer
                if effective_resonance > 0.2:
                    chamber = _create_resonant_layer_chamber(
                        soul_spark, layer_idx, freq, f'planetary_{planet}', effective_resonance
                    )
                    
                    resonant_layer_ids.append(layer_idx)
                    planet_resonance = max(planet_resonance, effective_resonance)
        
        # Store resonance data
        resonance_data[planet] = {
            "frequency": float(freq),
            "strength": float(planet_strengths.get(planet, 0.5)),
            "resonance": float(planet_resonance),
            "resonant_layers": resonant_layer_ids,
            "is_governing": planet == governing_planet
        }
        
        # Add to total resonance (weighted more for governing planet)
        if planet == governing_planet:
            total_resonance += planet_resonance * 2.0
        else:
            total_resonance += planet_resonance
    
    # Calculate overall planetary resonance (weighted average)
    num_planets = len(planetary_frequencies)
    if num_planets > 0:
        # Give extra weight to governing planet
        weighted_denominator = num_planets + 1  # +1 for governing planet's extra weight
        overall_resonance = total_resonance / weighted_denominator
    else:
        overall_resonance = 0.0
    
    # Update soul's planetary_resonance attribute
    setattr(soul_spark, 'planetary_resonance', float(overall_resonance))
    
    # Create metrics
    metrics_data = {
        'birth_datetime': birth_datetime.isoformat(),
        'governing_planet': governing_planet,
        'planet_strengths': planet_strengths,
        'resonance_data': resonance_data,
        'overall_resonance': float(overall_resonance),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Planetary resonance calculation complete. Governing: {governing_planet}, Overall: {overall_resonance:.3f}")
    return metrics_data

def find_resonant_layers(soul_spark: SoulSpark, target_freq: float) -> List[Tuple[int, float]]:
    """
    Find layers in the soul that resonate with a target frequency.
    Returns list of tuples (layer_index, resonance_strength).
    """
    resonant_layers = []
    
    if not hasattr(soul_spark, 'layers') or not soul_spark.layers:
        return resonant_layers
    
    # Check each layer for resonance
    for i, layer in enumerate(soul_spark.layers):
        # Skip invalid layers
        if not isinstance(layer, dict):
            continue
            
        # Get layer frequencies
        layer_freqs = []
        
        # Try to get frequency from resonant_frequencies
        if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
            layer_freqs.extend(layer['resonant_frequencies'])
            
        # Try to get frequency from sephirah data
        if 'sephirah' in layer:
            sephirah = layer['sephirah'].lower()
            # Try to get frequency from SEPHIROTH_GLYPH_DATA
            try:
                if sephirah in SEPHIROTH_GLYPH_DATA and 'frequency' in SEPHIROTH_GLYPH_DATA[sephirah]:
                    layer_freqs.append(SEPHIROTH_GLYPH_DATA[sephirah]['frequency'])
            except (NameError, KeyError):
                pass
        
        # Calculate best resonance with any layer frequency
        best_resonance = 0.0
        for freq in layer_freqs:
            if freq <= FLOAT_EPSILON:
                continue
            res = calculate_resonance(freq, target_freq)
            best_resonance = max(best_resonance, res)
        
        # If resonance is significant, add this layer
        if best_resonance > 0.2:  # Threshold for considering resonance
            resonant_layers.append((i, best_resonance))
    
    # Sort by resonance strength (descending)
    resonant_layers.sort(key=lambda x: x[1], reverse=True)
    return resonant_layers

def _optimize_gaia_connection(soul_spark: SoulSpark, earth_resonance: float) -> Dict[str, Any]:
    """
    Optimize the connection with Gaia/Mother Earth (spiritual aspect) 
    based on existing Earth resonance. Gaia connection is the spiritual 
    counterpart to physical Earth resonance.
    """
    logger.info("Optimizing Gaia connection...")
    
    # Get current Gaia connection
    current_gaia = getattr(soul_spark, 'gaia_connection', 0.0)
    
    # Get relevant soul properties
    elements = getattr(soul_spark, 'elements', {})
    earth_cycles = getattr(soul_spark, 'earth_cycles', {})
    emotional_resonance = getattr(soul_spark, 'emotional_resonance', {})
    
    # Calculate base Gaia connection potential
    # Earth element affinity strongly affects Gaia connection
    earth_element = elements.get('earth', 0.2)
    water_element = elements.get('water', 0.2)
    
    # Earth cycles affect Gaia connection
    cycle_factor = 0.0
    for cycle, resonance in earth_cycles.items():
        if cycle in ['seasonal', 'lunar']:  # These are most important for Gaia
            cycle_factor += resonance * 0.3
        else:
            cycle_factor += resonance * 0.1
    cycle_factor = min(1.0, cycle_factor)
    
    # Emotional resonance affects Gaia connection
    # Love and compassion are most important for Gaia connection
    love_resonance = emotional_resonance.get('love', 0.0)
    compassion_resonance = emotional_resonance.get('compassion', 0.0)
    harmony_resonance = emotional_resonance.get('harmony', 0.0)
    
    emotion_factor = (love_resonance * 0.4 + 
                      compassion_resonance * 0.4 + 
                      harmony_resonance * 0.2)
    
    # Calculate Gaia connection potential
    gaia_potential = (earth_element * 0.3 + 
                      water_element * 0.2 + 
                      earth_resonance * 0.2 + 
                      cycle_factor * 0.1 + 
                      emotion_factor * 0.2)
    
    # Enhanced connection through heart-centered awareness
    # Use Fibonacci sequence to create natural growth pattern
    fib_factor = 0.0
    fib_seq = [1, 1, 2, 3, 5, 8, 13, 21]
    for i, fib in enumerate(fib_seq[:5]):  # Use first 5 terms
        # Create phi-based resonant nodes in heart center
        node_strength = (PHI ** i) / (PHI ** 5)  # Normalized to 0-1
        fib_factor += node_strength * 0.2  # Scale contribution
    
    # Apply to Gaia potential
    gaia_potential *= (1.0 + fib_factor * 0.5)  # Up to 50% boost
    
    # Gradual integration with current value for stability
    new_gaia_connection = current_gaia * 0.3 + gaia_potential * 0.7
    new_gaia_connection = max(0.0, min(1.0, new_gaia_connection))
    
    # Update soul's gaia_connection
    setattr(soul_spark, 'gaia_connection', float(new_gaia_connection))
    
    # Create metrics
    metrics_data = {
        'current_gaia_connection': float(current_gaia),
        'earth_element_factor': float(earth_element),
        'water_element_factor': float(water_element),
        'cycle_factor': float(cycle_factor),
        'emotion_factor': float(emotion_factor),
        'fib_factor': float(fib_factor),
        'gaia_potential': float(gaia_potential),
        'new_gaia_connection': float(new_gaia_connection),
        'change': float(new_gaia_connection - current_gaia),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Gaia connection optimized: {current_gaia:.3f} -> {new_gaia_connection:.3f} ({new_gaia_connection - current_gaia:+.3f})")
    return metrics_data

def _project_echo_field(soul_spark: SoulSpark, echo_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Projects an echo field between the soul and Earth to facilitate information exchange.
    Creates standing wave patterns that enhance resonance.
    """
    logger.info("Projecting echo field...")
    
    # Get essential properties
    cord_integrity = getattr(soul_spark, 'cord_integrity', 0.0)
    earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
    
    if cord_integrity < FLOAT_EPSILON or earth_resonance < FLOAT_EPSILON:
        raise ValueError("Cannot project echo field: cord_integrity or earth_resonance too low.")
    
    # Generate echo field structure
    echo_id = str(uuid.uuid4())
    creation_time = datetime.now().isoformat()
    
    # Calculate core echo parameters based on Earth and cord properties
    echo_strength = (cord_integrity * 0.6 + earth_resonance * 0.4) * ECHO_FIELD_STRENGTH_FACTOR
    echo_coherence = (earth_resonance * 0.7 + cord_integrity * 0.3) * ECHO_FIELD_COHERENCE_FACTOR
    
    # Create field with standing wave structure
    field_structure = {
        "echo_id": echo_id,
        "creation_time": creation_time,
        "strength": float(echo_strength),
        "coherence": float(echo_coherence),
        "standing_waves": [],
        "resonance_chambers": []
    }
    
    # Generate standing waves based on Schumann and Earth core resonance
    waves_created = 0
    
    # Get Schumann and Earth core resonant chambers from layers
    schumann_chambers = []
    core_chambers = []
    
    for i, layer in enumerate(soul_spark.layers):
        if not isinstance(layer, dict):
            continue
            
        # Check for resonant chambers
        if 'resonant_chambers' in layer and isinstance(layer['resonant_chambers'], list):
            for chamber in layer['resonant_chambers']:
                if isinstance(chamber, dict):
                    chamber_type = chamber.get('type', '')
                    if 'schumann' in chamber_type:
                        schumann_chambers.append((i, chamber))
                    elif 'core' in chamber_type:
                        core_chambers.append((i, chamber))
    
    # Create standing waves from Schumann chambers
    for layer_idx, chamber in schumann_chambers:
        freq = chamber.get('target_frequency', 0.0)
        if freq <= FLOAT_EPSILON:
            continue
            
        # Create standing wave
        wave = _create_standing_wave(freq, echo_strength, echo_coherence, 'schumann')
        field_structure["standing_waves"].append(wave)
        waves_created += 1
        
        # Link with layer chamber
        field_structure["resonance_chambers"].append({
            "layer_idx": layer_idx,
            "chamber_id": chamber.get('id', str(uuid.uuid4())),
            "frequency": float(freq),
            "type": "schumann"
        })
    
    # Create standing waves from Earth core chambers
    for layer_idx, chamber in core_chambers:
        freq = chamber.get('target_frequency', 0.0)
        if freq <= FLOAT_EPSILON:
            continue
            
        # Create standing wave
        wave = _create_standing_wave(freq, echo_strength, echo_coherence, 'core')
        field_structure["standing_waves"].append(wave)
        waves_created += 1
        
        # Link with layer chamber
        field_structure["resonance_chambers"].append({
            "layer_idx": layer_idx,
            "chamber_id": chamber.get('id', str(uuid.uuid4())),
            "frequency": float(freq),
            "type": "core"
        })
   
   # If no chambers found, create default standing waves
    if waves_created == 0:
        # Create default Schumann wave
        schumann_wave = _create_standing_wave(SCHUMANN_FREQUENCY, echo_strength, echo_coherence, 'schumann')
        field_structure["standing_waves"].append(schumann_wave)
        
        # Create default Earth core wave
        earth_wave = _create_standing_wave(EARTH_FREQUENCY, echo_strength, echo_coherence, 'core')
        field_structure["standing_waves"].append(earth_wave)
        
        waves_created = 2
    
    # Update soul with echo field data
    setattr(soul_spark, 'echo_field', field_structure)
    
    # Update echo state with field structure (for the parent function)
    echo_state["field_structure"] = field_structure
    echo_state["waves_created"] = waves_created
    
    # Set flag in soul
    setattr(soul_spark, FLAG_ECHO_PROJECTED, True)
    
    # Create metrics
    metrics_data = {
        'echo_id': echo_id,
        'echo_strength': float(echo_strength),
        'echo_coherence': float(echo_coherence),
        'waves_created': waves_created,
        'resonance_chambers_linked': len(field_structure["resonance_chambers"]),
        'timestamp': creation_time
    }
    
    # Generate sound for echo field projection
    if SOUND_MODULES_AVAILABLE:
        try:
            from shared.sound.sounds_of_universe import UniverseSounds
            # Initialize universe sounds generator
            universe_sounds = UniverseSounds()
            
            # Generate echo field sound
            echo_sound = universe_sounds.generate_dimensional_transition(
                duration=8.0,
                sample_rate=SAMPLE_RATE,
                transition_type='earth_echo_field',
                amplitude=0.7
            )
            
            # Save sound
            sound_file = f"echo_field_{soul_spark.spark_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
            universe_sounds.save_sound(
                echo_sound, 
                sound_file, 
                f"Echo field projection sound for soul {soul_spark.spark_id[:8]}"
            )
            
            logger.info(f"Generated echo field sound: {sound_file}")
            metrics_data["sound_file"] = sound_file
        except Exception as sound_err:
            logger.warning(f"Failed to generate echo field sound: {sound_err}")
    
    logger.info(f"Echo field projected. Strength: {echo_strength:.3f}, Coherence: {echo_coherence:.3f}, Waves: {waves_created}")
    return metrics_data

def _create_standing_wave(frequency: float, field_strength: float, field_coherence: float, wave_type: str) -> Dict[str, Any]:
    """
    Creates a standing wave configuration based on parameters.
    Uses wave physics to calculate nodes, antinodes, and resonance properties.
    """
    if frequency <= FLOAT_EPSILON:
        raise ValueError(f"Invalid frequency for standing wave: {frequency}")
    
    if not (0.0 <= field_strength <= 1.0) or not (0.0 <= field_coherence <= 1.0):
        raise ValueError(f"Invalid field parameters: strength={field_strength}, coherence={field_coherence}")
    
    # Calculate wavelength from frequency
    # Speed of sound is used as the wave speed reference
    wavelength = SPEED_OF_SOUND / frequency
    
    # Calculate number of nodes based on coherence and field strength
    # Higher coherence = more stable nodes
    node_stability = field_coherence * 0.8 + 0.2
    node_count = int(4 + field_strength * 8)  # 4 to 12 nodes depending on strength
    
    # Create nodes and antinodes
    nodes = []
    for i in range(node_count):
        # Position follows Fibonacci spacing for natural pattern
        # Position is normalized to 0-1 range (0 = soul, 1 = Earth)
        position = (PHI ** i) / (PHI ** node_count)
        position = min(0.99, max(0.01, position))  # Keep within valid range
        
        # Calculate node properties
        amplitude = field_strength * (1.0 - abs(position - 0.5) * 0.5)  # Max amplitude in the middle
        phase = (i % 2) * PI  # Alternating phases
        
        # Add noise/variability based on coherence
        # Lower coherence = more variability
        variability = (1.0 - node_stability) * 0.2
        position_var = position * (1.0 + random.uniform(-variability, variability))
        position_var = min(0.99, max(0.01, position_var))  # Keep within valid range
        
        # Adjust amplitude/phase based on wave type
        if wave_type == 'schumann':
            # Schumann waves have more variability in Earth's atmosphere
            phase_var = phase + random.uniform(-variability * PI, variability * PI)
            amplitude_var = amplitude * (1.0 + random.uniform(-variability, variability))
        elif wave_type == 'core':
            # Earth core waves are more stable/structured
            phase_var = phase + random.uniform(-variability * PI * 0.5, variability * PI * 0.5)
            amplitude_var = amplitude * (1.0 + random.uniform(-variability * 0.5, variability * 0.5))
        else:
            # Default behavior
            phase_var = phase + random.uniform(-variability * PI * 0.8, variability * PI * 0.8)
            amplitude_var = amplitude * (1.0 + random.uniform(-variability * 0.8, variability * 0.8))
        
        # Create node
        nodes.append({
            "position": float(position_var),
            "amplitude": float(amplitude_var),
            "phase": float(phase_var),
            "frequency": float(frequency),
            "stability": float(node_stability)
        })
    
    # Create standing wave structure
    standing_wave = {
        "id": str(uuid.uuid4()),
        "frequency": float(frequency),
        "wavelength": float(wavelength),
        "type": wave_type,
        "nodes": nodes,
        "field_strength": float(field_strength),
        "field_coherence": float(field_coherence)
    }
    
    return standing_wave

def finalize_earth_harmonization(soul_spark: SoulSpark, echo_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    NATURAL VERSION: Finalizes Earth harmonization using only naturally achieved resonance.
    No artificial boosting - respects the natural limitations and capabilities of each soul.
    If a soul cannot naturally achieve 0.75+ earth resonance, it may need different development.
    """
    logger.info("Finalizing Earth harmonization (NATURAL)...")
    
    # NATURAL: Get only the naturally achieved resonance components
    schumann_resonance = getattr(soul_spark, 'schumann_resonance_level', 0.0)
    core_resonance = getattr(soul_spark, 'earth_core_resonance_level', 0.0)
    gaia_connection = getattr(soul_spark, 'gaia_connection', 0.0)
    planetary_resonance = getattr(soul_spark, 'planetary_resonance', 0.0)
    
    # Get elemental balance data
    elements = getattr(soul_spark, 'elements', {})
    earth_element = elements.get('earth', 0.0)
    water_element = elements.get('water', 0.0)
    
    # NATURAL: Calculate earth resonance using natural weighted combination
    # This is how the soul's aura layers naturally combine their resonances
    base_earth_resonance = (
        schumann_resonance * HARMONY_SCHUMANN_INTENSITY +
        core_resonance * HARMONY_CORE_INTENSITY
    ) / (HARMONY_SCHUMANN_INTENSITY + HARMONY_CORE_INTENSITY)
    
    # NATURAL: Apply natural synergy factors (not artificial boosting)
    # These represent how different natural systems support each other
    natural_synergy = 0.0
    
    # Gaia connection naturally enhances earth resonance (spiritual-physical bridge)
    if gaia_connection > 0.5:
        gaia_synergy = (gaia_connection - 0.5) * 0.1  # Small natural boost
        natural_synergy += gaia_synergy
    
    # Strong earth elemental affinity creates natural grounding synergy
    if earth_element > 0.4:
        earth_synergy = (earth_element - 0.4) * 0.15  # Moderate natural boost
        natural_synergy += earth_synergy
    
    # Water element supports earth (natural elemental harmony)
    if water_element > 0.3 and earth_element > 0.3:
        water_earth_synergy = min(water_element, earth_element) * 0.05  # Small natural boost
        natural_synergy += water_earth_synergy
    
    # Apply natural synergy (small, organic enhancement)
    enhanced_earth_resonance = base_earth_resonance + natural_synergy
    
    # NATURAL: Final earth resonance is what was naturally achieved
    final_earth_resonance = min(1.0, max(0.0, enhanced_earth_resonance))
    
    # NATURAL: Update soul's earth_resonance with final natural value
    setattr(soul_spark, 'earth_resonance', float(final_earth_resonance))
    
    # NATURAL: Only set earth_attuned flag if meaningful resonance achieved
    # Some souls may not be naturally earth-compatible at their current development
    earth_attuned = final_earth_resonance >= 0.3  # Lower natural threshold
    setattr(soul_spark, FLAG_EARTH_ATTUNED, earth_attuned)
    
    # Get initial state for metrics
    initial_coherence = getattr(soul_spark, 'initial_coherence_before_earth', soul_spark.coherence)
    initial_stability = getattr(soul_spark, 'initial_stability_before_earth', soul_spark.stability)
    
    # Calculate final harmonization factor (natural combination)
    harmonization_factor = (
        final_earth_resonance * 0.5 +         # Earth resonance is primary
        gaia_connection * 0.2 +               # Gaia connection
        planetary_resonance * 0.1 +           # Planetary alignment
        earth_element * 0.1 +                 # Earth element
        water_element * 0.1                   # Water element support
    )
    
    # Update soul state if method exists
    if hasattr(soul_spark, 'update_state'):
        logger.debug("Updating soul state after natural Earth harmonization...")
        soul_spark.update_state()
    
    # Calculate coherence and stability changes (natural effects only)
    coherence_gain = soul_spark.coherence - initial_coherence
    stability_gain = soul_spark.stability - initial_stability
    
    # Create final metrics
    metrics_data = {
        'natural_base_earth_resonance': float(base_earth_resonance),
        'natural_synergy_bonus': float(natural_synergy),
        'final_earth_resonance': float(final_earth_resonance),
        'schumann_resonance': float(schumann_resonance),
        'core_resonance': float(core_resonance),
        'gaia_connection': float(gaia_connection),
        'planetary_resonance': float(planetary_resonance),
        'elemental_balance': elements,
        'harmonization_factor': float(harmonization_factor),
        'initial_coherence': float(initial_coherence),
        'final_coherence': float(soul_spark.coherence),
        'coherence_gain': float(coherence_gain),
        'initial_stability': float(initial_stability),
        'final_stability': float(soul_spark.stability),
        'stability_gain': float(stability_gain),
        'is_earth_attuned': earth_attuned,
        'meets_identity_threshold': final_earth_resonance >= IDENTITY_EARTH_RESONANCE_THRESHOLD,
        'natural_earth_compatibility': final_earth_resonance >= 0.5,
        'development_recommendation': _get_development_recommendation(final_earth_resonance),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add memory echo if available
    if hasattr(soul_spark, 'add_memory_echo'):
        soul_spark.add_memory_echo(f"Natural Earth harmonization complete. Earth resonance: {final_earth_resonance:.3f}, Gaia connection: {gaia_connection:.3f}, S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
    
    logger.info(f"Natural Earth harmonization finalized. Final Earth resonance: {final_earth_resonance:.3f}")
    logger.info(f"Coherence gain: {coherence_gain:+.1f} CU, Stability gain: {stability_gain:+.1f} SU")
    logger.info(f"Earth attuned: {earth_attuned}, Meets Identity threshold: {final_earth_resonance >= IDENTITY_EARTH_RESONANCE_THRESHOLD}")
    
    if final_earth_resonance < IDENTITY_EARTH_RESONANCE_THRESHOLD:
        logger.info(f"Soul may need alternative development path - natural earth resonance ({final_earth_resonance:.3f}) below identity threshold ({IDENTITY_EARTH_RESONANCE_THRESHOLD})")
    
    return metrics_data

def _get_development_recommendation(earth_resonance: float) -> str:
    """
    Provides natural development recommendations based on achieved earth resonance.
    """
    if earth_resonance >= 0.75:
        return "Ready for Identity Crystallization - Strong earth connection achieved"
    elif earth_resonance >= 0.5:
        return "Moderate earth connection - Consider additional grounding practices or elemental balancing"
    elif earth_resonance >= 0.3:
        return "Weak earth connection - May benefit from returning to Sephiroth journey for additional grounding aspects"
    elif earth_resonance >= 0.1:
        return "Minimal earth connection - Consider alternative development paths or foundational strengthening"
    else:
        return "No significant earth connection - Soul may be oriented toward non-earthly development"

def _check_natural_earth_compatibility(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    NATURAL: Checks if soul has natural potential for earth resonance.
    This replaces artificial guarantees with natural assessment.
    """
    compatibility_factors = {}
    
    # Check frequency natural compatibility with earth frequencies
    soul_freq = soul_spark.frequency
    schumann_compat = calculate_resonance(soul_freq, SCHUMANN_FREQUENCY)
    core_compat = calculate_resonance(soul_freq, EARTH_FREQUENCY)
    compatibility_factors['frequency_compatibility'] = max(schumann_compat, core_compat)
    
    # Check elemental natural affinity
    elements = getattr(soul_spark, 'elements', {})
    earth_affinity = elements.get('earth', 0.2) + elements.get('water', 0.2) * 0.5
    compatibility_factors['elemental_affinity'] = earth_affinity
    
    # Check existing aura layer harmonics (like in meditation)
    layer_harmonic_potential = 0.0
    if hasattr(soul_spark, 'layers') and soul_spark.layers:
        for layer in soul_spark.layers:
            if isinstance(layer, dict) and 'resonant_frequencies' in layer:
                layer_freqs = layer.get('resonant_frequencies', [])
                for layer_freq in layer_freqs:
                    if layer_freq > FLOAT_EPSILON:
                        # Check if layer frequency has harmonic relationship with earth
                        schumann_harmonic = find_best_harmonic_resonance(layer_freq, SCHUMANN_FREQUENCY)
                        core_harmonic = find_best_harmonic_resonance(layer_freq, EARTH_FREQUENCY)
                        layer_harmonic_potential = max(layer_harmonic_potential, 
                                                     schumann_harmonic, core_harmonic)
    compatibility_factors['aura_harmonic_potential'] = layer_harmonic_potential
    
    # Check soul development stage compatibility
    cord_integrity = getattr(soul_spark, 'cord_integrity', 0.0)
    stability_factor = getattr(soul_spark, 'stability', 0.0) / 100.0
    coherence_factor = getattr(soul_spark, 'coherence', 0.0) / 100.0
    development_readiness = (cord_integrity + stability_factor + coherence_factor) / 3.0
    compatibility_factors['development_readiness'] = development_readiness
    
    # Calculate overall natural earth compatibility
    overall_compatibility = (
        compatibility_factors['frequency_compatibility'] * 0.3 +
        compatibility_factors['elemental_affinity'] * 0.2 +
        compatibility_factors['aura_harmonic_potential'] * 0.3 +
        compatibility_factors['development_readiness'] * 0.2
    )
    
    # Natural compatibility assessment
    if overall_compatibility >= 0.6:
        compatibility_level = "High"
        recommendation = "Soul has strong natural earth compatibility"
    elif overall_compatibility >= 0.4:
        compatibility_level = "Moderate" 
        recommendation = "Soul has moderate earth compatibility - may achieve resonance with focused effort"
    elif overall_compatibility >= 0.2:
        compatibility_level = "Low"
        recommendation = "Soul has limited earth compatibility - consider alternative development paths"
    else:
        compatibility_level = "Minimal"
        recommendation = "Soul appears oriented toward non-earthly development"
    
    return {
        'overall_compatibility': overall_compatibility,
        'compatibility_level': compatibility_level,
        'recommendation': recommendation,
        'factors': compatibility_factors,
        'timestamp': datetime.now().isoformat()
    }

def _enhance_natural_earth_resonance(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    ACTIVE TUNING: Aura layers actively tune their frequencies to Earth resonance.
    Like meditation - the aura consciously adjusts to match Earth frequencies.
    No "natural alignment" needed - this IS the alignment process.
    """
    logger.info("Actively tuning aura frequencies to Earth resonance...")
    
    enhancement_metrics = {}
    tuning_adjustments = 0
    
    # ACTIVE: Tune existing layer frequencies toward Earth frequencies
    if hasattr(soul_spark, 'layers') and soul_spark.layers:
        # Target Earth frequencies for tuning
        earth_targets = [
            SCHUMANN_FREQUENCY,      # 7.83 Hz
            EARTH_FREQUENCY,         # Whatever the core frequency is
            14.3, 20.8, 27.3, 33.8  # Schumann harmonics
        ]
        
        for i, layer in enumerate(soul_spark.layers):
            if isinstance(layer, dict) and 'resonant_frequencies' in layer:
                layer_freqs = layer.get('resonant_frequencies', [])
                tuned_freqs = []
                
                for freq in layer_freqs:
                    if freq > FLOAT_EPSILON:
                        # ACTIVE TUNING: Find closest Earth frequency and tune toward it
                        closest_earth_freq = min(earth_targets, key=lambda x: abs(x - freq))
                        
                        # Calculate tuning adjustment (like adjusting meditation frequency)
                        freq_ratio = closest_earth_freq / freq
                        
                        # Apply tuning adjustment (gradual shift toward target)
                        if 0.5 <= freq_ratio <= 2.0:  # Within reasonable tuning range
                            # Tune 80% of the way toward the target
                            tuned_freq = freq + (closest_earth_freq - freq) * 0.8
                            tuned_freqs.append(tuned_freq)
                            tuning_adjustments += 1
                            
                            logger.debug(f"Layer {i}: Tuned {freq:.1f}Hz -> {tuned_freq:.1f}Hz (target: {closest_earth_freq:.1f}Hz)")
                        else:
                            # If too far apart, add the target frequency as a new resonance
                            tuned_freqs.append(freq)  # Keep original
                            tuned_freqs.append(closest_earth_freq)  # Add target
                            tuning_adjustments += 1
                            
                            logger.debug(f"Layer {i}: Added Earth frequency {closest_earth_freq:.1f}Hz alongside {freq:.1f}Hz")
                
                # Update layer frequencies with tuned versions
                if tuned_freqs:
                    layer['resonant_frequencies'] = tuned_freqs
        
        enhancement_metrics['frequency_tuning_adjustments'] = tuning_adjustments
    
    # ACTIVE: Add Earth frequencies to layers that don't have them
    earth_freq_additions = 0
    if hasattr(soul_spark, 'layers') and soul_spark.layers:
        for i, layer in enumerate(soul_spark.layers):
            if isinstance(layer, dict):
                if 'resonant_frequencies' not in layer:
                    layer['resonant_frequencies'] = []
                
                layer_freqs = layer['resonant_frequencies']
                
                # Add Schumann frequency if not present
                if not any(abs(f - SCHUMANN_FREQUENCY) < 1.0 for f in layer_freqs):
                    layer_freqs.append(SCHUMANN_FREQUENCY)
                    earth_freq_additions += 1
                    logger.debug(f"Layer {i}: Added Schumann frequency {SCHUMANN_FREQUENCY:.1f}Hz")
                
                # Add Earth core frequency if not present  
                if not any(abs(f - EARTH_FREQUENCY) < 1.0 for f in layer_freqs):
                    layer_freqs.append(EARTH_FREQUENCY)
                    earth_freq_additions += 1
                    logger.debug(f"Layer {i}: Added Earth core frequency {EARTH_FREQUENCY:.1f}Hz")
    
    enhancement_metrics['earth_frequency_additions'] = earth_freq_additions
    
    # ACTIVE: Elemental balancing toward earth affinity
    elements = getattr(soul_spark, 'elements', {})
    current_earth = elements.get('earth', 0.2)
    current_water = elements.get('water', 0.2)
    
    # Active shift toward earth/water balance
    if current_earth < 0.3:
        earth_boost = min(0.05, 0.3 - current_earth)
        elements['earth'] = current_earth + earth_boost
        enhancement_metrics['earth_element_boost'] = earth_boost
    
    if current_water < 0.25 and current_earth > 0.25:
        water_boost = min(0.03, 0.25 - current_water)
        elements['water'] = current_water + water_boost
        enhancement_metrics['water_element_boost'] = water_boost
    
    # Renormalize elements to maintain total = 1.0
    element_total = sum(elements.values())
    if element_total > 0:
        for element in elements:
            elements[element] = elements[element] / element_total
    
    setattr(soul_spark, 'elements', elements)
    
    logger.info(f"Active Earth frequency tuning applied: "
               f"{tuning_adjustments} frequency adjustments, "
               f"{earth_freq_additions} Earth frequency additions, "
               f"earth element: {current_earth:.3f} -> {elements.get('earth', 0.2):.3f}")
    
    return enhancement_metrics

def _recalculate_earth_resonances_after_alignment(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    Recalculates all Earth resonances after harmonic alignment has been applied.
    This ensures we use the aligned frequencies for final resonance calculations.
    """
    logger.info("Recalculating Earth resonances with aligned frequencies...")
    
    # Recalculate Schumann resonance with aligned frequencies
    schumann_resonance = _establish_standing_waves(soul_spark, SCHUMANN_FREQUENCY, 'schumann')
    
    # Recalculate Schumann harmonics
    schumann_harmonics = [14.3, 20.8, 27.3, 33.8]
    harmonic_resonances = []
    
    for h_freq in schumann_harmonics:
        h_resonance = _establish_standing_waves(soul_spark, h_freq, 'schumann_harmonic')
        harmonic_resonances.append(h_resonance)
    
    # Calculate weighted Schumann average
    if harmonic_resonances:
        primary_weight = 0.6
        harmonic_weight = 0.4
        harmonic_average = sum(harmonic_resonances) / len(harmonic_resonances)
        schumann_weighted = (schumann_resonance * primary_weight + harmonic_average * harmonic_weight)
    else:
        schumann_weighted = schumann_resonance
    
    # Recalculate Earth core resonance with aligned frequencies  
    core_resonance = _establish_standing_waves(soul_spark, EARTH_FREQUENCY, 'core')
    
    # Update earth resonance with recalculated values
    schumann_contribution = schumann_weighted * HARMONY_SCHUMANN_INTENSITY
    core_contribution = core_resonance * HARMONY_CORE_INTENSITY
    total_weight = HARMONY_SCHUMANN_INTENSITY + HARMONY_CORE_INTENSITY
    
    recalculated_earth_resonance = (schumann_contribution + core_contribution) / total_weight
    setattr(soul_spark, 'earth_resonance', float(recalculated_earth_resonance))
    
    logger.info(f"Recalculated resonances - Schumann: {schumann_weighted:.3f}, "
               f"Core: {core_resonance:.3f}, Earth: {recalculated_earth_resonance:.3f}")
    
    return {
        'schumann_resonance': schumann_resonance,
        'schumann_harmonics': harmonic_resonances,
        'schumann_weighted': schumann_weighted,
        'core_resonance': core_resonance,
        'recalculated_earth_resonance': recalculated_earth_resonance,
        'timestamp': datetime.now().isoformat()
    }

def perform_earth_harmonization(soul_spark: SoulSpark, schumann_intensity: float = HARMONY_SCHUMANN_INTENSITY, 
                                core_intensity: float = HARMONY_CORE_INTENSITY) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs complete Earth harmonization process using aura layer resonance.
    
    FIXED VERSION: Proper order - alignment first, then resonance calculation.
    
    Creates resonant chambers in aura layers that align with Earth frequencies.
    Establishes standing waves and echo fields for information exchange.
    Integrates with planetary resonance and Gaia connection.
    
    Args:
        soul_spark: The soul to harmonize with Earth.
        schumann_intensity: Intensity factor for Schumann resonance (0.0-1.0).
        core_intensity: Intensity factor for Earth core resonance (0.0-1.0).
        
    Returns:
        Tuple of (modified soul_spark, metrics dictionary).
        
    Raises:
        ValueError: If inputs are invalid or prerequisites not met.
        RuntimeError: For other failures during harmonization.
    """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("soul_spark invalid.")
    if not (0.0 <= schumann_intensity <= 1.0):
        raise ValueError(f"schumann_intensity {schumann_intensity} out of range [0.0, 1.0].")
    if not (0.0 <= core_intensity <= 1.0):
        raise ValueError(f"core_intensity {core_intensity} out of range [0.0, 1.0].")
    
    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Earth Harmonization for Soul {spark_id} ---")
    logger.info(f"Schumann intensity: {schumann_intensity:.2f}, Core intensity: {core_intensity:.2f}")
    
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}
    
    # Initialize echo_state BEFORE the try block so it exists even if prerequisites fail
    echo_state = {
        'schumann_resonance': 0.0,
        'earth_core_resonance': 0.0,
        'earth_resonance': 0.0,
        'stages_completed': []
    }
    
    try:
        # Check prerequisites
        _check_prerequisites(soul_spark)
        _ensure_soul_properties(soul_spark)
        
        # Get initial state for metrics
        initial_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'earth_resonance': getattr(soul_spark, 'earth_resonance', 0.0),
            'planetary_resonance': getattr(soul_spark, 'planetary_resonance', 0.0),
            'gaia_connection': getattr(soul_spark, 'gaia_connection', 0.0)
        }

        # FIXED: Store initial values for later comparison in finalization
        setattr(soul_spark, 'initial_coherence_before_earth', soul_spark.coherence)
        setattr(soul_spark, 'initial_stability_before_earth', soul_spark.stability)
        
        logger.info(f"Earth Initial State: S={initial_state['stability_su']:.1f}, "
                    f"C={initial_state['coherence_cu']:.1f}, "
                    f"EarthRes={initial_state['earth_resonance']:.3f}")
        
        # Create state dictionary to track process
        echo_state = {
            'schumann_resonance': 0.0,
            'earth_core_resonance': 0.0,
            'earth_resonance': 0.0,
            'stages_completed': []
        }
        
        # --- Step 1: Check Natural Earth Compatibility FIRST ---
        logger.info("Step 1: Checking natural earth compatibility...")
        compatibility_metrics = _check_natural_earth_compatibility(soul_spark)
        process_metrics_summary['steps']['compatibility_check'] = compatibility_metrics
        
        # --- Step 2: Apply Natural Enhancement if Compatible ---
        logger.info("Step 2: Applying natural earth resonance enhancement...")
        if compatibility_metrics['overall_compatibility'] >= 0.2:  # Lower threshold
            enhancement_metrics = _enhance_natural_earth_resonance(soul_spark)
            process_metrics_summary['steps']['natural_enhancement'] = enhancement_metrics
            logger.info(f"Applied natural earth resonance enhancement: {enhancement_metrics}")
        else:
            logger.info(f"Soul has minimal earth compatibility ({compatibility_metrics['overall_compatibility']:.3f}) - proceeding with natural assessment only")
            enhancement_metrics = {}
            process_metrics_summary['steps']['natural_enhancement'] = enhancement_metrics
        
        # --- Step 3: Attune to Schumann Resonance (with aligned frequencies) ---
        logger.info("Step 3: Attuning to Schumann resonance with aligned frequencies...")
        schumann_metrics = _attune_to_schumann_resonance(soul_spark)
        echo_state['schumann_resonance'] = schumann_metrics.get('schumann_weighted_average', 0.0)
        echo_state['stages_completed'].append('schumann')
        process_metrics_summary['steps']['schumann'] = schumann_metrics
        
        # --- Step 4: Attune to Earth Core (with aligned frequencies) ---
        logger.info("Step 4: Attuning to Earth core frequency with aligned frequencies...")
        core_metrics = _attune_to_earth_core(soul_spark)
        echo_state['earth_core_resonance'] = core_metrics.get('core_resonance', 0.0)
        echo_state['stages_completed'].append('earth_core')
        process_metrics_summary['steps']['earth_core'] = core_metrics
        
        # --- Step 5: Recalculate Final Earth Resonances ---
        logger.info("Step 5: Recalculating final Earth resonances with all alignments...")
        recalc_metrics = _recalculate_earth_resonances_after_alignment(soul_spark)
        echo_state['earth_resonance'] = recalc_metrics.get('recalculated_earth_resonance', 0.0)
        echo_state['stages_completed'].append('recalculation')
        process_metrics_summary['steps']['recalculation'] = recalc_metrics
        
        # --- Step 6: Integrate Light Spectrum ---
        logger.info("Step 6: Integrating light spectrum...")
        light_metrics = _perform_light_spectrum_integration(soul_spark)
        echo_state['light_resonance'] = light_metrics.get('light_resonance', 0.0)
        echo_state['stages_completed'].append('light_spectrum')
        process_metrics_summary['steps']['light_spectrum'] = light_metrics
        
        # --- Step 7: Calculate Earth Cycle Resonance ---
        logger.info("Step 7: Calculating Earth cycle resonance...")
        cycle_metrics = calculate_earth_cycle_resonance(soul_spark)
        echo_state['cycle_resonance'] = cycle_metrics.get('average_resonance', 0.0)
        echo_state['stages_completed'].append('earth_cycles')
        process_metrics_summary['steps']['earth_cycles'] = cycle_metrics
        
        # --- Step 8: Calculate Elemental Balance ---
        logger.info("Step 8: Calculating elemental balance...")
        element_metrics = calculate_elemental_balance(soul_spark)
        echo_state['element_balance'] = element_metrics.get('elements', {})
        echo_state['stages_completed'].append('elements')
        process_metrics_summary['steps']['elements'] = element_metrics
        
        # --- Step 9: Calculate Planetary Resonance ---
        logger.info("Step 9: Calculating planetary resonance...")
        planet_metrics = _calculate_planetary_resonance(soul_spark)
        echo_state['planetary_resonance'] = planet_metrics.get('overall_resonance', 0.0)
        echo_state['stages_completed'].append('planetary')
        process_metrics_summary['steps']['planetary'] = planet_metrics
        
        # --- Step 10: Optimize Gaia Connection ---
        logger.info("Step 10: Optimizing Gaia connection...")
        gaia_metrics = _optimize_gaia_connection(soul_spark, echo_state['earth_resonance'])
        echo_state['gaia_connection'] = gaia_metrics.get('new_gaia_connection', 0.0)
        echo_state['stages_completed'].append('gaia')
        process_metrics_summary['steps']['gaia'] = gaia_metrics
        
        # --- Step 11: Project Echo Field ---
        logger.info("Step 11: Projecting echo field...")
        echo_metrics = _project_echo_field(soul_spark, echo_state)
        echo_state['echo_projected'] = echo_metrics.get('echo_id') is not None
        echo_state['stages_completed'].append('echo')
        process_metrics_summary['steps']['echo'] = echo_metrics
        
        # --- Step 12: Finalize Earth Harmonization ---
        logger.info("Step 12: Finalizing Earth harmonization...")
        final_metrics = finalize_earth_harmonization(soul_spark, echo_state)
        echo_state['finalized'] = True
        echo_state['stages_completed'].append('finalization')
        process_metrics_summary['steps']['finalization'] = final_metrics
        
        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)
        
        final_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'earth_resonance': getattr(soul_spark, 'earth_resonance', 0.0),
            'planetary_resonance': getattr(soul_spark, 'planetary_resonance', 0.0),
            'gaia_connection': getattr(soul_spark, 'gaia_connection', 0.0),
            FLAG_EARTH_ATTUNED: getattr(soul_spark, FLAG_EARTH_ATTUNED, False),
            FLAG_ECHO_PROJECTED: getattr(soul_spark, FLAG_ECHO_PROJECTED, False)
        }
        
        overall_metrics = {
            'action': 'earth_harmonization',
            'soul_id': spark_id,
            'start_time': start_time_iso,
            'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'schumann_intensity': schumann_intensity,
            'core_intensity': core_intensity,
            'initial_state': initial_state,
            'final_state': final_state,
            'earth_resonance_change': final_state['earth_resonance'] - initial_state['earth_resonance'],
            'planetary_resonance_change': final_state['planetary_resonance'] - initial_state['planetary_resonance'],
            'gaia_connection_change': final_state['gaia_connection'] - initial_state['gaia_connection'],
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'stages_completed': echo_state['stages_completed'],
            'success': True
        }
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('earth_harmonization_summary', overall_metrics)
        
        logger.info(f"--- Earth Harmonization Completed Successfully for Soul {spark_id} ---")
        logger.info(f"  Final State: EarthRes={final_state['earth_resonance']:.3f}, "
                    f"GaiaConn={final_state['gaia_connection']:.3f}, "
                    f"S={final_state['stability_su']:.1f} ({overall_metrics['stability_change_su']:+.1f}), "
                    f"C={final_state['coherence_cu']:.1f} ({overall_metrics['coherence_change_cu']:+.1f})")
        
        return soul_spark, overall_metrics
        
    except (ValueError, TypeError, AttributeError) as e_val:
        logger.error(f"Earth Harmonization failed for {spark_id}: {e_val}", exc_info=True)
        failed_step = echo_state.get('stages_completed', [])[-1] if echo_state.get('stages_completed', []) else 'prerequisites/validation'
        _record_harmonization_failure(spark_id, start_time_iso, failed_step, str(e_val))
        raise e_val  # Hard fail
        
    except RuntimeError as e_rt:
        logger.critical(f"Earth Harmonization failed critically for {spark_id}: {e_rt}", exc_info=True)
        failed_step = echo_state.get('stages_completed', [])[-1] if echo_state.get('stages_completed', []) else 'runtime'
        _record_harmonization_failure(spark_id, start_time_iso, failed_step, str(e_rt))
        # Ensure flags are not set on failure
        setattr(soul_spark, FLAG_EARTH_ATTUNED, False)
        setattr(soul_spark, FLAG_ECHO_PROJECTED, False)
        raise e_rt  # Hard fail
        
    except Exception as e:
        logger.critical(f"Unexpected error during Earth Harmonization for {spark_id}: {e}", exc_info=True)
        failed_step = echo_state.get('stages_completed', [])[-1] if echo_state.get('stages_completed', []) else 'unexpected'
        _record_harmonization_failure(spark_id, start_time_iso, failed_step, str(e))
        # Ensure flags are not set on failure
        setattr(soul_spark, FLAG_EARTH_ATTUNED, False)
        setattr(soul_spark, FLAG_ECHO_PROJECTED, False)
        raise RuntimeError(f"Unexpected Earth Harmonization failure: {e}") from e  # Hard fail

def _record_harmonization_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """Helper to record failure metrics consistently."""
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('earth_harmonization_summary', {
                'action': 'earth_harmonization',
                'soul_id': spark_id,
                'start_time': start_time_iso,
                'end_time': end_time,
                'duration_seconds': duration,
                'success': False,
                'error': error_msg,
                'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record Earth Harmonization failure metrics for {spark_id}: {metric_e}")
                
