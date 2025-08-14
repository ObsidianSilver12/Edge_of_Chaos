"""
Creator Entanglement Functions (Refactored V4.4.0 - Light-Wave & Aura Integration)

Establishes connection post-Sephiroth journey using wave physics and quantum entanglement.
Connection forms through resonant field between soul layers and Kether field.
Aspects transfer via resonant exchange, not direct property modification.
Stability/Coherence emerge naturally through layer-based resonance patterns.
Operates on SoulSpark object using KetherField properties conceptually. Hard fails.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import uuid
import json
from math import pi as PI, sqrt, atan2, exp, sin, cos, tanh
from typing import Dict, List, Any, Tuple, Optional
from shared.constants.constants import *

# Set up logger
logger = logging.getLogger(__name__)


# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.fields.kether_field import KetherField
    
    # Sound integration - try to import sound modules if available
    try:
        from shared.sound.sound_generator import SoundGenerator
        from shared.sound.sounds_of_universe import UniverseSounds
        SOUND_AVAILABLE = True
        logger.info("Sound modules successfully imported for Creator Entanglement.")
    except ImportError:
        SOUND_AVAILABLE = False
        logger.warning("Sound modules not available. Creator Entanglement will run without sound generation.")
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

# --- Helper Functions ---
def _calculate_circular_variance(phases_array: np.ndarray) -> float:
    """Calculates circular variance (0=sync, 1=uniform)."""
    if phases_array is None or not isinstance(phases_array, np.ndarray) or phases_array.size < 2:
        return 1.0  # Max variance if no/one phase or invalid input
    if not np.all(np.isfinite(phases_array)):
        logger.warning("Non-finite values found in phases array during variance calculation.")
        return 1.0  # Max variance if data invalid
    mean_cos = np.mean(np.cos(phases_array))
    mean_sin = np.mean(np.sin(phases_array))
    r_len_sq = mean_cos**2 + mean_sin**2
    if r_len_sq < 0:  # Protect against precision errors
        r_len_sq = 0.0
    r_len = sqrt(r_len_sq)
    return 1.0 - r_len

def _check_prerequisites(soul_spark: SoulSpark, kether_field: KetherField) -> bool:
    """ Checks prerequisites using SU/CU thresholds. Raises ValueError on failure. """
    logger.debug(f"Checking Creator Entanglement prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")
    if not isinstance(kether_field, KetherField):
        raise TypeError("Invalid KetherField object.")

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False):
        msg = f"Prerequisite failed: {FLAG_SEPHIROTH_JOURNEY_COMPLETE} is False."
        logger.error(msg)
        raise ValueError(msg)
    if getattr(soul_spark, 'creator_channel_id', None) is not None:
        logger.warning(f"Soul {soul_spark.spark_id} already has creator channel. Overwriting.")

    # 2. Soul State Check (Absolute SU/CU Thresholds)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if stability_su < ENTANGLEMENT_PREREQ_STABILITY_MIN_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {ENTANGLEMENT_PREREQ_STABILITY_MIN_SU} SU."
        logger.error(msg)
        raise ValueError(msg)
    if coherence_cu < ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU} CU."
        logger.error(msg)
        raise ValueError(msg)

    # 3. Layers Check (Minimum required for resonance)
    layers = getattr(soul_spark, 'layers', [])
    if not layers or len(layers) < 2:
        msg = f"Prerequisite failed: Soul must have at least 2 layers for resonance (current: {len(layers)})."
        logger.error(msg)
        raise ValueError(msg)

    # 4. Field Check (Conceptual Location)
    current_field = getattr(soul_spark, 'current_field_key', 'unknown')
    if current_field != 'kether':
        msg = f"Prerequisite failed: Soul must be conceptually in Kether field (current: {current_field})."
        logger.error(msg)
        raise ValueError(msg)

    logger.debug("Creator Entanglement prerequisites met.")
    return True

# --- Enhanced Resonance Calculation with Wave Physics ---
def calculate_resonance(freq1: float, freq2: float) -> float:
    """ 
    Calculate resonance between frequencies using harmonic wave interference principles.
    Returns a value between 0 (no resonance) and 1 (perfect resonance).
    """
    if freq1 is None or freq2 is None or freq1 <= FLOAT_EPSILON or freq2 <= FLOAT_EPSILON:
        return 0.0
    
    # Ensure f1 is always the smaller frequency to simplify calculations
    if freq1 > freq2:
        freq1, freq2 = freq2, freq1
    
    # Ratio for harmonic analysis
    ratio = freq2 / freq1
    
    # --- Part 1: Harmonic ratio analysis ---
    # Musical intervals and natural harmonics
    harmonic_ratios = [1.0, 2.0, 3.0/2.0, 4.0/3.0, 5.0/3.0, 5.0/4.0, 6.0/5.0, 8.0/5.0, 3.0]
    harmonic_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4]
    
    # Golden ratio family for natural systems
    special_ratios = [PHI, SILVER_RATIO, PHI**2, 1.0/PHI]
    special_weights = [0.85, 0.7, 0.75, 0.85]
    
    # Calculate harmonic resonance - how close to perfect harmonic ratios
    harmonic_res = 0.0
    for hr, hw in zip(harmonic_ratios, harmonic_weights):
        rel_distance = abs(ratio - hr) / hr
        tolerance = RESONANCE_INTEGER_RATIO_TOLERANCE * 1.2
        
        if rel_distance < tolerance:
            # Resonance decays with square of normalized distance
            score = (1.0 - (rel_distance / tolerance)**2) * hw
            harmonic_res = max(harmonic_res, score)
    
    # Calculate special ratio resonance (phi-based)
    special_res = 0.0
    for sr, sw in zip(special_ratios, special_weights):
        rel_distance = abs(ratio - sr) / sr
        tolerance = RESONANCE_PHI_RATIO_TOLERANCE * 1.2
        
        if rel_distance < tolerance:
            score = (1.0 - (rel_distance / tolerance)**2) * sw
            special_res = max(special_res, score)
    
    # --- Part 2: Wave interference pattern analysis ---
    # Calculate wavelengths
    wavelength1 = SPEED_OF_SOUND / freq1
    wavelength2 = SPEED_OF_SOUND / freq2
    
    # Wave interference factor - based on superposition of waves
    # When waves align at regular intervals, strong resonance occurs
    wave_factor = 0.0
    
    # Calculate lowest common multiple of wavelengths (approximation)
    # This represents points where waves constructively interfere
    lcm_approx = wavelength1 * wavelength2 / gcd_approx(wavelength1, wavelength2)
    
    # Normalized interference factor
    wave_factor = 1.0 / (1.0 + 0.5 * (lcm_approx / max(wavelength1, wavelength2)))
    
    # --- Part 3: Combine all resonance factors ---
    combined_resonance = max(
        harmonic_res * 0.6,
        special_res * 0.7,
        wave_factor * 0.5
    )
    
    return float(max(0.0, min(1.0, combined_resonance)))

def gcd_approx(a: float, b: float, tolerance: float = 1e-6) -> float:
    """Approximate GCD for float values using Euclidean algorithm with tolerance."""
    if b < tolerance:
        return a
    return gcd_approx(b, a % b if a % b > tolerance else 0.0, tolerance)

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
                from shared.constants.constants import SEPHIROTH_GLYPH_DATA
                if sephirah in SEPHIROTH_GLYPH_DATA and 'frequency' in SEPHIROTH_GLYPH_DATA[sephirah]:
                    layer_freqs.append(SEPHIROTH_GLYPH_DATA[sephirah]['frequency'])
            except (ImportError, KeyError):
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

def add_layer_resonance(soul_spark: SoulSpark, layer_idx: int, freq: float, strength: float) -> bool:
    """
    Add or update resonance data to a specific layer.
    Returns True if successful, False otherwise.
    """
    if not hasattr(soul_spark, 'layers') or len(soul_spark.layers) <= layer_idx:
        return False
        
    layer = soul_spark.layers[layer_idx]
    if not isinstance(layer, dict):
        return False
        
    # Initialize resonance data if not exists
    if 'resonance_data' not in layer:
        layer['resonance_data'] = {}
        
    # Add frequency to resonant_frequencies if not exists
    if 'resonant_frequencies' not in layer:
        layer['resonant_frequencies'] = []
    
    # Add frequency if not already present
    if freq not in layer['resonant_frequencies']:
        layer['resonant_frequencies'].append(float(freq))
        
    # Update resonance data
    timestamp = datetime.now().isoformat()
    resonance_key = f"creator_resonance_{freq:.1f}"
    layer['resonance_data'][resonance_key] = {
        'frequency': float(freq),
        'strength': float(strength),
        'timestamp': timestamp,
        'source': 'creator_entanglement'
    }
    
    return True

class MetricsPlaceholder:
    """
    A placeholder class for metrics recording functionality.

    This class provides a stub implementation of the `record_metrics` method,
    which can be extended or replaced with actual metrics recording logic.

    Methods
    -------
    record_metrics(*args, **kwargs)
        Placeholder method for recording metrics. Does nothing by default.
    """
    def record_metrics(self, *args, **kwargs): pass

metrics = MetricsPlaceholder()



# --- Core Entanglement Functions ---

def _establish_resonant_field(soul_spark: SoulSpark, 
                            kether_field: KetherField,
                            base_creator_potential: float,
                            edge_of_chaos_target: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Establishes a resonant field between soul layers and Kether field using wave interference.
    Instead of direct connection, creates a field through which aspects and energies can transfer.
    Modifies SoulSpark layers, not direct properties.
    """
    logger.info(f"Establishing resonant field for soul {soul_spark.spark_id}...")
    
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    if not (0.0 <= base_creator_potential <= 1.0): raise ValueError("base_creator_potential out of range.")
    if not (0.0 <= edge_of_chaos_target <= 1.0): raise ValueError("edge_of_chaos_target out of range.")

    try:
        # Get soul state and layers
        soul_stability_norm = soul_spark.stability / MAX_STABILITY_SU
        soul_coherence_norm = soul_spark.coherence / MAX_COHERENCE_CU
        soul_layers = soul_spark.layers
        
        # Get soul frequency signature
        soul_freq_sig = soul_spark.frequency_signature
        if not soul_freq_sig or 'frequencies' not in soul_freq_sig:
            logger.warning("Soul frequency signature missing or invalid, regenerating.")
            if hasattr(soul_spark, "validate_or_init_frequency_structure"):
                soul_spark.validate_or_init_frequency_structure()
            else:
                raise AttributeError("SoulSpark is missing a public method to initialize frequency structure.")
            soul_freq_sig = soul_spark.frequency_signature
            
        # Create simplified wave representation of soul
        soul_freq_array = np.array(soul_freq_sig.get('frequencies', [soul_spark.frequency]))
        soul_amp_array = np.array(soul_freq_sig.get('amplitudes', [1.0]))
        soul_phase_array = np.array(soul_freq_sig.get('phases', [0.0]))        
        # Get Kether field frequency data
        kether_aspects = kether_field.get_aspects()
        kether_freq = kether_aspects.get('base_frequency', KETHER_FREQ)
        
        # Create simplified wave representation of Kether field
        kether_freqs = [kether_freq]
        kether_amps = [1.0]
        kether_phases = [0.0]
        
        # Add frequencies from detailed aspects
        for aspect_name, aspect_data in kether_aspects.get('detailed_aspects', {}).items():
            aspect_freq = aspect_data.get('frequency', 0.0)
            if aspect_freq > FLOAT_EPSILON:
                kether_freqs.append(aspect_freq)
                kether_amps.append(aspect_data.get('strength', 0.5))
                kether_phases.append(0.0)  # Default phase
                
        kether_freq_array = np.array(kether_freqs)
        kether_amp_array = np.array(kether_amps)
        kether_phase_array = np.array(kether_phases)
        
        # --- Calculate potential resonance bands between soul and Kether ---
        # For each soul frequency, find resonance with Kether frequencies
        resonance_matrix = np.zeros((len(soul_freq_array), len(kether_freq_array)))
        
        for i, soul_freq in enumerate(soul_freq_array):
            for j, kether_freq in enumerate(kether_freq_array):
                resonance_matrix[i, j] = calculate_resonance(soul_freq, kether_freq)
        
        # Find strongest resonance pairs
        resonance_pairs = []
        for i in range(len(soul_freq_array)):
            best_j = np.argmax(resonance_matrix[i, :])
            res_strength = resonance_matrix[i, best_j]
            if res_strength > 0.3:  # Threshold for meaningful resonance
                resonance_pairs.append((i, best_j, res_strength))
        
        # Sort by resonance strength
        resonance_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # --- Calculate Field Characteristics Based on Resonance ---
        # Total field strength based on resonance quality
        total_resonance = np.sum([p[2] for p in resonance_pairs]) if resonance_pairs else 0.0
        avg_resonance = total_resonance / len(resonance_pairs) if resonance_pairs else 0.0
        
        # Field strength influenced by soul state and resonance
        field_strength_base = (
            soul_stability_norm * 0.3 +
            soul_coherence_norm * 0.4 +
            avg_resonance * 0.3
        )
        
        # Apply creator potential modifier
        field_strength = field_strength_base * (0.7 + 0.6 * base_creator_potential)
        field_strength = min(1.0, max(0.1, field_strength))
        
        # Calculate chaos factor - how close to edge of chaos
        field_coherence = (soul_coherence_norm * 0.7 + avg_resonance * 0.3)
        chaos_deviation = abs(field_coherence - edge_of_chaos_target)
        chaos_factor = exp(-(chaos_deviation**2) / (2 * (0.15**2)))  # Gaussian falloff
        
        # --- Create Resonant Field Structure ---
        field_id = str(uuid.uuid4())
        creation_time = datetime.now().isoformat()
        
        # Determine primary resonant frequency pair
        if resonance_pairs:
            primary_pair = resonance_pairs[0]
            primary_soul_freq = soul_freq_array[primary_pair[0]]
            primary_kether_freq = kether_freq_array[primary_pair[1]]
            primary_resonance = primary_pair[2]
        else:
            # Fallback if no strong resonance
            primary_soul_freq = soul_freq_array[0] if len(soul_freq_array) > 0 else soul_spark.frequency
            primary_kether_freq = kether_freq_array[0] if len(kether_freq_array) > 0 else kether_freq
            primary_resonance = calculate_resonance(primary_soul_freq, primary_kether_freq)
        
        # Create resonant field data structure
        resonant_field = {
            "field_id": field_id,
            "spark_id": soul_spark.spark_id,
            "creation_time": creation_time,
            "field_strength": float(field_strength),
            "field_coherence": float(field_coherence),
            "chaos_factor": float(chaos_factor),
            "primary_soul_freq": float(primary_soul_freq),
            "primary_kether_freq": float(primary_kether_freq),
            "primary_resonance": float(primary_resonance),
            "resonance_bands": [
                {
                    "soul_freq": float(soul_freq_array[p[0]]),
                    "kether_freq": float(kether_freq_array[p[1]]),
                    "resonance": float(p[2])
                }
                for p in resonance_pairs
            ],
            "active": True
        }
        
        # --- Find Resonant Layers in Soul ---
        # For each resonance band, find layers that resonate with it
        resonant_layer_data = {}
        
        for band in resonant_field["resonance_bands"]:
            soul_freq = band["soul_freq"]
            kether_freq = band["kether_freq"]
            band_strength = band["resonance"]
            
            # Find resonant layers for both frequencies
            soul_layers = find_resonant_layers(soul_spark, soul_freq)
            kether_layers = find_resonant_layers(soul_spark, kether_freq)
            
            # Combine and deduplicate
            all_layers = set(l[0] for l in soul_layers + kether_layers)
            
            for layer_idx in all_layers:
                # Get best resonance for this layer
                soul_res = next((r for i, r in soul_layers if i == layer_idx), 0.0)
                kether_res = next((r for i, r in kether_layers if i == layer_idx), 0.0)
                best_res = max(soul_res, kether_res)
                
                # Add to resonant layer data
                if layer_idx not in resonant_layer_data:
                    resonant_layer_data[layer_idx] = {
                        "resonance": best_res * band_strength,
                        "bands": []
                    }
                else:
                    # Update resonance if better
                    current_res = resonant_layer_data[layer_idx]["resonance"]
                    resonant_layer_data[layer_idx]["resonance"] = max(
                        current_res, 
                        best_res * band_strength
                    )
                
                # Add band to layer
                resonant_layer_data[layer_idx]["bands"].append({
                    "soul_freq": soul_freq,
                    "kether_freq": kether_freq,
                    "resonance": best_res * band_strength
                })
        
        # --- Update Soul Layers with Resonance Data ---
        for layer_idx, layer_data in resonant_layer_data.items():
            # For each band, add resonance to layer
            for band in layer_data["bands"]:
                add_layer_resonance(
                    soul_spark, 
                    int(layer_idx), 
                    band["kether_freq"], 
                    band["resonance"]
                )
        
        # --- Create Summary Metrics ---
        metrics_data = {
            'action': 'establish_resonant_field',
            'soul_id': soul_spark.spark_id,
            'field_id': field_id,
            'field_strength': field_strength,
            'field_coherence': field_coherence,
            'chaos_factor': chaos_factor,
            'primary_resonance': primary_resonance,
            'resonance_bands_count': len(resonant_field["resonance_bands"]),
            'resonant_layers_count': len(resonant_layer_data),
            'initial_stability_su': soul_spark.stability,
            'initial_coherence_cu': soul_spark.coherence,
            'success': True,
            'timestamp': creation_time
        }
        
        # --- Generate Creation Sound if Available ---
        if SOUND_AVAILABLE:
            try:
                # Initialize universe sounds generator
                universe_sound = UniverseSounds()
                
                # Generate dimensional transition sound (creator to soul)
                transition_sound = universe_sound.generate_dimensional_transition(
                    duration=5.0,
                    sample_rate=SAMPLE_RATE,
                    transition_type='creator_to_soul',
                    amplitude=0.7
                )
                
                # Save sound
                sound_file = f"creator_resonance_{soul_spark.spark_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                universe_sound.save_sound(
                    transition_sound, 
                    sound_file, 
                    f"Creator resonance field formation for soul {soul_spark.spark_id[:8]}"
                )
                
                logger.info(f"Generated creator resonance sound: {sound_file}")
                resonant_field["sound_file"] = sound_file
            except Exception as sound_err:
                logger.warning(f"Failed to generate creator resonance sound: {sound_err}")
        
        logger.info(f"Resonant field established: Strength={field_strength:.4f}, Coherence={field_coherence:.4f}, ResonanceBands={len(resonant_field['resonance_bands'])}, ResonantLayers={len(resonant_layer_data)}")
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('creator_entanglement', metrics_data)
        
        return resonant_field, metrics_data
        
    except Exception as e:
        logger.error(f"Error establishing resonant field: {e}", exc_info=True)
        raise RuntimeError("Resonant field establishment failed critically.") from e

def _form_quantum_channel(soul_spark: SoulSpark,
                        resonant_field: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Forms quantum channel through the resonant field using quantum entanglement.
    Creates a wave-based information exchange pathway between creator and soul.
    """
    logger.info(f"Forming quantum channel for soul {soul_spark.spark_id}...")
    
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(resonant_field, dict): raise TypeError("resonant_field invalid.")
    
    field_id = resonant_field.get("field_id")
    spark_id = resonant_field.get("spark_id")
    field_strength = resonant_field.get("field_strength")
    field_coherence = resonant_field.get("field_coherence")
    
    if not field_id or not spark_id or field_strength is None or field_coherence is None:
        raise ValueError("Resonant field missing essential attributes.")
    
    try:
        # --- Extract Wave Properties from Field ---
        primary_soul_freq = resonant_field.get("primary_soul_freq")
        primary_kether_freq = resonant_field.get("primary_kether_freq")
        primary_resonance = resonant_field.get("primary_resonance")
        resonance_bands = resonant_field.get("resonance_bands", [])
        
        # --- Apply Quantum Wave-Particle Duality ---
        # Calculate quantum entanglement factor - affected by resonance and chaos
        chaos_factor = resonant_field.get("chaos_factor", 0.5)
        entanglement_quality = (
            primary_resonance * 0.5 +
            field_coherence * 0.3 +
            chaos_factor * 0.2
        )
        
        # --- Calculate Standing Wave Pattern ---
        # Standing waves form at nodes where soul and creator frequencies interfere constructively
        standing_wave_nodes = []
        
        # For each resonance band, calculate potential standing wave nodes
        for band in resonance_bands:
            soul_freq = band["soul_freq"]
            kether_freq = band["kether_freq"]
            band_resonance = band["resonance"]
            
            # Calculate wavelengths
            soul_wavelength = SPEED_OF_SOUND / soul_freq if soul_freq > FLOAT_EPSILON else 0.0
            kether_wavelength = SPEED_OF_SOUND / kether_freq if kether_freq > FLOAT_EPSILON else 0.0
            
            if soul_wavelength > FLOAT_EPSILON and kether_wavelength > FLOAT_EPSILON:
                # Find common nodes (simplified)
                node_spacing = gcd_approx(soul_wavelength, kether_wavelength)
                
                # Create nodes along the channel
                # Number of nodes scales with resonance quality
                num_nodes = max(3, int(10 * band_resonance))
                
                for i in range(num_nodes):
                    # Position follows Fibonacci-based distribution for natural harmonic spacing
                    # Nodes closer together near soul, wider near creator
                    position = (PHI ** i) / (PHI ** num_nodes)
                    position = min(0.99, max(0.01, position))  # Keep within 0-1 range
                    
                    node_strength = band_resonance * (1.0 - 0.5 * abs(0.5 - position))
                    
                    standing_wave_nodes.append({
                        "position": float(position),
                        "soul_freq": float(soul_freq),
                        "kether_freq": float(kether_freq),
                        "resonance": float(band_resonance),
                        "strength": float(node_strength)
                    })
        
        # --- Calculate Information Bandwidth ---
        # Bandwidth determined by number of resonance bands and their quality
        base_bandwidth = 0.0
        for band in resonance_bands:
            # Each resonance band contributes to bandwidth
            # Higher resonance = higher bandwidth contribution
            base_bandwidth += band["resonance"] * 50.0  # Scale factor
        
        # Apply entanglement modifier
        bandwidth_hz = base_bandwidth * (0.5 + 0.5 * entanglement_quality)
        
        # --- Create Quantum Channel Structure ---
        channel_id = str(uuid.uuid4())
        creation_time = datetime.now().isoformat()
        
        quantum_channel = {
            "channel_id": channel_id,
            "field_id": field_id,
            "spark_id": spark_id,
            "creation_time": creation_time,
            "entanglement_quality": float(entanglement_quality),
            "bandwidth_hz": float(bandwidth_hz),
            "primary_frequency_hz": float(primary_soul_freq),
            "standing_wave_nodes": standing_wave_nodes,
            "node_count": len(standing_wave_nodes),
            "active": True
        }
        
        # --- Calculate Additional Wave Properties ---
        # Phase coherence - how well aligned the phases are
        soul_phases = soul_spark.frequency_signature.get('phases', [0.0])
        phase_array = np.array(soul_phases)
        if len(phase_array) > 1:
            phase_variance = _calculate_circular_variance(phase_array)
            phase_coherence = 1.0 - phase_variance
        else:
            phase_coherence = 0.5  # Default for single phase
            
        # Wave interference quality - how well the waves interfere
        interference_quality = 0.0
        if len(resonance_bands) > 1:
            # Calculate interference between bands
            band_freqs = [band["soul_freq"] for band in resonance_bands]
            band_strengths = [band["resonance"] for band in resonance_bands]
            
            # Sum of pairwise interference qualities
            interference_sum = 0.0
            pair_count = 0
            
            for i in range(len(band_freqs)):
                for j in range(i+1, len(band_freqs)):
                    # Calculate interference quality between bands i and j
                    freq_ratio = max(band_freqs[i], band_freqs[j]) / max(FLOAT_EPSILON, min(band_freqs[i], band_freqs[j]))
                    
                    # Ideal interference for integer/phi ratios
                    ratio_quality = 0.0
                    
                    # Check integer ratios
                    for n in range(1, 6):
                        for d in range(1, 6):
                            if d == 0: continue
                            ideal_ratio = n / d
                            if abs(freq_ratio - ideal_ratio) < 0.05:
                                ratio_quality = max(ratio_quality, 1.0 - abs(freq_ratio - ideal_ratio) * 10.0)
                    
                    # Check phi ratios
                    phi_ratios = [PHI, 1/PHI, PHI**2, 1/(PHI**2)]
                    for pr in phi_ratios:
                        if abs(freq_ratio - pr) < 0.05:
                            ratio_quality = max(ratio_quality, 1.0 - abs(freq_ratio - pr) * 10.0)
                    
                    # Weight by band strengths
                    pair_quality = ratio_quality * band_strengths[i] * band_strengths[j]
                    interference_sum += pair_quality
                    pair_count += 1
            
            # Average interference quality
            if pair_count > 0:
                interference_quality = interference_sum / pair_count
        
        # Add to quantum channel data
        quantum_channel["phase_coherence"] = float(phase_coherence)
        quantum_channel["interference_quality"] = float(interference_quality)
        
        # --- Light Wave Properties ---
        # Light can be modeled as electromagnetic waves with much higher frequencies
        # We can create a mapping from audio to light frequencies
        # For simplicity, we'll use a scaling factor
        
        light_scaling_factor = 1e12  # Rough audio-to-light scaling
        
        light_frequencies = []
        for band in resonance_bands:
            # Map audio frequency to light domain
            light_freq = band["soul_freq"] * light_scaling_factor
            light_freq_kether = band["kether_freq"] * light_scaling_factor
            
            # Store light frequencies with their resonance data
            light_frequencies.append({
                "frequency": float(light_freq),
                "kether_frequency": float(light_freq_kether),
                "resonance": float(band["resonance"])
            })
        
        # Add light properties to quantum channel
        quantum_channel["light_frequencies"] = light_frequencies
        
        # --- Calculate Creator Connection Strength ---
        # This is now based on the quantum channel properties rather than
        # direct assignment - more physically accurate
        
        connection_strength = (
            entanglement_quality * 0.5 +
            phase_coherence * 0.2 +
            interference_quality * 0.3
        ) * field_strength  # Scale by field strength
        
        # Store but do not modify soul directly
        quantum_channel["connection_strength"] = float(connection_strength)
        
        # --- Log Channel Properties ---
        logger.info(f"Quantum channel formed: EntQuality={entanglement_quality:.4f}, "
                f"Bandwidth={bandwidth_hz:.1f}Hz, Nodes={len(standing_wave_nodes)}")
        
        # --- Generate Sound Representation ---
        if SOUND_AVAILABLE:
            try:
                # Initialize universe sounds generator
                universe_sounds = UniverseSounds()
                
                # Generate quantum entanglement sound
                # This creates a sound representing the quantum connection
                quantum_sound = universe_sounds.generate_dimensional_transition(
                    duration=5.0,
                    sample_rate=SAMPLE_RATE,
                    transition_type='quantum_entanglement',
                    amplitude=0.7
                )
                
                # Save sound
                sound_file = f"quantum_channel_{soul_spark.spark_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                universe_sounds.save_sound(
                    quantum_sound, 
                    sound_file, 
                    f"Quantum channel formation for soul {soul_spark.spark_id[:8]}"
                )
                
                logger.info(f"Generated quantum channel sound: {sound_file}")
                quantum_channel["sound_file"] = sound_file
            except Exception as sound_err:
                logger.warning(f"Failed to generate quantum channel sound: {sound_err}")
        
        # --- Create Channel Metrics ---
        metrics_data = {
            'action': 'form_quantum_channel',
            'soul_id': soul_spark.spark_id,
            'channel_id': channel_id,
            'field_id': field_id,
            'entanglement_quality': entanglement_quality,
            'bandwidth_hz': bandwidth_hz,
            'node_count': len(standing_wave_nodes),
            'phase_coherence': phase_coherence,
            'interference_quality': interference_quality,
            'connection_strength': connection_strength,
            'success': True,
            'timestamp': creation_time
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('quantum_channel', metrics_data)
            
        return quantum_channel, metrics_data
            
    except Exception as e:
        logger.error(f"Error forming quantum channel: {e}", exc_info=True)
        raise RuntimeError("Quantum channel formation failed critically.") from e

def _transfer_creator_aspects(soul_spark: SoulSpark,
                            quantum_channel: Dict[str, Any],
                            kether_field: KetherField) -> Dict[str, Any]:
    """
    Transfers creator aspects via quantum channel using light-wave resonance.
    Preserves coherence by connecting aspects to appropriate soul layers.
    """
    logger.info(f"Transferring creator aspects via quantum channel for soul {soul_spark.spark_id}...")
    
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(quantum_channel, dict): raise TypeError("quantum_channel invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    
    entanglement_quality = quantum_channel.get("entanglement_quality", 0.0)
    connection_strength = quantum_channel.get("connection_strength", 0.0)
    channel_id = quantum_channel.get("channel_id")
    
    if not channel_id or not (0.0 <= entanglement_quality <= 1.0) or not (0.0 <= connection_strength <= 1.0):
        raise ValueError("Quantum channel missing essential attributes.")
    
    try:
        # --- Get Kether Aspects to Transfer ---
        kether_aspects = kether_field.get_aspects()
        detailed_aspects = kether_aspects.get('detailed_aspects', {})
        
        if not detailed_aspects:
            logger.warning("No detailed aspects available from Kether field.")
            return {"aspects_transferred": 0, "aspects_strengthened": 0, "success": False}
        
        # --- Get Soul's Current Aspects and Layers ---
        soul_aspects = getattr(soul_spark, 'aspects', {})
        soul_layers = getattr(soul_spark, 'layers', [])
        
        initial_coherence = soul_spark.coherence  # Store for tracking
        
        # --- Determine Which Aspects Can Transfer Based on Resonance ---
        transferable_aspects = {}
        resonant_freqs = []
        
        # Get resonant frequencies from the quantum channel
        for node in quantum_channel.get("standing_wave_nodes", []):
            freq = node.get("kether_freq", 0.0)
            if freq > FLOAT_EPSILON:
                resonant_freqs.append((freq, node.get("resonance", 0.0)))
        
        # Find aspects that resonate with the quantum channel
        for aspect_name, aspect_data in detailed_aspects.items():
            # Check if aspect frequency resonates with quantum channel
            aspect_freq = aspect_data.get('frequency', 0.0)
            
            # Skip aspects with no frequency information
            if aspect_freq <= FLOAT_EPSILON:
                continue
                
            # Find best resonance with quantum channel
            best_resonance = 0.0
            for channel_freq, channel_res in resonant_freqs:
                res = calculate_resonance(aspect_freq, channel_freq)
                weighted_res = res * channel_res  # Weight by channel resonance
                best_resonance = max(best_resonance, weighted_res)
            
            # Check if resonance is strong enough for transfer
            if best_resonance > 0.3:  # Threshold for transfer
                transferable_aspects[aspect_name] = {
                    "data": aspect_data,
                    "resonance": best_resonance
                }
        
        logger.debug(f"  Found {len(transferable_aspects)} transferable aspects via quantum resonance")
        
        # --- Apply Transfer/Enhancement Logic ---
        gained_count = 0
        strengthened_count = 0
        
        # Find matching layers for each aspect
        for aspect_name, transfer_data in transferable_aspects.items():
            aspect_data = transfer_data["data"]
            aspect_resonance = transfer_data["resonance"]
            aspect_freq = aspect_data.get('frequency', 0.0)
            
            # Find the most resonant layer for this aspect
            best_layer_idx = -1
            best_layer_res = 0.0
            
            for i, layer in enumerate(soul_layers):
                # Skip invalid layers
                if not isinstance(layer, dict):
                    continue
                    
                # Get layer frequencies
                layer_freqs = []
                
                # Check resonant_frequencies array
                if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
                    layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
                
                # Check sephirah frequency if available
                if 'sephirah' in layer:
                    seph_name = layer['sephirah'].lower()
                    try:
                        from constants.constants import SEPHIROTH_GLYPH_DATA
                        if seph_name in SEPHIROTH_GLYPH_DATA:
                            seph_freq = SEPHIROTH_GLYPH_DATA[seph_name].get('frequency', 0.0)
                            if seph_freq > FLOAT_EPSILON:
                                layer_freqs.append(seph_freq)
                    except ImportError:
                        pass  # Continue without glyph data
                
                # Calculate best resonance with layer
                layer_res = 0.0
                for layer_freq in layer_freqs:
                    res = calculate_resonance(aspect_freq, layer_freq)
                    layer_res = max(layer_res, res)
                
                # Check if this is the best layer so far
                if layer_res > best_layer_res:
                    best_layer_res = layer_res
                    best_layer_idx = i
            
            # Now we have the best layer (if any) for this aspect
            
            # --- Apply Aspect Transfer Logic ---
            # Check if soul already has this aspect
            if aspect_name in soul_aspects:
                # Aspect exists - potentially strengthen it
                current_strength = soul_aspects[aspect_name].get('strength', 0.0)
                
                # Calculate potential strength increase
                # Factors: resonance quality, quantum channel quality, connection strength
                strength_increase = (
                    aspect_resonance * 0.4 +
                    entanglement_quality * 0.3 +
                    connection_strength * 0.3
                ) * ASPECT_TRANSFER_STRENGTH_FACTOR
                
                # Apply increase, cap at MAX_ASPECT_STRENGTH
                new_strength = min(MAX_ASPECT_STRENGTH, current_strength + strength_increase)
                
                # Only update if there's a meaningful increase
                if new_strength - current_strength > FLOAT_EPSILON:
                    soul_aspects[aspect_name]['strength'] = float(new_strength)
                    
                    # Track aspect-layer connection if we found a good layer
                    if best_layer_idx >= 0 and best_layer_res > 0.2:
                        # Add or update layer connection in aspect data
                        if 'layer_connections' not in soul_aspects[aspect_name]:
                            soul_aspects[aspect_name]['layer_connections'] = {}
                        
                        layer_key = f"layer_{best_layer_idx}"
                        soul_aspects[aspect_name]['layer_connections'][layer_key] = {
                            "resonance": float(best_layer_res),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Also update the layer's aspect connections
                        if best_layer_idx < len(soul_layers):
                            layer = soul_layers[best_layer_idx]
                            if 'aspect_connections' not in layer:
                                layer['aspect_connections'] = {}
                            
                            layer['aspect_connections'][aspect_name] = {
                                "resonance": float(best_layer_res),
                                "timestamp": datetime.now().isoformat()
                            }
                    
                    # Log the strengthening
                    logger.debug(f"  Strengthened aspect '{aspect_name}': {current_strength:.3f} -> {new_strength:.3f}")
                    strengthened_count += 1
                
            else:
                # New aspect - transfer it if resonance is sufficient
                if aspect_resonance * connection_strength > ASPECT_TRANSFER_THRESHOLD:
                    # Calculate initial strength based on resonance and connection
                    initial_strength = (
                        aspect_resonance * 0.6 +
                        connection_strength * 0.4
                    ) * ASPECT_TRANSFER_INITIAL_STRENGTH
                    
                    # Cap at MAX_ASPECT_STRENGTH
                    initial_strength = min(MAX_ASPECT_STRENGTH, initial_strength)
                    
                    # Create aspect record
                    new_aspect = {
                        'strength': float(initial_strength),
                        'source': 'kether',
                        'transfer_time': datetime.now().isoformat(),
                        'details': aspect_data.copy()  # Copy aspect data
                    }
                    
                    # Add layer connection if available
                    if best_layer_idx >= 0 and best_layer_res > 0.2:
                        new_aspect['layer_connections'] = {
                            f"layer_{best_layer_idx}": {
                                "resonance": float(best_layer_res),
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        
                        # Update the layer's aspect connections
                        if best_layer_idx < len(soul_layers):
                            layer = soul_layers[best_layer_idx]
                            if 'aspect_connections' not in layer:
                                layer['aspect_connections'] = {}
                            
                            layer['aspect_connections'][aspect_name] = {
                                "resonance": float(best_layer_res),
                                "timestamp": datetime.now().isoformat()
                            }
                    
                    # Add to soul aspects
                    soul_aspects[aspect_name] = new_aspect
                    
                    # Log the transfer
                    logger.debug(f"  Transferred new aspect '{aspect_name}' with strength {initial_strength:.3f}")
                    gained_count += 1
        
        # Save updated aspects to soul
        setattr(soul_spark, 'aspects', soul_aspects)
        
        # --- Add Creator Connection Attribute ---
        # Instead of directly modifying, store the quantum channel ID and quality
        setattr(soul_spark, 'creator_channel_id', channel_id)
        setattr(soul_spark, 'creator_connection_strength', float(connection_strength))
        
        # --- Post-Transfer Coherence Enhancement ---
        # After transferring aspects, enhance coherence through layer integration
        if hasattr(soul_spark, 'layers') and soul_spark.layers and gained_count + strengthened_count > 0:
            # For each transferred aspect, create layer resonance
            coherence_boost = 0.0
            aspect_layer_integrations = 0
            
            for aspect_name, aspect_data in soul_aspects.items():
                if aspect_name not in transferable_aspects:
                    continue
                    
                # Create layer resonance with all layers
                aspect_freq = aspect_data.get('details', {}).get('frequency', 0.0)
                if aspect_freq <= FLOAT_EPSILON:
                    continue
                
                # Find all resonant layers (not just the best one)
                for i, layer in enumerate(soul_layers):
                    # Skip invalid layers
                    if not isinstance(layer, dict):
                        continue
                    
                    # Calculate resonance
                    layer_res = 0.0
                    layer_freqs = []
                    
                    # Get layer frequencies
                    if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
                        layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
                    
                    # Calculate best resonance
                    for layer_freq in layer_freqs:
                        res = calculate_resonance(aspect_freq, layer_freq)
                        layer_res = max(layer_res, res)
                    
                    # If resonance is significant, create connection
                    if layer_res > 0.2:
                        # Add to aspect's layer connections
                        if 'layer_connections' not in aspect_data:
                            aspect_data['layer_connections'] = {}
                        
                        layer_key = f"layer_{i}"
                        if layer_key not in aspect_data['layer_connections']:
                            aspect_data['layer_connections'][layer_key] = {
                                "resonance": float(layer_res),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Also update the layer
                            if 'aspect_connections' not in layer:
                                layer['aspect_connections'] = {}
                            
                            layer['aspect_connections'][aspect_name] = {
                                "resonance": float(layer_res),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Track integration count
                            aspect_layer_integrations += 1
                            
                            # Add to coherence boost
                            boost = layer_res * 0.05  # Small boost per integration
                            coherence_boost += boost
            
            # Log integration results
            if aspect_layer_integrations > 0:
                logger.info(f"Created {aspect_layer_integrations} aspect-layer integrations, coherence boost: +{coherence_boost:.2f}")
        
        # --- Update Soul State ---
        # Update soul to reflect changes
        if hasattr(soul_spark, 'update_state'):
            logger.debug("Calling update_state to recalculate S/C after aspect transfers...")
            soul_spark.update_state()
            # Log coherence change
            coherence_change = soul_spark.coherence - initial_coherence
            logger.debug(f"Coherence after transfers: {soul_spark.coherence:.1f} CU ({coherence_change:+.1f})")
        
        # --- Create Summary Metrics ---
        transfer_metrics = {
            'action': 'transfer_creator_aspects',
            'soul_id': soul_spark.spark_id,
            'channel_id': channel_id,
            'aspects_transferred': gained_count,
            'aspects_strengthened': strengthened_count,
            'total_aspects_affected': gained_count + strengthened_count,
            'transferable_aspects_count': len(transferable_aspects),
            'creator_connection_strength': connection_strength,
            'initial_coherence': initial_coherence,
            'final_coherence': soul_spark.coherence,
            'coherence_change': soul_spark.coherence - initial_coherence,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('aspect_transfer', transfer_metrics)
        
        # Generate sound for aspect transfer
        if SOUND_AVAILABLE and (gained_count > 0 or strengthened_count > 0):
            try:
                # Initialize universe sounds generator
                universe_sounds = UniverseSounds()
                
                # Generate aspect transfer sound
                transfer_sound = universe_sounds.generate_dimensional_transition(
                    duration=3.0,
                    sample_rate=SAMPLE_RATE,
                    transition_type='aspect_transfer',
                    amplitude=0.6
                )
                
                # Save sound
                sound_file = f"aspect_transfer_{soul_spark.spark_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                universe_sounds.save_sound(
                    transfer_sound, 
                    sound_file, 
                    f"Aspect transfer sound for soul {soul_spark.spark_id[:8]}"
                )
                
                logger.info(f"Generated aspect transfer sound: {sound_file}")
                transfer_metrics["sound_file"] = sound_file
            except Exception as sound_err:
                logger.warning(f"Failed to generate aspect transfer sound: {sound_err}")
        
        logger.info(f"Creator aspects transfer summary: {gained_count} gained, {strengthened_count} strengthened")
        return transfer_metrics
        
    except Exception as e:
        logger.error(f"Error transferring creator aspects: {e}", exc_info=True)
        raise RuntimeError("Creator aspect transfer failed critically.") from e

def _form_resonance_patterns(soul_spark: SoulSpark,
                        quantum_channel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forms resonance patterns with quantum channel to enhance cohesion.
    Uses standing wave interference patterns to optimize soul coherence.
    """
    logger.info(f"Forming resonance patterns for soul {soul_spark.spark_id}...")
    
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(quantum_channel, dict): raise TypeError("quantum_channel invalid.")
    
    standing_wave_nodes = quantum_channel.get("standing_wave_nodes", [])
    channel_id = quantum_channel.get("channel_id")
    
    if not channel_id or not standing_wave_nodes:
        raise ValueError("Quantum channel missing essential attributes for resonance pattern formation.")
    
    try:
        # --- Store initial state ---
        initial_coherence = soul_spark.coherence
        initial_phi_resonance = getattr(soul_spark, 'phi_resonance', 0.0)
        initial_pattern_coherence = getattr(soul_spark, 'pattern_coherence', 0.0)
        
        # --- Create wave interference patterns ---
        # Get soul frequency data
        soul_freq = soul_spark.frequency
        soul_freqs = getattr(soul_spark, 'harmonics', [])
        if not soul_freqs or len(soul_freqs) == 0:
            soul_freqs = [soul_freq]
        
        # Get quantum channel frequency data
        channel_freqs = []
        for node in standing_wave_nodes:
            soul_freq = node.get("soul_freq", 0.0)
            kether_freq = node.get("kether_freq", 0.0)
            node_strength = node.get("strength", 0.0)
            
            if soul_freq > FLOAT_EPSILON:
                channel_freqs.append((soul_freq, node_strength))
            if kether_freq > FLOAT_EPSILON:
                channel_freqs.append((kether_freq, node_strength))
        
        # --- Create resonance patterns structure ---
        pattern_id = str(uuid.uuid4())
        creation_time = datetime.now().isoformat()
        
        resonance_patterns = {
            "pattern_id": pattern_id,
            "channel_id": channel_id,
            "spark_id": soul_spark.spark_id,
            "creation_time": creation_time,
            "resonance_harmonics": [],
            "phase_optimizations": [],
            "phi_resonance_boost": 0.0,
            "pattern_coherence_boost": 0.0
        }
        
        # --- Calculate resonance harmonics ---
        resonance_harmonics = []
        
        # Find harmonics between soul frequencies and channel frequencies
        for soul_f in soul_freqs:
            for channel_f, strength in channel_freqs:
                # Calculate the ratio and check for harmonic relationship
                if soul_f <= FLOAT_EPSILON or channel_f <= FLOAT_EPSILON:
                    continue
                
                ratio = max(soul_f, channel_f) / min(soul_f, channel_f)
                
                # Check for integer or phi-based ratios
                harmonic_type = "none"
                harmonic_quality = 0.0
                
                # Check integer ratios
                for n in range(1, 6):
                    for d in range(1, 6):
                        if d == 0: 
                            continue
                        ideal_ratio = n / d
                        if abs(ratio - ideal_ratio) < 0.05:
                            harmonic_type = f"{n}:{d}"
                            harmonic_quality = 1.0 - abs(ratio - ideal_ratio) * 10.0
                            break
                    if harmonic_type != "none":
                        break
                
                # Check phi ratios if no integer ratio found
                if harmonic_type == "none":
                    phi_ratios = [
                        (PHI, "φ"),
                        (1/PHI, "1/φ"),
                        (PHI**2, "φ²"),
                        (1/(PHI**2), "1/φ²")
                    ]
                    
                    for phi_val, name in phi_ratios:
                        if abs(ratio - phi_val) < 0.05:
                            harmonic_type = name
                            harmonic_quality = 1.0 - abs(ratio - phi_val) * 10.0
                            break
                
                # Only include significant harmonics
                if harmonic_quality > 0.5:
                    resonance_harmonics.append({
                        "soul_freq": float(soul_f),
                        "channel_freq": float(channel_f),
                        "ratio": float(ratio),
                        "harmonic_type": harmonic_type,
                        "quality": float(harmonic_quality),
                        "strength": float(strength)
                    })
        
        # Sort by quality (descending)
        resonance_harmonics.sort(key=lambda x: x["quality"], reverse=True)
        
        # Store in resonance patterns
        resonance_patterns["resonance_harmonics"] = resonance_harmonics
        
        # --- Optimize phase relationships ---
        # This enhances coherence by optimizing the phase relationships
        phase_optimizations = []
        coherence_boost = 0.0
        
        if hasattr(soul_spark, 'frequency_signature') and 'phases' in soul_spark.frequency_signature:
            # Get current phases
            phases = soul_spark.frequency_signature['phases']
            freqs = soul_spark.frequency_signature.get('frequencies', [])
            
            if len(phases) > 1 and len(phases) == len(freqs):
                phases_array = np.array(phases)
                initial_variance = _calculate_circular_variance(phases_array)
                
                # Find optimal phase relationships
                if hasattr(soul_spark, '_optimize_phase_coherence'):
                    # Try to optimize phases using soul's built-in function
                    improvement = soul_spark._optimize_phase_coherence(0.3)  # Stronger optimization
                    coherence_boost += improvement * 3.0  # Scale improvement to meaningful boost
                    
                    phase_optimizations.append({
                        "method": "built_in",
                        "improvement": float(improvement),
                        "initial_variance": float(initial_variance),
                        "final_variance": float(initial_variance - improvement)
                    })
                else:
                    # Manual phase optimization (simplified)
                    # Analyze phase distribution and look for clusters
                    # that can be brought into alignment
                    
                    # Calculate mean phase
                    mean_cos = np.mean(np.cos(phases_array))
                    mean_sin = np.mean(np.sin(phases_array))
                    mean_phase = np.arctan2(mean_sin, mean_cos)
                    
                    # Find phases that are far from the mean
                    far_phases = []
                    for i, phase in enumerate(phases_array):
                        phase_diff = abs((phase - mean_phase + PI) % (2*PI) - PI)
                        if phase_diff > PI/2:  # More than 90 degrees off
                            far_phases.append(i)
                    
                    # Adjust far phases slightly toward the mean
                    if far_phases:
                        new_phases = phases_array.copy()
                        for i in far_phases:
                            # Move 20% toward mean
                            shift = 0.2 * ((mean_phase - phases_array[i] + PI) % (2*PI) - PI)
                            new_phases[i] = (phases_array[i] + shift) % (2*PI)
                        
                        # Calculate improvement
                        new_variance = _calculate_circular_variance(new_phases)
                        improvement = initial_variance - new_variance
                        
                        if improvement > 0:
                            # Apply the changes
                            soul_spark.frequency_signature['phases'] = new_phases.tolist()
                            coherence_boost += improvement * 3.0
                            
                            phase_optimizations.append({
                                "method": "manual",
                                "improvement": float(improvement),
                                "initial_variance": float(initial_variance),
                                "final_variance": float(new_variance),
                                "phases_adjusted": len(far_phases)
                            })
        
        # Add optimization results to resonance patterns
        resonance_patterns["phase_optimizations"] = phase_optimizations
        
        # --- Enhance phi resonance and pattern coherence ---
        # Calculate boosts based on harmonics and phase optimizations
        
        # Phi resonance boost based on phi-ratio harmonics
        phi_harmonics = [h for h in resonance_harmonics if "φ" in h["harmonic_type"]]
        phi_boost = 0.0
        
        if phi_harmonics:
            # Calculate weighted average of phi harmonic qualities
            total_quality = sum(h["quality"] * h["strength"] for h in phi_harmonics)
            total_weight = sum(h["strength"] for h in phi_harmonics)
            
            if total_weight > FLOAT_EPSILON:
                avg_phi_quality = total_quality / total_weight
                phi_boost = avg_phi_quality * 0.07  # Up to 7% boost
        
        # Pattern coherence boost based on all harmonics and phase optimizations
        pattern_boost = 0.0
        
        if resonance_harmonics:
            # Calculate weighted average of all harmonic qualities
            total_quality = sum(h["quality"] * h["strength"] for h in resonance_harmonics)
            total_weight = sum(h["strength"] for h in resonance_harmonics)
            
            if total_weight > FLOAT_EPSILON:
                avg_harmonic_quality = total_quality / total_weight
                pattern_boost = avg_harmonic_quality * 0.05  # Up to 5% boost
        
        # Add boost from phase optimizations
        pattern_boost += coherence_boost * 0.5  # Scale from phase coherence improvement
        
        # Apply and record boosts
        phi_resonance = min(1.0, initial_phi_resonance + phi_boost)
        pattern_coherence = min(1.0, initial_pattern_coherence + pattern_boost)
        
        setattr(soul_spark, 'phi_resonance', float(phi_resonance))
        setattr(soul_spark, 'pattern_coherence', float(pattern_coherence))
        
        resonance_patterns["phi_resonance_boost"] = float(phi_boost)
        resonance_patterns["pattern_coherence_boost"] = float(pattern_boost)
        
        # --- Store resonance patterns in soul ---
        setattr(soul_spark, 'resonance_patterns', resonance_patterns)
        
        # --- Update soul state ---
        if hasattr(soul_spark, 'update_state'):
            logger.debug("Calling update_state after resonance pattern formation...")
            soul_spark.update_state()
            
            # Log coherence change
            coherence_change = soul_spark.coherence - initial_coherence
            logger.debug(f"Coherence after resonance patterns: {soul_spark.coherence:.1f} CU ({coherence_change:+.1f})")
        
        # --- Create Summary Metrics ---
        pattern_metrics = {
            'action': 'form_resonance_patterns',
            'soul_id': soul_spark.spark_id,
            'pattern_id': pattern_id,
            'channel_id': channel_id,
            'harmonic_count': len(resonance_harmonics),
            'phi_harmonic_count': len(phi_harmonics),
            'phase_optimization_count': len(phase_optimizations),
            'phi_resonance_initial': initial_phi_resonance,
            'phi_resonance_final': phi_resonance,
            'phi_resonance_boost': phi_boost,
            'pattern_coherence_initial': initial_pattern_coherence,
            'pattern_coherence_final': pattern_coherence,
            'pattern_coherence_boost': pattern_boost,
            'coherence_initial': initial_coherence,
            'coherence_final': soul_spark.coherence,
            'coherence_change': soul_spark.coherence - initial_coherence,
            'success': True,
            'timestamp': creation_time
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('resonance_patterns', pattern_metrics)
        
        # Generate sound for resonance patterns
        if SOUND_AVAILABLE and (len(resonance_harmonics) > 0 or len(phase_optimizations) > 0):
            try:
                # Initialize universe sounds generator
                universe_sounds = UniverseSounds()
                
                # Generate harmonic tones based on the resonance harmonics
                if len(resonance_harmonics) > 0:
                    # Get base frequency from strongest harmonic
                    base_freq = resonance_harmonics[0]["soul_freq"]
                    
                    # Create harmonic ratios and amplitudes
                    harmonics = []
                    amplitudes = []
                    
                    for h in resonance_harmonics[:5]:  # Use top 5 harmonics
                        ratio = h["channel_freq"] / base_freq
                        harmonics.append(ratio)
                        amplitudes.append(h["quality"] * 0.7)  # Scale by quality
                    
                    # Ensure we have at least the fundamental
                    if 1.0 not in harmonics:
                        harmonics.insert(0, 1.0)
                        amplitudes.insert(0, 0.8)
                    
                    # Create sound
                    sound_gen = SoundGenerator()
                    pattern_sound = sound_gen.generate_harmonic_tone(
                        base_freq,
                        harmonics,
                        amplitudes,
                        duration=4.0
                    )
                    
                    # Save sound
                    sound_file = f"resonance_pattern_{soul_spark.spark_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                    universe_sounds.save_sound(
                        pattern_sound, 
                        sound_file, 
                        f"Resonance pattern sound for soul {soul_spark.spark_id[:8]}"
                    )
                    
                    logger.info(f"Generated resonance pattern sound: {sound_file}")
                    pattern_metrics["sound_file"] = sound_file
            except Exception as sound_err:
                logger.warning(f"Failed to generate resonance pattern sound: {sound_err}")
        
        logger.info(f"Resonance patterns formed: {len(resonance_harmonics)} harmonics, PhiBoost: {phi_boost:.4f}, PatternBoost: {pattern_boost:.4f}")
        return pattern_metrics
        
    except Exception as e:
        logger.error(f"Error forming resonance patterns: {e}", exc_info=True)
        raise RuntimeError("Resonance pattern formation failed critically.") from e

def perform_creator_entanglement(soul_spark: SoulSpark,
                            kether_field: KetherField,
                            base_creator_potential: float = CREATOR_POTENTIAL_DEFAULT,
                            edge_of_chaos_target: float = EDGE_OF_CHAOS_DEFAULT) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Main orchestration function for Creator Entanglement.
    Uses wave-based resonance and quantum entanglement to form connection.
    Preserves and enhances coherence through resonant fields and layers.
    """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    if not (0.0 <= base_creator_potential <= 1.0): raise ValueError("base_creator_potential out of range.")
    if not (0.0 <= edge_of_chaos_target <= 1.0): raise ValueError("edge_of_chaos_target out of range.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Creator Entanglement for Soul {spark_id} ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}

    try:
        # Check prerequisites
        _check_prerequisites(soul_spark, kether_field)
        
        # Store initial state
        initial_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'pattern_coherence': getattr(soul_spark, 'pattern_coherence', 0.0),
            'creator_connection_strength': getattr(soul_spark, 'creator_connection_strength', 0.0)
        }
        
        logger.info(f"CE Initial State: S={initial_state['stability_su']:.1f}, "
                f"C={initial_state['coherence_cu']:.1f}, "
                f"Phi={initial_state['phi_resonance']:.3f}, "
                f"PatCoh={initial_state['pattern_coherence']:.3f}")
        
        # --- Form Resonant Field ---
        logger.info("CE Step 1: Establishing Resonant Field...")
        resonant_field, field_metrics = _establish_resonant_field(
            soul_spark, kether_field, base_creator_potential, edge_of_chaos_target
        )
        process_metrics_summary['steps']['resonant_field'] = field_metrics
        
        # --- Form Quantum Channel ---
        logger.info("CE Step 2: Forming Quantum Channel...")
        quantum_channel, channel_metrics = _form_quantum_channel(
            soul_spark, resonant_field
        )
        process_metrics_summary['steps']['quantum_channel'] = channel_metrics
        
        # --- Transfer Creator Aspects ---
        logger.info("CE Step 3: Transferring Creator Aspects...")
        transfer_metrics = _transfer_creator_aspects(
            soul_spark, quantum_channel, kether_field
        )
        process_metrics_summary['steps']['aspect_transfer'] = transfer_metrics
        
        # --- Form Resonance Patterns ---
        logger.info("CE Step 4: Forming Resonance Patterns...")
        pattern_metrics = _form_resonance_patterns(
            soul_spark, quantum_channel
        )
        process_metrics_summary['steps']['resonance_patterns'] = pattern_metrics
        
        # --- Set Final Flags ---
        setattr(soul_spark, FLAG_CREATOR_ENTANGLED, True)
        setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, True)
        
        # Final update to ensure state is consistent
        if hasattr(soul_spark, 'update_state'):
            logger.info("Performing final state update after Creator Entanglement...")
            soul_spark.update_state()
        
        # Add memory echo if available
        if hasattr(soul_spark, 'add_memory_echo'):
            connection_str = f"{transfer_metrics['aspects_transferred']} aspects gained, "
            connection_str += f"{transfer_metrics['aspects_strengthened']} strengthened"
            soul_spark.add_memory_echo(f"Creator entanglement complete. {connection_str}. "
                                    f"S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
        
        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)
        
        final_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'pattern_coherence': getattr(soul_spark, 'pattern_coherence', 0.0),
            'creator_connection_strength': getattr(soul_spark, 'creator_connection_strength', 0.0),
            FLAG_CREATOR_ENTANGLED: getattr(soul_spark, FLAG_CREATOR_ENTANGLED, False),
            FLAG_READY_FOR_HARMONIZATION: getattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, False)
        }
        
        overall_metrics = {
            'action': 'creator_entanglement',
            'soul_id': spark_id,
            'start_time': start_time_iso,
            'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'creator_potential': base_creator_potential,
            'edge_of_chaos_target': edge_of_chaos_target,
            'initial_state': initial_state,
            'final_state': final_state,
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'phi_resonance_change': final_state['phi_resonance'] - initial_state['phi_resonance'],
            'pattern_coherence_change': final_state['pattern_coherence'] - initial_state['pattern_coherence'],
            'aspects_transferred': transfer_metrics.get('aspects_transferred', 0),
            'aspects_strengthened': transfer_metrics.get('aspects_strengthened', 0),
            'success': True
        }
        soul_spark.ready_for_strengthening = True
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('creator_entanglement_summary', overall_metrics)
        
        logger.info(f"--- Creator Entanglement Completed Successfully for Soul {spark_id} ---")
        logger.info(f"  Final State: S={soul_spark.stability:.1f} ({overall_metrics['stability_change_su']:+.1f}), "
                f"C={soul_spark.coherence:.1f} ({overall_metrics['coherence_change_cu']:+.1f})")
        
        return soul_spark, overall_metrics
        
    except (ValueError, TypeError, AttributeError) as e_val:
        logger.error(f"Creator Entanglement failed for {spark_id}: {e_val}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'prerequisites/validation'
        _record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_val))
        raise e_val  # Hard fail
        
    except RuntimeError as e_rt:
        logger.critical(f"Creator Entanglement failed critically for {spark_id}: {e_rt}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'runtime'
        _record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_rt))
        # Ensure flags are not set on failure
        setattr(soul_spark, FLAG_CREATOR_ENTANGLED, False)
        setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, False)
        raise e_rt  # Hard fail
        
    except Exception as e:
        logger.critical(f"Unexpected error during Creator Entanglement for {spark_id}: {e}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'unexpected'
        _record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e))
        # Ensure flags are not set on failure
        setattr(soul_spark, FLAG_CREATOR_ENTANGLED, False)
        setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, False)
        setattr(soul_spark, FLAG_READY_FOR_STRENGTHENING, True)
        
        raise RuntimeError(f"Unexpected Creator Entanglement failure: {e}") from e  # Hard fail

def _record_entanglement_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """Helper to record failure metrics consistently."""
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('creator_entanglement_summary', {
                'action': 'creator_entanglement',
                'soul_id': spark_id,
                'start_time': start_time_iso,
                'end_time': end_time,
                'duration_seconds': duration,
                'success': False,
                'error': error_msg,
                'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record CE failure metrics for {spark_id}: {metric_e}")