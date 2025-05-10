"""
Life Cord Formation Functions (Refactored V4.4.0 - Wave Physics & Aura Integration)

Manifests the life cord based on Creator connection quality, anchored by the soul.
Uses wave-based physics including waveguide principles, quantum entanglement,
and standing wave patterns. Implements proper acoustic and light transmission
through the cord. Creates harmonic nodes based on Fibonacci sequences.
Integrates with soul's aura layers rather than modifying core properties.
Modifies the SoulSpark object instance directly. Uses constants. Hard fails.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
import uuid
from constants.constants import *
from typing import Dict, List, Any, Tuple, Optional
from math import pi as PI, sqrt, exp, sin, cos, tanh


# --- Logging ---
logger = logging.getLogger(__name__)

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    # Import resonance calculation
    from .creator_entanglement import calculate_resonance
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}.")
    raise ImportError(f"Core dependencies missing: {e}") from e

# --- Sound Module Imports ---
try:
    from sound.sound_generator import SoundGenerator
    from sound.sounds_of_universe import UniverseSounds
    SOUND_MODULES_AVAILABLE = True
except ImportError:
    logger.warning("Sound modules not available. Life cord formation will use simulated sound.")
    SOUND_MODULES_AVAILABLE = False

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

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds. Raises ValueError on failure. """
    logger.debug(f"Checking life cord prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False):
        msg = f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_LIFE_CORD}."
        logger.error(msg); raise ValueError(msg)

    # 2. Minimum Stability and Coherence (Absolute SU/CU)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if stability_su < 0 or coherence_cu < 0:
        msg = "Prerequisite failed: Soul missing stability or coherence attributes."
        logger.error(msg); raise AttributeError(msg)

    if stability_su < CORD_STABILITY_THRESHOLD_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {CORD_STABILITY_THRESHOLD_SU} SU."
        logger.error(msg); raise ValueError(msg)
    if coherence_cu < CORD_COHERENCE_THRESHOLD_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {CORD_COHERENCE_THRESHOLD_CU} CU."
        logger.error(msg); raise ValueError(msg)

    # 3. Energy Check (Done in main function before cost)

    if getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_CORD_FORMATION_COMPLETE}. Re-running.")

    logger.debug("Life cord prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. Raises error if missing/invalid. """
    logger.debug(f"Ensuring properties for life cord formation (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'position', 'field_radius',
                'field_strength', 'creator_connection_strength', 'energy', 'layers']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Life Cord: {missing}")

    if not hasattr(soul_spark, 'life_cord'): setattr(soul_spark, 'life_cord', {})
    if not hasattr(soul_spark, 'cord_integrity'): setattr(soul_spark, 'cord_integrity', 0.0)

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    pos = getattr(soul_spark, 'position')
    if not isinstance(pos, list) or len(pos)!=3: raise ValueError(f"Invalid position: {pos}")
    if soul_spark.energy < CORD_ACTIVATION_ENERGY_COST:
        raise ValueError(f"Insufficient energy ({soul_spark.energy:.1f} SEU) for cord activation cost ({CORD_ACTIVATION_ENERGY_COST} SEU).")

    # Check for layers presence
    if not soul_spark.layers or len(soul_spark.layers) < 2:
        raise ValueError("Soul must have at least 2 aura layers for life cord formation.")

    logger.debug("Soul properties ensured for Life Cord.")

def _find_resonant_layer_indices(soul_spark: SoulSpark, frequency: float) -> List[int]:
    """
    Find indices of layers that resonate with a given frequency.
    Uses calculate_resonance to find layers with natural affinity.
    """
    resonant_indices = []
    
    for i, layer in enumerate(soul_spark.layers):
        if not isinstance(layer, dict):
            continue
            
        # Get layer frequencies
        layer_freqs = []
        if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
            layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
        
        # Find best resonance with this layer
        best_resonance = 0.0
        for layer_freq in layer_freqs:
            res = calculate_resonance(frequency, layer_freq)
            best_resonance = max(best_resonance, res)
        
        # Add layer if resonance is significant
        if best_resonance > 0.3:  # Only include strong resonances
            resonant_indices.append(i)
    
    return resonant_indices

def _create_waveguide_structure(soul_frequency: float, length: float, diameter: float, 
                              integrity: float = 0.8) -> Dict[str, Any]:
    """
    Creates a waveguide structure for the life cord based on physical principles.
    Includes acoustic impedance, wave velocity, and attenuation factors.
    """
    if soul_frequency <= FLOAT_EPSILON or length <= FLOAT_EPSILON or diameter <= FLOAT_EPSILON:
        raise ValueError("Invalid waveguide parameters: frequency, length, or diameter is not positive.")
    
    if not 0.0 <= integrity <= 1.0:
        raise ValueError(f"Integrity must be between 0.0 and 1.0, got {integrity}")
    
    # Calculate basic waveguide properties
    wavelength = SPEED_OF_SOUND / soul_frequency
    cutoff_frequency = 1.8412 * SPEED_OF_SOUND / (PI * diameter)
    propagation_constant = 2 * PI / wavelength
    
    # Calculate attenuation based on integrity (lower integrity = higher attenuation)
    attenuation = (1.0 - integrity) * 0.05  # dB/m
    
    # Calculate acoustic impedance (simplified model)
    impedance = AIR_DENSITY * SPEED_OF_SOUND / (PI * (diameter/2)**2)
    
    # Calculate velocity factor (acoustic waves travel differently in waveguide)
    if soul_frequency > cutoff_frequency:
        velocity_factor = sqrt(1 - (cutoff_frequency/soul_frequency)**2)
    else:
        velocity_factor = 0.5  # Below cutoff, waves still propagate but differently
    
    # Create waveguide structure
    waveguide = {
        "length": float(length),
        "diameter": float(diameter),
        "wavelength": float(wavelength),
        "cutoff_frequency": float(cutoff_frequency),
        "propagation_constant": float(propagation_constant),
        "attenuation": float(attenuation),
        "acoustic_impedance": float(impedance),
        "velocity_factor": float(velocity_factor),
        "resonant_modes": [],# Will be filled with modes
        "primary_frequency_hz": float(soul_frequency),  # Add this line
        "bandwidth_hz": float(soul_frequency * 0.1),  # Add this line
    }
    
    # Calculate resonant modes in the waveguide
    num_modes = 5  # Calculate first 5 modes
    for m in range(num_modes):
        mode_freq = (m + 1) * SPEED_OF_SOUND / (2 * length)
        mode_wavelength = SPEED_OF_SOUND / mode_freq
        
        # Check if mode can propagate
        can_propagate = mode_freq > cutoff_frequency
        
        # Calculate mode integrity (higher modes have lower integrity)
        mode_integrity = integrity * (1.0 - 0.1 * m)
        
        waveguide["resonant_modes"].append({
            "mode_number": m + 1,
            "frequency": float(mode_freq),
            "wavelength": float(mode_wavelength),
            "can_propagate": can_propagate,
            "integrity_factor": float(mode_integrity)
        })
    
    return waveguide

def _calculate_fibonacci_node_positions(num_nodes: int) -> List[float]:
    """
    Calculate node positions based on Fibonacci sequence for optimal stability.
    Returns positions normalized to 0-1 range.
    """
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive.")
    
    # Generate enough Fibonacci numbers
    fib_sequence = [1, 1]
    while len(fib_sequence) < num_nodes + 5:  # Get more than needed for calculations
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    
    # Use Fibonacci numbers to calculate positions
    positions = []
    max_pos = fib_sequence[num_nodes + 2]  # Use this for normalization
    
    for i in range(num_nodes):
        # Use golden ratio properties for natural distribution
        # This creates a distribution that clumps toward the ends
        pos = fib_sequence[i + 2] / max_pos
        
        # Adjust to ensure positions are well-distributed
        # Apply a smoothing function to get better spread
        adjusted_pos = 0.05 + 0.9 * (1 - (1 - pos)**2)
        
        positions.append(adjusted_pos)
    
    # Sort positions (Fibonacci sequence will already be ascending,
    # but the smoothing might change the order slightly)
    positions.sort()
    
    return positions

# --- Core Life Cord Functions ---

def _establish_soul_earth_anchors(soul_spark: SoulSpark) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Establishes connection anchors between soul and Earth using wave physics.
    Creates structures that resonate rather than directly transfer energy.
    """
    logger.info("Establishing soul-Earth connection anchors...")
    
    # Get soul properties for calculations
    soul_stability = soul_spark.stability / MAX_STABILITY_SU  # Normalize to 0-1
    soul_coherence = soul_spark.coherence / MAX_COHERENCE_CU  # Normalize to 0-1
    soul_frequency = soul_spark.frequency
    creator_connection = getattr(soul_spark, 'creator_connection_strength', 0.0)
    
    # Calculate anchor strengths based on soul properties
    soul_anchor_strength = (soul_stability * 0.6 + soul_coherence * 0.4) * ANCHOR_STRENGTH_MODIFIER
    soul_anchor_resonance = (soul_coherence * 0.7 + soul_stability * 0.3) * ANCHOR_RESONANCE_MODIFIER
    
    # Find resonant layers for soul anchor
    soul_anchor_layer_indices = _find_resonant_layer_indices(soul_spark, soul_frequency)
    
    # If no resonant layers found, find the most suitable one
    if not soul_anchor_layer_indices:
        # Use outermost layer (typically the connection interface)
        soul_anchor_layer_indices = [len(soul_spark.layers) - 1]
    
    # Create soul anchor
    soul_anchor = {
        "position": [float(p) for p in soul_spark.position],
        "frequency": float(soul_frequency),
        "strength": float(max(0.1, min(1.0, soul_anchor_strength))),
        "resonance": float(max(0.1, min(1.0, soul_anchor_resonance))),
        "layer_indices": soul_anchor_layer_indices,
        "creator_connection": float(creator_connection)
    }
    
    # Calculate Earth anchor properties
    earth_frequency = EARTH_FREQUENCY
    earth_anchor_strength = EARTH_ANCHOR_STRENGTH
    earth_anchor_resonance = EARTH_ANCHOR_RESONANCE
    
    # Create Earth anchor position (below soul)
    earth_anchor_pos = [float(p) for p in soul_anchor["position"]]
    earth_anchor_pos[2] -= 100.0  # Simplified - place below soul
    
    # Create Earth anchor
    earth_anchor = {
        "position": earth_anchor_pos,
        "frequency": float(earth_frequency),
        "strength": float(earth_anchor_strength),
        "resonance": float(earth_anchor_resonance)
    }
    
    # Calculate resonance between soul and Earth
    resonance = calculate_resonance(soul_frequency, earth_frequency)
    
    # Enhance resonance with creator connection (acts as bridge)
    if creator_connection > FLOAT_EPSILON:
        resonance = resonance * (1.0 + creator_connection * 0.5)
    
    # Normalize to 0-1 range
    connection_strength = min(1.0, max(0.2, resonance))
    
    logger.info(f"Anchor points established. Soul: {soul_frequency:.1f}Hz (S={soul_anchor_strength:.3f}), "
               f"Earth: {earth_frequency:.1f}Hz (S={earth_anchor_strength:.3f}), "
               f"Resonance: {connection_strength:.3f}")
    
    return soul_anchor, earth_anchor, connection_strength

def _form_bidirectional_waveguide(soul_anchor: Dict[str, Any], earth_anchor: Dict[str, Any],
                               connection_strength: float, complexity: float) -> Dict[str, Any]:
    """
    Forms a bidirectional waveguide between soul and Earth using wave physics.
    Creates a structure that can propagate both energy and information.
    """
    logger.info("Forming bidirectional waveguide...")
    
    # Calculate waveguide parameters
    soul_freq = soul_anchor["frequency"]
    earth_freq = earth_anchor["frequency"]
    
    # Calculate distance between anchors (simplified for example)
    soul_pos = np.array(soul_anchor["position"])
    earth_pos = np.array(earth_anchor["position"])
    distance = np.linalg.norm(soul_pos - earth_pos)
    
    # Scale waveguide diameter based on connection strength and complexity
    base_diameter = 10.0  # Base diameter in arbitrary units
    diameter = base_diameter * (0.5 + 0.5 * connection_strength) * (0.7 + 0.3 * complexity)
    
    # Create main waveguide for soul frequency
    soul_waveguide = _create_waveguide_structure(
        soul_frequency=soul_freq,
        length=distance,
        diameter=diameter,
        integrity=connection_strength
    )
    
    # Create secondary waveguide for Earth frequency
    earth_waveguide = _create_waveguide_structure(
        soul_frequency=earth_freq,
        length=distance,
        diameter=diameter * 0.8,  # Slightly smaller
        integrity=connection_strength * 0.9  # Slightly weaker
    )
    
    # Calculate bidirectional properties
    # - How well energy flows from soul to Earth
    soul_to_earth_efficiency = connection_strength * (soul_anchor["strength"] / 2 + earth_anchor["resonance"] / 2)
    
    # - How well energy flows from Earth to soul
    earth_to_soul_efficiency = connection_strength * (earth_anchor["strength"] / 2 + soul_anchor["resonance"] / 2)
    
    # - How well information flows bidirectionally (quantum entanglement factor)
    quantum_efficiency = connection_strength * min(soul_anchor["resonance"], earth_anchor["resonance"]) * 1.2
    quantum_efficiency = min(1.0, quantum_efficiency)  # Cap at 1.0
    
    # Calculate waveguide impedance matching
    # - Better matching = more efficient energy transfer
    impedance_ratio = soul_waveguide["acoustic_impedance"] / earth_waveguide["acoustic_impedance"]
    impedance_match = 4 * impedance_ratio / ((1 + impedance_ratio)**2)  # 1.0 = perfect match
    
    # Create waveguide structure
    waveguide = {
        "soul_waveguide": soul_waveguide,
        "earth_waveguide": earth_waveguide,
        "length": float(distance),
        "diameter": float(diameter),
        "soul_to_earth_efficiency": float(soul_to_earth_efficiency),
        "earth_to_soul_efficiency": float(earth_to_soul_efficiency),
        "quantum_efficiency": float(quantum_efficiency),
        "impedance_match": float(impedance_match),
        "bidirectional_factor": float((soul_to_earth_efficiency + earth_to_soul_efficiency) / 2)
    }
    
    logger.info(f"Bidirectional waveguide formed. Length={distance:.1f}, Diameter={diameter:.1f}, "
               f"S→E={soul_to_earth_efficiency:.3f}, E→S={earth_to_soul_efficiency:.3f}, "
               f"Quantum={quantum_efficiency:.3f}")
    
    return waveguide

def _create_harmonic_nodes(waveguide: Dict[str, Any], complexity: float) -> List[Dict[str, Any]]:
    """
    Creates harmonic nodes in the life cord based on Fibonacci patterns.
    These nodes act as resonators and energy amplifiers/attenuators.
    """
    logger.info("Creating harmonic nodes...")
    
    # Calculate number of nodes based on complexity
    num_nodes = int(HARMONIC_NODE_COUNT_BASE + complexity * HARMONIC_NODE_COUNT_FACTOR)
    
    # Calculate node positions using Fibonacci sequence
    node_positions = _calculate_fibonacci_node_positions(num_nodes)
    
    # Get waveguide properties for node calculations
    length = waveguide["length"]
    soul_waveguide = waveguide["soul_waveguide"]
    earth_waveguide = waveguide["earth_waveguide"]
    
    # Create nodes with harmonic properties
    nodes = []
    
    for i, position in enumerate(node_positions):
        # Calculate node properties based on position and Fibonacci sequence
        # Position affects the node's resonant properties
        
        # Determine node type (different types of nodes have different properties)
        node_type = ""
        type_idx = i % 3
        if type_idx == 0:
            # Phi-based harmonic node
            phi_exp = (i//3) % 5 + 1
            freq_ratio = PHI**phi_exp
            node_type = f"phi^{phi_exp}"
        elif type_idx == 1:
            # Integer-ratio harmonic node
            int_mult = (i//3) % 7 + 2
            freq_ratio = float(int_mult)
            node_type = f"int*{int_mult}"
        else:
            # Silver ratio harmonic node
            silver_exp = (i//3) % 3 + 1
            freq_ratio = SILVER_RATIO**silver_exp
            node_type = f"silver^{silver_exp}"
        
        # Calculate frequencies from the waveguides
        soul_freq = soul_waveguide["resonant_modes"][0]["frequency"] * freq_ratio
        earth_freq = earth_waveguide["resonant_modes"][0]["frequency"] * freq_ratio
        
        # Calculate amplitude based on position and complexity
        base_amplitude = HARMONIC_NODE_AMP_BASE + complexity * HARMONIC_NODE_AMP_FACTOR_COMPLEX
        position_factor = 1.0 - abs(position - 0.5) * HARMONIC_NODE_AMP_FALLOFF
        amplitude = base_amplitude * position_factor
        
        # Calculate phase offset (phi-based for natural oscillation)
        phase_offset = PI * PHI * position
        
        # Create node
        node = {
            "position": float(position),
            "absolute_position": float(position * length),
            "soul_frequency": float(soul_freq),
            "earth_frequency": float(earth_freq),
            "amplitude": float(max(0.0, min(1.0, amplitude))),
            "phase_offset": float(phase_offset),
            "type": node_type,
            "standing_wave_antinode": i % 2 == 0  # Alternating antinodes and nodes
        }
        
        nodes.append(node)
    
    logger.info(f"Created {len(nodes)} harmonic nodes using Fibonacci positioning.")
    return nodes

def _integrate_with_aura_layers(soul_spark: SoulSpark, waveguide: Dict[str, Any],
                             harmonic_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Integrates the life cord with soul's aura layers for enhanced coherence.
    Creates resonant connections between cord nodes and aura layers.
    """
    logger.info("Integrating life cord with aura layers...")
    
    # Get soul properties
    soul_frequency = soul_spark.frequency
    soul_layers = soul_spark.layers
    
    # Track layer integrations
    layer_integrations = []
    total_resonance = 0.0
    
    # Track which nodes connect to which layers
    node_layer_connections = {}
    
    # Integrate each harmonic node with resonant aura layers
    for i, node in enumerate(harmonic_nodes):
        node_freq = node["soul_frequency"]
        node_position = node["position"]
        
        # Find resonant layers for this node
        resonant_layer_indices = _find_resonant_layer_indices(soul_spark, node_freq)
        
        # If no resonant layers found, find closest layer by frequency
        if not resonant_layer_indices and soul_layers:
            closest_layer_idx = 0
            closest_resonance = 0.0
            
            for j, layer in enumerate(soul_layers):
                if not isinstance(layer, dict):
                    continue
                    
                # Get layer frequencies
                layer_freqs = []
                if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
                    layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
                
                if not layer_freqs:
                    continue
                
                # Find best resonance
                best_res = 0.0
                for layer_freq in layer_freqs:
                    res = calculate_resonance(node_freq, layer_freq)
                    best_res = max(best_res, res)
                
                if best_res > closest_resonance:
                    closest_resonance = best_res
                    closest_layer_idx = j
            
            # Add closest layer if resonance is at least minimal
            if closest_resonance > 0.1:
                resonant_layer_indices = [closest_layer_idx]
        
        # Create connections for this node
        node_connections = []
        
        for layer_idx in resonant_layer_indices:
            # Calculate connection strength based on node and layer properties
            layer = soul_layers[layer_idx]
            
            # Get layer properties
            layer_strength = layer.get('strength', 0.5) if isinstance(layer, dict) else 0.5
            
            # Calculate connection parameters
            connection_strength = layer_strength * node["amplitude"] * 0.8
            
            # Add cord resonance to layer
            if isinstance(layer, dict):
                # Add life cord resonance to this layer
                if 'life_cord_resonance' not in layer:
                    layer['life_cord_resonance'] = {}
                
                # Add node connection
                node_key = f"node_{i}"
                layer['life_cord_resonance'][node_key] = {
                    "frequency": float(node_freq),
                    "strength": float(connection_strength),
                    "position": float(node_position),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add cord frequencies to layer's resonant frequencies
                if 'resonant_frequencies' not in layer:
                    layer['resonant_frequencies'] = []
                
                if node_freq not in layer['resonant_frequencies']:
                    layer['resonant_frequencies'].append(float(node_freq))
            
            # Add connection to tracking
            node_connections.append({
                "layer_idx": layer_idx,
                "strength": float(connection_strength)
            })
            
            # Add to total resonance
            total_resonance += connection_strength
            
            # Add to layer integrations
            layer_integrations.append({
                "node_idx": i,
                "layer_idx": layer_idx,
                "frequency": float(node_freq),
                "strength": float(connection_strength)
            })
        
        # Store connections for this node
        node_layer_connections[i] = node_connections
    
    # Calculate overall integration factor
    integration_factor = 0.0
    if layer_integrations:
        avg_resonance = total_resonance / len(layer_integrations)
        integration_factor = avg_resonance * min(1.0, len(layer_integrations) / 10.0)
    
    # Create integration metrics
    integration_metrics = {
        "layer_integrations": layer_integrations,
        "total_resonance": float(total_resonance),
        "integration_factor": float(integration_factor),
        "integrated_layers": len(set(li["layer_idx"] for li in layer_integrations)),
        "node_layer_map": node_layer_connections
    }
    
    logger.info(f"Integrated life cord with {integration_metrics['integrated_layers']} aura layers. "
               f"Integration factor: {integration_factor:.3f}")
    
    return integration_metrics

def _create_light_pathways(waveguide: Dict[str, Any], harmonic_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates light pathways through the cord waveguide, connecting harmonic nodes
    with bidirectional energy propagation channels. Uses quantum entanglement
    properties for non-local information exchange.
    
    Args:
        waveguide: The waveguide structure with physical properties
        harmonic_nodes: List of harmonic nodes placed along the cord
        
    Returns:
        Dict containing light pathway metrics and properties
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If pathway creation fails
    """
    logger.info("LC Step: Creating Light Pathways...")
    
    if not isinstance(waveguide, dict):
        raise ValueError("Waveguide must be a dictionary")
    if not isinstance(harmonic_nodes, list) or not harmonic_nodes:
        raise ValueError("Harmonic nodes must be a non-empty list")
    
    try:
        # Print all waveguide keys for debugging
        logger.debug(f"Waveguide keys in _create_light_pathways: {list(waveguide.keys())}")
        # Extract key waveguide properties
        primary_freq = waveguide.get("primary_frequency_hz", 0.0)
        bandwidth_hz = waveguide.get("bandwidth_hz", 0.0)
        stability_factor = waveguide.get("stability_factor", 0.0)
        elasticity = waveguide.get("elasticity_factor", 0.0)
        
        logger.debug(f"Extracted frequencies: primary={primary_freq}, bandwidth={bandwidth_hz}")
        
        if primary_freq <= FLOAT_EPSILON or bandwidth_hz <= FLOAT_EPSILON:
            # Provide more detailed error
            all_keys = ", ".join(waveguide.keys())
            raise ValueError(f"Invalid waveguide frequencies: primary={primary_freq}, bandwidth={bandwidth_hz}. Available keys: {all_keys}")
        
        # Sort nodes by position for coherent pathway formation
        sorted_nodes = sorted(harmonic_nodes, key=lambda n: n.get("position", 0.0))
        
        # Create light mapping structures
        light_frequencies = []
        light_pathways = []
        
        # Determine light spectrum scaling factor
        # Light frequency = audio frequency * scaling factor
        light_scaling_factor = 1e12  # Standard audio-to-light scaling
        
        # Base metrics
        total_pathway_strength = 0.0
        max_entanglement = 0.0
        quantum_tunneling_factor = 0.0
        
        # Process each node pair to create pathways
        for i in range(len(sorted_nodes) - 1):
            node1 = sorted_nodes[i]
            node2 = sorted_nodes[i + 1]
            
            # Calculate positions and frequencies
            pos1 = node1.get("position", 0.0)
            pos2 = node2.get("position", 0.0)
            freq1 = node1.get("frequency_hz", primary_freq)
            freq2 = node2.get("frequency_hz", primary_freq)
            
            # Skip invalid node pairs
            if pos1 >= pos2 or freq1 <= FLOAT_EPSILON or freq2 <= FLOAT_EPSILON:
                continue
                
            # Calculate wavelengths
            wavelength1 = SPEED_OF_LIGHT / (freq1 * light_scaling_factor)
            wavelength2 = SPEED_OF_LIGHT / (freq2 * light_scaling_factor)
            
            # Determine optimal pathway properties using quantum optics principles
            # Path length in normalized units (0-1 scale)
            path_length = pos2 - pos1
            
            # Interference quality - based on wavelength relationship
            # Perfect constructive interference at integer wavelength ratios
            wavelength_ratio = max(wavelength1, wavelength2) / min(wavelength1, wavelength2)
            interference_quality = 0.0
            
            # Check for resonant wavelength ratios (integers and Phi-based)
            for n in range(1, 6):
                for d in range(1, 6):
                    if d == 0:
                        continue
                    ratio = n / d
                    if abs(wavelength_ratio - ratio) < 0.05:
                        interference_quality = max(interference_quality, 
                                                 1.0 - abs(wavelength_ratio - ratio) * 10.0)
            
            # Check PHI-based ratios
            phi_ratios = [PHI, 1/PHI, PHI**2, 1/(PHI**2)]
            for phi_ratio in phi_ratios:
                if abs(wavelength_ratio - phi_ratio) < 0.05:
                    interference_quality = max(interference_quality, 
                                             1.0 - abs(wavelength_ratio - phi_ratio) * 10.0)
            
            # If no resonant ratio found, use a base minimum
            if interference_quality < 0.2:
                interference_quality = 0.2
                
            # Quantum entanglement strength - decreases with physical distance
            # but has non-local properties based on frequency resonance
            direct_entanglement = max(0.0, min(1.0, 
                                            0.8 - 0.6 * path_length + 0.4 * interference_quality))
            
            # Non-local quantum correlation based on frequency
            freq_resonance = calculate_resonance(freq1, freq2)
            nonlocal_factor = 0.3 + 0.7 * freq_resonance
            
            # Combined entanglement quality
            entanglement_quality = min(1.0, direct_entanglement * 0.6 + nonlocal_factor * 0.4)
            max_entanglement = max(max_entanglement, entanglement_quality)
            
            # Quantum tunneling probability - important for information transfer
            # through potential barriers in the waveguide
            barrier_height = 1.0 - min(stability_factor, elasticity)
            tunneling_factor = min(1.0, 
                                 0.3 * entanglement_quality + 
                                 0.2 * (1.0 - path_length) + 
                                 0.5 * (1.0 - barrier_height))
            
            quantum_tunneling_factor = max(quantum_tunneling_factor, tunneling_factor)
            
            # Calculate overall pathway strength
            pathway_strength = min(1.0, 
                                 0.4 * entanglement_quality + 
                                 0.4 * interference_quality + 
                                 0.2 * tunneling_factor)
            
            total_pathway_strength += pathway_strength
            
            # Map audio frequencies to light spectrum
            light_freq1 = freq1 * light_scaling_factor
            light_freq2 = freq2 * light_scaling_factor
            
            # Convert to wavelengths in nm
            light_wavelength1 = (SPEED_OF_LIGHT / light_freq1) * 1e9
            light_wavelength2 = (SPEED_OF_LIGHT / light_freq2) * 1e9
            
            # Get colors from wavelengths
            color1 = _wavelength_to_color(light_wavelength1)
            color2 = _wavelength_to_color(light_wavelength2)
            
            # Create light frequency mapping
            light_frequencies.extend([
                {
                    "audio_freq": float(freq1),
                    "light_freq": float(light_freq1),
                    "wavelength_nm": float(light_wavelength1),
                    "color": color1
                },
                {
                    "audio_freq": float(freq2),
                    "light_freq": float(light_freq2),
                    "wavelength_nm": float(light_wavelength2),
                    "color": color2
                }
            ])
            
            # Create pathway structure
            light_pathways.append({
                "start_node_idx": i,
                "end_node_idx": i + 1,
                "start_position": float(pos1),
                "end_position": float(pos2),
                "start_frequency": float(freq1),
                "end_frequency": float(freq2),
                "interference_quality": float(interference_quality),
                "entanglement_quality": float(entanglement_quality),
                "tunneling_factor": float(tunneling_factor),
                "strength": float(pathway_strength),
                "light_properties": {
                    "start_wavelength_nm": float(light_wavelength1),
                    "end_wavelength_nm": float(light_wavelength2),
                    "start_color": color1,
                    "end_color": color2
                }
            })
        
        # Calculate overall light pathway metrics
        avg_pathway_strength = (total_pathway_strength / len(light_pathways) 
                             if light_pathways else 0.0)
        
        # Create final light pathways structure
        light_pathways_structure = {
            "pathways": light_pathways,
            "light_frequencies": light_frequencies,
            "avg_pathway_strength": float(avg_pathway_strength),
            "max_entanglement_quality": float(max_entanglement),
            "quantum_tunneling_factor": float(quantum_tunneling_factor),
            "total_pathway_count": len(light_pathways)
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "pathway_count": len(light_pathways),
                "avg_pathway_strength": avg_pathway_strength,
                "max_entanglement": max_entanglement,
                "quantum_tunneling_factor": quantum_tunneling_factor,
                "light_frequencies_mapped": len(light_frequencies),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('life_cord_light_pathways', metrics_data)
        
        logger.info(f"Light pathways created: {len(light_pathways)} paths, Avg Strength: {avg_pathway_strength:.4f}")
        return light_pathways_structure
        
    except Exception as e:
        logger.error(f"Error creating light pathways: {e}", exc_info=True)
        raise RuntimeError("Light pathway creation failed critically.") from e

def _calculate_information_bandwidth(waveguide: Dict[str, Any], 
                                   light_pathways: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates theoretical and effective information bandwidth based on 
    waveguide properties and light pathway characteristics.
    
    Args:
        waveguide: The waveguide structure
        light_pathways: The light pathway structure
        
    Returns:
        Dict containing bandwidth calculations
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If calculation fails
    """
    logger.info("LC Step: Calculating Information Bandwidth...")
    
    if not isinstance(waveguide, dict):
        raise ValueError("Waveguide must be a dictionary")
    if not isinstance(light_pathways, dict):
        raise ValueError("Light pathways must be a dictionary")
    
    try:
        # Extract key properties
        base_bandwidth_hz = waveguide.get("bandwidth_hz", 0.0)
        stability_factor = waveguide.get("stability_factor", 0.0)
        avg_pathway_strength = light_pathways.get("avg_pathway_strength", 0.0)
        max_entanglement = light_pathways.get("max_entanglement_quality", 0.0)
        quantum_tunneling = light_pathways.get("quantum_tunneling_factor", 0.0)
        pathway_count = light_pathways.get("total_pathway_count", 0)
        
        if base_bandwidth_hz <= FLOAT_EPSILON:
            raise ValueError(f"Invalid base bandwidth: {base_bandwidth_hz}")
        
        # Calculate theoretical maximum bandwidth
        # Base audio bandwidth + quantum multiplier effect
        quantum_multiplier = 1.0 + max_entanglement * 3.0
        theoretical_max_bw = base_bandwidth_hz * quantum_multiplier
        
        # Calculate effective bandwidth considering pathway quality
        pathway_quality_factor = min(1.0, 
                                  0.4 * avg_pathway_strength + 
                                  0.3 * max_entanglement + 
                                  0.3 * quantum_tunneling)
        
        # Parallel pathway factor - more pathways enable more information flow
        parallel_factor = min(1.0, pathway_count / 10.0) * 0.5 + 0.5
        
        # Stability affects consistent information flow
        stability_effect = 0.3 + 0.7 * stability_factor
        
        # Calculate effective bandwidth
        effective_bandwidth = (theoretical_max_bw * 
                             pathway_quality_factor * 
                             parallel_factor * 
                             stability_effect)
        
        # Calculate information capacity in theoretical bits per second
        # Using Shannon's information theory as a basis
        # C = B * log2(1 + S/N)
        # Where B is bandwidth, S is signal power, N is noise power
        signal_to_noise = 1.0 + 9.0 * pathway_quality_factor * stability_factor
        bits_per_second = effective_bandwidth * np.log2(signal_to_noise)
        
        # Calculate quantum information capacity
        # Entanglement enables quantum information transfer (qubits)
        qubit_efficiency = max_entanglement * quantum_tunneling
        qubits_per_second = bits_per_second * qubit_efficiency
        
        # Create bandwidth structure
        bandwidth_metrics = {
            "base_bandwidth_hz": float(base_bandwidth_hz),
            "theoretical_max_bandwidth_hz": float(theoretical_max_bw),
            "effective_bandwidth_hz": float(effective_bandwidth),
            "pathway_quality_factor": float(pathway_quality_factor),
            "parallel_pathway_factor": float(parallel_factor),
            "stability_effect": float(stability_effect),
            "quantum_multiplier": float(quantum_multiplier),
            "information_capacity_bps": float(bits_per_second),
            "quantum_information_capacity_qps": float(qubits_per_second)
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "base_bandwidth_hz": float(base_bandwidth_hz),
                "theoretical_max_bandwidth_hz": float(theoretical_max_bw),
                "effective_bandwidth_hz": float(effective_bandwidth),
                "information_capacity_bps": float(bits_per_second),
                "quantum_information_capacity_qps": float(qubits_per_second),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('life_cord_bandwidth', metrics_data)
        
        logger.info(f"Information bandwidth calculated: Base={base_bandwidth_hz:.1f}Hz, "
                   f"Effective={effective_bandwidth:.1f}Hz, "
                   f"Capacity={bits_per_second:.1f}bps")
        return bandwidth_metrics
        
    except Exception as e:
        logger.error(f"Error calculating information bandwidth: {e}", exc_info=True)
        raise RuntimeError("Information bandwidth calculation failed.") from e

def _create_standing_wave_nodes(cord_structure: Dict[str, Any], 
                              earth_connection: float) -> Dict[str, Any]:
    """
    Creates standing wave patterns between soul and Earth anchor points,
    enhancing cord stability through resonant nodes positioned according
    to Fibonacci patterns.
    
    Args:
        cord_structure: The developing cord structure
        earth_connection: Earth connection strength factor (0-1)
        
    Returns:
        Dict containing standing wave metrics
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If wave creation fails
    """
    logger.info("LC Step: Creating Standing Wave Nodes...")
    
    if not isinstance(cord_structure, dict):
        raise ValueError("Cord structure must be a dictionary")
    if not isinstance(earth_connection, (int, float)) or not (0.0 <= earth_connection <= 1.0):
        raise ValueError(f"Earth connection must be between 0.0 and 1.0, got {earth_connection}")
    
    try:
        # Extract key properties
        primary_freq = cord_structure.get("primary_frequency_hz", 0.0)
        soul_primary_freq = cord_structure.get("soul_primary_freq", 0.0)
        earth_freq = cord_structure.get("earth_freq", 0.0)
        stability_factor = cord_structure.get("stability_factor", 0.0)
        
        if primary_freq <= FLOAT_EPSILON:
            raise ValueError(f"Invalid primary frequency: {primary_freq}")
        
        # Calculate fundamental wavelength for standing wave
        # Using primary frequency as the basis
        fundamental_wavelength = SPEED_OF_SOUND / primary_freq
        
        # Use Fibonacci sequence for node positioning
        # This creates a naturalistic, harmonious pattern
        fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34]
        fib_total = sum(fib_seq)
        
        # Number of nodes based on earth connection and stability
        node_count = int(3 + earth_connection * 7)
        node_count = min(node_count, len(fib_seq))
        
        standing_wave_nodes = []
        total_amplitude = 0.0
        
        # Create nodes with Fibonacci-based positioning
        for i in range(node_count):
            # Position follows Fibonacci spacing
            fib_position = sum(fib_seq[:i+1]) / fib_total
            
            # Calculate node properties
            # Amplitude is maximum at center, decreases toward ends
            amplitude = (0.5 + 0.5 * np.sin(PI * fib_position)) * earth_connection
            
            # Phase alternates between peaks and troughs
            phase = PI * i
            
            # Frequency gradually shifts from soul to earth frequency
            node_freq = (soul_primary_freq * (1.0 - fib_position) + 
                       earth_freq * fib_position)
            
            # Calculate wavelength at this node
            node_wavelength = SPEED_OF_SOUND / node_freq
            
            # Determine node type (antinode or node)
            node_type = "antinode" if i % 2 == 0 else "node"
            
            # Add to nodes list
            standing_wave_nodes.append({
                "position": float(fib_position),
                "amplitude": float(amplitude),
                "phase": float(phase),
                "frequency_hz": float(node_freq),
                "wavelength": float(node_wavelength),
                "type": node_type
            })
            
            total_amplitude += amplitude
        
        # Calculate standing wave metrics
        avg_amplitude = total_amplitude / node_count if node_count > 0 else 0.0
        
        # Calculate coherence of the standing wave
        # Perfect standing waves have consistent nodes and antinodes
        amplitude_std = np.std([node["amplitude"] for node in standing_wave_nodes]) if standing_wave_nodes else 1.0
        coherence_factor = max(0.0, 1.0 - amplitude_std)
        
        # Calculate stability enhancement from standing wave
        stability_boost = (earth_connection * 0.5 + 
                         coherence_factor * 0.3 + 
                         avg_amplitude * 0.2) * 0.2
        
        # Create standing wave structure
        standing_wave_structure = {
            "nodes": standing_wave_nodes,
            "fundamental_wavelength": float(fundamental_wavelength),
            "node_count": node_count,
            "average_amplitude": float(avg_amplitude),
            "coherence_factor": float(coherence_factor),
            "stability_boost": float(stability_boost)
        }
        
        # Update cord structure with standing wave
        cord_structure["standing_wave"] = standing_wave_structure
        
        # Boost stability with standing wave effect
        new_stability = min(1.0, stability_factor + stability_boost)
        cord_structure["stability_factor"] = float(new_stability)
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "node_count": node_count,
                "average_amplitude": float(avg_amplitude),
                "coherence_factor": float(coherence_factor),
                "stability_boost": float(stability_boost),
                "new_stability_factor": float(new_stability),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('life_cord_standing_wave', metrics_data)
        
        logger.info(f"Standing wave nodes created: {node_count} nodes, "
                   f"Coherence: {coherence_factor:.4f}, "
                   f"Stability Boost: +{stability_boost:.4f}")
        return standing_wave_structure
        
    except Exception as e:
        logger.error(f"Error creating standing wave nodes: {e}", exc_info=True)
        raise RuntimeError("Standing wave creation failed critically.") from e

def _enhance_cord_with_sound(cord_structure: Dict[str, Any],
                           sound_type: str = "harmonic") -> Dict[str, Any]:
    """
    Enhances cord properties using sound harmonics that strengthen 
    the waveguide characteristics and integrity.
    
    Args:
        cord_structure: The developing cord structure
        sound_type: Type of sound enhancement (harmonic, quantum, resonant)
        
    Returns:
        Dict containing sound enhancement metrics
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If sound enhancement fails
    """
    logger.info(f"LC Step: Enhancing Cord with {sound_type.capitalize()} Sound...")
    
    if not isinstance(cord_structure, dict):
        raise ValueError("Cord structure must be a dictionary")
    if not isinstance(sound_type, str) or not sound_type:
        raise ValueError("Sound type must be a non-empty string")
    
    try:
        # Extract key properties
        primary_freq = cord_structure.get("primary_frequency_hz", 0.0)
        bandwidth_hz = cord_structure.get("bandwidth_hz", 0.0)
        stability_factor = cord_structure.get("stability_factor", 0.0)
        elasticity = cord_structure.get("elasticity_factor", 0.0)
        
        if primary_freq <= FLOAT_EPSILON:
            raise ValueError(f"Invalid primary frequency: {primary_freq}")
        
        # Generate sound based on type
        sound_duration = 5.0  # seconds
        sound_amplitude = 0.7  # normalized amplitude
        
        sound_data = None
        sound_file = None
        
        # Check if sound modules are available
        if not 'SoundGenerator' in globals() and not SOUND_MODULES_AVAILABLE:
            logger.warning("Sound modules unavailable. Creating theoretical sound model only.")
            sound_available = False
        else:
            sound_available = SOUND_MODULES_AVAILABLE
        
        # Initialize sound metrics
        sound_enhancement = {
            "type": sound_type,
            "frequency_hz": float(primary_freq),
            "duration_seconds": float(sound_duration),
            "amplitude": float(sound_amplitude),
            "has_physical_sound": False,
            "enhancement_factors": {}
        }
        
        # Create physical sound if available
        if sound_available:
            try:
                # Create sound generator
                sound_gen = SoundGenerator(sample_rate=SAMPLE_RATE)
                universe_sounds = UniverseSounds(sample_rate=SAMPLE_RATE)
                
                if sound_type == "harmonic":
                    # Create harmonic chord based on cord frequency
                    harmonics = [1.0, PHI, 2.0, PHI*2, 3.0]
                    amplitudes = [0.8, 0.6, 0.5, 0.4, 0.3]
                    sound_data = sound_gen.generate_harmonic_tone(
                        primary_freq, harmonics, amplitudes, sound_duration)
                    
                elif sound_type == "quantum":
                    # Use stellar sound for quantum properties
                    sound_data = universe_sounds.generate_stellar_sound(
                        "sun", sound_duration, sound_amplitude)
                    
                elif sound_type == "resonant":
                    # Use dimensional transition for resonant properties
                    sound_data = universe_sounds.generate_dimensional_transition(
                        sound_duration, SAMPLE_RATE, 
                        "cord_to_earth", sound_amplitude)
                
                # Save sound file if data was generated
                if sound_data is not None:
                    sound_file = f"life_cord_{sound_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                    file_path = sound_gen.save_sound(sound_data, sound_file, 
                                                 f"Life Cord {sound_type.capitalize()} Sound")
                    
                    sound_enhancement["sound_file"] = sound_file
                    sound_enhancement["file_path"] = file_path
                    sound_enhancement["has_physical_sound"] = True
                    
                    logger.info(f"Generated {sound_type} sound for cord enhancement: {file_path}")
            
            except Exception as sound_err:
                logger.warning(f"Error generating physical sound: {sound_err}. Using theoretical model.")
        
        # Calculate enhancement factors regardless of physical sound
        # These represent the theoretical effect of the sound on the cord
        
        # Base enhancement from sound type
        if sound_type == "harmonic":
            # Harmonic sound enhances cord stability and elasticity
            stability_boost = 0.15
            elasticity_boost = 0.10
            bandwidth_boost = 0.05
            
        elif sound_type == "quantum":
            # Quantum sound enhances bandwidth and provides some stability
            stability_boost = 0.05
            elasticity_boost = 0.05
            bandwidth_boost = 0.20
            
        elif sound_type == "resonant":
            # Resonant sound enhances elasticity and bandwidth
            stability_boost = 0.08
            elasticity_boost = 0.15
            bandwidth_boost = 0.12
            
        else:
            # Default modest enhancements
            stability_boost = 0.05
            elasticity_boost = 0.05
            bandwidth_boost = 0.05
        
        # Calculate new property values with sound enhancement
        new_stability = min(1.0, stability_factor + stability_boost)
        new_elasticity = min(1.0, elasticity + elasticity_boost)
        new_bandwidth = bandwidth_hz * (1.0 + bandwidth_boost)
        
        # Update cord structure with enhanced properties
        cord_structure["stability_factor"] = float(new_stability)
        cord_structure["elasticity_factor"] = float(new_elasticity)
        cord_structure["bandwidth_hz"] = float(new_bandwidth)
        
        # Store enhancement factors
        sound_enhancement["enhancement_factors"] = {
            "stability_boost": float(stability_boost),
            "elasticity_boost": float(elasticity_boost),
            "bandwidth_boost": float(bandwidth_boost),
            "new_stability": float(new_stability),
            "new_elasticity": float(new_elasticity),
            "new_bandwidth_hz": float(new_bandwidth)
        }
        
        # Add sound enhancement to cord structure
        if not "sound_enhancements" in cord_structure:
            cord_structure["sound_enhancements"] = []
        
        cord_structure["sound_enhancements"].append(sound_enhancement)
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "sound_type": sound_type,
                "has_physical_sound": sound_enhancement["has_physical_sound"],
                "stability_boost": float(stability_boost),
                "elasticity_boost": float(elasticity_boost),
                "bandwidth_boost": float(bandwidth_boost),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('life_cord_sound_enhancement', metrics_data)
        
        logger.info(f"Cord enhanced with {sound_type} sound: "
                   f"Stability +{stability_boost:.3f}, "
                   f"Elasticity +{elasticity_boost:.3f}, "
                   f"Bandwidth +{bandwidth_boost*100:.1f}%")
        return sound_enhancement
        
    except Exception as e:
        logger.error(f"Error enhancing cord with sound: {e}", exc_info=True)
        raise RuntimeError(f"Sound enhancement failed critically: {e}") from e

def _optimize_layered_stability(soul_spark: SoulSpark, cord_structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimizes cord stability by creating resonant connections with soul layers,
    establishing bidirectional energy exchange pathways that strengthen
    the cord's integration with the soul's aura structure.
    
    Args:
        soul_spark: The soul being optimized
        cord_structure: The developing cord structure
        
    Returns:
        Dict containing layer optimization metrics
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If optimization fails
    """
    logger.info("LC Step: Optimizing Layered Stability...")
    
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not isinstance(cord_structure, dict):
        raise ValueError("Cord structure must be a dictionary")
    
    try:
        # Extract key properties
        primary_freq = cord_structure.get("primary_frequency_hz", 0.0)
        stability_factor = cord_structure.get("stability_factor", 0.0)
        soul_layers = getattr(soul_spark, 'layers', [])
        
        if primary_freq <= FLOAT_EPSILON:
            raise ValueError(f"Invalid primary frequency: {primary_freq}")
        if not soul_layers:
            logger.warning("No soul layers found for optimization")
            return {"layer_connections": [], "stability_boost": 0.0}
        
        # Get harmonic nodes and light pathways if available
        harmonic_nodes = cord_structure.get("harmonic_nodes", [])
        light_pathways_structure = cord_structure.get("light_pathways", {})
        light_frequencies = light_pathways_structure.get("light_frequencies", [])
        
        # Initialize layer connections data
        layer_connections = []
        resonant_layers = 0
        total_resonance = 0.0
        
        # Process each soul layer to establish resonance
        for layer_idx, layer in enumerate(soul_layers):
            if not isinstance(layer, dict):
                continue
                
            # Get layer frequencies if available
            layer_freqs = []
            if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
                layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
            
            # If layer has no frequencies, skip
            if not layer_freqs:
                continue
                
            # Calculate resonance with cord frequencies
            best_resonance = 0.0
            best_cord_freq = primary_freq
            best_layer_freq = 0.0
            
            # Check resonance with primary frequency
            for layer_freq in layer_freqs:
                res = calculate_resonance(primary_freq, layer_freq)
                if res > best_resonance:
                    best_resonance = res
                    best_cord_freq = primary_freq
                    best_layer_freq = layer_freq
            
            # Check resonance with harmonic nodes
            for node in harmonic_nodes:
                node_freq = node.get("frequency_hz", 0.0)
                if node_freq <= FLOAT_EPSILON:
                    continue
                    
                for layer_freq in layer_freqs:
                    res = calculate_resonance(node_freq, layer_freq)
                    if res > best_resonance:
                        best_resonance = res
                        best_cord_freq = node_freq
                        best_layer_freq = layer_freq
            
            # Check resonance with light frequencies
            for light_freq_data in light_frequencies:
                audio_freq = light_freq_data.get("audio_freq", 0.0)
                if audio_freq <= FLOAT_EPSILON:
                    continue
                    
                for layer_freq in layer_freqs:
                    res = calculate_resonance(audio_freq, layer_freq)
                    if res > best_resonance:
                        best_resonance = res
                        best_cord_freq = audio_freq
                        best_layer_freq = layer_freq
            
            # If no significant resonance found (below threshold), skip this layer
            if best_resonance < 0.2:
                continue
                
            # Create resonant connection with this layer
            connection_strength = min(1.0, best_resonance * 1.2)  # Slight boost
            
            # Calculate stability boost from this connection
            layer_stability_boost = connection_strength * 0.05  # Max 5% boost per layer
            
            # Create connection data
            connection = {
                "layer_idx": layer_idx,
                "cord_frequency": float(best_cord_freq),
                "layer_frequency": float(best_layer_freq),
                "resonance": float(best_resonance),
                "connection_strength": float(connection_strength),
                "stability_boost": float(layer_stability_boost)
            }
            
            # Add to layer connections list
            layer_connections.append(connection)
            resonant_layers += 1
            total_resonance += best_resonance
            
            # Store connection in layer data for future reference
            if 'cord_connections' not in layer:
                layer['cord_connections'] = []
                
            layer['cord_connections'].append({
                "frequency": float(best_cord_freq),
                "resonance": float(best_resonance),
                "strength": float(connection_strength),
                "timestamp": datetime.now().isoformat()
            })
        
        # Calculate overall stability boost and coherence enhancement
        avg_resonance = total_resonance / resonant_layers if resonant_layers > 0 else 0.0
        overall_stability_boost = min(0.2, resonant_layers * 0.03)  # Cap at 20%
        overall_coherence_boost = min(0.15, avg_resonance * 0.2)  # Cap at 15%
        
        # Update cord structure with new stability factor
        new_stability = min(1.0, stability_factor + overall_stability_boost)
        cord_structure["stability_factor"] = float(new_stability)
        
        # Create optimization metrics
        optimization_metrics = {
            "layer_connections": layer_connections,
            "resonant_layers": resonant_layers,
            "total_resonance": float(total_resonance),
            "average_resonance": float(avg_resonance),
            "stability_boost": float(overall_stability_boost),
            "coherence_boost": float(overall_coherence_boost),
            "new_stability_factor": float(new_stability)
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "resonant_layers": resonant_layers,
                "average_resonance": float(avg_resonance),
                "stability_boost": float(overall_stability_boost),
                "coherence_boost": float(overall_coherence_boost),
                "new_stability_factor": float(new_stability),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('life_cord_layer_optimization', metrics_data)
        
        logger.info(f"Layered stability optimized: {resonant_layers} layers connected, "
                   f"Avg Resonance: {avg_resonance:.4f}, "
                   f"Stability Boost: +{overall_stability_boost:.4f}")
        
        return optimization_metrics
        
    except Exception as e:
        logger.error(f"Error optimizing layered stability: {e}", exc_info=True)
        raise RuntimeError("Layered stability optimization failed critically.") from e

def _wavelength_to_color(wavelength_nm: float) -> str:
    """
    Convert a wavelength in nanometers to a named color in the visible spectrum.
    
    Args:
        wavelength_nm: Wavelength in nanometers
        
    Returns:
        String name of the corresponding color
        
    Raises:
        ValueError: If wavelength is outside visible spectrum
    """
    if not isinstance(wavelength_nm, (int, float)):
        raise ValueError(f"Wavelength must be a number, got {type(wavelength_nm)}")
    
    if wavelength_nm < 380 or wavelength_nm > 750:
        # Return non-visible designations
        if wavelength_nm < 380:
            return "ultraviolet"
        else:
            return "infrared"
    
    # Visible spectrum color mapping
    if 380 <= wavelength_nm < 450:
        return "violet"
    elif 450 <= wavelength_nm < 485:
        return "blue"
    elif 485 <= wavelength_nm < 500:
        return "cyan"
    elif 500 <= wavelength_nm < 565:
        return "green"
    elif 565 <= wavelength_nm < 590:
        return "yellow"
    elif 590 <= wavelength_nm < 625:
        return "orange"
    elif 625 <= wavelength_nm <= 750:
        return "red"
    
    # This should not be reached due to the earlier range check
    return "unknown"

def _calculate_earth_resonance(soul_spark: SoulSpark, cord_integrity: float) -> float:
    """
    Calculate soul's resonance with Earth based on cord integrity and connection.
    This measures how effectively the soul can interact with the physical plane.
    
    Args:
        soul_spark: The soul being evaluated
        cord_integrity: Life cord integrity factor (0-1)
        
    Returns:
        Earth resonance factor (0-1)
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If calculation fails
    """
    logger.info("LC Step: Calculating Earth Resonance...")
    
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not isinstance(cord_integrity, (int, float)) or not (0.0 <= cord_integrity <= 1.0):
        raise ValueError(f"Cord integrity must be between 0.0 and 1.0, got {cord_integrity}")
    
    try:
        # Get core soul properties
        soul_frequency = getattr(soul_spark, 'frequency', 0.0)
        if soul_frequency <= FLOAT_EPSILON:
            raise ValueError("Soul frequency must be positive")
        
        soul_coherence = getattr(soul_spark, 'coherence', 0.0) / MAX_COHERENCE_CU
        soul_stability = getattr(soul_spark, 'stability', 0.0) / MAX_STABILITY_SU
        
        # Get physical and Gaia frequency factors
        earth_freq = EARTH_FREQUENCY
        schumann_freq = SCHUMANN_FREQUENCY
        
        # Calculate base resonance with Earth physical frequency
        base_resonance = calculate_resonance(soul_frequency, earth_freq)
        
        # Calculate resonance with Schumann frequency (spiritual connection)
        schumann_resonance = calculate_resonance(soul_frequency, schumann_freq)
        
        # Calculate soul-Earth-Gaia harmonic triad
        # This represents alignment of physical, energetic, and spiritual connection
        triad_factor = calculate_resonance(soul_frequency, (earth_freq + schumann_freq) / 2)
        
        # Calculate Earth connection strength
        # - Cord integrity is the main factor (physical connection)
        # - Soul coherence affects energy exchange quality
        # - Soul stability ensures connection remains steady
        earth_connection = (
            cord_integrity * 0.6 +      # Main connection via cord
            soul_coherence * 0.2 +      # Quality of energy exchange
            soul_stability * 0.1 +      # Stability of connection
            triad_factor * 0.1          # Harmonic alignment bonus
        )
        
        # Calculate final Earth resonance
        # This is a combination of physical connection and frequency resonance
        earth_resonance = (
            earth_connection * 0.7 +    # Physical connection is primary
            base_resonance * 0.2 +      # Physical frequency resonance
            schumann_resonance * 0.1    # Spiritual/energetic resonance
        )
        
        # Ensure result is in valid range
        earth_resonance = max(0.0, min(1.0, earth_resonance))
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "cord_integrity": float(cord_integrity),
                "base_resonance": float(base_resonance),
                "schumann_resonance": float(schumann_resonance),
                "triad_factor": float(triad_factor),
                "earth_connection": float(earth_connection),
                "earth_resonance": float(earth_resonance),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('life_cord_earth_resonance', metrics_data)
        
        logger.info(f"Earth resonance calculated: {earth_resonance:.4f} "
                   f"(Cord: {cord_integrity:.3f}, Base Res: {base_resonance:.3f}, "
                   f"Schumann: {schumann_resonance:.3f})")
        
        return float(earth_resonance)
        
    except Exception as e:
        logger.error(f"Error calculating Earth resonance: {e}", exc_info=True)
        raise RuntimeError("Earth resonance calculation failed critically.") from e

def _finalize_cord_integration(soul_spark: SoulSpark, 
                            cord_structure: Dict[str, Any], 
                            earth_resonance: float) -> Dict[str, Any]:
    """
    Finalize the integration of the life cord with the soul, ensuring proper
    energy flow and establishing the full connection between soul and Earth.
    
    Args:
        soul_spark: The soul being connected
        cord_structure: The complete cord structure
        earth_resonance: The calculated Earth resonance factor
        
    Returns:
        Dict containing final integration metrics
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If integration fails
    """
    logger.info("LC Step: Finalizing Cord Integration...")
    
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not isinstance(cord_structure, dict):
        raise ValueError("Cord structure must be a dictionary")
    if not isinstance(earth_resonance, (int, float)) or not (0.0 <= earth_resonance <= 1.0):
        raise ValueError(f"Earth resonance must be between 0.0 and 1.0, got {earth_resonance}")
    
    try:
        # Get essential metrics from cord structure
        stability_factor = cord_structure.get("stability_factor", 0.0)
        integrity_factor = cord_structure.get("integrity_factor", 0.0)
        bandwidth = cord_structure.get("bandwidth_hz", 0.0)
        
        # Get key components
        light_pathways = cord_structure.get("light_pathways", {})
        sound_enhancements = cord_structure.get("sound_enhancements", [])
        standing_wave = cord_structure.get("standing_wave", {})
        
        # Calculate overall integration quality
        avg_pathway_strength = light_pathways.get("avg_pathway_strength", 0.0)
        standing_wave_coherence = standing_wave.get("coherence_factor", 0.0)
        
        # Integration factors
        physical_integration = min(1.0, earth_resonance * 1.2)  # Slight boost to earth resonance
        energetic_integration = min(1.0, avg_pathway_strength * 1.1)  # Light pathways
        structural_integration = min(1.0, stability_factor)  # Stability represents structural integrity
        
        # Calculate overall integration
        overall_integration = (
            physical_integration * 0.4 +
            energetic_integration * 0.3 +
            structural_integration * 0.3
        )
        
        # Finalize cord integrity
        final_integrity = (
            integrity_factor * 0.5 +
            overall_integration * 0.3 +
            earth_resonance * 0.2
        )
        
        # Ensure results are in valid range
        overall_integration = max(0.0, min(1.0, overall_integration))
        final_integrity = max(0.0, min(1.0, final_integrity))
        
        # Create finalization time
        finalization_time = datetime.now().isoformat()
        
        # Store results in soul_spark
        setattr(soul_spark, "cord_integrity", float(final_integrity))
        setattr(soul_spark, "earth_resonance", float(earth_resonance))
        setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, True)
        
        # Store final cord structure in soul_spark
        final_cord = {
            "integrity_factor": float(final_integrity),
            "stability_factor": float(stability_factor),
            "bandwidth_hz": float(bandwidth),
            "overall_integration": float(overall_integration),
            "earth_resonance": float(earth_resonance),
            "creation_timestamp": cord_structure.get("creation_timestamp", ""),
            "finalization_timestamp": finalization_time
        }
        
        # Add summary of key structures
        if light_pathways:
            final_cord["light_pathways_summary"] = {
                "count": light_pathways.get("total_pathway_count", 0),
                "avg_strength": float(avg_pathway_strength),
                "quantum_tunneling": float(light_pathways.get("quantum_tunneling_factor", 0.0))
            }
            
        if standing_wave:
            final_cord["standing_wave_summary"] = {
                "node_count": standing_wave.get("node_count", 0),
                "coherence_factor": float(standing_wave_coherence),
                "average_amplitude": float(standing_wave.get("average_amplitude", 0.0))
            }
            
        if sound_enhancements:
            final_cord["sound_enhancements_summary"] = [
                {"type": se.get("type", ""), "boost": se.get("enhancement_factors", {}).get("stability_boost", 0.0)}
                for se in sound_enhancements
            ]
        
        # Set the complete structure in soul_spark
        setattr(soul_spark, "life_cord", final_cord)
        
        # Store last modified timestamp
        setattr(soul_spark, "last_modified", finalization_time)
        
        # Create finalization metrics
        finalization_metrics = {
            "cord_integrity": float(final_integrity),
            "earth_resonance": float(earth_resonance),
            "overall_integration": float(overall_integration),
            "physical_integration": float(physical_integration),
            "energetic_integration": float(energetic_integration),
            "structural_integration": float(structural_integration),
            "finalization_timestamp": finalization_time
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('life_cord_finalization', finalization_metrics)
        
        logger.info(f"Life cord integration finalized: Integrity={final_integrity:.4f}, "
                   f"Earth Resonance={earth_resonance:.4f}, "
                   f"Overall Integration={overall_integration:.4f}")
        
        return finalization_metrics
        
    except Exception as e:
        logger.error(f"Error finalizing cord integration: {e}", exc_info=True)
        raise RuntimeError("Cord integration finalization failed critically.") from e

def form_life_cord(soul_spark: SoulSpark, intensity: float = 0.7, complexity: float = 0.5) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Forms the life cord connecting the soul to Earth, enabling incarnation.
    Establishes a stable energetic connection that facilitates the flow of
    information and energy between the spiritual and physical realms.
    
    Args:
        soul_spark: The soul forming the life cord
        intensity: Intensity of the cord formation process (0.1-1.0)
        complexity: Complexity/sophistication of the resulting cord (0.1-1.0)
        
    Returns:
        Tuple of (modified soul_spark, process_metrics)
        
    Raises:
        ValueError: For invalid inputs
        RuntimeError: If cord formation fails
    """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("soul_spark must be a SoulSpark instance")
    if not isinstance(intensity, (int, float)) or not (0.1 <= intensity <= 1.0):
        raise ValueError(f"Intensity must be between 0.1 and 1.0, got {intensity}")
    if not isinstance(complexity, (int, float)) or not (0.1 <= complexity <= 1.0):
        raise ValueError(f"Complexity must be between 0.1 and 1.0, got {complexity}")
    
    spark_id = getattr(soul_spark, "spark_id", "unknown")
    logger.info(f"--- Beginning Life Cord Formation for Soul {spark_id} (Int={intensity:.2f}, Cmplx={complexity:.2f}) ---")
    start_time = datetime.now().isoformat()
    
    # Prepare metrics container
    process_metrics = {
        "steps": {},
        "soul_id": spark_id,
        "start_time": start_time,
        "intensity": float(intensity),
        "complexity": float(complexity),
        "success": False  # Will be set to True if successful
    }
    
    try:
        # --- 1. Check Prerequisites ---
        logger.info("Life Cord Step 1: Checking Prerequisites...")
        _ensure_soul_properties(soul_spark)
        _check_prerequisites(soul_spark)
        
        # Energy check before proceeding
        initial_energy = soul_spark.energy
        if initial_energy < CORD_ACTIVATION_ENERGY_COST:
            raise ValueError(f"Insufficient energy ({initial_energy:.1f} SEU) for cord formation. Required: {CORD_ACTIVATION_ENERGY_COST} SEU")
        
        # Apply energy cost
        soul_spark.energy -= CORD_ACTIVATION_ENERGY_COST
        
        # --- 2. Establish Anchor Points ---
        logger.info("Life Cord Step 2: Establishing Anchor Points...")
        soul_anchor, earth_anchor, connection_strength = _establish_soul_earth_anchors(soul_spark)
        process_metrics["steps"]["anchor_points"] = {
            "soul_anchor": soul_anchor,
            "earth_anchor": earth_anchor,
            "connection_strength": float(connection_strength)
        }
        
        # --- 3. Form Bidirectional Waveguide ---
        logger.info("Life Cord Step 3: Forming Bidirectional Waveguide...")
        waveguide = _form_bidirectional_waveguide(soul_anchor, earth_anchor, connection_strength, complexity)
        process_metrics["steps"]["waveguide"] = {
            "length": float(waveguide["length"]),
            "diameter": float(waveguide["diameter"]),
            "soul_to_earth_efficiency": float(waveguide["soul_to_earth_efficiency"]),
            "earth_to_soul_efficiency": float(waveguide["earth_to_soul_efficiency"]),
            "quantum_efficiency": float(waveguide["quantum_efficiency"])
        }
        
        # Debug the waveguide to check frequency properties
        logger.debug(f"Waveguide created with keys: {waveguide.keys()}")
        logger.debug(f"Waveguide frequency properties: primary_frequency_hz={waveguide.get('primary_frequency_hz', 'NOT SET')}, bandwidth_hz={waveguide.get('bandwidth_hz', 'NOT SET')}")
        
        # Ensure waveguide has required frequency properties
        if 'primary_frequency_hz' not in waveguide or waveguide['primary_frequency_hz'] <= FLOAT_EPSILON:
            soul_frequency = getattr(soul_spark, 'frequency', None)
            if soul_frequency and soul_frequency > FLOAT_EPSILON:
                logger.info(f"Setting missing primary_frequency_hz to soul frequency: {soul_frequency}")
                waveguide['primary_frequency_hz'] = float(soul_frequency)
            else:
                logger.error(f"Soul frequency invalid or missing: {soul_frequency}")
                raise ValueError(f"Cannot create waveguide without valid frequency. Soul frequency: {soul_frequency}")
        
        if 'bandwidth_hz' not in waveguide or waveguide['bandwidth_hz'] <= FLOAT_EPSILON:
            # Set bandwidth to 10% of primary frequency
            waveguide['bandwidth_hz'] = float(waveguide['primary_frequency_hz'] * 0.1)
            logger.info(f"Setting missing bandwidth_hz to 10% of primary: {waveguide['bandwidth_hz']}")
        
        # --- 4. Create Harmonic Nodes ---
        logger.info("Life Cord Step 4: Creating Harmonic Nodes...")
        harmonic_nodes = _create_harmonic_nodes(waveguide, complexity)
        process_metrics["steps"]["harmonic_nodes"] = {
            "node_count": len(harmonic_nodes)
        }
        
        # --- 5. Create Light Pathways ---
        logger.info("Life Cord Step 5: Creating Light Pathways...")
        light_pathways = _create_light_pathways(waveguide, harmonic_nodes)
        process_metrics["steps"]["light_pathways"] = {
            "pathway_count": light_pathways["total_pathway_count"],
            "avg_pathway_strength": float(light_pathways["avg_pathway_strength"]),
            "max_entanglement_quality": float(light_pathways["max_entanglement_quality"])
        }
        
        # --- 6. Calculate Information Bandwidth ---
        logger.info("Life Cord Step 6: Calculating Information Bandwidth...")
        bandwidth_metrics = _calculate_information_bandwidth(waveguide, light_pathways)
        process_metrics["steps"]["bandwidth"] = {
            "effective_bandwidth_hz": float(bandwidth_metrics["effective_bandwidth_hz"]),
            "information_capacity_bps": float(bandwidth_metrics["information_capacity_bps"])
        }
        
        # --- 7. Integrate with Aura Layers ---
        logger.info("Life Cord Step 7: Integrating with Aura Layers...")
        # Create temporary cord structure to integrate with layers
        temp_cord_structure = {
            "primary_frequency_hz": soul_anchor["frequency"],
            "earth_freq": earth_anchor["frequency"],
            "soul_primary_freq": soul_anchor["frequency"],
            "bandwidth_hz": bandwidth_metrics["effective_bandwidth_hz"],
            "creation_timestamp": start_time,
            "integrity_factor": connection_strength * 0.8,  # Initial integrity
            "stability_factor": connection_strength * 0.7,  # Initial stability
            "elasticity_factor": 0.5,  # Default elasticity
            "waveguide": waveguide,
            "harmonic_nodes": harmonic_nodes,
            "light_pathways": light_pathways
        }
        
        layer_integration = _integrate_with_aura_layers(soul_spark, waveguide, harmonic_nodes)
        process_metrics["steps"]["layer_integration"] = {
            "integrated_layers": layer_integration["integrated_layers"],
            "total_resonance": float(layer_integration["total_resonance"])
        }
        
        # --- 8. Create Standing Wave Patterns ---
        logger.info("Life Cord Step 8: Creating Standing Wave Patterns...")
        standing_wave = _create_standing_wave_nodes(temp_cord_structure, connection_strength * complexity)
        process_metrics["steps"]["standing_wave"] = {
            "node_count": standing_wave["node_count"],
            "coherence_factor": float(standing_wave["coherence_factor"]),
            "stability_boost": float(standing_wave["stability_boost"])
        }
        
        # --- 9. Enhance with Sound ---
        logger.info("Life Cord Step 9: Enhancing with Sound...")
        
        # Sound type based on intensity and complexity
        if intensity > 0.8:
            sound_type = "quantum"  # High intensity - quantum properties
        elif complexity > 0.7:
            sound_type = "harmonic"  # High complexity - harmonic structures
        else:
            sound_type = "resonant"  # Default - resonant properties
            
        sound_enhancement = _enhance_cord_with_sound(temp_cord_structure, sound_type)
        process_metrics["steps"]["sound_enhancement"] = {
            "sound_type": sound_type,
            "has_physical_sound": sound_enhancement["has_physical_sound"],
            "stability_boost": float(sound_enhancement["enhancement_factors"]["stability_boost"])
        }
        
        # --- 10. Optimize Layered Stability ---
        logger.info("Life Cord Step 10: Optimizing Layered Stability...")
        layer_optimization = _optimize_layered_stability(soul_spark, temp_cord_structure)
        process_metrics["steps"]["layer_optimization"] = {
            "resonant_layers": layer_optimization["resonant_layers"],
            "stability_boost": float(layer_optimization["stability_boost"]),
            "coherence_boost": float(layer_optimization["coherence_boost"])
        }
        
        # --- 11. Calculate Earth Resonance ---
        logger.info("Life Cord Step 11: Calculating Earth Resonance...")
        earth_resonance = _calculate_earth_resonance(soul_spark, temp_cord_structure["integrity_factor"])
        process_metrics["steps"]["earth_resonance"] = {
            "earth_resonance": float(earth_resonance)
        }
        
        # --- 12. Finalize Cord Integration ---
        logger.info("Life Cord Step 12: Finalizing Cord Integration...")
        finalization = _finalize_cord_integration(soul_spark, temp_cord_structure, earth_resonance)
        process_metrics["steps"]["finalization"] = {
            "cord_integrity": float(finalization["cord_integrity"]),
            "earth_resonance": float(finalization["earth_resonance"]),
            "overall_integration": float(finalization["overall_integration"])
        }
        
        # --- 13. Update Soul State ---
        logger.info("Life Cord Step 13: Updating Soul State...")
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            logger.debug(f"Updated soul state after cord formation: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        else:
            logger.warning("Soul update_state method not found. Skipping soul state update.")
        
        # --- Finalize Process Metrics ---
        end_time = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time)
        start_time_dt = datetime.fromisoformat(start_time)
        duration_seconds = (end_time_dt - start_time_dt).total_seconds()
        
        process_metrics["end_time"] = end_time
        process_metrics["duration_seconds"] = duration_seconds
        process_metrics["success"] = True
        process_metrics["final_cord_integrity"] = float(soul_spark.cord_integrity)
        process_metrics["final_earth_resonance"] = float(soul_spark.earth_resonance)
        process_metrics["energy_consumed"] = float(CORD_ACTIVATION_ENERGY_COST)
        
        # Log summary
        logger.info(f"--- Life Cord Formation Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Final Metrics: Integrity={soul_spark.cord_integrity:.4f}, "
                   f"Earth Resonance={soul_spark.earth_resonance:.4f}, "
                   f"Duration={duration_seconds:.2f}s")
        
        # Record overall metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('life_cord_summary', process_metrics)
        
        return soul_spark, process_metrics
        
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Life cord formation failed for {spark_id}: {e}")
        # Set failure flag
        setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False)
        
        # Record failure metrics
        process_metrics["success"] = False
        process_metrics["error"] = str(e)
        process_metrics["end_time"] = datetime.now().isoformat()
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('life_cord_summary', process_metrics)
            
        raise  # Re-raise original error
        
    except Exception as e:
        logger.critical(f"Unexpected error during life cord formation for {spark_id}: {e}", exc_info=True)
        # Set failure flag
        setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False)
        
        # Record failure metrics
        process_metrics["success"] = False
        process_metrics["error"] = str(e)
        process_metrics["end_time"] = datetime.now().isoformat()
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('life_cord_summary', process_metrics)
            
        raise RuntimeError(f"Unexpected life cord formation failure: {e}") from e
