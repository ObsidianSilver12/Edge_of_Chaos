"""
Life Cord Formation Functions (Refactored V5.0.0 - Resonant Channel Principles)

Creates an indestructible divine life cord with two anchors (earth and spiritual)
and communication channels based on natural resonance principles. The soul must
adjust its resonance to match the anchors to establish communication through biomimetic
adaptation similar to meditation practice. The cord itself remains perfect divine energy
while the soul adapts its frequency temporarily without changing its base nature.

Records the resonance journey using edge of chaos principles for metrics tracking.
Hard fails on critical errors with no fallbacks for reliable debugging.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import uuid
from constants.constants import *
from typing import Dict, List, Any, Tuple, Optional
from math import pi as PI, sqrt, exp, sin, cos, tanh, log


# --- Logging ---
logger = logging.getLogger(__name__)

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    
    # Try to import resonance calculation from creator_entanglement
    try:
        from .creator_entanglement import calculate_resonance
    except (ImportError, AttributeError):
        # Define a local version if import fails
        def calculate_resonance(freq1: float, freq2: float) -> float:
            """Calculate natural resonance between frequencies."""
            if freq1 <= FLOAT_EPSILON or freq2 <= FLOAT_EPSILON:
                return 0.0
                
            # Calculate logarithmic difference (perceptual)
            log_diff = abs(np.log(freq1) - np.log(freq2))
            tolerance = 0.15  # Resonance width parameter
            
            # Main resonance (Gaussian distribution - natural resonance curve)
            main_resonance = np.exp(-log_diff**2 / (2 * tolerance**2))
            
            # Check for harmonic relationships (integer ratios and phi)
            ratio = max(freq1, freq2) / min(freq1, freq2)
            harmonic_resonance = 0.0
            
            # Integer ratios (natural harmonics)
            for i in range(1, 6):
                for j in range(1, 6):
                    if j == 0: continue
                    target_ratio = float(i) / float(j)
                    ratio_diff = abs(ratio - target_ratio)
                    if ratio_diff < 0.1 * target_ratio:
                        harmonic_score = 1.0 - (ratio_diff/(0.1 * target_ratio))
                        harmonic_resonance = max(harmonic_resonance, harmonic_score)
            
            # Phi ratios (natural growth patterns)
            phi_ratios = [PHI, 1/PHI, PHI**2, 1/(PHI**2)]
            for phi_ratio in phi_ratios:
                ratio_diff = abs(ratio - phi_ratio)
                if ratio_diff < 0.1 * phi_ratio:
                    phi_score = 1.0 - (ratio_diff/(0.1 * phi_ratio))
                    harmonic_resonance = max(harmonic_resonance, phi_score)
                    
            # Combine using weighted average (natural principle)
            final_resonance = 0.7 * main_resonance + 0.3 * harmonic_resonance
            return max(0.0, min(1.0, final_resonance))

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

# --- Prerequisite Validation ---

def _check_prerequisites(soul_spark: SoulSpark) -> None:
    """
    Checks prerequisites using SU/CU thresholds. 
    Raises ValueError on failure - hard fail with no fallbacks.
    """
    logger.debug(f"Checking life cord prerequisites for soul {soul_spark.spark_id}...")
    
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False):
        raise ValueError(f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_LIFE_CORD}.")

    # 2. Minimum Stability and Coherence (Absolute SU/CU)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    
    if stability_su < 0 or coherence_cu < 0:
        raise AttributeError("Prerequisite failed: Soul missing stability or coherence attributes.")

    if stability_su < CORD_STABILITY_THRESHOLD_SU:
        raise ValueError(f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {CORD_STABILITY_THRESHOLD_SU} SU.")
        
    if coherence_cu < CORD_COHERENCE_THRESHOLD_CU:
        raise ValueError(f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {CORD_COHERENCE_THRESHOLD_CU} CU.")

    logger.debug("Life cord prerequisites met.")


def _ensure_soul_properties(soul_spark: SoulSpark) -> None:
    """
    Ensure soul has necessary properties. 
    Raises error if missing/invalid - hard fail for reliability.
    """
    logger.debug(f"Ensuring properties for life cord formation (Soul {soul_spark.spark_id})...")
    
    required = [
        'frequency', 'stability', 'coherence', 'position', 'field_radius',
        'field_strength', 'creator_connection_strength', 'energy'
    ]
    
    missing = [attr for attr in required if not hasattr(soul_spark, attr)]
    if missing:
        raise AttributeError(f"SoulSpark missing essential attributes for Life Cord: {missing}")

    if not hasattr(soul_spark, 'life_cord'): 
        setattr(soul_spark, 'life_cord', {})
        
    if not hasattr(soul_spark, 'cord_integrity'): 
        setattr(soul_spark, 'cord_integrity', 0.0)

    if soul_spark.frequency <= FLOAT_EPSILON: 
        raise ValueError("Soul frequency must be positive.")
        
    pos = getattr(soul_spark, 'position')
    if not isinstance(pos, list) or len(pos) != 3: 
        raise ValueError(f"Invalid position: {pos}")
        
    if soul_spark.energy < CORD_ACTIVATION_ENERGY_COST:
        raise ValueError(f"Insufficient energy ({soul_spark.energy:.1f} SEU) for cord activation cost ({CORD_ACTIVATION_ENERGY_COST} SEU).")

    logger.debug("Soul properties ensured for Life Cord.")


# --- Core Implementation ---

def _establish_divine_cord() -> Dict[str, Any]:
    """
    Creates the divine cord structure with perfect, indestructible properties.
    The divine cord is composed of pure creator energy that follows fundamental
    universal laws rather than biological ones.
    
    Returns:
        Dict containing the divine cord properties
    """
    logger.info("LC Step: Establishing Divine Cord...")
    
    creation_timestamp = datetime.now().isoformat()
    
    # Create the perfect divine cord
    divine_cord = {
        "divine_properties": {
            "integrity": 1.0,               # Perfect integrity - indestructible
            "stability": 1.0,               # Perfect stability - unwavering
            "elasticity": 1.0,              # Perfect elasticity - adapts without distortion
            "resonance": 1.0,               # Perfect resonance - pure creator frequency
            "creation_timestamp": creation_timestamp,
            "divine_energy_purity": 1.0,    # 100% pure creator energy
            "divine_energy_level": 1.0,     # Full creator energy
            "divine_laws": {
                "immutability": 1.0,        # Cannot be corrupted
                "omnipresence": 1.0,        # Spans dimensions seamlessly
                "efficiency": 1.0           # Perfect energy transmission
            }
        },
        "creation_timestamp": creation_timestamp
    }
    
    logger.info("Divine Cord established with perfect properties.")
    return divine_cord


def _create_spiritual_anchor(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    Creates the spiritual anchor based on soul properties and creator connection.
    This anchor connects to the spiritual aspect of the soul and requires
    the soul to resonate upward toward creator frequency.
    
    Args:
        soul_spark: The SoulSpark object
        
    Returns:
        Dict containing the spiritual anchor properties
    """
    logger.info("LC Step: Creating Spiritual Anchor...")
    
    # Get soul properties
    soul_frequency = getattr(soul_spark, 'frequency', 432.0)
    creator_connection = getattr(soul_spark, 'creator_connection_strength', 0.0)
    soul_coherence = getattr(soul_spark, 'coherence', 0.0) / MAX_COHERENCE_CU
    soul_position = getattr(soul_spark, 'position', [0, 0, 0])
    
    # Calculate spiritual anchor frequency using natural scaling
    creator_freq = 528.0  # Divine frequency
    
    # Apply a natural phi-based blending for spiritual frequency
    phi_factor = PHI / (PHI + 1)  # ~0.618
    blend_factor = creator_connection * phi_factor
    spiritual_freq = soul_frequency * (1.0 - blend_factor) + creator_freq * blend_factor
    
    # Calculate position (above soul) - natural geometric placement
    spiritual_pos = [float(p) for p in soul_position]
    spiritual_pos[2] += soul_frequency / 43.2  # Natural height based on frequency
    
    # Calculate resonance target using natural curve
    # Higher creator connection = more precise matching needed (exponential curve)
    resonance_base = 0.5 + (1.0 - exp(-creator_connection * 2.0)) * 0.3
    resonance_target = max(0.6, min(0.9, resonance_base))
    
    # Calculate tolerance based on soul coherence (more coherent = narrower tolerance)
    # Natural logarithmic relationship for tolerance scaling
    tolerance_base = 0.25
    coherence_factor = max(0.0, min(1.0, soul_coherence))
    resonance_tolerance = tolerance_base * (1.0 - coherence_factor * 0.6)
    
    # Create spiritual anchor
    spiritual_anchor = {
        "position": spiritual_pos,
        "frequency": float(spiritual_freq),
        "resonance_target": float(resonance_target),
        "resonance_tolerance": float(resonance_tolerance),
        "creator_connection": float(creator_connection),
        "soul_frequency_base": float(soul_frequency),
        "creator_frequency": float(creator_freq),
        "metadata": {
            "creation_timestamp": datetime.now().isoformat(),
            "geometric_principles": "phi_based_positioning",
            "frequency_principles": "creator_soul_resonance_blend"
        }
    }
    
    logger.info(f"Spiritual Anchor created at frequency {spiritual_freq:.2f}Hz "
               f"with resonance target {resonance_target:.3f} "
               f"(tolerance: {resonance_tolerance:.3f})")
    
    return spiritual_anchor


def _create_earth_anchor(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    Creates the earth anchor based on Earth/Gaia frequencies and the soul's
    previous interaction with Malkuth during the Sephiroth journey.
    
    Args:
        soul_spark: The SoulSpark object to extract Malkuth resonance data
        
    Returns:
        Dict containing the earth anchor properties
    """
    logger.info("LC Step: Creating Earth Anchor...")
    
    # Earth frequencies
    earth_freq = EARTH_FREQUENCY
    schumann_freq = SCHUMANN_FREQUENCY
    
    # Check for previous Malkuth interaction to inform frequency blending
    malkuth_frequency = None
    
    # Look in interaction_history for Malkuth frequency data
    if hasattr(soul_spark, 'interaction_history') and soul_spark.interaction_history:
        for interaction in soul_spark.interaction_history:
            if interaction.get('type') == 'sephirah_layer_formation' and interaction.get('sephirah') == 'malkuth':
                # Found Malkuth interaction record
                logger.debug("Found previous Malkuth interaction data in soul history")
                malkuth_frequency = interaction.get('sephirah_frequency', None)
                break
    
    # Calculate combined earth frequency with natural weighting
    if malkuth_frequency and malkuth_frequency > 0:
        # Use Malkuth resonance experience to improve earth connection
        # Natural system 3-way blending with Fibonacci-based weights
        earth_weight = 5/13    # Fibonacci ratio
        schumann_weight = 3/13  # Fibonacci ratio
        malkuth_weight = 5/13   # Fibonacci ratio
        
        combined_earth_freq = (
            earth_freq * earth_weight +
            schumann_freq * schumann_weight +
            malkuth_frequency * malkuth_weight
        )
        logger.debug(f"Using Malkuth frequency ({malkuth_frequency:.2f}Hz) to enhance Earth anchor")
    else:
        # Default golden ratio weighting without Malkuth data
        earth_weight = PHI / (PHI + 1)      # ~0.618
        schumann_weight = 1 / (PHI + 1)     # ~0.382
        combined_earth_freq = earth_freq * earth_weight + schumann_freq * schumann_weight
    
    # Create position (below - where the brain will be)
    earth_pos = [0.0, 0.0, -100.0]
    
    # Earth resonance is naturally more forgiving than spiritual
    # Base resonance target on earth_resonance if available
    earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
    
    # Non-linear sigmoid curve for resonance target based on existing earth resonance
    resonance_target = 0.5 + 0.3 * (1 / (1 + exp(-10 * (earth_resonance - 0.5))))
    
    # Natural tolerance curve - higher than spiritual anchor
    resonance_tolerance = 0.15 + 0.1 * (1 - earth_resonance)
    
    # Create earth anchor
    earth_anchor = {
        "position": earth_pos,
        "frequency": float(combined_earth_freq),
        "resonance_target": float(resonance_target),
        "resonance_tolerance": float(resonance_tolerance),
        "earth_base_frequency": float(earth_freq),
        "schumann_frequency": float(schumann_freq),
        "malkuth_frequency": float(malkuth_frequency) if malkuth_frequency else None,
        "gaia_connection": float(1.0),  # Perfect Gaia connection for anchor
        "metadata": {
            "creation_timestamp": datetime.now().isoformat(),
            "frequency_principles": "golden_ratio_blend",
            "resonance_principles": "sigmoid_natural_curve"
        }
    }
    
    logger.info(f"Earth Anchor created at frequency {combined_earth_freq:.2f}Hz "
               f"with resonance target {resonance_target:.3f} "
               f"(tolerance: {resonance_tolerance:.3f})")
    
    return earth_anchor


def _form_communication_channels(spiritual_anchor: Dict[str, Any], 
                               earth_anchor: Dict[str, Any],
                               soul_spark: SoulSpark,
                               complexity: float) -> Dict[str, Any]:
    """
    Forms the communication channels between spiritual and earth anchors
    using biomimetic principles for bandwidth and resistance calculation.
    
    Args:
        spiritual_anchor: The spiritual anchor properties
        earth_anchor: The earth anchor properties
        soul_spark: The SoulSpark object for properties and metrics
        complexity: Complexity factor for channel properties
        
    Returns:
        Dict containing the channel properties
    """
    logger.info("LC Step: Forming Communication Channels...")
    
    # Get anchor frequencies
    spiritual_freq = spiritual_anchor["frequency"]
    earth_freq = earth_anchor["frequency"]
    soul_freq = soul_spark.frequency
    
    # Get soul properties for natural bandwidth calculation
    coherence = soul_spark.coherence / MAX_COHERENCE_CU  # Normalize to 0-1
    creator_connection = getattr(soul_spark, 'creator_connection_strength', 0.0)
    earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
    
    # Calculate spiritual channel properties using natural scaling
    # Bandwidth scales with frequency * complexity * golden ratio
    spiritual_bandwidth = spiritual_freq * complexity * (0.5 + 0.5 * PHI/2) * (0.7 + 0.3 * creator_connection)
    
    # Resistance is inversely proportional to creator connection (natural inverse relationship)
    # Lower resistance = better connection
    spiritual_resistance = max(0.1, 0.5 * (1.0 - creator_connection * 0.8))
    
    # Get threshold from anchor
    spiritual_threshold = spiritual_anchor["resonance_target"]
    
    # Calculate earth channel properties using natural scaling
    # Bandwidth scales with frequency * complexity * golden ratio
    earth_bandwidth = earth_freq * complexity * (0.5 + 0.5 * PHI/2) * (0.7 + 0.3 * earth_resonance)
    
    # Resistance is inversely proportional to earth resonance 
    earth_resistance = max(0.1, 0.5 * (1.0 - earth_resonance * 0.8))
    
    # Get threshold from anchor
    earth_threshold = earth_anchor["resonance_target"]
    
    # Calculate channel distance using 3D coordinates
    spiritual_pos = spiritual_anchor["position"]
    earth_pos = earth_anchor["position"]
    distance = np.sqrt(sum((spiritual_pos[i] - earth_pos[i])**2 for i in range(3)))
    
    # Calculate quantum entanglement based on coherence (non-linear natural scaling)
    quantum_entanglement = tanh(coherence * 2.5)  # Hyperbolic tangent gives natural saturation curve
    
    # Calculate shield strength using golden ratio scaling
    em_shield_base = 0.7
    em_shield_bonus = 0.3 * (1.0 - exp(-coherence * 3.0))  # Natural exponential approach
    em_shield_strength = em_shield_base + em_shield_bonus
    
    # Create the channels structure
    channels = {
        "spiritual_channel": {
            "state": "inactive",  # Initially inactive until soul resonates
            "bandwidth_hz": float(spiritual_bandwidth),
            "resistance": float(spiritual_resistance),
            "resonance_threshold": float(spiritual_threshold),
            "frequency": float(spiritual_freq),
            "soul_resonant_range": [soul_freq * 0.9, soul_freq * 1.5],  # Natural range for resonance
            "creation_timestamp": datetime.now().isoformat()
        },
        "earth_channel": {
            "state": "inactive",  # Initially inactive until soul resonates
            "bandwidth_hz": float(earth_bandwidth),
            "resistance": float(earth_resistance),
            "resonance_threshold": float(earth_threshold),
            "frequency": float(earth_freq),
            "soul_resonant_range": [soul_freq * 0.5, soul_freq * 1.1],  # Natural range for resonance
            "creation_timestamp": datetime.now().isoformat()
        },
        "channel_properties": {
            "length": float(distance),
            "electromagnetic_shield_strength": float(em_shield_strength),
            "stability": float(1.0),  # Perfect stability (divine)
            "quantum_entanglement": float(quantum_entanglement),
            "light_speed_factor": float(1.0),  # Instantaneous communication (divine)
            "creator_energy_flow": float(1.0),  # Perfect energy flow (divine)
            "creation_timestamp": datetime.now().isoformat()
        }
    }
    
    logger.info(f"Communication Channels formed: "
               f"Spiritual (BW: {spiritual_bandwidth:.2f}Hz, R: {spiritual_resistance:.3f}), "
               f"Earth (BW: {earth_bandwidth:.2f}Hz, R: {earth_resistance:.3f}), "
               f"Distance: {distance:.1f}, Shield: {em_shield_strength:.3f}, "
               f"QE: {quantum_entanglement:.3f}")
    
    return channels


def _test_soul_channel_resonance(soul_spark: SoulSpark, 
                               channel_type: str,
                               channel: Dict[str, Any], 
                               anchor: Dict[str, Any],
                               max_steps: int) -> Dict[str, Any]:
    """
    Tests if the soul can resonate with a channel by biomimetically adjusting its frequency.
    Uses edge of chaos principles to find the resonance point, mimicking how consciousness
    adjusts frequency during meditation.
    
    Args:
        soul_spark: The SoulSpark object
        channel_type: Type of channel ('spiritual' or 'earth')
        channel: The channel properties
        anchor: The anchor properties
        max_steps: Maximum number of frequency adjustment steps
        
    Returns:
        Dict containing the resonance test results
    """
    logger.info(f"LC Step: Testing Soul-{channel_type.capitalize()} Channel Resonance...")
    
    # Get channel and soul properties
    channel_freq = channel["frequency"]
    resonance_threshold = channel["resonance_threshold"]
    resonance_tolerance = anchor["resonance_tolerance"]
    soul_frequency = getattr(soul_spark, 'frequency', 432.0)
    
    # Set up resonance finding variables
    steps = 0
    resonance_achieved = False
    current_resonance = calculate_resonance(soul_frequency, channel_freq)
    best_resonance = current_resonance
    best_frequency = soul_frequency
    current_frequency = soul_frequency
    
    # Calculate initial difficulty based on resonance gap
    frequency_gap = abs(np.log(soul_frequency) - np.log(channel_freq))
    initial_difficulty = tanh(frequency_gap * 1.5)  # Natural scaling with hyperbolic tangent
    
    # Track resonance path for edge of chaos analysis
    resonance_path = []
    frequency_adjustments = []
    
    # Calculate required resonance target with tolerance
    required_resonance = resonance_threshold - resonance_tolerance
    
    logger.info(f"Starting {channel_type} resonance test: "
               f"Soul: {soul_frequency:.2f}Hz, Channel: {channel_freq:.2f}Hz, "
               f"Initial Resonance: {current_resonance:.4f}, Target: {required_resonance:.4f}")
    
    # Record edge of chaos transitions
    eoc_transitions = 0
    last_distance_to_eoc = abs(current_resonance - 0.618)
    
    # Begin resonance finding process - biomimetic frequency adjustment
    while steps < max_steps and not resonance_achieved:
        steps += 1
        
        # Record current state
        resonance_path.append(float(current_resonance))
        
        # Check if resonance is achieved
        if current_resonance >= required_resonance:
            resonance_achieved = True
            logger.info(f"{channel_type.capitalize()} resonance achieved after {steps} steps: {current_resonance:.4f}")
            break
        
        # Calculate how far we are from edge of chaos (EOC = 0.618)
        # In natural systems, the edge of chaos is often at the golden ratio point
        current_distance_to_eoc = abs(current_resonance - 0.618)
        
        # Detect if we crossed the edge of chaos threshold
        if last_distance_to_eoc > 0.05 and current_distance_to_eoc <= 0.05:
            eoc_transitions += 1
            logger.debug(f"Edge of chaos transition detected at step {steps}")
        
        last_distance_to_eoc = current_distance_to_eoc
        
        # Apply biomimetic frequency adjustment strategy
        # At edge of chaos, make larger adjustments to break through
        # Further from edge, make smaller, more careful adjustments
        if current_distance_to_eoc < 0.1:
            # Near edge of chaos - larger adjustment to break through
            # Natural systems often make "jumps" at critical thresholds
            adjustment_scale = 0.12 * (1.0 - (current_distance_to_eoc / 0.1))
        else:
            # Away from edge - smaller adjustments (careful exploration)
            # Natural systems explore parameters more cautiously away from transitions
            adjustment_scale = 0.05 * (1.0 - current_resonance) * (current_distance_to_eoc / 0.2)
        
        # Calculate frequency adjustment direction
        frequency_ratio = current_frequency / channel_freq
        
        # Direction depends on whether we're above or below target frequency
        # Natural systems tend to approach targets from both directions
        if frequency_ratio > 1.0:
            # Current frequency too high
            adjustment = -adjustment_scale * current_frequency
        else:
            # Current frequency too low
            adjustment = adjustment_scale * current_frequency
        
        # Apply phi-based correction if close to phi ratio
        # Natural systems often seek out golden ratio relationships
        phi_ratio = PHI
        inverse_phi = 1.0 / PHI
        
        phi_distances = [
            abs(frequency_ratio - phi_ratio),
            abs(frequency_ratio - inverse_phi),
            abs(frequency_ratio - phi_ratio**2),
            abs(frequency_ratio - inverse_phi**2)
        ]
        
        min_phi_distance = min(phi_distances)
        if min_phi_distance < 0.1:
            # We're close to a phi relationship, leverage it
            # Find which phi relationship is closest
            phi_patterns = [PHI, 1.0/PHI, PHI**2, 1.0/(PHI**2)]
            closest_phi = phi_patterns[phi_distances.index(min_phi_distance)]
            
            # Target that phi relationship
            target = channel_freq * closest_phi
            phi_adjustment = (target - current_frequency) * 0.2
            
            # Blend normal adjustment with phi-targeting (weighted by proximity)
            phi_weight = 1.0 - (min_phi_distance / 0.1)
            adjustment = adjustment * (1.0 - phi_weight) + phi_adjustment * phi_weight
            logger.debug(f"Applied phi correction with weight {phi_weight:.3f}")
        
        # Apply slight randomization for natural exploration
        # Natural systems have inherent variability in adjustments
        noise_amplitude = 0.02 * abs(adjustment) * (1.0 - resonance_threshold)
        adjustment += np.random.normal(0, noise_amplitude)
        
        # Record adjustment
        frequency_adjustments.append(float(adjustment))
        
        # Apply adjustment
        current_frequency += adjustment
        
        # Apply natural bounds to frequency 
        # Ensure within resonant range for safety
        if "soul_resonant_range" in channel:
            min_safe = channel["soul_resonant_range"][0]
            max_safe = channel["soul_resonant_range"][1]
            current_frequency = max(min_safe, min(max_safe, current_frequency))
        
        # Calculate new resonance
        current_resonance = calculate_resonance(current_frequency, channel_freq)
        
        # Track best resonance
        if current_resonance > best_resonance:
            best_resonance = current_resonance
            best_frequency = current_frequency
    
    # Calculate final difficulty using biomimetic criteria
    normalized_steps = min(1.0, steps / max_steps)
    log_factor = 0.5 * (1.0 + tanh((steps - max_steps/2) / (max_steps/4)))
    effort_factor = normalized_steps * log_factor
    
    # Higher intensity at edge of chaos - natural systems expend more energy at transitions
    eoc_intensity = 0.3 * (eoc_transitions / max(1, steps/10))
    
    # Combine initial gap with effort and transitions for natural difficulty metric
    difficulty = initial_difficulty * 0.4 + effort_factor * 0.4 + eoc_intensity * 0.2
    difficulty = max(0.0, min(1.0, difficulty))
    
    # Create resonance results
    resonance_results = {
        "resonance_achieved": resonance_achieved,
        "steps_taken": steps,
        "max_steps": max_steps,
        "initial_resonance": float(calculate_resonance(soul_frequency, channel_freq)),
        "final_resonance": float(current_resonance),
        "best_resonance": float(best_resonance),
        "initial_frequency": float(soul_frequency),
        "final_frequency": float(current_frequency),
        "best_frequency": float(best_frequency),
        "channel_frequency": float(channel_freq),
        "resonance_difficulty": float(difficulty),
        "resonance_path": resonance_path,
        "frequency_adjustments": frequency_adjustments,
        "edge_of_chaos_transitions": eoc_transitions,
        "creation_timestamp": datetime.now().isoformat()
    }
    
    # Record results in channel
    channel["state"] = "active" if resonance_achieved else "inactive"
    channel["resonance_result"] = resonance_results
    
    logger.info(f"{channel_type.capitalize()} Channel Test Results: "
               f"Achieved: {resonance_achieved}, "
               f"Steps: {steps}/{max_steps}, "
               f"Final Resonance: {current_resonance:.4f}, "
               f"Difficulty: {difficulty:.4f}, "
               f"EOC Transitions: {eoc_transitions}")
    
    return resonance_results

def _record_resonance_metrics(soul_spark: SoulSpark,
                            spiritual_results: Dict[str, Any],
                            earth_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Records metrics about the resonance tests to the soul spark.
    Updates soul properties based on natural principles.
    
    Args:
        soul_spark: The SoulSpark object
        spiritual_results: Results from spiritual channel test
        earth_results: Results from earth channel test
        
    Returns:
        Dict containing combined metrics
    """
    logger.info("LC Step: Recording Resonance Metrics...")
    
    # Extract key results
    spiritual_achieved = spiritual_results["resonance_achieved"]
    earth_achieved = earth_results["resonance_achieved"]
    spiritual_steps = spiritual_results["steps_taken"]
    earth_steps = earth_results["steps_taken"]
    spiritual_difficulty = spiritual_results["resonance_difficulty"]
    earth_difficulty = earth_results["resonance_difficulty"]
    
    # Calculate overall metrics using natural principles
    # Overall success only if both channels activated
    overall_success = spiritual_achieved and earth_achieved
    
    # Average difficulty with golden ratio weighting - natural bias toward earth difficulty
    # Earth is more important for soul's immediate future
    phi_weight_earth = PHI / (PHI + 1)  # ~0.618
    phi_weight_spiritual = 1 / (PHI + 1)  # ~0.382
    avg_difficulty = (
        earth_difficulty * phi_weight_earth + 
        spiritual_difficulty * phi_weight_spiritual
    )
    
    # Total steps with biomimetic analysis
    total_steps = spiritual_steps + earth_steps
    
    # Calculate step efficiency ratio (natural log curve)
    # This measures how efficiently the soul found resonance
    max_possible_steps = spiritual_results["max_steps"] + earth_results["max_steps"]
    if overall_success and max_possible_steps > 0:
        step_efficiency = 1.0 - (total_steps / max_possible_steps)
        # Apply natural log curve to emphasize differences
        step_efficiency = max(0.0, min(1.0, -0.5 * log(0.1 + 0.9 * (1.0 - step_efficiency))))
    else:
        step_efficiency = 0.0
    
    # Create combined metrics
    communication_metrics = {
        "spiritual_resonance_achieved": spiritual_achieved,
        "earth_resonance_achieved": earth_achieved,
        "spiritual_resonance_steps": spiritual_steps,
        "earth_resonance_steps": earth_steps,
        "spiritual_resonance_difficulty": float(spiritual_difficulty),
        "earth_resonance_difficulty": float(earth_difficulty),
        "overall_resonance_success": overall_success,
        "overall_resonance_difficulty": float(avg_difficulty),
        "total_resonance_steps": total_steps,
        "step_efficiency": float(step_efficiency),
        "resonance_timestamp": datetime.now().isoformat(),
        "edge_of_chaos_transitions": {
            "spiritual": spiritual_results.get("edge_of_chaos_transitions", 0),
            "earth": earth_results.get("edge_of_chaos_transitions", 0),
            "total": spiritual_results.get("edge_of_chaos_transitions", 0) + 
                     earth_results.get("edge_of_chaos_transitions", 0)
        }
    }
    
    # Calculate earth resonance for soul property using biomimetic principles
    # This is a measure of how well the soul can communicate with Earth
    if earth_achieved:
        # Successful resonance - calculate based on final resonance
        base_resonance = earth_results["final_resonance"] 
        
        # Apply efficiency bonus (natural systems reward efficient adaptation)
        efficiency_bonus = step_efficiency * 0.2
        
        earth_resonance = min(1.0, base_resonance + efficiency_bonus)
    else:
        # Failed resonance - use best achieved value with penalty
        earth_resonance = earth_results["best_resonance"] * 0.7  # 30% penalty
    
    # Record to soul spark - these become permanent attributes
    setattr(soul_spark, "earth_resonance", float(earth_resonance))
    setattr(soul_spark, "cord_resonance_steps", total_steps)
    setattr(soul_spark, "cord_resonance_difficulty", float(avg_difficulty))
    
    # For tracking resonance capacity without changing the base frequency
    if earth_achieved and spiritual_achieved:
        best_earth_freq = earth_results["best_frequency"]
        best_spiritual_freq = spiritual_results["best_frequency"]
        
        setattr(soul_spark, "earth_resonant_frequency", float(best_earth_freq))
        setattr(soul_spark, "spiritual_resonant_frequency", float(best_spiritual_freq))
    
    logger.info(f"Resonance Metrics Recorded: "
               f"Overall Success: {overall_success}, "
               f"Overall Difficulty: {avg_difficulty:.4f}, "
               f"Total Steps: {total_steps}, "
               f"Earth Resonance: {earth_resonance:.4f}, "
               f"Step Efficiency: {step_efficiency:.4f}")
    
    # Record metrics to tracking system if available
    if METRICS_AVAILABLE:
        metrics.record_metrics('life_cord_resonance', {
            'soul_id': soul_spark.spark_id,
            'spiritual_resonance_achieved': spiritual_achieved,
            'earth_resonance_achieved': earth_achieved,
            'spiritual_resonance_steps': spiritual_steps,
            'earth_resonance_steps': earth_steps,
            'overall_resonance_success': overall_success,
            'overall_resonance_difficulty': avg_difficulty,
            'earth_resonance_value': earth_resonance,
            'step_efficiency': step_efficiency,
            'edge_of_chaos_transitions_total': communication_metrics["edge_of_chaos_transitions"]["total"],
            'timestamp': communication_metrics["resonance_timestamp"]
        })
    
    return communication_metrics


def _finalize_life_cord(soul_spark: SoulSpark, 
                       cord_structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalizes the life cord and sets appropriate flags on the soul spark.
    The divine cord maintains perfect integrity regardless of resonance success.
    
    Args:
        soul_spark: The SoulSpark object
        cord_structure: The complete cord structure
        
    Returns:
        Dict containing finalization metrics
    """
    logger.info("LC Step: Finalizing Life Cord...")
    
    # Extract key information
    communication_metrics = cord_structure.get("communication_metrics", {})
    overall_success = communication_metrics.get("overall_resonance_success", False)
    
    # Cord integrity is always perfect (divine energy)
    cord_integrity = 1.0
    
    # Get earth resonance from communication metrics or use stored value
    earth_resonance = getattr(soul_spark, "earth_resonance", 0.0)
    earth_resonance = max(0.0, min(1.0, earth_resonance))
    
    # Create finalization time
    finalization_time = datetime.now().isoformat()
    
    # Create sigil data for future gateway key
    sigil_data = {
        "type": "life_cord",
        "creation_timestamp": cord_structure.get("creation_timestamp", ""),
        "finalization_timestamp": finalization_time,
        "soul_id": soul_spark.spark_id,
        "surface_data": {
            "resonance_steps": communication_metrics.get("total_resonance_steps", 0),
            "resonance_difficulty": float(communication_metrics.get("overall_resonance_difficulty", 0.0)),
            "earth_resonance": float(earth_resonance),
            "step_efficiency": float(communication_metrics.get("step_efficiency", 0.0))
        },
        "hidden_data": {
            "spiritual_channel": cord_structure.get("channels", {}).get("spiritual_channel", {}),
            "earth_channel": cord_structure.get("channels", {}).get("earth_channel", {}),
            "spiritual_anchor": cord_structure.get("anchors", {}).get("spiritual", {}),
            "earth_anchor": cord_structure.get("anchors", {}).get("earth", {})
        },
        "edge_of_chaos_data": communication_metrics.get("edge_of_chaos_transitions", {})
    }
    
    # Store sigil data in soul_spark for later retrieval by sigil creator
    if not hasattr(soul_spark, "sigil_data"):
        setattr(soul_spark, "sigil_data", {})
    
    soul_spark.sigil_data["life_cord"] = sigil_data
    
    # Set final flags and properties on soul spark
    setattr(soul_spark, "cord_integrity", float(cord_integrity))
    setattr(soul_spark, "earth_resonance", float(earth_resonance))
    setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, True)
    setattr(soul_spark, FLAG_READY_FOR_EARTH, True)
    
    # Update last modified timestamp
    setattr(soul_spark, "last_modified", finalization_time)
    
    # Store simplified final cord structure in soul_spark
    final_cord = {
        "integrity": float(cord_integrity),
        "creation_timestamp": cord_structure.get("creation_timestamp", ""),
        "finalization_timestamp": finalization_time,
        "earth_resonance": float(earth_resonance),
        "spiritual_resonance": float(communication_metrics.get("spiritual_resonance_difficulty", 0.0)),
        "spiritual_channel_active": communication_metrics.get("spiritual_resonance_achieved", False),
        "earth_channel_active": communication_metrics.get("earth_resonance_achieved", False),
        "resonance_steps": communication_metrics.get("total_resonance_steps", 0),
        "resonance_difficulty": float(communication_metrics.get("overall_resonance_difficulty", 0.0)),
        "step_efficiency": float(communication_metrics.get("step_efficiency", 0.0)),
        "earth_anchor_frequency": cord_structure.get("anchors", {}).get("earth", {}).get("frequency", 0.0),
        "spiritual_anchor_frequency": cord_structure.get("anchors", {}).get("spiritual", {}).get("frequency", 0.0)
    }
    
    # Set the complete cord structure in soul_spark
    setattr(soul_spark, "life_cord", final_cord)
    
    # Record metrics
    if METRICS_AVAILABLE:
        metrics.record_metrics('life_cord_finalization', {
            'soul_id': soul_spark.spark_id,
            'cord_integrity': cord_integrity,
            'earth_resonance': earth_resonance,
            'overall_success': overall_success,
            'spiritual_channel_active': communication_metrics.get("spiritual_resonance_achieved", False),
            'earth_channel_active': communication_metrics.get("earth_resonance_achieved", False),
            'timestamp': finalization_time
        })
    
    # Create finalization metrics
    finalization_metrics = {
        "cord_integrity": float(cord_integrity),
        "earth_resonance": float(earth_resonance),
        "spiritual_resonance": float(communication_metrics.get("spiritual_resonance_difficulty", 0.0)),
        "resonance_steps": communication_metrics.get("total_resonance_steps", 0),
        "resonance_difficulty": float(communication_metrics.get("overall_resonance_difficulty", 0.0)),
        "overall_success": overall_success,
        "step_efficiency": float(communication_metrics.get("step_efficiency", 0.0)),
        "finalization_timestamp": finalization_time
    }
    
    logger.info(f"Life Cord Finalized: "
               f"Integrity: {cord_integrity:.2f} (Divine), "
               f"Earth Resonance: {earth_resonance:.4f}, "
               f"Success: {overall_success}, "
               f"Efficiency: {communication_metrics.get('step_efficiency', 0.0):.4f}")
    
    # Add memory echo if soul has this capacity
    if hasattr(soul_spark, 'add_memory_echo'):
        success_str = "successful" if overall_success else "partial"
        soul_spark.add_memory_echo(
            f"Life cord formation {success_str}. "
            f"Earth resonance: {earth_resonance:.3f}. "
            f"Cord integrity: {cord_integrity:.2f} (perfect divine energy)."
        )
    
    return finalization_metrics


# --- Main Function ---

def form_life_cord(soul_spark: SoulSpark, intensity: float = 0.7, 
                 complexity: float = 0.5) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Forms the divine life cord connecting the soul to Earth and Creator.
    Creates perfect divine cord with two anchors and channels.
    The soul must learn to resonate with the anchors to establish communication,
    adapting its frequency temporarily without changing its base nature.
    
    Args:
        soul_spark: The SoulSpark object
        intensity: Factor affecting resonance finding process (0.1-1.0)
        complexity: Factor affecting channel structure complexity (0.1-1.0)
        
    Returns:
        Tuple of (modified soul_spark, process_metrics)
    """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("soul_spark must be a SoulSpark instance.")
        
    if not isinstance(intensity, (int, float)) or not (0.1 <= intensity <= 1.0):
        raise ValueError(f"Intensity must be between 0.1 and 1.0, got {intensity}")
        
    if not isinstance(complexity, (int, float)) or not (0.1 <= complexity <= 1.0):
        raise ValueError(f"Complexity must be between 0.1 and 1.0, got {complexity}")
    
    spark_id = getattr(soul_spark, "spark_id", "unknown_spark")
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
        # --- Check Prerequisites ---
        logger.info("Life Cord Step 1: Checking Prerequisites...")
        _ensure_soul_properties(soul_spark)
        _check_prerequisites(soul_spark)
        
        # Energy check before proceeding - hard fail
        initial_energy = soul_spark.energy
        if initial_energy < CORD_ACTIVATION_ENERGY_COST:
            raise ValueError(f"Insufficient energy ({initial_energy:.1f} SEU) for cord formation. Required: {CORD_ACTIVATION_ENERGY_COST} SEU")
        
        # Apply energy cost
        soul_spark.energy -= CORD_ACTIVATION_ENERGY_COST
        
        # --- Create Divine Cord ---
        logger.info("Life Cord Step 2: Creating Divine Cord...")
        cord_structure = _establish_divine_cord()
        process_metrics["steps"]["divine_cord"] = {
            "divine_properties": cord_structure["divine_properties"]
        }
        
        # --- Create Anchors ---
        logger.info("Life Cord Step 3: Creating Anchors...")
        spiritual_anchor = _create_spiritual_anchor(soul_spark)
        earth_anchor = _create_earth_anchor(soul_spark)
        
        # Add anchors to cord structure
        cord_structure["anchors"] = {
            "spiritual": spiritual_anchor,
            "earth": earth_anchor
        }
        
        process_metrics["steps"]["anchors"] = {
            "spiritual_anchor": {k: v for k, v in spiritual_anchor.items() if k != "position"},
            "earth_anchor": {k: v for k, v in earth_anchor.items() if k != "position"}
        }
        
        # --- Form Communication Channels ---
        logger.info("Life Cord Step 4: Forming Communication Channels...")
        channels = _form_communication_channels(
            spiritual_anchor, 
            earth_anchor,
            soul_spark,
            complexity
        )
        
        # Add channels to cord structure
        cord_structure["channels"] = channels
        
        process_metrics["steps"]["channels"] = {
            "spiritual_channel": {k: v for k, v in channels["spiritual_channel"].items() if k != "resonance_result"},
            "earth_channel": {k: v for k, v in channels["earth_channel"].items() if k != "resonance_result"},
            "channel_properties": channels["channel_properties"]
        }
        
        # --- Test Soul-Channel Resonance ---
        logger.info("Life Cord Step 5: Testing Soul-Channel Resonance...")
        
        # Calculate maximum resonance steps based on intensity
        # Use natural scaling with Fibonacci sequence
        fib_basis = int(13 * intensity)  # 13 is a Fibonacci number
        max_steps = fib_basis * 8  # Natural multiple
        
        # Test spiritual channel
        spiritual_results = _test_soul_channel_resonance(
            soul_spark,
            "spiritual",
            channels["spiritual_channel"],
            spiritual_anchor,
            max_steps
        )
        
        # Test earth channel
        earth_results = _test_soul_channel_resonance(
            soul_spark,
            "earth",
            channels["earth_channel"],
            earth_anchor,
            max_steps
        )
        
        process_metrics["steps"]["resonance_tests"] = {
            "spiritual_results": {k: v for k, v in spiritual_results.items() 
                               if k not in ["resonance_path", "frequency_adjustments"]},
            "earth_results": {k: v for k, v in earth_results.items() 
                           if k not in ["resonance_path", "frequency_adjustments"]}
        }
        
        # --- Record Communication Metrics ---
        logger.info("Life Cord Step 6: Recording Communication Metrics...")
        communication_metrics = _record_resonance_metrics(
            soul_spark,
            spiritual_results,
            earth_results
        )
        
        # Add communication metrics to cord structure
        cord_structure["communication_metrics"] = communication_metrics
        
        process_metrics["steps"]["communication_metrics"] = communication_metrics
        
        # --- Finalize Life Cord ---
        logger.info("Life Cord Step 7: Finalizing Life Cord...")
        finalization = _finalize_life_cord(soul_spark, cord_structure)
        process_metrics["steps"]["finalization"] = finalization
        
        # --- Generate Sound If Available ---
        if SOUND_MODULES_AVAILABLE:
            try:
                logger.info("Generating life cord formation sound...")
                
                # Create sound generator
                sound_gen = SoundGenerator(sample_rate=SAMPLE_RATE)
                
                # Create chord based on spiritual and earth frequencies
                spiritual_freq = spiritual_anchor["frequency"]
                earth_freq = earth_anchor["frequency"]
                
                # Create harmonic structure with phi ratios - natural harmony
                frequencies = [
                    spiritual_freq,
                    earth_freq,
                    spiritual_freq * PHI,
                    earth_freq / PHI,
                    spiritual_freq * 2.0,
                    earth_freq * 2.0
                ]
                
                # Amplitudes based on resonance success - natural weighting
                spiritual_amp = 0.8 if spiritual_results["resonance_achieved"] else 0.4
                earth_amp = 0.8 if earth_results["resonance_achieved"] else 0.4
                
                # Golden ratio falloff for harmonics - natural harmonic distribution
                amplitudes = [
                    spiritual_amp,
                    earth_amp,
                    spiritual_amp / PHI,
                    earth_amp / PHI,
                    spiritual_amp / (PHI**2),
                    earth_amp / (PHI**2)
                ]
                
                # Generate the sacred chord
                sound_data = sound_gen.generate_sacred_chord(
                    frequencies, 
                    amplitudes, 
                    5.0
                )
                
                # Save the sound
                timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
                sound_file = f"life_cord_{spark_id}_{timestamp_str}.wav"
                file_path = sound_gen.save_sound(
                    sound_data, 
                    sound_file, 
                    f"Life Cord Formation - Soul {spark_id}"
                )
                
                process_metrics["sound_file"] = sound_file
                process_metrics["sound_path"] = file_path
                
                logger.info(f"Life cord sound generated: {file_path}")
                
            except Exception as sound_err:
                logger.warning(f"Failed to generate life cord sound: {sound_err}")
        
        # --- Complete Process Metrics ---
        end_time = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time)
        start_time_dt = datetime.fromisoformat(start_time)
        duration_seconds = (end_time_dt - start_time_dt).total_seconds()
        
        process_metrics["end_time"] = end_time
        process_metrics["duration_seconds"] = duration_seconds
        process_metrics["success"] = True
        process_metrics["cord_integrity"] = 1.0  # Divine perfection
        process_metrics["earth_resonance"] = float(getattr(soul_spark, "earth_resonance", 0.0))
        process_metrics["energy_consumed"] = float(CORD_ACTIVATION_ENERGY_COST)
        process_metrics["spiritual_channel_active"] = spiritual_results["resonance_achieved"]
        process_metrics["earth_channel_active"] = earth_results["resonance_achieved"]
        
        # Log summary with natural system principles emphasized
        logger.info(f"--- Life Cord Formation Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Final Metrics: Divine Integrity=1.0 (perfect), "
                   f"Earth Resonance={process_metrics['earth_resonance']:.4f}, "
                   f"Spiritual Channel: {'Active' if spiritual_results['resonance_achieved'] else 'Inactive'}, "
                   f"Earth Channel: {'Active' if earth_results['resonance_achieved'] else 'Inactive'}, "
                   f"Total Steps: {communication_metrics['total_resonance_steps']}, "
                   f"Efficiency: {communication_metrics.get('step_efficiency', 0.0):.4f}, "
                   f"Duration: {duration_seconds:.2f}s")
        
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
        process_metrics["error_type"] = type(e).__name__
        process_metrics["end_time"] = datetime.now().isoformat()
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('life_cord_summary', process_metrics)
            
        raise  # Re-raise original error for hard fail
        
    except Exception as e:
        logger.critical(f"Unexpected error during life cord formation for {spark_id}: {e}", exc_info=True)
        # Set failure flag
        setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False)
        
        # Record failure metrics
        process_metrics["success"] = False
        process_metrics["error"] = str(e)
        process_metrics["error_type"] = "Unexpected"
        process_metrics["end_time"] = datetime.now().isoformat()
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('life_cord_summary', process_metrics)
            
        raise RuntimeError(f"Unexpected life cord formation failure: {e}") from e










# """
# Life Cord Formation Functions (Refactored V5.0.0 - Divine Channel Implementation)

# Creates an indestructible divine life cord with two anchors (earth and spiritual)
# and communication channels. The soul must adjust its resonance to match the anchors
# to establish communication. Records the difficulty and steps required to establish
# connection using edge of chaos principles.

# The life cord itself is perfect divine energy and does not change based on soul properties.
# Instead, the soul must learn to resonate with the anchors to open the channels.
# """

# import logging
# import numpy as np
# import os
# import sys
# from datetime import datetime
# import time
# import uuid
# from constants.constants import *
# from typing import Dict, List, Any, Tuple, Optional
# from math import pi as PI, sqrt, exp, sin, cos, tanh


# # --- Logging ---
# logger = logging.getLogger(__name__)

# # --- Dependency Imports ---
# try:
#     from stage_1.soul_spark.soul_spark import SoulSpark
#     # Import resonance calculation
#     from .creator_entanglement import calculate_resonance
# except ImportError as e:
#     logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}.")
#     raise ImportError(f"Core dependencies missing: {e}") from e

# # --- Sound Module Imports ---
# try:
#     from sound.sound_generator import SoundGenerator
#     from sound.sounds_of_universe import UniverseSounds
#     SOUND_MODULES_AVAILABLE = True
# except ImportError:
#     logger.warning("Sound modules not available. Life cord formation will use simulated sound.")
#     SOUND_MODULES_AVAILABLE = False

# # --- Metrics Tracking ---
# try:
#     import metrics_tracking as metrics
#     METRICS_AVAILABLE = True
# except ImportError:
#     logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
#     METRICS_AVAILABLE = False
#     class MetricsPlaceholder:
#         def record_metrics(self, *args, **kwargs): pass
#     metrics = MetricsPlaceholder()

# # --- Helper Functions ---

# def _check_prerequisites(soul_spark: SoulSpark) -> bool:
#     """ Checks prerequisites using SU/CU thresholds. Raises ValueError on failure. """
#     logger.debug(f"Checking life cord prerequisites for soul {soul_spark.spark_id}...")
#     if not isinstance(soul_spark, SoulSpark):
#         raise TypeError("Invalid SoulSpark object.")

#     # 1. Stage Completion Check
#     if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False):
#         msg = f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_LIFE_CORD}."
#         logger.error(msg); raise ValueError(msg)

#     # 2. Minimum Stability and Coherence (Absolute SU/CU)
#     stability_su = getattr(soul_spark, 'stability', -1.0)
#     coherence_cu = getattr(soul_spark, 'coherence', -1.0)
#     if stability_su < 0 or coherence_cu < 0:
#         msg = "Prerequisite failed: Soul missing stability or coherence attributes."
#         logger.error(msg); raise AttributeError(msg)

#     if stability_su < CORD_STABILITY_THRESHOLD_SU:
#         msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {CORD_STABILITY_THRESHOLD_SU} SU."
#         logger.error(msg); raise ValueError(msg)
#     if coherence_cu < CORD_COHERENCE_THRESHOLD_CU:
#         msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {CORD_COHERENCE_THRESHOLD_CU} CU."
#         logger.error(msg); raise ValueError(msg)

#     # 3. Energy Check (Done in main function before cost)

#     if getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False):
#         logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_CORD_FORMATION_COMPLETE}. Re-running.")

#     logger.debug("Life cord prerequisites met.")
#     return True

# def _ensure_soul_properties(soul_spark: SoulSpark):
#     """ Ensure soul has necessary properties. Raises error if missing/invalid. """
#     logger.debug(f"Ensuring properties for life cord formation (Soul {soul_spark.spark_id})...")
#     required = ['frequency', 'stability', 'coherence', 'position', 'field_radius',
#                 'field_strength', 'creator_connection_strength', 'energy', 'layers']
#     if not all(hasattr(soul_spark, attr) for attr in required):
#         missing = [attr for attr in required if not hasattr(soul_spark, attr)]
#         raise AttributeError(f"SoulSpark missing essential attributes for Life Cord: {missing}")

#     if not hasattr(soul_spark, 'life_cord'): setattr(soul_spark, 'life_cord', {})
#     if not hasattr(soul_spark, 'cord_integrity'): setattr(soul_spark, 'cord_integrity', 0.0)

#     if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
#     pos = getattr(soul_spark, 'position')
#     if not isinstance(pos, list) or len(pos)!=3: raise ValueError(f"Invalid position: {pos}")
#     if soul_spark.energy < CORD_ACTIVATION_ENERGY_COST:
#         raise ValueError(f"Insufficient energy ({soul_spark.energy:.1f} SEU) for cord activation cost ({CORD_ACTIVATION_ENERGY_COST} SEU).")

#     # Check for layers presence
#     if not soul_spark.layers or len(soul_spark.layers) < 2:
#         raise ValueError("Soul must have at least 2 aura layers for life cord formation.")

#     logger.debug("Soul properties ensured for Life Cord.")

# def _establish_divine_cord() -> Dict[str, Any]:
#     """
#     Creates the divine cord structure with perfect, indestructible properties.
#     The divine cord is composed of pure creator energy.
    
#     Returns:
#         Dict containing the divine cord properties
#     """
#     logger.info("LC Step: Establishing Divine Cord...")
    
#     # Create timestamp for cord creation
#     creation_timestamp = datetime.now().isoformat()
    
#     # Create the divine cord with perfect properties
#     divine_cord = {
#         "divine_properties": {
#             "integrity": 1.0,               # Perfect, indestructible
#             "stability": 1.0,               # Perfect stability
#             "elasticity": 1.0,              # Perfect elasticity
#             "resonance": 1.0,               # Perfect resonance
#             "creation_timestamp": creation_timestamp,
#             "divine_energy_purity": 1.0,    # Pure creator energy
#             "divine_energy_level": 1.0      # Full divine energy
#         },
#         "creation_timestamp": creation_timestamp
#     }
    
#     logger.info("Divine Cord established with perfect properties.")
#     return divine_cord

# def _create_spiritual_anchor(soul_spark: SoulSpark) -> Dict[str, Any]:
#     """
#     Creates the spiritual anchor based on soul properties.
#     This anchor connects to the spiritual aspect of the soul.
    
#     Args:
#         soul_spark: The SoulSpark object
        
#     Returns:
#         Dict containing the spiritual anchor properties
#     """
#     logger.info("LC Step: Creating Spiritual Anchor...")
    
#     # Get soul properties
#     soul_frequency = getattr(soul_spark, 'frequency', 432.0)
#     creator_connection = getattr(soul_spark, 'creator_connection_strength', 0.0)
#     soul_coherence = getattr(soul_spark, 'coherence', 0.0) / MAX_COHERENCE_CU
#     soul_position = getattr(soul_spark, 'position', [0, 0, 0])
    
#     # Calculate spiritual anchor frequency
#     # Based on soul's frequency with slight elevation toward creator frequency
#     creator_freq = 528.0  # Divine frequency
#     spiritual_freq = soul_frequency * (1.0 - creator_connection * 0.3) + creator_freq * (creator_connection * 0.3)
    
#     # Calculate position (above soul)
#     spiritual_pos = [float(p) for p in soul_position]
#     spiritual_pos[2] += 10.0  # Place above soul
    
#     # Calculate resonance target (how closely the soul must match to connect)
#     # Based on creator connection strength - stronger connection = more precise matching needed
#     base_tolerance = 0.2
#     resonance_target = max(0.7, 0.7 + 0.2 * creator_connection)
#     resonance_tolerance = max(0.05, base_tolerance * (1.0 - creator_connection * 0.5))
    
#     # Create spiritual anchor
#     spiritual_anchor = {
#         "position": spiritual_pos,
#         "frequency": float(spiritual_freq),
#         "resonance_target": float(resonance_target),
#         "resonance_tolerance": float(resonance_tolerance),
#         "creator_connection": float(creator_connection),
#         "soul_frequency_base": float(soul_frequency),
#         "creator_frequency": float(creator_freq)
#     }
    
#     logger.info(f"Spiritual Anchor created at frequency {spiritual_freq:.2f}Hz "
#                f"with resonance target {resonance_target:.3f} "
#                f"(tolerance: {resonance_tolerance:.3f})")
    
#     return spiritual_anchor

# def _create_earth_anchor() -> Dict[str, Any]:
#     """
#     Creates the earth anchor based on Earth/Gaia frequencies.
#     This anchor connects to the physical realm.
    
#     Returns:
#         Dict containing the earth anchor properties
#     """
#     logger.info("LC Step: Creating Earth Anchor...")
    
#     # Earth frequencies
#     earth_freq = EARTH_FREQUENCY
#     schumann_freq = SCHUMANN_FREQUENCY
    
#     # Calculate combined earth frequency with Schumann resonance influence
#     combined_earth_freq = earth_freq * 0.7 + schumann_freq * 0.3
    
#     # Create position (below - where the brain will be)
#     earth_pos = [0.0, 0.0, -100.0]
    
#     # Earth resonance is more forgiving than spiritual
#     resonance_target = 0.6
#     resonance_tolerance = 0.2
    
#     # Create earth anchor
#     earth_anchor = {
#         "position": earth_pos,
#         "frequency": float(combined_earth_freq),
#         "resonance_target": float(resonance_target),
#         "resonance_tolerance": float(resonance_tolerance),
#         "earth_base_frequency": float(earth_freq),
#         "schumann_frequency": float(schumann_freq),
#         "gaia_connection": float(1.0)  # Perfect Gaia connection for anchor
#     }
    
#     logger.info(f"Earth Anchor created at frequency {combined_earth_freq:.2f}Hz "
#                f"with resonance target {resonance_target:.3f} "
#                f"(tolerance: {resonance_tolerance:.3f})")
    
#     return earth_anchor

# def _form_communication_channels(spiritual_anchor: Dict[str, Any], 
#                               earth_anchor: Dict[str, Any],
#                               complexity: float) -> Dict[str, Any]:
#     """
#     Forms the communication channels between spiritual and earth anchors.
    
#     Args:
#         spiritual_anchor: The spiritual anchor properties
#         earth_anchor: The earth anchor properties
#         complexity: Complexity factor for channel properties
        
#     Returns:
#         Dict containing the channel properties
#     """
#     logger.info("LC Step: Forming Communication Channels...")
    
#     # Get anchor frequencies
#     spiritual_freq = spiritual_anchor["frequency"]
#     earth_freq = earth_anchor["frequency"]
    
#     # Calculate spiritual channel properties
#     spiritual_bandwidth = spiritual_freq * 0.2 * (0.7 + 0.3 * complexity)
#     spiritual_resistance = max(0.1, 0.4 - 0.2 * complexity)
#     spiritual_threshold = spiritual_anchor["resonance_target"]
    
#     # Calculate earth channel properties
#     earth_bandwidth = earth_freq * 0.3 * (0.7 + 0.3 * complexity)
#     earth_resistance = max(0.2, 0.5 - 0.2 * complexity)
#     earth_threshold = earth_anchor["resonance_target"]
    
#     # Calculate channel distance
#     spiritual_pos = spiritual_anchor["position"]
#     earth_pos = earth_anchor["position"]
#     distance = np.sqrt(sum((spiritual_pos[i] - earth_pos[i])**2 for i in range(3)))
    
#     # Create the channels structure
#     channels = {
#         "spiritual_channel": {
#             "state": "inactive",  # Initially inactive until soul resonates
#             "bandwidth": float(spiritual_bandwidth),
#             "resistance": float(spiritual_resistance),
#             "resonance_threshold": float(spiritual_threshold),
#             "frequency": float(spiritual_freq)
#         },
#         "earth_channel": {
#             "state": "inactive",  # Initially inactive until soul resonates
#             "bandwidth": float(earth_bandwidth),
#             "resistance": float(earth_resistance),
#             "resonance_threshold": float(earth_threshold),
#             "frequency": float(earth_freq)
#         },
#         "channel_properties": {
#             "length": float(distance),
#             "electromagnetic_shield_strength": float(0.9),  # Strong shield
#             "stability": float(1.0),  # Perfect stability
#             "quantum_entanglement": float(1.0)  # Perfect entanglement
#         }
#     }
    
#     logger.info(f"Communication Channels formed: "
#                f"Spiritual (BW: {spiritual_bandwidth:.2f}Hz, R: {spiritual_resistance:.3f}), "
#                f"Earth (BW: {earth_bandwidth:.2f}Hz, R: {earth_resistance:.3f}), "
#                f"Distance: {distance:.1f}, Shield: 0.9")
    
#     return channels

# def _test_soul_channel_resonance(soul_spark: SoulSpark, 
#                                channel_type: str,
#                                channel: Dict[str, Any], 
#                                anchor: Dict[str, Any],
#                                max_steps: int) -> Dict[str, Any]:
#     """
#     Tests if the soul can resonate with a channel by adjusting its frequency.
#     Uses edge of chaos principles to find the resonance point.
    
#     Args:
#         soul_spark: The SoulSpark object
#         channel_type: Type of channel ('spiritual' or 'earth')
#         channel: The channel properties
#         anchor: The anchor properties
#         max_steps: Maximum number of steps to attempt resonance
        
#     Returns:
#         Dict containing the resonance test results
#     """
#     logger.info(f"LC Step: Testing Soul-{channel_type.capitalize()} Channel Resonance...")
    
#     # Get channel and soul properties
#     channel_freq = channel["frequency"]
#     resonance_threshold = channel["resonance_threshold"]
#     resonance_tolerance = anchor["resonance_tolerance"]
#     soul_frequency = getattr(soul_spark, 'frequency', 432.0)
    
#     # Set up resonance finding variables
#     steps = 0
#     resonance_achieved = False
#     current_resonance = calculate_resonance(soul_frequency, channel_freq)
#     best_resonance = current_resonance
#     best_frequency = soul_frequency
#     current_frequency = soul_frequency
    
#     # Calculate initial difficulty based on resonance gap
#     frequency_gap = abs(soul_frequency - channel_freq) / max(soul_frequency, channel_freq)
#     initial_difficulty = frequency_gap
    
#     # Track resonance path for edge of chaos analysis
#     resonance_path = []
#     frequency_adjustments = []
    
#     # Calculate required resonance target with tolerance
#     required_resonance = resonance_threshold - resonance_tolerance
    
#     logger.info(f"Starting {channel_type} resonance test: "
#                f"Soul: {soul_frequency:.2f}Hz, Channel: {channel_freq:.2f}Hz, "
#                f"Initial Resonance: {current_resonance:.4f}, Target: {required_resonance:.4f}")
    
#     # Begin resonance finding process
#     while steps < max_steps and not resonance_achieved:
#         steps += 1
        
#         # Record current state
#         resonance_path.append(float(current_resonance))
        
#         # Check if resonance is achieved
#         if current_resonance >= required_resonance:
#             resonance_achieved = True
#             logger.info(f"{channel_type.capitalize()} resonance achieved after {steps} steps: {current_resonance:.4f}")
#             break
        
#         # Calculate how far we are from edge of chaos (EOC = 0.618)
#         eoc_distance = abs(current_resonance - 0.618)
        
#         # At edge of chaos, make larger adjustments
#         # Further from edge, make smaller, more careful adjustments
#         if eoc_distance < 0.1:
#             # Near edge of chaos - larger adjustment to break through
#             adjustment_scale = 0.1
#         else:
#             # Away from edge - smaller adjustments
#             adjustment_scale = 0.05 * (1.0 - current_resonance)
        
#         # Calculate frequency adjustment direction
#         frequency_ratio = current_frequency / channel_freq
        
#         if frequency_ratio > 1.0:
#             # Current frequency too high
#             adjustment = -adjustment_scale * current_frequency
#         else:
#             # Current frequency too low
#             adjustment = adjustment_scale * current_frequency
        
#         # Apply phi-based correction if close to phi ratio
#         phi_ratio = PHI / frequency_ratio if frequency_ratio < 1.0 else frequency_ratio / PHI
#         if abs(phi_ratio - 1.0) < 0.1:
#             # We're close to a phi relationship, leverage it
#             if frequency_ratio < 1.0:
#                 target = channel_freq / PHI
#             else:
#                 target = channel_freq * PHI
            
#             adjustment = (target - current_frequency) * 0.2
        
#         # Record adjustment
#         frequency_adjustments.append(float(adjustment))
        
#         # Apply adjustment
#         current_frequency += adjustment
        
#         # Calculate new resonance
#         current_resonance = calculate_resonance(current_frequency, channel_freq)
        
#         # Track best resonance
#         if current_resonance > best_resonance:
#             best_resonance = current_resonance
#             best_frequency = current_frequency
    
#     # Calculate final difficulty
#     normalized_steps = min(1.0, steps / max_steps)
#     difficulty = initial_difficulty * (0.3 + 0.7 * normalized_steps)
    
#     # Create resonance results
#     resonance_results = {
#         "resonance_achieved": resonance_achieved,
#         "steps_taken": steps,
#         "max_steps": max_steps,
#         "initial_resonance": float(calculate_resonance(soul_frequency, channel_freq)),
#         "final_resonance": float(current_resonance),
#         "best_resonance": float(best_resonance),
#         "initial_frequency": float(soul_frequency),
#         "final_frequency": float(current_frequency),
#         "best_frequency": float(best_frequency),
#         "channel_frequency": float(channel_freq),
#         "resonance_difficulty": float(difficulty),
#         "resonance_path": resonance_path,
#         "frequency_adjustments": frequency_adjustments,
#         "eoc_transitions": sum(1 for i in range(1, len(resonance_path)) 
#                              if abs(resonance_path[i-1] - 0.618) > 0.05 and 
#                              abs(resonance_path[i] - 0.618) < 0.05)
#     }
    
#     # Record results in channel
#     channel["state"] = "active" if resonance_achieved else "inactive"
#     channel["resonance_result"] = resonance_results
    
#     logger.info(f"{channel_type.capitalize()} Channel Test Results: "
#                f"Achieved: {resonance_achieved}, "
#                f"Steps: {steps}/{max_steps}, "
#                f"Final Resonance: {current_resonance:.4f}, "
#                f"Difficulty: {difficulty:.4f}")
    
#     return resonance_results

# def _record_resonance_metrics(soul_spark: SoulSpark,
#                            spiritual_results: Dict[str, Any],
#                            earth_results: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Records metrics about the resonance tests to the soul spark.
    
#     Args:
#         soul_spark: The SoulSpark object
#         spiritual_results: Results from spiritual channel test
#         earth_results: Results from earth channel test
        
#     Returns:
#         Dict containing combined metrics
#     """
#     logger.info("LC Step: Recording Resonance Metrics...")
    
#     # Extract key results
#     spiritual_achieved = spiritual_results["resonance_achieved"]
#     earth_achieved = earth_results["resonance_achieved"]
#     spiritual_steps = spiritual_results["steps_taken"]
#     earth_steps = earth_results["steps_taken"]
#     spiritual_difficulty = spiritual_results["resonance_difficulty"]
#     earth_difficulty = earth_results["resonance_difficulty"]
    
#     # Calculate overall metrics
#     overall_success = spiritual_achieved and earth_achieved
#     avg_difficulty = (spiritual_difficulty + earth_difficulty) / 2
#     total_steps = spiritual_steps + earth_steps
    
#     # Create combined metrics
#     communication_metrics = {
#         "spiritual_resonance_achieved": spiritual_achieved,
#         "earth_resonance_achieved": earth_achieved,
#         "spiritual_resonance_steps": spiritual_steps,
#         "earth_resonance_steps": earth_steps,
#         "spiritual_resonance_difficulty": float(spiritual_difficulty),
#         "earth_resonance_difficulty": float(earth_difficulty),
#         "overall_resonance_success": overall_success,
#         "overall_resonance_difficulty": float(avg_difficulty),
#         "total_resonance_steps": total_steps,
#         "resonance_timestamp": datetime.now().isoformat()
#     }
    
#     # Calculate earth resonance for soul property
#     # This is a measure of how well the soul can communicate with Earth
#     earth_resonance = earth_results["best_resonance"] if earth_achieved else earth_results["final_resonance"]
    
#     # Record to soul spark
#     setattr(soul_spark, "earth_resonance", float(earth_resonance))
#     setattr(soul_spark, "cord_resonance_steps", total_steps)
#     setattr(soul_spark, "cord_resonance_difficulty", float(avg_difficulty))
    
#     logger.info(f"Resonance Metrics Recorded: "
#                f"Overall Success: {overall_success}, "
#                f"Overall Difficulty: {avg_difficulty:.4f}, "
#                f"Total Steps: {total_steps}, "
#                f"Earth Resonance: {earth_resonance:.4f}")
    
#     return communication_metrics

# def _finalize_life_cord(soul_spark: SoulSpark, 
#                       cord_structure: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Finalizes the life cord and sets appropriate flags on the soul spark.
    
#     Args:
#         soul_spark: The SoulSpark object
#         cord_structure: The complete cord structure
        
#     Returns:
#         Dict containing finalization metrics
#     """
#     logger.info("LC Step: Finalizing Life Cord...")
    
#     # Extract key information
#     communication_metrics = cord_structure.get("communication_metrics", {})
#     overall_success = communication_metrics.get("overall_resonance_success", False)
    
#     # Set cord integrity - always perfect as it's divine energy
#     cord_integrity = 1.0
    
#     # Get earth resonance from communication metrics or use stored value
#     earth_resonance = getattr(soul_spark, "earth_resonance", 0.0)
    
#     # Make sure it's within valid range
#     earth_resonance = max(0.0, min(1.0, earth_resonance))
    
#     # Create finalization time
#     finalization_time = datetime.now().isoformat()
    
#     # Create sigil data for future gateway key
#     sigil_data = {
#         "type": "life_cord",
#         "creation_timestamp": cord_structure.get("creation_timestamp", ""),
#         "finalization_timestamp": finalization_time,
#         "soul_id": soul_spark.spark_id,
#         "surface_data": {
#             "resonance_steps": communication_metrics.get("total_resonance_steps", 0),
#             "resonance_difficulty": float(communication_metrics.get("overall_resonance_difficulty", 0.0)),
#             "earth_resonance": float(earth_resonance)
#         },
#         "hidden_data": {
#             "spiritual_channel": cord_structure.get("channels", {}).get("spiritual_channel", {}),
#             "earth_channel": cord_structure.get("channels", {}).get("earth_channel", {}),
#             "spiritual_anchor": cord_structure.get("anchors", {}).get("spiritual", {}),
#             "earth_anchor": cord_structure.get("anchors", {}).get("earth", {})
#         }
#     }
    
#     # Store sigil data in soul_spark for later retrieval by sigil creator
#     if not hasattr(soul_spark, "sigil_data"):
#         setattr(soul_spark, "sigil_data", {})
    
#     soul_spark.sigil_data["life_cord"] = sigil_data
    
#     # Set final flags and properties on soul spark
#     setattr(soul_spark, "cord_integrity", float(cord_integrity))
#     setattr(soul_spark, "earth_resonance", float(earth_resonance))
#     setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, True)
#     setattr(soul_spark, FLAG_READY_FOR_EARTH, True)
    
#     # Update last modified timestamp
#     setattr(soul_spark, "last_modified", finalization_time)
    
#     # Store simplified final cord structure in soul_spark
#     final_cord = {
#         "integrity": float(cord_integrity),
#         "creation_timestamp": cord_structure.get("creation_timestamp", ""),
#         "finalization_timestamp": finalization_time,
#         "earth_resonance": float(earth_resonance),
#         "spiritual_resonance": float(communication_metrics.get("spiritual_resonance_difficulty", 0.0)),
#         "spiritual_channel_active": communication_metrics.get("spiritual_resonance_achieved", False),
#         "earth_channel_active": communication_metrics.get("earth_resonance_achieved", False),
#         "resonance_steps": communication_metrics.get("total_resonance_steps", 0),
#         "resonance_difficulty": float(communication_metrics.get("overall_resonance_difficulty", 0.0))
#     }
    
#     # Set the complete cord structure in soul_spark
#     setattr(soul_spark, "life_cord", final_cord)
    
#     # Create finalization metrics
#     finalization_metrics = {
#         "cord_integrity": float(cord_integrity),
#         "earth_resonance": float(earth_resonance),
#         "spiritual_resonance": float(communication_metrics.get("spiritual_resonance_difficulty", 0.0)),
#         "resonance_steps": communication_metrics.get("total_resonance_steps", 0),
#         "resonance_difficulty": float(communication_metrics.get("overall_resonance_difficulty", 0.0)),
#         "overall_success": overall_success,
#         "finalization_timestamp": finalization_time
#     }
    
#     logger.info(f"Life Cord Finalized: "
#                f"Integrity: {cord_integrity:.2f} (Divine), "
#                f"Earth Resonance: {earth_resonance:.4f}, "
#                f"Success: {overall_success}")
    
#     return finalization_metrics

# def form_life_cord(soul_spark: SoulSpark, intensity: float = 0.7, 
#                 complexity: float = 0.5) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Forms the divine life cord connecting the soul to Earth and Creator.
#     Creates perfect divine cord with two anchors and channels.
#     The soul must learn to resonate with the anchors to establish communication.
    
#     Args:
#         soul_spark: The SoulSpark object
#         intensity: Factor affecting resonance finding process
#         complexity: Factor affecting channel structure complexity
        
#     Returns:
#         Tuple of (modified soul_spark, process_metrics)
#     """
#     # --- Input Validation ---
#     if not isinstance(soul_spark, SoulSpark):
#         raise TypeError("soul_spark must be a SoulSpark instance.")
#     if not isinstance(intensity, (int, float)) or not (0.1 <= intensity <= 1.0):
#         raise ValueError(f"Intensity must be between 0.1 and 1.0, got {intensity}")
#     if not isinstance(complexity, (int, float)) or not (0.1 <= complexity <= 1.0):
#         raise ValueError(f"Complexity must be between 0.1 and 1.0, got {complexity}")
    
#     spark_id = getattr(soul_spark, "spark_id", "unknown_spark")
#     logger.info(f"--- Beginning Life Cord Formation for Soul {spark_id} (Int={intensity:.2f}, Cmplx={complexity:.2f}) ---")
#     start_time = datetime.now().isoformat()
    
#     # Prepare metrics container
#     process_metrics = {
#         "steps": {},
#         "soul_id": spark_id,
#         "start_time": start_time,
#         "intensity": float(intensity),
#         "complexity": float(complexity),
#         "success": False  # Will be set to True if successful
#     }
    
#     try:
#         # --- Check Prerequisites ---
#         logger.info("Life Cord Step 1: Checking Prerequisites...")
#         _ensure_soul_properties(soul_spark)
#         _check_prerequisites(soul_spark)
        
#         # Energy check before proceeding
#         initial_energy = soul_spark.energy
#         if initial_energy < CORD_ACTIVATION_ENERGY_COST:
#             raise ValueError(f"Insufficient energy ({initial_energy:.1f} SEU) for cord formation. Required: {CORD_ACTIVATION_ENERGY_COST} SEU")
        
#         # Apply energy cost
#         soul_spark.energy -= CORD_ACTIVATION_ENERGY_COST
        
#         # --- Create Divine Cord ---
#         logger.info("Life Cord Step 2: Creating Divine Cord...")
#         cord_structure = _establish_divine_cord()
#         process_metrics["steps"]["divine_cord"] = {
#             "divine_properties": cord_structure["divine_properties"]
#         }
        
#         # --- Create Anchors ---
#         logger.info("Life Cord Step 3: Creating Anchors...")
#         spiritual_anchor = _create_spiritual_anchor(soul_spark)
#         earth_anchor = _create_earth_anchor()
        
#         # Add anchors to cord structure
#         cord_structure["anchors"] = {
#             "spiritual": spiritual_anchor,
#             "earth": earth_anchor
#         }
        
#         process_metrics["steps"]["anchors"] = {
#             "spiritual_anchor": {k: v for k, v in spiritual_anchor.items() if k != "position"},
#             "earth_anchor": {k: v for k, v in earth_anchor.items() if k != "position"}
#         }
        
#         # --- Form Communication Channels ---
#         logger.info("Life Cord Step 4: Forming Communication Channels...")
#         channels = _form_communication_channels(spiritual_anchor, earth_anchor, complexity)
        
#         # Add channels to cord structure
#         cord_structure["channels"] = channels
        
#         process_metrics["steps"]["channels"] = {
#             "spiritual_channel": {k: v for k, v in channels["spiritual_channel"].items() if k != "resonance_result"},
#             "earth_channel": {k: v for k, v in channels["earth_channel"].items() if k != "resonance_result"},
#             "channel_properties": channels["channel_properties"]
#         }
        
#         # --- Test Soul-Channel Resonance ---
#         logger.info("Life Cord Step 5: Testing Soul-Channel Resonance...")
        
#         # Calculate maximum resonance steps based on intensity
#         max_steps = int(144 * intensity)  # Scale steps by intensity
        

# # Test spiritual channel
#         spiritual_results = _test_soul_channel_resonance(
#             soul_spark,
#             "spiritual",
#             channels["spiritual_channel"],
#             spiritual_anchor,
#             max_steps
#         )
        
#         # Test earth channel
#         earth_results = _test_soul_channel_resonance(
#             soul_spark,
#             "earth",
#             channels["earth_channel"],
#             earth_anchor,
#             max_steps
#         )
        
#         process_metrics["steps"]["resonance_tests"] = {
#             "spiritual_results": {k: v for k, v in spiritual_results.items() 
#                                if k not in ["resonance_path", "frequency_adjustments"]},
#             "earth_results": {k: v for k, v in earth_results.items() 
#                            if k not in ["resonance_path", "frequency_adjustments"]}
#         }
        
#         # --- Record Communication Metrics ---
#         logger.info("Life Cord Step 6: Recording Communication Metrics...")
#         communication_metrics = _record_resonance_metrics(
#             soul_spark,
#             spiritual_results,
#             earth_results
#         )
        
#         # Add communication metrics to cord structure
#         cord_structure["communication_metrics"] = communication_metrics
        
#         process_metrics["steps"]["communication_metrics"] = communication_metrics
        
#         # --- Finalize Life Cord ---
#         logger.info("Life Cord Step 7: Finalizing Life Cord...")
#         finalization = _finalize_life_cord(soul_spark, cord_structure)
#         process_metrics["steps"]["finalization"] = finalization
        
#         # --- Generate Sound If Available ---
#         if SOUND_MODULES_AVAILABLE:
#             try:
#                 logger.info("Generating life cord formation sound...")
                
#                 # Create sound generator
#                 sound_gen = SoundGenerator(sample_rate=SAMPLE_RATE)
                
#                 # Create chord based on spiritual and earth frequencies
#                 spiritual_freq = spiritual_anchor["frequency"]
#                 earth_freq = earth_anchor["frequency"]
                
#                 # Create harmonic structure with phi ratios
#                 frequencies = [
#                     spiritual_freq,
#                     earth_freq,
#                     spiritual_freq * PHI,
#                     earth_freq / PHI,
#                     spiritual_freq * 2.0
#                 ]
                
#                 # Amplitudes based on resonance success
#                 spiritual_amp = 0.8 if spiritual_results["resonance_achieved"] else 0.4
#                 earth_amp = 0.8 if earth_results["resonance_achieved"] else 0.4
                
#                 amplitudes = [
#                     spiritual_amp,
#                     earth_amp,
#                     spiritual_amp * 0.7,
#                     earth_amp * 0.7,
#                     spiritual_amp * 0.5
#                 ]
                
#                 # Generate the chord
#                 sound_data = sound_gen.generate_sacred_chord(
#                     frequencies, 
#                     amplitudes, 
#                     5.0
#                 )
                
#                 # Save the sound
#                 sound_file = f"life_cord_{spark_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
#                 file_path = sound_gen.save_sound(
#                     sound_data, 
#                     sound_file, 
#                     f"Life Cord Formation - Soul {spark_id}"
#                 )
                
#                 process_metrics["sound_file"] = sound_file
#                 process_metrics["sound_path"] = file_path
                
#                 logger.info(f"Life cord sound generated: {file_path}")
                
#             except Exception as sound_err:
#                 logger.warning(f"Failed to generate life cord sound: {sound_err}")
        
#         # --- Complete Process Metrics ---
#         end_time = datetime.now().isoformat()
#         end_time_dt = datetime.fromisoformat(end_time)
#         start_time_dt = datetime.fromisoformat(start_time)
#         duration_seconds = (end_time_dt - start_time_dt).total_seconds()
        
#         process_metrics["end_time"] = end_time
#         process_metrics["duration_seconds"] = duration_seconds
#         process_metrics["success"] = True
#         process_metrics["cord_integrity"] = 1.0  # Divine perfection
#         process_metrics["earth_resonance"] = float(getattr(soul_spark, "earth_resonance", 0.0))
#         process_metrics["energy_consumed"] = float(CORD_ACTIVATION_ENERGY_COST)
        
#         # Log summary
#         logger.info(f"--- Life Cord Formation Completed Successfully for Soul {spark_id} ---")
#         logger.info(f"Final Metrics: Divine Integrity=1.0, "
#                    f"Earth Resonance={process_metrics['earth_resonance']:.4f}, "
#                    f"Spiritual Channel: {'Active' if communication_metrics['spiritual_resonance_achieved'] else 'Inactive'}, "
#                    f"Earth Channel: {'Active' if communication_metrics['earth_resonance_achieved'] else 'Inactive'}, "
#                    f"Total Steps: {communication_metrics['total_resonance_steps']}, "
#                    f"Duration: {duration_seconds:.2f}s")
        
#         # Record overall metrics
#         if METRICS_AVAILABLE:
#             metrics.record_metrics('life_cord_summary', process_metrics)
        
#         return soul_spark, process_metrics
        
#     except (ValueError, TypeError, AttributeError) as e:
#         logger.error(f"Life cord formation failed for {spark_id}: {e}")
#         # Set failure flag
#         setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False)
        
#         # Record failure metrics
#         process_metrics["success"] = False
#         process_metrics["error"] = str(e)
#         process_metrics["end_time"] = datetime.now().isoformat()
        
#         if METRICS_AVAILABLE:
#             metrics.record_metrics('life_cord_summary', process_metrics)
            
#         raise  # Re-raise original error
        
#     except Exception as e:
#         logger.critical(f"Unexpected error during life cord formation for {spark_id}: {e}", exc_info=True)
#         # Set failure flag
#         setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False)
        
#         # Record failure metrics
#         process_metrics["success"] = False
#         process_metrics["error"] = str(e)
#         process_metrics["end_time"] = datetime.now().isoformat()
        
#         if METRICS_AVAILABLE:
#             metrics.record_metrics('life_cord_summary', process_metrics)
            
#         raise RuntimeError(f"Unexpected life cord formation failure: {e}") from e
