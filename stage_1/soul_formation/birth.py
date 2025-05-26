"""
Birth Process Functions (Refactored V4.5.0 - Wave Physics & Aura Integration)

Handles birth into physical incarnation using proper wave physics principles.
Implements energy transformation through wave mechanics, spectral memory veil,
acoustic birth signature, and layer-based physical integration. Uses standing
waves for brain-soul connection and maintains aura layer integrity throughout.
Modifies SoulSpark directly. Uses hard fails, proper logging and metrics.
"""

import logging
import numpy as np
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import time
import uuid
from datetime import datetime
from math import sqrt, pi as PI, exp, sin, cos, tanh
from constants.constants import *
from stage_1.evolve.core.evolve_constants import *
from stage_1.evolve.brain_structure.brain_seed import BrainSeed

# --- Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.evolve.brain_structure.brain_seed import BrainSeed, create_brain_seed
    from stage_1.evolve.brain_structure.brain_soul_attachment import attach_soul_to_brain, distribute_soul_aspects
    from stage_1.evolve.core.mother_resonance import create_mother_resonance_data
    from stage_1.evolve.mycelial.mother_integration import create_mother_integration_controller
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies for Birth: {e}", exc_info=True)
    raise ImportError(f"Core dependencies missing for Birth: {e}") from e

# --- Sound Module Imports ---
try:
    from sound.sound_generator import SoundGenerator
    from sound.sounds_of_universe import UniverseSounds
    SOUND_MODULES_AVAILABLE = True
except ImportError:
    logger.warning("Sound modules not available. Birth process will use simulated sound.")
    SOUND_MODULES_AVAILABLE = False

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not available. Metrics will not be recorded.")
    METRICS_AVAILABLE = False

# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites including identity flags. Raises ValueError on failure. """
    logger.debug(f"Checking birth prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("Invalid SoulSpark object.")
    if not getattr(soul_spark, FLAG_READY_FOR_BIRTH, False): msg = f"Prereq failed: {FLAG_READY_FOR_BIRTH} missing."; logger.error(msg); raise ValueError(msg)
    if not getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False): msg = "Prereq failed: Identity not crystallized."; logger.error(msg); raise ValueError(msg)
    cord_integrity = getattr(soul_spark, "cord_integrity", -1.0); earth_resonance = getattr(soul_spark, "earth_resonance", -1.0)
    if cord_integrity < 0 or earth_resonance < 0: msg = "Prereq failed: Missing cord_integrity or earth_resonance."; logger.error(msg); raise AttributeError(msg)
    if cord_integrity < BIRTH_PREREQ_CORD_INTEGRITY_MIN: msg = f"Prereq failed: Cord integrity ({cord_integrity:.3f}) < {BIRTH_PREREQ_CORD_INTEGRITY_MIN})"; logger.error(msg); raise ValueError(msg)
    if earth_resonance < BIRTH_PREREQ_EARTH_RESONANCE_MIN: msg = f"Prereq failed: Earth Resonance ({earth_resonance:.3f}) < {BIRTH_PREREQ_EARTH_RESONANCE_MIN})"; logger.error(msg); raise ValueError(msg)
    logger.debug("Birth prerequisites (Flags, Factors) met. Energy check pending.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties for birth. Raises Error if fails. """
    logger.debug(f"Ensuring properties for birth (Soul {soul_spark.spark_id})...")
    required = ['frequency','stability','coherence','spiritual_energy','physical_energy','cord_integrity','earth_resonance','life_cord','aspects','cumulative_sephiroth_influence','layers']
    if not all(hasattr(soul_spark, attr) for attr in required): missing=[attr for attr in required if not hasattr(soul_spark, attr)]; raise AttributeError(f"SoulSpark missing attributes for Birth: {missing}")
    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    if not isinstance(soul_spark.life_cord, dict): raise TypeError("Missing 'life_cord' dict.")
    if soul_spark.spiritual_energy < 0: raise ValueError("Spiritual energy cannot be negative.")
    if not soul_spark.layers or len(soul_spark.layers) < 2: raise ValueError("Soul must have at least 2 aura layers for birth process.")
    defaults={"memory_veil":None,"breath_pattern":None,"physical_integration":0.0,"incarnated":False,"birth_time":None,"brain_connection":None}
    for attr, default in defaults.items():
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None: setattr(soul_spark, attr, default)
    logger.debug("Soul properties ensured for Birth.")

def _calculate_soul_completeness(soul_spark: SoulSpark) -> float:
    """ Calculates a 0-1 factor representing soul completeness/integration post-journey. """
    stab_factor=soul_spark.stability/MAX_STABILITY_SU; coh_factor=soul_spark.coherence/MAX_COHERENCE_CU; infl_factor=soul_spark.cumulative_sephiroth_influence; aspect_count=len(soul_spark.aspects); aspect_factor=min(1.0, aspect_count/60.0)
    completeness=(stab_factor*0.25 + coh_factor*0.25 + infl_factor*0.30 + aspect_factor*0.20)
    return max(0.0, min(1.0, completeness))

def _find_resonant_layer_indices(soul_spark: SoulSpark, frequency: float) -> List[int]:
    """
    Find indices of layers that resonate with a given frequency.
    Uses quantum resonance calculation to find layers with natural affinity.
    """
    resonant_indices = []
    for i, layer in enumerate(soul_spark.layers):
        if not isinstance(layer, dict): continue
        # Get layer frequencies
        layer_freqs = []
        if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
            layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
        # Find best resonance with this layer
        best_resonance = 0.0
        for layer_freq in layer_freqs:
            # Calculate resonance factor using quantum principles
            freq_ratio = max(frequency, layer_freq) / min(frequency, layer_freq)
            # Check for resonant ratios (integer or phi-based)
            integer_resonance = min(abs(freq_ratio - round(freq_ratio)) for i in range(1, 6))
            phi_resonance = min(abs(freq_ratio - PHI), abs(freq_ratio - 1/PHI))
            # Combine resonance factors
            res = 1.0 - min(integer_resonance, phi_resonance)
            best_resonance = max(best_resonance, res)
        # Add layer if resonance is significant
        if best_resonance > 0.3:
            resonant_indices.append(i)
    return resonant_indices

# --- Mother Profile Helper ---
def _create_or_get_mother_profile(provided_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates mother profile if not provided, or validates and returns provided profile.
    """
    if provided_profile:
        logger.debug("Using provided mother profile")
        return provided_profile
    
    logger.info("Creating default mother resonance profile for birth")
    try:
        # Create mother resonance data
        mother_data = create_mother_resonance_data()
        logger.debug(f"Created mother profile with {len(mother_data)} attributes")
        return mother_data
    except Exception as e:
        logger.warning(f"Failed to create mother profile: {e}. Using minimal profile.")
        # Return minimal profile as fallback
        return {
            'nurturing_capacity': 0.7,
            'love_resonance': 0.8,
            'spiritual': {'connection': 0.6}
        }

def _load_existing_brain_seed(soul_spark: SoulSpark) -> BrainSeed:
    """
    Loads existing brain seed from previous stages instead of creating new one.
    """
    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    
    # Try to load from various possible locations
    possible_paths = [
        f"output/brain_seeds/brain_{spark_id}_developed.json",
        f"output/brain_seeds/brain_{spark_id}_basic.json",
        f"output/brain_seeds/brain_{spark_id}.json"
    ]
    
    for brain_path in possible_paths:
        try:
            if os.path.exists(brain_path):
                logger.info(f"Loading existing brain seed from {brain_path}")
                brain_seed = BrainSeed.load_state(brain_path)
                logger.debug(f"Loaded brain seed with {brain_seed.base_energy_level:.2E} BEU")
                return brain_seed
        except Exception as e:
            logger.warning(f"Failed to load brain seed from {brain_path}: {e}")
            continue
    
    # If no existing brain seed found, this is an error
    raise FileNotFoundError(f"No existing brain seed found for soul {spark_id}. "
                           f"Brain seed should be created in earlier stages before birth.")

# --- Core Birth Functions ---

def _create_energy_transformation_waveguide(soul_spark: SoulSpark, required_joules: float, 
                                          buffer_factor: float) -> Tuple[float, Dict[str, Any]]:
    """
    Creates a waveguide for spiritual to physical energy transformation using wave physics.
    Transforms energy while preserving coherence and total energy conservation.
    """
    logger.info("Birth Phase: Creating Energy Transformation Waveguide...")
    
    # Calculate required energy with buffer
    total_joules_needed = required_joules * buffer_factor
    required_physical_seu = total_joules_needed * ENERGY_SCALE_FACTOR
    
    # Check available energy
    if soul_spark.spiritual_energy < required_physical_seu:
        raise ValueError(f"Insufficient spiritual energy ({soul_spark.spiritual_energy:.1f} SEU) for birth ({required_physical_seu:.1f} SEU).")
    
    try:
        # Get soul properties for waveguide calculations
        soul_frequency = soul_spark.frequency
        soul_coherence = soul_spark.coherence / MAX_COHERENCE_CU  # Normalize to 0-1
        
        # Calculate waveguide properties using wave physics
        wavelength = SPEED_OF_SOUND / soul_frequency  # Base wavelength
        propagation_constant = 2 * PI / wavelength
        resonant_wavelength = wavelength * buffer_factor  # Scale with buffer
        
        # Calculate quantum efficiency based on coherence
        quantum_efficiency = 0.7 + 0.3 * soul_coherence
        
        # Calculate energy transfer coefficient using wave equations
        # Using quantum tunneling principles for efficiency
        transfer_coefficient = quantum_efficiency * (1.0 - exp(-buffer_factor))
        
        # Calculate actual energy transferred considering wave properties
        energy_converted_seu = required_physical_seu * transfer_coefficient
        
        # Calculate reflection coefficient (energy that remains spiritual)
        reflection_coefficient = 1.0 - transfer_coefficient
        
        # Apply energy transformation with conservation principles
        physical_energy_gain = energy_converted_seu
        spiritual_energy_loss = energy_converted_seu
        
        # Update soul energy values
        soul_spark.physical_energy = physical_energy_gain
        soul_spark.spiritual_energy -= spiritual_energy_loss
        
        # Create waveguide metrics
        waveguide_metrics = {
            "energy_converted_seu": float(energy_converted_seu),
            "buffer_factor_used": float(buffer_factor),
            "brain_need_joules": float(required_joules),
            "waveguide_wavelength_m": float(resonant_wavelength),
            "propagation_constant": float(propagation_constant),
            "quantum_efficiency": float(quantum_efficiency),
            "transfer_coefficient": float(transfer_coefficient),
            "reflection_coefficient": float(reflection_coefficient),
            "conservation_error": float(abs(spiritual_energy_loss - physical_energy_gain))
        }
        
        logger.info(f"Energy waveguide created. Converted: {energy_converted_seu:.1f} SEU, Efficiency: {quantum_efficiency:.3f}")
        return energy_converted_seu, waveguide_metrics
        
    except Exception as e:
        logger.error(f"Error creating energy transformation waveguide: {e}", exc_info=True)
        raise RuntimeError("Energy transformation failed critically.") from e

def _create_spectral_memory_veil(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """
    Creates a spectral memory veil using light physics principles.
    Implements frequency-specific attenuation across different bands.
    """
    logger.info("Birth Phase: Creating Spectral Memory Veil...")
    
    if not(0.1 <= intensity <= 1.0):
        raise ValueError("Intensity out of range.")
    
    try:
        # Get soul properties for spectral calculations
        soul_coherence_norm = soul_spark.coherence / MAX_COHERENCE_CU
        soul_frequency = soul_spark.frequency
        
        # Calculate base veil strength using quantum field principles
        veil_strength = BIRTH_VEIL_STRENGTH_BASE + intensity * BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR
        veil_strength = max(0.0, min(1.0, veil_strength))
        logger.debug(f"Memory Veil: Intensity={intensity:.2f}, CohNorm={soul_coherence_norm:.3f} -> VeilStrength={veil_strength:.4f}")
        
        # Initialize spectral band definitions (based on soul frequency)
        # Use frequency multipliers to create bands
        base_freq = soul_frequency
        bands = {
            "infrared": (base_freq * 0.1, base_freq * 0.5),  # Long-term/deep memories
            "red": (base_freq * 0.5, base_freq * 0.8),       # Emotional memories
            "orange": (base_freq * 0.8, base_freq * 0.9),    # Sensory memories
            "yellow": (base_freq * 0.9, base_freq),          # Intellectual memories
            "green": (base_freq, base_freq * 1.2),           # Heart/connection memories
            "blue": (base_freq * 1.2, base_freq * 1.5),      # Spiritual knowledge
            "violet": (base_freq * 1.5, base_freq * 2.0),    # Higher consciousness
            "ultraviolet": (base_freq * 2.0, base_freq * 3.0) # Beyond-physical memories
        }
        
        # Create veil details structure
        deployment_time = datetime.now().isoformat()
        veil_details = {
            "strength_factor": veil_strength,
            "deployment_time": deployment_time,
            "spectral_bands": bands,
            "band_attenuation": {},
            "aspect_retention": {}
        }
        
        # Calculate spectral attenuation for each band
        # Different bands have different attenuation levels
        for band_name, (low_freq, high_freq) in bands.items():
            # Calculate band-specific attenuation
            base_attenuation = veil_strength
            
            # Apply band-specific modifiers
            if band_name in ["infrared", "ultraviolet"]:
                # Strongest veil for deepest and highest memories
                attenuation = base_attenuation * 1.2
            elif band_name in ["red", "violet"]:
                # Strong veil for emotional and higher consciousness
                attenuation = base_attenuation * 1.1
            elif band_name in ["green"]:
                # Reduced veil for heart/connection memories
                attenuation = base_attenuation * 0.8
            else:
                attenuation = base_attenuation
            
            # Apply coherence resistance factor
            coherence_resistance = soul_coherence_norm * VEIL_COHERENCE_RESISTANCE_FACTOR
            final_attenuation = max(0.0, min(1.0, attenuation - coherence_resistance))
            
            # Store band attenuation
            veil_details["band_attenuation"][band_name] = float(final_attenuation)
        
        affected_count = 0
        # Apply to each aspect using spectral mapping
        if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict):
            raise AttributeError("Cannot deploy veil: Soul aspects missing.")
            
        for aspect_name, aspect_data in soul_spark.aspects.items():
            if not isinstance(aspect_data, dict): continue
            
            # Determine aspect's spectral band based on frequency characteristics
            aspect_freq = aspect_data.get('frequency', soul_frequency)
            aspect_band = "yellow"  # Default band
            
            # Find which band contains this aspect
            for band_name, (low_freq, high_freq) in bands.items():
                if low_freq <= aspect_freq <= high_freq:
                    aspect_band = band_name
                    break
            
            # Get band-specific attenuation
            band_attenuation = veil_details["band_attenuation"].get(aspect_band, veil_strength)
            
            # Calculate retention as inverse of attenuation
            base_retention = VEIL_BASE_RETENTION
            type_mod = BIRTH_VEIL_MEMORY_RETENTION_MODS.get(aspect_name, 0.0)
            
            # Enhanced retention calculation using quantum field equations
            coherence_resistance = soul_coherence_norm * VEIL_COHERENCE_RESISTANCE_FACTOR
            
            # Calculate final retention using wave interference principles
            retention_factor = base_retention + coherence_resistance + type_mod
            final_retention = retention_factor * (1.0 - band_attenuation * 0.9)
            final_retention = max(0.01, min(1.0, final_retention))
            
            aspect_data['retention_factor'] = float(final_retention)
            aspect_data['spectral_band'] = aspect_band
            veil_details['aspect_retention'][aspect_name] = {
                "retention_factor": float(final_retention),
                "spectral_band": aspect_band,
                "band_attenuation": float(band_attenuation)
            }
            affected_count += 1
        
        # Store veil in soul
        setattr(soul_spark, "memory_veil", veil_details)
        if hasattr(soul_spark, 'memory_retention'): delattr(soul_spark, 'memory_retention')
        soul_spark.last_modified = deployment_time
        
        # Calculate average retention
        avg_retention = np.mean([data["retention_factor"] for data in veil_details['aspect_retention'].values()]) if veil_details['aspect_retention'] else 0.0
        
        # Prepare metrics
        phase_metrics = {
            "veil_strength_factor": veil_strength,
            "aspects_veiled_count": affected_count,
            "avg_retention_factor": avg_retention,
            "spectral_bands_count": len(bands),
            "timestamp": deployment_time
        }
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('birth_spectral_memory_veil', phase_metrics)
        
        logger.info(f"Spectral memory veil deployed. Strength:{veil_strength:.3f}, Aspects:{affected_count}, AvgRet:{avg_retention:.3f}")
        return phase_metrics
        
    except Exception as e:
        logger.error(f"Error deploying spectral memory veil: {e}", exc_info=True)
        raise RuntimeError("Memory veil deployment failed critically.") from e

def _generate_acoustic_birth_signature(soul_spark: SoulSpark, base_frequency: float,
                                   intensity: float) -> np.ndarray:
    """
    Generates an acoustic birth signature using sound physics.
    Creates a unique sound pattern that encodes the soul's identity in physical form.
    """
    if not SOUND_MODULES_AVAILABLE:
        logger.warning("Sound modules unavailable. Using simulated acoustic signature.")
        # Create simulated signature (simple sine wave)
        duration = 3.0  # seconds
        sample_rate = SAMPLE_RATE if 'SAMPLE_RATE' in globals() else 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signature = 0.7 * np.sin(2 * PI * base_frequency * t)
        return signature
    
    try:
        # Initialize sound generators
        sound_gen = SoundGenerator(sample_rate=SAMPLE_RATE)
        universe_sounds = UniverseSounds(sample_rate=SAMPLE_RATE)
        
        # Get soul properties for signature generation
        soul_name = getattr(soul_spark, "name", "Unnamed")
        soul_freq = soul_spark.frequency
        
        # Calculate signature properties
        birth_freq = base_frequency
        duration = 5.0  # seconds
        
        # Generate fundamental tone with harmonics based on soul frequency
        harmonics = [1.0, PHI, 2.0, PHI*2, 3.0]  # Harmonic series including Phi ratios
        amplitudes = [0.7, 0.5, 0.4, 0.25, 0.15]  # Decreasing amplitudes
        
        # Generate harmonic tone
        birth_tone = sound_gen.generate_harmonic_tone(birth_freq, harmonics, amplitudes, duration)
        
        # Add name frequency modulation
        name_len = len(soul_name)
        if name_len > 0:
            # Convert name to frequency modulator
            name_mod = np.zeros_like(birth_tone)
            for i, char in enumerate(soul_name):
                char_val = ord(char) / 128.0  # Normalize to 0-1 range
                mod_freq = birth_freq * (0.5 + char_val)
                mod_phase = i / name_len * 2 * PI
                mod_amp = 0.2 * (name_len - i) / name_len  # Decreasing amplitude
                
                # Create time array
                t = np.linspace(0, duration, len(birth_tone), endpoint=False)
                
                # Add modulation component
                name_mod += mod_amp * np.sin(2 * PI * mod_freq * t + mod_phase)
            
            # Apply name modulation
            birth_tone = birth_tone * (1.0 + 0.3 * name_mod)
        
        # Add dimensional transition effect
        dim_transition = universe_sounds.generate_dimensional_transition(
            duration=duration,
            sample_rate=SAMPLE_RATE,
            transition_type='soul_to_physical',
            amplitude=0.5
        )
        
        # Mix components
        signature = 0.7 * birth_tone + 0.3 * dim_transition
        
        # Apply envelope
        env_fade = int(0.2 * len(signature))  # 20% fade in/out
        if env_fade > 0:
            # Create fade in/out envelope
            fade_in = np.linspace(0, 1, env_fade)
            fade_out = np.linspace(1, 0, env_fade)
            env = np.ones(len(signature))
            env[:env_fade] = fade_in
            env[-env_fade:] = fade_out
            
            # Apply envelope
            signature = signature * env
        
        # Normalize final signature
        max_amp = np.max(np.abs(signature))
        if max_amp > 0:
            signature = signature / max_amp * 0.9  # Scale to 90% of maximum amplitude
        
        logger.info(f"Generated acoustic birth signature: Freq={birth_freq:.1f}Hz, Duration={duration:.1f}s")
        return signature
        
    except Exception as e:
        logger.error(f"Error generating acoustic birth signature: {e}", exc_info=True)
        # Fall back to simple sine wave if generation fails
        logger.warning("Falling back to simple sine wave signature")
        duration = 3.0  # seconds
        sample_rate = SAMPLE_RATE if 'SAMPLE_RATE' in globals() else 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signature = 0.7 * np.sin(2 * PI * base_frequency * t)
        return signature

def _connect_to_physical_form(soul_spark: SoulSpark, intensity: float, mother_profile: Optional[Dict]) -> Tuple[float, float, Dict[str, Any]]:
    """ 
    Calculates potential connection strength (0-1) and acceptance (0-1) using wave physics.
    Implements constructive interference between soul and physical frequencies.
    """
    logger.info("Birth Phase: Calculating Physical Form Connection Potential...")
    if not(0.1<=intensity<=1.0): raise ValueError("Intensity out of range.")
    try:
        earth_resonance=soul_spark.earth_resonance; cord_integrity=soul_spark.cord_integrity
        mother_nurturing=mother_profile.get('nurturing_capacity',0.5) if mother_profile else 0.5
        mother_spiritual=mother_profile.get('spiritual',{}).get('connection',0.5) if mother_profile else 0.5
        mother_love=mother_profile.get('love_resonance',0.5) if mother_profile else 0.5
        
        # Enhanced connection calculation using wave interference principles
        # Phase coherence between soul and physical frequencies
        phase_coherence = 0.5 + 0.5 * cos(earth_resonance * PI)  # 0-1 range
        
        # Wave impedance matching between soul and physical form
        impedance_match = 4 * (cord_integrity * mother_spiritual) / ((cord_integrity + mother_spiritual)**2)
        
        # Calculate base connection strength using wave physics
        base_strength = (earth_resonance * BIRTH_CONN_WEIGHT_RESONANCE + cord_integrity * BIRTH_CONN_WEIGHT_INTEGRITY) * (1.0 + mother_spiritual * BIRTH_CONN_MOTHER_STRENGTH_FACTOR)
        
        # Apply phase coherence and impedance matching
        connection_factor = intensity * BIRTH_CONN_STRENGTH_FACTOR
        connection_strength = min(BIRTH_CONN_STRENGTH_CAP, 
                                max(0.0, base_strength * (1.0 + connection_factor) * 
                                   phase_coherence * (0.7 + 0.3 * impedance_match)))
        
        logger.debug(f"Physical Conn: PhaseCoherence={phase_coherence:.3f}, ImpedanceMatch={impedance_match:.3f}, Strength={connection_strength:.4f}")
        
        # Calculate acceptance using wave resonance principles
        trauma_base = intensity * BIRTH_CONN_TRAUMA_FACTOR
        trauma_reduction = mother_nurturing * BIRTH_CONN_MOTHER_TRAUMA_REDUCTION + mother_love * 0.1
        trauma_level = max(0.0, min(1.0, trauma_base - trauma_reduction))
        
        # Use resonant absorption equation for acceptance calculation
        acceptance_base = max(BIRTH_ACCEPTANCE_MIN, 1.0 - trauma_level * BIRTH_CONN_ACCEPTANCE_TRAUMA_FACTOR)
        acceptance_boost = mother_love * BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR
        
        # Calculate final acceptance with resonance principles
        acceptance = min(1.0, max(0.0, acceptance_base + acceptance_boost))
        logger.debug(f"Physical Form Acceptance: TraumaLevel={trauma_level:.3f}, AcceptFactor={acceptance:.4f}")
        
        # Create enhanced metrics with wave physics properties
        phase_metrics = {
            "connection_strength_factor": float(connection_strength),
            "form_acceptance_factor": float(acceptance),
            "trauma_level_factor": float(trauma_level),
            "phase_coherence": float(phase_coherence),
            "impedance_match": float(impedance_match),
            "mother_influence_applied": mother_profile is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('birth_physical_connection', phase_metrics)
            
        logger.info(f"Physical connection potential calculated. ConnFactor:{connection_strength:.3f}, AcceptFactor:{acceptance:.3f}")
        return float(connection_strength), float(acceptance), phase_metrics
        
    except Exception as e:
        logger.error(f"Error calculating physical form connection: {e}", exc_info=True)
        raise RuntimeError("Physical form connection calculation failed.") from e

def _integrate_first_breath(soul_spark: SoulSpark, physical_connection: float, 
                          form_acceptance: float, intensity: float) -> Dict[str, Any]:
    """
    Integrates first breath using acoustic wave principles.
    Creates standing wave patterns between soul and physical form.
    """
    logger.info("Birth Phase: Integrating first breath...")
    if not(0.0<=physical_connection<=1.0): raise ValueError("physical_connection invalid.")
    if not(0.0<=form_acceptance<=1.0): raise ValueError("form_acceptance invalid.")
    if not(0.1<=intensity<=1.0): raise ValueError("Intensity invalid.")
    
    try:
        earth_resonance = soul_spark.earth_resonance
        
        # Calculate breath parameters using wave acoustics
        breath_amplitude = max(0.0, min(1.0, BIRTH_BREATH_AMP_BASE + intensity * BIRTH_BREATH_AMP_INTENSITY_FACTOR))
        breath_depth = max(0.0, min(1.0, BIRTH_BREATH_DEPTH_BASE + intensity * BIRTH_BREATH_DEPTH_INTENSITY_FACTOR))
        logger.debug(f"First Breath: Amp={breath_amplitude:.3f}, Depth={breath_depth:.3f}")
        
        # Generate acoustic birth signature based on earth frequency
        birth_frequency = EARTH_BREATH_FREQUENCY  # Base frequency for breath
        acoustic_signature = _generate_acoustic_birth_signature(
            soul_spark, birth_frequency, intensity)
        
        # Save birth signature if sound modules available
        signature_path = None
        if SOUND_MODULES_AVAILABLE:
            try:
                sound_gen = SoundGenerator(sample_rate=SAMPLE_RATE)
                soul_id = getattr(soul_spark, 'spark_id', 'unknown')
                filename = f"birth_signature_{soul_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                signature_path = sound_gen.save_sound(
                    acoustic_signature, filename, f"Birth Signature for Soul {soul_id}")
                logger.info(f"Saved birth acoustic signature to {signature_path}")
            except Exception as e:
                logger.warning(f"Failed to save birth acoustic signature: {e}")
        
        # Calculate integration using standing wave principles
        # Create standing wave between soul and earth frequencies
        integration_strength = (physical_connection * BIRTH_FINAL_INTEGRATION_WEIGHT_CONN +
                              form_acceptance * BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT +
                              breath_depth * BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH)
        
        # Apply earth resonance enhancement
        integration_strength += earth_resonance * 0.1
        
        # Calculate resonance with soul layers
        layer_resonance = 0.0
        resonant_layer_indices = _find_resonant_layer_indices(soul_spark, birth_frequency)
        if resonant_layer_indices:
            # Calculate average resonance enhancement from layers
            layer_resonance = 0.1 * len(resonant_layer_indices) / len(soul_spark.layers)
            integration_strength += layer_resonance
            
            # Store breath resonance in resonant layers
            breath_time = datetime.now().isoformat()
            for layer_idx in resonant_layer_indices:
                if layer_idx < len(soul_spark.layers):
                    layer = soul_spark.layers[layer_idx]
                    if isinstance(layer, dict):
                        if 'breath_resonance' not in layer:
                            layer['breath_resonance'] = {}
                        layer['breath_resonance']['frequency'] = float(birth_frequency)
                        layer['breath_resonance']['amplitude'] = float(breath_amplitude)
                        layer['breath_resonance']['timestamp'] = breath_time
        
        # Calculate final integration factor
        total_integration_factor = min(1.0, max(0.0, integration_strength))
        logger.debug(f"First Breath -> IntegrationFactor={total_integration_factor:.4f} (LayerRes={layer_resonance:.3f})")
        
        # Create breath pattern with enhanced wave physics
        breath_time = datetime.now().isoformat()
        breath_config = {
            "target_frequency_hz": float(EARTH_BREATH_FREQUENCY),
            "initial_amplitude_factor": breath_amplitude,
            "initial_depth_factor": breath_depth,
            "integration_achieved_factor": total_integration_factor,
            "resonant_layer_indices": resonant_layer_indices,
"layer_resonance_factor": float(layer_resonance),
            "acoustic_signature_generated": signature_path is not None,
            "timestamp": breath_time
        }
        
        # Store breath pattern and integration factor in soul
        setattr(soul_spark, "breath_pattern", breath_config)
        setattr(soul_spark, "physical_integration", float(total_integration_factor))
        soul_spark.last_modified = breath_time
        
        # Create phase metrics
        phase_metrics = {
            "physical_integration_factor": total_integration_factor,
            "layer_resonance_factor": layer_resonance,
            "resonant_layers_count": len(resonant_layer_indices),
            "acoustic_signature_saved": signature_path is not None,
            "timestamp": breath_time
        }
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('birth_first_breath', phase_metrics)
            
        logger.info(f"First breath integrated. Physical Integration Factor: {total_integration_factor:.4f}")
        return phase_metrics
        
    except Exception as e:
        logger.error(f"Error integrating first breath: {e}", exc_info=True)
        raise RuntimeError("First breath integration failed critically.") from e

def _create_brain_soul_standing_waves(soul_spark: SoulSpark, brain_seed: BrainSeed) -> Dict[str, Any]:
    """
    Creates standing wave patterns between brain and soul using wave physics.
    Implements resonant nodes at key connection points for optimal coherence.
    """
    logger.info("Birth Phase: Creating Brain-Soul Standing Waves...")
    
    try:
        # Get soul properties for wave calculations
        soul_frequency = soul_spark.frequency
        physical_integration = soul_spark.physical_integration
        
        # Calculate fundamental frequencies
        brain_frequency = getattr(brain_seed, 'resonant_frequency', soul_frequency * 0.8)
        
        # Calculate wavelength and propagation parameters
        soul_wavelength = SPEED_OF_LIGHT / (soul_frequency * 1e6)  # Convert to appropriate scale
        brain_wavelength = SPEED_OF_LIGHT / (brain_frequency * 1e6)
        
        # Calculate average wavelength for standing wave
        avg_wavelength = (soul_wavelength + brain_wavelength) / 2
        
        # Calculate connection distance (simplified)
        connection_distance = avg_wavelength * 10  # Example distance
        
        # Calculate wave number and propagation constant
        wave_number = 2 * PI / avg_wavelength
        propagation_constant = wave_number
        
        # Calculate number of nodes based on physical integration
        # More nodes = stronger connection
        num_nodes = int(3 + physical_integration * 7)
        
        # Create nodes using Fibonacci sequence for optimal spacing
        fibonacci = [1, 1, 2, 3, 5, 8, 13, 21]
        fib_total = sum(fibonacci[:num_nodes])
        
        # Initialize node data
        nodes = []
        for i in range(min(num_nodes, len(fibonacci))):
            # Calculate normalized position
            position = sum(fibonacci[:i+1]) / fib_total
            
            # Calculate amplitude using standing wave equation
            amplitude = 0.5 + 0.5 * np.sin(PI * position)
            
            # Calculate node frequency - smooth transition from soul to brain
            node_freq = soul_frequency * (1 - position) + brain_frequency * position
            
            # Calculate phase relationship
            phase = position * PI
            
            # Create node data
            node = {
                "position": float(position),
                "amplitude": float(amplitude),
                "frequency": float(node_freq),
                "phase": float(phase),
                "is_antinode": i % 2 == 0  # Alternate nodes and antinodes
            }
            
            nodes.append(node)
        
        # Calculate impedance matching between soul and brain
        impedance_ratio = soul_frequency / brain_frequency
        impedance_match = 4 * impedance_ratio / ((1 + impedance_ratio) ** 2)  # Max at ratio=1
        
        # Calculate wave transmission coefficient
        transmission = physical_integration * impedance_match
        
        # Create standing wave structure
        standing_waves = {
            "soul_frequency": float(soul_frequency),
            "brain_frequency": float(brain_frequency),
            "wavelength": float(avg_wavelength),
            "propagation_constant": float(propagation_constant),
            "impedance_match": float(impedance_match),
            "transmission_coefficient": float(transmission),
            "nodes": nodes,
            "node_count": len(nodes)
        }
        
        # Update brain seed with connection properties
        if hasattr(brain_seed, 'wave_connection'):
            brain_seed.wave_connection = standing_waves
        else:
            setattr(brain_seed, 'wave_connection', standing_waves)
        
        logger.info(f"Brain-Soul standing waves created. Nodes:{len(nodes)}, Transmission:{transmission:.4f}")
        return standing_waves
        
    except Exception as e:
        logger.error(f"Error creating brain-soul standing waves: {e}", exc_info=True)
        raise RuntimeError("Brain-soul standing wave creation failed critically.") from e

def _integrate_layers_with_physical_form(soul_spark: SoulSpark, brain_seed: BrainSeed,
                                      physical_integration: float) -> Dict[str, Any]:
    """
    Integrates soul layers with physical form using resonant field principles.
    Creates impedance matching between layers and body for optimal energy transfer.
    """
    logger.info("Birth Phase: Integrating Layers with Physical Form...")
    
    try:
        # Get soul properties for integration
        soul_layers = soul_spark.layers
        if not soul_layers:
            raise ValueError("Soul has no layers to integrate")
        
        # Get brain properties
        brain_frequency = getattr(brain_seed, 'resonant_frequency', soul_spark.frequency * 0.8)
        
        # Integration metrics
        integrated_layers = 0
        total_resonance = 0.0
        layer_integrations = []
        
        # Process each layer for integration
        for layer_idx, layer in enumerate(soul_layers):
            if not isinstance(layer, dict):
                continue
                
            # Get layer frequencies
            layer_freqs = []
            if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
                layer_freqs.extend([f for f in layer['resonant_frequencies'] if f > FLOAT_EPSILON])
            else:
                # Use default frequency based on layer index
                default_freq = soul_spark.frequency * (0.5 + 0.1 * layer_idx)
                layer_freqs = [default_freq]
            
            # Calculate best resonance with physical form
            best_resonance = 0.0
            best_freq = 0.0
            
            for layer_freq in layer_freqs:
                # Calculate resonance with brain frequency
                brain_res = _calculate_frequency_resonance(layer_freq, brain_frequency)
                
                # Find best resonance
                if brain_res > best_resonance:
                    best_resonance = brain_res
                    best_freq = layer_freq
            
            # Skip layers with poor resonance
            if best_resonance < 0.2:
                continue
                
            # Calculate layer-specific integration strength
            layer_strength = layer.get('strength', 0.5)
            integration_strength = physical_integration * best_resonance * layer_strength
            
            # Create physical integration data
            integration_time = datetime.now().isoformat()
            physical_integration_data = {
                "frequency": float(best_freq),
                "resonance": float(best_resonance),
                "integration_strength": float(integration_strength),
                "timestamp": integration_time
            }
            
            # Store integration data in layer
            if 'physical_integration' not in layer:
                layer['physical_integration'] = {}
            
            layer['physical_integration'] = physical_integration_data
            
            # Add to metrics
            integrated_layers += 1
            total_resonance += best_resonance
            
            layer_integrations.append({
                "layer_idx": layer_idx,
                "frequency": float(best_freq),
                "resonance": float(best_resonance),
                "integration_strength": float(integration_strength)
            })
        
        # Calculate overall layer integration factor
        avg_resonance = total_resonance / integrated_layers if integrated_layers > 0 else 0.0
        overall_integration = physical_integration * avg_resonance
        
        # Create integration metrics
        integration_metrics = {
            "integrated_layers_count": integrated_layers,
            "total_layers_count": len(soul_layers),
            "average_resonance": float(avg_resonance),
            "overall_integration_factor": float(overall_integration),
            "layer_integrations": layer_integrations
        }
        
        logger.info(f"Layer integration complete. Integrated:{integrated_layers}/{len(soul_layers)}, AvgRes:{avg_resonance:.4f}")
        return integration_metrics
        
    except Exception as e:
        logger.error(f"Error integrating layers with physical form: {e}", exc_info=True)
        raise RuntimeError("Layer integration failed critically.") from e

def _calculate_frequency_resonance(freq1: float, freq2: float) -> float:
    """
    Calculates resonance between two frequencies using wave principles.
    Checks for integer ratios, phi ratios, and harmonic relationships.
    """
    if freq1 <= FLOAT_EPSILON or freq2 <= FLOAT_EPSILON:
        return 0.0
        
    # Calculate frequency ratio
    ratio = max(freq1, freq2) / min(freq1, freq2)
    
    # Check for integer ratio resonance
    integer_resonance = 1.0
    for i in range(1, 6):
        for j in range(1, 6):
            if j == 0: continue
            int_ratio = i / j
            if abs(ratio - int_ratio) < RESONANCE_INTEGER_RATIO_TOLERANCE:
                integer_resonance = 1.0 - abs(ratio - int_ratio) / RESONANCE_INTEGER_RATIO_TOLERANCE
                break
    
    # Check for Phi-related resonance
    phi_resonance = 0.0
    phi_ratios = [PHI, 1/PHI, PHI**2, 1/(PHI**2)]
    for phi_ratio in phi_ratios:
        if abs(ratio - phi_ratio) < RESONANCE_PHI_RATIO_TOLERANCE:
            phi_resonance = 1.0 - abs(ratio - phi_ratio) / RESONANCE_PHI_RATIO_TOLERANCE
            break
    
    # Take the best resonance type
    return max(integer_resonance, phi_resonance)

def _finalize_birth_state(soul_spark: SoulSpark) -> Dict[str, Any]:
    """ 
    Applies final adjustments to soul state due to embodiment using wave physics.
    Ensures coherence is preserved through proper frequency stabilization.
    """
    logger.info("Birth Phase: Finalizing soul state...")
    try:
        initial_freq = soul_spark.frequency
        initial_stability = soul_spark.stability
        physical_integration = soul_spark.physical_integration
        
        # Calculate frequency shift using wave physics
        # Find new equilibrium frequency based on physical integration
        freq_shift_percentage = BIRTH_FINAL_FREQ_SHIFT_FACTOR * (1.0 - physical_integration)
        
        # Frequency shift is proportional to integration quality
        # Better integration = less frequency disruption
        freq_reduction = initial_freq * freq_shift_percentage
        new_freq = max(FLOAT_EPSILON * 10, initial_freq - freq_reduction)
        soul_spark.frequency = float(new_freq)
        logger.debug(f"Finalize State: Freq {initial_freq:.1f}->{new_freq:.1f}")
        
        # Update frequency structure
        if hasattr(soul_spark, '_validate_or_init_frequency_structure'):
            soul_spark._validate_or_init_frequency_structure()
        else:
            raise AttributeError("Missing frequency structure update method.")
        
        # Calculate stability adjustment using wave interference principles
        # Physical integration creates constructive/destructive interference
        stability_penalty_percentage = BIRTH_FINAL_STABILITY_PENALTY_FACTOR * (1.0 - physical_integration)
        stability_reduction = initial_stability * stability_penalty_percentage
        
        # Calculate new stability
        new_stability = max(0.0, min(MAX_STABILITY_SU, initial_stability - stability_reduction))
        soul_spark.stability = float(new_stability)
        logger.debug(f"Finalize State: Stability {initial_stability:.1f}->{new_stability:.1f}")
        
        # Mark soul as incarnated
        setattr(soul_spark, FLAG_INCARNATED, True)
        birth_time = datetime.now().isoformat()
        setattr(soul_spark, "birth_time", birth_time)
        soul_spark.last_modified = birth_time
        
        # Create finalization metrics
        phase_metrics = {
            "final_frequency_hz": new_freq,
            "frequency_change_hz": new_freq - initial_freq,
            "final_stability_su": new_stability,
            "stability_change_su": new_stability - initial_stability,
            "birth_timestamp": birth_time,
            "success": True
        }
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('birth_finalization', phase_metrics)
            
        logger.info(f"Birth state finalized. Final Freq: {new_freq:.1f} Hz, Final Stability: {new_stability:.1f} SU")
        return phase_metrics
        
    except Exception as e:
        logger.error(f"Error finalizing birth state: {e}", exc_info=True)
        raise RuntimeError("Birth finalization failed critically.") from e

# --- Orchestration Function ---
def perform_birth(soul_spark: SoulSpark,
                 intensity: float = BIRTH_INTENSITY_DEFAULT,
                 mother_profile: Optional[Dict[str, Any]] = None) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ 
    Performs complete birth process using wave physics.
    Transforms spiritual energy to physical, creates brain-soul connection,
    and implements proper layer integration with physical form.
    """
    # --- Input Validation & Setup ---
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("soul_spark invalid.")
    if not (0.1 <= intensity <= 1.0):
        raise ValueError("Intensity invalid.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    log_msg_suffix = "with Mother Profile" if mother_profile else "without Mother Profile"
    logger.info(f"--- Starting Birth Process for Soul {spark_id} (Int={intensity:.2f}) {log_msg_suffix} ---")
    
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}
    brain_seed = None

    try:
        # --- Create or Get Mother Profile ---
        mother_profile = _create_or_get_mother_profile(mother_profile)
        logger.info(f"Using mother profile with nurturing: {mother_profile.get('nurturing_capacity', 'unknown')}")
        
        # --- Check Prerequisites and Properties ---
        _ensure_soul_properties(soul_spark)
        _check_prerequisites(soul_spark)
        
        # Store initial state for metrics
        initial_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'spiritual_energy_seu': soul_spark.spiritual_energy,
            'physical_energy_seu': soul_spark.physical_energy,
            'earth_resonance': soul_spark.earth_resonance,
            'cord_integrity': soul_spark.cord_integrity,
            'physical_integration': soul_spark.physical_integration
        }
        logger.info(f"Birth Initial State: S={initial_state['stability_su']:.1f}, C={initial_state['coherence_cu']:.1f}, E_spirit={initial_state['spiritual_energy_seu']:.1f}")

        # --- 1. Calculate Connection Potential with Wave Physics ---
        logger.info("Birth Step 1: Calculating Connection Potential...")
        connection_strength, form_acceptance, metrics_conn = _connect_to_physical_form(soul_spark, intensity, mother_profile)
        process_metrics_summary['steps']['connection_potential'] = metrics_conn

        # --- 2. Energy Transformation Using Waveguide Physics ---
        logger.info("Birth Step 2: Energy Transformation...")
        soul_completeness = _calculate_soul_completeness(soul_spark)
        buffer_factor = 1.0 + (0.4 * soul_completeness)
        required_joules = ENERGY_BRAIN_14_DAYS_JOULES
        
        # Create energy transformation waveguide
        energy_converted_seu, energy_metrics = _create_energy_transformation_waveguide(
            soul_spark, required_joules, buffer_factor)
        process_metrics_summary['steps']['energy_transformation'] = energy_metrics

        # --- 3. Minimal Brain Seed Creation with Wave Integration ---
        logger.info("Birth Step 3: Creating Brain Seed & Wave Integration...")
        
        # Calculate Brain Energy Units
        physical_joules = soul_spark.physical_energy / ENERGY_SCALE_FACTOR
        total_available_beu = physical_joules * BRAIN_ENERGY_SCALE_FACTOR
        logger.debug(f"Converting {soul_spark.physical_energy:.1f} SEU to {total_available_beu:.2E} BEU.")
        
        # Allocate BEU with enhanced ratios
        core_beu = total_available_beu * BIRTH_ALLOC_SEED_CORE
        mycelial_beu = total_available_beu * (BIRTH_ALLOC_REGIONS + BIRTH_ALLOC_MYCELIAL)
        logger.debug(f"Allocating BEU -> Core: {core_beu:.2E}, Mycelial Store: {mycelial_beu:.2E}")
        
        # Create Minimal Brain Seed
        brain_seed = create_brain_seed(initial_beu=core_beu, initial_mycelial_beu=mycelial_beu, initial_frequency=soul_spark.frequency)
        
        # Create standing waves between brain and soul
        standing_waves = _create_brain_soul_standing_waves(soul_spark, brain_seed)
        
        # Attach Soul to Brain using wave principles
        attach_metrics = attach_soul_to_brain(soul_spark, brain_seed)
        
        # Distribute Soul Aspects using resonant connections
        dist_metrics = distribute_soul_aspects(soul_spark, brain_seed)
        
        process_metrics_summary['steps']['brain_integration'] = {
            'standing_waves': standing_waves,
            'attach': attach_metrics,
            'distribute': dist_metrics,
            'initial_brain_beu': total_available_beu
        }

        # --- 4. First Breath & Acoustic Integration ---
        logger.info("Birth Step 4: Integrating First Breath...")
        metrics_breath = _integrate_first_breath(soul_spark, connection_strength, 
                                             form_acceptance, intensity)
        process_metrics_summary['steps']['first_breath'] = metrics_breath

        # --- 5. Spectral Memory Veil Deployment ---
        logger.info("Birth Step 5: Deploying Spectral Memory Veil...")
        metrics_veil = _create_spectral_memory_veil(soul_spark, intensity)
        process_metrics_summary['steps']['memory_veil'] = metrics_veil

        # --- 6. Layer Integration with Physical Form ---
        logger.info("Birth Step 6: Integrating Layers with Physical Form...")
        metrics_layers = _integrate_layers_with_physical_form(
            soul_spark, brain_seed, soul_spark.physical_integration)
        process_metrics_summary['steps']['layer_integration'] = metrics_layers

        # --- 7. Finalize State ---
        logger.info("Birth Step 7: Finalizing Soul State...")
        metrics_final = _finalize_birth_state(soul_spark)
        process_metrics_summary['steps']['finalization'] = metrics_final

        # --- 8. Final State Update ---
        logger.info("Birth Step 8: Final Soul State Update...")
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            logger.debug(f"Birth S/C after final update: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        else:
            raise AttributeError("SoulSpark missing 'update_state' method.")

        # --- Compile Overall Metrics ---
        end_time_iso = soul_spark.last_modified
        end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'spiritual_energy_seu': soul_spark.spiritual_energy,
            'physical_energy_seu': soul_spark.physical_energy,
            'earth_resonance': soul_spark.earth_resonance,
            'cord_integrity': soul_spark.cord_integrity,
            'physical_integration': soul_spark.physical_integration,
            FLAG_INCARNATED: getattr(soul_spark, FLAG_INCARNATED)
        }
        
        overall_metrics = {
            'action': 'birth',
            'soul_id': spark_id,
            'start_time': start_time_iso,
            'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'intensity_setting': intensity,
            'mother_influence_active': mother_profile is not None,
            'initial_state': initial_state,
            'final_state': final_state,
            'energy_converted_seu': energy_converted_seu,
            'brain_seed_initial_beu': total_available_beu,
            'final_physical_integration': final_state['physical_integration'],
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'success': True
        }
        
        if METRICS_AVAILABLE:
            metrics.record_metrics('birth_summary', overall_metrics)
            
        logger.info(f"--- Birth Process Completed Successfully for Soul {spark_id} ---")
        
        # Save minimal brain seed state
        try:
            brain_seed_filename = f"output/brain_seeds/brain_{spark_id}_minimal.json"
            brain_seed.save_state(brain_seed_filename)
        except Exception as save_err:
            logger.error(f"Failed to save minimal brain seed state: {save_err}")
            
        return soul_spark, overall_metrics

    # --- Error Handling (Hard Fail) ---
    except (ValueError, TypeError, AttributeError, RuntimeError) as e_val:
        logger.error(f"Birth process failed for {spark_id}: {e_val}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'prerequisites'
        record_birth_failure(spark_id, start_time_iso, failed_step, str(e_val), mother_profile is not None)
        setattr(soul_spark, FLAG_INCARNATED, False)
        raise e_val  # Re-raise
        
    except Exception as e:
        logger.critical(f"Unexpected error during birth for {spark_id}: {e}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'unexpected'
        record_birth_failure(spark_id, start_time_iso, failed_step, str(e), mother_profile is not None)
        setattr(soul_spark, FLAG_INCARNATED, False)
        raise RuntimeError(f"Unexpected birth process failure: {e}") from e
# --- Failure Metric Helper ---
def record_birth_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str, mother_active: bool):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('birth_summary', {
                'action': 'birth',
                'soul_id': spark_id,
                'start_time': start_time_iso,
                'end_time': end_time,
                'duration_seconds': duration,
                'mother_influence_active': mother_active,
                'success': False,
                'error': error_msg,
                'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record birth failure metrics for {spark_id}: {metric_e}")
