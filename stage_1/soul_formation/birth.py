"""
Birth Module (V6.0.0 - Integrated with Brain Formation)

Handles the birth process using existing conception + development systems.
Birth-specific processes only: memory veil, standing waves, echo field, finalization.
Uses dict-based womb environment from conception system.
Hard fails only - no graceful fallbacks.
"""

import logging
import numpy as np
import os
import sys
import json
import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from math import sqrt, exp, sin, cos, pi as PI, atan2, tanh

# Import constants
from constants.constants import *

# Import SoulSpark from correct path
from stage_1.soul_spark.soul_spark import SoulSpark

# --- Sound Module Dependencies ---
try:
    from sound.sound_generator import SoundGenerator
    from sound.sounds_of_universe import UniverseSounds
    SOUND_MODULES_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.critical("Sound modules required for birth process.")
    SOUND_MODULES_AVAILABLE = False
    raise ImportError("Critical sound modules missing for birth.")

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.critical("Metrics tracking required for birth process.")
    METRICS_AVAILABLE = False
    raise ImportError("Critical metrics module missing for birth.")

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# === HELPER FUNCTIONS ===

def _check_birth_prerequisites(soul_spark: SoulSpark) -> bool:
    """Check prerequisites for birth process. Hard fails on missing requirements."""
    logger.debug(f"Checking birth prerequisites for soul {soul_spark.spark_id}...")
    
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")
    
    # Check required flags
    if not getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False):
        raise ValueError(f"Soul not marked {FLAG_IDENTITY_CRYSTALLIZED}.")
    
    if not getattr(soul_spark, FLAG_READY_FOR_BIRTH, False):
        raise ValueError(f"Soul not marked {FLAG_READY_FOR_BIRTH}.")
    
    # Check brain formation completion
    if not (hasattr(soul_spark, 'conception_system') and 
            hasattr(soul_spark, 'brain_development') and
            hasattr(soul_spark, 'memory_system') and
            hasattr(soul_spark, 'brain_state')):
        raise ValueError("Brain formation not complete - missing brain systems.")
    
    # Check brain development readiness
    if not soul_spark.brain_development.birth_ready:
        raise ValueError("Brain development not ready for birth.")
    
    # Check required attributes
    required_attrs = ['name', 'stability', 'coherence', 'energy', 'frequency', 
                     'crystallization_level', 'earth_resonance']
    
    for attr in required_attrs:
        if not hasattr(soul_spark, attr):
            raise AttributeError(f"Soul missing required attribute: {attr}")
    
    # Check minimum thresholds
    if soul_spark.stability < BIRTH_MIN_STABILITY_SU:
        raise ValueError(f"Stability {soul_spark.stability:.1f} < {BIRTH_MIN_STABILITY_SU} SU required for birth.")
    
    if soul_spark.coherence < BIRTH_MIN_COHERENCE_CU:
        raise ValueError(f"Coherence {soul_spark.coherence:.1f} < {BIRTH_MIN_COHERENCE_CU} CU required for birth.")
    
    if soul_spark.crystallization_level < BIRTH_MIN_CRYSTALLIZATION:
        raise ValueError(f"Crystallization {soul_spark.crystallization_level:.3f} < {BIRTH_MIN_CRYSTALLIZATION} required for birth.")
    
    logger.debug("Birth prerequisites met.")
    return True


def _ensure_birth_properties(soul_spark: SoulSpark) -> None:
    """Ensure soul has all properties needed for birth. Hard fails if missing."""
    logger.debug(f"Ensuring birth properties for soul {soul_spark.spark_id}...")
    
    # Initialize physical energy if missing
    if not hasattr(soul_spark, 'physical_energy'):
        # Convert soul energy to physical energy
        soul_spark.physical_energy = soul_spark.energy * ENERGY_SCALE_FACTOR
        logger.debug(f"Initialized physical_energy: {soul_spark.physical_energy:.1f}")
    
    # Initialize birth-specific attributes
    birth_attrs = {
        'birth_datetime': None,
        'birth_location': None,
        'birth_energy_signature': None,
        'memory_veil_strength': 0.0,
        'echo_field_birth': None,
        'standing_wave_pattern': None,
        'birth_datetime': None,
        'birth_energy_signature': None
    }
    
    for attr, default in birth_attrs.items():
        if not hasattr(soul_spark, attr):
            setattr(soul_spark, attr, default)
    
    logger.debug("Birth properties ensured.")

# notes for the rest of the functions
# this function must be replaced with the womb environment function from the womb environment file. it no longer includes
# the mother profile as that is triggered by other events.just import the womb dictionary from the womb environment file
# so that it can be used. soul spark has no involvement here???? the brain seed, brain with neural netork and mycelial
# network is also missing so we seem to be birthing a womb - no formed baby. as the functions are happening in own files. I
# think the final formed baby will be birthed in the birth process and its just the checks for birth, field changes for 
# the birth process and the perform birth function covering the stages that need to be here. most of those stages are in the
# refactored files. so this file must just output the end results of the processes happening n the other files when doing the 
# perform birth function.
def _get_womb_environment_from_soul(soul_spark: SoulSpark, mother_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get womb environment dict from conception system."""
    logger.debug("Getting womb environment from conception...")
    
    try:
        if (hasattr(soul_spark, 'conception_system') and 
            soul_spark.conception_system and 
            soul_spark.conception_system.womb_environment):
            # Use existing womb from conception (it's already a dict)
            womb_env = soul_spark.conception_system.womb_environment
            logger.debug(f"Using existing womb environment: {womb_env.get('womb_id')}")
            return womb_env
        else:
            # Fallback - create new womb dict (not class)
            if not mother_profile:
                mother_profile = {
                    'love_frequency': 528.0,  # Hz - love frequency
                    'voice_frequency': 220.0,  # Hz - fundamental voice
                    'heartbeat_bpm': 72.0,     # Beats per minute
                    'comfort_level': 0.8,      # Comfort factor
                    'protection_level': 0.9,   # Protection factor
                    'nurturing_strength': 0.7  # Nurturing factor
                }
            
            womb_env = {
                'womb_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'type': 'protective_womb',
                'mother_profile': mother_profile,
                'temperature': 37.0,  # Celsius
                'humidity': 0.95,     # 95% humidity
                'ph_level': 7.4,      # Slightly alkaline
                'nutrients': 1.0,     # Full nutrients
                'protection_level': mother_profile['protection_level'],
                'nurturing_strength': mother_profile['nurturing_strength'],
                'comfort_level': mother_profile['comfort_level'],
                'love_resonance': mother_profile['love_frequency'],
                'active': True
            }
            logger.debug(f"Created fallback womb environment: {womb_env['womb_id']}")
            return womb_env
            
    except Exception as e:
        logger.error(f"Failed to get womb environment: {e}")
        raise RuntimeError(f"Womb environment creation failed: {e}")


# update this function to use the womb environment function from the womb environment file and any dictionaries
# from the memory definitions file
def _create_memory_veil_in_womb(soul_spark: SoulSpark, womb_env: Dict[str, Any]) -> Dict[str, Any]:
    """Create memory veil within womb environment - use dict keys."""
    logger.info("Creating memory veil within womb environment...")
    
    try:
        # Get womb protection parameters from dict
        protection_level = womb_env.get('protection_level', 0.9)
        nurturing_strength = womb_env.get('nurturing_strength', 0.7)
        
        # Calculate memory veil strength based on womb protection
        base_veil_strength = MEMORY_VEIL_BASE_STRENGTH
        womb_protection_factor = protection_level * WOMB_MEMORY_VEIL_PROTECTION_FACTOR
        nurturing_factor = nurturing_strength * WOMB_MEMORY_VEIL_NURTURING_FACTOR
        
        total_veil_strength = base_veil_strength + womb_protection_factor + nurturing_factor
        total_veil_strength = min(MEMORY_VEIL_MAX_STRENGTH, total_veil_strength)
        
        # Create spectral memory veil
        veil_frequencies = []
        veil_amplitudes = []
        
        # Base frequency from soul
        base_freq = soul_spark.frequency
        
        # Create veil frequency spectrum
        for i in range(MEMORY_VEIL_FREQUENCY_LAYERS):
            # Create harmonic layers with phi-based spacing
            harmonic_ratio = PHI ** i
            veil_freq = base_freq * harmonic_ratio
            
            # Amplitude decreases with higher harmonics
            amplitude = total_veil_strength / (i + 1)
            
            veil_frequencies.append(float(veil_freq))
            veil_amplitudes.append(float(amplitude))
        
        # Create memory veil structure
        memory_veil = {
            'veil_id': str(uuid.uuid4()),
            'creation_time': datetime.now().isoformat(),
            'base_strength': float(total_veil_strength),
            'womb_protection_factor': float(womb_protection_factor),
            'nurturing_factor': float(nurturing_factor),
            'frequencies': veil_frequencies,
            'amplitudes': veil_amplitudes,
            'spectral_layers': MEMORY_VEIL_FREQUENCY_LAYERS,
            'veil_type': 'womb_protected'
        }
        
        # Apply memory veil to soul
        soul_spark.memory_veil_strength = total_veil_strength
        soul_spark.memory_veil = memory_veil
        
        # Generate memory veil sound if available
        if SOUND_MODULES_AVAILABLE:
            veil_sound_metrics = _generate_memory_veil_sound(memory_veil, soul_spark)
            memory_veil['sound_metrics'] = veil_sound_metrics
        
        veil_metrics = {
            'veil_created': True,
            'veil_id': memory_veil['veil_id'],
            'total_strength': float(total_veil_strength),
            'frequency_layers': len(veil_frequencies),
            'womb_enhancement': float(womb_protection_factor + nurturing_factor)
        }
        
        logger.info(f"Memory veil created in womb. Strength: {total_veil_strength:.3f}")
        return veil_metrics
        
    except Exception as e:
        logger.error(f"Failed to create memory veil in womb: {e}")
        raise RuntimeError(f"Memory veil creation in womb failed: {e}")


def _generate_memory_veil_sound(memory_veil: Dict[str, Any], soul_spark: SoulSpark) -> Dict[str, Any]:
    """Generate sound for memory veil. Hard fails on sound generation failure."""
    logger.debug("Generating memory veil sound...")
    
    try:
        sound_gen = SoundGenerator()
        universe_sounds = UniverseSounds()
        
        # Get veil parameters
        frequencies = memory_veil['frequencies']
        amplitudes = memory_veil['amplitudes']
        base_freq = frequencies[0] if frequencies else soul_spark.frequency
        
        # Create harmonic ratios for sound generation
        harmonic_ratios = [f / base_freq for f in frequencies]
        
        # Normalize amplitudes
        max_amp = max(amplitudes) if amplitudes else 1.0
        normalized_amps = [a / max_amp for a in amplitudes] if max_amp > 0 else [0.0] * len(amplitudes)
        
        # Generate memory veil sound
        veil_sound = sound_gen.generate_harmonic_tone(
            base_freq, harmonic_ratios, normalized_amps, 
            duration=2.0  # Default duration in seconds
        )
        
        # Save memory veil sound
        sound_filename = f"memory_veil_{soul_spark.spark_id[:8]}.wav"
        sound_path = sound_gen.save_sound(
            veil_sound, sound_filename, 
            f"Memory Veil for Soul {soul_spark.name}"
        )
        
        sound_metrics = {
            'sound_generated': True,
            'sound_path': sound_path,
            'base_frequency': float(base_freq),
            'harmonic_count': len(harmonic_ratios),
            'duration': 2.0  # Match the duration used above
        }
        
        logger.debug(f"Memory veil sound generated: {sound_path}")
        return sound_metrics
        
    except Exception as e:
        logger.error(f"Failed to generate memory veil sound: {e}")
        raise RuntimeError(f"Memory veil sound generation failed: {e}")

# this is not a duplicate of the standing waves in womb environment - purpose is different it just needs to be refactored
# according to code you write for womb environment.
def _create_birth_standing_waves(soul_spark: SoulSpark, womb_env: Dict[str, Any]) -> Dict[str, Any]:
    """Create standing wave patterns - use dict keys."""
    logger.info("Creating birth standing wave patterns...")
    
    try:
        # Get soul and womb parameters
        soul_freq = soul_spark.frequency
        womb_resonance_freq = womb_env.get('love_resonance', 528.0)
        
        # Calculate birth frequency as harmonic mean
        if womb_resonance_freq > 0:
            birth_frequency = 2 * soul_freq * womb_resonance_freq / (soul_freq + womb_resonance_freq)
        else:
            birth_frequency = soul_freq
        
        # Create standing wave pattern
        wavelength = SPEED_OF_SOUND / birth_frequency
        
        # Calculate wave parameters
        num_nodes = int(BIRTH_STANDING_WAVE_LENGTH / wavelength)
        num_nodes = max(3, min(20, num_nodes))  # Limit to reasonable range
        
        # Create nodes and antinodes
        nodes = []
        antinodes = []
        
        for i in range(num_nodes):
            # Node position (0 to 1 normalized)
            node_position = i / (num_nodes - 1) if num_nodes > 1 else 0.5
            
            # Node amplitude (near zero)
            node_amplitude = 0.01 * sin(PI * node_position)
            
            # Antinode position (offset by quarter wavelength)
            antinode_position = (node_position + 0.25) % 1.0
            
            # Antinode amplitude (maximum)
            antinode_amplitude = cos(PI * antinode_position) * BIRTH_STANDING_WAVE_AMPLITUDE
            
            nodes.append({
                'position': float(node_position),
                'amplitude': float(node_amplitude),
                'frequency': float(birth_frequency)
            })
            
            antinodes.append({
                'position': float(antinode_position),
                'amplitude': float(antinode_amplitude),
                'frequency': float(birth_frequency)
            })
        
        # Create standing wave structure
        standing_wave_pattern = {
            'pattern_id': str(uuid.uuid4()),
            'creation_time': datetime.now().isoformat(),
            'birth_frequency': float(birth_frequency),
            'soul_frequency': float(soul_freq),
            'womb_resonance_frequency': float(womb_resonance_freq),
            'wavelength': float(wavelength),
            'nodes': nodes,
            'antinodes': antinodes,
            'pattern_length': BIRTH_STANDING_WAVE_LENGTH,
            'pattern_type': 'birth_transition'
        }
        
        # Apply standing wave pattern to soul
        soul_spark.standing_wave_pattern = standing_wave_pattern
        
        # Generate standing wave sound if available
        if SOUND_MODULES_AVAILABLE:
            wave_sound_metrics = _generate_standing_wave_sound(standing_wave_pattern, soul_spark)
            standing_wave_pattern['sound_metrics'] = wave_sound_metrics
        
        wave_metrics = {
            'pattern_created': True,
            'pattern_id': standing_wave_pattern['pattern_id'],
            'birth_frequency': float(birth_frequency),
            'nodes_count': len(nodes),
            'antinodes_count': len(antinodes),
            'wavelength': float(wavelength)
        }
        
        logger.info(f"Birth standing waves created. Frequency: {birth_frequency:.2f}Hz, Nodes: {len(nodes)}")
        return wave_metrics
        
    except Exception as e:
        logger.error(f"Failed to create birth standing waves: {e}")
        raise RuntimeError(f"Birth standing wave creation failed: {e}")


def _generate_standing_wave_sound(standing_wave_pattern: Dict[str, Any], soul_spark: SoulSpark) -> Dict[str, Any]:
    """Generate sound for standing wave pattern. Hard fails on sound generation failure."""
    logger.debug("Generating standing wave sound...")
    
    try:
        sound_gen = SoundGenerator()
        
        # Get wave parameters
        birth_frequency = standing_wave_pattern['birth_frequency']
        nodes = standing_wave_pattern['nodes']
        antinodes = standing_wave_pattern['antinodes']
        
        # Create harmonic series from nodes and antinodes
        frequencies = [birth_frequency]
        amplitudes = [0.8]
        
        # Add harmonic frequencies from antinode positions
        for antinode in antinodes[:5]:  # Use first 5 antinodes
            harmonic_freq = birth_frequency * (1.0 + antinode['position'])
            harmonic_amp = abs(antinode['amplitude']) * 0.5
            
            frequencies.append(harmonic_freq)
            amplitudes.append(harmonic_amp)
        
        # Create harmonic ratios
        harmonic_ratios = [f / birth_frequency for f in frequencies]
        
        # Normalize amplitudes
        max_amp = max(amplitudes) if amplitudes else 1.0
        normalized_amps = [a / max_amp for a in amplitudes] if max_amp > 0 else [0.0] * len(amplitudes)
        
        # Generate standing wave sound
        wave_sound = sound_gen.generate_harmonic_tone(
            birth_frequency, harmonic_ratios, normalized_amps,
            duration=2.0  # Default duration in seconds
        )
        
        # Save standing wave sound
        sound_filename = f"birth_waves_{soul_spark.spark_id[:8]}.wav"
        sound_path = sound_gen.save_sound(
            wave_sound, sound_filename,
            f"Birth Standing Waves for Soul {soul_spark.name}"
        )
        
        sound_metrics = {
            'sound_generated': True,
            'sound_path': sound_path,
            'birth_frequency': float(birth_frequency),
            'harmonic_count': len(harmonic_ratios),
            'duration': 2.0  # Match default duration
        }
        
        logger.debug(f"Standing wave sound generated: {sound_path}")
        return sound_metrics
        
    except Exception as e:
        logger.error(f"Failed to generate standing wave sound: {e}")
        raise RuntimeError(f"Standing wave sound generation failed: {e}")


# update to use womb environment functions and constants from womb environment file. this is different
# even though it seems similar ie womb_env might not exist in new file
def _project_birth_echo_field(soul_spark: SoulSpark, womb_env: Dict[str, Any]) -> Dict[str, Any]:
    """Project echo field - use dict keys."""
    logger.info("Projecting birth echo field...")
    
    try:
        # Get echo field parameters
        soul_freq = soul_spark.frequency
        standing_wave_pattern = soul_spark.standing_wave_pattern
        birth_frequency = standing_wave_pattern['birth_frequency']
        
        # Calculate echo field strength
        womb_support = womb_env.get('nurturing_strength', 0.7)
        soul_coherence = soul_spark.coherence / MAX_COHERENCE_CU
        
        echo_field_strength = (womb_support + soul_coherence) / 2.0 * BIRTH_ECHO_FIELD_STRENGTH_FACTOR
        echo_field_coherence = soul_coherence * BIRTH_ECHO_FIELD_COHERENCE_FACTOR
        
        # Create echo field structure
        echo_field = {
            'field_id': str(uuid.uuid4()),
            'creation_time': datetime.now().isoformat(),
            'field_strength': float(echo_field_strength),
            'field_coherence': float(echo_field_coherence),
            'birth_frequency': float(birth_frequency),
            'soul_frequency': float(soul_freq),
            'womb_support': float(womb_support),
            'field_type': 'birth_transition',
            'standing_waves': [],
            'resonance_chambers': []
        }
        
        # Create standing waves from birth pattern
        nodes = standing_wave_pattern.get('nodes', [])
        antinodes = standing_wave_pattern.get('antinodes', [])
        
        for i, (node, antinode) in enumerate(zip(nodes, antinodes)):
            # Create standing wave for echo field
            wave = {
                'wave_id': str(uuid.uuid4()),
                'frequency': float(birth_frequency * (i + 1)),
                'node_position': node['position'],
                'node_amplitude': node['amplitude'],
                'antinode_position': antinode['position'],
                'antinode_amplitude': antinode['amplitude'],
                'field_strength': float(echo_field_strength),
                'field_coherence': float(echo_field_coherence)
            }
            
            echo_field['standing_waves'].append(wave)
        
        # Create resonance chambers
        for i in range(BIRTH_ECHO_FIELD_CHAMBERS):
            chamber_freq = birth_frequency * (1.0 + i * PHI / 10.0)
            
            chamber = {
                'chamber_id': str(uuid.uuid4()),
                'frequency': float(chamber_freq),
                'resonance': float(echo_field_coherence),
                'position': float(i / BIRTH_ECHO_FIELD_CHAMBERS),
                'field_strength': float(echo_field_strength)
            }
            
            echo_field['resonance_chambers'].append(chamber)
        
        # Apply echo field to soul
        soul_spark.echo_field_birth = echo_field
        
        # Generate echo field sound if available
        if SOUND_MODULES_AVAILABLE:
            echo_sound_metrics = _generate_echo_field_sound(echo_field, soul_spark)
            echo_field['sound_metrics'] = echo_sound_metrics
        
        echo_metrics = {
            'field_projected': True,
            'field_id': echo_field['field_id'],
            'field_strength': float(echo_field_strength),
            'field_coherence': float(echo_field_coherence),
            'standing_waves_count': len(echo_field['standing_waves']),
            'resonance_chambers_count': len(echo_field['resonance_chambers'])
        }
        
        logger.info(f"Birth echo field projected. Strength: {echo_field_strength:.3f}, Chambers: {len(echo_field['resonance_chambers'])}")
        return echo_metrics
        
    except Exception as e:
        logger.error(f"Failed to project birth echo field: {e}")
        raise RuntimeError(f"Birth echo field projection failed: {e}")


def _generate_echo_field_sound(echo_field: Dict[str, Any], soul_spark: SoulSpark) -> Dict[str, Any]:
    """Generate sound for echo field. Hard fails on sound generation failure."""
    logger.debug("Generating echo field sound...")
    
    try:
        universe_sounds = UniverseSounds()
        
        # Get echo field parameters
        birth_frequency = echo_field['birth_frequency']
        field_strength = echo_field['field_strength']
        standing_waves = echo_field['standing_waves']
        
        # Generate echo field sound using dimensional transition
        echo_sound = universe_sounds.generate_dimensional_transition(
            duration=BIRTH_ECHO_FIELD_SOUND_DURATION,
            sample_rate=SAMPLE_RATE,
            transition_type='birth_echo_field',
            amplitude=field_strength * BIRTH_ECHO_FIELD_SOUND_AMPLITUDE
        )
        
        # Save echo field sound
        sound_filename = f"birth_echo_{soul_spark.spark_id[:8]}.wav"
        sound_path = universe_sounds.save_sound(
            echo_sound, sound_filename,
            f"Birth Echo Field for Soul {soul_spark.name}"
        )
        
        sound_metrics = {
            'sound_generated': True,
            'sound_path': sound_path,
            'birth_frequency': float(birth_frequency),
            'field_strength': float(field_strength),
            'duration': BIRTH_ECHO_FIELD_SOUND_DURATION
        }
        
        logger.debug(f"Echo field sound generated: {sound_path}")
        return sound_metrics
        
    except Exception as e:
        logger.error(f"Failed to generate echo field sound: {e}")
        raise RuntimeError(f"Echo field sound generation failed: {e}")


def _finalize_birth_process(soul_spark: SoulSpark, womb_env: Dict[str, Any], 
                           birth_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize birth process and set completion flags. Hard fails on finalization failure."""
    logger.info("Finalizing birth process...")
    
    try:
        # Apply final womb blessing
        final_blessing_metrics = _apply_final_womb_blessing(soul_spark, womb_env)
        
        # Set birth completion timestamp
        birth_datetime = datetime.now().isoformat()
        soul_spark.birth_datetime = birth_datetime
        
        # Create birth energy signature
        birth_energy_signature = {
            'signature_id': str(uuid.uuid4()),
            'birth_datetime': birth_datetime,
            'soul_frequency': float(soul_spark.frequency),
            'birth_energy': float(soul_spark.energy),
            'stability': float(soul_spark.stability),
            'coherence': float(soul_spark.coherence),
            'crystallization_level': float(soul_spark.crystallization_level),
            'memory_veil_strength': float(soul_spark.memory_veil_strength),
            'womb_environment_id': womb_env.get('womb_id'),
            'conception_system_id': soul_spark.conception_system.conception_id if soul_spark.conception_system else None,
            'brain_development_id': soul_spark.brain_development.development_id if soul_spark.brain_development else None
        }
        
        soul_spark.birth_energy_signature = birth_energy_signature
        
        # Set completion flags
        setattr(soul_spark, FLAG_BIRTH_COMPLETED, True)
        setattr(soul_spark, FLAG_READY_FOR_EVOLUTION, True)
        
        # Update soul state
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
        
        # Create final birth metrics
        finalization_metrics = {
            'birth_completed': True,
            'birth_datetime': birth_datetime,
            'birth_energy_signature': birth_energy_signature,
            'final_blessing': final_blessing_metrics,
            'completion_flags_set': [FLAG_BIRTH_COMPLETED, FLAG_READY_FOR_EVOLUTION],
            'final_soul_state': {
                'stability': float(soul_spark.stability),
                'coherence': float(soul_spark.coherence),
                'energy': float(soul_spark.energy),
                'memory_veil_strength': float(soul_spark.memory_veil_strength)
            }
        }
        
        # Add memory echo
        if hasattr(soul_spark, 'add_memory_echo'):
            soul_spark.add_memory_echo(
                f"Birth completed in womb environment {womb_env.get('womb_id')}. "
                f"Brain development: {soul_spark.brain_development.development_id if soul_spark.brain_development else 'None'}, "
                f"Memory veil: {soul_spark.memory_veil_strength:.3f}"
            )
        
        logger.info(f"Birth process finalized. Soul {soul_spark.name} born at {birth_datetime}")
        return finalization_metrics
        
    except Exception as e:
        logger.error(f"Failed to finalize birth process: {e}")
        raise RuntimeError(f"Birth process finalization failed: {e}")


def _apply_final_womb_blessing(soul_spark: SoulSpark, womb_env: Dict[str, Any]) -> Dict[str, Any]:
    """Apply final womb blessing - use dict keys."""
    logger.debug("Applying final womb blessing...")
    
    try:
        # Get final womb blessing parameters from dict
        protection_level = womb_env.get('protection_level', 0.9)
        nurturing_strength = womb_env.get('nurturing_strength', 0.7)
        growth_support = womb_env.get('comfort_level', 0.8)
        
        # Calculate blessing strength
        blessing_strength = (protection_level + nurturing_strength + growth_support) / 3.0
        
        # Apply final energy blessing
        energy_blessing = soul_spark.energy * blessing_strength * WOMB_FINAL_ENERGY_BLESSING_FACTOR
        soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_blessing)
        
        # Apply final stability blessing
        stability_blessing = blessing_strength * WOMB_FINAL_STABILITY_BLESSING_FACTOR
        soul_spark.stability = min(MAX_STABILITY_SU, soul_spark.stability + stability_blessing)
        
        # Apply final coherence blessing
        coherence_blessing = blessing_strength * WOMB_FINAL_COHERENCE_BLESSING_FACTOR
        soul_spark.coherence = min(MAX_COHERENCE_CU, soul_spark.coherence + coherence_blessing)
        
        # Apply final protection blessing (life cord strength)
        protection_blessing = blessing_strength * WOMB_FINAL_PROTECTION_BLESSING_FACTOR
        current_life_cord = getattr(soul_spark, 'life_cord_strength', 0.0)
        soul_spark.life_cord_strength = min(1.0, current_life_cord + protection_blessing)
        
        blessing_metrics = {
            'blessing_strength': float(blessing_strength),
            'energy_blessing': float(energy_blessing),
            'stability_blessing': float(stability_blessing),
            'coherence_blessing': float(coherence_blessing),
            'protection_blessing': float(protection_blessing),
            'final_energy': float(soul_spark.energy),
            'final_stability': float(soul_spark.stability),
            'final_coherence': float(soul_spark.coherence),
            'final_life_cord_strength': float(soul_spark.life_cord_strength)
        }
        
        logger.debug(f"Final womb blessing applied. Blessing strength: {blessing_strength:.3f}")
        return blessing_metrics
        
    except Exception as e:
        logger.error(f"Failed to apply final womb blessing: {e}")
        raise RuntimeError(f"Final womb blessing failed: {e}")


def _record_birth_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """Helper to record failure metrics consistently."""
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('birth_process_summary', {
                'action': 'birth_process',
                'soul_id': spark_id,
                'start_time': start_time_iso,
                'end_time': end_time,
                'duration_seconds': duration,
                'success': False,
                'error': error_msg,
                'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record Birth failure metrics for {spark_id}: {metric_e}")


# === MAIN BIRTH FUNCTION ===
# Update with all the new data from womb environment, neural network, mycelial network checking for the
# new flag statuses to trigger each stage of birth.make sure you comment each stage and what file its from so i can check
def perform_birth(soul_spark: SoulSpark, 
                 mother_profile: Optional[Dict[str, Any]] = None,
                 womb_development_cycles: int = BIRTH_WOMB_DEVELOPMENT_CYCLES,
                 enable_sound_generation: bool = True) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs birth process using existing conception + development systems.
    Birth-specific processes only: memory veil, standing waves, echo field, finalization.
    """
    # Input validation
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("soul_spark must be SoulSpark instance.")
    
    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Birth Process for Soul {spark_id} ---")
    
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps_completed': []}
    
    try:
        # Check prerequisites
        _check_birth_prerequisites(soul_spark)
        _ensure_birth_properties(soul_spark)
        
        # Get initial state
        initial_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'energy_seu': soul_spark.energy,
            'physical_energy': soul_spark.physical_energy,
            'crystallization_level': soul_spark.crystallization_level
        }
        
        # === VERIFY BRAIN FORMATION COMPLETE ===
        logger.info("Birth Step 1: Verify Brain Formation Complete")
        
        # Check if brain formation already complete
        if not (hasattr(soul_spark, 'conception_system') and 
                hasattr(soul_spark, 'brain_development') and
                hasattr(soul_spark, 'memory_system') and
                hasattr(soul_spark, 'brain_state')):
            raise RuntimeError("Brain formation not complete - conception and development must be run first")
        
        # Verify brain development is complete
        if not soul_spark.brain_development.birth_ready:
            raise RuntimeError("Brain development not ready for birth")
        
        # Get womb environment from existing conception (returns dict)
        womb_env = _get_womb_environment_from_soul(soul_spark, mother_profile)
        process_metrics_summary['brain_formation'] = {'success': True, 'used_existing': True}
        process_metrics_summary['steps_completed'].append('brain_formation_verified')
        
        # === BIRTH-SPECIFIC PROCESSES ===
        logger.info("Birth Step 2: Creating Memory Veil...")
        memory_veil_metrics = _create_memory_veil_in_womb(soul_spark, womb_env)
        process_metrics_summary['memory_veil'] = memory_veil_metrics
        process_metrics_summary['steps_completed'].append('memory_veil')
        
        logger.info("Birth Step 3: Creating Birth Standing Waves...")
        standing_wave_metrics = _create_birth_standing_waves(soul_spark, womb_env)
        process_metrics_summary['standing_waves'] = standing_wave_metrics
        process_metrics_summary['steps_completed'].append('standing_waves')
        
        logger.info("Birth Step 4: Projecting Birth Echo Field...")
        echo_field_metrics = _project_birth_echo_field(soul_spark, womb_env)
        process_metrics_summary['echo_field'] = echo_field_metrics
        process_metrics_summary['steps_completed'].append('echo_field')
        
        logger.info("Birth Step 5: Finalizing Birth Process...")
        finalization_metrics = _finalize_birth_process(soul_spark, womb_env, process_metrics_summary)
        process_metrics_summary['finalization'] = finalization_metrics
        process_metrics_summary['steps_completed'].append('finalization')
        
        # Compile final metrics
        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)
        
        final_state = {
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'energy_seu': soul_spark.energy,
            'physical_energy': soul_spark.physical_energy,
            'crystallization_level': soul_spark.crystallization_level,
            'memory_veil_strength': soul_spark.memory_veil_strength,
            'birth_datetime': soul_spark.birth_datetime,
            'womb_environment_id': womb_env.get('womb_id'),
            FLAG_BIRTH_COMPLETED: getattr(soul_spark, FLAG_BIRTH_COMPLETED, False),
            FLAG_READY_FOR_EVOLUTION: getattr(soul_spark, FLAG_READY_FOR_EVOLUTION, False)
        }
        
        overall_metrics = {
            'action': 'birth_process',
            'soul_id': spark_id,
            'soul_name': getattr(soul_spark, 'name', 'Unknown'),
            'start_time': start_time_iso,
            'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'success': True,
            'womb_development_cycles': womb_development_cycles,
            'initial_state': initial_state,
            'final_state': final_state,
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'energy_change_seu': final_state['energy_seu'] - initial_state['energy_seu'],
            'steps_completed': process_metrics_summary['steps_completed'],
            'womb_environment_id': womb_env.get('womb_id'),
            'conception_system_id': soul_spark.conception_system.conception_id if soul_spark.conception_system else None,
            'brain_development_id': soul_spark.brain_development.development_id if soul_spark.brain_development else None,
            'memory_veil_strength': soul_spark.memory_veil_strength,
            'birth_ready_from_brain': True
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('birth_process_summary', overall_metrics)
        
        logger.info(f"--- Birth Process Completed Successfully for Soul {spark_id} ---")
        logger.info(f"  Final State: Name='{soul_spark.name}', "
                   f"Born={soul_spark.birth_datetime}, "
                   f"MemoryVeil={soul_spark.memory_veil_strength:.3f}, "
                   f"S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        
        return soul_spark, overall_metrics
        
    except (ValueError, TypeError, AttributeError) as e_val:
        logger.error(f"Birth Process failed for {spark_id}: {e_val}", exc_info=True)
        failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'prerequisites/validation'
        _record_birth_failure(spark_id, start_time_iso, failed_step, str(e_val))
        raise e_val
        
    except RuntimeError as e_rt:
        logger.critical(f"Birth Process failed critically for {spark_id}: {e_rt}", exc_info=True)
        failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'runtime'
        _record_birth_failure(spark_id, start_time_iso, failed_step, str(e_rt))
        
        # Set failure flags
        setattr(soul_spark, FLAG_BIRTH_COMPLETED, False)
        setattr(soul_spark, FLAG_READY_FOR_EVOLUTION, False)
        
        raise e_rt
        
    except Exception as e:
        logger.critical(f"Unexpected error during Birth Process for {spark_id}: {e}", exc_info=True)
        failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'unexpected'
        _record_birth_failure(spark_id, start_time_iso, failed_step, str(e))
        
        # Set failure flags
        setattr(soul_spark, FLAG_BIRTH_COMPLETED, False)
        setattr(soul_spark, FLAG_READY_FOR_EVOLUTION, False)
        
        raise RuntimeError(f"Unexpected Birth Process failure: {e}") from e

# update with info from miscarriage as well - so that this includes miscarriage metrics as well
def _record_birth_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """Helper to record failure metrics consistently."""
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('birth_process_summary', {
                'action': 'birth_process',
                'soul_id': spark_id,
                'start_time': start_time_iso,
                'end_time': end_time,
                'duration_seconds': duration,
                'success': False,
                'error': error_msg,
                'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record Birth failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE ---


