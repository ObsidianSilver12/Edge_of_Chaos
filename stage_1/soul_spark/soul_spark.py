# --- START OF FILE src/stage_1/soul_formation/soul_spark.py ---

"""
Soul Spark Module (Refactored V4.3.7 - Emergence Principle, Strict PEP8)

Defines the SoulSpark class. Initialization relies on data derived
from field emergence. Stability/Coherence calculated based on internal
factors and external influence factors. Initial arbitrary scaling removed.
Adheres strictly to PEP 8 formatting, especially line length and breaks.
Assumes `from constants.constants import *`.
"""

import logging
import os
import sys
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from math import log10, pi as PI, exp, sqrt, tanh
import numpy as np # Make sure numpy is imported as np
from constants.constants import * # Import constants for the module
# --- Constants Import ---
# Assume constants are imported at the top level
try:
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Could not import constants: {e}. SoulSpark cannot function.")
    sys.exit(1)

# --- Determine Required Geometric Attributes ---
# Calculate this list at the module level based on constants
_GEOM_ATTRS_TO_ADD: List[str] = [] # Initialize as empty list
try:
    _REQUIRED_GEOM_ATTR_KEYS = set()
    # Check if GEOMETRY_EFFECTS exists and is a dictionary
    if 'GEOMETRY_EFFECTS' in globals() and isinstance(GEOMETRY_EFFECTS, dict):
        for effects_dict in GEOMETRY_EFFECTS.values():
            # Ensure the value associated with a geometry is also a dict
            if isinstance(effects_dict, dict):
                _REQUIRED_GEOM_ATTR_KEYS.update(effects_dict.keys())
    else:
         logging.warning("GEOMETRY_EFFECTS constant not found or not a dictionary in constants.py.")
except Exception as e:
    logging.error(f"Error determining geometric attributes: {e}")
    _REQUIRED_GEOM_ATTR_KEYS = set()

# --- Logging ---
logger = logging.getLogger(__name__)
try:
    logger.setLevel(LOG_LEVEL)
except NameError:
    logger.warning("LOG_LEVEL constant not found, using default INFO level.")
    logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    log_formatter = logging.Formatter(LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

# --- Determine Required Geometric Attributes (Calculated within this module) ---
_GEOM_ATTRS_TO_ADD: List[str] = [] # Initialize as empty list
try:
    _REQUIRED_GEOM_ATTR_KEYS = set()
    # Use GEOMETRY_EFFECTS directly since we used 'import *'
    if isinstance(GEOMETRY_EFFECTS, dict):
        for effects_dict in GEOMETRY_EFFECTS.values():
            if isinstance(effects_dict, dict):
                _REQUIRED_GEOM_ATTR_KEYS.update(effects_dict.keys())

    # List of base SoulSpark attributes that might appear in GEOMETRY_EFFECTS
    # Ensure this list is comprehensive based on your SoulSpark attributes
    _EXISTING_SOUL_ATTRS_FOR_EFFECTS = [
        'stability', 'coherence', 'phi_resonance', 'yin_yang_balance',
        'harmony', 'field_integration', 'frequency', 'energy', 'resonance',
        'creator_alignment', 'pattern_coherence', 'cord_integrity',
        'earth_resonance', 'elemental_alignment', 'cycle_synchronization',
        'planetary_resonance', 'gaia_connection', 'name_resonance',
        'response_level', 'heartbeat_entrainment', 'crystallization_level',
        'attribute_coherence', 'physical_integration', 'memory_retention',
        'state_stability', 'guff_influence_factor',
        'cumulative_sephiroth_influence', 'creator_connection_strength',
        'field_radius', 'field_strength', 'physical_energy', 'spiritual_energy',
        'consciousness_frequency',
        # Add any other core attributes here if they might be targets of effects
    ]

    # Filter out keys that are existing attributes or likely modifiers
    _MODIFIER_SUFFIXES = ('_boost', '_factor', '_push', '_gain', '_modifier', '_delta')
    _GEOM_ATTRS_TO_ADD = sorted([
        attr for attr in _REQUIRED_GEOM_ATTR_KEYS
        if attr not in _EXISTING_SOUL_ATTRS_FOR_EFFECTS
        and not attr.endswith(_MODIFIER_SUFFIXES)
    ])
    logger.debug(f"Identified geometric attributes to add to SoulSpark: {_GEOM_ATTRS_TO_ADD}")

except NameError:
     logger.error("Constant 'GEOMETRY_EFFECTS' not found or not imported correctly. "
                  "Cannot determine geometric attributes.")
except Exception as e:
    logger.error(f"Error processing GEOMETRY_EFFECTS: {e}", exc_info=True)
         
# --- Logging ---
logger = logging.getLogger(__name__)
try:
    logger.setLevel(LOG_LEVEL)
except NameError:
    logger.warning("LOG_LEVEL constant not found, using default INFO level.")
    logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    log_formatter = logging.Formatter(LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

# --- Visualization Placeholder ---
VISUALIZATION_ENABLED = False


class SoulSpark:
    """
    Represents the evolving soul entity. S/C calculated from underlying factors.
    Initialization expects `initial_data` derived from the emergence process.
    """

    # --- __init__ ---
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None,
                 spark_id: Optional[str] = None):
        """
        Initialize a new SoulSpark instance using provided initial_data.
        Calculates initial SU/CU based on this data without arbitrary scaling.
        """
        self.spark_id: str = spark_id or str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.last_modified: str = self.creation_time
        data = initial_data if initial_data is not None else {}
        self.toroidal_flow_strength = float(data.get('toroidal_flow_strength', 0.05)) 
        if not data:
            raise ValueError("SoulSpark requires non-empty initial_data.")

        try:
            self.frequency: float = float(data['frequency'])
            self.energy: float = float(data['energy'])
            if self.frequency <= FLOAT_EPSILON:
                raise ValueError("Initial frequency must be positive.")
            if self.energy <= 0:
                raise ValueError("Initial energy must be positive.")
            max_init_energy = MAX_SOUL_ENERGY_SEU * 1.5
            if self.energy > max_init_energy:
                logger.warning(f"Initial energy {self.energy:.1f} SEU high, clamping to {max_init_energy:.1f}.")
                self.energy = max_init_energy
        except KeyError as ke:
            raise ValueError(f"Missing required key in initial_data: {ke}") from ke
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid core attribute in initial_data: {e}") from e

        self.resonance = float(data.get('resonance', 0.1))
        self.creator_alignment = float(data.get('creator_alignment', 0.0))
        self.phi_resonance = float(data.get('phi_resonance', 0.15))
        self.pattern_coherence = float(data.get('pattern_coherence', 0.15))
        self.harmony = float(data.get('harmony', 0.15)) # This is the 0-1 factor
        self.cord_integrity = float(data.get('cord_integrity', 0.0))
        self.field_integration = float(data.get('field_integration', 0.0))
        self.earth_resonance = float(data.get('earth_resonance', 0.0))
        self.elemental_alignment = float(data.get('elemental_alignment', 0.0))
        self.cycle_synchronization = float(data.get('cycle_synchronization', 0.0))
        self.planetary_resonance = float(data.get('planetary_resonance', 0.0))
        self.gaia_connection = float(data.get('gaia_connection', 0.0))
        self.name_resonance = float(data.get('name_resonance', 0.0))
        self.response_level = float(data.get('response_level', 0.0))
        self.heartbeat_entrainment = float(data.get('heartbeat_entrainment', 0.0))
        self.yin_yang_balance = float(data.get('yin_yang_balance', 0.5))
        self.crystallization_level = float(data.get('crystallization_level', 0.0))
        self.attribute_coherence = float(data.get('attribute_coherence', 0.0))
        self.physical_integration = float(data.get('physical_integration', 0.0))
        self.memory_retention = float(data.get('memory_retention', 1.0))
        self.state_stability = float(data.get('state_stability', 0.2))
        self.guff_influence_factor = float(data.get('guff_influence_factor', 0.0))
        self.cumulative_sephiroth_influence = float(data.get('cumulative_sephiroth_influence', 0.0))

        self.harmonics = data.get('harmonics', [])
        self.frequency_signature = data.get('frequency_signature', {})
        self._validate_or_init_frequency_structure() 
        self.frequency_history = data.get('frequency_history', [self.frequency] * 5)
        if not self.frequency_history: self.frequency_history = [self.frequency] * 5
        self.aspects: Dict[str, Dict[str, Any]] = data.get('aspects', {})
        for attr_name in _GEOM_ATTRS_TO_ADD:
            setattr(self, attr_name, float(data.get(attr_name, 0.0)))
        self.layers: List[Dict[str, Any]] = data.get('layers', [])
        self.interaction_history: List[Dict[str, Any]] = data.get('interaction_history', [])
        self.memory_echoes: List[str] = data.get('memory_echoes', [])
        flag_names = [f for f in globals() if f.startswith('FLAG_')]
        for flag_const_name in flag_names:
            flag_key = globals()[flag_const_name]
            default_flag = True if flag_key == FLAG_READY_FOR_GUFF else False
            setattr(self, flag_key, bool(data.get(flag_key, default_flag)))

        default_pos = [d / 2.0 for d in GRID_SIZE] 
        self.position: List[float] = data.get('position', default_pos)
        self.current_field_key: str = data.get('current_field_key', 'void')
        self.field_radius: float = float(data.get('field_radius', 1.0))
        self.field_strength: float = float(data.get('field_strength', 0.5))
        self.creator_channel_id: Optional[str] = data.get('creator_channel_id')
        self.creator_connection_strength: float = float(data.get('creator_connection_strength', 0.0))
        self.resonance_patterns: Dict[str, Any] = data.get('resonance_patterns', {})
        self.life_cord: Optional[Dict[str, Any]] = data.get('life_cord')
        self.elements: Dict[str, float] = data.get('elements', {})
        self.earth_cycles: Dict[str, float] = data.get('earth_cycles', {})
        self.name: Optional[str] = data.get('name')
        self.gematria_value: int = int(data.get('gematria_value', 0))
        self.voice_frequency: float = float(data.get('voice_frequency', 0.0))
        self.soul_color: Optional[str] = data.get('soul_color')
        self.color_frequency: float = float(data.get('color_frequency', 0.0))
        self.soul_frequency: float = float(data.get('soul_frequency', self.frequency))
        self.sephiroth_aspect: Optional[str] = data.get('sephiroth_aspect')
        self.elemental_affinity: Optional[str] = data.get('elemental_affinity')
        self.platonic_symbol: Optional[str] = data.get('platonic_symbol')
        default_emotions = {'love': 0.0, 'joy': 0.0, 'peace': 0.0, 'harmony': 0.0, 'compassion': 0.0}
        loaded_emotions = data.get('emotional_resonance', {})
        self.emotional_resonance: Dict[str, float] = {k: float(loaded_emotions.get(k, default_emotions.get(k, 0.0))) for k in default_emotions}
        self.identity_metrics: Optional[Dict[str, Any]] = data.get('identity_metrics')
        self.sacred_geometry_imprint: Optional[str] = data.get('sacred_geometry_imprint')
        self.geometric_pattern: Optional[str] = data.get('geometric_pattern')
        self.birth_time: Optional[str] = data.get('birth_time')
        self.memory_veil: Optional[Dict[str, Any]] = data.get('memory_veil')
        self.breath_pattern: Optional[Dict[str, Any]] = data.get('breath_pattern')
        self.physical_energy: float = float(data.get('physical_energy', 0.0))
        self.spiritual_energy: float = data.get('spiritual_energy', max(0.0, self.energy - self.physical_energy))
        self.consciousness_state: str = data.get('consciousness_state', 'spark')
        self.consciousness_frequency: float = float(data.get('consciousness_frequency', self.frequency * 0.1))

        self.stability: float = self._calculate_stability_score()
        self.stability = max(0.0, min(MAX_STABILITY_SU, self.stability))
        if self.stability < 10.0: logger.warning(f"Initial stability low: {self.stability:.1f} SU.")
        self.coherence: float = self._calculate_coherence_score()
        self.coherence = max(0.0, min(MAX_COHERENCE_CU, self.coherence))
        if self.coherence < 10.0: logger.warning(f"Initial coherence low: {self.coherence:.1f} CU.")

        self._validate_attributes()
        logger.info(f"SoulSpark initialized: ID={self.spark_id}, Field={self.current_field_key}, Freq={self.frequency:.2f}Hz, E={self.energy:.2f}SEU, S={self.stability:.1f}SU, C={self.coherence:.1f}CU")

    @property
    def energy_joules(self) -> float:
        return self.energy * ENERGY_UNSCALE_FACTOR

    def _validate_or_init_frequency_structure(self, num_harmonics: int = 2, force_regen_signature: bool = False): # Added force_regen
        regen_needed = force_regen_signature # Start with force_regen
        if not regen_needed: # Only check other conditions if not forced
            if not hasattr(self,'harmonics') or not isinstance(self.harmonics,list) or len(self.harmonics) == 0:
                self.harmonics = []
                regen_needed = True
            if (not hasattr(self,'frequency_signature') or 
                not isinstance(self.frequency_signature,dict) or
                'frequencies' not in self.frequency_signature or 
                'phases' not in self.frequency_signature or
                'amplitudes' not in self.frequency_signature or
                (isinstance(self.frequency_signature.get('frequencies'), (list, np.ndarray)) and 
                len(self.frequency_signature.get('frequencies', [])) == 0) or
                # Ensure base_frequency exists before comparing
                ('base_frequency' in self.frequency_signature and 
                 abs(self.frequency_signature.get('base_frequency',0.0)-self.frequency) > FLOAT_EPSILON) or
                # Add check if base_frequency is missing entirely
                ('base_frequency' not in self.frequency_signature)
                ):
                self.frequency_signature = {}
                regen_needed = True
        
        if regen_needed:
            logger.debug(f"Regenerating freq structure for {self.spark_id} (Forced: {force_regen_signature}).")
            self._generate_phi_integer_harmonics(num_harmonics)
        else:
            # logger.debug(f"Using existing freq structure for {self.spark_id}") # Can be noisy
            for key in ['frequencies','amplitudes','phases']:
                if key in self.frequency_signature and isinstance(self.frequency_signature[key],list):
                    self.frequency_signature[key] = np.array(self.frequency_signature[key])
            if 'frequencies' in self.frequency_signature:
                if isinstance(self.frequency_signature['frequencies'], np.ndarray):
                    self.harmonics = self.frequency_signature['frequencies'].tolist()
                else:
                    self.harmonics = self.frequency_signature.get('frequencies', [])

    def _generate_phi_integer_harmonics(self, num_harmonics: int = 2):
        if self.frequency <= FLOAT_EPSILON:
            self.frequency_signature={'base_frequency':self.frequency,'frequencies':[],'amplitudes':[],'phases':[],'num_frequencies':0}; self.harmonics=[]; return
        base_freq=self.frequency; frequencies=[base_freq]; amplitudes=[1.0]; phi_count=num_harmonics//2;
        for i in range(1,phi_count+1):
            freq = base_freq*(PHI**i)
            if freq/base_freq > 10: continue
            frequencies.append(freq); amplitudes.append(0.8/(PHI**(i*0.8)))
        int_count=num_harmonics-phi_count
        for i in range(2,int_count+2):
            freq=base_freq*i
            if freq/base_freq>10: continue
            frequencies.append(freq); amplitudes.append(0.9/(i**0.9))
        unique_freqs_amps=sorted(list(set(zip(frequencies,amplitudes))))
        if not unique_freqs_amps: unique_freqs_amps=[(base_freq,1.0)]
        final_frequencies,final_amplitudes=zip(*unique_freqs_amps); final_frequencies=np.array(final_frequencies); final_amplitudes=np.array(final_amplitudes)
        max_amp = np.max(final_amplitudes) if final_amplitudes.size > 0 else 0.0
        if max_amp > FLOAT_EPSILON: final_amplitudes /= max_amp; final_amplitudes = np.clip(final_amplitudes, 0, 1)
        else: final_frequencies = np.array([base_freq]); final_amplitudes = np.array([1.0])
        initial_phases=(final_frequencies/base_freq)*np.pi; initial_coherence_estimate=getattr(self,'pattern_coherence',0.15); phase_noise_mag=np.pi*(1.0-initial_coherence_estimate)*0.5; phase_noise=np.random.uniform(-phase_noise_mag,phase_noise_mag,len(final_frequencies)); final_phases=(initial_phases+phase_noise)%(2*np.pi)
        self.frequency_signature={'base_frequency':float(base_freq),'frequencies':final_frequencies.tolist(),'amplitudes':final_amplitudes.tolist(),'phases':final_phases.tolist(),'num_frequencies':len(final_frequencies)}
        self.harmonics=final_frequencies.tolist() # Ensure self.harmonics is updated
        # logger.debug(f"Generated Phi/Int harmonic structure for {self.spark_id}. NumHarmonics: {len(self.harmonics)}")

    def _optimize_phase_coherence(self, target_factor: float = 0.2) -> float:  # Increase default 0.1->0.2
        """
        Optimize phase coherence by reducing circular variance.
        Returns improvement amount (0-1 scale).
        """
        if not isinstance(self.frequency_signature, dict): return 0.0
        if 'phases' not in self.frequency_signature: return 0.0
        phases_data = self.frequency_signature['phases']
        if phases_data is None: return 0.0
        if isinstance(phases_data, list) and len(phases_data) == 0: return 0.0
        phases = np.array(phases_data) if not isinstance(phases_data, np.ndarray) else phases_data
        if phases.size <= 1: return 0.0
        
        mean_cos = np.mean(np.cos(phases))
        mean_sin = np.mean(np.sin(phases))
        initial_r = np.sqrt(mean_cos**2 + mean_sin**2)
        initial_var = 1.0 - initial_r
        mean_phase = np.arctan2(mean_sin, mean_cos)
        
        # Stronger optimization factor (0.2 default instead of 0.1)
        new_phases = phases * (1.0 - target_factor) + mean_phase * target_factor
        new_phases = new_phases % (2 * np.pi)
        
        new_mean_cos = np.mean(np.cos(new_phases))
        new_mean_sin = np.mean(np.sin(new_phases))
        new_r = np.sqrt(new_mean_cos**2 + new_mean_sin**2)
        new_var = 1.0 - new_r
        
        if new_var < initial_var - FLOAT_EPSILON:
            self.frequency_signature['phases'] = new_phases.tolist()
            logger.debug(f"Phase coherence optimized: {initial_var:.4f} -> {new_var:.4f}")
            return (initial_var - new_var)
        return 0.0
    
    def _calculate_stability_score(self) -> float:
        try:
            freq_stability_factor=0.0; variance=np.nan; relative_variance=np.inf; mean_freq=self.frequency
            if len(self.frequency_history)>=5:
                recent_freqs=np.array(self.frequency_history[-5:]); mean_freq=np.mean(recent_freqs)
                if mean_freq>FLOAT_EPSILON: variance=np.var(recent_freqs); relative_variance=variance/(mean_freq**2); freq_stability_factor=exp(-relative_variance*STABILITY_VARIANCE_PENALTY_K*0.1)
            else: freq_stability_factor=0.1
            layers_count=len(getattr(self,'layers',[])); aspects_count=len(getattr(self,'aspects',{}))
            aspect_strengths=[a.get('strength',0.0) for a in getattr(self,'aspects',{}).values()]
            avg_aspect_strength=sum(aspect_strengths)/max(1,len(aspect_strengths)) if aspect_strengths else 0.0
            phi_resonance=getattr(self,'phi_resonance',0.1); creator_alignment=getattr(self,'creator_alignment',0.0)
            layer_contrib=tanh(layers_count/5.0)*STABILITY_PATTERN_WEIGHT_LAYERS
            aspect_contrib=(tanh(aspects_count/20.0+avg_aspect_strength)*STABILITY_PATTERN_WEIGHT_ASPECTS)
            phi_contrib=phi_resonance*STABILITY_PATTERN_WEIGHT_PHI
            align_contrib=creator_alignment*STABILITY_PATTERN_WEIGHT_ALIGNMENT
            pattern_integrity_factor=np.clip(layer_contrib+aspect_contrib+phi_contrib+align_contrib,0.0,1.0)
            guff_influence=getattr(self,'guff_influence_factor',0.0)
            seph_influence=getattr(self,'cumulative_sephiroth_influence',0.0)
            field_influence_factor=np.clip(guff_influence+seph_influence,0.0,1.0)
            torus_factor = getattr(self, 'toroidal_flow_strength', 0.0)
            stability_score = (freq_stability_factor * STABILITY_WEIGHT_FREQ +
                   pattern_integrity_factor * STABILITY_WEIGHT_PATTERN +
                   field_influence_factor * STABILITY_WEIGHT_FIELD +
                   torus_factor * STABILITY_WEIGHT_TORUS 
                  ) * MAX_STABILITY_SU
            light_energy_integrity = 0.0
            try:
                # Calculate light-energy integrity based on phi and frequency stability
                if hasattr(self, 'phi_resonance') and hasattr(self, 'frequency_history'):
                    phi_factor = getattr(self, 'phi_resonance', 0.1)
                    if len(self.frequency_history) >= 5:
                        freq_variance = np.var(self.frequency_history[-5:])
                        if self.frequency > FLOAT_EPSILON:
                            # Normalize variance relative to frequency
                            rel_variance = freq_variance / (self.frequency**2)
                            # Calculate stability component (higher with lower variance)
                            freq_stability_component = exp(-rel_variance * 5.0)
                            # Combine with phi resonance for light-energy integrity
                            light_energy_integrity = (freq_stability_component * 0.6 + phi_factor * 0.4) * 0.2
                            logger.debug(f"  _calc_stability: Light-energy integrity={light_energy_integrity:.4f} "
                                        f"(phi={phi_factor:.3f}, freq_stability={freq_stability_component:.3f})")
            except Exception as e:
                logger.error(f"Error calculating light-energy integrity: {e}", exc_info=True)
                light_energy_integrity = 0.0

            # Add light-energy integrity to stability score
            stability_score += light_energy_integrity * MAX_STABILITY_SU

            # Log detailed calculation
            logger.debug(f"  _calc_stability: Final score with light factor = {stability_score:.1f} SU")


            total_weight = (STABILITY_WEIGHT_FREQ + STABILITY_WEIGHT_PATTERN +
                            STABILITY_WEIGHT_FIELD + STABILITY_WEIGHT_TORUS)
            if abs(total_weight - 1.0) > FLOAT_EPSILON and total_weight > FLOAT_EPSILON:
                stability_score *= (1.0 / total_weight) # Normalize if weights don't sum to 1
                logger.debug(f"Stability weights (sum={total_weight:.3f}) normalized.") # Log only if normalized

            # logger.debug(f"  _calc_stability: TorusFactor={torus_factor:.3f}, Weight={STABILITY_WEIGHT_TORUS:.2f}")
            # logger.debug(f"  _calc_stability: FreqFactor={freq_stability_factor:.3f} (RelVar={relative_variance:.4E})")
            # logger.debug(f"  _calc_stability: PatternFactor={pattern_integrity_factor:.3f} (L={layers_count}, A={aspects_count}, AvgStr={avg_aspect_strength:.2f}, PhiRes={phi_resonance:.2f}, Align={creator_alignment:.2f})")
            # logger.debug(f"  _calc_stability: FieldInfluence={field_influence_factor:.3f} (Guff={guff_influence:.3f}, Seph={seph_influence:.3f})")
            # logger.debug(f"  _calc_stability: WEIGHTS: Freq={STABILITY_WEIGHT_FREQ:.2f}, Pattern={STABILITY_WEIGHT_PATTERN:.2f}, Field={STABILITY_WEIGHT_FIELD:.2f}, Torus={STABILITY_WEIGHT_TORUS:.2f}")
            # logger.debug(f"  _calc_stability: Final Score = {stability_score:.1f} SU")
            return max(0.0,min(MAX_STABILITY_SU,stability_score))
        except Exception as e: logger.error(f"Error calculating stability score: {e}",exc_info=True); return getattr(self,'stability',0.0)


    def _calculate_coherence_score(self) -> float:
        """ Calculates coherence (0-100 CU) from underlying factors. Includes detailed logging and robust error handling. """
        logger.debug(f"--- Running _calculate_coherence_score for {self.spark_id} with DETAILED LOGGING ---")
        try: 
            # --- Initialize factors with defaults ---
            phase_coherence_factor = 0.1
            harmonic_purity_factor = 0.1 # This was the variable name used for the calculated harmonic purity
            pattern_coherence_factor = getattr(self, 'pattern_coherence', 0.0)
            guff_influence = getattr(self, 'guff_influence_factor', 0.0)
            seph_influence = getattr(self, 'cumulative_sephiroth_influence', 0.0)
            creator_factor = getattr(self, 'creator_connection_strength', 0.0)
            torus_factor = getattr(self, 'toroidal_flow_strength', 0.0)
            
            # --- DETAILED LOGGING: Initial attribute values ---
            logger.debug(f"  COHERENCE DEBUG - Initial Attributes:")
            logger.debug(f"    pattern_coherence = {pattern_coherence_factor:.4f}")
            logger.debug(f"    guff_influence = {guff_influence:.4f}")
            logger.debug(f"    seph_influence = {seph_influence:.4f}")
            logger.debug(f"    creator_factor = {creator_factor:.4f}")
            logger.debug(f"    torus_factor = {torus_factor:.4f}")
            logger.debug(f"    harmony (not used) = {getattr(self, 'harmony', 0.0):.4f}")

            circ_variance = 1.0 
            phases_debug = "N/A" 
            logger.debug(f"Phases data for coherence calculation: {phases_debug}")

            # --- 1. Phase Alignment Component ---
            freq_sig = getattr(self, 'frequency_signature', {})
            if isinstance(freq_sig, dict) and 'phases' in freq_sig:
                phases_data = freq_sig.get('phases')
                
                # DETAILED LOGGING: Phases data
                logger.debug(f"  COHERENCE DEBUG - Phases Data: type={type(phases_data).__name__}, empty={phases_data is None or (isinstance(phases_data, (list, np.ndarray)) and len(phases_data) == 0)}")
                
                if isinstance(phases_data, (list, np.ndarray)):
                    phases = np.array(phases_data) 
                    phases_debug = f"{phases.size}"
                    
                    # DETAILED LOGGING: Phases array
                    if phases.size > 0:
                        logger.debug(f"  COHERENCE DEBUG - Phases Array: size={phases.size}, min={np.min(phases) if phases.size > 0 else 'N/A'}, max={np.max(phases) if phases.size > 0 else 'N/A'}")
                    
                    if phases.size > 1:
                        mean_cos = np.mean(np.cos(phases))
                        mean_sin = np.mean(np.sin(phases))
                        sqrt_arg = max(0.0, mean_cos**2 + mean_sin**2)
                        mean_r = sqrt(sqrt_arg) 
                        circ_variance = 1.0 - mean_r 
                        phase_coherence_factor = np.clip(1.0 - circ_variance, 0.0, 1.0)
                        
                        # DETAILED LOGGING: Phase calculation
                        logger.debug(f"  COHERENCE DEBUG - Phase Calc: mean_cos={mean_cos:.4f}, mean_sin={mean_sin:.4f}, mean_r={mean_r:.4f}, circ_variance={circ_variance:.4f}")
                        
                    elif phases.size == 1:
                        phase_coherence_factor = 0.75 
                        circ_variance = 0.25 
                        logger.debug(f"  COHERENCE DEBUG - Single Phase: Used default phase_coherence_factor={phase_coherence_factor:.4f}")
                else: 
                    logger.warning(f"COHERENCE DEBUG - 'phases' data is invalid type ({type(phases_data)}) or None.")
                    phases_debug = f"InvalidType({type(phases_data).__name__})"
            else: 
                logger.warning(f"COHERENCE DEBUG - frequency_signature invalid type ({type(freq_sig).__name__}) or missing 'phases'.")
                phases_debug = "MissingKey/NotDict"

            # --- 2. Harmonic Purity Component ---
            deviations = []
            light_resonance_factor = 0.0
            harmonics_list = getattr(self, 'harmonics', [])
            base_freq = self.frequency

            # Ensure base_freq is valid before proceeding
            if harmonics_list and isinstance(base_freq, (int, float)) and base_freq > FLOAT_EPSILON:
                valid_harmonics = [h for h in harmonics_list if isinstance(h, (int, float)) and h > FLOAT_EPSILON]
                ratios = [h / base_freq for h in valid_harmonics]
                
                if ratios:
                    # Calculate first-order deviations (standard from original method)
                    primary_deviations = []
                    phi_resonances = 0
                    integer_resonances = 0
                    
                    for r in ratios:
                        if not np.isfinite(r): 
                            continue  # Skip invalid ratios
                            
                        # Check simple integer ratios with detailed tracking
                        int_devs = [abs(r-n) for n in range(1,6)]
                        min_int_dev = min(int_devs)
                        if min_int_dev < 0.05:  # Strong integer resonance
                            integer_resonances += 1
                            
                        # Check Phi ratios with detailed tracking
                        phi_devs = [abs(r - PHI), abs(r - (1/PHI)), abs(r - PHI**2), abs(r - (1/PHI**2))]
                        min_phi_dev = min(phi_devs)
                        if min_phi_dev < 0.05:  # Strong phi resonance
                            phi_resonances += 1
                            
                        # Store minimum deviation
                        primary_deviations.append(min(min_int_dev, min_phi_dev))
                    
                    # Calculate second-order harmonic interactions (new)
                    # This simulates light-like interference patterns between harmonics
                    secondary_deviations = []
                    if len(valid_harmonics) >= 2:
                        for i in range(len(valid_harmonics)):
                            for j in range(i+1, len(valid_harmonics)):
                                # Get difference and ratio between harmonics
                                h1, h2 = valid_harmonics[i], valid_harmonics[j]
                                freq_diff = abs(h1 - h2)
                                freq_ratio = max(h1, h2) / min(h1, h2)
                                
                                # Check if difference forms a resonant pattern
                                diff_factor = 0.0
                                if freq_diff > FLOAT_EPSILON:
                                    # Check if difference is near integer multiple of base freq
                                    diff_multiple = freq_diff / base_freq
                                    diff_dev = abs(diff_multiple - round(diff_multiple))
                                    diff_factor = exp(-diff_dev * 20.0)  # Sharp falloff
                                
                                # Check if ratio forms a resonant pattern
                                ratio_factor = 0.0
                                if freq_ratio > 1.0:
                                    # Check if ratio is near integer or phi
                                    int_ratio_dev = min(abs(freq_ratio - round(freq_ratio)), 1.0)
                                    phi_ratio_dev = min(abs(freq_ratio - PHI), abs(freq_ratio - (1/PHI)))
                                    ratio_dev = min(int_ratio_dev, phi_ratio_dev)
                                    ratio_factor = exp(-ratio_dev * 15.0)  # Sharp falloff
                                
                                # Combined effect (interference pattern)
                                combined_factor = diff_factor * 0.3 + ratio_factor * 0.7
                                secondary_deviations.append(1.0 - combined_factor)  # Convert to deviation
                    
                    # Calculate light resonance based on harmonic relationships
                    # More integer and phi resonances = stronger light-like resonance
                    total_harmonics = len(valid_harmonics)
                    if total_harmonics > 0:
                        integer_ratio = integer_resonances / total_harmonics
                        phi_ratio = phi_resonances / total_harmonics
                        light_resonance_factor = (integer_ratio * 0.6 + phi_ratio * 0.4) * 0.7
                    
                    # Combine primary and secondary deviations
                    if primary_deviations:
                        deviations.extend(primary_deviations)
                    if secondary_deviations:
                        deviations.extend(secondary_deviations)
                        
                    # Calculate overall purity factor
                    avg_deviation = np.mean(deviations) if deviations else 10.0
                    if not np.isfinite(avg_deviation): 
                        avg_deviation = 10.0
                        
                    harmonic_purity_factor = exp(-avg_deviation * 10.0)
                    
                    # Log significant harmonic patterns
                    if integer_resonances > 2 or phi_resonances > 1:
                        # Only log this message periodically to avoid spamming the log
                        if not hasattr(self, '_last_harmonic_log_time'):
                            self._last_harmonic_log_time = 0.0
                        
                        current_time = time.time()
                        # Only log once every 5 seconds for the same soul
                        if current_time - self._last_harmonic_log_time > 5.0:
                            logger.info(f"Soul {self.spark_id[:8]} has strong harmonic pattern: "
                                    f"{integer_resonances} integer, {phi_resonances} phi resonances")
                            self._last_harmonic_log_time = current_time
                        
            else:  # No harmonics or invalid base_freq
                logger.debug(f"_calc_coherence: No valid harmonics or base frequency for soul {self.spark_id[:8]}")
                harmonic_purity_factor = 0.1
                
            harmonic_purity_factor = np.clip(harmonic_purity_factor, 0.0, 1.0)

            # --- 3. Field Influence Component ---
            seph_influence_enhanced = np.power(seph_influence, 0.8) if seph_influence > FLOAT_EPSILON else 0.0
            field_influence_factor = np.clip(guff_influence * 0.4 + seph_influence_enhanced * 0.6, 0.0, 1.0)
            
            # DETAILED LOGGING: Field influence calculation
            logger.debug(f"  COHERENCE DEBUG - Field Influence: guff={guff_influence:.4f}, seph_enhanced={seph_influence_enhanced:.4f}, combined={field_influence_factor:.4f}")

            # --- 4. Weighted Sum ---
            # Check for valid factors
            factors = [phase_coherence_factor, harmonic_purity_factor, pattern_coherence_factor,
                    field_influence_factor, creator_factor, torus_factor]
            
            # DETAILED LOGGING: All factors before sum
            logger.debug(f"  COHERENCE DEBUG - All Factors:")
            logger.debug(f"    phase_coherence_factor = {phase_coherence_factor:.4f}")
            logger.debug(f"    harmonic_purity_factor = {harmonic_purity_factor:.4f}")
            logger.debug(f"    pattern_coherence_factor = {pattern_coherence_factor:.4f}")
            logger.debug(f"    field_influence_factor = {field_influence_factor:.4f}")
            logger.debug(f"    creator_factor = {creator_factor:.4f}")
            logger.debug(f"    torus_factor = {torus_factor:.4f}")
            
            if not all(isinstance(f, (int, float)) and np.isfinite(f) for f in factors):
                logger.error(f"COHERENCE DEBUG - One or more coherence factors non-numeric/non-finite! Factors: {factors}")
                return float(getattr(self, 'coherence', 0.0))
            
            # THE CRITICAL POINT: This sum MUST match the weights defined in constants.py
            # DETAILED LOGGING: Weight values
            logger.debug(f"  COHERENCE DEBUG - All Weights:")
            logger.debug(f"    COHERENCE_WEIGHT_PHASE = {COHERENCE_WEIGHT_PHASE:.4f}")
            logger.debug(f"    COHERENCE_WEIGHT_HARMONY = {COHERENCE_WEIGHT_HARMONY:.4f}")
            logger.debug(f"    COHERENCE_WEIGHT_PATTERN = {COHERENCE_WEIGHT_PATTERN:.4f}")
            logger.debug(f"    COHERENCE_WEIGHT_FIELD = {COHERENCE_WEIGHT_FIELD:.4f}")
            logger.debug(f"    COHERENCE_WEIGHT_CREATOR = {COHERENCE_WEIGHT_CREATOR:.4f}")
            logger.debug(f"    COHERENCE_WEIGHT_TORUS = {COHERENCE_WEIGHT_TORUS:.4f}")
            
            light_boost = light_resonance_factor * 0.15  # Scale appropriately

            raw_coherence_score = (phase_coherence_factor * COHERENCE_WEIGHT_PHASE +
                                harmonic_purity_factor * COHERENCE_WEIGHT_HARMONY +
                                pattern_coherence_factor * COHERENCE_WEIGHT_PATTERN +
                                field_influence_factor * COHERENCE_WEIGHT_FIELD +
                                creator_factor * COHERENCE_WEIGHT_CREATOR +
                                torus_factor * COHERENCE_WEIGHT_TORUS +
                                light_boost)  # Add light resonance
            
            logger.debug(f"_calc_coherence: Including light_resonance_factor={light_resonance_factor:.4f}, light_boost={light_boost:.4f}")
            
            # DETAILED LOGGING: Raw score calculation
            logger.debug(f"  COHERENCE DEBUG - Raw Score Components:")
            logger.debug(f"    phase: {phase_coherence_factor:.4f} * {COHERENCE_WEIGHT_PHASE:.4f} = {phase_coherence_factor * COHERENCE_WEIGHT_PHASE:.4f}")
            logger.debug(f"    harmonic: {harmonic_purity_factor:.4f} * {COHERENCE_WEIGHT_HARMONY:.4f} = {harmonic_purity_factor * COHERENCE_WEIGHT_HARMONY:.4f}")
            logger.debug(f"    pattern: {pattern_coherence_factor:.4f} * {COHERENCE_WEIGHT_PATTERN:.4f} = {pattern_coherence_factor * COHERENCE_WEIGHT_PATTERN:.4f}")
            logger.debug(f"    field: {field_influence_factor:.4f} * {COHERENCE_WEIGHT_FIELD:.4f} = {field_influence_factor * COHERENCE_WEIGHT_FIELD:.4f}")
            logger.debug(f"    creator: {creator_factor:.4f} * {COHERENCE_WEIGHT_CREATOR:.4f} = {creator_factor * COHERENCE_WEIGHT_CREATOR:.4f}")
            logger.debug(f"    torus: {torus_factor:.4f} * {COHERENCE_WEIGHT_TORUS:.4f} = {torus_factor * COHERENCE_WEIGHT_TORUS:.4f}")
            logger.debug(f"    raw_sum = {raw_coherence_score:.4f}")

            # --- 5. Normalization ---
            total_weight = (COHERENCE_WEIGHT_PHASE + COHERENCE_WEIGHT_HARMONY +
                            COHERENCE_WEIGHT_PATTERN + COHERENCE_WEIGHT_FIELD +
                            COHERENCE_WEIGHT_CREATOR + COHERENCE_WEIGHT_TORUS)
            
            normalized_score = raw_coherence_score
            if abs(total_weight - 1.0) > FLOAT_EPSILON:
                if total_weight > FLOAT_EPSILON:
                    logger.debug(f"  COHERENCE DEBUG - Normalizing. Total weight = {total_weight:.4f} != 1.0")
                    normalized_score = raw_coherence_score / total_weight
                else:
                    logger.error(f"COHERENCE DEBUG - Total weight zero/negative ({total_weight:.4f}). Setting score to 0.")
                    normalized_score = 0.0
            
            # --- 6. Scale and Clamp ---
            final_coherence_cu = normalized_score * MAX_COHERENCE_CU
            clamped_coherence = max(0.0, min(MAX_COHERENCE_CU, final_coherence_cu))
            
            # DETAILED LOGGING: Final calculation
            logger.debug(f"  COHERENCE DEBUG - Final Calculation:")
            logger.debug(f"    raw_coherence_score = {raw_coherence_score:.4f}")
            logger.debug(f"    total_weight = {total_weight:.4f}")
            logger.debug(f"    normalized_score = {normalized_score:.4f}")
            logger.debug(f"    MAX_COHERENCE_CU = {MAX_COHERENCE_CU:.1f}")
            logger.debug(f"    final_coherence_cu = {final_coherence_cu:.4f}")
            logger.debug(f"    clamped_coherence = {clamped_coherence:.4f}")

            if not isinstance(clamped_coherence, (int, float)) or not np.isfinite(clamped_coherence):
                logger.error(f"COHERENCE DEBUG - Invalid final value: {clamped_coherence}. Returning 0.0")
                return 0.0
                
            logger.debug(f"--- End of _calculate_coherence_score for {self.spark_id} - returning {clamped_coherence:.4f} ---")
            return float(clamped_coherence)

        except Exception as e:
            logger.error(f"CRITICAL ERROR calculating coherence score: {e}", exc_info=True)
            return float(getattr(self, 'coherence', 0.0)) # Fallback
    
    def update_state(self):
            logger.debug(f"--- Running update_state for {self.spark_id} ---")
            if hasattr(self,'frequency_history') and isinstance(self.frequency_history,list):
                self.frequency_history.append(self.frequency);
                if len(self.frequency_history)>20: self.frequency_history.pop(0)
            elif not hasattr(self,'frequency_history'): self.frequency_history=[self.frequency]*5
            original_stability = getattr(self,'stability', 0.0)
            original_coherence = getattr(self,'coherence', 0.0)

            # logger.debug(f"  update_state PRE-CALC FACTORS: Freq={self.frequency:.1f}, PatternCoh={self.pattern_coherence:.4f}, "
            #             f"GuffInf={self.guff_influence_factor:.4f}, SephInf={self.cumulative_sephiroth_influence:.4f}, "
            #             f"CreatorConn={self.creator_connection_strength:.4f}, Torus={self.toroidal_flow_strength:.4f}, "
            #             f"HarmonyFactor={self.harmony:.4f}, EarthRes={self.earth_resonance:.4f}, "
            #             f"#Harmonics={len(self.harmonics)}, #Phases={len(self.frequency_signature.get('phases', []))}")
            try:
                new_stability = self._calculate_stability_score()
                new_coherence = self._calculate_coherence_score()
                if new_stability is None or not isinstance(new_stability, (int, float)):
                    logger.warning(f"Stability calculation returned None or non-numeric. Using previous value: {original_stability}")
                    new_stability = original_stability
                if new_coherence is None or not isinstance(new_coherence, (int, float)):
                    logger.warning(f"Coherence calculation returned None or non-numeric. Using previous value: {original_coherence}")
                    new_coherence = original_coherence
                
                new_stability = max(0.0, min(MAX_STABILITY_SU, new_stability)) 
                new_coherence = max(0.0, min(MAX_COHERENCE_CU, new_coherence))
                
                # logger.debug(f"  update_state: Calculated New S={new_stability:.1f}, New C={new_coherence:.1f}. Current S={original_stability:.1f}, C={original_coherence:.1f}")
                
                stability_changed = abs(new_stability - original_stability) > FLOAT_EPSILON
                coherence_changed = abs(new_coherence - original_coherence) > FLOAT_EPSILON

                self.stability = float(new_stability)
                self.coherence = float(new_coherence)
                
                if stability_changed: 
                    print(f"DEBUG: SoulSpark {self.spark_id} stability updated: {original_stability:.2f} -> {self.stability:.2f}")
                if coherence_changed: 
                    print(f"DEBUG: SoulSpark {self.spark_id} coherence updated: {original_coherence:.2f} -> {self.coherence:.2f}")
                    
                # logger.debug(f"  update_state: VERIFICATION - After assignment S={self.stability:.1f}, C={self.coherence:.1f}")
                # logger.debug(f"--- Finished update_state. Final S={self.stability:.1f}, C={self.coherence:.1f} ---")
            except Exception as e:
                logger.error(f"Error during update_state: {e}", exc_info=True)
                self.stability = original_stability
                self.coherence = original_coherence
            self._validate_soul_state()
            
    def add_layer(self, source: str, density_map: Dict[str, Any], color_hex: str, timestamp: str) -> None:
        """Add a new aura layer to the soul."""
        if not hasattr(self, 'layers'):
            self.layers = []
            
        new_layer = {
            'source': source,
            'density': density_map,  # Store density_map directly under 'density' key
            'color_hex': color_hex,
            'timestamp': timestamp
        }
        
        self.layers.append(new_layer)
        logger.info(f"Added layer from {source} to soul {self.spark_id}.")

    def get_coherence_color(self) -> Tuple[int, int, int]:
        try: norm=self.coherence/MAX_COHERENCE_CU; norm=max(0.0,min(1.0,norm))
        except: norm=0.1
        if norm>=0.98: return (255,255,255);
        if norm<=0.02: return (50,0,0);
        hue=(1.0-norm)*300.0; sat=0.8+norm*0.2; val=0.7+norm*0.3; h_i=int(hue/60)%6; f=hue/60-h_i; p=val*(1-sat); q=val*(1-f*sat); t=val*(1-(1-f)*sat)
        if h_i==0: r,g,b=val,t,p; 
        elif h_i==1: r,g,b=q,val,p 
        elif h_i==2: r,g,b=p,val,t 
        elif h_i==3: r,g,b=p,q,val 
        elif h_i==4: r,g,b=t,p,val 
        else: r,g,b=val,p,q
        return (int(r*255),int(g*255),int(b*255))

    def get_pattern_distortion(self) -> float:
        try: norm=self.stability/MAX_STABILITY_SU; norm=max(0.0,min(1.0,norm)); return 1.0-norm
        except: return 0.9

    def add_memory_echo(self, event_description: str):
        if not hasattr(self,'memory_echoes') or not isinstance(self.memory_echoes,list): self.memory_echoes=[]
        self.memory_echoes.append(f"{datetime.now().isoformat()}: {event_description}"); # logger.debug(f"Memory echo added: '{event_description}'")

    def get_spark_metrics(self) -> Dict[str, Any]:
        try:
             life_cord_data=getattr(self,'life_cord',{}) or {}
             core_metrics={'spark_id':self.spark_id,'name':getattr(self,'name',None),'current_state':getattr(self,'consciousness_state','unknown'),'current_field':getattr(self,'current_field_key','unknown'),'frequency_hz':self.frequency,'stability_su':self.stability,'coherence_cu':self.coherence,'energy_seu':self.energy,'physical_energy_seu':getattr(self,'physical_energy',0.0),'spiritual_energy_seu':getattr(self,'spiritual_energy',0.0),'phi_resonance':self.phi_resonance,'creator_alignment':self.creator_alignment,'earth_resonance':self.earth_resonance,'cord_integrity':self.cord_integrity,'physical_integration':self.physical_integration,'crystallization_level':self.crystallization_level,'aspects_count':len(self.aspects),'layer_count':len(self.layers),'age_seconds':(datetime.now()-datetime.fromisoformat(self.creation_time)).total_seconds(),}
             flag_names=[f for f in globals() if f.startswith('FLAG_')];
             for flag_const_name in flag_names: flag_key=globals()[flag_const_name]; core_metrics[flag_key]=getattr(self,flag_key,False)
             extended_metrics={'position':self.position,'aspects':self.aspects,'layers':self.layers,'interaction_history_count':len(self.interaction_history),'geometric_pattern':getattr(self,'geometric_pattern',None),'sacred_geometry_imprint':getattr(self,'sacred_geometry_imprint',None),'platonic_symbol':getattr(self,'platonic_symbol',None),'elements':self.elements,'emotional_resonance':self.emotional_resonance,'frequency_signature':self.frequency_signature,'life_cord':life_cord_data,'identity_details':{'gematria':self.gematria_value,'name_resonance':self.name_resonance,'voice_frequency_hz':self.voice_frequency,'response_level':self.response_level,'soul_color':self.soul_color,'color_frequency_hz':self.color_frequency,'soul_frequency_hz':self.soul_frequency,'sephiroth_aspect':self.sephiroth_aspect,'elemental_affinity':self.elemental_affinity,'yin_yang_balance':self.yin_yang_balance,'attribute_coherence':self.attribute_coherence,},'geometric_attributes':{attr:getattr(self,attr,0.0) for attr in _GEOM_ATTRS_TO_ADD},'birth_details':{'birth_time':self.birth_time,'memory_retention':self.memory_retention,},'state_details':{'consciousness_frequency_hz':self.consciousness_frequency,'state_stability':self.state_stability,},'memory_echo_count':len(self.memory_echoes),'last_modified':self.last_modified,'creation_time':self.creation_time,}
             metadata={'timestamp':datetime.now().isoformat(),'metrics_version':'4.3.7_emergence','units':{'energy':'SEU','stability':'SU','coherence':'CU','frequency':'Hz'},'energy_scale_joules_per_seu':ENERGY_UNSCALE_FACTOR,}
             return {'core':core_metrics,'extended':extended_metrics,'metadata':metadata}
        except Exception as e: logger.error(f"Error generating metrics: {e}",exc_info=True); return {'core':{'spark_id':getattr(self,'spark_id','err'),'error':str(e)},'extended':{},'metadata':{'error':True}}

    def _validate_attributes(self):
        # logger.debug(f"Validating attributes for soul {self.spark_id}..."); # Can be noisy
        numeric_attrs_config={'energy':(FLOAT_EPSILON,MAX_SOUL_ENERGY_SEU*1.5),'stability':(0.0,MAX_STABILITY_SU),'coherence':(0.0,MAX_COHERENCE_CU),'frequency':(FLOAT_EPSILON,None),'voice_frequency':(0.0,None),'color_frequency':(0.0,None),'soul_frequency':(FLOAT_EPSILON,None),'consciousness_frequency':(0.0,None),'gematria_value':(0,None,int),'physical_energy':(0.0,MAX_SOUL_ENERGY_SEU),'spiritual_energy':(0.0,MAX_SOUL_ENERGY_SEU),'resonance':(0.0,1.0),'creator_alignment':(0.0,1.0),'phi_resonance':(0.0,1.0),'pattern_coherence':(0.0,1.0),'harmony':(0.0,1.0),'cord_integrity':(0.0,1.0),'field_integration':(0.0,1.0),'earth_resonance':(0.0,1.0),'elemental_alignment':(0.0,1.0),'cycle_synchronization':(0.0,1.0),'planetary_resonance':(0.0,1.0),'gaia_connection':(0.0,1.0),'name_resonance':(0.0,1.0),'response_level':(0.0,1.0),'heartbeat_entrainment':(0.0,1.0),'yin_yang_balance':(0.0,1.0),'crystallization_level':(0.0,1.0),'attribute_coherence':(0.0,1.0),'physical_integration':(0.0,1.0),'memory_retention':(0.0,1.0),'state_stability':(0.0,1.0),**{attr:(0.0,1.0) for attr in _GEOM_ATTRS_TO_ADD}}
        for attr,config in numeric_attrs_config.items():
             if not hasattr(self,attr): continue
             min_val,max_val=config[0],config[1]; expected_type=config[2] if len(config)>2 else (int,float); val=getattr(self,attr,None)
             if val is None: continue
             if not isinstance(val,expected_type):
                 try: val=int(round(float(val))) if expected_type==int or int in expected_type else float(val); setattr(self,attr,val)
                 except: logger.critical(f"VALIDATION FAIL: Cannot convert {attr}"); raise TypeError(f"Attr {attr} invalid type {type(val)}")
             current_val=getattr(self,attr)
             if not isinstance(current_val,(int,float)) or not np.isfinite(current_val): raise ValueError(f"Attr {attr} non-finite: {repr(current_val)}")
             clamped_val=current_val; clamped=False
             if min_val is not None and current_val<min_val: clamped_val=min_val; clamped=True
             if max_val is not None and current_val>max_val: clamped_val=max_val; clamped=True
             if clamped and abs(clamped_val-current_val)>FLOAT_EPSILON: logger.warning(f"Clamping '{attr}' ({current_val}) -> {clamped_val}"); setattr(self,attr,clamped_val)
        dict_attrs=['aspects','emotional_resonance','elements','earth_cycles','frequency_signature','resonance_patterns','life_cord','memory_veil','breath_pattern','identity_metrics']; list_attrs=['harmonics','memory_echoes','position','layers','interaction_history','frequency_history']
        for attr in dict_attrs:
             if hasattr(self,attr) and getattr(self,attr) is not None and not isinstance(getattr(self,attr),dict): raise TypeError(f"Attr {attr} must be dict/None")
        for attr in list_attrs:
             if hasattr(self,attr) and not isinstance(getattr(self,attr),list): raise TypeError(f"Attr {attr} must be list")
        if not(isinstance(self.position,list) and len(self.position)==3 and all(isinstance(p,(int,float)) for p in self.position)): raise ValueError(f"Position invalid: {self.position}")
        for layer in getattr(self,'layers',[]):
            if not isinstance(layer, dict): 
                raise ValueError(f"Layer is not a dictionary: {layer}")
            if 'sephirah' not in layer and 'source' not in layer:
                raise ValueError(f"Layer missing required 'sephirah' or 'source' key: {layer}")
        for aspect_name, aspect_detail in getattr(self,'aspects',{}).items():
            if not isinstance(aspect_detail,dict) or 'strength' not in aspect_detail or not isinstance(aspect_detail['strength'],(int,float)): raise ValueError(f"Invalid aspect structure for '{aspect_name}': {aspect_detail}")
            if not (0.0<=aspect_detail['strength']<=MAX_ASPECT_STRENGTH): aspect_detail['strength']=max(0.0,min(MAX_ASPECT_STRENGTH,aspect_detail['strength']))
        # logger.debug("Attribute validation complete.") # Can be noisy

    def _validate_soul_state(self):
        try:
            if not hasattr(self,'stability') or not isinstance(self.stability,(int,float)) or not (0.0<=self.stability<=MAX_STABILITY_SU): logger.error(f"VALIDATION ERROR: Stability invalid: {getattr(self,'stability','MISSING')}")
            if not hasattr(self,'coherence') or not isinstance(self.coherence,(int,float)) or not (0.0<=self.coherence<=MAX_COHERENCE_CU): logger.error(f"VALIDATION ERROR: Coherence invalid: {getattr(self,'coherence','MISSING')}")
        except Exception as e: logger.error(f"VALIDATION ERROR: Exception: {e}")

    def save_spark_data(self, file_path: str) -> bool:
        logger.info(f"Saving soul {self.spark_id} to {file_path}...")
        try:
            data_to_save={attr:getattr(self,attr) for attr in self.__dict__};
            class NumpyEncoder(json.JSONEncoder):
                def default(self,o):
                    if isinstance(o,np.ndarray): return o.tolist()
                    if isinstance(o,(np.int_,np.intc,np.intp,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64)): return int(o)
                    if isinstance(o,(np.float_,np.float16,np.float32,np.float64)): return float(o)
                    if isinstance(o,(datetime,uuid.UUID)): return str(o)
                    try: return super().default(o)
                    except TypeError: return f"<Unserializable:{type(o).__name__}>"
            os.makedirs(os.path.dirname(file_path),exist_ok=True);
            with open(file_path,'w') as f: json.dump(data_to_save,f,indent=2,cls=NumpyEncoder)
            logger.info("Save successful."); return True
        except Exception as e: logger.error(f"Error saving {self.spark_id}: {e}",exc_info=True); raise

    @classmethod
    def load_from_file(cls, file_path: str) -> 'SoulSpark':
        logger.info(f"Loading soul from {file_path}...")
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(file_path,'r') as f: loaded_data=json.load(f)
            if not isinstance(loaded_data,dict): raise ValueError("Invalid data format.")
            init_args_data={}; required_keys=['frequency','energy'];
            for key in required_keys:
                 if key not in loaded_data: raise ValueError(f"Loaded data missing '{key}'.")
                 init_args_data[key]=loaded_data[key]
            for key in ['harmonics','frequency_signature','phi_resonance','pattern_coherence','harmony']:
                 if key in loaded_data: init_args_data[key]=loaded_data[key]
            spark_id=loaded_data.get('spark_id'); instance=cls(initial_data=init_args_data,spark_id=spark_id)
            for attr,value in loaded_data.items():
                 if isinstance(value,dict) and attr=='frequency_signature':
                      for k_sig in ['frequencies','amplitudes','phases']: # Renamed k to k_sig
                           if k_sig in value and isinstance(value[k_sig],list): value[k_sig]=np.array(value[k_sig])
                 try: setattr(instance,attr,value)
                 except AttributeError: logger.warning(f"Could not set attr '{attr}' during load.")
            instance._validate_or_init_frequency_structure(); instance._validate_attributes();
            logger.info(f"Soul spark loaded: ID={instance.spark_id}"); return instance
        except Exception as e: logger.error(f"Error loading soul: {e}",exc_info=True); raise RuntimeError(f"Failed load: {e}") from e

    def __str__(self) -> str:
        name=getattr(self,'name',self.spark_id[:8]); state=getattr(self,'consciousness_state','N/A'); field=getattr(self,'current_field_key','N/A'); eng=getattr(self,'energy',0.0); stab=getattr(self,'stability',0.0); coh=getattr(self,'coherence',0.0); cryst_flag_name = FLAG_IDENTITY_CRYSTALLIZED if 'FLAG_IDENTITY_CRYSTALLIZED' in globals() else 'identity_crystallized_flag_missing' ;cryst=getattr(self,cryst_flag_name,False)
        return (f"SoulSpark(Name: {name}, ID: {self.spark_id[:8]}, Field: {field}, State: {state}, E:{eng:.1f}SEU, S:{stab:.1f}SU, C:{coh:.1f}CU, Cryst:{cryst})")

    def __repr__(self) -> str:
        return (f"<SoulSpark id='{self.spark_id}' name='{getattr(self,'name',None)}' E={self.energy:.1f} S={self.stability:.1f} C={self.coherence:.1f}>")

# --- END OF FILE src/stage_1/soul_formation/soul_spark.py ---










