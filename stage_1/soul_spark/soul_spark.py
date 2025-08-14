# --- START OF FILE src/stage_1/soul_formation/soul_spark.py ---

"""
Soul Spark Module (Refactored V4.3.7 - Emergence Principle, Strict PEP8)

Defines the SoulSpark class. Initialization relies on data derived
from field emergence. Stability/Coherence calculated based on internal
factors and external influence factors. Initial arbitrary scaling removed.
Adheres strictly to PEP 8 formatting, especially line length and breaks.
Assumes `from shared.constants.constants import *`.
"""

import logging
import os
import sys
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
from math import log10, pi as PI, exp, sqrt, tanh

if TYPE_CHECKING:
    from fields.field_controller import FieldController
import numpy as np # Make sure numpy is imported as np
from shared.constants.constants import * # Import constants for the module
# --- Constants Import ---
# Assume constants are imported at the top level
try:
    from shared.constants.constants import *
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

        self.resonance = float(data.get('resonance', 0.0))
        self.creator_alignment = float(data.get('creator_alignment', 0.0))
        self.phi_resonance = float(data.get('phi_resonance', 0.0))
        self.pattern_coherence = float(data.get('pattern_coherence', 0.0))
        self.harmony = float(data.get('harmony', 0.0)) 
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
    
    @classmethod
    def create_from_field_emergence(cls, field_controller: 'FieldController', 
                                    coords: Optional[Tuple[int, int, int]] = None) -> 'SoulSpark':
        """
        Creates a soul spark naturally from field properties at edge of chaos.
        This replaces arbitrary initialization with field-derived properties.
        """
        logger.info("Creating soul spark through field emergence...")
        
        # Find optimal emergence location (edge of chaos) if not specified
        if coords is None:
            optimal_coords = field_controller.find_optimal_development_location()
            coords = (int(optimal_coords[0]), int(optimal_coords[1]), int(optimal_coords[2]))
            logger.info(f"Found optimal emergence location: {coords}")
        
        # Get field properties at emergence point
        field_properties = field_controller.get_properties_at(coords)
        
        # DEBUG: Log field properties
        logger.info(f"Field properties at {coords}:")
        logger.info(f"  Energy: {field_properties.get('energy_seu', 'N/A')} SEU")
        logger.info(f"  Coherence: {field_properties.get('coherence_cu', 'N/A')} CU")
        logger.info(f"  Stability: {field_properties.get('stability_su', 'N/A')} SU")
        logger.info(f"  Order factor: {field_properties.get('order_factor', 'N/A')}")
        logger.info(f"  Chaos factor: {field_properties.get('chaos_factor', 'N/A')}")
        logger.info(f"  Edge of chaos: {field_properties.get('edge_of_chaos', 'N/A')}")
        
        # Validate field properties
        if not field_properties or 'error' in field_properties:
            raise RuntimeError(f"Cannot emerge soul: Invalid field properties at {coords}")
        
        # Create initial data from field properties
        initial_data = cls._extract_field_properties_for_emergence(field_properties, coords)
        
        # DEBUG: Log extracted initial data
        logger.info("Extracted initial data for soul:")
        logger.info(f"  Energy: {initial_data.get('energy')} SEU")
        logger.info(f"  Pattern coherence: {initial_data.get('pattern_coherence')}")
        logger.info(f"  Phi resonance: {initial_data.get('phi_resonance')}")
        logger.info(f"  Harmony: {initial_data.get('harmony')}")
        logger.info(f"  Toroidal flow: {initial_data.get('toroidal_flow_strength')}")
        
        # Generate unique spark ID
        timestamp = datetime.now().isoformat()
        spark_id = f"Soul_{timestamp.replace(':', '-').replace('.', '_')}_{coords[0]}_{coords[1]}_{coords[2]}"
        
        # Create the soul spark
        soul_spark = cls(initial_data=initial_data, spark_id=spark_id)
        
        # DEBUG: Log actual soul values after creation
        logger.info("Soul spark after creation:")
        logger.info(f"  Energy: {soul_spark.energy:.1f} SEU")
        logger.info(f"  Stability: {soul_spark.stability:.1f} SU")
        logger.info(f"  Coherence: {soul_spark.coherence:.1f} CU")
        logger.info(f"  Pattern coherence: {soul_spark.pattern_coherence:.3f}")
        logger.info(f"  Phi resonance: {soul_spark.phi_resonance:.3f}")
        logger.info(f"  Harmony: {soul_spark.harmony:.3f}")
        logger.info(f"  Toroidal flow: {getattr(soul_spark, 'toroidal_flow_strength', 'N/A')}")
        
        # Set emergence flags
        setattr(soul_spark, FLAG_READY_FOR_GUFF, True)
        
        return soul_spark

    @staticmethod
    def _extract_field_properties_for_emergence(field_props: Dict[str, Any], 
                                            coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extracts and transforms field properties into initial soul data."""
        
        # Extract field values with logging
        field_energy = field_props.get('energy_seu', VOID_BASE_ENERGY_SEU)
        field_coherence = field_props.get('coherence_cu', VOID_BASE_COHERENCE_CU)
        field_stability = field_props.get('stability_su', VOID_BASE_STABILITY_SU)
        field_order = field_props.get('order_factor', 0.5)
        field_chaos = field_props.get('chaos_factor', 0.5)
        edge_of_chaos = field_props.get('edge_of_chaos', 0.5)
        
        logger.debug(f"Field values - Energy: {field_energy}, Coherence: {field_coherence}, Stability: {field_stability}")
        logger.debug(f"Field factors - Order: {field_order}, Chaos: {field_chaos}, EoC: {edge_of_chaos}")
        
        # Ensure all core values are above minimum thresholds
        initial_pattern_coherence = max(0.2, field_order * 0.8)  # Minimum 0.2
        initial_phi_resonance = max(0.2, edge_of_chaos * 0.8)    # Minimum 0.2
        initial_harmony = max(0.2, field_order * 0.7)           # Minimum 0.2
        initial_torus = max(0.1, edge_of_chaos * field_chaos * 0.8)  # Minimum 0.1
        
        logger.debug(f"Initial soul values - Pattern: {initial_pattern_coherence}, Phi: {initial_phi_resonance}")
        logger.debug(f"Initial soul values - Harmony: {initial_harmony}, Torus: {initial_torus}")
        # In create_from_field_emergence, add debug logging:
        logger.info(f"Field properties at {coords}:")
        logger.info(f"  Energy: {field_props.get('energy_seu', 'N/A')} SEU")
        logger.info(f"  Order factor: {field_props.get('order_factor', 'N/A')}")
        logger.info(f"  Edge of chaos: {field_props.get('edge_of_chaos', 'N/A')}")
        
        initial_data = {
            # Core values
            'energy': float(field_energy),
            'frequency': float(field_props.get('frequency_hz', 432.0)),
            
            # Ensure minimum viable values for coherence calculation
            'pattern_coherence': initial_pattern_coherence,
            'phi_resonance': initial_phi_resonance,
            'harmony': initial_harmony,
            'toroidal_flow_strength': initial_torus,
            
            # Give initial sparks a small amount of creator connection
            # This represents the spark's origin from the field itself
            'creator_alignment': max(0.1, edge_of_chaos * 0.3),  # Some connection
            'creator_connection_strength': max(0.1, edge_of_chaos * 0.2),  # Some strength
            
            # Field context
            'current_field_key': 'void',
            'position': list(coords),
            
            # Other required fields...
            'guff_influence_factor': 0.0,
            'cumulative_sephiroth_influence': 0.0,
            'resonance': max(0.1, field_order * 0.5),
            'yin_yang_balance': 0.5,
            
            # Initialize collections
            'aspects': {},
            'layers': [],
            'interaction_history': [],
            'memory_echoes': [],
            'frequency_history': [field_props.get('frequency_hz', 432.0)] * 5,
        }
        
        # Create harmonics and frequency signature...
        base_freq = initial_data['frequency']
        harmonics = [base_freq, base_freq * 2.0, base_freq * 3.0, base_freq * GOLDEN_RATIO, base_freq / GOLDEN_RATIO]
        initial_data['harmonics'] = harmonics
        
        # Create frequency signature with proper phases
        frequencies = np.array(harmonics)
        amplitudes = np.array([1.0, 0.5, 0.3, 0.4, 0.2])
        
        # Use field coherence to determine phase alignment
        coherence_factor = min(1.0, field_coherence / MAX_COHERENCE_CU)
        if coherence_factor > 0.1:  # If field has some coherence
            # Create more aligned phases
            phase_spread = (1.0 - coherence_factor) * PI * 0.5
            phases = np.array([0.0, PI/4, PI/2, 3*PI/4, PI]) + np.random.uniform(-phase_spread, phase_spread, 5)
        else:
            # Random phases for low field coherence
            phases = np.random.uniform(0, 2*PI, 5)
        
        phases = phases % (2 * PI)
        
        initial_data['frequency_signature'] = {
            'base_frequency': float(base_freq),
            'frequencies': frequencies.tolist(),
            'amplitudes': amplitudes.tolist(),
            'phases': phases.tolist(),
            'num_frequencies': len(frequencies)
        }
        
        # Create initial void layer
        timestamp = datetime.now().isoformat()
        void_layer = {
            'source': 'void',
            'density': {
                'base_density': max(0.2, field_order * 0.8),
                'uniformity': max(0.2, 1.0 - field_chaos * 0.5)
            },
            'color_hex': '#1a1a2e',
            'timestamp': timestamp
        }
        initial_data['layers'] = [void_layer]
        
        return initial_data

    @staticmethod
    def _calculate_emergence_quality(field_props: Dict[str, Any]) -> float:
        """
        Calculates a quality metric for emergence based on field properties.
        Higher quality indicates better conditions for stable soul development.
        """
        edge_of_chaos = field_props.get('edge_of_chaos', 0.0)
        field_order = field_props.get('order_factor', 0.5)
        field_energy = field_props.get('energy_seu', 0.0)
        pattern_influence = field_props.get('pattern_influence', 0.0)
        
        # Quality factors
        edge_quality = edge_of_chaos  # Direct measure
        order_quality = 1.0 - abs(field_order - 0.618)  # Closer to golden ratio is better
        energy_quality = min(1.0, field_energy / (VOID_BASE_ENERGY_SEU * 2.0))  # Good energy levels
        pattern_quality = min(1.0, pattern_influence * 2.0)  # Pattern presence helps
        
        # Weighted combination
        quality = (edge_quality * 0.4 + order_quality * 0.3 + 
                energy_quality * 0.2 + pattern_quality * 0.1)
        
        return float(np.clip(quality, 0.0, 1.0))

    
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
        """
        Calculates stability as an emergent property rather than a weighted sum.
        Stability emerges from the integrity of the soul's structure and its 
        resilience to perturbation.
        """
        try:
            # --- Frequency stability component ---
            freq_stability_factor = 0.0
            variance = np.nan
            relative_variance = np.inf
            mean_freq = self.frequency
            
            if len(self.frequency_history) >= 5:
                recent_freqs = np.array(self.frequency_history[-5:])
                mean_freq = np.mean(recent_freqs)
                if mean_freq > FLOAT_EPSILON:
                    variance = np.var(recent_freqs)
                    relative_variance = variance / (mean_freq**2)
                    # Natural stability - inverse exponential relationship with variance
                    freq_stability_factor = np.exp(-relative_variance * 20.0)
            else:
                freq_stability_factor = 0.1
                
            # --- Structural components ---
            
            # Layers represent developed soul structure
            layers_count = len(getattr(self, 'layers', []))
            layer_density = np.tanh(layers_count / 7.0)  # Natural saturation curve
            
            # Aspects represent soul qualities that contribute to identity
            aspects_count = len(getattr(self, 'aspects', {}))
            aspect_strengths = [a.get('strength', 0.0) for a in getattr(self, 'aspects', {}).values()]
            avg_aspect_strength = np.mean(aspect_strengths) if aspect_strengths else 0.0
            aspect_density = np.tanh(aspects_count / 15.0) * np.clip(avg_aspect_strength, 0.1, 1.0)
            
            # Phi resonance aligns with universal patterns
            phi_resonance = getattr(self, 'phi_resonance', 0.1)
            
            # Creator alignment provides foundational connection
            creator_alignment = getattr(self, 'creator_alignment', 0.0)
            
            # Toroidal flow creates a self-reinforcing energy pattern
            torus_factor = getattr(self, 'toroidal_flow_strength', 0.0)
            
            # Pattern coherence organizes internal information
            pattern_coherence = getattr(self, 'pattern_coherence', 0.0)
            
            # External field influences
            guff_influence = getattr(self, 'guff_influence_factor', 0.0)
            seph_influence = getattr(self, 'cumulative_sephiroth_influence', 0.0)
            
            # --- Natural stability emergence ---
            
            # Core structural integrity from layers and aspects
            structural_integrity = (layer_density * 0.6) + (aspect_density * 0.4)
            
            # Core resonant integrity from phi resonance and pattern coherence
            resonant_integrity = np.sqrt(phi_resonance * pattern_coherence)
            
            # Foundational connection from creator alignment and field influences
            field_strength = np.clip((guff_influence + seph_influence * 1.5) / 2.0, 0.0, 1.0)
            foundational_connection = np.sqrt(creator_alignment * field_strength)
            
            # Dynamic sustainability from frequency stability and toroidal flow
            dynamic_sustainability = np.sqrt(freq_stability_factor * max(0.1, torus_factor))
            
            # --- Resilience factors ---
            
            # Natural systems develop resilience as they mature
            # This is the ability to recover from disturbance
            resilience = 0.0
            
            # Resilience comes from diversity and redundancy
            if aspects_count > 5:
                # Diverse aspects provide alternative pathways for stability
                aspect_diversity = min(1.0, aspects_count / 20.0)
                resilience += aspect_diversity * 0.3
            
            # Resilience comes from energetic reserves
            energy_reserve_factor = min(1.0, self.energy / (MAX_SOUL_ENERGY_SEU * 0.2))
            resilience += energy_reserve_factor * 0.2
            
            # Resilience comes from strong integration of components
            component_avg = np.mean([structural_integrity, resonant_integrity, 
                                foundational_connection, dynamic_sustainability])
            if component_avg > 0.5:
                integration_factor = component_avg - 0.5
                resilience += integration_factor * 0.5
                
            # Cap resilience
            resilience = min(0.5, resilience)
            
            # --- Final stability calculation ---
            
            # Base stability from core components
            base_components = [
                structural_integrity, 
                resonant_integrity,
                foundational_connection,
                dynamic_sustainability
            ]
            
            # Calculate weighted geometric mean
            # Natural systems require all components to be present for true stability
            component_product = np.prod([max(c, 0.05) for c in base_components])  # Avoid zeros
            base_stability = component_product ** (1.0 / len(base_components))
            
            # Apply resilience bonus - mature systems can maintain stability through disruption
            stability_with_resilience = base_stability * (1.0 + resilience)
            
            # Convert to stability units with natural limits
            stability_su = np.clip(stability_with_resilience * MAX_STABILITY_SU, 0.0, MAX_STABILITY_SU)
            
            logger.debug(f"Stability components: Structure={structural_integrity:.3f}, Resonance={resonant_integrity:.3f}, "
                        f"Foundation={foundational_connection:.3f}, Dynamics={dynamic_sustainability:.3f}")
            logger.debug(f"Resilience: {resilience:.3f}, Final stability: {stability_su:.2f} SU")
            
            return float(stability_su)
        except Exception as e:
            logger.error(f"Error calculating natural stability: {e}", exc_info=True)
            return getattr(self, 'stability', 0.0)


    # --- In stage_1/soul_spark/soul_spark.py ---

    def _calculate_coherence_score(self) -> float:
        """
        Calculates coherence as an emergent property with detailed logging.
        """
        try: 
            logger.debug(f"=== Starting coherence calculation for {self.spark_id} ===")
            
            # Initialize foundation variables
            phase_coherence = 0.0
            harmonic_purity = 0.0
            
            # Calculate phase coherence - a measure of order in the frequency signature
            freq_sig = getattr(self, 'frequency_signature', {})
            logger.debug(f"Frequency signature type: {type(freq_sig)}")
            
            if isinstance(freq_sig, dict) and 'phases' in freq_sig:
                phases_data = freq_sig.get('phases')
                logger.debug(f"Phases data: {phases_data}")
                
                if phases_data is not None and len(phases_data) > 1:
                    phases = np.array(phases_data) if not isinstance(phases_data, np.ndarray) else phases_data
                    logger.debug(f"Phases shape: {phases.shape}")
                    
                    # Calculate circular variance for phase coherence
                    mean_cos = np.mean(np.cos(phases))
                    mean_sin = np.mean(np.sin(phases))
                    resultant_length = np.sqrt(mean_cos**2 + mean_sin**2)
                    phase_coherence = resultant_length  # 0 = random phases, 1 = aligned phases
                    logger.debug(f"Phase coherence calculated: {phase_coherence}")
            
            # Calculate harmonic purity
            harmonics_list = getattr(self, 'harmonics', [])
            base_freq = self.frequency
            logger.debug(f"Harmonics list: {harmonics_list}")
            logger.debug(f"Base frequency: {base_freq}")
            
            if harmonics_list and isinstance(base_freq, (int, float)) and base_freq > FLOAT_EPSILON:
                valid_harmonics = [h for h in harmonics_list if isinstance(h, (int, float)) and h > FLOAT_EPSILON]
                logger.debug(f"Valid harmonics: {valid_harmonics}")
                
                if valid_harmonics and base_freq > FLOAT_EPSILON:
                    ratios = [h / base_freq for h in valid_harmonics]
                    logger.debug(f"Harmonic ratios: {ratios}")
                    
                    total_deviation = 0.0
                    count = 0
                    
                    for r in ratios:
                        if not np.isfinite(r) or r <= FLOAT_EPSILON: 
                            continue
                        
                        # Check integer ratios
                        min_int_dev = float('inf')
                        for n in range(1, 6):
                            int_dev = abs(r - n)
                            min_int_dev = min(min_int_dev, int_dev)
                        
                        # Check Phi ratios
                        phi_devs = [abs(r - PHI), abs(r - (1/PHI)), abs(r - PHI**2), abs(r - (1/PHI**2))]
                        min_phi_dev = min(phi_devs)
                        
                        # Use the smaller deviation
                        min_dev = min(min_int_dev, min_phi_dev)
                        # Normalize by ratio to make it scale-independent
                        normalized_dev = min_dev / r
                        total_deviation += normalized_dev
                        count += 1
                    
                    if count > 0:
                        avg_deviation = total_deviation / count
                        # Convert deviation to purity (1 - deviation)
                        harmonic_purity = max(0.0, 1.0 - avg_deviation)
                        logger.debug(f"Harmonic purity calculated: {harmonic_purity}")
                    else:
                        logger.warning(f"No valid ratios counted")
                else:
                    logger.warning(f"No valid harmonics found")
            else:
                logger.warning(f"Invalid harmonics or base frequency")
            
            # Get other factors
            pattern_coherence = getattr(self, 'pattern_coherence', 0.0)
            phi_resonance = getattr(self, 'phi_resonance', 0.0)
            harmony = getattr(self, 'harmony', 0.0)
            torus_factor = getattr(self, 'toroidal_flow_strength', 0.0)
            
            logger.debug(f"Pattern coherence: {pattern_coherence}")
            logger.debug(f"Phi resonance: {phi_resonance}")
            logger.debug(f"Harmony: {harmony}")
            logger.debug(f"Toroidal flow: {torus_factor}")
            
            # External field influences
            guff_influence = getattr(self, 'guff_influence_factor', 0.0)
            seph_influence = getattr(self, 'cumulative_sephiroth_influence', 0.0)
            creator_factor = getattr(self, 'creator_connection_strength', 0.0)
            
            logger.debug(f"Guff influence: {guff_influence}")
            logger.debug(f"Sephiroth influence: {seph_influence}")
            logger.debug(f"Creator factor: {creator_factor}")
            
            # --- Natural coherence emergence ---
            
            # Foundation: Pattern coherence x Phi resonance creates structural integrity
            structural_integrity = np.sqrt(max(FLOAT_EPSILON, pattern_coherence * phi_resonance))
            logger.debug(f"Structural integrity: {structural_integrity}")
            
            # Harmonic expression: Phase coherence x Harmonic purity creates resonant clarity
            resonant_clarity = np.sqrt(max(FLOAT_EPSILON, phase_coherence * harmonic_purity))
            logger.debug(f"Resonant clarity: {resonant_clarity}")
            
            # Flow dynamics: Harmony x Toroidal flow creates sustainable dynamics
            flow_dynamics = np.sqrt(max(FLOAT_EPSILON, harmony * torus_factor))
            logger.debug(f"Flow dynamics: {flow_dynamics}")
            
            # External connection: Creator connection x Field influences creates contextual integration
            field_strength = np.clip((guff_influence + seph_influence * 1.5) / 2.0, 0.0, 1.0)
            contextual_integration = np.sqrt(max(FLOAT_EPSILON, creator_factor * field_strength))
            logger.debug(f"Field strength: {field_strength}")
            logger.debug(f"Contextual integration: {contextual_integration}")
            
            # For new souls without external connections, use base coherence
            if contextual_integration < 0.1:
                contextual_integration = 0.1 + (pattern_coherence * 0.1)  # Give minimum base
                logger.debug(f"Adjusted contextual integration for new soul: {contextual_integration}")
            
            # Apply natural damping to prevent artificial perfection
            dampened_components = []
            components = [structural_integrity, resonant_clarity, flow_dynamics, contextual_integration]
            
            for i, component in enumerate(components):
                # Natural systems rarely reach perfect states - apply mild asymptotic curve
                # Ensure component is not zero to avoid log errors
                if component <= FLOAT_EPSILON:
                    dampened = 0.1  # Minimum threshold for empty components
                else:
                    dampened = 1.0 - np.exp(-3.0 * component)
                dampened_components.append(dampened)
                logger.debug(f"Component {i} ({component:.3f}) dampened to {dampened:.3f}")
            
            # Calculate the geometric mean to enforce interdependence
            # Ensure no zero components
            component_product = np.prod([max(FLOAT_EPSILON, c) for c in dampened_components])
            num_components = len(dampened_components)
            logger.debug(f"Component product: {component_product}")
            
            # Apply an emergent quality - systems can become more than the sum of their parts
            emergence_bonus = 0.0
            if all(c > 0.5 for c in dampened_components):
                # When all components are reasonably strong, emergence creates additional coherence
                emergence_bonus = np.mean([c - 0.5 for c in dampened_components]) * 0.2
                logger.debug(f"Emergence bonus applied: {emergence_bonus}")
            
            # Final coherence calculation using geometric mean with emergence
            base_coherence = component_product ** (1.0 / num_components)
            final_coherence = np.clip(base_coherence + emergence_bonus, 0.0, 1.0)
            coherence_cu = final_coherence * MAX_COHERENCE_CU
            
            logger.debug(f"Base coherence: {base_coherence}, Final coherence: {final_coherence}")
            logger.debug(f"Final coherence CU: {coherence_cu}")
            
            logger.debug(f"=== Coherence calculation complete: {coherence_cu} CU ===")
            return float(coherence_cu)
            
        except Exception as e:
            logger.error(f"Error in coherence calculation: {e}", exc_info=True)
            # Return a minimal coherence based on what we can calculate
            pattern_coherence = getattr(self, 'pattern_coherence', 0.2)
            phi_resonance = getattr(self, 'phi_resonance', 0.2)
            fallback_coherence = np.sqrt(pattern_coherence * phi_resonance) * MAX_COHERENCE_CU * 0.5
            logger.error(f"Returning fallback coherence: {fallback_coherence}")
            return float(fallback_coherence)

    def update_state(self):
        """Updates stability and coherence based on current attributes."""
        logger.debug(f"--- Running update_state for {self.spark_id} ---")
        
        # Update frequency history
        if hasattr(self, 'frequency_history') and isinstance(self.frequency_history, list):
            self.frequency_history.append(self.frequency)
            if len(self.frequency_history) > 20:
                self.frequency_history.pop(0)
        elif not hasattr(self, 'frequency_history'):
            self.frequency_history = [self.frequency] * 5
        
        # Store original values for comparison
        original_stability = getattr(self, 'stability', 0.0)
        original_coherence = getattr(self, 'coherence', 0.0)
        
        # Log current factors for debugging
        logger.debug(f"  update_state PRE-CALC FACTORS: Freq={self.frequency:.1f}, PatternCoh={self.pattern_coherence:.4f}, "
                    f"GuffInf={self.guff_influence_factor:.4f}, SephInf={self.cumulative_sephiroth_influence:.4f}, "
                    f"CreatorConn={self.creator_connection_strength:.4f}, Torus={self.toroidal_flow_strength:.4f}, "
                    f"Harmony={self.harmony:.4f}, EarthRes={self.earth_resonance:.4f}")
        
        try:
            # Calculate new stability and coherence
            new_stability = self._calculate_stability_score()
            new_coherence = self._calculate_coherence_score()
            
            # Validate results
            if new_stability is None or not isinstance(new_stability, (int, float)):
                logger.critical(f"CRITICAL FAILURE: Stability calculation returned invalid result for spark {self.spark_id}")
                raise RuntimeError(f"Stability calculation failed for spark {self.spark_id}")
            if new_coherence is None or not isinstance(new_coherence, (int, float)):
                logger.critical(f"CRITICAL FAILURE: Coherence calculation returned invalid result for spark {self.spark_id}")
                raise RuntimeError(f"Coherence calculation failed for spark {self.spark_id}")
            
            # Clamp values to valid range
            new_stability = max(0.0, min(MAX_STABILITY_SU, new_stability)) 
            new_coherence = max(0.0, min(MAX_COHERENCE_CU, new_coherence))
            
            # Check if values changed significantly
            stability_changed = abs(new_stability - original_stability) > FLOAT_EPSILON
            coherence_changed = abs(new_coherence - original_coherence) > FLOAT_EPSILON
            
            # Update soul attributes
            self.stability = float(new_stability)
            self.coherence = float(new_coherence)
            
            # Log changes for debugging
            if stability_changed or coherence_changed:
                logger.info(f"SoulSpark {self.spark_id[:8]} metrics updated: S:{original_stability:.2f}→{self.stability:.2f}, "
                        f"C:{original_coherence:.2f}→{self.coherence:.2f}")
            
            logger.debug(f"--- Finished update_state. Final S={self.stability:.1f}, C={self.coherence:.1f} ---")
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
        try: 
            norm = self.coherence / MAX_COHERENCE_CU
            norm = max(0.0, min(1.0, norm))
        except Exception as e:
            logger.critical(f"CRITICAL FAILURE: Cannot calculate coherence color for spark {self.spark_id}: {e}")
            raise RuntimeError(f"Coherence color calculation failed for spark {self.spark_id}") from e
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
        try: 
            norm = self.stability / MAX_STABILITY_SU
            norm = max(0.0, min(1.0, norm))
            return 1.0 - norm
        except Exception as e:
            logger.critical(f"CRITICAL FAILURE: Cannot calculate pattern distortion for spark {self.spark_id}: {e}")
            raise RuntimeError(f"Pattern distortion calculation failed for spark {self.spark_id}") from e

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
                    if isinstance(o,(np.integer,np.floating)): return int(o) if isinstance(o,np.integer) else float(o)
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










