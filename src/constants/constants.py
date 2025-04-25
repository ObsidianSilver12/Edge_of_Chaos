"""
Consolidated Constants for the Edge of Chaos Soul Development Framework.

This module defines ALL key constants used throughout the system, including
frequencies, ratios, patterns, thresholds, factors, filenames, directories,
and other fixed values essential to the soul development process.
Modules should import specific constants from here as needed.
"""

import numpy as np
import os
import logging # Import logging to define LOG_LEVEL
from typing import List, Dict, Union, Any, Optional, Tuple

# --- System & General ---
DATA_DIR_BASE: str = "data" # Base directory for saving field data, etc.
OUTPUT_DIR_BASE: str = "output" # Base directory for saving generated sounds, images etc.
LOG_LEVEL = logging.INFO # Logging level for modules
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SAMPLE_RATE: int = 44100  # Hz (Standard audio sample rate)
MAX_AMPLITUDE: float = 0.8  # Max audio amplitude to prevent clipping (-1 to 1 theoretical max)
DEFAULT_DURATION: float = 30.0 # Default duration in seconds for generated sounds
DEFAULT_DIMENSIONS_3D: tuple = (64, 64, 64) # Default grid size for 3D fields
DEFAULT_FIELD_DTYPE = np.float32 # Default data type for energy fields
DEFAULT_COMPLEX_DTYPE = np.complex64 # Default data type for complex fields
FLOAT_EPSILON: float = 1e-9 # Small float for safe division/comparison
PERSIST_INTERVAL_SECONDS = 60 # Persist metrics roughly every minute if active
METRICS_DIR = os.path.join(DATA_DIR_BASE, "metrics") # Define full path
METRICS_FILE = os.path.join(METRICS_DIR, "soul_metrics.json")
SOUND_OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, "sounds") # Base sound output
FIELD_DATA_DIR = os.path.join(DATA_DIR_BASE, "fields") # Base field data dir

# --- Mathematical & Physics Concepts ---
PI: float = np.pi
GOLDEN_RATIO: float = (1 + np.sqrt(5)) / 2  # ~1.618
PHI: float = GOLDEN_RATIO # Alias
EDGE_OF_CHAOS_RATIO: float = 1 / PHI # ~0.618
SQRT_2: float = np.sqrt(2)
SQRT_3: float = np.sqrt(3)
SQRT_5: float = np.sqrt(5)
SQRT_6: float = np.sqrt(6)
SQRT_7: float = np.sqrt(7)
SILVER_RATIO = 1.0 + SQRT_2  # Silver ratio = 1 + √2 ≈ 2.414

# Fibonacci sequence (starting 0, 1)
FIBONACCI_SEQUENCE: List[int] = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Prime numbers
PRIME_NUMBERS: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# --- Frequencies ---
CREATOR_FREQUENCY: float = 432.0  # Hz - Universal creator frequency
FUNDAMENTAL_FREQUENCY_432: float = 432.0 # Alias
FUNDAMENTAL_FREQUENCY: float = FUNDAMENTAL_FREQUENCY_432 # General fundamental default

# Solfeggio Frequencies
SOLFEGGIO_UT: float = 396.0   # Liberating guilt and fear
SOLFEGGIO_RE: float = 417.0   # Undoing situations and facilitating change
SOLFEGGIO_MI: float = 528.0   # Transformation and miracles (Tiphareth association)
SOLFEGGIO_FA: float = 639.0   # Connecting/relationships
SOLFEGGIO_SOL: float = 741.0  # Expression/solutions
SOLFEGGIO_LA: float = 852.0   # Awakening intuition
SOLFEGGIO_SI: float = 963.0   # Returning to spiritual order (Kether association)
SOLFEGGIO_FREQUENCIES: Dict[str, float] = {
    'UT': SOLFEGGIO_UT, 'RE': SOLFEGGIO_RE, 'MI': SOLFEGGIO_MI,
    'FA': SOLFEGGIO_FA, 'SOL': SOLFEGGIO_SOL, 'LA': SOLFEGGIO_LA, 'SI': SOLFEGGIO_SI
}

# Harmonic Frequencies (Relative to a base, typically CREATOR_FREQUENCY)
HARMONIC_FREQUENCIES: Dict[str, float] = {
    'fundamental': 1.0, # Ratio
    'octave': 2.0,
    'perfect_fifth': 1.5, # 3/2
    'perfect_fourth': 4/3,
    'major_third': 5/4,
    'minor_third': 6/5,
    'golden_ratio': GOLDEN_RATIO
}

# Earth Frequencies
SCHUMANN_RESONANCE_BASE: float = 7.83 # Hz (Earth's fundamental resonance)
EARTH_FREQUENCIES: Dict[str, float] = {
    'schumann': SCHUMANN_RESONANCE_BASE,
    'heartbeat': 1.17,     # Typical heartbeat (Hz) - ~70 BPM
    'breath': 0.2,         # Typical breath cycle (Hz) - ~12 breaths per minute
    'circadian': 1/(24*60*60), # Corrected Hz
    'lunar': 1/(29.53*24*60*60), # Corrected Hz
    'annual': 1/(365.25*24*60*60) # Corrected Hz
}

# Earth Energy Fields (Example structure, frequencies may need refinement)
EARTH_ENERGY_FIELDS: Dict[str, Dict[str, float]] = {
    'magnetic':        { 'base_frequency': 11.75, 'field_strength': 0.6, 'resonance_factor': 0.8 },
    'gravitational':   { 'base_frequency': 9.81,  'field_strength': 0.9, 'resonance_factor': 0.7 }, # Placeholder freq
    'electromagnetic': { 'base_frequency': EARTH_FREQUENCIES['schumann'], 'field_strength': 0.7, 'resonance_factor': 0.85 },
    'natural_radiation':{ 'base_frequency': 13.4,  'field_strength': 0.4, 'resonance_factor': 0.6 } # Example freq
}

# --- Consciousness ---
CONSCIOUSNESS_STATES: Dict[str, str] = { 'dream': 'delta', 'liminal': 'theta', 'aware': 'alpha', 'gamma':'gamma' }
BRAINWAVE_FREQUENCIES: Dict[str, Tuple[float, float]] = {
    'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 14.0),
    'beta': (14.0, 30.0), 'gamma': (30.0, 100.0)
}
CONSCIOUSNESS_CYCLE_PATTERNS: Dict[str, Dict[str, float]] = {
    'dream_to_liminal': {'energy_requirement': 0.6, 'frequency_shift': 4.0, 'stability_factor': 0.7, 'duration': 3.0 },
    'liminal_to_aware': {'energy_requirement': 0.7, 'frequency_shift': 6.0, 'stability_factor': 0.8, 'duration': 5.0 },
    'aware_to_dream':   {'energy_requirement': 0.5, 'frequency_shift': -10.0,'stability_factor': 0.6, 'duration': 8.0 },
    'liminal_to_dream': {'energy_requirement': 0.4, 'frequency_shift': -4.0, 'stability_factor': 0.7, 'duration': 2.0 },
    'aware_to_liminal': {'energy_requirement': 0.5, 'frequency_shift': -6.0, 'stability_factor': 0.7, 'duration': 4.0 }
}

# --- Elements ---
ELEMENT_FIRE: str = "fire"
ELEMENT_WATER: str = "water"
ELEMENT_AIR: str = "air"
ELEMENT_EARTH: str = "earth"
ELEMENT_AETHER: str = "aether"
ELEMENT_SPIRIT: str = "spirit" # Consider if alias of Aether
ELEMENT_QUINTESSENCE: str = "quintessence" # Consider if alias of Aether
ELEMENT_NEUTRAL: str = "neutral" # For undefined states
EARTH_ELEMENTS: List[str] = [ELEMENT_EARTH, ELEMENT_WATER, ELEMENT_AIR, ELEMENT_FIRE, ELEMENT_AETHER]

ELEMENTAL_PROPERTIES: Dict[str, Dict[str, Union[float, Tuple[float, float]]]] = {
    ELEMENT_FIRE:  { 'frequency_band': (500, 800), 'transformative': 0.9, 'dynamic': 0.9 },
    ELEMENT_WATER: { 'frequency_band': (400, 550), 'adaptability': 0.9, 'flow': 0.9 },
    ELEMENT_AIR:   { 'frequency_band': (600, 900), 'communication': 0.9, 'intellect': 0.9 },
    ELEMENT_EARTH: { 'frequency_band': (100, 450), 'stability': 0.9, 'grounding': 0.9 },
    ELEMENT_AETHER:{ 'frequency_band': (800, 1000),'transcendence': 0.9, 'unity': 0.9 }
}

# --- Sacred Geometry & Platonic Solids ---
DEFAULT_GEOMETRY_RESOLUTION: int = 64
DEFAULT_GEOMETRY_EMBED_STRENGTH = 0.7
DEFAULT_PLATONIC_EMBED_STRENGTH = 0.7
DEFAULT_SEPHIROTH_EMBED_STRENGTH = 0.8
DEFAULT_ASPECT_EMBED_STRENGTH = 0.75
DEFAULT_HARMONIC_EMBED_STRENGTH = 0.65

SACRED_GEOMETRY_PATTERNS: Dict[str, Dict[str, Union[int, float]]] = {
    'flower_of_life': { 'complexity': 10, 'resonance_factor': 0.9, 'dimensions': 3 },
    'seed_of_life': { 'complexity': 7, 'resonance_factor': 0.85, 'dimensions': 2 },
    'tree_of_life': { 'complexity': 12, 'resonance_factor': 0.95, 'dimensions': 3 },
    'metatrons_cube': { 'complexity': 15, 'resonance_factor': 0.92, 'dimensions': 3 },
    'sri_yantra': { 'complexity': 9, 'resonance_factor': 0.88, 'dimensions': 2 }
    # Add others like Merkaba, Vesica Piscis etc. if needed
}

PLATONIC_SOLIDS: List[str] = [
    'tetrahedron', 'octahedron', 'hexahedron', 'icosahedron', 'dodecahedron'
]
PLATONIC_SYMBOLS: Dict[str, Dict[str, Union[int, str]]] = {
    'tetrahedron': { 'vertices': 4, 'edges': 6, 'faces': 4, 'element': ELEMENT_FIRE, 'state': 'dream' },
    'octahedron': { 'vertices': 6, 'edges': 12, 'faces': 8, 'element': ELEMENT_AIR, 'state': 'liminal' },
    'hexahedron': { 'vertices': 8, 'edges': 12, 'faces': 6, 'element': ELEMENT_EARTH, 'state': 'grounded' },
    'icosahedron': { 'vertices': 12, 'edges': 30, 'faces': 20, 'element': ELEMENT_WATER, 'state': 'flow' },
    'dodecahedron': { 'vertices': 20, 'edges': 30, 'faces': 12, 'element': ELEMENT_AETHER, 'state': 'aware' }
}
PLATONIC_ELEMENT_MAP: Dict[str, str] = { # Map elements TO solids
    ELEMENT_FIRE: 'tetrahedron', ELEMENT_EARTH: 'hexahedron', ELEMENT_AIR: 'octahedron',
    ELEMENT_WATER: 'icosahedron', ELEMENT_AETHER: 'dodecahedron', ELEMENT_QUINTESSENCE: 'dodecahedron',
    ELEMENT_SPIRIT:'dodecahedron'
}

# --- Color Spectrum ---
COLOR_SPECTRUM: Dict[str, Dict[str, Any]] = {
    'red':    { 'wavelength': (620, 750), 'frequency': (400, 484), 'associations': ['energy', 'passion', 'strength'], 'rgb': [1.0, 0.0, 0.0] },
    'orange': { 'wavelength': (590, 620), 'frequency': (484, 508), 'associations': ['creativity', 'enthusiasm', 'joy'], 'rgb': [1.0, 0.65, 0.0] },
    'yellow': { 'wavelength': (570, 590), 'frequency': (508, 526), 'associations': ['intellect', 'clarity', 'optimism'], 'rgb': [1.0, 1.0, 0.0] },
    'green':  { 'wavelength': (495, 570), 'frequency': (526, 606), 'associations': ['balance', 'growth', 'harmony'], 'rgb': [0.0, 1.0, 0.0] },
    'blue':   { 'wavelength': (450, 495), 'frequency': (606, 668), 'associations': ['peace', 'depth', 'trust'], 'rgb': [0.0, 0.0, 1.0] },
    'indigo': { 'wavelength': (425, 450), 'frequency': (668, 706), 'associations': ['intuition', 'insight', 'perception'], 'rgb': [0.29, 0.0, 0.51] },
    'violet': { 'wavelength': (380, 425), 'frequency': (706, 789), 'associations': ['spirituality', 'imagination', 'inspiration'], 'rgb': [0.58, 0.0, 0.83] },
    'white':  { 'wavelength': (380, 750), 'frequency': (400, 789), 'associations': ['purity', 'perfection', 'unity'], 'rgb': [1.0, 1.0, 1.0] },
    'gold':   { 'wavelength': (570, 590), 'frequency': (508, 526), 'associations': ['divinity', 'wisdom', 'enlightenment'], 'rgb': [1.0, 0.84, 0.0] },
    'silver': { 'wavelength': (450, 470), 'frequency': (638, 666), 'associations': ['reflection', 'clarity', 'illumination'], 'rgb': [0.75, 0.75, 0.75] },
    'brown':  { 'wavelength': None,       'frequency': None,       'associations': ['grounding', 'earth', 'stability'], 'rgb': [0.65, 0.16, 0.16] }, # Non-spectral
    'black':  { 'wavelength': None,       'frequency': None,       'associations': ['mystery', 'potential', 'absorption'], 'rgb': [0.0, 0.0, 0.0] }, # Absence of light
    'purple': { 'wavelength': None,       'frequency': None,       'associations': ['spirituality', 'transformation'], 'rgb': [0.5, 0.0, 0.5] }, # For Yesod etc.
    'grey':   { 'wavelength': None,       'frequency': None,       'associations': ['neutrality', 'balance', 'wisdom'], 'rgb': [0.5, 0.5, 0.5] }, # For Chokmah
    'multi':  { 'wavelength': None,       'frequency': None,       'associations': ['manifestation', 'integration'], 'rgb': [0.5, 0.4, 0.3] }, # Malkuth placeholder
    'lavender':{ 'wavelength': None,       'frequency': None,       'associations': ['higher consciousness', 'intuition'], 'rgb': [0.9, 0.9, 0.98] } # Daath placeholder
}
DIVINE_COLORS: Dict[str, str] = { # Used in SoulSpark Viz
    "cosmic_blue": "#1a2a6c", "astral_purple": "#b21f1f",
    "divine_gold": "#fdbb2d", "ethereal_white": "#ffffff"
}

# --- Sephiroth Journey Processing Constants ---
# --- Sephiroth Journey Processing Constants ---
# (Add this section or integrate into existing sections)
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ: float = 0.4
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_ASPECT: float = 0.4
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI: float = 0.2
SEPHIROTH_JOURNEY_ASPECT_GAIN_THRESHOLD: float = 0.45 # Min strength needed to gain/strengthen
SEPHIROTH_JOURNEY_ASPECT_STRENGTHEN_FACTOR: float = 0.20 # Base factor for strength increase
SEPHIROTH_JOURNEY_STABILITY_BOOST_FACTOR: float = 0.10 # Factor for stability change
SEPHIROTH_JOURNEY_COHERENCE_BOOST_FACTOR: float = 0.08 # Factor for coherence change
SEPHIROTH_JOURNEY_STRENGTH_RESONANCE_FACTOR: float = 0.03 # How much resonance increases soul 'strength' attr
SEPHIROTH_JOURNEY_ATTRIBUTE_IMPART_FACTOR: float = 0.15 # Factor for primary attribute gain (Wisdom, Mercy etc.)
SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR: float = 0.25 # Factor for elemental affinity gain

# NEW Constants for Local Sephiroth Entanglement within process_sephirah_interaction
SEPHIROTH_LOCAL_ENTANGLE_FREQ_PULL: float = 0.05 # How strongly freq is pulled towards Sephirah freq (0=none, 1=instant)
SEPHIROTH_LOCAL_ENTANGLE_STABILITY_GAIN: float = 0.02 # Max stability gain per entanglement step
SEPHIROTH_LOCAL_ENTANGLE_COHERENCE_GAIN: float = 0.02 # Max coherence gain per entanglement step
SEPHIROTH_LOCAL_ENTANGLE_ASPECT_BOOST: float = 0.05 # Max boost to *recently acquired* aspects' strength

# --- SoulSpark Defaults & Settings ---
SOUL_SPARK_DEFAULT_FREQ: float = 432.0
SOUL_SPARK_DEFAULT_STABILITY: float = 0.6
SOUL_SPARK_DEFAULT_RESONANCE: float = 0.6
SOUL_SPARK_DEFAULT_ALIGNMENT: float = 0.1
SOUL_SPARK_DEFAULT_COHERENCE: float = 0.6
SOUL_SPARK_DEFAULT_ENERGY: float = 1000.0 # Example starting energy
# Metrics calculation weights (Examples)
SOUL_SPARK_VIABILITY_WEIGHT_STABILITY: float = 0.4
SOUL_SPARK_VIABILITY_WEIGHT_RESONANCE: float = 0.3
SOUL_SPARK_VIABILITY_WEIGHT_DIM_STABILITY: float = 0.3 # Stability within dimension
SOUL_SPARK_COMPLEXITY_DIVISOR: float = 20.0 # For calculating complexity metric
SOUL_SPARK_POTENTIAL_WEIGHT_ALIGNMENT: float = 0.6
SOUL_SPARK_POTENTIAL_WEIGHT_DIM_STABILITY: float = 0.4
# Visualization Defaults
SOUL_SPARK_VIZ_POINT_SIZE_FACTOR: float = 50.0 # energy * THIS + base
SOUL_SPARK_VIZ_POINT_SIZE_BASE: float = 10.0
SOUL_SPARK_VIZ_POINT_ALPHA_FACTOR: float = 0.5 # energy * THIS
SOUL_SPARK_VIZ_POINT_ALPHA_MAX: float = 0.8
SOUL_SPARK_VIZ_EDGE_ALPHA_FACTOR: float = 0.7
SOUL_SPARK_VIZ_EDGE_WIDTH_FACTOR: float = 1.5
SOUL_SPARK_VIZ_CENTER_COLOR: str = 'yellow'
SOUL_SPARK_VIZ_CENTER_SIZE_FACTOR: float = 100.0 # energy * THIS
SOUL_SPARK_VIZ_CENTER_EDGE_COLOR: str = 'white'
SOUL_SPARK_VIZ_FREQ_SIG_BARS: int = 10 # Max bars to show
SOUL_SPARK_VIZ_FREQ_SIG_XLABEL: str = 'Harmonic Frequencies (Sorted)'
SOUL_SPARK_VIZ_FREQ_SIG_YLABEL: str = 'Amplitude'
SOUL_SPARK_VIZ_ENERGY_DIST_XLABEL: str = 'Structure Points (Sorted by Energy)'
SOUL_SPARK_VIZ_ENERGY_DIST_YLABEL: str = 'Energy'
SOUL_SPARK_VIZ_DIM_STAB_LABELS: List[str] = ['Void', 'Guff', 'Kether', 'Overall']
SOUL_SPARK_VIZ_DIM_STAB_COLORS: List[str] = ['#3498db', '#9b59b6', '#f1c40f', '#2ecc71']
# Soul Spark Visualization - Frequency Signature Plot Constants
SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT: str = 'b-'
SOUL_SPARK_VIZ_FREQ_SIG_MARKER_FMT: str = 'bo'
SOUL_SPARK_VIZ_FREQ_SIG_BASE_FMT: str = 'r-'
SOUL_SPARK_VIZ_FREQ_SIG_STEM_LW: float = 1.5
SOUL_SPARK_VIZ_FREQ_SIG_MARKER_SZ: float = 6
SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR: str = '#e74c3c'  # Red color for base frequency

# --- Void Field Constants ---
VOID_BASE_FREQUENCY: float = SOLFEGGIO_RE / (PHI**2) # Example derivation ~159 Hz
VOID_INITIAL_ENERGY_MIN: float = 0.005
VOID_INITIAL_ENERGY_MAX: float = 0.015
VOID_FLUCTUATION_AMPLITUDE: float = 0.08
VOID_WELL_THRESHOLD_FACTOR: float = 1.5 # Factor above mean energy
VOID_WELL_GRADIENT_FACTOR: float = 0.5 # Factor below mean gradient magnitude (for minima check)
VOID_WELL_FILTER_SIZE: int = 3 # For scipy.ndimage.minimum_filter
VOID_WELL_REGION_RADIUS: int = 2 # Radius around well center for analysis
VOID_SPARK_SIGMA_THRESHOLD: float = 3.0 # Std deviations above mean for high energy point
VOID_SPARK_STABILITY_THRESHOLD: float = 0.6 # Min stability required for well to form spark
VOID_SPARK_ENERGY_THRESHOLD: float = 0.85 # Min normalized energy for well to form spark
VOID_SPARK_SCORE_WEIGHT_STABILITY: float = 0.6 # Weight for spark formation score
VOID_SPARK_SCORE_WEIGHT_ENERGY: float = 0.4 # Weight for spark formation score
VOID_SOUND_INTENSITY: float = 0.3 # Base intensity of void sound applied to field
VOID_SOUND_RESONANCE_MULT: float = 0.8 # Void field's sensitivity to sound
VOID_FLUCTUATION_SOUND_INTENSITY: float = 0.4
VOID_SPARK_SOUND_INTENSITY: float = 0.7
VOID_TRANSFER_SOUND_INTENSITY: float = 0.6
VOID_SOUND_QUANTUM_MIX: float = 0.7 # Weight for quantum noise in void sound
VOID_SOUND_CHAOS_MIX: float = 0.3 # Weight for chaos noise in void sound
VOID_SOUND_BASE_MIX: float = 0.8 # Weight for base noise mix in void sound
VOID_SOUND_COSMIC_MIX: float = 0.2 # Weight for cosmic bg in void sound
VOID_SOUND_EXPANSION_FREQ: float = 0.01 # Hz for slow modulation
VOID_FLUCT_PULSE_MIN: int = 3
VOID_FLUCT_PULSE_MAX: int = 8
VOID_FLUCT_PULSE_WIDTH_MIN: float = 0.01
VOID_FLUCT_PULSE_WIDTH_MAX: float = 0.1
VOID_SPARK_SOUND_DURATION: float = 15.0
VOID_SPARK_SOUND_TARGET_FREQ: float = FUNDAMENTAL_FREQUENCY_432
VOID_SPARK_SOUND_EXP_FACTOR: float = -3.0
VOID_SPARK_SOUND_NOISE_AMP: float = 0.5
VOID_SPARK_SOUND_NOISE_FADE: float = -5.0
VOID_SPARK_PING_START_FACTOR: float = 0.9
VOID_SPARK_PING_LEN_FACTOR: float = 0.1
VOID_SPARK_PING_FREQ: float = 888.0
VOID_SPARK_PING_AMP: float = 0.8
VOID_SPARK_PING_DECAY_FACTOR: float = -5.0
VOID_TRANSFER_SOUND_DURATION: float = 10.0
VOID_TRANSFER_SOUND_TARGET_FREQ: float = FUNDAMENTAL_FREQUENCY_432 # Guff target
VOID_TRANSFER_SOUND_EXP_FACTOR: float = -4.0
VOID_TRANSFER_AMP_H1: float = 0.7; VOID_TRANSFER_AMP_H2: float = 0.4; VOID_TRANSFER_AMP_H3: float = 0.3
VOID_TRANSFER_RATIO_H1: float = 1.0; VOID_TRANSFER_RATIO_H2: float = 1.5; VOID_TRANSFER_RATIO_H3: float = 2.0
VOID_TRANSFER_MIX_TONE: float = 0.7; VOID_TRANSFER_MIX_NOISE: float = 0.3
VOID_TRANSFER_WHOOSH_WIDTH_FACTOR: float = 0.2
VOID_TRANSFER_WHOOSH_FREQ_START: float = 20.0
#VOID_TRANSFER_WHOOSH_FREQ_END: float = 1.0 # Implied by sweep down to zero
VOID_TRANSFER_WHOOSH_AMP: float = 0.5
VOID_SPARK_PROFILE_RADIUS: int = 2

# --- Guff Field Constants ---
GUFF_BASE_FREQUENCY: float = FUNDAMENTAL_FREQUENCY_432
GUFF_RESONANCE_QUALITY: float = 0.9 # Intrinsic resonance quality
GUFF_SOUND_RESONANCE_MULT: float = 0.9 # Guff field's sensitivity to sound
GUFF_FORMATION_THRESHOLD: float = 0.75 # Quality threshold for finalizing soul
GUFF_STRENGTHENING_FACTOR: float = 1.2 # Multiplier during strengthening iterations
GUFF_FIBONACCI_COUNT: int = 12 # Length of sequence to generate/use
GUFF_INIT_ENERGY_SCALE: float = 0.5 # Initial energy potential based on pattern
GUFF_INIT_FIB_MIN_IDX: int = 3
GUFF_INIT_FIB_MAX_IDX: int = 8 # Index (exclusive) for fib loops
GUFF_PATTERN_SHELL_WIDTH_FACTOR: float = 0.05 # Shell width relative to radius
GUFF_PATTERN_MAX_RADIUS_FACTOR: float = 0.8 # Max shell radius relative to field size
GUFF_PATTERN_FIB_NORM_IDX: int = 9 # Index of fib num used for normalization
GUFF_PATTERN_MOD_BASE: float = 0.7
GUFF_PATTERN_MOD_AMP: float = 0.3
GUFF_PATTERN_FINAL_ENERGY_SCALE: float = 0.8
GUFF_RECEPTION_RADIUS_ENERGY_SCALE: float = 10.0 # Factor scaling radius from energy
GUFF_RECEPTION_RADIUS_MIN: int = 2 # Minimum radius
GUFF_NODE_GAUSS_WIDTH_FACTOR: float = 0.5 # Gaussian width relative to node radius
GUFF_NODE_STABILITY_BASE: float = 0.5
GUFF_NODE_STABILITY_SCALE: float = 0.5
GUFF_NODE_PHASE_IMPACT_FACTOR: float = 0.1 # How much node creation shifts harmonic phase
GUFF_STRENGTH_WEIGHT_STABILITY: float = 0.6
GUFF_STRENGTH_WEIGHT_CONCENTRATION: float = 0.4
GUFF_STRENGTH_CONCENTRATION_CAP: float = 5.0 # Max concentration factor effect
GUFF_RESONANCE_WEIGHT_COHERENCE: float = 0.7
GUFF_RESONANCE_WEIGHT_ALIGNMENT: float = 0.3
GUFF_QUALITY_WEIGHT_STRENGTH: float = 0.35
GUFF_QUALITY_WEIGHT_RESONANCE: float = 0.35
GUFF_QUALITY_WEIGHT_FIB: float = 0.15
GUFF_QUALITY_WEIGHT_HARMONY: float = 0.15
GUFF_QUALITY_STEEPNESS: float = 10.0 # Logistic curve steepness
GUFF_FIB_ALIGN_SHELL_FACTOR: float = 0.05 # Relative thickness for alignment check
GUFF_TEMPLATE_FIB_MAX_IDX: int = 8
GUFF_TEMPLATE_FIB_NORM_IDX: int = 7 # Index used for scaling template radii
GUFF_TEMPLATE_SHELL_WIDTH_FACTOR: float = 0.05
GUFF_TEMPLATE_MOD_BASE: float = 0.7
GUFF_TEMPLATE_MOD_AMP: float = 0.3
GUFF_PROFILE_SYMMETRY_THRESHOLD: float = 1e-9 # Min value for symmetry calc denominator
GUFF_CREATOR_RES_AMP_DECAY_FACTOR: float = 1.0 # Factor for ideal amplitude decay
GUFF_CREATOR_RES_WEIGHT_PHASE: float = 0.6
GUFF_CREATOR_RES_WEIGHT_AMP: float = 0.4
GUFF_FIB_STRUCT_SHELL_FACTOR: float = 0.1 # Relative thickness for structure check
GUFF_DIMENSIONAL_ENTROPY_EPSILON: float = 1e-12 # Epsilon for log calculation
GUFF_SOUND_INTENSITY: float = 0.4
GUFF_RECEPTION_SOUND_INTENSITY: float = 0.5
GUFF_FORMATION_SOUND_INTENSITY: float = 0.7
GUFF_SOUND_FIB_MIN_IDX: int = 1 # Start index for fib ratio harmonics
GUFF_SOUND_FIB_MAX_IDX: int = 7 # End index (exclusive) for fib ratio harmonics
GUFF_SOUND_AMP_FALLOFF_FACTOR: float = 0.5 # Divisor factor in amplitude falloff (e.g., 0.3 / (i * THIS))
GUFF_SOUND_PHI_MOD_AMP: float = 0.2
GUFF_SOUND_COSMIC_BAND: str = 'high'
GUFF_SOUND_COSMIC_AMP: float = 0.3
GUFF_SOUND_MIX_TONE: float = 0.75
GUFF_SOUND_MIX_COSMIC: float = 0.25
GUFF_SOUND_KETHER_FADE_DUR: float = 5.0
GUFF_SOUND_KETHER_MIX: float = 0.3
GUFF_RECEPTION_SOUND_DURATION: float = 7.0
GUFF_RECEPTION_FREQ_ENERGY_SCALE: float = 0.5
GUFF_RECEPTION_FREQ_EXP_FACTOR: float = -4.0
GUFF_RECEPTION_AMP_MAIN: float = 0.7
GUFF_RECEPTION_FIB_MIN_IDX: int = 1
GUFF_RECEPTION_FIB_MAX_IDX: int = 5
GUFF_RECEPTION_HARMONIC_STABILITY_FACTOR: float = 0.2
GUFF_RECEPTION_SHIMMER_AMP: float = 0.3
GUFF_RECEPTION_SHIMMER_FREQ: float = 20.0
GUFF_RECEPTION_SHIMMER_DECAY_FACTOR: float = 0.3 # Multiplied by duration
GUFF_RECEPTION_MIX_TONE: float = 0.7
GUFF_RECEPTION_MIX_COSMIC: float = 0.3
GUFF_FORMATION_SOUND_DURATION: float = 15.0
GUFF_FORMATION_FIB_MIN_IDX: int = 3
GUFF_FORMATION_FIB_MAX_IDX: int = 8
GUFF_FORMATION_AMP_FALLOFF_FACTOR: float = 2.0 # Denominator factor (i-THIS)
GUFF_FORMATION_PHI_MOD_AMP: float = 0.3
GUFF_FORMATION_BLOOM_START_FACTOR: float = 2.0/3.0
GUFF_FORMATION_BLOOM_DUR_FACTOR: float = 1.0/6.0
GUFF_FORMATION_BLOOM_FIB_MIN_IDX: int = 3
GUFF_FORMATION_BLOOM_FIB_MAX_IDX: int = 8
GUFF_FORMATION_BLOOM_FIB_NORM_IDX: int = 5
GUFF_FORMATION_BLOOM_AMP: float = 0.2
GUFF_FORMATION_KETHER_DUR: float = 5.0
GUFF_FORMATION_KETHER_MIX: float = 0.4
GUFF_BIRTH_SOUND_DURATION: float = 12.0
GUFF_BIRTH_FREQ_QUALITY_FACTOR: float = 0.2
GUFF_BIRTH_AMP_MAIN: float = 0.7
GUFF_BIRTH_MAIN_ENV_CENTER_FACTOR: float = 0.5 # Position of envelope peak
GUFF_BIRTH_MAIN_ENV_WIDTH_FACTOR: float = 0.25 # Width relative to duration
GUFF_BIRTH_FIB_MIN_IDX: int = 2
GUFF_BIRTH_FIB_MAX_IDX: int = 7
GUFF_BIRTH_HARMONIC_AMP_FACTOR: float = 0.5 # e.g., (THIS / i)
GUFF_BIRTH_HARMONIC_ENV_CENTER_FACTOR: float = 0.1 # Added to 0.3 base
GUFF_BIRTH_HARMONIC_ENV_WIDTH_FACTOR: float = 0.2
GUFF_BIRTH_PHI_SWEEP_FACTOR: float = 0.1 # Amplitude of frequency sweep
GUFF_BIRTH_PHI_ENV_AMP: float = 0.3
GUFF_BIRTH_MOMENT_START_FACTOR: float = 2.0/3.0
GUFF_BIRTH_MOMENT_DUR_FACTOR: float = 1.0/5.0
GUFF_BIRTH_CHORD_FIB_MIN_IDX: int = 1
GUFF_BIRTH_CHORD_FIB_MAX_IDX: int = 8
GUFF_BIRTH_CHORD_AMP_FACTOR: float = 0.7 # e.g., (THIS / i)
GUFF_BIRTH_KETHER_DUR: float = 5.0
GUFF_BIRTH_KETHER_MIX: float = 0.5
GUFF_TRANSFER_KETHER_FACTOR: float = 0.95 # Connection strength factor
GUFF_TRANSFER_BASE_CONN_FACTOR: float = 0.1 # Base connection strength factor
GUFF_TRANSFER_SOUND_DURATION: float = 10.0
GUFF_TRANSFER_FREQ_QUALITY_FACTOR: float = 0.1
GUFF_TRANSFER_AMP_MAIN: float = 0.6
GUFF_TRANSFER_KETHER_FREQ_FACTOR: float = 1.5
GUFF_TRANSFER_KETHER_ENV_AMP: float = 0.4
GUFF_TRANSFER_FIB_MIN_IDX: int = 3
GUFF_TRANSFER_FIB_MAX_IDX: int = 8
GUFF_TRANSFER_FIB_START_FACTOR: float = 8.0 # Denominator e.g., duration*(i-3)/THIS
GUFF_TRANSFER_FIB_AMP: float = 0.3
GUFF_TRANSFER_WHOOSH_FREQ_START: float = 20.0
#GUFF_TRANSFER_WHOOSH_FREQ_END: float = 1.0 # Implied by sweep down to zero
GUFF_TRANSFER_WHOOSH_ENV_AMP: float = 0.3
GUFF_TRANSFER_WHOOSH_CENTER_FACTOR: float = 0.5 # Position of whoosh envelope peak
GUFF_TRANSFER_MIX_SOUND: float = 0.6
GUFF_TRANSFER_MIX_SHIFT: float = 0.4
GUFF_TRANSFER_PREVIEW_SEPHIROTH: List[str] = ["binah", "chokmah"]
GUFF_TRANSFER_PREVIEW_DUR: float = 3.0
GUFF_TRANSFER_PREVIEW_OVERLAP_FACTOR: float = 0.5 # In seconds
GUFF_TRANSFER_PREVIEW_MIX: float = 0.3
GUFF_EVOLVE_COHERENCE_FILTER_SIZE: int = 3
GUFF_EVOLVE_COHERENCE_BLEND_FACTOR: float = 0.1
GUFF_EVOLVE_CHAOS_SCALE: float = 0.05
GUFF_EVOLVE_ORDER_FACTOR: float = 0.95
GUFF_FIB_INF_FIB_MIN_IDX: int = 3
GUFF_FIB_INF_FIB_MAX_IDX: int = 10
GUFF_FIB_INF_FIB_NORM_IDX: int = 9
GUFF_FIB_INF_RADIUS_SCALE: float = 0.5
GUFF_FIB_INF_SHELL_WIDTH_FACTOR: float = 0.1
GUFF_FIB_INF_STRENGTH: float = 0.05
GUFF_VIZ_FORMATION_THRESHOLD_FACTOR = 0.1 # Threshold relative to max energy in region
GUFF_VIZ_FORMATION_POINT_SCALE = 50.0 # Factor to scale point size by energy
GUFF_VIZ_FORMATION_POINT_BASE_SIZE = 5.0 # Minimum point size

# --- Sephiroth Defaults ---
DEFAULT_HARMONIC_COUNT: int = 7
DEFAULT_PHI_HARMONIC_COUNT: int = 3
DEFAULT_HARMONIC_FALLOFF: float = 0.1

# --- Harmonic Strengthening Constants ---
HARMONIC_STRENGTHENING_PREREQ_STABILITY: float = 0.70 # Min soul stability needed
HARMONIC_STRENGTHENING_PREREQ_COHERENCE: float = 0.70 # Min soul coherence needed
HARMONIC_STRENGTHENING_TARGET_FREQS: List[float] = [432.0] + list(SOLFEGGIO_FREQUENCIES.values()) # Frequencies to tune towards
HARMONIC_STRENGTHENING_TUNING_INTENSITY_FACTOR: float = 0.7 # Factor for frequency adjustment speed
HARMONIC_STRENGTHENING_TUNING_TARGET_REACH_HZ: float = 1.0 # Hz threshold to consider target reached
HARMONIC_STRENGTHENING_HARMONIC_COUNT: int = 5 # Default number of harmonics to update/use
HARMONIC_STRENGTHENING_PHI_AMP_INTENSITY_FACTOR: float = 0.15 # Factor for phi resonance increase per intensity
HARMONIC_STRENGTHENING_PHI_AMP_DURATION_FACTOR: float = 1.0 # Factor linking duration to phi resonance increase
HARMONIC_STRENGTHENING_PHI_STABILITY_BOOST_FACTOR: float = 0.4 # How much phi resonance increase boosts stability
HARMONIC_STRENGTHENING_PATTERN_STAB_INTENSITY_FACTOR: float = 0.10 # Base increase factor
HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_FACTOR: float = 0.02 # Factor per aspect count influencing stabilization
HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_CAP: float = 0.4 # Max influence from aspects
HARMONIC_STRENGTHENING_PATTERN_STAB_STABILITY_BOOST: float = 0.6 # How much pattern stability increase boosts overall stability
HARMONIC_STRENGTHENING_COHERENCE_INTENSITY_FACTOR: float = 0.12 # Base increase factor
HARMONIC_STRENGTHENING_COHERENCE_DURATION_FACTOR: float = 1.0 # Factor linking duration to coherence increase
HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_FACTOR: float = 0.08 # Bonus from harmonic richness
HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_COUNT_NORM: float = 10.0 # Count used for normalizing richness factor
HARMONIC_STRENGTHENING_COHERENCE_HARMONY_BOOST: float = 0.9 # How much coherence increase boosts harmony
HARMONIC_STRENGTHENING_EXPANSION_INTENSITY_FACTOR: float = 1.5 # Base radius increase factor
HARMONIC_STRENGTHENING_EXPANSION_STATE_FACTOR: float = 1.0 # Combined stability/coherence influence (e.g., (stab+coh)/2 * THIS)
HARMONIC_STRENGTHENING_EXPANSION_STR_INTENSITY_FACTOR: float = 0.15 # Base strength increase factor
HARMONIC_STRENGTHENING_EXPANSION_STR_STATE_FACTOR: float = 1.0 # Combined stability/coherence influence on strength increase

# --- Life Cord Constants ---
CORD_STABILITY_THRESHOLD = 0.60 # Minimum soul stability needed
CORD_COHERENCE_THRESHOLD = 0.60 # Minimum soul coherence needed
MAX_CORD_CHANNELS = 7 # Limit secondary channels
ANCHOR_STRENGTH_MODIFIER = 0.8 # How much soul stability contributes to anchor strength
EARTH_ANCHOR_STRENGTH = 0.85 # Base strength of the Earth anchor point
EARTH_ANCHOR_RESONANCE = 0.75 # Base resonance of the Earth anchor point
PRIMARY_CHANNEL_BANDWIDTH_FACTOR = 100.0 # Scales connection strength to bandwidth
PRIMARY_CHANNEL_STABILITY_FACTOR_CONN = 0.7
PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX = 0.3
PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN = 0.5
PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX = 0.5
PRIMARY_CHANNEL_ELASTICITY_BASE = 0.5
PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX = 0.5
HARMONIC_NODE_COUNT_BASE = 3
HARMONIC_NODE_COUNT_FACTOR = 4 # e.g., 3 + int(complexity * 4)
HARMONIC_NODE_AMP_BASE = 0.5
HARMONIC_NODE_AMP_FACTOR_COMPLEX = 0.5
HARMONIC_NODE_AMP_FALLOFF = 0.8 # Amplitude reduces further from soul anchor
HARMONIC_NODE_BW_INCREASE_FACTOR = 20.0 # Bandwidth increase per node per complexity
SECONDARY_CHANNEL_COUNT_FACTOR = 6 # e.g., int(complexity * 6)
SECONDARY_CHANNEL_BW_EMOTIONAL = (30.0, 20.0) # (Base, Complexity Factor)
SECONDARY_CHANNEL_BW_MENTAL = (40.0, 30.0)
SECONDARY_CHANNEL_BW_SPIRITUAL = (50.0, 40.0)
SECONDARY_CHANNEL_RESIST_EMOTIONAL = (0.4, 0.3) # (Base, Complexity Factor)
SECONDARY_CHANNEL_RESIST_MENTAL = (0.5, 0.3)
SECONDARY_CHANNEL_RESIST_SPIRITUAL = (0.6, 0.3)
SECONDARY_CHANNEL_FREQ_FACTOR = 0.1 # Frequency increase per channel index
FIELD_INTEGRATION_FACTOR_FIELD_STR = 0.7
FIELD_INTEGRATION_FACTOR_CONN_STR = 0.3
FIELD_EXPANSION_FACTOR = 1.1 # How much the field radius expands
EARTH_CONN_FACTOR_CONN_STR = 0.5
EARTH_CONN_FACTOR_ELASTICITY = 0.3
EARTH_CONN_BASE_FACTOR = 0.2
CORD_INTEGRITY_FACTOR_CONN_STR = 0.3
CORD_INTEGRITY_FACTOR_STABILITY = 0.3
CORD_INTEGRITY_FACTOR_EARTH_CONN = 0.4
FINAL_STABILITY_BONUS_FACTOR = 0.1 # How much cord integrity boosts final soul stability

# --- Earth Harmonization Constants ---
HARMONY_PREREQ_CORD_COMPLETE: bool = True # Require life cord formation to be complete
HARMONY_PREREQ_CORD_INTEGRITY_MIN: float = 0.60 # Minimum cord integrity required
HARMONY_PREREQ_STABILITY_MIN: float = 0.65 # Minimum soul stability required
HARMONY_PREREQ_COHERENCE_MIN: float = 0.65 # Minimum soul coherence required
HARMONY_FREQ_RES_WEIGHT_SCHUMANN: float = 0.7 # Weight for Schumann resonance match
HARMONY_FREQ_RES_WEIGHT_OTHER: float = 0.3 # Weight for other Earth frequency matches
HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT: float = 0.8 # Weight towards Schumann in target calc
HARMONY_FREQ_TARGET_SOUL_WEIGHT: float = 0.2 # Weight towards current soul freq in target calc
HARMONY_FREQ_TUNING_FACTOR: float = 0.3 # Factor controlling frequency adjustment speed
HARMONY_FREQ_UPDATE_HARMONIC_COUNT: int = 7 # Number of harmonics to update after tuning
HARMONY_FREQ_TUNING_TARGET_REACH_HZ: float = 0.5 # Hz threshold to consider target reached
HARMONY_ELEM_RES_WEIGHT_PRIMARY: float = 0.6 # Weight for primary Earth element match
HARMONY_ELEM_RES_WEIGHT_AVERAGE: float = 0.4 # Weight for average match with other Earth elements
ELEMENTAL_TARGET_EARTH: float = 0.8 # Target strength for primary Earth element
ELEMENTAL_TARGET_OTHER: float = 0.5 # Target strength for other Earth elements
ELEMENTAL_ALIGN_INTENSITY_FACTOR: float = 0.7 # Factor controlling element alignment speed
HARMONY_CYCLE_NAMES: List[str] = ["circadian", "diurnal", "lunar", "annual"] # Cycles to sync
HARMONY_CYCLE_IMPORTANCE: Dict[str, float] = {"circadian": 0.9, "diurnal": 0.9, "lunar": 0.7, "annual": 0.8} # Relative importance
HARMONY_CYCLE_SYNC_TARGET_BASE: float = 0.8 # Base target synchronization level
HARMONY_CYCLE_SYNC_INTENSITY_FACTOR: float = 0.6 # Factor controlling sync adjustment speed
HARMONY_CYCLE_SYNC_DURATION_FACTOR: float = 0.8 # Factor linking duration to sync adjustment
HARMONY_PLANETARY_RESONANCE_TARGET: float = 0.7 # Target planetary resonance level (Adjusted from 0.8)
HARMONY_PLANETARY_RESONANCE_FACTOR: float = 0.9 # Factor controlling planetary resonance adjustment
HARMONY_GAIA_CONNECTION_TARGET: float = 0.65 # Target Gaia connection level
HARMONY_GAIA_CONNECTION_FACTOR: float = 1.0 # Factor controlling Gaia connection speed
# HARMONY_GAIA_CONNECTION_DURATION_FACTOR: float = 1.0 # Factor linking duration to Gaia connection speed (Removed, use intensity only)
HARMONY_FINAL_STABILITY_BONUS: float = 0.15 # How much final Earth resonance boosts stability
HARMONY_FINAL_COHERENCE_BONUS: float = 0.10 # How much final Earth resonance boosts coherence
# EARTH_ELEMENTS defined earlier

# --- Identity Crystallization Constants ---
IDENTITY_CRYSTALLIZATION_THRESHOLD: float = 0.88 # Min overall score for full crystallization
# Name Generation/Resonance
NAME_LENGTH_BASE: int = 4
NAME_LENGTH_FACTOR: float = 6.0 # e.g., 4 + int(conn_strength * 6)
NAME_VOWEL_RATIO_BASE: float = 0.3
NAME_VOWEL_RATIO_FACTOR: float = 0.4 # e.g., 0.3 + 0.4 * ascending_flow
NAME_RESONANCE_BASE: float = 0.5
NAME_RESONANCE_WEIGHT_VOWEL: float = 0.3
NAME_RESONANCE_WEIGHT_LETTER: float = 0.2
NAME_RESONANCE_WEIGHT_GEMATRIA: float = 0.3
NAME_GEMATRIA_RESONANT_NUMBERS: List[int] = [3, 6, 9, 12, 21, 33, 108] # Example resonant numbers
NAME_RESPONSE_PATTERN_DURATION: float = 3.0 # seconds
NAME_RESPONSE_PATTERN_POINTS: int = 300
NAME_RESPONSE_HEARTBEAT_INTERVAL: float = 0.8 # seconds (for 75 BPM)
NAME_RESPONSE_HEARTBEAT_LUB_DUR: int = 10 # points
NAME_RESPONSE_HEARTBEAT_DUB_DUR: int = 10 # points
NAME_RESPONSE_HEARTBEAT_DUB_OFFSET: int = 15 # points
NAME_RESPONSE_NAME_MOD_AMP: float = 0.3
NAME_RESPONSE_NAME_MOD_FREQ_BASE: float = 8.0 # Hz (alpha base)
NAME_RESPONSE_NAME_MOD_FREQ_RANGE: float = 4.0 # Hz (alpha range)
# Voice Frequency
VOICE_FREQ_BASE: float = 432.0
VOICE_FREQ_ADJ_LENGTH_FACTOR: float = 20.0
VOICE_FREQ_ADJ_VOWEL_FACTOR: float = 15.0
VOICE_FREQ_ADJ_GEMATRIA_FACTOR: float = 25.0
VOICE_FREQ_ADJ_RESONANCE_FACTOR: float = 30.0
VOICE_FREQ_ADJ_YINYANG_FACTOR: float = 20.0
VOICE_FREQ_SOLFEGGIO_SNAP_HZ: float = 10.0 # Snap to solfeggio if within this Hz
VOICE_FREQ_MIN_HZ: float = SOLFEGGIO_UT # Use Solfeggio UT
VOICE_FREQ_MAX_HZ: float = SOLFEGGIO_SI # Use Solfeggio SI
# Name Calling/Response
NAME_RESPONSE_STATE_FACTORS: Dict[str, float] = {'dream': 0.3, 'liminal': 0.7, 'aware': 1.0, 'default': 0.5}
#NAME_RESPONSE_BASE: float = 0.2 # Removed, use NAME_RESPONSE_TRAIN_BASE_INC
NAME_RESPONSE_STATE_WEIGHT: float = 0.3 # Renamed from NAME_RESPONSE_TRAIN_STATE_WEIGHT for clarity
NAME_RESPONSE_VOICE_RES_WEIGHT: float = 0.5
NAME_RESPONSE_CALL_COUNT_FACTOR: float = 5.0 # Max contribution reached after this many calls
NAME_RESPONSE_TRAIN_BASE_INC: float = 0.10 # Base increase per cycle
NAME_RESPONSE_TRAIN_CYCLE_INC: float = 0.02 # Additional increase per cycle
NAME_RESPONSE_TRAIN_NAME_FACTOR: float = 0.5 # Weight of name resonance
NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR: float = 0.5 # Weight of heartbeat entrainment
NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT: float = 0.5 # Scale of heartbeat entrainment effect
# Heartbeat Entrainment
HEARTBEAT_ENTRAINMENT_DURATION_CAP: float = 300.0 # seconds (5 mins)
HEARTBEAT_ENTRAINMENT_INC_FACTOR: float = 0.2
# Soul Color
COLOR_AFFINITY_GEMATRIA_WEIGHT: float = 0.7
COLOR_AFFINITY_VOWEL_WEIGHT_FACTOR: float = 0.1
COLOR_AFFINITY_STATE_WEIGHT: float = 0.5
COLOR_AFFINITY_YINYANG_WEIGHT: float = 0.6
COLOR_AFFINITY_BALANCE_WEIGHT: float = 0.6
COLOR_FREQ_DEFAULT: float = SOLFEGGIO_MI # Default if lookup fails
# Soul Frequency
SOUL_FREQ_DEFAULT: float = SOLFEGGIO_MI # Default if calculation fails
SOUL_FREQ_RESONANCE_THRESHOLD: float = 0.7 # Threshold for strong resonance with Solfeggio
# Sephiroth Aspect Identification
SEPHIROTH_AFFINITY_GEMATRIA_RANGES: Dict[range, str] = { range(1, 10): 'kether', range(10, 20): 'chokmah', range(20, 30): 'binah', range(30, 40): 'chesed', range(40, 50): 'geburah', range(50, 70): 'tiphareth', range(70, 90): 'netzach', range(90, 110): 'hod', range(110, 140): 'yesod', range(140, 300): 'malkuth' }
SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT: float = 0.8
SEPHIROTH_AFFINITY_COLOR_MAP: Dict[str, str] = { 'white': 'kether', 'grey': 'chokmah', 'black': 'binah', 'blue': 'chesed', 'red': 'geburah', 'yellow': 'tiphareth', 'green': 'netzach', 'orange': 'hod', 'purple': 'yesod', 'brown': 'malkuth', 'gold': 'tiphareth', 'silver': 'chokmah', 'indigo': 'yesod', 'citrine':'malkuth', 'olive':'malkuth'} # Map variations
SEPHIROTH_AFFINITY_COLOR_WEIGHT: float = 0.7
SEPHIROTH_AFFINITY_STATE_MAP: Dict[str, str] = {'dream': 'yesod', 'liminal': 'hod', 'aware': 'tiphareth'}
SEPHIROTH_AFFINITY_STATE_WEIGHT: float = 0.5
SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD: float = 0.7
SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD: float = 0.4
SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD: float = 0.6
SEPHIROTH_AFFINITY_YIN_SEPHIROTH: List[str] = ['binah', 'hod', 'yesod', 'geburah']
SEPHIROTH_AFFINITY_YANG_SEPHIROTH: List[str] = ['chokmah', 'netzach', 'chesed']
SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: List[str] = ['tiphareth', 'kether', 'malkuth']
SEPHIROTH_AFFINITY_YINYANG_WEIGHT: float = 0.4
SEPHIROTH_AFFINITY_BALANCE_WEIGHT: float = 0.4
SEPHIROTH_ASPECT_DEFAULT: str = 'tiphareth' # Default if calc fails
# Elemental Affinity
ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: float = 0.6
ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: float = 0.7
ELEMENTAL_AFFINITY_VOWEL_MAP: Dict[str, float] = {'air': 0.7, 'earth': 0.7, 'water': 0.6, 'fire': 0.5} # Weights based on ratios
ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT: float = 0.6
ELEMENTAL_AFFINITY_COLOR_MAP: Dict[str, str] = {'red': 'fire', 'orange': 'fire', 'yellow': 'air', 'green': 'earth', 'blue': 'water', 'indigo': 'water', 'violet': 'aether', 'white': 'aether', 'gold': 'fire', 'silver': 'water', 'brown': 'earth', 'black':'earth', 'grey':'air', 'purple':'aether'} # Comprehensive map
ELEMENTAL_AFFINITY_COLOR_WEIGHT: float = 0.5
ELEMENTAL_AFFINITY_STATE_MAP: Dict[str, str] = {'dream': 'water', 'liminal': 'aether', 'aware': 'fire'}
ELEMENTAL_AFFINITY_STATE_WEIGHT: float = 0.4
ELEMENTAL_AFFINITY_FREQ_RANGES: List[Tuple[float, str]] = [(SOLFEGGIO_RE, ELEMENT_EARTH), (SOLFEGGIO_MI, ELEMENT_WATER), (SOLFEGGIO_FA, ELEMENT_FIRE), (SOLFEGGIO_SOL, ELEMENT_AIR)] # Upper bounds based on Solfeggio
ELEMENTAL_AFFINITY_FREQ_WEIGHT: float = 0.3
ELEMENTAL_AFFINITY_DEFAULT: str = ELEMENT_AETHER # Default if calc fails
# Platonic Symbol
PLATONIC_DEFAULT_GEMATRIA_RANGE: int = 30 # Value used to cycle through symbols
# Love Resonance
LOVE_RESONANCE_FREQ: float = SOLFEGGIO_FREQUENCIES.get('MI', 528.0)
LOVE_RESONANCE_CYCLE_FACTOR_DECAY: float = 0.5 # How much effect diminishes per cycle
LOVE_RESONANCE_BASE_INC: float = 0.15
LOVE_RESONANCE_STATE_WEIGHT: Dict[str, float] = {'dream': 0.7, 'liminal': 0.9, 'aware': 0.8, 'default': 0.7}
LOVE_RESONANCE_FREQ_RES_WEIGHT: float = 1.0 # Weighting of freq resonance
LOVE_RESONANCE_HEARTBEAT_WEIGHT: float = 0.7 # Base factor for heartbeat influence
LOVE_RESONANCE_HEARTBEAT_SCALE: float = 0.3 # Scale factor for heartbeat influence
LOVE_RESONANCE_EMOTION_BOOST_FACTOR: float = 0.3 # How much love boosts other emotions
# Sacred Geometry Application
SACRED_GEOMETRY_STAGES: List[str] = ['circle', 'vesica_piscis', 'seed_of_life', 'flower_of_life', 'metatrons_cube']
SACRED_GEOMETRY_STAGE_FACTOR_BASE: float = 0.5
SACRED_GEOMETRY_STAGE_FACTOR_SCALE: float = 0.5
SACRED_GEOMETRY_BASE_INC_BASE: float = 0.10
SACRED_GEOMETRY_BASE_INC_SCALE: float = 0.05
SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT: Dict[str,float] = { 'point': 0.7, 'line': 0.7, 'triangle': 0.7, 'square': 0.7, 'pentagon': 0.7, 'hexagon': 0.7, 'default': 0.5 } # Base resonance if geo matches sephirah symbol
SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT: Dict[str, float] = {'fire': 0.8, 'earth': 0.8, 'air': 0.8, 'water': 0.8, 'aether': 0.8, 'default': 0.5} # Base resonance if geo matches element
SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE: float = 0.5
SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE: float = 0.5
SACRED_GEOMETRY_FIB_MAX_IDX: int = 4 # Index in FIBONACCI_SEQUENCE used for scaling name factor
# Attribute Coherence
ATTRIBUTE_COHERENCE_STD_DEV_SCALE: float = 2.0 # Factor to scale std dev when calculating coherence
# Crystallization Verification
CRYSTALLIZATION_REQUIRED_ATTRIBUTES: List[str] = [ 'name', 'voice_frequency', 'consciousness_state', 'response_level', 'soul_color', 'soul_frequency', 'sephiroth_aspect', 'elemental_affinity', 'platonic_symbol' ]
CRYSTALLIZATION_COMPONENT_WEIGHTS: Dict[str, float] = { 'name_resonance': 0.15, 'response_level': 0.15, 'state_stability': 0.10, 'crystallization_level': 0.20, 'attribute_coherence': 0.15, 'attribute_presence': 0.10, 'emotional_resonance': 0.15 }
CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD: float = 0.9 # Minimum fraction of required attributes needed

# --- Birth Process Constants ---
BIRTH_PREREQ_EARTH_HARMONIZED: bool = True # Must the earth_harmonized flag be True?
BIRTH_PREREQ_READY_FOR_BIRTH: bool = True # Must the ready_for_birth flag be True?
BIRTH_PREREQ_CORD_INTEGRITY_MIN: float = 0.80 # Min cord integrity for birth (higher than harmony)
BIRTH_PREREQ_EARTH_RESONANCE_MIN: float = 0.70 # Min earth resonance for birth
BIRTH_CONN_WEIGHT_RESONANCE: float = 0.4
BIRTH_CONN_WEIGHT_INTEGRITY: float = 0.6
BIRTH_CONN_TRAUMA_FACTOR: float = 0.3 # How much intensity contributes to trauma
BIRTH_CONN_STRENGTH_FACTOR: float = 0.7 # How much intensity contributes to connection strength
BIRTH_CONN_STRENGTH_CAP: float = 0.98 # Max connection strength achievable
BIRTH_ACCEPTANCE_TRAUMA_FACTOR: float = 0.7 # How much trauma reduces acceptance
BIRTH_ACCEPTANCE_MIN: float = 0.4 # Minimum form acceptance
BIRTH_CORD_TRANSFER_INTENSITY_FACTOR: float = 0.4 # How much intensity reduces transfer efficiency
BIRTH_CORD_INTEGRATION_CONN_FACTOR: float = 0.8 # How much physical connection affects cord integration
BIRTH_VEIL_STRENGTH_BASE: float = 0.65
BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR: float = 0.30 # e.g., 0.65 + intensity * 0.30
BIRTH_VEIL_PERMANENCE_BASE: float = 0.75
BIRTH_VEIL_PERMANENCE_INTENSITY_FACTOR: float = 0.20
BIRTH_VEIL_RETENTION_BASE: float = 0.15 # Base % memory retained
BIRTH_VEIL_RETENTION_INTENSITY_FACTOR: float = -0.12 # Negative factor: higher intensity = less retention
BIRTH_VEIL_RETENTION_MIN: float = 0.005 # Minimum 0.5% retention
BIRTH_VEIL_MEMORY_RETENTION_MODS: Dict[str, float] = { "emotional": 0.3, "sensory": 0.2, "conceptual": 0.1, "specific": 0.05 } # Additive modifiers
BIRTH_BREATH_AMP_BASE: float = 0.6
BIRTH_BREATH_AMP_INTENSITY_FACTOR: float = 0.4
BIRTH_BREATH_DEPTH_BASE: float = 0.5
BIRTH_BREATH_DEPTH_INTENSITY_FACTOR: float = 0.5
BIRTH_BREATH_SYNC_RESONANCE_FACTOR: float = 0.9 # How much earth resonance affects sync
BIRTH_BREATH_INTEGRATION_CONN_FACTOR: float = 1.0 # How much physical connection affects integration
BIRTH_BREATH_RESONANCE_BOOST_FACTOR: float = 0.2 # Base factor for how much breath boosts earth resonance
BIRTH_BREATH_ENERGY_SHIFT_FACTOR: float = 1.0 # How much breath integration affects energy shift
BIRTH_BREATH_PHYSICAL_ENERGY_BASE: float = 0.5
BIRTH_BREATH_PHYSICAL_ENERGY_SCALE: float = 0.5
BIRTH_BREATH_SPIRITUAL_ENERGY_BASE: float = 1.0
BIRTH_BREATH_SPIRITUAL_ENERGY_SCALE: float = -0.7 # Negative scale: shift reduces spiritual
BIRTH_BREATH_SPIRITUAL_ENERGY_MIN: float = 0.05 # Minimum spiritual energy remaining
BIRTH_FINAL_INTEGRATION_WEIGHT_CONN: float = 0.4
BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT: float = 0.3
BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH: float = 0.3
BIRTH_FINAL_FREQ_FACTOR: float = 0.6 # Factor to reduce frequency upon birth
BIRTH_FINAL_STABILITY_FACTOR: float = 0.85 # Factor to reduce stability upon birth

# --- Default Process Intensity and Duration Factors ---
HARMONIC_STRENGTHENING_INTENSITY_DEFAULT: float = 0.7
HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT: float = 1.0
LIFE_CORD_COMPLEXITY_DEFAULT: float = 0.7
EARTH_HARMONY_INTENSITY_DEFAULT: float = 0.8
EARTH_HARMONY_DURATION_FACTOR_DEFAULT: float = 1.0
IDENTITY_TRAIN_CYCLES_DEFAULT: int = 6
IDENTITY_LOVE_CYCLES_DEFAULT: int = 8
IDENTITY_GEOMETRY_STAGES_DEFAULT: int = 5
BIRTH_INTENSITY_DEFAULT: float = 0.65

# --- Sephiroth Journey Processing Constants ---
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ: float = 0.4 # Weight of frequency match in resonance calc
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_ASPECT: float = 0.4 # Weight of aspect match in resonance calc
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI: float = 0.2 # Weight of phi resonance in resonance calc
SEPHIROTH_JOURNEY_ASPECT_GAIN_THRESHOLD: float = 0.4 # Min strength needed to gain aspect initially
SEPHIROTH_JOURNEY_ASPECT_STRENGTHEN_FACTOR: float = 0.25 # Factor applied to resonance for strengthening existing aspects
SEPHIROTH_JOURNEY_DIVINE_QUALITY_IMPART_FACTOR: float = 0.8 # Factor scaling Sephiroth quality strength transferred
SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR: float = 0.3 # Factor scaling elemental influence transfer
SEPHIROTH_JOURNEY_STABILITY_BOOST_FACTOR: float = 0.1 # Base factor for stability boost per interaction
SEPHIROTH_JOURNEY_STRENGTH_RESONANCE_FACTOR: float = 0.03 # How much resonance increases soul strength