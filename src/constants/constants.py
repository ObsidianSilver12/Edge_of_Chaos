"""
Constants for the Edge of Chaos Soul Development Framework.

This module defines key constants used throughout the system, including
frequencies, ratios, patterns, and other fixed values essential to the
soul development process.
"""

import numpy as np

# Fundamental constants
DIMENSIONS = 10  # Number of dimensions in the void field
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # ~1.618
EDGE_OF_CHAOS_RATIO = GOLDEN_RATIO / (GOLDEN_RATIO + 1)  # ~0.618
CREATOR_FREQUENCY = 432.0  # Hz - Universal creator frequency

# Fibonacci sequence
FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Prime numbers
PRIME_NUMBERS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Chaos-order ratios for different stages
CHAOS_ORDER_RATIOS = {
    'void': 0.618,
    'spark': 0.5,
    'guff': 0.3,
    'sephiroth': 0.4,
    'earth': 0.2
}

# Solfeggio frequencies
SOLFEGGIO_FREQUENCIES = {
    'UT': 396.0,   # Liberating guilt and fear
    'RE': 417.0,   # Undoing situations and facilitating change
    'MI': 528.0,   # Transformation and miracles
    'FA': 639.0,   # Connecting/relationships
    'SOL': 741.0,  # Expression/solutions
    'LA': 852.0,   # Awakening intuition
    'SI': 963.0    # Returning to spiritual order
}

# Harmonic frequencies related to creator frequency
HARMONIC_FREQUENCIES = {
    'fundamental': CREATOR_FREQUENCY,
    'octave': CREATOR_FREQUENCY * 2,
    'perfect_fifth': CREATOR_FREQUENCY * 3/2,
    'perfect_fourth': CREATOR_FREQUENCY * 4/3,
    'major_third': CREATOR_FREQUENCY * 5/4,
    'minor_third': CREATOR_FREQUENCY * 6/5,
    'golden_ratio': CREATOR_FREQUENCY * GOLDEN_RATIO
}

# Earth frequencies
EARTH_FREQUENCIES = {
    'schumann': 7.83,      # Schumann resonance (Hz)
    'heartbeat': 1.17,     # Typical heartbeat (Hz) - ~70 BPM
    'breath': 0.2,         # Typical breath cycle (Hz) - ~12 breaths per minute
    'circadian': 1/86400,  # Circadian rhythm (Hz) - 1 cycle per day
    'lunar': 1/2551443,    # Lunar cycle (Hz) - 1 cycle per 29.5 days
    'annual': 1/31536000   # Annual cycle (Hz) - 1 cycle per 365 days
}

# Earth energy fields
EARTH_ENERGY_FIELDS = {
    'magnetic': {
        'base_frequency': 11.75,
        'field_strength': 0.6,
        'resonance_factor': 0.8
    },
    'gravitational': {
        'base_frequency': 9.81,
        'field_strength': 0.9,
        'resonance_factor': 0.7
    },
    'electromagnetic': {
        'base_frequency': 7.83,
        'field_strength': 0.7,
        'resonance_factor': 0.85
    },
    'natural_radiation': {
        'base_frequency': 13.4,
        'field_strength': 0.4,
        'resonance_factor': 0.6
    }
}

# Consciousness states and associated brainwave frequencies
CONSCIOUSNESS_STATES = {
    'dream': 'delta',
    'liminal': 'theta',
    'aware': 'alpha'
}

BRAINWAVE_FREQUENCIES = {
    'delta': (0.5, 4.0),     # Deep sleep
    'theta': (4.0, 8.0),     # Drowsiness/meditation
    'alpha': (8.0, 14.0),    # Relaxed/calm
    'beta': (14.0, 30.0),    # Alert/working
    'gamma': (30.0, 100.0)   # High-level cognition
}

# Sacred geometry patterns
SACRED_GEOMETRY_PATTERNS = {
    'flower_of_life': {
        'complexity': 10,
        'resonance_factor': 0.9,
        'dimensions': 3,
        'base_energy': 100
    },
    'seed_of_life': {
        'complexity': 7,
        'resonance_factor': 0.85,
        'dimensions': 2,
        'base_energy': 80
    },
    'tree_of_life': {
        'complexity': 12,
        'resonance_factor': 0.95,
        'dimensions': 3,
        'base_energy': 120
    },
    'metatrons_cube': {
        'complexity': 15,
        'resonance_factor': 0.92,
        'dimensions': 3,
        'base_energy': 150
    },
    'sri_yantra': {
        'complexity': 9,
        'resonance_factor': 0.88,
        'dimensions': 2,
        'base_energy': 90
    }
}

# Platonic solids
PLATONIC_SOLIDS = [
    'tetrahedron',   # Fire element
    'octahedron',    # Air element
    'hexahedron',    # Earth element
    'icosahedron',   # Water element
    'dodecahedron'   # Aether element
]

PLATONIC_SYMBOLS = {
    'tetrahedron': {
        'vertices': 4,
        'edges': 6,
        'faces': 4,
        'element': 'fire',
        'state': 'dream'
    },
    'octahedron': {
        'vertices': 6,
        'edges': 12,
        'faces': 8,
        'element': 'air',
        'state': 'liminal'
    },
    'hexahedron': {  # Cube
        'vertices': 8,
        'edges': 12,
        'faces': 6,
        'element': 'earth',
        'state': 'grounded'
    },
    'icosahedron': {
        'vertices': 12,
        'edges': 30,
        'faces': 20,
        'element': 'water',
        'state': 'flow'
    },
    'dodecahedron': {
        'vertices': 20,
        'edges': 30,
        'faces': 12,
        'element': 'aether',
        'state': 'aware'
    }
}

# Color spectrum
COLOR_SPECTRUM = {
    'red': {
        'wavelength': (620, 750),
        'frequency': (400, 484),
        'associations': ['energy', 'passion', 'strength']
    },
    'orange': {
        'wavelength': (590, 620),
        'frequency': (484, 508),
        'associations': ['creativity', 'enthusiasm', 'joy']
    },
    'yellow': {
        'wavelength': (570, 590),
        'frequency': (508, 526),
        'associations': ['intellect', 'clarity', 'optimism']
    },
    'green': {
        'wavelength': (495, 570),
        'frequency': (526, 606),
        'associations': ['balance', 'growth', 'harmony']
    },
    'blue': {
        'wavelength': (450, 495),
        'frequency': (606, 668),
        'associations': ['peace', 'depth', 'trust']
    },
    'indigo': {
        'wavelength': (425, 450),
        'frequency': (668, 706),
        'associations': ['intuition', 'insight', 'perception']
    },
    'violet': {
        'wavelength': (380, 425),
        'frequency': (706, 789),
        'associations': ['spirituality', 'imagination', 'inspiration']
    },
    'white': {
        'wavelength': (380, 750),
        'frequency': (400, 789),
        'associations': ['purity', 'perfection', 'unity']
    },
    'gold': {
        'wavelength': (570, 590),
        'frequency': (508, 526),
        'associations': ['divinity', 'wisdom', 'enlightenment']
    },
    'silver': {
        'wavelength': (450, 470),
        'frequency': (638, 666),
        'associations': ['reflection', 'clarity', 'illumination']
    }
}

# Sephiroth aspects
SEPHIROTH_ASPECTS = {
    'kether': {
        'name': 'Crown',
        'frequency': 963.0,
        'color': 'white',
        'symbol': 'point',
        'aspects': ['divine_will', 'unity', 'oneness'],
        'element': 'aether'
    },
    'chokmah': {
        'name': 'Wisdom',
        'frequency': 852.0,
        'color': 'blue',
        'symbol': 'line',
        'aspects': ['wisdom', 'revelation', 'insight'],
        'element': 'air'
    },
    'binah': {
        'name': 'Understanding',
        'frequency': 741.0,
        'color': 'black',
        'symbol': 'triangle',
        'aspects': ['understanding', 'receptivity', 'pattern_recognition'],
        'element': 'water'
    },
    'chesed': {
        'name': 'Mercy',
        'frequency': 639.0,
        'color': 'purple',
        'symbol': 'square',
        'aspects': ['mercy', 'compassion', 'expansion'],
        'element': 'water'
    },
    'geburah': {
        'name': 'Severity',
        'frequency': 528.0,
        'color': 'red',
        'symbol': 'pentagon',
        'aspects': ['severity', 'discipline', 'discernment'],
        'element': 'fire'
    },
    'tiphareth': {
        'name': 'Beauty',
        'frequency': 528.0,
        'color': 'gold',
        'symbol': 'hexagon',
        'aspects': ['beauty', 'harmony', 'balance'],
        'element': 'fire'
    },
    'netzach': {
        'name': 'Victory',
        'frequency': 417.0,
        'color': 'green',
        'symbol': 'heptagon',
        'aspects': ['victory', 'emotion', 'appreciation'],
        'element': 'water'
    },
    'hod': {
        'name': 'Glory',
        'frequency': 396.0,
        'color': 'orange',
        'symbol': 'octagon',
        'aspects': ['glory', 'intellect', 'communication'],
        'element': 'air'
    },
    'yesod': {
        'name': 'Foundation',
        'frequency': 285.0,
        'color': 'indigo',
        'symbol': 'nonagon',
        'aspects': ['foundation', 'dreams', 'subconscious'],
        'element': 'aether'
    },
    'malkuth': {
        'name': 'Kingdom',
        'frequency': 174.0,
        'color': 'brown',
        'symbol': 'decagon',
        'aspects': ['kingdom', 'manifestation', 'physicality'],
        'element': 'earth'
    }
}

# Entanglement patterns
ENTANGLEMENT_PATTERNS = {
    'creator': {
        'complexity': 7,
        'resonance_factor': 0.95,
        'base_energy': 150
    },
    'divine': {
        'complexity': 5,
        'resonance_factor': 0.9,
        'base_energy': 120
    },
    'spiritual': {
        'complexity': 3,
        'resonance_factor': 0.85,
        'base_energy': 100
    },
    'physical': {
        'complexity': 2,
        'resonance_factor': 0.8,
        'base_energy': 80
    }
}

# Quantum coupling strength
QUANTUM_COUPLING_STRENGTH = 0.7

# Consciousness cycle patterns
CONSCIOUSNESS_CYCLE_PATTERNS = {
    'dream_to_liminal': {
        'energy_requirement': 0.6,
        'frequency_shift': 4.0,  # Hz increase
        'stability_factor': 0.7,
        'duration': 3.0  # seconds
    },
    'liminal_to_aware': {
        'energy_requirement': 0.7,
        'frequency_shift': 6.0,  # Hz increase
        'stability_factor': 0.8,
        'duration': 5.0  # seconds
    },
    'aware_to_dream': {
        'energy_requirement': 0.5,
        'frequency_shift': -10.0,  # Hz decrease
        'stability_factor': 0.6,
        'duration': 8.0  # seconds
    },
    'liminal_to_dream': {
        'energy_requirement': 0.4,
        'frequency_shift': -4.0,  # Hz decrease
        'stability_factor': 0.7,
        'duration': 2.0  # seconds
    },
    'aware_to_liminal': {
        'energy_requirement': 0.5,
        'frequency_shift': -6.0,  # Hz decrease
        'stability_factor': 0.7,
        'duration': 4.0  # seconds
    }
}

# Gateway key mappings
GATEWAY_KEYS = {
    'tetrahedron': ['tiphareth', 'netzach', 'hod'],
    'octahedron': ['binah', 'kether', 'chokmah', 'chesed', 'tiphareth', 'geburah'],
    'hexahedron': ['hod', 'netzach', 'chesed', 'chokmah', 'binah', 'geburah'],
    'icosahedron': ['kether', 'chesed', 'geburah'],
    'dodecahedron': ['hod', 'netzach', 'chesed', 'daath', 'geburah']
}

# Elemental properties
ELEMENTAL_PROPERTIES = {
    'fire': {
        'frequency': 528.0,
        'energy': 0.9,
        'transformative': 0.95,
        'resonance': 0.85
    },
    'water': {
        'frequency': 417.0,
        'energy': 0.7,
        'adaptability': 0.95,
        'resonance': 0.9
    },
    'air': {
        'frequency': 396.0,
        'energy': 0.6,
        'communication': 0.95,
        'resonance': 0.8
    },
    'earth': {
        'frequency': 174.0,
        'energy': 0.85,
        'stability': 0.95,
        'resonance': 0.75
    },
    'aether': {
        'frequency': 963.0,
        'energy': 0.95,
        'transcendence': 0.95,
        'resonance': 0.95
    }
}

"""
Constants Module

Central repository for constants used across the Soul Development Framework.
Categorized for better organization.
"""

import numpy as np
import logging # Use logging level constant

# --- General Simulation & System ---
SAMPLE_RATE: int = 44100  # Hz (Standard audio sample rate)
MAX_AMPLITUDE: float = 0.8  # Max audio amplitude to prevent clipping (-1 to 1 theoretical max)
DEFAULT_DURATION: float = 30.0 # Default duration in seconds for generated sounds
DEFAULT_DIMENSIONS_3D: tuple = (256, 256, 256) # Default grid size for 3D fields
DEFAULT_FIELD_DTYPE = np.float32 # Default data type for energy fields
DEFAULT_COMPLEX_DTYPE = np.complex64 # Default data type for complex fields
DEFAULT_GEOMETRY_RESOLUTION: int = 64 # Default grid resolution for generating geometry patterns
MAX_SNAPSHOT_ELEMENTS: int = 8192 # Limit size of array snapshots saved in state (e.g., 16^3*2)
DATA_DIR_BASE: str = "data" # Base directory for saving field data, etc.
OUTPUT_DIR_BASE: str = "output/sounds" # Base directory for saving generated sounds
LOG_LEVEL = logging.INFO # Logging level for modules
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
FLOAT_EPSILON: float = 1e-9 # Small float for safe division/comparison

# --- Mathematical & Physics Concepts ---
PI: float = np.pi
PHI: float = (1 + np.sqrt(5)) / 2  # Golden Ratio ~1.618
GOLDEN_RATIO: float = PHI # Alias
EDGE_OF_CHAOS_RATIO: float = 1 / PHI # ~0.618 - Used in void field dynamics
SQRT_2: float = np.sqrt(2)
SQRT_3: float = np.sqrt(3)
SQRT_5: float = np.sqrt(5)
SQRT_6: float = np.sqrt(6)
SQRT_7: float = np.sqrt(7)
# Add physical constants if needed for *calculations* (not just names)
# SPEED_OF_LIGHT = 299792458 etc.

# --- Frequencies ---
# Solfeggio Frequencies (Examples)
SOLFEGGIO_174: float = 174.0 # Grounding (Malkuth Base / Earth)
SOLFEGGIO_285: float = 285.0 # Healing/Energy Fields (Air Base for Octahedron)
SOLFEGGIO_396: float = 396.0 # Liberation (Geburah Base / Fire Base for Tetrahedron)
SOLFEGGIO_417: float = 417.0 # Change (Binah Base / Water Base for Icosahedron)
SOLFEGGIO_528: float = 528.0 # Transformation/Love (Tiphareth Base / Aether Base for Dodecahedron)
SOLFEGGIO_639: float = 639.0 # Relationships (Chesed Base)
SOLFEGGIO_741: float = 741.0 # Expression/Intuition (Chokmah Base / Hod Base)
SOLFEGGIO_852: float = 852.0 # Intuition (Yesod Base)
SOLFEGGIO_963: float = 963.0 # Pineal Gland/Divinity (Kether Base)
# Other Bases
FUNDAMENTAL_FREQUENCY_432: float = 432.0 # Alternative 'sacred' tuning standard
SCHUMANN_RESONANCE_BASE: float = 7.83 # Hz (Earth's fundamental resonance)

# --- Void Field Specific Constants ---
VOID_BASE_FREQUENCY: float = SOLFEGGIO_741 / (PHI**3) # ~175.6 Hz Example derivation
VOID_INITIAL_ENERGY_MIN: float = 0.005
VOID_INITIAL_ENERGY_MAX: float = 0.015
VOID_FLUCTUATION_AMPLITUDE: float = 0.08
VOID_WELL_THRESHOLD_FACTOR: float = 1.5 # Factor above mean energy
VOID_WELL_GRADIENT_FACTOR: float = 0.5 # Factor below mean gradient magnitude (for minima check)
VOID_WELL_FILTER_SIZE: int = 3
VOID_WELL_REGION_RADIUS: int = 2
VOID_SPARK_SIGMA_THRESHOLD: float = 3.0 # For high energy point detection
VOID_SPARK_STABILITY_THRESHOLD: float = 0.6
VOID_SPARK_ENERGY_THRESHOLD: float = 0.85 # Normalized threshold
VOID_SOUND_INTENSITY: float = 0.3
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
VOID_TRANSFER_WHOOSH_FREQ: float = 5.0
VOID_TRANSFER_WHOOSH_AMP: float = 0.5
VOID_SOUND_RESONANCE_MULT: float = 0.8 # Field's own sensitivity to sound
VOID_SPARK_PROFILE_RADIUS: int = 2

# --- Guff Field Specific Constants ---
GUFF_BASE_FREQUENCY: float = FUNDAMENTAL_FREQUENCY_432
GUFF_RESONANCE_QUALITY: float = 0.9
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
GUFF_TRANSFER_WHOOSH_FREQ_END: float = 1.0
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

# --- Sephiroth Defaults (Used by Dictionary if Aspect files fail) ---
# It's better if Aspect Dictionary fails hard, but keep these as ultimate fallback
DEFAULT_HARMONIC_COUNT: int = 7
DEFAULT_PHI_HARMONIC_COUNT: int = 3
DEFAULT_HARMONIC_FALLOFF: float = 0.1

# --- Elements / Colors / Planets (Already defined in draft, ensure comprehensive) ---
# (Keep element strings, color strings, planet strings)

# --- File Paths ---
# (Keep LOG_FILE_PATH, METRICS_DIR, etc. - construct using os.path.join)
METRICS_DIR = os.path.join(DATA_DIR_BASE, "metrics")
SOUND_OUTPUT_DIR = OUTPUT_DIR_BASE
FIELD_DATA_DIR = os.path.join(DATA_DIR_BASE, "fields")

# --- Module Specific Constants ---
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