"""
Central Constants for the Soul Development Framework (Version 4.2 - Reorganized)

Consolidated and validated constants for simulation parameters, physics,
field properties, soul defaults, stage thresholds, geometry, and brain development.
All duplicates have been identified and marked with (D).
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from enum import Enum

# =============================================================================
# CORE PHYSICS & MATHEMATICAL CONSTANTS
# =============================================================================

# Mathematical Constants
PI: float = np.pi
GOLDEN_RATIO: float = (1 + np.sqrt(5)) / 2.0  # Phi (~1.618)
PHI: float = GOLDEN_RATIO
# (D) PHI = 1.618033988749895  # Duplicate from evolve constants
SILVER_RATIO: float = 1 + np.sqrt(2)
EDGE_OF_CHAOS_RATIO: float = 1.0 / PHI
EDGE_OF_CHAOS_DEFAULT = 0.618  # Golden ratio - optimal edge of chaos parameter
FLOAT_EPSILON: float = 1e-9
# (D) FLOAT_EPSILON = 1e-9  # Duplicate from evolve constants

# Physics Constants
PLANCK_CONSTANT_H = 6.626e-34  # J*s
SPEED_OF_LIGHT: float = 299792458.0  # Speed of light in m/s
# (D) SPEED_OF_LIGHT = 299792458.0  # Duplicate from evolve constants
SPEED_OF_SOUND: float = 343.0  # Speed of sound in m/s at 20C air
# (D) SPEED_OF_SOUND = 343.0  # Duplicate definition
AIR_DENSITY: float = 1.225  # Air density in kg/m³ at sea level

# Frequency Constants
SCHUMANN_FREQUENCY: float = 7.83  # Hz
FUNDAMENTAL_FREQUENCY_432: float = 432.0
INITIAL_SPARK_BASE_FREQUENCY_HZ: float = 432.0
KETHER_FREQ: float = 963.0  # Example Kether base freq for Guff resonance calc
LOVE_RESONANCE_FREQ = 528.0
EARTH_FREQUENCY = 136.10
EARTH_BREATH_FREQUENCY = 0.2

# =============================================================================
# ENERGY UNITS & CONVERSIONS
# =============================================================================

# Soul Energy Units (SEU) - Primary energy system
MAX_SOUL_ENERGY_SEU: float = 1e6
INITIAL_SPARK_ENERGY_SEU: float = 500.0
VOID_BASE_ENERGY_SEU: float = 10.0
PASSIVE_ENERGY_DISSIPATION_RATE_SEU_PER_SEC: float = 0.1
CORD_ACTIVATION_ENERGY_COST: float = 50.0  # SEU cost for activating a life cord

# Brain Energy Units (BEU) - Secondary energy system
BRAIN_MIN_ENERGY_BEU: float = 1.0e6  # Minimum Brain Energy Units for viability
MIN_BRAIN_SEED_ENERGY: float = 1.0  # Minimum BEU for brain seed operation
MAX_BRAIN_SEED_CAPACITY: float = 1000.0  # Maximum energy capacity for brain seed

# Energy Scale Factors & Conversions
ENERGY_SCALE_FACTOR: float = 1e14  # 1 SEU = 1e-14 Joules (Matches Synapse Energy)
# (D) ENERGY_SCALE_FACTOR: float = 1e14  # Duplicate definition
BRAIN_ENERGY_SCALE_FACTOR: float = 1e12  # 1 BEU = 1e-12 Joules (1 picoJoule)
ENERGY_UNSCALE_FACTOR: float = 1e-14  # 1 Joule = 1e14 SEU (Inverse for reporting)

# Joules Conversions (Clearer naming)
JOULES_PER_SEU: float = 1e-14  # 1 SEU = 1e-14 Joules (Matches Synapse Energy)
SEU_PER_JOULE: float = 1e14    # 1 Joule = 1e14 SEU (Inverse)
JOULES_PER_BEU: float = 1e-12  # 1 BEU = 1e-12 Joules (1 picoJoule)
BEU_PER_JOULE: float = 1e12    # 1 Joule = 1e12 BEU
# (D) BRAIN_ENERGY_UNIT_PER_JOULE: float = 1e12  # Duplicate of BEU_PER_JOULE

# Synapse Energy
SYNAPSE_ENERGY_JOULES: float = 1e-14  # Energy of one synaptic firing in Joules
# (D) SYNAPSE_ENERGY_JOULES: float = JOULES_PER_SEU  # Duplicate reference

# Derived Energy Calculations
SYNAPSES_COUNT_FOR_MIN_ENERGY: int = int(1e9)  # 1 billion synapses for baseline calc
SECONDS_IN_14_DAYS: float = 14 * 24 * 60 * 60
ENERGY_BRAIN_14_DAYS_JOULES: float = SYNAPSE_ENERGY_JOULES * SYNAPSES_COUNT_FOR_MIN_ENERGY * SECONDS_IN_14_DAYS

# =============================================================================
# GRID & DIMENSIONAL CONSTANTS  
# =============================================================================

# Primary Grid Dimensions (Brain/Field)
GRID_DIMENSIONS: Tuple[int, int, int] = (256, 256, 256)  # Primary brain structure grid
# (D) GRID_SIZE: Tuple[int, int, int] = (144, 144, 144)  # Duplicate/different field grid
# (D) BRAIN_GRID_SIZE: Tuple[int, int, int] = (256, 256, 256)  # Duplicate of GRID_DIMENSIONS
# (D) BRAIN_DEFAULT_SIZE_X = 256  # Duplicate component
# (D) BRAIN_DEFAULT_SIZE_Y = 256  # Duplicate component  
# (D) BRAIN_DEFAULT_SIZE_Z = 256  # Duplicate component

# Field System Grid (Different from brain grid)
FIELD_SYSTEM_GRID_SIZE: Tuple[int, int, int] = (64, 64, 64)  # Field controller grid

# Block/Cell Constants
DEFAULT_BLOCK_COUNT = 1000  # Approximate number of blocks to divide brain into
DEFAULT_CELL_DENSITY = 0.05  # Default 5% of cells are active
BOUNDARY_CELL_DENSITY = 1.0  # 100% of cells are active at boundaries
ACTIVE_CELL_THRESHOLD = 0.1  # Energy level to consider a cell active

# =============================================================================
# STABILITY & COHERENCE SYSTEM
# =============================================================================

# Units & Ranges
MAX_STABILITY_SU: float = 100.0
MAX_COHERENCE_CU: float = 100.0
INITIAL_STABILITY_CALC_FACTOR: float = 0.2  # Initial scale applied to calculated S
INITIAL_COHERENCE_CALC_FACTOR: float = 0.15  # Initial scale applied to calculated C

# Stability Score Calculation Weights (Sum = 1.0)
STABILITY_WEIGHT_FREQ: float = 0.30     # Contribution from frequency stability
STABILITY_WEIGHT_PATTERN: float = 0.35   # Contribution from internal structure
STABILITY_WEIGHT_FIELD: float = 0.20     # Contribution from external field influences
STABILITY_WEIGHT_TORUS: float = 0.15

# Pattern Component Weights (within STABILITY_WEIGHT_PATTERN)
STABILITY_PATTERN_WEIGHT_LAYERS: float = 0.3
STABILITY_PATTERN_WEIGHT_ASPECTS: float = 0.3
STABILITY_PATTERN_WEIGHT_PHI: float = 0.2
STABILITY_PATTERN_WEIGHT_ALIGNMENT: float = 0.2

# Coherence Score Calculation Weights (Sum = 1.0)
COHERENCE_WEIGHT_PHASE = 0.20           # Weight for phase coherence
COHERENCE_WEIGHT_HARMONY = 0.20         # Weight for harmony factor
COHERENCE_WEIGHT_HARMONIC_PURITY = 0.10  # Weight for harmonic purity
COHERENCE_WEIGHT_PATTERN = 0.15         # Weight for pattern coherence
COHERENCE_WEIGHT_FIELD = 0.15           # Weight for field influence
COHERENCE_WEIGHT_CREATOR = 0.10         # Weight for creator connection
COHERENCE_WEIGHT_TORUS = 0.10           # Weight for toroidal flow

# Verification of weight sums
assert abs((COHERENCE_WEIGHT_PHASE + COHERENCE_WEIGHT_HARMONY + 
            COHERENCE_WEIGHT_HARMONIC_PURITY + COHERENCE_WEIGHT_PATTERN + 
            COHERENCE_WEIGHT_FIELD + COHERENCE_WEIGHT_CREATOR + 
            COHERENCE_WEIGHT_TORUS) - 1.0) < 0.000001, "Coherence weights must sum to 1.0"

# Other Calculation Factors
STABILITY_VARIANCE_PENALTY_K: float = 50.0  # How much freq variance hurts stability

# =============================================================================
# BRAIN REGIONS & STRUCTURE
# =============================================================================

# Region Names
REGION_FRONTAL: str = 'frontal'
REGION_PARIETAL: str = 'parietal'
REGION_TEMPORAL: str = 'temporal'
REGION_OCCIPITAL: str = 'occipital'
REGION_LIMBIC: str = 'limbic'
REGION_BRAIN_STEM: str = 'brain_stem'
REGION_CEREBELLUM: str = 'cerebellum'

# Region List
BRAIN_REGIONS: List[str] = [REGION_FRONTAL, REGION_PARIETAL, REGION_TEMPORAL, 
                           REGION_OCCIPITAL, REGION_LIMBIC, REGION_BRAIN_STEM, REGION_CEREBELLUM]

# Region Proportions (Sums to 1.0)
REGION_PROPORTIONS: Dict[str, float] = {
    REGION_FRONTAL: 0.28,    # Updated from evolve constants (was 0.30)
    REGION_PARIETAL: 0.15,   # Updated from evolve constants (was 0.18)  
    REGION_TEMPORAL: 0.17,   # Updated from evolve constants (was 0.20)
    REGION_OCCIPITAL: 0.13,  # Updated from evolve constants (was 0.12)
    REGION_LIMBIC: 0.10,     # Updated from evolve constants (was 0.08)
    REGION_BRAIN_STEM: 0.03, # Updated from evolve constants (was 0.02)
    REGION_CEREBELLUM: 0.14  # Updated from evolve constants (was 0.10)
}

# Region Locations (Normalized coordinates)
REGION_LOCATIONS = {
    REGION_FRONTAL: (0.3, 0.7, 0.5),    # Front upper part
    REGION_PARIETAL: (0.7, 0.7, 0.5),   # Rear upper part
    REGION_TEMPORAL: (0.5, 0.4, 0.3),   # Side middle part
    REGION_OCCIPITAL: (0.8, 0.5, 0.5),  # Rear part
    REGION_LIMBIC: (0.5, 0.5, 0.4),     # Central part
    REGION_BRAIN_STEM: (0.5, 0.3, 0.2), # Lower central part
    REGION_CEREBELLUM: (0.7, 0.3, 0.3)  # Lower rear part
}

# Region Default Frequencies (Resting state targets)
REGION_DEFAULT_FREQUENCIES: Dict[str, float] = {
    REGION_FRONTAL: 13.0,    # Updated from evolve (was 18.0) - primarily beta waves
    REGION_PARIETAL: 10.0,   # Alpha/beta mix
    REGION_TEMPORAL: 9.0,    # Updated from evolve (was 10.0) - primarily alpha waves
    REGION_OCCIPITAL: 11.0,  # Updated from evolve (was 12.0) - alpha/beta mix
    REGION_LIMBIC: 6.0,      # Theta waves
    REGION_BRAIN_STEM: 4.0,  # Updated from evolve (was 3.0) - delta/theta mix
    REGION_CEREBELLUM: 8.0   # Alpha waves
}

# Brain Wave Types
BRAIN_FREQUENCIES = {
    'delta': (0.5, 4),      # Deep sleep
    'theta': (4, 8),        # Drowsy, meditation 
    'beta': (13, 30),       # Alert, active
    'gamma': (30, 100),     # High cognition
    'lambda': (100, 400)    # Higher spiritual states
}

# (D) BRAIN_WAVE_TYPES: Dict[str, Tuple[float, float]] = {  # Duplicate of BRAIN_FREQUENCIES
#     'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
#     'beta': (13.0, 30.0), 'gamma': (30.0, 100.0), 'lambda': (100.0, 200.0)
# }

# Brain Complexity Thresholds
BRAIN_COMPLEXITY_THRESHOLDS: Dict[str, float] = {
    'energy_coverage': 0.3,     # Minimum energy coverage for complexity
    'mycelial_coverage': 0.2,   # Minimum mycelial coverage
    'avg_resonance': 0.3,       # Average resonance threshold
    'avg_energy': 0.2,          # Min average energy across brain
    'field_initialized': True   # All fields must be initialized
}

# =============================================================================
# CONSCIOUSNESS & BRAIN STATES
# =============================================================================

# Consciousness Activation Thresholds
CONSCIOUSNESS_ACTIVATION_THRESHOLDS: Dict[str, float] = {
    'dream': 0.4,      # Threshold for dream state activation
    'liminal': 0.6,    # Threshold for liminal state activation  
    'aware': 0.8       # Threshold for aware state activation
}

# Brain States for Field Modulation
BRAIN_STATE_DORMANT: str = "dormant"
BRAIN_STATE_FORMATION: str = "brain_formation"
BRAIN_STATE_AWARE_RESTING: str = "aware_resting"
BRAIN_STATE_AWARE_PROCESSING: str = "aware_input_processing"
BRAIN_STATE_ACTIVE_THOUGHT: str = "active_thought"
BRAIN_STATE_DREAMING: str = "dreaming"
BRAIN_STATE_LIMINAL_TRANSITION: str = "liminal_transition"
BRAIN_STATE_SOUL_ATTACHED_SETTLING: str = "soul_attached_settling"

# State Frequencies
LIMINAL_BASE_FREQUENCY = 3.5      # Delta/theta boundary
DREAM_BASE_FREQUENCY = 5.5        # Theta wave dominant
AWARENESS_BASE_FREQUENCY = 9.0    # Alpha wave dominant
STATE_TRANSITION_THRESHOLD = 0.6  # Percentage of cells needed for state transition

# Soul Attachment
SOUL_ATTACHMENT_COMPLEXITY_THRESHOLD: float = 0.85  # Overall complexity score needed

# =============================================================================
# MYCELIAL NETWORK CONSTANTS
# =============================================================================

# Basic Network Properties
MYCELIAL_DEFAULT_DENSITY: float = 0.1  # Default mycelial density for pathways
# (D) MYCELIAL_DEFAULT_DENSITY: float = 0.1  # Duplicate definition
MYCELIAL_MAXIMUM_PATHWAY_LENGTH: int = 200  # Maximum distance for mycelial connections
# (D) MYCELIAL_MAXIMUM_PATHWAY_LENGTH: int = int(GRID_DIMENSIONS[0] * 0.80)  # Duplicate (80% of brain extent)

# Quantum Properties
QUANTUM_ENTANGLEMENT_FREQUENCY: float = 432.0  # Frequency for quantum entanglement
QUANTUM_SEEDS_PER_SUBREGION: int = 2         # Number of quantum seeds per subregion
# (D) QUANTUM_SEEDS_PER_SUBREGION_TARGET: int = 1  # Duplicate (different target)
QUANTUM_EFFICIENCY: float = 0.98             # Quantum efficiency factor

# Energy and Routes
DEFAULT_SEED_COUNT_PER_REGION = 3        # Default number of seeds per region
MYCELIAL_QUANTUM_RANGE = 50              # Maximum distance for quantum entanglement
MYCELIAL_ENERGY_EFFICIENCY = 0.95        # Energy transfer efficiency
ENERGY_ROUTE_MAX_DISTANCE = 100          # Maximum distance for direct energy routes
DEFAULT_SEED_ENERGY_CAPACITY = 50.0      # Base energy capacity for mycelial seeds
DEFAULT_INITIAL_ENERGY_RATIO = 0.2       # Seeds start with 20% of capacity
ENERGY_TRANSFER_QUANTUM_BONUS = 0.98     # Efficiency for quantum connections
ENERGY_TRANSFER_LOSS_PER_DISTANCE = 0.005 # Energy loss per unit distance

# =============================================================================
# BOUNDARY SYSTEM
# =============================================================================

# Boundary Types
BOUNDARY_TYPE_SHARP = "sharp"       # Clear delineation
BOUNDARY_TYPE_GRADUAL = "gradual"   # Gradual transition
BOUNDARY_TYPE_PERMEABLE = "permeable"  # Very permeable

# Boundary Parameters
BOUNDARY_PARAMETERS = {
    BOUNDARY_TYPE_SHARP: {
        "transition_width": 1,
        "permeability": 0.3
    },
    BOUNDARY_TYPE_GRADUAL: {
        "transition_width": 3,
        "permeability": 0.6
    },
    BOUNDARY_TYPE_PERMEABLE: {
        "transition_width": 5,
        "permeability": 0.9
    }
}

# Region Boundary Mappings
REGION_BOUNDARIES = {
    (REGION_FRONTAL, REGION_PARIETAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_PARIETAL, REGION_OCCIPITAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_TEMPORAL, REGION_FRONTAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_TEMPORAL, REGION_PARIETAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_LIMBIC, REGION_FRONTAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_TEMPORAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_PARIETAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_OCCIPITAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_BRAIN_STEM): BOUNDARY_TYPE_PERMEABLE,
    (REGION_BRAIN_STEM, REGION_CEREBELLUM): BOUNDARY_TYPE_SHARP,
}

# Sound Pattern Constants
SOUND_SHARP_FREQUENCY = 15.0      # Base frequency for sharp boundaries
SOUND_GRADUAL_FREQUENCY = 10.0    # Base frequency for gradual boundaries 
SOUND_PERMEABLE_FREQUENCY = 7.0   # Base frequency for permeable boundaries
SOUND_FREQUENCY_VARIATION = 1.0   # Random variation in boundary sounds

# =============================================================================
# FIELD SYSTEM CONSTANTS
# =============================================================================

# Void Field Properties
VOID_BASE_FREQUENCY_RANGE: Tuple[float, float] = (10.0, 1000.0)
VOID_BASE_STABILITY_SU: float = 20.0  # Baseline SU towards which Void drifts
VOID_BASE_COHERENCE_CU: float = 15.0  # Baseline CU towards which Void drifts
VOID_CHAOS_ORDER_BALANCE: float = 0.5

# Sephiroth Properties
SEPHIROTH_DEFAULT_RADIUS: float = 8.0
SEPHIROTH_INFLUENCE_FALLOFF: float = 1.5
DEFAULT_PHI_HARMONIC_COUNT: int = 3

# Guff Properties  
GUFF_RADIUS_FACTOR: float = 0.3
GUFF_CAPACITY: int = 100
GUFF_TARGET_ENERGY_SEU: float = MAX_SOUL_ENERGY_SEU * 0.95 * 0.9  # 90% of Kether
GUFF_TARGET_STABILITY_SU: float = 98.0 * 0.95  # 95% of Kether
GUFF_TARGET_COHERENCE_CU: float = 98.0 * 0.95  # 95% of Kether

# Field Dynamics
HARMONIC_RESONANCE_ENERGY_BOOST: float = 0.012
WAVE_PROPAGATION_SPEED: float = 0.2
ENERGY_DISSIPATION_RATE = 0.002
FIELD_CHAOS_LEVEL_DEFAULT: float = 0.02  # Default 2% randomness/noise
FIELD_STATE_CACHE_ENABLED: bool = True   # Enable caching of state calculations
FIELD_STATIC_GRID_SPACING: int = 8       # For static electromagnetic foundation pattern
FIELD_MAX_PROPAGATION_RADIUS_FACTOR: float = 0.6  # Max propagation radius factor

# Safe Frequency Ranges for Fields
FIELD_STATIC_BASE_FREQUENCY_HZ_RANGE: Tuple[float, float] = (0.05, 1.0)
FIELD_DEVELOPMENT_FREQUENCY_HZ_RANGE: Tuple[float, float] = (0.5, 10.0)
FIELD_SOUL_RESONANCE_FREQUENCY_HZ_RANGE: Tuple[float, float] = (7.0, 15.0)
FIELD_ACTIVE_THOUGHT_FREQUENCY_HZ_RANGE: Tuple[float, float] = (12.0, 35.0)
FIELD_AVOID_FREQUENCY_RANGES_HZ: List[Tuple[float, float]] = [
    (40.0, 70.0),    # Avoid AC power line frequencies
    (700.0, 2700.0)  # Avoid RF bands (cellular, WiFi, Bluetooth)
]

# =============================================================================
# SEPHIROTH SYSTEM
# =============================================================================

# Sephiroth Absolute Potentials
SEPHIROTH_ENERGY_POTENTIALS_SEU: Dict[str, float] = {
    'kether': MAX_SOUL_ENERGY_SEU * 0.95,
    'chokmah': MAX_SOUL_ENERGY_SEU * 0.85,
    'binah': MAX_SOUL_ENERGY_SEU * 0.80,
    'daath': MAX_SOUL_ENERGY_SEU * 0.70,
    'chesed': MAX_SOUL_ENERGY_SEU * 0.75,
    'geburah': MAX_SOUL_ENERGY_SEU * 0.65,
    'tiphareth': MAX_SOUL_ENERGY_SEU * 0.70,
    'netzach': MAX_SOUL_ENERGY_SEU * 0.60,
    'hod': MAX_SOUL_ENERGY_SEU * 0.55,
    'yesod': MAX_SOUL_ENERGY_SEU * 0.45,
    'malkuth': MAX_SOUL_ENERGY_SEU * 0.30
}

SEPHIROTH_TARGET_STABILITY_SU: Dict[str, float] = {
    'kether': 98.0, 'chokmah': 90.0, 'binah': 92.0, 'daath': 85.0,
    'chesed': 88.0, 'geburah': 80.0, 'tiphareth': 95.0, 'netzach': 85.0,
    'hod': 82.0, 'yesod': 88.0, 'malkuth': 75.0
}

SEPHIROTH_TARGET_COHERENCE_CU: Dict[str, float] = {
    'kether': 98.0, 'chokmah': 92.0, 'binah': 90.0, 'daath': 88.0,
    'chesed': 90.0, 'geburah': 82.0, 'tiphareth': 95.0, 'netzach': 88.0,
    'hod': 85.0, 'yesod': 90.0, 'malkuth': 70.0
}

# Sephiroth Glyph Data
SEPHIROTH_GLYPH_DATA: Dict[str, Dict[str, Any]] = {
    'kether': {
        'platonic': 'dodecahedron', 'sigil': 'Point/Crown',
        'gematria_keys': ['Kether', 'Crown', 'Will', 'Unity', 1],
        'fibonacci': [1, 1]
    },
    'chokmah': {
        'platonic': 'sphere', 'sigil': 'Line/Wheel',
        'gematria_keys': ['Chokmah', 'Wisdom', 'Father', 2],
        'fibonacci': [2]
    },
    'binah': {
        'platonic': 'icosahedron', 'sigil': 'Triangle/Womb',
        'gematria_keys': ['Binah', 'Understanding', 'Mother', 3],
        'fibonacci': [3]
    },
    'chesed': {
        'platonic': 'hexahedron', 'sigil': 'Square/Solid',
        'gematria_keys': ['Chesed', 'Mercy', 'Grace', 4],
        'fibonacci': [5]
    },
    'geburah': {
        'platonic': 'tetrahedron', 'sigil': 'Pentagon/Sword',
        'gematria_keys': ['Geburah', 'Severity', 'Strength', 5],
        'fibonacci': [8]
    },
    'tiphareth': {
        'platonic': 'octahedron', 'sigil': 'Hexagram/Sun',
        'gematria_keys': ['Tiphareth', 'Beauty', 'Harmony', 6],
        'fibonacci': [13]
    },
    'netzach': {
        'platonic': 'icosahedron', 'sigil': 'Heptagon/Victory',
        'gematria_keys': ['Netzach', 'Victory', 'Endurance', 7],
        'fibonacci': [21]
    },
    'hod': {
        'platonic': 'octahedron', 'sigil': 'Octagon/Splendor',
        'gematria_keys': ['Hod', 'Splendor', 'Glory', 8],
        'fibonacci': [34]
    },
    'yesod': {
        'platonic': 'icosahedron', 'sigil': 'Nonagon/Foundation',
        'gematria_keys': ['Yesod', 'Foundation', 'Moon', 9],
        'fibonacci': [55]
    },
    'malkuth': {
        'platonic': 'hexahedron', 'sigil': 'CrossInCircle/Kingdom',
        'gematria_keys': ['Malkuth', 'Kingdom', 'Shekhinah', 'Earth', 10],
        'fibonacci': [89]
    },
    'daath': {
        'platonic': 'sphere', 'sigil': 'VoidPoint',
        'gematria_keys': ['Daath', 'Knowledge', 'Abyss', 11],
        'fibonacci': []
    }
}

# Sephiroth Frequencies and Adjustments
SEPHIROTH_FREQ_NUDGE_FACTOR = 0.04

# =============================================================================
# GEOMETRY & SACRED PATTERNS
# =============================================================================

# Platonic Solids
PLATONIC_SOLIDS: List[str] = ['tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron', 'sphere', 'merkaba']
AVAILABLE_PLATONIC_SOLIDS: List[str] = ["tetrahedron", "hexahedron", "octahedron", "dodecahedron", "icosahedron", "sphere", "merkaba"]

# Sacred Geometry Patterns
SACRED_GEOMETRY_STAGES: List[str] = ["seed_of_life", "flower_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "64_tetrahedron"]
AVAILABLE_GEOMETRY_PATTERNS: List[str] = ["flower_of_life", "seed_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "egg_of_life", "fruit_of_life", "germ_of_life", "sri_yantra", "star_tetrahedron", "64_tetrahedron"]

# Platonic Base Frequencies
PLATONIC_BASE_FREQUENCIES: Dict[str, float] = {
    'tetrahedron': 396.0, 'hexahedron': 285.0, 'octahedron': 639.0,
    'dodecahedron': 963.0, 'icosahedron': 369.0, 'sphere': 432.0, 'merkaba': 528.0
}

# Geometry Base Frequencies
GEOMETRY_BASE_FREQUENCIES: Dict[str, float] = {
    'point': 963.0, 'line': 852.0, 'triangle': 396.0, 'square': 285.0,
    'pentagon': 417.0, 'hexagram': 528.0, 'heptagon': 741.0, 'octagon': 741.0,
    'nonagon': 852.0, 'cross/cube': 174.0, 'vesica_piscis': 444.0,
    'flower_of_life': 528.0, 'seed_of_life': 432.0, 'tree_of_life': 528.0,
    'metatrons_cube': 639.0, 'merkaba': 741.0, 'vector_equilibrium': 639.0
}

# Platonic Harmonic Ratios
PLATONIC_HARMONIC_RATIOS: Dict[str, List[float]] = {
    'tetrahedron': [1.0, 2.0, 3.0, 5.0],
    'hexahedron': [1.0, 2.0, 4.0, 8.0],
    'octahedron': [1.0, 1.5, 2.0, 3.0],
    'dodecahedron': [1.0, PHI, 2.0, PHI*2, 3.0],
    'icosahedron': [1.0, 1.5, 2.0, 2.5, 3.0],
    'sphere': [1.0, 1.5, 2.0, 2.5, 3.0, PHI, 4.0, 5.0],
    'merkaba': [1.0, 1.5, 2.0, 3.0, PHI, 4.0]
}

# Geometry Effects
GEOMETRY_EFFECTS: Dict[str, Dict[str, float]] = {
    'tetrahedron': {'energy_focus': 0.1, 'transformative_capacity': 0.07},
    'hexahedron': {'stability_factor_boost': 0.15, 'grounding': 0.12, 'energy_containment': 0.08},
    'octahedron': {'yin_yang_balance_push': 0.0, 'coherence_factor_boost': 0.1, 'stability_factor_boost': 0.05},
    'dodecahedron': {'unity_connection': 0.15, 'phi_resonance_boost': 0.12, 'transcendence': 0.1},
    'icosahedron': {'emotional_flow': 0.12, 'adaptability': 0.1, 'coherence_factor_boost': 0.08},
    'sphere': {'potential_realization': 0.1, 'unity_connection': 0.05},
    'merkaba': {'stability_factor_boost': 0.1, 'transformative_capacity': 0.12, 'field_resilience': 0.08},
    'flower_of_life': {'harmony_boost': 0.12, 'structural_integration': 0.1},
    'seed_of_life': {'potential_realization': 0.1, 'stability_factor_boost': 0.08},
    'vesica_piscis': {'yin_yang_balance_push': 0.0, 'connection_boost': 0.09},
    'tree_of_life': {'harmony_boost': 0.1, 'structural_integration': 0.1, 'connection_boost': 0.08},
    'metatrons_cube': {'structural_integration': 0.12, 'connection_boost': 0.12},
    'vector_equilibrium': {'yin_yang_balance_push': 0.0, 'zero_point_attunement': 0.15},
    '64_tetrahedron': {'structural_integration': 0.15, 'energy_containment': 0.1}
}

DEFAULT_GEOMETRY_EFFECT: Dict[str, float] = {'stability_factor_boost': 0.01}

# Element to Platonic Mapping
PLATONIC_ELEMENT_MAP: Dict[str, str] = {
    'earth': 'hexahedron', 'water': 'icosahedron', 'fire': 'tetrahedron',
    'air': 'octahedron', 'aether': 'dodecahedron', 'spirit': 'dodecahedron',
    'void': 'sphere', 'light': 'merkaba'
}

# Influence Strengths
GEOMETRY_VOID_INFLUENCE_STRENGTH: float = 0.15
PLATONIC_VOID_INFLUENCE_STRENGTH: float = 0.20

# =============================================================================
# RESONANCE & ASPECTS
# =============================================================================

# Resonance Tolerances
RESONANCE_INTEGER_RATIO_TOLERANCE: float = 0.02
RESONANCE_PHI_RATIO_TOLERANCE: float = 0.03

# Sephiroth Journey Resonance
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ: float = 0.5
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM: float = 0.3
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI: float = 0.2

# Aspect Properties
SEPHIROTH_ASPECT_TRANSFER_FACTOR: float = 0.2
MAX_ASPECT_STRENGTH: float = 1.0
ASPECT_TRANSFER_THRESHOLD = 0.3  # Minimum resonance * connection for aspect transfer
ASPECT_TRANSFER_STRENGTH_FACTOR = 0.15  # Strength increase factor for existing aspects
ASPECT_TRANSFER_INITIAL_STRENGTH = 0.4  # Initial strength for newly transferred aspects

# Anchor Properties
ANCHOR_RESONANCE_MODIFIER: float = 0.85  # Modifier for anchor resonance calculations
ANCHOR_STRENGTH_MODIFIER: float = 0.6   # Matches existing constant
# (D) ANCHOR_STRENGTH_MODIFIER: float = 0.6  # Duplicate definition

# =============================================================================
# TRANSFER & INFLUENCE RATES
# =============================================================================

# Energy Transfer Rates
ENERGY_TRANSFER_RATE_K: float = 0.05           # Base rate for SEU transfer
GUFF_ENERGY_TRANSFER_RATE_K: float = 0.2       # SEU transfer rate in Guff
SEPHIROTH_ENERGY_EXCHANGE_RATE_K: float = 0.05 # SEU exchange rate during Sephirah interaction

# Influence Factor Rates
GUFF_INFLUENCE_RATE_K: float = 0.05      # How much each Guff step increments guff_influence_factor
SEPHIRAH_INFLUENCE_RATE_K: float = 0.15  # How much each Sephirah interaction increments cumulative_sephiroth_influence

# =============================================================================
# SPARK EMERGENCE & HARMONIZATION
# =============================================================================

# Spark Constants
SPARK_EOC_ENERGY_YIELD_FACTOR: float = 50.0    # How much EoC multiplies base potential
SPARK_FIELD_ENERGY_CATALYST_FACTOR: float = 0.05  # Fraction of local void energy added as catalyst
SPARK_SEED_GEOMETRY: str = 'sphere'             # Which platonic harmonic ratios seed the spark
SPARK_INITIAL_FACTOR_EOC_SCALE: float = 0.4     # How much EoC boosts initial phi_resonance, pattern_coherence
SPARK_INITIAL_FACTOR_ORDER_SCALE: float = 0.2   # How much local void order boosts initial factors
SPARK_INITIAL_FACTOR_PATTERN_SCALE: float = 0.1 # How much local void pattern boosts initial factors
SPARK_INITIAL_FACTOR_BASE: float = 0.1          # Minimum base value for initial factors

# Harmonization Constants
HARMONIZATION_ITERATIONS: int = 244             # Number of internal harmonization steps
HARMONIZATION_PATTERN_COHERENCE_RATE: float = 0.003  # Rate factor builds towards 1.0
HARMONIZATION_PHI_RESONANCE_RATE: float = 0.002      # Rate factor builds towards 1.0
HARMONIZATION_HARMONY_RATE: float = 0.0015           # Rate factor builds towards 1.0
HARMONIZATION_ENERGY_GAIN_RATE: float = 0.25         # SEU gain per iteration scaled by internal order
HARMONIZATION_PHASE_ADJUST_RATE = 0.01               # Small adjustment per iteration
HARMONIZATION_HARMONIC_ADJUST_RATE = 0.005           # Even smaller adjustment
HARMONIZATION_CIRC_VAR_THRESHOLD = 0.15              # Only adjust if variance > 15%
HARMONIZATION_HARM_DEV_THRESHOLD = 0.08              # Only adjust if average deviation > 8%
HARMONIZATION_TORUS_RATE: float = 0.002              # Rate factor builds towards 1.0

# =============================================================================
# HARMONIC STRENGTHENING SYSTEM
# =============================================================================

# Trigger Thresholds
HS_TRIGGER_STABILITY_SU = 99.0          # Target stability threshold
HS_TRIGGER_COHERENCE_CU = 99.0          # Target coherence threshold
HS_TRIGGER_PHASE_COHERENCE = 0.98       # Target phase coherence 
HS_TRIGGER_HARMONIC_PURITY = 0.98       # Target harmonic purity
HS_TRIGGER_FACTOR_THRESHOLD = 0.98      # Target for phi, pattern, harmony, torus

# Cycle Control
HS_MAX_CYCLES: int = 256                # Maximum refinement iterations
HS_UPDATE_STATE_INTERVAL: int = 10      # Cycles between S/C recalculations

# Base Improvement Rates
HS_BASE_RATE_PHI = 0.01                 # Base rate for phi improvement
HS_BASE_RATE_PCOH = 0.01                # Base rate for pattern coherence
HS_BASE_RATE_HARMONY = 0.01             # Base rate for harmony
HS_BASE_RATE_TORUS = 0.01               # Base rate for torus

# Frequency Structure Refinement
HS_PHASE_ADJUST_RATE = 0.01             # Rate for phase adjustment
HS_HARMONIC_ADJUST_RATE = 0.01          # Rate for harmonic adjustment

# Energy Adjustment
HS_ENERGY_ADJUST_FACTOR: float = 5.0    # Scales SEU gain/loss based on delta_harmony

# =============================================================================
# CREATOR ENTANGLEMENT SYSTEM
# =============================================================================

# Entanglement Thresholds
CE_RESONANCE_THRESHOLD: float = 0.75    # Min resonance score considered 'strong'
CE_CONNECTION_FREQ_WEIGHT: float = 0.6    # Weight of frequency resonance in connection strength
CE_CONNECTION_COHERENCE_WEIGHT: float = 0.4  # Weight of soul's coherence in connection strength
CE_PATTERN_MODULATION_STRENGTH: float = 0.05  # How strongly Kether's geometry influences soul

# Default Values
CREATOR_POTENTIAL_DEFAULT = 0.7  # Default creator potential for entanglement
ENTANGLEMENT_PREREQ_STABILITY_MIN_SU = 30.0   # Minimum stability required
ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU = 25.0   # Minimum coherence required

# =============================================================================
# SOLFEGGIO & SOUND FREQUENCIES
# =============================================================================

# Solfeggio Frequencies
SOLFEGGIO_FREQUENCIES: Dict[str, float] = {
    'UT': 396.0, 'RE': 417.0, 'MI': 528.0, 'FA': 639.0,
    'SOL': 741.0, 'LA': 852.0, 'SI': 963.0
}

# Harmonic Strengthening Target Frequencies
HARMONIC_STRENGTHENING_TARGET_FREQS: List[float] = sorted(
    list(SOLFEGGIO_FREQUENCIES.values()) + [FUNDAMENTAL_FREQUENCY_432]
)

# Audio Constants
SAMPLE_RATE: int = 44100
MAX_AMPLITUDE: float = 0.95

# =============================================================================
# STAGE FLAGS & PREREQUISITES
# =============================================================================

# Stage Readiness Flags
FLAG_READY_FOR_GUFF = "ready_for_guff"
FLAG_GUFF_STRENGTHENED = "guff_strengthened"
FLAG_READY_FOR_JOURNEY = "ready_for_journey"
FLAG_SEPHIROTH_JOURNEY_COMPLETE = "sephiroth_journey_complete"
FLAG_READY_FOR_ENTANGLEMENT = "ready_for_entanglement"
FLAG_READY_FOR_COMPLETION = "ready_for_completion"
FLAG_CREATOR_ENTANGLED = "creator_entangled"
FLAG_SPARK_HARMONIZED = "spark_harmonized"
FLAG_READY_FOR_HARMONIZATION = "ready_for_harmonization"
FLAG_READY_FOR_STRENGTHENING = "ready_for_strengthening"
FLAG_HARMONICALLY_STRENGTHENED = "harmonically_strengthened"
FLAG_READY_FOR_LIFE_CORD = "ready_for_life_cord"
FLAG_CORD_FORMATION_COMPLETE = "cord_formation_complete"
FLAG_READY_FOR_EARTH = "ready_for_earth"
FLAG_EARTH_HARMONIZED = "earth_harmonized"
FLAG_EARTH_ATTUNED = "earth_attuned"
FLAG_READY_FOR_IDENTITY = "ready_for_identity"
FLAG_IDENTITY_CRYSTALLIZED = "identity_crystallized"
FLAG_READY_FOR_BIRTH = "ready_for_birth"
FLAG_BIRTH_COMPLETED = "birth_completed"
FLAG_READY_FOR_EVOLUTION = "ready_for_evolution"
FLAG_INCARNATED = "incarnated"
FLAG_ECHO_PROJECTED = "echo_projected"
FLAG_SOUL_ATTACHED_TO_BRAIN = "soul_attached_to_brain"

# Stage Prerequisites (Using SU/CU)
ENTANGLEMENT_PREREQ_STABILITY_MIN_SU: float = 75.0
ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU: float = 75.0
# (D) ENTANGLEMENT_PREREQ_STABILITY_MIN_SU = 30.0  # Duplicate with different value
# (D) ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU = 25.0  # Duplicate with different value

HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU: float = 70.0
HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU: float = 70.0
CORD_STABILITY_THRESHOLD_SU: float = 80.0
CORD_COHERENCE_THRESHOLD_CU: float = 80.0
HARMONY_PREREQ_CORD_INTEGRITY_MIN: float = 0.70
HARMONY_PREREQ_STABILITY_MIN_SU: float = 75.0
HARMONY_PREREQ_COHERENCE_MIN_CU: float = 75.0
IDENTITY_STABILITY_THRESHOLD_SU: float = 85.0
IDENTITY_COHERENCE_THRESHOLD_CU: float = 85.0
IDENTITY_EARTH_RESONANCE_THRESHOLD: float = 0.75
IDENTITY_CRYSTALLIZATION_THRESHOLD: float = 0.85
BIRTH_PREREQ_CORD_INTEGRITY_MIN: float = 0.80
BIRTH_PREREQ_EARTH_RESONANCE_MIN: float = 0.75

# Birth Prerequisites
BIRTH_MIN_STABILITY_SU = 75.0
BIRTH_MIN_COHERENCE_CU = 75.0
BIRTH_MIN_CRYSTALLIZATION = 0.75

# =============================================================================
# EARTH HARMONIZATION CONSTANTS
# =============================================================================

# Earth Anchor Properties
EARTH_ANCHOR_RESONANCE: float = 0.9
EARTH_ANCHOR_STRENGTH: float = 0.9
EARTH_CYCLE_FREQ_SCALING: float = 1.0e9

# Echo Field Properties
ECHO_FIELD_STRENGTH_FACTOR: float = 0.85
HARMONY_SCHUMANN_INTENSITY: float = 0.6
HARMONY_CORE_INTENSITY: float = 0.4

# Earth Frequencies
EARTH_FREQUENCIES: Dict[str, float] = {
    "schumann": 7.83,
    "geomagnetic": 11.75,
    "core_resonance": EARTH_FREQUENCY,
    "breath_cycle": EARTH_BREATH_FREQUENCY,
    "heartbeat_cycle": 1.2,
    "circadian_cycle": 1.0/(24*3600)
}

# Earth Elements
EARTH_ELEMENTS: List[str] = ["earth", "water", "fire", "air", "aether"]

# Element Colors for Visualization
ELEMENT_COLORS = {
    'earth': '#8B4513',
    'air': '#87CEEB', 
    'fire': '#FF4500',
    'water': '#1E90FF',
    'aether': '#E6E6FA'
}

# Surface Resonance
SURFACE_RESONANCE_FACTOR = 0.5
SURFACE_RESONANCE_HARMONIC_THRESHOLD = 0.3

# Echo Attunement Constants
ECHO_PROJECTION_ENERGY_COST_FACTOR: float = 0.02  # % of spiritual energy cost
ECHO_ATTUNEMENT_CYCLES: int = 84        # Number of attunement cycles
ECHO_ATTUNEMENT_RATE: float = 0.8       # Base rate of state shift per cycle
ECHO_FIELD_COHERENCE_FACTOR: float = 0.75  # Coherence factor for echo field
ATTUNEMENT_RESONANCE_THRESHOLD: float = 0.3   # Min resonance for attunement shift

# Elemental Alignment
ELEMENTAL_TARGET_EARTH: float = 0.8  # Target alignment for primary earth element
ELEMENTAL_TARGET_OTHER: float = 0.4  # Target alignment for other elements
ELEMENTAL_ALIGN_INTENSITY_FACTOR: float = 0.2  # Rate for elemental alignment nudge

# Planetary Frequencies
PLANETARY_FREQUENCIES: Dict[str, float] = {
    "Sun": 126.22, "Moon": 210.42, "Mercury": 141.27, "Venus": 221.23,
    "Earth": 194.71, "Mars": 144.72, "Jupiter": 183.58, "Saturn": 147.85,
    "Uranus": 207.36, "Neptune": 211.44, "Pluto": 140.25
}

# Planetary Attunement
PLANETARY_ATTUNEMENT_CYCLES: int = 21  # Cycles for planetary resonance step
PLANETARY_RESONANCE_RATE: float = 0.0095  # Rate for planetary resonance factor gain

# Gaia Connection
GAIA_CONNECTION_CYCLES: int = 13  # Cycles for Gaia connection step
GAIA_CONNECTION_FACTOR: float = 0.023  # Rate for Gaia connection factor gain

# Stress Feedback
STRESS_FEEDBACK_FACTOR: float = 0.0024  # How much echo discordance impacts main soul

# Harmony Cycles
HARMONY_CYCLE_NAMES: List[str] = ["circadian", "heartbeat", "breath"]
HARMONY_CYCLE_IMPORTANCE: Dict[str, float] = {"circadian": 0.6, "heartbeat": 0.8, "breath": 1.0}

# =============================================================================
# IDENTITY CRYSTALLIZATION (ASTROLOGY)
# =============================================================================

# Zodiac Signs (13 signs format)
ZODIAC_SIGNS: List[Dict[str, str]] = [
    {"name": "Aries", "symbol": "♈", "start_date": "April 19", "end_date": "May 13"},
    {"name": "Taurus", "symbol": "♉", "start_date": "May 14", "end_date": "June 19"},
    {"name": "Gemini", "symbol": "♊", "start_date": "June 20", "end_date": "July 20"},
    {"name": "Cancer", "symbol": "♋", "start_date": "July 21", "end_date": "August 9"},
    {"name": "Leo", "symbol": "♌", "start_date": "August 10", "end_date": "September 15"},
    {"name": "Virgo", "symbol": "♍", "start_date": "September 16", "end_date": "October 30"},
    {"name": "Libra", "symbol": "♎", "start_date": "October 31", "end_date": "November 22"},
    {"name": "Scorpio", "symbol": "♏", "start_date": "November 23", "end_date": "November 29"},
    {"name": "Ophiuchus", "symbol": "⛎", "start_date": "November 30", "end_date": "December 17"},
    {"name": "Sagittarius", "symbol": "♐", "start_date": "December 18", "end_date": "January 18"},
    {"name": "Capricorn", "symbol": "♑", "start_date": "January 19", "end_date": "February 15"},
    {"name": "Aquarius", "symbol": "♒", "start_date": "February 16", "end_date": "March 11"},
    {"name": "Pisces", "symbol": "♓", "start_date": "March 12", "end_date": "April 18"}
]

# Zodiac Traits (abbreviated for space - full traits available in original)
ZODIAC_TRAITS: Dict[str, Dict[str, List[str]]] = {
    "Aries": {
        "positive": ["Courageous","Energetic","Adventurous","Enthusiastic","Confident"],
        "negative": ["Impulsive","Impatient","Aggressive","Reckless","Self-centered"]
    },
    "Taurus": {
        "positive": ["Reliable","Patient","Practical","Devoted","Persistent"],
        "negative": ["Possessive","Stubborn","Materialistic","Self-indulgent","Inflexible"]
    },
    # ... (other signs follow same pattern)
}

# Astrology Constants
ASTROLOGY_MAX_POSITIVE_TRAITS: int = 5
ASTROLOGY_MAX_NEGATIVE_TRAITS: int = 2
NAME_GEMATRIA_RESONANT_NUMBERS: List[int] = [3, 7, 9, 11, 13, 22]

# Color Spectrum
COLOR_SPECTRUM: Dict[str, Dict] = {
    "red": {"frequency": (400, 480), "hex": "#FF0000"},
    "orange": {"frequency": (480, 510), "hex": "#FFA500"},
    "gold": {"frequency": (510, 530), "hex": "#FFD700"},
    "yellow": {"frequency": (530, 560), "hex": "#FFFF00"},
    "green": {"frequency": (560, 610), "hex": "#00FF00"},
    "blue": {"frequency": (610, 670), "hex": "#0000FF"},
    "indigo": {"frequency": (670, 700), "hex": "#4B0082"},
    "violet": {"frequency": (700, 790), "hex": "#8A2BE2"},
    "white": {"frequency": (400, 790), "hex": "#FFFFFF"},
    "black": {"frequency": (0, 0), "hex": "#000000"},
    "silver": {"frequency": (0, 0), "hex": "#C0C0C0"},
    "magenta": {"frequency": (0, 0), "hex": "#FF00FF"},
    "grey": {"frequency": (0,0), "hex": "#808080"},
    "earth_tones": {"frequency": (150, 250), "hex": "#A0522D"},
    "lavender": {"frequency": (700, 750), "hex": "#E6E6FA"},
    "brown": {"frequency": (150, 250), "hex": "#A52A2A"}
}

COLOR_FREQ_DEFAULT: float = 500.0

# =============================================================================
# LIFE CORD SYSTEM
# =============================================================================

# Life Cord Properties
LIFE_CORD_COMPLEXITY_DEFAULT: float = 0.7
LIFE_CORD_FREQUENCIES = {'primary': 528.0}

# Primary Channel Properties
PRIMARY_CHANNEL_BANDWIDTH_FACTOR: float = 200.0
PRIMARY_CHANNEL_STABILITY_FACTOR_CONN: float = 0.7
PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX: float = 0.3
PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN: float = 0.6
PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX: float = 0.4
PRIMARY_CHANNEL_ELASTICITY_BASE: float = 0.5
PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX: float = 0.3

# Harmonic Nodes
HARMONIC_NODE_COUNT_BASE: int = 3
HARMONIC_NODE_COUNT_FACTOR: float = 15.0
HARMONIC_NODE_AMP_BASE: float = 0.4
HARMONIC_NODE_AMP_FACTOR_COMPLEX: float = 0.4
HARMONIC_NODE_AMP_FALLOFF: float = 0.6
HARMONIC_NODE_BW_INCREASE_FACTOR: float = 5.0

# Secondary Channels
MAX_CORD_CHANNELS: int = 7
SECONDARY_CHANNEL_COUNT_FACTOR: float = 6.0
SECONDARY_CHANNEL_FREQ_FACTOR: float = 0.1

# Channel Types
SECONDARY_CHANNEL_BW_EMOTIONAL: tuple[float,float] = (10.0,30.0)
SECONDARY_CHANNEL_RESIST_EMOTIONAL: tuple[float,float] = (0.4,0.3)
SECONDARY_CHANNEL_BW_MENTAL: tuple[float,float] = (15.0,40.0)
SECONDARY_CHANNEL_RESIST_MENTAL: tuple[float,float] = (0.5,0.3)
SECONDARY_CHANNEL_BW_SPIRITUAL: tuple[float,float] = (20.0,50.0)
SECONDARY_CHANNEL_RESIST_SPIRITUAL: tuple[float,float] = (0.6,0.3)

# Field Integration
FIELD_INTEGRATION_FACTOR_FIELD_STR: float = 0.6
FIELD_INTEGRATION_FACTOR_CONN_STR: float = 0.4
FIELD_EXPANSION_FACTOR: float = 1.05

# Cord Integrity
CORD_INTEGRITY_FACTOR_CONN_STR: float = 0.4
CORD_INTEGRITY_FACTOR_STABILITY: float = 0.3
CORD_INTEGRITY_FACTOR_EARTH_CONN: float = 0.3

# Final Stability
FINAL_STABILITY_BONUS_FACTOR: float = 0.15

# =============================================================================
# BIRTH PROCESS CONSTANTS
# =============================================================================

# Birth Process Configuration
BIRTH_WOMB_DEVELOPMENT_CYCLES = 3  # Number of abstract development cycles
# (D) BIRTH_WOMB_DEVELOPMENT_CYCLES = 3  # Duplicate definition

# Birth Energy Allocation
BIRTH_ALLOC_SEED_CORE: float = 0.4   # 40% of energy to brain seed core
# (D) BIRTH_ALLOC_SEED_CORE: float = 0.10  # Duplicate with different value
BIRTH_ALLOC_REGIONS: float = 0.3     # 30% to brain regions  
# (D) BIRTH_ALLOC_REGIONS: float = 0.30  # Duplicate definition
BIRTH_ALLOC_MYCELIAL: float = 0.3    # 30% to mycelial networks
# (D) BIRTH_ALLOC_MYCELIAL: float = 0.60  # Duplicate with different value

# Birth Connection Properties
BIRTH_INTENSITY_DEFAULT: float = 0.7
BIRTH_CONN_WEIGHT_RESONANCE: float = 0.6
BIRTH_CONN_WEIGHT_INTEGRITY: float = 0.4
BIRTH_CONN_STRENGTH_FACTOR: float = 0.5
BIRTH_CONN_STRENGTH_CAP: float = 0.95
BIRTH_CONN_MOTHER_STRENGTH_FACTOR: float = 0.1

# Birth Trauma & Acceptance
BIRTH_CONN_TRAUMA_FACTOR: float = 0.3
BIRTH_CONN_ACCEPTANCE_TRAUMA_FACTOR: float = 0.4
BIRTH_CONN_MOTHER_TRAUMA_REDUCTION: float = 0.2
BIRTH_ACCEPTANCE_MIN: float = 0.2
BIRTH_ACCEPTANCE_TRAUMA_FACTOR: float = 0.8
BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR: float = 0.1

# Birth Cord Transfer
BIRTH_CORD_TRANSFER_INTENSITY_FACTOR: float = 0.2
BIRTH_CORD_MOTHER_EFFICIENCY_FACTOR: float = 0.1
BIRTH_CORD_INTEGRATION_CONN_FACTOR: float = 0.9
BIRTH_CORD_MOTHER_INTEGRATION_FACTOR: float = 0.08

# Birth Veil Properties
BIRTH_VEIL_STRENGTH_BASE: float = 0.6
BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR: float = 0.3
BIRTH_VEIL_PERMANENCE_BASE: float = 0.7
BIRTH_VEIL_PERMANENCE_INTENSITY_FACTOR: float = 0.25
BIRTH_VEIL_RETENTION_BASE: float = 0.1
BIRTH_VEIL_RETENTION_INTENSITY_FACTOR: float = -0.05
BIRTH_VEIL_RETENTION_MIN: float = 0.02
BIRTH_VEIL_MOTHER_RETENTION_FACTOR: float = 0.02

# Memory Retention Modifiers
BIRTH_VEIL_MEMORY_RETENTION_MODS: Dict[str, float] = {
    'core_identity': 0.1,
    'creator_connection': 0.05,
    'journey_lessons': 0.02,
    'specific_details': -0.05
}

# Birth Breath Properties
BIRTH_BREATH_AMP_BASE: float = 0.5
BIRTH_BREATH_AMP_INTENSITY_FACTOR: float = 0.3
BIRTH_BREATH_DEPTH_BASE: float = 0.6
BIRTH_BREATH_DEPTH_INTENSITY_FACTOR: float = 0.2
BIRTH_BREATH_SYNC_RESONANCE_FACTOR: float = 0.8
BIRTH_BREATH_MOTHER_SYNC_FACTOR: float = 0.3
BIRTH_BREATH_INTEGRATION_CONN_FACTOR: float = 0.7
BIRTH_BREATH_RESONANCE_BOOST_FACTOR: float = 0.1
BIRTH_BREATH_MOTHER_RESONANCE_BOOST: float = 0.15
BIRTH_BREATH_ENERGY_SHIFT_FACTOR: float = 0.15
BIRTH_BREATH_MOTHER_ENERGY_BOOST: float = 0.1

# Birth Physical/Spiritual Energy
BIRTH_BREATH_PHYSICAL_ENERGY_BASE: float = 0.5
BIRTH_BREATH_PHYSICAL_ENERGY_SCALE: float = 0.8
BIRTH_BREATH_SPIRITUAL_ENERGY_BASE: float = 0.7
BIRTH_BREATH_SPIRITUAL_ENERGY_SCALE: float = -0.5
BIRTH_BREATH_SPIRITUAL_ENERGY_MIN: float = 0.1

# Birth Final Integration
BIRTH_FINAL_INTEGRATION_WEIGHT_CONN: float = 0.4
BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT: float = 0.3
BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH: float = 0.3
BIRTH_FINAL_MOTHER_INTEGRATION_BOOST: float = 0.05
BIRTH_FINAL_FREQ_FACTOR: float = 0.8
BIRTH_FINAL_STABILITY_FACTOR: float = 0.9
BIRTH_ENERGY_BUFFER_FACTOR: float = 1.4

# Birth Frequency & Stability Adjustments
BIRTH_FINAL_FREQ_SHIFT_FACTOR: float = 0.05  # Max % frequency drop due to density
BIRTH_FINAL_STABILITY_PENALTY_FACTOR: float = 0.03  # Max % stability drop due to shock

# Birth Attachment
BIRTH_ATTACHMENT_MIN_CORD_INTEGRITY: float = 0.75  # Min cord integrity for attachment

# Birth Standing Wave Properties
BIRTH_STANDING_WAVE_LENGTH = 10.0  # meters
BIRTH_STANDING_WAVE_AMPLITUDE = 0.8
BIRTH_STANDING_WAVE_SOUND_DURATION = 8.0
BIRTH_STANDING_WAVE_SOUND_AMPLITUDE = 0.5

# Birth Echo Field Properties
BIRTH_ECHO_FIELD_STRENGTH_FACTOR = 0.7
BIRTH_ECHO_FIELD_COHERENCE_FACTOR = 0.8
BIRTH_ECHO_FIELD_CHAMBERS = 5
BIRTH_ECHO_FIELD_SOUND_DURATION = 12.0
BIRTH_ECHO_FIELD_SOUND_AMPLITUDE = 0.6

# =============================================================================
# WOMB ENVIRONMENT CONSTANTS
# =============================================================================

# Womb Enhancement Factors
WOMB_ENERGY_ENHANCEMENT_FACTOR = 0.2
WOMB_FREQUENCY_STABILIZATION_FACTOR = 0.15
WOMB_GROWTH_ENHANCEMENT_FACTOR = 0.25
WOMB_BRAIN_ENERGY_INTEGRATION_FACTOR = 0.1
WOMB_BRAIN_RESONANCE_INTEGRATION_FACTOR = 0.05
WOMB_BRAIN_PROTECTION_INTEGRATION_FACTOR = 0.08
WOMB_ENERGY_CONNECTION_ENHANCEMENT = 0.3
WOMB_RESONANCE_ENHANCEMENT = 0.2
WOMB_MYCELIAL_ENHANCEMENT = 0.25

# Womb Stability & Coherence Boosts
WOMB_SOUL_STABILITY_BOOST = 5.0  # SU
WOMB_BRAIN_STABILITY_BOOST = 0.1
WOMB_LIFE_CORD_ENHANCEMENT = 0.15

# Womb Memory Protection
WOMB_MEMORY_VEIL_PROTECTION_FACTOR = 0.2
WOMB_MEMORY_VEIL_NURTURING_FACTOR = 0.15
WOMB_STRESS_THRESHOLD = 0.7

# Womb Final Blessings
WOMB_FINAL_ENERGY_BLESSING_FACTOR = 0.1
WOMB_FINAL_STABILITY_BLESSING_FACTOR = 3.0  # SU
WOMB_FINAL_COHERENCE_BLESSING_FACTOR = 3.0  # CU
WOMB_FINAL_PROTECTION_BLESSING_FACTOR = 0.1

# =============================================================================
# MEMORY & VEIL SYSTEM
# =============================================================================

# Memory Constants
MEMORY_ACTIVATION_ENERGY = 0.1    # Energy required to activate a memory
MEMORY_FREQUENCY_TOLERANCE = 1.0  # Default frequency tolerance for searches
MEMORY_RESONANCE_THRESHOLD = 0.5  # Minimum resonance for memory activation
MEMORY_FRAGMENT_ENERGY_THRESHOLD = 0.05  # Minimum energy to activate memory fragment
# (D) MEMORY_FRAGMENT_ENERGY_THRESHOLD = 0.05  # Duplicate definition

# Memory Veil Constants
MEMORY_VEIL_BASE_STRENGTH = 0.8
MEMORY_VEIL_MAX_STRENGTH = 0.95
MEMORY_VEIL_FREQUENCY_LAYERS = 7
MEMORY_VEIL_SOUND_DURATION = 10.0
MEMORY_VEIL_SOUND_AMPLITUDE = 0.3
VEIL_BASE_RETENTION: float = 0.05  # Base retention factor before coherence bonus
VEIL_COHERENCE_RESISTANCE_FACTOR: float = 0.15  # How much coherence resists veil

# =============================================================================
# BRAIN SEED & CREATOR ENERGY
# =============================================================================

# Brain Seed Properties
DEFAULT_BRAIN_SEED_FREQUENCY: float = 7.83  # Default frequency (Schumann resonance)
SEED_FIELD_RADIUS: float = 5.0              # Energy field radius around seed
BRAIN_FREQUENCY_SCALE: float = 0.85         # Scale factor for brain vs soul frequency

# Creator Energy Management
CREATOR_ENERGY_BASE_FREQUENCY_HZ: float = 432.0  # Default frequency of Creator Energy
BRAIN_SEED_NURTURING_REQUEST_THRESHOLD_FACTOR: float = 0.2  # Request threshold (20% of capacity)
BRAIN_SEED_NURTURING_ENERGY_INCREMENT_FACTOR: float = 0.1   # Request increment (10% of capacity)
INITIAL_SPARK_ENERGY_FROM_CREATOR_FACTOR: float = 0.4       # Proportion from Creator

# =============================================================================
# CELL ID PREFIXES & ORGANIZATION
# =============================================================================

# Cell ID Prefixes
PREFIX_BLOCK = "B"         # Block ID prefix
PREFIX_CELL = "C"          # Regular cell ID prefix
PREFIX_BOUNDARY = "BC"     # Boundary cell ID prefix
PREFIX_MYCELIAL = "MS"     # Mycelial seed ID prefix
PREFIX_ROUTE = "R"         # Route ID prefix

# =============================================================================
# NATURAL SCALING & ENERGY CYCLES
# =============================================================================

# Energy Conversion Factors (Natural Proportional Relationships)
NATURAL_ENERGY_TO_HARMONY = 0.05   # Energy units needed for 1% harmony increase
NATURAL_HARMONY_TO_ENERGY = 20.0   # Energy generated by 1% harmony improvement
NATURAL_ENERGY_CYCLE_RATIO = 0.02  # % of energy used per cycle (sustainable rate)

# Natural Scaling Factors
NATURAL_EFFORT_SCALING = 4.0       # Exponential effort scaling for approaching perfection
NATURAL_STAGNATION_THRESHOLD = 10  # Cycles of no change to detect equilibrium
NATURAL_VIABILITY_THRESHOLD = 0.5  # Minimum capacity needed for further development

# =============================================================================
# PATHS & LOGGING
# =============================================================================

# Directory Paths
DATA_DIR_BASE: str = "output"
OUTPUT_DIR_BASE: str = "output"

# Logging Configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Metrics
PERSIST_INTERVAL_SECONDS: int = 60

# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

# Soul Spark Visualization
SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT: str = 'grey'
SOUL_SPARK_VIZ_FREQ_SIG_MARKER_FMT: str = 'bo'
SOUL_SPARK_VIZ_FREQ_SIG_BASE_FMT: str = 'r-'
SOUL_SPARK_VIZ_FREQ_SIG_STEM_LW: float = 1.5
SOUL_SPARK_VIZ_FREQ_SIG_MARKER_SZ: float = 5.0
SOUL_SPARK_VIZ_FREQ_SIG_XLABEL: str = 'Frequency (Hz)'
SOUL_SPARK_VIZ_FREQ_SIG_YLABEL: str = 'Amplitude'
SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR: str = 'red'

# =============================================================================
# STAGE-SPECIFIC PARAMETERS
# =============================================================================

# Guff Strengthening
GUFF_STRENGTHENING_DURATION: float = 10.0

# Sephiroth Journey
SEPHIROTH_JOURNEY_ATTRIBUTE_IMPART_FACTOR: float = 0.05
SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR: float = 0.04

# Entanglement
ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR: float = 0.15
ENTANGLEMENT_STABILITY_BOOST_FACTOR: float = 0.1
ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE: float = 0.6
ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE: float = 0.4
ENTANGLEMENT_RESONANCE_BOOST_FACTOR: float = 0.05
ENTANGLEMENT_STABILIZATION_ITERATIONS: int = 5
ENTANGLEMENT_STABILIZATION_FACTOR_STRENGTH: float = 1.005

# Harmonic Strengthening
HARMONIC_STRENGTHENING_INTENSITY_DEFAULT: float = 0.7
HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT: float = 1.0
HARMONIC_STRENGTHENING_TUNING_INTENSITY_FACTOR: float = 0.1
HARMONIC_STRENGTHENING_TUNING_TARGET_REACH_HZ: float = 1.0
HARMONIC_STRENGTHENING_PHI_AMP_INTENSITY_FACTOR: float = 0.05
HARMONIC_STRENGTHENING_PHI_STABILITY_BOOST_FACTOR: float = 0.3
HARMONIC_STRENGTHENING_PATTERN_STAB_INTENSITY_FACTOR: float = 0.04
HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_FACTOR: float = 0.01
HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_CAP: float = 0.1
HARMONIC_STRENGTHENING_PATTERN_STAB_STABILITY_BOOST: float = 0.4
HARMONIC_STRENGTHENING_COHERENCE_INTENSITY_FACTOR: float = 0.08
HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_COUNT_NORM: float = 10.0
HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_FACTOR: float = 0.03
HARMONIC_STRENGTHENING_COHERENCE_HARMONY_BOOST: float = 0.25
HARMONIC_STRENGTHENING_EXPANSION_INTENSITY_FACTOR: float = 0.1
HARMONIC_STRENGTHENING_EXPANSION_STATE_FACTOR: float = 1.0
HARMONIC_STRENGTHENING_EXPANSION_STR_INTENSITY_FACTOR: float = 0.03
HARMONIC_STRENGTHENING_EXPANSION_STR_STATE_FACTOR: float = 0.5
HARMONIC_STRENGTHENING_HARMONIC_COUNT: int = 7

# Earth Connection
EARTH_CONN_FACTOR_CONN_STR: float = 0.5
EARTH_CONN_FACTOR_ELASTICITY: float = 0.3
EARTH_CONN_BASE_FACTOR: float = 0.1
EARTH_HARMONY_INTENSITY_DEFAULT: float = 0.7
EARTH_HARMONY_DURATION_FACTOR_DEFAULT: float = 1.0

# Harmony Cycle Sync
HARMONY_CYCLE_SYNC_TARGET_BASE: float = 0.9
HARMONY_CYCLE_SYNC_INTENSITY_FACTOR: float = 0.1
HARMONY_CYCLE_SYNC_DURATION_FACTOR: float = 1.0
HARMONY_PLANETARY_RESONANCE_TARGET: float = 0.85
HARMONY_PLANETARY_RESONANCE_FACTOR: float = 0.15
HARMONY_GAIA_CONNECTION_TARGET: float = 0.90
HARMONY_GAIA_CONNECTION_FACTOR: float = 0.2
HARMONY_FINAL_STABILITY_BONUS: float = 0.08
HARMONY_FINAL_COHERENCE_BONUS: float = 0.08

# Harmony Frequency Tuning
HARMONY_FREQ_TUNING_FACTOR: float = 0.15
HARMONY_FREQ_TUNING_TARGET_REACH_HZ: float = 1.0
HARMONY_FREQ_UPDATE_HARMONIC_COUNT: int = 5
HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT: float = 0.6
HARMONY_FREQ_TARGET_SOUL_WEIGHT: float = 0.2
HARMONY_FREQ_TARGET_CORE_WEIGHT: float = 0.2
HARMONY_FREQ_RES_WEIGHT_SCHUMANN: float = 0.7
HARMONY_FREQ_RES_WEIGHT_OTHER: float = 0.3
HARMONY_ELEM_RES_WEIGHT_PRIMARY: float = 0.6
HARMONY_ELEM_RES_WEIGHT_AVERAGE: float = 0.4

# =============================================================================
# IDENTITY CRYSTALLIZATION EXTENDED
# =============================================================================

# Name Resonance
NAME_RESONANCE_BASE: float = 0.1
NAME_RESONANCE_WEIGHT_VOWEL: float = 0.3
NAME_RESONANCE_WEIGHT_LETTER: float = 0.2
NAME_RESONANCE_WEIGHT_GEMATRIA: float = 0.4

# Voice Frequency
VOICE_FREQ_BASE: float = 220.0
VOICE_FREQ_ADJ_LENGTH_FACTOR: float = -50.0
VOICE_FREQ_ADJ_VOWEL_FACTOR: float = 80.0
VOICE_FREQ_ADJ_GEMATRIA_FACTOR: float = 40.0
VOICE_FREQ_ADJ_RESONANCE_FACTOR: float = 60.0
VOICE_FREQ_ADJ_YINYANG_FACTOR: float = -70.0
VOICE_FREQ_MIN_HZ: float = 80.0
VOICE_FREQ_MAX_HZ: float = 600.0
VOICE_FREQ_SOLFEGGIO_SNAP_HZ: float = 5.0

# Sephiroth Affinity
SEPHIROTH_ASPECT_DEFAULT: str = "tiphareth"
SEPHIROTH_AFFINITY_GEMATRIA_RANGES: Dict[range, str] = {
    range(1, 50): "malkuth",
    range(50, 80): "yesod",
    range(80, 110): "hod",
    range(110, 140): "netzach",
    range(140, 180): "tiphareth",
    range(180, 220): "geburah",
    range(220, 260): "chesed",
    range(260, 300): "binah",
    range(300, 350): "chokmah",
    range(350, 1000): "kether"
}

SEPHIROTH_AFFINITY_COLOR_MAP: Dict[str, str] = {
    "white": "kether", "grey": "chokmah", "black": "binah",
    "blue": "chesed", "red": "geburah", "yellow": "tiphareth",
    "gold": "tiphareth", "green": "netzach", "orange": "hod",
    "violet": "yesod", "purple": "yesod", "brown": "malkuth",
    "earth_tones": "malkuth", "silver": "daath", "lavender": "daath"
}

SEPHIROTH_AFFINITY_STATE_MAP: Dict[str, str] = {
    "spark": "kether", "dream": "yesod", "formative": "malkuth",
    "aware": "tiphareth", "integrated": "kether", "harmonized": "chesed"
}

# Affinity Weights
SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD: float = 0.80
SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD: float = 0.35
SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD: float = 0.65
SEPHIROTH_AFFINITY_YIN_SEPHIROTH: List[str] = ["binah", "geburah", "hod"]
SEPHIROTH_AFFINITY_YANG_SEPHIROTH: List[str] = ["chokmah", "chesed", "netzach"]
SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: List[str] = ["kether", "tiphareth", "yesod", "malkuth", "daath"]
SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT: float = 0.2
SEPHIROTH_AFFINITY_COLOR_WEIGHT: float = 0.25
SEPHIROTH_AFFINITY_STATE_WEIGHT: float = 0.15
SEPHIROTH_AFFINITY_YINYANG_WEIGHT: float = 0.1
SEPHIROTH_AFFINITY_BALANCE_WEIGHT: float = 0.1

# Elemental Affinity
ELEMENTAL_AFFINITY_DEFAULT: str = "aether"
ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: float = 0.55
ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: float = 0.70
ELEMENTAL_AFFINITY_VOWEL_MAP: Dict[str, float] = {
    'air': 0.2, 'earth': 0.2, 'water': 0.15, 'fire': 0.15
}
ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT: float = 0.3
ELEMENTAL_AFFINITY_COLOR_WEIGHT: float = 0.2
ELEMENTAL_AFFINITY_STATE_WEIGHT: float = 0.1
ELEMENTAL_AFFINITY_FREQ_WEIGHT: float = 0.1

ELEMENTAL_AFFINITY_FREQ_RANGES: List[Tuple[float, str]] = [
    (150, 'earth'), (300, 'water'), (500, 'fire'), (750, 'air'), (float('inf'), 'aether')
]

ELEMENTAL_AFFINITY_COLOR_MAP: Dict[str, str] = {
    "red": "fire", "orange": "fire", "brown": "earth", "earth_tones": "earth",
    "yellow": "air", "green": "earth/water", "blue": "water", "indigo": "water/aether",
    "violet": "aether", "white": "aether", "black": "earth", "grey": "air",
    "silver": "aether", "gold": "fire", "lavender": "aether"
}

ELEMENTAL_AFFINITY_STATE_MAP: Dict[str, str] = {
    "spark": "fire", "dream": "water", "formative": "earth",
    "aware": "air", "integrated": "aether", "harmonized": "water"
}

# Name Response Training
NAME_RESPONSE_TRAIN_BASE_INC: float = 0.02
NAME_RESPONSE_TRAIN_CYCLE_INC: float = 0.005
NAME_RESPONSE_TRAIN_NAME_FACTOR: float = 0.5
NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR: float = 0.8
NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT: float = 0.4

NAME_RESPONSE_STATE_FACTORS: Dict[str, float] = {
    'spark': 0.2, 'dream': 0.5, 'formative': 0.7, 'aware': 1.0,
    'integrated': 1.2, 'harmonized': 1.1, 'default': 0.8
}

# Heartbeat Entrainment
HEARTBEAT_ENTRAINMENT_INC_FACTOR: float = 0.05
HEARTBEAT_ENTRAINMENT_DURATION_CAP: float = 300.0

# Love Resonance
LOVE_RESONANCE_BASE_INC: float = 0.03
LOVE_RESONANCE_CYCLE_FACTOR_DECAY: float = 0.3
LOVE_RESONANCE_STATE_WEIGHT: Dict[str, float] = {
    'spark': 0.1, 'dream': 0.6, 'formative': 0.8, 'aware': 1.0,
    'integrated': 1.2, 'harmonized': 1.1, 'default': 0.7
}
LOVE_RESONANCE_FREQ_RES_WEIGHT: float = 0.5
LOVE_RESONANCE_HEARTBEAT_WEIGHT: float = 0.3
LOVE_RESONANCE_HEARTBEAT_SCALE: float = 0.4
LOVE_RESONANCE_EMOTION_BOOST_FACTOR: float = 0.1

# Sacred Geometry Integration
SACRED_GEOMETRY_STAGE_FACTOR_BASE: float = 1.0
SACRED_GEOMETRY_STAGE_FACTOR_SCALE: float = 0.5
SACRED_GEOMETRY_BASE_INC_BASE: float = 0.01
SACRED_GEOMETRY_BASE_INC_SCALE: float = 0.005

SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT: Dict[str, float] = {
    'tetrahedron': 1.1, 'hexahedron': 1.1, 'octahedron': 1.1, 'dodecahedron': 1.2,
    'icosahedron': 1.1, 'sphere': 1.0, 'point': 1.0, 'line': 1.0, 'triangle': 1.0,
    'square': 1.0, 'pentagon': 1.1, 'hexagram': 1.2, 'heptagon': 1.1, 'octagon': 1.1,
    'nonagon': 1.1, 'cross/cube': 1.1, 'vesica_piscis': 1.1, 'default': 1.0
}

SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT: Dict[str, float] = {
    'fire': 1.1, 'earth': 1.1, 'air': 1.1, 'water': 1.1, 'aether': 1.2,
    'light': 1.1, 'shadow': 0.9, 'default': 1.0
}

SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE: float = 0.8
SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE: float = 0.4
FIBONACCI_SEQUENCE: List[int] = [1, 1, 2, 3, 5, 8, 13, 21]
SACRED_GEOMETRY_FIB_MAX_IDX: int = 5

# Crystallization
ATTRIBUTE_COHERENCE_STD_DEV_SCALE: float = 2.0
CRYSTALLIZATION_REQUIRED_ATTRIBUTES: List[str] = [
    'name', 'soul_color', 'soul_frequency', 'sephiroth_aspect', 'elemental_affinity',
    'platonic_symbol', 'crystallization_level', 'attribute_coherence', 'voice_frequency'
]

CRYSTALLIZATION_COMPONENT_WEIGHTS: Dict[str, float] = {
    'name_resonance': 0.1, 'response_level': 0.1, 'state_stability': 0.1,
    'crystallization_level': 0.3, 'attribute_coherence': 0.2, 'attribute_presence': 0.1,
    'emotional_resonance': 0.1
}

CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD: float = 0.9
PLATONIC_DEFAULT_GEMATRIA_RANGE: int = 50

# =============================================================================
# END OF CONSTANTS
# =============================================================================

# Final validation
assert len(BRAIN_REGIONS) == len(REGION_PROPORTIONS), "Region lists must match"
assert abs(sum(REGION_PROPORTIONS.values()) - 1.0) < 0.001, "Region proportions must sum to 1.0"




















# # --- START OF constants.py ---

# """
# Central Constants for the Soul Development Framework (Version 4.2 - Principle-Driven S/C Calc)

# Consolidated and validated constants for simulation parameters, physics,
# field properties (absolute units/potentials), soul defaults (new units),
# stage thresholds/factors/flags (updated), sound generation,
# mappings, geometry/glyphs, logging. Includes S/C calculation weights and influence rates.
# """

# import numpy as np
# import logging
# from typing import Dict, List, Tuple, Any

# # --- Global Simulation & Physics ---
# SAMPLE_RATE: int = 44100
# MAX_AMPLITUDE: float = 0.95
# FLOAT_EPSILON: float = 1e-9
# PI: float = np.pi
# PLANCK_CONSTANT_H = 6.626e-34 # J*s
# GOLDEN_RATIO: float = (1 + np.sqrt(5)) / 2.0 # Phi (~1.618)
# PHI: float = GOLDEN_RATIO
# SILVER_RATIO: float = 1 + np.sqrt(2)
# EDGE_OF_CHAOS_RATIO: float = 1.0 / PHI
# SCHUMANN_FREQUENCY: float = 7.83 # Hz
# CORD_ACTIVATION_ENERGY_COST: float = 50.0 # SEU cost for activating a life cord
# EDGE_OF_CHAOS_DEFAULT = 0.618 # Golden ratio - optimal edge of chaos parameter
# # --- Paths & Logging ---
# DATA_DIR_BASE: str = "output"
# LOG_LEVEL = logging.INFO # Use INFO, but DEBUG in specific files for tracing
# LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# OUTPUT_DIR_BASE: str = "output"
# SPEED_OF_LIGHT: float = 299792458.0  # Speed of light in m/s
# SPEED_OF_SOUND: float = 343.0  # Speed of sound in m/s at 20C air
# ANCHOR_RESONANCE_MODIFIER: float = 0.85  # Modifier for anchor resonance calculations
# ANCHOR_STRENGTH_MODIFIER: float = 0.6  # Matches existing constant on line 359
# AIR_DENSITY: float = 1.225  # Air density in kg/m³ at sea level
# # --- Energy Units (Biomimetic Scaled Joules) ---



# # --- Soul Core Units & Ranges ---
# INITIAL_SPARK_BASE_FREQUENCY_HZ: float = 432.0
# INITIAL_SPARK_ENERGY_SEU: float = 500.0
# MAX_SOUL_ENERGY_SEU: float = 1e6
# PASSIVE_ENERGY_DISSIPATION_RATE_SEU_PER_SEC: float = 0.1

# # --- Spark Emergence & Harmonization Constants ---
# SPARK_EOC_ENERGY_YIELD_FACTOR: float = 50.0   # TUNE: How much EoC (0-1) multiplies base potential for initial energy.
# SPARK_FIELD_ENERGY_CATALYST_FACTOR: float = 0.05 # TUNE: Fraction of local void energy added as catalyst.
# SPARK_SEED_GEOMETRY: str = 'sphere'           # TUNE: Which platonic harmonic ratios seed the spark ('sphere', 'seed_of_life', etc.)
# SPARK_INITIAL_FACTOR_EOC_SCALE: float = 0.4   # TUNE: How much EoC (0-1) boosts initial phi_resonance, pattern_coherence etc.
# SPARK_INITIAL_FACTOR_ORDER_SCALE: float = 0.2 # TUNE: How much local void order (0-1) boosts initial factors.
# SPARK_INITIAL_FACTOR_PATTERN_SCALE: float = 0.1# TUNE: How much local void pattern (0-1) boosts initial factors.
# SPARK_INITIAL_FACTOR_BASE: float = 0.1        # TUNE: Minimum base value for initial factors like phi_resonance.
# SEPHIROTH_FREQ_NUDGE_FACTOR = 0.04
# HARMONIZATION_ITERATIONS: int = 244             # TUNE: Number of internal harmonization steps.
# HARMONIZATION_PATTERN_COHERENCE_RATE: float = 0.003 # TUNE: Rate factor builds towards 1.0 during harmonization.
# HARMONIZATION_PHI_RESONANCE_RATE: float = 0.002   # TUNE: Rate factor builds towards 1.0 during harmonization.
# HARMONIZATION_HARMONY_RATE: float = 0.0015        # TUNE: Rate factor builds towards 1.0 during harmonization.
# HARMONIZATION_ENERGY_GAIN_RATE: float = 0.25      # TUNE: SEU gain per iteration scaled by internal order proxy.
# HARMONIZATION_PHASE_ADJUST_RATE = 0.01 #(Small adjustment per iteration)
# HARMONIZATION_HARMONIC_ADJUST_RATE = 0.005 #(Even smaller adjustment)
# HARMONIZATION_CIRC_VAR_THRESHOLD = 0.15 #(Only adjust if variance is > 15%)
# HARMONIZATION_HARM_DEV_THRESHOLD = 0.08 #(Only adjust if average deviation > 8%)
# HARMONIZATION_TORUS_RATE: float = 0.002 # TUNE: Rate factor builds towards 1.0 during harmonization.

# PLATONIC_HARMONIC_RATIOS: Dict[str, List[float]] = {
#     'tetrahedron': [1.0, 2.0, 3.0, 5.0],
#     'hexahedron': [1.0, 2.0, 4.0, 8.0],
#     'octahedron': [1.0, 1.5, 2.0, 3.0],
#     'dodecahedron': [1.0, PHI, 2.0, PHI*2, 3.0],
#     'icosahedron': [1.0, 1.5, 2.0, 2.5, 3.0],
#     'sphere': [1.0, 1.5, 2.0, 2.5, 3.0, PHI, 4.0, 5.0], # Default Seed?
#     'merkaba': [1.0, 1.5, 2.0, 3.0, PHI, 4.0]
# }

# # --- Creator Entanglement (CE) - Resonance & Connection Constants
# CE_RESONANCE_THRESHOLD: float = 0.75    # Min resonance score (0-1) considered 'strong' for certain effects (e.g., pattern modulation?)
# CE_CONNECTION_FREQ_WEIGHT: float = 0.6    # Weight of frequency resonance score in connection strength calculation
# CE_CONNECTION_COHERENCE_WEIGHT: float = 0.4 # Weight of soul's coherence (normalized) in connection strength calculation
# CE_PATTERN_MODULATION_STRENGTH: float = 0.05 # How strongly Kether's geometry influences soul's pattern coherence during connection
# CREATOR_POTENTIAL_DEFAULT = 0.7  # Default creator potential for entanglement
# ENTANGLEMENT_PREREQ_STABILITY_MIN_SU = 30.0  # Minimum stability required for entanglement
# ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU = 25.0  # Minimum coherence required for entanglement
# SPEED_OF_SOUND = 343.0  # Speed of sound in m/s, used for wavelength calculations
# ASPECT_TRANSFER_THRESHOLD = 0.3  # Minimum resonance * connection for aspect transfer
# ASPECT_TRANSFER_STRENGTH_FACTOR = 0.15  # Strength increase factor for existing aspects
# ASPECT_TRANSFER_INITIAL_STRENGTH = 0.4  # Initial strength for newly transferred aspects

# # --- Stability & Coherence Units & Ranges ---
# MAX_STABILITY_SU: float = 100.0
# MAX_COHERENCE_CU: float = 100.0
# INITIAL_STABILITY_CALC_FACTOR: float = 0.2 # Initial scale applied to calculated S
# INITIAL_COHERENCE_CALC_FACTOR: float = 0.15 # Initial scale applied to calculated C

# # --- NEW: Stability Score Calculation Weights (Sum should ideally be 1.0) ---
# STABILITY_WEIGHT_FREQ: float = 0.30     # Contribution from frequency stability
# STABILITY_WEIGHT_PATTERN: float = 0.35   # Contribution from internal structure (layers, aspects, patterns)
# STABILITY_WEIGHT_FIELD: float = 0.20       # Contribution from external field influences (Guff, Sephiroth)
# STABILITY_WEIGHT_TORUS: float   = 0.15
# # Factors used WITHIN the pattern component calculation
# STABILITY_PATTERN_WEIGHT_LAYERS: float = 0.3
# STABILITY_PATTERN_WEIGHT_ASPECTS: float = 0.3
# STABILITY_PATTERN_WEIGHT_PHI: float = 0.2
# STABILITY_PATTERN_WEIGHT_ALIGNMENT: float = 0.2

# # --- Harmonic Strengthening (HS) - Targeted Refinement Cycle Constants (NEW for V4.3.8+) ---

# # Thresholds to trigger refinement for specific aspects (Scores 0-1, SU/CU 0-100)
# HS_TRIGGER_STABILITY_SU = 99.0          # Target stability threshold
# HS_TRIGGER_COHERENCE_CU = 99.0          # Target coherence threshold
# HS_TRIGGER_PHASE_COHERENCE = 0.98       # Target phase coherence 
# HS_TRIGGER_HARMONIC_PURITY = 0.98       # Target harmonic purity
# HS_TRIGGER_FACTOR_THRESHOLD = 0.98      # Target for phi, pattern, harmony, torus
# # Cycle Control
# HS_MAX_CYCLES: int =  256                # Maximum refinement iterations if thresholds not met
# HS_UPDATE_STATE_INTERVAL: int = 10         # How many cycles between S/C recalculations via update_state()
# # Base rates for factor improvements per cycle (modulated by current state)
# HS_BASE_RATE_PHI = 0.01                 # Base rate for phi improvement
# HS_BASE_RATE_PCOH = 0.01                # Base rate for pattern coherence
# HS_BASE_RATE_HARMONY = 0.01             # Base rate for harmony
# HS_BASE_RATE_TORUS = 0.01               # Base rate for torus
# # Adjustment rates for frequency structure refinement per cycle
# HS_PHASE_ADJUST_RATE = 0.01             # Rate for phase adjustment
# HS_HARMONIC_ADJUST_RATE = 0.01          # Rate for harmonic adjustment
# # Energy adjustment based on harmony changes during HS
# HS_ENERGY_ADJUST_FACTOR: float = 5.0        # Scales SEU gain/loss based on delta_harmony in a cycle
# # --- NEW: Coherence Score Calculation Weights (Sum should ideally be 1.0) ---
# COHERENCE_WEIGHT_PHASE = 0.20           # Weight for phase coherence
# COHERENCE_WEIGHT_HARMONY = 0.20         # Weight for harmony factor
# COHERENCE_WEIGHT_HARMONIC_PURITY = 0.10  # Weight for harmonic purity
# COHERENCE_WEIGHT_PATTERN = 0.15         # Weight for pattern coherence
# COHERENCE_WEIGHT_FIELD = 0.15           # Weight for field influence
# COHERENCE_WEIGHT_CREATOR = 0.10         # Weight for creator connection
# COHERENCE_WEIGHT_TORUS = 0.10           # Weight for toroidal flow
# # Verify sum is exactly 1.0
# assert abs((COHERENCE_WEIGHT_PHASE + COHERENCE_WEIGHT_HARMONY + 
#             COHERENCE_WEIGHT_HARMONIC_PURITY + COHERENCE_WEIGHT_PATTERN + 
#             COHERENCE_WEIGHT_FIELD + COHERENCE_WEIGHT_CREATOR + 
#             COHERENCE_WEIGHT_TORUS) - 1.0) < 0.000001, "Coherence weights must sum to 1.0"

# # Other Calculation Factors
# STABILITY_VARIANCE_PENALTY_K: float = 50.0 # How much freq variance hurts stability

# # --- Field System ---
# GRID_SIZE: Tuple[int, int, int] = (144, 144, 144)
# VOID_BASE_ENERGY_SEU: float = 10.0
# VOID_BASE_FREQUENCY_RANGE: Tuple[float, float] = (10.0, 1000.0)
# VOID_BASE_STABILITY_SU: float = 20.0 # Baseline SU towards which Void drifts
# VOID_BASE_COHERENCE_CU: float = 15.0 # Baseline CU towards which Void drifts
# VOID_CHAOS_ORDER_BALANCE: float = 0.5
# SEPHIROTH_DEFAULT_RADIUS: float = 8.0
# SEPHIROTH_INFLUENCE_FALLOFF: float = 1.5
# DEFAULT_PHI_HARMONIC_COUNT: int = 3
# GUFF_RADIUS_FACTOR: float = 0.3
# GUFF_CAPACITY: int = 100

# # --- Sephiroth Absolute Potentials ---
# SEPHIROTH_ENERGY_POTENTIALS_SEU: Dict[str, float] = { 'kether': MAX_SOUL_ENERGY_SEU * 0.95, 'chokmah': MAX_SOUL_ENERGY_SEU * 0.85, 'binah': MAX_SOUL_ENERGY_SEU * 0.80, 'daath': MAX_SOUL_ENERGY_SEU * 0.70, 'chesed': MAX_SOUL_ENERGY_SEU * 0.75, 'geburah': MAX_SOUL_ENERGY_SEU * 0.65, 'tiphareth': MAX_SOUL_ENERGY_SEU * 0.70, 'netzach': MAX_SOUL_ENERGY_SEU * 0.60, 'hod': MAX_SOUL_ENERGY_SEU * 0.55, 'yesod': MAX_SOUL_ENERGY_SEU * 0.45, 'malkuth': MAX_SOUL_ENERGY_SEU * 0.30 }
# SEPHIROTH_TARGET_STABILITY_SU: Dict[str, float] = { 'kether': 98.0, 'chokmah': 90.0, 'binah': 92.0, 'daath': 85.0, 'chesed': 88.0, 'geburah': 80.0, 'tiphareth': 95.0, 'netzach': 85.0, 'hod': 82.0, 'yesod': 88.0, 'malkuth': 75.0 }
# SEPHIROTH_TARGET_COHERENCE_CU: Dict[str, float] = { 'kether': 98.0, 'chokmah': 92.0, 'binah': 90.0, 'daath': 88.0, 'chesed': 90.0, 'geburah': 82.0, 'tiphareth': 95.0, 'netzach': 88.0, 'hod': 85.0, 'yesod': 90.0, 'malkuth': 70.0 }
# GUFF_TARGET_ENERGY_SEU: float = SEPHIROTH_ENERGY_POTENTIALS_SEU['kether'] * 0.9
# GUFF_TARGET_STABILITY_SU: float = SEPHIROTH_TARGET_STABILITY_SU['kether'] * 0.95
# GUFF_TARGET_COHERENCE_CU: float = SEPHIROTH_TARGET_COHERENCE_CU['kether'] * 0.95
# KETHER_FREQ: float = 963.0 # Example Kether base freq for Guff resonance calc

# # --- Transfer & Influence Rates ---
# ENERGY_TRANSFER_RATE_K: float = 0.05           # Base rate for SEU transfer
# GUFF_ENERGY_TRANSFER_RATE_K: float = 0.2       # SEU transfer rate in Guff
# SEPHIROTH_ENERGY_EXCHANGE_RATE_K: float = 0.05 # SEU exchange rate during Sephirah interaction
# # --- NEW: Influence Factor Rates ---
# GUFF_INFLUENCE_RATE_K: float = 0.05  # *** TUNE: How much each Guff step increments guff_influence_factor (0-1)
# SEPHIRAH_INFLUENCE_RATE_K: float = 0.15 # *** TUNE: How much each Sephirah interaction increments cumulative_sephiroth_influence (0-1)

# # --- Resonance & Aspects ---
# RESONANCE_INTEGER_RATIO_TOLERANCE: float = 0.02
# RESONANCE_PHI_RATIO_TOLERANCE: float = 0.03
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ: float = 0.5
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM: float = 0.3
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI: float = 0.2
# SEPHIROTH_ASPECT_TRANSFER_FACTOR: float = 0.2
# MAX_ASPECT_STRENGTH: float = 1.0

# # --- Geometric & Platonic Constants ---
# PLATONIC_BASE_FREQUENCIES: Dict[str, float] = {'tetrahedron': 396.0, 'hexahedron': 285.0, 'octahedron': 639.0, 'dodecahedron': 963.0, 'icosahedron': 369.0, 'sphere': 432.0, 'merkaba': 528.0 }
# GEOMETRY_EFFECTS: Dict[str, Dict[str, float]] = {'tetrahedron': {'energy_focus': 0.1, 'transformative_capacity': 0.07}, 'hexahedron': {'stability_factor_boost': 0.15, 'grounding': 0.12, 'energy_containment': 0.08}, 'octahedron': {'yin_yang_balance_push': 0.0, 'coherence_factor_boost': 0.1, 'stability_factor_boost': 0.05}, 'dodecahedron': {'unity_connection': 0.15, 'phi_resonance_boost': 0.12, 'transcendence': 0.1}, 'icosahedron': {'emotional_flow': 0.12, 'adaptability': 0.1, 'coherence_factor_boost': 0.08}, 'sphere': {'potential_realization': 0.1, 'unity_connection': 0.05}, 'merkaba': {'stability_factor_boost': 0.1, 'transformative_capacity': 0.12, 'field_resilience': 0.08}, 'flower_of_life': {'harmony_boost': 0.12, 'structural_integration': 0.1}, 'seed_of_life': {'potential_realization': 0.1, 'stability_factor_boost': 0.08}, 'vesica_piscis': {'yin_yang_balance_push': 0.0, 'connection_boost': 0.09}, 'tree_of_life': {'harmony_boost': 0.1, 'structural_integration': 0.1, 'connection_boost': 0.08}, 'metatrons_cube': {'structural_integration': 0.12, 'connection_boost': 0.12}, 'vector_equilibrium': {'yin_yang_balance_push': 0.0, 'zero_point_attunement': 0.15}, '64_tetrahedron': {'structural_integration': 0.15, 'energy_containment': 0.1} }
# DEFAULT_GEOMETRY_EFFECT: Dict[str, float] = {'stability_factor_boost': 0.01}
# SEPHIROTH_GLYPH_DATA: Dict[str, Dict[str, Any]] = {'kether': { 'platonic': 'dodecahedron', 'sigil': 'Point/Crown', 'gematria_keys': ['Kether', 'Crown', 'Will', 'Unity', 1], 'fibonacci': [1, 1] }, 'chokmah': { 'platonic': 'sphere', 'sigil': 'Line/Wheel', 'gematria_keys': ['Chokmah', 'Wisdom', 'Father', 2], 'fibonacci': [2] }, 'binah': { 'platonic': 'icosahedron', 'sigil': 'Triangle/Womb', 'gematria_keys': ['Binah', 'Understanding', 'Mother', 3], 'fibonacci': [3] }, 'chesed': { 'platonic': 'hexahedron', 'sigil': 'Square/Solid', 'gematria_keys': ['Chesed', 'Mercy', 'Grace', 4], 'fibonacci': [5] }, 'geburah': { 'platonic': 'tetrahedron', 'sigil': 'Pentagon/Sword', 'gematria_keys': ['Geburah', 'Severity', 'Strength', 5], 'fibonacci': [8] }, 'tiphareth': { 'platonic': 'octahedron', 'sigil': 'Hexagram/Sun', 'gematria_keys': ['Tiphareth', 'Beauty', 'Harmony', 6], 'fibonacci': [13] }, 'netzach': { 'platonic': 'icosahedron', 'sigil': 'Heptagon/Victory', 'gematria_keys': ['Netzach', 'Victory', 'Endurance', 7], 'fibonacci': [21] }, 'hod': { 'platonic': 'octahedron', 'sigil': 'Octagon/Splendor', 'gematria_keys': ['Hod', 'Splendor', 'Glory', 8], 'fibonacci': [34] }, 'yesod': { 'platonic': 'icosahedron', 'sigil': 'Nonagon/Foundation', 'gematria_keys': ['Yesod', 'Foundation', 'Moon', 9], 'fibonacci': [55] }, 'malkuth': { 'platonic': 'hexahedron', 'sigil': 'CrossInCircle/Kingdom', 'gematria_keys': ['Malkuth', 'Kingdom', 'Shekhinah', 'Earth', 10], 'fibonacci': [89] }, 'daath': { 'platonic': 'sphere', 'sigil': 'VoidPoint', 'gematria_keys': ['Daath', 'Knowledge', 'Abyss', 11], 'fibonacci': [] }, }

# # --- Physics / Field Dynamics ---
# HARMONIC_RESONANCE_ENERGY_BOOST: float = 0.012 # Influence factor on energy in VoidField resonance calc
# WAVE_PROPAGATION_SPEED: float = 0.2           # Rate for VoidField energy propagation
# ENERGY_DISSIPATION_RATE = 0.002              # Rate for VoidField energy dissipation

# # --- Stage Prerequisites (Using SU/CU) ---
# ENTANGLEMENT_PREREQ_STABILITY_MIN_SU: float = 75.0; 
# ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU: float = 75.0
# HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU: float = 70.0; 
# HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU: float = 70.0
# CORD_STABILITY_THRESHOLD_SU: float = 80.0; 
# CORD_COHERENCE_THRESHOLD_CU: float = 80.0
# HARMONY_PREREQ_CORD_INTEGRITY_MIN: float = 0.70; 
# HARMONY_PREREQ_STABILITY_MIN_SU: float = 75.0; 
# HARMONY_PREREQ_COHERENCE_MIN_CU: float = 75.0
# IDENTITY_STABILITY_THRESHOLD_SU: float = 85.0; 
# IDENTITY_COHERENCE_THRESHOLD_CU: float = 85.0
# IDENTITY_EARTH_RESONANCE_THRESHOLD: float = 0.75; 
# IDENTITY_CRYSTALLIZATION_THRESHOLD: float = 0.85
# BIRTH_PREREQ_CORD_INTEGRITY_MIN: float = 0.80; 
# BIRTH_PREREQ_EARTH_RESONANCE_MIN: float = 0.75

# # --- Readiness & Completion Flags ---
# FLAG_READY_FOR_GUFF="ready_for_guff";
# FLAG_GUFF_STRENGTHENED="guff_strengthened"; 
# FLAG_READY_FOR_JOURNEY="ready_for_journey"; 
# FLAG_SEPHIROTH_JOURNEY_COMPLETE="sephiroth_journey_complete"; 
# FLAG_READY_FOR_ENTANGLEMENT="ready_for_entanglement"; 
# FLAG_READY_FOR_COMPLETION="ready_for_completion"; 
# FLAG_CREATOR_ENTANGLED = "creator_entangled"  # Flag indicating successful creator entanglement
# FLAG_SPARK_HARMONIZED = "spark_harmonized"  # New Flag
# FLAG_READY_FOR_HARMONIZATION = "ready_for_harmonization"  # Flag indicating readiness for harmonization
# FLAG_READY_FOR_STRENGTHENING="ready_for_strengthening"; 
# FLAG_HARMONICALLY_STRENGTHENED="harmonically_strengthened"; 
# FLAG_READY_FOR_LIFE_CORD="ready_for_life_cord"; 
# FLAG_CORD_FORMATION_COMPLETE="cord_formation_complete"; 
# FLAG_READY_FOR_EARTH="ready_for_earth"; 
# FLAG_EARTH_HARMONIZED="earth_harmonized"; 
# FLAG_EARTH_ATTUNED = "earth_attuned"; # From Earth Harmony
# FLAG_READY_FOR_IDENTITY="ready_for_identity"; 
# FLAG_IDENTITY_CRYSTALLIZED="identity_crystallized"; 
# FLAG_READY_FOR_BIRTH="ready_for_birth"; 
# FLAG_BIRTH_COMPLETED = "birth_completed"
# FLAG_READY_FOR_EVOLUTION = "ready_for_evolution"
# FLAG_INCARNATED="incarnated"
# FLAG_ECHO_PROJECTED = "echo_projected"; 
# FLAG_EARTH_ATTUNED = "earth_attuned" # New Flags

# # --- Other Stage Parameters (Factors & Defaults - REVIEW those affecting SU/CU) ---
# # Constants for later stages are kept here, but the stage logic may need adjustment
# # to modify influence factors instead of direct SU/CU boosts if we want full consistency.
# GUFF_STRENGTHENING_DURATION: float = 10.0
# SEPHIROTH_JOURNEY_ATTRIBUTE_IMPART_FACTOR: float = 0.05
# SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR: float = 0.04
# ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR: float = 0.15 # OK
# ENTANGLEMENT_STABILITY_BOOST_FACTOR: float = 0.1 # *** REVIEW: Scales SU gain? ***
# ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE: float = 0.6; 
# ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE: float = 0.4
# ENTANGLEMENT_RESONANCE_BOOST_FACTOR: float = 0.05 # OK
# ENTANGLEMENT_STABILIZATION_ITERATIONS: int = 5
# ENTANGLEMENT_STABILIZATION_FACTOR_STRENGTH: float = 1.005 # OK
# HARMONIC_STRENGTHENING_INTENSITY_DEFAULT: float = 0.7; 
# HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT: float = 1.0
# FUNDAMENTAL_FREQUENCY_432: float = 432.0
# SOLFEGGIO_FREQUENCIES: Dict[str, float] = { 'UT': 396.0, 'RE': 417.0, 'MI': 528.0, 'FA': 639.0, 'SOL': 741.0, 'LA': 852.0, 'SI': 963.0 }
# HARMONIC_STRENGTHENING_TARGET_FREQS: List[float] = sorted(list(SOLFEGGIO_FREQUENCIES.values()) + [FUNDAMENTAL_FREQUENCY_432])
# HARMONIC_STRENGTHENING_TUNING_INTENSITY_FACTOR: float = 0.1; 
# HARMONIC_STRENGTHENING_TUNING_TARGET_REACH_HZ: float = 1.0
# HARMONIC_STRENGTHENING_PHI_AMP_INTENSITY_FACTOR: float = 0.05 # OK
# HARMONIC_STRENGTHENING_PHI_STABILITY_BOOST_FACTOR: float = 0.3 # *** REVIEW: Scales SU gain? ***
# HARMONIC_STRENGTHENING_PATTERN_STAB_INTENSITY_FACTOR: float = 0.04 # OK
# HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_FACTOR: float = 0.01; 
# HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_CAP: float = 0.1
# HARMONIC_STRENGTHENING_PATTERN_STAB_STABILITY_BOOST: float = 0.4 # *** REVIEW: Scales SU gain? ***
# HARMONIC_STRENGTHENING_COHERENCE_INTENSITY_FACTOR: float = 0.08 # *** REVIEW: Scales CU gain? ***
# HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_COUNT_NORM: float = 10.0
# HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_FACTOR: float = 0.03 # *** REVIEW: Scales CU gain? ***
# HARMONIC_STRENGTHENING_COHERENCE_HARMONY_BOOST: float = 0.25 # OK
# HARMONIC_STRENGTHENING_EXPANSION_INTENSITY_FACTOR: float = 0.1 # OK
# HARMONIC_STRENGTHENING_EXPANSION_STATE_FACTOR: float = 1.0
# HARMONIC_STRENGTHENING_EXPANSION_STR_INTENSITY_FACTOR: float = 0.03 # OK
# HARMONIC_STRENGTHENING_EXPANSION_STR_STATE_FACTOR: float = 0.5
# HARMONIC_STRENGTHENING_HARMONIC_COUNT: int = 7
# LIFE_CORD_COMPLEXITY_DEFAULT: float = 0.7
# PRIMARY_CHANNEL_BANDWIDTH_FACTOR: float = 200.0; 
# PRIMARY_CHANNEL_STABILITY_FACTOR_CONN: float = 0.7; 
# PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX: float = 0.3
# PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN: float = 0.6; 
# PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX: float = 0.4
# PRIMARY_CHANNEL_ELASTICITY_BASE: float = 0.5; 
# PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX: float = 0.3
# HARMONIC_NODE_COUNT_BASE: int = 3; 
# HARMONIC_NODE_COUNT_FACTOR: float = 15.0
# HARMONIC_NODE_AMP_BASE: float = 0.4; 
# HARMONIC_NODE_AMP_FACTOR_COMPLEX: float = 0.4; 
# HARMONIC_NODE_AMP_FALLOFF: float = 0.6
# HARMONIC_NODE_BW_INCREASE_FACTOR: float = 5.0
# MAX_CORD_CHANNELS: int = 7; 
# SECONDARY_CHANNEL_COUNT_FACTOR: float = 6.0; 
# SECONDARY_CHANNEL_FREQ_FACTOR: float = 0.1
# SECONDARY_CHANNEL_BW_EMOTIONAL: tuple[float,float]=(10.0,30.0); 
# SECONDARY_CHANNEL_RESIST_EMOTIONAL: tuple[float,float]=(0.4,0.3)
# SECONDARY_CHANNEL_BW_MENTAL: tuple[float,float]=(15.0,40.0); 
# SECONDARY_CHANNEL_RESIST_MENTAL: tuple[float,float]=(0.5,0.3)
# SECONDARY_CHANNEL_BW_SPIRITUAL: tuple[float,float]=(20.0,50.0); 
# SECONDARY_CHANNEL_RESIST_SPIRITUAL: tuple[float,float]=(0.6,0.3)
# FIELD_INTEGRATION_FACTOR_FIELD_STR: float = 0.6; 
# FIELD_INTEGRATION_FACTOR_CONN_STR: float = 0.4
# FIELD_EXPANSION_FACTOR: float = 1.05
# CORD_INTEGRITY_FACTOR_CONN_STR: float = 0.4; 
# CORD_INTEGRITY_FACTOR_STABILITY: float = 0.3; 
# CORD_INTEGRITY_FACTOR_EARTH_CONN: float = 0.3
# FINAL_STABILITY_BONUS_FACTOR: float = 0.15 # *** REVIEW: Scales SU bonus? ***


# #Earth Harmonisation (Earth Resonance) Constants
# EARTH_ANCHOR_RESONANCE: float = 0.9
# ANCHOR_STRENGTH_MODIFIER: float = 0.6; 
# EARTH_ANCHOR_STRENGTH: float = 0.9; 
# EARTH_CYCLE_FREQ_SCALING: float = 1.0e9
# ECHO_FIELD_STRENGTH_FACTOR: float = 0.85
# HARMONY_SCHUMANN_INTENSITY: float = 0.6
# HARMONY_CORE_INTENSITY: float = 0.4
# EARTH_FREQUENCY = 136.10; 
# EARTH_BREATH_FREQUENCY = 0.2
# EARTH_CONN_FACTOR_CONN_STR: float = 0.5; 
# EARTH_CONN_FACTOR_ELASTICITY: float = 0.3; 
# EARTH_CONN_BASE_FACTOR: float = 0.1
# EARTH_HARMONY_INTENSITY_DEFAULT: float = 0.7; 
# EARTH_HARMONY_DURATION_FACTOR_DEFAULT: float = 1.0
# EARTH_FREQUENCIES: Dict[str, float] = { "schumann": 7.83, "geomagnetic": 11.75, "core_resonance": EARTH_FREQUENCY, "breath_cycle": EARTH_BREATH_FREQUENCY, "heartbeat_cycle": 1.2, "circadian_cycle": 1.0/(24*3600)}
# EARTH_ELEMENTS: List[str] = ["earth", "water", "fire", "air", "aether"]

# # Surface resonance constants
# SURFACE_RESONANCE_FACTOR = 0.5
# SURFACE_RESONANCE_HARMONIC_THRESHOLD = 0.3

# # Element colors for layer visualization
# ELEMENT_COLORS = {
#     'earth': '#8B4513',
#     'air': '#87CEEB', 
#     'fire': '#FF4500',
#     'water': '#1E90FF',
#     'aether': '#E6E6FA'
# }

# # --- Earth Harmonization (Echo Attunement) Constants (NEW V4.3.8+) ---
# ECHO_PROJECTION_ENERGY_COST_FACTOR: float = 0.02 # % of spiritual energy cost
# ECHO_ATTUNEMENT_CYCLES: int = 84        # Number of attunement cycles
# ECHO_ATTUNEMENT_RATE: float = 0.8       # Base rate of state shift per cycle (modulated by resonance)
# ECHO_FIELD_COHERENCE_FACTOR: float = 0.75 # Coherence factor for echo field
# ATTUNEMENT_RESONANCE_THRESHOLD: float = 0.3   # Min resonance for attunement shift to occur
# ELEMENTAL_TARGET_EARTH: float = 0.8 # Target alignment for primary earth element
# ELEMENTAL_TARGET_OTHER: float = 0.4 # Target alignment for other elements
# ELEMENTAL_ALIGN_INTENSITY_FACTOR: float = 0.2 # Rate for elemental alignment nudge
# PLANETARY_FREQUENCIES: Dict[str, float] = { # Example Frequencies (Hz) - NEEDS REAL VALUES/SOURCE
#     "Sun": 126.22, "Moon": 210.42, "Mercury": 141.27, "Venus": 221.23, "Earth": 194.71,
#     "Mars": 144.72, "Jupiter": 183.58, "Saturn": 147.85, "Uranus": 207.36,
#     "Neptune": 211.44, "Pluto": 140.25 }
# PLANETARY_ATTUNEMENT_CYCLES: int = 21 # Cycles for planetary resonance step
# PLANETARY_RESONANCE_RATE: float = 0.0095 # Rate for planetary resonance factor gain
# GAIA_CONNECTION_CYCLES: int = 13 # Cycles for Gaia connection step
# GAIA_CONNECTION_FACTOR: float = 0.023 # Rate for Gaia connection factor gain
# STRESS_FEEDBACK_FACTOR: float = 0.0024 # How much echo discordance impacts main soul stability variance
# HARMONY_CYCLE_NAMES: List[str] = ["circadian", "heartbeat", "breath"]; 
# HARMONY_CYCLE_IMPORTANCE: Dict[str, float] = {"circadian": 0.6, "heartbeat": 0.8, "breath": 1.0}
# HARMONY_CYCLE_SYNC_TARGET_BASE: float = 0.9; 
# HARMONY_CYCLE_SYNC_INTENSITY_FACTOR: float = 0.1; 
# HARMONY_CYCLE_SYNC_DURATION_FACTOR: float = 1.0
# HARMONY_PLANETARY_RESONANCE_TARGET: float = 0.85; 
# HARMONY_PLANETARY_RESONANCE_FACTOR: float = 0.15
# HARMONY_GAIA_CONNECTION_TARGET: float = 0.90; 
# HARMONY_GAIA_CONNECTION_FACTOR: float = 0.2
# HARMONY_FINAL_STABILITY_BONUS: float = 0.08 # *** REVIEW: Scales SU bonus? ***
# HARMONY_FINAL_COHERENCE_BONUS: float = 0.08 # *** REVIEW: Scales CU bonus? ***
# HARMONY_CYCLE_SYNC_TARGET_BASE: float = 0.9
# HARMONY_CYCLE_SYNC_INTENSITY_FACTOR: float = 0.1
# HARMONY_FREQ_TUNING_FACTOR: float = 0.15 # Re-add if needed by Earth Harmony freq attunement
# HARMONY_FREQ_TUNING_TARGET_REACH_HZ: float = 1.0; 
# HARMONY_FREQ_UPDATE_HARMONIC_COUNT: int = 5
# HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT: float = 0.6; 
# HARMONY_FREQ_TARGET_SOUL_WEIGHT: float = 0.2; 
# HARMONY_FREQ_TARGET_CORE_WEIGHT: float = 0.2
# HARMONY_FREQ_RES_WEIGHT_SCHUMANN: float = 0.7; 
# HARMONY_FREQ_RES_WEIGHT_OTHER: float = 0.3
# HARMONY_ELEM_RES_WEIGHT_PRIMARY: float = 0.6; 
# HARMONY_ELEM_RES_WEIGHT_AVERAGE: float = 0.4



# # --- Identity Crystallization (Astrology) Constants (NEW V4.3.8+) ---
# ZODIAC_SIGNS: List[Dict[str, str]] = [ # Using 13 signs format
#     {"name": "Aries", "symbol": "♈", "start_date": "April 19", "end_date": "May 13"},
#     {"name": "Taurus", "symbol": "♉", "start_date": "May 14", "end_date": "June 19"},
#     {"name": "Gemini", "symbol": "♊", "start_date": "June 20", "end_date": "July 20"},
#     {"name": "Cancer", "symbol": "♋", "start_date": "July 21", "end_date": "August 9"},
#     {"name": "Leo", "symbol": "♌", "start_date": "August 10", "end_date": "September 15"},
#     {"name": "Virgo", "symbol": "♍", "start_date": "September 16", "end_date": "October 30"},
#     {"name": "Libra", "symbol": "♎", "start_date": "October 31", "end_date": "November 22"},
#     {"name": "Scorpio", "symbol": "♏", "start_date": "November 23", "end_date": "November 29"},
#     {"name": "Ophiuchus", "symbol": "⛎", "start_date": "November 30", "end_date": "December 17"},
#     {"name": "Sagittarius", "symbol": "♐", "start_date": "December 18", "end_date": "January 18"},
#     {"name": "Capricorn", "symbol": "♑", "start_date": "January 19", "end_date": "February 15"},
#     {"name": "Aquarius", "symbol": "♒", "start_date": "February 16", "end_date": "March 11"},
#     {"name": "Pisces", "symbol": "♓", "start_date": "March 12", "end_date": "April 18"}
# ]

# ZODIAC_TRAITS: Dict[str, Dict[str, List[str]]] = { # Provided trait lists
#     "Aries": {"positive": ["Courageous","Energetic","Adventurous","Enthusiastic","Confident","Dynamic","Independent","Pioneering","Ambitious","Passionate"], "negative": ["Impulsive","Impatient","Aggressive","Reckless","Self-centered","Stubborn","Short-tempered","Competitive","Blunt","Domineering"]},
#     "Taurus": {"positive": ["Reliable","Patient","Practical","Devoted","Persistent","Loyal","Stable","Loving","Determined","Sensual"], "negative": ["Possessive","Stubborn","Materialistic","Self-indulgent","Inflexible","Jealous","Resistant to change","Greedy","Lazy","Rigid"]},
#     "Gemini": {"positive": ["Versatile","Communicative","Curious","Adaptable","Intellectual","Social","Witty","Quick-minded","Expressive","Inquisitive"], "negative": ["Inconsistent","Restless","Superficial","Indecisive","Nervous","Unreliable","Scattered","Dual-natured","Gossipy","Easily bored"]},
#     "Cancer": {"positive": ["Nurturing","Intuitive","Emotional","Protective","Sympathetic","Compassionate","Empathetic","Tenacious","Caring","Loyal"], "negative": ["Moody","Overly sensitive","Clingy","Insecure","Suspicious","Manipulative","Pessimistic","Guarded","Self-pitying","Vengeful"]},
#     "Leo": {"positive": ["Generous","Charismatic","Creative","Enthusiastic","Confident","Loyal","Warm-hearted","Leaders","Optimistic","Passionate"], "negative": ["Arrogant","Domineering","Attention-seeking","Dramatic","Stubborn","Self-centered","Bossy","Prideful","Vain","Inflexible"]},
#     "Virgo": {"positive": ["Analytical","Practical","Meticulous","Reliable","Diligent","Intelligent","Modest","Precise","Helpful","Orderly"], "negative": ["Overly critical","Perfectionist","Obsessive","Fussy","Anxious","Judgmental","Overthinking","Rigid","Skeptical","Nitpicking"]},
#     "Libra": {"positive": ["Diplomatic","Fair-minded","Cooperative","Social","Harmonious","Charming","Balanced","Idealistic","Graceful","Tactful"], "negative": ["Indecisive","People-pleasing","Superficial","Unreliable","Avoidant","Flirtatious","Detached","Conflict-avoiding","Vain","Non-confrontational"]},
#     "Scorpio": {"positive": ["Passionate","Resourceful","Brave","Loyal","Determined","Focused","Intuitive","Magnetic","Ambitious","Perceptive"], "negative": ["Jealous","Secretive","Manipulative","Untrusting","Vengeful","Obsessive","Possessive","Controlling","Suspicious","Intense"]},
#     "Ophiuchus": {"positive": ["Healer","Truth-seeker","Visionary","Magnetic","Wise","Mystical","Intellectual","Compassionate","Enlightened","Ambitious"], "negative": ["Jealous","Secretive","Power-seeking","Elitist","Temperamental","Arrogant","Egotistical","Envious","Critical","Mistrusting"]},
#     "Sagittarius": {"positive": ["Optimistic","Freedom-loving","Adventurous","Philosophical","Expansive","Honest","Enthusiastic","Open-minded","Idealistic","Generous"], "negative": ["Tactless","Inconsistent","Restless","Overconfident","Reckless","Superficial","Irresponsible","Blunt","Impatient","Uncommitted"]},
#     "Capricorn": {"positive": ["Disciplined","Responsible","Patient","Ambitious","Resourceful","Practical","Loyal","Organized","Persistent","Reserved"], "negative": ["Pessimistic","Stubborn","Detached","Unforgiving","Status-conscious","Conservative","Rigid","Materialistic","Cold","Workaholic"]},
#     "Aquarius": {"positive": ["Progressive","Original","Humanitarian","Independent","Intellectual","Inventive","Friendly","Visionary","Innovative","Idealistic"], "negative": ["Detached","Unpredictable","Stubborn","Aloof","Eccentric","Rebellious","Unemotional","Inconsistent","Impersonal","Contrary"]},
#     "Pisces": {"positive": ["Compassionate","Intuitive","Imaginative","Gentle","Empathetic","Creative","Spiritual","Adaptable","Artistic","Dreamy"], "negative": ["Escapist","Overly sensitive","Indecisive","Self-pitying","Unrealistic","Delusional","Impressionable","Dependent","Procrastinating","Easily influenced"]}
# }
# ASTROLOGY_MAX_POSITIVE_TRAITS: int = 5
# ASTROLOGY_MAX_NEGATIVE_TRAITS: int = 2
# NAME_GEMATRIA_RESONANT_NUMBERS: List[int] = [3, 7, 9, 11, 13, 22]
# NAME_RESONANCE_BASE: float = 0.1; 
# NAME_RESONANCE_WEIGHT_VOWEL: float = 0.3; 
# NAME_RESONANCE_WEIGHT_LETTER: float = 0.2; 
# NAME_RESONANCE_WEIGHT_GEMATRIA: float = 0.4
# VOICE_FREQ_BASE: float = 220.0; 
# VOICE_FREQ_ADJ_LENGTH_FACTOR: float = -50.0; 
# VOICE_FREQ_ADJ_VOWEL_FACTOR: float = 80.0; 
# VOICE_FREQ_ADJ_GEMATRIA_FACTOR: float = 40.0; 
# VOICE_FREQ_ADJ_RESONANCE_FACTOR: float = 60.0; 
# VOICE_FREQ_ADJ_YINYANG_FACTOR: float = -70.0
# VOICE_FREQ_MIN_HZ: float = 80.0; 
# VOICE_FREQ_MAX_HZ: float = 600.0; 
# VOICE_FREQ_SOLFEGGIO_SNAP_HZ: float = 5.0
# COLOR_SPECTRUM: Dict[str, Dict] = { "red": {"frequency": (400, 480), "hex": "#FF0000"}, "orange": {"frequency": (480, 510), "hex": "#FFA500"}, "gold": {"frequency": (510, 530), "hex": "#FFD700"}, "yellow": {"frequency": (530, 560), "hex": "#FFFF00"}, "green": {"frequency": (560, 610), "hex": "#00FF00"}, "blue": {"frequency": (610, 670), "hex": "#0000FF"}, "indigo": {"frequency": (670, 700), "hex": "#4B0082"}, "violet": {"frequency": (700, 790), "hex": "#8A2BE2"}, "white": {"frequency": (400, 790), "hex": "#FFFFFF"}, "black": {"frequency": (0, 0), "hex": "#000000"}, "silver": {"frequency": (0, 0), "hex": "#C0C0C0"}, "magenta": {"frequency": (0, 0), "hex": "#FF00FF"}, "grey": {"frequency": (0,0), "hex": "#808080"}, "earth_tones": {"frequency": (150, 250), "hex": "#A0522D"}, "lavender": {"frequency": (700, 750), "hex": "#E6E6FA"}, "brown": {"frequency": (150, 250), "hex": "#A52A2A"} }
# COLOR_FREQ_DEFAULT: float = 500.0
# SEPHIROTH_ASPECT_DEFAULT: str = "tiphareth"; 
# SEPHIROTH_AFFINITY_GEMATRIA_RANGES: Dict[range, str] = {range(1, 50): "malkuth", range(50, 80): "yesod", range(80, 110): "hod", range(110, 140): "netzach", range(140, 180): "tiphareth", range(180, 220): "geburah", range(220, 260): "chesed", range(260, 300): "binah", range(300, 350): "chokmah", range(350, 1000): "kether"}
# SEPHIROTH_AFFINITY_COLOR_MAP: Dict[str, str] = {"white": "kether", "grey": "chokmah", "black": "binah", "blue": "chesed", "red": "geburah", "yellow": "tiphareth", "gold": "tiphareth", "green": "netzach", "orange": "hod", "violet": "yesod", "purple": "yesod", "brown": "malkuth", "earth_tones": "malkuth", "silver": "daath", "lavender": "daath"}
# SEPHIROTH_AFFINITY_STATE_MAP: Dict[str, str] = {"spark":"kether", "dream": "yesod", "formative": "malkuth", "aware": "tiphareth", "integrated": "kether", "harmonized": "chesed"}
# SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD: float = 0.80; 
# SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD: float = 0.35; 
# SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD: float = 0.65
# SEPHIROTH_AFFINITY_YIN_SEPHIROTH: List[str] = ["binah", "geburah", "hod"]; 
# SEPHIROTH_AFFINITY_YANG_SEPHIROTH: List[str] = ["chokmah", "chesed", "netzach"]; 
# SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: List[str] = ["kether", "tiphareth", "yesod", "malkuth", "daath"]
# SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT: float = 0.2; 
# SEPHIROTH_AFFINITY_COLOR_WEIGHT: float = 0.25; 
# SEPHIROTH_AFFINITY_STATE_WEIGHT: float = 0.15; 
# SEPHIROTH_AFFINITY_YINYANG_WEIGHT: float = 0.1; 
# SEPHIROTH_AFFINITY_BALANCE_WEIGHT: float = 0.1
# ELEMENTAL_AFFINITY_DEFAULT: str = "aether"; 
# ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: float = 0.55; 
# ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: float = 0.70
# ELEMENTAL_AFFINITY_VOWEL_MAP: Dict[str, float] = {'air': 0.2, 'earth': 0.2, 'water': 0.15, 'fire': 0.15}
# ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT: float = 0.3; 
# ELEMENTAL_AFFINITY_COLOR_WEIGHT: float = 0.2; 
# ELEMENTAL_AFFINITY_STATE_WEIGHT: float = 0.1; 
# ELEMENTAL_AFFINITY_FREQ_WEIGHT: float = 0.1
# ELEMENTAL_AFFINITY_FREQ_RANGES: List[Tuple[float, str]] = [(150, 'earth'), (300, 'water'), (500, 'fire'), (750, 'air'), (float('inf'), 'aether')]
# ELEMENTAL_AFFINITY_COLOR_MAP: Dict[str, str] = {"red": "fire", "orange": "fire", "brown": "earth", "earth_tones": "earth", "yellow": "air", "green": "earth/water", "blue": "water", "indigo": "water/aether", "violet": "aether", "white": "aether", "black": "earth", "grey": "air", "silver": "aether", "gold": "fire", "lavender": "aether"}
# ELEMENTAL_AFFINITY_STATE_MAP: Dict[str, str] = {"spark":"fire", "dream": "water", "formative": "earth", "aware": "air", "integrated": "aether", "harmonized": "water"}
# LOVE_RESONANCE_FREQ = 528.0
# PLATONIC_ELEMENT_MAP: Dict[str, str] = {'earth': 'hexahedron', 'water': 'icosahedron', 'fire': 'tetrahedron', 'air': 'octahedron', 'aether': 'dodecahedron', 'spirit': 'dodecahedron', 'void': 'sphere', 'light':'merkaba'}
# PLATONIC_SOLIDS: List[str] = ['tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron', 'sphere', 'merkaba']
# PLATONIC_DEFAULT_GEMATRIA_RANGE: int = 50
# NAME_RESPONSE_TRAIN_BASE_INC: float = 0.02; 
# NAME_RESPONSE_TRAIN_CYCLE_INC: float = 0.005; 
# NAME_RESPONSE_TRAIN_NAME_FACTOR: float = 0.5; 
# NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR: float = 0.8; 
# NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT: float = 0.4
# NAME_RESPONSE_STATE_FACTORS: Dict[str, float] = {'spark': 0.2, 'dream': 0.5, 'formative': 0.7, 'aware': 1.0, 'integrated': 1.2, 'harmonized': 1.1, 'default': 0.8}
# HEARTBEAT_ENTRAINMENT_INC_FACTOR: float = 0.05; 
# HEARTBEAT_ENTRAINMENT_DURATION_CAP: float = 300.0
# LOVE_RESONANCE_BASE_INC: float = 0.03; 
# LOVE_RESONANCE_CYCLE_FACTOR_DECAY: float = 0.3
# LOVE_RESONANCE_STATE_WEIGHT: Dict[str, float] = {'spark': 0.1, 'dream': 0.6, 'formative': 0.8, 'aware': 1.0, 'integrated': 1.2, 'harmonized': 1.1, 'default': 0.7}
# LOVE_RESONANCE_FREQ_RES_WEIGHT: float = 0.5; 
# LOVE_RESONANCE_HEARTBEAT_WEIGHT: float = 0.3; 
# LOVE_RESONANCE_HEARTBEAT_SCALE: float = 0.4
# LOVE_RESONANCE_EMOTION_BOOST_FACTOR: float = 0.1
# SACRED_GEOMETRY_STAGES: List[str] = ["seed_of_life", "flower_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "64_tetrahedron"]
# SACRED_GEOMETRY_STAGE_FACTOR_BASE: float = 1.0; 
# SACRED_GEOMETRY_STAGE_FACTOR_SCALE: float = 0.5
# SACRED_GEOMETRY_BASE_INC_BASE: float = 0.01; 
# SACRED_GEOMETRY_BASE_INC_SCALE: float = 0.005
# SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT: Dict[str, float] = {'tetrahedron': 1.1, 'hexahedron': 1.1, 'octahedron': 1.1, 'dodecahedron': 1.2, 'icosahedron': 1.1, 'sphere': 1.0, 'point': 1.0, 'line': 1.0, 'triangle': 1.0, 'square': 1.0, 'pentagon': 1.1, 'hexagram': 1.2, 'heptagon': 1.1, 'octagon': 1.1, 'nonagon': 1.1, 'cross/cube': 1.1, 'vesica_piscis': 1.1, 'default': 1.0}
# SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT: Dict[str, float] = {'fire': 1.1, 'earth': 1.1, 'air': 1.1, 'water': 1.1, 'aether': 1.2, 'light': 1.1, 'shadow': 0.9, 'default': 1.0}
# SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE: float = 0.8; 
# SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE: float = 0.4
# FIBONACCI_SEQUENCE: List[int] = [1, 1, 2, 3, 5, 8, 13, 21]; 
# SACRED_GEOMETRY_FIB_MAX_IDX: int = 5
# ATTRIBUTE_COHERENCE_STD_DEV_SCALE: float = 2.0
# CRYSTALLIZATION_REQUIRED_ATTRIBUTES: List[str] = ['name', 'soul_color', 'soul_frequency', 'sephiroth_aspect', 'elemental_affinity', 'platonic_symbol', 'crystallization_level', 'attribute_coherence', 'voice_frequency']
# CRYSTALLIZATION_COMPONENT_WEIGHTS: Dict[str, float] = { 'name_resonance': 0.1, 'response_level': 0.1, 'state_stability': 0.1, 'crystallization_level': 0.3, 'attribute_coherence': 0.2, 'attribute_presence': 0.1, 'emotional_resonance': 0.1 }
# CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD: float = 0.9
# BIRTH_INTENSITY_DEFAULT: float = 0.7
# BIRTH_CONN_WEIGHT_RESONANCE: float = 0.6; 
# BIRTH_CONN_WEIGHT_INTEGRITY: float = 0.4
# BIRTH_CONN_STRENGTH_FACTOR: float = 0.5; 
# BIRTH_CONN_STRENGTH_CAP: float = 0.95; 
# BIRTH_CONN_MOTHER_STRENGTH_FACTOR: float = 0.1
# BIRTH_CONN_TRAUMA_FACTOR: float = 0.3; 
# BIRTH_CONN_ACCEPTANCE_TRAUMA_FACTOR: float = 0.4 
# BIRTH_CONN_MOTHER_TRAUMA_REDUCTION: float = 0.2
# BIRTH_ACCEPTANCE_MIN: float = 0.2; 
# BIRTH_ACCEPTANCE_TRAUMA_FACTOR: float = 0.8; 
# BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR: float = 0.1
# BIRTH_CORD_TRANSFER_INTENSITY_FACTOR: float = 0.2; 
# BIRTH_CORD_MOTHER_EFFICIENCY_FACTOR: float = 0.1
# BIRTH_CORD_INTEGRATION_CONN_FACTOR: float = 0.9; 
# BIRTH_CORD_MOTHER_INTEGRATION_FACTOR: float = 0.08
# BIRTH_VEIL_STRENGTH_BASE: float = 0.6; 
# BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR: float = 0.3
# BIRTH_VEIL_PERMANENCE_BASE: float = 0.7; 
# BIRTH_VEIL_PERMANENCE_INTENSITY_FACTOR: float = 0.25
# BIRTH_VEIL_RETENTION_BASE: float = 0.1; 
# BIRTH_VEIL_RETENTION_INTENSITY_FACTOR: float = -0.05; 
# BIRTH_VEIL_RETENTION_MIN: float = 0.02; 
# BIRTH_VEIL_MOTHER_RETENTION_FACTOR: float = 0.02
# BIRTH_VEIL_MEMORY_RETENTION_MODS: Dict[str, float] = {'core_identity': 0.1, 'creator_connection': 0.05, 'journey_lessons': 0.02, 'specific_details': -0.05}
# BIRTH_BREATH_AMP_BASE: float = 0.5; 
# BIRTH_BREATH_AMP_INTENSITY_FACTOR: float = 0.3
# BIRTH_BREATH_DEPTH_BASE: float = 0.6; 
# BIRTH_BREATH_DEPTH_INTENSITY_FACTOR: float = 0.2
# BIRTH_BREATH_SYNC_RESONANCE_FACTOR: float = 0.8; 
# BIRTH_BREATH_MOTHER_SYNC_FACTOR: float = 0.3
# BIRTH_BREATH_INTEGRATION_CONN_FACTOR: float = 0.7
# BIRTH_BREATH_RESONANCE_BOOST_FACTOR: float = 0.1; 
# BIRTH_BREATH_MOTHER_RESONANCE_BOOST: float = 0.15
# BIRTH_BREATH_ENERGY_SHIFT_FACTOR: float = 0.15; 
# BIRTH_BREATH_MOTHER_ENERGY_BOOST: float = 0.1
# BIRTH_BREATH_PHYSICAL_ENERGY_BASE: float = 0.5; 
# BIRTH_BREATH_PHYSICAL_ENERGY_SCALE: float = 0.8
# BIRTH_BREATH_SPIRITUAL_ENERGY_BASE: float = 0.7; 
# BIRTH_BREATH_SPIRITUAL_ENERGY_SCALE: float = -0.5; 
# BIRTH_BREATH_SPIRITUAL_ENERGY_MIN: float = 0.1
# BIRTH_FINAL_INTEGRATION_WEIGHT_CONN: float = 0.4; 
# BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT: float = 0.3; 
# BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH: float = 0.3
# BIRTH_FINAL_MOTHER_INTEGRATION_BOOST: float = 0.05
# BIRTH_FINAL_FREQ_FACTOR: float = 0.8
# BIRTH_FINAL_STABILITY_FACTOR: float = 0.9 # *** REVIEW: Multiplies SU? Change to modify influence? ***
# BIRTH_ENERGY_BUFFER_FACTOR: float = 1.4 # Target max buffer (scaled by soul completeness)
# BIRTH_ALLOC_SEED_CORE: float = 0.10 # Proportion of BEU for Seed Core
# BIRTH_ALLOC_REGIONS: float = 0.30 # Proportion of BEU for Region Dev
# BIRTH_ALLOC_MYCELIAL: float = 0.60 # Proportion of BEU for Mycelial Store
# VEIL_BASE_RETENTION: float = 0.05 # Base retention factor before coherence bonus
# VEIL_COHERENCE_RESISTANCE_FACTOR: float = 0.15 # How much coherence resists veil (scales bonus)
# BIRTH_FINAL_FREQ_SHIFT_FACTOR: float = 0.05 # Max % frequency drop due to density
# BIRTH_FINAL_STABILITY_PENALTY_FACTOR: float = 0.03 # Max % stability drop due to shock
# BIRTH_ATTACHMENT_MIN_CORD_INTEGRITY: float = 0.75 # Min cord integrity for attachment
# BIRTH_ALLOC_REGIONS: float = 0.30 # Proportion of BEU for Region Dev
# BIRTH_ALLOC_MYCELIAL: float = 0.60 # Proportion of BEU for Mycelial Store
# LIFE_CORD_FREQUENCIES = {'primary': 528.0}
# # Birth Prerequisites
# BIRTH_MIN_STABILITY_SU = 75.0
# BIRTH_MIN_COHERENCE_CU = 75.0
# BIRTH_MIN_CRYSTALLIZATION = 0.75

# # Birth Process Configuration
# BIRTH_WOMB_DEVELOPMENT_CYCLES = 3
# BIRTH_ALLOC_SEED_CORE = 0.4  # 40% of energy to brain seed core
# BIRTH_ALLOC_REGIONS = 0.3    # 30% to brain regions
# BIRTH_ALLOC_MYCELIAL = 0.3   # 30% to mycelial networks

# # (M) CRITICAL DIMENSION UPDATE
# GRID_DIMENSIONS: Tuple[int, int, int] = (256, 256, 256)
# BRAIN_GRID_SIZE: Tuple[int, int, int] = (256, 256, 256) # Ensure consistency

# # Brain Constants
# INITIAL_SPARK_ENERGY_FROM_CREATOR_FACTOR: float = 0.4 # Proportion of Soul's total potential physical energy that comes as the initial spark via BrainSeed
# BRAIN_MIN_ENERGY_BEU: float = 1.0e6  # Minimum Brain Energy Units for viability (after spark)
# BRAIN_DEFAULT_SIZE_X = 256
# BRAIN_DEFAULT_SIZE_Y = 256
# BRAIN_DEFAULT_SIZE_Z = 256



# # Birth Standing Wave Constants
# BIRTH_STANDING_WAVE_LENGTH = 10.0  # meters
# BIRTH_STANDING_WAVE_AMPLITUDE = 0.8
# BIRTH_STANDING_WAVE_SOUND_DURATION = 8.0
# BIRTH_STANDING_WAVE_SOUND_AMPLITUDE = 0.5

# # Birth Echo Field Constants
# BIRTH_ECHO_FIELD_STRENGTH_FACTOR = 0.7
# BIRTH_ECHO_FIELD_COHERENCE_FACTOR = 0.8
# BIRTH_ECHO_FIELD_CHAMBERS = 5
# BIRTH_ECHO_FIELD_SOUND_DURATION = 12.0
# BIRTH_ECHO_FIELD_SOUND_AMPLITUDE = 0.6


# # Define fallback constants in case import fails
# BRAIN_FREQUENCIES = {
#     'delta': (0.5, 4),      # Deep sleep
#     'theta': (4, 8),        # Drowsy, meditation 
#     'beta': (13, 30),       # Alert, active
#     'gamma': (30, 100),     # High cognition
#     'lambda': (100, 400)    # Higher spiritual states
# }

# # --- Brain Seed Constants ---
# MIN_BRAIN_SEED_ENERGY: float = 1.0         # Minimum BEU for brain seed operation
# DEFAULT_BRAIN_SEED_FREQUENCY: float = 7.83 # Default frequency (Schumann resonance)
# SEED_FIELD_RADIUS: float = 5.0             # Energy field radius around seed
# MAX_BRAIN_SEED_CAPACITY: float = 1000.0    # Maximum energy capacity for brain seed
# BRAIN_FREQUENCY_SCALE: float = 0.85        # Scale factor for brain vs soul frequency
# ENERGY_SCALE_FACTOR: float = 1e14
# BRAIN_ENERGY_UNIT_PER_JOULE: float = 1e12    # 1 Joule = 1e12 BEU (or 1 BEU = 1e-12 Joules)
# # For clarity, let's define BEU to Joules:
# JOULES_PER_BEU: float = 1e-12 # 1 BEU = 1 picoJoule


# # --- Brain Structure Constants ---
# BRAIN_GRID_SIZE: Tuple[int, int, int] = (256, 256, 256)  # Brain structure grid dimensions
# BRAIN_REGIONS: List[str] = ['frontal', 'parietal', 'temporal', 'occipital', 'limbic', 'brain_stem', 'cerebellum']
# BRAIN_COMPLEXITY_THRESHOLDS: Dict[str, float] = {
#     'energy_coverage': 0.3,     # Minimum energy coverage for complexity
#     'mycelial_coverage': 0.2,   # Minimum mycelial coverage
#     'avg_resonance': 0.3        # Average resonance threshold
# }

# # --- Consciousness Activation Thresholds ---
# CONSCIOUSNESS_ACTIVATION_THRESHOLDS: Dict[str, float] = {
#     'dream': 0.4,      # Threshold for dream state activation
#     'liminal': 0.6,    # Threshold for liminal state activation  
#     'aware': 0.8       # Threshold for aware state activation
# }

# # --- Brain Region Constants ---
# REGION_FRONTAL: str = 'frontal'
# REGION_PARIETAL: str = 'parietal'
# REGION_TEMPORAL: str = 'temporal'
# REGION_OCCIPITAL: str = 'occipital'
# REGION_LIMBIC: str = 'limbic'
# REGION_BRAIN_STEM: str = 'brain_stem'
# REGION_CEREBELLUM: str = 'cerebellum'

# # --- Energy Units ---
# SYNAPSE_ENERGY_JOULES: float = 1e-14       # Energy of one synaptic firing in Joules
# # SOUL Energy Scale: SEU <-> Joules
# ENERGY_SCALE_FACTOR: float = 1e14          # 1 SEU = 1e-14 Joules (Matches Synapse Energy)
# ENERGY_UNSCALE_FACTOR: float = 1e-14       # 1 Joule = 1e14 SEU (Inverse for reporting) # <<< ADD THIS BACK
# # BRAIN Energy Scale: BEU <-> Joules
# BRAIN_ENERGY_SCALE_FACTOR: float = 1e12    # 1 BEU = 1e-12 Joules (1 picoJoule) # <<< ENSURE THIS IS PRESENT
# BRAIN_ENERGY_UNIT_PER_JOULE: float = 1e12 

# # For clarity, let's define BEU to Joules:
# JOULES_PER_SEU: float = 1e-14 # 1 SEU = 1e-14 Joules (Matches Synapse Energy)
# SEU_PER_JOULE: float = 1e14   # 1 Joule = 1e14 SEU (Inverse)

# # BRAIN Energy Scale: BEU <-> Joules
# JOULES_PER_BEU: float = 1e-12 # 1 BEU = 1e-12 Joules (1 picoJoule)
# BEU_PER_JOULE: float = 1e12   # 1 Joule = 1e12 BEU

# # Derived Minimum Brain Energy (based on 1 synaptic firing per second for 14 days for an arbitrary number of synapses)
# # This defines a baseline operational energy requirement for a minimally viable brain.
# SYNAPSES_COUNT_FOR_MIN_ENERGY: int = int(1e9) # Assume 1 billion synapses for baseline calc
# SYNAPSE_ENERGY_JOULES: float = JOULES_PER_SEU # Energy of one synaptic firing in Joules (1 SEU's worth)
# SECONDS_IN_14_DAYS: float = 14 * 24 * 60 * 60
# ENERGY_BRAIN_14_DAYS_JOULES: float = SYNAPSE_ENERGY_JOULES * SYNAPSES_COUNT_FOR_MIN_ENERGY * SECONDS_IN_14_DAYS
# BRAIN_MIN_ENERGY_BEU: float = ENERGY_BRAIN_14_DAYS_JOULES * BEU_PER_JOULE # Minimum BEU for brain viability



# # Womb Environment Constants
# WOMB_ENERGY_ENHANCEMENT_FACTOR = 0.2
# WOMB_FREQUENCY_STABILIZATION_FACTOR = 0.15
# WOMB_GROWTH_ENHANCEMENT_FACTOR = 0.25
# WOMB_BRAIN_ENERGY_INTEGRATION_FACTOR = 0.1
# WOMB_BRAIN_RESONANCE_INTEGRATION_FACTOR = 0.05
# WOMB_BRAIN_PROTECTION_INTEGRATION_FACTOR = 0.08
# WOMB_ENERGY_CONNECTION_ENHANCEMENT = 0.3
# WOMB_RESONANCE_ENHANCEMENT = 0.2
# WOMB_MYCELIAL_ENHANCEMENT = 0.25
# WOMB_SOUL_STABILITY_BOOST = 5.0  # SU
# WOMB_BRAIN_STABILITY_BOOST = 0.1
# WOMB_LIFE_CORD_ENHANCEMENT = 0.15
# WOMB_MEMORY_VEIL_PROTECTION_FACTOR = 0.2
# WOMB_MEMORY_VEIL_NURTURING_FACTOR = 0.15
# WOMB_STRESS_THRESHOLD = 0.7
# WOMB_FINAL_ENERGY_BLESSING_FACTOR = 0.1
# WOMB_FINAL_STABILITY_BLESSING_FACTOR = 3.0  # SU
# WOMB_FINAL_COHERENCE_BLESSING_FACTOR = 3.0  # CU
# WOMB_FINAL_PROTECTION_BLESSING_FACTOR = 0.1



# # --- Metrics Tracking ---
# PERSIST_INTERVAL_SECONDS: int = 60

# # --- Visualization Defaults ---
# SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT: str = 'grey'; 
# SOUL_SPARK_VIZ_FREQ_SIG_MARKER_FMT: str = 'bo'
# SOUL_SPARK_VIZ_FREQ_SIG_BASE_FMT: str = 'r-'; 
# SOUL_SPARK_VIZ_FREQ_SIG_STEM_LW: float = 1.5
# SOUL_SPARK_VIZ_FREQ_SIG_MARKER_SZ: float = 5.0; 
# SOUL_SPARK_VIZ_FREQ_SIG_XLABEL: str = 'Frequency (Hz)'
# SOUL_SPARK_VIZ_FREQ_SIG_YLABEL: str = 'Amplitude'; 
# SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR: str = 'red'

# # --- Geometry Lists ---
# AVAILABLE_GEOMETRY_PATTERNS: List[str] = ["flower_of_life", "seed_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "egg_of_life", "fruit_of_life", "germ_of_life", "sri_yantra", "star_tetrahedron", "64_tetrahedron"]
# AVAILABLE_PLATONIC_SOLIDS: List[str] = ["tetrahedron", "hexahedron", "octahedron", "dodecahedron", "icosahedron", "sphere", "merkaba"]
# GEOMETRY_BASE_FREQUENCIES: Dict[str, float] = { 'point': 963.0, 'line': 852.0, 'triangle': 396.0, 'square': 285.0, 'pentagon': 417.0, 'hexagram': 528.0, 'heptagon': 741.0, 'octagon': 741.0, 'nonagon': 852.0, 'cross/cube': 174.0, 'vesica_piscis': 444.0, 'flower_of_life': 528.0, 'seed_of_life': 432.0, 'tree_of_life': 528.0, 'metatrons_cube': 639.0, 'merkaba': 741.0, 'vector_equilibrium': 639.0 }
# PLATONIC_HARMONIC_RATIOS: Dict[str, List[float]] = { 'tetrahedron': [1.0, 2.0, 3.0, 5.0], 'hexahedron': [1.0, 2.0, 4.0, 8.0], 'octahedron': [1.0, 1.5, 2.0, 3.0], 'dodecahedron': [1.0, PHI, 2.0, PHI*2, 3.0], 'icosahedron': [1.0, 1.5, 2.0, 2.5, 3.0], 'sphere': [1.0, 1.5, 2.0, 2.5, 3.0, PHI, 4.0, 5.0], 'merkaba': [1.0, 1.5, 2.0, 3.0, PHI, 4.0] }
# GEOMETRY_VOID_INFLUENCE_STRENGTH: float = 0.15
# PLATONIC_VOID_INFLUENCE_STRENGTH: float = 0.20

# # Energy conversion factors (natural proportional relationships)
# NATURAL_ENERGY_TO_HARMONY = 0.05   # Energy units needed for 1% harmony increase
# NATURAL_HARMONY_TO_ENERGY = 20.0   # Energy generated by 1% harmony improvement
# NATURAL_ENERGY_CYCLE_RATIO = 0.02  # % of energy used per cycle (sustainable rate)

# # Natural scaling factors
# NATURAL_EFFORT_SCALING = 4.0       # Exponential effort scaling for approaching perfection
# NATURAL_STAGNATION_THRESHOLD = 10  # Cycles of no change to detect equilibrium
# NATURAL_VIABILITY_THRESHOLD = 0.5  # Minimum capacity needed for further natural development

# # Memory fragment constants
# MEMORY_FRAGMENT_ENERGY_THRESHOLD = 0.05  # Minimum energy to activate memory fragment

# # Mycelial network constants  
# MYCELIAL_DEFAULT_DENSITY = 0.1  # Default mycelial density for pathways

# # Memory Veil Constants
# MEMORY_VEIL_BASE_STRENGTH = 0.8
# MEMORY_VEIL_MAX_STRENGTH = 0.95
# MEMORY_VEIL_FREQUENCY_LAYERS = 7
# MEMORY_VEIL_SOUND_DURATION = 10.0
# MEMORY_VEIL_SOUND_AMPLITUDE = 0.3

# # --- Mycelial Network Constants ---
# MYCELIAL_MAXIMUM_PATHWAY_LENGTH: int = 200  # Maximum distance for mycelial connections
# QUANTUM_ENTANGLEMENT_FREQUENCY: float = 432.0  # Frequency for quantum entanglement
# QUANTUM_SEEDS_PER_SUBREGION: int = 2         # Number of quantum seeds per subregion
# QUANTUM_EFFICIENCY: float = 0.98             # Quantum efficiency factor

# # --- BrainSeed & Nurturing Flow (Creator Energy Management) ---
# CREATOR_ENERGY_BASE_FREQUENCY_HZ: float = 432.0 # Default frequency of Creator Energy channeled by BrainSeed
# BRAIN_SEED_NURTURING_REQUEST_THRESHOLD_FACTOR: float = 0.2 # If BrainStructure total mycelial energy drops below 20% of its *estimated full operational capacity*, request more.
# BRAIN_SEED_NURTURING_ENERGY_INCREMENT_FACTOR: float = 0.1 # Request 10% of *estimated full operational capacity* when BrainSeed channels more energy.

# # --- Field Dynamics Constants ---
# FIELD_CHAOS_LEVEL_DEFAULT: float = 0.02 # Default 2% randomness/noise for biological realism in field modulations
# FIELD_STATE_CACHE_ENABLED: bool = True  # Enable caching of state-specific field modulation calculations
# FIELD_STATIC_GRID_SPACING: int = 8      # For DPI-like static electromagnetic foundation pattern (every Nth cell)
# FIELD_MAX_PROPAGATION_RADIUS_FACTOR: float = 0.6 # Max dynamic field propagation radius as factor of smallest brain dimension

# # Brain States for Field Modulation (extend as needed)
# BRAIN_STATE_DORMANT: str = "dormant"
# BRAIN_STATE_FORMATION: str = "brain_formation"
# BRAIN_STATE_AWARE_RESTING: str = "aware_resting"       # General resting state post-formation
# BRAIN_STATE_AWARE_PROCESSING: str = "aware_input_processing" # Actively processing external/internal stimuli
# BRAIN_STATE_ACTIVE_THOUGHT: str = "active_thought"     # Focused internal cognitive activity
# BRAIN_STATE_DREAMING: str = "dreaming"
# BRAIN_STATE_LIMINAL_TRANSITION: str = "liminal_transition"
# BRAIN_STATE_SOUL_ATTACHED_SETTLING: str = "soul_attached_settling" # State immediately after soul attachment

# # Safe Frequency Ranges for Fields (to avoid simulated interference with host system - values are illustrative)
# FIELD_STATIC_BASE_FREQUENCY_HZ_RANGE: Tuple[float, float] = (0.05, 1.0) # Very low, ultra-stable foundation
# FIELD_DEVELOPMENT_FREQUENCY_HZ_RANGE: Tuple[float, float] = (0.5, 10.0) # Delta, Theta, Low Alpha during growth
# FIELD_SOUL_RESONANCE_FREQUENCY_HZ_RANGE: Tuple[float, float] = (7.0, 15.0) # Theta, Alpha for soul interaction field
# FIELD_ACTIVE_THOUGHT_FREQUENCY_HZ_RANGE: Tuple[float, float] = (12.0, 35.0) # Beta, Low Gamma for active cognition
# FIELD_AVOID_FREQUENCY_RANGES_HZ: List[Tuple[float, float]] = [
#     (40.0, 70.0),    # Avoid typical AC power line frequencies and their harmonics
#     (700.0, 2700.0)  # Avoid common RF bands (cellular, WiFi, Bluetooth - broader range for safety)
# ]

# # --- Brain Development & Complexity ---
# SOUL_ATTACHMENT_COMPLEXITY_THRESHOLD: float = 0.85 # Overall complexity score (0-1) needed for soul attachment

# # Brain Region Definitions (ensure these are used consistently by brain_structure.py)
# REGION_FRONTAL: str = 'frontal'
# REGION_PARIETAL: str = 'parietal'
# REGION_TEMPORAL: str = 'temporal'
# REGION_OCCIPITAL: str = 'occipital'
# REGION_LIMBIC: str = 'limbic'
# REGION_BRAIN_STEM: str = 'brain_stem'
# REGION_CEREBELLUM: str = 'cerebellum'

# # Illustrative proportions and default resting state frequencies
# REGION_PROPORTIONS: Dict[str, float] = {
#     REGION_FRONTAL: 0.30, REGION_PARIETAL: 0.18, REGION_TEMPORAL: 0.20,
#     REGION_OCCIPITAL: 0.12, REGION_LIMBIC: 0.08, REGION_BRAIN_STEM: 0.02,
#     REGION_CEREBELLUM: 0.10
# } # Sums to 1.0
# REGION_DEFAULT_FREQUENCIES: Dict[str, float] = { # Resting state target frequencies for regions
#     REGION_FRONTAL: 18.0, REGION_PARIETAL: 10.0, REGION_TEMPORAL: 10.0, # Adjusted Temporal
#     REGION_OCCIPITAL: 12.0, REGION_LIMBIC: 6.0, REGION_BRAIN_STEM: 3.0,
#     REGION_CEREBELLUM: 8.0
# }
# BRAIN_WAVE_TYPES: Dict[str, Tuple[float, float]] = { # (min_hz, max_hz)
#     'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
#     'beta': (13.0, 30.0), 'gamma': (30.0, 100.0), 'lambda': (100.0, 200.0)
# }

# # --- Mycelial Network Constants ---
# MYCELIAL_DEFAULT_DENSITY: float = 0.1 # Base density for newly formed mycelial pathways
# MYCELIAL_MAXIMUM_PATHWAY_LENGTH: int = int(GRID_DIMENSIONS[0] * 0.80) # Max 80% of brain extent
# QUANTUM_ENTANGLEMENT_FREQUENCY: float = 432.0  # Hz for mycelial seed beacons (if different from Creator energy)
# QUANTUM_SEEDS_PER_SUBREGION_TARGET: int = 1 # Target 1 quantum seed node per subregion initially
# QUANTUM_EFFICIENCY: float = 0.98 # For quantum communication simulations

# # --- Memory System ---
# MEMORY_FRAGMENT_ENERGY_THRESHOLD: float = 0.05 # Min energy for a fragment to be considered substantial or activatable
# FLAG_SOUL_ATTACHED_TO_BRAIN = "soul_attached_to_brain" # Flag set on BrainGrid instance

# FIELD_SYSTEM_GRID_SIZE: Tuple[int, int, int] = (64, 64, 64) # Example if FieldController uses a grid
# VOID_BASE_ENERGY_SEU: float = 10.0

# BIRTH_WOMB_DEVELOPMENT_CYCLES = 3 # Number of abstract development 

# WOMB_ENERGY_ENHANCEMENT_FACTOR = 0.2 # How much womb *enhances soul's processing* of energy

# # --- START OF FILE stage_1/evolve/evolve_constants.py ---

# """
# Constants for Stage 1 Brain Evolution.

# This file defines constants used throughout the Stage 1 brain evolution process,
# including grid dimensions, region definitions, energy parameters, and frequencies.
# """

# import numpy as np
# from enum import Enum

# # --- General Constants ---
# FLOAT_EPSILON = 1e-9
# PHI = 1.618033988749895  # Golden ratio
# SPEED_OF_LIGHT = 299792458.0  # m/s

# # --- Brain Grid Constants (Simplified) ---
# GRID_DIMENSIONS = (256, 256, 256)  # 3D grid size
# DEFAULT_BLOCK_COUNT = 1000   # Approximate number of blocks to divide brain into

# # --- Region Names (Unchanged) ---
# REGION_FRONTAL = "frontal"
# REGION_PARIETAL = "parietal"
# REGION_TEMPORAL = "temporal"
# REGION_OCCIPITAL = "occipital"
# REGION_LIMBIC = "limbic"
# REGION_BRAIN_STEM = "brain_stem"
# REGION_CEREBELLUM = "cerebellum"

# # --- Region Proportions (Unchanged) ---
# REGION_PROPORTIONS = {
#     REGION_FRONTAL: 0.28,
#     REGION_PARIETAL: 0.15,
#     REGION_TEMPORAL: 0.17,
#     REGION_OCCIPITAL: 0.13,
#     REGION_LIMBIC: 0.10,
#     REGION_BRAIN_STEM: 0.03,
#     REGION_CEREBELLUM: 0.14
# }

# # --- Region Locations (Normalized coordinates) ---
# REGION_LOCATIONS = {
#     REGION_FRONTAL: (0.3, 0.7, 0.5),    # Front upper part
#     REGION_PARIETAL: (0.7, 0.7, 0.5),   # Rear upper part
#     REGION_TEMPORAL: (0.5, 0.4, 0.3),   # Side middle part
#     REGION_OCCIPITAL: (0.8, 0.5, 0.5),  # Rear part
#     REGION_LIMBIC: (0.5, 0.5, 0.4),     # Central part
#     REGION_BRAIN_STEM: (0.5, 0.3, 0.2), # Lower central part
#     REGION_CEREBELLUM: (0.7, 0.3, 0.3)  # Lower rear part
# }

# # --- Region Default Frequencies ---
# REGION_DEFAULT_FREQUENCIES = {
#     REGION_FRONTAL: 13.0,      # Primarily beta waves
#     REGION_PARIETAL: 10.0,     # Alpha/beta mix
#     REGION_TEMPORAL: 9.0,      # Primarily alpha waves
#     REGION_OCCIPITAL: 11.0,    # Alpha/beta mix
#     REGION_LIMBIC: 6.0,        # Theta waves
#     REGION_BRAIN_STEM: 4.0,    # Delta/theta mix
#     REGION_CEREBELLUM: 8.0     # Alpha waves
# }

# # --- Cell Density Parameters ---
# DEFAULT_CELL_DENSITY = 0.05  # Default 5% of cells are active
# BOUNDARY_CELL_DENSITY = 1.0  # 100% of cells are active at boundaries
# ACTIVE_CELL_THRESHOLD = 0.1  # Energy level to consider a cell active

# # --- Boundary Types ---
# BOUNDARY_TYPE_SHARP = "sharp"       # Clear delineation
# BOUNDARY_TYPE_GRADUAL = "gradual"   # Gradual transition
# BOUNDARY_TYPE_PERMEABLE = "permeable"  # Very permeable

# # --- Boundary Parameters ---
# BOUNDARY_PARAMETERS = {
#     BOUNDARY_TYPE_SHARP: {
#         "transition_width": 1,
#         "permeability": 0.3
#     },
#     BOUNDARY_TYPE_GRADUAL: {
#         "transition_width": 3,
#         "permeability": 0.6
#     },
#     BOUNDARY_TYPE_PERMEABLE: {
#         "transition_width": 5,
#         "permeability": 0.9
#     }
# }

# # --- Region Boundary Mappings ---
# REGION_BOUNDARIES = {
#     (REGION_FRONTAL, REGION_PARIETAL): BOUNDARY_TYPE_GRADUAL,
#     (REGION_PARIETAL, REGION_OCCIPITAL): BOUNDARY_TYPE_GRADUAL,
#     (REGION_TEMPORAL, REGION_FRONTAL): BOUNDARY_TYPE_GRADUAL,
#     (REGION_TEMPORAL, REGION_PARIETAL): BOUNDARY_TYPE_GRADUAL,
#     (REGION_LIMBIC, REGION_FRONTAL): BOUNDARY_TYPE_PERMEABLE,
#     (REGION_LIMBIC, REGION_TEMPORAL): BOUNDARY_TYPE_PERMEABLE,
#     (REGION_LIMBIC, REGION_PARIETAL): BOUNDARY_TYPE_PERMEABLE,
#     (REGION_LIMBIC, REGION_OCCIPITAL): BOUNDARY_TYPE_PERMEABLE,
#     (REGION_LIMBIC, REGION_BRAIN_STEM): BOUNDARY_TYPE_PERMEABLE,
#     (REGION_BRAIN_STEM, REGION_CEREBELLUM): BOUNDARY_TYPE_SHARP,
# }

# # --- Mycelial Network Constants (Simplified) ---
# DEFAULT_SEED_COUNT_PER_REGION = 3  # Default number of seeds per region
# MYCELIAL_QUANTUM_RANGE = 50        # Maximum distance for quantum entanglement
# MYCELIAL_ENERGY_EFFICIENCY = 0.95  # Energy transfer efficiency
# ENERGY_ROUTE_MAX_DISTANCE = 100    # Maximum distance for direct energy routes

# # --- Cell ID Prefixes ---
# PREFIX_BLOCK = "B"         # Block ID prefix
# PREFIX_CELL = "C"          # Regular cell ID prefix
# PREFIX_BOUNDARY = "BC"     # Boundary cell ID prefix
# PREFIX_MYCELIAL = "MS"     # Mycelial seed ID prefix
# PREFIX_ROUTE = "R"         # Route ID prefix

# # --- Memory Constants ---
# MEMORY_ACTIVATION_ENERGY = 0.1    # Energy required to activate a memory
# MEMORY_FREQUENCY_TOLERANCE = 1.0  # Default frequency tolerance for searches
# MEMORY_RESONANCE_THRESHOLD = 0.5  # Minimum resonance for memory activation

# # --- Sound Pattern Constants ---
# SOUND_SHARP_FREQUENCY = 15.0      # Base frequency for sharp boundaries
# SOUND_GRADUAL_FREQUENCY = 10.0    # Base frequency for gradual boundaries 
# SOUND_PERMEABLE_FREQUENCY = 7.0   # Base frequency for permeable boundaries
# SOUND_FREQUENCY_VARIATION = 1.0   # Random variation in boundary sounds

# # --- Energy Constants ---
# DEFAULT_SEED_ENERGY_CAPACITY = 50.0   # Base energy capacity for mycelial seeds
# DEFAULT_INITIAL_ENERGY_RATIO = 0.2    # Seeds start with 20% of capacity
# ENERGY_TRANSFER_QUANTUM_BONUS = 0.98  # Efficiency for quantum connections
# ENERGY_TRANSFER_LOSS_PER_DISTANCE = 0.005  # Energy loss per unit distance

# # --- Brain Seed Constants ---
# MIN_BRAIN_SEED_ENERGY = 1.0       # Minimum energy for brain seed operation
# DEFAULT_BRAIN_SEED_FREQUENCY = 7.83  # Schumann frequency default
# SEED_FIELD_RADIUS = 5.0           # Energy field radius around seed
# MYCELIAL_MAXIMUM_PATHWAY_LENGTH = 200  # Max distance for mycelial connections

# # --- Brain Complexity Thresholds ---
# BRAIN_COMPLEXITY_THRESHOLDS = {
#     'energy_coverage': 0.3,        # Min percentage of cells with energy > 0.1
#     'mycelial_coverage': 0.2,      # Min percentage of cells with mycelial density > 0.1
#     'avg_resonance': 0.3,          # Min average resonance across brain
#     'avg_energy': 0.2,             # Min average energy across brain
#     'field_initialized': True      # All fields must be initialized
# }

# # --- Consciousness Activation Thresholds ---
# CONSCIOUSNESS_ACTIVATION_THRESHOLDS = {
#     'dream': 0.4,                  # Brain complexity for dream state
#     'liminal': 0.6,                # Brain complexity for liminal transitions
#     'aware': 0.8                   # Brain complexity for aware state
# }

# # --- State Constants ---
# LIMINAL_BASE_FREQUENCY = 3.5      # Delta/theta boundary
# DREAM_BASE_FREQUENCY = 5.5        # Theta wave dominant
# AWARENESS_BASE_FREQUENCY = 9.0    # Alpha wave dominant
# STATE_TRANSITION_THRESHOLD = 0.6  # Percentage of cells needed for state transition

# # --- END OF FILE constants.py ---



























