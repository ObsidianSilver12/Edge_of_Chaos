# --- START OF constants.py ---

"""
Central Constants for the Soul Development Framework (Version 4.2 - Principle-Driven S/C Calc)

Consolidated and validated constants for simulation parameters, physics,
field properties (absolute units/potentials), soul defaults (new units),
stage thresholds/factors/flags (updated), sound generation,
mappings, geometry/glyphs, logging. Includes S/C calculation weights and influence rates.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any

# --- Global Simulation & Physics ---
SAMPLE_RATE: int = 44100
MAX_AMPLITUDE: float = 0.95
FLOAT_EPSILON: float = 1e-9
PI: float = np.pi
PLANCK_CONSTANT_H = 6.626e-34 # J*s
GOLDEN_RATIO: float = (1 + np.sqrt(5)) / 2.0 # Phi (~1.618)
PHI: float = GOLDEN_RATIO
SILVER_RATIO: float = 1 + np.sqrt(2)
EDGE_OF_CHAOS_RATIO: float = 1.0 / PHI
SCHUMANN_FREQUENCY: float = 7.83 # Hz
CORD_ACTIVATION_ENERGY_COST: float = 50.0 # SEU cost for activating a life cord
EDGE_OF_CHAOS_DEFAULT = 0.618 # Golden ratio - optimal edge of chaos parameter
# --- Paths & Logging ---
DATA_DIR_BASE: str = "output"
LOG_LEVEL = logging.INFO # Use INFO, but DEBUG in specific files for tracing
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
OUTPUT_DIR_BASE: str = "output"
SPEED_OF_LIGHT: float = 299792458.0  # Speed of light in m/s
ANCHOR_RESONANCE_MODIFIER: float = 0.85  # Modifier for anchor resonance calculations
ANCHOR_STRENGTH_MODIFIER: float = 0.6  # Matches existing constant on line 359
AIR_DENSITY: float = 1.225  # Air density in kg/m³ at sea level
# --- Energy Units (Biomimetic Scaled Joules) ---

# --- Soul Core Units & Ranges ---
INITIAL_SPARK_BASE_FREQUENCY_HZ: float = 432.0
INITIAL_SPARK_ENERGY_SEU: float = 500.0
MAX_SOUL_ENERGY_SEU: float = 1e6
PASSIVE_ENERGY_DISSIPATION_RATE_SEU_PER_SEC: float = 0.1

# --- Spark Emergence & Harmonization Constants ---
SPARK_EOC_ENERGY_YIELD_FACTOR: float = 50.0   # TUNE: How much EoC (0-1) multiplies base potential for initial energy.
SPARK_FIELD_ENERGY_CATALYST_FACTOR: float = 0.05 # TUNE: Fraction of local void energy added as catalyst.
SPARK_SEED_GEOMETRY: str = 'sphere'           # TUNE: Which platonic harmonic ratios seed the spark ('sphere', 'seed_of_life', etc.)
SPARK_INITIAL_FACTOR_EOC_SCALE: float = 0.4   # TUNE: How much EoC (0-1) boosts initial phi_resonance, pattern_coherence etc.
SPARK_INITIAL_FACTOR_ORDER_SCALE: float = 0.2 # TUNE: How much local void order (0-1) boosts initial factors.
SPARK_INITIAL_FACTOR_PATTERN_SCALE: float = 0.1# TUNE: How much local void pattern (0-1) boosts initial factors.
SPARK_INITIAL_FACTOR_BASE: float = 0.1        # TUNE: Minimum base value for initial factors like phi_resonance.
SEPHIROTH_FREQ_NUDGE_FACTOR = 0.04
HARMONIZATION_ITERATIONS: int = 244             # TUNE: Number of internal harmonization steps.
HARMONIZATION_PATTERN_COHERENCE_RATE: float = 0.003 # TUNE: Rate factor builds towards 1.0 during harmonization.
HARMONIZATION_PHI_RESONANCE_RATE: float = 0.002   # TUNE: Rate factor builds towards 1.0 during harmonization.
HARMONIZATION_HARMONY_RATE: float = 0.0015        # TUNE: Rate factor builds towards 1.0 during harmonization.
HARMONIZATION_ENERGY_GAIN_RATE: float = 0.25      # TUNE: SEU gain per iteration scaled by internal order proxy.
HARMONIZATION_PHASE_ADJUST_RATE = 0.01 #(Small adjustment per iteration)
HARMONIZATION_HARMONIC_ADJUST_RATE = 0.005 #(Even smaller adjustment)
HARMONIZATION_CIRC_VAR_THRESHOLD = 0.15 #(Only adjust if variance is > 15%)
HARMONIZATION_HARM_DEV_THRESHOLD = 0.08 #(Only adjust if average deviation > 8%)
HARMONIZATION_TORUS_RATE: float = 0.002 # TUNE: Rate factor builds towards 1.0 during harmonization.

PLATONIC_HARMONIC_RATIOS: Dict[str, List[float]] = {
    'tetrahedron': [1.0, 2.0, 3.0, 5.0],
    'hexahedron': [1.0, 2.0, 4.0, 8.0],
    'octahedron': [1.0, 1.5, 2.0, 3.0],
    'dodecahedron': [1.0, PHI, 2.0, PHI*2, 3.0],
    'icosahedron': [1.0, 1.5, 2.0, 2.5, 3.0],
    'sphere': [1.0, 1.5, 2.0, 2.5, 3.0, PHI, 4.0, 5.0], # Default Seed?
    'merkaba': [1.0, 1.5, 2.0, 3.0, PHI, 4.0]
}

# --- Stability & Coherence Units & Ranges ---
MAX_STABILITY_SU: float = 100.0
MAX_COHERENCE_CU: float = 100.0
INITIAL_STABILITY_CALC_FACTOR: float = 0.2 # Initial scale applied to calculated S
INITIAL_COHERENCE_CALC_FACTOR: float = 0.15 # Initial scale applied to calculated C

# --- NEW: Stability Score Calculation Weights (Sum should ideally be 1.0) ---
STABILITY_WEIGHT_FREQ: float = 0.30     # Contribution from frequency stability
STABILITY_WEIGHT_PATTERN: float = 0.35   # Contribution from internal structure (layers, aspects, patterns)
STABILITY_WEIGHT_FIELD: float = 0.20       # Contribution from external field influences (Guff, Sephiroth)
STABILITY_WEIGHT_TORUS: float   = 0.15
# Factors used WITHIN the pattern component calculation
STABILITY_PATTERN_WEIGHT_LAYERS: float = 0.3
STABILITY_PATTERN_WEIGHT_ASPECTS: float = 0.3
STABILITY_PATTERN_WEIGHT_PHI: float = 0.2
STABILITY_PATTERN_WEIGHT_ALIGNMENT: float = 0.2

# --- Harmonic Strengthening (HS) - Targeted Refinement Cycle Constants (NEW for V4.3.8+) ---

# Thresholds to trigger refinement for specific aspects (Scores 0-1, SU/CU 0-100)
HS_TRIGGER_STABILITY_SU: float = 95.0     # If Stability is below this, HS might target it
HS_TRIGGER_COHERENCE_CU: float = 95.0     # If Coherence is below this, HS might target it
HS_TRIGGER_PHASE_COHERENCE: float = 0.90    # Target for (1.0 - Circular Variance)
HS_TRIGGER_HARMONIC_PURITY: float = 0.90    # Target for (1.0 - Harmonic Deviation)
HS_TRIGGER_FACTOR_THRESHOLD: float = 0.95   # Target for Phi, P.Coh, Harmony, Torus factors (0-1)

# Cycle Control
HS_MAX_CYCLES: int =  144                # Maximum refinement iterations if thresholds not met
HS_UPDATE_STATE_INTERVAL: int = 10         # How many cycles between S/C recalculations via update_state()

# Base rates for factor improvements per cycle (modulated by current state)
HS_BASE_RATE_PHI: float = 0.003             # Base rate for Phi Resonance improvement
HS_BASE_RATE_PCOH: float = 0.004            # Base rate for Pattern Coherence improvement
HS_BASE_RATE_HARMONY: float = 0.005         # Base rate for Harmony improvement
HS_BASE_RATE_TORUS: float = 0.006           # Base rate for Toroidal Flow improvement

# Adjustment rates for frequency structure refinement per cycle
HS_PHASE_ADJUST_RATE: float = 0.005         # Max adjustment factor applied during phase optimization nudge
HS_HARMONIC_ADJUST_RATE: float = 0.002       # Max adjustment factor applied during harmonic purity nudge

# Energy adjustment based on harmony changes during HS
HS_ENERGY_ADJUST_FACTOR: float = 5.0        # Scales SEU gain/loss based on delta_harmony in a cycle


# --- Creator Entanglement (CE) - Resonance & Connection Constants
CE_RESONANCE_THRESHOLD: float = 0.75    # Min resonance score (0-1) considered 'strong' for certain effects (e.g., pattern modulation?)
CE_CONNECTION_FREQ_WEIGHT: float = 0.6    # Weight of frequency resonance score in connection strength calculation
CE_CONNECTION_COHERENCE_WEIGHT: float = 0.4 # Weight of soul's coherence (normalized) in connection strength calculation
CE_PATTERN_MODULATION_STRENGTH: float = 0.05 # How strongly Kether's geometry influences soul's pattern coherence during connection
CREATOR_POTENTIAL_DEFAULT = 0.7  # Default creator potential for entanglement
ENTANGLEMENT_PREREQ_STABILITY_MIN_SU = 30.0  # Minimum stability required for entanglement
ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU = 25.0  # Minimum coherence required for entanglement
SPEED_OF_SOUND = 343.0  # Speed of sound in m/s, used for wavelength calculations
ASPECT_TRANSFER_THRESHOLD = 0.3  # Minimum resonance * connection for aspect transfer
ASPECT_TRANSFER_STRENGTH_FACTOR = 0.15  # Strength increase factor for existing aspects
ASPECT_TRANSFER_INITIAL_STRENGTH = 0.4  # Initial strength for newly transferred aspects

# --- NEW: Coherence Score Calculation Weights (Sum should ideally be 1.0) ---
COHERENCE_WEIGHT_PHASE: float = 0.20      # Contribution from phase alignment
COHERENCE_WEIGHT_HARMONY: float = 0.10   # Contribution from harmonic purity/alignment
COHERENCE_WEIGHT_PATTERN: float = 0.25    # Contribution from pattern_coherence factor
COHERENCE_WEIGHT_FIELD: float = 0.10     # Contribution from external field influences
COHERENCE_WEIGHT_CREATOR: float = 0.10  # Contribution from creator connection strength
COHERENCE_WEIGHT_TORUS = 0.25  # Contribution from toroidal coherence

# Other Calculation Factors
STABILITY_VARIANCE_PENALTY_K: float = 50.0 # How much freq variance hurts stability

# --- Field System ---
GRID_SIZE: Tuple[int, int, int] = (64, 64, 64)
VOID_BASE_ENERGY_SEU: float = 10.0
VOID_BASE_FREQUENCY_RANGE: Tuple[float, float] = (10.0, 1000.0)
VOID_BASE_STABILITY_SU: float = 20.0 # Baseline SU towards which Void drifts
VOID_BASE_COHERENCE_CU: float = 15.0 # Baseline CU towards which Void drifts
VOID_CHAOS_ORDER_BALANCE: float = 0.5
SEPHIROTH_DEFAULT_RADIUS: float = 8.0
SEPHIROTH_INFLUENCE_FALLOFF: float = 1.5
DEFAULT_PHI_HARMONIC_COUNT: int = 3
GUFF_RADIUS_FACTOR: float = 0.3
GUFF_CAPACITY: int = 100

# --- Sephiroth Absolute Potentials ---
SEPHIROTH_ENERGY_POTENTIALS_SEU: Dict[str, float] = { 'kether': MAX_SOUL_ENERGY_SEU * 0.95, 'chokmah': MAX_SOUL_ENERGY_SEU * 0.85, 'binah': MAX_SOUL_ENERGY_SEU * 0.80, 'daath': MAX_SOUL_ENERGY_SEU * 0.70, 'chesed': MAX_SOUL_ENERGY_SEU * 0.75, 'geburah': MAX_SOUL_ENERGY_SEU * 0.65, 'tiphareth': MAX_SOUL_ENERGY_SEU * 0.70, 'netzach': MAX_SOUL_ENERGY_SEU * 0.60, 'hod': MAX_SOUL_ENERGY_SEU * 0.55, 'yesod': MAX_SOUL_ENERGY_SEU * 0.45, 'malkuth': MAX_SOUL_ENERGY_SEU * 0.30 }
SEPHIROTH_TARGET_STABILITY_SU: Dict[str, float] = { 'kether': 98.0, 'chokmah': 90.0, 'binah': 92.0, 'daath': 85.0, 'chesed': 88.0, 'geburah': 80.0, 'tiphareth': 95.0, 'netzach': 85.0, 'hod': 82.0, 'yesod': 88.0, 'malkuth': 75.0 }
SEPHIROTH_TARGET_COHERENCE_CU: Dict[str, float] = { 'kether': 98.0, 'chokmah': 92.0, 'binah': 90.0, 'daath': 88.0, 'chesed': 90.0, 'geburah': 82.0, 'tiphareth': 95.0, 'netzach': 88.0, 'hod': 85.0, 'yesod': 90.0, 'malkuth': 70.0 }
GUFF_TARGET_ENERGY_SEU: float = SEPHIROTH_ENERGY_POTENTIALS_SEU['kether'] * 0.9
GUFF_TARGET_STABILITY_SU: float = SEPHIROTH_TARGET_STABILITY_SU['kether'] * 0.95
GUFF_TARGET_COHERENCE_CU: float = SEPHIROTH_TARGET_COHERENCE_CU['kether'] * 0.95
KETHER_FREQ: float = 963.0 # Example Kether base freq for Guff resonance calc

# --- Transfer & Influence Rates ---
ENERGY_TRANSFER_RATE_K: float = 0.05           # Base rate for SEU transfer
GUFF_ENERGY_TRANSFER_RATE_K: float = 0.2       # SEU transfer rate in Guff
SEPHIROTH_ENERGY_EXCHANGE_RATE_K: float = 0.05 # SEU exchange rate during Sephirah interaction
# --- NEW: Influence Factor Rates ---
GUFF_INFLUENCE_RATE_K: float = 0.05  # *** TUNE: How much each Guff step increments guff_influence_factor (0-1)
SEPHIRAH_INFLUENCE_RATE_K: float = 0.15 # *** TUNE: How much each Sephirah interaction increments cumulative_sephiroth_influence (0-1)

# --- Resonance & Aspects ---
RESONANCE_INTEGER_RATIO_TOLERANCE: float = 0.02
RESONANCE_PHI_RATIO_TOLERANCE: float = 0.03
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ: float = 0.5
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM: float = 0.3
SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI: float = 0.2
SEPHIROTH_ASPECT_TRANSFER_FACTOR: float = 0.2
MAX_ASPECT_STRENGTH: float = 1.0

# --- Geometric & Platonic Constants ---
PLATONIC_BASE_FREQUENCIES: Dict[str, float] = {'tetrahedron': 396.0, 'hexahedron': 285.0, 'octahedron': 639.0, 'dodecahedron': 963.0, 'icosahedron': 369.0, 'sphere': 432.0, 'merkaba': 528.0 }
GEOMETRY_EFFECTS: Dict[str, Dict[str, float]] = {'tetrahedron': {'energy_focus': 0.1, 'transformative_capacity': 0.07}, 'hexahedron': {'stability_factor_boost': 0.15, 'grounding': 0.12, 'energy_containment': 0.08}, 'octahedron': {'yin_yang_balance_push': 0.0, 'coherence_factor_boost': 0.1, 'stability_factor_boost': 0.05}, 'dodecahedron': {'unity_connection': 0.15, 'phi_resonance_boost': 0.12, 'transcendence': 0.1}, 'icosahedron': {'emotional_flow': 0.12, 'adaptability': 0.1, 'coherence_factor_boost': 0.08}, 'sphere': {'potential_realization': 0.1, 'unity_connection': 0.05}, 'merkaba': {'stability_factor_boost': 0.1, 'transformative_capacity': 0.12, 'field_resilience': 0.08}, 'flower_of_life': {'harmony_boost': 0.12, 'structural_integration': 0.1}, 'seed_of_life': {'potential_realization': 0.1, 'stability_factor_boost': 0.08}, 'vesica_piscis': {'yin_yang_balance_push': 0.0, 'connection_boost': 0.09}, 'tree_of_life': {'harmony_boost': 0.1, 'structural_integration': 0.1, 'connection_boost': 0.08}, 'metatrons_cube': {'structural_integration': 0.12, 'connection_boost': 0.12}, 'vector_equilibrium': {'yin_yang_balance_push': 0.0, 'zero_point_attunement': 0.15}, '64_tetrahedron': {'structural_integration': 0.15, 'energy_containment': 0.1} }
DEFAULT_GEOMETRY_EFFECT: Dict[str, float] = {'stability_factor_boost': 0.01}
SEPHIROTH_GLYPH_DATA: Dict[str, Dict[str, Any]] = {'kether': { 'platonic': 'dodecahedron', 'sigil': 'Point/Crown', 'gematria_keys': ['Kether', 'Crown', 'Will', 'Unity', 1], 'fibonacci': [1, 1] }, 'chokmah': { 'platonic': 'sphere', 'sigil': 'Line/Wheel', 'gematria_keys': ['Chokmah', 'Wisdom', 'Father', 2], 'fibonacci': [2] }, 'binah': { 'platonic': 'icosahedron', 'sigil': 'Triangle/Womb', 'gematria_keys': ['Binah', 'Understanding', 'Mother', 3], 'fibonacci': [3] }, 'chesed': { 'platonic': 'hexahedron', 'sigil': 'Square/Solid', 'gematria_keys': ['Chesed', 'Mercy', 'Grace', 4], 'fibonacci': [5] }, 'geburah': { 'platonic': 'tetrahedron', 'sigil': 'Pentagon/Sword', 'gematria_keys': ['Geburah', 'Severity', 'Strength', 5], 'fibonacci': [8] }, 'tiphareth': { 'platonic': 'octahedron', 'sigil': 'Hexagram/Sun', 'gematria_keys': ['Tiphareth', 'Beauty', 'Harmony', 6], 'fibonacci': [13] }, 'netzach': { 'platonic': 'icosahedron', 'sigil': 'Heptagon/Victory', 'gematria_keys': ['Netzach', 'Victory', 'Endurance', 7], 'fibonacci': [21] }, 'hod': { 'platonic': 'octahedron', 'sigil': 'Octagon/Splendor', 'gematria_keys': ['Hod', 'Splendor', 'Glory', 8], 'fibonacci': [34] }, 'yesod': { 'platonic': 'icosahedron', 'sigil': 'Nonagon/Foundation', 'gematria_keys': ['Yesod', 'Foundation', 'Moon', 9], 'fibonacci': [55] }, 'malkuth': { 'platonic': 'hexahedron', 'sigil': 'CrossInCircle/Kingdom', 'gematria_keys': ['Malkuth', 'Kingdom', 'Shekhinah', 'Earth', 10], 'fibonacci': [89] }, 'daath': { 'platonic': 'sphere', 'sigil': 'VoidPoint', 'gematria_keys': ['Daath', 'Knowledge', 'Abyss', 11], 'fibonacci': [] }, }

# --- Physics / Field Dynamics ---
HARMONIC_RESONANCE_ENERGY_BOOST: float = 0.012 # Influence factor on energy in VoidField resonance calc
WAVE_PROPAGATION_SPEED: float = 0.2           # Rate for VoidField energy propagation
ENERGY_DISSIPATION_RATE = 0.002              # Rate for VoidField energy dissipation

# --- Stage Prerequisites (Using SU/CU) ---
ENTANGLEMENT_PREREQ_STABILITY_MIN_SU: float = 75.0; 
ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU: float = 75.0
HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU: float = 70.0; 
HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU: float = 70.0
CORD_STABILITY_THRESHOLD_SU: float = 80.0; 
CORD_COHERENCE_THRESHOLD_CU: float = 80.0
HARMONY_PREREQ_CORD_INTEGRITY_MIN: float = 0.70; 
HARMONY_PREREQ_STABILITY_MIN_SU: float = 75.0; 
HARMONY_PREREQ_COHERENCE_MIN_CU: float = 75.0
IDENTITY_STABILITY_THRESHOLD_SU: float = 85.0; 
IDENTITY_COHERENCE_THRESHOLD_CU: float = 85.0
IDENTITY_EARTH_RESONANCE_THRESHOLD: float = 0.75; 
IDENTITY_CRYSTALLIZATION_THRESHOLD: float = 0.85
BIRTH_PREREQ_CORD_INTEGRITY_MIN: float = 0.80; 
BIRTH_PREREQ_EARTH_RESONANCE_MIN: float = 0.75

# --- Readiness & Completion Flags ---
FLAG_READY_FOR_GUFF="ready_for_guff";
FLAG_GUFF_STRENGTHENED="guff_strengthened"; 
FLAG_READY_FOR_JOURNEY="ready_for_journey"; 
FLAG_SEPHIROTH_JOURNEY_COMPLETE="sephiroth_journey_complete"; 
FLAG_READY_FOR_ENTANGLEMENT="ready_for_entanglement"; 
FLAG_READY_FOR_COMPLETION="ready_for_completion"; 
FLAG_CREATOR_ENTANGLED = "creator_entangled"  # Flag indicating successful creator entanglement
FLAG_READY_FOR_HARMONIZATION = "ready_for_harmonization"  # Flag indicating readiness for harmonization
FLAG_READY_FOR_STRENGTHENING="ready_for_strengthening"; 
FLAG_HARMONICALLY_STRENGTHENED="harmonically_strengthened"; 
FLAG_READY_FOR_LIFE_CORD="ready_for_life_cord"; 
FLAG_CORD_FORMATION_COMPLETE="cord_formation_complete"; 
FLAG_READY_FOR_EARTH="ready_for_earth"; 
FLAG_EARTH_HARMONIZED="earth_harmonized"; 
FLAG_EARTH_ATTUNED = "earth_attuned"; # From Earth Harmony
FLAG_READY_FOR_IDENTITY="ready_for_identity"; 
FLAG_IDENTITY_CRYSTALLIZED="identity_crystallized"; 
FLAG_READY_FOR_BIRTH="ready_for_birth"; 
FLAG_INCARNATED="incarnated"
FLAG_ECHO_PROJECTED = "echo_projected"; 
FLAG_EARTH_ATTUNED = "earth_attuned" # New Flags

# --- Other Stage Parameters (Factors & Defaults - REVIEW those affecting SU/CU) ---
# Constants for later stages are kept here, but the stage logic may need adjustment
# to modify influence factors instead of direct SU/CU boosts if we want full consistency.
GUFF_STRENGTHENING_DURATION: float = 10.0
SEPHIROTH_JOURNEY_ATTRIBUTE_IMPART_FACTOR: float = 0.05
SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR: float = 0.04
ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR: float = 0.15 # OK
ENTANGLEMENT_STABILITY_BOOST_FACTOR: float = 0.1 # *** REVIEW: Scales SU gain? ***
ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE: float = 0.6; 
ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE: float = 0.4
ENTANGLEMENT_RESONANCE_BOOST_FACTOR: float = 0.05 # OK
ENTANGLEMENT_STABILIZATION_ITERATIONS: int = 5
ENTANGLEMENT_STABILIZATION_FACTOR_STRENGTH: float = 1.005 # OK
HARMONIC_STRENGTHENING_INTENSITY_DEFAULT: float = 0.7; 
HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT: float = 1.0
FUNDAMENTAL_FREQUENCY_432: float = 432.0
SOLFEGGIO_FREQUENCIES: Dict[str, float] = { 'UT': 396.0, 'RE': 417.0, 'MI': 528.0, 'FA': 639.0, 'SOL': 741.0, 'LA': 852.0, 'SI': 963.0 }
HARMONIC_STRENGTHENING_TARGET_FREQS: List[float] = sorted(list(SOLFEGGIO_FREQUENCIES.values()) + [FUNDAMENTAL_FREQUENCY_432])
HARMONIC_STRENGTHENING_TUNING_INTENSITY_FACTOR: float = 0.1; 
HARMONIC_STRENGTHENING_TUNING_TARGET_REACH_HZ: float = 1.0
HARMONIC_STRENGTHENING_PHI_AMP_INTENSITY_FACTOR: float = 0.05 # OK
HARMONIC_STRENGTHENING_PHI_STABILITY_BOOST_FACTOR: float = 0.3 # *** REVIEW: Scales SU gain? ***
HARMONIC_STRENGTHENING_PATTERN_STAB_INTENSITY_FACTOR: float = 0.04 # OK
HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_FACTOR: float = 0.01; 
HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_CAP: float = 0.1
HARMONIC_STRENGTHENING_PATTERN_STAB_STABILITY_BOOST: float = 0.4 # *** REVIEW: Scales SU gain? ***
HARMONIC_STRENGTHENING_COHERENCE_INTENSITY_FACTOR: float = 0.08 # *** REVIEW: Scales CU gain? ***
HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_COUNT_NORM: float = 10.0
HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_FACTOR: float = 0.03 # *** REVIEW: Scales CU gain? ***
HARMONIC_STRENGTHENING_COHERENCE_HARMONY_BOOST: float = 0.25 # OK
HARMONIC_STRENGTHENING_EXPANSION_INTENSITY_FACTOR: float = 0.1 # OK
HARMONIC_STRENGTHENING_EXPANSION_STATE_FACTOR: float = 1.0
HARMONIC_STRENGTHENING_EXPANSION_STR_INTENSITY_FACTOR: float = 0.03 # OK
HARMONIC_STRENGTHENING_EXPANSION_STR_STATE_FACTOR: float = 0.5
HARMONIC_STRENGTHENING_HARMONIC_COUNT: int = 7
LIFE_CORD_COMPLEXITY_DEFAULT: float = 0.7
PRIMARY_CHANNEL_BANDWIDTH_FACTOR: float = 200.0; 
PRIMARY_CHANNEL_STABILITY_FACTOR_CONN: float = 0.7; 
PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX: float = 0.3
PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN: float = 0.6; 
PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX: float = 0.4
PRIMARY_CHANNEL_ELASTICITY_BASE: float = 0.5; 
PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX: float = 0.3
HARMONIC_NODE_COUNT_BASE: int = 3; 
HARMONIC_NODE_COUNT_FACTOR: float = 15.0
HARMONIC_NODE_AMP_BASE: float = 0.4; 
HARMONIC_NODE_AMP_FACTOR_COMPLEX: float = 0.4; 
HARMONIC_NODE_AMP_FALLOFF: float = 0.6
HARMONIC_NODE_BW_INCREASE_FACTOR: float = 5.0
MAX_CORD_CHANNELS: int = 7; 
SECONDARY_CHANNEL_COUNT_FACTOR: float = 6.0; 
SECONDARY_CHANNEL_FREQ_FACTOR: float = 0.1
SECONDARY_CHANNEL_BW_EMOTIONAL: tuple[float,float]=(10.0,30.0); 
SECONDARY_CHANNEL_RESIST_EMOTIONAL: tuple[float,float]=(0.4,0.3)
SECONDARY_CHANNEL_BW_MENTAL: tuple[float,float]=(15.0,40.0); 
SECONDARY_CHANNEL_RESIST_MENTAL: tuple[float,float]=(0.5,0.3)
SECONDARY_CHANNEL_BW_SPIRITUAL: tuple[float,float]=(20.0,50.0); 
SECONDARY_CHANNEL_RESIST_SPIRITUAL: tuple[float,float]=(0.6,0.3)
FIELD_INTEGRATION_FACTOR_FIELD_STR: float = 0.6; 
FIELD_INTEGRATION_FACTOR_CONN_STR: float = 0.4
FIELD_EXPANSION_FACTOR: float = 1.05
CORD_INTEGRITY_FACTOR_CONN_STR: float = 0.4; 
CORD_INTEGRITY_FACTOR_STABILITY: float = 0.3; 
CORD_INTEGRITY_FACTOR_EARTH_CONN: float = 0.3
FINAL_STABILITY_BONUS_FACTOR: float = 0.15 # *** REVIEW: Scales SU bonus? ***


#EEarth Harmonisation (Earth Resonance) Constants
EARTH_ANCHOR_RESONANCE: float = 0.9
ANCHOR_STRENGTH_MODIFIER: float = 0.6; 
EARTH_ANCHOR_STRENGTH: float = 0.9; 
EARTH_CYCLE_FREQ_SCALING: float = 1.0e9
ECHO_FIELD_STRENGTH_FACTOR: float = 0.85
HARMONY_SCHUMANN_INTENSITY: float = 0.6
HARMONY_CORE_INTENSITY: float = 0.4
EARTH_FREQUENCY = 136.10; 
EARTH_BREATH_FREQUENCY = 0.2
EARTH_CONN_FACTOR_CONN_STR: float = 0.5; 
EARTH_CONN_FACTOR_ELASTICITY: float = 0.3; 
EARTH_CONN_BASE_FACTOR: float = 0.1
EARTH_HARMONY_INTENSITY_DEFAULT: float = 0.7; 
EARTH_HARMONY_DURATION_FACTOR_DEFAULT: float = 1.0
EARTH_FREQUENCIES: Dict[str, float] = { "schumann": 7.83, "geomagnetic": 11.75, "core_resonance": EARTH_FREQUENCY, "breath_cycle": EARTH_BREATH_FREQUENCY, "heartbeat_cycle": 1.2, "circadian_cycle": 1.0/(24*3600)}
EARTH_ELEMENTS: List[str] = ["earth", "water", "fire", "air", "aether"]

# Surface resonance constants
SURFACE_RESONANCE_FACTOR = 0.5
SURFACE_RESONANCE_HARMONIC_THRESHOLD = 0.3

# Element colors for layer visualization
ELEMENT_COLORS = {
    'earth': '#8B4513',
    'air': '#87CEEB', 
    'fire': '#FF4500',
    'water': '#1E90FF',
    'aether': '#E6E6FA'
}

# --- Earth Harmonization (Echo Attunement) Constants (NEW V4.3.8+) ---
ECHO_PROJECTION_ENERGY_COST_FACTOR: float = 0.02 # % of spiritual energy cost
ECHO_ATTUNEMENT_CYCLES: int = 84        # Number of attunement cycles
ECHO_ATTUNEMENT_RATE: float = 0.8       # Base rate of state shift per cycle (modulated by resonance)
ECHO_FIELD_COHERENCE_FACTOR: float = 0.75 # Coherence factor for echo field
ATTUNEMENT_RESONANCE_THRESHOLD: float = 0.3   # Min resonance for attunement shift to occur
ELEMENTAL_TARGET_EARTH: float = 0.8 # Target alignment for primary earth element
ELEMENTAL_TARGET_OTHER: float = 0.4 # Target alignment for other elements
ELEMENTAL_ALIGN_INTENSITY_FACTOR: float = 0.2 # Rate for elemental alignment nudge
PLANETARY_FREQUENCIES: Dict[str, float] = { # Example Frequencies (Hz) - NEEDS REAL VALUES/SOURCE
    "Sun": 126.22, "Moon": 210.42, "Mercury": 141.27, "Venus": 221.23, "Earth": 194.71,
    "Mars": 144.72, "Jupiter": 183.58, "Saturn": 147.85, "Uranus": 207.36,
    "Neptune": 211.44, "Pluto": 140.25 }
PLANETARY_ATTUNEMENT_CYCLES: int = 21 # Cycles for planetary resonance step
PLANETARY_RESONANCE_RATE: float = 0.0095 # Rate for planetary resonance factor gain
GAIA_CONNECTION_CYCLES: int = 13 # Cycles for Gaia connection step
GAIA_CONNECTION_FACTOR: float = 0.023 # Rate for Gaia connection factor gain
STRESS_FEEDBACK_FACTOR: float = 0.0024 # How much echo discordance impacts main soul stability variance
HARMONY_CYCLE_NAMES: List[str] = ["circadian", "heartbeat", "breath"]; 
HARMONY_CYCLE_IMPORTANCE: Dict[str, float] = {"circadian": 0.6, "heartbeat": 0.8, "breath": 1.0}
HARMONY_CYCLE_SYNC_TARGET_BASE: float = 0.9; 
HARMONY_CYCLE_SYNC_INTENSITY_FACTOR: float = 0.1; 
HARMONY_CYCLE_SYNC_DURATION_FACTOR: float = 1.0
HARMONY_PLANETARY_RESONANCE_TARGET: float = 0.85; 
HARMONY_PLANETARY_RESONANCE_FACTOR: float = 0.15
HARMONY_GAIA_CONNECTION_TARGET: float = 0.90; 
HARMONY_GAIA_CONNECTION_FACTOR: float = 0.2
HARMONY_FINAL_STABILITY_BONUS: float = 0.08 # *** REVIEW: Scales SU bonus? ***
HARMONY_FINAL_COHERENCE_BONUS: float = 0.08 # *** REVIEW: Scales CU bonus? ***
HARMONY_CYCLE_SYNC_TARGET_BASE: float = 0.9
HARMONY_CYCLE_SYNC_INTENSITY_FACTOR: float = 0.1
HARMONY_FREQ_TUNING_FACTOR: float = 0.15 # Re-add if needed by Earth Harmony freq attunement
HARMONY_FREQ_TUNING_TARGET_REACH_HZ: float = 1.0; 
HARMONY_FREQ_UPDATE_HARMONIC_COUNT: int = 5
HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT: float = 0.6; 
HARMONY_FREQ_TARGET_SOUL_WEIGHT: float = 0.2; 
HARMONY_FREQ_TARGET_CORE_WEIGHT: float = 0.2
HARMONY_FREQ_RES_WEIGHT_SCHUMANN: float = 0.7; 
HARMONY_FREQ_RES_WEIGHT_OTHER: float = 0.3
HARMONY_ELEM_RES_WEIGHT_PRIMARY: float = 0.6; 
HARMONY_ELEM_RES_WEIGHT_AVERAGE: float = 0.4



# --- Identity Crystallization (Astrology) Constants (NEW V4.3.8+) ---
ZODIAC_SIGNS: List[Dict[str, str]] = [ # Using 13 signs format
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

ZODIAC_TRAITS: Dict[str, Dict[str, List[str]]] = { # Provided trait lists
    "Aries": {"positive": ["Courageous","Energetic","Adventurous","Enthusiastic","Confident","Dynamic","Independent","Pioneering","Ambitious","Passionate"], "negative": ["Impulsive","Impatient","Aggressive","Reckless","Self-centered","Stubborn","Short-tempered","Competitive","Blunt","Domineering"]},
    "Taurus": {"positive": ["Reliable","Patient","Practical","Devoted","Persistent","Loyal","Stable","Loving","Determined","Sensual"], "negative": ["Possessive","Stubborn","Materialistic","Self-indulgent","Inflexible","Jealous","Resistant to change","Greedy","Lazy","Rigid"]},
    "Gemini": {"positive": ["Versatile","Communicative","Curious","Adaptable","Intellectual","Social","Witty","Quick-minded","Expressive","Inquisitive"], "negative": ["Inconsistent","Restless","Superficial","Indecisive","Nervous","Unreliable","Scattered","Dual-natured","Gossipy","Easily bored"]},
    "Cancer": {"positive": ["Nurturing","Intuitive","Emotional","Protective","Sympathetic","Compassionate","Empathetic","Tenacious","Caring","Loyal"], "negative": ["Moody","Overly sensitive","Clingy","Insecure","Suspicious","Manipulative","Pessimistic","Guarded","Self-pitying","Vengeful"]},
    "Leo": {"positive": ["Generous","Charismatic","Creative","Enthusiastic","Confident","Loyal","Warm-hearted","Leaders","Optimistic","Passionate"], "negative": ["Arrogant","Domineering","Attention-seeking","Dramatic","Stubborn","Self-centered","Bossy","Prideful","Vain","Inflexible"]},
    "Virgo": {"positive": ["Analytical","Practical","Meticulous","Reliable","Diligent","Intelligent","Modest","Precise","Helpful","Orderly"], "negative": ["Overly critical","Perfectionist","Obsessive","Fussy","Anxious","Judgmental","Overthinking","Rigid","Skeptical","Nitpicking"]},
    "Libra": {"positive": ["Diplomatic","Fair-minded","Cooperative","Social","Harmonious","Charming","Balanced","Idealistic","Graceful","Tactful"], "negative": ["Indecisive","People-pleasing","Superficial","Unreliable","Avoidant","Flirtatious","Detached","Conflict-avoiding","Vain","Non-confrontational"]},
    "Scorpio": {"positive": ["Passionate","Resourceful","Brave","Loyal","Determined","Focused","Intuitive","Magnetic","Ambitious","Perceptive"], "negative": ["Jealous","Secretive","Manipulative","Untrusting","Vengeful","Obsessive","Possessive","Controlling","Suspicious","Intense"]},
    "Ophiuchus": {"positive": ["Healer","Truth-seeker","Visionary","Magnetic","Wise","Mystical","Intellectual","Compassionate","Enlightened","Ambitious"], "negative": ["Jealous","Secretive","Power-seeking","Elitist","Temperamental","Arrogant","Egotistical","Envious","Critical","Mistrusting"]},
    "Sagittarius": {"positive": ["Optimistic","Freedom-loving","Adventurous","Philosophical","Expansive","Honest","Enthusiastic","Open-minded","Idealistic","Generous"], "negative": ["Tactless","Inconsistent","Restless","Overconfident","Reckless","Superficial","Irresponsible","Blunt","Impatient","Uncommitted"]},
    "Capricorn": {"positive": ["Disciplined","Responsible","Patient","Ambitious","Resourceful","Practical","Loyal","Organized","Persistent","Reserved"], "negative": ["Pessimistic","Stubborn","Detached","Unforgiving","Status-conscious","Conservative","Rigid","Materialistic","Cold","Workaholic"]},
    "Aquarius": {"positive": ["Progressive","Original","Humanitarian","Independent","Intellectual","Inventive","Friendly","Visionary","Innovative","Idealistic"], "negative": ["Detached","Unpredictable","Stubborn","Aloof","Eccentric","Rebellious","Unemotional","Inconsistent","Impersonal","Contrary"]},
    "Pisces": {"positive": ["Compassionate","Intuitive","Imaginative","Gentle","Empathetic","Creative","Spiritual","Adaptable","Artistic","Dreamy"], "negative": ["Escapist","Overly sensitive","Indecisive","Self-pitying","Unrealistic","Delusional","Impressionable","Dependent","Procrastinating","Easily influenced"]}
}
ASTROLOGY_MAX_POSITIVE_TRAITS: int = 5
ASTROLOGY_MAX_NEGATIVE_TRAITS: int = 2
NAME_GEMATRIA_RESONANT_NUMBERS: List[int] = [3, 7, 9, 11, 13, 22]
NAME_RESONANCE_BASE: float = 0.1; 
NAME_RESONANCE_WEIGHT_VOWEL: float = 0.3; 
NAME_RESONANCE_WEIGHT_LETTER: float = 0.2; 
NAME_RESONANCE_WEIGHT_GEMATRIA: float = 0.4
VOICE_FREQ_BASE: float = 220.0; 
VOICE_FREQ_ADJ_LENGTH_FACTOR: float = -50.0; 
VOICE_FREQ_ADJ_VOWEL_FACTOR: float = 80.0; 
VOICE_FREQ_ADJ_GEMATRIA_FACTOR: float = 40.0; 
VOICE_FREQ_ADJ_RESONANCE_FACTOR: float = 60.0; 
VOICE_FREQ_ADJ_YINYANG_FACTOR: float = -70.0
VOICE_FREQ_MIN_HZ: float = 80.0; 
VOICE_FREQ_MAX_HZ: float = 600.0; 
VOICE_FREQ_SOLFEGGIO_SNAP_HZ: float = 5.0
COLOR_SPECTRUM: Dict[str, Dict] = { "red": {"frequency": (400, 480), "hex": "#FF0000"}, "orange": {"frequency": (480, 510), "hex": "#FFA500"}, "gold": {"frequency": (510, 530), "hex": "#FFD700"}, "yellow": {"frequency": (530, 560), "hex": "#FFFF00"}, "green": {"frequency": (560, 610), "hex": "#00FF00"}, "blue": {"frequency": (610, 670), "hex": "#0000FF"}, "indigo": {"frequency": (670, 700), "hex": "#4B0082"}, "violet": {"frequency": (700, 790), "hex": "#8A2BE2"}, "white": {"frequency": (400, 790), "hex": "#FFFFFF"}, "black": {"frequency": (0, 0), "hex": "#000000"}, "silver": {"frequency": (0, 0), "hex": "#C0C0C0"}, "magenta": {"frequency": (0, 0), "hex": "#FF00FF"}, "grey": {"frequency": (0,0), "hex": "#808080"}, "earth_tones": {"frequency": (150, 250), "hex": "#A0522D"}, "lavender": {"frequency": (700, 750), "hex": "#E6E6FA"}, "brown": {"frequency": (150, 250), "hex": "#A52A2A"} }
COLOR_FREQ_DEFAULT: float = 500.0
SEPHIROTH_ASPECT_DEFAULT: str = "tiphareth"; 
SEPHIROTH_AFFINITY_GEMATRIA_RANGES: Dict[range, str] = {range(1, 50): "malkuth", range(50, 80): "yesod", range(80, 110): "hod", range(110, 140): "netzach", range(140, 180): "tiphareth", range(180, 220): "geburah", range(220, 260): "chesed", range(260, 300): "binah", range(300, 350): "chokmah", range(350, 1000): "kether"}
SEPHIROTH_AFFINITY_COLOR_MAP: Dict[str, str] = {"white": "kether", "grey": "chokmah", "black": "binah", "blue": "chesed", "red": "geburah", "yellow": "tiphareth", "gold": "tiphareth", "green": "netzach", "orange": "hod", "violet": "yesod", "purple": "yesod", "brown": "malkuth", "earth_tones": "malkuth", "silver": "daath", "lavender": "daath"}
SEPHIROTH_AFFINITY_STATE_MAP: Dict[str, str] = {"spark":"kether", "dream": "yesod", "formative": "malkuth", "aware": "tiphareth", "integrated": "kether", "harmonized": "chesed"}
SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD: float = 0.80; 
SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD: float = 0.35; 
SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD: float = 0.65
SEPHIROTH_AFFINITY_YIN_SEPHIROTH: List[str] = ["binah", "geburah", "hod"]; 
SEPHIROTH_AFFINITY_YANG_SEPHIROTH: List[str] = ["chokmah", "chesed", "netzach"]; 
SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: List[str] = ["kether", "tiphareth", "yesod", "malkuth", "daath"]
SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT: float = 0.2; 
SEPHIROTH_AFFINITY_COLOR_WEIGHT: float = 0.25; 
SEPHIROTH_AFFINITY_STATE_WEIGHT: float = 0.15; 
SEPHIROTH_AFFINITY_YINYANG_WEIGHT: float = 0.1; 
SEPHIROTH_AFFINITY_BALANCE_WEIGHT: float = 0.1
ELEMENTAL_AFFINITY_DEFAULT: str = "aether"; 
ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: float = 0.55; 
ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: float = 0.70
ELEMENTAL_AFFINITY_VOWEL_MAP: Dict[str, float] = {'air': 0.2, 'earth': 0.2, 'water': 0.15, 'fire': 0.15}
ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT: float = 0.3; 
ELEMENTAL_AFFINITY_COLOR_WEIGHT: float = 0.2; 
ELEMENTAL_AFFINITY_STATE_WEIGHT: float = 0.1; 
ELEMENTAL_AFFINITY_FREQ_WEIGHT: float = 0.1
ELEMENTAL_AFFINITY_FREQ_RANGES: List[Tuple[float, str]] = [(150, 'earth'), (300, 'water'), (500, 'fire'), (750, 'air'), (float('inf'), 'aether')]
ELEMENTAL_AFFINITY_COLOR_MAP: Dict[str, str] = {"red": "fire", "orange": "fire", "brown": "earth", "earth_tones": "earth", "yellow": "air", "green": "earth/water", "blue": "water", "indigo": "water/aether", "violet": "aether", "white": "aether", "black": "earth", "grey": "air", "silver": "aether", "gold": "fire", "lavender": "aether"}
ELEMENTAL_AFFINITY_STATE_MAP: Dict[str, str] = {"spark":"fire", "dream": "water", "formative": "earth", "aware": "air", "integrated": "aether", "harmonized": "water"}
LOVE_RESONANCE_FREQ = 528.0
PLATONIC_ELEMENT_MAP: Dict[str, str] = {'earth': 'hexahedron', 'water': 'icosahedron', 'fire': 'tetrahedron', 'air': 'octahedron', 'aether': 'dodecahedron', 'spirit': 'dodecahedron', 'void': 'sphere', 'light':'merkaba'}
PLATONIC_SOLIDS: List[str] = ['tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron', 'sphere', 'merkaba']
PLATONIC_DEFAULT_GEMATRIA_RANGE: int = 50
NAME_RESPONSE_TRAIN_BASE_INC: float = 0.02; 
NAME_RESPONSE_TRAIN_CYCLE_INC: float = 0.005; 
NAME_RESPONSE_TRAIN_NAME_FACTOR: float = 0.5; 
NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR: float = 0.8; 
NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT: float = 0.4
NAME_RESPONSE_STATE_FACTORS: Dict[str, float] = {'spark': 0.2, 'dream': 0.5, 'formative': 0.7, 'aware': 1.0, 'integrated': 1.2, 'harmonized': 1.1, 'default': 0.8}
HEARTBEAT_ENTRAINMENT_INC_FACTOR: float = 0.05; 
HEARTBEAT_ENTRAINMENT_DURATION_CAP: float = 300.0
LOVE_RESONANCE_BASE_INC: float = 0.03; 
LOVE_RESONANCE_CYCLE_FACTOR_DECAY: float = 0.3
LOVE_RESONANCE_STATE_WEIGHT: Dict[str, float] = {'spark': 0.1, 'dream': 0.6, 'formative': 0.8, 'aware': 1.0, 'integrated': 1.2, 'harmonized': 1.1, 'default': 0.7}
LOVE_RESONANCE_FREQ_RES_WEIGHT: float = 0.5; 
LOVE_RESONANCE_HEARTBEAT_WEIGHT: float = 0.3; 
LOVE_RESONANCE_HEARTBEAT_SCALE: float = 0.4
LOVE_RESONANCE_EMOTION_BOOST_FACTOR: float = 0.1
SACRED_GEOMETRY_STAGES: List[str] = ["seed_of_life", "flower_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "64_tetrahedron"]
SACRED_GEOMETRY_STAGE_FACTOR_BASE: float = 1.0; 
SACRED_GEOMETRY_STAGE_FACTOR_SCALE: float = 0.5
SACRED_GEOMETRY_BASE_INC_BASE: float = 0.01; 
SACRED_GEOMETRY_BASE_INC_SCALE: float = 0.005
SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT: Dict[str, float] = {'tetrahedron': 1.1, 'hexahedron': 1.1, 'octahedron': 1.1, 'dodecahedron': 1.2, 'icosahedron': 1.1, 'sphere': 1.0, 'point': 1.0, 'line': 1.0, 'triangle': 1.0, 'square': 1.0, 'pentagon': 1.1, 'hexagram': 1.2, 'heptagon': 1.1, 'octagon': 1.1, 'nonagon': 1.1, 'cross/cube': 1.1, 'vesica_piscis': 1.1, 'default': 1.0}
SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT: Dict[str, float] = {'fire': 1.1, 'earth': 1.1, 'air': 1.1, 'water': 1.1, 'aether': 1.2, 'light': 1.1, 'shadow': 0.9, 'default': 1.0}
SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE: float = 0.8; 
SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE: float = 0.4
FIBONACCI_SEQUENCE: List[int] = [1, 1, 2, 3, 5, 8, 13, 21]; 
SACRED_GEOMETRY_FIB_MAX_IDX: int = 5
ATTRIBUTE_COHERENCE_STD_DEV_SCALE: float = 2.0
CRYSTALLIZATION_REQUIRED_ATTRIBUTES: List[str] = ['name', 'soul_color', 'soul_frequency', 'sephiroth_aspect', 'elemental_affinity', 'platonic_symbol', 'crystallization_level', 'attribute_coherence', 'voice_frequency']
CRYSTALLIZATION_COMPONENT_WEIGHTS: Dict[str, float] = { 'name_resonance': 0.1, 'response_level': 0.1, 'state_stability': 0.1, 'crystallization_level': 0.3, 'attribute_coherence': 0.2, 'attribute_presence': 0.1, 'emotional_resonance': 0.1 }
CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD: float = 0.9
BIRTH_INTENSITY_DEFAULT: float = 0.7
BIRTH_CONN_WEIGHT_RESONANCE: float = 0.6; 
BIRTH_CONN_WEIGHT_INTEGRITY: float = 0.4
BIRTH_CONN_STRENGTH_FACTOR: float = 0.5; 
BIRTH_CONN_STRENGTH_CAP: float = 0.95; 
BIRTH_CONN_MOTHER_STRENGTH_FACTOR: float = 0.1
BIRTH_CONN_TRAUMA_FACTOR: float = 0.3; 
BIRTH_CONN_ACCEPTANCE_TRAUMA_FACTOR: float = 0.4 
BIRTH_CONN_MOTHER_TRAUMA_REDUCTION: float = 0.2
BIRTH_ACCEPTANCE_MIN: float = 0.2; 
BIRTH_ACCEPTANCE_TRAUMA_FACTOR: float = 0.8; 
BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR: float = 0.1
BIRTH_CORD_TRANSFER_INTENSITY_FACTOR: float = 0.2; 
BIRTH_CORD_MOTHER_EFFICIENCY_FACTOR: float = 0.1
BIRTH_CORD_INTEGRATION_CONN_FACTOR: float = 0.9; 
BIRTH_CORD_MOTHER_INTEGRATION_FACTOR: float = 0.08
BIRTH_VEIL_STRENGTH_BASE: float = 0.6; 
BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR: float = 0.3
BIRTH_VEIL_PERMANENCE_BASE: float = 0.7; 
BIRTH_VEIL_PERMANENCE_INTENSITY_FACTOR: float = 0.25
BIRTH_VEIL_RETENTION_BASE: float = 0.1; 
BIRTH_VEIL_RETENTION_INTENSITY_FACTOR: float = -0.05; 
BIRTH_VEIL_RETENTION_MIN: float = 0.02; 
BIRTH_VEIL_MOTHER_RETENTION_FACTOR: float = 0.02
BIRTH_VEIL_MEMORY_RETENTION_MODS: Dict[str, float] = {'core_identity': 0.1, 'creator_connection': 0.05, 'journey_lessons': 0.02, 'specific_details': -0.05}
BIRTH_BREATH_AMP_BASE: float = 0.5; 
BIRTH_BREATH_AMP_INTENSITY_FACTOR: float = 0.3
BIRTH_BREATH_DEPTH_BASE: float = 0.6; 
BIRTH_BREATH_DEPTH_INTENSITY_FACTOR: float = 0.2
BIRTH_BREATH_SYNC_RESONANCE_FACTOR: float = 0.8; 
BIRTH_BREATH_MOTHER_SYNC_FACTOR: float = 0.3
BIRTH_BREATH_INTEGRATION_CONN_FACTOR: float = 0.7
BIRTH_BREATH_RESONANCE_BOOST_FACTOR: float = 0.1; 
BIRTH_BREATH_MOTHER_RESONANCE_BOOST: float = 0.15
BIRTH_BREATH_ENERGY_SHIFT_FACTOR: float = 0.15; 
BIRTH_BREATH_MOTHER_ENERGY_BOOST: float = 0.1
BIRTH_BREATH_PHYSICAL_ENERGY_BASE: float = 0.5; 
BIRTH_BREATH_PHYSICAL_ENERGY_SCALE: float = 0.8
BIRTH_BREATH_SPIRITUAL_ENERGY_BASE: float = 0.7; 
BIRTH_BREATH_SPIRITUAL_ENERGY_SCALE: float = -0.5; 
BIRTH_BREATH_SPIRITUAL_ENERGY_MIN: float = 0.1
BIRTH_FINAL_INTEGRATION_WEIGHT_CONN: float = 0.4; 
BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT: float = 0.3; 
BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH: float = 0.3
BIRTH_FINAL_MOTHER_INTEGRATION_BOOST: float = 0.05
BIRTH_FINAL_FREQ_FACTOR: float = 0.8
BIRTH_FINAL_STABILITY_FACTOR: float = 0.9 # *** REVIEW: Multiplies SU? Change to modify influence? ***
BIRTH_ENERGY_BUFFER_FACTOR: float = 1.4 # Target max buffer (scaled by soul completeness)
BIRTH_ALLOC_SEED_CORE: float = 0.10 # Proportion of BEU for Seed Core
BIRTH_ALLOC_REGIONS: float = 0.30 # Proportion of BEU for Region Dev
BIRTH_ALLOC_MYCELIAL: float = 0.60 # Proportion of BEU for Mycelial Store
VEIL_BASE_RETENTION: float = 0.05 # Base retention factor before coherence bonus
VEIL_COHERENCE_RESISTANCE_FACTOR: float = 0.15 # How much coherence resists veil (scales bonus)
BIRTH_FINAL_FREQ_SHIFT_FACTOR: float = 0.05 # Max % frequency drop due to density
BIRTH_FINAL_STABILITY_PENALTY_FACTOR: float = 0.03 # Max % stability drop due to shock
BIRTH_ATTACHMENT_MIN_CORD_INTEGRITY: float = 0.75 # Min cord integrity for attachment
BIRTH_ALLOC_REGIONS: float = 0.30 # Proportion of BEU for Region Dev
BIRTH_ALLOC_MYCELIAL: float = 0.60 # Proportion of BEU for Mycelial Store
LIFE_CORD_FREQUENCIES = {'primary': 528.0}
# Define fallback constants in case import fails
BRAIN_FREQUENCIES = {
    'delta': (0.5, 4),      # Deep sleep
    'theta': (4, 8),        # Drowsy, meditation 
    'beta': (13, 30),       # Alert, active
    'gamma': (30, 100),     # High cognition
    'lambda': (100, 400)    # Higher spiritual states
}

# --- Energy Units ---
SYNAPSE_ENERGY_JOULES: float = 1e-14       # Energy of one synaptic firing in Joules
# SOUL Energy Scale: SEU <-> Joules
ENERGY_SCALE_FACTOR: float = 1e14          # 1 SEU = 1e-14 Joules (Matches Synapse Energy)
ENERGY_UNSCALE_FACTOR: float = 1e-14       # 1 Joule = 1e14 SEU (Inverse for reporting) # <<< ADD THIS BACK
# BRAIN Energy Scale: BEU <-> Joules
BRAIN_ENERGY_SCALE_FACTOR: float = 1e12    # 1 BEU = 1e-12 Joules (1 picoJoule) # <<< ENSURE THIS IS PRESENT
BRAIN_ENERGY_UNIT_PER_JOULE: float = 1e12 
ENERGY_BRAIN_14_DAYS_JOULES: float = 1e-14 * 24 * 60 * 60 * 14 # Energy of one synaptic firing in Joules over 14 days
ENERGY_BRAIN_14_DAYS_SEU: float = ENERGY_BRAIN_14_DAYS_JOULES * ENERGY_SCALE_FACTOR # Energy of one synaptic firing in SEU over 14 days



# --- Metrics Tracking ---
PERSIST_INTERVAL_SECONDS: int = 60

# --- Visualization Defaults ---
SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT: str = 'grey'; 
SOUL_SPARK_VIZ_FREQ_SIG_MARKER_FMT: str = 'bo'
SOUL_SPARK_VIZ_FREQ_SIG_BASE_FMT: str = 'r-'; 
SOUL_SPARK_VIZ_FREQ_SIG_STEM_LW: float = 1.5
SOUL_SPARK_VIZ_FREQ_SIG_MARKER_SZ: float = 5.0; 
SOUL_SPARK_VIZ_FREQ_SIG_XLABEL: str = 'Frequency (Hz)'
SOUL_SPARK_VIZ_FREQ_SIG_YLABEL: str = 'Amplitude'; 
SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR: str = 'red'

# --- Geometry Lists ---
AVAILABLE_GEOMETRY_PATTERNS: List[str] = ["flower_of_life", "seed_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "egg_of_life", "fruit_of_life", "germ_of_life", "sri_yantra", "star_tetrahedron", "64_tetrahedron"]
AVAILABLE_PLATONIC_SOLIDS: List[str] = ["tetrahedron", "hexahedron", "octahedron", "dodecahedron", "icosahedron", "sphere", "merkaba"]
GEOMETRY_BASE_FREQUENCIES: Dict[str, float] = { 'point': 963.0, 'line': 852.0, 'triangle': 396.0, 'square': 285.0, 'pentagon': 417.0, 'hexagram': 528.0, 'heptagon': 741.0, 'octagon': 741.0, 'nonagon': 852.0, 'cross/cube': 174.0, 'vesica_piscis': 444.0, 'flower_of_life': 528.0, 'seed_of_life': 432.0, 'tree_of_life': 528.0, 'metatrons_cube': 639.0, 'merkaba': 741.0, 'vector_equilibrium': 639.0 }
PLATONIC_HARMONIC_RATIOS: Dict[str, List[float]] = { 'tetrahedron': [1.0, 2.0, 3.0, 5.0], 'hexahedron': [1.0, 2.0, 4.0, 8.0], 'octahedron': [1.0, 1.5, 2.0, 3.0], 'dodecahedron': [1.0, PHI, 2.0, PHI*2, 3.0], 'icosahedron': [1.0, 1.5, 2.0, 2.5, 3.0], 'sphere': [1.0, 1.5, 2.0, 2.5, 3.0, PHI, 4.0, 5.0], 'merkaba': [1.0, 1.5, 2.0, 3.0, PHI, 4.0] }
GEOMETRY_VOID_INFLUENCE_STRENGTH: float = 0.15
PLATONIC_VOID_INFLUENCE_STRENGTH: float = 0.20

# --- END OF FILE constants.py ---



























