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

# --- Paths & Logging ---
DATA_DIR_BASE: str = "output"
LOG_LEVEL = logging.INFO # Use INFO, but DEBUG in specific files for tracing
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
OUTPUT_DIR_BASE: str = "output"

# --- Energy Units (Biomimetic Scaled Joules) ---
SYNAPSE_ENERGY_JOULES: float = 1e-14
ENERGY_SCALE_FACTOR: float = 1e14 # 1 SEU = 1.0 unit
ENERGY_UNSCALE_FACTOR: float = 1e-14
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

HARMONIZATION_ITERATIONS: int = 144             # TUNE: Number of internal harmonization steps.
HARMONIZATION_PATTERN_COHERENCE_RATE: float = 0.003 # TUNE: Rate factor builds towards 1.0 during harmonization.
HARMONIZATION_PHI_RESONANCE_RATE: float = 0.002   # TUNE: Rate factor builds towards 1.0 during harmonization.
HARMONIZATION_HARMONY_RATE: float = 0.0015        # TUNE: Rate factor builds towards 1.0 during harmonization.
HARMONIZATION_ENERGY_GAIN_RATE: float = 0.1       # TUNE: SEU gain per iteration scaled by internal order proxy.

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
STABILITY_WEIGHT_FREQ: float = 0.30       # Contribution from frequency stability
STABILITY_WEIGHT_PATTERN: float = 0.50   # Contribution from internal structure (layers, aspects, patterns)
STABILITY_WEIGHT_FIELD: float = 0.20       # Contribution from external field influences (Guff, Sephiroth)
# Factors used WITHIN the pattern component calculation
STABILITY_PATTERN_WEIGHT_LAYERS: float = 0.4
STABILITY_PATTERN_WEIGHT_ASPECTS: float = 0.3
STABILITY_PATTERN_WEIGHT_PHI: float = 0.1
STABILITY_PATTERN_WEIGHT_ALIGNMENT: float = 0.2

# --- NEW: Coherence Score Calculation Weights (Sum should ideally be 1.0) ---
COHERENCE_WEIGHT_PHASE: float = 0.35      # Contribution from phase alignment
COHERENCE_WEIGHT_HARMONY: float = 0.25    # Contribution from harmonic purity/alignment
COHERENCE_WEIGHT_PATTERN: float = 0.15    # Contribution from pattern_coherence factor
COHERENCE_WEIGHT_FIELD: float = 0.15      # Contribution from external field influences
COHERENCE_WEIGHT_CREATOR: float = 0.10    # Contribution from creator connection strength

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
SEPHIRAH_INFLUENCE_RATE_K: float = 0.1 # *** TUNE: How much each Sephirah interaction increments cumulative_sephiroth_influence (0-1)

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

# --- Brain Sustenance Target ---
BRAIN_SUSTENANCE_POWER_W: float = 20.0
BRAIN_SUSTENANCE_POWER_SEU_PER_SEC: float = BRAIN_SUSTENANCE_POWER_W * ENERGY_SCALE_FACTOR

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
FLAG_READY_FOR_STRENGTHENING="ready_for_strengthening"; 
FLAG_HARMONICALLY_STRENGTHENED="harmonically_strengthened"; 
FLAG_READY_FOR_LIFE_CORD="ready_for_life_cord"; 
FLAG_CORD_FORMATION_COMPLETE="cord_formation_complete"; 
FLAG_READY_FOR_EARTH="ready_for_earth"; 
FLAG_EARTH_HARMONIZED="earth_harmonized"; 
FLAG_READY_FOR_IDENTITY="ready_for_identity"; 
FLAG_IDENTITY_CRYSTALLIZED="identity_crystallized"; 
FLAG_READY_FOR_BIRTH="ready_for_birth"; 
FLAG_INCARNATED="incarnated"

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
ANCHOR_STRENGTH_MODIFIER: float = 0.6; EARTH_ANCHOR_STRENGTH: float = 0.9; 
EARTH_ANCHOR_RESONANCE: float = 0.9
EARTH_FREQUENCY = 136.10; EARTH_BREATH_FREQUENCY = 0.2
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
EARTH_CONN_FACTOR_CONN_STR: float = 0.5; 
EARTH_CONN_FACTOR_ELASTICITY: float = 0.3; 
EARTH_CONN_BASE_FACTOR: float = 0.1
CORD_INTEGRITY_FACTOR_CONN_STR: float = 0.4; 
CORD_INTEGRITY_FACTOR_STABILITY: float = 0.3; 
CORD_INTEGRITY_FACTOR_EARTH_CONN: float = 0.3
FINAL_STABILITY_BONUS_FACTOR: float = 0.15 # *** REVIEW: Scales SU bonus? ***
EARTH_HARMONY_INTENSITY_DEFAULT: float = 0.7; 
EARTH_HARMONY_DURATION_FACTOR_DEFAULT: float = 1.0
EARTH_FREQUENCIES: Dict[str, float] = { "schumann": 7.83, "geomagnetic": 11.75, "core_resonance": 
EARTH_FREQUENCY, "breath_cycle": EARTH_BREATH_FREQUENCY, "heartbeat_cycle": 1.2, "circadian_cycle": 1.0/(24*3600)}
HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT: float = 0.6; 
HARMONY_FREQ_TARGET_SOUL_WEIGHT: float = 0.2; 
HARMONY_FREQ_TARGET_CORE_WEIGHT: float = 0.2
HARMONY_FREQ_RES_WEIGHT_SCHUMANN: float = 0.7; 
HARMONY_FREQ_RES_WEIGHT_OTHER: float = 0.3
HARMONY_ELEM_RES_WEIGHT_PRIMARY: float = 0.6; 
HARMONY_ELEM_RES_WEIGHT_AVERAGE: float = 0.4
HARMONY_FREQ_TUNING_FACTOR: float = 0.15; 
HARMONY_FREQ_TUNING_TARGET_REACH_HZ: float = 1.0; 
HARMONY_FREQ_UPDATE_HARMONIC_COUNT: int = 5
EARTH_ELEMENTS: List[str] = ["earth", "water", "fire", "air", "aether"]
ELEMENTAL_TARGET_EARTH: float = 0.8; 
ELEMENTAL_TARGET_OTHER: float = 0.4; 
ELEMENTAL_ALIGN_INTENSITY_FACTOR: float = 0.2
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



























# # --- START OF FILE constants.py ---

# """
# Central Constants for the Soul Development Framework (Version 2.3 - Geometry Integrated)

# Consolidated and validated constants for simulation parameters, physics,
# field properties, soul defaults, stage thresholds/factors/flags,
# sound generation, mappings, logging, and NEW geometric/platonic properties.
# """

# import numpy as np
# import logging
# from typing import Dict, List, Tuple, Any

# # --- Global Simulation & Physics ---
# SAMPLE_RATE: int = 44100
# MAX_AMPLITUDE: float = 0.95
# FLOAT_EPSILON: float = 1e-9
# PI: float = np.pi
# GOLDEN_RATIO: float = (1 + np.sqrt(5)) / 2.0 # Phi (~1.618)
# PHI: float = GOLDEN_RATIO # Alias for convenience
# SILVER_RATIO: float = 1 + np.sqrt(2)
# EDGE_OF_CHAOS_RATIO: float = 1.0 / PHI # Target balance (~0.618)
# VOID_BASE_FREQUENCY_RANGE: Tuple[float, float] = (0.1, 1000.0) # Hz range for Void Field
# NOISE_GEN_AVAILABLE: bool = True # Flag for noise generation availability

# # --- Paths & Logging ---
# DATA_DIR_BASE: str = "output"
# LOG_LEVEL = logging.INFO
# LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# OUTPUT_DIR_BASE: str = "output"

# # --- Energy Units (Biomimetic Scaled Joules) ---
# SYNAPSE_ENERGY_JOULES: float = 1e-14       # Base unit: Energy of one synaptic transmission
# ENERGY_SCALE_FACTOR: float = 1e14        # Scales Joules to simulation units (1 SEU = 1.0 unit)
# ENERGY_UNSCALE_FACTOR: float = 1e-14       # Converts simulation units (SEU) back to Joules
# # --- Soul Core Units & Ranges ---
# INITIAL_SPARK_BASE_FREQUENCY_HZ: float = 432.0 # Default starting frequency if not specified
# INITIAL_SPARK_ENERGY_SEU: float = 500.0      # Base starting potential energy reservoir in SEU
# MAX_SOUL_ENERGY_SEU: float = 1e6             # Max potential energy (1 Million SEU = 1e-8 Joules)
# PASSIVE_ENERGY_DISSIPATION_RATE_SEU_PER_SEC: float = 0.1 # Passive energy loss rate

# # --- Stability & Coherence Units & Ranges ---
# MAX_STABILITY_SU: float = 100.0              # Scale for Stability Score (0-100)
# MAX_COHERENCE_CU: float = 100.0              # Scale for Coherence Score (0-100)
# # Initial Calculation Factors (Used if calculating from initial freq/harmonics)
# INITIAL_STABILITY_CALC_FACTOR: float = 0.2     # Scales initial SU score calculation
# INITIAL_COHERENCE_CALC_FACTOR: float = 0.15    # Scales initial CU score calculation
# # Calculation Weights (Used by SoulSpark._calculate_... methods)
# STABILITY_PATTERN_INTEGRITY_WEIGHT: float = 0.7  # Weight for pattern/layer integrity in SU score
# STABILITY_HARMONIC_CONSISTENCY_WEIGHT: float = 0.3 # Weight for harmonic stability in SU score
# COHERENCE_PHASE_ALIGNMENT_WEIGHT: float = 0.6    # Weight for phase alignment in CU score
# COHERENCE_HARMONIC_PURITY_WEIGHT: float = 0.4    # Weight for harmonic purity in CU score
# STABILITY_VARIANCE_PENALTY_K: float = 50.0       # Penalty factor for frequency variance in SU calc

# # --- Field System ---
# # Void Field Baseline Properties
# VOID_BASE_ENERGY_SEU: float = 10.0           # Base energy potential in Void (SEU)
# VOID_BASE_STABILITY_SU: float = 20.0           # Base stability level in Void (SU)
# VOID_BASE_COHERENCE_CU: float = 15.0           # Base coherence level in Void (CU)
# VOID_CHAOS_ORDER_BALANCE: float = 0.5          # Target chaos/order balance (0-1) used for SU/CU calculations

# PLANCK_CONSTANT_H = 6.626e-34  # Planck constant in J⋅s
# MAX_ASPECT_STRENGTH = 1.0
# MAX_SOUL_ENERGY_SEU = 1e6
# MAX_STABILITY_SU = 100.0
# MAX_COHERENCE_CU = 100.0
# FLOAT_EPSILON = 1e-9
# ENERGY_SCALE_FACTOR = 1e14
# ENERGY_UNSCALE_FACTOR = 1e-14
# INITIAL_SPARK_ENERGY_SEU = 500.0
# INITIAL_SPARK_BASE_FREQUENCY_HZ = 432.0 # For dummy init in load
# STABILITY_VARIANCE_PENALTY_K = 50.0
# COHERENCE_PHASE_ALIGNMENT_WEIGHT = 0.6
# COHERENCE_HARMONIC_PURITY_WEIGHT = 0.4
# GRID_SIZE = (64, 64, 64) # For default position
# ENTANGLEMENT_STABILIZATION_FACTOR_STRENGTH = 0.1 # Factor for strength



# # --- Geometric & Platonic Frequencies & Harmonics ---

# # Base frequencies (as specified by user)
# PLATONIC_BASE_FREQUENCIES: Dict[str, float] = {
#     'tetrahedron': 396.0, # UT Solfeggio (assoc. with Geburah sometimes, used for fire aspect)
#     'hexahedron': 285.0, # Earth Tone/Healing Tissue (assoc. with Malkuth/Earth)
#     'octahedron': 639.0, # FA Solfeggio (assoc. with Tiphareth/Air/Connection)
#     'dodecahedron': 963.0, # SI Solfeggio (assoc. with Kether/Aether/Crown)
#     'icosahedron': 369.0, # Quantum/Manifestation Frequency? (assoc. with Yesod/Water)
#     'sphere': 432.0, # Default/Neutral/Aether
#     'merkaba': 528.0 # MI Solfeggio (Love/Transformation)
# }
# # Harmonic Series Generators (Examples - can be functions or lists)
# def get_integer_harmonics(base_freq: float, count: int = 4) -> List[float]:
#     if base_freq <= FLOAT_EPSILON: return []
#     return [base_freq * (n + 1) for n in range(count)]

# def get_phi_harmonics(base_freq: float, count: int = 3) -> List[float]:
#     if base_freq <= FLOAT_EPSILON: return []
#     return [base_freq * (PHI ** n) for n in range(count + 1)] # Include base

# PLATONIC_HARMONIC_SERIES: Dict[str, List[float]] = { # Pre-calculated examples for simplicity
#     'tetrahedron': get_integer_harmonics(PLATONIC_BASE_FREQUENCIES['tetrahedron'], 4),
#     'hexahedron': get_integer_harmonics(PLATONIC_BASE_FREQUENCIES['hexahedron'], 4),
#     'octahedron': get_integer_harmonics(PLATONIC_BASE_FREQUENCIES['octahedron'], 4),
#     'dodecahedron': get_phi_harmonics(PLATONIC_BASE_FREQUENCIES['dodecahedron'], 3),
#     'icosahedron': get_phi_harmonics(PLATONIC_BASE_FREQUENCIES['icosahedron'], 3),
#     'sphere': get_integer_harmonics(PLATONIC_BASE_FREQUENCIES['sphere'], 3),
#     'merkaba': get_phi_harmonics(PLATONIC_BASE_FREQUENCIES['merkaba'], 4)
# }

# # --- Fundamental & Solfeggio Frequencies ---
# FUNDAMENTAL_FREQUENCY_432: float = 432.0
# SOLFEGGIO_FREQUENCIES: Dict[str, float] = {
#     'UT': 396.0,  # Liberation from fear and guilt
#     'RE': 417.0,  # Transformation and resonance
#     'MI': 528.0,  # Transformation and miracles
#     'FA': 639.0,  # Connecting and relationships
#     'SOL': 741.0, # Awakening intuition
#     'LA': 852.0,  # Returning to spiritual order
#     'SI': 963.0   # Awakening and returning to oneness
# }

# # --- SoulSpark Defaults & Properties ---
# # Unchanged unless new defaults needed
# SOUL_SPARK_DEFAULT_FREQ: float = 432.0
# SOUL_SPARK_DEFAULT_STABILITY: float = 0.6
# SOUL_SPARK_DEFAULT_RESONANCE: float = 0.6
# SOUL_SPARK_DEFAULT_COHERENCE: float = 0.6
# SOUL_SPARK_DEFAULT_ALIGNMENT: float = 0.1
# SOUL_SPARK_DEFAULT_PHI_RESONANCE: float = 0.5
# SOUL_SPARK_DEFAULT_ENERGY: float = 0.5 # Initial energy level (0-1 scale relative to Guff max?)
# SOUL_FREQ_DEFAULT: float = SOUL_SPARK_DEFAULT_FREQ # Default 'soul_frequency' if calculation fails

# # Effects of geometric patterns on soul attributes (Factors applied to change calculation)
# # Keys MUST match SoulSpark attribute names or influence factors
# GEOMETRY_EFFECTS: Dict[str, Dict[str, float]] = {
#     # Platonic Solids
#     'tetrahedron': {'energy_focus': 0.07, 'transformative_capacity': 0.05}, # Fire
#     'hexahedron': {'stability_factor_boost': 0.1, 'grounding': 0.1, 'energy_containment': 0.05}, # Earth
#     'octahedron': {'yin_yang_balance_push': 0.0, 'coherence_factor_boost': 0.08, 'stability_factor_boost': 0.03}, # Air (push towards 0.5 balance)
#     'dodecahedron': {'unity_connection': 0.1, 'phi_resonance_boost': 0.1, 'transcendence': 0.08}, # Aether
#     'icosahedron': {'emotional_flow': 0.1, 'adaptability': 0.07, 'coherence_factor_boost': 0.05}, # Water
#     # Other Geometries
#     'sphere': {'potential_realization': 0.1, 'unity_connection': 0.05},
#     'merkaba': {'stability_factor_boost': 0.08, 'transformative_capacity': 0.1, 'field_resilience': 0.05}, # Protection
#     'flower_of_life': {'harmony_boost': 0.1, 'structural_integration': 0.08},
#     'seed_of_life': {'potential_realization': 0.08, 'stability_factor_boost': 0.06}, # Foundation
#     'vesica_piscis': {'yin_yang_balance_push': 0.0, 'connection_boost': 0.07},
#     'tree_of_life': {'harmony_boost': 0.08, 'structural_integration': 0.08, 'connection_boost': 0.06},
#     'metatrons_cube': {'structural_integration': 0.1, 'connection_boost': 0.1}, # Structure
#     'vector_equilibrium': {'yin_yang_balance_push': 0.0, 'zero_point_attunement': 0.1}, # Balance
#     '64_tetrahedron': {'structural_integration': 0.12, 'energy_containment': 0.08} # Complex Structure
# }
# # Default effect if geometry not found
# DEFAULT_GEOMETRY_EFFECT: Dict[str, float] = {'stability_factor_boost': 0.01}

# # --- Sephiroth Absolute Potentials (CRITICAL TUNING AREA) ---
# # These define the environment's influence strength in the new units.
# SEPHIROTH_ENERGY_POTENTIALS_SEU: Dict[str, float] = { # *** TUNE ALL VALUES ***
#     'kether': MAX_SOUL_ENERGY_SEU * 0.95, 'chokmah': MAX_SOUL_ENERGY_SEU * 0.85,
#     'binah': MAX_SOUL_ENERGY_SEU * 0.80, 'daath': MAX_SOUL_ENERGY_SEU * 0.70,
#     'chesed': MAX_SOUL_ENERGY_SEU * 0.75, 'geburah': MAX_SOUL_ENERGY_SEU * 0.65,
#     'tiphareth': MAX_SOUL_ENERGY_SEU * 0.70, 'netzach': MAX_SOUL_ENERGY_SEU * 0.60,
#     'hod': MAX_SOUL_ENERGY_SEU * 0.55, 'yesod': MAX_SOUL_ENERGY_SEU * 0.45,
#     'malkuth': MAX_SOUL_ENERGY_SEU * 0.30
# }
# SEPHIROTH_TARGET_STABILITY_SU: Dict[str, float] = { # *** TUNE ALL VALUES *** (Target SU 0-100)
#     'kether': 98.0, 'chokmah': 90.0, 'binah': 92.0, 'daath': 85.0,
#     'chesed': 88.0, 'geburah': 80.0, 'tiphareth': 95.0, 'netzach': 85.0,
#     'hod': 82.0, 'yesod': 88.0, 'malkuth': 75.0
# }
# SEPHIROTH_TARGET_COHERENCE_CU: Dict[str, float] = { # *** TUNE ALL VALUES *** (Target CU 0-100)
#     'kether': 98.0, 'chokmah': 92.0, 'binah': 90.0, 'daath': 88.0,
#     'chesed': 90.0, 'geburah': 82.0, 'tiphareth': 95.0, 'netzach': 88.0,
#     'hod': 85.0, 'yesod': 90.0, 'malkuth': 70.0
# }

# # --- Sephiroth Glyph Data (Example Structure - NEEDS TO BE POPULATED) ---

# # Sigils are symbolic names, actual visual representation is separate
# SEPHIROTH_GLYPH_DATA: Dict[str, Dict[str, Any]] = {
#     'kether': { 'platonic': 'dodecahedron', 'sigil': 'Point/Crown', 'gematria_keys': ['Kether', 'Crown', 'Will', 'Unity', 1], 'fibonacci': [1, 1] },
#     'chokmah': { 'platonic': 'sphere', 'sigil': 'Line/Wheel', 'gematria_keys': ['Chokmah', 'Wisdom', 'Father', 2], 'fibonacci': [2] },
#     'binah': { 'platonic': 'icosahedron', 'sigil': 'Triangle/Womb', 'gematria_keys': ['Binah', 'Understanding', 'Mother', 3], 'fibonacci': [3] }, # Changed platonic based on common water association
#     'chesed': { 'platonic': 'hexahedron', 'sigil': 'Square/Solid', 'gematria_keys': ['Chesed', 'Mercy', 'Grace', 4], 'fibonacci': [5] },
#     'geburah': { 'platonic': 'tetrahedron', 'sigil': 'Pentagon/Sword', 'gematria_keys': ['Geburah', 'Severity', 'Strength', 5], 'fibonacci': [8] },
#     'tiphareth': { 'platonic': 'octahedron', 'sigil': 'Hexagram/Sun', 'gematria_keys': ['Tiphareth', 'Beauty', 'Harmony', 6], 'fibonacci': [13] },
#     'netzach': { 'platonic': 'icosahedron', 'sigil': 'Heptagon/Victory', 'gematria_keys': ['Netzach', 'Victory', 'Endurance', 7], 'fibonacci': [21] }, # Often associated with Venus/Watery aspects
#     'hod': { 'platonic': 'octahedron', 'sigil': 'Octagon/Splendor', 'gematria_keys': ['Hod', 'Splendor', 'Glory', 8], 'fibonacci': [34] }, # Often associated with Mercury/Airy aspects
#     'yesod': { 'platonic': 'icosahedron', 'sigil': 'Nonagon/Foundation', 'gematria_keys': ['Yesod', 'Foundation', 'Moon', 9], 'fibonacci': [55] }, # Strong water/lunar association
#     'malkuth': { 'platonic': 'hexahedron', 'sigil': 'CrossInCircle/Kingdom', 'gematria_keys': ['Malkuth', 'Kingdom', 'Shekhinah', 'Earth', 10], 'fibonacci': [89] },
#     'daath': { 'platonic': 'sphere', 'sigil': 'VoidPoint', 'gematria_keys': ['Daath', 'Knowledge', 'Abyss', 11], 'fibonacci': [] },
# }

# WAVE_PROPAGATION_SPEED: float = 0.2 # *** ADDED BACK *** Grid units per step (diffusion speed)
# ENERGY_DISSIPATION_RATE: float = 0.002 # *** ADDED BACK *** Rate relative to SEU now

# # --- Brain Sustenance Target ---
# BRAIN_SUSTENANCE_POWER_W: float = 20.0         # Approx. power consumption (Joules/sec)
# BRAIN_SUSTENANCE_POWER_SEU_PER_SEC: float = BRAIN_SUSTENANCE_POWER_W * ENERGY_SCALE_FACTOR # Target power output post-birth

# # --- Stage Prerequisites (Using SU/CU) ---
# ENTANGLEMENT_PREREQ_STABILITY_MIN_SU: float = 75.0 # Lowered slightly
# ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU: float = 75.0 # Lowered slightly
# HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU: float = 70.0
# HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU: float = 70.0
# CORD_STABILITY_THRESHOLD_SU: float = 80.0
# CORD_COHERENCE_THRESHOLD_CU: float = 80.0
# HARMONY_PREREQ_CORD_INTEGRITY_MIN: float = 0.70 # Keep 0-1
# HARMONY_PREREQ_STABILITY_MIN_SU: float = 75.0
# HARMONY_PREREQ_COHERENCE_MIN_CU: float = 75.0
# IDENTITY_STABILITY_THRESHOLD_SU: float = 85.0
# IDENTITY_COHERENCE_THRESHOLD_CU: float = 85.0
# IDENTITY_EARTH_RESONANCE_THRESHOLD: float = 0.75 # Keep 0-1
# IDENTITY_CRYSTALLIZATION_THRESHOLD: float = 0.85 # Keep 0-1 overall score
# BIRTH_PREREQ_CORD_INTEGRITY_MIN: float = 0.80 # Keep 0-1
# BIRTH_PREREQ_EARTH_RESONANCE_MIN: float = 0.75 # Keep 0-1

# # --- Readiness Flags (Unchanged) ---
# FLAG_READY_FOR_GUFF = "ready_for_guff"
# FLAG_READY_FOR_JOURNEY = "ready_for_journey"
# FLAG_READY_FOR_ENTANGLEMENT = "ready_for_entanglement"
# FLAG_READY_FOR_COMPLETION = "ready_for_completion" # *** ADDED BACK *** General flag post-entanglement
# FLAG_READY_FOR_STRENGTHENING = "ready_for_strengthening" # Specific flag for HS
# FLAG_READY_FOR_LIFE_CORD = "ready_for_life_cord"
# FLAG_READY_FOR_EARTH = "ready_for_earth"
# FLAG_READY_FOR_IDENTITY = "ready_for_identity"
# FLAG_READY_FOR_BIRTH = "ready_for_birth"

# # --- Stage Completion Flags (Unchanged) ---
# FLAG_GUFF_STRENGTHENED = "guff_strengthened"
# FLAG_SEPHIROTH_JOURNEY_COMPLETE = "sephiroth_journey_complete"
# FLAG_HARMONICALLY_STRENGTHENED = "harmonically_strengthened"
# FLAG_CORD_FORMATION_COMPLETE = "cord_formation_complete"
# FLAG_EARTH_HARMONIZED = "earth_harmonized"
# FLAG_IDENTITY_CRYSTALLIZED = "identity_crystallized"
# FLAG_INCARNATED = "incarnated"

# # --- Stage Prerequisites (Using SU/CU thresholds) ---
# # *** TUNE THESE THRESHOLDS based on expected state after previous stage ***
# ENTANGLEMENT_PREREQ_STABILITY_MIN_SU: float = 75.0
# ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU: float = 75.0
# HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU: float = 70.0 # Lowered? HS aims to increase these.
# HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU: float = 70.0
# CORD_STABILITY_THRESHOLD_SU: float = 80.0
# CORD_COHERENCE_THRESHOLD_CU: float = 80.0
# HARMONY_PREREQ_CORD_INTEGRITY_MIN: float = 0.70 # Keep 0-1
# HARMONY_PREREQ_STABILITY_MIN_SU: float = 75.0
# HARMONY_PREREQ_COHERENCE_MIN_CU: float = 75.0
# IDENTITY_STABILITY_THRESHOLD_SU: float = 85.0 # Identity requires high stability/coherence
# IDENTITY_COHERENCE_THRESHOLD_CU: float = 85.0
# IDENTITY_EARTH_RESONANCE_THRESHOLD: float = 0.75 # Keep 0-1
# IDENTITY_CRYSTALLIZATION_THRESHOLD: float = 0.85 # Keep 0-1 overall score
# BIRTH_PREREQ_CORD_INTEGRITY_MIN: float = 0.80 # Keep 0-1
# BIRTH_PREREQ_EARTH_RESONANCE_MIN: float = 0.75 # Keep 0-1

# # --- Stage Parameters ---
# # Guff Region (Targets derived from Kether's potential - review factors if needed)
# GUFF_TARGET_ENERGY_SEU: float = SEPHIROTH_ENERGY_POTENTIALS_SEU['kether'] * 0.9 # Factor applied to Kether potential
# GUFF_TARGET_STABILITY_SU: float = SEPHIROTH_TARGET_STABILITY_SU['kether'] * 0.95 # Factor applied to Kether target
# GUFF_TARGET_COHERENCE_CU: float = SEPHIROTH_TARGET_COHERENCE_CU['kether'] * 0.95 # Factor applied to Kether target
# # Guff Strengthening (Unchanged)
# GUFF_RADIUS_FACTOR: float = 0.3
# GUFF_STRENGTHENING_DURATION: float = 10.0 # *** TUNE: How long in Guff? Affects total gain.
# GUFF_STRENGTHENING_ENERGY_RATE: float = 0.3
# GUFF_STRENGTHENING_STABILITY_RATE: float = 0.05
# GUFF_STRENGTHENING_COHERENCE_RATE: float = 0.05
# GUFF_CAPACITY: int = 100 # Max number of souls in Guff

# # --- Transfer Rates (CRITICAL TUNING AREA) ---
# # These control how quickly the soul changes state. Higher = faster changes.
# ENERGY_TRANSFER_RATE_K: float = 0.05           # *** TUNE: Base rate for SEU transfer between soul/environment
# GUFF_ENERGY_TRANSFER_RATE_K: float = 0.1       # *** TUNE: SEU transfer rate specifically in Guff
# SEPHIROTH_ENERGY_EXCHANGE_RATE_K: float = 0.02 # *** TUNE: SEU exchange rate during Sephirah layer formation
# STABILITY_TRANSFER_RATE_K: float = 1       # *** TUNE: Rate of SU change towards target
# COHERENCE_TRANSFER_RATE_K: float = 1        # *** TUNE: Rate of CU change towards target
# GUFF_STABILITY_TRANSFER_RATE_K: float = 2   # *** TUNE: SU change rate in Guff
# GUFF_COHERENCE_TRANSFER_RATE_K: float = 2   # *** TUNE: CU change rate in Guff

# # --- Resonance & Aspects ---
# SEPHIROTH_DEFAULT_RADIUS: float = 8.0          # *** ADDED BACK *** Grid units (scaled in controller)
# SEPHIROTH_INFLUENCE_FALLOFF: float = 1.5       # Rate of influence decay
# DEFAULT_PHI_HARMONIC_COUNT: int = 3            # Default harmonic count if not specified
# HARMONIC_RESONANCE_ENERGY_BOOST: float = 0.012 # *** ADDED BACK *** Factor applied to energy in VoidField dynamics
# HARMONIC_RESONANCE_THRESHOLD: float = 0.05     # *** ENSURE this is still used or if detailed calc replaced it *** Log 
# RESONANCE_INTEGER_RATIO_TOLERANCE: float = 0.02 # *** TUNE: How close to integer ratio for resonance?
# RESONANCE_PHI_RATIO_TOLERANCE: float = 0.03     # *** TUNE: How close to phi ratio for resonance?
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ: float = 0.5 # *** TUNE: Weighting for final resonance score
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM: float = 0.3 # *** TUNE: Weighting for final resonance score
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI: float = 0.2 # *** TUNE: Weighting for final resonance score
# SEPHIROTH_ASPECT_TRANSFER_FACTOR: float = 0.2    # *** TUNE: Scales how much aspect strength (0-1) is transferred per interaction based on resonance.
# SEPHIROTH_JOURNEY_ATTRIBUTE_IMPART_FACTOR: float = 0.05 # Scales 0-1 attribute gain
# SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR: float = 0.04 # Scales 0-1 element gain
# SEPHIROTH_LOCAL_ENTANGLE_FREQ_PULL: float = 0.05 # Factor for frequency pull
# SEPHIROTH_LOCAL_ENTANGLE_STABILITY_GAIN_FACTOR: float = 0.01 # Influence on stability calc
# SEPHIROTH_LOCAL_ENTANGLE_COHERENCE_GAIN_FACTOR: float = 0.01 # Influence on coherence calc
# SEPHIROTH_LOCAL_ENTANGLE_ASPECT_BOOST: float = 0.02 # 0-1 boost factor for aspect strength
# MAX_ASPECT_STRENGTH: float = 1.0               # (Fixed Scale 0-1)

# # Sephiroth Journey Processing (Updated with Geometric Resonance/Transformation)
# SEPHIROTH_PRIMARY_ATTRIBUTE_MAP = {
#     'kether': 'divine_will',
#     'chokmah': 'divine_wisdom',
#     'binah': 'divine_understanding',
#     'chesed': 'divine_mercy',
#     'geburah': 'divine_severity',
#     'tiphereth': 'divine_beauty',
#     'netzach': 'divine_victory',
#     'hod': 'divine_splendor',
#     'yesod': 'divine_foundation',
#     'malkuth': 'divine_kingdom'
# }
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_FREQ: float = 0.4 # Reduced weight
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_ASPECT: float = 0.2 # Reduced weight
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_PHI: float = 0.1 # Reduced weight
# SEPHIROTH_JOURNEY_RESONANCE_WEIGHT_GEOM: float = 0.3 # NEW: Weight for geometric resonance
# SEPHIROTH_JOURNEY_ASPECT_STRENGTHEN_FACTOR: float = 0.1 # Scales initial strength gain
# SEPHIROTH_JOURNEY_STABILITY_BOOST_FACTOR: float = 0.02
# SEPHIROTH_JOURNEY_COHERENCE_BOOST_FACTOR: float = 0.02
# SEPHIROTH_JOURNEY_ENERGY_EXCHANGE_RATE: float = 0.05
# # Sephiroth Journey
# SEPHIROTH_JOURNEY_ATTRIBUTE_IMPART_FACTOR: float = 0.05 # Scales 0-1 attribute gain
# SEPHIROTH_JOURNEY_ELEMENTAL_IMPART_FACTOR: float = 0.04 # Scales 0-1 element gain
# SEPHIROTH_LOCAL_ENTANGLE_FREQ_PULL: float = 0.05 # Factor for frequency pull
# SEPHIROTH_LOCAL_ENTANGLE_STABILITY_GAIN_FACTOR: float = 0.01 # Factor influencing stability calc
# SEPHIROTH_LOCAL_ENTANGLE_COHERENCE_GAIN_FACTOR: float = 0.01 # Factor influencing coherence calc
# SEPHIROTH_LOCAL_ENTANGLE_ASPECT_BOOST: float = 0.02 # 0-1 boost factor for aspect strength
# # NEW: Geometric Transformation Factors (per interaction step)
# GEOM_TRANSFORM_STABILITY_FACTOR: Dict[str, float] = { # Boost/Penalty to Stability
#     'tetrahedron': 0.003, 'hexahedron': 0.005, 'octahedron': 0.004,
#     'dodecahedron': 0.002, 'icosahedron': -0.001, 'sphere': 0.001, 'merkaba': 0.006
# }
# GEOM_TRANSFORM_COHERENCE_FACTOR: Dict[str, float] = { # Boost/Penalty to Coherence
#     'tetrahedron': -0.001, 'hexahedron': 0.002, 'octahedron': 0.005,
#     'dodecahedron': 0.006, 'icosahedron': 0.003, 'sphere': 0.001, 'merkaba': 0.004
# }
# GEOM_TRANSFORM_PHI_RESONANCE_FACTOR: Dict[str, float] = { # Boost/Penalty to Phi Resonance
#     'tetrahedron': 0.001, 'hexahedron': -0.002, 'octahedron': 0.002,
#     'dodecahedron': 0.008, 'icosahedron': 0.005, 'sphere': 0.001, 'merkaba': 0.006
# }

# # Creator Entanglement (Unchanged)
# ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR: float = 0.15 # Increased boost for 0-1 alignment
# ENTANGLEMENT_STABILITY_BOOST_FACTOR: float = 0.1 # *** TUNE: Factor scaling SU boost based on avg aspect efficiency
# ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE: float = 0.6 # 0-1 factor
# ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE: float = 0.4 # 0-1 factor
# ENTANGLEMENT_RESONANCE_BOOST_FACTOR: float = 0.05 # Influence on resonance calc (0-1)
# ENTANGLEMENT_STABILIZATION_ITERATIONS: int = 5 # Number of iterations for stabilization

# # Harmonic Strengthening (Unchanged)
# HARMONIC_STRENGTHENING_INTENSITY_DEFAULT: float = 0.7
# HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT: float = 1.0
# HARMONIC_STRENGTHENING_PHI_AMP_INTENSITY_FACTOR: float = 0.05 # Influence on phi resonance (0-1)
# HARMONIC_STRENGTHENING_PHI_STABILITY_BOOST_FACTOR: float = 0.3 # *** TUNE: Factor scaling SU boost based on phi 
# HARMONIC_STRENGTHENING_PREREQ_STABILITY: float = 0.80
# HARMONIC_STRENGTHENING_PREREQ_COHERENCE: float = 0.80
# HARMONIC_STRENGTHENING_INTENSITY_DEFAULT: float = 0.7
# HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT: float = 1.0
# HARMONIC_STRENGTHENING_TARGET_FREQS: List[float] = sorted(list(SOLFEGGIO_FREQUENCIES.values()) + [FUNDAMENTAL_FREQUENCY_432])
# HARMONIC_STRENGTHENING_TUNING_INTENSITY_FACTOR: float = 0.1
# HARMONIC_STRENGTHENING_TUNING_TARGET_REACH_HZ: float = 1.0
# HARMONIC_STRENGTHENING_PATTERN_STAB_INTENSITY_FACTOR: float = 0.04
# HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_FACTOR: float = 0.01
# HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_CAP: float = 0.1
# HARMONIC_STRENGTHENING_PATTERN_STAB_STABILITY_BOOST: float = 0.4 # *** TUNE: Factor scaling SU boost based on pattern 
# HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_COUNT_NORM: float = 10.0
# HARMONIC_STRENGTHENING_COHERENCE_INTENSITY_FACTOR: float = 0.08 # Base CU boost factor
# HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_FACTOR: float = 0.03 # Additional CU boost from harmonics
# HARMONIC_STRENGTHENING_COHERENCE_HARMONY_BOOST: float = 0.25 # Boost to harmony (0-1) based on CU gain
# HARMONIC_STRENGTHENING_EXPANSION_INTENSITY_FACTOR: float = 0.1
# HARMONIC_STRENGTHENING_EXPANSION_STATE_FACTOR: float = 1.0
# HARMONIC_STRENGTHENING_EXPANSION_STR_INTENSITY_FACTOR: float = 0.03
# HARMONIC_STRENGTHENING_EXPANSION_STR_STATE_FACTOR: float = 0.5
# HARMONIC_STRENGTHENING_HARMONIC_COUNT = 7 # Used in soul spark for harmonic structure generation

# # Life Cord Formation (Unchanged)
# CORD_STABILITY_THRESHOLD: float = 0.85
# CORD_COHERENCE_THRESHOLD: float = 0.85
# LIFE_CORD_COMPLEXITY_DEFAULT: float = 0.7
# ANCHOR_STRENGTH_MODIFIER: float = 0.6
# EARTH_ANCHOR_STRENGTH: float = 0.9
# EARTH_ANCHOR_RESONANCE: float = 0.9
# FINAL_STABILITY_BONUS_FACTOR: float = 0.15 # *** TUNE: Factor scaling SU bonus based on cord integrity (0-1)
# # Life Cord Constants
# PRIMARY_CHANNEL_BANDWIDTH_FACTOR = 100.0
# PRIMARY_CHANNEL_STABILITY_FACTOR_CONN = 0.6
# PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX = 0.4

# # Secondary Channel Base Resistance Values and Complexity Factors (0-1 scores)
# SECONDARY_CHANNEL_RESIST_EMOTIONAL = (0.4, 0.3)  # (base, complexity_factor)
# SECONDARY_CHANNEL_RESIST_MENTAL = (0.5, 0.3)     # (base, complexity_factor)
# SECONDARY_CHANNEL_RESIST_SPIRITUAL = (0.6, 0.3)   # (base, complexity_factor)

# # Earth anchor frequency is not fixed, it's the Earth's core resonant frequency
# EARTH_FREQUENCY = 136.10 # Hz (Ohm tone - example)
# EARTH_BREATH_FREQUENCY = 0.2 # Hz (~12 breaths/min)
# PRIMARY_CHANNEL_BANDWIDTH_FACTOR: float = 200.0
# PRIMARY_CHANNEL_STABILITY_FACTOR_CONN: float = 0.7; 
# PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX: float = 0.3
# PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN: float = 0.6; 
# PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX: float = 0.4
# PRIMARY_CHANNEL_ELASTICITY_BASE: float = 0.5; 
# PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX: float = 0.3
# HARMONIC_NODE_COUNT_BASE: int = 3; 
# HARMONIC_NODE_COUNT_FACTOR: float = 15.0
# HARMONIC_NODE_AMP_BASE: float = 0.4; 
# HARMONIC_NODE_AMP_FACTOR_COMPLEX: float = 0.4
# HARMONIC_NODE_AMP_FALLOFF: float = 0.6; 
# HARMONIC_NODE_BW_INCREASE_FACTOR: float = 5.0
# MAX_CORD_CHANNELS: int = 7; 
# SECONDARY_CHANNEL_COUNT_FACTOR: float = 6.0
# SECONDARY_CHANNEL_FREQ_FACTOR: float = 0.1
# SECONDARY_CHANNEL_BW_EMOTIONAL: tuple[float, float] = (10.0, 30.0); 

# SECONDARY_CHANNEL_BW_MENTAL: tuple[float, float] = (15.0, 40.0); 
# SECONDARY_CHANNEL_RESIST_MENTAL: tuple[float, float] = (0.5, 0.3)
# SECONDARY_CHANNEL_BW_SPIRITUAL: tuple[float, float] = (20.0, 50.0); 
# SECONDARY_CHANNEL_RESIST_SPIRITUAL: tuple[float, float] = (0.6, 0.3)
# FIELD_INTEGRATION_FACTOR_FIELD_STR: float = 0.6; 
# FIELD_INTEGRATION_FACTOR_CONN_STR: float = 0.4
# FIELD_EXPANSION_FACTOR: float = 1.05
# EARTH_CONN_FACTOR_CONN_STR: float = 0.5; 
# EARTH_CONN_FACTOR_ELASTICITY: float = 0.3; 
# EARTH_CONN_BASE_FACTOR: float = 0.1
# CORD_INTEGRITY_FACTOR_CONN_STR: float = 0.4; 
# CORD_INTEGRITY_FACTOR_STABILITY: float = 0.3; 
# CORD_INTEGRITY_FACTOR_EARTH_CONN: float = 0.3


# # Earth Harmonization (Updated Earth Frequencies)
# HARMONY_PREREQ_CORD_INTEGRITY_MIN: float = 0.70
# HARMONY_PREREQ_STABILITY_MIN: float = 0.80
# HARMONY_PREREQ_COHERENCE_MIN: float = 0.80
# EARTH_HARMONY_INTENSITY_DEFAULT: float = 0.7
# EARTH_HARMONY_DURATION_FACTOR_DEFAULT: float = 1.0
# HARMONY_FINAL_STABILITY_BONUS: float = 0.08 # *** TUNE: Factor scaling SU bonus based on earth resonance (0-1)
# HARMONY_FINAL_COHERENCE_BONUS: float = 0.08 # *** TUNE: Factor scaling CU bonus based on earth resonance (0-1)
# # Earth Frequencies (Example values - more research needed for accurate mapping)
# EARTH_FREQUENCIES: Dict[str, float] = {
#     "schumann": 7.83,          # Base Schumann resonance
#     "geomagnetic": 11.75,      # Geomagnetic field related
#     "core_resonance": EARTH_FREQUENCY, # 136.10 Hz (Ohm)
#     "breath_cycle": EARTH_BREATH_FREQUENCY,  # ~12 breaths/min -> 0.2 Hz
#     "heartbeat_cycle": 1.2,   # ~72 bpm -> 1.2 Hz
#     "circadian_cycle": 1.0 / (24 * 3600), # Very low frequency
# }
# HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT: float = 0.6 # Increased Schumann importance
# HARMONY_FREQ_TARGET_SOUL_WEIGHT: float = 0.2
# HARMONY_FREQ_TARGET_CORE_WEIGHT: float = 0.2 # NEW: Weight for core resonance
# HARMONY_FREQ_RES_WEIGHT_SCHUMANN: float = 0.7 # Weight for frequency resonance calculation
# HARMONY_FREQ_RES_WEIGHT_OTHER: float = 0.3
# HARMONY_ELEM_RES_WEIGHT_PRIMARY: float = 0.6 # Weight for primary element match
# HARMONY_ELEM_RES_WEIGHT_AVERAGE: float = 0.4
# HARMONY_FREQ_TUNING_FACTOR: float = 0.15; HARMONY_FREQ_TUNING_TARGET_REACH_HZ: float = 1.0
# HARMONY_FREQ_UPDATE_HARMONIC_COUNT: int = 5
# EARTH_ELEMENTS: List[str] = ["earth", "water", "fire", "air", "aether"] # Consistent list
# ELEMENTAL_TARGET_EARTH: float = 0.8; ELEMENTAL_TARGET_OTHER: float = 0.4
# ELEMENTAL_ALIGN_INTENSITY_FACTOR: float = 0.2
# HARMONY_CYCLE_NAMES: List[str] = ["circadian", "heartbeat", "breath"]
# HARMONY_CYCLE_IMPORTANCE: Dict[str, float] = {"circadian": 0.6, "heartbeat": 0.8, "breath": 1.0}
# HARMONY_CYCLE_SYNC_TARGET_BASE: float = 0.9; HARMONY_CYCLE_SYNC_INTENSITY_FACTOR: float = 0.1; HARMONY_CYCLE_SYNC_DURATION_FACTOR: float = 1.0
# HARMONY_PLANETARY_RESONANCE_TARGET: float = 0.85; HARMONY_PLANETARY_RESONANCE_FACTOR: float = 0.15
# HARMONY_GAIA_CONNECTION_TARGET: float = 0.90; HARMONY_GAIA_CONNECTION_FACTOR: float = 0.2


# # Identity Crystallization (Unchanged - logic uses existing constants)
# IDENTITY_STABILITY_THRESHOLD_SU: float = 85.0 # Threshold SU required to START identity stage
# IDENTITY_COHERENCE_THRESHOLD_CU: float = 85.0 # Threshold CU required to START identity stage
# IDENTITY_EARTH_RESONANCE_THRESHOLD: float = 0.75
# NAME_GEMATRIA_RESONANT_NUMBERS: List[int] = [3, 7, 9, 11, 13, 22]
# NAME_RESONANCE_BASE: float = 0.1; NAME_RESONANCE_WEIGHT_VOWEL: float = 0.3; NAME_RESONANCE_WEIGHT_LETTER: float = 0.2; NAME_RESONANCE_WEIGHT_GEMATRIA: float = 0.4
# VOICE_FREQ_BASE: float = 220.0; VOICE_FREQ_ADJ_LENGTH_FACTOR: float = -50.0; VOICE_FREQ_ADJ_VOWEL_FACTOR: float = 80.0; VOICE_FREQ_ADJ_GEMATRIA_FACTOR: float = 40.0; VOICE_FREQ_ADJ_RESONANCE_FACTOR: float = 60.0; VOICE_FREQ_ADJ_YINYANG_FACTOR: float = -70.0
# VOICE_FREQ_MIN_HZ: float = 80.0; VOICE_FREQ_MAX_HZ: float = 600.0; VOICE_FREQ_SOLFEGGIO_SNAP_HZ: float = 5.0
# COLOR_SPECTRUM: Dict[str, Dict] = { # Ensure alignment with sephiroth_data colors
#     "red": {"frequency": (400, 480), "hex": "#FF0000"}, 
#     "orange": {"frequency": (480, 510), "hex": "#FFA500"},
#     "gold": {"frequency": (510, 530), "hex": "#FFD700"}, 
#     "yellow": {"frequency": (530, 560), "hex": "#FFFF00"},
#     "green": {"frequency": (560, 610), "hex": "#00FF00"}, 
#     "blue": {"frequency": (610, 670), "hex": "#0000FF"},
#     "indigo": {"frequency": (670, 700), "hex": "#4B0082"}, 
#     "violet": {"frequency": (700, 790), "hex": "#8A2BE2"},
#     "white": {"frequency": (400, 790), "hex": "#FFFFFF"}, 
#     "black": {"frequency": (0, 0), "hex": "#000000"},
#     "silver": {"frequency": (0, 0), "hex": "#C0C0C0"}, 
#     "magenta": {"frequency": (0, 0), "hex": "#FF00FF"},
#     "grey": {"frequency": (0,0), "hex": "#808080"}, 
#     "earth_tones": {"frequency": (150, 250), "hex": "#A0522D"},
#     "lavender": {"frequency": (700, 750), "hex": "#E6E6FA"}, 
#     "brown": {"frequency": (150, 250), "hex": "#A52A2A"} # Added brown
# }
# COLOR_FREQ_DEFAULT: float = 500.0
# SACRED_GEOMETRY_BASE_INC_BASE: float = 0.01 # Base increase for crystallization level (0-1 score) per stage
# SACRED_GEOMETRY_STAGE_FACTOR_SCALE: float = 0.5 # How much later stages increase level more
# SEPHIROTH_ASPECT_DEFAULT: str = "tiphareth"
# SEPHIROTH_AFFINITY_GEMATRIA_RANGES: Dict[range, str] = {range(1, 50): "malkuth", range(50, 80): "yesod", range(80, 110): "hod", range(110, 140): "netzach", range(140, 180): "tiphareth", range(180, 220): "geburah", range(220, 260): "chesed", range(260, 300): "binah", range(300, 350): "chokmah", range(350, 1000): "kether"}
# SEPHIROTH_AFFINITY_COLOR_MAP: Dict[str, str] = {"white": "kether", "grey": "chokmah", "black": "binah", "blue": "chesed", "red": "geburah", "yellow": "tiphareth", "gold": "tiphareth", "green": "netzach", "orange": "hod", "violet": "yesod", "purple": "yesod", "brown": "malkuth", "earth_tones": "malkuth", "silver": "daath", "lavender": "daath"}
# SEPHIROTH_AFFINITY_STATE_MAP: Dict[str, str] = {"spark":"kether", "dream": "yesod", "formative": "malkuth", "aware": "tiphareth", "integrated": "kether", "harmonized": "chesed"}
# SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD: float = 0.80
# SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD: float = 0.35; SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD: float = 0.65
# SEPHIROTH_AFFINITY_YIN_SEPHIROTH: List[str] = ["binah", "geburah", "hod"]; SEPHIROTH_AFFINITY_YANG_SEPHIROTH: List[str] = ["chokmah", "chesed", "netzach"]; SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: List[str] = ["kether", "tiphareth", "yesod", "malkuth", "daath"]
# SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT: float = 0.2; SEPHIROTH_AFFINITY_COLOR_WEIGHT: float = 0.25; SEPHIROTH_AFFINITY_STATE_WEIGHT: float = 0.15; SEPHIROTH_AFFINITY_YINYANG_WEIGHT: float = 0.1; SEPHIROTH_AFFINITY_BALANCE_WEIGHT: float = 0.1
# ELEMENTAL_AFFINITY_DEFAULT: str = "aether"
# ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: float = 0.55; ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: float = 0.70
# ELEMENTAL_AFFINITY_VOWEL_MAP: Dict[str, float] = {'air': 0.2, 'earth': 0.2, 'water': 0.15, 'fire': 0.15}
# ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT: float = 0.3; ELEMENTAL_AFFINITY_COLOR_WEIGHT: float = 0.2; ELEMENTAL_AFFINITY_STATE_WEIGHT: float = 0.1; ELEMENTAL_AFFINITY_FREQ_WEIGHT: float = 0.1
# ELEMENTAL_AFFINITY_FREQ_RANGES: List[Tuple[float, str]] = [(150, 'earth'), (300, 'water'), (500, 'fire'), (750, 'air'), (float('inf'), 'aether')]
# ELEMENTAL_AFFINITY_COLOR_MAP: Dict[str, str] = {"red": "fire", "orange": "fire", "brown": "earth", "earth_tones": "earth", "yellow": "air", "green": "earth/water", "blue": "water", "indigo": "water/aether", "violet": "aether", "white": "aether", "black": "earth", "grey": "air", "silver": "aether", "gold": "fire", "lavender": "aether"} # Added brown
# ELEMENTAL_AFFINITY_STATE_MAP: Dict[str, str] = {"spark":"fire", "dream": "water", "formative": "earth", "aware": "air", "integrated": "aether", "harmonized": "water"}
# LOVE_RESONANCE_FREQ = 528.0 # Define Love Resonance frequency
# PLATONIC_ELEMENT_MAP: Dict[str, str] = {'earth': 'hexahedron', 'water': 'icosahedron', 'fire': 'tetrahedron', 'air': 'octahedron', 'aether': 'dodecahedron', 'spirit': 'dodecahedron', 'void': 'sphere', 'light':'merkaba'}
# PLATONIC_SOLIDS: List[str] = ['tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron', 'sphere', 'merkaba']
# PLATONIC_DEFAULT_GEMATRIA_RANGE: int = 50
# NAME_RESPONSE_TRAIN_BASE_INC: float = 0.02; NAME_RESPONSE_TRAIN_CYCLE_INC: float = 0.005
# NAME_RESPONSE_TRAIN_NAME_FACTOR: float = 0.5
# NAME_RESPONSE_STATE_FACTORS: Dict[str, float] = {'spark': 0.2, 'dream': 0.5, 'formative': 0.7, 'aware': 1.0, 'integrated': 1.2, 'harmonized': 1.1, 'default': 0.8}
# NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR: float = 0.8; NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT: float = 0.4
# HEARTBEAT_ENTRAINMENT_INC_FACTOR: float = 0.05; HEARTBEAT_ENTRAINMENT_DURATION_CAP: float = 300.0
# LOVE_RESONANCE_BASE_INC: float = 0.03; LOVE_RESONANCE_CYCLE_FACTOR_DECAY: float = 0.3
# LOVE_RESONANCE_STATE_WEIGHT: Dict[str, float] = {'spark': 0.1, 'dream': 0.6, 'formative': 0.8, 'aware': 1.0, 'integrated': 1.2, 'harmonized': 1.1, 'default': 0.7}
# LOVE_RESONANCE_FREQ_RES_WEIGHT: float = 0.5; LOVE_RESONANCE_HEARTBEAT_WEIGHT: float = 0.3; LOVE_RESONANCE_HEARTBEAT_SCALE: float = 0.4
# LOVE_RESONANCE_EMOTION_BOOST_FACTOR: float = 0.1
# SACRED_GEOMETRY_STAGES: List[str] = ["seed_of_life", "flower_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "64_tetrahedron"]
# SACRED_GEOMETRY_STAGE_FACTOR_BASE: float = 1.0; SACRED_GEOMETRY_STAGE_FACTOR_SCALE: float = 0.5
# SACRED_GEOMETRY_BASE_INC_BASE: float = 0.01; SACRED_GEOMETRY_BASE_INC_SCALE: float = 0.005
# SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT: Dict[str, float] = {'tetrahedron': 1.1, 'hexahedron': 1.1, 'octahedron': 1.1, 'dodecahedron': 1.2, 'icosahedron': 1.1, 'sphere': 1.0, 'point': 1.0, 'line': 1.0, 'triangle': 1.0, 'square': 1.0, 'pentagon': 1.1, 'hexagram': 1.2, 'heptagon': 1.1, 'octagon': 1.1, 'nonagon': 1.1, 'cross/cube': 1.1, 'vesica_piscis': 1.1, 'default': 1.0}
# SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT: Dict[str, float] = {'fire': 1.1, 'earth': 1.1, 'air': 1.1, 'water': 1.1, 'aether': 1.2, 'light': 1.1, 'shadow': 0.9, 'default': 1.0}
# SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE: float = 0.8; SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE: float = 0.4
# FIBONACCI_SEQUENCE: List[int] = [1, 1, 2, 3, 5, 8, 13, 21]; SACRED_GEOMETRY_FIB_MAX_IDX: int = 5
# ATTRIBUTE_COHERENCE_STD_DEV_SCALE: float = 2.0
# IDENTITY_CRYSTALLIZATION_THRESHOLD: float = 0.85 # Threshold to pass stage
# CRYSTALLIZATION_REQUIRED_ATTRIBUTES: List[str] = ['name', 'soul_color', 'soul_frequency', 'sephiroth_aspect', 'elemental_affinity', 'platonic_symbol', 'crystallization_level', 'attribute_coherence', 'voice_frequency']
# CRYSTALLIZATION_COMPONENT_WEIGHTS: Dict[str, float] = { 'name_resonance': 0.1, 'response_level': 0.1, 'state_stability': 0.1, 'crystallization_level': 0.3, 'attribute_coherence': 0.2, 'attribute_presence': 0.1, 'emotional_resonance': 0.1 }
# CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD: float = 0.9 # Min % of required attrs present

# # Birth Process (Unchanged)
# BIRTH_PREREQ_CORD_INTEGRITY_MIN: float = 0.80
# BIRTH_PREREQ_EARTH_RESONANCE_MIN: float = 0.75
# BIRTH_INTENSITY_DEFAULT: float = 0.7
# BIRTH_FINAL_FREQ_FACTOR: float = 0.8 # Multiplier for frequency change
# BIRTH_FINAL_STABILITY_FACTOR: float = 0.9 # Factor influencing stability calc
# BIRTH_CONN_WEIGHT_RESONANCE: float = 0.6; 
# BIRTH_CONN_WEIGHT_INTEGRITY: float = 0.4
# BIRTH_CONN_STRENGTH_FACTOR: float = 0.5; 
# BIRTH_CONN_STRENGTH_CAP: float = 0.95
# BIRTH_CONN_TRAUMA_FACTOR: float = 0.3
# BIRTH_ACCEPTANCE_MIN: float = 0.2; 
# BIRTH_ACCEPTANCE_TRAUMA_FACTOR: float = 0.8
# BIRTH_CORD_TRANSFER_INTENSITY_FACTOR: float = 0.2
# BIRTH_CORD_INTEGRATION_CONN_FACTOR: float = 0.9
# BIRTH_VEIL_STRENGTH_BASE: float = 0.6; 
# BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR: float = 0.3
# BIRTH_VEIL_PERMANENCE_BASE: float = 0.7; 
# BIRTH_VEIL_PERMANENCE_INTENSITY_FACTOR: float = 0.25
# BIRTH_VEIL_RETENTION_BASE: float = 0.1; 
# BIRTH_VEIL_RETENTION_INTENSITY_FACTOR: float = -0.05
# BIRTH_VEIL_RETENTION_MIN: float = 0.02
# BIRTH_VEIL_MEMORY_RETENTION_MODS: Dict[str, float] = {'core_identity': 0.1, 'creator_connection': 0.05, 'journey_lessons': 0.02, 'specific_details': -0.05}
# BIRTH_BREATH_AMP_BASE: float = 0.5; 
# BIRTH_BREATH_AMP_INTENSITY_FACTOR: float = 0.3
# BIRTH_BREATH_DEPTH_BASE: float = 0.6; 
# BIRTH_BREATH_DEPTH_INTENSITY_FACTOR: float = 0.2
# BIRTH_BREATH_SYNC_RESONANCE_FACTOR: float = 0.8
# BIRTH_BREATH_INTEGRATION_CONN_FACTOR: float = 0.7
# BIRTH_BREATH_RESONANCE_BOOST_FACTOR: float = 0.1
# BIRTH_BREATH_ENERGY_SHIFT_FACTOR: float = 0.15
# BIRTH_BREATH_PHYSICAL_ENERGY_BASE: float = 0.5; 
# BIRTH_BREATH_PHYSICAL_ENERGY_SCALE: float = 0.8
# BIRTH_BREATH_SPIRITUAL_ENERGY_BASE: float = 0.7; 
# BIRTH_BREATH_SPIRITUAL_ENERGY_SCALE: float = -0.5
# BIRTH_BREATH_SPIRITUAL_ENERGY_MIN: float = 0.1
# BIRTH_FINAL_INTEGRATION_WEIGHT_CONN: float = 0.4; 
# BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT: float = 0.3; BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH: float = 0.3
# BIRTH_FINAL_FREQ_FACTOR: float = 0.8 # Multiplier for frequency change
# BIRTH_FINAL_STABILITY_FACTOR: float = 0.9 # Factor influencing stability calc
# BIRTH_CONN_MOTHER_STRENGTH_FACTOR: float = 0.05
# BIRTH_CONN_MOTHER_TRAUMA_REDUCTION: float = 0.2
# BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR: float = 0.1
# BIRTH_CORD_MOTHER_EFFICIENCY_FACTOR: float = 0.1
# BIRTH_CORD_MOTHER_INTEGRATION_FACTOR: float = 0.08
# BIRTH_VEIL_MOTHER_RETENTION_FACTOR: float = 0.02
# BIRTH_BREATH_MOTHER_SYNC_FACTOR: float = 0.3
# BIRTH_BREATH_MOTHER_RESONANCE_BOOST: float = 0.15
# BIRTH_BREATH_MOTHER_ENERGY_BOOST: float = 0.1
# BIRTH_FINAL_MOTHER_INTEGRATION_BOOST: float = 0.05
# BIRTH_FINAL_FREQ_FACTOR: float = 0.8 # Multiplier for frequency change Hz -> Hz
# BIRTH_FINAL_STABILITY_FACTOR: float = 0.9 # Multiplier for SU -> SU change

# # --- Metrics Tracking ---
# PERSIST_INTERVAL_SECONDS: int = 60 # How often metrics tries to save

# # --- Visualization Defaults ---
# # Keep visualization defaults as they are specific to the current visualization code
# SOUL_SPARK_VIZ_FREQ_SIG_STEM_FMT: str = 'grey'; SOUL_SPARK_VIZ_FREQ_SIG_MARKER_FMT: str = 'bo'
# SOUL_SPARK_VIZ_FREQ_SIG_BASE_FMT: str = 'r-'; SOUL_SPARK_VIZ_FREQ_SIG_STEM_LW: float = 1.5
# SOUL_SPARK_VIZ_FREQ_SIG_MARKER_SZ: float = 5.0; SOUL_SPARK_VIZ_FREQ_SIG_XLABEL: str = 'Frequency (Hz)'
# SOUL_SPARK_VIZ_FREQ_SIG_YLABEL: str = 'Amplitude'; SOUL_SPARK_VIZ_FREQ_SIG_BASE_COLOR: str = 'red'

# # Add sacred geometry pattern list (ensure matches imported modules)
# AVAILABLE_GEOMETRY_PATTERNS: List[str] = ["flower_of_life", "seed_of_life", "vesica_piscis", "tree_of_life", "metatrons_cube", "merkaba", "vector_equilibrium", "egg_of_life", "fruit_of_life", "germ_of_life", "sri_yantra", "star_tetrahedron", "64_tetrahedron"]
# AVAILABLE_PLATONIC_SOLIDS: List[str] = ["tetrahedron", "hexahedron", "octahedron", "dodecahedron", "icosahedron", "sphere", "merkaba"]

# GEOMETRY_BASE_FREQUENCIES: Dict[str, float] = {
#     'point': 963.0, # Kether
#     'line': 852.0, # Chokmah
#     'triangle': 396.0, # Binah (also Tetrahedron/Octahedron/Icosahedron)
#     'square': 285.0, # Chesed (also Hexahedron) - Adjusted from user example for consistency
#     'pentagon': 417.0, # Geburah
#     'hexagram': 528.0, # Tiphareth
#     'heptagon': 741.0, # Netzach
#     'octagon': 741.0, # Hod - Same as Netzach for symmetry?
#     'nonagon': 852.0, # Yesod - Same as Chokmah for connection?
#     'cross/cube': 174.0, # Malkuth
#     'vesica_piscis': 444.0, # Daath
#     'flower_of_life': 528.0, # Complex, relates to Tiphareth
#     'seed_of_life': 432.0, # Foundational
#     'tree_of_life': 528.0, # Harmonizing
#     'metatrons_cube': 639.0, # Connective
#     'merkaba': 741.0, # Vehicle
#     'vector_equilibrium': 639.0 # Balancing
# }

# KETHER_FREQ: float = 963.0 # Kether frequency

# # Simple Integer Harmonic Series (Example)
# GEOMETRY_HARMONIC_RATIOS: Dict[str, List[float]] = {
#     name: [1.0, 2.0, 3.0, 4.0] for name in GEOMETRY_BASE_FREQUENCIES # Default integer series
# }
# PLATONIC_HARMONIC_RATIOS: Dict[str, List[float]] = {
#     'tetrahedron': [1.0, 2.0, 3.0, 5.0], # More energetic?
#     'hexahedron': [1.0, 2.0, 4.0, 8.0], # Stable powers of 2?
#     'octahedron': [1.0, 1.5, 2.0, 3.0], # Balanced
#     'dodecahedron': [1.0, PHI, 2.0, PHI*2, 3.0], # Phi-based
#     'icosahedron': [1.0, 1.5, 2.0, 2.5, 3.0], # Flowing
#     'sphere': [1.0, 1.5, 2.0, 2.5, 3.0, PHI, 4.0, 5.0], # All potential
#     'merkaba': [1.0, 1.5, 2.0, 3.0, PHI, 4.0] # Combined
# }
# # Add specific overrides if needed, e.g.,
# # GEOMETRY_HARMONIC_RATIOS['flower_of_life'] = [1.0, PHI, 2.0, 3.0, 5.0]

# # Energetic Effect Keywords/Modifiers (Used in Sephiroth Journey Processing)
# # --- Geometric Effects (Factors applied to change calculations) ---
# # *** REVIEW / TUNE these factors carefully ***
# # Keys should match SoulSpark attributes or represent influence types
# GEOMETRY_EFFECTS: Dict[str, Dict[str, float]] = { # Example values need tuning
#     'tetrahedron': {'energy_focus': 0.1, 'transformative_capacity': 0.07}, # Fire
#     'hexahedron': {'stability_factor_boost': 0.15, 'grounding': 0.12, 'energy_containment': 0.08}, # Earth
#     'octahedron': {'yin_yang_balance_push': 0.0, 'coherence_factor_boost': 0.1, 'stability_factor_boost': 0.05}, # Air
#     'dodecahedron': {'unity_connection': 0.15, 'phi_resonance_boost': 0.12, 'transcendence': 0.1}, # Aether
#     'icosahedron': {'emotional_flow': 0.12, 'adaptability': 0.1, 'coherence_factor_boost': 0.08}, # Water
#     'sphere': {'potential_realization': 0.1, 'unity_connection': 0.05},
#     'merkaba': {'stability_factor_boost': 0.1, 'transformative_capacity': 0.12, 'field_resilience': 0.08},
#     'flower_of_life': {'harmony_boost': 0.12, 'structural_integration': 0.1},
#     'seed_of_life': {'potential_realization': 0.1, 'stability_factor_boost': 0.08},
#     'vesica_piscis': {'yin_yang_balance_push': 0.0, 'connection_boost': 0.09},
#     'tree_of_life': {'harmony_boost': 0.1, 'structural_integration': 0.1, 'connection_boost': 0.08},
#     'metatrons_cube': {'structural_integration': 0.12, 'connection_boost': 0.12},
#     'vector_equilibrium': {'yin_yang_balance_push': 0.0, 'zero_point_attunement': 0.15},
#     '64_tetrahedron': {'structural_integration': 0.15, 'energy_containment': 0.1}
# }
# # Add default effect if needed
# DEFAULT_GEOMETRY_EFFECT: Dict[str, float] = {'stability_factor_boost': 0.01}

# # Influence Strength on Void Field (How strongly geometry modifies the void)
# GEOMETRY_VOID_INFLUENCE_STRENGTH: float = 0.15 # Base strength for modifying void props
# PLATONIC_VOID_INFLUENCE_STRENGTH: float = 0.20 # Platonics have stronger base influence?


# --- END OF FILE constants.py ---
