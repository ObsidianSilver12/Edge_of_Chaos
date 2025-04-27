"""
Brain Region Definitions

Defines the comprehensive properties of brain regions, sub-regions,
and memory types for the brain formation process.

These definitions serve as the foundation for creating the 3D brain structure.
"""

import numpy as np
from enum import Enum

# ---- Hemisphere Definitions ----
HEMISPHERES = {
    'left': {'proportion': 0.5, 'bias': 'logical', 'primary_function': 'analytical'},
    'right': {'proportion': 0.5, 'bias': 'creative', 'primary_function': 'holistic'}
}

# ---- Color Palette (Specific unique shades) ----
class ColorShades(Enum):
    # Blues
    ROYAL_BLUE_1 = "#0a1f5c"
    ROYAL_BLUE_2 = "#14347c"
    ROYAL_BLUE_3 = "#1e4a9c"
    ROYAL_BLUE_4 = "#285fbc"
    ROYAL_BLUE_5 = "#3275dc"
    ROYAL_BLUE_6 = "#3c8afc"
    ROYAL_BLUE_7 = "#46a0fc"
    
    # Greens
    EMERALD_1 = "#0a3c1e"
    EMERALD_2 = "#14522a"
    EMERALD_3 = "#1e6836"
    EMERALD_4 = "#287e42"
    EMERALD_5 = "#32944e"
    EMERALD_6 = "#3caa5a"
    EMERALD_7 = "#46c066"
    
    # Reds
    RUBY_1 = "#5c0a1a"
    RUBY_2 = "#7c1423"
    RUBY_3 = "#9c1e2c"
    RUBY_4 = "#bc2835"
    RUBY_5 = "#dc323e"
    RUBY_6 = "#fc3c47"
    RUBY_7 = "#fc4650"
    
    # Golds
    GOLD_1 = "#5c4a0a"
    GOLD_2 = "#7c6414"
    GOLD_3 = "#9c7e1e"
    GOLD_4 = "#bc9828"
    GOLD_5 = "#dcb232"
    GOLD_6 = "#fccc3c"
    GOLD_7 = "#fcd646"
    
    # Purples
    VIOLET_1 = "#3c0a5c"
    VIOLET_2 = "#52147c"
    VIOLET_3 = "#681e9c"
    VIOLET_4 = "#7e28bc"
    VIOLET_5 = "#9432dc"
    VIOLET_6 = "#aa3cfc"
    VIOLET_7 = "#c046fc"
    
    # Oranges
    AMBER_1 = "#5c2a0a"
    AMBER_2 = "#7c3a14"
    AMBER_3 = "#9c4a1e"
    AMBER_4 = "#bc5a28"
    AMBER_5 = "#dc6a32"
    AMBER_6 = "#fc7a3c"
    AMBER_7 = "#fc8a46"
    
    # Teals
    TEAL_1 = "#0a5c4a"
    TEAL_2 = "#147c64"
    TEAL_3 = "#1e9c7e"
    TEAL_4 = "#28bc98"
    TEAL_5 = "#32dcb2"
    TEAL_6 = "#3cfccc"
    TEAL_7 = "#46fcd6"
    
    # Silvers
    SILVER_1 = "#4a4a4a"
    SILVER_2 = "#5a5a5a"
    SILVER_3 = "#6a6a6a"
    SILVER_4 = "#7a7a7a"
    SILVER_5 = "#8a8a8a"
    SILVER_6 = "#9a9a9a"
    SILVER_7 = "#aaaaaa"


# ---- Standard Glyphs ----
# References to glyphs stored elsewhere - for now just placeholders
class BrainGlyphs(Enum):
    FRONTAL_GLYPH = "frontal_glyph"
    PARIETAL_GLYPH = "parietal_glyph"
    TEMPORAL_GLYPH = "temporal_glyph"
    OCCIPITAL_GLYPH = "occipital_glyph"
    LIMBIC_GLYPH = "limbic_glyph"
    CEREBELLUM_GLYPH = "cerebellum_glyph"
    BRAIN_STEM_GLYPH = "brain_stem_glyph"
    
    PREFRONTAL_GLYPH = "prefrontal_glyph"
    MOTOR_CORTEX_GLYPH = "motor_cortex_glyph"
    SENSORY_CORTEX_GLYPH = "sensory_cortex_glyph"
    VISUAL_CORTEX_GLYPH = "visual_cortex_glyph"
    AUDITORY_CORTEX_GLYPH = "auditory_cortex_glyph"
    HIPPOCAMPUS_GLYPH = "hippocampus_glyph"
    AMYGDALA_GLYPH = "amygdala_glyph"
    THALAMUS_GLYPH = "thalamus_glyph"
    HYPOTHALAMUS_GLYPH = "hypothalamus_glyph"


# ---- Platonic Solids ----
class PlatonicSolids(Enum):
    TETRAHEDRON = "tetrahedron"
    CUBE = "cube"  # Hexahedron
    OCTAHEDRON = "octahedron"
    DODECAHEDRON = "dodecahedron"
    ICOSAHEDRON = "icosahedron"


# ---- Major Brain Regions ----
MAJOR_REGIONS = {
    'frontal': {
        'proportion': 0.28,
        'location_bias': (0.7, 0.5, 0.2),  # (x, y, z) bias in unit cube
        'function': 'executive_control',
        'default_wave': 'beta',
        'wave_frequency_hz': 18.5,
        'color': ColorShades.ROYAL_BLUE_7,
        'sound_base_note': 'C4',
        'glyph': BrainGlyphs.FRONTAL_GLYPH,
        'sub_regions': ['prefrontal', 'orbitofrontal', 'motor_cortex', 'broca']
    },
    'parietal': {
        'proportion': 0.19,
        'location_bias': (0.4, 0.5, 0.6),
        'function': 'sensory_integration',
        'default_wave': 'alpha',
        'wave_frequency_hz': 10.2,
        'color': ColorShades.EMERALD_5,
        'sound_base_note': 'E4',
        'glyph': BrainGlyphs.PARIETAL_GLYPH,
        'sub_regions': ['somatosensory_cortex', 'superior_parietal', 'inferior_parietal']
    },
    'temporal': {
        'proportion': 0.22,
        'location_bias': (0.3, 0.2, 0.5),
        'function': 'auditory_language_memory',
        'default_wave': 'alpha',
        'wave_frequency_hz': 9.7,
        'color': ColorShades.GOLD_6,
        'sound_base_note': 'G4',
        'glyph': BrainGlyphs.TEMPORAL_GLYPH,
        'sub_regions': ['primary_auditory', 'wernicke', 'medial_temporal', 'fusiform']
    },
    'occipital': {
        'proportion': 0.14,
        'location_bias': (0.3, 0.8, 0.5),
        'function': 'visual_processing',
        'default_wave': 'alpha',
        'wave_frequency_hz': 11.3,
        'color': ColorShades.VIOLET_4,
        'sound_base_note': 'B4',
        'glyph': BrainGlyphs.OCCIPITAL_GLYPH,
        'sub_regions': ['primary_visual', 'secondary_visual', 'visual_association']
    },
    'limbic': {
        'proportion': 0.11,
        'location_bias': (0.5, 0.5, 0.5),  # Central
        'function': 'emotion_memory',
        'default_wave': 'theta',
        'wave_frequency_hz': 6.8,
        'color': ColorShades.RUBY_3,
        'sound_base_note': 'D4',
        'glyph': BrainGlyphs.LIMBIC_GLYPH,
        'sub_regions': ['hippocampus', 'amygdala', 'thalamus', 'hypothalamus', 'cingulate']
    },
    'cerebellum': {
        'proportion': 0.14,
        'location_bias': (0.5, 0.9, 0.2),  # Low back
        'function': 'motor_coordination',
        'default_wave': 'alpha',
        'wave_frequency_hz': 9.3,
        'color': ColorShades.TEAL_3,
        'sound_base_note': 'A3',
        'glyph': BrainGlyphs.CEREBELLUM_GLYPH,
        'sub_regions': ['anterior_lobe', 'posterior_lobe', 'flocculonodular']
    },
    'brain_stem': {
        'proportion': 0.06,
        'location_bias': (0.5, 0.8, 0.1),  # Bottom center
        'function': 'basic_life_functions',
        'default_wave': 'delta',
        'wave_frequency_hz': 2.5,
        'color': ColorShades.SILVER_2,
        'sound_base_note': 'F3',
        'glyph': BrainGlyphs.BRAIN_STEM_GLYPH,
        'sub_regions': ['midbrain', 'pons', 'medulla']
    }
}


# ---- Sub Regions ----
SUB_REGIONS = {
    # Frontal Sub-Regions
    'prefrontal': {
        'parent': 'frontal',
        'proportion': 0.4,  # Proportion of parent region
        'platonic_solid': PlatonicSolids.DODECAHEDRON,
        'function': 'planning_decision_making',
        'wave_frequency_hz': 19.3,
        'color': ColorShades.ROYAL_BLUE_5,
        'sound_modifier': 'harmonic_fifth',
        'glyph': BrainGlyphs.PREFRONTAL_GLYPH,
    },
    'orbitofrontal': {
        'parent': 'frontal',
        'proportion': 0.15,
        'platonic_solid': PlatonicSolids.ICOSAHEDRON,
        'function': 'reward_evaluation',
        'wave_frequency_hz': 17.8,
        'color': ColorShades.ROYAL_BLUE_3,
        'sound_modifier': 'harmonic_third',
        'glyph': None,  # No specific glyph
    },
    'motor_cortex': {
        'parent': 'frontal',
        'proportion': 0.3,
        'platonic_solid': PlatonicSolids.CUBE,
        'function': 'movement_control',
        'wave_frequency_hz': 18.9,
        'color': ColorShades.ROYAL_BLUE_6,
        'sound_modifier': 'perfect_fourth',
        'glyph': BrainGlyphs.MOTOR_CORTEX_GLYPH,
    },
    'broca': {
        'parent': 'frontal',
        'proportion': 0.15,
        'platonic_solid': PlatonicSolids.TETRAHEDRON,
        'function': 'speech_production',
        'wave_frequency_hz': 17.5,
        'color': ColorShades.ROYAL_BLUE_2,
        'sound_modifier': 'minor_third',
        'glyph': None,
    },
    
    # Parietal Sub-Regions
    'somatosensory_cortex': {
        'parent': 'parietal',
        'proportion': 0.4,
        'platonic_solid': PlatonicSolids.CUBE,
        'function': 'touch_body_awareness',
        'wave_frequency_hz': 10.5,
        'color': ColorShades.EMERALD_6,
        'sound_modifier': 'perfect_fifth',
        'glyph': BrainGlyphs.SENSORY_CORTEX_GLYPH,
    },
    'superior_parietal': {
        'parent': 'parietal',
        'proportion': 0.3,
        'platonic_solid': PlatonicSolids.OCTAHEDRON,
        'function': 'spatial_coordination',
        'wave_frequency_hz': 10.1,
        'color': ColorShades.EMERALD_4,
        'sound_modifier': 'major_third',
        'glyph': None,
    },
    'inferior_parietal': {
        'parent': 'parietal',
        'proportion': 0.3,
        'platonic_solid': PlatonicSolids.ICOSAHEDRON,
        'function': 'language_mathematics',
        'wave_frequency_hz': 9.8,
        'color': ColorShades.EMERALD_3,
        'sound_modifier': 'minor_seventh',
        'glyph': None,
    },
    
    # Temporal Sub-Regions
    'primary_auditory': {
        'parent': 'temporal',
        'proportion': 0.25,
        'platonic_solid': PlatonicSolids.TETRAHEDRON,
        'function': 'sound_processing',
        'wave_frequency_hz': 9.9,
        'color': ColorShades.GOLD_7,
        'sound_modifier': 'perfect_octave',
        'glyph': BrainGlyphs.AUDITORY_CORTEX_GLYPH,
    },
    'wernicke': {
        'parent': 'temporal',
        'proportion': 0.2,
        'platonic_solid': PlatonicSolids.DODECAHEDRON,
        'function': 'language_comprehension',
        'wave_frequency_hz': 9.5,
        'color': ColorShades.GOLD_5,
        'sound_modifier': 'major_sixth',
        'glyph': None,
    },
    'medial_temporal': {
        'parent': 'temporal',
        'proportion': 0.3,
        'platonic_solid': PlatonicSolids.ICOSAHEDRON,
        'function': 'memory_formation',
        'wave_frequency_hz': 8.8,
        'color': ColorShades.GOLD_3,
        'sound_modifier': 'perfect_fourth',
        'glyph': None,
    },
    'fusiform': {
        'parent': 'temporal',
        'proportion': 0.25,
        'platonic_solid': PlatonicSolids.CUBE,
        'function': 'face_recognition',
        'wave_frequency_hz': 9.2,
        'color': ColorShades.GOLD_4,
        'sound_modifier': 'minor_third',
        'glyph': None,
    },
    
    # Occipital Sub-Regions
    'primary_visual': {
        'parent': 'occipital',
        'proportion': 0.4,
        'platonic_solid': PlatonicSolids.TETRAHEDRON,
        'function': 'basic_visual_processing',
        'wave_frequency_hz': 11.5,
        'color': ColorShades.VIOLET_5,
        'sound_modifier': 'major_third',
        'glyph': BrainGlyphs.VISUAL_CORTEX_GLYPH,
    },
    'secondary_visual': {
        'parent': 'occipital',
        'proportion': 0.35,
        'platonic_solid': PlatonicSolids.OCTAHEDRON,
        'function': 'complex_visual_processing',
        'wave_frequency_hz': 11.0,
        'color': ColorShades.VIOLET_3,
        'sound_modifier': 'perfect_fifth',
        'glyph': None,
    },
    'visual_association': {
        'parent': 'occipital',
        'proportion': 0.25,
        'platonic_solid': PlatonicSolids.DODECAHEDRON,
        'function': 'visual_integration',
        'wave_frequency_hz': 10.7,
        'color': ColorShades.VIOLET_6,
        'sound_modifier': 'minor_seventh',
        'glyph': None,
    },
    
    # Limbic Sub-Regions
    'hippocampus': {
        'parent': 'limbic',
        'proportion': 0.2,
        'platonic_solid': PlatonicSolids.DODECAHEDRON,
        'function': 'memory_formation',
        'wave_frequency_hz': 6.5,
        'color': ColorShades.RUBY_4,
        'sound_modifier': 'major_third',
        'glyph': BrainGlyphs.HIPPOCAMPUS_GLYPH,
    },
    'amygdala': {
        'parent': 'limbic',
        'proportion': 0.15,
        'platonic_solid': PlatonicSolids.ICOSAHEDRON,
        'function': 'emotional_processing',
        'wave_frequency_hz': 7.2,
        'color': ColorShades.RUBY_5,
        'sound_modifier': 'minor_third',
        'glyph': BrainGlyphs.AMYGDALA_GLYPH,
    },
    'thalamus': {
        'parent': 'limbic',
        'proportion': 0.25,
        'platonic_solid': PlatonicSolids.CUBE,
        'function': 'sensory_relay',
        'wave_frequency_hz': 8.1,
        'color': ColorShades.RUBY_2,
        'sound_modifier': 'perfect_fifth',
        'glyph': BrainGlyphs.THALAMUS_GLYPH,
    },
    'hypothalamus': {
        'parent': 'limbic',
        'proportion': 0.15,
        'platonic_solid': PlatonicSolids.TETRAHEDRON,
        'function': 'homeostasis',
        'wave_frequency_hz': 5.9,
        'color': ColorShades.RUBY_6,
        'sound_modifier': 'octave_down',
        'glyph': BrainGlyphs.HYPOTHALAMUS_GLYPH,
    },
    'cingulate': {
        'parent': 'limbic',
        'proportion': 0.25,
        'platonic_solid': PlatonicSolids.OCTAHEDRON,
        'function': 'emotion_regulation',
        'wave_frequency_hz': 7.5,
        'color': ColorShades.RUBY_1,
        'sound_modifier': 'major_seventh',
        'glyph': None,
    },
    
    # Cerebellum Sub-Regions
    'anterior_lobe': {
        'parent': 'cerebellum',
        'proportion': 0.4,
        'platonic_solid': PlatonicSolids.OCTAHEDRON,
        'function': 'skilled_movement',
        'wave_frequency_hz': 9.4,
        'color': ColorShades.TEAL_5,
        'sound_modifier': 'perfect_fourth',
        'glyph': None,
    },
    'posterior_lobe': {
        'parent': 'cerebellum',
        'proportion': 0.5,
        'platonic_solid': PlatonicSolids.ICOSAHEDRON,
        'function': 'movement_coordination',
        'wave_frequency_hz': 9.1,
        'color': ColorShades.TEAL_2,
        'sound_modifier': 'major_sixth',
        'glyph': None,
    },
    'flocculonodular': {
        'parent': 'cerebellum',
        'proportion': 0.1,
        'platonic_solid': PlatonicSolids.TETRAHEDRON,
        'function': 'balance_coordination',
        'wave_frequency_hz': 8.9,
        'color': ColorShades.TEAL_7,
        'sound_modifier': 'minor_seventh',
        'glyph': None,
    },
    
    # Brain Stem Sub-Regions
    'midbrain': {
        'parent': 'brain_stem',
        'proportion': 0.3,
        'platonic_solid': PlatonicSolids.OCTAHEDRON,
        'function': 'visual_auditory_reflexes',
        'wave_frequency_hz': 3.1,
        'color': ColorShades.SILVER_5,
        'sound_modifier': 'perfect_fifth',
        'glyph': None,
    },
    'pons': {
        'parent': 'brain_stem',
        'proportion': 0.35,
        'platonic_solid': PlatonicSolids.CUBE,
        'function': 'breathing_facial_expressions',
        'wave_frequency_hz': 2.7,
        'color': ColorShades.SILVER_3,
        'sound_modifier': 'minor_third',
        'glyph': None,
    },
    'medulla': {
        'parent': 'brain_stem',
        'proportion': 0.35,
        'platonic_solid': PlatonicSolids.TETRAHEDRON,
        'function': 'autonomic_functions',
        'wave_frequency_hz': 1.8,
        'color': ColorShades.SILVER_6,
        'sound_modifier': 'octave_down',
        'glyph': None,
    }
}


# ---- Memory Types ----
MEMORY_TYPES = {
    'survival': {
        'base_frequency_hz': 4.2,
        'signal_pattern': 'high_amplitude_rapid',
        'decay_rate': 0.0001,
        'structure_distribution': 0.7,  # 70% in structure
        'brain_distribution': 0.3,     # 30% distributed
        'fullness_threshold': 0.95,    # When to switch to full distribution
        'color_identifier': ColorShades.RUBY_1,
        'sound_pattern': 'staccato_low',
        'primary_structure': 'limbic',
        'secondary_structures': ['brain_stem', 'frontal']
    },
    'emotional': {
        'base_frequency_hz': 5.8,
        'signal_pattern': 'oscillating_medium',
        'decay_rate': 0.0008,
        'structure_distribution': 0.7,
        'brain_distribution': 0.3,
        'fullness_threshold': 0.9,
        'color_identifier': ColorShades.AMBER_3,
        'sound_pattern': 'flowing_mid',
        'primary_structure': 'limbic',
        'secondary_structures': ['temporal', 'frontal']
    },
    'procedural': {
        'base_frequency_hz': 9.3,
        'signal_pattern': 'steady_rhythmic',
        'decay_rate': 0.00005,
        'structure_distribution': 0.8,
        'brain_distribution': 0.2,
        'fullness_threshold': 0.95,
        'color_identifier': ColorShades.TEAL_4,
        'sound_pattern': 'rhythmic_sustained',
        'primary_structure': 'cerebellum',
        'secondary_structures': ['frontal', 'parietal']
    },
    'semantic': {
        'base_frequency_hz': 11.7,
        'signal_pattern': 'complex_structured',
        'decay_rate': 0.0003,
        'structure_distribution': 0.6,
        'brain_distribution': 0.4,
        'fullness_threshold': 0.9,
        'color_identifier': ColorShades.ROYAL_BLUE_1,
        'sound_pattern': 'harmonic_complex',
        'primary_structure': 'temporal',
        'secondary_structures': ['frontal', 'parietal']
    },
    'episodic': {
        'base_frequency_hz': 7.1,
        'signal_pattern': 'burst_sequence',
        'decay_rate': 0.001,
        'structure_distribution': 0.7,
        'brain_distribution': 0.3,
        'fullness_threshold': 0.85,
        'color_identifier': ColorShades.GOLD_2,
        'sound_pattern': 'melodic_varied',
        'primary_structure': 'temporal',
        'secondary_structures': ['limbic', 'frontal']
    },
    'working': {
        'base_frequency_hz': 14.5,
        'signal_pattern': 'rapid_changing',
        'decay_rate': 0.05,
        'structure_distribution': 0.9,
        'brain_distribution': 0.1,
        'fullness_threshold': 0.85,
        'color_identifier': ColorShades.EMERALD_7,
        'sound_pattern': 'staccato_high',
        'primary_structure': 'frontal',
        'secondary_structures': ['parietal']
    },
    'dimensional': {
        'base_frequency_hz': 18.3,
        'signal_pattern': 'geometric_harmonic',
        'decay_rate': 0.0002,
        'structure_distribution': 0.5,
        'brain_distribution': 0.5,
        'fullness_threshold': 0.92,
        'color_identifier': ColorShades.VIOLET_7,
        'sound_pattern': 'resonant_complex',
        'primary_structure': 'parietal',
        'secondary_structures': ['occipital', 'frontal']
    },
    'ephemeral': {
        'base_frequency_hz': 23.7,
        'signal_pattern': 'transient_fluctuating',
        'decay_rate': 0.1,
        'structure_distribution': 0.4,
        'brain_distribution': 0.6,
        'fullness_threshold': 0.8,
        'color_identifier': ColorShades.SILVER_7,
        'sound_pattern': 'ethereal_light',
        'primary_structure': 'frontal',
        'secondary_structures': ['temporal', 'limbic']
    }
}


# ---- Sound Modifiers ----
SOUND_MODIFIERS = {
    'harmonic_fifth': {'interval': 1.5, 'timbre_shift': 0.2, 'amplitude_modifier': 1.1},
    'harmonic_third': {'interval': 1.25, 'timbre_shift': 0.15, 'amplitude_modifier': 1.05},
    'perfect_fourth': {'interval': 1.33, 'timbre_shift': 0.1, 'amplitude_modifier': 1.0},
    'perfect_fifth': {'interval': 1.5, 'timbre_shift': 0.25, 'amplitude_modifier': 1.15},
    'perfect_octave': {'interval': 2.0, 'timbre_shift': 0.3, 'amplitude_modifier': 1.2},
    'major_third': {'interval': 1.25, 'timbre_shift': 0.12, 'amplitude_modifier': 1.02},
    'minor_third': {'interval': 1.2, 'timbre_shift': 0.1, 'amplitude_modifier': 0.98},
    'major_sixth': {'interval': 1.67, 'timbre_shift': 0.22, 'amplitude_modifier': 1.1},
    'minor_seventh': {'interval': 1.78, 'timbre_shift': 0.18, 'amplitude_modifier': 1.05},
    'major_seventh': {'interval': 1.87, 'timbre_shift': 0.25, 'amplitude_modifier': 1.12},
    'octave_down': {'interval': 0.5, 'timbre_shift': -0.3, 'amplitude_modifier': 0.9}
}


# ---- Brain Wave Types ----
BRAIN_WAVE_TYPES = {
    'delta': {'frequency_range': (0.5, 4.0), 'state': 'deep_sleep'},
    'theta': {'frequency_range': (4.0, 8.0), 'state': 'drowsy_meditation'},
    'alpha': {'frequency_range': (8.0, 13.0), 'state': 'relaxed_aware'},
    'beta': {'frequency_range': (13.0, 30.0), 'state': 'alert_focused'},
    'gamma': {'frequency_range': (30.0, 100.0), 'state': 'high_cognition'},
    'lambda': {'frequency_range': (100.0, 200.0), 'state': 'insight_transcendence'}
}


# ---- Signal Patterns for Memory Types ----
SIGNAL_PATTERNS = {
    'high_amplitude_rapid': {
        'amplitude_range': (0.7, 1.0),
        'frequency_modifier': 1.2,
        'waveform': 'square_with_noise',
        'burst_pattern': 'short_intense'
    },
    'oscillating_medium': {
        'amplitude_range': (0.4, 0.8),
        'frequency_modifier': 0.9,
        'waveform': 'sine_complex',
        'burst_pattern': 'rhythmic_varying'
    },
    'steady_rhythmic': {
        'amplitude_range': (0.5, 0.7),
        'frequency_modifier': 1.0,
        'waveform': 'triangular',
        'burst_pattern': 'continuous'
    },
    'complex_structured': {
        'amplitude_range': (0.3, 0.6),
        'frequency_modifier': 1.1,
        'waveform': 'composite_harmonic',
        'burst_pattern': 'organized_complex'
    },
    'burst_sequence': {
        'amplitude_range': (0.5, 0.9),
        'frequency_modifier': 0.95,
        'waveform': 'sawtooth_modified',
        'burst_pattern': 'sequential_bursts'
    },
    'rapid_changing': {
        'amplitude_range': (0.4, 0.7),
        'frequency_modifier': 1.3,
        'waveform': 'variable_complex',
        'burst_pattern': 'chaotic_rapid'
    },
    'geometric_harmonic': {
        'amplitude_range': (0.6, 0.8),
        'frequency_modifier': 1.15,
        'waveform': 'golden_ratio_based',
        'burst_pattern': 'fractal_recurring'
    },
    'transient_fluctuating': {
        'amplitude_range': (0.2, 0.5),
        'frequency_modifier': 1.4,
        'waveform': 'ephemeral_noise',
        'burst_pattern': 'transient_sporadic'
    }
}


# ---- Sound Patterns for Memory Types ----
SOUND_PATTERNS = {
    'staccato_low': {
        'base_note_shift': -1.0,
        'attack': 0.05,
        'decay': 0.1,
        'sustain': 0.3,
        'release': 0.2,
        'harmonics': [1.0, 0.5, 0.25]
    },
    'flowing_mid': {
        'base_note_shift': 0.0,
        'attack': 0.2,
        'decay': 0.3,
        'sustain': 0.6,
        'release': 0.5,
        'harmonics': [1.0, 0.7, 0.4, 0.2]
    },
    'rhythmic_sustained': {
        'base_note_shift': -0.5,
        'attack': 0.1,
        'decay': 0.2,
        'sustain': 0.8,
        'release': 0.3,
        'harmonics': [1.0, 0.6, 0.3, 0.15, 0.08]
    },
    'harmonic_complex': {
        'base_note_shift': 0.5,
        'attack': 0.15,
        'decay': 0.25,
        'sustain': 0.7,
        'release': 0.4,
        'harmonics': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
    },
    'melodic_varied': {
        'base_note_shift': 0.25,
        'attack': 0.18,
        'decay': 0.3,
        'sustain': 0.5,
        'release': 0.6,
        'harmonics': [1.0, 0.7, 0.5, 0.2, 0.1]
    },
    'staccato_high': {
        'base_note_shift': 1.0,
        'attack': 0.02,
        'decay': 0.1,
        'sustain': 0.2,
        'release': 0.1,
        'harmonics': [1.0, 0.4, 0.1]
    },
    'resonant_complex': {
        'base_note_shift': 0.75,
        'attack': 0.25,
        'decay': 0.35,
        'sustain': 0.6,
        'release': 0.8,
        'harmonics': [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1]
    },
    'ethereal_light': {
        'base_note_shift': 1.5,
        'attack': 0.35,
        'decay': 0.5,
        'sustain': 0.3,
        'release': 0.9,
        'harmonics': [0.7, 1.0, 0.9, 0.5, 0.3, 0.1]
    }
}


# ---- Boundary Definitions ----
BOUNDARY_TYPES = {
    'sharp': {
        'transition_width': 3,  # pixels/voxels
        'gradient_pattern': 'step',
        'permeability': 0.2
    },
    'gradual': {
        'transition_width': 10,
        'gradient_pattern': 'linear',
        'permeability': 0.6
    },
    'diffuse': {
        'transition_width': 20,
        'gradient_pattern': 'exponential',
        'permeability': 0.8
    },
    'oscillating': {
        'transition_width': 15,
        'gradient_pattern': 'sine_wave',
        'permeability': 0.5
    },
    'fractal': {
        'transition_width': 12,
        'gradient_pattern': 'fractal_noise',
        'permeability': 0.4
    }
}

# Define which boundary type exists between regions
REGION_BOUNDARIES = {
    ('frontal', 'parietal'): 'gradual',
    ('frontal', 'temporal'): 'gradual',
    ('parietal', 'temporal'): 'gradual',
    ('parietal', 'occipital'): 'sharp',
    ('temporal', 'occipital'): 'sharp',
    ('limbic', 'frontal'): 'diffuse',
    ('limbic', 'parietal'): 'diffuse',
    ('limbic', 'temporal'): 'diffuse',
    ('limbic', 'occipital'): 'oscillating',
    ('cerebellum', 'brain_stem'): 'sharp',
    ('cerebellum', 'occipital'): 'fractal',
    ('brain_stem', 'limbic'): 'fractal',
    # Default boundary type if not specified
    'default': 'gradual'
}

# ---- Global Constants ----
# Standard 3D Grid Dimensions (Fibonacci-based)
GRID_DIMENSIONS = (233, 233, 233)

# Golden Ratio for sacred proportions
GOLDEN_RATIO = 1.618033988749895

# Base frequency for entire brain
BASE_BRAIN_FREQUENCY = 7.83  # Earth's Schumann resonance

# Standard tolerance for frequency matching
FREQUENCY_MATCH_TOLERANCE = 0.05

# Energy distribution constants
ENERGY_DISTRIBUTION = {
    'frontal': 0.28,
    'parietal': 0.19,
    'temporal': 0.22,
    'occipital': 0.14,
    'limbic': 0.11,
    'cerebellum': 0.14,
    'brain_stem': 0.06
}

# Define mapping helpers to get parent regions and other relationships
def get_sub_regions_for_parent(parent_region):
    """Get all sub-regions for a specified parent region."""
    return [sr for sr, data in SUB_REGIONS.items() if data['parent'] == parent_region]

def get_all_region_names():
    """Get all region names (major and sub) as a list."""
    return list(MAJOR_REGIONS.keys()) + list(SUB_REGIONS.keys())

def get_parent_region(sub_region):
    """Get the parent region for a specified sub-region."""
    if sub_region in SUB_REGIONS:
        return SUB_REGIONS[sub_region]['parent']
    elif sub_region in MAJOR_REGIONS:
        return sub_region  # Major regions are their own parent
    return None

def get_platonic_solid_for_region(region_name):
    """Get the platonic solid for a region."""
    if region_name in SUB_REGIONS:
        return SUB_REGIONS[region_name]['platonic_solid']
    # For major regions, combine platonics of sub-regions or use default
    if region_name in MAJOR_REGIONS:
        sub_regions = get_sub_regions_for_parent(region_name)
        if sub_regions:
            # Just return the most common platonic among sub-regions for simplicity
            platonic_counts = {}
            for sr in sub_regions:
                ps = SUB_REGIONS[sr]['platonic_solid']
                platonic_counts[ps] = platonic_counts.get(ps, 0) + 1
            return max(platonic_counts.items(), key=lambda x: x[1])[0]
        else:
            return PlatonicSolids.CUBE  # Default
    return None

