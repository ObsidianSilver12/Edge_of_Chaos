# region_definitions.py
"""
Brain Region Definitions - ANATOMICALLY CORRECT VERSION WITH COMPLETE PROPERTIES

Defines the comprehensive properties of REAL brain regions and sub-regions
based on actual neuroanatomy with all sound, frequency, and boundary data.

Architecture: Grid → Regions → Sub-regions → Blocks
No artificial hemispheric division - brain works as integrated system.
"""

import numpy as np
from enum import Enum

# ---- Import missing dependencies ----
from shared.constants.constants import *

# ---- BRAIN-SPECIFIC COLOR SHADES (Unique from framework COLOR_SPECTRUM) ----
class ColorShades(Enum):
    """Brain-specific color shades that don't conflict with framework colors."""

    # Frontal Cortex - Various shades of blue (avoiding pure blue #0000FF)
    FRONTAL_BLUE = "#4169E1"          # Royal Blue

    # Parietal Cortex - Various shades of green (avoiding pure green #00FF00)
    PARIETAL_GREEN = "#228B22"        # Forest Green

    # Temporal Cortex - Various shades of gold/amber (avoiding pure gold #FFD700)
    TEMPORAL_GOLD = "#DAA520"         # Goldenrod

    # Occipital Cortex - Various shades of purple (avoiding pure violet #8A2BE2)
    OCCIPITAL_PURPLE = "#9370DB"      # Medium Purple

    # Limbic System - Various shades of red (avoiding pure red #FF0000)
    LIMBIC_RED = "#DC143C"            # Crimson

    # Cerebellum - Teal shades (not in framework spectrum)
    CEREBELLUM_TEAL = "#008B8B"       # Dark Cyan

    # Brainstem - Gray shades (avoiding pure grey #808080)
    BRAINSTEM_GRAY = "#696969"        # Dim Gray

    # Additional unique brain colors for sub-regions
    NEURAL_CORAL = "#FF7F50"          # Coral
    SYNAPTIC_TURQUOISE = "#40E0D0"    # Turquoise
    MYCELIAL_OLIVE = "#808000"        # Olive
    BORDER_SLATE = "#708090"          # Slate Gray
    FIELD_NAVY = "#000080"            # Navy
    COSMIC_MAROON = "#800000"         # Maroon
    QUANTUM_PLUM = "#DDA0DD"          # Plum

# Verification that brain colors don't conflict with framework colors
FRAMEWORK_HEX_COLORS = {
    "#FF0000", "#FFA500", "#FFD700", "#FFFF00", "#00FF00", "#0000FF", 
    "#4B0082", "#8A2BE2", "#FFFFFF", "#000000", "#C0C0C0", "#FF00FF", 
    "#808080", "#A0522D", "#E6E6FA", "#A52A2A"
}

BRAIN_HEX_COLORS = {color.value for color in ColorShades}

# Assert no conflicts
assert not FRAMEWORK_HEX_COLORS.intersection(BRAIN_HEX_COLORS), \
    f"Brain colors conflict with framework colors: {FRAMEWORK_HEX_COLORS.intersection(BRAIN_HEX_COLORS)}"

# ---- REAL Neuroanatomical Regions and Sub-regions WITH COMPLETE PROPERTIES ----

MAJOR_REGIONS = {
    'frontal_cortex': {
        'proportion': 0.32,  # Reduced from 0.35 to help balance total
        'location_bias': (0.2, 0.3, 0.5),  # Front of brain
        'function': 'executive_control_motor',
        'default_wave': 'beta',
        'wave_frequency_hz': 18.5,
        'color': ColorShades.FRONTAL_BLUE,
        'sound_base_note': 'C4',
        'boundary_type': 'gradual',
        'sub_regions': [
            'prefrontal_cortex', 'primary_motor_cortex', 'premotor_cortex',
            'supplementary_motor_area', 'broca_area'
        ]
    },

    'parietal_cortex': {
        'proportion': 0.19,  # Reduced from 0.20
        'location_bias': (0.4, 0.7, 0.6),  # Top-back
        'function': 'sensory_integration_spatial',
        'default_wave': 'alpha',
        'wave_frequency_hz': 10.2,
        'color': ColorShades.PARIETAL_GREEN,
        'sound_base_note': 'E4',
        'boundary_type': 'sharp',
        'sub_regions': [
            'primary_somatosensory_cortex', 'secondary_somatosensory_cortex',
            'posterior_parietal_cortex', 'superior_parietal_lobule', 'inferior_parietal_lobule'
        ]
    },

    'temporal_cortex': {
        'proportion': 0.17,  # Reduced from 0.18
        'location_bias': (0.3, 0.2, 0.3),  # Sides, lower
        'function': 'auditory_language_memory',
        'default_wave': 'theta',
        'wave_frequency_hz': 9.7,
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_base_note': 'G4',
        'boundary_type': 'gradual',
        'sub_regions': [
            'primary_auditory_cortex', 'wernicke_area', 'superior_temporal_gyrus',
            'middle_temporal_gyrus', 'inferior_temporal_gyrus', 'hippocampus', 'parahippocampal_gyrus'
        ]
    },

    'occipital_cortex': {
        'proportion': 0.12,  # Kept same - smallest cortical region
        'location_bias': (0.8, 0.5, 0.5),  # Back of brain
        'function': 'visual_processing',
        'default_wave': 'alpha',
        'wave_frequency_hz': 11.3,
        'color': ColorShades.OCCIPITAL_PURPLE,
        'sound_base_note': 'B4',
        'boundary_type': 'sharp',
        'sub_regions': [
            'primary_visual_cortex', 'secondary_visual_cortex', 'visual_area_v3',
            'visual_area_v4', 'visual_area_v5_mt'
        ]
    },

    'limbic_system': {
        'proportion': 0.08,  # Kept same - appropriate for deep structures
        'location_bias': (0.5, 0.5, 0.4),  # Deep, central
        'function': 'emotion_motivation_memory',
        'default_wave': 'theta',
        'wave_frequency_hz': 6.8,
        'color': ColorShades.LIMBIC_RED,
        'sound_base_note': 'D4',
        'boundary_type': 'diffuse',
        'sub_regions': [
            'amygdala', 'anterior_cingulate', 'posterior_cingulate', 'insula', 'orbitofrontal_cortex'
        ]
    },

    'cerebellum': {
        'proportion': 0.10,  # Kept same - anatomically accurate
        'location_bias': (0.7, 0.2, 0.2),  # Back, bottom
        'function': 'motor_learning_coordination',
        'default_wave': 'alpha',
        'wave_frequency_hz': 9.3,
        'color': ColorShades.CEREBELLUM_TEAL,
        'sound_base_note': 'A3',
        'boundary_type': 'fractal',
        'sub_regions': [
            'cerebellar_cortex', 'deep_cerebellar_nuclei', 'vestibulocerebellum',
            'spinocerebellum', 'cerebrocerebellum'
        ]
    },

    'brainstem': {
        'proportion': 0.02,  # Reduced from 0.07 - brainstem is actually quite small
        'location_bias': (0.6, 0.5, 0.1),  # Central, bottom
        'function': 'vital_functions_arousal',
        'default_wave': 'delta',
        'wave_frequency_hz': 2.5,
        'color': ColorShades.BRAINSTEM_GRAY,
        'sound_base_note': 'F3',
        'boundary_type': 'oscillating',
        'sub_regions': [
            'midbrain', 'pons', 'medulla', 'reticular_formation'
        ]
    }
}

# ---- Sub-region Properties WITH COMPLETE SOUND AND FREQUENCY DATA ----
SUB_REGIONS = {
    # FRONTAL CORTEX Sub-regions
    'prefrontal_cortex': {
        'parent': 'frontal_cortex',
        'proportion': 0.40,
        'function': 'executive_control_working_memory',
        'brodmann_areas': [9, 10, 11, 12, 46, 47],
        'wave_frequency_hz': 19.3,
        'default_wave': 'beta',
        'color': ColorShades.FRONTAL_BLUE,
        'sound_modifier': 'harmonic_fifth',
        'sound_pattern': 'harmonic_complex',
        'boundary_type': 'gradual',
        'connections': ['temporal_cortex', 'parietal_cortex', 'limbic_system']
    },
    'primary_motor_cortex': {
        'parent': 'frontal_cortex', 
        'proportion': 0.25,
        'function': 'voluntary_movement_control',
        'brodmann_areas': [4],
        'wave_frequency_hz': 18.9,
        'default_wave': 'beta',
        'color': ColorShades.FRONTAL_BLUE,
        'sound_modifier': 'perfect_fourth',
        'sound_pattern': 'rhythmic_sustained',
        'boundary_type': 'sharp',
        'connections': ['cerebellum', 'brainstem', 'parietal_cortex']
    },
    'premotor_cortex': {
        'parent': 'frontal_cortex',
        'proportion': 0.20,
        'function': 'movement_planning',
        'brodmann_areas': [6],
        'wave_frequency_hz': 18.2,
        'default_wave': 'beta',
        'color': ColorShades.FRONTAL_BLUE,
        'sound_modifier': 'major_third',
        'sound_pattern': 'flowing_mid',
        'boundary_type': 'gradual',
        'connections': ['primary_motor_cortex', 'parietal_cortex']
    },
    'supplementary_motor_area': {
        'parent': 'frontal_cortex',
        'proportion': 0.10,
        'function': 'complex_movement_sequences',
        'brodmann_areas': [6],
        'wave_frequency_hz': 17.8,
        'default_wave': 'beta',
        'color': ColorShades.FRONTAL_BLUE,
        'sound_modifier': 'harmonic_third',
        'sound_pattern': 'melodic_varied',
        'boundary_type': 'diffuse',
        'connections': ['primary_motor_cortex', 'premotor_cortex']
    },
    'broca_area': {
        'parent': 'frontal_cortex',
        'proportion': 0.05,
        'function': 'speech_production',
        'brodmann_areas': [44, 45],
        'wave_frequency_hz': 17.5,
        'default_wave': 'beta',
        'color': ColorShades.FRONTAL_BLUE,
        'sound_modifier': 'minor_third',
        'sound_pattern': 'staccato_high',
        'boundary_type': 'sharp',
        'connections': ['wernicke_area', 'primary_motor_cortex']
    },
    
    # PARIETAL CORTEX Sub-regions
    'primary_somatosensory_cortex': {
        'parent': 'parietal_cortex',
        'proportion': 0.30,
        'function': 'touch_pressure_temperature',
        'brodmann_areas': [1, 2, 3],
        'wave_frequency_hz': 10.5,
        'default_wave': 'alpha',
        'color': ColorShades.PARIETAL_GREEN,
        'sound_modifier': 'perfect_fifth',
        'sound_pattern': 'flowing_mid',
        'boundary_type': 'sharp',
        'connections': ['primary_motor_cortex', 'secondary_somatosensory_cortex']
    },
    'secondary_somatosensory_cortex': {
        'parent': 'parietal_cortex',
        'proportion': 0.20,
        'function': 'complex_tactile_processing',
        'brodmann_areas': [40, 43],
        'wave_frequency_hz': 10.1,
        'default_wave': 'alpha',
        'color': ColorShades.PARIETAL_GREEN,
        'sound_modifier': 'major_third',
        'sound_pattern': 'harmonic_complex',
        'boundary_type': 'gradual',
        'connections': ['primary_somatosensory_cortex', 'insula']
    },
    'posterior_parietal_cortex': {
        'parent': 'parietal_cortex',
        'proportion': 0.25,
        'function': 'spatial_attention_navigation',
        'brodmann_areas': [7],
        'wave_frequency_hz': 9.8,
        'default_wave': 'alpha',
        'color': ColorShades.PARIETAL_GREEN,
        'sound_modifier': 'minor_seventh',
        'sound_pattern': 'ethereal_light',
        'boundary_type': 'diffuse',
        'connections': ['occipital_cortex', 'frontal_cortex']
    },
    'superior_parietal_lobule': {
        'parent': 'parietal_cortex',
        'proportion': 0.15,
        'function': 'spatial_processing_reaching',
        'brodmann_areas': [5, 7],
        'wave_frequency_hz': 10.3,
        'default_wave': 'alpha',
        'color': ColorShades.PARIETAL_GREEN,
        'sound_modifier': 'perfect_octave',
        'sound_pattern': 'resonant_complex',
        'boundary_type': 'gradual',
        'connections': ['primary_motor_cortex', 'occipital_cortex']
    },
    'inferior_parietal_lobule': {
        'parent': 'parietal_cortex',
        'proportion': 0.10,
        'function': 'language_mathematics_integration',
        'brodmann_areas': [39, 40],
        'wave_frequency_hz': 9.9,
        'default_wave': 'alpha',
        'color': ColorShades.PARIETAL_GREEN,
        'sound_modifier': 'major_sixth',
        'sound_pattern': 'melodic_varied',
        'boundary_type': 'oscillating',
        'connections': ['temporal_cortex', 'frontal_cortex']
    },
    
    # TEMPORAL CORTEX Sub-regions  
    'primary_auditory_cortex': {
        'parent': 'temporal_cortex',
        'proportion': 0.20,
        'function': 'basic_sound_processing',
        'brodmann_areas': [41, 42],
        'wave_frequency_hz': 9.9,
        'default_wave': 'theta',
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_modifier': 'perfect_octave',
        'sound_pattern': 'staccato_low',
        'boundary_type': 'sharp',
        'connections': ['superior_temporal_gyrus']
    },
    'wernicke_area': {
        'parent': 'temporal_cortex', 
        'proportion': 0.15,
        'function': 'language_comprehension',
        'brodmann_areas': [22],
        'wave_frequency_hz': 9.5,
        'default_wave': 'theta',
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_modifier': 'major_sixth',
        'sound_pattern': 'harmonic_complex',
        'boundary_type': 'gradual',
        'connections': ['broca_area', 'inferior_parietal_lobule']
    },
    'superior_temporal_gyrus': {
        'parent': 'temporal_cortex',
        'proportion': 0.20,
        'function': 'complex_auditory_processing',
        'brodmann_areas': [22, 41, 42],
        'wave_frequency_hz': 9.2,
        'default_wave': 'theta',
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_modifier': 'perfect_fourth',
        'sound_pattern': 'flowing_mid',
        'boundary_type': 'gradual',
        'connections': ['primary_auditory_cortex', 'frontal_cortex']
    },
    'middle_temporal_gyrus': {
        'parent': 'temporal_cortex',
        'proportion': 0.15,
        'function': 'semantic_processing',
        'brodmann_areas': [21],
        'wave_frequency_hz': 8.8,
        'default_wave': 'theta',
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_modifier': 'minor_third',
        'sound_pattern': 'melodic_varied',
        'boundary_type': 'diffuse',
        'connections': ['inferior_temporal_gyrus', 'frontal_cortex']
    },
    'inferior_temporal_gyrus': {
        'parent': 'temporal_cortex',
        'proportion': 0.15,
        'function': 'object_face_recognition',
        'brodmann_areas': [20],
        'wave_frequency_hz': 8.5,
        'default_wave': 'theta',
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_modifier': 'major_third',
        'sound_pattern': 'resonant_complex',
        'boundary_type': 'fractal',
        'connections': ['occipital_cortex', 'hippocampus']
    },
    'hippocampus': {
        'parent': 'temporal_cortex',
        'proportion': 0.10,
        'function': 'memory_formation_spatial_navigation',
        'brodmann_areas': ['hippocampal_formation'],
        'wave_frequency_hz': 6.5,
        'default_wave': 'theta',
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_modifier': 'octave_down',
        'sound_pattern': 'ethereal_light',
        'boundary_type': 'diffuse',
        'connections': ['parahippocampal_gyrus', 'frontal_cortex']
    },
    'parahippocampal_gyrus': {
        'parent': 'temporal_cortex',
        'proportion': 0.05,
        'function': 'memory_context_scene_processing',
        'brodmann_areas': [36, 37],
        'wave_frequency_hz': 6.8,
        'default_wave': 'theta',
        'color': ColorShades.TEMPORAL_GOLD,
        'sound_modifier': 'major_seventh',
        'sound_pattern': 'staccato_high',
        'boundary_type': 'oscillating',
        'connections': ['hippocampus', 'occipital_cortex']
    },
    
    # OCCIPITAL CORTEX Sub-regions
    'primary_visual_cortex': {
        'parent': 'occipital_cortex',
        'proportion': 0.40,
        'function': 'edge_orientation_basic_vision',
        'brodmann_areas': [17],
        'wave_frequency_hz': 11.5,
        'default_wave': 'alpha',
        'color': ColorShades.OCCIPITAL_PURPLE,
        'sound_modifier': 'major_third',
        'sound_pattern': 'staccato_low',
        'boundary_type': 'sharp',
        'connections': ['secondary_visual_cortex']
    },
    'secondary_visual_cortex': {
        'parent': 'occipital_cortex',
        'proportion': 0.25,
        'function': 'visual_feature_integration',
        'brodmann_areas': [18],
        'wave_frequency_hz': 11.0,
        'default_wave': 'alpha',
        'color': ColorShades.OCCIPITAL_PURPLE,
        'sound_modifier': 'perfect_fifth',
        'sound_pattern': 'flowing_mid',
        'boundary_type': 'gradual',
        'connections': ['primary_visual_cortex', 'visual_area_v3', 'visual_area_v4']
    },
    'visual_area_v3': {
        'parent': 'occipital_cortex',
        'proportion': 0.15,
        'function': 'motion_form_processing',
        'brodmann_areas': [19],
        'wave_frequency_hz': 10.7,
        'default_wave': 'alpha',
        'color': ColorShades.OCCIPITAL_PURPLE,
        'sound_modifier': 'minor_seventh',
        'sound_pattern': 'rhythmic_sustained',
        'boundary_type': 'diffuse',
        'connections': ['secondary_visual_cortex', 'visual_area_v5_mt']
    },
    'visual_area_v4': {
        'parent': 'occipital_cortex',
        'proportion': 0.15,
        'function': 'color_form_processing',
        'brodmann_areas': [19],
        'wave_frequency_hz': 10.9,
        'default_wave': 'alpha',
        'color': ColorShades.OCCIPITAL_PURPLE,
        'sound_modifier': 'harmonic_fifth',
        'sound_pattern': 'harmonic_complex',
        'boundary_type': 'fractal',
        'connections': ['secondary_visual_cortex', 'inferior_temporal_gyrus']
    },
    'visual_area_v5_mt': {
        'parent': 'occipital_cortex',
        'proportion': 0.05,
        'function': 'motion_detection',
        'brodmann_areas': [19],
        'wave_frequency_hz': 10.2,
        'default_wave': 'alpha',
        'color': ColorShades.OCCIPITAL_PURPLE,
        'sound_modifier': 'perfect_octave',
        'sound_pattern': 'staccato_high',
        'boundary_type': 'oscillating',
        'connections': ['visual_area_v3', 'posterior_parietal_cortex']
    },
    
    # LIMBIC SYSTEM Sub-regions
    'amygdala': {
        'parent': 'limbic_system',
        'proportion': 0.25,
        'function': 'fear_emotion_threat_detection',
        'brodmann_areas': ['amygdaloid_complex'],
        'wave_frequency_hz': 7.2,
        'default_wave': 'theta',
        'color': ColorShades.LIMBIC_RED,
        'sound_modifier': 'minor_third',
        'sound_pattern': 'staccato_low',
        'boundary_type': 'diffuse',
        'connections': ['hippocampus', 'prefrontal_cortex', 'brainstem']
    },
    'anterior_cingulate': {
        'parent': 'limbic_system',
        'proportion': 0.25,
        'function': 'emotion_regulation_conflict_monitoring',
        'brodmann_areas': [24, 32],
        'wave_frequency_hz': 7.5,
        'default_wave': 'theta',
        'color': ColorShades.LIMBIC_RED,
        'sound_modifier': 'major_seventh',
        'sound_pattern': 'flowing_mid',
        'boundary_type': 'gradual',
        'connections': ['prefrontal_cortex', 'amygdala']
    },
    'posterior_cingulate': {
        'parent': 'limbic_system',
        'proportion': 0.20,
        'function': 'self_awareness_default_mode',
        'brodmann_areas': [23, 31],
        'wave_frequency_hz': 7.8,
        'default_wave': 'theta',
        'color': ColorShades.LIMBIC_RED,
        'sound_modifier': 'perfect_fifth',
        'sound_pattern': 'ethereal_light',
        'boundary_type': 'diffuse',
        'connections': ['prefrontal_cortex', 'parietal_cortex']
    },
    'insula': {
        'parent': 'limbic_system',
        'proportion': 0.20,
        'function': 'interoception_body_awareness',
        'brodmann_areas': ['insular_cortex'],
        'wave_frequency_hz': 8.1,
        'default_wave': 'theta',
        'color': ColorShades.LIMBIC_RED,
        'sound_modifier': 'harmonic_third',
        'sound_pattern': 'resonant_complex',
        'boundary_type': 'oscillating',
        'connections': ['secondary_somatosensory_cortex', 'amygdala']
    },
    'orbitofrontal_cortex': {
        'parent': 'limbic_system',
        'proportion': 0.10,
        'function': 'reward_decision_making',
        'brodmann_areas': [11, 47],
        'wave_frequency_hz': 8.5,
        'default_wave': 'theta',
        'color': ColorShades.LIMBIC_RED,
        'sound_modifier': 'major_sixth',
        'sound_pattern': 'melodic_varied',
        'boundary_type': 'fractal',
        'connections': ['amygdala', 'prefrontal_cortex']
    },
    
    # CEREBELLUM Sub-regions
    'cerebellar_cortex': {
        'parent': 'cerebellum',
        'proportion': 0.40,
        'function': 'motor_learning_fine_tuning',
        'brodmann_areas': ['cerebellar_cortex'],
        'wave_frequency_hz': 9.4,
        'default_wave': 'alpha',
        'color': ColorShades.CEREBELLUM_TEAL,
        'sound_modifier': 'perfect_fourth',
        'sound_pattern': 'rhythmic_sustained',
        'boundary_type': 'fractal',
        'connections': ['deep_cerebellar_nuclei', 'primary_motor_cortex']
    },
    'deep_cerebellar_nuclei': {
        'parent': 'cerebellum',
        'proportion': 0.20,
        'function': 'motor_output_coordination',
        'brodmann_areas': ['deep_nuclei'],
        'wave_frequency_hz': 9.1,
        'default_wave': 'alpha',
        'color': ColorShades.CEREBELLUM_TEAL,
        'sound_modifier': 'major_sixth',
        'sound_pattern': 'harmonic_complex',
        'boundary_type': 'sharp',
        'connections': ['cerebellar_cortex', 'brainstem']
    },
    'vestibulocerebellum': {
        'parent': 'cerebellum',
        'proportion': 0.15,
        'function': 'balance_eye_movements',
        'brodmann_areas': ['flocculonodular_lobe'],
        'wave_frequency_hz': 8.9,
        'default_wave': 'alpha',
        'color': ColorShades.CEREBELLUM_TEAL,
        'sound_modifier': 'minor_seventh',
        'sound_pattern': 'staccato_high',
        'boundary_type': 'oscillating',
        'connections': ['brainstem']
    },
    'spinocerebellum': {
        'parent': 'cerebellum',
        'proportion': 0.15,
        'function': 'posture_locomotion',
        'brodmann_areas': ['vermis_intermediate'],
        'wave_frequency_hz': 9.0,
        'default_wave': 'alpha',
        'color': ColorShades.CEREBELLUM_TEAL,
        'sound_modifier': 'perfect_octave',
        'sound_pattern': 'flowing_mid',
        'boundary_type': 'gradual',
        'connections': ['primary_motor_cortex', 'brainstem']
    },
    'cerebrocerebellum': {
        'parent': 'cerebellum',
        'proportion': 0.10,
        'function': 'cognitive_functions_planning',
        'brodmann_areas': ['lateral_hemispheres'],
        'wave_frequency_hz': 9.2,
        'default_wave': 'alpha',
        'color': ColorShades.CEREBELLUM_TEAL,
        'sound_modifier': 'major_third',
        'sound_pattern': 'ethereal_light',
        'boundary_type': 'diffuse',
        'connections': ['prefrontal_cortex', 'parietal_cortex']
    },
    
    # BRAINSTEM Sub-regions
    'midbrain': {
        'parent': 'brainstem',
        'proportion': 0.25,
        'function': 'eye_movement_visual_auditory_reflexes',
        'brodmann_areas': ['midbrain_structures'],
        'wave_frequency_hz': 3.1,
        'default_wave': 'delta',
        'color': ColorShades.BRAINSTEM_GRAY,
        'sound_modifier': 'perfect_fifth',
        'sound_pattern': 'staccato_low',
        'boundary_type': 'oscillating',
        'connections': ['occipital_cortex', 'temporal_cortex']
    },
    'pons': {
        'parent': 'brainstem',
        'proportion': 0.35,
        'function': 'sleep_arousal_facial_sensation',
        'brodmann_areas': ['pontine_structures'],
        'wave_frequency_hz': 2.7,
        'default_wave': 'delta',
        'color': ColorShades.BRAINSTEM_GRAY,
        'sound_modifier': 'minor_third',
        'sound_pattern': 'rhythmic_sustained',
        'boundary_type': 'gradual',
        'connections': ['cerebellum', 'reticular_formation']
    },
    'medulla': {
        'parent': 'brainstem',
        'proportion': 0.25,
        'function': 'breathing_heart_rate_blood_pressure',
        'brodmann_areas': ['medullary_structures'],
        'wave_frequency_hz': 1.8,
        'default_wave': 'delta',
        'color': ColorShades.BRAINSTEM_GRAY,
        'sound_modifier': 'octave_down',
        'sound_pattern': 'flowing_mid',
        'boundary_type': 'sharp',
        'connections': ['spinal_cord']
    },
    'reticular_formation': {
        'parent': 'brainstem',
        'proportion': 0.15,
        'function': 'arousal_sleep_wake_consciousness',
        'brodmann_areas': ['reticular_structures'],
        'wave_frequency_hz': 2.2,
        'default_wave': 'delta',
        'color': ColorShades.BRAINSTEM_GRAY,
        'sound_modifier': 'harmonic_fifth',
        'sound_pattern': 'resonant_complex',
        'boundary_type': 'diffuse',
        'connections': ['frontal_cortex', 'limbic_system']
    }
}

# ---- Sound Modifiers (COMPLETE) ----
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

# ---- Brain Wave Types (COMPLETE) ----
BRAIN_WAVE_TYPES = {
    'delta': {'frequency_range': (0.5, 4.0), 'state': 'deep_sleep'},
    'theta': {'frequency_range': (4.0, 8.0), 'state': 'drowsy_meditation'},
    'alpha': {'frequency_range': (8.0, 13.0), 'state': 'relaxed_aware'},
    'beta': {'frequency_range': (13.0, 30.0), 'state': 'alert_focused'},
    'gamma': {'frequency_range': (30.0, 100.0), 'state': 'high_cognition'},
    'lambda': {'frequency_range': (100.0, 200.0), 'state': 'insight_transcendence'}
}

# ---- Sound Patterns for sub regions ----
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

# ---- Boundary Definitions (COMPLETE) ----
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
    ('frontal_cortex', 'parietal_cortex'): 'gradual',
    ('frontal_cortex', 'temporal_cortex'): 'gradual',
    ('parietal_cortex', 'temporal_cortex'): 'gradual',
    ('parietal_cortex', 'occipital_cortex'): 'sharp',
    ('temporal_cortex', 'occipital_cortex'): 'sharp',
    ('limbic_system', 'frontal_cortex'): 'diffuse',
    ('limbic_system', 'parietal_cortex'): 'diffuse',
    ('limbic_system', 'temporal_cortex'): 'diffuse',
    ('limbic_system', 'occipital_cortex'): 'oscillating',
    ('cerebellum', 'brainstem'): 'sharp',
    ('cerebellum', 'occipital_cortex'): 'fractal',
    ('brainstem', 'limbic_system'): 'fractal',
    # Default boundary type if not specified
    'default': 'gradual'
}

# ---- Global Constants ----
# Standard 3D Grid Dimensions (Updated to match brain_structure.py)
GRID_DIMENSIONS = (256, 256, 256)

# Golden Ratio for sacred proportions
GOLDEN_RATIO = 1.618033988749895

# Base frequency for entire brain
BASE_BRAIN_FREQUENCY = 7.83  # Earth's Schumann resonance

# Standard tolerance for frequency matching
FREQUENCY_MATCH_TOLERANCE = 0.05

# Energy distribution constants
ENERGY_DISTRIBUTION = {
    'frontal_cortex': 0.35,
    'parietal_cortex': 0.20,
    'temporal_cortex': 0.18,
    'occipital_cortex': 0.12,
    'limbic_system': 0.08,
    'cerebellum': 0.10,
    'brainstem': 0.07
}

# ---- Helper Functions ----
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

def get_region_connections(region_name):
    """Get anatomical connections for a region."""
    if region_name in SUB_REGIONS:
        return SUB_REGIONS[region_name].get('connections', [])
    return []

def get_brodmann_areas(region_name):
    """Get Brodmann areas for a region."""
    if region_name in SUB_REGIONS:
        return SUB_REGIONS[region_name].get('brodmann_areas', [])
    return []

def get_region_frequency(region_name):
    """Get the wave frequency for a region."""
    if region_name in SUB_REGIONS:
        return SUB_REGIONS[region_name]['wave_frequency_hz']
    elif region_name in MAJOR_REGIONS:
        return MAJOR_REGIONS[region_name]['wave_frequency_hz']
    return BASE_BRAIN_FREQUENCY

def get_region_wave_type(region_name):
    """Get the default brain wave type for a region."""
    if region_name in SUB_REGIONS:
        return SUB_REGIONS[region_name]['default_wave']
    elif region_name in MAJOR_REGIONS:
        return MAJOR_REGIONS[region_name]['default_wave']
    return 'alpha'

def get_sound_properties(region_name):
    """Get complete sound properties for a region."""
    if region_name in SUB_REGIONS:
        sub_region = SUB_REGIONS[region_name]
        parent = MAJOR_REGIONS[sub_region['parent']]
        return {
            'base_note': parent['sound_base_note'],
            'frequency_hz': sub_region['wave_frequency_hz'],
            'wave_type': sub_region['default_wave'],
            'sound_modifier': sub_region.get('sound_modifier', 'perfect_fifth'),
            'sound_pattern': sub_region.get('sound_pattern', 'flowing_mid'),
            'boundary_type': sub_region.get('boundary_type', 'gradual')
        }
    elif region_name in MAJOR_REGIONS:
        region = MAJOR_REGIONS[region_name]
        return {
            'base_note': region['sound_base_note'],
            'frequency_hz': region['wave_frequency_hz'],
            'wave_type': region['default_wave'],
            'sound_modifier': 'perfect_fifth',
            'sound_pattern': 'flowing_mid',
            'boundary_type': region.get('boundary_type', 'gradual')
        }
    return None

def validate_anatomical_structure():
    """Validate that the anatomical structure is consistent."""
    errors = []

    # Check that all sub-regions have valid parents
    for sub_region, data in SUB_REGIONS.items():
        parent = data['parent']
        if parent not in MAJOR_REGIONS:
            errors.append(f"Sub-region '{sub_region}' has invalid parent '{parent}'")

    # Check that proportions sum to approximately 1.0 for each major region
    for major_region in MAJOR_REGIONS.keys():
        sub_regions = get_sub_regions_for_parent(major_region)
        if sub_regions:
            total_proportion = sum(SUB_REGIONS[sr]['proportion'] for sr in sub_regions)
            if abs(total_proportion - 1.0) > 0.01:
                errors.append(f"Region '{major_region}' sub-region proportions sum to {total_proportion:.3f}, not 1.0")

    # Check that major region proportions sum to approximately 1.0
    total_major_proportion = sum(data['proportion'] for data in MAJOR_REGIONS.values())
    if abs(total_major_proportion - 1.0) > 0.01:
        errors.append(f"Major region proportions sum to {total_major_proportion:.3f}, not 1.0")

    # Check frequency alignment with brain wave types
    for region_name, region_data in SUB_REGIONS.items():
        wave_type = region_data.get('default_wave')
        frequency = region_data.get('wave_frequency_hz')
        if wave_type and frequency:
            wave_range = BRAIN_WAVE_TYPES[wave_type]['frequency_range']
            if not (wave_range[0] <= frequency <= wave_range[1]):
                errors.append(f"Region '{region_name}' frequency {frequency}Hz not in {wave_type} range {wave_range}")

    return errors
    return []

# Validate on import
_validation_errors = validate_anatomical_structure()
if _validation_errors:
    import warnings
    for error in _validation_errors:
        warnings.warn(f"Anatomical structure validation error: {error}")

# Export
__all__ = [
    'MAJOR_REGIONS', 'SUB_REGIONS', 'ColorShades', 'SOUND_MODIFIERS', 
    'BRAIN_WAVE_TYPES', 'SOUND_PATTERNS', 'BOUNDARY_TYPES', 'REGION_BOUNDARIES',
    'ENERGY_DISTRIBUTION', 'BASE_BRAIN_FREQUENCY', 'FREQUENCY_MATCH_TOLERANCE',
    'get_sub_regions_for_parent', 'get_all_region_names', 'get_parent_region',
    'get_region_connections', 'get_brodmann_areas', 'get_region_frequency',
    'get_region_wave_type', 'get_sound_properties', 'validate_anatomical_structure'
]