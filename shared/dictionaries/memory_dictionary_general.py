MEMORY_TYPES = {
        'ephemeral': {
            'memory_type_id': 1,
            'memory_frequency_hz': 23.7,
            'decay_rate': 0.1,
            'preferred_storage_duration_hours': 1,
            'typical_content': 'temporary thoughts, immediate sensory data'
        },
        'short_term_working': {
                'memory_type_id': 2,
                'memory_frequency_hz': 18.2,
                'decay_rate': 0.05,
                'preferred_storage_duration_hours': 24,
                'typical_content': "daily experiences, active processing"
            },
        'transitional': {
            'memory_type_id': 3,
            'memory_frequency_hz': 14.8,
            'decay_rate': 0.02,
            'preferred_storage_duration_hours': 168,
            'typical_content': 'learning consolidation, pattern formation'
        },
        'long_term_semantic': {
            'memory_type_id': 4,
            'memory_frequency_hz': 12.1,
            'decay_rate': 0.01,
            'preferred_storage_duration_hours': 8760,
            'typical_content': "concepts, knowledge, meanings"
        },
        'long_term_episodic': {
            'memory_type_id': 5,
            'memory_frequency_hz': 9.8,
            'decay_rate': 0.005,
            'preferred_storage_duration_hours': 17520,
            'typical_content': "personal experiences, events, narratives"
        },
        'procedural': {
            'memory_type_id': 6,
            'memory_frequency_hz': 7.4,
            'decay_rate': 0.002,
            'preferred_storage_duration_hours': 35040,
            'typical_content': "skills, habits, motor patterns"
        },
        'emotional_imprint': {
            'memory_type_id': 7,
            'memory_frequency_hz': 5.9,
            'decay_rate': 0.001,
            'preferred_storage_duration_hours': 87600,
            'typical_content': "emotional associations, trauma, joy"
        },
        'core_identity': {
            'memory_type_id': 8,
            'memory_frequency_hz': 3.2,
            'decay_rate': 0.0001,
            'preferred_storage_duration_hours': 525600,
            'typical_content': "fundamental self-concepts, identity aspects"
        },
        'soul_essence': {
            'memory_type_id': 9,
            'memory_frequency_hz': 1.1,
            'decay_rate': 0.00001,
            'preferred_storage_duration_hours': 8760000,
            'typical_content': "sephiroth aspects, spiritual traits, soul nature, spiritual journey"
        },
        'survival': {
            'memory_type_id': 10,
            'memory_frequency_hz': 0.5,
            'decay_rate': 0.000001,
            'preferred_storage_duration_hours': 17520000,
            'typical_content': "instincts, survival mechanisms, primal responses, basic needs satisfaction"
        }
    }

# Signal patterns for memory encoding field/energy dynamics 
SIGNAL_PATTERNS = {
    'simple_continuous': {
        'amplitude_range': (0.1, 0.3),
        'frequency_modifier': 1.0,
        'waveform': 'sine',
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



# this needs to be corrected so we can position the sensory data around its memory fragment - do we rather do this as a radius of 360 around the central fragment instead of using a number?
# this positions the sensory data around its memory fragment using a radius and angle system
DODECASTOR_ROUTING_MAP = {
    'auditory_node_position': {'radius': 1.0, 'angle': 0},          # 0 degrees
    'visual_node_position': {'radius': 1.0, 'angle': 60},           # 60 degrees
    'verbal_node_position': {'radius': 1.0, 'angle': 120},          # 120 degrees
    'psychic_node_position': {'radius': 1.0, 'angle': 180},         # 180 degrees
    'emotional_node_position': {'radius': 1.0, 'angle': 240},       # 240 degrees
    'energetic_node_position': {'radius': 1.0, 'angle': 300},       # 300 degrees
}