# --- memory_definitions.py V8 ---
"""
Memory definitions for subconscious world map and conscious thought processes.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

# Memory type definitions and their attributes.
MEMORY_TYPES = {
    1: {
        "memory_type_id": 1,
        "name": "ephemeral",
        "base_frequency_hz": 23.7,
        "decay_rate": 0.1,
        "preferred_storage_duration_hours": 1,
        "typical_content": "temporary thoughts, immediate sensory data"
    },
    2: {
        "memory_type_id": 2,
        "name": "short_term_working",
        "base_frequency_hz": 18.2,
        "decay_rate": 0.05,
        "preferred_storage_duration_hours": 24,
        "typical_content": "daily experiences, active processing"
    },
    3: {
        "memory_type_id": 3,
        "name": "transitional",
        "base_frequency_hz": 14.8,
        "decay_rate": 0.02,
        "preferred_storage_duration_hours": 168,
        "typical_content": "learning consolidation, pattern formation"
    },
    4: {
        "memory_type_id": 4,
        "name": "long_term_semantic",
        "base_frequency_hz": 12.1,
        "decay_rate": 0.01,
        "preferred_storage_duration_hours": 8760,
        "typical_content": "concepts, knowledge, meanings"
    },
    5: {
        "memory_type_id": 5,
        "name": "long_term_episodic",
        "base_frequency_hz": 9.8,
        "decay_rate": 0.005,
        "preferred_storage_duration_hours": 17520,
        "typical_content": "personal experiences, events, narratives"
    },
    6: {
        "memory_type_id": 6,
        "name": "procedural",
        "base_frequency_hz": 7.4,
        "decay_rate": 0.002,
        "preferred_storage_duration_hours": 35040,
        "typical_content": "skills, habits, motor patterns"
    },
    7: {
        "memory_type_id": 7,
        "name": "emotional_imprint",
        "base_frequency_hz": 5.9,
        "decay_rate": 0.001,
        "preferred_storage_duration_hours": 87600,
        "typical_content": "emotional associations, trauma, joy"
    },
    8: {
        "memory_type_id": 8,
        "name": "core_identity",
        "base_frequency_hz": 3.2,
        "decay_rate": 0.0001,
        "preferred_storage_duration_hours": 525600,
        "typical_content": "fundamental self-concepts, identity aspects"
    },
    9: {
        "memory_type_id": 9,
        "name": "soul_essence",
        "base_frequency_hz": 1.1,
        "decay_rate": 0.00001,
        "preferred_storage_duration_hours": 8760000,
        "typical_content": "sephiroth aspects, spiritual traits, soul nature"
    }
}

# Signal patterns for memory encoding
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

# Brain States and their characteristics to be used as modifiers for memory encoding and thought processing
BRAIN_STATES = {
    # Waking States with Alertness Levels
    'hyper_alert': {
        'brain_state_id': 1,
        'name': 'hyper_alert',
        'description': 'Heightened awareness, danger response, or intense focus',
        'dominant_frequency_range': (20, 30),  # High Beta waves
        'processing_speed_modifier': 1.3,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.8,
        'default_processing_intensity': 0.9
    },
    'focused': {
        'brain_state_id': 2,
        'name': 'focused',
        'description': 'Deliberate attention and concentration',
        'dominant_frequency_range': (15, 20),  # Beta waves
        'processing_speed_modifier': 1.2,
        'pattern_sensitivity': 0.8,
        'emotional_sensitivity': 0.6,
        'default_processing_intensity': 0.8
    },
    'relaxed_alert': {
        'brain_state_id': 3,
        'name': 'relaxed_alert',
        'description': 'Calm but attentive state',
        'dominant_frequency_range': (12, 15),  # Low Beta waves
        'processing_speed_modifier': 1.0,
        'pattern_sensitivity': 0.7,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.7
    },
    'learning': {
        'brain_state_id': 4,
        'name': 'learning',
        'description': 'Active information acquisition and processing',
        'dominant_frequency_range': (10, 14),  # Beta-Alpha boundary
        'processing_speed_modifier': 1.1,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.6,
        'default_processing_intensity': 0.8
    },
    'autopilot': {
        'brain_state_id': 5,
        'name': 'autopilot',
        'description': 'Routine tasks with minimal conscious attention',
        'dominant_frequency_range': (8, 12),  # Alpha waves
        'processing_speed_modifier': 0.8,
        'pattern_sensitivity': 0.5,
        'emotional_sensitivity': 0.4,
        'default_processing_intensity': 0.5
    },
    'drowsy': {
        'brain_state_id': 6,
        'name': 'drowsy',
        'description': 'Reduced alertness, tired but awake',
        'dominant_frequency_range': (7, 10),  # Alpha-Theta boundary
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.4,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.4
    },
    
    # Sleep Cycle States
    'sleep_onset': {
        'brain_state_id': 7,
        'name': 'sleep_onset',
        'description': 'Transition from wakefulness to sleep',
        'dominant_frequency_range': (4, 8),  # Theta waves
        'processing_speed_modifier': 0.5,
        'pattern_sensitivity': 0.6,
        'emotional_sensitivity': 0.7,
        'default_processing_intensity': 0.4
    },
    'light_sleep_n1': {
        'brain_state_id': 8,
        'name': 'light_sleep_n1',
        'description': 'First stage of NREM sleep',
        'dominant_frequency_range': (4, 7),  # Theta waves
        'processing_speed_modifier': 0.4,
        'pattern_sensitivity': 0.5,
        'emotional_sensitivity': 0.6,
        'default_processing_intensity': 0.3
    },
    'light_sleep_n2': {
        'brain_state_id': 9,
        'name': 'light_sleep_n2',
        'description': 'Second stage of NREM sleep with sleep spindles',
        'dominant_frequency_range': (12, 14),  # Sleep spindles
        'processing_speed_modifier': 0.3,
        'pattern_sensitivity': 0.4,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.3
    },
    'deep_sleep_n3': {
        'brain_state_id': 10,
        'name': 'deep_sleep_n3',
        'description': 'Slow wave sleep, deep NREM',
        'dominant_frequency_range': (0.5, 4),  # Delta waves
        'processing_speed_modifier': 0.2,
        'pattern_sensitivity': 0.3,
        'emotional_sensitivity': 0.2,
        'default_processing_intensity': 0.2
    },
    'rem_light': {
        'brain_state_id': 11,
        'name': 'rem_light',
        'description': 'Early REM sleep with lower intensity',
        'dominant_frequency_range': (4, 8),  # Theta waves
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.7,
        'emotional_sensitivity': 0.8,
        'default_processing_intensity': 0.5
    },
    'rem_intense': {
        'brain_state_id': 12,
        'name': 'rem_intense',
        'description': 'Deep REM sleep with vivid dreaming',
        'dominant_frequency_range': (20, 40),  # High frequency mixed
        'processing_speed_modifier': 0.7,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.9,
        'default_processing_intensity': 0.6
    },
    
    # Altered Consciousness States
    'meditation_light': {
        'brain_state_id': 13,
        'name': 'meditation_light',
        'description': 'Early stage meditation with relaxed awareness',
        'dominant_frequency_range': (8, 12),  # Alpha waves
        'processing_speed_modifier': 0.7,
        'pattern_sensitivity': 0.6,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.4
    },
    'meditation_deep': {
        'brain_state_id': 14,
        'name': 'meditation_deep',
        'description': 'Deep meditation with heightened awareness',
        'dominant_frequency_range': (4, 7),  # Theta waves
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.8,
        'emotional_sensitivity': 0.7,
        'default_processing_intensity': 0.5
    },
    'liminal_hypnagogic': {
        'brain_state_id': 15,
        'name': 'liminal_hypnagogic',
        'description': 'Transitional state between wakefulness and sleep',
        'dominant_frequency_range': (4, 8),  # Theta waves
        'processing_speed_modifier': 0.5,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.8,
        'default_processing_intensity': 0.6
    },
    'liminal_hypnopompic': {
        'brain_state_id': 16,
        'name': 'liminal_hypnopompic',
        'description': 'Transitional state between sleep and wakefulness',
        'dominant_frequency_range': (4, 10),  # Theta-Alpha mix
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.8,
        'emotional_sensitivity': 0.7,
        'default_processing_intensity': 0.5
    }
}

# do we rather do this as a radius of 360 around the central fragment?
DODECASTOR_ROUTING_MAP = {
    'auditory_node_position': '1',
    'visual_node_position': '2',
    'verbal_node_position': '3',
    'psychic_node_position': '4',
    'emotional_node_position': '5',
    'energetic_node_position': '6',
    'resonance_node_position': '7',
    'category_similarity_node_position': '8',
    'temporal_sequence_node_position': '9',
    'spatial_similarity_node_position': '10',
}



SENSORY_MATRIX = {
    'visual': {
        # Core Identification
        'visual_id': None,
        'semantic_world_process_map_id': None,
        
        # Image Source and Format
        'image_type': None,  # 'photograph', 'illustration', 'visualization', 'dream', 'hallucination', etc.
        'image_path': 'stage_1/brain_formation/brain/images/',
        'image_format': '.png',
        'image_source': None,  # 'camera', 'generated', 'memory', 'imagination', etc.
        'capture_time': None,  # When the image was captured/generated
        # 'image_classification': None,  # General classification of image content
        
        # Technical Image Properties
        'resolution': None,  # Image resolution (width x height)
        'bit_depth': None,  # Color bit depth
        'aspect_ratio': None,  # Width to height ratio
        'file_size': None,  # Size in bytes
        # 'compression_level': None,  # Level of compression if applicable
        
        # Visual Quality Parameters
        'color_palette': None,  # Dominant colors in the image
        'contrast': 0.0,  # Level of contrast (0.0-1.0)
        'brightness': 0.0,  # Overall brightness (0.0-1.0)
        'saturation': 0.0,  # Color saturation (0.0-1.0)
        'hue': 0.0,  # Dominant hue (0-360 degrees)
        'sharpness': 0.0,  # Image sharpness/focus (0.0-1.0)
        'noise_level': 0.0,  # Amount of visual noise (0.0-1.0)
        # 'blur_amount': 0.0,  # Amount of blur (0.0-1.0)
        # 'dynamic_range': 0.0,  # Range from darkest to brightest (0.0-1.0)
        
        # # Color Analysis
        # 'color_distribution': {},  # Distribution of colors by percentage
        # 'color_harmony': None,  # Type of color harmony (complementary, analogous, etc.)
        # 'color_temperature': 0.0,  # Warm to cool temperature
        # 'color_mood': None,  # Mood conveyed by color scheme
        # 'color_symbolism': {},  # Symbolic meanings of colors present
        # 'color_psychology_effects': [],  # Psychological effects of color scheme
        
        # Light Properties
        'lighting_direction': None,  # Direction of main light source
        'lighting_type': None,  # 'natural', 'artificial', 'mixed', etc.
        # 'lighting_quality': None,  # 'harsh', 'soft', 'diffuse', etc.
        'shadows_presence': 0.0,  # Presence of shadows (0.0-1.0)
        'shadow_direction': None,  # Direction of shadows
        # 'shadow_hardness': 0.0,  # Hardness of shadow edges (0.0-1.0)
        'highlights_presence': 0.0,  # Presence of highlights (0.0-1.0)
        # 'luminosity_pattern': None,  # Pattern of light distribution
        'reflections_presence': 0.0,  # Presence of reflections (0.0-1.0)
        
        # Spatial Properties
        # 'perspective_type': None,  # 'linear', 'atmospheric', 'isometric', etc.
        # 'depth_perception': 0.0,  # Sense of depth (0.0-1.0)
        'depth_map': None,  # Depth map if available
        'spatial_relationships': {},  # Relationships between objects in space
        'foreground_elements': [],  # Elements in the foreground
        'midground_elements': [],  # Elements in the midground
        'background_elements': [],  # Elements in the background
        # 'scale_indicators': [],  # Elements that indicate scale
        'vanishing_points': [],  # Vanishing points in perspective
        'horizon_line': None,  # Position of horizon line if present
        
        # Composition
        'composition_type': None,  # 'rule of thirds', 'golden ratio', 'symmetrical', etc.
        # 'visual_balance': 0.0,  # Balance of visual elements (0.0-1.0)
        # 'visual_weight_distribution': {},  # Distribution of visual weight
        'focal_points': [],  # Main points of visual interest
        'visual_flow': None,  # Eye movement patterns
        # 'framing_technique': None,  # How the subject is framed
        'negative_space_usage': 0.0,  # Empty space ratio
        # 'cropping': None,  # How the image is cropped
        
        # # Content Analysis
        # 'visual_content': None,  # General description of content
        # 'visual_description': None,  # Detailed description of what is seen
        # 'subject_matter': None,  # Main subject of the image
        # 'identified_visual_elements': [],  # Individual elements identified
        # 'scene_type': None,  # 'landscape', 'portrait', 'still life', etc.
        # 'scene_setting': None,  # Where the scene takes place
        # 'time_of_day': None,  # Time of day depicted
        # 'weather_conditions': None,  # Weather conditions depicted
        # 'season_depicted': None,  # Season depicted
        # 'cultural_indicators': [],  # Cultural elements present
        # 'historical_period': None,  # Historical period depicted if applicable
        
        # Object Recognition
        'objects_detected': [],  # Objects detected in the image
        'object_count': {},  # Count of each type of object
        # 'object_sizes': {},  # Relative sizes of objects
        'object_positions': {},  # Positions of objects
        'object_relationships': {},  # How objects relate to each other
        # 'object_occlusions': [],  # Objects partially hidden by others
        # 'object_states': {},  # States of objects (e.g., 'open', 'broken')
        
        # # Living Beings
        # 'people_present': False,  # Whether people are present
        # 'number_of_people': 0,  # Number of people if present
        # 'facial_expressions': [],  # Facial expressions of people
        # 'body_language': [],  # Body language of people
        # 'clothing_styles': [],  # Clothing styles of people
        # 'animals_present': False,  # Whether animals are present
        # 'animal_types': [],  # Types of animals if present
        # 'plant_life': [],  # Plant life present in the image
        
        # Pattern Recognition
        # 'visual_texture': None,  # Overall texture quality
        'texture_types': [],  # Types of textures present
        # 'texture_contrasts': 0.0,  # Contrast between textures (0.0-1.0)
        # 'pattern_presence': 0.0,  # Presence of patterns (0.0-1.0)
        # 'pattern_types': [],  # Types of patterns present
        # 'pattern_regularity': 0.0,  # Regularity of patterns (0.0-1.0)
        # 'visual_pattern_type': None,  # Specific type of visual pattern
        # 'pattern_symbolism': {},  # Symbolic meanings of patterns
        'geometric_shapes': [],  # Basic shapes detected
        'line_orientations': [],  # Major line directions
        'symmetry_detected': None,

        # Sacred Geometry (pattern matching)
        'golden_ratio_presence': 0.0,  # Mathematical detection
        'platonic_solids': [],  # If detected in image
        'fibonacci_patterns': [],  # Spiral detection
        'geometric_harmonics': [], 
        
        # # Movement and Time
        # 'implied_movement': 0.0,  # Sense of movement in still image (0.0-1.0)
        # 'movement_direction': None,  # Direction of implied movement
        # 'movement_speed': 0.0,  # Implied speed of movement (0.0-1.0)
        # 'temporal_indicators': [],  # Elements indicating time passage
        # 'narrative_moment': None,  # Moment in a narrative if applicable
        # 'before_moment': None,  # What might have happened before
        # 'after_moment': None,  # What might happen after
        
        # # Geometric Analysis
        # 'geometric_shape_identification': [],  # Geometric shapes identified
        # 'dominant_shapes': [],  # Dominant shapes in the image
        # 'shape_arrangements': [],  # How shapes are arranged
        # 'line_types': [],  # Types of lines (straight, curved, etc.)
        # 'line_directions': [],  # Directions of major lines
        # 'angular_relationships': [],  # Relationships between angles
        # 'symmetry_types': [],  # Types of symmetry present
        # 'proportional_relationships': [],  # Proportional relationships
        
        # # Sacred Geometry
        # 'sacred_geometry_identification': None,  # Sacred geometry identified
        # 'sacred_geometry_content': None,  # Content related to sacred geometry
        # 'sacred_geometry_associated': None,  # Associated sacred geometry
        # 'platonics_solid_identification': None,  # Platonic solids identified
        # 'platonics_solid_content': None,  # Content related to platonic solids
        # 'golden_ratio_presence': 0.0,  # Presence of golden ratio (0.0-1.0)
        # 'fibonacci_patterns': [],  # Fibonacci patterns identified
        # 'mandala_elements': [],  # Mandala-like elements
        
        # Movement Detection (for video)
        'motion_vectors': [],  # If video input
        'optical_flow': None,
        'temporal_changes': [],  # Changes between frames
        
        # Semantic Content
        'scene_type': None,  # Indoor/outdoor/abstract
        'identified_concepts': [],  # What the image contains
        'visual_symbols': [],  # Recognized symbols
        'text_detected': [],  # OCR results if applicable

        # # Symbolic and Conceptual
        # 'symbolism_present': 0.0,  # Presence of symbolism (0.0-1.0)
        # 'symbols_identified': [],  # Specific symbols identified
        # 'symbolic_meanings': {},  # Meanings of identified symbols
        # 'metaphorical_content': [],  # Metaphorical content
        # 'allegory_elements': [],  # Elements of allegory
        # 'archetypes_present': [],  # Archetypal elements present
        # 'visual_related_concepts': [],  # Concepts related to the visual
        # 'emotional_evocation': {},  # Emotions evoked by the image
        
        # Glyphs and Special Elements
        'glyphs_present': False,  # Glyphs present in the image
        # 'glyph_associated': None,  # Associated glyphs
        # 'glyph_presence': 0.0,  # Presence of glyphs (0.0-1.0)
        # 'glyph_interpretation': None,  # Interpretation of glyphs
        # 'sigils_present': [],  # Sigils present in the image
        # 'magical_symbols': [],  # Magical symbols present
        # 'religious_iconography': [],  # Religious iconography present
        
        # # Technical Visual Processing
        # 'edge_detection': 0.0,  # Edge detection strength (0.0-1.0)
        # 'feature_extraction': [],  # Features extracted from image
        # 'encoded_metadata_surface': None,  # Surface metadata encoding
        # 'encoded_metadata_hidden': None,  # Hidden metadata encoding
        # 'steganographic_content': None,  # Hidden content in the image
        # 'watermarks_present': False,  # Whether watermarks are present
        
        # # Visual Impact and Response
        # 'visual_impact_score': 0.0,  # Overall impact of the image (0.0-1.0)
        # 'attention_capture': 0.0,  # How well it captures attention (0.0-1.0)
        # 'memorability_score': 0.0,  # How memorable the image is (0.0-1.0)
        # 'aesthetic_rating': 0.0,  # Aesthetic quality rating (0.0-1.0)
        # 'emotional_response': None,  # Primary emotional response
        # 'cognitive_challenge': 0.0,  # Level of cognitive challenge (0.0-1.0)
        
        # # Processing Metadata
        # 'signal_pattern': None,  # Pattern of visual signal processing
        # 'processing_confidence': 0.0,  # Confidence in visual processing (0.0-1.0)
        # 'processing_time': 0.0,  # Time taken to process the visual
        # 'attention_distribution': {},  # Distribution of attention across image
        # 'interpretation_confidence': 0.0  # Confidence in interpretation (0.0-1.0)
    },
    'auditory': {
        # Core Identification
        'auditory_id': None,
        'semantic_world_process_map_id': None,
        # Audio Source
        'audio_source': None,  # 'speech', 'music', 'environmental', 'mixed'
        'capture_time': None,
        'duration': 0.0,
        
        # Technical Properties (measurable)
        'sample_rate': None,
        'bit_depth': None,
        'channels': None,  # Mono/stereo/surround
        'file_format': None,
        
        # Frequency Analysis
        'frequency_spectrum': {},  # FFT results
        'fundamental_frequency': None,  # F0 detection
        'harmonics': [],  # Harmonic series
        'spectral_centroid': 0.0,  # Brightness measure
        'spectral_rolloff': 0.0,  # Frequency cutoff
        
        # Amplitude Analysis
        'amplitude_envelope': [],  # Volume over time
        'peak_amplitude': 0.0,
        'rms_amplitude': 0.0,  # Average loudness
        'dynamic_range': 0.0,
        
        # Temporal Properties
        'onset_times': [],  # When sounds start
        'tempo': None,  # If rhythmic
        'beat_positions': [],  # Beat detection
        'rhythm_pattern': None,
        
        # Spatial Audio (if multichannel)
        'spatial_position': None,  # Sound source location
        'distance_cues': None,  # Near/far estimation
        'reverb_amount': 0.0,  # Environmental size cue
        
        # Speech Detection (if applicable)
        'speech_detected': False,
        'voice_characteristics': {
            'pitch_range': None,
            'speaking_rate': None,
            'voice_quality': None,  # Detected timbre
        },
        
        # Pattern Recognition
        'melodic_patterns': [],  # Musical phrases
        'rhythmic_patterns': [],  # Repeating rhythms
        'timbral_signatures': [],  # Instrument/source identification
    },
    'verbal': {
        # Core Identification
        'verbal_id': None,
        'semantic_world_process_map_id': None,
        
        # Communication Direction
        'communication_type': None,  # 'heard', 'spoken', 'internal_dialogue', 'read', 'written'
        # 'communication_direction': None,  # 'incoming', 'outgoing', 'internal'
        # 'communication_medium': None,  # 'in_person', 'phone', 'digital', 'broadcast', etc.
        
        # Content Structure
        'content': None,  # The actual text/speech
        'content_length': 0,  # Word/character count
        'content_structure': None,  # Paragraph/dialogue/list

                # Linguistic Analysis (subconscious parsing)
        'tokens': [],  # Tokenized content
        'parts_of_speech': {},  # POS tagging
        'syntax_tree': None,  # Parse tree if applicable
        'dependencies': [],  # Grammatical dependencies
        
        # Semantic Analysis
        'entities': [],  # Named entities
        'keywords': [],  # Key terms extracted
        'topics': [],  # Topic modeling results
        'semantic_embeddings': [],  # Vector representations
        
        # Phonetic Properties (if spoken)
        'phonemes': [],  # Phonetic transcription
        'stress_patterns': [],
        'intonation_contour': [],
        
        # Morphological Analysis
        'word_roots': {},
        'affixes': {},
        'compound_structures': [],
        
        # Contextual Meaning
        'ambiguous_terms': {},  # Words with multiple meanings
        'resolved_references': {},  # Pronoun resolution
        'implied_meanings': [],  # Subtext detection
        
        # Pattern Detection
        'linguistic_patterns': [],  # Recurring structures
        'stylistic_features': [],  # Writing style markers
        'register': None,  # Formal/informal/technical
    },

    #     # Speaker/Source Information
    #     'speaker_id': None,  # Identity of speaker if known
    #     'speaker_characteristics': {
    #         'gender': None,
    #         'approximate_age': None,
    #         'accent': None,
    #         'voice_quality': None,  # 'raspy', 'clear', 'deep', 'high-pitched', etc.
    #         'emotional_tone': None,  # Emotional quality of the voice
    #         'speaking_rate': 0.0,  # Words per minute or relative speed
    #         'volume': 0.0,  # Volume level (0.0-1.0)
    #         'familiarity': 0.0  # How familiar the voice is (0.0-1.0)
    #     },
        
    #     # Acoustic Properties
    #     'acoustic_properties': {
    #         'amplitude': 0.0,  # Overall loudness
    #         'pitch_average': 0.0,  # Average pitch in Hz
    #         'pitch_range': [0.0, 0.0],  # Range of pitch variation
    #         'timbre': None,  # Tonal quality
    #         'rhythm': None,  # Rhythmic pattern of speech
    #         'prosody': None,  # Intonation pattern
    #         'background_noise_level': 0.0,  # Level of background noise (0.0-1.0)
    #         'audio_clarity': 0.0,  # Clarity of the audio (0.0-1.0)
    #         'spatial_direction': None,  # Direction sound is coming from
    #         'reverberation': 0.0,  # Amount of echo/reverberation (0.0-1.0)
    #         'audio_distortions': []  # Any distortions in the audio
    #     },
        
    #     # Language Characteristics
    #     'language_type': None,  # 'natural', 'programming', 'formal', 'informal', etc.
    #     'language_name': None,  # English, Spanish, Python, etc.
    #     'language_variant': None,  # Dialect, regional variant
    #     'language_formality': 0.0,  # Level of formality (0.0-1.0)
    #     'language_complexity': 0.0,  # Complexity level (0.0-1.0)
    #     'language_style': None,  # 'academic', 'conversational', 'poetic', etc.
    #     'originating_language': None,  # Original language if translated
        
    #     # Content
    #     'language_content': None,  # The actual verbal content
    #     'content_length': 0,  # Length in words, characters, or time
    #     'content_format': None,  # Format of content (sentence, paragraph, etc.)
    #     'content_structure': None,  # Structure of content (narrative, Q&A, etc.)
    #     'content_purpose': None,  # Purpose of communication (inform, request, etc.)
    #     'content_topic': None,  # Main topic of communication
    #     'content_subtopics': [],  # Subtopics covered
    #     'content_keywords': [],  # Key terms in the content
        
    #     # Linguistic Analysis
    #     'linguistic_pattern_type': None,  # Type of linguistic pattern
    #     'linguistic_pattern': None,  # Specific linguistic pattern identified
    #     'identified_linguistic_elements': [],  # Linguistic elements identified
    #     'grammar_structure': None,  # Grammatical structure
    #     'sentence_types': [],  # Types of sentences used
    #     'grammar_rules': [],  # Grammar rules applied
    #     'punctuation_identified': [],  # Punctuation used
    #     'speech_parts': {},  # Distribution of parts of speech
    #     'tense_usage': {},  # Tenses used
    #     'voice_usage': None,  # Active/passive voice usage
    #     'modality': None,  # Indicative, subjunctive, imperative, etc.
        
    #     # Word-Level Analysis
    #     'vocabulary_level': 0.0,  # Sophistication of vocabulary (0.0-1.0)
    #     'word_frequency_pattern': {},  # Frequency of word usage
    #     'word_resonance_score': 0.0,  # Emotional resonance of words (0.0-1.0)
    #     'rare_words': [],  # Uncommon words used
    #     'jargon_terms': [],  # Specialized terminology used
    #     'neologisms': [],  # Newly coined terms
    #     'word_origins': {},  # Etymology of notable words
        
    #     # Morphological Analysis
    #     'morphemes_identified': [],  # Morphemes identified in content
    #     'root_words': {},  # Root words identified
    #     'prefixes': {},  # Prefixes used
    #     'suffixes': {},  # Suffixes used
    #     'combining_vowels': {},  # Combining vowels used
    #     'compound_words': [],  # Compound words identified
    #     'word_formation_patterns': [],  # Patterns of word formation
        
    #     # Semantic Analysis
    #     'semantic_fields': [],  # Semantic domains covered
    #     'semantic_roles': {},  # Semantic roles in sentences
    #     'semantic_relations': [],  # Relations between concepts
    #     'semantic_density': 0.0,  # Density of meaning (0.0-1.0)
    #     'ambiguity_level': 0.0,  # Level of ambiguity (0.0-1.0)
    #     'dictionary_definitions': {},  # Definitions of key terms
    #     'synonyms': {},  # Synonyms of key terms
    #     'antonyms': {},  # Antonyms of key terms
    #     'context_examples': [],  # Examples of usage in context
    #     'language_related_concepts': [],  # Concepts related to content
        
    #     # Phonetic Analysis
    #     'phonetic_pronunciation': {},  # Phonetic transcription
    #     'phonemes_used': [],  # Phonemes identified
    #     'syllable_structure': [],  # Structure of syllables
    #     'stress_patterns': [],  # Patterns of stress
    #     'phonological_processes': [],  # Phonological processes observed
    #     'accent_features': [],  # Features of accent
    #     'speech_impediments': [],  # Any speech impediments noted
        
    #     # Pragmatic Analysis
    #     'speech_acts': [],  # Types of speech acts (promise, request, etc.)
    #     'conversational_implicatures': [],  # Implied meanings
    #     'presuppositions': [],  # Assumed information
    #     'politeness_strategies': [],  # Strategies for politeness
    #     'cooperative_principles': [],  # Adherence to conversational maxims
    #     'discourse_markers': [],  # Markers of discourse structure
    #     'turn_taking_patterns': [],  # Patterns of conversational turns
        
    #     # Numerical/Symbolic Aspects
    #     'gematria_pattern': None,  # Numerical pattern in words
    #     'gematria_connections': [],  # Connections revealed through gematria
    #     'symbolic_associations': {},  # Symbolic meanings of words/phrases
    #     'numerological_significance': {},  # Numerological meanings
        
    #     # Metaphysical Connections
    #     'glyph_associated': None,  # Associated glyphs
    #     'sacred_geometry_associated': None,  # Associated sacred geometry
    #     'platonics_solid_associated': None,  # Associated platonic solids
    #     'chakra_associations': [],  # Associations with energy centers
    #     'elemental_associations': [],  # Associations with elements
        
    #     # Communication Effectiveness
    #     'clarity_score': 0.0,  # Clarity of communication (0.0-1.0)
    #     'persuasiveness_score': 0.0,  # Persuasiveness (0.0-1.0)
    #     'emotional_impact_score': 0.0,  # Emotional impact (0.0-1.0)
    #     'memorability_score': 0.0,  # How memorable the content is (0.0-1.0)
    #     'authenticity_score': 0.0,  # Perceived authenticity (0.0-1.0)
        
    #     # Response Indicators
    #     'response_required': False,  # Whether a response is expected
    #     'response_urgency': 0.0,  # Urgency of response needed (0.0-1.0)
    #     'response_options': [],  # Possible response options
    #     'response_constraints': [],  # Constraints on responses
        
    #     # Processing Metadata
    #     'processing_confidence': 0.0,  # Confidence in processing (0.0-1.0)
    #     'processing_difficulty': 0.0,  # Difficulty in processing (0.0-1.0)
    #     'misunderstanding_risk': 0.0,  # Risk of misunderstanding (0.0-1.0)
    #     'attention_required': 0.0  # Attention needed for processing (0.0-1.0)
    # },

    'psychic': {
        # Core Identification
        'psychic_id': None,
        'semantic_world_process_map_id': None,
        
        # Experience Classification
        'psychic_event_type': None,  # 'precognition', 'telepathy', 'clairvoyance', 'empathic', 'channeling', etc.
        'psychic_experience': None,  # Description of the psychic experience
        # 'psychic_intensity': 0.0,  # Intensity of the psychic experience (0.0-1.0)
        # 'psychic_clarity': 0.0,  # Clarity of the psychic information (0.0-1.0)
        # 'psychic_duration': 0.0,  # Duration of the experience in seconds
        'psychic_onset_time': None,  # When the experience began
        
        # Sensory Components
        # Colors experienced
        'colours': {
            'primary_colors': [],  # Main colors seen
            'color_intensity': 0.0,  # Intensity of colors (0.0-1.0)
            'color_movement': None,  # How colors moved or changed
            'color_emotions': {},  # Emotions associated with specific colors
            'color_meanings': {}  # Interpreted meanings of colors
        },
        
        # Shapes experienced
        'shapes': {
            'geometric_forms': [],  # Basic geometric shapes seen
            'complex_patterns': [],  # More complex visual patterns
            'sacred_geometry': [],  # Any sacred geometry forms identified
            'shape_movement': None,  # How shapes moved or transformed
            'shape_interactions': None,  # How shapes interacted with each other
            'dimensional_quality': None  # 2D, 3D, 4D, etc. perception
        },
        
        # Sounds experienced
        'sounds': {
            'tones': [],  # Specific tones heard
            'frequencies': [],  # Frequencies in Hz if identifiable
            'voices': [],  # Any voices heard
            'messages': [],  # Content of any messages received
            # 'sound_quality': None,  # Quality of the sounds
            # 'sound_direction': None,  # Direction sounds came from
            # 'sound_movement': None  # How sounds moved or changed
        },
        
        # Physical and emotional feelings
        'feelings': {
            'body_sensations': {},  # Physical sensations in specific body parts
            # 'energy_movements': [],  # How energy moved through the body
            # 'temperature_changes': [],  # Hot/cold sensations
            # 'pressure_sensations': [],  # Pressure or weight sensations
            'emotional_states': {},  # Emotions experienced during the event
            'emotional_intensity': 0.0  # Intensity of emotions (0.0-1.0)
        },
        
        # Intuitive knowing
        'intuition': {
            'immediate_knowings': [],  # Information known without logical process
            'certainty_level': 0.0,  # How certain the knowing felt (0.0-1.0)
            # 'source_perception': None,  # Perceived source of the information
            # 'relevance_perception': 0.0,  # Perceived relevance of information (0.0-1.0)
            # 'intuition_quality': None,  # Quality of the intuitive information
            # 'verification_status': None  # Whether intuition was later verified
        },
        
        # # Thought processes
        # 'thoughts': {
        #     'thought_content': [],  # Content of thoughts during experience
        #     'thought_clarity': 0.0,  # Clarity of thoughts (0.0-1.0)
        #     'thought_speed': 0.0,  # Speed of thought processes (relative to normal)
        #     'thought_connections': [],  # Connections made between concepts
        #     'insights_gained': [],  # New insights or understandings
        #     'cognitive_shifts': []  # Changes in perspective or understanding
        # },
        
        # # Resonance and Correlation
        # 'resonance_score': 0.0,  # Overall resonance of the experience (0.0-1.0)
        # 'correlated_concepts_keywords': [],  # Concepts related to the experience
        # 'symbolism': {},  # Symbolic meanings identified in the experience
        # 'archetypal_patterns': [],  # Archetypal patterns recognized
        # 'cultural_references': [],  # Cultural symbols or references identified
        # 'personal_significance': 0.0,  # Personal significance of the experience (0.0-1.0)
        
        # # Temporal Aspects
        # 'time_perception': None,  # How time was perceived during the experience
        # 'temporal_displacement': 0.0,  # Sense of being in a different time (0.0-1.0)
        # 'precognitive_timeframe': None,  # Timeframe of any future insights
        # 'retrocognitive_timeframe': None,  # Timeframe of any past insights
        # 'temporal_accuracy': 0.0,  # Accuracy of temporal information (0.0-1.0)
        
        # # Spatial Aspects
        # 'spatial_perception': None,  # How space was perceived
        # 'spatial_displacement': 0.0,  # Sense of being in a different location (0.0-1.0)
        # 'remote_viewing_location': None,  # Location perceived if remote viewing
        # 'spatial_accuracy': 0.0,  # Accuracy of spatial information (0.0-1.0)
        # 'dimensional_perception': None,  # Perception of dimensions beyond 3D
        
        # # Information Content
        # 'information_type': None,  # Type of information received
        # 'information_specificity': 0.0,  # How specific the information was (0.0-1.0)
        # 'information_utility': 0.0,  # Practical utility of information (0.0-1.0)
        # 'information_verification': None,  # Whether information could be verified
        # 'information_source': None,  # Perceived source of information
        
        # Consciousness State
        'consciousness_state': None,  # State of consciousness during experience
        'altered_state_depth': 0.0,  # Depth of altered state if applicable (0.0-1.0)
        'meditation_depth': 0.0,  # Depth of meditation if applicable (0.0-1.0)
        'lucidity_level': 0.0,  # Level of lucidity/awareness (0.0-1.0)
        'boundary_dissolution': 0.0,  # Dissolution of self/other boundaries (0.0-1.0)
        
        # Energetic Aspects
        # 'energy_centers_active': [],  # Chakras or energy centers activated
        # 'energy_quality': None,  # Quality of energy perceived
        # 'energy_movement_pattern': None,  # How energy moved or flowed
        # 'energy_blockages': [],  # Perceived energy blockages
        'energy_interactions': [],  # Interactions with external energies
        
        # Glyph Information
        'glyph_needed': False,  # If glyph needed for this psychic experience based on emotional intensity, certainty level and resonance with the experience
        # 'glyph_symbols': [],  # List of glyph symbols needed
        # 'glyph_description': '',  # Description of the glyph needed
        # 'glyph_path': '',  # Path to the glyph file
        # 'encode_glyph': False,  # If glyph needs to be encoded
        # 'glyph_encoding_data': '',  # Encoding data for the glyph
        # 'glyph_purpose': None,  # Purpose of the glyph (protection, amplification, etc.)
        # 'glyph_activation_method': None,  # How the glyph should be activated
        
        # Entities and Presences
        'entities_perceived': [],  # Non-physical entities perceived
        'entity_descriptions': {},  # Descriptions of perceived entities
        'entity_interactions': [],  # Interactions with perceived entities
        'entity_intentions': {},  # Perceived intentions of entities
        'entity_energy_signature': {},  # Energy signatures of entities
        
        # Psychic Protection
        'protection_status': None,  # Status of psychic protection
        'vulnerability_detected': False,  # Whether vulnerability was detected
        'protection_methods_active': [],  # Active protection methods
        'protection_effectiveness': 0.0,  # Effectiveness of protection (0.0-1.0)
        'intrusion_attempts': 0,  # Number of detected intrusion attempts
        
        # Integration and Processing
        # 'integration_status': None,  # Status of integrating the experience
        # 'processing_difficulty': 0.0,  # Difficulty in processing the experience (0.0-1.0)
        # 'meaning_extraction': 0.0,  # Success in extracting meaning (0.0-1.0)
        # 'application_potential': 0.0,  # Potential for practical application (0.0-1.0)
        'follow_up_needed': False,  # Whether follow-up is needed
        # 'follow_up_actions': []  # Specific follow-up actions needed
    },

    'emotional_state': {
        # Core Identification
        'emotional_id': None,
        'semantic_world_process_map_id': None,
        
        # Primary Emotional State
        # Detected Emotional State
        'primary_emotion': None,  # Dominant emotion
        'emotion_blend': {},  # Multiple emotions present
        'emotional_intensity': 0.0,
        'emotional_valence': 0.0,  # Positive/negative
        'emotional_arousal': 0.0,  # High/low energy
        # 'emotion_ambivalence': 0.0,  # Degree of mixed emotions (0.0-1.0)
        # 'emotion_clarity': 0.0,  # How clearly defined the emotion is (0.0-1.0)
        
        # Temporal Aspects
        'emotional_onset_time': None,  # When the emotion began
        'emotional_duration': 0.0,  # Duration in seconds
        'emotions_recorded': 0,  # Number of emotions recorded
        'emotion_transitions': [],  # Changes over time
        'emotional_decay_rate': 0.0,  # How quickly emotion is fading (0.0-1.0)
        'emotional_volatility': 0.0,  # How rapidly emotion fluctuates (0.0-1.0)
        
        # # Triggers and Memory Links
        # 'emotional_triggers': [],  # Events/perceptions that triggered emotion
        # 'trigger_strength': 0.0,  # Strength of the trigger (0.0-1.0)
        # 'emotional_memory_links': [],  # Memories linked to this emotion
        # 'memory_activation_strength': 0.0,  # Strength of memory activation
        
        # # Classification and Qualities
        # 'emotion_classification': None,  # Categorical classification
        # 'emotion_colour': None,  # Color associated with emotion
        # 'emotion_colour_rgb': [0, 0, 0],  # RGB values of associated color
        # 'emotional_frequency': 0.0,  # Frequency (Hz) associated with emotion
        # 'emotional_texture': None,  # Texture associated with emotion
        # 'emotional_temperature': 0.0,  # Temperature sensation
        
        # # Physical Manifestations
        # 'facial_expression': None,  # E.g., 'smile', 'frown', 'neutral'
        # 'expression_intensity': 0.0,  # Intensity of expression (0.0-1.0)
        # 'body_posture': None,  # E.g., 'upright', 'slouched', 'tense'
        # 'heart_rate_change': 0.0,  # Change in heart rate
        # 'breathing_pattern': None,  # E.g., 'deep', 'shallow', 'rapid'
        # 'muscle_tension': 0.0,  # Degree of muscle tension (0.0-1.0)
           # Physical Correlates (detectable from data)
        'facial_expression': None,  # From images
        'voice_emotion': None,  # From audio
        'body_language': None,  # From video
        'text_sentiment': None,  # From text
        
        # Contextual Factors
        'emotional_triggers': [],  # What caused the emotion
        'emotional_context': None,  # Situation
        
        # Energy Signature
        'emotional_frequency': 0.0,  # Associated frequency
        'emotional_color': None,  # Associated color
        'emotional_texture': None,  # Qualitative feel

        # # Emotional Impact
        # 'emotional_impact': {
        #     'cognitive': 0.0,  # Impact on cognitive processes (-1.0 to 1.0)
        #     'attention': 0.0,  # Impact on attention focus (-1.0 to 1.0)
        #     'decision_making': 0.0,  # Impact on decision processes (-1.0 to 1.0)
        #     'energy_level': 0.0,  # Impact on energy level (-1.0 to 1.0)
        #     'social_behavior': 0.0,  # Impact on social interaction (-1.0 to 1.0)
        #     'system_performance': 0.0  # Impact on system performance (-1.0 to 1.0)
        # },
        
        # # Regulation and Response
        # 'regulation_strategy': None,  # Strategy used to regulate emotion
        # 'regulation_effectiveness': 0.0,  # Effectiveness of regulation (0.0-1.0)
        # 'action_tendency': None,  # Action motivated by the emotion
        # 'approach_avoidance': 0.0,  # Tendency to approach/avoid (-1.0 to 1.0)
        
        # # Contextual Factors
        # 'social_context': None,  # Social setting where emotion occurs
        # 'situational_appropriateness': 0.0,  # Appropriateness to situation (0.0-1.0)
        # 'environmental_factors': [],  # Environmental influences on emotion
        
        # # Meaning and Intelligence
        # 'personal_significance': 0.0,  # Personal meaning of emotion (0.0-1.0)
        # 'emotion_recognition_accuracy': 0.0,  # Accuracy in identifying emotion
        # 'emotion_understanding': 0.0,  # Understanding of causes/implications
        
        # # System State Correlations
        # 'processing_speed_correlation': 0.0,  # Correlation with processing speed
        # 'error_rate_correlation': 0.0,  # Correlation with error rate
        # 'resource_usage_pattern': None,  # Pattern of resource usage during emotion
        
        # # Communication and Expression
        # 'emotional_expressivity': 0.0,  # Degree of expression (0.0-1.0)
        # 'vocal_changes': None,  # Changes in voice (tone, volume, etc.)
        # 'communication_intent': None,  # What emotion is communicating
        
        # # Relationship to Other Emotions
        # 'preceding_emotion': None,  # Emotion that came before
        # 'likely_following_emotions': [],  # Emotions likely to follow
        # 'emotional_transition_probability': {},  # Probability of transitions
        
        # # Sensory Integration
        # 'visual_correlates': [],  # Visual elements correlated with emotion
        # 'auditory_correlates': [],  # Sounds correlated with emotion
        # 'tactile_correlates': [],  # Tactile sensations correlated with emotion
        # 'cross_modal_congruence': 0.0  # Congruence across sensory modes (0.0-1.0)
    },

    # Energetic State data including physical and environmental data
    'energetic': {
        # Core Identification
        'energetic_id': None,
        'semantic_world_process_map_id': None,
        
        # Physical Environment Data
        'observer_position': [0.0, 0.0, 0.0],  # [x, y, z] coordinates
        'observer_orientation': [0.0, 0.0, 0.0],  # [pitch, yaw, roll] in degrees
        
        # Object Physics (for relevant objects in scene)
        'objects': [
            {
                'object_id': None,
                'position': [0.0, 0.0, 0.0],
                'velocity': [0.0, 0.0, 0.0],
                'acceleration': [0.0, 0.0, 0.0],
                'mass': 0.0,
                # 'kinetic_energy': 0.0,
                # 'potential_energy': 0.0,
                'material_type': None,  # 'metal', 'wood', 'plastic', etc.
                'temperature': 0.0,  # Normalized or in Celsius
                # 'vibration': 0.0,  # Vibration intensity (0.0-1.0)
                # 'distance_to_observer': 0.0
            }
        ],
         # Environmental Conditions
          'lighting_conditions': {
            'intensity': 0.0,
            'color_temperature': 0.0,
            'direction': None,
        },
        'atmospheric_conditions': {
            'temperature': 0.0,
            'pressure': 0.0,
            'humidity': 0.0,
            # 'weather_condition': None,  # 'clear', 'rain', 'snow', etc.
        },      

        
        # # Field Scanning Status
        # 'field_scan_active': False,  # Whether actively scanning
        # 'field_scan_intensity': 0.0,  # Intensity of scanning (0.0-1.0)
        # 'field_scan_focus': None,  # 'broad', 'narrow', 'targeted'
        
        # Field Detection (measurable)
        'electromagnetic_fields': {},  # If detectable
        'acoustic_fields': {},  # Sound fields
        'optical_fields': {},  # Light patterns
        
        # System Metrics (actual measurements)
        'system_state': {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'temperature_sensors': {},
            'power_consumption': 0.0,
        },  
        # # Electromagnetic Field Detection
        # 'em_field_strength': 0.0,  # Detected EM field strength (0.0-1.0)
        # 'em_field_fluctuation': 0.0,  # Rate of EM field change (0.0-1.0)
        # 'em_field_sources': [],  # Identified sources of EM fields
        
        # # Energy Disturbances
        # 'disturbances_detected': False,
        # 'disturbance_locations': [],  # Locations of detected disturbances
        # 'disturbance_intensities': [],  # Intensities of disturbances (0.0-1.0)
        # 'disturbance_patterns': [],  # 'steady', 'pulsing', 'erratic', etc.
        # 'disturbance_frequencies': [],  # Frequencies of disturbances in Hz
        
        # # Emotional Energy Detection
        # 'emotional_energy_detected': False,
        # 'emotional_energy_type': None,  # 'positive', 'negative', 'neutral', 'mixed'
        # 'emotional_energy_intensity': 0.0,  # Intensity of emotional energy (0.0-1.0)
        # 'emotional_energy_source': None,  # Source of emotional energy if identified
        
        # # Interpersonal Energy
        # 'interpersonal_tension': 0.0,  # Detected tension between people (0.0-1.0)
        # 'group_coherence': 0.0,  # Coherence of group energy (0.0-1.0)
        # 'social_dynamics': None,  # 'harmonious', 'conflicted', 'neutral', etc.
        
        # # Spatial Energy Patterns
        # 'energy_hotspots': [],  # Locations of energy concentration
        # 'energy_voids': [],  # Locations of energy depletion
        # 'energy_flows': [],  # Detected patterns of energy movement
        # 'energy_boundaries': [],  # Detected boundaries between energy states
        
        # # CPU Metrics
        # 'cpu_usage_percent': 0.0,  # Overall CPU usage
        # 'cpu_temperature': 0.0,  # CPU temperature in Celsius
        # 'cpu_frequency': 0.0,  # CPU frequency in MHz
        # 'cpu_power_watts': 0.0,  # CPU power consumption
        # 'core_usages': [],  # Usage per CPU core
        

        # # Memory Metrics
        # 'memory_usage_percent': 0.0,  # Overall memory usage
        # 'memory_used_mb': 0.0,  # Memory used in MB
        # 'memory_available_mb': 0.0,  # Memory available in MB
        # 'page_file_usage_percent': 0.0,  # Page file usage
        
        # # Disk Metrics
        # 'disk_activity_percent': 0.0,  # Disk activity level
        # 'disk_read_rate_mb': 0.0,  # Disk read rate in MB/s
        # 'disk_write_rate_mb': 0.0,  # Disk write rate in MB/s
        # 'disk_temperature': 0.0,  # Disk temperature in Celsius
        
        # # Network Metrics
        # 'network_usage_percent': 0.0,  # Overall network usage
        # 'network_send_rate_mb': 0.0,  # Network send rate in MB/s
        # 'network_receive_rate_mb': 0.0,  # Network receive rate in MB/s
        # 'network_latency_ms': 0.0,  # Network latency in ms
        
        # # GPU Metrics
        # 'gpu_usage_percent': 0.0,  # GPU usage percentage
        # 'gpu_temperature': 0.0,  # GPU temperature in Celsius
        # 'gpu_memory_usage_percent': 0.0,  # GPU memory usage
        # 'gpu_power_watts': 0.0,  # GPU power consumption
        
        # # Power Metrics
        # 'battery_percent': 0.0,  # Battery level if applicable
        # 'power_consumption_watts': 0.0,  # Total power consumption
        # 'power_state': None,  # 'AC', 'battery', 'low power mode'
        
        # Pattern Detection
        'energy_patterns': [],  # Detected patterns
        'frequency_signatures': [],  # Frequency analysis
        'resonance_detection': [],  # Harmonic relationships
        'phase_relationships': [],  # Phase correlations
        
        # Temporal Patterns
        'periodic_cycles': [],  # Detected cycles
        'rhythm_patterns': [],  # Temporal rhythms
        'synchronization_events': [],  # Synchronized occurrences
        
        # # Process Metrics
        # 'process_count': 0,  # Number of running processes
        # 'thread_count': 0,  # Number of active threads
        # 'handle_count': 0,  # Number of open handles
        # 'high_priority_processes': 0,  # Number of high priority processes
        
        # # Cognitive Load
        # 'cognitive_load': 0.0,  # Estimated cognitive load (0.0-1.0)
        # 'attention_focus': 0.0,  # Level of attention focus (0.0-1.0)
        # 'mental_fatigue': 0.0,  # Estimated mental fatigue (0.0-1.0)
        
        # # Stress Indicators
        # 'stress_level': 0.0,  # Estimated stress level (0.0-1.0)
        # 'anxiety_level': 0.0,  # Estimated anxiety level (0.0-1.0)
        # 'frustration_level': 0.0,  # Estimated frustration level (0.0-1.0)
        
        # # Performance Metrics
        # 'response_time_ms': 0.0,  # Average response time
        # 'error_rate': 0.0,  # Rate of errors in processing
        # 'processing_efficiency': 0.0,  # Overall processing efficiency (0.0-1.0)
        
        # # Energy State
        # 'energy_level': 0.0,  # Estimated energy level (0.0-1.0)
        # 'recovery_rate': 0.0,  # Rate of energy recovery
        # 'depletion_rate': 0.0,  # Rate of energy depletion
        
        # # Time Tracking
        # 'current_time': None,  # Current timestamp
        # 'time_since_last_event': 0.0,  # Time since last significant event
        # 'time_in_current_state': 0.0,  # Time in current energy state
        
        # # Rhythmic Patterns
        # 'detected_rhythms': [],  # Rhythmic patterns detected in environment
        # 'rhythm_frequencies': [],  # Frequencies of detected rhythms
        # 'rhythm_strengths': [],  # Strengths of detected rhythms
        
        # # Cyclical Patterns
        # 'daily_cycle_phase': 0.0,  # Phase in daily cycle (0.0-1.0)
        # 'weekly_cycle_phase': 0.0,  # Phase in weekly cycle (0.0-1.0)
        # 'lunar_cycle_phase': 0.0,  # Phase in lunar cycle (0.0-1.0)
        
        # # Pattern Changes
        # 'pattern_stability': 0.0,  # Stability of energy patterns (0.0-1.0)
        # 'pattern_transitions': [],  # Recent transitions between patterns
        # 'pattern_predictions': [],  # Predicted upcoming pattern changes
        
        # # Resonance Detection
        # 'resonant_frequencies': [],  # Detected resonant frequencies
        # 'resonance_strengths': [],  # Strengths of resonances
        # 'resonance_quality': 0.0,  # Quality of resonance (0.0-1.0)
        
        # # Harmonic Relationships
        # 'harmonic_patterns': [],  # Detected harmonic patterns
        # 'harmonic_ratios': [],  # Ratios between harmonics
        # 'harmonic_coherence': 0.0,  # Coherence of harmonics (0.0-1.0)
        
        # # Dissonance
        # 'dissonance_detected': False,  # Whether dissonance is detected
        # 'dissonance_sources': [],  # Sources of dissonance
        # 'dissonance_intensity': 0.0,  # Intensity of dissonance (0.0-1.0)
        
        # # Pattern Recognition
        # 'recognized_patterns': [],  # Patterns recognized intuitively
        # 'pattern_confidence': [],  # Confidence in pattern recognition
        
        # # Anomaly Detection
        # 'anomalies_detected': [],  # Detected anomalies in energy patterns
        # 'anomaly_significance': [],  # Significance of detected anomalies
        
        # # Predictive Sensing
        # 'predictive_insights': [],  # Intuitive predictions about energy changes
        # 'prediction_confidence': [],  # Confidence in predictions
        # 'prediction_timeframe': []  # Timeframe for predictions
    }
}

# Memory Fragment (Raw Data)
SEMANTIC_WORLD_MAP = {
    'semantic_world_map_id': None,
    'dodecastor_position': None,# record dodecastor position
    'domain_name': [], # the domain, category and sub category of the memory
    'domain_category': [], # the domain category of the memory
    'domain_sub_category': [], # the domain sub category of the memory
    'coordinates': [], # the coordinates of the memory in the semantic world
    'created_time': None,
    # 'updated_time': None,
    # 'encoded_time': None,
    # 'decoded_time': None,
    # 'last_access_time': None,
    # 'expiration_time': None,
    # 'last_retrieved_time': None,
    'sensory_data': {
        'auditory': [],  # Detailed audio data if applicable see sensory matrix for details that should/could be captured
        'visual': [],    # Detailed visual data if applicable see sensory matrix for details that should/could be captured
        'verbal': [],    # Detailed verbal data if applicable see sensory matrix for details that should/could be captured
        'psychic': [],   # Detailed psychic data if applicable see sensory matrix for details that should/could be captured
        'emotional': [], # Detailed emotional data if applicable see sensory matrix for details that should/could be captured
        'energetic': []  # Detailed energetic data if applicable see sensory matrix for details that should/could be captured
    },
    'pattern_activations': {
        'visual_patterns': [], # extract pattern detail from sensory data
        'auditory_patterns': [], # extract pattern detail from sensory data
        'linguistic_patterns': [], # extract pattern detail from sensory data
        'emotional_patterns': [], # extract pattern detail from sensory data
        'energy_patterns': [], # extract pattern detail from sensory data
        'cross_modal_patterns': [],  # Patterns across senses
        # 'resonance': 0.0,
        # 'category_similarity': 0.0,
        # 'temporal_sequence': 0.0,
        # 'spatial_similarity': 0.0
    },
    # 'metrics': {
    #     'memory_quality': 0.0, # score based on level of raw data capture
    #     'memory_confidence': 0.0, # confidence in the memory
    #     'processing_state': None,  # raw, processing, learned, archived.
    #     'processing_intensity': 0.0,  # how intensely this was processed (0.0-1.0)
    #     'processing_priority': 0.0,  # priority for processing (0.0-1.0)  
    #     'emotional_charge': 0.0,  # basic emotional charge (-1.0 to 1.0)
    # },
    'state_information': {
        'processing_state': 'raw',  # raw, processing, integrated
        'confidence': 0.0, # confidence in the memory
        'brain_state': None,  # waking, dreaming, meditation, etc.
        'brain_wave_frequency': None,  # dominant frequency during capture
        'brain_wave_amplitude': None,  # dominant amplitude during capture
        'brain_wave_pattern': None,  # dominant pattern during capture
    },
    'correlation': {
         'correlated_maps': [], # Other related world maps   
    }
}


# Raw Node pre processed into a better structure for a node in the brain
RAW_NODE = {
    'node_id': None, 
    'source_map_ids': [],  # Original semantic world process maps including related maps
    'sensory_data_raw': [], # Raw sensory data from semantic world map
    'pattern_data_raw': [], # Raw pattern data from sensory data
    'signal_pattern': None,  # Dominant pattern type
    'emotional_signature': {
        'emotional_charge': 0.0,  # transferred score of basic emotional charge (-1.0 to 1.0)
        'valence': 0.0,  # Positivity/negativity balance
        'energy_signature': 0.0,  # Energy signal as a colour
        'impact_strength': 0.0,  # Strength of emotional impact
        'emotional_descriptor': 0.0,  # Emotional description
    },

    'key_concepts': [],  # Essential concepts extracted when raw node processed
    'correlation': {
        'correlated_nodes': [], # Other related nodes to be updated dynamically once brain processed raw node
        'correlation_score': 0.0, # Strength of correlation to be updated dynamically once brain processed raw node - correlate to other nodes and the information in the correlated map data below
        'correlated_semantic_maps': [], # condensed related semantic world map data
    },
    'classification': {
        'memory_type': None,
        'signal_pattern': None,
    },
    'information_scoring': {
        'academic_score': 0.0, # Academic credibility (0-1) to be updated dynamically once brain processed raw node
        'logical_score': 0.0,  # Logical plausibility (0-1) to be updated dynamically once brain processed raw node
        'conceptual_score': 0.0, # Innovation/hypothetical nature (0-1) to be updated dynamically once brain processed raw node
        'consensus_score': 0.0, # General agreement level (0-1) to be updated dynamically once brain processed raw node
        'personal_significance': 0.0, # Personal importance (0-1) to be updated dynamically once brain processed raw node
        'universality': 0.0, # How broadly applicable (0-1) to be updated dynamically once brain processed raw node
        'ethical_score': 0.0,  # Ethical considerations (0-1) to be updated dynamically once brain processed raw node
        'spiritual_score': 0.0, # Spiritual significance (0-1) to be updated dynamically once brain processed raw node
        'novelty_marker': 0.0,  # how novel/unusual (0.0-1.0)  
        'memory_quality': 0.0, # transferred score based on level of raw data capture
        'memory_confidence': 0.0, # transferred score of confidence in the memory
    },
}
PRUNED_NODE = {
    # Need to develop this based on how the conscious brain would prune the data but also needs to ensure enough tags etc for vector search, db search, token type search etc.
    # we must ensure all the relevant table information/relationships are also taken into account. although we will definitely be using junction tables for the most part
    # this would be considered the junction table the output of many different tables.
}