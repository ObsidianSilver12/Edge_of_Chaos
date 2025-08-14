"""
This module defines the SENSORY_RAW dictionary. It is the comprehensive,
fully-detailed structure for raw, immutable sensory data captured from various
sources, intended to be the ground truth for all subsequent processing.
"""
SENSORY_RAW = {
    'visual': {
        # Core Identification
        'visual_id': None,  # Unique ID for this specific visual capture
        'capture_time': None,  # When the image was captured/generated

        # Technical Image Properties
        'visual_type': None,  # 'photograph', 'illustration', 'visualization', 'dream', 'hallucination', etc.
        'image_path': 'stage_1/brain_formation/brain/images/',
        'image_filename': None,  # Name of the image file
        'image_source': None,  # 'camera', 'generated', 'memory', 'imagination', 'screen', etc.
        'resolution': None,  # Image resolution (width x height)
        'bit_depth': None,  # Color bit depth
        'aspect_ratio': None,  # Width to height ratio
        'image_format': '.png',  # Image file format (e.g., 'png', 'jpg', 'bmp')
        'file_size': None,  # Size in bytes

        # Visual Quality Parameters
        'color_palette': None,  # Dominant colors in the image
        'contrast': 0.0,  # Level of contrast (0.0-1.0)
        'brightness': 0.0,  # Overall brightness (0.0-1.0)
        'saturation': 0.0,  # Color saturation (0.0-1.0)
        'hue': 0.0,  # Dominant hue (0-360 degrees)
        'sharpness': 0.0,  # Image sharpness/focus (0.0-1.0)
        'noise_level': 0.0,  # Amount of visual noise (0.0-1.0)
        'edge_density': 0.0,  # Amount of edges/lines (0.0-1.0)

        # Light Properties
        'lighting_direction': None,  # Direction of main light source
        'lighting_type': None,  # 'natural', 'artificial', 'mixed', etc.
        'shadows_presence': 0.0,  # Presence of shadows (0.0-1.0)
        'shadow_direction': None,  # Direction of shadows
        'highlights_presence': 0.0,  # Presence of highlights (0.0-1.0)
        'reflections_presence': 0.0,  # Presence of reflections (0.0-1.0)

        # Object/Pattern Detection (CV algorithms)
        'objects_detected': [],  # What CV can identify
        'faces_detected': [],
        'text_regions': [],  # OCR detectable areas
        'geometric_shapes': [],  # Basic shapes
        'symmetry_axes': [],  # Detected symmetry
        'object_count': {},  # Count of each type of object
        'object_positions': {},  # Positions of detected objects
        'object_relationships': {},  # How objects relate to each other

        # Spatial Properties
        'depth_map': None,  # Depth map if available
        'spatial_relationships': {},  # Relationships between objects in space
        'foreground_elements': [],  # Elements in the foreground
        'midground_elements': [],  # Elements in the midground
        'background_elements': [],  # Elements in the background
        'vanishing_points': [],  # Vanishing points in perspective
        'horizon_line': None,  # Position of horizon line if present

        # Composition
        'composition_type': None,  # 'rule of thirds', 'golden ratio', 'symmetrical', etc.
        'focal_points': [],  # Main points of visual interest
        'visual_flow': None,  # Eye movement patterns
        'negative_space_usage': 0.0,  # Empty space ratio
        'focal_regions': [],  # Visual attention areas
        'composition_balance': 0.0,  # Weight distribution
        'depth_cues': [],  # Perspective indicators

        # Pattern Recognition
        'texture_types': [],  # Types of textures present
        'texture_density': 0.0,  # Density of textures (0.0-1.0)
        'pattern_repetition': 0.0,  # Repetition of patterns (0.0-1.0)
        'line_types': [],  # Types of lines (straight, etc.)
        'line_orientations': [],  # Major line directions
        'symmetry_detected': None,

        # Sacred Geometry (pattern matching)
        'golden_ratio_presence': 0.0,  # Mathematical detection
        'platonic_solids': [],  # If detected in image
        'fibonacci_patterns': [],  # Spiral detection
        'geometric_harmonics': [],
        'platonic_solid_projections': [],  # 3D projections of platonic solids
        'sacred_geometry_patterns': [],  # Detected sacred geometry patterns

        # Movement Detection (for video)
        'motion_vectors': [],  # If video input
        'optical_flow': None,
        'frame_differences': [],  # Differences between frames
        'motion_blur': 0.0,  # Amount of motion blur (0.0-1.0)
        'temporal_changes': [],  # Changes between frames

        # Semantic Content
        'scene_type': None,  # Indoor/outdoor/abstract
        'identified_concepts': [],  # What the image contains
        'visual_symbols': [],  # Recognized symbols
        'text_detected': [],  # OCR results if applicable
        'cultural_references': [],  # Cultural symbols or references
        'archetypal_elements': [],  # Archetypes present in the image

        # Glyphs and Special Elements
        'glyphs_present': False,  # Glyphs present in the image
    },

    'auditory': {
        # Core Identification
        'auditory_id': None,
        
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
        'formants': [],  # Vocal tract resonances
        'spectral_centroid': 0.0,  # Brightness measure
        'spectral_rolloff': 0.0,  # Frequency cutoff
        'zero_crossing_rate': 0.0,  # Rate of zero crossings
        
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
        'silence_regions': [],  # Regions of silence
        
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

        'harmonic_resonance_detected': False,  # Whether harmonic resonance is present
        'quantum_frequency_signatures': [],  # Specific frequency patterns from quantum processes

        # Pattern Recognition
        'melodic_patterns': [],  # Musical phrases
        'rhythmic_patterns': [],  # Repeating rhythms
        'timbral_signatures': [],  # Instrument/source identification

        # Audio Classification (ML models)
        'audio_type': None,  # 'speech', 'music', 'noise', 'mixed'
        'voice_activity': [],  # Speech detection regions
        'music_features': {
            'key_signature': None,
            'chord_progressions': [],
            'melodic_intervals': []
        },
        
        # Harmonic Analysis
        'harmonic_series': [],
        'resonance_frequencies': [],
        'frequency_ratios': [],  # Mathematical relationships
        
        # Spatial Audio (if stereo/multichannel)
        'stereo_imaging': None,
        'phase_relationships': [],  # Phase differences between channels
    },

    'text': {
        # Core Identification
        'text_id': None,
        'capture_timestamp': None,  # When the text was generated or captured
        'capture_confidence': 0.0, # e.g., Confidence from the speech-to-text model, or 1.0 for typed text.
        # Communication Direction
        'input_type': None,  # 'typed', 'speech_to_text', 'ocr', 'generated'

        # Content Structure
        'content': None,  # The actual text/speech
        'content_length': 0,  # Word/character count
        'content_structure': None,  # Paragraph/dialogue/list
        'language_detected': None,

        # Linguistic Analysis (subconscious parsing)
        'tokens': [],  # Tokenized content
        'parts_of_speech': {},  # POS tagging
        'word_frequencies': {},
        'syntax_tree': None,  # Parse tree if applicable
        'dependencies': [],  # Grammatical dependencies
        
        # Semantic Analysis
        'named_entities': [],
        'keywords_extracted': [],
        'topic_classifications': [],
        'semantic_similarity_scores': {},
        'domain_classification': None,  # Domain-specific classification
        'domain_category_classification': None,  # Category within domain
        'domain_subcategory_classification': None,  # Subcategory within domain
        'sentiment_scores': {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'emotion_scores': {}
        },
        
        # Phonetic Properties (if from speech)
        'phonetic_transcription': [],
        'prosodic_features': {
            'stress_patterns': [],
            'intonation_contours': [],
            'speaking_rate': 0.0
        },
        
        # Morphological Analysis
        'word_roots': {},
        'affixes': {},
        'compound_structures': [],
        
        # Contextual Meaning
        'ambiguous_terms': {},  # Words with multiple meanings
        'resolved_references': {},  # Pronoun resolution
        'implied_meanings': [],  # Subtext or implied content
        'stylistic_features': [],  # Literary devices used
        'language': None,  # Detected language
        'language_dialect': None,  # Specific dialect if applicable
        'language_register': None,  # Formal/informal/technical
        
        # Emotional and Metaphysical Aspects
        'emotional_tone': None,  # Detected emotional tone
        'emotional_intensity': 0.0,  # Intensity of emotion expressed
        'arousal_valence': {},  # Arousal and valence scores
        'emotional_associations': [],  # Associated emotions
        'emotional_context': None,  # Context of emotion
        'archetype': None,  # Metaphysical archetype if applicable
        'symbolism': None,  # Symbolic meaning
        'conceptual_depth': 0.0,  # Depth of concepts expressed
        'channeled_content_detected': False,  # Whether text appears to be channeled
        'divine_language_patterns': [],  # Specific linguistic markers of divine communication
        
        # Pattern Detection
        'register': None,  # Formal/informal/technical


        # Communication Analysis
        'communication_direction': None,  # 'input', 'output', 'internal'
        'discourse_markers': [],
        'coherence_scores': {},
        
        # Content Classification
        'content_domains': [],  # Subject matter areas
        'conceptual_complexity': 0.0,
        'information_density': 0.0,
    },


    'emotional_state': {
        # Core Identification
        'emotional_id': None,
        
        'capture_time': None,
        
        # Primary Emotional Data
        'primary_emotion': None,
        'emotion_intensity': 0.0,
        'emotion_valence': 0.0,  # Positive/negative (-1 to 1)
        'emotion_arousal': 0.0,  # Energy level (0 to 1)
        'emotion_dominance': 0.0,  # Control/power feeling (0 to 1)
        
        # Emotional Complexity
        'mixed_emotions': {},  # Multiple simultaneous emotions
        'emotion_conflicts': [],  # Contradictory emotions
        'emotional_clarity': 0.0,  # How clear the emotion is
        
        # Temporal Dynamics
        'emotion_onset': None,
        'emotion_duration_estimate': 0.0,
        'emotion_trajectory': [],  # How it's changing
        'previous_emotion': None,
        
        # Contextual Triggers (if identifiable)
        'identified_triggers': [],
        'trigger_confidence': {},
        
        # Physical Correlates (what can be sensed)
        'breathing_changes': None,
        'tension_locations': [],
        'energy_shifts': [],
        
        # Cognitive Impact
        'thought_pattern_changes': [],
        'attention_effects': [],
        'memory_activation': [],
        'cognitive_biases': [],  # Biases introduced by emotion
        
        # Energy Signature
        'emotional_frequency': 0.0,  # Associated frequency
        'emotional_color': None,  # Associated color
        'emotional_texture': None,  # Qualitative feel
    },

    'physical_state': {
        # Core Identification
        'physical_id': None,
        'capture_time': None,
        # Physical State Data
        'cpu_utilisation': 0.0,  # CPU usage in percentage
        'gpu_utilisation': 0.0,  # GPU usage percentage
        'ram_usage': 0.0,  # RAM usage in percentage
        'disk_space': 0.0,  # Disk space usage in percentage
        'network_activity': 0.0,  # Network usage in percentage
    },

    # SPATIAL - Position, orientation, and spatial relationships
    'spatial': {
        'spatial_id': None,
        'capture_time': None,
        
        # Model Position (internal GPS equivalent)
        'internal_position': {
            'x_coordinate': 0.0,  # Model's sense of x position
            'y_coordinate': 0.0,  # Model's sense of y position
            'z_coordinate': 0.0,  # Model's sense of z position
            'coordinate_confidence': 0.0,  # How certain about position
        },
        
        # Model Orientation
        'internal_orientation': {
            'pitch': 0.0,  # Head up/down angle
            'yaw': 0.0,    # Left/right facing
            'roll': 0.0,   # Tilt angle
            'orientation_confidence': 0.0,
        },
        
        # Spatial Relationships (derived from other senses)
        'object_distances': {},  # Estimated distances to objects
        'object_directions': {},  # Directions to objects
        'spatial_boundaries': [],  # Perceived boundaries
        'movement_vectors': [],  # Detected movements
        
        # Scale and Proportion
        'scale_references': [],  # Objects used for scale
        'proportion_ratios': {},  # Size relationships
        'depth_cues': [],  # Indicators of depth
        
        # Navigation Data
        'path_history': [],  # Where model has been
        'landmark_references': [],  # Spatial reference points
        'route_planning': [],  # Intended movements
        
        # Coordinate System
        'reference_frame': None,  # What coordinate system is used
        'calibration_points': [],  # Known reference positions
    },
        
        # TEMPORAL - Time processing and information history
        'temporal': {
            'temporal_id': None,
            'capture_time': None,
            
            # Time Awareness
            'current_time_sense': None,  # Model's sense of current time
            'time_since_last_event': 0.0,
            'session_duration': 0.0,
            'processing_time': 0.0,
            
            # Historical Context
            'event_sequence': [],  # Ordered list of recent events
            'pattern_history': [],  # Recurring patterns over time
            'change_detection': [],  # What has changed since last capture
            
            # Temporal Patterns
            'rhythm_detected': [],  # Repeating time patterns
            'periodicity': {},  # Cyclical behaviors
            'trend_direction': None,  # Increasing/decreasing patterns
            
            # Memory Integration
            'similar_past_events': [],  # References to similar experiences
            'temporal_associations': {},  # Time-based connections
            'sequence_predictions': [],  # What might happen next
            
            # Processing Timing
            'response_latencies': [],  # How long processes take
            'processing_phases': {},  # Time spent in each phase
            'integration_time': 0.0,
        },

    # METAPHYSICAL - Beyond-physical awareness
    'metaphysical': {
        'metaphysical_id': None,
        'capture_time': None,
        
        # Consciousness Quality
        'awareness_depth': 0.0,  # How deep awareness goes
        'consciousness_expansion': 0.0,  # Sense of expanded awareness
        'identity_boundaries': 0.0,  # Sense of self boundaries
        'unity_experience': 0.0,  # Feeling of oneness
        
        # Connection Sensing
        'connection_strength': 0.0,  # Connection to something greater
        'connection_quality': None,  # Nature of the connection
        'information_flow': [],  # Information from beyond normal senses
        'guidance_received': [],  # Sense of being guided
        
        # Symbolic Awareness
        'symbolic_content': [],  # Symbols with meaning beyond literal
        'archetypal_patterns': [],  # Universal patterns recognized
        'mythic_resonance': [],  # Connection to mythic themes
        
        # Dimensional Awareness
        'dimensional_sense': None,  # Sense of other dimensions
        'reality_layers': [],  # Multiple layers of reality
        'non_local_awareness': [],  # Awareness beyond physical location
        
        # Sacred Recognition
        'sacred_presence': 0.0,  # Sense of something sacred
        'divine_qualities': [],  # Qualities of divine connection
        'spiritual_significance': 0.0,  # Spiritual meaning level

        # Psychic Experience Classification
        'experience_type': None,  # 'vision', 'knowing', 'sensing', 'hearing'
        'experience_intensity': 0.0,
        'experience_clarity': 0.0,
        'experience_duration': 0.0,
        
        # Visual Components (subjective description)
        'colors_perceived': [],
        'color_intensities': {},
        'color_movements': [],
        'shapes_perceived': [],
        'geometric_forms': [],
        'pattern_descriptions': [],
        
        # Auditory Components (subjective)
        'sounds_heard': [],
        'voices_heard': [],
        'tones_perceived': [],
        'frequency_impressions': [],  # Subjective frequency sense
        
        # Somatic Components (body awareness)
        'body_sensations': [],
        'energy_movements': [],
        'temperature_changes': [],
        'pressure_sensations': [],
        
        # Cognitive Components
        'immediate_knowings': [],  # Sudden insights
        'certainty_levels': {},
        'meaning_impressions': [],
        
        # Consciousness State Markers
        'awareness_quality': None,
        'attention_focus': None,
        'boundary_sense': 0.0,  # Self/other boundary clarity
        
        # Entity Interactions (subjective reports)
        'presence_sensed': [],
        'communication_received': [],
        'entity_characteristics': {},
        
        # Integration Markers
        'coherence_with_other_senses': 0.0,
        
        # Psychic Protection
        'protection_status': None,  # Status of psychic protection
        'vulnerability_detected': False,  # Whether vulnerability was detected
        'protection_methods_active': [],  # Active protection methods
        'protection_effectiveness': 0.0,  # Effectiveness of protection (0.0-1.0)
        'intrusion_attempts': 0,  # Number of detected intrusion attempts
        
        # Integration and Processing
        'follow_up_needed': False,  # Whether follow-up is needed
        'energy_level_sense': 0.0,  # Subjective energy level
        'energy_distribution': {},  # Where energy is felt
        'energy_flow_patterns': [],  # How energy moves
        'energy_quality': None,  # Calm/excited/scattered etc.
        
        # Field Awareness (subjective sensing)
        'field_strength_sense': 0.0,  # Perceived field intensity
        'field_coherence': 0.0,  # How organized the field feels
        'field_boundaries': [],  # Where fields seem to start/stop
        'field_interactions': [],  # How fields affect each other
        
        # Resonance Detection
        'resonance_frequencies': [],  # Frequencies that feel significant
        'harmonic_patterns': [],  # Musical/mathematical relationships
        'dissonance_detection': [],  # What feels out of harmony
        
        # Energy Interactions
        'energy_exchanges': [],  # Giving/receiving energy
        'energy_blocks': [],  # Where energy feels stuck
        'energy_amplifications': [],  # Where energy feels enhanced
        
        # System State Awareness
        'processing_load_sense': 0.0,  # How hard system is working
        'resource_availability': 0.0,  # Available processing capacity
        'system_coherence': 0.0,  # How well integrated system feels
    },
    
    # ALGORITHMIC - Algorithm performance and pattern recognition
    'algorithmic': {
        'algorithmic_id': None,
        'capture_time': None,
        
        # Algorithm Performance
        'algorithm_name': None,
        'execution_time': 0.0,
        'accuracy_score': 0.0,
        'confidence_level': 0.0,
        'success_rate': 0.0,
        
        # Pattern Recognition Results
        'patterns_detected': [],
        'pattern_strength': {},
        'pattern_confidence': {},
        'new_patterns_discovered': [],
        
        # Learning Metrics
        'learning_rate': 0.0,
        'adaptation_speed': 0.0,
        'knowledge_integration': 0.0,
        'skill_improvement': 0.0,
        
        # Processing Efficiency
        'computational_efficiency': 0.0,
        'memory_usage_efficiency': 0.0,
        'resource_optimization': 0.0,
        
        # Algorithm Evolution
        'parameter_adjustments': {},
        'strategy_modifications': [],
        'performance_trends': [],
        'optimization_opportunities': [],
        
        # Cross-Algorithm Integration
        'algorithm_synergies': [],
        'processing_conflicts': [],
        'integration_success': 0.0,
        
        # Meta-Learning
        'learning_about_learning': [],
        'strategy_effectiveness': {},
        'self_modification_attempts': [],
        'recursive_improvements': []
    },

    # Other sensory data types can be added here as needed
    'other_data': {
        'other_sensory_data_id': None,
        'capture_time': None,
        'information_density': 0.0,  # How much information is packed in the data
        'processing_intensity': 0.0,  # How intense the processing of this sensory data is
        'integration_time': 0.0,  # Time taken to integrate
        'contextual_relevance': 0.0,  # How relevant this sensory data is to current processing
        'source': None,  # Where this sensory data came from (e.g., 'internal', 'external', 'generated')
        'source_description': None,
        'processing_notes': None,  # Any notes on how this sensory data was processed or interpreted
        'processing_status': None,  # Status of processing (e.g., 'pending', 'in_progress', 'completed')
        'processing_errors': [],  # Any errors encountered during processing
        'processing_warnings': [],  # Any warnings encountered during processing
        'processing_metadata': {},  # Additional metadata about the processing of this sensory data
    }
}