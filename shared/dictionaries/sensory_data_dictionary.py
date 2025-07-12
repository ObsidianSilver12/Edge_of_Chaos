
# passive part of the subconscious recording of data but can be back propogated from the node when information is processed or learned later.

SENSORY_RAW = {
    'visual': {
        # Core Identification
        'visual_id': None,

        # Image Source and Format
        'image_type': None,  # 'photograph', 'illustration', 'visualization', 'dream', 'hallucination', etc.
        'image_path': 'stage_1/brain_formation/brain/images/',
        'image_format': '.png',
        'image_source': None,  # 'camera', 'generated', 'memory', 'imagination', etc.
        'capture_time': None,  # When the image was captured/generated

        # Technical Image Properties
        'resolution': None,  # Image resolution (width x height)
        'bit_depth': None,  # Color bit depth
        'aspect_ratio': None,  # Width to height ratio
        'file_size': None,  # Size in bytes
        
        # Visual Quality Parameters
        'color_palette': None,  # Dominant colors in the image
        'contrast': 0.0,  # Level of contrast (0.0-1.0)
        'brightness': 0.0,  # Overall brightness (0.0-1.0)
        'saturation': 0.0,  # Color saturation (0.0-1.0)
        'hue': 0.0,  # Dominant hue (0-360 degrees)
        'sharpness': 0.0,  # Image sharpness/focus (0.0-1.0)
        'noise_level': 0.0,  # Amount of visual noise (0.0-1.0)
        
        # Light Properties
        'lighting_direction': None,  # Direction of main light source
        'lighting_type': None,  # 'natural', 'artificial', 'mixed', etc.
        'shadows_presence': 0.0,  # Presence of shadows (0.0-1.0)
        'shadow_direction': None,  # Direction of shadows
        'highlights_presence': 0.0,  # Presence of highlights (0.0-1.0)
        'reflections_presence': 0.0,  # Presence of reflections (0.0-1.0)
        
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
        
        # Object Recognition
        'objects_detected': [],  # Objects detected in the image
        'object_count': {},  # Count of each type of object
        'object_positions': {},  # Positions of objects
        'object_relationships': {},  # How objects relate to each other
        
        # Pattern Recognition
        'texture_types': [],  # Types of textures present
        'geometric_shapes': [],  # Basic shapes detected
        'line_orientations': [],  # Major line directions
        'symmetry_detected': None,

        # Sacred Geometry (pattern matching)
        'golden_ratio_presence': 0.0,  # Mathematical detection
        'platonic_solids': [],  # If detected in image
        'fibonacci_patterns': [],  # Spiral detection
        'geometric_harmonics': [], 

        'divine_connection_strength': 0.0,  # 0.0-1.0 intensity of divine presence
        'quantum_entanglement_indicators': [],  # Visual signs of quantum connection
        'creator_aspect_manifestations': [],  # Visible signs of transferred aspects  
        'light_transmission_quality': 0.0,  # 0.0-1.0 clarity of divine light
        
        # Movement Detection (for video)
        'motion_vectors': [],  # If video input
        'optical_flow': None,
        'temporal_changes': [],  # Changes between frames
        
        # Semantic Content
        'scene_type': None,  # Indoor/outdoor/abstract
        'identified_concepts': [],  # What the image contains
        'visual_symbols': [],  # Recognized symbols
        'text_detected': [],  # OCR results if applicable
        
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

        'harmonic_resonance_detected': False,  # Whether harmonic resonance is present
        'quantum_frequency_signatures': [],  # Specific frequency patterns from quantum processes

        # Pattern Recognition
        'melodic_patterns': [],  # Musical phrases
        'rhythmic_patterns': [],  # Repeating rhythms
        'timbral_signatures': [],  # Instrument/source identification
    },
    'text': {
        # Core Identification
        'text_id': None,
        
        # Communication Direction
        'communication_type': None,  # 'heard', 'spoken', 'internal_dialogue', 'read', 'written'

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
        'domain_classification': None,  # Domain-specific classification
        'domain_category_classification': None,  # Category within domain
        'domain_subcategory_classification': None,  # Subcategory within domain
        
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
       'ambiguous_terms': [],  # Subtext detection

        'channeled_content_detected': False,  # Whether text appears to be channeled
        'divine_language_patterns': [],  # Specific linguistic markers of divine communication
        
        # Pattern Detection
        'linguistic_patterns': [],  # Recurring structures
        'stylistic_features': [],  # Writing style markers
        'register': None,  # Formal/informal/technical
    },

    'psychic': {
        # Core Identification
        'psychic_id': None,
        
        # Experience Classification
        'psychic_event_type': None,  # 'precognition', 'telepathy', 'clairvoyance', 'empathic', 'channeling', etc.
        'psychic_experience': None,  # Description of the psychic experience
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
        },
        
        # Physical and emotional feelings
        'feelings': {
            'body_sensations': [],  # Physical sensations in specific body parts
            'emotional_states': [],  # Emotions experienced during the event
            'emotional_intensity': 0.0  # Intensity of emotions (0.0-1.0)
        },
        
        # Intuitive knowing
        'intuition': {
            'immediate_knowings': [],  # Information known without logical process
            'certainty_level': 0.0,  # How certain the knowing felt (0.0-1.0)
        },
        
        # Consciousness State
        'consciousness_state': None,  # State of consciousness during experience
        'altered_state_depth': 0.0,  # Depth of altered state if applicable (0.0-1.0)
        'meditation_depth': 0.0,  # Depth of meditation if applicable (0.0-1.0)
        'lucidity_level': 0.0,  # Level of lucidity/awareness (0.0-1.0)
        'boundary_dissolution': 0.0,  # Dissolution of self/other boundaries (0.0-1.0)
        
        # Energetic Aspects
        'energy_interactions': [],  # Interactions with external energies
        
        # Glyph Information
        'glyph_needed': False,  # If glyph needed for this psychic experience based on emotional intensity, certainty level and resonance with the experience
        
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
        'follow_up_needed': False,  # Whether follow-up is needed
    },

    'emotional_state': {
        # Core Identification
        'emotional_id': None,
        
        # Primary Emotional State
        'primary_emotion': None,  # Dominant emotion
        'emotion_blend': {},  # Multiple emotions present
        'emotional_intensity': 0.0,
        'emotional_valence': 0.0,  # Positive/negative
        'emotional_arousal': 0.0,  # High/low energy
        'emotion_ambivalence': 0.0,  # Degree of mixed emotions (0.0-1.0)
        'emotion_clarity': 0.0,  # How clearly defined the emotion is (0.0-1.0)
        
        # Temporal Aspects
        'emotional_onset_time': None,  # When the emotion began
        'emotional_duration': 0.0,  # Duration in seconds
        'emotions_recorded': 0,  # Number of emotions recorded
        'emotion_transitions': [],  # Changes over time
        'emotional_decay_rate': 0.0,  # How quickly emotion is fading (0.0-1.0)
        'emotional_volatility': 0.0,  # How rapidly emotion fluctuates (0.0-1.0)
        
        # Contextual Factors
        'emotional_triggers': [],  # What caused the emotion
        'emotional_context': None,  # Situation
        
        # Energy Signature
        'emotional_frequency': 0.0,  # Associated frequency
        'emotional_color': None,  # Associated color
        'emotional_texture': None,  # Qualitative feel
    },

    # Energetic State data including physical and environmental data
    'energetic': {
        # Core Identification
        'energetic_id': None,
        
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
                'material_type': None,  # 'metal', 'wood', 'plastic', etc.
                'temperature': 0.0,  # Normalized or in Celsius
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
        },      
        
        # Field Detection (measurable)
        'electromagnetic_fields': [],  # If detectable
        'acoustic_fields': [],  # Sound fields
        'optical_fields': [],  # Light patterns
        
        # System Metrics (actual measurements)
        'system_state': {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'temperature_sensors': [],
            'power_consumption': 0.0,
        },
    'process_context': {
        'process_id': None,  # ID of the process generating this sensory data
        'process_name': None,  # Name of the process (e.g., 'creator_entanglement')
        'process_phase': None,  # Phase within the process
        'soul_id': None,  # ID of the soul experiencing this
        'glyph_id': None,  # Associated glyph for steganographic encoding
        'metrics_reference': {},  # Reference to associated process metrics
        'integration_data': {}  # Data needed for integration with other captures
        }
    }
}







