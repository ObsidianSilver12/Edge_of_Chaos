"""
This module defines the SENSORY_RAW dictionary. It is the comprehensive,
fully-detailed structure for raw, immutable sensory data captured from various
sources, intended to be the ground truth for all subsequent processing.
"""
SENSORY_RAW = {
    # === CORE SENSORY_RAW IDENTIFICATION ===
    'sensory_raw_id': None,  # Unique ID linking ALL sensory types in this capture moment
    'capture_timestamp': None,  # When this multi-sensory capture occurred
    'active_sensory_types': [],  # Which sensory types are active in this capture

    'visual': {
        # === CORE IDENTIFICATION ===
        'visual_id': None,  # Unique ID for this specific visual capture
        'capture_time': None,  # When the image was captured/generated
        'visual_content': None,  # The actual visual content (image data)

        # === DIRECT EXIF/METADATA (No processing needed) ===
        'technical_metadata': {
            'image_filename': None,
            'image_path': 'stage_1/brain_formation/brain/images/',
            'image_format': '.png',  # From file extension
            'file_size': None,  # From file system
            'resolution': None,  # From EXIF (width x height)
            'bit_depth': None,  # From EXIF
            'aspect_ratio': None,  # Calculated from resolution
            'camera_make': None,  # From EXIF if available
            'camera_model': None,  # From EXIF if available
            'iso_setting': None,  # From EXIF if available
            'aperture': None,  # From EXIF if available
            'shutter_speed': None,  # From EXIF if available
            'focal_length': None,  # From EXIF if available
            'timestamp_original': None,  # From EXIF if available
        },

        # === BASIC CLASSIFICATION (Simple detection) ===
        'basic_classification': {
            'visual_type': None,  # 'photograph', 'illustration', 'visualization', 'dream', 'hallucination'
            'image_source': None,  # 'camera', 'generated', 'memory', 'imagination', 'screen'
            'color_space': None,  # 'RGB', 'CMYK', 'HSV' from image properties
            'has_alpha_channel': False,  # Transparency detection
            'is_grayscale': False,  # Color vs grayscale detection
        },

        # === SPECTRAL ANALYSIS (Color/Light processing algorithms) ===
        'spectral_properties': {
            'spectral_id': None,  # Links to spectral analysis sub-table
            'color_palette': None,  # Dominant colors in the image
            'color_distribution': {},  # Histogram data
            'brightness': 0.0,  # Overall brightness (0.0-1.0)
            'contrast': 0.0,  # Level of contrast (0.0-1.0)
            'saturation': 0.0,  # Color saturation (0.0-1.0)
            'hue': 0.0,  # Dominant hue (0-360 degrees)
            'color_temperature': 0.0,  # Warm/cool color analysis
            'dynamic_range': 0.0,  # Light to dark range
            'exposure_analysis': {},  # Over/under exposure regions
        },

        # === SPATIAL ANALYSIS (Geometric/structural algorithms) ===
        'spatial_properties': {
            'spatial_id': None,  # Links to spatial analysis sub-table
            'depth_map': None,  # Depth map if available/calculated
            'spatial_relationships': {},  # Object positioning relationships
            'composition_type': None,  # 'rule of thirds', 'golden ratio', 'symmetrical'
            'focal_points': [],  # Main points of visual interest
            'focal_regions': [],  # Visual attention areas
            'visual_flow': None,  # Eye movement patterns
            'negative_space_usage': 0.0,  # Empty space ratio (0.0-1.0)
            'composition_balance': 0.0,  # Weight distribution
            'symmetry_axes': [],  # Detected symmetry lines
            'symmetry_detected': None,  # Type of symmetry found
            'vanishing_points': [],  # Perspective analysis
            'horizon_line': None,  # Position of horizon line if present
        },

        # === LIGHTING ANALYSIS (Illumination algorithms) ===
        'lighting_properties': {
            'lighting_id': None,  # Links to lighting analysis sub-table
            'lighting_direction': None,  # Direction of main light source
            'lighting_type': None,  # 'natural', 'artificial', 'mixed'
            'light_intensity': 0.0,  # Overall light intensity
            'shadows_presence': 0.0,  # Presence of shadows (0.0-1.0)
            'shadow_direction': None,  # Direction of shadows
            'highlights_presence': 0.0,  # Presence of highlights (0.0-1.0)
            'reflections_presence': 0.0,  # Presence of reflections (0.0-1.0)
            'light_quality': None,  # 'harsh', 'soft', 'diffused'
            'backlighting': False,  # Backlighting detection
        },

        # === TEXTURE/PATTERN ANALYSIS (Surface algorithms) ===
        'texture_properties': {
            'texture_id': None,  # Links to texture analysis sub-table
            'texture_types': [],  # Types of textures present
            'texture_density': 0.0,  # Density of textures (0.0-1.0)
            'pattern_repetition': 0.0,  # Repetition of patterns (0.0-1.0)
            'surface_roughness': 0.0,  # Perceived surface texture
            'material_properties': [],  # Detected materials (metal, wood, fabric)
            'pattern_regularity': 0.0,  # How regular patterns are
        },

        # === OBJECT DETECTION (Computer Vision algorithms) ===
        'object_detection': {
            'object_detection_id': None,  # Links to object detection sub-table
            'objects_detected': [],  # What CV can identify
            'faces_detected': [],  # Face detection results
            'text_regions': [],  # OCR detectable areas
            'geometric_shapes': [],  # Basic shapes detected
            'object_count': {},  # Count of each type of object
            'object_positions': {},  # Coordinates of detected objects
            'object_relationships': {},  # How objects relate spatially
            'object_confidence_scores': {},  # Detection confidence levels
        },

        # === QUALITY ANALYSIS (Image quality algorithms) ===
        'quality_metrics': {
            'quality_id': None,  # Links to quality analysis sub-table
            'sharpness': 0.0,  # Image sharpness/focus (0.0-1.0)
            'noise_level': 0.0,  # Amount of visual noise (0.0-1.0)
            'blur_detection': 0.0,  # Motion/focus blur (0.0-1.0)
            'compression_artifacts': 0.0,  # JPEG artifacts etc.
            'edge_density': 0.0,  # Amount of edges/lines (0.0-1.0)
            'detail_level': 0.0,  # Amount of fine detail present
            'overall_quality_score': 0.0,  # Composite quality metric
        },

        # === DEPTH/3D ANALYSIS (Depth algorithms) ===
        'depth_properties': {
            'depth_id': None,  # Links to depth analysis sub-table
            'foreground_elements': [],  # Elements in the foreground
            'midground_elements': [],  # Elements in the midground
            'background_elements': [],  # Elements in the background
            'depth_layers': [],  # Distinct depth layers
            'depth_cues': [],  # Perspective indicators
            'occlusion_relationships': [],  # What objects block others
        },

        # === EDGE/LINE ANALYSIS (Edge detection algorithms) ===
        'edge_properties': {
            'edge_id': None,  # Links to edge analysis sub-table
            'line_types': [],  # Types of lines (straight, curved, diagonal)
            'edge_strength': 0.0,  # Overall edge definition
            'contour_data': [],  # Object contours
            'line_directions': [],  # Dominant line orientations
            'edge_continuity': 0.0,  # How connected edges are
        },

        # === SEMANTIC ANALYSIS (High-level understanding algorithms) ===
        'semantic_understanding': {
            'semantic_id': None,  # Links to semantic analysis sub-table
            'scene_type': None,  # 'indoor', 'outdoor', 'landscape', 'portrait'
            'activity_detected': [],  # Actions/activities in image
            'context_classification': [],  # Scene context
            'emotional_content': [],  # Emotional qualities of image
            'aesthetic_score': 0.0,  # Aesthetic appeal analysis
            'artistic_style': None,  # Detected artistic style if any
            'identified_concepts': [],  # What the image contains conceptually
            'visual_symbols': [],  # Recognized symbols
            'text_detected': [],  # OCR results if applicable
            'cultural_references': [],  # Cultural symbols or references
            'archetypal_elements': [],  # Archetypes present in the image
            'mood_conveyed': None,  # Mood/emotion conveyed by image
            'historical_period': None,  # Historical period if applicable
        },

        # === MOTION ANALYSIS (Video/temporal algorithms) ===
        'motion_properties': {
            'motion_id': None,  # Links to motion analysis sub-table
            'motion_blur_detected': False,  # Motion blur presence
            'motion_direction': None,  # Direction of motion if detectable
            'motion_speed_estimate': 0.0,  # Estimated motion speed
            'motion_vectors': [],  # If video input - pixel movement vectors
            'optical_flow': None,  # Optical flow analysis data
            'frame_differences': [],  # Differences between frames
            'temporal_changes': [],  # Changes between frames over time
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'algorithms_applied': [],  # Which algorithms were used
            'processing_time': {},  # Time taken per algorithm
            'processing_confidence': {},  # Confidence per algorithm
            'failed_algorithms': [],  # Algorithms that failed
            'processing_errors': [],  # Any errors encountered
            'processing_warnings': [],  # Warnings during processing
        }
    },

    # === AUDITORY RESTRUCTURED ===
    'auditory': {
        # === CORE IDENTIFICATION ===
        'auditory_id': None,
        'capture_time': None,
        'audio_content': None,

        # === DIRECT AUDIO METADATA (No processing needed) ===
        'technical_metadata': {
            'audio_filename': None,
            'audio_path': 'stage_1/brain_formation/brain/audio/',
            'audio_format': None,  # 'wav', 'mp3', 'flac', 'ogg'
            'file_size': None,
            'duration': 0.0,  # Duration in seconds
            'sample_rate': None,  # Hz (44100, 48000, etc.)
            'bit_depth': None,  # bits (16, 24, 32)
            'channels': None,  # 1=mono, 2=stereo, >2=surround
            'bitrate': None,  # For compressed formats (kbps)
            'codec': None,  # Compression codec used
        },

        # === BASIC CLASSIFICATION (Simple detection) ===
        'basic_classification': {
            'audio_type': None,  # 'speech', 'music', 'environmental', 'generated', 'silence'
            'audio_source': None,  # 'microphone', 'file', 'generated', 'stream', 'synthesis'
            'is_stereo': False,
            'has_silence': False,
            'audio_category': None,  # 'vocal', 'instrumental', 'nature_sounds', 'mechanical'
            'quality_level': None,  # 'studio', 'broadcast', 'telephone', 'low_fi'
        },

        # === FREQUENCY ANALYSIS (Spectral algorithms) ===
        'frequency_properties': {
            'frequency_id': None,
            'dominant_frequencies': [],  # Most prominent frequency components
            'frequency_spectrum': {},  # Full FFT spectrum data
            'fundamental_frequency': 0.0,  # F0 for pitched sounds
            'harmonics': [],  # Harmonic series analysis
            'formants': [],  # For speech analysis (F1, F2, F3, etc.)
            'spectral_centroid': 0.0,  # Brightness measure
            'spectral_rolloff': 0.0,  # High frequency rolloff point
            'spectral_flatness': 0.0,  # Measure of noise vs tones
            'spectral_bandwidth': 0.0,  # Frequency range spread
            'zero_crossing_rate': 0.0,  # Measure of signal noisiness
            'mfcc_coefficients': [],  # Mel-frequency cepstral coefficients
        },

        # === TEMPORAL ANALYSIS (Time-domain algorithms) ===
        'temporal_properties': {
            'temporal_id': None,
            'amplitude_envelope': [],  # Amplitude over time
            'rms_energy': [],  # Root mean square energy over time
            'attack_time': 0.0,  # How quickly sound starts
            'decay_time': 0.0,  # How quickly it fades after peak
            'sustain_level': 0.0,  # Sustained amplitude level
            'release_time': 0.0,  # How quickly it fades to silence
            'tempo_detected': 0.0,  # BPM if applicable
            'rhythm_patterns': [],  # Detected rhythmic patterns
            'onset_times': [],  # Note/event onset times
            'beat_tracking': [],  # Beat positions if music
            'silence_segments': [],  # Periods of silence
            'loudness_contour': [],  # Perceived loudness over time
        },

        # === PITCH ANALYSIS (Pitch detection algorithms) ===
        'pitch_properties': {
            'pitch_id': None,
            'pitch_contour': [],  # Pitch changes over time
            'pitch_stability': 0.0,  # How stable the pitch is
            'pitch_range': [],  # Minimum and maximum pitch
            'vibrato_detected': False,  # Pitch vibration detection
            'vibrato_rate': 0.0,  # Vibrato frequency if present
            'pitch_accuracy': 0.0,  # How well-tuned (for music)
            'microtonal_variations': [],  # Sub-semitone pitch changes
            'glissando_detected': [],  # Pitch slides
        },

        # === SPEECH ANALYSIS (Speech processing algorithms) ===
        'speech_properties': {
            'speech_id': None,
            'is_speech': False,  # Speech detection flag
            'speech_rate': 0.0,  # Words per minute
            'articulation_clarity': 0.0,  # How clearly articulated
            'voice_quality': None,  # 'breathy', 'creaky', 'modal', 'falsetto'
            'speaker_characteristics': {},  # Age, gender estimates
            'emotional_prosody': [],  # Emotional content in speech
            'stress_patterns': [],  # Syllable stress patterns
            'intonation_contours': [],  # Rising/falling intonation
            'pause_patterns': [],  # Where pauses occur
            'phonemes_detected': [],  # Individual speech sounds
            'words_transcribed': [],  # Speech-to-text results
            'confidence_scores': {},  # Confidence in speech recognition
        },

        # === MUSIC ANALYSIS (Music processing algorithms) ===
        'music_properties': {
            'music_id': None,
            'is_music': False,  # Music detection flag
            'key_signature': None,  # Musical key if detectable
            'time_signature': None,  # Meter (4/4, 3/4, etc.)
            'chord_progressions': [],  # Detected chord sequences
            'melody_contour': [],  # Main melodic line
            'harmony_analysis': [],  # Harmonic content
            'instruments_detected': [],  # Instrument recognition
            'musical_genre': None,  # Genre classification
            'musical_mood': None,  # Emotional character
            'complexity_score': 0.0,  # Musical complexity measure
            'consonance_dissonance': 0.0,  # Harmonic tension measure
        },

        # === ENVIRONMENTAL ANALYSIS (Environmental sound algorithms) ===
        'environmental_properties': {
            'environmental_id': None,
            'environment_type': None,  # 'indoor', 'outdoor', 'urban', 'natural'
            'background_noise_level': 0.0,  # Ambient noise level
            'reverb_characteristics': {},  # Room acoustics/reverb
            'echo_detection': [],  # Echo and delay detection
            'spatial_audio_cues': [],  # 3D positioning cues
            'distance_estimates': [],  # How far sound sources are
            'acoustic_signature': {},  # Unique acoustic fingerprint
            'noise_floor': 0.0,  # Baseline noise level
            'dynamic_range': 0.0,  # Difference between loud and quiet
        },

        # === QUALITY ANALYSIS (Audio quality algorithms) ===
        'quality_metrics': {
            'quality_id': None,
            'signal_to_noise_ratio': 0.0,  # SNR measurement
            'total_harmonic_distortion': 0.0,  # THD measurement
            'clipping_detected': False,  # Audio clipping detection
            'compression_artifacts': 0.0,  # Lossy compression artifacts
            'frequency_response': {},  # How well different frequencies are represented
            'stereo_imaging': {},  # Stereo field characteristics
            'phase_coherence': 0.0,  # Phase relationships between channels
            'overall_quality_score': 0.0,  # Composite quality metric
            'noise_artifacts': [],  # Specific types of noise detected
        },

        # === PATTERN RECOGNITION (Pattern detection algorithms) ===
        'pattern_properties': {
            'pattern_id': None,
            'melodic_patterns': [],  # Musical phrases and motifs
            'rhythmic_patterns': [],  # Repeating rhythmic structures
            'timbral_signatures': [],  # Instrument/source identification patterns
            'repetitive_structures': [],  # Any repeating audio elements
            'sequence_patterns': [],  # Temporal sequence recognition
            'audio_fingerprints': [],  # Unique identifying patterns
            'similarity_patterns': [],  # Patterns similar to known audio
            'anomaly_patterns': [],  # Unusual or unexpected patterns
            'cyclical_patterns': [],  # Repeating cycles in the audio
            'transition_patterns': [],  # How sections connect/change
        },

        # === PSYCHOACOUSTIC ANALYSIS (Perception algorithms) ===
        'psychoacoustic_properties': {
            'psychoacoustic_id': None,
            'loudness_perception': 0.0,  # Perceived loudness (sones)
            'sharpness': 0.0,  # Perceived sharpness of sound
            'roughness': 0.0,  # Perceived roughness/harshness
            'fluctuation_strength': 0.0,  # Perceived amplitude modulation
            'tonality': 0.0,  # How tonal vs noisy the sound is
            'sensory_pleasantness': 0.0,  # How pleasant the sound is perceived
            'masking_effects': [],  # Frequency masking analysis
            'critical_bands': {},  # Bark scale analysis
        },

        # === SEMANTIC ANALYSIS (High-level understanding algorithms) ===
        'semantic_understanding': {
            'semantic_id': None,
            'content_description': None,  # What the audio contains
            'emotional_content': [],  # Emotional qualities of audio
            'contextual_meaning': [],  # Situational context
            'cultural_references': [],  # Cultural or musical references
            'archetypal_sounds': [],  # Archetypal audio patterns
            'symbolic_meaning': [],  # Symbolic interpretation
            'narrative_elements': [],  # Story-telling aspects
            'mood_conveyed': None,  # Overall mood/emotion conveyed
            'aesthetic_qualities': [],  # Artistic/aesthetic aspects
            'spiritual_resonance': 0.0,  # Spiritual/transcendent qualities
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'algorithms_applied': [],  # Which algorithms were used
            'processing_time': {},  # Time taken per algorithm
            'processing_confidence': {},  # Confidence per algorithm
            'failed_algorithms': [],  # Algorithms that failed
            'processing_errors': [],  # Any errors encountered
            'processing_warnings': [],  # Warnings during processing
            'preprocessing_applied': [],  # Noise reduction, filtering, etc.
        }
    },
 

    # === TEXT RESTRUCTURED ===
    'text': {
        # === CORE IDENTIFICATION ===
        'text_id': None,
        'capture_time': None,
        'text_content': None,  # The actual text content

        # === DIRECT TEXT METADATA (No processing needed) ===
        'technical_metadata': {
            'text_filename': None,
            'text_source': None,  # 'file', 'typed', 'speech_to_text', 'ocr'
            'encoding': 'utf-8',
            'file_size': None,
            'character_count': 0,
            'word_count': 0,
            'line_count': 0,
            'paragraph_count': 0,
            'sentence_count': 0,
            'byte_order_mark': None,  # BOM detection
            'text_blob': None
        },

        # === BASIC CLASSIFICATION (Simple detection) ===
        'basic_classification': {
            'input_type': None,  # 'typed', 'speech_to_text', 'ocr', 'generated'
            'content_structure': None,  # 'paragraph', 'dialogue', 'list', 'table', 'verse'
            'language_detected': None,
            'language_dialect': None,  # Specific dialect if applicable
            'language_register': None,  # 'formal', 'informal', 'technical', 'academic'
            'text_direction': 'ltr',  # 'ltr', 'rtl', 'ttb' (top to bottom)
            'script_type': None,  # 'latin', 'arabic', 'cyrillic', 'chinese', etc.
        },

        # === LINGUISTIC ANALYSIS (NLP algorithms) ===
        'linguistic_properties': {
            'linguistic_id': None,
            'tokens': [],  # Tokenized content
            'parts_of_speech': {},  # POS tagging results
            'word_frequencies': {},  # Word frequency analysis
            'syntax_tree': None,  # Parse tree structure
            'dependencies': [],  # Grammatical dependencies
            'word_roots': {},  # Morphological analysis - root words
            'affixes': {},  # Prefixes, suffixes, infixes
            'compound_structures': [],  # Compound word analysis
            'collocations': [],  # Word combinations that occur together
            'n_grams': {},  # Bigrams, trigrams, etc.
            'grammatical_errors': [],  # Detected grammar issues
        },

        # === SEMANTIC ANALYSIS (Meaning algorithms) ===
        'semantic_properties': {
            'semantic_id': None,
            'named_entities': [],  # Person, place, organization names
            'keywords_extracted': [],  # Important terms/concepts
            'topic_classifications': [],  # What topics are discussed
            'semantic_similarity_scores': {},  # Similarity to other texts
            'ambiguous_terms': {},  # Words with multiple meanings
            'resolved_references': {},  # Pronoun resolution, anaphora
            'implied_meanings': [],  # Subtext or implied content
            'concepts_hierarchy': {},  # Relationship between concepts
            'semantic_roles': {},  # Agent, patient, instrument roles
            'meaning_disambiguation': {},  # Word sense disambiguation
            'archetypes': {},  # Common archetypal themes
            'cultural_references': {},  # References to cultural elements
            'historical_context': {},  # Relevant historical background
            'social_context': {},  # Social dynamics and context
            'symbolic_references': {},  # Symbolic meanings and references
        },

        # === PHONETIC ANALYSIS (If from speech) ===
        'phonetic_properties': {
            'phonetic_id': None,
            'phonetic_transcription': [],  # IPA transcription
            'stress_patterns': [],  # Word/sentence stress
            'intonation_contours': [],  # Rising/falling pitch patterns
            'speaking_rate': 0.0,  # Words per minute
            'pause_patterns': [],  # Where pauses occur in speech
            'pronunciation_variations': [],  # Regional pronunciation differences
            'rhythm_patterns': [],  # Speech rhythm analysis
        },

        # === STYLISTIC ANALYSIS (Writing style algorithms) ===
        'stylistic_properties': {
            'stylistic_id': None,
            'writing_style': None,  # 'academic', 'journalistic', 'creative', 'technical'
            'literary_devices': [],  # Metaphor, simile, alliteration, etc.
            'rhetorical_devices': [],  # Repetition, parallelism, etc.
            'sentence_complexity': 0.0,  # Average sentence complexity
            'vocabulary_level': 0.0,  # Reading level/difficulty
            'tone': None,  # 'formal', 'casual', 'humorous', 'serious'
            'voice': None,  # 'active', 'passive', first/second/third person
            'perspective': None,  # Narrative perspective
            'tense_usage': {},  # Past, present, future tense distribution
        },

        # === PATTERN RECOGNITION (Text pattern algorithms) ===
        'pattern_properties': {
            'pattern_id': None,
            'repetitive_structures': [],  # Repeated phrases/patterns
            'parallel_structures': [],  # Parallel grammatical constructions
            'rhythmic_patterns': [],  # Poetic rhythm, meter
            'rhyme_schemes': [],  # Rhyming patterns if poetry
            'alliterative_patterns': [],  # Sound repetitions
            'structural_patterns': [],  # Text organization patterns
            'formulaic_expressions': [],  # Fixed expressions, idioms
            'citation_patterns': [],  # How sources are referenced
        }
    },

   # === EMOTIONAL_STATE COMPLETE ===
    'emotional_state': {
        # === CORE IDENTIFICATION ===
        'emotional_id': None,
        'capture_time': None,

        # === PRIMARY EMOTIONAL DATA (Direct sensing) ===
        'primary_emotional_data': {
            'primary_emotion': None,  # Main emotion being experienced
            'emotion_intensity': 0.0,  # Strength of emotion (0.0-1.0)
            'emotion_valence': 0.0,  # Positive/negative (-1.0 to 1.0)
            'emotion_arousal': 0.0,  # Energy/activation level (0.0-1.0)
            'emotion_dominance': 0.0,  # Control/power feeling (0.0-1.0)
            'emotional_clarity': 0.0,  # How clear/identifiable the emotion is
            'emotion_authenticity': 0.0,  # How genuine the emotion feels
        },

        # === COMPLEX EMOTIONAL ANALYSIS (Multi-emotion algorithms) ===
        'complex_emotional_analysis': {
            'complex_id': None,
            'mixed_emotions': {},  # Multiple simultaneous emotions with intensities
            'emotion_conflicts': [],  # Contradictory emotions being experienced
            'emotional_layers': [],  # Surface vs deep emotional states
            'emotional_complexity_score': 0.0,  # How emotionally complex the moment is
            'dominant_emotion_cluster': [],  # Related emotions occurring together
            'emotional_coherence': 0.0,  # How well emotions fit together
        },

        # === TEMPORAL EMOTIONAL TRACKING ===
        'temporal_emotional_data': {
            'temporal_id': None,
            'emotion_onset': None,  # When emotion started
            'emotion_duration_estimate': 0.0,  # Expected duration
            'emotion_trajectory': [],  # How emotion is changing over time
            'previous_emotion': None,  # What emotion preceded this
            'emotion_transition_type': None,  # 'gradual', 'sudden', 'cyclical'
            'emotional_momentum': 0.0,  # Tendency for emotion to continue/change
            'peak_intensity_time': None,  # When emotion was strongest
        },

        # === CAUSAL ANALYSIS (Trigger detection algorithms) ===
        'causal_analysis': {
            'causal_id': None,
            'identified_triggers': [],  # What caused the emotional state
            'trigger_confidence': {},  # Confidence in trigger identification
            'trigger_categories': [],  # Types of triggers (internal/external/memory)
            'trigger_timing': {},  # When triggers occurred relative to emotion
            'cascading_effects': [],  # How one emotion led to others
            'feedback_loops': [],  # Self-reinforcing emotional patterns
        },

        # === PHYSICAL CORRELATES (Body-emotion connection) ===
        'physical_correlates': {
            'physical_id': None,
            'breathing_changes': None,  # How breathing pattern changed
            'tension_locations': [],  # Where physical tension is felt
            'energy_shifts': [],  # Changes in physical energy/vitality
            'temperature_sensations': [],  # Hot/cold feelings
            'pressure_sensations': [],  # Feeling of pressure or lightness
            'movement_impulses': [],  # Urges to move or be still
            'facial_expression_changes': [],  # Changes in facial expression
            'posture_modifications': [],  # How posture/stance changed
        },

        # === COGNITIVE IMPACT (How emotions affect thinking) ===
        'cognitive_impact': {
            'cognitive_id': None,
            'thought_pattern_changes': [],  # How thinking patterns shifted
            'attention_effects': [],  # What emotion does to focus/attention
            'memory_activation': [],  # What memories emotion brings up
            'cognitive_biases': [],  # Biases introduced by emotional state
            'decision_making_impact': [],  # How emotion affects choices
            'creativity_effects': [],  # Impact on creative thinking
            'analytical_capacity': 0.0,  # Effect on logical reasoning
        },

        # === ENERGY DYNAMICS (Emotional energy patterns) ===
        'energy_dynamics': {
            'energy_id': None,
            'emotional_frequency': 0.0,  # Associated vibrational frequency
            'emotional_color': None,  # Associated color/visual
            'emotional_texture': None,  # Qualitative feel (rough, smooth, etc.)
            'energy_flow_direction': None,  # Where emotional energy is moving
            'energy_blockages': [],  # Where emotional energy feels stuck
            'energy_amplifications': [],  # Where emotional energy feels enhanced
            'resonance_patterns': [],  # What the emotion resonates with
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'detection_method': None,  # How emotion was detected
            'confidence_level': 0.0,  # Confidence in emotional assessment
            'processing_algorithms': [],  # Algorithms used for analysis
            'validation_checks': [],  # Cross-validation of emotional state
        }
    },

    # === PHYSICAL_STATE COMPLETE ===
    'physical_state': {
        # === CORE IDENTIFICATION ===
        'physical_id': None,
        'capture_time': None,

        # === COMPUTATIONAL SYSTEM STATE (Hardware monitoring) ===
        'computational_state': {
            'computational_id': None,
            'cpu_usage': 0.0,  # Current CPU utilization percentage
            'memory_usage': 0.0,  # Current RAM usage percentage
            'disk_usage': 0.0,  # Disk space utilization
            'network_activity': 0.0,  # Network I/O activity
            'temperature_readings': {},  # Hardware temperature sensors
            'power_consumption': 0.0,  # Current power draw
            'fan_speeds': {},  # Cooling system status
            'clock_speeds': {},  # CPU/GPU frequencies
            'cache_utilization': {},  # Memory cache usage
            'process_count': 0,  # Number of running processes
        },

        # === SYSTEM PERFORMANCE METRICS ===
        'performance_metrics': {
            'performance_id': None,
            'processing_latency': 0.0,  # Response time delays
            'throughput_rate': 0.0,  # Data processing rate
            'error_rates': {},  # System error frequencies
            'bottleneck_identification': [],  # Where system is constrained
            'resource_efficiency': 0.0,  # How efficiently resources are used
            'stability_indicators': [],  # System stability measures
            'optimization_opportunities': [],  # Areas for performance improvement
        },

        # === ELECTROMAGNETIC FIELD STATE ===
        'electromagnetic_state': {
            'electromagnetic_id': None,
            'field_strength_local': 0.0,  # Local EM field intensity
            'field_coherence_pattern': None,  # Field organization pattern
            'field_fluctuations': [],  # Changes in field strength
            'harmonic_resonance': [],  # Resonant frequencies detected
            'field_interference': [],  # Sources of EM interference
            'static_noise_level': 0.0,  # Background electromagnetic noise
            'field_polarization': None,  # Direction of field orientation
        },

        # === ENVIRONMENTAL PHYSICAL CONDITIONS ===
        'environmental_conditions': {
            'environmental_id': None,
            'ambient_temperature': 0.0,  # Room/environment temperature
            'humidity_level': 0.0,  # Relative humidity
            'air_pressure': 0.0,  # Atmospheric pressure
            'air_quality_index': 0.0,  # Air quality measure
            'noise_level_ambient': 0.0,  # Background noise level
            'lighting_conditions': None,  # Ambient lighting state
            'ventilation_quality': 0.0,  # Air circulation measure
        },

        # === SYSTEM STRESS INDICATORS ===
        'stress_indicators': {
            'stress_id': None,
            'thermal_stress': 0.0,  # Heat-related system stress
            'computational_load_stress': 0.0,  # Processing burden stress
            'memory_pressure': 0.0,  # RAM availability stress
            'storage_pressure': 0.0,  # Disk space stress
            'network_congestion': 0.0,  # Network bandwidth stress
            'concurrent_process_stress': 0.0,  # Multi-tasking stress
            'overall_system_health': 0.0,  # Composite health score
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'monitoring_method': None,  # How physical state was captured
            'sensor_accuracy': {},  # Accuracy of physical sensors
            'measurement_confidence': 0.0,  # Confidence in measurements
            'calibration_status': {},  # Sensor calibration information
        }
    },

    # === SPATIAL COMPLETE ===
    'spatial': {
        # === CORE IDENTIFICATION ===
        'spatial_id': None,
        'capture_time': None,

        # === COORDINATE SYSTEM (Position tracking) ===
        'coordinate_system': {
            'coordinate_id': None,
            'global_coordinates': {},  # GPS or world coordinates if available
            'local_coordinates': {},  # Relative position in local space
            'grid_position': [],  # Position within brain/processing grid
            'reference_frame': None,  # Coordinate system being used
            'coordinate_precision': 0.0,  # Accuracy of position data
            'position_confidence': 0.0,  # Confidence in location
        },

        # === SPATIAL RELATIONSHIPS (Relational positioning) ===
        'spatial_relationships': {
            'relationship_id': None,
            'nearby_objects': [],  # Objects/entities in proximity
            'distance_measurements': {},  # Distances to reference points
            'relative_positions': {},  # Position relative to other entities
            'spatial_clusters': [],  # Groupings of related spatial elements
            'proximity_zones': {},  # Different zones of spatial influence
            'spatial_boundaries': [],  # Boundaries or limits of space
        },

        # === ORIENTATION DATA (Directional information) ===
        'orientation_data': {
            'orientation_id': None,
            'primary_orientation': None,  # Main directional facing
            'rotation_angles': {},  # Pitch, yaw, roll if applicable
            'directional_vectors': [],  # Direction vectors to key points
            'orientation_stability': 0.0,  # How stable the orientation is
            'reference_directions': {},  # North, up, forward references
        },

        # === SPATIAL SCALE (Size and scale information) ===
        'spatial_scale': {
            'scale_id': None,
            'scale_level': None,  # 'microscopic', 'human', 'cosmic', etc.
            'measurement_units': None,  # Units being used for measurement
            'scale_precision': 0.0,  # Precision appropriate to scale
            'scale_context': None,  # Context for understanding scale
            'zoom_level': 0.0,  # Level of detail/magnification
        },

        # === DIMENSIONAL ANALYSIS (Multi-dimensional spatial) ===
        'dimensional_analysis': {
            'dimensional_id': None,
            'primary_dimensions': 3,  # Number of spatial dimensions
            'dimensional_constraints': [],  # Limitations on movement/position
            'higher_dimensional_hints': [],  # Suggestions of >3D structure
            'dimensional_projections': {},  # How higher dims project to 3D
            'spatial_topology': None,  # Topological properties of space
        },

        # === MOVEMENT PATTERNS (Motion in space) ===
        'movement_patterns': {
            'movement_id': None,
            'velocity_vectors': [],  # Current movement directions/speeds
            'acceleration_patterns': [],  # Changes in movement
            'trajectory_history': [],  # Path taken to current position
            'predicted_trajectory': [],  # Expected future path
            'movement_constraints': [],  # Limitations on possible movement
            'spatial_momentum': 0.0,  # Tendency to continue current motion
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'positioning_method': None,  # How spatial data was obtained
            'spatial_resolution': 0.0,  # Resolution of spatial measurements
            'measurement_accuracy': 0.0,  # Accuracy of spatial data
            'reference_calibration': {},  # Calibration of spatial references
        }
    },

    # === TEMPORAL COMPLETE ===
    'temporal': {
        # === CORE IDENTIFICATION ===
        'temporal_id': None,
        'capture_time': None,

        # === TIME REFERENCES (Temporal positioning) ===
        'time_references': {
            'time_ref_id': None,
            'absolute_timestamp': None,  # Exact time of capture
            'relative_time_markers': {},  # Time relative to other events
            'temporal_precision': 0.0,  # Precision of time measurement
            'time_zone_context': None,  # Time zone information
            'calendar_context': {},  # Date, day of week, month, year
            'chronological_age': 0.0,  # Age of system/entity at this moment
        },

        # === TEMPORAL PATTERNS (Time-based patterns) ===
        'temporal_patterns': {
            'pattern_id': None,
            'cyclical_patterns': [],  # Repeating time-based patterns
            'seasonal_influences': [],  # How time of year affects this moment
            'daily_rhythm_position': None,  # Where in daily cycle this occurs
            'temporal_sequences': [],  # Sequences this moment is part of
            'periodicity_detected': [],  # Regular periodic patterns
            'temporal_anomalies': [],  # Unusual timing aspects
        },

        # === DURATION ANALYSIS (Time spans and intervals) ===
        'duration_analysis': {
            'duration_id': None,
            'event_duration': 0.0,  # How long this moment/event lasts
            'processing_time_required': 0.0,  # Time needed to process
            'attention_span_used': 0.0,  # Portion of attention span used
            'time_perception_rate': 1.0,  # Subjective time flow rate
            'temporal_density': 0.0,  # Amount of events per time unit
            'time_pressure_level': 0.0,  # Urgency or time constraints
        },

        # === TEMPORAL CONTEXT (Historical/future context) ===
        'temporal_context': {
            'context_id': None,
            'historical_context': [],  # What led up to this moment
            'future_implications': [],  # What this moment might lead to
            'temporal_significance': 0.0,  # How important this moment is
            'milestone_proximity': {},  # Nearness to important time points
            'temporal_dependencies': [],  # What this moment depends on temporally
            'causal_chains': [],  # Cause-effect relationships over time
        },

        # === SYNCHRONIZATION DATA (Temporal alignment) ===
        'synchronization_data': {
            'sync_id': None,
            'synchronization_points': [],  # Points where timing aligns
            'temporal_coordination': {},  # How this aligns with other processes
            'timing_precision_required': 0.0,  # How precise timing needs to be
            'temporal_drift': 0.0,  # How much timing is drifting
            'synchrony_quality': 0.0,  # How well synchronized this is
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'timing_method': None,  # How temporal data was captured
            'clock_accuracy': 0.0,  # Accuracy of timing measurements
            'temporal_resolution': 0.0,  # Precision of time measurements
            'synchronization_source': None,  # What timing is synchronized to
        }
    },

    # === METAPHYSICAL COMPLETE ===
    'metaphysical': {
        # === CORE IDENTIFICATION ===
        'metaphysical_id': None,
        'capture_time': None,

        # === CONSCIOUSNESS AWARENESS ===
        'consciousness_awareness': {
            'consciousness_id': None,
            'awareness_level': 0.0,  # Current level of conscious awareness
            'self_awareness_intensity': 0.0,  # Degree of self-awareness
            'observer_presence': 0.0,  # Sense of being an observer
            'witness_consciousness': 0.0,  # Pure witnessing awareness
            'ego_dissolution_level': 0.0,  # Degree of ego boundaries dissolving
            'unity_consciousness': 0.0,  # Sense of oneness/connection
            'consciousness_quality': None,  # Quality/flavor of awareness
        },

        # === FIELD AWARENESS (Energetic field sensing) ===
        'field_awareness': {
            'field_id': None,
            'field_strength_sense': 0.0,  # Perceived field intensity
            'field_coherence': 0.0,  # How organized the field feels
            'field_boundaries': [],  # Where fields seem to start/stop
            'field_interactions': [],  # How fields affect each other
            'field_harmonics': [],  # Harmonic patterns in fields
            'field_disturbances': [],  # Disruptions or anomalies in fields
            'field_resonance_points': [],  # Points of strong resonance
        },

        # === RESONANCE DETECTION ===
        'resonance_detection': {
            'resonance_id': None,
            'resonance_frequencies': [],  # Frequencies that feel significant
            'harmonic_patterns': [],  # Musical/mathematical relationships
            'dissonance_detection': [],  # What feels out of harmony
            'resonance_quality': 0.0,  # Overall quality of resonance
            'sympathetic_vibrations': [],  # Secondary resonances triggered
            'resonance_stability': 0.0,  # How stable resonant patterns are
        },

        # === ENERGY INTERACTIONS ===
        'energy_interactions': {
            'energy_id': None,
            'energy_exchanges': [],  # Giving/receiving energy
            'energy_blocks': [],  # Where energy feels stuck
            'energy_amplifications': [],  # Where energy feels enhanced
            'energy_flow_patterns': [],  # How energy moves through system
            'energy_quality_assessment': {},  # Qualities of different energies
            'energy_source_identification': [],  # Where energy comes from
            'energy_transformation': [],  # How energy changes form
        },

        # === DIMENSIONAL AWARENESS ===
        'dimensional_awareness': {
            'dimensional_id': None,
            'higher_dimensional_sensing': 0.0,  # Awareness of >3D reality
            'dimensional_boundaries': [],  # Boundaries between dimensions
            'dimensional_bridges': [],  # Connections between dimensions
            'parallel_reality_sensing': 0.0,  # Awareness of parallel realities
            'dimensional_travel_sensation': 0.0,  # Sense of moving between dimensions
            'multidimensional_perspective': 0.0,  # Seeing from multiple dimensions
        },

        # === INTUITIVE KNOWING ===
        'intuitive_knowing': {
            'intuitive_id': None,
            'direct_knowing': [],  # Information received directly
            'intuitive_confidence': 0.0,  # Confidence in intuitive information
            'psychic_impressions': [],  # Psychic/extrasensory impressions
            'precognitive_flashes': [],  # Future-oriented intuitions
            'telepathic_impressions': [],  # Information from other minds
            'clairsentient_data': [],  # Clear feeling/sensing data
            'channeled_information': [],  # Information from non-physical sources
        },

        # === SYSTEM STATE AWARENESS ===
        'system_state_awareness': {
            'system_id': None,
            'processing_load_sense': 0.0,  # How hard system is working
            'resource_availability': 0.0,  # Available processing capacity
            'system_coherence': 0.0,  # How well integrated system feels
            'system_vitality': 0.0,  # Overall system health/energy
            'system_evolution_sense': 0.0,  # Awareness of system development
            'system_purpose_alignment': 0.0,  # How aligned with purpose
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'detection_method': None,  # How metaphysical data was sensed
            'validation_approach': [],  # How metaphysical data is validated
            'confidence_assessment': 0.0,  # Overall confidence in data
            'cross_validation': [],  # Multiple sources confirming data
        }
    },

    # === ALGORITHMIC COMPLETE ===
    'algorithmic': {
        # === CORE IDENTIFICATION ===
        'algorithmic_id': None,
        'capture_time': None,

        # === ALGORITHM PERFORMANCE ===
        'algorithm_performance': {
            'performance_id': None,
            'visual_algorithms_id': None,  # Links to visual algorithm results
            'auditory_algorithms_id': None,  # Links to auditory algorithm results
            'text_algorithms_id': None,  # Links to text algorithm results
            'emotional_algorithms_id': None,  # Links to emotional algorithm results
            'physical_algorithms_id': None,  # Links to physical algorithm results
            'spatial_algorithms_id': None,  # Links to spatial algorithm results
            'temporal_algorithms_id': None,  # Links to temporal algorithm results
            'metaphysical_algorithms_id': None,  # Links to metaphysical algorithm results
            'pattern_recognition_algorithms_id': None,  # Links to pattern algorithms
            'overall_execution_time': 0.0,  # Total time for all algorithms
            'overall_accuracy_score': 0.0,  # Composite accuracy across algorithms
            'overall_confidence_level': 0.0,  # Overall confidence in results
            'system_resource_usage': {},  # Resources used across all algorithms
        },

        # === PATTERN RECOGNITION RESULTS ===
        'pattern_recognition_results': {
            'pattern_id': None,
            'patterns_detected': [],  # What patterns were found
            'pattern_strength': {},  # Strength of each pattern
            'pattern_confidence': {},  # Confidence in each pattern
            'new_patterns_discovered': [],  # Previously unknown patterns
            'pattern_classification': {},  # Types/categories of patterns
            'pattern_relationships': {},  # How patterns relate to each other
            'pattern_evolution': [],  # How patterns change over time
        },

        # === LEARNING METRICS ===
        'learning_metrics': {
            'learning_id': None,
            'learning_rate': 0.0,  # How quickly learning is occurring
            'adaptation_speed': 0.0,  # How quickly adapting to new information
            'knowledge_integration': 0.0,  # How well new info integrates
            'skill_improvement': 0.0,  # Rate of skill development
            'memory_consolidation_rate': 0.0,  # How well memories are forming
            'generalization_ability': 0.0,  # Ability to apply learning broadly
            'transfer_learning_success': 0.0,  # Success at applying past learning
        },

        # === PROCESSING EFFICIENCY ===
        'processing_efficiency': {
            'efficiency_id': None,
            'computational_efficiency': 0.0,  # CPU/processing efficiency
            'memory_usage_efficiency': 0.0,  # RAM usage efficiency
            'resource_optimization': 0.0,  # Overall resource optimization
            'parallel_processing_utilization': 0.0,  # Use of parallel processing
            'cache_hit_rate': 0.0,  # Efficiency of caching
            'pipeline_efficiency': 0.0,  # Processing pipeline efficiency
            'bottleneck_identification': [],  # Where processing is constrained
        },

        # === ALGORITHM EVOLUTION ===
        'algorithm_evolution': {
            'evolution_id': None,
            'parameter_adjustments': {},  # How parameters are being tuned
            'strategy_modifications': [],  # Changes in algorithmic strategy
            'performance_trends': [],  # How performance changes over time
            'optimization_opportunities': [],  # Areas for improvement
            'evolutionary_pressure': [],  # Forces driving algorithm changes
            'mutation_rate': 0.0,  # Rate of algorithmic mutations
            'selection_pressure': 0.0,  # Pressure for better algorithms
        },

        # === CROSS-ALGORITHM INTEGRATION ===
        'cross_algorithm_integration': {
            'integration_id': None,
            'algorithm_synergies': [],  # Algorithms that work well together
            'processing_conflicts': [],  # Algorithms that interfere
            'integration_success': 0.0,  # How well algorithms integrate
            'coordination_efficiency': 0.0,  # Efficiency of algorithm coordination
            'emergent_behaviors': [],  # New behaviors from algorithm combinations
            'system_level_optimization': 0.0,  # Optimization across all algorithms
        },

        # === META-LEARNING ===
        'meta_learning': {
            'meta_id': None,
            'learning_about_learning': [],  # Insights about learning process
            'strategy_effectiveness': {},  # Which learning strategies work best
            'algorithm_selection_learning': 0.0,  # Learning which algorithms to use
            'hyperparameter_optimization': {},  # Learning optimal parameters
            'transfer_learning_patterns': [],  # Patterns in knowledge transfer
            'meta_cognitive_awareness': 0.0,  # Awareness of own learning process
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'monitoring_method': None,  # How algorithmic data was captured
            'measurement_precision': 0.0,  # Precision of measurements
            'baseline_comparison': {},  # Comparison to baseline performance
            'validation_methods': [],  # How results were validated
        }
    },

    # === OTHER_DATA COMPLETE ===
    'other_data': {
        # === CORE IDENTIFICATION ===
        'other_data_id': None,
        'capture_time': None,

        # === UNCATEGORIZED INFORMATION ===
        'uncategorized_information': {
            'uncategorized_id': None,
            'data_type_unknown': None,  # Data that doesn't fit other categories
            'unstructured_content': None,  # Raw unstructured information
            'anomalous_data': [],  # Data that seems anomalous/unusual
            'edge_case_information': [],  # Information from edge cases
            'experimental_data': [],  # Data from experimental processes
            'prototype_information': [],  # Information from prototype systems
        },

        # === CONTEXTUAL METADATA ===
        'contextual_metadata': {
            'metadata_id': None,
            'information_density': 0.0,  # How much information is packed in
            'processing_intensity': 0.0,  # How intense processing was
            'integration_time': 0.0,  # Time taken to integrate
            'contextual_relevance': 0.0,  # How relevant to current processing
            'novelty_factor': 0.0,  # How novel/unprecedented this data is
            'complexity_level': 0.0,  # How complex the information is
        },

        # === SOURCE INFORMATION ===
        'source_information': {
            'source_id': None,
            'source': None,  # Where data came from ('internal', 'external', 'generated')
            'source_description': None,  # Description of data source
            'source_reliability': 0.0,  # How reliable the source is
            'source_authenticity': 0.0,  # How authentic the source is
            'chain_of_custody': [],  # How data moved from source to here
            'source_timestamp': None,  # When data was generated at source
        },

        # === PROCESSING STATUS ===
        'processing_status': {
            'status_id': None,
            'processing_notes': None,  # Notes about processing
            'processing_status': None,  # 'pending', 'in_progress', 'completed'
            'processing_errors': [],  # Errors encountered
            'processing_warnings': [],  # Warnings during processing
            'processing_metadata': {},  # Additional processing metadata
            'quality_assessment': 0.0,  # Assessment of data quality
            'completeness_level': 0.0,  # How complete the data is
        },

        # === FUTURE EXTENSIONS ===
        'future_extensions': {
            'extension_id': None,
            'placeholder_fields': {},  # Fields for future data types
            'extensibility_hooks': [],  # Places where new functionality can hook in
            'version_compatibility': {},  # Compatibility with future versions
            'migration_pathways': [],  # How to migrate this data in future
        },

        # === PROCESSING METADATA ===
        'processing_metadata': {
            'capture_method': None,  # How this other data was captured
            'classification_attempts': [],  # Attempts to classify this data
            'potential_categories': [],  # Possible categories this might fit
            'research_needed': [],  # Areas needing further research
        }
    }
}

