"""
SENSORY_PATTERNS Dictionary - Individual Sense Analysis + Basic Cross-Modal Patterns
Analyzes patterns of each sense individually in detail, then does basic cross-modal
pattern analysis. Includes basic meaning and interpretation from pattern analysis.
"""

SENSORY_PATTERNS = {
    # === CORE IDENTIFICATION ===
    'sensory_patterns_id': None,
    'sensory_raw_id': None,  # Links back to the sensory_raw capture
    'pattern_analysis_start_time': None,
    'pattern_analysis_end_time': None,
    'mycelial_network_id': None,

    # === PARTICIPATING SENSORY DATA ===
    'participating_senses': {
        'active_sensory_types': [],  # Which of the 10 senses have data
        'sensory_data_quality': {},  # Quality score per sense (0.0-1.0)
        'primary_sense': None,  # Dominant sense in this capture
        'sensory_completeness': 0.0,  # How much sensory data captured overall
    },

    # === INDIVIDUAL SENSE PATTERN ANALYSIS ===
    'individual_sense_patterns': {
        # Visual Pattern Analysis (if present)
        'visual_patterns': {
            'visual_id': None,  # Links to visual data
            'dominant_colors': [],  # Main colors detected
            'primary_objects': [],  # Objects identified
            'scene_type': None,  # Indoor/outdoor/portrait/landscape
            'composition_patterns': [],  # Rule of thirds, symmetry, etc.
            'visual_complexity': 0.0,  # Complexity score (0.0-1.0)
            'motion_patterns': [],  # Movement or stillness patterns
            'lighting_patterns': [],  # Light/shadow patterns
            'texture_patterns': [],  # Surface texture patterns
            'spatial_patterns': [],  # Depth and spatial arrangement
            'pattern_confidence': 0.0,  # Confidence in visual analysis
            'basic_visual_story': None,  # Basic interpretation of what's seen
        },
        
        # Auditory Pattern Analysis (if present)
        'auditory_patterns': {
            'auditory_id': None,  # Links to auditory data
            'sound_type': None,  # Speech, music, environmental, silence
            'dominant_frequencies': [],  # Main frequency components
            'rhythm_patterns': [],  # Rhythmic structures detected
            'volume_patterns': [],  # Loudness variation patterns
            'temporal_patterns': [],  # Time-based sound patterns
            'harmonic_patterns': [],  # Harmonic relationships
            'noise_patterns': [],  # Background noise characteristics
            'audio_complexity': 0.0,  # Complexity score (0.0-1.0)
            'pattern_confidence': 0.0,  # Confidence in audio analysis
            'basic_audio_story': None,  # Basic interpretation of what's heard
        },
        
        # Text Pattern Analysis (if present)
        'text_patterns': {
            'text_id': None,  # Links to text data
            'language_patterns': [],  # Language structure patterns
            'semantic_patterns': [],  # Meaning patterns detected
            'emotional_patterns': [],  # Emotional tone patterns
            'complexity_patterns': [],  # Text complexity patterns
            'topic_patterns': [],  # Subject matter patterns
            'style_patterns': [],  # Writing style patterns
            'structure_patterns': [],  # Organizational patterns
            'text_complexity': 0.0,  # Complexity score (0.0-1.0)
            'pattern_confidence': 0.0,  # Confidence in text analysis
            'basic_text_story': None,  # Basic interpretation of text meaning
        },
        
        # Emotional State Pattern Analysis (if present)
        'emotional_patterns': {
            'emotional_state_id': None,  # Links to emotional data
            'primary_emotion_pattern': None,  # Main emotional pattern
            'emotional_intensity_pattern': 0.0,  # Intensity patterns
            'emotional_change_patterns': [],  # How emotions change
            'emotional_trigger_patterns': [],  # What triggers emotions
            'emotional_complexity': 0.0,  # Emotional complexity score
            'pattern_confidence': 0.0,  # Confidence in emotional analysis
            'basic_emotional_story': None,  # Basic interpretation of emotional state
        },
        
        # Physical State Pattern Analysis (if present)
        'physical_patterns': {
            'physical_state_id': None,  # Links to physical data
            'system_performance_patterns': [],  # System performance patterns
            'resource_usage_patterns': [],  # Resource utilization patterns
            'environmental_patterns': [],  # Environmental condition patterns
            'energy_patterns': [],  # Energy consumption patterns
            'physical_complexity': 0.0,  # Physical state complexity
            'pattern_confidence': 0.0,  # Confidence in physical analysis
            'basic_physical_story': None,  # Basic interpretation of physical state
        },
        
        # Spatial Pattern Analysis (if present)
        'spatial_patterns': {
            'spatial_id': None,  # Links to spatial data
            'location_patterns': [],  # Position-related patterns
            'movement_patterns': [],  # Movement or positioning patterns
            'orientation_patterns': [],  # Directional patterns
            'scale_patterns': [],  # Size and scale patterns
            'spatial_complexity': 0.0,  # Spatial complexity score
            'pattern_confidence': 0.0,  # Confidence in spatial analysis
            'basic_spatial_story': None,  # Basic interpretation of spatial context
        },
        
        # Temporal Pattern Analysis (if present)
        'temporal_patterns': {
            'temporal_id': None,  # Links to temporal data
            'timing_patterns': [],  # Time-related patterns
            'sequence_patterns': [],  # Sequential patterns
            'duration_patterns': [],  # Duration-related patterns
            'cyclical_patterns': [],  # Repeating time patterns
            'temporal_complexity': 0.0,  # Temporal complexity score
            'pattern_confidence': 0.0,  # Confidence in temporal analysis
            'basic_temporal_story': None,  # Basic interpretation of temporal context
        },
        
        # Metaphysical Pattern Analysis (if present)
        'metaphysical_patterns': {
            'metaphysical_id': None,  # Links to metaphysical data
            'consciousness_patterns': [],  # Awareness-related patterns
            'energy_field_patterns': [],  # Field-related patterns
            'resonance_patterns': [],  # Resonance patterns detected
            'intuitive_patterns': [],  # Intuitive information patterns
            'metaphysical_complexity': 0.0,  # Metaphysical complexity score
            'pattern_confidence': 0.0,  # Confidence in metaphysical analysis
            'basic_metaphysical_story': None,  # Basic interpretation of metaphysical state
        },
        
        # Algorithmic Pattern Analysis (if present)
        'algorithmic_patterns': {
            'algorithmic_id': None,  # Links to algorithmic data
            'performance_patterns': [],  # Algorithm performance patterns
            'learning_patterns': [],  # Learning-related patterns
            'efficiency_patterns': [],  # Efficiency patterns
            'error_patterns': [],  # Error and correction patterns
            'algorithmic_complexity': 0.0,  # Algorithmic complexity score
            'pattern_confidence': 0.0,  # Confidence in algorithmic analysis
            'basic_algorithmic_story': None,  # Basic interpretation of algorithmic state
        },
        
        # Other Data Pattern Analysis (if present)
        'other_patterns': {
            'other_data_id': None,  # Links to other data
            'unclassified_patterns': [],  # Patterns that don't fit other categories
            'emergent_patterns': [],  # Unexpected patterns discovered
            'anomaly_patterns': [],  # Anomalous patterns detected
            'other_complexity': 0.0,  # Other data complexity score
            'pattern_confidence': 0.0,  # Confidence in other data analysis
            'basic_other_story': None,  # Basic interpretation of other data
        },
    },

    # === BASIC CROSS-MODAL PATTERN ANALYSIS ===
    'cross_modal_patterns': {
        # Basic Correlations Between Senses
        'correlation_matrix': {},  # Correlation coefficients between available senses
        'similarity_scores': {},  # Similarity measures between senses
        'complementary_patterns': [],  # How senses complement each other
        'conflicting_patterns': [],  # Where senses provide conflicting information
        'reinforcing_patterns': [],  # Where senses reinforce each other
        
        # Temporal Alignment
        'temporal_synchronization': {},  # How senses align in time
        'sequence_patterns': [],  # Sequential relationships between senses
        'timing_correlations': {},  # Timing-based correlations
        
        # Basic Cross-Modal Stories
        'integrated_story_elements': [],  # Story elements that span multiple senses
        'cross_modal_themes': [],  # Themes that appear across senses
        'multi_sensory_coherence': 0.0,  # How well senses fit together (0.0-1.0)
    },

    # === PATTERN STRENGTH ANALYSIS ===
    'pattern_strength': {
        'individual_pattern_strengths': {},  # Strength per sense (0.0-1.0)
        'cross_modal_pattern_strength': 0.0,  # Overall cross-modal strength
        'pattern_confidence_scores': {},  # Confidence per pattern type
        'statistical_significance': {},  # Statistical significance measures
        'pattern_consistency': 0.0,  # How consistent patterns are
        'pattern_clarity': 0.0,  # How clear/distinct patterns are
    },

    # === BASIC MEANING INTERPRETATION ===
    'basic_meaning': {
        'primary_story': None,  # Main story/meaning from pattern analysis
        'supporting_themes': [],  # Supporting themes identified
        'context_summary': None,  # Basic context from all available senses
        'emotional_tone': None,  # Overall emotional character
        'significance_level': 0.0,  # How significant this capture seems (0.0-1.0)
        'coherence_assessment': 0.0,  # How coherent the overall meaning is
        'novelty_assessment': 0.0,  # How novel/unusual this seems
        'meaning_confidence': 0.0,  # Confidence in basic meaning interpretation
    },

    # === PROCESSING QUALITY ===
    'processing_quality': {
        'analysis_completeness': 0.0,  # How complete the analysis is (0.0-1.0)
        'data_quality_assessment': 0.0,  # Quality of input data
        'pattern_detection_quality': 0.0,  # Quality of pattern detection
        'cross_modal_analysis_quality': 0.0,  # Quality of cross-modal analysis
        'overall_processing_quality': 0.0,  # Overall quality score
        'processing_warnings': [],  # Any warnings during processing
        'processing_errors': [],  # Any errors encountered
    },

    # === FRAGMENT READINESS ===
    'fragment_readiness': {
        'ready_for_fragment_creation': False,  # Ready for advanced fragment processing
        'minimum_quality_met': False,  # Minimum quality thresholds met
        'pattern_analysis_complete': False,  # Pattern analysis completed successfully
        'basic_meaning_extracted': False,  # Basic meaning successfully extracted
        'cross_modal_analysis_sufficient': False,  # Cross-modal analysis sufficient
    },

    # === PROCESSING METADATA ===
    'processing_metadata': {
        'algorithms_used': [],  # Which algorithm categories were used
        'processing_time_per_sense': {},  # Processing time per sense
        'total_processing_time': 0.0,  # Total processing duration
        'computational_resources_used': {},  # Resources consumed
        'algorithm_effectiveness': {},  # How effective different approaches were
    }
}














