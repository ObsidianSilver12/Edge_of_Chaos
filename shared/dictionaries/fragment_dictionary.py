"""
This module defines the FRAGMENT dictionary. Fragments are consolidated
memory structures created by the mycelial network from raw sensory data
and patterns. They represent the intermediate stage between raw sensory
capture and final nodes.
"""

FRAGMENT = {
    # === Core Identification & Lineage ===
    'fragment_id': None,  # Unique fragment identifier
    # ARCHITECTURAL IMPROVEMENT: Explicitly links this fragment to the
    # coordination event that created it, ensuring a traceable history.
    'source_coordination_id': None,
    'creation_time': None,  # When fragment was created
    'source_sensory_types': [],  # List of sensory types that contributed to this fragment
    'mycelial_network_id': None,  # ID of mycelial network that created this fragment

    # === Processing State (Automation Flag) ===
    'processing_stage': 'raw',  # 'raw', 'patterns_identified', 'consolidated', 'ready_for_node'
    'node_creation_flag': False,  # Flag indicating readiness for node conversion
    'validation_status': 'pending',  # 'pending', 'in_progress', 'validated', 'rejected'
    'last_updated': None,

    # === Raw Sensory Data Collection ===
    # ARCHITECTURAL IMPROVEMENT: This section is now for holding IDs, not
    # full data, to prevent duplication. The full data is in SENSORY_RAW.
    'raw_sensory_data': {
        'visual_id': None,  # Visual sensory data ID if present
        'auditory_id': None,  # Auditory sensory data ID if present
        'text_id': None,  # Text/linguistic data ID if present
        'emotional_state_id': None,  # Emotional state data ID if present
        'physical_state_id': None,  # Physical/computational state ID if present
        'spatial_id': None,  # Spatial positioning and orientation ID if present
        'temporal_id': None,  # Temporal awareness and timing ID if present
        'metaphysical_id': None,  # Beyond-physical awareness ID if present
        'algorithmic_id': None,  # Algorithm performance and learning ID if present
        'other_data_id': None  # Other sensory data ID if present
    },

    # === Pattern Recognition Results ===
    'identified_patterns': {
        'cross_sensory_patterns': [],  # Patterns identified across sensory types
        'individual_patterns': {},  # Patterns within individual sensory types
        'pattern_correlations': {},  # Correlation scores between patterns
        'pattern_confidence': {},  # Confidence scores for identified patterns
        'pattern_metadata': {}  # Additional pattern information
    },

    # === Semantic Labeling ===
    'semantic_labels': {
        'primary_concepts': [],  # Main concepts identified
        'secondary_concepts': [],  # Related concepts
        'descriptive_labels': [],  # Descriptive tags
        'categorical_labels': [],  # Category assignments
        'emotional_labels': [],  # Emotional content labels
        'contextual_labels': []  # Contextual information
    },
    
    # === Basic Semantic Meaning ===
    'basic_meaning': {
        'summary': None,  # Brief summary of fragment content
        'significance': 0.0,  # Significance score (0.0-1.0)
        'coherence': 0.0,  # Internal coherence score (0.0-1.0)
        'completeness': 0.0,  # Data completeness score (0.0-1.0)
        'novelty': 0.0,  # Novelty score (0.0-1.0)
        'clarity': 0.0  # Clarity score (0.0-1.0)
    },
    
    # === Memory Type Assignment ===
    'assigned_memory_type': {
        'memory_type_id': None,
        'memory_frequency_hz': None,
        'decay_rate': None,
        'preferred_storage_duration_hours': None,
        'assignment_confidence': 0.0
    },
    
    # === Energy and Processing Metrics ===
    'energy_metrics': {
        'creation_energy_used': 0.0,
        'processing_energy_used': 0.0,
        'pattern_recognition_energy': 0.0,
        'semantic_processing_energy': 0.0,
        'total_energy_consumed': 0.0
    },
    
    # === Quality Scores ===
    'quality_scores': {
        'data_quality': 0.0,  # Quality of raw sensory data (0.0-1.0)
        'pattern_quality': 0.0,  # Quality of pattern identification (0.0-1.0)
        'semantic_quality': 0.0,  # Quality of semantic labeling (0.0-1.0)
        'overall_quality': 0.0,  # Overall fragment quality (0.0-1.0)
        'readiness_score': 0.0  # Readiness for node conversion (0.0-1.0)
    },
    
    # === Brain State Context ===
    'brain_state_context': {
        'brain_state_during_creation': None,
        'cognitive_state_during_creation': None,
        'processing_intensity': 0.0,
        'attention_level': 0.0,
        'emotional_state': None
    },
    
    # === Relationships ===
    'relationships': {
        'related_fragments': [],  # IDs of related fragments
        'temporal_relationships': [],  # Time-based relationships
        'causal_relationships': [],  # Cause-effect relationships
        'conceptual_relationships': [],  # Concept-based relationships
        'similarity_scores': {}  # Similarity scores with other fragments
    },
    
    # === Validation Tracking ===
    'validation_tracking': {
        'validation_attempts': 0,
        'validation_errors': [],
        'validation_warnings': [],
        'validation_notes': [],
        'neural_network_feedback': []
    },
    
    # === Storage and Retrieval ===
    'storage_info': {
        'storage_location': None,  # Physical storage location/path
        'grid_coordinates': None,  # Temporary grid coordinates if assigned
        'access_count': 0,
        'last_accessed': None,
        'retrieval_contexts': []  # Contexts in which fragment was retrieved
    },
    
    # === Metadata ===
    'metadata': {
        'creation_context': {},  # Context when fragment was created
        'processing_history': [],  # History of processing operations
        'version': 1.0,  # Fragment version for updates
        'tags': [],  # Additional tags for categorization
        'notes': []  # Any additional notes about the fragment
    }
}
