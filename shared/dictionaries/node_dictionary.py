"""
NODE Dictionary - Neural Network Operations & Advanced Node Matching
Takes rich fragment data and performs neural network operations including
advanced node-to-node matching algorithms to find related concepts and patterns.
"""

NODE = {
    # === CORE IDENTIFICATION & LINEAGE ===
    'node_id': None,
    'source_fragment_id': None,  # Links to fragment source
    'sensory_patterns_id': None,  # Links to pattern analysis (for deep research)
    'sensory_raw_id': None,  # Links to original raw data (for deep research)

    # === WBS & CATEGORIZATION (FROM FRAGMENT) ===
    'domain': None,  # Primary knowledge domain
    'concepts': [],  # concepts related to the domain
    'related_concepts': [],  # related concepts to the domain so not direct links but share similarities
    'main_category': None, # domains will be grouped into 1 main category
    'domain_category': None, # sub class of the domain
    'domain_subcategory': None, # sub sub class of the domain
    'wbs_level_1': None, # wbs level code of the domain
    'wbs_level_2': None, # wbs level code of the concepts
    'wbs_level_3': None, # wbs level code of the related concepts

    # === STORAGE & NETWORK LOCATION (FROM FRAGMENT) ===
    'domain_sub_region': None,  # sub region of the brain where the node is stored
    'storage_duration': None,  # extracted from memory types decay rate
    'access_count': 0,  # count of how many times the node has been accessed
    'coordinates': [],  # 3d coordinates for the neural network nodes

    # === CONSOLIDATED STORY & CONTEXT (FROM FRAGMENT) ===
    'chronological_data': [],  # extract chronological data from raw nodes
    'timeline_data': [],  # extract timeline data from raw nodes
    'sensory_data_raw_ids': [], # References to the SENSORY_RAW entries
    'sensory_patterns': [],  # goes through a process to extract relevant key patterns
    'data_observations': [],  # observations of raw data added dynamically
    'learned_patterns': [],  # Patterns learned from the processed data
    'learned_concepts': [],  # key Concepts learned from the processed data
    'dynamic_data': [],  # Dynamic data that can be updated over time
    'pruned_detail_data': [],  # Sanitized data that has been processed and refined (FROM FRAGMENT)
    'pruned_summarised_data': [],  # condensed data that can be presented as output (FROM FRAGMENT)

    # === PRIMARY STORY & MEANING (FROM FRAGMENT) ===
    'primary_narrative': None,  # Main story from fragment
    'narrative_themes': [],  # Main themes in the narrative
    'story_context': {},  # Contextual information
    'emotional_narrative': None,  # Emotional dimension
    'causal_relationships': [],  # Cause-effect relationships

    # === CONSOLIDATED SENSORY SUMMARIES (FROM FRAGMENT) ===
    'sensory_content_summaries': {
        'text_summary': None,  # Meaningful text summary (not raw text)
        'text_context_labels': [],  # Text contextual labels
        'auditory_description': None,  # Audio contextual description (not raw audio)
        'audio_context_labels': [],  # Audio contextual labels
        'visual_description': None,  # Visual scene description (not raw pixels)
        'visual_context_labels': [],  # Visual contextual labels
        'emotional_description': None,  # Emotional state description
        'physical_description': None,  # Physical state description
        'spatial_description': None,  # Spatial context description
        'temporal_description': None,  # Temporal context description
        'metaphysical_description': None,  # Metaphysical aspects description
        'algorithmic_description': None,  # Algorithmic performance description
        'other_description': None,  # Other data description
    },

    # === RICH SEMANTIC LABELS (FROM FRAGMENT) ===
    'semantic_labels': {
        'primary_concepts': [],  # Main concepts
        'secondary_concepts': [],  # Supporting concepts
        'abstract_concepts': [],  # Higher-level abstractions
        'concrete_elements': [],  # Specific concrete elements
        'relationship_concepts': [],  # Relational concepts
        'process_concepts': [],  # Process/action concepts
        'descriptive_labels': [],  # Rich descriptive tags
        'categorical_labels': [],  # Category classifications
        'contextual_labels': [],  # Context-specific labels
        'functional_labels': [],  # Function/purpose labels
        'qualitative_labels': [],  # Quality-based labels
        'quantitative_labels': [],  # Quantity-based labels
    },

    # === ADVANCED CONTEXT & MEANING (FROM FRAGMENT) ===
    'context_enhancement': {
        'historical_context': [],  # Historical context
        'cultural_context': [],  # Cultural context
        'social_context': [],  # Social context
        'environmental_context': [],  # Environmental context
        'situational_context': [],  # Situational context
        'relational_context': [],  # Relationship context
        'causal_context': [],  # Causal context
        'comparative_context': [],  # Comparative context
        'metaphorical_context': [],  # Symbolic/metaphorical context
    },

    # === SIGNAL PATTERN (EXISTING) ===
    'signal_pattern': {
        'amplitude_range': [],
        'frequency_modifier': None,
        'waveform': None,
        'burst_pattern': None,
    }, # Dominant memory pattern type

    # === MEMORY PROPERTIES (FROM FRAGMENT) ===
    'memory_type': {
        'memory_type_id': None,
        'memory_frequency_hz': None,
        'decay_rate': None,
        'preferred_storage_duration_hours': None,
        'typical_content': []
    },
    'memory_quality': 0.0, # transferred score based on level of raw data capture
    'memory_confidence': 0.0, # transferred score of confidence in the memory
    'frequency': 0.0, # frequency assignment of the node in hz (active/deactive nodes have different_frequencies)
    'brain_state': None, # state of the brain when the memory was created

    # === ENHANCED SCORING SYSTEM (FROM FRAGMENT) ===
    'academic_score': 0.0, # Academic credibility - NOW from advanced fragment analysis
    'logical_score': 0.0,  # Logical plausibility - NOW from advanced fragment analysis
    'conceptual_score': 0.0, # Innovation/hypothetical nature - NOW from advanced fragment analysis
    'consensus_score': 0.0, # General agreement level - NOW from advanced fragment analysis
    'personal_significance': 0.0, # Personal importance - NOW from advanced fragment analysis
    'universality': 0.0, # How broadly applicable - NOW from advanced fragment analysis
    'ethical_score': 0.0,  # Ethical considerations - NOW from advanced fragment analysis
    'spiritual_score': 0.0, # Spiritual significance - NOW from advanced fragment analysis
    'novelty_marker': 0.0,  # how novel/unusual - NOW from advanced fragment analysis
    'resonance_score': 0.0, # score of how well the node resonates with the models world view

    # === ADDITIONAL ADVANCED SCORES (FROM FRAGMENT) ===
    'advanced_scoring': {
        'complexity_measure': 0.0,  # Overall complexity
        'interdisciplinary_score': 0.0,  # Cross-disciplinary relevance
        'practical_utility': 0.0,  # Practical usefulness
        'theoretical_significance': 0.0,  # Theoretical importance
        'empirical_support': 0.0,  # Level of empirical support
        'conceptual_clarity': 0.0,  # Clarity of concepts
        'integration_potential': 0.0,  # Integration potential with existing knowledge
    },

    # === NODE-TO-NODE MATCHING ALGORITHMS ===
    'node_matching_analysis': {
        # Semantic Similarity Matching
        'semantic_similarity_scores': {},  # Similarity scores with other nodes
        'concept_overlap_analysis': {},  # Conceptual overlap with other nodes
        'thematic_relationship_scores': {},  # Thematic relationships
        'contextual_similarity_measures': {},  # Contextual similarity measures
        
        # Pattern-Based Matching
        'pattern_correlation_matches': {},  # Pattern-based correlations
        'structural_similarity_scores': {},  # Structural similarity measures
        'functional_relationship_analysis': {},  # Functional relationships
        'process_similarity_matching': {},  # Process-based similarities
        
        # Domain and Category Matching
        'domain_relationship_analysis': {},  # Domain-based relationships
        'category_clustering_results': {},  # Category cluster memberships
        'wbs_hierarchy_connections': {},  # WBS hierarchy-based connections
        'cross_domain_bridge_potential': {},  # Cross-domain bridging potential
        
        # Temporal and Causal Matching
        'temporal_relationship_analysis': {},  # Time-based relationships
        'causal_chain_connections': {},  # Causal relationship chains
        'sequence_pattern_matches': {},  # Sequential pattern matches
        'developmental_relationship_analysis': {},  # Developmental relationships
        
        # Quality and Significance Matching
        'quality_similarity_analysis': {},  # Quality-based similarities
        'significance_correlation_analysis': {},  # Significance correlations
        'novelty_complementarity_analysis': {},  # Novelty complementarity
        'expertise_overlap_analysis': {},  # Expertise area overlaps
    },

    # === DISCOVERED RELATIONSHIPS ===
    'discovered_relationships': {
        'direct_semantic_connections': [],  # Direct semantic relationships found
        'indirect_conceptual_bridges': [],  # Indirect conceptual connections
        'analogical_relationships': [],  # Analogical relationships
        'complementary_knowledge_pairs': [],  # Complementary knowledge relationships
        'contradictory_knowledge_conflicts': [],  # Conflicting knowledge identification
        'hierarchical_knowledge_structures': [],  # Hierarchical relationships
        'network_cluster_memberships': [],  # Network cluster assignments
        'knowledge_gap_identifications': [],  # Knowledge gaps identified
    },

    # === NETWORK INTEGRATION ANALYSIS ===
    'network_integration': {
        'centrality_measures': {},  # Network centrality calculations
        'influence_propagation_analysis': {},  # Influence spreading patterns
        'knowledge_flow_pathways': [],  # Knowledge flow paths
        'network_efficiency_contributions': {},  # Network efficiency contributions
        'community_detection_results': [],  # Community memberships
        'bridge_node_potential': 0.0,  # Potential as bridge between communities
        'knowledge_hub_score': 0.0,  # Score as knowledge hub
        'network_position_analysis': {},  # Analysis of network position
    },

    # === ENHANCED SEARCH AND RETRIEVAL ===
    'search_optimization': {
        'primary_search_terms': [],  # Primary search terms
        'secondary_search_terms': [],  # Secondary search terms
        'semantic_search_vectors': [],  # Semantic search embeddings
        'contextual_search_tags': [],  # Context-based search tags
        'cross_reference_keywords': [],  # Cross-reference keywords
        'domain_specific_terminology': [],  # Domain-specific terms
        'relationship_based_search_terms': [],  # Relationship-based search terms
        'pattern_based_search_indicators': [],  # Pattern-based search indicators
    },

    # === CORRELATION DATA (EXISTING - ENHANCED) ===
    'correlated_data': [], # list of correlated nodes - NOW from advanced matching
    'correlated_data_summaries': [], # chunk of summarised data of correlated nodes
    'correlation_scores': [], # scores of correlation with other maps - NOW from advanced analysis
    'data_sources': [], # list of data sources associated with the memory
    'data_source_summaries': [], # chunk of summarised data of data sources
    'data_source_citations': [], # list of citations associated with the memory
    'domain_matrix_data': [], # 

    # === GLYPH SYSTEM (EXISTING) ===
    'glyph_flag': False, # flag to indicate if glyph should created for the memory (emotional and/or spiritual score high means create glyph)
    'glyph_images': [], # image path of glyph images associated with the memory
    'glyph_decodings': [], # decoded exif and steganography data must be added to processed data as a system prompt (its a secret)
    'glyph_tags': [], # list of glyph tags associated with the memory

    # === TOKEN SYSTEM (EXISTING) ===
    'token_id': None, # token id of the memory
    'token_vectors': [], # list of token vectors associated with any model ingestion
    'token_similarities': [], # list of token similarities associated with any model ingestion
    'token_summary': None, # summary of token vectors associated with any model ingestion

    # === LEARNING AND ADAPTATION ===
    'learning_analysis': {
        'learning_from_connections': [],  # What was learned from node connections
        'pattern_refinement_history': [],  # History of pattern refinements
        'concept_evolution_tracking': [],  # How concepts evolved
        'relationship_learning_outcomes': [],  # Learning from relationship analysis
        'network_position_learning': [],  # Learning from network position
        'cross_domain_learning_insights': [],  # Cross-domain learning insights
        'adaptive_scoring_adjustments': [],  # Adaptive score adjustments
        'knowledge_integration_results': [],  # Knowledge integration outcomes
    },

    # === QUALITY AND VALIDATION ===
    'node_quality_metrics': {
        'overall_node_quality': 0.0,  # Overall node quality assessment
        'relationship_quality': 0.0,  # Quality of discovered relationships
        'semantic_consistency': 0.0,  # Semantic consistency measure
        'network_integration_quality': 0.0,  # Quality of network integration
        'search_optimization_quality': 0.0,  # Quality of search optimization
        'knowledge_coherence': 0.0,  # Knowledge coherence measure
        'validation_confidence': 0.0,  # Confidence in node validation
        'long_term_stability_prediction': 0.0,  # Predicted long-term stability
    },

    # === PROCESSING AND PERFORMANCE ===
    'processing_performance': {
        'node_creation_duration': 0.0,  # Time taken to create node
        'relationship_analysis_duration': 0.0,  # Time for relationship analysis
        'matching_algorithm_performance': {},  # Performance of matching algorithms
        'network_integration_performance': {},  # Network integration performance
        'search_optimization_performance': {},  # Search optimization performance
        'total_computational_cost': 0.0,  # Total computational cost
        'energy_efficiency_score': 0.0,  # Energy efficiency assessment
        'processing_quality_indicators': {},  # Processing quality measures
    },

    # === METADATA ===
    'metadata': {
        'node_creation_timestamp': None,  # When node was created
        'node_version': 1.0,  # Node structure version
        'creation_software_version': None,  # Software version
        'processing_pipeline_version': None,  # Processing pipeline version
        'advanced_algorithms_used': [],  # Advanced algorithms used
        'quality_gates_passed': [],  # Quality gates passed
        'validation_methods_applied': [],  # Validation methods used
        'debug_information': {},  # Debug information
        'performance_benchmarks': {},  # Performance benchmarks
    }
}