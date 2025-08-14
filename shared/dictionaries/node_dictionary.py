NODE = {
    # === Core Identification & Lineage ===
    'node_id': None,
    # ARCHITECTURAL IMPROVEMENT: Explicitly links this node to the
    # fragment that was its source material.
    'source_fragment_id': None,

    # === WBS & Categorization ===
    'domain': None,
    'concepts': [],  # concepts related to the domain
    'related_concepts': [],  # related concepts to the domain so not direct links but share similarities
    'main_category': None, # domains will be grouped into 1 main category
    'domain_category': None, # sub class of the domain
    'domain_subcategory': None, # sub sub class of the domain
    'wbs_level_1': None, # wbs level code of the domain
    'wbs_level_2': None, # wbs level code of the concepts
    'wbs_level_3': None, # wbs level code of the related concepts

    # === Storage & Network Location ===
    'domain_sub_region': None,  # sub region of the brain where the node is stored
    'storage_duration': None,  # extracted from memory types decay rate
    'access_count': 0,  # count of how many times the node has been accessed
    'coordinates': [],  # 3d coordinates for the neural network nodes

    # === Extracted & Learned Content ===
    'chronological_data': [],  # extract chronological data from raw nodes
    'timeline_data': [],  # extract timeline data from raw nodes
    # ARCHITECTURAL IMPROVEMENT: Holds IDs to the original sensory data, not copies.
    'sensory_data_raw_ids': [], # References to the SENSORY_RAW entries
    'sensory_patterns': [],  # goes through a process to extract relevant key patterns
    'data_observations': [],  # observations of raw data added dynamically
    'learned_patterns': [],  # Patterns learned from the processed data
    'learned_concepts': [],  # key Concepts learned from the processed data
    'dynamic_data': [],  # Dynamic data that can be updated over time
    'pruned_detail_data': [],  # Sanitized data that has been processed and refined
    'pruned_summarised_data': [],  # condensed data that can be presented as output
    'signal_pattern': {
        'amplitude_range': [],
        'frequency_modifier': None,
        'waveform': None,
        'burst_pattern': None,
    }, # Dominant memory pattern type

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
    'correlated_data': [], # list of correlated nodes
    'correlated_data_summaries': [], # chunk of summarised data of correlated nodes
    'correlation_scores': [], # scores of correlation with other maps
    'data_sources': [], # list of data sources associated with the memory
    'data_source_summaries': [], # chunk of summarised data of data sources
    'data_source_citations': [], # list of citations associated with the memory
    'domain_matrix_data': [], # 
    'glyph_flag': False, # flag to indicate if glyph should created for the memory (emotional and/or spiritual score high means create glyph)
    'glyph_images': [], # image path of glyph images associated with the memory
    'glyph_decodings': [], # decoded exif and steganography data must be added to processed data as a system prompt (its a secret)
    'glyph_tags': [], # list of glyph tags associated with the memory
    'token_id': None, # token id of the memory
    'token_vectors': [], # list of token vectors associated with any model ingestion
    'token_similarities': [], # list of token similarities associated with any model ingestion
    'token_summary': None, # summary of token vectors associated with any model ingestion
    'academic_score': 0.0, # Academic credibility (0-1) to be updated dynamically once brain processed raw node
    'logical_score': 0.0,  # Logical plausibility (0-1) to be updated dynamically once brain processed raw node
    'conceptual_score': 0.0, # Innovation/hypothetical nature (0-1) to be updated dynamically once brain processed raw node
    'consensus_score': 0.0, # General agreement level (0-1) to be updated dynamically once brain processed raw node
    'personal_significance': 0.0, # Personal importance (0-1) to be updated dynamically once brain processed raw node
    'universality': 0.0, # How broadly applicable (0-1) to be updated dynamically once brain processed raw node
    'ethical_score': 0.0,  # Ethical considerations (0-1) to be updated dynamically once brain processed raw node
    'spiritual_score': 0.0, # Spiritual significance (0-1) to be updated dynamically once brain processed raw node
    'novelty_marker': 0.0,  # how novel/unusual (0.0-1.0)  
    'resonance_score': 0.0, # score of how well the node resonates with the models world view. may need refinement of what scores contribute to this score
    'harmonic_score': 0.0, # score of energy dynamics between connections
    'emotional_charge': 0.0,  # transferred score of basic emotional charge (-1.0 to 1.0)
    'emotional_descriptor': None,  # Emotional description
    'valence': 0.0,  # Positivity/negativity balance
    'energy_signature': 0.0,  # Energy signal as a colour
    'impact_strength': 0.0,  # Strength of emotional impact
    'training_confidence': 0.0, # Confidence in the information learned based on training data (0-1)
    'confidence_score': 0.0, # Confidence in the information overall there is evidence that hypothesis is plausible, training data good, patterns have high confidence etc (0-1)
}