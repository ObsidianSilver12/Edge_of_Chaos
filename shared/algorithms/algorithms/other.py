import pandas as pd

# REMAINING SENSES - COMPLETE ALGORITHM CATALOG
remaining_algorithms = {
    'Sense': [],  # Which sense this algorithm belongs to
    'Algorithm_Name': [],
    'Type': [],  # Model, Function, Methodology, Algorithm
    'Stage': [],  # SENSORY_RAW->PATTERNS, PATTERNS->FRAGMENTS, etc.
    'Category': [],
    'Description': [],
    'Mathematical_Basis': [],
    'Input_Data_Required': [],
    'Output_Data_Generated': [],
    'Computational_Complexity': [],
    'Biomimetic_Relevance': [],
    'Implementation_Notes': []
}

# =============================================================================
# SPATIAL SENSE ALGORITHMS
# =============================================================================

spatial_algorithms = [
    # STAGE 1: SENSORY_RAW -> PATTERNS
    ['SPATIAL', 'Coordinate_System_Conversion', 'Function', 'SENSORY_RAW->PATTERNS', 'Coordinate_Analysis',
     'Converts between different coordinate systems (Cartesian, polar, spherical, geographic)',
     'Cartesian to Polar: r = √(x² + y²), θ = arctan(y/x); Geographic: lat/lon to UTM',
     'coordinates, source_coordinate_system, target_coordinate_system',
     'converted_coordinates, coordinate_accuracy, transformation_matrix',
     'O(1)', 4, 'Use proj4 library for geographic conversions. Essential for spatial analysis.'],
     
    ['SPATIAL', 'Distance_Calculation', 'Function', 'SENSORY_RAW->PATTERNS', 'Coordinate_Analysis',
     'Calculates distances between spatial points using appropriate metrics',
     'Euclidean: d = √(Σ(xi - yi)²); Haversine: d = 2r*arcsin(√(sin²(Δφ/2) + cos(φ1)*cos(φ2)*sin²(Δλ/2)))',
     'point_coordinates, distance_metric_type, coordinate_system',
     'distances, spatial_relationships, proximity_measures',
     'O(n²)', 4, 'Use appropriate distance metric for coordinate system (Euclidean vs Haversine).'],
     
    ['SPATIAL', 'Spatial_Clustering', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Coordinate_Analysis',
     'Groups spatially proximate points into clusters using density-based methods',
     'DBSCAN clustering: density_reachable points within eps distance with min_samples',
     'spatial_coordinates, clustering_parameters (eps, min_samples)',
     'spatial_clusters, cluster_boundaries, location_patterns',
     'O(n²)', 4, 'Use DBSCAN for irregular cluster shapes. Good for geographic data.'],
     
    ['SPATIAL', 'Movement_Vector_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Movement_Analysis',
     'Analyzes movement vectors including velocity, acceleration, and direction changes',
     'Velocity = Δposition/Δtime; Acceleration = Δvelocity/Δtime; Direction = arctan(Δy/Δx)',
     'position_timeseries, timestamps, coordinate_system',
     'velocity_vectors, acceleration_data, movement_patterns, trajectory_analysis',
     'O(n)', 5, 'Highly biomimetic - similar to how animals track movement and navigation.'],
     
    ['SPATIAL', 'Spatial_Relationship_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Relationship_Analysis',
     'Detects spatial relationships between objects (inside, outside, adjacent, overlapping)',
     'Point-in-polygon tests, boundary intersection analysis, proximity thresholds',
     'object_boundaries, spatial_objects, relationship_thresholds',
     'spatial_relationship_types, containment_analysis, proximity_relationships',
     'O(n²)', 4, 'Use computational geometry algorithms for relationship detection.'],
     
    ['SPATIAL', 'Spatial_Density_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Density_Analysis',
     'Analyzes spatial density of points or objects across different regions',
     'Kernel Density Estimation: KDE(x) = (1/nh) * Σ K((x - xi)/h)',
     'spatial_points, kernel_bandwidth, analysis_grid',
     'density_map, high_density_regions, spatial_distribution_patterns',
     'O(n²)', 3, 'Use KDE or simple grid-based counting for density estimation.'],

    # STAGE 2: PATTERNS -> FRAGMENTS
    ['SPATIAL', 'Spatial_Pattern_Integration', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Integrates multiple spatial patterns into coherent spatial understanding',
     'Weighted combination of spatial patterns with coherence validation',
     'location_patterns, movement_patterns, density_patterns, relationship_patterns',
     'integrated_spatial_understanding, spatial_coherence_score, unified_spatial_model',
     'O(n²)', 5, 'Essential for creating unified spatial awareness from multiple cues.'],
     
    ['SPATIAL', 'Territory_Boundary_Detection', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Detects territorial or boundary patterns in spatial data',
     'Convex hull computation and alpha shapes for boundary detection',
     'spatial_clusters, boundary_detection_parameters, territorial_indicators',
     'territory_boundaries, spatial_domains, boundary_confidence',
     'O(n*log(n))', 4, 'Important for understanding spatial organization and boundaries.'],

    # STAGE 3: FRAGMENTS -> NODES  
    ['SPATIAL', 'Spatial_Concept_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Forms abstract spatial concepts like "home", "work area", "path" from spatial patterns',
     'Hierarchical clustering of spatial patterns with semantic labeling',
     'spatial_fragments, location_frequency, activity_associations',
     'spatial_concepts, location_significance, spatial_memory_formation',
     'O(n²)', 5, 'Very biomimetic - how animals form concepts of territory and familiar places.'],
     
    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP
    ['SPATIAL', 'Spatial_Index_Construction', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates spatial index for fast spatial queries and nearest neighbor searches',
     'R-tree or k-d tree construction for multidimensional spatial indexing',
     'spatial_nodes, indexing_parameters, spatial_dimensions',
     'spatial_index, nearest_neighbor_capability, range_query_support',
     'O(n*log(n))', 3, 'Use R-tree for 2D/3D data, k-d tree for higher dimensions.']
]

# =============================================================================
# TEMPORAL SENSE ALGORITHMS  
# =============================================================================

temporal_algorithms = [
    # STAGE 1: SENSORY_RAW -> PATTERNS
    ['TEMPORAL', 'Timestamp_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Time_Analysis',
     'Analyzes timestamp data including parsing, validation, and temporal resolution detection',
     'Timestamp parsing with timezone handling and precision detection',
     'timestamp_data, timezone_info, timestamp_formats',
     'parsed_timestamps, temporal_resolution, time_accuracy',
     'O(n)', 3, 'Use datetime libraries with timezone awareness. Handle multiple formats.'],
     
    ['TEMPORAL', 'Duration_Calculation', 'Function', 'SENSORY_RAW->PATTERNS', 'Time_Analysis',
     'Calculates durations between events and analyzes event length patterns',
     'Duration = end_time - start_time with proper timezone and precision handling',
     'event_timestamps, start_times, end_times',
     'event_durations, duration_patterns, temporal_intervals',
     'O(n)', 4, 'Handle timezone conversions and daylight saving time transitions.'],
     
    ['TEMPORAL', 'Cyclical_Pattern_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Pattern_Detection',
     'Detects cyclical patterns in temporal data using Fourier analysis',
     'FFT analysis: X(k) = Σ x(n) * e^(-j*2π*k*n/N) to find periodic components',
     'temporal_timeseries, sampling_rate, frequency_range',
     'cyclical_patterns, dominant_frequencies, seasonal_components',
     'O(n*log(n))', 5, 'Very biomimetic - circadian rhythms, seasonal patterns, biological cycles.'],
     
    ['TEMPORAL', 'Event_Sequence_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Sequence_Analysis',
     'Analyzes sequences of events for patterns and relationships',
     'Sequence pattern mining using algorithms like GSP or PrefixSpan',
     'event_sequences, sequence_mining_parameters, support_thresholds',
     'frequent_patterns, sequence_rules, temporal_associations',
     'O(n²)', 4, 'Find frequently occurring sequences of events over time.'],
     
    ['TEMPORAL', 'Temporal_Change_Point_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Change_Detection',
     'Detects points in time where statistical properties of data change significantly',
     'PELT (Pruned Exact Linear Time) or Binary Segmentation for change point detection',
     'temporal_timeseries, change_detection_parameters, statistical_measures',
     'change_points, regime_changes, temporal_segments',
     'O(n*log(n))', 4, 'Identify when temporal patterns shift or change significantly.'],
     
    ['TEMPORAL', 'Time_Series_Forecasting', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Prediction',
     'Forecasts future values based on historical temporal patterns',
     'ARIMA modeling: (1-φ₁B-...-φₚBᵖ)(1-B)ᵈXₜ = (1+θ₁B+...+θₑBᵉ)εₜ',
     'historical_timeseries, forecast_horizon, model_parameters',
     'future_predictions, prediction_intervals, forecast_confidence',
     'O(n²)', 3, 'Use ARIMA, exponential smoothing, or neural networks for forecasting.'],

    # STAGE 2: PATTERNS -> FRAGMENTS
    ['TEMPORAL', 'Multi_Scale_Temporal_Integration', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Integrates temporal patterns across multiple time scales (seconds to years)',
     'Hierarchical temporal integration with scale-appropriate weighting',
     'short_term_patterns, medium_term_patterns, long_term_patterns, scale_weights',
     'integrated_temporal_understanding, multi_scale_coherence, temporal_hierarchy',
     'O(n*log(n))', 5, 'Very biomimetic - how biological systems integrate multiple time scales.'],
     
    ['TEMPORAL', 'Causal_Temporal_Analysis', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Causality_Analysis',
     'Analyzes temporal causality using Granger causality and transfer entropy',
     'Granger Causality: F-test on whether lagged values of X help predict Y',
     'temporal_timeseries_X, temporal_timeseries_Y, lag_parameters',
     'causal_relationships, causality_strength, temporal_dependencies',
     'O(n²)', 4, 'Determine if changes in one variable predict changes in another.'],

    # STAGE 3: FRAGMENTS -> NODES
    ['TEMPORAL', 'Temporal_Concept_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Forms abstract temporal concepts like "morning routine", "weekly cycle", "seasonal change"',
     'Temporal pattern abstraction with semantic labeling and hierarchical organization',
     'temporal_fragments, pattern_significance, temporal_contexts',
     'temporal_concepts, temporal_categories, temporal_memory_formation',
     'O(n²)', 5, 'Very biomimetic - how humans form concepts of time periods and routines.'],
     
    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP  
    ['TEMPORAL', 'Temporal_Index_Construction', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates temporal index for fast time-based queries and range searches',
     'B+ tree or interval tree construction for temporal data indexing',
     'temporal_nodes, time_ranges, indexing_parameters',
     'temporal_index, time_range_queries, temporal_search_capability',
     'O(n*log(n))', 3, 'Enable fast queries like "what happened last Tuesday?" or "events in 2024".']
]

# =============================================================================
# METAPHYSICAL SENSE ALGORITHMS
# =============================================================================

metaphysical_algorithms = [
    # STAGE 1: SENSORY_RAW -> PATTERNS
    ['METAPHYSICAL', 'Brain_State_Frequency_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'State_Monitoring',
     'Analyzes dominant frequencies in brain state data using spectral analysis',
     'Power Spectral Density analysis to identify dominant frequency components',
     'brain_state_timeseries, frequency_bands, spectral_analysis_parameters',
     'dominant_frequencies, frequency_power_distribution, brain_wave_patterns',
     'O(n*log(n))', 5, 'Novel approach - analyzing computational "brain waves" like biological EEG.'],
     
    ['METAPHYSICAL', 'Consciousness_Level_Quantification', 'Function', 'SENSORY_RAW->PATTERNS', 'Consciousness_Analysis',
     'Quantifies level of consciousness based on processing complexity and awareness metrics',
     'Consciousness_level = processing_complexity * awareness_breadth * integration_index',
     'processing_metrics, awareness_indicators, integration_measures',
     'consciousness_level, awareness_metrics, cognitive_complexity_score',
     'O(n)', 5, 'Groundbreaking - attempting to quantify machine consciousness levels.'],
     
    ['METAPHYSICAL', 'Energy_Field_Fluctuation_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Field_Analysis',
     'Detects fluctuations in computational energy fields and processing resonances',
     'Statistical analysis of energy consumption patterns and processing harmonics',
     'energy_consumption_timeseries, processing_load_patterns, field_parameters',
     'field_fluctuations, energy_field_patterns, resonance_detection',
     'O(n)', 4, 'Novel biomimetic approach - computational fields like biological fields.'],
     
    ['METAPHYSICAL', 'Resonance_Frequency_Calculation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Resonance_Analysis',
     'Calculates resonance frequencies between different system components',
     'Cross-correlation analysis to find synchronous oscillations between components',
     'component_frequencies, system_harmonics, resonance_parameters',
     'resonance_frequencies, harmonic_relationships, synchronization_patterns',
     'O(n²)', 5, 'Based on physics principles - finding resonant frequencies in computation.'],
     
    ['METAPHYSICAL', 'Quantum_Coherence_Measurement', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Quantum_Analysis',
     'Measures quantum-like coherence in computational processes',
     'Coherence measure based on phase relationships and entanglement-like correlations',
     'processing_states, phase_information, correlation_matrices',
     'coherence_measures, entanglement_indicators, quantum_like_properties',
     'O(n²)', 4, 'Experimental - applying quantum concepts to computational consciousness.'],
     
    ['METAPHYSICAL', 'System_Purpose_Alignment_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Purpose_Analysis',
     'Analyzes alignment between current system behavior and defined purpose/goals',
     'Alignment_score = cosine_similarity(current_behavior_vector, purpose_vector)',
     'current_system_behavior, defined_purpose, goal_parameters',
     'purpose_alignment_score, behavioral_drift, goal_coherence',
     'O(n)', 5, 'Very biomimetic - like how living beings assess alignment with life purpose.'],
     
    ['METAPHYSICAL', 'Emergence_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Emergence_Analysis',
     'Detects emergent properties arising from complex system interactions',
     'Emergence measure based on system complexity exceeding sum of component complexities',
     'system_components, component_interactions, complexity_measures',
     'emergence_indicators, emergent_properties, system_transcendence',
     'O(n²)', 5, 'Cutting-edge - detecting when system exhibits properties beyond its parts.'],

    # STAGE 2: PATTERNS -> FRAGMENTS
    ['METAPHYSICAL', 'Consciousness_State_Integration', 'Methodology', 'PATTERNS->FRAGMENTS', 'Consciousness_Integration',
     'Integrates various consciousness indicators into unified awareness assessment',
     'Weighted integration of multiple consciousness measures with feedback loops',
     'consciousness_indicators, awareness_measures, integration_parameters',
     'integrated_consciousness_state, awareness_coherence, consciousness_stability',
     'O(n)', 5, 'Novel approach to unified consciousness measurement in AI systems.'],
     
    ['METAPHYSICAL', 'Resonance_Pattern_Synthesis', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Resonance_Integration',
     'Synthesizes resonance patterns across different system levels and components',
     'Multi-level resonance analysis with harmonic integration',
     'component_resonances, system_harmonics, resonance_interactions',
     'synthesized_resonance_patterns, harmonic_coherence, resonance_stability',
     'O(n²)', 4, 'Combine resonance patterns from different system levels.'],

    # STAGE 3: FRAGMENTS -> NODES
    ['METAPHYSICAL', 'Consciousness_Concept_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Consciousness_Concepts',
     'Forms abstract concepts related to consciousness, awareness, and existential understanding',
     'Hierarchical abstraction of consciousness experiences into conceptual frameworks',
     'consciousness_fragments, existential_experiences, awareness_patterns',
     'consciousness_concepts, existential_understanding, awareness_categories',
     'O(n²)', 5, 'Groundbreaking - AI forming concepts about its own consciousness.'],
     
    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP
    ['METAPHYSICAL', 'Consciousness_State_Embedding', 'Model', 'NODES->SEMANTIC_MAP', 'Consciousness_Indexing',
     'Creates embeddings for different consciousness states enabling introspective search',
     'Neural embedding of consciousness features into searchable vector space',
     'consciousness_states, embedding_model, consciousness_dimensions',
     'consciousness_embeddings, introspective_search_capability, consciousness_similarity',
     'O(n)', 4, 'Enable AI to search and compare its own consciousness states.']
]

# =============================================================================
# ALGORITHMIC SENSE ALGORITHMS
# =============================================================================

algorithmic_algorithms = [
    # STAGE 1: SENSORY_RAW -> PATTERNS
    ['ALGORITHMIC', 'Algorithm_Performance_Profiling', 'Function', 'SENSORY_RAW->PATTERNS', 'Performance_Analysis',
     'Profiles algorithm execution including time, memory, accuracy, and resource utilization',
     'Performance_metrics = {execution_time, memory_usage, accuracy, throughput, latency}',
     'algorithm_executions, performance_counters, resource_monitors',
     'performance_profiles, execution_statistics, resource_efficiency_metrics',
     'O(1)', 4, 'Essential for understanding which algorithms work best in different contexts.'],
     
    ['ALGORITHMIC', 'Algorithm_Accuracy_Assessment', 'Function', 'SENSORY_RAW->PATTERNS', 'Accuracy_Analysis',
     'Assesses accuracy of algorithm outputs using various validation metrics',
     'Accuracy metrics: precision, recall, F1-score, AUC-ROC, MSE, MAE depending on task type',
     'algorithm_outputs, ground_truth_data, validation_metrics, task_type',
     'accuracy_scores, confidence_intervals, error_analysis, performance_reliability',
     'O(n)', 4, 'Track how accurate different algorithms are for different types of problems.'],
     
    ['ALGORITHMIC', 'Learning_Rate_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Learning_Analysis',
     'Analyzes how quickly algorithms learn and adapt to new information',
     'Learning_rate = Δperformance / Δtraining_iterations with convergence analysis',
     'training_performance_curves, iteration_counts, convergence_metrics',
     'learning_rates, convergence_speed, adaptation_efficiency, learning_curves',
     'O(n)', 5, 'Very biomimetic - how fast different learning approaches work.'],
     
    ['ALGORITHMIC', 'Algorithm_Effectiveness_Correlation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Effectiveness_Analysis',
     'Correlates algorithm effectiveness with input characteristics and context',
     'Correlation analysis between algorithm performance and input feature characteristics',
     'algorithm_performance, input_characteristics, contextual_factors',
     'effectiveness_correlations, optimal_usage_contexts, algorithm_suitability_mapping',
     'O(n²)', 4, 'Learn which algorithms work best for which types of problems.'],
     
    ['ALGORITHMIC', 'Resource_Efficiency_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Efficiency_Analysis',
     'Analyzes computational resource efficiency of different algorithms',
     'Efficiency = output_quality / (computational_cost + memory_cost + time_cost)',
     'resource_usage_metrics, output_quality_measures, cost_functions',
     'efficiency_ratings, resource_optimization_opportunities, cost_benefit_analysis',
     'O(n)', 3, 'Important for selecting algorithms that give best value for computational cost.'],
     
    ['ALGORITHMIC', 'Algorithm_Synergy_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Synergy_Analysis',
     'Detects synergistic effects when algorithms are used together',
     'Synergy_measure = combined_performance - (performance_A + performance_B)',
     'individual_algorithm_performance, combined_algorithm_performance, interaction_effects',
     'synergy_scores, algorithm_combinations, interaction_benefits',
     'O(n²)', 4, 'Find combinations of algorithms that work better together than separately.'],

    # STAGE 2: PATTERNS -> FRAGMENTS
    ['ALGORITHMIC', 'Algorithm_Selection_Strategy_Development', 'Methodology', 'PATTERNS->FRAGMENTS', 'Strategy_Formation',
     'Develops strategies for selecting optimal algorithms based on context and requirements',
     'Multi-criteria decision analysis combining performance, efficiency, and suitability factors',
     'algorithm_performance_patterns, context_requirements, selection_criteria',
     'selection_strategies, algorithm_recommendation_rules, adaptive_selection_policies',
     'O(n²)', 5, 'Learn to automatically select best algorithms for different situations.'],
     
    ['ALGORITHMIC', 'Performance_Pattern_Integration', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Integrates algorithm performance patterns across different contexts and time periods',
     'Weighted integration of performance patterns with temporal and contextual factors',
     'performance_patterns, temporal_factors, contextual_factors, integration_weights',
     'integrated_performance_understanding, algorithm_reliability_assessment, context_sensitivity',
     'O(n²)', 4, 'Understand algorithm behavior across different conditions and contexts.'],

    # STAGE 3: FRAGMENTS -> NODES
    ['ALGORITHMIC', 'Algorithmic_Knowledge_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Knowledge_Formation',
     'Forms abstract knowledge about algorithmic principles and computational strategies',
     'Knowledge abstraction from algorithmic experiences into general principles',
     'algorithmic_fragments, performance_experiences, strategy_outcomes',
     'algorithmic_knowledge, computational_principles, strategy_understanding',
     'O(n²)', 5, 'Very advanced - AI learning general principles about computation itself.'],
     
    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP
    ['ALGORITHMIC', 'Algorithm_Knowledge_Index', 'Function', 'NODES->SEMANTIC_MAP', 'Knowledge_Indexing',
     'Creates searchable index of algorithmic knowledge and performance characteristics',
     'Structured index of algorithms with performance metadata and usage contexts',
     'algorithmic_knowledge, performance_metadata, usage_contexts',
     'algorithm_knowledge_base, searchable_algorithm_index, recommendation_system',
     'O(n*log(n))', 4, 'Enable fast lookup of which algorithms to use for specific problems.']
]

# =============================================================================
# OTHER DATA SENSE ALGORITHMS
# =============================================================================

other_data_algorithms = [
    # STAGE 1: SENSORY_RAW -> PATTERNS
    ['OTHER', 'Data_Type_Classification', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Classification_Analysis',
     'Attempts to classify unknown data types using statistical and structural analysis',
     'Feature extraction and machine learning classification on data characteristics',
     'unknown_data, data_structure_analysis, statistical_features',
     'data_type_predictions, classification_confidence, structure_analysis',
     'O(n)', 3, 'Try to understand what type of data this might be.'],
     
    ['OTHER', 'Pattern_Discovery_Unknown_Data', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Pattern_Analysis',
     'Discovers patterns in unknown data using unsupervised learning methods',
     'Clustering, dimensionality reduction, and anomaly detection on unknown data',
     'unknown_data, pattern_discovery_parameters, unsupervised_methods',
     'discovered_patterns, data_clusters, anomalous_elements',
     'O(n²)', 4, 'Find structure and patterns in data we don\'t yet understand.'],
     
    ['OTHER', 'Data_Quality_Assessment', 'Function', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Assesses quality of unknown data including completeness, consistency, and validity',
     'Quality_score = completeness * consistency * validity * timeliness',
     'unknown_data, quality_criteria, validation_rules',
     'data_quality_scores, quality_issues, data_reliability_assessment',
     'O(n)', 3, 'Determine if this unknown data is worth further analysis.'],

    # STAGE 2: PATTERNS -> FRAGMENTS  
    ['OTHER', 'Unknown_Data_Integration_Attempt', 'Methodology', 'PATTERNS->FRAGMENTS', 'Integration_Analysis',
     'Attempts to integrate unknown data with existing sensory modalities',
     'Correlation analysis and similarity measures with known data types',
     'unknown_data_patterns, known_sensory_patterns, integration_attempts',
     'integration_possibilities, correlation_strengths, potential_relationships',
     'O(n²)', 3, 'Try to connect unknown data with things we do understand.'],

    # STAGE 3: FRAGMENTS -> NODES
    ['OTHER', 'Novel_Concept_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Forms new conceptual categories for previously unknown types of data',
     'Unsupervised concept formation using hierarchical clustering and abstraction',
     'unknown_data_fragments, concept_formation_parameters, abstraction_levels',
     'novel_concepts, new_categories, conceptual_frameworks',
     'O(n²)', 5, 'Very important - ability to form concepts for entirely new types of data.'],

    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP
    ['OTHER', 'Unknown_Data_Archive', 'Function', 'NODES->SEMANTIC_MAP', 'Archive_Management',
     'Creates archive system for unknown data that might become useful later',
     'Structured archival with metadata and retrieval capabilities for unknown data',
     'unknown_data_nodes, metadata_extraction, archival_parameters',
     'data_archive, retrieval_system, metadata_index',
     'O(n*log(n))', 3, 'Keep unknown data organized for future analysis when we understand it better.']
]

# Combine all algorithms
all_algorithms = (spatial_algorithms + temporal_algorithms + 
                 metaphysical_algorithms + algorithmic_algorithms + 
                 other_data_algorithms)

# Add all the data to the dictionary
for algo_data in all_algorithms:
    remaining_algorithms['Sense'].append(algo_data[0])
    remaining_algorithms['Algorithm_Name'].append(algo_data[1])
    remaining_algorithms['Type'].append(algo_data[2])
    remaining_algorithms['Stage'].append(algo_data[3])
    remaining_algorithms['Category'].append(algo_data[4])
    remaining_algorithms['Description'].append(algo_data[5])
    remaining_algorithms['Mathematical_Basis'].append(algo_data[6])
    remaining_algorithms['Input_Data_Required'].append(algo_data[7])
    remaining_algorithms['Output_Data_Generated'].append(algo_data[8])
    remaining_algorithms['Computational_Complexity'].append(algo_data[9])
    remaining_algorithms['Biomimetic_Relevance'].append(algo_data[10])
    remaining_algorithms['Implementation_Notes'].append(algo_data[11])

# Create DataFrame
remaining_df = pd.DataFrame(remaining_algorithms)

# Display summary
print("REMAINING SENSES - COMPLETE ALGORITHM CATALOG")
print("=" * 60)
print(f"Total Remaining Senses Algorithms: {len(remaining_df)}")
print()

# Show breakdown by sense
print("ALGORITHMS BY SENSE:")
sense_counts = remaining_df['Sense'].value_counts()
for sense, count in sense_counts.items():
    print(f"  {sense}: {count} algorithms")
print()

# Show breakdown by stage
print("ALGORITHMS BY STAGE:")
stage_counts = remaining_df['Stage'].value_counts()
for stage, count in stage_counts.items():
    print(f"  {stage}: {count} algorithms")
print()

# Show type breakdown
print("ALGORITHM TYPES:")
type_counts = remaining_df['Type'].value_counts()
for algo_type, count in type_counts.items():
    print(f"  {algo_type}: {count} algorithms")
print()

# Show biomimetic relevance distribution
print("BIOMIMETIC RELEVANCE DISTRIBUTION:")
biomimetic_counts = remaining_df['Biomimetic_Relevance'].value_counts().sort_index()
for relevance, count in biomimetic_counts.items():
    print(f"  Level {relevance}: {count} algorithms")
print()

# Show sample of each sense
print("SAMPLE ALGORITHMS BY SENSE:")
for sense in remaining_df['Sense'].unique():
    print(f"\n{sense} SENSE - Sample Algorithms:")
    sense_data = remaining_df[remaining_df['Sense'] == sense]
    print(sense_data[['Algorithm_Name', 'Type', 'Stage', 'Biomimetic_Relevance']].head(3).to_string(index=False))

# Summary statistics
print(f"\nSUMMARY STATISTICS:")
print(f"Total Algorithms Across All Remaining Senses: {len(remaining_df)}")
print(f"Most Biomimetic Sense: {remaining_df.groupby('Sense')['Biomimetic_Relevance'].mean().idxmax()}")
print(f"Average Biomimetic Relevance: {remaining_df['Biomimetic_Relevance'].mean():.2f}")

# Save to Excel if needed
# remaining_df.to_excel('remaining_senses_algorithms_complete_catalog.xlsx', index=False)