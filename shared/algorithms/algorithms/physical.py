import pandas as pd

# PHYSICAL STATE SENSE - COMPLETE ALGORITHM CATALOG
physical_algorithms = {
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
# STAGE 1: SENSORY_RAW -> PATTERNS
# =============================================================================

algorithms_data = [
    # SYSTEM METRICS COLLECTION
    ['CPU_Usage_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors CPU utilization percentage across all cores and individual core usage',
     'CPU_usage = (cpu_time_busy / total_cpu_time) * 100',
     'system_process_info, sampling_interval, core_count',
     'cpu_usage (0.0-100.0), per_core_usage, cpu_load_average',
     'O(1)', 4, 'Use psutil.cpu_percent(). Sample over time intervals for accuracy.'],
     
    ['Memory_Usage_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors RAM usage including available, used, cached, and buffered memory',
     'Memory_usage = (used_memory / total_memory) * 100',
     'system_memory_info, virtual_memory_stats',
     'memory_usage (0.0-100.0), available_memory, memory_pressure_indicator',
     'O(1)', 4, 'Use psutil.virtual_memory(). Track both physical and virtual memory.'],
     
    ['Disk_Usage_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors disk space utilization across all mounted filesystems',
     'Disk_usage = (used_space / total_space) * 100',
     'filesystem_info, mounted_drives, disk_partitions',
     'disk_usage (0.0-100.0), free_space, disk_io_statistics',
     'O(n)', 3, 'Use psutil.disk_usage(). Monitor all mounted drives separately.'],
     
    ['Network_Activity_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors network I/O including bytes sent/received, packets, and connection counts',
     'Network_activity = bytes_sent + bytes_received per unit time',
     'network_interface_info, connection_stats, bandwidth_limits',
     'network_activity (bytes/sec), connection_count, bandwidth_utilization',
     'O(1)', 3, 'Use psutil.net_io_counters(). Track per-interface statistics.'],
     
    ['Temperature_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors hardware temperature sensors for CPU, GPU, and other components',
     'Temperature readings from hardware sensors (direct hardware access)',
     'hardware_sensors, sensor_locations, temperature_thresholds',
     'temperature_readings (°C), thermal_zones, overheating_alerts',
     'O(1)', 5, 'Use psutil.sensors_temperatures(). Very biomimetic - like body temperature.'],
     
    ['Power_Consumption_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors system power consumption and battery status if available',
     'Power_consumption = voltage * current (from hardware sensors)',
     'power_sensors, battery_info, power_management_settings',
     'power_consumption (watts), battery_level, power_efficiency',
     'O(1)', 4, 'Use psutil.sensors_battery(). Important for energy awareness.'],
     
    ['Process_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors running processes including PID, CPU usage, memory usage, and status',
     'Process_stats aggregation across all running processes',
     'process_list, process_details, system_process_table',
     'process_count, top_processes, process_resource_usage',
     'O(n)', 3, 'Use psutil.process_iter(). Track process lifecycle and resource usage.'],
     
    ['Fan_Speed_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'System_Monitoring',
     'Monitors cooling system fan speeds and thermal management',
     'Fan_speed readings from hardware sensors (RPM measurements)',
     'fan_sensors, cooling_zones, thermal_management_settings',
     'fan_speeds (RPM), cooling_efficiency, thermal_management_status',
     'O(1)', 4, 'Use psutil.sensors_fans(). Indicates system thermal stress.'],

    # PERFORMANCE ANALYSIS ALGORITHMS
    ['CPU_Load_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Performance_Analysis',
     'Analyzes CPU load patterns over time to identify performance bottlenecks',
     'Load_average = exponential_moving_average(cpu_usage) with different time windows',
     'cpu_usage_timeseries, load_average_intervals, analysis_windows',
     'cpu_load_patterns, performance_bottlenecks, load_trend_analysis',
     'O(n)', 4, 'Analyze 1min, 5min, 15min load averages. Identify usage patterns.'],
     
    ['Memory_Pressure_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Performance_Analysis',
     'Analyzes memory pressure indicators including page faults and swap usage',
     'Memory_pressure = swap_usage + page_fault_rate + cache_hit_ratio',
     'memory_usage_stats, page_fault_counts, swap_activity',
     'memory_pressure_level, swap_efficiency, memory_bottleneck_detection',
     'O(n)', 4, 'High page faults and swap usage indicate memory pressure.'],
     
    ['Disk_IO_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Performance_Analysis',
     'Analyzes disk I/O patterns to identify storage bottlenecks and access patterns',
     'IO_analysis = read_rate + write_rate + seek_time + queue_depth',
     'disk_io_stats, read_write_patterns, disk_queue_metrics',
     'io_performance_patterns, storage_bottlenecks, disk_efficiency',
     'O(n)', 3, 'Track IOPS, throughput, and latency patterns over time.'],
     
    ['System_Responsiveness_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Performance_Analysis',
     'Analyzes overall system responsiveness using response time measurements',
     'Responsiveness = 1 / (average_response_time + response_time_variance)',
     'response_times, system_latency_measurements, user_interaction_delays',
     'responsiveness_score, latency_patterns, performance_degradation_indicators',
     'O(n)', 4, 'Measure system response to standard operations over time.'],
     
    ['Resource_Contention_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Performance_Analysis',
     'Identifies resource contention between processes and system components',
     'Contention_score = resource_wait_time / total_processing_time',
     'process_resource_usage, resource_locks, waiting_times',
     'resource_contention_levels, bottleneck_identification, resource_conflicts',
     'O(n²)', 3, 'Analyze which processes are competing for the same resources.'],

    # THERMAL ANALYSIS ALGORITHMS
    ['Thermal_Stress_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Thermal_Analysis',
     'Analyzes thermal stress patterns and heat distribution across system components',
     'Thermal_stress = (current_temp - baseline_temp) / thermal_design_power',
     'temperature_readings, baseline_temperatures, thermal_design_limits',
     'thermal_stress_levels, heat_distribution_patterns, cooling_efficiency',
     'O(n)', 5, 'Very biomimetic - like body temperature regulation analysis.'],
     
    ['Temperature_Gradient_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Thermal_Analysis',
     'Analyzes temperature gradients between different system components',
     'Temperature_gradient = ΔT / Δposition between thermal zones',
     'multi_zone_temperatures, component_positions, thermal_conductivity',
     'temperature_gradients, thermal_imbalances, hotspot_identification',
     'O(n)', 4, 'Identify thermal hotspots and cooling inefficiencies.'],
     
    ['Thermal_Response_Modeling', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Thermal_Analysis',
     'Models thermal response to computational load changes',
     'Thermal_response = thermal_mass * dT/dt + thermal_resistance * power_dissipation',
     'temperature_timeseries, computational_load, thermal_constants',
     'thermal_response_model, temperature_prediction, thermal_time_constants',
     'O(n)', 4, 'Physics-based thermal modeling for temperature prediction.'],
     
    ['Cooling_Efficiency_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Thermal_Analysis',
     'Analyzes cooling system efficiency based on fan speeds and temperature control',
     'Cooling_efficiency = heat_removed / (fan_power + pump_power)',
     'fan_speeds, temperature_reductions, power_consumption, cooling_curves',
     'cooling_efficiency, thermal_management_effectiveness, cooling_optimization',
     'O(n)', 4, 'Evaluate how effectively cooling systems manage thermal loads.'],

    # ENERGY ANALYSIS ALGORITHMS
    ['Power_Efficiency_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Energy_Analysis',
     'Analyzes power efficiency by correlating computational work with energy consumption',
     'Power_efficiency = computational_work_completed / energy_consumed',
     'power_consumption, computational_tasks, performance_metrics',
     'power_efficiency_ratio, energy_optimization_opportunities, efficiency_trends',
     'O(n)', 4, 'Important for sustainable computing and energy awareness.'],
     
    ['Energy_Usage_Pattern_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Energy_Analysis',
     'Analyzes patterns in energy usage over different time scales and workloads',
     'Pattern_analysis using time series decomposition and spectral analysis',
     'power_consumption_timeseries, workload_patterns, usage_cycles',
     'energy_usage_patterns, consumption_cycles, power_demand_forecasting',
     'O(n*log(n))', 3, 'Use FFT to find periodic patterns in energy consumption.'],
     
    ['Battery_Health_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Energy_Analysis',
     'Analyzes battery health including charge cycles, capacity degradation, and efficiency',
     'Battery_health = current_capacity / original_capacity * cycle_count_factor',
     'battery_stats, charge_cycles, capacity_measurements, charging_patterns',
     'battery_health_score, capacity_degradation_rate, charging_efficiency',
     'O(n)', 4, 'Track battery aging and predict replacement needs.'],
     
    ['Dynamic_Power_Scaling_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Energy_Analysis',
     'Analyzes effectiveness of dynamic power scaling and frequency adjustments',
     'Power_scaling_efficiency = performance_maintained / power_reduced',
     'cpu_frequencies, power_states, performance_scaling, workload_adaptation',
     'scaling_efficiency, power_state_transitions, adaptive_power_management',
     'O(n)', 3, 'Evaluate how well system adapts power usage to computational needs.'],

    # SYSTEM HEALTH ALGORITHMS
    ['Overall_System_Health_Assessment', 'Methodology', 'SENSORY_RAW->PATTERNS', 'Health_Analysis',
     'Combines multiple system metrics into overall health score using weighted factors',
     'Health_score = Σ(weight_i * normalized_metric_i) where weights sum to 1',
     'cpu_health, memory_health, disk_health, thermal_health, power_health',
     'overall_system_health (0.0-1.0), health_factor_contributions, critical_issues',
     'O(n)', 4, 'Biomimetic - like overall body health assessment combining vital signs.'],
     
    ['Hardware_Degradation_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Health_Analysis',
     'Detects hardware degradation by tracking performance metrics over time',
     'Degradation_rate = (baseline_performance - current_performance) / time_elapsed',
     'performance_baselines, current_performance_metrics, time_series_analysis',
     'degradation_indicators, hardware_aging_patterns, maintenance_predictions',
     'O(n)', 4, 'Track hardware aging patterns to predict maintenance needs.'],
     
    ['System_Stability_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Health_Analysis',
     'Analyzes system stability using crash frequency, error rates, and uptime patterns',
     'Stability_score = uptime / (uptime + downtime) * (1 - error_rate)',
     'system_uptime, crash_logs, error_frequencies, stability_events',
     'stability_score, reliability_metrics, failure_pattern_analysis',
     'O(n)', 4, 'Essential for assessing system reliability and robustness.'],
     
    ['Performance_Baseline_Establishment', 'Function', 'SENSORY_RAW->PATTERNS', 'Health_Analysis',
     'Establishes performance baselines during optimal operating conditions',
     'Baseline = statistical_mode(performance_metrics) during optimal_conditions',
     'historical_performance_data, optimal_condition_indicators, statistical_analysis',
     'performance_baselines, optimal_operating_ranges, deviation_thresholds',
     'O(n*log(n))', 3, 'Establish what "normal" performance looks like for comparison.'],
     
    ['Anomaly_Detection_System_Metrics', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Health_Analysis',
     'Detects anomalous system behavior using statistical and machine learning methods',
     'Anomaly_score = mahalanobis_distance(current_metrics, normal_distribution)',
     'system_metrics_timeseries, normal_behavior_model, anomaly_threshold',
     'anomaly_scores, anomalous_events, behavioral_deviations',
     'O(n)', 4, 'Use isolation forest or statistical methods to detect unusual system behavior.'],

    # ENVIRONMENTAL MONITORING
    ['Environmental_Condition_Monitoring', 'Function', 'SENSORY_RAW->PATTERNS', 'Environmental_Analysis',
     'Monitors environmental conditions that affect system performance',
     'Environmental_impact = temperature_factor * humidity_factor * pressure_factor',
     'ambient_temperature, humidity_sensors, atmospheric_pressure, air_quality',
     'environmental_conditions, environmental_stress_factors, operating_environment_assessment',
     'O(1)', 3, 'Monitor external factors that influence system performance.'],
     
    ['Air_Quality_Assessment', 'Function', 'SENSORY_RAW->PATTERNS', 'Environmental_Analysis',
     'Assesses air quality around system including dust, particle count, and ventilation',
     'Air_quality_index based on particulate matter and ventilation effectiveness',
     'particulate_sensors, dust_levels, ventilation_measurements, air_circulation',
     'air_quality_index, dust_contamination_levels, ventilation_effectiveness',
     'O(1)', 3, 'Important for maintaining clean operating environment.'],
     
    ['Vibration_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Environmental_Analysis',
     'Analyzes mechanical vibrations that could affect system stability',
     'Vibration_analysis using FFT to identify problematic frequencies',
     'accelerometer_data, vibration_sensors, mechanical_resonance_frequencies',
     'vibration_spectrum, mechanical_stress_indicators, resonance_problems',
     'O(n*log(n))', 3, 'Use accelerometer data and FFT analysis for vibration patterns.'],
     
    ['Electromagnetic_Interference_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Environmental_Analysis',
     'Detects electromagnetic interference that could affect system operation',
     'EMI_analysis using spectrum analysis of electromagnetic fields',
     'electromagnetic_field_sensors, radio_frequency_spectrum, interference_sources',
     'emi_levels, interference_sources, electromagnetic_compatibility',
     'O(n*log(n))', 2, 'Monitor EMI that could cause system instability or errors.'],

    # =============================================================================
    # STAGE 2: PATTERNS -> FRAGMENTS
    # =============================================================================
    
    ['System_Performance_Pattern_Integration', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Integrates multiple system performance patterns into coherent performance assessment',
     'Weighted integration of CPU, memory, disk, network, and thermal patterns',
     'cpu_patterns, memory_patterns, disk_patterns, network_patterns, thermal_patterns',
     'integrated_performance_profile, performance_bottleneck_identification, system_efficiency',
     'O(n)', 4, 'Combine multiple performance indicators into unified assessment.'],
     
    ['Resource_Correlation_Analysis', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Analyzes correlations between different system resources and their interdependencies',
     'Cross-correlation analysis between different resource utilization patterns',
     'resource_utilization_patterns, correlation_analysis, dependency_mapping',
     'resource_correlations, bottleneck_relationships, resource_dependencies',
     'O(n²)', 4, 'Understand how different system resources affect each other.'],
     
    ['Temporal_Performance_Analysis', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Analyzes performance patterns over different time scales',
     'Multi-scale temporal analysis: seconds, minutes, hours, days, weeks',
     'performance_timeseries, multiple_time_scales, seasonal_patterns',
     'temporal_performance_patterns, cyclic_behaviors, long_term_trends',
     'O(n*log(n))', 4, 'Identify daily, weekly, seasonal patterns in system performance.'],
     
    ['Workload_Characterization', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Characterizes different types of computational workloads and their resource requirements',
     'Statistical clustering of workload patterns and resource usage profiles',
     'workload_metrics, resource_usage_profiles, task_classifications',
     'workload_categories, resource_requirements, workload_predictions',
     'O(n²)', 4, 'Understand different types of computational work and their needs.'],
     
    ['Performance_Trend_Analysis', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Analyzes long-term performance trends and predicts future performance changes',
     'Time series trend analysis with seasonal decomposition and forecasting',
     'historical_performance_data, trend_analysis_parameters, forecasting_models',
     'performance_trends, future_performance_predictions, capacity_planning_insights',
     'O(n*log(n))', 3, 'Use time series analysis to predict future performance needs.'],

    # =============================================================================
    # STAGE 3: FRAGMENTS -> NODES
    # =============================================================================
    
    ['System_State_Concept_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Forms abstract concepts of different system states based on integrated performance data',
     'Hierarchical clustering of system states into conceptual categories',
     'system_performance_fragments, state_clustering_parameters, concept_abstraction',
     'system_state_concepts, state_categories, operational_modes',
     'O(n²)', 5, 'Create high-level understanding of different system operational states.'],
     
    ['Performance_Model_Learning', 'Model', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Learns predictive models of system performance under different conditions',
     'Machine learning models relating system conditions to performance outcomes',
     'system_conditions, performance_outcomes, model_training_data',
     'performance_prediction_models, condition_performance_relationships, model_accuracy',
     'O(n²)', 4, 'Learn to predict system performance based on current conditions.'],
     
    ['Resource_Optimization_Strategy_Learning', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Learns optimal resource allocation strategies based on performance history',
     'Reinforcement learning for optimal resource allocation under different workloads',
     'resource_allocation_history, performance_outcomes, optimization_objectives',
     'optimization_strategies, resource_allocation_policies, efficiency_improvements',
     'O(n²)', 4, 'Learn how to optimally allocate resources for different workloads.'],
     
    ['System_Health_Memory_Formation', 'Algorithm', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Forms long-term memory of system health states and their causes',
     'Associative memory linking system conditions to health outcomes',
     'health_states, causal_conditions, health_outcome_associations',
     'health_memory_associations, diagnostic_knowledge, health_prediction_capability',
     'O(n²)', 5, 'Very biomimetic - learn what conditions lead to good/poor health.'],

    # =============================================================================
    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP
    # =============================================================================
    
    ['System_State_Embedding', 'Model', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates dense vector embeddings for different system states enabling similarity search',
     'Neural embedding of system state features into low-dimensional vectors',
     'system_state_features, embedding_model, embedding_dimensions',
     'system_state_embeddings, state_similarity_vectors, clustering_capability',
     'O(n)', 3, 'Enable similarity search for system states and conditions.'],
     
    ['Performance_Pattern_Index', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates searchable index of performance patterns for analysis and optimization',
     'Inverted index of performance patterns with metadata and context',
     'performance_patterns, pattern_metadata, indexing_parameters',
     'performance_pattern_index, searchable_performance_data, pattern_retrieval',
     'O(n*log(n))', 3, 'Enable fast search and analysis of historical performance patterns.'],
     
    ['System_Health_Timeline_Index', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates temporal index for system health data enabling time-based health queries',
     'Time-series database optimized for system health and performance data',
     'health_timeline_data, temporal_metadata, health_indexing_parameters',
     'health_timeline_index, temporal_health_search, chronological_health_access',
     'O(n*log(n))', 4, 'Enable queries about system health over time periods.'],
     
    ['Predictive_Maintenance_Index', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates index for predictive maintenance based on system degradation patterns',
     'Maintenance prediction index based on component aging and failure patterns',
     'degradation_patterns, maintenance_history, failure_predictions',
     'maintenance_schedule_index, predictive_maintenance_alerts, component_lifecycle_tracking',
     'O(n*log(n))', 4, 'Enable proactive maintenance scheduling based on predicted needs.'],
     
    ['System_Optimization_Knowledge_Base', 'Methodology', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates knowledge base of system optimization strategies and their effectiveness',
     'Structured knowledge representation of optimization strategies and outcomes',
     'optimization_strategies, effectiveness_measurements, context_conditions',
     'optimization_knowledge_base, strategy_recommendations, optimization_search',
     'O(n²)', 4, 'Accumulate knowledge about what optimizations work under different conditions.']
]

# Add all the data to the dictionary
for algo_data in algorithms_data:
    physical_algorithms['Algorithm_Name'].append(algo_data[0])
    physical_algorithms['Type'].append(algo_data[1])
    physical_algorithms['Stage'].append(algo_data[2])
    physical_algorithms['Category'].append(algo_data[3])
    physical_algorithms['Description'].append(algo_data[4])
    physical_algorithms['Mathematical_Basis'].append(algo_data[5])
    physical_algorithms['Input_Data_Required'].append(algo_data[6])
    physical_algorithms['Output_Data_Generated'].append(algo_data[7])
    physical_algorithms['Computational_Complexity'].append(algo_data[8])
    physical_algorithms['Biomimetic_Relevance'].append(algo_data[9])
    physical_algorithms['Implementation_Notes'].append(algo_data[10])

# Create DataFrame
physical_df = pd.DataFrame(physical_algorithms)

# Display summary
print("PHYSICAL STATE SENSE - COMPLETE ALGORITHM CATALOG")
print("=" * 50)
print(f"Total Physical State Algorithms: {len(physical_df)}")
print(f"Stage 1 (SENSORY_RAW->PATTERNS): {len(physical_df[physical_df['Stage'] == 'SENSORY_RAW->PATTERNS'])}")
print(f"Stage 2 (PATTERNS->FRAGMENTS): {len(physical_df[physical_df['Stage'] == 'PATTERNS->FRAGMENTS'])}")
print(f"Stage 3 (FRAGMENTS->NODES): {len(physical_df[physical_df['Stage'] == 'FRAGMENTS->NODES'])}")
print(f"Stage 4 (NODES->SEMANTIC_MAP): {len(physical_df[physical_df['Stage'] == 'NODES->SEMANTIC_MAP'])}")
print()

# Show category breakdown
print("ALGORITHM CATEGORIES:")
category_counts = physical_df['Category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count} algorithms")
print()

# Show type breakdown
print("ALGORITHM TYPES:")
type_counts = physical_df['Type'].value_counts()
for algo_type, count in type_counts.items():
    print(f"  {algo_type}: {count} algorithms")
print()

# Show biomimetic relevance distribution
print("BIOMIMETIC RELEVANCE DISTRIBUTION:")
biomimetic_counts = physical_df['Biomimetic_Relevance'].value_counts().sort_index()
for relevance, count in biomimetic_counts.items():
    print(f"  Level {relevance}: {count} algorithms")
print()

# Show sample of the complete data
print("SAMPLE ALGORITHM DETAILS:")
print(physical_df[['Algorithm_Name', 'Type', 'Stage', 'Category', 'Biomimetic_Relevance']].head(10).to_string(index=False))

# Save to Excel if needed
# physical_df.to_excel('physical_state_algorithms_complete_catalog.xlsx', index=False)