# memory types determine the coordinate placement within the brain grid. there is random placement in the 
# mirror grid when it a fragment is created. memory fragment placement is determined by the observed date from where brain scans highlight certain areas
# when thinking of certain things plus we apply some dimensionality by using the 3d map to place less 
# relevant/used data further from the main domain. Memory fragments 
memory_type = {
    "memory_type_id": "UUID",
    "memory_frequency_hz": "",
    "decay_rate":"",
    "storage_duration":"",
    "brain_sub_region_storage":"",
    "typical_content": "str",
    "memory_type_description": "str",
}

mycelial_seed = {
    "mycelial_seed_id": "UUID",
    "mycelial_seed_creation_time": "",
    "mycelial_seed_last_accessed_time": "",
    "mycelial_seed_access_count": 0,
    "mycelial_seed_state": "",  # active, inactive, archived
    "mycelial_seed_coordinates": "",  # coordinates in the mirror brain grid
}

signal_pattern = {
    "signal_pattern_id": "UUID",
    "signal_pattern_name": "",
    "signal_pattern_description": "",
    "amplitude_range": "",
    "frequency_modifier": "",
    "waveform": "",
    "burst_pattern": ""
}

brain_states = {
        "brain_state_id": "UUID",
        "brain_state_name": "",
        "brain_state_description": "",
        "dominant_frequency_range": "",
        "processing_speed_modifier": "",
        "pattern_sensitivity": "",
        "emotional_sensitivity": "",
        "default_processing_intensity": ""
}

fragment = {
    "fragment_ID": "UUID",
    "fragment_created_time": "",
    "fragment_access_frequency": "",
    "fragment_access_count": "",
    "temporal_references": "",
    "sensory_modality_count": "",
    "fragment_coordinates": "",
    "fragment_frequency_hz": "", # frequency of fragment in hz depends on state all active,inactive,archived have own frequency
    "fragment_quality": "", # calculated score of quality in the memory average of quality from each sensory input
    "fragment_confidence": "",  # calculated score of confidence in the memory average of confidence from each sensory input
    "fragment_state": "", # active, inactive, archived
    "fragment_detail_blob": "", # blob of data for the fragment
    "fk_signal_pattern_id": "",
    "fk_memory_type_id": "",
    "fk_brain_state_id": ""
}

# we would need tables to store algorithm outcomes so that we can have the intermediate step before node is populated



nodes = {
    "node_id": "UUID",
    "node_state": "",
    "node_coordinates": "",
    "node_connections": [],
    "node_confidence": "",
    "node_quality": "",
    "node_domain": "", # this is the initial classification of the node which creates a record in wbs level 1
    "node_created_time": "",
    "node_last_accessed_time": "",
    "node_accessed_count": "",
    "node_processing_intensity": "", # how much processing is required to access the node
    "node_summary_blob": "", # blob of data for the node goes through a summarisation step extracting key info from all fragments
    "fk_node_cluster_route_id": "", # routes to other clusters of nodes
    "fk_wbs_structure_ID": "", # this is the connection to full wbs table
}

node_cluster_routes = {
    "node_cluster_route_id": "UUID",
    "route_connections_count": 0,
    "route_connections": [], # array of node IDs that this route connects to
    "route_quality": "",
    "route_confidence": "",
    "route_created_time": "",
    "route_last_accessed_time": "",
    "route_accessed_count": "",
    "route_state": "", # active, inactive, archived
}

