def get_brain_structure_dictionary():
    """
    Return the complete brain structure dictionary model showing
    the hierarchical organization of the brain.
    """
    
    brain_model = {
        "regions": {
            "region_id": "UUID (PK)",
            "code": "String",          # One-letter code (C, D, B, L)
            "name": "String",          # Full name (cerebrum, cerebellum, etc.)
            "description": "Text",
            "volume_proportion": "Float",  # Proportion of total brain volume
            "cell_density": "Float",    # Base cell density for this region
            "base_frequency": "Float",  # Base frequency in Hz
            "position_x": "Float",      # Normalized position (0-1)
            "position_y": "Float",
            "position_z": "Float",
            "is_active": "Boolean"
        },
        
        "subregions": {
            "subregion_id": "UUID (PK)",
            "region_id": "Foreign Key -> regions",
            "code": "String",          # One-letter code (F, P, T, etc.)
            "subregion_number": "Integer",  # To distinguish multiple of same type
            "name": "String",          # Full name (frontal, parietal, etc.)
            "description": "Text",
            "volume_proportion": "Float",  # Proportion within parent region
            "cell_density_multiplier": "Float",  # Multiplier to parent density
            "base_frequency": "Float",  # Base frequency in Hz
            "position_x": "Float",      # Normalized position (0-1)
            "position_y": "Float",
            "position_z": "Float",
            "is_active": "Boolean"
        },
        
        "functional_areas": {
            "functional_area_id": "UUID (PK)",
            "subregion_id": "Foreign Key -> subregions",
            "code": "String",          # Identifier code
            "area_number": "Integer",   # To distinguish multiple of same type
            "name": "String",          # Name (prefrontal_cortex, etc.)
            "description": "Text",
            "volume_proportion": "Float",  # Proportion within subregion
            "cell_density_multiplier": "Float",  # Multiplier to subregion density
            "base_frequency": "Float",  # Base frequency in Hz
            "position_x": "Float",      # Normalized position (0-1)
            "position_y": "Float",
            "position_z": "Float",
            "is_active": "Boolean"
        },
        
        "blocks": {
            "block_id": "UUID (PK)",
            "block_number": "Integer",  # Sequential number within parent
            "region_id": "Foreign Key -> regions",
            "subregion_id": "Foreign Key -> subregions",
            "functional_area_id": "Foreign Key -> functional_areas",
            "block_code": "String",     # Block identifier code (e.g., "023")
            "position_x": "Integer",    # Actual grid position
            "position_y": "Integer",
            "position_z": "Integer",
            "size_x": "Integer",        # Block dimensions
            "size_y": "Integer",
            "size_z": "Integer",
            "cell_capacity": "Integer", # Max cells in this block
            "active_cell_count": "Integer", # Currently active cells
            "is_boundary": "Boolean",   # Whether this is a boundary block
            "boundary_ids": "Array",    # List of boundaries this block participates in
            "base_frequency": "Float",  # Base frequency for this block
            "resonance": "Float",       # Base resonance value
            "coherence": "Float",       # Base coherence value
            "stability": "Float",       # Base stability value
            "energy_level": "Float",    # Current energy level
            "creation_time": "Timestamp"
        },
        
        "cells": {
            "cell_id": "UUID (PK)",
            "cell_number": "Integer",   # Sequential number within block
            "block_id": "Foreign Key -> blocks",
            "position_x": "Integer",    # Actual grid position
            "position_y": "Integer",
            "position_z": "Integer",
            "full_address": "String",   # Full hierarchical address (e.g., "F1-023-12345")
            "is_active": "Boolean",     # Whether cell is currently active
            "is_boundary": "Boolean",   # Whether this is a boundary cell
            "is_transmission_only": "Boolean", # Pure transmission vs storage
            "transmits_sound": "Boolean", # Whether cell transmits sound
            "energy_level": "Float",    # Current energy level
            "frequency": "Float",       # Current frequency
            "resonance": "Float",       # Current resonance value
            "coherence": "Float",       # Current coherence value
            "stability": "Float",       # Current stability value
            "memory_id": "String",      # Reference to stored memory (if any)
            "soul_presence": "Float",   # Level of soul presence (0-1)
            "activation_time": "Timestamp", # When cell was last activated
            "last_update": "Timestamp", # When cell was last updated
            "creation_time": "Timestamp"
        },
        
        "brain_boundaries": {
            "boundary_id": "UUID (PK)",
            "source_region_id": "Foreign Key -> regions",
            "target_region_id": "Foreign Key -> regions",
            "source_subregion_id": "Foreign Key -> subregions",
            "target_subregion_id": "Foreign Key -> subregions",
            "boundary_type": "String",  # sharp, gradual, permeable
            "permeability": "Float",    # Permeability value (0-1)
            "transition_width": "Integer", # Width in cells
            "sound_frequency": "Float", # Base sound frequency for this boundary
            "blocks_involved": "Array", # List of blocks that form this boundary
            "cell_count": "Integer",    # Number of cells in this boundary
            "is_active": "Boolean",
            "creation_time": "Timestamp"
        },
        
        "cell_field_values": {
            "field_value_id": "UUID (PK)",
            "cell_id": "Foreign Key -> cells",
            "field_type": "String",     # energy, frequency, resonance, coherence, stability
            "value": "Float",           # Current field value
            "last_update": "Timestamp"
        },
        
        "mycelial_seeds": {
            "seed_id": "UUID (PK)",
            "region_id": "Foreign Key -> regions",
            "subregion_id": "Foreign Key -> subregions",
            "block_id": "Foreign Key -> blocks",
            "seed_code": "String",      # Seed identifier (e.g., "MS_F1_3")
            "position_x": "Integer",    # Actual grid position
            "position_y": "Integer",
            "position_z": "Integer",
            "energy_capacity": "Float", # Maximum energy storage
            "current_energy": "Float",  # Current stored energy
            "base_frequency": "Float",  # Base frequency in Hz
            "resonance": "Float",       # Current resonance value
            "stability": "Float",       # Stability value
            "coherence": "Float",       # Coherence value
            "connection_count": "Integer", # Number of connections to other seeds
            "quantum_connections": "Integer", # Number of quantum connections
            "is_active": "Boolean",
            "creation_time": "Timestamp"
        },
        
        "mycelial_connections": {
            "connection_id": "UUID (PK)",
            "source_seed_id": "Foreign Key -> mycelial_seeds",
            "target_seed_id": "Foreign Key -> mycelial_seeds",
            "connection_type": "String", # standard, quantum, hybrid
            "is_quantum": "Boolean",    # Whether quantum entangled
            "distance": "Float",        # Distance between seeds
            "efficiency": "Float",      # Energy transfer efficiency
            "bandwidth": "Float",       # Maximum energy per transmission
            "transmission_time": "Float", # Time for energy to transfer
            "energy_loss_rate": "Float", # Energy loss per unit distance
            "is_active": "Boolean",
            "creation_time": "Timestamp",
            "last_used": "Timestamp"
        },
        
        "energy_routes": {
            "route_id": "UUID (PK)",
            "route_number": "Integer",  # Route identifier (e.g., 20)
            "source_address": "String", # Source address (e.g., "E24")
            "target_address": "String", # Target address
            "source_seed_id": "Foreign Key -> mycelial_seeds",
            "target_seed_id": "Foreign Key -> mycelial_seeds",
            "path_description": "Text", # Description of path
            "path_points": "Array",     # List of coordinates along path
            "distance": "Float",        # Total route distance
            "efficiency": "Float",      # Energy transfer efficiency
            "bandwidth": "Float",       # Maximum energy per transmission
            "transmission_time": "Float", # Time for energy to transfer
            "energy_loss_rate": "Float", # Energy loss per transmission
            "frequency_pattern": "String", # Pattern for packet transmission
            "is_active": "Boolean",
            "creation_time": "Timestamp",
            "last_used": "Timestamp"
        },
        
        "coordinates": {
            "coordinate_id": "UUID (PK)",
            "x": "Integer",
            "y": "Integer",
            "z": "Integer",
            "region_id": "Foreign Key -> regions", 
            "subregion_id": "Foreign Key -> subregions",
            "functional_area_id": "Foreign Key -> functional_areas",
            "block_id": "Foreign Key -> blocks",
            "cell_id": "Foreign Key -> cells",
            "address": "String",        # Hierarchical address (e.g., "F1-023-12345")
            "original_x": "Integer",    # Original position (if moved)
            "original_y": "Integer",
            "original_z": "Integer",
            "is_active": "Boolean",
            "creation_time": "Timestamp",
            "last_update": "Timestamp"
        },
        
        "cell_states": {
            "state_id": "UUID (PK)",
            "cell_id": "Foreign Key -> cells",
            "state_type": "String",     # active, inactive, boundary, resonant
            "energy_level": "Float",
            "frequency": "Float",
            "resonance": "Float",
            "coherence": "Float",
            "stability": "Float",
            "soul_presence": "Float",   # Level of soul presence (0-1)
            "timestamp": "Timestamp"
        },
        
        "field_state": {
            "field_state_id": "UUID (PK)",
            "timestamp": "Timestamp",
            "state_type": "String",     # baseline, current, snapshot
            "region_id": "Foreign Key -> regions",
            "subregion_id": "Foreign Key -> subregions",
            "block_id": "Foreign Key -> blocks",
            "avg_energy": "Float",      # Average energy in this area
            "avg_frequency": "Float",   # Average frequency
            "avg_resonance": "Float",   # Average resonance
            "avg_coherence": "Float",   # Average coherence
            "avg_stability": "Float",   # Average stability
            "active_cell_count": "Integer", # Number of active cells
            "total_cell_count": "Integer", # Total cells in this area
            "field_stability": "Float", # Overall field stability
            "field_coherence": "Float", # Overall field coherence
            "description": "Text"
        },
        
        "consciousness_state": {
            "state_id": "UUID (PK)",
            "timestamp": "Timestamp",
            "state_name": "String",     # liminal, dream, awareness
            "state_level": "Float",     # Intensity level of state (0-1)
            "dominant_frequency": "Float", # Dominant frequency of this state
            "energy_level": "Float",    # Overall energy level
            "coherence": "Float",       # Overall coherence
            "resonance": "Float",       # Overall resonance
            "stability": "Float",       # Overall stability
            "active_regions": "Array",  # List of most active regions
            "active_cell_percentage": "Float", # Percentage of active cells
            "state_description": "Text" # Description of state
        },
        
        "wave_patterns": {
            "pattern_id": "UUID (PK)",
            "pattern_name": "String",   # Name of the wave pattern
            "pattern_type": "String",   # standing, traveling, resonant
            "base_frequency": "Float",  # Base frequency of pattern
            "amplitude": "Float",       # Wave amplitude
            "phase": "Float",           # Wave phase
            "harmonics": "Array",       # List of harmonic frequencies
            "propagation_speed": "Float", # Speed of wave propagation
            "attenuation_rate": "Float", # Energy loss during propagation
            "region_ids": "Array",      # Regions affected by this pattern
            "is_active": "Boolean",
            "creation_time": "Timestamp"
        },
        
        "sound_patterns": {
            "sound_id": "UUID (PK)",
            "pattern_name": "String",   # Name of the sound pattern
            "pattern_type": "String",   # boundary, oscillation, harmonic
            "base_frequency": "Float",  # Base frequency of sound
            "amplitude": "Float",       # Sound amplitude
            "harmonics": "Array",       # List of harmonic frequencies
            "phase_pattern": "String",  # Pattern of phase changes
            "boundary_id": "Foreign Key -> brain_boundaries", # Related boundary
            "cells_involved": "Array",  # List of cells transmitting this sound
            "description": "Text",
            "is_active": "Boolean",
            "creation_time": "Timestamp"
        },
        
        "brain_seed": {
            "seed_id": "UUID (PK)",
            "position_x": "Integer",    # Actual grid position
            "position_y": "Integer",
            "position_z": "Integer",
            "region_id": "Foreign Key -> regions",
            "subregion_id": "Foreign Key -> subregions",
            "block_id": "Foreign Key -> blocks",
            "cell_id": "Foreign Key -> cells",
            "address": "String",        # Hierarchical address
            "base_energy_level": "Float", # Core energy level
            "mycelial_energy_store": "Float", # Mycelial network stored energy
            "energy_capacity": "Float", # Maximum energy capacity
            "base_frequency_hz": "Float", # Base frequency
            "frequency_harmonics": "Array", # Harmonic frequencies
            "resonance": "Float",       # Resonance value
            "coherence": "Float",       # Coherence value
            "stability": "Float",       # Stability value
            "complexity": "Integer",    # Seed complexity level
            "formation_progress": "Float", # Progress toward full formation
            "energy_generators": "Array", # List of energy generator info
            "soul_connected": "Boolean", # Whether connected to soul
            "soul_connection_strength": "Float", # Connection strength to soul
            "creation_time": "Timestamp",
            "last_updated": "Timestamp"
        },
        
        "memory_fragments": {
            "fragment_id": "UUID (PK)",
            "cell_id": "Foreign Key -> cells",
            "address": "String",        # Hierarchical address 
            "content": "Text",          # Fragment content
            "frequency_hz": "Float",    # Fragment frequency
            "energy_level": "Float",    # Fragment energy level
            "activated": "Boolean",     # Whether fragment is activated
            "resonance": "Float",       # Resonance value
            "coherence": "Float",       # Coherence value
            "is_soul_aspect": "Boolean", # Whether fragment is a soul aspect
            "associations": "Array",    # List of associated fragment IDs
            "meta_tags": "Object",      # Metadata tags for fragment
            "creation_time": "Timestamp",
            "last_accessed": "Timestamp",
            "access_count": "Integer"
        },
        
        "energy_transfers": {
            "transfer_id": "UUID (PK)",
            "source_type": "String",    # seed, cell, node, generator
            "source_id": "String",      # ID of source
            "target_type": "String",    # seed, cell, node, fragment
            "target_id": "String",      # ID of target
            "route_id": "Foreign Key -> energy_routes",
            "energy_amount": "Float",   # Amount of energy transferred
            "energy_delivered": "Float", # Amount actually delivered
            "energy_lost": "Float",     # Amount lost in transfer
            "efficiency": "Float",      # Transfer efficiency
            "transfer_time": "Float",   # Time taken for transfer
            "packet_frequency": "Float", # Frequency of transfer packet
            "timestamp": "Timestamp",
            "success": "Boolean"        # Whether transfer succeeded
        },
        
        "soul_presence_map": {
            "map_id": "UUID (PK)",
            "timestamp": "Timestamp",
            "region_id": "Foreign Key -> regions",
            "subregion_id": "Foreign Key -> subregions",
            "block_id": "Foreign Key -> blocks",
            "average_presence": "Float", # Average soul presence in area
            "presence_cells": "Integer", # Number of cells with presence
            "total_cells": "Integer",   # Total cells in area
            "dominant_frequency": "Float", # Dominant frequency in area
            "state_description": "Text"
        }
    }
    
    return brain_model