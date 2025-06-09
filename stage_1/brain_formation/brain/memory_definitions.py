# --- memory_definitions.py V8 ---
"""
Enhanced memory definitions for brain formation with full Stage 2 structure
but core implementation for Stage 1 (sephiroth and identity fragments only)
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

# Memory types from original definitions plus enhancements
MEMORY_TYPES = {
    1: {
        "memory_type_id": 1,
        "name": "ephemeral",
        "base_frequency_hz": 23.7,
        "decay_rate": 0.1,
        "preferred_storage_duration_hours": 1,
        "typical_content": "temporary thoughts, immediate sensory data"
    },
    2: {
        "memory_type_id": 2,
        "name": "short_term_working",
        "base_frequency_hz": 18.2,
        "decay_rate": 0.05,
        "preferred_storage_duration_hours": 24,
        "typical_content": "daily experiences, active processing"
    },
    3: {
        "memory_type_id": 3,
        "name": "transitional",
        "base_frequency_hz": 14.8,
        "decay_rate": 0.02,
        "preferred_storage_duration_hours": 168,
        "typical_content": "learning consolidation, pattern formation"
    },
    4: {
        "memory_type_id": 4,
        "name": "long_term_semantic",
        "base_frequency_hz": 12.1,
        "decay_rate": 0.01,
        "preferred_storage_duration_hours": 8760,
        "typical_content": "concepts, knowledge, meanings"
    },
    5: {
        "memory_type_id": 5,
        "name": "long_term_episodic",
        "base_frequency_hz": 9.8,
        "decay_rate": 0.005,
        "preferred_storage_duration_hours": 17520,
        "typical_content": "personal experiences, events, narratives"
    },
    6: {
        "memory_type_id": 6,
        "name": "procedural",
        "base_frequency_hz": 7.4,
        "decay_rate": 0.002,
        "preferred_storage_duration_hours": 35040,
        "typical_content": "skills, habits, motor patterns"
    },
    7: {
        "memory_type_id": 7,
        "name": "emotional_imprint",
        "base_frequency_hz": 5.9,
        "decay_rate": 0.001,
        "preferred_storage_duration_hours": 87600,
        "typical_content": "emotional associations, trauma, joy"
    },
    8: {
        "memory_type_id": 8,
        "name": "core_identity",
        "base_frequency_hz": 3.2,
        "decay_rate": 0.0001,
        "preferred_storage_duration_hours": 525600,
        "typical_content": "fundamental self-concepts, identity aspects"
    },
    9: {
        "memory_type_id": 9,
        "name": "soul_essence",
        "base_frequency_hz": 1.1,
        "decay_rate": 0.00001,
        "preferred_storage_duration_hours": 8760000,
        "typical_content": "sephiroth aspects, spiritual traits, soul nature"
    }
}

# Signal patterns for memory encoding
SIGNAL_PATTERNS = {
    'simple_continuous': {
        'amplitude_range': (0.1, 0.3),
        'frequency_modifier': 1.0,
        'waveform': 'sine',
        'burst_pattern': 'continuous'
    },
    'complex_structured': {
        'amplitude_range': (0.3, 0.6),
        'frequency_modifier': 1.1,
        'waveform': 'composite_harmonic',
        'burst_pattern': 'organized_complex'
    },
    'burst_sequence': {
        'amplitude_range': (0.5, 0.9),
        'frequency_modifier': 0.95,
        'waveform': 'sawtooth_modified',
        'burst_pattern': 'sequential_bursts'
    },
    'rapid_changing': {
        'amplitude_range': (0.4, 0.7),
        'frequency_modifier': 1.3,
        'waveform': 'variable_complex',
        'burst_pattern': 'chaotic_rapid'
    },
    'geometric_harmonic': {
        'amplitude_range': (0.6, 0.8),
        'frequency_modifier': 1.15,
        'waveform': 'golden_ratio_based',
        'burst_pattern': 'fractal_recurring'
    },
    'transient_fluctuating': {
        'amplitude_range': (0.2, 0.5),
        'frequency_modifier': 1.4,
        'waveform': 'ephemeral_noise',
        'burst_pattern': 'transient_sporadic'
    }
}

def create_memory_fragment_structure(
    content: str,
    fragment_type: str,  # 'sephiroth' or 'identity'
    brain_region: str,
    coordinates: tuple,
    sephiroth_aspect: str = None,
    identity_aspect: str = None
) -> Dict[str, Any]:
    """
    Create a complete memory fragment structure ready for Stage 2 enhancement.
    Stage 1 uses core fields, Stage 2 will populate search optimization fields.
    """
    
    # Determine memory type based on fragment type
    if fragment_type == 'sephiroth':
        memory_type = MEMORY_TYPES[9]  # soul_essence
        domain_name = 'sephiroth_traits'
        domain_description = 'spiritual traits and aspects from sephiroth journey'
    elif fragment_type == 'identity':
        memory_type = MEMORY_TYPES[8]  # core_identity  
        domain_name = 'identity_aspects'
        domain_description = 'fundamental identity aspects and self-concepts'
    else:
        raise ValueError(f"Unknown fragment type: {fragment_type}")
    
    # Generate unique IDs
    node_id = str(uuid.uuid4())
    domain_id = str(uuid.uuid4())
    
    # Create timestamp
    creation_time = datetime.now().isoformat()
    
    # Memory fragment structure (full Stage 2 ready)
    memory_fragment = {
        # Core identifiers (Stage 1 active)
        "node_id": node_id,
        "coordinates": coordinates,
        "brain_region": brain_region,
        "previous_coordinate": None,  # Set if fragment moves
        
        # Content and classification (Stage 1 active)
        "content": content,
        "fragment_type": fragment_type,
        "sephiroth_aspect": sephiroth_aspect,
        "identity_aspect": identity_aspect,
        
        # Temporal properties (Stage 1 active)
        "creation_timestamp": creation_time,
        "last_accessed": creation_time,
        "decay_rate": memory_type["decay_rate"],
        "storage_duration_hours": memory_type["preferred_storage_duration_hours"],
        "energy_expenditure": memory_type["base_frequency_hz"] * 0.1,  # Normal synaptic energy
        
        # Frequency properties (Stage 1 active)
        "frequency_hz_active": memory_type["base_frequency_hz"],
        "frequency_hz_inactive": memory_type["base_frequency_hz"] * 0.1,
        "current_state": "inactive",  # Start inactive until retrieved
        
        # Node properties (Stage 1 active)
        "is_node": False,  # Memory fragment, not neural node
        
        # Domain structure (Stage 1 active)
        "domain": {
            "domain_id": domain_id,
            "name": domain_name,
            "description": domain_description,
            "wbs_level_id": 1,
            "memory_type_id": memory_type["memory_type_id"]
        },
        
        # Memory type details (Stage 1 active)
        "memory_type": {
            "memory_type_id": memory_type["memory_type_id"],
            "name": memory_type["name"],
            "base_frequency_hz": memory_type["base_frequency_hz"],
            "decay_rate": memory_type["decay_rate"],
            "preferred_storage_duration_hours": memory_type["preferred_storage_duration_hours"]
        },
        
        # Signal patterns (Stage 1 basic)
        "signal_patterns": SIGNAL_PATTERNS.get('geometric_harmonic', {}),
        
        # Search optimization fields (Stage 2 ready, basic values for Stage 1)
        "semantic_meaning": [],  # Will be populated in Stage 2
        "synonyms": [],  # Will be populated in Stage 2
        "antonyms": [],  # Will be populated in Stage 2
        
        # Similarity and confidence scores (Stage 2 ready, basic for Stage 1)
        "similarity_scores": {
            "domain_concept_similarity": 0.0,  # Will be calculated in Stage 2
            "confidence_score": 1.0,  # High confidence for sephiroth/identity
            "source_reliability": 1.0  # Perfect reliability for soul aspects
        },
        
        # Pattern recognition scores (Stage 2 ready, basic for Stage 1)
        "pattern_scores": {
            "visual_pattern": 0.0,  # Will be developed in Stage 2
            "auditory_pattern": 0.0,  # Will be developed in Stage 2
            "emotional_pattern": 0.8 if fragment_type == 'identity' else 0.6,  # Basic emotional relevance
            "prediction_accuracy": 0.0,  # Will be learned in Stage 2
            "learning_potential": 0.9 if fragment_type == 'sephiroth' else 0.7  # High for spiritual aspects
        },
        
        # Meta properties for Stage 2 enhancement
        "meta_tags": {
            "origin": "soul_formation",
            "stability": "high",
            "accessibility": "core",
            "stage_1_ready": True,
            "stage_2_ready": True
        },
        
        # Training and development markers (Stage 2 ready)
        "training_markers": {
            "reinforcement_count": 0,
            "last_reinforcement": None,
            "prediction_success_rate": 0.0,
            "connection_strength": 0.5,  # Medium initial strength
            "pruning_resistance": 0.9 if fragment_type == 'sephiroth' else 0.8  # High resistance to pruning
        }
    }
    
    return memory_fragment

def get_memory_type_by_id(memory_type_id: int) -> Dict[str, Any]:
    """Get memory type definition by ID."""
    return MEMORY_TYPES.get(memory_type_id, {})

def get_signal_pattern_by_name(pattern_name: str) -> Dict[str, Any]:
    """Get signal pattern definition by name."""
    return SIGNAL_PATTERNS.get(pattern_name, {})

def calculate_memory_priority(fragment_type: str, access_frequency: int = 0) -> float:
    """Calculate memory priority for retrieval and storage decisions."""
    base_priority = {
        'sephiroth': 0.9,  # Very high priority
        'identity': 0.8,   # High priority
    }.get(fragment_type, 0.5)
    
    # Access frequency bonus (logarithmic)
    import math
    access_bonus = math.log(access_frequency + 1) * 0.1
    
    return min(1.0, base_priority + access_bonus)

def create_simplified_memory_fragment(content: str, fragment_type: str, coordinates: tuple) -> Dict[str, Any]:
    """Create simplified memory fragment for basic operations (Stage 1 minimal)."""
    memory_type = MEMORY_TYPES[9] if fragment_type == 'sephiroth' else MEMORY_TYPES[8]
    
    return {
        "node_id": str(uuid.uuid4()),
        "content": content,
        "fragment_type": fragment_type,
        "coordinates": coordinates,
        "frequency_hz": memory_type["base_frequency_hz"],
        "decay_rate": memory_type["decay_rate"],
        "creation_timestamp": datetime.now().isoformat(),
        "active": False
    }
