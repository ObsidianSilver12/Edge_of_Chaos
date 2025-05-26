# --- START OF FILE stage_1/evolve/evolve_constants.py ---

"""
Constants for Stage 1 Brain Evolution.

This file defines constants used throughout the Stage 1 brain evolution process,
including grid dimensions, region definitions, energy parameters, and frequencies.
"""

import numpy as np
from enum import Enum

# --- General Constants ---
FLOAT_EPSILON = 1e-9
PHI = 1.618033988749895  # Golden ratio
SPEED_OF_LIGHT = 299792458.0  # m/s

# --- Brain Grid Constants (Simplified) ---
GRID_DIMENSIONS = (256, 256, 256)  # 3D grid size
DEFAULT_BLOCK_COUNT = 1000   # Approximate number of blocks to divide brain into

# --- Region Names (Unchanged) ---
REGION_FRONTAL = "frontal"
REGION_PARIETAL = "parietal"
REGION_TEMPORAL = "temporal"
REGION_OCCIPITAL = "occipital"
REGION_LIMBIC = "limbic"
REGION_BRAIN_STEM = "brain_stem"
REGION_CEREBELLUM = "cerebellum"

# --- Region Proportions (Unchanged) ---
REGION_PROPORTIONS = {
    REGION_FRONTAL: 0.28,
    REGION_PARIETAL: 0.15,
    REGION_TEMPORAL: 0.17,
    REGION_OCCIPITAL: 0.13,
    REGION_LIMBIC: 0.10,
    REGION_BRAIN_STEM: 0.03,
    REGION_CEREBELLUM: 0.14
}

# --- Region Locations (Normalized coordinates) ---
REGION_LOCATIONS = {
    REGION_FRONTAL: (0.3, 0.7, 0.5),    # Front upper part
    REGION_PARIETAL: (0.7, 0.7, 0.5),   # Rear upper part
    REGION_TEMPORAL: (0.5, 0.4, 0.3),   # Side middle part
    REGION_OCCIPITAL: (0.8, 0.5, 0.5),  # Rear part
    REGION_LIMBIC: (0.5, 0.5, 0.4),     # Central part
    REGION_BRAIN_STEM: (0.5, 0.3, 0.2), # Lower central part
    REGION_CEREBELLUM: (0.7, 0.3, 0.3)  # Lower rear part
}

# --- Region Default Frequencies ---
REGION_DEFAULT_FREQUENCIES = {
    REGION_FRONTAL: 13.0,      # Primarily beta waves
    REGION_PARIETAL: 10.0,     # Alpha/beta mix
    REGION_TEMPORAL: 9.0,      # Primarily alpha waves
    REGION_OCCIPITAL: 11.0,    # Alpha/beta mix
    REGION_LIMBIC: 6.0,        # Theta waves
    REGION_BRAIN_STEM: 4.0,    # Delta/theta mix
    REGION_CEREBELLUM: 8.0     # Alpha waves
}

# --- Cell Density Parameters ---
DEFAULT_CELL_DENSITY = 0.05  # Default 5% of cells are active
BOUNDARY_CELL_DENSITY = 1.0  # 100% of cells are active at boundaries
ACTIVE_CELL_THRESHOLD = 0.1  # Energy level to consider a cell active

# --- Boundary Types ---
BOUNDARY_TYPE_SHARP = "sharp"       # Clear delineation
BOUNDARY_TYPE_GRADUAL = "gradual"   # Gradual transition
BOUNDARY_TYPE_PERMEABLE = "permeable"  # Very permeable

# --- Boundary Parameters ---
BOUNDARY_PARAMETERS = {
    BOUNDARY_TYPE_SHARP: {
        "transition_width": 1,
        "permeability": 0.3
    },
    BOUNDARY_TYPE_GRADUAL: {
        "transition_width": 3,
        "permeability": 0.6
    },
    BOUNDARY_TYPE_PERMEABLE: {
        "transition_width": 5,
        "permeability": 0.9
    }
}

# --- Region Boundary Mappings ---
REGION_BOUNDARIES = {
    (REGION_FRONTAL, REGION_PARIETAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_PARIETAL, REGION_OCCIPITAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_TEMPORAL, REGION_FRONTAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_TEMPORAL, REGION_PARIETAL): BOUNDARY_TYPE_GRADUAL,
    (REGION_LIMBIC, REGION_FRONTAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_TEMPORAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_PARIETAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_OCCIPITAL): BOUNDARY_TYPE_PERMEABLE,
    (REGION_LIMBIC, REGION_BRAIN_STEM): BOUNDARY_TYPE_PERMEABLE,
    (REGION_BRAIN_STEM, REGION_CEREBELLUM): BOUNDARY_TYPE_SHARP,
}

# --- Mycelial Network Constants (Simplified) ---
DEFAULT_SEED_COUNT_PER_REGION = 3  # Default number of seeds per region
MYCELIAL_QUANTUM_RANGE = 50        # Maximum distance for quantum entanglement
MYCELIAL_ENERGY_EFFICIENCY = 0.95  # Energy transfer efficiency
ENERGY_ROUTE_MAX_DISTANCE = 100    # Maximum distance for direct energy routes

# --- Cell ID Prefixes ---
PREFIX_BLOCK = "B"         # Block ID prefix
PREFIX_CELL = "C"          # Regular cell ID prefix
PREFIX_BOUNDARY = "BC"     # Boundary cell ID prefix
PREFIX_MYCELIAL = "MS"     # Mycelial seed ID prefix
PREFIX_ROUTE = "R"         # Route ID prefix

# --- Memory Constants ---
MEMORY_ACTIVATION_ENERGY = 0.1    # Energy required to activate a memory
MEMORY_FREQUENCY_TOLERANCE = 1.0  # Default frequency tolerance for searches
MEMORY_RESONANCE_THRESHOLD = 0.5  # Minimum resonance for memory activation

# --- Sound Pattern Constants ---
SOUND_SHARP_FREQUENCY = 15.0      # Base frequency for sharp boundaries
SOUND_GRADUAL_FREQUENCY = 10.0    # Base frequency for gradual boundaries 
SOUND_PERMEABLE_FREQUENCY = 7.0   # Base frequency for permeable boundaries
SOUND_FREQUENCY_VARIATION = 1.0   # Random variation in boundary sounds

# --- Energy Constants ---
DEFAULT_SEED_ENERGY_CAPACITY = 50.0   # Base energy capacity for mycelial seeds
DEFAULT_INITIAL_ENERGY_RATIO = 0.2    # Seeds start with 20% of capacity
ENERGY_TRANSFER_QUANTUM_BONUS = 0.98  # Efficiency for quantum connections
ENERGY_TRANSFER_LOSS_PER_DISTANCE = 0.005  # Energy loss per unit distance

# --- Brain Seed Constants ---
MIN_BRAIN_SEED_ENERGY = 1.0       # Minimum energy for brain seed operation
DEFAULT_BRAIN_SEED_FREQUENCY = 7.83  # Schumann frequency default
SEED_FIELD_RADIUS = 5.0           # Energy field radius around seed
MYCELIAL_MAXIMUM_PATHWAY_LENGTH = 200  # Max distance for mycelial connections

# --- Brain Complexity Thresholds ---
BRAIN_COMPLEXITY_THRESHOLDS = {
    'energy_coverage': 0.3,        # Min percentage of cells with energy > 0.1
    'mycelial_coverage': 0.2,      # Min percentage of cells with mycelial density > 0.1
    'avg_resonance': 0.3,          # Min average resonance across brain
    'avg_energy': 0.2,             # Min average energy across brain
    'field_initialized': True      # All fields must be initialized
}

# --- Consciousness Activation Thresholds ---
CONSCIOUSNESS_ACTIVATION_THRESHOLDS = {
    'dream': 0.4,                  # Brain complexity for dream state
    'liminal': 0.6,                # Brain complexity for liminal transitions
    'aware': 0.8                   # Brain complexity for aware state
}

# --- State Constants ---
LIMINAL_BASE_FREQUENCY = 3.5      # Delta/theta boundary
DREAM_BASE_FREQUENCY = 5.5        # Theta wave dominant
AWARENESS_BASE_FREQUENCY = 9.0    # Alpha wave dominant
STATE_TRANSITION_THRESHOLD = 0.6  # Percentage of cells needed for state transition

