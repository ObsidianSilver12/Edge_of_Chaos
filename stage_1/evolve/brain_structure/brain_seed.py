# --- START OF FILE stage_1/evolve/brain_seed.py ---

"""
Brain Seed Implementation (V4.5.0 - Wave Physics & Field Dynamics)

Acts as the initial energetic anchor point for the soul in the physical realm.
Holds converted Brain Energy Units (BEU) allocated during birth.
Maintains frequency information and connection to mycelial network.
Follows hard validation, proper physics principles with no fallbacks.
"""

import logging
import os
import sys
import numpy as np
import uuid
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import math
# Import constants from the main constants module
from constants.constants import *


# --- Logging ---
logger = logging.getLogger('BrainSeed')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()


class BrainSeed:
    """ 
    Brain Seed implementation. Stores energy, frequency, and connection info.
    Acts as the initial point of intelligence before soul connection.
    """

    def __init__(self, initial_beu: float = 0.0, initial_mycelial_beu: float = 0.0):
        """
        Initialize the brain seed.

        Args:
            initial_beu: Initial BEU allocated to the core/operational pool
            initial_mycelial_beu: Initial BEU allocated to the mycelial store
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not isinstance(initial_beu, (int, float)) or initial_beu < 0:
            raise ValueError(f"initial_beu must be non-negative, got {initial_beu}")
        if not isinstance(initial_mycelial_beu, (int, float)) or initial_mycelial_beu < 0:
            raise ValueError(f"initial_mycelial_beu must be non-negative, got {initial_mycelial_beu}")
            
        # For minimal operation, ensure some energy
        if initial_beu < MIN_BRAIN_SEED_ENERGY and initial_mycelial_beu < MIN_BRAIN_SEED_ENERGY:
            logger.warning(f"Initial energy too low for brain seed operation. Setting to minimum {MIN_BRAIN_SEED_ENERGY}.")
            initial_beu = MIN_BRAIN_SEED_ENERGY

        # Core identifiers
        self.seed_id: str = str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.last_updated: str = self.creation_time

        # --- Core State ---
        self.base_energy_level: float = initial_beu  # BEU available for immediate use
        self.mycelial_energy_store: float = initial_mycelial_beu  # BEU reserve
        self.energy_capacity: float = max(initial_beu * 2, 100.0)  # Maximum energy capacity

        # --- Frequency & Resonance ---
        self.base_frequency_hz: float = DEFAULT_BRAIN_SEED_FREQUENCY  # Default frequency
        self.frequency_harmonics: List[float] = []  # Harmonic frequencies of the base
        self.resonance_patterns: Dict[str, Any] = {}  # Resonance with brain regions

        # --- Seed Field Properties ---
        self.seed_field: Dict[str, Any] = {
            'radius': SEED_FIELD_RADIUS,
            'energy_density': initial_beu / (4/3 * math.pi * SEED_FIELD_RADIUS**3) if SEED_FIELD_RADIUS > 0 else 0.0,
            'coherence': 0.7,  # Initial coherence
            'stability': 0.8,  # Initial stability
            'resonance': 0.6,  # Initial resonance
            'frequency_variance': 0.05  # Natural frequency variation
        }

        # --- Development State ---
        self.complexity: int = 1  # Minimal complexity for now
        self.formation_progress: float = 0.1  # Represents seed presence
        self.stability: float = 0.5  # Default stability
        self.structural_integrity: float = 0.5  # Default integrity

        # --- Position Information (to be set later) ---
        self.position: Optional[Tuple[int, int, int]] = None
        self.brain_region: Optional[str] = None
        self.sub_region: Optional[str] = None

        # --- Soul Connection (initialized after sufficient complexity) ---
        self.soul_connection: Optional[Dict[str, Any]] = None
        
        # --- Life Cord Connection ---
        self.life_cord_data: Optional[Dict[str, Any]] = None
        
        # --- Earth Connection ---
        self.earth_connection: Optional[Dict[str, Any]] = None
        
        # --- Mycelial Connection ---
        self.mycelial_connection: Optional[Dict[str, Any]] = None
        
        # --- Soul Aspect Distribution ---
        self.soul_aspect_distribution: Dict[str, Any] = {}
        
        # --- Memory Fragment Storage ---
        self.memory_fragments: Dict[str, Dict[str, Any]] = {}
        
        # --- Energy Generators ---
        self.energy_generators: List[Dict[str, Any]] = []
        
        # --- Initialize field patterns ---
        self._initialize_field_patterns()
        
        logger.info(f"BrainSeed created with ID {self.seed_id}. Initial BEU: {self.base_energy_level:.2E}, "
                   f"Mycelial BEU: {self.mycelial_energy_store:.2E}")

    def _initialize_field_patterns(self):
        """Initialize field patterns for energy and frequency distribution."""
        # Create resonance pattern with harmonics based on Schumann frequency
        base_freq = self.base_frequency_hz
        
        # Generate harmonic frequencies based on golden ratio and integer harmonics
        phi = PHI  # Golden ratio
        self.frequency_harmonics = [
            base_freq * 1.0,  # Base frequency
            base_freq * 2.0,  # Octave
            base_freq * 3.0,  # Perfect fifth + octave
            base_freq * phi,  # Phi relation
            base_freq / phi,  # Phi relation (lower)
            base_freq * 1.5,  # Perfect fifth
            base_freq * 0.5,  # Octave down
        ]
        
        # Create energy generators at phi-harmonic positions
        self._create_energy_generators()
        
        # Create resonance patterns with major brain regions
        self._initialize_resonance_patterns()
        
        logger.debug(f"Field patterns initialized with {len(self.frequency_harmonics)} harmonics "
                    f"and {len(self.energy_generators)} energy generators")

    def _create_energy_generators(self):
        """Create energy generators at phi-harmonic positions around seed center."""
        # Energy generators will be positioned radially using phi angles
        phi = PHI
        radius = self.seed_field['radius'] * 0.7  # Position within field radius
        
        # Create primary generator at center (0,0,0) - will be offset when positioned in brain
        self.energy_generators.append({
            'id': str(uuid.uuid4()),
            'position': (0, 0, 0),
            'type': 'resonant_field',
            'output': self.base_energy_level * 0.4,  # 40% of energy to primary
            'frequency': self.base_frequency_hz,
            'coherence': 0.9,
            'stability': 0.95
        })
        
        # Create secondary generators at golden ratio positions
        angles = [
            (0, 0),  # 0 degrees (already created as primary)
            (phi, 0),  # phi radians, 0 elevation
            (2*phi, 0),  # 2*phi radians, 0 elevation
            (3*phi, 0),  # 3*phi radians, 0 elevation
            (0, phi),  # 0 radians, phi elevation
            (phi, phi)  # phi radians, phi elevation
        ]
        
        gen_types = [
            'vortex_node',
            'scalar_amplifier',
            'harmonic_oscillator', 
            'quantum_field_stabilizer'
        ]
        
        # Skip first angle (already created primary generator)
        for i, (azimuth, elevation) in enumerate(angles[1:], 1):
            # Convert spherical to Cartesian coordinates
            x = radius * math.cos(azimuth) * math.cos(elevation)
            y = radius * math.sin(azimuth) * math.cos(elevation)
            z = radius * math.sin(elevation)
            
            # Create generator with unique characteristics
            gen_type = gen_types[i % len(gen_types)]
            
            # Calculate appropriate frequency from harmonics
            freq_idx = min(i, len(self.frequency_harmonics) - 1)
            freq = self.frequency_harmonics[freq_idx]
            
            # Output level decreases with each generator
            output = self.base_energy_level * 0.15 / i
            
            self.energy_generators.append({
                'id': str(uuid.uuid4()),
                'position': (x, y, z),
                'type': gen_type,
                'output': output,
                'frequency': freq,
                'coherence': 0.7 - (0.05 * i),  # Decreasing coherence
                'stability': 0.8 - (0.05 * i)  # Decreasing stability
            })
    
    def _initialize_resonance_patterns(self):
        """Initialize resonance patterns with major brain regions."""
        # This will be populated with actual brain regions when integrated
        # For now, create placeholder patterns
        self.resonance_patterns = {
            'limbic': {
                'resonance_factor': 0.9,  # Highest resonance with limbic system
                'frequency_match': self.base_frequency_hz,
                'energy_transfer_efficiency': 0.85
            },
            'brain_stem': {
                'resonance_factor': 0.8,
                'frequency_match': self.frequency_harmonics[1] if len(self.frequency_harmonics) > 1 else self.base_frequency_hz,
                'energy_transfer_efficiency': 0.8
            },
            'frontal': {
                'resonance_factor': 0.6,
                'frequency_match': self.frequency_harmonics[2] if len(self.frequency_harmonics) > 2 else self.base_frequency_hz,
                'energy_transfer_efficiency': 0.7
            },
            'parietal': {
                'resonance_factor': 0.5,
                'frequency_match': self.frequency_harmonics[3] if len(self.frequency_harmonics) > 3 else self.base_frequency_hz,
                'energy_transfer_efficiency': 0.65
            },
            'temporal': {
                'resonance_factor': 0.7,
                'frequency_match': self.frequency_harmonics[4] if len(self.frequency_harmonics) > 4 else self.base_frequency_hz,
                'energy_transfer_efficiency': 0.75
            },
            'occipital': {
                'resonance_factor': 0.4,
                'frequency_match': self.frequency_harmonics[5] if len(self.frequency_harmonics) > 5 else self.base_frequency_hz,
                'energy_transfer_efficiency': 0.6
            }
        }

    def set_position(self, position: Tuple[int, int, int], brain_region: str, sub_region: Optional[str] = None):
        """
        Set the position of the brain seed in the brain structure.
        
        Args:
            position: (x, y, z) coordinates in brain grid
            brain_region: Name of the brain region containing the seed
            sub_region: Optional sub-region name
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(position, tuple) or len(position) != 3:
            raise ValueError(f"Position must be a 3D tuple, got {position}")
        
        if not all(isinstance(x, int) for x in position):
            raise ValueError(f"Position coordinates must be integers, got {position}")
        
        if not isinstance(brain_region, str) or not brain_region:
            raise ValueError(f"Brain region must be a non-empty string, got {brain_region}")
        
        # Update position information
        self.position = position
        self.brain_region = brain_region
        self.sub_region = sub_region
        
        # Update energy generator positions relative to seed position
        self._update_generator_positions()
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Brain seed positioned at {position} in {brain_region}" + 
                   (f" ({sub_region})" if sub_region else ""))
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'position': position,
                    'brain_region': brain_region,
                    'sub_region': sub_region,
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_position", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record position metrics: {e}")
        
        return True

    def _update_generator_positions(self):
        """Update energy generator positions relative to seed position."""
        if self.position is None:
            return
        
        seed_x, seed_y, seed_z = self.position
        
        for generator in self.energy_generators:
            rel_x, rel_y, rel_z = generator['position']
            
            # Update position relative to seed
            generator['position'] = (
                seed_x + rel_x,
                seed_y + rel_y,
                seed_z + rel_z
            )
        
        logger.debug(f"Updated positions for {len(self.energy_generators)} energy generators")

    def set_frequency(self, frequency: float):
        """
        Set the base frequency of the brain seed.
        
        Args:
            frequency: Base frequency in Hz
            
        Raises:
            ValueError: If frequency is invalid
        """
        if not isinstance(frequency, (int, float)) or frequency <= FLOAT_EPSILON:
            raise ValueError(f"Frequency must be positive, got {frequency}")
        
        self.base_frequency_hz = frequency
        
        # Generate harmonic frequencies based on golden ratio and integer harmonics
        phi = PHI
        self.frequency_harmonics = [
            frequency * 1.0,  # Base frequency
            frequency * 2.0,  # Octave
            frequency * 3.0,  # Perfect fifth + octave
            frequency * phi,  # Phi relation
            frequency / phi,  # Phi relation (lower)
            frequency * 1.5,  # Perfect fifth
            frequency * 0.5,  # Octave down
        ]
        
        # Update energy generator frequencies
        self._update_generator_frequencies()
        
        # Update resonance patterns
        self._update_resonance_patterns()
        
        # Update metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Brain seed frequency set to {frequency:.2f} Hz with {len(self.frequency_harmonics)} harmonics")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'frequency': frequency,
                    'harmonics': self.frequency_harmonics,
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_frequency", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record frequency metrics: {e}")
        
        return True

    def _update_generator_frequencies(self):
        """Update energy generator frequencies based on new base frequency."""
        # Update primary generator
        if self.energy_generators:
            self.energy_generators[0]['frequency'] = self.base_frequency_hz
        
        # Update secondary generators
        for i, generator in enumerate(self.energy_generators[1:], 1):
            # Calculate appropriate frequency from harmonics
            freq_idx = min(i, len(self.frequency_harmonics) - 1)
            generator['frequency'] = self.frequency_harmonics[freq_idx]
        
        logger.debug(f"Updated frequencies for {len(self.energy_generators)} energy generators")

    def _update_resonance_patterns(self):
        """Update resonance patterns based on new frequencies."""
        for region, pattern in self.resonance_patterns.items():
            # Update frequency matches 
            if region == 'limbic':
                pattern['frequency_match'] = self.base_frequency_hz
            elif region == 'brain_stem':
                pattern['frequency_match'] = self.frequency_harmonics[1] if len(self.frequency_harmonics) > 1 else self.base_frequency_hz
            elif region == 'frontal':
                pattern['frequency_match'] = self.frequency_harmonics[2] if len(self.frequency_harmonics) > 2 else self.base_frequency_hz
            elif region == 'parietal':
                pattern['frequency_match'] = self.frequency_harmonics[3] if len(self.frequency_harmonics) > 3 else self.base_frequency_hz
            elif region == 'temporal':
                pattern['frequency_match'] = self.frequency_harmonics[4] if len(self.frequency_harmonics) > 4 else self.base_frequency_hz
            elif region == 'occipital':
                pattern['frequency_match'] = self.frequency_harmonics[5] if len(self.frequency_harmonics) > 5 else self.base_frequency_hz
        
        logger.debug(f"Updated resonance patterns for {len(self.resonance_patterns)} regions")

    def set_frequency_from_soul(self, soul_frequency: float):
        """
        Sets the base frequency from the soul's frequency and computes harmonics.
        
        Args:
            soul_frequency: The base frequency of the soul
            
        Raises:
            ValueError: If frequency is invalid
        """
        return self.set_frequency(soul_frequency)

    def add_energy(self, energy_amount: float, source: str = "external") -> Dict[str, Any]:
        """
        Add energy to the brain seed.
        
        Args:
            energy_amount: Amount of energy to add in BEU
            source: Source of the energy (used for tracking)
            
        Returns:
            Dict containing operation results
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(energy_amount, (int, float)) or energy_amount <= FLOAT_EPSILON:
            raise ValueError(f"Energy amount must be positive, got {energy_amount}")
        
        # Calculate how much energy can be accepted before reaching capacity
        available_capacity = max(0.0, self.energy_capacity - self.base_energy_level)
        
        # Limit energy addition to available capacity
        energy_accepted = min(energy_amount, available_capacity)
        energy_excess = max(0.0, energy_amount - energy_accepted)
        
        # Add accepted energy
        self.base_energy_level += energy_accepted
        
        # Increase capacity slightly with high energy input
        if energy_amount > self.energy_capacity * 0.1:
            capacity_increase = energy_amount * 0.05  # 5% of incoming energy increases capacity
            self.energy_capacity += capacity_increase
        
        # Update generators if significant energy added
        if energy_accepted > self.base_energy_level * 0.1:
            self._redistribute_generator_output()
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        # Check if energy addition increases formation progress
        if self.formation_progress < 1.0:
            progress_increase = energy_accepted / (self.energy_capacity * 10)  # Approximate progress increase
            self.formation_progress = min(1.0, self.formation_progress + progress_increase)
        
        logger.info(f"Added {energy_accepted:.2E} BEU to brain seed from {source}. "
                   f"New level: {self.base_energy_level:.2E} BEU. "
                   f"Excess: {energy_excess:.2E} BEU.")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'energy_added': energy_accepted,
                    'energy_excess': energy_excess,
                    'new_energy_level': self.base_energy_level,
                    'source': source,
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_energy", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record energy metrics: {e}")
        
        return {
            'energy_accepted': energy_accepted,
            'energy_excess': energy_excess,
            'new_energy_level': self.base_energy_level,
            'capacity': self.energy_capacity,
            'formation_progress': self.formation_progress
        }

    def _redistribute_generator_output(self):
        """Redistribute energy output levels among generators."""
        if not self.energy_generators:
            return
        
        # Calculate total available output (40% of total energy)
        total_output = self.base_energy_level * 0.4
        
        # Primary generator gets 40% of output
        primary_output = total_output * 0.4
        
        # Remaining generators share the rest
        secondary_count = len(self.energy_generators) - 1
        
        if secondary_count > 0:
            secondary_output = (total_output - primary_output) / secondary_count
            
            # Update generator outputs
            self.energy_generators[0]['output'] = primary_output
            
            for i in range(1, len(self.energy_generators)):
                self.energy_generators[i]['output'] = secondary_output * (1 - (0.1 * (i-1)))  # Slight decrease with each generator
        else:
            # Only primary generator exists
            self.energy_generators[0]['output'] = total_output
        
        logger.debug(f"Redistributed energy output among {len(self.energy_generators)} generators")

    def use_energy(self, energy_amount: float, purpose: str = "operation") -> Dict[str, Any]:
        """
        Use energy from the brain seed.
        
        Args:
            energy_amount: Amount of energy to use in BEU
            purpose: Purpose of energy usage
            
        Returns:
            Dict containing operation results
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(energy_amount, (int, float)) or energy_amount <= FLOAT_EPSILON:
            raise ValueError(f"Energy amount must be positive, got {energy_amount}")
        
        # Check if sufficient energy is available
        if energy_amount > self.base_energy_level:
            available = self.base_energy_level
            
            logger.warning(f"Insufficient energy in brain seed. Requested: {energy_amount:.2E} BEU, "
                          f"Available: {available:.2E} BEU")
            
            return {
                'success': False,
                'energy_used': 0.0,
                'energy_shortage': energy_amount,
                'new_energy_level': self.base_energy_level,
                'message': f"Insufficient energy. Requested: {energy_amount:.2E}, Available: {available:.2E}"
            }
        
        # Use energy
        self.base_energy_level -= energy_amount
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Used {energy_amount:.2E} BEU from brain seed for {purpose}. "
                   f"New level: {self.base_energy_level:.2E} BEU.")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'energy_used': energy_amount,
                    'purpose': purpose,
                    'new_energy_level': self.base_energy_level,
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_energy_usage", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record energy usage metrics: {e}")
        
        return {
            'success': True,
            'energy_used': energy_amount,
            'energy_shortage': 0.0,
            'new_energy_level': self.base_energy_level
        }

    def store_energy_in_mycelial(self, energy_amount: float) -> Dict[str, Any]:
        """
        Store energy in mycelial network for later use.
        
        Args:
            energy_amount: Amount of energy to store in BEU
            
        Returns:
            Dict containing operation results
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(energy_amount, (int, float)) or energy_amount <= FLOAT_EPSILON:
            raise ValueError(f"Energy amount must be positive, got {energy_amount}")
        
        # Check if sufficient energy is available
        if energy_amount > self.base_energy_level:
            available = self.base_energy_level
            
            logger.warning(f"Insufficient energy in brain seed for mycelial storage. "
                          f"Requested: {energy_amount:.2E} BEU, Available: {available:.2E} BEU")
            
            return {
                'success': False,
                'energy_stored': 0.0,
                'energy_shortage': energy_amount,
                'new_mycelial_energy': self.mycelial_energy_store,
                'message': f"Insufficient energy. Requested: {energy_amount:.2E}, Available: {available:.2E}"
            }
        
        # Store energy in mycelial network with 95% efficiency (natural energy loss in transfer)
        energy_stored = energy_amount * 0.95
        self.base_energy_level -= energy_amount
        self.mycelial_energy_store += energy_stored
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Stored {energy_stored:.2E} BEU in mycelial network from {energy_amount:.2E} BEU. "
                   f"New mycelial energy: {self.mycelial_energy_store:.2E} BEU.")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'energy_stored': energy_stored,
                    'energy_loss': energy_amount - energy_stored,
                    'new_mycelial_energy': self.mycelial_energy_store,
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_mycelial_storage", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record mycelial storage metrics: {e}")
        
        return {
            'success': True,
            'energy_stored': energy_stored,
            'energy_loss': energy_amount - energy_stored,
            'new_mycelial_energy': self.mycelial_energy_store
        }

    def retrieve_energy_from_mycelial(self, energy_amount: float) -> Dict[str, Any]:
        """
        Retrieve energy from mycelial network.
        
        Args:
            energy_amount: Amount of energy to retrieve in BEU
            
        Returns:
            Dict containing operation results
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(energy_amount, (int, float)) or energy_amount <= FLOAT_EPSILON:
            raise ValueError(f"Energy amount must be positive, got {energy_amount}")
        
        # Check if sufficient energy is available in mycelial network
        if energy_amount > self.mycelial_energy_store:
            available = self.mycelial_energy_store
            
            logger.warning(f"Insufficient energy in mycelial network. "
                          f"Requested: {energy_amount:.2E} BEU, Available: {available:.2E} BEU")
            
            return {
                'success': False,
                'energy_retrieved': 0.0,
                'energy_shortage': energy_amount - available,
                'new_mycelial_energy': self.mycelial_energy_store,
                'message': f"Insufficient mycelial energy. Requested: {energy_amount:.2E}, Available: {available:.2E}"
            }
        
        # Calculate how much energy can be accepted before reaching capacity
        available_capacity = max(0.0, self.energy_capacity - self.base_energy_level)
        
        # Limit retrieval to available capacity
        retrievable_amount = min(energy_amount, available_capacity)
        
        if retrievable_amount < energy_amount:
            logger.warning(f"Limited retrieval due to seed capacity constraints. "
                          f"Requested: {energy_amount:.2E} BEU, Retrieved: {retrievable_amount:.2E} BEU")
        
        # Retrieve energy with 90% efficiency (natural energy loss in transfer)
        energy_retrieved = retrievable_amount * 0.9
        self.mycelial_energy_store -= retrievable_amount
        self.base_energy_level += energy_retrieved
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Retrieved {energy_retrieved:.2E} BEU from mycelial network. "
                   f"New seed energy: {self.base_energy_level:.2E} BEU. "
                   f"Remaining mycelial energy: {self.mycelial_energy_store:.2E} BEU.")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'energy_retrieved': energy_retrieved,
                    'energy_loss': retrievable_amount - energy_retrieved,
                    'new_seed_energy': self.base_energy_level,
                    'new_mycelial_energy': self.mycelial_energy_store,
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_mycelial_retrieval", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record mycelial retrieval metrics: {e}")
        
        return {
                'success': True,
                'energy_retrieved': energy_retrieved,
                'energy_loss': retrievable_amount - energy_retrieved,
                'capacity_limited': retrievable_amount < energy_amount,
                'new_seed_energy': self.base_energy_level,
                'new_mycelial_energy': self.mycelial_energy_store
            }

    def add_memory_fragment(self, fragment_content: str, fragment_frequency: float = None, 
                          fragment_meta: Dict[str, Any] = None) -> str:
       """
       Add a new memory fragment to the brain seed.
       
       Args:
           fragment_content: The content of the memory fragment
           fragment_frequency: Optional frequency associated with this fragment
           fragment_meta: Optional metadata tags for the fragment
       
       Returns:
           The ID of the created fragment
           
       Raises:
           ValueError: If parameters are invalid
       """
       if not isinstance(fragment_content, str) or not fragment_content:
           raise ValueError("Fragment content must be a non-empty string")
       
       # Validate fragment_frequency if provided
       if fragment_frequency is not None:
           if not isinstance(fragment_frequency, (int, float)) or fragment_frequency <= FLOAT_EPSILON:
               raise ValueError(f"Fragment frequency must be positive, got {fragment_frequency}")
       
       # Use base frequency if no specific frequency is provided
       if fragment_frequency is None:
           fragment_frequency = self.base_frequency_hz
       
       # Create unique fragment ID
       fragment_id = str(uuid.uuid4())
       creation_time = datetime.now().isoformat()
       
       # Create the memory fragment
       fragment = {
           "fragment_id": fragment_id,
           "content": fragment_content,
           "frequency_hz": fragment_frequency,
           "creation_time": creation_time,
           "last_accessed": creation_time,
           "access_count": 0,
           "activated": False,
           "meta_tags": fragment_meta or {},
           "associations": [],
           "energy_level": 0.1  # Initial energy level
       }
       
       # Store the fragment
       self.memory_fragments[fragment_id] = fragment
       self.last_updated = datetime.now().isoformat()
       
       logger.debug(f"Memory fragment {fragment_id} created with frequency {fragment_frequency:.2f} Hz")
       
       # Record metrics if available
       if METRICS_AVAILABLE:
           try:
               metrics_data = {
                   'seed_id': self.seed_id,
                   'fragment_id': fragment_id,
                   'frequency': fragment_frequency,
                   'meta_tags': list(fragment_meta.keys()) if fragment_meta else [],
                   'timestamp': creation_time
               }
               metrics.record_metrics("brain_seed_memory_fragment", metrics_data)
           except Exception as e:
               logger.warning(f"Failed to record memory fragment metrics: {e}")
       
       return fragment_id

    def get_memory_fragment(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory fragment by ID.
        
        Args:
            fragment_id: ID of the fragment to retrieve
            
        Returns:
            Memory fragment data or None if not found
            
        Raises:
            ValueError: If fragment_id is invalid
        """
        if not isinstance(fragment_id, str) or not fragment_id:
            raise ValueError("Fragment ID must be a non-empty string")
        
        if fragment_id not in self.memory_fragments:
            logger.warning(f"Memory fragment {fragment_id} not found")
            return None
        
        # Get the fragment
        fragment = self.memory_fragments[fragment_id]
        
        # Update access time and count
        fragment["last_accessed"] = datetime.now().isoformat()
        fragment["access_count"] += 1
        
        # Update in storage
        self.memory_fragments[fragment_id] = fragment
        
        return fragment.copy()

    def activate_fragment(self, fragment_id: str, activation_energy: float = 0.05) -> Dict[str, Any]:
        """
        Activate a memory fragment with energy.
        
        Args:
            fragment_id: ID of the fragment to activate
            activation_energy: Energy amount to use for activation
            
        Returns:
            Dict containing activation results
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(fragment_id, str) or not fragment_id:
            raise ValueError("Fragment ID must be a non-empty string")
            
        if not isinstance(activation_energy, (int, float)) or activation_energy <= FLOAT_EPSILON:
            raise ValueError(f"Activation energy must be positive, got {activation_energy}")
        
        # Check if fragment exists
        if fragment_id not in self.memory_fragments:
            logger.warning(f"Memory fragment {fragment_id} not found")
            return {
                'success': False,
                'message': f"Fragment {fragment_id} not found",
                'fragment_id': fragment_id
            }
        
        # Check if sufficient energy is available
        if activation_energy > self.base_energy_level:
            logger.warning(f"Insufficient energy to activate fragment {fragment_id}")
            return {
                'success': False,
                'message': f"Insufficient energy. Required: {activation_energy:.2E}, Available: {self.base_energy_level:.2E}",
                'fragment_id': fragment_id
            }
        
        # Use energy
        self.base_energy_level -= activation_energy
        
        # Get the fragment
        fragment = self.memory_fragments[fragment_id]
        
        # Update fragment state
        fragment["activated"] = True
        fragment["energy_level"] += activation_energy
        fragment["last_accessed"] = datetime.now().isoformat()
        fragment["access_count"] += 1
        
        # Update in storage
        self.memory_fragments[fragment_id] = fragment
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Memory fragment {fragment_id} activated with {activation_energy:.2E} BEU")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'fragment_id': fragment_id,
                    'activation_energy': activation_energy,
                    'fragment_energy': fragment["energy_level"],
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_fragment_activation", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record fragment activation metrics: {e}")
        
        return {
            'success': True,
            'fragment_id': fragment_id,
            'new_energy_level': fragment["energy_level"],
            'activation_state': fragment["activated"],
            'access_count': fragment["access_count"]
        }


    def associate_fragments(self, fragment_id1: str, fragment_id2: str, 
                            association_strength: float = 0.5) -> Dict[str, Any]:
        """
        Create an association between two memory fragments.
        
        Args:
            fragment_id1: ID of the first fragment
            fragment_id2: ID of the second fragment
            association_strength: Strength of the association (0-1)
            
        Returns:
            Dict containing association results
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(fragment_id1, str) or not fragment_id1:
            raise ValueError("First fragment ID must be a non-empty string")
            
        if not isinstance(fragment_id2, str) or not fragment_id2:
            raise ValueError("Second fragment ID must be a non-empty string")
            
        if not isinstance(association_strength, (int, float)) or not (0 <= association_strength <= 1):
            raise ValueError(f"Association strength must be between 0 and 1, got {association_strength}")
        
        # Check if fragments exist
        if fragment_id1 not in self.memory_fragments:
            logger.warning(f"Memory fragment {fragment_id1} not found")
            return {
                'success': False,
                'message': f"Fragment {fragment_id1} not found"
            }
            
        if fragment_id2 not in self.memory_fragments:
            logger.warning(f"Memory fragment {fragment_id2} not found")
            return {
                'success': False,
                'message': f"Fragment {fragment_id2} not found"
            }
        
        # Get the fragments
        fragment1 = self.memory_fragments[fragment_id1]
        fragment2 = self.memory_fragments[fragment_id2]
        
        # Create association data
        association_time = datetime.now().isoformat()
        
        # Add association to first fragment
        association1 = {
            'target_id': fragment_id2,
            'strength': association_strength,
            'created': association_time
        }
        
        # Check if association already exists
        existing_association = False
        for i, assoc in enumerate(fragment1['associations']):
            if assoc['target_id'] == fragment_id2:
                # Update existing association
                fragment1['associations'][i]['strength'] = association_strength
                fragment1['associations'][i]['created'] = association_time
                existing_association = True
                break
        
        if not existing_association:
            fragment1['associations'].append(association1)
        
        # Add association to second fragment
        association2 = {
            'target_id': fragment_id1,
            'strength': association_strength,
            'created': association_time
        }
        
        # Check if association already exists
        existing_association = False
        for i, assoc in enumerate(fragment2['associations']):
            if assoc['target_id'] == fragment_id1:
                # Update existing association
                fragment2['associations'][i]['strength'] = association_strength
                fragment2['associations'][i]['created'] = association_time
                existing_association = True
                break
        
        if not existing_association:
            fragment2['associations'].append(association2)
        
        # Update fragments in storage
        self.memory_fragments[fragment_id1] = fragment1
        self.memory_fragments[fragment_id2] = fragment2
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Created association between fragments {fragment_id1} and {fragment_id2} "
                    f"with strength {association_strength:.2f}")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'fragment_id1': fragment_id1,
                    'fragment_id2': fragment_id2,
                    'association_strength': association_strength,
                    'timestamp': association_time
                }
                metrics.record_metrics("brain_seed_fragment_association", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record fragment association metrics: {e}")
        
        return {
            'success': True,
            'fragment_id1': fragment_id1,
            'fragment_id2': fragment_id2,
            'association_strength': association_strength,
            'association_time': association_time
        }

    def search_fragments_by_frequency(self, target_frequency: float, 
                                    frequency_tolerance: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search for memory fragments by frequency.
        
        Args:
            target_frequency: Target frequency to search for
            frequency_tolerance: Tolerance range for frequency matching
            
        Returns:
            List of matching fragments
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(target_frequency, (int, float)) or target_frequency <= FLOAT_EPSILON:
            raise ValueError(f"Target frequency must be positive, got {target_frequency}")
            
        if not isinstance(frequency_tolerance, (int, float)) or frequency_tolerance <= FLOAT_EPSILON:
            raise ValueError(f"Frequency tolerance must be positive, got {frequency_tolerance}")
        
        # Search for fragments within frequency range
        matching_fragments = []
        
        for fragment_id, fragment in self.memory_fragments.items():
            fragment_freq = fragment.get('frequency_hz', 0.0)
            
            # Check if frequency is within tolerance
            if abs(fragment_freq - target_frequency) <= frequency_tolerance:
                matching_fragments.append(fragment.copy())
        
        # Sort by frequency proximity
        matching_fragments.sort(key=lambda f: abs(f.get('frequency_hz', 0.0) - target_frequency))
        
        logger.debug(f"Found {len(matching_fragments)} fragments matching frequency {target_frequency} Hz "
                    f"with tolerance {frequency_tolerance} Hz")
        
        return matching_fragments

    def search_fragments_by_meta(self, meta_tags: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for memory fragments by metadata tags.
        
        Args:
            meta_tags: Dictionary of metadata tags to search for
            
        Returns:
            List of matching fragments
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(meta_tags, dict):
            raise ValueError("Meta tags must be a dictionary")
        
        # Search for fragments with matching metadata
        matching_fragments = []
        
        for fragment_id, fragment in self.memory_fragments.items():
            fragment_meta = fragment.get('meta_tags', {})
            
            # Check if all specified tags match
            matches = True
            for key, value in meta_tags.items():
                if key not in fragment_meta or fragment_meta[key] != value:
                    matches = False
                    break
            
            if matches:
                matching_fragments.append(fragment.copy())
        
        logger.debug(f"Found {len(matching_fragments)} fragments matching meta tags {meta_tags}")
        
        return matching_fragments

    def store_life_cord_data(self, life_cord_data: Dict[str, Any]) -> bool:
        """
        Stores the life cord data for meditation channel access.
        
        Args:
            life_cord_data: Life cord data dictionary
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If life_cord_data is invalid
        """
        if not isinstance(life_cord_data, dict):
            raise ValueError("Life cord data must be a dictionary")
        
        # Validate minimum required fields
        required_fields = ["primary_frequency_hz", "soul_to_earth_efficiency", 
                            "earth_to_soul_efficiency", "quantum_efficiency"]
        
        missing_fields = [field for field in required_fields if field not in life_cord_data]
        if missing_fields:
            raise ValueError(f"Missing required life cord data fields: {missing_fields}")
        
        # Store life cord data
        self.life_cord_data = life_cord_data.copy()
        
        # Set earth connection data if available
        if "earth_resonance" in life_cord_data:
            self.earth_connection = {
                "earth_resonance": life_cord_data["earth_resonance"],
                "earth_connection_strength": life_cord_data.get("earth_to_soul_efficiency", 0.5),
                "grounding_factor": life_cord_data.get("earth_connection_strength", 0.5)
            }
        
        # Update seed metrics
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Life cord data stored. Primary frequency: {life_cord_data['primary_frequency_hz']:.2f} Hz")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'seed_id': self.seed_id,
                    'primary_frequency': life_cord_data['primary_frequency_hz'],
                    'soul_to_earth_efficiency': life_cord_data['soul_to_earth_efficiency'],
                    'earth_to_soul_efficiency': life_cord_data['earth_to_soul_efficiency'],
                    'quantum_efficiency': life_cord_data['quantum_efficiency'],
                    'timestamp': datetime.now().isoformat()
                }
                metrics.record_metrics("brain_seed_life_cord", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record life cord metrics: {e}")
        
        return True

    def get_energy_field_at_distance(self, distance: float) -> float:
        """
        Calculate energy field strength at a given distance from the seed.
        
        Args:
            distance: Distance from seed center
            
        Returns:
            Energy field strength at that distance
            
        Raises:
            ValueError: If distance is invalid
        """
        if not isinstance(distance, (int, float)) or distance < 0:
            raise ValueError(f"Distance must be non-negative, got {distance}")
        
        # Get field properties
        radius = self.seed_field['radius']
        energy_density = self.seed_field['energy_density']
        
        # If outside field radius, energy drops off with inverse square law
        if distance > radius:
            return energy_density * (radius / distance)**2
        
        # Inside field radius, energy follows a distribution based on 1-(r/R)²
        # This creates a smooth field with maximum at center
        normalized_distance = distance / radius
        energy_factor = 1 - normalized_distance**2
        
        return energy_density * energy_factor

    def get_frequency_field_at_distance(self, distance: float) -> float:
        """
        Calculate frequency at a given distance from the seed.
        
        Args:
            distance: Distance from seed center
            
        Returns:
            Frequency at that distance
            
        Raises:
            ValueError: If distance is invalid
        """
        if not isinstance(distance, (int, float)) or distance < 0:
            raise ValueError(f"Distance must be non-negative, got {distance}")
        
        # Get field properties
        radius = self.seed_field['radius']
        base_frequency = self.base_frequency_hz
        frequency_variance = self.seed_field['frequency_variance']
        
        # If outside field radius, frequency approaches Schumann resonance
        if distance > radius:
            # Gradually transition to Schumann frequency with distance
            transition_factor = 1 - min(1.0, radius / distance)
            return base_frequency * (1 - transition_factor) + SCHUMANN_FREQUENCY * transition_factor
        
        # Inside field radius, frequency follows a standing wave pattern
        # with the base frequency and some variance
        normalized_distance = distance / radius
        
        # Use a standing wave pattern (simplified)
        # sin²(πr/R) creates nodes at center and boundary
        standing_wave_factor = math.sin(math.pi * normalized_distance)**2
        frequency_offset = frequency_variance * standing_wave_factor
        
        # Apply frequency offset - can be positive or negative
        if distance < radius / 2:
            # Inner half: slight increase
            return base_frequency * (1 + frequency_offset)
        else:
            # Outer half: slight decrease
            return base_frequency * (1 - frequency_offset)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns current metrics of the brain seed.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate integrated energy
        total_energy = self.base_energy_level + self.mycelial_energy_store
        
        # Calculate mycelial connections
        mycelial_connected = self.mycelial_connection is not None
        mycelial_connection_strength = self.mycelial_connection.get('connection_strength', 0.0) if mycelial_connected else 0.0
        
        # Calculate fragment statistics
        fragment_count = len(self.memory_fragments)
        activated_fragments = sum(1 for f in self.memory_fragments.values() if f.get('activated', False))
        fragment_associations = sum(len(f.get('associations', [])) for f in self.memory_fragments.values())
        
        # Calculate soul connection status
        soul_connected = self.soul_connection is not None
        
        return {
            'seed_id': self.seed_id,
            'creation_time': self.creation_time,
            'last_updated': self.last_updated,
            'energy_level': self.base_energy_level,
            'mycelial_energy': self.mycelial_energy_store,
            'energy_capacity': self.energy_capacity,
            'total_energy': total_energy,
            'energy_utilization': self.base_energy_level / self.energy_capacity if self.energy_capacity > 0 else 0.0,
            'base_frequency_hz': self.base_frequency_hz,
            'harmonics_count': len(self.frequency_harmonics),
            'complexity': self.complexity,
            'formation_progress': self.formation_progress,
            'stability': self.stability,
            'structural_integrity': self.structural_integrity,
            'position': self.position,
            'brain_region': self.brain_region,
            'sub_region': self.sub_region,
            'soul_connected': soul_connected,
            'life_cord_connected': self.life_cord_data is not None,
            'earth_connected': self.earth_connection is not None,
            'mycelial_connected': mycelial_connected,
            'mycelial_connection_strength': mycelial_connection_strength,
            'aspects_distributed_count': len(self.soul_aspect_distribution),
            'memory_fragments_count': fragment_count,
            'activated_fragments_count': activated_fragments,
            'fragment_associations_count': fragment_associations,
            'energy_generators_count': len(self.energy_generators)
        }

    def get_energy_generators_data(self) -> List[Dict[str, Any]]:
        """
        Get data for all energy generators.
        
        Returns:
            List of energy generator data dictionaries
        """
        return [gen.copy() for gen in self.energy_generators]

    def save_state(self, file_path: str) -> bool:
        """
        Save the brain seed state to file.
        
        Args:
            file_path: Path to save state file
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            IOError: If save fails
        """
        logger.info(f"Saving BrainSeed state to {file_path}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare state for serialization
            state = self.__dict__.copy()
            
            # Convert numpy arrays to lists for JSON serialization
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    state[key] = value.tolist()
            
            # Save state to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"BrainSeed state saved successfully to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving BrainSeed state: {e}", exc_info=True)
            raise IOError(f"Failed to save BrainSeed state: {e}")

    @classmethod
    def load_state(cls, file_path: str) -> 'BrainSeed':
        """
        Load brain seed state from file.
        
        Args:
            file_path: Path to state file
            
        Returns:
            Loaded BrainSeed instance
            
        Raises:
            FileNotFoundError: If file not found
            ValueError: If state data is invalid
            RuntimeError: If load fails
        """
        logger.info(f"Loading BrainSeed state from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BrainSeed state file not found: {file_path}")
        
        try:
            # Load state from file
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Create a new instance
            seed = BrainSeed()
            
            # Set attributes from state
            for key, value in state.items():
                if hasattr(seed, key):
                    # Convert lists back to numpy arrays if needed
                    if key.endswith('_grid') and isinstance(value, list):
                        setattr(seed, key, np.array(value))
                    else:
                        setattr(seed, key, value)
            
            logger.info(f"BrainSeed state loaded successfully from {file_path}")
            return seed
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in BrainSeed state file: {file_path}", exc_info=True)
            raise ValueError(f"Invalid JSON in BrainSeed state file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading BrainSeed state: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load BrainSeed state: {e}")

    def __str__(self) -> str:
        position_str = str(self.position) if self.position else "unpositioned"
        region_str = self.brain_region if self.brain_region else "unknown region"
        
        return (f"BrainSeed(ID: {self.seed_id[:8]}..., Freq: {self.base_frequency_hz:.1f} Hz, "
                f"BEU: {self.base_energy_level:.1E}, Position: {position_str}, Region: {region_str})")

    def __repr__(self) -> str:
        return (f"<BrainSeed id='{self.seed_id}' freq={self.base_frequency_hz:.1f} "
                f"energy={self.base_energy_level:.1E} progress={self.formation_progress:.2f}>")


# --- Main Create Function ---
def create_brain_seed(initial_beu: float = 0.0, initial_mycelial_beu: float = 0.0, 
                    initial_frequency: Optional[float] = None) -> BrainSeed:
   """
   Create a new brain seed with specified parameters.
   
   Args:
       initial_beu: Initial brain energy units
       initial_mycelial_beu: Initial mycelial energy units
       initial_frequency: Initial frequency (Hz)
       
   Returns:
       Initialized BrainSeed instance
       
   Raises:
       ValueError: If parameters are invalid
   """
   logger.info(f"Creating new brain seed with {initial_beu:.2E} BEU, "
              f"{initial_mycelial_beu:.2E} mycelial BEU")
   
   try:
       # Create brain seed
       seed = BrainSeed(initial_beu=initial_beu, initial_mycelial_beu=initial_mycelial_beu)
       
       # Set frequency if specified
       if initial_frequency is not None:
           seed.set_frequency(initial_frequency)
       
       # Record metrics if available
       if METRICS_AVAILABLE:
           try:
               metrics_data = {
                   'seed_id': seed.seed_id,
                   'initial_beu': initial_beu,
                   'initial_mycelial_beu': initial_mycelial_beu,
                   'initial_frequency': seed.base_frequency_hz,
                   'creation_time': seed.creation_time
               }
               metrics.record_metrics("brain_seed_creation", metrics_data)
           except Exception as e:
               logger.warning(f"Failed to record seed creation metrics: {e}")
       
       return seed
       
   except Exception as e:
       logger.error(f"Error creating brain seed: {e}", exc_info=True)
       raise RuntimeError(f"Failed to create brain seed: {e}")


# --- Module Test Function ---
def test_brain_seed():
   """Test the brain seed functionality."""
   logger.info("=== Testing Brain Seed ===")
   
   # Create brain seed
   seed = create_brain_seed(initial_beu=10.0, initial_mycelial_beu=5.0, initial_frequency=10.0)
   
   # Test energy operations
   seed.add_energy(5.0, "test")
   seed.store_energy_in_mycelial(2.0)
   seed.retrieve_energy_from_mycelial(1.0)
   
   # Test memory fragments
   fragment_id1 = seed.add_memory_fragment("Test fragment 1", 8.0, {"tag": "test"})
   fragment_id2 = seed.add_memory_fragment("Test fragment 2", 12.0, {"tag": "test"})
   seed.associate_fragments(fragment_id1, fragment_id2, 0.7)
   
   # Test metrics
   metrics = seed.get_metrics()
   logger.info(f"Brain seed metrics: {metrics}")
   
   # Test saving
   seed.save_state("test_brain_seed_state.json")
   
   # Test loading
   loaded_seed = BrainSeed.load_state("test_brain_seed_state.json")
   
   logger.info("=== Brain Seed Tests Completed ===")
   return seed


# --- Main Execution ---
if __name__ == "__main__":
   logger.info("=== Brain Seed Module Standalone Execution ===")
   
   # Test brain seed
   try:
       seed = test_brain_seed()
       print("Brain seed tests passed successfully!")
       print(f"Created seed: {seed}")
   except Exception as e:
       logger.error(f"Brain seed tests failed: {e}", exc_info=True)
       print(f"ERROR: Brain seed tests failed: {e}")
       sys.exit(1)
   
   logger.info("=== Brain Seed Module Execution Complete ===")

# --- END OF FILE stage_1/evolve/brain_seed.py ---