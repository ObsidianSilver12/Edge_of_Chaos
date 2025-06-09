# --- brain_seed.py V8 ---

"""
Brain seed - energy spark that triggers brain development.
Like sperm+egg - provides initial energy burst to start growth.
the brain seed is saved and flag is set to indicate that the brain seed has been created.
thiswill trigger the brain formation process.
"""

import logging
import uuid
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import math

# Import constants
from constants.constants import *


# --- Logging Setup ---
logger = logging.getLogger("Conception")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class BrainSeed:
    """
    Brain seed - energy spark that triggers brain development.
    Like sperm+egg - provides initial energy burst to start growth.
    """
    def __init__(self, dimensions: Tuple[int, int, int] = GRID_DIMENSIONS):
        """Initialize conception system."""
        self.conception_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.dimensions = dimensions
        
        # Seed state
        self.brain_seed = None
        self.is_active = True
        self.seed_energy = 0.0
        self.seed_position = None
        
        # Womb environment
        self.womb_environment = None
        self.mother_resonance_active = False
        
        # Energy tracking
        self.creator_energy_added = 0.0
        self.mycelial_storage_energy = 0.0
        
        logger.info(f"Conception system initialized: {self.conception_id[:8]}")

    def create_brain_seed(self) -> Dict[str, Any]:
        """
        Create a brain seed at the edge of chaos.
        Simple energy spark to trigger brain development.
        """
        logger.info("Creating brain seed at edge of chaos")
        
        try:
            # Calculate edge of chaos parameters
            chaos_frequency = 432.0 * math.sqrt(2)  # Sacred frequency at chaos edge
            chaos_energy = random.uniform(0.618, 1.618)  # Golden ratio range
            
            # Create seed with minimal complexity
            self.brain_seed = {
                'seed_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'type': 'brain_seed',
                'frequency': chaos_frequency,
                'energy': chaos_energy,
                'active': True,
                'edge_of_chaos': True,
                'position': None,  # Will be set when placed
                'merkaba_enhanced': False,
                'creator_energy': 0.0,
                'phi_resonance': PHI,
                'sacred_geometry': 'seed_of_life'
            }
            
            self.seed_energy = chaos_energy
            self.is_active = True
            
            seed_metrics = {
                'success': True,
                'seed_id': self.brain_seed['seed_id'],
                'chaos_frequency': chaos_frequency,
                'chaos_energy': chaos_energy,
                'phi_resonance': PHI,
                'edge_of_chaos': True
            }
            
            logger.info(f"Brain seed created: {chaos_frequency:.2f}Hz, {chaos_energy:.3f} energy")
            
            return seed_metrics
            
        except Exception as e:
            logger.error(f"Failed to create brain seed: {e}")
            return {'success': False, 'error': str(e)}    

    def add_creator_energy(self, energy_amount: float = None) -> Dict[str, Any]:
        """
        Add pure energy from creator to brain seed.
        """
        if not self.brain_seed:
            return {'success': False, 'reason': 'no_brain_seed'}
        
        try:
            # Default energy amount if not specified
            if energy_amount is None:
                energy_amount = random.uniform(2.0, 5.0)  # Creator energy range
            
            # Add creator energy to seed
            self.brain_seed['creator_energy'] += energy_amount
            self.brain_seed['energy'] += energy_amount
            self.creator_energy_added += energy_amount
            
            # Enhance frequency with creator energy
            frequency_boost = energy_amount * 10.0  # 10Hz per energy unit
            self.brain_seed['frequency'] += frequency_boost
            
            # Mark as creator-enhanced
            self.brain_seed['creator_enhanced'] = True
            
            creator_metrics = {
                'success': True,
                'energy_added': energy_amount,
                'total_creator_energy': self.brain_seed['creator_energy'],
                'new_seed_energy': self.brain_seed['energy'],
                'frequency_boost': frequency_boost,
                'new_frequency': self.brain_seed['frequency']
            }
            
            logger.info(f"Creator energy added: {energy_amount:.2f}, total: {self.brain_seed['creator_energy']:.2f}")
            
            return creator_metrics
            
        except Exception as e:
            logger.error(f"Failed to add creator energy: {e}")
            return {'success': False, 'error': str(e)}
    
    def find_edge_of_chaos_locations(self) -> List[Tuple[int, int, int]]:
        """Find positions at edge of chaos for optimal mycelial storage."""
        try:
            chaos_positions = []
            
            # Search around brain seed position for edge of chaos
            search_radius = 20
            for dx in range(-search_radius, search_radius + 1, 5):
                for dy in range(-search_radius, search_radius + 1, 5):
                    for dz in range(-search_radius, search_radius + 1, 5):
                        x = self.seed_position[0] + dx
                        y = self.seed_position[1] + dy
                        z = self.seed_position[2] + dz
                        
                        # Check bounds
                        if (0 <= x < self.dimensions[0] and
                            0 <= y < self.dimensions[1] and
                            0 <= z < self.dimensions[2]):
                            
                            # Calculate chaos metric (distance from center + randomness)
                            center_dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            randomness = random.random()
                            chaos_metric = center_dist * randomness
                            
                            # Edge of chaos is moderate chaos metric
                            if 0.3 <= chaos_metric <= 0.7:
                                chaos_positions.append((x, y, z))
            
            # Sort by chaos metric (best edge of chaos first)
            return chaos_positions[:5]  # Return top 5 positions
            
        except Exception as e:
            logger.warning(f"Edge of chaos detection failed: {e}")
            return []
        
    def place_brain_seed(self, position: Tuple[int, int, int] = None) -> Dict[str, Any]:
        """
        Place brain seed in womb environment at edge of chaos where it will spark brain growth.
        Returns a dictionary with details about the placement process. Saves data
        to the brain seed dictionary. Updates flag status to BRAIN_SEED_PLACED
        """
        if not self.brain_seed:
            logger.error("Cannot place brain seed: no seed created")
            return {'success': False, 'reason': 'no_brain_seed'}
        
        try:
            # Calculate optimal position if not provided
            if position is None:
                # Place at center of brain area with some variance
                center_x = self.dimensions[0] // 2
                center_y = self.dimensions[1] // 2  
                center_z = self.dimensions[2] // 2
                
                # Add variance for uniqueness (±10% of dimension)
                variance_x = random.randint(-self.dimensions[0]//10, self.dimensions[0]//10)
                variance_y = random.randint(-self.dimensions[1]//10, self.dimensions[1]//10)
                variance_z = random.randint(-self.dimensions[2]//10, self.dimensions[2]//10)
                
                position = (
                    max(26, min(self.dimensions[0]-26, center_x + variance_x)),  # Stay within brain area
                    max(26, min(self.dimensions[1]-26, center_y + variance_y)),
                    max(26, min(self.dimensions[2]-26, center_z + variance_z))
                )
            
            # Validate position is within brain area (not in external buffer)
            if (position[0] < 26 or position[0] >= self.dimensions[0]-26 or
                position[1] < 26 or position[1] >= self.dimensions[1]-26 or
                position[2] < 26 or position[2] >= self.dimensions[2]-26):
                raise ValueError(f"Position {position} is outside brain area")
            
            # Calculate edge of chaos metric for this position
            center = (self.dimensions[0]//2, self.dimensions[1]//2, self.dimensions[2]//2)
            distance_from_center = np.sqrt(sum((position[i] - center[i])**2 for i in range(3)))
            edge_chaos_metric = distance_from_center / (self.dimensions[0]//4)  # Normalize
            
            # Update brain seed with position
            self.brain_seed['position'] = position
            self.brain_seed['placement_time'] = datetime.now().isoformat()
            self.brain_seed['edge_chaos_metric'] = edge_chaos_metric
            self.brain_seed['placed'] = True
            
            # Store position for later use
            self.seed_position = position
            
            # Set flag
            setattr(self, FLAG_BRAIN_SEED_PLACED, True)
            
            placement_metrics = {
                'success': True,
                'position': position,
                'edge_chaos_metric': edge_chaos_metric,
                'distance_from_center': distance_from_center,
                'placement_time': self.brain_seed['placement_time'],
                'flag_set': FLAG_BRAIN_SEED_PLACED
            }
            
            logger.info(f"Brain seed placed at {position}, edge chaos metric: {edge_chaos_metric:.3f}")
            
            return placement_metrics
            
        except Exception as e:
            logger.error(f"Failed to place brain seed: {e}")
            return {'success': False, 'error': str(e)}

    def strengthen_brain_seed(self) -> Dict[str, Any]:
        """
        Strengthen brain seed energy dramatically with merkaba and golden ratios.
        Returns a dictionary with details about the strengthening process. Saves data
        to the brain seed dictionary. Updates the brain seed dictionary with the new energy level.
        Updates flag status to BRAIN_SEED_READY
        """
        if not self.brain_seed:
            return {'success': False, 'reason': 'no_brain_seed'}
        
        try:
            initial_energy = self.brain_seed['energy']
            if initial_energy < MIN_BRAIN_SEED_ENERGY:
                return {'success': False, 'reason': 'insufficient_energy'}
            
            # Apply merkaba enhancement
            merkaba_energy = initial_energy * 1.618  # Golden ratio multiplier
            self.brain_seed['energy'] += merkaba_energy
            self.brain_seed['merkaba_enhanced'] = True
            
            # Apply golden ratio frequencies
            base_freq = self.brain_seed['frequency']
            phi_freq = base_freq * 1.618
            self.brain_seed['frequency'] = phi_freq
            self.brain_seed['phi_enhanced'] = True
            
            # Apply sacred geometry boost
            geometry_boost = initial_energy * 0.618  # Minor golden ratio
            self.brain_seed['energy'] += geometry_boost
            self.brain_seed['sacred_geometry'] = 'flower_of_life'
            
            # Calculate total strengthening
            total_energy_gained = merkaba_energy + geometry_boost
            final_energy = self.brain_seed['energy']
            strengthening_factor = final_energy / initial_energy
            
            # Update seed state
            self.brain_seed['strengthened'] = True
            self.brain_seed['strengthening_time'] = datetime.now().isoformat()
            self.seed_energy = final_energy
            
            # Set flag
            setattr(self, FLAG_BRAIN_SEED_READY, True)
            
            strengthening_metrics = {
                'success': True,
                'initial_energy': initial_energy,
                'merkaba_energy_added': merkaba_energy,
                'geometry_boost': geometry_boost,
                'total_energy_gained': total_energy_gained,
                'final_energy': final_energy,
                'strengthening_factor': strengthening_factor,
                'phi_frequency': phi_freq,
                'enhancements': ['merkaba', 'golden_ratios', 'sacred_geometry'],
                'flag_set': FLAG_BRAIN_SEED_READY
            }
            
            logger.info(f"Brain seed strengthened: {strengthening_factor:.2f}x energy boost")
            
            return strengthening_metrics
            
        except Exception as e:
            logger.error(f"Failed to strengthen brain seed: {e}")
            return {'success': False, 'error': str(e)}     
        
    def save_brain_seed(self) -> Dict[str, Any]:
        """
        Save the brain seed to the brain seed dictionary. 
        Updates flag status to BRAIN_SEED_SAVED
        """
        if not self.brain_seed:
            logger.error("Cannot save brain seed: no seed created")
            return {'success': False, 'reason': 'no_brain_seed'}
        
        try:
            # Validate seed is ready for saving
            if not self.brain_seed.get('placed', False):
                return {'success': False, 'reason': 'seed_not_placed'}
            
            if not self.brain_seed.get('strengthened', False):
                return {'success': False, 'reason': 'seed_not_strengthened'}
            
            # Add final save timestamp
            self.brain_seed['saved_time'] = datetime.now().isoformat()
            self.brain_seed['saved'] = True
            
            # Calculate final seed state
            seed_integrity = (
                (1.0 if self.brain_seed.get('merkaba_enhanced', False) else 0.7) *
                (1.0 if self.brain_seed.get('phi_enhanced', False) else 0.8) *
                (1.0 if self.brain_seed.get('creator_enhanced', False) else 0.9)
            )
            
            self.brain_seed['seed_integrity'] = seed_integrity
            
            # Set flag to trigger brain development
            setattr(self, FLAG_BRAIN_SEED_SAVED, True)
            
            save_metrics = {
                'success': True,
                'seed_id': self.brain_seed['seed_id'],
                'position': self.brain_seed['position'],
                'final_energy': self.brain_seed['energy'],
                'final_frequency': self.brain_seed['frequency'],
                'seed_integrity': seed_integrity,
                'save_time': self.brain_seed['saved_time'],
                'flag_set': FLAG_BRAIN_SEED_SAVED
            }
            
            logger.info(f"Brain seed saved successfully. Integrity: {seed_integrity:.3f}")
            
            return save_metrics
            
        except Exception as e:
            logger.error(f"Failed to save brain seed: {e}")
            return {'success': False, 'error': str(e)}
