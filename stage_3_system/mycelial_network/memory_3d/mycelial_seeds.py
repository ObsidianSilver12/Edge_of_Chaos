# mycelial_seeds.py V9 - FIXED WITH ENTANGLEMENT
"""
Mycelial Seeds System - Energy Towers, Quantum Communication & Field Modulation
Manages mycelial seed operations after brain formation. Includes proper entanglement
energy usage and integration with stage_3_system energy operations.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import uuid
import random
import numpy as np

# Import constants and stage_3_system integration
from shared.constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("MycelialSeeds")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MycelialSeedsSystem:
    """
    Mycelial Seeds System for Stage 1 Formation and Stage 3 Operations.
    
    Stage 1 Responsibilities:
    - Load seeds created during brain formation
    - Activate seeds for entanglement processes
    - Handle quantum communication channel establishment
    
    Stage 3 Integration:
    - Coordinate with stage_3_system/mycelial_network/energy/energy_system.py
    - Provide field modulation capabilities
    - Manage ongoing seed operations
    """
    
    def __init__(self, brain_structure_reference: Dict[str, Any] = None):
        """Initialize mycelial seeds system with brain structure reference."""
        self.system_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure_ref = brain_structure_reference
        
        # --- Seed State Management ---
        self.active_seeds = {}              # Currently active seeds
        self.dormant_seeds = {}             # Seeds in dormant state
        self.entangled_seeds = {}           # Seeds participating in entanglement
        
        # --- Communication and Field Modulation ---
        self.communication_channels = {}    # Open quantum channels
        self.field_modulations = {}         # Active field modulations
        self.entanglement_pairs = {}        # Seed-to-seed entanglement mapping
        
        # --- Energy Tracking ---
        self.seed_energy_states = {}        # Track energy levels per seed
        self.energy_transfer_history = []   # History of energy transfers
        self.entanglement_energy_used = 0.0 # Total energy used for entanglement
        
        # --- Operational Metrics ---
        self.metrics = {
            'total_seeds': 0,
            'active_seeds_count': 0,
            'dormant_seeds_count': 0,
            'entangled_seeds_count': 0,
            'communication_channels_open': 0,
            'field_modulations_active': 0,
            'total_activations': 0,
            'total_energy_used': 0.0,
            'entanglement_energy_used': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        
        # --- Field Modulation Capabilities ---
        self.field_modulation_types = {
            'healing': {'frequency_range': (40.0, 100.0), 'energy_multiplier': 1.5},
            'amplification': {'frequency_range': (7.83, 40.0), 'energy_multiplier': 2.0},
            'disruption': {'frequency_range': (100.0, 200.0), 'energy_multiplier': 1.8},
            'chaos_induction': {'frequency_range': (200.0, 400.0), 'energy_multiplier': 2.5}
        }
        
        logger.info(f"🌱 Mycelial seeds system initialized: {self.system_id[:8]}")
    
    def load_seeds_from_brain_structure(self) -> Dict[str, Any]:
        """Load all mycelial seeds from brain structure (created during brain formation)."""
        logger.info("🌱 Loading mycelial seeds from brain structure...")
        
        try:
            if not self.brain_structure_ref:
                raise RuntimeError("Brain structure reference not provided")
            
            seeds_loaded = 0
            
            # Load seeds from brain structure's mycelial_seeds collection
            brain_seeds = self.brain_structure_ref.get('mycelial_seeds', {})
            
            if not brain_seeds:
                logger.warning("No mycelial seeds found in brain structure")
                return {'seeds_loaded': 0, 'reason': 'no_seeds_in_brain'}
            
            for seed_id, seed_data in brain_seeds.items():
                # Initialize seed in dormant state with proper frequency
                dormant_seed = {
                    'seed_id': seed_id,
                    'coordinates': seed_data['coordinates'],
                    'region': seed_data['region'],
                    'sub_region': seed_data['sub_region'],
                    'hierarchical_name': seed_data['hierarchical_name'],
                    
                    # Frequency management
                    'seed_frequency': MYCELIAL_SEED_FREQUENCY_BASE,
                    'status_frequency': MYCELIAL_SEED_FREQUENCY_DORMANT,
                    'status': 'dormant',
                    
                    # Energy state
                    'energy_level': 0.0,
                    'energy_storage_capacity': seed_data.get('energy_storage_capacity', 
                                                           SYNAPSE_ENERGY_JOULES * MYCELIAL_SEED_BASE_ENERGY_MULTIPLIER),
                    
                    # Operational state
                    'active': False,
                    'quantum_entangled': False,
                    'field_modulation_active': False,
                    'communication_channels': [],
                    
                    # History
                    'creation_time': seed_data.get('creation_time'),
                    'last_activated': None,
                    'activation_count': 0,
                    'entanglement_count': 0
                }
                
                self.dormant_seeds[seed_id] = dormant_seed
                seeds_loaded += 1
            
            # Update metrics
            self.metrics['total_seeds'] = seeds_loaded
            self.metrics['dormant_seeds_count'] = seeds_loaded
            self._update_metrics()
            
            logger.info(f"✅ Loaded {seeds_loaded} mycelial seeds from brain structure")
            
            return {
                'seeds_loaded': seeds_loaded,
                'dormant_seeds': len(self.dormant_seeds),
                'load_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to load seeds from brain structure: {e}")
            raise RuntimeError(f"Seed loading failed: {e}") from e
    
    def activate_seed_for_entanglement(self, seed_id: str, entanglement_target: str = None, 
                                     entanglement_type: str = 'quantum_communication') -> Dict[str, Any]:
        """
        Activate mycelial seed for entanglement processes with proper energy usage.
        
        Args:
            seed_id: ID of seed to activate
            entanglement_target: Target for entanglement (another seed ID, brain node, etc.)
            entanglement_type: Type of entanglement ('quantum_communication', 'mirror_grid', 'soul_connection')
        
        Returns:
            Entanglement activation details
        """
        logger.info(f"🔗 Activating seed {seed_id[:8]} for entanglement - type: {entanglement_type}")
        
        try:
            if seed_id not in self.dormant_seeds:
                raise ValueError(f"Seed {seed_id} not found in dormant seeds or already active")
            
            # Calculate entanglement energy requirements
            base_energy = SYNAPSE_ENERGY_JOULES * MYCELIAL_SEED_BASE_ENERGY_MULTIPLIER * SEU_PER_JOULE
            
            entanglement_multipliers = {
                'quantum_communication': MYCELIAL_SEED_COMMUNICATION_BOOST,
                'mirror_grid': MYCELIAL_SEED_TRANSFER_BOOST,
                'soul_connection': MYCELIAL_SEED_FIELD_MODULATION_BOOST,
                'brain_node_link': MYCELIAL_SEED_BASE_ENERGY_MULTIPLIER * 1.5
            }
            
            energy_multiplier = entanglement_multipliers.get(entanglement_type, 1.0)
            entanglement_energy = base_energy * energy_multiplier
            
            # Move seed from dormant to active with entanglement
            seed_data = self.dormant_seeds.pop(seed_id)
            
            # Update seed for entanglement
            seed_data.update({
                'status': 'active',
                'status_frequency': MYCELIAL_SEED_FREQUENCY_ACTIVE,
                'active': True,
                'quantum_entangled': True,
                'entanglement_type': entanglement_type,
                'entanglement_target': entanglement_target,
                'energy_level': entanglement_energy,
                'activation_time': datetime.now().isoformat(),
                'activation_count': seed_data['activation_count'] + 1,
                'entanglement_count': seed_data['entanglement_count'] + 1,
                'last_activated': datetime.now().isoformat()
            })
            
            # Store as active and entangled seed
            self.active_seeds[seed_id] = seed_data
            self.entangled_seeds[seed_id] = seed_data
            
            # Create entanglement mapping if target specified
            if entanglement_target:
                entanglement_id = str(uuid.uuid4())
                entanglement_data = {
                    'entanglement_id': entanglement_id,
                    'seed_id': seed_id,
                    'target_id': entanglement_target,
                    'entanglement_type': entanglement_type,
                    'established_time': datetime.now().isoformat(),
                    'energy_used': entanglement_energy,
                    'quantum_state': 'entangled',
                    'coherence_level': 1.0,
                    'communication_active': True
                }
                
                self.entanglement_pairs[entanglement_id] = entanglement_data
                seed_data['entanglement_id'] = entanglement_id
            
            # Track energy usage
            self.seed_energy_states[seed_id] = {
                'current_energy': entanglement_energy,
                'initial_energy': entanglement_energy,
                'usage_rate': 0.05,  # SEU per minute for entanglement maintenance
                'last_updated': datetime.now().isoformat(),
                'entanglement_type': entanglement_type
            }
            
            # Update energy tracking
            self.entanglement_energy_used += entanglement_energy
            self.energy_transfer_history.append({
                'transfer_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'action': 'entanglement_activation',
                'seed_id': seed_id,
                'energy_amount': entanglement_energy,
                'entanglement_type': entanglement_type,
                'target': entanglement_target
            })
            
            # Update metrics
            self.metrics['active_seeds_count'] += 1
            self.metrics['dormant_seeds_count'] -= 1
            self.metrics['entangled_seeds_count'] += 1
            self.metrics['total_activations'] += 1
            self.metrics['total_energy_used'] += entanglement_energy
            self.metrics['entanglement_energy_used'] += entanglement_energy
            self._update_metrics()
            
            activation_result = {
                'success': True,
                'seed_id': seed_id,
                'entanglement_type': entanglement_type,
                'energy_used': entanglement_energy,
                'entanglement_target': entanglement_target,
                'entanglement_id': entanglement_data.get('entanglement_id') if entanglement_target else None,
                'activation_time': seed_data['activation_time']
            }
            
            logger.info(f"✅ Seed {seed_id[:8]} activated for {entanglement_type}")
            logger.info(f"   Energy used: {entanglement_energy:.1f} SEU")
            if entanglement_target:
                logger.info(f"   Entangled with: {entanglement_target[:8]}")
            
            return activation_result
            
        except Exception as e:
            logger.error(f"Failed to activate seed for entanglement: {e}")
            raise RuntimeError(f"Seed entanglement activation failed: {seed_id}") from e
    
    def activate_seed_for_communication(self, seed_id: str, target_coordinates: Tuple[int, int, int] = None) -> Dict[str, Any]:
        """Activate mycelial seed for quantum communication channel."""
        logger.info(f"📡 Activating seed {seed_id[:8]} for communication...")
        
        try:
            if seed_id not in self.dormant_seeds:
                raise ValueError(f"Seed {seed_id} not found in dormant seeds")
            
            # Use entanglement activation with communication type
            activation_result = self.activate_seed_for_entanglement(
                seed_id, 
                f"coords_{target_coordinates[0]}_{target_coordinates[1]}_{target_coordinates[2]}" if target_coordinates else None,
                'quantum_communication'
            )
            
            # Create communication channel
            channel_id = str(uuid.uuid4())
            communication_channel = {
                'channel_id': channel_id,
                'seed_id': seed_id,
                'seed_coordinates': self.active_seeds[seed_id]['coordinates'],
                'target_coordinates': target_coordinates,
                'frequency': self.active_seeds[seed_id]['seed_frequency'],
                'established_time': datetime.now().isoformat(),
                'channel_strength': 1.0,
                'quantum_entangled': True,
                'status': 'open',
                'data_transmitted': 0,
                'last_transmission': None
            }
            
            self.communication_channels[channel_id] = communication_channel
            self.active_seeds[seed_id]['communication_channels'].append(channel_id)
            
            # Update metrics
            self.metrics['communication_channels_open'] += 1
            self._update_metrics()
            
            activation_result['communication_channel_id'] = channel_id
            
            logger.info(f"✅ Communication channel {channel_id[:8]} established")
            
            return activation_result
            
        except Exception as e:
            logger.error(f"Failed to activate seed for communication: {e}")
            raise RuntimeError(f"Communication activation failed: {seed_id}") from e
    
    def activate_field_modulation(self, seed_id: str, modulation_type: str, 
                                 target_region: str = None, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Activate field modulation capabilities for a seed.
        
        Args:
            seed_id: Seed to use for modulation
            modulation_type: 'healing', 'amplification', 'disruption', 'chaos_induction'
            target_region: Specific region to target
            intensity: Modulation intensity (0.1 to 2.0)
        """
        logger.info(f"🌊 Activating field modulation: {modulation_type} - seed {seed_id[:8]}")
        
        try:
            if modulation_type not in self.field_modulation_types:
                raise ValueError(f"Invalid modulation type: {modulation_type}")
            
            if seed_id not in self.active_seeds:
                # Activate seed first if dormant
                if seed_id in self.dormant_seeds:
                    self.activate_seed_for_entanglement(seed_id, None, 'field_modulation')
                else:
                    raise ValueError(f"Seed {seed_id} not found")
            
            modulation_config = self.field_modulation_types[modulation_type]
            
            # Calculate modulation energy
            base_energy = SYNAPSE_ENERGY_JOULES * MYCELIAL_SEED_FIELD_MODULATION_BOOST * SEU_PER_JOULE
            modulation_energy = base_energy * modulation_config['energy_multiplier'] * intensity
            
            # Create field modulation
            modulation_id = str(uuid.uuid4())
            field_modulation = {
                'modulation_id': modulation_id,
                'seed_id': seed_id,
                'modulation_type': modulation_type,
                'target_region': target_region,
                'intensity': intensity,
                'frequency_range': modulation_config['frequency_range'],
                'energy_used': modulation_energy,
                'start_time': datetime.now().isoformat(),
                'status': 'active',
                'effects_generated': 0,
                'last_effect_time': None
            }
            
            self.field_modulations[modulation_id] = field_modulation
            
            # Update seed state
            seed_data = self.active_seeds[seed_id]
            seed_data['field_modulation_active'] = True
            seed_data['current_modulation_id'] = modulation_id
            seed_data['energy_level'] += modulation_energy
            
            # Track energy
            if seed_id in self.seed_energy_states:
                self.seed_energy_states[seed_id]['current_energy'] += modulation_energy
            
            self.energy_transfer_history.append({
                'transfer_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'action': 'field_modulation_activation',
                'seed_id': seed_id,
                'energy_amount': modulation_energy,
                'modulation_type': modulation_type,
                'intensity': intensity
            })
            
            # Update metrics
            self.metrics['field_modulations_active'] += 1
            self.metrics['total_energy_used'] += modulation_energy
            self._update_metrics()
            
            logger.info(f"✅ Field modulation {modulation_type} activated")
            logger.info(f"   Energy: {modulation_energy:.1f} SEU, Intensity: {intensity}")
            
            return {
                'success': True,
                'modulation_id': modulation_id,
                'modulation_type': modulation_type,
                'energy_used': modulation_energy,
                'frequency_range': modulation_config['frequency_range'],
                'intensity': intensity
            }
            
        except Exception as e:
            logger.error(f"Failed to activate field modulation: {e}")
            raise RuntimeError(f"Field modulation activation failed: {seed_id}") from e
    
    def deactivate_seed(self, seed_id: str, return_energy: bool = True) -> Dict[str, Any]:
        """Deactivate a seed and optionally return energy to storage."""
        logger.info(f"💤 Deactivating seed {seed_id[:8]}")
        
        try:
            if seed_id not in self.active_seeds:
                raise ValueError(f"Seed {seed_id} not found in active seeds")
            
            seed_data = self.active_seeds.pop(seed_id)
            
            # Calculate energy to return
            energy_to_return = 0.0
            if return_energy and seed_id in self.seed_energy_states:
                energy_state = self.seed_energy_states[seed_id]
                energy_to_return = energy_state['current_energy'] * 0.8  # 80% recovery rate
            
            # Close communication channels
            channels_closed = 0
            for channel_id in seed_data.get('communication_channels', []):
                if channel_id in self.communication_channels:
                    self.communication_channels[channel_id]['status'] = 'closed'
                    channels_closed += 1
            
            # Deactivate field modulation
            if seed_data.get('field_modulation_active'):
                modulation_id = seed_data.get('current_modulation_id')
                if modulation_id and modulation_id in self.field_modulations:
                    self.field_modulations[modulation_id]['status'] = 'inactive'
                    self.metrics['field_modulations_active'] -= 1
            
            # Update seed state and move to dormant
            seed_data.update({
                'status': 'dormant',
                'status_frequency': MYCELIAL_SEED_FREQUENCY_DORMANT,
                'active': False,
                'quantum_entangled': False,
                'field_modulation_active': False,
                'energy_level': 0.0,
                'deactivation_time': datetime.now().isoformat()
            })
            
            self.dormant_seeds[seed_id] = seed_data
            
            # Remove from entangled seeds
            if seed_id in self.entangled_seeds:
                del self.entangled_seeds[seed_id]
            
            # Clean up energy state
            if seed_id in self.seed_energy_states:
                del self.seed_energy_states[seed_id]
            
            # Update metrics
            self.metrics['active_seeds_count'] -= 1
            self.metrics['dormant_seeds_count'] += 1
            self.metrics['entangled_seeds_count'] -= 1
            self.metrics['communication_channels_open'] -= channels_closed
            self._update_metrics()
            
            logger.info(f"✅ Seed {seed_id[:8]} deactivated")
            if energy_to_return > 0:
                logger.info(f"   Energy returned: {energy_to_return:.1f} SEU")
            
            return {
                'success': True,
                'seed_id': seed_id,
                'energy_returned': energy_to_return,
                'channels_closed': channels_closed,
                'deactivation_time': seed_data['deactivation_time']
            }
            
        except Exception as e:
            logger.error(f"Failed to deactivate seed: {e}")
            raise RuntimeError(f"Seed deactivation failed: {seed_id}") from e
    
    def get_seeds_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all seeds with specified status."""
        try:
            if status == 'active':
                return list(self.active_seeds.values())
            elif status == 'dormant':
                return list(self.dormant_seeds.values())
            elif status == 'entangled':
                return list(self.entangled_seeds.values())
            else:
                raise ValueError(f"Invalid status: {status}")
                
        except Exception as e:
            logger.error(f"Failed to get seeds by status: {e}")
            raise RuntimeError(f"Seed status query failed: {status}") from e
    
    def get_seeds_in_region(self, region: str, sub_region: str = None) -> List[Dict[str, Any]]:
        """Get all seeds in specified region/sub-region."""
        try:
            seeds_in_region = []
            
            # Search in all seed collections
            all_seeds = {**self.active_seeds, **self.dormant_seeds, **self.entangled_seeds}
            
            for seed_data in all_seeds.values():
                if seed_data['region'] == region:
                    if sub_region is None or seed_data['sub_region'] == sub_region:
                        seeds_in_region.append(seed_data)
            
            return seeds_in_region
            
        except Exception as e:
            logger.error(f"Failed to get seeds in region: {e}")
            raise RuntimeError(f"Region seed query failed: {region}") from e
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            self._update_metrics()
            
            status = {
                'system_id': self.system_id,
                'creation_time': self.creation_time,
                'current_time': datetime.now().isoformat(),
                'metrics': self.metrics.copy(),
                'energy_summary': {
                    'total_energy_used': self.metrics['total_energy_used'],
                    'entanglement_energy_used': self.metrics['entanglement_energy_used'],
                    'active_energy_states': len(self.seed_energy_states),
                    'energy_transfers_recorded': len(self.energy_transfer_history)
                },
                'operational_summary': {
                    'active_seeds': len(self.active_seeds),
                    'dormant_seeds': len(self.dormant_seeds),
                    'entangled_seeds': len(self.entangled_seeds),
                    'communication_channels': len(self.communication_channels),
                    'field_modulations': len(self.field_modulations),
                    'entanglement_pairs': len(self.entanglement_pairs)
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise RuntimeError(f"System status query failed") from e
    
    def _update_metrics(self):
        """Update internal metrics."""
        try:
            self.metrics.update({
                'active_seeds_count': len(self.active_seeds),
                'dormant_seeds_count': len(self.dormant_seeds),
                'entangled_seeds_count': len(self.entangled_seeds),
                'communication_channels_open': sum(1 for ch in self.communication_channels.values() 
                                                 if ch.get('status') == 'open'),
                'field_modulations_active': sum(1 for mod in self.field_modulations.values() 
                                              if mod.get('status') == 'active'),
                'entanglement_energy_used': self.entanglement_energy_used,
                'last_updated': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")
    
    def cleanup_inactive_resources(self):
        """Clean up inactive communication channels and field modulations."""
        logger.info("🧹 Cleaning up inactive resources...")
        
        try:
            cleaned_channels = 0
            cleaned_modulations = 0
            
            # Clean up closed communication channels
            channels_to_remove = []
            for channel_id, channel_data in self.communication_channels.items():
                if channel_data.get('status') == 'closed':
                    channels_to_remove.append(channel_id)
            
            for channel_id in channels_to_remove:
                del self.communication_channels[channel_id]
                cleaned_channels += 1
            
            # Clean up inactive field modulations
            modulations_to_remove = []
            for mod_id, mod_data in self.field_modulations.items():
                if mod_data.get('status') == 'inactive':
                    modulations_to_remove.append(mod_id)
            
            for mod_id in modulations_to_remove:
                del self.field_modulations[mod_id]
                cleaned_modulations += 1
            
            if cleaned_channels > 0 or cleaned_modulations > 0:
                logger.info(f"✅ Cleaned {cleaned_channels} channels, {cleaned_modulations} modulations")
            
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"Failed to cleanup inactive resources: {e}")


# --- INTEGRATION FUNCTIONS ---

def create_mycelial_seeds_system(brain_structure: Dict[str, Any]) -> MycelialSeedsSystem:
    """Create and initialize mycelial seeds system from brain structure."""
    logger.info("🌱 Creating mycelial seeds system...")
    
    try:
        # Create system
        seeds_system = MycelialSeedsSystem(brain_structure)
        
        # Load seeds from brain structure
        load_result = seeds_system.load_seeds_from_brain_structure()
        
        if load_result['seeds_loaded'] == 0:
            logger.warning("No seeds loaded - check brain structure")
        
        logger.info(f"✅ Mycelial seeds system created: {load_result['seeds_loaded']} seeds loaded")
        
        return seeds_system
        
    except Exception as e:
        logger.error(f"Failed to create mycelial seeds system: {e}")
        raise RuntimeError(f"Mycelial seeds system creation failed: {e}") from e


