"""
Energy System - Stage 3 System Operations
Handles all energy transfers, returns, and tracking for active brain operations.
Works with energy_storage.py (creation) and mycelial_seeds.py (seed operations).
"""
from shared.constants.constants import *
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import uuid
import random

# --- Logging Setup ---
logger = logging.getLogger("EnergySystem")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class EnergySystem:
    """
    Energy System for Stage 3 Operations
    
    Responsibilities:
    - Track energy levels in active nodes and seeds
    - Handle energy transfers between components
    - Process energy returns after operations
    - Monitor energy thresholds and alerts
    - Coordinate with mycelial seeds for energy distribution
    """
    
    def __init__(self, energy_storage_reference: Dict[str, Any] = None):
        """Initialize energy system with reference to energy storage."""
        self.system_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.energy_storage_ref = energy_storage_reference
        
        # Active tracking (what energy_storage.py doesn't do)
        self.active_nodes_energy = {}      # Track energy in active nodes
        self.active_seeds_energy = {}      # Track energy in active seeds
        self.energy_transfers = []         # History of all transfers
        self.energy_returns = []           # History of all returns
        self.processing_sessions = {}      # Active processing sessions
        
        # System metrics
        self.metrics = {
            'total_energy_transferred': 0.0,
            'total_energy_returned': 0.0,
            'active_processing_sessions': 0,
            'completed_processing_sessions': 0,
            'energy_efficiency': 1.0,
            'last_updated': datetime.now().isoformat()
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'low_energy_percentage': 20.0,
            'critical_energy_percentage': 10.0,
            'high_usage_rate': 50.0  # SEU per minute
        }
        
        logger.info(f"Energy system initialized: {self.system_id[:8]}")
    
    def start_mycelial_processing(self, seed_id: str, processing_type: str = 'sensory_capture') -> str:
        """
        Start mycelial processing session - gives seed 10x synapse energy.
        
        Args:
            seed_id: ID of mycelial seed to energize
            processing_type: Type of processing ('sensory_capture', 'communication', 'field_modulation')
        
        Returns:
            Processing session ID
        """
        logger.info(f"🌱 Starting mycelial processing for seed {seed_id[:8]} - {processing_type}")
        
        try:
            # Calculate energy based on processing type
            base_energy = SYNAPSE_ENERGY_JOULES * MYCELIAL_SEED_BASE_ENERGY_MULTIPLIER * SEU_PER_JOULE
            
            energy_boost = {
                'sensory_capture': MYCELIAL_SEED_TRANSFER_BOOST,
                'communication': MYCELIAL_SEED_COMMUNICATION_BOOST,
                'field_modulation': MYCELIAL_SEED_FIELD_MODULATION_BOOST
            }.get(processing_type, 1.0)
            
            total_energy = base_energy * energy_boost
            
            # Create processing session
            session_id = str(uuid.uuid4())
            processing_session = {
                'session_id': session_id,
                'seed_id': seed_id,
                'processing_type': processing_type,
                'start_time': datetime.now().isoformat(),
                'initial_energy': total_energy,
                'remaining_energy': total_energy,
                'steps_completed': 0,
                'total_steps': SENSORY_CAPTURE_STEPS,
                'energy_per_step': ENERGY_TRANSFER_STEP_AMOUNT,
                'processing_fee_per_step': total_energy * SUBCONSCIOUS_PROCESSING_FEE,
                'status': 'active'
            }
            
            # Track active seed energy
            self.active_seeds_energy[seed_id] = {
                'seed_id': seed_id,
                'current_energy': total_energy,
                'session_id': session_id,
                'last_updated': datetime.now().isoformat()
            }
            
            # Store processing session
            self.processing_sessions[session_id] = processing_session
            self.metrics['active_processing_sessions'] += 1
            
            # Log transfer
            self.energy_transfers.append({
                'transfer_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'from': 'energy_storage',
                'to': seed_id,
                'amount': total_energy,
                'purpose': f'mycelial_processing_{processing_type}',
                'session_id': session_id
            })
            
            self.metrics['total_energy_transferred'] += total_energy
            self._update_metrics()
            
            logger.info(f"✅ Processing session started: {session_id[:8]}")
            logger.info(f"   Energy allocated: {total_energy:.2f} SEU")
            logger.info(f"   Steps planned: {SENSORY_CAPTURE_STEPS}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start mycelial processing: {e}")
            raise RuntimeError(f"Mycelial processing start failed: {e}") from e
    
    def transfer_energy_for_step(self, session_id: str, step_number: int) -> Dict[str, Any]:
        """
        Transfer energy for a single processing step (1 SEU + variance).
        
        Args:
            session_id: Processing session ID
            step_number: Current step number (1-10)
        
        Returns:
            Transfer result details
        """
        logger.debug(f"⚡ Transferring energy for step {step_number} in session {session_id[:8]}")
        
        try:
            if session_id not in self.processing_sessions:
                raise ValueError(f"Processing session {session_id} not found")
            
            session = self.processing_sessions[session_id]
            seed_id = session['seed_id']
            
            # Calculate transfer amount with variance
            base_transfer = session['energy_per_step']
            variance = random.uniform(-ENERGY_TRANSFER_VARIANCE, ENERGY_TRANSFER_VARIANCE)
            transfer_amount = base_transfer * (1 + variance)
            
            # Calculate processing fee (0.5% per step)
            processing_fee = session['processing_fee_per_step']
            
            # Check if enough energy in seed
            if session['remaining_energy'] < (transfer_amount + processing_fee):
                raise RuntimeError(f"Insufficient energy in seed {seed_id} for step {step_number}")
            
            # Update session energy
            session['remaining_energy'] -= (transfer_amount + processing_fee)
            session['steps_completed'] = step_number
            session['last_step_time'] = datetime.now().isoformat()
            
            # Update seed energy tracking
            if seed_id in self.active_seeds_energy:
                self.active_seeds_energy[seed_id]['current_energy'] = session['remaining_energy']
                self.active_seeds_energy[seed_id]['last_updated'] = datetime.now().isoformat()
            
            # Record transfer
            transfer_result = {
                'transfer_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'step_number': step_number,
                'transfer_amount': transfer_amount,
                'processing_fee': processing_fee,
                'remaining_energy': session['remaining_energy'],
                'variance_applied': variance,
                'status': 'completed'
            }
            
            self.energy_transfers.append(transfer_result)
            self.metrics['total_energy_transferred'] += transfer_amount
            
            logger.debug(f"✅ Step {step_number} energy transferred: {transfer_amount:.3f} SEU")
            logger.debug(f"   Processing fee: {processing_fee:.3f} SEU")
            logger.debug(f"   Remaining in seed: {session['remaining_energy']:.2f} SEU")
            
            return transfer_result
            
        except Exception as e:
            logger.error(f"Failed to transfer energy for step: {e}")
            raise RuntimeError(f"Energy transfer failed: {e}") from e
    
    def complete_subconscious_processing(self, session_id: str) -> Dict[str, Any]:
        """
        Complete subconscious processing - return remaining energy to storage.
        
        Args:
            session_id: Processing session ID
        
        Returns:
            Completion details with energy return
        """
        logger.info(f"🧠 Completing subconscious processing for session {session_id[:8]}")
        
        try:
            if session_id not in self.processing_sessions:
                raise ValueError(f"Processing session {session_id} not found")
            
            session = self.processing_sessions[session_id]
            seed_id = session['seed_id']
            
            # Calculate energy return (remaining energy after all steps)
            remaining_energy = session['remaining_energy']
            
            # Update session status
            session['status'] = 'completed_subconscious'
            session['completion_time'] = datetime.now().isoformat()
            session['energy_returned'] = remaining_energy
            
            # Record energy return
            energy_return = {
                'return_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'from': seed_id,
                'to': 'energy_storage',
                'amount': remaining_energy,
                'return_type': 'subconscious_completion',
                'processing_steps_completed': session['steps_completed']
            }
            
            self.energy_returns.append(energy_return)
            self.metrics['total_energy_returned'] += remaining_energy
            
            # Clean up active seed tracking
            if seed_id in self.active_seeds_energy:
                del self.active_seeds_energy[seed_id]
            
            # Update metrics
            self.metrics['active_processing_sessions'] -= 1
            self.metrics['completed_processing_sessions'] += 1
            self._update_metrics()
            
            # Create flag for neural network
            neural_flag = {
                'flag_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'flag_type': 'subconscious_processing_complete',
                'session_id': session_id,
                'seed_id': seed_id,
                'ready_for_conscious_processing': True
            }
            
            logger.info(f"✅ Subconscious processing completed")
            logger.info(f"   Energy returned: {remaining_energy:.2f} SEU")
            logger.info(f"   Steps completed: {session['steps_completed']}/{session['total_steps']}")
            logger.info(f"   Neural flag created: {neural_flag['flag_id'][:8]}")
            
            return {
                'completion_details': session,
                'energy_return': energy_return,
                'neural_flag': neural_flag
            }
            
        except Exception as e:
            logger.error(f"Failed to complete subconscious processing: {e}")
            raise RuntimeError(f"Subconscious processing completion failed: {e}") from e
    
    def start_conscious_processing(self, neural_flag: Dict[str, Any]) -> str:
        """
        Start conscious processing - allocate 3x10 synapse energy.
        
        Args:
            neural_flag: Flag from completed subconscious processing
        
        Returns:
            Conscious processing session ID
        """
        logger.info(f"🧠💡 Starting conscious processing from flag {neural_flag['flag_id'][:8]}")
        
        try:
            # Calculate conscious processing energy (3x more than subconscious)
            base_energy = SYNAPSE_ENERGY_JOULES * MYCELIAL_SEED_BASE_ENERGY_MULTIPLIER * SEU_PER_JOULE
            conscious_energy = base_energy * CONSCIOUS_PROCESSING_MULTIPLIER
            
            # Create conscious processing session
            conscious_session_id = str(uuid.uuid4())
            conscious_session = {
                'session_id': conscious_session_id,
                'session_type': 'conscious_processing',
                'parent_session_id': neural_flag['session_id'],
                'seed_id': neural_flag['seed_id'],
                'start_time': datetime.now().isoformat(),
                'allocated_energy': conscious_energy,
                'remaining_energy': conscious_energy,
                'processing_stage': 'semantic_analysis',
                'status': 'active'
            }
            
            # Track as active processing
            self.processing_sessions[conscious_session_id] = conscious_session
            self.metrics['active_processing_sessions'] += 1
            
            # Log energy allocation
            self.energy_transfers.append({
                'transfer_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'from': 'energy_storage',
                'to': 'conscious_processor',
                'amount': conscious_energy,
                'purpose': 'conscious_processing',
                'session_id': conscious_session_id
            })
            
            self.metrics['total_energy_transferred'] += conscious_energy
            self._update_metrics()
            
            logger.info(f"✅ Conscious processing started: {conscious_session_id[:8]}")
            logger.info(f"   Energy allocated: {conscious_energy:.2f} SEU")
            
            return conscious_session_id
            
        except Exception as e:
            logger.error(f"Failed to start conscious processing: {e}")
            raise RuntimeError(f"Conscious processing start failed: {e}") from e
    
    def get_energy_status(self) -> Dict[str, Any]:
        """Get comprehensive energy system status."""
        return {
            'system_id': self.system_id,
            'creation_time': self.creation_time,
            'active_seeds_count': len(self.active_seeds_energy),
            'active_processing_sessions': self.metrics['active_processing_sessions'],
            'completed_sessions': self.metrics['completed_processing_sessions'],
            'total_transfers': len(self.energy_transfers),
            'total_returns': len(self.energy_returns),
            'energy_efficiency': self.metrics['energy_efficiency'],
            'last_updated': self.metrics['last_updated']
        }
    
    def _update_metrics(self):
        """Update system metrics."""
        # Calculate energy efficiency
        if self.metrics['total_energy_transferred'] > 0:
            self.metrics['energy_efficiency'] = (
                self.metrics['total_energy_returned'] / 
                self.metrics['total_energy_transferred']
            )
        
        self.metrics['last_updated'] = datetime.now().isoformat()


