"""
Birth Module

This module implements the process of birth transitioning a soul from the
spiritual realm to physical incarnation. It manages the soul's attachment
to the physical body, memory veil, and first breath integration.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import time
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='birth.log'
)
logger = logging.getLogger('birth')

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
try:
    from void.soul_spark import SoulSpark
    import metrics_tracking as metrics
    from constants import EARTH_FREQUENCIES, GOLDEN_RATIO
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    # Define fallback constants
    EARTH_FREQUENCIES = {
        "schumann": 7.83,
        "heartbeat": 1.2,  # ~72 bpm
        "breath": 0.2      # ~12 breaths per minute
    }
    GOLDEN_RATIO = 1.618033988749895


class Birth:
    """
    Birth Process Manager
    
    This class handles the process of birthing a soul into physical incarnation,
    including the attachment to the physical form, memory veil deployment,
    and first breath integration.
    """
    
    def __init__(self, soul_spark):
        """
        Initialize the birth process.
        
        Args:
            soul_spark: The soul spark to birth
        """
        self.soul_spark = soul_spark
        self.birth_id = str(uuid.uuid4())
        
        # Process state tracking
        self.birth_complete = False
        self.current_phase = None
        self.phase_results = {}
        
        # Physical form connection
        self.physical_connection = 0.0
        self.form_acceptance = 0.0
        self.memory_veil_deployed = False
        self.first_breath_complete = False
        
        # Process metrics
        self.metrics = {
            "start_time": 0.0,
            "phase_metrics": {},
            "final_integration": 0.0,
            "birth_duration": 0.0
        }
        
        logger.info(f"Birth process initialized for soul spark {getattr(soul_spark, 'id', 'unknown')}")
    
    def begin_birth(self, intensity=0.8):
        """
        Begin the birth process.
        
        Args:
            intensity (float): Intensity of the birth process (0.1-1.0)
                Higher intensity results in a faster, more traumatic birth
                Lower intensity is gentler but may result in weaker physical connection
                
        Returns:
            bool: Success status
        """
        logger.info(f"Beginning birth process with intensity {intensity}")
        self.metrics["start_time"] = time.time()
        
        # Check prerequisites
        if not self._check_prerequisites():
            logger.error("Soul does not meet prerequisites for birth")
            return False
        
        # Phase 1: Physical Form Connection
        self.current_phase = "physical_connection"
        success = self._connect_to_physical_form(intensity)
        if not success:
            logger.error("Physical form connection failed")
            return False
            
        # Phase 2: Life Cord Transfer
        self.current_phase = "life_cord_transfer"
        success = self._transfer_life_cord(intensity)
        if not success:
            logger.error("Life cord transfer failed")
            return False
            
        # Phase 3: Memory Veil Deployment
        self.current_phase = "memory_veil"
        success = self._deploy_memory_veil(intensity)
        if not success:
            logger.error("Memory veil deployment failed")
            return False
            
        # Phase 4: First Breath Integration
        self.current_phase = "first_breath"
        success = self._first_breath_integration(intensity)
        if not success:
            logger.error("First breath integration failed")
            return False
            
        # Phase 5: Finalize Birth
        self.current_phase = "birth_finalization"
        success = self._finalize_birth()
        if not success:
            logger.error("Birth finalization failed")
            return False
        
        # Complete birth process
        self.birth_complete = True
        birth_duration = time.time() - self.metrics["start_time"]
        self.metrics["birth_duration"] = birth_duration
        
        # Record metrics
        self._record_metrics()
        
        logger.info(f"Birth completed in {birth_duration:.2f} seconds")
        return True
    
    def _check_prerequisites(self):
        """
        Check if prerequisites for birth are met.
        
        Returns:
            bool: True if prerequisites are met
        """
        # Check for Earth harmonization
        if not getattr(self.soul_spark, "earth_harmonized", False):
            logger.warning("Soul has not been harmonized with Earth")
            return False
            
        # Check if ready for birth
        if not getattr(self.soul_spark, "ready_for_birth", False):
            logger.warning("Soul not ready for birth")
            return False
            
        # Check if life cord is intact
        cord_integrity = getattr(self.soul_spark, "cord_integrity", 0.0)
        if cord_integrity < 0.7:  # Higher threshold for birth
            logger.warning(f"Life cord integrity too low for birth: {cord_integrity:.2f}")
            return False
            
        # Check for sufficient Earth resonance
        earth_resonance = getattr(self.soul_spark, "earth_resonance", 0.0)
        if earth_resonance < 0.6:
            logger.warning(f"Earth resonance too low for birth: {earth_resonance:.2f}")
            return False
            
        return True
    
    def _connect_to_physical_form(self, intensity):
        """
        Connect the soul to its physical form.
        
        Args:
            intensity (float): Intensity of connection
            
        Returns:
            bool: Success status
        """
        logger.info("Connecting to physical form")
        
        # Get soul properties relevant for physical connection
        earth_resonance = getattr(self.soul_spark, "earth_resonance", 0.6)
        cord_integrity = getattr(self.soul_spark, "cord_integrity", 0.7)
        
        # Calculate base connection strength
        base_strength = (earth_resonance * 0.4 + cord_integrity * 0.6)
        
        # Intensity affects how strongly the connection is made
        # Higher intensity results in stronger connection but more trauma
        trauma_factor = intensity * 0.3
        connection_factor = intensity * 0.7
        
        # Calculate form connection with factors
        max_potential = base_strength * (1.0 + connection_factor)
        connection_strength = min(0.95, max_potential) # Cap at 0.95
        
        # Calculate trauma level
        trauma_level = trauma_factor * intensity
        
        # Calculate form acceptance - inversely related to trauma
        acceptance = max(0.5, 1.0 - trauma_level * 0.6)
        
        # Store connection metrics
        self.physical_connection = connection_strength
        self.form_acceptance = acceptance
        
        # Store phase results
        self.phase_results["physical_connection"] = {
            "base_strength": base_strength,
            "trauma_level": trauma_level,
            "connection_strength": connection_strength,
            "form_acceptance": acceptance
        }
        
        # Record phase metrics
        self.metrics["phase_metrics"]["physical_connection"] = {
            "connection_strength": connection_strength,
            "form_acceptance": acceptance,
            "trauma_level": trauma_level
        }
        
        logger.info(f"Physical form connection established: {connection_strength:.2f}")
        logger.info(f"Form acceptance: {acceptance:.2f}, Trauma level: {trauma_level:.2f}")
        
        return True
    
    def _transfer_life_cord(self, intensity):
        """
        Transfer the life cord to the physical form.
        
        Args:
            intensity (float): Intensity of transfer
            
        Returns:
            bool: Success status
        """
        logger.info("Transferring life cord")
        
        # Get life cord from soul
        life_cord = getattr(self.soul_spark, "life_cord", None)
        if not life_cord:
            logger.error("Soul does not have a life cord")
            return False
            
        # Get cord integrity
        cord_integrity = getattr(self.soul_spark, "cord_integrity", 0.7)
        
        # Calculate transfer efficiency based on intensity and integrity
        # Lower intensity allows for more careful, complete transfer
        transfer_efficiency = cord_integrity * (1.0 - intensity * 0.3)
        
        # Calculate form integration based on physical connection
        form_integration = self.physical_connection * 0.8
        
        # Calculate new cord integrity after transfer
        new_integrity = cord_integrity * transfer_efficiency
        
        # Calculate bandwidth reduction due to transfer
        original_bandwidth = life_cord.get("bandwidth", 100)
        new_bandwidth = original_bandwidth * transfer_efficiency
        
        # Update life cord properties
        updated_cord = life_cord.copy() if isinstance(life_cord, dict) else {}
        if "bandwidth" in updated_cord:
            updated_cord["bandwidth"] = new_bandwidth
        
        updated_cord["form_integration"] = form_integration
        updated_cord["physical_anchored"] = True
        
        # Store phase results
        self.phase_results["life_cord_transfer"] = {
            "transfer_efficiency": transfer_efficiency,
            "form_integration": form_integration,
            "original_integrity": cord_integrity,
            "new_integrity": new_integrity,
            "original_bandwidth": original_bandwidth,
            "new_bandwidth": new_bandwidth
        }
        
        # Record phase metrics
        self.metrics["phase_metrics"]["life_cord_transfer"] = {
            "transfer_efficiency": transfer_efficiency,
            "form_integration": form_integration,
            "integrity_change": new_integrity - cord_integrity,
            "bandwidth_reduction": original_bandwidth - new_bandwidth
        }
        
        # Update soul properties
        setattr(self.soul_spark, "life_cord", updated_cord)
        setattr(self.soul_spark, "cord_integrity", new_integrity)
        setattr(self.soul_spark, "form_integration", form_integration)
        
        logger.info(f"Life cord transferred with efficiency: {transfer_efficiency:.2f}")
        logger.info(f"New cord integrity: {new_integrity:.2f}, Form integration: {form_integration:.2f}")
        
        return True
    
    def _deploy_memory_veil(self, intensity):
        """
        Deploy the memory veil to limit pre-birth memories.
        
        Args:
            intensity (float): Intensity of memory veil deployment
            
        Returns:
            bool: Success status
        """
        logger.info("Deploying memory veil")
        
        # Higher intensity creates a stronger memory veil
        veil_strength = 0.6 + intensity * 0.3
        
        # Calculate veil permanence - how lasting the veil is
        veil_permanence = 0.7 + intensity * 0.2
        
        # Calculate memory retention - what percentage of memories remain accessible
        memory_retention = max(0.01, 0.2 - intensity * 0.15)  # Min 1% retention
        
        # Memory types and their retention modifiers
        memory_types = {
            "emotional": 0.3,  # Emotional memories are more likely to be retained
            "sensory": 0.2,    # Sensory memories somewhat retained
            "conceptual": 0.1,  # Conceptual memories mostly veiled
            "specific": 0.05    # Specific memories almost entirely veiled
        }
        
        # Calculate retention for each memory type
        memory_retentions = {}
        for mem_type, mod in memory_types.items():
            retention = min(1.0, memory_retention + mod)
            memory_retentions[mem_type] = retention
        
        # Create veil configuration
        veil_config = {
            "strength": veil_strength,
            "permanence": veil_permanence,
            "base_retention": memory_retention,
            "memory_retentions": memory_retentions,
            "deployment_time": time.time()
        }
        
        # Store phase results
        self.phase_results["memory_veil"] = veil_config.copy()
        
        # Record phase metrics
        self.metrics["phase_metrics"]["memory_veil"] = {
            "veil_strength": veil_strength,
            "veil_permanence": veil_permanence,
            "memory_retention": memory_retention
        }
        
        # Update soul properties
        setattr(self.soul_spark, "memory_veil", veil_config)
        setattr(self.soul_spark, "memory_retention", memory_retention)
        
        # Mark veil as deployed
        self.memory_veil_deployed = True
        
        logger.info(f"Memory veil deployed with strength: {veil_strength:.2f}")
        logger.info(f"Memory retention: {memory_retention:.2f}, Permanence: {veil_permanence:.2f}")
        
        return True

    def _first_breath_integration(self, intensity):
        """
        Integrate the first breath to anchor the soul to physical existence.
        
        Args:
            intensity (float): Intensity of the integration
            
        Returns:
            bool: Success status
        """
        logger.info("Integrating first breath")
        
        # Earth frequency baseline (natural breath rhythm)
        earth_breath_freq = EARTH_FREQUENCIES.get("breath", 0.2)  # ~12 breaths per minute
        
        # Intensity affects the breath pattern
        # Higher intensity creates a more dramatic breath (birth cry)
        breath_amplitude = 0.6 + intensity * 0.4
        breath_depth = 0.5 + intensity * 0.5
        
        # Calculate breath synchronization factor
        earth_resonance = getattr(self.soul_spark, "earth_resonance", 0.6)
        breath_sync = earth_resonance * 0.9
        
        # Breath integration based on physical connection
        integration_strength = self.physical_connection * breath_sync
        
        # Calculate resonance enhancement
        resonance_boost = breath_sync * breath_depth * 0.2
        new_earth_resonance = min(1.0, earth_resonance + resonance_boost)
        
        # Calculate energetic shift from breath (spiritual to physical)
        energy_shift = breath_depth * integration_strength
        physical_energy = 0.5 + energy_shift * 0.5
        spiritual_energy = max(0.1, 1.0 - energy_shift * 0.7)
        
        # Create breath configuration
        breath_config = {
            "frequency": earth_breath_freq,
            "amplitude": breath_amplitude,
            "depth": breath_depth,
            "sync_factor": breath_sync,
            "integration_strength": integration_strength,
            "physical_energy": physical_energy,
            "spiritual_energy": spiritual_energy,
            "timestamp": time.time()
        }
        
        # Store phase results
        self.phase_results["first_breath"] = breath_config.copy()
        
        # Record phase metrics
        self.metrics["phase_metrics"]["first_breath"] = {
            "integration_strength": integration_strength,
            "resonance_boost": resonance_boost,
            "physical_energy": physical_energy,
            "spiritual_energy": spiritual_energy
        }
        
        # Update soul properties
        setattr(self.soul_spark, "breath_pattern", breath_config)
        setattr(self.soul_spark, "earth_resonance", new_earth_resonance)
        setattr(self.soul_spark, "physical_energy", physical_energy)
        setattr(self.soul_spark, "spiritual_energy", spiritual_energy)
        
        # Mark first breath as complete
        self.first_breath_complete = True
        
        logger.info(f"First breath integrated with strength: {integration_strength:.2f}")
        logger.info(f"Physical energy: {physical_energy:.2f}, Spiritual energy: {spiritual_energy:.2f}")
        
        return True
    
    def _finalize_birth(self):
        """
        Finalize the birth process and transition to physical life.
        
        Returns:
            bool: Success status
        """
        logger.info("Finalizing birth process")
        
        # Check if all phases complete
        if not self.memory_veil_deployed or not self.first_breath_complete:
            logger.warning("Not all phases complete, birth may be incomplete")
            
        # Calculate final integration level
        connection_strength = self.physical_connection * 0.4
        form_acceptance = self.form_acceptance * 0.3
        
        # Get breath integration if available
        breath_config = getattr(self.soul_spark, "breath_pattern", {})
        breath_integration = breath_config.get("integration_strength", 0.7) * 0.3
        
        # Calculate total integration
        total_integration = connection_strength + form_acceptance + breath_integration
        
        # Adjust core soul properties for physical existence
        original_frequency = getattr(self.soul_spark, "frequency", 528.0)
        physical_frequency = original_frequency * 0.5  # Reduce frequency for physical plane
        
        original_stability = getattr(self.soul_spark, "stability", 0.7)
        physical_stability = original_stability * 0.8  # Slight reduction in stability
        
        # Update soul properties
        setattr(self.soul_spark, "frequency", physical_frequency)
        setattr(self.soul_spark, "stability", physical_stability)
        setattr(self.soul_spark, "physical_integration",
