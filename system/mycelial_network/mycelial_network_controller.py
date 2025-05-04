# --- mycelial_network_controller.py - Controller for the mycelial network system ---

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os
import json

# Import mycelial network
from system.mycelial_network.mycelial_network import MycelialNetwork
from system.mycelial_network.monitoring.state_monitoring_controller import StateMonitoringController
from system.mycelial_network.energy.energy_system import EnergySystem
from system.mycelial_network.memory_3d.memory_structure import MemoryStructure


# Configure logging
logger = logging.getLogger("MycelialNetworkController")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MycelialNetworkController:
    """
    Controller for the mycelial network system.
    Coordinates operations between the mycelial network, memory, and energy systems.
    """
    
    def __init__(self, brain_grid):
        """
        Initialize the mycelial network controller.
        
        Args:
            brain_grid: Reference to the brain grid structure
        """
        self.brain_grid = brain_grid
        self.mycelial_network = MycelialNetwork(brain_grid)
        self.initialized = False
        self.monitoring_active = False
        
        # Integration tracking
        self.energy_system_integrated = False
        self.memory_system_integrated = False
        self.processing_system_integrated = False
        self.subconscious_integrated = False
        
        # Operation tracking
        self.last_energy_distribution = None
        self.last_frequency_propagation = None
        self.operations_log = []
        
        # Store configuration
        self.config = {
            "auto_energy_distribution": True,
            "enable_frequency_propagation": True,
            "memory_pathway_creation": "auto",
            "system_monitoring_interval": 60,  # seconds
            "subconscious_integration_level": "standard"
        }
        
        logger.info("Mycelial network controller initialized")
    
    def initialize_system(self, seed_position: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Initialize the mycelial network system.
        
        Args:
            seed_position: Optional position for the brain seed
            
        Returns:
            Dict containing initialization results
        """
        logger.info("Initializing mycelial network system")
        
        # Initialize mycelial network
        network_init = self.mycelial_network.initialize_network(seed_position)
        
        # Initialize tracking
        self.initialized = network_init
        self.operations_log.append({
            "operation": "system_initialization",
            "timestamp": datetime.now().isoformat(),
            "success": network_init,
            "seed_position": seed_position
        })
        
        # Return initialization results
        return {
            "success": network_init,
            "system_initialized": self.initialized,
            "timestamp": datetime.now().isoformat()
        }
    
    def integrate_energy_system(self) -> Dict:
        """
        Integrate with the energy system.
        
        Returns:
            Dict containing integration results
        """
        if not self.initialized:
            return {"success": False, "error": "System not initialized"}
        
        logger.info("Integrating with energy system")
        
        try:
            # Import energy system components
            # This would typically connect to your energy system modules
            from system.mycelial_network.energy.energy_system import EnergySystem
            
            # Perform integration logic
            # This is a placeholder for actual integration code
            
            self.energy_system_integrated = True
            self.operations_log.append({
                "operation": "energy_system_integration",
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            return {
                "success": True,
                "energy_system_integrated": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Energy system integration failed: {e}")
            self.operations_log.append({
                "operation": "energy_system_integration",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": f"Energy system integration failed: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def integrate_memory_system(self) -> Dict:
        """
        Integrate with the memory system.
        
        Returns:
            Dict containing integration results
        """
        if not self.initialized:
            return {"success": False, "error": "System not initialized"}
        
        logger.info("Integrating with memory system")
        
        try:
            # Import memory system components
            # This would typically connect to your memory system modules
            from system.mycelial_network.memory_3d.memory_structure import MemoryStructure
            
            # Perform integration logic
            # This is a placeholder for actual integration code
            
            self.memory_system_integrated = True
            self.operations_log.append({
                "operation": "memory_system_integration",
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            return {
                "success": True,
                "memory_system_integrated": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Memory system integration failed: {e}")
            self.operations_log.append({
                "operation": "memory_system_integration",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": f"Memory system integration failed: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def distribute_soul_energy(self, seed_position: Tuple[int, int, int], 
                              energy_amount: float, soul_properties: Dict) -> Dict:
        """
        Distribute soul energy through the mycelial network.
        
        Args:
            seed_position: Position of brain seed (x, y, z)
            energy_amount: Amount of energy to distribute (BEU)
            soul_properties: Dictionary containing soul properties
            
        Returns:
            Dict containing distribution results
        """
        if not self.initialized:
            return {"success": False, "error": "System not initialized"}
        
        logger.info(f"Distributing {energy_amount:.2f} BEU of soul energy")
      
        # Extract soul properties
        soul_frequency = soul_properties.get('frequency', 432.0)
        soul_stability = soul_properties.get('stability', 0.5) / 100.0  # Normalize from 0-100 to 0-1
        soul_coherence = soul_properties.get('coherence', 0.5) / 100.0  # Normalize from 0-100 to 0-1
        


        # Distribute energy through mycelial network
        distribution_results = self.mycelial_network.distribute_energy(
            seed_position, energy_amount, soul_frequency, soul_stability, soul_coherence
        )
        
        # Propagate frequencies if enabled
        if self.config["enable_frequency_propagation"] and distribution_results["success"]:
            propagation_results = self.mycelial_network.propagate_frequencies(
                soul_frequency, soul_coherence
            )
            distribution_results["frequency_propagation"] = propagation_results
        
        # Create memory pathways for soul aspects if enabled and aspects provided
        if (self.config["memory_pathway_creation"] in ["auto", "enabled"] and 
            "aspects" in soul_properties and 
            distribution_results["success"]):
            
            aspects = soul_properties["aspects"]
            pathway_results = self.mycelial_network.create_memory_pathways(aspects)
            distribution_results["memory_pathways"] = pathway_results
        
        # Update tracking
        self.last_energy_distribution = distribution_results
        self.operations_log.append({
            "operation": "soul_energy_distribution",
            "timestamp": datetime.now().isoformat(),
            "success": distribution_results["success"],
            "energy_amount": energy_amount
        })
        
        return distribution_results
    
    def start_monitoring(self) -> Dict:
        """
        Start monitoring the mycelial network system.
        
        Returns:
            Dict containing monitoring start results
        """
        if not self.initialized:
            return {"success": False, "error": "System not initialized"}
        
        logger.info("Starting mycelial network monitoring")
        
        try:
            from .monitoring.state_monitoring_controller import StateMonitoringController
            state_monitor = StateMonitoringController(self.brain_grid, mycelial_network=self.mycelial_network)
            # State monitoring code...
            
            self.monitoring_active = True
            self.operations_log.append({
                "operation": "monitoring_start",
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            return {
                "success": True,
                "monitoring_active": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Monitoring start failed: {e}")
            self.operations_log.append({
                "operation": "monitoring_start",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": f"Monitoring start failed: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive status of the mycelial network system.
        
        Returns:
            Dict containing system status
        """
        # Get mycelial network metrics
        network_metrics = self.mycelial_network.get_network_metrics() if self.initialized else {
            "initialized": False
        }
        
        # Compile system status
        system_status = {
            "initialized": self.initialized,
            "energy_system_integrated": self.energy_system_integrated,
            "memory_system_integrated": self.memory_system_integrated,
            "processing_system_integrated": self.processing_system_integrated,
            "subconscious_integrated": self.subconscious_integrated,
            "monitoring_active": self.monitoring_active,
            "mycelial_network": network_metrics,
            "config": self.config,
            "operations_count": len(self.operations_log),
            "last_operation": self.operations_log[-1] if self.operations_log else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return system_status
    
    def update_config(self, config_updates: Dict) -> Dict:
        """
        Update system configuration.
        
        Args:
            config_updates: Dictionary containing configuration updates
            
        Returns:
            Dict containing update results
        """
        logger.info(f"Updating mycelial network configuration: {config_updates}")
        
        # Update configuration
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Log operation
        self.operations_log.append({
            "operation": "config_update",
            "timestamp": datetime.now().isoformat(),
            "updates": config_updates
        })
        
        return {
            "success": True,
            "updated_config": self.config,
            "timestamp": datetime.now().isoformat()
        }
    
def save_system_state(self, output_path: str) -> Dict:
    """
    Save the current system state to disk.
    
    Args:
        output_path: Path to save the system state
        
    Returns:
        Dict containing save results
    """
    logger.info(f"Saving mycelial network system state to {output_path}")
    
    try:
        # Compile system state
        system_state = {
            "initialized": self.initialized,
            "energy_system_integrated": self.energy_system_integrated,
            "memory_system_integrated": self.memory_system_integrated,
            "processing_system_integrated": self.processing_system_integrated,
            "subconscious_integrated": self.subconscious_integrated,
            "monitoring_active": self.monitoring_active,
            "config": self.config,
            "operations_log": self.operations_log,
            "network_metrics": self.mycelial_network.get_network_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(system_state, f, indent=2, default=str)
        
        logger.info(f"System state saved to {output_path}")
        return {
            "success": True,
            "output_path": output_path,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error saving system state: {e}")
        return {
            "success": False,
            "error": f"Failed to save system state: {e}",
            "timestamp": datetime.now().isoformat()
        }

def load_system_state(self, input_path: str) -> Dict:
    """
    Load system state from disk.
    
    Args:
        input_path: Path to load the system state from
        
    Returns:
        Dict containing load results
    """
    logger.info(f"Loading mycelial network system state from {input_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(input_path):
            return {
                "success": False,
                "error": f"File not found: {input_path}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Load from file
        with open(input_path, 'r') as f:
            system_state = json.load(f)
        
        # Validate state data
        if not isinstance(system_state, dict) or "initialized" not in system_state:
            return {
                "success": False,
                "error": "Invalid system state data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Update controller state
        self.initialized = system_state.get("initialized", False)
        self.energy_system_integrated = system_state.get("energy_system_integrated", False)
        self.memory_system_integrated = system_state.get("memory_system_integrated", False)
        self.processing_system_integrated = system_state.get("processing_system_integrated", False)
        self.subconscious_integrated = system_state.get("subconscious_integrated", False)
        self.monitoring_active = system_state.get("monitoring_active", False)
        self.config = system_state.get("config", self.config)
        self.operations_log = system_state.get("operations_log", [])
        
        # Note: This doesn't restore the actual mycelial network density and energy grids
        # Those would need to be reconstructed separately
        
        logger.info(f"System state loaded from {input_path}")
        return {
            "success": True,
            "loaded_state": {k: v for k, v in system_state.items() if k != "operations_log"},
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error loading system state: {e}")
        return {
            "success": False,
            "error": f"Failed to load system state: {e}",
            "timestamp": datetime.now().isoformat()
        }

def create_subconscious_integration(self, learning_paths: Optional[List[str]] = None) -> Dict:
    """
    Integrate with the subconscious system and create pathways for learning.
    
    Args:
        learning_paths: Optional list of specific learning pathways to integrate
        
    Returns:
        Dict containing integration results
    """
    if not self.initialized:
        return {"success": False, "error": "System not initialized"}
    
    logger.info("Creating subconscious integration pathways")
    
    try:
        # Default learning paths if none specified
        if not learning_paths:
            learning_paths = [
                "emotional_maturity",
                "creative_conceptualisation",
                "language_learning",
                "logic_learning",
                "spiritual_progress"
            ]
        
        # Placeholder for subconscious integration logic
        # This would typically connect to your subconscious modules
        
        # Simulate pathway creation results
        pathways_created = 0
        integration_results = {}
        
        for path in learning_paths:
            # Placeholder for actual integration logic
            success = True  # Simulated success
            
            if success:
                pathways_created += 1
                integration_results[path] = {"success": True}
            else:
                integration_results[path] = {"success": False, "error": "Integration failed"}
        
        self.subconscious_integrated = pathways_created > 0
        self.operations_log.append({
            "operation": "subconscious_integration",
            "timestamp": datetime.now().isoformat(),
            "success": self.subconscious_integrated,
            "pathways_created": pathways_created,
            "learning_paths": learning_paths
        })
        
        return {
            "success": self.subconscious_integrated,
            "pathways_created": pathways_created,
            "integration_results": integration_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Subconscious integration failed: {e}")
        return {
            "success": False,
            "error": f"Subconscious integration failed: {e}",
            "timestamp": datetime.now().isoformat()
        }

def process_homeostatic_feedback(self, feedback: Dict) -> Dict:
    """
    Process homeostatic feedback to adjust the mycelial network.
    
    Args:
        feedback: Dictionary containing homeostatic feedback data
        
    Returns:
            Dict containing processing results
    """
    if not self.initialized:
        return {"success": False, "error": "System not initialized"}
    
    logger.info("Processing homeostatic feedback for mycelial network")
    
    try:
        # Extract feedback data
        stress_level = feedback.get("stress_level", 0.5)
        energy_balance = feedback.get("energy_balance", 0.0)
        regional_imbalances = feedback.get("regional_imbalances", {})
        suggested_actions = feedback.get("suggested_actions", [])
        
        # Placeholder for actual feedback processing logic
        # This would typically make adjustments to the mycelial network
        
        # Track processing results
        adjustments_made = 0
        regions_adjusted = []
        
        # Apply feedback
        if "adjust_network_density" in suggested_actions:
            # Placeholder for density adjustment logic
            adjustments_made += 1
        
        if "redistribute_energy" in suggested_actions:
            # Placeholder for energy redistribution logic
            adjustments_made += 1
        
        if "reinforce_pathways" in suggested_actions:
            # Placeholder for pathway reinforcement logic
            adjustments_made += 1
        
        # Process regional imbalances
        for region, imbalance in regional_imbalances.items():
            # Placeholder for regional adjustment logic
            regions_adjusted.append(region)
            adjustments_made += 1
        
        self.operations_log.append({
            "operation": "homeostatic_feedback_processing",
            "timestamp": datetime.now().isoformat(),
            "feedback_summary": {
                "stress_level": stress_level,
                "energy_balance": energy_balance,
                "regions_with_imbalances": list(regional_imbalances.keys())
            },
            "adjustments_made": adjustments_made
        })
        
        return {
            "success": True,
            "adjustments_made": adjustments_made,
            "regions_adjusted": regions_adjusted,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Homeostatic feedback processing failed: {e}")
        return {
            "success": False,
            "error": f"Feedback processing failed: {e}",
            "timestamp": datetime.now().isoformat()
        }

def shutdown_system(self) -> Dict:
    """
    Safely shutdown the mycelial network system.
    
    Returns:
        Dict containing shutdown results
    """
    logger.info("Shutting down mycelial network system")
    
    try:
        # Perform shutdown operations
        if self.monitoring_active:
            # Stop monitoring
            self.monitoring_active = False
        
        # Reset mycelial network
        if self.initialized:
            self.mycelial_network.reset_network()
        
        # Update state
        self.initialized = False
        self.energy_system_integrated = False
        self.memory_system_integrated = False
        self.processing_system_integrated = False
        self.subconscious_integrated = False
        
        # Log operation
        self.operations_log.append({
            "operation": "system_shutdown",
            "timestamp": datetime.now().isoformat(),
            "success": True
        })
        
        return {
            "success": True,
            "system_shutdown": True,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"System shutdown failed: {e}")
        return {
            "success": False,
            "error": f"System shutdown failed: {e}",
            "timestamp": datetime.now().isoformat()
        }