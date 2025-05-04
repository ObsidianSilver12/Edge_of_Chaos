# --- energy_system.py - Basic energy system for mycelial network ---

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger("EnergySystem")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class EnergySystem:
    """
    Energy system for managing and tracking energy distribution in the brain.
    Handles energy conservation, storage, and conversion between different types.
    """
    
    def __init__(self, brain_grid=None, mycelial_network=None):
        """Initialize the energy system"""
        self.brain_grid = brain_grid
        self.mycelial_network = mycelial_network
        self.initialized = False
        self.creation_time = datetime.now().isoformat()
        
        # Energy stores
        self.total_beu = 0.0  # Brain Energy Units
        self.mycelial_store = 0.0  # Stored for future use
        self.operational_level = 0.0  # Currently in use
        self.available_level = 0.0  # Available for allocation
        
        # Energy audit
        self.energy_allocated = {}  # Tracking energy by function
        self.energy_conversions = []  # History of conversions
        self.energy_consumption_rate = 0.0  # BEU/second
        
        # Constants
        self.BEU_TO_JOULE = 1e-12  # Conversion factor
        self.MIN_OPERATIONAL_ENERGY = 100.0  # Minimum BEU for operation
        self.BASELINE_CONSUMPTION_BEU = 0.1  # BEU/second at rest
        
        logger.info("Energy system initialized")
    
    def initialize_with_energy(self, initial_beu, mycelial_beu=None):
        """
        Initialize the energy system with starting energy levels
        
        Args:
            initial_beu: Initial Brain Energy Units
            mycelial_beu: Optional energy for mycelial store
        """
        self.total_beu = float(initial_beu)
        
        # Split energy between stores
        if mycelial_beu is not None:
            self.mycelial_store = float(mycelial_beu)
            self.available_level = self.total_beu - self.mycelial_store
        else:
            # Default split: 60% mycelial, 40% available
            self.mycelial_store = self.total_beu * 0.6
            self.available_level = self.total_beu * 0.4
        
        # Set operational energy (minimum needed)
        self.operational_level = min(self.MIN_OPERATIONAL_ENERGY, self.available_level * 0.5)
        self.available_level -= self.operational_level
        
        self.initialized = True
        logger.info(f"Energy system initialized with {self.total_beu:.2E} BEU total, "
                   f"{self.mycelial_store:.2E} BEU in mycelial store")
        
        return {
            "success": True,
            "total_beu": self.total_beu,
            "mycelial_store": self.mycelial_store,
            "operational_level": self.operational_level,
            "available_level": self.available_level
        }
    
    def allocate_energy(self, amount, purpose, coordinates=None):
        """
        Allocate energy from available pool for a specific purpose
        
        Args:
            amount: Amount of BEU to allocate
            purpose: Purpose identifier
            coordinates: Optional brain coordinates
            
        Returns:
            Success status and allocation details
        """
        if not self.initialized:
            return {"success": False, "error": "Energy system not initialized"}
        
        if amount <= 0:
            return {"success": False, "error": "Invalid energy amount"}
        
        if amount > self.available_level:
            # Try to draw from mycelial store if available energy is insufficient
            if amount <= (self.available_level + self.mycelial_store):
                shortfall = amount - self.available_level
                self.mycelial_store -= shortfall
                self.available_level += shortfall
                logger.info(f"Drew {shortfall:.2E} BEU from mycelial store for {purpose}")
            else:
                return {"success": False, "error": "Insufficient energy available"}
        
        # Perform allocation
        self.available_level -= amount
        
        # Record allocation
        if purpose not in self.energy_allocated:
            self.energy_allocated[purpose] = 0.0
        self.energy_allocated[purpose] += amount
        
        # If brain grid and coordinates provided, update energy grid
        if self.brain_grid is not None and coordinates is not None:
            x, y, z = coordinates
            if (0 <= x < self.brain_grid.dimensions[0] and 
                0 <= y < self.brain_grid.dimensions[1] and 
                0 <= z < self.brain_grid.dimensions[2]):
                self.brain_grid.energy_grid[x, y, z] += amount * 0.01  # Scale for grid
        
        logger.info(f"Allocated {amount:.2E} BEU for {purpose}")
        return {
            "success": True,
            "amount": amount,
            "purpose": purpose,
            "remaining_available": self.available_level,
            "remaining_mycelial": self.mycelial_store
        }
    
    def release_energy(self, amount, purpose):
        """
        Release previously allocated energy back to available pool
        
        Args:
            amount: Amount of BEU to release
            purpose: Purpose identifier
            
        Returns:
            Success status and release details
        """
        if not self.initialized:
            return {"success": False, "error": "Energy system not initialized"}
        
        if amount <= 0:
            return {"success": False, "error": "Invalid energy amount"}
        
        # Check if this purpose has allocated energy
        if purpose not in self.energy_allocated or self.energy_allocated[purpose] < amount:
            return {"success": False, "error": "Energy not previously allocated for this purpose"}
        
        # Release energy
        self.energy_allocated[purpose] -= amount
        self.available_level += amount
        
        logger.info(f"Released {amount:.2E} BEU from {purpose}")
        return {
            "success": True,
            "amount": amount,
            "purpose": purpose,
            "remaining_allocated": self.energy_allocated[purpose],
            "available_level": self.available_level
        }
    
    def transfer_to_mycelial(self, amount):
        """Transfer energy from available pool to mycelial store"""
        if not self.initialized:
            return {"success": False, "error": "Energy system not initialized"}
        
        if amount <= 0:
            return {"success": False, "error": "Invalid energy amount"}
        
        if amount > self.available_level:
            return {"success": False, "error": "Insufficient available energy"}
        
        self.available_level -= amount
        self.mycelial_store += amount
        
        logger.info(f"Transferred {amount:.2E} BEU to mycelial store")
        return {
            "success": True,
            "amount": amount,
            "available_level": self.available_level,
            "mycelial_store": self.mycelial_store
        }
    
    def get_energy_for_search(self, node_count, complexity):
        """
        Calculate and allocate energy needed for search operation
        
        Args:
            node_count: Estimated number of nodes to search
            complexity: Search complexity factor (0-1)
            
        Returns:
            Success status and energy allocation
        """
        # Calculate based on nodes and complexity
        base_energy = node_count * 0.01  # Base energy per node
        complexity_factor = 1.0 + complexity * 2.0  # Scale with complexity
        total_energy = base_energy * complexity_factor
        
        # Allocate the energy
        allocation = self.allocate_energy(total_energy, "search")
        
        return allocation
    
    def get_energy_status(self):
        """Get current energy system status"""
        return {
            "initialized": self.initialized,
            "total_beu": self.total_beu,
            "mycelial_store": self.mycelial_store,
            "operational_level": self.operational_level,
            "available_level": self.available_level,
            "allocated": self.energy_allocated,
            "consumption_rate_beu_per_second": self.energy_consumption_rate,
            "creation_time": self.creation_time
        }