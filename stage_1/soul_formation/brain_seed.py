# --- START OF FILE stage_2/brain_development/brain_seed.py ---

"""
brain_seed.py - Minimal Brain Seed Placeholder (V4.3.8+ Simplified)

Acts as the initial energetic anchor point for the soul in the physical realm.
Holds converted Brain Energy Units (BEU) allocated during birth.
Detailed structure and development are deferred.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import json
import numpy as np # Keep for potential save/load compatibility

# --- Logging ---
logger = logging.getLogger('BrainSeed')
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants (Assume imported or defined globally) ---
try:
    from constants.constants import FLOAT_EPSILON, BRAIN_ENERGY_UNIT_PER_JOULE # Need these
except ImportError:
    logging.error("BrainSeed using fallback constants.")
    FLOAT_EPSILON = 1e-9; BRAIN_ENERGY_UNIT_PER_JOULE = 1e12

class BrainSeed:
    """ Minimal Brain Seed placeholder. Stores initial energy and connection info. """

    def __init__(self, resonant_soul_id: str, initial_beu: float = 0.0, initial_mycelial_beu: float = 0.0):
        """
        Initialize the minimal brain seed.

        Args:
            resonant_soul_id (str): The ID of the soul attaching.
            initial_beu (float): Initial BEU allocated to the core/operational pool.
            initial_mycelial_beu (float): Initial BEU allocated to the mycelial store.
        """
        if not isinstance(resonant_soul_id, str) or not resonant_soul_id:
            raise ValueError("resonant_soul_id must be a non-empty string.")
        if not isinstance(initial_beu, (int, float)) or initial_beu < 0:
            raise ValueError("initial_beu must be non-negative.")
        if not isinstance(initial_mycelial_beu, (int, float)) or initial_mycelial_beu < 0:
            raise ValueError("initial_mycelial_beu must be non-negative.")

        self.resonant_soul_id: str = resonant_soul_id
        self.creation_time: str = datetime.now().isoformat()
        self.last_updated: str = self.creation_time

        # --- Core State ---
        self.base_energy_level: float = initial_beu # BEU available for immediate use
        self.mycelial_energy_store: float = initial_mycelial_beu # BEU reserve

        # --- Placeholders for Future Development ---
        self.complexity: int = 1 # Minimal complexity for now
        self.formation_progress: float = 0.1 # Represents seed presence
        self.stability: float = 0.5 # Default stability
        self.structural_integrity: float = 0.5 # Default integrity
        self.soul_connection: Optional[Dict[str, Any]] = None # Link to soul
        self.soul_aspect_distribution: Optional[Dict[str, Any]] = {} # Store distributed aspects here

        logger.info(f"Minimal BrainSeed created for Soul {self.resonant_soul_id}. Initial BEU: {self.base_energy_level:.2E}, Mycelial BEU: {self.mycelial_energy_store:.2E}")

    def get_metrics(self) -> Dict[str, Any]:
        """ Returns current metrics of the minimal brain seed. """
        return {
            'resonant_soul_id': self.resonant_soul_id,
            'energy_level_beu': self.base_energy_level,
            'mycelial_energy_store_beu': self.mycelial_energy_store,
            'complexity': self.complexity,
            'formation_progress': self.formation_progress,
            'stability': self.stability,
            'structural_integrity': self.structural_integrity,
            'last_updated': self.last_updated,
            'soul_connected': self.soul_connection is not None,
            'aspects_distributed_count': len(self.soul_aspect_distribution) if self.soul_aspect_distribution else 0
        }

    def __str__(self) -> str:
        return f"BrainSeed(SoulID: {self.resonant_soul_id}, BEU: {self.base_energy_level:.2E}, MycelialBEU: {self.mycelial_energy_store:.2E})"

    def __repr__(self) -> str:
        return f"<BrainSeed id='{self.resonant_soul_id}' energy={self.base_energy_level:.1E}>"

    def save_state(self, file_path: str):
        """ Saves the minimal state. Raises IOError if fails. """
        logger.info(f"Saving minimal BrainSeed state to {file_path}")
        try:
            state = self.__dict__.copy()
            state_serializable = json.loads(json.dumps(state, default=str)) # Basic serialization
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f: json.dump(state_serializable, f, indent=2)
            logger.info("Minimal BrainSeed state saved successfully."); return True
        except Exception as e: logger.error(f"Error saving minimal BrainSeed state: {e}", exc_info=True); raise IOError(f"Failed save minimal BrainSeed state: {e}") from e

    @classmethod
    def load_state(cls, file_path: str):
        """ Loads minimal state. Raises error if fails. """
        logger.info(f"Loading minimal BrainSeed state from {file_path}")
        if not os.path.exists(file_path): raise FileNotFoundError(f"BrainSeed state file not found: {file_path}")
        try:
            with open(file_path,'r') as f: state = json.load(f)
            if not isinstance(state, dict) or 'resonant_soul_id' not in state: raise ValueError("Invalid or incomplete state data.")
            # Recreate instance with loaded data
            instance = cls(resonant_soul_id=state['resonant_soul_id'])
            for key, value in state.items():
                if hasattr(instance, key): setattr(instance, key, value)
                else: logger.warning(f"Attr '{key}' from save file not in minimal BrainSeed. Skipping.")
            logger.info(f"Minimal BrainSeed state loaded for Soul {instance.resonant_soul_id}.")
            return instance
        except Exception as e: logger.error(f"Error loading minimal BrainSeed state: {e}", exc_info=True); raise RuntimeError(f"Failed load minimal BrainSeed state: {e}") from e


# --- Factory Function (Simplified) ---
def create_brain_seed(resonant_soul, initial_beu: float = 0.0, initial_mycelial_beu: float = 0.0) -> BrainSeed:
    """ Creates a minimal BrainSeed instance. Fails hard. """
    logger.info(f"Factory: Creating minimal brain seed for Soul {resonant_soul.spark_id}")
    try:
        if not hasattr(resonant_soul, 'spark_id'): raise TypeError("resonant_soul object missing required attributes.")
        brain_seed = BrainSeed(resonant_soul.spark_id, initial_beu, initial_mycelial_beu)
        logger.info(f"Factory: Minimal brain seed created.")
        return brain_seed
    except Exception as e:
        logger.error(f"Factory: Failed to create minimal brain seed: {e}", exc_info=True)
        raise RuntimeError("Minimal brain seed factory creation failed.") from e

# --- END OF FILE stage_2/brain_development/brain_seed.py ---