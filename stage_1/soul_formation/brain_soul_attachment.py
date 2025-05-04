# --- START OF FILE stage_2/brain_development/brain_soul_attachment.py ---

"""
brain_soul_attachment.py - Module for connecting the soul to the brain. V4.3.8+ Simplified

Handles basic attachment via life cord structure.
Distributes aspects conceptually to the minimal BrainSeed.
Hard fails on critical errors. Assumes constants are available.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# --- Logging ---
logger = logging.getLogger('BrainSoulAttachment')
if not logger.handlers:
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
try:
    from constants.constants import FLOAT_EPSILON, LIFE_CORD_FREQUENCIES # Minimal constants
    # Import resonance calculation function if needed for connection strength?
    # from stage_1.soul_formation.creator_entanglement import calculate_resonance # Optional
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Imports failed: {e}. BrainSoulAttachment cannot function.")
    raise ImportError(f"Essential dependencies missing: {e}") from e
except NameError as e:
    logging.critical(f"CRITICAL ERROR: Missing required constant definition: {e}. BrainSoulAttachment cannot function.")
    raise NameError(f"Essential constant missing: {e}") from e

# --- Dependency Imports ---
try:
    from .brain_seed import BrainSeed
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

# --- Simplified Attachment Function ---
def attach_soul_to_brain(soul_spark: SoulSpark, brain_seed: BrainSeed) -> Dict[str, Any]:
    """ Establishes a basic link between SoulSpark life_cord and minimal BrainSeed. Fails hard. """
    logger.info(f"Attaching soul {soul_spark.spark_id} to minimal brain seed...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(brain_seed, BrainSeed): raise TypeError("brain_seed invalid.")

    life_cord_data = getattr(soul_spark, 'life_cord', None)
    if not isinstance(life_cord_data, dict): msg = "SoulSpark missing valid life_cord."; logger.error(msg); raise AttributeError(msg)
    cord_integrity = soul_spark.cord_integrity
    BIRTH_ATTACHMENT_MIN_CORD_INTEGRITY = 0.75 # Define or get from constants
    if cord_integrity < BIRTH_ATTACHMENT_MIN_CORD_INTEGRITY: msg = f"Cord integrity ({cord_integrity:.3f}) too low (< {BIRTH_ATTACHMENT_MIN_CORD_INTEGRITY})."; logger.error(msg); raise ValueError(msg)

    try:
        connection_timestamp = datetime.now().isoformat()
        # Calculate a simple connection strength based on cord properties
        connection_strength = (cord_integrity * 0.6 +
                               life_cord_data.get('stability_factor', 0.5) * 0.4)
        connection_strength = max(0.0, min(1.0, connection_strength))

        connection = {
            'connection_strength': float(connection_strength),
            'establishment_timestamp': connection_timestamp,
            'brain_seed_id': brain_seed.resonant_soul_id, # Link by ID
            'life_cord_primary_freq_hz': life_cord_data.get('primary_frequency_hz')
        }

        # Store connection details on both objects
        soul_spark.brain_connection = connection.copy()
        brain_seed.soul_connection = connection.copy()
        soul_spark.last_modified = connection_timestamp
        brain_seed.last_updated = connection_timestamp

        logger.info(f"Minimal Soul-brain attachment established. Strength={connection['connection_strength']:.3f}")
        return {'success': True, 'connection_strength': connection['connection_strength'], 'timestamp': connection_timestamp}

    except Exception as e:
        logger.error(f"Error during minimal soul-brain attachment: {e}", exc_info=True)
        raise RuntimeError("Minimal soul-brain attachment failed critically.") from e

# --- Simplified Aspect Distribution ---
def distribute_soul_aspects(soul_spark: SoulSpark, brain_seed: BrainSeed) -> Dict[str, Any]:
    """ Conceptually distributes soul aspects to the minimal BrainSeed. Fails hard. """
    logger.info("Distributing soul aspects conceptually to minimal brain seed...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(brain_seed, BrainSeed): raise TypeError("brain_seed invalid.")
    if not hasattr(brain_seed, 'soul_connection'): raise RuntimeError("Soul-brain connection not established.")

    if not hasattr(soul_spark, 'aspects') or not soul_spark.aspects:
        logger.warning("Soul has no aspects to distribute.")
        brain_seed.soul_aspect_distribution = {} # Ensure attribute exists
        return {'success': True, 'soul_aspects_count': 0, 'integration_level': 0.0}

    try:
        # Store a simplified representation of aspects in the brain seed
        # This doesn't involve complex mapping, just transferring the data conceptually
        aspect_summary = {}
        for name, data in soul_spark.aspects.items():
             aspect_summary[name] = {
                 'strength': data.get('strength', 0.0),
                 # Include retention factor after veil is applied
                 'retention_factor': data.get('retention_factor', 1.0)
             }

        brain_seed.soul_aspect_distribution = aspect_summary # Store summary directly
        integration_level = np.mean([d.get('strength', 0) * d.get('retention_factor', 1) for d in aspect_summary.values()]) if aspect_summary else 0.0
        brain_seed.last_updated = datetime.now().isoformat()

        logger.info(f"Conceptually distributed {len(aspect_summary)} soul aspects to brain seed. Avg Effective Strength: {integration_level:.3f}")
        return {'success': True, 'soul_aspects_count': len(aspect_summary), 'integration_level': float(integration_level)}

    except Exception as e:
        logger.error(f"Error distributing soul aspects: {e}", exc_info=True)
        raise RuntimeError("Soul aspect distribution failed critically.") from e

# --- END OF FILE stage_2/brain_development/brain_soul_attachment.py ---