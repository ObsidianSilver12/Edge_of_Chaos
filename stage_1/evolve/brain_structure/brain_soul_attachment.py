# --- START OF FILE stage_1/evolve/brain_soul_attachment.py ---

"""
Brain-Soul Attachment Module (V4.5.0 - Wave Physics & Field Dynamics)

Handles the attachment of a soul to a brain structure after sufficient complexity.
Places the soul in the limbic/brain stem region with proper dimensioning.
Establishes energy connections and distributes soul aspects to memory fragments.
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
from constants.constants import *

# --- Logging ---
logger = logging.getLogger('BrainSoulAttachment')
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


class BrainSoulAttachment:
    """
    Handles attachment of a soul to a brain structure after sufficient complexity.
    """
    
    def __init__(self):
        """Initialize the brain-soul attachment handler."""
        self.attachment_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        
        # Attachment state
        self.attachment_completed = False
        self.brain_id = None
        self.soul_id = None
        self.soul_position = None
        self.attachment_metrics = {}
        
        logger.info("BrainSoulAttachment initialized")
    
    def attach_soul_to_brain_method(self, soul_spark, brain_structure, brain_seed) -> Dict[str, Any]:
        """
        Attach a soul to a brain structure through the brain seed.
        
        Args:
            soul_spark: The soul spark to attach
            brain_structure: The brain structure
            brain_seed: The brain seed (integration point)
            
        Returns:
            Dict containing attachment metrics
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If attachment fails
        """
        # Validate inputs
        if not hasattr(soul_spark, 'soul_id'):
            raise ValueError("Invalid soul_spark object. Missing soul_id.")
            
        if not hasattr(brain_structure, 'brain_id'):
            raise ValueError("Invalid brain_structure object. Missing brain_id.")
            
        if not hasattr(brain_seed, 'seed_id'):
            raise ValueError("Invalid brain_seed object. Missing seed_id.")
        
        self.brain_id = brain_structure.brain_id
        self.soul_id = soul_spark.soul_id
        
        logger.info(f"Attaching soul {soul_spark.soul_id} to brain {brain_structure.brain_id}")
        
        # Initialize metrics
        attachment_metrics = {
            'attachment_id': self.attachment_id,
            'soul_id': soul_spark.soul_id,
            'brain_id': brain_structure.brain_id,
            'seed_id': brain_seed.seed_id,
            'attachment_start': datetime.now().isoformat(),
            'attachment_phases': {}
        }
        
        try:
            # Check brain complexity
            if not self._check_brain_complexity(brain_structure):
                raise ValueError("Brain structure lacks sufficient complexity for soul attachment.")
            
            # Phase 1: Find optimal position for soul placement in limbic/brain stem region
            soul_position = self._find_soul_placement_region(brain_structure, brain_seed)
            self.soul_position = soul_position
            
            # Verify soul size compatibility with region
            region_size = self._get_region_size(brain_structure, soul_position)
            if not self._is_soul_compatible_with_region(soul_spark, region_size):
                raise ValueError(f"Soul size incompatible with region size {region_size}")
            
            # Phase 1 metrics
            sx, sy, sz = soul_position
            region_at_position = brain_structure.region_grid[sx, sy, sz]
            subregion_at_position = brain_structure.sub_region_grid[sx, sy, sz]
            
            phase1_metrics = {
                'soul_position': soul_position,
                'region': region_at_position,
                'subregion': subregion_at_position if subregion_at_position else "none",
                'region_size': region_size,
                'soul_size': self._get_soul_size(soul_spark),
                'position_field_properties': {
                    'energy': float(brain_structure.energy_grid[sx, sy, sz]),
                    'frequency': float(brain_structure.frequency_grid[sx, sy, sz]),
                    'stability': float(brain_structure.stability_grid[sx, sy, sz]),
                    'coherence': float(brain_structure.coherence_grid[sx, sy, sz]),
                    'resonance': float(brain_structure.resonance_grid[sx, sy, sz]),
                    'soul_presence': float(brain_structure.soul_presence_grid[sx, sy, sz])
                }
            }
            attachment_metrics['attachment_phases']['soul_placement'] = phase1_metrics
            
            # Phase 2: Create soul container in brain
            soul_container_metrics = self._create_soul_container(soul_spark, brain_structure, soul_position, region_size)
            attachment_metrics['attachment_phases']['soul_container'] = soul_container_metrics
            
            # Phase 3: Connect life cord from soul to brain
            life_cord_metrics = self._connect_life_cord(soul_spark, brain_structure, brain_seed, soul_position)
            attachment_metrics['attachment_phases']['life_cord'] = life_cord_metrics
            
            # Phase 4: Distribute soul aspects to temporal region as memory fragments
            aspect_distribution_metrics = self._distribute_soul_aspects(soul_spark, brain_structure, brain_seed)
            attachment_metrics['attachment_phases']['aspect_distribution'] = aspect_distribution_metrics
            
            # Phase 5: Establish energy connection between soul and brain
            energy_connection_metrics = {
                'connection_strength': 1.0,  # Default connection strength
                'connection_status': 'established'
            }
            attachment_metrics['attachment_phases']['energy_connection'] = energy_connection_metrics
            
            # Update brain seed with soul connection information
            brain_seed.soul_connection = {
                'soul_id': soul_spark.soul_id,
                'connection_established': True,
                'connection_time': datetime.now().isoformat(),
                'connection_strength': energy_connection_metrics['connection_strength'],
                'soul_position': soul_position,
                'life_cord_active': life_cord_metrics['life_cord_active']
            }
            
            # Mark attachment as complete
            self.attachment_completed = True
            self.last_updated = datetime.now().isoformat()
            self.attachment_metrics = attachment_metrics
            
            # Final metrics
            attachment_metrics['attachment_end'] = datetime.now().isoformat()
            attachment_metrics['success'] = True
            
            logger.info(f"Soul attachment completed successfully at position {soul_position}")
            return attachment_metrics
            
        except Exception as e:
            logger.error(f"Error during soul attachment: {e}", exc_info=True)
            attachment_metrics['attachment_end'] = datetime.now().isoformat()
            attachment_metrics['success'] = False
            attachment_metrics['error'] = str(e)
            
            # Store failed metrics
            self.attachment_metrics = attachment_metrics
            
            raise RuntimeError(f"Soul attachment failed: {e}")        
        
    def _check_brain_complexity(self, brain_structure) -> bool:
        """
        Check if brain has sufficient complexity for soul attachment.
        
        Args:
            brain_structure: The brain structure
            
        Returns:
            True if brain has sufficient complexity, False otherwise
        """
        # For minimal brain seed attachment, we just need basic structure
        # Check if brain_structure has the minimal required attributes
        required_attrs = ['dimensions', 'region_grid', 'energy_grid', 'frequency_grid', 
                         'resonance_grid', 'coherence_grid', 'soul_presence_grid']
        
        for attr in required_attrs:
            if not hasattr(brain_structure, attr):
                logger.warning(f"Brain structure missing required attribute: {attr}")
                return False
        
        # Check if brain has energy
        if hasattr(brain_structure, 'energy_grid'):
            avg_energy = np.mean(brain_structure.energy_grid)
            if avg_energy < 0.01:  # Very minimal energy requirement
                logger.warning(f"Insufficient energy in brain structure: {avg_energy:.3f}")
                return False
        
        logger.info("Brain structure meets minimal complexity requirements for soul attachment")
        return True
    
    def _find_soul_placement_region(self, brain_structure, brain_seed) -> Tuple[int, int, int]:
        """
        Find the optimal region for soul placement.
        For minimal brain seed, use seed position or brain center.
        
        Args:
            brain_structure: The brain structure
            brain_seed: The brain seed (used for field interactions)
            
        Returns:
            Tuple of (x, y, z) coordinates for soul placement
        """
        logger.info("Finding optimal soul placement region")
        
        # For minimal brain seed, use seed position if available
        if hasattr(brain_seed, 'position') and brain_seed.position:
            logger.info(f"Using brain seed position: {brain_seed.position}")
            return brain_seed.position
        
        # Otherwise use center of brain structure
        center = (brain_structure.dimensions[0] // 2,
                 brain_structure.dimensions[1] // 2,
                 brain_structure.dimensions[2] // 2)
        logger.info(f"Using brain center position: {center}")
        return center
    
    def _get_region_size(self, brain_structure, position: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Get the size of the region at the specified position.
        For minimal brain seed, return reasonable default size.
        
        Args:
            brain_structure: The brain structure
            position: Position to get region size for
            
        Returns:
            Tuple of (x_size, y_size, z_size) for region dimensions
        """
        # For minimal brain seed, return default size based on brain dimensions
        return (min(64, brain_structure.dimensions[0]),
                min(64, brain_structure.dimensions[1]),
                min(64, brain_structure.dimensions[2]))
    
    def _get_soul_size(self, soul_spark) -> Tuple[int, int, int]:
        """
        Get the size of the soul spark.
        
        Args:
            soul_spark: The soul spark
            
        Returns:
            Tuple of (x_size, y_size, z_size) for soul dimensions
        """
        # If soul has explicit size, use it
        if hasattr(soul_spark, 'size'):
            return soul_spark.size
        
        # Default: use a 16x16x16 cube size for minimal attachment
        return (16, 16, 16)
    
    def _is_soul_compatible_with_region(self, soul_spark, region_size: Tuple[int, int, int]) -> bool:
        """
        Check if soul size is compatible with region size.
        
        Args:
            soul_spark: The soul spark
            region_size: Region size tuple
            
        Returns:
            True if compatible, False otherwise
        """
        soul_size = self._get_soul_size(soul_spark)
        
        # Soul should fit within region
        return (soul_size[0] <= region_size[0] and
                soul_size[1] <= region_size[1] and
                soul_size[2] <= region_size[2])
    
    def _create_soul_container(self, soul_spark, brain_structure, 
                             soul_position: Tuple[int, int, int], 
                             region_size: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Create a minimal soul container in the brain structure.
        
        Args:
            soul_spark: The soul spark
            brain_structure: The brain structure
            soul_position: Position for soul container
            region_size: Size of the region
            
        Returns:
            Dict with soul container metrics
        """
        logger.info(f"Creating minimal soul container at position {soul_position}")
        
        # Get soul size
        soul_size = self._get_soul_size(soul_spark)
        sx, sy, sz = soul_position
        
        # Soul container should be centered at soul position
        half_x, half_y, half_z = soul_size[0] // 2, soul_size[1] // 2, soul_size[2] // 2
        
        # Ensure container is within brain bounds
        x_min = max(0, sx - half_x)
        x_max = min(brain_structure.dimensions[0] - 1, sx + half_x)
        y_min = max(0, sy - half_y)
        y_max = min(brain_structure.dimensions[1] - 1, sy + half_y)
        z_min = max(0, sz - half_z)
        z_max = min(brain_structure.dimensions[2] - 1, sz + half_z)
        
        # Calculate actual container size
        container_size = (
            x_max - x_min + 1,
            y_max - y_min + 1,
            z_max - z_min + 1
        )
        
        # Set up soul field properties
        soul_frequency = getattr(soul_spark, 'frequency', DEFAULT_BRAIN_SEED_FREQUENCY)
        
        # Fill container with soul presence
        cells_filled = 0
        
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for z in range(z_min, z_max + 1):
                    # Calculate distance from center
                    dx, dy, dz = x - sx, y - sy, z - sz
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Set soul presence with falloff
                    if dist <= max(soul_size) // 2:
                        falloff = 1.0 - (dist / (max(soul_size) // 2 + 0.1))
                        
                        # Set soul presence
                        brain_structure.soul_presence_grid[x, y, z] = max(
                            brain_structure.soul_presence_grid[x, y, z],
                            0.8 * falloff
                        )
                        
                        # Set soul frequency
                        if hasattr(brain_structure, 'soul_frequency_grid'):
                            brain_structure.soul_frequency_grid[x, y, z] = soul_frequency
                        
                        cells_filled += 1
        
        logger.info(f"Minimal soul container created with {cells_filled} cells filled")
        
        # Return metrics
        return {
            'container_position': soul_position,
            'container_size': container_size,
            'soul_size': soul_size,
            'cells_filled': cells_filled,
            'soul_frequency': float(soul_frequency)
        }
    
    def _connect_life_cord(self, soul_spark, brain_structure, brain_seed, 
                         soul_position: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Connect the life cord from soul to brain (minimal version).
        
        Args:
            soul_spark: The soul spark
            brain_structure: The brain structure
            brain_seed: The brain seed
            soul_position: Position of soul container
            
        Returns:
            Dict with life cord metrics
        """
        logger.info(f"Connecting life cord from soul to brain seed")
        
        # Extract life cord data from soul if available
        life_cord_data = {}
        if hasattr(soul_spark, 'life_cord') and isinstance(soul_spark.life_cord, dict):
            life_cord_data = soul_spark.life_cord
        else:
            # Create minimal life cord data
            life_cord_data = {
                'primary_frequency_hz': getattr(soul_spark, 'frequency', DEFAULT_BRAIN_SEED_FREQUENCY),
                'soul_to_earth_efficiency': 0.8,
                'earth_to_soul_efficiency': 0.8,
                'quantum_efficiency': 0.9
            }
        
        # Store life cord data in brain seed if it has storage method
        if hasattr(brain_seed, 'store_life_cord_data'):
            brain_seed.store_life_cord_data(life_cord_data)
        else:
            # Store as simple attribute
            brain_seed.life_cord_data = life_cord_data
        
        life_cord_active = True
        
        logger.info("Life cord connected successfully")
        
        # Return metrics
        return {
            'life_cord_active': life_cord_active,
            'cord_type': 'minimal',
            'efficiency': 0.8,
            'primary_frequency': float(life_cord_data['primary_frequency_hz'])
        }
    
    def _distribute_soul_aspects(self, soul_spark, brain_structure, brain_seed) -> Dict[str, Any]:
        """
        Distribute soul aspects to memory system (minimal version).
        
        Args:
            soul_spark: The soul spark
            brain_structure: The brain structure
            brain_seed: The brain seed
            
        Returns:
            Dict with aspect distribution metrics
        """
        logger.info("Distributing soul aspects to memory system")
        
        # Extract soul aspects if available
        soul_aspects = {}
        if hasattr(soul_spark, 'aspects') and isinstance(soul_spark.aspects, dict):
            soul_aspects = soul_spark.aspects
        
        if not soul_aspects:
            logger.info("No soul aspects found for distribution")
            return {
                'aspects_distributed': 0,
                'fragments_created': 0,
                'distribution_success': True,
                'message': "No aspects to distribute"
            }
        
        aspects_distributed = 0
        fragments_created = 0
        
        # For minimal version, just store aspects in brain seed
        for aspect_id, aspect_data in soul_aspects.items():
            try:
                # Convert aspect data to fragment format
                if isinstance(aspect_data, dict):
                    aspect_content = json.dumps(aspect_data)
                else:
                    aspect_content = str(aspect_data)
                
                # Store in brain seed if it has memory fragment methods
                if hasattr(brain_seed, 'add_memory_fragment'):
                    fragment_id = brain_seed.add_memory_fragment(
                        fragment_content=aspect_content,
                        fragment_frequency=getattr(soul_spark, 'frequency', DEFAULT_BRAIN_SEED_FREQUENCY),
                        fragment_meta={'aspect_id': aspect_id, 'origin': 'soul'}
                    )
                    fragments_created += 1
                else:
                    # Store as simple attribute
                    if not hasattr(brain_seed, 'soul_aspects'):
                        brain_seed.soul_aspects = {}
                    brain_seed.soul_aspects[aspect_id] = aspect_data
                
                aspects_distributed += 1
                logger.debug(f"Aspect {aspect_id} distributed successfully")
                
            except Exception as e:
                logger.warning(f"Failed to distribute aspect {aspect_id}: {e}")
        
        logger.info(f"Soul aspect distribution complete: {aspects_distributed} aspects distributed")
        
        # Return metrics
        return {
            'aspects_distributed': aspects_distributed,
            'fragments_created': fragments_created,
            'distribution_success': aspects_distributed > 0,
            'message': f"Distributed {aspects_distributed} aspects"
        }


# --- Standalone Functions for Birth Module ---

def attach_soul_to_brain(soul_spark, brain_seed) -> Dict[str, Any]:
    """
    Standalone function to attach soul to brain through brain seed.
    This is a simplified version for the birth process.
    
    Args:
        soul_spark: The soul spark to attach
        brain_seed: The brain seed (acts as minimal brain structure)
        
    Returns:
        Dict containing attachment metrics
    """
    logger.info(f"Attaching soul {getattr(soul_spark, 'spark_id', 'unknown')} to minimal brain seed")
    
    try:
        # For minimal brain seed attachment, we create a simplified structure
        # Create a minimal brain structure object that has the required attributes
        class MinimalBrainStructure:
            def __init__(self, dimensions=(64, 64, 64)):
                self.dimensions = dimensions
                self.brain_id = getattr(brain_seed, 'seed_id', 'minimal_brain')
                
                # Initialize minimal grids
                self.region_grid = np.full(dimensions, 'core', dtype=object)
                self.sub_region_grid = np.full(dimensions, 'center', dtype=object)
                self.energy_grid = np.ones(dimensions) * 0.1
                self.frequency_grid = np.ones(dimensions) * getattr(soul_spark, 'frequency', DEFAULT_BRAIN_SEED_FREQUENCY)
                self.resonance_grid = np.ones(dimensions) * 0.5
                self.coherence_grid = np.ones(dimensions) * 0.5
                self.soul_presence_grid = np.zeros(dimensions)
                
                # Optional grids
                if not hasattr(self, 'soul_frequency_grid'):
                    self.soul_frequency_grid = np.zeros(dimensions)
        
        # Create minimal brain structure
        brain_structure = MinimalBrainStructure()
        
        # Use the attachment class
        attachment_handler = BrainSoulAttachment()
        
        # Perform attachment
        attachment_metrics = attachment_handler.attach_soul_to_brain_method(
            soul_spark, brain_structure, brain_seed)
        
        # Store brain connection info in soul
        if hasattr(soul_spark, '__dict__'):
            soul_spark.brain_connection = {
                'brain_seed_id': getattr(brain_seed, 'seed_id', 'unknown'),
                'attachment_time': datetime.now().isoformat(),
                'attachment_success': attachment_metrics.get('success', False)
            }
        
        logger.info("Soul attachment to minimal brain seed completed successfully")
        return attachment_metrics
        
    except Exception as e:
        logger.error(f"Error in standalone attach_soul_to_brain: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'attachment_time': datetime.now().isoformat()
        }


def distribute_soul_aspects(soul_spark, brain_seed) -> Dict[str, Any]:
    """
    Standalone function to distribute soul aspects to brain seed memory system.
    
    Args:
        soul_spark: The soul spark with aspects
        brain_seed: The brain seed to store aspects in
        
    Returns:
        Dict containing distribution metrics
    """
    logger.info(f"Distributing soul aspects to brain seed memory system")
    
    try:
        # Extract soul aspects if available
        soul_aspects = {}
        if hasattr(soul_spark, 'aspects') and isinstance(soul_spark.aspects, dict):
            soul_aspects = soul_spark.aspects
        
        if not soul_aspects:
            logger.info("No soul aspects found for distribution")
            return {
                'aspects_distributed': 0,
                'fragments_created': 0,
                'distribution_success': True,
                'message': "No aspects to distribute"
            }
        
        aspects_distributed = 0
        fragments_created = 0
        
        # Distribute aspects to brain seed
        for aspect_id, aspect_data in soul_aspects.items():
            try:
                # Convert aspect data to storable format
                if isinstance(aspect_data, dict):
                    aspect_content = json.dumps(aspect_data)
                else:
                    aspect_content = str(aspect_data)
                
                # Store in brain seed
                if hasattr(brain_seed, 'add_memory_fragment'):
                    # Use brain seed's memory fragment system
                    fragment_id = brain_seed.add_memory_fragment(
                        fragment_content=aspect_content,
                        fragment_frequency=getattr(soul_spark, 'frequency', DEFAULT_BRAIN_SEED_FREQUENCY),
                        fragment_meta={'aspect_id': aspect_id, 'origin': 'soul'}
                    )
                    fragments_created += 1
                    logger.debug(f"Created memory fragment {fragment_id} for aspect {aspect_id}")
                else:
                    # Store as simple attribute
                    if not hasattr(brain_seed, 'soul_aspects'):
                        brain_seed.soul_aspects = {}
                    brain_seed.soul_aspects[aspect_id] = aspect_data
                    logger.debug(f"Stored aspect {aspect_id} as attribute")
                
                aspects_distributed += 1
                
            except Exception as e:
                logger.warning(f"Failed to distribute aspect {aspect_id}: {e}")
        
        # Create associations between aspects if brain seed supports it
        if hasattr(brain_seed, 'associate_fragments') and fragments_created > 1:
            try:
                # Simple sequential association
                fragment_ids = []
                if hasattr(brain_seed, 'memory_fragments'):
                    fragment_ids = list(brain_seed.memory_fragments.keys())[-fragments_created:]
                
                for i in range(len(fragment_ids) - 1):
                    brain_seed.associate_fragments(
                        fragment_ids[i], fragment_ids[i+1], 0.5)
                
            except Exception as e:
                logger.warning(f"Failed to create aspect associations: {e}")
        
        logger.info(f"Soul aspect distribution complete: {aspects_distributed} aspects, {fragments_created} fragments")
        
        return {
            'aspects_distributed': aspects_distributed,
            'fragments_created': fragments_created,
            'distribution_success': aspects_distributed > 0,
            'aspect_count': len(soul_aspects)
        }
        
    except Exception as e:
        logger.error(f"Error in standalone distribute_soul_aspects: {e}", exc_info=True)
        return {
            'aspects_distributed': 0,
            'fragments_created': 0,
            'distribution_success': False,
            'error': str(e)
        }


# --- END OF FILE stage_1/evolve/brain_soul_attachment.py ---