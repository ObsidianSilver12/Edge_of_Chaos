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
from shared.constants.constants import *

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

def attach_soul_to_brain(soul_spark, brain_structure, enhanced_network=None):
    """
    Attach soul to brain using current architecture.
    
    Args:
        soul_spark: The soul spark to attach
        brain_structure: HybridBrainStructure instance
        enhanced_network: Enhanced mycelial network for coordination
    """
    logger.info(f"Attaching soul {getattr(soul_spark, 'soul_id', 'unknown')} to brain structure")
    
    try:
        # Find optimal position using brain structure's complexity analysis
        soul_position = find_optimal_soul_position(brain_structure)
        
        # Create soul container in brain structure (sparse storage)
        container_result = create_soul_container_sparse(soul_spark, brain_structure, soul_position)
        
        # Connect life cord
        life_cord_result = connect_life_cord_simple(soul_spark, brain_structure, soul_position)
        
        # Distribute soul aspects to memory system
        if enhanced_network and hasattr(enhanced_network, 'memory_fragment_system'):
            aspect_result = distribute_soul_aspects_enhanced(
                soul_spark, enhanced_network.memory_fragment_system, enhanced_network
            )
        else:
            aspect_result = {'aspects_distributed': 0, 'message': 'No memory system available'}
        
        # Store soul connection info
        if hasattr(soul_spark, '__dict__'):
            soul_spark.brain_connection = {
                'brain_id': brain_structure.brain_id,
                'position': soul_position,
                'attachment_time': datetime.now().isoformat(),
                'container_cells': container_result.get('cells_filled', 0),
                'life_cord_active': life_cord_result.get('active', False)
            }
        
        logger.info("Soul attachment completed successfully")
        
        return {
            'success': True,
            'soul_position': soul_position,
            'container_result': container_result,
            'life_cord_result': life_cord_result,
            'aspect_distribution': aspect_result
        }
        
    except Exception as e:
        logger.error(f"Soul attachment failed: {e}")
        return {'success': False, 'error': str(e)}


def find_optimal_soul_position(brain_structure):
    """Find optimal soul position using brain structure's existing systems."""
    # Use brain structure's existing limbic region center
    if hasattr(brain_structure, 'regions') and 'limbic' in brain_structure.regions:
        return brain_structure.regions['limbic']['center']
    
    # Fallback to brain center
    return (
        brain_structure.dimensions[0] // 2,
        brain_structure.dimensions[1] // 2,
        brain_structure.dimensions[2] // 2
    )


def create_soul_container_sparse(soul_spark, brain_structure, position):
    """Create soul container using sparse storage."""
    x, y, z = position
    soul_size = (16, 16, 16)  # Default soul container size
    cells_filled = 0
    
    # Use brain structure's sparse active_cells storage
    for dx in range(-soul_size[0]//2, soul_size[0]//2):
        for dy in range(-soul_size[1]//2, soul_size[1]//2):
            for dz in range(-soul_size[2]//2, soul_size[2]//2):
                nx, ny, nz = x + dx, y + dy, z + dz
                
                if (0 <= nx < brain_structure.dimensions[0] and 
                    0 <= ny < brain_structure.dimensions[1] and 
                    0 <= nz < brain_structure.dimensions[2]):
                    
                    # Add to sparse active cells
                    cell_position = (nx, ny, nz)
                    if cell_position not in brain_structure.active_cells:
                        brain_structure.active_cells[cell_position] = {}
                    
                    # Set soul properties
                    brain_structure.active_cells[cell_position].update({
                        'soul_presence': 0.8,
                        'frequency': getattr(soul_spark, 'frequency', 7.83),
                        'energy': 0.5,
                        'soul_container': True
                    })
                    
                    # Assign region
                    brain_structure.region_assignments[cell_position] = 'limbic'
                    cells_filled += 1
    
    return {
        'success': True,
        'cells_filled': cells_filled,
        'position': position,
        'size': soul_size
    }


def connect_life_cord_simple(soul_spark, brain_structure, position):
    """Connect life cord using current architecture."""
    life_cord_data = getattr(soul_spark, 'life_cord', {
        'primary_frequency_hz': getattr(soul_spark, 'frequency', 7.83),
        'efficiency': 0.8
    })
    
    # Store life cord connection in brain structure
    if hasattr(brain_structure, 'soul_connections'):
        brain_structure.soul_connections = {}
    
    brain_structure.soul_connections[position] = {
        'soul_id': getattr(soul_spark, 'soul_id', 'unknown'),
        'life_cord_data': life_cord_data,
        'connection_time': datetime.now().isoformat()
    }
    
    return {
        'active': True,
        'frequency': life_cord_data['primary_frequency_hz'],
        'efficiency': life_cord_data.get('efficiency', 0.8)
    }


def distribute_soul_aspects_enhanced(soul_spark, memory_system, enhanced_network):
    """Distribute soul aspects using enhanced network coordination."""
    if not hasattr(soul_spark, 'aspects'):
        return {'aspects_distributed': 0, 'message': 'No soul aspects found'}
    
    aspects_distributed = 0
    fragments_created = []
    
    for aspect_id, aspect_data in soul_spark.aspects.items():
        try:
            # Use enhanced network for optimal placement
            if hasattr(enhanced_network, 'coordinate_memory_placement'):
                placement_result = enhanced_network.coordinate_memory_placement(aspect_data)
                region = placement_result.get('region', 'temporal')
                position = placement_result.get('position')
            else:
                region = 'temporal'
                position = None
            
            # Create memory fragment
            fragment_id = memory_system.add_fragment(
                content=aspect_data,
                region=region,
                position=position,
                frequency=getattr(soul_spark, 'frequency', 7.83),
                meta_tags={'aspect_id': aspect_id, 'origin': 'soul'},
                origin='soul'
            )
            
            fragments_created.append(fragment_id)
            aspects_distributed += 1
            
        except Exception as e:
            logger.warning(f"Failed to distribute aspect {aspect_id}: {e}")
    
    return {
        'aspects_distributed': aspects_distributed,
        'fragments_created': fragments_created,
        'total_aspects': len(soul_spark.aspects)
    }


# --- END OF FILE stage_1/evolve/brain_soul_attachment.py ---