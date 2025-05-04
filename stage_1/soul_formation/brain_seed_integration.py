# --- brain_seed_integration.py - Integrates brain seed with brain structure and mycelial network ---

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# --- Logging ---
logger = logging.getLogger('BrainSeedIntegration')
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
try:
    from constants.constants import FLOAT_EPSILON, BRAIN_ENERGY_UNIT_PER_JOULE
except ImportError:
    logging.error("BrainSeedIntegration using fallback constants.")
    FLOAT_EPSILON = 1e-9
    BRAIN_ENERGY_UNIT_PER_JOULE = 1e12

class BrainSeedIntegration:
    """
    Handles the integration of a BrainSeed with the brain structure and mycelial network.
    This is part of the one-time process during incarnation.
    """
    
    def __init__(self):
        """Initialize the brain seed integration handler"""
        self.integration_completed = False
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        self.integration_metrics = {}
        
        logger.info("BrainSeedIntegration initialized")
    
    def integrate_seed_with_structure(self, brain_seed, brain_structure, 
                                     seed_position: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Integrate the brain seed with the brain structure.
        
        Args:
            brain_seed: The BrainSeed instance
            brain_structure: The BrainGrid instance
            seed_position: Optional position for seed (if None, optimal position is determined)
            
        Returns:
            Dict containing integration metrics
        """
        logger.info(f"Integrating brain seed {brain_seed.resonant_soul_id} with brain structure")
        
        # Validate inputs
        if not hasattr(brain_seed, 'resonant_soul_id'):
            msg = "Invalid brain_seed object. Missing resonant_soul_id."
            logger.error(msg)
            raise ValueError(msg)
            
        if not hasattr(brain_structure, 'energy_grid'):
            msg = "Invalid brain_structure object. Missing energy_grid."
            logger.error(msg)
            raise ValueError(msg)
        
        # Initialize metrics
        integration_metrics = {
            'seed_id': brain_seed.resonant_soul_id,
            'integration_start': datetime.now().isoformat(),
            'seed_energy_beu': brain_seed.base_energy_level,
            'mycelial_energy_beu': brain_seed.mycelial_energy_store,
            'integration_phases': {}
        }
        
        try:
            # Phase 1: Position seed in brain structure
            if seed_position is None:
                # Find optimal position
                seed_position = brain_structure.find_optimal_seed_position()
                logger.info(f"Optimal seed position determined: {seed_position}")
            
            x, y, z = seed_position
            if not (0 <= x < brain_structure.dimensions[0] and 
                   0 <= y < brain_structure.dimensions[1] and
                   0 <= z < brain_structure.dimensions[2]):
                raise ValueError(f"Seed position {seed_position} out of brain structure bounds")
            
            # Determine brain region at seed position
            seed_region = brain_structure.region_grid[x, y, z]
            seed_subregion = brain_structure.sub_region_grid[x, y, z]
            
            # Phase 1 metrics
            phase1_metrics = {
                'seed_position': seed_position,
                'seed_region': seed_region if seed_region else "unknown",
                'seed_subregion': seed_subregion if seed_subregion else "unknown",
                'position_field_properties': {
                    'energy': float(brain_structure.energy_grid[x, y, z]),
                    'frequency': float(brain_structure.frequency_grid[x, y, z]),
                    'stability': float(brain_structure.stability_grid[x, y, z]),
                    'coherence': float(brain_structure.coherence_grid[x, y, z])
                }
            }
            integration_metrics['integration_phases']['positioning'] = phase1_metrics
            
            # Phase 2: Energy distribution to brain structure
            # Handle mycelial network connection
            try:
                # Try to import the mycelial network controller with correct path
                from system.mycelial_network.mycelial_network_controller import MycelialNetworkController
                
                # Initialize network controller
                network_controller = MycelialNetworkController(brain_structure)
                
                # Initialize network
                network_init = network_controller.initialize_system(seed_position)
                
                # Prepare soul properties for energy distribution
                soul_properties = {
                    'frequency': brain_seed.soul_connection.get('life_cord_primary_freq_hz', 432.0),
                    'stability': brain_seed.stability * 100,  # Convert 0-1 to 0-100
                    'coherence': brain_seed.structural_integrity * 100,  # Convert 0-1 to 0-100
                    'aspects': brain_seed.soul_aspect_distribution
                }
                
                # Distribute energy
                energy_distribution = network_controller.distribute_soul_energy(
                    seed_position, 
                    brain_seed.base_energy_level + brain_seed.mycelial_energy_store,
                    soul_properties
                )
                
                # If we get here, mycelial network is working
                mycelial_network_active = True
                
                # Phase 2 metrics with mycelial network
                phase2_metrics = {
                    'mycelial_network_active': True,
                    'network_initialization': network_init.get('success', False),
                    'energy_distributed': energy_distribution.get('total_energy_distributed', 0.0),
                    'distribution_efficiency': energy_distribution.get('distribution_efficiency', 0.0),
                    'cells_energized': energy_distribution.get('cells_energized', 0),
                    'region_coverage': energy_distribution.get('region_distribution', {})
                }
                
            except (ImportError, AttributeError, Exception) as e:
                # Mycelial network unavailable - implement fallback
                logger.warning(f"Mycelial network unavailable: {e}. Using fallback energy distribution.")
                
                # Direct energy distribution to brain structure
                energy_amount = brain_seed.base_energy_level + brain_seed.mycelial_energy_store
                
                # Create basic energy distribution in a sphere around seed
                energy_radius = 10  # cells
                total_distributed = 0
                cells_energized = 0
                region_distribution = {}
                
                # Distribute energy in sphere around seed
                for dx in range(-energy_radius, energy_radius + 1):
                    for dy in range(-energy_radius, energy_radius + 1):
                        for dz in range(-energy_radius, energy_radius + 1):
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < brain_structure.dimensions[0] and 
                                0 <= ny < brain_structure.dimensions[1] and 
                                0 <= nz < brain_structure.dimensions[2]):
                                
                                # Calculate distance for falloff
                                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                                if dist <= energy_radius:
                                    # Energy decreases with distance from seed
                                    energy_factor = 1.0 - (dist / energy_radius)
                                    cell_energy = energy_amount * energy_factor * 0.01  # Small portion per cell
                                    
                                    # Apply energy to brain grid
                                    brain_structure.energy_grid[nx, ny, nz] += cell_energy
                                    total_distributed += cell_energy
                                    cells_energized += 1
                                    
                                    # Track region distribution
                                    region = brain_structure.region_grid[nx, ny, nz]
                                    if region:
                                        if region not in region_distribution:
                                            region_distribution[region] = 0
                                        region_distribution[region] += cell_energy
                
                # Phase 2 metrics with fallback
                phase2_metrics = {
                    'mycelial_network_active': False,
                    'energy_distributed': float(total_distributed),
                    'distribution_efficiency': float(total_distributed / energy_amount),
                    'cells_energized': cells_energized,
                    'region_distribution': {k: float(v) for k, v in region_distribution.items()},
                    'fallback_method': 'spherical_distribution'
                }
                
                # Mark mycelial network as not active
                mycelial_network_active = False
            
            # Save energy distribution metrics
            integration_metrics['integration_phases']['energy_distribution'] = phase2_metrics
            
            # Phase 3: Set initial soul presence at seed position
            # Create high soul presence at seed position and immediate vicinity
            presence_radius = 3
            for dx in range(-presence_radius, presence_radius + 1):
                for dy in range(-presence_radius, presence_radius + 1):
                    for dz in range(-presence_radius, presence_radius + 1):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < brain_structure.dimensions[0] and 
                            0 <= ny < brain_structure.dimensions[1] and 
                            0 <= nz < brain_structure.dimensions[2]):
                            
                            # Calculate distance-based falloff
                            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                            if dist <= presence_radius:
                                # Higher presence closer to seed
                                presence = max(0.0, 1.0 - (dist / presence_radius))
                                brain_structure.soul_presence_grid[nx, ny, nz] = presence
                                
                                # Also set soul frequency
                                soul_frequency = brain_seed.soul_connection.get('life_cord_primary_freq_hz', 432.0)
                                brain_structure.soul_frequency_grid[nx, ny, nz] = soul_frequency
            
            # Mark seed position with maximum presence
            brain_structure.soul_presence_grid[x, y, z] = 1.0
            soul_frequency = brain_seed.soul_connection.get('life_cord_primary_freq_hz', 432.0)
            brain_structure.soul_frequency_grid[x, y, z] = soul_frequency
            
            # Update total soul filled cells count
            brain_structure.soul_filled_cells = np.sum(brain_structure.soul_presence_grid > 0.1)
            
            # Phase 3 metrics
            phase3_metrics = {
                'initial_presence_radius': presence_radius,
                'seed_presence': 1.0,
                'cells_with_presence': int(brain_structure.soul_filled_cells),
                'presence_coverage_percent': float(
                    brain_structure.soul_filled_cells / brain_structure.total_grid_cells * 100
                )
            }
            integration_metrics['integration_phases']['initial_presence'] = phase3_metrics
            
            # Phase 4: Create memory pathways for soul aspects if mycelial network is active
            if mycelial_network_active and hasattr(brain_seed, 'soul_aspect_distribution') and brain_seed.soul_aspect_distribution:
                try:
                    # Create memory pathways
                    memory_pathways = network_controller.mycelial_network.create_memory_pathways(
                        brain_seed.soul_aspect_distribution
                    )
                    
                    # Phase 4 metrics
                    phase4_metrics = {
                        'pathways_created': memory_pathways.get('pathways_created', 0),
                        'aspects_connected': memory_pathways.get('aspects_connected', 0),
                        'aspect_pathways': memory_pathways.get('aspect_pathways', {})
                    }
                except Exception as e:
                    # Handle failure in memory pathway creation
                    logger.warning(f"Memory pathway creation failed: {e}")
                    phase4_metrics = {
                        'pathways_created': 0,
                        'error': str(e),
                        'aspect_count': len(brain_seed.soul_aspect_distribution)
                    }
            else:
                # No mycelial network or no aspects to distribute
                phase4_metrics = {
                    'pathways_created': 0,
                    'mycelial_network_active': mycelial_network_active,
                    'aspect_count': len(brain_seed.soul_aspect_distribution) if hasattr(brain_seed, 'soul_aspect_distribution') else 0
                }
            
            integration_metrics['integration_phases']['memory_pathways'] = phase4_metrics
            
            # Update brain structure with integration timestamp
            brain_structure.last_updated = datetime.now().isoformat()
            
            # Mark integration as complete
            self.integration_completed = True
            self.last_updated = datetime.now().isoformat()
            self.integration_metrics = integration_metrics
            
            # Final metrics
            integration_metrics['integration_end'] = datetime.now().isoformat()
            integration_metrics['success'] = True
            
            logger.info(f"Brain seed integration successful at position {seed_position}")
            return integration_metrics
            
        except Exception as e:
            logger.error(f"Error during brain seed integration: {e}", exc_info=True)
            integration_metrics['integration_end'] = datetime.now().isoformat()
            integration_metrics['success'] = False
            integration_metrics['error'] = str(e)
            self.integration_metrics = integration_metrics
            raise RuntimeError(f"Brain seed integration failed: {e}") from e
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get the current integration status and metrics"""
        return {
            'completed': self.integration_completed,
            'creation_time': self.creation_time,
            'last_updated': self.last_updated,
            'metrics_summary': {
                'success': self.integration_metrics.get('success', False),
                'integration_time': (
                    datetime.fromisoformat(self.integration_metrics.get('integration_end', self.last_updated)) -
                    datetime.fromisoformat(self.integration_metrics.get('integration_start', self.creation_time))
                ).total_seconds() if 'integration_start' in self.integration_metrics else 0,
                'energy_distributed': self.integration_metrics.get('integration_phases', {}).get(
                    'energy_distribution', {}).get('energy_distributed', 0.0),
                'cells_with_presence': self.integration_metrics.get('integration_phases', {}).get(
                    'initial_presence', {}).get('cells_with_presence', 0)
            } if self.integration_metrics else {}
        }


def integrate_brain_seed(brain_seed, brain_structure, seed_position: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
    """
    Helper function to integrate a brain seed with brain structure.
    
    Args:
        brain_seed: The BrainSeed instance
        brain_structure: The BrainGrid instance
        seed_position: Optional position for seed (if None, optimal position is determined)
        
    Returns:
        Dict containing integration results
    """
    integrator = BrainSeedIntegration()
    return integrator.integrate_seed_with_structure(brain_seed, brain_structure, seed_position)