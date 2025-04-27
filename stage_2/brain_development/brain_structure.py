"""
brain_structure.py - Module for developing and managing brain structure.

This module handles the development of brain hemispheres, regions,
and the integration of sacred geometry and platonic solids.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any

# Import constants when available
try:
    from soul.constants import (
        SACRED_GEOMETRY,
        PLATONIC_SOLIDS,
        BRAIN_FREQUENCIES,
        GOLDEN_RATIO,
        FIBONACCI_SEQUENCE
    )
except ImportError:
    # Default constants if module not available
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    BRAIN_FREQUENCIES = {
        'delta': (0.5, 4),      # Deep sleep
        'theta': (4, 8),        # Drowsy, meditation
        'alpha': (8, 13),       # Relaxed, calm
        'beta': (13, 30),       # Alert, active
        'gamma': (30, 100),     # High cognition
        'lambda': (100, 400)    # Higher spiritual states
    }
    SACRED_GEOMETRY = {
        'vesica_piscis': {'complexity': 2, 'dimensions': 2},
        'seed_of_life': {'complexity': 7, 'dimensions': 2},
        'flower_of_life': {'complexity': 19, 'dimensions': 2},
        'tree_of_life': {'complexity': 10, 'dimensions': 2},
        'metatrons_cube': {'complexity': 13, 'dimensions': 3},
        'sri_yantra': {'complexity': 9, 'dimensions': 2},
        'torus': {'complexity': 1, 'dimensions': 3}
    }
    PLATONIC_SOLIDS = {
        'tetrahedron': {'vertices': 4, 'faces': 4, 'element': 'fire'},
        'hexahedron': {'vertices': 8, 'faces': 6, 'element': 'earth'},
        'octahedron': {'vertices': 6, 'faces': 8, 'element': 'air'},
        'dodecahedron': {'vertices': 20, 'faces': 12, 'element': 'aether'},
        'icosahedron': {'vertices': 12, 'faces': 20, 'element': 'water'}
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainStructure')

def develop_hemispheres(brain_seed):
    """
    Develop the left and right hemispheres with sacred geometry patterns.
    Each hemisphere will have its own geometry, frequency, and characteristics.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to develop hemispheres for
        
    Returns:
        dict: Development metrics
    """
    logger.info("Developing brain hemispheres with sacred geometry")
    
    # Check if brain seed is ready for hemisphere development
    if brain_seed.formation_progress < 0.1:
        logger.warning("Brain seed not sufficiently formed for hemisphere development")
        return {
            'success': False,
            'reason': 'insufficient_formation',
            'required_progress': 0.1,
            'current_progress': brain_seed.formation_progress
        }
    
    # Call brain seed's internal hemisphere development
    development_result = brain_seed.develop_initial_structure()
    
    if not development_result['success']:
        logger.warning(f"Hemisphere development failed: {development_result.get('reason', 'unknown')}")
        return development_result
    
    # Apply advanced sacred geometry patterns to each hemisphere
    apply_sacred_geometry_to_hemispheres(brain_seed)
    
    # Create inter-hemisphere connections
    create_corpus_callosum(brain_seed)
    
    logger.info("Hemispheres successfully developed with sacred geometry patterns")
    
    return {
        'success': True,
        'left_hemisphere': {
            'complexity': brain_seed.hemisphere_structure['left']['complexity'],
            'patterns': brain_seed.hemisphere_structure['left'].get('sacred_geometry', []),
            'frequency_range': [f['frequency'] for f in brain_seed.hemisphere_structure['left'].get('frequency_range', [])]
        },
        'right_hemisphere': {
            'complexity': brain_seed.hemisphere_structure['right']['complexity'],
            'patterns': brain_seed.hemisphere_structure['right'].get('sacred_geometry', []),
            'frequency_range': [f['frequency'] for f in brain_seed.hemisphere_structure['right'].get('frequency_range', [])]
        },
        'balance': development_result.get('balance_factor', 0.5),
        'progress': brain_seed.formation_progress
    }

def apply_sacred_geometry_to_hemispheres(brain_seed):
    """
    Apply sacred geometry patterns to hemisphere structures.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to apply sacred geometry to
    """
    # Define pattern application process for left hemisphere (logical/ordered)
    left_patterns = brain_seed.hemisphere_structure['left'].get('sacred_geometry', [])
    
    # Calculate pattern scaling factors based on hemisphere complexity
    left_scale = 0.5 + (0.5 * brain_seed.hemisphere_structure['left']['complexity'] / 10)
    
    # Create geometry structures
    left_geometry = []
    for pattern_name in left_patterns:
        if pattern_name in SACRED_GEOMETRY:
            # Create pattern instance
            pattern = {
                'name': pattern_name,
                'scale': left_scale * (0.8 + (0.4 * np.random.random())),
                'rotation': [np.random.random() * 2 * np.pi for _ in range(3)],
                'position': np.array([-0.3, 0, 0]) + np.array([
                    (np.random.random() - 0.5) * 0.2,
                    (np.random.random() - 0.5) * 0.2,
                    (np.random.random() - 0.5) * 0.2
                ]),
                'complexity': SACRED_GEOMETRY[pattern_name]['complexity'],
                'dimensions': SACRED_GEOMETRY[pattern_name]['dimensions'],
                'frequencies': [
                    10 + (SACRED_GEOMETRY[pattern_name]['complexity'] * 2 * (1 + np.random.random()))
                    for _ in range(3)
                ],
                'intensity': 0.7 + (0.3 * np.random.random())
            }
            left_geometry.append(pattern)
    
    # Apply to left hemisphere
    brain_seed.hemisphere_structure['left']['geometry_structures'] = left_geometry
    
    # Define pattern application process for right hemisphere (creative/fluid)
    right_patterns = brain_seed.hemisphere_structure['right'].get('sacred_geometry', [])
    
    # Calculate pattern scaling factors based on hemisphere complexity
    right_scale = 0.5 + (0.5 * brain_seed.hemisphere_structure['right']['complexity'] / 10)
    
    # Create geometry structures
    right_geometry = []
    for pattern_name in right_patterns:
        if pattern_name in SACRED_GEOMETRY:
            # Create pattern instance
            pattern = {
                'name': pattern_name,
                'scale': right_scale * (0.8 + (0.4 * np.random.random())),
                'rotation': [np.random.random() * 2 * np.pi for _ in range(3)],
                'position': np.array([0.3, 0, 0]) + np.array([
                    (np.random.random() - 0.5) * 0.2,
                    (np.random.random() - 0.5) * 0.2,
                    (np.random.random() - 0.5) * 0.2
                ]),
                'complexity': SACRED_GEOMETRY[pattern_name]['complexity'],
                'dimensions': SACRED_GEOMETRY[pattern_name]['dimensions'],
                'frequencies': [
                    7 + (SACRED_GEOMETRY[pattern_name]['complexity'] * 1.5 * (1 + np.random.random()))
                    for _ in range(3)
                ],
                'intensity': 0.7 + (0.3 * np.random.random())
            }
            right_geometry.append(pattern)
    
    # Apply to right hemisphere
    brain_seed.hemisphere_structure['right']['geometry_structures'] = right_geometry
    
    logger.info(f"Applied {len(left_geometry)} sacred geometry patterns to left hemisphere")
    logger.info(f"Applied {len(right_geometry)} sacred geometry patterns to right hemisphere")

def create_corpus_callosum(brain_seed):
    """
    Create corpus callosum connections between hemispheres.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to create connections for
    """
    # Create connection structure
    corpus_callosum = {
        'fibers': [],
        'connection_strength': 0.0,
        'bandwidth': 0.0,
        'stability': 0.0
    }
    
    # Create connection fibers
    fiber_count = 5 + int(brain_seed.complexity * 1.5)
    
    for i in range(fiber_count):
        # Position along z-axis (-0.5 to 0.5)
        z_pos = -0.5 + (i / (fiber_count - 1))
        
        # Create fiber
        fiber = {
            'id': f'cc_fiber_{i}',
            'position_start': np.array([-0.2, 0, z_pos]),
            'position_end': np.array([0.2, 0, z_pos]),
            'thickness': 0.02 + (0.03 * np.random.random()),
            'signal_speed': 0.7 + (0.3 * np.random.random()),
            'frequency': 8 + (12 * np.random.random()),  # Alpha-beta range for inter-hemisphere
            'connection_type': random.choice([
                'info_transfer', 'synchronization', 'modulation',
                'suppression', 'reinforcement'
            ])
        }
        
        corpus_callosum['fibers'].append(fiber)
    
    # Calculate overall metrics
    corpus_callosum['connection_strength'] = 0.6 + (0.4 * brain_seed.stability)
    corpus_callosum['bandwidth'] = fiber_count * 10
    corpus_callosum['stability'] = 0.7 + (0.3 * np.random.random())
    
    # Store corpus callosum
    brain_seed.corpus_callosum = corpus_callosum
    
    logger.info(f"Created corpus callosum with {fiber_count} fibers")

def develop_regions(brain_seed):
    """
    Develop brain regions with platonic structures.
    Each region will have specialized platonic solids and frequency patterns.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to develop regions for
        
    Returns:
        dict: Development metrics
    """
    logger.info("Developing brain regions with platonic structures")
    
    # Check if hemispheres are developed
    if not (brain_seed.hemisphere_structure['left'].get('developed', False) and 
            brain_seed.hemisphere_structure['right'].get('developed', False)):
        logger.warning("Cannot develop regions before hemispheres are formed")
        return {
            'success': False,
            'reason': 'hemispheres_not_developed',
            'progress': brain_seed.formation_progress
        }
    
    # Call brain seed's internal region development
    development_result = brain_seed.develop_brain_regions()
    
    if not development_result['success']:
        logger.warning(f"Region development failed: {development_result.get('reason', 'unknown')}")
        return development_result
    
    # Apply advanced platonic structures to regions
    apply_platonic_structures_to_regions(brain_seed)
    
    logger.info("Brain regions successfully developed with platonic structures")
    
    return {
        'success': True,
        'regions_developed': development_result.get('regions_developed', []),
        'region_count': development_result.get('region_count', 0),
        'balance_factor': development_result.get('balance_factor', 0.5),
        'progress': brain_seed.formation_progress
    }

def apply_platonic_structures_to_regions(brain_seed):
    """
    Apply platonic structures to brain regions.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to apply platonic structures to
    """
    # Iterate through each region
    for region_name, region in brain_seed.region_structure.items():
        # Get base platonic structure
        base_structure = region.get('platonic_structure', {})
        solid_name = base_structure.get('solid', random.choice(list(PLATONIC_SOLIDS.keys())))
        
        if solid_name not in PLATONIC_SOLIDS:
            continue
            
        # Get platonic solid properties
        solid_info = PLATONIC_SOLIDS[solid_name]
        
        # Create enhanced platonic structure
        enhanced_structure = {
            'solid': solid_name,
            'element': solid_info['element'],
            'vertices': solid_info['vertices'],
            'faces': solid_info['faces'],
            'scale': base_structure.get('scale', 0.5) * (0.8 + (0.4 * np.random.random())),
            'rotation': [np.random.random() * 2 * np.pi for _ in range(3)],
            'position': np.mean([p['position'] for p in region['pockets']], axis=0),
            'frequency': region.get('frequency', {}).get('frequency', 10.0),
            'nested_structures': []
        }
        
        # Add nested platonic structures for more complexity
        nest_count = 1 + int(region['complexity'] / 3)
        
        for i in range(nest_count):
            # Select a different solid for nesting
            nested_solid = random.choice([s for s in PLATONIC_SOLIDS.keys() if s != solid_name])
            nested_info = PLATONIC_SOLIDS[nested_solid]
            
            # Calculate nesting parameters
            scale_factor = 0.3 + (0.4 * (i / nest_count))
            
            # Create nested structure
            nested = {
                'solid': nested_solid,
                'element': nested_info['element'],
                'vertices': nested_info['vertices'],
                'faces': nested_info['faces'],
                'scale': scale_factor,
                'rotation': [np.random.random() * 2 * np.pi for _ in range(3)],
                'frequency': enhanced_structure['frequency'] * GOLDEN_RATIO * (0.9 + (0.2 * np.random.random())),
                'connection_points': []
            }
            
            # Create connection points between nested and main structure
            connection_count = min(3, random.randint(1, nested_info['vertices'] // 2))
            
            for j in range(connection_count):
                # Create connection point
                connection = {
                    'strength': 0.5 + (0.5 * np.random.random()),
                    'frequency': (enhanced_structure['frequency'] + nested['frequency']) / 2,
                    'position': np.array([
                        (np.random.random() - 0.5) * nested['scale'],
                        (np.random.random() - 0.5) * nested['scale'],
                        (np.random.random() - 0.5) * nested['scale']
                    ])
                }
                
                nested['connection_points'].append(connection)
            
            enhanced_structure['nested_structures'].append(nested)
        
        # Update the region's platonic structure
        brain_seed.region_structure[region_name]['platonic_structure'] = enhanced_structure
    
    logger.info(f"Applied enhanced platonic structures to {len(brain_seed.region_structure)} regions")

def create_regional_pockets(brain_seed):
    """
    Create pockets within each region with varying frequencies and colors.
    This provides the complexity needed for unique brain formation.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to create pockets for
        
    Returns:
        dict: Pocket creation metrics
    """
    logger.info("Creating regional pockets with frequency variations")
    
    # Check if regions exist
    if not brain_seed.region_structure:
        logger.warning("Cannot create pockets before regions are developed")
        return {
            'success': False,
            'reason': 'regions_not_developed',
            'progress': brain_seed.formation_progress
        }
    
    # Track metrics
    pocket_metrics = {}
    total_pockets = 0
    
    # Create pockets for each region
    for region_name, region in brain_seed.region_structure.items():
        # Get region complexity
        complexity = region.get('complexity', brain_seed.complexity)
        
        # Calculate number of pockets
        pocket_count = max(5, int(complexity * 2))
        
        # Create pockets if not already present
        if 'pockets' not in region or not region['pockets']:
            region['pockets'] = brain_seed._create_region_pockets(region_name, complexity)
        
        # Track metrics
        pocket_metrics[region_name] = {
            'count': len(region['pockets']),
            'avg_size': np.mean([p['size'] for p in region['pockets']]),
            'frequency_ranges': set([p['frequency']['band'] for p in region['pockets']])
        }
        
        total_pockets += len(region['pockets'])
    
    logger.info(f"Created total of {total_pockets} pockets across {len(brain_seed.region_structure)} regions")
    
    return {
        'success': True,
        'total_pockets': total_pockets,
        'region_metrics': pocket_metrics,
        'progress': brain_seed.formation_progress
    }

def apply_white_noise(brain_seed):
    """
    Apply white noise to unstructured areas of the brain.
    This creates background activity in empty pockets.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to apply white noise to
        
    Returns:
        dict: White noise application metrics
    """
    logger.info("Applying white noise to unstructured brain areas")
    
    # Use brain seed's built-in white noise function
    result = brain_seed.apply_white_noise()
    
    # Add additional variations to the white noise
    if result['success']:
        _enhance_white_noise(brain_seed)
    
    return result

def _enhance_white_noise(brain_seed):
    """
    Enhance white noise with additional spectral characteristics.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to enhance white noise for
    """
    # Iterate through regions
    for region_name, region in brain_seed.region_structure.items():
        if 'white_noise_frequencies' not in region:
            continue
            
        # Add pink noise component (1/f spectrum - more natural)
        pink_noise = []
        for i in range(10):
            # Frequency decreases, amplitude increases (1/f relationship)
            freq = 1.0 + (30.0 / (i + 1))
            amplitude = 0.02 + (0.03 / np.sqrt(i + 1))
            
            pink_noise.append({
                'frequency': freq,
                'amplitude': amplitude * region['white_noise_level'],
                'phase': 2 * np.pi * np.random.random()
            })
        
        # Add brown noise component (1/f² spectrum - deeper tones)
        brown_noise = []
        for i in range(5):
            # Frequency decreases, amplitude increases even faster (1/f² relationship)
            freq = 0.5 + (10.0 / (i + 1))
            amplitude = 0.01 + (0.04 / ((i + 1) ** 2))
            
            brown_noise.append({
                'frequency': freq,
                'amplitude': amplitude * region['white_noise_level'],
                'phase': 2 * np.pi * np.random.random()
            })
        
        # Add to existing white noise
        region['enhanced_noise'] = {
            'white': region['white_noise_frequencies'],
            'pink': pink_noise,
            'brown': brown_noise,
            'composite_level': region['white_noise_level']
        }
    
    logger.info("Enhanced white noise with pink and brown components")

def prepare_for_soul_attachment(brain_seed):
    """
    Prepare the brain for soul attachment by creating attachment points
    and resonance channels.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to prepare
        
    Returns:
        dict: Preparation metrics
    """
    logger.info("Preparing brain for soul attachment")
    
    # Use brain seed's built-in preparation function
    result = brain_seed.prepare_for_soul_attachment()
    
    if result['success']:
        # Enhance attachment points with additional properties
        _enhance_attachment_points(brain_seed)
    
    return result

def _enhance_attachment_points(brain_seed):
    """
    Enhance soul attachment points with additional properties.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to enhance attachment points for
    """
    if not hasattr(brain_seed, 'attachment_points'):
        return
        
    # Iterate through attachment points
    for point in brain_seed.attachment_points:
        # Add resonance harmonics
        point['resonance_harmonics'] = []
        
        # Create harmonic series based on point's base frequency
        base_freq = point['frequency']
        
        # Use Fibonacci ratios for harmonics (more spiritual)
        for i in range(1, 6):
            if i < len(FIBONACCI_SEQUENCE):
                ratio = FIBONACCI_SEQUENCE[i] / FIBONACCI_SEQUENCE[i-1]
                harmonic = {
                    'frequency': base_freq * ratio,
                    'amplitude': 0.9 / (i + 1),
                    'phase': 2 * np.pi * np.random.random()
                }
                point['resonance_harmonics'].append(harmonic)
        
        # Add energy transfer properties
        point['energy_transfer'] = {
            'input_capacity': point['capacity'] * 0.4,
            'output_capacity': point['capacity'] * 0.6,
            'efficiency': 0.7 + (0.3 * point['strength']),
            'directional_bias': np.random.random()  # 0=brain to soul, 1=soul to brain
        }
        
        # Add spiritual properties
        point['spiritual_properties'] = {
            'receptivity': 0.5 + (0.5 * np.random.random()),
            'clarity': 0.6 + (0.4 * point['strength']),
            'dimensional_resonance': 0.3 + (0.7 * np.random.random())
        }
    
    logger.info(f"Enhanced {len(brain_seed.attachment_points)} soul attachment points")

def verify_brain_structure(brain_seed):
    """
    Verify the complete brain structure for consistency and readiness.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to verify
        
    Returns:
        dict: Verification results
    """
    logger.info("Verifying complete brain structure")
    
    # Check hemispheres
    hemispheres_check = {
        'developed': (brain_seed.hemisphere_structure['left'].get('developed', False) and 
                     brain_seed.hemisphere_structure['right'].get('developed', False)),
        'sacred_geometry': hasattr(brain_seed.hemisphere_structure['left'], 'geometry_structures') and
                          hasattr(brain_seed.hemisphere_structure['right'], 'geometry_structures'),
        'corpus_callosum': hasattr(brain_seed, 'corpus_callosum')
    }
    
    # Check regions
    regions_check = {
        'developed': len(brain_seed.region_structure) >= 5,
        'platonic_structures': all('platonic_structure' in region for region in brain_seed.region_structure.values()),
        'pockets': all('pockets' in region and len(region['pockets']) > 0 for region in brain_seed.region_structure.values()),
        'white_noise': all('white_noise_level' in region for region in brain_seed.region_structure.values())
    }
    
    # Check attachment readiness
    attachment_check = {
        'attachment_points': hasattr(brain_seed, 'attachment_points') and len(brain_seed.attachment_points) > 0,
        'resonance_channels': hasattr(brain_seed, 'resonance_channels') and len(brain_seed.resonance_channels) > 0
    }
    
    # Calculate overall readiness
    hemispheres_ready = all(hemispheres_check.values())
    regions_ready = all(regions_check.values())
    attachment_ready = all(attachment_check.values())
    
    overall_ready = hemispheres_ready and regions_ready and attachment_ready
    
    logger.info(f"Brain structure verification: {'PASSED' if overall_ready else 'FAILED'}")
    
    return {
        'success': overall_ready,
        'hemispheres_ready': hemispheres_ready,
        'regions_ready': regions_ready,
        'attachment_ready': attachment_ready,
        'hemispheres_check': hemispheres_check,
        'regions_check': regions_check,
        'attachment_check': attachment_check,
        'formation_progress': brain_seed.formation_progress,
        'stability': brain_seed.stability
    }

def get_brain_structure_metrics(brain_seed):
    """
    Get comprehensive metrics about the brain structure.
    
    Parameters:
        brain_seed (BrainSeed): The brain seed to get metrics for
        
    Returns:
        dict: Complete brain structure metrics
    """
    # Get base metrics from brain seed
    base_metrics = brain_seed.get_metrics()
    
    # Add detailed structure metrics
    structure_metrics = {
        'hemispheres': {
            'count': 2,
            'left': {
                'complexity': brain_seed.hemisphere_structure['left'].get('complexity', 0),
                'geometry_patterns': brain_seed.hemisphere_structure['left'].get('sacred_geometry', []),
                'energy': brain_seed.hemisphere_structure['left'].get('energy', 0)
            },
            'right': {
                'complexity': brain_seed.hemisphere_structure['right'].get('complexity', 0),
                'geometry_patterns': brain_seed.hemisphere_structure['right'].get('sacred_geometry', []),
                'energy': brain_seed.hemisphere_structure['right'].get('energy', 0)
            }
        },
        'regions': {
            'count': len(brain_seed.region_structure),
            'types': list(brain_seed.region_structure.keys()),
            'total_pockets': sum(len(region.get('pockets', [])) for region in brain_seed.region_structure.values()),
            'platonic_solids': [region.get('platonic_structure', {}).get('solid', 'unknown') 
                               for region in brain_seed.region_structure.values()]
        }
    }
    
    # Add corpus callosum metrics if available
    if hasattr(brain_seed, 'corpus_callosum'):
        structure_metrics['corpus_callosum'] = {
            'fiber_count': len(brain_seed.corpus_callosum.get('fibers', [])),
            'connection_strength': brain_seed.corpus_callosum.get('connection_strength', 0),
            'bandwidth': brain_seed.corpus_callosum.get('bandwidth', 0)
        }
    
    # Add attachment metrics if available
    if hasattr(brain_seed, 'attachment_points'):
        structure_metrics['soul_attachment'] = {
            'attachment_points': len(brain_seed.attachment_points),
            'primary_points': sum(1 for p in brain_seed.attachment_points if p.get('purpose', '') == 'primary_connection'),
            'region_points': sum(1 for p in brain_seed.attachment_points if p.get('purpose', '') == 'region_connection'),
            'hemisphere_points': sum(1 for p in brain_seed.attachment_points if p.get('purpose', '') == 'hemisphere_connection')
        }
    
    # Add resonance metrics if available
    if hasattr(brain_seed, 'resonance_channels'):
        structure_metrics['resonance'] = {
            'channel_count': len(brain_seed.resonance_channels),
            'channel_types': [channel.get('name', 'unknown') for channel in brain_seed.resonance_channels],
            'frequency_range': [
                min(channel.get('frequency', 0) for channel in brain_seed.resonance_channels),
                max(channel.get('frequency', 0) for channel in brain_seed.resonance_channels)
            ]
        }
    
    # Combine all metrics
    combined_metrics = {**base_metrics, 'structure': structure_metrics}
    
    return combined_metrics