"""
brain_soul_attachment.py - Module for connecting the soul to the brain through the life cord.

This module handles the attachment of the soul to the brain via the life cord,
establishing resonance, and distributing soul aspects throughout the brain.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Import constants when available
try:
    from soul.constants import (
        GOLDEN_RATIO,
        FIBONACCI_SEQUENCE,
        SEPHIROTH_ASPECTS,
        LIFE_CORD_FREQUENCIES
    )
except ImportError:
    # Default constants if module not available
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    SEPHIROTH_ASPECTS = {
        'kether': {'quality': 'divine_will', 'frequency': 963.0},
        'chokmah': {'quality': 'wisdom', 'frequency': 852.0},
        'binah': {'quality': 'understanding', 'frequency': 741.0},
        'chesed': {'quality': 'mercy', 'frequency': 639.0},
        'geburah': {'quality': 'severity', 'frequency': 528.0},
        'tiphareth': {'quality': 'beauty', 'frequency': 417.0},
        'netzach': {'quality': 'victory', 'frequency': 396.0},
        'hod': {'quality': 'splendor', 'frequency': 285.0},
        'yesod': {'quality': 'foundation', 'frequency': 174.0},
        'malkuth': {'quality': 'kingdom', 'frequency': 128.0}
    }
    LIFE_CORD_FREQUENCIES = {
        'primary': 528.0,
        'secondary': [396.0, 417.0, 639.0, 741.0, 852.0, 963.0]
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainSoulAttachment')

def attach_soul_to_brain(life_cord, brain_seed):
    """
    Attach the soul to the brain through the life cord.
    This establishes the primary connection between spiritual and physical.
    
    Parameters:
        life_cord: The life cord connecting to the soul
        brain_seed: The brain seed prepared for attachment
        
    Returns:
        dict: Attachment metrics
    """
    logger.info("Attaching soul to brain through life cord")
    
    # Check if the brain is prepared for attachment
    if not hasattr(brain_seed, 'attachment_points') or not brain_seed.attachment_points:
        logger.warning("Brain not prepared for soul attachment")
        return {
            'success': False,
            'reason': 'brain_not_prepared',
            'required': 'attachment_points'
        }
    
    # Check if the life cord is properly formed
    if not hasattr(life_cord, 'connection_strength') or life_cord.connection_strength < 0.5:
        logger.warning("Life cord not sufficiently developed for attachment")
        return {
            'success': False,
            'reason': 'life_cord_insufficient',
            'connection_strength': getattr(life_cord, 'connection_strength', 0)
        }
    
    # Create connection structure
    connection = {
        'brain_attachment_points': brain_seed.attachment_points,
        'soul_connection_points': [],
        'resonance_channels': brain_seed.resonance_channels,
        'connection_strength': 0.0,
        'resonance_coherence': 0.0,
        'energy_flow': {
            'brain_to_soul': 0.0,
            'soul_to_brain': 0.0
        },
        'established_time': 0.0
    }
    
    # Create soul connection points corresponding to brain attachment points
    soul_connection_points = _create_soul_connection_points(life_cord, brain_seed.attachment_points)
    connection['soul_connection_points'] = soul_connection_points
    
    # Establish primary connections
    connection = _establish_primary_connections(connection, life_cord)
    
    # Create resonance field between brain and soul
    connection = _create_resonance_field(connection, life_cord, brain_seed)
    
    # Calculate overall connection strength
    primary_strength = connection.get('primary_connection_strength', 0.5)
    resonance_coherence = connection.get('resonance_coherence', 0.5)
    
    connection['connection_strength'] = 0.4 * primary_strength + 0.6 * resonance_coherence
    
    # Set establishment time
    connection['established_time'] = 1.0
    
    # Store connection in both life cord and brain seed
    life_cord.brain_connection = connection
    brain_seed.soul_connection = connection
    
    logger.info(f"Soul-brain attachment established with strength {connection['connection_strength']:.2f}")
    
    return {
        'success': True,
        'connection_strength': connection['connection_strength'],
        'resonance_coherence': connection['resonance_coherence'],
        'attachment_points': len(connection['brain_attachment_points']),
        'soul_connection_points': len(connection['soul_connection_points'])
    }

def _create_soul_connection_points(life_cord, brain_attachment_points):
    """
    Create soul connection points corresponding to brain attachment points.
    
    Parameters:
        life_cord: The life cord connecting to the soul
        brain_attachment_points: Attachment points in the brain
        
    Returns:
        list: Soul connection points
    """
    soul_connection_points = []
    
    # Use life cord's cord structure and energy channels
    cord_structure = getattr(life_cord, 'cord_structure', {})
    
    # Get nodes from life cord
    nodes = cord_structure.get('nodes', [])
    
    # Get channels from life cord
    channels = cord_structure.get('channels', [])
    
    # Create matching connection points for each brain attachment point
    for i, brain_point in enumerate(brain_attachment_points):
        # Select appropriate node/channel from life cord
        node_idx = i % len(nodes) if nodes else None
        channel_idx = i % len(channels) if channels else None
        
        # Base position and properties from life cord node
        position = None
        frequency = LIFE_CORD_FREQUENCIES['primary']
        strength = 0.5
        
        if node_idx is not None and nodes:
            node = nodes[node_idx]
            position = node.get('position', np.array([0, 0, 0]))
            frequency = node.get('frequency', frequency)
            strength = node.get('strength', strength)
        
        # Add properties from channel if available
        if channel_idx is not None and channels:
            channel = channels[channel_idx]
            frequency = channel.get('frequency', frequency)
        
        # Create connection point
        purpose = brain_point.get('purpose', 'unknown')
        point = {
            'id': f'soul_connection_{i}',
            'position': position,
            'frequency': frequency,
            'strength': strength,
            'purpose': purpose,
            'corresponding_brain_point_id': brain_point.get('id', f'unknown_{i}'),
            'resonance_factor': _calculate_resonance(frequency, brain_point.get('frequency', frequency))
        }
        
        soul_connection_points.append(point)
    
    logger.info(f"Created {len(soul_connection_points)} soul connection points")
    
    return soul_connection_points

def _calculate_resonance(freq1, freq2):
    """Calculate resonance between two frequencies."""
    # Ensure we're working with positive values
    freq1 = abs(freq1)
    freq2 = abs(freq2)
    
    # Handle identical frequencies
    if abs(freq1 - freq2) < 0.001:
        return 1.0
    
    # Calculate ratio with larger frequency in numerator
    ratio = max(freq1, freq2) / min(freq1, freq2)
    
    # Perfect resonance for simple harmonic ratios
    harmonic_ratios = [1.0, 2.0, 3.0, 4.0, 1.5, 3.0/2.0, 4.0/3.0, GOLDEN_RATIO]
    
    # Find distance to closest harmonic ratio
    min_distance = min(abs(ratio - hr) for hr in harmonic_ratios)
    
    # Transform distance to resonance (closer = higher resonance)
    resonance = 1.0 / (1.0 + 5.0 * min_distance)
    
    return resonance

def _establish_primary_connections(connection, life_cord):
    """
    Establish primary connections between brain and soul points.
    
    Parameters:
        connection: The connection structure
        life_cord: The life cord
        
    Returns:
        dict: Updated connection structure
    """
    # Create matching connections between brain and soul points
    connections = []
    
    brain_points = connection['brain_attachment_points']
    soul_points = connection['soul_connection_points']
    
    # Match points by purpose
    matched_points = []
    
    # First match primary connections
    primary_brain_points = [p for p in brain_points if p.get('purpose') == 'primary_connection']
    primary_soul_points = [p for p in soul_points if p.get('purpose') == 'primary_connection']
    
    for i in range(min(len(primary_brain_points), len(primary_soul_points))):
        brain_point = primary_brain_points[i]
        soul_point = primary_soul_points[i]
        
        conn = {
            'brain_point_id': brain_point.get('id', f'unknown_brain_{i}'),
            'soul_point_id': soul_point.get('id', f'unknown_soul_{i}'),
            'strength': (brain_point.get('strength', 0.5) + soul_point.get('strength', 0.5)) / 2,
            'frequency': (brain_point.get('frequency', 10.0) + soul_point.get('frequency', 10.0)) / 2,
            'resonance': soul_point.get('resonance_factor', 0.5),
            'purpose': 'primary',
            'bandwidth': 80.0 + (20.0 * np.random.random())
        }
        
        connections.append(conn)
        matched_points.append((brain_point.get('id', ''), soul_point.get('id', '')))
    
    # Next match region connections
    region_brain_points = [p for p in brain_points if p.get('purpose') == 'region_connection']
    region_soul_points = [p for p in soul_points if p.get('purpose') == 'region_connection']
    
    for i in range(min(len(region_brain_points), len(region_soul_points))):
        brain_point = region_brain_points[i]
        soul_point = region_soul_points[i]
        
        conn = {
            'brain_point_id': brain_point.get('id', f'unknown_brain_{i}'),
            'soul_point_id': soul_point.get('id', f'unknown_soul_{i}'),
            'strength': (brain_point.get('strength', 0.5) + soul_point.get('strength', 0.5)) / 2,
            'frequency': (brain_point.get('frequency', 10.0) + soul_point.get('frequency', 10.0)) / 2,
            'resonance': soul_point.get('resonance_factor', 0.5),
            'purpose': 'region',
            'region': brain_point.get('region', 'unknown'),
            'bandwidth': 40.0 + (20.0 * np.random.random())
        }
        
        connections.append(conn)
        matched_points.append((brain_point.get('id', ''), soul_point.get('id', '')))
    
    # Last match hemisphere connections
    hemi_brain_points = [p for p in brain_points if p.get('purpose') == 'hemisphere_connection']
    hemi_soul_points = [p for p in soul_points if p.get('purpose') == 'hemisphere_connection']
    
    for i in range(min(len(hemi_brain_points), len(hemi_soul_points))):
        brain_point = hemi_brain_points[i]
        soul_point = hemi_soul_points[i]
        
        conn = {
            'brain_point_id': brain_point.get('id', f'unknown_brain_{i}'),
            'soul_point_id': soul_point.get('id', f'unknown_soul_{i}'),
            'strength': (brain_point.get('strength', 0.5) + soul_point.get('strength', 0.5)) / 2,
            'frequency': (brain_point.get('frequency', 10.0) + soul_point.get('frequency', 10.0)) / 2,
            'resonance': soul_point.get('resonance_factor', 0.5),
            'purpose': 'hemisphere',
            'hemisphere': brain_point.get('hemisphere', 'unknown'),
            'bandwidth': 60.0 + (20.0 * np.random.random())
        }
        
        connections.append(conn)
        matched_points.append((brain_point.get('id', ''), soul_point.get('id', '')))
    
    # Match any remaining unmatched points
    unmatched_brain_points = [p for p in brain_points if not any(p.get('id', '') == pair[0] for pair in matched_points)]
    unmatched_soul_points = [p for p in soul_points if not any(p.get('id', '') == pair[1] for pair in matched_points)]
    
    for i in range(min(len(unmatched_brain_points), len(unmatched_soul_points))):
        brain_point = unmatched_brain_points[i]
        soul_point = unmatched_soul_points[i]
        
        conn = {
            'brain_point_id': brain_point.get('id', f'unknown_brain_{i}'),
            'soul_point_id': soul_point.get('id', f'unknown_soul_{i}'),
            'strength': (brain_point.get('strength', 0.5) + soul_point.get('strength', 0.5)) / 2,
            'frequency': (brain_point.get('frequency', 10.0) + soul_point.get('frequency', 10.0)) / 2,
            'resonance': _calculate_resonance(
                brain_point.get('frequency', 10.0),
                soul_point.get('frequency', 10.0)
            ),
            'purpose': 'auxiliary',
            'bandwidth': 30.0 + (20.0 * np.random.random())
        }
        
        connections.append(conn)
    
    # Store connections
    connection['connections'] = connections
    
    # Calculate primary connection strength
    primary_strengths = [c['strength'] * c['resonance'] for c in connections if c['purpose'] == 'primary']
    connection['primary_connection_strength'] = np.mean(primary_strengths) if primary_strengths else 0.5
    
    logger.info(f"Established {len(connections)} primary connections between brain and soul")
    
    return connection

def _create_resonance_field(connection, life_cord, brain_seed):
    """
    Create resonance field between brain and soul.
    
    Parameters:
        connection: The connection structure
        life_cord: The life cord
        brain_seed: The brain seed
        
    Returns:
        dict: Updated connection structure
    """
    # Create resonance field based on brain and soul frequency patterns
    resonance_field = {
        'coherence': 0.0,
        'intensity': 0.0,
        'frequency_bands': {},
        'harmonic_nodes': []
    }
    
    # Collect brain frequencies
    brain_frequencies = []
    
    # Add hemisphere frequencies
    for hemi in ['left', 'right']:
        if 'frequency_range' in brain_seed.hemisphere_structure[hemi]:
            for freq_data in brain_seed.hemisphere_structure[hemi]['frequency_range']:
                brain_frequencies.append({
                    'frequency': freq_data.get('frequency', 10.0),
                    'source': f'{hemi}_hemisphere',
                    'importance': 0.8
                })
    
    # Add region frequencies
    for region_name, region in brain_seed.region_structure.items():
        if 'frequency' in region:
            brain_frequencies.append({
                'frequency': region['frequency'].get('frequency', 10.0),
                'source': f'region_{region_name}',
                'importance': 0.7
            })
    
    # Collect soul frequencies from life cord
    soul_frequencies = []
    
    # Add life cord frequencies
    if hasattr(life_cord, 'cord_structure'):
        # Add node frequencies
        if 'nodes' in life_cord.cord_structure:
            for node in life_cord.cord_structure['nodes']:
                if 'frequency' in node:
                    soul_frequencies.append({
                        'frequency': node['frequency'],
                        'source': 'life_cord_node',
                        'importance': 0.9
                    })
        
        # Add channel frequencies
        if 'channels' in life_cord.cord_structure:
            for channel in life_cord.cord_structure['channels']:
                if 'frequency' in channel:
                    soul_frequencies.append({
                        'frequency': channel['frequency'],
                        'source': 'life_cord_channel',
                        'importance': 0.7
                    })
    
    # Create resonance bands based on brain-soul frequency interactions
    resonance_bands = {}
    
    # Common frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100),
        'lambda': (100, 400)
    }
    
    # Calculate resonance for each band
    for band_name, (band_min, band_max) in bands.items():
        # Filter frequencies in this band
        brain_band_freqs = [f for f in brain_frequencies 
                           if band_min <= f['frequency'] <= band_max]
        soul_band_freqs = [f for f in soul_frequencies 
                          if band_min <= f['frequency'] <= band_max]
        
        # Skip if no frequencies in this band
        if not brain_band_freqs or not soul_band_freqs:
            continue
        
        # Calculate average resonance between all frequency pairs
        resonance_values = []
        
        for brain_freq in brain_band_freqs:
            for soul_freq in soul_band_freqs:
                # Calculate resonance
                resonance = _calculate_resonance(brain_freq['frequency'], soul_freq['frequency'])
                
                # Weight by importance
                weighted_resonance = resonance * brain_freq['importance'] * soul_freq['importance']
                resonance_values.append(weighted_resonance)
        
        # Calculate average resonance for this band
        band_resonance = np.mean(resonance_values)
        
        # Store in bands
        resonance_bands[band_name] = {
            'resonance': band_resonance,
            'brain_frequencies': [f['frequency'] for f in brain_band_freqs],
            'soul_frequencies': [f['frequency'] for f in soul_band_freqs],
            'harmonic_nodes': []
        }
        
        # Create harmonic nodes at points of high resonance
        for brain_freq in brain_band_freqs:
            for soul_freq in soul_band_freqs:
                resonance = _calculate_resonance(brain_freq['frequency'], soul_freq['frequency'])
                
                # Create harmonic node for high resonance pairs
                if resonance > 0.8:
                    node = {
                        'brain_frequency': brain_freq['frequency'],
                        'soul_frequency': soul_freq['frequency'],
                        'resonance': resonance,
                        'band': band_name,
                        'stability': 0.5 + (0.5 * resonance),
                        'energy': 10.0 * resonance
                    }
                    
                    resonance_bands[band_name]['harmonic_nodes'].append(node)
                    resonance_field['harmonic_nodes'].append(node)
    
    # Store resonance bands
    resonance_field['frequency_bands'] = resonance_bands
    
    # Calculate overall field coherence
    band_coherence_values = [band['resonance'] for band in resonance_bands.values()]
    resonance_field['coherence'] = np.mean(band_coherence_values) if band_coherence_values else 0.5
    
    # Calculate field intensity
    resonance_field['intensity'] = 0.4 + (0.6 * resonance_field['coherence'])
    
    # Store resonance field
    connection['resonance_field'] = resonance_field
    connection['resonance_coherence'] = resonance_field['coherence']
    
    logger.info(f"Created resonance field with coherence {resonance_field['coherence']:.2f}")
    logger.info(f"Established {len(resonance_field['harmonic_nodes'])} harmonic nodes")
    
    return connection

def distribute_soul_aspects(life_cord, brain_seed):
    """
    Distribute soul aspects throughout brain regions via resonance.
    This creates a harmonized blend of soul and brain.
    
    Parameters:
        life_cord: The life cord connecting to the soul
        brain_seed: The brain seed with soul connection
        
    Returns:
        dict: Distribution metrics
    """
    logger.info("Distributing soul aspects throughout brain regions")
    
    # Check if soul-brain connection is established
    if not hasattr(brain_seed, 'soul_connection'):
        logger.warning("Soul-brain connection not established")
        return {
            'success': False,
            'reason': 'connection_not_established'
        }
    
    # Get resonant soul from life cord
    resonant_soul = getattr(life_cord, 'resonant_soul', None)
    
    if resonant_soul is None:
        logger.warning("Resonant soul not found in life cord")
        return {
            'success': False,
            'reason': 'resonant_soul_missing'
        }
    
    # Create aspect distribution map
    distribution = {
        'aspects': {},
        'region_mappings': {},
        'hemisphere_mappings': {},
        'integration_level': 0.0
    }
    
    # Map soul aspects to brain regions based on resonance
    distribution = _map_aspects_to_regions(distribution, resonant_soul, brain_seed)
    
    # Create distributed soul field throughout brain
    distribution = _create_distributed_soul_field(distribution, resonant_soul, brain_seed)
    
    # Calculate integration level
    mapped_aspects = sum(len(aspects) for aspects in distribution['region_mappings'].values())
    total_aspects = len(distribution['aspects'])
    
    integration_percentage = mapped_aspects / total_aspects if total_aspects > 0 else 0
    distribution['integration_level'] = 0.5 + (0.5 * integration_percentage)
    
    # Store distribution in brain seed
    brain_seed.soul_aspect_distribution = distribution
    
    logger.info(f"Distributed {len(distribution['aspects'])} soul aspects with {mapped_aspects} region mappings")
    logger.info(f"Soul integration level: {distribution['integration_level']:.2f}")
    
    return {
        'success': True,
        'soul_aspects': len(distribution['aspects']),
        'mapped_aspects': mapped_aspects,
        'region_mappings': len(distribution['region_mappings']),
        'hemisphere_mappings': len(distribution['hemisphere_mappings']),
        'integration_level': distribution['integration_level']
    }

def _map_aspects_to_regions(distribution, resonant_soul, brain_seed):
    """
    Map soul aspects to brain regions based on resonance.
    
    Parameters:
        distribution: The distribution structure
        resonant_soul: The resonant soul
        brain_seed: The brain seed
        
    Returns:
        dict: Updated distribution structure
    """
    # Get soul aspects (from resonant soul or use default sephiroth aspects)
    soul_aspects = {}
    
    # Use soul aspects if available
    if hasattr(resonant_soul, 'aspects'):
        soul_aspects = resonant_soul.aspects
    else:
        # Use default sephiroth aspects
        soul_aspects = SEPHIROTH_ASPECTS
    
    # Store aspects
    distribution['aspects'] = soul_aspects
    
    # Define region-aspect affinity mappings
    region_affinities = {
        'frontal': ['kether', 'chokmah', 'binah'],  # Higher cognitive functions
        'parietal': ['tiphareth', 'chesed', 'geburah'],  # Integration, balance
        'temporal': ['netzach', 'hod'],  # Memory, language, creativity
        'occipital': ['yesod'],  # Visual processing, foundation
        'limbic': ['tiphareth', 'yesod'],  # Emotion, memory
        'cerebellum': ['malkuth'],  # Physical coordination
        'brainstem': ['malkuth', 'yesod']  # Basic functions, foundation
    }
    
    # For each region, map aspects based on natural affinity and resonance
    for region_name, region in brain_seed.region_structure.items():
        # Get preferred aspects for this region
        preferred_aspects = region_affinities.get(region_name, [])
        
        # Create region mapping
        distribution['region_mappings'][region_name] = []
        
        # First map preferred aspects
        for aspect_name in preferred_aspects:
            if aspect_name in soul_aspects:
                aspect = soul_aspects[aspect_name]
                
                # Calculate resonance between aspect and region frequencies
                aspect_freq = aspect.get('frequency', 0)
                region_freq = region['frequency'].get('frequency', 0) if 'frequency' in region else 0
                
                resonance = _calculate_resonance(aspect_freq, region_freq)
                
                # Create mapping
                mapping = {
                    'aspect': aspect_name,
                    'region': region_name,
                    'resonance': resonance,
                    'influence': 0.5 + (0.5 * resonance),
                    'integration': 0.3 + (0.7 * resonance)
                }
                
                distribution['region_mappings'][region_name].append(mapping)
        
        # Find other resonant aspects for this region
        region_freq = region['frequency'].get('frequency', 0) if 'frequency' in region else 0
        
        for aspect_name, aspect in soul_aspects.items():
            # Skip already mapped preferred aspects
            if aspect_name in preferred_aspects:
                continue
                
            # Calculate resonance
            aspect_freq = aspect.get('frequency', 0)
            resonance = _calculate_resonance(aspect_freq, region_freq)
            
            # Map highly resonant aspects
            if resonance > 0.7:
                mapping = {
                    'aspect': aspect_name,
                    'region': region_name,
                    'resonance': resonance,
                    'influence': 0.3 + (0.5 * resonance),
                    'integration': 0.2 + (0.5 * resonance)
                }
                
                distribution['region_mappings'][region_name].append(mapping)
    
    # Map aspects to hemispheres
    hemisphere_affinities = {
        'left': ['chokmah', 'binah', 'geburah', 'hod'],  # Logical, analytical
        'right': ['chesed', 'netzach', 'tiphareth', 'yesod']  # Creative, intuitive
    }
    
    # For each hemisphere, map aspects
    for hemi, hemisphere in brain_seed.hemisphere_structure.items():
        # Get preferred aspects for this hemisphere
        preferred_aspects = hemisphere_affinities.get(hemi, [])
        
        # Create hemisphere mapping
        distribution['hemisphere_mappings'][hemi] = []
        
        # Map preferred aspects
        for aspect_name in preferred_aspects:
            if aspect_name in soul_aspects:
                aspect = soul_aspects[aspect_name]
                
                # Create mapping
                mapping = {
                    'aspect': aspect_name,
                    'hemisphere': hemi,
                    'influence': 0.7,
                    'integration': 0.6
                }
                
                distribution['hemisphere_mappings'][hemi].append(mapping)
    
    return distribution

def _create_distributed_soul_field(distribution, resonant_soul, brain_seed):
    """
    Create distributed soul field throughout brain.
    
    Parameters:
        distribution: The distribution structure
        resonant_soul: The resonant soul
        brain_seed: The brain seed
        
    Returns:
        dict: Updated distribution structure
    """
    # Create soul field
    soul_field = {
        'overall_intensity': 0.0,
        'overall_coherence': 0.0,
        'regions': {},
        'hemispheres': {}
    }
    
    # Create field in each mapped region
    for region_name, mappings in distribution['region_mappings'].items():
        # Skip empty mappings
        if not mappings:
            continue
        
        # Get region
        region = brain_seed.region_structure.get(region_name, {})
        
        # Create field
        field = {
            'intensity': 0.0,
            'coherence': 0.0,
            'aspect_intensities': {},
            'pockets': []
        }
        
        # Calculate aspect intensities
        total_influence = sum(m['influence'] for m in mappings)
        
        for mapping in mappings:
            aspect_name = mapping['aspect']
            influence = mapping['influence']
            
            # Normalize influence
            normalized_influence = influence / total_influence if total_influence > 0 else 0
            
            # Store aspect intensity
            field['aspect_intensities'][aspect_name] = normalized_influence
            
            # Create soul pockets in region pockets
            if 'pockets' in region:
                for i, pocket in enumerate(region['pockets']):
                    # Only create soul pocket in some brain pockets
                    if np.random.random() < normalized_influence:
                        soul_pocket = {
                            'aspect': aspect_name,
                            'intensity': normalized_influence * (0.7 + 0.3 * np.random.random()),
                            'position': pocket['position'],
                            'frequency': SEPHIROTH_ASPECTS[aspect_name]['frequency'],
                            'color': _get_aspect_color(aspect_name),
                            'brain_pocket_id': pocket.get('id', f'unknown_{i}')
                        }
                        field['pockets'].append(soul_pocket)
        
        # Calculate field properties
        field['intensity'] = 0.3 + (0.7 * min(1.0, total_influence))
        field['coherence'] = 0.5 + (0.3 * np.random.random())
        
        # Store field
        soul_field['regions'][region_name] = field
    
    # Create field in each mapped hemisphere
    for hemi, mappings in distribution['hemisphere_mappings'].items():
        # Skip empty mappings
        if not mappings:
            continue
        
        # Create field
        field = {
            'intensity': 0.0,
            'coherence': 0.0,
            'aspect_intensities': {}
        }
        
        # Calculate aspect intensities
        total_influence = sum(m['influence'] for m in mappings)
        
        for mapping in mappings:
            aspect_name = mapping['aspect']
            influence = mapping['influence']
            
            # Normalize influence
            normalized_influence = influence / total_influence if total_influence > 0 else 0
            
            # Store aspect intensity
            field['aspect_intensities'][aspect_name] = normalized_influence
        
        # Calculate field properties
        field['intensity'] = 0.4 + (0.6 * min(1.0, total_influence))
        field['coherence'] = 0.6 + (0.3 * np.random.random())
        
        # Store field
        soul_field['hemispheres'][hemi] = field
    
    # Calculate overall field properties
    region_intensities = [r['intensity'] for r in soul_field['regions'].values()]
    hemisphere_intensities = [h['intensity'] for h in soul_field['hemispheres'].values()]
    
    # Overall intensity is weighted average of region and hemisphere intensities
    if region_intensities and hemisphere_intensities:
        region_avg = np.mean(region_intensities)
        hemi_avg = np.mean(hemisphere_intensities)
        
        soul_field['overall_intensity'] = 0.4 * region_avg + 0.6 * hemi_avg
    elif region_intensities:
        soul_field['overall_intensity'] = np.mean(region_intensities)
    elif hemisphere_intensities:
        soul_field['overall_intensity'] = np.mean(hemisphere_intensities)
    else:
        soul_field['overall_intensity'] = 0.0
    
    # Calculate coherence
    region_coherences = [r['coherence'] for r in soul_field['regions'].values()]
    hemisphere_coherences = [h['coherence'] for h in soul_field['hemispheres'].values()]
    
    if region_coherences and hemisphere_coherences:
        region_avg = np.mean(region_coherences)
        hemi_avg = np.mean(hemisphere_coherences)
        
        soul_field['overall_coherence'] = 0.3 * region_avg + 0.7 * hemi_avg
    elif region_coherences:
        soul_field['overall_coherence'] = np.mean(region_coherences)
    elif hemisphere_coherences:
        soul_field['overall_coherence'] = np.mean(hemisphere_coherences)
    else:
        soul_field['overall_coherence'] = 0.0
    
    # Store soul field
    distribution['soul_field'] = soul_field
    
    logger.info(f"Created distributed soul field with intensity {soul_field['overall_intensity']:.2f} "
               f"and coherence {soul_field['overall_coherence']:.2f}")
    
    return distribution

def _get_aspect_color(aspect_name):
    """Generate a color for a soul aspect."""
    # Define colors for each sephiroth aspect
    aspect_colors = {
        'kether': {'r': 255, 'g': 255, 'b': 255},  # White
        'chokmah': {'r': 200, 'g': 200, 'b': 255},  # Light blue
        'binah': {'r': 127, 'g': 0, 'b': 255},  # Purple
        'chesed': {'r': 0, 'g': 0, 'b': 255},  # Blue
        'geburah': {'r': 255, 'g': 0, 'b': 0},  # Red
        'tiphareth': {'r': 255, 'g': 215, 'b': 0},  # Gold
        'netzach': {'r': 0, 'g': 255, 'b': 0},  # Green
        'hod': {'r': 255, 'g': 165, 'b': 0},  # Orange
        'yesod': {'r': 130, 'g': 0, 'b': 255},  # Violet
        'malkuth': {'r': 139, 'g': 69, 'b': 19}   # Brown
    }
    
    # Return color for aspect or a default color
    return aspect_colors.get(aspect_name, {'r': 128, 'g': 128, 'b': 128})