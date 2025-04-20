"""
Creator Entanglement Module

This module implements the mechanisms for establishing quantum connection between
a soul spark and the creator (Kether). It enables resonance matching, aspect transfer,
and soul stabilization through creator alignment.

The creator entanglement process is a critical step that occurs after initial soul
formation in the Guff field and prepares the soul for its journey through the Sephiroth.

Author: Soul Development Framework Team
"""

import numpy as np
import logging
import uuid
from datetime import datetime
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from stage_1.sephiroth.kether_aspects import KetherAspects, AspectType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='creator_entanglement.log'
)
logger = logging.getLogger('creator_entanglement')

class CreatorEntanglement:
    """
    Implements mechanisms for soul-creator quantum entanglement.
    
    This class handles the quantum connection between a soul spark and the creator (Kether),
    enabling resonance matching, aspect transfer, and strengthening of the soul structure
    through creator alignment.
    """
    
    def __init__(self, creator_resonance=0.8, edge_of_chaos_ratio=0.618):
        """
        Initialize a new Creator Entanglement processor.
        
        Args:
            creator_resonance (float): Base strength of creator resonance (0-1)
            edge_of_chaos_ratio (float): Ratio for edge of chaos (default: golden ratio inverse)
        """
        self.entanglement_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.creator_resonance = creator_resonance
        self.edge_of_chaos_ratio = edge_of_chaos_ratio
        
        # Load Kether aspects for creator properties
        self.kether_aspects = KetherAspects()
        
        # Entanglement properties
        self.quantum_channels = {}
        self.resonance_patterns = {}
        self.aspect_transfer_metrics = {}
        
        logger.info(f"Creator Entanglement processor initialized with ID: {self.entanglement_id}")
        logger.info(f"Creator resonance: {creator_resonance}, Edge of chaos: {edge_of_chaos_ratio}")
    
    def establish_quantum_connection(self, soul_spark, connection_strength=None):
        """
        Establish the initial quantum connection between a soul spark and the creator.
        
        This forms the foundation for all subsequent entanglement processes.
        
        Args:
            soul_spark: The SoulSpark object to entangle
            connection_strength (float): Optional override for connection strength
            
        Returns:
            dict: Information about the established connection
        """
        # Use provided strength or default to creator_resonance
        if connection_strength is None:
            connection_strength = self.creator_resonance
        
        # Generate a unique channel ID
        channel_id = str(uuid.uuid4())
        
        # Calculate base connection properties
        stability = soul_spark.stability if hasattr(soul_spark, 'stability') else 0.7
        resonance = soul_spark.resonance if hasattr(soul_spark, 'resonance') else 0.7
        
        # Connection quality depends on spark properties and creator resonance
        connection_quality = (0.4 * stability + 0.4 * resonance + 0.2 * connection_strength)
        
        # Apply edge of chaos for optimal entanglement
        # Entanglement occurs most effectively at the edge of chaos
        chaos_factor = 1.0 - abs(connection_quality - self.edge_of_chaos_ratio) / self.edge_of_chaos_ratio
        
        # Final connection strength with chaos factor
        effective_strength = connection_strength * chaos_factor
        
        # Create the quantum channel
        quantum_channel = {
            'channel_id': channel_id,
            'spark_id': soul_spark.spark_id if hasattr(soul_spark, 'spark_id') else 'unknown',
            'creation_time': datetime.now().isoformat(),
            'connection_strength': effective_strength,
            'connection_quality': connection_quality,
            'stability': stability,
            'resonance': resonance,
            'chaos_factor': chaos_factor,
            'active': True,
            'frequency_signature': self._calculate_frequency_signature(soul_spark)
        }
        
        # Store the channel
        self.quantum_channels[channel_id] = quantum_channel
        
        # Update soul properties if supported
        if hasattr(soul_spark, 'creator_channel_id'):
            soul_spark.creator_channel_id = channel_id
        
        if hasattr(soul_spark, 'creator_connection_strength'):
            soul_spark.creator_connection_strength = effective_strength
            
        if hasattr(soul_spark, 'creator_alignment'):
            soul_spark.creator_alignment *= (1.0 + 0.1 * effective_strength)
            soul_spark.creator_alignment = min(1.0, soul_spark.creator_alignment)
        
        logger.info(f"Established quantum connection for spark {quantum_channel['spark_id']}")
        logger.info(f"Channel ID: {channel_id}, Strength: {effective_strength:.4f}")
        
        return quantum_channel
    
    def _calculate_frequency_signature(self, soul_spark):
        """
        Calculate the frequency signature for soul-creator resonance.
        
        This signature defines how the soul and creator frequencies interact.
        
        Args:
            soul_spark: The SoulSpark object
            
        Returns:
            dict: Frequency signature information
        """
        # Get creator frequencies from Kether aspects
        creator_freqs = self.kether_aspects.get_frequencies()
        
        # Get soul frequencies if available
        soul_freqs = {}
        if hasattr(soul_spark, 'frequency_signature') and soul_spark.frequency_signature:
            if 'frequencies' in soul_spark.frequency_signature:
                soul_freqs = {f'soul_{i}': freq 
                             for i, freq in enumerate(soul_spark.frequency_signature['frequencies'])}
        
        # Combine frequencies with weights
        combined_freqs = {}
        
        # Add creator frequencies with higher weight
        for name, freq in creator_freqs.items():
            combined_freqs[f'creator_{name}'] = {
                'frequency': freq,
                'weight': 0.7 * self.creator_resonance
            }
        
        # Add soul frequencies with lower weight
        for name, freq in soul_freqs.items():
            combined_freqs[name] = {
                'frequency': freq,
                'weight': 0.3
            }
            
        # Calculate resonance points (frequencies where creator and soul align)
        resonance_points = self._find_resonance_points(creator_freqs.values(), soul_freqs.values())
        
        # Construct frequency signature
        signature = {
            'creator_frequencies': creator_freqs,
            'soul_frequencies': soul_freqs,
            'combined_frequencies': combined_freqs,
            'resonance_points': resonance_points,
            'primary_resonance': max(resonance_points) if resonance_points else 0
        }
        
        return signature
    
    def _find_resonance_points(self, creator_freqs, soul_freqs):
        """
        Find frequency points where creator and soul resonance aligns.
        
        Args:
            creator_freqs (list): Creator frequency values
            soul_freqs (list): Soul frequency values
            
        Returns:
            list: Resonance point frequencies
        """
        # Convert inputs to lists if they're not already
        if not isinstance(creator_freqs, list):
            creator_freqs = list(creator_freqs)
        
        if not isinstance(soul_freqs, list):
            soul_freqs = list(soul_freqs)
            
        # If we don't have both sets, return empty list
        if not creator_freqs or not soul_freqs:
            return []
            
        resonance_points = []
        
        # Find harmonic relationships between frequencies
        for c_freq in creator_freqs:
            for s_freq in soul_freqs:
                # Check for direct resonance (same frequency)
                if abs(c_freq - s_freq) < 0.01 * c_freq:
                    resonance_points.append(c_freq)
                    continue
                    
                # Check for harmonic resonance (integer multiples)
                ratio = c_freq / s_freq if s_freq > 0 else 0
                inverse_ratio = s_freq / c_freq if c_freq > 0 else 0
                
                # Check if close to integer ratios up to 5:1
                for i in range(1, 6):
                    if abs(ratio - i) < 0.05 or abs(inverse_ratio - i) < 0.05:
                        # Use the lower frequency as the resonance point
                        resonance_points.append(min(c_freq, s_freq))
                        break
                        
                # Check for golden ratio resonance
                phi = (1 + np.sqrt(5)) / 2  # ~1.618
                if abs(ratio - phi) < 0.05 or abs(inverse_ratio - phi) < 0.05:
                    # Use the phi-modified frequency
                    resonance_points.append(min(c_freq, s_freq) * phi)
                    
        # Return unique resonance points, sorted
        return sorted(list(set(resonance_points)))
    
    def form_resonance_patterns(self, soul_spark, channel_id=None):
        """
        Form resonance patterns between soul and creator.
        
        These patterns strengthen the connection and prepare for aspect transfer.
        
        Args:
            soul_spark: The SoulSpark object
            channel_id (str): Optional specific channel ID to use
            
        Returns:
            dict: Information about the formed resonance patterns
        """
        # Get channel ID if not provided
        if channel_id is None:
            if hasattr(soul_spark, 'creator_channel_id'):
                channel_id = soul_spark.creator_channel_id
            else:
                # No channel exists yet, create one
                channel = self.establish_quantum_connection(soul_spark)
                channel_id = channel['channel_id']
        
        # Verify channel exists
        if channel_id not in self.quantum_channels:
            logger.warning(f"Channel {channel_id} not found. Creating new channel.")
            channel = self.establish_quantum_connection(soul_spark)
            channel_id = channel['channel_id']
            
        # Get the channel
        channel = self.quantum_channels[channel_id]
        
        # Calculate resonance pattern based on frequency signature
        signature = channel['frequency_signature']
        
        # Design resonance patterns based on creator aspects
        primary_aspects = self.kether_aspects.get_primary_aspects()
        creator_aspects = self.kether_aspects.get_creator_aspects()
        
        # Start with primary patterns from most important aspects
        patterns = {}
        
        # Add primary aspects with highest weights
        for name, aspect in primary_aspects.items():
            patterns[name] = {
                'frequency': aspect['frequency'],
                'strength': aspect['strength'] * channel['connection_strength'],
                'pattern_type': 'primary',
                'description': aspect['description'],
                'color': aspect['color'],
                'element': aspect['element'],
                'keywords': aspect['keywords']
            }
            
        # Add creator aspects with medium weights
        for name, aspect in creator_aspects.items():
            patterns[name] = {
                'frequency': aspect['frequency'],
                'strength': aspect['strength'] * channel['connection_strength'] * 0.8,
                'pattern_type': 'creator',
                'description': aspect['description'],
                'color': aspect['color'],
                'element': aspect['element'],
                'keywords': aspect['keywords']
            }
        
        # Generate harmonic patterns from resonance points
        if 'resonance_points' in signature and signature['resonance_points']:
            for i, point in enumerate(signature['resonance_points']):
                patterns[f'harmonic_{i}'] = {
                    'frequency': point,
                    'strength': 0.7 * channel['connection_strength'],
                    'pattern_type': 'harmonic',
                    'description': f'Harmonic resonance at {point:.2f} Hz',
                    'color': (255, 255, 255),  # Default white
                    'element': 'harmony',
                    'keywords': ['harmonic', 'resonance', 'alignment']
                }
                
        # Calculate overall pattern coherence
        pattern_strengths = [p['strength'] for p in patterns.values()]
        total_strength = sum(pattern_strengths)
        pattern_coherence = total_strength / len(patterns) if patterns else 0
        
        # Store resonance pattern
        resonance_data = {
            'channel_id': channel_id,
            'spark_id': channel['spark_id'],
            'patterns': patterns,
            'pattern_count': len(patterns),
            'pattern_coherence': pattern_coherence,
            'formation_time': datetime.now().isoformat()
        }
        
        self.resonance_patterns[channel_id] = resonance_data
        
        # Update soul properties if supported
        if hasattr(soul_spark, 'resonance_patterns'):
            soul_spark.resonance_patterns = patterns
            
        if hasattr(soul_spark, 'pattern_coherence'):
            soul_spark.pattern_coherence = pattern_coherence
            
        if hasattr(soul_spark, 'resonance'):
            soul_spark.resonance *= (1.0 + 0.05 * pattern_coherence)
            soul_spark.resonance = min(1.0, soul_spark.resonance)
        
        logger.info(f"Formed {len(patterns)} resonance patterns for spark {channel['spark_id']}")
        logger.info(f"Pattern coherence: {pattern_coherence:.4f}")
        
        return resonance_data
    
    def transfer_creator_aspects(self, soul_spark, channel_id=None, aspect_types=None):
        """
        Transfer creator aspects to the soul through the quantum channel.
        
        This enriches the soul with specific creator properties and strengthens
        its connection to the divine source.
        
        Args:
            soul_spark: The SoulSpark object
            channel_id (str): Optional specific channel ID to use
            aspect_types (list): Optional specific aspect types to transfer
            
        Returns:
            dict: Information about the transferred aspects
        """
        # Get channel ID if not provided
        if channel_id is None:
            if hasattr(soul_spark, 'creator_channel_id'):
                channel_id = soul_spark.creator_channel_id
            else:
                # No channel exists yet, create channel and resonance
                channel = self.establish_quantum_connection(soul_spark)
                channel_id = channel['channel_id']
                self.form_resonance_patterns(soul_spark, channel_id)
        
        # Verify channel exists and has resonance patterns
        if channel_id not in self.quantum_channels:
            logger.warning(f"Channel {channel_id} not found. Creating new channel.")
            channel = self.establish_quantum_connection(soul_spark)
            channel_id = channel['channel_id']
            self.form_resonance_patterns(soul_spark, channel_id)
            
        if channel_id not in self.resonance_patterns:
            logger.warning(f"Resonance patterns for channel {channel_id} not found. Creating patterns.")
            self.form_resonance_patterns(soul_spark, channel_id)
            
        # Get the channel and resonance data
        channel = self.quantum_channels[channel_id]
        resonance_data = self.resonance_patterns[channel_id]
        
        # Determine which aspect types to transfer
        if aspect_types is None:
            # Default: transfer all aspects except dimensional (those come later from Sephiroth)
            aspect_types = [
                AspectType.PRIMARY,
                AspectType.CREATOR,
                AspectType.ELEMENTAL,
                AspectType.CHAKRA,
                AspectType.HARMONIC,
                AspectType.YIN_YANG,
                AspectType.STRUCTURAL,
                AspectType.CONSCIOUSNESS
            ]
            
        # Filter aspects by type
        transferable_aspects = {}
        
        for aspect_type in aspect_types:
            aspects = self.kether_aspects.get_aspects_by_type(aspect_type)
            for name, aspect in aspects.items():
                # Calculate transfer efficiency based on resonance
                if name in resonance_data['patterns']:
                    pattern = resonance_data['patterns'][name]
                    efficiency = pattern['strength']
                else:
                    # Lower efficiency for aspects without specific resonance
                    efficiency = 0.6 * channel['connection_strength']
                    
                # Apply connection quality factor
                efficiency *= channel['connection_quality']
                
                # Calculate transferred strength
                transferred_strength = aspect['strength'] * efficiency
                
                # Add to transferable aspects
                transferable_aspects[name] = {
                    'original_strength': aspect['strength'],
                    'transfer_efficiency': efficiency,
                    'transferred_strength': transferred_strength,
                    'aspect_type': aspect_type.name,
                    'frequency': aspect['frequency'],
                    'color': aspect['color'],
                    'element': aspect['element'],
                    'description': aspect['description'],
                    'keywords': aspect['keywords']
                }
        
        # Calculate transfer metrics
        total_original = sum(a['original_strength'] for a in transferable_aspects.values())
        total_transferred = sum(a['transferred_strength'] for a in transferable_aspects.values())
        average_efficiency = total_transferred / total_original if total_original > 0 else 0
        
        # Update soul with transferred aspects
        if hasattr(soul_spark, 'creator_aspects'):
            soul_spark.creator_aspects = transferable_aspects
        
        # Track overall transfer impact on soul properties
        if hasattr(soul_spark, 'creator_alignment'):
            soul_spark.creator_alignment = min(1.0, soul_spark.creator_alignment + 0.1 * average_efficiency)
            
        if hasattr(soul_spark, 'stability'):
            soul_spark.stability = min(1.0, soul_spark.stability + 0.05 * average_efficiency)
            
        # Store transfer metrics
        transfer_metrics = {
            'channel_id': channel_id,
            'spark_id': channel['spark_id'],
            'aspect_count': len(transferable_aspects),
            'total_original_strength': total_original,
            'total_transferred_strength': total_transferred,
            'average_efficiency': average_efficiency,
            'transfer_time': datetime.now().isoformat(),
            'transferred_aspects': transferable_aspects
        }
        
        self.aspect_transfer_metrics[channel_id] = transfer_metrics
        
        logger.info(f"Transferred {len(transferable_aspects)} creator aspects to spark {channel['spark_id']}")
        logger.info(f"Average transfer efficiency: {average_efficiency:.4f}")
        
        return transfer_metrics
    
    def stabilize_soul_creator_relationship(self, soul_spark, channel_id=None, iterations=3):
        """
        Stabilize the quantum relationship between soul and creator.
        
        This iterative process strengthens the connection and ensures
        long-term stability for the soul's journey.
        
        Args:
            soul_spark: The SoulSpark object
            channel_id (str): Optional specific channel ID to use
            iterations (int): Number of stabilization iterations
            
        Returns:
            dict: Information about the stabilization process
        """
        # Get channel ID if not provided
        if channel_id is None:
            if hasattr(soul_spark, 'creator_channel_id'):
                channel_id = soul_spark.creator_channel_id
            else:
                # No channel exists yet, create full connection
                channel = self.establish_quantum_connection(soul_spark)
                channel_id = channel['channel_id']
                self.form_resonance_patterns(soul_spark, channel_id)
                self.transfer_creator_aspects(soul_spark, channel_id)
        
        # Verify channel exists with all prerequisite processes
        if channel_id not in self.quantum_channels:
            logger.warning(f"Channel {channel_id} not found. Creating complete connection.")
            channel = self.establish_quantum_connection(soul_spark)
            channel_id = channel['channel_id']
            self.form_resonance_patterns(soul_spark, channel_id)
            self.transfer_creator_aspects(soul_spark, channel_id)
            
        # Get the channel
        channel = self.quantum_channels[channel_id]
        
        # Track metrics for stabilization process
        initial_metrics = {
            'connection_strength': channel['connection_strength'],
            'connection_quality': channel['connection_quality']
        }
        
        if hasattr(soul_spark, 'stability'):
            initial_metrics['stability'] = soul_spark.stability
            
        if hasattr(soul_spark, 'creator_alignment'):
            initial_metrics['creator_alignment'] = soul_spark.creator_alignment
            
        if hasattr(soul_spark, 'resonance'):
            initial_metrics['resonance'] = soul_spark.resonance
            
        # Perform stabilization iterations
        for i in range(iterations):
            # Strengthen connection through iterative resonance
            if channel_id in self.resonance_patterns:
                patterns = self.resonance_patterns[channel_id]['patterns']
                
                # Increase strength of each pattern slightly
                for name in patterns:
                    patterns[name]['strength'] = min(1.0, patterns[name]['strength'] * 1.05)
                    
                # Update pattern coherence
                pattern_strengths = [p['strength'] for p in patterns.values()]
                total_strength = sum(pattern_strengths)
                pattern_coherence = total_strength / len(patterns) if patterns else 0
                self.resonance_patterns[channel_id]['pattern_coherence'] = pattern_coherence
                
                # Update soul property if available
                if hasattr(soul_spark, 'pattern_coherence'):
                    soul_spark.pattern_coherence = pattern_coherence
            
            # Strengthen quantum channel
            channel['connection_strength'] = min(1.0, channel['connection_strength'] * 1.03)
            channel['connection_quality'] = min(1.0, channel['connection_quality'] * 1.02)
            
            # Update soul properties
            if hasattr(soul_spark, 'creator_connection_strength'):
                soul_spark.creator_connection_strength = channel['connection_strength']
                
            if hasattr(soul_spark, 'creator_alignment'):
                soul_spark.creator_alignment = min(1.0, soul_spark.creator_alignment * 1.03)
                
            if hasattr(soul_spark, 'stability'):
                soul_spark.stability = min(1.0, soul_spark.stability * 1.02)
                
            if hasattr(soul_spark, 'resonance'):
                soul_spark.resonance = min(1.0, soul_spark.resonance * 1.02)
                
            logger.info(f"Completed stabilization iteration {i+1}/{iterations} for spark {channel['spark_id']}")
            
        # Calculate final metrics
        final_metrics = {
            'connection_strength': channel['connection_strength'],
            'connection_quality': channel['connection_quality']
        }
        
        if hasattr(soul_spark, 'stability'):
            final_metrics['stability'] = soul_spark.stability
            
        if hasattr(soul_spark, 'creator_alignment'):
            final_metrics['creator_alignment'] = soul_spark.creator_alignment
            
        if hasattr(soul_spark, 'resonance'):
            final_metrics['resonance'] = soul_spark.resonance
            
        # Calculate improvements
        improvements = {
            key: final_metrics[key] - initial_metrics[key]
            for key in final_metrics
            if key in initial_metrics
        }
        
        # Store stabilization results
        stabilization_result = {
            'channel_id': channel_id,
            'spark_id': channel['spark_id'],
            'iterations': iterations,
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'improvements': improvements,
            'stabilization_time': datetime.now().isoformat()
        }
        
        logger.info(f"Completed soul-creator relationship stabilization for spark {channel['spark_id']}")
        logger.info(f"Connection strength improved by {improvements.get('connection_strength', 0):.4f}")
        logger.info(f"Creator alignment improved by {improvements.get('creator_alignment', 0):.4f}")
        
        return stabilization_result
    
    def enable_bidirectional_communication(self, soul_spark, channel_id=None, basic_only=True):
        """
        Enable bidirectional communication between soul and creator.
        
        This establishes channels for information and energy exchange. If basic_only
        is True, only implements simple energy transfer without full gateway mechanics.
        
        Args:
            soul_spark: The SoulSpark object
            channel_id (str): Optional specific channel ID to use
            basic_only (bool): Whether to implement only basic communication without gateways
            
        Returns:
            dict: Information about the communication channels
        """
        # Get channel ID if not provided
        if channel_id is None:
            if hasattr(soul_spark, 'creator_channel_id'):
                channel_id = soul_spark.creator_channel_id
            else:
                # No channel exists yet, create one
                channel = self.establish_quantum_connection(soul_spark)
                channel_id = channel['channel_id']
        
        # Verify channel exists
        if channel_id not in self.quantum_channels:
            logger.warning(f"Channel {channel_id} not found. Creating new channel.")
            channel = self.establish_quantum_connection(soul_spark)
            channel_id = channel['channel_id']
            
        # Get the channel
        channel = self.quantum_channels[channel_id]
        
        # Get or create resonance patterns
        if channel_id not in self.resonance_patterns:
            self.form_resonance_patterns(soul_spark, channel_id)
            
        resonance_data = self.resonance_patterns[channel_id]
        
        # Create communication channels
        communication_channels = {
            'energy_transfer': {
                'direction': 'bidirectional',
                'bandwidth': 0.8 * channel['connection_strength'],
                'stability': 0.7 * channel['connection_quality'],
                'active': True
            },
            'aspect_transfer': {
                'direction': 'creator_to_soul',
                'bandwidth': 0.7 * channel['connection_strength'],
                'stability': 0.8 * channel['connection_quality'],
                'active': True
            },
            'resonance_feedback': {
                'direction': 'bidirectional',
                'bandwidth': 0.6 * channel['connection_strength'],
                'stability': 0.75 * channel['connection_quality'],
                'active': True
            }
        }
        
        # If not basic only, add gateway-related channels
        if not basic_only:
            # These would require gateway implementation
            communication_channels['information_exchange'] = {
                'direction': 'bidirectional',
                'bandwidth': 0.5 * channel['connection_strength'],
                'stability': 0.6 * channel['connection_quality'],
                'active': False,  # Not active until gateway implementation
                'requires_gateway': True
            }
            
            communication_channels['consciousness_stream'] = {
                'direction': 'bidirectional',
                'bandwidth': 0.4 * channel['connection_strength'],
                'stability': 0.5 * channel['connection_quality'],
                'active': False,  # Not active until gateway implementation
                'requires_gateway': True
            }
            
        # Update channel with communication capabilities
        channel['communication_channels'] = communication_channels
        
        # Update soul properties if supported
        if hasattr(soul_spark, 'communication_channels'):
            soul_spark.communication_channels = communication_channels
            
        # Calculate overall communication capability
        active_channels = [c for c in communication_channels.values() if c['active']]
        
        if active_channels:
            avg_bandwidth = sum(c['bandwidth'] for c in active_channels) / len(active_channels)
            avg_stability = sum(c['stability'] for c in active_channels) / len(active_channels)
            
            comm_capability = avg_bandwidth * avg_stability
        else:
            comm_capability = 0
            
        # Store communication capability
        channel['communication_capability'] = comm_capability
        
        if hasattr(soul_spark, 'communication_capability'):
            soul_spark.communication_capability = comm_capability
            
        logger.info(f"Enabled {len(active_channels)} communication channels for spark {channel['spark_id']}")
        logger.info(f"Communication capability: {comm_capability:.4f}")
        
        return {
            'channel_id': channel_id,
            'spark_id': channel['spark_id'],
            'communication_channels': communication_channels,
            'communication_capability': comm_capability,
            'basic_only': basic_only,
            'enabling_time': datetime.now().isoformat()
        }
    
    def run_full_entanglement_process(self, soul_spark, with_communication=True, basic_comm_only=True):
        """
        Run the complete entanglement process for a soul spark.
        
        This executes all steps from connection establishment to stabilization.
        
        Args:
            soul_spark: The SoulSpark object to entangle
            with_communication (bool): Whether to enable communication channels
            basic_comm_only (bool): Whether to use only basic communication without gateways
            
        Returns:
            dict: Results of the full entanglement process
        """
        # Step 1: Establish quantum connection
        connection = self.establish_quantum_connection(soul_spark)
        channel_id = connection['channel_id']
        
        # Step 2: Form resonance patterns
        resonance = self.form_resonance_patterns(soul_spark, channel_id)
        
        # Step 3: Transfer creator aspects
        transfer = self.transfer_creator_aspects(soul_spark, channel_id)
        
        # Step 4: Stabilize the relationship
        stabilization = self.stabilize_soul_creator_relationship(soul_spark, channel_id, iterations=3)
        
        # Step 5: Enable communication if requested
        communication = None
        if with_communication:
            communication = self.enable_bidirectional_communication(
                soul_spark, 
                channel_id,
                basic_only=basic_comm_only
            )
            
        # Compile the complete results
        results = {
            'spark_id': soul_spark.spark_id if hasattr(soul_spark, 'spark_id') else 'unknown',
            'channel_id': channel_id,
            'process_time': datetime.now().isoformat(),
            'connection': connection,
            'resonance': resonance,
            'transfer': transfer,
            'stabilization': stabilization,
            'communication': communication
        }
        
        # Update any final properties on the soul spark
        if hasattr(soul_spark, 'entanglement_complete'):
            soul_spark.entanglement_complete = True
        
        logger.info(f"Completed full entanglement process for spark {results['spark_id']}")
        logger.info(f"Final creator alignment: {stabilization['final_metrics'].get('creator_alignment', 'N/A')}")
        
        return results
    
    def save_entanglement_data(self, output_dir="output", filename=None):
        """
        Save all entanglement data to file.
        
        Args:
            output_dir (str): Directory to save output files
            filename (str): Optional custom filename
            
        Returns:
            str: Path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"creator_entanglement_{self.entanglement_id[:8]}.json"
            
        save_path = os.path.join(output_dir, filename)
        
        # Compile entanglement data
        entanglement_data = {
            'entanglement_id': self.entanglement_id,
            'creation_time': self.creation_time,
            'creator_resonance': self.creator_resonance,
            'edge_of_chaos_ratio': self.edge_of_chaos_ratio,
            'quantum_channels': self.quantum_channels,
            'resonance_patterns': self.resonance_patterns,
            'aspect_transfer_metrics': self.aspect_transfer_metrics,
            'export_time': datetime.now().isoformat()
        }
        
        # Save to file
        try:
            with open(save_path, 'w') as f:
                json.dump(entanglement_data, f, indent=2)
                
            logger.info(f"Entanglement data saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving entanglement data: {str(e)}")
            return None
    
    def get_channel_info(self, channel_id):
        """
        Get information about a specific quantum channel.
        
        Args:
            channel_id (str): The channel ID to retrieve
            
        Returns:
            dict: Channel information or None if not found
        """
        return self.quantum_channels.get(channel_id)
    
    def get_spark_channels(self, spark_id):
        """
        Get all channels for a specific soul spark.
        
        Args:
            spark_id (str): The spark ID to search for
            
        Returns:
            list: List of channel information for this spark
        """
        return [
            channel for channel in self.quantum_channels.values()
            if channel['spark_id'] == spark_id
        ]
    
    def __str__(self):
        """String representation of the Creator Entanglement processor."""
        active_channels = sum(1 for c in self.quantum_channels.values() if c['active'])
        
        return (f"Creator Entanglement (ID: {self.entanglement_id})\n"
                f"Creation Time: {self.creation_time}\n"
                f"Creator Resonance: {self.creator_resonance}\n"
                f"Edge of Chaos Ratio: {self.edge_of_chaos_ratio}\n"
                f"Quantum Channels: {len(self.quantum_channels)} ({active_channels} active)\n"
                f"Resonance Patterns: {len(self.resonance_patterns)}\n"
                f"Aspect Transfers: {len(self.aspect_transfer_metrics)}")


if __name__ == "__main__":
    # Example usage
    entanglement = CreatorEntanglement(
        creator_resonance=0.8,
        edge_of_chaos_ratio=0.618
    )
    
    # To fully test this, we would need a SoulSpark object
    # Here's a simplified test using a mock spark
    try:
        from soul_formation.soul_spark import SoulSpark
        
        # Create a test spark
        spark = SoulSpark(creator_resonance=0.7)
        
        # Run the full entanglement process
        results = entanglement.run_full_entanglement_process(
            spark,
            with_communication=True,
            basic_comm_only=True
        )
        
        print("\nEntanglement Results:")
        print(f"Spark ID: {results['spark_id']}")
        print(f"Channel ID: {results['channel_id']}")
        print(f"Aspect Count: {results['transfer']['aspect_count']}")
        print(f"Transfer Efficiency: {results['transfer']['average_efficiency']:.4f}")
        
        if 'creator_alignment' in results['stabilization']['final_metrics']:
            alignment = results['stabilization']['final_metrics']['creator_alignment']
            print(f"Final Creator Alignment: {alignment:.4f}")
        
        # Save entanglement data
        save_path = entanglement.save_entanglement_data()
        if save_path:
            print(f"\nEntanglement data saved to: {save_path}")
            
    except ImportError:
        print("\nSoulSpark module not available for full testing")
        print("Creating minimal mock object for demonstration")
        
        # Create a minimal mock object
        class MockSpark:
            def __init__(self):
                self.spark_id = str(uuid.uuid4())
                self.stability = 0.7
                self.resonance = 0.7
                self.creator_alignment = 0.6
                self.frequency_signature = {
                    'frequencies': [432.0, 528.0, 639.0],
                    'amplitudes': [0.8, 0.7, 0.6],
                    'phases': [0, np.pi/4, np.pi/2],
                    'num_frequencies': 3
                }
        
        mock_spark = MockSpark()
        
        # Run basic entanglement with mock object
        channel = entanglement.establish_quantum_connection(mock_spark)
        print(f"\nEstablished channel {channel['channel_id']} for mock spark")
        print(f"Connection strength: {channel['connection_strength']:.4f}")
        
    print("\n" + str(entanglement))