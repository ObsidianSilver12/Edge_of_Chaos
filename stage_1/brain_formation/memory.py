# --- memory.py (V6.0.0 - Simplified Memory System) ---

"""
Memory3D - Simplified 3D memory coordinate system for baby brain.

Just coordinates + memory type for initial development.
Random distribution initially, only map to major regions when becoming nodes.
No complex tags/knowledge graphs yet - save for later learning phases.
"""

import logging
import uuid
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Import constants
from constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("Memory3D")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Memory type to major region mapping (for when fragments become nodes)
MEMORY_TYPE_TO_MAJOR_REGION = {
    'survival': 'brain_stem',
    'emotional': 'limbic', 
    'procedural': 'cerebellum',
    'semantic': 'temporal',
    'episodic': 'temporal',
    'working': 'frontal',
    'dimensional': 'parietal',
    'ephemeral': 'frontal'
}

# Memory type frequencies (Hz)
MEMORY_TYPE_FREQUENCIES = {
    'survival': 4.2,
    'emotional': 5.8,
    'procedural': 9.3,
    'semantic': 11.7,
    'episodic': 7.1,
    'working': 14.5,
    'dimensional': 18.3,
    'ephemeral': 23.7
}

# Memory type decay rates (per time unit)
MEMORY_TYPE_DECAY_RATES = {
    'survival': 0.0001,
    'emotional': 0.0008,
    'procedural': 0.00005,
    'semantic': 0.0003,
    'episodic': 0.001,
    'working': 0.05,
    'dimensional': 0.0002,
    'ephemeral': 0.1
}


class Memory3DClassification:
    """
    Memory3D - Simplified 3D memory coordinate system for baby brain.
    
    Key simplifications:
    - Just 3D coordinates + memory type
    - Random placement initially (not region-specific)
    - Only 8 memory types (no complex tags)
    - Convert to proper nodes when classified
    """
    
    def __init__(self, brain_structure=None):
        """Initialize memory system with brain structure reference."""
        self.memory_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure = brain_structure
        
        # Memory storage
        self.memory_fragments = {}  # fragment_id -> memory_fragment
        self.memory_nodes = {}      # node_id -> memory_node (classified fragments)
        self.active_memories = set()  # Set of active memory IDs
        
        # Coordinate tracking
        self.used_coordinates = set()  # Track used positions
        
        logger.info(f"Memory3D system initialized: {self.memory_id[:8]}")
    
    def get_random_coordinates(self) -> Tuple[int, int, int]:
        """
        Get random coordinates in brain space.
        Random distribution initially (not region-specific).
        """
        if not self.brain_structure:
            # Default brain dimensions if no structure
            x = random.randint(0, 255)
            y = random.randint(0, 255) 
            z = random.randint(0, 255)
        else:
            # Use actual brain dimensions
            x = random.randint(0, self.brain_structure.dimensions[0] - 1)
            y = random.randint(0, self.brain_structure.dimensions[1] - 1)
            z = random.randint(0, self.brain_structure.dimensions[2] - 1)
        
        return (x, y, z)
    
    def classify_memory_type(self, content: str, context: str = None) -> str:
        """
        Classify memory type based on content.
        Simple classification for baby brain.
        """
        content_lower = content.lower()
        
        # Simple keyword-based classification
        if any(word in content_lower for word in ['danger', 'safe', 'food', 'pain', 'comfort']):
            return 'survival'
        elif any(word in content_lower for word in ['happy', 'sad', 'love', 'fear', 'angry']):
            return 'emotional'
        elif any(word in content_lower for word in ['how', 'move', 'walk', 'grab', 'touch']):
            return 'procedural'
        elif any(word in content_lower for word in ['word', 'meaning', 'name', 'what', 'definition']):
            return 'semantic'
        elif any(word in content_lower for word in ['when', 'where', 'yesterday', 'remember', 'happened']):
            return 'episodic'
        elif any(word in content_lower for word in ['think', 'now', 'current', 'doing', 'focus']):
            return 'working'
        elif any(word in content_lower for word in ['shape', 'size', 'space', 'where', 'direction']):
            return 'dimensional'
        else:
            return 'ephemeral'  # Default for unclassified
    
    def add_memory_fragment(self, content: str, memory_type: str = None, 
                           coordinates: Tuple[int, int, int] = None) -> Dict[str, Any]:
        """
        Add memory fragment with random placement.
        """
        fragment_id = str(uuid.uuid4())
        
        # Auto-classify if not provided
        if memory_type is None:
            memory_type = self.classify_memory_type(content)
        
        # Get random coordinates if not provided
        if coordinates is None:
            coordinates = self.get_random_coordinates()
        
        # Create memory fragment
        fragment = {
            'id': fragment_id,
            'content': content,
            'memory_type': memory_type,
            'coordinates': coordinates,
            'frequency': MEMORY_TYPE_FREQUENCIES[memory_type],
            'decay_rate': MEMORY_TYPE_DECAY_RATES[memory_type],
            'active': True,
            'created_time': datetime.now().isoformat(),
            'access_count': 0,
            'last_accessed': None,
            'classification_level': 1  # Just basic classification
        }
        
        # Store fragment
        self.memory_fragments[fragment_id] = fragment
        self.active_memories.add(fragment_id)
        self.used_coordinates.add(coordinates)
        
        # Place in brain structure if available
        if self.brain_structure:
            self.brain_structure.set_field_value(coordinates, 'memory_fragment', fragment_id)
            self.brain_structure.set_field_value(coordinates, 'memory_type', memory_type)
            self.brain_structure.set_field_value(coordinates, 'memory_frequency', fragment['frequency'])
        
        logger.debug(f"Memory fragment created: {memory_type} at {coordinates}")
        
        return fragment
    
    def memory_type_mapping(self, memory_type: str) -> str:
        """
        Map memory type to target major region (for when becoming nodes).
        """
        return MEMORY_TYPE_TO_MAJOR_REGION.get(memory_type, 'frontal')
    
    def find_position_near_major_region(self, region_name: str) -> Tuple[int, int, int]:
        """
        Find position near specified major region.
        Used when converting fragments to nodes.
        """
        if not self.brain_structure or region_name not in self.brain_structure.regions:
            # Fallback to random position
            return self.get_random_coordinates()
        
        region = self.brain_structure.regions[region_name]
        center = region['center']
        radius = int(region['radius'] * 0.8)  # Stay within 80% of region
        
        # Try to find position near region center
        for attempt in range(10):
            # Random offset from center
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            dz = random.randint(-radius, radius)
            
            x = max(0, min(self.brain_structure.dimensions[0] - 1, center[0] + dx))
            y = max(0, min(self.brain_structure.dimensions[1] - 1, center[1] + dy))
            z = max(0, min(self.brain_structure.dimensions[2] - 1, center[2] + dz))
            
            position = (x, y, z)
            
            # Use position if not already occupied
            if position not in self.used_coordinates:
                return position
        
        # Fallback if no free position found
        return self.get_random_coordinates()
    
    def convert_memory_to_node(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert memory fragment to proper node when sufficiently classified.
        Only map to major region when becoming node.
        """
        if fragment_id not in self.memory_fragments:
            logger.warning(f"Fragment {fragment_id} not found for conversion")
            return None
        
        fragment = self.memory_fragments[fragment_id]
        
        # Check if fragment is sufficiently accessed/classified
        if fragment['access_count'] < 3:  # Minimum access threshold
            logger.debug(f"Fragment {fragment_id} not accessed enough for node conversion")
            return None
        
        # Create node from fragment
        node_id = str(uuid.uuid4())
        
        # Get target major region for this memory type
        target_region = self.memory_type_mapping(fragment['memory_type'])
        new_coordinates = self.find_position_near_major_region(target_region)
        
        # Create memory node
        node = {
            'id': node_id,
            'original_fragment_id': fragment_id,
            'content': fragment['content'],
            'memory_type': fragment['memory_type'],
            'coordinates': new_coordinates,
            'frequency': fragment['frequency'],
            'decay_rate': fragment['decay_rate'],
            'active': True,
            'node_type': 'classified_memory_node',
            'target_region': target_region,
            'created_time': fragment['created_time'],
            'converted_time': datetime.now().isoformat(),
            'access_count': fragment['access_count'],
            'classification_level': 2  # Upgraded to node level
        }
        
        # Remove old coordinates from brain structure
        if self.brain_structure:
            old_coords = fragment['coordinates']
            self.brain_structure.set_field_value(old_coords, 'memory_fragment', None)
            self.brain_structure.set_field_value(old_coords, 'memory_type', None)
            self.used_coordinates.discard(old_coords)
            
            # Set new coordinates
            self.brain_structure.set_field_value(new_coordinates, 'memory_node', node_id)
            self.brain_structure.set_field_value(new_coordinates, 'memory_type', node['memory_type'])
            self.brain_structure.set_field_value(new_coordinates, 'memory_frequency', node['frequency'])
            self.used_coordinates.add(new_coordinates)
        
        # Store node and remove fragment
        self.memory_nodes[node_id] = node
        del self.memory_fragments[fragment_id]
        self.active_memories.discard(fragment_id)
        self.active_memories.add(node_id)
        
        logger.info(f"Memory fragment converted to node: {fragment['memory_type']} -> {target_region}")
        
        return node
    
    def retrieve_memory(self, memory_id: str = None, memory_type: str = None, 
                       coordinates: Tuple[int, int, int] = None, content_search: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve memory fragments or nodes based on criteria.
        """
        results = []
        
        # Search in fragments
        for frag_id, fragment in self.memory_fragments.items():
            if memory_id and frag_id != memory_id:
                continue
            if memory_type and fragment['memory_type'] != memory_type:
                continue
            if coordinates and fragment['coordinates'] != coordinates:
                continue
            if content_search and content_search.lower() not in fragment['content'].lower():
                continue
            
            # Update access info
            fragment['access_count'] += 1
            fragment['last_accessed'] = datetime.now().isoformat()
            
            results.append(fragment.copy())
        
        # Search in nodes
        for node_id, node in self.memory_nodes.items():
            if memory_id and node_id != memory_id:
                continue
            if memory_type and node['memory_type'] != memory_type:
                continue
            if coordinates and node['coordinates'] != coordinates:
                continue
            if content_search and content_search.lower() not in node['content'].lower():
                continue
            
            # Update access info
            node['access_count'] += 1
            node['last_accessed'] = datetime.now().isoformat()
            
            results.append(node.copy())
        
        return results
    
    def memory_pattern_recognition(self) -> Dict[str, Any]:
        """
        Basic pattern recognition in memories.
        Triggered during dreaming state.
        """
        patterns_found = []
        
        # Group memories by type
        type_groups = {}
        for memory in list(self.memory_fragments.values()) + list(self.memory_nodes.values()):
            mem_type = memory['memory_type']
            if mem_type not in type_groups:
                type_groups[mem_type] = []
            type_groups[mem_type].append(memory)
        
        # Look for simple patterns within types
        for mem_type, memories in type_groups.items():
            if len(memories) >= 3:  # Need at least 3 for pattern
                # Simple content similarity pattern
                content_words = []
                for memory in memories:
                    content_words.extend(memory['content'].lower().split())
                
                # Find common words
                word_counts = {}
                for word in content_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                common_words = [word for word, count in word_counts.items() if count >= 2]
                
                if common_words:
                    patterns_found.append({
                        'pattern_type': 'content_similarity',
                        'memory_type': mem_type,
                        'common_elements': common_words[:5],  # Top 5 common words
                        'memory_count': len(memories),
                        'strength': len(common_words) / len(memories)
                    })
        
        pattern_metrics = {
            'patterns_found': len(patterns_found),
            'patterns': patterns_found,
            'recognition_time': datetime.now().isoformat()
        }
        
        logger.info(f"Pattern recognition found {len(patterns_found)} patterns")
        
        return pattern_metrics
    
    def memory_pruning(self) -> Dict[str, Any]:
        """
        Prune memories based on decay rate and frequency of use.
        Makes inactive memories removable to conserve energy.
        """
        pruned_fragments = 0
        pruned_nodes = 0
        
        current_time = datetime.now()
        
        # Prune fragments
        fragments_to_remove = []
        for frag_id, fragment in self.memory_fragments.items():
            # Calculate decay based on time and access
            created_time = datetime.fromisoformat(fragment['created_time'])
            age_hours = (current_time - created_time).total_seconds() / 3600
            
            # Apply decay rate
            decay_amount = fragment['decay_rate'] * age_hours
            
            # Reduce decay if frequently accessed
            if fragment['access_count'] > 0:
                decay_amount *= (1.0 / (1.0 + fragment['access_count'] * 0.1))
            
            # Mark for removal if highly decayed
            if decay_amount > 0.8:  # 80% decay threshold
                fragments_to_remove.append(frag_id)
        
        # Remove decayed fragments
        for frag_id in fragments_to_remove:
            fragment = self.memory_fragments[frag_id]
            
            # Remove from brain structure
            if self.brain_structure:
                coords = fragment['coordinates']
                self.brain_structure.set_field_value(coords, 'memory_fragment', None)
                self.brain_structure.set_field_value(coords, 'memory_type', None)
                self.used_coordinates.discard(coords)
            
            # Mark as inactive (save coordinates for potential reactivation)
            fragment['active'] = False
            fragment['pruned_time'] = datetime.now().isoformat()
            
            del self.memory_fragments[frag_id]
            self.active_memories.discard(frag_id)
            pruned_fragments += 1
        
        # Prune nodes (less aggressive)
        nodes_to_remove = []
        for node_id, node in self.memory_nodes.items():
            created_time = datetime.fromisoformat(node['created_time'])
            age_hours = (current_time - created_time).total_seconds() / 3600
            
            # Nodes decay slower
            decay_amount = node['decay_rate'] * age_hours * 0.5
            
            # Even more resistance for frequently accessed nodes
            if node['access_count'] > 0:
                decay_amount *= (1.0 / (1.0 + node['access_count'] * 0.2))
            
            # Higher threshold for node removal
            if decay_amount > 0.9:  # 90% decay threshold
                nodes_to_remove.append(node_id)
        
        # Remove decayed nodes
        for node_id in nodes_to_remove:
            node = self.memory_nodes[node_id]
            
            # Remove from brain structure
            if self.brain_structure:
                coords = node['coordinates']
                self.brain_structure.set_field_value(coords, 'memory_node', None)
                self.brain_structure.set_field_value(coords, 'memory_type', None)
                self.used_coordinates.discard(coords)
            
            # Mark as inactive
            node['active'] = False
            node['pruned_time'] = datetime.now().isoformat()
            
            del self.memory_nodes[node_id]
            self.active_memories.discard(node_id)
            pruned_nodes += 1
        
        pruning_metrics = {
            'pruned_fragments': pruned_fragments,
            'pruned_nodes': pruned_nodes,
            'total_pruned': pruned_fragments + pruned_nodes,
            'active_fragments_remaining': len(self.memory_fragments),
            'active_nodes_remaining': len(self.memory_nodes),
            'pruning_time': datetime.now().isoformat()
        }
        
        logger.info(f"Memory pruning complete: {pruned_fragments} fragments, {pruned_nodes} nodes pruned")
        
        return pruning_metrics
    
    def memory_to_node_trigger(self) -> List[Dict[str, Any]]:
        """
        Check fragments for node conversion eligibility.
        Triggered event for monitoring when fragments become nodes.
        """
        conversions = []
        
        # Check all fragments for conversion eligibility
        for frag_id, fragment in list(self.memory_fragments.items()):
            # Criteria for node conversion:
            # 1. Accessed at least 3 times
            # 2. Has been active for some time
            # 3. Content is not too ephemeral
            
            if (fragment['access_count'] >= 3 and 
                fragment['memory_type'] != 'ephemeral' and
                fragment['active']):
                
                node = self.convert_memory_to_node(frag_id)
                if node:
                    conversions.append({
                        'original_fragment_id': frag_id,
                        'new_node_id': node['id'],
                        'memory_type': node['memory_type'],
                        'target_region': node['target_region'],
                        'conversion_time': node['converted_time']
                    })
        
        logger.info(f"Memory to node conversions: {len(conversions)} fragments converted")
        
        return conversions
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        # Count by type
        fragment_counts = {}
        node_counts = {}
        
        for fragment in self.memory_fragments.values():
            mem_type = fragment['memory_type']
            fragment_counts[mem_type] = fragment_counts.get(mem_type, 0) + 1
        
        for node in self.memory_nodes.values():
            mem_type = node['memory_type']
            node_counts[mem_type] = node_counts.get(mem_type, 0) + 1
        
        return {
            'memory_system_id': self.memory_id,
            'creation_time': self.creation_time,
            'total_fragments': len(self.memory_fragments),
            'total_nodes': len(self.memory_nodes),
            'total_active': len(self.active_memories),
            'fragment_counts_by_type': fragment_counts,
            'node_counts_by_type': node_counts,
            'coordinates_used': len(self.used_coordinates),
            'brain_structure_connected': self.brain_structure is not None
        }


# === UTILITY FUNCTIONS ===

def create_memory_system(brain_structure=None) -> Memory3DClassification:
    """Create memory system with optional brain structure."""
    return Memory3DClassification(brain_structure)


def demonstrate_memory_system():
    """Demonstrate the simplified memory system."""
    print("\n=== Memory3D System Demonstration ===")
    
    try:
        # Create memory system
        memory_system = create_memory_system()
        
        print("1. Adding memory fragments...")
        
        # Add various memory types
        test_memories = [
            ("I saw a red circle", "episodic"),
            ("Red means stop", "semantic"),
            ("How to grab things", "procedural"),
            ("Mom's voice is comforting", "emotional"),
            ("Stay away from hot stove", "survival"),
            ("Currently thinking about shapes", "working"),
            ("The ball is round", "dimensional"),
            ("Random thought", "ephemeral")
        ]
        
        fragments_added = 0
        for content, mem_type in test_memories:
            fragment = memory_system.add_memory_fragment(content, mem_type)
            fragments_added += 1
            print(f"   Added {mem_type}: {content[:30]}...")
        
        print(f"   Total fragments added: {fragments_added}")
        
        print("2. Accessing memories to increase usage...")
        
        # Access some memories multiple times
        all_fragments = list(memory_system.memory_fragments.keys())
        for i in range(5):  # 5 access cycles
            for frag_id in all_fragments[:4]:  # Access first 4 fragments
                results = memory_system.retrieve_memory(memory_id=frag_id)
                if results:
                    print(f"   Accessed: {results[0]['memory_type']}")
        
        print("3. Converting fragments to nodes...")
        
        # Trigger fragment to node conversions
        conversions = memory_system.memory_to_node_trigger()
        print(f"   Conversions made: {len(conversions)}")
        
        for conversion in conversions:
            print(f"   {conversion['memory_type']} -> {conversion['target_region']}")
        
        print("4. Pattern recognition...")
        
        # Run pattern recognition
        patterns = memory_system.memory_pattern_recognition()
        print(f"   Patterns found: {patterns['patterns_found']}")
        
        print("5. Memory pruning...")
        
        # Run memory pruning
        pruning_result = memory_system.memory_pruning()
        print(f"   Fragments pruned: {pruning_result['pruned_fragments']}")
        print(f"   Nodes pruned: {pruning_result['pruned_nodes']}")
        
        # Final statistics
        stats = memory_system.get_memory_statistics()
        print(f"\nFinal Statistics:")
        print(f"   Total Fragments: {stats['total_fragments']}")
        print(f"   Total Nodes: {stats['total_nodes']}")
        print(f"   Total Active: {stats['total_active']}")
        print(f"   Coordinates Used: {stats['coordinates_used']}")
        
        print("\nMemory system demonstration completed successfully!")
        
        return memory_system
        
    except Exception as e:
        print(f"ERROR: Memory system demonstration failed: {e}")
        return None


# === MAIN EXECUTION ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demonstrate memory system
    demo_memory = demonstrate_memory_system()
    
    if demo_memory:
        print("\nSimplified Memory3D system ready for baby brain!")
    else:
        print("\nERROR: Memory system demonstration failed")

# --- End of memory.py ---


# define difference between a seed, a fragment and a node.
# a seed is an empty structure that is used as a vessel to channel energy though for soul spark and the brain spark.
# a fragment is a memory that has little or no classification yet that must be stored in the 3d space used for ephemeral
# memories. a node is a thought that is classified and stored in the 3d space.most thoughts start from a memory fragment 
# its something model sees, hears,touches, smells, tastes, feels, thinks, imagines, dreams, remembers, etc.
# the baby brain isnt that complex and initial processes need to be relatively simple but encompass the range of processes
# we expect the brain and mycelial network to perform. however the processes will be more flexible/neuroplastic and simplified
# to start with.
# can refer to memory structure and functions for more of the conceptual information  - however we have not firmed up the 
# exact hierarchy that will work best would refine that based on training cycles after brain developed.


# class Memory3DClassification:
#     """
#     Memory3D - create the 3D memory coordinate system for the brain.
#     """
#     def __init__(self):

    
#     def classify_memory_type():
#         """
#         Classify memory type in accordance with assumed memory type with time-decay. this will determine where 
#         the memory is stored in the 3D space and will determine when a trigger process must fire to decay the memory.
#         the memory type is mapped to brain regions and will be assigned to that region in the 3D space
#         in accordance with the full coordinate mapping. memory types: survival, emotional, procedural, semantic, 
#         episodic, working, dimensional, ephemeral.all memories without proper categorisation are assigned to the
#         ephemeral type memory type initially. 
#         """

#     def memory_type_mapping():
#         """
#         map memory types to specific brain regions and in accordance with the coordinates assigned. if a region 
#         is full use place in nearest region until the memory is decayed. decayed memories are removed from the 3D space.
#         but original coordinates are saved. If a thought process evokes a pathway with decayed memories they can be revived
#         by triggering the memory remap process. this will be more natural in process i.e. will work on associations so not
#         all decayed get revived. the higher the association the more likely the memory will be revived.
#         """

#     def domain_mapping():
#         """
#         map memory fragments to specific domains once known. we will start with a high-level list of domains to start with based on
#         a babies known universe and then we will refine the list as we go along. domains are more detailed than categories to avoid oversimplification and to 
#         allow for more flexibility by creating a wider search net to start from. for example language is too narrow a classification 
#         so a broader domain example could be english language. basic initial domains: personal development, english language, universal patterns, 
#         spiritual awakening, the 6 senses,  geometry, algebra, childhood learning
#         """

#     def temporal_mapping():
#         """
#         map the temporal coordinate ie the time and space of a memory. closer in time and space for recent, further away for more distant memories.
#         temporal references decay over time but also depend on usage frequency. the more the knowledge is accessed the more time decay is put off.
#         define decay as a field of temporal mapping
#         """

#     def concepts_mapping():
#         """
#         map the concepts related to the domains and assign coordinates to them based on the concept classification. Example: childhood learning concepts: 
#         colors, shapes, numbers, letters, sounds, textures, smells, tastes, movements, objects, actions, events, stories, songs, games, puzzles. 
#         english language: vocabulary, grammar, punctuation, phonetics, antonyms and synonyms, spelling etc.
#         personal development: moral values, ethics, personal values, personal beliefs, personal growth, empathy, compassion, forgiveness, gratitude
#         spiritual awakening: enlightenment, vibrations, intuition, soul, spirit, consciousness, universal awareness, global religions 
#         universal patterns: complexity theory, recursion, fractals, chaos theory, self-organization, emergence, holons, sacred geometry, dna, cymatics, astrology, numerology
#         the sixth senses: sounds, sight, touch, taste, smell, intuition
#         geometry: angles, shapes, 3d, 2d space, fundamentals
#         algebra: fundamentals
#         """

#     def related_concepts_mapping():
#         """
#         map the related concepts related to the concepts and assign coordinates to them based on the related concept classification. 
       
#         """

#     def assign_coordinates():
#         """
#         Assign coordinates to memory fragments based on the wbs type classification level 1-domain, level 2-concept, level 3 related concept
#         for level 3 we need to determine if we should break it down further or if it is sufficient to leave it as is.
#         """

#     def additive_tags_mapping():
#         """
#         map the additional tags to concepts and related concepts for better search and understanding 
#         we may need to understand this level a bit more as to how we can classify cleanly. then do we add in: synonyms,
#         similar hypotheses, pattern similarity, resonance, semantic meaning, semantic similarity, opposites, personal meta tags etc to 
#         the additonal fields of each node or do we determine that certain things should be nodes on its own. The purpose of the 
#         additional tags is to have personal tags, additive information etc for better
#         search capability and to help with the understanding of the concepts, better reasoning etc. So instead of just big blobs of 
#         info we have fields that can be used to find better patterns or to search more effectively.
#         """

#     def retrieve_memory():
#         """
#         retrieve memory from memory fragments based on the wbs type classification level 1-domain, level 2-concept, level 3 related concept, 
#         or meta tags or frequency or memory type and/or one of the other fields. 
#         """

#     def retrieve_node():
#         """
#         retrieve a node based on the wbs type classification level 1-domain, level 2-concept, level 3 related concept, 
#         or meta tags
#         """

#     def memory_pattern_recognition():
#         """
#         this is a triggered event that always happens within the dreaming state. it is triggered from state management and runs this 
#         process to recognise patterns in the data. we will need to determine neural network strategies to use for this. so even though
#         we are processing from subconscious it is still going to use common strategies. strategies that may be different include: use
#         of additional tags to better match semantics, use of frequency or resonance to match patters, use of memory fragment frequency
#         to determine thoughts that need classification or reclassification if node became inactive over time as that node was not being
#         used frequently enough to prevent decay.
#         """

#     def memory_pruning():
#         """
#         memories are pruned based on decay rate and frequency of use. they become inactive with an inactive frequency that can be searched.
#         inactive memories or nodes (if classified as nodes) no longer exist in the brain grid as a coordinate. the coordinate is saved and 
#         if the memory/node becomes active again because it has a related concept or concept to something being thought about it will reactivate
#         if it has a high level of relevance to the subject. this makes it a more natural and realistic memory system. It also follows the 
#         principle of energy conservation.
#         """

#     def memory_to_node():
#         """
#         triggered event that happens when a memory is classified as a node because it meets this criteria: it has a level 1,2 and 3 classification
#         and is not a node already. it is then assigned a node coordinate and added to the brain grid.
#         """



