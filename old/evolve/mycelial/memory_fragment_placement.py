# memory_fragment_placement.py - Enhanced Network Integration

"""
Memory Fragment System Integration with Enhanced Mycelial Network

Adds optimal placement coordination while maintaining existing fragment functionality.
Integrates with enhanced network for placement optimization without content modification.
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("MemoryFragmentPlacement")


# === Enhanced Network Interface for Memory Fragment System ===

def add_enhanced_network_interface_to_memory_system(memory_fragment_system, enhanced_network):
    """
    Add enhanced network interface to existing memory fragment system.
    
    Args:
        memory_fragment_system: Existing MemoryFragmentSystem instance
        enhanced_network: Enhanced mycelial network for placement coordination
    """
    logger.info("Adding enhanced network interface to memory fragment system")
    
    # Store reference to enhanced network
    memory_fragment_system.enhanced_mycelial_network = enhanced_network
    
    # Store original add_fragment method
    memory_fragment_system._original_add_fragment = memory_fragment_system.add_fragment
    
    # Create enhanced add_fragment method
    def enhanced_add_fragment(self, content, region=None, position=None, 
                            frequency=None, meta_tags=None, origin="perception"):
        """
        Enhanced add_fragment with network coordination for optimal placement.
        Uses enhanced network for placement optimization when region not specified.
        """
        # If no region specified and enhanced network available, get optimal placement
        if not region and hasattr(self, 'enhanced_mycelial_network') and self.enhanced_mycelial_network:
            try:
                placement_result = self.enhanced_mycelial_network.coordinate_memory_placement(
                    content, target_region=None
                )
                
                if placement_result.get('success', False):
                    region = placement_result.get('region', 'temporal')
                    if not position:
                        position = placement_result.get('position')
                    
                    logger.debug(f"Enhanced network provided optimal placement: {region} at {position}")
                    
                    # Add placement metadata
                    if meta_tags is None:
                        meta_tags = {}
                    meta_tags['placement_method'] = 'enhanced_network_optimized'
                    meta_tags['placement_score'] = placement_result.get('placement_score', 0.0)
                
            except Exception as e:
                logger.warning(f"Enhanced network placement failed, using fallback: {e}")
                region = region or 'temporal'  # Fallback to temporal region
        
        # Use original add_fragment logic with potentially optimized region/position
        return self._original_add_fragment(content, region, position, frequency, meta_tags, origin)
    
    # Bind enhanced method to memory system
    import types
    memory_fragment_system.add_fragment = types.MethodType(enhanced_add_fragment, memory_fragment_system)
    
    # Add placement reporting method
    def report_placement_statistics(self):
        """Report placement statistics to enhanced network."""
        if hasattr(self, 'enhanced_mycelial_network') and self.enhanced_mycelial_network:
            try:
                # Count fragments by placement method
                placement_methods = {}
                optimized_placements = 0
                
                for fragment_id, fragment in self.fragments.items():
                    placement_method = fragment.get('meta_tags', {}).get('placement_method', 'standard')
                    placement_methods[placement_method] = placement_methods.get(placement_method, 0) + 1
                    
                    if placement_method == 'enhanced_network_optimized':
                        optimized_placements += 1
                
                # Report to enhanced network metrics
                if hasattr(self.enhanced_mycelial_network, 'metrics'):
                    self.enhanced_mycelial_network.metrics['optimized_fragment_placements'] = optimized_placements
                    self.enhanced_mycelial_network.metrics['total_fragment_placements'] = len(self.fragments)
                
                logger.info(f"Placement statistics: {optimized_placements}/{len(self.fragments)} optimized")
                
                return {
                    'total_fragments': len(self.fragments),
                    'optimized_placements': optimized_placements,
                    'placement_methods': placement_methods
                }
                
            except Exception as e:
                logger.error(f"Error reporting placement statistics: {e}")
                return {'error': str(e)}
        
        return {'error': 'Enhanced network not available'}
    
    # Bind reporting method
    memory_fragment_system.report_placement_statistics = types.MethodType(
        report_placement_statistics, memory_fragment_system
    )
    
    logger.info("Enhanced network interface added to memory fragment system")


# === Enhanced Mycelial Network Memory Coordination ===

def enhance_mycelial_network_memory_coordination(enhanced_network):
    """
    Add memory fragment coordination capabilities to enhanced mycelial network.
    
    Args:
        enhanced_network: Enhanced mycelial network instance
    """
    logger.info("Adding memory coordination to enhanced mycelial network")
    
    def coordinate_memory_placement(self, memory_fragment, target_region: str = None) -> Dict[str, Any]:
        """
        Coordinate optimal memory fragment placement using brain structure analysis.
        
        Args:
            memory_fragment: Memory fragment content
            target_region: Optional target region
            
        Returns:
            Dict with placement recommendations
        """
        if not self.brain_structure:
            return {'success': False, 'reason': 'No brain structure available'}
        
        try:
            # Determine optimal region based on current brain state and fragment type
            if not target_region:
                target_region = self._determine_optimal_region_for_fragment(memory_fragment)
            
            # Find optimal position within region
            optimal_position = self._find_optimal_position_in_region(target_region, memory_fragment)
            
            if optimal_position:
                # Calculate placement score based on multiple factors
                placement_score = self._calculate_placement_score(target_region, optimal_position, memory_fragment)
                
                return {
                    'success': True,
                    'region': target_region,
                    'position': optimal_position,
                    'placement_score': placement_score,
                    'optimization_factors': {
                        'brain_state': self.consciousness_orchestrator.current_state_name,
                        'region_activity': self._get_region_activity_level(target_region),
                        'field_strength': self._get_field_strength_at_position(optimal_position)
                    }
                }
            else:
                return {'success': False, 'reason': 'No suitable position found in target region'}
            
        except Exception as e:
            logger.error(f"Error coordinating memory placement: {e}")
            return {'success': False, 'error': str(e)}
    
    def _determine_optimal_region_for_fragment(self, memory_fragment) -> str:
        """Determine optimal brain region for memory fragment."""
        current_state = self.consciousness_orchestrator.current_state_name
        
        # Fragment type analysis
        fragment_str = str(memory_fragment).lower()
        
        # Emotional content → limbic
        if any(word in fragment_str for word in ['emotion', 'feeling', 'fear', 'joy', 'love', 'anger']):
            return 'limbic'
        
        # Visual content → occipital
        if any(word in fragment_str for word in ['visual', 'image', 'see', 'color', 'light']):
            return 'occipital'
        
        # Motor content → cerebellum
        if any(word in fragment_str for word in ['movement', 'motor', 'coordination', 'balance']):
            return 'cerebellum'
        
        # Language content → temporal
        if any(word in fragment_str for word in ['word', 'language', 'speech', 'sound', 'voice']):
            return 'temporal'
        
        # Executive/planning content → frontal
        if any(word in fragment_str for word in ['plan', 'decision', 'execute', 'control']):
            return 'frontal'
        
        # State-based preferences
        state_preferences = {
            'dream': 'temporal',        # Dreams in memory regions
            'aware': 'frontal',         # Awareness in executive regions
            'liminal': 'limbic',        # Integration in emotional regions
            'formation': 'brain_stem',  # Formation in core regions
            'soul_attached_settling': 'limbic'  # Soul settling in limbic
        }
        
        return state_preferences.get(current_state, 'temporal')  # Default to temporal
    
    def _find_optimal_position_in_region(self, region_name: str, memory_fragment) -> Optional[Tuple[int, int, int]]:
        """Find optimal position within specified brain region."""
        if region_name not in self.brain_structure.regions:
            logger.warning(f"Region {region_name} not found in brain structure")
            return None
        
        region = self.brain_structure.regions[region_name]
        center = region['center']
        radius = region['radius']
        
        # Sample positions within region
        best_position = None
        best_score = -1.0
        
        # Try multiple positions around region center
        import random
        for _ in range(20):  # Sample 20 positions
            # Generate position within region radius
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.uniform(0, radius * 0.8)  # Stay within 80% of radius
            
            offset_x = int(distance * math.cos(angle))
            offset_y = int(distance * math.sin(angle))
            offset_z = int(distance * math.sin(angle * 0.5))  # Less variation in z
            
            candidate_x = max(0, min(center[0] + offset_x, self.brain_structure.dimensions[0] - 1))
            candidate_y = max(0, min(center[1] + offset_y, self.brain_structure.dimensions[1] - 1))
            candidate_z = max(0, min(center[2] + offset_z, self.brain_structure.dimensions[2] - 1))
            
            candidate_position = (candidate_x, candidate_y, candidate_z)
            
            # Score this position
            score = self._score_position_for_placement(candidate_position, memory_fragment)
            
            if score > best_score:
                best_score = score
                best_position = candidate_position
        
        return best_position
    
    def _score_position_for_placement(self, position: Tuple[int, int, int], memory_fragment) -> float:
        """Score a position for memory fragment placement."""
        try:
            x, y, z = position
            score = 0.0
            
            # Factor 1: Energy level at position (higher is better)
            if hasattr(self.brain_structure, 'get_field_value'):
                energy = self.brain_structure.get_field_value(position, 'energy')
                score += energy * 0.3
            
            # Factor 2: Field dynamics strength
            if self.field_dynamics:
                field_value = self.field_dynamics.get_combined_field_value_at_point(position)
                score += field_value * 0.2
            
            # Factor 3: Avoid high stress areas
            current_stress = self.stress_monitor.current_stress_levels.get('overall', 0.0)
            stress_penalty = current_stress * 0.1
            score -= stress_penalty
            
            # Factor 4: Phi resonance zones (preferred)
            if hasattr(self.brain_structure, 'phi_resonance_zones'):
                for zone_id, zone in self.brain_structure.phi_resonance_zones.items():
                    zone_pos = zone.get('anchor_position', (0, 0, 0))
                    distance = math.sqrt((x - zone_pos[0])**2 + (y - zone_pos[1])**2 + (z - zone_pos[2])**2)
                    if distance <= zone.get('resonance_radius', 10):
                        score += 0.15  # Bonus for phi resonance zones
            
            # Factor 5: Edge of chaos zones (optimal for processing)
            if hasattr(self.brain_structure, 'edge_of_chaos_zones'):
                for zone_id, zone in self.brain_structure.edge_of_chaos_zones.items():
                    zone_pos = zone.get('position', (0, 0, 0))
                    distance = math.sqrt((x - zone_pos[0])**2 + (y - zone_pos[1])**2 + (z - zone_pos[2])**2)
                    if distance <= 5:  # Close to edge of chaos
                        score += zone.get('optimization_potential', 0.0) * 0.1
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring position {position}: {e}")
            return 0.0
    
    def _calculate_placement_score(self, region: str, position: Tuple[int, int, int], 
                                 memory_fragment) -> float:
        """Calculate overall placement score for the chosen region and position."""
        try:
            # Base score from position scoring
            position_score = self._score_position_for_placement(position, memory_fragment)
            
            # Regional appropriateness score
            region_score = self._score_region_appropriateness(region, memory_fragment)
            
            # Current brain state compatibility
            state_score = self._score_state_compatibility(region)
            
            # Weighted combination
            total_score = (position_score * 0.5 + region_score * 0.3 + state_score * 0.2)
            
            return min(1.0, max(0.0, total_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating placement score: {e}")
            return 0.5  # Neutral score on error
    
    def _score_region_appropriateness(self, region: str, memory_fragment) -> float:
        """Score how appropriate a region is for the memory fragment type."""
        fragment_str = str(memory_fragment).lower()
        
        # Region scoring based on content type
        region_scores = {
            'limbic': {
                'emotional_keywords': ['emotion', 'feeling', 'fear', 'joy', 'love', 'anger', 'comfort'],
                'base_score': 0.8
            },
            'temporal': {
                'memory_keywords': ['memory', 'remember', 'past', 'experience', 'learn'],
                'base_score': 0.7
            },
            'frontal': {
                'executive_keywords': ['plan', 'decision', 'control', 'execute', 'think'],
                'base_score': 0.6
            },
            'occipital': {
                'visual_keywords': ['visual', 'see', 'image', 'color', 'light', 'sight'],
                'base_score': 0.5
            }
        }
        
        if region in region_scores:
            region_data = region_scores[region]
            base_score = region_data['base_score']
            
            # Check for relevant keywords
            keyword_matches = 0
            for keyword_type, keywords in region_data.items():
                if keyword_type != 'base_score' and isinstance(keywords, list):
                    for keyword in keywords:
                        if keyword in fragment_str:
                            keyword_matches += 1
            
            # Boost score based on keyword matches
            keyword_boost = min(0.2, keyword_matches * 0.05)
            return base_score + keyword_boost
        
        return 0.4  # Default score for unknown regions
    
    def _score_state_compatibility(self, region: str) -> float:
        """Score how compatible the region is with current brain state."""
        current_state = self.consciousness_orchestrator.current_state_name
        
        # State-region compatibility matrix
        compatibility = {
            'dream': {
                'temporal': 0.9,    # Dreams favor memory regions
                'limbic': 0.8,      # Emotional processing
                'frontal': 0.4,     # Less executive activity
                'occipital': 0.6    # Some visual processing
            },
            'aware': {
                'frontal': 0.9,     # High executive activity
                'parietal': 0.8,    # Integration
                'temporal': 0.7,    # Memory access
                'limbic': 0.6       # Some emotional processing
            },
            'liminal': {
                'limbic': 0.9,      # Integration through emotion
                'temporal': 0.7,    # Memory integration
                'frontal': 0.6,     # Some planning
                'brain_stem': 0.5   # Core processing
            }
        }
        
        if current_state in compatibility and region in compatibility[current_state]:
            return compatibility[current_state][region]
        
        return 0.5  # Neutral compatibility
    
    def _get_region_activity_level(self, region_name: str) -> float:
        """Get current activity level of a brain region."""
        # Check if region is in current active regions
        if region_name in self.consciousness_orchestrator.current_state.active_regions if hasattr(self.consciousness_orchestrator.current_state, 'active_regions') else set():
            return 0.8
        
        # Base activity based on region type
        base_activities = {
            'brain_stem': 0.9,   # Always highly active
            'limbic': 0.7,       # Generally active
            'temporal': 0.6,     # Moderate activity
            'frontal': 0.5,      # Variable activity
            'parietal': 0.5,     # Variable activity
            'occipital': 0.4,    # Lower baseline
            'cerebellum': 0.6    # Moderate activity
        }
        
        return base_activities.get(region_name, 0.5)
    
    def _get_field_strength_at_position(self, position: Tuple[int, int, int]) -> float:
        """Get field strength at position."""
        if self.field_dynamics:
            return self.field_dynamics.get_combined_field_value_at_point(position)
        return 0.0
    
    # Bind methods to enhanced network
    import types
    enhanced_network.coordinate_memory_placement = types.MethodType(coordinate_memory_placement, enhanced_network)
    enhanced_network._determine_optimal_region_for_fragment = types.MethodType(_determine_optimal_region_for_fragment, enhanced_network)
    enhanced_network._find_optimal_position_in_region = types.MethodType(_find_optimal_position_in_region, enhanced_network)
    enhanced_network._score_position_for_placement = types.MethodType(_score_position_for_placement, enhanced_network)
    enhanced_network._calculate_placement_score = types.MethodType(_calculate_placement_score, enhanced_network)
    enhanced_network._score_region_appropriateness = types.MethodType(_score_region_appropriateness, enhanced_network)
    enhanced_network._score_state_compatibility = types.MethodType(_score_state_compatibility, enhanced_network)
    enhanced_network._get_region_activity_level = types.MethodType(_get_region_activity_level, enhanced_network)
    enhanced_network._get_field_strength_at_position = types.MethodType(_get_field_strength_at_position, enhanced_network)
    
    logger.info("Memory coordination capabilities added to enhanced mycelial network")


# === Integration Functions ===

def integrate_memory_system_with_enhanced_network(memory_fragment_system, enhanced_network):
    """
    Complete integration of memory fragment system with enhanced mycelial network.
    
    Args:
        memory_fragment_system: Memory fragment system instance
        enhanced_network: Enhanced mycelial network instance
    """
    logger.info("Integrating memory fragment system with enhanced mycelial network")
    
    try:
        # Add memory coordination to enhanced network
        enhance_mycelial_network_memory_coordination(enhanced_network)
        
        # Add enhanced network interface to memory system
        add_enhanced_network_interface_to_memory_system(memory_fragment_system, enhanced_network)
        
        # Store reference in enhanced network
        enhanced_network.memory_fragment_system = memory_fragment_system
        
        # Test integration
        test_result = test_memory_integration(memory_fragment_system, enhanced_network)
        
        logger.info(f"Memory system integration completed. Test result: {test_result['success']}")
        
        return {
            'success': True,
            'integration_test': test_result,
            'memory_system_enhanced': True,
            'enhanced_network_coordinated': True
        }
        
    except Exception as e:
        logger.error(f"Memory system integration failed: {e}")
        return {'success': False, 'error': str(e)}


def test_memory_integration(memory_fragment_system, enhanced_network):
    """Test the memory fragment integration."""
    try:
        # Test 1: Enhanced placement
        test_fragment = "Test emotional memory fragment with joy and happiness"
        
        # This should use enhanced network for placement
        fragment_id = memory_fragment_system.add_fragment(test_fragment)
        
        # Check if fragment was created
        fragment = memory_fragment_system.get_fragment(fragment_id)
        
        if not fragment:
            return {'success': False, 'reason': 'Fragment not created'}
        
        # Test 2: Check placement metadata
        placement_method = fragment.get('meta_tags', {}).get('placement_method')
        
        if placement_method != 'enhanced_network_optimized':
            return {'success': False, 'reason': f'Expected enhanced placement, got {placement_method}'}
        
        # Test 3: Placement statistics
        stats = memory_fragment_system.report_placement_statistics()
        
        if 'total_fragments' not in stats:
            return {'success': False, 'reason': 'Placement statistics not available'}
        
        return {
            'success': True,
            'fragment_created': fragment_id,
            'placement_method': placement_method,
            'placement_stats': stats
        }
        
    except Exception as e:
        logger.error(f"Memory integration test failed: {e}")
        return {'success': False, 'error': str(e)}