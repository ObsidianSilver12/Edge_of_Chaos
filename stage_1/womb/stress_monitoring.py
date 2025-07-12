# --- stress_monitoring.py V8 CORRECTED ---
"""
Event-driven stress monitoring system triggered only at specific brain formation milestones.
Monitors stress at 3 critical points and triggers progressive responses: mother resonance → womb healing → miscarriage.
Only uses functions that actually exist in integrated files.
"""

import logging
import uuid
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import math

# Import constants
from shared.constants.constants import *

# Import mother resonance system (only use functions that exist)
from mother_resonance import generate_mother_resonance_profile, create_mother_resonance_data

# --- Logging Setup ---
logger = logging.getLogger("Conception")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class StressMonitoring:
    """
    Event-driven stress monitoring system for critical brain formation milestones.
    Triggers only at specific flag events, not continuous monitoring.
    """
    
    def __init__(self):
        """Initialize the stress monitoring system."""
        self.logger = logging.getLogger("Conception.StressMonitoring")
        self.logger.info("Initializing event-driven stress monitoring system")
        
        # Development stage thresholds (adjusted for pregnancy-like sensitivity)
        self.stage_thresholds = {
            'early': {      # After brain structure (like first trimester - more sensitive)
                'mother_resonance': 0.5,
                'womb_healing': 0.7,
                'miscarriage': 0.8
            },
            'mid': {        # After neural synapses (mid development)
                'mother_resonance': 0.6,
                'womb_healing': 0.8,
                'miscarriage': 0.9
            },
            'late': {       # After mycelial entanglement (more resilient)
                'mother_resonance': 0.7,
                'womb_healing': 0.85,
                'miscarriage': 0.95
            }
        }
        
        # Current state
        self.monitoring_enabled = True
        self.miscarriage_risk_active = True  # Disabled during birth
        self.healing_attempts = 0
        self.max_healing_attempts = 3
        
        # Integration references (only store what's passed)
        self.mycelial_network = None
        
        # Metrics
        self.stress_events = []
        self.interventions_applied = []

    def monitor_stress_at_brain_structure_complete(self, brain_structure=None, energy_storage=None):
        """
        Monitor stress after BRAIN_STRUCTURE_CREATED flag - EARLY stage (most sensitive).
        This is the first critical monitoring point.
        """
        try:
            self.logger.info("=== STRESS CHECK: Brain Structure Complete (Early Stage) ===")
            
            if not self.monitoring_enabled:
                return
            
            # Calculate stress based on available data only
            stress_level = self._calculate_early_stage_stress(brain_structure, energy_storage)
            
            self.logger.info(f"Early stage stress level: {stress_level:.3f}")
            
            # Record stress event
            stress_event = {
                'stage': 'early',
                'trigger': 'BRAIN_STRUCTURE_CREATED',
                'stress_level': stress_level,
                'timestamp': datetime.now().isoformat()
            }
            self.stress_events.append(stress_event)
            
            # Apply interventions based on early stage thresholds
            self._apply_interventions_for_stage('early', stress_level)
            
        except Exception as e:
            self.logger.error(f"Error in early stage stress monitoring: {e}")
            # If monitoring fails, assume moderate stress and trigger mother resonance
            self._apply_mother_resonance('early', 0.6, "monitoring_failure")

    def monitor_stress_at_synapses_complete(self, neural_network=None, brain_structure=None, energy_storage=None):
        """
        Monitor stress after FLAG_SUB_REGION_SYNAPSES_ADDED - MID stage.
        This is the second critical monitoring point.
        """
        try:
            self.logger.info("=== STRESS CHECK: Synapses Complete (Mid Stage) ===")
            
            if not self.monitoring_enabled:
                return
            
            # Calculate stress based on neural network complexity
            stress_level = self._calculate_mid_stage_stress(neural_network, brain_structure, energy_storage)
            
            self.logger.info(f"Mid stage stress level: {stress_level:.3f}")
            
            # Record stress event
            stress_event = {
                'stage': 'mid',
                'trigger': 'FLAG_SUB_REGION_SYNAPSES_ADDED',
                'stress_level': stress_level,
                'timestamp': datetime.now().isoformat()
            }
            self.stress_events.append(stress_event)
            
            # Apply interventions based on mid stage thresholds
            self._apply_interventions_for_stage('mid', stress_level)
            
        except Exception as e:
            self.logger.error(f"Error in mid stage stress monitoring: {e}")
            self._apply_mother_resonance('mid', 0.7, "monitoring_failure")

    def monitor_stress_at_seeds_entangled(self, mycelial_network=None, neural_network=None, 
                                         brain_structure=None, energy_storage=None):
        """
        Monitor stress after FLAG_SEEDS_ENTANGLED - LATE stage (most resilient).
        This is the third and final critical monitoring point.
        """
        try:
            self.logger.info("=== STRESS CHECK: Seeds Entangled (Late Stage) ===")
            
            if not self.monitoring_enabled:
                return
            
            # Store mycelial network reference for potential miscarriage
            self.mycelial_network = mycelial_network
            
            # Calculate stress based on full system complexity
            stress_level = self._calculate_late_stage_stress(mycelial_network, neural_network, 
                                                           brain_structure, energy_storage)
            
            self.logger.info(f"Late stage stress level: {stress_level:.3f}")
            
            # Record stress event
            stress_event = {
                'stage': 'late',
                'trigger': 'FLAG_SEEDS_ENTANGLED',
                'stress_level': stress_level,
                'timestamp': datetime.now().isoformat()
            }
            self.stress_events.append(stress_event)
            
            # Apply interventions based on late stage thresholds
            self._apply_interventions_for_stage('late', stress_level)
            
        except Exception as e:
            self.logger.error(f"Error in late stage stress monitoring: {e}")
            self._apply_mother_resonance('late', 0.8, "monitoring_failure")

    def _calculate_early_stage_stress(self, brain_structure, energy_storage) -> float:
        """Calculate stress level for early brain formation stage using ONLY existing attributes."""
        try:
            stress_factors = []
            
            # Check energy levels - ONLY use confirmed existing attributes
            if energy_storage and hasattr(energy_storage, 'energy_storage'):
                energy_dict = energy_storage.energy_storage
                current_energy = energy_dict.get('energy_available_seu', 0)
                capacity = energy_dict.get('capacity_seu', 100.0)
                energy_ratio = current_energy / capacity if capacity > 0 else 0.5
                
                if energy_ratio < 0.3:
                    stress_factors.append(0.7)  # High stress from low energy
                elif energy_ratio < 0.5:
                    stress_factors.append(0.4)  # Moderate stress
                else:
                    stress_factors.append(0.1)  # Low stress
            else:
                stress_factors.append(0.2)  # Unknown energy = mild stress
            
            # Check brain structure - ONLY use confirmed existing attributes
            if brain_structure and hasattr(brain_structure, 'field_integrity'):
                field_integrity_dict = brain_structure.field_integrity
                test_passed = field_integrity_dict.get('test_passed', False)
                
                if not test_passed:
                    stress_factors.append(0.6)  # High stress from field issues
                else:
                    stress_factors.append(0.05)  # Low stress
            else:
                stress_factors.append(0.15)  # Unknown field = mild stress
            
            # Add random biological variance (5-15%)
            variance = random.uniform(0.05, 0.15)
            stress_factors.append(variance)
            
            # Calculate average stress
            avg_stress = sum(stress_factors) / len(stress_factors) if stress_factors else 0.3
            
            return min(1.0, max(0.0, avg_stress))
            
        except Exception as e:
            self.logger.debug(f"Error calculating early stage stress: {e}")
            return 0.4  # Moderate stress fallback

    def _calculate_mid_stage_stress(self, neural_network, brain_structure, energy_storage) -> float:
        """Calculate stress level for mid brain formation stage using ONLY existing attributes."""
        try:
            stress_factors = []
            
            # Check neural network - ONLY use confirmed existing attributes
            if neural_network and hasattr(neural_network, 'nodes'):
                nodes_dict = neural_network.nodes
                synapses_dict = getattr(neural_network, 'synapses', {})
                
                node_count = len(nodes_dict)
                synapse_count = len(synapses_dict)
                
                # Count active nodes
                active_nodes = sum(1 for node in nodes_dict.values() 
                                 if node.get('status') == 'active')
                
                if node_count > 0:
                    activity_ratio = active_nodes / node_count
                    if activity_ratio > 0.8:
                        stress_factors.append(0.6)  # High activity stress
                    elif activity_ratio > 0.5:
                        stress_factors.append(0.3)  # Moderate activity
                    else:
                        stress_factors.append(0.1)  # Normal activity
                
                # Stress from synapse complexity
                if synapse_count > 5000:  # Arbitrary complexity threshold
                    stress_factors.append(0.4)
                elif synapse_count > 2000:
                    stress_factors.append(0.2)
                else:
                    stress_factors.append(0.05)
            else:
                stress_factors.append(0.25)  # Unknown neural state
            
            # Include early stage factors with lower weight
            early_stress = self._calculate_early_stage_stress(brain_structure, energy_storage)
            stress_factors.append(early_stress * 0.6)  # Reduced weight for early factors
            
            # Calculate weighted average
            avg_stress = sum(stress_factors) / len(stress_factors) if stress_factors else 0.4
            
            return min(1.0, max(0.0, avg_stress))
            
        except Exception as e:
            self.logger.debug(f"Error calculating mid stage stress: {e}")
            return 0.5  # Moderate stress fallback

    def _calculate_late_stage_stress(self, mycelial_network, neural_network, brain_structure, energy_storage) -> float:
        """Calculate stress level for late brain formation stage using ONLY existing attributes."""
        try:
            stress_factors = []
            
            # Check mycelial network - ONLY use confirmed existing attributes
            if mycelial_network and hasattr(mycelial_network, 'metrics'):
                metrics_dict = mycelial_network.metrics
                errors = metrics_dict.get('errors', [])
                error_count = len(errors)
                
                if error_count > 3:
                    stress_factors.append(0.7)  # High stress from mycelial errors
                elif error_count > 1:
                    stress_factors.append(0.4)  # Moderate stress
                else:
                    stress_factors.append(0.1)  # Low stress
                
                # Check entanglement success using confirmed attributes
                if hasattr(mycelial_network, 'active_seeds'):
                    active_seeds_list = mycelial_network.active_seeds
                    active_seed_count = len(active_seeds_list)
                    
                    if active_seed_count < 5:
                        stress_factors.append(0.5)  # Stress from low activity
                    else:
                        stress_factors.append(0.1)  # Good activity
            else:
                stress_factors.append(0.3)  # Unknown mycelial state
            
            # Include mid stage factors with reduced weight
            mid_stress = self._calculate_mid_stage_stress(neural_network, brain_structure, energy_storage)
            stress_factors.append(mid_stress * 0.5)  # Further reduced weight
            
            # Calculate weighted average
            avg_stress = sum(stress_factors) / len(stress_factors) if stress_factors else 0.3
            
            return min(1.0, max(0.0, avg_stress))
            
        except Exception as e:
            self.logger.debug(f"Error calculating late stage stress: {e}")
            return 0.4  # Moderate stress fallback

    def _apply_interventions_for_stage(self, stage: str, stress_level: float):
        """Apply appropriate interventions based on stage thresholds."""
        try:
            thresholds = self.stage_thresholds[stage]
            
            if stress_level >= thresholds['miscarriage'] and self.miscarriage_risk_active:
                self.logger.error(f"CRITICAL STRESS in {stage} stage: {stress_level:.3f} - TRIGGERING MISCARRIAGE")
                self._trigger_miscarriage(stage, stress_level)
                
            elif stress_level >= thresholds['womb_healing']:
                self.logger.warning(f"High stress in {stage} stage: {stress_level:.3f} - applying womb healing")
                self._apply_womb_healing(stage, stress_level)
                
            elif stress_level >= thresholds['mother_resonance']:
                self.logger.info(f"Elevated stress in {stage} stage: {stress_level:.3f} - applying mother resonance")
                self._apply_mother_resonance(stage, stress_level, "elevated_stress")
                
            else:
                self.logger.info(f"Normal stress in {stage} stage: {stress_level:.3f} - no intervention needed")
            
        except Exception as e:
            self.logger.error(f"Error applying interventions for {stage} stage: {e}")

    def _apply_mother_resonance(self, stage: str, stress_level: float, reason: str):
        """Apply mother resonance using only existing functions."""
        try:
            self.logger.info(f"Applying mother resonance for {stage} stage (reason: {reason})")
            
            # Use actual mother resonance functions that exist
            mother_profile = generate_mother_resonance_profile()
            mother_data = create_mother_resonance_data()
            
            # Simple stress reduction based on mother's love resonance
            love_resonance = mother_profile.love_resonance
            stress_reduction = love_resonance * 0.3  # Max 30% reduction
            
            # Record intervention
            intervention = {
                'type': 'mother_resonance',
                'stage': stage,
                'reason': reason,
                'pre_stress': stress_level,
                'stress_reduction': stress_reduction,
                'mother_love_resonance': love_resonance,
                'timestamp': datetime.now().isoformat()
            }
            self.interventions_applied.append(intervention)
            
            self.logger.info(f"Mother resonance applied: {stress_reduction:.3f} stress reduction "
                           f"(love resonance: {love_resonance:.3f})")
            
        except Exception as e:
            self.logger.error(f"Failed to apply mother resonance: {e}")

    def _apply_womb_healing(self, stage: str, stress_level: float):
        """Apply womb healing as second-level intervention."""
        try:
            self.logger.warning(f"Applying womb healing for {stage} stage")
            
            self.healing_attempts += 1
            
            # Check healing limits
            if self.healing_attempts > self.max_healing_attempts:
                self.logger.error(f"Maximum healing attempts ({self.max_healing_attempts}) exceeded")
                self._trigger_miscarriage(stage, stress_level, "healing_failed")
                return
            
            # Simple healing effectiveness based on stage
            stage_effectiveness = {
                'early': 0.5,   # Less effective in early sensitive stage
                'mid': 0.7,     # More effective in mid stage
                'late': 0.8     # Most effective in late resilient stage
            }
            
            healing_effectiveness = stage_effectiveness.get(stage, 0.6)
            stress_reduction = healing_effectiveness * 0.4  # Max 40% reduction
            
            # Record intervention
            intervention = {
                'type': 'womb_healing',
                'stage': stage,
                'attempt': self.healing_attempts,
                'pre_stress': stress_level,
                'stress_reduction': stress_reduction,
                'effectiveness': healing_effectiveness,
                'timestamp': datetime.now().isoformat()
            }
            self.interventions_applied.append(intervention)
            
            # Check if healing was sufficient
            post_stress = stress_level - stress_reduction
            if post_stress <= self.stage_thresholds[stage]['mother_resonance']:
                self.logger.info(f"Womb healing successful: stress reduced to {post_stress:.3f}")
                # Trigger healing sleep cycle flag
                setattr(self, FLAG_STRESS_RELIEVED, True)
            else:
                self.logger.warning(f"Womb healing insufficient: stress still {post_stress:.3f}")
                # If still above miscarriage threshold, trigger miscarriage
                if post_stress >= self.stage_thresholds[stage]['miscarriage']:
                    self._trigger_miscarriage(stage, post_stress, "healing_insufficient")
            
        except Exception as e:
            self.logger.error(f"Failed to apply womb healing: {e}")
            # If healing fails, trigger miscarriage
            self._trigger_miscarriage(stage, stress_level, "healing_error")

    def _trigger_miscarriage(self, stage: str, stress_level: float, reason: str = "critical_stress"):
        """Trigger miscarriage using mycelial network soul return process."""
        try:
            self.logger.error(f"TRIGGERING MISCARRIAGE: {stage} stage, stress={stress_level:.3f}, reason={reason}")
            
            # Record miscarriage event
            miscarriage_event = {
                'type': 'miscarriage',
                'stage': stage,
                'stress_level': stress_level,
                'reason': reason,
                'healing_attempts': self.healing_attempts,
                'timestamp': datetime.now().isoformat()
            }
            self.interventions_applied.append(miscarriage_event)
            
            # Set miscarriage flag
            setattr(self, FLAG_MISCARRY, True)
            
            # Call mycelial network to return soul to liminal space
            if self.mycelial_network and hasattr(self.mycelial_network, 'activate_liminal_state_miscarriage'):
                self.logger.error("Returning soul to liminal space via mycelial network...")
                soul_package = self.mycelial_network.activate_liminal_state_miscarriage()
                self.logger.error(f"Soul returned with {soul_package.get('extracted_energy', 0):.2f} SEU")
            else:
                self.logger.error("CRITICAL: Mycelial network unavailable for soul return")
            
            # Set simulation termination flag
            setattr(self, FLAG_SIMULATION_TERMINATED, True)
            
            self.logger.error("MISCARRIAGE PROCESS COMPLETE - SIMULATION TERMINATED")
            
            # Hard fail - raise exception to terminate
            raise RuntimeError(f"MISCARRIAGE: {stage} stage stress {stress_level:.3f} - {reason}")
            
        except Exception as e:
            if "MISCARRIAGE:" in str(e):
                raise  # Re-raise miscarriage exception
            else:
                self.logger.error(f"Error during miscarriage process: {e}")
                setattr(self, FLAG_SIMULATION_TERMINATED, True)
                raise RuntimeError(f"MISCARRIAGE PROCESS FAILED: {e}")

    def disable_miscarriage_for_birth(self):
        """Disable miscarriage risk during birth phase (mother resonance only)."""
        self.miscarriage_risk_active = False
        self.logger.info("Miscarriage risk disabled for birth phase - mother resonance still available")

    def get_stress_monitoring_report(self) -> Dict[str, Any]:
        """Generate final stress monitoring report."""
        try:
            return {
                'stress_events': self.stress_events,
                'interventions_applied': self.interventions_applied,
                'total_healing_attempts': self.healing_attempts,
                'miscarriage_occurred': any(i.get('type') == 'miscarriage' for i in self.interventions_applied),
                'final_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating stress report: {e}")
            return {'error': str(e)}

# ===== END OF CORRECTED STRESS_MONITORING.PY =====

# Simple test function
def test_stress_monitoring():
    """Test the corrected event-driven stress monitoring."""
    try:
        monitor = StressMonitoring()
        
        # Test early stage
        monitor.monitor_stress_at_brain_structure_complete()
        
        # Test mid stage  
        monitor.monitor_stress_at_synapses_complete()
        
        # Test late stage
        monitor.monitor_stress_at_seeds_entangled()
        
        # Get report
        report = monitor.get_stress_monitoring_report()
        print(f"Stress monitoring test completed: {len(report['stress_events'])} events")
        
        return True
        
    except Exception as e:
        print(f"Stress monitoring test failed: {e}")
        return False

if __name__ == "__main__":
    test_stress_monitoring()

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        