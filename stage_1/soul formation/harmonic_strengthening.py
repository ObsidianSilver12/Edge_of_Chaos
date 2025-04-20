"""
Harmonic Strengthening

This module implements the harmonic strengthening process for souls,
enhancing soul stability, coherence and resonance through harmonic frequencies
and phi-resonance amplification.

The harmonic strengthening process occurs after the initial soul formation
through the Sephiroth and prepares the soul for life cord formation.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='harmonic_strengthening.log'
)
logger = logging.getLogger('harmonic_strengthening')

# Add parent directory to path to import from parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
try:
    from void.soul_spark import SoulSpark
    import metrics_tracking as metrics
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# Constants
GOLDEN_RATIO = 1.618033988749895
PHI_HARMONICS = [GOLDEN_RATIO ** n for n in range(1, 8)]
BASE_FREQUENCIES = [432.0, 528.0, 639.0, 741.0, 852.0, 963.0]


class HarmonicStrengthening:
    """
    Harmonic Strengthening Process
    
    Enhances soul stability, coherence and resonance through harmonic frequencies
    and phi-resonance amplification. This prepares the soul for life cord formation
    and physical incarnation.
    """
    
    def __init__(self, soul_spark):
        """
        Initialize the harmonic strengthening process.
        
        Args:
            soul_spark: The soul spark to be strengthened
        """
        self.soul_spark = soul_spark
        
        # Process state tracking
        self.process_complete = False
        self.current_phase = None
        self.phase_results = {}
        
        # Harmonic metrics
        self.metrics = {
            "initial_stability": getattr(soul_spark, "stability", 0.0),
            "initial_coherence": getattr(soul_spark, "coherence", 0.0),
            "initial_frequency": getattr(soul_spark, "frequency", 0.0),
            "phase_metrics": {},
            "final_stability": 0.0,
            "final_coherence": 0.0,
            "improvement_percentage": 0.0,
            "process_duration": 0.0
        }
        
        # Ensure soul has necessary properties
        self._ensure_soul_properties()
        
        logger.info(f"Harmonic Strengthening initialized for soul spark {soul_spark.id}")
    
    def _ensure_soul_properties(self):
        """Ensure soul has all necessary properties for strengthening."""
        # Check if soul has completed formation
        if not getattr(self.soul_spark, "formation_complete", False):
            logger.warning("Soul has not completed formation process")
            setattr(self.soul_spark, "formation_complete", True)
        
        # Ensure frequency is set
        if not hasattr(self.soul_spark, "frequency"):
            logger.warning("Soul frequency not set, initializing with default")
            setattr(self.soul_spark, "frequency", 528.0)
            
        # Ensure stability is set
        if not hasattr(self.soul_spark, "stability"):
            logger.warning("Soul stability not set, initializing with default")
            setattr(self.soul_spark, "stability", 0.6)
            
        # Ensure coherence is set
        if not hasattr(self.soul_spark, "coherence"):
            logger.warning("Soul coherence not set, initializing with default")
            setattr(self.soul_spark, "coherence", 0.6)
            
        # Initialize harmonics if not present
        if not hasattr(self.soul_spark, "harmonics"):
            logger.info("Initializing soul harmonics")
            base_freq = getattr(self.soul_spark, "frequency", 528.0)
            harmonics = [base_freq * n for n in range(1, 6)]
            setattr(self.soul_spark, "harmonics", harmonics)
    
    def strengthen(self, intensity=0.7, duration=1.0):
        """
        Perform the complete harmonic strengthening process.
        
        Args:
            intensity (float): Intensity of the strengthening (0.1-1.0)
            duration (float): Relative duration of the process (0.1-2.0)
            
        Returns:
            bool: Success status
        """
        logger.info(f"Beginning strengthening process with intensity={intensity}, duration={duration}")
        start_time = time.time()
        
        # Record initial metrics
        self._record_initial_metrics()
        
        # Phase 1: Frequency Tuning
        self.current_phase = "frequency_tuning"
        success = self._perform_frequency_tuning(intensity)
        if not success:
            logger.error("Frequency tuning phase failed")
            return False
            
        # Phase 2: Phi Resonance Amplification
        self.current_phase = "phi_resonance"
        success = self._perform_phi_resonance(intensity, duration)
        if not success:
            logger.error("Phi resonance phase failed")
            return False
            
        # Phase 3: Harmonic Pattern Stabilization
        self.current_phase = "pattern_stabilization"
        success = self._perform_pattern_stabilization(intensity)
        if not success:
            logger.error("Pattern stabilization phase failed")
            return False
            
        # Phase 4: Coherence Enhancement
        self.current_phase = "coherence_enhancement"
        success = self._perform_coherence_enhancement(intensity, duration)
        if not success:
            logger.error("Coherence enhancement phase failed")
            return False
            
        # Phase 5: Resonance Field Expansion
        self.current_phase = "field_expansion"
        success = self._perform_field_expansion(intensity)
        if not success:
            logger.error("Field expansion phase failed")
            return False
        
        # Complete process
        self.process_complete = True
        process_duration = time.time() - start_time
        self.metrics["process_duration"] = process_duration
        
        # Record final metrics
        self._record_final_metrics()
        
        logger.info(f"Strengthening process completed in {process_duration:.2f} seconds")
        return True
    
    def _perform_frequency_tuning(self, intensity):
        """
        Tune the soul's frequency to optimal harmonic ratios.
        
        Args:
            intensity (float): Intensity of the tuning
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning frequency tuning phase")
        
        # Get current frequency
        current_freq = getattr(self.soul_spark, "frequency", 528.0)
        
        # Find closest harmonic base frequency
        closest_base = min(BASE_FREQUENCIES, key=lambda f: abs(f - current_freq))
        
        # Calculate tuning amount based on intensity
        freq_diff = closest_base - current_freq
        tuning_amount = freq_diff * intensity
        
        # Apply tuning
        new_freq = current_freq + tuning_amount
        setattr(self.soul_spark, "frequency", new_freq)
        
        # Update harmonics
        harmonics = [new_freq * n for n in range(1, 6)]
        setattr(self.soul_spark, "harmonics", harmonics)
        
        # Store phase results
        self.phase_results["frequency_tuning"] = {
            "original_frequency": current_freq,
            "target_frequency": closest_base,
            "new_frequency": new_freq,
            "harmonics": harmonics
        }
        
        # Record phase metrics
        self.metrics["phase_metrics"]["frequency_tuning"] = {
            "improvement": abs(tuning_amount) / current_freq,
            "target_reached": abs(new_freq - closest_base) < 1.0
        }
        
        logger.info(f"Frequency tuned from {current_freq:.2f} to {new_freq:.2f} Hz")
        return True
    
    def _perform_phi_resonance(self, intensity, duration):
        """
        Amplify the soul's phi resonance patterns.
        
        Args:
            intensity (float): Intensity of the amplification
            duration (float): Relative duration of the phase
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning phi resonance amplification phase")
        
        # Get current properties
        frequency = getattr(self.soul_spark, "frequency", 528.0)
        phi_resonance = getattr(self.soul_spark, "phi_resonance", 0.5)
        
        # Calculate phi harmonic series
        phi_harmonics = []
        for i in range(1, 8):
            # Apply phi ratio in both directions for complete series
            phi_up = frequency * (GOLDEN_RATIO ** i)
            phi_down = frequency / (GOLDEN_RATIO ** i)
            phi_harmonics.extend([phi_up, phi_down])
        
        # Sort harmonics
        phi_harmonics.sort()
        
        # Increase phi resonance based on intensity and duration
        phi_increase = 0.2 * intensity * duration
        new_phi_resonance = min(1.0, phi_resonance + phi_increase)
        
        # Apply increased phi resonance
        setattr(self.soul_spark, "phi_resonance", new_phi_resonance)
        setattr(self.soul_spark, "phi_harmonics", phi_harmonics)
        
        # Increase stability based on phi resonance improvement
        stability_increase = phi_increase * 0.3
        current_stability = getattr(self.soul_spark, "stability", 0.6)
        new_stability = min(1.0, current_stability + stability_increase)
        setattr(self.soul_spark, "stability", new_stability)
        
        # Store phase results
        self.phase_results["phi_resonance"] = {
            "original_phi_resonance": phi_resonance,
            "new_phi_resonance": new_phi_resonance,
            "phi_harmonics": phi_harmonics[:5],  # Store first 5 harmonics
            "stability_increase": stability_increase
        }
        
        # Record phase metrics
        self.metrics["phase_metrics"]["phi_resonance"] = {
            "improvement": (new_phi_resonance - phi_resonance) / max(0.1, phi_resonance),
            "stability_gain": stability_increase
        }
        
        logger.info(f"Phi resonance amplified from {phi_resonance:.2f} to {new_phi_resonance:.2f}")
        return True
    
    def _perform_pattern_stabilization(self, intensity):
        """
        Stabilize the soul's harmonic patterns.
        
        Args:
            intensity (float): Intensity of the stabilization
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning pattern stabilization phase")
        
        # Get current properties
        stability = getattr(self.soul_spark, "stability", 0.6)
        pattern_stability = getattr(self.soul_spark, "pattern_stability", 0.5)
        aspects = getattr(self.soul_spark, "aspects", {})
        
        # Calculate aspect influence on pattern stability
        aspect_influence = min(0.3, len(aspects) * 0.01)
        
        # Calculate stabilization amount
        base_increase = 0.15 * intensity
        aspect_bonus = aspect_influence * intensity
        total_increase = base_increase + aspect_bonus
        
        # Apply pattern stabilization
        new_pattern_stability = min(1.0, pattern_stability + total_increase)
        setattr(self.soul_spark, "pattern_stability", new_pattern_stability)
        
        # Apply general stability increase
        stability_increase = total_increase * 0.7
        new_stability = min(1.0, stability + stability_increase)
        setattr(self.soul_spark, "stability", new_stability)
        
        # Store phase results
        self.phase_results["pattern_stabilization"] = {
            "original_pattern_stability": pattern_stability,
            "new_pattern_stability": new_pattern_stability,
            "stability_increase": stability_increase,
                        "aspect_influence": aspect_influence
        }
        
        # Record phase metrics
        self.metrics["phase_metrics"]["pattern_stabilization"] = {
            "improvement": (new_pattern_stability - pattern_stability) / max(0.1, pattern_stability),
            "stability_gain": stability_increase
        }
        
        logger.info(f"Pattern stability increased from {pattern_stability:.2f} to {new_pattern_stability:.2f}")
        return True
    
    def _perform_coherence_enhancement(self, intensity, duration):
        """
        Enhance the soul's coherence through harmonic resonance.
        
        Args:
            intensity (float): Intensity of the enhancement
            duration (float): Relative duration of the phase
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning coherence enhancement phase")
        
        # Get current properties
        coherence = getattr(self.soul_spark, "coherence", 0.6)
        harmonics = getattr(self.soul_spark, "harmonics", [])
        
        # Calculate harmonic coherence factor
        harmonic_factor = min(1.0, len(harmonics) * 0.1)
        
        # Calculate coherence increase based on intensity, duration and harmonics
        base_increase = 0.15 * intensity * duration
        harmonic_bonus = harmonic_factor * intensity * 0.1
        total_increase = base_increase + harmonic_bonus
        
        # Apply coherence enhancement
        new_coherence = min(1.0, coherence + total_increase)
        setattr(self.soul_spark, "coherence", new_coherence)
        
        # Update harmony rating
        harmony = getattr(self.soul_spark, "harmony", 0.5)
        harmony_increase = total_increase * 0.8
        new_harmony = min(1.0, harmony + harmony_increase)
        setattr(self.soul_spark, "harmony", new_harmony)
        
        # Store phase results
        self.phase_results["coherence_enhancement"] = {
            "original_coherence": coherence,
            "new_coherence": new_coherence,
            "harmony_increase": harmony_increase,
            "harmonic_factor": harmonic_factor
        }
        
        # Record phase metrics
        self.metrics["phase_metrics"]["coherence_enhancement"] = {
            "improvement": (new_coherence - coherence) / max(0.1, coherence),
            "harmony_gain": harmony_increase
        }
        
        logger.info(f"Coherence enhanced from {coherence:.2f} to {new_coherence:.2f}")
        return True
    
    def _perform_field_expansion(self, intensity):
        """
        Expand the soul's resonance field.
        
        Args:
            intensity (float): Intensity of the expansion
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning resonance field expansion phase")
        
        # Get current properties
        field_radius = getattr(self.soul_spark, "field_radius", 3.0)
        field_strength = getattr(self.soul_spark, "field_strength", 0.5)
        
        # Calculate expansion and strengthening
        radius_increase = 2.0 * intensity
        strength_increase = 0.2 * intensity
        
        # Apply expansion
        new_radius = field_radius + radius_increase
        new_strength = min(1.0, field_strength + strength_increase)
        
        setattr(self.soul_spark, "field_radius", new_radius)
        setattr(self.soul_spark, "field_strength", new_strength)
        
        # Store phase results
        self.phase_results["field_expansion"] = {
            "original_radius": field_radius,
            "new_radius": new_radius,
            "original_strength": field_strength,
            "new_strength": new_strength
        }
        
        # Record phase metrics
        self.metrics["phase_metrics"]["field_expansion"] = {
            "radius_improvement": radius_increase / field_radius,
            "strength_improvement": strength_increase / max(0.1, field_strength)
        }
        
        logger.info(f"Field expanded from radius {field_radius:.2f} to {new_radius:.2f}")
        return True
    
    def _record_initial_metrics(self):
        """Record initial metrics before strengthening process."""
        # Record soul spark properties
        metrics.record_metric(
            "harmonic_strengthening", 
            "initial_frequency", 
            getattr(self.soul_spark, "frequency", 0.0)
        )
        
        metrics.record_metric(
            "harmonic_strengthening",
            "initial_stability",
            getattr(self.soul_spark, "stability", 0.0)
        )
        
        metrics.record_metric(
            "harmonic_strengthening",
            "initial_coherence",
            getattr(self.soul_spark, "coherence", 0.0)
        )
        
        # Record phi resonance if present
        if hasattr(self.soul_spark, "phi_resonance"):
            metrics.record_metric(
                "harmonic_strengthening",
                "initial_phi_resonance",
                self.soul_spark.phi_resonance
            )
    
    def _record_final_metrics(self):
        """Record final metrics after strengthening process."""
        # Update final metrics in our metrics dict
        self.metrics["final_stability"] = getattr(self.soul_spark, "stability", 0.0)
        self.metrics["final_coherence"] = getattr(self.soul_spark, "coherence", 0.0)
        
        # Calculate improvement percentage
        initial_combined = (self.metrics["initial_stability"] + self.metrics["initial_coherence"]) / 2
        final_combined = (self.metrics["final_stability"] + self.metrics["final_coherence"]) / 2
        
        if initial_combined > 0:
            improvement = (final_combined - initial_combined) / initial_combined * 100
            self.metrics["improvement_percentage"] = improvement
        
        # Record metrics to central system
        metrics.record_metric(
            "harmonic_strengthening", 
            "final_frequency", 
            getattr(self.soul_spark, "frequency", 0.0)
        )
        
        metrics.record_metric(
            "harmonic_strengthening",
            "final_stability",
            getattr(self.soul_spark, "stability", 0.0)
        )
        
        metrics.record_metric(
            "harmonic_strengthening",
            "final_coherence",
            getattr(self.soul_spark, "coherence", 0.0)
        )
        
        metrics.record_metric(
            "harmonic_strengthening",
            "improvement_percentage",
            self.metrics["improvement_percentage"]
        )
        
        metrics.record_metric(
            "harmonic_strengthening",
            "process_duration",
            self.metrics["process_duration"]
        )
        
        # Record phi resonance if present
        if hasattr(self.soul_spark, "phi_resonance"):
            metrics.record_metric(
                "harmonic_strengthening",
                "final_phi_resonance",
                self.soul_spark.phi_resonance
            )
        
        logger.info("Final metrics recorded for harmonic strengthening")


# Example usage
if __name__ == "__main__":
    # Create a soul spark
    from void.soul_spark import SoulSpark
    
    soul = SoulSpark()
    
    # Set some initial properties
    soul.frequency = 528.0
    soul.stability = 0.6
    soul.coherence = 0.6
    soul.formation_complete = True
    
    # Initialize harmonic strengthening
    strengthening = HarmonicStrengthening(soul)
    
    # Perform strengthening
    strengthening.strengthen(intensity=0.8, duration=1.2)
    
    # Print results
    print(f"Soul ID: {soul.id}")
    print(f"Frequency: {soul.frequency:.2f} Hz")
    print(f"Stability: {soul.stability:.2f}")
    print(f"Coherence: {soul.coherence:.2f}")
    print(f"Phi Resonance: {getattr(soul, 'phi_resonance', 0.0):.2f}")
    print(f"Field Radius: {getattr(soul, 'field_radius', 0.0):.2f}")
    print(f"Improvement: {strengthening.metrics['improvement_percentage']:.2f}%")
