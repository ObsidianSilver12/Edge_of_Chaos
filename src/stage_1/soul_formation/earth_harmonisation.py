"""
Earth Harmonization

This module implements the process of harmonizing the soul with Earth's frequencies
and energetic patterns, preparing it for physical incarnation. The process attunes
the soul to Earth's natural rhythms, elements, and planetary vibrations.

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
    filename='earth_harmonisation.log'
)
logger = logging.getLogger('earth_harmonisation')

# Add parent directory to path to import from parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
try:
    from void.soul_spark import SoulSpark
    from sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    import metrics_tracking as metrics
    from constants import EARTH_FREQUENCIES
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    # Define fallback constants
    EARTH_FREQUENCIES = {
        "schumann": 7.83,    # Schumann resonance first harmonic
        "geomagnetic": 11.3, # Earth's geomagnetic field
        "diurnal": 1.0 / 24.0, # Daily cycle (Hz - cycles per hour)
        "lunar": 1.0 / (24.0 * 29.5), # Lunar cycle (Hz)
        "annual": 1.0 / (24.0 * 365.25) # Annual cycle (Hz)
    }

# Constants
EARTH_ELEMENTS = ["earth", "water", "air", "fire", "ether"]


class EarthHarmonization:
    """
    Earth Harmonization Process
    
    This class implements the process of harmonizing a soul with Earth's frequencies
    and energetic patterns, preparing it for physical incarnation.
    """
    
    def __init__(self, soul_spark):
        """
        Initialize the Earth harmonization process.
        
        Args:
            soul_spark: The soul spark to harmonize with Earth
        """
        self.soul_spark = soul_spark
        
        # Process state tracking
        self.harmonization_complete = False
        self.current_phase = None
        self.phase_results = {}
        
        # Harmonization metrics
        self.metrics = {
            "initial_resonance": 0.0,
            "phase_metrics": {},
            "final_resonance": 0.0,
            "process_duration": 0.0
        }
        
        # Earth attunements
        self.attunements = {
            "frequency": 0.0,
            "elemental": {},
            "cycles": {},
            "planetary": 0.0,
            "gaia_connection": 0.0
        }
        
        logger.info(f"Earth Harmonization initialized for soul spark {getattr(soul_spark, 'id', 'unknown')}")
    
    def harmonize(self, intensity=0.7, duration=1.0):
        """
        Perform the complete Earth harmonization process.
        
        Args:
            intensity (float): Intensity of harmonization (0.1-1.0)
            duration (float): Relative duration of the process (0.1-2.0)
            
        Returns:
            bool: Success status
        """
        logger.info(f"Beginning Earth harmonization with intensity={intensity}, duration={duration}")
        start_time = time.time()
        
        # Check prerequisites
        if not self._check_prerequisites():
            logger.error("Prerequisites not met for Earth harmonization")
            return False
        
        # Calculate initial Earth resonance
        initial_resonance = self._calculate_earth_resonance()
        self.metrics["initial_resonance"] = initial_resonance
        
        logger.info(f"Initial Earth resonance: {initial_resonance:.2f}")
        
        # Phase 1: Frequency Attunement
        self.current_phase = "frequency_attunement"
        success = self._perform_frequency_attunement(intensity, duration)
        if not success:
            logger.error("Frequency attunement phase failed")
            return False
            
        # Phase 2: Elemental Alignment
        self.current_phase = "elemental_alignment"
        success = self._perform_elemental_alignment(intensity)
        if not success:
            logger.error("Elemental alignment phase failed")
            return False
            
        # Phase 3: Cycle Synchronization
        self.current_phase = "cycle_synchronization"
        success = self._perform_cycle_synchronization(intensity, duration)
        if not success:
            logger.error("Cycle synchronization phase failed")
            return False
            
        # Phase 4: Planetary Resonance
        self.current_phase = "planetary_resonance"
        success = self._perform_planetary_resonance(intensity)
        if not success:
            logger.error("Planetary resonance phase failed")
            return False
            
        # Phase 5: Gaia Connection
        self.current_phase = "gaia_connection"
        success = self._perform_gaia_connection(intensity, duration)
        if not success:
            logger.error("Gaia connection phase failed")
            return False
        
        # Complete harmonization
        self.harmonization_complete = True
        process_duration = time.time() - start_time
        self.metrics["process_duration"] = process_duration
        
        # Calculate final Earth resonance
        final_resonance = self._calculate_earth_resonance()
        self.metrics["final_resonance"] = final_resonance
        
        # Record metrics
        self._record_metrics()
        
        # Update soul
        self._update_soul_properties()
        
        logger.info(f"Earth harmonization completed in {process_duration:.2f} seconds")
        logger.info(f"Final Earth resonance: {final_resonance:.2f}")
        
        return True
    
    def _check_prerequisites(self):
        """
        Check if prerequisites for Earth harmonization are met.
        
        Returns:
            bool: True if prerequisites are met
        """
        # Check for life cord
        if not getattr(self.soul_spark, "cord_formation_complete", False):
            logger.warning("Life cord formation not complete")
            return False
            
        # Check if ready for Earth
        if not getattr(self.soul_spark, "ready_for_earth", False):
            logger.warning("Soul not ready for Earth harmonization")
            return False
            
        # Check for sufficient cord integrity
        cord_integrity = getattr(self.soul_spark, "cord_integrity", 0.0)
        if cord_integrity < 0.5:
            logger.warning(f"Life cord integrity too low: {cord_integrity:.2f}")
            return False
            
        return True
    
    def _calculate_earth_resonance(self):
        """
        Calculate the soul's current resonance with Earth.
        
        Returns:
            float: Earth resonance level (0.0-1.0)
        """
        # Get soul properties
        soul_freq = getattr(self.soul_spark, "frequency", 528.0)
        soul_elements = getattr(self.soul_spark, "elements", {})
        
        # Calculate frequency resonance component
        freq_factors = []
        for name, earth_freq in EARTH_FREQUENCIES.items():
            # Calculate harmonic relationship
            if earth_freq > 0 and soul_freq > 0:
                # Use the smaller ratio to get a value between 0-1
                ratio = min(earth_freq / soul_freq, soul_freq / earth_freq)
                freq_factors.append(ratio)
        
        # Average frequency resonance
        if freq_factors:
            freq_resonance = sum(freq_factors) / len(freq_factors)
        else:
            freq_resonance = 0.3  # Default value
        
        # Calculate elemental resonance component
        elem_resonance = 0.0
        earth_element_count = len(EARTH_ELEMENTS)
        
        if soul_elements:
            # Sum up soul's earth element strengths
            total_strength = 0.0
            for elem in EARTH_ELEMENTS:
                if elem in soul_elements:
                    total_strength += soul_elements[elem]
            
            # Calculate average elemental resonance
            if earth_element_count > 0:
                elem_resonance = total_strength / earth_element_count
        
        # Calculate overall resonance (weighted)
        overall_resonance = freq_resonance * 0.6 + elem_resonance * 0.4
        
        return overall_resonance
    
    def _perform_frequency_attunement(self, intensity, duration):
        """
        Attune the soul's frequency to Earth's primary frequencies.
        
        Args:
            intensity (float): Intensity of attunement
            duration (float): Relative duration factor
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning frequency attunement phase")
        
        # Get current soul frequency
        soul_freq = getattr(self.soul_spark, "frequency", 528.0)
        
        # Calculate weighted Earth frequency target
        # The Schumann resonance is most important, but we blend with others
        target_freq = (
            EARTH_FREQUENCIES["schumann"] * 0.5 +
            EARTH_FREQUENCIES["geomagnetic"] * 0.3 +
            soul_freq * 0.2  # Keep some of the soul's original frequency
        )
        
        # Calculate attunement amount based on intensity and duration
        attunement_factor = intensity * duration * 0.5
        freq_shift = (target_freq - soul_freq) * attunement_factor
        
        # Apply frequency shift
        new_freq = soul_freq + freq_shift
        
        # Calculate attunement level - avoid division by zero
        if abs(target_freq - soul_freq) > 0.001:
            attunement_level = attunement_factor * min(1.0, abs(freq_shift) / abs(target_freq - soul_freq))
        else:
            attunement_level = attunement_factor
        
        # Store results
        self.attunements["frequency"] = attunement_level
        self.phase_results["frequency_attunement"] = {
            "original_frequency": soul_freq,
            "target_frequency": target_freq,
            "new_frequency": new_freq,
            "attunement_level": attunement_level
        }
        
        # Update soul frequency
        setattr(self.soul_spark, "frequency", new_freq)
        
        # Update harmonics
        if hasattr(self.soul_spark, "harmonics"):
            harmonics = getattr(self.soul_spark, "harmonics")
            new_harmonics = [new_freq * (i+1) for i in range(len(harmonics))]
            setattr(self.soul_spark, "harmonics", new_harmonics)
        
        # Record phase metrics
        self.metrics["phase_metrics"]["frequency_attunement"] = {
            "frequency_shift": freq_shift,
            "attunement_level": attunement_level
        }
        
        logger.info(f"Frequency attunement complete: {soul_freq:.2f} Hz → {new_freq:.2f} Hz")
        logger.info(f"Attunement level: {attunement_level:.2f}")
        
        return True
    
    def _perform_elemental_alignment(self, intensity):
        """
        Align the soul with Earth's elemental forces.
        
        Args:
            intensity (float): Intensity of alignment
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning elemental alignment phase")
        
        # Get current soul elements
        soul_elements = getattr(self.soul_spark, "elements", {})
        
        # Copy to avoid modifying during iteration
        soul_elements = soul_elements.copy() if soul_elements else {}
        
        # Calculate alignment for each element
        alignment_results = {}
        
        # Get malkuth aspect for its element
        try:
            malkuth_aspects = aspect_dictionary.get_aspects("malkuth")
            malkuth_element = malkuth_aspects.get("element", "earth").lower() if malkuth_aspects else "earth"
        except (NameError, AttributeError):
            # Fall back if aspect_dictionary not available
            malkuth_element = "earth"
        
        for element in EARTH_ELEMENTS:
            # Get current element strength or default to 0
            current_strength = soul_elements.get(element, 0.0)
            
            # Calculate target strength - emphasize Earth's primary element
            if element == malkuth_element:
                target_strength = 0.8  # Primary element
            else:
                target_strength = 0.5  # Other elements
            
            # Calculate adjustment amount
            adjustment = (target_strength - current_strength) * intensity * 0.7
            
            # Apply adjustment
            new_strength = current_strength + adjustment
            new_strength = max(0.1, min(1.0, new_strength))
            
            # Store new strength
            soul_elements[element] = new_strength
            
            # Calculate alignment level for this element
            alignment_level = 1.0 - abs(target_strength - new_strength) / max(0.1, target_strength)
            
            # Store results
            alignment_results[element] = {
                "original_strength": current_strength,
                "target_strength": target_strength,
                "new_strength": new_strength,
                "alignment_level": alignment_level
            }
        
        # Calculate overall elemental alignment
        overall_alignment = sum(r["alignment_level"] for r in alignment_results.values()) / len(EARTH_ELEMENTS)
        
        # Store results
        self.attunements["elemental"] = {
            "elements": soul_elements,
            "overall_alignment": overall_alignment
        }
        
        self.phase_results

        self.phase_results["elemental_alignment"] = {
            "element_results": alignment_results,
            "overall_alignment": overall_alignment
        }
        
        # Update soul elements
        setattr(self.soul_spark, "elements", soul_elements)
        
        # Record phase metrics
        self.metrics["phase_metrics"]["elemental_alignment"] = {
            "overall_alignment": overall_alignment,
            "element_alignments": {e: r["alignment_level"] for e, r in alignment_results.items()}
        }
        
        logger.info(f"Elemental alignment complete with overall level: {overall_alignment:.2f}")
        return True
    
    def _perform_cycle_synchronization(self, intensity, duration):
        """
        Synchronize the soul with Earth's natural cycles.
        
        Args:
            intensity (float): Intensity of synchronization
            duration (float): Relative duration factor
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning cycle synchronization phase")
        
        # Get current soul cycles or initialize
        soul_cycles = getattr(self.soul_spark, "earth_cycles", {})
        
        # Copy to avoid modifying during iteration
        soul_cycles = soul_cycles.copy() if soul_cycles else {}
        
        # Calculate synchronization for each cycle
        sync_results = {}
        earth_cycles = {
            "diurnal": {"freq": EARTH_FREQUENCIES["diurnal"], "importance": 0.9},
            "lunar": {"freq": EARTH_FREQUENCIES["lunar"], "importance": 0.7},
            "annual": {"freq": EARTH_FREQUENCIES["annual"], "importance": 0.8}
        }
        
        for cycle_name, cycle_info in earth_cycles.items():
            # Get current synchronization or default
            current_sync = soul_cycles.get(cycle_name, 0.3)
            
            # Calculate target synchronization - based on importance
            target_sync = cycle_info["importance"]
            
            # Calculate adjustment amount
            adjustment = (target_sync - current_sync) * intensity * duration * 0.8
            
            # Apply adjustment
            new_sync = current_sync + adjustment
            new_sync = max(0.1, min(1.0, new_sync))
            
            # Store new synchronization
            soul_cycles[cycle_name] = new_sync
            
            # Calculate sync level for this cycle
            if target_sync > 0:
                sync_level = 1.0 - abs(target_sync - new_sync) / target_sync
            else:
                sync_level = 0.5  # Default if target is zero
            
            # Store results
            sync_results[cycle_name] = {
                "original_sync": current_sync,
                "target_sync": target_sync,
                "new_sync": new_sync,
                "sync_level": sync_level
            }
        
        # Calculate overall cycle synchronization
        overall_sync = sum(r["sync_level"] for r in sync_results.values()) / len(earth_cycles)
        
        # Store results
        self.attunements["cycles"] = {
            "cycle_syncs": soul_cycles,
            "overall_sync": overall_sync
        }
        
        self.phase_results["cycle_synchronization"] = {
            "cycle_results": sync_results,
            "overall_sync": overall_sync
        }
        
        # Update soul cycles
        setattr(self.soul_spark, "earth_cycles", soul_cycles)
        
        # Record phase metrics
        self.metrics["phase_metrics"]["cycle_synchronization"] = {
            "overall_sync": overall_sync,
            "cycle_syncs": {c: r["sync_level"] for c, r in sync_results.items()}
        }
        
        logger.info(f"Cycle synchronization complete with overall level: {overall_sync:.2f}")
        return True
    
    def _perform_planetary_resonance(self, intensity):
        """
        Attune the soul to planetary resonance.
        
        Args:
            intensity (float): Intensity of attunement
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning planetary resonance phase")
        
        # Get current planetary resonance or initialize
        current_resonance = getattr(self.soul_spark, "planetary_resonance", 0.3)
        
        # Calculate target resonance
        target_resonance = 0.7  # Moderate-high planetary connection
        
        # Get malkuth aspects to enhance planetary connection
        try:
            malkuth_aspects = aspect_dictionary.get_aspects("malkuth")
        except (NameError, AttributeError):
            malkuth_aspects = None
        
        # Calculate adjustment based on soul and malkuth aspects
        adjustment = (target_resonance - current_resonance) * intensity * 0.9
        
        # Apply adjustment
        new_resonance = current_resonance + adjustment
        new_resonance = max(0.1, min(1.0, new_resonance))
        
        # Calculate resonance level
        if target_resonance > 0:
            resonance_level = 1.0 - abs(target_resonance - new_resonance) / target_resonance
        else:
            resonance_level = 0.5  # Default if target is zero
        
        # Store results
        self.attunements["planetary"] = resonance_level
        
        self.phase_results["planetary_resonance"] = {
            "original_resonance": current_resonance,
            "target_resonance": target_resonance,
            "new_resonance": new_resonance,
            "resonance_level": resonance_level
        }
        
        # Update soul planetary resonance
        setattr(self.soul_spark, "planetary_resonance", new_resonance)
        
        # Record phase metrics
        self.metrics["phase_metrics"]["planetary_resonance"] = {
            "resonance_level": resonance_level,
            "adjustment": adjustment
        }
        
        logger.info(f"Planetary resonance complete with level: {resonance_level:.2f}")
        return True
    
    def _perform_gaia_connection(self, intensity, duration):
        """
        Establish connection with Gaia consciousness.
        
        Args:
            intensity (float): Intensity of connection
            duration (float): Relative duration factor
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning Gaia connection phase")
        
        # Get current Gaia connection or initialize
        current_connection = getattr(self.soul_spark, "gaia_connection", 0.2)
        
        # Calculate target connection
        target_connection = 0.6  # Moderate Gaia connection
        
        # Calculate adjustment based on intensity and duration
        adjustment = (target_connection - current_connection) * intensity * duration
        
        # Apply adjustment
        new_connection = current_connection + adjustment
        new_connection = max(0.1, min(0.9, new_connection))  # Cap at 0.9
        
        # Calculate connection level
        if target_connection > 0:
            connection_level = 1.0 - abs(target_connection - new_connection) / target_connection
        else:
            connection_level = 0.5  # Default if target is zero
        
        # Store results
        self.attunements["gaia_connection"] = connection_level
        
        self.phase_results["gaia_connection"] = {
            "original_connection": current_connection,
            "target_connection": target_connection,
            "new_connection": new_connection,
            "connection_level": connection_level
        }
        
        # Update soul Gaia connection
        setattr(self.soul_spark, "gaia_connection", new_connection)
        
        # Record phase metrics
        self.metrics["phase_metrics"]["gaia_connection"] = {
            "connection_level": connection_level,
            "adjustment": adjustment
        }
        
        logger.info(f"Gaia connection established with level: {connection_level:.2f}")
        return True
    
    def _update_soul_properties(self):
        """Update the soul with Earth harmonization properties."""
        # Set Earth harmonization properties
        setattr(self.soul_spark, "earth_harmonized", True)
        setattr(self.soul_spark, "earth_resonance", self.metrics["final_resonance"])
        setattr(self.soul_spark, "earth_attunements", self.attunements.copy())
        
        # Adjust soul properties based on harmonization
        
        # Stability increase
        current_stability = getattr(self.soul_spark, "stability", 0.7)
        stability_bonus = self.metrics["final_resonance"] * 0.15
        new_stability = min(1.0, current_stability + stability_bonus)
        setattr(self.soul_spark, "stability", new_stability)
        
        # Coherence increase
        current_coherence = getattr(self.soul_spark, "coherence", 0.7)
        coherence_bonus = self.metrics["final_resonance"] * 0.1
        new_coherence = min(1.0, current_coherence + coherence_bonus)
        setattr(self.soul_spark, "coherence", new_coherence)
        
        # Mark as ready for birth
        setattr(self.soul_spark, "ready_for_birth", True)
        
        logger.info(f"Soul updated with Earth harmonization properties")
        logger.info(f"New stability: {new_stability:.2f}, New coherence: {new_coherence:.2f}")
    
    def _record_metrics(self):
        """Record final metrics after Earth harmonization."""
        try:
            # Record to central metrics system
            metrics.record_metrics(
                "earth_harmonization",
                "initial_resonance",
                self.metrics["initial_resonance"]
            )
            
            metrics.record_metrics(
                "earth_harmonization",
                "final_resonance",
                self.metrics["final_resonance"]
            )
            
            metrics.record_metrics(
                "earth_harmonization",
                "process_duration",
                self.metrics["process_duration"]
            )
            
            # Record attunements
            for attunement_type, value in self.attunements.items():
                if isinstance(value, dict) and "overall_alignment" in value:
                    # Record overall alignment/sync for complex attunements
                    metrics.record_metrics(
                        "earth_harmonization",
                        f"{attunement_type}_alignment",
                        value["overall_alignment"]
                    )
                elif not isinstance(value, dict):
                    # Record simple attunement values
                    metrics.record_metrics(
                        "earth_harmonization",
                        f"{attunement_type}_attunement",
                        value
                    )
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
        
        logger.info("Final metrics recorded for Earth harmonization")


# Example usage
if __name__ == "__main__":
    # Create a simple test class if SoulSpark is not available
    try:
        from void.soul_spark import SoulSpark
        soul = SoulSpark()
    except ImportError:
        # Create a simple test object
        class TestSoulSpark:
            def __init__(self):
                self.id = "test-soul-1"
                self.frequency = 528.0
                self.stability = 0.75
                self.coherence = 0.75
                self.cord_formation_complete = True
                self.ready_for_earth = True
                self.cord_integrity = 0.7
                self.elements = {}
        
        soul = TestSoulSpark()
    
    # Initialize Earth harmonization
    harmonization = EarthHarmonization(soul)
    
    # Perform harmonization
    harmonization.harmonize(intensity=0.8, duration=1.2)
    
    # Print results
    print(f"Soul ID: {getattr(soul, 'id', 'unknown')}")
    print(f"Earth Resonance: {soul.earth_resonance:.2f}")
    print(f"Frequency: {soul.frequency:.2f} Hz")
    print(f"Elements: {soul.elements}")
    print(f"Earth Cycles: {getattr(soul, 'earth_cycles', {})}")
    print(f"Gaia Connection: {soul.gaia_connection:.2f}")
    print(f"Ready for Birth: {soul.ready_for_birth}")
