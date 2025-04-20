"""
Soul Formation - Sephiroth Journey

This module implements the soul's journey through the Sephiroth dimensions,
forming the soul's structural and energetic composition through interactions
with each dimension.

The soul traverses the Tree of Life in a specific sequence, forming entanglements
and acquiring aspects from each Sephirah based on its resonance patterns.

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
    filename='soul_formation.log'
)
logger = logging.getLogger('soul_formation')

# Add parent directory to path to import from parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
try:
    from sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    from sephiroth.sephiroth_controller import SephirothController
    from void.soul_spark import SoulSpark
    import metrics_tracking as metrics
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class SoulFormation:
    """
    Soul Formation through Sephiroth Journey
    
    Manages the soul's journey through the Sephiroth dimensions, forming
    the soul's structural and energetic composition through interactions
    with each dimension.
    """
    
    def __init__(self, soul_spark, sephiroth_controller=None):
        """
        Initialize the soul formation process.
        
        Args:
            soul_spark: The soul spark to be formed
            sephiroth_controller: Controller for Sephiroth navigation (optional)
        """
        self.soul_spark = soul_spark
        
        # Create a new controller if none provided
        if sephiroth_controller is None:
            self.sephiroth_controller = SephirothController()
        else:
            self.sephiroth_controller = sephiroth_controller
            
        # Journey state tracking
        self.current_sephirah = None
        self.journey_path = []
        self.journey_complete = False
        
        # Initialize soul layers
        self.layers = {}
        self.entanglements = {}
        self.aspect_acquisitions = {}
        
        # Track metrics
        self.metrics = {
            "resonance_scores": {},
            "aspect_acquisitions": {},
            "journey_duration": 0,
            "entanglement_strengths": {},
            "transformation_metrics": {}
        }
        
        logger.info(f"Soul Formation initialized for soul spark {soul_spark.id}")
    
    def begin_journey(self, journey_type="traditional"):
        """
        Begin the soul's journey through the Sephiroth.
        
        Args:
            journey_type (str): Type of journey to undertake
                "traditional": Follows the traditional path up the Tree of Life
                "personalized": Creates a custom path based on soul resonance
                "direct": Direct path through strongest resonance points
                
        Returns:
            bool: Success status
        """
        logger.info(f"Beginning {journey_type} journey for soul spark {self.soul_spark.id}")
        
        # Set journey path based on type
        if journey_type == "traditional":
            self.journey_path = self._get_traditional_path()
        elif journey_type == "personalized":
            self.journey_path = self._get_personalized_path()
        elif journey_type == "direct":
            self.journey_path = self._get_direct_path()
        else:
            logger.error(f"Unknown journey type: {journey_type}")
            return False
            
        # Begin at the first Sephirah
        if not self.journey_path:
            logger.error("Journey path is empty")
            return False
            
        self.current_sephirah = self.journey_path[0]
        
        # Record initial state
        self._record_initial_metrics()
        
        logger.info(f"Journey begun with path: {self.journey_path}")
        return True
    
    def _get_traditional_path(self):
        """
        Get the traditional path up the Tree of Life.
        
        Returns:
            list: Ordered list of Sephiroth to visit
        """
        # Traditional path starts at Malkuth and moves upward
        return [
            "malkuth",  # Kingdom - Physical reality
            "yesod",    # Foundation - Astral/Ethereal
            "hod",      # Glory - Intellectual
            "netzach",  # Victory - Emotional
            "tiphareth",# Beauty - Harmony/Self
            "geburah",  # Severity - Discipline/Strength
            "chesed",   # Mercy - Compassion/Love
            "binah",    # Understanding - Divine feminine
            "chokmah",  # Wisdom - Divine masculine
            "kether"    # Crown - Divine unity
        ]
    
    def _get_personalized_path(self):
        """
        Generate a personalized path based on soul resonance.
        
        Returns:
            list: Customized path through the Sephiroth
        """
        # Calculate resonance with each Sephirah
        resonance_scores = {}
        for sephirah in aspect_dictionary.sephiroth_names:
            resonance = self._calculate_resonance(sephirah)
            resonance_scores[sephirah] = resonance
            
        # Record resonance scores
        self.metrics["resonance_scores"] = resonance_scores
        
        # Always start with Malkuth and end with Kether
        path = ["malkuth"]
        remaining = [s for s in aspect_dictionary.sephiroth_names 
                    if s not in ["malkuth", "kether"]]
        
        # Order remaining Sephiroth by resonance
        remaining.sort(key=lambda s: resonance_scores.get(s, 0), reverse=True)
        
        # Create path
        path.extend(remaining)
        path.append("kether")
        
        return path
    
    def _get_direct_path(self):
        """
        Generate a direct path through strongest resonance points.
        
        Returns:
            list: Direct path through key Sephiroth
        """
        # Calculate resonance with each Sephirah
        resonance_scores = {}
        for sephirah in aspect_dictionary.sephiroth_names:
            resonance = self._calculate_resonance(sephirah)
            resonance_scores[sephirah] = resonance
        
        # Record resonance scores
        self.metrics["resonance_scores"] = resonance_scores
        
        # Select top 5 resonating Sephiroth plus Malkuth and Kether
        top_sephiroth = sorted(resonance_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Always include Malkuth and Kether
        must_include = ["malkuth", "kether"]
        
        # Start with Malkuth
        path = ["malkuth"]
        
        # Add top resonating (that aren't already in must_include)
        added = 0
        for sephirah, _ in top_sephiroth:
            if sephirah not in must_include and added < 3:
                path.append(sephirah)
                added += 1
        
        # End with Kether
        if "kether" not in path:
            path.append("kether")
            
        return path
    
    def _calculate_resonance(self, sephirah):
        """
        Calculate soul resonance with a specific Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            float: Resonance score (0-1)
        """
        # Get aspects for this Sephirah
        aspects = aspect_dictionary.get_aspects(sephirah)
        
        if not aspects:
            return 0.0
            
        # Base resonance calculation
        primary_aspects = aspects.get("primary_aspects", [])
        phi_harmonic_count = aspects.get("phi_harmonic_count", 0)
        frequency_modifier = aspects.get("frequency_modifier", 0.75)
        
        # Get soul properties for resonance calculation
        soul_frequency = getattr(self.soul_spark, "frequency", 528.0)
        soul_phi_resonance = getattr(self.soul_spark, "phi_resonance", 0.5)
        soul_pattern = getattr(self.soul_spark, "pattern_stability", 0.6)
        
        # Calculate frequency resonance
        sephirah_freq = 432.0 * frequency_modifier
        freq_ratio = min(soul_frequency / sephirah_freq, 
                       sephirah_freq / soul_frequency)
        freq_resonance = freq_ratio * 0.8 + 0.2  # Scale to 0.2-1.0
        
        # Calculate phi resonance
        phi_resonance = (soul_phi_resonance * phi_harmonic_count) / 7.0
        phi_resonance = min(1.0, phi_resonance)
        
        # Count matching aspects
        soul_aspects = getattr(self.soul_spark, "aspects", [])
        aspect_matches = sum(1 for a in soul_aspects if a in primary_aspects)
        aspect_resonance = aspect_matches / max(1, len(primary_aspects))
        
        # Combine for total resonance
        total_resonance = (
            freq_resonance * 0.4 +
            phi_resonance * 0.3 +
            aspect_resonance * 0.3
        )
        
        # Add slight randomness for variability (±5%)
        randomness = 1.0 + (np.random.random() - 0.5) * 0.1
        total_resonance *= randomness
        
        # Ensure in range 0-1
        total_resonance = max(0.0, min(1.0, total_resonance))
        
        return total_resonance
    
    def _record_initial_metrics(self):
        """Record initial metrics before journey begins."""
        # Record soul spark properties
        metrics.record_metric(
            "soul_formation", 
            "initial_frequency", 
            getattr(self.soul_spark, "frequency", 0.0)
        )
        
        metrics.record_metric(
            "soul_formation",
            "initial_stability",
            getattr(self.soul_spark, "stability", 0.0)
        )
        
        # Record journey properties
        metrics.record_metric(
            "soul_formation",
            "journey_path",
            self.journey_path
        )
    
    def process_current_sephirah(self):
        """
        Process the current Sephirah in the journey.
        
        This entangles the soul with the current Sephirah and 
        acquires aspects from it.
        
        Returns:
            bool: Success status
        """
        if not self.current_sephirah:
            logger.error("No current Sephirah to process")
            return False
            
        logger.info(f"Processing Sephirah: {self.current_sephirah}")
        
        # Get aspects for this Sephirah
        aspects = aspect_dictionary.get_aspects(self.current_sephirah)
        
        if not aspects:
            logger.error(f"No aspects found for {self.current_sephirah}")
            return False
            
        # Create entanglement
        entanglement_strength = self._create_entanglement(self.current_sephirah)
        self.entanglements[self.current_sephirah] = entanglement_strength
        
        # Acquire aspects
        acquired_aspects = self._acquire_aspects(self.current_sephirah, entanglement_strength)
        self.aspect_acquisitions[self.current_sephirah] = acquired_aspects
        
        # Form soul layer
        self._form_soul_layer(self.current_sephirah, aspects, entanglement_strength)
        
        # Record metrics
        metrics.record_metric(
            "soul_formation",
            f"entanglement_{self.current_sephirah}",
            entanglement_strength
        )
        
        metrics.record_metric(
            "soul_formation",
            f"aspects_{self.current_sephirah}",
            acquired_aspects
        )
        
        logger.info(f"Processed {self.current_sephirah}: "
                   f"Entanglement={entanglement_strength:.2f}, "
                   f"Aspects={len(acquired_aspects)}")
        
        return True
    
    def _create_entanglement(self, sephirah):
        """
        Create entanglement between soul and Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            float: Entanglement strength (0-1)
        """
        # Calculate resonance
        resonance = self._calculate_resonance(sephirah)
        
        # Base entanglement strength on resonance and stability
        soul_stability = getattr(self.soul_spark, "stability", 0.5)
        soul_coherence = getattr(self.soul_spark, "coherence", 0.5)
        
        # Calculate base entanglement strength
        entanglement = resonance * 0.7 + soul_stability * 0.15 + soul_coherence * 0.15
        
        # Ensure proper range
        entanglement = max(0.3, min(1.0, entanglement))
        
        # Add slight randomness for uniqueness
        randomness = 1.0 + (np.random.random() - 0.5) * 0.1
        entanglement = entanglement * randomness
        entanglement = max(0.3, min(1.0, entanglement))
        
        # Integrate with Sephiroth controller
        result = self.sephiroth_controller.create_entanglement(
            self.soul_spark, sephirah, entanglement
        )
        
        # Update soul properties based on entanglement
        self._update_soul_properties(sephirah, entanglement)
        
        logger.info(f"Created entanglement with {sephirah}: {entanglement:.2f}")
        return entanglement
    
    def _acquire_aspects(self, sephirah, entanglement_strength):
        """
        Acquire aspects from the current Sephirah based on entanglement strength.
        
        Args:
            sephirah (str): Name of the Sephirah
            entanglement_strength (float): Strength of entanglement
            
        Returns:
            dict: Acquired aspects with their strengths
        """
        # Get aspects for this Sephirah
        sephirah_aspects = aspect_dictionary.get_aspects(sephirah)
        
        if not sephirah_aspects:
            logger.warning(f"No aspects found for {sephirah}")
            return {}
            
        # Get primary and secondary aspects
        primary_aspects = sephirah_aspects.get("primary_aspects", [])
        secondary_aspects = sephirah_aspects.get("secondary_aspects", [])
        
        acquired = {}
        
        # Determine threshold for acquisition based on entanglement
        primary_threshold = 0.3  # Always acquire some primary aspects
        secondary_threshold = 0.5  # Need stronger entanglement for secondary
        
        # Acquire primary aspects
        for aspect in primary_aspects:
            # Calculate acquisition strength based on entanglement
            # Strong entanglement = stronger aspect acquisition
            aspect_strength = entanglement_strength * (0.7 + np.random.random() * 0.3)
            
            # Ensure minimum strength for complete soul formation
            aspect_strength = max(0.3, aspect_strength)
            
            # Only acquire if above threshold
            if aspect_strength >= primary_threshold:
                acquired[aspect] = aspect_strength
        
        # Acquire secondary aspects if entanglement is strong enough
        if entanglement_strength >= secondary_threshold:
            for aspect in secondary_aspects:
                # Secondary aspects are generally weaker
                aspect_strength = entanglement_strength * (0.4 + np.random.random() * 0.3)
                
                # Only acquire if above threshold
                if aspect_strength >= secondary_threshold:
                    acquired[aspect] = aspect_strength
        
        # Add aspects to soul
        self._add_aspects_to_soul(acquired)
        
        logger.info(f"Acquired {len(acquired)} aspects from {sephirah}")
        return acquired
    
    def _add_aspects_to_soul(self, aspects):
        """
        Add the acquired aspects to the soul.
        
        Args:
            aspects (dict): Aspects with their strengths
        """
        # Get existing soul aspects
        soul_aspects = getattr(self.soul_spark, "aspects", {})
        
        # Update existing aspects or add new ones
        for aspect, strength in aspects.items():
            if aspect in soul_aspects:
                # Strengthen existing aspect (max 1.0)
                soul_aspects[aspect] = min(1.0, soul_aspects[aspect] + strength * 0.3)
            else:
                # Add new aspect
                soul_aspects[aspect] = strength
        
        # Update soul
        setattr(self.soul_spark, "aspects", soul_aspects)
    
    def _form_soul_layer(self, sephirah, aspects, entanglement_strength):
        """
        Form a soul layer corresponding to the current Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            aspects (dict): Sephirah aspects
            entanglement_strength (float): Strength of entanglement
        """
        # Create a new layer
        layer = {
            "sephirah": sephirah,
            "entanglement_strength": entanglement_strength,
            "formation_time": time.time(),
            "frequency": aspects.get("frequency_modifier", 0.75) * 432.0,
            "color": aspects.get("color", "white"),
            "element": aspects.get("element", "spirit"),
            "planetary_correspondence": aspects.get("planetary_correspondence", None),
            "geometric_correspondence": aspects.get("geometric_correspondence", None)
        }
        
        # Add layer to soul
        self.layers[sephirah] = layer
        
        # Update soul properties based on new layer
        self._integrate_layer(layer)
        
        logger.info(f"Formed soul layer for {sephirah}")
    
    def _integrate_layer(self, layer):
        """
        Integrate a new layer into the soul structure.
        
        Args:
            layer (dict): Layer data
        """
        # Get current soul properties
        soul_frequency = getattr(self.soul_spark, "frequency", 528.0)
        soul_stability = getattr(self.soul_spark, "stability", 0.5)
        soul_coherence = getattr(self.soul_spark, "coherence", 0.5)
        
        # Integrate layer frequency - weighted average based on entanglement
        layer_weight = layer["entanglement_strength"]
        new_frequency = (soul_frequency + layer["frequency"] * layer_weight) / (1 + layer_weight)
        
        # Update stability and coherence
        stability_boost = layer["entanglement_strength"] * 0.1
        coherence_boost = layer["entanglement_strength"] * 0.08
        
        new_stability = min(1.0, soul_stability + stability_boost)
        new_coherence = min(1.0, soul_coherence + coherence_boost)
        
        # Update soul properties
        setattr(self.soul_spark, "frequency", new_frequency)
        setattr(self.soul_spark, "stability", new_stability)
        setattr(self.soul_spark, "coherence", new_coherence)
        
        # Add element to soul's elemental composition
        self._add_element(layer["element"], layer["entanglement_strength"])
        
        logger.info(f"Integrated layer: frequency={new_frequency:.2f}, "
                   f"stability={new_stability:.2f}, coherence={new_coherence:.2f}")
    
    def _add_element(self, element, strength):
        """
        Add an elemental component to the soul.
        
        Args:
            element (str): Element name
            strength (float): Element strength
        """
        # Get current elemental composition
        elements = getattr(self.soul_spark, "elements", {})
        
        # Add or strengthen element
        if element in elements:
            elements[element] = min(1.0, elements[element] + strength * 0.2)
        else:
            elements[element] = strength * 0.7
            
        # Update soul
        setattr(self.soul_spark, "elements", elements)
    
    def _update_soul_properties(self, sephirah, entanglement_strength):
        """
        Update soul properties based on Sephirah entanglement.
        
        Args:
            sephirah (str): Name of the Sephirah
            entanglement_strength (float): Strength of entanglement
        """
        # Different Sephiroth affect the soul in different ways
        if sephirah == "kether":
            # Kether increases divine connection and highest consciousness
            self._update_soul_attribute("divine_connection", 0.15 * entanglement_strength)
            self._update_soul_attribute("consciousness_level", 0.2 * entanglement_strength)
            
        elif sephirah == "chokmah":
            # Chokmah increases wisdom and insight
            self._update_soul_attribute("wisdom", 0.2 * entanglement_strength)
            self._update_soul_attribute("insight", 0.15 * entanglement_strength)
            
        elif sephirah == "binah":
            # Binah increases understanding and pattern recognition
            self._update_soul_attribute("understanding", 0.2 * entanglement_strength)
            self._update_soul_attribute("pattern_recognition", 0.15 * entanglement_strength)
            
        elif sephirah == "chesed":
            # Chesed increases compassion and love
            self._update_soul_attribute("compassion", 0.2 * entanglement_strength)
            self._update_soul_attribute("love", 0.15 * entanglement_strength)
            
        elif sephirah == "geburah":
            # Geburah increases strength and discipline
            self._update_soul_attribute("strength", 0.2 * entanglement_strength)
            self._update_soul_attribute("discipline", 0.15 * entanglement_strength)
            
        elif sephirah == "tiphareth":
            # Tiphareth increases harmony and self-awareness
            self._update_soul_attribute("harmony", 0.2 * entanglement_strength)
            self._update_soul_attribute("self_awareness", 0.15 * entanglement_strength)
            
        elif sephirah == "netzach":
            # Netzach increases emotion and creativity
            self._update_soul_attribute("emotional_capacity", 0.2 * entanglement_strength)
            self._update_soul_attribute("creativity", 0.15 * entanglement_strength)
            
        elif sephirah == "hod":
            # Hod increases intellect and communication
            self._update_soul_attribute("intellect", 0.2 * entanglement_strength)
            self._update_soul_attribute("communication", 0.15 * entanglement_strength)
            
        elif sephirah == "yesod":
            # Yesod increases foundation and dreams
            self._update_soul_attribute("foundation", 0.2 * entanglement_strength)
            self._update_soul_attribute("dream_capacity", 0.15 * entanglement_strength)
            
        elif sephirah == "malkuth":
            # Malkuth increases groundedness and manifestation
            self._update_soul_attribute("groundedness", 0.2 * entanglement_strength)
            self._update_soul_attribute("manifestation", 0.15 * entanglement_strength)
    
    def _update_soul_attribute(self, attribute, value):
        """
        Update a soul attribute, creating it if it doesn't exist.
        
        Args:
            attribute (str): Attribute name
            value (float): Value to add
        """
        current = getattr(self.soul_spark, attribute, 0.0)
        setattr(self.soul_spark, attribute, min(1.0, current + value))
    
    def advance_to_next_sephirah(self):
        """
        Advance to the next Sephirah in the journey.
        
        Returns:
            bool: True if advanced, False if journey complete
        """
        if not self.journey_path:
            logger.error("No journey path defined")
            return False
            
        # Find current position
        try:
            current_index = self.journey_path.index(self.current_sephirah)
        except ValueError:
            logger.error(f"Current Sephirah {self.current_sephirah} not in journey path")
            return False
            
        # Check if this is the last Sephirah
        if current_index >= len(self.journey_path) - 1:
            logger.info("Journey complete - reached final Sephirah")
            self.journey_complete = True
            return False
            
        # Advance to next
        self.current_sephirah = self.journey_path[current_index + 1]
        logger.info(f"Advanced to next Sephirah: {self.current_sephirah}")
        
        return True
    
    def complete_journey(self):
        """
        Complete the journey and finalize the soul formation.
        
        Returns:
            bool: Success status
        """
        if not self.journey_complete:
            logger.warning("Completing journey before reaching final Sephirah")
            
        # Ensure all Sephiroth have been entangled with
        missing_sephiroth = [s for s in aspect_dictionary.sephiroth_names 
                          if s not in self.entanglements]
        
        # Create minimal entanglements with any missing Sephiroth to ensure completeness
        for sephirah in missing_sephiroth:
            logger.info(f"Creating minimal entanglement with missing Sephirah: {sephirah}")
            entanglement_strength = 0.3 + np.random.random() * 0.1
            self.entanglements[sephirah] = entanglement_strength
            
            # Acquire minimal aspects
            acquired_aspects = self._acquire_aspects(sephirah, entanglement_strength)
            self.aspect_acquisitions[sephirah] = acquired_aspects
            
            # Form minimal soul layer
            aspects = aspect_dictionary.get_aspects(sephirah)
            if aspects:
                self._form_soul_layer(sephirah, aspects, entanglement_strength)
        
        # Finalize soul structure
        self._finalize_soul_structure()
        
        # Record completion
        self.journey_complete = True
        self.metrics["journey_duration"] = time.time() - self.metrics.get("start_time", time.time())
        
        # Record final metrics
        self._record_final_metrics()
        
        logger.info("Soul formation journey completed")
        return True
    
    def _finalize_soul_structure(self):
        """Finalize the soul structure after journey completion."""
        # Calculate overall stability and coherence
        total_entanglement = sum(self.entanglements.values())
        avg_entanglement = total_entanglement / max(1, len(self.entanglements))
        
        # Increase stability based on complete journey
        stability_bonus = 0.1 + avg_entanglement * 0.1
        current_stability = getattr(self.soul_spark, "stability", 0.6)
        new_stability = min(1.0, current_stability + stability_bonus)
        
        # Increase coherence based on journey
        coherence_bonus = 0.1 + avg_entanglement * 0.15
        current_coherence = getattr(self.soul_spark, "coherence", 0.6)
        new_coherence = min(1.0, current_coherence + coherence_bonus)
        
        # Update soul properties
        setattr(self.soul_spark, "stability", new_stability)
        setattr(self.soul_spark, "coherence", new_coherence)
        setattr(self.soul_spark, "formation_complete", True)
        setattr(self.soul_spark, "sephiroth_entanglements", self.entanglements.copy())
        
        logger.info(f"Finalized soul structure: stability={new_stability:.2f}, coherence={new_coherence:.2f}")
    
    def _record_final_metrics(self):
        """Record final metrics after journey completion."""
        # Record soul properties
        metrics.record_metric(
            "soul_formation", 
            "final_frequency", 
            getattr(self.soul_spark, "frequency", 0.0)
        )
        
        metrics.record_metric(
            "soul_formation",
            "final_stability",
            getattr(self.soul_spark, "stability", 0.0)
        )
        
        metrics.record_metric(
            "soul_formation",
            "final_coherence",
            getattr(self.soul_spark, "coherence", 0.0)
        )
        
        # Record entanglements
        metrics.record_metric(
            "soul_formation",
            "sephiroth_entanglements",
            self.entanglements
        )
        
        # Record aspects
        metrics.record_metric(
            "soul_formation",
            "acquired_aspects",
            getattr(self.soul_spark, "aspects", {})
        )
        
        # Record elemental composition
        metrics.record_metric(
            "soul_formation",
            "elemental_composition",
            getattr(self.soul_spark, "elements", {})
        )
        
        logger.info("Final metrics recorded for soul formation")


# Example usage
if __name__ == "__main__":
    # Create a soul spark
    from void.soul_spark import SoulSpark
    
    soul = SoulSpark()
    
    # Initialize soul formation
    formation = SoulFormation(soul)
    
    # Begin journey
    formation.begin_journey("personalized")
    
    # Process all Sephiroth in the journey
    while True:
        formation.process_current_sephirah()
        if not formation.advance_to_next_sephirah():
            break
    
    # Complete the journey
    formation.complete_journey()
    
    # Print final soul properties
    print(f"Soul ID: {soul.id}")
    print(f"Frequency: {soul.frequency:.2f}")
    print(f"Stability: {soul.stability:.2f}")
    print(f"Coherence: {soul.coherence:.2f}")
    print(f"Elements: {soul.elements}")
    print(f"Number of aspects: {len(soul.aspects)}")


