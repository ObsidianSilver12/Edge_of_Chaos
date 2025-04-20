"""
Daath Aspects Module

This module defines the specific aspects for Daath, Knowledge - the hidden 
Sephirah representing hidden knowledge, the abyss, and the connecting point
between the supernal triangle and the lower Sephiroth.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.daath')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    KNOWLEDGE = auto()
    HIDDEN = auto()
    GATEWAY = auto()
    TRANSITION = auto()
    VOID = auto()
    INTEGRATION = auto()

class DaathAspects:
    """
    Daath (Knowledge) Aspects
    
    Contains the specific aspects, frequencies, and properties of Daath,
    the hidden Sephirah representing divine knowledge, the abyss, and
    the threshold between the supernal and the manifest.
    """
    
    def __init__(self):
        """Initialize Daath aspects"""
        logger.info("Initializing Daath aspects")
        self.base_frequency = 819.0  # Hz (specific to Daath's resonance)
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Daath"""
        aspects = {
            # Primary aspects of Daath
            "hidden_knowledge": {
                "type": AspectType.KNOWLEDGE,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "The hidden knowledge beyond ordinary comprehension"
            },
            "abyss": {
                "type": AspectType.VOID,
                "frequency": self.base_frequency * 1.207,
                "strength": 0.98,
                "description": "The void between the supernal and the manifest"
            },
            "gateway": {
                "type": AspectType.GATEWAY,
                "frequency": self.base_frequency * 1.382,
                "strength": 0.96,
                "description": "The gateway between worlds"
            },
            "hidden_sephirah": {
                "type": AspectType.HIDDEN,
                "frequency": self.base_frequency * 0.786,
                "strength": 0.95,
                "description": "The invisible dimension of the Tree of Life"
            },
            "threshold": {
                "type": AspectType.TRANSITION,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.94,
                "description": "The threshold of consciousness"
            },
            "integration_point": {
                "type": AspectType.INTEGRATION,
                "frequency": self.base_frequency * 2.0,
                "strength": 0.92,
                "description": "The point where supernal knowledge integrates with lower worlds"
            },
            "concealed_light": {
                "type": AspectType.HIDDEN,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.90,
                "description": "The light concealed within darkness"
            },
            "crossing_point": {
                "type": AspectType.TRANSITION,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.89,
                "description": "The crossing point between realms"
            },
            "unmanifest_potential": {
                "type": AspectType.VOID,
                "frequency": self.base_frequency * 1.888,
                "strength": 0.88,
                "description": "The unmanifest potential waiting to be expressed"
            },
            "divine_knowledge": {
                "type": AspectType.KNOWLEDGE,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.87,
                "description": "Knowledge of the divine nature of reality"
            },
            
            # Extended aspects unique to Daath
            "akashic_records": {
                "type": AspectType.KNOWLEDGE,
                "frequency": self.base_frequency * 1.272,
                "strength": 0.86,
                "description": "The cosmic memory of all that is, was, or will be"
            },
            "cosmic_void": {
                "type": AspectType.VOID,
                "frequency": self.base_frequency * 2.618,
                "strength": 0.84,
                "description": "The void between dimensions"
            },
            "hidden_face": {
                "type": AspectType.HIDDEN,
                "frequency": self.base_frequency * 2.236,
                "strength": 0.82,
                "description": "The hidden face of the divine"
            },
            "universal_memory": {
                "type": AspectType.KNOWLEDGE,
                "frequency": self.base_frequency * 1.414,
                "strength": 0.80,
                "description": "The memory matrix of the universe"
            },
            "veil": {
                "type": AspectType.HIDDEN,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.78,
                "description": "The veil separating worlds and states of consciousness"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "kether": {
                "name": "Hidden Connection",
                "strength": 0.92,
                "quality": "divine_knowledge",
                "description": "The hidden connection to the divine source"
            },
            "chokmah": {
                "name": "Wisdom's Shadow",
                "strength": 0.85,
                "quality": "hidden_wisdom",
                "description": "The hidden aspects of divine wisdom"
            },
            "binah": {
                "name": "Form's Gateway",
                "strength": 0.90,
                "quality": "formless_to_form",
                "description": "The gateway from the formless to form"
            },
            "tiphareth": {
                "name": "Hidden Bridge",
                "strength": 0.87,
                "quality": "higher_self_connection",
                "description": "The hidden bridge to the higher self"
            }
        }
    
    def get_primary_aspects(self):
        """
        Get the primary aspects of Daath.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        primary_keys = [
            "hidden_knowledge", "abyss", "gateway", 
            "hidden_sephirah", "threshold"
        ]
        return {k: self.aspects[k] for k in primary_keys if k in self.aspects}
    
    def get_aspect(self, aspect_name):
        """
        Get a specific aspect by name.
        
        Args:
            aspect_name (str): Name of the aspect
            
        Returns:
            dict: The aspect data or None if not found
        """
        return self.aspects.get(aspect_name, None)
    
    def get_all_aspects(self):
        """
        Get all Daath aspects.
        
        Returns:
            dict: Dictionary of all aspects
        """
        return self.aspects
    
    def get_relationship(self, sephirah_name):
        """
        Get relationship with another Sephirah.
        
        Args:
            sephirah_name (str): Name of the other Sephirah
            
        Returns:
            dict: Relationship data or None if not defined
        """
        return self.relationships.get(sephirah_name.lower(), None)
    
    def export_resonant_frequencies(self):
        """
        Export the resonant frequencies of Daath aspects.
        
        Returns:
            list: List of frequency values
        """
        return [aspect["frequency"] for aspect in self.aspects.values()]
    
    def calculate_resonance(self, frequency):
        """
        Calculate how strongly a given frequency resonates with Daath.
        
        Args:
            frequency (float): The frequency to test
            
        Returns:
            float: Resonance value between 0-1
        """
        resonances = []
        
        for aspect in self.aspects.values():
            aspect_freq = aspect["frequency"]
            # Calculate resonance - closer frequencies resonate more strongly
            ratio = min(frequency / aspect_freq, aspect_freq / frequency)
            # Apply harmonic principles (octaves resonate strongly)
            harmonic_factor = 1.0
            if abs(np.log2(frequency / aspect_freq) % 1) < 0.1:
                harmonic_factor = 2.0
                
            resonance = ratio * harmonic_factor * aspect["strength"]
            resonances.append(resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
