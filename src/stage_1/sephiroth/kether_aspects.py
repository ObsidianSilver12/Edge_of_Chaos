"""
Enhanced Kether Aspects Module

This module defines the specific aspects for Kether, Crown - the first
Sephirah representing divine unity, pure being, and the highest point of the Tree.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.kether')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    UNITY = auto()
    DIVINITY = auto()
    WILL = auto()
    BEING = auto()
    LIGHT = auto()
    TRANSCENDENCE = auto()
    PRIMARY = 1
    CREATOR = 2
    DIMENSIONAL = 3
    ELEMENTAL = 4
    CHAKRA = 5
    HARMONIC = 6
    YIN_YANG = 7
    STRUCTURAL = 8
    CONSCIOUSNESS = 9

class KetherAspects:
    """
    Kether (Crown) Aspects
    
    Contains the specific aspects, frequencies, and properties of Kether,
    the first Sephirah representing divine unity, pure being, divine will,
    and the highest point of emanation on the Tree of Life.
    """
    
    def __init__(self):
        """Initialize Kether aspects"""
        logger.info("Initializing Kether aspects")
        
        # Base properties (from dictionary)
        self.name = "kether"
        self.title = "Crown"
        self.position = 1
        self.pillar = "middle"
        self.element = "quintessence"
        self.color = "pure white"
        self.divine_name = "Eheieh"
        self.archangel = "Metatron"
        self.consciousness = "Yechidah"
        self.planetary_correspondence = None
        self.geometric_correspondence = "point"
        self.chakra_correspondence = "crown"
        self.body_correspondence = "crown of head"
        self.frequency_modifier = 1.0  # Highest frequency (pure source)
        self.phi_harmonic_count = 7    # Strongest phi connection
        self.harmonic_count = 12       # Most complex harmonics
        
        # Base frequency (from original aspect file)
        self.base_frequency = 963.0  # Hz (Base creator frequency * 2.23)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Kether"""
        aspects = {
            # Primary aspects of Kether
            "divine_unity": {
                "type": AspectType.UNITY,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "The perfect unity beyond all division"
            },
            "divine_will": {
                "type": AspectType.WILL,
                "frequency": self.base_frequency * 1.207,
                "strength": 0.99,
                "description": "The primordial will of the divine"
            },
            "pure_being": {
                "type": AspectType.BEING,
                "frequency": self.base_frequency * 1.382,
                "strength": 0.98,
                "description": "Pure being beyond all qualities"
            },
            "crown_consciousness": {
                "type": AspectType.TRANSCENDENCE,
                "frequency": self.base_frequency * 0.786,
                "strength": 0.97,
                "description": "The highest level of divine consciousness"
            },
            "limitless_light": {
                "type": AspectType.LIGHT,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.96,
                "description": "The limitless light (Ain Soph Aur)"
            },
            "primordial_point": {
                "type": AspectType.UNITY,
                "frequency": self.base_frequency * 2.0,
                "strength": 0.95,
                "description": "The primordial point from which all emanates"
            },
            "divine_singularity": {
                "type": AspectType.UNITY,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.94,
                "description": "The singular source of all manifestation"
            },
            "transcendence": {
                "type": AspectType.TRANSCENDENCE,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.93,
                "description": "The transcendent aspect beyond all limitation"
            },
            "absolute_unity": {
                "type": AspectType.UNITY,
                "frequency": self.base_frequency * 1.888,
                "strength": 0.92,
                "description": "The absolute unity prior to all division"
            },
            "divine_essence": {
                "type": AspectType.DIVINITY,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.91,
                "description": "The essential nature of divinity itself"
            },
            
            # Extended aspects unique to Kether
            "cosmic_breath": {
                "type": AspectType.BEING,
                "frequency": self.base_frequency * 1.272,
                "strength": 0.90,
                "description": "The cosmic breath that initiates creation"
            },
            "hidden_face": {
                "type": AspectType.TRANSCENDENCE,
                "frequency": self.base_frequency * 2.618,
                "strength": 0.89,
                "description": "The hidden face of divinity"
            },
            "ancient_of_days": {
                "type": AspectType.DIVINITY,
                "frequency": self.base_frequency * 2.236,
                "strength": 0.88,
                "description": "The ancient of days aspect of divinity"
            },
            "divine_potential": {
                "type": AspectType.BEING,
                "frequency": self.base_frequency * 1.414,
                "strength": 0.87,
                "description": "The divine potential containing all possibilities"
            },
            "ineffable_presence": {
                "type": AspectType.DIVINITY,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.86,
                "description": "The ineffable presence beyond comprehension"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "chokmah": {
                "name": "Crown to Wisdom",
                "strength": 0.95,
                "quality": "divine_emanation",
                "description": "The emanation of divine wisdom from unity"
            },
            "binah": {
                "name": "Crown to Understanding",
                "strength": 0.95,
                "quality": "divine_form",
                "description": "The emanation of divine form from unity"
            },
                "tiphareth": {
                    "name": "Crown to Beauty",
                    "strength": 0.90,
                    "quality": "divine_illumination",
                    "path_name": "Gimel",
                    "tarot": "High Priestess",
                    "description": "The direct illumination of the divine light to the central point"
                },
                "daath": {
                    "name": "Crown to Knowledge",
                    "strength": 0.93,
                    "quality": "divine_gnosis",
                    "description": "The hidden connection to the abyss of knowledge"
                }
            }
    
    def get_metadata(self):
        """
        Get the metadata for Kether.
        
        Returns:
            dict: Dictionary of metadata
        """
        return {
            "name": self.name,
            "title": self.title,
            "position": self.position,
            "pillar": self.pillar,
            "element": self.element,
            "color": self.color,
            "divine_name": self.divine_name,
            "archangel": self.archangel,
            "consciousness": self.consciousness,
            "planetary_correspondence": self.planetary_correspondence,
            "geometric_correspondence": self.geometric_correspondence,
            "chakra_correspondence": self.chakra_correspondence,
            "body_correspondence": self.body_correspondence,
            "frequency_modifier": self.frequency_modifier,
            "phi_harmonic_count": self.phi_harmonic_count,
            "harmonic_count": self.harmonic_count
        }
    
    def get_primary_aspects(self):
        """
        Get the primary aspects of Kether.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        # For Kether, aspects with strength >= 0.93 are considered primary
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.93}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Kether.
        
        Returns:
            dict: Dictionary of secondary aspects
        """
        # For Kether, aspects with strength < 0.93 are considered secondary
        return {k: v for k, v in self.aspects.items() if v["strength"] < 0.93}
    
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
        Get all aspects.
        
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
        Export the resonant frequencies of Kether.
        
        Returns:
            list: List of frequency values
        """
        frequencies = [aspect["frequency"] for aspect in self.aspects.values()]
        
        # Add base and divine frequencies
        frequencies.extend([
            self.base_frequency,
            self.base_frequency * 1.618,  # Golden ratio
            self.base_frequency * 2.0,    # Octave
            432.0  # Creator base
        ])
        
        return frequencies
    
    def calculate_resonance(self, frequency):
        """
        Calculate how strongly a given frequency resonates with Kether.
        
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

    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Kether.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Will",
            "strength": 1.0,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Kether.
        
        Returns:
            float: Stability modifier value
        """
        return self.frequency_modifier
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Kether.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Kether.
        
        Returns:
            dict: Dimensional position information
        """
        return {
            "level": self.position,
            "pillar": self.pillar
        }
    
    def get_platonic_solid(self):
        """
        Get the associated platonic solid for Kether.
        
        Returns:
            str: Name of platonic solid or None
        """
        return self.geometric_correspondence
