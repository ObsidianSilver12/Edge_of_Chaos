"""
Enhanced Binah Aspects Module

This module defines the specific aspects for Binah, Understanding - the third
Sephirah representing understanding, receptivity, and form.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.binah')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    UNDERSTANDING = auto()  # Understanding aspects
    FEMININE = auto()       # Feminine principle aspects
    FORM = auto()           # Form/structure aspects
    RECEPTIVITY = auto()    # Receptive aspects
    LIMITATION = auto()     # Limiting/defining aspects
    TIME = auto()           # Temporal aspects
    PATTERN = auto()        # Pattern aspects

class BinahAspects:
    """
    Binah (Understanding) Aspects
    
    Contains the specific aspects, frequencies, and properties of Binah,
    the third Sephirah representing understanding, receptivity, and form.
    """
    
    def __init__(self):
        """Initialize Binah aspects"""
        logger.info("Initializing Binah aspects")
        
        # Base properties (from dictionary)
        self.name = "binah"
        self.title = "Understanding"
        self.position = 3
        self.pillar = "left"
        self.element = "water"
        self.color = "black"
        self.divine_name = "YHVH Elohim"
        self.archangel = "Tzaphkiel"
        self.consciousness = "Neshamah"
        self.planetary_correspondence = "Saturn"
        self.geometric_correspondence = "triangle"
        self.chakra_correspondence = "third eye"
        self.body_correspondence = "left brain"
        self.frequency_modifier = 0.9
        self.phi_harmonic_count = 6
        self.harmonic_count = 10
        
        # Base frequency for Binah (receptive understanding)
        self.base_frequency = 417.0  # Hz (Solfeggio frequency for change/transformation)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Binah"""
        aspects = {
            # Primary aspects of Binah
            "divine_understanding": {
                "type": AspectType.UNDERSTANDING,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Divine understanding and comprehension"
            },
            "form_creation": {
                "type": AspectType.FORM,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.98,
                "description": "Creation of form and structure"
            },
            "feminine_principle": {
                "type": AspectType.FEMININE,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.97,
                "description": "The feminine/yin principle"
            },
            "divine_receptivity": {
                "type": AspectType.RECEPTIVITY,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.96,
                "description": "Divine receptivity and matrix"
            },
            "limitation": {
                "type": AspectType.LIMITATION,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.95,
                "description": "Limitation and restriction for manifestation"
            },
            "pattern_recognition": {
                "type": AspectType.PATTERN,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.94,
                "description": "Recognition of divine patterns"
            },
            "divine_womb": {
                "type": AspectType.FEMININE,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.93,
                "description": "The divine womb of creation"
            },
            "time_structure": {
                "type": AspectType.TIME,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.92,
                "description": "The structure of time and space"
            },
            "crystallization": {
                "type": AspectType.FORM,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.91,
                "description": "Crystallization of divine energy into form"
            },
            
            # Extended aspects unique to Binah
            "divine_mother": {
                "type": AspectType.FEMININE,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.89,
                "description": "The divine mother principle"
            },
            "karmic_structure": {
                "type": AspectType.PATTERN,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.87,
                "description": "The structure of karmic patterns"
            },
            "divine_darkness": {
                "type": AspectType.FEMININE,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.86,
                "description": "The divine darkness of creation"
            },
            "sacred_geometry": {
                "type": AspectType.FORM,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.85,
                "description": "Sacred geometric patterns and matrices"
            },
            "comprehension": {
                "type": AspectType.UNDERSTANDING,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.83,
                "description": "Deep comprehension of divine principles"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "kether": {
                "name": "Understanding to Crown",
                "strength": 0.95,
                "quality": "potential",
                "path_name": "Aleph",
                "tarot": "Fool",
                "description": "The path of potential and pure consciousness"
            },
            "chokmah": {
                "name": "Understanding to Wisdom",
                "strength": 0.95,
                "quality": "creation",
                "path_name": "Daleth",
                "tarot": "Empress",
                "description": "The path of creation through divine masculine and feminine"
            },
            "tiphareth": {
                "name": "Understanding to Beauty",
                "strength": 0.85,
                "quality": "understanding",
                "path_name": "Zain",
                "tarot": "Lovers",
                "description": "The path of understanding and formation"
            },
            "geburah": {
                "name": "Understanding to Severity",
                "strength": 0.85,
                "quality": "restriction",
                "path_name": "Zayin",
                "tarot": "Lovers",
                "description": "The path of restriction and discipline"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Binah.
        
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
        Get the primary aspects of Binah.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.90}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Binah.
        
        Returns:
            dict: Dictionary of secondary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] < 0.90}
    
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
    
    def calculate_resonance(self, frequency):
        """
        Calculate how strongly a frequency resonates with Binah.
        
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
        
        # Add Saturn resonance (Binah corresponds to Saturn)
        saturn_resonance = self._calculate_saturn_resonance(frequency)
        resonances.append(saturn_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_saturn_resonance(self, frequency):
        """Calculate resonance with Saturn frequencies"""
        # Saturn's orbit frequency (translated to audible range)
        saturn_freq = 147.85  # Hz (derived from orbital period)
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / saturn_freq, saturn_freq / frequency)
        
        # Check for harmonic relationships
        harmonic = 1.0
        if abs(np.log2(frequency / saturn_freq) % 1) < 0.1:
            harmonic = 2.0
            
        return ratio * harmonic * 0.85
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Binah.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Understanding",
            "strength": 0.95,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Binah.
        
        Returns:
            float: Stability modifier value
        """
        # Binah is stable due to its form-giving nature
        return self.frequency_modifier + 0.05
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Binah.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Binah.
        
        Returns:
            dict: Dimensional position information
        """
        return {
            "level": self.position,
            "pillar": self.pillar,
            "supernal": True  # Part of the supernal triad
        }
    
    def get_platonic_solid(self):
        """
        Get the associated platonic solid for Binah.
        
        Returns:
            str: Name of platonic solid
        """
        # Binah is associated with structure
        return "octahedron"
    
    # Binah-specific methods
    
    def calculate_form_matrix(self, pattern=None):
        """
        Calculate a form matrix for a pattern.
        
        This is specific to Binah as the form-giving Sephirah.
        
        Args:
            pattern (dict): Optional pattern data
            
        Returns:
            dict: Form matrix information
        """
        # Implementation would vary based on pattern data
        dimensions = 3
        stability = 0.85
        
        if pattern and isinstance(pattern, dict):
            # Add pattern complexity modification
            if "complexity" in pattern:
                dimensions = max(3, min(7, 3 + pattern["complexity"] * 4))
            
            # Add pattern stability modification
            if "structure" in pattern:
                stability = 0.7 + 0.3 * pattern["structure"]
        
        # Create matrix template based on dimensions and stability
        return {
            "dimensions": dimensions,
            "stability": stability,
            "receptivity": 0.9,
                        "form_quality": stability * 0.9,
            "pattern_integrity": 0.7 + 0.3 * stability
        }
    
    def get_time_cycles(self):
        """
        Get the time cycles associated with Binah.
        
        Returns:
            dict: Time cycle information
        """
        return {
            "cosmic": {
                "period": 25920.0,  # Great Year (Precession of Equinoxes)
                "frequency": 1.0/25920.0,
                "influence": 0.95
            },
            "saturn": {
                "period": 29.5,  # Years
                "frequency": 1.0/29.5,
                "influence": 0.9
            },
            "karmic": {
                "period": 7.0,  # Years
                "frequency": 1.0/7.0,
                "influence": 0.85
            },
            "gates": [
                {"name": "Birth", "transition": 0.2},
                {"name": "Maturity", "transition": 0.4},
                {"name": "Wisdom", "transition": 0.7},
                {"name": "Death", "transition": 0.9}
            ]
        }


