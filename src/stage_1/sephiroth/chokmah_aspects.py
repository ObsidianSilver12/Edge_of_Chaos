"""
Enhanced Chokmah Aspects Module

This module defines the specific aspects for Chokmah, Wisdom - the second
Sephirah representing wisdom, revelation, and masculine dynamic force.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.chokmah')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    WISDOM = auto()      # Wisdom/insight aspects
    DYNAMISM = auto()    # Dynamic/active force aspects
    MASCULINE = auto()   # Masculine principle aspects
    REVELATION = auto()  # Revelation aspects
    INSPIRATION = auto() # Inspiration aspects
    FORCE = auto()       # Force/energy aspects
    COSMIC = auto()      # Cosmic energy aspects

class ChokmahAspects:
    """
    Chokmah (Wisdom) Aspects
    
    Contains the specific aspects, frequencies, and properties of Chokmah,
    the second Sephirah representing wisdom, revelation, and dynamic force.
    """
    
    def __init__(self):
        """Initialize Chokmah aspects"""
        logger.info("Initializing Chokmah aspects")
        
        # Base properties (from dictionary)
        self.name = "chokmah"
        self.title = "Wisdom"
        self.position = 2
        self.pillar = "right"
        self.element = "fire"
        self.color = "grey"
        self.divine_name = "Yah"
        self.archangel = "Raziel"
        self.consciousness = "Chiah"
        self.planetary_correspondence = "Uranus / Zodiac"
        self.geometric_correspondence = "line"
        self.chakra_correspondence = "third eye"
        self.body_correspondence = "right brain"
        self.frequency_modifier = 0.95  # Very high frequency
        self.phi_harmonic_count = 6
        self.harmonic_count = 10
        
        # Base frequency for Chokmah (high frequency for wisdom)
        self.base_frequency = 741.0  # Hz (Solfeggio frequency for expression and solutions)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Chokmah"""
        aspects = {
            # Primary aspects of Chokmah
            "divine_wisdom": {
                "type": AspectType.WISDOM,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Divine wisdom and insight"
            },
            "dynamic_force": {
                "type": AspectType.DYNAMISM,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.98,
                "description": "Dynamic outward-moving force"
            },
            "masculine_principle": {
                "type": AspectType.MASCULINE,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.96,
                "description": "The masculine/yang principle"
            },
            "divine_revelation": {
                "type": AspectType.REVELATION,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.97,
                "description": "Divine revelation and vision"
            },
            "cosmic_energy": {
                "type": AspectType.COSMIC,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.95,
                "description": "Pure cosmic energy and power"
            },
            "primordial_force": {
                "type": AspectType.FORCE,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.94,
                "description": "Primordial creative force"
            },
            "divine_will": {
                "type": AspectType.FORCE,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.93,
                "description": "Expression of divine will and purpose"
            },
            "universal_law": {
                "type": AspectType.WISDOM,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.92,
                "description": "Universal laws and principles"
            },
            "inspiration": {
                "type": AspectType.INSPIRATION,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.91,
                "description": "Divine inspiration and insight"
            },
            
            # Extended aspects unique to Chokmah
            "divine_father": {
                "type": AspectType.MASCULINE,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.88,
                "description": "The divine father principle"
            },
            "zodiacal_wisdom": {
                "type": AspectType.COSMIC,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.87,
                "description": "Zodiacal patterns and cosmic order"
            },
            "dual_polarity": {
                "type": AspectType.DYNAMISM,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.86,
                "description": "Creation of duality and polarity"
            },
            "lightning_flash": {
                "type": AspectType.FORCE,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.85,
                "description": "The lightning flash of creation"
            },
            "creative_vision": {
                "type": AspectType.REVELATION,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.83,
                "description": "Creative vision and foresight"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "kether": {
                "name": "Wisdom to Crown",
                "strength": 0.95,
                "quality": "emanation",
                "path_name": "Beth",
                "tarot": "Magician",
                "description": "The path of pure emanation from the divine source"
            },
            "binah": {
                "name": "Wisdom to Understanding",
                "strength": 0.95,
                "quality": "creation",
                "path_name": "Daleth",
                "tarot": "Empress",
                "description": "The path of creation through divine masculine and feminine"
            },
            "tiphareth": {
                "name": "Wisdom to Beauty",
                "strength": 0.85,
                "quality": "consciousness",
                "path_name": "Heh",
                "tarot": "Emperor",
                "description": "The path of consciousness and divine order"
            },
            "chesed": {
                "name": "Wisdom to Mercy",
                "strength": 0.85,
                "quality": "benevolence",
                "path_name": "Vav",
                "tarot": "Hierophant",
                "description": "The path of benevolence and loving power"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Chokmah.
        
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
        Get the primary aspects of Chokmah.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.90}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Chokmah.
        
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
        Calculate how strongly a frequency resonates with Chokmah.
        
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
        
        # Add Uranus resonance (Chokmah corresponds to Uranus)
        uranus_resonance = self._calculate_uranus_resonance(frequency)
        resonances.append(uranus_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_uranus_resonance(self, frequency):
        """Calculate resonance with Uranus frequencies"""
        # Uranus's orbit frequency (translated to audible range)
        uranus_freq = 207.36  # Hz (derived from orbital period)
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / uranus_freq, uranus_freq / frequency)
        
        # Check for harmonic relationships
        harmonic = 1.0
        if abs(np.log2(frequency / uranus_freq) % 1) < 0.1:
            harmonic = 2.0
            
        return ratio * harmonic * 0.9
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Chokmah.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Wisdom",
            "strength": 0.95,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Chokmah.
        
        Returns:
            float: Stability modifier value
        """
        # Chokmah is dynamic, less stable than Kether
        return self.frequency_modifier - 0.05
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Chokmah.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Chokmah.
        
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
        Get the associated platonic solid for Chokmah.
        
        Returns:
            str: Name of platonic solid
        """
        # Chokmah is associated with the octahedron (dynamic)
        return "octahedron"
    
    # Chokmah-specific methods
    
    def get_wisdom_revelation(self, subject=None):
        """
        Generate wisdom revelation on a subject.
        
        This is specific to Chokmah as the Sephirah of Wisdom.
        
        Args:
            subject (str): Optional subject for the revelation
            
        Returns:
            dict: Wisdom revelation information
        """
        # Implementation would depend on specific requirements
        wisdom_types = ["insight", "foresight", "cosmic_pattern", "principle", "law"]
        
        return {
            "type": np.random.choice(wisdom_types),
            "strength": 0.7 + 0.3 * np.random.random(),
            "subject": subject or "universal",
            "clarity": 0.6 + 0.4 * np.random.random(),
            "insight": "The revelation of divine patterns manifesting through cosmic order"
        }
    
    def calculate_dynamic_force(self, entity=None):
        """
        Calculate dynamic force value for an entity.
        
        This is specific to Chokmah as the Sephirah of dynamic force.
        
        Args:
            entity (dict): Optional entity data
            
        Returns:
            float: Dynamic force value between 0-1
        """
        # Base dynamic force
        base_force = 0.8
        
        # Modify based on entity properties if provided
        if entity and isinstance(entity, dict):
            # Add masculine energy modification
            if "masculine_energy" in entity:
                base_force *= 0.7 + 0.3 * entity["masculine_energy"]
            
            # Add wisdom modification
            if "wisdom" in entity:
                base_force *= 0.8 + 0.2 * entity["wisdom"]
        
        return min(1.0, base_force)


