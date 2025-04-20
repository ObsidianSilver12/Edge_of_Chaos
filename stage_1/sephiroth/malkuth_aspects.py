"""
Enhanced Malkuth Aspects Module

This module defines the specific aspects for Malkuth, Kingdom - the tenth
Sephirah representing physical manifestation, material world, and stability.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.malkuth')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    # Original types expanded for Malkuth's material nature
    UNITY = auto()
    DIVINITY = auto()
    MANIFESTATION = auto()  # Specific to material manifestation
    STABILITY = auto()      # Physical stability
    NATURE = auto()         # Natural forces
    ELEMENTAL = auto()      # Elemental forces
    PHYSICAL = auto()       # Physical properties
    CYCLICAL = auto()       # Natural cycles
    VITALITY = auto()       # Life energy
    CRYSTALLINE = auto()    # Structure and form

class MalkuthAspects:
    """
    Malkuth (Kingdom) Aspects
    
    Contains the specific aspects, frequencies, and properties of Malkuth,
    the tenth Sephirah representing physical manifestation, stability,
    and the material world at the base of the Tree of Life.
    """
    
    def __init__(self):
        """Initialize Malkuth aspects"""
        logger.info("Initializing Malkuth aspects")
        
        # Base properties (from dictionary)
        self.name = "malkuth"
        self.title = "Kingdom"
        self.position = 10
        self.pillar = "middle"
        self.element = "earth"
        self.color = "brown/citrine/olive/black"
        self.divine_name = "Adonai ha-Aretz"
        self.archangel = "Sandalphon"
        self.consciousness = "Nefesh (outer)"
        self.planetary_correspondence = "Earth"
        self.geometric_correspondence = "cube"
        self.chakra_correspondence = "root"
        self.body_correspondence = "whole physical body"
        self.frequency_modifier = 0.55  # Lowest frequency (most dense)
        self.phi_harmonic_count = 1     # Minimal phi-connection
        self.harmonic_count = 4         # Simplest harmonic structure
        
        # Base frequency for Malkuth (lower than Kether, representing density)
        self.base_frequency = 174.0  # Hz (Lowest Solfeggio frequency - for pain relief and grounding)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Malkuth"""
        aspects = {
            # Primary aspects of Malkuth
            "physical_manifestation": {
                "type": AspectType.MANIFESTATION,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "The physical manifestation of divine energy into form"
            },
            "stability": {
                "type": AspectType.STABILITY,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.99,
                "description": "Stability and grounding in the material world"
            },
            "material_world": {
                "type": AspectType.PHYSICAL,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.98,
                "description": "The material world and physical reality"
            },
            "earth_element": {
                "type": AspectType.ELEMENTAL,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.97,
                "description": "The element of earth in its purest form"
            },
            "grounding": {
                "type": AspectType.STABILITY,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.96,
                "description": "Grounding energy connecting spirit to matter"
            },
            "physical_body": {
                "type": AspectType.PHYSICAL,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.95,
                "description": "The physical body as vessel for consciousness"
            },
            "nature_cycles": {
                "type": AspectType.CYCLICAL,
                "frequency": self.base_frequency * 1.272,
                "strength": 0.94,
                "description": "The natural cycles of the Earth and seasons"
            },
            "four_elements": {
                "type": AspectType.ELEMENTAL,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.93,
                "description": "The integration of all four elements in physical form"
            },
            "mineral_kingdom": {
                "type": AspectType.CRYSTALLINE,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.92,
                "description": "The mineral kingdom and crystal structures"
            },
            "vital_life_force": {
                "type": AspectType.VITALITY,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.91,
                "description": "The vital life force animating physical forms"
            },
            
            # Extended aspects unique to Malkuth
            "biodiversity": {
                "type": AspectType.NATURE,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.89,
                "description": "The diverse forms of life in the physical realm"
            },
            "physical_laws": {
                "type": AspectType.PHYSICAL,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.88,
                "description": "The physical laws governing matter and energy"
            },
            "ecological_balance": {
                "type": AspectType.NATURE,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.87,
                "description": "The balance and harmony of natural ecosystems"
            },
            "time_cycles": {
                "type": AspectType.CYCLICAL,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.86,
                "description": "The cycles of time in the physical dimension"
            },
            "earth_magnetism": {
                "type": AspectType.PHYSICAL,
                "frequency": self.base_frequency * 7.83,  # Schumann resonance
                "strength": 0.85,
                "description": "The electromagnetic field of the Earth"
            },
            "sacred_geometry_physical": {
                "type": AspectType.CRYSTALLINE,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.84,
                "description": "Sacred geometry manifested in physical structures"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        # For Malkuth, relationships are primarily upward-oriented
        return {
            "yesod": {
                "name": "Kingdom to Foundation",
                "strength": 0.95,
                "quality": "manifestation",
                "path_name": "Tav",
                "tarot": "World",
                "description": "The path of manifestation from astral to physical"
            },
            "hod": {
                "name": "Kingdom to Glory",
                "strength": 0.85,
                "quality": "intellect",
                "path_name": "Shin",
                "tarot": "Judgment",
                "description": "The path of intellectual understanding of physical forms"
            },
            "netzach": {
                "name": "Kingdom to Victory",
                "strength": 0.85,
                "quality": "nature",
                "path_name": "Qoph",
                "tarot": "Moon",
                "description": "The path of natural forces and emotional connection to Earth"
            },
            "tiphareth": {
                "name": "Kingdom to Beauty",
                "strength": 0.75,
                "quality": "harmony",
                "path_name": "Samekh",
                "tarot": "Temperance",
                "description": "The path of harmony between spirit and matter"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Malkuth.
        
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
        Get the primary aspects of Malkuth.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        # For Malkuth, aspects with strength >= 0.91 are considered primary
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.91}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Malkuth.
        
        Returns:
            dict: Dictionary of secondary aspects
        """
        # For Malkuth, aspects with strength < 0.91 are considered secondary
        return {k: v for k, v in self.aspects.items() if v["strength"] < 0.91}
    
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
        Export the resonant frequencies of Malkuth.
        
        Returns:
            list: List of frequency values
        """
        # Include Earth-specific frequencies
        earth_frequencies = [
            7.83,   # Schumann resonance
            14.3,   # Second Schumann harmonic
            20.8,   # Third Schumann harmonic
            27.3,   # Fourth Schumann harmonic
            33.8    # Fifth Schumann harmonic
        ]
        
        # Add scaled Earth frequencies
        earth_frequencies = [self.base_frequency * (f/7.83) for f in earth_frequencies]
        
        # Add aspect frequencies
        aspect_frequencies = [aspect["frequency"] for aspect in self.aspects.values()]
        
        return aspect_frequencies + earth_frequencies
    
    # Malkuth-specific method for Earth resonance
    def calculate_earth_resonance(self, frequency):
        """
        Calculate how strongly a frequency resonates with Earth's natural frequencies.
        
        Args:
            frequency (float): The frequency to test
            
        Returns:
            float: Earth resonance value between 0-1
        """
        # Earth's key frequencies (Schumann resonances)
        earth_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]
        
        # Calculate resonance with each Earth frequency
        max_resonance = 0.0
        for earth_freq in earth_freqs:
            # Calculate ratio (always < 1.0)
            ratio = min(frequency / earth_freq, earth_freq / frequency)
            
            # Calculate resonance based on harmonic principles
            harmonic_factor = 1.0
            if abs(np.log2(frequency / earth_freq) % 1) < 0.1:
                harmonic_factor = 2.0
            
            resonance = ratio * harmonic_factor * 0.8  # Lower overall resonance for Earth
            max_resonance = max(max_resonance, resonance)
        
        return max_resonance
    
    def calculate_resonance(self, frequency):
        """
        Calculate how strongly a frequency resonates with Malkuth.
        
        Args:
            frequency (float): The frequency to test
            
        Returns:
            float: Resonance value between 0-1
        """
        # Combine aspect resonance and Earth resonance
        aspect_resonances = []
        
        for aspect in self.aspects.values():
            aspect_freq = aspect["frequency"]
            # Calculate resonance - closer frequencies resonate more strongly
            ratio = min(frequency / aspect_freq, aspect_freq / frequency)
            # Apply harmonic principles (octaves resonate strongly)
            harmonic_factor = 1.0
            if abs(np.log2(frequency / aspect_freq) % 1) < 0.1:
                harmonic_factor = 2.0
                
            resonance = ratio * harmonic_factor * aspect["strength"]
            aspect_resonances.append(resonance)
        
        # Get Earth resonance
        earth_resonance = self.calculate_earth_resonance(frequency)
        
        # Combine resonances (stronger influence from Earth for Malkuth)
        aspect_max = max(aspect_resonances) if aspect_resonances else 0.0
        
        # Weight Earth resonance more heavily for Malkuth
        combined_resonance = 0.6 * earth_resonance + 0.4 * aspect_max
        
        return combined_resonance

    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Malkuth.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Manifestation",
            "strength": 0.9,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Malkuth.
        
        Returns:
            float: Stability modifier value
        """
        # Malkuth is very stable in the physical world
        return self.frequency_modifier + 0.2
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Malkuth.
        
        Returns:
            float: Resonance multiplier value
        """
        # Lower phi harmonic resonance for Malkuth
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Malkuth.
        
        Returns:
            dict: Dimensional position information
        """
        return {
            "level": self.position,
            "pillar": self.pillar,
            "depth": 10.0  # Physical manifestation depth
        }
    
    def get_platonic_solid(self):
        """
        Get the associated platonic solid for Malkuth.
        
        Returns:
            str: Name of platonic solid
        """
        # Malkuth is associated with the cube (hexahedron)
        return "hexahedron"
    
    # Malkuth-specific methods
    
    def get_elemental_quaternary(self):
        """
        Get the elemental quaternary of Malkuth.
        
        Returns:
            dict: The four elemental aspects of Malkuth
        """
        return {
            "earth": {
                "color": "citrine", 
                "direction": "north",
                "archangel": "Uriel",
                "strength": 1.0
            },
            "air": {
                "color": "yellow",
                "direction": "east",
                "archangel": "Raphael",
                "strength": 0.7
            },
            "water": {
                "color": "blue",
                "direction": "west", 
                "archangel": "Gabriel",
                "strength": 0.8
            },
            "fire": {
                "color": "red",
                "direction": "south",
                "archangel": "Michael",
                "strength": 0.6
            }
        }
    
    def get_natural_cycles(self):
        """
        Get the natural cycles associated with Malkuth.
        
        Returns:
            dict: Natural cycle information
        """
        return {
            "diurnal": {
                "period": 24.0,  # hours
                "frequency": 1.0/24.0,  # cycles per hour
                "strength": 0.95
            },
            "lunar": {
                "period": 29.53,  # days
                "frequency": 1.0/29.53,  # cycles per day
                "strength": 0.9
            },
            "seasonal": {
                "period": 365.25,  # days
                "frequency": 1.0/365.25,  # cycles per day
                "strength": 0.85
            },
            "biorhythm": {
                "physical": {
                    "period": 23.0,  # days
                    "frequency": 1.0/23.0  # cycles per day
                },
                "emotional": {
                    "period": 28.0,  # days
                    "frequency": 1.0/28.0  # cycles per day
                },
                "intellectual": {
                    "period": 33.0,  # days
                    "frequency": 1.0/33.0  # cycles per day
                }
            }
        }
    
    def calculate_physical_resonance(self, physical_entity):
        """
        Calculate how strongly a physical entity resonates with Malkuth.
        
        Args:
            physical_entity (dict): Entity with physical properties
            
        Returns:
            float: Physical resonance value between 0-1
        """
        # Sample implementation - would depend on physical_entity structure
        if not isinstance(physical_entity, dict):
            return 0.0
            
        resonance_factors = []
        
        # Element resonance
        if "element" in physical_entity:
            element = physical_entity["element"].lower()
            element_strength = {
                "earth": 1.0,
                "water": 0.8,
                "air": 0.6,
                "fire": 0.7
            }.get(element, 0.0)
            resonance_factors.append(element_strength)
        
        # Form resonance
        if "form" in physical_entity:
            form = physical_entity["form"].lower()
            form_strength = {
                "crystal": 0.95,
                "organic": 0.9,
                "liquid": 0.75,
                "gas": 0.5,
                "plasma": 0.3
            }.get(form, 0.6)
            resonance_factors.append(form_strength)
        
        # Density resonance
        if "density" in physical_entity:
            density = physical_entity["density"]
            # Normalize to 0-1 (assuming density 0-10 scale)
            density_normalized = min(1.0, max(0.0, density / 10.0))
            resonance_factors.append(density_normalized)
        
        # Calculate average resonance if we have factors
        if resonance_factors:
            return sum(resonance_factors) / len(resonance_factors)
        else:
            return 0.5  # Default medium resonance


