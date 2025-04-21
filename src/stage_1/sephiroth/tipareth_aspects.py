"""
Enhanced Tiphareth Aspects Module

This module defines the specific aspects for Tiphareth, Beauty - the sixth
Sephirah representing harmony, balance, beauty, and the central solar point.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.tiphareth')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    HARMONY = auto()    # Harmonizing aspects
    BEAUTY = auto()     # Beauty aspects
    BALANCE = auto()    # Balance/equilibrium aspects  
    SOLAR = auto()      # Solar/light aspects
    HEALING = auto()    # Healing/restoration aspects
    TRANSFORMATION = auto()  # Transformation aspects
    INTEGRATION = auto()     # Integration aspects
    CONSCIOUSNESS = auto()   # Consciousness aspects

class TipharethAspects:
    """
    Tiphareth (Beauty) Aspects
    
    Contains the specific aspects, frequencies, and properties of Tiphareth,
    the sixth Sephirah representing beauty, harmony, balance, and the central
    point of integration on the Tree of Life.
    """
    
    def __init__(self):
        """Initialize Tiphareth aspects"""
        logger.info("Initializing Tiphareth aspects")
        
        # Base properties (from dictionary)
        self.name = "tiphareth"
        self.title = "Beauty"
        self.position = 6
        self.pillar = "middle"
        self.element = "air/fire"
        self.color = "gold/yellow"
        self.divine_name = "YHVH Eloah ve-Daath"
        self.archangel = "Raphael"
        self.consciousness = "Ruach (center)"
        self.planetary_correspondence = "Sun"
        self.geometric_correspondence = "hexahedron"
        self.chakra_correspondence = "heart"
        self.body_correspondence = "heart/chest"
        self.frequency_modifier = 0.75  # Central balancing frequency
        self.phi_harmonic_count = 6     # Strong phi-resonance
        self.harmonic_count = 10        # Rich harmonics
        
        # Base frequency (solar frequency)
        self.base_frequency = 528.0  # Hz (Solfeggio frequency for transformation and miracles)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Tiphareth"""
        aspects = {
            # Primary aspects of Tiphareth
            "harmony": {
                "type": AspectType.HARMONY,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Harmony and resonance throughout the Tree"
            },
            "beauty": {
                "type": AspectType.BEAUTY,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.99,
                "description": "Divine beauty and perfection of form"
            },
            "balance": {
                "type": AspectType.BALANCE,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.98,
                "description": "Perfect balance between opposing forces"
            },
            "solar_consciousness": {
                "type": AspectType.CONSCIOUSNESS,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.97,
                "description": "Solar consciousness and illumination"
            },
            "integration": {
                "type": AspectType.INTEGRATION,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.96,
                "description": "Integration of higher and lower aspects"
            },
            "heart_center": {
                "type": AspectType.HEALING,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.95,
                "description": "The spiritual and energetic heart center"
            },
            "transformation": {
                "type": AspectType.TRANSFORMATION,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.94,
                "description": "Transformative spiritual processes"
            },
            "healing": {
                "type": AspectType.HEALING,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.93,
                "description": "Divine healing and restoration"
            },
            "higher_self": {
                "type": AspectType.CONSCIOUSNESS,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.92,
                "description": "Connection to the higher self and divine identity"
            },
            "sacred_identity": {
                "type": AspectType.CONSCIOUSNESS,
                "frequency": self.base_frequency * 1.272,
                "strength": 0.91,
                "description": "The sacred identity and divine name"
            },
            
            # Extended aspects unique to Tiphareth
            "christ_consciousness": {
                "type": AspectType.CONSCIOUSNESS,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.90,
                "description": "Christ consciousness and divine love"
            },
            "resurrection": {
                "type": AspectType.TRANSFORMATION,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.89,
                "description": "Death and rebirth of the spiritual self"
            },
            "solar_radiance": {
                "type": AspectType.SOLAR,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.88,
                "description": "Radiant solar energy and vitality"
            },
            "sacrifice": {
                "type": AspectType.TRANSFORMATION,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.87,
                "description": "Sacred sacrifice for spiritual transformation"
            },
            "equilibrium": {
                "type": AspectType.BALANCE,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.86,
                "description": "Perfect equilibrium of all forces"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        # Tiphareth has relationships in all directions
        return {
            "kether": {
                "name": "Beauty to Crown",
                "strength": 0.90,
                "quality": "divine_illumination",
                "path_name": "Gimel",
                "tarot": "High Priestess",
                "description": "The path of divine illumination from the crown"
            },
            "chokmah": {
                "name": "Beauty to Wisdom",
                "strength": 0.85,
                "quality": "consciousness",
                "path_name": "Heh",
                "tarot": "Emperor",
                "description": "The path of wisdom and consciousness"
            },
            "binah": {
                "name": "Beauty to Understanding",
                "strength": 0.85,
                "quality": "understanding",
                "path_name": "Zain",
                "tarot": "Lovers",
                "description": "The path of understanding and formation"
            },
            "chesed": {
                "name": "Beauty to Mercy",
                "strength": 0.85,
                "quality": "healing",
                "path_name": "Yod",
                "tarot": "Hermit",
                "description": "The path of compassion and healing"
            },
            "geburah": {
                "name": "Beauty to Severity",
                "strength": 0.85,
                "quality": "purification",
                "path_name": "Lamed",
                "tarot": "Justice",
                "description": "The path of strength and purification"
            },
            "netzach": {
                "name": "Beauty to Victory",
                "strength": 0.80,
                "quality": "beauty",
                "path_name": "Nun",
                "tarot": "Death",
                "description": "The path of beauty and harmony in emotion"
            },
            "hod": {
                "name": "Beauty to Glory",
                "strength": 0.80,
                "quality": "revelation",
                "path_name": "Ayin",
                "tarot": "Devil",
                "description": "The path of truth and revelation"
            },
            "yesod": {
                "name": "Beauty to Foundation",
                "strength": 0.80,
                "quality": "reflection",
                "path_name": "Samekh",
                "tarot": "Temperance",
                "description": "The path of reflection and balance"
            },
            "malkuth": {
                "name": "Beauty to Kingdom",
                "strength": 0.75,
                "quality": "incarnation",
                "path_name": "Resh",
                "tarot": "Sun",
                "description": "The path of manifestation and incarnation"
            },
            "daath": {
                "name": "Beauty to Knowledge",
                "strength": 0.70,
                "quality": "gnosis",
                "description": "The hidden path of knowledge and spiritual awakening"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Tiphareth.
        
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
        Get the primary aspects of Tiphareth.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        # For Tiphareth, aspects with strength >= 0.91 are considered primary
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.91}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Tiphareth.
        
        Returns:
            dict: Dictionary of secondary aspects
        """
        # For Tiphareth, aspects with strength < 0.91 are considered secondary
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
        Export the resonant frequencies of Tiphareth.
        
        Returns:
            list: List of frequency values
        """
        frequencies = [aspect["frequency"] for aspect in self.aspects.values()]
        
        # Add solar frequencies
        solar_frequencies = [
            528.0,  # Transformation and DNA repair
            639.0,  # Connection and relationships
            741.0,  # Expression and solutions
            852.0,  # Returning to spiritual order
            396.0,  # Liberation from fear
            417.0   # Facilitating change
        ]
        
        return frequencies + solar_frequencies
    
    def calculate_resonance(self, frequency):
        """
        Calculate how strongly a frequency resonates with Tiphareth.
        
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
        
        # Add solar resonance (Tiphareth corresponds to the Sun)
        solar_resonance = self._calculate_solar_resonance(frequency)
        resonances.append(solar_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_solar_resonance(self, frequency):
        """Calculate resonance with solar frequencies"""
        # The Sun's fundamental "tone" (translated to audible frequency)
        solar_freq = 126.22  # Hz
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / solar_freq, solar_freq / frequency)
        
        # Check for octave relationships
        harmonic = 1.0
        if abs(np.log2(frequency / solar_freq) % 1) < 0.1:
            harmonic = 2.0
            
        # Solar resonance is amplified in Tiphareth
        return ratio * harmonic * 0.95
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Tiphareth.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Beauty and Harmony",
            "strength": 1.0,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Tiphareth.
        
        Returns:
            float: Stability modifier value
        """
        return self.frequency_modifier
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Tiphareth.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Tiphareth.
        
        Returns:
            dict: Dimensional position information
        """
        return {
            "level": self.position,
            "pillar": self.pillar,
            "center": True  # Tiphareth is at the center of the Tree
        }
    
    def get_platonic_solid(self):
        """
        Get the associated platonic solid for Tiphareth.
        
        Returns:
            str: Name of platonic solid
        """
        return self.geometric_correspondence
    
    # Tiphareth-specific methods
    
    def calculate_heart_resonance(self, entity):
        """
        Calculate heart-centered resonance for an entity.
        
        This is specific to Tiphareth as the heart center of the Tree.
        
        Args:
            entity (dict): Entity with heart-related properties
            
        Returns:
            float: Heart resonance value between 0-1
        """
        # Sample implementation - would depend on entity structure
        if not isinstance(entity, dict):
            return 0.0
            
        resonance_factors = []
        
        # Love presence
        if "love_quotient" in entity:
            love = entity["love_quotient"]
            # Normalize to 0-1
            love_normalized = min(1.0, max(0.0, love))
            resonance_factors.append(love_normalized)
        
        # Harmony
        if "harmony" in entity:
            harmony = entity["harmony"]
            # Normalize to 0-1
            harmony_normalized = min(1.0, max(0.0, harmony))
            resonance_factors.append(harmony_normalized)
        
        # Integration
        if "integration" in entity:
            integration = entity["integration"]
            # Normalize to 0-1
            integration_normalized = min(1.0, max(0.0, integration))
            resonance_factors.append(integration_normalized)
        
        # Calculate average resonance if we have factors
        if resonance_factors:
            return sum(resonance_factors) / len(resonance_factors)
        else:
            return 0.6  # Default medium-high resonance for Tiphareth
    
    def get_solar_calendar(self):
        """
        Get the solar calendar associations for Tiphareth.
        
        Returns:
            dict: Solar calendar information
        """
        return {
            "solstices": {
                "summer": {
                    "significance": "Maximum light",
                    "resonance": 1.0
                },
                "winter": {
                    "significance": "Rebirth of light",
                    "resonance": 0.9
                }
            },
            "equinoxes": {
                "spring": {
                    "significance": "Balance of light/dark with ascending light",
                    "resonance": 0.95
                },
                "autumn": {
                    "significance": "Balance of light/dark with descending light",
                    "resonance": 0.85
                }
            },
            "solar_hours": {
                "dawn": 0.8,
                "noon": 1.0,
                "dusk": 0.8,
                "midnight": 0.6
            }
        }