"""
Enhanced Chesed Aspects Module

This module defines the specific aspects for Chesed, Mercy - the fourth
Sephirah representing mercy, compassion, and expansion.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.chesed')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    MERCY = auto()       # Mercy/compassion aspects
    EXPANSION = auto()   # Expansion/growth aspects
    LOVE = auto()        # Love aspects
    ABUNDANCE = auto()   # Abundance/prosperity aspects
    ORDER = auto()       # Order/organization aspects
    BENEVOLENCE = auto() # Benevolence aspects
    MAJESTY = auto()     # Majesty/dignity aspects

class ChesedAspects:
    """
    Chesed (Mercy) Aspects
    
    Contains the specific aspects, frequencies, and properties of Chesed,
    the fourth Sephirah representing mercy, compassion, expansion, and love.
    """
    
    def __init__(self):
        """Initialize Chesed aspects"""
        logger.info("Initializing Chesed aspects")
        
        # Base properties (from dictionary)
        self.name = "chesed"
        self.title = "Mercy"
        self.position = 4
        self.pillar = "right"
        self.element = "water/air"
        self.color = "blue"
        self.divine_name = "El"
        self.archangel = "Tzadkiel"
        self.consciousness = "Ruach (higher)"
        self.planetary_correspondence = "Jupiter"
        self.geometric_correspondence = "tetrahedron"
        self.chakra_correspondence = "heart"
        self.body_correspondence = "right arm"
        self.frequency_modifier = 0.85
        self.phi_harmonic_count = 5
        self.harmonic_count = 9
        
        # Base frequency for Chesed (loving frequencies)
        self.base_frequency = 639.0  # Hz (Solfeggio frequency for relationships)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Chesed"""
        aspects = {
            # Primary aspects of Chesed
            "divine_mercy": {
                "type": AspectType.MERCY,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Divine mercy and compassion"
            },
            "expansion": {
                "type": AspectType.EXPANSION,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.98,
                "description": "Expansion and growth"
            },
            "divine_love": {
                "type": AspectType.LOVE,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.97,
                "description": "Divine unconditional love"
            },
            "abundance": {
                "type": AspectType.ABUNDANCE,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.96,
                "description": "Abundance and prosperity"
            },
            "organization": {
                "type": AspectType.ORDER,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.95,
                "description": "Divine order and organization"
            },
            "benevolence": {
                "type": AspectType.BENEVOLENCE,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.94,
                "description": "Benevolence and kindness"
            },
            "majesty": {
                "type": AspectType.MAJESTY,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.93,
                "description": "Divine majesty and dignity"
            },
            "grace": {
                "type": AspectType.MERCY,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.92,
                "description": "Divine grace"
            },
            "forgiveness": {
                "type": AspectType.MERCY,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.91,
                "description": "Divine forgiveness"
            },
            
            # Extended aspects unique to Chesed
            "jupiter_blessing": {
                "type": AspectType.ABUNDANCE,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.89,
                "description": "Jupiter's blessing of expansion"
            },
            "giving": {
                "type": AspectType.MERCY,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.88,
                "description": "Generous giving and charity"
            },
            "justice": {
                "type": AspectType.ORDER,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.86,
                "description": "Merciful justice"
            },
            "healing_love": {
                "type": AspectType.LOVE,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.85,
                "description": "Healing through divine love"
            },
            "prosperity": {
                "type": AspectType.ABUNDANCE,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.83,
                "description": "Divine prosperity and abundance"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "chokmah": {
                "name": "Mercy to Wisdom",
                "strength": 0.85,
                "quality": "benevolence",
                "path_name": "Vav",
                "tarot": "Hierophant",
                "description": "The path of benevolence and loving wisdom"
            },
            "tiphareth": {
                "name": "Mercy to Beauty",
                "strength": 0.85,
                "quality": "healing",
                "path_name": "Yod",
                "tarot": "Hermit",
                "description": "The path of compassion and healing"
            },
            "geburah": {
                "name": "Mercy to Severity",
                "strength": 0.9,
                "quality": "balance",
                "path_name": "Teth",
                "tarot": "Strength",
                "description": "The path of balanced judgment"
            },
            "netzach": {
                "name": "Mercy to Victory",
                "strength": 0.8,
                "quality": "love",
                "path_name": "Kaph",
                "tarot": "Wheel of Fortune",
                "description": "The path of loving success"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Chesed.
        
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
        Get the primary aspects of Chesed.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.90}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Chesed.
        
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
        Calculate how strongly a frequency resonates with Chesed.
        
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
        
        # Add Jupiter resonance (Chesed corresponds to Jupiter)
        jupiter_resonance = self._calculate_jupiter_resonance(frequency)
        resonances.append(jupiter_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_jupiter_resonance(self, frequency):
        """Calculate resonance with Jupiter frequencies"""
        # Jupiter's orbit frequency (translated to audible range)
        jupiter_freq = 183.58  # Hz (derived from orbital period)
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / jupiter_freq, jupiter_freq / frequency)
        
        # Check for harmonic relationships
        harmonic = 1.0
        if abs(np.log2(frequency / jupiter_freq) % 1) < 0.1:
            harmonic = 2.0
            
        return ratio * harmonic * 0.88
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Chesed.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Mercy",
            "strength": 0.95,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Chesed.
        
        Returns:
            float: Stability modifier value
        """
        return self.frequency_modifier
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Chesed.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Chesed.
        
        Returns:
            dict: Dimensional position information
        """
        return {
            "level": self.position,
            "pillar": self.pillar,
            "moral_plane": True  # Part of the moral/ethical triad
        }
    
    def get_platonic_solid(self):
        """
        Get the associated platonic solid for Chesed.
        
        Returns:
            str: Name of platonic solid
        """
        return self.geometric_correspondence
    
    # Chesed-specific methods
    
    def calculate_abundance_flow(self, entity=None):
        """
        Calculate abundance flow for an entity.
        
        This is specific to Chesed as the Sephirah of abundance.
        
        Args:
            entity (dict): Optional entity data
            
        Returns:
            dict: Abundance flow information
        """
        # Base abundance values
        base_flow = {
            "energy": 0.85,
            "resources": 0.80,
            "love": 0.90,
            "opportunities": 0.75
        }
        
        # Modify based on entity properties if provided
        if entity and isinstance(entity, dict):
            # Add love modification
            if "love_capacity" in entity:
                love_factor = 0.7 + 0.3 * entity["love_capacity"]
                base_flow["love"] *= love_factor
                base_flow["energy"] *= 0.9 + 0.1 * love_factor
            
            # Add generosity modification
            if "generosity" in entity:
                generosity_factor = 0.6 + 0.4 * entity["generosity"]
                base_flow["resources"] *= generosity_factor
                base_flow["opportunities"] *= generosity_factor
        
        # Ensure values are within bounds
        for key in base_flow:
            base_flow[key] = min(1.0, max(0.0, base_flow[key]))
        
        return base_flow
    
    def apply_mercy_blessing(self, entity=None, intensity=0.5):
        """
        Apply a mercy blessing to an entity.
        
        Args:
            entity (dict): Optional entity data
            intensity (float): Intensity of blessing (0-1)
            
        Returns:
            dict: Blessing effect information
        """
        # Validate input
        intensity = min(1.0, max(0.0, intensity))
        
        # Calculate blessing effects
        effects = {
            "healing": 0.6 + 0.4 * intensity,
            "forgiveness": 0.7 + 0.3 * intensity,
            "expansion": 0.5 + 0.5 * intensity,
            "duration": 7 * intensity  # Days
        }
        
        # If entity provided, add specific effects
        if entity and isinstance(entity, dict):
            # Store blessing in entity if possible
            if "blessings" in entity and isinstance(entity["blessings"], list):
                entity["blessings"].append({
                    "type": "mercy",
                    "strength": intensity,
                    "effects": effects
                })
        
        return effects



