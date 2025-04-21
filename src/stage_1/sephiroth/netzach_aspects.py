"""
Enhanced Netzach Aspects Module

This module defines the specific aspects for Netzach, Victory - the seventh
Sephirah representing victory, emotion, and natural forces.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.netzach')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    VICTORY = auto()     # Victory aspects
    EMOTION = auto()     # Emotional aspects
    NATURE = auto()      # Nature/natural forces aspects
    BEAUTY = auto()      # Beauty aspects
    LOVE = auto()        # Love aspects
    DESIRE = auto()      # Desire aspects
    PASSION = auto()     # Passion aspects
    ART = auto()         # Artistic aspects

class NetzachAspects:
    """
    Netzach (Victory) Aspects
    
    Contains the specific aspects, frequencies, and properties of Netzach,
    the seventh Sephirah representing victory, emotion, and natural forces.
    """
    
    def __init__(self):
        """Initialize Netzach aspects"""
        logger.info("Initializing Netzach aspects")
        
        # Base properties (from dictionary)
        self.name = "netzach"
        self.title = "Victory"
        self.position = 7
        self.pillar = "right"
        self.element = "water/fire"
        self.color = "green"
        self.divine_name = "YHVH Tzabaoth"
        self.archangel = "Haniel"
        self.consciousness = "Nefesh (higher)"
        self.planetary_correspondence = "Venus"
        self.geometric_correspondence = "icosahedron"
        self.chakra_correspondence = "heart/throat"
        self.body_correspondence = "right leg"
        self.frequency_modifier = 0.7
        self.phi_harmonic_count = 3
        self.harmonic_count = 7
        
        # Base frequency for Netzach (emotional frequencies)
        self.base_frequency = 528.0  # Hz (Solfeggio frequency for transformation/love)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Netzach"""
        aspects = {
            # Primary aspects of Netzach
            "divine_victory": {
                "type": AspectType.VICTORY,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Divine victory and success"
            },
            "emotional_intelligence": {
                "type": AspectType.EMOTION,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.98,
                "description": "Emotional intelligence and fluidity"
            },
            "natural_forces": {
                "type": AspectType.NATURE,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.97,
                "description": "Connection to natural forces"
            },
            "artistic_beauty": {
                "type": AspectType.BEAUTY,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.96,
                "description": "Artistic beauty and aesthetics"
            },
            "passion": {
                "type": AspectType.PASSION,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.95,
                "description": "Divine passion and enthusiasm"
            },
            "emotional_love": {
                "type": AspectType.LOVE,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.94,
                "description": "Emotional love and connection"
            },
            "desire": {
                "type": AspectType.DESIRE,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.93,
                "description": "Constructive desire and longing"
            },
            "group_harmony": {
                "type": AspectType.VICTORY,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.92,
                "description": "Group harmony and collective victory"
            },
            "creative_expression": {
                "type": AspectType.ART,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.91,
                "description": "Creative expression and artistry"
            },
            
            # Extended aspects unique to Netzach
            "venusian_attraction": {
                "type": AspectType.LOVE,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.89,
                "description": "Venusian principles of attraction"
            },
            "emotional_healing": {
                "type": AspectType.EMOTION,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.88,
                "description": "Emotional healing and wholeness"
            },
            "sensuality": {
                "type": AspectType.DESIRE,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.86,
                "description": "Divine sensuality and pleasure"
            },
            "musical_harmony": {
                "type": AspectType.ART,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.85,
                "description": "Musical harmony and resonance"
            },
            "natural_cycles": {
                "type": AspectType.NATURE,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.83,
                "description": "Natural cycles and rhythms"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "chesed": {
                "name": "Victory to Mercy",
                "strength": 0.8,
                "quality": "love",
                "path_name": "Kaph",
                "tarot": "Wheel of Fortune",
                "description": "The path of loving success"
            },
            "tiphareth": {
                "name": "Victory to Beauty",
                "strength": 0.8,
                "quality": "beauty",
                "path_name": "Nun",
                "tarot": "Death",
                "description": "The path of transformative beauty"
            },
            "hod": {
                "name": "Victory to Splendor",
                "strength": 0.8,
                "quality": "integration",
                "path_name": "Peh",
                "tarot": "Tower",
                "description": "The path of integrating emotion and intellect"
            },
            "yesod": {
                "name": "Victory to Foundation",
                "strength": 0.75,
                "quality": "imagination",
                "path_name": "Tzaddi",
                "tarot": "Star",
                "description": "The path of emotional imagination"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Netzach.
        
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
        Get the primary aspects of Netzach.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.90}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Netzach.
        
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
        Calculate how strongly a frequency resonates with Netzach.
        
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
        
        # Add Venus resonance (Netzach corresponds to Venus)
        venus_resonance = self._calculate_venus_resonance(frequency)
        resonances.append(venus_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_venus_resonance(self, frequency):
        """Calculate resonance with Venus frequencies"""
        # Venus's orbit frequency (translated to audible range)
        venus_freq = 221.23  # Hz (derived from orbital period)
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / venus_freq, venus_freq / frequency)
        
        # Check for harmonic relationships
        harmonic = 1.0
        if abs(np.log2(frequency / venus_freq) % 1) < 0.1:
            harmonic = 2.0
            
        return ratio * harmonic * 0.86
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Netzach.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Victory",
            "strength": 0.95,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Netzach.
        
        Returns:
            float: Stability modifier value
        """
        return self.frequency_modifier - 0.05  # Less stable due to emotional nature
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Netzach.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Netzach.
        
        Returns:
            dict: Dimensional position information
        """
        return {
            "level": self.position,
            "pillar": self.pillar,
            "astral_plane": True  # Part of the astral/psychological triad
        }
    
    def get_platonic_solid(self):
        """
        Get the associated platonic solid for Netzach.
        
        Returns:
            str: Name of platonic solid
        """
        return self.geometric_correspondence
    
    # Netzach-specific methods
    
    def calculate_emotional_response(self, stimulus=None, entity=None):
        """
        Calculate emotional response to a stimulus.
        
        This is specific to Netzach as the Sephirah of emotions.
        
        Args:
            stimulus (dict): Optional stimulus data
            entity (dict): Optional entity data
            
        Returns:
            dict: Emotional response information
        """
        # Base emotional response
        base_response = {
            "love": 0.75,
            "joy": 0.70,
            "passion": 0.80,
            "empathy": 0.65,
            "inspiration": 0.60
        }
        
        # Modify based on stimulus if provided
        if stimulus and isinstance(stimulus, dict):
            # Add beauty modification
            if "beauty" in stimulus:
                beauty_factor = 0.6 + 0.4 * stimulus["beauty"]
                base_response["love"] *= beauty_factor
                base_response["inspiration"] *= beauty_factor
            
            # Add emotional resonance modification
            if "emotional_tone" in stimulus:
                tone = stimulus["emotional_tone"]
                if tone == "joyful":
                    base_response["joy"] *= 1.3
                elif tone == "loving":
                    base_response["love"] *= 1.3
                elif tone == "passionate":
                    base_response["passion"] *= 1.3
        
        # Modify based on entity if provided
        if entity and isinstance(entity, dict):
            # Add emotional sensitivity modification
            if "emotional_sensitivity" in entity:
                sensitivity = entity["emotional_sensitivity"]
                for key in base_response:
                    base_response[key] *= 0.7 + 0.3 * sensitivity
        
        # Ensure values are within bounds
        for key in base_response:
            base_response[key] = min(1.0, max(0.0, base_response[key]))
        
        return base_response
    
    def generate_artistic_inspiration(self, art_form="music", intensity=0.7):
        """
        Generate artistic inspiration for a specific art form.
        
        Args:
            art_form (str): Type of art form
            intensity (float): Intensity of inspiration (0-1)
            
        Returns:
            dict: Inspiration data
        """
        # Validate inputs
        intensity = min(1.0, max(0.0, intensity))
        valid_art_forms = ["music", "visual", "dance", "poetry", "drama"]
        art_form = art_form if art_form in valid_art_forms else "music"
        
        # Base inspiration values
        base_inspiration = {
            "creativity": 0.7 + 0.3 * intensity,
            "emotional_depth": 0.8 + 0.2 * intensity,
            "beauty_perception": 0.6 + 0.4 * intensity,
            "technical_skill": 0.5 + 0.3 * intensity,
            "duration": 3 + 4 * intensity  # Hours
        }
        
        # Art form specific adjustments
        if art_form == "music":
            base_inspiration["rhythm"] = 0.7 + 0.3 * intensity
            base_inspiration["harmony"] = 0.8 + 0.2 * intensity
            base_inspiration["melody"] = 0.75 + 0.25 * intensity
        elif art_form == "visual":
            base_inspiration["color"] = 0.8 + 0.2 * intensity
            base_inspiration["form"] = 0.7 + 0.3 * intensity
            base_inspiration["composition"] = 0.75 + 0.25 * intensity
        elif art_form == "dance":
            base_inspiration["movement"] = 0.85 + 0.15 * intensity
            base_inspiration["expression"] = 0.8 + 0.2 * intensity
            base_inspiration["rhythm"] = 0.75 + 0.25 * intensity
        elif art_form == "poetry":
            base_inspiration["imagery"] = 0.8 + 0.2 * intensity
            base_inspiration["rhythm"] = 0.7 + 0.3 * intensity
            base_inspiration["metaphor"] = 0.85 + 0.15 * intensity
        elif art_form == "drama":
            base_inspiration["character"] = 0.75 + 0.25 * intensity
            base_inspiration["emotion"] = 0.85 + 0.15 * intensity
            base_inspiration["narrative"] = 0.7 + 0.3 * intensity
        
        return {
            "art_form": art_form,
            "intensity": intensity,
            "inspiration": base_inspiration
        }


