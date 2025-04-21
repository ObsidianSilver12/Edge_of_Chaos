"""
Enhanced Hod Aspects Module

This module defines the specific aspects for Hod, Glory - the eighth
Sephirah representing intellect, communication, and analysis.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.hod')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    GLORY = auto()           # Glory/splendor aspects
    INTELLECT = auto()       # Intellectual aspects
    COMMUNICATION = auto()   # Communication aspects
    ANALYSIS = auto()        # Analytical aspects
    LOGIC = auto()           # Logical aspects
    LANGUAGE = auto()        # Language aspects
    LEARNING = auto()        # Learning/teaching aspects
    SCIENCE = auto()         # Scientific aspects

class HodAspects:
    """
    Hod (Glory) Aspects
    
    Contains the specific aspects, frequencies, and properties of Hod,
    the eighth Sephirah representing intellect, communication, and analysis.
    """
    
    def __init__(self):
        """Initialize Hod aspects"""
        logger.info("Initializing Hod aspects")
        
        # Base properties (from dictionary)
        self.name = "hod"
        self.title = "Glory"
        self.position = 8
        self.pillar = "left"
        self.element = "water/air"
        self.color = "orange"
        self.divine_name = "Elohim Tzabaoth"
        self.archangel = "Michael"
        self.consciousness = "Nefesh (lower)"
        self.planetary_correspondence = "Mercury"
        self.geometric_correspondence = "dodecahedron"
        self.chakra_correspondence = "throat"
        self.body_correspondence = "left leg"
        self.frequency_modifier = 0.65
        self.phi_harmonic_count = 3
        self.harmonic_count = 7
        
        # Base frequency for Hod (intellectual frequencies)
        self.base_frequency = 741.0  # Hz (Solfeggio frequency for intuition/expression)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Hod"""
        aspects = {
            # Primary aspects of Hod
            "divine_glory": {
                "type": AspectType.GLORY,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Divine glory and splendor"
            },
            "intellect": {
                "type": AspectType.INTELLECT,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.98,
                "description": "Divine intellect and reasoning"
            },
            "communication": {
                "type": AspectType.COMMUNICATION,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.97,
                "description": "Divine communication and expression"
            },
            "analysis": {
                "type": AspectType.ANALYSIS,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.96,
                "description": "Analysis and discernment"
            },
            "logic": {
                "type": AspectType.LOGIC,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.95,
                "description": "Divine logic and reason"
            },
            "language": {
                "type": AspectType.LANGUAGE,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.94,
                "description": "Language and symbols"
            },
            "learning": {
                "type": AspectType.LEARNING,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.93,
                "description": "Learning and teaching"
            },
            "systems": {
                "type": AspectType.SCIENCE,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.92,
                "description": "Systems and structures"
            },
            "science": {
                "type": AspectType.SCIENCE,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.91,
                "description": "Scientific understanding"
            },
            
            # Extended aspects unique to Hod
            "mercurial_intellect": {
                "type": AspectType.INTELLECT,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.89,
                "description": "Mercurial, quick intellect"
            },
            "writing": {
                "type": AspectType.COMMUNICATION,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.88,
                "description": "The art of writing and recording"
            },
            "technology": {
                "type": AspectType.SCIENCE,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.86,
                "description": "Technology and innovation"
            },
            "magical_names": {
                "type": AspectType.LANGUAGE,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.85,
                "description": "Magical names and words of power"
            },
            "hermetic_wisdom": {
                "type": AspectType.LEARNING,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.83,
                "description": "Hermetic wisdom and knowledge"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "geburah": {
                "name": "Glory to Severity",
                "strength": 0.8,
                "quality": "structure",
                "path_name": "Mem",
                "tarot": "Hanged Man",
                "description": "The path of structured judgment"
            },
            "tiphareth": {
                "name": "Glory to Beauty",
                "strength": 0.8,
                "quality": "revelation",
                "path_name": "Ayin",
                "tarot": "Devil",
                "description": "The path of intellectual revelation"
            },
            "netzach": {
                "name": "Glory to Victory",
                "strength": 0.8,
                "quality": "integration",
                "path_name": "Peh",
                "tarot": "Tower",
                "description": "The path of integrating intellect and emotion"
            },
            "yesod": {
                "name": "Glory to Foundation",
                "strength": 0.75,
                "quality": "communication",
                "path_name": "Qoph",
                "tarot": "Moon",
                "description": "The path of processing and communication"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Hod.
        
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
        Get the primary aspects of Hod.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.90}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Hod.
        
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
        Calculate how strongly a frequency resonates with Hod.
        
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
        
        # Add Mercury resonance (Hod corresponds to Mercury)
        mercury_resonance = self._calculate_mercury_resonance(frequency)
        resonances.append(mercury_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_mercury_resonance(self, frequency):
        """Calculate resonance with Mercury frequencies"""
        # Mercury's orbit frequency (translated to audible range)
        mercury_freq = 141.27  # Hz (derived from orbital period)
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / mercury_freq, mercury_freq / frequency)
        
        # Check for harmonic relationships
        harmonic = 1.0
        if abs(np.log2(frequency / mercury_freq) % 1) < 0.1:
            harmonic = 2.0
            
        return ratio * harmonic * 0.85
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Hod.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Glory",
            "strength": 0.95,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Hod.
        
        Returns:
            float: Stability modifier value
        """
        return self.frequency_modifier + 0.05  # More stable due to logical nature
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Hod.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Hod.
        
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
        Get the associated platonic solid for Hod.
        
        Returns:
            str: Name of platonic solid
        """
        return self.geometric_correspondence
    
    # Hod-specific methods
    
    def analyze_concept(self, concept=None, complexity=0.5):
        """
        Analyze a concept using Hod's intellectual aspects.
        
        This is specific to Hod as the Sephirah of intellect and analysis.
        
        Args:
            concept (dict): Optional concept data
            complexity (float): Complexity level of analysis (0-1)
            
        Returns:
            dict: Analysis results
        """
        # Validate inputs
        complexity = min(1.0, max(0.0, complexity))
        
        # Base analysis values
        base_analysis = {
            "logical_structure": 0.7 + 0.3 * complexity,
            "symbolic_representation": 0.65 + 0.35 * complexity,
            "linguistic_expression": 0.75 + 0.25 * complexity,
            "systemic_integration": 0.6 + 0.4 * complexity,
            "practical_application": 0.5 + 0.5 * complexity
        }
        
        # Modify based on concept if provided
        if concept and isinstance(concept, dict):
            # Add clarity modification
            if "clarity" in concept:
                clarity_factor = 0.6 + 0.4 * concept["clarity"]
                base_analysis["logical_structure"] *= clarity_factor
                base_analysis["symbolic_representation"] *= clarity_factor
            
            # Add complexity modification
            if "internal_complexity" in concept:
                complexity_factor = 0.5 + 0.5 * concept["internal_complexity"]
                base_analysis["systemic_integration"] *= complexity_factor
                base_analysis["practical_application"] *= (2.0 - complexity_factor) * 0.5  # Inverse relationship
        
        # Ensure values are within bounds
        for key in base_analysis:
            base_analysis[key] = min(1.0, max(0.0, base_analysis[key]))
        
        return base_analysis
    
    def formulate_communication(self, message=None, target=None, clarity=0.8):
        """
        Formulate a clear communication of ideas.
        
        Args:
            message (dict): Optional message content
            target (dict): Optional target audience
            clarity (float): Desired clarity level (0-1)
            
        Returns:
            dict: Communication effectiveness data
        """
        # Validate inputs
        clarity = min(1.0, max(0.0, clarity))
        
        # Base communication effectiveness
        base_effectiveness = {
            "clarity": 0.7 + 0.3 * clarity,
            "precision": 0.65 + 0.35 * clarity,
            "structure": 0.6 + 0.4 * clarity,
            "persuasiveness": 0.5 + 0.3 * clarity,
            "memorability": 0.55 + 0.25 * clarity
        }
        
        # Adjust based on message if provided
        if message and isinstance(message, dict):
            # Add complexity adjustment
            if "complexity" in message:
                complexity = message["complexity"]
                complexity_factor = 1.0 - 0.5 * complexity  # More complex = less clarity
                base_effectiveness["clarity"] *= complexity_factor
                base_effectiveness["memorability"] *= complexity_factor
            
            # Add importance adjustment
            if "importance" in message:
                importance = message["importance"]
                base_effectiveness["persuasiveness"] *= 0.7 + 0.3 * importance
        
        # Adjust based on target if provided
        if target and isinstance(target, dict):
            # Add receptivity adjustment
            if "receptivity" in target:
                receptivity = target["receptivity"]
                for key in base_effectiveness:
                    base_effectiveness[key] *= 0.7 + 0.3 * receptivity
            
            # Add educational level adjustment
            if "education_level" in target:
                education = target["education_level"]
                base_effectiveness["precision"] *= 0.6 + 0.4 * education
                base_effectiveness["structure"] *= 0.7 + 0.3 * education
        
        # Ensure values are within bounds
        for key in base_effectiveness:
            base_effectiveness[key] = min(1.0, max(0.0, base_effectiveness[key]))
        
        return base_effectiveness
