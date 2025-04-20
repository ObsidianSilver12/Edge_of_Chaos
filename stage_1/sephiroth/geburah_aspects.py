"""
Enhanced Geburah Aspects Module

This module defines the specific aspects for Geburah, Severity - the fifth
Sephirah representing severity, discipline, and judgment.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.geburah')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    SEVERITY = auto()    # Severity aspects
    JUDGMENT = auto()    # Judgment aspects
    DISCIPLINE = auto()  # Discipline/restriction aspects
    STRENGTH = auto()    # Strength aspects
    COURAGE = auto()     # Courage aspects
    ELIMINATION = auto() # Elimination aspects
    PROTECTION = auto()  # Protection aspects

class GeburahAspects:
    """
    Geburah (Severity) Aspects
    
    Contains the specific aspects, frequencies, and properties of Geburah,
    the fifth Sephirah representing severity, discipline, and judgment.
    """
    
    def __init__(self):
        """Initialize Geburah aspects"""
        logger.info("Initializing Geburah aspects")
        
        # Base properties (from dictionary)
        self.name = "geburah"
        self.title = "Severity"
        self.position = 5
        self.pillar = "left"
        self.element = "fire"
        self.color = "red"
        self.divine_name = "Elohim Gibor"
        self.archangel = "Khamael"
        self.consciousness = "Ruach (lower)"
        self.planetary_correspondence = "Mars"
        self.geometric_correspondence = "octahedron"
        self.chakra_correspondence = "solar plexus"
        self.body_correspondence = "left arm"
        self.frequency_modifier = 0.8
        self.phi_harmonic_count = 4
        self.harmonic_count = 8
        
        # Base frequency for Geburah (transformative frequencies)
        self.base_frequency = 396.0  # Hz (Solfeggio frequency for liberation)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Geburah"""
        aspects = {
            # Primary aspects of Geburah
            "divine_severity": {
                "type": AspectType.SEVERITY,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Divine severity and judgment"
            },
            "discipline": {
                "type": AspectType.DISCIPLINE,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.98,
                "description": "Divine discipline and order"
            },
            "strength": {
                "type": AspectType.STRENGTH,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.97,
                "description": "Divine strength and power"
            },
            "righteous_judgment": {
                "type": AspectType.JUDGMENT,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.96,
                "description": "Righteous judgment and discernment"
            },
            "restriction": {
                "type": AspectType.DISCIPLINE,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.95,
                "description": "Divine restriction and limitation"
            },
            "elimination": {
                "type": AspectType.ELIMINATION,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.94,
                "description": "Elimination of the unnecessary"
            },
            "courage": {
                "type": AspectType.COURAGE,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.93,
                "description": "Divine courage and valor"
            },
            "protection": {
                "type": AspectType.PROTECTION,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.92,
                "description": "Divine protection and boundaries"
            },
            "purification": {
                "type": AspectType.ELIMINATION,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.91,
                "description": "Purification through divine fire"
            },
            
            # Extended aspects unique to Geburah
            "divine_warrior": {
                "type": AspectType.STRENGTH,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.89,
                "description": "The divine warrior principle"
            },
            "boundaries": {
                "type": AspectType.PROTECTION,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.88,
                "description": "Setting divine boundaries"
            },
            "discernment": {
                "type": AspectType.JUDGMENT,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.86,
                "description": "Sharp discernment and insight"
            },
            "karmic_justice": {
                "type": AspectType.JUDGMENT,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.85,
                "description": "Karmic justice and balance"
            },
            "divine_sacrifice": {
                "type": AspectType.ELIMINATION,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.83,
                "description": "Willing sacrifice for higher purpose"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "binah": {
                "name": "Severity to Understanding",
                "strength": 0.85,
                "quality": "restriction",
                "path_name": "Zayin",
                "tarot": "Lovers",
                "description": "The path of restriction and limitation"
            },
            "tiphareth": {
                "name": "Severity to Beauty",
                "strength": 0.85,
                "quality": "purification",
                "path_name": "Lamed",
                "tarot": "Justice",
                "description": "The path of purification through fire"
            },
            "chesed": {
                "name": "Severity to Mercy",
                "strength": 0.9,
                "quality": "balance",
                "path_name": "Teth",
                "tarot": "Strength",
                "description": "The path of balanced judgment"
            },
            "hod": {
                "name": "Severity to Splendor",
                "strength": 0.8,
                "quality": "structure",
                "path_name": "Mem",
                "tarot": "Hanged Man",
                "description": "The path of structured judgment"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Geburah.
        
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
        Get the primary aspects of Geburah.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.90}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Geburah.
        
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
        Calculate how strongly a frequency resonates with Geburah.
        
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
        
        # Add Mars resonance (Geburah corresponds to Mars)
        mars_resonance = self._calculate_mars_resonance(frequency)
        resonances.append(mars_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_mars_resonance(self, frequency):
        """Calculate resonance with Mars frequencies"""
        # Mars's orbit frequency (translated to audible range)
        mars_freq = 144.72  # Hz (derived from orbital period)
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / mars_freq, mars_freq / frequency)
        
        # Check for harmonic relationships
        harmonic = 1.0
        if abs(np.log2(frequency / mars_freq) % 1) < 0.1:
            harmonic = 2.0
            
        return ratio * harmonic * 0.87
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Geburah.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Severity",
            "strength": 0.95,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Geburah.
        
        Returns:
            float: Stability modifier value
        """
        return self.frequency_modifier - 0.05  # Less stable due to transformative nature
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Geburah.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Geburah.
        
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
        Get the associated platonic solid for Geburah.
        
        Returns:
            str: Name of platonic solid
        """
        return self.geometric_correspondence
    
    # Geburah-specific methods
    
    def calculate_boundary_strength(self, entity=None):
        """
        Calculate boundary strength for an entity.
        
        This is specific to Geburah as the Sephirah of boundaries.
        
        Args:
            entity (dict): Optional entity data
            
        Returns:
            dict: Boundary strength information
        """
        # Base boundary values
        base_boundaries = {
            "physical": 0.75,
            "emotional": 0.70
                        "physical": 0.75,
            "emotional": 0.70,
            "mental": 0.80,
            "spiritual": 0.65
        }
        
        # Modify based on entity properties if provided
        if entity and isinstance(entity, dict):
            # Add strength modification
            if "strength" in entity:
                strength_factor = 0.7 + 0.3 * entity["strength"]
                base_boundaries["physical"] *= strength_factor
            
            # Add discipline modification
            if "discipline" in entity:
                discipline_factor = 0.6 + 0.4 * entity["discipline"]
                base_boundaries["mental"] *= discipline_factor
                base_boundaries["emotional"] *= (0.8 + 0.2 * discipline_factor)
        
        # Ensure values are within bounds
        for key in base_boundaries:
            base_boundaries[key] = min(1.0, max(0.0, base_boundaries[key]))
        
        return base_boundaries
    
    def apply_purification(self, entity=None, intensity=0.5):
        """
        Apply a purification process to an entity.
        
        Args:
            entity (dict): Optional entity data
            intensity (float): Intensity of purification (0-1)
            
        Returns:
            dict: Purification effect information
        """
        # Validate input
        intensity = min(1.0, max(0.0, intensity))
        
        # Calculate purification effects
        effects = {
            "removal": 0.5 + 0.5 * intensity,
            "refinement": 0.6 + 0.4 * intensity,
            "strengthening": 0.4 + 0.6 * intensity,
            "duration": 5 * intensity  # Days
        }
        
        # If entity provided, add specific effects
        if entity and isinstance(entity, dict):
            # Store purification in entity if possible
            if "processes" in entity and isinstance(entity["processes"], list):
                entity["processes"].append({
                    "type": "purification",
                    "strength": intensity,
                    "effects": effects
                })
        
        return effects
