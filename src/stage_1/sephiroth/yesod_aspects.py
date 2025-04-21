"""
Enhanced Yesod Aspects Module

This module defines the specific aspects for Yesod, Foundation - the ninth
Sephirah representing foundation, dreams, and the subconscious.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sephiroth_aspects.yesod')

class AspectType(Enum):
    """Enumeration of aspect types for categorization"""
    FOUNDATION = auto()     # Foundation aspects
    DREAM = auto()          # Dream aspects
    SUBCONSCIOUS = auto()   # Subconscious aspects
    IMAGINATION = auto()    # Imagination aspects
    MEMORY = auto()         # Memory aspects
    CYCLES = auto()         # Cyclical pattern aspects
    PURIFICATION = auto()   # Purification aspects
    ASTRAL = auto()         # Astral/subtle energy aspects

class YesodAspects:
    """
    Yesod (Foundation) Aspects
    
    Contains the specific aspects, frequencies, and properties of Yesod,
    the ninth Sephirah representing foundation, dreams, and the subconscious.
    """
    
    def __init__(self):
        """Initialize Yesod aspects"""
        logger.info("Initializing Yesod aspects")
        
        # Base properties (from dictionary)
        self.name = "yesod"
        self.title = "Foundation"
        self.position = 9
        self.pillar = "middle"
        self.element = "air"
        self.color = "purple"
        self.divine_name = "Shaddai El Chai"
        self.archangel = "Gabriel"
        self.consciousness = "Nefesh (foundation)"
        self.planetary_correspondence = "Moon"
        self.geometric_correspondence = "spheroid"
        self.chakra_correspondence = "sacral"
        self.body_correspondence = "reproductive organs"
        self.frequency_modifier = 0.6
        self.phi_harmonic_count = 2
        self.harmonic_count = 6
        
        # Base frequency for Yesod (lunar frequencies)
        self.base_frequency = 852.0  # Hz (Solfeggio frequency for intuition)
        
        # Initialize the detailed aspects and relationships
        self.aspects = self._initialize_aspects()
        self.relationships = self._initialize_relationships()
        
    def _initialize_aspects(self):
        """Initialize the specific aspects of Yesod"""
        aspects = {
            # Primary aspects of Yesod
            "divine_foundation": {
                "type": AspectType.FOUNDATION,
                "frequency": self.base_frequency,
                "strength": 1.0,
                "description": "Divine foundation and structure"
            },
            "dream_consciousness": {
                "type": AspectType.DREAM,
                "frequency": self.base_frequency * 1.125,
                "strength": 0.98,
                "description": "Dream consciousness and awareness"
            },
            "subconscious_mind": {
                "type": AspectType.SUBCONSCIOUS,
                "frequency": self.base_frequency * 1.333,
                "strength": 0.97,
                "description": "Subconscious mind and processes"
            },
            "imagination": {
                "type": AspectType.IMAGINATION,
                "frequency": self.base_frequency * 1.618,  # Golden ratio
                "strength": 0.96,
                "description": "Divine imagination and creativity"
            },
            "memory": {
                "type": AspectType.MEMORY,
                "frequency": self.base_frequency * 0.882,
                "strength": 0.95,
                "description": "Memory and record keeping"
            },
            "cyclical_patterns": {
                "type": AspectType.CYCLES,
                "frequency": self.base_frequency * 0.75,
                "strength": 0.94,
                "description": "Cyclical patterns and rhythms"
            },
            "purification": {
                "type": AspectType.PURIFICATION,
                "frequency": self.base_frequency * 1.5,
                "strength": 0.93,
                "description": "Purification and filtration"
            },
            "astral_projection": {
                "type": AspectType.ASTRAL,
                "frequency": self.base_frequency * 0.667,
                "strength": 0.92,
                "description": "Astral projection and travel"
            },
            "subtle_energy": {
                "type": AspectType.ASTRAL,
                "frequency": self.base_frequency * 2.0,  # Octave
                "strength": 0.91,
                "description": "Subtle energy perception and manipulation"
            },
            
            # Extended aspects unique to Yesod
            "lunar_cycles": {
                "type": AspectType.CYCLES,
                "frequency": self.base_frequency * 3.14159,  # Pi
                "strength": 0.89,
                "description": "Lunar cycles and influences"
            },
            "dreamwork": {
                "type": AspectType.DREAM,
                "frequency": self.base_frequency * 1.414,  # √2
                "strength": 0.88,
                "description": "Dreamwork and dream interpretation"
            },
            "psychic_awareness": {
                "type": AspectType.ASTRAL,
                "frequency": self.base_frequency * 1.732,  # √3
                "strength": 0.86,
                "description": "Psychic awareness and sensitivity"
            },
            "sexual_energy": {
                "type": AspectType.FOUNDATION,
                "frequency": self.base_frequency * 0.5,
                "strength": 0.85,
                "description": "Sexual energy and creative force"
            },
            "collective_unconscious": {
                "type": AspectType.SUBCONSCIOUS,
                "frequency": self.base_frequency * 1.111,
                "strength": 0.83,
                "description": "Connection to collective unconscious"
            }
        }
        
        return aspects
    
    def _initialize_relationships(self):
        """Initialize relationships with other Sephiroth"""
        return {
            "tiphareth": {
                "name": "Foundation to Beauty",
                "strength": 0.8,
                "quality": "reflection",
                "path_name": "Samekh",
                "tarot": "Temperance",
                "description": "The path of reflecting higher consciousness"
            },
            "netzach": {
                "name": "Foundation to Victory",
                "strength": 0.75,
                "quality": "imagination",
                "path_name": "Tzaddi",
                "tarot": "Star",
                "description": "The path of emotional imagination"
            },
            "hod": {
                "name": "Foundation to Glory",
                "strength": 0.75,
                "quality": "communication",
                "path_name": "Qoph",
                "tarot": "Moon",
                "description": "The path of processing and communication"
            },
            "malkuth": {
                "name": "Foundation to Kingdom",
                "strength": 0.7,
                "quality": "manifestation",
                "path_name": "Tav",
                "tarot": "World",
                "description": "The path of manifestation into physical reality"
            }
        }
    
    def get_metadata(self):
        """
        Get the metadata for Yesod.
        
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
        Get the primary aspects of Yesod.
        
        Returns:
            dict: Dictionary of primary aspects
        """
        return {k: v for k, v in self.aspects.items() if v["strength"] >= 0.90}
    
    def get_secondary_aspects(self):
        """
        Get the secondary aspects of Yesod.
        
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
        Calculate how strongly a frequency resonates with Yesod.
        
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
        
        # Add Moon resonance (Yesod corresponds to Moon)
        moon_resonance = self._calculate_moon_resonance(frequency)
        resonances.append(moon_resonance)
        
        # Return maximum resonance
        return max(resonances) if resonances else 0.0
    
    def _calculate_moon_resonance(self, frequency):
        """Calculate resonance with Moon frequencies"""
        # Moon's orbit frequency (translated to audible range)
        moon_freq = 210.42  # Hz (derived from lunar cycle)
        
        # Calculate ratio (always < 1.0)
        ratio = min(frequency / moon_freq, moon_freq / frequency)
        
        # Check for harmonic relationships
        harmonic = 1.0
        if abs(np.log2(frequency / moon_freq) % 1) < 0.1:
            harmonic = 2.0
            
        return ratio * harmonic * 0.88
    
    # Methods for compatibility with the existing dictionary API
    
    def get_divine_quality(self):
        """
        Get the divine quality of Yesod.
        
        Returns:
            dict: Divine quality information
        """
        return {
            "name": "Divine Foundation",
            "strength": 0.95,
            "primary_aspects": list(self.get_primary_aspects().keys()),
            "secondary_aspects": list(self.get_secondary_aspects().keys())
        }
    
    def get_stability_modifier(self):
        """
        Get the stability modifier for Yesod.
        
        Returns:
            float: Stability modifier value
        """
        return self.frequency_modifier - 0.05  # Less stable due to dream/fluid nature
    
    def get_resonance_multiplier(self):
        """
        Get the resonance multiplier for Yesod.
        
        Returns:
            float: Resonance multiplier value
        """
        return self.phi_harmonic_count / 7.0  # Normalized to 0-1 range
    
    def get_dimensional_position(self):
        """
        Get the dimensional position of Yesod.
        
        Returns:
            dict: Dimensional position information
        """
        return {
            "level": self.position,
            "pillar": self.pillar,
            "astral_plane": True,  # Part of the astral/psychological triad
            "foundation_role": True  # Acts as foundation between Malkuth and higher Sephiroth
        }
    
    def get_platonic_solid(self):
        """
        Get the associated platonic solid for Yesod.
        
        Returns:
            str: Name of platonic solid
        """
        return self.geometric_correspondence
    
    # Yesod-specific methods
    
    def process_dream_state(self, dream_content=None, consciousness_level=0.3):
        """
        Process dream state experiences and content.
        
        This is specific to Yesod as the Sephirah of dreams and the subconscious.
        
        Args:
            dream_content (dict): Optional dream content data
            consciousness_level (float): Level of consciousness during dream (0-1)
            
        Returns:
            dict: Dream processing results
        """
        # Validate inputs
        consciousness_level = min(1.0, max(0.0, consciousness_level))
        
        # Base dream processing values
        base_processing = {
            "recall_clarity": 0.4 + 0.6 * consciousness_level,
            "symbolic_content": 0.8 - 0.3 * consciousness_level,  # More symbolic at lower consciousness
            "emotional_intensity": 0.65 + 0.2 * (0.5 - abs(0.5 - consciousness_level)),  # Peaks at mid-level
            "integration_potential": 0.3 + 0.7 * consciousness_level,
            "spiritual_insight": 0.5 + 0.5 * consciousness_level
        }
        
        # Modify based on dream content if provided
        if dream_content and isinstance(dream_content, dict):
            # Add lucidity modification
            if "lucidity" in dream_content:
                lucidity = dream_content["lucidity"]
                base_processing["recall_clarity"] *= 0.6 + 0.4 * lucidity
                base_processing["integration_potential"] *= 0.7 + 0.3 * lucidity
            
            # Add emotional content modification
            if "emotional_content" in dream_content:
                emotional = dream_content["emotional_content"]
                base_processing["emotional_intensity"] *= 0.5 + 0.5 * emotional
                base_processing["symbolic_content"] *= 0.7 + 0.3 * emotional
            
            # Add spiritual content modification
            if "spiritual_content" in dream_content:
                spiritual = dream_content["spiritual_content"]
                base_processing["spiritual_insight"] *= 0.6 + 0.4 * spiritual
        
        # Ensure values are within bounds
        for key in base_processing:
            base_processing[key] = min(1.0, max(0.0, base_processing[key]))
        
        return base_processing
    
    def evaluate_astral_connections(self, connections=None, receptivity=0.5):
        """
        Evaluate astral connections and subtle energy pathways.
        
        Args:
            connections (list): Optional list of connection points
            receptivity (float): Receptivity to astral influences (0-1)
            
        Returns:
            dict: Astral connection evaluation
        """
        # Validate inputs
        receptivity = min(1.0, max(0.0, receptivity))
        
        # Base astral connection values
        base_connections = {
            "clarity": 0.6 + 0.4 * receptivity,
            "strength": 0.5 + 0.5 * receptivity,
            "duration": 0.4 + 0.6 * receptivity,
            "information_fidelity": 0.45 + 0.55 * receptivity,
            "protection_level": 0.7 - 0.2 * receptivity  # Higher receptivity = slightly lower protection
        }
        
        # Connection-specific adjustments if provided
        if connections and isinstance(connections, list):
            # Calculate number of active connections
            num_connections = len(connections)
            
            # Adjust based on number of connections (too many dilutes strength)
            connection_factor = 1.0 if num_connections <= 3 else 3.0 / num_connections
            
            base_connections["clarity"] *= 0.8 + 0.2 * connection_factor
            base_connections["strength"] *= 0.7 + 0.3 * connection_factor
            
            # Process individual connection points if they have attributes
            lunar_influence = 0
            higher_sephiroth = 0
            
            for connection in connections:
                if isinstance(connection, dict):
                    if connection.get("type") == "lunar":
                        lunar_influence += 1
                    if connection.get("level", 0) < 9:  # Higher Sephiroth
                        higher_sephiroth += 1
            
            # Apply lunar influences (enhances receptivity)
            if lunar_influence > 0:
                lunar_factor = min(1.0, lunar_influence / 3.0)
                base_connections["clarity"] *= 1.0 + 0.2 * lunar_factor
                base_connections["duration"] *= 1.0 + 0.3 * lunar_factor
            
            # Apply higher Sephiroth influences (enhances information quality)
            if higher_sephiroth > 0:
                higher_factor = min(1.0, higher_sephiroth / 5.0)
                base_connections["information_fidelity"] *= 1.0 + 0.3 * higher_factor
                base_connections["protection_level"] *= 1.0 + 0.2 * higher_factor
        
        # Ensure values are within bounds
        for key in base_connections:
            base_connections[key] = min(1.0, max(0.0, base_connections[key]))
        
        return base_connections
    
    def calculate_cycle_influence(self, cycle_type="lunar", phase=0.5):
        """
        Calculate the influence of natural cycles on Yesod energies.
        
        Args:
            cycle_type (str): Type of cycle (lunar, solar, etc.)
            phase (float): Current phase of the cycle (0-1)
            
        Returns:
            dict: Cycle influence data
        """
        # Validate inputs
        phase = min(1.0, max(0.0, phase))
        valid_cycles = ["lunar", "solar", "seasonal", "circadian"]
        cycle_type = cycle_type if cycle_type in valid_cycles else "lunar"
        
        # Base influence values
        base_influence = {
            "strength": 0.0,  # Will be set based on cycle type
            "receptivity": 0.0,
            "dream_clarity": 0.0,
            "subconsciousness_access": 0.0,
            "manifestation_potential": 0.0
        }
        
        # Cycle-specific patterns
        if cycle_type == "lunar":
            # Lunar cycle (peaks at full moon - 0.5)
            peak_phase = 0.5
            amplitude = 0.8
            base_influence["strength"] = 0.7 + 0.3 * (1.0 - 2.0 * abs(phase - peak_phase))
            
            # Receptivity peaks at full moon
            base_influence["receptivity"] = 0.5 + 0.5 * (1.0 - 2.0 * abs(phase - peak_phase))
            
            # Dream clarity peaks just before full moon
            dream_peak = 0.45
            base_influence["dream_clarity"] = 0.4 + 0.6 * (1.0 - 2.0 * abs(phase - dream_peak))
            
            # Subconscious access peaks at new moon
            subcon_peak = 0.0
            base_influence["subconsciousness_access"] = 0.5 + 0.5 * (1.0 - 2.0 * abs(phase - subcon_peak))
            
            # Manifestation potential peaks after full moon
            manifest_peak = 0.6
            base_influence["manifestation_potential"] = 0.4 + 0.6 * (1.0 - 2.0 * abs(phase - manifest_peak))
            
        elif cycle_type == "circadian":
            # Circadian rhythm (peaks at night - 0.75)
            peak_phase = 0.75
            base_influence["strength"] = 0.4 + 0.6 * (1.0 - 2.0 * abs(phase - peak_phase))
            
            # Dream clarity peaks in early morning
            dream_peak = 0.85
            base_influence["dream_clarity"] = 0.3 + 0.7 * (1.0 - 2.0 * abs(phase - dream_peak))
            
            # Subconscious access peaks at night
            subcon_peak = 0.75
            base_influence["subconsciousness_access"] = 0.4 + 0.6 * (1.0 - 2.0 * abs(phase - subcon_peak))
            
            # Other values follow the main cycle
            base_influence["receptivity"] = 0.5 + 0.5 * (1.0 - 2.0 * abs(phase - peak_phase))
            base_influence["manifestation_potential"] = 0.6 + 0.4 * (1.0 - 2.0 * abs(phase - (peak_phase + 0.125) % 1.0))
            
        elif cycle_type == "seasonal":
            # Seasonal cycle (peaks at winter solstice for introspection - 0.0)
            winter_peak = 0.0  # Winter solstice
            summer_peak = 0.5  # Summer solstice
            
            # Overall strength has two peaks - winter for introspection, summer for manifestation
            winter_strength = 0.8 * (1.0 - 2.0 * abs(phase - winter_peak))
            summer_strength = 0.6 * (1.0 - 2.0 * abs(phase - summer_peak))
            base_influence["strength"] = 0.6 + 0.4 * max(winter_strength, summer_strength)
            
            # Dream clarity peaks in winter
            base_influence["dream_clarity"] = 0.5 + 0.5 * (1.0 - 2.0 * abs(phase - winter_peak))
            
            # Subconscious access peaks in autumn/winter
            autumn_peak = 0.75  # Autumn equinox
            autumn_winter = 0.7 * (1.0 - 2.0 * abs(phase - autumn_peak)) + 0.8 * (1.0 - 2.0 * abs(phase - winter_peak))
            base_influence["subconsciousness_access"] = 0.4 + 0.6 * (autumn_winter / 1.5)
            
            # Manifestation potential peaks in spring/summer
            spring_peak = 0.25  # Spring equinox
            spring_summer = 0.7 * (1.0 - 2.0 * abs(phase - spring_peak)) + 0.8 * (1.0 - 2.0 * abs(phase - summer_peak))
            base_influence["manifestation_potential"] = 0.4 + 0.6 * (spring_summer / 1.5)
            
            # Receptivity follows the main strength
            base_influence["receptivity"] = base_influence["strength"] * 0.9
            
        elif cycle_type == "solar":
            # Solar cycle (peaks at dawn and dusk - 0.25 and 0.75)
            dawn_peak = 0.25
            dusk_peak = 0.75
            
            # Overall strength has two peaks
            dawn_strength = 0.7 * (1.0 - 2.0 * abs(phase - dawn_peak))
            dusk_strength = 0.8 * (1.0 - 2.0 * abs(phase - dusk_peak))
            base_influence["strength"] = 0.5 + 0.5 * max(dawn_strength, dusk_strength)
            
            # Dream clarity peaks before dawn
            predawn_peak = 0.2
            base_influence["dream_clarity"] = 0.4 + 0.6 * (1.0 - 2.0 * abs(phase - predawn_peak))
            
            # Subconscious access peaks at midnight
            midnight_peak = 0.0
            base_influence["subconsciousness_access"] = 0.5 + 0.5 * (1.0 - 2.0 * abs(phase - midnight_peak))
            
            # Manifestation potential peaks at noon
            noon_peak = 0.5
            base_influence["manifestation_potential"] = 0.6 + 0.4 * (1.0 - 2.0 * abs(phase - noon_peak))
            
            # Receptivity peaks at dusk
            base_influence["receptivity"] = 0.5 + 0.5 * (1.0 - 2.0 * abs(phase - dusk_peak))
        
        # Ensure values are within bounds
        for key in base_influence:
            base_influence[key] = min(1.0, max(0.0, base_influence[key]))
        
        return base_influence
    
    def stabilize_foundation(self, entity=None, stability_level=0.7):
        """
        Stabilize the foundational aspects of an entity.
        
        This is a key function of Yesod as the Foundation Sephirah.
        
        Args:
            entity (dict): Optional entity data
            stability_level (float): Desired stability level (0-1)
            
        Returns:
            dict: Foundation stabilization results
        """
        # Validate inputs
        stability_level = min(1.0, max(0.0, stability_level))
        
        # Base stabilization values
        base_stabilization = {
            "grounding": 0.5 + 0.5 * stability_level,
            "coherence": 0.6 + 0.4 * stability_level,
            "integration": 0.4 + 0.6 * stability_level,
            "solidity": 0.55 + 0.45 * stability_level,
            "durability": 0.5 + 0.5 * stability_level
        }
        
        # Modify based on entity if provided
        if entity and isinstance(entity, dict):
            # Add current stability modification
            if "current_stability" in entity:
                current = entity["current_stability"]
                delta = stability_level - current
                
                # Harder to stabilize unstable entities
                if delta > 0:
                    if current < 0.3:  # Very unstable
                        effectiveness = 0.6
                    elif current < 0.7:  # Moderately stable
                        effectiveness = 0.8
                    else:  # Already quite stable
                        effectiveness = 0.9
                        
                    # Apply effectiveness factor
                    for key in base_stabilization:
                        base_stabilization[key] *= effectiveness
            
            # Add resonance modification
            if "yesod_resonance" in entity:
                resonance = entity["yesod_resonance"]
                for key in base_stabilization:
                    base_stabilization[key] *= 0.7 + 0.3 * resonance
        
        # Create effect duration based on stability level
        effect_duration = {
            "unit": "days",
            "value": 1 + int(7 * stability_level)  # 1-8 days
        }
        
        # Ensure values are within bounds
        for key in base_stabilization:
            base_stabilization[key] = min(1.0, max(0.0, base_stabilization[key]))
        
        return {
            "stabilization": base_stabilization,
            "duration": effect_duration
        }
    
    def establish_astral_gateway(self, direction="upward", purpose="connection", strength=0.7):
        """
        Establish an astral gateway to other dimensions or consciousness states.
        
        Yesod serves as a gateway between physical reality and higher dimensions.
        
        Args:
            direction (str): Gateway direction ('upward', 'downward', 'horizontal')
            purpose (str): Gateway purpose ('connection', 'communication', 'travel')
            strength (float): Gateway strength (0-1)
            
        Returns:
            dict: Gateway properties
        """
        # Validate inputs
        strength = min(1.0, max(0.0, strength))
        valid_directions = ["upward", "downward", "horizontal"]
        direction = direction if direction in valid_directions else "upward"
        valid_purposes = ["connection", "communication", "travel", "observation"]
        purpose = purpose if purpose in valid_purposes else "connection"
        
        # Base gateway properties
        base_gateway = {
            "stability": 0.6 + 0.4 * strength,
            "clarity": 0.5 + 0.5 * strength,
            "bandwidth": 0.4 + 0.6 * strength,
            "security": 0.7 + 0.3 * strength,
            "duration": int(30 + 90 * strength)  # 30-120 minutes
        }
        
        # Direction-specific adjustments
        if direction == "upward":  # Toward higher Sephiroth
            base_gateway["stability"] *= 0.9  # Slightly less stable
            base_gateway["clarity"] *= 1.2  # Clearer
            base_gateway["bandwidth"] *= 0.8  # Lower bandwidth
        elif direction == "downward":  # Toward Malkuth
            base_gateway["stability"] *= 1.1  # More stable
            base_gateway["clarity"] *= 0.9  # Less clear
            base_gateway["bandwidth"] *= 1.2  # Higher bandwidth
        else:  # Horizontal (to parallel dimensions)
            base_gateway["stability"] *= 0.8  # Less stable
            base_gateway["security"] *= 0.9  # Less secure
            
        # Purpose-specific adjustments
        if purpose == "connection":
            base_gateway["stability"] *= 1.1
            base_gateway["duration"] *= 1.2
        elif purpose == "communication":
            base_gateway["bandwidth"] *= 1.2
            base_gateway["clarity"] *= 1.1
        elif purpose == "travel":
            base_gateway["security"] *= 1.2
            base_gateway["stability"] *= 0.9
        elif purpose == "observation":
            base_gateway["clarity"] *= 1.3
            base_gateway["security"] *= 1.1
            base_gateway["bandwidth"] *= 0.8
        
        # Calculate destination based on direction
        destination = None
        if direction == "upward":
            if strength > 0.8:
                destination = "tiphareth"  # High strength can reach to Beauty
            elif strength > 0.6:
                destination = ["netzach", "hod"]  # Medium strength reaches Victory/Glory
            else:
                destination = ["netzach", "hod"]  # Lower strength still connects but weaker
        elif direction == "downward":
            destination = "malkuth"  # Earth dimension
        else:  # Horizontal
            destination = "parallel_dimensions"
        
        # Ensure values are within bounds
        for key in base_gateway:
            if isinstance(base_gateway[key], (int, float)) and not isinstance(base_gateway[key], bool):
                base_gateway[key] = min(1.0, max(0.0, base_gateway[key]))
        
        return {
            "direction": direction,
            "purpose": purpose,
            "strength": strength,
            "destination": destination,
            "properties": base_gateway
        }


