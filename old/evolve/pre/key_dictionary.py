"""
Key Dictionary Module

This module serves as the central repository for Platonic key definitions and
Sephirotic connections. It maps each key to its corresponding dimensions and
provides functions for retrieving the appropriate key for dimensional traversal.

Encoding in dimensional gateways functions as a mirror - information patterns
must be readable in both forward and backward directions, reflecting the
bidirectional nature of dimensional resonance.
"""

import math
from metrics_tracker import MetricsTracker

# Constants for key calculations
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # Golden ratio for sacred geometry

# Base frequencies for each Platonic key
BASE_FREQUENCIES = {
    "Tetrahedron": 528.0,  # Fire (Transformation) Hz
    "Cube": 432.0,         # Earth (Grounding) Hz
    "Octahedron": 396.0,   # Air (Liberation) Hz
    "Icosahedron": 417.0,  # Water (Flow) Hz
    "Dodecahedron": 639.0, # Aether (Connection) Hz
    "Merkaba": 963.0       # Light Vehicle (Awakening) Hz
}

# Precise Sephiroth connections for each Platonic key based on discovery
# These are the exact connections that form each Platonic solid in the Tree of Life
PLATONIC_CONNECTIONS = {
    "Tetrahedron": ["Tiphareth", "Netzach", "Hod"],
    "Cube": ["Hod", "Netzach", "Chesed", "Chokmah", "Binah", "Geburah"],
    "Octahedron": ["Binah", "Kether", "Chokmah", "Chesed", "Tiphareth", "Geburah"],
    "Icosahedron": ["Kether", "Chesed", "Geburah"],
    "Dodecahedron": ["Hod", "Netzach", "Chesed", "Daath", "Geburah"]
}

# Sephiroth dimensional properties
SEPHIROTH_PROPERTIES = {
    "Kether": {
        "title": "Crown",
        "dimension": "Divine Unity",
        "frequency": 963.0,
        "color": (1.0, 1.0, 1.0),  # White
        "element": "Pure Light",
        "core_geometry": "Point",
        "resonance_pattern": "Cosmic Oneness",
        "position": (0, 4.5, 0)  # Precise positioning (x, y, z)
    },
    "Chokmah": {
        "title": "Wisdom",
        "dimension": "Divine Masculine",
        "frequency": 852.0,
        "color": (0.8, 0.8, 0.8),  # Grey
        "element": "Water/Aether",
        "core_geometry": "Line",
        "resonance_pattern": "Expansion",
        "position": (-1.5, 3.5, 0)
    },
    "Binah": {
        "title": "Understanding",
        "dimension": "Divine Feminine",
        "frequency": 741.0,
        "color": (0.2, 0.2, 0.2),  # Black/Dark
        "element": "Water/Aether",
        "core_geometry": "Triangle",
        "resonance_pattern": "Formative",
        "position": (1.5, 3.5, 0)
    },
    "Daath": {
        "title": "Knowledge",
        "dimension": "Hidden Gateway",
        "frequency": 729.0,
        "color": (0.5, 0.0, 0.5),  # Purple (hidden)
        "element": "Void",
        "core_geometry": "Hidden Point",
        "resonance_pattern": "Dimensional Nexus",
        "position": (0, 3.0, 0)  # Hidden position
    },
    "Chesed": {
        "title": "Mercy",
        "dimension": "Compassion",
        "frequency": 639.0,
        "color": (0.0, 0.0, 1.0),  # Blue
        "element": "Water",
        "core_geometry": "Tetrahedron",
        "resonance_pattern": "Abundance",
        "position": (-2.25, 2.5, 0)
    },
    "Geburah": {
        "title": "Severity",
        "dimension": "Discipline",
        "frequency": 528.0,
        "color": (1.0, 0.0, 0.0),  # Red
        "element": "Fire",
        "core_geometry": "Octahedron",
        "resonance_pattern": "Purification",
        "position": (2.25, 2.5, 0)
    },
    "Tiphareth": {
        "title": "Beauty",
        "dimension": "Harmony",
        "frequency": 528.0,
        "color": (1.0, 1.0, 0.0),  # Yellow/Gold
        "element": "Fire/Air",
        "core_geometry": "Hexagon",
        "resonance_pattern": "Integration",
        "position": (0, 2.0, 0)
    },
    "Netzach": {
        "title": "Victory",
        "dimension": "Emotion",
        "frequency": 417.0,
        "color": (0.0, 1.0, 0.0),  # Green
        "element": "Air/Water",
        "core_geometry": "Pentagon",
        "resonance_pattern": "Inspiration",
        "position": (-2.25, 0.5, 0)
    },
    "Hod": {
        "title": "Splendor",
        "dimension": "Intellect",
        "frequency": 396.0,
        "color": (1.0, 0.65, 0.0),  # Orange
        "element": "Air/Earth",
        "core_geometry": "Square",
        "resonance_pattern": "Communication",
        "position": (2.25, 0.5, 0)
    },
    "Yesod": {
        "title": "Foundation",
        "dimension": "Subconscious",
        "frequency": 285.0,
        "color": (0.5, 0.0, 0.5),  # Purple
        "element": "Air/Earth",
        "core_geometry": "Circle",
        "resonance_pattern": "Dream",
        "position": (0, -0.5, 0)
    },
    "Malkuth": {
        "title": "Kingdom",
        "dimension": "Physical",
        "frequency": 174.0,
        "color": (0.55, 0.27, 0.07),  # Brown/Earth
        "element": "Earth",
        "core_geometry": "Cube",
        "resonance_pattern": "Manifestation",
        "position": (0, -2.0, 0)
    }
}

# Element compatibility matrix
ELEMENT_COMPATIBILITY = {
    "Fire": {
        "Fire": 1.0, "Air": 0.8, "Water": 0.4, "Earth": 0.6, 
        "Aether": 0.7, "Pure Light": 0.9, "Void": 0.5
    },
    "Earth": {
        "Fire": 0.6, "Air": 0.5, "Water": 0.8, "Earth": 1.0, 
        "Aether": 0.6, "Pure Light": 0.7, "Void": 0.7
    },
    "Air": {
        "Fire": 0.8, "Air": 1.0, "Water": 0.7, "Earth": 0.5, 
        "Aether": 0.8, "Pure Light": 0.8, "Void": 0.6
    },
    "Water": {
        "Fire": 0.4, "Air": 0.7, "Water": 1.0, "Earth": 0.8, 
        "Aether": 0.7, "Pure Light": 0.8, "Void": 0.5
    },
    "Aether": {
        "Fire": 0.7, "Air": 0.8, "Water": 0.7, "Earth": 0.6, 
        "Aether": 1.0, "Pure Light": 0.9, "Void": 0.9
    }
}

# Tracker for key dictionary operations
tracker = MetricsTracker("key_dictionary")


def get_dimensional_key(from_sephirah, to_sephirah):
    """
    Returns the appropriate Platonic key for traversing between two Sephiroth.
    
    Args:
        from_sephirah: Source Sephirah name
        to_sephirah: Target Sephirah name
        
    Returns:
        Name of the Platonic key that connects these Sephiroth
    """
    tracker.record_metric("key_lookup", f"Looking for key between {from_sephirah} and {to_sephirah}")
    
    # Check if both Sephiroth exist
    if from_sephirah not in SEPHIROTH_PROPERTIES or to_sephirah not in SEPHIROTH_PROPERTIES:
        tracker.record_metric("key_lookup_error", f"Invalid Sephirah name")
        return None
    
    # Check each key's connections to find one that includes both Sephiroth
    for key_name, connections in PLATONIC_CONNECTIONS.items():
        if from_sephirah in connections and to_sephirah in connections:
            tracker.record_metric("key_found", key_name)
            return key_name
    
    # If no direct key is found, use Merkaba as default
    tracker.record_metric("key_default", "Using Merkaba as default key")
    return "Merkaba"


def get_sephirah_properties(sephirah_name):
    """
    Returns the properties of a specific Sephirah dimension.
    
    Args:
        sephirah_name: Name of the Sephirah
        
    Returns:
        Dictionary of Sephirah properties
    """
    if sephirah_name in SEPHIROTH_PROPERTIES:
        return SEPHIROTH_PROPERTIES[sephirah_name]
    else:
        return None


def calculate_element_compatibility(element1, element2):
    """
    Calculates the compatibility between two elemental forces.
    
    Args:
        element1: First element
        element2: Second element
        
    Returns:
        Compatibility coefficient (0.0-1.0)
    """
    # Handle composite elements
    if isinstance(element1, list):
        # Calculate average compatibility with all elements in the list
        compatibilities = [calculate_element_compatibility(e, element2) for e in element1]
        return sum(compatibilities) / len(compatibilities)
        
    if isinstance(element2, list):
        # Calculate average compatibility with all elements in the list
        compatibilities = [calculate_element_compatibility(element1, e) for e in element2]
        return sum(compatibilities) / len(compatibilities)
    
    # Look up compatibility in matrix
    if element1 in ELEMENT_COMPATIBILITY and element2 in ELEMENT_COMPATIBILITY[element1]:
        return ELEMENT_COMPATIBILITY[element1][element2]
    else:
        return 0.5  # Default for unknown combinations


def create_mirrored_encoding(information):
    """
    Creates a mirrored encoding pattern that can be read in both directions.
    This reflects the bidirectional nature of dimensional gateways.
    
    Args:
        information: The information to encode
        
    Returns:
        Mirrored encoding pattern
    """
    # Convert information to string if not already
    info_str = str(information)
    
    # Create forward/backward palindromic structure
    # For complex data, we use a more sophisticated approach than simple reversal
    forward = info_str
    separator = "|"  # Dimensional gateway marker
    
    # Create reversed version with transformation for bidirectional reading
    # This isn't just a simple reverse but a phase-conjugate mirror
    backward = ""
    for char in reversed(info_str):
        # Apply phase conjugation (symbolic representation)
        # In a full implementation, this would be a more complex transformation
        if char.isalpha():
            # Rotate characters within their case (a→z, b→y, etc.)
            if char.islower():
                backward += chr(219 - ord(char))  # 219 = ord('a') + ord('z')
            else:
                backward += chr(155 - ord(char))  # 155 = ord('A') + ord('Z')
        elif char.isdigit():
            # Complement digits (0→9, 1→8, etc.)
            backward += str(9 - int(char))
        else:
            # Keep special characters as is
            backward += char
    
    # Combine with dimensional gateway separator
    mirrored = forward + separator + backward
    
    # Calculate encoding properties
    encoding_properties = {
        "original_length": len(info_str),
        "mirrored_length": len(mirrored),
        "symmetry_point": len(forward),
        "palindromic_integrity": _calculate_palindromic_integrity(mirrored)
    }
    
    return {
        "content": mirrored,
        "properties": encoding_properties
    }


def _calculate_palindromic_integrity(mirrored_content):
    """
    Calculates how well the mirrored content maintains information integrity
    when read in both directions.
    
    Args:
        mirrored_content: The mirrored content to analyze
        
    Returns:
        Integrity coefficient (0.0-1.0)
    """
    # Find separator position
    separator_pos = mirrored_content.find("|")
    if separator_pos == -1:
        return 0.0
    
    # Split into forward and backward parts
    forward = mirrored_content[:separator_pos]
    backward = mirrored_content[separator_pos+1:]
    
    # Check if backward is valid phase conjugate of forward
    if len(backward) != len(forward):
        return 0.5  # Partial integrity
    
    # Check character by character
    matches = 0
    for i in range(len(forward)):
        f_char = forward[i]
        b_char = backward[len(backward)-1-i]
        
        # Apply reverse of the phase conjugation to compare
        expected_b_char = b_char
        if b_char.isalpha():
            if b_char.islower():
                expected_b_char = chr(219 - ord(b_char))
            else:
                expected_b_char = chr(155 - ord(b_char))
        elif b_char.isdigit():
            expected_b_char = str(9 - int(b_char))
            
        if f_char == expected_b_char:
            matches += 1
    
    return matches / len(forward)


# Gateway connection strength between Sephiroth
def calculate_gateway_strength(from_sephirah, to_sephirah):
    """
    Calculates the natural gateway strength between two Sephiroth
    based on their properties and positions.
    
    Args:
        from_sephirah: Source Sephirah name
        to_sephirah: Target Sephirah name
        
    Returns:
        Gateway strength coefficient (0.0-1.0)
    """
    # Get Sephirah properties
    source = get_sephirah_properties(from_sephirah)
    target = get_sephirah_properties(to_sephirah)
    
    if not source or not target:
        return 0.0
    
    # Calculate frequency resonance
    freq_ratio = min(source["frequency"], target["frequency"]) / max(source["frequency"], target["frequency"])
    
    # Calculate elemental compatibility
    element_comp = calculate_element_compatibility(source["element"], target["element"])
    
    # Calculate spatial proximity (inverse of distance)
    source_pos = source["position"]
    target_pos = target["position"]
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(source_pos, target_pos)))
    proximity = 1.0 / (1.0 + distance)  # Normalize to 0-1 range
    
    # Calculate final gateway strength
    strength = (freq_ratio * 0.4) + (element_comp * 0.3) + (proximity * 0.3)
    
    return min(strength, 1.0)  # Cap at 1.0
