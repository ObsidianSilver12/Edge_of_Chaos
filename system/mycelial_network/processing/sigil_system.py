"""
glyph_system.py - Module for creating and encoding meaningful glyphs.

This module handles the creation of glyphs for meaningful data, encoding with
surface-level exif data and hidden steganography for deeper information.
"""

import numpy as np
import logging
import random
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GlyphSystem')

class GlyphSystem:
    """
    System for creating and encoding glyphs with metadata and hidden information.
    
    Glyphs are symbolic representations that contain both visual meaning and
    encoded metadata. They are used to emphasize meaningful data, especially for
    spiritual, metaphysical, and dimensional information.
    """
    
    def __init__(self):
        """Initialize the glyph system."""
        # Base symbols library - mapping concept categories to base symbols
        self.base_symbols = {
            'spiritual': ['circle', 'spiral', 'triangle', 'vesica_piscis', 'star', 'lotus'],
            'metaphysical': ['pentagram', 'hexagram', 'metatron', 'tree_of_life', 'yin_yang', 'infinity'],
            'dimensional': ['cube', 'tesseract', 'torus', 'merkaba', 'octahedron', 'icosahedron'],
            'energy': ['wave', 'ray', 'flame', 'lightning', 'vortex', 'aurora'],
            'elemental': ['earth', 'water', 'fire', 'air', 'aether', 'void'],
            'consciousness': ['eye', 'crown', 'brain', 'mind', 'dream', 'awakening'],
            'emotional': ['heart', 'tears', 'smile', 'embrace', 'release', 'flow'],
            'temporal': ['hourglass', 'clock', 'cycle', 'infinity', 'moment', 'eternity'],
            'creative': ['brush', 'note', 'dance', 'voice', 'inspiration', 'vision'],
            'technological': ['circuit', 'node', 'network', 'interface', 'algorithm', 'quantum']
        }
        
        # Platonic solids - for 3D glyph components
        self.platonic_forms = {
            'tetrahedron': {'vertices': 4, 'faces': 4, 'element': 'fire'},
            'hexahedron': {'vertices': 8, 'faces': 6, 'element': 'earth'},
            'octahedron': {'vertices': 6, 'faces': 8, 'element': 'air'},
            'dodecahedron': {'vertices': 20, 'faces': 12, 'element': 'aether'},
            'icosahedron': {'vertices': 12, 'faces': 20, 'element': 'water'}
        }
        
        # Sacred geometry patterns
        self.sacred_patterns = {
            'flower_of_life': {'complexity': 19, 'dimensions': 2, 'meaning': 'creation'},
            'seed_of_life': {'complexity': 7, 'dimensions': 2, 'meaning': 'beginning'},
            'tree_of_life': {'complexity': 10, 'dimensions': 2, 'meaning': 'connection'},
            'metatrons_cube': {'complexity': 13, 'dimensions': 3, 'meaning': 'balance'},
            'sri_yantra': {'complexity': 9, 'dimensions': 2, 'meaning': 'harmony'},
            'golden_spiral': {'complexity': 8, 'dimensions': 2, 'meaning': 'growth'},
            'torus': {'complexity': 6, 'dimensions': 3, 'meaning': 'flow'}
        }
        
        # Color meanings for glyph coloration
        self.color_meanings = {
            'red': {'element': 'fire', 'chakra': 'root', 'emotion': 'passion'},
            'orange': {'element': 'fire-earth', 'chakra': 'sacral', 'emotion': 'creativity'},
            'yellow': {'element': 'air', 'chakra': 'solar_plexus', 'emotion': 'confidence'},
            'green': {'element': 'earth', 'chakra': 'heart', 'emotion': 'love'},
            'blue': {'element': 'water', 'chakra': 'throat', 'emotion': 'expression'},
            'indigo': {'element': 'aether', 'chakra': 'third_eye', 'emotion': 'intuition'},
            'violet': {'element': 'spirit', 'chakra': 'crown', 'emotion': 'connection'},
            'white': {'element': 'light', 'chakra': 'transpersonal', 'emotion': 'purity'},
            'black': {'element': 'void', 'chakra': 'earth_star', 'emotion': 'mystery'},
            'gold': {'element': 'divine', 'chakra': 'god_head', 'emotion': 'enlightenment'},
            'silver': {'element': 'moon', 'chakra': 'causal', 'emotion': 'reflection'}
        }
        
        # Encoding methods for metadata
        self.encoding_methods = {
            'surface': {
                'exif': {'visibility': 'high', 'capacity': 'medium'},
                'color': {'visibility': 'high', 'capacity': 'low'},
                'pattern': {'visibility': 'high', 'capacity': 'low'},
                'shape': {'visibility': 'high', 'capacity': 'low'}
            },
            'hidden': {
                'steganography': {'visibility': 'very_low', 'capacity': 'high'},
                'frequency': {'visibility': 'low', 'capacity': 'medium'},
                'geometric_encoding': {'visibility': 'low', 'capacity': 'medium'},
                'quantum_marker': {'visibility': 'very_low', 'capacity': 'low'}
            }
        }
        
        # Track created glyphs
        self.created_glyphs = {}
        
        logger.info("Glyph system initialized")
    
    def create_glyph_for_concept(self, concept, metadata=None):
        """
        Create an appropriate glyph for a given concept.
        
        Parameters:
            concept (dict): Concept information including category, name, etc.
            metadata (dict, optional): Additional metadata to encode
            
        Returns:
            dict: Created glyph
        """
        if not metadata:
            metadata = {}
        
        # Extract key concept information
        concept_name = concept.get('name', 'unknown')
        concept_category = concept.get('category', 'spiritual')
        concept_importance = concept.get('importance', 0.5)
        concept_spiritual_level = concept.get('spiritual_level', 0.5)
        
        # Generate unique ID for the glyph
        glyph_id = self._generate_glyph_id(concept_name, concept_category)
        
        # Select appropriate base symbol
        base_symbol = self._select_base_symbol(concept_category, concept_name)
        
        # Determine complexity based on concept importance
        complexity = 0.3 + (0.7 * concept_importance)
        
        # Determine dimensionality based on spiritual level
        dimensions = 2 if concept_spiritual_level < 0.7 else 3
        
        # Select color palette
        primary_color, secondary_colors = self._select_colors(concept)
        
        # Create glyph structure
        glyph = {
            'id': glyph_id,
            'concept_name': concept_name,
            'concept_category': concept_category,
            'base_symbol': base_symbol,
            'complexity': complexity,
            'dimensions': dimensions,
            'primary_color': primary_color,
            'secondary_colors': secondary_colors,
            'creation_timestamp': datetime.now(),
            'components': [],
            'sacred_geometry': [],
            'platonic_forms': [],
            'encoded_metadata': {},
            'steganographic_data': {}
        }
        
        # Add sacred geometry components
        glyph['sacred_geometry'] = self._add_sacred_geometry(concept, complexity)
        
        # Add platonic forms for 3D glyphs
        if dimensions == 3:
            glyph['platonic_forms'] = self._add_platonic_forms(concept)
        
        # Create visual components based on concept attributes
        glyph['components'] = self._create_glyph_components(concept, complexity, dimensions)
        
        # Encode surface metadata
        glyph['encoded_metadata'] = self._encode_surface_metadata(concept, metadata)
        
        # Store glyph
        self.created_glyphs[glyph_id] = glyph
        
        logger.info(f"Created glyph {glyph_id} for concept {concept_name} ({concept_category})")
        
        return glyph
    
    def _generate_glyph_id(self, concept_name, concept_category):
        """Generate a unique ID for a glyph."""
        # Create a base string using concept name, category, and timestamp
        base_string = f"{concept_name}:{concept_category}:{datetime.now().isoformat()}"
        
        # Generate hash
        hash_obj = hashlib.sha256(base_string.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Use first 12 characters for ID
        return f"glyph_{hash_hex[:12]}"
    
    def _select_base_symbol(self, concept_category, concept_name):
        """Select an appropriate base symbol for the concept."""
        # Check if category exists
        if concept_category in self.base_symbols:
            # Get symbols for this category
            category_symbols = self.base_symbols[concept_category]
            
            # Try to find a symbol that matches part of the concept name
            matching_symbols = [
                symbol for symbol in category_symbols 
                if symbol in concept_name or concept_name in symbol
            ]
            
            if matching_symbols:
                return random.choice(matching_symbols)
            else:
                # Otherwise pick a random symbol from the category
                return random.choice(category_symbols)
        else:
            # Default to a spiritual symbol if category not found
            return random.choice(self.base_symbols['spiritual'])
    
    def _select_colors(self, concept):
        """Select appropriate colors for the glyph based on concept attributes."""
        # Get concept attributes that might influence color
        category = concept.get('category', 'spiritual')
        emotion = concept.get('emotion', None)
        element = concept.get('element', None)
        chakra = concept.get('chakra', None)
        
        primary_color = None
        
        # Try to select primary color based on concept attributes
        if chakra and any(chakra == attrs['chakra'] for color, attrs in self.color_meanings.items()):
            primary_color = next(color for color, attrs in self.color_meanings.items() 
                                if attrs['chakra'] == chakra)
        elif element and any(element == attrs['element'] for color, attrs in self.color_meanings.items()):
            primary_color = next(color for color, attrs in self.color_meanings.items() 
                                if attrs['element'] == element)
        elif emotion and any(emotion == attrs['emotion'] for color, attrs in self.color_meanings.items()):
            primary_color = next(color for color, attrs in self.color_meanings.items() 
                                if attrs['emotion'] == emotion)
        else:
            # Category-based color selection
            if category == 'spiritual':
                primary_color = random.choice(['violet', 'indigo', 'gold'])
            elif category == 'metaphysical':
                primary_color = random.choice(['indigo', 'violet', 'white'])
            elif category == 'dimensional':
                primary_color = random.choice(['blue', 'indigo', 'violet'])
            elif category == 'energy':
                primary_color = random.choice(['yellow', 'orange', 'red'])
            elif category == 'elemental':
                primary_color = random.choice(['green', 'blue', 'red', 'white', 'brown'])
            elif category == 'consciousness':
                primary_color = random.choice(['indigo', 'violet', 'white'])
            elif category == 'emotional':
                primary_color = random.choice(['green', 'pink', 'blue'])
            elif category == 'temporal':
                primary_color = random.choice(['blue', 'silver', 'black'])
            elif category == 'creative':
                primary_color = random.choice(['orange', 'yellow', 'pink'])
            elif category == 'technological':
                primary_color = random.choice(['blue', 'silver', 'green'])
            else:
                primary_color = random.choice(list(self.color_meanings.keys()))
        
        # Select complementary colors for secondary colors
        color_wheel = list(self.color_meanings.keys())
        try:
            primary_index = color_wheel.index(primary_color)
            
            # Choose complementary colors (opposite and adjacent)
            opposite_index = (primary_index + len(color_wheel) // 2) % len(color_wheel)
            adjacent1_index = (primary_index + 1) % len(color_wheel)
            adjacent2_index = (primary_index - 1) % len(color_wheel)
            
            secondary_colors = [
                color_wheel[opposite_index],
                color_wheel[adjacent1_index],
                color_wheel[adjacent2_index]
            ]
        except ValueError:
            # Fallback if primary color isn't in the wheel
            secondary_colors = random.sample(color_wheel, 3)
        
        return primary_color, secondary_colors
    
    def _add_sacred_geometry(self, concept, complexity):
        """Add sacred geometry patterns to the glyph."""
        spiritual_level = concept.get('spiritual_level', 0.5)
        
        # Determine number of sacred geometry patterns based on complexity and spiritual level
        pattern_count = max(1, min(3, int(1 + complexity * 2 + spiritual_level)))
        
        # Select patterns
        selected_patterns = []
        available_patterns = list(self.sacred_patterns.keys())
        
        for _ in range(pattern_count):
            if available_patterns:
                # Choose a pattern that matches the complexity level
                suitable_patterns = [
                    p for p in available_patterns
                    if abs(self.sacred_patterns[p]['complexity'] / 20 - complexity) < 0.3
                ]
                
                if not suitable_patterns:
                    suitable_patterns = available_patterns
                
                pattern = random.choice(suitable_patterns)
                pattern_data = self.sacred_patterns[pattern].copy()
                
                # Add scaling factor and rotation
                pattern_data['scale'] = 0.5 + (0.5 * random.random())
                pattern_data['rotation'] = random.random() * 360
                pattern_data['pattern_name'] = pattern
                
                selected_patterns.append(pattern_data)
                
                # Remove to avoid duplication (optional)
                if pattern in available_patterns:
                    available_patterns.remove(pattern)
        
        return selected_patterns
    
    def _add_platonic_forms(self, concept):
        """Add platonic solids for 3D glyphs."""
        element = concept.get('element', None)
        complexity = concept.get('complexity', 0.5)
        
        # Determine number of platonic forms based on complexity
        form_count = max(1, min(3, int(1 + complexity * 2)))
        
        # Select forms
        selected_forms = []
        
        # Try to match by element first
        if element:
            matching_forms = [
                form for form, data in self.platonic_forms.items()
                if data['element'] == element
            ]
            
            if matching_forms:
                selected_forms.append(random.choice(matching_forms))
                form_count -= 1
        
        # Add additional forms
        available_forms = list(self.platonic_forms.keys())
        for _ in range(form_count):
            if available_forms:
                form = random.choice(available_forms)
                form_data = self.platonic_forms[form].copy()
                
                # Add scaling factor and rotation
                form_data['scale'] = 0.5 + (0.5 * random.random())
                form_data['rotation'] = [
                    random.random() * 360,
                    random.random() * 360,
                    random.random() * 360
                ]
                form_data['form_name'] = form
                
                selected_forms.append(form_data)
                
                # Remove to avoid duplication
                if form in available_forms:
                    available_forms.remove(form)
        
        return selected_forms
    
    def _create_glyph_components(self, concept, complexity, dimensions):
        """Create visual components for the glyph based on concept attributes."""
        components = []
        
        # Get concept attributes
        attributes = concept.get('attributes', {})
        
        # Calculate number of components based on complexity
        component_count = max(3, min(12, int(5 + complexity * 10)))
        
        # Component types
        component_types = ['point', 'line', 'curve', 'circle', 'polygon', 'spiral']
        if dimensions == 3:
            component_types.extend(['sphere', 'cylinder', 'cone', 'torus'])
        
        # Create components
        for i in range(component_count):
            # Select component type
            component_type = random.choice(component_types)
            
            # Create base component
            component = {
                'type': component_type,
                'position': self._generate_position(i, component_count, dimensions),
                'scale': 0.2 + (0.8 * random.random()),
                'rotation': random.random() * 360 if dimensions == 2 else [
                    random.random() * 360,
                    random.random() * 360,
                    random.random() * 360
                ],
                'color': random.choice([concept.get('primary_color', 'white')] + 
                                     concept.get('secondary_colors', ['blue', 'green']))
            }
            
            # Add type-specific properties
            if component_type == 'point':
                component['size'] = 0.02 + (0.08 * random.random())
            
            elif component_type == 'line':
                component['length'] = 0.2 + (0.8 * random.random())
                component['thickness'] = 0.01 + (0.05 * random.random())
                component['start_point'] = component['position']
                component['end_point'] = self._generate_position(i+1, component_count, dimensions)
            
            elif component_type == 'curve':
                component['control_points'] = [
                    self._generate_position(i+j, component_count, dimensions)
                    for j in range(3)
                ]
                component['thickness'] = 0.01 + (0.05 * random.random())
            
            elif component_type == 'circle':
                component['radius'] = 0.1 + (0.4 * random.random())
                component['fill'] = random.random() > 0.5
                component['thickness'] = 0.01 + (0.03 * random.random())
            
            elif component_type == 'polygon':
                component['sides'] = random.choice([3, 4, 5, 6, 7, 8, 9, 10, 12])
                component['radius'] = 0.1 + (0.3 * random.random())
                component['fill'] = random.random() > 0.5
                component['thickness'] = 0.01 + (0.03 * random.random())
            
            elif component_type == 'spiral':
                component['turns'] = 1 + int(3 * random.random())
                component['expansion'] = 0.05 + (0.2 * random.random())
                component['thickness'] = 0.01 + (0.03 * random.random())
            
            elif component_type == 'sphere':
                component['radius'] = 0.1 + (0.3 * random.random())
                component['fill'] = random.random() > 0.3
                component['resolution'] = 8 + int(16 * random.random())
            
            elif component_type == 'cylinder':
                component['radius'] = 0.1 + (0.2 * random.random())
                component['height'] = 0.2 + (0.4 * random.random())
                component['fill'] = random.random() > 0.3
            
            elif component_type == 'cone':
                component['base_radius'] = 0.1 + (0.2 * random.random())
                component['height'] = 0.2 + (0.4 * random.random())
                component['fill'] = random.random() > 0.3
            
            elif component_type == 'torus':
                component['major_radius'] = 0.15 + (0.2 * random.random())
                component['minor_radius'] = 0.05 + (0.1 * random.random())
                component['fill'] = random.random() > 0.3
            
            # Add component to list
            components.append(component)
        
        return components
    
    def _generate_position(self, index, total, dimensions):
        """Generate a position for a component based on index and total components."""
        # Create a balanced distribution of components
        if dimensions == 2:
            # Use polar coordinates for 2D
            angle = 2 * math.pi * index / total
            radius = 0.2 + (0.8 * random.random())
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            return [x, y]
        else:
            # Use spherical coordinates for 3D
            phi = 2 * math.pi * index / total
            theta = math.pi * random.random()
            radius = 0.2 + (0.8 * random.random())
            
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            
            return [x, y, z]
    
    def _encode_surface_metadata(self, concept, metadata):
        """Encode surface-level metadata in the glyph."""
        # Start with basic concept metadata
        encoded_metadata = {
            'concept_name': concept.get('name', 'unknown'),
            'concept_category': concept.get('category', 'unknown'),
            'creation_date': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Add essential metadata for search
        if 'description' in concept:
            encoded_metadata['description'] = concept['description']
        
        if 'keywords' in concept:
            encoded_metadata['keywords'] = concept['keywords']
        
        if 'creator' in concept:
            encoded_metadata['creator'] = concept['creator']
        
        # Add additional metadata
        for key, value in metadata.items():
            encoded_metadata[key] = value
        
        return encoded_metadata
    
    def encode_steganographic_data(self, glyph_id, hidden_data):
        """
        Encode hidden steganographic data in the glyph.
        
        Parameters:
            glyph_id (str): ID of the glyph to encode data into
            hidden_data (dict): Data to hide within the glyph
            
        Returns:
            bool: True if encoding successful
        """
        if glyph_id not in self.created_glyphs:
            logger.warning(f"Cannot encode steganographic data: glyph {glyph_id} not found")
            return False
        
        glyph = self.created_glyphs[glyph_id]
        
        # Create encoding structure
        encoding = {
            'encoding_method': 'steganography',
            'encoding_timestamp': datetime.now().isoformat(),
            'data': {}
        }
        
        # Process each data item for encoding
        for key, value in hidden_data.items():
            # Generate a unique encoding pattern based on the key and value
            hash_base = f"{key}:{str(value)}:{random.random()}"
            hash_obj = hashlib.sha256(hash_base.encode())
            encoding_pattern = hash_obj.hexdigest()
            
            # Store the data with its encoding pattern
            encoding['data'][key] = {
                'value': value,
                'encoding_pattern': encoding_pattern
            }
        
        # Store encoded data in the glyph
        glyph['steganographic_data'] = encoding
        
        logger.info(f"Encoded steganographic data into glyph {glyph_id}")
        return True
    
    def decode_steganographic_data(self, glyph_id):
        """
        Decode hidden steganographic data from a glyph.
        
        Parameters:
            glyph_id (str): ID of the glyph to decode data from
            
        Returns:
            dict: Decoded hidden data
        """
        if glyph_id not in self.created_glyphs:
            logger.warning(f"Cannot decode steganographic data: glyph {glyph_id} not found")
            return {}
        
        glyph = self.created_glyphs[glyph_id]
        
        # Check if glyph has steganographic data
        if 'steganographic_data' not in glyph or 'data' not in glyph['steganographic_data']:
            logger.warning(f"No steganographic data found in glyph {glyph_id}")
            return {}
        
        # Extract the hidden data
        decoded_data = {}
        for key, encoding in glyph['steganographic_data']['data'].items():
            decoded_data[key] = encoding['value']
        
        return decoded_data
    
    def render_glyph_ascii(self, glyph_id):
        """
        Render a simple ASCII representation of the glyph for visualization.
        
        Parameters:
            glyph_id (str): ID of the glyph to render
            
        Returns:
            str: ASCII representation of the glyph
        """
        if glyph_id not in self.created_glyphs:
            return "Glyph not found"
        
        glyph = self.created_glyphs[glyph_id]
        
        # Create a simple ASCII representation based on glyph type
        base_symbol = glyph['base_symbol']
        ascii_art = ""
        
        # Top border
        ascii_art += "+------ Glyph: " + glyph['concept_name'] + " ------+\n"
        
        # Symbol representation
        if base_symbol in ['circle', 'sphere']:
            ascii_art += "          @@@@          \n"
            ascii_art += "        @@@@@@@@        \n"
            ascii_art += "      @@@@@@@@@@@@      \n"
            ascii_art += "      @@@@@@@@@@@@      \n"
            ascii_art += "      @@@@@@@@@@@@      \n"
            ascii_art += "        @@@@@@@@        \n"
            ascii_art += "          @@@@          \n"
        
        elif base_symbol in ['triangle']:
            ascii_art += "           /\\           \n"
            ascii_art += "          /  \\          \n"
            ascii_art += "         /    \\         \n"
            ascii_art += "        /      \\        \n"
            ascii_art += "       /        \\       \n"
            ascii_art += "      /          \\      \n"
            ascii_art += "     /____________\\     \n"
        
        elif base_symbol in ['star', 'pentagram']:
            ascii_art += "           *            \n"
            ascii_art += "          / \\           \n"
            ascii_art += "         /   \\          \n"
            ascii_art += "    *---*-----*---*     \n"
            ascii_art += "     \\ /       \\ /      \n"
            ascii_art += "      *---------*       \n"
        
        elif base_symbol in ['spiral']:
            ascii_art += "          @@@@          \n"
            ascii_art += "        @@    @@        \n"
            ascii_art += "       @   @@   @       \n"
            ascii_art += "       @  @  @  @       \n"
            ascii_art += "       @   @@   @       \n"
            ascii_art += "        @@    @@        \n"
            ascii_art += "          @@@@          \n"
        
        else:
            ascii_art += "      [" + base_symbol + "]       \n"
            ascii_art += "                          \n"
            ascii_art += "                          \n"
            ascii_art += "                          \n"
        
        # Information footer
        ascii_art += "+---------------------------+\n"
        ascii_art += "Type: " + glyph['concept_category'] + "\n"
        ascii_art += "Dim: " + str(glyph['dimensions']) + "D\n"
        ascii_art += "Color: " + glyph['primary_color'] + "\n"
        
        return ascii_art
    
    def get_glyph_metadata(self, glyph_id):
        """
        Get complete metadata for a glyph.
        
        Parameters:
            glyph_id (str): ID of the glyph
            
        Returns:
            dict: Complete glyph metadata
        """
        if glyph_id not in self.created_glyphs:
            logger.warning(f"Glyph {glyph_id} not found")
            return {}
        
        glyph = self.created_glyphs[glyph_id]
        
        # Create metadata summary
        metadata = {
            'id': glyph_id,
            'concept': {
                'name': glyph['concept_name'],
                'category': glyph['concept_category']
            },
            'visual': {
                'base_symbol': glyph['base_symbol'],
                'dimensions': glyph['dimensions'],
                'complexity': glyph['complexity'],
                'colors': {
                    'primary': glyph['primary_color'],
                    'secondary': glyph['secondary_colors']
                }
            },
            'structure': {
                'sacred_geometry_count': len(glyph.get('sacred_geometry', [])),
                'platonic_forms_count': len(glyph.get('platonic_forms', [])),
                'component_count': len(glyph.get('components', []))
            },
            'metadata': {
                'surface': glyph.get('encoded_metadata', {}),
                'has_hidden_data': 'steganographic_data' in glyph and len(glyph['steganographic_data']) > 0
            },
            'creation_info': {
                'timestamp': glyph.get('creation_timestamp', datetime.now()).isoformat()
            }
        }
        
        return metadata
    
    def search_glyphs(self, search_criteria):
        """
        Search for glyphs matching specific criteria.
        
        Parameters:
            search_criteria (dict): Search criteria
            
        Returns:
            list: Matching glyph IDs
        """
        matching_glyphs = []
        
        for glyph_id, glyph in self.created_glyphs.items():
            matches = True
            
            # Check each search criterion
            for key, value in search_criteria.items():
                if key == 'concept_name':
                    if value.lower() not in glyph['concept_name'].lower():
                        matches = False
                        break
                
                elif key == 'concept_category':
                    if value != glyph['concept_category']:
                        matches = False
                        break
                
                elif key == 'base_symbol':
                    if value != glyph['base_symbol']:
                        matches = False
                        break
                
                elif key == 'dimensions':
                    if value != glyph['dimensions']:
                        matches = False
                        break
                
                elif key == 'min_complexity':
                    if value > glyph['complexity']:
                        matches = False
                        break
                
                elif key == 'max_complexity':
                    if value < glyph['complexity']:
                        matches = False
                        break
                
                elif key == 'color':
                    if value != glyph['primary_color'] and value not in glyph['secondary_colors']:
                        matches = False
                        break
                
                elif key == 'has_sacred_geometry':
                    has_geometry = len(glyph.get('sacred_geometry', [])) > 0
                    if value != has_geometry:
                        matches = False
                        break
                
                elif key == 'has_platonic_forms':
                    has_forms = len(glyph.get('platonic_forms', [])) > 0
                    if value != has_forms:
                        matches = False
                        break
                
                elif key == 'has_hidden_data':
                    has_hidden_data = 'steganographic_data' in glyph and len(glyph['steganographic_data']) > 0
                    if value != has_hidden_data:
                        matches = False
                        break
                
                elif key == 'metadata_key':
                    if 'encoded_metadata' not in glyph or value not in glyph['encoded_metadata']:
                        matches = False
                        break
                
                elif key == 'metadata_value':
                    if 'encoded_metadata' not in glyph:
                        matches = False
                        break
                    
                    found = False
                    for meta_value in glyph['encoded_metadata'].values():
                        if isinstance(meta_value, str) and value in meta_value:
                            found = True
                            break
                    
                    if not found:
                        matches = False
                        break
            
            # If all criteria matched, add to results
            if matches:
                matching_glyphs.append(glyph_id)
        
        return matching_glyphs


# Factory function to create GlyphSystem
def create_glyph_system():
    """
    Create a new glyph system.
    
    Returns:
        GlyphSystem: A new glyph system instance
    """
    logger.info("Creating new glyph system")
    return GlyphSystem()