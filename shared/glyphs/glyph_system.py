# glyph_system.py
"""
Redesigned Glyph System

Creates two types of glyphs:
1. Gateway Key Glyphs: Double circle stargate with Sephirah sigils, platonic shapes, and key sigils
2. Normal Glyphs: Double circle stargate with sacred geometry and sigils (up to 3)

All glyphs are black line art that generate images and store to proper folders for encoding.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import math
import uuid

# Import our libraries
from shared.dictionaries.sigils_code import create_sigils_dictionary
from shared.tools.encode import create_image_encoder

# Import sacred geometry and platonic functions
try:
    from shared.sacred_geometry.seed_of_life import SeedOfLife
    from shared.sacred_geometry.flower_of_life import FlowerOfLife
    from shared.sacred_geometry.fruit_of_life import get_base_glyph_elements as get_fruit_elements
    from shared.sacred_geometry.tree_of_life import TreeOfLife # Assuming class with the method does not exist
    from shared.sacred_geometry.metatrons_cube import MetatronsCube
    from shared.sacred_geometry.sri_yantra import SriYantra
    from shared.sacred_geometry.vesica_piscis import get_base_glyph_elements as get_vesica_elements
    from shared.sacred_geometry.egg_of_life import get_base_glyph_elements as get_egg_elements
    from shared.sacred_geometry.germ_of_life import get_base_glyph_elements as get_germ_elements
    from shared.sacred_geometry.vector_equilibrium import get_base_glyph_elements as get_ve_elements
    from shared.sacred_geometry.star_tetrahedron import get_base_glyph_elements as get_star_tetra_elements
    from shared.sacred_geometry.merkaba import Merkaba

    # Platonic Solids
    from shared.platonics.tetrahedron import get_base_glyph_elements as get_tetra_elements
    from shared.platonics.hexahedron import get_base_glyph_elements as get_hexa_elements
    from shared.platonics.octahedron import get_base_glyph_elements as get_octa_elements
    from shared.platonics.dodecahedron import get_base_glyph_elements as get_dodeca_elements
    from shared.platonics.icosahedron import get_base_glyph_elements as get_icosa_elements
    
    GEOMETRY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import geometry modules: {e}")
    GEOMETRY_AVAILABLE = False

logger = logging.getLogger('GlyphSystem')

class GlyphSystem:
    """
    Redesigned Glyph System for creating stargate-style glyphs
    
    Creates two types:
    - Gateway Key Glyphs: For Sephirah connections with keys
    - Normal Glyphs: For general purpose with sacred geometry
    """
    
    def __init__(self):
        """Initialize the glyph system"""
        self.sigils_dict = create_sigils_dictionary()
        self.encoder = create_image_encoder()
        
        # Create required directories
        self.to_encode_path = Path("shared/assets/to_encode")
        self.encoded_path = Path("shared/assets/encoded")
        self.sephirah_path = Path("shared/assets/sephirah")
        
        for path in [self.to_encode_path, self.encoded_path, self.sephirah_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Sephirah data
        self.sephirah_data = {
            'keter': {'color': '#FFFFFF', 'element': 'divine_light', 'position': (0, 3)},
            'chokmah': {'color': '#C0C0C0', 'element': 'wisdom', 'position': (2.5, 2)},
            'binah': {'color': '#000000', 'element': 'understanding', 'position': (-2.5, 2)},
            'chesed': {'color': '#0000FF', 'element': 'mercy', 'position': (2.5, 0)},
            'geburah': {'color': '#FF0000', 'element': 'severity', 'position': (-2.5, 0)},
            'tipheret': {'color': '#FFFF00', 'element': 'beauty', 'position': (0, 0)},
            'netzach': {'color': '#00FF00', 'element': 'victory', 'position': (2.5, -2)},
            'hod': {'color': '#FFA500', 'element': 'splendor', 'position': (-2.5, -2)},
            'yesod': {'color': '#800080', 'element': 'foundation', 'position': (0, -2)},
            'malkuth': {'color': '#8B4513', 'element': 'kingdom', 'position': (0, -4)}
        }
        
        # Platonic solids with element associations
        self.platonic_elements = {
            'tetrahedron': 'fire',
            'hexahedron': 'earth', 
            'octahedron': 'air',
            'icosahedron': 'water',
            'dodecahedron': 'aether'
        }
        
        # Sacred geometry patterns
        self.sacred_geometry_patterns = [
            'seed_of_life', 'flower_of_life', 'tree_of_life', 
            'metatrons_cube', 'vesica_piscis'
        ]
        
        # Track created glyphs
        self.created_glyphs = {}
        
        logger.info("Redesigned Glyph System initialized")
    
    def create_gateway_key_glyph(self, key_concept: Dict[str, Any], sephirah_connections: List[str]) -> str:
        """
        Create a Gateway Key Glyph with stargate structure
        
        Args:
            key_concept: Information about the key (name, purpose, etc.)
            sephirah_connections: List of Sephirah names this key connects to
            
        Returns:
            Glyph ID for the created gateway key glyph
        """
        glyph_id = f"gateway_key_{uuid.uuid4().hex[:8]}"
        
        # Create the glyph
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw the stargate (two circles)
        outer_circle = patches.Circle((0, 0), 4.5, fill=False, edgecolor='black', linewidth=3)
        inner_circle = patches.Circle((0, 0), 3.0, fill=False, edgecolor='black', linewidth=3)
        
        # Fill outer ring black, inner circle white
        outer_ring = patches.Circle((0, 0), 4.5, fill=True, facecolor='black')
        inner_white = patches.Circle((0, 0), 3.0, fill=True, facecolor='white')
        
        ax.add_patch(outer_ring)
        ax.add_patch(inner_white)
        ax.add_patch(outer_circle)
        ax.add_patch(inner_circle)
        
        # Place Sephirah sigils between outer and inner circles
        sephirah_sigils = []
        if sephirah_connections:
            angle_step = 2 * math.pi / len(sephirah_connections)
            radius = 3.75  # Between circles
            
            for i, sephirah in enumerate(sephirah_connections):
                if sephirah in self.sephirah_data:
                    angle = i * angle_step
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    
                    # Get unique sigil for this Sephirah
                    sigil = self.sigils_dict.get_sigil('sacred_geometry')
                    if sigil:
                        sephirah_sigils.append(sigil)
                        ax.text(x, y, sigil, fontsize=24, ha='center', va='center', 
                               color='white', weight='bold')
        
        # Place platonic shape in inner circle
        platonic_shape = None
        if GEOMETRY_AVAILABLE:
            # Select platonic based on key concept
            element = key_concept.get('element', 'aether')
            for platonic, plat_element in self.platonic_elements.items():
                if plat_element == element:
                    platonic_shape = platonic
                    break
            
            if not platonic_shape:
                platonic_shape = 'dodecahedron'  # Default
            
            # Draw platonic shape (simplified)
            self._draw_platonic_shape(ax, platonic_shape, center=(0, 0), size=1.5)
        
        # Place key sigil in center of platonic
        key_sigil = self.sigils_dict.get_sigil('mystical')
        if key_sigil:
            ax.text(0, 0, key_sigil, fontsize=32, ha='center', va='center', 
                   color='black', weight='bold')
        
        # Save glyph image
        image_filename = f"{glyph_id}.png"
        image_path = self.to_encode_path / image_filename
        
        plt.savefig(image_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Store glyph data
        glyph_data = {
            'id': glyph_id,
            'type': 'gateway_key',
            'concept': key_concept,
            'sephirah_connections': sephirah_connections,
            'sephirah_sigils': sephirah_sigils,
            'platonic_shape': platonic_shape,
            'key_sigil': key_sigil,
            'image_path': str(image_path),
            'creation_time': datetime.now().isoformat()
        }
        
        self.created_glyphs[glyph_id] = glyph_data
        
        logger.info(f"Created gateway key glyph {glyph_id} for {key_concept.get('name', 'unknown')}")
        return glyph_id
    
    def create_normal_glyph(self, concept: Dict[str, Any], complexity: float = 0.5) -> str:
        """
        Create a normal glyph with stargate structure
        
        Args:
            concept: Information about the concept
            complexity: Complexity level (0.0-1.0) determines number of sigils
            
        Returns:
            Glyph ID for the created normal glyph
        """
        glyph_id = f"normal_glyph_{uuid.uuid4().hex[:8]}"
        
        # Create the glyph
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw the stargate (two circles) - all black lines
        outer_circle = patches.Circle((0, 0), 4.5, fill=False, edgecolor='black', linewidth=3)
        inner_circle = patches.Circle((0, 0), 3.0, fill=False, edgecolor='black', linewidth=3)
        
        ax.add_patch(outer_circle)
        ax.add_patch(inner_circle)
        
        # Place sacred geometry at top between circles
        geometry_pattern = None
        if GEOMETRY_AVAILABLE:
            geometry_pattern = self._select_sacred_geometry(concept)
            if geometry_pattern:
                self._draw_sacred_geometry(ax, geometry_pattern, position=(0, 3.75), size=0.8)
        
        # Place platonic shape in inner circle
        platonic_shape = self._select_platonic_shape(concept)
        if platonic_shape and GEOMETRY_AVAILABLE:
            self._draw_platonic_shape(ax, platonic_shape, center=(0, 0), size=1.5)
        
        # Place sigils in center (1-3 based on complexity)
        num_sigils = min(3, max(1, int(complexity * 3) + 1))
        center_sigils = []
        
        for i in range(num_sigils):
            sigil = self.sigils_dict.get_sigil('mystical')
            if sigil:
                center_sigils.append(sigil)
                
                # Position sigils
                if num_sigils == 1:
                    x, y = 0, 0
                elif num_sigils == 2:
                    x, y = (-0.3 + i * 0.6), 0
                else:  # 3 sigils
                    angle = i * (2 * math.pi / 3) - math.pi/2
                    x = 0.4 * math.cos(angle)
                    y = 0.4 * math.sin(angle)
                
                ax.text(x, y, sigil, fontsize=24, ha='center', va='center', 
                       color='black', weight='bold')
        
        # Save glyph image
        image_filename = f"{glyph_id}.png"
        image_path = self.to_encode_path / image_filename
        
        plt.savefig(image_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Store glyph data
        glyph_data = {
            'id': glyph_id,
            'type': 'normal',
            'concept': concept,
            'complexity': complexity,
            'sacred_geometry': geometry_pattern,
            'platonic_shape': platonic_shape,
            'center_sigils': center_sigils,
            'image_path': str(image_path),
            'creation_time': datetime.now().isoformat()
        }
        
        self.created_glyphs[glyph_id] = glyph_data
        
        logger.info(f"Created normal glyph {glyph_id} for {concept.get('name', 'unknown')}")
        return glyph_id
    
    def encode_glyph(self, glyph_id: str, metadata: Dict[str, Any], hidden_data: str) -> str:
        """
        Encode a glyph with EXIF metadata and steganography, then move to encoded folder
        
        Args:
            glyph_id: ID of glyph to encode
            metadata: Metadata for EXIF encoding
            hidden_data: Data for steganographic encoding
            
        Returns:
            Path to encoded image
        """
        if glyph_id not in self.created_glyphs:
            raise ValueError(f"Glyph {glyph_id} not found")
        
        glyph = self.created_glyphs[glyph_id]
        source_path = glyph['image_path']
        
        # Create encoded filename
        encoded_filename = f"encoded_{glyph_id}.png"
        encoded_path = self.encoded_path / encoded_filename
        
        # Encode the image
        self.encoder.encode_full_data(source_path, metadata, hidden_data, str(encoded_path))
        
        # Update glyph data
        glyph['encoded_path'] = str(encoded_path)
        glyph['encoded_metadata'] = metadata
        glyph['encoding_time'] = datetime.now().isoformat()
        
        logger.info(f"Encoded glyph {glyph_id} to {encoded_path}")
        return str(encoded_path)
    
    def _draw_sacred_geometry(self, ax, pattern: str, position: Tuple[float, float], size: float):
        """Draw sacred geometry pattern at specified position"""
        if not GEOMETRY_AVAILABLE:
            return
        
        try:
            x, y = position
            
            if pattern == 'seed_of_life':
                # Simple 7-circle pattern
                center_radius = size * 0.15
                for i in range(6):
                    angle = i * math.pi / 3
                    circle_x = x + size * 0.3 * math.cos(angle)
                    circle_y = y + size * 0.3 * math.sin(angle)
                    circle = patches.Circle((circle_x, circle_y), center_radius, 
                                          fill=False, edgecolor='black', linewidth=1)
                    ax.add_patch(circle)
                
                # Center circle
                center_circle = patches.Circle((x, y), center_radius, 
                                             fill=False, edgecolor='black', linewidth=1)
                ax.add_patch(center_circle)
            
            elif pattern == 'flower_of_life':
                # Extended flower pattern
                center_radius = size * 0.1
                for ring in range(2):
                    for i in range(6 if ring == 0 else 12):
                        angle = i * math.pi / (3 if ring == 0 else 6)
                        radius = size * (0.2 if ring == 0 else 0.35)
                        circle_x = x + radius * math.cos(angle)
                        circle_y = y + radius * math.sin(angle)
                        circle = patches.Circle((circle_x, circle_y), center_radius, 
                                              fill=False, edgecolor='black', linewidth=1)
                        ax.add_patch(circle)
            
            elif pattern == 'vesica_piscis':
                # Two overlapping circles
                radius = size * 0.3
                circle1 = patches.Circle((x - radius/2, y), radius, 
                                       fill=False, edgecolor='black', linewidth=1)
                circle2 = patches.Circle((x + radius/2, y), radius, 
                                       fill=False, edgecolor='black', linewidth=1)
                ax.add_patch(circle1)
                ax.add_patch(circle2)
            
            # Add other patterns as needed
            
        except Exception as e:
            logger.warning(f"Could not draw sacred geometry {pattern}: {e}")
    
    def _draw_platonic_shape(self, ax, shape: str, center: Tuple[float, float], size: float):
        """Draw simplified platonic shape"""
        if not GEOMETRY_AVAILABLE:
            return
        
        try:
            x, y = center
            
            if shape == 'tetrahedron':
                # Triangle
                height = size * math.sqrt(3) / 2
                points = np.array([
                    [x, y + height * 2/3],
                    [x - size/2, y - height/3],
                    [x + size/2, y - height/3],
                    [x, y + height * 2/3]
                ])
                ax.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
            
            elif shape == 'hexahedron':
                # Square/cube
                square = patches.Rectangle((x - size/2, y - size/2), size, size, 
                                         fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(square)
            
            elif shape == 'octahedron':
                # Diamond
                points = np.array([
                    [x, y + size/2],
                    [x + size/2, y],
                    [x, y - size/2],
                    [x - size/2, y],
                    [x, y + size/2]
                ])
                ax.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
            
            elif shape == 'dodecahedron':
                # Pentagon
                angles = np.linspace(0, 2*math.pi, 6)
                points = np.array([[x + size/2 * math.cos(a), y + size/2 * math.sin(a)] for a in angles])
                ax.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
            
            elif shape == 'icosahedron':
                # Star pattern
                outer_angles = np.linspace(0, 2*math.pi, 6)
                inner_angles = np.linspace(math.pi/5, 2*math.pi + math.pi/5, 6)
                
                for i in range(5):
                    x1 = x + size/2 * math.cos(outer_angles[i])
                    y1 = y + size/2 * math.sin(outer_angles[i])
                    x2 = x + size/4 * math.cos(inner_angles[i])
                    y2 = y + size/4 * math.sin(inner_angles[i])
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
            
        except Exception as e:
            logger.warning(f"Could not draw platonic shape {shape}: {e}")
    
    def _select_sacred_geometry(self, concept: Dict[str, Any]) -> Optional[str]:
        """Select appropriate sacred geometry based on concept"""
        category = concept.get('category', 'spiritual')
        
        if category in ['spiritual', 'consciousness']:
            return 'seed_of_life'
        elif category in ['creation', 'creative']:
            return 'flower_of_life'
        elif category in ['connection', 'relationship']:
            return 'vesica_piscis'
        else:
            import random
            return random.choice(self.sacred_geometry_patterns)
    
    def _select_platonic_shape(self, concept: Dict[str, Any]) -> Optional[str]:
        """Select appropriate platonic shape based on concept"""
        element = concept.get('element')
        
        if element:
            for shape, shape_element in self.platonic_elements.items():
                if shape_element == element:
                    return shape
        
        # Default selection based on category
        category = concept.get('category', 'spiritual')
        if category in ['energy', 'fire']:
            return 'tetrahedron'
        elif category in ['stability', 'earth']:
            return 'hexahedron'
        elif category in ['air', 'movement']:
            return 'octahedron'
        elif category in ['water', 'flow']:
            return 'icosahedron'
        else:
            return 'dodecahedron'  # Default to aether/spirit
    
    def get_glyph_data(self, glyph_id: str) -> Optional[Dict[str, Any]]:
        """Get complete data for a glyph"""
        return self.created_glyphs.get(glyph_id)
    
    def list_glyphs(self, glyph_type: Optional[str] = None) -> List[str]:
        """List all created glyphs, optionally filtered by type"""
        if glyph_type:
            return [gid for gid, glyph in self.created_glyphs.items() 
                   if glyph.get('type') == glyph_type]
        return list(self.created_glyphs.keys())


# Factory function
def create_glyph_system() -> GlyphSystem:
    """Create a new glyph system instance"""
    return GlyphSystem()