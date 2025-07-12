# glyphs/sigil_constants.py
"""
Sigil Constants for Glyph System

Defines paths, rendering constants, and configuration for the glyph system.
Works with the existing base_glyph_generation.py architecture.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Base directories for different asset types
GLYPHS_BASE_PATH = Path("glyphs")
ASSETS_BASE_PATH = Path("shared/assets")

# Sacred geometry and platonic base images (for use in composition)
SACRED_GEOMETRY_BASES_PATH = ASSETS_BASE_PATH / "sacred_geometry"
PLATONIC_SOLID_BASES_PATH = ASSETS_BASE_PATH / "platonics"

# Glyph workflow paths
GLYPHS_TO_ENCODE_PATH = ASSETS_BASE_PATH / "to_encode"
GLYPHS_ENCODED_PATH = ASSETS_BASE_PATH / "encoded"
SEPHIRAH_IMAGES_PATH = ASSETS_BASE_PATH / "sephirah"

# Create directories if they don't exist
for path in [SACRED_GEOMETRY_BASES_PATH, PLATONIC_SOLID_BASES_PATH, 
             GLYPHS_TO_ENCODE_PATH, GLYPHS_ENCODED_PATH, SEPHIRAH_IMAGES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RENDERING CONSTANTS
# =============================================================================

# Image properties
DEFAULT_SIGIL_RESOLUTION = (1024, 1024)  # (width, height)
DEFAULT_DPI = 300
DEFAULT_IMAGE_FORMAT = "PNG"

# Colors
DEFAULT_LINE_COLOR_BASE_IMAGE = '#000000'  # Black lines
TRANSPARENT_BACKGROUND = (0.0, 0.0, 0.0, 0.0)  # RGBA transparent
WHITE_BACKGROUND = '#FFFFFF'
BLACK_BACKGROUND = '#000000'

# Line properties
DEFAULT_LINE_WIDTH = 2.0
THIN_LINE_WIDTH = 1.0
THICK_LINE_WIDTH = 3.0

# Glyph structure constants
OUTER_CIRCLE_RADIUS = 4.5
INNER_CIRCLE_RADIUS = 3.0
SEPHIRAH_SIGIL_RADIUS = 3.75  # Between outer and inner circles
PLATONIC_SIZE = 1.5
SIGIL_SIZE_SMALL = 0.4
SIGIL_SIZE_MEDIUM = 0.6
SIGIL_SIZE_LARGE = 0.8

# =============================================================================
# GLYPH TYPE CONFIGURATIONS
# =============================================================================

# Gateway Key Glyph Configuration
GATEWAY_KEY_CONFIG = {
    'outer_circle_color': BLACK_BACKGROUND,
    'inner_circle_color': WHITE_BACKGROUND,
    'outer_circle_fill': True,
    'inner_circle_fill': True,
    'sephirah_sigil_color': WHITE_BACKGROUND,  # White text on black ring
    'platonic_color': DEFAULT_LINE_COLOR_BASE_IMAGE,
    'key_sigil_color': DEFAULT_LINE_COLOR_BASE_IMAGE,
    'line_width': THICK_LINE_WIDTH
}

# Normal Glyph Configuration
NORMAL_GLYPH_CONFIG = {
    'circle_color': DEFAULT_LINE_COLOR_BASE_IMAGE,
    'circle_fill': False,
    'sacred_geometry_color': DEFAULT_LINE_COLOR_BASE_IMAGE,
    'platonic_color': DEFAULT_LINE_COLOR_BASE_IMAGE,
    'sigil_color': DEFAULT_LINE_COLOR_BASE_IMAGE,
    'line_width': DEFAULT_LINE_WIDTH
}

# =============================================================================
# SEPHIRAH SPECIFIC CONSTANTS
# =============================================================================

# Sephirah colors (from existing SEPHIROTH_GLYPH_DATA)
SEPHIRAH_COLORS = {
    'kether': '#FFFFFF',     # White
    'chokmah': '#C0C0C0',    # Silver/Grey  
    'binah': '#000000',      # Black
    'chesed': '#0000FF',     # Blue
    'geburah': '#FF0000',    # Red
    'tiphareth': '#FFFF00',  # Yellow/Gold
    'netzach': '#00FF00',    # Green
    'hod': '#FFA500',        # Orange
    'yesod': '#800080',      # Purple/Violet
    'malkuth': '#8B4513',    # Brown/Earth
    'daath': '#708090'       # Slate Grey (hidden sephirah)
}

# Sephirah sigil mappings (using existing SEPHIROTH_GLYPH_DATA)
SEPHIRAH_SIGILS = {
    'kether': 'Point/Crown',
    'chokmah': 'Line/Wheel', 
    'binah': 'Triangle/Womb',
    'chesed': 'Square/Solid',
    'geburah': 'Pentagon/Sword',
    'tiphareth': 'Hexagram/Sun',
    'netzach': 'Heptagon/Victory',
    'hod': 'Octagon/Splendor',
    'yesod': 'Nonagon/Foundation',
    'malkuth': 'CrossInCircle/Kingdom',
    'daath': 'VoidPoint'
}

# Platonic associations (from existing SEPHIROTH_GLYPH_DATA)
SEPHIRAH_PLATONICS = {
    'kether': 'dodecahedron',
    'chokmah': 'sphere',
    'binah': 'icosahedron',
    'chesed': 'hexahedron',
    'geburah': 'tetrahedron',
    'tiphareth': 'octahedron',
    'netzach': 'icosahedron',
    'hod': 'octahedron',
    'yesod': 'icosahedron',
    'malkuth': 'hexahedron',
    'daath': 'sphere'
}

# =============================================================================
# UNICODE SIGIL CATEGORIES
# =============================================================================

# Available Unicode sigil categories for use with existing sigils_dictionary
SIGIL_CATEGORIES = [
    'sacred_geometry',
    'spiritual', 
    'elemental',
    'mystical',
    'cosmic',
    'alchemical',
    'runic',
    'mathematical',
    'arrows'
]

# =============================================================================
# ENCODING CONSTANTS
# =============================================================================

# Metadata tags for EXIF encoding
GLYPH_SOFTWARE_TAG = b'SoulGlyphSystem'
GLYPH_DESCRIPTION_TAG = 'Soul Development Framework Glyph'

# Steganography settings
LSB_BITS_PER_CHANNEL = 1  # Use 1 LSB per color channel
STEGANOGRAPHY_HEADER_BITS = 32  # 32 bits for message length

# =============================================================================
# INTEGRATION WITH BASE_GLYPH_GENERATION
# =============================================================================

# These match the parameters expected by base_glyph_generation.py
BASE_GLYPH_GENERATION_CONFIG = {
    'center_2d_param': (0.0, 0.0),
    'center_3d_param': (0.0, 0.0, 0.0),
    'radius_param': 1.0,
    'edge_length_param': 2.0,
    'data_gen_resolution_param': 100
}

# File naming conventions
BASE_GLYPH_FILENAME_SUFFIX = "_base.png"
GATEWAY_GLYPH_PREFIX = "gateway_key_"
NORMAL_GLYPH_PREFIX = "normal_glyph_"
ENCODED_GLYPH_PREFIX = "encoded_"

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Maximum limits
MAX_SIGILS_PER_NORMAL_GLYPH = 3
MAX_SEPHIRAH_CONNECTIONS_PER_KEY = 10
MIN_GLYPH_COMPLEXITY = 0.0
MAX_GLYPH_COMPLEXITY = 1.0

# File size limits (in bytes)
MAX_GLYPH_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_STEGANOGRAPHY_PAYLOAD_SIZE = 1024 * 1024  # 1MB

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    'invalid_sephirah': "Invalid Sephirah name provided: {}",
    'sigil_unavailable': "No available sigils in category: {}",
    'file_not_found': "Required base glyph file not found: {}",
    'encoding_failed': "Failed to encode glyph: {}",
    'invalid_complexity': "Complexity must be between {} and {}",
    'max_sigils_exceeded': "Maximum {} sigils allowed per normal glyph",
    'geometry_generation_failed': "Failed to generate {} geometry: {}"
}