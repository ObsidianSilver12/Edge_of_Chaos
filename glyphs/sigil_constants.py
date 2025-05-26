# --- START OF FILE glyphs/sigil_constants.py ---
"""
Constants specific to the Sigil Generation System.
"""
import numpy as np # For PI if not in main constants

# --- Sigil Categories (as per your list) ---
SIGIL_CATEGORY_MATHEMATICAL = "mathematical"
SIGIL_CATEGORY_GEOMETRIC = "geometric"
SIGIL_CATEGORY_ASTROLOGICAL = "astrological"
SIGIL_CATEGORY_ALCHEMICAL = "alchemical"
SIGIL_CATEGORY_MYSTICAL = "mystical"
SIGIL_CATEGORY_ELEMENTAL = "elemental"
SIGIL_CATEGORY_RUNIC = "runic"
SIGIL_CATEGORY_ANCIENT = "ancient"
SIGIL_CATEGORY_YIJING = "yijing"
SIGIL_CATEGORY_ZODIACAL = "zodiacal" # Distinct from Astrological? Or subset? For now, separate.
SIGIL_CATEGORY_RELIGIOUS = "religious"
SIGIL_CATEGORY_MUSICAL = "musical"
SIGIL_CATEGORY_MISCELLANEOUS = "miscellaneous"
SIGIL_CATEGORY_ANCHOR_SOUL = "ANCHOR_SOUL"
SIGIL_CATEGORY_ANCHOR_EARTH = "ANCHOR_EARTH"
# Add other specific system-defined categories like "GATEWAY_KEY" if needed

ALL_SIGIL_CATEGORIES = [
    SIGIL_CATEGORY_MATHEMATICAL, SIGIL_CATEGORY_GEOMETRIC, SIGIL_CATEGORY_ASTROLOGICAL,
    SIGIL_CATEGORY_ALCHEMICAL, SIGIL_CATEGORY_MYSTICAL, SIGIL_CATEGORY_ELEMENTAL,
    SIGIL_CATEGORY_RUNIC, SIGIL_CATEGORY_ANCIENT, SIGIL_CATEGORY_YIJING,
    SIGIL_CATEGORY_ZODIACAL, SIGIL_CATEGORY_RELIGIOUS, SIGIL_CATEGORY_MUSICAL,
    SIGIL_CATEGORY_MISCELLANEOUS, SIGIL_CATEGORY_ANCHOR_SOUL, SIGIL_CATEGORY_ANCHOR_EARTH
]

# --- Image Generation Parameters ---
DEFAULT_SIGIL_RESOLUTION = 512  # Pixels (e.g., 512x512 for single structure)
MAX_CORE_SYMBOLS = 3
CORE_SYMBOL_PADDING_FACTOR = 0.05 # Percentage of Platonic solid's bounding box
DEFAULT_LINE_WIDTH = 2 # Pixels
DEFAULT_LINE_COLOR_BASE_IMAGE = (0, 0, 0) # Black lines for base assets
TRANSPARENT_BACKGROUND = (0, 0, 0, 0) # RGBA for transparent

# --- Output Directories ---
# Base paths for assets and encoded glyphs (relative to project root or configurable)
BASE_ASSETS_PATH = "glyphs/assets"
SACRED_GEOMETRY_BASES_PATH = f"{BASE_ASSETS_PATH}/sacred_geometry_bases"
PLATONIC_SOLID_BASES_PATH = f"{BASE_ASSETS_PATH}/platonic_solid_bases"

ENCODED_GLYPHS_BASE_PATH = "glyph_resonance/encoded_glyphs"
DYNAMIC_SIGILS_PATH = f"{ENCODED_GLYPHS_BASE_PATH}/dynamic"
ANCHOR_SIGILS_PATH = f"{ENCODED_GLYPHS_BASE_PATH}/anchors"
CUSTOM_GLYPHS_PATH = f"{ENCODED_GLYPHS_BASE_PATH}/custom" # For your pre-designed ones

# --- EXIF Data ---
SIGIL_SYSTEM_SOFTWARE_TAG = "SoulForgeSigilSystemV1.0"
SIGIL_VERIFICATION_PHRASE = "RevelatioArcanum" # "Revelation of the Secret"

# --- Steganography ---
STEG_MESSAGE_LENGTH_BITS = 32 # Number of bits to store the length of the hidden message

# --- Default Frequencies for Coloring (Hz) ---
# (These can be overridden by specific geometry/platonic data if available)
DEFAULT_SACRED_GEOMETRY_COLOR_FREQ = 432.0 # A4 tuning
DEFAULT_PLATONIC_SOLID_COLOR_FREQ = 256.0  # Approx C4 (Middle C)

# --- Color Mapping (Example - can be expanded) ---
# For mapping frequencies to colors. This is a simplified model.
# A more sophisticated model would use light physics (wavelengths).
COLOR_FREQ_MAP = {
    (0, 100): (255, 0, 0, 255),        # Red for low frequencies
    (101, 300): (255, 165, 0, 255),    # Orange
    (301, 500): (255, 255, 0, 255),    # Yellow
    (501, 700): (0, 255, 0, 255),      # Green
    (701, 900): (0, 0, 255, 255),      # Blue
    (901, 1200): (75, 0, 130, 255),    # Indigo
    (1201, 10000): (148, 0, 211, 255)  # Violet for high frequencies
}
DEFAULT_LINE_RGBA_COLOR = (200, 200, 255, 255) # Light ethereal blue/white for dynamic coloring

# --- Sound for Inter-Ring Space ---
# Example parameters for a mixed noise background
UNIVERSAL_SIGIL_SOUND_PARAMS = {
    "type": "mixed_noise_v1",
    "segments": [
        {'type': 'pink', 'duration_s': 2.0, 'amplitude': 0.03, 'fade_out_s': 0.2},
        {'type': 'brown', 'duration_s': 2.0, 'amplitude': 0.02, 'fade_in_s': 0.2, 'fade_out_s': 0.2},
        {'type': 'white', 'duration_s': 1.0, 'amplitude': 0.01, 'fade_in_s': 0.2}
    ],
    "overall_amplitude_envelope": {'attack': 0.1, 'decay': 0.5, 'sustain_level': 0.7, 'release': 1.0},
    "target_rms_normalization": -25.0 # Target RMS in dBFS for the final mixed sound
}
# Path for pre-generated universal sounds if that approach is taken
UNIVERSAL_SOUND_ASSETS_PATH = "sound/universal_backgrounds"


# --- Golden Ratio (often used in sacred geometry) ---
# Moved here from main constants if it's primarily for sigils/glyphs
# If used more broadly, keep in main constants.py
# For now, assuming it's fine in main constants.py and imported with '*'
# PHI = (1 + np.sqrt(5)) / 2 # Example if it was here

# --- Mathematical constants ---
# PI = np.pi # Example if it was here

# --- Speed of Light (m/s) - for conceptual light/freq mapping ---
# SPEED_OF_LIGHT = 299792458.0 # Example if it was here

# --- SymPy rendering options (if used) ---
SYMPY_PREVIEW_OPTIONS = ["-D", "200"] # DPI for rendering SymPy expressions

# --- Font for Unicode Symbols ---
# This is system-dependent. Try common ones.
# On Windows: "Segoe UI Symbol", "Arial Unicode MS"
# On Linux: "DejaVu Sans", "Noto Sans Symbols2"
# On macOS: "Apple Symbols"
UNICODE_SYMBOL_FONT_PATHS = {
    "Windows": "C:/Windows/Fonts/seguiemj.ttf", # Segoe UI Emoji/Symbol
    "Linux": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Or Noto
    "Darwin": "/System/Library/Fonts/Apple Symbols.ttf"
}
DEFAULT_UNICODE_FONT_SIZE = 32 # Relative size, will be scaled

# --- Logging ---
SIGIL_LOG_LEVEL = "INFO" # Default logging level for sigil system
SIGIL_LOG_FORMAT = '%(asctime)s - SIGIL_SYS - %(levelname)s - %(module)s.%(funcName)s: %(message)s'

# --- Tagging constants from problem description ---
DOMAIN_TAG = "domain"
CONCEPT_TAG = "concept"
RELATED_CONCEPTS_TAG = "related_concepts"
PERSONAL_TAGS_TAG = "personal_tags"
META_TAGS_TAG = "meta_tags"
# --- END OF FILE glyphs/sigil_constants.py ---