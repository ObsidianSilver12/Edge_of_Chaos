# --- START OF FILE glyphs/sigil_components.py ---
"""
Defines the Sigil class and related data structures like
Unicode symbol mappings.
"""
import logging
import uuid
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import sigil-specific constants
try:
    from .sigil_constants import *
except ImportError:
    # Fallback if run directly or constants not found in this specific path
    # This is less ideal but makes the file parseable standalone for testing.
    # In a real run, constants.py from project root should be accessible.
    SIGIL_SYSTEM_SOFTWARE_TAG = "SoulForgeSigilSystemV1.0_fallback"
    SIGIL_VERIFICATION_PHRASE = "RevelatioArcanum_fallback"
    # ... add other critical fallbacks from sigil_constants.py if needed ...
    DOMAIN_TAG = "domain"
    CONCEPT_TAG = "concept"
    RELATED_CONCEPTS_TAG = "related_concepts"
    PERSONAL_TAGS_TAG = "personal_tags"
    META_TAGS_TAG = "meta_tags"

# Setup logger for this module
logger = logging.getLogger(__name__)
# Configure logger if not already configured (e.g. if run standalone)
if not logger.handlers:
    try:
        log_level_sc = getattr(logging, SIGIL_LOG_LEVEL, logging.INFO)
        log_fmt_sc = SIGIL_LOG_FORMAT
    except NameError: # sigil_constants might not have loaded
        log_level_sc = logging.INFO
        log_fmt_sc = '%(asctime)s - SIGIL_COMP - %(levelname)s - %(message)s'
    logger.setLevel(log_level_sc)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(log_fmt_sc))
    logger.addHandler(ch)


# --- Unicode Symbol Data (Curated Lists) ---
# This should be expanded significantly.
UNICODE_SYMBOLS_BY_CATEGORY: Dict[str, Dict[str, Dict[str, Any]]] = {
    SIGIL_CATEGORY_MATHEMATICAL: {
        '∑': {'codepoint': 'U+2211', 'description': 'N-Ary Summation', 'keywords': ['sum', 'total', 'integration', 'aggregate']},
        '∫': {'codepoint': 'U+222B', 'description': 'Integral', 'keywords': ['integration', 'calculus', 'area', 'continuous sum']},
        '∞': {'codepoint': 'U+221E', 'description': 'Infinity', 'keywords': ['limitless', 'unbounded', 'eternal', 'transfinite']},
        'π': {'codepoint': 'U+03C0', 'description': 'Greek Small Letter Pi', 'keywords': ['pi', 'circle', 'ratio', 'transcendental']},
        '∆': {'codepoint': 'U+2206', 'description': 'Increment / Laplace Operator', 'keywords': ['change', 'difference', 'triangle', 'delta']},
        '√': {'codepoint': 'U+221A', 'description': 'Square Root', 'keywords': ['root', 'origin', 'radix']},
    },
    SIGIL_CATEGORY_ALCHEMICAL: {
        '☉': {'codepoint': 'U+2609', 'description': 'Sun', 'keywords': ['gold', 'sol', 'consciousness', 'self', 'masculine', 'spirit']},
        '☽': {'codepoint': 'U+263D', 'description': 'Moon', 'keywords': ['silver', 'luna', 'subconscious', 'reflection', 'feminine', 'soul']},
        '♀': {'codepoint': 'U+2640', 'description': 'Venus / Copper', 'keywords': ['venus', 'copper', 'love', 'beauty', 'harmony', 'feminine']},
        '♂': {'codepoint': 'U+2642', 'description': 'Mars / Iron', 'keywords': ['mars', 'iron', 'war', 'action', 'masculine', 'strength']},
        '🜁': {'codepoint': 'U+1F701', 'description': 'Alchemical Symbol for Air', 'keywords': ['air', 'intellect', 'communication', 'breath']},
        '🜂': {'codepoint': 'U+1F702', 'description': 'Alchemical Symbol for Fire', 'keywords': ['fire', 'energy', 'passion', 'transformation']},
        '🜃': {'codepoint': 'U+1F703', 'description': 'Alchemical Symbol for Water', 'keywords': ['water', 'emotion', 'intuition', 'flow']},
        '🜄': {'codepoint': 'U+1F704', 'description': 'Alchemical Symbol for Earth', 'keywords': ['earth', 'stability', 'matter', 'grounding']},
        '🜔': {'codepoint': 'U+1F714', 'description': 'Alchemical Symbol for Salt', 'keywords': ['salt', 'body', 'crystallization', 'purification']},
        '🜍': {'codepoint': 'U+1F70D', 'description': 'Alchemical Symbol for Sulfur', 'keywords': ['sulfur', 'soul', 'consciousness', 'active principle']},
        '🜐': {'codepoint': 'U+1F710', 'description': 'Alchemical Symbol for Mercury', 'keywords': ['mercury', 'quicksilver', 'spirit', 'messenger', 'mind']},
    },
    SIGIL_CATEGORY_ELEMENTAL: { # Can overlap with Alchemical but focus on pure elements
        '🔥': {'codepoint': 'U+1F525', 'description': 'Fire Emoji', 'keywords': ['fire', 'energy', 'passion']},
        '💧': {'codepoint': 'U+1F4A7', 'description': 'Droplet Emoji (Water)', 'keywords': ['water', 'emotion', 'flow']},
        '💨': {'codepoint': 'U+1F4A8', 'description': 'Dashing Away Emoji (Air)', 'keywords': ['air', 'wind', 'breath']},
        '🌍': {'codepoint': 'U+1F30D', 'description': 'Earth Globe Europe-Africa Emoji', 'keywords': ['earth', 'grounding', 'stability']},
        '🌌': {'codepoint': 'U+1F30C', 'description': 'Milky Way Emoji (Aether/Spirit)', 'keywords': ['aether', 'spirit', 'cosmos', 'universe']},
    },
    SIGIL_CATEGORY_RUNIC: { # Elder Futhark examples
        'ᚠ': {'codepoint': 'U+16A0', 'description': 'Fehu', 'keywords': ['wealth', 'cattle', 'abundance', 'luck']},
        'ᚢ': {'codepoint': 'U+16A2', 'description': 'Uruz', 'keywords': ['strength', 'aurochs', 'power', 'perseverance']},
        'ᚦ': {'codepoint': 'U+16A6', 'description': 'Thurisaz', 'keywords': ['giant', 'thorn', 'protection', 'conflict']},
    },
    SIGIL_CATEGORY_YIJING: { # Trigrams
        '☰': {'codepoint': 'U+2630', 'description': 'Trigram for Heaven (Qian)', 'keywords': ['heaven', 'creative', 'strong', 'masculine']},
        '☱': {'codepoint': 'U+2631', 'description': 'Trigram for Lake (Dui)', 'keywords': ['lake', 'joyful', 'open', 'youngest daughter']},
    },
    # ... Add more categories and symbols
    SIGIL_CATEGORY_MISCELLANEOUS: {
         '⚛': {'codepoint': 'U+269B', 'description': 'Atom Symbol', 'keywords': ['atomic', 'science', 'particle', 'energy']},
         '⚜': {'codepoint': 'U+269C', 'description': 'Fleur-de-lis', 'keywords': ['royalty', 'purity', 'light']},
    }
}


class Sigil:
    """
    Represents a single sigil, its components, visual representation,
    and embedded data.
    """
    def __init__(self,
                 category: str,
                 purpose_keywords: List[str],
                 creator_id: str = "system_default",
                 difficulty: float = 0.5,
                 complexity: float = 0.5,
                 intended_application: str = "general_resonance",
                 target_resonance_descriptor: Optional[str] = None):
        """
        Initializes a Sigil object. Visual components are set separately.

        Args:
            category (str): The primary category of the sigil (e.g., "mathematical").
            purpose_keywords (List[str]): Keywords describing the sigil's intent.
            creator_id (str): Identifier for the entity creating the sigil.
            difficulty (float): Estimated difficulty to create/charge/understand (0-1).
            complexity (float): Structural or conceptual complexity (0-1).
            intended_application (str): Brief description of how the sigil is meant to be used.
            target_resonance_descriptor (Optional[str]): Description of the intended resonant target.
        """
        if category not in ALL_SIGIL_CATEGORIES: # Use constant from sigil_constants
            logger.warning(f"Sigil category '{category}' not in predefined list. Adding to miscellaneous.")
            self.category: str = SIGIL_CATEGORY_MISCELLANEOUS
        else:
            self.category: str = category

        self.sigil_id: str = str(uuid.uuid4())
        self.creation_date: str = datetime.now().isoformat()
        self.creator_id: str = creator_id
        self.purpose_keywords: List[str] = sorted(list(set(purpose_keywords))) # Ensure unique and sorted

        # Visual Components (to be populated by drawing/composition logic)
        self.outer_circle_params: Dict[str, Any] = {}
        self.inner_circle_params: Dict[str, Any] = {}
        self.inter_ring_geometry: Optional[Dict[str, Any]] = None # e.g., {'type': 'flower_of_life', 'line_color_freq_hz': 432.0, 'base_image_path': '...'}
        self.platonic_solid: Optional[Dict[str, Any]] = None   # e.g., {'type': 'tetrahedron', 'line_color_freq_hz': 396.0, 'base_image_path': '...'}
        self.symbolic_core: List[Dict[str, Any]] = []          # List of 1-3 symbols

        # Data Storage Paths
        self.image_path: Optional[str] = None
        self.metadata_blob_path: Optional[str] = None # Path to the main JSON blob for this sigil

        # Technical/Operational Data
        self.difficulty_rating: float = max(0.0, min(1.0, difficulty))
        self.complexity_rating: float = max(0.0, min(1.0, complexity))
        self.intended_application: str = intended_application
        self.target_resonance_descriptor: Optional[str] = target_resonance_descriptor
        self.version: str = "1.0.0"
        self.content_hash: Optional[str] = None # Populated by finalize_sigil

        # Data to be embedded
        self.surface_data_exif: Dict[str, Any] = {} # For EXIF UserComment (limited size)
        self.main_blob_data: Dict[str, Any] = {} # For the primary JSON blob file
        # self.hidden_data_steg: Dict[str, Any] = {} # Data for steganography (less emphasis now, can be part of main_blob_data initially)

        # Derived Properties (Calculated based on components)
        self.dominant_light_properties: Dict[str, Any] = {}
        self.resonant_sound_properties: Dict[str, Any] = {}
        self.overall_frequency_signature: Dict[str, Any] = {}

        logger.debug(f"Sigil class initialized for ID {self.sigil_id} (Category: {self.category})")

    def set_visual_components(self,
                              outer_circle: Dict[str, Any],
                              inner_circle: Dict[str, Any],
                              inter_ring_geom: Optional[Dict[str, Any]],
                              platonic: Optional[Dict[str, Any]],
                              core_symbols: List[Dict[str, Any]]):
        """Sets the visual components that define the sigil's appearance."""
        self.outer_circle_params = outer_circle
        self.inner_circle_params = inner_circle
        self.inter_ring_geometry = inter_ring_geom
        self.platonic_solid = platonic
        if len(core_symbols) > MAX_CORE_SYMBOLS: # Use constant
            logger.warning(f"Too many core symbols ({len(core_symbols)}). Truncating to {MAX_CORE_SYMBOLS}.")
        self.symbolic_core = core_symbols[:MAX_CORE_SYMBOLS]
        logger.debug(f"Visual components set for sigil {self.sigil_id}.")

    def _generate_content_hash(self) -> str:
        """Generates a SHA256 hash based on the sigil's core defining visual and semantic components."""
        data_to_hash = {
            "category": self.category,
            "purpose": self.purpose_keywords, # Already sorted
            "inter_ring_type": self.inter_ring_geometry['type'] if self.inter_ring_geometry else None,
            "platonic_type": self.platonic_solid['type'] if self.platonic_solid else None,
            # Sort core symbols by their actual symbol string for consistent hashing
            "core_symbols_values": sorted([s['symbol'] for s in self.symbolic_core if 'symbol' in s])
        }
        json_str = json.dumps(data_to_hash, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _calculate_derived_properties(self):
        """
        Calculates dominant light, resonant sound, and overall frequency signature
        based on the chosen visual components. Placeholder for complex physics.
        """
        logger.debug(f"Calculating derived properties for sigil {self.sigil_id}...")
        # --- Dominant Light Properties ---
        # Simplistic: average color of components or based on primary intent.
        # A more complex model would simulate light interference from geometric forms.
        # For now, use color from inter_ring_geometry if present, else platonic, else default.
        dominant_hex = "#FFFFFF" # Default white light
        associated_light_freqs_thz = [430.0, 770.0] # Broad visible spectrum
        base_light_freq_for_coloring = DEFAULT_SACRED_GEOMETRY_COLOR_FREQ # From sigil_constants

        if self.inter_ring_geometry and 'line_color_freq_hz' in self.inter_ring_geometry:
            base_light_freq_for_coloring = self.inter_ring_geometry['line_color_freq_hz']
        elif self.platonic_solid and 'line_color_freq_hz' in self.platonic_solid:
            base_light_freq_for_coloring = self.platonic_solid['line_color_freq_hz']
        
        # Conceptual: Convert this audio-range frequency to a light color
        # This mapping needs a proper physics-based or esoteric model.
        # For now, using a placeholder function.
        try:
            r_light, g_light, b_light = map_frequency_to_rgb_color(base_light_freq_for_coloring) # Needs to be defined
            dominant_hex = f"#{r_light:02x}{g_light:02x}{b_light:02x}"
            # Conceptual light frequencies (e.g., primary and a harmonic)
            light_freq_1_thz = (base_light_freq_for_coloring * 1e9) / SPEED_OF_LIGHT # Placeholder scaling
            associated_light_freqs_thz = [light_freq_1_thz * 1e-12, light_freq_1_thz * PHI * 1e-12]

        except NameError: # SPEED_OF_LIGHT or PHI might not be loaded if constants failed
            logger.warning("SPEED_OF_LIGHT or PHI not available for light property calculation.")
        except Exception as e:
            logger.warning(f"Could not calculate dominant hex: {e}")


        self.dominant_light_properties = {
            'dominant_hex': dominant_hex,
            'associated_light_frequencies_thz': associated_light_freqs_thz,
            'primary_source_frequency_for_color_hz': base_light_freq_for_coloring
        }

        # --- Resonant Sound Properties ---
        # Collect frequencies from components
        all_frequencies = []
        if self.inter_ring_geometry and 'line_color_freq_hz' in self.inter_ring_geometry:
            all_frequencies.append(self.inter_ring_geometry['line_color_freq_hz'])
        if self.platonic_solid and 'line_color_freq_hz' in self.platonic_solid:
            all_frequencies.append(self.platonic_solid['line_color_freq_hz'])
        for core_sym in self.symbolic_core:
            # Assume core symbols might have an associated conceptual frequency
            # This needs to be defined in UNICODE_SYMBOLS_BY_CATEGORY
            if 'conceptual_freq_hz' in core_sym:
                all_frequencies.append(core_sym['conceptual_freq_hz'])
        
        if not all_frequencies: all_frequencies.append(432.0) # Default
        
        primary_resonant_freq = np.median(all_frequencies) # Use median as a robust central tendency
        harmonics = [primary_resonant_freq * r for r in [1.0, PHI, 2.0, 1/PHI, 3.0/2.0]]
        self.resonant_sound_properties = {
            'primary_resonant_frequency_hz': float(primary_resonant_freq),
            'harmonics_hz': [float(h) for h in harmonics[:5]], # Top 5
            'timbre_description': f"{self.category}_{self.inter_ring_geometry['type'] if self.inter_ring_geometry else 'open'}_{self.platonic_solid['type'] if self.platonic_solid else 'core'}",
            'universal_background_params': UNIVERSAL_SIGIL_SOUND_PARAMS # From sigil_constants
        }

        # --- Overall Frequency Signature ---
        # A composite or key frequency representing the sigil's core vibration
        self.overall_frequency_signature = {'primary_hz': float(primary_resonant_freq)}
        logger.debug("Derived properties calculated.")


    def finalize_sigil_data(self):
        """
        Populates the data dictionaries after all components are set and derived properties calculated.
        Generates content hash.
        """
        self._calculate_derived_properties() # Calculate light/sound/freq first
        self.content_hash = self._generate_content_hash()

        # --- Surface Data (for EXIF UserComment and main JSON blob) ---
        self.surface_data_exif = {
            "sigil_id": self.sigil_id,
            "creation_date": self.creation_date,
            "creator_id": self.creator_id,
            "sigil_version": self.version,
            "category": self.category,
            "purpose_keywords_summary": ", ".join(self.purpose_keywords[:5]) + ('...' if len(self.purpose_keywords) > 5 else ''),
            "verification_phrase": SIGIL_VERIFICATION_PHRASE, # From sigil_constants
            "content_hash_sha256": self.content_hash,
            "image_dimensions_px": f"{DEFAULT_SIGIL_RESOLUTION}x{DEFAULT_SIGIL_RESOLUTION}", # Assuming square for now
            "core_symbol_count": len(self.symbolic_core),
        }

        # --- Main Blob Data (for the sigil_id.json file) ---
        self.main_blob_data = {
            "sigil_id": self.sigil_id,
            "creation_date": self.creation_date,
            "creator_id": self.creator_id,
            "sigil_version": self.version,
            "category": self.category,
            "purpose_keywords": self.purpose_keywords,
            "verification_phrase": SIGIL_VERIFICATION_PHRASE,
            "content_hash_sha256": self.content_hash,

            "visual_components": {
                "outer_circle_params": self.outer_circle_params,
                "inner_circle_params": self.inner_circle_params,
                "inter_ring_geometry": self.inter_ring_geometry,
                "platonic_solid": self.platonic_solid,
                "symbolic_core": self.symbolic_core,
            },
            "derived_properties": {
                "dominant_light_properties": self.dominant_light_properties,
                "resonant_sound_properties": self.resonant_sound_properties,
                "overall_frequency_signature": self.overall_frequency_signature,
            },
            "operational_data": {
                "difficulty_rating": self.difficulty_rating,
                "complexity_rating": self.complexity_rating,
                "intended_application": self.intended_application,
                "target_resonance_descriptor": self.target_resonance_descriptor,
                "steps_to_charge": 0, # Example, could be calculated or set
                "current_charge_level": 0.0,
            },
            "tagging_blob": { # Using the keys you specified
                DOMAIN_TAG: "DefaultDomain", # Placeholder, to be filled by specific logic
                CONCEPT_TAG: "DefaultConcept", # Placeholder
                RELATED_CONCEPTS_TAG: [], # Placeholder
                PERSONAL_TAGS_TAG: [], # Placeholder
                META_TAGS_TAG: ["dynamic_sigil_system"], # Placeholder
            },
            "steganographic_payload_summary": { # Info about what *could* be in steganography
                "intended_content_type": "supplementary_esoteric_data",
                "estimated_size_bytes": 0 # This would be updated if steganography is used
            },
            # --- Hidden/Occult Data Blob (part of the main JSON for now, not separate file) ---
            # This is where the "Meaning-Driven/Occult Information" goes
            "hidden_occult_meaning": {
                "deeper_intent": "To harmonize [X] with [Y] through resonant alignment.", # Example
                "esoteric_correspondences": { # Example
                    "planet": "Jupiter (Expansion)",
                    "element_implied": self.platonic_solid['element'] if self.platonic_solid and 'element' in self.platonic_solid else "Universal",
                    "number": len(self.symbolic_core) + (self.inter_ring_geometry['circles'] if self.inter_ring_geometry and 'circles' in self.inter_ring_geometry else 0)
                },
                "activation_phrase_suggestion": "Fiat Lux Resonantia" # Example
            }
        }
        logger.debug(f"Sigil data finalized for {self.sigil_id}.")

    def __str__(self):
        return f"Sigil(ID: {self.sigil_id[:8]}..., Category: {self.category}, Purpose: {self.purpose_keywords[:3]})"

    def __repr__(self):
        return f"<Sigil id='{self.sigil_id}' category='{self.category}'>"


# --- Helper: Frequency to Color Mapping ---
def map_frequency_to_rgb_color(frequency_hz: float) -> Tuple[int, int, int]:
    """
    Maps an audio-range frequency to an RGB color tuple (0-255).
    This is a conceptual mapping, not strictly physics-based for visible light.
    """
    if frequency_hz <= 0: return (128, 128, 128) # Grey for invalid

    # Normalize frequency to a 0-1 range (e.g., within 20Hz to 20kHz log scale)
    log_min = np.log10(20)
    log_max = np.log10(20000)
    norm_freq = (np.log10(max(20,min(20000,frequency_hz))) - log_min) / (log_max - log_min)
    norm_freq = np.clip(norm_freq, 0.0, 1.0)

    # Map normalized frequency to Hue (0-360 degrees)
    # Example: Low freqs red, mid green, high blue/violet
    hue = norm_freq * 300.0 # 0-300 range: Red -> Green -> Blue -> Purple part of spectrum
    saturation = 0.8 + norm_freq * 0.2 # Higher frequencies more saturated
    value = 0.7 + (1.0 - norm_freq) * 0.3 # Lower frequencies brighter (value)

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


if __name__ == '__main__':
    logger.info("Testing Sigil Component Initialization...")
    test_sigil = Sigil(
        category=SIGIL_CATEGORY_ALCHEMICAL,
        purpose_keywords=["transformation", "purification", "prima_materia"],
        creator_id="test_system",
        difficulty=0.7,
        complexity=0.6,
        intended_application="meditation_focus_object",
        target_resonance_descriptor="Alchemical Gold (Sol)"
    )
    test_sigil.set_visual_components(
        outer_circle={'radius_factor': 1.0, 'color': '#CCCCCC', 'line_width': 3},
        inner_circle={'radius_factor': 0.75, 'color': '#AAAAAA', 'line_width': 2},
        inter_ring_geom={'type': 'vesica_piscis', 'line_color_freq_hz': 432.0, 'base_image_path': 'glyphs/assets/sacred_geometry_bases/vesica_piscis_base.png'},
        platonic={'type': 'hexahedron', 'line_color_freq_hz': 256.0, 'base_image_path': 'glyphs/assets/platonic_solid_bases/hexahedron_base.png'},
        core_symbols=[
            {'symbol': '☉', 'unicode_codepoint': 'U+2609', 'meaning_tags': ['sun', 'gold']},
            {'symbol': '🜍', 'unicode_codepoint': 'U+1F70D', 'meaning_tags': ['sulfur', 'soul']},
        ]
    )
    test_sigil.finalize_sigil_data()
    print(f"Test Sigil Created: {test_sigil}")
    print(f"Content Hash: {test_sigil.content_hash}")
    print(f"Surface Data for EXIF (first few): {dict(list(test_sigil.surface_data_exif.items())[:3])}")
    print(f"Dominant Light: {test_sigil.dominant_light_properties}")
    print(f"Resonant Sound: {test_sigil.resonant_sound_properties}")

    # Test color mapping
    print(f"Color for 100Hz: {map_frequency_to_rgb_color(100)}")
    print(f"Color for 432Hz: {map_frequency_to_rgb_color(432)}")
    print(f"Color for 1000Hz: {map_frequency_to_rgb_color(1000)}")
    print(f"Color for 8000Hz: {map_frequency_to_rgb_color(8000)}")

# --- END OF FILE glyphs/sigil_components.py ---