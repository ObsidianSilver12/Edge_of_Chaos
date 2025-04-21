"""
Sephiroth Field 3D Implementation

This module implements a 3D field for Sephiroth dimensions,
managing energy potential, resonance, and interaction with aspects,
sacred geometry, and platonic solids within a 3D space.

Author: Soul Development Framework Team
Refactored to 3D with strict error handling.
"""

import numpy as np
import logging
import os
import uuid
from datetime import datetime
import scipy.ndimage # For Gaussian smoothing
from typing import Optional, Tuple, Dict, Any, List

# --- Constants ---
try:
    # Attempt to import constants, raise error if essential ones are missing
    from src.constants import (
        DEFAULT_DIMENSIONS_3D, DEFAULT_FIELD_DTYPE, DEFAULT_COMPLEX_DTYPE,
        PHI, PI, DEFAULT_GEOMETRY_EMBED_STRENGTH, DEFAULT_PLATONIC_EMBED_STRENGTH,
        ELEMENT_FIRE, ELEMENT_WATER, ELEMENT_AIR, ELEMENT_EARTH,
        ELEMENT_AETHER, ELEMENT_SPIRIT, ELEMENT_QUINTESSENCE, ELEMENT_NEUTRAL,
        SOLFEGGIO_MI, # Example Solfeggio freq used in embedding
        DATA_DIR_BASE, LOG_LEVEL, LOG_FORMAT
    )
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. Cannot proceed.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
# Adjust paths based on your actual project structure
try:
    from src.field_system_3d import Field3D # Import the 3D base class
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    from sound.sephiroth_sound_integration import SephirothSoundIntegration
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import core dependencies for SephirothField3D: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

# --- Geometry/Platonic Embedder Imports ---
# Attempt to import all embedders. Log warnings if missing, but don't fail initialization.
GEOMETRY_EMBEDDERS_AVAILABLE = True
EMBED_FUNCTIONS = {}
try:
    from stage_1.sacred_geometry.flower_of_life import embed_flower_in_field
    EMBED_FUNCTIONS["flower_of_life"] = embed_flower_in_field
    from stage_1.sacred_geometry.seed_of_life import embed_seed_in_field
    EMBED_FUNCTIONS["seed_of_life"] = embed_seed_in_field
    from stage_1.sacred_geometry.merkaba import embed_merkaba_in_field
    EMBED_FUNCTIONS["merkaba"] = embed_merkaba_in_field
    from stage_1.sacred_geometry.metatrons_cube import embed_metatrons_cube_in_field
    EMBED_FUNCTIONS["metatrons_cube"] = embed_metatrons_cube_in_field
    from stage_1.sacred_geometry.sri_yantra import embed_sri_yantra_in_field
    EMBED_FUNCTIONS["sri_yantra"] = embed_sri_yantra_in_field
    from stage_1.sacred_geometry.tree_of_life import embed_tree_of_life_in_field
    EMBED_FUNCTIONS["tree_of_life"] = embed_tree_of_life_in_field
    from stage_1.sacred_geometry.vesica_piscis import embed_vesica_in_field
    EMBED_FUNCTIONS["vesica_piscis"] = embed_vesica_in_field
    # Import others if they exist (add here)
    # from stage_1.sacred_geometry.fruit_of_life import embed_fruit_in_field
    # EMBED_FUNCTIONS["fruit_of_life"] = embed_fruit_in_field

    from stage_1.platonic_solids.tetrahedron import embed_tetrahedron_in_field
    EMBED_FUNCTIONS["tetrahedron"] = embed_tetrahedron_in_field
    from stage_1.platonic_solids.hexahedron import embed_hexahedron_in_field
    EMBED_FUNCTIONS["hexahedron"] = embed_hexahedron_in_field
    from stage_1.platonic_solids.octahedron import embed_octahedron_in_field
    EMBED_FUNCTIONS["octahedron"] = embed_octahedron_in_field
    from stage_1.platonic_solids.dodecahedron import embed_dodecahedron_in_field
    EMBED_FUNCTIONS["dodecahedron"] = embed_dodecahedron_in_field
    from stage_1.platonic_solids.icosahedron import embed_icosahedron_in_field
    EMBED_FUNCTIONS["icosahedron"] = embed_icosahedron_in_field
except ImportError as e:
    logging.warning(f"Could not import all geometry/platonic embedders: {e}. Embedding features will be limited.")
    GEOMETRY_EMBEDDERS_AVAILABLE = False

# Configure logging
log_file_path = os.path.join("logs", "sephiroth_field_3d.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('sephiroth_field_3d')

class SephirothField3D(Field3D): # Inherit from 3D base
    """
    3D field implementation for Sephiroth fields.
    Manages energy potential, resonance, and interaction with aspects,
    sacred geometry, and platonic solids within a 3D space.
    Strict error handling enforced.
    """

    def __init__(self, sephirah: str, dimensions: tuple = DEFAULT_DIMENSIONS_3D,
                 data_dir: str = DATA_DIR_BASE, **kwargs):
        """
        Initialize a Sephiroth field. Fails if essential data is missing.

        Args:
            sephirah (str): The name of the Sephirah (case-insensitive).
            dimensions (tuple): Dimensions of the 3D field (tuple of 3 positive integers).
            data_dir (str): Base directory to store field data.
            **kwargs: Additional keyword arguments for the base class (e.g., initial_energy).

        Raises:
            ValueError: If Sephirah name is invalid, dimensions are incorrect,
                        or essential Sephiroth properties cannot be loaded.
            RuntimeError: If critical dependencies like aspect_dictionary are missing.
        """
        self.sephirah = sephirah.lower()
        field_name = f"sephiroth_{self.sephirah}"

        # --- Validate Sephirah and Load Properties FIRST ---
        if not ASPECT_DICT_AVAILABLE:
            raise RuntimeError("CRITICAL: SephirothAspectDictionary failed to load.")
        if self.sephirah not in aspect_dictionary.sephiroth_names:
            raise ValueError(f"Invalid Sephirah name provided: '{sephirah}'")
        self._load_sephirah_properties() # Fails internally if essential props missing

        # --- Initialize Base Class ---
        try:
            super().__init__(field_name=field_name, dimensions=dimensions, data_dir=data_dir, **kwargs)
        except ValueError as ve: # Catch init errors from base (like bad dimensions)
             logger.critical(f"Error initializing base Field3D for {self.sephirah}: {ve}")
             raise # Re-raise critical init error

        # --- Initialize Sound Integration ---
        # Requires self.base_frequency to be loaded first
        self.sound_integration = None
        self.sound_enabled = False
        if self.base_frequency is not None and self.base_frequency > 0:
            try:
                self.sound_integration = SephirothSoundIntegration(base_frequency=self.base_frequency)
                # Check if the underlying SoundGenerator loaded correctly
                if self.sound_integration.sound_generator is not None:
                    self.sound_enabled = True
                    logger.info(f"Sound integration enabled for {self.sephirah}.")
                else:
                    logger.warning(f"Sound integration initialized but base SoundGenerator missing for {self.sephirah}. Sound disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize SephirothSoundIntegration for {self.sephirah}: {e}", exc_info=True)
        else:
             logger.warning(f"Sound integration disabled for {self.sephirah} due to invalid base frequency.")


        # --- Initialize Field State ---
        self._initialize_field_state() # Applies elemental pattern, normalizes

        # --- Update Metrics ---
        self.update_field_metrics() # Calculate initial metrics

        # --- Initialize Field Sound ---
        self.field_sound = None # Waveform array
        self.current_soundscape = None
        self._initialize_field_sound() # Applies sound to initialized field

        logger.info(f"SephirothField3D '{self.field_name}' (UUID: {self.uuid}) initialized successfully.")

    def _load_sephirah_properties(self):
        """Load properties, failing hard if essential data is missing."""
        logger.debug(f"Loading properties for {self.sephirah}...")
        self.aspects_data = aspect_dictionary.get_aspects(self.sephirah) # Load the full dict
        if not self.aspects_data or not self.aspects_data.get("detailed_aspects"):
            raise ValueError(f"FATAL: Could not load aspects data for {self.sephirah}")

        # Load properties, checking for existence and validity
        self.title = self.aspects_data.get("title", self.sephirah.capitalize())
        self.position = self.aspects_data.get("position")
        self.pillar = self.aspects_data.get("pillar")
        self.element = self.aspects_data.get("element", ELEMENT_NEUTRAL).lower()
        self.color = self.aspects_data.get("color")
        self.divine_name = self.aspects_data.get("divine_name")
        self.archangel = self.aspects_data.get("archangel")
        self.planetary_correspondence = self.aspects_data.get("planetary_correspondence")
        self.geometric_correspondence = self.aspects_data.get("geometric_correspondence")
        self.platonic_solid = aspect_dictionary.get_platonic_solid(self.sephirah)

        # Load modifiers and counts, using aspect_dictionary for validated access
        self.frequency_modifier = aspect_dictionary.get_frequency_modifier(self.sephirah) # Assumes this method exists
        self.phi_harmonic_count = aspect_dictionary.get_phi_harmonic_count(self.sephirah) # Assumes this method exists
        self.harmonic_count = aspect_dictionary.get_harmonic_count(self.sephirah) # Assumes this method exists
        self.stability_modifier = aspect_dictionary.get_stability_modifier(self.sephirah)
        self.resonance_multiplier = aspect_dictionary.get_resonance_multiplier(self.sephirah)
        self.dimensional_position = aspect_dictionary.get_dimensional_position(self.sephirah)
        self.divine_quality = aspect_dictionary.get_divine_quality(self.sephirah)

        # Get base frequency directly from the loaded instance via aspect_dictionary
        aspect_instance = aspect_dictionary.load_aspect_instance(self.sephirah)
        if aspect_instance and hasattr(aspect_instance, 'base_frequency'):
            freq = aspect_instance.base_frequency
            if isinstance(freq, (int, float)) and freq > 0:
                self.base_frequency = freq
            else:
                 raise ValueError(f"FATAL: Invalid base_frequency ({freq}) loaded for {self.sephirah}.")
        else:
             raise ValueError(f"FATAL: Cannot load base_frequency for {self.sephirah}.")

        # Check for None in essential properties after loading
        essential_props = ['position', 'pillar', 'element', 'color', 'divine_name', 'archangel',
                           'base_frequency', 'stability_modifier', 'resonance_multiplier']
        for prop in essential_props:
             if getattr(self, prop, None) is None:
                  # It might be acceptable for some props like planet to be None (Kether)
                  # But others are essential for function. Raise error for critical ones.
                  if prop in ['base_frequency', 'stability_modifier', 'resonance_multiplier']:
                      raise ValueError(f"FATAL: Essential property '{prop}' is None for {self.sephirah}.")
                  else:
                      logger.warning(f"Property '{prop}' is None for {self.sephirah}.")


        logger.info(f"Loaded properties for {self.sephirah}: Freq={self.base_frequency:.2f} Hz, Element={self.element}")

    def _initialize_field_state(self):
        """Set the initial energy state of the 3D field."""
        # Base class init already adds noise/base energy
        self._embed_specific_energy(percentage=0.4)
        self._normalize_field()
        logger.info(f"Initialized base field state for {self.sephirah}.")

    def _initialize_field_sound(self):
        """Initialize the field sound based on Sephiroth characteristics."""
        if not self.sound_enabled or self.sound_integration is None:
             logger.info(f"Sound disabled or integration unavailable for {self.sephirah}.")
             return
        try:
            # Generate waveform directly
            waveform = self.sound_integration.generate_sephiroth_field_sound(
                self.sephirah, duration=15.0, save=False
            )
            if waveform is None or len(waveform) == 0:
                # FAIL HARD: If the initial sound cannot be generated, log critical error.
                # Depending on design, could raise exception or just disable sound.
                logger.critical(f"CRITICAL: Failed to generate initial field sound waveform for {self.sephirah}. Sound features may fail.")
                self.sound_enabled = False # Disable sound for this instance
                self.field_sound = None
            else:
                self.field_sound = waveform # Store numpy array
                logger.info(f"Initialized {self.sephirah} field sound waveform.")
                # Apply sound to the initialized field
                self.apply_sound_to_field(self.field_sound, intensity=0.6)
        except Exception as e:
            logger.error(f"Error initializing field sound for {self.sephirah}: {e}", exc_info=True)
            self.sound_enabled = False
            self.field_sound = None

    # --- Element Embedding Functions (Keep 3D versions) ---
    def _embed_specific_energy(self, percentage):
        """Embed Sephiroth-specific energy in the 3D field."""
        if not (0 < percentage <= 1.0):
             logger.warning(f"Invalid percentage {percentage} for embedding. Skipping.")
             return
        logger.info(f"Embedding {self.sephirah} ({self.element}) energy (Strength: {percentage:.2f})")
        element_lower = self.element.lower()
        embed_func_map = {
             ELEMENT_FIRE: self._embed_fire_pattern, ELEMENT_WATER: self._embed_water_pattern,
             ELEMENT_AIR: self._embed_air_pattern, ELEMENT_EARTH: self._embed_earth_pattern,
             ELEMENT_AETHER: self._embed_aether_pattern, ELEMENT_SPIRIT: self._embed_aether_pattern,
             ELEMENT_QUINTESSENCE: self._embed_aether_pattern # Map aliases to aether
        }
        embed_func = embed_func_map.get(element_lower, self._embed_default_pattern)
        if embed_func == self._embed_default_pattern:
            logger.warning(f"Using default energy pattern for element '{self.element}'.")

        try:
            embed_func(percentage) # Modifies self.energy_potential and normalizes
            self._apply_element_resonance_sound() # Apply sound after embedding
        except Exception as e:
             logger.error(f"Error during energy embedding for {self.element}: {e}", exc_info=True)
             # Should this be fatal? Maybe not, field can still exist without pattern.

    def _embed_fire_pattern(self, percentage):
        indices=np.indices(self.dimensions,dtype=DEFAULT_FIELD_DTYPE); center=np.array([d//2 for d in self.dimensions],dtype=DEFAULT_FIELD_DTYPE)
        osc=(np.sin(indices[0]*PI/20)+np.sin(indices[1]*PI/15)+np.sin(indices[2]*PI/25))/3.0
        dist_sq=np.sum((indices-center[:,None,None,None])**2,axis=0); dist=np.sqrt(dist_sq)
        max_dist=np.sqrt(np.sum((np.array(self.dimensions)/2)**2)); norm_dist=dist/max(1e-6,max_dist)
        pattern=np.exp(-norm_dist)*(1.0+0.5*osc); self.energy_potential+=pattern*percentage; self._normalize_field()

    def _embed_water_pattern(self, percentage):
        indices=np.indices(self.dimensions,dtype=DEFAULT_FIELD_DTYPE)
        waves=(np.sin(indices[0]*PI/40+indices[1]*PI/30)+np.sin(indices[1]*PI/35+indices[2]*PI/45)+np.sin(indices[2]*PI/30+indices[0]*PI/40))/3.0
        pattern=0.5+0.5*waves; self.energy_potential+=pattern*percentage; self._normalize_field()

    def _embed_air_pattern(self, percentage):
        indices=np.indices(self.dimensions,dtype=DEFAULT_FIELD_DTYPE)
        pattern=0.7+0.2*np.sin(indices[0]*PI/10)+0.2*np.sin(indices[1]*PI/12)+0.2*np.sin(indices[2]*PI/8)
        pattern+=np.random.normal(0,0.1,self.dimensions); min_p,max_p=np.min(pattern),np.max(pattern)
        if max_p>min_p: pattern=(pattern-min_p)/(max_p-min_p)
        else: pattern=np.zeros_like(pattern)
        self.energy_potential+=pattern*percentage; self._normalize_field()

    def _embed_earth_pattern(self, percentage):
        indices=np.indices(self.dimensions,dtype=DEFAULT_FIELD_DTYPE); center=np.array([d//2 for d in self.dimensions],dtype=DEFAULT_FIELD_DTYPE)
        grid=np.abs(np.sin(indices[0]*PI/40))*np.abs(np.sin(indices[1]*PI/40))*np.abs(np.sin(indices[2]*PI/40))
        dist_sq=np.sum((indices-center[:,None,None,None])**2,axis=0); dist=np.sqrt(dist_sq)
        max_dist=np.sqrt(np.sum((np.array(self.dimensions)/2)**2)); norm_dist=dist/max(1e-6,max_dist)
        central=np.exp(-norm_dist*2); pattern=0.3*grid+0.7*central
        self.energy_potential+=pattern*percentage; self._normalize_field()

    def _embed_aether_pattern(self, percentage):
        indices=np.indices(self.dimensions,dtype=DEFAULT_FIELD_DTYPE); center=np.array([d//2 for d in self.dimensions],dtype=DEFAULT_FIELD_DTYPE)
        waves=(np.sin(indices[0]*PI/(20*PHI))+np.sin(indices[1]*PI/(20*PHI**2))+np.sin(indices[2]*PI/(20*PHI**3)))/3.0
        dist_sq=np.sum((indices-center[:,None,None,None])**2,axis=0); dist=np.sqrt(dist_sq)
        max_dist=np.sqrt(np.sum((np.array(self.dimensions)/2)**2)); norm_dist=dist/max(1e-6,max_dist)
        central=np.exp(-norm_dist**PHI); pattern=0.5*waves+0.5*central
        self.energy_potential+=pattern*percentage; self._normalize_field()

    def _embed_default_pattern(self, percentage):
        logger.warning(f"Using default energy pattern for element '{self.element}'.")
        indices=np.indices(self.dimensions,dtype=DEFAULT_FIELD_DTYPE); center=np.array([d//2 for d in self.dimensions],dtype=DEFAULT_FIELD_DTYPE)
        waves=(np.sin(indices[0]*PI/30)+np.sin(indices[1]*PI/30)+np.sin(indices[2]*PI/30))/3.0
        dist_sq=np.sum((indices-center[:,None,None,None])**2,axis=0); dist=np.sqrt(dist_sq)
        max_dist=np.sqrt(np.sum((np.array(self.dimensions)/2)**2)); norm_dist=dist/max(1e-6,max_dist)
        central=np.exp(-norm_dist); pattern=0.6*waves+0.4*central
        self.energy_potential+=pattern*percentage; self._normalize_field()

    def _apply_element_resonance_sound(self):
        if not self.sound_enabled or self.sound_integration is None: return
        try:
            element_sound = self.sound_integration.generate_element_resonance(
                sephirah=self.sephirah, element=self.element, duration=10.0 )
            if element_sound is not None and len(element_sound)>0:
                self.apply_sound_to_field(element_sound, intensity=0.5)
                logger.info(f"Applied {self.element} resonance sound to {self.sephirah}")
            else: logger.warning(f"Failed to generate element resonance sound for {self.element}")
        except Exception as e: logger.error(f"Failed to apply element resonance sound: {e}")

    # --- Harmonic Application (Keep 3D version) ---
    def _apply_harmonic_to_field(self, harmonic: Dict[str, float]):
        """Applies a single harmonic wave to the 3D field energy."""
        if not isinstance(harmonic, dict) or not all(k in harmonic for k in ['frequency', 'amplitude', 'phase']):
            logger.error(f"Invalid harmonic data received: {harmonic}"); return
        frequency = harmonic['frequency']; amplitude = harmonic['amplitude']; phase = harmonic['phase']
        if frequency <= 0 or amplitude < 0: return # Silently skip invalid harmonics

        try:
            indices = np.indices(self.dimensions, dtype=DEFAULT_FIELD_DTYPE)
            base_wl_hz=100.0; base_wl_grid=50.0;
            wl = base_wl_grid * (base_wl_hz / frequency) * self.resonance_multiplier
            wl = max(2.0, wl) # Enforce minimum wavelength

            wave = sum(np.sin(2 * PI * indices[i] / wl + phase + i * PI / 3) for i in range(3)) / 3.0 * amplitude

            # Element scaling
            el = self.element.lower(); scale = 1.0
            if el==ELEMENT_FIRE: scale=0.5+0.5*min(1.,frequency/500)
            elif el==ELEMENT_WATER: scale=0.5+0.5*max(0.,1-abs(frequency-300)/300)
            elif el==ELEMENT_AIR: scale=0.5+0.5*min(1.,frequency/400)
            elif el==ELEMENT_EARTH: scale=0.5+0.5*max(0.,1-frequency/300)
            elif el in [ELEMENT_AETHER,ELEMENT_SPIRIT,ELEMENT_QUINTESSENCE]: scale=0.5+0.5*(1-min(1.,abs(frequency-SOLFEGGIO_MI)/500))

            self.energy_potential += wave * scale * self.stability_modifier
            # Note: Normalization should ideally happen *after* all harmonics/sounds are applied in a step.
            # self._normalize_field() # Moved out

        except Exception as e:
            logger.error(f"Error applying harmonic {frequency}Hz: {e}", exc_info=True)


    def apply_sound_to_field(self, sound_waveform: np.ndarray, intensity: float = 0.5):
        """Applies sound waveform modulation to the 3D field energy potential."""
        if not self.sound_enabled: return False
        if sound_waveform is None or sound_waveform.size == 0: logger.warning("Empty sound waveform."); return False
        if not isinstance(sound_waveform, np.ndarray): raise TypeError("sound_waveform must be a numpy array.")
        intensity = max(0.0, min(1.0, intensity))
        if intensity == 0: return False

        try:
            max_abs = np.max(np.abs(sound_waveform));
            norm_sound = sound_waveform / max_abs if max_abs > 1e-9 else np.zeros_like(sound_waveform)
            sound_rms = np.sqrt(np.mean(norm_sound**2))
            res_mult = self.resonance_multiplier
            mod_factor = 1.0 + (sound_rms * intensity * res_mult) # Multiplicative factor

            self.energy_potential *= mod_factor
            # Normalization should happen after all updates in a simulation step
            # self._normalize_field() # Removed - call explicitly after all effects in a step

            logger.info(f"Applied sound modulation (RMS:{sound_rms:.3f}, Intensity:{intensity:.2f}) to {self.sephirah}")
            return True
        except Exception as e:
            logger.error(f"Error applying sound waveform to field: {e}", exc_info=True)
            return False

    # --- Sacred Geometry & Platonic Solid Application (Using 3D Embedders) ---
    def apply_sacred_geometry(self, geometry_type: str, strength: float = DEFAULT_GEOMETRY_EMBED_STRENGTH, position: Optional[tuple] = None):
        """Applies a sacred geometry pattern using 3D embedders."""
        if not GEOMETRY_EMBEDDERS_AVAILABLE:
            logger.error(f"Cannot apply {geometry_type}: Embedder functions unavailable.")
            return False # Fail hard if embedders missing

        geo_lower = geometry_type.lower()
        if geo_lower not in EMBED_FUNCTIONS:
            logger.error(f"Unsupported or unavailable geometry type for embedding: {geometry_type}")
            return False # Fail hard if specific embedder missing

        if position is None: position = tuple(d // 2 for d in self.dimensions)
        elif len(position) != 3: raise ValueError("Position must be a 3D tuple.")
        strength = max(0.0, min(1.0, strength))

        try:
            embed_func = EMBED_FUNCTIONS[geo_lower]
            size_param = min(self.dimensions) // 8 # Example size - adjust per geometry if needed
            logger.debug(f"Applying {geometry_type} at {position} with strength {strength}")

            # Embedder modifies self.energy_potential IN PLACE
            embed_func(self.energy_potential, position, size_param, strength)

            self._normalize_field() # Normalize after modification
            self._apply_sacred_geometry_sound(geometry_type, strength)
            logger.info(f"Applied {geometry_type} to {self.sephirah} field")
            return True
        except Exception as e:
            logger.error(f"Error applying sacred geometry {geometry_type}: {e}", exc_info=True)
            return False # Indicate failure


    def _apply_sacred_geometry_sound(self, geometry_type, strength):
        """Applies sound associated with sacred geometry."""
        if not self.sound_enabled or self.sound_integration is None: return
        try:
            sound = self.sound_integration.generate_sacred_geometry_sound(geometry_type, self.sephirah, strength, 8.0)
            if sound is not None and len(sound) > 0: self.apply_sound_to_field(sound, intensity=strength*0.8)
            else: logger.warning(f"Failed to generate sound for geometry {geometry_type}")
        except Exception as e: logger.error(f"Error applying sacred geometry sound: {e}")


    def apply_platonic_solid(self, solid_type: Optional[str] = None, strength: float = DEFAULT_PLATONIC_EMBED_STRENGTH):
        """Applies a platonic solid pattern using 3D embedders."""
        if not GEOMETRY_EMBEDDERS_AVAILABLE:
             logger.error(f"Cannot apply platonic solid: Embedder functions unavailable.")
             return False

        if solid_type is None: solid_type = self.platonic_solid
        if not solid_type: logger.warning(f"No platonic solid specified or assigned for {self.sephirah}"); return False
        solid_lower = solid_type.lower()

        if solid_lower not in EMBED_FUNCTIONS:
             logger.error(f"Unsupported or unavailable platonic solid embedder: {solid_type}")
             return False

        strength = max(0.0, min(1.0, strength))
        try:
            embed_func = EMBED_FUNCTIONS[solid_lower]
            center = tuple(d // 2 for d in self.dimensions)
            edge_length = min(self.dimensions) // 4 # Example size parameter

            logger.debug(f"Applying {solid_type} at {center} with strength {strength}")
            # Embedder modifies self.energy_potential IN PLACE, using element for pattern
            embed_func(self.energy_potential, center, edge_length, strength, self.element)

            self._normalize_field()
            self._apply_platonic_solid_sound(solid_type)
            logger.info(f"Applied {solid_type} to {self.sephirah} field")
            return True
        except Exception as e:
            logger.error(f"Error applying platonic solid {solid_type}: {e}", exc_info=True)
            return False


    def _apply_platonic_solid_sound(self, solid_type):
        """Applies sound associated with platonic solid."""
        if not self.sound_enabled or self.sound_integration is None: return
        try:
            sound = self.sound_integration.generate_platonic_solid_sound(solid_type, self.sephirah, self.element, 10.0)
            if sound is not None and len(sound) > 0: self.apply_sound_to_field(sound, intensity=0.6)
            else: logger.warning(f"Failed to generate sound for platonic solid {solid_type}")
        except Exception as e: logger.error(f"Error applying platonic solid sound: {e}")


    # --- Stabilization (Keep 3D version using SciPy) ---
    def _stabilize_field(self):
        """Stabilize the 3D field using Gaussian filter and edge containment."""
        logger.debug("Stabilizing 3D field")
        try:
            sigma = max(1.0, min(self.dimensions) / 64.0)
            # Use reflect mode for boundaries to avoid edge artifacts from smoothing
            smoothed = scipy.ndimage.gaussian_filter(self.energy_potential, sigma=sigma, mode='reflect')
            blend = 0.3; self.energy_potential = (1.0 - blend)*self.energy_potential + blend*smoothed

            # Harmonic stabilization pattern
            indices = np.indices(self.dimensions, dtype=DEFAULT_FIELD_DTYPE)
            stab_pattern = sum(np.sin(2*PI*indices[i]/(10*PHI**(i+1))) for i in range(3)) / 3.0
            stab_pattern *= 0.1 * self.stability_modifier # Reduced strength
            self.energy_potential += stab_pattern

            self._apply_edge_containment_3d()
            self._normalize_field() # Normalize after stabilization steps
            logger.info(f"Field stabilized with sigma {sigma:.2f}, modifier {self.stability_modifier:.2f}")
        except Exception as e:
             logger.error(f"Error during field stabilization: {e}", exc_info=True)


    # --- Keep _apply_edge_containment_3d ---
    def _apply_edge_containment_3d(self):
        """Apply containment to 3D field edges."""
        try:
            combined_mask = np.ones_like(self.energy_potential, dtype=np.float32)
            for i, dim_size in enumerate(self.dimensions):
                edge_falloff = np.ones(dim_size, dtype=np.float32)
                width = max(1, dim_size // 10); width = min(width, dim_size // 2) # Prevent width > half
                if width < 1: continue
                ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
                edge_falloff[:width] = ramp; edge_falloff[-width:] = ramp[::-1]
                shape=[1]*3; shape[i]=dim_size
                combined_mask *= edge_falloff.reshape(shape)
            self.energy_potential *= combined_mask
        except Exception as e: logger.error(f"Error applying edge containment: {e}", exc_info=True)


    # --- Metrics Update (Keep 3D version) ---
    def update_field_metrics(self):
        """Update metrics about the 3D field's current state."""
        # (Implementation from previous step, ensure it uses self.energy_potential)
        if not hasattr(self, 'metrics'): self.metrics = {}
        try:
            energy = self.energy_potential
            metrics = self.metrics # Update existing dict
            metrics['energy_mean'] = float(np.mean(energy))
            metrics['energy_std'] = float(np.std(energy))
            metrics['energy_min'] = float(np.min(energy))
            metrics['energy_max'] = float(np.max(energy))
            metrics['stability'] = float(1.0 - min(1.0, np.var(energy) * 10))

            mid_x, mid_y, mid_z = [d // 2 for d in self.dimensions]
            slices = [energy[:, mid_y, mid_z], energy[mid_x, :, mid_z], energy[mid_x, mid_y, :]]
            ratios = []
            for s in slices:
                 if s.size > 1:
                     fft_vals = np.abs(np.fft.rfft(s))[1:]
                     if fft_vals.size > 0: ratios.append(np.max(fft_vals) / max(1e-10, np.mean(fft_vals)))
                     else: ratios.append(0)
                 else: ratios.append(0)
            metrics['pattern_strength'] = float(min(1.0, np.mean(ratios) / 10.0)) if ratios else 0.0
            metrics['coherence'] = float(self._calculate_spatial_coherence_3d())
            metrics['resonance'] = self.calculate_frequency_resonance(self.base_frequency)
            metrics['timestamp'] = datetime.now().isoformat()
            logger.debug(f"Updated 3D field metrics for {self.sephirah}")
            return metrics
        except Exception as e:
             logger.error(f"Error updating field metrics: {e}", exc_info=True)
             return self.metrics # Return existing metrics on error

    # --- Keep _calculate_spatial_coherence_3d ---
    def _calculate_spatial_coherence_3d(self):
        """Calculate spatial coherence for the 3D field."""
        coherence_sum = 0.0; count = 0
        try:
            for axis in range(3):
                idx1=[slice(None)]*3; idx2=[slice(None)]*3; idx1[axis]=slice(None,-1); idx2[axis]=slice(1,None)
                diff = np.abs(self.energy_potential[tuple(idx1)] - self.energy_potential[tuple(idx2)])
                coherence_axis = 1.0 - np.minimum(1.0, diff); coherence_sum += np.sum(coherence_axis); count += diff.size
            return coherence_sum / count if count > 0 else 0.0
        except Exception as e: logger.error(f"Coherence calculation error: {e}"); return 0.0

    # --- Keep calculate_frequency_resonance ---
    def calculate_frequency_resonance(self, frequency: float) -> float:
         """Calculate resonance of the field with a given frequency."""
         # Delegate to aspect instance if possible, otherwise use field-based calculation
         if aspect_dictionary:
             aspect_instance = aspect_dictionary.load_aspect_instance(self.sephirah)
             if aspect_instance and hasattr(aspect_instance, 'calculate_resonance'):
                 try:
                      return aspect_instance.calculate_resonance(frequency)
                 except Exception as e:
                      logger.warning(f"Error calling aspect resonance for {self.sephirah}: {e}")
         # Fallback field-based calculation
         if self.base_frequency <= 0: return 0.0
         freq_diff = abs(frequency - self.base_frequency)
         width = self.base_frequency * 0.2 # Example: 20% width
         resonance = np.exp(-(freq_diff**2) / (2 * max(1e-6, width**2)))
         return resonance * self.resonance_multiplier


    # --- Visualization Methods (Keep 3D versions) ---
    def visualize_field_3d(self, threshold=0.3, show_overlay=True, title=None, save_path=None):
         # (Implementation from previous step, ensure imports are handled)
        try:
             import matplotlib.pyplot as plt
             from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
             logger.error("Matplotlib or mpl_toolkits not found. Cannot visualize 3D field.")
             return None
        # (Rest of the 3D visualization code from previous step...)
        fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
        mask = self.energy_potential > threshold
        if not np.any(mask): logger.warning(f"No voxels above threshold {threshold} for 3D visualization."); plt.close(fig); return None
        x, y, z = np.indices(self.dimensions); cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=threshold, vmax=np.max(self.energy_potential[mask]))
        colors = cmap(norm(self.energy_potential)); colors[..., 3] = 0.4 * mask
        ax.voxels(x[mask], y[mask], z[mask], mask[mask], facecolors=colors[mask], edgecolor='k', linewidth=0.1) # Pass coordinates
        if title is None: title = f"{self.sephirah.capitalize()} 3D Field (> {threshold:.2f})"
        ax.set_title(title); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        if show_overlay:
             ax.text2D(0.05, 0.95, f"E: {self.element.capitalize()}", transform=ax.transAxes, color='white', bbox=dict(facecolor='black', alpha=0.5))
             if self.platonic_solid: ax.text2D(0.05, 0.90, f"P: {self.platonic_solid.capitalize()}", transform=ax.transAxes, color='white', bbox=dict(facecolor='black', alpha=0.5))
        max_dim = np.max(self.dimensions); mid_pt = np.array(self.dimensions)/2
        ax.set_xlim(mid_pt[0]-max_dim/2, mid_pt[0]+max_dim/2); ax.set_ylim(mid_pt[1]-max_dim/2, mid_pt[1]+max_dim/2); ax.set_zlim(mid_pt[2]-max_dim/2, mid_pt[2]+max_dim/2)
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight'); logger.info(f"3D Viz saved: {save_path}")
        # Don't call plt.show() here, return the figure object
        return fig


    def visualize_field_2d(self, axis='z', slice_index=None, show_contour=True, title=None, save_path=None):
        # (Implementation from previous step, ensure imports are handled)
        try:
             import matplotlib.pyplot as plt
        except ImportError:
             logger.error("Matplotlib not found. Cannot visualize 2D slice.")
             return None
        # (Rest of the 2D visualization code from previous step...)
        dims = self.dimensions
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map: raise ValueError("Axis must be 'x', 'y', or 'z'")
        axis_idx = axis_map[axis]
        if slice_index is None: slice_index = dims[axis_idx] // 2
        slice_index = max(0, min(slice_index, dims[axis_idx]-1)) # Clamp index

        if axis == 'z': field_2d = self.energy_potential[:, :, slice_index].T; xl, yl = 'X', 'Y'; extent = [0, dims[0], 0, dims[1]]
        elif axis == 'y': field_2d = self.energy_potential[:, slice_index, :].T; xl, yl = 'X', 'Z'; extent = [0, dims[0], 0, dims[2]]
        else: # axis == 'x'
             field_2d = self.energy_potential[slice_index, :, :].T; xl, yl = 'Y', 'Z'; extent = [0, dims[1], 0, dims[2]]

        fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(field_2d, origin='lower', cmap='viridis', extent=extent)
        plt.colorbar(im, ax=ax, label='Energy Potential')
        if show_contour and np.ptp(field_2d) > 1e-6:
            levels = np.linspace(np.min(field_2d), np.max(field_2d), 10)
            ax.contour(field_2d, colors='white', alpha=0.3, levels=levels, extent=extent)
        if title is None: title = f"{self.sephirah.capitalize()} Slice ({axis.upper()}={slice_index})"
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect('equal')
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight'); logger.info(f"2D Slice saved: {save_path}")
        return fig

    # --- Keep Soul Transformation methods (ensure checks for None/errors) ---
    def transform_soul(self, soul, duration=1.0):
        """Transform a soul in this 3D Sephiroth field."""
        if not hasattr(soul, 'aspects'): raise AttributeError("Soul object missing 'aspects'")
        duration = max(0.1, duration) # Ensure positive duration
        logger.info(f"Transforming soul in {self.sephirah} field for {duration} units")
        try:
            initial_state = self._get_soul_state(soul)
            resonance = self._calculate_soul_resonance(soul)
            strength = resonance * min(1.0, duration / 5.0)
            transformation = self._apply_transformation(soul, strength)

            transformation_sound = self._generate_transformation_sound(soul, strength, resonance)
            if transformation_sound is not None: self.apply_sound_to_field(transformation_sound, intensity=0.4)

            final_state = self._get_soul_state(soul)
            path_record = { "soul_id": getattr(soul, 'id', 'unknown'), "sephirah": self.sephirah,
                            "entry_time": datetime.now().isoformat(), "duration": duration,
                            "resonance": resonance, "transformation_strength": strength,
                            "aspects_gained": transformation.get("aspects_gained", []),
                            "initial_state": initial_state, "final_state": final_state }
            if not hasattr(self, 'soul_paths'): self.soul_paths = []
            self.soul_paths.append(path_record) # Log the path

            logger.info(f"Soul transformation complete: resonance={resonance:.4f}, strength={strength:.4f}")
            return { "success": True, "resonance": resonance, "strength": strength, **transformation }
        except Exception as e:
            logger.error(f"Error during soul transformation in {self.sephirah}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # --- (Keep _generate_transformation_sound, _get_soul_state, _calculate_soul_resonance, _apply_transformation) ---
    # --- Ensure they handle potential errors and invalid soul objects gracefully ---
    def _generate_transformation_sound(self, soul, strength, resonance):
        if not self.sound_enabled or self.sound_integration is None: return None
        try:
            return self.sound_integration.generate_transformation_sound(
                 self.sephirah, self.element, strength, resonance, 5.0)
        except Exception as e: logger.error(f"Error generating transformation sound: {e}"); return None

    def _get_soul_state(self, soul):
        if soul is None: return {}
        return { "aspect_count": len(getattr(soul, 'aspects', {})),
                 "resonance_profile": getattr(soul, 'resonance_profile', {}),
                 "strength": getattr(soul, 'strength', 0.0),
                 "stability": getattr(soul, 'stability', 0.0),
                 "current_aspects": list(getattr(soul, 'aspects', {}).keys()) }

    def _calculate_soul_resonance(self, soul):
        if soul is None: return 0.0
        base_resonance = 0.5; soul_aspects = getattr(soul, 'aspects', {})
        sephirah_primary = set(self.aspects_data.get('primary_aspects', [])) # Use loaded aspects_data
        match_count = sum(1 for aspect in soul_aspects if aspect in sephirah_primary)
        aspect_factor = min(1.0, match_count / len(sephirah_primary)) if sephirah_primary else 0.0
        soul_frequency = getattr(soul, 'primary_frequency', None)
        freq_factor = self.calculate_frequency_resonance(soul_frequency) if soul_frequency else 0.0
        resonance = (0.4*base_resonance + 0.4*aspect_factor + 0.2*freq_factor) * self.resonance_multiplier
        return max(0.0, min(1.0, resonance))

    def _apply_transformation(self, soul, strength):
        if soul is None or not hasattr(soul, 'aspects'): return {"error": "Invalid soul object"}
        results = {"aspects_gained": [], "aspects_strengthened": [], "divine_qualities_imparted": []}
        primary_aspects = self.aspects_data.get('primary_aspects', [])
        secondary_aspects = self.aspects_data.get('secondary_aspects', [])
        detailed_aspects = self.aspects_data.get('detailed_aspects', {})

        num_primary = max(1, int(len(primary_aspects) * strength))
        for aspect_name in primary_aspects[:num_primary]:
            if aspect_name not in soul.aspects:
                soul.aspects[aspect_name] = {'strength': 0.3+(0.4*strength), 'source': self.sephirah, 'data': detailed_aspects.get(aspect_name, {}), 'time': datetime.now().isoformat()}
                results["aspects_gained"].append(aspect_name)
            else:
                current = soul.aspects[aspect_name].get('strength', 0.0)
                soul.aspects[aspect_name]['strength'] = min(1.0, current + (0.2 * strength))
                results["aspects_strengthened"].append(aspect_name)

        num_secondary = max(0, int(len(secondary_aspects) * strength * 0.5))
        for aspect_name in secondary_aspects[:num_secondary]:
             if aspect_name not in soul.aspects:
                 soul.aspects[aspect_name] = {'strength': 0.2+(0.3*strength), 'source': self.sephirah, 'data': detailed_aspects.get(aspect_name, {}), 'time': datetime.now().isoformat()}
                 results["aspects_gained"].append(aspect_name)

        # Impart divine qualities
        if self.divine_quality:
             quality_name = self.divine_quality.get('name', 'Unknown'); quality_strength = self.divine_quality.get('strength', 0.5) * strength
             if not hasattr(soul, 'divine_qualities'): soul.divine_qualities = {}
             soul.divine_qualities[quality_name] = {'strength': quality_strength, 'source': self.sephirah, 'time': datetime.now().isoformat()}
             results["divine_qualities_imparted"].append(quality_name)

        # Apply element influence
        if self.element and hasattr(soul, 'elemental_influence'):
             if not isinstance(soul.elemental_influence, dict): soul.elemental_influence = {}
             current = soul.elemental_influence.get(self.element, 0.0)
             soul.elemental_influence[self.element] = min(1.0, current + (0.3 * strength))

        # Update stability and strength
        if hasattr(soul, 'stability'): soul.stability = min(1.0, getattr(soul, 'stability', 0.5) + (0.1 * strength * self.stability_modifier))
        if hasattr(soul, 'strength'): soul.strength = min(1.0, getattr(soul, 'strength', 0.5) + len(results["aspects_gained"]) * 0.05 + (0.05 * strength))

        logger.info(f"Transformation applied to soul: gained={len(results['aspects_gained'])}, strengthened={len(results['aspects_strengthened'])}")
        return results

    # --- Gateway Preparation (Keep 3D version) ---
    def prepare_for_gateway(self, gateway_key: str):
        """Prepare the 3D field for gateway operations."""
        logger.info(f"Preparing {self.sephirah} field for {gateway_key} gateway")
        gateway_key_lower = gateway_key.lower()

        if not self._verify_gateway_compatibility(gateway_key_lower):
            logger.error(f"{gateway_key} gateway not compatible with {self.sephirah}. Preparation aborted.")
            return {"success": False, "reason": "incompatible_gateway"}

        # Apply relevant sacred geometry
        geo_map = {"tetrahedron": "merkaba", "octahedron": "flower_of_life",
                   "hexahedron": "seed_of_life", "icosahedron": "sri_yantra",
                   "dodecahedron": "metatrons_cube"}
        geometry_to_apply = geo_map.get(gateway_key_lower)
        if geometry_to_apply:
            success_geo = self.apply_sacred_geometry(geometry_to_apply, strength=0.8)
            if not success_geo: logger.warning(f"Failed to apply sacred geometry {geometry_to_apply} for gateway.")
        else: logger.warning(f"No specific sacred geometry defined for gateway {gateway_key}")

        # Apply the platonic solid itself (stronger effect)
        success_platonic = self.apply_platonic_solid(gateway_key_lower, strength=0.9)
        if not success_platonic: logger.warning(f"Failed to apply platonic solid {gateway_key} for gateway.")

        # Stabilize and amplify
        self._stabilize_field()
        self._amplify_gateway_frequencies(gateway_key_lower)

        gateway_metrics = self._calculate_gateway_metrics(gateway_key_lower)
        # Determine overall success based on critical steps
        overall_success = success_platonic # Require at least platonic solid embedding
        logger.info(f"Gateway preparation complete. Success: {overall_success}")
        return {"success": overall_success, **gateway_metrics}

    # --- Keep _verify_gateway_compatibility, _amplify_gateway_frequencies, ---
    # --- _calculate_gateway_metrics, _calculate_gateway_resonance ---
    def _verify_gateway_compatibility(self, gateway_key):
        if aspect_dictionary is None: return False
        sephiroth_in_gateway = aspect_dictionary.get_gateway_sephiroth(gateway_key)
        return self.sephirah in sephiroth_in_gateway

    def _amplify_gateway_frequencies(self, gateway_key):
        freq_map = { "tetrahedron": [396., 417., 528.], "octahedron": [285., 396., 528.],
                     "hexahedron": [174., 285., 396.], "icosahedron": [417., 528., 639.],
                     "dodecahedron": [528., 639., 741.] }
        frequencies = freq_map.get(gateway_key, [396., 417., 528.])
        logger.info(f"Amplifying gateway frequencies for {gateway_key}: {frequencies}")
        for freq in frequencies:
             harmonic = {'frequency': freq, 'amplitude': 0.7, 'phase': 0.0}
             self._apply_harmonic_to_field(harmonic)
        self._normalize_field()

    def _calculate_gateway_metrics(self, gateway_key):
        self.update_field_metrics() # Ensure metrics are fresh
        metrics = { "stability": self.metrics.get('stability', 0.0), "energy_level": self.metrics.get('energy_level', 0.0),
                    "coherence": self.metrics.get('coherence', 0.0), "resonance": self._calculate_gateway_resonance(gateway_key),
                    "gateway_key": gateway_key, "timestamp": datetime.now().isoformat() }
        logger.info(f"Gateway metrics: S={metrics['stability']:.2f}, R={metrics['resonance']:.2f}")
        return metrics

    def _calculate_gateway_resonance(self, gateway_key):
        element_map = {"tetrahedron":ELEMENT_FIRE, "octahedron":ELEMENT_AIR, "hexahedron":ELEMENT_EARTH,
                       "icosahedron":ELEMENT_WATER, "dodecahedron":ELEMENT_AETHER}
        gateway_element = element_map.get(gateway_key); element_match = 0.0
        if self.element == gateway_element: element_match = 1.0
        elif gateway_element is not None: element_match = 0.5 # Partial compatibility
        else: element_match = 0.2

        energy_dist = np.histogram(self.energy_potential, bins=10, range=(0, 1))[0]
        if np.sum(energy_dist)>0: energy_dist=energy_dist/np.sum(energy_dist)
        else: energy_dist = np.zeros(10)

        pattern_match = 0.0
        if gateway_key=="tetrahedron": pattern_match=min(1.,np.sum(energy_dist[-3:])*2)
        elif gateway_key=="octahedron": pattern_match=max(0.,min(1.,1-np.std(energy_dist)*5))
        elif gateway_key=="hexahedron": pattern_match=min(1.,np.sum(energy_dist[3:7])*2)
        elif gateway_key=="icosahedron": pattern_match=min(1.,np.sum([energy_dist[i]*(i/9.0) for i in range(10)])*2)
        elif gateway_key=="dodecahedron":
            phi_ratios=[(1/PHI)**(5-i) for i in range(10)]; phi_ratios=np.array(phi_ratios)/np.sum(phi_ratios)
            pattern_match=max(0.,1-np.sum(np.abs(energy_dist-phi_ratios))/2)

        resonance=(0.6*element_match + 0.4*pattern_match)*self.resonance_multiplier
        return max(0.,min(1., resonance))

    # --- Keep get_field_state (already adapted) ---
    def get_field_state(self) -> Dict[str, Any]:
        """Returns the current comprehensive state of the 3D field."""
        # (Implementation from previous step is fine)
        self.update_field_metrics()
        state = { "field_name": self.field_name, "class": self.__class__.__name__, "uuid": self.uuid,
                  "sephirah": self.sephirah, "title": self.title, "element": self.element, "platonic_solid": self.platonic_solid,
                  "dimensions": self.dimensions, "metrics": self.metrics.copy(), "divine_quality": self.divine_quality,
                  "stability_modifier": self.stability_modifier, "resonance_multiplier": self.resonance_multiplier,
                  "dimensional_position": self.dimensional_position, "base_frequency": self.base_frequency,
                  "timestamp": datetime.now().isoformat(),
                  "primary_aspects": self.aspects_data.get("primary_aspects", []),
                  "secondary_aspects": self.aspects_data.get("secondary_aspects", []),
                  "energy_potential_mean": self.metrics.get('energy_mean', 0.0), # Use metric
                  "energy_potential_std": self.metrics.get('energy_std', 0.0) }
        return state

    # --- Keep __str__ and __repr__ (already adapted) ---
    def __str__(self):
        metrics = self.metrics; energy = metrics.get('energy_mean', 'N/A'); stability = metrics.get('stability', 'N/A')
        energy_str = f"{energy:.4f}" if isinstance(energy, (int, float)) else energy
        stability_str = f"{stability:.4f}" if isinstance(stability, (int, float)) else stability
        aspect_count = len(self.aspects_data.get('primary_aspects', [])) + len(self.aspects_data.get('secondary_aspects', []))
        return (f"SephirothField3D: {self.sephirah.capitalize()} ({self.title})\n"
                f"  Element: {self.element.capitalize()} | Platonic: {self.platonic_solid.capitalize() if self.platonic_solid else 'None'}\n"
                f"  Dims: {self.dimensions} | Base Freq: {self.base_frequency:.2f} Hz\n"
                f"  Energy: {energy_str} | Stability: {stability_str} | Resonance Mult: {self.resonance_multiplier:.2f}\n"
                f"  Aspects Defined: {aspect_count}")

    def __repr__(self):
         return f"<{self.__class__.__name__} name='{self.field_name}' sephirah='{self.sephirah}' dims={self.dimensions} uuid='{self.uuid}'>"



        
        #


