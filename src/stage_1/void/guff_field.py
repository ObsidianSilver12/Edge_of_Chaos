# Gemini Code
"""
GuffField3D Module

Implements the 3D Guff field for soul spark strengthening and creator harmonization.
Represents the dimensional space where sparks align with creator frequencies.

Author: Soul Development Framework Team - Refactored to 3D with Type Hinting & Strict Errors
"""

import numpy as np
import logging
import os
import uuid
from datetime import datetime
import json
from typing import Optional, Tuple, Dict, Any, List, Union

# Use try-except for optional dependencies
try:
    import scipy.ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some features like smoothing might be limited.")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # For 3D plots if needed
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    logging.warning("Matplotlib not found. Visualization methods will be disabled.")

# --- Constants ---
# Attempt to import constants, raise error if essential ones are missing
try:
    from src.constants import (
        DEFAULT_DIMENSIONS_3D, DEFAULT_FIELD_DTYPE, DEFAULT_COMPLEX_DTYPE,
        PHI, GOLDEN_RATIO, PI, GUFF_BASE_FREQUENCY, GUFF_RESONANCE_QUALITY,
        GUFF_FORMATION_THRESHOLD, GUFF_STRENGTHENING_FACTOR, GUFF_FIBONACCI_COUNT,
        GUFF_SOUND_INTENSITY, GUFF_RECEPTION_SOUND_INTENSITY, GUFF_FORMATION_SOUND_INTENSITY,
        OUTPUT_DIR_BASE, DATA_DIR_BASE, SAMPLE_RATE, MAX_AMPLITUDE,
        LOG_LEVEL, LOG_FORMAT, ELEMENT_AETHER, ELEMENT_QUINTESSENCE, ELEMENT_SPIRIT,
        SOLFEGGIO_963 # Example Kether freq needed for sound gen
    )
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. GuffField3D cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
# Adjust paths as needed
SOUND_MODULES_AVAILABLE = False
SoundGenerator = NoiseGenerator = UniverseSounds = SephirothSoundIntegration = None
try:
    from sound.sound_generator import SoundGenerator
    from sound.white_noise import NoiseGenerator
    from sound.sounds_of_universe import UniverseSounds
    from sound.sephiroth_sound_integration import SephirothSoundIntegration
    SOUND_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import sound modules for GuffField3D: {e}. Sound features disabled.")

# Needed for relationship calculations etc.
try:
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    ASPECT_DICT_AVAILABLE = True
except ImportError:
    logging.critical("Failed to import SephirothAspectDictionary. GuffField3D cannot function.")
    aspect_dictionary = None
    ASPECT_DICT_AVAILABLE = False


# Configure logging
log_file_path = os.path.join("logs", "guff_field_3d.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('guff_field_3d')

class GuffField3D(object): # Inherit from object
    """
    Implementation of the 3D Guff field. Strengthens soul sparks via resonance.
    Enforces strict error handling.
    """

    def __init__(self, dimensions: Tuple[int, int, int] = DEFAULT_DIMENSIONS_3D,
                 field_name: str = "guff_field", data_dir: str = DATA_DIR_BASE):
        """
        Initialize a new 3D Guff field. Fails hard on invalid configuration.
        """
        # --- Input Validation ---
        if not isinstance(dimensions, tuple) or len(dimensions) != 3 or not all(isinstance(d, int) and d > 0 for d in dimensions):
            raise ValueError(f"Dimensions must be a tuple of 3 positive integers, got {dimensions}")
        if not field_name or not isinstance(field_name, str):
            raise ValueError("Field name must be a non-empty string.")

        self.field_name = field_name
        self.dimensions = dimensions
        self.uuid = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.data_dir = os.path.join(data_dir, "fields", field_name)
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create data directory {self.data_dir}: {e}")
            raise

        logger.info(f"Initializing GuffField3D '{self.field_name}' (UUID: {self.uuid}) with dimensions {self.dimensions}")

        # --- Initialize 3D Field Arrays ---
        try:
            self.energy_potential: np.ndarray = np.zeros(dimensions, dtype=DEFAULT_FIELD_DTYPE)
            self.harmonic_field: np.ndarray = np.zeros(dimensions, dtype=DEFAULT_COMPLEX_DTYPE)
        except MemoryError as me:
            logger.critical(f"CRITICAL: Insufficient memory for Guff field arrays {dimensions}. {me}")
            raise
        except Exception as e:
            logger.critical(f"CRITICAL: Error initializing Guff field arrays: {e}", exc_info=True)
            raise

        # --- Properties ---
        self.golden_ratio: float = GOLDEN_RATIO
        self.resonance_quality: float = GUFF_RESONANCE_QUALITY
        self.soul_formation_threshold: float = GUFF_FORMATION_THRESHOLD
        self.strengthening_factor: float = GUFF_STRENGTHENING_FACTOR
        if not (0 < self.resonance_quality <= 1): raise ValueError("GUFF_RESONANCE_QUALITY must be between 0 and 1")
        if not (0 < self.soul_formation_threshold < 1): raise ValueError("GUFF_FORMATION_THRESHOLD must be between 0 and 1")
        if self.strengthening_factor <= 1.0: raise ValueError("GUFF_STRENGTHENING_FACTOR must be > 1.0")

        # Fibonacci sequence
        self.fibonacci_sequence: List[int] = self._generate_fibonacci_sequence(GUFF_FIBONACCI_COUNT)

        # --- Initialize Sound Components ---
        self.sound_enabled: bool = SOUND_MODULES_AVAILABLE
        self.sound_output_dir: str = os.path.join(OUTPUT_DIR_BASE, "guff")
        os.makedirs(self.sound_output_dir, exist_ok=True)
        self.sound_generator: Optional[SoundGenerator] = None
        self.noise_gen: Optional[NoiseGenerator] = None
        self.universe_sounds: Optional[UniverseSounds] = None
        self.sephiroth_sounds: Optional[SephirothSoundIntegration] = None
        self.guff_sound: Optional[np.ndarray] = None # Waveform array
        self.base_frequency: float = 0.0

        if self.sound_enabled:
            self._initialize_sound_objects() # Fails hard if objects cannot init
            self.base_frequency = GUFF_BASE_FREQUENCY
            if self.base_frequency <= 0: raise ValueError("GUFF_BASE_FREQUENCY must be positive.")
        else:
            logger.warning("Sound modules failed to import. Sound features disabled for GuffField.")

        # --- Initialize Field State and Sound ---
        self.metrics: Dict[str, Any] = {"creation_time": self.creation_time}
        self._initialize_field() # Fills arrays
        if self.sound_enabled:
            self._initialize_sound_system() # Applies initial sound

        logger.info(f"Guff field '{self.field_name}' initialized successfully.")

    def _initialize_sound_objects(self):
        """Helper to initialize sound objects, failing hard if unavailable."""
        if not SOUND_MODULES_AVAILABLE:
             raise RuntimeError("Essential sound modules are not available for GuffField.")
        try:
            self.sound_generator = SoundGenerator(output_dir=self.sound_output_dir, sample_rate=SAMPLE_RATE)
            self.noise_gen = NoiseGenerator(output_dir=self.sound_output_dir, sample_rate=SAMPLE_RATE)
            self.universe_sounds = UniverseSounds(output_dir=self.sound_output_dir, sample_rate=SAMPLE_RATE)
            if 'SephirothSoundIntegration' in globals() and SephirothSoundIntegration is not None:
                 self.sephiroth_sounds = SephirothSoundIntegration(output_dir=os.path.join(self.sound_output_dir, "sephiroth"))
                 if self.sephiroth_sounds.sound_generator is None:
                      logger.warning("SephirothSoundIntegration loaded, but its internal generator missing.")
            else:
                 logger.warning("SephirothSoundIntegration class not available.")
                 self.sephiroth_sounds = None
            logger.info("Guff sound objects initialized successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize Guff sound objects: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize sound objects for GuffField: {e}") from e

    def _generate_fibonacci_sequence(self, length: int) -> List[int]:
        """Generate Fibonacci sequence."""
        if length <= 0: return []
        if length == 1: return [0]
        sequence = [0, 1]
        while len(sequence) < length:
            next_val = sequence[-1] + sequence[-2]
            sequence.append(next_val)
        return sequence

    def _initialize_field(self):
        """Initialize the 3D field with creator harmonics."""
        try:
            harmonic_base = np.zeros(self.dimensions, dtype=DEFAULT_FIELD_DTYPE)
            indices = np.indices(self.dimensions, dtype=DEFAULT_FIELD_DTYPE) # Use field dtype

            # Add harmonic waves based on Fibonacci wavelengths
            # Use fewer terms for faster init, ensure fib > 0
            fib_count = min(8, len(self.fibonacci_sequence))
            for i in range(3, fib_count):
                fib = self.fibonacci_sequence[i]
                if fib <= 0: continue
                wavelength = float(max(1, fib))

                for dim_idx in range(3):
                     harmonic_base += 0.1 * np.sin(2 * PI * indices[dim_idx] / wavelength)

            # Normalize base pattern
            min_h, max_h = np.min(harmonic_base), np.max(harmonic_base)
            if max_h > min_h: harmonic_base = (harmonic_base - min_h) / (max_h - min_h)
            else: harmonic_base.fill(0.0)

            self.energy_potential = (harmonic_base * 0.5).astype(DEFAULT_FIELD_DTYPE)

            # Initialize harmonic field
            amplitudes = np.sqrt(self.energy_potential)
            center = np.array([d / 2.0 for d in self.dimensions], dtype=DEFAULT_FIELD_DTYPE)
            dist_sq = np.sum((indices - center[:, None, None, None])**2, axis=0)
            distance = np.sqrt(dist_sq)
            max_dist = np.sqrt(np.sum((np.array(self.dimensions) / 2.0)**2))
            norm_dist = distance / max(1e-6, max_dist)
            phases = norm_dist * 2 * PI * self.golden_ratio

            self.harmonic_field = amplitudes.astype(DEFAULT_COMPLEX_DTYPE) * np.exp(1j * phases).astype(DEFAULT_COMPLEX_DTYPE)
            self._normalize_field()
            logger.info("Guff field baseline initialized with 3D Fibonacci harmonics")
        except Exception as e:
             logger.critical(f"CRITICAL: Error initializing Guff field state: {e}", exc_info=True)
             raise RuntimeError(f"Failed to initialize Guff field state: {e}") from e

    def _initialize_sound_system(self):
        """Initialize sound system for the Guff field. Fails hard."""
        if not self.sound_enabled:
             logger.warning("Guff sound system initialization skipped: sound disabled.")
             return
        try:
            self.guff_sound = self.generate_guff_sound(duration=15.0)
            if self.guff_sound is None or len(self.guff_sound) == 0:
                raise RuntimeError("Failed to generate critical initial Guff sound.")

            logger.info("Guff field sound system initialized")
            success = self.apply_sound_to_field(self.guff_sound, intensity=GUFF_SOUND_INTENSITY)
            if not success:
                 raise RuntimeError("Failed to apply initial Guff sound to field.")

        except Exception as e:
            logger.critical(f"CRITICAL: Error initializing Guff sound system: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Guff sound system: {e}") from e

    def _normalize_field(self):
        """Normalize field energy (clipping) and update harmonic field."""
        try:
             self.energy_potential = np.clip(self.energy_potential, 0.0, 1.0)
             energy_nonneg = np.maximum(0.0, self.energy_potential)
             current_phases = np.angle(self.harmonic_field)
             new_amplitudes = np.sqrt(energy_nonneg)
             self.harmonic_field = new_amplitudes.astype(DEFAULT_COMPLEX_DTYPE) * np.exp(1j * current_phases).astype(DEFAULT_COMPLEX_DTYPE)
        except Exception as e:
             logger.error(f"Error during Guff field normalization: {e}")
             # Consider if this should raise an error

    def initialize_field_patterns(self) -> Dict[str, Any]:
        """Initialize the 3D Guff field with creator harmonic patterns."""
        logger.info("Initializing 3D Guff field patterns...")
        try:
            center = tuple(d // 2 for d in self.dimensions)
            indices = np.indices(self.dimensions, dtype=DEFAULT_FIELD_DTYPE)
            distance = np.sqrt(np.sum((indices - np.array(center)[:,None,None,None])**2, axis=0))
            max_dist = np.sqrt(np.sum((np.array(self.dimensions)/2.0)**2))

            pattern = np.zeros(self.dimensions, dtype=np.float64)
            valid_fibs = []
            # Use up to 10th fib, or less if sequence is shorter
            max_fib_idx = min(10, len(self.fibonacci_sequence))
            fib_norm_base = self.fibonacci_sequence[max_fib_idx-1] if max_fib_idx > 3 and self.fibonacci_sequence[max_fib_idx-1]>0 else 1

            for i in range(3, max_fib_idx):
                fib = self.fibonacci_sequence[i]
                if fib <= 0: continue
                radius = (fib / fib_norm_base) * max_dist * 0.8 # Scale radius
                if radius <= 0: continue
                valid_fibs.append(fib)
                shell_width = max(1.0, radius * 0.05)
                shell = np.exp(-(distance - radius)**2 / (2 * shell_width**2))
                pattern += shell

            # Modulation
            modulation = np.ones(self.dimensions)
            for i in range(3):
                mod_wavelength = self.dimensions[i] / self.golden_ratio
                if mod_wavelength > 0: modulation *= 0.5 + 0.5 * np.sin(2 * PI * indices[i] / mod_wavelength)
            pattern *= (0.7 + 0.3 * modulation)

            # Normalize and apply
            max_p = np.max(pattern); pattern = pattern / max(max_p, 1e-9) if max_p > 0 else np.zeros_like(pattern)
            self.energy_potential = (pattern * 0.8).astype(DEFAULT_FIELD_DTYPE)
            amplitudes = np.sqrt(self.energy_potential)
            phases = np.random.uniform(0, 2 * PI, self.dimensions)
            self.harmonic_field = amplitudes.astype(DEFAULT_COMPLEX_DTYPE) * np.exp(1j * phases).astype(DEFAULT_COMPLEX_DTYPE)

            # Apply initial sound
            if self.sound_enabled and self.guff_sound is not None:
                self.apply_sound_to_field(self.guff_sound, intensity=GUFF_SOUND_INTENSITY)

            self._normalize_field()
            return { "pattern_type": "fibonacci_spherical_3d", "golden_ratio": self.golden_ratio,
                     "fibonacci_used": valid_fibs, "max_energy": float(np.max(self.energy_potential)),
                     "mean_energy": float(np.mean(self.energy_potential)), "timestamp": datetime.now().isoformat() }
        except Exception as e:
             logger.error(f"Error initializing field patterns: {e}", exc_info=True)
             raise RuntimeError(f"Failed to initialize Guff field patterns: {e}") from e

    def receive_spark(self, spark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Receive a 3D soul spark. Raises ValueError on invalid input."""
        if not isinstance(spark_data, dict) or not spark_data.get("transfer_complete", False):
            raise ValueError("Invalid or incomplete spark data received.")
        void_pos = spark_data.get("position"); void_dims_data = spark_data.get("void_field_metrics", {}).get("dimensions")
        if void_pos is None or void_dims_data is None or len(void_pos)!=3 or len(void_dims_data)!=3:
            raise ValueError("Spark data missing valid 3D void position or dimensions.")
        # Ensure void dimensions are positive integers
        void_dims = tuple(int(max(1,d)) for d in void_dims_data) # Force positive int

        spark_id = spark_data.get("spark_id", "unknown_spark")
        logger.info(f"Receiving spark {spark_id} from 3D Void at {void_pos}")

        try:
            # Map position safely
            guff_pos = tuple(min(int(p*self.dimensions[i]/vd), self.dimensions[i]-1) for i,(p,vd) in enumerate(zip(void_pos, void_dims)))
            # Calculate radius safely
            max_energy_void = max(1e-9, spark_data.get("void_field_metrics",{}).get("max_energy",1.0))
            energy_norm = spark_data.get("energy",0) / max_energy_void
            radius = max(2, int(10*energy_norm))
            stability = max(0.0, min(1.0, spark_data.get("stability", 0.5)))

            # Create node BEFORE generating sound
            self._create_resonance_node(guff_pos, radius, stability)

            # Generate/apply sound
            sound_path = spark_data.get("reception_sound_path")
            if self.sound_enabled and not sound_path:
                reception_sound = self.generate_reception_sound(spark_data) # Updates spark_data dict
                if reception_sound is not None:
                    self.apply_sound_to_field(reception_sound, intensity=GUFF_RECEPTION_SOUND_INTENSITY)
                    sound_path = spark_data.get("reception_sound_path") # Get updated path

            reception_info = { "spark_id": spark_id, "void_position": void_pos, "guff_position": guff_pos,
                               "reception_radius": radius, "reception_energy": float(self.energy_potential[guff_pos]),
                               "reception_timestamp": datetime.now().isoformat(),
                               "reception_sound_path": sound_path, "reception_complete": True }
            logger.info(f"Spark reception complete at {guff_pos} (3D)")
            return reception_info
        except Exception as e:
             logger.error(f"Error receiving spark {spark_id}: {e}", exc_info=True)
             raise RuntimeError(f"Failed receiving spark {spark_id}: {e}") from e

    def _create_resonance_node(self, position: Tuple[int,int,int], radius: int, stability: float):
        """Create a 3D resonance node. Raises errors on failure."""
        if len(position)!=3 or not all(0 <= p < d for p,d in zip(position, self.dimensions)): raise ValueError(f"Invalid node position {position} for dims {self.dimensions}")
        if radius <= 0: raise ValueError("Radius must be positive")
        stability = max(0.0, min(1.0, stability))
        logger.debug(f"Creating 3D node at {position}, r={radius}, stability={stability:.3f}")

        try:
            slices = tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
            region_shape = tuple(s.stop - s.start for s in slices)
            if any(s == 0 for s in region_shape): logger.warning(f"Node region {slices} is empty, skipping."); return

            # Use meshgrid for efficient distance calc within region
            grid_coords = np.meshgrid(*(np.arange(s.start, s.stop) for s in slices), indexing='ij')
            dist_sq = np.sum((grid_coords - np.array(position)[:,None,None,None])**2, axis=0)
            distances = np.sqrt(dist_sq)

            effective_radius = max(1.0, radius * 0.5)
            energy_profile = np.exp(-distances**2 / (2 * effective_radius**2)) * (0.5 + 0.5 * stability)

            # Apply energy additively
            self.energy_potential[slices] += energy_profile.astype(self.energy_potential.dtype)

            # Update harmonic field
            region_harmonic = self.harmonic_field[slices]
            new_amplitudes = np.sqrt(np.maximum(0, self.energy_potential[slices]))
            current_phases = np.angle(region_harmonic)
            phase_adjustment = 2*PI * self.golden_ratio * energy_profile * 0.1
            new_phases = current_phases + phase_adjustment
            self.harmonic_field[slices] = new_amplitudes.astype(DEFAULT_COMPLEX_DTYPE) * np.exp(1j*new_phases).astype(DEFAULT_COMPLEX_DTYPE)

            self._normalize_field()

        except Exception as e:
             logger.error(f"Error creating resonance node at {position}: {e}", exc_info=True)
             raise RuntimeError(f"Failed to create resonance node at {position}") from e # Fail hard

    def strengthen_soul_formation(self, reception_data: Dict[str, Any], iterations: int = 10) -> Dict[str, Any]:
        """Strengthen the 3D soul formation. Fails hard on error."""
        if not isinstance(reception_data, dict) or not reception_data.get("reception_complete", False):
            raise ValueError("Cannot strengthen: Invalid or incomplete reception data.")
        position = reception_data.get("guff_position"); radius = reception_data.get("reception_radius")
        if position is None or radius is None or len(position)!=3 or radius<=0:
            raise ValueError("Cannot strengthen: Invalid position or radius in reception data.")
        if not isinstance(iterations, int) or iterations <= 0:
             raise ValueError("Iterations must be a positive integer.")

        logger.info(f"Strengthening soul formation at {position} (3D) for {iterations} iterations")
        try:
            template = self._create_formation_template_3d(position, radius)
            strength_prog = []; resonance_prog = []

            for i in range(iterations):
                strength_factor = 1.0 + (i / iterations) * self.strengthening_factor
                self._apply_formation_template_3d(template, position, radius, strength_factor) # Can raise errors
                f_strength = self._calculate_formation_strength_3d(position, radius)
                f_resonance = self._calculate_formation_resonance_3d(position, radius)
                strength_prog.append(f_strength); resonance_prog.append(f_resonance)

            # Sound generation and application
            sound_path = None
            if self.sound_enabled:
                 formation_sound = self.generate_soul_formation_sound(template) # Can fail hard
                 if formation_sound is not None:
                      region_slices = tuple(slice(max(0, p - radius), min(dim, p + radius + 1)) for p, dim in zip(position, self.dimensions))
                      success_apply = self.apply_sound_to_field(formation_sound, intensity=GUFF_FORMATION_SOUND_INTENSITY, region=region_slices)
                      if not success_apply: logger.warning("Failed to apply formation sound.")
                      sound_path = template.get("formation_sound_path") # Get path if set

            formation_quality = self.calculate_formation_quality_3d(position, radius)
            result = { # Populate result dictionary as before...
                       "spark_id": reception_data["spark_id"], "position": position, "radius": radius, "iterations": iterations,
                       "initial_strength": strength_prog[0] if strength_prog else 0, "final_strength": strength_prog[-1] if strength_prog else 0,
                       "initial_resonance": resonance_prog[0] if resonance_prog else 0, "final_resonance": resonance_prog[-1] if resonance_prog else 0,
                       "formation_quality": formation_quality, "strength_progression": strength_prog, "resonance_progression": resonance_prog,
                       "timestamp": datetime.now().isoformat(), "formation_sound_path": sound_path }
            logger.info(f"Soul formation strengthened: quality={formation_quality:.4f}")
            return result
        except Exception as e:
             logger.error(f"Error strengthening soul formation: {e}", exc_info=True)
             raise RuntimeError(f"Failed to strengthen soul formation: {e}") from e # Fail hard

    def _create_formation_template_3d(self, position: tuple, radius: int) -> Dict[str, Any]:
        """Create a 3D soul formation template. Fails hard on error."""
        # (Implementation from previous step, ensure robust calculations)
        if len(position)!=3 or radius<=0: raise ValueError("Invalid position/radius for template.")
        logger.debug(f"Creating 3D formation template at {position}, r={radius}")
        try:
            indices=np.indices(self.dimensions, dtype=DEFAULT_FIELD_DTYPE)
            dist=np.sqrt(np.sum((indices-np.array(position)[:,None,None,None])**2, axis=0))
            template=np.zeros(self.dimensions, dtype=DEFAULT_FIELD_DTYPE); fib_radii=[]
            fib_max = self.fibonacci_sequence[7] if len(self.fibonacci_sequence)>7 and self.fibonacci_sequence[7]>0 else 1
            if fib_max == 0: raise ValueError("Fibonacci base for scaling is zero.")

            for i in range(3,min(8,len(self.fibonacci_sequence))):
                fib=self.fibonacci_sequence[i];
                if fib<=0: continue
                fib_r=(fib/fib_max)*radius
                if fib_r<=0: continue
                fib_radii.append(fib_r); width=max(1.,fib_r*0.05)
                template+=np.exp(-(dist-fib_r)**2/(2*width**2))

            rel_x=indices[0]-position[0]; rel_y=indices[1]-position[1]; rel_z=indices[2]-position[2]
            azimuth=np.arctan2(rel_y,rel_x); elevation=np.arctan2(rel_z,np.sqrt(rel_x**2+rel_y**2+1e-9))
            phi_mod=np.sin(azimuth*self.golden_ratio+elevation*2.0); modulation=0.7+0.3*phi_mod
            template*=modulation; max_t=np.max(template); template=template/max(max_t,1e-9) if max_t>0 else np.zeros_like(template)

            return {"template":template, "position":position, "radius":radius, "fibonacci_radii":fib_radii, "timestamp":datetime.now().isoformat()}
        except Exception as e:
             logger.error(f"Error creating formation template: {e}", exc_info=True)
             raise RuntimeError("Template creation failed.") from e

    def _apply_formation_template_3d(self, template_data: Dict[str, Any], position: tuple, radius: int, strength_factor: float):
        """Apply a 3D formation template. Fails hard on error."""
        if not isinstance(template_data, dict) or "template" not in template_data: raise ValueError("Invalid template data.")
        if len(position)!=3 or radius<=0 or strength_factor <= 0: raise ValueError("Invalid position/radius/strength.")

        try:
            template_field = template_data["template"]
            slices = tuple(slice(max(0,p-radius), min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
            region_shape = tuple(s.stop - s.start for s in slices)
            if any(s==0 for s in region_shape): return # Skip empty region

            # Extract template slice corresponding to the field region
            template_slices = []
            for i in range(3):
                 start = slices[i].start; stop = slices[i].stop
                 if start < 0 or stop > template_field.shape[i]: raise IndexError("Template application region exceeds template bounds.")
                 template_slices.append(slice(start, stop))
            template_region = template_field[tuple(template_slices)]

            if template_region.shape != region_shape: # Should not happen with IndexError check, but double-check
                 raise ValueError(f"Shape mismatch between template region {template_region.shape} and field slices {region_shape}.")

            # Apply template additively
            self.energy_potential[slices] += template_region.astype(self.energy_potential.dtype) * strength_factor

            # Update harmonic field phase
            region_harmonic = self.harmonic_field[slices]
            amplitudes = np.sqrt(np.maximum(0, self.energy_potential[slices]))
            phases = np.angle(region_harmonic)
            phase_adjustment = 2*PI*self.golden_ratio*template_region*strength_factor*0.1
            new_phases = phases + phase_adjustment
            self.harmonic_field[slices] = amplitudes.astype(DEFAULT_COMPLEX_DTYPE) * np.exp(1j*new_phases).astype(DEFAULT_COMPLEX_DTYPE)

            # No normalization here - done after the loop in strengthen_soul_formation
        except Exception as e:
            logger.error(f"Error applying formation template at {position}: {e}", exc_info=True)
            raise RuntimeError("Formation template application failed.") from e


    def _calculate_formation_strength_3d(self, position: tuple, radius: int) -> float:
        """Calculate 3D formation strength. Returns 0.0 on error."""
        # (Implementation from previous step, return 0.0 on error)
        if len(position)!=3 or radius<=0: return 0.0
        try:
            slices=tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
            region=self.energy_potential[slices];
            if region.size==0: return 0.0
            mean_e=np.mean(region); std_e=np.std(region); field_mean=np.mean(self.energy_potential)
            stability = 1.0 / (1.0 + std_e / max(1e-9, mean_e))
            concentration = mean_e / max(1e-9, field_mean)
            strength = 0.6*stability + 0.4*min(concentration, 5.0)/5.0
            return max(0.0, min(1.0, strength))
        except Exception as e:
             logger.warning(f"Error calculating strength at {position}: {e}")
             return 0.0

    def _calculate_formation_resonance_3d(self, position: tuple, radius: int) -> float:
        """Calculate 3D formation resonance. Returns 0.0 on error."""
        # (Implementation from previous step, return 0.0 on error)
        if len(position)!=3 or radius<=0: return 0.0
        try:
            slices=tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
            region_h=self.harmonic_field[slices];
            if region_h.size==0: return 0.0
            phases=np.angle(region_h); coherence=np.abs(np.mean(np.exp(1j*phases)))
            ideal_phi=2*PI/self.golden_ratio; align=np.mean(np.cos(phases-ideal_phi)); align_norm=(align+1)/2.0
            resonance = 0.7*coherence + 0.3*align_norm
            return max(0.0, min(1.0, resonance))
        except Exception as e:
             logger.warning(f"Error calculating resonance at {position}: {e}")
             return 0.0

    def calculate_formation_quality_3d(self, position: tuple, radius: int) -> float:
        """Calculate 3D formation quality. Returns 0.0 on error."""
        # (Implementation from previous step, return 0.0 on error)
        if len(position)!=3 or radius<=0: return 0.0
        try:
            strength=self._calculate_formation_strength_3d(position,radius); resonance=self._calculate_formation_resonance_3d(position,radius)
            fib_align=self._calculate_fibonacci_alignment_3d(position,radius); geo_harmony=self._calculate_geometric_harmony_3d(position,radius)
            q_raw = (0.35*strength + 0.35*resonance + 0.15*fib_align + 0.15*geo_harmony)
            k=10; quality = 1.0 / (1.0 + np.exp(-k * (q_raw - self.soul_formation_threshold)))
            return quality
        except Exception as e:
             logger.warning(f"Error calculating formation quality at {position}: {e}")
             return 0.0

    def _calculate_fibonacci_alignment_3d(self, position: tuple, radius: int) -> float:
        """Calculate 3D Fibonacci alignment. Returns 0.0 on error."""
        # (Implementation from previous step, return 0.0 on error)
        if len(position)!=3 or radius<=0: return 0.0
        try:
            slices=tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
            region=self.energy_potential[slices];
            if region.size==0: return 0.0
            indices=np.indices(region.shape); abs_indices=[indices[i]+slices[i].start for i in range(3)]
            distances=np.sqrt(np.sum((abs_indices-np.array(position)[:,None,None,None])**2, axis=0))
            scores=[]; region_mean=np.mean(region)
            if region_mean < 1e-9: return 0.0
            fib_max=self.fibonacci_sequence[7] if len(self.fibonacci_sequence)>7 and self.fibonacci_sequence[7]>0 else 1
            if fib_max==0: return 0.0
            for i in range(3,min(8,len(self.fibonacci_sequence))):
                fib=self.fibonacci_sequence[i];
                if fib<=0: continue
                fib_r=(fib/fib_max)*radius
                if fib_r<=0: continue
                thickness=max(1.0, radius*0.05); mask=(np.abs(distances-fib_r)<thickness)
                if np.any(mask):
                    shell_mean=np.mean(region[mask]); score=1.0-abs(shell_mean-region_mean)/region_mean
                    scores.append(max(0.0, score))
            alignment=np.mean(scores) if scores else 0.0; return max(0.0, min(1.0, alignment))
        except Exception as e:
             logger.warning(f"Error calculating Fibonacci alignment at {position}: {e}")
             return 0.0

    def _calculate_geometric_harmony_3d(self, position: tuple, radius: int) -> float:
         """Calculate 3D geometric harmony (Simplified). Returns 0.0 on error."""
         # (Implementation from previous step, return 0.0 on error)
         if len(position)!=3 or radius<=0: return 0.0
         try:
             slices=tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
             region=self.energy_potential[slices];
             if region.size<8: return 0.0
             scores=[]
             centers=tuple(s//2 for s in region.shape)
             for axis in range(3):
                 split=centers[axis]
                 if split<=0 or split>=region.shape[axis]-1: continue
                 idx1=[slice(None)]*3; idx1[axis]=slice(0,split)
                 idx2=[slice(None)]*3; idx2[axis]=slice(None,region.shape[axis]-split-1,-1)
                 h1=region[tuple(idx1)]; h2=region[tuple(idx2)]
                 min_s=tuple(min(s1,s2) for s1,s2 in zip(h1.shape,h2.shape))
                 if any(s==0 for s in min_s): continue
                 trim1=tuple(slice(0,s) for s in min_s); trim2=tuple(slice(0,s) for s in min_s)
                 h1_t=h1[trim1]; h2_t=h2[trim2]
                 mean_h1=np.mean(h1_t)
                 if mean_h1 > 1e-9: scores.append(max(0.0, 1.0-(np.mean(np.abs(h1_t-h2_t))/mean_h1)))
             harmony=np.mean(scores) if scores else 0.0; return max(0.0, min(1.0, harmony))
         except Exception as e:
              logger.warning(f"Error calculating geometric harmony at {position}: {e}")
              return 0.0

    # --- Sound Generation Methods (Keep logic, ensure safe calls & fail hard if needed) ---
    # --- Added check for sound_enabled and helper _get_sound_object ---
    def _get_sound_object(self, obj_name: str) -> Any:
        """Helper to get a required sound object, raising error if missing/disabled."""
        if not self.sound_enabled: raise RuntimeError(f"Sound generation failed: Sound system is disabled.")
        sound_obj = getattr(self, obj_name, None)
        if sound_obj is None: raise RuntimeError(f"Sound generation failed: Required object '{obj_name}' is not available.")
        return sound_obj

    def generate_guff_sound(self, duration: float = 30.0) -> Optional[np.ndarray]:
         # Uses _get_sound_object internally now
         snd_gen = self._get_sound_object('sound_generator'); uni_sounds = self._get_sound_object('universe_sounds')
         seph_sounds = getattr(self, 'sephiroth_sounds', None) # Optional
         sample_rate = snd_gen.sample_rate
         if duration <= 0: raise ValueError("Duration must be positive.")
         try:
             num_samples=int(duration*sample_rate); time=np.linspace(0,duration,num_samples)
             base_tone=0.5*np.sin(2*PI*self.base_frequency*time); harmonic_tones=np.zeros_like(base_tone)
             fib_seq = self.fibonacci_sequence
             for i in range(1, min(7, len(fib_seq)-1)):
                 if fib_seq[i] <= 0: continue
                 fib_ratio = fib_seq[i+1]/fib_seq[i]; freq=self.base_frequency*fib_ratio
                 amp=0.3/max(1.,i*0.5); harmonic_tones+=amp*np.sin(2*PI*freq*time)
             phi_mod=0.2*np.sin(2*PI*self.base_frequency/self.golden_ratio*time); creator_tones=base_tone+harmonic_tones+phi_mod
             cosmic_bg = uni_sounds.generate_cosmic_background(duration,0.3,sample_rate=sample_rate,frequency_band='high')
             if cosmic_bg is None: cosmic_bg = np.zeros_like(creator_tones)
             min_l=min(len(creator_tones),len(cosmic_bg)); creator_tones=creator_tones[:min_l]; cosmic_bg=cosmic_bg[:min_l]
             guff_sound=0.75*creator_tones+0.25*cosmic_bg; max_amp=np.max(np.abs(guff_sound)); guff_sound=guff_sound/max(max_amp,1e-9)*MAX_AMPLITUDE

             # Add Kether sound if available
             if seph_sounds and hasattr(seph_sounds, 'get_sephiroth_sound'):
                 kether_aspects = seph_sounds.get_sephiroth_sound('kether', duration=5.0)
                 if kether_aspects is not None and len(kether_aspects) > 0:
                      kether_len=min(len(kether_aspects),len(guff_sound)); fade_in=np.linspace(0,1,kether_len)
                      guff_sound[-kether_len:] += 0.3*kether_aspects[:kether_len]*fade_in
                      max_amp=np.max(np.abs(guff_sound)); guff_sound/=max(max_amp,1e-9)*MAX_AMPLITUDE

             timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"guff_field_{timestamp}.wav"
             if not snd_gen.save_sound(guff_sound, filename=fn): raise IOError("Failed to save Guff sound.")
             return guff_sound
         except Exception as e: logger.error(f"Error generating Guff sound: {e}"); raise RuntimeError("Guff sound gen failed") from e

    # --- (Implement other generate_*_sound methods with similar fail-hard logic) ---
    def generate_reception_sound(self, spark_data: Dict[str, Any], duration: float = 7.0) -> Optional[np.ndarray]:
        snd_gen = self._get_sound_object('sound_generator'); uni_sounds = self._get_sound_object('universe_sounds')
        if not isinstance(spark_data, dict): raise ValueError("Invalid spark_data for reception sound.")
        if duration <= 0: raise ValueError("Duration must be positive.")
        sample_rate = self.sample_rate; num_samples = int(duration * sample_rate)
        try:
            time = np.linspace(0, duration, num_samples); energy_norm = spark_data.get("energy",0)/max(1e-6, spark_data.get("void_field_metrics",{}).get("max_energy",1.0))
            f_start=self.base_frequency*0.8; f_end=self.base_frequency*(1.0+0.5*energy_norm)
            if f_start <=0 or f_end <=0: raise ValueError("Invalid frequencies for reception sound.")
            freq_curve = f_start+(f_end-f_start)*(1-np.exp(-4*time/max(1e-6,duration))); phase=np.cumsum(freq_curve/sample_rate)
            sound = 0.7*np.sin(2*PI*phase)
            stability = spark_data.get("stability", 0.5)
            fib_seq = self.fibonacci_sequence
            for i in range(1, min(5, len(fib_seq)-1)):
                 if fib_seq[i] <= 0: continue
                 fib_ratio=fib_seq[i+1]/fib_seq[i]; h_freq=self.base_frequency*fib_ratio; amp=0.2*stability/i
                 phase_offset=np.random.uniform(0,2*PI); sound+=amp*np.sin(2*PI*h_freq*time+phase_offset)
            shimmer = 0.3*np.sin(2*PI*20*time)*np.exp(-time/(duration*0.3)); sound+=shimmer
            max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE
            noise = uni_sounds.generate_dimensional_transition(duration,sample_rate,'void_to_guff')
            if noise is not None: min_l=min(len(sound),len(noise)); sound=0.7*sound[:min_l]+0.3*noise[:min_l]
            max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE
            timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"spark_reception_{timestamp}.wav"
            filepath = snd_gen.save_sound(sound, filename=fn)
            if not filepath: raise IOError("Failed to save reception sound.")
            spark_data["reception_sound_path"] = os.path.basename(filepath) # Relative path
            return sound
        except Exception as e: logger.error(f"Reception sound gen error: {e}"); raise RuntimeError("Reception sound gen failed") from e

    def generate_soul_formation_sound(self, template_data: Dict[str, Any], duration: float = 15.0) -> Optional[np.ndarray]:
        snd_gen=self._get_sound_object('sound_generator'); seph_sounds=getattr(self,'sephiroth_sounds',None)
        if not isinstance(template_data, dict): raise ValueError("Invalid template data for formation sound.")
        if duration <= 0: raise ValueError("Duration must be positive.")
        sample_rate = self.sample_rate; num_samples = int(duration * sample_rate)
        try:
            time = np.linspace(0, duration, num_samples); sound = np.zeros_like(time)
            fib_seq = self.fibonacci_sequence
            for i in range(3, min(8, len(fib_seq))):
                 if fib_seq[3] <= 0 or fib_seq[i] <= 0: continue # Need base fib > 0
                 fib_ratio=fib_seq[i]/fib_seq[3]; freq=self.base_frequency*fib_ratio; amp=0.6/max(1,i-2)
                 fade_in=np.sqrt(np.clip(time*i/max(1e-6,duration),0,1)); sound+=amp*fade_in*np.sin(2*PI*freq*time)
            phi_mod=0.3*np.sin(2*PI*(self.base_frequency/self.golden_ratio)*time); sound+=phi_mod
            # Add bloom
            bloom_p=int(2*duration/3*sample_rate); bloom_l=int(duration/6*sample_rate)
            if bloom_p+bloom_l <= num_samples:
                 bloom_t=np.linspace(0,1,bloom_l); bloom_env=np.sin(PI*bloom_t); bloom_chord=np.zeros(bloom_l)
                 fib5 = fib_seq[5] if len(fib_seq)>5 and fib_seq[5]>0 else 1
                 for i in range(3,min(8,len(fib_seq))):
                      if fib_seq[i] <=0: continue
                      freq = self.base_frequency*fib_seq[i]/max(1e-6, fib5)
                      bloom_chord+=0.2*np.sin(2*PI*freq*np.linspace(0,duration/6,bloom_l)) # Use correct time for bloom
                 sound[bloom_p:bloom_p+bloom_l]+=bloom_chord*bloom_env

            max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE
            # Add Kether resonance if possible
            if seph_sounds and hasattr(seph_sounds, 'get_harmonic_resonance'):
                kether_tones = seph_sounds.get_harmonic_resonance('kether', duration=5.0)
                if kether_tones is not None and len(kether_tones)>0:
                     kether_l=min(len(kether_tones),int(sample_rate*5)); fade_in=np.linspace(0,1,kether_l)
                     end_p=max(0,len(sound)-kether_l)
                     sound[end_p:end_p+kether_l]+=0.4*kether_tones[:min(kether_l,len(sound)-end_p)]*fade_in
                     max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE

            timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"soul_formation_{timestamp}.wav"
            filepath = snd_gen.save_sound(sound, filename=fn)
            if not filepath: raise IOError("Failed to save soul formation sound.")
            template_data["formation_sound_path"] = os.path.basename(filepath)
            return sound
        except Exception as e: logger.error(f"Soul formation sound gen error: {e}"); raise RuntimeError("Soul formation sound gen failed") from e

    def generate_birth_sound(self, formation_data: Dict[str, Any], duration: float = 12.0) -> Optional[np.ndarray]:
        snd_gen=self._get_sound_object('sound_generator'); seph_sounds=getattr(self,'sephiroth_sounds',None)
        if not isinstance(formation_data, dict): raise ValueError("Invalid formation data for birth sound.")
        if duration <= 0: raise ValueError("Duration must be positive.")
        sample_rate = self.sample_rate; num_samples = int(duration * sample_rate)
        try:
            time=np.linspace(0,duration,num_samples); quality=formation_data.get("formation_quality",0.75)
            f0=self.base_frequency*(0.9+0.2*quality); sound=np.zeros_like(time)
            if f0 <=0 : raise ValueError("Invalid base frequency f0 for birth sound.")
            envelope=np.exp(-0.5*((time-duration/2)/max(1e-6,duration/4))**2); sound+=0.7*envelope*np.sin(2*PI*f0*time)
            fib_seq = self.fibonacci_sequence
            for i in range(2, min(7, len(fib_seq)-1)):
                 if fib_seq[i-1]<=0 or fib_seq[i]<=0: continue
                 fib_r=fib_seq[i]/fib_seq[i-1]; h_freq=f0*fib_r; h_dur=duration*(0.3+0.1*i)
                 h_env=np.exp(-0.5*((time-h_dur)/max(1e-6,duration/5))**2); sound+=(0.5/i)*h_env*np.sin(2*PI*h_freq*time)
            phi_freq=f0/self.golden_ratio; phi_sweep=phi_freq*(1+0.1*np.sin(2*PI*time/max(1e-6,duration)))
            phi_env=0.3*(1-np.cos(2*PI*time/max(1e-6,duration))); sound+=phi_env*np.sin(2*PI*phi_sweep*time)
            # Add birth moment
            birth_m=int(2*duration/3*sample_rate); birth_l=int(duration/5*sample_rate)
            if birth_m+birth_l <= num_samples:
                 cresc_env=(np.linspace(0,1,birth_l)**2); birth_chord=np.zeros(birth_l)
                 birth_t_local = np.linspace(0, duration/5, birth_l)
                 for i in range(1,8):
                      if i<=0: continue
                      birth_chord+=(0.7/i)*np.sin(2*PI*f0*i*birth_t_local)
                 sound[birth_m:birth_m+birth_l]+=birth_chord*cresc_env

            max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE
            # Add Kether resonance
            if seph_sounds and hasattr(seph_sounds, 'get_harmonic_resonance'):
                creator_tones = seph_sounds.get_harmonic_resonance('kether', duration=5.0)
                if creator_tones is not None and len(creator_tones)>0:
                     creator_l=min(len(creator_tones),int(sample_rate*5))
                     if creator_l > 0 and len(sound)>=creator_l:
                          fade_out=np.linspace(1,0,creator_l); fade_in=np.linspace(0,1,creator_l)
                          end_p=len(sound)-creator_l
                          sound[end_p:]*=fade_out; sound[end_p:]+=0.5*creator_tones[:creator_l]*fade_in
                          max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE

            timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"soul_birth_{timestamp}.wav"
            filepath = snd_gen.save_sound(sound, filename=fn)
            if not filepath: raise IOError("Failed to save birth sound.")
            formation_data["birth_sound_path"] = os.path.basename(filepath)
            return sound
        except Exception as e: logger.error(f"Birth sound gen error: {e}"); raise RuntimeError("Birth sound gen failed") from e

    def generate_sephiroth_transfer_sound(self, soul_data: Dict[str, Any], duration: float = 10.0) -> Optional[np.ndarray]:
        snd_gen=self._get_sound_object('sound_generator'); uni_sounds=self._get_sound_object('universe_sounds'); seph_sounds=getattr(self,'sephiroth_sounds',None)
        if not isinstance(soul_data, dict): raise ValueError("Invalid soul data for transfer sound.")
        if duration <= 0: raise ValueError("Duration must be positive.")
        sample_rate = self.sample_rate; num_samples = int(duration * sample_rate)
        try:
             time=np.linspace(0,duration,num_samples); quality=soul_data.get("formation_quality",0.8)
             f0=self.base_frequency*(0.95+0.1*quality); sound=np.zeros_like(time)
             if f0<=0: raise ValueError("Invalid base frequency f0 for transfer sound.")
             freq_rise=f0*(1+time/max(1e-6,duration)); envelope=np.sqrt(np.linspace(0,1,len(time)))
             sound+=0.6*envelope*np.sin(2*PI*freq_rise*time)
             kether_f=f0*1.5; kether_env=0.4*(np.linspace(0,1,len(time))**2); sound+=kether_env*np.sin(2*PI*kether_f*time)
             fib_seq = self.fibonacci_sequence
             for i in range(3, min(8, len(fib_seq)-1)):
                  if fib_seq[i-1]<=0 or fib_seq[i]<=0: continue
                  fib_r=fib_seq[i]/fib_seq[i-1]; h_freq=f0*fib_r; start_t=duration*(i-3)/8.0
                  h_env=np.zeros_like(time); idx=(time>=start_t); h_env[idx]=np.linspace(0,1,sum(idx))**0.5
                  sound+=0.3*h_env*np.sin(2*PI*h_freq*time)
             whoosh_f=np.linspace(20,1,len(time)); whoosh_env=0.3*np.exp(-4*(time-duration/2)**2/max(1e-6,duration**2))
             sound+=whoosh_env*np.sin(2*PI*whoosh_f*time)
             max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE
             noise = uni_sounds.generate_dimensional_transition(duration,sample_rate,'guff_to_sephiroth')
             if noise is not None: min_l=min(len(sound),len(noise)); sound=0.6*sound[:min_l]+0.4*noise[:min_l]
             max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE
             # Add Sephiroth previews
             if seph_sounds and hasattr(seph_sounds, 'get_preview_tone'):
                  first_sephiroth = ["binah", "chokmah"] # Example path start
                  for i, sephirah in enumerate(first_sephiroth):
                       preview = seph_sounds.get_preview_tone(sephirah, duration=3.0)
                       if preview is not None and len(preview)>0:
                            prev_l=min(len(preview),int(sample_rate*3))
                            start_p=max(0,len(sound)-prev_l-int(i*sample_rate*0.5)) # Slightly overlap
                            end_p=min(len(sound),start_p+prev_l)
                            if start_p < end_p:
                                 fade_in=np.linspace(0,1,end_p-start_p)
                                 sound[start_p:end_p]+=0.3*fade_in*preview[:end_p-start_p]
             max_amp=np.max(np.abs(sound)); sound=sound/max(max_amp,1e-9)*MAX_AMPLITUDE

             timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"sephiroth_transfer_{timestamp}.wav"
             filepath = snd_gen.save_sound(sound, filename=fn)
             if not filepath: raise IOError("Failed to save Sephiroth transfer sound.")
             soul_data["transfer_sound_path"] = os.path.basename(filepath)
             return sound
         except Exception as e: logger.error(f"Sephiroth transfer sound gen error: {e}"); raise RuntimeError("Transfer sound gen failed") from e

    # --- finalize_soul_formation (Ensure 3D profile/metric calls) ---
    def finalize_soul_formation(self, formation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize 3D soul formation. Fails hard on error or low quality."""
        if not isinstance(formation_data, dict): raise ValueError("Invalid formation_data")
        quality = formation_data.get("formation_quality", 0)
        if quality < self.soul_formation_threshold:
            raise ValueError(f"Cannot finalize: Quality {quality:.3f} < Threshold {self.soul_formation_threshold}")
        position = formation_data.get("position"); radius = formation_data.get("radius")
        if position is None or radius is None or len(position)!=3 or radius<=0:
            raise ValueError("Cannot finalize: Invalid position or radius.")
        spark_id = formation_data.get("spark_id", "unknown_spark")
        logger.info(f"Finalizing soul formation {spark_id} at {position} (3D)")

        try:
            profile = self._calculate_formation_profile_3d(position, radius) # Use 3D
            birth_sound_path = formation_data.get("birth_sound_path")
            if self.sound_enabled and not birth_sound_path:
                 birth_sound = self.generate_birth_sound(formation_data) # Updates formation_data
                 if birth_sound is None: raise RuntimeError("Failed to generate birth sound for finalization.")
                 birth_sound_path = formation_data.get("birth_sound_path")

            # Snapshot (limit size)
            max_snapshot_elements = 4096
            slices = tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
            energy_snap = self.energy_potential[slices].copy()
            harmonic_snap = self.harmonic_field[slices].copy()
            energy_list = energy_snap.tolist() if energy_snap.size<=max_snapshot_elements else None
            h_real_list = np.real(harmonic_snap).tolist() if harmonic_snap.size<=max_snapshot_elements else None
            h_imag_list = np.imag(harmonic_snap).tolist() if harmonic_snap.size<=max_snapshot_elements else None

            soul_data = {
                "spark_id": spark_id, "position": position, "radius": radius, "formation_quality": quality,
                "final_strength": formation_data["final_strength"], "final_resonance": formation_data["final_resonance"],
                "profile": profile, # 3D profile
                "field_snapshot": {"energy": energy_list, "harmonic_real": h_real_list, "harmonic_imag": h_imag_list},
                "creator_resonance": self._calculate_creator_resonance_3d(position, radius),
                "golden_ratio_alignment": self._calculate_golden_ratio_alignment_3d(position, radius),
                "fibonacci_structure": self._calculate_fibonacci_structure_3d(position, radius),
                "dimensional_properties": self._calculate_dimensional_properties_3d(position, radius),
                "formation_timestamp": datetime.now().isoformat(),
                "birth_sound_path": birth_sound_path, # Relative path
                "ready_for_sephiroth": True
            }
            logger.info(f"Soul formation finalized: quality={quality:.4f}")
            return soul_data
        except Exception as e:
             logger.error(f"Error finalizing soul formation {spark_id}: {e}", exc_info=True)
             raise RuntimeError(f"Finalization failed for spark {spark_id}: {e}") from e


    # --- Keep 3D Profile/Metric Calculations (_calculate_*_3d) ---
    # --- (Implementations from previous step, ensure robustness) ---
    def _calculate_formation_profile_3d(self, position: tuple, radius: int) -> Dict[str, Any]:
        # (Implementation from previous step)
        slices=tuple(slice(max(0,p-r),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
        region_e=self.energy_potential[slices]; region_h=self.harmonic_field[slices]
        if region_e.size==0: raise ValueError("Cannot profile empty region")
        energy={"total":float(np.sum(region_e)),"mean":float(np.mean(region_e)),"max":float(np.max(region_e)),"std":float(np.std(region_e))}
        phases=np.angle(region_h); amps=np.abs(region_h)
        harmonic={"phase_coherence":float(np.abs(np.mean(np.exp(1j*phases)))),"mean_amplitude":float(np.mean(amps))}
        sym_scores=[]
        for axis in range(3):
            split=region_e.shape[axis]//2
            if split>0 and split<region_e.shape[axis]-1:
                idx1=[slice(None)]*3; idx1[axis]=slice(0,split)
                idx2=[slice(None)]*3; idx2[axis]=slice(None,region_e.shape[axis]-split-1,-1)
                h1=region_e[tuple(idx1)]; h2=region_e[tuple(idx2)]
                min_s=tuple(min(s1,s2) for s1,s2 in zip(h1.shape,h2.shape))
                if any(s==0 for s in min_s): continue
                h1_t=h1[tuple(slice(0,s) for s in min_s)]; h2_t=h2[tuple(slice(0,s) for s in min_s)]
                mean_h1=np.mean(h1_t);
                if mean_h1>1e-9: sym_scores.append(max(0.0, 1.0-(np.mean(np.abs(h1_t-h2_t))/mean_h1)))
        symmetry={"overall_symmetry":float(np.mean(sym_scores)) if sym_scores else 0.0}
        return {"energy":energy, "harmonic":harmonic, "symmetry":symmetry}

    def _calculate_creator_resonance_3d(self, position: tuple, radius: int) -> Dict[str, float]:
        # (Implementation from previous step)
        slices=tuple(slice(max(0,p-r),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
        region_h=self.harmonic_field[slices]; region_e=self.energy_potential[slices]
        if region_h.size==0: return {"overall_resonance": 0.0}
        phases=np.angle(region_h); amps=np.abs(region_h)
        phase_align=(np.mean(np.cos(phases-(2*PI/self.golden_ratio)))+1)/2.0
        # Simplified amplitude resonance (correlation with ideal decay)
        amp_res=0.5 # Default
        if np.std(amps) > 1e-9:
            indices=np.indices(region_h.shape); abs_indices=[indices[i]+slices[i].start for i in range(3)]
            distances=np.sqrt(np.sum((abs_indices-np.array(position)[:,None,None,None])**2,axis=0))
            ideal_decay=np.exp(-distances/max(1e-6,radius))
            if np.std(ideal_decay) > 1e-9:
                 norm_amps=(amps-np.mean(amps))/np.std(amps); norm_ideal=(ideal_decay-np.mean(ideal_decay))/np.std(ideal_decay)
                 amp_res=(np.mean(norm_amps*norm_ideal)+1)/2.0
        overall=(0.6*phase_align+0.4*amp_res)
        return {"phase_alignment":float(phase_align),"amplitude_resonance":float(amp_res),"overall_resonance":float(overall)}

    def _calculate_golden_ratio_alignment_3d(self, position: tuple, radius: int) -> Dict[str, Any]:
        # Placeholder - requires complex geometric analysis
        logger.warning("3D Golden Ratio alignment calculation is a placeholder.")
        return {"overall_alignment": 0.5 + np.random.rand()*0.1} # Return plausible value

    def _calculate_fibonacci_structure_3d(self, position: tuple, radius: int) -> Dict[str, Any]:
        # (Implementation using _calculate_fibonacci_alignment_3d)
        alignment = self._calculate_fibonacci_alignment_3d(position, radius)
        # Placeholder shell count
        return {"fibonacci_progression": alignment, "shell_count": 5}

    def _calculate_dimensional_properties_3d(self, position: tuple, radius: int) -> Dict[str, Any]:
        # (Implementation from previous step)
        slices=tuple(slice(max(0,p-r),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
        region=self.energy_potential[slices];
        if region.size<8: return {"dimensional_balance":0.0,"energy_concentration":0.0, "dimensional_entropy":[]}
        dim_entropy=[]; max_entropies=[]
        for axis in range(3):
             sum_axes=tuple(i for i in range(3) if i!=axis)
             dim_dist=np.sum(region,axis=sum_axes)
             if np.sum(dim_dist)>1e-10:
                  norm_dist=dim_dist/np.sum(dim_dist)
                  entropy=-np.sum(norm_dist*np.log2(norm_dist+1e-12))
                  max_e=np.log2(len(dim_dist)) if len(dim_dist)>0 else 0
                  dim_entropy.append(entropy/max_e if max_e>0 else 0); max_entropies.append(max_e)
             else: dim_entropy.append(0.0); max_entropies.append(0.0)
        mean_entropy = np.mean(dim_entropy) if dim_entropy else 0.0
        std_entropy = np.std(dim_entropy) if dim_entropy else 0.0
        balance = 1.0 - (std_entropy / max(1e-9, mean_entropy)) if mean_entropy > 0 else 0.0
        concentration = np.max(region) / max(1e-9, np.mean(region))
        return {"dimensional_entropy":dim_entropy,"dimensional_balance":float(max(0,balance)),"energy_concentration":float(concentration)}


    # --- Keep prepare_for_sephiroth_transfer & _calculate_sephiroth_connections ---
    # --- (Logic is independent of field dimensionality) ---
    def prepare_for_sephiroth_transfer(self, soul_data: Dict[str, Any]) -> Dict[str, Any]:
         """Prepare 3D formed soul for transfer. Raises ValueError on error."""
         if not isinstance(soul_data, dict) or not soul_data.get("ready_for_sephiroth", False):
             raise ValueError("Soul not ready for Sephiroth transfer")
         spark_id = soul_data.get("spark_id", "unknown_spark")
         logger.info(f"Preparing soul {spark_id} for Sephiroth transfer from 3D Guff.")
         try:
             # Sound Generation
             transfer_sound_path = soul_data.get("transfer_sound_path")
             if self.sound_enabled and not transfer_sound_path:
                  transfer_sound = self.generate_sephiroth_transfer_sound(soul_data) # Modifies soul_data
                  if transfer_sound is None: raise RuntimeError("Failed to generate transfer sound.")
                  transfer_sound_path = soul_data.get("transfer_sound_path")

             sephiroth_connections = self._calculate_sephiroth_connections(soul_data)
             guff_metrics = self.get_field_metrics() # Get 3D metrics

             transfer_data = { "spark_id": spark_id, "formation_quality": soul_data["formation_quality"],
                               "creator_resonance": soul_data.get("creator_resonance", {}), "sephiroth_connections": sephiroth_connections,
                               "transfer_sound_path": transfer_sound_path, "guff_field_metrics": guff_metrics,
                               "transfer_timestamp": datetime.now().isoformat(), "transfer_prepared": True }
             logger.info(f"Soul {spark_id} prepared for Sephiroth transfer.")
             return transfer_data
         except Exception as e:
              logger.error(f"Error preparing soul {spark_id} for transfer: {e}", exc_info=True)
              raise RuntimeError(f"Transfer preparation failed: {e}") from e

    def _calculate_sephiroth_connections(self, soul_data: Dict[str, Any]) -> Dict[str, float]:
         # (Logic from previous step)
         if not ASPECT_DICT_AVAILABLE: return {} # Cannot function without dictionary
         quality = soul_data.get("formation_quality", 0.5)
         connections = {name: 0.0 for name in aspect_dictionary.sephiroth_names if name != 'daath'}
         connections["kether"] = max(0.0, min(1.0, 0.95 * quality))
         for sephirah in connections:
             if sephirah != "kether": connections[sephirah] = max(0.0, min(1.0, 0.1 * quality))
         # Example boost
         # first_sephiroth = ["binah", "chokmah"]
         # for sephirah in first_sephiroth:
         #     if sephirah in connections: connections[sephirah] = min(1.0, connections[sephirah] + 0.1 * quality)
         return connections


    # --- Keep get_field_metrics (3D version) ---
    def get_field_metrics(self) -> Dict[str, Any]:
        """Get metrics about the current state of the 3D Guff field."""
        # (Implementation from previous step)
        try:
             energy=self.energy_potential; harmonic=self.harmonic_field
             total_e=np.sum(energy); mean_e=np.mean(energy); max_e=np.max(energy); std_e=np.std(energy)
             phases=np.angle(harmonic); coherence=np.abs(np.mean(np.exp(1j*phases))) if harmonic.size>0 else 0.0
             phi_phase=2*PI/self.golden_ratio; phi_align=(np.mean(np.cos(phases-phi_phase))+1)/2.0 if harmonic.size>0 else 0.0
             return {"total_energy":float(total_e),"mean_energy":float(mean_e),"max_energy":float(max_e),"energy_std":float(std_e),
                     "phase_coherence":float(coherence),"phi_alignment":float(phi_align),"resonance_quality":float(self.resonance_quality),
                     "dimensions":self.dimensions,"timestamp":datetime.now().isoformat()}
        except Exception as e: logger.error(f"Error getting Guff metrics: {e}"); return {}


    # --- Keep visualization methods (3D versions) ---
    def visualize_field_slice(self, axis: str = 'z', slice_index: Optional[int] = None, show: bool = True, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualize a 2D slice of the 3D Guff field."""
        # (Implementation from previous step)
        if not VISUALIZATION_ENABLED: logger.warning("Visualization disabled."); return None
        try:
             dims=self.dimensions; axis_map={'x':0,'y':1,'z':2}; axis_idx=axis_map.get(axis)
             if axis_idx is None: raise ValueError("Axis must be 'x','y','z'")
             if slice_index is None: slice_index=dims[axis_idx]//2
             slice_index=max(0,min(slice_index,dims[axis_idx]-1))
             if axis=='z': field_2d=self.energy_potential[:,:,slice_index].T; xl,yl='X','Y'; ext=[0,dims[0],0,dims[1]]
             elif axis=='y': field_2d=self.energy_potential[:,slice_index,:].T; xl,yl='X','Z'; ext=[0,dims[0],0,dims[2]]
             else: field_2d=self.energy_potential[slice_index,:,:].T; xl,yl='Y','Z'; ext=[0,dims[1],0,dims[2]]
             fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(field_2d,cmap='viridis',origin='lower',extent=ext)
             fig.colorbar(im,ax=ax,label='Energy Potential'); ax.set_title(f'Guff Field 3D Slice ({axis.upper()}={slice_index})')
             ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect('equal')
             if save_path: plt.savefig(save_path,dpi=300,bbox_inches='tight'); logger.info(f"Saved Guff slice: {save_path}")
             if show: plt.show()
             else: plt.close(fig)
             return fig
        except Exception as e: logger.error(f"Error visualizing Guff slice: {e}"); return None

    def visualize_formation(self, soul_data: Dict[str, Any], show: bool = True, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualize a 3D soul formation."""
        # (Implementation from previous step)
        if not VISUALIZATION_ENABLED: logger.warning("Visualization disabled."); return None
        if not isinstance(soul_data,dict): logger.error("Invalid soul_data."); return None
        try:
            pos=soul_data.get("position"); radius=soul_data.get("radius");
            if pos is None or radius is None or len(pos)!=3 or radius<=0: logger.error("Invalid pos/radius."); return None
            fig=plt.figure(figsize=(12,10)); ax=fig.add_subplot(111,projection='3d')
            slices=tuple(slice(max(0,int(p-radius*2)), min(d,int(p+radius*2+1))) for p,d in zip(pos,self.dimensions))
            region_e=self.energy_potential[slices];
            if region_e.size==0: logger.warning("Viz region empty."); plt.close(fig); return None
            x_r,y_r,z_r=np.indices(region_e.shape); x_a=x_r+slices[0].start; y_a=y_r+slices[1].start; z_a=z_r+slices[2].start
            energy_v=region_e.flatten(); threshold=0.1*np.max(region_e); mask=energy_v>threshold
            if not np.any(mask): logger.warning("No points above viz threshold."); plt.close(fig); return None
            cmap=plt.cm.plasma; norm=plt.Normalize(vmin=threshold,vmax=np.max(region_e))
            colors=cmap(norm(energy_v[mask]))
            ax.scatter(x_a.flatten()[mask],y_a.flatten()[mask],z_a.flatten()[mask], c=colors,s=energy_v[mask]*50+5,alpha=0.6)
            ax.scatter(pos[0],pos[1],pos[2],color='cyan',s=150,marker='*',label='Center')
            q=soul_data.get("formation_quality",0); s=soul_data.get("final_strength",0)
            ax.set_title(f"Formation (Q:{q:.3f},S:{s:.3f}) @ ({pos[0]},{pos[1]},{pos[2]})"); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend()
            max_r=max(x_a.max()-x_a.min(),y_a.max()-y_a.min(),z_a.max()-z_a.min()); mid_x=(x_a.max()+x_a.min())/2; mid_y=(y_a.max()+y_a.min())/2; mid_z=(z_a.max()+z_a.min())/2
            ax.set_xlim(mid_x-max_r/2,mid_x+max_r/2); ax.set_ylim(mid_y-max_r/2,mid_y+max_r/2); ax.set_zlim(mid_z-max_r/2,mid_z+max_r/2)
            if save_path: plt.savefig(save_path,dpi=300,bbox_inches='tight'); logger.info(f"Saved formation viz: {save_path}")
            if show: plt.show()
            else: plt.close(fig)
            return fig
        except Exception as e: logger.error(f"Error visualizing formation: {e}"); plt.close(fig); return None


    # --- evolve_field (Keep 3D version) ---
    def evolve_field(self, time_step: float = 0.1, iterations: int = 1):
        """Evolve the 3D field over time. Fails hard on error."""
        if not SCIPY_AVAILABLE: raise RuntimeError("Cannot evolve field: SciPy not available for filtering.")
        if time_step <= 0 or iterations <= 0: raise ValueError("time_step and iterations must be positive.")
        logger.debug(f"Evolving Guff field for {iterations} steps, dt={time_step}")
        try:
            for _ in range(iterations):
                phase_change=2*PI*self.energy_potential*time_step
                self.harmonic_field*=np.exp(1j*phase_change).astype(DEFAULT_COMPLEX_DTYPE)
                amplitude=np.abs(self.harmonic_field); phases=np.angle(self.harmonic_field)
                complex_phase_vector=np.exp(1j*phases); n_size=3
                local_mean_vector=scipy.ndimage.uniform_filter(complex_phase_vector, size=n_size, mode='reflect')
                local_coherence=np.abs(local_mean_vector)
                blend=0.1; self.energy_potential=((1.0-blend)*self.energy_potential + blend*amplitude*local_coherence).astype(DEFAULT_FIELD_DTYPE)
                chaos=0.05*np.random.random(self.dimensions); order=0.95; self.energy_potential=order*self.energy_potential+chaos
                self._apply_fibonacci_influence_3d()
                self._normalize_field()
            return True # Indicate success
        except Exception as e:
            logger.error(f"Error evolving Guff field: {e}", exc_info=True)
            raise RuntimeError("Field evolution failed.") from e

    # --- Keep _apply_fibonacci_influence_3d ---
    def _apply_fibonacci_influence_3d(self):
        # (Implementation from previous step)
        try:
            indices=np.indices(self.dimensions,dtype=DEFAULT_FIELD_DTYPE); center=np.array([d//2 for d in self.dimensions]);
            dist=np.sqrt(np.sum((indices-center[:,None,None,None])**2,axis=0))
            max_dist=np.sqrt(np.sum((np.array(self.dimensions)/2)**2))
            pattern=np.zeros_like(self.energy_potential)
            fib_max = self.fibonacci_sequence[9] if len(self.fibonacci_sequence)>9 and self.fibonacci_sequence[9]>0 else 1
            if fib_max == 0: return # Cannot scale
            for i in range(3,min(10,len(self.fibonacci_sequence))):
                 fib=self.fibonacci_sequence[i];
                 if fib<=0: continue
                 fib_d=(fib/fib_max)*max_dist*0.5 # Scale distance
                 if fib_d <= 0: continue
                 width=max(1.0,fib_d*0.1); shell=np.exp(-(dist-fib_d)**2/(2*width**2))
                 pattern+=shell
            max_p=np.max(pattern); pattern=pattern/max(max_p,1e-9) if max_p>0 else np.zeros_like(pattern)
            self.energy_potential = (0.95*self.energy_potential + 0.05*pattern).astype(DEFAULT_FIELD_DTYPE)
        except Exception as e: logger.error(f"Error applying fibonacci influence: {e}")


    # --- Keep State Management methods (adapted for 3D) ---
    def get_state(self) -> Dict[str, Any]:
        """Returns the current comprehensive state of the 3D Guff field."""
        # (Implementation similar to VoidField3D.get_state)
        state = { "field_name": self.field_name, "class": self.__class__.__name__, "uuid": self.uuid,
                  "dimensions": self.dimensions, "creation_time": self.creation_time, "metrics": self.get_field_metrics(),
                  "base_frequency": self.base_frequency, "resonance_quality": self.resonance_quality,
                  "formation_threshold": self.soul_formation_threshold, "strengthening_factor": self.strengthening_factor,
                  "sound_enabled": self.sound_enabled }
        return state

    def save_state(self, filename: str = None, include_arrays: bool = False) -> Optional[str]:
        """Saves the Guff field state."""
        # (Implementation similar to VoidField3D.save_state, saving energy and harmonic fields)
        if filename is None: filename=f"{self.field_name}_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        json_filepath=os.path.join(self.data_dir, f"{filename}.json")
        state_data=self.get_state(); array_paths={}
        if include_arrays:
            path_ep=self._save_array(self.energy_potential, f"{filename}_energy");
            if path_ep: array_paths["energy_potential_path"] = path_ep
            path_hf=self._save_array(self.harmonic_field, f"{filename}_harmonic");
            if path_hf: array_paths["harmonic_field_path"] = path_hf
            state_data.update(array_paths)
        try:
            with open(json_filepath,'w') as f: json.dump(state_data,f,indent=2,default=str)
            logger.info(f"Saved GuffField3D state to {json_filepath}"); return json_filepath
        except Exception as e: logger.error(f"Failed to save GuffField3D state: {e}"); return None

    def load_state(self, filepath: str, load_arrays: bool = True) -> bool:
        """Loads the Guff field state."""
        # (Implementation similar to VoidField3D.load_state, loading energy and harmonic fields)
        if not os.path.exists(filepath): logger.error(f"State file not found: {filepath}"); return False
        try:
             with open(filepath,'r') as f: state_data=json.load(f)
             if tuple(state_data.get("dimensions",[])) != self.dimensions: raise ValueError("Dimension mismatch")
             # Restore properties
             self.uuid=state_data.get("uuid",self.uuid); self.metrics=state_data.get("metrics",self.metrics)
             self.base_frequency=state_data.get("base_frequency",self.base_frequency)
             self.resonance_quality=state_data.get("resonance_quality",self.resonance_quality)
             # ... restore other parameters ...
             if load_arrays:
                  base_dir=os.path.dirname(filepath)
                  ep_path=state_data.get("energy_potential_path")
                  hf_path=state_data.get("harmonic_field_path")
                  if ep_path and os.path.exists(os.path.join(base_dir,ep_path)):
                       loaded_ep=np.load(os.path.join(base_dir,ep_path))
                       if loaded_ep.shape==self.dimensions: self.energy_potential=loaded_ep.astype(DEFAULT_FIELD_DTYPE)
                  if hf_path and os.path.exists(os.path.join(base_dir,hf_path)):
                       loaded_hf=np.load(os.path.join(base_dir,hf_path))
                       if loaded_hf.shape==self.dimensions: self.harmonic_field=loaded_hf.astype(DEFAULT_COMPLEX_DTYPE)
             logger.info(f"Loaded GuffField3D state from {filepath}"); return True
        except Exception as e: logger.error(f"Failed to load GuffField3D state: {e}"); return False

    def _save_array(self, array: np.ndarray, name_prefix: str) -> Optional[str]:
        """Helper to save a numpy array."""
        # (Implementation from previous step)
        array_filename = f"{name_prefix}.npy"
        array_filepath = os.path.join(self.data_dir, array_filename)
        try: np.save(array_filepath, array); logger.debug(f"Saved array '{name_prefix}'"); return os.path.basename(array_filepath)
        except Exception as e: logger.error(f"Failed to save array '{name_prefix}': {e}"); return None

    # --- Keep __str__ and __repr__ ---
    def __str__(self):
        return f"GuffField3D(name='{self.field_name}', dims={self.dimensions}, base_freq={self.base_frequency:.2f})"
    def __repr__(self):
        return f"<GuffField3D name='{self.field_name}' uuid='{self.uuid}'>"
    
    