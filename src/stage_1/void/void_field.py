"""
VoidField3D Module

Implements the 3D void field where soul spark formation occurs.
Represents the primordial quantum field manifesting potential via fluctuations.

Author: Soul Development Framework Team - Refactored to 3D with Strict Error Handling & Constants
"""

import numpy as np
import logging
import os
import uuid
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# --- Optional Dependencies ---
try:
    import scipy.ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Potential well identification requires SciPy.")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    logging.warning("Matplotlib not found. Visualization methods will be disabled.")

# --- Constants ---
try:
    from src.constants import (
        DEFAULT_DIMENSIONS_3D, DEFAULT_FIELD_DTYPE, DEFAULT_COMPLEX_DTYPE,
        PI, EDGE_OF_CHAOS_RATIO, VOID_BASE_FREQUENCY,
        VOID_FLUCTUATION_AMPLITUDE, VOID_SPARK_STABILITY_THRESHOLD,
        VOID_SPARK_ENERGY_THRESHOLD, VOID_SOUND_INTENSITY,
        VOID_FLUCTUATION_SOUND_INTENSITY, VOID_SPARK_SOUND_INTENSITY,
        VOID_TRANSFER_SOUND_INTENSITY, OUTPUT_DIR_BASE, DATA_DIR_BASE,
        SAMPLE_RATE, MAX_AMPLITUDE, LOG_LEVEL, LOG_FORMAT, FLOAT_EPSILON,
        VOID_INITIAL_ENERGY_MIN, VOID_INITIAL_ENERGY_MAX,
        VOID_SPARK_SIGMA_THRESHOLD, VOID_WELL_THRESHOLD_FACTOR,
        VOID_WELL_FILTER_SIZE, VOID_WELL_REGION_RADIUS,
        VOID_SOUND_QUANTUM_MIX, VOID_SOUND_CHAOS_MIX, VOID_SOUND_BASE_MIX, VOID_SOUND_COSMIC_MIX,
        VOID_SOUND_EXPANSION_FREQ, VOID_FLUCT_PULSE_MIN, VOID_FLUCT_PULSE_MAX,
        VOID_FLUCT_PULSE_WIDTH_MIN, VOID_FLUCT_PULSE_WIDTH_MAX, VOID_SPARK_SOUND_DURATION,
        VOID_SPARK_SOUND_TARGET_FREQ, VOID_SPARK_SOUND_EXP_FACTOR, VOID_SPARK_SOUND_NOISE_AMP,
        VOID_SPARK_SOUND_NOISE_FADE, VOID_SPARK_PING_START_FACTOR, VOID_SPARK_PING_LEN_FACTOR,
        VOID_SPARK_PING_FREQ, VOID_SPARK_PING_AMP, VOID_SPARK_PING_DECAY_FACTOR,
        VOID_TRANSFER_SOUND_DURATION, VOID_TRANSFER_SOUND_TARGET_FREQ, VOID_TRANSFER_SOUND_EXP_FACTOR,
        VOID_TRANSFER_AMP_H1, VOID_TRANSFER_AMP_H2, VOID_TRANSFER_AMP_H3,
        VOID_TRANSFER_RATIO_H1, VOID_TRANSFER_RATIO_H2, VOID_TRANSFER_RATIO_H3,
        VOID_TRANSFER_MIX_TONE, VOID_TRANSFER_MIX_NOISE, VOID_TRANSFER_WHOOSH_WIDTH_FACTOR,
        VOID_TRANSFER_WHOOSH_FREQ, VOID_TRANSFER_WHOOSH_AMP, VOID_SOUND_RESONANCE_MULT,
        VOID_SPARK_PROFILE_RADIUS
    )
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. VoidField3D cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
SOUND_MODULES_AVAILABLE = False
SoundGenerator = NoiseGenerator = UniverseSounds = None
try:
    from sound.sound_generator import SoundGenerator
    from sound.noise_generator import NoiseGenerator
    from sound.sounds_of_universe import UniverseSounds
    SOUND_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import sound modules for VoidField3D: {e}. Sound features disabled.")

GEOMETRY_MODULES_AVAILABLE = False
EMBED_FUNCTIONS = {}
try:
    from stage_1.sacred_geometry.flower_of_life import embed_flower_in_field
    EMBED_FUNCTIONS["flower_of_life"] = embed_flower_in_field
    from stage_1.sacred_geometry.seed_of_life import embed_seed_in_field
    EMBED_FUNCTIONS["seed_of_life"] = embed_seed_in_field
    from stage_1.sacred_geometry.merkaba import embed_merkaba_in_field
    EMBED_FUNCTIONS["merkaba"] = embed_merkaba_in_field
    GEOMETRY_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import all geometry embedders for VoidField3D: {e}. Embedding limited.")

# --- Logging Setup ---
log_file_path = os.path.join("logs", "void_field_3d.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('void_field_3d')

class VoidField3D(object):
    """
    Implementation of the 3D void field where soul sparks form.
    Enforces strict error handling.
    """

    def __init__(self, dimensions: Tuple[int, int, int] = DEFAULT_DIMENSIONS_3D,
                 field_name: str = "void_field", data_dir: str = DATA_DIR_BASE):
        """Initialize a new 3D void field. Fails hard on invalid configuration."""
        # --- Input Validation ---
        if not isinstance(dimensions, tuple) or len(dimensions) != 3 or not all(isinstance(d, int) and d > 0 for d in dimensions):
            raise ValueError(f"Dimensions must be a tuple of 3 positive integers, got {dimensions}")
        if not field_name or not isinstance(field_name, str):
            raise ValueError("Field name must be a non-empty string.")

        self.field_name: str = field_name
        self.dimensions: Tuple[int, int, int] = dimensions
        self.uuid: str = str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.data_dir: str = os.path.join(data_dir, "fields", field_name)
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except OSError as e: logger.critical(f"CRITICAL: Failed to create data directory {self.data_dir}: {e}"); raise

        logger.info(f"Initializing VoidField3D '{self.field_name}' (UUID: {self.uuid}) with dimensions {self.dimensions}")

        # --- Initialize 3D Field Arrays ---
        try:
            self.energy_potential: np.ndarray = np.zeros(dimensions, dtype=DEFAULT_FIELD_DTYPE)
            self.quantum_state: np.ndarray = np.zeros(dimensions, dtype=DEFAULT_COMPLEX_DTYPE)
        except MemoryError as me: logger.critical(f"CRITICAL: Insufficient memory {dimensions}. {me}"); raise
        except Exception as e: logger.critical(f"CRITICAL: Error initializing field arrays: {e}"); raise

        # --- Properties ---
        self.planck_constant: float = 1.0 # Simulation constant
        self.edge_of_chaos_ratio: float = EDGE_OF_CHAOS_RATIO
        self.fluctuation_amplitude: float = VOID_FLUCTUATION_AMPLITUDE
        if not (0 < self.edge_of_chaos_ratio < 1): raise ValueError("EDGE_OF_CHAOS_RATIO invalid")
        if self.fluctuation_amplitude < 0: raise ValueError("VOID_FLUCTUATION_AMPLITUDE invalid")

        # --- State Tracking ---
        self.potential_wells: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {"creation_time": self.creation_time}

        # --- Initialize Sound Components ---
        self.sound_enabled: bool = SOUND_MODULES_AVAILABLE
        self.sound_output_dir: str = os.path.join(OUTPUT_DIR_BASE, "void")
        os.makedirs(self.sound_output_dir, exist_ok=True)
        self.sound_generator: Optional[SoundGenerator] = None
        self.noise_gen: Optional[NoiseGenerator] = None
        self.universe_sounds: Optional[UniverseSounds] = None
        self.void_sound: Optional[np.ndarray] = None
        self.base_frequency: float = 0.0

        if self.sound_enabled:
            self._initialize_sound_objects() # Fails hard if objects cannot init
            self.base_frequency = VOID_BASE_FREQUENCY
            if self.base_frequency <= 0: raise ValueError("VOID_BASE_FREQUENCY must be positive.")
        else:
            logger.warning("Sound modules failed to import. Sound features disabled for VoidField.")

        # --- Initialize Field State and Sound ---
        self._initialize_field()
        if self.sound_enabled:
            self._initialize_sound_system()

        logger.info(f"Void field '{self.field_name}' initialized successfully.")

    def _initialize_sound_objects(self):
        """Helper to initialize sound objects, failing hard if unavailable."""
        if not SOUND_MODULES_AVAILABLE: raise RuntimeError("Essential sound modules are not available for VoidField.")
        try:
            self.sound_generator = SoundGenerator(output_dir=self.sound_output_dir, sample_rate=SAMPLE_RATE)
            self.noise_gen = NoiseGenerator(output_dir=self.sound_output_dir, sample_rate=SAMPLE_RATE)
            self.universe_sounds = UniverseSounds(output_dir=self.sound_output_dir, sample_rate=SAMPLE_RATE)
            logger.info("Void sound objects initialized successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize Void sound objects: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize sound objects for VoidField: {e}") from e

    def _initialize_field(self):
        """Initialize the 3D field with baseline quantum vacuum energy."""
        try:
            vacuum_energy = np.random.uniform(VOID_INITIAL_ENERGY_MIN, VOID_INITIAL_ENERGY_MAX, self.dimensions).astype(DEFAULT_FIELD_DTYPE)
            vacuum_energy[vacuum_energy < 0] = 0
            self.energy_potential = vacuum_energy
            amplitudes = np.sqrt(self.energy_potential)
            phases = np.random.uniform(0, 2 * PI, self.dimensions)
            self.quantum_state = amplitudes.astype(DEFAULT_COMPLEX_DTYPE) * np.exp(1j * phases).astype(DEFAULT_COMPLEX_DTYPE)
            self._normalize_field()
            logger.info("Void field baseline initialized (3D)")
        except Exception as e:
             logger.critical(f"CRITICAL: Error initializing void field state: {e}", exc_info=True)
             raise RuntimeError(f"Failed to initialize void field state: {e}") from e

    def _initialize_sound_system(self):
        """Initialize sound system for the Void field. Fails hard."""
        if not self.sound_enabled: return
        try:
            self.void_sound = self.generate_void_sound(duration=10.0)
            if self.void_sound is None or len(self.void_sound) == 0:
                raise RuntimeError("Failed to generate critical initial void sound.")
            logger.info("Void field sound system initialized")
            success = self.apply_sound_to_field(self.void_sound, intensity=VOID_SOUND_INTENSITY)
            if not success:
                 raise RuntimeError("Failed to apply initial void sound to field.")
        except Exception as e:
             logger.critical(f"CRITICAL: Error initializing sound system: {e}", exc_info=True)
             raise RuntimeError(f"Failed to initialize sound system: {e}") from e

    def _normalize_field(self):
        """Normalize field energy (clipping) and update quantum state."""
        try:
             self.energy_potential = np.clip(self.energy_potential, 0.0, 1.0)
             energy_nonneg = np.maximum(0.0, self.energy_potential)
             current_phases = np.angle(self.quantum_state)
             new_amplitudes = np.sqrt(energy_nonneg)
             self.quantum_state = new_amplitudes.astype(DEFAULT_COMPLEX_DTYPE) * np.exp(1j * current_phases).astype(DEFAULT_COMPLEX_DTYPE)
        except Exception as e:
             logger.error(f"Error during field normalization: {e}", exc_info=True)
             # Avoid raising error for normalization itself

    def _normalize_sound(self, sound_array: np.ndarray, target_amplitude: float) -> np.ndarray:
        """Normalizes sound array. Fails hard."""
        if sound_array is None or sound_array.size == 0: raise ValueError("Cannot normalize empty array.")
        try:
            max_abs = np.max(np.abs(sound_array))
            if not np.isfinite(max_abs): raise ValueError("Cannot normalize array with NaN/Inf.")
            if max_abs > FLOAT_EPSILON: normalized = sound_array / max_abs * target_amplitude
            else: normalized = np.zeros_like(sound_array)
            return np.clip(normalized, -target_amplitude, target_amplitude).astype(np.float32)
        except Exception as e:
            logger.error(f"Error during sound normalization: {e}", exc_info=True)
            raise RuntimeError("Normalization failed.") from e

    def embed_sacred_geometry(self, geometry_type: str, strength: float = 1.0, position: Optional[Tuple[int, int, int]] = None):
        """Embed a sacred geometry pattern into the 3D field. Fails hard."""
        if not GEOMETRY_MODULES_AVAILABLE:
             raise RuntimeError(f"Cannot embed {geometry_type}: Geometry embedders not available.")
        if position is None: position = tuple(d // 2 for d in self.dimensions)
        elif len(position) != 3 or not all(isinstance(p, int) for p in position):
             raise ValueError("Position must be a tuple of 3 integers.")
        strength = max(0.0, min(1.0, strength))

        geometry_type_lower = geometry_type.lower()
        if geometry_type_lower not in EMBED_FUNCTIONS:
             raise ValueError(f"Unsupported or unavailable geometry type: {geometry_type}")

        logger.info(f"Embedding {geometry_type} at {position} with strength {strength}")
        embed_func = EMBED_FUNCTIONS[geometry_type_lower]
        # Use constants for default sizes if available
        size_param = min(self.dimensions) // 8 # DEFAULT_GEOMETRY_SIZE_FACTOR?
        if geometry_type_lower == 'seed_of_life': size_param = min(self.dimensions) // 10 # DEFAULT_SEED_SIZE_FACTOR?

        try:
            # Embedder modifies self.energy_potential in place
            embed_func(self.energy_potential, position, size_param, strength)
            self._normalize_field()
            logger.info(f"Successfully embedded {geometry_type}")
            return {'type': geometry_type, 'position': position, 'strength': strength, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error embedding {geometry_type}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to embed {geometry_type}: {e}") from e

    def simulate_quantum_fluctuations(self, iterations: int = 10, amplitude: Optional[float] = None):
        """Simulate quantum fluctuations in the 3D void field. Fails hard."""
        if amplitude is None: amplitude = self.fluctuation_amplitude
        if not isinstance(amplitude, (int, float)) or amplitude < 0: raise ValueError("Amplitude must be non-negative.")
        if not isinstance(iterations, int) or iterations <= 0: raise ValueError("Iterations must be positive.")

        logger.info(f"Simulating {iterations} quantum fluctuations (3D) with amplitude {amplitude:.3f}")
        high_energy_points = []
        try:
            for i in range(iterations):
                fluctuations = np.random.normal(0, amplitude, self.dimensions) + 1j * np.random.normal(0, amplitude, self.dimensions)
                self.quantum_state = (self.quantum_state + fluctuations.astype(DEFAULT_COMPLEX_DTYPE))
                new_energy = np.abs(self.quantum_state)**2
                chaos_mask = np.random.random(self.dimensions) < self.edge_of_chaos_ratio
                chaotic_flucts = np.random.normal(0, amplitude*2, self.dimensions)*chaos_mask
                new_energy += np.abs(chaotic_flucts)
                self.energy_potential = np.maximum(0, new_energy).astype(DEFAULT_FIELD_DTYPE)
                self._normalize_field()

                mean_e=np.mean(self.energy_potential); std_e=np.std(self.energy_potential)
                threshold = mean_e + VOID_SPARK_SIGMA_THRESHOLD * std_e
                points_indices = np.where(self.energy_potential > threshold)
                for j in range(len(points_indices[0])):
                    point = tuple(int(points_indices[d][j]) for d in range(3))
                    energy = float(self.energy_potential[point])
                    high_energy_points.append({'position':point,'energy':energy,'iteration':i})

            # Fluctuation sound
            if self.sound_enabled and iterations >= 5:
                try:
                    fluctuation_sound = self.generate_quantum_fluctuation_sound(amplitude=amplitude)
                    if fluctuation_sound is not None:
                        if not self.apply_sound_to_field(fluctuation_sound, intensity=VOID_FLUCTUATION_SOUND_INTENSITY):
                             logger.warning("Failed to apply fluctuation sound.")
                except RuntimeError as sound_err: # Catch failure from sound gen
                     logger.warning(f"Fluctuation sound processing failed: {sound_err}")

            metrics = self.get_field_metrics() # Fails hard if metrics calculation fails
            result = {'iterations': iterations, 'amplitude': amplitude, 'high_energy_points': high_energy_points, **metrics}
            return result
        except Exception as e:
             logger.critical(f"CRITICAL Error during quantum fluctuation simulation: {e}", exc_info=True)
             raise RuntimeError(f"Quantum fluctuation failed: {e}") from e

    def identify_potential_wells(self, threshold_factor: float = VOID_WELL_THRESHOLD_FACTOR) -> List[Dict[str, Any]]:
        """Identify potential wells in 3D field. Raises RuntimeError on failure."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Cannot identify potential wells: SciPy is not available.")
        if not (isinstance(threshold_factor, (int, float)) and threshold_factor > 0):
             raise ValueError(f"Invalid threshold_factor: {threshold_factor}")

        self.potential_wells = []
        logger.info(f"Identifying potential wells (3D) with threshold factor {threshold_factor:.2f}...")
        try:
            gradients = np.gradient(self.energy_potential)
            gradient_magnitude = np.sqrt(np.sum([g**2 for g in gradients], axis=0))
            filter_size = VOID_WELL_FILTER_SIZE
            local_min_gradient = scipy.ndimage.minimum_filter(gradient_magnitude, size=filter_size, mode='reflect')

            with np.errstate(invalid='ignore'): # Ignore potential invalid comparisons
                minima_mask = (gradient_magnitude == local_min_gradient) & np.isfinite(gradient_magnitude)
                # Add check for low gradient value relative to mean
                minima_mask &= (gradient_magnitude < np.mean(gradient_magnitude) * VOID_WELL_GRADIENT_FACTOR)

            energy_threshold = np.mean(self.energy_potential) * threshold_factor
            threshold_mask = (self.energy_potential > energy_threshold)
            wells_mask = minima_mask & threshold_mask
            well_indices = np.where(wells_mask)
            num_wells_found = len(well_indices[0])
            logger.info(f"Found {num_wells_found} potential well candidates.")

            wells_list = []
            for i in range(num_wells_found):
                pos = tuple(int(well_indices[dim_idx][i]) for dim_idx in range(3))
                well_energy = float(self.energy_potential[pos])
                well_gradient = float(gradient_magnitude[pos])

                radius = VOID_WELL_REGION_RADIUS
                slices = tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(pos,(radius,)*3,self.dimensions))
                region = self.energy_potential[slices]
                if region.size > 1:
                     depth = float(np.max(region) - well_energy)
                     region_std = np.std(region)
                     stability = 1.0 / (region_std + FLOAT_EPSILON) if region_std > FLOAT_EPSILON else 1e10 # Use epsilon
                else: depth = 0.0; stability = 0.0

                wells_list.append({ 'position': pos, 'energy': well_energy, 'gradient': well_gradient,
                                    'depth': depth, 'stability': stability, 'timestamp': datetime.now().isoformat(),
                                    'id': str(uuid.uuid4()) })

            self.potential_wells = sorted(wells_list, key=lambda w: w['stability'] * w['depth'], reverse=True)
            logger.info(f"Identified {len(self.potential_wells)} potential wells meeting criteria.")
            return self.potential_wells
        except Exception as e:
             logger.error(f"Error identifying potential wells: {e}", exc_info=True)
             self.potential_wells = [] # Clear wells on error
             raise RuntimeError("Potential well identification failed.") from e

    # --- Sound Generation Methods (Fail hard using _get_sound_object) ---
    def generate_void_sound(self, duration: float = 30.0) -> np.ndarray:
        """Generates void sound. Fails hard."""
        noise_gen = self._get_sound_object('noise_gen')
        uni_sounds = self._get_sound_object('universe_sounds')
        snd_gen = self._get_sound_object('sound_generator')
        if duration <= 0: raise ValueError("Duration must be positive.")
        try:
            sample_rate=self.sample_rate; num_samples = int(duration * sample_rate)
            quantum_noise = noise_gen.generate_noise('quantum', duration, 0.7)
            edge_chaos = noise_gen.generate_noise('edge_of_chaos', duration, 0.5)
            cosmic_bg = uni_sounds.generate_cosmic_background(duration, 0.4)
            if quantum_noise is None: raise RuntimeError("Quantum noise generation failed.")
            if edge_chaos is None: raise RuntimeError("Edge of chaos noise generation failed.")
            if cosmic_bg is None: cosmic_bg = np.zeros(num_samples, dtype=DEFAULT_FIELD_DTYPE) # Allow fallback

            min_len = min(len(quantum_noise), len(edge_chaos), len(cosmic_bg), num_samples)
            void_base = VOID_SOUND_QUANTUM_MIX*quantum_noise[:min_len] + VOID_SOUND_CHAOS_MIX*edge_chaos[:min_len]
            void_sound = VOID_SOUND_BASE_MIX*void_base + VOID_SOUND_COSMIC_MIX*cosmic_bg[:min_len]
            # Add expansion mod
            if VOID_SOUND_EXPANSION_FREQ > 0:
                time = np.linspace(0, duration, min_len, endpoint=False)
                void_sound *= (0.8 + 0.2 * np.sin(2 * PI * VOID_SOUND_EXPANSION_FREQ * time)).astype(np.float32)

            void_sound = self._normalize_sound(void_sound, MAX_AMPLITUDE)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"void_field_{timestamp}.wav"
            if not snd_gen.save_sound(void_sound, filename=fn, description="Void Field Base"):
                 raise IOError("Failed to save void sound.")
            return void_sound
        except Exception as e: logger.error(f"Void sound gen error: {e}"); raise RuntimeError("Void sound generation failed") from e

    def generate_quantum_fluctuation_sound(self, amplitude: float, duration: float = 5.0) -> np.ndarray:
        """Generates fluctuation sound. Fails hard."""
        noise_gen = self._get_sound_object('noise_gen'); snd_gen = self._get_sound_object('sound_generator')
        if duration <= 0 or amplitude < 0: raise ValueError("Duration positive, amplitude non-negative.")
        try:
            noise = noise_gen.generate_noise('quantum', duration, amplitude)
            if noise is None: raise RuntimeError("Quantum noise component failed.")
            num_samples=len(noise); time=np.linspace(0,duration,num_samples)
            for _ in range(np.random.randint(VOID_FLUCT_PULSE_MIN, VOID_FLUCT_PULSE_MAX)):
                 pt=np.random.uniform(0,duration); pw=np.random.uniform(VOID_FLUCT_PULSE_WIDTH_MIN, VOID_FLUCT_PULSE_WIDTH_MAX)
                 noise += amplitude * np.exp(-(time-pt)**2 / (2*max(FLOAT_EPSILON,pw**2)))
            noise = self._normalize_sound(noise, MAX_AMPLITUDE)
            timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"quantum_fluctuation_{timestamp}.wav"
            if not snd_gen.save_sound(noise, filename=fn): raise IOError("Failed to save fluctuation sound.")
            return noise
        except Exception as e: logger.error(f"Fluctuation sound gen error: {e}"); raise RuntimeError("Fluctuation sound gen failed") from e

    def generate_spark_formation_sound(self, spark_result: Dict[str, Any]) -> np.ndarray:
        """Generates spark formation sound. Fails hard."""
        noise_gen=self._get_sound_object('noise_gen'); snd_gen=self._get_sound_object('sound_generator')
        if not isinstance(spark_result, dict) or not spark_result.get("spark_formed"): raise ValueError("Invalid spark_result.")
        try:
            duration=VOID_SPARK_SOUND_DURATION; sample_rate=self.sample_rate; num_samples=int(duration*sample_rate)
            time=np.linspace(0,duration,num_samples); base_f=self.base_frequency; target_f=VOID_SPARK_SOUND_TARGET_FREQ
            if base_f <= 0 or target_f <= 0: raise ValueError("Invalid frequencies for spark sound.")
            freq_curve=base_f+(target_f-base_f)*(1-np.exp(VOID_SPARK_SOUND_EXP_FACTOR*time/max(FLOAT_EPSILON,duration))); phase=np.cumsum(freq_curve/sample_rate)
            sound = (VOID_TRANSFER_AMP_H1*np.sin(2*PI*phase*VOID_TRANSFER_RATIO_H1) +
                     VOID_TRANSFER_AMP_H2*np.sin(2*PI*phase*VOID_TRANSFER_RATIO_H2) +
                     VOID_TRANSFER_AMP_H3*np.sin(2*PI*phase*VOID_TRANSFER_RATIO_H3)).astype(np.float32)
            noise = noise_gen.generate_noise('white', duration, VOID_SPARK_SOUND_NOISE_AMP)
            if noise is not None: sound += noise[:num_samples] * np.exp(VOID_SPARK_SOUND_NOISE_FADE * time / max(FLOAT_EPSILON, duration))
            # Add ping
            ping_t=int(VOID_SPARK_PING_START_FACTOR*num_samples); ping_l=int(VOID_SPARK_PING_LEN_FACTOR*num_samples); ping_f=VOID_SPARK_PING_FREQ
            if ping_t+ping_l <= num_samples and ping_f > 0 and ping_l > 0:
                 ping_tl=np.arange(ping_l)/sample_rate; ping_w=VOID_SPARK_PING_AMP*np.sin(2*PI*ping_f*ping_tl)*np.exp(VOID_SPARK_PING_DECAY_FACTOR*ping_tl/max(FLOAT_EPSILON,ping_l/sample_rate))
                 sound[ping_t:ping_t+ping_l] += ping_w[:len(sound[ping_t:ping_t+ping_l])]
            sound = self._normalize_sound(sound, MAX_AMPLITUDE)
            timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"spark_formation_{timestamp}.wav"
            filepath = snd_gen.save_sound(sound, filename=fn)
            if not filepath: raise IOError("Failed to save spark formation sound.")
            spark_result["formation_sound_path"] = os.path.basename(filepath)
            return sound
        except Exception as e: logger.error(f"Spark formation sound gen error: {e}"); raise RuntimeError("Spark formation sound gen failed") from e

    def generate_transfer_sound(self, spark_data: Dict[str, Any]) -> np.ndarray:
        """Generates transfer sound. Fails hard."""
        uni_sounds=self._get_sound_object('universe_sounds'); snd_gen=self._get_sound_object('sound_generator')
        if not isinstance(spark_data, dict): raise ValueError("Invalid spark_data.")
        try:
             duration=VOID_TRANSFER_SOUND_DURATION; sample_rate=self.sample_rate; num_samples=int(duration*sample_rate)
             time=np.linspace(0,duration,num_samples); void_f=self.base_frequency; guff_f=VOID_TRANSFER_SOUND_TARGET_FREQ
             if void_f <= 0 or guff_f <= 0: raise ValueError("Invalid frequencies for transfer sound.")
             freq_curve=void_f+(guff_f-void_f)*(1-np.exp(VOID_TRANSFER_SOUND_EXP_FACTOR*time/max(FLOAT_EPSILON,duration))); phase=np.cumsum(freq_curve/sample_rate)
             sound = (VOID_TRANSFER_AMP_H1*np.sin(2*PI*phase*VOID_TRANSFER_RATIO_H1) +
                      VOID_TRANSFER_AMP_H2*np.sin(2*PI*phase*VOID_TRANSFER_RATIO_H2) +
                      VOID_TRANSFER_AMP_H3*np.sin(2*PI*phase*VOID_TRANSFER_RATIO_H3)).astype(np.float32)
             noise = uni_sounds.generate_dimensional_transition(duration,sample_rate,'void_to_guff')
             if noise is not None:
                 min_l=min(len(sound),len(noise)); sound=VOID_TRANSFER_MIX_TONE*sound[:min_l]+VOID_TRANSFER_MIX_NOISE*noise[:min_l]
             # Add whoosh
             mid=num_samples//2; width=int(VOID_TRANSFER_WHOOSH_WIDTH_FACTOR*num_samples); start=mid-width//2; end=start+width
             if 0<=start and end<num_samples: sound[start:end]+=VOID_TRANSFER_WHOOSH_AMP*np.sin(2*PI*VOID_TRANSFER_WHOOSH_FREQ*np.linspace(0,1,end-start))
             sound = self._normalize_sound(sound, MAX_AMPLITUDE)
             timestamp=datetime.now().strftime("%Y%m%d%H%M%S"); fn=f"spark_transfer_{timestamp}.wav"
             filepath = snd_gen.save_sound(sound, filename=fn)
             if not filepath: raise IOError("Failed to save transfer sound.")
             spark_data["transfer_sound_path"] = os.path.basename(filepath)
             return sound
        except Exception as e: logger.error(f"Transfer sound gen error: {e}"); raise RuntimeError("Transfer sound gen failed") from e

    # --- apply_sound_to_field (Keep 3D version, ensure fail hard internally) ---
    def apply_sound_to_field(self, sound_waveform: np.ndarray, intensity: float = 0.5) -> bool:
        """Applies sound waveform modulation to the 3D field energy potential."""
        if not self.sound_enabled: logger.warning("Sound disabled, cannot apply."); return False
        if sound_waveform is None or sound_waveform.size == 0: logger.warning("Empty sound waveform."); return False
        if not isinstance(sound_waveform, np.ndarray): raise TypeError("sound_waveform must be a numpy array.")
        intensity = max(0.0, min(1.0, intensity))
        if intensity == 0: return False
        try:
            # Use internal strict normalize
            norm_sound = self._normalize_sound(sound_waveform, 1.0)
            sound_rms = np.sqrt(np.mean(norm_sound**2))
            # Use void specific resonance multiplier from constants
            mod_factor = 1.0 + (sound_rms * intensity * VOID_SOUND_RESONANCE_MULT)
            self.energy_potential *= mod_factor
            self._normalize_field() # Normalize after applying modulation
            logger.info(f"Applied sound modulation (RMS:{sound_rms:.3f}, Intensity:{intensity:.2f}) to {self.field_name}")
            return True
        except Exception as e:
            logger.error(f"Error applying sound waveform to field: {e}", exc_info=True)
            # Fail hard? Or just return False? Let's return False for application error.
            return False

    # --- check_spark_formation (Keep 3D, fail-hard version) ---
    def check_spark_formation(self, min_well_stability: float = VOID_SPARK_STABILITY_THRESHOLD,
                              min_energy_threshold: float = VOID_SPARK_ENERGY_THRESHOLD) -> Dict[str, Any]:
        """Checks for spark formation conditions in 3D wells. Fails hard on critical errors."""
        # (Implementation from previous step - ensuring it raises RuntimeError on internal failure)
        if not self.potential_wells:
            try: self.identify_potential_wells() # Attempt to find wells, can raise RuntimeError
            except RuntimeError: logger.error("Failed to identify wells during spark check."); return {"spark_formed": False, "reason": "Well identification failed", "timestamp": datetime.now().isoformat()}
        if not self.potential_wells: return {"spark_formed": False, "reason": "No potential wells found", "timestamp": datetime.now().isoformat()}

        best_well=None; best_score=-1.0
        try: max_field_energy = np.max(self.energy_potential)
        except Exception as e: raise RuntimeError("Failed to get max field energy for spark check.") from e
        if max_field_energy <= FLOAT_EPSILON: return {"spark_formed": False, "reason": "Field energy near zero", "timestamp": datetime.now().isoformat()}

        for well in self.potential_wells:
            energy_norm = well['energy'] / max_field_energy; stability = well['stability']; score = stability * energy_norm
            if (stability >= min_well_stability and energy_norm >= min_energy_threshold and score > best_score):
                best_well = well; best_score = score

        if best_well is not None:
            logger.info(f"Soul spark formed at {best_well['position']} (3D) with score {best_score:.4f}")
            result = { "spark_formed": True, "well": best_well, "formation_score": best_score, "position": best_well['position'],
                       "energy": best_well['energy'], "stability": best_well['stability'], "timestamp": datetime.now().isoformat(),
                       "spark_id": str(uuid.uuid4()), "dimensions": self.dimensions }
            if self.sound_enabled:
                try: self.generate_spark_formation_sound(result) # Can raise error
                except RuntimeError as sound_err: logger.warning(f"Spark formed, but formation sound failed: {sound_err}")
                # Don't fail spark formation if only sound fails, but log warning
            return result
        else:
            logger.info("No soul spark formed - requirements not met.")
            best_s = max([w['stability'] for w in self.potential_wells], default=0); best_e = max([w['energy'] for w in self.potential_wells], default=0)
            best_e_norm = best_e / max_field_energy
            return { "spark_formed": False, "reason": "No well met requirements", "best_stability": best_s,
                     "best_energy_norm": best_e_norm, "timestamp": datetime.now().isoformat() }


    # --- transfer_spark_to_guff (Keep 3D, fail-hard version) ---
    def transfer_spark_to_guff(self, spark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares a formed 3D spark for transfer. Raises ValueError/RuntimeError on failure."""
        # (Implementation from previous step - ensuring it raises errors)
        if not isinstance(spark_data, dict) or not spark_data.get("spark_formed", False): raise ValueError("Invalid spark_data.")
        if "position" not in spark_data or len(spark_data["position"]) != 3: raise ValueError("Spark data missing 3D position.")
        spark_id = spark_data.get("spark_id"); if not spark_id: raise ValueError("Spark data missing 'spark_id'.")
        logger.info(f"Preparing spark {spark_id} for transfer from 3D Void.")
        try:
            transfer_sound_path = spark_data.get("transfer_sound_path")
            if self.sound_enabled and not transfer_sound_path:
                transfer_sound = self.generate_transfer_sound(spark_data) # Can raise error
                if transfer_sound is not None:
                    if not self.apply_sound_to_field(transfer_sound, intensity=VOID_TRANSFER_SOUND_INTENSITY):
                         logger.warning("Failed to apply transfer sound.")
                    transfer_sound_path = spark_data.get("transfer_sound_path")
                else: logger.warning("Transfer sound generation failed.") # generate should have raised error

            void_metrics = self.get_field_metrics() # Can raise error
            spark_profile = self.get_spark_energy(spark_data["position"], radius=VOID_SPARK_PROFILE_RADIUS) # Can raise error

            transfer_data = { "spark_id": spark_id, "position": spark_data["position"], "energy": spark_data["energy"],
                              "stability": spark_data["stability"], "formation_score": spark_data["formation_score"],
                              "void_field_metrics": void_metrics, "spark_energy_profile": spark_profile,
                              "transfer_timestamp": datetime.now().isoformat(), "transfer_sound_path": transfer_sound_path,
                              "transfer_complete": True }
            logger.info(f"Spark {spark_id} prepared for transfer.")
            return transfer_data
        except Exception as e: logger.error(f"Error preparing spark {spark_id} for transfer: {e}"); raise RuntimeError("Transfer prep failed") from e

    # --- Keep get_field_metrics, get_spark_energy (3D versions, fail hard) ---
    def get_field_metrics(self) -> Dict[str, Any]:
        """Gets metrics about the 3D void field. Raises RuntimeError on failure."""
        # (Implementation from previous step)
        try:
            energy=self.energy_potential; quantum=self.quantum_state
            total_e=np.sum(energy); mean_e=np.mean(energy); max_e=np.max(energy); std_e=np.std(energy)
            if not all(np.isfinite(x) for x in [total_e, mean_e, max_e, std_e]): raise ValueError("NaN/Inf in energy metrics.")
            phases=np.angle(quantum); coherence=np.abs(np.mean(np.exp(1j*phases))) if quantum.size>0 else 0.0
            if not np.isfinite(coherence): raise ValueError("NaN/Inf in phase coherence.")
            grads=np.gradient(energy); grad_mag=np.sqrt(np.sum([g**2 for g in grads], axis=0))
            mean_g=np.mean(grad_mag); max_g=np.max(grad_mag)
            if not np.isfinite(mean_g) or not np.isfinite(max_g): raise ValueError("NaN/Inf in gradient metrics.")
            entropy=-np.sum((energy/(total_e+FLOAT_EPSILON))*np.log2(energy/(total_e+FLOAT_EPSILON)+FLOAT_EPSILON)) if total_e>FLOAT_EPSILON else 0.0
            return { "total_energy":float(total_e),"mean_energy":float(mean_e),"max_energy":float(max_e),"energy_std":float(std_e),
                     "phase_coherence":float(coherence),"mean_gradient":float(mean_g),"max_gradient":float(max_g),
                     "energy_entropy":float(entropy),"num_potential_wells":len(self.potential_wells),
                     "edge_of_chaos_ratio":self.edge_of_chaos_ratio,"dimensions":self.dimensions,"timestamp":datetime.now().isoformat() }
        except Exception as e: logger.error(f"Error getting void metrics: {e}"); raise RuntimeError("Metric calculation failed") from e

    def get_spark_energy(self, position: tuple, radius: int = VOID_SPARK_PROFILE_RADIUS):
        """Gets energy profile around 3D spark. Raises ValueError/RuntimeError."""
        # (Implementation from previous step)
        if len(position)!=3: raise ValueError("Position must be 3D tuple"); radius=max(1,int(radius))
        slices=tuple(slice(max(0,p-radius),min(d,p+r+1)) for p,r,d in zip(position,(radius,)*3,self.dimensions))
        energy_r=self.energy_potential[slices]; quantum_r=self.quantum_state[slices]
        if energy_r.size==0: raise ValueError(f"Spark energy region empty: {position} r={radius}")
        try:
            mean_g=0.0
            if all(s>1 for s in energy_r.shape) and SCIPY_AVAILABLE:
                 try: grads=np.gradient(energy_r); mean_g=np.mean(np.sqrt(np.sum([g**2 for g in grads],axis=0)))
                 except Exception as grad_e: logger.warning(f"Gradient calc failed for spark profile: {grad_e}") # Warn, don't fail
            phases=np.angle(quantum_r); coherence=np.abs(np.mean(np.exp(1j*phases))) if quantum_r.size>0 else 0.0
            # Check results for NaN/Inf
            results = {"total_energy":float(np.sum(energy_r)),"mean_energy":float(np.mean(energy_r)),"max_energy":float(np.max(energy_r)),
                       "energy_std":float(np.std(energy_r)),"energy_gradient":float(mean_g),"phase_coherence":float(coherence)}
            if not all(np.isfinite(v) for v in results.values()): raise ValueError("NaN/Inf detected in spark profile results.")
            return {"position":position,"radius":radius,**results,"timestamp":datetime.now().isoformat()}
        except Exception as e: logger.error(f"Error getting spark profile: {e}"); raise RuntimeError("Spark profile failed") from e

    # --- Keep State Management methods (fail hard versions) ---
    def get_state(self) -> Dict[str, Any]:
        """Returns the current comprehensive state of the 3D Void field."""
        # (Implementation from previous step)
        try: metrics = self.get_field_metrics() # Get current metrics
        except RuntimeError: metrics = self.metrics # Use last known metrics if calc fails now
        state = { "field_name": self.field_name, "class": self.__class__.__name__, "uuid": self.uuid,
                  "dimensions": self.dimensions, "creation_time": self.creation_time, "metrics": metrics,
                  "potential_well_count": len(self.potential_wells), "base_frequency": self.base_frequency,
                  "edge_of_chaos_ratio": self.edge_of_chaos_ratio, "fluctuation_amplitude": self.fluctuation_amplitude,
                  "sound_enabled": self.sound_enabled }
        return state

    def save_state(self, filename: str = None, include_arrays: bool = False) -> str:
        """Saves the Void field state. Fails hard on error."""
        # (Implementation from previous step, raises errors)
        if filename is None: filename=f"{self.field_name}_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        json_filepath=os.path.join(self.data_dir, f"{filename}.json"); state_data=self.get_state(); array_paths={}
        if include_arrays:
            path_ep=self._save_array(self.energy_potential, f"{filename}_energy");
            if path_ep: array_paths["energy_potential_path"]=path_ep
            else: raise IOError("Failed to save energy potential array.") # Fail hard if array save fails
            path_qs=self._save_array(self.quantum_state, f"{filename}_quantum");
            if path_qs: array_paths["quantum_state_path"]=path_qs
            else: raise IOError("Failed to save quantum state array.")
            state_data.update(array_paths)
        try:
            with open(json_filepath,'w') as f: json.dump(state_data,f,indent=2,default=str)
            logger.info(f"Saved VoidField3D state to {json_filepath}"); return json_filepath
        except Exception as e: logger.error(f"Failed to save VoidField3D state JSON: {e}"); raise IOError("State JSON saving failed") from e

    def load_state(self, filepath: str, load_arrays: bool = True) -> bool:
        """Loads the Void field state. Fails hard on error."""
        # (Implementation from previous step, raises errors)
        if not os.path.exists(filepath): raise FileNotFoundError(f"State file not found: {filepath}")
        try:
             with open(filepath,'r') as f: state_data=json.load(f)
             if tuple(state_data.get("dimensions",[])) != self.dimensions: raise ValueError("Dimension mismatch")
             self.uuid=state_data.get("uuid",self.uuid); self.metrics=state_data.get("metrics",self.metrics)
             # Restore properties strictly if they exist in file, otherwise keep defaults but warn
             self.base_frequency = float(state_data.get("base_frequency", self.base_frequency))
             self.edge_of_chaos_ratio = float(state_data.get("edge_of_chaos_ratio", self.edge_of_chaos_ratio))
             self.fluctuation_amplitude = float(state_data.get("fluctuation_amplitude", self.fluctuation_amplitude))
             self.sound_enabled = bool(state_data.get("sound_enabled", self.sound_enabled))
             # ... restore other parameters ...
             if load_arrays:
                  base_dir=os.path.dirname(filepath)
                  ep_path_rel=state_data.get("energy_potential_path"); qs_path_rel=state_data.get("quantum_state_path")
                  if not ep_path_rel or not qs_path_rel: raise ValueError("State file missing required array paths.")
                  ep_abs=os.path.join(base_dir,ep_path_rel); qs_abs=os.path.join(base_dir,qs_path_rel)
                  if not os.path.exists(ep_abs): raise FileNotFoundError(f"Energy file missing: {ep_abs}")
                  if not os.path.exists(qs_abs): raise FileNotFoundError(f"Quantum state file missing: {qs_abs}")
                  loaded_ep=np.load(ep_abs); loaded_qs=np.load(qs_abs)
                  if loaded_ep.shape!=self.dimensions: raise ValueError(f"Shape mismatch energy: {loaded_ep.shape} vs {self.dimensions}")
                  if loaded_qs.shape!=self.dimensions: raise ValueError(f"Shape mismatch quantum: {loaded_qs.shape} vs {self.dimensions}")
                  self.energy_potential=loaded_ep.astype(DEFAULT_FIELD_DTYPE)
                  self.quantum_state=loaded_qs.astype(DEFAULT_COMPLEX_DTYPE)
             logger.info(f"Loaded VoidField3D state from {filepath}"); return True
        except Exception as e: logger.error(f"Failed to load VoidField3D state: {e}"); raise RuntimeError("State loading failed") from e

    def _save_array(self, array: np.ndarray, name_prefix: str) -> str:
        """Helper to save array. Raises IOError on failure."""
        # (Implementation from previous step)
        array_filename = f"{name_prefix}.npy"
        array_filepath = os.path.join(self.data_dir, array_filename)
        try: np.save(array_filepath, array); logger.debug(f"Saved array '{name_prefix}'"); return os.path.basename(array_filepath)
        except Exception as e: logger.error(f"Failed to save array '{name_prefix}': {e}"); raise IOError(f"Failed to save array {name_prefix}") from e

    # --- Keep __str__ and __repr__ ---
    def __str__(self):
         return f"VoidField3D(name='{self.field_name}', dims={self.dimensions}, wells={len(self.potential_wells)})"
    def __repr__(self):
         return f"<VoidField3D name='{self.field_name}' uuid='{self.uuid}'>"





