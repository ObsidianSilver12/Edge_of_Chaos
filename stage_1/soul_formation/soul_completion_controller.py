# --- START OF FILE src/stage_1/soul_completion_controller.py ---

"""
Soul Completion Controller (Refactored V4.3.10 - Full Soul Formation & Corrected Constant Usage)

Orchestrates the complete soul formation process from Spark Harmonization through Birth.
Handles all stage functions directly, including Spark Harmonization, Guff Strengthening,
Sephiroth Journey, Creator Entanglement, Harmonic Strengthening, Life Cord, Earth
Harmonization, Identity Crystallization, and Birth.
Works with refactored stage functions that use wave physics and layer-based approaches.
Uses constants directly after wildcard import.
"""

import logging
import os
import sys
import uuid
import json
import time
import random
from datetime import datetime, timedelta # Added timedelta
from typing import Optional, Tuple, Dict, Any, List # type: ignore
# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants (Wildcard Import) ---
try:
    from constants.constants import *
    logger.setLevel(LOG_LEVEL) # Set logger level after successful import
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: constants.py failed import in soul_completion_controller.py: {e}")
    # # Define critical fallbacks if constants.py is missing
    # DATA_DIR_BASE = "output_fallback"
    # LOG_LEVEL = "INFO"
    # GRID_SIZE = (64,64,64)
    # FLOAT_EPSILON = 1e-9
    # PHI = (1 + 5**0.5) / 2
    # GUFF_STRENGTHENING_DURATION = 10.0
    # HARMONIC_STRENGTHENING_INTENSITY_DEFAULT=0.7
    # HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT=1.0
    # LIFE_CORD_COMPLEXITY_DEFAULT=0.7
    # EARTH_HARMONY_INTENSITY_DEFAULT=0.7 # Placeholder if this constant name changed
    # EARTH_HARMONY_DURATION_FACTOR_DEFAULT=1.0 # Placeholder
    # BIRTH_INTENSITY_DEFAULT=0.7
    # IDENTITY_CRYSTALLIZATION_THRESHOLD = 0.85
    # MAX_STABILITY_SU = 100.0
    # MAX_COHERENCE_CU = 100.0
    # # Define all FLAG constants as strings
    # FLAG_READY_FOR_BIRTH = 'ready_for_birth'
    # FLAG_READY_FOR_HARMONIZATION = 'ready_for_harmonization'
    # FLAG_READY_FOR_ENTANGLEMENT = 'ready_for_entanglement'
    # FLAG_SEPHIROTH_JOURNEY_COMPLETE = 'sephiroth_journey_complete'
    # FLAG_CREATOR_ENTANGLED = 'creator_entangled'
    # FLAG_HARMONICALLY_STRENGTHENED = 'harmonically_strengthened'
    # FLAG_READY_FOR_LIFE_CORD = 'ready_for_life_cord'
    # FLAG_CORD_FORMATION_COMPLETE = 'cord_formation_complete'
    # FLAG_READY_FOR_EARTH = 'ready_for_earth'
    # FLAG_EARTH_ATTUNED = 'earth_attuned'
    # FLAG_ECHO_PROJECTED = 'echo_projected'
    # FLAG_IDENTITY_CRYSTALLIZED = 'identity_crystallized'
    # FLAG_INCARNATED = 'incarnated'
    # Add other essential constants if the import fails catastrophically
    logger.warning(f"Constants not loaded, using fallback values. LOG_LEVEL set to {LOG_LEVEL}")
    logger.setLevel(getattr(logging, str(LOG_LEVEL).upper(), logging.INFO))


# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.fields.field_controller import FieldController
    from stage_1.fields.sephiroth_field import SephirothField

    from stage_1.soul_formation.spark_harmonization import perform_spark_harmonization
    from stage_1.soul_formation.guff_strengthening import perform_guff_strengthening
    from stage_1.soul_formation.sephiroth_journey_processing import process_sephirah_interaction
    from stage_1.soul_formation.creator_entanglement import perform_creator_entanglement
    from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening
    from stage_1.soul_formation.life_cord import form_life_cord
    from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization
    from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization
    from stage_1.soul_formation.birth import perform_birth

    try:
        import metrics_tracking as metrics
        METRICS_AVAILABLE = True
    except ImportError:
        logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
        METRICS_AVAILABLE = False
        class MetricsPlaceholder:
            def record_metrics(self, *args, **kwargs): pass
            def record_metric(self, *args, **kwargs): pass
            def persist_metrics(self, *args, **kwargs): pass
        metrics = MetricsPlaceholder()
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import stage modules/SoulSpark: {e}", exc_info=True)
    raise ImportError(f"Core stage dependencies missing: {e}") from e

# --- Visualization Import & Setup ---
VISUALIZATION_ENABLED = False # Default
VISUALIZATION_OUTPUT_DIR = os.path.join(DATA_DIR_BASE, "visuals", "soul_completion") # Consistent path
try:
    from stage_1.soul_formation.soul_visualizer import (
        visualize_soul_state,
        visualize_state_comparison
    )
    VISUALIZATION_ENABLED = True
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR_BASE, "completed_souls"), exist_ok=True) # For final soul data
    logger.info("Soul visualization module loaded successfully.")
except ImportError as ie:
    logger.warning(f"Soul visualization module not found: {ie}. Visualizations disabled.")
except Exception as e:
    logger.warning(f"Error setting up visualization: {e}. Visualizations disabled.")

# --- Mother Resonance Import ---
MOTHER_RESONANCE_AVAILABLE = False # Default
try:
    from glyphs.mother.mother_resonance import create_mother_resonance_data
    MOTHER_RESONANCE_AVAILABLE = True
    logger.info("Mother resonance module loaded successfully.")
except ImportError:
    logger.warning("Mother resonance module not found. Birth will proceed without mother influence or require explicit profile.")

# --- Helper Function for Metrics Display (Copied from root_controller) ---
def display_stage_metrics(stage_name, metrics_dict):
    """Prints a formatted summary of stage metrics."""
    skip_keys = {
        'success', 'error', 'failed_step', 'action', 'soul_id', 'start_time',
        'end_time', 'timestamp', 'duration_seconds', 'initial_state',
        'final_state', 'guff_properties_used', 'imparted_aspect_strengths',
        'aspects_touched_names', 'initial_state_changes', 'geometric_changes',
        'local_entanglement_changes', 'element_details', 'cycle_details',
        'components', 'missing_attributes', 'gained_aspect_names',
        'strengthened_aspect_names', 'transfers', 'memory_retentions',
        'layer_formation_changes', 'imparted_aspect_strengths_summary',
        'detailed_metrics', 'guff_targets_used', 'initial_stability_su',
        'initial_coherence_cu', 'initial_energy_seu',
        'initial_pattern_coherence', 'initial_phi_resonance',
        'initial_harmony', 'initial_toroidal_flow',
        'peak_stability_su_during_stabilization',
        'peak_coherence_cu_during_stabilization', 'step_metrics'
    }
    print(f"\n{'='*20} {stage_name} Metrics Summary {'='*20}")
    if not isinstance(metrics_dict, dict):
        print("  Invalid metrics format (not a dict).")
        print("=" * (40 + len(stage_name)))
        return
    if not metrics_dict:
        print("  No metrics captured.")
        print("=" * (40 + len(stage_name)))
        return

    success = metrics_dict.get('success')
    if success is not None: print(f"  Success: {success}")
    else:
        print("  Success: Unknown (key missing)")
        logger.warning(f"Metrics for '{stage_name}' missing 'success' key.")

    if not success and success is not None:
        print(f"  Error: {metrics_dict.get('error', 'Unknown')}")
        print(f"  Failed Step: {metrics_dict.get('failed_step', 'N/A')}")

    display_keys = sorted([
        str(k) for k in metrics_dict.keys() if str(k) not in skip_keys
    ])

    if not display_keys and success is not False:
        print("  (No specific metrics to display)")

    max_key_len = max(len(k.replace('_',' ').title()) for k in display_keys) if display_keys else 30

    for key in display_keys:
        value = metrics_dict[key]
        unit = ""
        key_display = key
        key_lower = key.lower()
        if key_lower.endswith('_seu'): unit=" SEU"; key_display=key[:-4]
        elif key_lower.endswith('_su'): unit=" SU"; key_display=key[:-3]
        elif key_lower.endswith('_cu'): unit=" CU"; key_display=key[:-3]
        elif key_lower.endswith('_hz'): unit=" Hz"; key_display=key[:-3]
        elif key_lower.endswith('_pct'): unit="%"; key_display=key[:-4]
        elif key_lower.endswith('_factor'): key_display=key[:-7]
        elif key_lower.endswith('_score'): key_display=key[:-6]
        elif key_lower.endswith('_level'): key_display=key[:-6]
        elif key_lower.endswith('_count'): key_display=key[:-6]
        elif key_lower.endswith('_ratio'): key_display=key[:-6]
        elif key_lower.endswith('_strength'): key_display=key[:-9]
        elif key_lower.endswith('_coherence'): key_display=key[:-10]
        elif key_lower.endswith('_resonance'): key_display=key[:-10]
        elif key_lower.endswith('_alignment'): key_display=key[:-10]
        elif key_lower.endswith('_integration'): key_display=key[:-12]

        if isinstance(value, float):
            if unit in [" SU", " CU", " Hz"]: formatted_val = f"{value:.1f}"
            elif unit == " SEU": formatted_val = f"{value:.2f}"
            elif unit == "%": formatted_val = f"{value:.1f}"
            elif '_gain' in key_lower: formatted_val = f"{value:+.2f}"
            elif any(sfx in key_lower for sfx in ['_factor','_score','_level','_resonance','_strength','_change','_integrity','_coherence','_ratio','_alignment','_integration']):
                formatted_val = f"{value:.3f}"
            else: formatted_val = f"{value:.3f}"
        elif isinstance(value, int): formatted_val = str(value)
        elif isinstance(value, bool): formatted_val = str(value)
        elif isinstance(value, list): formatted_val = f"<List ({len(value)} items)>"
        elif isinstance(value, dict): formatted_val = f"<Dict ({len(value)} keys)>"
        else: formatted_val = str(value)

        key_display_cleaned = key_display.replace('_', ' ').title()
        print(f"  {key_display_cleaned:<{max_key_len}} : {formatted_val}{unit}")
    print("=" * (40 + len(stage_name)))


CONTROLLER_METRIC_CATEGORY = "soul_completion_controller"

class SoulCompletionController:
    """
    Orchestrates complete soul formation from Spark Harmonization to Birth.
    """

    def __init__(self, data_dir: str = DATA_DIR_BASE, field_controller: Optional[FieldController] = None,
                 controller_id: Optional[str] = None, visualization_enabled: bool = VISUALIZATION_ENABLED):
        if not data_dir or not isinstance(data_dir, str): raise ValueError("Data directory invalid.")
        self.controller_id: str = controller_id or str(uuid.uuid4())
        self.creation_time: str = datetime.now().isoformat()
        self.output_dir: str = os.path.join(data_dir, "controller_data", f"soul_completion_{self.controller_id}")
        self.visualization_enabled = visualization_enabled
        # Use the global VISUALIZATION_OUTPUT_DIR for consistency
        self.visual_save_dir = VISUALIZATION_OUTPUT_DIR
        self.field_controller = field_controller
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.visualization_enabled:
                os.makedirs(self.visual_save_dir, exist_ok=True)
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create output dir {self.output_dir}: {e}")
            raise
        self.active_souls: Dict[str, Dict[str, Any]] = {}
        self.development_states: Dict[str, List[Tuple[SoulSpark, str]]] = {}
        logger.info(f"Initializing Soul Completion Controller (ID: {self.controller_id})")
        if METRICS_AVAILABLE:
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'status': 'initialized', 'controller_id': self.controller_id,
                'timestamp': self.creation_time, 'visualization_enabled': self.visualization_enabled
            })
        logger.info(f"Soul Completion Controller '{self.controller_id}' initialized.")

    def _run_stage(self, stage_func: callable, soul_spark: SoulSpark, stage_name_readable: str,
                   show_visuals: bool, pre_stage_vis_name: str, post_stage_vis_name: str,
                   **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        """Helper to run a generic stage with visualization and error handling."""
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Stage: {stage_name_readable} for {spark_id}...")
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name_readable, 'start_time': datetime.now().isoformat()}

        if self.visualization_enabled:
            if spark_id not in self.development_states: self.development_states[spark_id] = []
            try:
                self.development_states[spark_id].append((soul_spark, pre_stage_vis_name))
                visualize_soul_state(soul_spark, pre_stage_vis_name, self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"{pre_stage_vis_name} visualization failed: {vis_err}")

        try:
            _, stage_metrics = stage_func(soul_spark, **kwargs)

            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, post_stage_vis_name))
                    visualize_soul_state(soul_spark, post_stage_vis_name, self.visual_save_dir, show=show_visuals)
                except Exception as vis_err: logger.warning(f"{post_stage_vis_name} visualization failed: {vis_err}")

            display_stage_metrics(stage_name_readable, stage_metrics)
            logger.info(f"{stage_name_readable} Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            return soul_spark, stage_metrics
        except Exception as e:
            logger.error(f"{stage_name_readable} failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['error'] = str(e)
            raise RuntimeError(f"{stage_name_readable} failed: {e}") from e


    def run_spark_harmonization(self, soul_spark: SoulSpark, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        return self._run_stage(perform_spark_harmonization, soul_spark, "Spark Harmonization",
                               show_visuals, "Pre_Harmonization", "Post_Harmonization")

    def run_guff_strengthening(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
                               duration: float = GUFF_STRENGTHENING_DURATION, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        field_ctrl = field_controller or self.field_controller
        if not field_ctrl: raise ValueError("Field controller required for Guff strengthening.")
        try:
            current_field = getattr(soul_spark, 'current_field_key', None)
            if current_field != 'guff' and current_field != 'kether': # Allow if already in Kether but not yet Guff
                logger.info(f"Moving soul {soul_spark.spark_id} to Guff for strengthening...")
                field_ctrl.place_soul_in_guff(soul_spark)
        except Exception as move_err: raise RuntimeError(f"Failed to move soul to Guff: {move_err}") from move_err

        stage_result = self._run_stage(
            perform_guff_strengthening, soul_spark, "Guff Strengthening", show_visuals,
            "Pre_Guff_Strengthening", "Post_Guff_Strengthening",
            field_controller=field_ctrl, duration=duration
        )
        # Release soul from Guff after strengthening
        try: field_ctrl.release_soul_from_guff(soul_spark)
        except Exception as release_err: raise RuntimeError(f"Failed to release soul from Guff: {release_err}") from release_err
        return stage_result

    def run_sephiroth_journey(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
                              journey_duration_per_sephirah: float = 2.0, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        stage_name = "Sephiroth Journey"
        field_ctrl = field_controller or self.field_controller
        if not field_ctrl: raise ValueError("Field controller required for Sephiroth journey.")
        logger.info(f"Stage: {stage_name} for {spark_id}...")
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
        if self.visualization_enabled:
            if spark_id not in self.development_states: self.development_states[spark_id] = []
            try:
                self.development_states[spark_id].append((soul_spark, "Pre_Sephiroth_Journey"))
                visualize_soul_state(soul_spark, "Pre_Sephiroth_Journey", self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"Pre-Journey visualization failed: {vis_err}")
        try:
            journey_path = ["kether", "chokmah", "binah", "daath", "chesed",
                           "geburah", "tiphareth", "netzach", "hod", "yesod", "malkuth"]
            journey_step_metrics: Dict[str, Any] = {}
            for sephirah_name_iter in journey_path: # Renamed variable
                stage_id = f"Interaction ({sephirah_name_iter.capitalize()})"
                logger.info(f"  Entering {sephirah_name_iter.capitalize()}...")
                sephirah_influencer = field_ctrl.get_field(sephirah_name_iter)
                if not sephirah_influencer or not isinstance(sephirah_influencer, SephirothField):
                    raise RuntimeError(f"SephirothField missing for '{sephirah_name_iter}'.")
                _, step_metrics = process_sephirah_interaction(
                    soul_spark, sephirah_influencer, field_ctrl, journey_duration_per_sephirah
                )
                journey_step_metrics[sephirah_name_iter] = step_metrics
                display_stage_metrics(stage_id, step_metrics)
                logger.info(f"  Exiting {sephirah_name_iter.capitalize()}.")
                if self.visualization_enabled and sephirah_name_iter in ["kether", "tiphareth", "malkuth"]:
                    try: visualize_soul_state(soul_spark, f"Sephiroth_{sephirah_name_iter.capitalize()}", self.visual_save_dir, show=show_visuals)
                    except Exception as vis_err: logger.warning(f"Sephiroth {sephirah_name_iter} visualization failed: {vis_err}")
            setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, True)
            setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, True)
            journey_metrics = {'steps': journey_step_metrics, 'soul_id': spark_id, 'journey_path': journey_path,
                               'journey_duration_per_sephirah': journey_duration_per_sephirah,
                               'total_journey_duration': journey_duration_per_sephirah * len(journey_path), 'success': True}
            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, "Post_Sephiroth_Journey"))
                    visualize_soul_state(soul_spark, "Post_Sephiroth_Journey", self.visual_save_dir, show=show_visuals)
                except Exception as vis_err: logger.warning(f"Post-Journey visualization failed: {vis_err}")
            logger.info(f"{stage_name} Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            return soul_spark, journey_metrics
        except Exception as e:
            logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status']='failed'; self.active_souls[spark_id]['error']=str(e)
            setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False); setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, False)
            raise RuntimeError(f"{stage_name} failed: {e}") from e

    def run_creator_entanglement(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
                                show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        field_ctrl = field_controller or self.field_controller
        if not field_ctrl: raise ValueError("Field controller required for Creator entanglement.")
        if not getattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False): raise ValueError("Sephiroth journey not complete.")
        if not getattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, False): raise ValueError(f"Missing {FLAG_READY_FOR_ENTANGLEMENT} flag.")
        kether_influencer = field_ctrl.kether_field
        if not kether_influencer: raise RuntimeError("Kether field unavailable.")
        stage_result = self._run_stage(
            perform_creator_entanglement, soul_spark, "Creator Entanglement", show_visuals,
            "Pre_Creator_Entanglement", "Post_Creator_Entanglement", kether_field=kether_influencer
        )
        setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, True)
        return stage_result

    def run_harmonic_strengthening(self, soul_spark: SoulSpark, intensity: float = HARMONIC_STRENGTHENING_INTENSITY_DEFAULT,
                                  duration_factor: float = HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT,
                                  show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not getattr(soul_spark, FLAG_CREATOR_ENTANGLED, False): raise ValueError("Creator entanglement not complete.")
        if not getattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, False): raise ValueError(f"Missing {FLAG_READY_FOR_HARMONIZATION} flag.")
        stage_result = self._run_stage(
            perform_harmonic_strengthening, soul_spark, "Harmonic Strengthening", show_visuals,
            "Pre_Harmonic_Strengthening", "Post_Harmonic_Strengthening",
            intensity=intensity, duration_factor=duration_factor
        )
        setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, True)
        return stage_result

    def run_life_cord_formation(self, soul_spark: SoulSpark, intensity: float = 0.7, # Default as in root_controller
                            complexity: float = LIFE_CORD_COMPLEXITY_DEFAULT,
                            show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False): raise ValueError(f"Missing {FLAG_READY_FOR_LIFE_CORD} flag.")
        return self._run_stage(
            form_life_cord, soul_spark, "Life Cord Formation", show_visuals,
            "Pre_Life_Cord", "Post_Life_Cord",
            intensity=intensity, complexity=complexity
        )

    def run_earth_harmonization(self, soul_spark: SoulSpark, schumann_intensity: float = 0.7, # Default as in root_controller
                            core_intensity: float = 0.7, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False): raise ValueError("Life cord not formed.")
        if not getattr(soul_spark, FLAG_READY_FOR_EARTH, False): raise ValueError(f"Missing {FLAG_READY_FOR_EARTH} flag.")
        return self._run_stage(
            perform_earth_harmonization, soul_spark, "Earth Harmonization", show_visuals,
            "Pre_Earth_Harmonization", "Post_Earth_Harmonization",
            schumann_intensity=schumann_intensity, core_intensity=core_intensity
        )

    def run_identity_crystallization(self, soul_spark: SoulSpark, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not getattr(soul_spark, FLAG_EARTH_ATTUNED, False): raise ValueError("Not harmonized with Earth.")
        # Default kwargs for identity crystallization if not provided
        id_kwargs = {
            'train_cycles': kwargs.get('train_cycles', 5),
            'entrainment_bpm': kwargs.get('entrainment_bpm', 68.0),
            'entrainment_duration': kwargs.get('entrainment_duration', 60.0),
            'love_cycles': kwargs.get('love_cycles', 3),
            'geometry_stages': kwargs.get('geometry_stages', 3),
            'crystallization_threshold': kwargs.get('crystallization_threshold', IDENTITY_CRYSTALLIZATION_THRESHOLD)
        }
        if 'specified_name' in kwargs: id_kwargs['specified_name'] = kwargs['specified_name']

        return self._run_stage(
            perform_identity_crystallization, soul_spark, "Identity Crystallization", show_visuals,
            "Pre_Identity_Crystallization", "Post_Identity_Crystallization",
            **id_kwargs
        )

    def run_birth_process(self, soul_spark: SoulSpark, intensity: float = BIRTH_INTENSITY_DEFAULT,
                    mother_profile: Optional[Dict[str, Any]] = None, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False): raise ValueError("Identity not crystallized.")
        if not getattr(soul_spark, FLAG_READY_FOR_BIRTH, False): raise ValueError(f"Missing {FLAG_READY_FOR_BIRTH} flag.")

        # Create/get mother profile if not provided and module available
        final_mother_profile = mother_profile
        if not final_mother_profile and MOTHER_RESONANCE_AVAILABLE:
            try:
                mother_resonance_data = create_mother_resonance_data()
                final_mother_profile = {
                    'nurturing_capacity': mother_resonance_data.get('nurturing_capacity', 0.7),
                    'spiritual': mother_resonance_data.get('spiritual', {'connection': 0.6}),
                    'love_resonance': mother_resonance_data.get('love_resonance', 0.7)
                }
            except Exception as mother_err:
                logger.warning(f"Failed to create mother resonance profile for birth: {mother_err}")
                # Proceed without it if creation fails, birth function might handle None

        stage_result = self._run_stage(
            perform_birth, soul_spark, "Birth", show_visuals,
            "Pre_Birth", "Post_Birth",
            intensity=intensity, mother_profile=final_mother_profile
        )
        # Create final development comparison visualization
        if self.visualization_enabled and soul_spark.spark_id in self.development_states:
            try:
                compare_path = visualize_state_comparison(
                    self.development_states[soul_spark.spark_id], self.visual_save_dir, show=show_visuals
                )
                logger.info(f"Development comparison created: {compare_path}")
            except Exception as comp_err: logger.warning(f"Failed to create development comparison: {comp_err}")
        return stage_result


    def run_soul_completion(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
                        journey_duration_per_sephirah: float = 2.0, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        if spark_id in self.active_souls and self.active_souls[spark_id]['status'] == 'processing':
            raise RuntimeError(f"Soul {spark_id} is already being processed.")

        logger.info(f"--- Starting Soul Completion Process for Soul {spark_id} ---")
        start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
        completion_summary = {'soul_id': spark_id, 'stages': {}}
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': None, 'start_time': start_time_iso}

        try:
            field_ctrl = field_controller or self.field_controller
            if not field_ctrl: raise ValueError("Field controller required for soul completion process.")

            # --- Run Stages ---
            _, metrics1 = self.run_spark_harmonization(soul_spark, show_visuals=show_visuals)
            completion_summary['stages']['Spark Harmonization'] = metrics1

            guff_duration = kwargs.get('guff_duration', GUFF_STRENGTHENING_DURATION)
            _, metrics2 = self.run_guff_strengthening(soul_spark, field_ctrl, duration=guff_duration, show_visuals=show_visuals)
            completion_summary['stages']['Guff Strengthening'] = metrics2

            _, metrics3 = self.run_sephiroth_journey(soul_spark, field_ctrl, journey_duration_per_sephirah=journey_duration_per_sephirah, show_visuals=show_visuals)
            completion_summary['stages']['Sephiroth Journey'] = metrics3

            _, metrics4 = self.run_creator_entanglement(soul_spark, field_ctrl, show_visuals=show_visuals)
            completion_summary['stages']['Creator Entanglement'] = metrics4

            harmony_intensity = kwargs.get('harmony_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT)
            harmony_duration = kwargs.get('harmony_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT)
            _, metrics5 = self.run_harmonic_strengthening(soul_spark, intensity=harmony_intensity, duration_factor=harmony_duration, show_visuals=show_visuals)
            completion_summary['stages']['Harmonic Strengthening'] = metrics5

            cord_intensity = kwargs.get('life_cord_intensity', 0.7) # Default as per root_controller if not specified
            cord_complexity = kwargs.get('cord_complexity', LIFE_CORD_COMPLEXITY_DEFAULT)
            _, metrics6 = self.run_life_cord_formation(soul_spark, intensity=cord_intensity, complexity=cord_complexity, show_visuals=show_visuals)
            completion_summary['stages']['Life Cord Formation'] = metrics6

            schumann_intensity_val = kwargs.get('schumann_intensity', EARTH_HARMONY_INTENSITY_DEFAULT) # Renamed variable
            core_intensity_val = kwargs.get('core_intensity', EARTH_HARMONY_INTENSITY_DEFAULT) # Renamed variable
            _, metrics7 = self.run_earth_harmonization(soul_spark, schumann_intensity=schumann_intensity_val, core_intensity=core_intensity_val, show_visuals=show_visuals)
            completion_summary['stages']['Earth Harmonization'] = metrics7

            id_kwargs = {
                'specified_name': kwargs.get('specified_name'),
                'train_cycles': kwargs.get('train_cycles', 5),
                'entrainment_bpm': kwargs.get('entrainment_bpm', 68.0),
                'entrainment_duration': kwargs.get('entrainment_duration', 60.0),
                'love_cycles': kwargs.get('love_cycles', 3),
                'geometry_stages': kwargs.get('geometry_stages', 3),
                'crystallization_threshold': kwargs.get('crystallization_threshold', IDENTITY_CRYSTALLIZATION_THRESHOLD),
            }
            _, metrics8 = self.run_identity_crystallization(soul_spark, show_visuals=show_visuals, **id_kwargs)
            completion_summary['stages']['Identity Crystallization'] = metrics8

            birth_intensity_val = kwargs.get('birth_intensity', BIRTH_INTENSITY_DEFAULT) # Renamed variable
            mother_profile_val = kwargs.get('mother_profile') # Renamed variable
            _, metrics9 = self.run_birth_process(soul_spark, intensity=birth_intensity_val, mother_profile=mother_profile_val, show_visuals=show_visuals)
            completion_summary['stages']['Birth'] = metrics9

            # --- Finalization ---
            end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
            completion_summary['start_time'] = start_time_iso; completion_summary['end_time'] = end_time_iso
            completion_summary['duration_seconds'] = (end_time_dt - start_time_dt).total_seconds()
            completion_summary['success'] = True
            completion_summary['final_soul_state_summary'] = soul_spark.get_spark_metrics()['core']

            self.active_souls[spark_id]['status'] = 'completed'; self.active_souls[spark_id]['end_time'] = end_time_iso
            self.active_souls[spark_id]['current_stage'] = None
            self.active_souls[spark_id]['summary'] = {k: v for k, v in completion_summary.items() if k != 'stages'}
            self._save_completed_soul(soul_spark)

            overall_controller_metrics = {
                'controller_run': 'soul_completion', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': end_time_iso,
                'duration_seconds': completion_summary['duration_seconds'], 'success': True,
                'final_energy_seu': soul_spark.energy, 'final_stability_su': soul_spark.stability,
                'final_coherence_cu': soul_spark.coherence,
                'final_incarnated_status': getattr(soul_spark, FLAG_INCARNATED, False)
            }
            if METRICS_AVAILABLE: metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, overall_controller_metrics)
            logger.info(f"--- Soul Completion Process Finished Successfully for Soul {spark_id} ---")
            logger.info(f"Duration: {completion_summary['duration_seconds']:.2f}s")
            logger.info(f"Final Incarnated Status: {getattr(soul_spark, FLAG_INCARNATED, False)}")
            return soul_spark, overall_controller_metrics
        except Exception as e:
            failed_stage_name = self.active_souls[spark_id].get('current_stage', 'unknown')
            end_time_iso = datetime.now().isoformat()
            logger.error(f"Soul completion failed at stage '{failed_stage_name}': {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['end_time'] = end_time_iso
            self.active_souls[spark_id]['error'] = str(e)
            setattr(soul_spark, FLAG_INCARNATED, False)
            if METRICS_AVAILABLE:
                metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                    'controller_run': 'soul_completion', 'soul_id': spark_id,
                    'start_time': start_time_iso, 'end_time': end_time_iso,
                    'duration_seconds': (datetime.fromisoformat(end_time_iso) - start_time_dt).total_seconds(),
                    'success': False, 'error': str(e), 'failed_stage': failed_stage_name })
            raise RuntimeError(f"Soul completion process failed at stage '{failed_stage_name}': {e}") from e

    def _save_completed_soul(self, soul_spark: SoulSpark) -> bool:
        """Saves the final state of the completed soul."""
        spark_id = getattr(soul_spark, 'spark_id', None)
        if not spark_id: return False
        # Use a subdirectory within DATA_DIR_BASE for completed souls
        completed_souls_dir = os.path.join(DATA_DIR_BASE, "completed_souls")
        os.makedirs(completed_souls_dir, exist_ok=True)
        filename = f"soul_completed_{spark_id}.json"
        save_path = os.path.join(completed_souls_dir, filename)

        logger.info(f"Saving completed soul data for {spark_id} to {save_path}...")
        try:
            if hasattr(soul_spark, 'save_spark_data'):
                soul_spark.save_spark_data(save_path)
            else:
                # Fallback serialization if save_spark_data is missing
                soul_dict = {k: v for k, v in soul_spark.__dict__.items()
                             if not k.startswith('_') and not callable(v)}
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(soul_dict, f, indent=2, default=str) # Use default=str for broader compatibility

            logger.info(f"Soul {spark_id} saved successfully to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save completed soul {spark_id} to {save_path}: {e}", exc_info=True)
            return False


    def get_active_souls_status(self) -> Dict[str, Dict[str, Any]]:
        """Returns the status of all active souls being processed."""
        return {k: v.copy() for k, v in self.active_souls.items()}

    def get_soul_status(self, soul_id: str) -> Optional[Dict[str, Any]]:
        """Returns the status of a specific soul if it exists."""
        return self.active_souls.get(soul_id, {}).copy() if soul_id in self.active_souls else None

    def get_visualization_paths(self, soul_id: str) -> List[str]:
        """Returns paths to visualizations for a specific soul."""
        if not self.visualization_enabled or soul_id not in self.development_states:
            return []
        visual_files = []
        try:
            if os.path.exists(self.visual_save_dir):
                for filename in os.listdir(self.visual_save_dir):
                    if soul_id in filename and (filename.endswith('.png') or filename.endswith('.jpg')):
                        visual_files.append(os.path.join(self.visual_save_dir, filename))
        except Exception as e:
            logger.warning(f"Error getting visualization paths for {soul_id}: {e}")
        return sorted(visual_files)

    def cleanup_resources(self):
        """Cleans up any resources used by the controller."""
        try:
            self.active_souls.clear()
            if METRICS_AVAILABLE and hasattr(metrics, 'persist_metrics'):
                metrics.persist_metrics()
            shutdown_time = datetime.now().isoformat()
            logger.info(f"Soul Completion Controller '{self.controller_id}' shutting down at {shutdown_time}")
            return True
        except Exception as e:
            logger.error(f"Error during controller cleanup: {e}", exc_info=True)
            return False

    def __del__(self):
        """Destructor to ensure cleanup when object is deleted."""
        try: self.cleanup_resources()
        except: pass # Suppress errors in destructor

# --- END OF FILE src/stage_1/soul_completion_controller.py ---















# # --- START OF FILE src/stage_1/soul_completion_controller.py ---

# """
# Soul Completion Controller (Refactored V4.3.9 - Full Soul Formation)

# Orchestrates the complete soul formation process from Spark Harmonization through Birth.
# Handles all stage functions directly, including Spark Harmonization, Guff Strengthening,
# Sephiroth Journey, Creator Entanglement, Harmonic Strengthening, Life Cord, Earth 
# Harmonization, Identity Crystallization, and Birth.
# Works with refactored stage functions that use wave physics and layer-based approaches.
# """

# import logging
# import os
# import sys
# import uuid
# import json
# import time
# import random
# from datetime import datetime
# from typing import Optional, Tuple, Dict, Any, List
# from constants.constants import *
# # --- Logging ---
# logger = logging.getLogger(__name__)

# # --- Constants ---

#     # DATA_DIR_BASE = "output" # Need at least this
#     # MAX_STABILITY_SU = 100.0; MAX_COHERENCE_CU = 100.0
#     # GUFF_STRENGTHENING_DURATION = 10.0
#     # HARMONIC_STRENGTHENING_INTENSITY_DEFAULT=0.7; HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT=1.0
#     # LIFE_CORD_COMPLEXITY_DEFAULT=0.7; EARTH_HARMONY_INTENSITY_DEFAULT=0.7; EARTH_HARMONY_DURATION_FACTOR_DEFAULT=1.0
#     # BIRTH_INTENSITY_DEFAULT=0.7; IDENTITY_CRYSTALLIZATION_THRESHOLD = 0.85
#     # # Flags
#     # FLAG_READY_FOR_BIRTH = 'ready_for_birth'
#     # FLAG_READY_FOR_HARMONIZATION = 'ready_for_harmonization'
#     # FLAG_READY_FOR_ENTANGLEMENT = 'ready_for_entanglement'
#     # FLAG_SEPHIROTH_JOURNEY_COMPLETE = 'sephiroth_journey_complete'
#     # FLAG_CREATOR_ENTANGLED = 'creator_entangled'
#     # FLAG_HARMONICALLY_STRENGTHENED = 'harmonically_strengthened'
#     # FLAG_READY_FOR_LIFE_CORD = 'ready_for_life_cord'
#     # FLAG_CORD_FORMATION_COMPLETE = 'cord_formation_complete'
#     # FLAG_READY_FOR_EARTH = 'ready_for_earth'
#     # FLAG_EARTH_ATTUNED = 'earth_attuned'
#     # FLAG_ECHO_PROJECTED = 'echo_projected'
#     # FLAG_IDENTITY_CRYSTALLIZED = 'identity_crystallized'
#     # FLAG_INCARNATED = 'incarnated'


# # --- Dependency Imports ---
# try:
#     from stage_1.soul_spark.soul_spark import SoulSpark
#     from stage_1.fields.field_controller import FieldController
#     from stage_1.fields.sephiroth_field import SephirothField
    
#     # Import all stage functions
#     from stage_1.soul_formation.spark_harmonization import perform_spark_harmonization
#     from stage_1.soul_formation.guff_strengthening import perform_guff_strengthening
#     from stage_1.soul_formation.sephiroth_journey_processing import process_sephirah_interaction
#     from stage_1.soul_formation.creator_entanglement import perform_creator_entanglement
#     from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening
#     from stage_1.soul_formation.life_cord import form_life_cord
#     from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization
#     from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization
#     from stage_1.soul_formation.birth import perform_birth
    
#     # Import metrics if available
#     try:
#         import metrics_tracking as metrics
#         METRICS_AVAILABLE = True
#     except ImportError:
#         logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
#         METRICS_AVAILABLE = False
#         class MetricsPlaceholder:
#             def record_metrics(self, *args, **kwargs): pass
#             def record_metric(self, *args, **kwargs): pass
#             def persist_metrics(self, *args, **kwargs): pass
#         metrics = MetricsPlaceholder()
# except ImportError as e:
#     logger.critical(f"CRITICAL ERROR: Failed to import stage modules/SoulSpark: {e}", exc_info=True)
#     raise ImportError(f"Core stage dependencies missing: {e}") from e

# # --- Visualization Import & Setup ---
# VISUALIZATION_ENABLED = False
# VISUALIZATION_OUTPUT_DIR = os.path.join("output", "visuals")
# try:
#     # Import visualization functions
#     from stage_1.soul_formation.soul_visualizer import (
#         visualize_soul_state, 
#         visualize_state_comparison
#     )
#     VISUALIZATION_ENABLED = True
    
#     # Ensure visualization directories exist
#     os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
#     os.makedirs(os.path.join("output", "completed"), exist_ok=True)
    
#     logger.info("Soul visualization module loaded successfully.")
# except ImportError as ie:
#     logger.warning(f"Soul visualization module not found: {ie}. Visualizations disabled.")
# except Exception as e:
#     logger.warning(f"Error setting up visualization: {e}. Visualizations disabled.")

# # --- Mother Resonance Import ---
# try:
#     # Mother resonance import
#     from glyphs.mother.mother_resonance import create_mother_resonance_data
#     MOTHER_RESONANCE_AVAILABLE = True
#     logger.info("Mother resonance module loaded successfully.")
# except ImportError:
#     logger.warning("Mother resonance module not found. Birth will proceed without mother influence.")
#     MOTHER_RESONANCE_AVAILABLE = False

# # --- Helper Function for Metrics Display ---
# def display_stage_metrics(stage_name, metrics_dict):
#     """Prints a formatted summary of stage metrics."""
#     # Define skip keys inside the function
#     skip_keys = {
#         'success', 'error', 'failed_step', 'action', 'soul_id', 'start_time',
#         'end_time', 'timestamp', 'duration_seconds', 'initial_state',
#         'final_state', 'guff_properties_used', 'imparted_aspect_strengths',
#         'aspects_touched_names', 'initial_state_changes', 'geometric_changes',
#         'local_entanglement_changes', 'element_details', 'cycle_details',
#         'components', 'missing_attributes', 'gained_aspect_names',
#         'strengthened_aspect_names', 'transfers', 'memory_retentions',
#         'layer_formation_changes', 'imparted_aspect_strengths_summary',
#         'detailed_metrics', 'guff_targets_used', 'initial_stability_su',
#         'initial_coherence_cu', 'initial_energy_seu',
#         'initial_pattern_coherence', 'initial_phi_resonance',
#         'initial_harmony', 'initial_toroidal_flow',
#         'peak_stability_su_during_stabilization',
#         'peak_coherence_cu_during_stabilization', 'step_metrics'
#     }
#     print(f"\n{'='*20} {stage_name} Metrics Summary {'='*20}")
#     if not isinstance(metrics_dict, dict):
#         print("  Invalid metrics format (not a dict).")
#         print("=" * (40 + len(stage_name)))
#         return
#     if not metrics_dict:
#         print("  No metrics captured.")
#         print("=" * (40 + len(stage_name)))
#         return

#     success = metrics_dict.get('success')
#     if success is not None: print(f"  Success: {success}")
#     else:
#         # Handle missing 'success' key
#         print("  Success: Unknown (key missing)")
#         logger.warning(f"Metrics for '{stage_name}' missing 'success' key.")

#     if not success and success is not None:
#         print(f"  Error: {metrics_dict.get('error', 'Unknown')}")
#         print(f"  Failed Step: {metrics_dict.get('failed_step', 'N/A')}")

#     display_keys = sorted([
#         str(k) for k in metrics_dict.keys() if str(k) not in skip_keys
#     ])

#     if not display_keys and success is not False:
#         print("  (No specific metrics to display)")

#     max_key_len = max(len(k.replace('_',' ').title()) for k in display_keys) if display_keys else 30

#     for key in display_keys:
#         value = metrics_dict[key]
#         unit = ""
#         key_display = key

#         # Unit detection and key cleaning
#         key_lower = key.lower()
#         if key_lower.endswith('_seu'): unit=" SEU"; key_display=key[:-4]
#         elif key_lower.endswith('_su'): unit=" SU"; key_display=key[:-3]
#         elif key_lower.endswith('_cu'): unit=" CU"; key_display=key[:-3]
#         elif key_lower.endswith('_hz'): unit=" Hz"; key_display=key[:-3]
#         elif key_lower.endswith('_pct'): unit="%"; key_display=key[:-4]
#         elif key_lower.endswith('_factor'): key_display=key[:-7]
#         elif key_lower.endswith('_score'): key_display=key[:-6]
#         elif key_lower.endswith('_level'): key_display=key[:-6]
#         elif key_lower.endswith('_count'): key_display=key[:-6]
#         elif key_lower.endswith('_ratio'): key_display=key[:-6]
#         elif key_lower.endswith('_strength'): key_display=key[:-9]
#         elif key_lower.endswith('_coherence'): key_display=key[:-10]
#         elif key_lower.endswith('_resonance'): key_display=key[:-10]
#         elif key_lower.endswith('_alignment'): key_display=key[:-10]
#         elif key_lower.endswith('_integration'): key_display=key[:-12]

#         # Formatting based on type and unit
#         if isinstance(value, float):
#             if unit in [" SU", " CU", " Hz"]: formatted_val = f"{value:.1f}"
#             elif unit == " SEU": formatted_val = f"{value:.2f}"
#             elif unit == "%": formatted_val = f"{value:.1f}"
#             elif '_gain' in key_lower: formatted_val = f"{value:+.2f}"
#             elif any(suffix in key_lower for suffix in ['_factor', '_score',
#                       '_level', '_resonance', '_strength', '_change',
#                       '_integrity', '_coherence', '_ratio', '_alignment',
#                       '_integration']):
#                 formatted_val = f"{value:.3f}"
#             else: formatted_val = f"{value:.3f}"
#         elif isinstance(value, int): formatted_val = str(value)
#         elif isinstance(value, bool): formatted_val = str(value)
#         elif isinstance(value, list): formatted_val = f"<List ({len(value)} items)>"
#         elif isinstance(value, dict): formatted_val = f"<Dict ({len(value)} keys)>"
#         else: formatted_val = str(value)

#         key_display_cleaned = key_display.replace('_', ' ').title()
#         # Pad key display for alignment
#         print(f"  {key_display_cleaned:<{max_key_len}} : {formatted_val}{unit}")

#     print("=" * (40 + len(stage_name)))

# # Controller-specific metric category
# CONTROLLER_METRIC_CATEGORY = "soul_completion_controller"

# class SoulCompletionController:
#     """
#     Orchestrates complete soul formation from Spark Harmonization to Birth.
#     Runs all stage functions directly from their corresponding files.
#     """

#     def __init__(self, data_dir: str = DATA_DIR_BASE, field_controller: Optional[FieldController] = None,
#                  controller_id: Optional[str] = None, visualization_enabled: bool = VISUALIZATION_ENABLED):
#         """ Initialize the Soul Completion Controller. """
#         if not data_dir or not isinstance(data_dir, str): raise ValueError("Data directory invalid.")

#         self.controller_id: str = controller_id or str(uuid.uuid4())
#         self.creation_time: str = datetime.now().isoformat()
#         self.output_dir: str = os.path.join(data_dir, "controller_data", f"soul_completion_{self.controller_id}")
#         self.visualization_enabled = visualization_enabled
#         self.visual_save_dir = os.path.join(data_dir, "visuals")
#         self.field_controller = field_controller  # Can be None if not field operations needed

#         try: 
#             os.makedirs(self.output_dir, exist_ok=True)
#             if self.visualization_enabled:
#                 os.makedirs(self.visual_save_dir, exist_ok=True)
#         except OSError as e: 
#             logger.critical(f"CRITICAL: Failed to create output dir {self.output_dir}: {e}")
#             raise

#         self.active_souls: Dict[str, Dict[str, Any]] = {} # {soul_id: status_dict}
#         # Store development states for visualization comparisons
#         self.development_states: Dict[str, List[Tuple[SoulSpark, str]]] = {}
        
#         logger.info(f"Initializing Soul Completion Controller (ID: {self.controller_id})")
#         if METRICS_AVAILABLE:
#             metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
#                 'status': 'initialized', 
#                 'controller_id': self.controller_id,
#                 'timestamp': self.creation_time,
#                 'visualization_enabled': self.visualization_enabled
#             })
#         logger.info(f"Soul Completion Controller '{self.controller_id}' initialized.")

#     def run_spark_harmonization(self, soul_spark: SoulSpark, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#         """
#         Runs the spark harmonization stage.
        
#         Args:
#             soul_spark: The SoulSpark object to harmonize
#             show_visuals: Whether to display visualizations
            
#         Returns:
#             Tuple of (modified SoulSpark, stage metrics)
#         """
#         if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#         spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#         stage_name = "Spark Harmonization"
        
#         logger.info(f"Stage: {stage_name} for {spark_id}...")
#         self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
        
#         # Initialize development states for visualization
#         if self.visualization_enabled and spark_id not in self.development_states:
#             self.development_states[spark_id] = []
#             try:
#                 # Capture pre-harmonization state
#                 self.development_states[spark_id].append((soul_spark, "Pre_Harmonization"))
#                 visualize_soul_state(soul_spark, "Pre_Harmonization", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Initial visualization failed: {vis_err}")
        
#         try:
#             _, stage_metrics = perform_spark_harmonization(soul_spark)
            
#             # Visualization after harmonization
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Post_Harmonization"))
#                     visualize_soul_state(soul_spark, "Post_Harmonization", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Post-Harmonization visualization failed: {vis_err}")
            
#             display_stage_metrics(stage_name, stage_metrics)
#             logger.info(f"{stage_name} Complete. Stability: {soul_spark.stability:.1f} SU, Coherence: {soul_spark.coherence:.1f} CU")
            
#             return soul_spark, stage_metrics
            
#         except Exception as e:
#             logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#             self.active_souls[spark_id]['status'] = 'failed'
#             self.active_souls[spark_id]['error'] = str(e)
            
#             raise RuntimeError(f"{stage_name} failed: {e}") from e

#     def run_guff_strengthening(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
#                                duration: float = GUFF_STRENGTHENING_DURATION, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#         """
#         Runs the Guff strengthening stage.
        
#         Args:
#             soul_spark: The SoulSpark object to strengthen
#             field_controller: The field controller for Guff interaction
#             duration: Duration of the Guff strengthening process
#             show_visuals: Whether to display visualizations
            
#         Returns:
#             Tuple of (modified SoulSpark, stage metrics)
#         """
#         if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#         spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#         stage_name = "Guff Strengthening"
        
#         # Use provided field controller or the one from initialization
#         field_ctrl = field_controller or self.field_controller
#         if not field_ctrl:
#             raise ValueError("Field controller required for Guff strengthening.")
        
#         logger.info(f"Stage: {stage_name} for {spark_id}...")
#         self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
        
#         # Move soul to Guff if not already there
#         try:
#             current_field = getattr(soul_spark, 'current_field_key', None)
#             if current_field != 'guff':
#                 logger.info(f"Moving soul {spark_id} to Guff...")
#                 field_ctrl.place_soul_in_guff(soul_spark)
#         except Exception as move_err:
#             logger.error(f"Failed to move soul to Guff: {move_err}")
#             raise RuntimeError(f"Failed to move soul to Guff: {move_err}") from move_err
        
#         try:
#             # Visualization before Guff strengthening
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Pre_Guff_Strengthening"))
#                     visualize_soul_state(soul_spark, "Pre_Guff_Strengthening", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Pre-Guff visualization failed: {vis_err}")
            
#             _, stage_metrics = perform_guff_strengthening(soul_spark, field_ctrl, duration=duration)
            
#             # Visualization after Guff strengthening
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Post_Guff_Strengthening"))
#                     visualize_soul_state(soul_spark, "Post_Guff_Strengthening", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Post-Guff visualization failed: {vis_err}")
            
#             display_stage_metrics(stage_name, stage_metrics)
#             logger.info(f"{stage_name} Complete. Stability: {soul_spark.stability:.1f} SU, Coherence: {soul_spark.coherence:.1f} CU")
            
#             # Release soul from Guff
#             try:
#                 logger.info(f"Releasing soul {spark_id} from Guff...")
#                 field_ctrl.release_soul_from_guff(soul_spark)
#             except Exception as release_err:
#                 logger.error(f"Failed to release soul from Guff: {release_err}")
#                 raise RuntimeError(f"Failed to release soul from Guff: {release_err}") from release_err
            
#             return soul_spark, stage_metrics
            
#         except Exception as e:
#             logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#             self.active_souls[spark_id]['status'] = 'failed'
#             self.active_souls[spark_id]['error'] = str(e)
            
#             # Try to release from Guff even on failure
#             try:
#                 if getattr(soul_spark, 'current_field_key', None) == 'guff':
#                     field_ctrl.release_soul_from_guff(soul_spark)
#             except:
#                 pass
                
#             raise RuntimeError(f"{stage_name} failed: {e}") from e

#     def run_sephiroth_journey(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
#                               journey_duration_per_sephirah: float = 2.0, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#         """
#         Runs the Sephiroth journey process through all 10 Sephiroth.
        
#         Args:
#             soul_spark: The SoulSpark object to guide through journey
#             field_controller: The field controller for Sephiroth interaction
#             journey_duration_per_sephirah: Duration for each Sephirah interaction
#             show_visuals: Whether to display visualizations
            
#         Returns:
#             Tuple of (modified SoulSpark, journey metrics)
#         """
#         if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#         spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#         stage_name = "Sephiroth Journey"
        
#         # Use provided field controller or the one from initialization
#         field_ctrl = field_controller or self.field_controller
#         if not field_ctrl:
#             raise ValueError("Field controller required for Sephiroth journey.")
        
#         logger.info(f"Stage: {stage_name} for {spark_id}...")
#         self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
        
#         # Visualization before journey
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Pre_Sephiroth_Journey"))
#                 visualize_soul_state(soul_spark, "Pre_Sephiroth_Journey", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Pre-Journey visualization failed: {vis_err}")
        
#         try:
#             # Define the Sephiroth journey path
#             journey_path = ["kether", "chokmah", "binah", "daath", "chesed", 
#                            "geburah", "tiphareth", "netzach", "hod", 
#                            "yesod", "malkuth"]
            
#             journey_step_metrics = {}
            
#             # Process each Sephirah in the path
#             for sephirah_name in journey_path:
#                 stage_id = f"Interaction ({sephirah_name.capitalize()})"
#                 logger.info(f"  Entering {sephirah_name.capitalize()}...")
                
#                 # Get the Sephiroth field
#                 sephirah_influencer = field_ctrl.get_field(sephirah_name)
#                 if not sephirah_influencer or not isinstance(sephirah_influencer, SephirothField):
#                     raise RuntimeError(f"SephirothField missing for '{sephirah_name}'.")
                
#                 # Process the interaction
#                 _, step_metrics = process_sephirah_interaction(
#                     soul_spark, sephirah_influencer, field_ctrl, journey_duration_per_sephirah
#                 )
                
#                 journey_step_metrics[sephirah_name] = step_metrics
#                 display_stage_metrics(stage_id, step_metrics)
#                 logger.info(f"  Exiting {sephirah_name.capitalize()}.")
                
#                 # Optional visualization at important sephirah stages
#                 if self.visualization_enabled and sephirah_name in ["kether", "tiphareth", "malkuth"]:
#                     try:
#                         visualize_soul_state(soul_spark, f"Sephiroth_{sephirah_name.capitalize()}", 
#                                            self.visual_save_dir, show=show_visuals)
#                     except Exception as vis_err:
#                         logger.warning(f"Sephiroth {sephirah_name} visualization failed: {vis_err}")
            
#             # Set journey completion flags
#             setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, True)
#             setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, True)
            
#             # Final journey metrics
#             journey_metrics = {
#                 'steps': journey_step_metrics,
#                 'soul_id': spark_id,
#                 'journey_path': journey_path,
#                 'journey_duration_per_sephirah': journey_duration_per_sephirah,
#                 'total_journey_duration': journey_duration_per_sephirah * len(journey_path),
#                 'success': True
#             }
            
#             # Visualization after complete journey
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Post_Sephiroth_Journey"))
#                     visualize_soul_state(soul_spark, "Post_Sephiroth_Journey", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Post-Journey visualization failed: {vis_err}")
            
#             logger.info(f"{stage_name} Complete. Stability: {soul_spark.stability:.1f} SU, Coherence: {soul_spark.coherence:.1f} CU")
            
#             return soul_spark, journey_metrics
            
#         except Exception as e:
#             logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#             self.active_souls[spark_id]['status'] = 'failed'
#             self.active_souls[spark_id]['error'] = str(e)
            
#             # Ensure journey flags are not set on failure
#             setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False)
#             setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, False)
            
#             raise RuntimeError(f"{stage_name} failed: {e}") from e

#     def run_creator_entanglement(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
#                                 show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#         """
#         Runs the creator entanglement stage.
        
#         Args:
#             soul_spark: The SoulSpark object to entangle
#             field_controller: The field controller for Kether interaction
#             show_visuals: Whether to display visualizations
            
#         Returns:
#             Tuple of (modified SoulSpark, stage metrics)
#         """
#         if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#         spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#         stage_name = "Creator Entanglement"
        
#         # Use provided field controller or the one from initialization
#         field_ctrl = field_controller or self.field_controller
#         if not field_ctrl:
#             raise ValueError("Field controller required for Creator entanglement.")
        
#         logger.info(f"Stage: {stage_name} for {spark_id}...")
#         self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
        
#         try:
#             # Check prerequisites
#             if not getattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False):
#                 raise ValueError(f"Soul must complete Sephiroth journey before Creator entanglement.")
#             if not getattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, False):
#                 raise ValueError(f"Soul not ready for entanglement. Missing {FLAG_READY_FOR_ENTANGLEMENT} flag.")
            
#             # Get Kether field for entanglement
#             kether_influencer = field_ctrl.kether_field
#             if not kether_influencer:
#                 raise RuntimeError("Kether field unavailable.")
            
#             # Visualization before entanglement
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Pre_Creator_Entanglement"))
#                     visualize_soul_state(soul_spark, "Pre_Creator_Entanglement", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Pre-Entanglement visualization failed: {vis_err}")
            
#             # Perform entanglement
#             _, stage_metrics = perform_creator_entanglement(soul_spark, kether_influencer)
            
#             # Visualization after entanglement
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Post_Creator_Entanglement"))
#                     visualize_soul_state(soul_spark, "Post_Creator_Entanglement", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Post-Entanglement visualization failed: {vis_err}")
            
#             display_stage_metrics(stage_name, stage_metrics)
#             logger.info(f"{stage_name} Complete. Stability: {soul_spark.stability:.1f} SU, Coherence: {soul_spark.coherence:.1f} CU")
            
#             # Set flag for next stage
#             setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, True)
            
#             return soul_spark, stage_metrics
            
#         except Exception as e:
#             logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#             self.active_souls[spark_id]['status'] = 'failed'
#             self.active_souls[spark_id]['error'] = str(e)
            
#             # Ensure entanglement flags are not set on failure
#             setattr(soul_spark, FLAG_CREATOR_ENTANGLED, False)
#             setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, False)
            
#             raise RuntimeError(f"{stage_name} failed: {e}") from e

# def run_harmonic_strengthening(self, soul_spark: SoulSpark, intensity: float = HARMONIC_STRENGTHENING_INTENSITY_DEFAULT,
#                                   duration_factor: float = HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT,
#                                   show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#         """
#         Runs the harmonic strengthening stage.
        
#         Args:
#             soul_spark: The SoulSpark object to strengthen
#             intensity: Intensity of the strengthening process (0.1-1.0)
#             duration_factor: Duration factor for the process
#             show_visuals: Whether to display visualizations
            
#         Returns:
#             Tuple of (modified SoulSpark, stage metrics)
#         """
#         if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#         spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#         stage_name = FLAG_HARMONICALLY_STRENGTHENED.replace('_', ' ').title()
        
#         logger.info(f"Stage: {stage_name} for {spark_id}...")
#         self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
        
#         try:
#             # Check prerequisites
#             if not getattr(soul_spark, FLAG_CREATOR_ENTANGLED, False):
#                 raise ValueError(f"Soul must be creator entangled before harmonic strengthening.")
#             if not getattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, False):
#                 raise ValueError(f"Soul not ready for harmonization. Missing {FLAG_READY_FOR_HARMONIZATION} flag.")
            
#             # Visualization before strengthening
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Pre_Harmonic_Strengthening"))
#                     visualize_soul_state(soul_spark, "Pre_Harmonic_Strengthening", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Pre-Harmonic visualization failed: {vis_err}")
            
#             # Pre-state for logging
#             pre_stability = soul_spark.stability
#             pre_coherence = soul_spark.coherence
            
#             # Perform harmonic strengthening
#             _, stage_metrics = perform_harmonic_strengthening(
#                 soul_spark, intensity=intensity, duration_factor=duration_factor
#             )
            
#             # Visualization after strengthening
#             if self.visualization_enabled:
#                 try:
#                     self.development_states[spark_id].append((soul_spark, "Post_Harmonic_Strengthening"))
#                     visualize_soul_state(soul_spark, "Post_Harmonic_Strengthening", self.visual_save_dir, show=show_visuals)
#                 except Exception as vis_err:
#                     logger.warning(f"Post-Harmonic visualization failed: {vis_err}")
            
#             display_stage_metrics(stage_name, stage_metrics)
#             logger.info(f"{stage_name} Complete. Stability: {soul_spark.stability:.1f} SU ({soul_spark.stability-pre_stability:+.1f}), " +
#                        f"Coherence: {soul_spark.coherence:.1f} CU ({soul_spark.coherence-pre_coherence:+.1f})")
            
#             # Set flag for next stage
#             setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, True)
            
#             return soul_spark, stage_metrics
            
#         except Exception as e:
#             logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#             self.active_souls[spark_id]['status'] = 'failed'
#             self.active_souls[spark_id]['error'] = str(e)
            
#             # Ensure strengthening flags are not set on failure
#             setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False)
#             setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False)
            
#             raise RuntimeError(f"{stage_name} failed: {e}") from e

# def run_life_cord_formation(self, soul_spark: SoulSpark, intensity: float = 0.7, 
#                             complexity: float = LIFE_CORD_COMPLEXITY_DEFAULT,
#                             show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Runs the life cord formation stage.
    
#     Args:
#         soul_spark: The SoulSpark object to form life cord for
#         intensity: Intensity of the cord formation process (0.1-1.0)
#         complexity: Complexity of the cord structure (0.1-1.0)
#         show_visuals: Whether to display visualizations
        
#     Returns:
#         Tuple of (modified SoulSpark, stage metrics)
#     """
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     stage_name = FLAG_CORD_FORMATION_COMPLETE.replace('_', ' ').title()
    
#     logger.info(f"Stage: {stage_name} for {spark_id}...")
#     self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
    
#     try:
#         # Check prerequisites
#         if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False):
#             raise ValueError(f"Soul not ready for life cord formation. Missing {FLAG_READY_FOR_LIFE_CORD} flag.")
        
#         # Visualization before cord formation
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Pre_Life_Cord"))
#                 visualize_soul_state(soul_spark, "Pre_Life_Cord", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Pre-Cord visualization failed: {vis_err}")
        
#         # Perform life cord formation
#         _, stage_metrics = form_life_cord(
#             soul_spark, intensity=intensity, complexity=complexity
#         )
        
#         # Visualization after cord formation
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Post_Life_Cord"))
#                 visualize_soul_state(soul_spark, "Post_Life_Cord", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Post-Cord visualization failed: {vis_err}")
        
#         display_stage_metrics(stage_name, stage_metrics)
#         logger.info(f"{stage_name} Complete. Cord Integrity: {soul_spark.cord_integrity:.3f}, " +
#                     f"Earth Resonance: {soul_spark.earth_resonance:.3f}")
        
#         return soul_spark, stage_metrics
        
#     except Exception as e:
#         logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#         self.active_souls[spark_id]['status'] = 'failed'
#         self.active_souls[spark_id]['error'] = str(e)
        
#         # Ensure cord flags are not set on failure
#         setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False)
        
#         raise RuntimeError(f"{stage_name} failed: {e}") from e

# def run_earth_harmonization(self, soul_spark: SoulSpark, schumann_intensity: float = 0.7,
#                             core_intensity: float = 0.7, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Runs the Earth harmonization stage.
    
#     Args:
#         soul_spark: The SoulSpark object to harmonize with Earth
#         schumann_intensity: Intensity for Schumann resonance attunement (0.1-1.0)
#         core_intensity: Intensity for Earth core attunement (0.1-1.0)
#         show_visuals: Whether to display visualizations
        
#     Returns:
#         Tuple of (modified SoulSpark, stage metrics)
#     """
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     stage_name = FLAG_EARTH_ATTUNED.replace('_', ' ').title()
    
#     logger.info(f"Stage: {stage_name} for {spark_id}...")
#     self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
    
#     try:
#         # Check prerequisites
#         if not getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False):
#             raise ValueError(f"Soul must have formed life cord before Earth harmonization.")
#         if not getattr(soul_spark, FLAG_READY_FOR_EARTH, False):
#             raise ValueError(f"Soul not ready for Earth harmonization. Missing {FLAG_READY_FOR_EARTH} flag.")
        
#         # Visualization before Earth harmonization
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Pre_Earth_Harmonization"))
#                 visualize_soul_state(soul_spark, "Pre_Earth_Harmonization", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Pre-Earth visualization failed: {vis_err}")
        
#         # Pre-state for logging
#         pre_stability = soul_spark.stability
#         pre_coherence = soul_spark.coherence
#         pre_earth_res = getattr(soul_spark, 'earth_resonance', 0.0)
        
#         # Perform Earth harmonization
#         _, stage_metrics = perform_earth_harmonization(
#             soul_spark, schumann_intensity=schumann_intensity, core_intensity=core_intensity
#         )
        
#         # Visualization after Earth harmonization
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Post_Earth_Harmonization"))
#                 visualize_soul_state(soul_spark, "Post_Earth_Harmonization", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Post-Earth visualization failed: {vis_err}")
        
#         display_stage_metrics(stage_name, stage_metrics)
#         logger.info(f"{stage_name} Complete. Earth Resonance: {soul_spark.earth_resonance:.3f} ({soul_spark.earth_resonance-pre_earth_res:+.3f}), " +
#                     f"S: {soul_spark.stability:.1f} ({soul_spark.stability-pre_stability:+.1f}), " +
#                     f"C: {soul_spark.coherence:.1f} ({soul_spark.coherence-pre_coherence:+.1f})")
        
#         return soul_spark, stage_metrics
        
#     except Exception as e:
#         logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#         self.active_souls[spark_id]['status'] = 'failed'
#         self.active_souls[spark_id]['error'] = str(e)
        
#         # Ensure Earth flags are not set on failure
#         setattr(soul_spark, FLAG_EARTH_ATTUNED, False)
#         setattr(soul_spark, FLAG_ECHO_PROJECTED, False)
        
#         raise RuntimeError(f"{stage_name} failed: {e}") from e

# def run_identity_crystallization(self, soul_spark: SoulSpark, specified_name: Optional[str] = None,
#                                 train_cycles: int = 5, entrainment_bpm: float = 68.0,
#                                 entrainment_duration: float = 60.0, love_cycles: int = 3,
#                                 geometry_stages: int = 3, crystallization_threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD,
#                                 show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Runs the identity crystallization stage.
    
#     Args:
#         soul_spark: The SoulSpark object to crystallize identity for
#         specified_name: Optional name to use instead of user input
#         train_cycles: Number of training cycles for name response
#         entrainment_bpm: Beats per minute for heartbeat entrainment
#         entrainment_duration: Duration for entrainment process
#         love_cycles: Number of love resonance cycles
#         geometry_stages: Number of sacred geometry application stages
#         crystallization_threshold: Threshold for successful crystallization
#         show_visuals: Whether to display visualizations
        
#     Returns:
#         Tuple of (modified SoulSpark, stage metrics)
#     """
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     stage_name = FLAG_IDENTITY_CRYSTALLIZED.replace('_', ' ').title()
    
#     logger.info(f"Stage: {stage_name} for {spark_id}...")
#     self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
    
#     try:
#         # Check prerequisites
#         if not getattr(soul_spark, FLAG_EARTH_ATTUNED, False):
#             raise ValueError(f"Soul must be harmonized with Earth before identity crystallization.")
        
#         # Create identity kwargs
#         id_kwargs = {
#             'train_cycles': train_cycles,
#             'entrainment_bpm': entrainment_bpm,
#             'entrainment_duration': entrainment_duration,
#             'love_cycles': love_cycles,
#             'geometry_stages': geometry_stages,
#             'crystallization_threshold': crystallization_threshold
#         }
        
#         # Add specified name if provided
#         if specified_name:
#             id_kwargs['specified_name'] = specified_name
            
#         # Visualization before identity crystallization
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Pre_Identity_Crystallization"))
#                 visualize_soul_state(soul_spark, "Pre_Identity_Crystallization", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Pre-Identity visualization failed: {vis_err}")
        
#         # Perform identity crystallization
#         _, stage_metrics = perform_identity_crystallization(soul_spark, **id_kwargs)
        
#         # Visualization after identity crystallization
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Post_Identity_Crystallization"))
#                 visualize_soul_state(soul_spark, "Post_Identity_Crystallization", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Post-Identity visualization failed: {vis_err}")
        
#         display_stage_metrics(stage_name, stage_metrics)
#         logger.info(f"{stage_name} Complete. Name: {soul_spark.name}, " +
#                     f"Crystallization Level: {soul_spark.crystallization_level:.3f}")
        
#         return soul_spark, stage_metrics
        
#     except Exception as e:
#         logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#         self.active_souls[spark_id]['status'] = 'failed'
#         self.active_souls[spark_id]['error'] = str(e)
        
#         # Ensure crystallization flags are not set on failure
#         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False)
#         setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
        
#         raise RuntimeError(f"{stage_name} failed: {e}") from e

# def run_birth_process(self, soul_spark: SoulSpark, intensity: float = BIRTH_INTENSITY_DEFAULT,
#                     mother_profile: Optional[Dict[str, Any]] = None, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Runs the birth process stage.
    
#     Args:
#         soul_spark: The SoulSpark object to birth
#         intensity: Intensity of the birth process (0.1-1.0)
#         mother_profile: Optional mother profile for nurturing influence
#         show_visuals: Whether to display visualizations
        
#     Returns:
#         Tuple of (modified SoulSpark, stage metrics)
#     """
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     stage_name = "Birth"
    
#     logger.info(f"Stage: {stage_name} for {spark_id}...")
#     self.active_souls[spark_id] = {'status': 'processing', 'current_stage': stage_name, 'start_time': datetime.now().isoformat()}
    
#     try:
#         # Check prerequisites
#         if not getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False):
#             raise ValueError(f"Soul must have crystallized identity before birth.")
#         if not getattr(soul_spark, FLAG_READY_FOR_BIRTH, False):
#             raise ValueError(f"Soul not ready for birth. Missing {FLAG_READY_FOR_BIRTH} flag.")
        
#         # Create/get mother profile if not provided
#         if not mother_profile and MOTHER_RESONANCE_AVAILABLE:
#             try:
#                 mother_resonance_data = create_mother_resonance_data()
#                 mother_profile = {
#                     'nurturing_capacity': mother_resonance_data.get('nurturing_capacity', 0.7),
#                     'spiritual': mother_resonance_data.get('spiritual', {'connection': 0.6}),
#                     'love_resonance': mother_resonance_data.get('love_resonance', 0.7)
#                 }
#             except Exception as mother_err:
#                 logger.warning(f"Failed to create mother resonance profile: {mother_err}")
#                 mother_profile = None
        
#         # Visualization before birth process
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Pre_Birth"))
#                 visualize_soul_state(soul_spark, "Pre_Birth", self.visual_save_dir, show=show_visuals)
#             except Exception as vis_err:
#                 logger.warning(f"Pre-Birth visualization failed: {vis_err}")
        
#         # Perform birth process
#         _, stage_metrics = perform_birth(
#             soul_spark, intensity=intensity, mother_profile=mother_profile
#         )
        
#         # Visualization after birth process
#         if self.visualization_enabled:
#             try:
#                 self.development_states[spark_id].append((soul_spark, "Post_Birth"))
#                 visualize_soul_state(soul_spark, "Post_Birth", self.visual_save_dir, show=show_visuals)
                
#                 # Create development comparison visualization
#                 try:
#                     compare_path = visualize_state_comparison(
#                         self.development_states[spark_id],
#                         self.visual_save_dir,
#                         show=show_visuals
#                     )
#                     logger.info(f"Development comparison created: {compare_path}")
#                 except Exception as comp_err:
#                     logger.warning(f"Failed to create development comparison: {comp_err}")
                    
#             except Exception as vis_err:
#                 logger.warning(f"Post-Birth visualization failed: {vis_err}")
        
#         display_stage_metrics(stage_name, stage_metrics)
#         logger.info(f"{stage_name} Complete. Incarnated: {getattr(soul_spark, FLAG_INCARNATED, False)}")
        
#         return soul_spark, stage_metrics
        
#     except Exception as e:
#         logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
#         self.active_souls[spark_id]['status'] = 'failed'
#         self.active_souls[spark_id]['error'] = str(e)
        
#         # Ensure incarnation flag is not set on failure
#         setattr(soul_spark, FLAG_INCARNATED, False)
        
#         raise RuntimeError(f"{stage_name} failed: {e}") from e

# def run_soul_completion(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
#                         journey_duration_per_sephirah: float = 2.0, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Runs the complete sequence of soul formation stages from Spark Harmonization to Birth.
    
#     Args:
#         soul_spark: The SoulSpark object to process
#         field_controller: Optional field controller for field-related operations
#         journey_duration_per_sephirah: Duration for each Sephirah interaction
#         show_visuals: Whether to display visualizations
#         **kwargs: Optional parameters to override defaults for specific stages
        
#     Returns:
#         Tuple of (modified SoulSpark, overall metrics)
        
#     Raises:
#         TypeError, ValueError, RuntimeError
#     """
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     if spark_id in self.active_souls and self.active_souls[spark_id]['status'] == 'processing':
#         raise RuntimeError(f"Soul {spark_id} is already being processed.")

#     logger.info(f"--- Starting Soul Completion Process for Soul {spark_id} ---")
#     start_time_iso = datetime.now().isoformat()
#     start_time_dt = datetime.fromisoformat(start_time_iso)
#     completion_summary = {'soul_id': spark_id, 'stages': {}}
#     self.active_souls[spark_id] = {'status': 'processing', 'current_stage': None, 'start_time': start_time_iso}

#     try:
#         # Use provided field controller or the one from initialization
#         field_ctrl = field_controller or self.field_controller
#         if not field_ctrl:
#             raise ValueError("Field controller required for soul completion process.")
        
#         # --- Stage 1: Spark Harmonization ---
#         _, metrics1 = self.run_spark_harmonization(soul_spark, show_visuals=show_visuals)
#         completion_summary['stages']['Spark Harmonization'] = metrics1
        
#         # --- Stage 2: Guff Strengthening ---
#         guff_duration = kwargs.get('guff_duration', GUFF_STRENGTHENING_DURATION)
#         _, metrics2 = self.run_guff_strengthening(
#             soul_spark, field_ctrl, duration=guff_duration, show_visuals=show_visuals
#         )
#         completion_summary['stages']['Guff Strengthening'] = metrics2
        
#         # --- Stage 3: Sephiroth Journey ---
#         _, metrics3 = self.run_sephiroth_journey(
#             soul_spark, field_ctrl, journey_duration_per_sephirah=journey_duration_per_sephirah, 
#             show_visuals=show_visuals
#         )
#         completion_summary['stages']['Sephiroth Journey'] = metrics3
        
#         # --- Stage 4: Creator Entanglement ---
#         _, metrics4 = self.run_creator_entanglement(soul_spark, field_ctrl, show_visuals=show_visuals)
#         completion_summary['stages']['Creator Entanglement'] = metrics4
        
#         # --- Stage 5: Harmonic Strengthening ---
#         harmony_intensity = kwargs.get('harmony_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT)
#         harmony_duration = kwargs.get('harmony_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT)
#         _, metrics5 = self.run_harmonic_strengthening(
#             soul_spark, intensity=harmony_intensity, duration_factor=harmony_duration, 
#             show_visuals=show_visuals
#         )
#         completion_summary['stages']['Harmonic Strengthening'] = metrics5
        
#         # --- Stage 6: Life Cord Formation ---
#         # Calculate life cord intensity if not provided
#         if 'life_cord_intensity' in kwargs:
#             cord_intensity = kwargs['life_cord_intensity']
#         else:
#             # Dynamic calculation based on stability and coherence
#             normalized_stability = min(1.0, soul_spark.stability / MAX_STABILITY_SU)
#             normalized_coherence = min(1.0, soul_spark.coherence / MAX_COHERENCE_CU)
#             cord_intensity = 0.5 + (normalized_stability * 0.3 + normalized_coherence * 0.2)
#             cord_intensity = min(1.0, cord_intensity)
            
#         cord_complexity = kwargs.get('cord_complexity', LIFE_CORD_COMPLEXITY_DEFAULT)
#         _, metrics6 = self.run_life_cord_formation(
#             soul_spark, intensity=cord_intensity, complexity=cord_complexity, 
#             show_visuals=show_visuals
#         )
#         completion_summary['stages']['Life Cord Formation'] = metrics6
        
#         # --- Stage 7: Earth Harmonization ---
#         # Get earth intensities
#         schumann_intensity = kwargs.get('schumann_intensity', 0.7)
#         core_intensity = kwargs.get('core_intensity', 0.7)
        
#         _, metrics7 = self.run_earth_harmonization(
#             soul_spark, schumann_intensity=schumann_intensity, core_intensity=core_intensity, 
#             show_visuals=show_visuals
#         )
#         completion_summary['stages']['Earth Harmonization'] = metrics7
        
#         # --- Stage 8: Identity Crystallization ---
#         # Extract identity-specific kwargs
#         id_kwargs = {
#             'specified_name': kwargs.get('specified_name'),
#             'train_cycles': kwargs.get('train_cycles', 5),
#             'entrainment_bpm': kwargs.get('entrainment_bpm', 68),
#             'entrainment_duration': kwargs.get('entrainment_duration', 60),
#             'love_cycles': kwargs.get('love_cycles', 3),
#             'geometry_stages': kwargs.get('geometry_stages', 3),
#             'crystallization_threshold': kwargs.get('crystallization_threshold', IDENTITY_CRYSTALLIZATION_THRESHOLD),
#             'show_visuals': show_visuals
#         }
        
#         _, metrics8 = self.run_identity_crystallization(soul_spark, **id_kwargs)
#         completion_summary['stages']['Identity Crystallization'] = metrics8
        
#         # --- Stage 9: Birth Process ---
#         # Get birth parameters
#         birth_intensity = kwargs.get('birth_intensity', BIRTH_INTENSITY_DEFAULT)
#         mother_profile = kwargs.get('mother_profile')
        
#         _, metrics9 = self.run_birth_process(
#             soul_spark, intensity=birth_intensity, mother_profile=mother_profile, 
#             show_visuals=show_visuals
#         )
#         completion_summary['stages']['Birth'] = metrics9
        
#         # --- Finalization ---
#         end_time_iso = datetime.now().isoformat()
#         end_time_dt = datetime.fromisoformat(end_time_iso)
#         completion_summary['start_time'] = start_time_iso
#         completion_summary['end_time'] = end_time_iso
#         completion_summary['duration_seconds'] = (end_time_dt - start_time_dt).total_seconds()
#         completion_summary['success'] = True
#         completion_summary['final_soul_state_summary'] = soul_spark.get_spark_metrics()['core']

#         self.active_souls[spark_id]['status'] = 'completed'
#         self.active_souls[spark_id]['end_time'] = end_time_iso
#         self.active_souls[spark_id]['current_stage'] = None
#         self.active_souls[spark_id]['summary'] = {k: v for k, v in completion_summary.items() if k != 'stages'}

#         self._save_completed_soul(soul_spark)

#         # Record overall metrics for this controller's run
#         overall_controller_metrics = {
#             'controller_run': 'soul_completion', 'soul_id': spark_id,
#             'start_time': start_time_iso, 'end_time': end_time_iso,
#             'duration_seconds': completion_summary['duration_seconds'],
#             'success': True,
#             'final_energy_seu': soul_spark.energy,
#             'final_stability_su': soul_spark.stability,
#             'final_coherence_cu': soul_spark.coherence,
#             'final_incarnated_status': getattr(soul_spark, FLAG_INCARNATED, False)
#         }
#         if METRICS_AVAILABLE: metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, overall_controller_metrics)

#         logger.info(f"--- Soul Completion Process Finished Successfully for Soul {spark_id} ---")
#         logger.info(f"Duration: {completion_summary['duration_seconds']:.2f}s")
#         logger.info(f"Final Incarnated Status: {getattr(soul_spark, FLAG_INCARNATED, False)}")

#         return soul_spark, overall_controller_metrics

#     except Exception as e:
#         # Error handling for the soul completion process
#         failed_stage_name = self.active_souls[spark_id].get('current_stage', 'unknown')
#         end_time_iso = datetime.now().isoformat()
#         logger.error(f"Soul completion failed at stage '{failed_stage_name}': {e}", exc_info=True)
#         self.active_souls[spark_id]['status'] = 'failed'
#         self.active_souls[spark_id]['end_time'] = end_time_iso
#         self.active_souls[spark_id]['error'] = str(e)
#         setattr(soul_spark, FLAG_INCARNATED, False)
        
#         # Record failure metric
#         if METRICS_AVAILABLE:
#             metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
#                 'controller_run': 'soul_completion', 'soul_id': spark_id,
#                 'start_time': start_time_iso, 'end_time': end_time_iso,
#                 'duration_seconds': (datetime.fromisoformat(end_time_iso) - start_time_dt).total_seconds(),
#                 'success': False, 'error': str(e), 'failed_stage': failed_stage_name })
        
#         raise RuntimeError(f"Soul completion process failed at stage '{failed_stage_name}': {e}") from e

# def _save_completed_soul(self, soul_spark: SoulSpark) -> bool:
#     """Saves the final state of the completed soul."""
#     spark_id = getattr(soul_spark, 'spark_id', None)
#     if not spark_id: return False
#     filename = f"soul_completed_{spark_id}.json"
#     save_path = os.path.join(self.output_dir, filename)
#     logger.info(f"Saving completed soul data for {spark_id} to {save_path}...")
#     try:
#         if hasattr(soul_spark, 'save_spark_data'):
#             # Use soul's built-in serialization if available
#             soul_spark.save_spark_data(save_path)
#         else:
#             # Otherwise serialize using __dict__ with some exclusions
#             soul_dict = {k: v for k, v in soul_spark.__dict__.items() 
#                         if not k.startswith('_') and not callable(v)}
#             with open(save_path, 'w', encoding='utf-8') as f:
#                 json.dump(soul_dict, f, indent=2, default=str)
        
#         # Also save to completed souls directory
#         completed_dir = os.path.join("output", "completed")
#         os.makedirs(completed_dir, exist_ok=True)
#         completed_path = os.path.join(completed_dir, filename)
        
#         # Copy the file to completed directory
#         import shutil
#         shutil.copy2(save_path, completed_path)
        
#         logger.info(f"Soul {spark_id} saved successfully to {save_path} and {completed_path}")
#         return True
#     except Exception as e:
#         logger.error(f"Failed to save completed soul {spark_id}: {e}", exc_info=True)
#         return False

# def get_active_souls_status(self) -> Dict[str, Dict[str, Any]]:
#     """Returns the status of all active souls being processed."""
#     return {k: v.copy() for k, v in self.active_souls.items()}

# def get_soul_status(self, soul_id: str) -> Optional[Dict[str, Any]]:
#     """Returns the status of a specific soul if it exists."""
#     return self.active_souls.get(soul_id, {}).copy() if soul_id in self.active_souls else None

# def get_visualization_paths(self, soul_id: str) -> List[str]:
#     """Returns paths to visualizations for a specific soul."""
#     if not self.visualization_enabled or soul_id not in self.development_states:
#         return []
    
#     visual_files = []
#     try:
#         # Look for visualization files with this soul ID
#         if os.path.exists(self.visual_save_dir):
#             for filename in os.listdir(self.visual_save_dir):
#                 if soul_id in filename and (filename.endswith('.png') or filename.endswith('.jpg')):
#                     visual_files.append(os.path.join(self.visual_save_dir, filename))
#     except Exception as e:
#         logger.warning(f"Error getting visualization paths for {soul_id}: {e}")
    
#     return sorted(visual_files)

# def cleanup_resources(self):
#     """Cleans up any resources used by the controller."""
#     try:
#         # Clear any temporary resources
#         self.active_souls.clear()
        
#         # Force metrics persistence if available
#         if METRICS_AVAILABLE and hasattr(metrics, 'persist_metrics'):
#             metrics.persist_metrics()
            
#         # Log controller shutdown
#         shutdown_time = datetime.now().isoformat()
#         logger.info(f"Soul Completion Controller '{self.controller_id}' shutting down at {shutdown_time}")
        
#         return True
#     except Exception as e:
#         logger.error(f"Error during controller cleanup: {e}", exc_info=True)
#         return False

# def __del__(self):
#     """Destructor to ensure cleanup when object is deleted."""
#     try:
#         self.cleanup_resources()
#     except:
#         # Suppress errors in destructor
#         pass