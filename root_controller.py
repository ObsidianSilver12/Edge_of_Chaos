# --- START OF FILE root_controller.py ---

"""
Root Controller (Refactored V4.3.11 - Direct Full Soul Formation, Visualization)

Orchestrates the simulation from spark emergence through all stages directly.
Initializes SoulSpark via field sampling, then processes through Harmonization, Guff,
Journey, Creator Entanglement, Harmonic Strengthening, Life Cord, Earth Harmonization,
Identity Crystallization, and Birth.
Integrates mandatory visualization calls, including a comprehensive final report.
Uses updated units and principle-driven S/C logic. Hard fails on visualization errors.
Corrected constant usage (no 'const.' prefix after wildcard import).
This version assumes only ONE soul is processed per run_simulation call.
"""

import logging
import os
import sys
import random
import time
import traceback
import json
import uuid
import numpy as np
from datetime import datetime, timedelta # Added timedelta
from math import pi as PI_MATH # type: ignore
try:
    from typing import List, Optional, Dict, Any, Tuple
except ImportError:
    List = Dict = Any = Optional = Tuple = type(None) # type: ignore

# --- Constants Import (CRUCIAL - Wildcard Import) ---
try:
    from constants.constants import *
    # Add explicit checks for key constants needed here
    if 'LOG_LEVEL' not in locals(): raise NameError("LOG_LEVEL not defined")
    if 'GRID_SIZE' not in locals(): raise NameError("GRID_SIZE not defined")
    # ... (add all other necessary constant checks from your previous version) ...
    if 'DATA_DIR_BASE' not in locals(): raise NameError("DATA_DIR_BASE not defined")

    logger_check = logging.getLogger('ConstantsCheckRoot')
    logger_check.info(f"RootController: Constants loaded. LOG_LEVEL={LOG_LEVEL}, GRID_SIZE={GRID_SIZE}")
except ImportError as e:
    print(f"FATAL ROOT: Could not import constants.constants: {e}")
    sys.exit(1)
except NameError as e:
    print(f"FATAL ROOT: Constant definition missing after import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ROOT: Error loading constants: {e}")
    sys.exit(1)


# --- Logger Initialization ---
logger = logging.getLogger('root_controller')
log_file_path = os.path.join(DATA_DIR_BASE, "logs", "root_controller_run.log")
if not logger.handlers:
    log_level_int = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level_int)
    log_formatter = logging.Formatter(LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
        logger.info("RootController Logging configured (Terminal: INFO, File: DEBUG).")
    except Exception as log_err:
        logger.error(f"RootController File logging setup failed: {log_err}", exc_info=True)
        logger.info("RootController Logging configured (Terminal: INFO, File: Disabled).")
else:
    logger.info("RootController Logger 'root_controller' already has handlers.")

# --- Core Stage Function Imports ---
try:
    logger.debug("RootController: Importing core components and stage functions...")
    from stage_1.fields.field_controller import FieldController
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.fields.sephiroth_field import SephirothField # Needed for type hint

    from stage_1.soul_formation.spark_harmonization import perform_spark_harmonization
    from stage_1.soul_formation.guff_strengthening import perform_guff_strengthening
    from stage_1.soul_formation.sephiroth_journey_processing import process_sephirah_interaction
    from stage_1.soul_formation.creator_entanglement import perform_creator_entanglement
    from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening
    from stage_1.soul_formation.life_cord import form_life_cord
    from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization
    from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization
    from stage_1.soul_formation.birth import perform_birth
    logger.debug("RootController: Core components and stage functions imported successfully.")
except ImportError as e:
    logger.critical(f"FATAL ROOT: Could not import core component or stage function: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"FATAL ROOT: Error during core component/stage function imports: {e}", exc_info=True)
    sys.exit(1)

# --- Visualization Import & Setup ---
VISUALIZATION_ENABLED = False
VISUALIZATION_OUTPUT_DIR_ROOT = os.path.join(DATA_DIR_BASE, "visuals", "simulation_run") # Unified visual output
try:
    from stage_1.soul_formation.soul_visualizer import (
        visualize_soul_state,
        visualize_state_comparison,
        create_comprehensive_soul_report # New import
    )
    VISUALIZATION_ENABLED = True
    os.makedirs(VISUALIZATION_OUTPUT_DIR_ROOT, exist_ok=True)
    logger.info("RootController: Soul visualization module loaded successfully.")
except ImportError as ie:
    logger.critical(f"CRITICAL ROOT ERROR: Soul visualization module not found: {ie}. Visualizations are required. Aborting.")
    sys.exit(1)
except Exception as e:
    logger.critical(f"CRITICAL ROOT ERROR: Error setting up visualization: {e}. Aborting.", exc_info=True)
    sys.exit(1)

# --- Metrics Import & Setup ---
METRICS_AVAILABLE = False
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
    logger.info("RootController: Metrics tracking module loaded successfully.")
except ImportError:
    logger.warning("RootController: Metrics tracking module not found. Using placeholder.")
    class MetricsPlaceholder:
        @staticmethod
        def record_metric(*args, **kwargs): pass
        @staticmethod
        def record_metrics(*args, **kwargs): pass
        @staticmethod
        def persist_metrics(*args, **kwargs): pass
    metrics = MetricsPlaceholder() # type: ignore
except Exception as e: # Catch any error during metrics import
    logger.critical(f"CRITICAL ROOT ERROR metrics import: {e}", exc_info=True)
    class MetricsPlaceholderOnError: # type: ignore
        @staticmethod
        def record_metric(*args, **kwargs): pass
        @staticmethod
        def record_metrics(*args, **kwargs): pass
        @staticmethod
        def persist_metrics(*args, **kwargs): pass
    metrics = MetricsPlaceholderOnError() # type: ignore


# --- Mother Resonance Import ---
MOTHER_RESONANCE_AVAILABLE = False # Default
try:
    from stage_1.evolve.core.mother_resonance import create_mother_resonance_data
    MOTHER_RESONANCE_AVAILABLE = True
    logger.info("Mother resonance module loaded successfully for RootController.")
except ImportError:
    logger.warning("Mother resonance module not found for RootController. Birth will proceed without mother influence or require explicit profile.")


# --- Helper Function for Metrics Display (from soul_completion_controller) ---
def display_stage_metrics(stage_name, metrics_dict):
    # (Implementation is identical to the one in soul_completion_controller, so not repeated for brevity)
    # Ensure this helper is defined or imported if it's not directly in this file.
    # For this consolidated version, I'll assume it's defined here.
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
        # ... (rest of the unit logic from your display_stage_metrics) ...
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



# --- Main Simulation Logic ---
def run_simulation(num_souls: int = 1, # Changed to 1 as per your intent
                   journey_duration_per_sephirah: float = 2.0,
                   report_path_base: str = os.path.join(DATA_DIR_BASE, "reports"),
                   show_visuals: bool = False,
                   **kwargs # To pass stage-specific overrides
                   ) -> None:
    """ Runs the main simulation flow for a single soul. """
    if num_souls != 1:
        logger.warning("RootController is configured to run for a single soul. num_souls > 1 will only process the first.")
        num_souls = 1 # Enforce single soul

    logger.info("--- Starting Soul Development Simulation (V4.3.11 - Direct Full Soul Completion) ---")
    logger.info(f"Num Souls: {num_souls}, Sephirah Duration: {journey_duration_per_sephirah}")
    overall_start_time = time.time()
    simulation_start_iso = datetime.now().isoformat()
    
    # Since it's one soul, the summary directly becomes the final report data for that soul
    single_soul_final_summary: Optional[Dict[str, Any]] = None

    os.makedirs(report_path_base, exist_ok=True)
    # Report path will include soul ID later
    
    field_controller: Optional[FieldController] = None
    development_states: List[Tuple[SoulSpark, str]] = [] # For this single soul

    try:
        logger.info("RootController: Initializing Field Controller...")
        field_controller = FieldController(grid_size=GRID_SIZE)
        logger.info("RootController: Field Controller initialized.")

        # --- Process the Single Soul ---
        base_id_str = (
            f"{simulation_start_iso.replace(':','-').replace('.','')}"
            f"_Soul_001" # Only one soul
        )
        logger.info(f"\n===== Processing Soul (Base ID: {base_id_str}) =====")
        single_soul_start_time = time.time()
        process_summary: Dict[str, Any] = {'base_id': base_id_str, 'stages_metrics': {}} # Ensure type
        current_stage_name = "Pre-Emergence"
        soul_spark: Optional[SoulSpark] = None

        try:
            # --- Stage 1: Spark Emergence ---
            current_stage_name = "Spark Emergence"
            logger.info(f"RootController Stage: {current_stage_name}...")
            creation_location = field_controller.find_optimal_development_location()
            creation_location_coords = field_controller._coords_to_int_tuple(creation_location)
            soul_spark = SoulSpark.create_from_field_emergence(field_controller, creation_location_coords)
            process_summary['soul_id'] = soul_spark.spark_id
            logger.info(f"Soul Spark {soul_spark.spark_id} emerged at {creation_location_coords}.")
            emergence_metrics = {'success': True, **soul_spark.get_spark_metrics()['core']}
            process_summary['stages_metrics'][current_stage_name] = emergence_metrics
            if METRICS_AVAILABLE: metrics.record_metrics(f"root_{current_stage_name.lower().replace(' ','_')}", emergence_metrics)
            display_stage_metrics(current_stage_name, emergence_metrics)
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Spark_Emergence_Initial"))
                    visualize_soul_state(soul_spark, "Spark_Emergence_Initial", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name}: {vis_err}") from vis_err

            # --- Stage 2: Spark Harmonization ---
            current_stage_name = "Spark Harmonization"
            logger.info(f"RootController Stage: {current_stage_name}...")
            _, harm_metrics = perform_spark_harmonization(soul_spark, iterations=kwargs.get('harmonization_iterations', HARMONIZATION_ITERATIONS))
            process_summary['stages_metrics'][current_stage_name] = harm_metrics
            display_stage_metrics(current_stage_name, harm_metrics)
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Harmonization"))
                    visualize_soul_state(soul_spark, "Post_Harmonization", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name}: {vis_err}") from vis_err

            # --- Stage 3: Move to Guff & Guff Strengthening ---
            current_stage_name = "Guff Strengthening"
            logger.info(f"RootController Stage: {current_stage_name} (including move)...")
            field_controller.place_soul_in_guff(soul_spark)
            logger.info(f"Soul {soul_spark.spark_id} moved to Guff.")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Guff_Strengthening"))
                    visualize_soul_state(soul_spark, "Pre_Guff_Strengthening", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err
            _, guff_metrics = perform_guff_strengthening(
                soul_spark=soul_spark, field_controller=field_controller,
                duration=kwargs.get('guff_duration', GUFF_STRENGTHENING_DURATION)
            )
            process_summary['stages_metrics'][current_stage_name] = guff_metrics
            display_stage_metrics(current_stage_name, guff_metrics)
            field_controller.release_soul_from_guff(soul_spark) # Release after strengthening
            logger.info(f"Soul {soul_spark.spark_id} released from Guff.")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Guff_Strengthening"))
                    visualize_soul_state(soul_spark, "Post_Guff_Strengthening", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post): {vis_err}") from vis_err

            # --- Stage 4: Sephiroth Journey ---
            current_stage_name = "Sephiroth Journey"
            logger.info(f"RootController Stage: {current_stage_name}...")
            journey_path = ["kether", "chokmah", "binah", "daath", "chesed",
                           "geburah", "tiphareth", "netzach", "hod", "yesod", "malkuth"]
            journey_overall_metrics: Dict[str, Any] = {'steps': {}, 'success': True, 'total_duration': 0.0}
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Sephiroth_Journey"))
                    visualize_soul_state(soul_spark, "Pre_Sephiroth_Journey", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err

            for sephirah_name_iter in journey_path:
                stage_id_readable = f"Journey: {sephirah_name_iter.capitalize()}"
                logger.info(f"  Entering {sephirah_name_iter.capitalize()}...")
                sephirah_influencer = field_controller.get_field(sephirah_name_iter)
                if not sephirah_influencer or not isinstance(sephirah_influencer, SephirothField):
                    raise RuntimeError(f"SephirothField missing for '{sephirah_name_iter}'.")
                _, step_metrics = process_sephirah_interaction(
                    soul_spark=soul_spark, sephirah_influencer=sephirah_influencer,
                    field_controller=field_controller, duration=journey_duration_per_sephirah
                )
                journey_overall_metrics['steps'][sephirah_name_iter] = step_metrics
                journey_overall_metrics['total_duration'] += journey_duration_per_sephirah
                display_stage_metrics(stage_id_readable, step_metrics)
                logger.info(f"  Exiting {sephirah_name_iter.capitalize()}.")
                if VISUALIZATION_ENABLED and sephirah_name_iter in ["kether", "tiphareth", "malkuth"]:
                    try: visualize_soul_state(soul_spark, f"Journey_{sephirah_name_iter.capitalize()}", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                    except Exception as vis_err: raise RuntimeError(f"Visualization error at Journey ({sephirah_name_iter}): {vis_err}") from vis_err
            setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, True)
            setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, True)
            process_summary['stages_metrics'][current_stage_name] = journey_overall_metrics
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Sephiroth_Journey"))
                    visualize_soul_state(soul_spark, "Post_Sephiroth_Journey", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post): {vis_err}") from vis_err

            # --- Stage 5: Creator Entanglement ---
            current_stage_name = "Creator Entanglement"
            logger.info(f"RootController Stage: {current_stage_name}...")
            kether_influencer = field_controller.kether_field
            if not kether_influencer: raise RuntimeError("Kether field unavailable for Entanglement.")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Creator_Entanglement"))
                    visualize_soul_state(soul_spark, "Pre_Creator_Entanglement", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err
            _, entanglement_metrics = perform_creator_entanglement(
                soul_spark=soul_spark, kether_field=kether_influencer,
                base_creator_potential=kwargs.get('base_creator_potential', CREATOR_POTENTIAL_DEFAULT),
                edge_of_chaos_target=kwargs.get('edge_of_chaos_target', EDGE_OF_CHAOS_DEFAULT)
            )
            process_summary['stages_metrics'][current_stage_name] = entanglement_metrics
            display_stage_metrics(current_stage_name, entanglement_metrics)
            setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, True) # Set by perform_creator_entanglement
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Creator_Entanglement"))
                    visualize_soul_state(soul_spark, "Post_Creator_Entanglement", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post): {vis_err}") from vis_err

            # --- Stage 6: Harmonic Strengthening ---
            current_stage_name = FLAG_HARMONICALLY_STRENGTHENED.replace('_',' ').title()
            logger.info(f"RootController Stage: {current_stage_name}...")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Harmonic_Strengthening"))
                    visualize_soul_state(soul_spark, "Pre_Harmonic_Strengthening", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err
            _, hs_metrics = perform_harmonic_strengthening(
                soul_spark=soul_spark,
                intensity=kwargs.get('harmony_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT),
                duration_factor=kwargs.get('harmony_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT)
            )
            process_summary['stages_metrics'][current_stage_name] = hs_metrics
            display_stage_metrics(current_stage_name, hs_metrics)
            setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, True) # Set by perform_harmonic_strengthening
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Harmonic_Strengthening"))
                    visualize_soul_state(soul_spark, "Post_Harmonic_Strengthening", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post): {vis_err}") from vis_err

            # --- Stage 7: Life Cord Formation ---
            current_stage_name = FLAG_CORD_FORMATION_COMPLETE.replace('_',' ').title()
            logger.info(f"RootController Stage: {current_stage_name}...")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Life_Cord"))
                    visualize_soul_state(soul_spark, "Pre_Life_Cord", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err
            _, lc_metrics = form_life_cord(
                soul_spark=soul_spark,
                intensity=kwargs.get('life_cord_intensity', 0.7), # Using the default passed to SoulCompletionController
                complexity=kwargs.get('cord_complexity', LIFE_CORD_COMPLEXITY_DEFAULT)
            )
            process_summary['stages_metrics'][current_stage_name] = lc_metrics
            display_stage_metrics(current_stage_name, lc_metrics)
            # FLAG_READY_FOR_EARTH is set by form_life_cord
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Life_Cord"))
                    visualize_soul_state(soul_spark, "Post_Life_Cord", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post): {vis_err}") from vis_err

            # --- Stage 8: Earth Harmonization ---
            current_stage_name = FLAG_EARTH_ATTUNED.replace('_',' ').title()
            logger.info(f"RootController Stage: {current_stage_name}...")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Earth_Harmonization"))
                    visualize_soul_state(soul_spark, "Pre_Earth_Harmonization", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err
            _, eh_metrics = perform_earth_harmonization(
                soul_spark=soul_spark,
                schumann_intensity=kwargs.get('schumann_intensity', EARTH_HARMONY_INTENSITY_DEFAULT),
                core_intensity=kwargs.get('core_intensity', EARTH_HARMONY_INTENSITY_DEFAULT)
            )
            process_summary['stages_metrics'][current_stage_name] = eh_metrics
            display_stage_metrics(current_stage_name, eh_metrics)
            # FLAG_ECHO_PROJECTED is set by perform_earth_harmonization
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Earth_Harmonization"))
                    visualize_soul_state(soul_spark, "Post_Earth_Harmonization", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post): {vis_err}") from vis_err

            # --- Stage 9: Identity Crystallization ---
            current_stage_name = FLAG_IDENTITY_CRYSTALLIZED.replace('_',' ').title()
            logger.info(f"RootController Stage: {current_stage_name}...")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Identity_Crystallization"))
                    visualize_soul_state(soul_spark, "Pre_Identity_Crystallization", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err
            id_kwargs = {k:v for k,v in kwargs.items() if k in [
                'train_cycles', 'entrainment_bpm', 'entrainment_duration',
                'love_cycles', 'geometry_stages', 'crystallization_threshold'
            ]} # Filter relevant kwargs
            _, id_metrics = perform_identity_crystallization(soul_spark=soul_spark, **id_kwargs)
            process_summary['stages_metrics'][current_stage_name] = id_metrics
            display_stage_metrics(current_stage_name, id_metrics)
            # FLAG_READY_FOR_BIRTH is set by perform_identity_crystallization
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Identity_Crystallization"))
                    visualize_soul_state(soul_spark, "Post_Identity_Crystallization", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post): {vis_err}") from vis_err

            # --- Stage 10: Birth Process ---
            current_stage_name = "Birth"
            logger.info(f"RootController Stage: {current_stage_name}...")
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Pre_Birth"))
                    visualize_soul_state(soul_spark, "Pre_Birth", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Pre): {vis_err}") from vis_err

            birth_mother_profile = kwargs.get('mother_profile')
            if not birth_mother_profile and MOTHER_RESONANCE_AVAILABLE:
                try:
                    mother_resonance_data = create_mother_resonance_data()
                    birth_mother_profile = {
                        'nurturing_capacity': mother_resonance_data.get('nurturing_capacity', 0.7),
                        'spiritual': mother_resonance_data.get('spiritual', {'connection': 0.6}),
                        'love_resonance': mother_resonance_data.get('love_resonance', 0.7)
                    }
                except Exception as mother_err: logger.warning(f"Failed to create mother profile: {mother_err}")

            _, birth_metrics = perform_birth(
                soul_spark=soul_spark,
                intensity=kwargs.get('birth_intensity', BIRTH_INTENSITY_DEFAULT),
                mother_profile=birth_mother_profile
            )
            process_summary['stages_metrics'][current_stage_name] = birth_metrics
            display_stage_metrics(current_stage_name, birth_metrics)
            if VISUALIZATION_ENABLED:
                try:
                    development_states.append((soul_spark, "Post_Birth"))
                    visualize_soul_state(soul_spark, "Post_Birth", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                    # Final comparison visualization
                    comp_path = visualize_state_comparison(development_states, VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                    logger.info(f"Final development comparison visualization saved to: {comp_path}")
                except Exception as vis_err: raise RuntimeError(f"Visualization error at {current_stage_name} (Post/Comparison): {vis_err}") from vis_err

            # --- Generate Comprehensive Report for this Soul ---
            if VISUALIZATION_ENABLED and soul_spark:
                try:
                    logger.info(f"Generating comprehensive report for soul {soul_spark.spark_id}...")
                    report_file_path = create_comprehensive_soul_report(
                        soul_spark, "Soul_Development_Complete", VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals
                    )
                    logger.info(f"Comprehensive soul report saved to: {report_file_path}")
                    process_summary['comprehensive_report_path'] = report_file_path
                except Exception as report_err:
                    logger.error(f"Failed to generate comprehensive soul report for {soul_spark.spark_id}: {report_err}", exc_info=True)


            # --- Completion ---
            final_metrics_core = soul_spark.get_spark_metrics()['core']
            process_summary['final_soul_state'] = final_metrics_core
            process_summary['success'] = True
            process_summary['end_time'] = datetime.now().isoformat()
            process_summary['total_duration_seconds'] = time.time() - single_soul_start_time
            single_soul_final_summary = process_summary # Since it's one soul
            logger.info(f"===== Soul Processing Complete (ID: {soul_spark.spark_id}) (RootController) =====")
            logger.info(f"Total time for soul: {process_summary['total_duration_seconds']:.2f}s. Incarnated={getattr(soul_spark, FLAG_INCARNATED, False)}")


        except Exception as soul_err_main_loop: # Catch errors from any stage
            failed_stage_name_main = current_stage_name # Use the last known stage
            end_time_iso_main = datetime.now().isoformat()
            
            # Determine soul identifier for logging
            soul_identifier_for_log = base_id_str # Default to base_id_str
            if soul_spark and hasattr(soul_spark, 'spark_id') and soul_spark.spark_id:
                soul_identifier_for_log = soul_spark.spark_id
            
            # Use soul_identifier_for_log instead of trying to construct from soul_num
            logger.error(f"RootController: Soul processing (ID/BaseID: {soul_identifier_for_log}) failed at stage '{failed_stage_name_main}': {soul_err_main_loop}", exc_info=True)

            process_summary_fail_main: Dict[str, Any] = {
                'base_id': base_id_str, # Keep base_id for consistency if needed elsewhere
                'soul_id_on_fail': getattr(soul_spark, 'spark_id', None), # Actual soul_id if available
                'success': False,
                'failed_stage': failed_stage_name_main,
                'error': str(soul_err_main_loop),
                'end_time': end_time_iso_main,
                'total_duration_seconds': time.time() - single_soul_start_time
            }
            if soul_spark: process_summary_fail_main['final_soul_state_on_fail'] = soul_spark.get_spark_metrics().get('core', {})


            if soul_spark and VISUALIZATION_ENABLED:
                stage_fail_name_vis_main = f"Failed_At_{failed_stage_name_main.replace(' ','_')}"
                try: visualize_soul_state(soul_spark, stage_fail_name_vis_main, VISUALIZATION_OUTPUT_DIR_ROOT, show=show_visuals)
                except Exception as vis_e_main: logger.error(f"RootController: Failed to visualize failed state: {vis_e_main}")
            
            # Store the failure summary
            single_soul_final_summary = process_summary_fail_main
            
            logger.error(f"===== Soul (ID/BaseID: {soul_identifier_for_log}) Processing FAILED (RootController) =====")
            print(f"\n{'='*20} ERROR ROOT: Soul Failed {'='*20}\n Stage: {failed_stage_name_main}\n Error: {soul_err_main_loop}\n{'='*70}")
            raise soul_err_main_loop


        # --- Final Report for the single soul ---
        logger.info("RootController: Simulation loop finished. Generating final report for the soul...")
        final_report_data_single_soul = {
            'simulation_start_time': simulation_start_iso,
            'simulation_end_time': datetime.now().isoformat(),
            'total_duration_seconds': time.time() - overall_start_time,
            'parameters': {
                'num_souls': 1, # Hardcoded to 1
                'journey_duration_per_sephirah': journey_duration_per_sephirah,
                'grid_size': GRID_SIZE,
                **kwargs
            },
            'soul_result': single_soul_final_summary # Store the summary for the single soul
        }

        class NumpyEncoder(json.JSONEncoder): # Define encoder for saving report
            def default(self, o):
                if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(o)
                elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                    if np.isnan(o): return None
                    if np.isinf(o): return str(o)
                    return round(float(o), 6)
                elif isinstance(o, np.ndarray):
                    cleaned_list = []
                    for item in o.tolist():
                        if isinstance(item, float):
                            if np.isnan(item): cleaned_list.append(None)
                            elif np.isinf(item): cleaned_list.append(str(item))
                            else: cleaned_list.append(round(item, 6))
                        elif isinstance(item, (int, bool, str)) or item is None: cleaned_list.append(item)
                        else: cleaned_list.append(str(item))
                    return cleaned_list
                elif isinstance(o, (datetime, uuid.UUID)): return str(o)
                try: return super().default(o)
                except TypeError: logger.warning(f"NumpyEncoder fallback for type {type(o)}."); return str(o)

        soul_report_filename = f"soul_report_{base_id_str}.json"
        final_report_path_for_soul = os.path.join(report_path_base, soul_report_filename)

        if final_report_path_for_soul:
            try:
                with open(final_report_path_for_soul, 'w') as f:
                    json.dump(final_report_data_single_soul, f, cls=NumpyEncoder, indent=2)
                logger.info(f"RootController: Final report for soul {base_id_str} saved to {final_report_path_for_soul}")
            except Exception as report_err:
                logger.error(f"RootController: Failed save report for soul {base_id_str}: {report_err}", exc_info=True)
                print(f"ERROR ROOT saving report for soul {base_id_str} to {final_report_path_for_soul}")

        print("\n" + "=" * 80 + "\nROOT SIMULATION COMPLETE (Single Soul)\n" + "=" * 80)
        success_status = single_soul_final_summary.get('success', False) if single_soul_final_summary else False
        print(f"Processed 1 soul | Duration: {time.time() - overall_start_time:.2f}s")
        print(f"Success: {'Yes' if success_status else 'No'}")
        if final_report_path_for_soul: print(f"Report: {final_report_path_for_soul}")
        print(f"Visualizations (if enabled) saved to: {VISUALIZATION_OUTPUT_DIR_ROOT}")
        print("=" * 80)
        logger.info("--- RootController: Soul Development Simulation Finished ---")

    except Exception as main_err:
        logger.critical(f"RootController: Simulation aborted due to critical error: {main_err}", exc_info=True)
        print("\n" + "=" * 80 + "\nCRITICAL ROOT ERROR - SIMULATION ABORTED\n" +
              f"Error: {main_err}\nSee log file: {log_file_path}\n" + "=" * 80)
        sys.exit(1)
    finally:
        if METRICS_AVAILABLE:
            try:
                logger.info("RootController: Persisting final metrics...")
                metrics.persist_metrics()
                logger.info("RootController: Final metrics persisted.")
            except Exception as persist_e:
                logger.error(f"RootController: ERROR persisting metrics at end of simulation: {persist_e}")
        logger.info("RootController: Shutting down logging.")
        logging.shutdown()


# --- Main Execution Block ---
if __name__ == "__main__":
    print("DEBUG: Starting main execution block of root_controller.py...")
    try:
        # Parameters for the simulation run (ensure these match expected kwargs in stages if not using defaults)
        simulation_params = {
            "num_souls": 1,
            "journey_duration_per_sephirah": 1.5,
            "report_path_base": os.path.join(DATA_DIR_BASE, "reports", "simulation_runs"),
            "show_visuals": False,

            # --- Stage-specific kwargs from your original root_controller's __main__ ---
            # These will be passed via **kwargs to run_simulation and then to individual stages
            # Spark Harmonization (uses HARMONIZATION_ITERATIONS from constants)
            "harmonization_iterations": HARMONIZATION_ITERATIONS, # Example of passing it explicitly
            # Guff Strengthening (uses GUFF_STRENGTHENING_DURATION from constants)
            "guff_duration": GUFF_STRENGTHENING_DURATION * 0.8,
            # Creator Entanglement (uses CREATOR_POTENTIAL_DEFAULT, EDGE_OF_CHAOS_DEFAULT from constants)
            "base_creator_potential": CREATOR_POTENTIAL_DEFAULT,
            "edge_of_chaos_target": EDGE_OF_CHAOS_DEFAULT,
            # Harmonic Strengthening
            "harmony_intensity": HARMONIC_STRENGTHENING_INTENSITY_DEFAULT * 1.1,
            "harmony_duration_factor": HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT * 0.9,
            # Life Cord
            "life_cord_intensity": 0.85,
            "cord_complexity": LIFE_CORD_COMPLEXITY_DEFAULT * 1.1,
            # Earth Harmonization
            "schumann_intensity": EARTH_HARMONY_INTENSITY_DEFAULT,
            "core_intensity": 0.65,
            # Identity Crystallization
            "specified_name": None, #"Anima_Test_Root", # or None for user input
            "train_cycles": 6,
            "entrainment_bpm": 70.0,
            "entrainment_duration": 75.0,
            "love_cycles": 4,
            "geometry_stages": 4,
            "crystallization_threshold": IDENTITY_CRYSTALLIZATION_THRESHOLD * 0.98,
            # Birth
            "birth_intensity": BIRTH_INTENSITY_DEFAULT * 0.95,
            # "mother_profile": { # Example, will be generated if None and MOTHER_RESONANCE_AVAILABLE=True
            #     'nurturing_capacity': 0.75,
            #     'spiritual': {'connection': 0.65},
            #     'love_resonance': 0.85
            # }
        }

        run_simulation(**simulation_params)
        print("DEBUG: run_simulation completed successfully from __main__.")

    except Exception as e:
        log_func = logger.critical if logger.hasHandlers() else print
        log_func(f"FATAL ERROR in root_controller __main__: {e}", exc_info=True)
        print(f"\nFATAL ERROR in __main__: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("DEBUG: Main execution block of root_controller.py finished.")
        if METRICS_AVAILABLE:
            try:
                print("Persisting metrics from __main__ finally block...")
                metrics.persist_metrics()
                print("Metrics persisted from __main__ finally block.")
            except Exception as persist_e:
                print(f"ERROR persisting metrics from __main__ finally: {persist_e}")
        if logging.getLogger().hasHandlers():
            logging.shutdown()

# --- END OF FILE root_controller.py ---











# # --- START OF FILE root_controller.py ---

# """
# Root Controller (Refactored V4.3.9 - Emergence/Harmonization, Minimal Brain, Visualization)

# Orchestrates the simulation from spark emergence through all stages.
# Initializes SoulSpark via field sampling, harmonizes the spark,
# then proceeds through Guff, Journey, CE, HS, Cord, Earth, Identity, Birth.
# Birth uses a minimal BrainSeed placeholder. Integrates mandatory visualization calls.
# Uses updated units and principle-driven S/C logic. Hard fails on visualization errors.
# """

# import logging
# import os
# import random
# import sys
# import time
# import traceback
# import json
# import uuid
# import numpy as np
# from datetime import datetime
# from math import pi as PI_MATH
# try:
#     from typing import List, Optional, Dict, Any, Tuple
# except ImportError:
#     List = Dict = Any = Optional = Tuple = type(None) # Basic fallback

# # --- Constants Import (CRUCIAL) ---
# try:
#     from constants.constants import *  # noqa: F403 - Allow wildcard import
#     # Add explicit checks for key constants needed here if import * fails
#     if 'LOG_LEVEL' not in locals(): raise NameError("LOG_LEVEL not defined")
#     if 'GRID_SIZE' not in locals(): raise NameError("GRID_SIZE not defined")
#     if 'LOG_FORMAT' not in locals(): raise NameError("LOG_FORMAT not defined")
#     # Add checks for stage default constants if not using kwargs extensively
#     logger_check = logging.getLogger('ConstantsCheck')
#     logger_check.info(f"Constants loaded. LOG_LEVEL={LOG_LEVEL}, GRID_SIZE={GRID_SIZE}")
# except ImportError as e:
#     print(f"FATAL: Could not import constants.constants: {e}")
#     sys.exit(1)
# except NameError as e:
#     print(f"FATAL: Constant definition missing after import: {e}")
#     sys.exit(1)
# except Exception as e:
#     print(f"FATAL: Error loading constants: {e}")
#     sys.exit(1)



# # --- Logger Initialization ---
# logger = logging.getLogger('root_controller')
# log_file_path = os.path.join("logs", "root_controller_run.log")
# if not logger.handlers:
#     log_level_int = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO) # noqa: F405
#     logger.setLevel(log_level_int)
#     log_formatter = logging.Formatter(LOG_FORMAT) # noqa: F405
#     # Terminal Handler
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(logging.INFO) # Keep terminal concise
#     ch.setFormatter(log_formatter)
#     logger.addHandler(ch)
#     # File Handler
#     try:
#         os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
#         fh = logging.FileHandler(log_file_path, mode='w')
#         fh.setLevel(logging.DEBUG) # Capture details
#         fh.setFormatter(log_formatter)
#         logger.addHandler(fh)
#         logger.info("Logging configured (Terminal: INFO, File: DEBUG).")
#     except Exception as log_err:
#         logger.error(f"File logging setup failed: {log_err}", exc_info=True)
#         logger.info("Logging configured (Terminal: INFO, File: Disabled).")
# else:
#     logger.info("Logger 'root_controller' already has handlers.")

# # --- Core Controller & Stage Imports ---
# try:
#     logger.debug("Importing core components...")
#     from stage_1.fields.field_controller import FieldController
#     from stage_1.soul_spark.soul_spark import SoulSpark
#     from stage_1.fields.sephiroth_field import SephirothField
#     # Import stage functions - Adjust paths if structure differs
#     from stage_1.soul_formation.spark_harmonization import perform_spark_harmonization # noqa E501
#     from stage_1.soul_formation.guff_strengthening import perform_guff_strengthening # noqa E501
#     from stage_1.soul_formation.sephiroth_journey_processing import process_sephirah_interaction # noqa E501
#     from stage_1.soul_formation.creator_entanglement import perform_creator_entanglement # noqa E501
#     from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening # noqa E501
#     from stage_1.soul_formation.life_cord import form_life_cord
#     from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization # noqa E501
#     from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization # noqa E501
#     from stage_1.soul_formation.birth import perform_birth # Uses minimal brain
#     logger.debug("Core stages imported successfully.")
# except ImportError as e:
#     logger.critical(f"FATAL: Could not import core component: {e}", exc_info=True) # noqa E501
#     sys.exit(1)
# except Exception as e:
#     logger.critical(f"FATAL: Error during core component imports: {e}", exc_info=True) # noqa E501
#     sys.exit(1)

# try:
#     # Mother resonance import
#     from glyphs.mother.mother_resonance import create_mother_resonance_data
#     MOTHER_RESONANCE_AVAILABLE = True
#     logger.info("Mother resonance module loaded successfully.")
# except ImportError:
#     logger.warning("Mother resonance module not found. Birth will proceed without mother influence.")
#     MOTHER_RESONANCE_AVAILABLE = False

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
#     logger.critical(f"CRITICAL ERROR: Soul visualization module not found: {ie}")
#     logger.critical("Visualizations are required for the simulation. Please install the visualization module.")
#     logger.critical("Aborting simulation.")
#     sys.exit(1)
# except Exception as e:
#     logger.critical(f"CRITICAL ERROR: Error setting up visualization: {e}", exc_info=True)
#     logger.critical("Aborting simulation.")
#     sys.exit(1)

# # --- Metrics Import & Setup ---
# METRICS_AVAILABLE = False
# try:
#     import metrics_tracking as metrics
#     METRICS_AVAILABLE = True
#     logger.info("Metrics tracking module loaded successfully.")
# except ImportError:
#     logger.warning("Metrics tracking module not found. Using placeholder.")
#     class MetricsPlaceholder:
#         @staticmethod
#         def record_metric(*args, **kwargs): pass
#         @staticmethod
#         def record_metrics(*args, **kwargs): pass
#         @staticmethod
#         def persist_metrics(*args, **kwargs): pass
#     metrics = MetricsPlaceholder()
# except Exception as e:
#     logger.critical(f"CRITICAL ERROR metrics import: {e}", exc_info=True)
#     class MetricsPlaceholder:
#         @staticmethod
#         def record_metric(*args, **kwargs): pass
#         @staticmethod
#         def record_metrics(*args, **kwargs): pass
#         @staticmethod
#         def persist_metrics(*args, **kwargs): pass
#     metrics = MetricsPlaceholder()

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

#     max_key_len = max(len(k.replace('_',' ').title()) for k in display_keys) if display_keys else 30 # noqa E501

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
#         elif isinstance(value, list): formatted_val = f"<List ({len(value)} items)>" # noqa E501
#         elif isinstance(value, dict): formatted_val = f"<Dict ({len(value)} keys)>" # noqa E501
#         else: formatted_val = str(value)

#         key_display_cleaned = key_display.replace('_', ' ').title()
#         # Pad key display for alignment
#         print(f"  {key_display_cleaned:<{max_key_len}} : {formatted_val}{unit}")

#     print("=" * (40 + len(stage_name)))


# # --- Spark Emergence Function ---
# def create_spark_from_field(field_controller: FieldController,
#                            base_id: str,
#                            creation_location_coords: tuple[int, int, int]
#                            ) -> SoulSpark:
#     """ Creates SoulSpark via field sampling. Fails hard. """
#     # ... (Implementation unchanged) ...
#     logger.info(f"Attempting spark emergence at {creation_location_coords}...")
#     try:
#         local_props=field_controller.get_properties_at(creation_location_coords) # noqa F821
#         local_eoc=local_props.get('edge_of_chaos',0.5); initial_data={}
#         base_potential=INITIAL_SPARK_ENERGY_SEU # noqa F405
#         field_energy_catalyst=local_props.get('energy_seu',0.0)*SPARK_FIELD_ENERGY_CATALYST_FACTOR # noqa F405
#         eoc_yield_multiplier=1.0+(local_eoc*SPARK_EOC_ENERGY_YIELD_FACTOR) # noqa F405
#         initial_data['energy']=max(INITIAL_SPARK_ENERGY_SEU*0.5, base_potential*eoc_yield_multiplier+field_energy_catalyst) # noqa F405
#         base_freq=local_props.get('frequency_hz',INITIAL_SPARK_BASE_FREQUENCY_HZ) # noqa F405
#         initial_data['frequency']=max(FLOAT_EPSILON, base_freq+np.random.normal(0,base_freq*0.01)) # noqa F405
#         seed_geometry=SPARK_SEED_GEOMETRY # noqa F405
#         seed_ratios=PLATONIC_HARMONIC_RATIOS.get(seed_geometry, [1.0,1.5,2.0,PHI]) # noqa F405
#         initial_data['harmonics']=[initial_data['frequency']*r for r in seed_ratios] # noqa E501
#         local_coh_norm=local_props.get('coherence_cu',VOID_BASE_COHERENCE_CU)/MAX_COHERENCE_CU # noqa F405
#         phase_align=0.5+local_coh_norm*0.4; phase_noise=np.pi*(1.0-phase_align) # noqa E501
#         init_phases=(np.array(seed_ratios)*np.pi+np.random.uniform(-phase_noise,phase_noise,len(seed_ratios)))%(2*np.pi) # noqa E501
#         initial_data['frequency_signature']={'base_frequency':initial_data['frequency'],'frequencies':initial_data['harmonics'],'amplitudes':[1.0/(r**0.7) for r in seed_ratios],'phases':init_phases.tolist()} # noqa E501
#         local_stab_norm=local_props.get('stability_su',VOID_BASE_STABILITY_SU)/MAX_STABILITY_SU # noqa F405
#         local_order=local_props.get('order_factor',0.5); local_pattern=local_props.get('pattern_influence',0.0) # noqa E501
#         initial_data['phi_resonance']=np.clip(SPARK_INITIAL_FACTOR_BASE+local_eoc*SPARK_INITIAL_FACTOR_EOC_SCALE+local_order*SPARK_INITIAL_FACTOR_ORDER_SCALE,0.0,1.0) # noqa F405
#         initial_data['pattern_coherence']=np.clip(SPARK_INITIAL_FACTOR_BASE+local_eoc*SPARK_INITIAL_FACTOR_EOC_SCALE*1.1+local_pattern*SPARK_INITIAL_FACTOR_PATTERN_SCALE,0.0,1.0) # noqa F405
#         initial_data['harmony']=np.clip(SPARK_INITIAL_FACTOR_BASE+local_eoc*SPARK_INITIAL_FACTOR_EOC_SCALE*0.8+(local_stab_norm+local_coh_norm)/2.0*0.1,0.0,1.0) # noqa F405
#         initial_data['creator_alignment']=0.0; initial_data['guff_influence_factor']=0.0; initial_data['cumulative_sephiroth_influence']=0.0; initial_data['creator_connection_strength']=0.0 # noqa E501
#         spark_id=f"Soul_{base_id}_{random.randint(100,999)}"; logger.debug(f"Instantiating SoulSpark {spark_id}...") # noqa E501
#         soul_spark=SoulSpark(initial_data=initial_data, spark_id=spark_id) # noqa F821
#         return soul_spark
#     except Exception as e:
#         logger.critical(f"CRITICAL spark creation at {creation_location_coords}: {e}", exc_info=True) # noqa E501
#         raise RuntimeError(f"Failed create SoulSpark: {e}") from e


# # --- Main Simulation Logic ---
# def run_simulation(num_souls: int = 1,
#                    journey_duration_per_sephirah: float = 2.0,
#                    report_path: str = "output/reports/simulation_report_final.json", # noqa E501
#                    show_visuals: bool = False,
#                    **kwargs
#                    ) -> None:
#     """ Runs the main simulation flow. """
#     logger.info("--- Starting Soul Development Simulation (V4.3.9 with Visualization) ---")
#     logger.info(f"Num Souls: {num_souls}, Sephirah Duration: {journey_duration_per_sephirah}") # noqa E501
#     overall_start_time = time.time()
#     simulation_start_iso = datetime.now().isoformat()
#     all_souls_final_results = {}
#     field_controller: Optional[FieldController] = None
#     output_base_dir = os.path.dirname(report_path) if report_path else "output"
#     visual_save_dir = os.path.join(output_base_dir, "visuals")
    
#     # Setup for capturing development state
#     development_states = {}

#     try:
#         logger.info("Initializing Field Controller...")
#         field_controller = FieldController(grid_size=GRID_SIZE) # noqa F405
#         logger.info("Field Controller initialized.")

#         for i in range(num_souls):
#             soul_num = i + 1
#             base_id_str = (
#                 f"{simulation_start_iso.replace(':','').replace('-','').replace('.','')}" # noqa E501
#                 f"_{soul_num:03d}"
#             )
#             logger.info(f"\n===== Processing Soul {soul_num}/{num_souls} (Base ID: {base_id_str}) =====") # noqa E501
#             single_soul_start_time = time.time()
#             process_summary = {'base_id': base_id_str, 'stages_metrics': {}}
#             current_stage_name = "Pre-Emergence"
#             soul_spark: Optional[SoulSpark] = None

#             try:
#                 # --- Stage 1: Spark Emergence ---
#                 current_stage_name = "Spark Emergence"
#                 logger.info(f"Stage: {current_stage_name}...")
#                 creation_location = field_controller.find_optimal_development_location() # noqa E501
#                 creation_location_coords = field_controller._coords_to_int_tuple(creation_location) # noqa E501
#                 soul_spark = create_spark_from_field( # noqa F821
#                     field_controller, base_id_str, creation_location_coords
#                 ) # Fails hard
#                 soul_spark.position = creation_location
#                 soul_spark.current_field_key = 'void'
#                 process_summary['soul_id'] = soul_spark.spark_id
#                 logger.info(f"Soul Spark {soul_spark.spark_id} emerged at {creation_location_coords}.") # noqa E501
#                 emergence_metrics = {'success': True, **soul_spark.get_spark_metrics()['core']} # noqa F821
#                 process_summary['stages_metrics'][current_stage_name] = emergence_metrics # noqa E501
#                 display_stage_metrics(current_stage_name, emergence_metrics)
                
#                 # Essential visualization - Hard fails if visualization fails
#                 try:
#                     visualize_soul_state(soul_spark, "Spark_Emergence", visual_save_dir, show=show_visuals)
#                     # Store state for final comparison
#                     if soul_spark.spark_id not in development_states:
#                         development_states[soul_spark.spark_id] = []
#                     development_states[soul_spark.spark_id].append((soul_spark, "Spark_Emergence"))
#                 except Exception as vis_err:
#                     logger.critical(f"CRITICAL: Visualization failed at {current_stage_name}: {vis_err}")
#                     raise RuntimeError(f"Visualization error (required for simulation): {vis_err}")

#                 # --- Stage 2: Spark Harmonization ---
#                 current_stage_name = "Spark Harmonization"
#                 logger.info(f"Stage: {current_stage_name}...")
#                 _, harm_metrics = perform_spark_harmonization(soul_spark) # Fails hard # noqa F821
#                 process_summary['stages_metrics'][current_stage_name] = harm_metrics # noqa E501
#                 display_stage_metrics(current_stage_name, harm_metrics)
#                 logger.info("Spark Harmonization complete.")

#                 # --- Stage 3: Move to Guff ---
#                 current_stage_name="Move to Guff"
#                 logger.info(f"Stage: {current_stage_name}...")
#                 field_controller.place_soul_in_guff(soul_spark) # noqa F821
#                 logger.info("Soul moved to Guff.")
#                 display_stage_metrics(current_stage_name, {'success': True})

#                 # --- Stage 4: Guff Strengthening ---
#                 current_stage_name="Guff Strengthening"
#                 logger.info(f"Stage: {current_stage_name}...")
#                 _, guff_metrics = perform_guff_strengthening( # noqa F821
#                     soul_spark, field_controller, duration=GUFF_STRENGTHENING_DURATION # noqa F405
#                 )
#                 process_summary['stages_metrics'][current_stage_name]=guff_metrics
#                 display_stage_metrics(current_stage_name, guff_metrics)
#                 logger.info("Guff Strengthening complete.")

#                 # --- Stage 5: Release from Guff ---
#                 current_stage_name="Release from Guff"
#                 logger.info(f"Stage: {current_stage_name}...")
#                 field_controller.release_soul_from_guff(soul_spark) # noqa F821
#                 logger.info("Soul released from Guff.")
#                 display_stage_metrics(current_stage_name, {'success': True})

#                 # --- Stage 6: Sephiroth Journey ---
#                 current_stage_name = "Sephiroth Journey"
#                 logger.info(f"Stage: {current_stage_name}...")
#                 journey_path = ["kether", "chokmah", "binah", "daath", "chesed",
#                                 "geburah", "tiphareth", "netzach", "hod",
#                                 "yesod", "malkuth"]
#                 journey_step_metrics = {}
#                 for sephirah_name in journey_path:
#                     stage_id = f"Interaction ({sephirah_name.capitalize()})"
#                     logger.info(f"  Entering {sephirah_name.capitalize()}...")
#                     sephirah_influencer = field_controller.get_field(sephirah_name) # noqa F821
#                     if not sephirah_influencer or not isinstance(sephirah_influencer, SephirothField): # noqa F821
#                         raise RuntimeError(f"SephirothField missing for '{sephirah_name}'.") # noqa E501
#                     _, step_metrics = process_sephirah_interaction( # noqa F821
#                         soul_spark, sephirah_influencer, field_controller,
#                         journey_duration_per_sephirah
#                     ) # Fails hard
#                     journey_step_metrics[sephirah_name] = step_metrics
#                     display_stage_metrics(stage_id, step_metrics)
#                     logger.info(f"  Exiting {sephirah_name.capitalize()}.")
#                 setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, True) # noqa F405
#                 setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, True) # noqa F405
#                 process_summary['stages_metrics']['sephiroth_journey_steps'] = journey_step_metrics # noqa E501
#                 logger.info("Sephiroth Journey complete.")
                
#                 # Essential visualization - Hard fails if visualization fails
#                 try:
#                     visualize_soul_state(soul_spark, "Sephiroth_Journey_End", visual_save_dir, show=show_visuals)
#                     # Store state for final comparison
#                     if soul_spark.spark_id not in development_states:
#                         development_states[soul_spark.spark_id] = []
#                     development_states[soul_spark.spark_id].append((soul_spark, "Sephiroth_Journey_End"))
#                 except Exception as vis_err:
#                     logger.critical(f"CRITICAL: Visualization failed at {current_stage_name}: {vis_err}")
#                     raise RuntimeError(f"Visualization error (required for simulation): {vis_err}")

#                 # --- Stage 7: Creator Entanglement ---
#                 current_stage_name = "Creator Entanglement"
#                 logger.info(f"Stage: {current_stage_name}...")
#                 kether_influencer = field_controller.kether_field # noqa F821
#                 if not kether_influencer: raise RuntimeError("Kether field unavailable.") # Hard Fail # noqa E501
#                 _, entanglement_metrics = perform_creator_entanglement( # noqa F821
#                     soul_spark, kether_influencer
#                 ) # Fails hard
#                 process_summary['stages_metrics'][current_stage_name] = entanglement_metrics # noqa E501
#                 display_stage_metrics(current_stage_name, entanglement_metrics)
#                 logger.info("Creator Entanglement complete.")
#                 logger.info(f"STATE PRE-HS: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}") # noqa E501

#                 # --- Stage 8: Harmonic Strengthening ---
#                 hs_stage_name = FLAG_HARMONICALLY_STRENGTHENED.replace('_',' ').title() # noqa F405
#                 current_stage_name = hs_stage_name
#                 logger.info(f"Stage: {current_stage_name}...")
#                 hs_intensity = kwargs.get('hs_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT) # noqa F405
#                 hs_duration = kwargs.get('hs_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT) # noqa F405
#                 _, hs_metrics = perform_harmonic_strengthening( # noqa F821
#                     soul_spark, intensity=hs_intensity, duration_factor=hs_duration
#                 ) # Fails hard
#                 process_summary['stages_metrics'][current_stage_name] = hs_metrics
#                 display_stage_metrics(current_stage_name, hs_metrics)
#                 logger.info("Harmonic Strengthening complete.")
#                 logger.info(f"STATE PRE-LC: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}") # noqa E501

#                 # --- Stage 9: Life Cord Formation ---
#                 lc_stage_name = FLAG_CORD_FORMATION_COMPLETE.replace('_',' ').title()
#                 current_stage_name = lc_stage_name
#                 logger.info(f"Stage: {current_stage_name}...")

#                 # Strict intensity calculation - will raise error if attributes missing
#                 if not hasattr(soul_spark, 'stability') or not hasattr(soul_spark, 'coherence'):
#                     raise AttributeError("Soul missing stability or coherence for life cord intensity calculation")

#                 if soul_spark.stability <= 0 or soul_spark.coherence <= 0 or MAX_STABILITY_SU <= 0 or MAX_COHERENCE_CU <= 0:
#                     raise ValueError(f"Invalid stability or coherence values. Stability: {soul_spark.stability}, Coherence: {soul_spark.coherence}, MAX_STABILITY_SU: {MAX_STABILITY_SU}, MAX_COHERENCE_CU: {MAX_COHERENCE_CU}")

#                 # Strict, no-fallback intensity calculation
#                 normalized_stability = soul_spark.stability / MAX_STABILITY_SU
#                 normalized_coherence = soul_spark.coherence / MAX_COHERENCE_CU
#                 lc_intensity = min(1.0, (normalized_stability * 0.6 + normalized_coherence * 0.4))

#                 # Strict complexity calculation
#                 if not all(hasattr(soul_spark, attr) for attr in ['pattern_coherence', 'phi_resonance', 'harmony']):
#                     raise AttributeError("Soul missing required attributes for life cord complexity calculation")

#                 if any(getattr(soul_spark, attr, 0) <= 0 for attr in ['pattern_coherence', 'phi_resonance', 'harmony']):
#                     raise ValueError(f"Invalid attribute values. Pattern Coherence: {soul_spark.pattern_coherence}, Phi Resonance: {soul_spark.phi_resonance}, Harmony: {soul_spark.harmony}")

#                 lc_complexity = min(1.0, (
#                     soul_spark.pattern_coherence * 0.4 + 
#                     soul_spark.phi_resonance * 0.4 + 
#                     soul_spark.harmony * 0.2
#                 ))

#                 # No kwargs fallback - will raise error if not explicitly provided
#                 lc_intensity = kwargs['life_cord_intensity'] if 'life_cord_intensity' in kwargs else lc_intensity
#                 lc_complexity = kwargs['life_cord_complexity'] if 'life_cord_complexity' in kwargs else lc_complexity

#                 _, lc_metrics = form_life_cord(
#                     soul_spark, 
#                     intensity=lc_intensity, 
#                     complexity=lc_complexity
#                 ) # noqa F821

#                 # --- Stage 10: Earth Harmonization ---
#                 def _validate_float(value, name):
#                     if not isinstance(value, (int, float)):
#                         raise TypeError(f"{name} must be a number")
#                     if value <= 0:
#                         raise ValueError(f"{name} must be positive")

#                 # Strict earth resonance calculation
#                 _validate_float(soul_spark.stability, "Stability")
#                 _validate_float(MAX_STABILITY_SU, "MAX_STABILITY_SU")
#                 _validate_float(getattr(soul_spark, 'earth_resonance', 0), "Earth Resonance")

#                 normalized_earth_resonance = getattr(soul_spark, 'earth_resonance')
#                 normalized_stability = soul_spark.stability / MAX_STABILITY_SU
#                 earth_intensity = min(1.0, (normalized_earth_resonance * 0.6 + normalized_stability * 0.4))

#                 # Strict intensity override
#                 earth_intensity = kwargs['earth_intensity'] if 'earth_intensity' in kwargs else earth_intensity
#                 schumann_intensity = kwargs.get('schumann_intensity', earth_intensity)
#                 core_intensity = kwargs.get('core_intensity', earth_intensity)

#                 _, eh_metrics = perform_earth_harmonization(
#                     soul_spark, 
#                     schumann_intensity=schumann_intensity,
#                     core_intensity=core_intensity
#                 )

#                 # --- Stage 11: Identity Crystallization ---
#                 # Strict kwargs requirement
#                 required_id_kwargs = [
#                     'train_cycles', 
#                     'entrainment_bpm', 
#                     'entrainment_duration', 
#                     'love_cycles', 
#                     'geometry_stages', 
#                     'crystallization_threshold'
#                 ]

#                 # Check that all required kwargs are present
#                 missing_kwargs = [k for k in required_id_kwargs if k not in kwargs]
#                 if missing_kwargs:
#                     raise ValueError(f"Missing required kwargs for Identity Crystallization: {missing_kwargs}")

#                 _, id_metrics = perform_identity_crystallization(soul_spark, **{
#                     k: kwargs[k] for k in required_id_kwargs
#                 })

#                 # --- Stage 12: Birth Process ---
#                 # Strict mother profile and intensity validation
#                 if 'birth_intensity' not in kwargs:
#                     raise ValueError("birth_intensity must be explicitly provided for birth")

#                 birth_intensity = kwargs['birth_intensity']
#                 _validate_float(birth_intensity, "Birth Intensity")

#                 # Strict mother profile requirement
#                 if not MOTHER_RESONANCE_AVAILABLE and 'mother_profile' not in kwargs:
#                     raise ValueError("Mother profile required when mother resonance module is unavailable")

#                 birth_mother_profile = kwargs.get('mother_profile')
#                 if not birth_mother_profile and MOTHER_RESONANCE_AVAILABLE:
#                     mother_resonance_data = create_mother_resonance_data()
#                     # Strict validation of mother resonance data
#                     if not mother_resonance_data:
#                         raise ValueError("Failed to generate mother resonance profile")
                    
#                     birth_mother_profile = {
#                         'nurturing_capacity': mother_resonance_data.get('nurturing_capacity', 
#                                                 mother_resonance_data.get('teaching', {}).get('nurturing')),
#                         'spiritual': mother_resonance_data.get('spiritual', {'connection': 0.5}),
#                         'love_resonance': mother_resonance_data.get('love_resonance')
#                     }
                    
#                     # Validate each profile component
#                     for key, value in birth_mother_profile.items():
#                         if value is None:
#                             raise ValueError(f"Missing required mother profile component: {key}")

#                 _, birth_metrics = perform_birth(
#                     soul_spark,
#                     intensity=birth_intensity,
#                     mother_profile=birth_mother_profile
#                 )
#                 process_summary['stages_metrics'][current_stage_name] = birth_metrics # noqa E501
#                 display_stage_metrics(current_stage_name, birth_metrics)
#                 logger.info("Birth Process complete.")
                
#                 # --- Final State Visualization (Post-Birth) ---
#                 try:
#                     visualize_soul_state(soul_spark, "Post_Birth", visual_save_dir, show=show_visuals)
#                     # Add to development states and create final comparison
#                     if soul_spark.spark_id in development_states:
#                         development_states[soul_spark.spark_id].append((soul_spark, "Post_Birth"))
#                         try:
#                             compare_path = visualize_state_comparison(
#                                 development_states[soul_spark.spark_id],
#                                 visual_save_dir,
#                                 show=show_visuals
#                             )
#                             logger.info(f"Final development comparison created: {compare_path}")
#                         except Exception as comp_err:
#                             logger.error(f"Failed to create final development comparison: {comp_err}")
#                 except Exception as vis_err:
#                     logger.error(f"Post-Birth visualization failed: {vis_err}")
#                     # Not raising error here as this is after the critical birth process

#                 # --- Completion ---
#                 final_metrics = soul_spark.get_spark_metrics() # noqa F821
#                 process_summary['final_soul_state'] = final_metrics
#                 process_summary['success'] = True
#                 process_summary['end_time'] = datetime.now().isoformat()
#                 process_summary['total_duration_seconds'] = time.time() - single_soul_start_time # noqa E501
#                 all_souls_final_results[soul_spark.spark_id] = process_summary
#                 logger.info(f"===== Soul {soul_num}/{num_souls} (ID: {soul_spark.spark_id}) Processing Complete =====") # noqa E501
#                 logger.info(f"Total time: {process_summary['total_duration_seconds']:.2f}s. Incarnated={getattr(soul_spark, FLAG_INCARNATED, False)}") # noqa E501 F405

#             except Exception as soul_err:
#                 failed_stage_name = current_stage_name
#                 end_time_iso = datetime.now().isoformat()
#                 logger.error(f"Soul processing failed at stage '{failed_stage_name}': {soul_err}", exc_info=True) # noqa E501
#                 process_summary['success'] = False
#                 process_summary['failed_stage'] = failed_stage_name
#                 process_summary['error'] = str(soul_err)
#                 process_summary['end_time'] = end_time_iso
#                 process_summary['total_duration_seconds'] = time.time() - single_soul_start_time # noqa E501
#                 if soul_spark:
#                     try:
#                         process_summary['final_soul_state_on_fail'] = soul_spark.get_spark_metrics() # noqa F821
#                     except NameError as metrics_err:
#                         # Handle case where ENERGY_UNSCALE_FACTOR might still be missing
#                         logger.error(f"Could not get final metrics on fail: {metrics_err}") # noqa E501
#                         process_summary['final_soul_state_on_fail'] = {"error": "Metrics failed"} # noqa E501
#                 # Visualization on Fail
#                 if soul_spark and VISUALIZATION_ENABLED: # noqa F821
#                     stage_fail_name = f"Failed_At_{failed_stage_name.replace(' ','_')}" # noqa E501
#                     try:
#                         visualize_soul_state(soul_spark, stage_fail_name, visual_save_dir, show=show_visuals) # noqa F821
#                     except Exception as vis_e:
#                         logger.error(f"Failed to visualize failed state: {vis_e}")
#                 final_id = getattr(soul_spark, 'spark_id', base_id_str)
#                 all_souls_final_results[final_id] = process_summary
#                 logger.error(f"===== Soul {soul_num}/{num_souls} (ID: {final_id}) Processing FAILED =====") # noqa E501
#                 print(f"\n{'='*20} ERROR: Soul Failed {'='*20}\n Stage: {failed_stage_name}\n Error: {soul_err}\n{'='*70}") # noqa E501
#                 raise soul_err # Hard fail simulation loop

#         # --- Final Report ---
#         logger.info("Simulation loop finished. Generating final report...")
#         final_report_data = {
#             'simulation_start_time': simulation_start_iso,
#             'simulation_end_time': datetime.now().isoformat(),
#             'total_duration_seconds': time.time() - overall_start_time,
#             'parameters': {
#                 'num_souls': num_souls,
#                 'journey_duration_per_sephirah': journey_duration_per_sephirah,
#                 'grid_size': GRID_SIZE, # noqa F405
#                 **kwargs
#             },
#             'souls_processed': len(all_souls_final_results),
#             'results_per_soul': all_souls_final_results
#         }

#         # Define NumpyEncoder class locally before use
#         class NumpyEncoder(json.JSONEncoder):
#             def default(self, o):
#                 if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
#                                 np.int16, np.int32, np.int64, np.uint8,
#                                 np.uint16, np.uint32, np.uint64)):
#                     return int(o)
#                 elif isinstance(o, (np.float_, np.float16, np.float32,
#                                   np.float64)): # Use np.float_ alias
#                     if np.isnan(o): return None
#                     if np.isinf(o): return str(o)
#                     return round(float(o), 6)
#                 elif isinstance(o, np.ndarray):
#                     cleaned_list = []
#                     for item in o.tolist():
#                         if isinstance(item, float):
#                             if np.isnan(item): cleaned_list.append(None)
#                             elif np.isinf(item): cleaned_list.append(str(item))
#                             else: cleaned_list.append(round(item, 6))
#                         elif isinstance(item, (int, bool, str)) or item is None:
#                             cleaned_list.append(item)
#                         else: cleaned_list.append(str(item))
#                     return cleaned_list
#                 elif isinstance(o, (datetime, uuid.UUID)):
#                     return str(o)
#                 try:
#                     return super().default(o)
#                 except TypeError:
#                     logger.warning(f"NumpyEncoder fallback for type {type(o)}.")
#                     return str(o)

#         # Save report
#         if report_path:
#             try:
#                 report_dir = os.path.dirname(report_path)
#                 if report_dir: os.makedirs(report_dir, exist_ok=True)
#                 with open(report_path, 'w') as f:
#                     json.dump(final_report_data, f, cls=NumpyEncoder, indent=2)
#                 logger.info(f"Final report saved to {report_path}")
#             except Exception as report_err:
#                 logger.error(f"Failed save report: {report_err}", exc_info=True)
#                 print(f"ERROR saving report to {report_path}")

#         # Display Final Summary
#         print("\n" + "=" * 80)
#         print("SIMULATION COMPLETE")
#         total_processed = len(all_souls_final_results)
#         successful_souls = sum(1 for res in all_souls_final_results.values()
#                                if res.get('success'))
#         failed_souls = total_processed - successful_souls
#         print(f"Processed {num_souls} souls | Duration: {time.time() - overall_start_time:.2f}s") # noqa E501
#         print(f"Success: {successful_souls}/{total_processed} | Failed: {failed_souls}/{total_processed}") # noqa E501
#         if report_path: print(f"Report: {report_path}")
        
#         # Visualization summary
#         print("\nVisualization Summary:")
#         print(f"- Visualizations saved to: {visual_save_dir}")
#         print(f"- Soul state data saved to: {os.path.join('output', 'completed')}")
#         print("- Key visualization points: Spark Emergence, Sephiroth Journey End, Identity Crystallization, Pre-Birth, Post-Birth")
#         print("- Development comparisons created for each soul's journey")
        
#         print("=" * 80)
#         logger.info("--- Soul Development Simulation Finished ---")

#     # Main exception block for the simulation run
#     except Exception as main_err:
#         logger.critical(f"Simulation aborted: {main_err}", exc_info=True)
#         print("\n"+"="*80+"\nCRITICAL ERROR - SIMULATION ABORTED\n" +
#               f"Error: {main_err}\nSee log.\n"+"="*80)
#         sys.exit(1)


# # --- Main Execution Block ---
# if __name__ == "__main__":
#     print("DEBUG: Starting main execution block of root_controller...")
#     try:
#         # --- Simulation Parameters ---
#         simulation_params = {
#             "num_souls": 1,
#             "journey_duration_per_sephirah": 1.5,
#             "report_path": "output/reports/simulation_report_minimal_brain_v1.json",
#             "show_visuals": False,
            
#             # REQUIRED explicit overrides
#             "life_cord_intensity": 0.8,
#             "life_cord_complexity": 0.7,
            
#             "earth_intensity": 0.75,
#             "schumann_intensity": 0.7,
#             "core_intensity": 0.8,
            
#             "train_cycles": 5,
#             "entrainment_bpm": 68.0,
#             "entrainment_duration": 100.0,
#             "love_cycles": 4,
#             "geometry_stages": 3,
#             "crystallization_threshold": 0.85,
            
#             "birth_intensity": 0.9,
#             "mother_profile": {
#                 'nurturing_capacity': 0.7,
#                 'spiritual': {'connection': 0.6},
#                 'love_resonance': 0.8
#             }
#         }
#         try:
#             run_simulation(**simulation_params)
#             print("DEBUG: run_simulation completed.")
#         except Exception as e:
#             log_func = logger.critical if logger.handlers else print
#             log_func(f"FATAL ERROR in main execution: {e}", exc_info=True)
#             print(f"\nFATAL ERROR: {type(e).__name__}: {e}")
#             traceback.print_exc()
#             sys.exit(1)
#         finally:
#             print("DEBUG: Main execution block finished.")
#             if METRICS_AVAILABLE:
#                 try:
#                     print("Persisting metrics...")
#                     metrics.persist_metrics() # noqa F821
#                     print("Metrics persisted.")
#                 except Exception as persist_e:
#                     logger.error(f"ERROR persisting metrics: {persist_e}")
#                     print(f"ERROR persisting metrics: {persist_e}")
#             logging.shutdown()
#     except Exception as e:
#         print(f"Failed to initialize simulation parameters: {e}")
#         sys.exit(1)
# # --- END OF FILE root_controller.py ---
