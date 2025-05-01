# --- START OF FILE root_controller.py ---

"""
Root Controller (Refactored V4.3.7 - Emergence/Harmonization)

Orchestrates the simulation from spark emergence through all stages.
Initializes SoulSpark via field sampling, harmonizes the spark,
then proceeds through Guff, Journey, and Completion stages.
Uses updated units and principle-driven S/C logic where implemented.
Adheres strictly to PEP 8 formatting. Assumes `from constants.constants import *`.
"""

import logging
import os
import random
import sys
import time
import traceback
import json
import uuid
import numpy as np
from datetime import datetime
from math import pi as PI_MATH # Use specific alias to avoid conflict
try:
    from typing import List, Optional, Dict, Any, Tuple
except ImportError:
    List = Dict = Any = Optional = Tuple = None # Compatibility fallback

# --- Setup Project Root Path ---
# (Keep your existing path setup logic here if needed)
# Example:
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # Adjust levels as needed
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print(f"DEBUG: Added {project_root} to sys.path")

# --- Early Print Statement ---
print("DEBUG: Project path setup attempted. Starting core imports...")

# --- Constants Import (CRUCIAL) ---
try:
    from constants.constants import *
    print(f"DEBUG: Constants loaded. LOG_LEVEL={LOG_LEVEL}, GRID_SIZE={GRID_SIZE}, "
          f"MAX_SOUL_ENERGY_SEU={MAX_SOUL_ENERGY_SEU}")
except ImportError as e:
    print(f"FATAL: Could not import constants.constants: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Error loading constants: {e}")
    sys.exit(1)

# --- Core Controller & Stage Imports ---
try:
    print("DEBUG: Importing FieldController...")
    from stage_1.fields.field_controller import FieldController
    print("DEBUG: Importing SoulSpark...")
    from stage_1.soul_spark.soul_spark import SoulSpark
    print("DEBUG: Importing SephirothField...")
    from stage_1.fields.sephiroth_field import SephirothField
    print("DEBUG: Importing Stage Functions...")
    # Import ALL stage functions
    from stage_1.soul_formation.spark_harmonization import perform_spark_harmonization # NEW
    from stage_1.soul_formation.guff_strengthening import perform_guff_strengthening
    from stage_1.soul_formation.sephiroth_journey_processing import process_sephirah_interaction
    from stage_1.soul_formation.creator_entanglement import run_full_entanglement_process
    from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening
    from stage_1.soul_formation.life_cord import form_life_cord
    from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization
    from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization
    from stage_1.soul_formation.birth import perform_birth
    print("DEBUG: Core stages imported.")
except ImportError as e:
    print(f"FATAL: Could not import core components: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Error during core component imports: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Logger Initialization ---
logger = logging.getLogger('root_controller')
if not logger.hasHandlers():
    try: logger.setLevel(LOG_LEVEL)
    except NameError: logger.setLevel(logging.INFO) # Fallback
    log_formatter = logging.Formatter(LOG_FORMAT) # Assumes LOG_FORMAT exists
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)
    try: # File handler
        log_file_path = os.path.join("logs", "root_controller_run.log")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
        logger.info("Logging configured.")
    except Exception as log_err:
        logger.error(f"File logging disabled: {log_err}")
else:
    logger.info("Logger already configured.")

# --- Check for Mother Glyph ---
MOTHER_GLYPH_AVAILABLE = False
MOTHER_GLYPH_PATH = None
try:
    # Use path relative to project structure if constants defines it
    # Or hardcode here if necessary
    encoded_glyph_path = os.path.join(
        "glyphs", "glyph_resonance", "encoded_glyphs", "encoded_mother_sigil.jpeg"
    )
    if os.path.exists(encoded_glyph_path):
        MOTHER_GLYPH_PATH = encoded_glyph_path
        MOTHER_GLYPH_AVAILABLE = True
        logger.info("Mother glyph found.")
    else:
        logger.warning(f"Mother glyph not found at: {encoded_glyph_path}")
except Exception as e:
    logger.error(f"Error checking mother glyph: {e}")

# --- Metrics Import ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
    logger.info("Metrics tracking module loaded successfully.")
except ImportError:
    logger.warning("Metrics tracking module not found. Using placeholder.")
    class MetricsPlaceholder:
        record_metric = lambda *a, **kw: None
        record_metrics = lambda *a, **kw: None
        get_category_metrics = lambda *a, **kw: {}
        get_all_metrics = lambda *a, **kw: {}
        analyze_metrics = lambda *a, **kw: None
    metrics = MetricsPlaceholder()
    METRICS_AVAILABLE = False
except Exception as e:
    logger.critical(f"CRITICAL ERROR during metrics import: {e}", exc_info=True)
    # Use placeholder on critical error
    class MetricsPlaceholder: # Repeat placeholder definition
        record_metric = lambda *a, **kw: None
        record_metrics = lambda *a, **kw: None
        get_category_metrics = lambda *a, **kw: {}
        get_all_metrics = lambda *a, **kw: {}
        analyze_metrics = lambda *a, **kw: None
    metrics = MetricsPlaceholder()
    METRICS_AVAILABLE = False

# --- Helper Function for Metrics Display ---
def display_stage_metrics(stage_name, metrics_dict):
    """Prints a formatted summary of stage metrics."""
    # (Implementation from V4.3.6)
    print(f"\n{'='*20} {stage_name} Metrics Summary {'='*20}")
    if not metrics_dict: print("  No metrics captured."); print("="*(40+len(stage_name))); return
    success = metrics_dict.get('success');
    if success is not None: print(f"  Success: {success}");
    if not success: print(f"  Error: {metrics_dict.get('error','Unknown')}"); print(f"  Failed Step: {metrics_dict.get('failed_step','N/A')}")
    else:
        if 'error' not in metrics_dict: print("  Success: True (implied)")
        else: print(f"  Success: False (implied by error: {metrics_dict.get('error')})"); print(f"  Failed Step: {metrics_dict.get('failed_step','N/A')}")
    skip_keys = {'success','error','failed_step','action','soul_id','start_time','end_time','timestamp','duration_seconds','initial_state','final_state','guff_properties_used','imparted_aspect_strengths','aspects_touched_names','initial_state_changes','geometric_changes','local_entanglement_changes','element_details','cycle_details','components','missing_attributes','gained_aspect_names','strengthened_aspect_names','transfers','memory_retentions','layer_formation_changes','imparted_aspect_strengths_summary'}
    display_keys = sorted([str(k) for k in metrics_dict.keys() if str(k) not in skip_keys])
    for key in display_keys:
        value=metrics_dict[key]; unit=""; formatted_val=""
        if key.endswith('_seu'): unit=" SEU"; key_display=key[:-4]
        elif key.endswith('_su'): unit=" SU"; key_display=key[:-3]
        elif key.endswith('_cu'): unit=" CU"; key_display=key[:-3]
        elif key.endswith('_hz'): unit=" Hz"; key_display=key[:-3]
        elif key.endswith('_factor'): unit=""; key_display=key[:-7]
        elif key.endswith('_score'): unit=""; key_display=key[:-6]
        elif key.endswith('_level'): unit=""; key_display=key[:-6]
        elif key.endswith('_pct'): unit="%"; key_display=key[:-4]
        elif key.endswith('_count'): unit=""; key_display=key[:-6]
        else: key_display=key
        if isinstance(value,float):
            if unit in [" SU"," CU"," Hz"]: formatted_val=f"{value:.1f}"
            elif unit == " SEU": formatted_val=f"{value:.2f}"
            elif unit == "%": formatted_val=f"{value:.1f}"
            else: formatted_val=f"{value:.3f}"
        elif isinstance(value,int) and unit=="": formatted_val=str(value)
        elif isinstance(value,bool): formatted_val=str(value)
        elif isinstance(value,list): formatted_val = f"<List ({len(value)} items)>" if len(value)>5 else str(value)
        elif isinstance(value,dict): formatted_val=f"<Dict ({len(value)} keys)>"
        else: formatted_val=str(value)
        print(f"  {key_display.replace('_',' ').title():<30}: {formatted_val}{unit}")
    print("="*(40+len(stage_name)))


# --- Spark Emergence Function ---
def create_spark_from_field(field_controller: FieldController,
                           base_id: str,
                           creation_location_coords: Tuple[int, int, int]
                           ) -> SoulSpark:
    """
    Creates a SoulSpark by sampling field properties at a location,
    representing an energetic emergence event influenced by local conditions.
    """
    logger.info(f"Attempting spark emergence at {creation_location_coords}...")
    try:
        # 1. Get Local Field Properties at Emergence Point
        local_props = field_controller.get_properties_at(creation_location_coords)
        local_eoc = local_props.get('edge_of_chaos', 0.5) # Get EoC value here

        # 2. Derive Initial SoulSpark Data from Field Properties
        initial_data = {}

        # --- Energy: Base Potential + Field Catalyst + EoC Yield ---
        base_potential = INITIAL_SPARK_ENERGY_SEU # Base potential
        field_energy_catalyst = local_props.get('energy_seu', 0.0) * SPARK_FIELD_ENERGY_CATALYST_FACTOR
        # EoC determines the yield of converting potential to actual energy
        eoc_yield_multiplier = 1.0 + (local_eoc * SPARK_EOC_ENERGY_YIELD_FACTOR)
        initial_data['energy'] = base_potential * eoc_yield_multiplier + field_energy_catalyst
        # Ensure minimum energy, but allow exceeding MAX initially (will be clamped in __init__)
        initial_data['energy'] = max(INITIAL_SPARK_ENERGY_SEU * 0.5, initial_data['energy'])
        logger.debug(f"  Emergence Energy: {initial_data['energy']:.1f} SEU "
                     f"(YieldMult={eoc_yield_multiplier:.2f})")

        # --- Frequency & Harmonics: From Field + Seed Geometry ---
        base_freq = local_props.get('frequency_hz', INITIAL_SPARK_BASE_FREQUENCY_HZ)
        # Add small random variation around local field frequency
        initial_data['frequency'] = max(FLOAT_EPSILON,
                                      base_freq + np.random.normal(0, base_freq * 0.01)) # 1% variation
        # Seed harmonics from a base geometry defined in constants
        seed_geometry = SPARK_SEED_GEOMETRY
        seed_ratios = PLATONIC_HARMONIC_RATIOS.get(seed_geometry)
        if seed_ratios is None:
             logger.warning(f"Seed geometry '{seed_geometry}' not found in PLATONIC_HARMONIC_RATIOS. Using default.")
             seed_ratios = [1.0, 1.5, 2.0, PHI] # Default if not found
        initial_data['harmonics'] = [initial_data['frequency'] * r for r in seed_ratios]
        # Generate slightly ordered phases based on these ratios and local coherence
        local_coherence_norm = local_props.get('coherence_cu', VOID_BASE_COHERENCE_CU) / MAX_COHERENCE_CU
        phase_alignment_factor = 0.5 + local_coherence_norm * 0.4 # Map 0-1 coherence to 0.5-0.9 alignment
        phase_noise_mag = np.pi * (1.0 - phase_alignment_factor) # Less noise if more coherent
        initial_phases = (np.array(seed_ratios) * np.pi +
                          np.random.uniform(-phase_noise_mag, phase_noise_mag, len(seed_ratios)))
        initial_phases = (initial_phases % (2 * np.pi)).tolist() # Wrap phases 0-2pi
        initial_data['frequency_signature'] = {
            'base_frequency': initial_data['frequency'],
            'frequencies': initial_data['harmonics'],
            'amplitudes': [1.0 / (r**0.7) for r in seed_ratios], # Example amplitude falloff
            'phases': initial_phases
        }
        logger.debug(f"  Emergence Freq: {initial_data['frequency']:.1f} Hz, "
                     f"Harmonics from: {seed_geometry}, Phases alignment: {phase_alignment_factor:.2f}")

        # --- Initial Factors: Seeded by Local Field Order/EoC ---
        local_stability_norm = local_props.get('stability_su', VOID_BASE_STABILITY_SU) / MAX_STABILITY_SU
        local_order = local_props.get('order_factor', 0.5)
        local_pattern = local_props.get('pattern_influence', 0.0)
        # Use constants for scaling influence
        initial_data['phi_resonance'] = np.clip(SPARK_INITIAL_FACTOR_BASE +
                                                local_eoc * SPARK_INITIAL_FACTOR_EOC_SCALE +
                                                local_order * SPARK_INITIAL_FACTOR_ORDER_SCALE, 0.0, 1.0)
        initial_data['pattern_coherence'] = np.clip(SPARK_INITIAL_FACTOR_BASE +
                                                    local_eoc * SPARK_INITIAL_FACTOR_EOC_SCALE * 1.1 + # Slightly more boost from EoC?
                                                    local_pattern * SPARK_INITIAL_FACTOR_PATTERN_SCALE, 0.0, 1.0)
        initial_data['harmony'] = np.clip(SPARK_INITIAL_FACTOR_BASE +
                                          local_eoc * SPARK_INITIAL_FACTOR_EOC_SCALE * 0.8 + # Slightly less boost from EoC?
                                          (local_stability_norm + local_coherence_norm) / 2.0 * 0.1, 0.0, 1.0)
        # Set other specific factors to 0 initially unless seeded differently
        initial_data['creator_alignment'] = 0.0
        initial_data['guff_influence_factor'] = 0.0
        initial_data['cumulative_sephiroth_influence'] = 0.0
        initial_data['creator_connection_strength'] = 0.0
        # *** REMOVED LOOP for _GEOM_ATTRS_TO_ADD ***
        # SoulSpark.__init__ will handle setting geom attrs to 0.0 if not in data

        logger.debug(f"  Emergence Factors: PhiRes={initial_data['phi_resonance']:.2f}, "
                     f"P.Coh={initial_data['pattern_coherence']:.2f} (Based on EoC={local_eoc:.3f})")

        # 3. Create Spark ID
        spark_id = f"Soul_{base_id}_{random.randint(100,999)}" # Ensure uniqueness

        # 4. Instantiate SoulSpark
        logger.debug(f"  Instantiating SoulSpark with derived initial data...")
        soul_spark = SoulSpark(initial_data=initial_data, spark_id=spark_id)

        return soul_spark

    except Exception as e:
        logger.critical(f"CRITICAL ERROR during spark creation at {creation_location_coords}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create SoulSpark from field state: {e}") from e

# --- Main Simulation Logic ---
def run_simulation(num_souls: int = 1,
                   journey_duration_per_sephirah: float = 2.0, # Defaulted from log
                   report_path: str = "output/reports/simulation_report_final.json"
                   ) -> None:
    """
    Runs the main simulation flow including emergence and harmonization stages.
    """
    logger.info("--- Starting Soul Development Simulation (V4.3.7) ---")
    logger.info(f"Number of Souls: {num_souls}")
    logger.info(f"Duration per Sephirah Interaction: {journey_duration_per_sephirah}")

    overall_start_time = time.time()
    simulation_start_iso = datetime.now().isoformat()

    all_souls_final_results = {}
    field_controller: Optional[FieldController] = None # Type hint

    try:
        # --- 1. Initialize Field System ---
        logger.info("Initializing Field Controller and Environment...")
        try:
            field_controller = FieldController(grid_size=GRID_SIZE)
            logger.info("Field Controller initialized successfully.")
        except Exception as fc_init_err:
            logger.critical("CRITICAL: Error during FieldController initialization.", exc_info=True)
            raise RuntimeError("FieldController initialization failed.") from fc_init_err

        # --- Simulation Loop for Each Soul ---
        for i in range(num_souls):
            soul_num = i + 1
            # Generate base ID for spark creation function
            base_id_str = f"{simulation_start_iso.replace(':','').replace('-','').replace('.','')}_{soul_num:03d}"
            logger.info(f"\n===== Processing Soul {soul_num}/{num_souls} (Base ID: {base_id_str}) =====")
            single_soul_start_time = time.time()
            process_summary = {'base_id': base_id_str, 'stages_metrics': {}}
            current_stage_name = "Pre-Emergence"
            soul_spark: Optional[SoulSpark] = None # Initialize to None

            try:
                # --- Stage 1: Spark Emergence ---
                current_stage_name = "Spark Emergence"
                logger.info(f"Stage: {current_stage_name}...")
                # Find optimal location based on field controller's tracked points
                creation_location = field_controller.find_optimal_development_location()
                creation_location_coords = field_controller._coords_to_int_tuple(creation_location)
                # Create the spark using the new function
                soul_spark = create_spark_from_field(field_controller, base_id_str, creation_location_coords)
                soul_spark.position = creation_location # Assign float position
                soul_spark.current_field_key = 'void' # Ensure starting state
                process_summary['soul_id'] = soul_spark.spark_id # Store actual ID
                logger.info(f"Soul Spark {soul_spark.spark_id} emerged at "
                            f"{creation_location_coords}. State: {soul_spark}")
                process_summary['stages_metrics'][current_stage_name] = soul_spark.get_spark_metrics()['core']
                display_stage_metrics(current_stage_name, {'success': True, **soul_spark.get_spark_metrics()['core']})

                # --- Stage 2: Spark Harmonization ---
                current_stage_name = "Spark Harmonization"
                logger.info(f"Stage: {current_stage_name}...")
                # Use default iterations from constants
                _, harm_metrics = perform_spark_harmonization(soul_spark)
                process_summary['stages_metrics'][current_stage_name] = harm_metrics
                display_stage_metrics(current_stage_name, harm_metrics)
                logger.info(f"Spark Harmonization complete. State: {soul_spark}")

                # --- Stage 3: Move to Guff ---
                current_stage_name = "Move to Guff"
                logger.info(f"Stage: {current_stage_name}...")
                field_controller.place_soul_in_guff(soul_spark)
                logger.info(f"Soul moved to Guff. Position: {soul_spark.position}, "
                            f"FieldKey: {soul_spark.current_field_key}")
                display_stage_metrics(current_stage_name, {
                    'success': True, 'final_position': soul_spark.position,
                    'final_field': soul_spark.current_field_key
                })

                # --- Stage 4: Guff Strengthening ---
                current_stage_name = "Guff Strengthening"
                logger.info(f"Stage: {current_stage_name}...")
                _, guff_metrics = perform_guff_strengthening(
                    soul_spark, field_controller, duration=GUFF_STRENGTHENING_DURATION
                )
                process_summary['stages_metrics'][current_stage_name] = guff_metrics
                display_stage_metrics(current_stage_name, guff_metrics)
                logger.info("Guff Strengthening complete.")

                # --- Stage 5: Release from Guff ---
                current_stage_name = "Release from Guff"
                logger.info(f"Stage: {current_stage_name}...")
                field_controller.release_soul_from_guff(soul_spark)
                logger.info(f"Soul released from Guff. Position: {soul_spark.position}, "
                            f"FieldKey: {soul_spark.current_field_key}")
                display_stage_metrics(current_stage_name, {
                    'success': True, 'final_position': soul_spark.position,
                    'final_field': soul_spark.current_field_key
                })

                # --- Stage 6: Sephiroth Journey ---
                current_stage_name = "Sephiroth Journey"
                logger.info(f"Stage: {current_stage_name}...")
                journey_path = ["kether", "chokmah", "binah", "daath", "chesed",
                                "geburah", "tiphareth", "netzach", "hod",
                                "yesod", "malkuth"]
                journey_step_metrics = {}

                for sephirah_name in journey_path:
                    interaction_stage_name = f"Sephirah Interaction ({sephirah_name.capitalize()})"
                    logger.info(f"  Entering {sephirah_name.capitalize()}...")
                    sephirah_influencer = field_controller.get_field(sephirah_name)
                    if not sephirah_influencer or not isinstance(sephirah_influencer, SephirothField):
                        raise RuntimeError(f"Could not get SephirothField for '{sephirah_name}'.")
                    # Process interaction using principle-driven function
                    _, step_metrics = process_sephirah_interaction(
                        soul_spark, sephirah_influencer, field_controller,
                        journey_duration_per_sephirah
                    )
                    journey_step_metrics[sephirah_name] = step_metrics
                    display_stage_metrics(interaction_stage_name, step_metrics)
                    logger.info(f"  Exiting {sephirah_name.capitalize()}.")

                # Mark journey completion flags
                setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, True)
                setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, True)
                process_summary['stages_metrics']['sephiroth_journey_steps'] = journey_step_metrics
                logger.info("Sephiroth Journey complete.")

                # --- Stage 7: Creator Entanglement ---
                current_stage_name = "Creator Entanglement"
                logger.info(f"Stage: {current_stage_name}...")
                kether_influencer = field_controller.kether_field
                if not kether_influencer: raise RuntimeError("Kether field unavailable.")
                # Assume defaults for creator_resonance, edge_of_chaos_ratio
                _, entanglement_metrics = run_full_entanglement_process(
                    soul_spark, kether_influencer
                )
                process_summary['stages_metrics'][current_stage_name] = entanglement_metrics
                display_stage_metrics(current_stage_name, entanglement_metrics)
                logger.info("Creator Entanglement complete.")

                # --- Stage 8: Harmonic Strengthening ---
                current_stage_name = FLAG_HARMONICALLY_STRENGTHENED.replace('_', ' ').title()
                logger.info(f"Stage: {current_stage_name}...")
                _, hs_metrics = perform_harmonic_strengthening(soul_spark)
                process_summary['stages_metrics'][current_stage_name] = hs_metrics
                display_stage_metrics(current_stage_name, hs_metrics)
                logger.info("Harmonic Strengthening complete.")

                # --- Stage 9: Life Cord Formation ---
                current_stage_name = FLAG_CORD_FORMATION_COMPLETE.replace('_', ' ').title()
                logger.info(f"Stage: {current_stage_name}...")
                _, lc_metrics = form_life_cord(soul_spark)
                process_summary['stages_metrics'][current_stage_name] = lc_metrics
                display_stage_metrics(current_stage_name, lc_metrics)
                logger.info("Life Cord Formation complete.")

                # --- Stage 10: Earth Harmonization ---
                current_stage_name = FLAG_EARTH_HARMONIZED.replace('_', ' ').title()
                logger.info(f"Stage: {current_stage_name}...")
                _, eh_metrics = perform_earth_harmonization(soul_spark)
                process_summary['stages_metrics'][current_stage_name] = eh_metrics
                display_stage_metrics(current_stage_name, eh_metrics)
                logger.info("Earth Harmonization complete.")

                # --- Stage 11: Identity Crystallization ---
                current_stage_name = FLAG_IDENTITY_CRYSTALLIZED.replace('_', ' ').title()
                logger.info(f"Stage: {current_stage_name}...")
                # Pass None to prompt user for name
                _, id_metrics = perform_identity_crystallization(soul_spark, specified_name=None)
                process_summary['stages_metrics'][current_stage_name] = id_metrics
                display_stage_metrics(current_stage_name, id_metrics)
                logger.info("Identity Crystallization complete.")

                # --- Stage 12: Birth Process ---
                current_stage_name = "Birth"
                logger.info(f"Stage: {current_stage_name}...")
                _, birth_metrics = perform_birth(
                    soul_spark, intensity=BIRTH_INTENSITY_DEFAULT,
                    mother_profile=None, # Pass actual profile if available
                    use_encoded_glyph=MOTHER_GLYPH_AVAILABLE
                )
                process_summary['stages_metrics'][current_stage_name] = birth_metrics
                display_stage_metrics(current_stage_name, birth_metrics)
                logger.info(f"Birth Process complete. Mother glyph influence: {MOTHER_GLYPH_AVAILABLE}")

                # --- Soul Completion Finalization ---
                process_summary['final_soul_state'] = soul_spark.get_spark_metrics()
                process_summary['success'] = True
                process_summary['end_time'] = datetime.now().isoformat()
                process_summary['total_duration_seconds'] = time.time() - single_soul_start_time
                all_souls_final_results[soul_spark.spark_id] = process_summary
                logger.info(f"===== Soul {soul_num}/{num_souls} (ID: {soul_spark.spark_id}) Processing Complete =====")
                logger.info(f"  Total time for soul: {process_summary['total_duration_seconds']:.2f} seconds")
                logger.info(f"  Final Status: Incarnated={getattr(soul_spark, FLAG_INCARNATED, False)}")

            except Exception as soul_err:
                # Log error and store failure summary
                failed_stage_name = current_stage_name
                end_time_iso = datetime.now().isoformat()
                logger.error(f"Soul processing failed at stage '{failed_stage_name}': {soul_err}", exc_info=True)
                process_summary['success'] = False
                process_summary['failed_stage'] = failed_stage_name
                process_summary['error'] = str(soul_err)
                process_summary['end_time'] = end_time_iso
                process_summary['total_duration_seconds'] = time.time() - single_soul_start_time
                # Store final state even on failure if soul_spark object exists
                if soul_spark: process_summary['final_soul_state_on_fail'] = soul_spark.get_spark_metrics()
                # Ensure the actual soul ID is used if available, otherwise use base ID
                final_id = getattr(soul_spark, 'spark_id', base_id_str)
                all_souls_final_results[final_id] = process_summary
                logger.error(f"===== Soul {soul_num}/{num_souls} (ID: {final_id}) Processing FAILED =====")
                print(f"\n{'='*20} ERROR: Soul Processing Failed {'='*20}")
                print(f"  Stage: {failed_stage_name}")
                print(f"  Error: {soul_err}")
                print("="*70)

        # --- End of Simulation Loop ---

        # --- Final Report ---
        logger.info("Simulation loop finished. Generating final report...")
        final_report_data = {
            'simulation_start_time': simulation_start_iso,
            'simulation_end_time': datetime.now().isoformat(),
            'total_duration_seconds': time.time() - overall_start_time,
            'parameters': {
                'num_souls': num_souls,
                'journey_duration_per_sephirah': journey_duration_per_sephirah,
                'grid_size': GRID_SIZE,
                'mother_glyph_available': MOTHER_GLYPH_AVAILABLE,
                # Add other key parameters if needed
            },
            'souls_processed': len(all_souls_final_results),
            'results_per_soul': all_souls_final_results
        }

        if report_path:
            try:
                report_dir = os.path.dirname(report_path)
                if report_dir: os.makedirs(report_dir, exist_ok=True)
                with open(report_path, 'w') as f:
                    class NumpyEncoder(json.JSONEncoder): # Nested encoder class
                        def default(self, o):
                            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                                             np.int16, np.int32, np.int64, np.uint8,
                                             np.uint16, np.uint32, np.uint64)): return int(o)
                            elif isinstance(o, (np.float_, np.float16, np.float32,
                                               np.float64)): return float(o)
                            elif isinstance(o, np.ndarray): return o.tolist()
                            elif isinstance(o, (datetime, uuid.UUID)): return str(o)
                            try: return super().default(o)
                            except TypeError: return f"<Unserializable:{type(o).__name__}>"
                    json.dump(final_report_data, f, indent=2, cls=NumpyEncoder)
                logger.info(f"Final simulation report saved to: {report_path}")
            except TypeError as json_err:
                 logger.error(f"JSON Serialization Error saving report: {json_err}. Saving partial.", exc_info=True)
                 try:
                      partial_path = report_path.replace('.json','_partial.json')
                      partial_data = {k:v for k,v in final_report_data.items() if k!='results_per_soul'}
                      with open(partial_path, 'w') as f: json.dump(partial_data, f, indent=2, cls=NumpyEncoder)
                      logger.info(f"Partial report saved to {partial_path}")
                 except Exception as partial_err: logger.error(f"Failed to save partial report: {partial_err}")
            except Exception as report_err: logger.error(f"Failed to save final report: {report_err}", exc_info=True)

        # --- Display Final Summary ---
        print("\n" + "="*80)
        print(f"SIMULATION COMPLETE - Processed {num_souls} souls")
        print(f"Total duration: {time.time() - overall_start_time:.2f} seconds")
        successful_souls = sum(1 for res in all_souls_final_results.values() if res.get('success'))
        print(f"Successful souls: {successful_souls}/{num_souls}")
        print(f"Failed souls: {num_souls - successful_souls}/{num_souls}")
        if report_path: print(f"Report saved to: {report_path}")
        print("="*80)
        logger.info(f"--- Soul Development Simulation Finished ---")

    except Exception as main_err:
        logger.critical(f"Simulation aborted due to critical error: {main_err}", exc_info=True)
        print("\n" + "="*80); print("CRITICAL ERROR - SIMULATION ABORTED")
        print(f"Error: {main_err}"); print("See log for details."); print("="*80)
        sys.exit(1)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("DEBUG: Starting main execution block of root_controller...")
    try:
        # Configure simulation parameters here
        run_simulation(
            num_souls=1,
            journey_duration_per_sephirah=2.0, # Keep shorter duration from logs
            report_path="output/reports/simulation_report_emergence_v1.json"
        )
        print("DEBUG: run_simulation completed.")
    except Exception as e:
         log_func = logger.critical if logger.hasHandlers() else print
         log_func(f"FATAL ERROR in main execution block: {e}", exc_info=True)
         print(f"\nFATAL ERROR in main execution block: {type(e).__name__}: {e}")
         traceback.print_exc(); sys.exit(1)
    finally:
        print("DEBUG: Main execution block finished.")
        if METRICS_AVAILABLE:
            try:
                print("Attempting final metrics persistence...")
                metrics.persist_metrics()
                print("Final metrics persistence successful.")
            except Exception as persist_e:
                print(f"ERROR during final metrics persistence: {persist_e}")
        logging.shutdown()

# --- END OF FILE root_controller.py ---