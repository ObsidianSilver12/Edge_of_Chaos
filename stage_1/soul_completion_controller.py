# --- START OF FILE src/stage_1/soul_completion_controller.py ---


# whole file is now wrong - rewrite from scratch do not include any fallbacks only hard fails. this was refactored
# to try match the stage functions for evolve. that whole refactor was abandoned as it was grossly incomplete
# you will need to check that each part of the controller from soul spark to birth matches the refactored files from soul
# spark to birth and includes all the new files that have been created for the brain formation process.
# birth has to be refactored to match the new birth process as well and the new refactored files include reference to
# the life cord so that needs to be checked to see if it matches. new process is soul is passed through the life cord which
# must be attached to the brain stem and then the soul is guided to its new home in the brain.

"""
Soul Completion Controller (Refactored V4.3.11 - Full Soul Formation, Comp. Report)

Orchestrates the complete soul formation process from Spark Harmonization through Birth.
Handles all stage functions directly.
Works with refactored stage functions that use wave physics and layer-based approaches.
Uses constants directly after wildcard import.
Calls the new comprehensive soul report at the end of each soul's processing.
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
from stage_1.soul_formation.creator_entanglement import (
    save_creator_entanglement_sensory_data  # This function needs to be imported
)
from stage_1.soul_formation.sephiroth_journey_processing import (
    create_encoded_sephirah_sigil,
    capture_sephirah_interaction_sensory_data,
    save_sephiroth_journey_sensory_data
)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants (Wildcard Import) ---
try:
    from shared.constants.constants import *
    logger.setLevel(LOG_LEVEL) # Set logger level after successful import
except ImportError as e:
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'
    logger.critical(f"CRITICAL ERROR: constants.py failed import in soul_completion_controller.py: {e}")
    logger.warning(f"Constants not loaded, using fallback values. LOG_LEVEL set to {LOG_LEVEL}")
    logger.setLevel(getattr(logging, str(LOG_LEVEL).upper(), logging.INFO))
    if not logger.handlers: # Basic handler if constants failed and logger wasn't set up by root
        logging.basicConfig(level=getattr(logging, str(LOG_LEVEL).upper(), logging.INFO), format=LOG_FORMAT)

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
        metrics = MetricsPlaceholder() # type: ignore
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import stage modules/SoulSpark: {e}", exc_info=True)
    raise ImportError(f"Core stage dependencies missing: {e}") from e

# --- Visualization Import & Setup ---
VISUALIZATION_ENABLED = False # Default
VISUALIZATION_OUTPUT_DIR = os.path.join(DATA_DIR_BASE, "visuals", "soul_completion") # Consistent path
try:
    from stage_1.soul_formation.soul_visualizer import (
        visualize_soul_state,
        visualize_state_comparison,
        create_comprehensive_soul_report # New import
    )
    VISUALIZATION_ENABLED = True
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR_BASE, "completed_souls"), exist_ok=True)
    logger.info("Soul visualization module loaded successfully in SoulCompletionController.")
except ImportError as ie:
    logger.warning(f"Soul visualization module not found in SoulCompletionController: {ie}. Visualizations disabled.")
except Exception as e:
    logger.warning(f"Error setting up visualization in SoulCompletionController: {e}. Visualizations disabled.")

# --- Mother Resonance Import ---
MOTHER_RESONANCE_AVAILABLE = False # Default
try:
    from stage_1.womb.mother_resonance import create_mother_resonance_data
    MOTHER_RESONANCE_AVAILABLE = True
    logger.info("Mother resonance module loaded successfully in SoulCompletionController.")
except ImportError:
    logger.warning("Mother resonance module not found in SoulCompletionController. Birth will proceed without mother influence or require explicit profile.")

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
        self.visual_save_dir = VISUALIZATION_OUTPUT_DIR # Use the module-level constant
        self.field_controller = field_controller
        
        # Initialize mycelial components
        self.memory_fragment_system = None
        self.quantum_network = None
        self.mycelial_initialized = False
        
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
                # Use a copy or a new SoulSpark instance if the original is heavily modified in-place by visualize_soul_state
                # For now, assuming visualize_soul_state is non-destructive or its modifications are acceptable.
                self.development_states[spark_id].append((soul_spark, pre_stage_vis_name))
                visualize_soul_state(soul_spark, pre_stage_vis_name, self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"{pre_stage_vis_name} visualization failed: {vis_err}")

        try:
            # Ensure kwargs passed to stage_func are only what it expects, or that stage_func handles **kwargs
            # For now, passing all kwargs assuming stage functions can handle them or ignore extras.
            _, stage_metrics = stage_func(soul_spark=soul_spark, **kwargs)


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


    def run_spark_harmonization(self, soul_spark: SoulSpark, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        # iterations is a parameter for perform_spark_harmonization, get it from constants or pass via kwargs
        iterations = kwargs.get('harmonization_iterations', HARMONIZATION_ITERATIONS)
        
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Stage: Spark Harmonization for {spark_id}...")
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': 'Spark Harmonization', 'start_time': datetime.now().isoformat()}

        if self.visualization_enabled:
            if spark_id not in self.development_states: self.development_states[spark_id] = []
            try:
                self.development_states[spark_id].append((soul_spark, "Pre_Harmonization"))
                visualize_soul_state(soul_spark, "Pre_Harmonization", self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"Pre_Harmonization visualization failed: {vis_err}")

        try:
            _, stage_metrics = perform_spark_harmonization(soul_spark=soul_spark, iterations=iterations)

            # Initialize basic mycelial network after harmonization
            if hasattr(soul_spark, 'brain_structure') and soul_spark.brain_structure is not None:
                try:
                    logger.info("Initializing basic mycelial network")
                    brain_structure = soul_spark.brain_structure
                    seed_position = getattr(soul_spark, 'position', (32, 32, 32))
                    
                    # Initialize mycelial network
                    network_metrics = initialize_basic_network(brain_structure, seed_position)
                    stage_metrics['mycelial_initialization'] = network_metrics
                    
                    # Initialize memory fragment system
                    self.memory_fragment_system = MemoryFragmentSystem(brain_structure)
                    
                    self.mycelial_initialized = True
                    logger.info(f"Mycelial network initialized with {network_metrics['cells_affected']} cells")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize mycelial network: {e}")
                    stage_metrics['mycelial_initialization'] = {'error': str(e)}

            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, "Post_Harmonization"))
                    visualize_soul_state(soul_spark, "Post_Harmonization", self.visual_save_dir, show=show_visuals)
                except Exception as vis_err: logger.warning(f"Post_Harmonization visualization failed: {vis_err}")

            display_stage_metrics("Spark Harmonization", stage_metrics)
            logger.info(f"Spark Harmonization Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            return soul_spark, stage_metrics
        except Exception as e:
            logger.error(f"Spark Harmonization failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['error'] = str(e)
            raise RuntimeError(f"Spark Harmonization failed: {e}") from e
        

    def run_guff_strengthening(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
                            duration: float = GUFF_STRENGTHENING_DURATION, show_visuals: bool = False) -> Tuple[SoulSpark, Dict[str, Any]]:
        field_ctrl = field_controller or self.field_controller
        if not field_ctrl: raise ValueError("Field controller required for Guff strengthening.")
        
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Stage: Guff Strengthening for {spark_id}...")
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': 'Guff Strengthening', 'start_time': datetime.now().isoformat()}
        
        try:
            current_field = getattr(soul_spark, 'current_field_key', None)
            if current_field != 'guff' and current_field != 'kether': # Allow if already in Kether but not yet Guff
                logger.info(f"Moving soul {soul_spark.spark_id} to Guff for strengthening...")
                field_ctrl.place_soul_in_guff(soul_spark)
        except Exception as move_err: raise RuntimeError(f"Failed to move soul to Guff: {move_err}") from move_err

        if self.visualization_enabled:
            if spark_id not in self.development_states: self.development_states[spark_id] = []
            try:
                self.development_states[spark_id].append((soul_spark, "Pre_Guff_Strengthening"))
                visualize_soul_state(soul_spark, "Pre_Guff_Strengthening", self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"Pre_Guff_Strengthening visualization failed: {vis_err}")

        try:
            _, stage_metrics = perform_guff_strengthening(soul_spark=soul_spark, field_controller=field_ctrl, duration=duration)

            # Establish primary mycelial pathways
            if self.mycelial_initialized and hasattr(soul_spark, 'brain_structure'):
                try:
                    logger.info("Establishing primary mycelial pathways")
                    pathway_metrics = establish_primary_pathways(soul_spark.brain_structure)
                    stage_metrics['mycelial_pathways'] = pathway_metrics
                    
                    # Setup energy distribution channels
                    energy_metrics = setup_energy_distribution_channels(soul_spark.brain_structure)
                    stage_metrics['mycelial_energy'] = energy_metrics
                    
                    logger.info(f"Established {pathway_metrics['pathways_created']} pathways, "
                            f"{energy_metrics['channels_created']} energy channels")
                            
                except Exception as e:
                    logger.warning(f"Failed to establish mycelial pathways: {e}")
                    stage_metrics['mycelial_pathways'] = {'error': str(e)}

            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, "Post_Guff_Strengthening"))
                    visualize_soul_state(soul_spark, "Post_Guff_Strengthening", self.visual_save_dir, show=show_visuals)
                except Exception as vis_err: logger.warning(f"Post_Guff_Strengthening visualization failed: {vis_err}")

            display_stage_metrics("Guff Strengthening", stage_metrics)
            logger.info(f"Guff Strengthening Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            
            try: field_ctrl.release_soul_from_guff(soul_spark)
            except Exception as release_err: raise RuntimeError(f"Failed to release soul from Guff: {release_err}") from release_err
            
            return soul_spark, stage_metrics
        except Exception as e:
            logger.error(f"Guff Strengthening failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['error'] = str(e)
            raise RuntimeError(f"Guff Strengthening failed: {e}") from e

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
            journey_sensory_captures: Dict[str, Any] = {}
            for sephirah_name_iter in journey_path:
                stage_id = f"Interaction ({sephirah_name_iter.capitalize()})"
                logger.info(f"  Entering {sephirah_name_iter.capitalize()}...")
                sephirah_influencer = field_ctrl.get_field(sephirah_name_iter)
                if not sephirah_influencer or not isinstance(sephirah_influencer, SephirothField):
                    raise RuntimeError(f"SephirothField missing for '{sephirah_name_iter}'.")
                
                # Process Sephirah interaction (existing functionality)
                _, step_metrics = process_sephirah_interaction(
                    soul_spark=soul_spark, sephirah_influencer=sephirah_influencer,
                    field_controller=field_ctrl, duration=journey_duration_per_sephirah
                )
                journey_step_metrics[sephirah_name_iter] = step_metrics
                try:
                    # Create encoded Sephirah sigil with Gateway Key frequency
                    sigil_glyph_id, encoded_sigil_path = create_encoded_sephirah_sigil(
                        soul_spark, sephirah_name_iter, step_metrics
                    )
                    
                    # Capture complete sensory data for this Sephirah interaction
                    sephirah_sensory = capture_sephirah_interaction_sensory_data(
                        soul_spark, sephirah_name_iter, step_metrics, sigil_glyph_id, encoded_sigil_path
                    )
                    
                    journey_sensory_captures[sephirah_name_iter] = sephirah_sensory
                    logger.info(f"Captured sensory data for {sephirah_name_iter}: {sephirah_sensory['capture_id']}")
                    
                except Exception as sens_err:
                    logger.error(f"Failed to capture sensory data for {sephirah_name_iter}: {sens_err}")
                    journey_step_metrics[sephirah_name_iter] = {'error': str(sens_err)}
                logger.info(f"  Exiting {sephirah_name_iter.capitalize()}.")
                if self.visualization_enabled and sephirah_name_iter in ["kether", "tiphareth", "malkuth"]:
                    try: visualize_soul_state(soul_spark, f"Sephiroth_{sephirah_name_iter.capitalize()}", self.visual_save_dir, show=show_visuals)
                except Exception as sensory_err:
                    logger.warning(f"Failed to capture sensory data for {sephirah_name_iter}: {sensory_err}")
                    journey_sensory_captures[sephirah_name_iter] = {'error': str(sensory_err)}
                
                display_stage_metrics(stage_id, step_metrics)
                logger.info(f"  Exiting {sephirah_name_iter.capitalize()}.")
                if self.visualization_enabled and sephirah_name_iter in ["kether", "tiphareth", "malkuth"]:
                    try:
                        visualize_soul_state(soul_spark, f"Sephiroth_{sephirah_name_iter.capitalize()}", self.visual_save_dir, show=show_visuals)
                    except Exception as vis_err:
                        logger.warning(f"Sephiroth {sephirah_name_iter} visualization failed: {vis_err}")
            
            # Save complete Sephiroth journey sensory data for soul echos processing
            try:
                if journey_sensory_captures:
                    sensory_data_path = save_sephiroth_journey_sensory_data(
                        soul_spark, journey_sensory_captures, journey_path
                    )
                    journey_step_metrics['sensory_data_saved'] = {
                        'path': sensory_data_path,
                        'captures_count': len(journey_sensory_captures),
                        'success': True
                    }
                    logger.info(f"Saved Sephiroth journey sensory data: {sensory_data_path}")
                else:
                    logger.warning("No sensory captures to save")
                    
            except Exception as save_err:
                logger.error(f"Failed to save Sephiroth journey sensory data: {save_err}")
                journey_step_metrics['sensory_data_saved'] = {'error': str(save_err), 'success': False}

            # Create memory fragments for soul aspects gained during journey
            if self.memory_fragment_system and hasattr(soul_spark, 'aspects'):
                try:
                    logger.info("Creating memory fragments for soul aspects")
                    aspects = getattr(soul_spark, 'aspects', {})
                    
                    if aspects:
                        fragment_metrics = distribute_soul_aspects(
                            self.memory_fragment_system, 
                            soul_spark.brain_structure, 
                            aspects
                        )
                        journey_step_metrics['memory_fragments'] = fragment_metrics
                        logger.info(f"Created {fragment_metrics['aspects_distributed']} memory fragments")
                    
                except Exception as e:
                    logger.warning(f"Failed to create memory fragments: {e}")
                    journey_step_metrics['memory_fragments'] = {'error': str(e)}
            
            setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, True)
            setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, True)
            journey_metrics = {
                'steps': journey_step_metrics, 
                'soul_id': spark_id, 
                'journey_path': journey_path,
                'journey_duration_per_sephirah': journey_duration_per_sephirah,
                'total_journey_duration': journey_duration_per_sephirah * len(journey_path), 
                'sensory_captures': {
                    'total_captures': len(journey_sensory_captures),
                    'successful_captures': len([c for c in journey_sensory_captures.values() if 'error' not in c]),
                    'failed_captures': len([c for c in journey_sensory_captures.values() if 'error' in c]),
                    'captures_by_sephirah': list(journey_sensory_captures.keys())
                },
                'success': True
            }
            
            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, "Post_Sephiroth_Journey"))
                    visualize_soul_state(soul_spark, "Post_Sephiroth_Journey", self.visual_save_dir, show=show_visuals)
                except Exception as vis_err:
                    logger.warning(f"Post-Journey visualization failed: {vis_err}")
            logger.info(f"{stage_name} Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            return soul_spark, journey_metrics
        except Exception as e:
            logger.error(f"{stage_name} failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status']='failed'; self.active_souls[spark_id]['error']=str(e)
            setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False); setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, False)
            raise RuntimeError(f"{stage_name} failed: {e}") from e
    def run_creator_entanglement(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
                                show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        field_ctrl = field_controller or self.field_controller
        if not field_ctrl: raise ValueError("Field controller required for Creator entanglement.")
        if not getattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False): raise ValueError("Sephiroth journey not complete.")
        if not getattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, False): raise ValueError(f"Missing {FLAG_READY_FOR_ENTANGLEMENT} flag.")
        kether_influencer = field_ctrl.kether_field
        if not kether_influencer: raise RuntimeError("Kether field unavailable.")

        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Stage: Creator Entanglement for {spark_id}...")
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': 'Creator Entanglement', 'start_time': datetime.now().isoformat()}

        if self.visualization_enabled:
            if spark_id not in self.development_states: self.development_states[spark_id] = []
            try:
                self.development_states[spark_id].append((soul_spark, "Pre_Creator_Entanglement"))
                visualize_soul_state(soul_spark, "Pre_Creator_Entanglement", self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"Pre_Creator_Entanglement visualization failed: {vis_err}")

        try:
            # Extract relevant kwargs for perform_creator_entanglement
            entanglement_kwargs = {
                'base_creator_potential': kwargs.get('base_creator_potential', CREATOR_POTENTIAL_DEFAULT),
                'edge_of_chaos_target': kwargs.get('edge_of_chaos_target', EDGE_OF_CHAOS_DEFAULT)
            }
            
            _, stage_metrics = perform_creator_entanglement(soul_spark=soul_spark, kether_field=kether_influencer, **entanglement_kwargs)

            # Prepare mycelial network for soul attachment
            if self.mycelial_initialized and hasattr(soul_spark, 'brain_structure'):
                try:
                    logger.info("Preparing mycelial network for soul attachment")
                    preparation_metrics = prepare_for_soul_attachment(soul_spark.brain_structure)
                    stage_metrics['soul_preparation'] = preparation_metrics
                    
                    # Store soul attachment position for later use
                    soul_spark.soul_attachment_position = preparation_metrics['position']
                    
                    logger.info(f"Soul attachment prepared at {preparation_metrics['position']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare soul attachment: {e}")
                    stage_metrics['soul_preparation'] = {'error': str(e)}

            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, "Post_Creator_Entanglement"))
                    visualize_soul_state(soul_spark, "Post_Creator_Entanglement", self.visual_save_dir, show=show_visuals)
                except Exception as vis_err: logger.warning(f"Post_Creator_Entanglement visualization failed: {vis_err}")

            display_stage_metrics("Creator Entanglement", stage_metrics)
            logger.info(f"Creator Entanglement Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            
            setattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, True)
            return soul_spark, stage_metrics
        except Exception as e:
            logger.error(f"Creator Entanglement failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['error'] = str(e)
            raise RuntimeError(f"Creator Entanglement failed: {e}") from e

    def run_harmonic_strengthening(self, soul_spark: SoulSpark, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        """
        Runs the harmonic strengthening stage.
        
        Args:
            soul_spark: The SoulSpark object to strengthen
            show_visuals: Whether to display visualizations
            **kwargs: Optional parameters including harmony_intensity and harmony_duration_factor
            
        Returns:
            Tuple of (modified SoulSpark, stage metrics)
        """
        if not getattr(soul_spark, FLAG_CREATOR_ENTANGLED, False): 
            raise ValueError("Creator entanglement not complete.")
        if not getattr(soul_spark, FLAG_READY_FOR_HARMONIZATION, False): 
            raise ValueError(f"Missing {FLAG_READY_FOR_HARMONIZATION} flag.")

        # Prepare parameters ensuring sufficient cycles for full convergence
        hs_kwargs = {
            'intensity': kwargs.get('harmony_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT),
            'duration_factor': kwargs.get('harmony_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT)
        }
        
        # Log the max cycles that will be used
        max_cycles = int(HS_MAX_CYCLES * hs_kwargs['duration_factor'])
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Running Harmonic Strengthening on {spark_id} with up to {max_cycles} cycles")
        logger.info(f"Using intensity={hs_kwargs['intensity']:.2f}, aiming for 100% convergence")
        
        # Run the harmonic strengthening stage
        stage_result = self._run_stage(
            perform_harmonic_strengthening, soul_spark, "Harmonic Strengthening", show_visuals,
            "Pre_Harmonic_Strengthening", "Post_Harmonic_Strengthening",
            **hs_kwargs
        )
        
        # Extract convergence info from metrics for detailed logging
        _, metrics = stage_result
        cycles_run = metrics.get('cycles_run', 0)
        stability_converged = metrics.get('stability_converged', False)
        coherence_converged = metrics.get('coherence_converged', False)
        
        # Log detailed convergence status
        if stability_converged and coherence_converged:
            logger.info(f"Harmonic Strengthening achieved FULL convergence after {cycles_run} cycles!")
        elif cycles_run >= max_cycles:
            logger.warning(f"Harmonic Strengthening reached max cycles ({max_cycles}) without full convergence")
            if not stability_converged:
                logger.warning(f"Stability did not converge to maximum (current: {soul_spark.stability:.2f})")
            if not coherence_converged:
                logger.warning(f"Coherence did not converge to maximum (current: {soul_spark.coherence:.2f})")
        else:
            logger.info(f"Harmonic Strengthening met thresholds after {cycles_run} cycles")
        
        # Set ready for next stage
        setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, True)
        return stage_result

    def run_life_cord_formation(self, soul_spark: SoulSpark, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]: # Added kwargs
        if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False): raise ValueError(f"Missing {FLAG_READY_FOR_LIFE_CORD} flag.")

        lc_kwargs = {
            'intensity': kwargs.get('life_cord_intensity', 0.7), # Using provided default
            'complexity': kwargs.get('cord_complexity', LIFE_CORD_COMPLEXITY_DEFAULT)
        }
        return self._run_stage(
            form_life_cord, soul_spark, "Life Cord Formation", show_visuals,
            "Pre_Life_Cord", "Post_Life_Cord",
            **lc_kwargs
        )

    def run_earth_harmonization(self, soul_spark: SoulSpark, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]: # Added kwargs
        if not getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False): raise ValueError("Life cord not formed.")
        if not getattr(soul_spark, FLAG_READY_FOR_EARTH, False): raise ValueError(f"Missing {FLAG_READY_FOR_EARTH} flag.")

        eh_kwargs = {
            'schumann_intensity': kwargs.get('schumann_intensity', EARTH_HARMONY_INTENSITY_DEFAULT),
            'core_intensity': kwargs.get('core_intensity', EARTH_HARMONY_INTENSITY_DEFAULT)
        }
        return self._run_stage(
            perform_earth_harmonization, soul_spark, "Earth Harmonization", show_visuals,
            "Pre_Earth_Harmonization", "Post_Earth_Harmonization",
            **eh_kwargs
        )

    def run_identity_crystallization(self, soul_spark: SoulSpark, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not getattr(soul_spark, FLAG_EARTH_ATTUNED, False): raise ValueError("Not harmonized with Earth.")
        
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Stage: Identity Crystallization for {spark_id}...")
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': 'Identity Crystallization', 'start_time': datetime.now().isoformat()}

        if self.visualization_enabled:
            if spark_id not in self.development_states: self.development_states[spark_id] = []
            try:
                self.development_states[spark_id].append((soul_spark, "Pre_Identity_Crystallization"))
                visualize_soul_state(soul_spark, "Pre_Identity_Crystallization", self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"Pre_Identity_Crystallization visualization failed: {vis_err}")

        try:
            # Pass all kwargs directly, perform_identity_crystallization will pick what it needs
            _, stage_metrics = perform_identity_crystallization(soul_spark=soul_spark, **kwargs)

            # Initialize quantum communication network
            if hasattr(soul_spark, 'brain_structure') and not self.quantum_network:
                try:
                    logger.info("Initializing quantum communication network")
                    self.quantum_network = create_mycelial_quantum_network(soul_spark.brain_structure)
                    
                    quantum_state = self.quantum_network.get_network_state()
                    stage_metrics['quantum_network'] = quantum_state
                    
                    logger.info(f"Quantum network created with {quantum_state['total_seeds']} seeds, "
                            f"{quantum_state['total_entanglements']} entanglements")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize quantum network: {e}")
                    stage_metrics['quantum_network'] = {'error': str(e)}

            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, "Post_Identity_Crystallization"))
                    visualize_soul_state(soul_spark, "Post_Identity_Crystallization", self.visual_save_dir, show=show_visuals)
                except Exception as vis_err: logger.warning(f"Post_Identity_Crystallization visualization failed: {vis_err}")

            display_stage_metrics("Identity Crystallization", stage_metrics)
            logger.info(f"Identity Crystallization Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            return soul_spark, stage_metrics
        except Exception as e:
            logger.error(f"Identity Crystallization failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['error'] = str(e)
            raise RuntimeError(f"Identity Crystallization failed: {e}") from e

    def run_birth_process(self, soul_spark: SoulSpark, show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        if not getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False): raise ValueError("Identity not crystallized.")
        if not getattr(soul_spark, FLAG_READY_FOR_BIRTH, False): raise ValueError(f"Missing {FLAG_READY_FOR_BIRTH} flag.")

        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Stage: Birth Process for {spark_id}...")
        self.active_souls[spark_id] = {'status': 'processing', 'current_stage': 'Birth Process', 'start_time': datetime.now().isoformat()}

        if self.visualization_enabled:
            if spark_id not in self.development_states: self.development_states[spark_id] = []
            try:
                self.development_states[spark_id].append((soul_spark, "Pre_Birth"))
                visualize_soul_state(soul_spark, "Pre_Birth", self.visual_save_dir, show=show_visuals)
            except Exception as vis_err: logger.warning(f"Pre_Birth visualization failed: {vis_err}")

        try:
            birth_kwargs = {
                'intensity': kwargs.get('birth_intensity', BIRTH_INTENSITY_DEFAULT),
                'mother_profile': kwargs.get('mother_profile') # Will be None if not in kwargs
            }
            # Create/get mother profile if not provided and module available
            if not birth_kwargs['mother_profile'] and MOTHER_RESONANCE_AVAILABLE:
                try:
                    mother_resonance_data = create_mother_resonance_data()
                    birth_kwargs['mother_profile'] = {
                        'nurturing_capacity': mother_resonance_data.get('nurturing_capacity', 0.7),
                        'spiritual': mother_resonance_data.get('spiritual', {'connection': 0.6}),
                        'love_resonance': mother_resonance_data.get('love_resonance', 0.7),
                        'physical': mother_resonance_data.get('physical', {'health': 0.7}),
                        'emotional': mother_resonance_data.get('emotional', {'stability': 0.7})
                    }
                except Exception as mother_err:
                    logger.warning(f"Failed to create mother resonance data: {mother_err}. Using defaults.")

            _, stage_metrics = perform_birth(soul_spark=soul_spark, **birth_kwargs)

            # Create final soul connection channels
            if (self.mycelial_initialized and hasattr(soul_spark, 'brain_structure') and 
                hasattr(soul_spark, 'soul_attachment_position')):
                try:
                    logger.info("Creating soul connection channels")
                    
                    # Create dedicated soul connection channel
                    channel_metrics = create_soul_connection_channel(
                        soul_spark.brain_structure, 
                        soul_spark.soul_attachment_position
                    )
                    stage_metrics['soul_connection'] = channel_metrics
                    
                    # Connect to limbic region for emotional integration
                    limbic_metrics = connect_soul_to_limbic_region(
                        soul_spark.brain_structure, 
                        soul_spark.soul_attachment_position
                    )
                    stage_metrics['limbic_connection'] = limbic_metrics
                    
                    logger.info(f"Soul connection established with {channel_metrics['unique_cells']} channel cells, "
                            f"{limbic_metrics.get('emotional_centers', 0)} emotional centers")
                    
                except Exception as e:
                    logger.warning(f"Failed to create soul connection: {e}")
                    stage_metrics['soul_connection'] = {'error': str(e)}

            if self.visualization_enabled:
                try:
                    self.development_states[spark_id].append((soul_spark, "Post_Birth"))
                    visualize_soul_state(soul_spark, "Post_Birth", self.visual_save_dir, show=show_visuals)
                except Exception as vis_err: logger.warning(f"Post_Birth visualization failed: {vis_err}")

            display_stage_metrics("Birth Process", stage_metrics)
            logger.info(f"Birth Process Complete. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")
            return soul_spark, stage_metrics
        except Exception as e:
            logger.error(f"Birth Process failed for {spark_id}: {e}", exc_info=True)
            self.active_souls[spark_id]['status'] = 'failed'; self.active_souls[spark_id]['error'] = str(e)
            raise RuntimeError(f"Birth Process failed: {e}") from e

    """
    NEW METHOD to add to the class:
    """
    def cleanup_mycelial_resources(self):
        """Clean up mycelial system resources."""
        try:
            if self.memory_fragment_system:
                # Save memory fragments if needed
                logger.info("Cleaning up memory fragment system")
                
            if self.quantum_network:
                # Save quantum network state if needed  
                logger.info("Cleaning up quantum network")
                
            self.mycelial_initialized = False
            logger.info("Mycelial resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during mycelial cleanup: {e}")

    """
    COMPLETE cleanup_resources method replacement:
    """
    def cleanup_resources(self):
        """Cleans up any resources used by the controller."""
        try:
            self.cleanup_mycelial_resources()
            
            self.active_souls.clear()
            if METRICS_AVAILABLE and hasattr(metrics, 'persist_metrics'):
                metrics.persist_metrics()
            shutdown_time = datetime.now().isoformat()
            logger.info(f"Soul Completion Controller '{self.controller_id}' shutting down at {shutdown_time}")
            return True
        except Exception as e:
            logger.error(f"Error during controller cleanup: {e}", exc_info=True)
            return False
    def create_final_reports(self, soul_spark: SoulSpark, save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Creates comprehensive reports for the completed soul.
        
        Args:
            soul_spark: The completed SoulSpark object
            save_path: Optional path to save reports (defaults to completed_souls dir)
            
        Returns:
            Dict mapping report types to their file paths
        """
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Creating final reports for soul {spark_id}...")
        
        if not save_path:
            save_path = os.path.join(DATA_DIR_BASE, "completed_souls", spark_id)
        
        os.makedirs(save_path, exist_ok=True)
        report_paths = {}
        
        # Generate comprehensive soul report
        if VISUALIZATION_ENABLED:
            try:
                report_path = create_comprehensive_soul_report(
                    soul_spark, 
                    os.path.join(save_path, f"{spark_id}_comprehensive_report.html")
                )
                report_paths['comprehensive'] = report_path
                logger.info(f"Created comprehensive soul report at {report_path}")
            except Exception as report_err:
                logger.error(f"Failed to create comprehensive report: {report_err}")
        
        # Save soul state JSON
        try:
            soul_data = soul_spark.to_dict()
            json_path = os.path.join(save_path, f"{spark_id}_state.json")
            with open(json_path, 'w') as f:
                json.dump(soul_data, f, indent=2)
            report_paths['json'] = json_path
            logger.info(f"Saved soul state JSON to {json_path}")
        except Exception as json_err:
            logger.error(f"Failed to save soul state JSON: {json_err}")
        
        # Create development timeline visualization
        if VISUALIZATION_ENABLED and spark_id in self.development_states:
            try:
                timeline_path = os.path.join(save_path, f"{spark_id}_development_timeline.html")
                # Create development timeline visualization with all states
                visualize_state_comparison(
                    self.development_states[spark_id],
                    timeline_path,
                    show=False
                )
                report_paths['timeline'] = timeline_path
                logger.info(f"Created development timeline at {timeline_path}")
            except Exception as timeline_err:
                logger.error(f"Failed to create development timeline: {timeline_err}")
        
        # Record metrics about final reports
        if METRICS_AVAILABLE:
            metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                'action': 'create_final_reports',
                'soul_id': spark_id,
                'report_types': list(report_paths.keys()),
                'report_paths': report_paths,
                'success': bool(report_paths),
                'timestamp': datetime.now().isoformat()
            })
        
        return report_paths

    def run_complete_formation(self, soul_spark: SoulSpark, field_controller: Optional[FieldController] = None,
                              show_visuals: bool = False, **kwargs) -> Tuple[SoulSpark, Dict[str, Any]]:
        """
        Runs the complete soul formation process from Spark to Birth.
        
        Args:
            soul_spark: The initial SoulSpark object
            field_controller: Optional field controller (uses self.field_controller if None)
            show_visuals: Whether to display visualizations
            **kwargs: Optional parameters for various stages
            
        Returns:
            Tuple of (completed SoulSpark, complete metrics dictionary)
        """
        spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
        logger.info(f"Starting complete formation for soul {spark_id}...")
        
        # Record start time and set soul status
        start_time = datetime.now()
        self.active_souls[spark_id] = {
            'status': 'starting', 
            'start_time': start_time.isoformat()
        }
        
        # Initialize metrics
        all_metrics = {
            'soul_id': spark_id,
            'start_time': start_time.isoformat(),
            'stages': {},
            'success': False
        }
        
        # Use provided or internal field controller
        field_ctrl = field_controller or self.field_controller
        if not field_ctrl:
            raise ValueError("Field controller required for complete soul formation.")
        
        try:
            # Run all stages in sequence
            try:
                logger.info("STAGE 1: Spark Harmonization")
                soul_spark, harmonization_metrics = self.run_spark_harmonization(
                    soul_spark, 
                    show_visuals=show_visuals,
                    harmonization_iterations=kwargs.get('harmonization_iterations', HARMONIZATION_ITERATIONS)
                )
                all_metrics['stages']['spark_harmonization'] = harmonization_metrics
            except Exception as e:
                logger.error(f"Spark Harmonization failed: {e}")
                raise

            try:
                logger.info("STAGE 2: Guff Strengthening")
                soul_spark, guff_metrics = self.run_guff_strengthening(
                    soul_spark, 
                    field_controller=field_ctrl,
                    duration=kwargs.get('guff_duration', GUFF_STRENGTHENING_DURATION),
                    show_visuals=show_visuals
                )
                all_metrics['stages']['guff_strengthening'] = guff_metrics
            except Exception as e:
                logger.error(f"Guff Strengthening failed: {e}")
                raise

            try:
                logger.info("STAGE 3: Sephiroth Journey")
                journey_duration = kwargs.get('journey_duration_per_sephirah', 2.0)
                soul_spark, journey_metrics = self.run_sephiroth_journey(
                    soul_spark, 
                    field_controller=field_ctrl,
                    journey_duration_per_sephirah=journey_duration,
                    show_visuals=show_visuals
                )
                all_metrics['stages']['sephiroth_journey'] = journey_metrics
            except Exception as e:
                logger.error(f"Sephiroth Journey failed: {e}")
                raise

            try:
                logger.info("STAGE 4: Creator Entanglement")
                # Extract creator entanglement specific kwargs
                ce_kwargs = {
                    'base_creator_frequency': kwargs.get('base_creator_frequency', 432.0),
                    'resonance_intensity': kwargs.get('resonance_intensity', 0.7),
                    'entanglement_duration': kwargs.get('entanglement_duration', 3.0),
                    'quantum_coherence_threshold': kwargs.get('quantum_coherence_threshold', 0.8)
                }
                
                soul_spark, entanglement_metrics = perform_creator_entanglement(
                    soul_spark, 
                    kether_influencer, 
                    show_visuals=show_visuals, 
                    **ce_kwargs
                )
            
            # Save Creator Entanglement sensory data for soul echos processing
                try:
                    if 'sensory_captures' in entanglement_metrics and entanglement_metrics['sensory_captures']:
                        ce_sensory_data_path = save_creator_entanglement_sensory_data(
                            soul_spark, entanglement_metrics['sensory_captures']
                        )
                        entanglement_metrics['sensory_data_saved'] = {
                            'path': ce_sensory_data_path,
                            'captures_count': len(entanglement_metrics['sensory_captures']),
                            'success': True
                        }
                        logger.info(f"Saved Creator Entanglement sensory data: {ce_sensory_data_path}")
                    else:
                        logger.warning("No Creator Entanglement sensory captures to save")
                        entanglement_metrics['sensory_data_saved'] = {
                            'captures_count': 0,
                            'success': False,
                            'reason': 'no_captures'
                        }
                        
                except Exception as save_err:
                    logger.error(f"Failed to save Creator Entanglement sensory data: {save_err}")
                    entanglement_metrics['sensory_data_saved'] = {
                        'error': str(save_err),
                        'success': False
                    }
                
                all_metrics['stages']['creator_entanglement'] = entanglement_metrics
            except Exception as e:
                logger.error(f"Creator Entanglement failed: {e}")
                raise

            try:
                logger.info("STAGE 5: Harmonic Strengthening")
                # Extract harmonic strengthening specific kwargs
                hs_kwargs = {
                    'harmony_intensity': kwargs.get('harmony_intensity', HARMONIC_STRENGTHENING_INTENSITY_DEFAULT),
                    'harmony_duration_factor': kwargs.get('harmony_duration_factor', HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT)
                }
                soul_spark, harmonic_metrics = self.run_harmonic_strengthening(
                    soul_spark, 
                    show_visuals=show_visuals,
                    **hs_kwargs
                )
                all_metrics['stages']['harmonic_strengthening'] = harmonic_metrics
            except Exception as e:
                logger.error(f"Harmonic Strengthening failed: {e}")
                raise

            try:
                logger.info("STAGE 6: Life Cord Formation")
                # Extract life cord specific kwargs
                lc_kwargs = {
                    'intensity': kwargs.get('life_cord_intensity', 0.7),
                    'complexity': kwargs.get('cord_complexity', LIFE_CORD_COMPLEXITY_DEFAULT)
                }
                soul_spark, cord_metrics = self.run_life_cord_formation(
                    soul_spark, 
                    show_visuals=show_visuals,
                    **lc_kwargs
                )
                all_metrics['stages']['life_cord_formation'] = cord_metrics
            except Exception as e:
                logger.error(f"Life Cord Formation failed: {e}")
                raise

            try:
                logger.info("STAGE 7: Earth Harmonization")
                # Extract earth harmonization specific kwargs
                eh_kwargs = {
                    'schumann_intensity': kwargs.get('schumann_intensity', HARMONY_SCHUMANN_INTENSITY),
                    'core_intensity': kwargs.get('core_intensity', HARMONY_CORE_INTENSITY)
                }
                soul_spark, earth_metrics = self.run_earth_harmonization(
                    soul_spark, 
                    show_visuals=show_visuals,
                    **eh_kwargs
                )
                all_metrics['stages']['earth_harmonization'] = earth_metrics
            except Exception as e:
                logger.error(f"Earth Harmonization failed: {e}")
                raise

            try:
                logger.info("STAGE 8: Identity Crystallization")
                # Extract identity crystallization specific kwargs
                ic_kwargs = {
                    'train_cycles': kwargs.get('train_cycles', 7),
                    'entrainment_bpm': kwargs.get('entrainment_bpm', 72.0),
                    'entrainment_duration': kwargs.get('entrainment_duration', 120.0),
                    'love_cycles': kwargs.get('love_cycles', 5),
                    'geometry_stages': kwargs.get('geometry_stages', 2),
                    'crystallization_threshold': kwargs.get('crystallization_threshold', IDENTITY_CRYSTALLIZATION_THRESHOLD)
                }
                soul_spark, identity_metrics = self.run_identity_crystallization(
                    soul_spark, 
                    show_visuals=show_visuals,
                    **ic_kwargs
                )
                all_metrics['stages']['identity_crystallization'] = identity_metrics
            except Exception as e:
                logger.error(f"Identity Crystallization failed: {e}")
                raise

            try:
                logger.info("STAGE 9: Birth Process")
                # Extract birth process specific kwargs
                birth_kwargs = {
                    'birth_intensity': kwargs.get('birth_intensity', BIRTH_INTENSITY_DEFAULT),
                    'mother_profile': kwargs.get('mother_profile')
                }
                soul_spark, birth_metrics = self.run_birth_process(
                    soul_spark, 
                    show_visuals=show_visuals,
                    **birth_kwargs
                )
                all_metrics['stages']['birth_process'] = birth_metrics
            except Exception as e:
                logger.error(f"Birth Process failed: {e}")
                raise

            # Create comprehensive final reports
            try:
                logger.info("Creating Final Reports")
                report_paths = self.create_final_reports(soul_spark)
                all_metrics['report_paths'] = report_paths
            except Exception as e:
                logger.error(f"Final Reports generation failed: {e}")
                # Continue since reports are optional

            # Completion successful
            end_time = datetime.now()
            all_metrics['end_time'] = end_time.isoformat()
            all_metrics['duration_seconds'] = (end_time - start_time).total_seconds()
            all_metrics['success'] = True
            
            self.active_souls[spark_id] = {
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds()
            }
            
            # Record final metrics
            if METRICS_AVAILABLE:
                metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                    'action': 'complete_formation',
                    'soul_id': spark_id,
                    'success': True,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': (end_time - start_time).total_seconds(),
                    'stages_completed': list(all_metrics['stages'].keys())
                })
            
            logger.info(f"Complete formation SUCCESSFUL for soul {spark_id} in {(end_time - start_time).total_seconds():.1f}s")
            return soul_spark, all_metrics
            
        except Exception as e:
            # Record failure
            end_time = datetime.now()
            all_metrics['end_time'] = end_time.isoformat()
            all_metrics['duration_seconds'] = (end_time - start_time).total_seconds()
            all_metrics['error'] = str(e)
            all_metrics['success'] = False
            
            self.active_souls[spark_id] = {
                'status': 'failed',
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds()
            }
            
            # Record failure metrics
            if METRICS_AVAILABLE:
                metrics.record_metrics(CONTROLLER_METRIC_CATEGORY, {
                    'action': 'complete_formation',
                    'soul_id': spark_id,
                    'success': False,
                    'error': str(e),
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': (end_time - start_time).total_seconds(),
                    'stages_completed': list(all_metrics['stages'].keys())
                })
            
            logger.error(f"Complete formation FAILED for soul {spark_id}: {e}")
            raise RuntimeError(f"Complete soul formation failed: {e}") from e

    def get_active_souls_status(self) -> Dict[str, Dict[str, Any]]:
        """Returns the status of all active souls."""
        return self.active_souls.copy()

    def get_development_timeline(self, soul_id: str) -> List[Tuple[str, str]]:
        """Returns the development timeline for a specific soul."""
        if soul_id not in self.development_states:
            return []
        return [(getattr(soul, 'spark_id', 'unknown'), stage) 
                for soul, stage in self.development_states[soul_id]]              
