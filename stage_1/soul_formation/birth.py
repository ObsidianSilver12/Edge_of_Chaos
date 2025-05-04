# --- START OF FILE src/stage_1/soul_formation/birth.py ---

"""
Birth Process Functions (Refactored V4.3.8+ - Principle-Driven, Simplified Brain)

Handles birth into physical incarnation. Converts spiritual energy (SEU) to
physical brain energy (BEU) based on calculated needs + soul completeness buffer.
Creates a MINIMAL BrainSeed, allocates energy, connects soul, distributes aspects conceptually.
Deploys memory veil affecting per-aspect retention factors based on coherence.
Modifies SoulSpark directly. Hard fails on critical errors.
"""

import logging
import numpy as np
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import time
import uuid
from datetime import datetime

# --- Constants ---
# Add constants directly here for clarity during development
try:
    from constants.constants import *
    # Ensure necessary constants exist
    required = ['FLAG_READY_FOR_BIRTH', 'FLAG_IDENTITY_CRYSTALLIZED', 'BIRTH_PREREQ_CORD_INTEGRITY_MIN',
                'BIRTH_PREREQ_EARTH_RESONANCE_MIN', 'ENERGY_BRAIN_14_DAYS_JOULES', 'ENERGY_SCALE_FACTOR',
                'BRAIN_ENERGY_SCALE_FACTOR', 'BIRTH_ALLOC_SEED_CORE', 'BIRTH_ALLOC_MYCELIAL', # Using only these two allocations now
                'BIRTH_INTENSITY_DEFAULT', 'MAX_STABILITY_SU', 'MAX_COHERENCE_CU',
                'BIRTH_CONN_WEIGHT_RESONANCE', 'BIRTH_CONN_WEIGHT_INTEGRITY', 'BIRTH_CONN_MOTHER_STRENGTH_FACTOR',
                'BIRTH_CONN_STRENGTH_FACTOR', 'BIRTH_CONN_STRENGTH_CAP', 'BIRTH_CONN_TRAUMA_FACTOR',
                'BIRTH_CONN_MOTHER_TRAUMA_REDUCTION', 'BIRTH_ACCEPTANCE_MIN', 'BIRTH_ACCEPTANCE_TRAUMA_FACTOR',
                'BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR', 'BIRTH_VEIL_STRENGTH_BASE',
                'BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR', 'VEIL_BASE_RETENTION', 'VEIL_COHERENCE_RESISTANCE_FACTOR',
                'BIRTH_BREATH_AMP_BASE', 'BIRTH_BREATH_AMP_INTENSITY_FACTOR', 'BIRTH_BREATH_DEPTH_BASE',
                'BIRTH_BREATH_DEPTH_INTENSITY_FACTOR', 'EARTH_BREATH_FREQUENCY',
                'BIRTH_BREATH_INTEGRATION_CONN_FACTOR', 'BIRTH_FINAL_INTEGRATION_WEIGHT_CONN',
                'BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT', 'BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH',
                'BIRTH_FINAL_MOTHER_INTEGRATION_BOOST', 'BIRTH_FINAL_FREQ_SHIFT_FACTOR',
                'BIRTH_FINAL_STABILITY_PENALTY_FACTOR', 'FLAG_INCARNATED', 'FLOAT_EPSILON',
                'BIRTH_VEIL_MEMORY_RETENTION_MODS'] # Need this for veil calc
    for const_name in required:
        if const_name not in globals(): raise NameError(f"Constant '{const_name}' not found")
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Could not import constants: {e}. Birth process cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e
except NameError as e:
    logging.critical(f"CRITICAL ERROR: Missing required constant definition: {e}. Birth process cannot function.")
    raise NameError(f"Essential constant missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    # Import SIMPLIFIED Brain Seed factory and attachment functions
    # Assumes path is stage_2/brain_development/
    from stage_2.brain_development.brain_seed import BrainSeed, create_brain_seed
    from stage_2.brain_development.brain_soul_attachment import attach_soul_to_brain, distribute_soul_aspects
    # NOTE: Removed imports for brain_structure functions
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies for Birth: {e}", exc_info=True)
    raise ImportError(f"Core dependencies missing for Birth: {e}") from e

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()

# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites including identity flags. Raises ValueError on failure. """
    # ... (Implementation unchanged from previous correct version) ...
    logger.debug(f"Checking birth prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("Invalid SoulSpark object.")
    if not getattr(soul_spark, FLAG_READY_FOR_BIRTH, False): msg = f"Prereq failed: {FLAG_READY_FOR_BIRTH} missing."; logger.error(msg); raise ValueError(msg)
    if not getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False): msg = "Prereq failed: Identity not crystallized."; logger.error(msg); raise ValueError(msg)
    cord_integrity = getattr(soul_spark, "cord_integrity", -1.0); earth_resonance = getattr(soul_spark, "earth_resonance", -1.0)
    if cord_integrity < 0 or earth_resonance < 0: msg = "Prereq failed: Missing cord_integrity or earth_resonance."; logger.error(msg); raise AttributeError(msg)
    if cord_integrity < BIRTH_PREREQ_CORD_INTEGRITY_MIN: msg = f"Prereq failed: Cord integrity ({cord_integrity:.3f}) < {BIRTH_PREREQ_CORD_INTEGRITY_MIN})"; logger.error(msg); raise ValueError(msg)
    if earth_resonance < BIRTH_PREREQ_EARTH_RESONANCE_MIN: msg = f"Prereq failed: Earth Resonance ({earth_resonance:.3f}) < {BIRTH_PREREQ_EARTH_RESONANCE_MIN})"; logger.error(msg); raise ValueError(msg)
    logger.debug("Birth prerequisites (Flags, Factors) met. Energy check pending.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties for birth. Raises Error if fails. """
    # ... (Implementation unchanged from previous correct version) ...
    logger.debug(f"Ensuring properties for birth (Soul {soul_spark.spark_id})...")
    required = ['frequency','stability','coherence','spiritual_energy','physical_energy','cord_integrity','earth_resonance','life_cord','aspects','cumulative_sephiroth_influence'] # Added influence for completeness
    if not all(hasattr(soul_spark, attr) for attr in required): missing=[attr for attr in required if not hasattr(soul_spark, attr)]; raise AttributeError(f"SoulSpark missing attributes for Birth: {missing}")
    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    if not isinstance(soul_spark.life_cord, dict): raise TypeError("Missing 'life_cord' dict.")
    if soul_spark.spiritual_energy < 0: raise ValueError("Spiritual energy cannot be negative.")
    defaults={"memory_veil":None,"breath_pattern":None,"physical_integration":0.0,"incarnated":False,"birth_time":None,"brain_connection":None}
    for attr, default in defaults.items():
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None: setattr(soul_spark, attr, default)
    logger.debug("Soul properties ensured for Birth.")

def _calculate_soul_completeness(soul_spark: SoulSpark) -> float:
    """ Calculates a 0-1 factor representing soul completeness/integration post-journey. """
    # ... (Implementation unchanged from previous correct version) ...
    stab_factor=soul_spark.stability/MAX_STABILITY_SU; coh_factor=soul_spark.coherence/MAX_COHERENCE_CU; infl_factor=soul_spark.cumulative_sephiroth_influence; aspect_count=len(soul_spark.aspects); aspect_factor=min(1.0, aspect_count/60.0)
    completeness=(stab_factor*0.25 + coh_factor*0.25 + infl_factor*0.30 + aspect_factor*0.20)
    return max(0.0, min(1.0, completeness))

# --- Core Birth Functions ---

def _connect_to_physical_form(soul_spark: SoulSpark, intensity: float, mother_profile: Optional[Dict]) -> Tuple[float, float, Dict[str, Any]]:
    """ Calculates potential connection strength (0-1) and acceptance (0-1). Fails hard. """
    # ... (Implementation unchanged from previous correct version - Hard Fails) ...
    logger.info("Birth Phase: Calculating Physical Form Connection Potential...")
    if not(0.1<=intensity<=1.0): raise ValueError("Intensity out of range.")
    try:
        earth_resonance=soul_spark.earth_resonance; cord_integrity=soul_spark.cord_integrity
        mother_nurturing=mother_profile.get('nurturing_capacity',0.5) if mother_profile else 0.5; mother_spiritual=mother_profile.get('spiritual',{}).get('connection',0.5) if mother_profile else 0.5; mother_love=mother_profile.get('love_resonance',0.5) if mother_profile else 0.5
        base_strength=(earth_resonance*BIRTH_CONN_WEIGHT_RESONANCE+cord_integrity*BIRTH_CONN_WEIGHT_INTEGRITY)*(1.0+mother_spiritual*BIRTH_CONN_MOTHER_STRENGTH_FACTOR); connection_factor=intensity*BIRTH_CONN_STRENGTH_FACTOR; connection_strength=min(BIRTH_CONN_STRENGTH_CAP,max(0.0,base_strength*(1.0+connection_factor))); logger.debug(f"Physical Conn Strength -> {connection_strength:.4f}")
        trauma_base=intensity*BIRTH_CONN_TRAUMA_FACTOR; trauma_reduction=mother_nurturing*BIRTH_CONN_MOTHER_TRAUMA_REDUCTION; trauma_level=max(0.0,min(1.0,trauma_base-trauma_reduction)); acceptance_base=max(BIRTH_ACCEPTANCE_MIN,1.0-trauma_level*BIRTH_ACCEPTANCE_TRAUMA_FACTOR); acceptance_boost=mother_love*BIRTH_CONN_MOTHER_ACCEPTANCE_FACTOR; acceptance=min(1.0,max(0.0,acceptance_base+acceptance_boost)); logger.debug(f"Physical Form Acceptance -> {acceptance:.4f}")
        phase_metrics={"connection_strength_factor":float(connection_strength),"form_acceptance_factor":float(acceptance),"trauma_level_factor":float(trauma_level),"mother_influence_applied":mother_profile is not None,"timestamp":datetime.now().isoformat()}
        if METRICS_AVAILABLE: metrics.record_metrics('birth_physical_connection',phase_metrics)
        logger.info(f"Physical connection potential calculated. ConnFactor:{connection_strength:.3f}, AcceptFactor:{acceptance:.3f}")
        return float(connection_strength), float(acceptance), phase_metrics
    except Exception as e: logger.error(f"Error calculating physical form connection: {e}",exc_info=True); raise RuntimeError("Physical form connection calculation failed.") from e

def _deploy_memory_veil(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """ Deploys memory veil. Adds per-aspect retention factor (0-1). Fails hard. """
    # ... (Implementation unchanged from previous correct version - Hard Fails) ...
    logger.info("Birth Phase: Deploying memory veil...")
    if not(0.1<=intensity<=1.0): raise ValueError("Intensity out of range.")
    try:
        soul_coherence_norm=soul_spark.coherence/MAX_COHERENCE_CU; veil_strength=BIRTH_VEIL_STRENGTH_BASE+intensity*BIRTH_VEIL_STRENGTH_INTENSITY_FACTOR; veil_strength=max(0.0,min(1.0,veil_strength)); logger.debug(f"Memory Veil: Intensity={intensity:.2f}, CohNorm={soul_coherence_norm:.3f} -> VeilStrength={veil_strength:.4f}")
        deployment_time=datetime.now().isoformat(); veil_details={"strength_factor":veil_strength,"deployment_time":deployment_time,"aspect_retention":{}}; affected_count=0
        if not hasattr(soul_spark,'aspects') or not isinstance(soul_spark.aspects,dict): raise AttributeError("Cannot deploy veil: Soul aspects missing.")
        for aspect_name,aspect_data in soul_spark.aspects.items():
            if not isinstance(aspect_data,dict): continue
            base_retention=VEIL_BASE_RETENTION; coherence_resistance=soul_coherence_norm*VEIL_COHERENCE_RESISTANCE_FACTOR; type_mod=BIRTH_VEIL_MEMORY_RETENTION_MODS.get(aspect_name,0.0); retention_factor=base_retention+coherence_resistance+type_mod; final_retention=retention_factor*(1.0-veil_strength*0.9); final_retention=max(0.01,min(1.0,final_retention))
            aspect_data['retention_factor']=float(final_retention); veil_details['aspect_retention'][aspect_name]=float(final_retention); affected_count+=1; #logger.debug(f"Veil Applied - '{aspect_name}': FinalRetention={final_retention:.4f}")
        setattr(soul_spark,"memory_veil",veil_details);
        if hasattr(soul_spark,'memory_retention'): delattr(soul_spark,'memory_retention') # Remove old single factor
        soul_spark.last_modified=deployment_time
        avg_retention = np.mean(list(veil_details['aspect_retention'].values())) if veil_details['aspect_retention'] else 0.0
        phase_metrics={"veil_strength_factor":veil_strength,"aspects_veiled_count":affected_count,"avg_retention_factor":avg_retention,"timestamp":deployment_time}
        if METRICS_AVAILABLE: metrics.record_metrics('birth_memory_veil',phase_metrics)
        logger.info(f"Memory veil deployed. Strength:{veil_strength:.3f}, Aspects:{affected_count}, AvgRet:{avg_retention:.3f}")
        return phase_metrics
    except Exception as e: logger.error(f"Error deploying memory veil: {e}",exc_info=True); raise RuntimeError("Memory veil deployment failed critically.") from e

def _integrate_first_breath(soul_spark: SoulSpark, physical_connection: float, form_acceptance: float, intensity: float) -> Dict[str, Any]: # Pass acceptance
    """ Integrates first breath rhythm. Calculates physical integration factor. Fails hard. """
    logger.info("Birth Phase: Integrating first breath...")
    if not(0.0<=physical_connection<=1.0): raise ValueError("physical_connection invalid.")
    if not(0.0<=form_acceptance<=1.0): raise ValueError("form_acceptance invalid.")
    if not(0.1<=intensity<=1.0): raise ValueError("Intensity invalid.")
    try:
        earth_resonance=soul_spark.earth_resonance
        breath_amplitude=max(0.0,min(1.0,BIRTH_BREATH_AMP_BASE+intensity*BIRTH_BREATH_AMP_INTENSITY_FACTOR)); breath_depth=max(0.0,min(1.0,BIRTH_BREATH_DEPTH_BASE+intensity*BIRTH_BREATH_DEPTH_INTENSITY_FACTOR)); logger.debug(f"First Breath: Amp={breath_amplitude:.3f}, Depth={breath_depth:.3f}")
        integration_strength=(physical_connection*BIRTH_FINAL_INTEGRATION_WEIGHT_CONN+form_acceptance*BIRTH_FINAL_INTEGRATION_WEIGHT_ACCEPT+breath_depth*BIRTH_FINAL_INTEGRATION_WEIGHT_BREATH); integration_strength+=earth_resonance*0.1 # Small bonus
        total_integration_factor=min(1.0,max(0.0,integration_strength)); logger.debug(f"First Breath -> IntegrationFactor={total_integration_factor:.4f}")
        breath_time=datetime.now().isoformat(); breath_config={"target_frequency_hz":float(EARTH_BREATH_FREQUENCY),"initial_amplitude_factor":breath_amplitude,"initial_depth_factor":breath_depth,"integration_achieved_factor":total_integration_factor,"timestamp":breath_time}
        setattr(soul_spark,"breath_pattern",breath_config); setattr(soul_spark,"physical_integration",float(total_integration_factor)); soul_spark.last_modified=breath_time
        phase_metrics={"physical_integration_factor":total_integration_factor,"timestamp":breath_time}
        if METRICS_AVAILABLE: metrics.record_metrics('birth_first_breath',phase_metrics)
        logger.info(f"First breath integrated. Physical Integration Factor: {total_integration_factor:.4f}")
        return phase_metrics
    except Exception as e: logger.error(f"Error integrating first breath: {e}",exc_info=True); raise RuntimeError("First breath integration failed critically.") from e


def _finalize_birth_state(soul_spark: SoulSpark) -> Dict[str, Any]:
    """ Applies final adjustments to soul state due to embodiment. Fails hard. """
    logger.info("Birth Phase: Finalizing soul state...")
    try:
        initial_freq=soul_spark.frequency; initial_stability=soul_spark.stability; physical_integration=soul_spark.physical_integration
        freq_shift_percentage=BIRTH_FINAL_FREQ_SHIFT_FACTOR*(1.0-physical_integration); freq_reduction=initial_freq*freq_shift_percentage; new_freq=max(FLOAT_EPSILON*10,initial_freq-freq_reduction); soul_spark.frequency=float(new_freq); logger.debug(f"Finalize State: Freq {initial_freq:.1f}->{new_freq:.1f}")
        if hasattr(soul_spark,'_validate_or_init_frequency_structure'): soul_spark._validate_or_init_frequency_structure()
        else: raise AttributeError("Missing frequency structure update method.")
        stability_penalty_percentage=BIRTH_FINAL_STABILITY_PENALTY_FACTOR*(1.0-physical_integration); stability_reduction=initial_stability*stability_penalty_percentage; new_stability=max(0.0,min(MAX_STABILITY_SU,initial_stability-stability_reduction)); soul_spark.stability=float(new_stability); logger.debug(f"Finalize State: Stability {initial_stability:.1f}->{new_stability:.1f}")
        setattr(soul_spark,FLAG_INCARNATED,True); birth_time=datetime.now().isoformat(); setattr(soul_spark,"birth_time",birth_time); soul_spark.last_modified=birth_time
        phase_metrics={"final_frequency_hz":new_freq,"frequency_change_hz":new_freq-initial_freq,"final_stability_su":new_stability,"stability_change_su":new_stability-initial_stability,"birth_timestamp":birth_time,"success":True}
        if METRICS_AVAILABLE: metrics.record_metrics('birth_finalization',phase_metrics)
        logger.info(f"Birth state finalized. Final Freq: {new_freq:.1f} Hz, Final Stability: {new_stability:.1f} SU")
        return phase_metrics
    except Exception as e: logger.error(f"Error finalizing birth state: {e}",exc_info=True); raise RuntimeError("Birth finalization failed critically.") from e


# --- Orchestration Function ---
def perform_birth(soul_spark: SoulSpark,
                 intensity: float = BIRTH_INTENSITY_DEFAULT,
                 mother_profile: Optional[Dict[str, Any]] = None,
                 brain_complexity: int = 9
                 ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs complete birth process. Fails hard. """
    # --- Input Validation & Setup ---
    if not isinstance(soul_spark,SoulSpark): raise TypeError("soul_spark invalid.")
    if not (0.1<=intensity<=1.0): raise ValueError("Intensity invalid.")
    if not isinstance(brain_complexity,int) or not (1<=brain_complexity<=10): raise ValueError("brain_complexity invalid.")
    spark_id=getattr(soul_spark,'spark_id','unknown'); log_msg_suffix="with Mother Profile" if mother_profile else "without Mother Profile"; logger.info(f"--- Starting Birth Process for Soul {spark_id} (Int={intensity:.2f}) {log_msg_suffix} ---")
    start_time_iso=datetime.now().isoformat(); start_time_dt=datetime.fromisoformat(start_time_iso); process_metrics_summary={'steps':{}}; brain_seed=None

    try:
        _ensure_soul_properties(soul_spark); _check_prerequisites(soul_spark)
        initial_state={'stability_su':soul_spark.stability,'coherence_cu':soul_spark.coherence,'spiritual_energy_seu':soul_spark.spiritual_energy,'physical_energy_seu':soul_spark.physical_energy,'earth_resonance':soul_spark.earth_resonance,'cord_integrity':soul_spark.cord_integrity,'physical_integration':soul_spark.physical_integration}
        logger.info(f"Birth Initial State: S={initial_state['stability_su']:.1f}, C={initial_state['coherence_cu']:.1f}, E_spirit={initial_state['spiritual_energy_seu']:.1f}")

        # --- 1. Calculate Connection Potential ---
        logger.info("Birth Step 1: Calculating Connection Potential...")
        connection_strength, form_acceptance, metrics_conn = _connect_to_physical_form(soul_spark, intensity, mother_profile) # Fails hard
        process_metrics_summary['steps']['connection_potential'] = metrics_conn

        # --- 2. Energy Check & Conversion ---
        logger.info("Birth Step 2: Energy Check & Conversion...")
        soul_completeness=_calculate_soul_completeness(soul_spark); buffer_factor=1.0+(0.4*soul_completeness); required_joules=ENERGY_BRAIN_14_DAYS_JOULES; total_joules_needed=required_joules*buffer_factor; required_physical_seu=total_joules_needed*ENERGY_SCALE_FACTOR; logger.debug(f"Energy Need: BaseJ={required_joules:.2E}, SoulCompl={soul_completeness:.3f}, BufferF={buffer_factor:.2f} -> TotalSEU={required_physical_seu:.1f}")
        if soul_spark.spiritual_energy < required_physical_seu: raise ValueError(f"Insufficient spiritual energy ({soul_spark.spiritual_energy:.1f} SEU) for birth ({required_physical_seu:.1f} SEU).") # Hard Fail
        energy_converted_seu=required_physical_seu; soul_spark.physical_energy=energy_converted_seu; soul_spark.spiritual_energy-=energy_converted_seu; logger.info(f"Energy Converted: {energy_converted_seu:.1f} SEU.")
        process_metrics_summary['steps']['energy_conversion']={'energy_converted_seu':energy_converted_seu,'buffer_factor_used':buffer_factor,'brain_need_joules':required_joules}

        # --- 3. Brain Seed Creation & Integration ---
        logger.info("Birth Step 3: Creating MINIMAL Brain Seed & Integrating...")
        # 3a. Convert Soul's Physical Energy (SEU) to Brain Energy (BEU)
        physical_joules = soul_spark.physical_energy / ENERGY_SCALE_FACTOR
        total_available_beu = physical_joules * BRAIN_ENERGY_SCALE_FACTOR
        logger.debug(f"Converting {soul_spark.physical_energy:.1f} SEU to {total_available_beu:.2E} BEU.")
        # 3b. Allocate BEU
        core_beu = total_available_beu * BIRTH_ALLOC_SEED_CORE # Use only core allocation for minimal seed
        mycelial_beu = total_available_beu * (BIRTH_ALLOC_REGIONS + BIRTH_ALLOC_MYCELIAL) # Rest goes to store
        # 3c. Create Minimal Brain Seed (passes energy directly)
        brain_seed = create_brain_seed(soul_spark, initial_beu=core_beu, initial_mycelial_beu=mycelial_beu) # Fails hard
        logger.info(f"Minimal Brain Seed Created. Core BEU:{core_beu:.2E}, Mycelial BEU:{mycelial_beu:.2E}")
        # --- REMOVED calls to brain_structure development functions ---
        # 3d. Attach Soul to MINIMAL Brain Seed
        attach_metrics = attach_soul_to_brain(soul_spark, brain_seed) # Fails hard
        # 3e. Distribute Soul Aspects Conceptually
        dist_metrics = distribute_soul_aspects(soul_spark, brain_seed) # Fails hard
        process_metrics_summary['steps']['brain_integration']={'attach':attach_metrics,'distribute':dist_metrics,'initial_brain_beu':total_available_beu}

        # --- 4. First Breath & Physical Integration ---
        logger.info("Birth Step 4: Integrating First Breath...")
        # Use acceptance calculated in step 1
        metrics_breath = _integrate_first_breath(soul_spark, connection_strength, form_acceptance, intensity) # Fails hard
        process_metrics_summary['steps']['first_breath'] = metrics_breath

        # --- 5. Memory Veil Deployment ---
        logger.info("Birth Step 5: Deploying Memory Veil...")
        metrics_veil = _deploy_memory_veil(soul_spark, intensity) # Fails hard
        process_metrics_summary['steps']['memory_veil'] = metrics_veil

        # --- 6. Finalize State ---
        logger.info("Birth Step 6: Finalizing Soul State...")
        metrics_final = _finalize_birth_state(soul_spark) # Fails hard
        process_metrics_summary['steps']['finalization'] = metrics_final

        # --- 7. Final State Update ---
        logger.info("Birth Step 7: Final Soul State Update...")
        if hasattr(soul_spark,'update_state'): soul_spark.update_state(); logger.debug(f"Birth S/C after final update: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        else: raise AttributeError("SoulSpark missing 'update_state' method.")

        # --- Compile Overall Metrics ---
        end_time_iso=soul_spark.last_modified; end_time_dt=datetime.fromisoformat(end_time_iso)
        final_state={'stability_su':soul_spark.stability,'coherence_cu':soul_spark.coherence,'spiritual_energy_seu':soul_spark.spiritual_energy,'physical_energy_seu':soul_spark.physical_energy,'earth_resonance':soul_spark.earth_resonance,'cord_integrity':soul_spark.cord_integrity,'physical_integration':soul_spark.physical_integration,FLAG_INCARNATED:getattr(soul_spark,FLAG_INCARNATED)}
        overall_metrics={'action':'birth','soul_id':spark_id,'start_time':start_time_iso,'end_time':end_time_iso,'duration_seconds':(end_time_dt-start_time_dt).total_seconds(),'intensity_setting':intensity,'mother_influence_active':mother_profile is not None,'brain_complexity_setting':brain_complexity,'initial_state':initial_state,'final_state':final_state,'energy_converted_seu':energy_converted_seu,'brain_seed_initial_beu':total_available_beu,'final_physical_integration':final_state['physical_integration'],'stability_change_su':final_state['stability_su']-initial_state['stability_su'],'coherence_change_cu':final_state['coherence_cu']-initial_state['coherence_cu'],'success':True}
        if METRICS_AVAILABLE: metrics.record_metrics('birth_summary',overall_metrics)
        logger.info(f"--- Birth Process Completed Successfully for Soul {spark_id} ---")
        # Optionally save brain seed state here
        # try: brain_seed.save_state(f"output/brain_seeds/brain_{spark_id}.json")
        # except Exception as save_err: logger.error(f"Failed to save brain seed state: {save_err}")
        return soul_spark, overall_metrics

    # --- Error Handling (Hard Fail) ---
    except (ValueError, TypeError, AttributeError, RuntimeError) as e_val:
         logger.error(f"Birth process failed for {spark_id}: {e_val}", exc_info=False) # Less verbose for expected
         failed_step=list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'prerequisites'
         record_birth_failure(spark_id,start_time_iso,failed_step,str(e_val),mother_profile is not None)
         setattr(soul_spark,FLAG_INCARNATED,False); raise e_val # Re-raise
    except Exception as e:
         logger.critical(f"Unexpected error during birth for {spark_id}: {e}", exc_info=True)
         failed_step=list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'unexpected'
         record_birth_failure(spark_id,start_time_iso,failed_step,str(e),mother_profile is not None)
         setattr(soul_spark,FLAG_INCARNATED,False); raise RuntimeError(f"Unexpected birth process failure: {e}") from e


# --- Failure Metric Helper ---
def record_birth_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str, mother_active: bool):
    """ Helper to record failure metrics consistently. """
    # ... (Implementation unchanged from previous correct version) ...
    if METRICS_AVAILABLE:
        try:
            end_time=datetime.now().isoformat(); duration=(datetime.fromisoformat(end_time)-datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('birth_summary',{'action':'birth','soul_id':spark_id,'start_time':start_time_iso,'end_time':end_time,'duration_seconds':duration,'mother_influence_active':mother_active,'success':False,'error':error_msg,'failed_step':failed_step})
        except Exception as metric_e: logger.error(f"Failed record birth failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/birth.py ---
