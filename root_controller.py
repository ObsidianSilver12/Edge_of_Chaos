# --- START OF FILE root_controller.py (V5.5 - With Complete Field Visualization Support) ---
"""
Root Controller (V5.5 - With Complete Field Visualization Support)

This root controller serves as the highest-level orchestrator. It follows a clean
architectural pattern:
1. Initialize the environment (FieldController).
2. Initialize the agent (SoulCompletionController), injecting the environment into it.
3. Run the agent's main process.
4. Handle comprehensive visualization results including field visualizations and cleanup.

Updated to properly handle and log the complete field visualization system including
void field states, edge of chaos, sephiroth tree, and field system dashboard.
"""

import logging
import os
import sys
from datetime import datetime
import json
import time
import traceback

# --- TOP-LEVEL IMPORTS: Any failure here will stop the script immediately ---
try:
    from shared.constants.constants import GRID_DIMENSIONS, LOG_LEVEL, LOG_FORMAT, DATA_DIR_BASE, IDENTITY_CRYSTALLIZATION_THRESHOLD
    logger = logging.getLogger("RootController")
    log_file_path = os.path.join(DATA_DIR_BASE, "logs", "root_controller_run.log")
    if not logger.handlers:
        log_level_int = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
        logger.setLevel(log_level_int)
        log_formatter = logging.Formatter(LOG_FORMAT)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(log_formatter)
        logger.addHandler(ch)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
    logger.info("RootController Logging configured (Terminal: INFO, File: DEBUG).")

    # Import the controllers and systems
    from stage_1.fields.field_controller import FieldController
    try:
        from stage_1.soul_completion_controller import SoulCompletionController
    except ImportError as exc:
        raise ImportError("The module 'stage_1.soul_completion_controller' could not be found. Please ensure the file exists and the path is correct.") from exc
    import metrics_tracking as metrics
except (ImportError, NameError) as e:
    print(f"\n{'='*80}\nFATAL ROOT ERROR - FAILED TO INITIALIZE\n{'='*80}")
    print(f"An essential module or constant failed to import: {e}")
    traceback.print_exc()
    print("Please ensure all `__init__.py` files are in place and that the script is run from the project's root directory.")
    sys.exit(1)


def run_simulation(**kwargs):
    """Initializes the environment and runs a full soul completion simulation."""
    sim_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("--- Starting Soul Development Simulation Run: %s ---", sim_id)
    overall_start_time = time.time()
    field_controller = None

    try:
        # --- Step 1: Initialize the Environment ---
        logger.info("RootController Step 1: Initializing Field Controller (Environment)...")
        field_controller = FieldController(grid_size=GRID_DIMENSIONS)
        logger.info("RootController: Field Controller environment initialized successfully.")

        # --- Step 2: Initialize the Agent (Soul Controller) and inject the environment ---
        logger.info("RootController Step 2: Initializing Soul Completion Controller (Agent)...")
        soul_controller = SoulCompletionController(
            simulation_id=sim_id,
            field_controller=field_controller # Dependency Injection
        )
        logger.info("RootController: Soul Completion Controller initialized successfully.")

        # --- Step 3: Run the main process ---
        logger.info("RootController Step 3: Executing soul completion process...")
        
        # Log visualization directories that will be created
        visualization_base = soul_controller.visualization_dir
        logger.info(f"RootController: Visualizations will be saved to: {visualization_base}")
        logger.info("  - Soul evolution stages in: soul_evolution/")
        logger.info("  - Brain structures in: brain_structure/")  
        logger.info("  - Field states and dashboard in: fields/")
        logger.info("  - Comprehensive soul evolution in: soul_evolution_complete/")
        logger.info("  - Comprehensive brain structure in: brain_structure_complete/")
        
        soul_controller.run_full_soul_completion(**kwargs)
        logger.info("RootController: Soul completion process finished.")
        
        # Log comprehensive visualization results including field visualizations
        _log_comprehensive_visualization_results(soul_controller, visualization_base)
        
        # Log standard visualization summary including field files
        _log_standard_visualization_summary(visualization_base)

        # --- Final Summary ---
        total_duration = time.time() - overall_start_time
        success_status = soul_controller.results.get('success', False)
        logger.info("--- Simulation Run %s Finished ---", sim_id)
        logger.info("Total Duration: %.2f seconds", total_duration)
        logger.info("Final Status: %s", 'SUCCESS' if success_status else 'FAILED')
        
        if success_status:
            _handle_successful_simulation(soul_controller, sim_id)
        else:
            _handle_failed_simulation(soul_controller, sim_id)

    except (RuntimeError, ValueError, TypeError, AttributeError) as e:
        logger.critical("RootController: Simulation %s aborted due to a critical error: %s",
                      sim_id, e, exc_info=True)
        print(f"\n{'='*80}\nCRITICAL SIMULATION ERROR - RUN ABORTED\n{'='*80}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(f"See log file for full traceback: {log_file_path}")
        print("="*80)

    finally:
        # This block ensures that no matter what happens, we try to save our metrics.
        try:
            logger.info("RootController: Persisting final metrics...")
            metrics.persist_metrics()
            logger.info("RootController: Final metrics persisted.")
        except Exception as persist_e:
            logger.error("RootController: ERROR persisting metrics at end of simulation: %s", persist_e)
        logging.shutdown()


def _log_comprehensive_visualization_results(soul_controller, visualization_base):
    """Log results from comprehensive visualizations including field system results."""
    try:
        comprehensive_results = soul_controller.results.get('comprehensive_visualizations', {})
        
        if comprehensive_results:
            logger.info("=== COMPREHENSIVE VISUALIZATION RESULTS ===")
            
            # Soul Evolution Results
            soul_evolution = comprehensive_results.get('soul_evolution', {})
            if soul_evolution.get('success'):
                files_created = soul_evolution.get('files_created', [])
                total_visualizations = soul_evolution.get('total_visualizations', 0)
                logger.info(f"✓ Soul Evolution: {total_visualizations} comprehensive visualizations created")
                
                for file_info in files_created:
                    stage = file_info.get('stage', 'Unknown Stage')
                    filename = file_info.get('filename', 'Unknown File')
                    logger.info(f"  - {stage}: {filename}")
                    
                # Log summary file
                summary_file = soul_evolution.get('summary_file')
                if summary_file:
                    logger.info(f"  - Evolution Summary: {os.path.basename(summary_file)}")
            else:
                logger.warning("✗ Soul Evolution: Comprehensive visualization failed")
            
            # Brain Structure Results
            brain_structure = comprehensive_results.get('brain_structure', {})
            if brain_structure.get('success'):
                files_created = brain_structure.get('files_created', [])
                total_visualizations = brain_structure.get('total_visualizations', 0)
                logger.info(f"✓ Brain Structure: {total_visualizations} comprehensive visualizations created")
                
                for file_info in files_created:
                    viz_type = file_info.get('visualization', 'Unknown Type')
                    filename = file_info.get('filename', 'Unknown File')
                    logger.info(f"  - {viz_type}: {filename}")
                    
                # Log summary file
                summary_file = brain_structure.get('summary_file')
                if summary_file:
                    logger.info(f"  - Brain Summary: {os.path.basename(summary_file)}")
            else:
                logger.warning("✗ Brain Structure: Comprehensive visualization failed")
            
            # Field System Results (NEW)
            field_system = comprehensive_results.get('field_system', {})
            if field_system.get('success'):
                dashboard_created = field_system.get('dashboard_created', False)
                logger.info(f"✓ Field System: Comprehensive dashboard visualization {'created' if dashboard_created else 'failed'}")
                
                if dashboard_created:
                    logger.info("  - Field Dashboard: field_system_dashboard.png")
                    logger.info("  - Includes: Void field energy, Edge of Chaos, Sephiroth Tree, Field metrics")
                    logger.info("  - Includes: Phi resonance distribution, System status overview")
                else:
                    logger.warning("  - Field dashboard creation failed")
            else:
                logger.warning("✗ Field System: Dashboard visualization failed")
            
            # Check for field dashboard completion status from soul controller results
            field_dashboard_complete = soul_controller.results.get('field_dashboard_complete', False)
            if field_dashboard_complete:
                logger.info("✓ Field Dashboard: Successfully integrated with comprehensive visualization system")
            else:
                logger.warning("✗ Field Dashboard: Integration with comprehensive system incomplete")
                
            logger.info("=== END COMPREHENSIVE VISUALIZATION RESULTS ===")
        else:
            logger.info("No comprehensive visualization results found")
            
    except Exception as viz_log_err:
        logger.error(f"Error logging comprehensive visualization results: {viz_log_err}")


def _log_standard_visualization_summary(visualization_base):
    """Log summary of all visualizations including field visualizations."""
    try:
        viz_summary = {
            'soul_viz_files': 0,
            'soul_viz_comprehensive': 0,
            'brain_viz_files': 0, 
            'brain_viz_comprehensive': 0,
            'field_viz_files': 0,
            'field_viz_types': [],
            'total_png_files': 0,
            'total_html_files': 0,
            'total_json_files': 0
        }
        
        # Count standard soul evolution files
        soul_evolution_dir = os.path.join(visualization_base, "soul_evolution")
        if os.path.exists(soul_evolution_dir):
            soul_files = os.listdir(soul_evolution_dir)
            viz_summary['soul_viz_files'] = len([f for f in soul_files if f.endswith(('.json', '.html', '.png'))])
            
        # Count comprehensive soul evolution files
        soul_evolution_complete_dir = os.path.join(visualization_base, "soul_evolution_complete")
        if os.path.exists(soul_evolution_complete_dir):
            soul_comprehensive_files = os.listdir(soul_evolution_complete_dir)
            viz_summary['soul_viz_comprehensive'] = len([f for f in soul_comprehensive_files if f.endswith(('.png', '.json'))])
            
        # Count standard brain structure files
        brain_structure_dir = os.path.join(visualization_base, "brain_structure")
        if os.path.exists(brain_structure_dir):
            brain_files = os.listdir(brain_structure_dir)
            viz_summary['brain_viz_files'] = len([f for f in brain_files if f.endswith(('.json', '.html', '.png'))])
            
        # Count comprehensive brain structure files
        brain_structure_complete_dir = os.path.join(visualization_base, "brain_structure_complete")
        if os.path.exists(brain_structure_complete_dir):
            brain_comprehensive_files = os.listdir(brain_structure_complete_dir)
            viz_summary['brain_viz_comprehensive'] = len([f for f in brain_comprehensive_files if f.endswith(('.png', '.json'))])
            
        # Count field visualization files (ENHANCED)
        fields_dir = os.path.join(visualization_base, "fields")
        if os.path.exists(fields_dir):
            field_files = [f for f in os.listdir(fields_dir) if f.endswith('.png')]
            viz_summary['field_viz_files'] = len(field_files)
            
            # Categorize field visualization types based on filenames
            field_types = set()
            for filename in field_files:
                if 'void_field' in filename:
                    if 'energy' in filename:
                        field_types.add('Void Field Energy')
                    elif 'coherence' in filename:
                        field_types.add('Void Field Coherence')
                    elif 'frequency' in filename:
                        field_types.add('Void Field Frequency')
                elif 'edge_of_chaos' in filename:
                    field_types.add('Edge of Chaos')
                elif 'sephiroth_tree' in filename:
                    field_types.add('Sephiroth Tree')
                elif 'sephiroth_field' in filename:
                    field_types.add('Individual Sephiroth')
                elif 'soul_field_interaction' in filename:
                    field_types.add('Soul-Field Interaction')
                elif 'field_frequency_spectrum' in filename:
                    field_types.add('Frequency Spectrum')
                elif 'field_system_dashboard' in filename:
                    field_types.add('System Dashboard')
                    
            viz_summary['field_viz_types'] = sorted(list(field_types))
            
        # Count total files by type across all directories
        for root, dirs, files in os.walk(visualization_base):
            for file in files:
                if file.endswith('.png'):
                    viz_summary['total_png_files'] += 1
                elif file.endswith('.html'):
                    viz_summary['total_html_files'] += 1
                elif file.endswith('.json'):
                    viz_summary['total_json_files'] += 1
        
        logger.info("=== VISUALIZATION FILE SUMMARY ===")
        logger.info(f"Soul Evolution (Standard): {viz_summary['soul_viz_files']} files")
        logger.info(f"Soul Evolution (Comprehensive): {viz_summary['soul_viz_comprehensive']} files")
        logger.info(f"Brain Structure (Standard): {viz_summary['brain_viz_files']} files")
        logger.info(f"Brain Structure (Comprehensive): {viz_summary['brain_viz_comprehensive']} files")
        logger.info(f"Field Visualizations: {viz_summary['field_viz_files']} files")
        
        # Log field visualization types
        if viz_summary['field_viz_types']:
            logger.info("Field Visualization Types Created:")
            for field_type in viz_summary['field_viz_types']:
                logger.info(f"  - {field_type}")
        else:
            logger.info("No field visualizations detected")
            
        logger.info(f"Total PNG Files: {viz_summary['total_png_files']}")
        logger.info(f"Total HTML Files: {viz_summary['total_html_files']}")
        logger.info(f"Total JSON Files: {viz_summary['total_json_files']}")
        logger.info("=== END VISUALIZATION FILE SUMMARY ===")
        
    except Exception as viz_summary_err:
        logger.warning(f"Could not create detailed visualization summary: {viz_summary_err}")


def _handle_successful_simulation(soul_controller, sim_id):
    """Handle successful simulation completion and data organization."""
    try:
        logger.info("RootController: Organizing completed soul data...")
        import subprocess
        import sys
        
        # Extract soul information for proper naming
        soul_name = "Unknown"
        birth_date = "Unknown"
        soul_summary = soul_controller.results.get('soul_summary', {})
        
        # Try to extract soul name and birth date from various sources
        if 'soul_data' in soul_summary:
            soul_name = soul_summary['soul_data'].get('name', soul_name)
            birth_date = soul_summary['soul_data'].get('birth_date', birth_date)
        elif 'name' in soul_summary:
            soul_name = soul_summary.get('name', soul_name)
        elif 'soul_id' in soul_summary:
            # Fallback: use soul_id if available
            soul_name = soul_summary.get('soul_id', f'soul_{sim_id}')
        
        # Create meaningful model name: SoulName_BirthDate
        if birth_date != "Unknown":
            model_name = f"{soul_name}_{birth_date.replace('-', '')}"
        else:
            model_name = f"{soul_name}_{sim_id}"
        
        logger.info(f"Organizing soul data for: {model_name}")
        
        # Log comprehensive visualization completion status including field system
        comprehensive_results = soul_controller.results.get('comprehensive_visualizations', {})
        soul_evolution_success = comprehensive_results.get('soul_evolution', {}).get('success', False)
        brain_structure_success = comprehensive_results.get('brain_structure', {}).get('success', False)
        field_system_success = comprehensive_results.get('field_system', {}).get('success', False)
        
        logger.info(f"Comprehensive Soul Evolution Visualizations: {'✓ SUCCESS' if soul_evolution_success else '✗ FAILED'}")
        logger.info(f"Comprehensive Brain Structure Visualizations: {'✓ SUCCESS' if brain_structure_success else '✗ FAILED'}")
        logger.info(f"Comprehensive Field System Dashboard: {'✓ SUCCESS' if field_system_success else '✗ FAILED'}")
        
        # Call cleanup script to organize data
        cleanup_result = subprocess.run([
            sys.executable, 'simulation_cleanup.py', 
            '--model-name', model_name
        ], capture_output=True, text=True, cwd='.')
        
        if cleanup_result.returncode == 0:
            logger.info("RootController: Soul data organization completed successfully")
            print(f"\n{'='*80}")
            print("SOUL DATA ORGANIZED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Soul Name: {soul_name}")
            print(f"Birth Date: {birth_date}")
            print(f"Simulation ID: {sim_id}")
            print(f"Data organized in: shared/output/completed_souls/{model_name}_*")
            
            # Print comprehensive visualization status including field system
            success_count = sum([soul_evolution_success, brain_structure_success, field_system_success])
            if success_count == 3:
                print("✓ All comprehensive visualizations completed successfully")
                print("  - Soul evolution stages and progression")
                print("  - Brain structure and neural networks")
                print("  - Field system states and dashboard")
            elif success_count > 0:
                print(f"⚠ {success_count}/3 comprehensive visualizations completed successfully")
                if soul_evolution_success:
                    print("  ✓ Soul evolution visualizations")
                if brain_structure_success:
                    print("  ✓ Brain structure visualizations")
                if field_system_success:
                    print("  ✓ Field system dashboard")
            else:
                print("✗ Comprehensive visualizations failed")
                
            print(f"System cleaned and ready for next simulation")
            print(f"{'='*80}")
        else:
            logger.warning(f"RootController: Data organization completed with warnings: {cleanup_result.stderr}")
            
    except Exception as cleanup_err:
        logger.error(f"RootController: Failed to organize soul data: {cleanup_err}")
        print(f"WARNING: Could not organize completed soul data: {cleanup_err}")


def _handle_failed_simulation(soul_controller, sim_id):
    """Handle failed simulation cleanup."""
    logger.error("Failed at stage: %s", soul_controller.results.get('failed_stage', 'Unknown'))
    logger.error("Error: %s", soul_controller.results.get('error', 'Unknown'))
    
    # Clean up failed simulation data
    try:
        logger.info("RootController: Cleaning up failed simulation data...")
        import subprocess
        import sys
        
        # Call cleanup script with --failed flag to delete files instead of organizing them
        cleanup_result = subprocess.run([
            sys.executable, 'simulation_cleanup.py', 
            '--model-name', f'failed_soul_{sim_id}',
            '--failed'
        ], capture_output=True, text=True, cwd='.')
        
        if cleanup_result.returncode == 0:
            logger.info("RootController: Failed simulation cleanup completed successfully")
            print(f"\n{'='*80}")
            print("FAILED SIMULATION CLEANED UP")
            print(f"{'='*80}")
            print(f"Simulation ID: {sim_id}")
            print("All output files deleted, cache cleared")
            print("System ready for next simulation attempt")
            print(f"{'='*80}")
        else:
            logger.warning(f"RootController: Failed simulation cleanup completed with warnings: {cleanup_result.stderr}")
            
    except Exception as cleanup_err:
        logger.error(f"RootController: Failed to cleanup failed simulation: {cleanup_err}")
        print(f"WARNING: Could not cleanup failed simulation data: {cleanup_err}")


if __name__ == "__main__":
    simulation_params = {
        'train_cycles': 7,
        'entrainment_bpm': 72.0,
        'entrainment_duration': 120.0,
        'love_cycles': 5,
        'geometry_stages': 2,
        'crystallization_threshold': IDENTITY_CRYSTALLIZATION_THRESHOLD,  # Use constant from shared/constants
        'birth_intensity': 0.7,
    }
    run_simulation(**simulation_params)

# --- END OF FILE root_controller.py ---




















