# --- START OF FILE stage_1/soul_formation/soul_completion_controller.py (V5.4 - With Comprehensive Visualizations) ---
"""
Soul Completion Controller (V5.4 - With Comprehensive Visualizations)

Orchestrates the entire soul development and formation process. This controller
RECEIVES a pre-initialized FieldController from a higher-level orchestrator
(e.g., root_controller.py), thus avoiding circular dependencies and ensuring a
clean separation of concerns between environment and agent.

Added comprehensive soul evolution and brain structure visualizations using Plotly.
"""

import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

# --- Constants Import ---
try:
    from shared.constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical("FATAL: Could not import constants: %s. The simulation cannot run.", e)
    sys.exit(1)

# --- Logger Initialization ---
logger = logging.getLogger("SoulCompletionController")

# --- Core Component & Stage Function Imports ---
try:
    from stage_1.fields.field_controller import FieldController
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.fields.sephiroth_field import SephirothField
    from stage_1.soul_formation.spark_harmonization import perform_spark_harmonization
    from stage_1.soul_formation.guff_strengthening import perform_guff_strengthening
    from stage_1.soul_formation.sephiroth_journey_processing import process_sephirah_interaction
    from stage_1.soul_formation.creator_entanglement import perform_creator_entanglement
    from stage_1.soul_formation.harmonic_strengthening import perform_harmonic_strengthening
    from stage_1.soul_formation.life_cord import form_life_cord
    from stage_1.soul_formation.earth_harmonisation import perform_earth_harmonization
    from stage_1.soul_formation.identity_crystallization import perform_identity_crystallization
    from stage_1.womb.womb_environment import Womb
    from stage_1.soul_formation.birth import BirthProcess
    from stage_1.soul_formation.soul_evolution_visualizer import SoulVisualizer
    from stage_1.fields.field_visualization import FieldVisualizer
    from stage_1.brain_formation.brain_structure_visualizer import BrainVisualizer
    import metrics_tracking as metrics
except ImportError as e:
    logger.critical("FATAL: Could not import a required module: %s.", e, exc_info=True)
    sys.exit(1)

# --- New Comprehensive Visualizer Imports ---
try:
    from stage_1.soul_formation.soul_evolution_visualizer import SoulEvolutionVisualizer
    from stage_1.brain_formation.brain_structure_visualizer import BrainStructureVisualizer
    COMPREHENSIVE_VISUALIZERS_AVAILABLE = True
    logger.info("Comprehensive visualization modules loaded successfully")
except ImportError as e:
    logger.warning(f"Comprehensive visualization modules not available: {e}")
    SoulEvolutionVisualizer = None
    BrainStructureVisualizer = None
    COMPREHENSIVE_VISUALIZERS_AVAILABLE = False


class SoulCompletionController:
    """Orchestrates the full soul development and formation process."""

    def __init__(self, simulation_id: str, field_controller: FieldController):
        """
        Initialize the SoulCompletionController.

        Args:
            simulation_id (str): A unique ID for this simulation run.
            field_controller (FieldController): A pre-initialized FieldController instance.
        """
        if not isinstance(field_controller, FieldController):
            raise TypeError("A valid, pre-initialized FieldController must be provided.")

        self.simulation_id = simulation_id
        self.field_controller = field_controller  # Use the provided controller
        self.results: Dict[str, Any] = {
            'simulation_id': self.simulation_id, 'start_time': datetime.now().isoformat(),
            'end_time': None, 'success': False, 'failed_stage': None, 'error': None,
            'soul_summary': {}
        }
        self.visualization_dir = os.path.join(DATA_DIR_BASE, "visuals", self.simulation_id)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Initialize existing visualizers
        field_viz_dir = os.path.join(self.visualization_dir, "fields")
        self.field_visualizer = FieldVisualizer(output_dir=field_viz_dir)
        
        soul_viz_dir = os.path.join(self.visualization_dir, "soul_evolution")
        self.soul_visualizer = SoulVisualizer(output_dir=soul_viz_dir)
        
        brain_viz_dir = os.path.join(self.visualization_dir, "brain_structure")
        self.brain_visualizer = BrainVisualizer(output_dir=brain_viz_dir)
        
        # Initialize comprehensive visualizers if available
        if COMPREHENSIVE_VISUALIZERS_AVAILABLE:
            soul_evolution_viz_dir = os.path.join(self.visualization_dir, "soul_evolution_complete")
            self.soul_evolution_visualizer = SoulEvolutionVisualizer(soul_evolution_viz_dir)
            
            brain_structure_viz_dir = os.path.join(self.visualization_dir, "brain_structure_complete")  
            self.brain_structure_visualizer = BrainStructureVisualizer(brain_structure_viz_dir)
            
            logger.info("Comprehensive visualizers initialized")
        else:
            self.soul_evolution_visualizer = None
            self.brain_structure_visualizer = None
            logger.warning("Comprehensive visualizers not available")
        
        # These will be updated later when we have soul identity
        self.final_visualization_dir = None
        
        logger.info("Soul Completion Controller initialized for Simulation ID: %s", self.simulation_id)

    def _display_stage_metrics(self, stage_name: str, metrics_dict: Dict[str, Any]):
        """Prints a formatted summary of metrics for a completed stage."""
        logger.info("Metrics for stage '%s': %s", stage_name, json.dumps(metrics_dict, default=str, indent=2))

    def _handle_failure(self, stage_name: str, soul_id: str, error: Exception):
        """Logs, records, and propagates failures."""
        logger.error("CRITICAL FAILURE in stage '%s' for soul %s.", stage_name, soul_id, exc_info=True)
        self.results.update({
            'success': False, 'failed_stage': stage_name, 'error': f"{type(error).__name__}: {error}"
        })
        metrics.record_metrics("simulation_failure", {
            'simulation_id': self.simulation_id, 'soul_id': soul_id, 'failed_stage': stage_name,
            'error_type': type(error).__name__, 'error_message': str(error)
        })
        raise RuntimeError(f"Stage '{stage_name}' failed for soul {soul_id}") from error
    
    def _prepare_soul_data_for_visualization(self, soul_spark: SoulSpark, stage: str) -> Dict[str, Any]:
        """Convert soul spark to data format needed by SoulVisualizer"""
        try:
            # Extract soul data - HARD FAIL if critical attributes missing
            frequency = getattr(soul_spark, 'frequency', None)
            if frequency is None:
                raise RuntimeError(f"CRITICAL: Soul frequency is required for visualization but missing from soul {soul_spark.spark_id}")
                
            energy = getattr(soul_spark, 'energy', None)
            if energy is None:
                raise RuntimeError(f"CRITICAL: Soul energy is required for visualization but missing from soul {soul_spark.spark_id}")
                
            stability = getattr(soul_spark, 'stability', None)
            if stability is None:
                raise RuntimeError(f"CRITICAL: Soul stability is required for visualization but missing from soul {soul_spark.spark_id}")
                
            coherence = getattr(soul_spark, 'coherence', None)
            if coherence is None:
                raise RuntimeError(f"CRITICAL: Soul coherence is required for visualization but missing from soul {soul_spark.spark_id}")
            
            soul_data = {
                'name': getattr(soul_spark, 'name', getattr(soul_spark, 'spark_id', 'Unknown Soul')),
                'birth_date': getattr(soul_spark, 'conceptual_birth_datetime', getattr(soul_spark, 'birth_date', '2024-05-11')),
                'star_sign': getattr(soul_spark, 'zodiac_sign', getattr(soul_spark, 'star_sign', 'Sagittarius')),
                'primary_color': getattr(soul_spark, 'color', '#4CC9F0'),
                'frequency': frequency,
                'energy': energy,
                'coherence': coherence,
                'stability': stability,
                'complexity': getattr(soul_spark, 'complexity', 75),
                'stage': stage.lower().replace(' ', '_'),
                'simulation_id': self.simulation_id,
                'spark_id': getattr(soul_spark, 'spark_id', 'unknown')
            }
            
            # Add stage-specific attributes if available
            if hasattr(soul_spark, 'harmonic_resonance'):
                soul_data['harmonic_resonance'] = soul_spark.harmonic_resonance
            if hasattr(soul_spark, 'quantum_signature'):
                soul_data['quantum_signature'] = str(soul_spark.quantum_signature)
            if hasattr(soul_spark, 'crystallization_progress'):
                soul_data['crystallization_progress'] = soul_spark.crystallization_progress
                
            return soul_data
            
        except Exception as e:
            logger.warning(f"Error preparing soul data for visualization: {e}")
            # Return minimal data structure
            return {
                'name': 'Unknown Soul',
                'date_of_birth': '2024-05-11',
                'star_sign': 'Sagittarius',
                'primary_color': '#4CC9F0',
                'energy_level': 75,
                'coherence': 80,
                'complexity': 70,
                'stability': 75,
                'stage': stage.lower().replace(' ', '_'),
                'simulation_id': self.simulation_id
            }
    
    def _visualize_soul_evolution_stage(self, soul_spark: SoulSpark, stage: str):
        """Create visualization for current soul evolution stage"""
        logger.info(f"Starting visualization for stage: {stage}")
        
        try:
            # Create stage-specific visualization based on current stage
            if stage == "Spark Emergence":
                result = self.soul_visualizer.create_soul_spark_visualization(soul_spark, stage)
            elif stage == "Sephiroth Journey":
                result = self.soul_visualizer.create_sephiroth_journey_visualization(soul_spark, stage)
            elif stage == "Creator Entanglement":
                result = self.soul_visualizer.create_creator_entanglement_visualization(soul_spark, stage)
            elif stage == "Identity Crystallization":
                result = self.soul_visualizer.create_identity_crystallization_visualization(soul_spark, stage)
            else:
                raise ValueError(f"Unknown stage: {stage}")
                
            if not result.get('success'):
                raise RuntimeError(f"Visualization failed for stage {stage}")
                
            logger.info(f"Soul visualization completed successfully for stage: {stage}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Soul visualization failed for stage {stage}: {e}")
            raise RuntimeError(f"CRITICAL: Soul visualization failed for stage {stage}: {e}") from e
    
    def _create_final_evolution_visualization(self, soul_spark: SoulSpark):
        """Create complete evolution visualization at the end"""
        try:
            logger.info("Creating final complete soul evolution visualization...")
            
            # Use the existing comprehensive Plotly visualization system
            result_paths = self.soul_visualizer.create_complete_soul_progression(soul_spark)
            
            if result_paths:
                logger.info("Complete soul evolution visualization created successfully")
                self.results['soul_summary']['visualization_complete'] = True
                self.results['soul_summary']['final_visualizations'] = result_paths
                
                for viz_type, path in result_paths.items():
                    logger.info(f"  - {viz_type}: {path}")
            else:
                logger.warning("Complete evolution visualization failed - no visualizations created")
                
        except Exception as e:
            logger.error(f"Error creating complete evolution visualization: {e}")
    
    def _create_comprehensive_field_dashboard(self, soul_spark: SoulSpark):
        """Create comprehensive field system dashboard showing complete field evolution."""
        try:
            logger.info("Creating comprehensive field system dashboard...")
            
            # Create the field dashboard using the field visualizer
            dashboard_result = self.field_visualizer.create_field_dashboard(
                self.field_controller, show=False, save=True
            )
            
            if dashboard_result:
                logger.info("Comprehensive field dashboard created successfully")
                self.results['field_dashboard_complete'] = True
                return True
            else:
                logger.warning("Field dashboard creation failed")
                self.results['field_dashboard_complete'] = False
                return False
                
        except Exception as e:
            logger.error(f"Error creating field dashboard: {e}")
            self.results['field_dashboard_complete'] = False
            return False

    def _create_comprehensive_visualizations(self, soul_spark: SoulSpark):
        """Create comprehensive soul evolution and brain structure visualizations using the new Plotly system."""
        try:
            soul_name = getattr(soul_spark, 'name', 'Unknown')
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            
            logger.info(f"Creating comprehensive visualizations for {soul_name} ({spark_id})")
            
            visualization_results = {
                'soul_evolution': {'success': False, 'files': []},
                'brain_structure': {'success': False, 'files': []},
                'field_system': {'success': False, 'dashboard_created': False}
            }
            
            # Create comprehensive soul evolution visualization
            if self.soul_evolution_visualizer:
                try:
                    logger.info("Creating comprehensive soul evolution visualization...")
                    soul_result = self.soul_evolution_visualizer.create_complete_soul_evolution(soul_spark)
                    if soul_result.get('success'):
                        visualization_results['soul_evolution'] = soul_result
                        logger.info(f"Soul evolution visualization completed: {soul_result['total_visualizations']} files")
                        
                        # Log each file created
                        for file_info in soul_result.get('files_created', []):
                            logger.info(f"  - {file_info['stage']}: {file_info['filename']}")
                    else:
                        raise RuntimeError("Soul evolution visualization returned success=False")
                        
                except Exception as soul_viz_error:
                    logger.error(f"CRITICAL: Soul evolution visualization failed: {soul_viz_error}", exc_info=True)
                    # Hard fail as requested - do not continue if critical visualization fails
                    raise RuntimeError(f"Soul evolution visualization failed: {soul_viz_error}") from soul_viz_error
            else:
                logger.warning("Comprehensive soul evolution visualizer not available")
            
            # Create comprehensive brain structure visualization
            if self.brain_structure_visualizer and hasattr(soul_spark, 'brain_structure'):
                try:
                    logger.info("Creating comprehensive brain structure visualization...")
                    brain_result = self.brain_structure_visualizer.create_complete_brain_visualization(soul_spark)
                    if brain_result.get('success'):
                        visualization_results['brain_structure'] = brain_result
                        logger.info(f"Brain structure visualization completed: {brain_result['total_visualizations']} files")
                        
                        # Log each file created
                        for file_info in brain_result.get('files_created', []):
                            logger.info(f"  - {file_info['visualization']}: {file_info['filename']}")
                    else:
                        raise RuntimeError("Brain structure visualization returned success=False")
                        
                except Exception as brain_viz_error:
                    logger.error(f"CRITICAL: Brain structure visualization failed: {brain_viz_error}", exc_info=True)
                    # Hard fail as requested - do not continue if critical visualization fails
                    raise RuntimeError(f"Brain structure visualization failed: {brain_viz_error}") from brain_viz_error
            else:
                if not self.brain_structure_visualizer:
                    logger.warning("Comprehensive brain structure visualizer not available")
                else:
                    logger.info("No brain structure available for comprehensive visualization")
            
            # Create comprehensive field system dashboard
            try:
                logger.info("Creating comprehensive field system dashboard...")
                field_dashboard_success = self._create_comprehensive_field_dashboard(soul_spark)
                visualization_results['field_system'] = {
                    'success': field_dashboard_success,
                    'dashboard_created': field_dashboard_success
                }
                if field_dashboard_success:
                    logger.info("Field system dashboard completed successfully")
                else:
                    logger.warning("Field system dashboard creation failed")
                    
            except Exception as field_viz_error:
                logger.error(f"CRITICAL: Field system dashboard failed: {field_viz_error}", exc_info=True)
                # Hard fail as requested - do not continue if critical visualization fails
                raise RuntimeError(f"Field system dashboard failed: {field_viz_error}") from field_viz_error
            
            # Store results in controller
            self.results['comprehensive_visualizations'] = visualization_results
            
            # Update final visualization paths for moving to completed souls
            if visualization_results['soul_evolution'].get('success') or visualization_results['brain_structure'].get('success'):
                # Ensure final visualization directory is set
                if not self.final_visualization_dir:
                    birth_date = getattr(soul_spark, 'conceptual_birth_datetime', None) or getattr(soul_spark, 'birth_date', 'Unknown')
                    self._update_visualization_paths_for_final_location(soul_name, birth_date)
            
            logger.info("Comprehensive visualizations completed successfully")
            return visualization_results
            
        except Exception as e:
            logger.error(f"CRITICAL: Comprehensive visualization creation failed: {e}", exc_info=True)
            raise RuntimeError(f"Comprehensive visualization failed: {e}") from e

    def _move_comprehensive_visualizations_to_final_location(self, soul_spark: SoulSpark):
        """Move comprehensive visualization files to the final completed souls directory."""
        try:
            soul_name = getattr(soul_spark, 'name', 'Unknown')
            spark_id = getattr(soul_spark, 'spark_id', 'unknown')
            
            if not self.final_visualization_dir:
                logger.warning("Final visualization directory not set, cannot move comprehensive visualizations")
                return
                
            # Extract base completed souls directory from final_visualization_dir
            # final_visualization_dir is like: .../completed_souls/ModelName_ID_Date/visuals
            base_completed_dir = os.path.dirname(os.path.dirname(self.final_visualization_dir))
            
            moved_successfully = True
            
            # Move comprehensive soul evolution visualizations
            if self.soul_evolution_visualizer:
                try:
                    soul_moved = self.soul_evolution_visualizer.move_to_completed_souls(
                        soul_name, spark_id, base_completed_dir
                    )
                    if not soul_moved:
                        moved_successfully = False
                        logger.error("Failed to move comprehensive soul evolution visualizations")
                except Exception as e:
                    logger.error(f"Error moving comprehensive soul evolution visualizations: {e}")
                    moved_successfully = False
            
            # Move comprehensive brain structure visualizations
            if self.brain_structure_visualizer:
                try:
                    brain_moved = self.brain_structure_visualizer.move_to_completed_souls(
                        soul_name, spark_id, base_completed_dir
                    )
                    if not brain_moved:
                        moved_successfully = False
                        logger.error("Failed to move comprehensive brain structure visualizations")
                except Exception as e:
                    logger.error(f"Error moving comprehensive brain structure visualizations: {e}")
                    moved_successfully = False
            
            if moved_successfully:
                logger.info("All comprehensive visualizations moved to final location")
            else:
                logger.warning("Some comprehensive visualizations failed to move")
                
        except Exception as e:
            logger.error(f"Failed to move comprehensive visualizations: {e}", exc_info=True)
    
    def _create_final_plotly_health_scan(self, soul_spark: SoulSpark):
        """Create comprehensive Plotly health scan visualizations for the completed soul"""
        try:
            logger.info("Creating comprehensive Plotly health scan visualizations...")
            
            # Create all the Plotly visualizations
            plotly_results = self.soul_visualizer.create_complete_soul_progression(soul_spark)
            
            if plotly_results:
                logger.info("Plotly health scan visualizations created successfully:")
                for viz_type, file_path in plotly_results.items():
                    logger.info(f"  - {viz_type}: {file_path}")
                
                # Update results with Plotly visualization paths
                self.results['soul_summary']['plotly_visualizations'] = plotly_results
                self.results['soul_summary']['health_scan_complete'] = True
                
                # Get summary of all visualizations
                viz_summary = self.soul_visualizer.get_visualization_summary()
                self.results['soul_summary']['visualization_summary'] = viz_summary
                
                logger.info(f"Total visualizations created: {viz_summary.get('html_visualizations', 0)} HTML, {viz_summary.get('png_visualizations', 0)} PNG")
                
            else:
                logger.warning("No Plotly visualizations were created")
                self.results['soul_summary']['health_scan_complete'] = False
                
        except Exception as e:
            logger.error(f"Error creating Plotly health scan visualizations: {e}")
            self.results['soul_summary']['health_scan_complete'] = False

    def _create_final_brain_health_scan(self, soul_spark: SoulSpark):
        """Create comprehensive Plotly brain health scan visualizations"""
        try:
            logger.info("Creating comprehensive brain health scan visualizations...")
            
            # Check if soul has brain structure
            brain_structure = getattr(soul_spark, 'brain_structure', None)
            if not brain_structure:
                raise RuntimeError("CRITICAL: Brain structure is required for health scan visualization but not available. Brain formation code failed to create brain structure.")
            
            # Create all the brain visualizations - will HARD FAIL if data is missing
            brain_results = self.brain_visualizer.create_complete_brain_progression(brain_structure)
            
            if brain_results:
                logger.info("Brain health scan visualizations created successfully:")
                for viz_type, file_path in brain_results.items():
                    logger.info(f"  - {viz_type}: {file_path}")
                
                # Update results with brain visualization paths
                self.results['soul_summary']['brain_visualizations'] = brain_results
                self.results['soul_summary']['brain_health_scan_complete'] = True
                
                logger.info(f"Total brain visualizations created: {len(brain_results)}")
                
            else:
                raise RuntimeError("CRITICAL: Brain visualization creation failed - no visualizations were created")
                
        except Exception as e:
            logger.error(f"CRITICAL FAILURE: Brain health scan visualization failed: {e}")
            # Re-raise the exception to fail the entire simulation
            raise RuntimeError(f"CRITICAL: Brain health scan visualization failed: {e}") from e

    def _visualize_field_state(self, soul_spark: SoulSpark, stage: str):
        """Create field visualization for current stage - shows field evolution throughout soul formation."""
        try:
            logger.info(f"Creating field visualization for stage: {stage}")
            
            # Get void field for visualization
            void_field = self.field_controller.get_field('void')
            if not void_field:
                logger.warning(f"No void field available for visualization in stage: {stage}")
                return
            
            # 1. Visualize void field energy state
            self.field_visualizer.visualize_void_field_slice(
                void_field, property_name='energy',
                show=False, save=True
            )
            
            # 2. Visualize void field coherence if stage involves coherence changes
            if stage in ['Creator Entanglement', 'Identity Crystallization', 'Harmonic Strengthening']:
                self.field_visualizer.visualize_void_field_slice(
                    void_field, property_name='coherence',
                    show=False, save=True
                )
            
            # 3. Visualize Edge of Chaos for development stages
            if stage in ['Spark Emergence', 'Guff Strengthening', 'Harmonic Strengthening']:
                self.field_visualizer.visualize_edge_of_chaos(void_field, show=False, save=True)
            
            # 4. Visualize soul-field interaction if soul has established position
            if hasattr(soul_spark, 'position') and hasattr(soul_spark, 'current_field_key'):
                current_field = self.field_controller.get_field(soul_spark.current_field_key)
                if current_field:
                    self.field_visualizer.visualize_soul_field_interaction(
                        soul_spark, current_field, show=False, save=True
                    )
            
            # 5. Stage-specific field visualizations
            if stage == 'Sephiroth Journey':
                # Visualize the Tree of Life structure during journey
                self.field_visualizer.visualize_sephiroth_tree(
                    self.field_controller, show=False, save=True
                )
                
                # Visualize current Sephiroth field if soul is in one
                if hasattr(soul_spark, 'current_field_key') and soul_spark.current_field_key != 'void':
                    sephiroth_field = self.field_controller.get_field(soul_spark.current_field_key)
                    if sephiroth_field:
                        self.field_visualizer.visualize_sephiroth_field(
                            sephiroth_field, show=False, save=True
                        )
            
            # 6. Frequency spectrum analysis for frequency-related stages
            if stage in ['Creator Entanglement', 'Harmonic Strengthening', 'Identity Crystallization']:
                current_field = self.field_controller.get_field(soul_spark.current_field_key or 'void')
                if current_field:
                    position = getattr(soul_spark, 'position', None)
                    if position:
                        position_int = [int(p) for p in position[:3]]
                        self.field_visualizer.visualize_field_frequency_spectrum(
                            current_field, position=position_int, show=False, save=True
                        )
            
            logger.info(f"Field visualization completed for stage: {stage}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Field visualization failed for stage {stage}: {e}")
            raise RuntimeError(f"Field visualization failed for stage {stage}: {e}") from e

    def _visualize_brain_state(self, soul_spark: SoulSpark, stage: str):
        """Create visualization for current brain state - HARD FAIL if brain expected but missing"""
        # Check if soul has brain structure
        brain_structure = getattr(soul_spark, 'brain_structure', None)
        if not brain_structure:
            logger.info(f"No brain structure available for visualization in stage: {stage}")
            return
            
        logger.info(f"Creating brain visualization for stage: {stage}")
        
        # Extract brain data - will raise if critical data missing
        brain_data = self.brain_visualizer._extract_brain_data(brain_structure)
        
        # Create 3D brain structure visualization - will raise if fails
        brain_fig = self.brain_visualizer.create_3d_brain_structure(brain_data)
        
        # Save brain visualization
        brain_filename = f"brain_state_{stage.lower().replace(' ', '_')}_{soul_spark.spark_id}.png"
        brain_file_path = os.path.join(self.brain_visualizer.output_dir, brain_filename)
        brain_fig.write_image(brain_file_path, width=1920, height=1080, scale=2)
        
        logger.info(f"Brain visualization saved: {brain_file_path}")

    def _update_visualization_paths_for_final_location(self, soul_name: str, birth_date: str):
        """Update visualization paths to point to the final completed souls directory"""
        try:
            # Create the final directory name based on soul identity
            if birth_date != "Unknown" and birth_date:
                if isinstance(birth_date, str) and 'T' in birth_date:
                    # Extract date part from datetime
                    date_part = birth_date.split('T')[0].replace('-', '')
                elif isinstance(birth_date, str):
                    date_part = birth_date.replace('-', '')
                else:
                    date_part = str(birth_date).replace('-', '')
                model_name = f"{soul_name}_{self.simulation_id}_{date_part}"
            else:
                model_name = f"{soul_name}_{self.simulation_id}"
            
            # Set the final visualization directory
            self.final_visualization_dir = os.path.join(
                DATA_DIR_BASE.replace('output', 'shared/output'),
                "completed_souls", 
                model_name,
                "visuals"
            )
            
            # Create the directory structure
            os.makedirs(self.final_visualization_dir, exist_ok=True)
            os.makedirs(os.path.join(self.final_visualization_dir, "soul_evolution"), exist_ok=True)
            os.makedirs(os.path.join(self.final_visualization_dir, "brain_structure"), exist_ok=True)
            os.makedirs(os.path.join(self.final_visualization_dir, "fields"), exist_ok=True)
            
            logger.info(f"Final visualization directory set: {self.final_visualization_dir}")
            
        except Exception as e:
            logger.error(f"Failed to set up final visualization directory: {e}")

    def _copy_visualizations_to_final_location(self):
        """Copy all visualization files to the final completed souls directory"""
        if not self.final_visualization_dir:
            logger.warning("Final visualization directory not set, skipping copy")
            return
            
        try:
            import shutil
            
            # Copy soul evolution visualizations
            soul_src = os.path.join(self.visualization_dir, "soul_evolution")
            soul_dst = os.path.join(self.final_visualization_dir, "soul_evolution")
            if os.path.exists(soul_src):
                for file in os.listdir(soul_src):
                    if file.endswith(('.json', '.png', '.html')):
                        shutil.copy2(os.path.join(soul_src, file), os.path.join(soul_dst, file))
                        logger.info(f"Copied soul visualization: {file}")
            
            # Copy brain structure visualizations
            brain_src = os.path.join(self.visualization_dir, "brain_structure")
            brain_dst = os.path.join(self.final_visualization_dir, "brain_structure")
            if os.path.exists(brain_src):
                for file in os.listdir(brain_src):
                    if file.endswith(('.json', '.png', '.html')):
                        shutil.copy2(os.path.join(brain_src, file), os.path.join(brain_dst, file))
                        logger.info(f"Copied brain visualization: {file}")
                        
            # Copy field visualizations
            field_src = os.path.join(self.visualization_dir, "fields")
            field_dst = os.path.join(self.final_visualization_dir, "fields")
            if os.path.exists(field_src):
                for file in os.listdir(field_src):
                    if file.endswith(('.json', '.png', '.html')):
                        shutil.copy2(os.path.join(field_src, file), os.path.join(field_dst, file))
                        logger.info(f"Copied field visualization: {file}")
                        
            logger.info(f"All visualizations copied to: {self.final_visualization_dir}")
            
        except Exception as e:
            logger.error(f"Failed to copy visualizations to final location: {e}")

    def run_full_soul_completion(self, **kwargs):
        """Executes the entire soul formation pipeline for a single soul."""
        current_stage = "Initialization"
        soul_spark: Optional[SoulSpark] = None

        try:
            # --- STAGE 1: SPARK EMERGENCE ---
            current_stage = "Spark Emergence"
            logger.info("--- STAGE: %s ---", current_stage)
            # The field controller is now self.field_controller, which was passed in.
            soul_spark = SoulSpark.create_from_field_emergence(self.field_controller)
            self.results['soul_summary']['soul_id'] = soul_spark.spark_id
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)
            self._visualize_brain_state(soul_spark, current_stage)

            # --- STAGES 2-11: SPIRITUAL JOURNEY (PRE-INCARNATION) ---

            current_stage = "Spark Harmonization"
            logger.info("--- STAGE: %s ---", current_stage)
            soul_spark, _ = perform_spark_harmonization(soul_spark)
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)

            current_stage = "Guff Strengthening"
            logger.info("--- STAGE: %s ---", current_stage)
            print(f"DEBUG: About to call place_soul_in_guff for {soul_spark.spark_id}")
            self.field_controller.place_soul_in_guff(soul_spark)
            print(f"DEBUG: place_soul_in_guff completed. Soul field: {getattr(soul_spark, 'current_field_key', 'unknown')}, position: {getattr(soul_spark, 'position', 'unknown')}")
            soul_spark, _ = perform_guff_strengthening(soul_spark, self.field_controller)
            self.field_controller.release_soul_from_guff(soul_spark)
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)

            current_stage = "Sephiroth Journey"
            logger.info("--- STAGE: %s ---", current_stage)
            journey_path = ["kether", "chokmah", "binah", "daath", "chesed", "geburah",
                            "tiphareth", "netzach", "hod", "yesod", "malkuth"]
            for sephirah_name in journey_path:
                influencer = self.field_controller.get_field(sephirah_name)
                if not isinstance(influencer, SephirothField):
                    raise TypeError(f"Invalid influencer for {sephirah_name}")
                soul_spark, _ = process_sephirah_interaction(soul_spark, influencer, self.field_controller, duration=2.0)
            setattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, True)
            setattr(soul_spark, FLAG_READY_FOR_ENTANGLEMENT, True)
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)

            current_stage = "Creator Entanglement"
            logger.info("--- STAGE: %s ---", current_stage)
            kether_field = self.field_controller.kether_field
            if not kether_field: raise RuntimeError("KetherField is required.")
            soul_spark, _ = perform_creator_entanglement(soul_spark, kether_field)
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)

            current_stage = "Harmonic Strengthening"
            logger.info("--- STAGE: %s ---", current_stage)
            soul_spark, _ = perform_harmonic_strengthening(soul_spark)
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)

            current_stage = "Life Cord Formation"
            logger.info("--- STAGE: %s ---", current_stage)
            soul_spark, _ = form_life_cord(soul_spark)
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)

            current_stage = "Earth Harmonization"
            logger.info("--- STAGE: %s ---", current_stage)
            soul_spark, _ = perform_earth_harmonization(soul_spark)
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)

            current_stage = "Identity Crystallization"
            logger.info("--- STAGE: %s ---", current_stage)
            # Filter kwargs to only pass expected parameters for identity crystallization
            identity_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ['train_cycles', 'entrainment_bpm', 'entrainment_duration', 
                                    'love_cycles', 'geometry_stages', 'crystallization_threshold']}
            soul_spark, metrics_data = perform_identity_crystallization(soul_spark, **identity_kwargs)
            self.results['soul_summary'][current_stage] = metrics_data
            self._visualize_soul_evolution_stage(soul_spark, current_stage)
            self._visualize_field_state(soul_spark, current_stage)
            self._visualize_brain_state(soul_spark, current_stage)
            
            # Create special progression visualization for Identity Crystallization - HARD FAIL on errors
            # Use the new visualizer for pre and post crystallization states
            self._visualize_soul_evolution_stage(soul_spark, "Pre-Identity Crystallization")
            # Soul has now crystallized, visualize the post state
            self._visualize_soul_evolution_stage(soul_spark, "Post-Identity Crystallization")
            logger.info(f"Created crystallization progression visualization for {current_stage}")

            # --- STAGE 12: DELEGATION TO BIRTH PROCESS ---
            current_stage = "Birth Process Orchestration"
            logger.info("--- STAGE: %s ---", current_stage)
            logger.info("Handing off control to BirthProcess orchestrator...")
            womb = Womb()
            birth_process = BirthProcess(womb_environment=womb, soul_spark=soul_spark)
            birth_metrics = birth_process.perform_complete_birth()
            self.results['soul_summary'][current_stage] = birth_metrics
            self._display_stage_metrics(current_stage, birth_metrics)
            self._visualize_soul_evolution_stage(soul_spark, "Post-Birth")
            self._visualize_field_state(soul_spark, "Post-Birth")
            self._visualize_brain_state(soul_spark, "Post-Birth")

            # --- FINAL REPORTING ---
            current_stage = "Final Reporting"
            logger.info("--- STAGE: %s ---", current_stage)
            
            # Create final comprehensive soul evolution visualization
            self._create_final_evolution_visualization(soul_spark)
            
            # Create comprehensive Plotly health scan visualizations
            self._create_final_plotly_health_scan(soul_spark)
            
            # Create comprehensive brain health scan visualizations 
            self._create_final_brain_health_scan(soul_spark)
            
            # --- COMPREHENSIVE VISUALIZATIONS (AFTER ALL STAGES) ---
            logger.info("--- CREATING COMPREHENSIVE VISUALIZATIONS ---")
            try:
                self._create_comprehensive_visualizations(soul_spark)
            except Exception as viz_error:
                logger.error(f"CRITICAL: Comprehensive visualizations failed: {viz_error}")
                self._handle_failure("Comprehensive Visualizations", soul_spark.spark_id, viz_error)
                raise
            
            # Extract final soul identity information
            soul_name = getattr(soul_spark, 'name', 'Unknown')
            birth_date = getattr(soul_spark, 'conceptual_birth_datetime', None) or getattr(soul_spark, 'birth_date', 'Unknown')
            star_sign = getattr(soul_spark, 'zodiac_sign', None) or getattr(soul_spark, 'star_sign', 'Unknown')
            
            # Debug: Check what attributes the soul actually has
            logger.info(f"Soul attributes check:")
            logger.info(f"  name: {soul_name}")
            logger.info(f"  birth_date: {birth_date}")
            logger.info(f"  star_sign: {star_sign}")
            logger.info(f"  available attributes: {[attr for attr in dir(soul_spark) if not attr.startswith('_')]}")
            
            # --- MOVE VISUALIZATIONS TO FINAL LOCATION ---
            logger.info("--- MOVING VISUALIZATIONS TO FINAL LOCATION ---")
            try:
                # Set up final visualization directory and copy existing files
                if soul_name != 'Unknown':
                    self._update_visualization_paths_for_final_location(soul_name, birth_date)
                    self._copy_visualizations_to_final_location()
                    
                    # Move comprehensive visualizations
                    self._move_comprehensive_visualizations_to_final_location(soul_spark)
                
            except Exception as move_error:
                logger.error(f"Failed to move visualizations to final location: {move_error}")
                # Don't hard fail on move errors, just warn
            
            # Store soul identity in results for root controller
            self.results['soul_summary']['name'] = soul_name
            self.results['soul_summary']['birth_date'] = birth_date
            self.results['soul_summary']['star_sign'] = star_sign
            
            logger.info(f"Soul identity finalized: {soul_name} (Born: {birth_date}, Sign: {star_sign})")
            
            # Legacy report for compatibility - create a simple summary instead
            final_report_path = os.path.join(self.visualization_dir, "soul_completion_report.json")
            try:
                soul_summary = {
                    'soul_id': getattr(soul_spark, 'spark_id', 'unknown'),
                    'name': soul_name,
                    'birth_date': birth_date,
                    'star_sign': star_sign,
                    'completion_time': datetime.now().isoformat(),
                    'stages_completed': ['Spark Emergence', 'Sephiroth Journey', 'Creator Entanglement', 'Identity Crystallization', 'Birth'],
                    'final_metrics': {
                        'energy': getattr(soul_spark, 'energy', 0),
                        'coherence': getattr(soul_spark, 'coherence', 0),
                        'stability': getattr(soul_spark, 'stability', 0)
                    },
                    'visualization_complete': self.results['soul_summary'].get('visualization_complete', False),
                    'comprehensive_visualizations': self.results.get('comprehensive_visualizations', {})
                }
                
                with open(final_report_path, 'w') as f:
                    json.dump(soul_summary, f, indent=2)
                    
                logger.info(f"Soul completion report saved: {final_report_path}")
                
            except Exception as report_error:
                logger.warning(f"Could not create legacy report: {report_error}")
                final_report_path = "report_creation_failed"
                
            self.results['soul_summary']['final_report_path'] = final_report_path
            self.results['success'] = True
            
            logger.info("Soul completion simulation finished successfully with comprehensive visualization suite")

        except Exception as e:
            self._handle_failure(current_stage, getattr(soul_spark, 'spark_id', 'unassigned'), e)
        finally:
            self.results['end_time'] = datetime.now().isoformat()
            report_path = os.path.join(DATA_DIR_BASE, "reports", f"simulation_report_{self.simulation_id}.json")
            try:
                import numpy as np
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (np.integer, np.floating)):
                            return int(obj) if isinstance(obj, np.integer) else float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        return super(NumpyEncoder, self).default(obj)
                        
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, cls=NumpyEncoder, default=str)
                logger.info("Full simulation report saved to: %s", report_path)
            except Exception as json_err:
                logger.error("Failed to save final simulation report: %s", json_err)
            
            if hasattr(metrics, 'persist_metrics'):
                metrics.persist_metrics()

# --- END OF FILE soul_completion_controller.py ---
