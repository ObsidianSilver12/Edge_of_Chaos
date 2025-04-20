"""
Root Controller

This module provides the main execution control for the entire Soul Development Framework.
It orchestrates the complete process flow from Void field creation through birth.

Author: Soul Development Framework Team
"""

import os
import sys
import time
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('root_controller')

# Import controllers
try:
    from stage_1.void.void_field_controller import VoidFieldController
    from stage_1.void.guff_controller import GuffController
    from stage_1.sephiroth.sephiroth_controller import SephirothController
    from stage_1.soul_formation.soul_formation_controller import SoulFormationController
    import metrics_tracking as metrics
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error(traceback.format_exc())
    raise

class RootController:
    """
    Master controller for the entire Soul Development Framework.
    
    This class orchestrates the complete process flow from Void field
    creation through birth, coordinating all sub-controllers.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the root controller.
        
        Args:
            output_dir (str): Directory for output files
        """
        # Initialize state
        self.process_complete = False
        self.current_stage = None
        
        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize sub-controllers
        self.void_controller = None
        self.guff_controller = None
        self.sephiroth_controller = None
        self.soul_formation_controller = None
        
        # Initialize key objects
        self.soul_spark = None
        self.life_cord = None
        self.soul_identity = None
        
        # Initialize metrics
        self.metrics_tracker = metrics.MetricsTracker("root_controller")
        self.start_time = time.time()
        
        # Process configuration
        self.config = {
            "void_stage": {
                "enabled": True,
                "params": {}
            },
            "guff_initial_stage": {
                "enabled": True,
                "params": {}
            },
            "sephiroth_journey": {
                "enabled": True,
                "params": {}
            },
            "soul_formation": {
                "enabled": True,
                "params": {}
            },
            "guff_return_stage": {
                "enabled": True,
                "params": {}
            },
            "birth_process": {
                "enabled": True,
                "params": {}
            }
        }
        
        logger.info("Root Controller initialized")
    
    def set_configuration(self, config: Dict[str, Any]) -> None:
        """
        Set configuration for the soul development process.
        
        Args:
            config (dict): Configuration settings
        """
        if not isinstance(config, dict):
            logger.error("Configuration must be a dictionary")
            return
        
        # Update configuration
        for stage, settings in config.items():
            if stage in self.config:
                self.config[stage].update(settings)
        
        logger.info("Configuration updated")
    
    def run_complete_process(self, soul_name: Optional[str] = None) -> bool:
        """
        Run the complete process from Void field to birth.
        
        Args:
            soul_name (str, optional): Optional name for the soul
            
        Returns:
            bool: Success status
        """
        logger.info("Beginning complete soul development process")
        self.start_time = time.time()
        
        try:
            # 1. Void Field Creation and Spark Formation
            if self.config["void_stage"]["enabled"]:
                if not self.run_void_stage(self.config["void_stage"]["params"]):
                    logger.error("Void stage failed")
                    return False
            else:
                logger.info("Void stage skipped (disabled in config)")
            
            # 2. Guff Transfer and Initial Strengthening
            if self.config["guff_initial_stage"]["enabled"]:
                if not self.run_guff_initial_stage(self.config["guff_initial_stage"]["params"]):
                    logger.error("Initial Guff stage failed")
                    return False
            else:
                logger.info("Initial Guff stage skipped (disabled in config)")
            
            # 3. Sephiroth Journey
            if self.config["sephiroth_journey"]["enabled"]:
                if not self.run_sephiroth_journey(self.config["sephiroth_journey"]["params"]):
                    logger.error("Sephiroth journey failed")
                    return False
            else:
                logger.info("Sephiroth journey skipped (disabled in config)")
            
            # 4. Soul Formation Process
            if self.config["soul_formation"]["enabled"]:
                params = self.config["soul_formation"]["params"].copy()
                params["specified_name"] = soul_name
                if not self.run_soul_formation_stage(params):
                    logger.error("Soul formation stage failed")
                    return False
            else:
                logger.info("Soul formation skipped (disabled in config)")
            
            # 5. Guff Return for Final Strengthening
            if self.config["guff_return_stage"]["enabled"]:
                if not self.run_guff_return_stage(self.config["guff_return_stage"]["params"]):
                    logger.error("Guff return stage failed")
                    return False
            else:
                logger.info("Guff return stage skipped (disabled in config)")
            
            # Record completion
            self.process_complete = True
            process_time = time.time() - self.start_time
            
            # Record final metrics
            self.metrics_tracker.update_metrics("process_complete", True)
            self.metrics_tracker.update_metrics("total_process_time", process_time)
            self.metrics_tracker.update_metrics("soul_name", soul_name or getattr(self.soul_identity, "name", "Unnamed"))
            self.metrics_tracker.record()
            
            logger.info(f"Complete soul development process finished in {process_time:.2f} seconds")
            
            # Save all metrics
            metrics.save_metrics_to_file()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in soul development process: {e}")
            logger.error(traceback.format_exc())
            
            # Record failure metrics
            self.metrics_tracker.update_metrics("process_complete", False)
            self.metrics_tracker.update_metrics("error", str(e))
            self.metrics_tracker.update_metrics("failed_stage", self.current_stage)
            self.metrics_tracker.update_metrics("process_time", time.time() - self.start_time)
            self.metrics_tracker.record()
            
            # Save metrics even on failure
            metrics.save_metrics_to_file()
            
            return False
    
    def run_void_stage(self, params: Dict[str, Any] = None) -> bool:
        """
        Run the Void field creation and spark formation stage.
        
        Args:
            params (dict): Optional parameters for Void stage
            
        Returns:
            bool: Success status
        """
        logger.info("Starting Void stage - field creation and spark formation")
        self.current_stage = "void"
        params = params or {}
        
        try:
            # Initialize Void Field Controller
            self.void_controller = VoidFieldController()
            
            # Create Void field with sacred geometry patterns
            void_field_created = self.void_controller.create_void_field(
                dimensions=params.get("dimensions", (64, 64, 64)),
                embed_patterns=params.get("embed_patterns", True),
                edge_of_chaos_ratio=params.get("edge_of_chaos_ratio", 0.618)
            )
            
            if not void_field_created:
                logger.error("Failed to create Void field")
                return False
            
            # Create gravity wells at pattern intersections
            wells_created = self.void_controller.create_gravity_wells(
                well_count=params.get("well_count", 7),
                well_strength=params.get("well_strength", 0.8)
            )
            
            if not wells_created:
                logger.error("Failed to create gravity wells")
                return False
            
            # Initiate quantum fluctuations
            fluctuations_initiated = self.void_controller.initiate_quantum_fluctuations(
                intensity=params.get("fluctuation_intensity", 0.7),
                duration=params.get("fluctuation_duration", 1.0)
            )
            
            if not fluctuations_initiated:
                logger.error("Failed to initiate quantum fluctuations")
                return False
            
            # Form soul spark at intersection point
            self.soul_spark = self.void_controller.form_soul_spark(
                creator_resonance=params.get("creator_resonance", 0.75)
            )
            
            if self.soul_spark is None:
                logger.error("Failed to form soul spark")
                return False
            
            # Record metrics
            self.metrics_tracker.update_metrics("void_stage_complete", True)
            self.metrics_tracker.update_metrics("spark_formed", True)
            self.metrics_tracker.update_metrics("spark_id", getattr(self.soul_spark, "spark_id", None))
            self.metrics_tracker.update_metrics("spark_stability", getattr(self.soul_spark, "stability", 0.0))
            self.metrics_tracker.record()
            
            logger.info(f"Void stage complete - soul spark formed with ID {getattr(self.soul_spark, 'spark_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error in Void stage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_guff_initial_stage(self, params: Dict[str, Any] = None) -> bool:
        """
        Run the initial Guff transfer and strengthening stage.
        
        Args:
            params (dict): Optional parameters for Guff stage
            
        Returns:
            bool: Success status
        """
        logger.info("Starting initial Guff stage - transfer and strengthening")
        self.current_stage = "guff_initial"
        params = params or {}
        
        try:
            # Check that we have a soul spark
            if self.soul_spark is None:
                logger.error("Cannot proceed to Guff stage: no soul spark")
                return False
            
            # Initialize Guff Controller
            self.guff_controller = GuffController()
            
            # Create Guff field
            guff_field_created = self.guff_controller.create_guff_field(
                dimensions=params.get("dimensions", (64, 64, 64)),
                creator_presence=params.get("creator_presence", 0.85)
            )
            
            if not guff_field_created:
                logger.error("Failed to create Guff field")
                return False
            
            # Transfer spark to Guff field
            transfer_successful = self.guff_controller.transfer_spark_to_guff(
                self.soul_spark,
                transfer_method=params.get("transfer_method", "quantum_tunnel")
            )
            
            if not transfer_successful:
                logger.error("Failed to transfer spark to Guff field")
                return False
            
            # Form initial soul structure around spark
            soul_layer_formed = self.guff_controller.form_soul_layer(
                self.soul_spark,
                layer_type=params.get("layer_type", "divine")
            )
            
            if not soul_layer_formed:
                logger.error("Failed to form initial soul layer")
                return False
            
            # Strengthen soul through creator resonance
            soul_strengthened = self.guff_controller.strengthen_through_creator(
                self.soul_spark,
                intensity=params.get("strengthen_intensity", 0.8),
                duration=params.get("strengthen_duration", 1.0)
            )
            
            if not soul_strengthened:
                logger.error("Failed to strengthen soul through creator resonance")
                return False
            
            # Record metrics
            self.metrics_tracker.update_metrics("guff_initial_stage_complete", True)
            self.metrics_tracker.update_metrics("spark_transferred", True)
            self.metrics_tracker.update_metrics("soul_layer_formed", soul_layer_formed)
            self.metrics_tracker.update_metrics("initial_strengthening", soul_strengthened)
            self.metrics_tracker.update_metrics("strengthened_stability", getattr(self.soul_spark, "stability", 0.0))
            self.metrics_tracker.record()
            
            logger.info("Initial Guff stage complete - soul strengthened")
            return True
            
        except Exception as e:
            logger.error(f"Error in initial Guff stage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_sephiroth_journey(self, params: Dict[str, Any] = None) -> bool:
        """
        Run the Sephiroth journey stage.
        
        Args:
            params (dict): Optional parameters for Sephiroth journey
            
        Returns:
            bool: Success status
        """
        logger.info("Starting Sephiroth journey stage")
        self.current_stage = "sephiroth_journey"
        params = params or {}
        
        try:
            # Check that we have a soul spark
            if self.soul_spark is None:
                logger.error("Cannot proceed to Sephiroth journey: no soul spark")
                return False
            
            # Initialize Sephiroth Controller
            self.sephiroth_controller = SephirothController(self.soul_spark)
            
            # Set up the journey parameters
            journey_type = params.get("journey_type", "traditional")
            start_sephirah = params.get("start_sephirah", "malkuth")
            journey_duration = params.get("journey_duration", 1.0)
            
            # Begin the journey
            journey_started = self.sephiroth_controller.begin_journey(
                journey_type=journey_type,
                start_sephirah=start_sephirah
            )
            
            if not journey_started:
                logger.error("Failed to begin Sephiroth journey")
                return False
            
            # Traverse each Sephiroth in the journey
            journey_complete = self.sephiroth_controller.execute_journey(
                duration=journey_duration,
                acquire_aspects=params.get("acquire_aspects", True)
            )
            
            if not journey_complete:
                logger.error("Failed to complete Sephiroth journey")
                return False
            
            # Complete the journey and retrieve the enhanced soul
            self.soul_spark = self.sephiroth_controller.complete_journey()
            
            # Get journey metrics
            journey_metrics = self.sephiroth_controller.get_journey_metrics()
            
            # Record metrics
            self.metrics_tracker.update_metrics("sephiroth_journey_complete", True)
            self.metrics_tracker.update_metrics("journey_type", journey_type)
            self.metrics_tracker.update_metrics("aspects_acquired", journey_metrics.get("aspects_acquired", 0))
            self.metrics_tracker.update_metrics("dimensions_visited", journey_metrics.get("dimensions_visited", 0))
            self.metrics_tracker.record()
            
            logger.info(f"Sephiroth journey complete - visited {journey_metrics.get('dimensions_visited', 0)} dimensions")
            return True
            
        except Exception as e:
            logger.error(f"Error in Sephiroth journey stage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_soul_formation_stage(self, params: Dict[str, Any] = None) -> bool:
        """
        Run the complete soul formation process.
        
        Args:
            params (dict): Optional parameters for soul formation
            
        Returns:
            bool: Success status
        """
        logger.info("Starting Soul Formation stage")
        self.current_stage = "soul_formation"
        params = params or {}
        
        try:
            # Check that we have a soul spark
            if self.soul_spark is None:
                logger.error("Cannot proceed to Soul Formation: no soul spark")
                return False
            
            # Initialize Soul Formation Controller
            self.soul_formation_controller = SoulFormationController(self.soul_spark)
            
            # Run creator entanglement if not already done
            if params.get("run_creator_entanglement", True):
                entanglement_success = self.soul_formation_controller.run_creator_entanglement()
                if not entanglement_success:
                    logger.warning("Creator entanglement process did not complete successfully")
            
            # Run harmonic strengthening if needed
            if params.get("run_harmonic_strengthening", True):
                strengthen_success = self.soul_formation_controller.run_harmonic_strengthening()
                if not strengthen_success:
                    logger.warning("Harmonic strengthening process did not complete successfully")
            
            # Initialize earth harmonization
            if params.get("run_earth_harmonization", True):
                earth_success = self.soul_formation_controller.run_earth_harmonization()
                if not earth_success:
                    logger.warning("Earth harmonization process did not complete successfully")
            
            # Form life cord
            if params.get("run_life_cord_formation", True):
                life_cord_success = self.soul_formation_controller.run_life_cord_formation()
                if not life_cord_success:
                    logger.error("Life cord formation failed")
                    return False
                
                # Store life cord reference
                self.life_cord = self.soul_formation_controller.life_cord
            
            # Run identity crystallization
            if params.get("run_identity_crystallization", True):
                identity_success = self.soul_formation_controller.run_identity_crystallization(
                    specified_name=params.get("specified_name")
                )
                if not identity_success:
                    logger.warning("Identity crystallization process did not complete successfully")
                
                # Store soul identity reference
                self.soul_identity = self.soul_formation_controller.soul_identity
            
            # Run birth process
            if params.get("run_birth_process", True):
                birth_success = self.soul_formation_controller.run_birth_process()
                if not birth_success:
                    logger.warning("Birth process did not complete successfully")
            
            # Get formation metrics
            formation_metrics = self.soul_formation_controller.get_formation_metrics()
            
            # Record metrics
            self.metrics_tracker.update_metrics("soul_formation_stage_complete", True)
            self.metrics_tracker.update_metrics("soul_name", getattr(self.soul_identity, "name", "Unnamed"))
            self.metrics_tracker.update_metrics("crystallization_level", 
                formation_metrics.get("identity_crystallization", {}).get("crystallization_level", 0.0))
            self.metrics_tracker.update_metrics("life_cord_strength", 
                formation_metrics.get("life_cord_formation", {}).get("cord_strength", 0.0))
            self.metrics_tracker.update_metrics("formation_time", formation_metrics.get("total_formation_time", 0.0))
            self.metrics_tracker.record()
            
            logger.info(f"Soul Formation stage complete - soul identity: {getattr(self.soul_identity, 'name', 'Unnamed')}")
            return True
            
        except Exception as e:
            logger.error(f"Error in Soul Formation stage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_guff_return_stage(self, params: Dict[str, Any] = None) -> bool:
        """
        Run the Guff return stage for final strengthening.
        
        Args:
            params (dict): Optional parameters for Guff return
            
        Returns:
            bool: Success status
        """
        logger.info("Starting Guff return stage - final strengthening")
        self.current_stage = "guff_return"
        params = params or {}
        
        try:
            # Check that we have a formed soul
            if self.soul_spark is None or self.soul_identity is None:
                logger.error("Cannot proceed to Guff return: incomplete soul")
                return False
            
            # Ensure Guff Controller is initialized
            if self.guff_controller is None:
                logger.warning("Guff Controller not initialized, creating new instance")
                self.guff_controller = GuffController()
                
                # Create Guff field if needed
                guff_field_created = self.guff_controller.create_guff_field(
                    dimensions=params.get("dimensions", (64, 64, 64)),
                    creator_presence=params.get("creator_presence", 0.9)
                )
                
                if not guff_field_created:
                    logger.error("Failed to create Guff field for return")
                    return False
            
            # Transfer soul back to Guff
            transfer_back = self.guff_controller.transfer_soul_to_guff(
                self.soul_spark,
                self.soul_identity,
                transfer_method=params.get("transfer_method", "identity_gateway")
            )
            
            if not transfer_back:
                logger.error("Failed to transfer soul back to Guff")
                return False
            
            # Final strengthening with higher intensity
            final_strengthening = self.guff_controller.final_strengthening(
                self.soul_spark,
                self.soul_identity,
                intensity=params.get("strengthen_intensity", 0.9),
                duration=params.get("strengthen_duration", 1.5)
            )
            
            if not final_strengthening:
                logger.error("Failed to perform final strengthening")
                return False
            
            # Transfer soul back to Earth field
            return_to_earth = self.guff_controller.return_to_earth(
                self.soul_spark,
                self.soul_identity,
                self.life_cord
            )
            
            if not return_to_earth:
                logger.error("Failed to return soul to Earth field")
                return False
            
            # Record metrics
            self.metrics_tracker.update_metrics("guff_return_stage_complete", True)
            self.metrics_tracker.update_metrics("final_strengthening", final_strengthening)
            self.metrics_tracker.update_metrics("final_stability", getattr(self.soul_spark, "stability", 0.0))
            self.metrics_tracker.update_metrics("final_coherence", getattr(self.soul_spark, "coherence", 0.0))
            self.metrics_tracker.record()
            
            logger.info("Guff return stage complete - soul returned to Earth with final strengthening")
            return True
            
        except Exception as e:
            logger.error(f"Error in Guff return stage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_formed_soul(self) -> Tuple[Any, Any, Any]:
        """
        Get the completely formed soul components.
        
        Returns:
            tuple: (soul_spark, life_cord, soul_identity)
        """
        if not self.process_complete:
            logger.warning("Attempting to get formed soul before process completion")
        
        return (self.soul_spark, self.life_cord, self.soul_identity)
    
    def get_process_metrics(self) -> Dict[str, Any]:
        """
        Get complete metrics for the soul development process.
        
        Returns:
            dict: Process metrics
        """
        # Get metrics from tracker
        all_metrics = self.metrics_tracker.get_all()
        
        # Add final soul metrics if available
        if self.soul_spark is not None:
            all_metrics["spark_metrics"] = getattr(self.soul_spark, "get_metrics", lambda: {})()
        
        if self.soul_identity is not None:
            all_metrics["identity_metrics"] = getattr(self.soul_identity, "get_metrics", lambda: {})()
        
        return all_metrics
    
    def save_visualization(self, output_prefix: str = "soul_development") -> bool:
        """
        Save visualizations of the formed soul.
        
        Args:
            output_prefix (str): Prefix for output files
            
        Returns:
            bool: Success status
        """
        if not self.process_complete:
            logger.warning("Attempting to visualize soul before process completion")
        
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Visualize soul spark if available
            if self.soul_spark is not None and hasattr(self.soul_spark, "visualize_spark"):
                spark_path = os.path.join(viz_dir, f"{output_prefix}_spark_{timestamp}.png")
                self.soul_spark.visualize_spark(show=False, save_path=spark_path)
                logger.info(f"Saved soul spark visualization to: {spark_path}")
            
            # Visualize soul identity if available
            if self.soul_formation_controller is not None and hasattr(self.soul_formation_controller, "visualize_soul_identity"):
                identity_path = os.path.join(viz_dir, f"{output_prefix}_identity_{timestamp}.png")
                self.soul_formation_controller.visualize_soul_identity(
                    self.soul_identity, 
                    show=False, 
                    save_path=identity_path
                )
                logger.info(f"Saved soul identity visualization to: {identity_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")
            logger.error(traceback.format_exc())
            return False


# Example usage
if __name__ == "__main__":
    # Create controller
    controller = RootController()
    
    # Set configuration if needed
    config = {
        "void_stage": {
            "params": {
                "edge_of_chaos_ratio": 0.618,
                "well_count": 9
            }
        },
        "sephiroth_journey": {
            "params": {
                "journey_type": "traditional",
                "start_sephirah": "malkuth"
            }
        },
        "soul_formation": {
            "params": {
                "run_harmonic_strengthening": True
            }
        }
    }
    controller.set_configuration(config)
    
    # Run complete process
    soul_name = "TestSoul"
    success = controller.run_complete_process(soul_name)
    
    if success:
        print("Soul development process completed successfully!")
        
        # Get formed soul
        soul_spark, life_cord, soul_identity = controller.get_formed_soul()
        
        print(f"Soul Name: {soul_identity.name}")
        print(f"Soul Stability: {soul_spark.stability:.4f}")
        
        # Save visualizations
        controller.save_visualization(f"{soul_identity.name}_development")
        
        # Get process metrics
        metrics = controller.get_process_metrics()
        print(f"Process Time: {metrics.get('total_process_time', 0.0):.2f} seconds")
    else:
        print("Soul development process failed.")


