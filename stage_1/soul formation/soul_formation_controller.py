"""
Soul Formation Controller

This module orchestrates the complete soul formation process from initial spark
through Sephiroth journey, strengthening, identity crystallization, and birth.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='soul_formation_controller.log'
)
logger = logging.getLogger('soul_formation_controller')

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
try:
    from stage_1.soul_formation.creator_entanglement import CreatorEntanglement
    from stage_1.soul_formation.soul_formation import SoulFormation
    from stage_1.soul_formation.harmonic_strengthening import HarmonicStrengthening
    from stage_1.soul_formation.edge_of_chaos import EdgeOfChaos
    from stage_1.soul_formation.life_cord import LifeCord
    from stage_1.soul_formation.earth_harmonisation import EarthHarmonisation
    from stage_1.soul_formation.identity_crystallization import process_identity_crystallization
    from stage_1.soul_formation.birth import Birth
    
    from stage_1.sephiroth.sephiroth_controller import SephirothController
    from stage_1.void.soul_spark import SoulSpark
    import metrics_tracking as metrics
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class SoulFormationController:
    """
    Controller for the complete soul formation process.
    
    This class orchestrates all stages of soul formation from initial spark
    through creator entanglement, Sephiroth journey, harmonization, identity 
    crystallization, and birth.
    """
    
    def __init__(self, soul_spark=None):
        """
        Initialize the soul formation controller.
        
        Args:
            soul_spark (SoulSpark, optional): Existing soul spark, or None to create new
        """
        # Create or use existing soul spark
        if soul_spark is None:
            self.soul_spark = SoulSpark()
        else:
            self.soul_spark = soul_spark
            
        # Initialize controllers for each stage
        self.sephiroth_controller = SephirothController()
        
        # Process stages
        self.creator_entanglement = None
        self.soul_formation = None
        self.harmonic_strengthening = None
        self.edge_of_chaos = None
        self.life_cord = None
        self.earth_harmonisation = None
        self.birth_process = None
        
        # Formation state
        self.current_stage = None
        self.formation_complete = False
        self.formation_metrics = {}
        
        # Store the identity results
        self.soul_identity = None
        self.identity_metrics = None
        
        logger.info(f"Soul Formation Controller initialized for soul spark {self.soul_spark.id}")
        
    def run_full_formation(self, journey_type="traditional", skip_stages=None):
        """
        Run the complete soul formation process.
        
        Args:
            journey_type (str): Type of Sephiroth journey (traditional, personalized, direct)
            skip_stages (list): Optional list of stage names to skip
            
        Returns:
            bool: Success status
        """
        logger.info(f"Beginning full soul formation process for soul spark {self.soul_spark.id}")
        skip_stages = skip_stages or []
        
        start_time = time.time()
        
        # 1. Creator Entanglement
        if "creator_entanglement" not in skip_stages:
            success = self.run_creator_entanglement()
            if not success:
                logger.error("Creator Entanglement failed")
                return False
        
        # 2. Sephiroth Journey
        if "sephiroth_journey" not in skip_stages:
            success = self.run_sephiroth_journey(journey_type=journey_type)
            if not success:
                logger.error("Sephiroth Journey failed")
                return False
        
        # 3. Harmonic Strengthening
        if "harmonic_strengthening" not in skip_stages:
            success = self.run_harmonic_strengthening()
            if not success:
                logger.error("Harmonic Strengthening failed")
                return False
        
        # 4. Edge of Chaos Balancing
        if "edge_of_chaos" not in skip_stages:
            success = self.run_edge_of_chaos()
            if not success:
                logger.error("Edge of Chaos balancing failed")
                return False
        
        # 5. Life Cord Formation
        if "life_cord" not in skip_stages:
            success = self.run_life_cord_formation()
            if not success:
                logger.error("Life Cord Formation failed")
                return False
        
        # 6. Earth Harmonization
        if "earth_harmonisation" not in skip_stages:
            success = self.run_earth_harmonisation()
            if not success:
                logger.error("Earth Harmonisation failed")
                return False
        
        # 7. Identity Crystallization
        if "identity_crystallization" not in skip_stages:
            success = self.run_identity_crystallization()
            if not success:
                logger.error("Identity Crystallization failed")
                return False
        
        # 8. Birth Process
        if "birth" not in skip_stages:
            success = self.run_birth_process()
            if not success:
                logger.error("Birth Process failed")
                return False
        
        # Record formation completion
        self.formation_complete = True
        self.formation_metrics["total_formation_time"] = time.time() - start_time
        
        # Record final metrics
        self._record_final_metrics()
        
        logger.info(f"Soul formation completed successfully in {time.time() - start_time:.2f} seconds")
        return True
    
    def run_creator_entanglement(self):
        """
        Run the creator entanglement process.
        
        Returns:
            bool: Success status
        """
        logger.info("Starting Creator Entanglement process")
        self.current_stage = "creator_entanglement"
        
        try:
            # Initialize Creator Entanglement
            self.creator_entanglement = CreatorEntanglement(self.soul_spark)
            
            # Run the full entanglement process
            success = self.creator_entanglement.run_full_entanglement_process()
            
            # Record metrics
            self.formation_metrics["creator_entanglement"] = {
                "success": success,
                "connection_strength": getattr(self.soul_spark, "creator_connection", 0.0),
                "resonance_points": getattr(self.creator_entanglement, "resonance_points", 0)
            }
            
            logger.info(f"Creator Entanglement completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in Creator Entanglement: {e}")
            return False
    
    def run_sephiroth_journey(self, journey_type="traditional"):
        """
        Run the soul's journey through the Sephiroth.
        
        Args:
            journey_type (str): Type of journey to undertake (traditional, personalized, direct)
            
        Returns:
            bool: Success status
        """
        logger.info(f"Starting Sephiroth Journey process with {journey_type} path")
        self.current_stage = "sephiroth_journey"
        
        try:
            # Initialize Soul Formation
            self.soul_formation = SoulFormation(
                self.soul_spark, 
                sephiroth_controller=self.sephiroth_controller
            )
            
            # Begin journey
            journey_started = self.soul_formation.begin_journey(journey_type)
            if not journey_started:
                logger.error("Failed to begin Sephiroth journey")
                return False
            
            # Process each Sephirah in the journey
            while True:
                # Process current Sephirah
                processed = self.soul_formation.process_current_sephirah()
                if not processed:
                    logger.error(f"Failed to process Sephirah: {self.soul_formation.current_sephirah}")
                    return False
                
                # Advance to next Sephirah (returns False when journey is complete)
                if not self.soul_formation.advance_to_next_sephirah():
                    break
            
            # Complete the journey
            journey_completed = self.soul_formation.complete_journey()
            
            # Record metrics
            self.formation_metrics["sephiroth_journey"] = {
                "success": journey_completed,
                "journey_type": journey_type,
                "stability": getattr(self.soul_spark, "stability", 0.0),
                "coherence": getattr(self.soul_spark, "coherence", 0.0),
                "entanglements": getattr(self.soul_spark, "sephiroth_entanglements", {})
            }
            
            logger.info(f"Sephiroth Journey completed: {journey_completed}")
            return journey_completed
            
        except Exception as e:
            logger.error(f"Error in Sephiroth Journey: {e}")
            return False
    
    def run_harmonic_strengthening(self, intensity=0.75, duration=1.0):
        """
        Run the harmonic strengthening process.
        
        Args:
            intensity (float): Strengthening intensity (0.0-1.0)
            duration (float): Relative duration multiplier
            
        Returns:
            bool: Success status
        """
        logger.info(f"Starting Harmonic Strengthening process with intensity {intensity}")
        self.current_stage = "harmonic_strengthening"
        
        try:
            # Initialize Harmonic Strengthening
            self.harmonic_strengthening = HarmonicStrengthening(self.soul_spark)
            
            # Run strengthening process
            success = self.harmonic_strengthening.strengthen(intensity, duration)
            
            # Check if additional strengthening is needed
            stability = getattr(self.soul_spark, "stability", 0.0)
            coherence = getattr(self.soul_spark, "coherence", 0.0)
            
            if success and (stability < 0.75 or coherence < 0.75):
                logger.info("Metrics below threshold, applying additional strengthening")
                success = self.harmonic_strengthening.strengthen(0.85, 1.5)
            
            # Record metrics
            self.formation_metrics["harmonic_strengthening"] = {
                "success": success,
                "stability": getattr(self.soul_spark, "stability", 0.0),
                "coherence": getattr(self.soul_spark, "coherence", 0.0),
                "frequency": getattr(self.soul_spark, "frequency", 0.0),
                "phi_resonance": getattr(self.soul_spark, "phi_resonance", 0.0)
            }
            
            logger.info(f"Harmonic Strengthening completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in Harmonic Strengthening: {e}")
            return False
    
    def run_edge_of_chaos(self):
        """
        Run the edge of chaos balancing process.
        
        Returns:
            bool: Success status
        """
        logger.info("Starting Edge of Chaos balancing process")
        self.current_stage = "edge_of_chaos"
        
        try:
            # Initialize Edge of Chaos
            self.edge_of_chaos = EdgeOfChaos(self.soul_spark)
            
            # Run balancing process
            success = self.edge_of_chaos.balance()
            
            # Record metrics
            self.formation_metrics["edge_of_chaos"] = {
                "success": success,
                "chaos_order_ratio": getattr(self.edge_of_chaos, "chaos_order_ratio", 0.0),
                "complexity": getattr(self.edge_of_chaos, "complexity", 0.0)
            }
            
            logger.info(f"Edge of Chaos balancing completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in Edge of Chaos balancing: {e}")
            return False
    
    def run_life_cord_formation(self):
        """
        Run the life cord formation process.
        
        Returns:
            bool: Success status
        """
        logger.info("Starting Life Cord formation process")
        self.current_stage = "life_cord"
        
        try:
            # Initialize Life Cord
            self.life_cord = LifeCord(self.soul_spark)
            
            # Form the life cord
            success = self.life_cord.form()
            
            # Record metrics
            self.formation_metrics["life_cord"] = {
                "success": success,
                "strength": getattr(self.life_cord, "strength", 0.0),
                "flow_rate": getattr(self.life_cord, "flow_rate", 0.0),
                "connection_quality": getattr(self.life_cord, "connection_quality", 0.0)
            }
            
            logger.info(f"Life Cord formation completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in Life Cord formation: {e}")
            return False
    
    def run_earth_harmonisation(self):
        """
        Run the earth harmonization process.
        
        Returns:
            bool: Success status
        """
        logger.info("Starting Earth Harmonisation process")
        self.current_stage = "earth_harmonisation"
        
        try:
            # Initialize Earth Harmonisation
            self.earth_harmonisation = EarthHarmonisation(self.soul_spark, self.life_cord)
            
            # Run harmonization process
            success = self.earth_harmonisation.harmonize()
            
            # Record metrics
            self.formation_metrics["earth_harmonisation"] = {
                "success": success,
                "earth_resonance": getattr(self.earth_harmonisation, "earth_resonance", 0.0),
                "elemental_balance": getattr(self.earth_harmonisation, "elemental_balance", {})
            }
            
            logger.info(f"Earth Harmonisation completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in Earth Harmonisation: {e}")
            return False
    
    def run_identity_crystallization(self, specified_name=None):
        """
        Run the identity crystallization process.
        
        Args:
            specified_name (str, optional): Optional specified name for the soul
            
        Returns:
            bool: Success status
        """
        logger.info("Starting Identity Crystallization process")
        self.current_stage = "identity_crystallization"
        
        try:
            # Run identity crystallization process
            self.soul_identity, self.identity_metrics = process_identity_crystallization(
                self.life_cord, specified_name=specified_name
            )
            
            success = self.identity_metrics.get("is_fully_crystallized", False)
            
            # Record metrics
            self.formation_metrics["identity_crystallization"] = {
                "success": success,
                "name": self.identity_metrics.get("name", "Unnamed"),
                "soul_color": self.identity_metrics.get("soul_color", "Unknown"),
                "soul_frequency": self.identity_metrics.get("soul_frequency", 0.0),
                "crystallization_level": self.identity_metrics.get("crystallization_level", 0.0)
            }
            
            logger.info(f"Identity Crystallization completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in Identity Crystallization: {e}")
            return False
    
    def run_birth_process(self):
        """
        Run the birth process.
        
        Returns:
            bool: Success status
        """
        logger.info("Starting Birth process")
        self.current_stage = "birth"
        
        try:
            # Initialize Birth process
            self.birth_process = Birth(self.soul_spark, self.life_cord, self.soul_identity)
            
            # Run birth process
            success = self.birth_process.initiate()
            
            # Record metrics
            self.formation_metrics["birth"] = {
                "success": success,
                "birth_time": getattr(self.birth_process, "birth_time", 0.0),
                "birth_quality": getattr(self.birth_process, "birth_quality", 0.0)
            }
            
            logger.info(f"Birth process completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in Birth process: {e}")
            return False
    
    def _record_final_metrics(self):
        """Record final metrics for the complete formation process."""
        
        # Collect overall metrics
        overall_metrics = {
            "formation_complete": self.formation_complete,
            "stability": getattr(self.soul_spark, "stability", 0.0),
            "coherence": getattr(self.soul_spark, "coherence", 0.0),
            "identity": self.identity_metrics.get("name", "Unnamed") if self.identity_metrics else "Unnamed",
            "birth_successful": self.formation_metrics.get("birth", {}).get("success", False)
        }
        
        # Add to formation metrics
        self.formation_metrics["overall"] = overall_metrics
        
        # Record in metrics tracker
        metrics.record_metrics("soul_formation", overall_metrics)
        
        logger.info(f"Final formation metrics recorded: {overall_metrics}")
    
    def get_formation_metrics(self):
        """
        Get all metrics for the soul formation process.
        
        Returns:
            dict: Complete formation metrics
        """
        return self.formation_metrics
    
    def get_formed_soul(self):
        """
        Get the completely formed soul.
        
        Returns:
            tuple: (soul_spark, life_cord, soul_identity)
        """
        if not self.formation_complete:
            logger.warning("Attempting to get formed soul before formation is complete")
        
        return (self.soul_spark, self.life_cord, self.soul_identity)


# Example usage if this module is run directly
if __name__ == "__main__":
    # Create a new soul spark
    spark = SoulSpark()
    
    # Create controller
    controller = SoulFormationController(spark)
    
    # Run full formation process
    success = controller.run_full_formation()
    
    if success:
        # Get the formed soul
        soul_spark, life_cord, soul_identity = controller.get_formed_soul()
        
        print(f"Soul formation complete!")
        print(f"Soul Name: {soul_identity.name}")
        print(f"Soul Color: {soul_identity.soul_color}")
        print(f"Soul Frequency: {soul_identity.soul_frequency:.2f} Hz")
        
        # Display complete metrics
        metrics = controller.get_formation_metrics()
        print(f"\nFormation Time: {metrics['total_formation_time']:.2f} seconds")
        
        # Display overall metrics
        overall = metrics.get("overall", {})
        print(f"\nOverall Metrics:")
        print(f"Stability: {overall.get('stability', 0.0):.2f}")
        print(f"Coherence: {overall.get('coherence', 0.0):.2f}")
    else:
        print("Soul formation failed!")

