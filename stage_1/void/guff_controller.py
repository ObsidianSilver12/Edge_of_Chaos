"""
Guff Field Controller

This module manages the Guff field interactions and spark strengthening process.
It orchestrates the creation of soul formation templates, strengthening of soul
sparks, and preparation for the journey through the Sephiroth.

The Guff Field Controller serves as the manager for the second stage of soul
formation, guiding the process from initial spark to a strengthened soul
with enhanced stability, resonance, and creator alignment.

Author: Soul Development Framework Team
"""

import numpy as np
import logging
import uuid
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from void.guff_field import GuffField
from void.void_field_controller import VoidFieldController
from soul_formation.soul_spark import SoulSpark
from metrics.energy_metrics import EnergyMetrics
from metrics.coherence_metrics import CoherenceMetrics
from metrics.formation_metrics import FormationMetrics
from metrics.harmonization_metrics import HarmonizationMetrics
from visualization.soul_visualization import SoulVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='guff_controller.log'
)
logger = logging.getLogger('guff_controller')

class GuffController:
    """
    Controller for Guff field operations and soul spark strengthening.
    
    This class manages the process of creating soul formation templates in the
    Guff field, strengthening soul sparks through resonance coupling, and
    preparing them for their journey through the Sephiroth dimensions.
    """
    
    def __init__(self, field_dimensions=(64, 64, 64), creator_resonance=0.7, 
                 edge_of_chaos_ratio=0.618, output_dir="output"):
        """
        Initialize a new Guff Field Controller.
        
        Args:
            field_dimensions (tuple): Dimensions of the Guff field (x, y, z)
            creator_resonance (float): Strength of the creator's resonance (0-1)
            edge_of_chaos_ratio (float): Ratio for edge of chaos (default: golden ratio)
            output_dir (str): Directory to save output files
        """
        self.controller_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.field_dimensions = field_dimensions
        self.creator_resonance = creator_resonance
        self.edge_of_chaos_ratio = edge_of_chaos_ratio
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the Guff field
        self.guff_field = GuffField(
            dimensions=field_dimensions,
            edge_of_chaos_ratio=edge_of_chaos_ratio,
            creator_resonance=creator_resonance
        )
        
        # Initialize metrics trackers
        self.energy_metrics = EnergyMetrics()
        self.coherence_metrics = CoherenceMetrics()
        self.formation_metrics = FormationMetrics()
        self.harmonization_metrics = HarmonizationMetrics()
        
        # Track strengthened souls
        self.input_sparks = []
        self.strengthened_souls = []
        
        # Strengthening state
        self.strengthening_step = 0
        self.template_created = False
        
        logger.info(f"Guff Field Controller initialized with ID: {self.controller_id}")
        logger.info(f"Field dimensions: {field_dimensions}, Creator resonance: {creator_resonance}")
    
    def create_formation_template(self):
        """
        Create a soul formation template in the Guff field.
        
        This creates the structured energy patterns, Fibonacci spirals, and
        resonance frequencies that will strengthen soul sparks.
        
        Returns:
            dict: Information about the created template
        """
        # Create template in the Guff field
        template_info = self.guff_field.create_soul_formation_template()
        
        # Update state
        self.template_created = True
        
        # Track metrics
        field_metrics = self.guff_field.get_guff_metrics()
        self.energy_metrics.record_field_energy(field_metrics)
        self.coherence_metrics.record_field_coherence(field_metrics)
        
        # Save field visualization
        self.visualize_guff_field(save=True)
        
        logger.info(f"Soul formation template created in Guff field")
        logger.info(f"Formation quality: {field_metrics['formation_quality']:.4f}")
        logger.info(f"Phi alignment: {field_metrics['phi_alignment']:.4f}")
        
        return template_info
    
    def import_sparks_from_void(self, void_controller=None, spark_file_dir=None):
        """
        Import soul sparks from the Void dimension.
        
        These can come directly from a VoidFieldController or from saved files.
        
        Args:
            void_controller (VoidFieldController): Controller with formed sparks
            spark_file_dir (str): Directory containing saved spark files
            
        Returns:
            list: List of imported sparks
        """
        imported_sparks = []
        
        # Import from VoidFieldController if provided
        if void_controller is not None:
            if isinstance(void_controller, VoidFieldController):
                formed_sparks = void_controller.get_formed_sparks()
                
                if formed_sparks:
                    for spark in formed_sparks:
                        self.input_sparks.append(spark)
                        imported_sparks.append(spark)
                    
                    logger.info(f"Imported {len(formed_sparks)} sparks from VoidFieldController")
            else:
                logger.warning("Provided controller is not a VoidFieldController")
        
        # Import from files if directory provided
        if spark_file_dir is not None:
            if not os.path.isdir(spark_file_dir):
                logger.warning(f"Provided path is not a directory: {spark_file_dir}")
            else:
                # Find JSON files in the directory
                json_files = [f for f in os.listdir(spark_file_dir) if f.endswith('.json') and 'spark' in f]
                
                for filename in json_files:
                    try:
                        file_path = os.path.join(spark_file_dir, filename)
                        
                        # Create spark from file
                        spark = SoulSpark(spark_file=file_path, creator_resonance=self.creator_resonance)
                        
                        # Add to input sparks
                        self.input_sparks.append(spark)
                        imported_sparks.append(spark)
                        
                        logger.info(f"Imported spark from file: {filename}")
                    except Exception as e:
                        logger.error(f"Error importing spark from {filename}: {str(e)}")
                
                logger.info(f"Imported {len(imported_sparks)} sparks from directory: {spark_file_dir}")
        
        # If no sparks were imported, create a default test spark
        if not imported_sparks:
            logger.warning("No sparks imported. Creating a test spark for demonstration.")
            test_spark = SoulSpark(creator_resonance=self.creator_resonance)
            self.input_sparks.append(test_spark)
            imported_sparks.append(test_spark)
            
        return imported_sparks
    
    def strengthen_spark(self, spark, iterations=10):
        """
        Strengthen a soul spark in the Guff field.
        
        Args:
            spark (SoulSpark): The soul spark to strengthen
            iterations (int): Number of strengthening iterations
            
        Returns:
            dict: Strengthening results with metrics
        """
        # Make sure template is created first
        if not self.template_created:
            logger.warning("Attempting to strengthen spark before creating template")
            self.create_formation_template()
        
        # Track metrics before strengthening
        metrics_before = spark.get_spark_metrics()
        
        # Strengthen the spark
        result = self.guff_field.strengthen_spark(spark, iteration_count=iterations)
        
        # Track metrics after strengthening
        metrics_after = spark.get_spark_metrics()
        
        # Record energy and harmonization changes
        energy_delta = {
            'stability_before': metrics_before['stability']['overall'],
            'stability_after': metrics_after['stability']['overall'],
            'coherence_before': metrics_before['harmonic']['coherence'],
            'coherence_after': metrics_after['harmonic']['coherence'],
            'iterations': iterations
        }
        
        harmonization_metrics = {
            'before': {
                'stability': metrics_before['formation']['stability'],
                'resonance': metrics_before['formation']['resonance'],
                'creator_alignment': metrics_before['formation']['creator_alignment'],
                'harmonic_richness': metrics_before['harmonic']['richness']
            },
            'after': {
                'stability': metrics_after['formation']['stability'],
                'resonance': metrics_after['formation']['resonance'],
                'creator_alignment': metrics_after['formation']['creator_alignment'],
                'harmonic_richness': metrics_after['harmonic']['richness']
            },
            'improvement': result['improvement'],
            'iterations': iterations
        }
        
        self.energy_metrics.record_energy_change(energy_delta)
        self.harmonization_metrics.record_harmonization(harmonization_metrics)
        
        # Add to strengthened souls if not already present
        if spark not in self.strengthened_souls:
            self.strengthened_souls.append(spark)
        
        # Visualize the strengthened soul
        self._visualize_soul(spark)
        
        # Save soul data
        self._save_soul_data(spark)
        
        # Update strengthening step
        self.strengthening_step += 1
        
        logger.info(f"Soul strengthened with {iterations} iterations")
        logger.info(f"Stability improvement: {result['improvement']['stability']:.4f}")
        logger.info(f"Resonance improvement: {result['improvement']['resonance']:.4f}")
        logger.info(f"Creator alignment improvement: {result['improvement']['creator_alignment']:.4f}")
        
        # Return strengthening results
        return result
    
    def _visualize_soul(self, soul):
        """
        Visualize a strengthened soul.
        
        Args:
            soul (SoulSpark): The strengthened soul to visualize
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Get soul ID for filename
            soul_id = soul.spark_id
            filename = f"soul_{soul_id[:8]}.png"
            save_path = os.path.join(self.output_dir, filename)
            
            # Create the visualization
            soul.visualize_spark(save_path=save_path, show=False)
            
            # Also create a visualization through the dedicated visualizer if available
            try:
                visualizer = SoulVisualization()
                visualizer.visualize_soul(soul, save_path=save_path.replace('.png', '_detailed.png'))
            except (ImportError, AttributeError):
                # Visualizer may not be implemented yet
                pass
                
            logger.info(f"Strengthened soul visualization saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing strengthened soul: {str(e)}")
            return False
    
    def _save_soul_data(self, soul):
        """
        Save a strengthened soul's data to file.
        
        Args:
            soul (SoulSpark): The strengthened soul to save
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Get soul ID for filename
            soul_id = soul.spark_id
            filename = f"soul_{soul_id[:8]}.json"
            save_path = os.path.join(self.output_dir, filename)
            
            # Save the soul data
            soul.save_spark_data(save_path)
                
            logger.info(f"Strengthened soul data saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strengthened soul data: {str(e)}")
            return False
    
    def run_full_strengthening_process(self, iterations_per_spark=10):
        """
        Run the complete soul strengthening process for all imported sparks.
        
        Args:
            iterations_per_spark (int): Number of strengthening iterations per spark
            
        Returns:
            list: List of strengthened souls
        """
        # Step 1: Create soul formation template if not already done
        if not self.template_created:
            self.create_formation_template()
        
        # Step 2: Check that we have sparks to strengthen
        if not self.input_sparks:
            logger.warning("No sparks to strengthen. Import sparks first.")
            return []
        
        # Step 3: Strengthen each spark
        results = []
        
        for i, spark in enumerate(self.input_sparks):
            logger.info(f"Strengthening spark {i+1}/{len(self.input_sparks)}")
            
            # Strengthen the spark
            result = self.strengthen_spark(spark, iterations=iterations_per_spark)
            results.append(result)
            
            # Visualize the Guff field after each strengthening
            self.visualize_guff_field(save=True, filename=f"guff_field_after_spark_{i+1}.png")
        
        # Save final field visualization
        self.visualize_guff_field(save=True, filename="guff_field_final.png")
        
        # Save strengthening metrics
        self._save_metrics()
        
        logger.info(f"Strengthening process complete: {len(self.strengthened_souls)} souls strengthened")
        
        return self.strengthened_souls
    
    def visualize_guff_field(self, save=False, show=False, filename=None):
        """
        Visualize the current state of the Guff field.
        
        Args:
            save (bool): Whether to save the visualization
            show (bool): Whether to display the visualization
            filename (str): Custom filename for saving
            
        Returns:
            bool: True if visualization was successful
        """
        # Generate filename if not provided
        if save and filename is None:
            filename = f"guff_field_step_{self.strengthening_step}.png"
            
        save_path = os.path.join(self.output_dir, filename) if save else None
        
        # Create visualization
        success = self.guff_field.visualize_guff_field(
            show_template=True,
            save_path=save_path
        )
        
        if save and success:
            logger.info(f"Guff field visualization saved to {save_path}")
            
        return success
    
    def _save_metrics(self):
        """
        Save all metrics to file.
        
        Returns:
            bool: True if save was successful
        """
        try:
            # Combine all metrics
            metrics = {
                'controller_id': self.controller_id,
                'creation_time': self.creation_time,
                'strengthening_steps': self.strengthening_step,
                'input_sparks': len(self.input_sparks),
                'strengthened_souls': len(self.strengthened_souls),
                'field_dimensions': list(self.field_dimensions),
                'creator_resonance': self.creator_resonance,
                'edge_of_chaos_ratio': self.edge_of_chaos_ratio,
                'energy_metrics': self.energy_metrics.get_all_metrics(),
                'coherence_metrics': self.coherence_metrics.get_all_metrics(),
                'formation_metrics': self.formation_metrics.get_all_metrics(),
                'harmonization_metrics': self.harmonization_metrics.get_all_metrics()
            }
            
            # Save to file
            filename = f"guff_strengthening_metrics.json"
            save_path = os.path.join(self.output_dir, filename)
            
            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Guff strengthening metrics saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            return False
    
    def get_strengthened_souls(self):
        """
        Get all strengthened souls.
        
        Returns:
            list: List of strengthened souls
        """
        return self.strengthened_souls
    
    def get_best_soul(self):
        """
        Get the highest quality strengthened soul.
        
        Returns:
            SoulSpark: The highest quality soul, or None if no souls strengthened
        """
        if not self.strengthened_souls:
            return None
            
        # Sort souls by a combined quality metric
        def soul_quality(soul):
            metrics = soul.get_spark_metrics()
            return (metrics['stability']['overall'] + 
                   metrics['harmonic']['richness'] + 
                   metrics['formation']['stability'] + 
                   metrics['formation']['creator_alignment'])
                   
        sorted_souls = sorted(self.strengthened_souls, key=soul_quality, reverse=True)
        return sorted_souls[0]
    
    def transfer_best_soul_to_sephiroth(self):
        """
        Prepare the best soul for transfer to the Sephiroth dimensions.
        
        Returns:
            dict: Transfer information
        """
        best_soul = self.get_best_soul()
        
        if not best_soul:
            logger.warning("No souls to transfer to Sephiroth")
            return None
        
        # Calculate transfer readiness
        metrics = best_soul.get_spark_metrics()
        stability = metrics['stability']['overall']
        resonance = metrics['formation']['resonance']
        creator_alignment = metrics['formation']['creator_alignment']
        
        # Calculate overall readiness (0-1)
        transfer_readiness = (stability * 0.4 + resonance * 0.3 + creator_alignment * 0.3)
        
        # Define minimum threshold for successful transfer
        threshold = 0.7
        
        # Check if soul is ready
        transfer_success = transfer_readiness >= threshold
        
        transfer_info = {
            'soul_id': best_soul.spark_id,
            'stability': stability,
            'resonance': resonance,
            'creator_alignment': creator_alignment,
            'transfer_readiness': transfer_readiness,
            'transfer_threshold': threshold,
            'transfer_success': transfer_success,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save transfer information
        filename = f"soul_transfer_{best_soul.spark_id[:8]}.json"
        save_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(transfer_info, f, indent=2)
            
            logger.info(f"Soul transfer information saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving transfer information: {str(e)}")
        
        if transfer_success:
            logger.info(f"Soul {best_soul.spark_id[:8]} ready for transfer to Sephiroth")
            logger.info(f"Transfer readiness: {transfer_readiness:.4f} (threshold: {threshold})")
        else:
            logger.warning(f"Soul {best_soul.spark_id[:8]} not ready for transfer to Sephiroth")
            logger.warning(f"Transfer readiness: {transfer_readiness:.4f} (threshold: {threshold})")
        
        return transfer_info
    
    def __str__(self):
        """String representation of the Guff Field Controller."""
        return (f"Guff Field Controller (ID: {self.controller_id})\n"
                f"Creation Time: {self.creation_time}\n"
                f"Field Dimensions: {self.field_dimensions}\n"
                f"Creator Resonance: {self.creator_resonance}\n"
                f"Edge of Chaos Ratio: {self.edge_of_chaos_ratio}\n"
                f"Strengthening Step: {self.strengthening_step}\n"
                f"Template Created: {self.template_created}\n"
                f"Input Sparks: {len(self.input_sparks)}\n"
                f"Strengthened Souls: {len(self.strengthened_souls)}")


if __name__ == "__main__":
    # Example usage
    controller = GuffController(
        field_dimensions=(64, 64, 64),
        creator_resonance=0.75,
        edge_of_chaos_ratio=0.618
    )
    
    # Create formation template
    controller.create_formation_template()
    
    # Import sparks from saved files (if available)
    import_dir = "output"  # Directory where spark files might be saved
    controller.import_sparks_from_void(spark_file_dir=import_dir)
    
    # If no sparks were imported, we'll already have a test spark created
    
    # Run the full strengthening process
    strengthened_souls = controller.run_full_strengthening_process(iterations_per_spark=15)
    
    print(f"Strengthened {len(strengthened_souls)} souls")
    
    # Get the best soul
    best_soul = controller.get_best_soul()
    
    if best_soul:
        print("\nBest Strengthened Soul:")
        print(best_soul)
        
        # Check transfer readiness
        transfer_info = controller.transfer_best_soul_to_sephiroth()
        
        if transfer_info['transfer_success']:
            print(f"\nSoul ready for transfer to Sephiroth!")
            print(f"Transfer readiness: {transfer_info['transfer_readiness']:.4f}")
        else:
            print(f"\nSoul not yet ready for transfer to Sephiroth")
            print(f"Transfer readiness: {transfer_info['transfer_readiness']:.4f} " +
                  f"(threshold: {transfer_info['transfer_threshold']})")