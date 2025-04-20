"""
Void Field Controller

This module controls the void operations and soul spark formation process.
It orchestrates the embedding of sacred geometry patterns, quantum fluctuations,
and the detection and formation of soul sparks in the Void dimension.

The Void Field Controller serves as the manager for the initial stage of soul
formation, guiding the process from pure potentiality to the emergence of
coherent soul sparks.

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
from void.void_field import VoidField
from soul_formation.soul_spark import SoulSpark
from metrics.energy_metrics import EnergyMetrics
from metrics.coherence_metrics import CoherenceMetrics
from metrics.formation_metrics import FormationMetrics
from visualization.spark_visualization import SparkVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='void_field_controller.log'
)
logger = logging.getLogger('void_field_controller')

class VoidFieldController:
    """
    Controller for void field operations and soul spark formation.
    
    This class manages the process of embedding sacred geometry patterns in the
    void field, simulating quantum fluctuations, detecting potential wells for
    spark formation, and creating soul sparks at those wells.
    """
    
    def __init__(self, field_dimensions=(64, 64, 64), creator_resonance=0.7, 
                edge_of_chaos_ratio=0.618, output_dir="output"):
        """
        Initialize a new Void Field Controller.
        
        Args:
            field_dimensions (tuple): Dimensions of the void field (x, y, z)
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
        
        # Initialize the void field
        self.void_field = VoidField(
            dimensions=field_dimensions,
            edge_of_chaos_ratio=edge_of_chaos_ratio,
            creator_resonance=creator_resonance
        )
        
        # Initialize metrics trackers
        self.energy_metrics = EnergyMetrics()
        self.coherence_metrics = CoherenceMetrics()
        self.formation_metrics = FormationMetrics()
        
        # Track formed sparks
        self.formed_sparks = []
        self.potential_wells = []
        
        # Simulation state
        self.simulation_step = 0
        self.patterns_embedded = False
        self.wells_identified = False
        
        logger.info(f"Void Field Controller initialized with ID: {self.controller_id}")
        logger.info(f"Field dimensions: {field_dimensions}, Creator resonance: {creator_resonance}")
    
    def embed_sacred_patterns(self):
        """
        Embed sacred geometry patterns in the void field.
        
        This creates the structured energy patterns that will serve as
        potential wells for soul spark formation.
        
        Returns:
            dict: Information about embedded patterns
        """
        # Embed patterns in the void field
        patterns = self.void_field.embed_sacred_geometry()
        
        # Update state
        self.patterns_embedded = True
        
        # Track metrics
        field_metrics = self.void_field.get_void_metrics()
        self.energy_metrics.record_field_energy(field_metrics)
        self.coherence_metrics.record_field_coherence(field_metrics)
        
        # Save field visualization
        self.visualize_void_field(save=True)
        
        logger.info(f"Sacred geometry patterns embedded in void field")
        logger.info(f"Stability: {field_metrics['stability']:.4f}, Coherence: {field_metrics['coherence']:.4f}")
        
        return patterns
    
    def identify_potential_wells(self):
        """
        Identify potential wells in the void field.
        
        These wells are points where sacred geometry patterns intersect and
        create energy concentrations that can support soul spark formation.
        
        Returns:
            list: List of potential wells
        """
        # Make sure patterns are embedded first
        if not self.patterns_embedded:
            logger.warning("Attempting to identify wells before embedding patterns")
            self.embed_sacred_patterns()
        
        # Identify potential wells
        self.potential_wells = self.void_field.identify_potential_wells()
        
        # Update state
        self.wells_identified = True
        
        # Track metrics
        well_metrics = {
            'num_wells': len(self.potential_wells),
            'average_quality': np.mean([w['quality'] for w in self.potential_wells]) if self.potential_wells else 0,
            'max_quality': np.max([w['quality'] for w in self.potential_wells]) if self.potential_wells else 0,
            'min_quality': np.min([w['quality'] for w in self.potential_wells]) if self.potential_wells else 0
        }
        
        self.formation_metrics.record_well_formation(well_metrics)
        
        logger.info(f"Identified {len(self.potential_wells)} potential wells in void field")
        logger.info(f"Average well quality: {well_metrics['average_quality']:.4f}")
        
        return self.potential_wells
    
    def simulate_quantum_fluctuations(self, iterations=10, fluctuation_strength=0.03):
        """
        Simulate quantum fluctuations in the void field.
        
        These fluctuations can lead to soul spark formation when they occur
        at potential wells with sufficient quality.
        
        Args:
            iterations (int): Number of simulation iterations
            fluctuation_strength (float): Strength of quantum fluctuations
            
        Returns:
            list: List of sparks formed during the simulation
        """
        # Make sure wells are identified first
        if not self.wells_identified:
            logger.warning("Attempting to simulate fluctuations before identifying wells")
            self.identify_potential_wells()
        
        # Track metrics before simulation
        field_metrics_before = self.void_field.get_void_metrics()
        
        # Simulate quantum fluctuations
        spark_formations = self.void_field.simulate_quantum_fluctuations(
            iterations=iterations,
            fluctuation_strength=fluctuation_strength
        )
        
        # Track metrics after simulation
        field_metrics_after = self.void_field.get_void_metrics()
        
        # Record energy changes
        energy_delta = {
            'stability_before': field_metrics_before['stability'],
            'stability_after': field_metrics_after['stability'],
            'coherence_before': field_metrics_before['coherence'],
            'coherence_after': field_metrics_after['coherence'],
            'num_fluctuations': iterations,
            'fluctuation_strength': fluctuation_strength
        }
        
        self.energy_metrics.record_energy_change(energy_delta)
        
        # Process any formed sparks
        for spark_data in spark_formations:
            # Create a SoulSpark object
            spark = SoulSpark(spark_data=spark_data, creator_resonance=self.creator_resonance)
            
            # Add to formed sparks list
            self.formed_sparks.append(spark)
            
            # Record formation metrics
            self.formation_metrics.record_spark_formation(spark.get_spark_metrics())
            
            # Visualize the spark
            self._visualize_spark(spark)
            
            # Save spark data
            self._save_spark_data(spark)
        
        # Update simulation step
        self.simulation_step += iterations
        
        logger.info(f"Simulated {iterations} quantum fluctuation iterations")
        logger.info(f"Formed {len(spark_formations)} new soul sparks")
        
        return spark_formations
    
    def _visualize_spark(self, spark):
        """
        Visualize a formed soul spark.
        
        Args:
            spark (SoulSpark): The soul spark to visualize
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Get spark ID for filename
            spark_id = spark.spark_id
            filename = f"spark_{spark_id[:8]}.png"
            save_path = os.path.join(self.output_dir, filename)
            
            # Create the visualization
            spark.visualize_spark(save_path=save_path, show=False)
            
            # Also create a visualization through the dedicated visualizer if available
            try:
                visualizer = SparkVisualization()
                visualizer.visualize_spark(spark, save_path=save_path.replace('.png', '_detailed.png'))
            except (ImportError, AttributeError):
                # Visualizer may not be implemented yet
                pass
                
            logger.info(f"Soul spark visualization saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing soul spark: {str(e)}")
            return False
    
    def _save_spark_data(self, spark):
        """
        Save a formed soul spark's data to file.
        
        Args:
            spark (SoulSpark): The soul spark to save
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Get spark ID for filename
            spark_id = spark.spark_id
            filename = f"spark_{spark_id[:8]}.json"
            save_path = os.path.join(self.output_dir, filename)
            
            # Save the spark data
            spark.save_spark_data(save_path)
                
            logger.info(f"Soul spark data saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving soul spark data: {str(e)}")
            return False
    
    def run_full_spark_formation(self, target_sparks=3, max_iterations=100):
        """
        Run the complete soul spark formation process.
        
        This executes all steps from pattern embedding to spark formation,
        continuing until a target number of sparks are formed or the
        maximum iterations are reached.
        
        Args:
            target_sparks (int): Target number of sparks to form
            max_iterations (int): Maximum simulation iterations
            
        Returns:
            list: List of formed soul sparks
        """
        # Step 1: Embed sacred geometry patterns
        self.embed_sacred_patterns()
        
        # Step 2: Identify potential wells
        self.identify_potential_wells()
        
        # Step 3: Simulate quantum fluctuations until target reached
        total_iterations = 0
        iteration_batch = 10  # Number of iterations per batch
        
        while len(self.formed_sparks) < target_sparks and total_iterations < max_iterations:
            # Run a batch of fluctuation iterations
            new_sparks = self.simulate_quantum_fluctuations(
                iterations=iteration_batch,
                fluctuation_strength=0.02 + 0.01 * np.random.random()  # Slight randomization
            )
            
            total_iterations += iteration_batch
            
            # Visualize the void field periodically
            if total_iterations % 20 == 0 or new_sparks:
                self.visualize_void_field(save=True)
            
            # Adjust fluctuation batch size based on progress
            if len(self.formed_sparks) > 0 and len(self.formed_sparks) < target_sparks / 2:
                # More aggressive fluctuations if we have some but not enough sparks
                iteration_batch = 15
            elif len(self.formed_sparks) >= target_sparks / 2:
                # More careful, smaller batches as we approach target
                iteration_batch = 5
            
            logger.info(f"Formation progress: {len(self.formed_sparks)}/{target_sparks} sparks, " +
                       f"{total_iterations}/{max_iterations} iterations")
        
        # Save final field visualization
        self.visualize_void_field(save=True, filename="void_field_final.png")
        
        # Save formation metrics
        self._save_metrics()
        
        logger.info(f"Spark formation complete: {len(self.formed_sparks)}/{target_sparks} sparks formed")
        logger.info(f"Total iterations: {total_iterations}")
        
        return self.formed_sparks
    
    def visualize_void_field(self, save=False, show=False, filename=None):
        """
        Visualize the current state of the void field.
        
        Args:
            save (bool): Whether to save the visualization
            show (bool): Whether to display the visualization
            filename (str): Custom filename for saving
            
        Returns:
            bool: True if visualization was successful
        """
        # Generate filename if not provided
        if save and filename is None:
            filename = f"void_field_step_{self.simulation_step}.png"
            
        save_path = os.path.join(self.output_dir, filename) if save else None
        
        # Create visualization
        success = self.void_field.visualize_void_field(
            show_wells=True,
            show_sparks=True,
            save_path=save_path
        )
        
        if save and success:
            logger.info(f"Void field visualization saved to {save_path}")
            
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
                'simulation_steps': self.simulation_step,
                'sparks_formed': len(self.formed_sparks),
                'potential_wells': len(self.potential_wells),
                'field_dimensions': list(self.field_dimensions),
                'creator_resonance': self.creator_resonance,
                'edge_of_chaos_ratio': self.edge_of_chaos_ratio,
                'energy_metrics': self.energy_metrics.get_all_metrics(),
                'coherence_metrics': self.coherence_metrics.get_all_metrics(),
                'formation_metrics': self.formation_metrics.get_all_metrics()
            }
            
            # Save to file
            filename = f"void_formation_metrics.json"
            save_path = os.path.join(self.output_dir, filename)
            
            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Void formation metrics saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            return False
    
    def get_formed_sparks(self):
        """
        Get all formed soul sparks.
        
        Returns:
            list: List of formed SoulSpark objects
        """
        return self.formed_sparks
    
    def get_best_spark(self):
        """
        Get the highest quality soul spark.
        
        Returns:
            SoulSpark: The highest quality spark, or None if no sparks formed
        """
        if not self.formed_sparks:
            return None
            
        # Sort sparks by a combined quality metric
        def spark_quality(spark):
            metrics = spark.get_spark_metrics()
            return (metrics['stability']['overall'] + 
                   metrics['harmonic']['richness'] + 
                   metrics['formation']['stability'])
                   
        sorted_sparks = sorted(self.formed_sparks, key=spark_quality, reverse=True)
        return sorted_sparks[0]
    
    def __str__(self):
        """String representation of the Void Field Controller."""
        return (f"Void Field Controller (ID: {self.controller_id})\n"
                f"Creation Time: {self.creation_time}\n"
                f"Field Dimensions: {self.field_dimensions}\n"
                f"Creator Resonance: {self.creator_resonance}\n"
                f"Edge of Chaos Ratio: {self.edge_of_chaos_ratio}\n"
                f"Simulation Step: {self.simulation_step}\n"
                f"Patterns Embedded: {self.patterns_embedded}\n"
                f"Potential Wells: {len(self.potential_wells)}\n"
                f"Formed Sparks: {len(self.formed_sparks)}")


if __name__ == "__main__":
    # Example usage
    controller = VoidFieldController(
        field_dimensions=(64, 64, 64),
        creator_resonance=0.75,
        edge_of_chaos_ratio=0.618
    )
    
    # Run the full spark formation process
    formed_sparks = controller.run_full_spark_formation(target_sparks=3, max_iterations=100)
    
    print(f"Formed {len(formed_sparks)} soul sparks")
    
    # Get the best spark
    best_spark = controller.get_best_spark()
    
    if best_spark:
        print("\nBest Soul Spark:")
        print(best_spark)