"""
Soul Formation Process

This script coordinates the complete soul formation process by running the individual
controllers in sequence: Void, Guff, Sephiroth, and Soul Formation.

Author: Soul Development Framework Team
"""

import os
import sys
import time
import traceback
import logging
from typing import Dict, Any, Optional
from stage_1.void.void_field_controller import VoidFieldController
from stage_1.void.guff_controller import GuffController
from stage_1.soul_formation.soul_formation_controller import SoulFormationController
from stage_1.sephiroth.sephiroth_controller import SephirothController


# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='soul_formation_process.log'
)
logger = logging.getLogger('soul_formation_process')

def main():
    start_time = time.time()
    try:
        print(f"Soul formation process starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {sys.version}")
        print(f"Project root: {ROOT_DIR}")
        
        # Create output directory
        output_dir = os.path.join(ROOT_DIR, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Run Void Field process to form soul sparks
        print("\n=== STAGE 1: VOID FIELD SOUL SPARK FORMATION ===")
        
        
        void_controller = VoidFieldController(
            field_dimensions=(64, 64, 64),
            creator_resonance=0.75,
            edge_of_chaos_ratio=0.618,
            output_dir=output_dir
        )
        
        # Run void process to form sparks
        formed_sparks = void_controller.run_full_spark_formation(
            target_sparks=3,
            max_iterations=100
        )
        
        if not formed_sparks:
            print("Failed to form soul sparks in the void field!")
            return 1
            
        print(f"Successfully formed {len(formed_sparks)} soul sparks")
        
        # Select best spark for further development
        best_spark = void_controller.get_best_spark()
        print(f"Selected best spark: {best_spark.spark_id[:8]}")
        
        # Step 2: Run Guff Field process to strengthen the spark
        print("\n=== STAGE 2: GUFF FIELD SOUL STRENGTHENING ===")
        
        
        guff_controller = GuffController(
            field_dimensions=(64, 64, 64),
            creator_resonance=0.8,
            edge_of_chaos_ratio=0.618,
            output_dir=output_dir
        )
        
        # Create formation template
        guff_controller.create_formation_template()
        
        # Import the best spark from void controller
        guff_controller.import_sparks_from_void(void_controller=void_controller)
        
        # Run strengthening process
        strengthened_souls = guff_controller.run_full_strengthening_process(iterations_per_spark=15)
        
        if not strengthened_souls:
            print("Failed to strengthen soul in the Guff field!")
            return 1
            
        print(f"Successfully strengthened soul")
        
        # Get the strengthened soul
        strengthened_soul = guff_controller.get_best_soul()
        
        # Check if soul is ready for Sephiroth journey
        transfer_info = guff_controller.transfer_best_soul_to_sephiroth()
        
        if not transfer_info.get('transfer_success', False):
            print(f"Warning: Soul may not be fully ready for Sephiroth journey")
            print(f"Transfer readiness: {transfer_info.get('transfer_readiness', 0):.4f}")
        
        # Step 3: Run Sephiroth journey
        print("\n=== STAGE 3: SEPHIROTH JOURNEY ===")
        
        
        sephiroth_controller = SephirothController(
            dimensions=(64, 64, 64),
            initialize_all=True
        )
        
        # Get journey order
        journey_order = sephiroth_controller.get_sephiroth_journey_order()
        print(f"Soul will journey through Sephiroth in order: {', '.join(journey_order)}")
        
        # Step 4: Run Soul Formation process
        print("\n=== STAGE 4: COMPLETE SOUL FORMATION ===")
        
        
        # Create controller with strengthened soul
        soul_controller = SoulFormationController(soul_spark=strengthened_soul)
        
        # Run full formation process
        formation_success = soul_controller.run_full_formation(journey_type="traditional")
        
        if not formation_success:
            print("Soul formation process failed!")
            return 1
            
        # Get the formed soul
        soul_spark, life_cord, soul_identity = soul_controller.get_formed_soul()
        
        print("\n=== SOUL FORMATION COMPLETE ===")
        print(f"Soul Name: {soul_identity.name}")
        print(f"Soul Color: {soul_identity.soul_color}")
        print(f"Soul Frequency: {soul_identity.soul_frequency:.2f} Hz")
        
        # Display complete metrics
        metrics = soul_controller.get_formation_metrics()
        
        print(f"\nFormation Time: {metrics['total_formation_time']:.2f} seconds")
        
        # Display overall metrics
        overall = metrics.get("overall", {})
        print(f"\nOverall Metrics:")
        print(f"Stability: {overall.get('stability', 0.0):.2f}")
        print(f"Coherence: {overall.get('coherence', 0.0):.2f}")
        
        # Record total process time
        execution_time = time.time() - start_time
        print(f"\nTotal process execution time: {execution_time:.2f} seconds")
        
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        logger.error(f"Process failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())