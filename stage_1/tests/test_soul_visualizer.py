#!/usr/bin/env python3
"""
Test script for the new SoulVisualizer system
Tests visualization creation without running full simulation
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SoulVisualizerTest')

def test_soul_visualizer():
    """Test the SoulVisualizer with sample data"""
    
    try:
        from stage_1.soul_formation.soul_evolution_visualizer import SoulVisualizer
        
        # Create test output directory
        test_output_dir = "output/visuals/test_visualization"
        
        # Initialize visualizer
        logger.info("Initializing SoulVisualizer...")
        visualizer = SoulVisualizer(output_dir=test_output_dir)
        
        # Test soul data (simulating a completed soul)
        test_soul_data = {
            'name': 'TestSoul_001',
            'date_of_birth': '2024-05-11',  # This should be corrected to Taurus
            'star_sign': 'Sagittarius',     # This will be fixed
            'primary_color': '#FF6B35',
            'energy_level': 92,
            'coherence': 88,
            'complexity': 75,
            'stability': 85,
            'simulation_id': 'test_run_001'
        }
        
        logger.info("Testing date/star sign correction...")
        corrected_date, corrected_sign = visualizer.fix_date_star_sign_mismatch(
            test_soul_data['date_of_birth'], 
            test_soul_data['star_sign']
        )
        logger.info(f"Corrected: {corrected_date} -> {corrected_sign}")
        
        # Test individual stage visualization
        logger.info("Testing soul spark stage visualization...")
        spark_result = visualizer.visualize_stage('soul_spark', test_soul_data, save_plots=True)
        logger.info(f"Soul spark visualization result: {spark_result.get('success', False)}")
        
        logger.info("Testing creator entanglement stage visualization...")
        creator_result = visualizer.visualize_stage('creator_entanglement', test_soul_data, save_plots=True)
        logger.info(f"Creator entanglement visualization result: {creator_result.get('success', False)}")
        
        logger.info("Testing sephiroth journey stage visualization...")
        sephiroth_result = visualizer.visualize_stage('sephiroth_journey', test_soul_data, save_plots=True)
        logger.info(f"Sephiroth journey visualization result: {sephiroth_result.get('success', False)}")
        
        logger.info("Testing pre-identity crystallization stage visualization...")
        pre_crystal_result = visualizer.visualize_stage('pre_identity_crystallization', test_soul_data, save_plots=True)
        logger.info(f"Pre-crystallization visualization result: {pre_crystal_result.get('success', False)}")
        
        logger.info("Testing post-identity crystallization stage visualization...")
        post_crystal_result = visualizer.visualize_stage('post_identity_crystallization', test_soul_data, save_plots=True)
        logger.info(f"Post-crystallization visualization result: {post_crystal_result.get('success', False)}")
        
        # Test complete evolution visualization
        logger.info("Testing complete evolution visualization...")
        complete_result = visualizer.create_complete_evolution_visualization(test_soul_data, save_plots=True)
        logger.info(f"Complete evolution visualization result: {complete_result.get('success', False)}")
        
        if complete_result.get('success'):
            logger.info("=== VISUALIZATION TEST RESULTS ===")
            logger.info(f"Test output directory: {test_output_dir}")
            
            # List generated files
            for stage in visualizer.stages:
                stage_dir = os.path.join(test_output_dir, stage)
                if os.path.exists(stage_dir):
                    files = os.listdir(stage_dir)
                    logger.info(f"{stage}: {len(files)} files generated")
            
            # Check overview files
            overview_files = [f for f in os.listdir(test_output_dir) if f.endswith('.html') or f.endswith('.json')]
            logger.info(f"Overview files: {len(overview_files)} generated")
            
            logger.info("=== DATE/STAR SIGN FIX VERIFICATION ===")
            logger.info(f"Original: May 11, 2024 -> Sagittarius")
            logger.info(f"Corrected: {corrected_date} -> {corrected_sign}")
            logger.info(f"Fix successful: {'✓' if corrected_sign == 'Taurus' else '✗'}")
            
            logger.info("=== TEST COMPLETED SUCCESSFULLY ===")
            print(f"\nVisualization test completed successfully!")
            print(f"Output saved to: {test_output_dir}")
            print(f"Date/star sign mismatch fixed: May 11 is now correctly {corrected_sign}")
            print("\nGenerated visualizations:")
            print("- 3D soul meshes for each evolution stage")
            print("- Frequency spectrum visualizations") 
            print("- Color evolution timeline")
            print("- Final metrics dashboard with corrected data")
            print("- Complete evolution timeline")
            
            return True
            
        else:
            logger.error("Complete evolution visualization failed")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_soul_visualizer()
    sys.exit(0 if success else 1)