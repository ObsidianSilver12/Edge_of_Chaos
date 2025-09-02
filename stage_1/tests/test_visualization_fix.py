#!/usr/bin/env python3
"""
Quick test to verify the visualization fixes work correctly
"""

import sys
import os
sys.path.append('.')

from stage_1.soul_formation.soul_evolution_visualizer import SoulVisualizer

def test_visualizer_with_early_stage():
    """Test visualizer with early stage data (no identity yet)"""
    print("Testing Soul Visualizer with early stage data...")
    
    # Create visualizer
    visualizer = SoulVisualizer(output_dir="output/visuals/test_fix")
    
    # Test data for early stage (Spark Emergence) - MINIMAL DATA ONLY
    early_soul_data = {
        'spark_id': 'Soul_test_001',
        'energy_level': 75.5,     # Real computed value
        'coherence': 82.3,        # Real computed value
        'complexity': 45.7,       # Real computed value
        'stability': 68.9,        # Real computed value
        'frequency': 432.7,       # Real frequency from soul spark
        'stage': 'spark_emergence',
        'simulation_id': 'test_001'
        # NO name, birth_date, star_sign, primary_color for early stages
    }
    
    print(f"Testing with data: {early_soul_data}")
    
    try:
        # Test the visualization
        result = visualizer.visualize_stage('soul_spark', early_soul_data, save_plots=True, show_plots=False)
        print(f"Visualization result: {result}")
        
        if result.get('success'):
            print("✅ Early stage visualization PASSED")
        else:
            print(f"❌ Early stage visualization FAILED: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Visualization test failed with exception: {e}")
        import traceback
        traceback.print_exc()

def test_visualizer_with_late_stage():
    """Test visualizer with late stage data (full identity)"""
    print("\nTesting Soul Visualizer with late stage data...")
    
    visualizer = SoulVisualizer(output_dir="output/visuals/test_fix")
    
    # Test data for late stage (Post Identity Crystallization)
    late_soul_data = {
        'spark_id': 'Soul_test_002',
        'name': 'TestSoul',           # Real name from identity crystallization
        'date_of_birth': '2024-05-11', # Real birth date
        'star_sign': 'Taurus',        # Real star sign (corrected)
        'primary_color': '#FF6B35',   # Real color from color processing
        'energy_level': 88.2,         # Real computed value
        'coherence': 94.1,            # Real computed value
        'complexity': 76.8,           # Real computed value
        'stability': 91.5,            # Real computed value
        'frequency': 528.3,           # Real frequency from soul spark
        'voice_frequency': 440.0,     # Real voice frequency from identity crystallization
        'stage': 'post_identity_crystallization',
        'simulation_id': 'test_002'
    }
    
    print(f"Testing with data: {late_soul_data}")
    
    try:
        result = visualizer.visualize_stage('post_identity_crystallization', late_soul_data, save_plots=True, show_plots=False)
        print(f"Visualization result: {result}")
        
        if result.get('success'):
            print("✅ Late stage visualization PASSED")
        else:
            print(f"❌ Late stage visualization FAILED: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Visualization test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualizer_with_early_stage()
    test_visualizer_with_late_stage()
    print("\nVisualization fix test completed!")