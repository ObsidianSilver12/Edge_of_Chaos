#!/usr/bin/env python3
"""
Debug script to identify the real visualization issue
Tests each component separately to isolate the problem
"""

import sys
import os
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VisualizationDebug')

def test_imports():
    """Test all imports step by step"""
    print("=== TESTING IMPORTS ===")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except Exception as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except Exception as e:
        print(f"✗ Matplotlib failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")
        
        # Test basic plotly functionality
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
        print("✓ Plotly Figure creation works")
        
    except Exception as e:
        print(f"✗ Plotly failed: {e}")
        return False
    
    try:
        import pyvista as pv
        print("✓ PyVista imported successfully")
        
        # Test basic pyvista functionality
        sphere = pv.Sphere()
        print(f"✓ PyVista Sphere creation works: {type(sphere)}")
        
    except Exception as e:
        print(f"✗ PyVista failed: {e}")
        print("  This might be the issue - PyVista often has VTK/display problems")
        return False
    
    return True

def test_soul_visualizer_import():
    """Test SoulVisualizer import specifically"""
    print("\n=== TESTING SOUL VISUALIZER IMPORT ===")
    
    try:
        sys.path.append('.')
        from stage_1.soul_formation.soul_visualizer import SoulVisualizer
        print("✓ SoulVisualizer imported successfully")
        
        # Test initialization
        viz = SoulVisualizer("debug_test_output")
        print("✓ SoulVisualizer initialized successfully")
        
        return viz
        
    except Exception as e:
        print(f"✗ SoulVisualizer failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def test_simple_visualization(visualizer):
    """Test the simplest possible visualization"""
    print("\n=== TESTING SIMPLE VISUALIZATION ===")
    
    if not visualizer:
        print("✗ No visualizer available")
        return False
    
    try:
        # Test date/star sign fix (should work)
        print("1. Testing date/star sign fix...")
        fixed_date, fixed_sign = visualizer.fix_date_star_sign_mismatch("2024-05-11", "Sagittarius")
        print(f"   Result: {fixed_date} -> {fixed_sign}")
        
        if fixed_sign == "Taurus":
            print("   ✓ Date/star sign fix working")
        else:
            print(f"   ✗ Expected Taurus, got {fixed_sign}")
        
        # Test color evolution (Plotly only)
        print("2. Testing color evolution visualization...")
        test_data = {
            'name': 'DebugSoul',
            'date_of_birth': '2024-05-11',
            'star_sign': 'Sagittarius',
            'energy_level': 85
        }
        
        color_fig = visualizer.create_color_evolution_visualization(test_data)
        if color_fig:
            color_path = "debug_color_test.html"
            color_fig.write_html(color_path)
            print(f"   ✓ Color evolution saved: {color_path}")
        else:
            print("   ✗ Color evolution failed")
        
        # Test 3D mesh generation (might fail)
        print("3. Testing 3D mesh generation...")
        try:
            mesh = visualizer.generate_3d_soul_mesh('soul_spark', 0.5)
            if mesh is not None:
                print("   ✓ 3D mesh generation working")
            else:
                print("   ✗ 3D mesh generation returned None")
        except Exception as mesh_e:
            print(f"   ✗ 3D mesh generation failed: {mesh_e}")
        
        # Test frequency spectrum
        print("4. Testing frequency spectrum...")
        try:
            freq_fig = visualizer.create_frequency_spectrum_3d('soul_spark', test_data)
            if freq_fig is not None:
                freq_path = "debug_frequency_test.html"
                freq_fig.write_html(freq_path)
                print(f"   ✓ Frequency spectrum saved: {freq_path}")
            else:
                print("   ✗ Frequency spectrum returned None")
        except Exception as freq_e:
            print(f"   ✗ Frequency spectrum failed: {freq_e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple visualization test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main debug function"""
    print("SOUL VISUALIZER DEBUG ANALYSIS")
    print("=" * 50)
    
    # Test imports first
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ ISSUE FOUND: Import problems detected")
        print("The visualization system cannot work without all dependencies")
        return False
    
    # Test SoulVisualizer specifically
    visualizer = test_soul_visualizer_import()
    
    if not visualizer:
        print("\n❌ ISSUE FOUND: SoulVisualizer import/initialization failed")
        return False
    
    # Test simple visualization functions
    viz_ok = test_simple_visualization(visualizer)
    
    if not viz_ok:
        print("\n❌ ISSUE FOUND: Visualization functions not working")
        return False
    
    print("\n✅ ALL TESTS PASSED")
    print("The visualization system appears to be working correctly.")
    print("The issue might be:")
    print("1. Logging level not showing debug messages")
    print("2. Visualization calls not being made during simulation")
    print("3. Output directory permissions")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nRun this debug script to identify the specific issue.")
    sys.exit(0 if success else 1)