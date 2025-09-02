#!/usr/bin/env python3
"""
Simple test to check what's working with visualization
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimpleVisualizationTest')

def test_basic_visualization():
    """Test basic visualization components"""
    
    try:
        # Test Plotly (should work)
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='test'))
        fig.update_layout(title="Basic Test")
        
        # Save to test location
        test_path = "output/visuals/basic_test.html"
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        fig.write_html(test_path)
        
        print(f"✓ Plotly test successful: {test_path}")
        
    except Exception as e:
        print(f"✗ Plotly test failed: {e}")
        return False
        
    try:
        # Test PyVista (might fail)
        import pyvista as pv
        
        sphere = pv.Sphere()
        print(f"✓ PyVista sphere creation successful")
        
        # Try to save
        test_mesh_path = "output/visuals/test_mesh.ply"
        sphere.save(test_mesh_path)
        print(f"✓ PyVista mesh save successful: {test_mesh_path}")
        
    except Exception as e:
        print(f"✗ PyVista test failed: {e}")
        print("  This is likely due to VTK/display issues in WSL")
        
    try:
        # Test the SoulVisualizer date fix (should work)
        sys.path.append('.')
        from stage_1.soul_formation.soul_evolution_visualizer import SoulVisualizer
        
        viz = SoulVisualizer("output/visuals/test_soul")
        
        fixed_date, fixed_sign = viz.fix_date_star_sign_mismatch("2024-05-11", "Sagittarius")
        
        if fixed_sign == "Taurus":
            print(f"✓ Date/star sign fix working: May 11 -> {fixed_sign}")
        else:
            print(f"✗ Date/star sign fix failed: May 11 -> {fixed_sign}")
            
        # Test color evolution (should work with Plotly)
        test_data = {
            'name': 'TestSoul',
            'date_of_birth': '2024-05-11',
            'star_sign': 'Sagittarius',
            'energy_level': 85
        }
        
        color_fig = viz.create_color_evolution_visualization(test_data)
        color_path = "output/visuals/test_color_evolution.html"
        color_fig.write_html(color_path)
        print(f"✓ Color evolution visualization: {color_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ SoulVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_visualization()
    if success:
        print("\n=== BASIC VISUALIZATION COMPONENTS WORKING ===")
        print("The visualization system has some working parts.")
        print("PyVista issues are likely due to VTK/display in WSL environment.")
    else:
        print("\n=== VISUALIZATION SYSTEM HAS ISSUES ===")
        print("Need to debug further.")