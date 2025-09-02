#!/usr/bin/env python3
"""
Test script to verify visualization dependencies
"""

import sys
import traceback

def test_imports():
    """Test all required imports for visualization"""
    results = {}
    
    # Test basic imports
    try:
        import numpy as np
        results['numpy'] = "✓ OK"
    except Exception as e:
        results['numpy'] = f"✗ FAILED: {e}"
    
    try:
        import matplotlib.pyplot as plt
        results['matplotlib'] = "✓ OK"
    except Exception as e:
        results['matplotlib'] = f"✗ FAILED: {e}"
    
    # Test 3D visualization libraries
    try:
        import plotly.graph_objects as go
        results['plotly'] = "✓ OK"
    except Exception as e:
        results['plotly'] = f"✗ FAILED: {e}"
    
    try:
        import pyvista as pv
        results['pyvista'] = "✓ OK"
    except Exception as e:
        results['pyvista'] = f"✗ FAILED: {e}"
    
    # Test SoulVisualizer import
    try:
        sys.path.append('.')
        from stage_1.soul_formation.soul_evolution_visualizer import SoulVisualizer
        results['SoulVisualizer'] = "✓ OK"
    except Exception as e:
        results['SoulVisualizer'] = f"✗ FAILED: {e}"
        traceback.print_exc()
    
    return results

def test_soul_visualizer():
    """Test SoulVisualizer functionality"""
    try:
        sys.path.append('.')
        from stage_1.soul_formation.soul_evolution_visualizer import SoulVisualizer
        
        # Try to create visualizer
        visualizer = SoulVisualizer("test_output")
        
        # Test sample data
        test_data = {
            'name': 'TestSoul',
            'date_of_birth': '2024-05-11',
            'star_sign': 'Sagittarius',
            'energy_level': 80
        }
        
        # Test date/star sign fix
        fixed_date, fixed_sign = visualizer.fix_date_star_sign_mismatch(
            test_data['date_of_birth'], 
            test_data['star_sign']
        )
        
        return f"✓ SoulVisualizer test passed - {fixed_date} -> {fixed_sign}"
        
    except Exception as e:
        traceback.print_exc()
        return f"✗ SoulVisualizer test failed: {e}"

if __name__ == "__main__":
    print("Testing visualization dependencies...")
    print("=" * 50)
    
    import_results = test_imports()
    for package, status in import_results.items():
        print(f"{package:15}: {status}")
    
    print("\n" + "=" * 50)
    print("Testing SoulVisualizer functionality...")
    print(test_soul_visualizer())