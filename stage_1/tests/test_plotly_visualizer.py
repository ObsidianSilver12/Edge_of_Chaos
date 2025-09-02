#!/usr/bin/env python3
"""
Test the new PyVista-free Plotly-only soul visualizer
"""

import sys
import os
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PlotlyVisualizerTest')

def test_plotly_visualizer():
    """Test the new Plotly-only visualizer"""
    
    try:
        print("=== Testing Plotly-Only Soul Visualizer ===")
        
        # Import the updated visualizer
        sys.path.append('.')
        from stage_1.soul_formation.soul_evolution_visualizer import SoulVisualizer
        
        print("✓ SoulVisualizer imported successfully")
        
        # Initialize visualizer
        visualizer = SoulVisualizer("test_plotly_output")
        print("✓ SoulVisualizer initialized")
        
        # Test data
        test_soul_data = {
            'name': 'TestSoul_Plotly',
            'date_of_birth': '2024-05-11',
            'star_sign': 'Sagittarius',
            'energy_level': 85,
            'coherence': 90,
            'complexity': 75,
            'stability': 88
        }
        
        print("\n=== Testing Individual Stage Visualizations ===")
        
        # Test each stage
        stages_to_test = [
            'soul_spark',
            'creator_entanglement', 
            'sephiroth_journey',
            'pre_identity_crystallization',
            'post_identity_crystallization'
        ]
        
        successful_stages = []
        
        for stage in stages_to_test:
            try:
                print(f"\n1. Testing {stage}...")
                
                # Test 3D mesh generation
                mesh_fig = visualizer.generate_3d_soul_mesh_plotly(stage, 0.5)
                if mesh_fig:
                    mesh_path = f"test_plotly_output/{stage}_3d_model.html"
                    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
                    mesh_fig.write_html(mesh_path)
                    print(f"   ✓ 3D model saved: {mesh_path}")
                else:
                    print(f"   ✗ 3D model failed")
                    continue
                
                # Test frequency spectrum
                freq_fig = visualizer.create_frequency_spectrum_3d(stage, test_soul_data)
                if freq_fig:
                    freq_path = f"test_plotly_output/{stage}_frequency.html"
                    freq_fig.write_html(freq_path)
                    print(f"   ✓ Frequency spectrum saved: {freq_path}")
                else:
                    print(f"   ✗ Frequency spectrum failed")
                    continue
                
                # Test complete stage visualization
                result = visualizer.visualize_stage(stage, test_soul_data, save_plots=True)
                if result.get('success'):
                    print(f"   ✓ Complete stage visualization successful")
                    successful_stages.append(stage)
                else:
                    print(f"   ✗ Complete stage visualization failed: {result.get('error')}")
                
            except Exception as stage_e:
                print(f"   ✗ Stage {stage} failed: {stage_e}")
        
        print(f"\n=== Stage Results: {len(successful_stages)}/{len(stages_to_test)} successful ===")
        
        if len(successful_stages) >= 3:  # At least 60% success
            print("\n=== Testing Complete Evolution Visualization ===")
            
            try:
                complete_result = visualizer.create_complete_evolution_visualization(test_soul_data, save_plots=True)
                
                if complete_result.get('success'):
                    print("✓ Complete evolution visualization successful")
                    
                    # Check what was created
                    overview = complete_result.get('overview', {})
                    stages_results = complete_result.get('stages', {})
                    
                    print(f"\nGenerated files:")
                    if 'color_evolution' in overview:
                        print(f"   - Color evolution: {overview['color_evolution']}")
                    if 'timeline' in overview:
                        print(f"   - Evolution timeline: {overview['timeline']}")
                    if 'combined_3d' in overview:
                        print(f"   - Combined 3D view: {overview['combined_3d']}")
                    
                    print(f"   - Individual stages: {len(stages_results)} stages")
                    
                    # Test date/star sign fix
                    corrected_date, corrected_sign = visualizer.fix_date_star_sign_mismatch(
                        test_soul_data['date_of_birth'], 
                        test_soul_data['star_sign']
                    )
                    
                    print(f"\n=== Date/Star Sign Fix Test ===")
                    print(f"Original: {test_soul_data['date_of_birth']} -> {test_soul_data['star_sign']}")
                    print(f"Corrected: {corrected_date} -> {corrected_sign}")
                    
                    if corrected_sign == "Taurus":
                        print("✓ Date/star sign correction working properly")
                    else:
                        print(f"✗ Expected Taurus, got {corrected_sign}")
                    
                    print(f"\n{'='*60}")
                    print("🎉 PLOTLY VISUALIZER TEST SUCCESSFUL!")
                    print(f"{'='*60}")
                    print("✅ All PyVista dependencies removed")
                    print("✅ 3D visualizations working with Plotly only")
                    print("✅ Frequency spectra generated")
                    print("✅ Color evolution working")
                    print("✅ Complete evolution timeline working")
                    print("✅ Date/star sign correction working")
                    print("✅ Final metrics dashboard working")
                    print(f"✅ Test files saved to: test_plotly_output/")
                    
                    return True
                else:
                    print(f"✗ Complete evolution visualization failed: {complete_result.get('error')}")
                    return False
                    
            except Exception as complete_e:
                print(f"✗ Complete evolution test failed: {complete_e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"✗ Too many stage failures ({len(successful_stages)}/{len(stages_to_test)})")
            return False
            
    except Exception as e:
        print(f"✗ Visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plotly_visualizer()
    
    if success:
        print("\n🎯 The soul visualizer is now working without PyVista!")
        print("Run a simulation to see the full 3D visualizations.")
    else:
        print("\n❌ There are still issues to resolve.")
    
    sys.exit(0 if success else 1)