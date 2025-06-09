#!/usr/bin/env python3

"""
Quick test to verify brain structure fixes
"""

import sys
import os

# Add the project root to Python path
project_root = r"C:\Kim\Claude\Edge_of_Chaos"
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    print("Testing imports...")
    
    # Test constants import
    from constants.constants import *
    print("✅ Constants imported successfully")
    
    # Test region proportions sum
    total_proportion = sum(REGION_PROPORTIONS.values())
    print(f"📊 Region proportions sum: {total_proportion:.6f}")
    if abs(total_proportion - 1.0) < 0.001:
        print("✅ Region proportions sum correctly to 1.0")
    else:
        print(f"❌ Region proportions sum to {total_proportion}, not 1.0")
    
    # Test brain structure import
    sys.path.append(os.path.join(project_root, 'stage_1', 'brain_formation', 'brain'))
    from brain_structure import Brain
    print("✅ Brain structure imported successfully")
    
    # Test brain creation
    print("\nTesting brain creation...")
    brain = Brain()
    print(f"✅ Brain created with ID: {brain.brain_id[:8]}")
    
    # Test helper methods
    print("\nTesting helper methods...")
    
    hemisphere_props = brain.get_hemisphere_wave_properties()
    print(f"✅ Hemisphere properties: {len(hemisphere_props)} hemispheres")
    
    region_props = brain.get_region_wave_properties()
    print(f"✅ Region properties: {len(region_props)} regions")
    
    anatomical_map = brain.get_anatomical_position_mapping()
    print(f"✅ Anatomical mapping: {len(anatomical_map)} positions")
    
    # Test region configuration
    frontal_config = brain.get_region_configuration('frontal')
    print(f"✅ Frontal region config: {frontal_config['function']}")
    
    print("\n🎉 ALL BASIC TESTS PASSED!")
    
    # Now test brain development
    print("\nTesting brain development (this may take a moment)...")
    try:
        brain.trigger_brain_development()
        print("✅ Brain development completed successfully!")
        
        # Check final structure
        total_blocks = brain.brain['statistics']['total_blocks']
        hemispheres = brain.brain['statistics']['hemisphere_count']
        regions = brain.brain['statistics']['region_count']
        
        print(f"📊 Final brain statistics:")
        print(f"   - Hemispheres: {hemispheres}")
        print(f"   - Regions: {regions}")
        print(f"   - Total blocks: {total_blocks}")
        
        if total_blocks > 0:
            print("✅ Brain structure created successfully!")
        else:
            print("❌ No blocks created")
            
    except Exception as dev_error:
        print(f"❌ Brain development failed: {dev_error}")
        import traceback
        traceback.print_exc()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print(f"\nTest completed.")
