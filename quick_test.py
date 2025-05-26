#!/usr/bin/env python3
"""
Quick test to verify core imports work
"""

def test_imports():
    print("Testing core imports...")
    
    try:
        # Test constants
        print("1. Testing constants...")
        from constants.constants import MIN_BRAIN_SEED_ENERGY, DEFAULT_BRAIN_SEED_FREQUENCY
        print(f"   ✓ Constants loaded: {MIN_BRAIN_SEED_ENERGY}, {DEFAULT_BRAIN_SEED_FREQUENCY}")
        
        # Test brain seed
        print("2. Testing brain seed import...")
        from stage_1.evolve.brain_structure.brain_seed import create_brain_seed
        print("   ✓ create_brain_seed imported successfully")
        
        # Test birth module import
        print("3. Testing birth module import...")
        from stage_1.soul_formation.birth import perform_birth
        print("   ✓ perform_birth imported successfully")
        
        # Test field controller
        print("4. Testing field controller...")
        from stage_1.fields.field_controller import FieldController
        print("   ✓ FieldController imported successfully")
        
        # Test mycelial network controller
        print("5. Testing mycelial network controller...")
        from system.mycelial_network.mycelial_network_controller import MycelialNetworkController
        print("   ✓ MycelialNetworkController imported successfully")
        
        # Test soul completion controller
        print("6. Testing soul completion controller...")
        from stage_1.soul_formation.soul_completion_controller import SoulCompletionController
        print("   ✓ SoulCompletionController imported successfully")
        
        print("\n=== ALL IMPORTS SUCCESSFUL ===")
        return True
        
    except Exception as e:
        print(f"\n✗ IMPORT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
