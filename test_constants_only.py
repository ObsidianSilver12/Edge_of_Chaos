"""
Simple Python test script to verify constants are working
"""

def test_constants():
    try:
        # Test constants import
        from constants.constants import *
        print("✓ Constants imported successfully")
        
        # Test specific constants
        print(f"  MIN_BRAIN_SEED_ENERGY = {MIN_BRAIN_SEED_ENERGY}")
        print(f"  DEFAULT_BRAIN_SEED_FREQUENCY = {DEFAULT_BRAIN_SEED_FREQUENCY}")
        print(f"  SEED_FIELD_RADIUS = {SEED_FIELD_RADIUS}")
        print(f"  REGION_LIMBIC = {REGION_LIMBIC}")
        print(f"  REGION_TEMPORAL = {REGION_TEMPORAL}")
        
        return True
        
    except Exception as e:
        print(f"✗ Constants import failed: {e}")
        return False

if __name__ == "__main__":
    if test_constants():
        print("\n=== CONSTANTS TEST PASSED ===")
    else:
        print("\n=== CONSTANTS TEST FAILED ===")
