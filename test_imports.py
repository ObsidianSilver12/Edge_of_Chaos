#!/usr/bin/env python3
"""
Quick test to verify imports work properly
"""

try:
    # Test constants import
    from constants.constants import *
    print(f"✓ Constants loaded successfully")
    print(f"  MIN_BRAIN_SEED_ENERGY = {MIN_BRAIN_SEED_ENERGY}")
    print(f"  DEFAULT_BRAIN_SEED_FREQUENCY = {DEFAULT_BRAIN_SEED_FREQUENCY}")
    print(f"  SEED_FIELD_RADIUS = {SEED_FIELD_RADIUS}")
    
    # Test brain_seed import
    import sys
    sys.path.append('.')
    from stage_1.brain_formation.brain.brain_seed import BrainSeed, create_brain_seed
    print(f"✓ BrainSeed module imported successfully")
    
    # Test basic functionality
    seed = create_brain_seed(initial_beu=10.0, initial_frequency=8.0)
    print(f"✓ BrainSeed created: {seed}")
    
    print("\n=== IMPORTS SUCCESSFUL ===")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
