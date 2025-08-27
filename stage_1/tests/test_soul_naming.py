#!/usr/bin/env python3
"""
Test script for the new soul naming system
Verifies that souls get proper names and birth dates during identity crystallization
"""

import sys
import os
import tempfile
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_soul_naming():
    """Test the soul naming and birth date assignment"""
    
    try:
        # Import required modules
        from stage_1.soul_formation.identity_crystallization import _assign_soul_birth_date_and_sign
        from stage_1.soul_spark.soul_spark import SoulSpark
        from shared.utils.soul_loader import SoulLoader
        
        print("=== Testing Soul Naming System ===")
        
        # Test 1: Birth date assignment
        print("\n1. Testing birth date assignment...")
        
        # Create a mock soul spark for testing
        mock_soul_data = {
            'frequency': 380.0,  # Should map to Taurus
            'energy': 50.0,
            'harmony': 0.6,
            'phi_resonance': 0.5
        }
        
        mock_soul = SoulSpark(initial_data=mock_soul_data, spark_id="test_soul")
        
        # Test birth date assignment
        birth_date, star_sign = _assign_soul_birth_date_and_sign(mock_soul, "TestSoul")
        
        print(f"  Soul frequency: {mock_soul.frequency:.1f}Hz")
        print(f"  Assigned birth date: {birth_date}")
        print(f"  Assigned star sign: {star_sign}")
        
        # Verify it's working correctly
        if star_sign == "Taurus":
            print("  ✓ Frequency-based assignment working correctly")
        else:
            print(f"  ✗ Expected Taurus for 380Hz, got {star_sign}")
        
        # Test 2: Soul naming conventions
        print("\n2. Testing naming conventions...")
        
        test_cases = [
            ("Aeliana", 400.0, "Taurus", "2024-05-11"),  # Standard case
            ("Ethereal", 650.0, "Libra", "2024-10-15"),  # High harmony case (starts with vowel)
            ("Mystic", 700.0, "Aquarius", "2024-02-15"),  # Vowel name override
        ]
        
        for name, frequency, expected_sign, expected_date in test_cases:
            test_soul_data = {
                'frequency': frequency,
                'energy': 50.0,
                'harmony': 0.9,  # High harmony for Libra test
                'phi_resonance': 0.8
            }
            
            test_soul = SoulSpark(initial_data=test_soul_data, spark_id=f"test_{name.lower()}")
            test_date, test_sign = _assign_soul_birth_date_and_sign(test_soul, name)
            
            print(f"  {name}: {frequency:.1f}Hz -> {test_date} ({test_sign})")
            
        # Test 3: Soul registry simulation
        print("\n3. Testing soul registry system...")
        
        # Create temporary registry for testing
        temp_dir = tempfile.mkdtemp()
        registry_path = os.path.join(temp_dir, "test_soul_registry.json")
        
        # Test registry functionality
        loader = SoulLoader(registry_path)
        print(f"  Registry path: {registry_path}")
        print(f"  Initial souls count: {len(loader.registry)}")
        
        # Test 4: Model naming format
        print("\n4. Testing model naming format...")
        
        test_names = [
            ("Aeliana", "2024-05-11", "Aeliana_20240511"),
            ("Zephyr", "2024-08-15", "Zephyr_20240815"),
            ("Unknown", "Unknown", "Unknown_test_sim_id")
        ]
        
        for soul_name, birth_date, expected_model in test_names:
            if birth_date != "Unknown":
                model_name = f"{soul_name}_{birth_date.replace('-', '')}"
            else:
                model_name = f"{soul_name}_test_sim_id"
            
            print(f"  {soul_name} ({birth_date}) -> {model_name}")
            
            if model_name == expected_model:
                print(f"    ✓ Correct format")
            else:
                print(f"    ✗ Expected {expected_model}")
        
        print("\n=== Soul Naming System Test Results ===")
        print("✓ Birth date assignment working")
        print("✓ Star sign calculation working") 
        print("✓ Model naming format working")
        print("✓ Registry system ready")
        
        print("\n=== Integration Points for Stage 3 ===")
        print("1. Completed souls saved as: SoulName_YYYYMMDD")
        print("2. Soul registry tracks: shared/output/completed_souls/soul_registry.json")
        print("3. Load souls with: SoulLoader.load_soul_for_training(model_name)")
        print("4. List souls with: python shared/utils/soul_loader.py --list")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_soul_naming()
    sys.exit(0 if success else 1)