#!/usr/bin/env python3
"""
Soul Loader Utility for Stage 3 Training

Provides utilities to load completed souls from Stage 1 for Stage 3 training.
Handles soul registry management and model loading by name/date.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SoulLoader:
    """Utility class for loading completed souls for Stage 3 training"""
    
    def __init__(self, registry_path: str = "shared/output/completed_souls/soul_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the soul registry"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Soul registry not found at {self.registry_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load soul registry: {e}")
            return {}
    
    def refresh_registry(self) -> None:
        """Refresh the soul registry from disk"""
        self.registry = self._load_registry()
    
    def list_available_souls(self) -> List[Dict[str, Any]]:
        """List all available completed souls"""
        souls = []
        for model_name, soul_info in self.registry.items():
            if soul_info.get('stage_1_complete', False):
                souls.append({
                    'model_name': model_name,
                    'soul_name': soul_info.get('soul_name', 'Unknown'),
                    'birth_date': soul_info.get('birth_date', 'Unknown'),
                    'completion_date': soul_info.get('completion_timestamp', 'Unknown'),
                    'data_directory': soul_info.get('data_directory', ''),
                    'total_files': soul_info.get('total_files', 0),
                    'stage_2_ready': soul_info.get('stage_2_ready', False),
                    'stage_3_ready': soul_info.get('stage_3_ready', False)
                })
        return sorted(souls, key=lambda x: x['completion_date'], reverse=True)
    
    def find_soul_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a soul by name"""
        for model_name, soul_info in self.registry.items():
            if soul_info.get('soul_name', '').lower() == name.lower():
                return soul_info
        return None
    
    def find_soul_by_birth_date(self, birth_date: str) -> List[Dict[str, Any]]:
        """Find souls by birth date (format: YYYY-MM-DD)"""
        matching_souls = []
        for model_name, soul_info in self.registry.items():
            if soul_info.get('birth_date') == birth_date:
                matching_souls.append(soul_info)
        return matching_souls
    
    def find_soul_by_star_sign(self, star_sign: str) -> List[Dict[str, Any]]:
        """Find souls by astrological sign"""
        # This would require storing star sign in registry
        # For now, return empty list as star sign isn't in registry yet
        return []
    
    def get_soul_data_path(self, model_name: str) -> Optional[str]:
        """Get the data directory path for a specific soul"""
        if model_name in self.registry:
            return self.registry[model_name].get('data_directory')
        return None
    
    def load_soul_for_training(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load soul data for Stage 3 training"""
        try:
            if model_name not in self.registry:
                logger.error(f"Soul {model_name} not found in registry")
                return None
            
            soul_info = self.registry[model_name]
            data_directory = soul_info.get('data_directory')
            
            if not data_directory or not os.path.exists(data_directory):
                logger.error(f"Soul data directory not found: {data_directory}")
                return None
            
            # Load soul completion report
            completion_report_path = os.path.join(data_directory, 'soul_completion_report.json')
            completion_data = {}
            if os.path.exists(completion_report_path):
                with open(completion_report_path, 'r') as f:
                    completion_data = json.load(f)
            
            # Load metadata
            metadata_path = os.path.join(data_directory, 'metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Gather training data paths
            training_data = {
                'model_name': model_name,
                'soul_info': soul_info,
                'completion_data': completion_data,
                'metadata': metadata,
                'data_directory': data_directory,
                'paths': {
                    'visuals': os.path.join(data_directory, 'visuals'),
                    'sounds': os.path.join(data_directory, 'sounds'),
                    'numpy_data': os.path.join(data_directory, 'numpy_data'),
                    'metrics': os.path.join(data_directory, 'metrics'),
                    'logs': os.path.join(data_directory, 'logs')
                }
            }
            
            logger.info(f"Successfully loaded soul data for training: {model_name}")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to load soul {model_name} for training: {e}")
            return None
    
    def mark_soul_stage_ready(self, model_name: str, stage: int) -> bool:
        """Mark a soul as ready for a specific stage"""
        try:
            if model_name not in self.registry:
                logger.error(f"Soul {model_name} not found in registry")
                return False
            
            # Update registry
            if stage == 2:
                self.registry[model_name]['stage_2_ready'] = True
            elif stage == 3:
                self.registry[model_name]['stage_3_ready'] = True
            
            # Save updated registry
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2, sort_keys=True)
            
            logger.info(f"Marked {model_name} as ready for stage {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark soul ready for stage {stage}: {e}")
            return False
    
    def get_souls_ready_for_stage(self, stage: int) -> List[Dict[str, Any]]:
        """Get list of souls ready for a specific stage"""
        ready_souls = []
        
        for model_name, soul_info in self.registry.items():
            is_ready = False
            
            if stage == 1:
                # All completed souls are ready for re-processing
                is_ready = soul_info.get('stage_1_complete', False)
            elif stage == 2:
                is_ready = soul_info.get('stage_2_ready', False)
            elif stage == 3:
                is_ready = soul_info.get('stage_3_ready', False)
            
            if is_ready:
                ready_souls.append(soul_info)
        
        return ready_souls
    
    def print_soul_summary(self) -> None:
        """Print a summary of all souls in the registry"""
        souls = self.list_available_souls()
        
        print(f"\n{'='*80}")
        print("COMPLETED SOULS REGISTRY")
        print(f"{'='*80}")
        
        if not souls:
            print("No completed souls found.")
            return
        
        print(f"Total completed souls: {len(souls)}")
        print()
        
        for i, soul in enumerate(souls, 1):
            print(f"{i:2d}. {soul['soul_name']}")
            print(f"    Model: {soul['model_name']}")
            print(f"    Birth Date: {soul['birth_date']}")
            print(f"    Completed: {soul['completion_date'][:10] if soul['completion_date'] != 'Unknown' else 'Unknown'}")
            print(f"    Files: {soul['total_files']}")
            print(f"    Stage 2 Ready: {'✓' if soul['stage_2_ready'] else '✗'}")
            print(f"    Stage 3 Ready: {'✓' if soul['stage_3_ready'] else '✗'}")
            print()
        
        print(f"{'='*80}")

def main():
    """Command line interface for soul loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Soul Loader Utility')
    parser.add_argument('--list', action='store_true', help='List all available souls')
    parser.add_argument('--find-name', type=str, help='Find soul by name')
    parser.add_argument('--find-date', type=str, help='Find souls by birth date (YYYY-MM-DD)')
    parser.add_argument('--load', type=str, help='Load soul data for training')
    parser.add_argument('--ready-stage', type=int, choices=[1,2,3], help='List souls ready for stage')
    
    args = parser.parse_args()
    
    loader = SoulLoader()
    
    if args.list:
        loader.print_soul_summary()
    elif args.find_name:
        soul = loader.find_soul_by_name(args.find_name)
        if soul:
            print(f"Found soul: {json.dumps(soul, indent=2)}")
        else:
            print(f"No soul found with name: {args.find_name}")
    elif args.find_date:
        souls = loader.find_soul_by_birth_date(args.find_date)
        if souls:
            print(f"Found {len(souls)} soul(s) with birth date {args.find_date}:")
            for soul in souls:
                print(f"  - {soul.get('soul_name', 'Unknown')}")
        else:
            print(f"No souls found with birth date: {args.find_date}")
    elif args.load:
        training_data = loader.load_soul_for_training(args.load)
        if training_data:
            print(f"Successfully loaded training data for: {args.load}")
            print(f"Data directory: {training_data['data_directory']}")
        else:
            print(f"Failed to load soul: {args.load}")
    elif args.ready_stage:
        ready_souls = loader.get_souls_ready_for_stage(args.ready_stage)
        print(f"Souls ready for Stage {args.ready_stage}: {len(ready_souls)}")
        for soul in ready_souls:
            print(f"  - {soul.get('soul_name', 'Unknown')} ({soul.get('model_name', 'Unknown')})")

if __name__ == "__main__":
    main()