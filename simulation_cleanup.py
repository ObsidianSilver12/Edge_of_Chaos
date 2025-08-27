#!/usr/bin/env python3
"""
Simulation Cleanup and Data Organization Script

Handles:
1. Moving completed soul data to proper organized structure
2. Cleaning up temporary files and __pycache__ 
3. Organizing logs, metrics, visuals, and sounds by simulation run
4. Preparing for next simulation run

Usage: python simulation_cleanup.py [--model-name MODEL] [--dry-run]
"""

import os
import sys
import shutil
import glob
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simulation_cleanup')

def get_model_name() -> tuple:
    """Get model name from command line or generate default."""
    parser = argparse.ArgumentParser(description='Cleanup and organize simulation data')
    parser.add_argument('--model-name', type=str, help='Name of the soul model/simulation')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    parser.add_argument('--failed', action='store_true', help='Clean up after a failed simulation (delete files instead of organizing)')
    args = parser.parse_args()
    
    if args.model_name:
        return args.model_name, args.dry_run, args.failed
    
    # Generate default name based on timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"soul_simulation_{timestamp}", args.dry_run, args.failed

def find_soul_files(base_dir: str = "output") -> Dict[str, List[str]]:
    """Find all soul-related files to organize."""
    files_by_type = {
        'visuals': [],
        'sounds': [],
        'logs': [],
        'metrics': [],
        'numpy_data': [],
        'other': []
    }
    
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory {base_dir} does not exist")
        return files_by_type
    
    # Search for files in output directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, base_dir)
            
            # Categorize files
            if file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html')):
                files_by_type['visuals'].append(full_path)
            elif file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                files_by_type['sounds'].append(full_path)
            elif file.endswith('.log') or 'log' in file.lower():
                files_by_type['logs'].append(full_path)
            elif file.endswith('.json') and ('metric' in file.lower() or 'soul' in file.lower() or 'numpy_metadata' in file.lower()):
                files_by_type['metrics'].append(full_path)
            elif file.endswith(('.npy', '.npz', '.pkl')):
                files_by_type['numpy_data'].append(full_path)
            else:
                files_by_type['other'].append(full_path)
    
    return files_by_type

def cleanup_failed_simulation_data(model_name: str, dry_run: bool = False) -> Dict[str, int]:
    """Clean up data from failed simulation by deleting output files and clearing cache."""
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting cleanup of failed simulation: {model_name}")
    
    cleanup_stats = {
        'visuals_deleted': 0,
        'sounds_deleted': 0,
        'logs_deleted': 0,
        'metrics_deleted': 0,
        'numpy_data_deleted': 0,
        'pycache_removed': 0,
        'temp_files_removed': 0,
        'empty_dirs_removed': 0
    }
    
    # Find and delete all output files
    files_by_type = find_soul_files()
    
    for file_type, file_list in files_by_type.items():
        if file_type == 'other':
            continue
            
        for file_path in file_list:
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Deleting failed simulation file: {file_path}")
            
            if not dry_run:
                try:
                    os.remove(file_path)
                    cleanup_stats[f'{file_type}_deleted'] += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            else:
                cleanup_stats[f'{file_type}_deleted'] += 1
    
    # Clean up __pycache__ directories
    cleanup_stats['pycache_removed'] = cleanup_pycache(dry_run)
    
    # Clean up temporary files
    cleanup_stats['temp_files_removed'] = cleanup_temp_files(dry_run)
    
    # Clean up empty directories
    cleanup_stats['empty_dirs_removed'] = cleanup_empty_directories(dry_run=dry_run)
    
    # Create failure summary
    failure_summary = {
        'model_name': model_name,
        'failure_timestamp': datetime.now().isoformat(),
        'cleanup_stats': cleanup_stats,
        'status': 'failed_and_cleaned'
    }
    
    # Save failure summary if not dry run
    if not dry_run:
        failure_log_path = f"failure_cleanup_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(failure_log_path, 'w') as f:
                json.dump(failure_summary, f, indent=2)
            logger.info(f"Failure cleanup summary saved: {failure_log_path}")
        except Exception as e:
            logger.error(f"Could not save failure summary: {e}")
    
    total_deleted = sum(cleanup_stats.values())
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Failed simulation cleanup completed: {total_deleted} items processed")
    
    return cleanup_stats

def organize_completed_soul_data(model_name: str, dry_run: bool = False) -> str:
    """Organize all soul data into proper completed_souls structure."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    completed_dir = f"shared/output/completed_souls/{model_name}_{timestamp}"
    
    # Create directory structure
    subdirs = ['visuals', 'sounds', 'numpy_data', 'logs', 'metrics']
    
    if not dry_run:
        os.makedirs(completed_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(completed_dir, subdir), exist_ok=True)
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Creating completed soul directory: {completed_dir}")
    
    # Find and organize files
    files_by_type = find_soul_files()
    
    for file_type, file_list in files_by_type.items():
        if file_type == 'other':
            continue
            
        target_subdir = os.path.join(completed_dir, file_type)
        
        for file_path in file_list:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_subdir, filename)
            
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Moving {file_path} -> {target_path}")
            
            if not dry_run:
                try:
                    # Create subdirectories if they don't exist
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Move the file
                    shutil.move(file_path, target_path)
                except Exception as e:
                    logger.error(f"Failed to move {file_path}: {e}")
    
    # Create metadata file
    metadata = {
        'model_name': model_name,
        'completion_timestamp': datetime.now().isoformat(),
        'files_organized': {k: len(v) for k, v in files_by_type.items()},
        'total_files': sum(len(v) for v in files_by_type.values())
    }
    
    metadata_path = os.path.join(completed_dir, 'metadata.json')
    if not dry_run:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Created metadata file: {metadata_path}")
    
    # Update soul registry for Stage 3 model loading
    if not dry_run:
        _update_soul_registry(model_name, completed_dir, metadata)
    
    return completed_dir

def _update_soul_registry(model_name: str, completed_dir: str, metadata: Dict[str, Any]) -> None:
    """Update the soul registry for Stage 3 model loading"""
    try:
        registry_path = "shared/output/completed_souls/soul_registry.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Load existing registry or create new one
        registry = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing soul registry: {e}")
                registry = {}
        
        # Parse soul information from model name
        # Expected format: SoulName_YYYYMMDD or SoulName_simID
        soul_info = {
            'model_name': model_name,
            'completion_timestamp': metadata['completion_timestamp'],
            'data_directory': completed_dir,
            'total_files': metadata['total_files'],
            'stage_1_complete': True,
            'stage_2_ready': False,
            'stage_3_ready': False
        }
        
        # Try to extract name and birth date
        if '_' in model_name:
            parts = model_name.split('_', 1)
            soul_info['soul_name'] = parts[0]
            date_or_id = parts[1]
            
            # Check if the second part looks like a date (8 digits)
            if len(date_or_id) == 8 and date_or_id.isdigit():
                try:
                    # Convert YYYYMMDD to YYYY-MM-DD
                    formatted_date = f"{date_or_id[:4]}-{date_or_id[4:6]}-{date_or_id[6:8]}"
                    soul_info['birth_date'] = formatted_date
                except:
                    soul_info['simulation_id'] = date_or_id
            else:
                soul_info['simulation_id'] = date_or_id
        else:
            soul_info['soul_name'] = model_name
        
        # Add to registry
        registry[model_name] = soul_info
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2, sort_keys=True)
        
        logger.info(f"Soul registry updated: {model_name} added to {registry_path}")
        logger.info(f"Total souls in registry: {len(registry)}")
        
    except Exception as e:
        logger.error(f"Failed to update soul registry: {e}")

def cleanup_pycache(dry_run: bool = False) -> int:
    """Remove all __pycache__ directories recursively."""
    removed_count = 0
    
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Removing {pycache_path}")
            
            if not dry_run:
                try:
                    shutil.rmtree(pycache_path)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {pycache_path}: {e}")
            else:
                removed_count += 1
    
    return removed_count

def cleanup_temp_files(dry_run: bool = False) -> int:
    """Remove temporary files (.tmp, .temp, etc.) while preserving simulation_cleanup.py."""
    removed_count = 0
    temp_patterns = ['*.tmp', '*.temp', '*.log.*', '.DS_Store', 'Thumbs.db']
    
    # Files to always preserve
    preserve_files = {
        'simulation_cleanup.py',
        'requirements.txt',
        'README.md'
    }
    
    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern, recursive=True):
            # Skip preserved files
            filename = os.path.basename(temp_file)
            if filename in preserve_files:
                logger.info(f"Preserving essential file: {temp_file}")
                continue
                
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Removing temporary file: {temp_file}")
            
            if not dry_run:
                try:
                    os.remove(temp_file)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {temp_file}: {e}")
            else:
                removed_count += 1
    
    return removed_count

def cleanup_empty_directories(base_dir: str = "output", dry_run: bool = False) -> int:
    """Remove empty directories in output folder."""
    removed_count = 0
    
    if not os.path.exists(base_dir):
        return removed_count
    
    # Walk bottom-up to remove nested empty directories
    for root, dirs, files in os.walk(base_dir, topdown=False):
        if not dirs and not files:  # Directory is empty
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Removing empty directory: {root}")
            
            if not dry_run:
                try:
                    os.rmdir(root)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {root}: {e}")
            else:
                removed_count += 1
    
    return removed_count

def main():
    """Main cleanup function."""
    try:
        model_name, dry_run, is_failed = get_model_name()
        
        if is_failed:
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting FAILED simulation cleanup for model: {model_name}")
            
            # Clean up failed simulation by deleting files
            cleanup_stats = cleanup_failed_simulation_data(model_name, dry_run)
            
            # Summary for failed cleanup
            total_deleted = sum(cleanup_stats.values())
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Failed simulation cleanup completed!")
            logger.info(f"Total items deleted/cleaned: {total_deleted}")
            logger.info("System cleared and ready for next simulation")
            
            if dry_run:
                print(f"\n[DRY RUN] Failed simulation cleanup summary:")
                for key, value in cleanup_stats.items():
                    print(f"  {key}: {value}")
                print("\nThis was a dry run. To actually perform the cleanup, run without --dry-run flag.")
            else:
                print(f"\n{'='*80}")
                print("FAILED SIMULATION CLEANUP COMPLETED")
                print(f"{'='*80}")
                print(f"Model: {model_name}")
                print(f"Total items cleaned: {total_deleted}")
                print("All output files deleted, cache cleared")
                print("System ready for next simulation attempt")
                print(f"{'='*80}")
        
        else:
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting successful simulation cleanup for model: {model_name}")
            
            # 1. Organize completed soul data
            completed_dir = organize_completed_soul_data(model_name, dry_run)
            
            # 2. Clean up __pycache__ directories
            pycache_removed = cleanup_pycache(dry_run)
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Removed {pycache_removed} __pycache__ directories")
            
            # 3. Clean up temporary files
            temp_removed = cleanup_temp_files(dry_run)
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Removed {temp_removed} temporary files")
            
            # 4. Clean up empty directories
            empty_removed = cleanup_empty_directories(dry_run=dry_run)
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Removed {empty_removed} empty directories")
            
            # Summary for successful cleanup
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Successful simulation cleanup completed!")
            logger.info(f"Soul data organized in: {completed_dir}")
            logger.info(f"System ready for next simulation")
            
            if dry_run:
                print("\nThis was a dry run. To actually perform the cleanup, run without --dry-run flag.")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()