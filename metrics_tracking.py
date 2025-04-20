"""
Metrics Tracking Module

This module provides simplified metrics tracking functionality for the Soul Development Framework.
Rather than handling all detailed metrics (which are managed by individual components),
this module provides centralized recording of high-level metrics.

Author: Soul Development Framework Team
"""

import json
import os
import time
import logging
from typing import Dict, Any, List, Optional
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('metrics_tracking')

# Metrics storage
_metrics_store = {}
_metrics_history = {}
_session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_metrics_file_path = os.path.join("output", "metrics")

# Ensure metrics directory exists
os.makedirs(_metrics_file_path, exist_ok=True)

def record_metrics(category: str, metrics_data: Dict[str, Any]) -> bool:
    """
    Record metrics for a specific category.
    
    Args:
        category (str): Category of the metrics (e.g., 'soul_formation', 'harmonic_strengthening')
        metrics_data (dict): Dictionary of metrics to record
        
    Returns:
        bool: Success status
    """
    try:
        # Add timestamp
        metrics_with_time = metrics_data.copy()
        metrics_with_time['timestamp'] = time.time()
        
        # Store in the metrics store
        if category not in _metrics_store:
            _metrics_store[category] = []
        _metrics_store[category].append(metrics_with_time)
        
        # Store in history
        if category not in _metrics_history:
            _metrics_history[category] = []
        _metrics_history[category].append(metrics_with_time)
        
        # Log metrics recording
        logger.debug(f"Recorded metrics for {category}: {len(metrics_data)} values")
        
        return True
    except Exception as e:
        logger.error(f"Error recording metrics: {e}")
        return False

def get_latest_metrics(category: str) -> Dict[str, Any]:
    """
    Get the latest metrics for a specific category.
    
    Args:
        category (str): Category of the metrics
        
    Returns:
        dict: Latest metrics or empty dict if none
    """
    if category in _metrics_store and _metrics_store[category]:
        return _metrics_store[category][-1]
    return {}

def get_all_metrics(category: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all recorded metrics.
    
    Args:
        category (str, optional): Category to retrieve, or None for all categories
        
    Returns:
        dict: All metrics data
    """
    if category:
        return {category: _metrics_store.get(category, [])}
    return _metrics_store

def reset_metrics(category: str = None) -> bool:
    """
    Reset metrics for a category or all metrics.
    
    Args:
        category (str, optional): Category to reset, or None for all categories
        
    Returns:
        bool: Success status
    """
    global _metrics_store
    
    try:
        if category:
            if category in _metrics_store:
                _metrics_store[category] = []
        else:
            _metrics_store = {}
        
        logger.info(f"Reset metrics for {'all categories' if category is None else category}")
        return True
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        return False

def save_metrics_to_file(filename: str = None) -> bool:
    """
    Save all metrics to a JSON file.
    
    Args:
        filename (str, optional): Filename to save to, or None for auto-generated name
        
    Returns:
        bool: Success status
    """
    try:
        if filename is None:
            filename = f"metrics_{_session_id}.json"
        
        filepath = os.path.join(_metrics_file_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(_metrics_store, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving metrics to file: {e}")
        return False

def load_metrics_from_file(filename: str) -> bool:
    """
    Load metrics from a JSON file.
    
    Args:
        filename (str): Filename to load from
        
    Returns:
        bool: Success status
    """
    global _metrics_store
    
    try:
        filepath = os.path.join(_metrics_file_path, filename)
        
        with open(filepath, 'r') as f:
            loaded_metrics = json.load(f)
        
        _metrics_store = loaded_metrics
        
        logger.info(f"Loaded metrics from {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error loading metrics from file: {e}")
        return False

class MetricsTracker:
    """
    Class for tracking metrics within a specific component.
    Provides a more object-oriented approach to metrics tracking.
    """
    
    def __init__(self, category: str):
        """
        Initialize a metrics tracker for a specific category.
        
        Args:
            category (str): Category to track metrics for
        """
        self.category = category
        self.metrics = {}
    
    def update_metrics(self, key: str, value: Any) -> None:
        """
        Update a specific metrics key.
        
        Args:
            key (str): Metrics key
            value (Any): Metrics value
        """
        self.metrics[key] = value
    
    def update_multiple(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Update multiple metrics at once.
        
        Args:
            metrics_dict (dict): Dictionary of metrics to update
        """
        self.metrics.update(metrics_dict)
    
    def get_metric(self, key: str) -> Any:
        """
        Get a specific metric.
        
        Args:
            key (str): Metrics key
            
        Returns:
            Any: Metric value or None if not found
        """
        return self.metrics.get(key)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all metrics tracked by this tracker.
        
        Returns:
            dict: All tracked metrics
        """
        return self.metrics.copy()
    
    def record(self) -> bool:
        """
        Record all metrics to the central store.
        
        Returns:
            bool: Success status
        """
        return record_metrics(self.category, self.metrics)
    
    def reset(self) -> None:
        """Reset all metrics in this tracker."""
        self.metrics = {}


# Automatic session recording
def record_session_start():
    """Record the start of a metrics tracking session."""
    record_metrics("session", {
        "session_id": _session_id,
        "start_time": time.time(),
        "status": "started"
    })

def record_session_end():
    """Record the end of a metrics tracking session and save metrics."""
    record_metrics("session", {
        "session_id": _session_id,
        "end_time": time.time(),
        "status": "completed"
    })
    save_metrics_to_file()

# Record session start when module is imported
record_session_start()

# Register atexit handler to save metrics when program exits
import atexit
atexit.register(record_session_end)
