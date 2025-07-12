"""
Enhanced Metrics Tracking Module

Provides centralized metrics collection, storage, analysis, and terminal display for the
Soul Development Framework. Enforces stricter error handling for persistence
and retrieval with additional terminal output.

Author: Soul Development Framework Team - Enhanced with Terminal Display
"""

import logging
import os
import json
import numpy as np
import time
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

# --- Constants ---
try:
    # Import necessary constants
    from shared.constants.constants import LOG_LEVEL, LOG_FORMAT, DATA_DIR_BASE, PERSIST_INTERVAL_SECONDS
    METRICS_DIR = os.path.join(DATA_DIR_BASE, "metrics")
    METRICS_FILE = os.path.join(METRICS_DIR, "soul_metrics.json")
except ImportError as e:
    # Basic logging setup if constants failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print(f"METRICS WARNING: Failed to import essential constants: {e}. Using fallback values.")
    # Define fallbacks
    METRICS_DIR = "metrics"
    METRICS_FILE = os.path.join(METRICS_DIR, "soul_metrics.json")
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    PERSIST_INTERVAL_SECONDS = 60

# --- Configuration ---
# Set to True to enable terminal display of metrics as they're recorded
DISPLAY_METRICS_IN_TERMINAL = True
# Set to True to add formatting for better terminal readability
FORMAT_TERMINAL_OUTPUT = True

# --- Logging Setup ---
log_file_path = os.path.join("output/logs", "metrics_tracking.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('metrics_tracking')

# --- Global Metrics Store ---
_metrics_store: Dict[str, Dict[str, Any]] = {}
_metrics_lock = threading.Lock()  # Lock for thread safety
_last_persist_time: float = 0.0  # Track last persistence time

# --- Terminal Output Helper Functions ---
def _format_value_for_display(value: Any) -> str:
    """Format a value for terminal display with appropriate formatting."""
    if isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, dict):
        return f"<Dict with {len(value)} keys>"
    elif isinstance(value, list):
        return f"<List with {len(value)} items>"
    else:
        return str(value)

def _display_metric_in_terminal(category: str, name: str, value: Any) -> None:
    """Display a metric in the terminal with nice formatting."""
    if not DISPLAY_METRICS_IN_TERMINAL:
        return
    
    if FORMAT_TERMINAL_OUTPUT:
        category_str = f"[{category.upper()}]"
        formatted_value = _format_value_for_display(value)
        print(f"{category_str.ljust(20)} {name.ljust(30)} = {formatted_value}")
    else:
        print(f"METRIC: {category}.{name} = {value}")

def _display_metrics_dict_in_terminal(category: str, metrics_dict: Dict[str, Any]) -> None:
    """Display multiple metrics in the terminal with nice formatting."""
    if not DISPLAY_METRICS_IN_TERMINAL:
        return
        
    if FORMAT_TERMINAL_OUTPUT:
        print(f"\n{'-'*20} METRICS: {category.upper()} {'-'*20}")
        for name, value in metrics_dict.items():
            formatted_value = _format_value_for_display(value)
            print(f"  {name.ljust(30)} = {formatted_value}")
        print(f"{'-'*60}")
    else:
        for name, value in metrics_dict.items():
            print(f"METRIC: {category}.{name} = {value}")

# --- Core Functions ---

def record_metric(category: str, name: str, value: Any) -> None:
    """
    Record a single metric value with terminal display.

    Args:
        category (str): Metric category (e.g., 'simulation', 'field_guff'). Must be non-empty.
        name (str): Metric name within the category. Must be non-empty.
        value: The metric value (can be any JSON-serializable type).

    Raises:
        TypeError: If category or name are not strings.
        ValueError: If category or name are empty strings.
    """
    global _last_persist_time
    if not isinstance(category, str): raise TypeError("Metric category must be a string.")
    if not isinstance(name, str): raise TypeError("Metric name must be a string.")
    if not category: raise ValueError("Metric category cannot be empty.")
    if not name: raise ValueError("Metric name cannot be empty.")

    # Basic check for JSON serializability (optional but good practice)
    try:
        # Use default=str for common non-serializable types like numpy types
        json.dumps({name: value}, default=str)
    except TypeError as e:
        logger.error(f"Metric value for {category}.{name} may not be JSON serializable: {e}. Value type: {type(value)}")
        # Decide whether to raise error or just log warning
        # raise TypeError(f"Value for metric {category}.{name} is not JSON serializable.") from e

    with _metrics_lock:
        if category not in _metrics_store:
            _metrics_store[category] = {}
        _metrics_store[category][name] = value

    # Display the metric in the terminal
    _display_metric_in_terminal(category, name, value)
    
    logger.debug(f"Recorded metric: {category}.{name} = {value}")

    # --- Persistence Check ---
    current_time = time.time()
    if current_time - _last_persist_time > PERSIST_INTERVAL_SECONDS:
        try:
            persist_metrics() # Attempt to persist
            _last_persist_time = current_time # Update time only on successful persist
        except Exception as e:
             # Log error but don't stop metric recording if persistence fails
             logger.error(f"Periodic persistence failed: {e}")
             _last_persist_time = current_time # Update time anyway to avoid rapid retries


def record_metrics(category: str, metrics_dict: Dict[str, Any]) -> None:
    """
    Record multiple metrics for a category with terminal display.

    Args:
        category (str): Metric category. Must be non-empty.
        metrics_dict (dict): Dictionary of metric names and values.

    Raises:
        TypeError: If category is not a string or metrics_dict is not a dict.
        ValueError: If category is empty or metric names/values within the dict are invalid.
    """
    if not isinstance(category, str): raise TypeError("Metric category must be a string.")
    if not category: raise ValueError("Metric category cannot be empty.")
    if not isinstance(metrics_dict, dict): raise TypeError("metrics_dict must be a dictionary.")

    # Validate individual metrics before applying changes
    validated_metrics = {}
    for name, value in metrics_dict.items():
         if not isinstance(name, str) or not name:
              raise ValueError(f"Invalid metric name found in dict for category '{category}': '{name}'")
         try: # Check serializability
              json.dumps({name: value}, default=str)
              validated_metrics[name] = value
         except TypeError as e:
              raise TypeError(f"Value for metric {category}.{name} not JSON serializable: {e}") from e

    # Apply validated metrics
    with _metrics_lock:
        if category not in _metrics_store:
            _metrics_store[category] = {}
        _metrics_store[category].update(validated_metrics) # Use update for efficiency

    # Display the metrics in the terminal
    _display_metrics_dict_in_terminal(category, validated_metrics)
    
    logger.debug(f"Recorded multiple metrics for category: {category}")

    # Trigger persistence check (same logic as single record)
    global _last_persist_time
    current_time = time.time()
    if current_time - _last_persist_time > PERSIST_INTERVAL_SECONDS:
        try: persist_metrics(); _last_persist_time = current_time
        except Exception as e: logger.error("Periodic persistence failed after recording multiple metrics."); _last_persist_time = current_time


def get_metric(category: str, name: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Get a specific metric value safely.

    Args:
        category (str): Metric category.
        name (str): Metric name.
        default: Value to return if metric doesn't exist. Defaults to None.

    Returns:
        Metric value or default. Returns default if category/name invalid or not found.
    """
    if not isinstance(category, str) or not isinstance(name, str) or not category or not name:
         logger.warning(f"Invalid category ('{category}') or name ('{name}') provided to get_metric.")
         return default
    with _metrics_lock:
        # Use .get() for safe access
        return _metrics_store.get(category, {}).get(name, default)

def get_category_metrics(category: str) -> Dict[str, Any]:
    """
    Get a copy of all metrics for a category.

    Args:
        category (str): Metric category.

    Returns:
        dict: A copy of the category's metrics, or an empty dict if category invalid/not found.
    """
    if not isinstance(category, str) or not category:
         logger.warning(f"Invalid category ('{category}') provided to get_category_metrics.")
         return {}
    with _metrics_lock:
        # Return a copy to prevent external modification of the store
        return _metrics_store.get(category, {}).copy()

def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Get a deep copy of all recorded metrics.

    Returns:
        dict: A deep copy of the entire metrics store.
    """
    with _metrics_lock:
        # Use json loads/dumps for a reasonable deep copy of serializable data
        try:
            return json.loads(json.dumps(_metrics_store, default=str))
        except Exception as e:
             logger.error(f"Failed to create deep copy of metrics store: {e}")
             # Fallback to shallow copy (less safe)
             return _metrics_store.copy()


def clear_metrics(category: Optional[str] = None) -> None:
    """
    Clear metrics for a specific category or all categories.

    Args:
        category (str, optional): Category to clear. If None, clears all metrics.

    Raises:
        TypeError: If category is provided but is not a string.
    """
    global _metrics_store # Ensure we modify the global
    if category is not None and not isinstance(category, str):
        raise TypeError("Category to clear must be a string or None.")

    with _metrics_lock:
        if category:
            if category in _metrics_store:
                del _metrics_store[category] # Remove the category entirely
                logger.info(f"Cleared metrics for category: {category}")
                print(f"METRICS: Cleared all metrics for category: {category}")
            else:
                logger.warning(f"Attempted to clear non-existent metrics category: {category}")
        else:
            _metrics_store = {} # Reset the entire store
            logger.info("Cleared all metrics categories.")
            print("METRICS: Cleared all metrics categories.")


def persist_metrics() -> bool:
    """
    Persist current metrics to the JSON file. Fails hard on error.

    Returns:
        bool: True (Raises exception on failure).

    Raises:
        IOError: If the metrics directory cannot be created or file cannot be written.
        RuntimeError: If JSON serialization fails.
    """
    logger.debug(f"Attempting to persist metrics to {METRICS_FILE}")
    try:
        # Ensure directory exists just before writing
        os.makedirs(METRICS_DIR, exist_ok=True)

        # Perform copy inside lock, release lock before file IO
        with _metrics_lock:
            # Use default=str to handle potential numpy types etc.
            metrics_copy = json.loads(json.dumps(_metrics_store, default=str))

        # Write the deep copy to file
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics_copy, f, indent=2) # default=str already applied

        logger.info(f"Persisted metrics successfully to {METRICS_FILE}")
        return True
    except OSError as e:
        logger.error(f"OSError persisting metrics to {METRICS_FILE}: {e}", exc_info=True)
        raise IOError(f"Failed to create/write metrics file/directory: {e}") from e
    except TypeError as e:
        logger.error(f"Serialization error persisting metrics: {e}", exc_info=True)
        raise RuntimeError(f"Metrics data contains non-serializable types: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error persisting metrics: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during metrics persistence: {e}") from e


def load_metrics() -> bool:
    """
    Load metrics from the JSON file. Fails hard on error.

    Returns:
        bool: True (Raises exception on failure).

    Raises:
        FileNotFoundError: If the metrics file does not exist.
        IOError: If the file cannot be read.
        RuntimeError: If JSON decoding fails or loaded data is not a dictionary.
    """
    global _metrics_store, _last_persist_time # Ensure we modify globals
    if not os.path.exists(METRICS_FILE):
        logger.warning(f"Metrics file not found: {METRICS_FILE}. Initializing empty store.")
        # Raise error instead of returning False, as loading is expected if file exists
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")

    logger.debug(f"Attempting to load metrics from {METRICS_FILE}")
    try:
        with open(METRICS_FILE, 'r') as f:
            loaded_metrics = json.load(f)

        if not isinstance(loaded_metrics, dict):
            raise RuntimeError(f"Loaded metrics data is not a dictionary (Type: {type(loaded_metrics)}). File corrupt?")

        with _metrics_lock:
            # Clear existing metrics before loading to avoid merging issues
            _metrics_store = loaded_metrics
            _last_persist_time = time.time() # Reset last persist time after loading

        logger.info(f"Loaded metrics successfully from {METRICS_FILE}")
        print(f"METRICS: Loaded {len(loaded_metrics)} categories from {METRICS_FILE}")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON metrics file {METRICS_FILE}: {e}", exc_info=True)
        raise RuntimeError(f"Metrics file {METRICS_FILE} is corrupted or not valid JSON.") from e
    except IOError as e:
         logger.error(f"IOError loading metrics from {METRICS_FILE}: {e}", exc_info=True)
         raise IOError(f"Failed to read metrics file {METRICS_FILE}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading metrics: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during metrics loading: {e}") from e

# --- Analysis Function ---
def analyze_metrics(category: str) -> Optional[Dict[str, Any]]:
    """
    Perform basic analysis on metrics for a category with terminal display.

    Args:
        category (str): Category to analyze.

    Returns:
        dict: Analysis results (count, numeric stats), or None if category invalid/not found.
    """
    metrics_data = get_category_metrics(category) # Safely gets metrics or {}
    if not metrics_data:
        logger.warning(f"No metrics found for category '{category}' to analyze.")
        print(f"METRICS ANALYSIS: No data available for category '{category}'")
        return None # Return None if no data

    analysis = { "category": category, "metric_count": len(metrics_data) }
    numeric_values: Dict[str, Union[int, float]] = {}
    non_numeric_keys: List[str] = []

    for name, value in metrics_data.items():
        if isinstance(value, (int, float)) and np.isfinite(value): # Check for finite numbers
            numeric_values[name] = value
        else:
            non_numeric_keys.append(name)

    analysis["numeric_metric_count"] = len(numeric_values)
    analysis["non_numeric_keys"] = non_numeric_keys

    if numeric_values:
        values_list = list(numeric_values.values())
        analysis["numeric_total"] = float(np.sum(values_list))
        analysis["numeric_average"] = float(np.mean(values_list))
        analysis["numeric_min"] = float(np.min(values_list))
        analysis["numeric_max"] = float(np.max(values_list))
        analysis["numeric_std_dev"] = float(np.std(values_list))
    else:
         analysis["numeric_total"] = 0.0
         analysis["numeric_average"] = 0.0
         analysis["numeric_min"] = 0.0
         analysis["numeric_max"] = 0.0
         analysis["numeric_std_dev"] = 0.0

    # Display analysis in terminal
    if DISPLAY_METRICS_IN_TERMINAL:
        print(f"\n{'='*20} METRICS ANALYSIS: {category.upper()} {'='*20}")
        print(f"  Total metrics: {analysis['metric_count']}")
        print(f"  Numeric metrics: {analysis['numeric_metric_count']}")
        if analysis['numeric_metric_count'] > 0:
            print(f"  Average: {analysis['numeric_average']:.4f}")
            print(f"  Min: {analysis['numeric_min']:.4f}")
            print(f"  Max: {analysis['numeric_max']:.4f}")
            print(f"  Std Dev: {analysis['numeric_std_dev']:.4f}")
        if non_numeric_keys:
            print(f"  Non-numeric metrics: {len(non_numeric_keys)}")
            print(f"  Keys: {', '.join(non_numeric_keys[:5])}{'...' if len(non_numeric_keys) > 5 else ''}")
        print("="*(40 + len(category)))

    return analysis

def print_metrics_summary() -> None:
    """
    Print a summary of all metrics categories to the terminal.
    """
    all_metrics = get_all_metrics()
    
    print("\n" + "="*60)
    print("METRICS SUMMARY - ALL CATEGORIES")
    print("="*60)
    
    if not all_metrics:
        print("No metrics recorded yet.")
        print("="*60)
        return
        
    categories = list(all_metrics.keys())
    categories.sort()
    
    print(f"Total categories: {len(categories)}")
    for category in categories:
        metrics_count = len(all_metrics[category])
        numeric_count = sum(1 for v in all_metrics[category].values() if isinstance(v, (int, float)))
        print(f"  {category.ljust(25)}: {metrics_count} metrics ({numeric_count} numeric)")
    
    print("\nUse analyze_metrics(category) for detailed analysis of a specific category")
    print("="*60)


# --- Initial Load Attempt ---
# Try to load metrics when the module is imported. Failures here are logged but don't stop import.
try:
    if os.path.exists(METRICS_FILE):
        load_metrics() # Attempt load, raises errors on failure now
    else:
        logger.info("Metrics file does not exist on initial load. Starting fresh.")
        print("METRICS: No existing metrics file found. Starting with empty metrics store.")
except Exception as initial_load_e:
    logger.error(f"Error during initial metrics load: {initial_load_e}")
    print(f"METRICS WARNING: Failed to load existing metrics: {initial_load_e}")
    # Initialize empty store if load fails critically
    _metrics_store = {}


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Enhanced Metrics Tracking Module Example...")

    # Record some metrics
    try:
        record_metric("simulation", "run_id", str(uuid.uuid4()))
        record_metric("simulation", "start_time", time.time())
        record_metric("guff_field", "sparks_processed", 0)
        record_metrics("guff_field", {"initial_energy": 0.15, "stability": 0.92})
        record_metric("guff_field", "sparks_processed", 1) # Update metric
        record_metric("void_field", "wells_found", 15)
        record_metric("void_field", "last_spark_time", time.time())
        record_metric("errors", "initialization_errors", 0)
        record_metric("performance", "loop_time_ms", 15.3)
        print("Metrics recorded.")
    except Exception as e:
         print(f"ERROR recording metrics: {e}")

    # Get metrics
    try:
        sim_start = get_metric("simulation", "start_time")
        guff_metrics = get_category_metrics("guff_field")
        all_mets = get_all_metrics()
        print(f"\nSimulation Start Time: {sim_start}")
        print(f"Guff Field Metrics: {guff_metrics}")
        print(f"Total Categories: {len(all_mets)}")
    except Exception as e:
         print(f"ERROR getting metrics: {e}")

    # Print summary of all metrics
    print_metrics_summary()

    # Analyze metrics
    try:
        guff_analysis = analyze_metrics("guff_field")
        void_analysis = analyze_metrics("void_field")
        non_existent_analysis = analyze_metrics("imaginary_category")
    except Exception as e:
         print(f"ERROR analyzing metrics: {e}")

    # Persist explicitly
    try:
        print("\nExplicitly persisting metrics...")
        persist_metrics()
        print("Persistence successful.")
    except Exception as e:
         print(f"ERROR during explicit persistence: {e}")

    # Clear and reload
    try:
        print("\nClearing 'void_field' metrics...")
        clear_metrics("void_field")
        print("\nReloading metrics from file...")
        load_metrics() # Will fail if file doesn't exist or persistence failed
        print("\nClearing all metrics...")
        clear_metrics()
    except Exception as e:
        print(f"ERROR during clear/reload: {e}")


    print("\nEnhanced Metrics Tracking Module Example Finished.")
# --- END OF FILE metrics_tracking.py ---