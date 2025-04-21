"""
Metrics Tracking Module

Provides centralized metrics collection, storage, and analysis for the
Soul Development Framework. Enforces stricter error handling for persistence
and retrieval.

Author: Soul Development Framework Team - Refactored with Strict Error Handling
"""

import logging
import os
import json
import numpy as np
import time
import threading
import uuid
from typing import Dict, List, Any, Union, Optional

# --- Constants ---
try:
    # Import necessary constants, e.g., logging config, base data dir
    from src.constants import LOG_LEVEL, LOG_FORMAT, DATA_DIR_BASE
    METRICS_DIR = os.path.join(DATA_DIR_BASE, "metrics") # Define full path
    METRICS_FILE = os.path.join(METRICS_DIR, "soul_metrics.json")
    # Constant for how often to attempt persistence (e.g., every N seconds)
    PERSIST_INTERVAL_SECONDS = 60 # Persist roughly every minute if active
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. Metrics tracking may fail.")
    # Define fallbacks ONLY for script to load, but functionality will be impaired
    METRICS_DIR = "metrics"
    METRICS_FILE = os.path.join(METRICS_DIR, "soul_metrics.json")
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    PERSIST_INTERVAL_SECONDS = 60

# --- Logging Setup ---
log_file_path = os.path.join("logs", "metrics_tracking.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('metrics_tracking')

# --- Global Metrics Store ---
_metrics_store: Dict[str, Dict[str, Any]] = {}
_metrics_lock = threading.Lock() # Lock for thread safety
_last_persist_time: float = 0.0 # Track last persistence time

# --- Core Functions ---

def record_metric(category: str, name: str, value: Any) -> None:
    """
    Record a single metric value.

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
        json.dumps({name: value}, default=str) # Use default=str as a fallback for common types
    except TypeError as e:
        logger.error(f"Metric value for {category}.{name} may not be JSON serializable: {e}. Value type: {type(value)}")
        # Decide whether to raise error or just log warning
        # raise TypeError(f"Value for metric {category}.{name} is not JSON serializable.") from e

    with _metrics_lock:
        if category not in _metrics_store:
            _metrics_store[category] = {}
        _metrics_store[category][name] = value

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
    Record multiple metrics for a category.

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

    logger.debug(f"Recorded multiple metrics for category: {category}")

    # Trigger persistence check (same logic as single record)
    global _last_persist_time
    current_time = time.time()
    if current_time - _last_persist_time > PERSIST_INTERVAL_SECONDS:
        try: persist_metrics(); _last_persist_time = current_time
        except Exception: logger.error("Periodic persistence failed after recording multiple metrics."); _last_persist_time = current_time


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
            else:
                logger.warning(f"Attempted to clear non-existent metrics category: {category}")
        else:
            _metrics_store = {} # Reset the entire store
            logger.info("Cleared all metrics categories.")


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
            metrics_copy = json.loads(json.dumps(_metrics_store, default=str)) # Get deep copy first

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
            # Clear existing metrics before loading to avoid merging issues?
            # Or update? Let's update, allows partial loads if structure changes? No, safer to replace.
            _metrics_store = loaded_metrics
            _last_persist_time = time.time() # Reset last persist time after loading

        logger.info(f"Loaded metrics successfully from {METRICS_FILE}")
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
    Perform basic analysis on metrics for a category.

    Args:
        category (str): Category to analyze.

    Returns:
        dict: Analysis results (count, numeric stats), or None if category invalid/not found.
    """
    metrics = get_category_metrics(category) # Safely gets metrics or {}
    if not metrics:
        logger.warning(f"No metrics found for category '{category}' to analyze.")
        return None # Return None if no data

    analysis = { "category": category, "metric_count": len(metrics) }
    numeric_values: Dict[str, Union[int, float]] = {}
    non_numeric_keys: List[str] = []

    for name, value in metrics.items():
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

    return analysis


# --- Initial Load Attempt ---
# Try to load metrics when the module is imported. Failures here are logged but don't stop import.
try:
    if os.path.exists(METRICS_FILE):
        load_metrics() # Attempt load, which logs errors internally if it fails
    else:
        logger.info("Metrics file does not exist on initial load. Starting fresh.")
except Exception as initial_load_e:
    logger.error(f"Error during initial metrics load: {initial_load_e}")
    # Initialize empty store if load fails critically
    _metrics_store = {}


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Metrics Tracking Module Example...")

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
        # Example of potentially non-serializable (will log error if strict check enabled)
        # record_metric("debug", "numpy_array", np.array([1,2,3]))
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
        # print(f"All Metrics: {all_mets}") # Can be very large
        print(f"Total Categories: {len(all_mets)}")
    except Exception as e:
         print(f"ERROR getting metrics: {e}")

    # Analyze metrics
    try:
        guff_analysis = analyze_metrics("guff_field")
        print(f"\nAnalysis for 'guff_field': {guff_analysis}")
        void_analysis = analyze_metrics("void_field")
        print(f"Analysis for 'void_field': {void_analysis}")
        non_existent_analysis = analyze_metrics("imaginary_category")
        print(f"Analysis for 'imaginary_category': {non_existent_analysis}") # Should be None
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
        print(f"'void_field' metrics after clear: {get_category_metrics('void_field')}")
        print("Reloading metrics from file...")
        load_metrics() # This should raise FileNotFoundError if persist failed earlier
        print(f"'void_field' metrics after reload: {get_category_metrics('void_field')}")
        print("\nClearing all metrics...")
        clear_metrics()
        print(f"All metrics after clear: {get_all_metrics()}")
    except Exception as e:
        print(f"ERROR during clear/reload: {e}")


    print("\nMetrics Tracking Module Example Finished.")
