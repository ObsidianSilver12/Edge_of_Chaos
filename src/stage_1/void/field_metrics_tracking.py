"""
Metrics Tracking Module

This module provides classes for tracking various metrics during the soul formation process.
It includes classes for energy metrics, coherence metrics, and formation metrics.

Author: Soul Development Framework Team
"""

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='metrics_tracking.log'
)
logger = logging.getLogger('metrics_tracking')

class EnergyMetrics:
    """
    Class for tracking energy-related metrics in the field system.
    
    This tracks changes in energy levels, distributions, and patterns
    over time during the soul formation process.
    """
    
    def __init__(self):
        """Initialize a new energy metrics tracker."""
        self.metrics = {
            'field_energy': [],
            'energy_changes': [],
            'timestamps': []
        }
        logger.info("Energy metrics tracker initialized")
    
    def record_field_energy(self, metrics):
        """
        Record the energy state of a field.
        
        Args:
            metrics (dict): Energy metrics from the field
        """
        self.metrics['field_energy'].append(metrics)
        self.metrics['timestamps'].append(datetime.now().isoformat())
        logger.debug(f"Recorded field energy metrics: {metrics}")
        
    def record_energy_change(self, delta):
        """
        Record a change in energy state.
        
        Args:
            delta (dict): Energy change metrics
        """
        self.metrics['energy_changes'].append(delta)
        logger.debug(f"Recorded energy change: {delta}")
        
    def get_all_metrics(self):
        """
        Get all recorded energy metrics.
        
        Returns:
            dict: All energy metrics
        """
        return self.metrics
        
    def get_latest_metrics(self):
        """
        Get the most recent energy metrics.
        
        Returns:
            dict: Latest energy metrics
        """
        if self.metrics['field_energy']:
            return self.metrics['field_energy'][-1]
        return {}

class CoherenceMetrics:
    """
    Class for tracking coherence-related metrics in the field system.
    
    This tracks the coherence, stability, and harmony of field systems
    during the soul formation process.
    """
    
    def __init__(self):
        """Initialize a new coherence metrics tracker."""
        self.metrics = {
            'field_coherence': [],
            'coherence_patterns': [],
            'timestamps': []
        }
        logger.info("Coherence metrics tracker initialized")
    
    def record_field_coherence(self, metrics):
        """
        Record the coherence state of a field.
        
        Args:
            metrics (dict): Coherence metrics from the field
        """
        self.metrics['field_coherence'].append(metrics)
        self.metrics['timestamps'].append(datetime.now().isoformat())
        logger.debug(f"Recorded field coherence metrics: {metrics}")
        
    def record_coherence_pattern(self, pattern):
        """
        Record a specific coherence pattern.
        
        Args:
            pattern (dict): Pattern metrics
        """
        self.metrics['coherence_patterns'].append(pattern)
        logger.debug(f"Recorded coherence pattern: {pattern}")
        
    def get_all_metrics(self):
        """
        Get all recorded coherence metrics.
        
        Returns:
            dict: All coherence metrics
        """
        return self.metrics
        
    def get_latest_metrics(self):
        """
        Get the most recent coherence metrics.
        
        Returns:
            dict: Latest coherence metrics
        """
        if self.metrics['field_coherence']:
            return self.metrics['field_coherence'][-1]
        return {}

class FormationMetrics:
    """
    Class for tracking formation-related metrics during soul creation.
    
    This tracks well formation, spark emergence, and other formation events
    during the soul formation process.
    """
    
    def __init__(self):
        """Initialize a new formation metrics tracker."""
        self.metrics = {
            'well_formation': [],
            'spark_formation': [],
            'formation_events': [],
            'timestamps': []
        }
        logger.info("Formation metrics tracker initialized")
    
    def record_well_formation(self, metrics):
        """
        Record potential well formation metrics.
        
        Args:
            metrics (dict): Well formation metrics
        """
        self.metrics['well_formation'].append(metrics)
        self.metrics['timestamps'].append(datetime.now().isoformat())
        logger.debug(f"Recorded well formation metrics: {metrics}")
        
    def record_spark_formation(self, metrics):
        """
        Record soul spark formation metrics.
        
        Args:
            metrics (dict): Spark formation metrics
        """
        self.metrics['spark_formation'].append(metrics)
        self.metrics['timestamps'].append(datetime.now().isoformat())
        logger.debug(f"Recorded spark formation metrics: {metrics}")
        
    def record_formation_event(self, event):
        """
        Record any notable formation event.
        
        Args:
            event (dict): Event data
        """
        self.metrics['formation_events'].append(event)
        logger.debug(f"Recorded formation event: {event}")
        
    def get_all_metrics(self):
        """
        Get all recorded formation metrics.
        
        Returns:
            dict: All formation metrics
        """
        return self.metrics
        
    def get_latest_metrics(self):
        """
        Get the most recent formation metrics.
        
        Returns:
            dict: Latest formation metrics
        """
        latest = {}
        if self.metrics['well_formation']:
            latest['well_formation'] = self.metrics['well_formation'][-1]
        if self.metrics['spark_formation']:
            latest['spark_formation'] = self.metrics['spark_formation'][-1]
        return latest

# Combined metrics tracking class
class MetricsTracker:
    """
    Comprehensive metrics tracking system for the soul formation process.
    
    This combines energy, coherence, and formation metrics to provide
    a complete picture of the soul formation process.
    """
    
    def __init__(self):
        """Initialize a comprehensive metrics tracker."""
        self.energy_metrics = EnergyMetrics()
        self.coherence_metrics = CoherenceMetrics()
        self.formation_metrics = FormationMetrics()
        self.start_time = datetime.now()
        logger.info("Comprehensive metrics tracker initialized")
    
    def get_all_metrics(self):
        """
        Get all metrics from all trackers.
        
        Returns:
            dict: Combined metrics from all trackers
        """
        return {
            'energy': self.energy_metrics.get_all_metrics(),
            'coherence': self.coherence_metrics.get_all_metrics(),
            'formation': self.formation_metrics.get_all_metrics(),
            'start_time': self.start_time.isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds()
        }
    
    def save_metrics(self, filename="metrics.json"):
        """
        Save all metrics to a JSON file.
        
        Args:
            filename (str): Name of the file to save metrics to
            
        Returns:
            bool: True if save was successful
        """
        import json
        try:
            with open(filename, 'w') as f:
                json.dump(self.get_all_metrics(), f, indent=2, default=str)
                logger.info("Metrics saved to %s", filename)
            return True
        except Exception as e:
            logger.error(f"Failed to save metrics to {filename}: {e}")
            return False


