"""
Sephiroth Controller

This module manages the creation and access to Sephiroth dimensional fields.
It provides a centralized way to create, access, and manage the different
Sephiroth fields that the soul will interact with during its formation journey.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from stage_1.sephiroth.sephiroth_field import SephirothField
from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sephiroth_controller.log'
)
logger = logging.getLogger('sephiroth_controller')

class SephirothController:
    """
    Controller for managing Sephiroth dimensional fields.
    
    This class creates and provides access to the various Sephiroth fields
    that represent the different dimensions in the Tree of Life.
    """
    
    def __init__(self, dimensions=(64, 64, 64), initialize_all=False):
        """
        Initialize the Sephiroth controller.
        
        Args:
            dimensions (tuple): Dimensions for all Sephiroth fields
            initialize_all (bool): Whether to initialize all fields at startup
        """
        self.dimensions = dimensions
        
        # Dictionary to store Sephiroth fields
        self.fields = {}
        
        # Store field metrics for analysis
        self.field_metrics = {}
        
        if initialize_all:
            self.initialize_all_fields()
        
        logger.info(f"Sephiroth controller initialized with dimensions {dimensions}")
    
    def initialize_all_fields(self):
        """
        Initialize all Sephiroth fields.
        
        This creates all the dimensional fields for each Sephirah in the Tree of Life.
        """
        logger.info("Initializing all Sephiroth fields")
        
        # Get list of valid Sephiroth from aspect dictionary
        sephiroth_names = aspect_dictionary.sephiroth_names
        
        for sephirah in sephiroth_names:
            self.get_field(sephirah)  # This will create the field if it doesn't exist
            
            # Collect metrics for the field
            field_metrics = self.fields[sephirah].get_metrics()
            self.field_metrics[sephirah] = field_metrics
        
        logger.info(f"All {len(sephiroth_names)} Sephiroth fields initialized")
    
    def get_field(self, sephirah: str) -> SephirothField:
        """
        Get a specific Sephiroth field, creating it if it doesn't exist.
        
        Args:
            sephirah (str): Name of the Sephirah (e.g., "kether", "chokmah")
            
        Returns:
            SephirothField: The requested field
        """
        sephirah = sephirah.lower()
        
        # Return existing field if it exists
        if sephirah in self.fields:
            return self.fields[sephirah]
        
        # Otherwise create a new field
        try:
            # Calculate creator resonance based on position in Tree
            creator_resonance = self._calculate_creator_resonance(sephirah)
            
            # Create the field
            field = SephirothField(
                sephirah=sephirah, 
                dimensions=self.dimensions,
                creator_resonance=creator_resonance
            )
            
            # Store the field
            self.fields[sephirah] = field
            
            # Store metrics
            self.field_metrics[sephirah] = field.get_metrics()
            
            logger.info(f"Created {sephirah} field with frequency {field._get_sephirah_frequency():.2f} Hz")
            return field
            
        except Exception as e:
            logger.error(f"Error creating {sephirah} field: {e}")
            raise
    
    def _calculate_creator_resonance(self, sephirah: str) -> float:
        """
        Calculate how strongly a Sephirah resonates with the Creator.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            float: Resonance strength (0.0-1.0)
        """
        # Get aspects for this Sephirah
        aspects = aspect_dictionary.get_aspects(sephirah)
        
        if not aspects or 'position' not in aspects:
            return 0.7  # Default
        
        # Position in the Tree (higher position = closer to Creator)
        position = aspects['position']
        
        # Calculate based on position (Kether=1 has strongest resonance)
        resonance = 1.0 - ((position - 1) * 0.07)
        
        # Additional factors based on pillar
        if 'pillar' in aspects:
            if aspects['pillar'] == 'middle':
                resonance += 0.1  # Stronger in middle pillar
            elif aspects['pillar'] == 'right':
                resonance += 0.05  # Stronger in right pillar than left
        
        # Ensure reasonable range
        resonance = max(0.4, min(resonance, 0.95))
        
        return resonance
    
    def get_all_field_metrics(self) -> Dict[str, Dict]:
        """
        Get metrics for all initialized fields.
        
        Returns:
            Dict[str, Dict]: Dictionary of field metrics keyed by Sephirah name
        """
        # Refresh metrics for all fields
        for sephirah, field in self.fields.items():
            self.field_metrics[sephirah] = field.get_metrics()
            
        return self.field_metrics
    
    def get_sephiroth_journey_order(self) -> List[str]:
        """
        Get the recommended order for the soul's journey through Sephiroth.
        
        According to the specified journey requirements, we start with Yesod
        and end with Kether.
        
        Returns:
            List[str]: List of Sephiroth names in journey order
        """
        # As specified, we start with Yesod
        journey_order = ["yesod"]
        
        # Add other Sephiroth in order from lower to higher
        remaining = [s for s in aspect_dictionary.sephiroth_names 
                    if s != "yesod" and s != "kether" and s != "malkuth"]
        
        # Sort by position (higher position numbers are lower in the tree)
        remaining.sort(key=lambda s: aspect_dictionary.get_aspects(s).get('position', 10), reverse=True)
        
        # Add to journey order
        journey_order.extend(remaining)
        
        # End with Kether
        journey_order.append("kether")
        
        return journey_order
    
    def get_field_relationships(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get the relationships between all Sephiroth fields.
        
        Returns:
            Dict[str, Dict[str, Dict]]: Dictionary of relationships
        """
        relationships = {}
        
        for sephirah in self.fields:
            relationships[sephirah] = aspect_dictionary.get_all_relationships(sephirah)
            
        return relationships

# Example usage if this module is run directly
if __name__ == "__main__":
    # Create a controller and initialize all fields
    controller = SephirothController(initialize_all=True)
    
    # Get the journey order
    journey_order = controller.get_sephiroth_journey_order()
    print("Soul journey order through Sephiroth:")
    for i, sephirah in enumerate(journey_order):
        print(f"{i+1}. {sephirah.capitalize()}")
    
    # Print some metrics for each field
    print("\nSephiroth field metrics:")
    metrics = controller.get_all_field_metrics()
    for sephirah, field_metrics in metrics.items():
        print(f"\n{field_metrics['name']} ({field_metrics['title']}):")
        print(f"  Frequency: {field_metrics['frequency']:.2f} Hz")
        print(f"  Element: {field_metrics['element']}")
        print(f"  Position: {field_metrics['position']} (Pillar: {field_metrics['pillar']})")
        print(f"  Creator Resonance: {field_metrics['connection']['creator_resonance']:.4f}")
