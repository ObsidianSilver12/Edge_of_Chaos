"""
Sephiroth Aspect Dictionary

This module provides a centralized loader for Sephiroth aspects,
facilitating access to aspect information across the Tree of Life.

Author: Soul Development Framework Team
"""

import logging
import importlib
import sys
from typing import Dict, List, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sephiroth_aspects.log'
)
logger = logging.getLogger('sephiroth_aspects')

class SephirothAspectDictionary:
    """
    Central loader for Sephiroth aspects across the Tree of Life.
    
    This class provides a unified interface for accessing aspects across all Sephiroth,
    loading aspect files as needed and caching them for performance.
    """
    
    def __init__(self):
        """Initialize the Sephiroth aspect dictionary loader."""
        self.sephiroth_names = [
            "kether", "chokmah", "binah", 
            "chesed", "geburah", "tiphareth",
            "netzach", "hod", "yesod", "malkuth", "daath"
        ]
        
        # Dictionary to store loaded aspect instances
        self.aspect_instances = {}
        
        # Gateway mappings (moved from defaults)
        self.gateway_mappings = {
            "tetrahedron": ["tiphareth", "netzach", "hod"],
            "octahedron": ["binah", "kether", "chokmah", "chesed", "tiphareth", "geburah"],
            "hexahedron": ["hod", "netzach", "chesed", "chokmah", "binah", "geburah"],
            "icosahedron": ["kether", "chesed", "geburah"],
            "dodecahedron": ["hod", "netzach", "chesed", "binah", "geburah"]
        }
        
        logger.info("Sephiroth Aspect Dictionary initialized")
    
    def load_aspect_instance(self, sephirah: str) -> Any:
        """
        Load a specific Sephiroth aspect instance if available.
        
        Args:
            sephirah (str): Name of the Sephiroth
            
        Returns:
            object: The aspect instance or None if not available
        """
        sephirah = sephirah.lower()
        
        # Return cached instance if available
        if sephirah in self.aspect_instances:
            return self.aspect_instances[sephirah]
            
        try:
            # Construct the module path
            module_name = f"stage_1.sephiroth.{sephirah}_aspects"
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the class name (first letter capitalized + "Aspects")
            class_name = f"{sephirah.capitalize()}Aspects"
            
            # Get the class and create an instance
            aspect_class = getattr(module, class_name)
            aspect_instance = aspect_class()
            
            # Store the instance for future use
            self.aspect_instances[sephirah] = aspect_instance
            
            logger.info(f"Loaded aspect instance for {sephirah}")
            
            return aspect_instance
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to load aspect instance for {sephirah}: {str(e)}")
            return None
    
    def get_aspects(self, sephirah: str) -> Dict[str, Any]:
        """
        Get all aspects for a specific Sephiroth.
        
        Args:
            sephirah (str): Name of the Sephiroth
            
        Returns:
            dict: Dictionary of aspects or empty dict if not found
        """
        sephirah = sephirah.lower()
        
        # Load the aspect instance
        aspect_instance = self.load_aspect_instance(sephirah)
        
        if aspect_instance is None:
            logger.warning(f"No aspect instance found for {sephirah}")
            return {}
        
        # Get metadata
        metadata = aspect_instance.get_metadata()
        
        # Get detailed aspect information
        aspects = aspect_instance.get_all_aspects()
        
        # Combine into a dictionary that matches the expected format
        # for compatibility with existing code
        result = metadata.copy()
        
        # Add primary and secondary aspects as lists of names for backward compatibility
        primary_aspects = aspect_instance.get_primary_aspects()
        secondary_aspects = aspect_instance.get_secondary_aspects() if hasattr(aspect_instance, 'get_secondary_aspects') else {}
        
        result["primary_aspects"] = list(primary_aspects.keys())
        result["secondary_aspects"] = list(secondary_aspects.keys())
        
        # Add detailed aspects dictionary for new code
        result["detailed_aspects"] = aspects
        
        return result
    
    def get_aspect(self, sephirah: str, aspect_name: str) -> Dict[str, Any]:
        """
        Get a specific aspect from a Sephiroth.
        
        Args:
            sephirah (str): Name of the Sephiroth
            aspect_name (str): Name of the aspect
            
        Returns:
            dict: Aspect information or empty dict if not found
        """
        sephirah = sephirah.lower()
        
        # Load the aspect instance
        aspect_instance = self.load_aspect_instance(sephirah)
        
        if aspect_instance is None:
            return {}
            
        # Get the aspect
        return aspect_instance.get_aspect(aspect_name) or {}
    
    def get_relationship(self, sephirah1: str, sephirah2: str) -> Dict[str, Any]:
        """
        Get the relationship between two Sephiroth.
        
        Args:
            sephirah1 (str): First Sephiroth name
            sephirah2 (str): Second Sephiroth name
            
        Returns:
            dict: Relationship information or empty dict if no relationship
        """
        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()
        
        # Load the aspect instance for sephirah1
        aspect_instance = self.load_aspect_instance(sephirah1)
        
        if aspect_instance is None:
            return {}
            
        # Get the relationship
        relationship = aspect_instance.get_relationship(sephirah2)
        
        if relationship:
            return relationship
            
        # Try the reverse relationship
        aspect_instance2 = self.load_aspect_instance(sephirah2)
        
        if aspect_instance2 is None:
            return {}
            
        return aspect_instance2.get_relationship(sephirah1) or {}
    
    def get_all_relationships(self, sephirah: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all relationships for a specific Sephiroth.
        
        Args:
            sephirah (str): Name of the Sephiroth
            
        Returns:
            dict: Dictionary of all relationships
        """
        sephirah = sephirah.lower()
        
        # Load the aspect instance
        aspect_instance = self.load_aspect_instance(sephirah)
        
        if aspect_instance is None:
            return {}
            
        # Check if the instance has a method to get all relationships
        if hasattr(aspect_instance, 'get_all_relationships'):
            return aspect_instance.get_all_relationships()
            
        # Fall back to the relationships dictionary if available
        if hasattr(aspect_instance, 'relationships'):
            return aspect_instance.relationships
            
        return {}
    
    def get_gateway_sephiroth(self, gateway_key: str) -> List[str]:
        """
        Get the Sephiroth associated with a gateway key.
        
        Args:
            gateway_key (str): Name of the gateway key (e.g., "tetrahedron")
            
        Returns:
            list: List of Sephiroth names for this gateway
        """
        gateway_key = gateway_key.lower()
        
        if gateway_key in self.gateway_mappings:
            return self.gateway_mappings[gateway_key]
        
        return []
    
    def get_sephiroth_by_element(self, element: str) -> List[str]:
        """
        Get all Sephiroth associated with a specific element.
        
        Args:
            element (str): Element name
            
        Returns:
            list: List of Sephiroth names with this element
        """
        element = element.lower()
        sephiroth_with_element = []
        
        # Check each Sephirah
        for sephirah in self.sephiroth_names:
            aspect_instance = self.load_aspect_instance(sephirah)
                        if aspect_instance:
                metadata = aspect_instance.get_metadata()
                if metadata.get("element", "").lower() == element:
                    sephiroth_with_element.append(sephirah)
        
        return sephiroth_with_element
    
    def get_sephiroth_by_planet(self, planet: str) -> List[str]:
        """
        Get Sephiroth associated with a specific planet.
        
        Args:
            planet (str): Planet name
            
        Returns:
            list: List of Sephiroth names with this planetary correspondence
        """
        planet = planet.lower()
        sephiroth_with_planet = []
        
        # Check each Sephirah
        for sephirah in self.sephiroth_names:
            aspect_instance = self.load_aspect_instance(sephirah)
            if aspect_instance:
                metadata = aspect_instance.get_metadata()
                correspondence = metadata.get("planetary_correspondence", "")
                if correspondence and correspondence.lower() == planet:
                    sephiroth_with_planet.append(sephirah)
        
        return sephiroth_with_planet
    
    def get_divine_quality(self, sephirah: str) -> Dict[str, Any]:
        """
        Get the divine quality for a Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            dict: Divine quality information
        """
        aspect_instance = self.load_aspect_instance(sephirah)
        if aspect_instance and hasattr(aspect_instance, 'get_divine_quality'):
            return aspect_instance.get_divine_quality()
        return {}
    
    def get_stability_modifier(self, sephirah: str) -> float:
        """
        Get the stability modifier for a Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            float: Stability modifier value
        """
        aspect_instance = self.load_aspect_instance(sephirah)
        if aspect_instance and hasattr(aspect_instance, 'get_stability_modifier'):
            return aspect_instance.get_stability_modifier()
        return 0.5  # Default value
    
    def get_resonance_multiplier(self, sephirah: str) -> float:
        """
        Get the resonance multiplier for a Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            float: Resonance multiplier value
        """
        aspect_instance = self.load_aspect_instance(sephirah)
        if aspect_instance and hasattr(aspect_instance, 'get_resonance_multiplier'):
            return aspect_instance.get_resonance_multiplier()
        return 0.5  # Default value
    
    def get_dimensional_position(self, sephirah: str) -> Dict[str, Any]:
        """
        Get the dimensional position for a Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            dict: Dimensional position information
        """
        aspect_instance = self.load_aspect_instance(sephirah)
        if aspect_instance and hasattr(aspect_instance, 'get_dimensional_position'):
            return aspect_instance.get_dimensional_position()
        return {"level": 0, "pillar": "none"}  # Default values
    
    def get_platonic_solid(self, sephirah: str) -> Optional[str]:
        """
        Get the platonic solid for a Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            
        Returns:
            str: Platonic solid name or None
        """
        aspect_instance = self.load_aspect_instance(sephirah)
        if aspect_instance and hasattr(aspect_instance, 'get_platonic_solid'):
            return aspect_instance.get_platonic_solid()
        return None
    
    def calculate_resonance(self, sephirah: str, frequency: float) -> float:
        """
        Calculate how strongly a frequency resonates with a Sephirah.
        
        Args:
            sephirah (str): Name of the Sephirah
            frequency (float): Frequency to test
            
        Returns:
            float: Resonance value between 0-1
        """
        aspect_instance = self.load_aspect_instance(sephirah)
        if aspect_instance and hasattr(aspect_instance, 'calculate_resonance'):
            return aspect_instance.calculate_resonance(frequency)
        return 0.0  # Default value

# Create a singleton instance for easy importing
aspect_dictionary = SephirothAspectDictionary()
