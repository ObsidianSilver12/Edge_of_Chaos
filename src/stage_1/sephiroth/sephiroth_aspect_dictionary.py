"""
Sephiroth Aspect Dictionary

Provides a centralized loader and access point for Sephiroth aspects,
ensuring consistent data retrieval across the framework. Handles loading
of individual aspect modules and caches instances. Enforces strict error
handling for missing or invalid aspect data.

Author: Soul Development Framework Team - Refactored with Strict Error Handling
"""

import logging
import os
import importlib
import sys
from typing import Dict, List, Any, Tuple, Optional, Union

# --- Constants ---
# Attempt to import constants, raise error if essential ones are missing
try:
    from src.constants import (
        LOG_LEVEL, LOG_FORMAT,
        # Add any other constants this module might directly use (e.g., default counts/falloffs if needed as fallback)
        DEFAULT_HARMONIC_COUNT, DEFAULT_PHI_HARMONIC_COUNT, DEFAULT_HARMONIC_FALLOFF
    )
except ImportError:
     # Basic logging setup if constants failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    DEFAULT_HARMONIC_COUNT = 7
    DEFAULT_PHI_HARMONIC_COUNT = 3
    DEFAULT_HARMONIC_FALLOFF = 0.1
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants. Aspect Dictionary may use fallback defaults.")
    # Continue if possible, but log critical warning
    LOG_LEVEL=logging.INFO
    LOG_FORMAT='%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# Configure logging
log_file_path = os.path.join("logs", "sephiroth_aspect_dictionary.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('sephiroth_aspect_dictionary')

class SephirothAspectDictionary:
    """
    Central loader and cache for Sephiroth aspects.

    Loads aspect data from individual Sephiroth aspect modules (e.g., kether_aspects.py)
    and provides a unified interface to access properties, aspects, and relationships.
    Fails hard if aspect modules or critical data within them are missing/invalid.
    """

    def __init__(self, aspect_module_base_path: str = "stage_1.sephiroth"):
        """
        Initialize the Sephiroth aspect dictionary loader.

        Args:
            aspect_module_base_path (str): The base Python import path for aspect modules
                                           (e.g., 'src.stage_1.sephiroth').

        Raises:
            RuntimeError: If initialization fails due to critical missing components.
        """
        self.sephiroth_names: List[str] = [
            "kether", "chokmah", "binah", "daath", # Include Daath
            "chesed", "geburah", "tiphareth",
            "netzach", "hod", "yesod", "malkuth"
        ]
        self.aspect_module_base_path = aspect_module_base_path

        # Dictionary to cache loaded aspect instances {sephirah_name: instance}
        self._aspect_instances: Dict[str, Any] = {}

        # --- Pre-load and validate all aspect instances on initialization ---
        # This ensures all required modules exist and have necessary attributes upfront.
        self._preload_all_aspects()

        # Define gateway mappings (consider moving to constants or config)
        # Keys should be lowercase Platonic solid names
        self.gateway_mappings: Dict[str, List[str]] = {
            "tetrahedron": ["tiphareth", "netzach", "hod", "geburah"], # Fire related
            "octahedron": ["binah", "kether", "chokmah", "chesed", "tiphareth", "geburah", "hod"], # Air related
            "hexahedron": ["malkuth", "yesod", "hod", "geburah", "binah"], # Earth related
            "icosahedron": ["kether", "chesed", "geburah", "binah", "tiphareth", "yesod"], # Water related
            "dodecahedron": ["hod", "netzach", "chesed", "daath", "binah", "tiphareth", "kether", "chokmah"] # Aether related
        }

        logger.info("Sephiroth Aspect Dictionary initialized and preloaded.")

    def _load_aspect_module(self, sephirah: str) -> Any:
        """Loads the Python module for a given Sephirah."""
        module_path = f"{self.aspect_module_base_path}.{sephirah}_aspects"
        try:
            module = importlib.import_module(module_path)
            logger.debug(f"Successfully imported module: {module_path}")
            return module
        except ImportError as e:
            logger.critical(f"CRITICAL ERROR: Failed to import aspect module for '{sephirah}' at path '{module_path}'. Ensure the file exists and is importable. Error: {e}")
            raise ImportError(f"Missing aspect module for {sephirah}") from e # Fail hard

    def _instantiate_aspect_class(self, module: Any, sephirah: str) -> Any:
        """Instantiates the aspect class from a loaded module."""
        class_name = f"{sephirah.capitalize()}Aspects"
        try:
            aspect_class = getattr(module, class_name)
            instance = aspect_class()
            logger.debug(f"Successfully instantiated class: {class_name}")
            return instance
        except AttributeError:
            logger.critical(f"CRITICAL ERROR: Class '{class_name}' not found in module for '{sephirah}'.")
            raise AttributeError(f"Missing class {class_name} for {sephirah}") from None # Fail hard
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Failed to instantiate '{class_name}' for '{sephirah}': {e}", exc_info=True)
            raise RuntimeError(f"Instantiation failed for {class_name}: {e}") from e # Fail hard

    def _validate_aspect_instance(self, instance: Any, sephirah: str):
        """Checks if the loaded instance has essential attributes."""
        # Define essential attributes expected in each aspect class
        essential_attrs = [
            'name', 'title', 'position', 'pillar', 'element', 'color',
            'divine_name', 'archangel', 'planetary_correspondence', # Allow None for planet
            'base_frequency', 'frequency_modifier', 'phi_harmonic_count',
            'harmonic_count', 'stability_modifier', 'resonance_multiplier',
            'aspects', 'relationships', # The dictionaries/methods themselves
            'get_metadata', 'get_all_aspects', 'get_aspect', 'get_relationship',
            'calculate_resonance', 'get_platonic_solid', # Essential methods
            'get_dimensional_position', 'get_divine_quality'
        ]
        missing = [attr for attr in essential_attrs if not hasattr(instance, attr)]
        if missing:
            raise AttributeError(f"Aspect instance for '{sephirah}' is missing essential attributes: {', '.join(missing)}")

        # Validate base_frequency
        if not isinstance(instance.base_frequency, (int, float)) or instance.base_frequency <= 0:
            raise ValueError(f"Aspect instance for '{sephirah}' has invalid base_frequency: {instance.base_frequency}")

        # Optional: Add more checks for types or ranges of other attributes

        logger.debug(f"Aspect instance for {sephirah} validated successfully.")


    def _preload_all_aspects(self):
        """Loads and validates all aspect instances on initialization."""
        logger.info("Preloading all Sephiroth aspect instances...")
        for sephirah_name in self.sephiroth_names:
            if sephirah_name not in self._aspect_instances:
                try:
                    module = self._load_aspect_module(sephirah_name)
                    instance = self._instantiate_aspect_class(module, sephirah_name)
                    self._validate_aspect_instance(instance, sephirah_name) # Validate essential parts
                    self._aspect_instances[sephirah_name] = instance
                except (ImportError, AttributeError, ValueError, RuntimeError) as e:
                    # Log the critical error and re-raise to halt initialization
                    logger.critical(f"Failed to preload aspects for '{sephirah_name}': {e}")
                    raise RuntimeError(f"Critical preload failure for {sephirah_name}") from e
        logger.info("All aspect instances preloaded and validated.")

    def load_aspect_instance(self, sephirah: str) -> Any:
        """
        Get a specific Sephiroth aspect instance (already preloaded).

        Args:
            sephirah (str): Name of the Sephiroth (case-insensitive).

        Returns:
            object: The aspect instance.

        Raises:
            ValueError: If the sephirah name is invalid or was not loaded.
        """
        sephirah_lower = sephirah.lower()
        if sephirah_lower not in self._aspect_instances:
            # This should not happen if preloading worked, but check anyway.
            logger.error(f"Attempted to access non-preloaded aspect instance for '{sephirah}'. Preload might have failed.")
            raise ValueError(f"Aspect instance for '{sephirah}' not found or failed to load.")
        return self._aspect_instances[sephirah_lower]

    # --- Accessor Methods ---

    def get_aspects(self, sephirah: str) -> Dict[str, Any]:
        """
        Get combined metadata and aspect lists for a specific Sephiroth.

        Args:
            sephirah (str): Name of the Sephiroth.

        Returns:
            dict: Dictionary of combined aspects and metadata.

        Raises:
            ValueError: If the sephirah is invalid or its instance failed to load.
        """
        instance = self.load_aspect_instance(sephirah) # Will raise error if not loaded
        try:
            metadata = instance.get_metadata()
            all_aspects_detail = instance.get_all_aspects()
            # Ensure primary/secondary methods exist or derive from strengths
            if hasattr(instance, 'get_primary_aspects'):
                 primary = list(instance.get_primary_aspects().keys())
            else: # Fallback: derive from strength
                 primary = [k for k,v in all_aspects_detail.items() if v.get("strength",0) >= 0.9] # Example threshold
                 logger.warning(f"Deriving primary aspects by strength for {sephirah}.")

            if hasattr(instance, 'get_secondary_aspects'):
                 secondary = list(instance.get_secondary_aspects().keys())
            else:
                 secondary = [k for k,v in all_aspects_detail.items() if v.get("strength",0) < 0.9]
                 logger.warning(f"Deriving secondary aspects by strength for {sephirah}.")


            result = metadata.copy()
            result["primary_aspects"] = primary
            result["secondary_aspects"] = secondary
            result["detailed_aspects"] = all_aspects_detail # Include the full detail
            return result
        except Exception as e:
            logger.error(f"Error retrieving aspects for {sephirah}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get aspects for {sephirah}") from e

    def get_aspect(self, sephirah: str, aspect_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific detailed aspect from a Sephiroth.

        Args:
            sephirah (str): Name of the Sephiroth.
            aspect_name (str): Name of the aspect.

        Returns:
            dict: Aspect information dictionary, or None if not found.

        Raises:
            ValueError: If the sephirah is invalid or its instance failed to load.
        """
        instance = self.load_aspect_instance(sephirah)
        try:
            aspect_info = instance.get_aspect(aspect_name)
            if aspect_info is None:
                logger.warning(f"Aspect '{aspect_name}' not found for Sephirah '{sephirah}'.")
            return aspect_info # Return None if not found by the instance method
        except Exception as e:
             logger.error(f"Error retrieving aspect '{aspect_name}' for {sephirah}: {e}")
             return None # Return None on error, don't fail hard here

    def get_relationship(self, sephirah1: str, sephirah2: str) -> Optional[Dict[str, Any]]:
        """
        Get the relationship defined from sephirah1 to sephirah2.

        Args:
            sephirah1 (str): First Sephiroth name.
            sephirah2 (str): Second Sephiroth name.

        Returns:
            dict: Relationship information dictionary, or None if not defined.

        Raises:
            ValueError: If sephirah1 is invalid or its instance failed to load.
        """
        instance1 = self.load_aspect_instance(sephirah1)
        try:
            relationship = instance1.get_relationship(sephirah2.lower())
            # Don't check reverse here - get_relationship should be unidirectional lookup
            if relationship is None:
                 logger.debug(f"No direct relationship defined from {sephirah1} to {sephirah2}.")
            return relationship
        except Exception as e:
             logger.error(f"Error retrieving relationship {sephirah1}->{sephirah2}: {e}")
             return None

    def get_all_relationships(self, sephirah: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all relationships defined *from* a specific Sephiroth.

        Args:
            sephirah (str): Name of the Sephiroth.

        Returns:
            dict: Dictionary of all relationships originating from this Sephirah.

        Raises:
            ValueError: If the sephirah is invalid or its instance failed to load.
            AttributeError: If the instance doesn't have the expected relationship data structure.
        """
        instance = self.load_aspect_instance(sephirah)
        try:
            if hasattr(instance, 'get_all_relationships'): # Prefer specific method
                return instance.get_all_relationships()
            elif hasattr(instance, 'relationships'): # Fallback to attribute
                rels = instance.relationships
                if isinstance(rels, dict):
                    return rels
                else:
                    raise TypeError(f"'relationships' attribute for {sephirah} is not a dictionary.")
            else:
                raise AttributeError(f"Aspect instance for {sephirah} lacks relationship data access method/attribute.")
        except Exception as e:
            logger.error(f"Error retrieving all relationships for {sephirah}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get relationships for {sephirah}") from e


    # --- Methods to get specific properties across all Sephiroth ---
    # --- These provide data needed by SephirothFrequencies ---

    def get_harmonic_counts(self) -> Dict[str, int]:
        """Gets harmonic count for all Sephiroth."""
        counts = {}
        for name in self.sephiroth_names:
            try:
                instance = self.load_aspect_instance(name)
                counts[name] = getattr(instance, 'harmonic_count', DEFAULT_HARMONIC_COUNT)
            except (ValueError, AttributeError) as e:
                 logger.error(f"Failed to get harmonic_count for {name}: {e}. Using default.")
                 counts[name] = DEFAULT_HARMONIC_COUNT
        return counts

    def get_phi_harmonic_counts(self) -> Dict[str, int]:
        """Gets phi harmonic count for all Sephiroth."""
        counts = {}
        for name in self.sephiroth_names:
            try:
                instance = self.load_aspect_instance(name)
                counts[name] = getattr(instance, 'phi_harmonic_count', DEFAULT_PHI_HARMONIC_COUNT)
            except (ValueError, AttributeError) as e:
                 logger.error(f"Failed to get phi_harmonic_count for {name}: {e}. Using default.")
                 counts[name] = DEFAULT_PHI_HARMONIC_COUNT
        return counts

    def get_harmonic_falloff(self) -> Dict[str, float]:
        """Gets harmonic amplitude falloff rate for all Sephiroth."""
        falloffs = {}
        for name in self.sephiroth_names:
            try:
                instance = self.load_aspect_instance(name)
                # Assume falloff is defined, maybe add default later if needed in aspects class
                falloffs[name] = getattr(instance, 'harmonic_falloff', DEFAULT_HARMONIC_FALLOFF)
            except (ValueError, AttributeError) as e:
                 logger.error(f"Failed to get harmonic_falloff for {name}: {e}. Using default.")
                 falloffs[name] = DEFAULT_HARMONIC_FALLOFF
        return falloffs

    def get_frequency_modifier(self, sephirah: str) -> float:
        """Gets frequency modifier for a specific Sephirah."""
        try:
            instance = self.load_aspect_instance(sephirah)
            mod = getattr(instance, 'frequency_modifier', None)
            if mod is None or not isinstance(mod, (int, float)):
                 raise ValueError("Invalid frequency_modifier attribute.")
            return float(mod)
        except (ValueError, AttributeError) as e:
             logger.error(f"Failed to get frequency_modifier for {sephirah}: {e}")
             raise ValueError(f"Missing or invalid frequency_modifier for {sephirah}") from e

    # --- Other Accessor Methods (Keep implementations from previous step) ---
    def get_gateway_sephiroth(self, gateway_key: str) -> List[str]:
        gateway_key = gateway_key.lower()
        return self.gateway_mappings.get(gateway_key, [])

    def get_sephiroth_by_element(self, element: str) -> List[str]:
        element_lower = element.lower()
        found = []
        for name in self.sephiroth_names:
            try:
                instance = self.load_aspect_instance(name)
                if getattr(instance, 'element', '').lower() == element_lower:
                    found.append(name)
            except ValueError: continue # Skip if instance failed load
        return found

    def get_sephiroth_by_planet(self, planet: str) -> List[str]:
        planet_lower = planet.lower()
        found = []
        for name in self.sephiroth_names:
            try:
                instance = self.load_aspect_instance(name)
                if getattr(instance, 'planetary_correspondence', None) is not None:
                     if getattr(instance, 'planetary_correspondence').lower() == planet_lower:
                          found.append(name)
            except ValueError: continue
        return found

    # --- Compatibility Methods (ensure they call the instance methods) ---
    def get_divine_quality(self, sephirah: str) -> Dict[str, Any]:
        instance = self.load_aspect_instance(sephirah)
        return instance.get_divine_quality() # Delegate

    def get_stability_modifier(self, sephirah: str) -> float:
        instance = self.load_aspect_instance(sephirah)
        # Stability modifier might be directly on instance or via method
        if hasattr(instance, 'get_stability_modifier'):
             return instance.get_stability_modifier()
        elif hasattr(instance, 'stability_modifier'):
             return instance.stability_modifier
        else:
             raise AttributeError(f"Cannot find stability modifier for {sephirah}")

    def get_resonance_multiplier(self, sephirah: str) -> float:
        instance = self.load_aspect_instance(sephirah)
        if hasattr(instance, 'get_resonance_multiplier'):
             return instance.get_resonance_multiplier()
        elif hasattr(instance, 'resonance_multiplier'):
             return instance.resonance_multiplier
        else:
             raise AttributeError(f"Cannot find resonance multiplier for {sephirah}")

    def get_dimensional_position(self, sephirah: str) -> Dict[str, Any]:
        instance = self.load_aspect_instance(sephirah)
        return instance.get_dimensional_position()

    def get_platonic_solid(self, sephirah: str) -> Optional[str]:
        instance = self.load_aspect_instance(sephirah)
        return instance.get_platonic_solid()

    def calculate_resonance(self, sephirah: str, frequency: float) -> float:
        instance = self.load_aspect_instance(sephirah)
        return instance.calculate_resonance(frequency)

# --- Create a singleton instance for easy importing ---
# --- Perform initialization within a try-except block ---
try:
    aspect_dictionary = SephirothAspectDictionary()
    logger.info("Global Sephiroth Aspect Dictionary created successfully.")
except Exception as e:
    logger.critical(f"CRITICAL FAILURE: Could not initialize global Sephiroth Aspect Dictionary: {e}", exc_info=True)
    aspect_dictionary = None # Indicate failure

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Sephiroth Aspect Dictionary Example...")
    if aspect_dictionary is None:
        print("ERROR: Aspect dictionary failed to initialize. Cannot run examples.")
    else:
        print(f"Loaded aspects for {len(aspect_dictionary._aspect_instances)} Sephiroth.")

        # Get Kether aspects
        print("\n--- Kether Aspects ---")
        kether_data = aspect_dictionary.get_aspects("kether")
        if kether_data:
            print(f"Title: {kether_data.get('title')}")
            print(f"Base Freq: {kether_data.get('base_frequency')}")
            print(f"Primary Aspects: {kether_data.get('primary_aspects')}")
            print(f"Detailed 'divine_unity': {aspect_dictionary.get_aspect('kether', 'divine_unity')}")
        else:
            print("Could not retrieve Kether data.")

        # Get relationship
        print("\n--- Relationship Kether -> Chokmah ---")
        rel = aspect_dictionary.get_relationship("kether", "chokmah")
        if rel:
            print(f"Name: {rel.get('name')}, Quality: {rel.get('quality')}")
        else:
            print("Relationship not found or error occurred.")

        # Get gateway
        print("\n--- Gateway Sephiroth for Tetrahedron ---")
        gateway_seph = aspect_dictionary.get_gateway_sephiroth("tetrahedron")
        print(f"Sephiroth: {gateway_seph}")

        print("\nSephiroth Aspect Dictionary Example Finished.")
