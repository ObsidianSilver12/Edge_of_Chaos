# --- START OF FILE sephiroth_aspect_dictionary.py ---

"""
Sephiroth Aspect Dictionary (Refactored)

Loads and provides access to the centrally defined Sephiroth aspect data
from sephiroth_data.py.
"""

import logging
from typing import Dict, Any, List, Optional

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Data Import ---
try:
    # Import the data structure directly
    from shared.sephiroth.sephiroth_data import SEPHIROTH_ASPECT_DATA
    DATA_LOADED = True
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Could not import SEPHIROTH_ASPECT_DATA from sephiroth_data.py: {e}")
    SEPHIROTH_ASPECT_DATA = {} # Define as empty dict if import fails
    DATA_LOADED = False
except Exception as e:
    # Catch other potential errors during import
    logger.critical(f"CRITICAL ERROR: Unexpected error importing sephiroth_data: {e}", exc_info=True)
    SEPHIROTH_ASPECT_DATA = {}
    DATA_LOADED = False


class AspectDictionary:
    """
    Provides access to Sephiroth aspect information loaded from sephiroth_data.
    Acts as the interface for retrieving Sephiroth definitions.
    """

    def __init__(self):
        """
        Initialize the AspectDictionary.
        """
        if not DATA_LOADED or not SEPHIROTH_ASPECT_DATA:
            # Log critical error and potentially raise an exception or handle gracefully
            err_msg = "SEPHIROTH_ASPECT_DATA failed to load. AspectDictionary cannot function."
            logger.critical(err_msg)
            # Depending on desired behavior, either raise an error or continue with empty data
            raise RuntimeError(err_msg)
            # self.aspect_data = {} # Alternative: Continue with empty data
            # self.sephiroth_names = []
        else:
            self.aspect_data: Dict[str, Dict[str, Any]] = SEPHIROTH_ASPECT_DATA
            self.sephiroth_names: List[str] = list(self.aspect_data.keys())
            # Validate loaded data - basic check
            if len(self.sephiroth_names) != 11:
                 logger.warning(f"Expected 11 Sephiroth, but found {len(self.sephiroth_names)} in loaded data.")
            logger.info(f"AspectDictionary initialized with {len(self.sephiroth_names)} Sephiroth from sephiroth_data.py")

    def get_aspects(self, sephirah_name: str) -> Dict[str, Any]:
        """
        Gets the aspect data dictionary for a specific Sephirah.

        Args:
            sephirah_name: Name of the Sephirah (lowercase recommended).

        Returns:
            Dictionary with aspect information, or an empty dictionary if not found.
        """
        sephirah_lower = sephirah_name.lower()
        data = self.aspect_data.get(sephirah_lower)
        if data is None:
            logger.warning(f"No aspect data found for Sephirah: '{sephirah_name}'. Returning empty dictionary.")
            return {}
        return data.copy() # Return a copy to prevent modification of original data

    # --- Helper Methods ---
    def get_all_base_frequencies(self) -> Dict[str, float]:
        """Returns a dictionary mapping Sephiroth names to their base frequencies."""
        return {name: data.get('base_frequency', 0.0) for name, data in self.aspect_data.items()}

    def get_all_elements(self) -> Dict[str, Optional[str]]:
        """Returns a dictionary mapping Sephiroth names to their element."""
        return {name: data.get('element') for name, data in self.aspect_data.items()}

    def get_all_primary_colors(self) -> Dict[str, Optional[str]]:
        """Returns a dictionary mapping Sephiroth names to their primary color."""
        return {name: data.get('primary_color') for name, data in self.aspect_data.items()}

    def get_geometric_correspondence(self, sephirah_name: str) -> Optional[str]:
        """Gets the geometric correspondence for a specific Sephirah."""
        return self.aspect_data.get(sephirah_name.lower(), {}).get('geometric_correspondence')

    def get_platonic_affinity(self, sephirah_name: str) -> Optional[str]:
        """Gets the Platonic solid affinity for a specific Sephirah."""
        return self.aspect_data.get(sephirah_name.lower(), {}).get('platonic_affinity')

    def get_harmonic_signature_params(self, sephirah_name: str) -> Optional[Dict[str, Any]]:
        """Gets the harmonic signature parameters for a specific Sephirah."""
        return self.aspect_data.get(sephirah_name.lower(), {}).get('harmonic_signature_params')

# --- Create the singleton instance ---
# This instance will be imported by other modules
aspect_dictionary = None
if DATA_LOADED and SEPHIROTH_ASPECT_DATA:
    try:
        aspect_dictionary = AspectDictionary()
    except Exception as e:
        logger.critical(f"Failed to instantiate AspectDictionary: {e}", exc_info=True)
        # Ensure aspect_dictionary is None if instantiation fails
        aspect_dictionary = None
else:
    logger.critical("Cannot create AspectDictionary instance because data loading failed.")


# --- END OF FILE sephiroth_aspect_dictionary.py ---
