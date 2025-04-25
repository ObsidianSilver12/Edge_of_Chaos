"""
Sephiroth Aspect Dictionary (Reimplemented)

This version extracts aspect information from the field classes instead of
using separate aspect files.
"""

import logging
from typing import Dict, Any, List, Optional, Type

# Import all Sephiroth field classes
from stage_1.fields.kether_field import KetherField
from stage_1.fields.chokmah_field import ChokmahField
from stage_1.fields.binah_field import BinahField
from stage_1.fields.chesed_field import ChesedField
from stage_1.fields.geburah_field import GeburahField
from stage_1.fields.tiphareth_field import TipharethField
from stage_1.fields.netzach_field import NetzachField
from stage_1.fields.hod_field import HodField
from stage_1.fields.yesod_field import YesodField
from stage_1.fields.malkuth_field import MalkuthField
from stage_1.fields.daath_field import DaathField

logger = logging.getLogger(__name__)

class AspectDictionary:
    """
    Provides aspect information by extracting it from the field classes.
    Acts as a replacement for the original aspect_dictionary.
    """

    def __init__(self):
        self.field_classes = {
            'kether': KetherField,
            'chokmah': ChokmahField,
            'binah': BinahField,
            'chesed': ChesedField,
            'geburah': GeburahField,
            'tiphareth': TipharethField, 
            'netzach': NetzachField,
            'hod': HodField,
            'yesod': YesodField,
            'malkuth': MalkuthField,
            'daath': DaathField
        }
        self.sephiroth_names = list(self.field_classes.keys())
        logger.info(f"AspectDictionary initialized with {len(self.sephiroth_names)} Sephiroth")

    def get_aspects(self, sephirah_name: str) -> Dict[str, Any]:
        """
        Gets aspect information for a specific Sephirah by creating
        a temporary field instance.
        
        Args:
            sephirah_name: Name of the Sephirah (lowercase)
            
        Returns:
            Dictionary with aspect information
        """
        if sephirah_name.lower() not in self.field_classes:
            return {}
            
        try:
            # Create temporary field instance
            field_class = self.field_classes[sephirah_name.lower()]
            field = field_class()
            
            # Extract aspect information
            aspects = {
                'name': field.name,
                'divine_attribute': getattr(field, 'divine_attribute', None),
                'geometric_correspondence': getattr(field, 'geometric_correspondence', None),
                'element': getattr(field, 'element', None),
                'primary_color': getattr(field, 'primary_color', None),
                'base_frequency': field.base_frequency,
                'aspects': {name: data for name, data in field.aspects.items()}
            }
            return aspects
            
        except Exception as e:
            logger.error(f"Error getting aspects for {sephirah_name}: {e}")
            return {}
    
    def load_aspect_instance(self, sephirah_name: str) -> Any:
        """
        Creates a field instance to serve as an aspect instance.
        
        Args:
            sephirah_name: Name of the Sephirah (lowercase)
            
        Returns:
            Field instance
        """
        if sephirah_name.lower() not in self.field_classes:
            return None
        
        try:
            field_class = self.field_classes[sephirah_name.lower()]
            return field_class()
        except Exception as e:
            logger.error(f"Error loading aspect instance for {sephirah_name}: {e}")
            return None

# Create the singleton instance
aspect_dictionary = AspectDictionary()
