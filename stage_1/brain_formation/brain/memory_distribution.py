# --- memory_distribution.py ---
"""
Mmeory fragment distribution for aspects of the soul and identity only. brain formation stage will not
distribute any other type of memory fragments as baby is not yet capable of forming any other memories.
"""

import logging
import uuid
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import math

from constants.constants import *
from memory_definitions import *

# --- Logging Setup ---
logger = logging.getLogger("Conception")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MemoryDistribution:
    def __init__(self):
      self.MEMORY_TYPES = MEMORY_TYPES
      self.SIGNAL_PATTERNS = SIGNAL_PATTERNS

    def load_aspects(self):
        """Loads the aspects of the sephiroth the soul received and the identity aspects so that
        it can be assigned to the grid.
        If the aspects are not loaded must hard fail. Must Trigger on BRAIN_STRUCTURE_CREATED"""
    
    def output_aspects(self):
        """Outputs the aspects of the sephiroth the identity aspects as a json list with all 
        properties per aspect so that it can be mapped according to a simplified memory structure
        If the aspects are not output it must hard fail"""

    def assign_aspect_properties_to_memory_fragments(self):
        """assigns the properties of the aspects to the memory fragments according to the 
        simplified memory structure in memory definitions depending on if its sephiroth 
        aspects or identity aspects. saves this to a memory fragments dictionary or numpy array. 
        if the aspects are not assigned must hard fail. must include the additional information from 
        memory types and signal patterns"""


    def distribute_sephiroth_aspects(self):
        """distributes each aspect to the correct sub region grid area. save the coordinates (x,y,z) of 
        the grid area the aspects is saved in and a location identifier to be used later to retrieve 
        the aspects. save this coordinate to the memory fragment dictionary or numpy array. if the aspects are 
        not loaded must hard fail"""

    def distribute_identity_aspects(self):
        """distributes each aspect to the correct sub region grid area. save the coordinates (x,y,z) of 
        the grid area the aspects is saved in and a location identifier to be used later to retrieve 
        the aspects. save this coordinate to the memory fragment dictionary or numpy array. if the aspects are 
        not loaded must hard fail"""


 

    

