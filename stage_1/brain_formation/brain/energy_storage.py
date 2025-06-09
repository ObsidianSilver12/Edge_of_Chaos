# --- energy_storage.py V7---
"""
create a storage system for energy to be used by the brain after mycelial network is fully formed.
trigger based on flag set for MYCELIUM_NETWORK_CREATED.Storage system to be created within the limbic
system. store energy as value in a dictionary with key as energy type and value as energy amount. 
energy type is initial_creator_energy and value is a calculation of how mcuh energy would
typically be needed to run the run for a full 2 weeks without sustence. the creator energy received
may be more than this calculation by 5-10%. Apply a random multiplier to the calculation to account for this.
this will allow for each model to have some uniqueness.create a function to return the energy to storage and
to distribute energy to nodes or synapses as needed. simplest method is to monitor different flag statuses
to determine when energy should be distributed or returned
"""

import logging
import uuid
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import math
from constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("Conception")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class EnergyStorage:
    def __init__(self):
        self.energy_storage = {}
        self.energy_amount = 0 # initial value set to 0
        
        
    def create_energy_store(self):
        """
        create the energy store within the deep ...... sub region adjacent to the limbic system of the brain and 
        store energy values in a numpy array or list where we track energy initially stored, energy distributed
        and energy returned. we will use the energy value as a way to trigger some additional functions later
        to ensure there is always enough energy to support the brain. if brain does not have enough energy it will
        trigger a function to return energy from nodes or synapses as needed and will trigger energy creation 
        processes if needed. if energy is not massively out of balance we will not trigger any energy creation
        processes just recovery from nodes or synapses. The energy_amount value is calculated based on the 2 weeks energy
        with a factor between 5-10% added randomly per simulation. Triggered by flag SEEDS_ENTANGLED. 
        save flag as STORE_CREATED
        """

    def add_energy_to_node(self):
        """
        when a node is active/activated add energy to the nodes dictionary and do a local field calculation
        to determine if there are local disturbances around the x.y.z coordinate of the node. set flag to
        FIELD_DISTURBANCE if a local disturbance is found trigger def diagnose_repair_field(self):
        to stabilise the field. 
        """

    def diagnose_repair_field(self):
        """
        when a field disturbance is detected we must trigger a function to diagnose and repair the field. we will
        diagnose and repair per sub region based on the type of disturbance. we will trigger whichever repair is
        needed based on the disturbance type. this does not have to be overly complex for example we could determine
        that there are too many active nodes or synapses in a sub region which would trigger a field distortion so instead of doing
        an overly complex field calculation we could simply create some rules to repair the field. for example we could
        trigger a function to deactivate some nodes/synapses in the sub region to reduce the field distortion. we could trigger
        an energy creation process if needed to support the field based on the need ie if we are running a large thinking
        process with alot of active nodes/synapses then we would rather trigger an energy creation process to temporarily increase
        the energy levels in the field to support the process but if we are running a simple process with a few active nodes
        then we would rather trigger a node deactivation process to reduce the field distortion.
        """

    def remove_energy_from_node(self):
        """
        when a node is deactivated we must remove the energy from the nodes dictionary and return the energy to the 
        energy storage by updating the energy storage dictionary/numpy array.
        """

    def add_energy_to_synaptic_routes(self):
        """
        when a synaptic route is activated we must add the energy to the synaptic routes dictionary and store that value,
        then we must update the energy storage to remove that energy from the energy storage dictionary/numpy array.Set flag
        to ACTIVE_ROUTE
        """

    def remove_energy_from_synaptic_routes(self):
        """
        when a synaptic route is deactivated we must remove the energy from the synaptic routes dictionary and return the energy to the 
        energy storage by updating the energy storage dictionary/numpy array.Set flag to DEACTIVATED_ROUTE
        """
