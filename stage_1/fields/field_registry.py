# --- START OF FILE field_registry.py ---

"""
Field Registry Module

Provides a centralized registry for managing all field instances in the soul
development framework. Handles field creation, access, connections, and transitions.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
from typing import Dict, List, Any, Tuple, Optional, Type
from datetime import datetime

# Import field types
from stage_1.fields.base_field import BaseField
from stage_1.fields.void_field import VoidField
from stage_1.fields.sephiroth_field import SephirothField
from stage_1.fields.guff_field import GuffField

# Configure logging
logger = logging.getLogger(__name__)

class FieldRegistry:
    """
    Central registry for managing all field instances in the soul development framework.
    Implements the singleton pattern to ensure a single point of access.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'FieldRegistry':
        """
        Get the singleton instance of the field registry.
        Creates the instance if it doesn't exist yet.
        
        Returns:
            FieldRegistry: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """
        Initialize the field registry.
        Should only be called once through the get_instance method.
        
        Raises:
            RuntimeError: If attempting to instantiate multiple times
        """
        # Ensure singleton pattern
        if FieldRegistry._instance is not None:
            raise RuntimeError("FieldRegistry is a singleton and should be accessed via get_instance()")
            
        # Registry data structures
        self.fields: Dict[str, BaseField] = {}  # Dictionary of all fields by ID
        self.void_field_id: Optional[str] = None  # ID of the void field
        self.field_connections: Dict[str, Dict[str, Dict[str, Any]]] = {}  # Dictionary of field connections
        self.field_types: Dict[str, Type[BaseField]] = {}  # Dictionary of field classes by type name
        
        # Registry state
        self.initialized: bool = False
        self.creation_time: str = datetime.now().isoformat()
        
        # Register field types
        self._register_field_types()
        
        logger.info("Field Registry initialized")
    
    def _register_field_types(self) -> None:
        """
        Register all field types in the registry.
        
        Raises:
            RuntimeError: If field type registration fails
        """
        try:
            # Register base field types
            self.field_types["base"] = BaseField
            self.field_types["void"] = VoidField
            self.field_types["sephiroth"] = SephirothField
            self.field_types["guff"] = GuffField
            
            # Register specific Sephiroth field types
            # These would normally be imported and registered dynamically
            # For now, only register the ones we've created
            try:
                from stage_1.fields.chesed_field import ChesedField
                self.field_types["chesed"] = ChesedField
            except ImportError:
                logger.warning("ChesedField not found, skipping registration")
            
            # Additional field types would be registered here
            
            logger.info(f"Registered {len(self.field_types)} field types in registry")
        except Exception as e:
            error_msg = f"Failed to register field types: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def initialize_system(self) -> bool:
        """
        Initialize the field system by creating the void field and basic structure.
        
        Returns:
            True if initialization was successful
            
        Raises:
            RuntimeError: If system is already initialized or initialization fails
        """
        if self.initialized:
            raise RuntimeError("Field system is already initialized")
            
        try:
            # 1. Create the Void field
            void_field = self.create_field("void", "The Void", dimensions=(1000.0, 1000.0, 1000.0))
            self.void_field_id = void_field.field_id
            
            # 2. Create Kether as the first Sephiroth
            kether_position = (500.0, 500.0, 900.0)  # Near the top of the void
            kether_dimensions = (100.0, 100.0, 100.0)
            
            # Special case for Kether with direct instantiation
            # Normally we'd use SephirothField but for now we'll use BaseField
            kether_field = self.create_field(
                "base",
                "Kether - Crown",
                dimensions=kether_dimensions,
                base_frequency=963.0,
                resonance=0.99,
                stability=0.99,
                coherence=0.99
            )
            
            # Add Kether to the void
            void_field.add_contained_field(
                kether_field.field_id,
                "sephiroth",
                kether_position,
                kether_dimensions
            )
            
            # Connect void and Kether
            self.connect_fields(void_field.field_id, kether_field.field_id, "containment", 1.0)
            
            # 3. Create Guff as a pocket dimension in Kether
            guff_position = (kether_position[0], kether_position[1], kether_position[2] - 20.0)
            guff_dimensions = (88.0, 88.0, 88.0)
            
            guff_field = self.create_field(
                "guff",
                "Guff - Treasury of Souls",
                dimensions=guff_dimensions,
                base_frequency=963.0,
                resonance=0.92,
                stability=0.95,
                coherence=0.93,
                kether_field_id=kether_field.field_id
            )
            
            # Add Guff to Kether
            kether_field.add_contained_field(
                guff_field.field_id,
                "guff",
                (50.0, 50.0, 40.0),  # Position within Kether
                guff_dimensions
            )
            
            # Connect Kether and Guff
            self.connect_fields(kether_field.field_id, guff_field.field_id, "pocket", 0.95)
            
            # Mark system as initialized
            self.initialized = True
            logger.info("Field system initialized successfully with Void, Kether, and Guff")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize field system: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def create_field(self, field_type: str, name: str, **kwargs) -> BaseField:
        """
        Create a new field of the specified type.
        
        Args:
            field_type: Type of field to create (e.g., "void", "sephiroth", "guff")
            name: Name for the new field
            **kwargs: Additional parameters for field initialization
            
        Returns:
            The created field instance
            
        Raises:
            ValueError: If field_type is invalid
            RuntimeError: If field creation fails
        """
        if field_type not in self.field_types:
            raise ValueError(f"Unknown field type: {field_type}")
            
        try:
            # Get the field class
            field_class = self.field_types[field_type]
            
            # Create the field instance
            field = field_class(name=name, **kwargs)
            
            # Add to registry
            self.fields[field.field_id] = field
            
            logger.info(f"Created {field_type} field '{name}' with ID {field.field_id}")
            return field
            
        except Exception as e:
            error_msg = f"Failed to create {field_type} field '{name}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_field(self, field_id: str) -> BaseField:
        """
        Get a field from the registry by ID.
        
        Args:
            field_id: ID of the field to retrieve
            
        Returns:
            The field instance
            
        Raises:
            ValueError: If field_id is not found
        """
        if field_id not in self.fields:
            raise ValueError(f"Field with ID {field_id} not found in registry")
            
        return self.fields[field_id]
    
    def delete_field(self, field_id: str) -> bool:
        """
        Delete a field from the registry.
        
        Args:
            field_id: ID of the field to delete
            
        Returns:
            True if field was deleted
            
        Raises:
            ValueError: If field_id is not found or is the void field
            RuntimeError: If field deletion fails
        """
        if field_id not in self.fields:
            raise ValueError(f"Field with ID {field_id} not found in registry")
            
        if field_id == self.void_field_id:
            raise ValueError("Cannot delete the void field")
            
        try:
            # Remove field connections
            if field_id in self.field_connections:
                del self.field_connections[field_id]
                
            # Remove as target in other fields' connections
            for source_id, connections in self.field_connections.items():
                if field_id in connections:
                    del connections[field_id]
            
            # Remove from registry
            deleted_field = self.fields.pop(field_id)
            
            logger.info(f"Deleted {deleted_field.field_type} field '{deleted_field.name}' with ID {field_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete field {field_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def connect_fields(self, source_id: str, target_id: str, connection_type: str, strength: float) -> bool:
        """
        Create a connection between two fields.
        
        Args:
            source_id: ID of the source field
            target_id: ID of the target field
            connection_type: Type of connection (e.g., "portal", "overlap", "containment")
            strength: Strength of the connection (0.0-1.0)
            
        Returns:
            True if connection was created
            
        Raises:
            ValueError: If field IDs are invalid or connection parameters are invalid
            RuntimeError: If connection creation fails
        """
        if source_id not in self.fields:
            raise ValueError(f"Source field {source_id} not found in registry")
            
        if target_id not in self.fields:
            raise ValueError(f"Target field {target_id} not found in registry")
            
        if source_id == target_id:
            raise ValueError("Cannot connect a field to itself")
            
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Connection strength must be between 0.0 and 1.0, got {strength}")
            
        try:
            # Get fields
            source_field = self.fields[source_id]
            target_field = self.fields[target_id]
            
            # Create connection in source field
            source_field.connect_field(target_id, connection_type, strength)
            
            # Create connection in target field (with adjusted strength for direction)
            target_field.connect_field(source_id, connection_type, strength * 0.9)
            
            # Create connection in registry
            if source_id not in self.field_connections:
                self.field_connections[source_id] = {}
                
            self.field_connections[source_id][target_id] = {
                'type': connection_type,
                'strength': strength,
                'established_time': datetime.now().isoformat()
            }
            
            logger.info(f"Connected field {source_id} to {target_id} with {connection_type} connection (strength: {strength:.2f})")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect fields {source_id} and {target_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def disconnect_fields(self, source_id: str, target_id: str) -> bool:
        """
        Remove a connection between two fields.
        
        Args:
            source_id: ID of the source field
            target_id: ID of the target field
            
        Returns:
            True if connection was removed
            
        Raises:
            ValueError: If field IDs are invalid or connection does not exist
            RuntimeError: If disconnection fails
        """
        if source_id not in self.fields:
            raise ValueError(f"Source field {source_id} not found in registry")
            
        if target_id not in self.fields:
            raise ValueError(f"Target field {target_id} not found in registry")
            
        if source_id not in self.field_connections or target_id not in self.field_connections[source_id]:
            raise ValueError(f"No connection exists from {source_id} to {target_id}")
            
        try:
            # Get fields
            source_field = self.fields[source_id]
            target_field = self.fields[target_id]
            
            # Remove connection in source field
            source_field.disconnect_field(target_id)
            
            # Remove connection in target field
            try:
                target_field.disconnect_field(source_id)
            except ValueError:
                logger.warning(f"Reverse connection from {target_id} to {source_id} not found when disconnecting")
            
            # Remove connection in registry
            del self.field_connections[source_id][target_id]
            
            logger.info(f"Disconnected field {source_id} from {target_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to disconnect fields {source_id} and {target_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def transfer_entity(self, entity_id: str, source_field_id: str, target_field_id: str, 
                      source_position: Tuple[float, float, float], target_position: Tuple[float, float, float]) -> bool:
        """
        Transfer an entity from one field to another.
        
        Args:
            entity_id: ID of the entity to transfer
            source_field_id: ID of the source field
            target_field_id: ID of the target field
            source_position: Current position in source field
            target_position: Target position in target field
            
        Returns:
            True if entity was transferred
            
        Raises:
            ValueError: If field IDs are invalid, entity is not found, or positions are invalid
            RuntimeError: If transfer fails
        """
        if source_field_id not in self.fields:
            raise ValueError(f"Source field {source_field_id} not found in registry")
            
        if target_field_id not in self.fields:
            raise ValueError(f"Target field {target_field_id} not found in registry")
            
        # Check if fields are connected
        if source_field_id not in self.field_connections or target_field_id not in self.field_connections[source_field_id]:
            raise ValueError(f"Fields {source_field_id} and {target_field_id} are not connected")
            
        try:
            # Get fields
            source_field = self.fields[source_field_id]
            target_field = self.fields[target_field_id]
            
            # Check if entity exists in source field
            entity_found = False
            for entity in source_field.entities:
                if entity['id'] == entity_id:
                    entity_found = True
                    break
                    
            if not entity_found:
                raise ValueError(f"Entity {entity_id} not found in source field {source_field_id}")
            
            # Remove entity from source field
            source_field.remove_entity(entity_id)
            
            # Add entity to target field
            target_field.add_entity(entity_id, target_position)
            
            logger.info(f"Transferred entity {entity_id} from field {source_field_id} to {target_field_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to transfer entity {entity_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def calculate_field_resonance(self, field_id1: str, field_id2: str) -> float:
        """
        Calculate resonance between two fields.
        
        Args:
            field_id1: ID of first field
            field_id2: ID of second field
            
        Returns:
            Resonance value between 0.0 and 1.0
            
        Raises:
            ValueError: If field IDs are invalid
            RuntimeError: If resonance calculation fails
        """
        if field_id1 not in self.fields:
            raise ValueError(f"Field {field_id1} not found in registry")
            
        if field_id2 not in self.fields:
            raise ValueError(f"Field {field_id2} not found in registry")
            
        try:
            # Get fields
            field1 = self.fields[field_id1]
            field2 = self.fields[field_id2]
            
            # Calculate frequency resonance
            freq_resonance = field1.calculate_field_resonance(field2.base_frequency)
            
            # Calculate stability/coherence resonance
            stability_diff = abs(field1.stability - field2.stability)
            coherence_diff = abs(field1.coherence - field2.coherence)
            
            # Lower difference = higher resonance
            stability_coherence_resonance = 1.0 - (stability_diff + coherence_diff) / 2
            
            # Calculate connection resonance
            connection_resonance = 0.0
            if field_id1 in self.field_connections and field_id2 in self.field_connections[field_id1]:
                connection_resonance = self.field_connections[field_id1][field_id2]['strength']
            
            # Combine resonances with weights
            overall_resonance = (
                0.4 * freq_resonance +
                0.4 * stability_coherence_resonance +
                0.2 * connection_resonance
            )
            
            logger.debug(f"Calculated resonance between fields {field_id1} and {field_id2}: {overall_resonance:.4f}")
            return float(max(0.0, min(1.0, overall_resonance)))
            
        except Exception as e:
            error_msg = f"Failed to calculate resonance between fields {field_id1} and {field_id2}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_all_fields(self) -> Dict[str, Dict[str, Any]]:
        """
        Get basic information about all fields in the registry.
        
        Returns:
            Dictionary of field information by ID
        """
        field_info = {}
        
        for field_id, field in self.fields.items():
            field_info[field_id] = {
                'id': field_id,
                'name': field.name,
                'type': field.field_type,
                'dimensions': field.dimensions,
                'base_frequency': field.base_frequency,
                'stability': field.stability,
                'coherence': field.coherence,
                'entity_count': len(field.entities),
                'connection_count': len(field.connections)
            }
        
        return field_info
    
    def get_field_hierarchy(self) -> Dict[str, Any]:
        """
        Get the hierarchical structure of fields starting from the void.
        
        Returns:
            Hierarchical dictionary of field structure
            
        Raises:
            RuntimeError: If field hierarchy cannot be determined
        """
        if not self.void_field_id:
            raise RuntimeError("Void field not set - system may not be initialized")
            
        try:
            void_field = self.fields[self.void_field_id]
            
            # Start with void field
            hierarchy = {
                'id': void_field.field_id,
                'name': void_field.name,
                'type': void_field.field_type,
                'contained_fields': {}
            }
            
            # Add contained fields recursively
            for contained_id, contained_info in void_field.contained_fields.items():
                hierarchy['contained_fields'][contained_id] = self._get_field_subhierarchy(contained_id)
            
            return hierarchy
            
        except Exception as e:
            error_msg = f"Failed to get field hierarchy: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _get_field_subhierarchy(self, field_id: str) -> Dict[str, Any]:
        """
        Helper method to get the subhierarchy for a field.
        
        Args:
            field_id: ID of the field
            
        Returns:
            Hierarchical dictionary for the field and its contained fields
        """
        field = self.fields[field_id]
        
        # Create basic hierarchy node
        hierarchy = {
            'id': field.field_id,
            'name': field.name,
            'type': field.field_type,
            'contained_fields': {}
        }
        
        # Add contained fields if available
        if hasattr(field, 'contained_fields'):
            for contained_id, contained_info in field.contained_fields.items():
                hierarchy['contained_fields'][contained_id] = self._get_field_subhierarchy(contained_id)
        
        return hierarchy
    
    def get_registry_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the field registry.
        
        Returns:
            Dictionary of registry metrics
        """
        field_count = len(self.fields)
        field_type_counts = {}
        
        for field in self.fields.values():
            field_type = field.field_type
            if field_type not in field_type_counts:
                field_type_counts[field_type] = 0
            field_type_counts[field_type] += 1
        
        # Count connections
        connection_count = sum(len(connections) for connections in self.field_connections.values())
        
        # Count entities
        entity_count = sum(len(field.entities) for field in self.fields.values())
        
        metrics = {
            'created_time': self.creation_time,
            'initialized': self.initialized,
            'field_count': field_count,
            'field_type_counts': field_type_counts,
            'connection_count': connection_count,
            'entity_count': entity_count,
            'void_field_id': self.void_field_id,
            'registered_field_types': list(self.field_types.keys())
        }
        
        return metrics
    
    def __str__(self) -> str:
        """String representation of the field registry."""
        return f"FieldRegistry(fields={len(self.fields)}, initialized={self.initialized})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<FieldRegistry fields={len(self.fields)} field_types={len(self.field_types)} connections={sum(len(c) for c in self.field_connections.values())}>"

# --- END OF FILE field_registry.py ---